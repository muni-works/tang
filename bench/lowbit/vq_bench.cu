// Decode-GEMV speed of the low-bit formats (bench/lowbit): packed 3-bit
// affine (g3g64h, the GPTQ winner) and VQ, against 4-bit affine (the MLX a4
// layout) and a bf16 GEMV of the same shape. vq.cu is generated from tang-compile:
//   VQ_CU_OUT=bench/lowbit/vq.cu cargo test -p tang-compile --lib vq_cuda_src_dump -- --ignored
//   nvcc -O3 -arch=native -I bench/lowbit bench/lowbit/vq_bench.cu -o /tmp/vq_bench && /tmp/vq_bench
// Random codes/book (speed only; correctness is the CPU test against the
// dequant reference). Prints us/call and effective GB/s for each shape.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_bf16.h>
#include "vq.cu"

__global__ void bf16_gemv(const __nv_bfloat16* __restrict__ w, const float* __restrict__ x,
                          float* __restrict__ y, int rows, int cols) {
  int lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) return;
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w + (size_t)r * cols);
  float acc = 0.f;
  for (int i = lane; i < cols / 2; i += 32) {
    float2 f = __bfloat1622float2(w2[i]);
    acc += f.x * x[2 * i] + f.y * x[2 * i + 1];
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) y[r] = acc;
}

template <class F> float time_us(F f, int iters = 200) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 10; ++i) f();
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) f();
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  return ms * 1000.f / iters;
}

int main() {
  int shapes[][2] = {{4096, 4096}, {12288, 4096}, {4096, 12288}};
  for (auto& s : shapes) {
    int rows = s[0], cols = s[1];
    size_t n = (size_t)rows * cols, ncode = n / 4, nwords = (ncode * 11 + 31) / 32 + 1;
    std::vector<unsigned> hc(nwords); for (auto& v : hc) v = rand() * 2654435761u;
    std::vector<__half> hs(n / 128, __float2half(0.01f)), hb(2048 * 4);
    for (auto& v : hb) v = __float2half((rand() % 2001 - 1000) / 500.f);
    unsigned* dc; __half *ds, *db; float *dx, *dxr, *dy, *dsg; __nv_bfloat16* dw;
    cudaMalloc(&dc, nwords * 4); cudaMalloc(&ds, hs.size() * 2); cudaMalloc(&db, hb.size() * 2);
    cudaMalloc(&dx, cols * 4); cudaMalloc(&dxr, cols * 4); cudaMalloc(&dsg, cols * 4);
    cudaMalloc(&dy, rows * 4); cudaMalloc(&dw, n * 2);
    cudaMemcpy(dc, hc.data(), nwords * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dx, 0, cols * 4); cudaMemset(dsg, 0, cols * 4); cudaMemset(dw, 0, n * 2);
    cudaFuncSetAttribute(vq_gemv, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048 * 16);
    float tv = time_us([&] {
      vq_rotate<<<cols / 128, 128>>>(dx, dsg, dxr);
      vq_gemv<<<(rows + 7) / 8, 256, 2048 * 16>>>(dc, ds, db, dxr, dy, rows, cols);
    });
    float tb = time_us([&] { bf16_gemv<<<(rows + 7) / 8, 256>>>(dw, dx, dy, rows, cols); });
    double vb = nwords * 4.0 + hs.size() * 2.0 + hb.size() * 2.0, bb = n * 2.0;
    // packed affine: 3-bit (g3g64h, rotated) and 4-bit (the MLX a4 layout, no rotation)
    float ta[5]; double ab[5];
    for (int bits = 3; bits <= 4; ++bits) {
      size_t aw = (n * bits + 31) / 32 + 1, ng = n / 64;
      std::vector<unsigned> hac(aw); for (auto& v : hac) v = rand() * 2654435761u;
      std::vector<__half> hp(ng, __float2half(0.01f));
      unsigned* dac; __half *dsc, *dbi;
      cudaMalloc(&dac, aw * 4); cudaMalloc(&dsc, ng * 2); cudaMalloc(&dbi, ng * 2);
      cudaMemcpy(dac, hac.data(), aw * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(dsc, hp.data(), ng * 2, cudaMemcpyHostToDevice);
      cudaMemcpy(dbi, hp.data(), ng * 2, cudaMemcpyHostToDevice);
      ta[bits] = bits == 3 ? time_us([&] {
        vq_rotate<<<cols / 128, 128>>>(dx, dsg, dxr);
        aff3_gemv<<<(rows + 7) / 8, 256>>>(dac, dsc, dbi, dxr, dy, rows, cols);
      }) : time_us([&] { aff4_gemv<<<(rows + 7) / 8, 256>>>(dac, dsc, dbi, dx, dy, rows, cols); });
      ab[bits] = aw * 4.0 + ng * 4.0;
      cudaFree(dac); cudaFree(dsc); cudaFree(dbi);
    }
    printf("%5dx%-5d  bf16 %7.1f us %5.0f GB/s | a4 %6.1f us %5.0f GB/s | g3 %6.1f us %5.0f GB/s (%.2fx a4) | vq %6.1f us %5.0f GB/s (%.2fx a4)\n",
           rows, cols, tb, bb / tb / 1e3, ta[4], ab[4] / ta[4] / 1e3, ta[3], ab[3] / ta[3] / 1e3, ta[4] / ta[3],
           tv, vb / tv / 1e3, ta[4] / tv);
    cudaError_t e = cudaGetLastError(); if (e) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 1; }
    cudaFree(dc); cudaFree(ds); cudaFree(db); cudaFree(dx); cudaFree(dxr); cudaFree(dsg); cudaFree(dy); cudaFree(dw);
  }
}
