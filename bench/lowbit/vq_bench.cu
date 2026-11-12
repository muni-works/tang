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
#include <cmath>
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

// Every timed call reads a different copy of its weights (`f(i)` picks copy
// i % copies(bytes)), spanning > 4x the L2, so the numbers are DRAM, not
// L2 (A100: 40 MB L2; the 4-bit matrices are 8-25 MB).
size_t copies(size_t bytes) { return (160u << 20) / bytes + 1; }
template <class F> float time_us(F f, int iters = 200) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 10; ++i) f(i);
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) f(i);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  return ms * 1000.f / iters;
}

// Check aff{3,4}_gemv against the CPU reference vectors from
// `affine_vectors_dump` (tang-compile); returns the max relative error.
float check_affine(int bits) {
  char path[32]; snprintf(path, sizeof path, "aff%d.bin", bits);
  FILE* f = fopen(path, "rb"); if (!f) { printf("no %s (run affine_vectors_dump)\n", path); return -1; }
  unsigned rc[2]; fread(rc, 4, 2, f);
  size_t rows = rc[0], cols = rc[1], aw = rows * cols / 32 * bits, ng = rows * cols / 64;
  std::vector<unsigned> c(aw); std::vector<__half> p(2 * ng); std::vector<float> x(cols), y(rows), g(rows);
  fread(c.data(), 4, aw, f); fread(p.data(), 2, 2 * ng, f); fread(x.data(), 4, cols, f); fread(y.data(), 4, rows, f); fclose(f);
  unsigned* dc; __half2* dp; float *dx, *dy;
  cudaMalloc(&dc, aw * 4); cudaMalloc(&dp, ng * 4); cudaMalloc(&dx, cols * 4); cudaMalloc(&dy, rows * 4);
  cudaMemcpy(dc, c.data(), aw * 4, cudaMemcpyHostToDevice); cudaMemcpy(dp, p.data(), ng * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, x.data(), cols * 4, cudaMemcpyHostToDevice);
  float* dxs; cudaMalloc(&dxs, cols / 64 * 4);
  aff_xsum<<<cols / 64 / 8, 256>>>(dx, dxs, cols);
  if (bits == 3) aff3_gemv<<<(rows + 7) / 8, 256>>>(dc, dp, dx, dxs, dy, rows, cols);
  else aff4_gemv<<<(rows + 7) / 8, 256>>>(dc, dp, dx, dxs, dy, rows, cols);
  cudaMemcpy(g.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
  float worst = 0, norm = 0;
  for (size_t r = 0; r < rows; ++r) norm = fmaxf(norm, fabsf(y[r]));
  for (size_t r = 0; r < rows; ++r) worst = fmaxf(worst, fabsf(g[r] - y[r]) / fmaxf(norm, 1e-6f));
  cudaFree(dc); cudaFree(dp); cudaFree(dx); cudaFree(dy); cudaFree(dxs);
  printf("aff%d_gemv vs CPU reference: max rel err %.2e over %zu rows %s\n", bits, worst, rows, worst < 1e-4 ? "OK" : "FAIL");
  return worst;
}

int main() {
  if (check_affine(3) > 1e-4 || check_affine(4) > 1e-4) return 1;
  int shapes[][2] = {{4096, 4096}, {12288, 4096}, {4096, 12288}};
  for (auto& s : shapes) {
    int rows = s[0], cols = s[1];
    size_t n = (size_t)rows * cols, ncode = n / 4, nwords = (ncode * 11 + 31) / 32 + 1;
    std::vector<unsigned> hc(nwords); for (auto& v : hc) v = rand() * 2654435761u;
    std::vector<__half> hs(n / 128, __float2half(0.01f)), hb(2048 * 4);
    for (auto& v : hb) v = __float2half((rand() % 2001 - 1000) / 500.f);
    unsigned* dc; __half *ds, *db; float *dx, *dxr, *dy, *dsg; __nv_bfloat16* dw;
    size_t kv = copies(nwords * 4), kb = copies(n * 2);
    cudaMalloc(&dc, nwords * 4 * kv); cudaMalloc(&ds, hs.size() * 2); cudaMalloc(&db, hb.size() * 2);
    cudaMalloc(&dx, cols * 4); cudaMalloc(&dxr, cols * 4); cudaMalloc(&dsg, cols * 4);
    cudaMalloc(&dy, rows * 4); cudaMalloc(&dw, n * 2 * kb);
    cudaMemcpy(dc, hc.data(), nwords * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dx, 0, cols * 4); cudaMemset(dsg, 0, cols * 4); cudaMemset(dw, 0, n * 2 * kb);
    for (size_t k = 1; k < kv; ++k) cudaMemcpy(dc + k * nwords, dc, nwords * 4, cudaMemcpyDeviceToDevice);
    cudaFuncSetAttribute(vq_gemv, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048 * 16);
    float tv = time_us([&](int i) {
      vq_rotate<<<cols / 128, 128>>>(dx, dsg, dxr);
      vq_gemv<<<(rows + 7) / 8, 256, 2048 * 16>>>(dc + i % kv * nwords, ds, db, dxr, dy, rows, cols);
    });
    float tb = time_us([&](int i) { bf16_gemv<<<(rows + 7) / 8, 256>>>(dw + i % kb * n, dx, dy, rows, cols); });
    double vb = nwords * 4.0 + hs.size() * 2.0 + hb.size() * 2.0, bb = n * 2.0;
    // packed affine: 3-bit (g3g64h, rotated) and 4-bit (the MLX a4 layout, no rotation)
    float ta[5], trot = 0; double ab[5];
    float* dxs; cudaMalloc(&dxs, cols / 64 * 4);
    for (int bits = 3; bits <= 4; ++bits) {
      size_t aw = n / 32 * bits, ng = n / 64;
      std::vector<unsigned> hac(aw); for (auto& v : hac) v = rand() * 2654435761u;
      std::vector<__half> hp(2 * ng, __float2half(0.01f));
      unsigned* dac; __half2* dsb;
      size_t ka = copies(aw * 4 + ng * 4);
      cudaMalloc(&dac, aw * 4 * ka); cudaMalloc(&dsb, ng * 4 * ka);
      for (size_t k = 0; k < ka; ++k) {
        cudaMemcpy(dac + k * aw, hac.data(), aw * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dsb + k * ng, hp.data(), ng * 4, cudaMemcpyHostToDevice);
      }
      ta[bits] = bits == 3 ? time_us([&](int i) {
        vq_rotate<<<cols / 128, 128>>>(dx, dsg, dxr);
        aff_xsum<<<cols / 64 / 8, 256>>>(dxr, dxs, cols);
        aff3_gemv<<<(rows + 7) / 8, 256>>>(dac + i % ka * aw, dsb + i % ka * ng, dxr, dxs, dy, rows, cols);
      }) : time_us([&](int i) {
        aff_xsum<<<cols / 64 / 8, 256>>>(dx, dxs, cols);
        aff4_gemv<<<(rows + 7) / 8, 256>>>(dac + i % ka * aw, dsb + i % ka * ng, dx, dxs, dy, rows, cols);
      });
      ab[bits] = aw * 4.0 + ng * 4.0;
      if (bits == 3) trot = time_us([&](int) { vq_rotate<<<cols / 128, 128>>>(dx, dsg, dxr); aff_xsum<<<cols / 64 / 8, 256>>>(dxr, dxs, cols); });
      cudaFree(dac); cudaFree(dsb);
    }
    printf("%5dx%-5d  bf16 %7.1f us %5.0f GB/s | a4 %6.1f us %5.0f GB/s | g3 %6.1f us (rot+xsum %4.1f) %5.0f GB/s (%.2fx a4; gemv alone %.2fx) | vq %6.1f us %5.0f GB/s (%.2fx a4)\n",
           rows, cols, tb, bb / tb / 1e3, ta[4], ab[4] / ta[4] / 1e3, ta[3], trot, ab[3] / (ta[3] - trot) / 1e3, ta[4] / ta[3], ta[4] / (ta[3] - trot),
           tv, vb / tv / 1e3, ta[4] / tv);
    cudaError_t e = cudaGetLastError(); if (e) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 1; }
    cudaFree(dc); cudaFree(ds); cudaFree(db); cudaFree(dx); cudaFree(dxr); cudaFree(dsg); cudaFree(dy); cudaFree(dw);
  }
}
