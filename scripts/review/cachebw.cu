// Read bandwidth vs working-set size (CUDA), independent of any tang kernel.
// nvcc -O3 -arch=sm_86 cachebw.cu -o cachebw && ./cachebw
#include <cstdio>
__global__ void rd(const float4* x, float* o, unsigned n4, unsigned passes) {
  float4 a = make_float4(0, 0, 0, 0);
  unsigned nt = gridDim.x * blockDim.x, tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned p = 0; p < passes; p++)
    for (unsigned i = tid; i < n4; i += nt) {
      float4 v = x[(i + p * 64) % n4];
      a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
  if (a.x == 1234.5f) o[tid] = a.y + a.z + a.w;
}
int main() {
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int threads = sms * 1536, block = 256;
  float* o; cudaMalloc(&o, threads * 4);
  for (int kb : {256, 1024, 2048, 4096, 8192, 32768, 131072, 1048576}) {
    size_t size = (size_t)kb * 1024;
    float4* x; cudaMalloc(&x, size); cudaMemset(x, 1, size);
    unsigned n4 = size / 16, passes = (8ull << 30) / size; if (!passes) passes = 1;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e30;
    for (int r = 0; r < 5; r++) {
      cudaEventRecord(a);
      rd<<<threads / block, block>>>(x, o, n4, passes);
      cudaEventRecord(b); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
    }
    printf("%d KB  %.0f GB/s\n", kb, (double)size * passes / (best * 1e-3) / 1e9);
    cudaFree(x);
  }
}
