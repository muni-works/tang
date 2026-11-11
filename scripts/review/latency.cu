// Empty-kernel duration inside a CUDA graph, and pointer-chase memory latency.
// nvcc -O2 -arch=sm_90a -o latency latency.cu && ./latency
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
__global__ void empty() {}
__global__ void chase(const unsigned long long* __restrict__ p, unsigned long long start, int hops, unsigned long long* out, long long* cyc) {
    unsigned long long i = start;
    long long t0 = clock64();
    for (int h = 0; h < hops; h++) i = p[i];
    long long t1 = clock64();
    *out = i; *cyc = t1 - t0;
}
int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    printf("device %s, %d SMs, SM clock %d kHz (attribute)\n", pr.name, pr.multiProcessorCount, clk);
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    // empty kernels: N launches captured into one graph, replay timed; per-kernel = total / N
    for (int blocks : {1, pr.multiProcessorCount, 4 * pr.multiProcessorCount}) {
        const int N = 2000;
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < N; i++) empty<<<blocks, 256, 0, s>>>();
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        for (int w = 0; w < 3; w++) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        float best = 1e9;
        for (int r = 0; r < 10; r++) {
            CK(cudaEventRecord(a, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(b, s));
            CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
        }
        printf("empty kernel in a graph, %d blocks x 256: %.3f us each (best of 10 replays of %d)\n", blocks, best * 1000.f / N, N);
        // stream launches, for reference
        CK(cudaEventRecord(a, s));
        for (int i = 0; i < N; i++) empty<<<blocks, 256, 0, s>>>();
        CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b));
        printf("empty kernel, stream launches, %d blocks: %.3f us each\n", blocks, ms * 1000.f / N);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
    // pointer chase: one thread through a random single cycle of 64-byte-spaced slots
    for (size_t bytes : {(size_t)1 << 20, (size_t)32 << 20, (size_t)4 << 30}) {
        size_t slots = bytes / 64, n = bytes / 8;
        std::vector<unsigned long long> perm(slots);
        for (size_t i = 0; i < slots; i++) perm[i] = i;
        std::mt19937_64 rng(7); std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<unsigned long long> h(n, 0);
        for (size_t i = 0; i < slots; i++) h[perm[i] * 8] = perm[(i + 1) % slots] * 8;
        unsigned long long *d, *o; long long* c;
        CK(cudaMalloc(&d, bytes)); CK(cudaMalloc(&o, 8)); CK(cudaMalloc(&c, 8));
        CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
        const int hops = 20000;
        chase<<<1, 1>>>(d, perm[0] * 8, hops, o, c); CK(cudaDeviceSynchronize());   // warm-up (code, TLB for small sizes)
        // timed run on a part of the cycle the warm-up did not touch (else it is all L2 hits)
        const size_t far = std::min(slots - 1, (size_t)4 * hops);
        CK(cudaEventRecord(a)); chase<<<1, 1>>>(d, perm[far] * 8, hops, o, c); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); long long cyc; CK(cudaMemcpy(&cyc, c, 8, cudaMemcpyDeviceToHost));
        printf("pointer chase over %zu MB (%zu slots, 64 B apart): %.1f ns per hop (events), %.0f cycles per hop (clock64)\n",
               bytes >> 20, slots, ms * 1e6 / hops, (double)cyc / hops);
        CK(cudaFree(d)); CK(cudaFree(o)); CK(cudaFree(c));
    }
}
