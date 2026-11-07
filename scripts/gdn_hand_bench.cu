// Time tang-llm's hand GDN prefill recurrence (fl_gdn_wide_prep + fl_gdn_wide) for
// Qwen3-Coder-Next's heads on synthetic inputs, to compare with the chunked scan.
//
//   cargo run --release -p tang-compute --features cuda --example dump_coder_gdn > /tmp/coder_gdn.cu
//   nvcc -O3 -arch=sm_86 -std=c++17 -include /tmp/coder_gdn.cu scripts/gdn_hand_bench.cu -o gdn_hand
//   ./gdn_hand 4096 16384
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

template <class T> T* dev(size_t n, float lo, float hi) {
    std::vector<float> h(n * sizeof(T) / 4);
    for (auto& x : h) x = lo + (hi - lo) * (rand() / (float)RAND_MAX);
    T* d;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    for (int ai = 1; ai < argc; ai++) {
        const unsigned t = atoi(argv[ai]);
        const unsigned stride = GDN_B + GDN_HV;
        float* P = dev<float>((size_t)t * stride, -0.5f, 0.5f);
        float* H0 = dev<float>(3 * GDN_CONV, -0.5f, 0.5f);
        float* S = dev<float>((size_t)GDN_HV * 128 * 128, -0.01f, 0.01f);
        float* DT = dev<float>(GDN_HV, -0.1f, 0.1f);
        float* SA = dev<float>(GDN_HV, -0.6f, -0.4f);
        float* Wc = dev<float>(4 * GDN_CONV, -0.5f, 0.5f);
        float* Y = dev<float>((size_t)t * GDN_V, 0, 0);
        float* X = dev<float>((size_t)t * GW_XROW, 0, 0);
        u64 tabh[1] = {(u64)H0}, tabs[1] = {(u64)S};
        u64 *Htab, *Stab;
        CK(cudaMalloc(&Htab, 8));
        CK(cudaMalloc(&Stab, 8));
        CK(cudaMemcpy(Htab, tabh, 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(Stab, tabs, 8, cudaMemcpyHostToDevice));
        unsigned seg[4] = {0, t, 0, 0};
        unsigned* SEG;
        CK(cudaMalloc(&SEG, 16));
        CK(cudaMemcpy(SEG, seg, 16, cudaMemcpyHostToDevice));
        const float eps = 1e-6f;
        cudaEvent_t a, b, c;
        cudaEventCreate(&a); cudaEventCreate(&b); cudaEventCreate(&c);
        float best_p = 1e9, best_w = 1e9;
        for (int it = 0; it < 8; it++) {
            cudaEventRecord(a);
            fl_gdn_wide_prep<<<t, 256>>>(Htab, P, stride, DT, SA, SEG, eps, Wc, X);
            cudaEventRecord(b);
            fl_gdn_wide<<<dim3(GDN_HV, 2), 256>>>(Stab, Htab, P, stride, DT, SA, Y, SEG, eps, Wc, X);
            cudaEventRecord(c);
            CK(cudaEventSynchronize(c));
            float mp, mw;
            cudaEventElapsedTime(&mp, a, b);
            cudaEventElapsedTime(&mw, b, c);
            if (it > 1) { best_p = mp < best_p ? mp : best_p; best_w = mw < best_w ? mw : best_w; }
        }
        CK(cudaGetLastError());
        printf("hand GDN prefill HV=%d HK=%d T=%u: prep %.3f ms, recurrence %.3f ms\n", GDN_HV, GDN_HK, t, best_p, best_w);
        cudaFree(P); cudaFree(H0); cudaFree(S); cudaFree(Y); cudaFree(X);
    }
}
