// GLM-5.3 MoE gate|up streaming roofline (B300). Standalone:
//   nvcc -O3 -arch=sm_100 -std=c++17 -I crates/tang-llm/src/flash/glm5 bench/glm5-moe-roofline/roofline.cu -o roofline
// Layout as one routed layer at T=1: 8 experts x (gate, up) x ff=2048 rows x 2304 B (Q4_K, hid 4096)
// = 75.5 MB, picked from NE=128 experts per tensor (1.2 GB) so every rep is L2-cold.
// Prints µs and GB/s per kernel shape; "gun" is the production g5f_gun on random Q4_K bytes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#ifndef G5F_MINB
#define G5F_MINB 8
#endif
#define G5F_GUR 1
#define G5F_PF 0
#include "moe_cuda.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int HID = 4096, FF = 2048, RB = 2304, NE = 128, K = 8, R16 = RB / 16;  // 144 uint4 per row

// pure streaming: 8 experts' gate rows then up rows, each block reads U uint4 per thread x blockDim, grid-stride
template <int U>
__global__ void flat(const uint4* g, const uint4* u, const unsigned* ids, size_t per_e, float* out) {
    unsigned acc = 0;
    const size_t n = per_e * K * 2;
    for (size_t base = ((size_t)blockIdx.x * blockDim.x) * U + threadIdx.x; base < n; base += (size_t)gridDim.x * blockDim.x * U) {
        uint4 v[U];
        #pragma unroll
        for (int i = 0; i < U; i++) {
            size_t q = base + (size_t)i * blockDim.x;
            if (q >= n) { v[i] = make_uint4(0, 0, 0, 0); continue; }
            const size_t m = q / per_e, o = q % per_e;
            const uint4* src = (m < K ? g : u) + (size_t)ids[m % K] * per_e + o;
            v[i] = __ldg(src);
        }
        #pragma unroll
        for (int i = 0; i < U; i++) acc ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    }
    if (acc == 0x12345678u) out[0] = 1.f;
}

// warp-per-row-group: grid (FF / (WPB * R), K), block 32*WPB. A warp owns R rows of gate and the same
// R rows of up (2R rows x 144 uint4); W warps share a row (each takes 1/W of every row). All of a
// lane's loads are issued before any use (LPL = ceil(2R*144 / (32W)) registers of uint4).
template <int R, int W, int WPB, int MINB>
__global__ void __launch_bounds__(32 * WPB, MINB) rows(const uint4* g, const uint4* u, const unsigned* ids, float* out) {
    constexpr int PER = 2 * R * R16;               // uint4 per row group
    constexpr int LPL = (PER + 32 * W - 1) / (32 * W);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * WPB + warp;        // global warp in this expert
    const int grp = gw / W, sub = gw % W;
    const int r0 = grp * R;
    if (r0 >= FF) return;
    const size_t e = ids[blockIdx.y];
    const uint4* gp = g + (e * FF + r0) * R16;
    const uint4* up = u + (e * FF + r0) * R16;
    uint4 v[LPL];
    #pragma unroll
    for (int i = 0; i < LPL; i++) {
        const int q = sub * 32 + lane + i * 32 * W;
        if (q < PER) v[i] = __ldg(q < R * R16 ? gp + q : up + (q - R * R16));
        else v[i] = make_uint4(0, 0, 0, 0);
    }
    unsigned acc = 0;
    #pragma unroll
    for (int i = 0; i < LPL; i++) acc ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    if (acc == 0x12345678u) out[gw] = 1.f;
}

// gun's access pattern without the math: warp per (gate, up) row pair, lane on chunks lane + 32 i
// (UNR iterations unrolled), per chunk the 16 B header + 32 B nibbles of each row (+ X: the 32 B
// codes, 32 B residual codes and the scale of the activations)
template <int X, int UNR>
__global__ void __launch_bounds__(128, 8) gunlite(const unsigned char* g, const unsigned char* u, const unsigned* ids,
                                                  const unsigned char* xq, const unsigned char* xq2, const float* xd, float* out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.x * 4 + warp;
    const size_t e = ids[blockIdx.y];
    const unsigned char* w[2] = {g + (e * FF + r) * RB, u + (e * FF + r) * RB};
    unsigned acc = 0;
    #pragma unroll UNR
    for (int c = lane; c < HID / 32; c += 32) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const unsigned char* b = w[i] + (c >> 3) * 144;
            const uint4 hd = *(const uint4*)b;
            const uint4* qp = (const uint4*)(b + 16 + ((c & 7) >> 1) * 32);
            const uint4 q0 = qp[0], q1 = qp[1];
            acc ^= hd.x ^ hd.y ^ hd.z ^ hd.w ^ q0.x ^ q0.y ^ q0.z ^ q0.w ^ q1.x ^ q1.y ^ q1.z ^ q1.w;
        }
        if (X) {
            const uint4* xp = (const uint4*)(xq + c * 32);
            const uint4* yp = (const uint4*)(xq2 + c * 32);
            const uint4 x0 = xp[0], x1 = xp[1], y0 = yp[0], y1 = yp[1];
            acc ^= x0.x ^ x0.y ^ x0.z ^ x0.w ^ x1.x ^ x1.y ^ x1.z ^ x1.w ^ y0.x ^ y0.y ^ y0.z ^ y0.w ^ y1.x ^ y1.y ^ y1.z ^ y1.w ^ __float_as_uint(xd[c]);
        }
    }
    if (acc == 0x12345678u) out[r] = 1.f;
}

static float timeit(int reps, std::vector<unsigned*>& idsets, void (*f)(const unsigned*)) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f(idsets[0]);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int r = 0; r < reps; r++) f(idsets[r % idsets.size()]);
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms * 1e3f / reps;
}

static uint4 *G, *U;
static float* OUT;
static unsigned char *XQ, *XQ2;
static float *XD, *XF, *H;
static G5W SG, SU;  // shared expert (Q8_0) or none
static bool WITH_SH = false, NO_X2 = false;

template <int U_>
void run_flat(const unsigned* ids) {
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    flat<U_><<<sms * 8, 256>>>(G, U, ids, (size_t)FF * R16, OUT);
}
template <int R, int W, int WPB, int MINB>
void run_rows(const unsigned* ids) {
    const int warps = FF / R * W;
    rows<R, W, WPB, MINB><<<dim3((warps + WPB - 1) / WPB, K), 32 * WPB>>>(G, U, ids, OUT);
}
template <int X, int UNR>
void run_lite(const unsigned* ids) {
    gunlite<X, UNR><<<dim3(FF / 4, K), 128>>>((const unsigned char*)G, (const unsigned char*)U, ids, XQ, XQ2, XD, OUT);
}
void run_gun(const unsigned* ids) {
    G5W g{(unsigned long long)G, (unsigned long long)FF * RB, RB, 12}, u{(unsigned long long)U, (unsigned long long)FF * RB, RB, 12};
    const G5W none{0, 0, 0, 0};
    const int S = K + (WITH_SH ? 1 : 0);
    g5f_gun<<<dim3(FF / 4, S, 1), 128>>>(g, u, WITH_SH ? SG : none, WITH_SH ? SU : none, ids, K, NE, XF, XQ, NO_X2 ? nullptr : XQ2, XD, H, FF, FF,
                                         WITH_SH ? FF : 0, HID, 1e30f, 1e30f);
}

void run_gux(const unsigned* ids) {
    G5W g{(unsigned long long)G, (unsigned long long)FF * RB, RB, 12}, u{(unsigned long long)U, (unsigned long long)FF * RB, RB, 12};
    const G5W none{0, 0, 0, 0};
    const int S = K + (WITH_SH ? 1 : 0);
    g5f_gux<<<dim3(FF / G5F_GUX_R, S, 1), 128>>>(g, u, WITH_SH ? SG : none, WITH_SH ? SU : none, ids, K, NE, XF, XQ, NO_X2 ? nullptr : XQ2, XD, H, FF, FF,
                                     WITH_SH ? FF : 0, HID, 1e30f, 1e30f);
}

void run_gu4(const unsigned* ids) {
    G5W g{(unsigned long long)G, (unsigned long long)FF * RB, RB, 12}, u{(unsigned long long)U, (unsigned long long)FF * RB, RB, 12};
    const G5W none{0, 0, 0, 0};
    const int S = K + (WITH_SH ? 1 : 0);
    const unsigned sm = g5f_k4_smem(RB, HID, 2, G5F_K4_NR);
    static bool once = false;
    if (!once) { CK(cudaFuncSetAttribute(g5f_gu4, cudaFuncAttributeMaxDynamicSharedMemorySize, sm)); once = true; }
    g5f_gu4<<<dim3(FF / G5F_K4_R, S, 1), 128, sm>>>(g, u, WITH_SH ? SG : none, WITH_SH ? SU : none, ids, K, NE, XF, XQ, NO_X2 ? nullptr : XQ2, XD, H, FF, FF,
                                     WITH_SH ? FF : 0, HID, 1e30f, 1e30f);
}

// g5f_route as one routed layer at T=1 (F32 router 288 x 4096, sigmoid + bias top-8, q8x2 of x)
static void route_bench() {
    const int NE_ = 288;
    float *w, *x, *s, *bias, *xd, *wts; unsigned char *xq, *xq2; unsigned *ids, *cnt;
    CK(cudaMalloc(&w, (size_t)NE_ * HID * 4 * 16)); CK(cudaMalloc(&x, HID * 4)); CK(cudaMalloc(&s, NE_ * 4));
    CK(cudaMalloc(&bias, NE_ * 4)); CK(cudaMalloc(&xd, HID / 32 * 4)); CK(cudaMalloc(&wts, 64));
    CK(cudaMalloc(&xq, HID)); CK(cudaMalloc(&xq2, HID)); CK(cudaMalloc(&ids, 64)); CK(cudaMalloc(&cnt, 64));
    {
        std::vector<float> h((size_t)NE_ * HID * 16), hx(HID), hb(NE_);
        unsigned st = 3;
        for (auto& v : h) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f - 0.5f) * 0.05f; }
        for (auto& v : hx) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f - 0.5f); }
        for (auto& v : hb) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f - 0.5f) * 0.01f; }
        CK(cudaMemcpy(w, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x, hx.data(), HID * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bias, hb.data(), NE_ * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(cnt, 0, 64));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int cold = 0; cold < 2; cold++) {
        const int reps = 64;
        unsigned long long ph[4] = {0, 0, 0, 0};
        float tot = 0;
        for (int r = 0; r < reps; r++) {
            G5W g{(unsigned long long)(w + (size_t)(cold ? r % 16 : 0) * NE_ * HID), 0, HID * 4, 0};
            cudaEventRecord(a);
            g5f_route<<<dim3(NE_, 1), 128>>>(g, x, s, NE_, HID, xq, xq2, xd, bias, ids, wts, 8, 1, 2.5f, cnt);
            cudaEventRecord(b);
            CK(cudaEventSynchronize(b));
            float ms; cudaEventElapsedTime(&ms, a, b); tot += ms;
#ifdef G5F_ROUTE_TS
            std::vector<unsigned long long> ts(4 * 1024);
            CK(cudaMemcpyFromSymbol(ts.data(), g5f_route_ts, ts.size() * 8));
            unsigned long long t0 = ~0ull, t1 = 0, t2 = 0, t3 = 0, d0 = ~0ull;
            for (int e = 0; e < NE_; e++) { t0 = std::min(t0, ts[4 * e]); t1 = std::max(t1, ts[4 * e + 1]); d0 = std::min(d0, ts[4 * e + 1]);
                if (ts[4 * e + 3] >= ts[4 * e + 2] && ts[4 * e + 2] > t2) { t2 = ts[4 * e + 2]; t3 = ts[4 * e + 3]; } }
            unsigned long long tmax0 = 0; for (int e = 0; e < NE_; e++) tmax0 = std::max(tmax0, ts[4 * e]);
            ph[0] += tmax0 - t0; ph[1] += t1 - t0; ph[2] += t2 - t1; ph[3] += t3 - t2;
            std::vector<unsigned long long> z(4 * 1024, 0); CK(cudaMemcpyToSymbol(g5f_route_ts, z.data(), z.size() * 8));
#endif
        }
        printf("route %s: %.2f us/call (events)", cold ? "cold" : "warm", tot * 1e3f / reps);
#ifdef G5F_ROUTE_TS
        printf(" | last block start %.2f, all dots done %.2f, -> select start +%.2f, select +%.2f us",
               ph[0] / 1e3 / reps, ph[1] / 1e3 / reps, ph[2] / 1e3 / reps, ph[3] / 1e3 / reps);
#endif
        printf("\n");
    }
}

// down as one routed layer at T=1: 8 experts x hid 4096 rows x ff 2048 (Q5_K 1408 B / Q6_Kr 1680 B
// rows), 16 disjoint expert sets (L2-cold), + the shared Q8_0r slot; down_splitn vs down4
static void down_bench(int ty) {
    const int DH = HID, DK = FF, DNE = 128;
    const unsigned rb = ty == 13 ? DK / 256 * 176 : DK / 256 * 210;
    const size_t per = (size_t)DNE * DH * rb;
    unsigned char* W;
    CK(cudaMalloc(&W, per));
    {
        std::vector<unsigned char> h(per);
        unsigned st = 5;
        for (size_t i = 0; i < per; i++) { st = st * 1664525u + 1013904223u; h[i] = st >> 24; }
        const int nb = DK / 256;
        for (size_t r = 0; r < per / rb; r++)
            for (int b = 0; b < nb; b++) {
                unsigned char* q = h.data() + r * rb;
                if (ty == 13) { q[b * 176 + 0] = 0; q[b * 176 + 1] = 0x20; q[b * 176 + 2] = 0; q[b * 176 + 3] = 0x20; }
                else { q[nb * 208 + 2 * b] = 0; q[nb * 208 + 2 * b + 1] = 0x20; }
            }
        CK(cudaMemcpy(W, h.data(), per, cudaMemcpyHostToDevice));
    }
    const unsigned SRB = DK + DK / 16;  // Q8_0r
    unsigned char* sd; CK(cudaMalloc(&sd, (size_t)DH * SRB)); CK(cudaMemset(sd, 0, (size_t)DH * SRB));
    const int S = K + 1, hs = DK;
    unsigned char *hq, *hq2; float *hd, *hf, *yp, *wts;
    CK(cudaMalloc(&hq, S * hs)); CK(cudaMalloc(&hq2, S * hs)); CK(cudaMalloc(&hd, S * hs / 32 * 4)); CK(cudaMalloc(&hf, S * hs * 4));
    CK(cudaMalloc(&yp, (size_t)S * DH * 4)); CK(cudaMalloc(&wts, K * 4));
    {
        std::vector<unsigned char> q(S * hs); std::vector<float> d(S * hs / 32), w(K, 0.125f);
        unsigned st = 9;
        for (auto& v : q) { st = st * 1664525u + 1013904223u; v = (st >> 24) % 255 - 127; }
        for (size_t i = 0; i < d.size(); i++) d[i] = 0.01f * (1 + i % 5);
        CK(cudaMemcpy(hq, q.data(), q.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(hq2, q.data(), q.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(hd, d.data(), d.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(wts, w.data(), K * 4, cudaMemcpyHostToDevice));
    }
    std::vector<unsigned*> sets;
    for (int s_ = 0; s_ < 16; s_++) {
        unsigned hid_[K];
        for (int j = 0; j < K; j++) hid_[j] = (unsigned)((s_ * K + j * 37) % DNE);
        unsigned* d; CK(cudaMalloc(&d, sizeof hid_)); CK(cudaMemcpy(d, hid_, sizeof hid_, cudaMemcpyHostToDevice));
        sets.push_back(d);
    }
    const G5W dw{(unsigned long long)W, (unsigned long long)DH * rb, rb, (unsigned)(ty == 13 ? 13 : 14)};
    const G5W dwr{(unsigned long long)W, (unsigned long long)DH * rb, rb, (unsigned)(ty == 13 ? 13 : 114)};
    const G5W sdw{(unsigned long long)sd, 0, SRB, 108};
    const unsigned sm = g5f_k4_smem(rb, DK, 1, G5F_K4_DNR);
    CK(cudaFuncSetAttribute(g5f_down4, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    const double bytes = (double)K * DH * rb + (double)DH * SRB;
    auto run = [&](int which, const unsigned* ids) {
        if (which < 2)
            g5f_down_splitn<<<dim3(DH / (4 * G5F_DNR), S, 1), 128>>>(which == 0 ? dw : dwr, sdw, ids, wts, K, DNE, hf, hq, hq2, hd, hs, yp, DK, DK, DH);
        else
            g5f_down4<<<dim3(DH / G5F_K4_DR, S, 1), 128, sm>>>(dwr, sdw, ids, wts, K, DNE, hf, hq, hq2, hd, hs, yp, DK, DK, DH);
    };
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> ref;
    for (int which = 0; which < 3; which++) {
        run(which, sets[0]); CK(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for (int r = 0; r < 64; r++) run(which, sets[r % 16]);
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        run(which, sets[3]); CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t)S * DH); CK(cudaMemcpy(y.data(), yp, y.size() * 4, cudaMemcpyDeviceToHost));
        double md = 0, mx = 0;
        if (which == 1) ref = y;
        if (!ref.empty()) for (size_t i = 0; i < y.size(); i++) { mx = fmax(mx, fabs(ref[i])); md = fmax(md, fabs((double)y[i] - ref[i])); }
        printf("down ty %d %-22s cold %7.2f us %6.0f GB/s | max |dy| %.3e of %.3e\n", ty, which == 0 ? "down_splitn (raw)" : which == 1 ? "down_splitn (repacked)" : "down4",
               ms * 1e3f / 64, bytes / (ms * 1e3f / 64) / 1e3, md, mx);
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && argv[1][0] == 'd') { down_bench(13); down_bench(14); return 0; }
    if (argc > 1 && argv[1][0] == 'r') { route_bench(); return 0; }
    const size_t per = (size_t)NE * FF * RB;
    CK(cudaMalloc(&G, per)); CK(cudaMalloc(&U, per));
    // random bytes, but Q4_K d / dmin as small finite halves (0x2000 ~ 2^-7) so the math is not NaN
    {
        std::vector<unsigned char> h(per);
        unsigned s = 1;
        for (size_t i = 0; i < per; i++) { s = s * 1664525u + 1013904223u; h[i] = s >> 24; }
        for (size_t r = 0; r < per / 144; r++) { h[r * 144 + 0] = 0; h[r * 144 + 1] = 0x20; h[r * 144 + 2] = 0; h[r * 144 + 3] = 0x20; }
        CK(cudaMemcpy(G, h.data(), per, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(U, h.data(), per, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&OUT, 1 << 20)); CK(cudaMalloc(&XQ, HID)); CK(cudaMalloc(&XQ2, HID)); CK(cudaMalloc(&XD, HID / 32 * 4));
    CK(cudaMalloc(&XF, HID * 4)); CK(cudaMalloc(&H, 9 * FF * 4));
    CK(cudaMemset(XQ, 1, HID)); CK(cudaMemset(XQ2, 1, HID)); CK(cudaMemset(XD, 0, HID / 32 * 4)); CK(cudaMemset(XF, 0, HID * 4));
    // shared expert: Q8_0 rows of 4096 = 128 * 34 B = 4352 B, ffs = 2048, one matrix each (L2-warm-ish: same every rep, as in decode... no: in decode
    // every layer has its own; so rotate over 16 copies via estride is not possible -> one copy, reported separately)
    const unsigned SRB = 4352;
    unsigned char *sg, *su;
    CK(cudaMalloc(&sg, (size_t)FF * SRB)); CK(cudaMalloc(&su, (size_t)FF * SRB));
    CK(cudaMemset(sg, 0, (size_t)FF * SRB)); CK(cudaMemset(su, 0, (size_t)FF * SRB));
    SG = G5W{(unsigned long long)sg, 0, SRB, 8}; SU = G5W{(unsigned long long)su, 0, SRB, 8};
    // 16 disjoint id sets of 8 experts each (128 experts) -> each rep L2-cold
    std::vector<unsigned*> sets;
    for (int s = 0; s < 16; s++) {
        unsigned hid_[K];
        for (int j = 0; j < K; j++) hid_[j] = (unsigned)((s * K + j * 37) % NE);
        unsigned* d; CK(cudaMalloc(&d, sizeof hid_)); CK(cudaMemcpy(d, hid_, sizeof hid_, cudaMemcpyHostToDevice));
        sets.push_back(d);
    }
    std::vector<unsigned*> warm(1, sets[0]);
    const double bytes = 2.0 * K * FF * RB;
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    printf("SMs %d, bytes/rep %.2f MB\n", sms, bytes / 1e6);
    auto rep = [&](const char* name, void (*f)(const unsigned*)) {
        const float c = timeit(64, sets, f), w = timeit(64, warm, f);
        CK(cudaGetLastError());
        printf("%-34s cold %7.2f us %6.0f GB/s | warm %7.2f us %6.0f GB/s\n", name, c, bytes / c / 1e3, w, bytes / w / 1e3);
    };
    if (argc > 1) goto gun;
    rep("flat U1 (8 blk/SM x 256)", run_flat<1>);
    rep("flat U4", run_flat<4>);
    rep("flat U8", run_flat<8>);
    if (argc > 1) goto gun;
    rep("flat U16", run_flat<16>);
    rep("rows R1 W1 WPB4 MINB8", run_rows<1, 1, 4, 8>);
    rep("rows R1 W1 WPB4 MINB16", run_rows<1, 1, 4, 16>);
    rep("rows R1 W1 WPB8 MINB8", run_rows<1, 1, 8, 8>);
    rep("rows R1 W2 WPB4 MINB8", run_rows<1, 2, 4, 8>);
    rep("rows R1 W2 WPB8 MINB8", run_rows<1, 2, 8, 8>);
    rep("rows R1 W4 WPB4 MINB16", run_rows<1, 4, 4, 16>);
    rep("rows R1 W4 WPB8 MINB8", run_rows<1, 4, 8, 8>);
    rep("rows R2 W1 WPB4 MINB4", run_rows<2, 1, 4, 4>);
    rep("rows R2 W2 WPB4 MINB8", run_rows<2, 2, 4, 8>);
    rep("rows R4 W4 WPB4 MINB8", run_rows<4, 4, 4, 8>);
gun:
    if (argc > 1 && argv[1][0] == '4') {
        rep("gu4 routed only", run_gu4);
        WITH_SH = true;
        rep("gu4 + shared Q8_0", run_gu4);
        SG.ty = SU.ty = 108;
        rep("gu4 + shared Q8_0r (108)", run_gu4);
        rep("gux + shared Q8_0r (108)", run_gux);
        SG.ty = SU.ty = 8;
        WITH_SH = false;
        rep("gux routed only", run_gux);
        NO_X2 = true;
        rep("gu4 routed only, no residual x2", run_gu4);
        NO_X2 = false;
        goto cmp;
    }
    rep("gunlite loads only, no x, unr1", run_lite<0, 1>);
    rep("gunlite loads + x, unr1", run_lite<1, 1>);
    rep("gunlite loads + x, unr4", run_lite<1, 4>);
    rep("gun (production, routed only)", run_gun);
    WITH_SH = true;
    rep("gun (production, + shared Q8_0)", run_gun);
    WITH_SH = false;
    rep("gux routed only", run_gux);
    WITH_SH = true;
    rep("gux + shared Q8_0", run_gux);
    SG.ty = SU.ty = 108;
    rep("gun (+ shared repacked 108)", run_gun);
    rep("gux (+ shared repacked 108)", run_gux);
    SG.ty = SU.ty = 8;
    WITH_SH = false; NO_X2 = true;
    rep("gun (routed only, no residual x2)", run_gun);
cmp:
    {   // gu4 vs gun: max |dh| relative to max |h| (summation order differs)
        std::vector<unsigned char> xq(HID), x2(HID); std::vector<float> xd(HID / 32);
        unsigned st = 7;
        for (int i = 0; i < HID; i++) { st = st * 1664525u + 1013904223u; xq[i] = (st >> 24) % 255 - 127; x2[i] = (st >> 16) % 255 - 127; }
        for (int i = 0; i < HID / 32; i++) xd[i] = 0.01f * (1 + i % 7);
        cudaMemcpy(XQ, xq.data(), HID, cudaMemcpyHostToDevice); cudaMemcpy(XQ2, x2.data(), HID, cudaMemcpyHostToDevice);
        cudaMemcpy(XD, xd.data(), HID / 32 * 4, cudaMemcpyHostToDevice);
        WITH_SH = true; NO_X2 = false;
        std::vector<float> a(9 * FF), b(9 * FF);
        run_gun(sets[3]); cudaMemcpy(a.data(), H, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemset(H, 0xff, a.size() * 4);
        run_gu4(sets[3]); CK(cudaDeviceSynchronize()); cudaMemcpy(b.data(), H, b.size() * 4, cudaMemcpyDeviceToHost);
        double mx = 0, md = 0; int nan = 0;
        for (size_t i = 0; i < a.size(); i++) { mx = fmax(mx, fabs(a[i])); if (b[i] != b[i]) nan++; else md = fmax(md, fabs(a[i] - b[i])); }
        printf("gu4 vs gun: max |dh| %.3e, max |h| %.3e, nan %d\n", md, mx, nan);
        if (argc > 1) return 0;
    }
    {   // bit-identity gux vs gun (random Q4_K rows, nonzero activations, + shared Q8_0)
        std::vector<unsigned char> xq(HID), x2(HID); std::vector<float> xd(HID / 32);
        unsigned st = 7;
        for (int i = 0; i < HID; i++) { st = st * 1664525u + 1013904223u; xq[i] = (st >> 24) % 255 - 127; x2[i] = (st >> 16) % 255 - 127; }
        for (int i = 0; i < HID / 32; i++) xd[i] = 0.01f * (1 + i % 7);
        cudaMemcpy(XQ, xq.data(), HID, cudaMemcpyHostToDevice); cudaMemcpy(XQ2, x2.data(), HID, cudaMemcpyHostToDevice);
        cudaMemcpy(XD, xd.data(), HID / 32 * 4, cudaMemcpyHostToDevice);
        WITH_SH = true; NO_X2 = false;
        std::vector<float> a(9 * FF), b(9 * FF);
        run_gun(sets[3]); cudaMemcpy(a.data(), H, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemset(H, 0xff, a.size() * 4);
        run_gux(sets[3]); cudaMemcpy(b.data(), H, b.size() * 4, cudaMemcpyDeviceToHost);
        int diff = 0; for (size_t i = 0; i < a.size(); i++) diff += memcmp(&a[i], &b[i], 4) != 0;
        printf("gux vs gun: %d / %zu values differ (h[0]=%g)\n", diff, a.size(), a[0]);
    }
    return 0;
}
