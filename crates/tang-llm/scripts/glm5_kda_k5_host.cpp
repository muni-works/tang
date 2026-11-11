// Host check of crates/tang-llm/src/flash/glm5/kda_k5.cuh without a GPU: compiles the exact
// kernel source as C++ (the same stubs as glm5_moe_cuda_host.cpp, plus the MoE source for
// G5W/g5_deq/g5_dot), runs each kernel over an emulated grid, and compares against
// independent naive C++ references computed right here (fixture-shaped geometry: the KDA
// fixture layer, hidden 48, nh 4, hd 16 — the same shapes the Rust CPU reference tests run).
//
// This checks the kernel *logic* (indexing, gate forms, the f64 rms); NVRTC compilation and
// real GPU execution are the pod's cuda-gated engine tests (engine_gpu_matches_cpu).
// Run: crates/tang-llm/scripts/glm5_kda_k5_host.sh
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

#define __global__
#define __device__
#define __forceinline__ inline
struct K5Dim { unsigned x, y, z; };
static K5Dim blockIdx, blockDim, threadIdx;
static inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }

#include "../src/flash/glm5/moe_cuda.cuh"
#define K5_HOST_CHECK 1
#include "../src/flash/glm5/kda_k5.cuh"

static int failures = 0;

// emulate kernel<<<ceil(n/128), 128>>>(args...)
template <typename F, typename... A>
static void launch(int n, F kernel, A... args) {
    blockDim.x = 128;
    for (unsigned b = 0; b * 128 < (unsigned)n; b++)
        for (unsigned t = 0; t < 128; t++) {
            blockIdx.x = b;
            threadIdx.x = t;
            kernel(args...);
        }
}

static void report(const char* what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-44s max abs err %.3e  (tol %.1e)  %s\n", what, err, tol, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static double maxerr(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { std::printf("size mismatch %zu vs %zu\n", a.size(), b.size()); return 1e30; }
    double e = 0;
    for (size_t i = 0; i < a.size(); i++) e = std::max(e, (double)std::fabs(a[i] - b[i]));
    return e;
}

static G5W view(const void* p, size_t cols) {
    G5W w;
    w.base = (unsigned long long)p;
    w.estride = 0;                      // one matrix: e = 0
    w.row_bytes = (unsigned)(cols * 4); // f32 rows
    w.ty = 0;
    return w;
}

// deterministic pseudo-random f32 in [-1, 1) (xorshift64*), the mhc test's rnd
static unsigned long long RS = 0x9E3779B97F4A7C15ull;
static float rnd() {
    RS ^= RS >> 12; RS ^= RS << 25; RS ^= RS >> 27;
    return (float)((RS >> 40) & 0xffffff) / (float)(1 << 23) - 1.0f;
}

int main() {

// ---- REAL-MODEL GEOMETRY CHECK (the bug pod 20's garbage ids exposed):
// the fixture shapes passed while the real KdaLayerW layout failed. attn_q is
// [8192, 4096] Q8_0 (d_inner=8192 outputs, hidden=4096 inputs). Run k5_gemv on
// REAL geometry with synthetic Q8_0 weights whose dequant values are exact, and
// compare against a CPU dot over the same encoded bytes. This splits "kernel
// wrong at real shapes" from "wiring wrong".
{
    const unsigned ROWS = 8192, COLS = 4096, T = 1;
    // Q8_0: 32 elements per 34-byte block (f16 d + int8 qs[32]).
    const unsigned blocks = COLS / 32;
    const unsigned row_bytes = blocks * 34;
    std::vector<unsigned char> w(ROWS * row_bytes);
    std::vector<float> ref_w(ROWS * COLS);
    std::vector<float> x(T * COLS);
    unsigned seed = 7;
    auto rnd = [&]() { seed = seed * 1103515245 + 12345; return (seed >> 16) % 2000 / 1000.0f - 1.0f; };
    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned b = 0; b < blocks; b++) {
            // d = an exact positive f16 power of two in [1, 8]; qs = int8 ±1 so dequant is exact
            float d = (float)(1 << (b % 4));
            unsigned bits = (unsigned short)(((b % 4) + 15) << 10);
            unsigned char *blk = &w[(unsigned long long)r * row_bytes + b * 34];
            blk[0] = (unsigned char)(bits & 0xFF);
            blk[1] = (unsigned char)(bits >> 8);
            for (unsigned j = 0; j < 32; j++) {
                blk[2 + j] = (signed char)((j % 2) ? 1 : -1);
                ref_w[(unsigned long long)r * COLS + b * 32 + j] = d * (float)((signed char)blk[2 + j]);
            }
        }
    }
    for (auto &v : x) v = rnd();
    // CPU reference dot
    std::vector<float> ref(T * ROWS);
    for (unsigned r = 0; r < ROWS; r++) {
        float acc = 0.f;
        for (unsigned j = 0; j < COLS; j++) acc += ref_w[(unsigned long long)r * COLS + j] * x[j];
        ref[r] = acc;
    }
    // device path
    G5W gw{ (unsigned long long)(size_t)w.data(), (unsigned long long)ROWS * row_bytes, row_bytes, 8 };
    std::vector<float> y(T * ROWS, 0.f);
    // emulated k5_gemv grid
    for (unsigned i = 0; i < T * ROWS; i++) {
        unsigned t = i / ROWS, r = i % ROWS;
        y[(unsigned long long)t * ROWS + r] = g5_dot(gw, 0, r, x.data() + (unsigned long long)t * COLS, COLS);
    }
    // first-block dump: kernel dequant vs reference, elements 0..7 of row 0
    {
        const unsigned char *row0 = &w[0];
        std::printf("row0 block0 bytes:");
        for (unsigned j = 0; j < 8; j++) std::printf(" %02x", row0[j]);
        std::printf("\nkernel deq[0..7]:");
        for (unsigned j = 0; j < 8; j++) std::printf(" %.4f", g5_deq(8, row0, j));
        std::printf("\nref    deq[0..7]:");
        for (unsigned j = 0; j < 8; j++) std::printf(" %.4f", ref_w[j]);
        std::printf("\nkernel y[0..7]:");
        for (unsigned r = 0; r < 8; r++) std::printf(" %.4f", y[r]);
        std::printf("\nref_w[0..7]:");
        for (unsigned j = 0; j < 8; j++) std::printf(" %.4f", ref_w[j]);
        { float direct = 0.f; for (unsigned j = 0; j < COLS; j++) direct += ref_w[j] * x[j];
          std::printf("\ndirect dot ref_w[0..4095].x[0..4095] = %.4f", (double)direct); }
        std::printf("\nx[0..7]:");
        for (unsigned j = 0; j < 8; j++) std::printf(" %.4f", x[j]);
        std::printf("\nref    y[0..7]:");
        for (unsigned r = 0; r < 8; r++) std::printf(" %.4f", ref[r]);
    }
    float maxerr = 0.f;
    for (unsigned i = 0; i < T * ROWS; i++) maxerr = std::max(maxerr, std::abs(y[i] - ref[i]));
    std::printf("real-geometry k5_gemv (8192x4096 Q8_0): max abs err %.6g\n", (double)maxerr);
    if (maxerr > 1e-3) { std::puts("REAL-GEOMETRY K5_GEMV: FAIL"); std::exit(1); }
    std::puts("REAL-GEOMETRY K5_GEMV: ok");
}

    // fixture-layer geometry (meta.json: hidden 48, nh 4, hd 16, lb -5)
    const int HID = 48, NH = 4, HD = 16, DI = NH * HD, T = 100;
    const float LB = -5.0f, EPS = 1e-5f;

    std::vector<float> x(T * HID), wq(DI * HID), wk(DI * HID), wv(DI * HID);
    std::vector<float> fa(HD * HID), fb(DI * HD), dt(DI), rate(NH);
    std::vector<float> bp(NH * HID), ga(HD * HID), gb(DI * HD), on(HD), wo(HID * DI);
    for (float& v : x) v = rnd();
    for (float& v : wq) v = rnd();
    for (float& v : wk) v = rnd();
    for (float& v : wv) v = rnd();
    for (float& v : fa) v = rnd();
    for (float& v : fb) v = rnd();
    for (float& v : dt) v = rnd();
    for (float& v : rate) v = std::exp(rnd() * 0.2f);
    for (float& v : bp) v = rnd();
    for (float& v : ga) v = rnd();
    for (float& v : gb) v = rnd();
    for (float& v : on) v = rnd();
    for (float& v : wo) v = rnd();

    // ---- naive C++ references (the CPU reference's forms, f64 inner)
    auto ref_dot = [&](const float* w, const float* xt, int k) {
        double s = 0;
        for (int j = 0; j < k; j++) s += (double)w[j] * (double)xt[j];
        return (float)s;
    };
    std::vector<float> rq(T * DI), rk(T * DI), rv(T * DI);
    for (int t = 0; t < T; t++)
        for (int r = 0; r < DI; r++) {
            rq[t * DI + r] = ref_dot(&wq[r * HID], &x[t * HID], HID);
            rk[t * DI + r] = ref_dot(&wk[r * HID], &x[t * HID], HID);
            rv[t * DI + r] = ref_dot(&wv[r * HID], &x[t * HID], HID);
        }
    std::vector<float> rg(T * DI), rbeta(T * NH), gpre(T * DI);
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < NH; h++) {
            rbeta[t * NH + h] = 1.f / (1.f + std::exp(-ref_dot(&bp[h * HID], &x[t * HID], HID)));
            for (int j = 0; j < HD; j++) {
                float a = 0;
                for (int p = 0; p < HD; p++) {
                    float fav = 0;
                    for (int q = 0; q < HID; q++) fav += fa[p * HID + q] * x[t * HID + q];
                    a += fb[(h * HD + j) * HD + p] * fav;
                }
                gpre[t * DI + h * HD + j] = a + dt[h * HD + j];
                rg[t * DI + h * HD + j] = LB / (1.f + std::exp(-rate[h] * gpre[t * DI + h * HD + j]));
            }
        }
    }

    // ---- the kernels
    std::vector<float> qkv(T * 3 * DI), g1(T * HD), g(T * DI), beta(T * NH);
    launch(T * DI, k5_gemv, view(wq.data(), HID), 0u, x.data(), qkv.data() + 0, (unsigned)(3 * DI), (unsigned)T, (unsigned)DI, (unsigned)HID);
    launch(T * DI, k5_gemv, view(wk.data(), HID), 0u, x.data(), qkv.data() + DI, (unsigned)(3 * DI), (unsigned)T, (unsigned)DI, (unsigned)HID);
    launch(T * DI, k5_gemv, view(wv.data(), HID), 0u, x.data(), qkv.data() + 2 * DI, (unsigned)(3 * DI), (unsigned)T, (unsigned)DI, (unsigned)HID);
    // the stacked qkv only holds q at stride 3*DI; compare per block
    std::vector<float> qk(T * DI), vk(T * DI);
    for (int t = 0; t < T; t++) {
        std::memcpy(&qk[t * DI], &qkv[t * 3 * DI], DI * 4);
        std::memcpy(&vk[t * DI], &qkv[t * 3 * DI + 2 * DI], DI * 4);
    }
    std::vector<float> qkvmid(T * DI);
    for (int t = 0; t < T; t++) std::memcpy(&qkvmid[t * DI], &qkv[t * 3 * DI + DI], DI * 4);
    report("k5_gemv q into stacked qkv", maxerr(qk, rq), 2e-5);
    report("k5_gemv k into stacked qkv", maxerr(qkvmid, rk), 2e-5);
    report("k5_gemv v into stacked qkv", maxerr(vk, rv), 2e-5);

    launch(T * HD, k5_gemv, view(fa.data(), HID), 0u, x.data(), g1.data(), (unsigned)HD, (unsigned)T, (unsigned)HD, (unsigned)HID);
    launch(T * DI, k5_gemv, view(fb.data(), HD), 0u, g1.data(), g.data(), (unsigned)DI, (unsigned)T, (unsigned)DI, (unsigned)HD);
    std::vector<float> graw = g; // the raw f_b gemv output (pre-dt) for the softplus check
    launch(T * DI, k5_gate_finish, g.data(), dt.data(), rate.data(), (unsigned)NH, (unsigned)HD, LB, (unsigned)T);
    report("k5_gemv f_a/f_b + k5_gate_finish vs reference", maxerr(g, rg), 1e-4);

    launch(T * NH, k5_beta, view(bp.data(), HID), x.data(), beta.data(), (unsigned)T, (unsigned)HID, (unsigned)NH);
    report("k5_beta vs reference", maxerr(beta, rbeta), 1e-5);

    // gated norm + o_proj, against references over a synthetic core output
    std::vector<float> o(T * DI), g2a(T * HD), g2(T * DI), yk(T * DI), yout(T * HID);
    for (float& v : o) v = rnd();
    launch(T * HD, k5_gemv, view(ga.data(), HID), 0u, x.data(), g2a.data(), (unsigned)HD, (unsigned)T, (unsigned)HD, (unsigned)HID);
    launch(T * DI, k5_gemv, view(gb.data(), HD), 0u, g2a.data(), g2.data(), (unsigned)DI, (unsigned)T, (unsigned)DI, (unsigned)HD);
    launch(T * DI, k5_gated_norm, o.data(), on.data(), g2.data(), yk.data(), (unsigned)NH, (unsigned)HD, EPS, (unsigned)T);
    std::vector<float> yref(T * DI);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < DI; c++) {
            int h = c / HD, j = c % HD;
            double s = 0;
            for (int d = 0; d < HD; d++) { double v = o[t * DI + h * HD + d]; s += v * v; }
            float r = (float)(1.0 / std::sqrt(s / HD + (double)EPS));
            yref[t * DI + c] = o[t * DI + c] * r * on[j] / (1.f + std::exp(-g2[t * DI + c]));
        }
    report("k5_gated_norm vs reference", maxerr(yk, yref), 1e-6);
    launch(T * HID, k5_gemv, view(wo.data(), DI), 0u, yk.data(), yout.data(), (unsigned)HID, (unsigned)T, (unsigned)HID, (unsigned)DI);
    std::vector<float> oref(T * HID);
    for (int t = 0; t < T; t++)
        for (int r = 0; r < HID; r++) oref[t * HID + r] = ref_dot(&wo[r * DI], &yk[t * DI], DI);
    report("k5_gemv o_proj vs reference", maxerr(yout, oref), 2e-5);

    // the softplus form (lb absent -> NaN sentinel); the kernel gets the RAW gemv output
    // (it adds dt itself) — graw[i] must equal gpre[i] - dt[chan]
    {
        std::vector<float> gk = graw;
        launch(T * DI, k5_gate_finish, gk.data(), dt.data(), rate.data(), (unsigned)NH, (unsigned)HD, std::nanf(""), (unsigned)T);
        double e = 0;
        int wi = -1;
        for (int i = 0; i < T * DI; i++) {
            float v = rate[(i / HD) % NH] * (graw[i] + dt[i % (NH * HD)]);
            float want = -(v > 20.f ? v : std::log(1.f + std::exp(v)));
            double d = (double)std::fabs(gk[i] - want);
            if (d > e) { e = d; wi = i; }
        }
        float vw = rate[(wi / HD) % NH] * (graw[wi] + dt[wi % (NH * HD)]);
        report("k5_gate_finish softplus form", e, 1e-5);
    }

    std::printf(failures ? "FAILURES: %d\n" : "all ok (%d failures)\n", failures);
    return failures ? 1 : 0;
}
