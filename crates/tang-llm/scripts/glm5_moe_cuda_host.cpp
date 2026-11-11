// Host check of crates/tang-llm/src/flash/glm5/moe_cuda.cuh without a GPU: compiles the exact
// kernel source as C++ (a few CUDA names stubbed below), runs each kernel over an emulated grid
// (every blockIdx.x x threadIdx.x, block 128), and compares with the HF fixtures and llama.cpp's
// gguf-py dequantisation. Run: crates/tang-llm/scripts/glm5_moe_cuda_host.sh
// This checks the kernel *logic* (indexing, decoders, routing, clamps); NVRTC compilation and real
// GPU execution are the pod's cuda-gated tests.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#define __global__
#define __device__
#define __forceinline__ inline
struct G5Dim { unsigned x, y, z; };
static G5Dim blockIdx, blockDim, threadIdx;
static inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }

#include "../src/flash/glm5/moe_cuda.cuh"

static std::string DIR;
static int failures = 0;

static std::vector<unsigned char> rd(const std::string& name) {
    std::string fn = name;
    std::replace(fn.begin(), fn.end(), '.', '_');
    FILE* f = std::fopen((DIR + "/" + fn + ".bin").c_str(), "rb");
    if (!f) { std::fprintf(stderr, "missing fixture %s\n", fn.c_str()); std::exit(2); }
    std::vector<unsigned char> v;
    unsigned char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(f);
    return v;
}
static std::vector<float> rf(const std::string& name) {
    auto b = rd(name);
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

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

static G5W view(const void* p, size_t estride, size_t row_bytes, unsigned ty) {
    G5W w;
    w.base = (unsigned long long)p;
    w.estride = estride;
    w.row_bytes = (unsigned)row_bytes;
    w.ty = ty;
    return w;
}

const int HID = 64, NE = 16, K = 4, FF = 32;

struct Moe {
    G5W router, gate, up, down, sg, su, sd;
    const float* bias;
    bool shared;
    int ng, tg, ffs;
    float lim;
};

// the launch sequence of moe_gpu.rs::ffn
static std::vector<float> run_moe(const Moe& m, const std::vector<float>& x, int T,
                                  std::vector<unsigned>* ids_out = nullptr, std::vector<float>* wts_out = nullptr) {
    std::vector<float> s(T * NE), wts(T * K), h(T * K * FF), yp(T * K * HID), y(T * HID), ysh, hs;
    std::vector<unsigned> ids(T * K);
    launch(T * NE, g5_router_logits, m.router, x.data(), s.data(), T, NE, HID);
    launch(T, g5_router_select, (const float*)s.data(), m.bias, ids.data(), wts.data(), T, NE, K, m.ng, m.tg, 1, 2.5f);
    launch(T * K * FF, g5_expert_gu, m.gate, m.up, (const unsigned*)ids.data(), K, NE, x.data(), h.data(), T * K, FF, HID, m.lim);
    launch(T * K * HID, g5_expert_down, m.down, (const unsigned*)ids.data(), NE, (const float*)h.data(), yp.data(), T * K, FF, HID);
    if (m.shared) {
        hs.resize(T * m.ffs);
        ysh.resize(T * HID);
        launch(T * m.ffs, g5_expert_gu, m.sg, m.su, (const unsigned*)nullptr, 1, 0, x.data(), hs.data(), T, m.ffs, HID, m.lim);
        launch(T * HID, g5_expert_down, m.sd, (const unsigned*)nullptr, 0, (const float*)hs.data(), ysh.data(), T, m.ffs, HID);
    }
    launch(T * HID, g5_combine, (const float*)yp.data(), (const float*)wts.data(), m.shared ? (const float*)ysh.data() : nullptr, y.data(), T, K, HID);
    if (ids_out) *ids_out = ids;
    if (wts_out) *wts_out = wts;
    return y;
}

// per token: same expert set, max weight error
static double route_err(const std::vector<unsigned>& ids, const std::vector<float>& wts,
                        const std::vector<float>& want_ids, const std::vector<float>& want_wts, int T) {
    double e = 0;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < K; j++) {
            int hit = -1;
            for (int q = 0; q < K; q++) if ((unsigned)want_ids[t * K + q] == ids[t * K + j]) hit = q;
            if (hit < 0) return 1e30;
            e = std::max(e, (double)std::fabs(wts[t * K + j] - want_wts[t * K + hit]));
        }
    return e;
}

int main(int argc, char** argv) {
    DIR = argc > 1 ? argv[1] : "crates/tang-llm/tests/fixtures/glm5/moe";

    // 1. decoders vs gguf-py, on rows of 512 elements (2 K-quant blocks, 16 Q8_0/Q5_0 blocks;
    // the Q5_0 fixture's expected values come from the ggml formula — see glm5_moe_fixtures.py)
    const struct { const char* name; unsigned ty; } q[] = {{"f16", 1}, {"q8_0", 8}, {"q5_0", 6}, {"q4_k", 12}, {"q5_k", 13}, {"q6_k", 14}};
    for (auto& k : q) {
        auto raw = rd(std::string("quant_") + k.name + "_raw");
        auto want = rf(std::string("quant_") + k.name + "_want");
        std::vector<float> got(want.size());
        double amax = 0;
        for (size_t j = 0; j < want.size(); j++) { got[j] = g5_deq(k.ty, raw.data(), (int)j); amax = std::max(amax, (double)std::fabs(want[j])); }
        report((std::string("g5_deq ") + k.name + " vs gguf-py").c_str(), maxerr(got, want), 1e-6 * std::max(1.0, amax));
    }

    auto x = rf("x");
    const int T = (int)x.size() / HID;
    auto rw = rf("router.weight"), rb = rf("router.bias");

    // 2. router alone (via the full pipeline's ids/wts) and the limit-1 experts, no shared expert
    {
        auto gu = rf("experts_l1.gate_up_proj"), dn = rf("experts_l1.down_proj");
        Moe m{};
        m.router = view(rw.data(), 0, HID * 4, 0);
        m.bias = rb.data();
        m.gate = view(gu.data(), 2 * FF * HID * 4, HID * 4, 0);
        m.up = view(gu.data() + FF * HID, 2 * FF * HID * 4, HID * 4, 0);  // HF fused layout: up rows follow gate
        m.down = view(dn.data(), HID * FF * 4, FF * 4, 0);
        m.ng = 1; m.tg = 1; m.lim = 1.0f;
        std::vector<unsigned> ids;
        std::vector<float> wts;
        auto y = run_moe(m, x, T, &ids, &wts);
        report("router (ids + weights) vs HF", route_err(ids, wts, rf("router_out.indices"), rf("router_out.weights"), T), 1e-5);
        report("experts, swiglu limit 1 vs HF", maxerr(y, rf("experts_l1_out")), 1e-4);
    }
    // 3. group-limited router
    {
        auto gw = rf("router_g4.weight"), gb = rf("router_g4.bias");
        auto gu = rf("experts.gate_up_proj"), dn = rf("experts.down_proj");
        Moe m{};
        m.router = view(gw.data(), 0, HID * 4, 0);
        m.bias = gb.data();
        m.gate = view(gu.data(), 2 * FF * HID * 4, HID * 4, 0);
        m.up = view(gu.data() + FF * HID, 2 * FF * HID * 4, HID * 4, 0);
        m.down = view(dn.data(), HID * FF * 4, FF * 4, 0);
        m.ng = 4; m.tg = 2; m.lim = 10.0f;
        std::vector<unsigned> ids;
        std::vector<float> wts;
        run_moe(m, x, T, &ids, &wts);
        report("group-limited router (4 groups, keep 2) vs HF", route_err(ids, wts, rf("router_g4_out.indices"), rf("router_g4_out.weights"), T), 1e-5);
    }
    // 4. the whole MoE layer (router + experts + shared expert)
    {
        auto mw = rf("moe.gate.weight"), mb = rf("moe.gate.e_score_correction_bias");
        auto gu = rf("moe.experts.gate_up_proj"), dn = rf("moe.experts.down_proj");
        auto sg = rf("moe.shared_experts.gate_proj.weight"), su = rf("moe.shared_experts.up_proj.weight"), sd = rf("moe.shared_experts.down_proj.weight");
        Moe m{};
        m.router = view(mw.data(), 0, HID * 4, 0);
        m.bias = mb.data();
        m.gate = view(gu.data(), 2 * FF * HID * 4, HID * 4, 0);
        m.up = view(gu.data() + FF * HID, 2 * FF * HID * 4, HID * 4, 0);
        m.down = view(dn.data(), HID * FF * 4, FF * 4, 0);
        m.sg = view(sg.data(), 0, HID * 4, 0);
        m.su = view(su.data(), 0, HID * 4, 0);
        m.sd = view(sd.data(), 0, FF * 4, 0);
        m.shared = true; m.ffs = FF; m.ng = 1; m.tg = 1; m.lim = 10.0f;
        report("full MoE layer vs HF", maxerr(run_moe(m, x, T), rf("moe_out")), 1e-4);
    }
    // 5. dense FFN, limit 1 (ids = null path)
    {
        const int FD = 256;
        auto g = rf("dense_l1.gate_proj.weight"), u = rf("dense_l1.up_proj.weight"), d = rf("dense_l1.down_proj.weight");
        std::vector<float> h(T * FD), y(T * HID);
        launch(T * FD, g5_expert_gu, view(g.data(), 0, HID * 4, 0), view(u.data(), 0, HID * 4, 0), (const unsigned*)nullptr, 1, 0, x.data(), h.data(), T, FD, HID, 1.0f);
        launch(T * HID, g5_expert_down, view(d.data(), 0, FD * 4, 0), (const unsigned*)nullptr, 0, (const float*)h.data(), y.data(), T, FD, HID);
        report("dense FFN, swiglu limit 1 vs HF", maxerr(y, rf("dense_l1_out")), 1e-4);
    }
    // 6. sentinel id: an id >= ne contributes 0
    {
        float s[1] = {0.f};
        unsigned id[1] = {(unsigned)NE};
        float h[1] = {123.f};
        blockDim.x = 1; blockIdx.x = 0; threadIdx.x = 0;
        G5W z = view(s, 0, 4, 0);
        g5_expert_gu(z, z, id, 1, NE, s, h, 1, 1, 1, 10.f);
        report("sentinel expert id gives 0", std::fabs(h[0]), 0.0);
    }
    std::printf(failures ? "\n%d FAILED\n" : "\nall ok\n", failures);
    return failures ? 1 : 0;
}
