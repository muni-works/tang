// DS_MEGA lane: launch-floor probes and the persistent dense-glue kernel.

// One dependent step: every block bumps its own counter (the stream orders the launches).
extern "C" __global__ void mg_tick(int* p) { DS_PDL_ENTRY
    if (threadIdx.x == 0) p[blockIdx.x] += 1;
}

// A grid barrier as ds_hc_fused_ks's: cnt[0] arrivals, cnt[1] the generation; every block
// co-resident. Left consistent (cnt[0] = 0) for the next use.
__device__ __forceinline__ void mg_grid_sync(int* cnt, int nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile int* gen = cnt + 1;
        int g0 = *gen;
        __threadfence();
        if (atomicAdd(cnt, 1) == nb - 1) {
            cnt[0] = 0;
            __threadfence();
            atomicAdd(cnt + 1, 1);
        } else {
            while (*gen == g0) __nanosleep(32);
        }
        __threadfence();
    }
    __syncthreads();
}

// `rounds` grid barriers back to back, a counter bump between them.
extern "C" __global__ void mg_barrier_loop(int* p, int* cnt, int rounds) { DS_PDL_ENTRY
    int nb = gridDim.x;
    for (int r = 0; r < rounds; r++) {
        if (threadIdx.x == 0) p[blockIdx.x] += 1;
        mg_grid_sync(cnt, nb);
    }
}

// ---- the dense-glue megakernel (DS_MEGA=1) ----
//
// One persistent launch for the stretch between a layer's MoE (all-reduced) and its
// attention core, t <= 8 rows: what ds_hc_fused_ks, ds_ks_quant, ds_ks (wq_a), ds_ks (wkv),
// the ring copy, ds_qkv_post and ds_ks (wq_b) did as seven launches, separated by grid
// barriers instead of kernel boundaries. Every phase computes what its kernel did, in the
// same order, so the outputs are bit for bit those of the separate launches. Phases:
//   A  blocks < gx: the mHC stage (ds_hc_fused_ks's math, its barrier among those gx
//      blocks), and the block input's K-split quantization (ds_ks_quant) folded into its
//      store; blocks >= gx: the ring copy for a verify window and L2 prefetches of the
//      three projections' weights.
//   B  lin_mma wq_a, then lin_mma wkv (own partials / counters each)
//   C  ds_qkv_post's work items (token, part), its 1024-thread reductions replayed in
//      the same order by 256 threads
//   D  lin_mma wq_b
// The grid must be co-resident (the host sizes it from the occupancy).

struct MgArgs {
    // A
    const float* h;
    const float* h2;
    const float* res;
    const float* post;
    const float* comb;
    float* xo;
    float* hpart;
    const float* pre_i;
    float* hin;
    const float* scale;
    const float* base;
    float* pre_o;
    float* post_o;
    float* comb_o;
    unsigned char* xc;
    float* xs;
    float* ring;
    float* all;
    // B, C, D
    float* qr;
    float* kv;
    unsigned char* qc;
    float* qsc;
    const float2* tab;
    const int* P;
    float* q;
    float* part1;
    float* part2;
    float* part3;
    int* cnt1;
    int* cnt2;
    int* cnt3;
    int* gbar;  // [2] the grid barrier
    int* hbar;  // [2] phase A's barrier among gx blocks
    DsW w, nw, wqa, wkv, wqb, qn, kn;
    int t, hc, d, iters, gx, rd, win, ncopy;
    float eps, neps, reps, qeps;
    int S1, SB1, NB1, S2, SB2, NB2, S3, SB3, NB3;
    int pf_bytes;  // L2 prefetch budget of wq_b (bytes)
};

__device__ __forceinline__ void mg_pf(const void* p, long long n, int who, int nwho) {
    const char* c = (const char*)p;
    for (long long o = (long long)who * 128; o < n; o += (long long)nwho * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(c + o));
}

__device__ __forceinline__ long long mg_wbytes(const DsW& w) {
    return (long long)w.rows * (w.fmt == 3 ? w.cols / 2 : w.cols);
}

// ds_hc_fused_ks's body (hc_fused_dev, shared so the two cannot drift) for block bx of
// np = a.gx (t0 = 0), + the ks quant of hin.
__device__ void mg_hc(const MgArgs& a, int bx) {
    hc_fused_dev(a.h, a.h2, a.res, a.post, a.comb, a.xo, a.w, a.hpart, a.pre_i, a.hin, a.nw, a.scale, a.base,
                 a.pre_o, a.post_o, a.comb_o, a.hbar, a.t, a.hc, a.d, a.iters, a.eps, a.neps, a.reps, 0, bx, 0, a.gx,
                 a.xc, a.xs);
}

// A block's sum over virtual threads v = tid + 256 k (k < 4) of ds_qkv_post's 1024-thread
// reduction, in its order: per virtual thread the strided partial, the warp xor tree per
// virtual warp, then warp 0's tree over the 32 warp sums.
__device__ __forceinline__ float mg_sum1024(const float* r, int n, float* red) {
    int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        float s = 0.f;
        for (int i = tid + 256 * k; i < n; i += 1024) {
            float v = __ldcg(r + i);
            s += v * v;
        }
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[wp + 8 * k] = s;
    }
    __syncthreads();
    if (tid < 32) {
        float s = red[tid];
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (tid == 0) red[32] = s;
    }
    __syncthreads();
    float v = red[32];
    __syncthreads();
    return v;
}

__device__ void mg_qkv_post(const MgArgs& a, int ti, int isq) {
    __shared__ float red[33];
    DsW w = isq ? a.qn : a.kn;
    int n = w.cols;
    float* r = (isq ? a.qr : a.kv) + (long long)ti * n;
    float s = mg_sum1024(r, n, red);
    float rs = 1.f / sqrtf(s / n + a.qeps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] = bf(wget(w, 0, i) * (__ldcg(r + i) * rs));
    __syncthreads();
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (isq) {
        for (int g = wid; g < n / 32; g += nw) {
            float xv = r[g * 32 + lane], am = fabsf(xv);
#pragma unroll
            for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            float sc = pow2_ceil(fmaxf(am, 1e-4f) / 448.f);
            a.qc[(long long)ti * n + g * 32 + lane] =
                (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(xv / sc, -448.f), 448.f)));
            if (lane == 0) a.qsc[(long long)ti * (n / 32) + g] = sc;
        }
        return;
    }
    int half = a.rd / 2, pos = __ldcg(a.P) + ti;
    float* tp = r + (n - a.rd);
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float2 cs = a.tab[(long long)pos * half + j];
        float x = tp[2 * j], b = tp[2 * j + 1];
        tp[2 * j] = bf(x * cs.x - b * cs.y);
        tp[2 * j + 1] = bf(x * cs.y + b * cs.x);
    }
    __syncthreads();
    float* rr = a.ring + (long long)(pos % a.win) * n;
    float* ar = a.all ? a.all + (long long)(a.win + ti) * n : 0;
    for (int g = wid; g < n / 32; g += nw) {
        float x = r[g * 32 + lane], amax = fabsf(x);
#pragma unroll
        for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float sc = pow2_ceil(fmaxf(amax, 1e-4f) / 448.f);
        float v = bf(round_e4m3(fminf(fmaxf(x / sc, -448.f), 448.f)) * sc);
        r[g * 32 + lane] = v;
        rr[g * 32 + lane] = v;
        if (ar) ar[g * 32 + lane] = v;
    }
}

template <int F>
__device__ void mg_glue_body(const MgArgs& a) {
    int bx = blockIdx.x, nb = gridDim.x;
    // A
    if (bx < a.gx) {
        mg_hc(a, bx);
    } else {
        int who = (bx - a.gx) * blockDim.x + threadIdx.x, nwho = (nb - a.gx) * blockDim.x;
        for (long long e = who; e < a.ncopy; e += nwho) a.all[e] = a.ring[e];
        mg_pf((const void*)a.wqa.w, mg_wbytes(a.wqa), who, nwho);
        mg_pf((const void*)a.wkv.w, mg_wbytes(a.wkv), who, nwho);
        mg_pf((const void*)a.wqb.w, min((long long)a.pf_bytes, mg_wbytes(a.wqb)), who, nwho);
    }
    mg_grid_sync(a.gbar, nb);
    // B
    lin_mma<F, 0>(a.wqa, a.wqa, a.xc, a.xs, a.qr, a.t, 1, 0, 0.f, a.S1, a.SB1, a.NB1, a.part1, a.cnt1, 0);
    lin_mma<F, 0>(a.wkv, a.wkv, a.xc, a.xs, a.kv, a.t, 1, 0, 0.f, a.S2, a.SB2, a.NB2, a.part2, a.cnt2, 0);
    mg_grid_sync(a.gbar, nb);
    // C
    for (int it = bx; it < 2 * a.t; it += nb) mg_qkv_post(a, it >> 1, (it & 1) == 0);
    mg_grid_sync(a.gbar, nb);
    // D
    lin_mma<F, 0>(a.wqb, a.wqb, a.qc, a.qsc, a.q, a.t, 1, 0, 0.f, a.S3, a.SB3, a.NB3, a.part3, a.cnt3, 0);
}

extern "C" __global__ void __launch_bounds__(256) mg_glue_f2(MgArgs a) { DS_PDL_ENTRY mg_glue_body<2>(a); }
extern "C" __global__ void __launch_bounds__(256) mg_glue_f3(MgArgs a) { DS_PDL_ENTRY mg_glue_body<3>(a); }

// ---- programmatic dependent launch probes (DS_PDL) ----
__device__ __forceinline__ void mg_pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void mg_pdl_trigger() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// A dependent step with an independent prologue: stream n16 uint4 of "weights" (the part a
// PDL launch may run under its predecessor's tail), then wait on the predecessor and bump the
// counters it bumped (the dependent part). sink keeps the loads alive.
extern "C" __global__ void mg_pdl_step(const uint4* __restrict__ w, long long n16, int* p, unsigned int* sink) {
    mg_pdl_trigger();
    unsigned int acc = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
        uint4 v = __ldg(w + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    mg_pdl_wait();
    if (acc == 0x9e3779b9u) sink[0] = acc;
    if (threadIdx.x == 0) p[blockIdx.x] += 1;
}
