
// ---------------------------------------------------------------------------
// The fused KDA DECODE step (T = 1; docs/models/glm5-speed-plan.md step 2, kda_decode.rs).
// Compiled after the KDA prelude (KDA_H/KDA_D/KDA_KC/KDA_CONV) and moe_cuda.cuh (G5W,
// g5_dot). Every value is computed with the SAME expression as the naive kernel it
// replaces (k5_gemv / k5_beta / k5_gate_finish / kda_step / kda_conv_commit /
// k5_gated_norm), so the fused step equals the naive one to the bit; only the launch
// count changes (13 launches + 3 host round-trips -> 4 launches, no host traffic between).

// k5d_proj: every GEMV over the token's x, one thread per output row.
// rows [0, 3DI): q | k | v -> QKV (the stacked layout); [3DI, 3DI+HD): f_a -> G1;
// [.., +NH): beta (sigmoid) -> B; [.., +HD): g_a -> G2A. R0 = the first row this launch
// covers (3*DI when k5_gemv8 already wrote q/k/v). SKIP bit0: f_a is on k5_gemv8 (the
// naive path's per-matrix rule, kda_dev window()) — skip it here, the gemv8 launch wrote
// it; bit1: same for g_a. The f32-dequant g5_dot and gemv8's int8 dp4a round DIFFERENTLY,
// so a group must use the same kernel on both paths or the outputs diverge (the forget
// gate then amplifies the difference through the state — the first real-weights run).
// Grid ceil((3DI+2HD+NH-R0)/128).
extern "C" __global__ void k5d_proj(const G5W wq, const G5W wk, const G5W wv, const G5W fa,
                                    const G5W bp, const G5W ga, const float* x, float* qkv,
                                    float* g1, float* beta, float* g2a, unsigned DI,
                                    unsigned HD, unsigned NH, unsigned HID, unsigned R0,
                                    unsigned SKIP) {
    unsigned i = R0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI) { qkv[i] = g5_dot(wq, 0u, i, x, HID); return; }
    i -= DI;
    if (i < DI) { qkv[DI + i] = g5_dot(wk, 0u, i, x, HID); return; }
    i -= DI;
    if (i < DI) { qkv[2 * DI + i] = g5_dot(wv, 0u, i, x, HID); return; }
    i -= DI;
    if (i < HD) { if (!(SKIP & 1u)) g1[i] = g5_dot(fa, 0u, i, x, HID); return; }
    i -= HD;
    if (i < NH) { float a = g5_dot(bp, 0u, i, x, HID); beta[i] = 1.f / (1.f + expf(-a)); return; }
    i -= NH;
    if (i < HD) { if (!(SKIP & 2u)) g2a[i] = g5_dot(ga, 0u, i, x, HID); }
}

// k5d_gate2: the two hd-input GEMVs. rows [0, DI): f_b(g1) + dt_bias + the gate form
// (k5_gate_finish's expression) -> G; [DI, 2DI): g_b(g2a) -> G2. Grid ceil(2DI/128).
extern "C" __global__ void k5d_gate2(const G5W fb, const G5W gb, const float* g1,
                                     const float* g2a, const float* dt, const float* rate,
                                     float* g, float* g2, unsigned NH, unsigned HD, float LB) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x, DI = NH * HD;
    if (i < DI) {
        // the naive path stores the GEMV result as a float first, then adds dt: same here
        float raw = g5_dot(fb, 0u, i, g1, HD);
        unsigned h = i / HD;
        float pre = raw + dt[i];
        float out;
        if (LB == LB) {
            out = LB / (1.f + expf(-rate[h] * pre));
        } else {
            float v = rate[h] * pre;
            out = -(v > 20.f ? v : logf(1.f + expf(v)));
        }
        g[i] = out;
        return;
    }
    i -= DI;
    if (i < DI) g2[i] = g5_dot(gb, 0u, i, g2a, HD);
}

// k5d_step: kda_step (fused conv + delta rule, verbatim) + the T=1 conv-history commit
// (each thread owns its 3 channels: it read their history in the conv, then shifts it —
// no other thread touches those channels) + the gated output norm (k5_gated_norm's f64
// sum of squares in the same d order). Grid KDA_H, block KDA_D. Writes O (pre-norm, the
// debug tap) and Y (normed, gated).
extern "C" __global__ void k5d_step(const float* __restrict__ QKV, float* __restrict__ H0,
                                    const float* __restrict__ W, const float* __restrict__ G,
                                    const float* __restrict__ B, float* __restrict__ S,
                                    float* __restrict__ O, const float* __restrict__ ON,
                                    const float* __restrict__ G2, float* __restrict__ Y,
                                    float EPS) {
    unsigned int h = blockIdx.x, x = threadIdx.x, D = KDA_D;
    __shared__ float q[KDA_D], k[KDA_D], sk[KDA_D], os[KDA_D];
    float qv = 0.0f, kv = 0.0f, vv = 0.0f;
    const unsigned int qc = h * D + x, kch = KDA_H * D + h * D + x, vch = 2 * KDA_H * D + h * D + x;
    {
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[qc * (KDA_KC - 1) + i] : QKV[qc];
            acc = __fmaf_rn(W[qc * KDA_KC + i], e, acc);
        }
        qv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[kch * (KDA_KC - 1) + i] : QKV[kch];
            acc = __fmaf_rn(W[kch * KDA_KC + i], e, acc);
        }
        kv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[vch * (KDA_KC - 1) + i] : QKV[vch];
            acc = __fmaf_rn(W[vch * KDA_KC + i], e, acc);
        }
        vv = acc / (1.0f + expf(-acc));
    }
    // the commit (kda_conv_commit at T=1): history <- [hist[1..], x]
    {
        const unsigned int chs[3] = {qc, kch, vch};
        for (unsigned int c = 0; c < 3; c++) {
            float* hp = H0 + chs[c] * (KDA_KC - 1);
            for (unsigned int s = 0; s < KDA_KC - 1; s++)
                hp[s] = s + 1 < KDA_KC - 1 ? hp[s + 1] : QKV[chs[c]];
        }
    }
    q[x] = qv; k[x] = kv;
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (unsigned int a = 0; a < D; a++) { qs += q[a] * q[a]; ks += k[a] * k[a]; }
    qs = rsqrtf(qs + 1e-6f); ks = rsqrtf(ks + 1e-6f);
    float scale = rsqrtf((float)D);
    __syncthreads(); // every thread has read q[]/k[] raw before any rescales them
    q[x] *= qs * scale; k[x] *= ks;
    __syncthreads();
    float p = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((unsigned long long)h * D + a) * D + x;
        float s = (*sp) * expf(G[h * D + a]);
        *sp = s;
        p = __fmaf_rn(k[a], s, p);
    }
    sk[x] = p;
    __syncthreads();
    float delta = B[h] * (vv - sk[x]);
    float o = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((unsigned long long)h * D + a) * D + x;
        float s = __fmaf_rn(k[a], delta, *sp);
        *sp = s;
        o = __fmaf_rn(q[a], s, o);
    }
    O[h * D + x] = o;
    // gated norm over the head (k5_gated_norm: f64 mean of squares, d ascending)
    os[x] = o;
    __syncthreads();
    double ss = 0.0;
    for (unsigned int d = 0; d < D; d++) { double v = (double)os[d]; ss += v * v; }
    float r = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    Y[h * D + x] = os[x] * r * ON[x] / (1.f + expf(-G2[h * D + x]));
}

// ---------------------------------------------------------------------------
// k5d_tail: everything between the x projections and o_proj in ONE launch (G5F path):
// beta (k5_beta's lane chains + xor reduce), the gate rows (k5d_gate2), the step
// (k5d_step), and o_proj's activation quantise (k5_q8 over yb). One block a head
// (grid KDA_H, block KDA_D): every input the head needs is head-local —
//   G / G2 rows h*D .. h*D+D-1 (f_b / g_b rows of this head, over the hd-wide g1 / g2a),
//   beta[h] (b_proj row h over x), the 3 conv channels each thread owns, the state S[h].
// Same expressions in the same order as the kernels it replaces (bit-equal), except the
// work layout: the head's f_b / g_b rows and the b_proj row are staged into shared memory
// with coalesced 16 B loads (k5d_gate2 had each thread stream its own 136 B row byte by
// byte: 52 us), the f16 block scale is decoded once per 32 (g5_deq decoded it per element:
// same value), expf(G) once per row (k5d_step recomputed it in every thread: same value),
// and S[h] is read ONCE into registers and written ONCE (k5d_step read it twice, wrote
// it twice: 16 MB/layer of traffic for a 4 MB state).
// Requires (the host checks, else it runs the old launches): f_b, g_b, b_proj Q8_0;
// KDA_D % 32 == 0; HID <= K5D_HIDMAX, HID % 32 == 0.
#define K5D_HIDMAX 4096
#define K5D_RB (KDA_D / 32 * 34)
__device__ __forceinline__ void k5d_stage(unsigned char* sm, const unsigned char* src, unsigned len,
                                          unsigned tid, unsigned nt, int* delta) {
    const unsigned long long a = (unsigned long long)src & ~15ull;
    *delta = (int)((unsigned long long)src - a);
    const unsigned n16 = ((unsigned)*delta + len + 15) >> 4;
    // cp.async (no register round trip): every 16 B granule of all three stages is in
    // flight at once, under the state loads and the conv — the old load-then-store loop
    // paid ~20 serial DRAM round trips per block. Waited on before the gate rows.
    for (unsigned i = tid; i < n16; i += nt) {
        const unsigned d = (unsigned)__cvta_generic_to_shared((uint4*)sm + i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"((const uint4*)a + i) : "memory");
    }
}
extern "C" __global__ void __launch_bounds__(KDA_D) k5d_tail(
    const G5W fb, const G5W gb, const G5W bp, const float* __restrict__ X,
    const float* __restrict__ G1, const float* __restrict__ G2A, const float* __restrict__ DT,
    const float* __restrict__ RATE, float LB, const float* __restrict__ QKV,
    float* __restrict__ H0, const float* __restrict__ W, float* __restrict__ S,
    const float* __restrict__ ON, float EPS, float* __restrict__ BETA, float* __restrict__ G,
    float* __restrict__ G2, float* __restrict__ O, float* __restrict__ Y,
    unsigned char* __restrict__ XQ, unsigned MAXT, unsigned HID, unsigned TI) {
    const unsigned h = blockIdx.x, x = threadIdx.x, D = KDA_D, DI = KDA_H * KDA_D;
    const unsigned lane = x & 31, warp = x >> 5;
    __shared__ __align__(16) unsigned char wf[KDA_D * K5D_RB + 16], wg[KDA_D * K5D_RB + 16];
    __shared__ __align__(16) unsigned char wb[K5D_HIDMAX / 32 * 34 + 16];
    __shared__ float g1s[KDA_D], g2s[KDA_D], q[KDA_D], k[KDA_D], eg[KDA_D], os[KDA_D];
    __shared__ float bs;
    __shared__ double sq[KDA_D];
    // x for beta, staged with the weights (dynamic: HID * 4 + 16 B, the host sets the
    // attribute) — warp 0's 16 serial L2 round trips for x were ~4 us of the critical path
    extern __shared__ __align__(16) unsigned char xs[];
    int df, dg, db, dx;
    k5d_stage(wf, (const unsigned char*)(fb.base + (unsigned long long)h * D * fb.row_bytes), D * K5D_RB, x, D, &df);
    k5d_stage(wg, (const unsigned char*)(gb.base + (unsigned long long)h * D * gb.row_bytes), D * K5D_RB, x, D, &dg);
    k5d_stage(wb, (const unsigned char*)(bp.base + (unsigned long long)h * bp.row_bytes), HID / 32 * 34, x, D, &db);
    k5d_stage(xs, (const unsigned char*)X, HID * 4, x, D, &dx);
    g1s[x] = G1[x]; // f_a / g_a outputs are hd wide, shared by every head
    g2s[x] = G2A[x];
    // the state column this thread owns, read once (independent loads, all in flight)
    float sr[KDA_D];
#pragma unroll
    for (unsigned a = 0; a < KDA_D; a++) sr[a] = S[((unsigned long long)h * D + a) * D + x];
    // conv (k5d_step verbatim) + the T=1 commit of the thread's 3 channels
    float qv, kv, vv;
    const unsigned int qc = h * D + x, kch = KDA_H * D + h * D + x, vch = 2 * KDA_H * D + h * D + x;
    {
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[qc * (KDA_KC - 1) + i] : QKV[qc];
            acc = __fmaf_rn(W[qc * KDA_KC + i], e, acc);
        }
        qv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[kch * (KDA_KC - 1) + i] : QKV[kch];
            acc = __fmaf_rn(W[kch * KDA_KC + i], e, acc);
        }
        kv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[vch * (KDA_KC - 1) + i] : QKV[vch];
            acc = __fmaf_rn(W[vch * KDA_KC + i], e, acc);
        }
        vv = acc / (1.0f + expf(-acc));
        const unsigned int chs[3] = {qc, kch, vch};
        for (unsigned int c = 0; c < 3; c++) {
            float* hp = H0 + chs[c] * (KDA_KC - 1);
            for (unsigned int s = 0; s < KDA_KC - 1; s++)
                hp[s] = s + 1 < KDA_KC - 1 ? hp[s + 1] : QKV[chs[c]];
        }
    }
    q[x] = qv; k[x] = kv;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    // gate rows (k5d_gate2: g5_dot's j-ascending acc += deq * x, deq = d * q)
    float gv, g2v;
    {
        const unsigned char* rf = wf + df + x * K5D_RB;
        const unsigned char* rg = wg + dg + x * K5D_RB;
        float af = 0.f, ag = 0.f;
        for (unsigned b = 0; b < D / 32; b++) {
            const float dfw = g5_h2f(g5_u16(rf + 34 * b)), dgw = g5_h2f(g5_u16(rg + 34 * b));
#pragma unroll
            for (unsigned i = 0; i < 32; i++) {
                af = fmaf(dfw * (float)(signed char)rf[34 * b + 2 + i], g1s[32 * b + i], af);
                ag = fmaf(dgw * (float)(signed char)rg[34 * b + 2 + i], g2s[32 * b + i], ag);
            }
        }
        float pre = af + DT[h * D + x];
        if (LB == LB) {
            gv = LB / (1.f + expf(-RATE[h] * pre));
        } else {
            float v = RATE[h] * pre;
            gv = -(v > 20.f ? v : logf(1.f + expf(v)));
        }
        g2v = ag;
        G[h * D + x] = gv;
        G2[h * D + x] = g2v;
        eg[x] = expf(gv);
    }
    // beta (k5_beta: lane chain e0 = lane*8 + 256i, 8 a step, xor reduce, sigmoid)
    if (warp == 0) {
        const unsigned char* row = wb + db;
        float a = 0.f;
        for (int e0 = lane * 8; e0 < (int)HID; e0 += 256) {
            const unsigned char* bb = row + (e0 / 32) * 34;
            const float dw = g5_h2f(g5_u16(bb));
            const float* xp = (const float*)(xs + dx) + e0;
            const float xv[8] = {xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7]};
#pragma unroll
            for (int qq = 0; qq < 8; qq++) a = fmaf(dw * (float)(signed char)bb[2 + (e0 + qq) % 32], xv[qq], a);
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) { float bt = 1.f / (1.f + expf(-a)); BETA[h] = bt; bs = bt; }
    }
    float qs = 0.0f, ks = 0.0f;
    for (unsigned int a = 0; a < D; a++) { qs += q[a] * q[a]; ks += k[a] * k[a]; }
    qs = rsqrtf(qs + 1e-6f); ks = rsqrtf(ks + 1e-6f);
    float scale = rsqrtf((float)D);
    __syncthreads(); // q/k read raw by all, eg / bs written
    q[x] *= qs * scale; k[x] *= ks;
    __syncthreads();
    float p = 0.0f;
#pragma unroll
    for (unsigned int a = 0; a < KDA_D; a++) {
        float s = sr[a] * eg[a];
        sr[a] = s;
        p = __fmaf_rn(k[a], s, p);
    }
    float delta = bs * (vv - p);
    float o = 0.0f;
#pragma unroll
    for (unsigned int a = 0; a < KDA_D; a++) {
        float s = __fmaf_rn(k[a], delta, sr[a]);
        S[((unsigned long long)h * D + a) * D + x] = s;
        o = __fmaf_rn(q[a], s, o);
    }
    O[h * D + x] = o;
    os[x] = o;
    // (double)o * (double)o is exact (24-bit mantissas), so ss + v*v rounds once either
    // way: the squares are formed in parallel and thread 0 runs only the DADD chain
#ifdef K5D_NORM_EXACT
    sq[x] = (double)o * (double)o;
    __syncthreads();
    // the f64 norm once per head (thread 0) and broadcast: every thread used to run the
    // same 128-long f64 chain, and FP64 is scarce on the B300 (~65 cycles a dependent
    // DADD): same sums in the same order, bit-equal
    if (x == 0) {
        double ss = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < KDA_D; d++) ss += sq[d];
        g1s[0] = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    }
    __syncthreads();
#else
    // default (NOTES 2026-10-28 kda4): the same f64 squares summed as a tree (xor shuffle
    // within each warp, then the KDA_D/32 warp partials in order by thread 0): depth
    // 5 + KDA_D/32 dependent DADDs instead of KDA_D. A summation-order change in f64, so
    // the f32 r almost always rounds the same. TANG_GLM5_KDA_NORM=exact: the serial chain.
    {
        double v = (double)o * (double)o;
        for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o2);
        if (lane == 0) sq[warp] = v;
    }
    __syncthreads();
    if (x == 0) {
        double ss = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < KDA_D / 32; d++) ss += sq[d];
        g1s[0] = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    }
    __syncthreads();
#endif
    const float r = g1s[0];
    const float yv = os[x] * r * ON[x] / (1.f + expf(-g2v));
    Y[h * D + x] = yv;
    // o_proj's k5_q8 over yb (COLS = DI, t = 0): this warp's 32 lanes are one Q8 block
    if (XQ) {
        float amax = fabsf(yv);
        for (int o2 = 16; o2 > 0; o2 >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o2));
        const float d = amax / 127.0f;
        const int qi = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(yv / d), -127.f), 127.f);
        // TI = this row's index in the window's k5_q8 layout (0 for a decode step)
        XQ[(size_t)TI * DI + h * D + x] = (unsigned char)(signed char)qi;
        if (lane == 0) ((float*)(XQ + (size_t)MAXT * DI))[TI * (DI / 32) + (h * D + x) / 32] = d;
    }
}

// k5d_tailt: k5d_tail over the T rows of one window in ONE launch (NOTES 2026-10-29
// kda4 round 4): the weights are staged once, the state column stays in registers across
// rows (S written once, after the last row); per row the body is k5d_tail verbatim (same
// expressions, same order: bit-equal to T t=1 steps). CK != 0: after each row but the
// last, that row's conv history and state go to CK + row*CKSTRIDE (h0 region, H0N floats)
// and the state region after it - the per-row path's checkpoint layout.
extern "C" __global__ void __launch_bounds__(KDA_D) k5d_tailt(
    const G5W fb, const G5W gb, const G5W bp, const float* __restrict__ X_,
    const float* __restrict__ G1_, const float* __restrict__ G2A_, const float* __restrict__ DT,
    const float* __restrict__ RATE, float LB, const float* __restrict__ QKV_,
    float* __restrict__ H0, const float* __restrict__ W, float* __restrict__ S,
    const float* __restrict__ ON, float EPS, float* __restrict__ BETA_, float* __restrict__ G_,
    float* __restrict__ G2_, float* __restrict__ O_, float* __restrict__ Y_,
    unsigned char* __restrict__ XQ, unsigned MAXT, unsigned HID, unsigned T,
    float* __restrict__ CK, unsigned long long CKSTRIDE, unsigned long long H0N) {
    const unsigned h = blockIdx.x, x = threadIdx.x, D = KDA_D, DI = KDA_H * KDA_D;
    const unsigned lane = x & 31, warp = x >> 5;
    __shared__ __align__(16) unsigned char wf[KDA_D * K5D_RB + 16], wg[KDA_D * K5D_RB + 16];
    __shared__ __align__(16) unsigned char wb[K5D_HIDMAX / 32 * 34 + 16];
    __shared__ float g1s[KDA_D], g2s[KDA_D], q[KDA_D], k[KDA_D], eg[KDA_D], os[KDA_D];
    __shared__ float bs;
    __shared__ double sq[KDA_D];
    // x for beta, staged with the weights (dynamic: HID * 4 + 16 B, the host sets the
    // attribute) — warp 0's 16 serial L2 round trips for x were ~4 us of the critical path
    extern __shared__ __align__(16) unsigned char xs[];
    int df, dg, db, dx;
    k5d_stage(wf, (const unsigned char*)(fb.base + (unsigned long long)h * D * fb.row_bytes), D * K5D_RB, x, D, &df);
    k5d_stage(wg, (const unsigned char*)(gb.base + (unsigned long long)h * D * gb.row_bytes), D * K5D_RB, x, D, &dg);
    k5d_stage(wb, (const unsigned char*)(bp.base + (unsigned long long)h * bp.row_bytes), HID / 32 * 34, x, D, &db);
    // the state column this thread owns, read once for all T rows
    float sr[KDA_D];
#pragma unroll
    for (unsigned a = 0; a < KDA_D; a++) sr[a] = S[((unsigned long long)h * D + a) * D + x];
    for (unsigned TI = 0; TI < T; TI++) {
    if (TI) __syncthreads(); // the previous row is done with xs / g1s / q / k / eg / os / sq
    const float* X = X_ + (size_t)TI * HID; const float* G1 = G1_ + TI * D; const float* G2A = G2A_ + TI * D;
    const float* QKV = QKV_ + (size_t)TI * 3 * DI; float* BETA = BETA_ + TI * KDA_H;
    float* G = G_ + (size_t)TI * DI; float* G2 = G2_ + (size_t)TI * DI; float* O = O_ + (size_t)TI * DI; float* Y = Y_ + (size_t)TI * DI;
    const bool ck = CK && TI + 1 < T;
    float* CKH = CK + TI * CKSTRIDE; float* CKS = CKH + H0N;
    k5d_stage(xs, (const unsigned char*)X, HID * 4, x, D, &dx);
    g1s[x] = G1[x]; // f_a / g_a outputs are hd wide, shared by every head
    g2s[x] = G2A[x];
    // the state column this thread owns, read once (independent loads, all in flight)
    // conv (k5d_step verbatim) + the T=1 commit of the thread's 3 channels
    float qv, kv, vv;
    const unsigned int qc = h * D + x, kch = KDA_H * D + h * D + x, vch = 2 * KDA_H * D + h * D + x;
    {
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[qc * (KDA_KC - 1) + i] : QKV[qc];
            acc = __fmaf_rn(W[qc * KDA_KC + i], e, acc);
        }
        qv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[kch * (KDA_KC - 1) + i] : QKV[kch];
            acc = __fmaf_rn(W[kch * KDA_KC + i], e, acc);
        }
        kv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[vch * (KDA_KC - 1) + i] : QKV[vch];
            acc = __fmaf_rn(W[vch * KDA_KC + i], e, acc);
        }
        vv = acc / (1.0f + expf(-acc));
        const unsigned int chs[3] = {qc, kch, vch};
        for (unsigned int c = 0; c < 3; c++) {
            float* hp = H0 + chs[c] * (KDA_KC - 1);
            for (unsigned int s = 0; s < KDA_KC - 1; s++)
                hp[s] = s + 1 < KDA_KC - 1 ? hp[s + 1] : QKV[chs[c]];
            if (ck)
                for (unsigned int s = 0; s < KDA_KC - 1; s++) CKH[chs[c] * (KDA_KC - 1) + s] = hp[s];
        }
    }
    q[x] = qv; k[x] = kv;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    // gate rows (k5d_gate2: g5_dot's j-ascending acc += deq * x, deq = d * q)
    float gv, g2v;
    {
        const unsigned char* rf = wf + df + x * K5D_RB;
        const unsigned char* rg = wg + dg + x * K5D_RB;
        float af = 0.f, ag = 0.f;
        for (unsigned b = 0; b < D / 32; b++) {
            const float dfw = g5_h2f(g5_u16(rf + 34 * b)), dgw = g5_h2f(g5_u16(rg + 34 * b));
#pragma unroll
            for (unsigned i = 0; i < 32; i++) {
                af = fmaf(dfw * (float)(signed char)rf[34 * b + 2 + i], g1s[32 * b + i], af);
                ag = fmaf(dgw * (float)(signed char)rg[34 * b + 2 + i], g2s[32 * b + i], ag);
            }
        }
        float pre = af + DT[h * D + x];
        if (LB == LB) {
            gv = LB / (1.f + expf(-RATE[h] * pre));
        } else {
            float v = RATE[h] * pre;
            gv = -(v > 20.f ? v : logf(1.f + expf(v)));
        }
        g2v = ag;
        G[h * D + x] = gv;
        G2[h * D + x] = g2v;
        eg[x] = expf(gv);
    }
    // beta (k5_beta: lane chain e0 = lane*8 + 256i, 8 a step, xor reduce, sigmoid)
    if (warp == 0) {
        const unsigned char* row = wb + db;
        float a = 0.f;
        for (int e0 = lane * 8; e0 < (int)HID; e0 += 256) {
            const unsigned char* bb = row + (e0 / 32) * 34;
            const float dw = g5_h2f(g5_u16(bb));
            const float* xp = (const float*)(xs + dx) + e0;
            const float xv[8] = {xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7]};
#pragma unroll
            for (int qq = 0; qq < 8; qq++) a = fmaf(dw * (float)(signed char)bb[2 + (e0 + qq) % 32], xv[qq], a);
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) { float bt = 1.f / (1.f + expf(-a)); BETA[h] = bt; bs = bt; }
    }
    float qs = 0.0f, ks = 0.0f;
    for (unsigned int a = 0; a < D; a++) { qs += q[a] * q[a]; ks += k[a] * k[a]; }
    qs = rsqrtf(qs + 1e-6f); ks = rsqrtf(ks + 1e-6f);
    float scale = rsqrtf((float)D);
    __syncthreads(); // q/k read raw by all, eg / bs written
    q[x] *= qs * scale; k[x] *= ks;
    __syncthreads();
    float p = 0.0f;
#pragma unroll
    for (unsigned int a = 0; a < KDA_D; a++) {
        float s = sr[a] * eg[a];
        sr[a] = s;
        p = __fmaf_rn(k[a], s, p);
    }
    float delta = bs * (vv - p);
    float o = 0.0f;
#pragma unroll
    for (unsigned int a = 0; a < KDA_D; a++) {
        float s = __fmaf_rn(k[a], delta, sr[a]);
        sr[a] = s;
        if (TI + 1 == T) S[((unsigned long long)h * D + a) * D + x] = s;
        else if (ck) CKS[((unsigned long long)h * D + a) * D + x] = s;
        o = __fmaf_rn(q[a], s, o);
    }
    O[h * D + x] = o;
    os[x] = o;
    // (double)o * (double)o is exact (24-bit mantissas), so ss + v*v rounds once either
    // way: the squares are formed in parallel and thread 0 runs only the DADD chain
#ifdef K5D_NORM_EXACT
    sq[x] = (double)o * (double)o;
    __syncthreads();
    // the f64 norm once per head (thread 0) and broadcast: every thread used to run the
    // same 128-long f64 chain, and FP64 is scarce on the B300 (~65 cycles a dependent
    // DADD): same sums in the same order, bit-equal
    if (x == 0) {
        double ss = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < KDA_D; d++) ss += sq[d];
        g1s[0] = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    }
    __syncthreads();
#else
    // default (NOTES 2026-10-28 kda4): the same f64 squares summed as a tree (xor shuffle
    // within each warp, then the KDA_D/32 warp partials in order by thread 0): depth
    // 5 + KDA_D/32 dependent DADDs instead of KDA_D. A summation-order change in f64, so
    // the f32 r almost always rounds the same. TANG_GLM5_KDA_NORM=exact: the serial chain.
    {
        double v = (double)o * (double)o;
        for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o2);
        if (lane == 0) sq[warp] = v;
    }
    __syncthreads();
    if (x == 0) {
        double ss = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < KDA_D / 32; d++) ss += sq[d];
        g1s[0] = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    }
    __syncthreads();
#endif
    const float r = g1s[0];
    const float yv = os[x] * r * ON[x] / (1.f + expf(-g2v));
    Y[h * D + x] = yv;
    // o_proj's k5_q8 over yb (COLS = DI, t = 0): this warp's 32 lanes are one Q8 block
    if (XQ) {
        float amax = fabsf(yv);
        for (int o2 = 16; o2 > 0; o2 >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o2));
        const float d = amax / 127.0f;
        const int qi = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(yv / d), -127.f), 127.f);
        // TI = this row's index in the window's k5_q8 layout (0 for a decode step)
        XQ[(size_t)TI * DI + h * D + x] = (unsigned char)(signed char)qi;
        if (lane == 0) ((float*)(XQ + (size_t)MAXT * DI))[TI * (DI / 32) + (h * D + x) / 32] = d;
    }
    } // rows
}

// The window tail split in two (NOTES 2026-10-29 kda4 round 5), bit-equal per row to
// k5d_tail: k5d_pret (grid nh x T) does everything that does not depend on the
// recurrent state — conv (from the pre-window history + the window's own QKV rows), the
// gate rows, beta, the q/k norms — and writes per-row scratch SC [T][4][DI] (q, k, v,
// expf(gate)); k5d_stept (grid nh) commits the conv history and runs only the state step,
// o, the norm and yb's q8 per row, the state in registers.
__device__ __forceinline__ float k5d_seq(const float* H0, const float* QKV, unsigned c, unsigned j, unsigned DI) {
    // element j of channel c's sequence: the KDA_KC-1 history values, then the window rows
    return j < KDA_KC - 1 ? H0[c * (KDA_KC - 1) + j] : QKV[(size_t)(j - (KDA_KC - 1)) * 3 * DI + c];
}
extern "C" __global__ void __launch_bounds__(KDA_D) k5d_pret(
    const G5W fb, const G5W gb, const G5W bp, const float* __restrict__ X_,
    const float* __restrict__ G1_, const float* __restrict__ G2A_, const float* __restrict__ DT,
    const float* __restrict__ RATE, float LB, const float* __restrict__ QKV_,
    float* __restrict__ H0, const float* __restrict__ W, float* __restrict__ S,
    const float* __restrict__ ON, float EPS, float* __restrict__ BETA_, float* __restrict__ G_,
    float* __restrict__ G2_, float* __restrict__ SC, unsigned HID, unsigned T,
    float* __restrict__ CK, unsigned long long CKSTRIDE) {
    const unsigned TI = blockIdx.y, h = blockIdx.x, x = threadIdx.x, D = KDA_D, DI = KDA_H * KDA_D;
    const unsigned lane = x & 31, warp = x >> 5;
    __shared__ __align__(16) unsigned char wf[KDA_D * K5D_RB + 16], wg[KDA_D * K5D_RB + 16];
    __shared__ __align__(16) unsigned char wb[K5D_HIDMAX / 32 * 34 + 16];
    __shared__ float g1s[KDA_D], g2s[KDA_D], q[KDA_D], k[KDA_D], eg[KDA_D], os[KDA_D];
    __shared__ float bs;
    __shared__ double sq[KDA_D];
    // x for beta, staged with the weights (dynamic: HID * 4 + 16 B, the host sets the
    // attribute) — warp 0's 16 serial L2 round trips for x were ~4 us of the critical path
    extern __shared__ __align__(16) unsigned char xs[];
    int df, dg, db, dx;
    k5d_stage(wf, (const unsigned char*)(fb.base + (unsigned long long)h * D * fb.row_bytes), D * K5D_RB, x, D, &df);
    k5d_stage(wg, (const unsigned char*)(gb.base + (unsigned long long)h * D * gb.row_bytes), D * K5D_RB, x, D, &dg);
    k5d_stage(wb, (const unsigned char*)(bp.base + (unsigned long long)h * bp.row_bytes), HID / 32 * 34, x, D, &db);
    const float* X = X_ + (size_t)TI * HID; const float* G1 = G1_ + TI * D; const float* G2A = G2A_ + TI * D;
    const float* QKV = QKV_ + (size_t)TI * 3 * DI; float* BETA = BETA_ + TI * KDA_H;
    float* G = G_ + (size_t)TI * DI; float* G2 = G2_ + (size_t)TI * DI;
    const bool ck = CK && TI + 1 < T;
    float* CKH = CK + TI * CKSTRIDE;
    k5d_stage(xs, (const unsigned char*)X, HID * 4, x, D, &dx);
    g1s[x] = G1[x]; // f_a / g_a outputs are hd wide, shared by every head
    g2s[x] = G2A[x];
    // the state column this thread owns, read once (independent loads, all in flight)
    // conv (k5d_step verbatim) + the T=1 commit of the thread's 3 channels
    float qv, kv, vv;
    const unsigned int qc = h * D + x, kch = KDA_H * D + h * D + x, vch = 2 * KDA_H * D + h * D + x;
    {
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = k5d_seq(H0, QKV_, qc, TI + i, DI);
            acc = __fmaf_rn(W[qc * KDA_KC + i], e, acc);
        }
        qv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = k5d_seq(H0, QKV_, kch, TI + i, DI);
            acc = __fmaf_rn(W[kch * KDA_KC + i], e, acc);
        }
        kv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = k5d_seq(H0, QKV_, vch, TI + i, DI);
            acc = __fmaf_rn(W[vch * KDA_KC + i], e, acc);
        }
        vv = acc / (1.0f + expf(-acc));
        // the conv history after this row -> its checkpoint (H0 itself is committed by
        // k5d_stept: every row block here still reads the pre-window H0)
        const unsigned int chs[3] = {qc, kch, vch};
        if (ck)
            for (unsigned int c = 0; c < 3; c++)
                for (unsigned int s = 0; s < KDA_KC - 1; s++)
                    CKH[chs[c] * (KDA_KC - 1) + s] = k5d_seq(H0, QKV_, chs[c], TI + 1 + s, DI);
    }
    q[x] = qv; k[x] = kv;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    // gate rows (k5d_gate2: g5_dot's j-ascending acc += deq * x, deq = d * q)
    float gv, g2v;
    {
        const unsigned char* rf = wf + df + x * K5D_RB;
        const unsigned char* rg = wg + dg + x * K5D_RB;
        float af = 0.f, ag = 0.f;
        for (unsigned b = 0; b < D / 32; b++) {
            const float dfw = g5_h2f(g5_u16(rf + 34 * b)), dgw = g5_h2f(g5_u16(rg + 34 * b));
#pragma unroll
            for (unsigned i = 0; i < 32; i++) {
                af = fmaf(dfw * (float)(signed char)rf[34 * b + 2 + i], g1s[32 * b + i], af);
                ag = fmaf(dgw * (float)(signed char)rg[34 * b + 2 + i], g2s[32 * b + i], ag);
            }
        }
        float pre = af + DT[h * D + x];
        if (LB == LB) {
            gv = LB / (1.f + expf(-RATE[h] * pre));
        } else {
            float v = RATE[h] * pre;
            gv = -(v > 20.f ? v : logf(1.f + expf(v)));
        }
        g2v = ag;
        G[h * D + x] = gv;
        G2[h * D + x] = g2v;
        eg[x] = expf(gv);
    }
    // beta (k5_beta: lane chain e0 = lane*8 + 256i, 8 a step, xor reduce, sigmoid)
    if (warp == 0) {
        const unsigned char* row = wb + db;
        float a = 0.f;
        for (int e0 = lane * 8; e0 < (int)HID; e0 += 256) {
            const unsigned char* bb = row + (e0 / 32) * 34;
            const float dw = g5_h2f(g5_u16(bb));
            const float* xp = (const float*)(xs + dx) + e0;
            const float xv[8] = {xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7]};
#pragma unroll
            for (int qq = 0; qq < 8; qq++) a = fmaf(dw * (float)(signed char)bb[2 + (e0 + qq) % 32], xv[qq], a);
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) { float bt = 1.f / (1.f + expf(-a)); BETA[h] = bt; bs = bt; }
    }
    float qs = 0.0f, ks = 0.0f;
    for (unsigned int a = 0; a < D; a++) { qs += q[a] * q[a]; ks += k[a] * k[a]; }
    qs = rsqrtf(qs + 1e-6f); ks = rsqrtf(ks + 1e-6f);
    float scale = rsqrtf((float)D);
    __syncthreads(); // q/k read raw by all, eg / bs written
    float* sc = SC + (size_t)TI * 4 * DI + h * D + x;
    sc[0] = q[x] * (qs * scale); // q[x] *= qs * scale, as in k5d_tail
    sc[DI] = k[x] * ks;
    sc[2 * DI] = vv;
    sc[3 * DI] = eg[x];
}
extern "C" __global__ void __launch_bounds__(KDA_D) k5d_stept(
    const float* __restrict__ QKV_, float* __restrict__ H0, float* __restrict__ S,
    const float* __restrict__ ON, float EPS, const float* __restrict__ BETA_,
    const float* __restrict__ G2_, const float* __restrict__ SC, float* __restrict__ O_,
    float* __restrict__ Y_, unsigned char* __restrict__ XQ, unsigned MAXT, unsigned T,
    float* __restrict__ CK, unsigned long long CKSTRIDE, unsigned long long H0N) {
    const unsigned h = blockIdx.x, x = threadIdx.x, D = KDA_D, DI = KDA_H * KDA_D;
    const unsigned lane = x & 31, warp = x >> 5;
    __shared__ float q[KDA_D], k[KDA_D], eg[KDA_D], os[KDA_D], g1s[1];
    __shared__ double sq[KDA_D];
    // commit the conv history after the window's T rows (this thread's 3 channels)
    {
        const unsigned int chs[3] = {h * D + x, KDA_H * D + h * D + x, 2 * KDA_H * D + h * D + x};
        for (unsigned int c = 0; c < 3; c++) {
            float hv[KDA_KC - 1];
            for (unsigned int s = 0; s < KDA_KC - 1; s++) hv[s] = k5d_seq(H0, QKV_, chs[c], T + s, DI);
            for (unsigned int s = 0; s < KDA_KC - 1; s++) H0[chs[c] * (KDA_KC - 1) + s] = hv[s];
        }
    }
    float sr[KDA_D];
#pragma unroll
    for (unsigned a = 0; a < KDA_D; a++) sr[a] = S[((unsigned long long)h * D + a) * D + x];
    for (unsigned TI = 0; TI < T; TI++) {
    if (TI) __syncthreads();
    const float* sc = SC + (size_t)TI * 4 * DI + h * D;
    q[x] = sc[x]; k[x] = sc[DI + x]; eg[x] = sc[3 * DI + x];
    const float vv = sc[2 * DI + x];
    const float bs = BETA_[TI * KDA_H + h];
    const float g2v = G2_[(size_t)TI * DI + h * D + x];
    float* O = O_ + (size_t)TI * DI; float* Y = Y_ + (size_t)TI * DI;
    const bool ck = CK && TI + 1 < T;
    float* CKS = CK + TI * CKSTRIDE + H0N;
    __syncthreads();
    float p = 0.0f;
#pragma unroll
    for (unsigned int a = 0; a < KDA_D; a++) {
        float s = sr[a] * eg[a];
        sr[a] = s;
        p = __fmaf_rn(k[a], s, p);
    }
    float delta = bs * (vv - p);
    float o = 0.0f;
#pragma unroll
    for (unsigned int a = 0; a < KDA_D; a++) {
        float s = __fmaf_rn(k[a], delta, sr[a]);
        sr[a] = s;
        if (TI + 1 == T) S[((unsigned long long)h * D + a) * D + x] = s;
        else if (ck) CKS[((unsigned long long)h * D + a) * D + x] = s;
        o = __fmaf_rn(q[a], s, o);
    }
    O[h * D + x] = o;
    os[x] = o;
    // (double)o * (double)o is exact (24-bit mantissas), so ss + v*v rounds once either
    // way: the squares are formed in parallel and thread 0 runs only the DADD chain
#ifdef K5D_NORM_EXACT
    sq[x] = (double)o * (double)o;
    __syncthreads();
    // the f64 norm once per head (thread 0) and broadcast: every thread used to run the
    // same 128-long f64 chain, and FP64 is scarce on the B300 (~65 cycles a dependent
    // DADD): same sums in the same order, bit-equal
    if (x == 0) {
        double ss = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < KDA_D; d++) ss += sq[d];
        g1s[0] = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    }
    __syncthreads();
#else
    // default (NOTES 2026-10-28 kda4): the same f64 squares summed as a tree (xor shuffle
    // within each warp, then the KDA_D/32 warp partials in order by thread 0): depth
    // 5 + KDA_D/32 dependent DADDs instead of KDA_D. A summation-order change in f64, so
    // the f32 r almost always rounds the same. TANG_GLM5_KDA_NORM=exact: the serial chain.
    {
        double v = (double)o * (double)o;
        for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o2);
        if (lane == 0) sq[warp] = v;
    }
    __syncthreads();
    if (x == 0) {
        double ss = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < KDA_D / 32; d++) ss += sq[d];
        g1s[0] = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    }
    __syncthreads();
#endif
    const float r = g1s[0];
    const float yv = os[x] * r * ON[x] / (1.f + expf(-g2v));
    Y[h * D + x] = yv;
    // o_proj's k5_q8 over yb (COLS = DI, t = 0): this warp's 32 lanes are one Q8 block
    if (XQ) {
        float amax = fabsf(yv);
        for (int o2 = 16; o2 > 0; o2 >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o2));
        const float d = amax / 127.0f;
        const int qi = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(yv / d), -127.f), 127.f);
        // TI = this row's index in the window's k5_q8 layout (0 for a decode step)
        XQ[(size_t)TI * DI + h * D + x] = (unsigned char)(signed char)qi;
        if (lane == 0) ((float*)(XQ + (size_t)MAXT * DI))[TI * (DI / 32) + (h * D + x) / 32] = d;
    }
    } // rows
}
