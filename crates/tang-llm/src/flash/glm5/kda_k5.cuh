
// y[t*YSTRIDE + r] = dot(W row r of matrix e, x + t*COLS). Call per projection with the
// destination base pointer and stride (ROWS for a plain layout, 3*DI for stacked qkv).
extern "C" __global__ void k5_gemv(const G5W w, unsigned e, const float* x, float* y,
                                   unsigned ystride, unsigned T, unsigned ROWS, unsigned COLS) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = i / ROWS, r = i % ROWS;
    if (t >= T) return;
    y[(unsigned long long)t * ystride + r] = g5_dot(w, e, r, x + (unsigned long long)t * COLS, COLS);
}

// g1 in place: g1[i] = gate_form(g1[i] + dt[chan]) over nh*hd per token.
extern "C" __global__ void k5_gate_finish(float* g1, const float* dt, const float* rate,
                                          unsigned NH, unsigned HD, float LB, unsigned T) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NH * HD) return;
    unsigned chan = i % (NH * HD);
    unsigned h = chan / HD;
    float pre = g1[i] + dt[chan];
    float out;
    if (LB == LB) { // NaN means absent: the softplus form
        out = LB / (1.f + expf(-rate[h] * pre));
    } else {
        float v = rate[h] * pre; // softplus: log(1+e^v), pass-through above 20 (ggml)
        out = -(v > 20.f ? v : logf(1.f + expf(v)));
    }
    g1[i] = out;
}

// beta[t][h] = sigmoid(dot(b_proj row h, x[t])).
extern "C" __global__ void k5_beta(const G5W bp, const float* x, float* beta,
                                   unsigned T, unsigned HID, unsigned NH) {
    // one WARP per (t, h): the naive form gave one thread a serial 4096-element
    // dequant+dot (1.28 ms/launch, 45.8 ms per decode token across 34 layers, ~0 B read
    // because each thread streams its row alone). Warp lanes split the row 128-wide in
    // 8-element steps with the shuffle reduce — the same shape as k5_gemv8's dot.
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned i = blockIdx.x * (blockDim.x >> 5) + warp;
    const unsigned t = i / NH, h = i % NH;
    if (t >= T) return;
    float a = 0.f;
    // g5_deq takes the row as a byte pointer (the G5W base is a u64 device address)
    const unsigned char* row = (const unsigned char*)(bp.base + (unsigned long long)h * bp.row_bytes);
    for (int e0 = lane * 8; e0 < (int)HID; e0 += 256) {
        #pragma unroll
        for (int q = 0; q < 8; q++) a = fmaf(g5_deq(bp.ty, row, e0 + q), x[(size_t)t * HID + e0 + q], a);
    }
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) beta[(unsigned long long)t * NH + h] = 1.f / (1.f + expf(-a));
}

// y[t][c] = rms_norm(o[t][h..])[c] * on[c] * sigmoid(g2[t][c]), h = c / HD.
extern "C" __global__ void k5_gated_norm(const float* o, const float* on, const float* g2,
                                         float* y, unsigned NH, unsigned HD, float EPS,
                                         unsigned T) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NH * HD) return;
    unsigned t = i / (NH * HD), c = i % (NH * HD);
    unsigned h = c / HD, j = c % HD;
    const float* ot = o + ((unsigned long long)t * NH + h) * HD;
    double s = 0.0; // the host's rms_norm_mul: f64 sum of squares
    for (unsigned d = 0; d < HD; d++) { double v = (double)ot[d]; s += v * v; }
    float r = (float)(1.0 / sqrt((s / (double)HD) + (double)EPS));
    y[i] = ot[j] * r * on[j] / (1.f + expf(-g2[i]));
}




// ---------------------------------------------------------------------------
// k5_gemv8: the dequant-reuse GEMV for Q8_0 rows (docs/models/glm5-speed-plan.md step 1)
// — the Flash engine's fe_gemv8 shape (kernels.rs gemv8_body) cut down to the G5W view's
// one-type matrices. SCOPE: the KDA projections, all Q8_0 in the UD-Q4_K_XL GGUF
// (attn_q/k/v/output, ssm_f_a/f_b, ssm_beta, ssm_g_a/g_b). The MoE experts (Q4_K/Q5_K)
// stay on g5_expert_gu for now.
//
// Block 128 = K5_WARPS warps, ONE row per warp (grid.x covers ROWS/K5_WARPS blocks,
// grid.y = T). The row is staged into shared memory with aligned uint4 loads (the whole
// 8.7 KB row in flight vs the naive one-thread serial read); then the 32 lanes walk it
// in 8-element steps: e0 = lane*8 + 256*i, each step dp4a's 8 contiguous row bytes
// (Q8_0: block at e0/32, values at bytes 2 + e0%32 .. +8) against 8 int8 activation
// codes, scaled by that block's f16 d and the activation block's d. Shared use: 4 warps
// x 8.7 KB = 34.8 KB (under the 48 KB default; two rows a warp would need a dynamic-
// smem opt-in — later, if the measure says so).
//
// Activations: xq = fe_q8's layout — plain int8 codes [T][COLS], f32 per-32 scales
// [T][COLS/32], d = amax/127. NOT the f32 serial rounding: parity is the engine test's
// 1e-3 bar + the real-model ids (fe_gemv8's bar). F32 rows never come through here
// (the small per-layer f32 tensors are not GEMV weights).
#ifndef K5_HOST_CHECK
#define K5_WARPS 4

// int -> float via a magic add (kernels.rs i2f): exact for |q| < 2^22.
__device__ __forceinline__ float k5_i2f(int q) { return __int_as_float(0x4B400000 + q) - 12582912.0f; }

// x [T][COLS] f32 -> xq: plain int8 codes [T][COLS] then per-32 f32 scales [T][COLS/32]
// (fe_q8's rule: d = amax/127 over each 32, round half away). Grid (COLS/32, T), block 32.
// MAXT = the scratch's row cap (the codes array is sized MAXT rows; scales follow it).
extern "C" __global__ void k5_q8(const float* x, unsigned char* xq, unsigned COLS, unsigned MAXT) {
    const unsigned t = blockIdx.y, c = blockIdx.x, lane = threadIdx.x;
    const float v = x[(size_t)t * COLS + c * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    xq[(size_t)t * COLS + c * 32 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) ((float*)(xq + (size_t)MAXT * COLS))[t * (COLS / 32) + c] = d;
}

extern "C" __global__ void __launch_bounds__(128) k5_gemv8(
    const G5W w, unsigned e, const unsigned char* xq, const float* xd,
    float* y, unsigned ystride, unsigned T, unsigned ROWS, unsigned COLS) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ uint4 dsm[]; // per warp: one row, ceil(row_bytes/16)+1 uint4s
    const unsigned rb = w.row_bytes;
    const int row16 = (int)((rb + 15) >> 4);
    const unsigned long long mbase = w.base + (unsigned long long)e * w.estride;
    const int t = (int)blockIdx.y;
    if (t >= (int)T) return;
    const unsigned char* xrow = xq + (size_t)t * COLS;

    // Direct path (NOTES 2026-10-28 kda4): no shared staging. Each lane reads its own
    // 8 codes per step straight from global as aligned u32 words (+ the block's d word),
    // U steps' words issued before any dp4a. Same e0 = lane*8 + 256i walk, same
    // dp4a(w0, dp4a(w1, 0)) and fmaf chain and xor tree: bit-equal to the staged path.
    // Needs 4-byte-aligned rows (row_bytes % 4 == 0, base % 4 == 0); the dynamic smem
    // the launchers still pass goes unused. TANG-side switch: none (always on when aligned).
    if (((rb | (unsigned)mbase) & 3u) == 0u && (COLS & 255u) == 0u) {
        const unsigned row = blockIdx.x * K5_WARPS + warp;
        if (row >= ROWS) return;
        const unsigned char* src = (const unsigned char*)(mbase + (unsigned long long)row * rb);
        const float* xdt = xd + (size_t)t * (COLS / 32);
        const int k = lane & 3;
        const int nst = (int)(COLS >> 8);
        float acc = 0.f;
        constexpr int U = 8;
        for (int i0 = 0; i0 < nst; i0 += U) {
            unsigned wdw[U], u0[U], u1[U], u2[U];
#pragma unroll
            for (int j = 0; j < U; j++) {
                if (i0 + j < nst) {
                    const unsigned b = (unsigned)(lane >> 2) + 8u * (unsigned)(i0 + j);
                    const unsigned od = 34u * b, oq = od + 2u + 8u * (unsigned)k;
                    const unsigned* pq = (const unsigned*)(src + (oq & ~3u));
                    wdw[j] = __ldcs((const unsigned*)(src + (od & ~3u)));
                    u0[j] = __ldcs(pq);
                    u1[j] = __ldcs(pq + 1);
                    u2[j] = (oq & 3u) ? __ldcs(pq + 2) : 0u;
                }
            }
#pragma unroll
            for (int j = 0; j < U; j++) {
                if (i0 + j >= nst) break;
                const int e0 = lane * 8 + 256 * (i0 + j);
                const unsigned b = (unsigned)(e0 >> 5);
                const unsigned sh = ((34u * b + 2u) & 3u) * 8u;
                const float dw = g5_h2f((b & 1u) ? (wdw[j] >> 16) : (wdw[j] & 0xffffu));
                const int w0 = (int)__funnelshift_r(u0[j], u1[j], sh);
                const int w1 = (int)__funnelshift_r(u1[j], u2[j], sh);
                const uint2 xv = *(const uint2*)(xrow + e0);
                const float dx = xdt[e0 / 32];
                const int s = __dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0));
                acc = fmaf(dx * dw, k5_i2f(s), acc);
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[(unsigned long long)t * ystride + row] = acc;
        return;
    }

    {
        const unsigned row = blockIdx.x * K5_WARPS + warp;
        if (row >= ROWS) return;
        const unsigned char* src = (const unsigned char*)(mbase + (unsigned long long)row * rb);
        // +1 granule per warp: an unaligned row (delta > 0) spans row16+1 granules; spacing
        // warps by row16 let each warp's staging overwrite the next warp's first granule
        uint4* sm = dsm + (size_t)warp * (row16 + 1);
        const unsigned long long a = (unsigned long long)src & ~15ull;
        const int delta = (int)((unsigned long long)src - a);
        const int n16 = (delta + (int)rb + 15) >> 4;
        for (int i = lane; i < n16; i += 32) sm[i] = __ldg((const uint4*)a + i);
        // lanes read granules other lanes staged: the warp must sync first (missing in the
        // first 3090 run — the shadow check failed 68/68 windows at qkv.q)
        __syncwarp();
        const unsigned char* rowp = (const unsigned char*)sm + delta;
        float acc = 0.f;
        for (int e0 = lane * 8; e0 < (int)COLS; e0 += 256) {
            const unsigned char* b = rowp + (e0 / 32) * 34;
            const float dw = g5_h2f(g5_u16(b));
            // Q8_0 blocks are 34 B: b + 2 + e0%32 is NOT 8-byte aligned — assemble the two
            // 4-byte words from bytes (a direct u64 load faults: MISALIGNED/ILLEGAL_ADDRESS)
            const unsigned char* qb = b + 2 + (e0 % 32);
            const int w0 = (int)((unsigned)qb[0] | ((unsigned)qb[1] << 8) | ((unsigned)qb[2] << 16) | ((unsigned)qb[3] << 24));
            const int w1 = (int)((unsigned)qb[4] | ((unsigned)qb[5] << 8) | ((unsigned)qb[6] << 16) | ((unsigned)qb[7] << 24));
            const uint2 xv = *(const uint2*)(xrow + e0);
            const float dx = xd[(size_t)t * (COLS / 32) + e0 / 32];
            const int s = __dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0));
            acc = fmaf(dx * dw, k5_i2f(s), acc);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[(unsigned long long)t * ystride + row] = acc;
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// k5_qgemv: the T=1 Q8_0 GEMV pushed toward HBM bandwidth (TANG_GLM5_KDA_G5F=2). Several
// matrices that share one input x run in ONE launch (q | k | v | f_a | g_a: 107 MB, so the
// ramp-up/tail of a 35 MB launch is paid once, not three times). Segment table in device
// memory (built at load: weight bases and output pointers never move, so it is
// graph-capture safe): {base, y, row_bytes, rows}. One row a warp, K5Q warps a block
// (blockDim.x / 32, the launcher picks it so the staged rows fit 48 KB).
// Each lane issues ALL its 16-byte row loads first (ld.global.nc, no L1 allocate: the
// weights are read once), then parks them in shared memory, then walks the row's Q8_0
// blocks (block b = lane + 32 i: f16 d + 32 int8 codes, 8 dp4a against x's k5_q8 codes,
// one fmaf per 32 elements). Requires base % 16 == 0 and row_bytes % 16 == 0 (Q8_0 rows
// of 4096 / 8192 are 4352 / 8704 B); the launcher checks and falls back otherwise.
struct K5Seg { unsigned long long base; unsigned long long y; unsigned row_bytes; unsigned rows; };
__device__ __forceinline__ uint4 k5_ldnc(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
// MAXV = 16-byte granules a lane holds (rows up to 32 * MAXV * 16 B): k5_qgemv9 for
// 4352 B rows (8.5 granules a lane), k5_qgemv17 for 8704 B (o_proj).
// No shared-memory staging (B300 microbench, NOTES 2026-10-28 kda3): staging capped the
// rows in flight per SM at 228 KB / row_bytes and serialised load -> dot inside each warp
// (3.7 TB/s on q|k|v|f_a|g_a, 2.4 on o_proj). Each lane now reads its own Q8_0 blocks
// b = lane + 32 i straight from global as 9 aligned u32 words (a 34 B block starts at
// 34b & ~3; odd b: the block is words 0..8 shifted by 16 bits, even b: words 0..7 plus
// the low half of word 8), every word of the row issued up front (ld.global.cs). Same
// per-block integer dot, same fmaf chain over i and xor tree: bit-equal to the staged
// kernel. Word 8 of an even b never runs past the row: rb % 16 == 0 (host-checked).
// R rows a warp (TANG_GLM5_KDA_QR2=1: R = 2, flagged by bit 31 of nseg so the launch
// signature and the kernel names stay): rows 2p, 2p+1 of a segment share each lane's
// activation loads (x codes + xd), every word of both rows issued before any dp4a. Each
// row's chain is the R = 1 chain verbatim (same blocks, same order): bit-equal.
template <int K5Q_MAXNB, int R>
__device__ __forceinline__ void k5_qgemv_body_r(
    const K5Seg* __restrict__ segs, unsigned nseg, const unsigned char* __restrict__ xq,
    const float* __restrict__ xd, unsigned COLS) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned g = blockIdx.x * nw + warp;
    unsigned s = 0;
    for (; s < nseg; s++) {
        const unsigned rr = (segs[s].rows + R - 1) / R;
        if (g < rr) break;
        g -= rr;
    }
    if (s >= nseg) return;
    g *= R;
    const int nb = (int)(COLS >> 5);
    unsigned wd[R][K5Q_MAXNB][9];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const bool live = g + r < segs[s].rows;
        const unsigned char* row = (const unsigned char*)(segs[s].base + (unsigned long long)(g + r) * segs[s].row_bytes);
#pragma unroll
        for (int i = 0; i < K5Q_MAXNB; i++) {
            const int b = lane + 32 * i;
            if (live && b < nb) {
                const unsigned* p = (const unsigned*)(row + ((34u * (unsigned)b) & ~3u));
#pragma unroll
                for (int j = 0; j < 9; j++) wd[r][i][j] = __ldcs(p + j);
            }
        }
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
#pragma unroll
    for (int i = 0; i < K5Q_MAXNB; i++) {
        const int b = lane + 32 * i;
        if (b >= nb) break;
        const bool odd = b & 1;
        const int4* xv = (const int4*)(xq + 32 * b);
        const int4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
        const float xdb = xd[b];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float dw = g5_h2f(odd ? (wd[r][i][0] >> 16) : (wd[r][i][0] & 0xffffu));
            int w[8];
#pragma unroll
            for (int j = 0; j < 8; j++) w[j] = odd ? (int)wd[r][i][j + 1] : (int)__funnelshift_r(wd[r][i][j], wd[r][i][j + 1], 16);
            int si = __dp4a(w[0], x0.x, 0);
            si = __dp4a(w[1], x0.y, si); si = __dp4a(w[2], x0.z, si); si = __dp4a(w[3], x0.w, si);
            si = __dp4a(w[4], x1.x, si); si = __dp4a(w[5], x1.y, si); si = __dp4a(w[6], x1.z, si);
            si = __dp4a(w[7], x1.w, si);
            acc[r] = fmaf(xdb * dw, k5_i2f(si), acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float a = acc[r];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && g + r < segs[s].rows) ((float*)segs[s].y)[g + r] = a;
    }
}
template <int K5Q_MAXNB>
__device__ __forceinline__ void k5_qgemv_body(
    const K5Seg* __restrict__ segs, unsigned nseg, const unsigned char* __restrict__ xq,
    const float* __restrict__ xd, unsigned COLS) {
    if (nseg & 0x80000000u) k5_qgemv_body_r<K5Q_MAXNB, 2>(segs, nseg & 0x7fffffffu, xq, xd, COLS);
    else k5_qgemv_body_r<K5Q_MAXNB, 1>(segs, nseg, xq, xd, COLS);
}
extern "C" __global__ void __launch_bounds__(256) k5_qgemv9(const K5Seg* segs, unsigned nseg,
    const unsigned char* xq, const float* xd, unsigned COLS) { k5_qgemv_body<5>(segs, nseg, xq, xd, COLS); }
extern "C" __global__ void __launch_bounds__(256) k5_qgemv17(const K5Seg* segs, unsigned nseg,
    const unsigned char* xq, const float* xd, unsigned COLS) { k5_qgemv_body<9>(segs, nseg, xq, xd, COLS); }
// k5_qgemvt: k5_qgemv for the T (2..4) rows of ONE window (the MTP verify window,
// NOTES 2026-10-28 kda4 round 3): each lane loads its row's weight words once and keeps T
// accumulators, token u reading xq + u*COLS codes and xd + u*(COLS/32) scales (k5_q8's
// MAXT layout with t = u). Per token the chain is the T = 1 chain verbatim (same blocks,
// same dp4a order, same fmaf over i, same xor tree): bit-equal to T separate t=1 launches.
// Output row u of segment s at y + u*stride: stride = SF for s < NF (q|k|v share qkvb at
// 3*di), else the segment's rows.
template <int K5Q_MAXNB, int TT>
__device__ __forceinline__ void k5_qgemvt_body(
    const K5Seg* __restrict__ segs, unsigned nseg, const unsigned char* __restrict__ xq,
    const float* __restrict__ xd, unsigned COLS, unsigned NF, unsigned SF) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned g = blockIdx.x * nw + warp;
    unsigned s = 0;
    for (; s < nseg; s++) {
        if (g < segs[s].rows) break;
        g -= segs[s].rows;
    }
    if (s >= nseg) return;
    const int nb = (int)(COLS >> 5);
    const unsigned char* row = (const unsigned char*)(segs[s].base + (unsigned long long)g * segs[s].row_bytes);
    unsigned wd[K5Q_MAXNB][9];
#pragma unroll
    for (int i = 0; i < K5Q_MAXNB; i++) {
        const int b = lane + 32 * i;
        if (b < nb) {
            const unsigned* p = (const unsigned*)(row + ((34u * (unsigned)b) & ~3u));
#pragma unroll
            for (int j = 0; j < 9; j++) wd[i][j] = __ldcs(p + j);
        }
    }
    float acc[TT];
#pragma unroll
    for (int u = 0; u < TT; u++) acc[u] = 0.f;
    const unsigned nbx = COLS >> 5;
#pragma unroll
    for (int i = 0; i < K5Q_MAXNB; i++) {
        const int b = lane + 32 * i;
        if (b >= nb) break;
        const bool odd = b & 1;
        const float dw = g5_h2f(odd ? (wd[i][0] >> 16) : (wd[i][0] & 0xffffu));
        int w[8];
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = odd ? (int)wd[i][j + 1] : (int)__funnelshift_r(wd[i][j], wd[i][j + 1], 16);
#pragma unroll
        for (int u = 0; u < TT; u++) {
            const int4* xv = (const int4*)(xq + (size_t)u * COLS + 32 * b);
            const int4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
            int si = __dp4a(w[0], x0.x, 0);
            si = __dp4a(w[1], x0.y, si); si = __dp4a(w[2], x0.z, si); si = __dp4a(w[3], x0.w, si);
            si = __dp4a(w[4], x1.x, si); si = __dp4a(w[5], x1.y, si); si = __dp4a(w[6], x1.z, si);
            si = __dp4a(w[7], x1.w, si);
            acc[u] = fmaf(xd[u * nbx + b] * dw, k5_i2f(si), acc[u]);
        }
    }
    const unsigned long long st = s < NF ? SF : segs[s].rows;
#pragma unroll
    for (int u = 0; u < TT; u++) {
        float a = acc[u];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) ((float*)segs[s].y)[u * st + g] = a;
    }
}
#define K5QT(NAME, NB) \
extern "C" __global__ void __launch_bounds__(128) NAME(const K5Seg* segs, unsigned nseg, \
    const unsigned char* xq, const float* xd, unsigned COLS, unsigned NF, unsigned SF, unsigned T) { \
    switch (T) { \
    case 2: k5_qgemvt_body<NB, 2>(segs, nseg, xq, xd, COLS, NF, SF); break; \
    case 3: k5_qgemvt_body<NB, 3>(segs, nseg, xq, xd, COLS, NF, SF); break; \
    default: k5_qgemvt_body<NB, 4>(segs, nseg, xq, xd, COLS, NF, SF); break; } }
K5QT(k5_qgemvt9, 5)
K5QT(k5_qgemvt17, 9)
// k5_qgemv_q4k: k5_qgemv for Q4_K rows (the requantized KDA projections, NOTES 2026-10-27
// kda-q4). Same segment table, staging and launch shape; only the dot differs. A Q4_K
// super-block (144 B, 256 values) = f16 d, f16 dmin, 12 B of packed 6-bit (scale, min) for
// its 8 sub-blocks of 32, then 128 B of nibbles: the 32 bytes of pair p hold sub-block 2p in
// the low nibbles and 2p+1 in the high ones. Lane = sub-block sb = lane + 32 i:
//   w·x over sb = xd[sb] · (d·sc · Σ q·xq  −  dmin·m · Σ xq)
// (ggml's dequant w = d·sc·q − dmin·m, x ≈ xd · xq). Rows of 4096 (2304 B) take
// k5_qgemv_q4k5, rows of 8192 (4608 B, o_proj) k5_qgemv_q4k9.
__device__ __forceinline__ void k5_q4k_sm(const unsigned char* q, int j, int& sc, int& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
template <int K5Q_MAXV>
__device__ __forceinline__ void k5_qgemv_q4k_body(
    const K5Seg* __restrict__ segs, unsigned nseg, const unsigned char* __restrict__ xq,
    const float* __restrict__ xd, unsigned COLS) {
    extern __shared__ uint4 dsm[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned g = blockIdx.x * nw + warp;
    unsigned s = 0;
    for (; s < nseg; s++) {
        if (g < segs[s].rows) break;
        g -= segs[s].rows;
    }
    if (s >= nseg) return;
    const unsigned rb = segs[s].row_bytes;
    const int n16 = (int)(rb >> 4);
    const uint4* src = (const uint4*)(segs[s].base + (unsigned long long)g * rb);
    uint4* sm = dsm + (size_t)warp * n16;
    uint4 v[K5Q_MAXV];
#pragma unroll
    for (int i = 0; i < K5Q_MAXV; i++)
        if (lane + 32 * i < n16) v[i] = k5_ldnc(src + lane + 32 * i);
#pragma unroll
    for (int i = 0; i < K5Q_MAXV; i++)
        if (lane + 32 * i < n16) sm[lane + 32 * i] = v[i];
    __syncwarp();
    const unsigned char* row = (const unsigned char*)sm;
    float acc = 0.f;
    const int nsb = (int)(COLS >> 5);
    for (int sb = lane; sb < nsb; sb += 32) {
        const unsigned char* blk = row + 144 * (sb >> 3);
        const int j = sb & 7;
        const unsigned short* h = (const unsigned short*)blk;
        const float d = g5_h2f(h[0]), dmin = g5_h2f(h[1]);
        int sc, m;
        k5_q4k_sm(blk + 4, j, sc, m);
        const unsigned* qs = (const unsigned*)(blk + 16 + 32 * (j >> 1)); // 16 B aligned
        const int sh = 4 * (j & 1);
        const int4* xv = (const int4*)(xq + 32 * sb);
        const int4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
        const int xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        int si = 0, sx = 0;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            si = __dp4a((int)((qs[k] >> sh) & 0x0F0F0F0Fu), xs[k], si);
            sx = __dp4a(0x01010101, xs[k], sx);
        }
        acc = fmaf(xd[sb], d * (float)sc * k5_i2f(si) - dmin * (float)m * k5_i2f(sx), acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) ((float*)segs[s].y)[g] = acc;
}
extern "C" __global__ void __launch_bounds__(256) k5_qgemv_q4k5(const K5Seg* segs, unsigned nseg,
    const unsigned char* xq, const float* xd, unsigned COLS) { k5_qgemv_q4k_body<5>(segs, nseg, xq, xd, COLS); }
extern "C" __global__ void __launch_bounds__(256) k5_qgemv_q4k9(const K5Seg* segs, unsigned nseg,
    const unsigned char* xq, const float* xd, unsigned COLS) { k5_qgemv_q4k_body<9>(segs, nseg, xq, xd, COLS); }
#endif // K5_HOST_CHECK
