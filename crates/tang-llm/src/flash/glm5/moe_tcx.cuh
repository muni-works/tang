// ---- GLM-5 GGUF-native int8 MMA experts (TCX, ported from coder_kernels.rs cx_moe_*_q) ----
//
// Compiled after moe_cuda.cuh (G5W, g5_h2f, g5f_dot_any, g5f_warp_sum, g5f_row). Opt-in
// (`TANG_GLM5_TCX=1`, default off until GPU-verified); see moe_gpu.rs `ffn_tcx`.
//
// Pairs are (token t, slot j), p = t * S + j (S = top_k + shared), the same row index g5f_gu's h
// and g5f_down_split's yp use. g5x_plan groups the pairs by expert (shared expert = key ne,
// sentinel ids >= ne = key ne + 1) in chunks of at most 8 pairs (one MMA n-tile):
//   plan[0] = groups, plan[1 + g] = group g's first index into the list, plan[1 + ng] = n,
//   plan[G5X_LIST + i] = the pairs in (key, pair) order.
// The expert kernels launch a fixed grid of groups (blocks past plan[0] return), so a captured
// graph holds for any routing.
//
// Expert math (coder's gg_rows): the m16n8k32 A register c of lane (g, tig) at step s is row
// g + 8 (c & 1)'s qs word at 32 (s / 2) + 8 tig + 4 (c >> 1) (one uint2 per lane and sub-block
// pair: k-permuted A), and the B fragment is the token's codes at 8 tig (one uint2: the same
// permutation), so dot products and per-32 scales are unchanged. Per step: the codes MMA, the
// mins MMA (A' = m_j replicated: gives m_j sum(x)), the per-32 activation scale read from global,
// float epilogue acc += dx (d sc dot - dmin m sum). A2: GLM's two-level activations
// (x = d (q + q2 / 254), g5f_q8) as two more MMAs on q2 — the same 1e-5 activation band as the
// g5f path. Integer dots are exact, so only the float epilogue order differs from g5f (summation
// over 32-blocks in order, per row): tolerance-gated, not bitwise. Rows of other types (Q6_K,
// Q8_0, repacked) take g5f_dot_any per (row, pair).
#ifndef G5X_SGU
#define G5X_SGU 2
#endif
#ifndef G5X_SDN
#define G5X_SDN 1
#endif
#define G5X_NP 1024
#define G5X_G 8
#define G5X_LIST (2 + G5X_NP)

__device__ __forceinline__ void g5x_mma(int* d, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
#else
    d[0] = d[1] = d[2] = d[3] = 0;
#endif
}

// Group pairs 0 .. n by key. One block of 1024 threads, n <= G5X_NP. O(n^2) per block (n = 9 at
// T = 1, 36 at a K = 3 verify window).
extern "C" __global__ void __launch_bounds__(1024) g5x_plan(const unsigned* ids, int k, int ne, int S, int n, unsigned* plan) {
    __shared__ int key[G5X_NP];
    __shared__ int rank_s[G5X_NP];
    __shared__ unsigned char start[G5X_NP];
    const int i = threadIdx.x;
    if (i < n) {
        const int t = i / S, j = i % S;
        int kk = ne;
        if (j < k) {
            const unsigned e = ids[(size_t)t * k + j];
            kk = e >= (unsigned)ne ? ne + 1 : (int)e;
        }
        key[i] = kk;
    }
    __syncthreads();
    if (i < n) {
        const int me = key[i];
        int lt = 0, pos = 0;
        for (int q = 0; q < n; q++) {
            const int kq = key[q];
            lt += kq < me;
            pos += (kq == me) & (q < i);
        }
        rank_s[i] = lt + pos;
        start[i] = (pos % G5X_G) == 0;
        plan[G5X_LIST + lt + pos] = (unsigned)i;
    }
    __syncthreads();
    if (i < n && start[i]) {
        int gid = 0;
        for (int q = 0; q < n; q++) gid += start[q] && rank_s[q] < rank_s[i];
        plan[1 + gid] = (unsigned)rank_s[i];
    }
    if (i == 0) {
        int ng = 0;
        for (int q = 0; q < n; q++) ng += start[q];
        plan[0] = (unsigned)ng;
        plan[1 + ng] = (unsigned)n;
    }
}

__device__ __forceinline__ float g5x_i2f(int v) { return (float)v; }

__device__ __forceinline__ void g5x_sc_unpack(const unsigned* sc, unsigned& s03, unsigned& s47, unsigned& m03, unsigned& m47) {
    s03 = sc[0] & 0x3F3F3F3Fu;
    m03 = sc[1] & 0x3F3F3F3Fu;
    s47 = (sc[2] & 0x0F0F0F0Fu) | (((sc[0] >> 6) & 0x03030303u) << 4);
    m47 = ((sc[2] >> 4) & 0x0F0F0F0Fu) | (((sc[1] >> 6) & 0x03030303u) << 4);
}

struct G5xR { float d, dm; unsigned sc[3]; uint2 q[4]; uint2 h; };
template <int TY>
__device__ __forceinline__ void g5x_load(const unsigned char* b, int tig, G5xR& o) {
    constexpr int QS = TY == 12 ? 16 : 48;
    const unsigned dw = __ldg((const unsigned*)b);
    o.d = g5_h2f(dw & 0xFFFFu);
    o.dm = g5_h2f(dw >> 16);
    o.sc[0] = __ldg((const unsigned*)(b + 4));
    o.sc[1] = __ldg((const unsigned*)(b + 8));
    o.sc[2] = __ldg((const unsigned*)(b + 12));
    #pragma unroll
    for (int j = 0; j < 4; j++) o.q[j] = __ldg((const uint2*)(b + QS + j * 32 + tig * 8));
    if (TY == 13) o.h = __ldg((const uint2*)(b + 16 + tig * 8));
    else o.h = make_uint2(0, 0);
}

// Rows wa (g) and wb (g + 8) of a 16-row tile against up to 8 tokens: xg / xg2 = this lane's
// B token's codes (+ 8 tig), da / db = the per-32 scales of tokens 2 tig and 2 tig + 1.
template <int TY, bool A2>
__device__ __forceinline__ void g5x_rows(const unsigned char* wa, const unsigned char* wb, int m, const unsigned char* xg,
                                         const unsigned char* xg2, const float* da, const float* db, float acc[4],
                                         int sb0, int sb1) {
    constexpr int BB = TY == 12 ? 144 : 176;
    const int lane = threadIdx.x & 31, tig = lane & 3;
    const int na = 2 * tig, nb = 2 * tig + 1;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    if (sb0 >= sb1) return;
    G5xR ra, rb, pa, pb;
    g5x_load<TY>(wa + sb0 * BB, tig, ra);
    g5x_load<TY>(wb + sb0 * BB, tig, rb);
    for (int sb = sb0; sb < sb1; sb++) {
        if (sb + 1 < sb1) { g5x_load<TY>(wa + (sb + 1) * BB, tig, pa); g5x_load<TY>(wb + (sb + 1) * BB, tig, pb); }
        unsigned sa[2], ma[2], sbw[2], mbw[2];
        g5x_sc_unpack(ra.sc, sa[0], sa[1], ma[0], ma[1]);
        g5x_sc_unpack(rb.sc, sbw[0], sbw[1], mbw[0], mbw[1]);
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            const int qp = s >> 1;
            unsigned a[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const G5xR& r = (c & 1) ? rb : ra;
                const unsigned w = (c >> 1) ? r.q[qp].y : r.q[qp].x;
                unsigned v = (s & 1) ? (w >> 4) & 0x0F0F0F0Fu : w & 0x0F0F0F0Fu;
                if (TY == 13) v |= ((((c >> 1) ? r.h.y : r.h.x) >> s) & 0x01010101u) << 4;
                a[c] = v;
            }
            const uint2 bb = __ldg((const uint2*)(xg + sb * 256 + s * 32));
            int d[4], mm[4];
            g5x_mma(d, a, bb.x, bb.y);
            const unsigned sel = (unsigned)(s & 3) * 0x1111u;
            const unsigned mw0 = ma[s >> 2], mw1 = mbw[s >> 2];
            const unsigned am[4] = {__byte_perm(mw0, 0, sel), __byte_perm(mw1, 0, sel), __byte_perm(mw0, 0, sel),
                                    __byte_perm(mw1, 0, sel)};
            g5x_mma(mm, am, bb.x, bb.y);
            float fd[4], fm[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) { fd[c] = g5x_i2f(d[c]); fm[c] = g5x_i2f(mm[c]); }
            if (A2) {
                const uint2 b2 = __ldg((const uint2*)(xg2 + sb * 256 + s * 32));
                int d2[4], m2[4];
                g5x_mma(d2, a, b2.x, b2.y);
                g5x_mma(m2, am, b2.x, b2.y);
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    fd[c] = fmaf(g5x_i2f(d2[c]), 1.0f / 254.0f, fd[c]);
                    fm[c] = fmaf(g5x_i2f(m2[c]), 1.0f / 254.0f, fm[c]);
                }
            }
            const int s0 = (int)__byte_perm(sa[s >> 2], 0, 0x4440u | (s & 3));
            const int s1 = (int)__byte_perm(sbw[s >> 2], 0, 0x4440u | (s & 3));
            const float fs0 = ra.d * g5x_i2f(s0), fs1 = rb.d * g5x_i2f(s1);
            const int blk = sb * 8 + s;
            if (na < m) {
                const float dx = __ldg(da + blk);
                acc[0] = fmaf(dx, fmaf(fs0, fd[0], -ra.dm * fm[0]), acc[0]);
                acc[2] = fmaf(dx, fmaf(fs1, fd[2], -rb.dm * fm[2]), acc[2]);
            }
            if (nb < m) {
                const float dx = __ldg(db + blk);
                acc[1] = fmaf(dx, fmaf(fs0, fd[1], -ra.dm * fm[1]), acc[1]);
                acc[3] = fmaf(dx, fmaf(fs1, fd[3], -rb.dm * fm[3]), acc[3]);
            }
        }
        if (sb + 1 < sb1) { ra = pa; rb = pb; }
    }
}

// One group's 16-row tiles. DOWN = false: out = gu [n][2 hs], gate rows at [0, F), up rows at
// [hs, hs + F) (F = ff or ffs), raw dots (g5x_silu applies the limits). DOWN = true: out = yp
// [n][hid] = w_j * dot (shared slot weight 1; sentinel pairs 0). Grid (ceil(rows · SPL / 128),
// groups), block 256: 8 warps, SPL warps a tile splitting K by superblocks, then summed in order.
template <bool DOWN, bool A2>
__device__ __forceinline__ void g5x_body(G5W g, G5W u, G5W sg, G5W su, const unsigned* ids, const float* wts, int k,
                                         int ne, int S, const float* xf, const unsigned char* xq,
                                         const unsigned char* xq2, const float* xd, int xs, float* out, int hs,
                                         int ff, int ffs, int hid, const unsigned* plan) {
    constexpr int SPL = DOWN ? G5X_SDN : G5X_SGU;
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = plan[1 + gi], b = plan[2 + gi], m = b - a;
    const int p0 = plan[G5X_LIST + a];
    const int j0 = p0 % S;
    const bool shared = j0 >= k;
    const unsigned e = shared ? 0u : ids[(size_t)(p0 / S) * k + j0];
    const bool sentinel = !shared && e >= (unsigned)ne;
    const int F = shared ? ffs : ff;
    const int KA = DOWN ? F : hid;
    const int NR = DOWN ? hid : 2 * F;
    const int ks = warp % SPL;
    const int r0 = (blockIdx.x * (8 / SPL) + warp / SPL) * 16;
    const bool live = r0 < NR;
    const G5W mw = DOWN ? (shared ? sg : g) : (shared ? (r0 < F ? sg : su) : (r0 < F ? g : u));
    const int ro = DOWN ? r0 : (r0 < F ? r0 : r0 - F);
    const int oc = DOWN ? r0 : (r0 < F ? r0 : hs + r0 - F);
    const size_t ostride = DOWN ? (size_t)hid : (size_t)2 * hs;
    __shared__ int pair_s[G5X_G];
    __shared__ float w_s[G5X_G];
    __shared__ float4 red_s[8][32];
    if (threadIdx.x < G5X_G) {
        const int q = threadIdx.x;
        const int p = q < m ? (int)plan[G5X_LIST + a + q] : p0;
        pair_s[q] = p;
        const int t = p / S, j = p % S;
        w_s[q] = (!DOWN || j >= k) ? 1.f : wts[(size_t)t * k + j];
    }
    __syncthreads();
    // a pair's activation row: the token for gate|up, the pair itself for down
    auto arow = [&](int p) { return DOWN ? p : p / S; };
    if (sentinel) {
        if (DOWN && live && ks == 0)
            for (int q = 0; q < m; q++)
                if (lane < 16) out[(size_t)pair_s[q] * ostride + oc + lane] = 0.f;
        return;
    }
    if (live && mw.ty != 12 && mw.ty != 13) {
        const unsigned char* wbase = (const unsigned char*)mw.base + (size_t)e * mw.estride;
        for (int r = ks; r < 16; r += SPL) {
            const unsigned char* w = wbase + (size_t)(ro + r) * mw.row_bytes;
            for (int q = 0; q < m; q++) {
                const int x = arow(pair_s[q]);
                const float v = g5f_warp_sum(g5f_dot_any(mw.ty, w, xq + (size_t)x * xs, xq2 ? xq2 + (size_t)x * xs : 0,
                                                         xd + (size_t)x * (xs / 32), xf + (size_t)x * xs, KA));
                if (lane == 0) out[(size_t)pair_s[q] * ostride + oc + r] = w_s[q] * v;
            }
        }
        return;
    }
    const int gq = lane >> 2, tig = lane & 3;
    const int xb = arow(pair_s[gq < m ? gq : 0]);
    const unsigned char* xg = xq + (size_t)xb * xs + tig * 8;
    const unsigned char* xg2 = A2 ? xq2 + (size_t)xb * xs + tig * 8 : 0;
    const float* da = xd + (size_t)arow(pair_s[2 * tig < m ? 2 * tig : 0]) * (xs / 32);
    const float* db = xd + (size_t)arow(pair_s[2 * tig + 1 < m ? 2 * tig + 1 : 0]) * (xs / 32);
    const unsigned char* wa = (const unsigned char*)mw.base + (size_t)e * mw.estride + (size_t)(ro + gq) * mw.row_bytes;
    const unsigned char* wb = wa + (size_t)8 * mw.row_bytes;
    const int nsb = KA / 256;
    const int sb0 = ks * nsb / SPL, sb1 = (ks + 1) * nsb / SPL;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (live) {
        if (mw.ty == 12) g5x_rows<12, A2>(wa, wb, m, xg, xg2, da, db, acc, sb0, sb1);
        else g5x_rows<13, A2>(wa, wb, m, xg, xg2, da, db, acc, sb0, sb1);
    }
    if (SPL > 1) {
        red_s[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __syncthreads();
        if (ks != 0) return;
        for (int j = 1; j < SPL; j++) {
            const float4 v = red_s[warp + j][lane];
            acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
        }
    }
    if (!live) return;
    if (2 * tig < m) {
        const size_t o = (size_t)pair_s[2 * tig] * ostride + oc + gq;
        out[o] = w_s[2 * tig] * acc[0];
        out[o + 8] = w_s[2 * tig] * acc[2];
    }
    if (2 * tig + 1 < m) {
        const size_t o = (size_t)pair_s[2 * tig + 1] * ostride + oc + gq;
        out[o] = w_s[2 * tig + 1] * acc[1];
        out[o + 8] = w_s[2 * tig + 1] * acc[3];
    }
}

// gate|up: activations x / xq / xq2 / xd per token at stride hid.
extern "C" __global__ void __launch_bounds__(256) g5x_gu_q(G5W g, G5W u, G5W sg, G5W su, const unsigned* ids, int k,
                                                          int ne, int S, const float* x, const unsigned char* xq,
                                                          const float* xd, float* gu, int hs, int ff, int ffs, int hid,
                                                          const unsigned* plan) {
    g5x_body<false, false>(g, u, sg, su, ids, 0, k, ne, S, x, xq, 0, xd, hid, gu, hs, ff, ffs, hid, plan);
}
extern "C" __global__ void __launch_bounds__(256) g5x_gu_q2(G5W g, G5W u, G5W sg, G5W su, const unsigned* ids, int k,
                                                           int ne, int S, const float* x, const unsigned char* xq,
                                                           const unsigned char* xq2, const float* xd, float* gu, int hs,
                                                           int ff, int ffs, int hid, const unsigned* plan) {
    g5x_body<false, true>(g, u, sg, su, ids, 0, k, ne, S, x, xq, xq2, xd, hid, gu, hs, ff, ffs, hid, plan);
}
// down: activations h / hq / hq2 / hd per pair at stride hs.
extern "C" __global__ void __launch_bounds__(256) g5x_down_q(G5W d, G5W sd, const unsigned* ids, const float* wts, int k,
                                                            int ne, int S, const float* h, const unsigned char* hq,
                                                            const float* hd, float* yp, int hs, int ff, int ffs, int hid,
                                                            const unsigned* plan) {
    g5x_body<true, false>(d, d, sd, sd, ids, wts, k, ne, S, h, hq, 0, hd, hs, yp, hs, ff, ffs, hid, plan);
}
extern "C" __global__ void __launch_bounds__(256) g5x_down_q2(G5W d, G5W sd, const unsigned* ids, const float* wts,
                                                             int k, int ne, int S, const float* h,
                                                             const unsigned char* hq, const unsigned char* hq2,
                                                             const float* hd, float* yp, int hs, int ff, int ffs,
                                                             int hid, const unsigned* plan) {
    g5x_body<true, true>(d, d, sd, sd, ids, wts, k, ne, S, h, hq, hq2, hd, hs, yp, hs, ff, ffs, hid, plan);
}

// h[p][r] = silu(min(g, L)) * clamp(u, ±L) from gu [n][2 hs] (g5f_gu's epilogue); rows past the
// slot's width and sentinel pairs 0. Grid (ceil(hs / 128), n), block 128.
extern "C" __global__ void g5x_silu(const float* gu, float* h, const unsigned* ids, int k, int ne, int S, int hs, int ff,
                                    int ffs, float lim, float lims) {
    const int r = blockIdx.x * 128 + threadIdx.x, p = blockIdx.y;
    if (r >= hs) return;
    const int t = p / S, j = p % S;
    const bool sh = j >= k;
    const bool sent = !sh && ids[(size_t)t * k + j] >= (unsigned)ne;
    const int F = sh ? ffs : ff;
    float v = 0.f;
    if (!sent && r < F) {
        const float L = sh ? lims : lim;
        const float a = gu[(size_t)p * 2 * hs + r], b = gu[(size_t)p * 2 * hs + hs + r];
        const float gg = fminf(a, L), uu = fminf(fmaxf(b, -L), L);
        v = gg / (1.0f + expf(-gg)) * uu;
    }
    h[(size_t)p * hs + r] = v;
}
