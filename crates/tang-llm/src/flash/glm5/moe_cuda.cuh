// GLM-5.3-Flash MoE, CUDA (NVRTC). NAIVE ON PURPOSE (GATES.md: WORK -> RIGHT -> FAST): one thread
// per output value, plain loops, weights dequantised element by element straight from their GGUF
// bytes, f32 activations and accumulation. No shared memory, no warp shuffles, no int8 activations.
// Optimisation ideas are listed under "later" in moe.rs. The FAST path (g5f_*, TANG_GLM5_MOE=fast)
// is at the end of this file, device-only; the naive kernels stay the reference.
//
// The file only uses __global__/__device__, blockIdx/blockDim/threadIdx .x, expf/fminf/fmaxf and
// __uint_as_float, so scripts/glm5_moe_cuda_host.cpp can compile it as host C++ and execute every
// kernel with an emulated grid (that is how it is checked on a Mac with no CUDA).
//
// Weight view G5W: matrix e of a weight tensor starts at base + e * estride; row r of it at
// + r * row_bytes; ty is the ggml type id (0 F32, 1 F16, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K). A 2-D
// tensor is estride 0. GGUF fused experts [in, out, E] have estride = out * row_bytes.
//
// One MoE layer (T tokens, K = top_k, x [T][hid] f32 on device):
//   g5_router_logits  grid ceil(T*ne/128)          s    [T][ne]   sigmoid(router . x)
//   g5_router_select  grid ceil(T/128)             ids  [T][K] u32, wts [T][K] f32
//   g5_expert_gu      grid ceil(T*K*ff/128)        h    [T*K][ff] silu(min(g,L)) * clamp(u,±L)
//   g5_expert_down    grid ceil(T*K*hid/128)       yp   [T*K][hid]
//   shared expert:    g5_expert_gu / g5_expert_down with ids = 0 (pair = token, expert 0) -> ysh
//   g5_combine        grid ceil(T*hid/128)         y    [T][hid] = sum_k wts * yp + ysh
// Dense FFN (first 3 layers): g5_expert_gu + g5_expert_down with ids = 0, writing y directly.
// All kernels: block 128 (any size works; each thread bounds-checks its index).

// nvrtc has no <cmath>: use the bit pattern for +/−infinity.
#define G5_INF_F __uint_as_float(0x7f800000u)
#define G5_NEG_INF_F __uint_as_float(0xff800000u)

struct G5W {
    unsigned long long base;
    unsigned long long estride;
    unsigned row_bytes;
    unsigned ty;
};

__device__ __forceinline__ unsigned g5_u16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
__device__ __forceinline__ unsigned g5_u32(const unsigned char* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

// IEEE half -> float by bits (no intrinsics, so the host build is the same code).
__device__ __forceinline__ float g5_h2f(unsigned h) {
    const unsigned sign = (h & 0x8000u) << 16;
    int ex = (int)((h >> 10) & 0x1f);
    unsigned man = h & 0x3ffu;
    if (ex == 0x1f) return __uint_as_float(sign | 0x7f800000u | (man << 13));
    if (ex == 0) {
        if (man == 0) return __uint_as_float(sign);
        ex = 1;
        while (!(man & 0x400u)) { man <<= 1; ex--; }
        man &= 0x3ffu;
    }
    return __uint_as_float(sign | ((unsigned)(ex + 112) << 23) | (man << 13));
}

// ggml get_scale_min_k4
__device__ __forceinline__ void g5_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Element j of a row of type ty (ggml dequantize_row_* semantics). Unknown types give NaN.
__device__ float g5_deq(unsigned ty, const unsigned char* row, int j) {
    if (ty == 0) return __uint_as_float(g5_u32(row + 4 * j));
    if (ty == 1) return g5_h2f(g5_u16(row + 2 * j));
    if (ty == 8) {  // Q8_0: 32 per block, 34 B: f16 d, int8 qs[32]
        const unsigned char* b = row + (j / 32) * 34;
        return g5_h2f(g5_u16(b)) * (float)(signed char)b[2 + j % 32];
    }
    if (ty == 6) {  // Q5_0: 32 per block, 22 B: f16 d, 32 high bits, 16 nibble bytes
        const unsigned char* b = row + (j / 32) * 22;
        const unsigned qh = g5_u32(b + 2);
        const int e = j % 32, l = e & 15;
        const int q = ((e < 16 ? b[6 + l] : b[6 + l] >> 4) & 0xF) | (((qh >> (e < 16 ? l : l + 16)) & 1) << 4);
        return g5_h2f(g5_u16(b)) * (float)(q - 16);
    }
    if (ty == 12 || ty == 13) {  // Q4_K 144 B / Q5_K 176 B: f16 d, f16 dmin, scales[12], (qh[32]), qs[128]
        const unsigned char* b = row + (j / 256) * (ty == 12 ? 144 : 176);
        const int e = j % 256, c = e / 64, w = e % 64, hi = w / 32, l = w % 32;
        int sc, mn;
        g5_scale_min_k4(2 * c + hi, b + 4, &sc, &mn);
        const unsigned char* qs = b + (ty == 12 ? 16 : 48);
        int q = (qs[c * 32 + l] >> (4 * hi)) & 0xF;
        if (ty == 13) q += ((b[16 + l] >> (2 * c + hi)) & 1) << 4;
        return g5_h2f(g5_u16(b)) * (float)sc * (float)q - g5_h2f(g5_u16(b + 2)) * (float)mn;
    }
    if (ty == 14) {  // Q6_K 210 B: ql[128], qh[64], int8 scales[16], f16 d
        const unsigned char* b = row + (j / 256) * 210;
        const int e = j % 256, n = e / 128, r = e % 128, qd = r / 32, l = r % 32;
        // slice n: ql_n = b + n*64, qh_n = b + 128 + n*32. Per (l, qd):
        //   qd 0: lo = ql_n[l] lo nibble  qd 1: lo = ql_n[l+32] lo nibble
        //   qd 2: lo = ql_n[l] hi nibble  qd 3: lo = ql_n[l+32] hi nibble
        //   h2 = qh_n[l] >> (2*qd) & 3; the value's scale row is sc[2*qd + l/16].
        const unsigned char* ql = b + n * 64;
        const unsigned char* qh = b + 128 + n * 32;
        const int lo = (ql[l + (qd & 1) * 32] >> ((qd & 2) ? 4 : 0)) & 0xF;
        const int h2 = (qh[l] >> (2 * qd)) & 3;
        const int sc = (signed char)b[192 + n * 8 + (qd * 2 + l / 16)];
        return g5_h2f(g5_u16(b + 208)) * (float)sc * (float)((lo | (h2 << 4)) - 32);
    }
    return __uint_as_float(0x7fc00000u);
}

// The upload-repacked rows (moe_gpu.rs `repack_row`; the GGUF on disk is untouched), split so
// every 32-weight quant payload is 16-byte aligned for g5f_lane_dot32. Row of k weights:
//   108 (Q8_0):  int8 qs[k] | f16 d[k/32]                    (padded to 16 B)
//   114 (Q6_K):  per 256-block {ql[128] qh[64]} x nb | int8 sc[16] x nb | f16 d x nb  (nb = k/256)
// Same values as 8 / 14 (the bytes are only moved), so the element decode needs k.
#define G5_Q8R 108u
#define G5_Q6R 114u
__device__ float g5_deq_r(unsigned ty, const unsigned char* row, int j, int k) {
    if (ty == G5_Q8R) return g5_h2f(g5_u16(row + k + 2 * (j / 32))) * (float)(signed char)row[j];
    if (ty == G5_Q6R) {
        const int nb = k / 256, bi = j / 256, e = j % 256, n = e / 128, r = e % 128, qd = r / 32, l = r % 32;
        const unsigned char* ql = row + bi * 192 + n * 64;
        const unsigned char* qh = row + bi * 192 + 128 + n * 32;
        const int lo = (ql[l + (qd & 1) * 32] >> ((qd & 2) ? 4 : 0)) & 0xF;
        const int h2 = (qh[l] >> (2 * qd)) & 3;
        const int sc = (signed char)row[nb * 192 + bi * 16 + n * 8 + (qd * 2 + l / 16)];
        return g5_h2f(g5_u16(row + nb * 208 + 2 * bi)) * (float)sc * (float)((lo | (h2 << 4)) - 32);
    }
    return g5_deq(ty, row, j);
}

// dot(row r of matrix e of w, x[0..k))
__device__ float g5_dot(const G5W w, unsigned e, int r, const float* x, int k) {
    const unsigned char* row = (const unsigned char*)(w.base + (unsigned long long)e * w.estride +
                                                      (unsigned long long)r * w.row_bytes);
    float acc = 0.f;
    if (w.ty == G5_Q8R || w.ty == G5_Q6R) {
        for (int j = 0; j < k; j++) acc += g5_deq_r(w.ty, row, j, k) * x[j];
        return acc;
    }
    for (int j = 0; j < k; j++) acc += g5_deq(w.ty, row, j) * x[j];
    return acc;
}

// s[t][e] = sigmoid(dot(router row e, x[t]))
extern "C" __global__ void g5_router_logits(const G5W w, const float* x, float* s, int T, int ne, int hid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * ne) return;
    const int t = i / ne, e = i % ne;
    const float a = g5_dot(w, 0u, e, x + (long long)t * hid, hid);
    s[i] = 1.0f / (1.0f + expf(-a));
}

// noaux_tc selection, one thread per token: choice = s + bias (bias may be null); optional group
// limit (group score = sum of its top-2 choice values, keep topk_group groups, lower index wins
// ties); top-k on choice (lower index wins ties); weights = unbiased s of the chosen, divided by
// (sum + 1e-20) if norm, times scale (HF Glm5NextTextTopkRouter). n_group <= 64.
extern "C" __global__ void g5_router_select(const float* s, const float* bias, unsigned* ids, float* wts,
                                            int T, int ne, int k, int n_group, int topk_group, int norm,
                                            float scale) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* st = s + (long long)t * ne;
    unsigned* id = ids + (long long)t * k;
    float* wt = wts + (long long)t * k;
    const int gs = n_group > 1 ? ne / n_group : ne;
    unsigned long long gmask = 1ull;
    if (n_group > 1) {
        gmask = 0ull;
        for (int pick = 0; pick < topk_group; pick++) {
            float best = G5_NEG_INF_F;
            int bg = -1;
            for (int g = 0; g < n_group; g++) {
                if ((gmask >> g) & 1ull) continue;
                float a = G5_NEG_INF_F, b2 = G5_NEG_INF_F;
                for (int e = g * gs; e < (g + 1) * gs; e++) {
                    const float v = st[e] + (bias ? bias[e] : 0.f);
                    if (v > a) { b2 = a; a = v; } else if (v > b2) b2 = v;
                }
                if (bg < 0 || a + b2 > best) { best = a + b2; bg = g; }
            }
            gmask |= 1ull << bg;
        }
    }
    float wsum = 0.f;
    for (int j = 0; j < k; j++) {
        float best = G5_NEG_INF_F;
        int bi = -1;
        for (int e = 0; e < ne; e++) {
            if (!((gmask >> (e / gs)) & 1ull)) continue;
            bool taken = false;
            for (int q = 0; q < j; q++) taken = taken || id[q] == (unsigned)e;
            if (taken) continue;
            const float v = st[e] + (bias ? bias[e] : 0.f);
            if (bi < 0 || v > best) { best = v; bi = e; }
        }
        id[j] = bi < 0 ? (unsigned)ne : (unsigned)bi;
        wt[j] = bi < 0 ? 0.f : st[bi];
        wsum += wt[j];
    }
    for (int j = 0; j < k; j++) {
        if (norm) wt[j] = wt[j] / (wsum + 1e-20f);
        wt[j] = wt[j] * scale;
    }
}

// h[p][r] = silu(min(gate_r . x, L)) * clamp(up_r . x, -L, L) for pair p. With ids: pair p is
// (token p / k, expert ids[p]); an id >= ne (the "no expert" sentinel) gives 0. Without ids
// (null): pair p is token p on matrix 0 (shared expert, dense FFN). L = INFINITY disables.
extern "C" __global__ void g5_expert_gu(const G5W gate, const G5W up, const unsigned* ids, int k, int ne,
                                        const float* x, float* h, int npairs, int ff, int hid, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npairs * ff) return;
    const int p = i / ff, r = i % ff;
    const unsigned e = ids ? ids[p] : 0u;
    const int t = ids ? p / k : p;
    if (ids && e >= (unsigned)ne) { h[i] = 0.f; return; }
    const float* xt = x + (long long)t * hid;
    const float g = fminf(g5_dot(gate, e, r, xt, hid), limit);
    const float u = fminf(fmaxf(g5_dot(up, e, r, xt, hid), -limit), limit);
    h[i] = g / (1.0f + expf(-g)) * u;
}

// yp[p][c] = down_c . h[p] (pairs as in g5_expert_gu)
extern "C" __global__ void g5_expert_down(const G5W down, const unsigned* ids, int ne, const float* h,
                                          float* yp, int npairs, int ff, int hid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npairs * hid) return;
    const int p = i / hid, c = i % hid;
    const unsigned e = ids ? ids[p] : 0u;
    if (ids && e >= (unsigned)ne) { yp[i] = 0.f; return; }
    yp[i] = g5_dot(down, e, c, h + (long long)p * ff, ff);
}

// y[t][c] = sum_j wts[t][j] * yp[t*k + j][c] (+ ysh[t][c] if ysh is not null)
extern "C" __global__ void g5_combine(const float* yp, const float* wts, const float* ysh, float* y,
                                      int T, int k, int hid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * hid) return;
    const int t = i / hid, c = i % hid;
    float acc = 0.f;
    for (int j = 0; j < k; j++) acc += wts[t * k + j] * yp[((long long)t * k + j) * hid + c];
    if (ysh) acc += ysh[i];
    y[i] = acc;
}

// =============================================================================================
// FAST path (TANG_GLM5_MOE=fast; docs/models/glm5-speed-plan.md steps 1 + 5, expert side).
// Device-only (dp4a, warp shuffles, shared memory, cp.async), so it is skipped by the host-C++
// check build. Everything else above is unchanged and still the reference the tests pin.
//
// - Dequant-reuse GEMV: a WARP per output row (not a thread). The row's GGUF bytes are staged
//   into shared memory with coalesced 16-byte copies, then the 32 lanes walk it 8 weights at a
//   time: lane L takes elements L*8 + 256*i. Quantized rows (Q5_0, Q8_0, Q4_K, Q5_K, Q6_K) are
//   decoded to int8 lanes exactly as the Flash engine's decode8i (kernels.rs) and dotted with
//   __dp4a against int8 activations (per-32 scale d = amax/127, round half away — fe_q8's
//   rule); F32 / F16 rows use f32 activations. K-quant mins: dot = dx * (dw*s - mw*sum(xq)).
// - Grouped launch: ONE gate|up launch and ONE down launch per layer for all of a token's
//   routed experts + the shared expert. grid.y = token; a token's ids / weights sit in shared
//   memory; each warp loops over the token's slots. g5f_down also does the combine (weighted
//   sum over slots, then + shared), so there is no yp buffer and no g5_combine.
// - Dense FFN (first 3 layers): the same two kernels with ids = null (one slot, matrix 0).
//
// Slot layout of h: row (t * S + j), S = nr + ns, nr = ids ? k : 1, ns = shared ? 1 : 0; row
// stride hs (>= max(ff, ffs), multiple of 32). hq / hd = g5f_q8 of h (codes / per-32 scales).
// Weight views are G5W as above. Staging buffers: per warp 2 rows of row16 uint4s (dynamic
// smem = 4 warps * 2 * row16 * 16 B); row16 = 0 disables staging (rows read from global).
// =============================================================================================
#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
// min resident blocks per SM for g5f_gu / g5f_down (caps registers: 4 -> 128, 8 -> 64)
#ifndef G5F_MINB
#define G5F_MINB 1
#endif

__device__ __forceinline__ float g5f_i2f(int q) { return __int_as_float(0x4B400000 + q) - 12582912.0f; }
__device__ __forceinline__ float g5f_h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short g5f_rd16(const unsigned char* p) {
    return (unsigned short)p[0] | ((unsigned short)p[1] << 8);
}
// 8 bytes at any alignment: three aligned 32-bit loads + two funnel shifts (kernels.rs rd64).
// The third word only when p is unaligned (sh == 0 ignores it): an aligned read of a row's
// last 8 bytes used to load 4 bytes past the allocation (compute-sanitizer, g5f_gemv on the
// last Q8_0 row; NOTES 2026-10-29 kda4 round 6). Same value either way.
__device__ __forceinline__ unsigned long long g5f_rd64(const unsigned char* p) {
    const unsigned long long a = (unsigned long long)p;
    const unsigned* q = (const unsigned*)(a & ~3ull);
    const unsigned sh = (unsigned)(a & 3) * 8;
    const unsigned w0 = q[0], w1 = q[1], w2 = sh ? q[2] : 0u;
    const unsigned lo = __funnelshift_r(w0, w1, sh), hi = __funnelshift_r(w1, w2, sh);
    return (unsigned long long)lo | ((unsigned long long)hi << 32);
}
__device__ __forceinline__ unsigned g5f_spread1(unsigned b) {
    return (b & 1u) | ((b & 2u) << 7) | ((b & 4u) << 14) | ((b & 8u) << 21);
}
__device__ __forceinline__ void g5f_smk4(int j, const unsigned char* q, int& s, int& m) {
    if (j < 4) { s = q[j] & 63; m = q[j + 4] & 63; }
    else { s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// kernels.rs decode8i, verbatim in logic: weights e0..e0+8 (e0 % 8 == 0) as signed int8 lanes of
// (w0, w1), weight = dw * code - mw. Q4_K/Q5_K nibbles are NOT 8 contiguous bytes of one
// nibble plane: byte l of chunk j holds element (j*64 + l) low and (j*64 + 32 + l) high.
template <int TY>
__device__ __forceinline__ void g5f_decode8i(const unsigned char* row, int e0, int& w0, int& w1, float& dw, float& mw) {
    mw = 0.f;
    if (TY == 8) {
        const unsigned char* b = row + (e0 / 32) * 34;
        dw = g5f_h2f(g5f_rd16(b));
        const unsigned long long q = g5f_rd64(b + 2 + e0 % 32);
        w0 = (int)(unsigned)q; w1 = (int)(unsigned)(q >> 32);
    } else if (TY == 6) {
        const unsigned char* b = row + (e0 / 32) * 22;
        dw = g5f_h2f(g5f_rd16(b));
        const unsigned qh = (unsigned)b[2] | ((unsigned)b[3] << 8) | ((unsigned)b[4] << 16) | ((unsigned)b[5] << 24);
        const int j0 = e0 % 32, sh = (j0 >> 4) * 4;
        const unsigned long long q = g5f_rd64(b + 6 + (j0 & 15));
        const unsigned h = qh >> j0;
        const unsigned n0 = (((unsigned)q >> sh) & 0x0F0F0F0Fu) | (g5f_spread1(h & 0xF) << 4);
        const unsigned n1 = (((unsigned)(q >> 32) >> sh) & 0x0F0F0F0Fu) | (g5f_spread1((h >> 4) & 0xF) << 4);
        w0 = (int)__vsub4(n0, 0x10101010u); w1 = (int)__vsub4(n1, 0x10101010u);
    } else if (TY == 12 || TY == 13) {
        const int bb = TY == 12 ? 144 : 176;
        const unsigned char* b = row + (e0 / 256) * bb;
        const int e = e0 % 256, j = e / 64, w = e % 64, l = w % 32, up = w >> 5;
        int s, m;
        g5f_smk4(2 * j + up, b + 4, s, m);
        dw = g5f_h2f(g5f_rd16(b)) * g5f_i2f(s);
        mw = g5f_h2f(g5f_rd16(b + 2)) * g5f_i2f(m);
        const unsigned long long q = g5f_rd64(b + (TY == 12 ? 16 : 48) + j * 32 + l);
        unsigned n0 = ((unsigned)q >> (4 * up)) & 0x0F0F0F0Fu, n1 = ((unsigned)(q >> 32) >> (4 * up)) & 0x0F0F0F0Fu;
        if (TY == 13) {
            const unsigned long long qh = g5f_rd64(b + 16 + l);
            const int hb = 2 * j + up;
            n0 |= (((unsigned)qh >> hb) & 0x01010101u) << 4;
            n1 |= (((unsigned)(qh >> 32) >> hb) & 0x01010101u) << 4;
        }
        w0 = (int)n0; w1 = (int)n1;
    } else if (TY == 14) {
        const unsigned char* b = row + (e0 / 256) * 210;
        const int e = e0 % 256, n = e / 128, r = e % 128, qt = r / 32, l = r % 32;
        dw = g5f_h2f(g5f_rd16(b + 208)) * g5f_i2f((int)(signed char)b[192 + n * 8 + l / 16 + 2 * qt]);
        const unsigned long long ql = g5f_rd64(b + n * 64 + l + 32 * (qt & 1));
        const unsigned long long qh = g5f_rd64(b + 128 + n * 32 + l);
        const int ls = 4 * (qt >> 1), hs = 2 * qt;
        const unsigned n0 = (((unsigned)ql >> ls) & 0x0F0F0F0Fu) | ((((unsigned)qh >> hs) & 0x03030303u) << 4);
        const unsigned n1 = (((unsigned)(ql >> 32) >> ls) & 0x0F0F0F0Fu) | ((((unsigned)(qh >> 32) >> hs) & 0x03030303u) << 4);
        w0 = (int)__vsub4(n0, 0x20202020u); w1 = (int)__vsub4(n1, 0x20202020u);
    } else {
        w0 = w1 = 0; dw = 0.f;
    }
}

// This lane's share of dot(row, x) over k (k % 8 == 0). xq / xd: the token's int8 codes and
// per-32 scales; xq2 (null = off): the residual codes, x ~ d * (q + q2 / 254) (g5f_q8), which
// takes the activation error from ~1e-2 to ~1e-5 for 2 (4 with K-quant mins) more dp4a per 8
// weights — the decode is shared. xf: the token's f32 values (F32 / F16 rows).
// unroll of the 8-weight row loop in g5f_gux / g5f_dot_anyn (Q8_0 shared expert / dense rows):
// same order, more loads in flight. g5f_gemv keeps 1 (unrolling it was slower there).
#ifndef G5F_LDU
#define G5F_LDU 0
#endif
// one 8-weight step of g5f_lane_dot (weights e0..e0+8) into acc
template <int TY>
__device__ __forceinline__ void g5f_lane_step(const unsigned char* w, const unsigned char* xq, const unsigned char* xq2,
                                             const float* xd, const float* xf, int e0, float& acc) {
    if (TY == 0 || TY == 1) {
        float v[8];
        if (TY == 0) {
            const float4 a = *(const float4*)(w + 4 * e0), b = *(const float4*)(w + 4 * e0 + 16);
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
            const unsigned long long q0 = g5f_rd64(w + 2 * e0), q1 = g5f_rd64(w + 2 * e0 + 8);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                v[i] = g5f_h2f((unsigned short)(q0 >> (16 * i)));
                v[4 + i] = g5f_h2f((unsigned short)(q1 >> (16 * i)));
            }
        }
        const float4 a = *(const float4*)(xf + e0), c = *(const float4*)(xf + e0 + 4);
        acc = fmaf(v[0], a.x, acc); acc = fmaf(v[1], a.y, acc); acc = fmaf(v[2], a.z, acc); acc = fmaf(v[3], a.w, acc);
        acc = fmaf(v[4], c.x, acc); acc = fmaf(v[5], c.y, acc); acc = fmaf(v[6], c.z, acc); acc = fmaf(v[7], c.w, acc);
    } else {
        int w0, w1;
        float dw, mw;
        g5f_decode8i<TY>(w, e0, w0, w1, dw, mw);
        const uint2 xv = *(const uint2*)(xq + e0);
        const float dx = xd[e0 / 32];
        float sf = g5f_i2f(__dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0)));
        uint2 xv2 = make_uint2(0u, 0u);
        if (xq2) {
            xv2 = *(const uint2*)(xq2 + e0);
            sf = fmaf(g5f_i2f(__dp4a(w0, (int)xv2.x, __dp4a(w1, (int)xv2.y, 0))), 1.0f / 254.0f, sf);
        }
        if (TY == 12 || TY == 13) {
            float sx = g5f_i2f(__dp4a(0x01010101, (int)xv.x, __dp4a(0x01010101, (int)xv.y, 0)));
            if (xq2)
                sx = fmaf(g5f_i2f(__dp4a(0x01010101, (int)xv2.x, __dp4a(0x01010101, (int)xv2.y, 0))), 1.0f / 254.0f, sx);
            acc = fmaf(dx, fmaf(dw, sf, -mw * sx), acc);
        } else {
            acc = fmaf(dx * dw, sf, acc);
        }
    }
}
// U = 0: the compiler's own unrolling (g5f_gemv; the pre-G5F_LDU code); U > 0: #pragma unroll U
template <int TY, int U = 0>
__device__ __forceinline__ float g5f_lane_dot(const unsigned char* w, const unsigned char* xq, const unsigned char* xq2,
                                             const float* xd, const float* xf, int k) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    if (U == 0) {
        for (int e0 = lane * 8; e0 < k; e0 += 256) g5f_lane_step<TY>(w, xq, xq2, xd, xf, e0, acc);
    } else {
        #pragma unroll (U > 0 ? U : 1)
        for (int e0 = lane * 8; e0 < k; e0 += 256) g5f_lane_step<TY>(w, xq, xq2, xd, xf, e0, acc);
    }
    return acc;
}

// Q4_K / Q5_K at 32-weight granularity: lane L takes whole 32-chunks c = L + 32 i (one
// sub-block each), so the 6-bit scale / min decode happens once per 32 weights instead of once
// per 8, and the chunk's nibbles, high bits and int8 activations arrive as 16-byte loads (both
// block sizes, 144 and 176, are multiples of 16; the caller checks w is 16-aligned and
// k % 256 == 0). Same arithmetic as g5f_lane_dot (dot = dx * (dw * sum(q x) - mw * sum(x))),
// summed in a different order.
#ifndef G5F_DOT32
#define G5F_DOT32 1
#endif
#ifndef G5F_UNROLL
#define G5F_UNROLL 1
#endif
// chunk c's contribution (Q4_K / Q5_K)
template <int TY>
__device__ __forceinline__ float g5f_c32k(const unsigned char* w, int c, const unsigned char* xq, const unsigned char* xq2,
                                         const float* xd) {
    const int bb = TY == 12 ? 144 : 176;
    {
        const unsigned char* b = w + (c >> 3) * bb;
        const int j = c & 7, up = j & 1;
        int s, m;
        g5f_smk4(j, b + 4, s, m);
        const float dw = g5f_h2f(g5f_rd16(b)) * (float)s;
        const float mw = g5f_h2f(g5f_rd16(b + 2)) * (float)m;
        const uint4* qp = (const uint4*)(b + (TY == 12 ? 16 : 48) + (j >> 1) * 32);
        const uint4 q0 = qp[0], q1 = qp[1];
        unsigned n[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        #pragma unroll
        for (int i = 0; i < 8; i++) n[i] = (n[i] >> (4 * up)) & 0x0F0F0F0Fu;
        if (TY == 13) {
            const uint4* hp = (const uint4*)(b + 16);
            const uint4 h0 = hp[0], h1 = hp[1];
            const unsigned hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            #pragma unroll
            for (int i = 0; i < 8; i++) n[i] |= ((hw[i] >> j) & 0x01010101u) << 4;
        }
        const uint4* xp = (const uint4*)(xq + (size_t)c * 32);
        const uint4 x0 = xp[0], x1 = xp[1];
        const unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        int sf = 0, sx = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sf = __dp4a((int)n[i], (int)xw[i], sf);
            sx = __dp4a(0x01010101, (int)xw[i], sx);
        }
        float fsf = (float)sf, fsx = (float)sx;
        if (xq2) {
            const uint4* x2p = (const uint4*)(xq2 + (size_t)c * 32);
            const uint4 y0 = x2p[0], y1 = x2p[1];
            const unsigned yw[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
            int sf2 = 0, sx2 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                sf2 = __dp4a((int)n[i], (int)yw[i], sf2);
                sx2 = __dp4a(0x01010101, (int)yw[i], sx2);
            }
            fsf = fmaf((float)sf2, 1.0f / 254.0f, fsf);
            fsx = fmaf((float)sx2, 1.0f / 254.0f, fsx);
        }
        return xd[c] * fmaf(dw, fsf, -mw * fsx);
    }
}

// The repacked Q8_0 (108) / Q6_K (114) rows at 32-weight granularity, lane L on chunks
// c = L + 32 i as g5f_lane_dot32: two 16-byte payload loads per chunk (Q6_K: ql + qh, 32 B each),
// scales from the row's tail. dot = dx * dw * sum(q x) (Q6_K: two 16-weight halves, one int8
// scale each). k % 32 == 0 (108) / k % 256 == 0 (114); w 16-aligned (the repack guarantees both).
__device__ __forceinline__ int g5f_dp8(const unsigned* n, const uint4 a, const uint4 b, int i0, int cnt) {
    const unsigned xw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    int s = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        if (i >= i0 && i < i0 + cnt) s = __dp4a((int)n[i], (int)xw[i], s);
    return s;
}
// chunk c's contribution (repacked Q8_0 / Q6_K; k = row length)
template <int TY>
__device__ __forceinline__ float g5f_c32r(const unsigned char* w, int c, int k, const unsigned char* xq,
                                         const unsigned char* xq2, const float* xd) {
    const int nb = k / 256;
    {
        unsigned n[8];
        float d0, d1;
        if (TY == 108) {
            const uint4* qp = (const uint4*)(w + (size_t)c * 32);
            const uint4 q0 = qp[0], q1 = qp[1];
            n[0] = q0.x; n[1] = q0.y; n[2] = q0.z; n[3] = q0.w; n[4] = q1.x; n[5] = q1.y; n[6] = q1.z; n[7] = q1.w;
            d0 = d1 = g5f_h2f(g5f_rd16(w + k + 2 * c));
        } else {
            const int bi = c >> 3, j = c & 7, hn = j >> 2, qt = j & 3;
            const unsigned char* b = w + (size_t)bi * 192;
            const uint4* lp = (const uint4*)(b + hn * 64 + 32 * (qt & 1));
            const uint4* hp = (const uint4*)(b + 128 + hn * 32);
            const uint4 l0 = lp[0], l1 = lp[1], h0 = hp[0], h1 = hp[1];
            const unsigned lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
            const unsigned hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            const int ls = 4 * (qt >> 1), hs = 2 * qt;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                n[i] = __vsub4(((lw[i] >> ls) & 0x0F0F0F0Fu) | (((hw[i] >> hs) & 0x03030303u) << 4), 0x20202020u);
            const float d = g5f_h2f(g5f_rd16(w + nb * 208 + 2 * bi));
            const signed char* sc = (const signed char*)(w + nb * 192 + bi * 16 + hn * 8 + 2 * qt);
            d0 = d * (float)sc[0];
            d1 = d * (float)sc[1];
        }
        const uint4* xp = (const uint4*)(xq + (size_t)c * 32);
        const uint4 x0 = xp[0], x1 = xp[1];
        float f0 = (float)g5f_dp8(n, x0, x1, 0, 4), f1 = (float)g5f_dp8(n, x0, x1, 4, 4);
        if (xq2) {
            const uint4* x2p = (const uint4*)(xq2 + (size_t)c * 32);
            const uint4 y0 = x2p[0], y1 = x2p[1];
            f0 = fmaf((float)g5f_dp8(n, y0, y1, 0, 4), 1.0f / 254.0f, f0);
            f1 = fmaf((float)g5f_dp8(n, y0, y1, 4, 4), 1.0f / 254.0f, f1);
        }
        return xd[c] * fmaf(d0, f0, d1 * f1);
    }
}

// one 32-chunk dot (TY 12 / 13 / 108 / 114) over one row, or two rows a and b at once (b != 0:
// both rows' loads in flight per iteration); lane L on chunks L + 32 i
template <int TY>
__device__ __forceinline__ float g5f_c32(const unsigned char* w, int c, int k, const unsigned char* xq,
                                        const unsigned char* xq2, const float* xd) {
    if (TY == 12 || TY == 13) return g5f_c32k<TY>(w, c, xq, xq2, xd);
    return g5f_c32r<TY>(w, c, k, xq, xq2, xd);
}
template <int TY>
__device__ __forceinline__ float2 g5f_lane_dot32x(const unsigned char* wa, const unsigned char* wb, const unsigned char* xq,
                                                 const unsigned char* xq2, const float* xd, int k) {
    const int lane = threadIdx.x & 31;
    float a = 0.f, b = 0.f;
    #pragma unroll G5F_UNROLL
    for (int c = lane; c < k / 32; c += 32) {
        a += g5f_c32<TY>(wa, c, k, xq, xq2, xd);
        if (wb) b += g5f_c32<TY>(wb, c, k, xq, xq2, xd);
    }
    return make_float2(a, b);
}
template <int TY>
__device__ __forceinline__ float g5f_lane_dot32(const unsigned char* w, const unsigned char* xq, const unsigned char* xq2,
                                               const float* xd, int k) {
    return g5f_lane_dot32x<TY>(w, 0, xq, xq2, xd, k).x;
}

template <int U = 0>
__device__ __forceinline__ float g5f_dot_any(unsigned ty, const unsigned char* w, const unsigned char* xq,
                                            const unsigned char* xq2, const float* xd, const float* xf, int k) {
    switch (ty) {
        case 0: return g5f_lane_dot<0, U>(w, xq, xq2, xd, xf, k);
        case 1: return g5f_lane_dot<1, U>(w, xq, xq2, xd, xf, k);
        case 6: return g5f_lane_dot<6, U>(w, xq, xq2, xd, xf, k);
        case 8: return g5f_lane_dot<8, U>(w, xq, xq2, xd, xf, k);
        case 12:
            if (G5F_DOT32 && ((unsigned long long)w & 15) == 0 && (k & 255) == 0) return g5f_lane_dot32<12>(w, xq, xq2, xd, k);
            return g5f_lane_dot<12, U>(w, xq, xq2, xd, xf, k);
        case 13:
            if (G5F_DOT32 && ((unsigned long long)w & 15) == 0 && (k & 255) == 0) return g5f_lane_dot32<13>(w, xq, xq2, xd, k);
            return g5f_lane_dot<13, U>(w, xq, xq2, xd, xf, k);
        case 14: return g5f_lane_dot<14, U>(w, xq, xq2, xd, xf, k);
        case 108: return g5f_lane_dot32<108>(w, xq, xq2, xd, k);
        case 114: return g5f_lane_dot32<114>(w, xq, xq2, xd, k);
        default: return __uint_as_float(0x7fc00000u);
    }
}

// Two rows of the same type against the same activations: the 32-chunk types in one fused
// loop (G5F_FUSE2), the rest as two g5f_dot_any. Unsummed lane partials (wb null: .y = 0).
#ifndef G5F_FUSE2
#define G5F_FUSE2 1
#endif
__device__ __forceinline__ float2 g5f_dot_any2(unsigned ty, const unsigned char* wa, const unsigned char* wb,
                                              const unsigned char* xq, const unsigned char* xq2, const float* xd,
                                              const float* xf, int k) {
    const bool al = G5F_FUSE2 && (((unsigned long long)wa | (unsigned long long)wb) & 15) == 0;
    if (al && (k & 255) == 0 && G5F_DOT32) {
        if (ty == 12) return g5f_lane_dot32x<12>(wa, wb, xq, xq2, xd, k);
        if (ty == 13) return g5f_lane_dot32x<13>(wa, wb, xq, xq2, xd, k);
    }
    if (G5F_FUSE2 && ty == 108) return g5f_lane_dot32x<108>(wa, wb, xq, xq2, xd, k);
    if (G5F_FUSE2 && ty == 114) return g5f_lane_dot32x<114>(wa, wb, xq, xq2, xd, k);
    return make_float2(g5f_dot_any(ty, wa, xq, xq2, xd, xf, k), wb ? g5f_dot_any(ty, wb, xq, xq2, xd, xf, k) : 0.f);
}

__device__ __forceinline__ float g5f_warp_sum(float a) {
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    return a;
}

// G5F_PF (TANG_GLM5_MOE_PF, bit mask: 1 gate|up, 2 down): before its dot, a warp asks L2 for every 128-byte line of the
// rows it is about to stream (one prefetch.global.L2 per lane per line, no registers held), so
// a row's DRAM reads are all in flight at once instead of one chunk iteration at a time.
#ifndef G5F_PF
#define G5F_PF 0
#endif
__device__ __forceinline__ const unsigned char* g5f_row(const G5W w, unsigned e, int r) {
    return (const unsigned char*)(w.base + (unsigned long long)e * w.estride + (unsigned long long)r * w.row_bytes);
}
template <int BIT>
__device__ __forceinline__ void g5f_pf(const unsigned char* p, int bytes, int lane) {
    if (!(G5F_PF & BIT)) return;
    const unsigned long long a = (unsigned long long)p & ~127ull;
    const int n = (int)(((unsigned long long)p + bytes - a + 127) >> 7);
    for (int i = lane; i < n; i += 32) asm volatile("prefetch.global.L2 [%0];" :: "l"(a + 128ull * i));
}

// Issue the copy of row r of matrix e of w into sm (aligned 16-byte pieces; cp.async on sm_80+).
// Returns where the row starts (in sm, or the global row when not staged). Call g5f_stage_wait
// + __syncwarp before reading.
__device__ __forceinline__ const unsigned char* g5f_stage(const G5W w, unsigned e, int r, uint4* sm, int row16, int lane) {
    const unsigned char* src = (const unsigned char*)(w.base + (unsigned long long)e * w.estride +
                                                      (unsigned long long)r * w.row_bytes);
    if (row16 == 0 || w.ty == 0) return src;
    const unsigned long long a = (unsigned long long)src & ~15ull;
    const int delta = (int)((unsigned long long)src - a);
    const int n16 = (delta + (int)w.row_bytes + 15) >> 4;
    for (int i = lane; i < n16; i += 32) {
#if __CUDA_ARCH__ >= 800
        const unsigned d = (unsigned)__cvta_generic_to_shared(sm + i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"((const uint4*)a + i) : "memory");
#else
        sm[i] = __ldg((const uint4*)a + i);
#endif
    }
    return (const unsigned char*)sm + delta;
}
__device__ __forceinline__ void g5f_stage_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
    __syncwarp();
}

// x [rows][k] f32 -> q [rows][k] int8 codes, d [rows][k/32] scales, and (q2 != null) the
// residual codes q2 = round((x - d q) * 254 / d) (|x - d q| <= d/2, so |q2| <= 127).
// grid (ceil(k/128), rows), block 128: one warp per 32-chunk. k % 32 == 0.
extern "C" __global__ void g5f_q8(const float* x, unsigned char* q, unsigned char* q2, float* d, int k) {
    const int c = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (c * 32 >= k) return;
    const size_t i = (size_t)blockIdx.y * k + c * 32 + lane;
    const float v = x[i];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    // fmaxf drops NaN, so a poisoned (never written) input would quantise to plausible codes:
    // give its chunk a NaN scale instead, so the debug run sees it.
    const bool nan = __any_sync(0xffffffffu, v != v);
    const float dd = nan ? __uint_as_float(0x7fc00000u) : amax / 127.0f;
    const int qi = nan || dd == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / dd), -127.f), 127.f);
    q[i] = (unsigned char)(signed char)qi;
    if (q2) {
        const float r = nan || dd == 0.f ? 0.f : (v - dd * (float)qi) * 254.0f / dd;
        q2[i] = (unsigned char)(signed char)(int)fminf(fmaxf(roundf(r), -127.f), 127.f);
    }
    if (lane == 0) d[(size_t)blockIdx.y * (k / 32) + c] = dd;
}

// s[t][e] = sigmoid(router_e . x[t]) for an F32 router, f32 throughout. grid (ne, T), block 128:
// the block's 4 warps each take a quarter of the row (8-aligned), summed in a fixed order —
// 4x the warps of a warp-per-row router (288 rows were 72 blocks on a 132-SM part).
extern "C" __global__ void __launch_bounds__(128) g5f_router(const G5W w, const float* x, float* s, int ne, int hid) {
    __shared__ float red[4];
    const int e = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.y;
    const int chunk = ((hid / 8 + 3) / 4) * 8, q0 = warp * chunk, q1 = min(hid, q0 + chunk);
    const unsigned char* row = (const unsigned char*)(w.base + (unsigned long long)e * w.row_bytes);
    const float a = q1 > q0 ? g5f_warp_sum(g5f_lane_dot<0>(row + 4 * (size_t)q0, 0, 0, 0, x + (size_t)t * hid + q0, q1 - q0)) : 0.f;
    if (lane == 0) red[warp] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float v = (red[0] + red[1]) + (red[2] + red[3]);
        s[(size_t)t * ne + e] = 1.0f / (1.0f + expf(-v));
    }
}

// g5_router_select for n_group == 1, one WARP per token (grid T, block 32): k passes of a warp
// argmax over choice = s + bias, lower index on ties; weights as g5_router_select.
#ifndef G5F_SEL2
#define G5F_SEL2 2
#endif
// one warp: token t's top-k from s (sv / sc: 2 x ne floats of shared memory); the previous
// version (TANG_GLM5_MOE_SEL2=0)
__device__ __forceinline__ void g5f_select_warp0(const float* s, const float* bias, unsigned* ids, float* wts, int ne,
                                                int k, int norm, float scale, int t, float* sv, float* sc) {
    const int lane = threadIdx.x & 31;
    for (int e = lane; e < ne; e += 32) {
        const float v = __ldcg(s + (size_t)t * ne + e);
        sv[e] = v;
        sc[e] = v + (bias ? bias[e] : 0.f);
    }
    __syncwarp();
    const float* st = sv;
    unsigned* id = ids + (size_t)t * k;
    float* wt = wts + (size_t)t * k;
    unsigned taken = 0u;  // bit i: expert lane + 32 i is chosen (ne <= 1024)
    float wsum = 0.f;
    for (int j = 0; j < k; j++) {
        float best = G5_NEG_INF_F;
        int bi = -1;
        for (int i = 0; lane + 32 * i < ne; i++) {
            if ((taken >> i) & 1u) continue;
            const int e = lane + 32 * i;
            const float v = sc[e];
            if (bi < 0 || v > best) { best = v; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const bool take = oi >= 0 && (bi < 0 || ob > best || (ob == best && oi < bi));
            if (take) { best = ob; bi = oi; }
        }
        if (bi >= 0 && (bi & 31) == lane) taken |= 1u << (bi >> 5);
        const float w = bi < 0 ? 0.f : st[bi];
        wsum += w;
        if (lane == 0) {
            id[j] = bi < 0 ? (unsigned)ne : (unsigned)bi;
            wt[j] = w;
        }
    }
    __syncwarp();
    if (lane == 0)
        for (int j = 0; j < k; j++) {
            float w = wt[j];
            if (norm) w = w / (wsum + 1e-20f);
            wt[j] = w * scale;
        }
}
// one warp: token t's top-k from s (sv / sc: 2 x ne floats of shared memory). The choice values
// of a lane's experts (lane + 32 i) sit in registers (NPL per lane); each of the k rounds is a
// register scan plus a 5-step shuffle argmax, and lane j keeps round j's id / weight so the
// normalisation and the stores happen once at the end (same values, same float ops).
template <int NPL>
__device__ __forceinline__ void g5f_select_warpn(const float* s, const float* bias, unsigned* ids, float* wts, int ne,
                                                 int k, int norm, float scale, int t, float* sv) {
    const int lane = threadIdx.x & 31;
    float cv[NPL];
    #pragma unroll
    for (int i = 0; i < NPL; i++) {
        const int e = lane + 32 * i;
        float v = 0.f, c = 0.f;
        if (e < ne) {
            v = __ldcg(s + (size_t)t * ne + e);
            c = v + (bias ? bias[e] : 0.f);
            sv[e] = v;
        }
        cv[i] = c;
    }
    __syncwarp();
    unsigned taken = 0u;  // bit i: expert lane + 32 i is chosen
    float wsum = 0.f, myw = 0.f;
    unsigned myid = (unsigned)ne;
    for (int j = 0; j < k; j++) {
        float best = G5_NEG_INF_F;
        int bi = -1;
        #pragma unroll
        for (int i = 0; i < NPL; i++) {
            const int e = lane + 32 * i;
            if (e >= ne || ((taken >> i) & 1u)) continue;
            const float v = cv[i];
            if (bi < 0 || v > best) { best = v; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const bool take = oi >= 0 && (bi < 0 || ob > best || (ob == best && oi < bi));
            if (take) { best = ob; bi = oi; }
        }
        if (bi >= 0 && (bi & 31) == lane) taken |= 1u << (bi >> 5);
        const float w = bi < 0 ? 0.f : sv[bi];
        wsum += w;
        if (lane == j) {
            myid = bi < 0 ? (unsigned)ne : (unsigned)bi;
            myw = w;
        }
    }
    if (lane < k) {
        float w = myw;
        if (norm) w = w / (wsum + 1e-20f);
        ids[(size_t)t * k + lane] = myid;
        wts[(size_t)t * k + lane] = w * scale;
    }
}
// g5f_select_warpn with each round's argmax as two warp reductions (redux.sync, sm_80+) on
// order-preserving u32 keys (max key, then min expert index among the lanes holding it: the same
// lower-index tie rule) instead of a 5-step shuffle pair; chosen / absent experts get key 0
// (every non-NaN float maps above it). TANG_GLM5_MOE_SEL2=2 (default).
__device__ __forceinline__ unsigned g5f_ord(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
template <int NPL>
__device__ __forceinline__ void g5f_select_warpr(const float* s, const float* bias, unsigned* ids, float* wts, int ne,
                                                 int k, int norm, float scale, int t, float* sv) {
    const int lane = threadIdx.x & 31;
    unsigned key[NPL];
    #pragma unroll
    for (int i = 0; i < NPL; i++) {
        const int e = lane + 32 * i;
        key[i] = 0u;
        if (e < ne) {
            const float v = __ldcg(s + (size_t)t * ne + e);
            key[i] = g5f_ord(v + (bias ? bias[e] : 0.f));
            sv[e] = v;
        }
    }
    __syncwarp();
    float wsum = 0.f, myw = 0.f;
    unsigned myid = (unsigned)ne;
    for (int j = 0; j < k; j++) {
        unsigned bk = 0u, bi = 0xFFFFFFFFu;
        #pragma unroll
        for (int i = 0; i < NPL; i++)
            if (key[i] > bk) { bk = key[i]; bi = (unsigned)(lane + 32 * i); }
        const unsigned m = __reduce_max_sync(0xffffffffu, bk);
        const unsigned win = __reduce_min_sync(0xffffffffu, bk == m ? bi : 0xFFFFFFFFu);
        const bool none = m == 0u;
        if (!none && (int)(win & 31) == lane) {
            #pragma unroll
            for (int i = 0; i < NPL; i++)
                if ((unsigned)i == (win >> 5)) key[i] = 0u;
        }
        const float w = none ? 0.f : sv[win];
        wsum += w;
        if (lane == j) {
            myid = none ? (unsigned)ne : win;
            myw = w;
        }
    }
    if (lane < k) {
        float w = myw;
        if (norm) w = w / (wsum + 1e-20f);
        ids[(size_t)t * k + lane] = myid;
        wts[(size_t)t * k + lane] = w * scale;
    }
}
__device__ __forceinline__ void g5f_select_warp(const float* s, const float* bias, unsigned* ids, float* wts, int ne,
                                                int k, int norm, float scale, int t, float* sv, float* sc) {
#if __CUDA_ARCH__ >= 800
    if (G5F_SEL2 >= 2 && k <= 32 && ne <= 320) {
        g5f_select_warpr<10>(s, bias, ids, wts, ne, k, norm, scale, t, sv);
        return;
    }
    if (G5F_SEL2 >= 2 && k <= 32 && ne <= 1024) {
        g5f_select_warpr<32>(s, bias, ids, wts, ne, k, norm, scale, t, sv);
        return;
    }
#endif
    if (G5F_SEL2 && k <= 32 && ne <= 320) {
        g5f_select_warpn<10>(s, bias, ids, wts, ne, k, norm, scale, t, sv);
        return;
    }
    if (G5F_SEL2 && k <= 32) {
        g5f_select_warpn<32>(s, bias, ids, wts, ne, k, norm, scale, t, sv);
        return;
    }
    g5f_select_warp0(s, bias, ids, wts, ne, k, norm, scale, t, sv, sc);
}
extern "C" __global__ void g5f_select(const float* s, const float* bias, unsigned* ids, float* wts, int ne, int k,
                                      int norm, float scale) {
    __shared__ float sv[1024], sc[1024];  // score, choice = score + bias (ne <= 1024)
    g5f_select_warp(s, bias, ids, wts, ne, k, norm, scale, blockIdx.x, sv, sc);
}

// q8(x) + router + select in one launch (routed layers, F32 router, ne <= 1024): grid (ne, T),
// block 128. Block e computes s[t][e] as g5f_router; blocks e < hid/32 also quantise x chunk e
// (g5f_q8, warp 0; ne >= hid/32 is the caller's check); the last block to finish token t (cnt[t],
// zero on entry, reset on exit) runs the top-k select as g5f_select.
// TANG_GLM5_MOE_ROUTE_REL (default 1): the last-block hand-off as one acq_rel atomic
#ifndef G5F_ROUTE_REL
#define G5F_ROUTE_REL 1
#endif
// microbench phase stamps (bench/glm5-moe-roofline, -DG5F_ROUTE_TS): per block entry / dot done,
// the last block's select start / end, %globaltimer ns
#ifdef G5F_ROUTE_TS
__device__ unsigned long long g5f_route_ts[4 * 1024];
#define G5F_TS(i) do { if (threadIdx.x == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g5f_route_ts[i] = t_; } } while (0)
#else
#define G5F_TS(i) do {} while (0)
#endif
extern "C" __global__ void __launch_bounds__(128) g5f_route(
    const G5W w, const float* x, float* s, int ne, int hid, unsigned char* xq, unsigned char* xq2, float* xd,
    const float* bias, unsigned* ids, float* wts, int k, int norm, float scale, unsigned* cnt) {
    __shared__ float red[4];
    __shared__ float sv[1024], sc[1024];
    __shared__ unsigned last;
    const int e = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.y;
    G5F_TS(4 * e);
    if (warp == 0 && e < hid / 32) {
        const size_t i = (size_t)t * hid + e * 32 + lane;
        const float v = x[i];
        float amax = fabsf(v);
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const bool nan = __any_sync(0xffffffffu, v != v);
        const float dd = nan ? __uint_as_float(0x7fc00000u) : amax / 127.0f;
        const int qi = nan || dd == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / dd), -127.f), 127.f);
        xq[i] = (unsigned char)(signed char)qi;
        if (xq2) {
            const float r = nan || dd == 0.f ? 0.f : (v - dd * (float)qi) * 254.0f / dd;
            xq2[i] = (unsigned char)(signed char)(int)fminf(fmaxf(roundf(r), -127.f), 127.f);
        }
        if (lane == 0) xd[(size_t)t * (hid / 32) + e] = dd;
    }
    const int chunk = ((hid / 8 + 3) / 4) * 8, q0 = warp * chunk, q1 = min(hid, q0 + chunk);
    const unsigned char* row = (const unsigned char*)(w.base + (unsigned long long)e * w.row_bytes);
    const float a = q1 > q0 ? g5f_warp_sum(g5f_lane_dot<0>(row + 4 * (size_t)q0, 0, 0, 0, x + (size_t)t * hid + q0, q1 - q0)) : 0.f;
    if (lane == 0) red[warp] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float v = (red[0] + red[1]) + (red[2] + red[3]);
        s[(size_t)t * ne + e] = 1.0f / (1.0f + expf(-v));
        G5F_TS(4 * e + 1);
#if G5F_ROUTE_REL && __CUDA_ARCH__ >= 700
        // one acq_rel atomic instead of fence + atomic + fence (publishes s, and the last
        // block's thread 0 acquires every other block's s; bar.sync extends it to warp 0)
        unsigned old;
        asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt + t) : "memory");
        last = old == (unsigned)ne - 1;
#else
        __threadfence();
        last = atomicAdd(cnt + t, 1u) == (unsigned)ne - 1;
#endif
    }
    __syncthreads();
    if (!last || warp != 0) return;
#if !(G5F_ROUTE_REL && __CUDA_ARCH__ >= 700)
    __threadfence();
#endif
    G5F_TS(4 * e + 2);
    g5f_select_warp(s, bias, ids, wts, ne, k, norm, scale, t, sv, sc);
    if (lane == 0) cnt[t] = 0u;
    G5F_TS(4 * e + 3);
}

// h[t*S + j][r] = silu(min(gate_r . x_t, L)) * clamp(up_r . x_t, ±L) for slot j of token t
// (routed slots j < nr on matrix ids[t][j] — 0 for an id >= ne — then the shared slot on
// sgate/sup when sgate.base != 0, limit lims). grid (ceil(max(ff, ffs)/4), S, T), block 128,
// one (slot, row r) per warp (gate and up rows staged together): every active expert's rows
// are in flight at once instead of a warp walking the slots in turn.
extern "C" __global__ void __launch_bounds__(128, G5F_MINB) g5f_gu(
    const G5W gate, const G5W up, const G5W sgate, const G5W sup, const unsigned* ids, int k, int ne,
    const float* x, const unsigned char* xq, const unsigned char* xq2, const float* xd, float* h, int hs, int ff,
    int ffs, int hid,
    float lim, float lims, int row16) {
    extern __shared__ uint4 g5f_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = blockIdx.y, t = blockIdx.z;
    const int nr = ids ? k : 1, ns = sgate.base ? 1 : 0, S = nr + ns;
    const bool sh = j >= nr;
    const int r = blockIdx.x * 4 + warp;
    if (r >= (sh ? ffs : ff)) return;
    uint4* sm0 = g5f_sm + (size_t)(warp * 2) * row16;
    uint4* sm1 = sm0 + row16;
    const unsigned char* xqt = xq + (size_t)t * hid;
    const unsigned char* xq2t = xq2 ? xq2 + (size_t)t * hid : 0;
    const float* xdt = xd + (size_t)t * (hid / 32);
    const float* xft = x + (size_t)t * hid;
    float* hp = h + (size_t)(t * S + j) * hs;
    const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
    if (!sh && ids && e >= (unsigned)ne) {
        if (lane == 0) hp[r] = 0.f;
        return;
    }
    const G5W g = sh ? sgate : gate, u = sh ? sup : up;
    if (row16 == 0) {
        g5f_pf<1>(g5f_row(g, e, r), (int)g.row_bytes, lane);
        g5f_pf<1>(g5f_row(u, e, r), (int)u.row_bytes, lane);
    }
    const unsigned char* wg = g5f_stage(g, e, r, sm0, row16, lane);
    const unsigned char* wu = g5f_stage(u, e, r, sm1, row16, lane);
    g5f_stage_wait();
    float a, b;
    if (g.ty == u.ty) {
        const float2 p = g5f_dot_any2(g.ty, wg, wu, xqt, xq2t, xdt, xft, hid);
        a = g5f_warp_sum(p.x);
        b = g5f_warp_sum(p.y);
    } else {
        a = g5f_warp_sum(g5f_dot_any(g.ty, wg, xqt, xq2t, xdt, xft, hid));
        b = g5f_warp_sum(g5f_dot_any(u.ty, wu, xqt, xq2t, xdt, xft, hid));
    }
    if (lane == 0) {
        const float L = sh ? lims : lim;
        const float gg = fminf(a, L), uu = fminf(fmaxf(b, -L), L);
        hp[r] = gg / (1.0f + expf(-gg)) * uu;
    }
}

// y[t][c] = sum_{j < nr} wts[t][j] * (down_{ids[t][j]} row c . h[t*S + j]) (+ sdown row c .
// h[t*S + nr] if sdown.base != 0); with ids = null: one slot of weight 1 on matrix 0 (dense).
// Ids >= ne contribute nothing. grid (ceil(hid / (8 / sw)), T), block 128. sw (1, 2 or 4) warps
// share a row pair c0, c0+1: warp sub-index i takes slots i, i+sw, ... (two rows each, staged),
// then the group's first warp sums the sw partials in a fixed order (deterministic). sw = 4
// for the routed layers (9 slots: every expert's rows in flight at once), 1 for dense.
extern "C" __global__ void __launch_bounds__(128, G5F_MINB) g5f_down(
    const G5W down, const G5W sdown, const unsigned* ids, const float* wts, int k, int ne,
    const float* h, const unsigned char* hq, const unsigned char* hq2, const float* hd, int hs, float* y, int ff,
    int ffs, int hid,
    int row16, int sw) {
    __shared__ float red[4][2];
    extern __shared__ uint4 g5f_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.y;
    const int nr = ids ? k : 1, ns = sdown.base ? 1 : 0, S = nr + ns;
    uint4* sm0 = g5f_sm + (size_t)(warp * 2) * row16;
    uint4* sm1 = sm0 + row16;
    const int grp = warp / sw, sub = warp % sw;
    const int c0 = (blockIdx.x * (4 / sw) + grp) * 2;
    const bool live = c0 < hid, two = c0 + 1 < hid;
    float acc0 = 0.f, acc1 = 0.f;
    for (int j = sub; live && j < S; j += sw) {
        const bool sh = j >= nr;
        const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
        if (!sh && ids && e >= (unsigned)ne) continue;
        const float wj = sh ? 1.f : (ids ? wts[(size_t)t * k + j] : 1.f);
        const G5W w = sh ? sdown : down;
        const int kk = sh ? ffs : ff;
        const size_t row = (size_t)t * S + j;
        const unsigned char* q = hq + row * hs;
        const unsigned char* q2 = hq2 ? hq2 + row * hs : 0;
        const float* d = hd + row * (hs / 32);
        const float* f = h + row * hs;
        const unsigned char* w0 = g5f_stage(w, e, c0, sm0, row16, lane);
        const unsigned char* w1 = two ? g5f_stage(w, e, c0 + 1, sm1, row16, lane) : w0;
        g5f_stage_wait();
        const float2 p = g5f_dot_any2(w.ty, w0, two ? w1 : 0, q, q2, d, f, kk);
        const float a = g5f_warp_sum(p.x);
        const float b = two ? g5f_warp_sum(p.y) : 0.f;
        acc0 = fmaf(wj, a, acc0);
        acc1 = fmaf(wj, b, acc1);
        __syncwarp();
    }
    if (sw == 1) {
        if (live && lane == 0) {
            y[(size_t)t * hid + c0] = acc0;
            if (two) y[(size_t)t * hid + c0 + 1] = acc1;
        }
        return;
    }
    if (lane == 0) { red[warp][0] = acc0; red[warp][1] = acc1; }
    __syncthreads();
    if (live && sub == 0 && lane == 0) {
        float s0 = 0.f, s1 = 0.f;
        for (int i = 0; i < sw; i++) { s0 += red[warp + i][0]; s1 += red[warp + i][1]; }
        y[(size_t)t * hid + c0] = s0;
        if (two) y[(size_t)t * hid + c0 + 1] = s1;
    }
}

// The split down: yp[t*S + j][c] = w_j * (row c of slot j's down matrix . h[t*S + j]) — one warp
// per (slot, row pair), single-shot like g5f_gu, so every slot's rows stream at once instead of
// a warp walking 2-3 slots in turn; g5f_sum adds the S partials (fixed order). Sentinel ids
// write 0. grid (ceil(hid/8), S, T), block 128, staging as g5f_down.
extern "C" __global__ void __launch_bounds__(128, G5F_MINB) g5f_down_split(
    const G5W down, const G5W sdown, const unsigned* ids, const float* wts, int k, int ne,
    const float* h, const unsigned char* hq, const unsigned char* hq2, const float* hd, int hs, float* yp, int ff,
    int ffs, int hid, int row16) {
    extern __shared__ uint4 g5f_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = blockIdx.y, t = blockIdx.z;
    const int nr = ids ? k : 1, ns = sdown.base ? 1 : 0, S = nr + ns;
    const int c0 = (blockIdx.x * 4 + warp) * 2;
    if (c0 >= hid) return;
    const bool two = c0 + 1 < hid, sh = j >= nr;
    const size_t row = (size_t)t * S + j;
    float* out = yp + row * hid;
    const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
    if (!sh && ids && e >= (unsigned)ne) {
        if (lane == 0) { out[c0] = 0.f; if (two) out[c0 + 1] = 0.f; }
        return;
    }
    const float wj = sh ? 1.f : (ids ? wts[(size_t)t * k + j] : 1.f);
    const G5W w = sh ? sdown : down;
    const int kk = sh ? ffs : ff;
    uint4* sm0 = g5f_sm + (size_t)(warp * 2) * row16;
    uint4* sm1 = sm0 + row16;
    const unsigned char* q = hq + row * hs;
    const unsigned char* q2 = hq2 ? hq2 + row * hs : 0;
    const float* d = hd + row * (hs / 32);
    const float* f = h + row * hs;
    if (row16 == 0) g5f_pf<2>(g5f_row(w, e, c0), (int)w.row_bytes * (two ? 2 : 1), lane);
    const unsigned char* w0 = g5f_stage(w, e, c0, sm0, row16, lane);
    const unsigned char* w1 = two ? g5f_stage(w, e, c0 + 1, sm1, row16, lane) : w0;
    g5f_stage_wait();
    const float2 p = g5f_dot_any2(w.ty, w0, two ? w1 : 0, q, q2, d, f, kk);
    const float a = g5f_warp_sum(p.x);
    const float b = two ? g5f_warp_sum(p.y) : 0.f;
    if (lane == 0) {
        out[c0] = wj * a;
        if (two) out[c0 + 1] = wj * b;
    }
}

// y[t][c] = sum_{j < S} yp[t*S + j][c], j ascending. grid (ceil(hid/128), T), block 128.
extern "C" __global__ void g5f_sum(const float* yp, float* y, int S, int hid) {
    const int c = blockIdx.x * 128 + threadIdx.x, t = blockIdx.y;
    if (c >= hid) return;
    const float* p = yp + (size_t)t * S * hid + c;
    float acc = 0.f;
    for (int j = 0; j < S; j++) acc += p[(size_t)j * hid];
    y[(size_t)t * hid + c] = acc;
}

// The plain GEMV on the same row kernel: y[t * ystride + r] = (row r of matrix e of w) . x_t for
// r < rows, t < T (k = row length, k % 8 == 0). Any G5W type (g5f_dot_any). Activations, per token
// at stride xs elements (xs >= k, xs % 32 == 0): xq / xq2 / xd = g5f_q8 of x (xq2 null = one-level
// codes; k5_q8's codes + scales also work: xq2 null, xd = its scales, xs = COLS); xf = x f32 (read
// for F32 / F16 rows only, may be null otherwise). xd's per-token stride is xs / 32.
// grid (ceil(rows/8), ceil(T/G5F_TT)), block 128: two rows a warp, staged once (row16 > 0; dynamic
// smem 8 * row16 * 16 B as g5f_down) and dotted against up to G5F_TT tokens, so a prefill window
// reads W once per G5F_TT tokens instead of once per token.
#define G5F_TT 8
extern "C" __global__ void __launch_bounds__(128) g5f_gemv(
    const G5W w, unsigned e, const float* xf, const unsigned char* xq, const unsigned char* xq2, const float* xd,
    int xs, float* y, int ystride, int T, int rows, int k, int row16) {
    extern __shared__ uint4 g5f_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * 4 + warp) * 2;
    if (r0 >= rows) return;  // whole warp: no block-level sync below
    const bool two = r0 + 1 < rows;
    uint4* sm0 = g5f_sm + (size_t)(warp * 2) * row16;
    const unsigned char* w0 = g5f_stage(w, e, r0, sm0, row16, lane);
    const unsigned char* w1 = two ? g5f_stage(w, e, r0 + 1, sm0 + row16, row16, lane) : w0;
    g5f_stage_wait();
    const int t0 = blockIdx.y * G5F_TT;
    const int t1 = t0 + G5F_TT < T ? t0 + G5F_TT : T;
    for (int t = t0; t < t1; t++) {
        const size_t o = (size_t)t * xs;
        const unsigned char* q = xq ? xq + o : 0;
        const unsigned char* q2 = xq2 ? xq2 + o : 0;
        const float* d = xd ? xd + o / 32 : 0;
        const float* f = xf ? xf + o : 0;
        const float a = g5f_warp_sum(g5f_dot_any(w.ty, w0, q, q2, d, f, k));
        const float b = two ? g5f_warp_sum(g5f_dot_any(w.ty, w1, q, q2, d, f, k)) : 0.f;
        if (lane == 0) {
            y[(size_t)t * ystride + r0] = a;
            if (two) y[(size_t)t * ystride + r0 + 1] = b;
        }
    }
}


// ---- N rows a warp (TANG_GLM5_MOE_ROWS): more weight bytes in flight per lane -------------------
// Each row's lane partial is summed exactly as g5f_lane_dot32x / g5f_dot_any (chunks c = lane +
// 32 i ascending, one accumulator per row), so outputs are bit-identical to the 1-2 row kernels;
// only the number of independent row streams per iteration grows (all N rows' loads issue before
// any use). Non-32-chunk types (and misaligned Q4_K / Q5_K) take g5f_dot_any per row.
// Chunk c's contribution with the activations already in registers (x: codes, y: residual codes
// or unused when !has2, dx = xd[c]) and the block header / scales read as whole words: the same
// values and the same float ops as g5f_c32k / g5f_c32r, ~4x fewer load instructions per chunk.
__device__ __forceinline__ unsigned g5f_byte(unsigned w, int i) { return (w >> (8 * i)) & 0xFFu; }
template <int TY>
__device__ __forceinline__ float g5f_c32v(const unsigned char* w, int c, int k, const unsigned* xw, const unsigned* yw,
                                         bool has2, float dx, float fsx) {
    unsigned n[8];
    float d0, d1, mw = 0.f;
    if (TY == 12 || TY == 13) {
        const int bb = TY == 12 ? 144 : 176;
        const unsigned char* b = w + (c >> 3) * bb;
        const int j = c & 7, up = j & 1;
        const uint4 hd = *(const uint4*)b;  // d, dmin | scales[12]
        int s, m;
        if (j < 4) {
            s = (int)(g5f_byte(hd.y, j) & 63);
            m = (int)(g5f_byte(hd.z, j) & 63);
        } else {
            const unsigned q4 = g5f_byte(hd.w, j - 4), qm4 = g5f_byte(hd.y, j - 4), q0 = g5f_byte(hd.z, j - 4);
            s = (int)((q4 & 0xF) | ((qm4 >> 6) << 4));
            m = (int)((q4 >> 4) | ((q0 >> 6) << 4));
        }
        d0 = g5f_h2f((unsigned short)(hd.x & 0xFFFFu)) * (float)s;
        mw = g5f_h2f((unsigned short)(hd.x >> 16)) * (float)m;
        const uint4* qp = (const uint4*)(b + (TY == 12 ? 16 : 48) + (j >> 1) * 32);
        const uint4 q0 = qp[0], q1 = qp[1];
        n[0] = q0.x; n[1] = q0.y; n[2] = q0.z; n[3] = q0.w; n[4] = q1.x; n[5] = q1.y; n[6] = q1.z; n[7] = q1.w;
        #pragma unroll
        for (int i = 0; i < 8; i++) n[i] = (n[i] >> (4 * up)) & 0x0F0F0F0Fu;
        if (TY == 13) {
            const uint4* hp = (const uint4*)(b + 16);
            const uint4 h0 = hp[0], h1 = hp[1];
            const unsigned hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            #pragma unroll
            for (int i = 0; i < 8; i++) n[i] |= ((hw[i] >> j) & 0x01010101u) << 4;
        }
        int sf = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) sf = __dp4a((int)n[i], (int)xw[i], sf);
        float fsf = (float)sf;
        if (has2) {
            int sf2 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) sf2 = __dp4a((int)n[i], (int)yw[i], sf2);
            fsf = fmaf((float)sf2, 1.0f / 254.0f, fsf);
        }
        return dx * fmaf(d0, fsf, -mw * fsx);
    } else {
        const int nb = k / 256;
        if (TY == 108) {
            const uint4* qp = (const uint4*)(w + (size_t)c * 32);
            const uint4 q0 = qp[0], q1 = qp[1];
            n[0] = q0.x; n[1] = q0.y; n[2] = q0.z; n[3] = q0.w; n[4] = q1.x; n[5] = q1.y; n[6] = q1.z; n[7] = q1.w;
            d0 = d1 = g5f_h2f(*(const unsigned short*)(w + k + 2 * c));
        } else {
            const int bi = c >> 3, j = c & 7, hn = j >> 2, qt = j & 3;
            const unsigned char* b = w + (size_t)bi * 192;
            const uint4* lp = (const uint4*)(b + hn * 64 + 32 * (qt & 1));
            const uint4* hp = (const uint4*)(b + 128 + hn * 32);
            const uint4 l0 = lp[0], l1 = lp[1], h0 = hp[0], h1 = hp[1];
            const unsigned lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
            const unsigned hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            const int ls = 4 * (qt >> 1), hs = 2 * qt;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                n[i] = __vsub4(((lw[i] >> ls) & 0x0F0F0F0Fu) | (((hw[i] >> hs) & 0x03030303u) << 4), 0x20202020u);
            const float d = g5f_h2f(*(const unsigned short*)(w + nb * 208 + 2 * bi));
            const unsigned short sc2 = *(const unsigned short*)(w + nb * 192 + bi * 16 + hn * 8 + 2 * qt);
            d0 = d * (float)(int)(signed char)(sc2 & 0xFF);
            d1 = d * (float)(int)(signed char)(sc2 >> 8);
        }
        int a0 = 0, a1 = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) a0 = __dp4a((int)n[i], (int)xw[i], a0);
        #pragma unroll
        for (int i = 4; i < 8; i++) a1 = __dp4a((int)n[i], (int)xw[i], a1);
        float f0 = (float)a0, f1 = (float)a1;
        if (has2) {
            int b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) b0 = __dp4a((int)n[i], (int)yw[i], b0);
            #pragma unroll
            for (int i = 4; i < 8; i++) b1 = __dp4a((int)n[i], (int)yw[i], b1);
            f0 = fmaf((float)b0, 1.0f / 254.0f, f0);
            f1 = fmaf((float)b1, 1.0f / 254.0f, f1);
        }
        return dx * fmaf(d0, f0, d1 * f1);
    }
}
// chunk c's contribution of each of N rows (activations loaded once for all rows)
template <int TY, int N>
__device__ __forceinline__ void g5f_chunkn(const unsigned char* const* w, int c, int k, const unsigned char* xq,
                                          const unsigned char* xq2, const float* xd, float* p) {
    const uint4* xp = (const uint4*)(xq + (size_t)c * 32);
    const uint4 x0 = xp[0], x1 = xp[1];
    const unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    unsigned yw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (xq2) {
        const uint4* yp = (const uint4*)(xq2 + (size_t)c * 32);
        const uint4 y0 = yp[0], y1 = yp[1];
        yw[0] = y0.x; yw[1] = y0.y; yw[2] = y0.z; yw[3] = y0.w; yw[4] = y1.x; yw[5] = y1.y; yw[6] = y1.z; yw[7] = y1.w;
    }
    const float dx = xd[c];
    // sum of the chunk's activation codes (K-quant min term): row-independent, so once per
    // chunk for all N rows (same ints, same float ops as computing it per row)
    float fsx = 0.f;
    if (TY == 12 || TY == 13) {
        int sx = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) sx = __dp4a(0x01010101, (int)xw[i], sx);
        fsx = (float)sx;
        if (xq2) {
            int sx2 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) sx2 = __dp4a(0x01010101, (int)yw[i], sx2);
            fsx = fmaf((float)sx2, 1.0f / 254.0f, fsx);
        }
    }
    #pragma unroll
    for (int i = 0; i < N; i++) p[i] = g5f_c32v<TY>(w[i], c, k, xw, yw, xq2 != 0, dx, fsx);
}
template <int TY, int N>
__device__ __forceinline__ void g5f_lane_dot32n(const unsigned char* const* w, const unsigned char* xq,
                                               const unsigned char* xq2, const float* xd, int k, float* acc) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < N; i++) acc[i] = 0.f;
    #pragma unroll G5F_UNROLL
    for (int c = lane; c < k / 32; c += 32) {
        float p[N];
        g5f_chunkn<TY, N>(w, c, k, xq, xq2, xd, p);
        #pragma unroll
        for (int i = 0; i < N; i++) acc[i] += p[i];
    }
}
template <int N>
__device__ __forceinline__ void g5f_dot_anyn(unsigned ty, const unsigned char* const* w, const unsigned char* xq,
                                            const unsigned char* xq2, const float* xd, const float* xf, int k,
                                            float* acc) {
    unsigned long long o = 0;
    #pragma unroll
    for (int i = 0; i < N; i++) o |= (unsigned long long)w[i];
    const bool al = (o & 15) == 0 && (k & 255) == 0 && G5F_DOT32;
    if (ty == 12 && al) { g5f_lane_dot32n<12, N>(w, xq, xq2, xd, k, acc); return; }
    if (ty == 13 && al) { g5f_lane_dot32n<13, N>(w, xq, xq2, xd, k, acc); return; }
    if (ty == 108) { g5f_lane_dot32n<108, N>(w, xq, xq2, xd, k, acc); return; }
    if (ty == 114) { g5f_lane_dot32n<114, N>(w, xq, xq2, xd, k, acc); return; }
    #pragma unroll
    for (int i = 0; i < N; i++) acc[i] = g5f_dot_any<G5F_LDU>(ty, w[i], xq, xq2, xd, xf, k);
}

// g5f_gu with G5F_GUR rows a warp (gate and up of each: 2 * G5F_GUR streams). grid
// (ceil(max(ff, ffs) / (4 * G5F_GUR)), S, T), block 128, no staging. Same h as g5f_gu.
#ifndef G5F_GUR
#define G5F_GUR 2
#endif
extern "C" __global__ void __launch_bounds__(128, G5F_MINB) g5f_gun(
    const G5W gate, const G5W up, const G5W sgate, const G5W sup, const unsigned* ids, int k, int ne,
    const float* x, const unsigned char* xq, const unsigned char* xq2, const float* xd, float* h, int hs, int ff,
    int ffs, int hid, float lim, float lims) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = blockIdx.y, t = blockIdx.z;
    const int nr = ids ? k : 1;
    const bool sh = j >= nr;
    const int rows = sh ? ffs : ff;
    const int r0 = (blockIdx.x * 4 + warp) * G5F_GUR;
    if (r0 >= rows) return;
    const int S = nr + (sgate.base ? 1 : 0);
    float* hp = h + (size_t)(t * S + j) * hs;
    const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
    if (!sh && ids && e >= (unsigned)ne) {
        if (lane < G5F_GUR && r0 + lane < rows) hp[r0 + lane] = 0.f;
        return;
    }
    const unsigned char* xqt = xq + (size_t)t * hid;
    const unsigned char* xq2t = xq2 ? xq2 + (size_t)t * hid : 0;
    const float* xdt = xd + (size_t)t * (hid / 32);
    const float* xft = x + (size_t)t * hid;
    const G5W g = sh ? sgate : gate, u = sh ? sup : up;
    float a[G5F_GUR], b[G5F_GUR];
    if (g.ty == u.ty) {
        const unsigned char* w[2 * G5F_GUR];
        #pragma unroll
        for (int i = 0; i < G5F_GUR; i++) {
            const int r = r0 + i < rows ? r0 + i : r0;
            w[2 * i] = g5f_row(g, e, r);
            w[2 * i + 1] = g5f_row(u, e, r);
        }
        #pragma unroll
        for (int i = 0; i < 2 * G5F_GUR; i++) g5f_pf<1>(w[i], (int)g.row_bytes, lane);
        float p[2 * G5F_GUR];
        g5f_dot_anyn<2 * G5F_GUR>(g.ty, w, xqt, xq2t, xdt, xft, hid, p);
        #pragma unroll
        for (int i = 0; i < G5F_GUR; i++) { a[i] = g5f_warp_sum(p[2 * i]); b[i] = g5f_warp_sum(p[2 * i + 1]); }
    } else {
        const unsigned char* wg[G5F_GUR];
        const unsigned char* wu[G5F_GUR];
        #pragma unroll
        for (int i = 0; i < G5F_GUR; i++) {
            const int r = r0 + i < rows ? r0 + i : r0;
            wg[i] = g5f_row(g, e, r);
            wu[i] = g5f_row(u, e, r);
        }
        float pa[G5F_GUR], pb[G5F_GUR];
        g5f_dot_anyn<G5F_GUR>(g.ty, wg, xqt, xq2t, xdt, xft, hid, pa);
        g5f_dot_anyn<G5F_GUR>(u.ty, wu, xqt, xq2t, xdt, xft, hid, pb);
        #pragma unroll
        for (int i = 0; i < G5F_GUR; i++) { a[i] = g5f_warp_sum(pa[i]); b[i] = g5f_warp_sum(pb[i]); }
    }
    if (lane == 0) {
        const float L = sh ? lims : lim;
        #pragma unroll
        for (int i = 0; i < G5F_GUR; i++) {
            if (r0 + i >= rows) break;
            const float gg = fminf(a[i], L), uu = fminf(fmaxf(b[i], -L), L);
            hp[r0 + i] = gg / (1.0f + expf(-gg)) * uu;
        }
    }
}

// g5f_down_split with G5F_DNR rows a warp (G5F_DNR even). grid (ceil(hid / (4 * G5F_DNR)), S, T),
// block 128, no staging. Same yp as g5f_down_split (the pair-fused dot is the same per-row sum).
#ifndef G5F_DNR
#define G5F_DNR 4
#endif
extern "C" __global__ void __launch_bounds__(128, G5F_MINB) g5f_down_splitn(
    const G5W down, const G5W sdown, const unsigned* ids, const float* wts, int k, int ne,
    const float* h, const unsigned char* hq, const unsigned char* hq2, const float* hd, int hs, float* yp, int ff,
    int ffs, int hid) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.z;
    const int nr = ids ? k : 1, ns = sdown.base ? 1 : 0, S = nr + ns;
    // shared slot (Q8_0, slow 8-weight chain) as block row 0 so it is scheduled first, not as a tail
    const int j = ns ? (blockIdx.y == 0 ? nr : (int)blockIdx.y - 1) : (int)blockIdx.y;
    const int c0 = (blockIdx.x * 4 + warp) * G5F_DNR;
    if (c0 >= hid) return;
    const bool sh = j >= nr;
    const size_t row = (size_t)t * S + j;
    float* out = yp + row * hid;
    const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
    if (!sh && ids && e >= (unsigned)ne) {
        if (lane < G5F_DNR && c0 + lane < hid) out[c0 + lane] = 0.f;
        return;
    }
    const float wj = sh ? 1.f : (ids ? wts[(size_t)t * k + j] : 1.f);
    const G5W w = sh ? sdown : down;
    const int kk = sh ? ffs : ff;
    const unsigned char* q = hq + row * hs;
    const unsigned char* q2 = hq2 ? hq2 + row * hs : 0;
    const float* d = hd + row * (hs / 32);
    const float* f = h + row * hs;
    const unsigned char* wr[G5F_DNR];
    #pragma unroll
    for (int i = 0; i < G5F_DNR; i++) wr[i] = g5f_row(w, e, c0 + i < hid ? c0 + i : c0);
    g5f_pf<2>(wr[0], (int)w.row_bytes * (c0 + G5F_DNR <= hid ? G5F_DNR : hid - c0), lane);
    float p[G5F_DNR];
    g5f_dot_anyn<G5F_DNR>(w.ty, wr, q, q2, d, f, kk, p);
    float o[G5F_DNR];
    #pragma unroll
    for (int i = 0; i < G5F_DNR; i++) o[i] = g5f_warp_sum(p[i]);
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < G5F_DNR; i++)
            if (c0 + i < hid) out[c0 + i] = wj * o[i];
    }
}


// ---- round 4 (TANG_GLM5_MOE_K4=1, default off): activation-resident, lane-contiguous rows ------------
// NOT bit-identical to g5f_gux / g5f_down_splitn (the default): a thread
// sums 2 x 16 weights of one superblock per row and the row sum is a fixed shuffle + smem tree,
// so the float summation order differs (same ints, same per-chunk scale math up to association).
// Rows of Q4_K / Q5_K / Q6_Kr with k % 256 == 0, k <= 4096, 16-byte aligned views. TPR = 8 k/256
// threads a row (8 per superblock), 128 / TPR row slots a block. Thread p = tid % 8 of superblock
// sb owns positions pos = 16 (p & 1) .. +16 of two 32-chunks jl / jh: Q4_K / Q5_K jl = 2 (p >> 1),
// jh = jl + 1 (low / high nibbles of the same 16 qs bytes); Q6_Kr jl = 4 (p >> 2) + ((p >> 1) & 1),
// jh = jl + 2 (low / high nibbles of the same 16 ql bytes). Its activations (codes, residual
// codes, scales, K-quant min sums) stay in registers for every row; a row costs it one 16-byte
// weight load (+ the header / qh / scales, shared by the superblock's 8 threads).
#ifndef G5F_K4_R
#define G5F_K4_R 8
#endif
#ifndef G5F_K4_NR
#define G5F_K4_NR 4
#endif
#ifndef G5F_K4_DR
#define G5F_K4_DR 8
#endif
#ifndef G5F_K4_DNR
#define G5F_K4_DNR 4
#endif
// min resident blocks per SM of g5f_gu4 / g5f_down4 (4 -> <= 128 registers)
#ifndef G5F_K4_MINB
#define G5F_K4_MINB 4
#endif
struct G5fAct {
    uint4 xl, xh, yl, yh;
    // dl / dh: chunk scales (/ 254 with residual codes: the int dot is 254 x . n + y . n);
    // al / ah: dl / dh x the half-chunk code sum (K-quant min term, same units)
    float dl, dh, al, ah;
    // Q4_K / Q5_K scale bytes of chunks jl, jl + 1 (g5f_smk4 for both at once, branch-free):
    // s2 = ((lo ? y : w) >> sh & ma) | (y >> (sh + 2) & mb), m2 likewise from z / (w >> 4)
    unsigned sh, ma, mb;
    bool lo;
};
struct G5fRaw {
    uint4 q, h, s;  // qs / ql 16 B; Q4_K/Q5_K: header (d, dmin, scales) | Q6_Kr: qh 16 B; Q5_K: qh 16 B (s)
    unsigned sc;    // Q6_Kr: the 2 scale bytes (jl, jh) | d
};
__device__ __forceinline__ int g5f_dp4(const uint4 a, const uint4 b, int c) {
    c = __dp4a((int)a.x, (int)b.x, c);
    c = __dp4a((int)a.y, (int)b.y, c);
    c = __dp4a((int)a.z, (int)b.z, c);
    return __dp4a((int)a.w, (int)b.w, c);
}
template <int TY>
__device__ __forceinline__ void g5f_k4_chunks(int p, int& jl, int& jh) {
    if (TY == 114) { jl = 4 * (p >> 2) + ((p >> 1) & 1); jh = jl + 2; }
    else { jl = 2 * (p >> 1); jh = jl + 1; }
}
template <int TY, bool H2>
__device__ __forceinline__ G5fAct g5f_k4_act(const unsigned char* xq, const unsigned char* xq2, const float* xd,
                                             int sb, int p) {
    int jl, jh;
    g5f_k4_chunks<TY>(p, jl, jh);
    const int pos = 16 * (p & 1), cl = sb * 8 + jl, ch = sb * 8 + jh;
    G5fAct a;
    a.xl = *(const uint4*)(xq + cl * 32 + pos);
    a.xh = *(const uint4*)(xq + ch * 32 + pos);
    if (H2) {
        a.yl = *(const uint4*)(xq2 + cl * 32 + pos);
        a.yh = *(const uint4*)(xq2 + ch * 32 + pos);
    } else {
        a.yl = a.yh = make_uint4(0, 0, 0, 0);
    }
    const float u = H2 ? 1.0f / 254.0f : 1.0f;
    a.dl = xd[cl] * u;
    a.dh = xd[ch] * u;
    a.al = a.ah = 0.f;
    a.lo = jl < 4;
    a.sh = 8 * (jl & 3);
    a.ma = a.lo ? 0x3F3Fu : 0x0F0Fu;
    a.mb = a.lo ? 0u : 0x3030u;
    if (TY == 12 || TY == 13) {
        const uint4 one = make_uint4(0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u);
        int sl = g5f_dp4(one, a.xl, 0), shh = g5f_dp4(one, a.xh, 0);
        if (H2) {
            sl = g5f_dp4(one, a.yl, sl * 254);
            shh = g5f_dp4(one, a.yh, shh * 254);
        }
        a.al = a.dl * (float)sl;
        a.ah = a.dh * (float)shh;
    }
    return a;
}
// row w's bytes for this thread (superblock sb of a k-wide row, nb = k / 256 superblocks)
template <int TY>
__device__ __forceinline__ G5fRaw g5f_k4_load(const unsigned char* w, int sb, int p, int nb) {
    G5fRaw r;
    const int pos = 16 * (p & 1);
    if (TY == 12 || TY == 13) {
        const int bb = TY == 12 ? 144 : 176;
        const unsigned char* b = w + sb * bb;
        r.h = *(const uint4*)b;
        r.q = *(const uint4*)(b + (TY == 12 ? 16 : 48) + (p >> 1) * 32 + pos);
        if (TY == 13) r.s = *(const uint4*)(b + 16 + pos);
        r.sc = 0;
    } else if (TY == 108) {  // Q8_0r: codes, then the k / 32 halves
        const int c = sb * 8 + 2 * (p >> 1);
        r.q = *(const uint4*)(w + c * 32 + pos);
        r.h = *(const uint4*)(w + (c + 1) * 32 + pos);
        r.sc = *(const unsigned*)(w + nb * 256 + 2 * c);
    } else {  // 114: Q6_Kr
        const int hn = p >> 2, a = (p >> 1) & 1;
        const unsigned char* b = w + sb * 192;
        r.q = *(const uint4*)(b + hn * 64 + 32 * a + pos);
        r.h = *(const uint4*)(b + 128 + hn * 32 + pos);
        const unsigned long long s8 = *(const unsigned long long*)(w + nb * 192 + sb * 16 + hn * 8);
        const int o = 2 * a + (p & 1);  // scale bytes o (jl) and o + 4 (jh)
        const unsigned lo = (unsigned)(s8 >> (8 * o)) & 0xFFu, hi = (unsigned)(s8 >> (8 * o + 32)) & 0xFFu;
        r.sc = lo | (hi << 8) | ((unsigned)*(const unsigned short*)(w + nb * 208 + 2 * sb) << 16);
    }
    return r;
}
__device__ __forceinline__ uint4 g5f_u4op(uint4 a, unsigned sh, unsigned m) {
    return make_uint4((a.x >> sh) & m, (a.y >> sh) & m, (a.z >> sh) & m, (a.w >> sh) & m);
}
// row r's share of this thread: d (sl dl (254 x.n + y.n)_l + sh dh (...)_h) - dmin (ml al + mh ah)
// (Q6_Kr: no min term, signed scales). Integer dots and scale products, 4 / 2 int -> float.
template <int TY, bool H2>
__device__ __forceinline__ float g5f_k4_dot(const G5fRaw& r, const G5fAct& a, int p) {
    uint4 nl, nh;
    if (TY == 12 || TY == 13) {
        nl = g5f_u4op(r.q, 0, 0x0F0F0F0Fu);
        nh = g5f_u4op(r.q, 4, 0x0F0F0F0Fu);
        if (TY == 13) {
            const int jl = 2 * (p >> 1);
            const uint4 hl = g5f_u4op(r.s, jl, 0x01010101u), hh = g5f_u4op(r.s, jl + 1, 0x01010101u);
            nl.x |= hl.x << 4; nl.y |= hl.y << 4; nl.z |= hl.z << 4; nl.w |= hl.w << 4;
            nh.x |= hh.x << 4; nh.y |= hh.y << 4; nh.z |= hh.z << 4; nh.w |= hh.w << 4;
        }
        const unsigned A = a.lo ? r.h.y : r.h.w, B = a.lo ? r.h.z : (r.h.w >> 4);
        const unsigned s2 = ((A >> a.sh) & a.ma) | ((r.h.y >> (a.sh + 2)) & a.mb);
        const unsigned m2 = ((B >> a.sh) & a.ma) | ((r.h.z >> (a.sh + 2)) & a.mb);
        int il = g5f_dp4(nl, a.xl, 0), ih = g5f_dp4(nh, a.xh, 0);
        if (H2) {
            il = g5f_dp4(nl, a.yl, il * 254);
            ih = g5f_dp4(nh, a.yh, ih * 254);
        }
        const float d = g5f_h2f((unsigned short)(r.h.x & 0xFFFFu)), dm = g5f_h2f((unsigned short)(r.h.x >> 16));
        const float fs = fmaf(a.dl, (float)(il * (int)(s2 & 0xFFu)), a.dh * (float)(ih * (int)(s2 >> 8)));
        const float fm = fmaf(a.al, (float)(m2 & 0xFFu), a.ah * (float)(m2 >> 8));
        return fmaf(d, fs, -dm * fm);
    } else if (TY == 108) {
        int il = g5f_dp4(r.q, a.xl, 0), ih = g5f_dp4(r.h, a.xh, 0);
        if (H2) {
            il = g5f_dp4(r.q, a.yl, il * 254);
            ih = g5f_dp4(r.h, a.yh, ih * 254);
        }
        const float dl = g5f_h2f((unsigned short)(r.sc & 0xFFFFu)), dh = g5f_h2f((unsigned short)(r.sc >> 16));
        return fmaf(a.dl * dl, (float)il, a.dh * dh * (float)ih);
    } else {
        const int hs = 2 * ((p >> 1) & 1);
        const uint4 ql = g5f_u4op(r.q, 0, 0x0F0F0F0Fu), qh = g5f_u4op(r.q, 4, 0x0F0F0F0Fu);
        const uint4 bl = g5f_u4op(r.h, hs, 0x03030303u), bh = g5f_u4op(r.h, hs + 4, 0x03030303u);
        nl = make_uint4(__vsub4(ql.x | (bl.x << 4), 0x20202020u), __vsub4(ql.y | (bl.y << 4), 0x20202020u),
                        __vsub4(ql.z | (bl.z << 4), 0x20202020u), __vsub4(ql.w | (bl.w << 4), 0x20202020u));
        nh = make_uint4(__vsub4(qh.x | (bh.x << 4), 0x20202020u), __vsub4(qh.y | (bh.y << 4), 0x20202020u),
                        __vsub4(qh.z | (bh.z << 4), 0x20202020u), __vsub4(qh.w | (bh.w << 4), 0x20202020u));
        int il = g5f_dp4(nl, a.xl, 0), ih = g5f_dp4(nh, a.xh, 0);
        if (H2) {
            il = g5f_dp4(nl, a.yl, il * 254);
            ih = g5f_dp4(nh, a.yh, ih * 254);
        }
        const int sl = (int)(signed char)(r.sc & 0xFF), shh = (int)(signed char)((r.sc >> 8) & 0xFF);
        const float d = g5f_h2f((unsigned short)(r.sc >> 16));
        return d * fmaf(a.dl, (float)(il * sl), a.dh * (float)(ih * shh));
    }
}
// Sum V values over the TPR threads of each row slot: lanes swap halves of their value set
// (offsets min(16, TPR / 2) down, V - 1 shuffles), then plain butterflies; a warp's partial of
// value i lands in buf[(slot * WPS + warp-in-slot) * V + i]. Caller syncs, then g5f_k4_total.
template <int V>
__device__ __forceinline__ void g5f_k4_reduce(float* v, int tpr, float* buf) {
    const int lane = threadIdx.x & 31;
    int o = tpr < 32 ? tpr >> 1 : 16, idx = 0;
    #pragma unroll
    for (int n = V; n > 1; n >>= 1) {
        const bool up = (lane & o) != 0;
        #pragma unroll
        for (int i = 0; i < n / 2; i++) {
            const float send = up ? v[i] : v[i + n / 2], keep = up ? v[i + n / 2] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
        idx = 2 * idx + (up ? 1 : 0);
        o >>= 1;
    }
    float r = v[0];
    for (int q = o; q > 0; q >>= 1) r += __shfl_xor_sync(0xffffffffu, r, q);
    // writer: the lowest lane of each idx group (o == 0: every lane holds its own idx)
    const int low = o > 0 ? (lane & (2 * o - 1)) : 0;
    if (low == 0) {
        const int wps = tpr < 32 ? 1 : tpr >> 5;
        const int slot = threadIdx.x / tpr, wis = (threadIdx.x % tpr) >> 5;
        buf[(slot * wps + wis) * V + idx] = r;
    }
}
template <int V>
__device__ __forceinline__ float g5f_k4_total(const float* buf, int tpr, int slot, int i) {
    const int wps = tpr < 32 ? 1 : tpr >> 5;
    float s = buf[(slot * wps) * V + i];
    for (int w = 1; w < wps; w++) s += buf[(slot * wps + w) * V + i];
    return s;
}

// ---- gate|up, shared slot first (TANG_GLM5_MOE_GUX, default 8) -------------------------------
// g5f_gun's row dot with the block order changed: the shared expert (Q8_0 here, on the 8-weight
// fused-multiply chain of g5f_lane_dot) was block row y = S - 1, so its slow blocks were scheduled
// last and ran as a tail (microbench: +17 us on a 25 us routed gate|up). Here it is y = 0, routed
// slot j at y = j + 1, and its gate and up rows are separate warps (two chains in flight per row).
// Rows of a 32-chunk type: warp w on rows r0 + w, r0 + w + 4, ... as g5f_gun. Same per-row math,
// so h is bit-identical. grid (ceil(max(ff, ffs) / R), S, T), block 128, R = G5F_GUX_R.
#ifndef G5F_GUX_R
#define G5F_GUX_R 4
#endif
extern "C" __global__ void __launch_bounds__(128, G5F_MINB) g5f_gux(
    const G5W gate, const G5W up, const G5W sgate, const G5W sup, const unsigned* ids, int k, int ne,
    const float* x, const unsigned char* xq, const unsigned char* xq2, const float* xd, float* h, int hs, int ff,
    int ffs, int hid, float lim, float lims) {
    __shared__ float part[2 * G5F_GUX_R];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.z;
    const int nr = ids ? k : 1, ns = sgate.base ? 1 : 0, S = nr + ns;
    const int j = ns ? (blockIdx.y == 0 ? nr : (int)blockIdx.y - 1) : (int)blockIdx.y;
    const bool sh = j >= nr;
    const int rows = sh ? ffs : ff;
    const int r0 = blockIdx.x * G5F_GUX_R;
    if (r0 >= rows) return;
    float* hp = h + (size_t)(t * S + j) * hs;
    const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
    if (!sh && ids && e >= (unsigned)ne) {
        if (threadIdx.x < G5F_GUX_R && r0 + (int)threadIdx.x < rows) hp[r0 + threadIdx.x] = 0.f;
        return;
    }
    const unsigned char* xqt = xq + (size_t)t * hid;
    const unsigned char* xq2t = xq2 ? xq2 + (size_t)t * hid : 0;
    const float* xdt = xd + (size_t)t * (hid / 32);
    const float* xft = x + (size_t)t * hid;
    const G5W g = sh ? sgate : gate, u = sh ? sup : up;
    const bool al = G5F_DOT32 && ((g.base | u.base | g.estride | u.estride | g.row_bytes | u.row_bytes) & 15) == 0 &&
                    (hid & 255) == 0;
    const int ty = g.ty == u.ty ? (int)g.ty : -1;
    const bool chunk = ((ty == 12 || ty == 13) && al) || ty == 108 || ty == 114;
    const float L = sh ? lims : lim;
    if (chunk) {
        // the g5f_gun row dot, warp w on rows w, w + 4, ...: lane partials go chunk by chunk into
        // one accumulator (nvcc contracts dx * (...) + acc into an fma, so the chunk loop cannot be
        // split across warps without changing h)
        for (int rr = warp; rr < G5F_GUX_R; rr += 4) {
            if (r0 + rr >= rows) break;
            const unsigned char* w[2] = {g5f_row(g, e, r0 + rr), g5f_row(u, e, r0 + rr)};
            float p[2];
            g5f_dot_anyn<2>(g.ty, w, xqt, xq2t, xdt, xft, hid, p);
            const float a = g5f_warp_sum(p[0]), b = g5f_warp_sum(p[1]);
            if (lane == 0) {
                const float gg = fminf(a, L), uu = fminf(fmaxf(b, -L), L);
                hp[r0 + rr] = gg / (1.0f + expf(-gg)) * uu;
            }
        }
        return;
    }
    // one warp per (row, gate | up): the g5f_dot_any per row of g5f_gun, two chains in flight per row
    float* ab = part;
    for (int q = warp; q < 2 * G5F_GUX_R; q += 4) {
        const int rr = q >> 1;
        if (r0 + rr >= rows) break;
        const G5W v = (q & 1) ? u : g;
        const float sres = g5f_warp_sum(g5f_dot_any<G5F_LDU>(v.ty, g5f_row(v, e, r0 + rr), xqt, xq2t, xdt, xft, hid));
        if (lane == 0) ab[q] = sres;
    }
    __syncthreads();
    if (threadIdx.x < G5F_GUX_R && r0 + (int)threadIdx.x < rows) {
        const float a = ab[2 * threadIdx.x], b = ab[2 * threadIdx.x + 1];
        const float gg = fminf(a, L), uu = fminf(fmaxf(b, -L), L);
        hp[r0 + threadIdx.x] = gg / (1.0f + expf(-gg)) * uu;
    }
}

// ---- round 4 kernels (TANG_GLM5_MOE_K4) ----------------------------------------------------------
// q8r: allow Q8_0r rows (down only: on gate|up the shared Q8_0r rows would size the ring for 4352 B rows)
__device__ __forceinline__ bool g5f_k4_ok(const G5W w, int k, bool q8r = true) {
    const int nb = k >> 8;
    return ((w.base | w.estride | w.row_bytes) & 15) == 0 && (k & 255) == 0 && nb >= 1 && nb <= 16 &&
           (nb & (nb - 1)) == 0 && (w.ty == 12 || w.ty == 13 || (q8r && w.ty == 108) || w.ty == 114);
}
// G5F_K4_ST = 0: no staging, each thread loads its 16-byte pieces from global (NR rows in flight)
#ifndef G5F_K4_ST
#define G5F_K4_ST 1
#endif
// G5F_K4_NS stages of shared memory in flight per block (cp.async ring)
#ifndef G5F_K4_NS
#define G5F_K4_NS 3
#endif
__device__ __forceinline__ void g5f_cp16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
#else
    *(uint4*)dst = __ldg((const uint4*)src);
#endif
}
__device__ __forceinline__ void g5f_cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void g5f_cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif
}
// dynamic shared memory g5f_gu4 / g5f_down4 need: NS stages x M matrices x (rs x NR) rows
__host__ __device__ __forceinline__ unsigned g5f_k4_smem(unsigned row_bytes, int k, int m, int nr) {
    return G5F_K4_ST ? (unsigned)G5F_K4_NS * m * (4096 / (k > 0 ? k : 4096)) * nr * row_bytes : 0u;
}
// this block's rows r0 .. r0 + R - 1 of matrices m[0..M) (M = 2: gate, up; 1: down): row slot s
// takes rows r0 + s + rs i. A batch = rs x NR consecutive rows (contiguous bytes per matrix),
// copied into a G5F_K4_NS-deep shared ring by the whole block (cp.async, 16 B a thread a step)
// while earlier batches are reduced. fin(row, sums) gets the M sums of each valid row.
template <int TY, bool H2, int M, int R, int NR, typename F>
__device__ __forceinline__ void g5f_k4_rows2(const G5W* m, unsigned e, int r0, int rows, int k, const unsigned char* xq,
                                            const unsigned char* xq2, const float* xd, float* buf, F fin) {
    extern __shared__ uint4 g5f_k4sm[];
    constexpr int NS = G5F_K4_NS;
    const int nb = k >> 8, tpr = 8 * nb, rs = 128 / tpr, br = rs * NR;
    const int slot = threadIdx.x / tpr, tt = threadIdx.x % tpr, sb = tt >> 3, p = tt & 7;
    const int rb = (int)m[0].row_bytes, rb16 = rb >> 4, stm = br * rb16;  // uint4 per (stage, matrix)
    const int nbat = (min(R, rows - r0) + br - 1) / br;
    auto issue = [&](int b) {
        if (b < nbat) {
            const int q0 = b * br, n = min(br, min(R, rows - r0) - q0);
            uint4* st = g5f_k4sm + (size_t)(b % NS) * M * stm;
            #pragma unroll
            for (int j = 0; j < M; j++) {
                const uint4* src = (const uint4*)g5f_row(m[j], e, r0 + q0);
                for (int i = threadIdx.x; i < n * rb16; i += 128) g5f_cp16(st + j * stm + i, src + i);
            }
        }
        g5f_cp_commit();
    };
    if (G5F_K4_ST) {
        #pragma unroll
        for (int b = 0; b < NS - 1; b++) issue(b);
    }
    const G5fAct a = g5f_k4_act<TY, H2>(xq, xq2, xd, sb, p);
    for (int b = 0; b < nbat; b++) {
        if (G5F_K4_ST) {
            issue(b + NS - 1);
            g5f_cp_wait<NS - 1>();
            __syncthreads();
        }
        const unsigned char* st = (const unsigned char*)(g5f_k4sm + (size_t)(b % NS) * M * stm);
        float v[M * NR];
        #pragma unroll
        for (int i = 0; i < NR; i++) {
            #pragma unroll
            for (int j = 0; j < M; j++)
            {
                const int q_ = b * br + slot + rs * i;
                const unsigned char* src_ = G5F_K4_ST ? st + (size_t)j * stm * 16 + (size_t)(slot + rs * i) * rb
                                                      : g5f_row(m[j], e, q_ < R && r0 + q_ < rows ? r0 + q_ : r0);
                const G5fRaw w_ = g5f_k4_load<TY>(src_, sb, p, nb);
#ifdef G5F_K4_NOMATH
                v[j * NR + i] = __uint_as_float((w_.q.x ^ w_.q.y ^ w_.q.z ^ w_.q.w ^ w_.h.x) & 0x3fffffffu);
#else
                v[j * NR + i] = g5f_k4_dot<TY, H2>(w_, a, p);
#endif
            }
        }
        float* bf = buf + (b & 1) * 16 * M * NR;
        g5f_k4_reduce<M * NR>(v, tpr, bf);
        __syncthreads();
        if ((int)threadIdx.x < br) {
            const int s = threadIdx.x / NR, i = threadIdx.x % NR, q = b * br + s + rs * i;
            if (q < R && r0 + q < rows) {
                float t[M];
                #pragma unroll
                for (int j = 0; j < M; j++) t[j] = g5f_k4_total<M * NR>(bf, tpr, s, j * NR + i);
                fin(r0 + q, t);
            }
        }
    }
    g5f_cp_wait<0>();
}

template <int TY, int M, int R, int NR, typename F>
__device__ __forceinline__ void g5f_k4_rows(const G5W* m, unsigned e, int r0, int rows, int k, const unsigned char* xq,
                                            const unsigned char* xq2, const float* xd, float* buf, F fin) {
    if (xq2) g5f_k4_rows2<TY, true, M, R, NR>(m, e, r0, rows, k, xq, xq2, xd, buf, fin);
    else g5f_k4_rows2<TY, false, M, R, NR>(m, e, r0, rows, k, xq, xq2, xd, buf, fin);
}

// gate|up: grid (ceil(max(ff, ffs) / G5F_K4_R), S, T), block 128; slot order as g5f_gux (shared
// first). Slots whose rows are not K4 rows (the Q8_0 shared expert) run g5f_gux's row code.
extern "C" __global__ void __launch_bounds__(128, G5F_K4_MINB) g5f_gu4(
    const G5W gate, const G5W up, const G5W sgate, const G5W sup, const unsigned* ids, int k, int ne,
    const float* x, const unsigned char* xq, const unsigned char* xq2, const float* xd, float* h, int hs, int ff,
    int ffs, int hid, float lim, float lims) {
    __shared__ float buf[2 * 16 * 2 * G5F_K4_NR + 2 * G5F_K4_R];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.z;
    const int nr = ids ? k : 1, ns = sgate.base ? 1 : 0, S = nr + ns;
    const int j = ns ? (blockIdx.y == 0 ? nr : (int)blockIdx.y - 1) : (int)blockIdx.y;
    const bool sh = j >= nr;
    const int rows = sh ? ffs : ff;
    const int r0 = blockIdx.x * G5F_K4_R;
    if (r0 >= rows) return;
    float* hp = h + (size_t)(t * S + j) * hs;
    const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
    if (!sh && ids && e >= (unsigned)ne) {
        if (threadIdx.x < G5F_K4_R && r0 + (int)threadIdx.x < rows) hp[r0 + threadIdx.x] = 0.f;
        return;
    }
    const unsigned char* xqt = xq + (size_t)t * hid;
    const unsigned char* xq2t = xq2 ? xq2 + (size_t)t * hid : 0;
    const float* xdt = xd + (size_t)t * (hid / 32);
    const float* xft = x + (size_t)t * hid;
    const G5W m2[2] = {sh ? sgate : gate, sh ? sup : up};
    const G5W g = m2[0], u = m2[1];
    const float L = sh ? lims : lim;
    auto fin = [&](int r, const float* ab) {
        const float gg = fminf(ab[0], L), uu = fminf(fmaxf(ab[1], -L), L);
        hp[r] = gg / (1.0f + expf(-gg)) * uu;
    };
    if (g.ty == u.ty && g5f_k4_ok(g, hid, false) && g5f_k4_ok(u, hid, false)) {
        if (g.ty == 12) { g5f_k4_rows<12, 2, G5F_K4_R, G5F_K4_NR>(m2, e, r0, rows, hid, xqt, xq2t, xdt, buf, fin); return; }
        if (g.ty == 13) { g5f_k4_rows<13, 2, G5F_K4_R, G5F_K4_NR>(m2, e, r0, rows, hid, xqt, xq2t, xdt, buf, fin); return; }
        if (g.ty == 108) { g5f_k4_rows<108, 2, G5F_K4_R, G5F_K4_NR>(m2, e, r0, rows, hid, xqt, xq2t, xdt, buf, fin); return; }
        g5f_k4_rows<114, 2, G5F_K4_R, G5F_K4_NR>(m2, e, r0, rows, hid, xqt, xq2t, xdt, buf, fin);
        return;
    }
    // g5f_gux's row code over this block's G5F_K4_R rows
    const bool al = G5F_DOT32 && ((g.base | u.base | g.estride | u.estride | g.row_bytes | u.row_bytes) & 15) == 0 &&
                    (hid & 255) == 0;
    const int ty = g.ty == u.ty ? (int)g.ty : -1;
    if (((ty == 12 || ty == 13) && al) || ty == 108 || ty == 114) {
        for (int rr = warp; rr < G5F_K4_R; rr += 4) {
            if (r0 + rr >= rows) break;
            const unsigned char* w[2] = {g5f_row(g, e, r0 + rr), g5f_row(u, e, r0 + rr)};
            float p[2];
            g5f_dot_anyn<2>(g.ty, w, xqt, xq2t, xdt, xft, hid, p);
            const float a = g5f_warp_sum(p[0]), b = g5f_warp_sum(p[1]);
            if (lane == 0) {
                const float ab[2] = {a, b};
                fin(r0 + rr, ab);
            }
        }
        return;
    }
    float* ab = buf + 2 * 16 * 2 * G5F_K4_NR;
    for (int q = warp; q < 2 * G5F_K4_R; q += 4) {
        const int rr = q >> 1;
        if (r0 + rr >= rows) break;
        const G5W v = (q & 1) ? u : g;
        const float sres = g5f_warp_sum(g5f_dot_any<G5F_LDU>(v.ty, g5f_row(v, e, r0 + rr), xqt, xq2t, xdt, xft, hid));
        if (lane == 0) ab[q] = sres;
    }
    __syncthreads();
    if (threadIdx.x < G5F_K4_R && r0 + (int)threadIdx.x < rows) fin(r0 + threadIdx.x, ab + 2 * threadIdx.x);
}

// down: grid (ceil(hid / G5F_K4_DR), S, T), block 128; yp as g5f_down_splitn (summed by g5f_sum).
// Non-K4 slots (the Q8_0 shared expert) run g5f_down_splitn's code, G5F_DNR rows a warp.
extern "C" __global__ void __launch_bounds__(128, G5F_K4_MINB) g5f_down4(
    const G5W down, const G5W sdown, const unsigned* ids, const float* wts, int k, int ne,
    const float* h, const unsigned char* hq, const unsigned char* hq2, const float* hd, int hs, float* yp, int ff,
    int ffs, int hid) {
    __shared__ float buf[2 * 16 * G5F_K4_DNR];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.z;
    const int nr = ids ? k : 1, ns = sdown.base ? 1 : 0, S = nr + ns;
    const int j = ns ? (blockIdx.y == 0 ? nr : (int)blockIdx.y - 1) : (int)blockIdx.y;
    const int b0 = blockIdx.x * G5F_K4_DR;
    if (b0 >= hid) return;
    const bool sh = j >= nr;
    const size_t row = (size_t)t * S + j;
    float* out = yp + row * hid;
    const unsigned e = sh ? 0u : (ids ? ids[(size_t)t * k + j] : 0u);
    if (!sh && ids && e >= (unsigned)ne) {
        if (threadIdx.x < G5F_K4_DR && b0 + (int)threadIdx.x < hid) out[b0 + threadIdx.x] = 0.f;
        return;
    }
    const float wj = sh ? 1.f : (ids ? wts[(size_t)t * k + j] : 1.f);
    const G5W w = sh ? sdown : down;
    const int kk = sh ? ffs : ff;
    const unsigned char* q = hq + row * hs;
    const unsigned char* q2 = hq2 ? hq2 + row * hs : 0;
    const float* d = hd + row * (hs / 32);
    const float* f = h + row * hs;
    if (g5f_k4_ok(w, kk)) {
        auto fin = [&](int r, const float* s) { out[r] = wj * s[0]; };
        if (w.ty == 12) { g5f_k4_rows<12, 1, G5F_K4_DR, G5F_K4_DNR>(&w, e, b0, hid, kk, q, q2, d, buf, fin); return; }
        if (w.ty == 13) { g5f_k4_rows<13, 1, G5F_K4_DR, G5F_K4_DNR>(&w, e, b0, hid, kk, q, q2, d, buf, fin); return; }
        if (w.ty == 108) { g5f_k4_rows<108, 1, G5F_K4_DR, G5F_K4_DNR>(&w, e, b0, hid, kk, q, q2, d, buf, fin); return; }
        g5f_k4_rows<114, 1, G5F_K4_DR, G5F_K4_DNR>(&w, e, b0, hid, kk, q, q2, d, buf, fin);
        return;
    }
    for (int c0 = b0 + warp * G5F_DNR; c0 < b0 + G5F_K4_DR && c0 < hid; c0 += 4 * G5F_DNR) {
        const unsigned char* wr[G5F_DNR];
        #pragma unroll
        for (int i = 0; i < G5F_DNR; i++) wr[i] = g5f_row(w, e, c0 + i < hid ? c0 + i : c0);
        g5f_pf<2>(wr[0], (int)w.row_bytes * (c0 + G5F_DNR <= hid ? G5F_DNR : hid - c0), lane);
        float p[G5F_DNR];
        g5f_dot_anyn<G5F_DNR>(w.ty, wr, q, q2, d, f, kk, p);
        float o[G5F_DNR];
        #pragma unroll
        for (int i = 0; i < G5F_DNR; i++) o[i] = g5f_warp_sum(p[i]);
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < G5F_DNR; i++)
                if (c0 + i < hid && c0 + i < b0 + G5F_K4_DR) out[c0 + i] = wj * o[i];
        }
    }
}

#endif  // fast path
