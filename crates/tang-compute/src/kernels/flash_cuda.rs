//! CUDA kernels for the Flash-Next decode ops (`crate::flash` has the contracts, `cpu::flash`
//! the reference). One NVRTC module; every kernel is `fl_*` so names never collide with the
//! other LLM kernels in the per-name function cache. Shapes come from the `#define` prelude that
//! [`Geometry::cuda_prelude`](crate::flash::Geometry::cuda_prelude) generates; the window size
//! `T` (1..=8) is a kernel argument, per-window values come from the `win` record in device
//! memory.

pub const FLASH_CUDA_BODY: &str = r#"__device__ __forceinline__ float bf(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }
__device__ __forceinline__ unsigned short to_bf16(float x) {
    unsigned int b = __float_as_uint(x);
    return (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
// IEEE half to float, exact for normals, subnormals and zero (no FTZ in this module).
__device__ __forceinline__ float h2f(unsigned short h) {
    float f = __uint_as_float(((unsigned int)(h & 0x7fffu)) << 13) * 5.192296858534828e+33f;
    return (h & 0x8000u) ? -f : f;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ int warp_isum(int v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// ---- int8 activations ([QAct]) ----

// One lane per element of a 32-wide chunk: amax, d = amax / 127, q = round(x / d), the chunk's
// code sum, and the code stored at its permuted slot. `Q` is the [QAct] buffer for `m` rows of
// `k`; this writes row `r`, chunk `c`.
__device__ __forceinline__ void quant_chunk(float v, unsigned int* Q, unsigned int m, unsigned int k,
                                            unsigned int r, unsigned int c, unsigned int lane) {
    float a = warp_max(fabsf(v));
    float d = a / 127.0f;
    int q = 0;
    if (d != 0.0f) q = (int)fminf(fmaxf(roundf(v / d), -127.0f), 127.0f);
    int hx = warp_isum(q);
    unsigned int h = lane >> 4, b = (lane & 15) >> 2, f = lane & 3;
    unsigned char* qb = (unsigned char*)(Q + (u64)r * (k / 4) + c * 8);
    qb[(4 * h + f) * 4 + b] = (unsigned char)(signed char)q;
    if (lane == 0) {
        unsigned int nch = k / 32;
        Q[(u64)m * k / 4 + (u64)r * nch + c] = __float_as_uint(d);
        Q[(u64)m * k / 4 + (u64)m * nch + (u64)r * nch + c] = (unsigned int)hx;
    }
}

// Grid (ceil(K / 256), M), block 256: a warp per chunk.
extern "C" __global__ void fl_quantize(const float* __restrict__ X, unsigned int* __restrict__ Q,
                                       unsigned int M, unsigned int K) {
    unsigned int lane = threadIdx.x & 31, c = blockIdx.x * 8 + (threadIdx.x >> 5), r = blockIdx.y;
    if (c >= K / 32) return;
    quant_chunk(X[(u64)r * K + c * 32 + lane], Q, M, K, r, c, lane);
}

// ---- Q2_0 dot products ----

// Eight dp4a operands for a chunk's two code words: m[4h + f] byte b = code of element 16h+4b+f.
__device__ __forceinline__ void expand(uint2 w, int m[8]) {
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        m[f] = (int)((w.x >> (2 * f)) & 0x03030303u);
        m[4 + f] = (int)((w.y >> (2 * f)) & 0x03030303u);
    }
}
// Σ code · q over one chunk; `xw` points at the chunk's 8 activation words.
__device__ __forceinline__ int chunk_dot(const int* m, const unsigned int* __restrict__ xw) {
    uint4 a = ((const uint4*)xw)[0], b = ((const uint4*)xw)[1];
    int s = 0;
    s = __dp4a(m[0], (int)a.x, s); s = __dp4a(m[1], (int)a.y, s);
    s = __dp4a(m[2], (int)a.z, s); s = __dp4a(m[3], (int)a.w, s);
    s = __dp4a(m[4], (int)b.x, s); s = __dp4a(m[5], (int)b.y, s);
    s = __dp4a(m[6], (int)b.z, s); s = __dp4a(m[7], (int)b.w, s);
    return s;
}

// Multi-column GEMV, Y[t, o] = W[o] · x[t] for T <= 8 columns, three weight formats:
//   FMT 0: bf16 W [N][K] against f32 X [T][K];
//   FMT 1: repacked Q2_0 against int8 XQ ([QAct], m = T): per chunk fma(d_w d_x, S − hx, acc);
//   FMT 2: repacked Q4X (MLX affine, group 64) against int8 XQ: per chunk
//          fma(d_x, fma(s, S, b · hx), acc).
// Each warp owns R = 4 consecutive rows and loads one 16-byte weight vector per row per step,
// applying it to every column, so each activation load serves four rows. KS warps split K for
// one row group (KS = 1, 2, 4, 8; 8 / KS row groups per 256-thread block) and reduce through
// shared memory, which keeps short-N / long-K shapes (10240 -> 320) on many SMs.
// x · eight bf16 weights as one pinned chain (no contraction left to the compiler, so every
// window-width instantiation rounds alike), added to acc.
__device__ __forceinline__ float dot8_rn(uint4 p, float4 a, float4 b, float acc) {
    float s = __fmul_rn(bf(p.x & 0xffff), a.x);
    s = __fmaf_rn(bf(p.x >> 16), a.y, s); s = __fmaf_rn(bf(p.y & 0xffff), a.z, s);
    s = __fmaf_rn(bf(p.y >> 16), a.w, s); s = __fmaf_rn(bf(p.z & 0xffff), b.x, s);
    s = __fmaf_rn(bf(p.z >> 16), b.y, s); s = __fmaf_rn(bf(p.w & 0xffff), b.z, s);
    s = __fmaf_rn(bf(p.w >> 16), b.w, s);
    return __fadd_rn(acc, s);
}

template <int FMT, int T, int GR>
__device__ __forceinline__ void gemv_body(const float* __restrict__ X, const unsigned int* __restrict__ XQ,
                                          const unsigned char* __restrict__ W, float* __restrict__ Y,
                                          unsigned int K, unsigned int N, unsigned int KS, unsigned int OS) {
    __shared__ float red[8][GR * T];
    const unsigned int EPV = FMT == 0 ? 8 : (FMT == 1 ? 64 : (FMT == 2 ? 32 : 16));
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    unsigned int row0 = (blockIdx.x * groups + rg) * GR;
    unsigned int nv = K / EPV, v0 = kw * nv / KS, v1 = (kw + 1) * nv / KS;
    u64 rowbytes = FMT == 0 ? (u64)K * 2 : (FMT == 1 ? K / 4 : (FMT == 2 ? K / 2 : K));
    const unsigned short* sc = (const unsigned short*)(W + (u64)N * rowbytes);
    const unsigned short* bi = sc + (u64)N * (K / 64);
    unsigned int kb = K / 4, nch = K / 32;
    const unsigned int* xs = XQ + (u64)T * kb;
    const unsigned int* xh = xs + (u64)T * nch;
    float acc[GR][T];
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
    if (row0 < N) {
        for (unsigned int v = v0 + lane; v < v1; v += 32) {
            uint4 wv[GR];
            float ws[GR], wb[GR];
            #pragma unroll
            for (int r = 0; r < GR; r++) {
                unsigned int o = min(row0 + r, N - 1);
                wv[r] = ((const uint4*)(W + (u64)o * rowbytes))[v];
                if (FMT == 1) ws[r] = h2f(sc[(u64)o * (K / 64) + v]);
                if (FMT == 3) ws[r] = h2f(sc[(u64)o * (K / 32) + v / 2]);
                if (FMT == 2) {
                    ws[r] = bf(sc[(u64)o * (K / 64) + v / 2]);
                    wb[r] = bf(bi[(u64)o * (K / 64) + v / 2]);
                }
            }
            // dp4a operands, expanded once per row and reused for every column.
            int m[GR][16];
            if (FMT == 1) {
                #pragma unroll
                for (int r = 0; r < GR; r++) {
                    expand(make_uint2(wv[r].x, wv[r].y), &m[r][0]);
                    expand(make_uint2(wv[r].z, wv[r].w), &m[r][8]);
                }
            } else if (FMT == 2) {
                #pragma unroll
                for (int r = 0; r < GR; r++) {
                    unsigned int q[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        m[r][2 * i] = (int)(q[i] & 0x0f0f0f0fu);
                        m[r][2 * i + 1] = (int)((q[i] >> 4) & 0x0f0f0f0fu);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                if (FMT == 0) {
                    const float4* x4 = (const float4*)(X + (u64)t * K + v * 8);
                    float4 a = x4[0], b = x4[1];
                    #pragma unroll
                    for (int r = 0; r < GR; r++) acc[r][t] = dot8_rn(wv[r], a, b, acc[r][t]);
                } else if (FMT == 1) {
                    const unsigned int* xw = XQ + (u64)t * kb + v * 16;
                    float dx0 = __uint_as_float(xs[(u64)t * nch + 2 * v]);
                    float dx1 = __uint_as_float(xs[(u64)t * nch + 2 * v + 1]);
                    int hx0 = (int)xh[(u64)t * nch + 2 * v], hx1 = (int)xh[(u64)t * nch + 2 * v + 1];
                    #pragma unroll
                    for (int r = 0; r < GR; r++) {
                        int s0 = chunk_dot(&m[r][0], xw);
                        int s1 = chunk_dot(&m[r][8], xw + 8);
                        acc[r][t] = __fmaf_rn(ws[r] * dx0, (float)(s0 - hx0), acc[r][t]);
                        acc[r][t] = __fmaf_rn(ws[r] * dx1, (float)(s1 - hx1), acc[r][t]);
                    }
                } else if (FMT == 3) {
                    // Half a chunk: 16 int8 weights (activation order) against 4 activation words.
                    uint4 xw = *(const uint4*)(XQ + (u64)t * kb + v * 4);
                    float dx = __uint_as_float(xs[(u64)t * nch + v / 2]);
                    #pragma unroll
                    for (int r = 0; r < GR; r++) {
                        int S = __dp4a((int)wv[r].x, (int)xw.x, 0);
                        S = __dp4a((int)wv[r].y, (int)xw.y, S);
                        S = __dp4a((int)wv[r].z, (int)xw.z, S);
                        S = __dp4a((int)wv[r].w, (int)xw.w, S);
                        acc[r][t] = __fmaf_rn(ws[r] * dx, (float)S, acc[r][t]);
                    }
                } else {
                    const unsigned int* xw = XQ + (u64)t * kb + v * 8;
                    float dx = __uint_as_float(xs[(u64)t * nch + v]);
                    int hx = (int)xh[(u64)t * nch + v];
                    #pragma unroll
                    for (int r = 0; r < GR; r++) {
                        int S = chunk_dot(&m[r][0], xw);
                        float val = __fmaf_rn(ws[r], (float)S, wb[r] * (float)hx);
                        acc[r][t] = __fmaf_rn(dx, val, acc[r][t]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float v = warp_sum(acc[r][t]);
            if (KS == 1) {
                if (lane == 0 && row0 + r < N) Y[(u64)t * OS + row0 + r] = v;
            } else if (lane == 0) {
                red[warp][r * T + t] = v;
            }
        }
    if (KS == 1) return;
    __syncthreads();
    unsigned int i = threadIdx.x;
    if (i < groups * GR * T) {
        unsigned int g = i / (GR * T), rt = i % (GR * T), r = rt / T, t = rt % T;
        unsigned int o = (blockIdx.x * groups + g) * GR + r;
        float v = 0.0f;
        for (unsigned int k = 0; k < KS; k++) v += red[g * KS + k][rt];
        if (o < N) Y[(u64)t * OS + o] = v;
    }
}

#define GEMV(FMT, NAME, T) \
extern "C" __global__ void __launch_bounds__(256) fl_##NAME##_t##T( \
    const float* __restrict__ X, const unsigned int* __restrict__ XQ, \
    const unsigned char* __restrict__ W, float* __restrict__ Y, unsigned int K, unsigned int N, \
    unsigned int KS, unsigned int OS) { \
    gemv_body<FMT, T, (T == 1 ? 2 : (T <= 8 ? 4 : 1))>(X, XQ, W, Y, K, N, KS, OS); \
}
#define GEMV_ALL(FMT, NAME) GEMV(FMT, NAME, 1) GEMV(FMT, NAME, 2) GEMV(FMT, NAME, 3) GEMV(FMT, NAME, 4) \
    GEMV(FMT, NAME, 5) GEMV(FMT, NAME, 6) GEMV(FMT, NAME, 7) GEMV(FMT, NAME, 8)
GEMV_ALL(0, bf16_gemv)
GEMV_ALL(1, q2_gemv)
GEMV_ALL(2, q4x_gemv)
GEMV_ALL(3, q8x_gemv)
// Prefill-only widths (one row a warp: the per-row arithmetic is the same at any width).
GEMV(0, bf16_gemv, 16) GEMV(0, bf16_gemv, 32) GEMV(0, bf16_gemv, 64)
GEMV(1, q2_gemv, 16) GEMV(1, q2_gemv, 32) GEMV(1, q2_gemv, 64)
GEMV(2, q4x_gemv, 16) GEMV(2, q4x_gemv, 32) GEMV(2, q4x_gemv, 64)
GEMV(3, q8x_gemv, 16) GEMV(3, q8x_gemv, 32) GEMV(3, q8x_gemv, 64)

// ---- hyper-connections ----

// R[t][c] += y[t] * 2σ(inj[t][c] / 4). Grid (HIDDEN / 256, HC, T).
extern "C" __global__ void fl_hc_write(float* __restrict__ R, const float* __restrict__ Y,
                                       const float* __restrict__ I) {
    unsigned int d = blockIdx.x * 256 + threadIdx.x, c = blockIdx.y, t = blockIdx.z;
    float g = 2.0f / (1.0f + expf(-(I[t * HC + c] * 0.25f)));
    u64 i = ((u64)t * HC + c) * HIDDEN + d;
    R[i] = __fmaf_rn(Y[(u64)t * HIDDEN + d], g, R[i]);
}

// Per (stream, token): the pending write, then xn = R · rsqrt(mean R² + eps) · w. `mode` 0: no
// write; 1: R += Yp · 2σ(Ip / 4); 2: the same with Yp the MoE combine of PARTS (router weights
// Wr, shared-gate logit column `sg` of L, -1 for none), computed here exactly as fl_moe_combine
// does. Grid (HC, T), block 256.
extern "C" __global__ void fl_hc_norm(float* __restrict__ R, const float* __restrict__ Yp,
                                      const float* __restrict__ Ip, unsigned int mode,
                                      const float* __restrict__ Wn, float* __restrict__ XN, float eps,
                                      const float* __restrict__ PARTS, const float* __restrict__ Wr,
                                      const float* __restrict__ L, unsigned int stride, int sg) {
    __shared__ float red[8];
    unsigned int c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    float* row = R + ((u64)t * HC + c) * HIDDEN;
    float g = mode ? 2.0f / (1.0f + expf(-(Ip[t * HC + c] * 0.25f))) : 0.0f;
    float gs = (mode == 2 && sg >= 0) ? sigm(L[(u64)t * stride + sg]) : 0.0f;
    float v[10], ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned int d = tid + 256 * i;
        float x = row[d];
        if (mode) {
            float y;
            if (mode == 1) {
                y = Yp[(u64)t * HIDDEN + d];
            } else {
                y = 0.0f;
                #pragma unroll
                for (int k = 0; k < TOPK; k++)
                    y = __fmaf_rn(Wr[t * TOPK + k], PARTS[(u64)(t * TOPK + k) * HIDDEN + d], y);
                if (sg >= 0) y = __fmaf_rn(gs, PARTS[(u64)(SHARED_ROW + t) * HIDDEN + d], y);
            }
            x = __fmaf_rn(y, g, x);
            row[d] = x;
        }
        v[i] = x;
        ss = __fmaf_rn(x, x, ss);
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; w++) ss += red[w];
    float rs = 1.0f / sqrtf(ss / (float)HIDDEN + eps);
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned int d = tid + 256 * i;
        XN[(u64)t * HC * HIDDEN + c * HIDDEN + d] = v[i] * rs * Wn[c * HIDDEN + d];
    }
}

// lo[t][k] = silu((down[k] · xn[t]) / 4) for k < HC_LR, inj[t][c] = inject[c] · xn[t] for the
// next HC rows. Grid ceil(rows / 4), block 256: a block takes 4 rows (so each xn load serves
// four), its 8 warps split K = 10240 and reduce through shared memory.
extern "C" __global__ void __launch_bounds__(256) fl_hc_down(
    const float* __restrict__ XN, const unsigned short* __restrict__ Wd,
    const unsigned short* __restrict__ Wi, float* __restrict__ LO, float* __restrict__ INJ,
    unsigned int T, unsigned int rows) {
    __shared__ float red[8][4][8];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int row0 = blockIdx.x * 4;
    const unsigned short* w[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        unsigned int row = min(row0 + r, rows - 1);
        w[r] = (row < HC_LR ? Wd + (u64)row * HC * HIDDEN : Wi + (u64)(row - HC_LR) * HC * HIDDEN) + warp * 1280;
    }
    float acc[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int t = 0; t < 8; t++) acc[r][t] = 0.0f;
    for (int j = 0; j < 5; j++) {
        unsigned int i = lane + 32 * j;
        uint4 pq[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) pq[r] = ((const uint4*)w[r])[i];
        #pragma unroll
        for (unsigned int t = 0; t < 8; t++) {
            if (t < T) {
                const float4* x4 = (const float4*)(XN + (u64)t * HC * HIDDEN + warp * 1280 + i * 8);
                float4 a = x4[0], b = x4[1];
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    uint4 p = pq[r];
                    acc[r][t] = dot8_rn(p, a, b, acc[r][t]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (unsigned int t = 0; t < 8; t++)
            if (t < T) {
                float s = warp_sum(acc[r][t]);
                if (lane == 0) red[warp][r][t] = s;
            }
    __syncthreads();
    if (threadIdx.x < 4 * T) {
        unsigned int r = threadIdx.x / T, t = threadIdx.x % T, row = row0 + r;
        if (row < rows) {
            float v = 0.0f;
            for (int k = 0; k < 8; k++) v += red[k][r][t];
            if (row < HC_LR) LO[t * HC_LR + row] = silu(v * 0.25f);
            else INJ[t * HC + row - HC_LR] = v;
        }
    }
}

// x[t][d] = (Σ_c xn[t][c][d] · σ(up[c·HIDDEN + d] · lo[t])) / 4; with `quant`, x also as int8
// activations ([QAct], m = T rows of HIDDEN). Grid HIDDEN / 32, block 1024: a warp per d, so a
// block's 32 outputs are one quantization chunk.
template <int T>
__device__ __forceinline__ void hc_up_body(const float* __restrict__ XN, const float* __restrict__ LO,
                                           const unsigned short* __restrict__ Wu, float* __restrict__ X,
                                           unsigned int* __restrict__ XQ, unsigned int quant) {
    __shared__ __align__(16) float lo[T * HC_LR];
    __shared__ float xs[T][32];
    for (unsigned int i = threadIdx.x; i < T * HC_LR; i += 1024) lo[i] = LO[i];
    __syncthreads();
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, d = blockIdx.x * 32 + warp;
    // Repacked up rows: d's 160 vectors are [p][c]; lane l reads stream l % 4, groups l / 4 + 8j.
    const uint4* w = (const uint4*)(Wu + (u64)d * HC * HC_LR);
    uint4 q[5];
    #pragma unroll
    for (int j = 0; j < 5; j++) q[j] = w[lane + 32 * j];
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.0f;
    #pragma unroll
    for (int j = 0; j < 5; j++) {
        unsigned int p = (lane >> 2) + 8 * j;
        float wv[8] = {bf(q[j].x & 0xffff), bf(q[j].x >> 16), bf(q[j].y & 0xffff), bf(q[j].y >> 16),
                       bf(q[j].z & 0xffff), bf(q[j].z >> 16), bf(q[j].w & 0xffff), bf(q[j].w >> 16)};
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float4* l4 = (const float4*)(lo + t * HC_LR + p * 8);
            float4 a = l4[0], b = l4[1];
            float s8 = __fmul_rn(wv[0], a.x);
            s8 = __fmaf_rn(wv[1], a.y, s8); s8 = __fmaf_rn(wv[2], a.z, s8); s8 = __fmaf_rn(wv[3], a.w, s8);
            s8 = __fmaf_rn(wv[4], b.x, s8); s8 = __fmaf_rn(wv[5], b.y, s8); s8 = __fmaf_rn(wv[6], b.z, s8);
            s8 = __fmaf_rn(wv[7], b.w, s8);
            acc[t] = __fadd_rn(acc[t], s8);
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float v = acc[t];
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 16);
        float g = sigm(v);
        float g1 = __shfl_sync(0xffffffffu, g, 1), g2 = __shfl_sync(0xffffffffu, g, 2), g3 = __shfl_sync(0xffffffffu, g, 3);
        if (lane == 0) {
            const float* xn = XN + (u64)t * HC * HIDDEN + d;
            float x = __fmaf_rn(xn[0], g, 0.0f);
            x = __fmaf_rn(xn[HIDDEN], g1, x);
            x = __fmaf_rn(xn[2 * HIDDEN], g2, x);
            x = __fmaf_rn(xn[3 * HIDDEN], g3, x);
            x *= 0.25f;
            X[(u64)t * HIDDEN + d] = x;
            xs[t][warp] = x;
        }
    }
    if (!quant) return;
    __syncthreads();
    if (warp < T) quant_chunk(xs[warp][lane], XQ, T, HIDDEN, warp, blockIdx.x, lane);
}
#define HC_UP(T) \
extern "C" __global__ void __launch_bounds__(1024) fl_hc_up_t##T( \
    const float* __restrict__ XN, const float* __restrict__ LO, const unsigned short* __restrict__ Wu, \
    float* __restrict__ X, unsigned int* __restrict__ XQ, unsigned int quant) { \
    hc_up_body<T>(XN, LO, Wu, X, XQ, quant); \
}
HC_UP(1) HC_UP(2) HC_UP(3) HC_UP(4) HC_UP(5) HC_UP(6) HC_UP(7) HC_UP(8)

// ---- grid barrier and completion counters ----

// All blocks of a co-resident grid meet here. `bar` = [count, generation], self-resetting, so a
// captured graph can replay it.
__device__ void grid_bar(unsigned int* bar, unsigned int nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* gen = bar + 1;
        unsigned int g = *gen;
        __threadfence();
        if (atomicAdd(bar, 1u) == nb - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*gen == g) {}
        }
        __threadfence();
    }
    __syncthreads();
}

// True in exactly one block: the last of `nb` to arrive at `cnt` (which it resets). Everything
// the other blocks wrote before arriving is visible to it.
__device__ bool last_block(unsigned int* cnt, unsigned int nb) {
    __shared__ unsigned int is_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned int prev = atomicAdd(cnt, 1u);
        is_last = prev == nb - 1;
        if (is_last) atomicExch(cnt, 0u);
    }
    __syncthreads();
    if (is_last) __threadfence();
    return is_last;
}

// The whole hyper-connection read in one launch on a co-resident grid (512 threads a block):
// phase 0 the pending write and per-stream norm (fl_hc_norm's arithmetic), barrier, phase 1 the
// down / inject rows (two rows per block step, 16 warps splitting K), barrier, phase 2 the up
// projection and mean (fl_hc_up's arithmetic: a warp per output, a block per 32-wide
// quantization chunk). XN and LO are the hc scratch.
// acc + (w · (a, b)) for 8 bf16 weights in p, as one pinned fma chain.

// Eight int8 codes (one uint2) to floats: the byte goes into the low byte of 2^23 (biased by
// 128), so q = bits − (2^23 + 128), exact; full-rate ops only (prmt, fadd).
__device__ __forceinline__ void q8x8(uint2 w, float q[8]) {
    unsigned int lo = w.x ^ 0x80808080u, hi = w.y ^ 0x80808080u;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        q[j] = __fsub_rn(__int_as_float(__byte_perm(lo, 0x4B000000u, 0x7540u | j)), 8388736.0f);
        q[4 + j] = __fsub_rn(__int_as_float(__byte_perm(hi, 0x4B000000u, 0x7540u | j)), 8388736.0f);
    }
}
// acc + d · (q · (a, b)) for eight int8 codes, as one pinned chain.
__device__ __forceinline__ float dot8q_rn(uint2 w, float d, float4 a, float4 b, float acc) {
    float q[8];
    q8x8(w, q);
    float s = __fmul_rn(q[0], a.x);
    s = __fmaf_rn(q[1], a.y, s); s = __fmaf_rn(q[2], a.z, s); s = __fmaf_rn(q[3], a.w, s);
    s = __fmaf_rn(q[4], b.x, s); s = __fmaf_rn(q[5], b.y, s); s = __fmaf_rn(q[6], b.z, s);
    s = __fmaf_rn(q[7], b.w, s);
    return __fmaf_rn(d, s, acc);
}

// Q8: the matrices are flash::hc_q8 buffers (codes, then f16 scales per 32 at the next 16-byte
// boundary, by GGUF row); dot products take dot8q_rn instead of dot8_rn.
template <int T, bool Q8>
__device__ __forceinline__ void hc_fused_body(
    float* __restrict__ R, const float* __restrict__ Yp, const float* __restrict__ Ip, unsigned int mode,
    const float* __restrict__ Wn, float eps, const float* __restrict__ PARTS, const float* __restrict__ Wr,
    const float* __restrict__ L, unsigned int stride, int sg, const unsigned short* __restrict__ Wd,
    const unsigned short* __restrict__ Wi, unsigned int rows, const unsigned short* __restrict__ Wu,
    float* __restrict__ X, unsigned int* __restrict__ XQ, unsigned int quant, float* __restrict__ INJ,
    float* __restrict__ XN, float* __restrict__ LO, unsigned int* __restrict__ bar) {
    float* PART = LO;  // [4][rows][T] phase-1 partials (flash::hc_scratch_words)
    __shared__ float red[16][2 * T];
    __shared__ float xs[T][32];
    __shared__ __align__(16) float lo[T * HC_LR];
    unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nb = gridDim.x;
    // Phase 0: per (stream, token) row.
    for (unsigned int item = blockIdx.x; item < HC * T; item += nb) {
        unsigned int c = item % HC, t = item / HC;
        float* row = R + ((u64)t * HC + c) * HIDDEN;
        float g = mode ? 2.0f / (1.0f + expf(-(Ip[t * HC + c] * 0.25f))) : 0.0f;
        float gs = (mode == 2 && sg >= 0) ? sigm(L[(u64)t * stride + sg]) : 0.0f;
        float v[5], ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            unsigned int d = tid + 512 * i;
            float x = row[d];
            if (mode) {
                float y;
                if (mode == 1) {
                    y = Yp[(u64)t * HIDDEN + d];
                } else {
                    y = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < TOPK; k++)
                        y = __fmaf_rn(Wr[t * TOPK + k], PARTS[(u64)(t * TOPK + k) * HIDDEN + d], y);
                    if (sg >= 0) y = __fmaf_rn(gs, PARTS[(u64)(SHARED_ROW + t) * HIDDEN + d], y);
                }
                x = __fmaf_rn(y, g, x);
                row[d] = x;
            }
            v[i] = x;
            ss = __fmaf_rn(x, x, ss);
        }
        ss = warp_sum(ss);
        if (lane == 0) red[warp][0] = ss;
        __syncthreads();
        ss = 0.0f;
        for (int w = 0; w < 16; w++) ss += red[w][0];
        float rs = 1.0f / sqrtf(ss / (float)HIDDEN + eps);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            unsigned int d = tid + 512 * i;
            XN[(u64)t * HC * HIDDEN + c * HIDDEN + d] = v[i] * rs * Wn[c * HIDDEN + d];
        }
        __syncthreads();
    }
    grid_bar(bar, nb);
#ifdef HC_STOP1
    return;
#endif
    // Phase 1: items (8-row group rg, K quarter kq), 4 per group: warp w takes rows
    // 8 rg + 2 (w % 4) + {0, 1} over K slice kq · 2560 + (w / 4) · 640, so a block reads a quarter
    // of XN (four warps share each slice in L1) instead of all of it per row pair. Partial sums
    // per (kq, row, t) go to PART; phase 2 adds the quarters in order.
    const unsigned int KW = HC * HIDDEN;
    const unsigned int ngroups = (rows + 7) / 8;
    for (unsigned int item = blockIdx.x; item < 4 * ngroups; item += nb) {
        const unsigned int rg = item / 4, kq = item % 4, rp = warp % 4, ks = warp / 4;
        const unsigned int r0 = 8 * rg + 2 * rp, koff = kq * 2560 + ks * 640;
        const unsigned short* w[2];
        const unsigned char* wq[2];
        const unsigned short* ws[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            unsigned int row = min(r0 + r, rows - 1);
            w[r] = (row < HC_LR ? Wd + (u64)row * KW : Wi + (u64)(row - HC_LR) * KW) + koff;
            if (Q8) {
                const unsigned char* base = (const unsigned char*)(row < HC_LR ? Wd : Wi);
                unsigned int rr = row < HC_LR ? row : row - HC_LR, nr = row < HC_LR ? HC_LR : HC;
                wq[r] = base + (u64)rr * KW + koff;
                ws[r] = (const unsigned short*)(base + (((u64)nr * KW + 15) & ~15ull)) + (u64)rr * (KW / 32) + koff / 32;
            }
        }
        float acc[2][T];
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
        if (r0 < rows) {
            // All of the warp's weight loads (three steps) are issued before any math.
            uint4 p0[3], p1[3];
            uint2 c0[3], c1[3];
            float d0[3], d1[3];
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                unsigned int i = min(lane + 32 * j, 79u);
                if (Q8) {
                    c0[j] = __ldg((const uint2*)wq[0] + i); c1[j] = __ldg((const uint2*)wq[1] + i);
                    d0[j] = h2f(__ldg(ws[0] + i / 4)); d1[j] = h2f(__ldg(ws[1] + i / 4));
                } else {
                    p0[j] = __ldg((const uint4*)w[0] + i); p1[j] = __ldg((const uint4*)w[1] + i);
                }
            }
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                unsigned int i = lane + 32 * j;
                if (i >= 80) break;
                #pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4* x4 = (const float4*)(XN + (u64)t * HC * HIDDEN + koff + i * 8);
                    float4 a = x4[0], b = x4[1];
                    // Explicit fma chains: contraction must not depend on T (the instantiation),
                    // or a token's output would depend on the window it is computed in.
                    if (Q8) {
                        acc[0][t] = dot8q_rn(c0[j], d0[j], a, b, acc[0][t]);
                        acc[1][t] = dot8q_rn(c1[j], d1[j], a, b, acc[1][t]);
                    } else {
                        acc[0][t] = dot8_rn(p0[j], a, b, acc[0][t]);
                        acc[1][t] = dot8_rn(p1[j], a, b, acc[1][t]);
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int t = 0; t < T; t++) {
                float sm = warp_sum(acc[r][t]);
                if (lane == 0) red[warp][r * T + t] = sm;
            }
        __syncthreads();
        if (tid < 8 * T) {
            unsigned int rl = tid / T, t = tid % T, row = 8 * rg + rl;
            if (row < rows) {
                // Row rl is (rp = rl / 2, r = rl % 2): the four K slices are warps rp + 4 ks.
                unsigned int rpp = rl / 2, r = rl % 2;
                float v = red[rpp][r * T + t];
                for (int k = 1; k < 4; k++) v += red[rpp + 4 * k][r * T + t];
                PART[((u64)kq * rows + row) * T + t] = v;
            }
        }
        __syncthreads();
    }
    grid_bar(bar, nb);
#ifdef HC_STOP2
    return;
#endif
    // The bottleneck from the quarters (added in order), and the injection (block 0).
    if (blockIdx.x == 0)
        for (unsigned int i = tid; i < (rows - HC_LR) * T; i += 512) {
            unsigned int row = HC_LR + i / T, t = i % T;
            float v = PART[(u64)row * T + t];
            for (int k = 1; k < 4; k++) v += PART[((u64)k * rows + row) * T + t];
            INJ[t * HC + row - HC_LR] = v;
        }
    // Phase 2: 32 outputs per block step, two per warp.
    if (blockIdx.x >= HIDDEN / 32) return;
    for (unsigned int i = tid; i < T * HC_LR; i += 512) {
        unsigned int t = i / HC_LR, row = i % HC_LR;
        float v = PART[(u64)row * T + t];
        for (int k = 1; k < 4; k++) v += PART[((u64)k * rows + row) * T + t];
        lo[i] = silu(v * 0.25f);
    }
    __syncthreads();
    for (unsigned int q = blockIdx.x; q < HIDDEN / 32; q += nb) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            unsigned int dl = warp + 16 * half, d = q * 32 + dl;
            const uint4* w = (const uint4*)(Wu + (u64)d * HC * HC_LR);
            uint4 qv[5];
            uint2 cv[5];
            float dv[5];
            if (Q8) {
                const unsigned char* base = (const unsigned char*)Wu;
                const unsigned short* sc = (const unsigned short*)(base + (((u64)HC * HIDDEN * HC_LR + 15) & ~15ull));
                // Vector v = lane + 32 j holds GGUF row (c = lane % 4) · HIDDEN + d, k 8p .. 8p + 8.
                #pragma unroll
                for (int j = 0; j < 5; j++) {
                    unsigned int p = (lane >> 2) + 8 * j;
                    cv[j] = __ldg((const uint2*)(base + (u64)d * HC * HC_LR) + lane + 32 * j);
                    dv[j] = h2f(__ldg(sc + (u64)((lane & 3) * HIDDEN + d) * (HC_LR / 32) + p / 4));
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 5; j++) qv[j] = w[lane + 32 * j];
            }
            float acc[T];
            #pragma unroll
            for (int t = 0; t < T; t++) acc[t] = 0.0f;
            #pragma unroll
            for (int j = 0; j < 5; j++) {
                unsigned int p = (lane >> 2) + 8 * j;
                if (Q8) {
                    #pragma unroll
                    for (int t = 0; t < T; t++) {
                        const float4* l4 = (const float4*)(lo + t * HC_LR + p * 8);
                        acc[t] = dot8q_rn(cv[j], dv[j], l4[0], l4[1], acc[t]);
                    }
                    continue;
                }
                float wv[8] = {bf(qv[j].x & 0xffff), bf(qv[j].x >> 16), bf(qv[j].y & 0xffff), bf(qv[j].y >> 16),
                               bf(qv[j].z & 0xffff), bf(qv[j].z >> 16), bf(qv[j].w & 0xffff), bf(qv[j].w >> 16)};
                #pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4* l4 = (const float4*)(lo + t * HC_LR + p * 8);
                    float4 a = l4[0], b = l4[1];
                    float s8 = __fmul_rn(wv[0], a.x);
                    s8 = __fmaf_rn(wv[1], a.y, s8); s8 = __fmaf_rn(wv[2], a.z, s8); s8 = __fmaf_rn(wv[3], a.w, s8);
                    s8 = __fmaf_rn(wv[4], b.x, s8); s8 = __fmaf_rn(wv[5], b.y, s8); s8 = __fmaf_rn(wv[6], b.z, s8);
                    s8 = __fmaf_rn(wv[7], b.w, s8);
                    acc[t] = __fadd_rn(acc[t], s8);
                }
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                float v = acc[t];
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                v += __shfl_xor_sync(0xffffffffu, v, 16);
                float g = sigm(v);
                float g1 = __shfl_sync(0xffffffffu, g, 1), g2 = __shfl_sync(0xffffffffu, g, 2), g3 = __shfl_sync(0xffffffffu, g, 3);
                if (lane == 0) {
                    const float* xn = XN + (u64)t * HC * HIDDEN + d;
                    float x = __fmaf_rn(xn[0], g, 0.0f);
                    x = __fmaf_rn(xn[HIDDEN], g1, x);
                    x = __fmaf_rn(xn[2 * HIDDEN], g2, x);
                    x = __fmaf_rn(xn[3 * HIDDEN], g3, x);
                    x *= 0.25f;
                    X[(u64)t * HIDDEN + d] = x;
                    xs[t][dl] = x;
                }
            }
        }
        __syncthreads();
        if (quant && warp < T) quant_chunk(xs[warp][lane], XQ, T, HIDDEN, warp, q, lane);
        __syncthreads();
    }
}
#define HC_FUSED(T) \
extern "C" __global__ void __launch_bounds__(512, 2) fl_hc_fused_t##T( \
    float* R, const float* Yp, const float* Ip, unsigned int mode, const float* Wn, float eps, \
    const float* PARTS, const float* Wr, const float* L, unsigned int stride, int sg, \
    const unsigned short* Wd, const unsigned short* Wi, unsigned int rows, const unsigned short* Wu, \
    float* X, unsigned int* XQ, unsigned int quant, float* INJ, float* XN, float* LO, unsigned int* bar) { \
    hc_fused_body<T, false>(R, Yp, Ip, mode, Wn, eps, PARTS, Wr, L, stride, sg, Wd, Wi, rows, Wu, X, XQ, quant, \
                            INJ, XN, LO, bar); \
} \
extern "C" __global__ void __launch_bounds__(512, 2) fl_hc_fused_q8_t##T( \
    float* R, const float* Yp, const float* Ip, unsigned int mode, const float* Wn, float eps, \
    const float* PARTS, const float* Wr, const float* L, unsigned int stride, int sg, \
    const unsigned short* Wd, const unsigned short* Wi, unsigned int rows, const unsigned short* Wu, \
    float* X, unsigned int* XQ, unsigned int quant, float* INJ, float* XN, float* LO, unsigned int* bar) { \
    hc_fused_body<T, true>(R, Yp, Ip, mode, Wn, eps, PARTS, Wr, L, stride, sg, Wd, Wi, rows, Wu, X, XQ, quant, \
                           INJ, XN, LO, bar); \
}
HC_FUSED(1) HC_FUSED(2) HC_FUSED(3) HC_FUSED(4) HC_FUSED(5) HC_FUSED(6) HC_FUSED(7) HC_FUSED(8)
// Prefill-only widths (flash-serve).
HC_FUSED(16) HC_FUSED(32)

// Wide windows (prefill, M = 1..64 tokens): hc_fused_body with the token loop tiled by HW_TT.
// Phase-1 and phase-2 weights are loaded into registers once and run over every tile; shared
// memory and accumulators are a tile's. Each token's chains are the T kernels' (explicit fma
// chains, the same K split and reduction order), so its bits don't depend on the width.
#ifndef HW_TT
#define HW_TT 8
#endif
#ifndef HW_MINB
#define HW_MINB 1
#endif
#ifndef HW_T2
#define HW_T2 16
#endif
template <bool Q8>
__device__ __forceinline__ void hc_wide_body(
    float* __restrict__ R, const float* __restrict__ Yp, const float* __restrict__ Ip, unsigned int mode,
    const float* __restrict__ Wn, float eps, const float* __restrict__ PARTS, const float* __restrict__ Wr,
    const float* __restrict__ L, unsigned int stride, int sg, const unsigned short* __restrict__ Wd,
    const unsigned short* __restrict__ Wi, unsigned int rows, const unsigned short* __restrict__ Wu,
    float* __restrict__ X, unsigned int* __restrict__ XQ, unsigned int quant, float* __restrict__ INJ,
    float* __restrict__ XN, float* __restrict__ LO, unsigned int* __restrict__ bar, unsigned int M) {
    float* PART = LO;  // [4][rows][M]
    __shared__ float red[16][2 * HW_TT];
    unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nb = gridDim.x;
    for (unsigned int item = blockIdx.x; item < HC * M; item += nb) {
        unsigned int c = item % HC, t = item / HC;
        float* row = R + ((u64)t * HC + c) * HIDDEN;
        float g = mode ? 2.0f / (1.0f + expf(-(Ip[t * HC + c] * 0.25f))) : 0.0f;
        float gs = (mode == 2 && sg >= 0) ? sigm(L[(u64)t * stride + sg]) : 0.0f;
        float v[5], ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            unsigned int d = tid + 512 * i;
            float x = row[d];
            if (mode) {
                float y;
                if (mode == 1) {
                    y = Yp[(u64)t * HIDDEN + d];
                } else {
                    y = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < TOPK; k++)
                        y = __fmaf_rn(Wr[t * TOPK + k], PARTS[(u64)(t * TOPK + k) * HIDDEN + d], y);
                    if (sg >= 0) y = __fmaf_rn(gs, PARTS[(u64)((M > 8 ? WIDE_SHARED_ROW : SHARED_ROW) + t) * HIDDEN + d], y);
                }
                x = __fmaf_rn(y, g, x);
                row[d] = x;
            }
            v[i] = x;
            ss = __fmaf_rn(x, x, ss);
        }
        ss = warp_sum(ss);
        if (lane == 0) red[warp][0] = ss;
        __syncthreads();
        ss = 0.0f;
        for (int w = 0; w < 16; w++) ss += red[w][0];
        float rs = 1.0f / sqrtf(ss / (float)HIDDEN + eps);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            unsigned int d = tid + 512 * i;
            XN[(u64)t * HC * HIDDEN + c * HIDDEN + d] = v[i] * rs * Wn[c * HIDDEN + d];
        }
        __syncthreads();
    }
    grid_bar(bar, nb);
#ifdef HC_STOP1
    return;
#endif
    const unsigned int KW = HC * HIDDEN;
    const unsigned int ngroups = (rows + 7) / 8;
    for (unsigned int item = blockIdx.x; item < 4 * ngroups; item += nb) {
        const unsigned int rg = item / 4, kq = item % 4, rp = warp % 4, ks = warp / 4;
        const unsigned int r0 = 8 * rg + 2 * rp, koff = kq * 2560 + ks * 640;
        const unsigned short* w[2];
        const unsigned char* wq[2];
        const unsigned short* ws[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            unsigned int row = min(r0 + r, rows - 1);
            w[r] = (row < HC_LR ? Wd + (u64)row * KW : Wi + (u64)(row - HC_LR) * KW) + koff;
            if (Q8) {
                const unsigned char* base = (const unsigned char*)(row < HC_LR ? Wd : Wi);
                unsigned int rr = row < HC_LR ? row : row - HC_LR, nr = row < HC_LR ? HC_LR : HC;
                wq[r] = base + (u64)rr * KW + koff;
                ws[r] = (const unsigned short*)(base + (((u64)nr * KW + 15) & ~15ull)) + (u64)rr * (KW / 32) + koff / 32;
            }
        }
        uint4 p0[3], p1[3];
        uint2 c0[3], c1[3];
        float d0[3], d1[3];
        if (r0 < rows) {
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                unsigned int i = min(lane + 32 * j, 79u);
                if (Q8) {
                    c0[j] = __ldg((const uint2*)wq[0] + i); c1[j] = __ldg((const uint2*)wq[1] + i);
                    d0[j] = h2f(__ldg(ws[0] + i / 4)); d1[j] = h2f(__ldg(ws[1] + i / 4));
                } else {
                    p0[j] = __ldg((const uint4*)w[0] + i); p1[j] = __ldg((const uint4*)w[1] + i);
                }
            }
        }
        for (unsigned int t0 = 0; t0 < M; t0 += HW_TT) {
            float acc[2][HW_TT];
            #pragma unroll
            for (int r = 0; r < 2; r++)
                #pragma unroll
                for (int t = 0; t < HW_TT; t++) acc[r][t] = 0.0f;
            if (r0 < rows) {
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    unsigned int i = lane + 32 * j;
                    if (i >= 80) break;
                    #pragma unroll
                    for (int t = 0; t < HW_TT; t++) {
                        unsigned int tc = min(t0 + t, M - 1);
                        const float4* x4 = (const float4*)(XN + (u64)tc * HC * HIDDEN + koff + i * 8);
                        float4 a = x4[0], b = x4[1];
                        if (Q8) {
                            acc[0][t] = dot8q_rn(c0[j], d0[j], a, b, acc[0][t]);
                            acc[1][t] = dot8q_rn(c1[j], d1[j], a, b, acc[1][t]);
                        } else {
                            acc[0][t] = dot8_rn(p0[j], a, b, acc[0][t]);
                            acc[1][t] = dot8_rn(p1[j], a, b, acc[1][t]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < 2; r++)
                #pragma unroll
                for (int t = 0; t < HW_TT; t++) {
                    float sm = warp_sum(acc[r][t]);
                    if (lane == 0) red[warp][r * HW_TT + t] = sm;
                }
            __syncthreads();
            if (tid < 8 * HW_TT) {
                unsigned int rl = tid / HW_TT, t = tid % HW_TT, row = 8 * rg + rl;
                if (row < rows && t0 + t < M) {
                    unsigned int rpp = rl / 2, r = rl % 2;
                    float v = red[rpp][r * HW_TT + t];
                    for (int k = 1; k < 4; k++) v += red[rpp + 4 * k][r * HW_TT + t];
                    PART[((u64)kq * rows + row) * M + t0 + t] = v;
                }
            }
            __syncthreads();
        }
    }
    grid_bar(bar, nb);
#ifdef HC_STOP2
    return;
#endif
    if (blockIdx.x == 0)
        for (unsigned int i = tid; i < (rows - HC_LR) * M; i += 512) {
            unsigned int row = HC_LR + i / M, t = i % M;
            float v = PART[(u64)row * M + t];
            for (int k = 1; k < 4; k++) v += PART[((u64)k * rows + row) * M + t];
            INJ[t * HC + row - HC_LR] = v;
        }
    // Phase 2 (wide): the lo activations once to global (after PART), then warp items
    // (output d, token tile) over the whole grid, then the quantization as (token, chunk) items.
    float* LOW = LO + (u64)4 * (HC_LR + HC) * M;
    for (unsigned int i = blockIdx.x * 512 + tid; i < M * HC_LR; i += nb * 512) {
        unsigned int t = i / HC_LR, row = i % HC_LR;
        float v = PART[(u64)row * M + t];
        for (int k = 1; k < 4; k++) v += PART[((u64)k * rows + row) * M + t];
        LOW[i] = silu(v * 0.25f);
    }
    grid_bar(bar, nb);
    const unsigned int ntile = (M + HW_T2 - 1) / HW_T2, nw = nb * 16;
    for (unsigned int it = blockIdx.x * 16 + warp; it < HIDDEN * ntile; it += nw) {
        unsigned int d = it % HIDDEN, t0 = (it / HIDDEN) * HW_T2, tn = min(M - t0, (unsigned int)HW_T2);
        const uint4* w = (const uint4*)(Wu + (u64)d * HC * HC_LR);
        uint4 qv[5];
        uint2 cv[5];
        float dv[5];
        if (Q8) {
            const unsigned char* base = (const unsigned char*)Wu;
            const unsigned short* sc = (const unsigned short*)(base + (((u64)HC * HIDDEN * HC_LR + 15) & ~15ull));
            #pragma unroll
            for (int j = 0; j < 5; j++) {
                unsigned int p = (lane >> 2) + 8 * j;
                cv[j] = __ldg((const uint2*)(base + (u64)d * HC * HC_LR) + lane + 32 * j);
                dv[j] = h2f(__ldg(sc + (u64)((lane & 3) * HIDDEN + d) * (HC_LR / 32) + p / 4));
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 5; j++) qv[j] = __ldg(w + lane + 32 * j);
        }
        float acc[HW_T2];
        #pragma unroll
        for (int t = 0; t < HW_T2; t++) acc[t] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            unsigned int p = (lane >> 2) + 8 * j;
            if (Q8) {
                #pragma unroll
                for (int t = 0; t < HW_T2; t++) {
                    const float4* l4 = (const float4*)(LOW + (u64)(t0 + min((unsigned int)t, tn - 1)) * HC_LR + p * 8);
                    acc[t] = dot8q_rn(cv[j], dv[j], l4[0], l4[1], acc[t]);
                }
                continue;
            }
            uint4 qq = qv[j];
            float wv[8] = {bf(qq.x & 0xffff), bf(qq.x >> 16), bf(qq.y & 0xffff), bf(qq.y >> 16),
                           bf(qq.z & 0xffff), bf(qq.z >> 16), bf(qq.w & 0xffff), bf(qq.w >> 16)};
            #pragma unroll
            for (int t = 0; t < HW_T2; t++) {
                const float4* l4 = (const float4*)(LOW + (u64)(t0 + min((unsigned int)t, tn - 1)) * HC_LR + p * 8);
                float4 a = l4[0], b = l4[1];
                float s8 = __fmul_rn(wv[0], a.x);
                s8 = __fmaf_rn(wv[1], a.y, s8); s8 = __fmaf_rn(wv[2], a.z, s8); s8 = __fmaf_rn(wv[3], a.w, s8);
                s8 = __fmaf_rn(wv[4], b.x, s8); s8 = __fmaf_rn(wv[5], b.y, s8); s8 = __fmaf_rn(wv[6], b.z, s8);
                s8 = __fmaf_rn(wv[7], b.w, s8);
                acc[t] = __fadd_rn(acc[t], s8);
            }
        }
        #pragma unroll
        for (int t = 0; t < HW_T2; t++) {
            float v = acc[t];
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            float g = sigm(v);
            float g1 = __shfl_sync(0xffffffffu, g, 1), g2 = __shfl_sync(0xffffffffu, g, 2), g3 = __shfl_sync(0xffffffffu, g, 3);
            if (lane == 0 && t < tn) {
                const float* xn = XN + (u64)(t0 + t) * HC * HIDDEN + d;
                float x = __fmaf_rn(xn[0], g, 0.0f);
                x = __fmaf_rn(xn[HIDDEN], g1, x);
                x = __fmaf_rn(xn[2 * HIDDEN], g2, x);
                x = __fmaf_rn(xn[3 * HIDDEN], g3, x);
                x *= 0.25f;
                X[(u64)(t0 + t) * HIDDEN + d] = x;
            }
        }
    }
    if (!quant) return;
    grid_bar(bar, nb);
    for (unsigned int it = blockIdx.x * 16 + warp; it < M * (HIDDEN / 32); it += nw) {
        unsigned int t = it / (HIDDEN / 32), q = it % (HIDDEN / 32);
        quant_chunk(X[(u64)t * HIDDEN + q * 32 + lane], XQ, M, HIDDEN, t, q, lane);
    }
}
extern "C" __global__ void __launch_bounds__(512, HW_MINB) fl_hc_fusedw(
    float* R, const float* Yp, const float* Ip, unsigned int mode, const float* Wn, float eps,
    const float* PARTS, const float* Wr, const float* L, unsigned int stride, int sg,
    const unsigned short* Wd, const unsigned short* Wi, unsigned int rows, const unsigned short* Wu,
    float* X, unsigned int* XQ, unsigned int quant, float* INJ, float* XN, float* LO, unsigned int* bar,
    unsigned int M) {
    hc_wide_body<false>(R, Yp, Ip, mode, Wn, eps, PARTS, Wr, L, stride, sg, Wd, Wi, rows, Wu, X, XQ, quant,
                        INJ, XN, LO, bar, M);
}
extern "C" __global__ void __launch_bounds__(512, HW_MINB) fl_hc_fusedw_q8(
    float* R, const float* Yp, const float* Ip, unsigned int mode, const float* Wn, float eps,
    const float* PARTS, const float* Wr, const float* L, unsigned int stride, int sg,
    const unsigned short* Wd, const unsigned short* Wi, unsigned int rows, const unsigned short* Wu,
    float* X, unsigned int* XQ, unsigned int quant, float* INJ, float* XN, float* LO, unsigned int* bar,
    unsigned int M) {
    hc_wide_body<true>(R, Yp, Ip, mode, Wn, eps, PARTS, Wr, L, stride, sg, Wd, Wi, rows, Wu, X, XQ, quant,
                       INJ, XN, LO, bar, M);
}

// ---- Gated DeltaNet ----

// The k head v head `h` reads and the output gate: qwen4exp tiles v heads over k heads and
// gates with a sigmoid; a grouped geometry (qwen3next) defines both in the prelude.
#ifndef GDN_KH
#define GDN_KH(h) ((h) % GDN_HK)
#endif
#ifndef GDN_GATE
#define GDN_GATE(z) sigm(z)
#endif

// Grid GDN_CONV / 128 (one block per 128-channel head), block 128.
extern "C" __global__ void fl_gdn_conv(const float* __restrict__ P, unsigned int stride,
                                       const float* __restrict__ H0, const float* __restrict__ Wc,
                                       float* __restrict__ H, unsigned int T, float eps) {
    __shared__ float red[4];
    unsigned int head = blockIdx.x, j = threadIdx.x, ch = head * 128 + j;
    float4 w = ((const float4*)Wc)[ch];
    // Tokens in chunks of 8 (one chunk up to MAX_T; wider windows are prefill). e[i] is conv
    // input row t0 + i of [hist | P].
    for (unsigned int t0 = 0; t0 < T; t0 += 8) {
    float e[11];
    #pragma unroll
    for (unsigned int i = 0; i < 11; i++) {
        unsigned int p = t0 + i;
        e[i] = p < 3 ? H0[p * GDN_CONV + ch] : (p - 3 < T ? P[(u64)(p - 3) * stride + ch] : 0.0f);
    }
    #pragma unroll
    for (unsigned int tl = 0; tl < 8; tl++) {
        unsigned int t = t0 + tl;
        if (t >= T) break;
        float acc = e[tl] * w.x;
        acc = __fmaf_rn(e[tl + 1], w.y, acc);
        acc = __fmaf_rn(e[tl + 2], w.z, acc);
        acc = __fmaf_rn(e[tl + 3], w.w, acc);
        float v = silu(acc);
        if (head < 2 * GDN_HK) {
            float ss = warp_sum(v * v);
            if ((j & 31) == 0) red[j >> 5] = ss;
            __syncthreads();
            ss = ((red[0] + red[1]) + red[2]) + red[3];
            __syncthreads();
            v *= 1.0f / sqrtf(ss + eps);
        }
        H[(u64)t * GDN_CONV + ch] = v;
    }
    }
}

// hist <- the last 3 rows of [hist | qkv_0 .. qkv_{n-1}], n = min(win[1], T). Grid 40, block 256.
extern "C" __global__ void fl_gdn_conv_commit(float* __restrict__ H0, const float* __restrict__ P,
                                              unsigned int stride, const unsigned int* __restrict__ win,
                                              unsigned int T) {
    unsigned int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= GDN_CONV) return;
    unsigned int n = min(win[1], T);
    float old[3] = {H0[ch], H0[GDN_CONV + ch], H0[2 * GDN_CONV + ch]};
    float nw[3];
    #pragma unroll
    for (unsigned int s = 0; s < 3; s++) {
        unsigned int e = n + s;
        nw[s] = e < 3 ? old[e] : P[(u64)(e - 3) * stride + ch];
    }
    #pragma unroll
    for (unsigned int s = 0; s < 3; s++) H0[s * GDN_CONV + ch] = nw[s];
}

// The recurrence + gated norm. Grid GDN_HV (a block per v head), block 512 = 4 row groups of 32
// state rows × 128 columns; each thread holds its 32 state values in registers for the window.
// `commit`: run min(win[1], T) tokens and write the state; else run T, state untouched. Same
// code path either way, so commit's outputs are bitwise verify's.
template <bool FUSED>
__device__ __forceinline__ void gdn_step_body(
    float* __restrict__ S, const float* __restrict__ Hc, const float* __restrict__ P,
    unsigned int stride, const float* __restrict__ DT, const float* __restrict__ SA,
    const float* __restrict__ NW, float* __restrict__ Y, unsigned int T,
    const unsigned int* __restrict__ win, unsigned int commit, float eps,
    unsigned int* __restrict__ YQ, unsigned int quant, const float* __restrict__ H0,
    const float* __restrict__ Wc, unsigned int pr0 = 0, unsigned int plen = 0, unsigned int r0 = 0) {
    // Path rows (tree windows): path index i is window row i < plen ? pr0 + i : r0 + i - plen
    // (identity when plen = r0 = 0). The first plen rows are a parent chain's prefix: they
    // advance the state but their outputs belong to that chain's segment and aren't written.
    // commit & 2: state only (a commit replay), no outputs; the state arithmetic is unchanged.
    #define GDN_ROW(i) ((i) < plen ? pr0 + (i) : r0 + (i) - plen)
    const bool outs = !(commit & 2);
    __shared__ float sq[8][128], sk[8][128], sv[8][128], sz[8][128];
    __shared__ float sg[8], sb[8];
    __shared__ float red[4][128];
    __shared__ float red2[4];
    unsigned int hv = blockIdx.x, hk = GDN_KH(hv), tid = threadIdx.x, j = tid & 127, rg = tid >> 7;
    unsigned int n = commit ? min(win[1], T) : T;
    float st[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) st[r] = S[((u64)(rg * 32 + r) * GDN_HV + hv) * 128 + j];
    // Tokens in chunks of 8 (one chunk up to MAX_T; wider windows are prefill), the state in
    // registers throughout: each token's arithmetic is the same at any width.
    for (unsigned int t0 = 0; t0 < n; t0 += 8) {
    const unsigned int nn = min(n - t0, 8u);
    if (FUSED) {
        __shared__ float cred[2][4];
        unsigned int ch = rg == 0 ? hk * 128 + j : (rg == 1 ? GDN_HK * 128 + hk * 128 + j : 2 * GDN_HK * 128 + hv * 128 + j);
        float4 w = rg < 3 ? ((const float4*)Wc)[ch] : make_float4(0, 0, 0, 0);
        float e[11];
        #pragma unroll
        for (unsigned int i = 0; i < 11; i++) {
            unsigned int p = t0 + i;
            e[i] = rg >= 3 ? 0.0f : (p < 3 ? H0[p * GDN_CONV + ch] : (p - 3 < n ? P[(u64)GDN_ROW(p - 3) * stride + ch] : 0.0f));
        }
        #pragma unroll
        for (unsigned int t = 0; t < 8; t++) {
            if (t >= nn) break;
            float acc = e[t] * w.x;
            acc = __fmaf_rn(e[t + 1], w.y, acc);
            acc = __fmaf_rn(e[t + 2], w.z, acc);
            acc = __fmaf_rn(e[t + 3], w.w, acc);
            float v = silu(acc);
            float ss = warp_sum(v * v);
            if (rg < 2 && (j & 31) == 0) cred[rg][j >> 5] = ss;
            __syncthreads();
            if (rg < 2) {
                ss = ((cred[rg][0] + cred[rg][1]) + cred[rg][2]) + cred[rg][3];
                v *= 1.0f / sqrtf(ss + eps);
            }
            __syncthreads();
            if (rg == 0) sq[t][j] = v;
            else if (rg == 1) sk[t][j] = v;
            else if (rg == 2) sv[t][j] = v;
            else if (outs) sz[t][j] = P[(u64)GDN_ROW(t0 + t) * stride + GDN_Z + hv * 128 + j];
        }
    } else {
        for (unsigned int i = tid; i < nn * 128; i += 512) {
            unsigned int t = i >> 7, jj = i & 127;
            const float* hh = Hc + (u64)GDN_ROW(t0 + t) * GDN_CONV;
            sq[t][jj] = hh[hk * 128 + jj];
            sk[t][jj] = hh[GDN_HK * 128 + hk * 128 + jj];
            sv[t][jj] = hh[2 * GDN_HK * 128 + hv * 128 + jj];
            sz[t][jj] = P[(u64)GDN_ROW(t0 + t) * stride + GDN_Z + hv * 128 + jj];
        }
    }
    if (tid < nn) {
        float a = P[(u64)GDN_ROW(t0 + tid) * stride + GDN_A + hv] + DT[hv];
        float sp = a > 20.0f ? a : log1pf(expf(a));
        sg[tid] = expf(sp * SA[hv]);
        sb[tid] = sigm(P[(u64)GDN_ROW(t0 + tid) * stride + GDN_B + hv]);
    }
    __syncthreads();
    for (unsigned int t = 0; t < nn; t++) {
        float g = sg[t];
        float p = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            st[r] *= g;
            p = __fmaf_rn(st[r], sk[t][rg * 32 + r], p);
        }
        red[rg][j] = p;
        __syncthreads();
        float skj = ((red[0][j] + red[1][j]) + red[2][j]) + red[3][j];
        float d = (sv[t][j] - skj) * sb[t];
        if (!outs || t0 + t < plen) {
            // No output for this row: the state update alone (the same fmas as below).
            #pragma unroll
            for (int r = 0; r < 32; r++) st[r] = __fmaf_rn(sk[t][rg * 32 + r], d, st[r]);
            __syncthreads();
            continue;
        }
        p = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            st[r] = __fmaf_rn(sk[t][rg * 32 + r], d, st[r]);
            p = __fmaf_rn(st[r], sq[t][rg * 32 + r], p);
        }
        __syncthreads();
        red[rg][j] = p;
        __syncthreads();
        float o = (((red[0][j] + red[1][j]) + red[2][j]) + red[3][j]) * INV_SQRT_D;
        if (rg == 0) {
            float ss = warp_sum(o * o);
            if ((j & 31) == 0) red2[j >> 5] = ss;
        }
        __syncthreads();
        if (rg == 0) {
            float ss = ((red2[0] + red2[1]) + red2[2]) + red2[3];
            float rs = 1.0f / sqrtf(ss / 128.0f + eps);
            float y = o * rs * NW[j] * GDN_GATE(sz[t][j]);
            Y[(u64)GDN_ROW(t0 + t) * GDN_V + hv * 128 + j] = y;
            if (quant) quant_chunk(y, YQ, T, GDN_V, t0 + t, hv * 4 + (j >> 5), j & 31);
        }
        __syncthreads();
    }
    }
    if (commit) {
        #pragma unroll
        for (int r = 0; r < 32; r++) S[((u64)(rg * 32 + r) * GDN_HV + hv) * 128 + j] = st[r];
    }
    #undef GDN_ROW
}

extern "C" __global__ void __launch_bounds__(512) fl_gdn_step(
    float* __restrict__ S, const float* __restrict__ Hc, const float* __restrict__ P,
    unsigned int stride, const float* __restrict__ DT, const float* __restrict__ SA,
    const float* __restrict__ NW, float* __restrict__ Y, unsigned int T,
    const unsigned int* __restrict__ win, unsigned int commit, float eps,
    unsigned int* __restrict__ YQ, unsigned int quant) {
    gdn_step_body<false>(S, Hc, P, stride, DT, SA, NW, Y, T, win, commit, eps, YQ, quant, nullptr, nullptr);
}

// The conv (history read-only) and the recurrence in one launch: Hc unused, conv inputs from
// P's qkv columns, the history H0 and the conv weights Wc.
extern "C" __global__ void __launch_bounds__(512) fl_gdn_conv_step(
    float* __restrict__ S, const float* __restrict__ P,
    unsigned int stride, const float* __restrict__ DT, const float* __restrict__ SA,
    const float* __restrict__ NW, float* __restrict__ Y, unsigned int T,
    const unsigned int* __restrict__ win, unsigned int commit, float eps,
    unsigned int* __restrict__ YQ, unsigned int quant, const float* __restrict__ H0,
    const float* __restrict__ Wc) {
    gdn_step_body<true>(S, nullptr, P, stride, DT, SA, NW, Y, T, win, commit, eps, YQ, quant, H0, Wc);
}

#ifdef GDN_SLOTS
// Several sequences in one window (the Coder engine's batched decode): segment b is rows
// SEG[b][0] .. + SEG[b][1] of the window, all of sequence slot SEG[b][2], keeping SEG[b][3] of
// them on commit. Its recurrent state and conv history are at Stab[slot] / Htab[slot]. Each
// segment runs exactly as a lone window of its rows would (the same body). Grid (GDN_HV, nseg).
//
// Tree windows (chain-decomposed): SEG[b][3] with bit 31 set packs keep | pr0 << 8 | plen << 16
// (otherwise it is a plain keep, any size). A chain that
// branches off another after its first plen rows (window rows pr0 ..) runs those rows first,
// in its own registers, as the prefix of its path: shared rows are computed once in the
// window's GEMVs and MoE, and only the cheap recurrence is redone, with no state in HBM.
// keep counts path rows (prefix included). plen = 0 is the plain segment.
#define SEG_KEEP(w) ((w) >> 31 ? (w) & 0xffu : (w))
#define SEG_PR0(w) ((w) >> 31 ? ((w) >> 8) & 0xffu : 0u)
#define SEG_PLEN(w) ((w) >> 31 ? ((w) >> 16) & 0xffu : 0u)
extern "C" __global__ void __launch_bounds__(512) fl_gdn_conv_step_seg(
    const u64* __restrict__ Stab, const u64* __restrict__ Htab, const float* __restrict__ P,
    unsigned int stride, const float* __restrict__ DT, const float* __restrict__ SA,
    const float* __restrict__ NW, float* __restrict__ Y, const unsigned int* __restrict__ SEG,
    unsigned int commit, float eps, const float* __restrict__ Wc) {
    const unsigned int* sg = SEG + blockIdx.y * 4;
    const unsigned int r0 = sg[0], len = sg[1], slot = sg[2], pk = sg[3];
    const unsigned int plen = SEG_PLEN(pk), win[2] = {0u, SEG_KEEP(pk)};
    gdn_step_body<true>((float*)Stab[slot], nullptr, P, stride, DT, SA, NW, Y, plen + len, win,
                        commit, eps, nullptr, 0, (const float*)Htab[slot], Wc, SEG_PR0(pk), plen,
                        r0);
}

// Wide prefill (the Coder engine's wide windows, one segment): the recurrence with 8 lanes per
// value column. A state column S[:, j] evolves on its own given the token's k, q, g, b and
// v_j, so 8 lanes hold column j (16 state rows a lane) and its dot products are 3-step shuffle
// reductions: no block barrier per token, a short dependency chain. Block = 64 columns of head
// blockIdx.x; the block stages a chunk of tokens' conv + SiLU (+ l2 norm for q, k), g and b in
// shared memory. Writes the raw output o (before the gated norm) to Y; fl_gdn_wide_norm
// finishes it. Commits the state for all `len` rows (prefill). Not the narrow kernel's
// summation order (gated by flash-wide-check's KL bar). Grid (GDN_HV, 2), block 512.
#define GW_C 32
__device__ __forceinline__ float sum8(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v;
}
// fl_gdn_wide's per-token inputs, every token in parallel (they don't depend on the state):
// conv + SiLU of q, k, v, l2 norm of q and k, the decay g and beta b. X row t (GW_XROW floats):
// q [GDN_HK][128], k [GDN_HK][128], v [GDN_HV][128], g [GDN_HV], b [GDN_HV]. The same
// arithmetic as the recurrence kernel staged itself (bitwise). Grid n, block 256.
#define GW_XROW (2 * GDN_HK * 128 + GDN_HV * 128 + 2 * GDN_HV)
extern "C" __global__ void __launch_bounds__(256) fl_gdn_wide_prep(
    const u64* __restrict__ Htab, const float* __restrict__ P, unsigned int stride,
    const float* __restrict__ DT, const float* __restrict__ SA, const unsigned int* __restrict__ SEG,
    float eps, const float* __restrict__ Wc, float* __restrict__ X) {
    __shared__ float xs[2 * GDN_HK * 128];
    const unsigned int r0 = SEG[0], slot = SEG[2], t = blockIdx.x;
    const float* H0 = (const float*)Htab[slot];
    const float* Pr = P + (u64)r0 * stride;
    float* Xr = X + (u64)t * GW_XROW;
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (unsigned int ch = tid; ch < GDN_CONV; ch += 256) {
        const float4 w = ((const float4*)Wc)[ch];
        float e[4];
        #pragma unroll
        for (unsigned int q = 0; q < 4; q++) {
            const unsigned int p = t + q;
            e[q] = p < 3 ? H0[p * GDN_CONV + ch] : Pr[(u64)(p - 3) * stride + ch];
        }
        float acc = e[0] * w.x;
        acc = __fmaf_rn(e[1], w.y, acc);
        acc = __fmaf_rn(e[2], w.z, acc);
        acc = __fmaf_rn(e[3], w.w, acc);
        const float v = silu(acc);
        if (ch < 2 * GDN_HK * 128) xs[ch] = v;
        else Xr[ch] = v;
    }
    if (tid < GDN_HV) {
        const float a = Pr[(u64)t * stride + GDN_A + tid] + DT[tid];
        const float sp = a > 20.0f ? a : log1pf(expf(a));
        Xr[2 * GDN_HK * 128 + GDN_HV * 128 + tid] = expf(sp * SA[tid]);
        Xr[2 * GDN_HK * 128 + GDN_HV * 128 + GDN_HV + tid] = sigm(Pr[(u64)t * stride + GDN_B + tid]);
    }
    __syncthreads();
    for (unsigned int h = warp; h < 2 * GDN_HK; h += 8) {
        const float* r = xs + h * 128;
        const float x0 = r[lane], x1 = r[lane + 32], x2 = r[lane + 64], x3 = r[lane + 96];
        const float ss = warp_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
        const float rs = 1.0f / sqrtf(ss + eps);
        float* o = Xr + h * 128;
        o[lane] = x0 * rs; o[lane + 32] = x1 * rs; o[lane + 64] = x2 * rs; o[lane + 96] = x3 * rs;
    }
}

extern "C" __global__ void __launch_bounds__(256) fl_gdn_wide(
    const u64* __restrict__ Stab, const u64* __restrict__ Htab, const float* __restrict__ P,
    unsigned int stride, const float* __restrict__ DT, const float* __restrict__ SA,
    float* __restrict__ Y, const unsigned int* __restrict__ SEG, float eps, const float* __restrict__ Wc,
    const float* __restrict__ X) {
    __shared__ __align__(16) float sq[GW_C][128];
    __shared__ __align__(16) float sk[GW_C][128];
    __shared__ __align__(16) float sv[GW_C][64];
    __shared__ float sg[GW_C], sb[GW_C];
    const unsigned int r0 = SEG[0], n = SEG[1], slot = SEG[2];
    float* S = (float*)Stab[slot];
    const float* H0 = (const float*)Htab[slot];
    const float* Pr = P + (u64)r0 * stride;
    float* Yr = Y + (u64)r0 * GDN_V;
    const unsigned int hv = blockIdx.x, hk = GDN_KH(hv), tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31, sub = lane & 7;
    const unsigned int jl = (warp * 4 + (lane >> 3)) * 2, j = blockIdx.y * 64 + jl;
    float st0[16], st1[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const u64 o = ((u64)(32 * (i >> 2) + 4 * sub + (i & 3)) * GDN_HV + hv) * 128 + j;
        st0[i] = S[o];
        st1[i] = S[o + 1];
    }
    for (unsigned int t0 = 0; t0 < n; t0 += GW_C) {
        const unsigned int nn = min(n - t0, (unsigned int)GW_C);
        __syncthreads();
        // This chunk's inputs from fl_gdn_wide_prep: q, k of head hk, this block's 64 v
        // columns, g and b of head hv.
        for (unsigned int i = tid; i < GW_C * 80; i += 256) {
            const unsigned int t = i / 80, c = i % 80;
            if (t >= nn) continue;
            const float* xr = X + (u64)(t0 + t) * GW_XROW;
            float4 v4;
            if (c < 32) { v4 = ((const float4*)(xr + hk * 128))[c]; *(float4*)&sq[t][4 * c] = v4; }
            else if (c < 64) { v4 = ((const float4*)(xr + GDN_HK * 128 + hk * 128))[c - 32]; *(float4*)&sk[t][4 * (c - 32)] = v4; }
            else { v4 = ((const float4*)(xr + 2 * GDN_HK * 128 + hv * 128 + blockIdx.y * 64))[c - 64]; *(float4*)&sv[t][4 * (c - 64)] = v4; }
        }
        if (tid < nn) {
            const float* xr = X + (u64)(t0 + tid) * GW_XROW + 2 * GDN_HK * 128 + GDN_HV * 128;
            sg[tid] = xr[hv];
            sb[tid] = xr[GDN_HV + hv];
        }
        __syncthreads();
        for (unsigned int t = 0; t < nn; t++) {
            const float g = sg[t];
            float kk[16];
            #pragma unroll
            for (int q4 = 0; q4 < 4; q4++) {
                const float4 v4 = *(const float4*)&sk[t][32 * q4 + 4 * sub];
                kk[4 * q4] = v4.x; kk[4 * q4 + 1] = v4.y; kk[4 * q4 + 2] = v4.z; kk[4 * q4 + 3] = v4.w;
            }
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i += 2) {
                st0[i] *= g; st0[i + 1] *= g; st1[i] *= g; st1[i + 1] *= g;
                p0 = __fmaf_rn(st0[i], kk[i], p0);
                p1 = __fmaf_rn(st0[i + 1], kk[i + 1], p1);
                p2 = __fmaf_rn(st1[i], kk[i], p2);
                p3 = __fmaf_rn(st1[i + 1], kk[i + 1], p3);
            }
            const float pa = sum8(p0 + p1), pb = sum8(p2 + p3);
            const float2 vv = *(const float2*)&sv[t][jl];
            const float bt = sb[t];
            const float d0 = (vv.x - pa) * bt, d1 = (vv.y - pb) * bt;
            float qq[16];
            #pragma unroll
            for (int q4 = 0; q4 < 4; q4++) {
                const float4 v4 = *(const float4*)&sq[t][32 * q4 + 4 * sub];
                qq[4 * q4] = v4.x; qq[4 * q4 + 1] = v4.y; qq[4 * q4 + 2] = v4.z; qq[4 * q4 + 3] = v4.w;
            }
            float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i += 2) {
                st0[i] = __fmaf_rn(kk[i], d0, st0[i]);
                st0[i + 1] = __fmaf_rn(kk[i + 1], d0, st0[i + 1]);
                st1[i] = __fmaf_rn(kk[i], d1, st1[i]);
                st1[i + 1] = __fmaf_rn(kk[i + 1], d1, st1[i + 1]);
                o0 = __fmaf_rn(st0[i], qq[i], o0);
                o1 = __fmaf_rn(st0[i + 1], qq[i + 1], o1);
                o2 = __fmaf_rn(st1[i], qq[i], o2);
                o3 = __fmaf_rn(st1[i + 1], qq[i + 1], o3);
            }
            const float oa = sum8(o0 + o1), ob = sum8(o2 + o3);
            if (sub == 0)
                *(float2*)&Yr[(u64)(t0 + t) * GDN_V + hv * 128 + j] = make_float2(oa * INV_SQRT_D, ob * INV_SQRT_D);
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const u64 o = ((u64)(32 * (i >> 2) + 4 * sub + (i & 3)) * GDN_HV + hv) * 128 + j;
        S[o] = st0[i];
        S[o + 1] = st1[i];
    }
}

// fl_gdn_wide's gated norm, in place on Y: y = o · rms(o)⁻¹ · w · gate(z). Grid (len, GDN_HV),
// block 128.
extern "C" __global__ void fl_gdn_wide_norm(const float* __restrict__ P, unsigned int stride,
                                            const float* __restrict__ NW, float* __restrict__ Y,
                                            const unsigned int* __restrict__ SEG, float eps) {
    __shared__ float red[4];
    const unsigned int r0 = SEG[0], t = blockIdx.x, hv = blockIdx.y, j = threadIdx.x;
    if (t >= SEG[1]) return;
    float* y = Y + (u64)(r0 + t) * GDN_V + hv * 128;
    const float o = y[j];
    const float ss = warp_sum(o * o);
    if ((j & 31) == 0) red[j >> 5] = ss;
    __syncthreads();
    const float tot = ((red[0] + red[1]) + red[2]) + red[3];
    const float rs = 1.0f / sqrtf(tot / 128.0f + eps);
    y[j] = o * rs * NW[j] * GDN_GATE(P[(u64)(r0 + t) * stride + GDN_Z + hv * 128 + j]);
}

// fl_gdn_conv_commit per segment (as above). Grid (ceil(GDN_CONV / 256), nseg), block 256.
extern "C" __global__ void fl_gdn_conv_commit_seg(const u64* __restrict__ Htab, const float* __restrict__ P,
                                                  unsigned int stride, const unsigned int* __restrict__ SEG) {
    const unsigned int* sg = SEG + blockIdx.y * 4;
    const unsigned int r0 = sg[0], len = sg[1], slot = sg[2];
    unsigned int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= GDN_CONV) return;
    float* H0 = (float*)Htab[slot];
    const unsigned int pk = sg[3], pr0 = SEG_PR0(pk), plen = SEG_PLEN(pk);
    unsigned int n = min(SEG_KEEP(pk), plen + len);
    float old[3] = {H0[ch], H0[GDN_CONV + ch], H0[2 * GDN_CONV + ch]};
    float nw[3];
    #pragma unroll
    for (unsigned int s = 0; s < 3; s++) {
        unsigned int e = n + s;
        unsigned int i = e - 3;
        nw[s] = e < 3 ? old[e] : P[(u64)(i < plen ? pr0 + i : r0 + i - plen) * stride + ch];
    }
    #pragma unroll
    for (unsigned int s = 0; s < 3; s++) H0[s * GDN_CONV + ch] = nw[s];
}
#endif

// ---- MoE ----

// One warp, one token: top-10 by (logit desc, index asc) of `l`, weights e_k / Σ_top e_j with
// e_k = exp(l_k − l_max) in f64 summed in rank order. Lane k < 10 returns rank k.
__device__ __forceinline__ void router_warp(const float* __restrict__ l, unsigned int NE, unsigned int lane,
                                            unsigned int* id_out, float* w_out) {
    float v[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned int e = lane + 32 * i;
        v[i] = e < NE ? l[e] : NEG_INF;
    }
    unsigned int used = 0, my_id = 0;
    float my_v = 0.0f, top = 0.0f;
    for (int k = 0; k < TOPK; k++) {
        float bv = NEG_INF;
        unsigned int bi = 0xffffffffu;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned int e = lane + 32 * i;
            if (!((used >> i) & 1) && e < NE && (bi == 0xffffffffu || v[i] > bv)) { bv = v[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            unsigned int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (oi != 0xffffffffu && (bi == 0xffffffffu || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
        }
        if ((bi & 31) == lane) used |= 1u << (bi >> 5);
        if (k == 0) top = bv;
        if (lane == k) { my_id = bi; my_v = bv; }
    }
    double e = lane < TOPK ? exp((double)my_v - (double)top) : 0.0;
    double sum = 0.0;
    for (int k = 0; k < TOPK; k++) sum += __shfl_sync(0xffffffffu, e, k);
    *id_out = my_id;
    *w_out = (float)(e / sum);
}

// Grid T, block 32 (a warp per token).
extern "C" __global__ void fl_router_topk(const float* __restrict__ L, unsigned int stride,
                                          unsigned int NE, unsigned int* __restrict__ IDS,
                                          float* __restrict__ W) {
    unsigned int t = blockIdx.x, lane = threadIdx.x, id;
    float w;
    router_warp(L + (u64)t * stride, NE, lane, &id, &w);
    if (lane < TOPK) {
        IDS[t * TOPK + lane] = id;
        W[t * TOPK + lane] = w;
    }
}

// Inclusive scan of one value per thread over the first 128 threads of the block (every thread
// of the block must call it).
__device__ unsigned int scan128(unsigned int v, unsigned int* sc) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) sc[warp] = v;
    __syncthreads();
    unsigned int off = 0;
    for (unsigned int w = 0; w < warp; w++) off += sc[w];
    __syncthreads();
    return v + off;
}

// The [MoePlan] from router ids `ids` (shared memory, T·TOPK of them, visible to the block) and
// a residency table (EXPERTS addresses, word pairs). Needs at least 128 threads; every step is
// parallel over the routed entries (two scans), no serial loop.
__device__ void plan_core(const unsigned int* ids, const unsigned int* __restrict__ TABLE, u64 shared,
                          unsigned int* __restrict__ PLAN, unsigned int T) {
    __shared__ unsigned int first[PLAN_CAP], gid[PLAN_CAP], start[PLAN_CAP + 1];
    __shared__ unsigned int sc[32];
    __shared__ unsigned int tot_g, tot_e, tot_m;
    unsigned int n = T * TOPK, tid = threadIdx.x;
    u64 p = 0;
    unsigned int me = 0, f = tid;
    if (tid < n) {
        me = ids[tid];
        p = (u64)TABLE[2 * me] | ((u64)TABLE[2 * me + 1] << 32);
        for (unsigned int j = 0; j < tid; j++) if (ids[j] == me) { f = j; break; }
        first[tid] = f;
    }
    bool lead = tid < n && f == tid;
    bool res = lead && p != 0, miss = lead && p == 0;
    unsigned int cnt = 0;
    if (res) for (unsigned int j = tid; j < n; j++) cnt += ids[j] == me;
    // Group index of a resident leader, its entries' start, and the missing-list index.
    unsigned int g_incl = scan128(res ? 1u : 0u, sc);
    unsigned int s_incl = scan128(cnt, sc);
    unsigned int m_incl = scan128(miss ? 1u : 0u, sc);
    if (tid == blockDim.x - 1) { tot_g = g_incl; tot_e = s_incl; tot_m = m_incl; }
    if (res) {
        unsigned int g = g_incl - 1, st = s_incl - cnt;
        gid[tid] = g;
        start[g] = st;
        PLAN[PLAN_GP + 2 * g] = (unsigned int)p;
        PLAN[PLAN_GP + 2 * g + 1] = (unsigned int)(p >> 32);
        PLAN[PLAN_GS + g] = st;
    } else if (lead) {
        gid[tid] = 0xffffffffu;
    }
    if (miss) PLAN[PLAN_MISS + m_incl - 1] = me;
    __syncthreads();
    unsigned int ng = tot_g, s = tot_e;
    if (tid < n) {
        unsigned int g = gid[first[tid]];
        if (g != 0xffffffffu) {
            unsigned int rank = 0;
            for (unsigned int j = 0; j < tid; j++) rank += ids[j] == me;
            unsigned int e = start[g] + rank;
            PLAN[PLAN_ET + e] = tid / TOPK;
            PLAN[PLAN_ED + e] = tid;
        }
    }
    if (shared != 0) {
        if (tid < T) {
            PLAN[PLAN_ET + s + tid] = tid;
            PLAN[PLAN_ED + s + tid] = SHARED_ROW + tid;
        }
        if (tid == 0) {
            PLAN[PLAN_GP + 2 * ng] = (unsigned int)shared;
            PLAN[PLAN_GP + 2 * ng + 1] = (unsigned int)(shared >> 32);
            PLAN[PLAN_GS + ng] = s;
        }
        ng += 1;
        s += T;
    }
    if (tid == 0) {
        PLAN[PLAN_GS + ng] = s;
        PLAN[0] = ng; PLAN[1] = s; PLAN[2] = tot_m; PLAN[3] = 0;
    }
}

// Grid 1, block 128.
extern "C" __global__ void fl_moe_plan(const unsigned int* __restrict__ IDS, const unsigned int* __restrict__ TABLE,
                                       u64 shared, unsigned int* __restrict__ PLAN, unsigned int T) {
    __shared__ unsigned int ids[PLAN_CAP];
    for (unsigned int i = threadIdx.x; i < T * TOPK; i += blockDim.x) ids[i] = IDS[i];
    __syncthreads();
    plan_core(ids, TABLE, shared, PLAN, T);
}

// Router and plan in one launch: warp t routes token t (ids and weights to IDS / W), then the
// block builds the plan from those ids, or from FORCED when `forced` (a recorded routing, for
// replays and benchmarks). Grid 1, block 256.
extern "C" __global__ void fl_moe_route(const float* __restrict__ L, unsigned int stride, unsigned int NE,
                                        const unsigned int* __restrict__ FORCED, unsigned int forced,
                                        const unsigned int* __restrict__ TABLE, u64 shared,
                                        unsigned int* __restrict__ IDS, float* __restrict__ W,
                                        unsigned int* __restrict__ PLAN, unsigned int T) {
    __shared__ unsigned int ids[PLAN_CAP];
    unsigned int lane = threadIdx.x & 31;
    for (unsigned int t = threadIdx.x >> 5; t < T; t += blockDim.x >> 5) {
        unsigned int id;
        float w;
        router_warp(L + (u64)t * stride, NE, lane, &id, &w);
        if (lane < TOPK) {
            IDS[t * TOPK + lane] = id;
            W[t * TOPK + lane] = w;
            if (!forced) ids[t * TOPK + lane] = id;
        }
    }
    if (forced)
        for (unsigned int i = threadIdx.x; i < T * TOPK; i += blockDim.x) ids[i] = FORCED[i];
    __syncthreads();
    plan_core(ids, TABLE, shared, PLAN, T);
}

__device__ __forceinline__ const unsigned char* plan_blob(const unsigned int* PLAN, unsigned int g) {
    return (const unsigned char*)((u64)PLAN[PLAN_GP + 2 * g] | ((u64)PLAN[PLAN_GP + 2 * g + 1] << 32));
}

// exp with the pinned operation sequence of flash::pexp (bitwise the CPU's).
__device__ __forceinline__ float pexpf(float x) {
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    float n = rintf(x * __int_as_float(0x3fb8aa3b));
    float r = __fmaf_rn(-n, __int_as_float(0x35bfbe8e), __fmaf_rn(-n, __int_as_float(0x3f317200), x));
    float p = __int_as_float(0x39500d01);
    p = __fmaf_rn(p, r, __int_as_float(0x3ab60b61));
    p = __fmaf_rn(p, r, __int_as_float(0x3c088889));
    p = __fmaf_rn(p, r, __int_as_float(0x3d2aaaab));
    p = __fmaf_rn(p, r, __int_as_float(0x3e2aaaab));
    p = __fmaf_rn(p, r, 0.5f);
    p = __fmaf_rn(p, r, 1.0f);
    p = __fmaf_rn(p, r, 1.0f);
    return p * __int_as_float(((int)n + 127) << 23);
}
__device__ __forceinline__ float psilu(float x) { return x / (1.0f + pexpf(-x)); }
// MOE_DP4A: the wide module's MoE on the dp4a tiles (A/B against tile_mma).
__device__ __forceinline__ constexpr bool moe_dp4a() {
#ifdef MOE_DP4A
    return true;
#else
    return false;
#endif
}

// Σ code · q over half a chunk: 4 code bytes (16 weights) against one [QAct] uint4 (the half's 4
// activation words), field f of the code bytes pairing with word f.
__device__ __forceinline__ int half_dot(unsigned int w, uint4 x) {
    int s = __dp4a((int)(w & 0x03030303u), (int)x.x, 0);
    s = __dp4a((int)((w >> 2) & 0x03030303u), (int)x.y, s);
    s = __dp4a((int)((w >> 4) & 0x03030303u), (int)x.z, s);
    s = __dp4a((int)((w >> 6) & 0x03030303u), (int)x.w, s);
    return s;
}

// One 16-row expert tile against up to NE entries, in the pinned order of ExpertBlob::ORDER:
// lane = (r = lane / 4, quarter b = lane % 4) owns slots 2b, 2b + 1 (one chunk, both halves) of
// rows r and r + 8, and runs their chains over the rows' 128-weight groups: one 8-byte load per
// row and group (the warp reads two 256-byte lines), each activation load serving both rows.
// The tree: slot m + slot m + 4 is a shuffle with lane b ^ 2, then (l0 + l2) + (l1 + l3) with
// lane b ^ 1. On return, lanes with b = 0 hold rows r (o0) and r + 8 (o1). Entry e reads row
// row[e] of the [QAct] at X (m_rows rows, kb words wide).
// One load step of a tile: B groups' code words and scales for this lane's two rows.
#define TILE_B(G) ((G) % 2 == 0 ? 2 : 5)
template <int G> struct TStep {
    uint2 w[TILE_B(G)][2];
    unsigned short ds[TILE_B(G)][2];
};
template <int G>
__device__ __forceinline__ void tile_load(const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
                                          int g0, unsigned int lane, TStep<G>& st) {
    unsigned int r = lane >> 2, b = lane & 3;
    #pragma unroll
    for (int q = 0; q < TILE_B(G); q++)
        #pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            st.w[q][rr] = __ldg((const uint2*)(codes + (g0 + q) * 512 + (r + 8 * rr) * 32 + b * 8));
            st.ds[q][rr] = __ldg((const unsigned short*)(scales + (g0 + q) * 64 + (r + 8 * rr) * 4 + (b >> 1) * 2));
        }
}

// One 16-row expert tile against up to NE entries, in the pinned order of ExpertBlob::ORDER:
// lane = (r = lane / 4, quarter b = lane % 4) owns slots 2b, 2b + 1 (one chunk, both halves) of
// rows r and r + 8, and runs their chains over the rows' 128-weight groups: one 8-byte load per
// row and group (the warp reads two 256-byte lines), each activation load serving both rows.
// The tree: slot m + slot m + 4 is a shuffle with lane b ^ 2, then (l0 + l2) + (l1 + l3) with
// lane b ^ 1. On return, lanes with b = 0 hold rows r (o0) and r + 8 (o1). Entry e reads row
// row[e] of the [QAct] at X (m_rows rows, kb words wide). Step 0 comes from `pre` when given
// (the caller prefetched it, e.g. during its previous tile).
template <int G, int NE, bool CG = false>
__device__ __forceinline__ void tile_dot(const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
                                         const unsigned int* __restrict__ X, unsigned int kb, unsigned int m_rows,
                                         const unsigned int row[NE], unsigned int ne, unsigned int lane,
                                         float o0[NE], float o1[NE], const TStep<G>* pre = nullptr) {
    constexpr int B = TILE_B(G);
    const unsigned int nch = kb / 8;
    unsigned int b = lane & 3;
    // Entry e's activation words and (d, Σq) at word offsets from X (no per-entry pointers:
    // registers decide the occupancy here).
    const unsigned int* xs = X + (u64)m_rows * kb + b;
    const unsigned int hoff = m_rows * nch;
    if (G == B) {
        // One load step (the down tiles): entries outermost, so only one entry's chains are live
        // (registers set the occupancy); each entry's order is the same as below.
        TStep<G> cur;
        if (pre) cur = *pre; else tile_load<G>(codes, scales, 0, lane, cur);
        #pragma unroll
        for (int e = 0; e < NE; e++) {
            if (e >= ne) continue;
            float a2[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            #pragma unroll
            for (int q = 0; q < B; q++) {
                unsigned int c4 = 4 * q;
                const uint4* xq = (const uint4*)(X + row[e] * kb + b * 8 + c4 * 8);
                uint4 x0 = CG ? __ldcg(xq) : xq[0], x1 = CG ? __ldcg(xq + 1) : xq[1];
                float dx = __uint_as_float(CG ? __ldcg(xs + row[e] * nch + c4) : xs[row[e] * nch + c4]);
                int nh = -(int)(CG ? __ldcg(xs + hoff + row[e] * nch + c4) : xs[hoff + row[e] * nch + c4]);
                #pragma unroll
                for (int rr = 0; rr < 2; rr++) {
                    uint2 w = cur.w[q][rr];
                    int s0 = __dp4a((int)(w.x & 0x03030303u), (int)x0.x, nh);
                    s0 = __dp4a((int)((w.x >> 2) & 0x03030303u), (int)x0.y, s0);
                    s0 = __dp4a((int)((w.x >> 4) & 0x03030303u), (int)x0.z, s0);
                    s0 = __dp4a((int)((w.x >> 6) & 0x03030303u), (int)x0.w, s0);
                    int s1 = __dp4a((int)(w.y & 0x03030303u), (int)x1.x, 0);
                    s1 = __dp4a((int)((w.y >> 2) & 0x03030303u), (int)x1.y, s1);
                    s1 = __dp4a((int)((w.y >> 4) & 0x03030303u), (int)x1.z, s1);
                    s1 = __dp4a((int)((w.y >> 6) & 0x03030303u), (int)x1.w, s1);
                    float dd = h2f(cur.ds[q][rr]) * dx;
                    a2[rr][0] = __fmaf_rn((float)s0, dd, a2[rr][0]);
                    a2[rr][1] = __fmaf_rn((float)s1, dd, a2[rr][1]);
                }
            }
            #pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                float l0 = a2[rr][0] + __shfl_xor_sync(0xffffffffu, a2[rr][0], 2);
                float l1 = a2[rr][1] + __shfl_xor_sync(0xffffffffu, a2[rr][1], 2);
                float x = l0 + __shfl_xor_sync(0xffffffffu, l0, 1);
                float y = l1 + __shfl_xor_sync(0xffffffffu, l1, 1);
                if (rr == 0) o0[e] = x + y; else o1[e] = x + y;
            }
        }
        return;
    }
    float a[NE][2][2];
    #pragma unroll
    for (int e = 0; e < NE; e++)
        #pragma unroll
        for (int q = 0; q < 2; q++) { a[e][q][0] = 0.0f; a[e][q][1] = 0.0f; }
    for (int g0 = 0; g0 < G; g0 += B) {
        TStep<G> cur;
        if (pre && g0 == 0) cur = *pre; else tile_load<G>(codes, scales, g0, lane, cur);
#ifdef MOE_LOADONLY
        #pragma unroll
        for (int q = 0; q < B; q++) a[0][0][0] += __uint_as_float((cur.w[q][0].x ^ cur.w[q][1].y ^ cur.ds[q][0] ^ cur.ds[q][1]) & 0x3fffffffu);
#else
        #pragma unroll
        for (int q = 0; q < B; q++) {
            int op[2][2][4];
            float d[2];
            #pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                d[rr] = h2f(cur.ds[q][rr]);
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    op[rr][0][f] = (int)((cur.w[q][rr].x >> (2 * f)) & 0x03030303u);
                    op[rr][1][f] = (int)((cur.w[q][rr].y >> (2 * f)) & 0x03030303u);
                }
            }
            unsigned int c4 = 4 * (g0 + q);
            #pragma unroll
            for (int e = 0; e < NE; e++) {
                if (e < ne) {
                    const uint4* xq = (const uint4*)(X + row[e] * kb + b * 8 + c4 * 8);
                    uint4 x0 = xq[0], x1 = xq[1];
                    float dx = __uint_as_float(xs[row[e] * nch + c4]);
                    int nh = -(int)xs[hoff + row[e] * nch + c4];
                    #pragma unroll
                    for (int rr = 0; rr < 2; rr++) {
                        int s0 = __dp4a(op[rr][0][0], (int)x0.x, nh);
                        s0 = __dp4a(op[rr][0][1], (int)x0.y, s0);
                        s0 = __dp4a(op[rr][0][2], (int)x0.z, s0);
                        s0 = __dp4a(op[rr][0][3], (int)x0.w, s0);
                        int s1 = __dp4a(op[rr][1][0], (int)x1.x, 0);
                        s1 = __dp4a(op[rr][1][1], (int)x1.y, s1);
                        s1 = __dp4a(op[rr][1][2], (int)x1.z, s1);
                        s1 = __dp4a(op[rr][1][3], (int)x1.w, s1);
                        float dd = d[rr] * dx;
                        a[e][rr][0] = __fmaf_rn((float)s0, dd, a[e][rr][0]);
                        a[e][rr][1] = __fmaf_rn((float)s1, dd, a[e][rr][1]);
                    }
                }
            }
        }
#endif
    }
    #pragma unroll
    for (int e = 0; e < NE; e++) {
        if (e < ne) {
            #pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                float l0 = a[e][rr][0] + __shfl_xor_sync(0xffffffffu, a[e][rr][0], 2);
                float l1 = a[e][rr][1] + __shfl_xor_sync(0xffffffffu, a[e][rr][1], 2);
                float x = l0 + __shfl_xor_sync(0xffffffffu, l0, 1);
                float y = l1 + __shfl_xor_sync(0xffffffffu, l1, 1);
                if (rr == 0) o0[e] = x + y; else o1[e] = x + y;
            }
        }
    }
}

// ---- int8 tensor-core tile (prefill, wide module) ----
// tile_dot's arithmetic with the integer half-chunk sums from mma.sync m16n8k16 (exact, so any
// summation order gives the same int): per (row, entry, chunk b, half) the same chain over the
// groups, a = fma((float)(S [+ nh]), h2f(ds) * dx, a), and the same tree ((b0 + b2) + (b1 + b3))
// per half, then half 0 + half 1. Up to 16 entries (two n8 tiles); rows of entry e: rmap[e], or
// rbase + e without a map. out[row][e] for the 16 tile rows.
__device__ __forceinline__ void mma16816(int c[4], unsigned int a0, unsigned int a1, unsigned int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
}
template <int G, bool CG>
__device__ __forceinline__ void tile_mma(const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
                                         const unsigned int* __restrict__ X, unsigned int kb, unsigned int m_rows,
                                         const unsigned int* __restrict__ rmap, unsigned int rbase, unsigned int ne,
                                         unsigned int lane, float (*out)[17]) {
    const unsigned int nch = kb / 8, gid = lane >> 2, tig = lane & 3;
    const unsigned int* xs = X + (u64)m_rows * kb;
    const unsigned int hoff = m_rows * nch;
    #define TM_ROW(e) (rmap ? rmap[min((unsigned int)(e), ne - 1)] : rbase + min((unsigned int)(e), ne - 1))
    #pragma unroll 1
    for (unsigned int nt = 0; 8 * nt < ne; nt++) {
        const unsigned int rowB = TM_ROW(8 * nt + gid);
        const unsigned int rowE[2] = {TM_ROW(8 * nt + 2 * tig), TM_ROW(8 * nt + 2 * tig + 1)};
        float acc[2][2][4][2];  // [rr][ec][b][half]
        #pragma unroll
        for (int rr = 0; rr < 2; rr++)
            #pragma unroll
            for (int ec = 0; ec < 2; ec++)
                #pragma unroll
                for (int b = 0; b < 4; b++) { acc[rr][ec][b][0] = 0.0f; acc[rr][ec][b][1] = 0.0f; }
        #pragma unroll 2
        for (int g = 0; g < G; g++) {
            uint4 w[2][2];
            unsigned int dsw[2];
            #pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                const uint4* wp = (const uint4*)(codes + g * 512 + (gid + 8 * rr) * 32);
                w[rr][0] = __ldg(wp);
                w[rr][1] = __ldg(wp + 1);
                dsw[rr] = __ldg((const unsigned int*)(scales + g * 64 + (gid + 8 * rr) * 4));
            }
            const uint4* xp = (const uint4*)(X + (u64)rowB * kb + (4 * g) * 8);
            float dx[2][4];
            int nh[2][4];
            #pragma unroll
            for (int ec = 0; ec < 2; ec++) {
                const unsigned int ci = rowE[ec] * nch + 4 * g;
                uint4 d4 = CG ? __ldcg((const uint4*)(xs + ci)) : *(const uint4*)(xs + ci);
                uint4 h4 = CG ? __ldcg((const uint4*)(xs + hoff + ci)) : *(const uint4*)(xs + hoff + ci);
                dx[ec][0] = __uint_as_float(d4.x); dx[ec][1] = __uint_as_float(d4.y);
                dx[ec][2] = __uint_as_float(d4.z); dx[ec][3] = __uint_as_float(d4.w);
                nh[ec][0] = -(int)h4.x; nh[ec][1] = -(int)h4.y; nh[ec][2] = -(int)h4.z; nh[ec][3] = -(int)h4.w;
            }
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                // Chunk b's 8 activation words (both halves); this thread's B word is word tig of each.
                uint4 x0 = CG ? __ldcg(xp + 2 * b) : xp[2 * b], x1 = CG ? __ldcg(xp + 2 * b + 1) : xp[2 * b + 1];
                unsigned int xw[2] = {tig == 0 ? x0.x : tig == 1 ? x0.y : tig == 2 ? x0.z : x0.w,
                                      tig == 0 ? x1.x : tig == 1 ? x1.y : tig == 2 ? x1.z : x1.w};
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    unsigned int cw0 = (b < 2 ? (b == 0 ? (hf ? w[0][0].y : w[0][0].x) : (hf ? w[0][0].w : w[0][0].z))
                                              : (b == 2 ? (hf ? w[0][1].y : w[0][1].x) : (hf ? w[0][1].w : w[0][1].z)));
                    unsigned int cw1 = (b < 2 ? (b == 0 ? (hf ? w[1][0].y : w[1][0].x) : (hf ? w[1][0].w : w[1][0].z))
                                              : (b == 2 ? (hf ? w[1][1].y : w[1][1].x) : (hf ? w[1][1].w : w[1][1].z)));
                    int C[4];
                    mma16816(C, (cw0 >> (2 * tig)) & 0x03030303u, (cw1 >> (2 * tig)) & 0x03030303u, xw[hf]);
                    #pragma unroll
                    for (int rr = 0; rr < 2; rr++) {
                        float ds = h2f((unsigned short)(dsw[rr] >> (16 * (b >> 1))));
                        #pragma unroll
                        for (int ec = 0; ec < 2; ec++) {
                            float dd = ds * dx[ec][b];
                            int sv = hf ? C[2 * rr + ec] : C[2 * rr + ec] + nh[ec][b];
                            acc[rr][ec][b][hf] = __fmaf_rn((float)sv, dd, acc[rr][ec][b][hf]);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int rr = 0; rr < 2; rr++)
            #pragma unroll
            for (int ec = 0; ec < 2; ec++) {
                float (*a)[2] = acc[rr][ec];
                float x = (a[0][0] + a[2][0]) + (a[1][0] + a[3][0]);
                float y = (a[0][1] + a[2][1]) + (a[1][1] + a[3][1]);
                out[gid + 8 * rr][8 * nt + 2 * tig + ec] = x + y;
            }
    }
    #undef TM_ROW
}

// Gate and up rows of every planned expert for its tokens, h = psilu(gate)·up quantized per
// [QAct] into HQ (m = PLAN_CAP rows of FF, row = entry). A gu item is (32-row h chunk q, group g),
// 20 per group, for a 128-thread block: each warp takes one 16-row gu tile (8 h rows), the block
// gathers the chunk's h in shared memory and quantizes it.
template <int NE>
__device__ __forceinline__ void moe_gu_item(const unsigned int* __restrict__ XQ, unsigned int T,
                                            const unsigned int* __restrict__ PLAN, unsigned int* __restrict__ HQ,
                                            unsigned int q, unsigned int g, float (*hs)[32]) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tile = 4 * q + warp;
    const unsigned int kb = HIDDEN / 4;
    const unsigned char* blob = plan_blob(PLAN, g);
    unsigned int e0 = PLAN[PLAN_GS + g], ne = PLAN[PLAN_GS + g + 1] - e0;
    // Entries in passes of EC (the tile is re-read per pass, from L2): the chains of at most EC
    // entries are live, so a wide instantiation costs what a T=4 one does on narrow groups.
    constexpr int EC = NE > 4 ? 4 : NE;
    __syncthreads();
    if (NE > 8 && !moe_dp4a()) {
        // Wide module: tensor-core tiles of 16 entries (tile_mma; same bits as tile_dot).
        __shared__ float om[4][16][17];
        for (unsigned int p0 = 0; p0 < ne; p0 += 16) {
            unsigned int np = min(ne - p0, 16u);
            tile_mma<HIDDEN / 128, false>(blob + GU_CODES + (u64)tile * (HIDDEN / 128) * 512,
                                          blob + GU_SCALES + (u64)tile * (HIDDEN / 128) * 64, XQ, kb, T,
                                          PLAN + PLAN_ET + e0 + p0, 0, np, lane, om[warp]);
            __syncwarp();
            for (unsigned int i = lane; i < 4 * np; i += 32) {
                unsigned int k = i % 4, e = i / 4;
                hs[p0 + e][8 * warp + k] = psilu(om[warp][2 * k][e]) * om[warp][2 * k + 1][e];
                hs[p0 + e][8 * warp + 4 + k] = psilu(om[warp][2 * k + 8][e]) * om[warp][2 * k + 9][e];
            }
            __syncwarp();
        }
        __syncthreads();
        for (unsigned int e = warp; e < ne; e += 4) quant_chunk(hs[e][lane], HQ, PLAN_CAP, FF, e0 + e, q, lane);
        return;
    }
    for (unsigned int p0 = 0; p0 < ne; p0 += EC) {
        unsigned int np = min(ne - p0, (unsigned int)EC);
        unsigned int tok[EC];
        #pragma unroll
        for (int e = 0; e < EC; e++) tok[e] = e < np ? PLAN[PLAN_ET + e0 + p0 + e] : 0;
        float a0[EC], a1[EC];
        tile_dot<HIDDEN / 128, EC>(blob + GU_CODES + (u64)tile * (HIDDEN / 128) * 512,
                                   blob + GU_SCALES + (u64)tile * (HIDDEN / 128) * 64, XQ, kb, T, tok, np, lane, a0, a1);
        #pragma unroll
        for (int e = 0; e < EC; e++) {
            if (e < np) {
                // Lane 4s holds gu rows s (a0) and s + 8 (a1): gate (s even) or up (s odd) of h
                // rows 8·tile + s / 2 and 8·tile + 4 + s / 2.
                float u0 = __shfl_down_sync(0xffffffffu, a0[e], 4);
                float u1 = __shfl_down_sync(0xffffffffu, a1[e], 4);
                if ((lane & 7) == 0) {
                    hs[p0 + e][8 * warp + (lane >> 3)] = psilu(a0[e]) * u0;
                    hs[p0 + e][8 * warp + 4 + (lane >> 3)] = psilu(a1[e]) * u1;
                }
            }
        }
    }
    __syncthreads();
    for (unsigned int e = warp; e < ne; e += 4) quant_chunk(hs[e][lane], HQ, PLAN_CAP, FF, e0 + e, q, lane);
}

template <int NE>
__device__ __forceinline__ void moe_gu_body(const unsigned int* __restrict__ XQ, unsigned int T,
                                            const unsigned int* __restrict__ PLAN, unsigned int* __restrict__ HQ) {
    __shared__ float hs[NE][32];
    unsigned int items = (FF / 32) * PLAN[0];
    for (unsigned int item = blockIdx.x; item < items; item += gridDim.x)
        moe_gu_item<NE>(XQ, T, PLAN, HQ, item % (FF / 32), item / (FF / 32), hs);
}

#define MOE_GU(T) \
extern "C" __global__ void __launch_bounds__(128, 8) fl_moe_gu_t##T( \
    const unsigned int* __restrict__ XQ, unsigned int Tm, const unsigned int* __restrict__ PLAN, \
    unsigned int* __restrict__ HQ) { \
    moe_gu_body<T>(XQ, Tm, PLAN, HQ); \
}
MOE_GU(1) MOE_GU(2) MOE_GU(3) MOE_GU(4) MOE_GU(5) MOE_GU(6) MOE_GU(7) MOE_GU(8)
#ifdef FLASH_WIDE
extern "C" __global__ void __launch_bounds__(128, 4) fl_moe_gu_t64(
    const unsigned int* __restrict__ XQ, unsigned int Tm, const unsigned int* __restrict__ PLAN,
    unsigned int* __restrict__ HQ) {
    moe_gu_body<64>(XQ, Tm, PLAN, HQ);
}
#endif

// Down rows of every planned expert against its entries' quantized h (HQ), into PARTS[dst]:
// one 16-row down tile of group g for a warp (160 per group). CG: read HQ through L2 only (it
// was written in the same launch).
template <int NE, bool CG>
__device__ __forceinline__ void moe_down_tile(const unsigned int* __restrict__ HQ, const unsigned int* __restrict__ PLAN,
                                              float* __restrict__ PARTS, unsigned int tile, unsigned int g) {
    unsigned int lane = threadIdx.x & 31;
    const unsigned int kb = FF / 4;
    const unsigned char* blob = plan_blob(PLAN, g);
    unsigned int e0 = PLAN[PLAN_GS + g], ne_all = PLAN[PLAN_GS + g + 1] - e0;
    if (NE > 8 && !moe_dp4a()) {
        __shared__ float om[8][16][17];
        unsigned int w8 = (threadIdx.x >> 5) & 7;
        for (unsigned int p0 = 0; p0 < ne_all; p0 += 16) {
            unsigned int np = min(ne_all - p0, 16u);
            tile_mma<FF / 128, CG>(blob + DOWN_CODES + (u64)tile * (FF / 128) * 512,
                                   blob + DOWN_SCALES + (u64)tile * (FF / 128) * 64, HQ, kb, PLAN_CAP, nullptr,
                                   e0 + p0, np, lane, om[w8]);
            __syncwarp();
            for (unsigned int i = lane; i < 16 * np; i += 32) {
                unsigned int r = i % 16, e = i / 16, dst = PLAN[PLAN_ED + e0 + p0 + e];
                PARTS[(u64)dst * HIDDEN + 16 * tile + r] = om[w8][r][e];
            }
            __syncwarp();
        }
        return;
    }
    if (NE <= 8) {
        // Decode widths: one pass, the original code shape.
        unsigned int ne = ne_all;
        unsigned int ents[NE];
        #pragma unroll
        for (int e = 0; e < NE; e++) ents[e] = e0 + e;
        float a0[NE], a1[NE];
        tile_dot<FF / 128, NE, CG>(blob + DOWN_CODES + (u64)tile * (FF / 128) * 512,
                                   blob + DOWN_SCALES + (u64)tile * (FF / 128) * 64, HQ, kb, PLAN_CAP, ents, ne, lane,
                                   a0, a1);
        unsigned int my_dst = lane < ne ? PLAN[PLAN_ED + e0 + lane] : 0;
        #pragma unroll
        for (int e = 0; e < NE; e++) {
            unsigned int dst = __shfl_sync(0xffffffffu, my_dst, e);
            if (e < ne && (lane & 3) == 0) {
                float* out = PARTS + (u64)dst * HIDDEN + 16 * tile + (lane >> 2);
                out[0] = a0[e];
                out[8] = a1[e];
            }
        }
        return;
    }
    // Entries in passes of DC (one pass up to T = 8; wide groups re-read the tile from L2).
    constexpr int DC = NE > 8 ? 8 : NE;
    for (unsigned int p0 = 0; p0 < ne_all; p0 += DC) {
    unsigned int ne = min(ne_all - p0, (unsigned int)DC);
    unsigned int ents[DC];
    #pragma unroll
    for (int e = 0; e < DC; e++) ents[e] = e0 + p0 + e;
    float a0[DC], a1[DC];
    tile_dot<FF / 128, DC, CG>(blob + DOWN_CODES + (u64)tile * (FF / 128) * 512,
                               blob + DOWN_SCALES + (u64)tile * (FF / 128) * 64, HQ, kb, PLAN_CAP, ents, ne, lane,
                               a0, a1);
    unsigned int my_dst = lane < ne ? PLAN[PLAN_ED + e0 + p0 + lane] : 0;
    #pragma unroll
    for (int e = 0; e < DC; e++) {
        unsigned int dst = __shfl_sync(0xffffffffu, my_dst, e);
        if (e < ne && (lane & 3) == 0) {
            float* out = PARTS + (u64)dst * HIDDEN + 16 * tile + (lane >> 2);
            out[0] = a0[e];
            out[8] = a1[e];
        }
    }
    }
}

template <int NE>
__device__ __forceinline__ void moe_down_body(const unsigned int* __restrict__ HQ, const unsigned int* __restrict__ PLAN,
                                              float* __restrict__ PARTS) {
    unsigned int warp = threadIdx.x >> 5;
    unsigned int items = (HIDDEN / 16) * PLAN[0];
    for (unsigned int item = blockIdx.x * 8 + warp; item < items; item += gridDim.x * 8)
        moe_down_tile<NE, false>(HQ, PLAN, PARTS, item % (HIDDEN / 16), item / (HIDDEN / 16));
}

#define MOE_DOWN(T) \
extern "C" __global__ void __launch_bounds__(256, 3) fl_moe_down_t##T( \
    const unsigned int* __restrict__ HQ, const unsigned int* __restrict__ PLAN, float* __restrict__ PARTS) { \
    moe_down_body<T>(HQ, PLAN, PARTS); \
}
MOE_DOWN(1) MOE_DOWN(2) MOE_DOWN(3) MOE_DOWN(4) MOE_DOWN(5) MOE_DOWN(6) MOE_DOWN(7) MOE_DOWN(8)
#ifdef FLASH_WIDE
extern "C" __global__ void __launch_bounds__(256, 2) fl_moe_down_t64(
    const unsigned int* __restrict__ HQ, const unsigned int* __restrict__ PLAN, float* __restrict__ PARTS) {
    moe_down_body<64>(HQ, PLAN, PARTS);
}
#endif

// gu and down in one launch on a co-resident grid of 128-thread blocks: items are every group's
// 20 gu items, then every group's 40 down items (4 tiles, a warp each), taken in index order
// (block b: b, b + grid, ...). A down item of group g waits for g's 20 gu items (CNT[g], counted
// after each item's HQ is written), so the down weights stream while the last gu items finish
// instead of after a launch boundary. Each block first runs all its gu items, so a waiting block
// never holds back a gu item: no deadlock. The last block out resets CNT. Same per-tile
// arithmetic as the two kernels (bitwise).
template <int NE>
__device__ __forceinline__ void moe_fused_body(const unsigned int* __restrict__ XQ, unsigned int T,
                                               const unsigned int* __restrict__ PLAN, unsigned int* __restrict__ HQ,
                                               float* __restrict__ PARTS, unsigned int* __restrict__ CNT) {
    __shared__ float hs[NE][32];
    const unsigned int ng = PLAN[0], ngu = (FF / 32) * ng, items = ngu + (HIDDEN / 64) * ng;
    for (unsigned int item = blockIdx.x; item < items; item += gridDim.x) {
        if (item < ngu) {
            unsigned int g = item / (FF / 32);
            moe_gu_item<NE>(XQ, T, PLAN, HQ, item % (FF / 32), g, hs);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(CNT + g, 1u);
            }
        } else {
            unsigned int d = item - ngu, g = d / (HIDDEN / 64), tile = 4 * (d % (HIDDEN / 64)) + (threadIdx.x >> 5);
            if (threadIdx.x == 0) {
                volatile unsigned int* c = CNT + g;
                while (*c < FF / 32) __nanosleep(32);
                __threadfence();
            }
            __syncthreads();
            moe_down_tile<NE, true>(HQ, PLAN, PARTS, tile, g);
        }
    }
    if (last_block(CNT + PLAN_CAP, gridDim.x))
        for (unsigned int i = threadIdx.x; i < ng; i += blockDim.x) CNT[i] = 0;
}

#define MOE_FUSED(T) \
extern "C" __global__ void __launch_bounds__(128, 8) fl_moe_fused_t##T( \
    const unsigned int* __restrict__ XQ, unsigned int Tm, const unsigned int* __restrict__ PLAN, \
    unsigned int* __restrict__ HQ, float* __restrict__ PARTS, unsigned int* __restrict__ CNT) { \
    moe_fused_body<T>(XQ, Tm, PLAN, HQ, PARTS, CNT); \
}
MOE_FUSED(1) MOE_FUSED(2) MOE_FUSED(3) MOE_FUSED(4) MOE_FUSED(5) MOE_FUSED(6) MOE_FUSED(7) MOE_FUSED(8)
#ifdef FLASH_WIDE
extern "C" __global__ void __launch_bounds__(128, 4) fl_moe_fused_t64(
    const unsigned int* __restrict__ XQ, unsigned int Tm, const unsigned int* __restrict__ PLAN,
    unsigned int* __restrict__ HQ, float* __restrict__ PARTS, unsigned int* __restrict__ CNT) {
    moe_fused_body<64>(XQ, Tm, PLAN, HQ, PARTS, CNT);
}
#endif

// y[t] = Σ_i w[t][i] parts[t·10 + i] (+ σ(logit[t][sg]) parts[SHARED_ROW + t]). Grid
// (HIDDEN / 256, T).
extern "C" __global__ void fl_moe_combine(const float* __restrict__ PARTS, const float* __restrict__ W,
                                          const float* __restrict__ L, unsigned int stride, int sg,
                                          float* __restrict__ Y) {
    unsigned int t = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < TOPK; i++) acc = __fmaf_rn(W[t * TOPK + i], PARTS[(u64)(t * TOPK + i) * HIDDEN + d], acc);
    if (sg >= 0) acc = __fmaf_rn(sigm(L[(u64)t * stride + sg]), PARTS[(u64)(SHARED_ROW + t) * HIDDEN + d], acc);
    Y[(u64)t * HIDDEN + d] = acc;
}

// ---- QSA ----

// RMSNorm then NeoX rope on the first 64 dims of a D-wide head (D = 256 or 128), by the whole
// 256-thread block. `buf` is shared scratch of 256 floats; returns this thread's output (j < D).
__device__ float norm_rope(const float* __restrict__ src, const float* __restrict__ w, unsigned int D,
                           const float* __restrict__ COS, const float* __restrict__ SIN, unsigned int pos,
                           float eps, float* buf, float* red) {
    unsigned int j = threadIdx.x;
    float x = j < D ? src[j] : 0.0f;
    float ss = warp_sum(x * x);
    if ((j & 31) == 0) red[j >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
    for (int k = 0; k < 8; k++) ss += red[k];
    float rs = 1.0f / sqrtf(ss / (float)D + eps);
    float y = j < D ? x * rs * w[j] : 0.0f;
    buf[j] = y;
    __syncthreads();
    if (j < 32) {
        float c = COS[pos * 32 + j], s = SIN[pos * 32 + j];
        y = __fmaf_rn(buf[j], c, -(buf[j + 32] * s));
    } else if (j < 64) {
        float c = COS[pos * 32 + j - 32], s = SIN[pos * 32 + j - 32];
        y = __fmaf_rn(buf[j], c, buf[j - 32] * s);
    }
    __syncthreads();
    return y;
}

// Grid (34, T), block 256. Jobs: 0..23 q heads, 24..25 k heads, 26..27 v heads, 28..31 indexer
// q heads, 32 raw indexer key into the ring, 33 pool the block this token completes.
extern "C" __global__ void fl_qsa_prep(
    const float* __restrict__ P, unsigned int stride, const unsigned int* __restrict__ win,
    const float* __restrict__ QN, const float* __restrict__ KN, const float* __restrict__ IQN,
    const float* __restrict__ IKN, const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Q, unsigned short* __restrict__ KC, unsigned short* __restrict__ VC,
    float* __restrict__ RING, float* __restrict__ POOLED, unsigned int T, float eps) {
    __shared__ float buf[256], red[8], mean[128];
    unsigned int job = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    unsigned int pos0 = win[0], pos = pos0 + t;
    const float* p = P + (u64)t * stride;
    if (job < 24) {
        float y = norm_rope(p + job * 512, QN, 256, COS, SIN, pos, eps, buf, red);
        Q[((u64)t * QSA_HEADS + job) * QSA_D + j] = y;
    } else if (job < 26) {
        unsigned int g = job - 24;
        float y = norm_rope(p + QSA_K + g * 256, KN, 256, COS, SIN, pos, eps, buf, red);
        KC[((u64)pos * QSA_KV + g) * QSA_D + j] = to_bf16(y);
    } else if (job < 28) {
        unsigned int g = job - 26;
        VC[((u64)pos * QSA_KV + g) * QSA_D + j] = to_bf16(p[QSA_V + g * 256 + j]);
    } else if (job < 32) {
        unsigned int h = job - 28;
        float y = norm_rope(p + QSA_IQ + h * 128, IQN, 128, COS, SIN, pos, eps, buf, red);
        if (j < 128) Q[(u64)T * QSA_HEADS * QSA_D + ((u64)t * IDX_HEADS + h) * IDX_D + j] = y;
    } else if (job == 32) {
        if (j < 128) RING[(pos % QSA_RING) * IDX_D + j] = p[QSA_IK + j];
    } else {
        if (pos % 4 != 3) return;
        unsigned int b = pos / 4, c0 = 4 * b;
        if (j < 128) {
            float r[4];
            for (int i = 0; i < 4; i++) {
                unsigned int c = c0 + i;
                r[i] = c >= pos0 ? P[(u64)(c - pos0) * stride + QSA_IK + j] : RING[(c % QSA_RING) * IDX_D + j];
            }
            mean[j] = (((r[0] + r[1]) + r[2]) + r[3]) * 0.25f;
        }
        __syncthreads();
        float y = norm_rope(mean, IKN, 128, COS, SIN, c0, eps, buf, red);
        if (j < 128) POOLED[(u64)b * IDX_D + j] = y;
    }
}

// Block scores: grid (ceil(MAXB / 8), T), block 256, a warp per pooled block.
extern "C" __global__ void fl_qsa_scores(const float* __restrict__ POOLED, const float* __restrict__ IQ,
                                         const unsigned int* __restrict__ win, float* __restrict__ SC,
                                         unsigned int MAXB) {
    unsigned int t = blockIdx.y, lane = threadIdx.x & 31;
    unsigned int b = blockIdx.x * 8 + (threadIdx.x >> 5);
    unsigned int n_bid = (win[0] + t + 1) / 4;
    if (b >= n_bid || b >= MAXB) return;
    float4 pk = ((const float4*)(POOLED + (u64)b * IDX_D))[lane];
    float total = 0.0f;
    #pragma unroll
    for (int h = 0; h < IDX_HEADS; h++) {
        float4 q = ((const float4*)(IQ + ((u64)t * IDX_HEADS + h) * IDX_D))[lane];
        float a = q.x * pk.x;
        a = __fmaf_rn(q.y, pk.y, a);
        a = __fmaf_rn(q.z, pk.z, a);
        a = __fmaf_rn(q.w, pk.w, a);
        a = warp_sum(a);
        total += a > 0.0f ? a : 0.0f;
    }
    if (lane == 0) SC[(u64)t * MAXB + b] = total * INV_SQRT_D;
}

__device__ __forceinline__ unsigned int ordered(float x) {
    unsigned int b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Inclusive block scan of one value per thread (1024 threads): warp scans, then a scan of the
// 32 warp totals.
__device__ unsigned int block_scan(unsigned int v, unsigned int* sc) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) sc[warp] = v;
    __syncthreads();
    if (warp == 0) {
        unsigned int w = sc[lane];
        for (int o = 1; o < 32; o <<= 1) {
            unsigned int n = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += n;
        }
        sc[lane] = w;
    }
    __syncthreads();
    unsigned int r = v + (warp > 0 ? sc[warp - 1] : 0);
    __syncthreads();
    return r;
}

#define SEL_MAXB 8192
#define SEL_BLOCKS 512
// Cells token t selects: every cell up to 512 complete blocks, else 512 blocks plus the tail.
__device__ __forceinline__ unsigned int qsa_n_sel(unsigned int n_kv) {
    return n_kv / 4 <= SEL_BLOCKS ? n_kv : 4 * SEL_BLOCKS + n_kv % 4;
}
// Top cells per token. Grid T, block 1024.
extern "C" __global__ void __launch_bounds__(1024) fl_qsa_select(
    const float* __restrict__ SC, const unsigned int* __restrict__ win, unsigned int* __restrict__ IDS,
    unsigned int MAXB, unsigned int* __restrict__ BM, unsigned int mask) {
    __shared__ unsigned int keys[SEL_MAXB];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int scan[1024];
    __shared__ unsigned int sh_digit, sh_k;
    unsigned int t = blockIdx.x, tid = threadIdx.x;
    unsigned int n_kv = win[0] + t + 1;
    unsigned int* out = IDS + (u64)t * QSA_WIDTH;
    unsigned int n_bid = n_kv / 4, tail = n_kv % 4;
    if (n_bid <= SEL_BLOCKS) {
        for (unsigned int i = tid; i < n_kv; i += 1024) out[i] = i;
        if (mask)
            for (unsigned int b = tid; b < n_bid; b += 1024) atomicOr(&BM[b], 1u << t);
        return;
    }
    // The top 512 complete blocks by (score desc, block asc), then the tail's cells.
    const unsigned int need = SEL_BLOCKS, rem = 0;
    for (unsigned int b = tid; b < n_bid; b += 1024) keys[b] = ordered(SC[(u64)t * MAXB + b]);
    __syncthreads();
    unsigned int prefix = 0, pmask = 0, k = need;
    for (int pass = 0; pass < 4; pass++) {
        unsigned int shift = 24 - 8 * pass;
        for (unsigned int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (unsigned int b = tid; b < n_bid; b += 1024)
            if ((keys[b] & pmask) == prefix) atomicAdd(&hist[(keys[b] >> shift) & 255u], 1u);
        __syncthreads();
        if (tid < 32) {
            // Lane l owns digits 255 - 8l .. 248 - 8l: a warp scan finds the digit holding the
            // k-th largest key.
            unsigned int loc[8], sum = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) { loc[i] = hist[255 - 8 * tid - i]; sum += loc[i]; }
            unsigned int incl = sum;
            for (int o = 1; o < 32; o <<= 1) {
                unsigned int n = __shfl_up_sync(0xffffffffu, incl, o);
                if (tid >= o) incl += n;
            }
            unsigned int acc = incl - sum;
            if (acc < k && incl >= k) {
                for (int i = 0; i < 8; i++) {
                    if (acc + loc[i] >= k) { sh_digit = 255 - 8 * tid - i; sh_k = k - acc; break; }
                    acc += loc[i];
                }
            }
        }
        __syncthreads();
        prefix |= sh_digit << shift;
        pmask |= 255u << shift;
        k = sh_k;
        __syncthreads();
    }
    unsigned int v = prefix, take_eq = k;
    unsigned int per = (n_bid + 1023) / 1024, b0 = min(tid * per, n_bid), b1 = min(b0 + per, n_bid);
    unsigned int eq = 0;
    for (unsigned int b = b0; b < b1; b++) eq += keys[b] == v;
    unsigned int eq_before = block_scan(eq, scan) - eq;
    unsigned int cells = 0, r = eq_before;
    for (unsigned int b = b0; b < b1; b++) {
        unsigned int key = keys[b];
        unsigned int c = 0;
        if (key > v) c = 4;
        else if (key == v) {
            if (r < take_eq) c = (r == take_eq - 1 && rem > 0) ? rem : 4;
            r++;
        }
        cells += c;
    }
    unsigned int off = block_scan(cells, scan) - cells;
    r = eq_before;
    for (unsigned int b = b0; b < b1; b++) {
        unsigned int key = keys[b];
        unsigned int c = 0;
        if (key > v) c = 4;
        else if (key == v) {
            if (r < take_eq) c = (r == take_eq - 1 && rem > 0) ? rem : 4;
            r++;
        }
        for (unsigned int i = 0; i < c; i++) out[off++] = b * 4 + i;
        if (mask && c) atomicOr(&BM[b], 1u << t);
    }
    if (tid == 0)
        for (unsigned int i = 0; i < tail; i++) out[4 * SEL_BLOCKS + i] = n_bid * 4 + i;
}

// Split-K attention: grid (QSA_NCH, QSA_KV, T), block 256. A block reads its chunk of
// QSA_CHUNK selected cells' K and V once for the 12 q heads that share the kv head: scores a
// thread per (cell, 3 heads), each reading its cell's whole key row; softmax a warp per head;
// p·V a thread per (dim pair, half of the cells). Writes per head [m, l, acc[256]] partials.
extern "C" __global__ void __launch_bounds__(256) fl_qsa_attend(
    const float* __restrict__ Q, const unsigned short* __restrict__ KC, const unsigned short* __restrict__ VC,
    const unsigned int* __restrict__ IDS, const unsigned int* __restrict__ win, float* __restrict__ PART) {
    __shared__ __align__(16) float qs[12][256];
    __shared__ float s[12][QSA_CHUNK];
    __shared__ unsigned int cells[QSA_CHUNK];
    __shared__ __align__(8) float half[12][256];
    unsigned int ch = blockIdx.x, g = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    unsigned int lane = tid & 31, warp = tid >> 5;
    unsigned int n_sel = qsa_n_sel(win[0] + t + 1);
    unsigned int i0 = ch * QSA_CHUNK;
    if (i0 >= n_sel) return;
    unsigned int n = min((unsigned int)QSA_CHUNK, n_sel - i0);
    for (unsigned int i = tid; i < 12 * 256 / 4; i += 256)
        ((float4*)&qs[0][0])[i] = ((const float4*)(Q + ((u64)t * QSA_HEADS + g * 12) * QSA_D))[i];
    if (tid < n) cells[tid] = IDS[(u64)t * QSA_WIDTH + i0 + tid];
    __syncthreads();
    {
        unsigned int c = tid % QSA_CHUNK, hg = tid / QSA_CHUNK;  // 4 groups of 3 heads
        if (c < n) {
            const uint4* kr = (const uint4*)(KC + ((u64)cells[c] * QSA_KV + g) * QSA_D);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            #pragma unroll 4
            for (int v = 0; v < 32; v++) {
                uint4 kq = kr[v];
                float kv[8] = {bf(kq.x & 0xffff), bf(kq.x >> 16), bf(kq.y & 0xffff), bf(kq.y >> 16),
                               bf(kq.z & 0xffff), bf(kq.z >> 16), bf(kq.w & 0xffff), bf(kq.w >> 16)};
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    a0 = __fmaf_rn(kv[e], qs[3 * hg][8 * v + e], a0);
                    a1 = __fmaf_rn(kv[e], qs[3 * hg + 1][8 * v + e], a1);
                    a2 = __fmaf_rn(kv[e], qs[3 * hg + 2][8 * v + e], a2);
                }
            }
            s[3 * hg][c] = a0 * 0.0625f;
            s[3 * hg + 1][c] = a1 * 0.0625f;
            s[3 * hg + 2][c] = a2 * 0.0625f;
        }
    }
    __syncthreads();
    for (unsigned int h = warp; h < 12; h += 8) {
        float a = NEG_INF, b = NEG_INF;
        if (lane < n) a = s[h][lane];
        if (QSA_CHUNK > 32 && lane + 32 < n) b = s[h][lane + 32];
        float m = warp_max(fmaxf(a, b));
        float pa = lane < n ? expf(a - m) : 0.0f, pb = (QSA_CHUNK > 32 && lane + 32 < n) ? expf(b - m) : 0.0f;
        if (lane < n) s[h][lane] = pa;
        if (QSA_CHUNK > 32 && lane + 32 < n) s[h][lane + 32] = pb;
        float l = warp_sum(pa + pb);
        if (lane == 0) {
            float* part = PART + (((u64)t * QSA_HEADS + g * 12 + h) * QSA_NCH + ch) * (QSA_D + 2);
            part[0] = m;
            part[1] = l;
        }
    }
    __syncthreads();
    // p·V: thread (dims 2j, 2j+1; cells of parity `side`), then the two halves added.
    unsigned int j = tid & 127, side = tid >> 7;
    float acc0[12], acc1[12];
    #pragma unroll
    for (int h = 0; h < 12; h++) { acc0[h] = 0.0f; acc1[h] = 0.0f; }
    for (unsigned int i = side; i < n; i += 2) {
        unsigned int vv = ((const unsigned int*)(VC + ((u64)cells[i] * QSA_KV + g) * QSA_D))[j];
        float v0 = bf(vv & 0xffff), v1 = bf(vv >> 16);
        #pragma unroll
        for (int h = 0; h < 12; h++) {
            acc0[h] = __fmaf_rn(s[h][i], v0, acc0[h]);
            acc1[h] = __fmaf_rn(s[h][i], v1, acc1[h]);
        }
    }
    if (side == 1) {
        #pragma unroll
        for (int h = 0; h < 12; h++) { half[h][2 * j] = acc0[h]; half[h][2 * j + 1] = acc1[h]; }
    }
    __syncthreads();
    if (side == 0) {
        #pragma unroll
        for (int h = 0; h < 12; h++) {
            float* part = PART + (((u64)t * QSA_HEADS + g * 12 + h) * QSA_NCH + ch) * (QSA_D + 2) + 2;
            part[2 * j] = acc0[h] + half[h][2 * j];
            part[2 * j + 1] = acc1[h] + half[h][2 * j + 1];
        }
    }
}

// ---- QSA attention over the union of a window's selections ----

// The window's union of selected blocks: block b is in it when some token selected it (BM, one
// bit per token, from fl_qsa_select) or when it holds some token's incomplete tail. Writes the
// union's blocks ascending (UB) with their masks (UM) and the count (UC[0]), and clears BM for
// the next window. Grid 1, block 1024.
extern "C" __global__ void __launch_bounds__(1024) fl_qsa_union(
    unsigned int* __restrict__ BM, const unsigned int* __restrict__ win, unsigned int T,
    unsigned int* __restrict__ UB, unsigned int* __restrict__ UM, unsigned int* __restrict__ UC) {
    __shared__ unsigned int scan[1024];
    unsigned int tid = threadIdx.x, pos0 = win[0];
    unsigned int nb_hi = (pos0 + T + 3) / 4, tail_lo = (pos0 + 1) / 4;
    unsigned int per = (nb_hi + 1023) / 1024, b0 = min(tid * per, nb_hi), b1 = min(b0 + per, nb_hi);
    unsigned int cnt = 0;
    for (unsigned int b = b0; b < b1; b++) cnt += (BM[b] != 0 || b >= tail_lo);
    unsigned int off = block_scan(cnt, scan) - cnt;
    for (unsigned int b = b0; b < b1; b++) {
        unsigned int m = BM[b];
        if (m != 0 || b >= tail_lo) {
            UB[off] = b;
            UM[off] = m;
            off++;
        }
        BM[b] = 0;
    }
    if (tid == 1023) UC[0] = off;
}

// Token t sees cell c of a union block with selection mask m.
__device__ __forceinline__ bool qsa_member(unsigned int c, unsigned int m, unsigned int t, unsigned int pos0) {
    unsigned int n_kv = pos0 + t + 1;
    return c < n_kv && (((m >> t) & 1u) || c >= (n_kv / 4) * 4);
}

// Attention over the union: grid (QSA_UCH chunks of 16 union blocks, QSA_KV), block 256. Each
// chunk's keys and values are read once, in two 32-cell tiles, for every token's 12 heads that
// share the kv head; a token's cells outside its own selection score -inf, so each token's
// softmax is over exactly its selected cells. Online softmax across the tiles; partials per
// (token, head, chunk) [m, l, acc[256]] for fl_qsa_union_merge.
#define UTILE 32
extern "C" __global__ void __launch_bounds__(256) fl_qsa_attend_union(
    const float* __restrict__ Q, const unsigned short* __restrict__ KC, const unsigned short* __restrict__ VC,
    const unsigned int* __restrict__ UB, const unsigned int* __restrict__ UM, const unsigned int* __restrict__ UC,
    const unsigned int* __restrict__ win, float* __restrict__ PART, unsigned int T, unsigned int NCHU) {
    __shared__ __align__(16) unsigned short ks[UTILE][256];
    __shared__ __align__(16) unsigned short vs[UTILE][256];
    __shared__ __align__(16) float qs[12][256];
    __shared__ float s[12][UTILE];
    __shared__ float mrow[12], lrow[12], scl[12];
    __shared__ unsigned int cells[UTILE], masks[UTILE];
    unsigned int ch = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned int nub = UC[0], pos0 = win[0];
    if (ch * 16 >= nub) return;
    for (unsigned int t = 0; t < T; t++) {
        // Online softmax state for this token's heads lives in registers (acc) and smem (m, l).
        float acc[12];
        #pragma unroll
        for (int h = 0; h < 12; h++) acc[h] = 0.0f;
        if (tid < 12) { mrow[tid] = NEG_INF; lrow[tid] = 0.0f; }
        for (unsigned int i = tid; i < 12 * 256 / 4; i += 256)
            ((float4*)&qs[0][0])[i] = ((const float4*)(Q + ((u64)t * QSA_HEADS + g * 12) * QSA_D))[i];
        for (unsigned int st = 0; st < 2; st++) {
            unsigned int ub0 = ch * 16 + st * 8;
            __syncthreads();
            if (tid < UTILE) {
                unsigned int k = ub0 + tid / 4;
                cells[tid] = k < nub ? UB[k] * 4 + tid % 4 : 0xffffffffu;
                masks[tid] = k < nub ? UM[k] : 0;
            }
            __syncthreads();
            // K and V tiles: 32 rows of 512 B each, 16 bytes a thread per step.
            for (unsigned int i = tid; i < UTILE * 32; i += 256) {
                unsigned int c = i / 32, p = i % 32, cell = cells[c];
                uint4 kz = make_uint4(0, 0, 0, 0), vz = kz;
                if (cell != 0xffffffffu) {
                    kz = ((const uint4*)(KC + ((u64)cell * QSA_KV + g) * QSA_D))[p];
                    vz = ((const uint4*)(VC + ((u64)cell * QSA_KV + g) * QSA_D))[p];
                }
                ((uint4*)&ks[c][0])[p] = kz;
                ((uint4*)&vs[c][0])[p] = vz;
            }
            __syncthreads();
            // Scores: thread (cell, head group of 1.5 heads): heads hg and hg + 8 (hg < 4).
            {
                unsigned int c = tid % UTILE, hg = tid / UTILE;
                bool mem = cells[c] != 0xffffffffu && qsa_member(cells[c], masks[c], t, pos0);
                for (unsigned int h = hg; h < 12; h += 8) {
                    float a = 0.0f;
                    for (int v = 0; v < 32; v++) {
                        uint4 kq = ((const uint4*)&ks[c][0])[v];
                        float kv[8] = {bf(kq.x & 0xffff), bf(kq.x >> 16), bf(kq.y & 0xffff), bf(kq.y >> 16),
                                       bf(kq.z & 0xffff), bf(kq.z >> 16), bf(kq.w & 0xffff), bf(kq.w >> 16)};
                        #pragma unroll
                        for (int e = 0; e < 8; e++) a = __fmaf_rn(kv[e], qs[h][8 * v + e], a);
                    }
                    s[h][c] = mem ? a * 0.0625f : NEG_INF;
                }
            }
            __syncthreads();
            // Per head (a warp each): new max, rescale, probabilities.
            for (unsigned int h = warp; h < 12; h += 8) {
                float x = s[h][lane];
                float mt = warp_max(x);
                float mo = mrow[h], mn = fmaxf(mo, mt);
                float p = mn == NEG_INF ? 0.0f : expf(x - mn);
                s[h][lane] = p;
                float lt = warp_sum(p);
                if (lane == 0) {
                    float sc = mn == NEG_INF ? 1.0f : expf(mo - mn);
                    scl[h] = sc;
                    mrow[h] = mn;
                    lrow[h] = lrow[h] * sc + lt;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int h = 0; h < 12; h++) {
                float a = acc[h] * scl[h];
                for (int c = 0; c < UTILE; c++) a = __fmaf_rn(s[h][c], bf(vs[c][tid]), a);
                acc[h] = a;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int h = 0; h < 12; h++) {
            float* part = PART + (((u64)t * QSA_HEADS + g * 12 + h) * NCHU + ch) * (QSA_D + 2);
            if (tid == 0) { part[0] = mrow[h]; part[1] = lrow[h]; }
            part[2 + tid] = acc[h];
        }
        __syncthreads();
    }
}

// Merge the union chunks per (head, token) and gate, as fl_qsa_merge. Grid (QSA_HEADS, T).
extern "C" __global__ void fl_qsa_union_merge(const float* __restrict__ PART, const float* __restrict__ P,
                                              unsigned int stride, const unsigned int* __restrict__ UC,
                                              float* __restrict__ OUT, unsigned int* __restrict__ OQ,
                                              unsigned int quant, unsigned int T, unsigned int NCHU) {
    __shared__ float f[1024];
    __shared__ float sL;
    unsigned int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    unsigned int nch = (UC[0] + 15) / 16;
    const float* part = PART + ((u64)t * QSA_HEADS + h) * NCHU * (QSA_D + 2);
    if (j < 32) {
        float M = NEG_INF;
        for (unsigned int c = j; c < nch; c += 32) M = fmaxf(M, part[c * (QSA_D + 2)]);
        M = warp_max(M);
        float L = 0.0f;
        for (unsigned int c = j; c < nch; c += 32) {
            float m = part[c * (QSA_D + 2)];
            float w = m == NEG_INF ? 0.0f : expf(m - M);
            f[c] = w;
            L += part[c * (QSA_D + 2) + 1] * w;
        }
        L = warp_sum(L);
        if (j == 0) sL = L;
    }
    __syncthreads();
    float o = 0.0f;
    for (unsigned int c = 0; c < nch; c++)
        if (f[c] != 0.0f) o = __fmaf_rn(part[c * (QSA_D + 2) + 2 + j], f[c], o);
    float gate = P[(u64)t * stride + h * 512 + 256 + j];
    float y = o / sL * sigm(gate);
    OUT[(u64)t * QSA_HEADS * QSA_D + h * QSA_D + j] = y;
    if (quant) quant_chunk(y, OQ, T, QSA_HEADS * QSA_D, t, h * 8 + (j >> 5), j & 31);
}

// Merge the chunks and apply the sigmoid gate; with `OQ`, also the gated output as int8
// activations ([QAct], m = T rows of QSA_HEADS · QSA_D). Grid (QSA_HEADS, T), block 256.
extern "C" __global__ void fl_qsa_merge(const float* __restrict__ PART, const float* __restrict__ P,
                                        unsigned int stride, const unsigned int* __restrict__ win,
                                        float* __restrict__ OUT, unsigned int* __restrict__ OQ,
                                        unsigned int quant, unsigned int T) {
    __shared__ float f[QSA_NCH];
    __shared__ float sL;
    unsigned int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    unsigned int n_sel = qsa_n_sel(win[0] + t + 1);
    unsigned int nch = (n_sel + QSA_CHUNK - 1) / QSA_CHUNK;
    const float* part = PART + ((u64)t * QSA_HEADS + h) * QSA_NCH * (QSA_D + 2);
    if (j < 32) {
        // Chunk weights exp(m_c − M) and L = Σ l_c · w_c, by one warp.
        float M = NEG_INF;
        for (unsigned int c = j; c < nch; c += 32) M = fmaxf(M, part[c * (QSA_D + 2)]);
        M = warp_max(M);
        float L = 0.0f;
        for (unsigned int c = j; c < nch; c += 32) {
            float w = expf(part[c * (QSA_D + 2)] - M);
            f[c] = w;
            L += part[c * (QSA_D + 2) + 1] * w;
        }
        L = warp_sum(L);
        if (j == 0) sL = L;
    }
    __syncthreads();
    float o = 0.0f, L = sL;
    for (unsigned int c = 0; c < nch; c++) o = __fmaf_rn(part[c * (QSA_D + 2) + 2 + j], f[c], o);
    float gate = P[(u64)t * stride + h * 512 + 256 + j];
    float y = o / L * sigm(gate);
    OUT[(u64)t * QSA_HEADS * QSA_D + h * QSA_D + j] = y;
    if (quant) quant_chunk(y, OQ, T, QSA_HEADS * QSA_D, t, h * 8 + (j >> 5), j & 31);
}
"#;
