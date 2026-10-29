// DeepSeek-V4.1-Flash device kernels, correctness first: each mirrors one op of the CPU
// reference (forward.rs / quant.rs), including where it rounds to bf16, so every kernel has a
// CPU twin to test against. Activations are f32 holding bf16-rounded values; weights stay in
// their checkpoint format and are decoded in the inner loop.
//
// Weight formats (struct DsW.fmt):
//   0 f32 [rows][cols]
//   1 bf16 [rows][cols] (u16)
//   2 fp8 e4m3 [rows][cols] + e8m0 scales [ceil(rows/32)][ceil(cols/32)]
//   3 fp4 e2m1 packed two to a byte along cols, low nibble first, + e8m0 scales [rows][cols/32]

// Programmatic dependent launch (DS_PDL=1, pdl.rs): every kernel lets its stream successor
// launch at its entry and waits for its predecessor (and its memory) before its first read;
// both are no-ops for a plain launch and below sm_90.
#ifndef DS_PDL_ENTRY
#if __CUDA_ARCH__ >= 900
#define DS_PDL_TRIGGER asm volatile("griddepcontrol.launch_dependents;");
#define DS_PDL_WAIT asm volatile("griddepcontrol.wait;" ::: "memory");
#else
#define DS_PDL_TRIGGER
#define DS_PDL_WAIT
#endif
#define DS_PDL_ENTRY DS_PDL_TRIGGER DS_PDL_WAIT
#endif


// NVRTC has no <math.h> macros
#define DS_INF __int_as_float(0x7f800000)

struct DsW {
    unsigned long long w;
    unsigned long long s;
    int rows;
    int cols;
    int fmt;
};

__device__ __forceinline__ float bf(float x) {
    unsigned int b = __float_as_uint(x);
    if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x7fffffu)) return x;  // NaN
    b += 0x7fffu + ((b >> 16) & 1u);
    return __uint_as_float(b & 0xffff0000u);
}

__device__ __forceinline__ float e4m3(unsigned char v) {
    float sign = (v & 0x80) ? -1.f : 1.f;
    int e = (v >> 3) & 0xf, m = v & 7;
    if (e == 0) return sign * (float)m * 0.001953125f;  // 2^-9
    return sign * (1.f + m / 8.f) * exp2f((float)(e - 7));
}

__device__ __forceinline__ float e8m0(unsigned char v) {
    if (v == 0) return 5.877471754111438e-39f;  // 2^-127
    return __uint_as_float(((unsigned int)v) << 23);
}

__device__ __forceinline__ float fp4(unsigned char code) {
    const float g[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = g[code & 7];
    return (code & 8) ? -m : m;
}

__device__ __forceinline__ float pow2_ceil(float x) {
    unsigned int b = __float_as_uint(x);
    unsigned int e = (b >> 23) & 0xff;
    return __uint_as_float((e + ((b & 0x7fffffu) != 0)) << 23);
}

__device__ __forceinline__ float round_e4m3(float x) {
    float a = fabsf(x);
    if (a == 0.f) return 0.f * x;
    float step;
    if (a >= 0.015625f) {  // 2^-6: three mantissa bits in this binade
        int e = (int)((__float_as_uint(a) >> 23) & 0xff) - 127;
        step = exp2f((float)(e - 3));
    } else {
        step = 0.001953125f;
    }
    float q = fminf(rintf(a / step) * step, 448.f);
    return copysignf(q, x);
}

__device__ __forceinline__ float round_fp4(float x) {
    const float g[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float a = fminf(fabsf(x), 6.f);
    int best = 0;
    for (int c = 1; c < 8; c++) {
        float d = fabsf(a - g[c]), db = fabsf(a - g[best]);
        if (d < db || (d == db && (c % 2 == 0) && (best % 2 == 1))) best = c;
    }
    return copysignf(g[best], x);
}

__device__ __forceinline__ float wget(const DsW& w, int r, int c) {
    switch (w.fmt) {
        case 0:
            return ((const float*)w.w)[(long long)r * w.cols + c];
        case 1: {
            unsigned short h = ((const unsigned short*)w.w)[(long long)r * w.cols + c];
            return __uint_as_float(((unsigned int)h) << 16);
        }
        case 2: {
            unsigned char v = ((const unsigned char*)w.w)[(long long)r * w.cols + c];
            int sc = (w.cols + 31) / 32;
            return e4m3(v) * e8m0(((const unsigned char*)w.s)[(long long)(r / 32) * sc + c / 32]);
        }
        default: {
            unsigned char v = ((const unsigned char*)w.w)[(long long)r * (w.cols / 2) + c / 2];
            unsigned char code = (c & 1) ? (v >> 4) : (v & 0xf);
            return fp4(code) * e8m0(((const unsigned char*)w.s)[(long long)r * (w.cols / 32) + c / 32]);
        }
    }
}

// y[t][n] = Σ_k x[t][k] w[n][k], one warp per output; rounded to bf16 when `round`.
// Prefetch row r of an f32/bf16 `w` into L2, the warp's lanes striding its 128-byte lines: the
// GEMV loops below then wait on L2, not DRAM, per unrolled chunk. Touches no results.
__device__ __forceinline__ void row_prefetch(const DsW& w, int r, int lane) {
    if (w.fmt > 1) return;
    long long rb = (long long)w.cols * (w.fmt == 0 ? 4 : 2);
    const char* p = (const char*)w.w + r * rb;
    for (long long o = (long long)lane * 128; o < rb; o += 32 * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p + o));
}

extern "C" __global__ void ds_linear(DsW w, const float* x, float* y, int t, int round) { DS_PDL_ENTRY
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= t * w.rows) return;
    int ti = warp / w.rows, n = warp % w.rows;
    const float* xr = x + (long long)ti * w.cols;
    float s = 0.f;
    for (int k = lane; k < w.cols; k += 32) s = fmaf(xr[k], wget(w, n, k), s);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[(long long)ti * w.rows + n] = round ? bf(s) : s;
}

// ds_linear for t <= 8 rows, one warp per output column n across all t rows: the weight row
// is read once, not t times. Bit-identical to ds_linear (same per-lane order and shuffle tree).
extern "C" __global__ void ds_linear_t8(DsW w, const float* x, float* y, int t, int round) { DS_PDL_ENTRY
    int n = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (n >= w.rows) return;
    row_prefetch(w, n, lane);
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = 0.f;
    int k = lane;
    if (w.fmt <= 1) {
        // loads for U steps issued ahead of the in-order FMAs (memory-level parallelism)
        const int U = 8;
        for (; k + 32 * (U - 1) < w.cols; k += 32 * U) {
            float wv[U];
#pragma unroll
            for (int u = 0; u < U; u++) wv[u] = wget(w, n, k + 32 * u);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                if (i >= t) break;
                const float* xr = x + (long long)i * w.cols + k;
                float xv[U];
#pragma unroll
                for (int u = 0; u < U; u++) xv[u] = xr[32 * u];
#pragma unroll
                for (int u = 0; u < U; u++) s[i] = fmaf(xv[u], wv[u], s[i]);
            }
        }
    }
    for (; k < w.cols; k += 32) {
        float wv = wget(w, n, k);
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (i < t) s[i] = fmaf(x[(long long)i * w.cols + k], wv, s[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i >= t) break;
        float v = s[i];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[(long long)i * w.rows + n] = round ? bf(v) : v;
    }
}

// The vocab head: y[i][n] = x[i]·w[n] for bf16 `w` [rows][cols], t <= 8 rows of f32 `x`,
// 4 output columns per warp (each x load feeds 4 weight rows), 16-byte loads. Each lane sums
// its 8-column runs in ascending order, then the warp's shuffle tree: every row's result is
// independent of t (a 1-token decode and a verify window agree bit for bit), but not of
// `ds_linear`'s order. Needs cols % 8 == 0 and 16-byte-aligned rows of x.
extern "C" __global__ void ds_head_bf16(const unsigned short* w, int rows, int cols,
                                        const float* x, float* y, int t) { DS_PDL_ENTRY
    int n0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 4, lane = threadIdx.x & 31;
    if (n0 >= rows) return;
    float s[4][8];
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int i = 0; i < 8; i++) s[r][i] = 0.f;
    const unsigned short* wr[4];
#pragma unroll
    for (int r = 0; r < 4; r++) wr[r] = w + (long long)min(n0 + r, rows - 1) * cols;
    int nv = cols & ~255;
#pragma unroll 2
    for (int c = lane * 8; c < nv; c += 256) {
        float wf[4][8];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            uint4 v = __ldcs((const uint4*)(wr[r] + c));
            unsigned int u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wf[r][2 * j] = __uint_as_float(u[j] << 16);
                wf[r][2 * j + 1] = __uint_as_float(u[j] & 0xffff0000u);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i >= t) break;
            const float4* xp = (const float4*)(x + (long long)i * cols + c);
            float4 a = xp[0], b = xp[1];
            float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int j = 0; j < 8; j++) s[r][i] = fmaf(xv[j], wf[r][j], s[r][i]);
        }
    }
    for (int k = nv + lane; k < cols; k += 32) {
        float wf[4];
#pragma unroll
        for (int r = 0; r < 4; r++) wf[r] = __uint_as_float(((unsigned int)wr[r][k]) << 16);
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (i < t) {
                float xv = x[(long long)i * cols + k];
#pragma unroll
                for (int r = 0; r < 4; r++) s[r][i] = fmaf(xv, wf[r], s[r][i]);
            }
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i >= t) break;
            float v = s[r][i];
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && n0 + r < rows) y[(long long)i * rows + n0 + r] = v;
        }
    }
}

// Per-row argmax of x [rows][n], one 1024-thread block per row, as the host `argmax`: the
// first index of the maximum, NaN never greater (a NaN at index 0 wins, as the host fold).
extern "C" __global__ void ds_argmax_rows(const float* x, int n, int* out) { DS_PDL_ENTRY
    const float* r = x + (long long)blockIdx.x * n;
    float bv = -DS_INF;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = r[i];
        if (v != v) v = i == 0 ? DS_INF : -DS_INF;
        if (bi == 0x7fffffff || v > bv) {
            bv = v;
            bi = i;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) {
        sv[w] = bv;
        si[w] = bi;
    }
    __syncthreads();
    if (w == 0) {
        bv = lane < nw ? sv[lane] : -DS_INF;
        bi = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) out[blockIdx.x] = bi;
    }
}

// In place: fp8 quant-dequant per 32 with power-of-two scales (warp per group, lane per value).
extern "C" __global__ void ds_fp8_qdq(float* v, int n_groups) { DS_PDL_ENTRY
    int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (g >= n_groups) return;
    float* p = v + (long long)g * 32 + lane;
    float x = *p, amax = fabsf(x);
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float s = pow2_ceil(fmaxf(amax, 1e-4f) / 448.f);
    *p = bf(round_e4m3(fminf(fmaxf(x / s, -448.f), 448.f)) * s);
}

// In place: fp4 quant-dequant per `block`, e4m3 scales when `e4m3_scale`, else power of two.
extern "C" __global__ void ds_fp4_qdq(float* v, int n_groups, int block, int e4m3_scale) { DS_PDL_ENTRY
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    float* p = v + (long long)g * block;
    float amax = 0.f;
    for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(p[i]));
    float s = e4m3_scale ? round_e4m3(fmaxf(amax, 6.f * 0.001953125f) / 6.f)
                         : pow2_ceil(fmaxf(amax, 6.f * 1.1754944e-38f) / 6.f);
    for (int i = 0; i < block; i++) p[i] = bf(round_fp4(fminf(fmaxf(p[i] / s, -6.f), 6.f)) * s);
}

// In place: RMSNorm of rows of n with weight w (any format, one row), bf16 out. Block per row.
extern "C" __global__ void ds_rms(float* x, DsW w, int n, float eps) { DS_PDL_ENTRY
    __shared__ float red[32];
    float* r = x + (long long)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    float rs = 1.f / sqrtf(red[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] = bf(wget(w, 0, i) * (r[i] * rs));
}

// In place: rotate the last rd of each d-wide vector; row r (of per_row vectors) at position
// pos0 + r * pstride, or pos_fixed when >= 0. tab is [S][rd/2] of (cos, sin). One thread per pair.
extern "C" __global__ void ds_rope(float* v, const float2* tab, int rows, int per_row, int d, int rd,
                                   int pos0, int pstride, int pos_fixed, int inverse) { DS_PDL_ENTRY
    int half = rd / 2;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * per_row * half) return;
    int j = i % half;
    long long vec = i / half;
    int r = vec / per_row;
    int pos = pos_fixed >= 0 ? pos_fixed : pos0 + r * pstride;
    float2 cs = tab[(long long)pos * half + j];
    float s = inverse ? -cs.y : cs.y;
    float* t = v + vec * d + (d - rd);
    float a = t[2 * j], b = t[2 * j + 1];
    t[2 * j] = bf(a * cs.x - b * s);
    t[2 * j + 1] = bf(a * s + b * cs.x);
}

// Gathered attention with a per-head sink: block per (t, h), d threads (d <= 1024).
// q [T][H][d], kv [n][d], idxs [T][k] (-1 = none), o [T][H][d] (bf16-rounded).
// Scores live in shared memory (k floats), so k <= smem / 4.
// Keys live in two places: row r < split of kv0, else row r - split of kv1 (a window, then a
// source layer's compressed cache), so neither is copied.
#define DS_KROW(r) ((r) < split ? kv0 + (long long)(r) * d : kv1 + (long long)((r) - split) * d)

extern "C" __global__ void ds_sparse_attn(const float* q, const float* kv0, int split, const float* kv1,
                                          const float* sink, const int* idxs, float* o, int h, int d, int k,
                                          float scale) { DS_PDL_ENTRY
    extern __shared__ float sc[];
    __shared__ float red[32];
    __shared__ float den_s;
    int t = blockIdx.x / h, hi = blockIdx.x % h;
    const float* qv = q + ((long long)t * h + hi) * d;
    const int* ix = idxs + (long long)t * k;
    for (int j = 0; j < k; j++) {
        int r = ix[j];
        float part = 0.f;
        if (r >= 0) {
            const float* kr = DS_KROW(r);
            for (int i = threadIdx.x; i < d; i += blockDim.x) part += qv[i] * kr[i];
        }
        for (int o2 = 16; o2; o2 >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o2);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int w = 0; w < (blockDim.x + 31) / 32; w++) s += red[w];
            sc[j] = r >= 0 ? s * scale : -DS_INF;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -DS_INF;
        for (int j = 0; j < k; j++) mx = fmaxf(mx, sc[j]);
        mx = fmaxf(mx, -1e30f);
        float den = 0.f;
        for (int j = 0; j < k; j++) {
            float p = expf(sc[j] - mx);
            den += p;
            sc[j] = p;
        }
        den_s = den + expf(sink[hi] - mx);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < k; j++) {
            int r = ix[j];
            if (r >= 0) acc += bf(sc[j]) * DS_KROW(r)[i];
        }
        o[((long long)t * h + hi) * d + i] = bf(acc / den_s);
    }
}

// mHC mixes for one token per thread: mixes [T][mix] (already scaled by the row's rsqrt) ->
// pre [T][hc], post [T][hc], comb [T][hc][hc]. hc <= 8.
__device__ __forceinline__ void sinkhorn_row(const float* m, const float* scale, const float* base, float* pre,
                                             float* post, float* comb, int ti, int hc, int iters, float eps) {
    float c[64];
    for (int j = 0; j < hc; j++) {
        pre[ti * hc + j] = 1.f / (1.f + expf(-(m[j] * scale[0] + base[j]))) + eps;
        post[ti * hc + j] = 2.f / (1.f + expf(-(m[hc + j] * scale[1] + base[hc + j])));
    }
    for (int i = 0; i < hc * hc; i++) c[i] = m[2 * hc + i] * scale[2] + base[2 * hc + i];
    for (int j = 0; j < hc; j++) {
        float mx = -DS_INF, s = 0.f;
        for (int k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
        for (int k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
        for (int k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
    }
    for (int it = 0; it < iters; it++) {
        if (it > 0)
            for (int j = 0; j < hc; j++) {
                float s = 0.f;
                for (int k = 0; k < hc; k++) s += c[j * hc + k];
                for (int k = 0; k < hc; k++) c[j * hc + k] /= s + eps;
            }
        for (int k = 0; k < hc; k++) {
            float s = 0.f;
            for (int j = 0; j < hc; j++) s += c[j * hc + k];
            for (int j = 0; j < hc; j++) c[j * hc + k] /= s + eps;
        }
    }
    for (int i = 0; i < hc * hc; i++) comb[(long long)ti * hc * hc + i] = c[i];
}

extern "C" __global__ void ds_sinkhorn(const float* mixes, const float* scale, const float* base,
                                       float* pre, float* post, float* comb, int t, int hc, int iters, float eps) { DS_PDL_ENTRY
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= t) return;
    int mix = (2 + hc) * hc;
    sinkhorn_row(mixes + (long long)ti * mix, scale, base, pre, post, comb, ti, hc, iters, eps);
}

// ds_hc_scale then ds_sinkhorn in one launch: block per token (blockDim a multiple of 32,
// <= 1024), the row's rsqrt as ds_hc_scale (same reduction order), the scaled mixes
// (bf-exact: one f32 product each, as ds_hc_scale stores them) staged in shared memory, then
// thread 0 runs the token's Sinkhorn. `mixes` is left unscaled.
extern "C" __global__ void ds_hc_sinkhorn(const float* x, const float* mixes, int n, float neps, const float* scale,
                                          const float* base, float* pre, float* post, float* comb, int hc, int iters,
                                          float eps) { DS_PDL_ENTRY
    __shared__ float red[32];
    __shared__ float sm[80];
    int mix = (2 + hc) * hc;
    const float* r = x + (long long)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f;
        for (int w = 0; w < blockDim.x / 32; w++) a += red[w];
        red[0] = 1.f / sqrtf(a / n + neps);
    }
    __syncthreads();
    for (int m = threadIdx.x; m < mix; m += blockDim.x) sm[m] = __fmul_rn(mixes[(long long)blockIdx.x * mix + m], red[0]);
    __syncthreads();
    if (threadIdx.x == 0) sinkhorn_row(sm, scale, base, pre, post, comb, blockIdx.x, hc, iters, eps);
}

// ---- the rest of the forward -------------------------------------------------------------

// dst [rows][w] bytes = src [rows][pitch] bytes at column c0 (a weight's column slice).
extern "C" __global__ void ds_copy2d(unsigned char* dst, const unsigned char* src, int rows, int w, int pitch,
                                     int c0) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * w) return;
    dst[i] = src[(i / w) * pitch + c0 + i % w];
}

extern "C" __global__ void ds_copy(float* dst, const float* src, long long n) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// mixes[t][m] *= rsqrt(mean(x[t]^2) + eps), x rows of n. Block per row.
extern "C" __global__ void ds_hc_scale(const float* x, float* mixes, int n, int mix, float eps) { DS_PDL_ENTRY
    __shared__ float red[32];
    const float* r = x + (long long)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f;
        for (int w = 0; w < blockDim.x / 32; w++) a += red[w];
        red[0] = 1.f / sqrtf(a / n + eps);
    }
    __syncthreads();
    for (int m = threadIdx.x; m < mix; m += blockDim.x) mixes[(long long)blockIdx.x * mix + m] *= red[0];
}

// out[t][i] = bf(Σ_c pre[t][c] x[t][c][i]). Thread per (t, i).
extern "C" __global__ void ds_hc_pre(const float* x, const float* pre, float* out, int t, int hc, int d) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * d) return;
    int ti = i / d, k = i % d;
    float s = 0.f;
    for (int c = 0; c < hc; c++) s += pre[ti * hc + c] * x[((long long)ti * hc + c) * d + k];
    out[i] = bf(s);
}

// One element of the mHC post-mix: bf(post[t][k] h[t][i] + Σ_j comb[t][j][k] res[t][j][i]), where
// h is bf(h + h2) when h2 is given (ds_add_round folded in).
__device__ __forceinline__ float hc_post_at(const float* h, const float* h2, const float* res, const float* post,
                                            const float* comb, int ti, int k, int i, int hc, int d) {
    float acc = 0.f;
    for (int j = 0; j < hc; j++) acc += comb[((long long)ti * hc + j) * hc + k] * res[((long long)ti * hc + j) * d + i];
    long long hi = (long long)ti * d + i;
    float hv = h2 ? bf(h[hi] + h2[hi]) : h[hi];
    return bf(post[ti * hc + k] * hv + acc);
}

// out[t][k][i] = hc_post_at(..). Thread per element. h2 may be null.
extern "C" __global__ void ds_hc_post(const float* h, const float* h2, const float* res, const float* post,
                                      const float* comb, float* out, int t, int hc, int d) { DS_PDL_ENTRY
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)t * hc * d) return;
    int i = e % d;
    int k = (e / d) % hc;
    int ti = e / ((long long)d * hc);
    out[e] = hc_post_at(h, h2, res, post, comb, ti, k, i, hc, d);
}

// ds_hc_post then the next mHC mixes GEMM (ds_linear / ds_linear_t8, unrounded) in one
// launch, or (post null) the GEMM alone on the stream res: block (mix row n, 8 tokens), warp
// per token, so the long per-lane chains spread over the SMs and the weight row is read once
// per block. Grid (rows, ceil(t / 8)), block 32 min(t, 8). Each lane walks k = lane +
// 32 j in order (the linear's per-lane order, then its shuffle tree), producing x[ti][k] on
// the fly; the n == 0 warps also store the stream. d % 32 == 0, hc <= 8.
extern "C" __global__ void __launch_bounds__(256) ds_hc_post_mix(const float* __restrict__ h,
                                                                 const float* __restrict__ h2,
                                                                 const float* __restrict__ res,
                                                                 const float* __restrict__ post,
                                                                 const float* __restrict__ comb, float* __restrict__ out,
                                                                 DsW w, float* __restrict__ y, int t, int hc, int d) { DS_PDL_ENTRY
    const int U = 8;
    int ti = blockIdx.y * 8 + threadIdx.x / 32, n = blockIdx.x, lane = threadIdx.x & 31;
    if (ti >= t) return;
    float s = 0.f;
    float* o = out + (long long)ti * hc * d;
    const float* rr = res + (long long)ti * hc * d;
    const float* hr = h + (long long)ti * d;
    const float* h2r = h2 ? h2 + (long long)ti * d : nullptr;
    for (int c = 0; c < hc; c++) {
        float cj[8], pc = 0.f;
        if (post) {
            pc = post[ti * hc + c];
#pragma unroll
            for (int j = 0; j < 8; j++) cj[j] = j < hc ? comb[((long long)ti * hc + j) * hc + c] : 0.f;
        }
        // x[ti][c][i] (as hc_post_at), stored by the n == 0 warps
        auto xat = [&](int i) -> float {
            if (!post) return rr[(long long)c * d + i];
            float acc = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++)
                if (j < hc) acc += cj[j] * rr[(long long)j * d + i];
            float hv = h2r ? bf(hr[i] + h2r[i]) : hr[i];
            float v = bf(pc * hv + acc);
            if (n == 0) o[(long long)c * d + i] = v;
            return v;
        };
        int i0 = lane;
        for (; i0 + 32 * (U - 1) < d; i0 += 32 * U) {
            float xv[U], wv[U];
            if (post) {
                float acc[U], hv[U];
#pragma unroll
                for (int u = 0; u < U; u++) {
                    int i = i0 + 32 * u;
                    hv[u] = h2r ? bf(hr[i] + h2r[i]) : hr[i];
                    acc[u] = 0.f;
                }
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (j >= hc) break;
#pragma unroll
                    for (int u = 0; u < U; u++) acc[u] += cj[j] * rr[(long long)j * d + i0 + 32 * u];
                }
#pragma unroll
                for (int u = 0; u < U; u++) xv[u] = bf(pc * hv[u] + acc[u]);
                if (n == 0)
#pragma unroll
                    for (int u = 0; u < U; u++) o[(long long)c * d + i0 + 32 * u] = xv[u];
            } else {
#pragma unroll
                for (int u = 0; u < U; u++) xv[u] = rr[(long long)c * d + i0 + 32 * u];
            }
#pragma unroll
            for (int u = 0; u < U; u++) wv[u] = wget(w, n, c * d + i0 + 32 * u);
#pragma unroll
            for (int u = 0; u < U; u++) s = fmaf(xv[u], wv[u], s);
        }
        for (; i0 < d; i0 += 32) s = fmaf(xat(i0), wget(w, n, c * d + i0), s);
    }
    for (int q = 16; q; q >>= 1) s += __shfl_xor_sync(0xffffffffu, s, q);
    if (lane == 0) y[(long long)ti * w.rows + n] = s;
}

// The relaxed (default-path) ds_hc_post_mix: the same stream (bit for bit) and the next
// mixes GEMM split over K, as fixed-order partials part[t][np][rows] (np = gridDim.x) that
// ds_hc_stage sums. Block (256-wide chunk of k = c d + i, up to 8 tokens): each thread makes
// one stream element per token into shared memory, then warp per mix row n reduces the chunk.
// Grid (ceil(hc d / 256), ceil(t / 8)), block 256. post null: the GEMM alone on res.
extern "C" __global__ void __launch_bounds__(256) ds_hc_mix_ks(const float* __restrict__ h,
                                                               const float* __restrict__ h2,
                                                               const float* __restrict__ res,
                                                               const float* __restrict__ post,
                                                               const float* __restrict__ comb, float* __restrict__ out,
                                                               DsW w, float* __restrict__ part, int t, int hc, int d) { DS_PDL_ENTRY
    __shared__ float xs[8][256];
    int K = hc * d, k0 = blockIdx.x * 256, t0 = blockIdx.y * 8, tid = threadIdx.x;
    int nt = min(8, t - t0), k = k0 + tid;
    if (k < K) {
        int c = k / d, i = k - c * d;
#pragma unroll
        for (int q = 0; q < 8; q++) {
            if (q >= nt) break;
            int ti = t0 + q;
            const float* rr = res + (long long)ti * K;
            float v;
            if (post) {
                float acc = 0.f;
                for (int j = 0; j < hc; j++) acc += comb[((long long)ti * hc + j) * hc + c] * rr[(long long)j * d + i];
                long long hi = (long long)ti * d + i;
                float hv = h2 ? bf(h[hi] + h2[hi]) : h[hi];
                v = bf(post[ti * hc + c] * hv + acc);
                out[(long long)ti * K + k] = v;
            } else {
                v = rr[k];
            }
            xs[q][tid] = v;
        }
    } else {
        for (int q = 0; q < nt; q++) xs[q][tid] = 0.f;
    }
    __syncthreads();
    int lane = tid & 31, np = gridDim.x;
    for (int n = tid / 32; n < w.rows; n += 8) {
        float acc[8];
#pragma unroll
        for (int q = 0; q < 8; q++) acc[q] = 0.f;
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int kk = lane + 32 * u;
            float wv = k0 + kk < K ? wget(w, n, k0 + kk) : 0.f;
#pragma unroll
            for (int q = 0; q < 8; q++) acc[q] = fmaf(xs[q][kk], wv, acc[q]);
        }
#pragma unroll
        for (int q = 0; q < 8; q++) {
            float s = acc[q];
            for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0 && q < nt) part[((long long)(t0 + q) * np + blockIdx.x) * w.rows + n] = s;
        }
    }
}

// sinkhorn_row on one warp, bit for bit: lane l holds comb element l (hc * hc <= 32); every
// row / column sum is taken in sinkhorn_row's order (each lane redoing its own row's or
// column's sum from shuffles). All 32 lanes call it.
template <int HCT, bool FAST = false>
__device__ __forceinline__ void sinkhorn_warp_t(const float* m, const float* scale, const float* base, float* pre,
                                                float* post, float* comb, int ti, int hc_, int iters, float eps) {
    // HCT > 0: hc known at compile time (the loops unrolled, the same operations in the same order)
    const int hc = HCT > 0 ? HCT : hc_;
    int l = threadIdx.x & 31, j = l / hc, k = l % hc, nn = hc * hc;
    if (l < hc) {
        pre[ti * hc + l] = 1.f / (1.f + expf(-(m[l] * scale[0] + base[l]))) + eps;
        post[ti * hc + l] = 2.f / (1.f + expf(-(m[hc + l] * scale[1] + base[hc + l])));
    }
    float c = l < nn ? m[2 * hc + l] * scale[2] + base[2 * hc + l] : 0.f;
    float mx = -DS_INF;
#pragma unroll
    for (int q = 0; q < hc; q++) mx = fmaxf(mx, __shfl_sync(0xffffffffu, c, (j * hc + q) & 31));
    c = expf(c - mx);
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < hc; q++) s += __shfl_sync(0xffffffffu, c, (j * hc + q) & 31);
    c = c / s + eps;
    for (int it = 0; it < iters; it++) {
        if (it > 0) {
            s = 0.f;
#pragma unroll
            for (int q = 0; q < hc; q++) s += __shfl_sync(0xffffffffu, c, (j * hc + q) & 31);
            c = FAST ? __fdividef(c, s + eps) : c / (s + eps);
        }
        s = 0.f;
#pragma unroll
        for (int q = 0; q < hc; q++) s += __shfl_sync(0xffffffffu, c, (q * hc + k) & 31);
        c = FAST ? __fdividef(c, s + eps) : c / (s + eps);
    }
    if (l < nn) comb[(long long)ti * nn + l] = c;
}

__device__ __forceinline__ void sinkhorn_warp(const float* m, const float* scale, const float* base, float* pre,
                                              float* post, float* comb, int ti, int hc, int iters, float eps) {
    if (hc == 4)
        sinkhorn_warp_t<4>(m, scale, base, pre, post, comb, ti, hc, iters, eps);
    else
        sinkhorn_warp_t<0>(m, scale, base, pre, post, comb, ti, hc, iters, eps);
}

// The rest of an mHC stage in one launch, block (1024 threads) per token: ds_hc_sinkhorn on
// the stream x (its rsqrt reduced as that kernel's 256 threads do) -> pre_o/post/comb, and
// ds_hc_pre(x, pre_i) then ds_rms(w) (1024 threads, its reduction order) -> out.
extern "C" __global__ void ds_hc_stage(const float* x, const float* mixes, float neps, const float* scale,
                                       const float* base, float* pre_o, float* post, float* comb, int hc, int iters,
                                       float eps, const float* pre_i, float* out, DsW w, int d, float reps,
                                       int np) { DS_PDL_ENTRY
    __shared__ float ra[32], rb[32], sm[80];
    int mix = (2 + hc) * hc, n = hc * d, tid = threadIdx.x;
    const float* r = x + (long long)blockIdx.x * n;
    if (tid < 256) {
        float s = 0.f;
        for (int i = tid; i < n; i += 256) s += r[i] * r[i];
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if ((tid & 31) == 0) ra[tid / 32] = s;
    }
    float* orow = out + (long long)blockIdx.x * d;
    const float* pr = pre_i + blockIdx.x * hc;
    float s = 0.f;
    for (int i = tid; i < d; i += blockDim.x) {
        float v = 0.f;
        for (int c = 0; c < hc; c++) v += pr[c] * r[(long long)c * d + i];
        v = bf(v);
        orow[i] = v;
    }
    // the stored row read back, as ds_rms reads it
    for (int i = tid; i < d; i += blockDim.x) s += orow[i] * orow[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((tid & 31) == 0) rb[tid / 32] = s;
    __syncthreads();
    if (tid < 32) {
        s = tid < blockDim.x / 32 ? rb[tid] : 0.f;
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (tid == 0) rb[0] = s;
    }
    __syncthreads();
    float rs = 1.f / sqrtf(rb[0] / d + reps);
    for (int i = tid; i < d; i += blockDim.x) orow[i] = bf(wget(w, 0, i) * (orow[i] * rs));
    if (tid < 32) {
        float a = 0.f;
        for (int q = 0; q < 8; q++) a += ra[q];
        float rr = 1.f / sqrtf(a / n + neps);
        for (int m = tid; m < mix; m += 32) {
            // np == 1: the exact GEMM's row as is; else its K-split partials, in order
            const float* mp = mixes + (long long)blockIdx.x * np * mix + m;
            float mv = mp[0];
#pragma unroll 8
            for (int q = 1; q < np; q++) mv += mp[(long long)q * mix];
            sm[m] = __fmul_rn(mv, rr);
        }
        __syncwarp();
        if (hc * hc <= 32)
            sinkhorn_warp(sm, scale, base, pre_o, post, comb, blockIdx.x, hc, iters, eps);
        else if (tid == 0)
            sinkhorn_row(sm, scale, base, pre_o, post, comb, blockIdx.x, hc, iters, eps);
    }
}

// A whole relaxed mHC stage in one launch: ds_hc_post (post null: the stream res as is),
// the stage's mixes GEMM K-split, hc_pre(stream, pre_i) and the stream / block-input square
// sums as per-block partials; then a grid barrier (t <= 8, so one row of co-resident
// blocks; cnt two zeroed ints, left consistent for the next launch), after which every
// block scales its block-input chunk by rms(nw) (partials summed in a fixed order) and
// block b runs the exact Sinkhorn (sinkhorn_warp) of tokens b + np j. Block 256 = hc copies x
// ci = 256 / hc dims; grid ceil(d / ci). part [t][np][rows + 2].
// The body of ds_hc_fused_ks for block (bx, by) of np_ in x, shared with mg_glue (which also
// passes qc/qsc: the block input's K-split quantization folded into its store, as ds_ks_quant).
__device__ __forceinline__ unsigned int e4m3_byte(float v);
__device__ __forceinline__ void hc_fused_dev(
    const float* __restrict__ h, const float* __restrict__ h2, const float* __restrict__ res,
    const float* __restrict__ post, const float* __restrict__ comb, float* __restrict__ xo, DsW w,
    float* part, const float* __restrict__ pre_i, float* hin, DsW nw, const float* __restrict__ scale,
    const float* __restrict__ base, float* __restrict__ pre_o, float* __restrict__ post_o,
    float* __restrict__ comb_o, int* cnt, int t, int hc, int d, int iters, float eps, float neps, float reps, int stub, int bx, int by, int np_, unsigned char* qc, float* qsc) {
    // stub (timing only, the hc_glue bench): 1 no mixes GEMM, 2 no grid barrier, 4 no partial
    // sums, 8 no Sinkhorn, 16 no stream loads
    __shared__ float xs[8][256];
    __shared__ float rx[8][8], ro[8][8], sm[8][80], rs[8];
    int ci = 256 / hc, tid = threadIdx.x, lane = tid & 31, wp = tid / 32;
    int i0 = bx * ci, t0 = by * 8, nt = min(8, t - t0), np = np_;
    int c = tid / ci, i = i0 + tid % ci, R = w.rows, K = hc * d;
    bool ok = c < hc && i < d;
    // this warp's first 3 mixes rows (n = wp + 8 r) prefetched, overlapping the stream loads
    float wpre[3][8];
#pragma unroll
    for (int r = 0; r < 3; r++)
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int n = wp + 8 * r, e = lane + 32 * u, cc = e / ci, ii = i0 + e % ci;
            wpre[r][u] = n < R && cc < hc && ii < d ? wget(w, n, cc * d + ii) : 0.f;
        }
    float sx[8];
#pragma unroll
    for (int q = 0; q < 8; q++) {
        float v = 0.f;
        if (ok && q < nt && !(stub & 16)) {
            int ti = t0 + q;
            const float* rr = res + (long long)ti * K;
            if (post) {
                float acc = 0.f;
                for (int j = 0; j < hc; j++) acc += comb[((long long)ti * hc + j) * hc + c] * rr[(long long)j * d + i];
                long long hi = (long long)ti * d + i;
                float hv = h2 ? bf(h[hi] + h2[hi]) : h[hi];
                v = bf(post[ti * hc + c] * hv + acc);
                xo[(long long)ti * K + (long long)c * d + i] = v;
            } else {
                v = rr[(long long)c * d + i];
            }
        }
        xs[q][tid] = v;
        sx[q] = v * v;
    }
    __syncthreads();
    // hc_pre on the chunk (threads < ci), and the square sums
    float so[8], hv[8];
#pragma unroll
    for (int q = 0; q < 8; q++) {
        so[q] = 0.f;
        hv[q] = 0.f;
        if (tid < ci && i0 + tid < d && q < nt) {
            int ti = t0 + q;
            float v = 0.f;
            for (int cc = 0; cc < hc; cc++) v += pre_i[ti * hc + cc] * xs[q][cc * ci + tid];
            v = bf(v);
            hv[q] = v;
            so[q] = v * v;
        }
    }
#pragma unroll
    for (int q = 0; q < 8; q++) {
        float a = sx[q], b = so[q];
        for (int o = 16; o; o >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, o);
            b += __shfl_xor_sync(0xffffffffu, b, o);
        }
        if (lane == 0) {
            rx[q][wp] = a;
            ro[q][wp] = b;
        }
    }
    // the mixes GEMM on the chunk: warp per row
#pragma unroll
    for (int r = 0; r < 10; r++) {
        int n = wp + 8 * r;
        if (n >= R || (stub & 1)) break;
        float acc[8];
#pragma unroll
        for (int q = 0; q < 8; q++) acc[q] = 0.f;
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int e = lane + 32 * u, cc = e / ci, ii = i0 + e % ci;
            float wv = r < 3 ? wpre[r < 3 ? r : 0][u] : (cc < hc && ii < d ? wget(w, n, cc * d + ii) : 0.f);
#pragma unroll
            for (int q = 0; q < 8; q++) acc[q] = fmaf(xs[q][e], wv, acc[q]);
        }
        // the 8 tokens' warp sums in 9 shuffles (butterfly, halving the values each step);
        // lane l ends with token ((l >> 4) & 1) * 4 + ((l >> 3) & 1) * 2 + ((l >> 2) & 1)
        float b4[4], b2[2];
        bool u16 = lane & 16, u8 = lane & 8, u4 = lane & 4;
#pragma unroll
        for (int q = 0; q < 4; q++)
            b4[q] = (u16 ? acc[q + 4] : acc[q]) + __shfl_xor_sync(0xffffffffu, u16 ? acc[q] : acc[q + 4], 16);
#pragma unroll
        for (int q = 0; q < 2; q++)
            b2[q] = (u8 ? b4[q + 2] : b4[q]) + __shfl_xor_sync(0xffffffffu, u8 ? b4[q] : b4[q + 2], 8);
        float sq = (u4 ? b2[1] : b2[0]) + __shfl_xor_sync(0xffffffffu, u4 ? b2[0] : b2[1], 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        int qo = (u16 ? 4 : 0) + (u8 ? 2 : 0) + (u4 ? 1 : 0);
        if ((lane & 3) == 0 && qo < nt) part[((long long)(t0 + qo) * np + bx) * (R + 2) + n] = sq;
    }
    __syncthreads();
    if (tid < nt) {
        float a = 0.f, b = 0.f;
        for (int q = 0; q < 8; q++) {
            a += rx[tid][q];
            b += ro[tid][q];
        }
        float* pp = part + ((long long)(t0 + tid) * np + bx) * (R + 2) + R;
        pp[0] = a;
        pp[1] = b;
    }
    // grid barrier (t <= 8: one row of blocks, all co-resident): cnt[0] arrivals, cnt[1]
    // the generation
    __syncthreads();
    if (tid == 0 && !(stub & 2)) {
        volatile int* gen = cnt + 1;
        int g0 = *gen;
        __threadfence();
        if (atomicAdd(cnt, 1) == np - 1) {
            cnt[0] = 0;
            __threadfence();
            atomicAdd(cnt + 1, 1);
        } else {
            while (*gen == g0) __nanosleep(64);
        }
        __threadfence();
    }
    __syncthreads();
    // thread per value, the partials summed in block order: every block the block inputs'
    // square sums, block b also the mixes and stream square sums of tokens b + np j
    int own = bx < nt ? (nt - 1 - bx) / np + 1 : 0;
    for (int e = tid; e < (stub & 4 ? 0 : nt + own * (R + 1)); e += 256) {
        int q = e < nt ? e : bx + np * ((e - nt) / (R + 1));
        int m = e < nt ? R + 1 : (e - nt) % (R + 1);
        const float* pp = part + (long long)(t0 + q) * np * (R + 2) + m;
        float v = 0.f;
#pragma unroll 16
        for (int b = 0; b < np; b++) v += __ldcg(pp + (long long)b * (R + 2));
        if (m < R)
            sm[q][m] = v;
        else if (m == R)
            rx[q][0] = 1.f / sqrtf(v / K + neps);
        else
            rs[q] = 1.f / sqrtf(v / d + reps);
    }
    __syncthreads();
    if (tid < ci && i0 + tid < d) {
        float wv = wget(nw, 0, i0 + tid);
#pragma unroll
        for (int q = 0; q < 8; q++)
            if (q < nt) {
                long long o = (long long)(t0 + q) * d + i0 + tid;
                float xv = bf(wv * (hv[q] * rs[q]));
                hin[o] = xv;
                if (qc) {
                    // ds_ks_quant of the 32-group o / 32 (mg_glue: ci % 32 == 0, d % ci == 0)
                    float am = fabsf(xv);
#pragma unroll
                    for (int k = 16; k; k >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, k));
                    float sc = pow2_ceil(fmaxf(am, 1e-4f) / 448.f);
                    qc[o] = (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(xv / sc, -448.f), 448.f)));
                    if (lane == 0) qsc[o / 32] = sc;
                }
            }
    }
    if (wp < own && !(stub & 8)) {
        int q = bx + np * wp, ti = t0 + q;
        for (int m = lane; m < R; m += 32) sm[q][m] = __fmul_rn(sm[q][m], rx[q][0]);
        __syncwarp();
        // relaxed path: the Sinkhorn's divisions as __fdividef (~2 ulp of f32, far under bf16)
        if (hc == 4)
            sinkhorn_warp_t<4, true>(sm[q], scale, base, pre_o, post_o, comb_o, ti, hc, iters, eps);
        else if (hc * hc <= 32)
            sinkhorn_warp(sm[q], scale, base, pre_o, post_o, comb_o, ti, hc, iters, eps);
        else if (lane == 0)
            sinkhorn_row(sm[q], scale, base, pre_o, post_o, comb_o, ti, hc, iters, eps);
    }
}


extern "C" __global__ void __launch_bounds__(256) ds_hc_fused_ks(
    const float* __restrict__ h, const float* __restrict__ h2, const float* __restrict__ res,
    const float* __restrict__ post, const float* __restrict__ comb, float* __restrict__ xo, DsW w,
    float* part, const float* __restrict__ pre_i, float* hin, DsW nw, const float* __restrict__ scale,
    const float* __restrict__ base, float* __restrict__ pre_o, float* __restrict__ post_o,
    float* __restrict__ comb_o, int* cnt, int t, int hc, int d, int iters, float eps, float neps, float reps, int stub) { DS_PDL_ENTRY
    hc_fused_dev(h, h2, res, post, comb, xo, w, part, pre_i, hin, nw, scale, base, pre_o, post_o, comb_o, cnt, t, hc, d, iters, eps, neps, reps, stub, blockIdx.x, blockIdx.y, gridDim.x, 0, 0);
}

// out[t][c][i] = table[ids[t]][i] for every copy c (bf16 table). Thread per element.
extern "C" __global__ void ds_embed(const unsigned short* table, const int* ids, float* out, int t, int copies, int d) { DS_PDL_ENTRY
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)t * copies * d) return;
    int i = e % d;
    int ti = e / ((long long)d * copies);
    out[e] = __uint_as_float(((unsigned int)table[(long long)ids[ti] * d + i]) << 16);
}

// out[t][i] = bf(mean_c x[t][c][i]).
extern "C" __global__ void ds_mean_hc(const float* x, float* out, int t, int hc, int d) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * d) return;
    int ti = i / d, k = i % d;
    float s = 0.f;
    for (int c = 0; c < hc; c++) s += x[((long long)ti * hc + c) * d + k];
    out[i] = bf(s / hc);
}

__device__ __forceinline__ float swiglu1(float gv, float uv, const float* w, float lim) {
    if (lim > 0.f) {
        gv = fminf(gv, lim);
        uv = fminf(fmaxf(uv, -lim), lim);
    }
    float v = gv / (1.f + expf(-gv)) * uv;
    return bf(w ? *w * v : v);
}

// h[r][j] = bf(w[r] * silu(min(g, lim)) * clamp(u, ±lim)); w null for the shared expert.
extern "C" __global__ void ds_swiglu(const float* g, const float* u, const float* w, float* h, long long n, int inter, float lim) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = swiglu1(g[i], u[i], w ? w + i / inter : 0, lim);
}

// The router, one block per token: raw [t][ne] gate logits (f32) -> ids/weights [t][k].
// sqrt(softplus) scores, top-k of score + bias (ties to the lower id), weights normalized and
// scaled. k <= 16; dynamic shared memory: ne floats (score + bias) + ne bytes (used flags).
// Each round is a block argmax; the weight sum runs serially in selection order, as before.
__device__ __forceinline__ bool router_better(float v, int e, float bv, int be) {
    return be < 0 || (e >= 0 && (v > bv || (v == bv && e < be)));
}

extern "C" __global__ void ds_router(const float* raw, const float* bias, int* ids, float* wts, int t, int ne, int k,
                                     int norm, float route_scale) { DS_PDL_ENTRY
    extern __shared__ float sh[];
    float* sv = sh;
    unsigned char* used = (unsigned char*)(sh + ne);
    __shared__ float rv[32];
    __shared__ int re[32];
    __shared__ int sel[16];
    int ti = blockIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    const float* r = raw + (long long)ti * ne;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        float x = r[e];
        float sc = sqrtf(x > 20.f ? x : log1pf(expf(x)));
        sv[e] = sc + bias[e];
        used[e] = 0;
    }
    __syncthreads();
    for (int j = 0; j < k; j++) {
        int be = -1;
        float bv = 0.f;
        for (int e = threadIdx.x; e < ne; e += blockDim.x)
            if (!used[e] && router_better(sv[e], e, bv, be)) { be = e; bv = sv[e]; }
        for (int o = 16; o; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (router_better(ov, oe, bv, be)) { be = oe; bv = ov; }
        }
        if (lane == 0) { rv[wid] = bv; re[wid] = be; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < nw; q++)
                if (router_better(rv[q], re[q], bv, be)) { be = re[q]; bv = rv[q]; }
            sel[j] = be;
            used[be] = 1;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float w[16], sum = 0.f;
        for (int j = 0; j < k; j++) {
            float x = r[sel[j]];
            w[j] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
            sum += w[j];
        }
        for (int j = 0; j < k; j++) {
            float v = (norm && k > 1) ? w[j] / (sum + 1e-20f) : w[j];
            ids[ti * k + j] = sel[j];
            wts[ti * k + j] = v * route_scale;
        }
    }
}

// out[j][i] = x[rows[j]][i].
extern "C" __global__ void ds_gather_rows(const float* x, const int* rows, float* out, int n, int d) { DS_PDL_ENTRY
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)n * d) return;
    out[e] = x[(long long)rows[e / d] * d + e % d];
}

// y[rows[j]][i] += v[j][i] (rows unique within one launch).
extern "C" __global__ void ds_scatter_add(float* y, const float* v, const int* rows, int n, int d) { DS_PDL_ENTRY
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)n * d) return;
    y[(long long)rows[e / d] * d + e % d] += v[e];
}

// y[i] += v[i].
extern "C" __global__ void ds_add(float* y, const float* v, long long n) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += v[i];
}

// y[i] = bf(y[i] + v[i]).
extern "C" __global__ void ds_add_round(float* y, const float* v, long long n) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf(y[i] + v[i]);
}

// v[i] = bf(v[i] * s).
extern "C" __global__ void ds_scale_round(float* v, float s, long long n) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = bf(v[i] * s);
}

// v[i] = bf(v[i]).
extern "C" __global__ void ds_round(float* v, long long n) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = bf(v[i]);
}

// The ratio-r compressor's pooling: out[g][c] = Σ_j kv[g*r+j][c] softmax_j(sc[g*r+j][c]).
extern "C" __global__ void ds_pool(const float* kv, const float* sc, float* out, int groups, int r, int hd) { DS_PDL_ENTRY
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)groups * hd) return;
    int g = e / hd, c = e % hd;
    float mx = -DS_INF;
    for (int j = 0; j < r; j++) mx = fmaxf(mx, sc[((long long)g * r + j) * hd + c]);
    float s = 0.f;
    for (int j = 0; j < r; j++) s += expf(sc[((long long)g * r + j) * hd + c] - mx);
    float acc = 0.f;
    for (int j = 0; j < r; j++) {
        long long o = ((long long)g * r + j) * hd + c;
        acc += kv[o] * (expf(sc[o] - mx) / s);
    }
    out[e] = acc;
}

// The ratio-r compressor batched over a step's t positions P[0] + e (block e, 1024 threads as
// ds_rms): ds_pool of the ring as it stands once position e's row is written (slot j holds the
// batch's latest row at a position <= P[0] + e that is j mod r, else the old ring's), ds_round,
// then ds_rms(nw), into lat[e]. Bit for bit the per-position ring_write + pool + round + rms.
extern "C" __global__ void __launch_bounds__(1024) ds_comp_pool(const float* kst, const float* sst, const float* kv,
                                                                 const float* sc, float* lat, DsW nw, const int* P,
                                                                 int r, int hd, float eps) { DS_PDL_ENTRY
    __shared__ float red[32];
    int e = blockIdx.x, p0 = P[0], pe = p0 + e;
    float* out = lat + (long long)e * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float mx = -DS_INF;
        for (int j = 0; j < r; j++) {
            int q = pe - (pe - j) % r;
            float sv = q >= p0 ? sc[(long long)(q - p0) * hd + c] : sst[(long long)j * hd + c];
            mx = fmaxf(mx, sv);
        }
        float s = 0.f;
        for (int j = 0; j < r; j++) {
            int q = pe - (pe - j) % r;
            float sv = q >= p0 ? sc[(long long)(q - p0) * hd + c] : sst[(long long)j * hd + c];
            s += expf(sv - mx);
        }
        float acc = 0.f;
        for (int j = 0; j < r; j++) {
            int q = pe - (pe - j) % r;
            bool nw_ = q >= p0;
            float sv = nw_ ? sc[(long long)(q - p0) * hd + c] : sst[(long long)j * hd + c];
            float kv_ = nw_ ? kv[(long long)(q - p0) * hd + c] : kst[(long long)j * hd + c];
            acc += kv_ * (expf(sv - mx) / s);
        }
        out[c] = bf(acc);
    }
    __syncthreads();
    float s = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) s += out[i] * out[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    float rs = 1.f / sqrtf(red[0] / hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) out[i] = bf(wget(nw, 0, i) * (out[i] * rs));
}

// The rows a batch of t leaves in two rings of r slots (slot (P[0] + e) % r): only each slot's
// last write, so the batch's rows never race.
extern "C" __global__ void ds_ring_last2(float* ra, const float* a, float* rb, const float* b, int t, int hd, int r,
                                         const int* P) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * hd) return;
    int e = i / hd, c = i % hd;
    if (e + r < t) return;
    long long o = (long long)((P[0] + e) % r) * hd + c;
    ra[o] = a[i];
    rb[o] = b[i];
}

// A step's compressed rows, batched (block (e, job), 1024 threads): job 0 the latent lat[e]
// (hd), job 1 the indexer key ik[e] (ihd, after ds_rms(kn)); rope at max(P[0] + e + 1 - r, 0)
// (dg_rope's fixed_r), fp4 qdq (latent: blocks of 16, e4m3 scale; key: 32, pow2) as
// ds_fp4_qdq, into cache row (P[0] + e) / r, by the batch's last row for that cache row.
extern "C" __global__ void __launch_bounds__(1024) ds_comp_post(float* lat, int hd, float* ckv, float* ik, int ihd,
                                                                 DsW kn, float* idxk, const float2* tab, int rd,
                                                                 const int* P, int r, int t, float eps) { DS_PDL_ENTRY
    __shared__ float red[32];
    int e = blockIdx.x, job = blockIdx.y, p0 = P[0];
    int n = job ? ihd : hd, blk = job ? 32 : 16;
    float* x = job ? ik + (long long)e * ihd : lat + (long long)e * hd;
    if (job) {
        float s = 0.f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
        __syncthreads();
        if (threadIdx.x < 32) {
            s = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
            for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (threadIdx.x == 0) red[0] = s;
        }
        __syncthreads();
        float rs = 1.f / sqrtf(red[0] / n + eps);
        for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = bf(wget(kn, 0, i) * (x[i] * rs));
        __syncthreads();
    }
    int half = rd / 2, pos = max(p0 + e + 1 - r, 0);
    float* tp = x + (n - rd);
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float2 cs = tab[(long long)pos * half + j];
        float a = tp[2 * j], b = tp[2 * j + 1];
        tp[2 * j] = bf(a * cs.x - b * cs.y);
        tp[2 * j + 1] = bf(a * cs.y + b * cs.x);
    }
    __syncthreads();
    for (int g = threadIdx.x; g < n / blk; g += blockDim.x) {
        float* p = x + (long long)g * blk;
        float amax = 0.f;
        for (int i = 0; i < blk; i++) amax = fmaxf(amax, fabsf(p[i]));
        float s = job ? pow2_ceil(fmaxf(amax, 6.f * 1.1754944e-38f) / 6.f)
                      : round_e4m3(fmaxf(amax, 6.f * 0.001953125f) / 6.f);
        for (int i = 0; i < blk; i++) p[i] = bf(round_fp4(fminf(fmaxf(p[i] / s, -6.f), 6.f)) * s);
    }
    __syncthreads();
    if (e != t - 1 && (p0 + e + 1) % r != 0) return;
    float* dst = (job ? idxk : ckv) + (long long)((p0 + e) / r) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = x[i];
}

// The indexer's scores, warp per (t, j): score = bf(Σ_h bf(relu(bf(q_h·k_j)) * w[t][h])).
extern "C" __global__ void ds_index_score(const float* q, const float* ik, const float* w, float* score,
                                          int t, int width, int nh, int ihd) { DS_PDL_ENTRY
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= t * width) return;
    int ti = warp / width, j = warp % width;
    const float* kj = ik + (long long)j * ihd;
    float s = 0.f;
    for (int h = 0; h < nh; h++) {
        const float* qv = q + ((long long)ti * nh + h) * ihd;
        float d = 0.f;
        for (int i = lane; i < ihd; i += 32) d += qv[i] * kj[i];
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        s += bf(fmaxf(bf(d), 0.f) * w[ti * nh + h]);
    }
    if (lane == 0) score[warp] = bf(s);
}

// wo_a, block-diagonal over groups: mid[t][g][r] = bf(Σ_i o[t][g][i] w[g*ol + r][i]).
// Every token at once (up to DS_WO_T a pass), so each weight is read and decoded once rather
// than once a token. Each output's sum runs in the same order as a warp per output
// (lane-strided, then the xor tree), so the result is unchanged.
#define DS_WO_T 8
__device__ __forceinline__ float ds_fp8w(const unsigned char* wr, const unsigned char* sr, int i, int m) {
    // the e4m3 decode built from its bits (the same float as e4m3()); the column block's
    // scale is warp-uniform (i / 32 = m)
    unsigned int v = wr[i];
    int e = (v >> 3) & 0xf;
    float dv = e ? __uint_as_float(((v & 0x80u) << 24) | ((unsigned int)(e + 120) << 23) | ((v & 7u) << 20))
                 : ((v & 0x80u) ? -1.f : 1.f) * (float)(v & 7u) * 0.001953125f;
    return bf(dv * e8m0(sr[m]));
}

// Block of 256 threads per DS_WO_ROWS consecutive rows of one group (ol % DS_WO_ROWS == 0):
// the tokens' activations staged in shared memory 256 columns at a time, read by every warp.
#define DS_WO_ROWS 16
#define DS_WO_TC 256
extern "C" __global__ void __launch_bounds__(256) ds_wo_a(const float* o, DsW w, float* mid, int t, int g, int ol) { DS_PDL_ENTRY
    __shared__ float xs[DS_WO_T][DS_WO_TC];
    int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
    int base = blockIdx.x * DS_WO_ROWS, gi = base / ol;
    const int RW = DS_WO_ROWS / 8;  // rows a warp
    int r0 = base - gi * ol + wi * RW;
    int gd = w.cols, nsc = (gd + 31) / 32;
    long long ts = (long long)g * gd;
    for (int t0 = 0; t0 < t; t0 += DS_WO_T) {
        int nt = min(DS_WO_T, t - t0);
        float s[RW][DS_WO_T];
#pragma unroll
        for (int j = 0; j < RW; j++)
#pragma unroll
            for (int u = 0; u < DS_WO_T; u++) s[j][u] = 0.f;
        const float* ov = o + ((long long)t0 * g + gi) * gd;
        for (int c0 = 0; c0 < gd; c0 += DS_WO_TC) {
            __syncthreads();
            for (int e = threadIdx.x; e < DS_WO_T * DS_WO_TC; e += 256) {
                int u = e / DS_WO_TC, c = c0 + e % DS_WO_TC;
                xs[u][e % DS_WO_TC] = u < nt && c < gd ? ov[u * ts + c] : 0.f;
            }
            __syncthreads();
#pragma unroll 2
            for (int mm = 0; mm < DS_WO_TC / 32; mm++) {
                int i = c0 + mm * 32 + lane, m = i / 32;
                if (c0 + mm * 32 >= gd) break;
                float x[DS_WO_T];
#pragma unroll
                for (int u = 0; u < DS_WO_T; u++) x[u] = xs[u][mm * 32 + lane];
                bool in = i < gd;
#pragma unroll
                for (int j = 0; j < RW; j++) {
                    int row = gi * ol + r0 + j;
                    float wv = !in ? 0.f
                               : w.fmt == 2
                                   ? ds_fp8w((const unsigned char*)w.w + (long long)row * gd,
                                             (const unsigned char*)w.s + (long long)(row / 32) * nsc, i, m)
                                   : bf(wget(w, row, i));
                    if (in)
#pragma unroll
                        for (int u = 0; u < DS_WO_T; u++)
                            if (u < nt) s[j][u] += x[u] * wv;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < RW; j++)
#pragma unroll
            for (int u = 0; u < DS_WO_T; u++) {
                if (u >= nt) break;
                float v = s[j][u];
                for (int o2 = 16; o2; o2 >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o2);
                if (lane == 0) mid[((long long)(t0 + u) * g + gi) * ol + r0 + j] = bf(v);
            }
    }
}

// out[b][a][c] = in[a][b][c] for in [na][nb][nc]: the wo_a groups' rows to and from
// per-group contiguous blocks for the K-split GEMV.
extern "C" __global__ void ds_perm3(const float* in, float* out, int na, int nb, int nc) { DS_PDL_ENTRY
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)na * nb * nc) return;
    int c = e % nc, b = (e / nc) % nb, a = e / ((long long)nb * nc);
    out[((long long)b * na + a) * nc + c] = in[e];
}

// Engram's gate, block per (t, c): out = bf(h + sigmoid(signed sqrt(normalized h·qk·key)) val).
// kv [t][(hc + 1) d] (keys, then the shared value), qk [hc][d] = q_weight * k_weight.
extern "C" __global__ void ds_engram_gate(const float* x, const float* kv, const float* qk, float* out,
                                          int hc, int d, float eps) { DS_PDL_ENTRY
    __shared__ float red[3][32];
    int ti = blockIdx.x / hc, c = blockIdx.x % hc;
    const float* h = x + ((long long)ti * hc + c) * d;
    const float* key = kv + (long long)ti * (hc + 1) * d + (long long)c * d;
    const float* val = kv + (long long)ti * (hc + 1) * d + (long long)hc * d;
    float hh = 0.f, kk = 0.f, dot = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        hh += h[i] * h[i];
        kk += key[i] * key[i];
        dot += h[i] * qk[c * d + i] * key[i];
    }
    for (int o = 16; o; o >>= 1) {
        hh += __shfl_xor_sync(0xffffffffu, hh, o);
        kk += __shfl_xor_sync(0xffffffffu, kk, o);
        dot += __shfl_xor_sync(0xffffffffu, dot, o);
    }
    if ((threadIdx.x & 31) == 0) { red[0][threadIdx.x / 32] = hh; red[1][threadIdx.x / 32] = kk; red[2][threadIdx.x / 32] = dot; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f, e = 0.f;
        for (int w = 0; w < blockDim.x / 32; w++) { a += red[0][w]; b += red[1][w]; e += red[2][w]; }
        float rstd = 1.f / sqrtf(a / d + eps) * (1.f / sqrtf(b / d + eps));
        float dt = e * rstd * powf((float)d, -0.5f);
        float gs = sqrtf(fmaxf(fabsf(dt), 1e-6f));
        red[0][0] = 1.f / (1.f + expf(-copysignf(gs, dt)));
    }
    __syncthreads();
    float gate = red[0][0];
    float* o = out + ((long long)ti * hc + c) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = bf(h[i] + gate * val[i]);
}

// ---- fast GEMV (decode, t <= 8 per launch) for fp8 (fmt 2) and fp4 (fmt 3) weights ----
//
// Bit-identical to ds_linear: lane l of the warp owning a row accumulates x[k] * w[k] for
// k = l, l + 32, ... in increasing k with fmaf, then the same xor butterfly. The weights are
// read with one 16-byte load per lane (32 fp4 / 16 fp8 values), decoded in registers and
// transposed through shared memory so each lane gets its k = l (mod 32) column. Each warp owns
// LR = 4 weight rows (8 lanes load each row's consecutive 128 bytes per tile), so one
// activation read serves 4 rows, and all t <= TM token rows share each weight read.
// `qdq` fuses ds_fp8_qdq: each 32-group is exactly one warp-wide column step, so its amax is a
// warp max and the result is bit-identical to quantizing first.
// GU: rows 0,1 come from w (w1) and rows 2,3 from w3 at the same output rows; the epilogue
// emits h = swiglu1(bf(g), bf(u)) (ds_swiglu), `rw` the per-token route weights or null.

#define LR 4
#define LU 4  // tiles in flight per lane

__device__ __forceinline__ float fp4v(unsigned int c) {
    unsigned int m = c & 7, e = m >> 1;
    unsigned int b = e ? (((e + 126u) << 23) | ((m & 1u) << 22)) : ((m & 1u) ? (126u << 23) : 0u);
    return __uint_as_float(b | ((c & 8u) << 28));
}

__device__ __forceinline__ float e4m3v(unsigned int v) {
    unsigned int e = (v >> 3) & 0xf, m = v & 7;
    float a = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    return (v & 0x80) ? -a : a;
}

template <int FMT, int TM, int GU>
__device__ __forceinline__ void lin_fast(DsW w, DsW w3, const float* __restrict__ x, float* __restrict__ y, int t,
                                         int round, int qdq, const float* __restrict__ rw, float lim,
                                         const int* __restrict__ xi) {
    // chunks (32 columns) per tile per row: 8 lanes x 16 bytes
    const int CT = FMT == 3 ? 8 : 4;
    __shared__ float sh[4][LR][8][33];
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float(*S)[8][33] = sh[wib];
    const int OUT = GU ? LR / 2 : LR;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + wib) * OUT;
    int cols = w.cols, nch = cols / 32;
    int r = lane >> 3, sub = lane & 7;
    // the row this lane loads
    int nr = GU ? n0 + (r & 1) : n0 + r;
    const DsW& wr = (GU && r >= 2) ? w3 : w;
    bool rv = nr < w.rows;
    long long rbytes = FMT == 3 ? cols / 2 : cols;
    const unsigned char* wp = (const unsigned char*)wr.w + (rv ? (long long)nr * rbytes : 0);
    const unsigned char* sp = (const unsigned char*)wr.s;
    // x row of token ti: xi[ti] when gathering (grouped MoE), else ti
    const float* xr[TM];
#pragma unroll
    for (int ti = 0; ti < TM; ti++) xr[ti] = x + (long long)(xi && ti < t ? xi[ti] : ti) * cols;
    float acc[LR][TM];
#pragma unroll
    for (int i = 0; i < LR; i++)
#pragma unroll
        for (int j = 0; j < TM; j++) acc[i][j] = 0.f;
    int ntiles = (nch + CT - 1) / CT;
    for (int tg = 0; tg < ntiles; tg += LU) {
        uint4 buf[LU];
        float scl[LU];
#pragma unroll
        for (int u = 0; u < LU; u++) {
            int tile = tg + u;
            // fp4: this lane's chunk; fp8: this lane's half chunk
            int c = FMT == 3 ? tile * 8 + sub : (tile * 8 + sub) >> 1;
            bool ok = rv && tile < ntiles && c < nch;
            buf[u] = make_uint4(0, 0, 0, 0);
            scl[u] = 0.f;
            if (ok) {
                buf[u] = __ldg((const uint4*)(wp + (long long)(tile * 8 + sub) * 16));
                unsigned char sv = FMT == 3 ? __ldg(sp + (long long)nr * nch + c) : __ldg(sp + (long long)(nr / 32) * nch + c);
                scl[u] = e8m0(sv);
            }
        }
#pragma unroll
        for (int u = 0; u < LU; u++) {
            int tile = tg + u;
            if (tile >= ntiles) break;
            unsigned int q[4] = {buf[u].x, buf[u].y, buf[u].z, buf[u].w};
            if (FMT == 3) {
                float* d = &S[r][sub][0];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int b = 0; b < 8; b++) d[i * 8 + b] = fp4v((q[i] >> (4 * b)) & 15u) * scl[u];
            } else {
                float* d = &S[r][sub >> 1][(sub & 1) * 16];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int b = 0; b < 4; b++) d[i * 4 + b] = e4m3v((q[i] >> (8 * b)) & 255u) * scl[u];
            }
            __syncwarp();
            int nj = min(CT, nch - tile * CT);
            for (int j = 0; j < nj; j++) {
                int k = (tile * CT + j) * 32 + lane;
#pragma unroll
                for (int ti = 0; ti < TM; ti++) {
                    if (ti < t) {
                        float xv = xr[ti][k];
                        if (qdq) {
                            float a = fabsf(xv);
#pragma unroll
                            for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
                            float s = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
                            xv = bf(round_e4m3(fminf(fmaxf(xv / s, -448.f), 448.f)) * s);
                        }
#pragma unroll
                        for (int i = 0; i < LR; i++) acc[i][ti] = fmaf(xv, S[i][j][lane], acc[i][ti]);
                    }
                }
            }
            __syncwarp();
        }
    }
#pragma unroll
    for (int i = 0; i < LR; i++)
#pragma unroll
        for (int ti = 0; ti < TM; ti++)
#pragma unroll
            for (int o = 16; o; o >>= 1) acc[i][ti] += __shfl_xor_sync(0xffffffffu, acc[i][ti], o);
    if (lane != 0) return;
#pragma unroll
    for (int i = 0; i < OUT; i++) {
        int n = n0 + i;
        if (n >= w.rows) break;
#pragma unroll
        for (int ti = 0; ti < TM; ti++) {
            if (ti >= t) break;
            if (GU) {
                y[(long long)ti * w.rows + n] = swiglu1(bf(acc[i][ti]), bf(acc[i + 2][ti]), rw ? rw + ti : 0, lim);
            } else {
                float s = acc[i][ti];
                y[(long long)ti * w.rows + n] = round ? bf(s) : s;
            }
        }
    }
}

#define DS_LIN(F, T, G)                                                                                       \
    extern "C" __global__ void __launch_bounds__(128) ds_lin_f##F##_t##T##_g##G(                             \
        DsW w, DsW w3, const float* x, float* y, int t, int round, int qdq, const float* rw, float lim) { DS_PDL_ENTRY     \
        lin_fast<F, T, G>(w, w3, x, y, t, round, qdq, rw, lim, 0);                                            \
    }
#define DS_LIN_T(F, G) DS_LIN(F, 1, G) DS_LIN(F, 2, G) DS_LIN(F, 4, G) DS_LIN(F, 8, G)
DS_LIN_T(2, 0)
DS_LIN_T(3, 0)
DS_LIN_T(2, 1)
DS_LIN_T(3, 1)

// ---- grouped MoE: one launch per projection for every routed expert of a layer ----
//
// blockIdx.y is a slot of `elist` (ds_moe_plan: the live local experts ascending, -1 past
// them); blockIdx.x tiles the weight rows exactly as ds_lin_*. Expert e's rows are its CSR
// segment off[e]..off[e+1] (ds_route), at most T of them. Both are lin_fast, bit-identical to
// the per-expert ds_linear_csr loop: GU gathers its x rows through `rows` (xq: x already
// fp8-quantized) into h [t*k][inter] at the CSR positions, with SwiGLU and the route weights;
// down maps hq (h quantized) to per-slot partials part [t*k][d], summed by ds_moe_reduce.
#define DS_MOE(F, T)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) ds_moe_gu_f##F##_t##T(                                 \
        const DsW* w1s, const DsW* w3s, const int* elist, const int* off, const int* rows, const float* rw,  \
        const float* xq, float* h, float lim) { DS_PDL_ENTRY                                                               \
        int e = elist[blockIdx.y];                                                                            \
        if (e < 0) return;                                                                                    \
        int o = off[e], n = off[e + 1] - o;                                                                   \
        DsW w = w1s[e];                                                                                       \
        lin_fast<F, T, 1>(w, w3s[e], xq, h + (long long)o * w.rows, n, 1, 0, rw + o, lim, rows + o);          \
    }                                                                                                         \
    extern "C" __global__ void __launch_bounds__(128) ds_moe_down_f##F##_t##T(                               \
        const DsW* w2s, const int* elist, const int* off, const float* hq, float* part) { DS_PDL_ENTRY                    \
        int e = elist[blockIdx.y];                                                                            \
        if (e < 0) return;                                                                                    \
        int o = off[e], n = off[e + 1] - o;                                                                   \
        DsW w = w2s[e];                                                                                       \
        lin_fast<F, T, 0>(w, w, hq + (long long)o * w.cols, part + (long long)o * w.rows, n, 1, 0, 0, 0.f, 0); \
    }
#define DS_MOE_T(F) DS_MOE(F, 1) DS_MOE(F, 2) DS_MOE(F, 4) DS_MOE(F, 8)
DS_MOE_T(2)
DS_MOE_T(3)

// ---- tensor-core K-split GEMV (the default fast path; DS_EXACT=1 selects lin_fast above) ----
//
// Not bit-exact with ds_linear (the reduction order differs); outputs agree with it to within
// one bf16 ulp in rare elements. The activation is fp8-quantized per 32-group while it is
// staged in shared memory (bit-identical to ds_fp8_qdq; idempotent on an already quantized
// activation), and kept as its e4m3 value in f16 plus the group scale. Weights decode with a
// few integer ops to f16 (e4m3 is f16 × 2^8, e2m1 is f16 × 2^14, both exact), so each 32-column
// group of a 16-row × 8-token tile is two mma.m16n8k16 (f16 in, f32 accumulate), scaled by
// weight scale × activation scale into f32 accumulators. A warp owns 16 weight rows (GU: rows
// 0-7 of w1 and the same rows of w3); a block of 8 warps is RW = 8 / S row tiles × S K-slices,
// the slices meeting in shared memory in a fixed order (deterministic). Within an mma's k step
// columns are permuted freely (A and B agree), so each lane's weight bytes are contiguous.

#ifndef KS_UG
#define KS_UG 4  // groups in flight per warp (4 beats 8 on H100: shorter ragged tails per item)
#endif

__device__ __forceinline__ unsigned short f2h_exact(float v) {
    unsigned int b = __float_as_uint(v);
    unsigned int sg = (b >> 16) & 0x8000u, a = b & 0x7fffffffu;
    if (a == 0) return (unsigned short)sg;
    return (unsigned short)(sg | (((a >> 23) - 112u) << 10) | ((a >> 13) & 0x3ffu));
}

// two e4m3 in bits 8-15 and 24-31 -> f16x2 of value / 2^8
__device__ __forceinline__ unsigned int f8h(unsigned int u) { return (u & 0x80008000u) | ((u >> 1) & 0x3f803f80u); }
// two e2m1 in bits 12-15 and 28-31 -> f16x2 of value / 2^14
__device__ __forceinline__ unsigned int f4h(unsigned int u) { return (u & 0x80008000u) | ((u >> 3) & 0x0e000e00u); }

__device__ __forceinline__ void mma16816(float* c, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
                                         unsigned int b0, unsigned int b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// __CUDA_ARCH__ of the module, so the host can fall back to the exact kernels below sm_80.
extern "C" __global__ void ds_arch(int* out) { DS_PDL_ENTRY *out = __CUDA_ARCH__; }

// e4m3-valued float -> its e4m3 byte (|v| <= 448, on the grid)
__device__ __forceinline__ unsigned int e4m3_byte(float v) {
    unsigned int b = __float_as_uint(v), sg = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    if (a == 0) return sg;
    int e = (int)(a >> 23) - 127;
    if (e < -6) return sg | (unsigned int)(__uint_as_float(a) * 512.f);  // subnormal: m * 2^-9
    return sg | ((unsigned int)(e + 7) << 3) | ((a >> 20) & 7u);
}

// The activation, quantized once: codes [t][K] (e4m3 bytes) and scales [t][K/32] (pow2), bit
// for bit ds_fp8_qdq's. Warp per 32-group.
extern "C" __global__ void ds_ks_quant(const float* x, unsigned char* codes, float* sx, int n_groups) { DS_PDL_ENTRY
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= n_groups) return;
    float xv = x[(long long)gw * 32 + lane];
    float a = fabsf(xv);
#pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
    codes[(long long)gw * 32 + lane] = (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(xv / s, -448.f), 448.f)));
    if (lane == 0) sx[gw] = s;
}

// Persistent blocks: each stages the whole quantized activation ([t][K] codes, row pitch
// P = K + 32 bytes for conflict-free 8-byte reads, and scales) once, then takes work items
// (row group br, K slice sb) = blockIdx.x, += gridDim.x. With SB > 1 slices, partials go to
// part[sb][t][rows (× 2 for GU)] and the last block of a row group (cnt) sums them in order.
template <int FMT, int GU, int UG = KS_UG>
__device__ __forceinline__ void lin_mma(DsW w, DsW w3, const unsigned char* __restrict__ xc,
                                        const float* __restrict__ xsc, float* __restrict__ y, int t, int round,
                                        const float* __restrict__ rw, float lim, int S, int SB, int nbr, float* part,
                                        int* cnt, const int* __restrict__ xi, int xst = 0, int yst = 0,
                                        int it0 = -1, bool staged = false) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float red[8][4][32];
    __shared__ int last;
    int cols = w.cols, nch = cols / 32, P = ((cols + 127) / 128) * 128 + 32;
    long long xrs = xst ? xst : cols, srs = xst ? xst / 32 : nch, yrs = yst ? yst : w.rows;
    unsigned char* xs = smem;           // [t][P]
    float* sx = (float*)(smem + t * P);  // [t][nch], scale × 2^8 (code -> f16) × weight POW
    const float POW = (FMT == 3 ? 16384.f : 256.f) * 256.f;
    // staged: this block already holds the same x (ds_ks2's later items)
    for (int e = threadIdx.x; e < (staged ? 0 : t * cols / 16); e += blockDim.x) {
        int ti = e / (cols / 16), c = e % (cols / 16);
        long long xr = xi ? __ldg(xi + ti) : ti;
        *(uint4*)(xs + ti * P + c * 16) = __ldg((const uint4*)(xc + xr * xrs) + c);
    }
    for (int e = threadIdx.x; e < (staged ? 0 : t * nch); e += blockDim.x) {
        int ti = e / nch, c = e % nch;
        long long xr = xi ? __ldg(xi + ti) : ti;
        sx[e] = __ldg(xsc + xr * srs + c) * POW;
    }
    __syncthreads();
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    int RW = 8 / S, rwi = wib / S, si = wib % S;
    const int OUT = GU ? 8 : 16;
    const int R2 = GU ? 2 * w.rows : w.rows;
    long long rbytes = FMT == 3 ? cols / 2 : cols;
    int gpb = (nch + SB - 1) / SB;
    // it0 >= 0: this block takes that one item (the many-row prefill grid)
    for (int item = it0 >= 0 ? it0 : blockIdx.x; item < nbr * SB; item += it0 >= 0 ? nbr * SB : gridDim.x) {
        int br = item / SB, sb = item % SB;
        int n0 = (br * RW + rwi) * OUT;
        int ra = n0 + g, rb = GU ? n0 + g : n0 + g + 8;
        bool va = ra < w.rows, vb = rb < w.rows;
        const unsigned char* pa =
            (const unsigned char*)w.w + (va ? (long long)ra * rbytes : 0) + (FMT == 3 ? 4 : 8) * tig;
        const unsigned char* pb =
            (const unsigned char*)(GU ? w3.w : w.w) + (vb ? (long long)rb * rbytes : 0) + (FMT == 3 ? 4 : 8) * tig;
        const unsigned char* sa =
            (const unsigned char*)w.s + (FMT == 3 ? (long long)ra * nch : (long long)(ra / 32) * nch);
        const unsigned char* sbp =
            (const unsigned char*)(GU ? w3.s : w.s) + (FMT == 3 ? (long long)rb * nch : (long long)(rb / 32) * nch);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        int glo = sb * gpb, ghi = min(nch, glo + gpb);
        for (int q0 = glo + si; q0 < ghi; q0 += UG * S) {
            uint2 la[UG], lb[UG];
            unsigned char ca[UG], cb[UG];
#pragma unroll
            for (int u = 0; u < UG; u++) {
                int G = q0 + u * S;
                la[u] = lb[u] = make_uint2(0, 0);
                ca[u] = cb[u] = 0;
                if (G < ghi) {
                    if (va) {
                        if (FMT == 3) la[u].x = __ldg((const unsigned int*)(pa + (long long)G * 16));
                        else la[u] = __ldg((const uint2*)(pa + (long long)G * 32));
                        ca[u] = __ldg(sa + G);
                    }
                    if (vb) {
                        if (FMT == 3) lb[u].x = __ldg((const unsigned int*)(pb + (long long)G * 16));
                        else lb[u] = __ldg((const uint2*)(pb + (long long)G * 32));
                        cb[u] = __ldg(sbp + G);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < UG; u++) {
                int G = q0 + u * S;
                if (G >= ghi) break;
                uint2 X = g < t ? *(const uint2*)(xs + g * P + G * 32 + 8 * tig) : make_uint2(0, 0);
                float c[4] = {0.f, 0.f, 0.f, 0.f};
                if (FMT == 3) {
                    unsigned int wa = la[u].x, wb = lb[u].x;
                    mma16816(c, f4h(wa << 12), f4h(wb << 12), f4h(wa << 8), f4h(wb << 8),
                             f8h(__byte_perm(X.x, X.y, 0x4000)), f8h(__byte_perm(X.x, X.y, 0x5010)));
                    mma16816(c, f4h(wa << 4), f4h(wb << 4), f4h(wa), f4h(wb), f8h(__byte_perm(X.x, X.y, 0x6020)),
                             f8h(__byte_perm(X.x, X.y, 0x7030)));
                } else {
                    mma16816(c, f8h(la[u].x << 8), f8h(lb[u].x << 8), f8h(la[u].x), f8h(lb[u].x), f8h(X.x << 8),
                             f8h(X.x));
                    mma16816(c, f8h(la[u].y << 8), f8h(lb[u].y << 8), f8h(la[u].y), f8h(lb[u].y), f8h(X.y << 8),
                             f8h(X.y));
                }
                float x0 = 2 * tig < t ? sx[2 * tig * nch + G] : 0.f;
                float x1 = 2 * tig + 1 < t ? sx[(2 * tig + 1) * nch + G] : 0.f;
                float swa = e8m0(ca[u]), swb = e8m0(cb[u]);
                acc[0] = fmaf(c[0], swa * x0, acc[0]);
                acc[1] = fmaf(c[1], swa * x1, acc[1]);
                acc[2] = fmaf(c[2], swb * x0, acc[2]);
                acc[3] = fmaf(c[3], swb * x1, acc[3]);
            }
        }
        __syncthreads();  // red reuse across items
#pragma unroll
        for (int i = 0; i < 4; i++) red[wib][i][lane] = acc[i];
        __syncthreads();
        if (SB > 1) {
            for (int e = threadIdx.x; e < RW * OUT * t; e += blockDim.x) {
                int j = e % t, o = (e / t) % OUT, rg = e / (t * OUT);
                int n = (br * RW + rg) * OUT + o;
                if (n >= w.rows) continue;
                int ln = (o & 7) * 4 + j / 2, i = j & 1;
                float a = 0.f, b = 0.f;
                for (int s = 0; s < S; s++) {
                    a += red[rg * S + s][GU || o < 8 ? i : 2 + i][ln];
                    if (GU) b += red[rg * S + s][2 + i][ln];
                }
                float* pp = part + ((long long)sb * t + j) * R2;
                pp[n] = a;
                if (GU) pp[w.rows + n] = b;
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                last = atomicAdd(&cnt[br], 1) == SB - 1;
                if (last) cnt[br] = 0;
            }
            __syncthreads();
            if (!last) continue;
            __threadfence();
        }
        for (int e = threadIdx.x; e < RW * OUT * t; e += blockDim.x) {
            int j = e % t, o = (e / t) % OUT, rg = e / (t * OUT);
            int n = (br * RW + rg) * OUT + o;
            if (n >= w.rows) continue;
            // C fragment: (row g, token 2tig + i) in acc[i], (row g + 8, ...) in acc[2 + i]
            int ln = (o & 7) * 4 + j / 2, i = j & 1;
            float a = 0.f, b = 0.f;
            if (SB > 1) {
                for (int q = 0; q < SB; q++) {
                    const float* pp = part + ((long long)q * t + j) * R2;
                    a += __ldcg(pp + n);
                    if (GU) b += __ldcg(pp + w.rows + n);
                }
            } else {
                for (int s = 0; s < S; s++) {
                    a += red[rg * S + s][GU || o < 8 ? i : 2 + i][ln];
                    if (GU) b += red[rg * S + s][2 + i][ln];
                }
            }
            if (GU) y[(long long)j * yrs + n] = swiglu1(bf(a), bf(b), rw ? rw + j : 0, lim);
            else y[(long long)j * yrs + n] = round ? bf(a) : a;
        }
    }
}

// PDL prologue of ds_ks: L2 prefetches of this block's first work item's weight slice (its
// rows of w, and w3's for GU), issued before DS_PDL_WAIT so they stream under the predecessor.
// Only compiled with PDL on (DS_PDL_PF, set by Kernels::new): without PDL it costs wq_b 24.5 ->
// 28.1 us and wo_b 36.4 -> 39.8 at t = 6 on H100.
template <int FMT, int GU>
__device__ __forceinline__ void ks_pf(const DsW& w, const DsW& w3, int S, int SB) {
#if __CUDA_ARCH__ >= 900 && defined(DS_PDL_PF)
    int nch = w.cols / 32, gpb = (nch + SB - 1) / SB;
    int br = blockIdx.x / SB, sb = blockIdx.x % SB;
    int nr = (8 / S) * (GU ? 8 : 16), n0 = br * nr;
    long long rbytes = FMT == 3 ? w.cols / 2 : w.cols, gb = FMT == 3 ? 16 : 32;
    long long lo = sb * gpb * gb, hi = min(nch, (sb + 1) * gpb) * gb;
    int lines = (int)((hi - lo + 127) / 128);
    for (int e = threadIdx.x; e < nr * (1 + GU) * lines; e += blockDim.x) {
        int r = e / lines % nr, m = e / lines / nr, l = e % lines;
        if (n0 + r >= w.rows) continue;
        const char* base = (const char*)(m ? w3.w : w.w) + (long long)(n0 + r) * rbytes + lo + 128LL * l;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(base));
    }
#endif
}

#define DS_KS(F, G)                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) ds_ks_f##F##_g##G(                                        \
        DsW w, DsW w3, const unsigned char* xc, const float* xsc, float* y, int t, int round, const float* rw,  \
        float lim, int S, int SB, int nbr, float* part, int* cnt) { DS_PDL_TRIGGER                                             \
        ks_pf<F, G>(w, w3, S, SB); DS_PDL_WAIT                                                                  \
        lin_mma<F, G>(w, w3, xc, xsc, y, t, round, rw, lim, S, SB, nbr, part, cnt, 0);                           \
    }
// The K-split GEMV over t > 8 rows in one launch (prefill): block b is row group b / nc of
// 8-row chunk b % nc (chunks fastest, so a row group's weights are read from L2 by the nc
// blocks after the first), SB = 1, each row's result that of the t <= 8 kernel on its chunk.
// xc/xsc: the whole activation quantized once.
#define DS_KST(F, G)                                                                                            \
    extern "C" __global__ void __launch_bounds__(256) ds_kst_f##F##_g##G(                                       \
        DsW w, DsW w3, const unsigned char* xc, const float* xsc, float* y, int t, int round, const float* rw,  \
        float lim, int S, int nbr) { DS_PDL_ENTRY                                                                            \
        int nc = (t + 7) / 8, c = blockIdx.x % nc, br = blockIdx.x / nc, r0 = c * 8;                          \
        lin_mma<F, G>(w, w3, xc + (long long)r0 * w.cols, xsc + (long long)r0 * (w.cols / 32),                 \
                      y + (long long)r0 * w.rows, min(8, t - r0), round, rw ? rw + r0 : 0, lim, S, 1, nbr, 0, 0, \
                      0, 0, 0, br);                                                                               \
    }
DS_KST(2, 0)
DS_KST(3, 0)
DS_KST(2, 1)
DS_KST(3, 1)
// Two K-split GEMVs on the same quantized x in one launch (wq_a and wkv): items 0 .. na - 1 are
// w's (S, SB, nbr, part, cnt), the rest wb's; x staged once per block. Each output as its own
// ds_ks_f*_g0 launch, bit for bit.
#define DS_KS2(F)                                                                                               \
    extern "C" __global__ void __launch_bounds__(256) ds_ks2_f##F(                                              \
        DsW w, DsW wb, const unsigned char* xc, const float* xsc, float* y, float* yb, int t, int round, int S,  \
        int SB, int nbr, int Sb, int SBb, int nbrb, float* part, float* partb, int* cnt, int* cntb) {           \
        int na = nbr * SB, nt = na + nbrb * SBb;                                                                \
        for (int item = blockIdx.x; item < nt; item += gridDim.x) {                                            \
            bool st = item != blockIdx.x;                                                                       \
            if (item < na)                                                                                      \
                lin_mma<F, 0>(w, w, xc, xsc, y, t, round, 0, 0.f, S, SB, nbr, part, cnt, 0, 0, 0, item, st);    \
            else                                                                                                \
                lin_mma<F, 0>(wb, wb, xc, xsc, yb, t, round, 0, 0.f, Sb, SBb, nbrb, partb, cntb, 0, 0, 0,       \
                              item - na, st);                                                                   \
        }                                                                                                       \
    }
DS_KS2(2)
DS_KS2(3)
DS_KS(2, 0)
DS_KS(3, 0)
DS_KS(2, 1)
DS_KS(3, 1)

#ifndef KS_MOE_MINB
#define KS_MOE_MINB 4  // the grouped K-split MoE kernels' min blocks an SM (register cap)
#endif
// Grouped MoE on the K-split path: the DS_MOE contract (blockIdx.y a slot of `elist`, expert
// e's rows its CSR segment off[e]..off[e+1], at most 8) with lin_mma per slot, SB = 1 (no
// cross-block partials: the slots give the parallelism) and blockIdx.x striding the expert's
// nbr row groups. The activation codes/scales come from ds_ks_quant: GU gathers the token rows
// of xc [t][d] through `rows`; down reads hq's codes [t*k][inter] at the segment.
#define DS_MOE_KS(F)                                                                                           \
    extern "C" __global__ void __launch_bounds__(256, KS_MOE_MINB) ds_moe_ks_gu_f##F(                                       \
        const DsW* w1s, const DsW* w3s, const int* elist, const int* off, const int* rows, const float* rw,    \
        const unsigned char* xc, const float* xsc, float* h, float lim, int S, int nbr) { DS_PDL_ENTRY                      \
        int e = elist[blockIdx.y];                                                                              \
        if (e < 0) return;                                                                                      \
        int o = off[e], n = off[e + 1] - o;                                                                     \
        if (n <= 0) return;                                                                                     \
        DsW w = w1s[e];                                                                                         \
        lin_mma<F, 1>(w, w3s[e], xc, xsc, h + (long long)o * w.rows, n, 1, rw + o, lim, S, 1, nbr, 0, 0,       \
                      rows + o);                                                                                \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(256, KS_MOE_MINB) ds_moe_ks_down_f##F(                                     \
        const DsW* w2s, const int* elist, const int* off, const unsigned char* hc, const float* hsc,           \
        float* part, int S, int nbr) { DS_PDL_ENTRY                                                                          \
        int e = elist[blockIdx.y];                                                                              \
        if (e < 0) return;                                                                                      \
        int o = off[e], n = off[e + 1] - o;                                                                     \
        if (n <= 0) return;                                                                                     \
        DsW w = w2s[e];                                                                                         \
        lin_mma<F, 0>(w, w, hc + (long long)o * w.cols, hsc + (long long)o * (w.cols / 32),                    \
                      part + (long long)o * w.rows, n, 1, 0, 0.f, S, 1, nbr, 0, 0, 0);                          \
    }
DS_MOE_KS(2)
DS_MOE_KS(3)

// The grouped K-split MoE with the shared expert as one more slot (blockIdx.y == gridDim.y - 1
// when sh != 0): every token routed to it at weight 1 (no rw), its own format SF (the shared
// expert is fp8 where the routed ones are fp4), GU over x's codes in token order into
// h[n + ti], down from hc rows n.. into part[n + ti] for ds_moe_reduce's `sh` term. The
// routed slots are ds_moe_ks_*_f F's exactly.
// The shared expert's row items when its gate/up is this rank's slice of `rows` (8 a warp tile).
// `shs` (h's row stride, the full intermediate) places a slice's h rows where the full
// expert's would be, so the down's K slice reads them at the same pitch.
__device__ __forceinline__ int moe_sh_nbr(int rows, int S) { return (rows + 8 * (8 / S) - 1) / (8 * (8 / S)); }
#define DS_MOE_KS_SH(F, SF)                                                                                    \
    extern "C" __global__ void __launch_bounds__(256, KS_MOE_MINB) ds_moe_ks_gu_f##F##_s##SF(                              \
        const DsW* w1s, const DsW* w3s, const int* elist, const int* off, const int* rows, const float* rw,    \
        const unsigned char* xc, const float* xsc, float* h, float lim, int S, int nbr, DsW s1, DsW s3, int t, \
        int n, int shs) { DS_PDL_ENTRY                                                                          \
        if (t > 0 && blockIdx.y == gridDim.y - 1) {                                                             \
            lin_mma<SF, 1>(s1, s3, xc, xsc, h + (long long)n * shs, t, 1, 0, lim, S, 1, moe_sh_nbr(s1.rows, S), \
                           0, 0, 0, 0, shs);                                                                    \
            return;                                                                                             \
        }                                                                                                       \
        int e = elist[blockIdx.y];                                                                              \
        if (e < 0) return;                                                                                      \
        int o = off[e], m = off[e + 1] - o;                                                                     \
        if (m <= 0) return;                                                                                     \
        DsW w = w1s[e];                                                                                         \
        lin_mma<F, 1>(w, w3s[e], xc, xsc, h + (long long)o * w.rows, m, 1, rw + o, lim, S, 1, nbr, 0, 0,       \
                      rows + o);                                                                                \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(256, KS_MOE_MINB) ds_moe_ks_down_f##F##_s##SF(                            \
        const DsW* w2s, const int* elist, const int* off, const unsigned char* hc, const float* hsc,           \
        float* part, int S, int nbr, DsW s2, int t, int n, int shs) { DS_PDL_ENTRY                              \
        if (t > 0 && blockIdx.y == gridDim.y - 1) {                                                             \
            lin_mma<SF, 0>(s2, s2, hc + (long long)n * shs, hsc + (long long)n * (shs / 32),                    \
                           part + (long long)n * s2.rows, t, 1, 0, 0.f, S, 1, nbr, 0, 0, 0, shs);               \
            return;                                                                                             \
        }                                                                                                       \
        int e = elist[blockIdx.y];                                                                              \
        if (e < 0) return;                                                                                      \
        int o = off[e], m = off[e + 1] - o;                                                                     \
        if (m <= 0) return;                                                                                     \
        DsW w = w2s[e];                                                                                         \
        lin_mma<F, 0>(w, w, hc + (long long)o * w.cols, hsc + (long long)o * (w.cols / 32),                    \
                      part + (long long)o * w.rows, m, 1, 0, 0.f, S, 1, nbr, 0, 0, 0);                          \
    }
DS_MOE_KS_SH(3, 2)
DS_MOE_KS_SH(2, 2)
DS_MOE_KS_SH(3, 3)
DS_MOE_KS_SH(2, 3)

// The persistent form of ds_moe_ks_*_s (DS_MOE_KP): a 1-D grid sized to fill the GPU once. The
// work units are (slot, row item): the shared expert's nbr items first (cost 2: fp8, all t
// rows), then each live slot's (cost 1). Each block takes a contiguous cost range, so it
// mostly stays on one slot and restages the activation only when the slot changes; lin_mma's
// arithmetic per item is the grid kernels' exactly (bit-identical). Live slots: elist entries
// >= 0 with rows, in elist order, found by warp 0 (cap <= 64).
__device__ __forceinline__ int moe_kp_live(const int* elist, const int* off, int cap, int* sl) {
    __shared__ int nl;
    if (threadIdx.x < 32) {
        int base = 0;
        for (int j0 = 0; j0 < cap; j0 += 32) {
            int j = j0 + threadIdx.x, e = j < cap ? elist[j] : -1;
            bool live = e >= 0 && off[e + 1] > off[e];
            unsigned b = __ballot_sync(0xffffffffu, live);
            if (live) sl[base + __popc(b & ((1u << threadIdx.x) - 1))] = e;
            base += __popc(b);
        }
        if (threadIdx.x == 0) nl = base;
    }
    __syncthreads();
    return nl;
}
// block b's unit range [lo, hi): su shared units at cost cs, then the routed ones at cost cr
__device__ __forceinline__ void moe_kp_range(int su, int units, int cs, int cr, int* lo, int* hi) {
    long long S0 = (long long)cs * su, C = S0 + (long long)cr * (units - su), G = gridDim.x, b = blockIdx.x;
    auto at = [&](long long c) -> int {
        return c <= S0 ? (int)((c + cs - 1) / cs) : (int)(su + (c - S0 + cr - 1) / cr);
    };
    *lo = at(b * C / G);
    *hi = min(units, at((b + 1) * C / G));
}
#define DS_MOE_KP(F, SF)                                                                                       \
    extern "C" __global__ void __launch_bounds__(256, KS_MOE_MINB) ds_moe_kp_gu_f##F##_s##SF(                  \
        const DsW* w1s, const DsW* w3s, const int* elist, const int* off, const int* rows, const float* rw,    \
        const unsigned char* xc, const float* xsc, float* h, float lim, int S, int nbr, DsW s1, DsW s3, int t, \
        int n, int shs, int cap) { DS_PDL_ENTRY                                                                 \
        __shared__ int sl[64];                                                                                  \
        int nl = moe_kp_live(elist, off, cap, sl), nbs = moe_sh_nbr(s1.rows, S), su = t > 0 ? nbs : 0, lo, hi;  \
        /* cost an item: its weight bytes (fp8 2, fp4 1) */                                                     \
        moe_kp_range(su, su + nl * nbr, SF == 3 ? 1 : 2, F == 3 ? 1 : 2, &lo, &hi);                             \
        int prev = -2;                                                                                          \
        for (int u = lo; u < hi; u++) {                                                                         \
            int sidx = u < su ? -1 : (u - su) / nbr, item = u < su ? u : (u - su) % nbr;                        \
            bool staged = sidx == prev;                                                                         \
            prev = sidx;                                                                                        \
            if (sidx < 0) {                                                                                     \
                lin_mma<SF, 1>(s1, s3, xc, xsc, h + (long long)n * shs, t, 1, 0, lim, S, 1, nbs, 0, 0, 0, 0,    \
                               shs, item, staged);                                                              \
                continue;                                                                                       \
            }                                                                                                   \
            int e = sl[sidx], o = off[e], m = off[e + 1] - o;                                                   \
            DsW w = w1s[e];                                                                                     \
            lin_mma<F, 1>(w, w3s[e], xc, xsc, h + (long long)o * w.rows, m, 1, rw + o, lim, S, 1, nbr, 0, 0,   \
                          rows + o, 0, 0, item, staged);                                                        \
        }                                                                                                       \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(256, KS_MOE_MINB) ds_moe_kp_down_f##F##_s##SF(                \
        const DsW* w2s, const int* elist, const int* off, const unsigned char* hc, const float* hsc,           \
        float* part, int S, int nbr, DsW s2, int t, int n, int shs, int cap) { DS_PDL_ENTRY                     \
        __shared__ int sl[64];                                                                                  \
        int nl = moe_kp_live(elist, off, cap, sl), su = t > 0 ? nbr : 0, lo, hi;                                \
        /* cost an item: its weight bytes, shared (its K slice, SF) vs routed (F) */                           \
        int kr = nl > 0 ? w2s[sl[0]].cols : s2.cols;                                                            \
        int cs = max(1, (int)((8LL * s2.cols * (SF == 3 ? 1 : 2)) / ((long long)kr * (F == 3 ? 1 : 2))));      \
        moe_kp_range(su, su + nl * nbr, cs, 8, &lo, &hi);                                                       \
        int prev = -2;                                                                                          \
        for (int u = lo; u < hi; u++) {                                                                         \
            int sidx = u < su ? -1 : (u - su) / nbr, item = u < su ? u : (u - su) % nbr;                        \
            bool staged = sidx == prev;                                                                         \
            prev = sidx;                                                                                        \
            if (sidx < 0) {                                                                                     \
                lin_mma<SF, 0>(s2, s2, hc + (long long)n * shs, hsc + (long long)n * (shs / 32),                \
                               part + (long long)n * s2.rows, t, 1, 0, 0.f, S, 1, nbr, 0, 0, 0, shs, 0, item,   \
                               staged);                                                                         \
                continue;                                                                                       \
            }                                                                                                   \
            int e = sl[sidx], o = off[e], m = off[e + 1] - o;                                                   \
            DsW w = w2s[e];                                                                                     \
            lin_mma<F, 0>(w, w, hc + (long long)o * w.cols, hsc + (long long)o * (w.cols / 32),                \
                          part + (long long)o * w.rows, m, 1, 0, 0.f, S, 1, nbr, 0, 0, 0, 0, 0, item, staged);  \
        }                                                                                                       \
    }
DS_MOE_KP(3, 2)
DS_MOE_KP(2, 2)
DS_MOE_KP(3, 3)
DS_MOE_KP(2, 3)

// wo_a on the K-split path in one launch: block-diagonal over blockIdx.y = group gi (ol rows of
// w, its gd = w.cols columns of the token-major activation [t][g * gd]), quantized once by
// ds_ks_quant over the whole row (gd % 32 == 0, so the 32-groups are the per-group ones), into
// mid [t][g * ol] directly: no perms. SB = 1, blockIdx.x strides the group's nbr row groups.
#define DS_WO_KS(F)                                                                                            \
    extern "C" __global__ void __launch_bounds__(256) ds_wo_ks_f##F(DsW w, const unsigned char* xc,            \
                                                                     const float* xsc, float* y, int t, int ol, \
                                                                     int S, int nbr) { DS_PDL_ENTRY                            \
        long long gi = blockIdx.y, gd = w.cols, g = gridDim.y;                                                  \
        DsW v = w;                                                                                              \
        v.rows = ol;                                                                                            \
        if (F == 2) {                                                                                           \
            v.w += gi * ol * gd;                                                                                \
            v.s += (gi * ol / 32) * ((gd + 31) / 32);                                                           \
        } else {                                                                                                \
            v.w += gi * ol * gd / 2;                                                                            \
            v.s += gi * ol * gd / 32;                                                                           \
        }                                                                                                       \
        lin_mma<F, 0>(v, v, xc + gi * gd, xsc + gi * (gd / 32), y + gi * ol, t, 1, 0, 0.f, S, 1, nbr, 0, 0, 0,   \
                      (int)(g * gd), (int)(g * ol));                                                            \
    }
DS_WO_KS(2)
DS_WO_KS(3)

// The q/kv epilogue of the decode attention, block (token, part) of 1024 threads (as ds_rms):
// part 0 rms(qr) in place (bit for bit ds_rms) and its K-split codes/scales for wq_b (as
// ds_ks_quant); part 1 rms(kv) (ds_rms), rope at P[0] + token (dg_rope), fp8 qdq
// (ds_fp8_qdq), into kv, ring slot (P[0] + token) % win, and all[win + token] when all != 0.
extern "C" __global__ void __launch_bounds__(1024) ds_qkv_post(float* qr, DsW qn, unsigned char* qc, float* qsc,
                                                                float* kv, DsW kn, const float2* tab, int rd,
                                                                const int* P, float* ring, int win, float* all,
                                                                float eps) { DS_PDL_ENTRY
    __shared__ float red[32];
    int ti = blockIdx.x, isq = blockIdx.y == 0;
    DsW w = isq ? qn : kn;
    int n = w.cols;
    float* r = (isq ? qr : kv) + (long long)ti * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    float rs = 1.f / sqrtf(red[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] = bf(wget(w, 0, i) * (r[i] * rs));
    __syncthreads();
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (isq) {
        for (int g = wid; g < n / 32; g += nw) {
            float xv = r[g * 32 + lane], a = fabsf(xv);
#pragma unroll
            for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            float sc = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
            qc[(long long)ti * n + g * 32 + lane] =
                (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(xv / sc, -448.f), 448.f)));
            if (lane == 0) qsc[(long long)ti * (n / 32) + g] = sc;
        }
        return;
    }
    int half = rd / 2, pos = P[0] + ti;
    float* tp = r + (n - rd);
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float2 cs = tab[(long long)pos * half + j];
        float a = tp[2 * j], b = tp[2 * j + 1];
        tp[2 * j] = bf(a * cs.x - b * cs.y);
        tp[2 * j + 1] = bf(a * cs.y + b * cs.x);
    }
    __syncthreads();
    float* rr = ring + (long long)(pos % win) * n;
    float* ar = all ? all + (long long)(win + ti) * n : 0;
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

// ---- grouped MoE over 8-row tiles of the CSR segments (prefill, t > 8) ----
//
// blockIdx.y is a tile of `tiles` (ds_moe_plan_tiles: expert e | chunk c << 16, -1 past
// them): expert e's CSR rows off[e] + 8c .. min(off[e+1], off[e] + 8c + 8). Each tile is the
// t <= 8 contract on its rows, so one launch per projection covers any segment length; rows
// are independent, so per row the results are those of the per-expert loop (exact: bit for
// bit; K-split: its relaxed bar).
#define MOE_TILE                                                                                               \
    int tl = tiles[blockIdx.y];                                                                                \
    if (tl < 0) return;                                                                                        \
    int e = tl & 0xffff, o = off[e] + 8 * (tl >> 16), n = min(8, off[e + 1] - o);                             \
    if (n <= 0) return;
#define DS_MOE_TILES(F)                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) ds_moe_tgu_f##F(                                         \
        const DsW* w1s, const DsW* w3s, const int* tiles, const int* off, const int* rows, const float* rw,   \
        const float* xq, float* h, float lim) { DS_PDL_ENTRY                                                                \
        MOE_TILE DsW w = w1s[e];                                                                               \
        lin_fast<F, 8, 1>(w, w3s[e], xq, h + (long long)o * w.rows, n, 1, 0, rw + o, lim, rows + o);           \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) ds_moe_tdown_f##F(                                       \
        const DsW* w2s, const int* tiles, const int* off, const float* hq, float* part) { DS_PDL_ENTRY                     \
        MOE_TILE DsW w = w2s[e];                                                                               \
        lin_fast<F, 8, 0>(w, w, hq + (long long)o * w.cols, part + (long long)o * w.rows, n, 1, 0, 0, 0.f, 0); \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256) ds_moe_ks_tgu_f##F(                                      \
        const DsW* w1s, const DsW* w3s, const int* tiles, const int* off, const int* rows, const float* rw,   \
        const unsigned char* xc, const float* xsc, float* h, float lim, int S, int nbr) { DS_PDL_ENTRY                      \
        MOE_TILE DsW w = w1s[e];                                                                               \
        lin_mma<F, 1>(w, w3s[e], xc, xsc, h + (long long)o * w.rows, n, 1, rw + o, lim, S, 1, nbr, 0, 0,       \
                      rows + o);                                                                               \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256) ds_moe_ks_tdown_f##F(                                    \
        const DsW* w2s, const int* tiles, const int* off, const unsigned char* hc, const float* hsc,          \
        float* part, int S, int nbr) { DS_PDL_ENTRY                                                                         \
        MOE_TILE DsW w = w2s[e];                                                                               \
        lin_mma<F, 0>(w, w, hc + (long long)o * w.cols, hsc + (long long)o * (w.cols / 32),                    \
                      part + (long long)o * w.rows, n, 1, 0, 0.f, S, 1, nbr, 0, 0, 0);                         \
    }
DS_MOE_TILES(2)
DS_MOE_TILES(3)

// dg_rope's value of element col (>= d - rd) of a d-wide row whose rotated pair is (a, b), at
// position pos, bit for bit (inverse: by -theta). The fused q / o ropes of the tiled attention.
__device__ __forceinline__ float ds_rope_at(float a, float b, int col, int d, int rd, const float2* tab, int pos,
                                            int inverse) {
    int e = col - (d - rd), j = e >> 1;
    float2 cs = tab[(long long)pos * (rd / 2) + j];
    float s = inverse ? -cs.y : cs.y;
    return (e & 1) ? bf(a * s + b * cs.x) : bf(a * cs.x - b * s);
}

// ---- tiled sparse attention (default; DS_EXACT=1 selects ds_sparse_attn) ---------------------
//
// Three launches over a scratch buffer, same numerics as ds_sparse_attn up to summation order:
// 1. ds_attn_qk: block per (t, 64-head chunk, 32-key tile, 64-wide d slice): the key tile and
//    the heads' q slice in shared memory, every head scores every key of the tile (4 heads ×
//    2 keys a thread); partial dot products to part[t][d/64][h][k].
// 2. ds_attn_soft: warp per (t, h): score = Σ partials × scale (-inf for idx -1), softmax with
//    the sink in the denominator; P = bf(p) to pr[t][h][k], den to den[t][h].
// 3. ds_attn_pv: block per (t, 8 heads, 64-wide d slice), 4 key slices × 64 dims: each kv
//    value is read once for 8 heads; o = bf(Σ P kv / den).

extern "C" __global__ void __launch_bounds__(256) ds_attn_qk(const float* q, const float* kv0, int split,
                                                             const float* kv1, const int* idxs, float* part,
                                                             int h, int d, int k, const float2* tab, const int* Pp,
                                                             int rd, int add, int ps) { DS_PDL_ENTRY
    __shared__ float Qs[64][65];
    __shared__ float Ks[32][65];
    int nkt = (k + 31) / 32, nds = d / 64, nhc = h / 64;
    int b = blockIdx.x;
    int ds = b % nds;
    int kt = (b / nds) % nkt;
    int hc = (b / (nds * nkt)) % nhc;
    int ti = b / (nds * nkt * nhc);
    int tid = threadIdx.x;
    const int* ix = idxs + (long long)ti * k;
    for (int e = tid; e < 64 * 64; e += 256) {
        int hh = e / 64, i = e % 64;
        const float* qr = q + ((long long)ti * h + hc * 64 + hh) * d;
        int col = ds * 64 + i;
        float v = qr[col];
        if (tab && col >= d - rd) {
            int c0 = col - ((col - (d - rd)) & 1);
            v = ds_rope_at(qr[c0], qr[c0 + 1], col, d, rd, tab, Pp[0] + add + ti * ps, 0);
        }
        Qs[hh][i] = v;
    }
    for (int e = tid; e < 32 * 64; e += 256) {
        int kk = e / 64, i = e % 64;
        int j = kt * 32 + kk;
        int r = j < k ? ix[j] : -1;
        Ks[kk][i] = r >= 0 ? DS_KROW(r)[ds * 64 + i] : 0.f;
    }
    __syncthreads();
    int kq = tid & 15, hq = tid >> 4;
    float a[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll 8
    for (int i = 0; i < 64; i++) {
        float k0 = Ks[kq * 2][i], k1 = Ks[kq * 2 + 1][i];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            float qv = Qs[hq * 4 + m][i];
            a[m][0] = fmaf(qv, k0, a[m][0]);
            a[m][1] = fmaf(qv, k1, a[m][1]);
        }
    }
#pragma unroll
    for (int m = 0; m < 4; m++)
#pragma unroll
        for (int c = 0; c < 2; c++) {
            int j = kt * 32 + kq * 2 + c;
            if (j < k) part[(((long long)ti * nds + ds) * h + hc * 64 + hq * 4 + m) * k + j] = a[m][c];
        }
}

extern "C" __global__ void ds_attn_soft(const float* part, const float* sink, const int* idxs, float* pr,
                                        float* den, int t, int h, int d, int k, float scale) { DS_PDL_ENTRY
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= t * h) return;
    int ti = warp / h, hi = warp % h, nds = d / 64;
    const int* ix = idxs + (long long)ti * k;
    float* p = pr + (long long)warp * k;
    float mx = -DS_INF;
    for (int j = lane; j < k; j += 32) {
        float s = 0.f;
        for (int c = 0; c < nds; c++) s += part[(((long long)ti * nds + c) * h + hi) * k + j];
        s = ix[j] >= 0 ? s * scale : -DS_INF;
        p[j] = s;
        mx = fmaxf(mx, s);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    mx = fmaxf(mx, -1e30f);
    float sum = 0.f;
    for (int j = lane; j < k; j += 32) {
        float e = expf(p[j] - mx);
        sum += e;
        p[j] = bf(e);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) den[warp] = sum + expf(sink[hi] - mx);
}

// ds_attn_soft fused into ds_attn_pv (bit for bit the pair): block per (t, 8 heads, dsb
// consecutive 64-wide d slices); warp m computes head g * 8 + m's softmax exactly as
// ds_attn_soft into shared P and den, then the slices run ds_attn_pv's accumulation.
extern "C" __global__ void __launch_bounds__(256) ds_attn_spv(const float* part, const float* sink,
                                                              const float* kv0, int split, const float* kv1,
                                                              const int* idxs, float* o, int h, int d, int k,
                                                              float scale, int dsb, const float2* tab, const int* Pp,
                                                              int rd, int add, int ps) { DS_PDL_ENTRY
    extern __shared__ float P[];  // [8][k], then the k indices
    __shared__ float red[4][8][64];
    __shared__ float dens[8];
    int* Is = (int*)(P + 8 * k);
    int nds = d / 64, hg = h / 8, nb = nds / dsb;
    int b = blockIdx.x;
    int db = b % nb, g = (b / nb) % hg, ti = b / (nb * hg);
    int tid = threadIdx.x;
    for (int e = tid; e < k; e += 256) Is[e] = idxs[(long long)ti * k + e];
    __syncthreads();
    {
        int m = tid >> 5, lane = tid & 31, hi = g * 8 + m;
        float* p = P + m * k;
        float mx = -DS_INF;
        for (int j = lane; j < k; j += 32) {
            float s = 0.f;
            for (int c = 0; c < nds; c++) s += part[(((long long)ti * nds + c) * h + hi) * k + j];
            s = Is[j] >= 0 ? s * scale : -DS_INF;
            p[j] = s;
            mx = fmaxf(mx, s);
        }
#pragma unroll
        for (int o2 = 16; o2; o2 >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o2));
        mx = fmaxf(mx, -1e30f);
        float sum = 0.f;
        for (int j = lane; j < k; j += 32) {
            float e = expf(p[j] - mx);
            sum += e;
            p[j] = bf(e);
        }
#pragma unroll
        for (int o2 = 16; o2; o2 >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o2);
        if (lane == 0) dens[m] = sum + expf(sink[hi] - mx);
    }
    __syncthreads();
    int ks = tid >> 6, di = tid & 63;
    for (int ds = db * dsb; ds < (db + 1) * dsb; ds++) {
        int off = ds * 64 + di;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int j0 = ks; j0 < k; j0 += 32) {
            float v[8];
#pragma unroll
            for (int u = 0; u < 8; u++) {
                int j = j0 + 4 * u;
                int r = j < k ? Is[j] : -1;
                v[u] = r >= 0 ? DS_KROW(r)[off] : 0.f;
            }
#pragma unroll
            for (int u = 0; u < 8; u++) {
                int j = j0 + 4 * u;
                if (j < k)
#pragma unroll
                    for (int m = 0; m < 8; m++) acc[m] = fmaf(P[m * k + j], v[u], acc[m]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 8; m++) red[ks][m][di] = acc[m];
        __syncthreads();
        for (int e = tid; e < 8 * 64; e += 256) {
            int m = e / 64, c = e % 64;
            float s = red[0][m][c] + red[1][m][c] + red[2][m][c] + red[3][m][c];
            int hi = g * 8 + m, col = ds * 64 + c;
            float v = bf(s / dens[m]);
            if (tab && col >= d - rd) {
                // the inverse rope of the stored pair: the partner recomputed as it is stored
                int c2 = c ^ 1;
                float s2 = red[0][m][c2] + red[1][m][c2] + red[2][m][c2] + red[3][m][c2];
                float v2 = bf(s2 / dens[m]);
                bool odd = (col - (d - rd)) & 1;
                v = ds_rope_at(odd ? v2 : v, odd ? v : v2, col, d, rd, tab, Pp[0] + add + ti * ps, 1);
            }
            o[((long long)ti * h + hi) * d + col] = v;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) ds_attn_pv(const float* pr, const float* den, const float* kv0,
                                                             int split, const float* kv1, const int* idxs,
                                                             float* o, int h, int d, int k, const float2* tab,
                                                             const int* Pp, int rd, int add, int ps) { DS_PDL_ENTRY
    extern __shared__ float P[];  // [8][k], then the k indices
    __shared__ float red[4][8][64];
    int* Is = (int*)(P + 8 * k);
    int nds = d / 64, hg = h / 8;
    int b = blockIdx.x;
    int ds = b % nds, g = (b / nds) % hg, ti = b / (nds * hg);
    int tid = threadIdx.x;
    for (int e = tid; e < 8 * k; e += 256) P[e] = pr[((long long)ti * h + g * 8) * k + e];
    for (int e = tid; e < k; e += 256) Is[e] = idxs[(long long)ti * k + e];
    __syncthreads();
    int ks = tid >> 6, di = tid & 63;
    int off = ds * 64 + di;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j0 = ks; j0 < k; j0 += 32) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int j = j0 + 4 * u;
            int r = j < k ? Is[j] : -1;
            v[u] = r >= 0 ? DS_KROW(r)[off] : 0.f;
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int j = j0 + 4 * u;
            if (j < k)
#pragma unroll
                for (int m = 0; m < 8; m++) acc[m] = fmaf(P[m * k + j], v[u], acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < 8; m++) red[ks][m][di] = acc[m];
    __syncthreads();
    for (int e = tid; e < 8 * 64; e += 256) {
        int m = e / 64, c = e % 64;
        float s = red[0][m][c] + red[1][m][c] + red[2][m][c] + red[3][m][c];
        int hi = g * 8 + m, col = ds * 64 + c;
        float v = bf(s / den[ti * h + hi]);
        if (tab && col >= d - rd) {
            // the inverse rope of the stored pair (as ds_attn_spv)
            int c2 = c ^ 1;
            float s2 = red[0][m][c2] + red[1][m][c2] + red[2][m][c2] + red[3][m][c2];
            float v2 = bf(s2 / den[ti * h + hi]);
            bool odd = (col - (d - rd)) & 1;
            v = ds_rope_at(odd ? v2 : v, odd ? v : v2, col, d, rd, tab, Pp[0] + add + ti * ps, 1);
        }
        o[((long long)ti * h + hi) * d + col] = v;
    }
}

// ---- split-K decode attention (relaxed bar; d == 512, h % 32 == 0) -----------------------------
//
// ds_attn_fd: block per (t, 64-head group, key split s of S). q's 64 heads and the split's keys
// (16-key tiles) sit in shared memory as bf16 pairs (both are bf16-valued, so exactly); per tile
// the scores (thread: 2 heads x 2 keys), an online softmax per head (running max / sum,
// p = exp(s - m) unrounded), and o += P kv (thread: a dim pair, all 64 heads in registers).
// Out: part[t][S][h][d] (unnormalised) and ml[t][S][h][2] (running max, sum).
// ds_attn_fd_comb: block per (t, head): the splits merged with the sink in the denominator,
// o = bf(sum_s w_s part_s / den), the o rope (inverse) fused as ds_attn_pv's.
#define FD_KT 16
#define FD_QP 257
#define FD_SMEM ((64 * FD_QP + FD_KT * FD_QP + FD_KT * 64 + 3 * 64 + FD_KT) * 4)
__device__ __forceinline__ unsigned int fd_pack(float2 v) {
    return (__float_as_uint(v.x) >> 16) | (__float_as_uint(v.y) & 0xffff0000u);
}
extern "C" __global__ void __launch_bounds__(256) ds_attn_fd(const float* q, const float* kv0, int split,
                                                             const float* kv1, const int* idxs, float* part,
                                                             float* ml, int h, int d, int k, float scale, int S,
                                                             const float2* tab, const int* Pp, int rd, int add,
                                                             int ps, int, int, const int*, int, int) { DS_PDL_ENTRY
    extern __shared__ float fsm[];
    unsigned int* Qs = (unsigned int*)fsm;  // [64][FD_QP]
    unsigned int* Ks = Qs + 64 * FD_QP;     // [FD_KT][FD_QP]
    float* Ps = (float*)(Ks + FD_KT * FD_QP);  // [FD_KT][64]
    float* scal = Ps + FD_KT * 64;            // [64]
    float* mrun = scal + 64;
    float* lrun = mrun + 64;
    int* Is = (int*)(lrun + 64);  // [FD_KT]
    int hg = h / 64, b = blockIdx.x, tid = threadIdx.x;
    int si = b % S, g = (b / S) % hg, ti = b / (S * hg);
    for (int e = tid; e < 64 * 128; e += 256) {
        int hh = e / 128, c4 = e % 128, col = 4 * c4;
        const float* qr = q + ((long long)ti * h + g * 64 + hh) * d;
        float4 v = *(const float4*)(qr + col);
        if (tab && col >= d - rd) {
            int pos = Pp[0] + add + ti * ps;
            float a0 = ds_rope_at(v.x, v.y, col, d, rd, tab, pos, 0);
            float a1 = ds_rope_at(v.x, v.y, col + 1, d, rd, tab, pos, 0);
            float a2 = ds_rope_at(v.z, v.w, col + 2, d, rd, tab, pos, 0);
            float a3 = ds_rope_at(v.z, v.w, col + 3, d, rd, tab, pos, 0);
            v = make_float4(a0, a1, a2, a3);
        }
        Qs[hh * FD_QP + 2 * c4] = fd_pack(make_float2(v.x, v.y));
        Qs[hh * FD_QP + 2 * c4 + 1] = fd_pack(make_float2(v.z, v.w));
    }
    if (tid < 64) {
        mrun[tid] = -DS_INF;
        lrun[tid] = 0.f;
    }
    int ntile = (k + FD_KT - 1) / FD_KT;
    int t0 = (int)((long long)si * ntile / S), t1 = (int)((long long)(si + 1) * ntile / S);
    const int* ix = idxs + (long long)ti * k;
    float2 acc[64];
#pragma unroll
    for (int m = 0; m < 64; m++) acc[m] = make_float2(0.f, 0.f);
    int hp = (tid >> 3) * 2, kp = (tid & 7) * 2;
    for (int tt = t0; tt < t1; tt++) {
        __syncthreads();
        if (tid < FD_KT) {
            int j = tt * FD_KT + tid;
            Is[tid] = j < k ? ix[j] : -1;
        }
        __syncthreads();
        for (int e = tid; e < FD_KT * 128; e += 256) {
            int kk = e / 128, c4 = e % 128, r = Is[kk];
            float4 v = r >= 0 ? *(const float4*)(DS_KROW(r) + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
            Ks[kk * FD_QP + 2 * c4] = fd_pack(make_float2(v.x, v.y));
            Ks[kk * FD_QP + 2 * c4 + 1] = fd_pack(make_float2(v.z, v.w));
        }
        __syncthreads();
        float sc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
        const unsigned int *q0 = Qs + hp * FD_QP, *q1 = q0 + FD_QP;
        const unsigned int *k0 = Ks + kp * FD_QP, *k1 = k0 + FD_QP;
#pragma unroll 8
        for (int jj = 0; jj < 256; jj++) {
            unsigned int a = q0[jj], bq = q1[jj], c = k0[jj], e2 = k1[jj];
            float a0 = __uint_as_float(a << 16), a1 = __uint_as_float(a & 0xffff0000u);
            float b0 = __uint_as_float(bq << 16), b1 = __uint_as_float(bq & 0xffff0000u);
            float c0 = __uint_as_float(c << 16), c1 = __uint_as_float(c & 0xffff0000u);
            float e0 = __uint_as_float(e2 << 16), e1 = __uint_as_float(e2 & 0xffff0000u);
            sc[0][0] = fmaf(a1, c1, fmaf(a0, c0, sc[0][0]));
            sc[0][1] = fmaf(a1, e1, fmaf(a0, e0, sc[0][1]));
            sc[1][0] = fmaf(b1, c1, fmaf(b0, c0, sc[1][0]));
            sc[1][1] = fmaf(b1, e1, fmaf(b0, e0, sc[1][1]));
        }
        bool v0 = Is[kp] >= 0, v1 = Is[kp + 1] >= 0;
#pragma unroll
        for (int u = 0; u < 2; u++) {
            float s0 = v0 ? sc[u][0] * scale : -DS_INF, s1 = v1 ? sc[u][1] * scale : -DS_INF;
            float mt = fmaxf(s0, s1);
            for (int o = 1; o < 8; o <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            int hh = hp + u;
            float mo = mrun[hh], mn = fmaxf(mo, mt);
            float p0 = 0.f, p1 = 0.f, f = 1.f;
            if (mn > -DS_INF) {
                p0 = expf(s0 - mn);
                p1 = expf(s1 - mn);
                f = mo > -DS_INF ? expf(mo - mn) : 0.f;
            }
            float ls = p0 + p1;
            for (int o = 1; o < 8; o <<= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
            Ps[kp * 64 + hh] = p0;
            Ps[(kp + 1) * 64 + hh] = p1;
            __syncwarp();
            if ((tid & 7) == 0) {
                scal[hh] = f;
                mrun[hh] = mn;
                lrun[hh] = lrun[hh] * f + ls;
            }
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 64; m += 4) {
            float4 f = *(const float4*)(scal + m);
            acc[m].x *= f.x, acc[m].y *= f.x;
            acc[m + 1].x *= f.y, acc[m + 1].y *= f.y;
            acc[m + 2].x *= f.z, acc[m + 2].y *= f.z;
            acc[m + 3].x *= f.w, acc[m + 3].y *= f.w;
        }
        for (int kk = 0; kk < FD_KT; kk++) {
            unsigned int kw = Ks[kk * FD_QP + tid];
            float vx = __uint_as_float(kw << 16), vy = __uint_as_float(kw & 0xffff0000u);
            const float* pr = Ps + kk * 64;
#pragma unroll
            for (int m = 0; m < 64; m += 4) {
                float4 p = *(const float4*)(pr + m);
                acc[m].x = fmaf(p.x, vx, acc[m].x), acc[m].y = fmaf(p.x, vy, acc[m].y);
                acc[m + 1].x = fmaf(p.y, vx, acc[m + 1].x), acc[m + 1].y = fmaf(p.y, vy, acc[m + 1].y);
                acc[m + 2].x = fmaf(p.z, vx, acc[m + 2].x), acc[m + 2].y = fmaf(p.z, vy, acc[m + 2].y);
                acc[m + 3].x = fmaf(p.w, vx, acc[m + 3].x), acc[m + 3].y = fmaf(p.w, vy, acc[m + 3].y);
            }
        }
    }
    __syncthreads();
    long long base = ((long long)ti * S + si) * h + g * 64;
#pragma unroll
    for (int m = 0; m < 64; m++) *(float2*)(part + (base + m) * d + 2 * tid) = acc[m];
    if (tid < 64) {
        ml[(base + tid) * 2] = mrun[tid];
        ml[(base + tid) * 2 + 1] = lrun[tid];
    }
}

extern "C" __global__ void __launch_bounds__(256) ds_attn_fd_comb(const float* part, const float* ml,
                                                                  const float* sink, float* o, int h, int d,
                                                                  int S, const float2* tab, const int* Pp, int rd,
                                                                  int add, int ps, unsigned char* qc, float* qs) { DS_PDL_ENTRY
    int b = blockIdx.x, hi = b % h, ti = b / h, tid = threadIdx.x;
    float M = -DS_INF;
    for (int s = 0; s < S; s++) M = fmaxf(M, ml[(((long long)ti * S + s) * h + hi) * 2]);
    M = fmaxf(M, -1e30f);
    float den = expf(sink[hi] - M);
    float2 a = make_float2(0.f, 0.f);
    for (int s = 0; s < S; s++) {
        long long r = ((long long)ti * S + s) * h + hi;
        float m = ml[r * 2];
        float w = m > -DS_INF ? expf(m - M) : 0.f;
        den += w * ml[r * 2 + 1];
        float2 v = *(const float2*)(part + r * d + 2 * tid);
        a.x = fmaf(w, v.x, a.x);
        a.y = fmaf(w, v.y, a.y);
    }
    int col = 2 * tid;
    float x = bf(a.x / den), y = bf(a.y / den);
    if (tab && col >= d - rd) {
        int pos = Pp[0] + add + ti * ps;
        float x2 = ds_rope_at(x, y, col, d, rd, tab, pos, 1);
        y = ds_rope_at(x, y, col + 1, d, rd, tab, pos, 1);
        x = x2;
    }
    long long oi = ((long long)ti * h + hi) * d + col;
    *(float2*)(o + oi) = make_float2(x, y);
    if (qc) {
        // ds_ks_quant of o (wo_a's input) in the same pass: 16 threads a 32-group
        float a = fmaxf(fabsf(x), fabsf(y));
#pragma unroll
        for (int f = 8; f; f >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, f));
        float sc = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
        qc[oi] = (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(x / sc, -448.f), 448.f)));
        qc[oi + 1] = (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(y / sc, -448.f), 448.f)));
        if ((tid & 15) == 0) qs[oi / 32] = sc;
    }
}

// ds_attn_fd on the tensor cores (bf16 mma, f32 accumulate), same blocks and outputs: per
// 16-key tile the 64 x 16 scores (warp w: heads 16 (w / 2) .., keys 8 (w % 2) ..), the online
// softmax as ds_attn_fd (p then rounded to bf16 for the mma), and o += P kv (warp w: all 64
// heads, dims 64 w ..; the accumulators stay in mma fragments).
__device__ __forceinline__ void mma16816bf(float* c, unsigned int a0, unsigned int a1, unsigned int a2,
                                           unsigned int a3, unsigned int b0, unsigned int b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
#define FD_IMAX 1024
#define FDM_SMEM ((64 * FD_QP + FD_KT * FD_QP + 64 * 17 + 64 * 9 + 3 * 64 + FD_IMAX) * 4)
extern "C" __global__ void __launch_bounds__(256) ds_attn_fdm(const float* q, const float* kv0, int split,
                                                              const float* kv1, const int* idxs, float* part,
                                                              float* ml, int h, int d, int k, float scale, int S,
                                                              const float2* tab, const int* Pp, int rd, int add,
                                                              int ps, int wmode, int win, const int* wsrc, int kc,
                                                              int r0) { DS_PDL_ENTRY
    extern __shared__ float fsm[];
    unsigned int* Qs = (unsigned int*)fsm;       // [64][FD_QP]
    unsigned int* Ks = Qs + 64 * FD_QP;          // [FD_KT][FD_QP]
    float* Ss = (float*)(Ks + FD_KT * FD_QP);    // [64][17]
    unsigned int* Pb = (unsigned int*)(Ss + 64 * 17);  // [64][9]: bf16 key pairs
    float* scal = (float*)(Pb + 64 * 9);
    float* mrun = scal + 64;
    float* lrun = mrun + 64;
    int* Is = (int*)(lrun + 64);  // [FD_IMAX]: the split's key indices
    int hg = h / 64, b = blockIdx.x, tid = threadIdx.x;
    int si = b % S, g = (b / S) % hg, ti = b / (S * hg);
    int lane = tid & 31, w = tid >> 5, gq = lane >> 2, tq = lane & 3;
    for (int e = tid; e < 64 * 128; e += 256) {
        int hh = e / 128, c4 = e % 128, col = 4 * c4;
        const float* qr = q + ((long long)ti * h + g * 64 + hh) * d;
        float4 v = *(const float4*)(qr + col);
        if (tab && col >= d - rd) {
            int pos = Pp[0] + add + ti * ps;
            float a0 = ds_rope_at(v.x, v.y, col, d, rd, tab, pos, 0);
            float a1 = ds_rope_at(v.x, v.y, col + 1, d, rd, tab, pos, 0);
            float a2 = ds_rope_at(v.z, v.w, col + 2, d, rd, tab, pos, 0);
            float a3 = ds_rope_at(v.z, v.w, col + 3, d, rd, tab, pos, 0);
            v = make_float4(a0, a1, a2, a3);
        }
        Qs[hh * FD_QP + 2 * c4] = fd_pack(make_float2(v.x, v.y));
        Qs[hh * FD_QP + 2 * c4 + 1] = fd_pack(make_float2(v.z, v.w));
    }
    if (tid < 64) {
        mrun[tid] = -DS_INF;
        lrun[tid] = 0.f;
    }
    int ntile = (k + FD_KT - 1) / FD_KT;
    int t0 = (int)((long long)si * ntile / S), t1 = (int)((long long)(si + 1) * ntile / S);
    const int* ix = idxs + (long long)ti * k;
    for (int e = tid; e < (t1 - t0) * FD_KT; e += 256) {
        int j = t0 * FD_KT + e, v = -1;
        if (j < k) {
            if (!wmode) {
                v = ix[j];
            } else if (j >= win) {
                v = wsrc[(long long)ti * kc + j - win];  // the top-k columns (ds_idx_cat's)
            } else {
                // the window columns, as dg_win_idx modes 1 / 3 at sp = P[0]
                int sp = Pp[0];
                if (wmode == 1) {
                    int i = (sp % win + 1 + j) % win;
                    v = i > sp ? -1 : i;
                } else {
                    int qq = sp + r0 + ti, cnt = min(win, qq + 1), pos = qq + 1 - cnt + j;
                    v = j >= cnt ? -1 : (pos < sp ? pos % win : win + pos - sp);
                }
            }
        }
        Is[e] = v;
    }
    __syncthreads();
    // the next tile's keys in registers while this one computes (thread: 2 float4)
    float4 kn[FD_KT * 128 / 256];
    auto kload = [&](int tt) {
#pragma unroll
        for (int u = 0; u < FD_KT * 128 / 256; u++) {
            int e = tid + 256 * u, kk = e / 128, c4 = e % 128, r = Is[(tt - t0) * FD_KT + kk];
            kn[u] = r >= 0 ? *(const float4*)(DS_KROW(r) + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto kstore = [&]() {
#pragma unroll
        for (int u = 0; u < FD_KT * 128 / 256; u++) {
            int e = tid + 256 * u, kk = e / 128, c4 = e % 128;
            Ks[kk * FD_QP + 2 * c4] = fd_pack(make_float2(kn[u].x, kn[u].y));
            Ks[kk * FD_QP + 2 * c4 + 1] = fd_pack(make_float2(kn[u].z, kn[u].w));
        }
    };
    if (t0 < t1) kload(t0);
    float acc[4][8][4];
#pragma unroll
    for (int m = 0; m < 4; m++)
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[m][n][c] = 0.f;
    int sm0 = (w >> 1) * 16, sn0 = (w & 1) * 8;  // this warp's score tile
    int hp = (tid >> 3) * 2, kp = (tid & 7) * 2;  // the softmax mapping (as ds_attn_fd)
    for (int tt = t0; tt < t1; tt++) {
        const int* It = Is + (tt - t0) * FD_KT;
        __syncthreads();  // the previous tile's Ks consumed
        kstore();
        __syncthreads();
        if (tt + 1 < t1) kload(tt + 1);
        {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            const unsigned int *qa = Qs + (sm0 + gq) * FD_QP, *qb = qa + 8 * FD_QP;
            const unsigned int* kb = Ks + (sn0 + gq) * FD_QP;
#pragma unroll 8
            for (int ks = 0; ks < 32; ks++) {
                int o = ks * 8 + tq;
                mma16816bf(c, qa[o], qb[o], qa[o + 4], qb[o + 4], kb[o], kb[o + 4]);
            }
            Ss[(sm0 + gq) * 17 + sn0 + 2 * tq] = c[0];
            Ss[(sm0 + gq) * 17 + sn0 + 2 * tq + 1] = c[1];
            Ss[(sm0 + gq + 8) * 17 + sn0 + 2 * tq] = c[2];
            Ss[(sm0 + gq + 8) * 17 + sn0 + 2 * tq + 1] = c[3];
        }
        __syncthreads();
        bool v0 = It[kp] >= 0, v1 = It[kp + 1] >= 0;
#pragma unroll
        for (int u = 0; u < 2; u++) {
            int hh = hp + u;
            float s0 = v0 ? Ss[hh * 17 + kp] * scale : -DS_INF, s1 = v1 ? Ss[hh * 17 + kp + 1] * scale : -DS_INF;
            float mt = fmaxf(s0, s1);
            for (int o = 1; o < 8; o <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            float mo = mrun[hh], mn = fmaxf(mo, mt);
            float p0 = 0.f, p1 = 0.f, f = 1.f;
            if (mn > -DS_INF) {
                p0 = bf(expf(s0 - mn));
                p1 = bf(expf(s1 - mn));
                f = mo > -DS_INF ? expf(mo - mn) : 0.f;
            }
            float ls = p0 + p1;
            for (int o = 1; o < 8; o <<= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
            Pb[hh * 9 + kp / 2] = fd_pack(make_float2(p0, p1));
            __syncwarp();
            if ((tid & 7) == 0) {
                scal[hh] = f;
                mrun[hh] = mn;
                lrun[hh] = lrun[hh] * f + ls;
            }
        }
        __syncthreads();
        unsigned int pa[4][4];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            float f0 = scal[m * 16 + gq], f1 = scal[m * 16 + gq + 8];
#pragma unroll
            for (int n = 0; n < 8; n++) {
                acc[m][n][0] *= f0, acc[m][n][1] *= f0;
                acc[m][n][2] *= f1, acc[m][n][3] *= f1;
            }
            const unsigned int *r0 = Pb + (m * 16 + gq) * 9, *r1 = r0 + 8 * 9;
            pa[m][0] = r0[tq];
            pa[m][1] = r1[tq];
            pa[m][2] = r0[tq + 4];
            pa[m][3] = r1[tq + 4];
        }
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int dim = w * 64 + n * 8 + gq, wd = dim >> 1, sh = (dim & 1) * 16;
            unsigned int k0 = (Ks[(2 * tq) * FD_QP + wd] >> sh) & 0xffffu;
            unsigned int k1 = (Ks[(2 * tq + 1) * FD_QP + wd] >> sh) & 0xffffu;
            unsigned int k8 = (Ks[(2 * tq + 8) * FD_QP + wd] >> sh) & 0xffffu;
            unsigned int k9 = (Ks[(2 * tq + 9) * FD_QP + wd] >> sh) & 0xffffu;
            unsigned int b0 = k0 | (k1 << 16), b1 = k8 | (k9 << 16);
#pragma unroll
            for (int m = 0; m < 4; m++) mma16816bf(acc[m][n], pa[m][0], pa[m][1], pa[m][2], pa[m][3], b0, b1);
        }
    }
    __syncthreads();
    long long base = ((long long)ti * S + si) * h + g * 64;
#pragma unroll
    for (int m = 0; m < 4; m++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int dim = w * 64 + n * 8 + 2 * tq;
            *(float2*)(part + (base + m * 16 + gq) * d + dim) = make_float2(acc[m][n][0], acc[m][n][1]);
            *(float2*)(part + (base + m * 16 + gq + 8) * d + dim) = make_float2(acc[m][n][2], acc[m][n][3]);
        }
    if (tid < 64) {
        ml[(base + tid) * 2] = mrun[tid];
        ml[(base + tid) * 2 + 1] = lrun[tid];
    }
}

// ds_head_bf16 with one row a warp (y [t][rows] = x [t][cols] . w^T, bf16 w, no rounding), for
// the small bf16 linears (compressor wkv / wgate): block (8 rows, one 1024-column chunk), x's
// chunk staged in shared memory; per-chunk partials part[chunk][t][rows], the last block of a
// row group (cnt, left zero) sums them in chunk order. cols % 1024 == 0, t <= 8.
extern "C" __global__ void __launch_bounds__(256) ds_gemv_bf16_r1(const unsigned short* w, int rows, int cols,
                                                                  const float* x, float* y, int t, float* part,
                                                                  int* cnt) {
    __shared__ __align__(16) float xs[8][1024];
    __shared__ int last;
    int tid = threadIdx.x, lane = tid & 31, n = blockIdx.x * 8 + (tid >> 5), ch = blockIdx.y, nch = gridDim.y;
    int c0 = ch * 1024;
    for (int e = tid; e < t * 256; e += 256) {
        int i = e / 256, c4 = e % 256;
        *(float4*)&xs[i][4 * c4] = *(const float4*)(x + (long long)i * cols + c0 + 4 * c4);
    }
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = 0.f;
    const unsigned short* wr = w + (long long)min(n, rows - 1) * cols + c0;
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; u++) v[u] = __ldcs((const uint4*)(wr + u * 256 + lane * 8));
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 4; u++) {
        int c = u * 256 + lane * 8;
        unsigned int q[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
        float wf[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wf[2 * j] = __uint_as_float(q[j] << 16);
            wf[2 * j + 1] = __uint_as_float(q[j] & 0xffff0000u);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i >= t) break;
            float4 a = *(const float4*)&xs[i][c], b = *(const float4*)&xs[i][c + 4];
            float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 8; j++) s[i] = fmaf(xv[j], wf[j], s[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i >= t) break;
        float r = s[i];
        for (int o = 16; o; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
        if (lane == 0 && n < rows) part[((long long)ch * t + i) * rows + n] = r;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        last = atomicAdd(cnt + blockIdx.x, 1) == nch - 1;
        if (last) cnt[blockIdx.x] = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = tid; e < 8 * t; e += 256) {
        int i = e / 8, r = blockIdx.x * 8 + e % 8;
        if (r >= rows) continue;
        float a = 0.f;
        for (int c = 0; c < nch; c++) a += __ldcg(part + ((long long)c * t + i) * rows + r);
        y[(long long)i * rows + r] = a;
    }
}
