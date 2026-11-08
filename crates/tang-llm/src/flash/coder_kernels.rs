//! The CUDA kernels of the Qwen3-Coder-Next engine ([`super::coder`]): what a plain-residual,
//! indexer-free, all-VRAM qwen3next window needs beyond the native GEMV (`kernels::GEMV_SRC`,
//! compiled into the same module) and the GDN kernels of the flash module.
//!
//! Every kernel's arithmetic for a token is independent of the window width `t` (the same
//! reduction orders at any `t`), so a token's outputs are bitwise the same in a T=1 window, a
//! T=8 window and a verify window (`flash-tcheck`).

/// The `#define` prelude for a model's shapes, for windows of up to `maxt` rows (`MAX_T`, or the
/// wide prefill width, which also sets the int8 activation layout `XQ8_T`).
/// TCX expert kernels' K split (warps a 16-row tile): gate|up, down
/// (`TANG_TCX_SGU` / `TANG_TCX_SDN`, divisors of 8 and of the superblock count).
pub fn tcx_split() -> (usize, usize) {
    let n = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    (n("TANG_TCX_SGU", 2), n("TANG_TCX_SDN", 1))
}

pub fn prelude(c: &Shapes, maxt: usize) -> String {
    let (sgu, sdn) = tcx_split();
    let tiles = if std::env::var("TANG_CODER_TCX_TILES").is_ok_and(|v| v == "1") {
        "#define CX_TCX_TILES 1\n"
    } else {
        ""
    };
    format!(
        "#define CX_HID {hid}\n#define CX_NH {nh}\n#define CX_NKV {nkv}\n#define CX_HD {hd}\n\
         #define CX_ROT {rot}\n#define CX_G {g}\n#define CX_QK {qk}\n#define CX_QV {qv}\n\
         #define CX_FF {ff}\n#define CX_TOPK {topk}\n#define CX_NE {ne}\n#define CX_NP {np}\n\
         #define CX_CHUNK {chunk}\n#define CX_MAXT {maxt}\n#define XQ8_T {maxt}\n\
         #define CX_TCX_SGU {sgu}\n#define CX_TCX_SDN {sdn}\n{tiles}{kv}",
        hid = c.hidden,
        nh = c.n_head,
        nkv = c.n_kv,
        hd = c.head_dim,
        rot = c.n_rot,
        g = c.n_head / c.n_kv,
        qk = c.n_head * 2 * c.head_dim,
        qv = c.n_head * 2 * c.head_dim + c.n_kv * c.head_dim,
        ff = c.ff,
        topk = c.topk,
        ne = c.n_expert,
        np = maxt * (c.topk + 1),
        chunk = attn_chunk(),
        maxt = maxt,
        kv = if kv8() {
            format!(
                "#define CX_KV8 1\n#define CX_KVROW {}\n",
                kv_row_bytes(c.n_kv, c.head_dim)
            )
        } else {
            String::new()
        },
    )
}

/// Positions per split-K attention block (`TANG_CODER_CHUNK`, default 64).
pub fn attn_chunk() -> usize {
    std::env::var("TANG_CODER_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

pub use super::fa2::{fa2_smem, fa2_splits, kv8, kv_row_bytes, FA2_BM, FA2_BN};

/// The shapes the kernels are compiled for.
#[derive(Clone, Debug)]
pub struct Shapes {
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub ff: usize,
    pub topk: usize,
    pub n_expert: usize,
}

pub const SRC: &str = r#"
#define MAXT 8
#define NEG_INF_F __int_as_float(0xff800000)

__device__ __forceinline__ float sigm_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_f2(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ unsigned short f2bf(float x) {
    unsigned int b = __float_as_uint(x);
    return (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }

// Sum over a block (blockDim.x a multiple of 32, at most 1024); every thread gets the total, in
// a fixed order.
__device__ float cx_bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += sh[i];
    return s;
}

// x = rmsnorm(r) * w and its int8 codes xq (fe_q8's layout and rule: codes [XQ8_T][n], then
// per-32 f32 scales), after r += y when `add`. Grid t, block 1024 (or 256); n % blockDim == 0, n <= 16 blockDim.
__device__ void norm_q8_body(float* rt, float (&v)[16], int nv, const float* w, float* x, unsigned char* xq,
                             int n, float eps, int t) {
    __shared__ float sh[32];
    float ss = 0.f;
    for (int j = 0; j < nv; j++) ss = fmaf(v[j], v[j], ss);
    ss = cx_bsum(ss, sh);
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    const int lane = threadIdx.x & 31;
    float* xd = (float*)(xq + (size_t)XQ8_T * n);
    for (int j = 0; j < nv; j++) {
        const int i = threadIdx.x + j * blockDim.x;
        const float y = v[j] * inv * w[i];
        x[(size_t)t * n + i] = y;
        float amax = fabsf(y);
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(y / d), -127.f), 127.f);
        xq[(size_t)t * n + i] = (unsigned char)(signed char)q;
        if (lane == 0) xd[t * (n / 32) + i / 32] = d;
    }
}
extern "C" __global__ void cx_norm_q8(float* r, const float* y, const float* w, float* x, unsigned char* xq,
                                      int n, float eps, int add) {
    const int t = blockIdx.x, nv = n / blockDim.x;
    float* rt = r + (size_t)t * n;
    float v[16];
    for (int j = 0; j < nv; j++) {
        const int i = threadIdx.x + j * blockDim.x;
        float a = rt[i];
        if (add) { a += y[(size_t)t * n + i]; rt[i] = a; }
        v[j] = a;
    }
    norm_q8_body(rt, v, nv, w, x, xq, n, eps, t);
}

// r += Σ_k w_k parts[t·K + k] (rank order) + sigmoid(lg[t][NE]) parts[T·K + t] (the shared
// expert), y (when given) gets that sum alone; then x, xq = the next norm of r (as
// cx_norm_q8). Grid t, block 1024.
extern "C" __global__ void cx_moe_add_norm(float* r, const float* parts, const float* wts, const float* lg, int ls,
                                           int T, float* y, const float* w, float* x, unsigned char* xq, float eps) {
    const int t = blockIdx.x, n = CX_HID, nv = CX_HID / (int)blockDim.x;
    float* rt = r + (size_t)t * n;
    const float g = sigm_f(lg[(size_t)t * ls + CX_NE]);
    float v[16];
    for (int j = 0; j < nv; j++) {
        const int i = threadIdx.x + j * blockDim.x;
        float acc = 0.f;
        for (int k = 0; k < CX_TOPK; k++) acc += wts[t * CX_TOPK + k] * parts[((size_t)t * CX_TOPK + k) * n + i];
        acc += g * parts[((size_t)T * CX_TOPK + t) * n + i];
        if (y) y[(size_t)t * n + i] = acc;
        const float a = rt[i] + acc;
        rt[i] = a;
        v[j] = a;
    }
    norm_q8_body(rt, v, nv, w, x, xq, n, eps, t);
}

// Softmax over the router logits, top CX_TOPK (ties to the lower id), renormalised in f64.
// Grid t, block 32. Logits row stride `ls`; writes ids [t][TOPK], w [t][TOPK].
// One warp: token t's top-k into id_out / w_out.
__device__ __forceinline__ void route_core(const float* lg, int ls, int t, unsigned* id_out, float* w_out, float scale) {
    const int lane = threadIdx.x & 31;
    const float* l = lg + (size_t)t * ls;
    const int PER = (CX_NE + 31) / 32;
    float v[PER];
    float mx = NEG_INF_F;
    for (int i = 0; i < PER; i++) {
        const int e = lane + 32 * i;
        v[i] = e < CX_NE ? l[e] : NEG_INF_F;
        mx = fmaxf(mx, v[i]);
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    // p_e = exp(l_e - max) / sum over all e; renormalised over the top-k the sum cancels.
    const double sum = 1.0;
    unsigned used = 0;
    unsigned my_id = 0;
    double my_p = 0.0, wsum = 0.0;
    for (int k = 0; k < CX_TOPK; k++) {
        float bv = NEG_INF_F;
        int bi = 0x7fffffff;
        for (int i = 0; i < PER; i++) {
            const int e = lane + 32 * i;
            if (e < CX_NE && !((used >> i) & 1u) && (v[i] > bv || (v[i] == bv && e < bi))) { bv = v[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((bi & 31) == lane) used |= 1u << (bi >> 5);
        const double p = exp((double)bv - (double)mx) / sum;
        wsum += p;
        if (lane == k) { my_id = (unsigned)bi; my_p = p; }
    }
    // (flash-ref clamps Σ top p at 6.1e-5; the top-1 term alone is p_max / sum >= 1 / NE here.)
    if (lane < CX_TOPK) {
        float w = (float)(my_p / wsum);
        if (scale != 0.f && scale != 1.f) w *= scale;
        id_out[lane] = my_id;
        w_out[lane] = w;
    }
}

extern "C" __global__ void cx_route(const float* lg, int ls, unsigned* ids, float* wts, float scale) {
    route_core(lg, ls, blockIdx.x, ids + blockIdx.x * CX_TOPK, wts + blockIdx.x * CX_TOPK, scale);
}

// One row of a GGUF-typed weight against int8 codes `xc` (plain order) with per-32 scales `xd`.
template <int TY>
__device__ __forceinline__ float cx_dot8(const unsigned char* w, const unsigned char* xc, const float* xd, int k) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int e0 = lane * 8; e0 < k; e0 += 256) {
        int w0, w1;
        float dw, mw;
        decode8i<TY>(w, e0, w0, w1, dw, mw);
        const uint2 xv = *(const uint2*)(xc + e0);
        const float dx = xd[e0 / 32];
        const int s = __dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0));
        if (TY == 12 || TY == 13) {
            const int sx = __dp4a(0x01010101, (int)xv.x, __dp4a(0x01010101, (int)xv.y, 0));
            acc = fmaf(dx, fmaf(dw, i2f(s), -mw * i2f(sx)), acc);
        } else {
            acc = fmaf(dx * dw, i2f(s), acc);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
// Q4_K / Q5_K with each lane on 32 weights a step: 16 bytes of qs (sub-blocks 2j and 2j + 1,
// positions l0..l0 + 16, low and high nibbles), its two scale/min pairs unpacked once, 8 dp4a.
// Lane → (superblock lane / 8 of each group of 4, pair j, half l0). Rows and x must be 16-byte
// aligned (staged expert rows are).
template <int TY>
__device__ __forceinline__ float cx_dotq_fast(const unsigned char* w, const unsigned char* xc, const float* xd, int k) {
    const int lane = threadIdx.x & 31;
    const int BB = TY == 12 ? 144 : 176, QS = TY == 12 ? 16 : 48;
    const int sbl = lane >> 3, j = (lane & 7) >> 1, l0 = (lane & 1) * 16;
    float acc = 0.f;
    for (int sb = sbl; sb < k / 256; sb += 4) {
        const unsigned char* b = w + sb * BB;
        const float d = h2f(rd16(b)), dmin = h2f(rd16(b + 2));
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, b + 4, s0, m0);
        scale_min_k4(2 * j + 1, b + 4, s1, m1);
        const uint4 q = *(const uint4*)(b + QS + j * 32 + l0);
        uint4 hb = make_uint4(0, 0, 0, 0);
        if (TY == 13) hb = *(const uint4*)(b + 16 + l0);
        const int e = sb * 256 + j * 64 + l0;
        const uint4 x0 = *(const uint4*)(xc + e), x1 = *(const uint4*)(xc + e + 32);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {hb.x, hb.y, hb.z, hb.w};
        const unsigned xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
        int sa = 0, sh = 0, ca = 0, ch = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned lo = qw[i] & 0x0F0F0F0Fu, hi = (qw[i] >> 4) & 0x0F0F0F0Fu;
            if (TY == 13) {
                lo |= ((hw[i] >> (2 * j)) & 0x01010101u) << 4;
                hi |= ((hw[i] >> (2 * j + 1)) & 0x01010101u) << 4;
            }
            sa = __dp4a((int)lo, (int)xa[i], sa);
            sh = __dp4a((int)hi, (int)xb[i], sh);
            ca = __dp4a(0x01010101, (int)xa[i], ca);
            ch = __dp4a(0x01010101, (int)xb[i], ch);
        }
        const float dx0 = xd[e / 32], dx1 = xd[e / 32 + 1];
        acc = fmaf(dx0, fmaf(d * i2f(s0), i2f(sa), -dmin * i2f(m0) * i2f(ca)), acc);
        acc = fmaf(dx1, fmaf(d * i2f(s1), i2f(sh), -dmin * i2f(m1) * i2f(ch)), acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
// cx_dotq_fast on half a warp: lanes 0..16 do one row, 16..32 another (each half returns its
// own sum). For k = 512 (two superblocks: 16 lanes' worth), where the whole-warp version
// leaves half the lanes idle.
template <int TY>
__device__ __forceinline__ float cx_dotq_half(const unsigned char* w, const unsigned char* xc, const float* xd, int k) {
    const int lane = threadIdx.x & 15;
    const int BB = TY == 12 ? 144 : 176, QS = TY == 12 ? 16 : 48;
    const int sbl = lane >> 3, j = (lane & 7) >> 1, l0 = (lane & 1) * 16;
    float acc = 0.f;
    for (int sb = sbl; sb < k / 256; sb += 2) {
        const unsigned char* b = w + sb * BB;
        const float d = h2f(rd16(b)), dmin = h2f(rd16(b + 2));
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, b + 4, s0, m0);
        scale_min_k4(2 * j + 1, b + 4, s1, m1);
        const uint4 q = *(const uint4*)(b + QS + j * 32 + l0);
        uint4 hb = make_uint4(0, 0, 0, 0);
        if (TY == 13) hb = *(const uint4*)(b + 16 + l0);
        const int e = sb * 256 + j * 64 + l0;
        const uint4 x0 = *(const uint4*)(xc + e), x1 = *(const uint4*)(xc + e + 32);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {hb.x, hb.y, hb.z, hb.w};
        const unsigned xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
        int sa = 0, sh = 0, ca = 0, ch = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned lo = qw[i] & 0x0F0F0F0Fu, hi = (qw[i] >> 4) & 0x0F0F0F0Fu;
            if (TY == 13) {
                lo |= ((hw[i] >> (2 * j)) & 0x01010101u) << 4;
                hi |= ((hw[i] >> (2 * j + 1)) & 0x01010101u) << 4;
            }
            sa = __dp4a((int)lo, (int)xa[i], sa);
            sh = __dp4a((int)hi, (int)xb[i], sh);
            ca = __dp4a(0x01010101, (int)xa[i], ca);
            ch = __dp4a(0x01010101, (int)xb[i], ch);
        }
        const float dx0 = xd[e / 32], dx1 = xd[e / 32 + 1];
        acc = fmaf(dx0, fmaf(d * i2f(s0), i2f(sa), -dmin * i2f(m0) * i2f(ca)), acc);
        acc = fmaf(dx1, fmaf(d * i2f(s1), i2f(sh), -dmin * i2f(m1) * i2f(ch)), acc);
    }
    for (int o = 8; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
// cx_dotq_fast (HALF = 0) / cx_dotq_half (HALF = 1) for m <= 8 activation rows at once
// (xc + t * xs codes, xd + t * ds scales): each weight block decoded once, then every row's
// sums, each row's arithmetic and reduction exactly the single-row kernel's (bitwise).
template <int TY, int HALF>
__device__ __forceinline__ void cx_dotq_m(const unsigned char* w, const unsigned char* xc, int xs, const float* xd, int ds,
                                          int k, int m, float* acc) {
    const int lane = HALF ? (threadIdx.x & 15) : (threadIdx.x & 31);
    const int BB = TY == 12 ? 144 : 176, QS = TY == 12 ? 16 : 48;
    const int sbl = lane >> 3, j = (lane & 7) >> 1, l0 = (lane & 1) * 16;
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.f;
    for (int sb = sbl; sb < k / 256; sb += (HALF ? 2 : 4)) {
        const unsigned char* b = w + sb * BB;
        const float d = h2f(rd16(b)), dmin = h2f(rd16(b + 2));
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, b + 4, s0, m0);
        scale_min_k4(2 * j + 1, b + 4, s1, m1);
        const uint4 q = *(const uint4*)(b + QS + j * 32 + l0);
        uint4 hb = make_uint4(0, 0, 0, 0);
        if (TY == 13) hb = *(const uint4*)(b + 16 + l0);
        const int e = sb * 256 + j * 64 + l0;
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {hb.x, hb.y, hb.z, hb.w};
        unsigned lo[4], hi[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            lo[i] = qw[i] & 0x0F0F0F0Fu;
            hi[i] = (qw[i] >> 4) & 0x0F0F0F0Fu;
            if (TY == 13) {
                lo[i] |= ((hw[i] >> (2 * j)) & 0x01010101u) << 4;
                hi[i] |= ((hw[i] >> (2 * j + 1)) & 0x01010101u) << 4;
            }
        }
        const float ds0 = d * i2f(s0), ds1 = d * i2f(s1);
        const float dm0 = -dmin * i2f(m0), dm1 = -dmin * i2f(m1);
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= m) break;
            const unsigned char* xr = xc + (size_t)t * xs;
            const uint4 x0 = *(const uint4*)(xr + e), x1 = *(const uint4*)(xr + e + 32);
            const unsigned xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
            int sa = 0, sh = 0, ca = 0, ch = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                sa = __dp4a((int)lo[i], (int)xa[i], sa);
                sh = __dp4a((int)hi[i], (int)xb[i], sh);
                ca = __dp4a(0x01010101, (int)xa[i], ca);
                ch = __dp4a(0x01010101, (int)xb[i], ch);
            }
            const float* xdr = xd + t * ds;
            const float dx0 = xdr[e / 32], dx1 = xdr[e / 32 + 1];
            acc[t] = fmaf(dx0, fmaf(ds0, i2f(sa), dm0 * i2f(ca)), acc[t]);
            acc[t] = fmaf(dx1, fmaf(ds1, i2f(sh), dm1 * i2f(ch)), acc[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t >= m) break;
        float a = acc[t];
        for (int o = HALF ? 8 : 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        acc[t] = a;
    }
}
// Q4_K / Q5_K per-row dot reading a TCX tile (tcx.rs) instead of the GGUF row: the same lane
// mapping, the same 16 qs / qh bytes a lane (they sit in the tile as they are), the same
// scales and min values, the same arithmetic and reduction as cx_dotq_fast (HALF 0) /
// cx_dotq_half (HALF 1): bitwise equal. `tb` the row's tile column (16-aligned), `row` 0..16.
template <int TY, int HALF>
__device__ __forceinline__ float cx_dotq_tile(const unsigned char* tb, int row, const unsigned char* xc, const float* xd, int k) {
    const int lane = HALF ? (threadIdx.x & 15) : (threadIdx.x & 31);
    constexpr int TB = TY == 12 ? 2368 : 2880, SO = TY == 12 ? 2048 : 2560;
    const int sbl = lane >> 3, j = (lane & 7) >> 1, l0 = (lane & 1) * 16;
    float acc = 0.f;
    for (int sb = sbl; sb < k / 256; sb += (HALF ? 2 : 4)) {
        const unsigned char* b = tb + sb * TB;
        const unsigned dw = *(const unsigned*)(b + SO + row * 4);
        const float d = h2f((unsigned short)(dw & 0xFFFFu)), dmin = h2f((unsigned short)(dw >> 16));
        const unsigned char* sm = b + SO + 64 + row * 16;
        const int s0 = sm[2 * j], s1 = sm[2 * j + 1], m0 = sm[8 + 2 * j], m1 = sm[8 + 2 * j + 1];
        const uint4 q = *(const uint4*)(b + (j * 16 + row) * 32 + l0);
        uint4 hb = make_uint4(0, 0, 0, 0);
        if (TY == 13) hb = *(const uint4*)(b + 2048 + row * 32 + l0);
        const int e = sb * 256 + j * 64 + l0;
        const uint4 x0 = *(const uint4*)(xc + e), x1 = *(const uint4*)(xc + e + 32);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {hb.x, hb.y, hb.z, hb.w};
        const unsigned xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
        int sa = 0, sh = 0, ca = 0, ch = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned lo = qw[i] & 0x0F0F0F0Fu, hi = (qw[i] >> 4) & 0x0F0F0F0Fu;
            if (TY == 13) {
                lo |= ((hw[i] >> (2 * j)) & 0x01010101u) << 4;
                hi |= ((hw[i] >> (2 * j + 1)) & 0x01010101u) << 4;
            }
            sa = __dp4a((int)lo, (int)xa[i], sa);
            sh = __dp4a((int)hi, (int)xb[i], sh);
            ca = __dp4a(0x01010101, (int)xa[i], ca);
            ch = __dp4a(0x01010101, (int)xb[i], ch);
        }
        const float dx0 = xd[e / 32], dx1 = xd[e / 32 + 1];
        acc = fmaf(dx0, fmaf(d * i2f(s0), i2f(sa), -dmin * i2f(m0) * i2f(ca)), acc);
        acc = fmaf(dx1, fmaf(d * i2f(s1), i2f(sh), -dmin * i2f(m1) * i2f(ch)), acc);
    }
    for (int o = HALF ? 8 : 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
// A TCX row is passed around as its tile column's address plus the row (0..16) in the low bits
// (tile columns are 16-aligned).
__device__ __forceinline__ const unsigned char* tcx_rowp(const unsigned char* base, unsigned long long estride, int rb,
                                                         unsigned e, int r) {
    return base + (size_t)e * estride + (size_t)(r >> 4) * rb + (r & 15);
}
__device__ __forceinline__ float cx_dot(int ty, const unsigned char* w, const unsigned char* xc, const float* xd, int k, int fast) {
#ifdef CX_TCX_TILES
    if (ty == 112 || ty == 113) {
        const unsigned char* tb = (const unsigned char*)((unsigned long long)w & ~15ull);
        const int row = (int)((unsigned long long)w & 15);
        return ty == 112 ? cx_dotq_tile<12, 0>(tb, row, xc, xd, k) : cx_dotq_tile<13, 0>(tb, row, xc, xd, k);
    }
#endif
    if (fast && (((unsigned long long)w | (unsigned long long)xc) & 15) == 0) {
        if (ty == 12) return cx_dotq_fast<12>(w, xc, xd, k);
        if (ty == 13) return cx_dotq_fast<13>(w, xc, xd, k);
    }
    switch (ty) {
        case 8: return cx_dot8<8>(w, xc, xd, k);
        case 2: return cx_dot8<2>(w, xc, xd, k);
        case 20: return cx_dot8<20>(w, xc, xd, k);
        case 6: return cx_dot8<6>(w, xc, xd, k);
        case 42: return cx_dot8<42>(w, xc, xd, k);
        case 23: return cx_dot8<23>(w, xc, xd, k);
        case 11: return cx_dot8<11>(w, xc, xd, k);
        case 12: return cx_dot8<12>(w, xc, xd, k);
        case 13: return cx_dot8<13>(w, xc, xd, k);
        case 14: return cx_dot8<14>(w, xc, xd, k);
        default: return 0.f;
    }
}

struct ExW { unsigned long long base; unsigned long long estride; int ty; int row_bytes; };

// Stage `nb` bytes at `src` into this warp's shared buffer with aligned 16-byte loads; returns
// the row's first byte there.
__device__ __forceinline__ const unsigned char* cx_stage(const unsigned char* src, int nb, uint4* sm, int lane) {
    const unsigned long long a = (unsigned long long)src & ~15ull;
    const int delta = (int)((unsigned long long)src - a);
    const int n16 = (delta + nb + 15) >> 4;
    for (int i = lane; i < n16; i += 32) sm[i] = __ldg((const uint4*)a + i);
    return (const unsigned char*)sm + delta;
}

// cx_stage with cp.async (sm_80+; synchronous loads before that): the copy is in flight until
// cx_async_wait<N>() and a __syncwarp().
__device__ __forceinline__ const unsigned char* cx_stage_async(const unsigned char* src, int nb, uint4* sm, int lane) {
    const unsigned long long a = (unsigned long long)src & ~15ull;
    const int delta = (int)((unsigned long long)src - a);
    const int n16 = (delta + nb + 15) >> 4;
    for (int i = lane; i < n16; i += 32) {
#if __CUDA_ARCH__ >= 800
        const unsigned dst = (unsigned)__cvta_generic_to_shared(sm + i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"((const uint4*)a + i));
#else
        sm[i] = __ldg((const uint4*)a + i);
#endif
    }
    return (const unsigned char*)sm + delta;
}
__device__ __forceinline__ void cx_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cx_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Experts' gate and up rows, R rows a warp (all staged in shared memory before any is decoded),
// 8 warps a block. Pairs p < T·K are (token p / K, rank p % K) with the routed expert ids[p];
// p >= T·K is the shared expert for token p − T·K. out[p][r]: r < FF gate, else up.
// Grid (2 FF / (8 R), T · (K + 1)), block 256, dynamic smem 8 R × r16 × 16 bytes.
template <int R, bool STAGE = true>
__device__ __forceinline__ void moe_gu_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                            const unsigned char* xq, float* out, int T, int r16, int fast, int pbase) {
    extern __shared__ uint4 dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, p = pbase + blockIdx.y;
    const bool shared = p >= T * CX_TOPK;
    const int t = shared ? p - T * CX_TOPK : p / CX_TOPK;
    const unsigned e = shared ? 0u : ids[p];
    const unsigned char* w[R];
    int ty[R], rr[R];
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const int r = (blockIdx.x * 8 + warp) * R + j;
        rr[j] = r;
        const ExW& m = shared ? (r < CX_FF ? sg : su) : (r < CX_FF ? g : u);
        const int ro = r < CX_FF ? r : r - CX_FF;
        ty[j] = m.ty;
#ifdef CX_TCX_TILES
        if (m.ty >= 112) { w[j] = tcx_rowp((const unsigned char*)m.base, m.estride, m.row_bytes, e, ro); continue; }
#endif
        const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)ro * m.row_bytes;
        w[j] = STAGE ? cx_stage(src, m.row_bytes, dsm + (warp * R + j) * r16, lane) : src;
    }
    __syncwarp();
    const float* xd = (const float*)(xq + (size_t)XQ8_T * CX_HID);
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const float a = cx_dot(ty[j], w[j], xq + (size_t)t * CX_HID, xd + t * (CX_HID / 32), CX_HID, fast);
        if (lane == 0) out[(size_t)p * 2 * CX_FF + rr[j]] = a;
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                            const unsigned char* xq, float* out, int T, int r16, int fast,
                                                            int pbase) {
    moe_gu_body<2>(g, u, sg, su, ids, xq, out, T, r16, fast, pbase);
}

// h = silu(gate) * up per pair, then int8 (fe_q8's rule) into hq: codes [CX_NP][FF], then
// scales [CX_NP][FF / 32]. Grid (FF / 32, T · (K + 1)), block 32.
extern "C" __global__ void cx_silu_q(const float* gu, unsigned char* hq, int pbase) {
    const int c = blockIdx.x, p = pbase + blockIdx.y, lane = threadIdx.x;
    const float g = gu[(size_t)p * 2 * CX_FF + c * 32 + lane];
    const float u = gu[(size_t)p * 2 * CX_FF + CX_FF + c * 32 + lane];
    const float v = silu_f2(g) * u;
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    hq[(size_t)p * CX_FF + c * 32 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) ((float*)(hq + (size_t)CX_NP * CX_FF))[p * (CX_FF / 32) + c] = d;
}

// Experts' down rows (shared expert for p >= T·K): out[p][r], R rows a warp, staged. With
// FUSE, the block first makes pair p's int8 h from gate|up itself (cx_silu_q's arithmetic, into
// shared memory) instead of reading cx_silu_q's output.
// Grid (HID / (8 R), T · (K + 1)), block 256, dynamic smem 8 R × r16 × 16 bytes.
template <int R, bool FUSE, bool STAGE = true>
__device__ __forceinline__ void moe_down_body(ExW d, ExW sd, const unsigned* ids, const unsigned char* hq,
                                              const float* gu, float* out, int T, int r16, int fast, int pbase) {
    extern __shared__ uint4 dsm[];
    __shared__ __align__(16) unsigned char hc[CX_FF];
    __shared__ float hs[CX_FF / 32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, p = pbase + blockIdx.y;
    const bool shared = p >= T * CX_TOPK;
    const ExW& m = shared ? sd : d;
    const unsigned e = shared ? 0u : ids[p];
    const unsigned char* w[R];
    int rr[R];
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const int r = (blockIdx.x * 8 + warp) * R + j;
        rr[j] = r;
#ifdef CX_TCX_TILES
        if (m.ty >= 112) { w[j] = tcx_rowp((const unsigned char*)m.base, m.estride, m.row_bytes, e, r); continue; }
#endif
        const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)r * m.row_bytes;
        w[j] = STAGE ? cx_stage(src, m.row_bytes, dsm + (warp * R + j) * r16, lane) : src;
    }
    const unsigned char* hcodes;
    const float* hd;
    if (FUSE) {
        for (int i = threadIdx.x; i < CX_FF; i += blockDim.x) {
            const float g = gu[(size_t)p * 2 * CX_FF + i];
            const float u = gu[(size_t)p * 2 * CX_FF + CX_FF + i];
            const float v = silu_f2(g) * u;
            float amax = fabsf(v);
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const float dd = amax / 127.0f;
            const int q = dd == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / dd), -127.f), 127.f);
            hc[i] = (unsigned char)(signed char)q;
            if ((i & 31) == 0) hs[i / 32] = dd;
        }
        __syncthreads();
        hcodes = hc;
        hd = hs;
    } else {
        __syncwarp();
        hcodes = hq + (size_t)p * CX_FF;
        hd = (const float*)(hq + (size_t)CX_NP * CX_FF) + p * (CX_FF / 32);
    }
#ifdef CX_TCX_TILES
    if (R % 2 == 0 && fast >= 2 && CX_FF == 512 && m.ty >= 112 && (((unsigned long long)hcodes) & 15) == 0) {
        // The TCX tile twin of the half-warp branch below (bitwise equal).
        const int h = lane >> 4;
        #pragma unroll
        for (int j0 = 0; j0 < R; j0 += 2) {
            const unsigned long long wp = (unsigned long long)w[j0 + h];
            const unsigned char* tb = (const unsigned char*)(wp & ~15ull);
            const int row = (int)(wp & 15);
            const float a = m.ty == 112 ? cx_dotq_tile<12, 1>(tb, row, hcodes, hd, CX_FF)
                                        : cx_dotq_tile<13, 1>(tb, row, hcodes, hd, CX_FF);
            if ((lane & 15) == 0) out[(size_t)p * CX_HID + rr[j0 + h]] = a;
        }
        return;
    }
#endif
    const bool aligned = ((((unsigned long long)w[0]) | ((unsigned long long)w[R - 1]) | (unsigned long long)hcodes) & 15) == 0;
    if (R % 2 == 0 && fast >= 2 && CX_FF == 512 && (m.ty == 12 || m.ty == 13) && aligned) {
        // fast = 2: rows two at a time on the two halves of the warp.
        const int h = lane >> 4;
        #pragma unroll
        for (int j0 = 0; j0 < R; j0 += 2) {
            const float a = m.ty == 12 ? cx_dotq_half<12>(w[j0 + h], hcodes, hd, CX_FF)
                                       : cx_dotq_half<13>(w[j0 + h], hcodes, hd, CX_FF);
            if ((lane & 15) == 0) out[(size_t)p * CX_HID + rr[j0 + h]] = a;
        }
        return;
    }
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const float a = cx_dot(m.ty, w[j], hcodes, hd, CX_FF, fast);
        if (lane == 0) out[(size_t)p * CX_HID + rr[j]] = a;
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down(ExW d, ExW sd, const unsigned* ids,
                                                              const unsigned char* hq, float* out, int T, int r16, int fast,
                                                              int pbase) {
    moe_down_body<2, false>(d, sd, ids, hq, nullptr, out, T, r16, fast, pbase);
}

// The same per-pair kernels reading expert rows straight from global memory instead of staging
// them in shared memory first (at decode nothing reuses a staged row; staging adds a pass and
// holds shared memory that limits blocks an SM), with 2 or 4 rows a warp. Same rows, same
// cx_dot: outputs bitwise equal to the staged kernels. Grid x = rows / (8 R). TANG_CODER_MOE.
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_d2(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                               const unsigned char* xq, float* out, int T, int r16,
                                                               int fast, int pbase) {
    moe_gu_body<2, false>(g, u, sg, su, ids, xq, out, T, r16, fast, pbase);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_d4(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                               const unsigned char* xq, float* out, int T, int r16,
                                                               int fast, int pbase) {
    moe_gu_body<4, false>(g, u, sg, su, ids, xq, out, T, r16, fast, pbase);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_d2(ExW d, ExW sd, const unsigned* ids,
                                                                 const unsigned char* hq, float* out, int T, int r16,
                                                                 int fast, int pbase) {
    moe_down_body<2, false, false>(d, sd, ids, hq, nullptr, out, T, r16, fast, pbase);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_d4(ExW d, ExW sd, const unsigned* ids,
                                                                 const unsigned char* hq, float* out, int T, int r16,
                                                                 int fast, int pbase) {
    moe_down_body<4, false, false>(d, sd, ids, hq, nullptr, out, T, r16, fast, pbase);
}

// cx_moe_down_d2 / _d4 with SiLU(gate) * up and its int8 made in the block from gate|up (FUSE:
// cx_silu_q's arithmetic, a 32-wide chunk a warp), so the cx_silu_q launch goes away. hq is
// unused. Bitwise the unfused kernels' outputs.
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_d2f(ExW d, ExW sd, const unsigned* ids,
                                                                  const float* gu, float* out, int T, int r16,
                                                                  int fast, int pbase) {
    moe_down_body<2, true, false>(d, sd, ids, nullptr, gu, out, T, r16, fast, pbase);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_d4f(ExW d, ExW sd, const unsigned* ids,
                                                                  const float* gu, float* out, int T, int r16,
                                                                  int fast, int pbase) {
    moe_down_body<4, true, false>(d, sd, ids, nullptr, gu, out, T, r16, fast, pbase);
}

// Per-pair expert kernels with the rows double-buffered: each warp runs L groups of R rows,
// copying group i + 1 into shared memory (cp.async) while it decodes group i, so a warp always
// has a group's loads in flight. Same rows and cx_dot (and the down kernel's half-warp path) as
// the staged kernels: bitwise equal outputs. Grid x = rows / (8 R L); dynamic smem
// 2 × 8 R × r16 × 16 bytes. TANG_CODER_MOE=8.
template <int R, int L>
__device__ __forceinline__ void moe_gu_db_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                               const unsigned char* xq, float* out, int T, int r16, int fast, int pbase) {
    extern __shared__ uint4 dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, p = pbase + blockIdx.y;
    const bool shared = p >= T * CX_TOPK;
    const int t = shared ? p - T * CX_TOPK : p / CX_TOPK;
    const unsigned e = shared ? 0u : ids[p];
    const float* xd = (const float*)(xq + (size_t)XQ8_T * CX_HID);
    const unsigned char* w[2][R];
    int ty[2][R], rr[2][R];
    auto issue = [&](int it, int b) {
        #pragma unroll
        for (int j = 0; j < R; j++) {
            const int r = ((blockIdx.x * L + it) * 8 + warp) * R + j;
            rr[b][j] = r;
            const ExW& m = shared ? (r < CX_FF ? sg : su) : (r < CX_FF ? g : u);
            const int ro = r < CX_FF ? r : r - CX_FF;
            ty[b][j] = m.ty;
            const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)ro * m.row_bytes;
            w[b][j] = cx_stage_async(src, m.row_bytes, dsm + ((b * 8 + warp) * R + j) * r16, lane);
        }
        cx_async_commit();
    };
    issue(0, 0);
    #pragma unroll
    for (int it = 0; it < L; it++) {
        const int b = it & 1;
        if (it + 1 < L) {
            issue(it + 1, b ^ 1);
            cx_async_wait<1>();
        } else {
            cx_async_wait<0>();
        }
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < R; j++) {
            const float a = cx_dot(ty[b][j], w[b][j], xq + (size_t)t * CX_HID, xd + t * (CX_HID / 32), CX_HID, fast);
            if (lane == 0) out[(size_t)p * 2 * CX_FF + rr[b][j]] = a;
        }
        __syncwarp();
    }
}
template <int R, int L>
__device__ __forceinline__ void moe_down_db_body(ExW d, ExW sd, const unsigned* ids, const unsigned char* hq,
                                                 float* out, int T, int r16, int fast, int pbase) {
    extern __shared__ uint4 dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, p = pbase + blockIdx.y;
    const bool shared = p >= T * CX_TOPK;
    const ExW& m = shared ? sd : d;
    const unsigned e = shared ? 0u : ids[p];
    const unsigned char* hcodes = hq + (size_t)p * CX_FF;
    const float* hd = (const float*)(hq + (size_t)CX_NP * CX_FF) + p * (CX_FF / 32);
    const unsigned char* w[2][R];
    int rr[2][R];
    auto issue = [&](int it, int b) {
        #pragma unroll
        for (int j = 0; j < R; j++) {
            const int r = ((blockIdx.x * L + it) * 8 + warp) * R + j;
            rr[b][j] = r;
            const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)r * m.row_bytes;
            w[b][j] = cx_stage_async(src, m.row_bytes, dsm + ((b * 8 + warp) * R + j) * r16, lane);
        }
        cx_async_commit();
    };
    issue(0, 0);
    #pragma unroll
    for (int it = 0; it < L; it++) {
        const int b = it & 1;
        if (it + 1 < L) {
            issue(it + 1, b ^ 1);
            cx_async_wait<1>();
        } else {
            cx_async_wait<0>();
        }
        __syncwarp();
        const bool aligned = ((((unsigned long long)w[b][0]) | ((unsigned long long)w[b][R - 1]) | (unsigned long long)hcodes) & 15) == 0;
        if (R % 2 == 0 && fast >= 2 && CX_FF == 512 && (m.ty == 12 || m.ty == 13) && aligned) {
            const int h = lane >> 4;
            #pragma unroll
            for (int j0 = 0; j0 < R; j0 += 2) {
                const float a = m.ty == 12 ? cx_dotq_half<12>(w[b][j0 + h], hcodes, hd, CX_FF)
                                           : cx_dotq_half<13>(w[b][j0 + h], hcodes, hd, CX_FF);
                if ((lane & 15) == 0) out[(size_t)p * CX_HID + rr[b][j0 + h]] = a;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < R; j++) {
                const float a = cx_dot(m.ty, w[b][j], hcodes, hd, CX_FF, fast);
                if (lane == 0) out[(size_t)p * CX_HID + rr[b][j]] = a;
            }
        }
        __syncwarp();
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_db(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                               const unsigned char* xq, float* out, int T, int r16,
                                                               int fast, int pbase) {
    moe_gu_db_body<2, 4>(g, u, sg, su, ids, xq, out, T, r16, fast, pbase);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_db(ExW d, ExW sd, const unsigned* ids,
                                                                 const unsigned char* hq, float* out, int T, int r16,
                                                                 int fast, int pbase) {
    moe_down_db_body<2, 4>(d, sd, ids, hq, out, T, r16, fast, pbase);
}

// ---- grouped experts: each distinct expert's rows read once for all the pairs routed to it ----
//
// A plan (cx_moe_plan) groups pairs pbase .. pbase + n by expert (the shared expert is one more
// key, CX_NE): plan[0] = groups, plan[1 + g] = group g's first index into the list, plan[1 + ng]
// = n, and plan[CX_PLAN_LIST + i] the pairs in (expert, pair) order. The grouped kernels launch
// a fixed grid of min(n, CX_NE + 1) groups (blocks past plan[0] return), so a window's graph holds
// for any routing. Each pair's dot is the per-pair kernels' (same staged rows, same cx_dot), so
// outputs are bitwise those of cx_moe_gu / cx_moe_down. Block 128 (n <= CX_NP <= 128).
#define CX_PLAN_LIST (2 + CX_NP)
// (Only built where CX_NP fits its static shared memory; wide modules use cx_moe_plan_w.)
#if CX_NP <= 4096
extern "C" __global__ void cx_moe_plan(const unsigned* ids, int T, int pbase, int n, unsigned* plan) {
    const int i = threadIdx.x;
    __shared__ int key[CX_NP];
    __shared__ int first[CX_NP];
    if (i < n) {
        const int p = pbase + i;
        key[i] = p >= T * CX_TOPK ? CX_NE : (int)ids[p];
    }
    __syncthreads();
    if (i < n) {
        int f = 1;
        for (int q = 0; q < i; q++) if (key[q] == key[i]) { f = 0; break; }
        first[i] = f;
    }
    __syncthreads();
    if (i < n) {
        const int k = key[i];
        int rank = 0, gid = 0;
        for (int j = 0; j < n; j++) {
            const int kj = key[j];
            if (kj < k || (kj == k && j < i)) rank++;
            if (kj < k && first[j]) gid++;
        }
        plan[CX_PLAN_LIST + rank] = (unsigned)(pbase + i);
        if (first[i]) plan[1 + gid] = (unsigned)rank;
    }
    if (i == 0) {
        int ng = 0;
        for (int j = 0; j < n; j++) ng += first[j];
        plan[0] = (unsigned)ng;
        plan[1 + ng] = (unsigned)n;
    }
}

#endif

// cx_moe_plan for many pairs (wide prefill windows, up to CX_NP): one block of 1024 threads,
// an expert histogram, a prefix over the keys, then a scatter. Pairs within a group land in any
// order, which can't change a result (each pair's dot is its own). Same plan layout.
// An expert with more than CX_GCHUNK pairs gets one group per CX_GCHUNK of them (batch modules:
// the shared expert, which every row routes to, would otherwise be one serial group of B pairs).
#ifndef CX_GCHUNK
#define CX_GCHUNK (1 << 30)
#endif
extern "C" __global__ void __launch_bounds__(1024) cx_moe_plan_w(const unsigned* ids, int T, int pbase, int n, unsigned* plan) {
    __shared__ int cnt[CX_NE + 1], start[CX_NE + 1], gid[CX_NE + 1], fill[CX_NE + 1];
    __shared__ int ng;
    for (int k = threadIdx.x; k <= CX_NE; k += blockDim.x) { cnt[k] = 0; fill[k] = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int p = pbase + i;
        atomicAdd(&cnt[p >= T * CX_TOPK ? CX_NE : (int)ids[p]], 1);
    }
    __syncthreads();
    // Prefix over the keys (pairs and group counts), a warp-scan per 32 keys then over the
    // warps' totals; one thread per key writes its groups' starts.
    {
        __shared__ int wsum[32], wgrp[32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        int base_a = 0, base_g = 0;
        for (int k0 = 0; k0 <= CX_NE; k0 += blockDim.x) {
            const int kk = k0 + threadIdx.x;
            const int c = kk <= CX_NE ? cnt[kk] : 0;
            const int gc = (c + CX_GCHUNK - 1) / CX_GCHUNK;
            int ia = c, ig = gc;
            for (int o = 1; o < 32; o <<= 1) {
                const int va = __shfl_up_sync(0xffffffffu, ia, o), vg = __shfl_up_sync(0xffffffffu, ig, o);
                if (lane >= o) { ia += va; ig += vg; }
            }
            if (lane == 31) { wsum[wid] = ia; wgrp[wid] = ig; }
            __syncthreads();
            if (wid == 0) {
                const int nw = blockDim.x >> 5;
                int sa = lane < nw ? wsum[lane] : 0, sgp = lane < nw ? wgrp[lane] : 0;
                for (int o = 1; o < 32; o <<= 1) {
                    const int va = __shfl_up_sync(0xffffffffu, sa, o), vg = __shfl_up_sync(0xffffffffu, sgp, o);
                    if (lane >= o) { sa += va; sgp += vg; }
                }
                if (lane < nw) { wsum[lane] = sa; wgrp[lane] = sgp; }
            }
            __syncthreads();
            const int pa = base_a + (wid ? wsum[wid - 1] : 0) + ia - c;
            const int pg = base_g + (wid ? wgrp[wid - 1] : 0) + ig - gc;
            if (kk <= CX_NE) {
                start[kk] = pa;
                for (int o = 0, q = pg; o < c; o += CX_GCHUNK, q++) plan[1 + q] = (unsigned)(pa + o);
            }
            const int tot_a = wsum[(blockDim.x >> 5) - 1], tot_g = wgrp[(blockDim.x >> 5) - 1];
            __syncthreads();
            base_a += tot_a;
            base_g += tot_g;
        }
        if (threadIdx.x == 0) {
            ng = base_g;
            plan[0] = (unsigned)base_g;
            plan[1 + base_g] = (unsigned)n;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int p = pbase + i;
        const int k = p >= T * CX_TOPK ? CX_NE : (int)ids[p];
        plan[CX_PLAN_LIST + start[k] + atomicAdd(&fill[k], 1)] = (unsigned)p;
    }
}

// cx_moe_gu over groups: grid (2 FF / (8 R), groups).
template <int R>
__device__ __forceinline__ void moe_gu_grouped_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                    const unsigned char* xq, float* out, int T, int r16, int fast,
                                                    const unsigned* plan) {
    extern __shared__ uint4 dsm[];
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = plan[1 + gi], b = plan[2 + gi];
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const unsigned e = shared ? 0u : ids[p0];
    const unsigned char* w[R];
    int ty[R], rr[R];
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const int r = (blockIdx.x * 8 + warp) * R + j;
        rr[j] = r;
        const ExW& m = shared ? (r < CX_FF ? sg : su) : (r < CX_FF ? g : u);
        const int ro = r < CX_FF ? r : r - CX_FF;
        ty[j] = m.ty;
        const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)ro * m.row_bytes;
        w[j] = cx_stage(src, m.row_bytes, dsm + (warp * R + j) * r16, lane);
    }
    __syncwarp();
    const float* xd = (const float*)(xq + (size_t)XQ8_T * CX_HID);
    for (int q = a; q < b; q++) {
        const int p = plan[CX_PLAN_LIST + q];
        const int t = shared ? p - T * CX_TOPK : p / CX_TOPK;
        #pragma unroll
        for (int j = 0; j < R; j++) {
            const float v = cx_dot(ty[j], w[j], xq + (size_t)t * CX_HID, xd + t * (CX_HID / 32), CX_HID, fast);
            if (lane == 0) out[(size_t)p * 2 * CX_FF + rr[j]] = v;
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_g(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                              const unsigned char* xq, float* out, int T, int r16,
                                                              int fast, const unsigned* plan) {
    moe_gu_grouped_body<2>(g, u, sg, su, ids, xq, out, T, r16, fast, plan);
}

// cx_moe_down over groups: grid (HID / (8 R), groups).
template <int R>
__device__ __forceinline__ void moe_down_grouped_body(ExW d, ExW sd, const unsigned* ids, const unsigned char* hq,
                                                      float* out, int T, int r16, int fast, const unsigned* plan) {
    extern __shared__ uint4 dsm[];
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = plan[1 + gi], b = plan[2 + gi];
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const ExW& m = shared ? sd : d;
    const unsigned e = shared ? 0u : ids[p0];
    const unsigned char* w[R];
    int rr[R];
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const int r = (blockIdx.x * 8 + warp) * R + j;
        rr[j] = r;
        const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)r * m.row_bytes;
        w[j] = cx_stage(src, m.row_bytes, dsm + (warp * R + j) * r16, lane);
    }
    __syncwarp();
    for (int q = a; q < b; q++) {
        const int p = plan[CX_PLAN_LIST + q];
        const unsigned char* hcodes = hq + (size_t)p * CX_FF;
        const float* hd = (const float*)(hq + (size_t)CX_NP * CX_FF) + p * (CX_FF / 32);
        const bool aligned = ((((unsigned long long)w[0]) | ((unsigned long long)w[R - 1]) | (unsigned long long)hcodes) & 15) == 0;
        if (R == 2 && fast >= 2 && CX_FF == 512 && (m.ty == 12 || m.ty == 13) && aligned) {
            const int h = lane >> 4;
            const float v = m.ty == 12 ? cx_dotq_half<12>(w[h], hcodes, hd, CX_FF) : cx_dotq_half<13>(w[h], hcodes, hd, CX_FF);
            if ((lane & 15) == 0) out[(size_t)p * CX_HID + rr[h]] = v;
            continue;
        }
        #pragma unroll
        for (int j = 0; j < R; j++) {
            const float v = cx_dot(m.ty, w[j], hcodes, hd, CX_FF, fast);
            if (lane == 0) out[(size_t)p * CX_HID + rr[j]] = v;
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_g(ExW d, ExW sd, const unsigned* ids,
                                                                const unsigned char* hq, float* out, int T, int r16,
                                                                int fast, const unsigned* plan) {
    moe_down_grouped_body<2>(d, sd, ids, hq, out, T, r16, fast, plan);
}

// ---- streaming grouped experts (batch decode: 1-8 tokens an expert) ----
//
// cx_moe_gu_g / cx_moe_down_g with each warp owning NR rows of its group's expert (2 a step)
// instead of 2, the steps' rows staged with cp.async CX_SST steps ahead in a per-warp ring, so a
// block streams NR * 8 rows with loads in flight while it computes, rather than one short block
// per 16 rows (which is block-launch-bound at ~2 pairs a group). Every dot is the grouped
// kernels' (same staged bytes, same cx_dot / cx_dotq_half on them): bitwise equal.
// Grid (rows / (8 NR), groups), block 256, dynamic smem 8 * CX_SST * 2 * r16 * 16 bytes.
#ifndef CX_SST
#define CX_SST 1
#endif
#ifndef CX_SST_DOWN
#define CX_SST_DOWN 2
#endif
#ifndef CX_MT
#define CX_MT 1
#endif
#ifndef CX_GCHUNK_S
#define CX_GCHUNK_S 8
#endif
#ifndef CX_SNR_GU
#define CX_SNR_GU 8
#endif
// Rows a warp stages (and computes, two at a time) per ring step, gate|up only (down: 2).
#ifndef CX_SRP
#define CX_SRP 2
#endif
#ifndef CX_SNR_DOWN
#define CX_SNR_DOWN 16
#endif
template <int NR, bool DOWN, int SST>
__device__ __forceinline__ void moe_stream_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                const unsigned char* xq, float* out, int T, int r16, int fast,
                                                const unsigned* plan, int gi, int bx) {
    extern __shared__ uint4 dsm[];
    if (gi >= (int)plan[0]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = plan[1 + gi], b = plan[2 + gi];
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const unsigned e = shared ? 0u : ids[p0];
    // The group's (at most CX_GCHUNK) activations, staged once a block after the first weight
    // loads are issued: codes [m][K], then scales [m][K / 32].
    constexpr int KA = DOWN ? CX_FF : CX_HID;
    constexpr int ACT16 = (CX_GCHUNK_S * (KA + KA / 8)) / 16;
    unsigned char* act = (unsigned char*)dsm;
    float* actd = (float*)(act + CX_GCHUNK_S * KA);
    constexpr int RP = DOWN ? 2 : CX_SRP;
    uint4* ring = dsm + ACT16 + warp * (SST * RP * r16);
    const int row0 = (bx * 8 + warp) * NR;
    constexpr int NS = NR / RP;
    const unsigned char* wp[SST][RP];
    auto issue = [&](int st) {
        const int slot = st % SST;
        #pragma unroll
        for (int j = 0; j < RP; j++) {
            const int r = row0 + RP * st + j;
            const ExW& m = DOWN ? (shared ? sg : g) : (shared ? (r < CX_FF ? sg : su) : (r < CX_FF ? g : u));
            const int ro = DOWN ? r : (r < CX_FF ? r : r - CX_FF);
            const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)ro * m.row_bytes;
            wp[slot][j] = cx_stage_async(src, m.row_bytes, ring + (slot * RP + j) * r16, lane);
        }
        cx_async_commit();
    };
    #pragma unroll
    for (int st = 0; st < SST - 1; st++) {
        if (st < NS) issue(st); else cx_async_commit();
    }
    {
        const int m = b - a;
        const float* xdg = (const float*)(xq + (size_t)(DOWN ? CX_NP : XQ8_T) * KA);
        for (int i = threadIdx.x; i < m * (KA / 16); i += blockDim.x) {
            const int q = i / (KA / 16), c = i % (KA / 16);
            const int p = plan[CX_PLAN_LIST + a + q];
            const int t = DOWN ? p : (shared ? p - T * CX_TOPK : p / CX_TOPK);
            ((uint4*)act)[q * (KA / 16) + c] = *(const uint4*)(xq + (size_t)t * KA + c * 16);
        }
        for (int i = threadIdx.x; i < m * (KA / 32); i += blockDim.x) {
            const int q = i / (KA / 32), c = i % (KA / 32);
            const int p = plan[CX_PLAN_LIST + a + q];
            const int t = DOWN ? p : (shared ? p - T * CX_TOPK : p / CX_TOPK);
            actd[q * (KA / 32) + c] = xdg[(size_t)t * (KA / 32) + c];
        }
        __syncthreads();
    }
    for (int st = 0; st < NS; st++) {
        if (st + SST - 1 < NS) issue(st + SST - 1); else cx_async_commit();
        cx_async_wait<SST - 1>();
        __syncwarp();
        const int slot = st % SST;
        #pragma unroll
        for (int jj = 0; jj < RP / 2; jj++) {
        const unsigned char* w0 = wp[slot][2 * jj];
        const unsigned char* w1 = wp[slot][2 * jj + 1];
        const int rr0 = row0 + RP * st + 2 * jj;
        const int mt = b - a;
        if (DOWN) {
            const ExW& m = shared ? sg : g;
            const bool al = ((((unsigned long long)w0) | ((unsigned long long)w1) | (unsigned long long)act) & 15) == 0;
            if (CX_MT && fast >= 2 && CX_FF == 512 && (m.ty == 12 || m.ty == 13) && al && mt <= 8) {
                const int h = lane >> 4;
                const unsigned char* wh = h ? w1 : w0;
                float acc[8];
                if (m.ty == 12) cx_dotq_m<12, 1>(wh, act, CX_FF, actd, CX_FF / 32, CX_FF, mt, acc);
                else cx_dotq_m<13, 1>(wh, act, CX_FF, actd, CX_FF / 32, CX_FF, mt, acc);
                if ((lane & 15) == 0) {
                    #pragma unroll
                    for (int t = 0; t < 8; t++)
                        if (t < mt) out[(size_t)plan[CX_PLAN_LIST + a + t] * CX_HID + rr0 + h] = acc[t];
                }
                __syncwarp();
                continue;
            }
            for (int q = a; q < b; q++) {
                const int p = plan[CX_PLAN_LIST + q];
                const unsigned char* hcodes = act + (size_t)(q - a) * CX_FF;
                const float* hd = actd + (q - a) * (CX_FF / 32);
                const bool aligned = ((((unsigned long long)w0) | ((unsigned long long)w1) | (unsigned long long)hcodes) & 15) == 0;
                if (fast >= 2 && CX_FF == 512 && (m.ty == 12 || m.ty == 13) && aligned) {
                    const int h = lane >> 4;
                    const unsigned char* wh = h ? w1 : w0;
                    const float v = m.ty == 12 ? cx_dotq_half<12>(wh, hcodes, hd, CX_FF) : cx_dotq_half<13>(wh, hcodes, hd, CX_FF);
                    if ((lane & 15) == 0) out[(size_t)p * CX_HID + rr0 + h] = v;
                    continue;
                }
                const float v0 = cx_dot(m.ty, w0, hcodes, hd, CX_FF, fast);
                if (lane == 0) out[(size_t)p * CX_HID + rr0] = v0;
                const float v1 = cx_dot(m.ty, w1, hcodes, hd, CX_FF, fast);
                if (lane == 0) out[(size_t)p * CX_HID + rr0 + 1] = v1;
            }
        } else {
            int ty[2];
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int r = rr0 + j;
                ty[j] = (shared ? (r < CX_FF ? sg : su) : (r < CX_FF ? g : u)).ty;
            }
            const bool al = ((((unsigned long long)w0) | ((unsigned long long)w1) | (unsigned long long)act) & 15) == 0;
            if (CX_MT && fast && al && mt <= 8 && (ty[0] == 12 || ty[0] == 13) && ty[1] == ty[0]) {
                float a0[8], a1[8];
                if (ty[0] == 12) {
                    cx_dotq_m<12, 0>(w0, act, CX_HID, actd, CX_HID / 32, CX_HID, mt, a0);
                    cx_dotq_m<12, 0>(w1, act, CX_HID, actd, CX_HID / 32, CX_HID, mt, a1);
                } else {
                    cx_dotq_m<13, 0>(w0, act, CX_HID, actd, CX_HID / 32, CX_HID, mt, a0);
                    cx_dotq_m<13, 0>(w1, act, CX_HID, actd, CX_HID / 32, CX_HID, mt, a1);
                }
                if (lane == 0) {
                    #pragma unroll
                    for (int t = 0; t < 8; t++) {
                        if (t >= mt) break;
                        const size_t p = plan[CX_PLAN_LIST + a + t];
                        out[p * 2 * CX_FF + rr0] = a0[t];
                        out[p * 2 * CX_FF + rr0 + 1] = a1[t];
                    }
                }
                __syncwarp();
                continue;
            }
            for (int q = a; q < b; q++) {
                const int p = plan[CX_PLAN_LIST + q];
                const unsigned char* xc = act + (size_t)(q - a) * CX_HID;
                const float* xdq = actd + (q - a) * (CX_HID / 32);
                const float v0 = cx_dot(ty[0], w0, xc, xdq, CX_HID, fast);
                if (lane == 0) out[(size_t)p * 2 * CX_FF + rr0] = v0;
                const float v1 = cx_dot(ty[1], w1, xc, xdq, CX_HID, fast);
                if (lane == 0) out[(size_t)p * 2 * CX_FF + rr0 + 1] = v1;
            }
        }
        }
        __syncwarp();
    }
    cx_async_wait<0>();
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_s(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                              const unsigned char* xq, float* out, int T, int r16,
                                                              int fast, const unsigned* plan) {
    moe_stream_body<CX_SNR_GU, false, CX_SST>(g, u, sg, su, ids, xq, out, T, r16, fast, plan, blockIdx.y, blockIdx.x);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_s(ExW d, ExW sd, const unsigned* ids,
                                                                const unsigned char* hq, float* out, int T, int r16,
                                                                int fast, const unsigned* plan) {
    moe_stream_body<CX_SNR_DOWN, true, CX_SST_DOWN>(d, d, sd, sd, ids, hq, out, T, r16, fast, plan, blockIdx.y, blockIdx.x);
}

// Persistent, weight-major streaming experts: a grid of about SMs x k blocks, each pulling the
// next work item (a group and a 1/CX_PSPLIT slice of its expert's rows) off an atomic counter and
// streaming that slice through the warps' cp.async rings (CX_PSST stages), so a block moves one
// expert's rows contiguously instead of 8-16 blocks each touching a sliver of it. Every dot is
// moe_stream_body's (bitwise). ctr[0] is the work counter, ctr[1] counts finished blocks; the
// last block out zeroes both, so a captured graph replays with them at zero.
// Dynamic smem 8 * CX_PSST * 2 * r16 * 16 + the activations, as cx_moe_*_s.
#ifndef CX_PSST
#define CX_PSST 3
#endif
#ifndef CX_PSPLIT
#define CX_PSPLIT 1
#endif
#ifndef CX_PSPLIT_DOWN
#define CX_PSPLIT_DOWN 1
#endif
template <bool DOWN, int SPLIT>
__device__ __forceinline__ void moe_persist_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                 const unsigned char* xq, float* out, int T, int r16, int fast,
                                                 const unsigned* plan, unsigned* ctr) {
    constexpr int ROWS = DOWN ? CX_HID : 2 * CX_FF;
    constexpr int NR = ROWS / (8 * SPLIT);
    static_assert(NR % 2 == 0 && NR * 8 * SPLIT == ROWS, "persistent split");
    __shared__ int item;
    const int nitems = (int)plan[0] * SPLIT;
    for (;;) {
        __syncthreads();
        if (threadIdx.x == 0) item = (int)atomicAdd(ctr, 1u);
        __syncthreads();
        const int it = item;
        if (it >= nitems) break;
        moe_stream_body<NR, DOWN, CX_PSST>(g, u, sg, su, ids, xq, out, T, r16, fast, plan, it / SPLIT, it % SPLIT);
    }
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(ctr + 1, 1u) == gridDim.x - 1) {
            ctr[0] = 0u;
            ctr[1] = 0u;
            __threadfence();
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_p(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                              const unsigned char* xq, float* out, int T, int r16,
                                                              int fast, const unsigned* plan, unsigned* ctr) {
    moe_persist_body<false, CX_PSPLIT>(g, u, sg, su, ids, xq, out, T, r16, fast, plan, ctr);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_p(ExW d, ExW sd, const unsigned* ids,
                                                                const unsigned char* hq, float* out, int T, int r16,
                                                                int fast, const unsigned* plan, unsigned* ctr) {
    moe_persist_body<true, CX_PSPLIT_DOWN>(d, d, sd, sd, ids, hq, out, T, r16, fast, plan, ctr);
}

// ---- grouped experts on int8 tensor cores for batch windows (<= 8 tokens a group) ----
//
// SwapAB: each warp owns 16 rows of its group's expert (M), the group's <= 8 tokens are N, and
// K steps 32 weights (one Q4_K / Q5_K sub-block) a mma.sync m16n8k32 s8. A lane loads its rows'
// nibbles straight from global as 32-bit words (one superblock of loads in flight ahead of the
// math), the tokens' codes, scales and per-32 code sums sit in shared memory, and each sub-block's
// int32 result gets the row's scale and min in the epilogue:
// acc += dx * (d sc * dot - dmin m * sum(x)). Not the per-pair dot's summation order: gated by
// flash-batch-check's tolerance mode. Other weight types (the shared expert's Q8_0) take the
// per-token cx_dot. Grid (rows / 128, groups), block 256.
__device__ __forceinline__ void mma_s8_16832(int* d, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
#else
    d[0] = d[1] = d[2] = d[3] = 0;
#endif
}
// One superblock of a row's operands for this lane: d, dmin, the 12 scale bytes, and the qs
// words at tig*4 and 16 + tig*4 of each of the 4 sub-block pairs (and qh's for Q5_K).
struct SbK { float d, dm; unsigned sc[3]; unsigned q[4][2]; unsigned h[2]; };
template <int TY>
__device__ __forceinline__ void sbk_load(const unsigned char* b, int tig, SbK& o) {
    const int QS = TY == 12 ? 16 : 48;
    o.d = h2f(rd16(b));
    o.dm = h2f(rd16(b + 2));
    o.sc[0] = *(const unsigned*)(b + 4);
    o.sc[1] = *(const unsigned*)(b + 8);
    o.sc[2] = *(const unsigned*)(b + 12);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        o.q[j][0] = *(const unsigned*)(b + QS + j * 32 + tig * 4);
        o.q[j][1] = *(const unsigned*)(b + QS + j * 32 + 16 + tig * 4);
    }
    if (TY == 13) {
        o.h[0] = *(const unsigned*)(b + 16 + tig * 4);
        o.h[1] = *(const unsigned*)(b + 16 + 16 + tig * 4);
    } else {
        o.h[0] = o.h[1] = 0;
    }
}
// sbk_load from a TCX tile (tcx.rs): row g + 8 half of the tile at `t`, the same SbK the GGUF
// row gives (the 12 scale bytes re-packed from the unpacked 6-bit values: every bit is used,
// so they are the GGUF's bytes).
template <int TY>
__device__ __forceinline__ void sbk_load_t(const unsigned char* t, int g, int tig, int half, SbK& o) {
    constexpr int SO = TY == 12 ? 2048 : 2560;
    const int row = g + 8 * half;
    const unsigned dw = *(const unsigned*)(t + SO + row * 4);
    o.d = h2f((unsigned short)(dw & 0xFFFFu));
    o.dm = h2f((unsigned short)(dw >> 16));
    const uint4 sm = *(const uint4*)(t + SO + 64 + row * 16);
    // s0..3 = sm.x bytes, s4..7 = sm.y, m0..3 = sm.z, m4..7 = sm.w.
    const unsigned s03 = sm.x, s47 = sm.y, m03 = sm.z, m47 = sm.w;
    o.sc[0] = s03 | ((s47 >> 4) & 0x03030303u) << 6;
    o.sc[1] = m03 | ((m47 >> 4) & 0x03030303u) << 6;
    o.sc[2] = (s47 & 0x0F0F0F0Fu) | (m47 & 0x0F0F0F0Fu) << 4;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        o.q[j][0] = *(const unsigned*)(t + (j * 16 + row) * 32 + tig * 4);
        o.q[j][1] = *(const unsigned*)(t + (j * 16 + row) * 32 + 16 + tig * 4);
    }
    if (TY == 13) {
        o.h[0] = *(const unsigned*)(t + 2048 + row * 32 + tig * 4);
        o.h[1] = *(const unsigned*)(t + 2048 + row * 32 + 16 + tig * 4);
    } else {
        o.h[0] = o.h[1] = 0;
    }
}
#ifndef CX_MMA_PF
#define CX_MMA_PF 1
#endif
#ifndef CX_MMA_MINB
#define CX_MMA_MINB 1
#endif
// scale_min_k4 on the 12 scale bytes held as three words (sub-block j a compile-time constant).
__device__ __forceinline__ void scale_min_w(int j, const unsigned* sc, int& s, int& m) {
    auto by = [&](int i) { return (int)((sc[i >> 2] >> ((i & 3) * 8)) & 0xFFu); };
    if (j < 4) { s = by(j) & 63; m = by(j + 4) & 63; }
    else { s = (by(j + 4) & 0xF) | ((by(j - 4) >> 6) << 4); m = (by(j + 4) >> 4) | ((by(j) >> 6) << 4); }
}
template <int TY, bool DOWN>
__device__ __forceinline__ void moe_mma_rows(const unsigned char* wa, const unsigned char* wb, int rb, int m,
                                             const unsigned char* act, const float* actd, const int* asum,
                                             float acc[4]) {
    constexpr int K = DOWN ? CX_FF : CX_HID;
    constexpr int NSB = K / 256;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int BB = TY == 12 ? 144 : 176;
    const int na = 2 * tig, nb = 2 * tig + 1;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    SbK ra, rb2, na_, nb_;
    sbk_load<TY>(wa, tig, ra);
    sbk_load<TY>(wb, tig, rb2);
    // Token g's B operand base (rows past m read token 0's codes: their columns are dropped).
    const unsigned char* xg = act + (size_t)(g < m ? g : 0) * K;
    for (int sb = 0; sb < NSB; sb++) {
        if (CX_MMA_PF && sb + 1 < NSB) {
            sbk_load<TY>(wa + (sb + 1) * BB, tig, na_);
            sbk_load<TY>(wb + (sb + 1) * BB, tig, nb_);
        }
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            const int j = s >> 1, hi = s & 1;
            unsigned a[4];
            #pragma unroll
            for (int w = 0; w < 2; w++) {
                unsigned qa = hi ? (ra.q[j][w] >> 4) & 0x0F0F0F0Fu : ra.q[j][w] & 0x0F0F0F0Fu;
                unsigned qb = hi ? (rb2.q[j][w] >> 4) & 0x0F0F0F0Fu : rb2.q[j][w] & 0x0F0F0F0Fu;
                if (TY == 13) {
                    qa |= ((ra.h[w] >> s) & 0x01010101u) << 4;
                    qb |= ((rb2.h[w] >> s) & 0x01010101u) << 4;
                }
                a[w * 2] = qa;
                a[w * 2 + 1] = qb;
            }
            const int e = sb * 256 + s * 32;
            const unsigned b0 = *(const unsigned*)(xg + e + tig * 4);
            const unsigned b1 = *(const unsigned*)(xg + e + 16 + tig * 4);
            int d[4];
            mma_s8_16832(d, a, b0, b1);
            // Columns 2 tig, 2 tig + 1's code sums over the block: precomputed (asum), or token
            // g's from the quad's B fragments, then shuffled to the columns.
            int sqa, sqb;
            if (asum) {
                sqa = na < m ? asum[na * (K / 32) + e / 32] : 0;
                sqb = nb < m ? asum[nb * (K / 32) + e / 32] : 0;
            } else {
                int sq = __dp4a((int)b0, 0x01010101, __dp4a((int)b1, 0x01010101, 0));
                sq += __shfl_xor_sync(0xffffffffu, sq, 1);
                sq += __shfl_xor_sync(0xffffffffu, sq, 2);
                sqa = __shfl_sync(0xffffffffu, sq, (2 * tig) * 4);
                sqb = __shfl_sync(0xffffffffu, sq, (2 * tig + 1) * 4);
            }
            int s0, m0, s1, m1;
            scale_min_w(s, ra.sc, s0, m0);
            scale_min_w(s, rb2.sc, s1, m1);
            const float da = ra.d * i2f(s0), ma = ra.dm * i2f(m0);
            const float db = rb2.d * i2f(s1), mb = rb2.dm * i2f(m1);
            const int blk = e / 32;
            if (na < m) {
                const float dx = actd[na * (K / 32) + blk], sx = i2f(sqa);
                acc[0] = fmaf(dx, fmaf(da, i2f(d[0]), -ma * sx), acc[0]);
                acc[2] = fmaf(dx, fmaf(db, i2f(d[2]), -mb * sx), acc[2]);
            }
            if (nb < m) {
                const float dx = actd[nb * (K / 32) + blk], sx = i2f(sqb);
                acc[1] = fmaf(dx, fmaf(da, i2f(d[1]), -ma * sx), acc[1]);
                acc[3] = fmaf(dx, fmaf(db, i2f(d[3]), -mb * sx), acc[3]);
            }
        }
        if (sb + 1 < NSB) {
            if (CX_MMA_PF) { ra = na_; rb2 = nb_; }
            else { sbk_load<TY>(wa + (sb + 1) * BB, tig, ra); sbk_load<TY>(wb + (sb + 1) * BB, tig, rb2); }
        }
    }
}
template <bool DOWN>
__device__ __forceinline__ void moe_mma_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                             const unsigned char* xq, float* out, int T, int fast,
                                             const unsigned* plan) {
    extern __shared__ uint4 dsm[];
    constexpr int KA = DOWN ? CX_FF : CX_HID;
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = plan[1 + gi], b = plan[2 + gi], m = b - a;
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const unsigned e = shared ? 0u : ids[p0];
    unsigned char* act = (unsigned char*)dsm;
    float* actd = (float*)(act + 8 * KA);
    int* asum = (int*)(actd + 8 * (KA / 32));
    __shared__ int pair_s[8];
    const float* xdg = (const float*)(xq + (size_t)(DOWN ? CX_NP : XQ8_T) * KA);
    if (threadIdx.x < 8) pair_s[threadIdx.x] = threadIdx.x < m ? (int)plan[CX_PLAN_LIST + a + threadIdx.x] : 0;
    __syncthreads();
    for (int i = threadIdx.x; i < m * (KA / 16); i += blockDim.x) {
        const int q = i / (KA / 16), c = i % (KA / 16);
        const int p = pair_s[q];
        const int t = DOWN ? p : (shared ? p - T * CX_TOPK : p / CX_TOPK);
        ((uint4*)act)[i] = *(const uint4*)(xq + (size_t)t * KA + c * 16);
    }
    for (int i = threadIdx.x; i < m * (KA / 32); i += blockDim.x) {
        const int q = i / (KA / 32), c = i % (KA / 32);
        const int p = pair_s[q];
        const int t = DOWN ? p : (shared ? p - T * CX_TOPK : p / CX_TOPK);
        actd[i] = xdg[(size_t)t * (KA / 32) + c];
    }
    __syncthreads();
    const int r0 = (blockIdx.x * 8 + warp) * 16;
    const int gq = lane >> 2, tig = lane & 3;
    const ExW& mw = DOWN ? (shared ? sg : g) : (shared ? (r0 < CX_FF ? sg : su) : (r0 < CX_FF ? g : u));
    const int ro = DOWN ? r0 : (r0 < CX_FF ? r0 : r0 - CX_FF);
    const size_t ostride = DOWN ? CX_HID : 2 * CX_FF;
    const unsigned char* wbase = (const unsigned char*)mw.base + (size_t)e * mw.estride;
    if (mw.ty == 12 || mw.ty == 13) {
        const unsigned char* wa = wbase + (size_t)(ro + gq) * mw.row_bytes;
        const unsigned char* wb = wa + (size_t)8 * mw.row_bytes;
        float acc[4];
        if (mw.ty == 12) moe_mma_rows<12, DOWN>(wa, wb, 0, m, act, actd, nullptr, acc);
        else moe_mma_rows<13, DOWN>(wa, wb, 0, m, act, actd, nullptr, acc);
        if (2 * tig < m) {
            out[(size_t)pair_s[2 * tig] * ostride + r0 + gq] = acc[0];
            out[(size_t)pair_s[2 * tig] * ostride + r0 + gq + 8] = acc[2];
        }
        if (2 * tig + 1 < m) {
            out[(size_t)pair_s[2 * tig + 1] * ostride + r0 + gq] = acc[1];
            out[(size_t)pair_s[2 * tig + 1] * ostride + r0 + gq + 8] = acc[3];
        }
        return;
    }
    // Other types: the per-pair dot, a row at a time.
    for (int r = 0; r < 16; r++) {
        const unsigned char* w = wbase + (size_t)(ro + r) * mw.row_bytes;
        for (int q = 0; q < m; q++) {
            const float v = cx_dot(mw.ty, w, act + (size_t)q * KA, actd + q * (KA / 32), KA, fast);
            if (lane == 0) out[(size_t)pair_s[q] * ostride + r0 + r] = v;
        }
    }
}
extern "C" __global__ void __launch_bounds__(256, CX_MMA_MINB) cx_moe_gu_m(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                              const unsigned char* xq, float* out, int T, int r16,
                                                              int fast, const unsigned* plan) {
    moe_mma_body<false>(g, u, sg, su, ids, xq, out, T, fast, plan);
}
extern "C" __global__ void __launch_bounds__(256, CX_MMA_MINB) cx_moe_down_m(ExW d, ExW sd, const unsigned* ids,
                                                                const unsigned char* hq, float* out, int T, int r16,
                                                                int fast, const unsigned* plan) {
    moe_mma_body<true>(d, d, sd, sd, ids, hq, out, T, fast, plan);
}

// ---- TCX experts: repacked 16 x 256 tiles (tcx.rs) on int8 tensor cores ----
//
// A warp owns one 16-row tile column of its group's expert; the group's <= 8 tokens are N. A
// step of 32 columns is two m16n8k32 s8 MMAs: the codes (A straight from the tile, a shift and
// mask a register) and the sub-block mins replicated over the step (A' = m_j, so the MMA gives
// m_j * sum(x) accumulated over the superblock). The 6-bit scales multiply the code MMA's int32
// (one IMAD a value); activations are int8 with one scale a token a 256 columns (Q8_K-style), so
// the only float math is once a superblock: acc += dx (d I1 - dmin I2). Groups of other
// weight types (a Q8_0 shared expert) take the per-token cx_dot on the per-32 codes `xq`.
// Grid (rows / 128, groups), block 256; dynamic smem 8 K + 8 (K / 256) 4.
__device__ __forceinline__ void mma_s8_16832c(int* d, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
#ifndef CX_TCX_SGU
#define CX_TCX_SGU 4
#endif
#ifndef CX_TCX_SDN
#define CX_TCX_SDN 2
#endif
// h: qh words (row g, cols 4 tig..), (row g, 16 + 4 tig..), (row g + 8, ..), (row g + 8, 16 + ..).
struct TcxT { uint4 c[4]; uint4 h; unsigned dw0, dw1; uint4 s0, s1; };
template <bool Q5>
__device__ __forceinline__ void tcx_load(const unsigned char* t, int lane, TcxT& o) {
    const int g = lane >> 2, tig = lane & 3;
    // codes [q][row][32]: register c = word at (q, g + 8 (c & 1), 16 (c >> 1) + 4 tig).
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        const unsigned char* r0 = t + (q * 16 + g) * 32 + 4 * tig;
        o.c[q] = make_uint4(__ldg((const unsigned*)r0), __ldg((const unsigned*)(r0 + 256)),
                            __ldg((const unsigned*)(r0 + 16)), __ldg((const unsigned*)(r0 + 256 + 16)));
    }
    const unsigned char* sc = t + 2048;
    if (Q5) {
        const unsigned char* qh = t + 2048;
        o.h = make_uint4(__ldg((const unsigned*)(qh + g * 32 + 4 * tig)), __ldg((const unsigned*)(qh + g * 32 + 16 + 4 * tig)),
                         __ldg((const unsigned*)(qh + (g + 8) * 32 + 4 * tig)),
                         __ldg((const unsigned*)(qh + (g + 8) * 32 + 16 + 4 * tig)));
        sc += 512;
    }
    o.dw0 = __ldg((const unsigned*)(sc + g * 4));
    o.dw1 = __ldg((const unsigned*)(sc + (g + 8) * 4));
    o.s0 = __ldg((const uint4*)(sc + 64 + g * 16));
    o.s1 = __ldg((const uint4*)(sc + 64 + (g + 8) * 16));
}
__device__ __forceinline__ unsigned u4w(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
// xg: token g's permuted codes (+ 8 tig); da / db: tokens 2 tig, 2 tig + 1's per-256 scales.
// P32: per-32 activations (fe_q8 codes, unpermuted; da / db per-32 scales), float math a step.
template <bool Q5, int K, bool P32>
__device__ __forceinline__ void tcx_rows(const unsigned char* tile, int m, const unsigned char* xg, const float* da,
                                         const float* db, float acc[4], int sb0, int sb1, const TcxT& first) {
    constexpr int NSB = K / 256;
    constexpr int TB = Q5 ? 2880 : 2368;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int na = 2 * tig, nb = 2 * tig + 1;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    TcxT cur = first, nxt;
    for (int sb = sb0; sb < sb1; sb++) {
        if (sb + 1 < sb1) tcx_load<Q5>(tile + (sb + 1) * TB, lane, nxt);
        int i1[4] = {0, 0, 0, 0}, i2[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            const int qp = s >> 1;
            unsigned a[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const unsigned w = u4w(cur.c[qp], c);
                unsigned v = (s & 1) ? (w >> 4) & 0x0F0F0F0Fu : w & 0x0F0F0F0Fu;
                if (Q5) {
                    // register c: row half c & 1, column half c >> 1; bit s of those qh bytes.
                    const unsigned h = u4w(cur.h, 2 * (c & 1) + (c >> 1));
                    v |= s <= 4 ? (h << (4 - s)) & 0x10101010u : (h >> (s - 4)) & 0x10101010u;
                }
                a[c] = v;
            }
            uint2 bb;
            if (P32) {
                bb.x = __ldg((const unsigned*)(xg + sb * 256 + s * 32));
                bb.y = __ldg((const unsigned*)(xg + sb * 256 + s * 32 + 16));
            } else {
                bb = __ldg((const uint2*)(xg + sb * 256 + s * 32));
            }
            int d[4];
            mma_s8_16832(d, a, bb.x, bb.y);
            const unsigned sw0 = u4w(cur.s0, s >> 2), sw1 = u4w(cur.s1, s >> 2);
            const unsigned mw0 = u4w(cur.s0, 2 + (s >> 2)), mw1 = u4w(cur.s1, 2 + (s >> 2));
            const unsigned sel = (unsigned)(s & 3) * 0x1111u;
            const unsigned am[4] = {__byte_perm(mw0, 0, sel), __byte_perm(mw1, 0, sel), __byte_perm(mw0, 0, sel),
                                    __byte_perm(mw1, 0, sel)};
            const int s0 = (int)__byte_perm(sw0, 0, 0x4440u | (s & 3));
            const int s1 = (int)__byte_perm(sw1, 0, 0x4440u | (s & 3));
            if (P32) {
                int mm[4];
                mma_s8_16832(mm, am, bb.x, bb.y);
                const float d0 = h2f((unsigned short)(cur.dw0 & 0xFFFFu)), m0 = h2f((unsigned short)(cur.dw0 >> 16));
                const float d1 = h2f((unsigned short)(cur.dw1 & 0xFFFFu)), m1 = h2f((unsigned short)(cur.dw1 >> 16));
                const float fs0 = d0 * i2f(s0), fs1 = d1 * i2f(s1);
                const float fm0 = m0, fm1 = m1;  // the mins MMA already carries m_j
                const int blk = sb * 8 + s;
                if (na < m) {
                    const float dx = __ldg(da + blk);
                    acc[0] = fmaf(dx, fmaf(fs0, i2f(d[0]), -fm0 * i2f(mm[0])), acc[0]);
                    acc[2] = fmaf(dx, fmaf(fs1, i2f(d[2]), -fm1 * i2f(mm[2])), acc[2]);
                }
                if (nb < m) {
                    const float dx = __ldg(db + blk);
                    acc[1] = fmaf(dx, fmaf(fs0, i2f(d[1]), -fm0 * i2f(mm[1])), acc[1]);
                    acc[3] = fmaf(dx, fmaf(fs1, i2f(d[3]), -fm1 * i2f(mm[3])), acc[3]);
                }
                continue;
            }
            mma_s8_16832c(i2, am, bb.x, bb.y);
            i1[0] += s0 * d[0];
            i1[1] += s0 * d[1];
            i1[2] += s1 * d[2];
            i1[3] += s1 * d[3];
        }
        const float d0 = h2f((unsigned short)(cur.dw0 & 0xFFFFu)), m0 = h2f((unsigned short)(cur.dw0 >> 16));
        const float d1 = h2f((unsigned short)(cur.dw1 & 0xFFFFu)), m1 = h2f((unsigned short)(cur.dw1 >> 16));
        if (P32) { if (sb + 1 < sb1) cur = nxt; continue; }
        if (na < m) {
            const float dx = __ldg(da + sb);
            acc[0] = fmaf(dx, fmaf(d0, i2f(i1[0]), -m0 * i2f(i2[0])), acc[0]);
            acc[2] = fmaf(dx, fmaf(d1, i2f(i1[2]), -m1 * i2f(i2[2])), acc[2]);
        }
        if (nb < m) {
            const float dx = __ldg(db + sb);
            acc[1] = fmaf(dx, fmaf(d0, i2f(i1[1]), -m0 * i2f(i2[1])), acc[1]);
            acc[3] = fmaf(dx, fmaf(d1, i2f(i1[3]), -m1 * i2f(i2[3])), acc[3]);
        }
        if (sb + 1 < sb1) cur = nxt;
    }
}
template <bool DOWN, bool P32>
__device__ __forceinline__ void moe_tcx_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                             const unsigned char* xk, const unsigned char* xq, float* out, int T,
                                             int fast, const unsigned* plan) {
    extern __shared__ uint4 dsm[];
    constexpr int KA = DOWN ? CX_FF : CX_HID;
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = plan[1 + gi], b = plan[2 + gi], m = b - a;
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const unsigned e = shared ? 0u : ids[p0];
    // S warps a row tile, each a 1/S of the superblocks; summed in slice order.
    constexpr int S = DOWN ? CX_TCX_SDN : CX_TCX_SGU;
    constexpr int NSB = KA / 256;
    const int ks = warp % S;
    const int r0 = (blockIdx.x * (8 / S) + warp / S) * 16;
    const ExW& mw = DOWN ? (shared ? sg : g) : (shared ? (r0 < CX_FF ? sg : su) : (r0 < CX_FF ? g : u));
    const int ro = DOWN ? r0 : (r0 < CX_FF ? r0 : r0 - CX_FF);
    const size_t ostride = DOWN ? CX_HID : 2 * CX_FF;
    __shared__ int pair_s[8];
    __shared__ float4 red_s[8][32];
    if (threadIdx.x < 8) pair_s[threadIdx.x] = threadIdx.x < m ? (int)plan[CX_PLAN_LIST + a + threadIdx.x] : 0;
    __syncthreads();
    auto tok = [&](int p) { return DOWN ? p : (shared ? p - T * CX_TOPK : p / CX_TOPK); };
    const unsigned char* wbase = (const unsigned char*)mw.base + (size_t)e * mw.estride;
    if (mw.ty != 112 && mw.ty != 113) {
        // Not repacked (whole block shares the type): the per-token dot on the per-32 codes.
        const float* xdg = (const float*)(xq + (size_t)(DOWN ? CX_NP : XQ8_T) * KA);
        for (int r = ks; r < 16; r += S) {
            const unsigned char* w = wbase + (size_t)(ro + r) * mw.row_bytes;
            for (int q = 0; q < m; q++) {
                const int t = tok(pair_s[q]);
                const float v = cx_dot(mw.ty, w, xq + (size_t)t * KA, xdg + (size_t)t * (KA / 32), KA, fast);
                if (lane == 0) out[(size_t)pair_s[q] * ostride + r0 + r] = v;
            }
        }
        return;
    }
    // The first superblock's weights are in flight before the activations are staged.
    const unsigned char* tile = wbase + (size_t)(ro / 16) * mw.row_bytes;
    const int sb0 = ks * NSB / S, sb1 = (ks + 1) * NSB / S;
    TcxT first;
    if (mw.ty == 112) tcx_load<false>(tile + sb0 * 2368, lane, first);
    else tcx_load<true>(tile + sb0 * 2880, lane, first);
    // Activations straight from global (cx_q8k wrote them in B-fragment order; L1 / L2 hits).
    const unsigned char* xa = P32 ? xq : xk;
    constexpr int SB = P32 ? 32 : 256;
    const float* xdg = (const float*)(xa + (size_t)(DOWN ? CX_NP : XQ8_T) * KA);
    const int gq = lane >> 2, tig = lane & 3;
    const int tg_ = tok(pair_s[gq < m ? gq : 0]);
    const unsigned char* xg = xa + (size_t)tg_ * KA + tig * (P32 ? 4 : 8);
    const float* da = xdg + (size_t)tok(pair_s[2 * tig < m ? 2 * tig : 0]) * (KA / SB);
    const float* db = xdg + (size_t)tok(pair_s[2 * tig + 1 < m ? 2 * tig + 1 : 0]) * (KA / SB);
    float acc[4];
    if (mw.ty == 112) tcx_rows<false, KA, P32>(tile, m, xg, da, db, acc, sb0, sb1, first);
    else tcx_rows<true, KA, P32>(tile, m, xg, da, db, acc, sb0, sb1, first);
    if (S > 1) {
        red_s[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __syncthreads();
        if (ks != 0) return;
        for (int j = 1; j < S; j++) {
            const float4 v = red_s[warp + j][lane];
            acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
        }
    }
    if (2 * tig < m) {
        out[(size_t)pair_s[2 * tig] * ostride + r0 + gq] = acc[0];
        out[(size_t)pair_s[2 * tig] * ostride + r0 + gq + 8] = acc[2];
    }
    if (2 * tig + 1 < m) {
        out[(size_t)pair_s[2 * tig + 1] * ostride + r0 + gq] = acc[1];
        out[(size_t)pair_s[2 * tig + 1] * ostride + r0 + gq + 8] = acc[3];
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_t(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                             const unsigned char* xk, const unsigned char* xq,
                                                             float* out, int T, int fast, const unsigned* plan) {
    moe_tcx_body<false, false>(g, u, sg, su, ids, xk, xq, out, T, fast, plan);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_t32(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                               const unsigned char* xk, const unsigned char* xq,
                                                               float* out, int T, int fast, const unsigned* plan) {
    moe_tcx_body<false, true>(g, u, sg, su, ids, xk, xq, out, T, fast, plan);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_t(ExW d, ExW sd, const unsigned* ids,
                                                               const unsigned char* hk, const unsigned char* hq,
                                                               float* out, int T, int fast, const unsigned* plan) {
    moe_tcx_body<true, false>(d, d, sd, sd, ids, hk, hq, out, T, fast, plan);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_t32(ExW d, ExW sd, const unsigned* ids,
                                                                 const unsigned char* hk, const unsigned char* hq,
                                                                 float* out, int T, int fast, const unsigned* plan) {
    moe_tcx_body<true, true>(d, d, sd, sd, ids, hk, hq, out, T, fast, plan);
}
// Per-256 int8 activations (Q8_K-style, for the TCX experts): codes [rows][K] then f32 scales
// [rows][K / 256], d = amax / 127. Grid (K / 256, rows), block 256.
__device__ __forceinline__ void q8k_store(float v, unsigned char* xk, int nrows_max, int K, int row, int c) {
    __shared__ float sh[8];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = a;
    __syncthreads();
    float amax = sh[0];
    for (int i = 1; i < 8; i++) amax = fmaxf(amax, sh[i]);
    const float d = amax / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    // Permuted a 32 as the MMA's B fragment wants it: [tig][cols 4 tig.., 16 + 4 tig..].
    const int o = threadIdx.x & 31, pos = (threadIdx.x & ~31) + ((o & 15) >> 2) * 8 + (o >> 4) * 4 + (o & 3);
    xk[(size_t)row * K + c * 256 + pos] = (unsigned char)(signed char)q;
    if (threadIdx.x == 0) ((float*)(xk + (size_t)nrows_max * K))[row * (K / 256) + c] = d;
}
extern "C" __global__ void __launch_bounds__(256) cx_q8k(const float* x, unsigned char* xk) {
    const int c = blockIdx.x, t = blockIdx.y;
    q8k_store(x[(size_t)t * CX_HID + c * 256 + threadIdx.x], xk, XQ8_T, CX_HID, t, c);
}
extern "C" __global__ void __launch_bounds__(256) cx_silu_q8k(const float* gu, unsigned char* hk, int pbase) {
    const int c = blockIdx.x, p = pbase + blockIdx.y;
    const float gv = gu[(size_t)p * 2 * CX_FF + c * 256 + threadIdx.x];
    const float uv = gu[(size_t)p * 2 * CX_FF + CX_FF + c * 256 + threadIdx.x];
    q8k_store(silu_f2(gv) * uv, hk, CX_NP, CX_FF, p, c);
}

// ---- GGUF-native batch experts: TCX32's kernel on the GGUF Q4_K / Q5_K rows themselves ----
//
// The m16n8k32 A register c of lane (g, tig) at step s is row g + 8 (c & 1)'s qs word at
// 32 (s / 2) + 16 (c >> 1) + 4 tig (sbk_load's q words, low nibbles for even s, high for odd),
// so the rows feed the MMA as they are. Per superblock a lane unpacks its two rows' 6-bit scales
// and mins into byte lanes once (bit ops on the three scale words); per step: the codes MMA, the
// mins MMA (A' = m_j replicated: gives m_j sum(x)), per-32 activations read from global, float
// epilogue as cx_moe_*_t32. Grid (rows · S / 128, groups), block 256, no smem, no barrier past
// the plan.
__device__ __forceinline__ void sc_unpack(const unsigned* sc, unsigned& s03, unsigned& s47, unsigned& m03, unsigned& m47) {
    s03 = sc[0] & 0x3F3F3F3Fu;
    m03 = sc[1] & 0x3F3F3F3Fu;
    s47 = (sc[2] & 0x0F0F0F0Fu) | (((sc[0] >> 6) & 0x03030303u) << 4);
    m47 = ((sc[2] >> 4) & 0x0F0F0F0Fu) | (((sc[1] >> 6) & 0x03030303u) << 4);
}
// A row's superblock for gg_rows: lane tig takes qs bytes 32 j + 8 tig .. + 8 (one uint2 a j) and
// qh bytes 8 tig .. + 8. The MMA's k order inside a 32-column step is then a permutation of the
// columns (A register 0 = columns 8 tig .. + 4, register 2 = 8 tig + 4 ..), and the B fragment
// (x at 8 tig, one uint2) takes the same one: dot products and per-32 scales are unchanged.
struct GgR { float d, dm; unsigned sc[3]; uint2 q[4]; uint2 h; };
template <int TY>
__device__ __forceinline__ void gg_load(const unsigned char* b, int tig, GgR& o) {
    constexpr int QS = TY == 12 ? 16 : 48;
    const unsigned dw = __ldg((const unsigned*)b);
    o.d = h2f((unsigned short)(dw & 0xFFFFu));
    o.dm = h2f((unsigned short)(dw >> 16));
    o.sc[0] = __ldg((const unsigned*)(b + 4));
    o.sc[1] = __ldg((const unsigned*)(b + 8));
    o.sc[2] = __ldg((const unsigned*)(b + 12));
    #pragma unroll
    for (int j = 0; j < 4; j++) o.q[j] = __ldg((const uint2*)(b + QS + j * 32 + tig * 8));
    if (TY == 13) o.h = __ldg((const uint2*)(b + 16 + tig * 8));
    else o.h = make_uint2(0, 0);
}
template <int TY, int K>
__device__ __forceinline__ void gg_rows(const unsigned char* wa, const unsigned char* wb, int m, const unsigned char* xg,
                                        const float* da, const float* db, float acc[4], int sb0, int sb1) {
    constexpr int BB = TY == 12 ? 144 : 176;
    const int lane = threadIdx.x & 31, tig = lane & 3;
    const int na = 2 * tig, nb = 2 * tig + 1;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    GgR ra, rb, pa, pb;
    gg_load<TY>(wa + sb0 * BB, tig, ra);
    gg_load<TY>(wb + sb0 * BB, tig, rb);
    for (int sb = sb0; sb < sb1; sb++) {
        if (sb + 1 < sb1) { gg_load<TY>(wa + (sb + 1) * BB, tig, pa); gg_load<TY>(wb + (sb + 1) * BB, tig, pb); }
        unsigned sa[2], ma[2], sbw[2], mbw[2];
        sc_unpack(ra.sc, sa[0], sa[1], ma[0], ma[1]);
        sc_unpack(rb.sc, sbw[0], sbw[1], mbw[0], mbw[1]);
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            const int qp = s >> 1;
            unsigned a[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const GgR& r = (c & 1) ? rb : ra;
                const unsigned w = (c >> 1) ? r.q[qp].y : r.q[qp].x;
                unsigned v = (s & 1) ? (w >> 4) & 0x0F0F0F0Fu : w & 0x0F0F0F0Fu;
                if (TY == 13) v |= ((((c >> 1) ? r.h.y : r.h.x) >> s) & 0x01010101u) << 4;
                a[c] = v;
            }
            const uint2 bb = __ldg((const uint2*)(xg + sb * 256 + s * 32));
            int d[4], mm[4];
            mma_s8_16832(d, a, bb.x, bb.y);
            const unsigned sel = (unsigned)(s & 3) * 0x1111u;
            const unsigned mw0 = ma[s >> 2], mw1 = mbw[s >> 2];
            const unsigned am[4] = {__byte_perm(mw0, 0, sel), __byte_perm(mw1, 0, sel), __byte_perm(mw0, 0, sel),
                                    __byte_perm(mw1, 0, sel)};
            mma_s8_16832(mm, am, bb.x, bb.y);
            const int s0 = (int)__byte_perm(sa[s >> 2], 0, 0x4440u | (s & 3));
            const int s1 = (int)__byte_perm(sbw[s >> 2], 0, 0x4440u | (s & 3));
            const float fs0 = ra.d * i2f(s0), fs1 = rb.d * i2f(s1);
            const int blk = sb * 8 + s;
            if (na < m) {
                const float dx = __ldg(da + blk);
                acc[0] = fmaf(dx, fmaf(fs0, i2f(d[0]), -ra.dm * i2f(mm[0])), acc[0]);
                acc[2] = fmaf(dx, fmaf(fs1, i2f(d[2]), -rb.dm * i2f(mm[2])), acc[2]);
            }
            if (nb < m) {
                const float dx = __ldg(db + blk);
                acc[1] = fmaf(dx, fmaf(fs0, i2f(d[1]), -ra.dm * i2f(mm[1])), acc[1]);
                acc[3] = fmaf(dx, fmaf(fs1, i2f(d[3]), -rb.dm * i2f(mm[3])), acc[3]);
            }
        }
        if (sb + 1 < sb1) { ra = pa; rb = pb; }
    }
}
template <bool DOWN>
__device__ __forceinline__ void moe_gg_body(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids, const unsigned char* xq,
                                            float* out, int T, int fast, const unsigned* plan) {
    constexpr int KA = DOWN ? CX_FF : CX_HID;
    constexpr int S = DOWN ? CX_TCX_SDN : CX_TCX_SGU;
    constexpr int NSB = KA / 256;
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = plan[1 + gi], b = plan[2 + gi], m = b - a;
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const unsigned e = shared ? 0u : ids[p0];
    const int ks = warp % S;
    const int r0 = (blockIdx.x * (8 / S) + warp / S) * 16;
    const ExW& mw = DOWN ? (shared ? sg : g) : (shared ? (r0 < CX_FF ? sg : su) : (r0 < CX_FF ? g : u));
    const int ro = DOWN ? r0 : (r0 < CX_FF ? r0 : r0 - CX_FF);
    const size_t ostride = DOWN ? CX_HID : 2 * CX_FF;
    __shared__ int pair_s[8];
    __shared__ float4 red_s[8][32];
    if (threadIdx.x < 8) pair_s[threadIdx.x] = threadIdx.x < m ? (int)plan[CX_PLAN_LIST + a + threadIdx.x] : 0;
    __syncthreads();
    auto tok = [&](int p) { return DOWN ? p : (shared ? p - T * CX_TOPK : p / CX_TOPK); };
    const unsigned char* wbase = (const unsigned char*)mw.base + (size_t)e * mw.estride;
    const float* xdg = (const float*)(xq + (size_t)(DOWN ? CX_NP : XQ8_T) * KA);
    if (mw.ty != 12 && mw.ty != 13) {
        for (int r = ks; r < 16; r += S) {
            const unsigned char* w = wbase + (size_t)(ro + r) * mw.row_bytes;
            for (int q = 0; q < m; q++) {
                const int t = tok(pair_s[q]);
                const float v = cx_dot(mw.ty, w, xq + (size_t)t * KA, xdg + (size_t)t * (KA / 32), KA, fast);
                if (lane == 0) out[(size_t)pair_s[q] * ostride + r0 + r] = v;
            }
        }
        return;
    }
    const int gq = lane >> 2, tig = lane & 3;
    const unsigned char* xg = xq + (size_t)tok(pair_s[gq < m ? gq : 0]) * KA + tig * 8;
    const float* da = xdg + (size_t)tok(pair_s[2 * tig < m ? 2 * tig : 0]) * (KA / 32);
    const float* db = xdg + (size_t)tok(pair_s[2 * tig + 1 < m ? 2 * tig + 1 : 0]) * (KA / 32);
    const unsigned char* wa = wbase + (size_t)(ro + gq) * mw.row_bytes;
    const unsigned char* wb = wa + (size_t)8 * mw.row_bytes;
    const int sb0 = ks * NSB / S, sb1 = (ks + 1) * NSB / S;
    float acc[4];
    if (mw.ty == 12) gg_rows<12, KA>(wa, wb, m, xg, da, db, acc, sb0, sb1);
    else gg_rows<13, KA>(wa, wb, m, xg, da, db, acc, sb0, sb1);
    if (S > 1) {
        red_s[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __syncthreads();
        if (ks != 0) return;
        for (int j = 1; j < S; j++) {
            const float4 v = red_s[warp + j][lane];
            acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
        }
    }
    if (2 * tig < m) {
        out[(size_t)pair_s[2 * tig] * ostride + r0 + gq] = acc[0];
        out[(size_t)pair_s[2 * tig] * ostride + r0 + gq + 8] = acc[2];
    }
    if (2 * tig + 1 < m) {
        out[(size_t)pair_s[2 * tig + 1] * ostride + r0 + gq] = acc[1];
        out[(size_t)pair_s[2 * tig + 1] * ostride + r0 + gq + 8] = acc[3];
    }
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu_q(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                             const unsigned char* xq, float* out, int T, int fast,
                                                             const unsigned* plan) {
    moe_gg_body<false>(g, u, sg, su, ids, xq, out, T, fast, plan);
}
extern "C" __global__ void __launch_bounds__(256) cx_moe_down_q(ExW d, ExW sd, const unsigned* ids,
                                                               const unsigned char* hq, float* out, int T, int fast,
                                                               const unsigned* plan) {
    moe_gg_body<true>(d, d, sd, sd, ids, hq, out, T, fast, plan);
}
// ---- dense Q8X (Q8_0 repacked) on int8 tensor cores for batch windows ----
//
// A warp owns 16 rows of one segment (M) and all T <= 64 tokens (up to 8 n-tiles of 8); K steps
// 32 (one Q8_0 block) a mma.sync m16n8k32 s8 per n-tile, the rows' codes read straight from
// global as the A fragment (prefetched a 256-wide chunk ahead), the tokens' codes and scales
// staged in shared memory per chunk; acc += dx * dw * dot. Grid (rows / 64, splits), block 128.
// splits > 1: each split covers k / splits columns and writes partials [split][t][rows];
// cx_mma_q8x_red sums them in split order (deterministic). Not the GEMV's summation order.
#define MQ_KC 256
extern "C" __global__ void __launch_bounds__(128) cx_mma_q8x(const Seg* segs, int nseg, const unsigned char* xq, int k,
                                                             float* out, int ostride, int T, int splits, float* part,
                                                             int rows_total, const float* xf) {
    __shared__ uint4 xs[64 * MQ_KC / 16];
    __shared__ float xds[64 * (MQ_KC / 32)];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int NT = (T + 7) / 8;
    int row = (blockIdx.x * 4 + warp) * 16, s = 0;
    const int growrow = row;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    const bool ok = s < nseg;
    const Seg sg = ok ? segs[s] : segs[0];
    const unsigned char* wa = (const unsigned char*)sg.w + (size_t)(row + g) * sg.row_bytes;
    const unsigned char* wb = wa + (size_t)8 * sg.row_bytes;
    const float* xdb = (const float*)(xq + (size_t)XQ8_T * k);
    const int ks = k / splits, kb = blockIdx.y * ks;
    if (ok && sg.type != 108) {
        // F32 / BF16 segments (small ones: GDN beta / alpha): a plain dot a row and token at a
        // time over this split's columns, from the f32 activations.
        for (int r = 0; r < 16; r++) {
            const unsigned char* w = (const unsigned char*)sg.w + (size_t)(row + r) * sg.row_bytes;
            for (int t = 0; t < T; t++) {
                const float* xr = xf + (size_t)t * k;
                float a = 0.f;
                for (int e = kb + lane; e < kb + ks; e += 32) {
                    const float wv = sg.type == 0 ? ((const float*)w)[e] : bf2f(((const unsigned short*)w)[e]);
                    a = fmaf(wv, xr[e], a);
                }
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) {
                    if (splits == 1) out[(size_t)t * ostride + sg.out_off + row + r] = a;
                    else part[((size_t)blockIdx.y * T + t) * rows_total + growrow + r] = a;
                }
            }
        }
    }
    if (ok && (sg.type == 12 || sg.type == 13)) {
        // Q4_K / Q5_K segments (GDN's ssm_ba): the expert MMA tile, 8 tokens at a time, whole k;
        // token group i in split i % splits (the other splits add zeros).
        const unsigned char* wa = (const unsigned char*)sg.w + (size_t)(row + g) * sg.row_bytes;
        const unsigned char* wb = wa + (size_t)8 * sg.row_bytes;
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int m = min(8, T - t0);
            float a4[4] = {0.f, 0.f, 0.f, 0.f};
            if ((t0 / 8) % splits == (int)blockIdx.y && k == CX_HID) {
                if (sg.type == 12) moe_mma_rows<12, false>(wa, wb, 0, m, xq + (size_t)t0 * k, xdb + (size_t)t0 * (k / 32), nullptr, a4);
                else moe_mma_rows<13, false>(wa, wb, 0, m, xq + (size_t)t0 * k, xdb + (size_t)t0 * (k / 32), nullptr, a4);
            }
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int t = t0 + 2 * tig + j;
                if (t >= T) continue;
                if (splits == 1) {
                    out[(size_t)t * ostride + sg.out_off + row + g] = a4[j];
                    out[(size_t)t * ostride + sg.out_off + row + g + 8] = a4[2 + j];
                } else {
                    float* pp = part + ((size_t)blockIdx.y * T + t) * rows_total + growrow + g;
                    pp[0] = a4[j];
                    pp[8] = a4[2 + j];
                }
            }
        }
    }
    const bool mm = ok && sg.type == 108;
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    for (int k0 = kb; k0 < kb + ks; k0 += MQ_KC) {
        unsigned af[8][4];
        float dwa[8], dwb[8];
        if (mm) {
            #pragma unroll
            for (int st = 0; st < 8; st++) {
                const int e = k0 + st * 32;
                af[st][0] = *(const unsigned*)(wa + e + tig * 4);
                af[st][1] = *(const unsigned*)(wb + e + tig * 4);
                af[st][2] = *(const unsigned*)(wa + e + 16 + tig * 4);
                af[st][3] = *(const unsigned*)(wb + e + 16 + tig * 4);
                dwa[st] = h2f(rd16(wa + k + (e / 32) * 2));
                dwb[st] = h2f(rd16(wb + k + (e / 32) * 2));
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < T * (MQ_KC / 16); i += blockDim.x) {
            const int t = i / (MQ_KC / 16), c = i % (MQ_KC / 16);
            xs[i] = *(const uint4*)(xq + (size_t)t * k + k0 + c * 16);
        }
        for (int i = threadIdx.x; i < T * (MQ_KC / 32); i += blockDim.x) {
            const int t = i / (MQ_KC / 32), c = i % (MQ_KC / 32);
            xds[i] = xdb[(size_t)t * (k / 32) + k0 / 32 + c];
        }
        __syncthreads();
        if (!mm) continue;
        const unsigned char* xsb = (const unsigned char*)xs;
        #pragma unroll
        for (int st = 0; st < 8; st++) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                if (nt >= NT) break;
                const int tg = nt * 8 + g;
                const unsigned char* xr = xsb + (size_t)(tg < T ? tg : 0) * MQ_KC + st * 32;
                const unsigned b0 = *(const unsigned*)(xr + tig * 4), b1 = *(const unsigned*)(xr + 16 + tig * 4);
                int d[4];
                mma_s8_16832(d, af[st], b0, b1);
                const int t0 = nt * 8 + 2 * tig, t1 = t0 + 1;
                const float x0 = t0 < T ? xds[t0 * (MQ_KC / 32) + st] : 0.f;
                const float x1 = t1 < T ? xds[t1 * (MQ_KC / 32) + st] : 0.f;
                acc[nt][0] = fmaf(x0 * dwa[st], i2f(d[0]), acc[nt][0]);
                acc[nt][1] = fmaf(x1 * dwa[st], i2f(d[1]), acc[nt][1]);
                acc[nt][2] = fmaf(x0 * dwb[st], i2f(d[2]), acc[nt][2]);
                acc[nt][3] = fmaf(x1 * dwb[st], i2f(d[3]), acc[nt][3]);
            }
        }
    }
    if (!mm) return;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        if (nt >= NT) break;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int t = nt * 8 + 2 * tig + j;
            if (t >= T) continue;
            if (splits == 1) {
                out[(size_t)t * ostride + sg.out_off + row + g] = acc[nt][j];
                out[(size_t)t * ostride + sg.out_off + row + g + 8] = acc[nt][2 + j];
            } else {
                float* pp = part + ((size_t)blockIdx.y * T + t) * rows_total + growrow + g;
                pp[0] = acc[nt][j];
                pp[8] = acc[nt][2 + j];
            }
        }
    }
}
// The router (F32 / BF16, one segment) for a batch window: a block owns 16 rows and a 128-wide
// slice of k for all T <= 64 tokens, the tokens' slice in shared memory (padded rows), each lane
// two tokens (t, t + 32) of each of its warp's 2 rows. Partials [split][t][rows] for
// cx_mma_q8x_red. Grid (rows / 16 rounded up, k / 128), block 256.
extern "C" __global__ void __launch_bounds__(256) cx_router_b(const Seg* segs, int nseg, const float* x, int k, int T,
                                                              float* part, int rows_total) {
    __shared__ float xs[64][129];
    const int k0 = blockIdx.y * 128;
    for (int i = threadIdx.x; i < T * 128; i += blockDim.x) {
        const int t = i >> 7, c = i & 127;
        xs[t][c] = x[(size_t)t * k + k0 + c];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const int r = blockIdx.x * 16 + warp * 2 + j;
        if (r >= rows_total) continue;
        int rr = r, si = 0;
        while (si < nseg - 1 && rr >= segs[si].rows) { rr -= segs[si].rows; si++; }
        const Seg sg = segs[si];
        const unsigned char* w = (const unsigned char*)sg.w + (size_t)rr * sg.row_bytes;
        float a0 = 0.f, a1 = 0.f;
        const bool t1 = lane + 32 < T, t0 = lane < T;
        for (int c = 0; c < 128; c++) {
            const float wv = sg.type == 0 ? ((const float*)w)[k0 + c] : bf2f(((const unsigned short*)w)[k0 + c]);
            if (t0) a0 = fmaf(wv, xs[lane][c], a0);
            if (t1) a1 = fmaf(wv, xs[lane + 32][c], a1);
        }
        if (t0) part[((size_t)blockIdx.y * T + lane) * rows_total + r] = a0;
        if (t1) part[((size_t)blockIdx.y * T + lane + 32) * rows_total + r] = a1;
    }
}
// The router (F32 / BF16 segments) for a wide window: a plain f32 tiled GEMM, 64 rows × 64
// tokens a block, k in steps of 32 through shared memory, 4 × 4 outputs a thread. x is the f32
// activations [T][k]. out[t][row]. Grid (rows / 64 rounded up, T / 64 rounded up), block 256.
// cx_sgemm_wide, re-tiled: 64 rows x 128 tokens a block, 4 x 8 outputs a thread, k slabs of 16
// loaded as float4 (the next slab in registers while this one computes), each row's segment
// found once a block. Every output is the same fmaf chain over k ascending as cx_sgemm_wide
// (bitwise equal). Grid (rows / 64, T / 128), block 256.
template <int NT>
__device__ __forceinline__ void sgemm_wide2_body(const Seg* segs, int nseg, int rows_total,
                                                                 const float* x, int k, float* out, int ostride,
                                                                 int T) {
    // NT threads: 64 rows x NT / 2 tokens a block.
    constexpr int TB = NT / 2, RL = NT / 4;  // tokens a block; rows (tokens) a loader pass
    __shared__ const unsigned char* wrow[64];
    __shared__ int wty[64], wout[64];
    __shared__ __align__(16) float ws[16][64 + 4];
    __shared__ __align__(16) float xs[16][TB + 4];
    const int r0 = blockIdx.x * 64, t0 = blockIdx.y * TB, tid = threadIdx.x;
    for (int rI = tid; rI < 64; rI += NT) {
        const int rr = r0 + rI;
        const unsigned char* w = nullptr;
        int ty = 0, o = 0;
        if (rr < rows_total) {
            int q = rr, si = 0;
            while (si < nseg - 1 && q >= segs[si].rows) { q -= segs[si].rows; si++; }
            const Seg sg = segs[si];
            w = (const unsigned char*)sg.w + (size_t)q * sg.row_bytes;
            ty = sg.type;
            o = sg.out_off + q;
        }
        wrow[rI] = w; wty[rI] = ty; wout[rI] = o;
    }
    __syncthreads();
    const int lr = tid >> 2, lc = (tid & 3) * 4;  // loader: rows / tokens lr + RL m, k lc..lc+3
    constexpr int NW = 64 / RL;                    // weight float4s a thread (1 or 2)
    const unsigned char* wl[NW];
    bool wbf[NW];
    #pragma unroll
    for (int m = 0; m < NW; m++) { wl[m] = wrow[lr + RL * m]; wbf[m] = wty[lr + RL * m] != 0; }
    const float* xl[2];
    #pragma unroll
    for (int m = 0; m < 2; m++) { const int tt = t0 + lr + RL * m; xl[m] = tt < T ? x + (size_t)tt * k : nullptr; }
    auto ldw = [&](int m, int k0) -> float4 {
        if (!wl[m]) return make_float4(0.f, 0.f, 0.f, 0.f);
        if (!wbf[m]) return *(const float4*)((const float*)wl[m] + k0 + lc);
        const uint2 u = *(const uint2*)((const unsigned short*)wl[m] + k0 + lc);
        return make_float4(bf2f((unsigned short)(u.x & 0xffff)), bf2f((unsigned short)(u.x >> 16)),
                           bf2f((unsigned short)(u.y & 0xffff)), bf2f((unsigned short)(u.y >> 16)));
    };
    float4 pw[NW], px[2];
    #pragma unroll
    for (int m = 0; m < NW; m++) pw[m] = ldw(m, 0);
    #pragma unroll
    for (int m = 0; m < 2; m++) px[m] = xl[m] ? *(const float4*)(xl[m] + lc) : make_float4(0.f, 0.f, 0.f, 0.f);
    constexpr int NTX = TB / 8;  // token lanes: tokens 4 tx.. and TB / 2 + 4 tx..
    const int tx = tid % NTX, ty = tid / NTX;
    float acc[4][8] = {};
    for (int k0 = 0; k0 < k; k0 += 16) {
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < NW; m++) {
            ws[lc][lr + RL * m] = pw[m].x; ws[lc + 1][lr + RL * m] = pw[m].y;
            ws[lc + 2][lr + RL * m] = pw[m].z; ws[lc + 3][lr + RL * m] = pw[m].w;
        }
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            xs[lc][lr + RL * m] = px[m].x; xs[lc + 1][lr + RL * m] = px[m].y;
            xs[lc + 2][lr + RL * m] = px[m].z; xs[lc + 3][lr + RL * m] = px[m].w;
        }
        __syncthreads();
        if (k0 + 16 < k) {
            #pragma unroll
            for (int m = 0; m < NW; m++) pw[m] = ldw(m, k0 + 16);
            #pragma unroll
            for (int m = 0; m < 2; m++) if (xl[m]) px[m] = *(const float4*)(xl[m] + k0 + 16 + lc);
        }
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            // Rows 4 ty.., tokens 4 tx.. and 64 + 4 tx..: one 16-byte read of w, two of x a column
            // (conflict-free).
            const float4 av = *(const float4*)&ws[c][4 * ty];
            const float4 b0 = *(const float4*)&xs[c][4 * tx], b1 = *(const float4*)&xs[c][TB / 2 + 4 * tx];
            const float a[4] = {av.x, av.y, av.z, av.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = 4 * ty + i;
        if (!wrow[r]) continue;
        const int o = wout[r];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int tt = t0 + (j < 4 ? 4 * tx + j : TB / 2 + 4 * tx + j - 4);
            if (tt < T) out[(size_t)tt * ostride + o] = acc[i][j];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) cx_sgemm_wide2(const Seg* segs, int nseg, int rows_total,
                                                                 const float* x, int k, float* out, int ostride, int T) {
    sgemm_wide2_body<256>(segs, nseg, rows_total, x, k, out, ostride, T);
}
// 64 rows x 64 tokens a block (twice the blocks: a fuller last wave). Block 128.
extern "C" __global__ void __launch_bounds__(128) cx_sgemm_wide2s(const Seg* segs, int nseg, int rows_total,
                                                                  const float* x, int k, float* out, int ostride, int T) {
    sgemm_wide2_body<128>(segs, nseg, rows_total, x, k, out, ostride, T);
}

extern "C" __global__ void __launch_bounds__(256) cx_sgemm_wide(const Seg* segs, int nseg, int rows_total,
                                                                const float* x, int k, float* out, int ostride,
                                                                int T) {
    __shared__ float ws[32][65], xs[32][65];
    const int r0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < k; k0 += 32) {
        for (int i = threadIdx.x; i < 64 * 32; i += 256) {
            const int r = i >> 5, c = i & 31;
            const int rr = r0 + r, tt = t0 + r;
            float wv = 0.f;
            if (rr < rows_total) {
                int q = rr, si = 0;
                while (si < nseg - 1 && q >= segs[si].rows) { q -= segs[si].rows; si++; }
                const Seg sg = segs[si];
                const unsigned char* w = (const unsigned char*)sg.w + (size_t)q * sg.row_bytes;
                wv = sg.type == 0 ? ((const float*)w)[k0 + c] : bf2f(((const unsigned short*)w)[k0 + c]);
            }
            ws[c][r] = wv;
            xs[c][r] = tt < T ? x[(size_t)tt * k + k0 + c] : 0.f;
        }
        __syncthreads();
        #pragma unroll 8
        for (int c = 0; c < 32; c++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = ws[c][ty + 16 * i]; b[i] = xs[c][tx + 16 * i]; }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) acc[i][jj] = fmaf(a[i], b[jj], acc[i][jj]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int rr = r0 + ty + 16 * i;
        if (rr >= rows_total) continue;
        int q = rr, si = 0;
        while (si < nseg - 1 && q >= segs[si].rows) { q -= segs[si].rows; si++; }
        const int o = segs[si].out_off + q;
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            const int tt = t0 + tx + 16 * jj;
            if (tt < T) out[(size_t)tt * ostride + o] = acc[i][jj];
        }
    }
}
// Sum cx_mma_q8x's split partials in split order into out. Grid (rows_total / 128, T), block 128.
extern "C" __global__ void cx_mma_q8x_red(const Seg* segs, int nseg, const float* part, int splits, int rows_total,
                                          int T, float* out, int ostride) {
    const int gr = blockIdx.x * 128 + threadIdx.x, t = blockIdx.y;
    if (gr >= rows_total) return;
    int row = gr, s = 0;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    float v = 0.f;
    for (int z = 0; z < splits; z++) v += part[((size_t)z * T + t) * rows_total + gr];
    out[(size_t)t * ostride + segs[s].out_off + row] = v;
}

// ---- grouped expert GEMM on tensor cores (wide prefill) ----
//
// The grouped kernels above decode an expert's weights once per (token, expert) pair; at wide
// windows that decode is the cost. Here each warp decodes a 16-row × 16-column weight tile once
// (decode8i, plain order) and multiplies it by up to 64 of the group's tokens with int8 MMA
// (mma.sync m16n8k16, s8 × s8 → s32), then applies the row's sub-block scale (and min) and the
// token's activation scale per 16 columns, accumulating in f32. Not the GEMV's summation order:
// gated against narrow prefill by greedy tokens and KL, not bitwise. Same plan as cx_moe_gu_g.
// Grid (rows / 128, groups, 64-token tiles), block 256 (8 warps × 16 rows).
__device__ __forceinline__ void mma_s8_16816(int* d, int a0, int a1, unsigned b0) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0));
#else
    d[0] = d[1] = d[2] = d[3] = 0;
#endif
}

// One warp: rows wa / wb (16 apart... rows g and g + 8 of its tile) of a K-wide matrix against
// the activation rows `xrow(c, n)` (int8 codes, scales xd per 32 columns) of up to nch chunks of
// 8 columns; acc[c][0..1] = rows g × columns 2 tig, 2 tig + 1, acc[c][2..3] = row g + 8.
// Q8X (type 108, the dense layers' Q8_0 repacked): [K int8 codes][K / 32 f16 scales] a row.
__device__ __forceinline__ void decode8_q8x(const unsigned char* row, int K, int e0, int& w0, int& w1, float& dw) {
    w0 = *(const int*)(row + e0);
    w1 = *(const int*)(row + e0 + 4);
    dw = h2f(rd16(row + K + (e0 / 32) * 2));
}
template <int TY>
__device__ __forceinline__ void gemm_tile(const unsigned char* wa, const unsigned char* wb,
                                          const unsigned char* const* xrow, const float* const* xd, int nch,
                                          float acc[8][4], int K) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    for (int k0 = 0; k0 < K; k0 += 16) {
        int w0, w1;
        float dwa, mwa = 0.f, dwb, mwb = 0.f;
        const int e0 = k0 + 8 * (tig >> 1);
        if (TY == 108) decode8_q8x(wa, K, e0, w0, w1, dwa);
        else decode8i<TY>(wa, e0, w0, w1, dwa, mwa);
        const int a0 = (tig & 1) ? w1 : w0;
        if (TY == 108) decode8_q8x(wb, K, e0, w0, w1, dwb);
        else decode8i<TY>(wb, e0, w0, w1, dwb, mwb);
        const int a1 = (tig & 1) ? w1 : w0;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            if (c >= nch) break;
            const unsigned char* xr = xrow[c];  // this lane's B column (token g of chunk c)
            const unsigned b0 = xr ? *(const unsigned*)(xr + k0 + 4 * tig) : 0u;
            int d[4];
            mma_s8_16816(d, a0, a1, b0);
            int sx = __dp4a((int)b0, 0x01010101, 0);
            sx += __shfl_xor_sync(0xffffffffu, sx, 1);
            sx += __shfl_xor_sync(0xffffffffu, sx, 2);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int col = 2 * tig + j;
                const int sxc = __shfl_sync(0xffffffffu, sx, col * 4);
                const float* xdc = xd[c * 8 + col];
                const float dx = xdc ? xdc[k0 / 32] : 0.f;
                if (TY == 12 || TY == 13) {
                    acc[c][j] = fmaf(dx, fmaf(dwa, i2f(d[j]), -mwa * i2f(sxc)), acc[c][j]);
                    acc[c][2 + j] = fmaf(dx, fmaf(dwb, i2f(d[2 + j]), -mwb * i2f(sxc)), acc[c][2 + j]);
                } else {
                    acc[c][j] = fmaf(dx * dwa, i2f(d[j]), acc[c][j]);
                    acc[c][2 + j] = fmaf(dx * dwb, i2f(d[2 + j]), acc[c][2 + j]);
                }
            }
        }
    }
}

// The k32 tile for weight types whose scale (and min) is constant over each 32 columns (all
// but Q3_K / Q6_K, 16-column scales): one mma.sync m16n8k32 per 32 columns and chunk, and the
// row's and token's scales applied once per 32 columns instead of per 16. The tokens' f32
// scales and (for Q4_K / Q5_K mins) per-32 code sums are staged in shared memory a K slab at a
// time, so the inner loop has no shuffles and no scattered scale loads. Every thread of the
// block must call it (it syncs); `live` = false warps only help stage.
#define GT_KS 1024
template <int TY>
__device__ __forceinline__ void gt_dec(const unsigned char* row, int K, int e0, int& w0, int& w1, float& dw, float& mw) {
    if (TY == 108) { decode8_q8x(row, K, e0, w0, w1, dw); mw = 0.f; }
    else decode8i<TY>(row, e0, w0, w1, dw, mw);
}
template <int TY>
__device__ __forceinline__ void gemm_tile32(const unsigned char* wa, const unsigned char* wb, bool live,
                                            const unsigned char* const* xrow_s, const float* const* xd_s, int nch,
                                            float acc[8][4], int K, float (*sxd)[GT_KS / 32 + 1],
                                            int (*ssx)[GT_KS / 32 + 1]) {
    constexpr bool MIN = TY == 12 || TY == 13;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const unsigned char* xr[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) xr[c] = xrow_s[c * 8 + g];
    for (int ks = 0; ks < K; ks += GT_KS) {
        const int kn = min(GT_KS, K - ks), nb = kn / 32;
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * nb; i += blockDim.x) {
            const int t = i / nb, b = i - t * nb;
            const float* xd = xd_s[t];
            sxd[t][b] = xd ? xd[ks / 32 + b] : 0.f;
            if (MIN) {
                int sm = 0;
                const unsigned char* r = xrow_s[t];
                if (r) {
                    const uint4* q = (const uint4*)(r + ks + b * 32);
                    const uint4 u0 = q[0], u1 = q[1];
                    sm = __dp4a((int)u0.x, 0x01010101, sm); sm = __dp4a((int)u0.y, 0x01010101, sm);
                    sm = __dp4a((int)u0.z, 0x01010101, sm); sm = __dp4a((int)u0.w, 0x01010101, sm);
                    sm = __dp4a((int)u1.x, 0x01010101, sm); sm = __dp4a((int)u1.y, 0x01010101, sm);
                    sm = __dp4a((int)u1.z, 0x01010101, sm); sm = __dp4a((int)u1.w, 0x01010101, sm);
                }
                ssx[t][b] = sm;
            }
        }
        __syncthreads();
        if (!live) continue;
        #pragma unroll 4
        for (int k0 = ks; k0 < ks + kn; k0 += 32) {
            unsigned a[4];
            int w0, w1;
            float dwa, mwa, dwb, mwb;
            // The 32 columns permuted (the same for weights and tokens, so the dot is the
            // same sum): lane tig's 8 consecutive columns k0 + 8 tig.. are the MMA's k = 4 tig..
            // and 16 + 4 tig.., so each decode of 8 weights is used whole.
            const int e0 = k0 + 8 * tig;
            gt_dec<TY>(wa, K, e0, w0, w1, dwa, mwa);
            a[0] = (unsigned)w0;
            a[2] = (unsigned)w1;
            gt_dec<TY>(wb, K, e0, w0, w1, dwb, mwb);
            a[1] = (unsigned)w0;
            a[3] = (unsigned)w1;
            const int bi = (k0 - ks) >> 5;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                if (c >= nch) break;
                const unsigned char* x = xr[c];
                const uint2 xv = x ? *(const uint2*)(x + k0 + 8 * tig) : make_uint2(0u, 0u);
                const unsigned b0 = xv.x, b1 = xv.y;
                int d[4];
                mma_s8_16832(d, a, b0, b1);
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int t = c * 8 + 2 * tig + j;
                    const float dx = sxd[t][bi];
                    if (MIN) {
                        const float sx = i2f(ssx[t][bi]);
                        acc[c][j] = fmaf(dx, fmaf(dwa, i2f(d[j]), -mwa * sx), acc[c][j]);
                        acc[c][2 + j] = fmaf(dx, fmaf(dwb, i2f(d[2 + j]), -mwb * sx), acc[c][2 + j]);
                    } else {
                        acc[c][j] = fmaf(dx * dwa, i2f(d[j]), acc[c][j]);
                        acc[c][2 + j] = fmaf(dx * dwb, i2f(d[2 + j]), acc[c][2 + j]);
                    }
                }
            }
        }
    }
}

// gemm_tile32 for Q4_K / Q5_K rows: a superblock's operands (d, dmin, scales, this lane's qs
// words) loaded once per 256 columns with the next superblock's loads in flight (sbk_load, as
// the batch MMA kernels), instead of 4-5 scattered loads per 8 weights. Same staging and math.
template <int TY, bool TL = false>
__device__ __forceinline__ void gemm_tile_sbk(const unsigned char* wa, const unsigned char* wb, bool live,
                                              const unsigned char* const* xrow_s, const float* const* xd_s, int nch,
                                              float acc[8][4], int K, float (*sxd)[GT_KS / 32 + 1],
                                              int (*ssx)[GT_KS / 32 + 1]) {
    constexpr int BB = TY == 12 ? 144 : 176;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const unsigned char* xr[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) xr[c] = xrow_s[c * 8 + g];
    SbK ra, rb, pa, pb;
    // TL: wa is the rows' TCX tile column (wb unused), 2368 / 2880 bytes a superblock.
    constexpr int TB = TY == 12 ? 2368 : 2880;
    if (live) {
        if (TL) { sbk_load_t<TY>(wa, g, tig, 0, ra); sbk_load_t<TY>(wa, g, tig, 1, rb); }
        else { sbk_load<TY>(wa, tig, ra); sbk_load<TY>(wb, tig, rb); }
    }
    for (int ks = 0; ks < K; ks += GT_KS) {
        const int kn = min(GT_KS, K - ks), nb = kn / 32;
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * nb; i += blockDim.x) {
            const int t = i / nb, b = i - t * nb;
            const float* xd = xd_s[t];
            sxd[t][b] = xd ? xd[ks / 32 + b] : 0.f;
            int sm = 0;
            const unsigned char* r = xrow_s[t];
            if (r) {
                const uint4* q = (const uint4*)(r + ks + b * 32);
                const uint4 u0 = q[0], u1 = q[1];
                sm = __dp4a((int)u0.x, 0x01010101, sm); sm = __dp4a((int)u0.y, 0x01010101, sm);
                sm = __dp4a((int)u0.z, 0x01010101, sm); sm = __dp4a((int)u0.w, 0x01010101, sm);
                sm = __dp4a((int)u1.x, 0x01010101, sm); sm = __dp4a((int)u1.y, 0x01010101, sm);
                sm = __dp4a((int)u1.z, 0x01010101, sm); sm = __dp4a((int)u1.w, 0x01010101, sm);
            }
            ssx[t][b] = sm;
        }
        __syncthreads();
        if (!live) continue;
        for (int k0 = ks; k0 < ks + kn; k0 += 256) {
            const int sb = k0 / 256;
            if ((sb + 1) * 256 < K) {
                if (TL) { sbk_load_t<TY>(wa + (sb + 1) * TB, g, tig, 0, pa); sbk_load_t<TY>(wa + (sb + 1) * TB, g, tig, 1, pb); }
                else { sbk_load<TY>(wa + (sb + 1) * BB, tig, pa); sbk_load<TY>(wb + (sb + 1) * BB, tig, pb); }
            }
            #pragma unroll
            for (int st = 0; st < 8; st++) {
                const int j = st >> 1, hi = st & 1;
                unsigned a[4];
                #pragma unroll
                for (int w = 0; w < 2; w++) {
                    unsigned qa = hi ? (ra.q[j][w] >> 4) & 0x0F0F0F0Fu : ra.q[j][w] & 0x0F0F0F0Fu;
                    unsigned qb = hi ? (rb.q[j][w] >> 4) & 0x0F0F0F0Fu : rb.q[j][w] & 0x0F0F0F0Fu;
                    if (TY == 13) {
                        qa |= ((ra.h[w] >> st) & 0x01010101u) << 4;
                        qb |= ((rb.h[w] >> st) & 0x01010101u) << 4;
                    }
                    a[w * 2] = qa;
                    a[w * 2 + 1] = qb;
                }
                int s0, m0, s1, m1;
                scale_min_w(st, ra.sc, s0, m0);
                scale_min_w(st, rb.sc, s1, m1);
                const float da = ra.d * i2f(s0), ma = ra.dm * i2f(m0);
                const float db = rb.d * i2f(s1), mb = rb.dm * i2f(m1);
                const int e = k0 + st * 32, bi = (e - ks) >> 5;
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (c >= nch) break;
                    const unsigned char* x = xr[c];
                    const unsigned b0 = x ? *(const unsigned*)(x + e + 4 * tig) : 0u;
                    const unsigned b1 = x ? *(const unsigned*)(x + e + 16 + 4 * tig) : 0u;
                    int d[4];
                    mma_s8_16832(d, a, b0, b1);
                    #pragma unroll
                    for (int jj = 0; jj < 2; jj++) {
                        const int t = c * 8 + 2 * tig + jj;
                        const float dx = sxd[t][bi], sx = i2f(ssx[t][bi]);
                        acc[c][jj] = fmaf(dx, fmaf(da, i2f(d[jj]), -ma * sx), acc[c][jj]);
                        acc[c][2 + jj] = fmaf(dx, fmaf(db, i2f(d[2 + jj]), -mb * sx), acc[c][2 + jj]);
                    }
                }
            }
            if ((sb + 1) * 256 < K) { ra = pa; rb = pb; }
        }
    }
}

// The token operand staged in shared memory: 128-column slabs of the tile's 64 token rows,
// double-buffered with cp.async (all 8 warps share one copy instead of each loading its own
// from global, and the MMA's B reads become short-latency LDS). Row stride XS_STR bytes keeps
// the warp's fragment reads conflict-free: 160 (≡ 8 words mod 32) for the 8-byte reads of the
// permuted k32 tile, 144 (≡ 4) for the sbk tile's 4-byte reads. Same MMAs and accumulation
// order as gemm_tile32 / gemm_tile_sbk, so the results are bitwise equal.
#define XS_COLS 128
#define GT_SCR (2 * 64 * (GT_KS / 32 + 1))
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
template <int STR>
__device__ __forceinline__ void xs_issue(unsigned char* xs, const unsigned char* const* xrow_s, int k0) {
    // 64 rows x 8 16-byte pieces = 512 copies, 2 a thread (block 256).
    for (int i = threadIdx.x; i < 64 * (XS_COLS / 16); i += blockDim.x) {
        const int t = i >> 3, pc = i & 7;
        const unsigned char* r = xrow_s[t];
        cp16z(xs + t * STR + pc * 16, (r ? r : xrow_s[0]) + k0 + pc * 16, r != nullptr);
    }
    cp_commit();
}
// The tokens' scale and (Q4_K / Q5_K mins) code sum per 32 columns, interleaved so a lane's
// two tokens' four values are one 16-byte read: sds[b][t] = {dx, (float)sum}.
typedef float2 SdsT[64];
__device__ __forceinline__ void gt_stage_sds(bool MIN, const unsigned char* const* xrow_s, const float* const* xd_s,
                                             int ks, int nb, SdsT* sds) {
    for (int i = threadIdx.x; i < 64 * nb; i += blockDim.x) {
        const int b = i >> 6, t = i & 63;
        const float* xd = xd_s[t];
        int sm = 0;
        if (MIN) {
            const unsigned char* r = xrow_s[t];
            if (r) {
                const uint4* q = (const uint4*)(r + ks + b * 32);
                const uint4 u0 = q[0], u1 = q[1];
                sm = __dp4a((int)u0.x, 0x01010101, sm); sm = __dp4a((int)u0.y, 0x01010101, sm);
                sm = __dp4a((int)u0.z, 0x01010101, sm); sm = __dp4a((int)u0.w, 0x01010101, sm);
                sm = __dp4a((int)u1.x, 0x01010101, sm); sm = __dp4a((int)u1.y, 0x01010101, sm);
                sm = __dp4a((int)u1.z, 0x01010101, sm); sm = __dp4a((int)u1.w, 0x01010101, sm);
            }
        }
        sds[b][t] = make_float2(xd ? xd[ks / 32 + b] : 0.f, i2f(sm));
    }
}
template <int TY>
__device__ __forceinline__ void gemm_tile32s(const unsigned char* wa, const unsigned char* wb, bool live,
                                             const unsigned char* const* xrow_s, const float* const* xd_s, int nch,
                                             float acc[8][4], int K, SdsT* sds, unsigned char* xs) {
    constexpr bool MIN = TY == 12 || TY == 13;
    constexpr int STR = 160;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int nq = K / XS_COLS;
    // The weights' next 32 columns decoded a step ahead (their loads in flight during this
    // step's MMAs).
    int nw0a = 0, nw1a = 0, nw0b = 0, nw1b = 0;
    float ndwa = 0.f, nmwa = 0.f, ndwb = 0.f, nmwb = 0.f;
    if (live) {
        gt_dec<TY>(wa, K, 8 * tig, nw0a, nw1a, ndwa, nmwa);
        gt_dec<TY>(wb, K, 8 * tig, nw0b, nw1b, ndwb, nmwb);
    }
    xs_issue<STR>(xs, xrow_s, 0);
    for (int q = 0; q < nq; q++) {
        const int kb = q * XS_COLS;
        const int ks = kb - kb % GT_KS;
        if (kb == ks) {
            __syncthreads();
            gt_stage_sds(MIN, xrow_s, xd_s, ks, min(GT_KS, K - ks) / 32, sds);
        }
        unsigned char* cur = xs + (q & 1) * 64 * STR;
        if (q + 1 < nq) { xs_issue<STR>(xs + ((q + 1) & 1) * 64 * STR, xrow_s, kb + XS_COLS); cp_wait1(); }
        else cp_wait0();
        __syncthreads();
        if (live) {
            #pragma unroll
            for (int kk = 0; kk < XS_COLS; kk += 32) {
                const int k0 = kb + kk;
                unsigned a[4] = {(unsigned)nw0a, (unsigned)nw0b, (unsigned)nw1a, (unsigned)nw1b};
                const float dwa = ndwa, mwa = nmwa, dwb = ndwb, mwb = nmwb;
                if (k0 + 32 < K) {
                    const int e1 = k0 + 32 + 8 * tig;
                    gt_dec<TY>(wa, K, e1, nw0a, nw1a, ndwa, nmwa);
                    gt_dec<TY>(wb, K, e1, nw0b, nw1b, ndwb, nmwb);
                }
                const int bi = (k0 - ks) >> 5;
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (c >= nch) break;
                    const uint2 xv = *(const uint2*)(cur + (c * 8 + g) * STR + kk + 8 * tig);
                    int d[4];
                    mma_s8_16832(d, a, xv.x, xv.y);
                    const float4 q4 = *(const float4*)&sds[bi][c * 8 + 2 * tig];
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const float dx = j ? q4.z : q4.x;
                        if (MIN) {
                            const float sx = j ? q4.w : q4.y;
                            acc[c][j] = fmaf(dx, fmaf(dwa, i2f(d[j]), -mwa * sx), acc[c][j]);
                            acc[c][2 + j] = fmaf(dx, fmaf(dwb, i2f(d[2 + j]), -mwb * sx), acc[c][2 + j]);
                        } else {
                            acc[c][j] = fmaf(dx * dwa, i2f(d[j]), acc[c][j]);
                            acc[c][2 + j] = fmaf(dx * dwb, i2f(d[2 + j]), acc[c][2 + j]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}
template <int TY, int HALF, int STR>
__device__ __forceinline__ void sbks_half(const SbK& ra, const SbK& rb, const unsigned char* cur, int kb, int ks, int nch,
                                          float acc[8][4], const SdsT* sds) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
            #pragma unroll
            for (int s4 = 0; s4 < 4; s4++) {
                const int st = HALF * 4 + s4;
                const int j = st >> 1, hi = st & 1;
                unsigned a[4];
                #pragma unroll
                for (int w = 0; w < 2; w++) {
                    unsigned qa = hi ? (ra.q[j][w] >> 4) & 0x0F0F0F0Fu : ra.q[j][w] & 0x0F0F0F0Fu;
                    unsigned qb = hi ? (rb.q[j][w] >> 4) & 0x0F0F0F0Fu : rb.q[j][w] & 0x0F0F0F0Fu;
                    if (TY == 13) {
                        qa |= ((ra.h[w] >> st) & 0x01010101u) << 4;
                        qb |= ((rb.h[w] >> st) & 0x01010101u) << 4;
                    }
                    a[w * 2] = qa;
                    a[w * 2 + 1] = qb;
                }
                int s0, m0, s1, m1;
                scale_min_w(st, ra.sc, s0, m0);
                scale_min_w(st, rb.sc, s1, m1);
                const float da = ra.d * i2f(s0), ma = ra.dm * i2f(m0);
                const float db = rb.d * i2f(s1), mb = rb.dm * i2f(m1);
                const int e = kb + s4 * 32, bi = (e - ks) >> 5;
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (c >= nch) break;
                    const unsigned char* xr = cur + (c * 8 + g) * STR + s4 * 32;
                    const unsigned b0 = *(const unsigned*)(xr + 4 * tig);
                    const unsigned b1 = *(const unsigned*)(xr + 16 + 4 * tig);
                    int d[4];
                    mma_s8_16832(d, a, b0, b1);
                    const float4 q4 = *(const float4*)&sds[bi][c * 8 + 2 * tig];
                    #pragma unroll
                    for (int jj = 0; jj < 2; jj++) {
                        const float dx = jj ? q4.z : q4.x, sx = jj ? q4.w : q4.y;
                        acc[c][jj] = fmaf(dx, fmaf(da, i2f(d[jj]), -ma * sx), acc[c][jj]);
                        acc[c][2 + jj] = fmaf(dx, fmaf(db, i2f(d[2 + jj]), -mb * sx), acc[c][2 + jj]);
                    }
                }
            }
}
template <int TY>
__device__ __forceinline__ void gemm_tile_sbks(const unsigned char* wa, const unsigned char* wb, bool live,
                                               const unsigned char* const* xrow_s, const float* const* xd_s, int nch,
                                               float acc[8][4], int K, SdsT* sds, unsigned char* xs) {
    constexpr int BB = TY == 12 ? 144 : 176;
    constexpr int STR = 144;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    SbK ra, rb, pa, pb;
    if (live) { sbk_load<TY>(wa, tig, ra); sbk_load<TY>(wb, tig, rb); }
    const int nq = K / XS_COLS;
    xs_issue<STR>(xs, xrow_s, 0);
    for (int q = 0; q < nq; q++) {
        const int kb = q * XS_COLS;
        const int ks = kb - kb % GT_KS;
        if (kb == ks) {
            __syncthreads();
            gt_stage_sds(true, xrow_s, xd_s, ks, min(GT_KS, K - ks) / 32, sds);
        }
        unsigned char* cur = xs + (q & 1) * 64 * STR;
        if (q + 1 < nq) { xs_issue<STR>(xs + ((q + 1) & 1) * 64 * STR, xrow_s, kb + XS_COLS); cp_wait1(); }
        else cp_wait0();
        __syncthreads();
        if (live) {
            const int sb = kb / 256, half = (kb / XS_COLS) & 1;
            if (!half && (sb + 1) * 256 < K) {
                sbk_load<TY>(wa + (sb + 1) * BB, tig, pa);
                sbk_load<TY>(wb + (sb + 1) * BB, tig, pb);
            }
            if (half) sbks_half<TY, 1, STR>(ra, rb, cur, kb, ks, nch, acc, sds);
            else sbks_half<TY, 0, STR>(ra, rb, cur, kb, ks, nch, acc, sds);
            if (half && (sb + 1) * 256 < K) { ra = pa; rb = pb; }
        }
        __syncthreads();
    }
}

// `xrow_s` / `xd_s`: the tile's 64 token rows (codes, per-32 scales; null past the tile) in
// shared memory. Every thread of the block calls it.
// `k16`: the old per-16 tile for every type (no syncs), for blocks whose warps differ in type.
__device__ __forceinline__ void gemm_tile_any(int ty, const unsigned char* wa, const unsigned char* wb, bool live,
                                              const unsigned char* const* xrow_s, const float* const* xd_s, int nch,
                                              float acc[8][4], int K, float* scr, bool k16 = false, unsigned char* xs = nullptr) {
    if (k16 || ty == 11 || ty == 14) {
        if (!live) return;
        const int g = (threadIdx.x & 31) >> 2;
        const unsigned char* xr[8];
        #pragma unroll
        for (int c = 0; c < 8; c++) xr[c] = xrow_s[c * 8 + g];
        switch (ty) {
            case 12: gemm_tile<12>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 13: gemm_tile<13>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 14: gemm_tile<14>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 8: gemm_tile<8>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 108: gemm_tile<108>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 11: gemm_tile<11>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 2: gemm_tile<2>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 20: gemm_tile<20>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 6: gemm_tile<6>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 23: gemm_tile<23>(wa, wb, xr, xd_s, nch, acc, K); break;
            case 42: gemm_tile<42>(wa, wb, xr, xd_s, nch, acc, K); break;
            default: break;
        }
        return;
    }
    // One staging area for every instance (statics in the templates would each take their own).
    // `scr` (the caller's, GT_SCR floats of shared memory): the old tiles' sxd / ssx, or the
    // staged tiles' interleaved sds.
    float (*sxd)[GT_KS / 32 + 1] = (float (*)[GT_KS / 32 + 1])scr;
    int (*ssx)[GT_KS / 32 + 1] = (int (*)[GT_KS / 32 + 1])(scr + 64 * (GT_KS / 32 + 1));
    if (xs && K % XS_COLS == 0) {
        SdsT* sds = (SdsT*)scr;
        switch (ty) {
            case 12: gemm_tile_sbks<12>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 13: gemm_tile_sbks<13>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 8: gemm_tile32s<8>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 108: gemm_tile32s<108>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 2: gemm_tile32s<2>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 20: gemm_tile32s<20>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 6: gemm_tile32s<6>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 23: gemm_tile32s<23>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            case 42: gemm_tile32s<42>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sds, xs); return;
            default: break;
        }
    }
    switch (ty) {
        case 12: gemm_tile_sbk<12>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 13: gemm_tile_sbk<13>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
#ifdef CX_TCX_TILES
        case 112: gemm_tile_sbk<12, true>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 113: gemm_tile_sbk<13, true>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
#endif
        case 8: gemm_tile32<8>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 108: gemm_tile32<108>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 2: gemm_tile32<2>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 20: gemm_tile32<20>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 6: gemm_tile32<6>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 23: gemm_tile32<23>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        case 42: gemm_tile32<42>(wa, wb, live, xrow_s, xd_s, nch, acc, K, sxd, ssx); break;
        default: break;
    }
}

// Gate|up: rows of [gate; up] (2 FF), K = hidden, columns the group's tokens (xq rows).
extern "C" __global__ void __launch_bounds__(256) cx_moe_gemm_gu(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                                 const unsigned char* xq, float* out, int T,
                                                                 const unsigned* plan) {
    __shared__ const unsigned char* xrow_s[64];
    __shared__ const float* xd_s[64];
    __shared__ int pair_s[64];
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int a = plan[1 + gi], b = plan[2 + gi];
    const int m0 = a + blockIdx.z * 64;
    if (m0 >= b) return;
    const int cnt = min(64, b - m0), nch = (cnt + 7) / 8;
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const unsigned e = shared ? 0u : ids[p0];
    const float* xdb = (const float*)(xq + (size_t)XQ8_T * CX_HID);
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        const int p = i < cnt ? (int)plan[CX_PLAN_LIST + m0 + i] : -1;
        const int t = p < 0 ? 0 : (shared ? p - T * CX_TOPK : p / CX_TOPK);
        pair_s[i] = p;
        xrow_s[i] = p < 0 ? nullptr : xq + (size_t)t * CX_HID;
        xd_s[i] = p < 0 ? nullptr : xdb + (size_t)t * (CX_HID / 32);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tig = lane & 3;
    const int r0 = (blockIdx.x * 8 + warp) * 16;
    // Field by field (a reference to a by-value kernel parameter copies the structs to local
    // memory in every block).
    const bool gt = r0 < CX_FF;
    const unsigned long long mb = shared ? (gt ? sg.base : su.base) : (gt ? g.base : u.base);
    const unsigned long long ms = shared ? (gt ? sg.estride : su.estride) : (gt ? g.estride : u.estride);
    const int mty = shared ? (gt ? sg.ty : su.ty) : (gt ? g.ty : u.ty);
    const int mrb = shared ? (gt ? sg.row_bytes : su.row_bytes) : (gt ? g.row_bytes : u.row_bytes);
    const int ro = gt ? r0 : r0 - CX_FF;
    const unsigned char* wa = mty >= 112 ? tcx_rowp((const unsigned char*)mb, ms, mrb, e, ro)
                                         : (const unsigned char*)mb + (size_t)e * ms + (size_t)(ro + gq) * mrb;
    const unsigned char* wb = wa + (size_t)8 * mrb;
    float acc[8][4];
    #pragma unroll
    for (int c = 0; c < 8; c++) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;
    // Token tiles staged in smem (K = hidden: 16 slabs); the down GEMM (K = FF, 4 slabs) is
    // faster without.
    __shared__ __align__(16) unsigned char xs[2 * 64 * 160];
    // The common types called directly (through gemm_tile_any's switch the compiler outlines
    // the tile into a CALL, with acc and the operand registers through local memory).
    __shared__ __align__(16) float scr[GT_SCR];
    if (mty == 12) gemm_tile_sbks<12>(wa, wb, true, xrow_s, xd_s, nch, acc, CX_HID, (SdsT*)scr, xs);
    else if (mty == 13) gemm_tile_sbks<13>(wa, wb, true, xrow_s, xd_s, nch, acc, CX_HID, (SdsT*)scr, xs);
    else gemm_tile_any(mty, wa, wb, true, xrow_s, xd_s, nch, acc, CX_HID, scr, false, xs);
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        if (c >= nch) break;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int p = pair_s[c * 8 + 2 * tig + j];
            if (p < 0) continue;
            out[(size_t)p * 2 * CX_FF + r0 + gq] = acc[c][j];
            out[(size_t)p * 2 * CX_FF + r0 + gq + 8] = acc[c][2 + j];
        }
    }
}

// Down: rows HID of the expert's down matrix, K = FF, columns the group's pairs (their int8 h
// from cx_silu_q: codes [CX_NP][FF], then scales [CX_NP][FF / 32]).
extern "C" __global__ void __launch_bounds__(256) cx_moe_gemm_down(ExW d, ExW sd, const unsigned* ids,
                                                                   const unsigned char* hq, float* out, int T,
                                                                   const unsigned* plan) {
    __shared__ const unsigned char* xrow_s[64];
    __shared__ const float* xd_s[64];
    __shared__ int pair_s[64];
    const int gi = blockIdx.y;
    if (gi >= (int)plan[0]) return;
    const int a = plan[1 + gi], b = plan[2 + gi];
    const int m0 = a + blockIdx.z * 64;
    if (m0 >= b) return;
    const int cnt = min(64, b - m0), nch = (cnt + 7) / 8;
    const int p0 = plan[CX_PLAN_LIST + a];
    const bool shared = p0 >= T * CX_TOPK;
    const unsigned e = shared ? 0u : ids[p0];
    const float* hd = (const float*)(hq + (size_t)CX_NP * CX_FF);
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        const int p = i < cnt ? (int)plan[CX_PLAN_LIST + m0 + i] : -1;
        pair_s[i] = p;
        xrow_s[i] = p < 0 ? nullptr : hq + (size_t)p * CX_FF;
        xd_s[i] = p < 0 ? nullptr : hd + (size_t)p * (CX_FF / 32);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tig = lane & 3;
    const int r0 = (blockIdx.x * 8 + warp) * 16;
    const unsigned long long mb = shared ? sd.base : d.base, ms = shared ? sd.estride : d.estride;
    const int mty = shared ? sd.ty : d.ty, mrb = shared ? sd.row_bytes : d.row_bytes;
    const unsigned char* wa = mty >= 112 ? tcx_rowp((const unsigned char*)mb, ms, mrb, e, r0)
                                         : (const unsigned char*)mb + (size_t)e * ms + (size_t)(r0 + gq) * mrb;
    const unsigned char* wb = wa + (size_t)8 * mrb;
    float acc[8][4];
    #pragma unroll
    for (int c = 0; c < 8; c++) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;
    __shared__ __align__(16) float scr[GT_SCR];
    __shared__ __align__(16) unsigned char xs[2 * 64 * 160];
    if (mty == 13) gemm_tile_sbks<13>(wa, wb, true, xrow_s, xd_s, nch, acc, CX_FF, (SdsT*)scr, xs);
    else if (mty == 12) gemm_tile_sbks<12>(wa, wb, true, xrow_s, xd_s, nch, acc, CX_FF, (SdsT*)scr, xs);
    else gemm_tile_any(mty, wa, wb, true, xrow_s, xd_s, nch, acc, CX_FF, scr);
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        if (c >= nch) break;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int p = pair_s[c * 8 + 2 * tig + j];
            if (p < 0) continue;
            out[(size_t)p * CX_HID + r0 + gq] = acc[c][j];
            out[(size_t)p * CX_HID + r0 + gq + 8] = acc[c][2 + j];
        }
    }
}

// Q8X, 32 rows a warp (two m16 tiles: rows g, g + 8 and g + 16, g + 24 of the warp's 32), so
// each staged B fragment and scale read feeds two MMAs (the 16-row tile is bound by those smem
// reads). Same MMAs and accumulation order per output as gemm_tile32s (bitwise). Block 128.
__device__ __forceinline__ void gemm_q8x_32r(const unsigned char* w0r, int rb, const unsigned char* const* xrow_s,
                                             const float* const* xd_s, int nch, float acc[2][8][4], int K, SdsT* sds,
                                             unsigned char* xs) {
    constexpr int STR = 160;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const unsigned char* wr[4] = {w0r, w0r + (size_t)8 * rb, w0r + (size_t)16 * rb, w0r + (size_t)24 * rb};
    const int nq = K / XS_COLS;
    xs_issue<STR>(xs, xrow_s, 0);
    for (int q = 0; q < nq; q++) {
        const int kb = q * XS_COLS;
        const int ks = kb - kb % GT_KS;
        if (kb == ks) {
            __syncthreads();
            gt_stage_sds(false, xrow_s, xd_s, ks, min(GT_KS, K - ks) / 32, sds);
        }
        unsigned char* cur = xs + (q & 1) * 64 * STR;
        if (q + 1 < nq) { xs_issue<STR>(xs + ((q + 1) & 1) * 64 * STR, xrow_s, kb + XS_COLS); cp_wait1(); }
        else cp_wait0();
        __syncthreads();
        #pragma unroll 2
        for (int kk = 0; kk < XS_COLS; kk += 32) {
            const int k0 = kb + kk, e0 = k0 + 8 * tig;
            int w[4][2];
            float dw[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) decode8_q8x(wr[r], K, e0, w[r][0], w[r][1], dw[r]);
            const unsigned a0[4] = {(unsigned)w[0][0], (unsigned)w[1][0], (unsigned)w[0][1], (unsigned)w[1][1]};
            const unsigned a1[4] = {(unsigned)w[2][0], (unsigned)w[3][0], (unsigned)w[2][1], (unsigned)w[3][1]};
            const int bi = (k0 - ks) >> 5;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                if (c >= nch) break;
                const uint2 xv = *(const uint2*)(cur + (c * 8 + g) * STR + kk + 8 * tig);
                int d0[4], d1[4];
                mma_s8_16832(d0, a0, xv.x, xv.y);
                mma_s8_16832(d1, a1, xv.x, xv.y);
                const float4 q4 = *(const float4*)&sds[bi][c * 8 + 2 * tig];
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const float dx = j ? q4.z : q4.x;
                    acc[0][c][j] = fmaf(dx * dw[0], i2f(d0[j]), acc[0][c][j]);
                    acc[0][c][2 + j] = fmaf(dx * dw[1], i2f(d0[2 + j]), acc[0][c][2 + j]);
                    acc[1][c][j] = fmaf(dx * dw[2], i2f(d1[j]), acc[1][c][j]);
                    acc[1][c][2 + j] = fmaf(dx * dw[3], i2f(d1[2 + j]), acc[1][c][2 + j]);
                }
            }
        }
        __syncthreads();
    }
}
// cx_gemm_dense with 4 warps of 32 rows (128 rows a block, same grid). A block whose warps are
// not all Q8X in one segment each runs the k16 tile per 16 rows (no syncs).
extern "C" __global__ void __launch_bounds__(128) cx_gemm_dense32(const Seg* segs, int nseg, const unsigned char* xq,
                                                                  int k, float* out, int ostride, int T) {
    __shared__ const float* xd_s[64];
    __shared__ const unsigned char* xrow_s[64];
    __shared__ __align__(16) unsigned char xs[2 * 64 * 160];
    __shared__ __align__(16) float scr[GT_SCR];
    const int t0 = blockIdx.y * 64, cnt = min(64, T - t0), nch = (cnt + 7) / 8;
    const float* xdb = (const float*)(xq + (size_t)XQ8_T * k);
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        xd_s[i] = i < cnt ? xdb + (size_t)(t0 + i) * (k / 32) : nullptr;
        xrow_s[i] = i < cnt ? xq + (size_t)(t0 + i) * k : nullptr;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tig = lane & 3;
    int row = (blockIdx.x * 4 + warp) * 32, s = 0;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    const bool live = s < nseg;
    const bool fast = __syncthreads_and(live && segs[s].type == 108 && row + 32 <= segs[s].rows && k % XS_COLS == 0);
    if (fast) {
        const Seg sg = segs[s];
        float acc[2][8][4];
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int c = 0; c < 8; c++) acc[h][c][0] = acc[h][c][1] = acc[h][c][2] = acc[h][c][3] = 0.f;
        gemm_q8x_32r((const unsigned char*)sg.w + (size_t)(row + gq) * sg.row_bytes, sg.row_bytes, xrow_s, xd_s, nch,
                     acc, k, (SdsT*)scr, xs);
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                if (c >= nch) break;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int i = c * 8 + 2 * tig + j;
                    if (i >= cnt) continue;
                    float* o = out + (size_t)(t0 + i) * ostride + sg.out_off + row + 16 * h + gq;
                    o[0] = acc[h][c][j];
                    o[8] = acc[h][c][2 + j];
                }
            }
        return;
    }
    for (int h = 0; h < 2; h++) {
        int r = (blockIdx.x * 4 + warp) * 32 + 16 * h, s2 = 0;
        while (s2 < nseg && r >= segs[s2].rows) { r -= segs[s2].rows; s2++; }
        if (s2 >= nseg) return;
        const Seg sg = segs[s2];
        const unsigned char* wa = (const unsigned char*)sg.w + (size_t)(r + gq) * sg.row_bytes;
        float acc[8][4];
        #pragma unroll
        for (int c = 0; c < 8; c++) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;
        gemm_tile_any(sg.type, wa, wa + (size_t)8 * sg.row_bytes, true, xrow_s, xd_s, nch, acc, k, scr, true);
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            if (c >= nch) break;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int i = c * 8 + 2 * tig + j;
                if (i >= cnt) continue;
                float* o = out + (size_t)(t0 + i) * ostride + sg.out_off + r + gq;
                o[0] = acc[c][j];
                o[8] = acc[c][2 + j];
            }
        }
    }
}

// A dense matrix (stacked segments, as fe_gemv8 takes them) over the window's rows:
// out[t][out_off + row] for t < T, from xq's int8 codes ([XQ8_T][k], then [XQ8_T][k / 32]
// scales). Each warp's 16 rows lie in one segment (rows % 16 == 0, checked on the host).
// Grid (rows / 128, T / 64 rounded up), block 256.
extern "C" __global__ void __launch_bounds__(256) cx_gemm_dense(const Seg* segs, int nseg, const unsigned char* xq,
                                                                int k, float* out, int ostride, int T) {
    __shared__ const float* xd_s[64];
    __shared__ const unsigned char* xrow_s[64];
    const int t0 = blockIdx.y * 64, cnt = min(64, T - t0), nch = (cnt + 7) / 8;
    const float* xdb = (const float*)(xq + (size_t)XQ8_T * k);
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        xd_s[i] = i < cnt ? xdb + (size_t)(t0 + i) * (k / 32) : nullptr;
        xrow_s[i] = i < cnt ? xq + (size_t)(t0 + i) * k : nullptr;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tig = lane & 3;
    int row = (blockIdx.x * 8 + warp) * 16, s = 0;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    // Past the last row: stage with the block (the k32 tile syncs), compute nothing. A block
    // whose live warps sit in segments of different types takes the k16 tile (no syncs).
    const bool live = s < nseg;
    int r0 = blockIdx.x * 128, s0 = 0;
    while (s0 < nseg - 1 && r0 >= segs[s0].rows) { r0 -= segs[s0].rows; s0++; }
    const int ty0 = segs[s0].type;
    const Seg sg = segs[live ? s : s0];
    const bool k16 = !__syncthreads_and(sg.type == ty0);
    const unsigned char* wa = (const unsigned char*)sg.w + (size_t)((live ? row : 0) + gq) * sg.row_bytes;
    const unsigned char* wb = wa + (size_t)8 * sg.row_bytes;
    float acc[8][4];
    #pragma unroll
    for (int c = 0; c < 8; c++) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;
    __shared__ __align__(16) unsigned char xs[2 * 64 * 160];
    __shared__ __align__(16) float scr[GT_SCR];
    gemm_tile_any(sg.type, wa, wb, live, xrow_s, xd_s, nch, acc, k, scr, k16, xs);
    if (!live) return;
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        if (c >= nch) break;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int i = c * 8 + 2 * tig + j;
            if (i >= cnt) continue;
            float* o = out + (size_t)(t0 + i) * ostride + sg.out_off + row + gq;
            o[0] = acc[c][j];
            o[8] = acc[c][2 + j];
        }
    }
}

// ---- K-split dense GEMV ----
// A block of 8 warps handles 2 rows; the 4 warps of a row each take a contiguous quarter of k
// (k % 1024 == 0) and their partial sums are added in warp order. Same segments, outputs and
// activation formats as fe_gemv8 (int8 codes for quantized rows, f32 for F32/BF16 rows), so
// it drops in for it; at T = 1 a 2048-row matrix gets 8192 warps instead of 1024.
template <int T, int TY>
__device__ __forceinline__ void dotk8(const unsigned char* w, const unsigned char* xq, const float* xd, int k, int k0, int k1, float* acc) {
    const int lane = threadIdx.x & 31;
    for (int e0 = k0 + lane * 8; e0 < k1; e0 += 256) {
        int w0, w1;
        float dw, mw;
        decode8i<TY>(w, e0, w0, w1, dw, mw);
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const uint2 xv = *(const uint2*)(xq + (size_t)t * k + e0);
            const float dx = xd[t * (k / 32) + e0 / 32];
            const int s = __dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0));
            if (TY == 12 || TY == 13) {
                const int sx = __dp4a(0x01010101, (int)xv.x, __dp4a(0x01010101, (int)xv.y, 0));
                acc[t] = fmaf(dx, fmaf(dw, i2f(s), -mw * i2f(sx)), acc[t]);
            } else {
                acc[t] = fmaf(dx * dw, i2f(s), acc[t]);
            }
        }
    }
}
template <int T, int TY>
__device__ __forceinline__ void dotkf(const unsigned char* w, const float* x, int k, int k0, int k1, float* acc) {
    const int lane = threadIdx.x & 31;
    for (int e0 = k0 + lane * 8; e0 < k1; e0 += 256) {
        float v[8];
        decode8<TY>(w, e0, v);
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float4 a = *(const float4*)(x + (size_t)t * k + e0);
            const float4 c = *(const float4*)(x + (size_t)t * k + e0 + 4);
            float s = acc[t];
            s = fmaf(v[0], a.x, s); s = fmaf(v[1], a.y, s); s = fmaf(v[2], a.z, s); s = fmaf(v[3], a.w, s);
            s = fmaf(v[4], c.x, s); s = fmaf(v[5], c.y, s); s = fmaf(v[6], c.z, s); s = fmaf(v[7], c.w, s);
            acc[t] = s;
        }
    }
}
// Q8X: Q8_0 rows repacked at load (lossless): [k int8 codes][k / 32 f16 scales], so a lane
// reads 16 codes with one aligned load. Type id 108. (k / Q) % 512 == 0.
template <int T>
__device__ __forceinline__ void dotq8x(const unsigned char* w, const unsigned char* xq, const float* xd, int k, int k0, int k1, float* acc) {
    const int lane = threadIdx.x & 31;
    const unsigned short* sc = (const unsigned short*)(w + k);
    // Two 512-wide steps a pass, both loads issued before either is used ((k1 - k0) % 1024 == 0
    // at Q <= 2 for k = 2048 / 4096).
    for (int e0 = k0 + lane * 16; e0 < k1; e0 += 1024) {
        const int e1 = e0 + 512;
        const bool two = e1 < k1;
        const uint4 c0 = *(const uint4*)(w + e0);
        const uint4 c1 = two ? *(const uint4*)(w + e1) : make_uint4(0, 0, 0, 0);
        const float dw0 = h2f(sc[e0 / 32]), dw1 = two ? h2f(sc[e1 / 32]) : 0.f;
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const uint4 xv = *(const uint4*)(xq + (size_t)t * k + e0);
            const float dx = xd[t * (k / 32) + e0 / 32];
            const int s = __dp4a((int)c0.x, (int)xv.x, __dp4a((int)c0.y, (int)xv.y,
                          __dp4a((int)c0.z, (int)xv.z, __dp4a((int)c0.w, (int)xv.w, 0))));
            acc[t] = fmaf(dx * dw0, i2f(s), acc[t]);
            if (two) {
                const uint4 yv = *(const uint4*)(xq + (size_t)t * k + e1);
                const float dy = xd[t * (k / 32) + e1 / 32];
                const int u = __dp4a((int)c1.x, (int)yv.x, __dp4a((int)c1.y, (int)yv.y,
                              __dp4a((int)c1.z, (int)yv.z, __dp4a((int)c1.w, (int)yv.w, 0))));
                acc[t] = fmaf(dy * dw1, i2f(u), acc[t]);
            }
        }
    }
}
template <int T, int Q>
__device__ void gemvk_core(const Seg* segs, int nseg, const float* x, const unsigned char* xq, const float* xd, int k, float* out, int ostride) {
    __shared__ float part[8][T];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = warp / Q, q = warp % Q;
    int row = blockIdx.x * (8 / Q) + j, s = 0;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    const bool ok = s < nseg;
    if (ok) {
        const Seg sg = segs[s];
        const unsigned char* w = (const unsigned char*)sg.w + (size_t)row * sg.row_bytes;
        const int k0 = q * (k / Q), k1 = k0 + k / Q;
        switch (sg.type) {
            case 30: dotkf<T, 30>(w, x, k, k0, k1, acc); break;
            case 0: dotkf<T, 0>(w, x, k, k0, k1, acc); break;
            case 108: dotq8x<T>(w, xq, xd, k, k0, k1, acc); break;
            case 8: dotk8<T, 8>(w, xq, xd, k, k0, k1, acc); break;
            case 2: dotk8<T, 2>(w, xq, xd, k, k0, k1, acc); break;
            case 20: dotk8<T, 20>(w, xq, xd, k, k0, k1, acc); break;
            case 6: dotk8<T, 6>(w, xq, xd, k, k0, k1, acc); break;
            case 42: dotk8<T, 42>(w, xq, xd, k, k0, k1, acc); break;
            case 23: dotk8<T, 23>(w, xq, xd, k, k0, k1, acc); break;
            case 11: dotk8<T, 11>(w, xq, xd, k, k0, k1, acc); break;
            case 12: dotk8<T, 12>(w, xq, xd, k, k0, k1, acc); break;
            case 13: dotk8<T, 13>(w, xq, xd, k, k0, k1, acc); break;
            case 14: dotk8<T, 14>(w, xq, xd, k, k0, k1, acc); break;
            default: break;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) part[warp][t] = a;
    }
    __syncthreads();
    if (ok && q == 0 && lane < T) {
        float v = part[Q * j][lane];
        #pragma unroll
        for (int i = 1; i < Q; i++) v += part[Q * j + i][lane];
        out[(size_t)lane * ostride + segs[s].out_off + row] = v;
    }
}
template <int T, int Q>
__device__ __forceinline__ void gemvk_body(const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) {
    gemvk_core<T, Q>(segs, nseg, x, xq, (const float*)(xq + (size_t)XQ8_T * k), k, out, ostride);
}
#define GEMVK_T(T) extern "C" __global__ void __launch_bounds__(256) cx_gemvk_t##T( \
    const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvk_body<T, 4>(segs, nseg, x, xq, k, out, ostride); } \
    extern "C" __global__ void __launch_bounds__(256) cx_gemvk8_t##T( \
    const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvk_body<T, 8>(segs, nseg, x, xq, k, out, ostride); } \
    extern "C" __global__ void __launch_bounds__(256) cx_gemvk2_t##T( \
    const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvk_body<T, 2>(segs, nseg, x, xq, k, out, ostride); } \
    extern "C" __global__ void __launch_bounds__(256) cx_gemvk1_t##T( \
    const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvk_body<T, 1>(segs, nseg, x, xq, k, out, ostride); }
GEMVK_T(1) GEMVK_T(2) GEMVK_T(3) GEMVK_T(4) GEMVK_T(5) GEMVK_T(6) GEMVK_T(7) GEMVK_T(8)

// Q8X GEMV with more bytes in flight (`TANG_CODER_Q8XW=R`): a warp owns R whole rows, a lane
// issues every one of its k / 512 uint4 code loads for all R rows (16·R·k/512 bytes) before any
// math, then dots them with each of the T activation rows. All segments Q8X, k % 512 == 0,
// k <= 4096. Not bitwise the K-split kernels' order.
template <int T, int R>
__device__ void gemvq8w_core(const Seg* segs, int nseg, const unsigned char* xq, const float* xd, int k, float* out, int ostride) {
#if __CUDA_ARCH__ >= 900
    // programmatic dependent launch (TANG_CODER_PDL=1): let the next kernel start its own
    // weight stream now; this one waits for its predecessor before reading activations
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kl = k >> 9;                       // uint4 loads a lane a row
    int rr[R], ss[R];
    const unsigned char* wp[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = (blockIdx.x * 8 + warp) * R + r, s = 0;
        while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
        rr[r] = row; ss[r] = s;
        wp[r] = s < nseg ? (const unsigned char*)segs[s].w + (size_t)row * segs[s].row_bytes : nullptr;
    }
    uint4 c[R][8];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int j = 0; j < 8; j++)
            c[r][j] = (wp[r] && j < kl) ? __ldg((const uint4*)(wp[r] + (lane + 32 * j) * 16)) : make_uint4(0, 0, 0, 0);
    }
    float sw[R][8];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int j = 0; j < 8; j++)
            sw[r][j] = (wp[r] && j < kl) ? h2f(((const unsigned short*)(wp[r] + k))[(lane + 32 * j) >> 1]) : 0.f;
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float acc[R];
        #pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j >= kl) break;
            const int e = (lane + 32 * j) * 16;
            const uint4 xv = __ldg((const uint4*)(xq + (size_t)t * k + e));
            const float dx = __ldg(xd + t * (k / 32) + (e >> 5));
            #pragma unroll
            for (int r = 0; r < R; r++) {
                const int d = __dp4a((int)c[r][j].x, (int)xv.x, __dp4a((int)c[r][j].y, (int)xv.y,
                              __dp4a((int)c[r][j].z, (int)xv.z, __dp4a((int)c[r][j].w, (int)xv.w, 0))));
                acc[r] = fmaf(dx * sw[r][j], i2f(d), acc[r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            float a = acc[r];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0 && ss[r] < nseg) out[(size_t)t * ostride + segs[ss[r]].out_off + rr[r]] = a;
        }
    }
}
#define GEMVQ8W_T(T) \
    extern "C" __global__ void __launch_bounds__(256) cx_gemvq8w1_t##T(const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvq8w_core<T, 1>(segs, nseg, xq, (const float*)(xq + (size_t)XQ8_T * k), k, out, ostride); } \
    extern "C" __global__ void __launch_bounds__(256) cx_gemvq8w2_t##T(const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvq8w_core<T, 2>(segs, nseg, xq, (const float*)(xq + (size_t)XQ8_T * k), k, out, ostride); } \
    extern "C" __global__ void __launch_bounds__(256) cx_gemvq8w4_t##T(const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvq8w_core<T, 4>(segs, nseg, xq, (const float*)(xq + (size_t)XQ8_T * k), k, out, ostride); }
GEMVQ8W_T(1) GEMVQ8W_T(2) GEMVQ8W_T(3) GEMVQ8W_T(4) GEMVQ8W_T(5) GEMVQ8W_T(6) GEMVQ8W_T(7) GEMVQ8W_T(8)

// Batch modules (CX_MAXT > 8): the same K-split GEMV over T rows of a wider window, the rows
// starting at row t0 of the module's activations. The host passes x, xq (codes), xd (scales)
// and out already offset to row t0. A row's sum is the narrow kernel's (same lanes, same
// order, same warp split): T only sets how many rows a block carries.
#if CX_MAXT > 8
#define GEMVKB_T(Q, T) extern "C" __global__ void __launch_bounds__(256) cx_gemvkb##Q##_t##T( \
    const Seg* segs, int nseg, const float* x, const unsigned char* xq, const float* xd, int k, float* out, int ostride) { \
    gemvk_core<T, Q>(segs, nseg, x, xq, xd, k, out, ostride); }
GEMVKB_T(1, 8) GEMVKB_T(1, 16) GEMVKB_T(1, 32)
GEMVKB_T(2, 8) GEMVKB_T(2, 16) GEMVKB_T(2, 32)
GEMVKB_T(4, 8) GEMVKB_T(4, 16) GEMVKB_T(4, 32)
GEMVKB_T(8, 8) GEMVKB_T(8, 16) GEMVKB_T(8, 32)
#endif

// Rows of a window may belong to different sequences: row t is at position ROWS[t] of slot
// ROWS[CX_MAXT + t], whose bf16 K/V caches for this layer are at ktab[slot] / vtab[slot].
// Tree windows (several draft chains of one sequence): a row's keys past position
// ROWS[7 CX_MAXT + t] (its chain's root) live at cache index position + ROWS[6 CX_MAXT + t], so
// chains at the same depth don't share cache cells. Offset 0 (every other window) is the plain
// layout.
#define CX_OFF(rows, t, p) ((p) > (int)(rows)[7 * CX_MAXT + (t)] ? (p) + (int)(rows)[6 * CX_MAXT + (t)] : (p))

#ifdef CX_KV8
// int8 K/V cache (`TANG_CODER_KV=int8`): position p of a layer's K (or V) cache is one CX_KVROW-byte
// row, CX_NKV heads of CX_HD int8 codes, then an f16 scale (absmax / 127) for each group of KV8_G
// dims of each head, zero padding to 16 bytes. Value = code * scale of its group.
#define KV8_G 64
#define KV8_NG (CX_HD / KV8_G)
__device__ __forceinline__ float kv8_h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short kv8_f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
#define KV8_Q(base, p, g) ((const signed char*)(base) + (size_t)(p) * CX_KVROW + (g) * CX_HD)
// The scale of dim d of head g at position p.
#define KV8_S(base, p, g, d) kv8_h2f(((const unsigned short*)((const unsigned char*)(base) + (size_t)(p) * CX_KVROW + CX_NKV * CX_HD))[(g) * KV8_NG + (d) / KV8_G])
static_assert(CX_KVROW % 16 == 0 && CX_HD % KV8_G == 0 && KV8_G % 32 == 0, "int8 K/V rows stay 16-byte aligned");
// Max over this thread's group of KV8_G threads (whole warps), of non-negative values.
__device__ float kv8_gmax(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    const int w0 = (threadIdx.x / KV8_G) * (KV8_G / 32);
    float m = 0.f;
    for (int i = 0; i < KV8_G / 32; i++) m = fmaxf(m, sh[w0 + i]);
    return m;
}
// Q, K and V heads are rotated before K/V are quantized: y = H D x / sqrt(CX_HD), D a fixed +-1
// diagonal and H the Sylvester Hadamard matrix (QuaRot, arXiv 2404.00456). q.k is unchanged
// (the rotation is orthogonal), and outlier channels are spread over the head, so a per-head
// absmax scale wastes far fewer codes. Attention then produces H D o; cx_attn_merge undoes it
// (o = D H y / sqrt(CX_HD)). Block = CX_HD threads, thread d owns dim d; s is CX_HD floats.
static_assert((CX_HD & (CX_HD - 1)) == 0, "int8 K/V rotation: head dim a power of two");
__device__ __forceinline__ float kv8_sign(int d) {
    return ((0x5A3C96E1u >> (d & 31)) ^ (unsigned)(d >> 5) ^ (unsigned)(d >> 7)) & 1u ? -1.f : 1.f;
}
__device__ float kv8_wht(float x, int d, float* s) {
    __syncthreads();
    s[d] = x;
    __syncthreads();
    for (int h = 1; h < CX_HD; h <<= 1) {
        const float a = s[d], b = s[d ^ h];
        __syncthreads();
        s[d] = (d & h) ? b - a : a + b;
        __syncthreads();
    }
    return s[d] * rsqrtf((float)CX_HD);
}
__device__ __forceinline__ float kv8_rot(float x, int d, float* s) { return kv8_wht(x * kv8_sign(d), d, s); }
__device__ __forceinline__ float kv8_unrot(float y, int d, float* s) { return kv8_wht(y, d, s) * kv8_sign(d); }
// One head's value x (this thread's dim d) as an int8 code into the row at `row`, each group's
// scale by its first thread.
// Codes are rounded against the stored (f16) scale, so code * scale is what the readers see.
__device__ __forceinline__ void kv8_put(unsigned char* row, int g, int d, float x, float* sh) {
    const unsigned short sh16 = kv8_f2h(kv8_gmax(fabsf(x), sh) / 127.0f);
    const float sc = kv8_h2f(sh16), inv = sc > 0.f ? 1.0f / sc : 0.f;
    const int q = max(-127, min(127, __float2int_rn(x * inv)));
    row[g * CX_HD + d] = (unsigned char)(signed char)q;
    if (d % KV8_G == 0) ((unsigned short*)(row + CX_NKV * CX_HD))[g * KV8_NG + d / KV8_G] = sh16;
}
#endif

// Gated attention prep: q heads (rmsnorm * qn, NeoX rope on CX_ROT dims) into q [t][NH][HD];
// k heads (rmsnorm * kn, rope) and v into the row's slot caches at the row's position.
// Grid (t, NH + NKV), block HD.
extern "C" __global__ void cx_attn_prep(const float* proj, int stride, const float* qn, const float* kn,
                                        const float* rc, const float* rs, const unsigned* rows, float* q,
                                        const unsigned long long* ktab, const unsigned long long* vtab,
                                        float eps) {
    __shared__ float sh[32];
    __shared__ float sx[CX_HD];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const unsigned pos = rows[t], slot = rows[CX_MAXT + t];
    unsigned short* kc = (unsigned short*)ktab[slot];
    unsigned short* vc = (unsigned short*)vtab[slot];
    const float* row = proj + (size_t)t * stride;
    const bool isq = h < CX_NH;
    const int g = h - CX_NH;
    float x = isq ? row[h * 2 * CX_HD + d] : row[CX_QK + g * CX_HD + d];
    const float ss = cx_bsum(x * x, sh);
    x = x * (1.0f / sqrtf(ss / (float)CX_HD + eps)) * (isq ? qn[d] : kn[d]);
    sx[d] = x;
    __syncthreads();
    if (d < CX_ROT) {
        const int half = CX_ROT / 2, i = d % half;
        const float c = rc[(size_t)pos * half + i], s = rs[(size_t)pos * half + i];
        x = d < half ? sx[d] * c - sx[d + half] * s : sx[d - half] * s + sx[d] * c;
    }
    if (isq) {
#ifdef CX_KV8
        x = kv8_rot(x, d, sx);
#endif
        q[((size_t)t * CX_NH + h) * CX_HD + d] = x;
    } else {
#ifdef CX_KV8
        const size_t pr = (size_t)CX_OFF(rows, t, (int)pos) * CX_KVROW;
        kv8_put((unsigned char*)kc + pr, g, d, kv8_rot(x, d, sx), sh);
        kv8_put((unsigned char*)vc + pr, g, d, kv8_rot(row[CX_QV + g * CX_HD + d], d, sx), sh);
#else
        const size_t o = ((size_t)CX_OFF(rows, t, (int)pos) * CX_NKV + g) * CX_HD + d;
        kc[o] = f2bf(x);
        vc[o] = f2bf(row[CX_QV + g * CX_HD + d]);
#endif
    }
}

// Split-K causal attention: block (token c, kv group g, chunk j of CX_CHUNK positions) runs the
// CX_G query heads of group g over its chunk (4 warps; lane owns HD / 32 dims) and writes per
// head (m, l, acc[HD]) to part[c][g][j][G][HD + 2]. Chunks past the token's position write l = 0.
#define CX_DL (CX_HD / 32)
#ifdef CX_KV8
static_assert(CX_DL % 8 == 0 && KV8_G % CX_DL == 0, "int8 split-K: a lane's codes, whole 8-byte words in one scale group");
#endif
extern "C" __global__ void __launch_bounds__(128) cx_attn_part(const float* q, const unsigned long long* ktab,
                                                               const unsigned long long* vtab, const unsigned* rows,
                                                               float* part, int nchunk) {
    __shared__ float wm[4][CX_G], wl[4][CX_G];
    __shared__ float wacc[4][CX_HD];
    const int c = blockIdx.x, g = blockIdx.y, j = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = (int)rows[c];
    const unsigned slot = rows[CX_MAXT + c];
    const unsigned short* kc = (const unsigned short*)ktab[slot];
    const unsigned short* vc = (const unsigned short*)vtab[slot];
    float* out = part + (((size_t)c * CX_NKV + g) * nchunk + j) * CX_G * (CX_HD + 2);
    const int lo = j * CX_CHUNK;
    if (lo > pos) {
        if (threadIdx.x < CX_G) out[threadIdx.x * (CX_HD + 2) + 1] = 0.f;
        return;
    }
    const float scale = 1.0f / sqrtf((float)CX_HD);
    float qv[CX_G][CX_DL];
    for (int h = 0; h < CX_G; h++)
        for (int i = 0; i < CX_DL; i++) qv[h][i] = q[((size_t)c * CX_NH + g * CX_G + h) * CX_HD + lane * CX_DL + i] * scale;
    float m[CX_G], l[CX_G], acc[CX_G][CX_DL];
    for (int h = 0; h < CX_G; h++) {
        m[h] = NEG_INF_F;
        l[h] = 0.f;
        for (int i = 0; i < CX_DL; i++) acc[h][i] = 0.f;
    }
    const int hi = min(pos + 1, lo + CX_CHUNK);
    for (int p = lo + warp; p < hi; p += 4) {
        const size_t pc = (size_t)CX_OFF(rows, c, p);
        float kv[CX_DL], vv[CX_DL];
#ifdef CX_KV8
        // A lane's CX_DL codes are 8-byte aligned (rows and heads are 16-byte aligned).
        const float ks = KV8_S(kc, pc, g, lane * CX_DL), vs = KV8_S(vc, pc, g, lane * CX_DL);
        const uint2* kr = (const uint2*)(KV8_Q(kc, pc, g) + lane * CX_DL);
        const uint2* vr = (const uint2*)(KV8_Q(vc, pc, g) + lane * CX_DL);
        #pragma unroll
        for (int c8 = 0; c8 < CX_DL / 8; c8++) {
            const uint2 kq = kr[c8], vq = vr[c8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned kw = i < 4 ? kq.x : kq.y, vw = i < 4 ? vq.x : vq.y;
                kv[c8 * 8 + i] = (float)(signed char)(kw >> (8 * (i & 3))) * ks;
                vv[c8 * 8 + i] = (float)(signed char)(vw >> (8 * (i & 3))) * vs;
            }
        }
#else
        const unsigned short* kr = kc + (pc * CX_NKV + g) * CX_HD + lane * CX_DL;
        const unsigned short* vr = vc + (pc * CX_NKV + g) * CX_HD + lane * CX_DL;
        #pragma unroll
        for (int i = 0; i < CX_DL; i++) { kv[i] = bf2f(kr[i]); vv[i] = bf2f(vr[i]); }
#endif
        #pragma unroll
        for (int h = 0; h < CX_G; h++) {
            float sc = 0.f;
            #pragma unroll
            for (int i = 0; i < CX_DL; i++) sc = fmaf(qv[h][i], kv[i], sc);
            for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o);
            const float mn = fmaxf(m[h], sc), corr = expf(m[h] - mn), pr = expf(sc - mn);
            #pragma unroll
            for (int i = 0; i < CX_DL; i++) acc[h][i] = fmaf(pr, vv[i], acc[h][i] * corr);
            l[h] = l[h] * corr + pr;
            m[h] = mn;
        }
    }
    for (int h = 0; h < CX_G; h++)
        if (lane == 0) { wm[warp][h] = m[h]; wl[warp][h] = l[h]; }
    for (int h = 0; h < CX_G; h++) {
        __syncthreads();
        for (int i = 0; i < CX_DL; i++) wacc[warp][lane * CX_DL + i] = acc[h][i];
        __syncthreads();
        float M = NEG_INF_F;
        for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) M = fmaxf(M, wm[w][h]);
        float L = 0.f;
        for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) L += wl[w][h] * expf(wm[w][h] - M);
        for (int d = threadIdx.x; d < CX_HD; d += 128) {
            float o = 0.f;
            for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) o += wacc[w][d] * expf(wm[w][h] - M);
            out[h * (CX_HD + 2) + 2 + d] = o;
        }
        if (threadIdx.x == 0) { out[h * (CX_HD + 2)] = M; out[h * (CX_HD + 2) + 1] = L; }
    }
}

// Merge each (token, head)'s chunks and apply the sigmoid gate (the second half of the head's
// q projection). Grid (t, NH), block HD.
// ---- derived wide attention (TANG_CODER_ATTN=derived, flash/derived_attn.rs) ----
//
// The derived flash kernel reads fp16 heads-first operands: q16[NKV][G][TW][HD] (a kv head's G query
// heads folded into its rows), k16 / v16[NKV][NPAD][HD], pos[G * TW] (each row's position, -1 past the
// window's t rows). cx_attn_dq / cx_attn_dkv build them from the prepped queries and the slot's cache;
// cx_attn_dunpack writes the kernel's normalized output as one split (m 0, l 1) of cx_attn_part's
// partials, so cx_attn_merge applies the gate (and the int8 path's inverse rotation) as usual.
__device__ __forceinline__ unsigned short da_f2h(float f) {
    unsigned short h;
    f = fminf(fmaxf(f, -65504.f), 65504.f);
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
// Grid ceil(NKV * G * TW * HD / 256), block 256: one element a thread.
extern "C" __global__ void cx_attn_dq(const float* q, const unsigned* rows, int t, int tw, unsigned short* q16, float* pos) {
    const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t)CX_NKV * CX_G * tw * CX_HD) return;
    const int d = (int)(e % CX_HD);
    const size_t r = e / CX_HD; // (h * G + a) * tw + i
    const int i = (int)(r % tw), ha = (int)(r / tw); // ha = h * G + a: the query head
    q16[e] = i < t ? da_f2h(q[((size_t)i * CX_NH + ha) * CX_HD + d]) : (unsigned short)0;
    if (d == 0 && ha < CX_G) pos[(size_t)ha * tw + i] = i < t ? (float)rows[i] : -1.f;
}
// Grid ceil(NKV * npad * HD / 256), block 256. Keys past the window's last position are zeros (a
// masked P is 0, and 0 * a stale NaN would poison O).
extern "C" __global__ void cx_attn_dkv(const unsigned long long* ktab, const unsigned long long* vtab,
                                       const unsigned* rows, int t, int npad, unsigned short* k16,
                                       unsigned short* v16) {
    const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t)CX_NKV * npad * CX_HD) return;
    const int d = (int)(e % CX_HD);
    const int j = (int)((e / CX_HD) % npad), g = (int)(e / ((size_t)CX_HD * npad));
    const int top = (int)rows[t - 1];
    const unsigned slot = rows[CX_MAXT];
    float kf = 0.f, vf = 0.f;
    if (j <= top) {
#ifdef CX_KV8
        const void* kc = (const void*)ktab[slot];
        const void* vc = (const void*)vtab[slot];
        kf = (float)KV8_Q(kc, j, g)[d] * KV8_S(kc, j, g, d);
        vf = (float)KV8_Q(vc, j, g)[d] * KV8_S(vc, j, g, d);
#else
        const unsigned short* kc = (const unsigned short*)ktab[slot];
        const unsigned short* vc = (const unsigned short*)vtab[slot];
        const size_t o = ((size_t)j * CX_NKV + g) * CX_HD + d;
        kf = __uint_as_float((unsigned)kc[o] << 16);
        vf = __uint_as_float((unsigned)vc[o] << 16);
#endif
    }
    k16[e] = da_f2h(kf);
    v16[e] = da_f2h(vf);
}
// Grid ceil(t * NKV * G * (HD + 2) / 256), block 256: o[NKV][G][TW][HD] -> part[t][NKV][1][G][m, l, acc].
extern "C" __global__ void cx_attn_dunpack(const float* o, int t, int tw, float* part) {
    const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t hs = CX_HD + 2;
    if (e >= (size_t)t * CX_NKV * CX_G * hs) return;
    const int x = (int)(e % hs);
    const size_t r = e / hs; // (i * NKV + g) * G + a
    const int a = (int)(r % CX_G), g = (int)((r / CX_G) % CX_NKV), i = (int)(r / ((size_t)CX_G * CX_NKV));
    part[e] = x == 0 ? 0.f : x == 1 ? 1.f : o[(((size_t)g * CX_G + a) * tw + i) * CX_HD + (x - 2)];
}

extern "C" __global__ void cx_attn_merge(const float* part, int nchunk, const float* proj, int stride, float* out) {
    const int c = blockIdx.x, h = blockIdx.y, g = h / CX_G, hh = h % CX_G, d = threadIdx.x;
    const size_t hs = (size_t)CX_G * (CX_HD + 2);
    const float* pp = part + ((size_t)c * CX_NKV + g) * nchunk * hs + hh * (CX_HD + 2);
    float M = NEG_INF_F;
    for (int j = 0; j < nchunk; j++) if (pp[j * hs + 1] > 0.f) M = fmaxf(M, pp[j * hs]);
    float L = 0.f, o = 0.f;
    for (int j = 0; j < nchunk; j++) {
        const float* qq = pp + j * hs;
        if (qq[1] > 0.f) {
            const float f = expf(qq[0] - M);
            L += qq[1] * f;
            o += qq[2 + d] * f;
        }
    }
    const float gt = proj[(size_t)c * stride + h * 2 * CX_HD + CX_HD + d];
#ifdef CX_JITTER
    o *= 1.f + CX_JITTER * (float)((c * 131 + h * 7 + d) % 17 - 8) / 8.f;
#endif
#ifdef CX_KV8
    __shared__ float su[CX_HD];
    out[((size_t)c * CX_NH + h) * CX_HD + d] = kv8_unrot(o / L, d, su) * sigm_f(gt);
#else
    out[((size_t)c * CX_NH + h) * CX_HD + d] = (o / L) * sigm_f(gt);
#endif
}
// ---- FA2-style tensor-core prefill attention (wide windows, `TANG_CODER_ATTN=fa2`) ----
//
// Structure after FlashAttention-2 (Tri Dao, arXiv 2307.08691; github.com/Dao-AILab/flash-attention,
// BSD-3-Clause): Q tile resident in shared memory, K/V tiles streamed through a cp.async double
// buffer, S = Q K^T and O += P V on bf16 tensor cores (mma.sync m16n8k16, f32 accumulate), online
// softmax in registers with P reused straight from the S accumulators as the A operand.
//
// GQA: the CX_G query heads of a kv group read the same K/V, so the M dimension of a tile is
// (row, head) pairs: query vector v = row * CX_G + head, FA_BM = 64 of them a block (8 rows × 8
// heads at G = 8). Wide windows are one slot at consecutive positions, so the tile's last row has
// its highest position: keys past it are never loaded (fully masked tiles skipped), and only the
// diagonal key tile masks per vector. Long contexts also split the keys across blockIdx.z (each
// split writes (m, l, acc) in cx_attn_part's layout, nchunk = nsplit) so a 256-row window still
// fills the GPU; cx_attn_merge then merges the splits and applies the sigmoid gate.
// Shared rows are padded by 8 bf16 (FA_LD), which makes the fragment loads and ldmatrix
// conflict-free (same effect as FA2's XOR swizzle, simpler addressing).
// Grid (ceil(t * G / FA_BM), NKV, nsplit), block 128 (4 warps × 16 vectors), FA_SMEM dynamic smem.
// Not split-K's summation order (fp16 Q and P): gated by flash-wide-check's greedy + KL bar.
#define FA_BM 64
#define FA_BN 32
#define FA_LD (CX_HD + 8)
#ifdef CX_KV8
// int8 K/V tiles land (cp.async) in a double-buffered staging area [2][K, V][FA_BN][CX_HD] after the
// fp16 tiles, and the per-tile conversion pass dequantizes them into the fp16 tiles. The staging
// is the double buffer, so the fp16 tiles are single (FA_TB = 1): the conversion runs between the
// loop's two barriers, after every warp is done with the previous tile. 100,352 B at head dim 256,
// two blocks an SM (double fp16 tiles made it 134,144: one).
#define FA_TB 1
#define FA_KB(b) 0
#define FA_SMEM ((FA_BM + 2 * FA_TB * FA_BN) * FA_LD * 2 + 4 * FA_BN * CX_HD)
#else
#define FA_TB 2
#define FA_KB(b) (b)
#define FA_SMEM ((FA_BM + 2 * FA_TB * FA_BN) * FA_LD * 2)
#endif
static_assert(FA_BM % CX_G == 0, "FA2 tile: whole rows of G heads");
static_assert(CX_HD % 16 == 0, "FA2 tile: head dim a multiple of 16");

// Q, P and the K/V tiles go through the tensor cores as fp16, not bf16: the K/V cache is bf16, which
// fp16 holds exactly (8 mantissa bits into 11; clamped to fp16's range), and Q (rmsnormed, pre-scaled)
// and P (in [0, 1]) keep 3 more mantissa bits. bf16 Q and P cost KL ~4e-3 against split-K at 1000
// tokens (flash-wide-check); fp16 is 8x finer at the same mma rate.
#define FA_HMAX 65504.f
__device__ __forceinline__ unsigned fa_pack(float lo, float hi) {
#ifdef FA_BF16
    return (unsigned)f2bf(lo) | ((unsigned)f2bf(hi) << 16);
#elif __CUDA_ARCH__ >= 800
    unsigned r;
    lo = fminf(fmaxf(lo, -FA_HMAX), FA_HMAX);
    hi = fminf(fmaxf(hi, -FA_HMAX), FA_HMAX);
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#else
    return 0;
#endif
}
// Two bf16 (a K or V element pair) to two fp16, in place.
__device__ __forceinline__ unsigned fa_bf2h(unsigned x) {
    return fa_pack(__uint_as_float(x << 16), __uint_as_float(x & 0xffff0000u));
}
__device__ __forceinline__ void fa_mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                       unsigned b1) {
#if __CUDA_ARCH__ >= 800
#ifdef FA_BF16
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
#else
    __trap();
#endif
}
__device__ __forceinline__ void fa_cp16(unsigned short* dst, const unsigned short* src) {
#if __CUDA_ARCH__ >= 800
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
#else
    *(uint4*)dst = *(const uint4*)src;
#endif
}

extern "C" __global__ void __launch_bounds__(128) cx_attn_fa2(const float* q, const unsigned long long* ktab,
                                                              const unsigned long long* vtab, const unsigned* rows,
                                                              int t, float* part, int nsplit) {
    extern __shared__ __align__(16) unsigned char fa_sm[];
    unsigned short* sq = (unsigned short*)fa_sm;  // [FA_BM][FA_LD]
    unsigned short* sk = sq + FA_BM * FA_LD;     // [FA_TB][FA_BN][FA_LD]
    unsigned short* sv = sk + FA_TB * FA_BN * FA_LD; // [FA_TB][FA_BN][FA_LD]
#ifdef CX_KV8
    unsigned char* s8 = (unsigned char*)(sv + FA_TB * FA_BN * FA_LD); // [2][2][FA_BN][CX_HD]
#endif
    const int g = blockIdx.y, j = blockIdx.z, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t4 = lane & 3;
    const int nv = t * CX_G, v0 = blockIdx.x * FA_BM;
    const int vlast = min(nv, v0 + FA_BM) - 1;
    const int hi = (int)rows[vlast / CX_G], top = (int)rows[t - 1];
    // Split j covers keys [k0, k0 + span) of the window's [0, top]; this tile needs [k0, k1).
    const int span = ((top + nsplit) / nsplit + FA_BN - 1) / FA_BN * FA_BN;
    const int k0 = j * span, k1 = min(k0 + span, hi + 1);
    const size_t vs = (size_t)CX_G * (CX_HD + 2);
    if (k0 >= k1) {
        if (tid < FA_BM && v0 + tid < nv) {
            const int v = v0 + tid;
            part[(((size_t)(v / CX_G) * CX_NKV + g) * nsplit + j) * vs + (v % CX_G) * (CX_HD + 2) + 1] = 0.f;
        }
        return;
    }
    const unsigned slot = rows[CX_MAXT];
    const unsigned short* kc = (const unsigned short*)ktab[slot];
    const unsigned short* vc = (const unsigned short*)vtab[slot];
#ifdef CX_KV8
    auto load = [&](int kb, int b) {
        for (int e = tid; e < FA_BN * (CX_HD / 16); e += 128) {
            const int p = e / (CX_HD / 16), d = (e % (CX_HD / 16)) * 16, key = kb + p;
            unsigned char* dk = s8 + ((2 * b) * FA_BN + p) * CX_HD + d;
            unsigned char* dv = s8 + ((2 * b + 1) * FA_BN + p) * CX_HD + d;
            if (key < k1) {
                fa_cp16((unsigned short*)dk, (const unsigned short*)(KV8_Q(kc, key, g) + d));
                fa_cp16((unsigned short*)dv, (const unsigned short*)(KV8_Q(vc, key, g) + d));
            } else {
                *(uint4*)dk = make_uint4(0, 0, 0, 0);
                *(uint4*)dv = make_uint4(0, 0, 0, 0);
            }
        }
    };
#else
    auto load = [&](int kb, int b) {
        for (int e = tid; e < FA_BN * (CX_HD / 8); e += 128) {
            const int p = e / (CX_HD / 8), d = (e % (CX_HD / 8)) * 8, key = kb + p;
            unsigned short* dk = sk + (b * FA_BN + p) * FA_LD + d;
            unsigned short* dv = sv + (b * FA_BN + p) * FA_LD + d;
            if (key < k1) {
                const size_t o = ((size_t)key * CX_NKV + g) * CX_HD + d;
                fa_cp16(dk, kc + o);
                fa_cp16(dv, vc + o);
            } else {
                // Zeros, not stale cache: a masked P is 0, and 0 * NaN would poison O.
                *(uint4*)dk = make_uint4(0, 0, 0, 0);
                *(uint4*)dv = make_uint4(0, 0, 0, 0);
            }
        }
    };
#endif
    load(k0, 0);
    cx_async_commit();
    // Q tile, pre-scaled, as bf16 (overlaps the first K/V copy).
    const float scale = 1.0f / sqrtf((float)CX_HD);
    for (int e = tid; e < FA_BM * (CX_HD / 4); e += 128) {
        const int v = e / (CX_HD / 4), d = (e % (CX_HD / 4)) * 4, gv = v0 + v;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gv < nv)
            x = *(const float4*)&q[((size_t)(gv / CX_G) * CX_NH + g * CX_G + gv % CX_G) * CX_HD + d];
        *(uint2*)&sq[v * FA_LD + d] = make_uint2(fa_pack(x.x * scale, x.y * scale), fa_pack(x.z * scale, x.w * scale));
    }
    // This thread's accumulator rows: vectors va (C rows gq) and vb (gq + 8) of the warp's 16.
    const int va = v0 + warp * 16 + gq, vb = va + 8;
    const int posa = va < nv ? (int)rows[va / CX_G] : -1, posb = vb < nv ? (int)rows[vb / CX_G] : -1;
    float o[CX_HD / 8][4];
    #pragma unroll
    for (int n = 0; n < CX_HD / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float ma = NEG_INF_F, mb = NEG_INF_F, la = 0.f, lb = 0.f;
    const int nt = (k1 - k0 + FA_BN - 1) / FA_BN;
    for (int it = 0; it < nt; it++) {
        const int b = it & 1, kbase = k0 + it * FA_BN;
        if (it + 1 < nt) load(kbase + FA_BN, b ^ 1);
        cx_async_commit();
        cx_async_wait<1>();
        __syncthreads();
#ifdef CX_KV8
        // The landed int8 K/V tile, code * scale -> fp16 (bf16 under FA_BF16) into the tile. Keys past
        // k1 are zero codes with scale 0.
        for (int e = tid; e < 2 * FA_BN * (CX_HD / 8); e += 128) {
            const int r = e / (CX_HD / 8), d = (e % (CX_HD / 8)) * 8, isv = r >= FA_BN, p = r - isv * FA_BN;
            const int key = kbase + p;
            const float sc = key < k1 ? KV8_S(isv ? vc : kc, key, g, d) : 0.f;
            const uint2 x = *(const uint2*)(s8 + ((2 * b + isv) * FA_BN + p) * CX_HD + d);
            auto h2 = [&](unsigned w, int i) {
                return fa_pack((float)(signed char)(w >> (8 * i)) * sc, (float)(signed char)(w >> (8 * i + 8)) * sc);
            };
            *(uint4*)((isv ? sv : sk) + (FA_KB(b) * FA_BN + p) * FA_LD + d) = make_uint4(h2(x.x, 0), h2(x.x, 2), h2(x.y, 0), h2(x.y, 2));
        }
        __syncthreads();
#elif !defined(FA_BF16)
        // The landed K/V tile, bf16 -> fp16 in place (once a block, not per warp fragment).
        for (int e = tid; e < 2 * FA_BN * (CX_HD / 8); e += 128) {
            const int r = e / (CX_HD / 8), d = (e % (CX_HD / 8)) * 8;
            uint4* pv = (uint4*)((r < FA_BN ? sk + (b * FA_BN + r) * FA_LD : sv + (b * FA_BN + r - FA_BN) * FA_LD) + d);
            uint4 x = *pv;
            x.x = fa_bf2h(x.x);
            x.y = fa_bf2h(x.y);
            x.z = fa_bf2h(x.z);
            x.w = fa_bf2h(x.w);
            *pv = x;
        }
        __syncthreads();
#endif
        // S = Q K^T (16 vectors × FA_BN keys a warp).
        float s[FA_BN / 8][4];
        #pragma unroll
        for (int n = 0; n < FA_BN / 8; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
        const unsigned short* qa = sq + (warp * 16 + gq) * FA_LD + 2 * t4;
        const unsigned short* kr = sk + (FA_KB(b) * FA_BN + gq) * FA_LD + 2 * t4;
        #pragma unroll
        for (int kk = 0; kk < CX_HD / 16; kk++) {
            const unsigned a0 = *(const unsigned*)(qa + kk * 16), a1 = *(const unsigned*)(qa + 8 * FA_LD + kk * 16);
            const unsigned a2 = *(const unsigned*)(qa + kk * 16 + 8),
                           a3 = *(const unsigned*)(qa + 8 * FA_LD + kk * 16 + 8);
            #pragma unroll
            for (int n = 0; n < FA_BN / 8; n++) {
                const unsigned short* kp = kr + n * 8 * FA_LD + kk * 16;
                fa_mma(s[n], a0, a1, a2, a3, *(const unsigned*)kp, *(const unsigned*)(kp + 8));
            }
        }
        // Causal + split mask, then the online softmax (each row is spread over a lane quad).
        float mxa = NEG_INF_F, mxb = NEG_INF_F;
        #pragma unroll
        for (int n = 0; n < FA_BN / 8; n++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int key = kbase + n * 8 + 2 * t4 + (e & 1);
                if (key >= k1 || key > (e < 2 ? posa : posb)) s[n][e] = NEG_INF_F;
                if (e < 2) mxa = fmaxf(mxa, s[n][e]);
                else mxb = fmaxf(mxb, s[n][e]);
            }
        for (int x = 1; x < 4; x <<= 1) {
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, x));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, x));
        }
        const float na = fmaxf(ma, mxa), nb = fmaxf(mb, mxb);
        // A row with no live key yet keeps m = -inf; subtract 0 instead so exp gives 0, not NaN.
        const float sa = na == NEG_INF_F ? 0.f : na, sb = nb == NEG_INF_F ? 0.f : nb;
        const float ca = expf(ma - sa), cb = expf(mb - sb);
        ma = na;
        mb = nb;
        float ra = 0.f, rb = 0.f;
        #pragma unroll
        for (int n = 0; n < FA_BN / 8; n++) {
            s[n][0] = expf(s[n][0] - sa);
            s[n][1] = expf(s[n][1] - sa);
            s[n][2] = expf(s[n][2] - sb);
            s[n][3] = expf(s[n][3] - sb);
            ra += s[n][0] + s[n][1];
            rb += s[n][2] + s[n][3];
        }
        la = la * ca + ra;
        lb = lb * cb + rb;
        #pragma unroll
        for (int n = 0; n < CX_HD / 8; n++) {
            o[n][0] *= ca;
            o[n][1] *= ca;
            o[n][2] *= cb;
            o[n][3] *= cb;
        }
        // O += P V: P's A fragments are the S accumulators of n-tiles 2kk and 2kk + 1; V's B
        // fragments come from ldmatrix.trans (V is key-major in smem).
        #pragma unroll
        for (int kk = 0; kk < FA_BN / 16; kk++) {
            const unsigned p0 = fa_pack(s[2 * kk][0], s[2 * kk][1]), p1 = fa_pack(s[2 * kk][2], s[2 * kk][3]);
            const unsigned p2 = fa_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                           p3 = fa_pack(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            const unsigned short* vr = sv + (FA_KB(b) * FA_BN + kk * 16 + (lane & 15)) * FA_LD + (lane >> 4) * 8;
            #pragma unroll
            for (int dn = 0; dn < CX_HD / 16; dn++) {
                unsigned r0, r1, r2, r3;
#if __CUDA_ARCH__ >= 800
                const unsigned addr = (unsigned)__cvta_generic_to_shared(vr + dn * 16);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                             : "r"(addr));
#else
                r0 = r1 = r2 = r3 = 0;
#endif
                fa_mma(o[2 * dn], p0, p1, p2, p3, r0, r1);
                fa_mma(o[2 * dn + 1], p0, p1, p2, p3, r2, r3);
            }
        }
        __syncthreads();
    }
    for (int x = 1; x < 4; x <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, x);
        lb += __shfl_xor_sync(0xffffffffu, lb, x);
    }
    // Partials in cx_attn_part's layout: part[row][g][split][head][m, l, acc[HD]].
    if (va < nv) {
        float* out = part + (((size_t)(va / CX_G) * CX_NKV + g) * nsplit + j) * vs + (va % CX_G) * (CX_HD + 2);
        if (t4 == 0) { out[0] = ma; out[1] = la; }
        #pragma unroll
        for (int n = 0; n < CX_HD / 8; n++) { out[2 + n * 8 + 2 * t4] = o[n][0]; out[3 + n * 8 + 2 * t4] = o[n][1]; }
    }
    if (vb < nv) {
        float* out = part + (((size_t)(vb / CX_G) * CX_NKV + g) * nsplit + j) * vs + (vb % CX_G) * (CX_HD + 2);
        if (t4 == 0) { out[0] = mb; out[1] = lb; }
        #pragma unroll
        for (int n = 0; n < CX_HD / 8; n++) { out[2 + n * 8 + 2 * t4] = o[n][2]; out[3 + n * 8 + 2 * t4] = o[n][3]; }
    }
}
"#;
