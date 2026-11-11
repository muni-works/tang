//! Compressed-KV decode kernels (CUDA), a straight port of
//! [`super::kvdec_msl::KVDEC_MSL`] so both backends share the table layout
//! (`cram_codec::kvgpu::Batch`). One thread per lane = one (stream, block,
//! head-group) arithmetic-coded substream. Integer only up to the final
//! `float(q) * unit`: with RoPE off and an f32 destination the output is
//! bit-identical to the CPU decoder.
//!
//! On this backend every u32 table rides in an f32 buffer (how
//! [`super::super::ComputeDevice::upload_u32`] stores it), so the kernel
//! reinterprets with `__float_as_uint` / `__uint_as_float` and takes `float*`
//! arguments. The launch prepends [`super::llm_cuda::KV_ROW_CUDA`] via
//! `llm_cuda::kv_source` so `kv_t` / `kv_st` match the cache's kind.

/// The MSL body with the access qualifiers swapped for CUDA.
pub const KVDEC_CUDA: &str = r#"
__device__ __forceinline__ unsigned int uw(const float* b, unsigned int i) {
    return __float_as_uint(b[i]);
}
// NVRTC (no headers, C mode) lacks clamp for unsigned and the half intrinsics.
__device__ __forceinline__ unsigned int uclamp(unsigned int v, unsigned int lo,
                                               unsigned int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__device__ __forceinline__ float h2f(unsigned short h) {
    const unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp = (h & 0x7c00u) >> 10, man = h & 0x3ffu;
    unsigned int b;
    if (exp == 0u) {
        b = sign;  // zero (denormals never occur in these scales)
    } else if (exp == 31u) {
        b = sign | 0x7f800000u | (man << 13);
    } else {
        b = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    return __uint_as_float(b);
}
__device__ __forceinline__ void sw(float* b, unsigned int i, unsigned int v) {
    b[i] = __uint_as_float(v);
}
// channel table row: mu, rho, class, dense class (i32 words as u32 bit patterns).
// The words ride in f32 buffers: read the FLOAT and reinterpret its bits — passing the
// uint to __float_as_int would convert the VALUE to float first (uint → float) and
// destroy the pattern.
__device__ __forceinline__ int4 ci(const float* b, unsigned int i) {
    return make_int4(__float_as_int(b[i * 4u]), __float_as_int(b[i * 4u + 1u]),
                     __float_as_int(b[i * 4u + 2u]), __float_as_int(b[i * 4u + 3u]));
}

__device__ __forceinline__ unsigned int tkv_rb(const unsigned int* s, unsigned int a,
                                               unsigned int end) {
    return a < end ? (s[a >> 2] >> ((a & 3u) * 8u)) & 0xffu : 0u;
}
struct TkvRC {
    unsigned int code, range, pos, end;
};
__device__ __forceinline__ bool tkv_dp(TkvRC& d, const float* S, unsigned int p1) {
    p1 = uclamp(p1, 32u, 65504u);
    unsigned int bound = (d.range >> 16) * (65536u - p1);
    bool bit = d.code >= bound;
    if (bit) {
        d.code -= bound;
        d.range -= bound;
    } else {
        d.range = bound;
    }
    while (d.range < (1u << 24)) {
        d.range <<= 8;
        d.code = (d.code << 8) | tkv_rb(reinterpret_cast<const unsigned int*>(S), d.pos, d.end);
        d.pos++;
    }
    return bit;
}
__device__ __forceinline__ bool tkv_dm(TkvRC& d, const float* s, float* m) {
    unsigned int w = uw(m, 0);
    unsigned int f = w & 0xffffu, sl = w >> 16;
    bool b = tkv_dp(d, s, (f + sl) >> 1);
    if (b) {
        f += (65535u - f) >> 4;
        sl += (65535u - sl) >> 7;
    } else {
        f -= f >> 4;
        sl -= sl >> 7;
    }
    sw(m, 0, f | (sl << 16));
    return b;
}
__device__ __forceinline__ int tkv_get(TkvRC& d, const float* s, float* m) {
    if (tkv_dm(d, s, m)) return 0;
    bool neg = tkv_dp(d, s, 32768u);
    unsigned int k = 0;
    bool more = true;
    while (k < 62u && more) {
        // the unary-model index stalls at 23 (like the MSL min(k, 23u))
        const unsigned int um = k < 23u ? k : 23u;
        more = tkv_dm(d, s, m + 1u + um);
        if (more) k++;
    }
    unsigned int w = 1u;
    for (int i = int(k) - 1; i >= 0; i--) {
        // the mantissa bit for i = k-1 comes from the mant model at k
        const unsigned int at = (unsigned int)i + 1u;
        const bool is_mant = at == k;
        const unsigned int mm = k < 23u ? k : 23u;
        const bool bit = is_mant ? tkv_dm(d, s, m + 25u + mm) : tkv_dp(d, s, 32768u);
        w = (w << 1) | (unsigned int)bit;
    }
    return neg ? -(int)w : (int)w;
}

// One thread per lane. Args: S stream words, L lanes (16 words each), C per channel
// (4 words), P prior [class][act][49], M model scratch, O the cache, CS/SN RoPE
// ([pos][hd/2] each; equal to P when no lane has RoPE), nl lanes.
extern "C" __global__ void tkv_decode(const float* S, const float* L, const float* C,
                                      const float* P, float* M, kv_t* O, const float* CS,
                                      const float* SN, unsigned int nl) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nl) return;
    const float* l = L + gid * 16u;
    // hd = lane width (channels); lane-local i maps to channel cbase + i (first half) or
    // fhd/2 + cbase + i - hd/2 (second half), so RoPE pairs stay inside one lane
    const unsigned int pos0 = uw(l, 2), n = uw(l, 3), ch = uw(l, 4);
    const unsigned int tstride = uw(l, 6), hd = uw(l, 11), cbase = uw(l, 14), fhd = uw(l, 15);
    const unsigned long long out = uw(l, 5);
    float* m = M + uw(l, 7);
    const float u = l[8];
    const unsigned int sink = uw(l, 9);
    const int sdiv = (int)uw(l, 10);
    const bool rope = (uw(l, 12) & 1u) != 0u;
    const bool store = (uw(l, 12) & 2u) == 0u;  // bit 1: decode only (benchmark bound)
    // dense class indices are numbered in order of first appearance: copy each prior once
    unsigned int seen = 0u;
    for (unsigned int c = 0; c < hd; c++) {
        const int4 q = ci(C, ch + c);
        if ((unsigned int)q.w != seen) continue;
        seen++;
        const unsigned int dc = (unsigned int)q.w * 147u, cl = (unsigned int)q.z * 147u;
        for (unsigned int k = 0; k < 147u; k++) sw(m, dc + k, uw(P, cl + k));
    }
    TkvRC d;
    d.pos = uw(l, 0);
    d.end = uw(l, 0) + uw(l, 1);
    d.range = 0xffffffffu;
    d.code = 0u;
    for (int i = 0; i < 4; i++) {
        d.code = (d.code << 8) | tkv_rb(reinterpret_cast<const unsigned int*>(S), d.pos, d.end);
        d.pos++;
    }
    int prev[128];
    unsigned int act[128];
    float row[128];
    for (unsigned int c = 0; c < hd; c++) act[c] = 0u;
    const unsigned int half_dim = hd / 2u, fhalf = fhd / 2u;
    for (unsigned int t = 0; t < n; t++) {
        const int ts = (pos0 + t < sink) ? 1 : sdiv;
        for (unsigned int c = 0; c < hd; c++) {
            const int4 q = ci(C, ch + c);
            const int mu = q.x;
            const int pred = t == 0u ? mu : mu + ((q.y * (prev[c] - mu)) >> 4);
            const int qq = tkv_get(d, S, m + ((unsigned int)q.w * 3u + act[c]) * 49u);
            const int r = pred + qq * ts;
            prev[c] = r;
            act[c] = min((unsigned int)abs(qq), 2u);
            row[c] = (float)r * u;
        }
        kv_t* dst = O + out + (unsigned long long)t * tstride;
        const unsigned int tc = (pos0 + t) * fhalf + cbase;
        for (unsigned int i = 0; i < half_dim; i++) {
            float x0 = row[i], x1 = row[i + half_dim];
            if (rope) {
                const float cs = CS[tc + i], sn = SN[tc + i];
                const float y0 = x0 * cs - x1 * sn;
                x1 = x1 * cs + x0 * sn;
                x0 = y0;
            }
            if (store) {
                kv_st(dst + cbase + i, x0);
                kv_st(dst + fhalf + cbase + i, x1);
            } else if (x0 == 1.0e30f && x1 == 1.0e30f) {
                kv_st(dst, x0);  // never true; keeps the decode live
            }
        }
    }
}

// KIVI-style dequant. params: [bits, g, nkv, hd, tokens, is_key, page shift].
// Codes in value order [t][h][c]; groups: K (t/g, h, c), V (t, h, c/g); SZ = (scale, zero).
extern "C" __global__ void kivi_dequant(const float* Q, const float* SZ, kv_t* O,
                                        const float* p, const float* T) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int bits = uw(p, 0), g = uw(p, 1), nkv = uw(p, 2), hd = uw(p, 3);
    const unsigned int ntok = uw(p, 4);
    const unsigned int kvd = nkv * hd;
    if (gid >= ntok * kvd) return;
    const unsigned int t = gid / kvd, hc = gid % kvd;
    const unsigned int b = gid * bits;
    const unsigned int code = (uw(Q, b >> 5) >> (b & 31u)) & ((1u << bits) - 1u);
    const unsigned int gi = uw(p, 5) != 0u ? (t / g) * kvd + hc : t * (kvd / g) + hc / g;
    const unsigned int hw = uw(SZ, gi);
    const float2 sz = make_float2(h2f((unsigned short)(hw & 0xffffu)), h2f((unsigned short)(hw >> 16)));
    const unsigned int* tu = reinterpret_cast<const unsigned int*>(T);
    kv_st(O + kv_row(tu, uw(p, 6), t) * kvd + hc, (float)code * sz.x + sz.y);
}
"#;
