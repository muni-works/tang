//! CUDA kernels for autoregressive LLM inference, mirroring `llm_msl`: decode GEMVs over bf16
//! and MLX 4-bit weights, weight dequantization for prefill GEMMs, embedding lookups, the fused
//! attention prologue, activations, LayerNorm, and split-KV / tiled attention with sliding
//! windows and bidirectional blocks. Activations are f32 throughout.

/// Prepends the helpers every kernel source below shares.
macro_rules! with_common {
    ($body:literal) => {
        concat!(
            r#"
__device__ __forceinline__ float bf(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
#define NEG_INF __uint_as_float(0xff800000u)
typedef unsigned long long u64;
"#,
            $body
        )
    };
}

/// `y[r, n] = sum_k x[r, k] * W[n, k]` for a handful of rows (`M <= 8`), W stored `[N, K]` in
/// bf16 (`gemv_bf16`) or MLX 4-bit groups (`gemv_q4`: packed `[N, K/8]` u32, low nibble first,
/// bf16 scales and biases `[N, K/group]`, `w = scale * q + bias`).
///
/// Decode is memory-bound: one warp per output column reads its weight row once, coalesced,
/// and applies it to all `M` rows. Launch: grid ceil(N / 8), block 256 (8 warps).
pub const GEMV_CUDA: &str = with_common!(
    r#"
// Nibble `e` of `q` as a float, exactly (float)((q >> 4e) & 0xf): the nibble goes into the
// mantissa of 2^23 and 2^23 comes off, a logic op and an add at full rate instead of a
// quarter-rate int-to-float convert.
__device__ __forceinline__ float nib(unsigned int q, int e) {
    return __int_as_float(((q >> (4 * e)) & 0xfu) | 0x4B000000u) - 8388608.0f;
}

extern "C" __global__ void gemv_bf16(
    const float* __restrict__ X, const unsigned short* __restrict__ W, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N)
{
    unsigned int lane = threadIdx.x & 31;
    unsigned int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= N) return;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned short* w = W + (u64)n * K;
    if ((K & 7) == 0) {
        const uint4* w8 = (const uint4*)w;
        for (unsigned int i = lane; i < K / 8; i += 32) {
            uint4 p = w8[i];
            float wv[8] = {bf(p.x & 0xffff), bf(p.x >> 16), bf(p.y & 0xffff), bf(p.y >> 16),
                           bf(p.z & 0xffff), bf(p.z >> 16), bf(p.w & 0xffff), bf(p.w >> 16)};
            #pragma unroll
            for (unsigned int r = 0; r < 8; r++) {
                if (r < M) {
                    const float4* x4 = (const float4*)(X + (u64)r * K + i * 8);
                    float4 a = x4[0], b = x4[1];
                    acc[r] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w
                            + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
    } else {
        for (unsigned int i = lane; i < K; i += 32) {
            float wv = bf(w[i]);
            #pragma unroll
            for (unsigned int r = 0; r < 8; r++) if (r < M) acc[r] += wv * X[(u64)r * K + i];
        }
    }
    #pragma unroll
    for (unsigned int r = 0; r < 8; r++) {
        if (r < M) {
            float s = warp_sum(acc[r]);
            if (lane == 0) Y[(u64)r * N + n] = s;
        }
    }
}

// Each lane takes one packed word (8 weights) per step; K % 8 == 0 and group % 8 == 0.
extern "C" __global__ void gemv_q4(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group)
{
    unsigned int lane = threadIdx.x & 31;
    unsigned int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= N) return;
    unsigned int words = K / 8, G = K / group;
    const unsigned int* w = Wq + (u64)n * words;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned int i = lane; i < words; i += 32) {
        unsigned int q = w[i];
        u64 g = (u64)n * G + (i * 8) / group;
        float s = bf(S[g]), b = bf(B[g]);
        float qf[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) qf[e] = nib(q, e);
        #pragma unroll
        for (unsigned int r = 0; r < 8; r++) {
            if (r < M) {
                const float4* x4 = (const float4*)(X + (u64)r * K + i * 8);
                float4 a = x4[0], c = x4[1];
                float d = qf[0] * a.x + qf[1] * a.y + qf[2] * a.z + qf[3] * a.w
                        + qf[4] * c.x + qf[5] * c.y + qf[6] * c.z + qf[7] * c.w;
                float xs = (a.x + a.y + a.z + a.w) + (c.x + c.y + c.z + c.w);
                acc[r] += s * d + b * xs;
            }
        }
    }
    #pragma unroll
    for (unsigned int r = 0; r < 8; r++) {
        if (r < M) {
            float v = warp_sum(acc[r]);
            if (lane == 0) Y[(u64)r * N + n] = v;
        }
    }
}

// 4-bit `linear` for 1..8 rows (decode, batched decode): a warp per CPW output columns, lane
// `l` taking words l, l + 32, ... of each. A word is dequantized once (s * q + b) for every row,
// and the rows' activations are staged in shared memory a chunk of XW words at a time (each
// word's first and second four values in separate planes: conflict-free float4 reads at a lane
// stride of 16 bytes). M is a template parameter, so the row loop has no branches. Each row's
// sum runs in the same order whatever M is: a row decoded in a batch gets exactly the bits it
// gets alone. A K split (gridDim.y > 1, `wsplit` words a split) writes its own [M, N] slice of
// Y for `sum_splits`; the split depends on the shape alone, never on M, so this holds split
// too. Launch: grid (ceil(N / (Q4_WARPS * CPW)), splits), block Q4_WARPS * 32.
#define XW 128
template <int CPW, int M, int WARPS>
__device__ __forceinline__ void gemv_q4_xs_body(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int group,
    unsigned int wsplit)
{
    __shared__ __align__(16) float Xs[2][8][XW * 4];
    unsigned int tid = threadIdx.x, lane = tid & 31;
    unsigned int n0 = (blockIdx.x * WARPS + (tid >> 5)) * CPW;
    bool live = n0 < N;
    unsigned int words = K / 8, G = K / group;
    unsigned int wb = blockIdx.y * wsplit, we = min(wb + wsplit, words);
    Y += (u64)blockIdx.y * M * N;
    float acc[CPW][M];
    #pragma unroll
    for (int c = 0; c < CPW; c++)
        #pragma unroll
        for (int r = 0; r < M; r++) acc[c][r] = 0.0f;
    for (unsigned int w0 = wb; w0 < we; w0 += XW) {
        unsigned int nw = min(XW, we - w0);
        __syncthreads();
        // Fill: thread t takes half-words (float4s) t, t + 256, ... of each row.
        for (int r = 0; r < M; r++) {
            for (unsigned int h = tid; h < 2 * nw; h += WARPS * 32) {
                float4 v = ((const float4*)(X + (u64)r * K + (u64)w0 * 8))[h];
                ((float4*)Xs[h & 1][r])[h >> 1] = v;
            }
        }
        __syncthreads();
        if (!live) continue;
        #pragma unroll 2
        for (unsigned int j = lane; j < nw; j += 32) {
            unsigned int i = w0 + j;
            unsigned int q[CPW];
            float s[CPW], b[CPW];
            #pragma unroll
            for (int c = 0; c < CPW; c++) {
                unsigned int n = min(n0 + c, N - 1);
                q[c] = Wq[(u64)n * words + i];
                u64 g = (u64)n * G + (i * 8) / group;
                s[c] = bf(S[g]); b[c] = bf(B[g]);
            }
            // Dequantize once a column (s * q + b), shared by every row.
            float wf[CPW][8];
            #pragma unroll
            for (int c = 0; c < CPW; c++)
                #pragma unroll
                for (int e = 0; e < 8; e++) wf[c][e] = fmaf(nib(q[c], e), s[c], b[c]);
            #pragma unroll
            for (int r = 0; r < M; r++) {
                float4 a = ((const float4*)Xs[0][r])[j], e = ((const float4*)Xs[1][r])[j];
                #pragma unroll
                for (int c = 0; c < CPW; c++) {
                    float t = acc[c][r];
                    t = fmaf(wf[c][0], a.x, t); t = fmaf(wf[c][1], a.y, t);
                    t = fmaf(wf[c][2], a.z, t); t = fmaf(wf[c][3], a.w, t);
                    t = fmaf(wf[c][4], e.x, t); t = fmaf(wf[c][5], e.y, t);
                    t = fmaf(wf[c][6], e.z, t); t = fmaf(wf[c][7], e.w, t);
                    acc[c][r] = t;
                }
            }

        }
    }
    if (!live) return;
    #pragma unroll
    for (int c = 0; c < CPW; c++) {
        #pragma unroll
        for (int r = 0; r < M; r++) {
            float v = warp_sum(acc[c][r]);
            if (lane == 0 && n0 + c < N) Y[(u64)r * N + n0 + c] = v;
        }
    }
}

// `gemv_q4_xs_body` with 128-bit weight loads, for K % 32 == 0 and group % 32 == 0 (every MLX
// model): per XW-word stage, lane `l` takes words 4l .. 4l + 3 of each of its CPW columns as one
// uint4 (32 weights, one scale group), issued before the stage's activations are staged so the
// loads overlap the fill. Activations are staged as plain rows of floats, float4 `i` at
// `i ^ ((i >> 3) & 7)` within its 8-float4 line, so a quarter-warp's eight float4 reads land on
// eight different banks. Same split and batch-invariance contract as `gemv_q4_xs_body`.
__device__ __forceinline__ unsigned int xv_swz(unsigned int i) { return i ^ ((i >> 3) & 7u); }
template <int CPW, int M, int WARPS>
__device__ __forceinline__ void gemv_q4_xv_body(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int group,
    unsigned int wsplit)
{
    __shared__ __align__(16) float4 Xs[M][XW * 2];
    unsigned int tid = threadIdx.x, lane = tid & 31;
    unsigned int n0 = (blockIdx.x * WARPS + (tid >> 5)) * CPW;
    bool live = n0 < N;
    unsigned int words = K / 8, G = K / group;
    unsigned int wb = blockIdx.y * wsplit, we = min(wb + wsplit, words);
    Y += (u64)blockIdx.y * M * N;
    float acc[CPW][M];
    #pragma unroll
    for (int c = 0; c < CPW; c++)
        #pragma unroll
        for (int r = 0; r < M; r++) acc[c][r] = 0.0f;
    for (unsigned int w0 = wb; w0 < we; w0 += XW) {
        unsigned int nw = min(XW, we - w0);
        unsigned int i = w0 + 4 * lane;        // this lane's first word
        bool mine = live && 4 * lane < nw;
        uint4 q[CPW];
        float s[CPW], b[CPW];
        if (mine) {
            #pragma unroll
            for (int c = 0; c < CPW; c++) {
                unsigned int n = min(n0 + c, N - 1);
                q[c] = *(const uint4*)(Wq + (u64)n * words + i);
                u64 g = (u64)n * G + (i * 8) / group;
                s[c] = bf(S[g]); b[c] = bf(B[g]);
            }
        }
        __syncthreads();
        for (int r = 0; r < M; r++) {
            const float4* xr = (const float4*)(X + (u64)r * K + (u64)w0 * 8);
            for (unsigned int h = tid; h < 2 * nw; h += WARPS * 32) Xs[r][xv_swz(h)] = xr[h];
        }
        __syncthreads();
        if (!mine) continue;
        #pragma unroll
        for (int c = 0; c < CPW; c++) {
            unsigned int qs[4] = {q[c].x, q[c].y, q[c].z, q[c].w};
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                float wf[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) wf[e] = fmaf(nib(qs[w], e), s[c], b[c]);
                unsigned int h = 8 * lane + 2 * w;
                #pragma unroll
                for (int r = 0; r < M; r++) {
                    float4 a = Xs[r][xv_swz(h)], e = Xs[r][xv_swz(h + 1)];
                    float t = acc[c][r];
                    t = fmaf(wf[0], a.x, t); t = fmaf(wf[1], a.y, t);
                    t = fmaf(wf[2], a.z, t); t = fmaf(wf[3], a.w, t);
                    t = fmaf(wf[4], e.x, t); t = fmaf(wf[5], e.y, t);
                    t = fmaf(wf[6], e.z, t); t = fmaf(wf[7], e.w, t);
                    acc[c][r] = t;
                }
            }
        }
    }
    if (!live) return;
    #pragma unroll
    for (int c = 0; c < CPW; c++) {
        #pragma unroll
        for (int r = 0; r < M; r++) {
            float v = warp_sum(acc[c][r]);
            if (lane == 0 && n0 + c < N) Y[(u64)r * N + n0 + c] = v;
        }
    }
}

#define Q4_WARPS 8
#define GEMV_Q4_XS(CPW, M)                                                                    \
extern "C" __global__ void __launch_bounds__(Q4_WARPS * 32) gemv_q4_xs##CPW##_m##M(           \
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,                         \
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,               \
    float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int group,                \
    unsigned int wsplit)                                                                      \
{ gemv_q4_xs_body<CPW, M, Q4_WARPS>(X, Wq, S, B, Y, K, N, group, wsplit); }
#define GEMV_Q4_XV(CPW, M)                                                                    \
extern "C" __global__ void __launch_bounds__(Q4_WARPS * 32) gemv_q4_xv##CPW##_m##M(           \
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,                         \
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,               \
    float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int group,                \
    unsigned int wsplit)                                                                      \
{ gemv_q4_xv_body<CPW, M, Q4_WARPS>(X, Wq, S, B, Y, K, N, group, wsplit); }
GEMV_Q4_XV(4, 1)
GEMV_Q4_XV(4, 2)
GEMV_Q4_XV(4, 3)
GEMV_Q4_XV(4, 4)
GEMV_Q4_XV(4, 5)
GEMV_Q4_XV(4, 6)
GEMV_Q4_XV(4, 7)
GEMV_Q4_XV(4, 8)
GEMV_Q4_XS(4, 1)
GEMV_Q4_XS(4, 2)
GEMV_Q4_XS(4, 3)
GEMV_Q4_XS(4, 4)
GEMV_Q4_XS(4, 5)
GEMV_Q4_XS(4, 6)
GEMV_Q4_XS(4, 7)
GEMV_Q4_XS(4, 8)

// A few rows (2..32: speculative-decoding verify, short incremental prefills), one output
// column per thread: each thread streams its column's weights (64 K per step, prefetched a
// step ahead with the step's activations: 128 K, 64 or 256 bytes per column), widens each weight once, and applies it to every row,
// whose activations come from shared memory as broadcasts (Xs[k][row]). Rows are padded to
// MR (8, 16 or 32). A K split (gridDim.y > 1) writes its own [M, N] slice of Y for
// `sum_splits`. Needs K % 128 == 0 (and group % 64 == 0 for 4-bit).
// Launch: grid (ceil(N / 256), splits), block 256.
template <int MR, bool IS_Q4, int KC>
__device__ __forceinline__ void gemm_cols_body(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    const unsigned short* __restrict__ Wb, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N, unsigned int group, unsigned int k_split)
{
    __shared__ __align__(16) float Xs[KC * MR];
    constexpr int NW = IS_Q4 ? KC / 32 : KC / 8;   // uint4 per column per step
    unsigned int tid = threadIdx.x, n = blockIdx.x * 256 + tid;
    unsigned int kb = blockIdx.y * k_split, ke = min(kb + k_split, K);
    Y += (u64)blockIdx.y * M * N;
    unsigned int nc = min(n, N - 1);
    unsigned int G = K / group;

    uint4 wr[NW];
    float sc[NW / 2 + 1], bi[NW / 2 + 1];
    auto fetch = [&](unsigned int k0) {
        #pragma unroll
        for (int u = 0; u < NW; u++)
            wr[u] = IS_Q4 ? *(const uint4*)(Wq + (u64)nc * (K / 8) + k0 / 8 + u * 4)
                          : *(const uint4*)(Wb + (u64)nc * K + k0 + u * 8);
        if (IS_Q4) {
            #pragma unroll
            for (int u = 0; u < NW / 2; u++) {
                u64 g = (u64)nc * G + (k0 + 64 * u) / group;
                sc[u] = bf(S[g]); bi[u] = bf(B[g]);
            }
        }
    };
    float acc[MR];
    #pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.0f;

    // X loader: element tid + 256 * i of the [MR, KC] chunk.
    constexpr int NX = KC * MR / 256;
    float xr[NX];
    auto fetch_x = [&](unsigned int k0) {
        #pragma unroll
        for (int i = 0; i < NX; i++) {
            unsigned int e = tid + 256 * i, r = e / KC, kk = e % KC;
            xr[i] = (r < M) ? X[(u64)r * K + k0 + kk] : 0.0f;
        }
    };
    if (kb < ke) { fetch(kb); fetch_x(kb); }
    for (unsigned int k0 = kb; k0 < ke; k0 += KC) {
        #pragma unroll
        for (int i = 0; i < NX; i++) {
            unsigned int e = tid + 256 * i, r = e / KC, kk = e % KC;
            Xs[kk * MR + r] = xr[i];
        }
        uint4 w4[NW];
        #pragma unroll
        for (int u = 0; u < NW; u++) w4[u] = wr[u];
        float s4[NW / 2 + 1], b4[NW / 2 + 1];
        #pragma unroll
        for (int u = 0; u < NW / 2 + 1; u++) { s4[u] = sc[u]; b4[u] = bi[u]; }
        __syncthreads();
        if (k0 + KC < ke) { fetch(k0 + KC); fetch_x(k0 + KC); }
        #pragma unroll
        for (int u = 0; u < NW; u++) {
            unsigned int wv[4] = {w4[u].x, w4[u].y, w4[u].z, w4[u].w};
            #pragma unroll
            for (int e = 0; e < (IS_Q4 ? 32 : 8); e++) {
                float w;
                if (IS_Q4) w = fmaf(nib(wv[e / 8], e % 8), s4[u / 2], b4[u / 2]);
                else w = bf((unsigned short)(wv[e / 2] >> (16 * (e % 2))));
                const float4* xr = (const float4*)(Xs + (u * (IS_Q4 ? 32 : 8) + e) * MR);
                #pragma unroll
                for (int r4 = 0; r4 < MR / 4; r4++) {
                    float4 x = xr[r4];
                    acc[4 * r4] += w * x.x;
                    acc[4 * r4 + 1] += w * x.y;
                    acc[4 * r4 + 2] += w * x.z;
                    acc[4 * r4 + 3] += w * x.w;
                }
            }
        }
        __syncthreads();
    }
    if (n < N) {
        #pragma unroll
        for (int r = 0; r < MR; r++)
            if (r < M) Y[(u64)r * N + n] = acc[r];
    }
}

#define GEMM_COLS(NAME, MR)                                                                   \
extern "C" __global__ void __launch_bounds__(256) NAME##_q4(                                  \
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,                         \
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,               \
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group, \
    unsigned int k_split)                                                                     \
{                                                                                             \
    gemm_cols_body<MR, true, 128>(X, Wq, S, B, nullptr, Y, M, K, N, group, k_split);               \
}                                                                                             \
extern "C" __global__ void __launch_bounds__(256) NAME##_bf16(                                \
    const float* __restrict__ X, const unsigned short* __restrict__ W, float* __restrict__ Y, \
    unsigned int M, unsigned int K, unsigned int N, unsigned int k_split)                     \
{                                                                                             \
    gemm_cols_body<MR, false, 128>(X, nullptr, nullptr, nullptr, W, Y, M, K, N, 64, k_split);      \
}

GEMM_COLS(gemm_cols8, 8)
GEMM_COLS(gemm_cols16, 16)
GEMM_COLS(gemm_cols32, 32)

// Small-batch GEMM (9..32 rows: verify steps, short incremental prefills) straight off packed
// weights: block (128 output columns, K split), 256 threads, 64-deep K chunks. The weight tile
// is widened into shared memory once per chunk and shared by every row, X is staged transposed;
// each thread owns 4 rows x 4 columns (rows (tid / 32) * 4 .., columns tid % 32 + 32 * i).
// The next chunk's global loads are issued before the current chunk's math. With a K split
// (gridDim.y > 1) each split writes its own `[M, N]` slice of Y for `sum_splits`.
// Needs M <= 32 and K % 64 == 0 (and group % 8 == 0 for 4-bit).
#define GS_BN 128
#define GS_BK 64
#define GS_XS 36
template <bool IS_Q4>
__device__ __forceinline__ void gemm_small_body(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    const unsigned short* __restrict__ Wb, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N, unsigned int group, unsigned int k_split)
{
    __shared__ __align__(16) float Ws[GS_BK * GS_BN];
    __shared__ __align__(16) float Xs[GS_BK * GS_XS];
    unsigned int tid = threadIdx.x;
    unsigned int n0 = blockIdx.x * GS_BN;
    unsigned int kb = blockIdx.y * k_split, ke = min(kb + k_split, K);
    Y += (u64)blockIdx.y * M * N;

    // Loader roles: weight column wc (two threads per column, 32 k each), X column xk.
    unsigned int wc = tid >> 1, part = tid & 1, wn = n0 + wc;
    unsigned int xk = tid & 63, xr = (tid >> 6) * 8;
    uint4 wreg[IS_Q4 ? 1 : 4];
    float xreg[8];

    auto fetch = [&](unsigned int k0) {
        if (IS_Q4) {
            wreg[0] = (wn < N) ? *(const uint4*)(Wq + (u64)wn * (K / 8) + k0 / 8 + part * 4)
                               : make_uint4(0, 0, 0, 0);
        } else {
            #pragma unroll
            for (int u = 0; u < (IS_Q4 ? 1 : 4); u++)
                wreg[u] = (wn < N) ? *(const uint4*)(Wb + (u64)wn * K + k0 + part * 32 + u * 8)
                                   : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++)
            xreg[i] = (xr + i < M) ? X[(u64)(xr + i) * K + k0 + xk] : 0.0f;
    };
    auto stage = [&](unsigned int k0) {
        if (IS_Q4) {
            unsigned int w4[4] = {wreg[0].x, wreg[0].y, wreg[0].z, wreg[0].w};
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                unsigned int kk = part * 32 + u * 8;
                float sc = 0.0f, bi = 0.0f;
                if (wn < N) {
                    u64 g = (u64)wn * (K / group) + (k0 + kk) / group;
                    sc = bf(S[g]); bi = bf(B[g]);
                }
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    Ws[(kk + e) * GS_BN + wc] = sc * (float)((w4[u] >> (4 * e)) & 0xf) + bi;
            }
        } else {
            #pragma unroll
            for (int u = 0; u < (IS_Q4 ? 1 : 4); u++) {
                unsigned int w4[4] = {wreg[u].x, wreg[u].y, wreg[u].z, wreg[u].w};
                unsigned int kk = part * 32 + u * 8;
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    Ws[(kk + 2 * e) * GS_BN + wc] = bf(w4[e] & 0xffff);
                    Ws[(kk + 2 * e + 1) * GS_BN + wc] = bf(w4[e] >> 16);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) Xs[xk * GS_XS + xr + i] = xreg[i];
    };

    unsigned int tc = tid & 31, tr = (tid >> 5) * 4;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    if (kb < ke) fetch(kb);
    for (unsigned int k0 = kb; k0 < ke; k0 += GS_BK) {
        stage(k0);
        __syncthreads();
        if (k0 + GS_BK < ke) fetch(k0 + GS_BK);
        #pragma unroll 8
        for (int kk = 0; kk < GS_BK; kk++) {
            float4 x = *(const float4*)(Xs + kk * GS_XS + tr);
            const float* w = Ws + kk * GS_BN + tc;
            float wv[4] = {w[0], w[32], w[64], w[96]};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[0][j] += x.x * wv[j];
                acc[1][j] += x.y * wv[j];
                acc[2][j] += x.z * wv[j];
                acc[3][j] += x.w * wv[j];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (tr + i >= M) break;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int n = n0 + tc + 32 * j;
            if (n < N) Y[(u64)(tr + i) * N + n] = acc[i][j];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gemm_small_q4(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group,
    unsigned int k_split)
{
    gemm_small_body<true>(X, Wq, S, B, nullptr, Y, M, K, N, group, k_split);
}

extern "C" __global__ void __launch_bounds__(256) gemm_small_bf16(
    const float* __restrict__ X, const unsigned short* __restrict__ W, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N, unsigned int k_split)
{
    gemm_small_body<false>(X, nullptr, nullptr, nullptr, W, Y, M, K, N, 8, k_split);
}

// Y[i] = sum over splits of P[s * n + i].
extern "C" __global__ void sum_splits(
    const float* __restrict__ P, float* __restrict__ Y, unsigned int n, unsigned int splits)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (unsigned int k = 0; k < splits; k++) s += P[(u64)k * n + i];
    Y[i] = s;
}
"#
);

/// Widen rows `row0 .. row0 + rows` of a `[N, K]` weight to f32 (`[rows, K]`), for prefill
/// GEMMs through cuBLAS. `dequant_bf16`: one thread per element; `dequant_q4`: one thread per
/// packed word. Launch: 256 threads, enough blocks to cover the range.
pub const DEQUANT_CUDA: &str = with_common!(
    r#"
extern "C" __global__ void dequant_bf16(
    const unsigned short* __restrict__ W, float* __restrict__ Y,
    unsigned int row0, unsigned int rows, unsigned int K)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * K) return;
    Y[i] = bf(W[(u64)row0 * K + i]);
}

extern "C" __global__ void dequant_q4(
    const unsigned int* __restrict__ Wq, const unsigned short* __restrict__ S,
    const unsigned short* __restrict__ B, float* __restrict__ Y,
    unsigned int row0, unsigned int rows, unsigned int K, unsigned int group)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;   // word within the chunk
    unsigned int words = K / 8;
    if (i >= (u64)rows * words) return;
    u64 r = (u64)row0 + i / words;
    unsigned int k = (unsigned int)(i % words) * 8;
    unsigned int q = Wq[r * words + k / 8];
    u64 g = r * (K / group) + k / group;
    float s = bf(S[g]), b = bf(B[g]);
    float* y = Y + i * 8;
    #pragma unroll
    for (int e = 0; e < 8; e++) y[e] = s * (float)((q >> (4 * e)) & 0xf) + b;
}
"#
);

/// Embedding lookups from bf16 and 4-bit tables into f32 rows. `ids` are u32 bit patterns.
/// Launch: one thread per output element.
pub const EMBED_CUDA: &str = with_common!(
    r#"
extern "C" __global__ void embedding_bf16(
    const unsigned short* __restrict__ W, const unsigned int* __restrict__ ids,
    float* __restrict__ Y, unsigned int seq_len, unsigned int dim)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)seq_len * dim) return;
    u64 s = gid / dim, d = gid % dim;
    Y[gid] = bf(W[(u64)ids[s] * dim + d]);
}

extern "C" __global__ void embedding_q4(
    const unsigned int* __restrict__ Wq, const unsigned short* __restrict__ S,
    const unsigned short* __restrict__ B, const unsigned int* __restrict__ ids,
    float* __restrict__ Y, unsigned int seq_len, unsigned int dim, unsigned int group)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)seq_len * dim) return;
    u64 s = gid / dim;
    unsigned int d = (unsigned int)(gid % dim);
    u64 row = ids[s];
    unsigned int q = (Wq[row * (dim / 8) + d / 8] >> (4 * (d % 8))) & 0xf;
    u64 g = row * (dim / group) + d / group;
    Y[gid] = bf(S[g]) * (float)q + bf(B[g]);
}
"#
);

/// How kernels that read or write a KV cache see its elements, prepended to their source (see
/// [`kv_source`]): `kv_t` is the stored type, `kv1`/`kv4` load one or four as float, `kv_st`
/// stores one (rounding to nearest even for bf16). Attention math stays f32 either way.
pub const KV_F32_CUDA: &str = r#"
#define KV_BF16 0
typedef float kv_t;
__device__ __forceinline__ float kv1(const kv_t* p) { return *p; }
__device__ __forceinline__ float4 kv4(const kv_t* p) { return *(const float4*)p; }
__device__ __forceinline__ void kv_st(kv_t* p, float x) { *p = x; }
"#;

/// [`KV_F32_CUDA`] for bfloat16 caches (raw bits in `unsigned short`).
pub const KV_BF16_CUDA: &str = r#"
#define KV_BF16 1
typedef unsigned short kv_t;
__device__ __forceinline__ float kv1(const kv_t* p) {
    return __uint_as_float(((unsigned int)*p) << 16);
}
__device__ __forceinline__ float4 kv4(const kv_t* p) {
    uint2 u = *(const uint2*)p;
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                       __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
__device__ __forceinline__ void kv_st(kv_t* p, float x) {
    unsigned int b = __float_as_uint(x);
    *p = (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
"#;

/// The row of a cache buffer that position `j` lives in: `j` itself (`sh == 0`), or through
/// the block table `T` of a paged cache (blocks of `1 << sh` positions).
pub const KV_ROW_CUDA: &str = r#"
__device__ __forceinline__ unsigned long long kv_row(const unsigned int* T, unsigned int sh,
                                                     unsigned int j) {
    return sh == 0 ? (unsigned long long)j
                   : ((unsigned long long)T[j >> sh] << sh) | (j & ((1u << sh) - 1u));
}
"#;

/// `src` (one of the KV-reading sources: [`ATTN_PREP_CUDA`], [`ATTENTION_CUDA`]) for f32 or
/// bf16 caches. Built once per variant.
pub fn kv_source(src: &'static str, bf16: bool) -> &'static str {
    use std::sync::Mutex;
    static BUILT: Mutex<Vec<(usize, bool, &'static str)>> = Mutex::new(Vec::new());
    let mut built = BUILT.lock().unwrap();
    let id = src.as_ptr() as usize;
    if let Some(&(_, _, s)) = built.iter().find(|(p, b, _)| *p == id && *b == bf16) {
        return s;
    }
    let head = if bf16 { KV_BF16_CUDA } else { KV_F32_CUDA };
    let s: &'static str = Box::leak(format!("{head}{KV_ROW_CUDA}{src}").into_boxed_str());
    built.push((id, bf16, s));
    s
}

/// The fused attention prologue, `attention_prep`: one warp per (head, token) over the q, k and
/// v heads of a fused QKV row (grid `(nh + 2*nkv, seq)`, block 32). q/k heads get the optional
/// per-head RMS norm and half-split RoPE at `pos + s`; q goes to `Qo`, k and v straight into
/// the caches (`kv_t`, see [`kv_source`]) at `pos + s`. Any even head dim up to 256.
pub const ATTN_PREP_CUDA: &str = with_common!(
    r#"
// Head `h` of row `s`, at position `p` of the cache whose block table is `T`.
__device__ __forceinline__ void prep_one(
    const float* __restrict__ QKV, const float* __restrict__ QN, const float* __restrict__ KN,
    const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Qo, kv_t* __restrict__ KC, kv_t* __restrict__ VC,
    unsigned int nh, unsigned int nkv, unsigned int hd, unsigned int p, float eps,
    unsigned int has_qn, unsigned int has_kn, const unsigned int* __restrict__ T,
    unsigned int sh, unsigned int h, unsigned int s, float* xs)
{
    unsigned int lane = threadIdx.x;
    unsigned int qd = nh * hd, kvd = nkv * hd, row = qd + 2 * kvd, half = hd / 2;
    const float* src = QKV + (u64)s * row + (u64)h * hd;  // heads are contiguous: q.., k.., v..

    if (h >= nh + nkv) {  // v head: straight into the cache
        kv_t* dst = VC + kv_row(T, sh, p) * kvd + (u64)(h - nh - nkv) * hd;
        for (unsigned int d = lane; d < hd; d += 32) kv_st(dst + d, src[d]);
        return;
    }
    bool is_q = h < nh;
    bool has_norm = is_q ? has_qn != 0 : has_kn != 0;
    float ss = 0.0f;
    for (unsigned int d = lane; d < hd; d += 32) { float v = src[d]; xs[d] = v; ss += v * v; }
    if (has_norm) {
        const float* w = is_q ? QN : KN;
        float r = rsqrtf(warp_sum(ss) / (float)hd + eps);
        for (unsigned int d = lane; d < hd; d += 32) xs[d] *= r * w[d];
    }
    __syncwarp();
    u64 t = (u64)p * half;
    if (is_q) {
        float* dst = Qo + (u64)s * qd + (u64)h * hd;
        for (unsigned int i = lane; i < half; i += 32) {
            float c = COS[t + i], sn = SIN[t + i];
            float x0 = xs[i], x1 = xs[i + half];
            dst[i] = x0 * c - x1 * sn;
            dst[i + half] = x1 * c + x0 * sn;
        }
        return;
    }
    kv_t* dst = KC + kv_row(T, sh, p) * kvd + (u64)(h - nh) * hd;
    for (unsigned int i = lane; i < half; i += 32) {
        float c = COS[t + i], sn = SIN[t + i];
        float x0 = xs[i], x1 = xs[i + half];
        kv_st(dst + i, x0 * c - x1 * sn);
        kv_st(dst + i + half, x1 * c + x0 * sn);
    }
}

extern "C" __global__ void attention_prep(
    const float* __restrict__ QKV, const float* __restrict__ QN, const float* __restrict__ KN,
    const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Qo, kv_t* __restrict__ KC, kv_t* __restrict__ VC,
    unsigned int nh, unsigned int nkv, unsigned int hd, unsigned int pos, float eps,
    unsigned int has_qn, unsigned int has_kn, const unsigned int* __restrict__ T,
    unsigned int sh)
{
    __shared__ float xs[256];
    prep_one(QKV, QN, KN, COS, SIN, Qo, KC, VC, nh, nkv, hd, pos + blockIdx.y, eps, has_qn,
             has_kn, T, sh, blockIdx.x, blockIdx.y, xs);
}

// Rows of different sequences (batched decode): row `s` is at position `RP[s]` of the cache
// whose block table starts at `T + RT[s]`.
extern "C" __global__ void attention_prep_rows(
    const float* __restrict__ QKV, const float* __restrict__ QN, const float* __restrict__ KN,
    const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Qo, kv_t* __restrict__ KC, kv_t* __restrict__ VC,
    unsigned int nh, unsigned int nkv, unsigned int hd, float eps,
    unsigned int has_qn, unsigned int has_kn, const unsigned int* __restrict__ T,
    unsigned int sh, const unsigned int* __restrict__ RP, const unsigned int* __restrict__ RT)
{
    __shared__ float xs[256];
    unsigned int s = blockIdx.y;
    prep_one(QKV, QN, KN, COS, SIN, Qo, KC, VC, nh, nkv, hd, RP[s], eps, has_qn, has_kn,
             T + RT[s], sh, blockIdx.x, s, xs);
}
"#
);

/// Decoder-layer glue: half-split RoPE, activations, residual add, LayerNorm.
pub const FUSED_CUDA: &str = with_common!(
    r#"

// Half-split RoPE: rotates (x[i], x[i + half]) by the angle for start_pos + s. One thread per
// (s, h, i < half).
extern "C" __global__ void rope_half(
    const float* __restrict__ X, const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Y, unsigned int seq_len, unsigned int n_heads, unsigned int head_dim,
    unsigned int start_pos)
{
    unsigned int half = head_dim / 2;
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)seq_len * n_heads * half) return;
    u64 i = gid % half, sh = gid / half, s = sh / n_heads;
    u64 base = sh * head_dim;
    u64 t = (start_pos + s) * half + i;
    float c = COS[t], sn = SIN[t];
    float x0 = X[base + i], x1 = X[base + i + half];
    Y[base + i] = x0 * c - x1 * sn;
    Y[base + i + half] = x1 * c + x0 * sn;
}

extern "C" __global__ void add_f32(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ Y, unsigned int n)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = A[i] + B[i];
}

// GELU, tanh approximation. The tanh argument is clamped as on Metal (it's ±1 well before ±10).
__device__ __forceinline__ float gelu(float g) {
    float t = tanhf(fminf(fmaxf(0.7978845608f * (g + 0.044715f * g * g * g), -10.0f), 10.0f));
    return 0.5f * g * (1.0f + t);
}

extern "C" __global__ void gelu_tanh(const float* __restrict__ X, float* __restrict__ Y, unsigned int n)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = gelu(X[i]);
}

// GELU, exact: 0.5 x (1 + erf(x / sqrt 2)).
extern "C" __global__ void gelu_erf(const float* __restrict__ X, float* __restrict__ Y, unsigned int n)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = 0.5f * X[i] * (1.0f + erff(X[i] * 0.70710678118654752f));
}

// Over a fused gate/up projection `[rows, 2*ff]`: gelu_tanh(gate) * up and silu(gate) * up.
extern "C" __global__ void geglu_split(
    const float* __restrict__ GU, float* __restrict__ Y, unsigned int rows, unsigned int ff)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)rows * ff) return;
    u64 r = gid / ff, i = gid % ff;
    Y[gid] = gelu(GU[r * 2 * ff + i]) * GU[r * 2 * ff + ff + i];
}

extern "C" __global__ void swiglu_split(
    const float* __restrict__ GU, float* __restrict__ Y, unsigned int rows, unsigned int ff)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)rows * ff) return;
    u64 r = gid / ff, i = gid % ff;
    float g = GU[r * 2 * ff + i];
    Y[gid] = g / (1.0f + expf(-g)) * GU[r * 2 * ff + ff + i];
}

// LayerNorm with weight and bias: one block of 256 threads per row.
extern "C" __global__ void layer_norm(
    const float* __restrict__ X, const float* __restrict__ W, const float* __restrict__ B,
    float* __restrict__ Y, unsigned int dim, float eps)
{
    __shared__ float red[8];
    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* x = X + (u64)blockIdx.x * dim;
    float s = 0.0f;
    for (unsigned int i = tid; i < dim; i += 256) s += x[i];
    s = warp_sum(s);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    float mean = 0.0f;
    for (int g = 0; g < 8; g++) mean += red[g];
    mean /= (float)dim;
    __syncthreads();
    float v = 0.0f;
    for (unsigned int i = tid; i < dim; i += 256) { float d = x[i] - mean; v += d * d; }
    v = warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float var = 0.0f;
    for (int g = 0; g < 8; g++) var += red[g];
    float inv = rsqrtf(var / (float)dim + eps);
    float* y = Y + (u64)blockIdx.x * dim;
    for (unsigned int i = tid; i < dim; i += 256) y[i] = (x[i] - mean) * inv * W[i] + B[i];
}
"#
);

/// Attention over a KV cache: q/out `[q_len, nh*D]`, K/V `[pos, nkv*D]` (GQA: `nh % nkv == 0`),
/// f32 or bf16 (`kv_t`, see [`kv_source`]; math in f32).
/// Query `qi` sits at position `cache_start + qi` and sees keys `0 ..= cache_start + qi`, or
/// with a sliding window (`window > 0`) only the last `window` of those; with `bidir` it sees
/// keys up to `cache_start + q_len` instead (image tokens, vision encoders). D at most 256.
///
/// Split-KV ("flash decoding"), for decode and short batches:
/// `attn_partial`: grid (head, split, qi), block 256 (8 warps). Each warp runs an online softmax
/// over its share of the split's keys (lanes hold D/32 dims; a score is one warp sum), then the
/// warps merge and write (max, sum, acc[D]) for the split. Splits start at key `base` (decode
/// skips keys no window can see).
/// `attn_combine`: grid (head, qi), block 32; rescales and sums the splits.
///
/// Tiled prefill, `attn_prefill`: grid (ceil(q_len / BQ), head), block 256. A block holds BQ
/// query rows (16, or 32 when D <= 128) in shared memory and walks 16-key tiles: S = Q·Kᵀ for
/// the tile, an online-softmax update per row, then O = diag(α)·O + P·V, so each K/V row is
/// read once per BQ queries.
pub const ATTENTION_CUDA: &str = with_common!(
    r#"
#define MAXV 8   // max D / 32
// Decode splits: one per ATTN_SPLIT_KEYS keys, at most ATTN_MAX_SPLITS (`decode_splits` in
// cuda/llm.rs, which must agree). 64 keys fills an H200 from a context of ~250 at 32 heads.
#define ATTN_SPLIT_KEYS 64
#define ATTN_MAX_SPLITS 128

// Query `qi`'s keys `j0..j1` (one split of them) for head `head`, into its split's slot of P.
__device__ __forceinline__ void attn_partial_one(
    const float* __restrict__ Q, const kv_t* __restrict__ K, const kv_t* __restrict__ V,
    float* __restrict__ P, unsigned int nh, unsigned int nkv, unsigned int D,
    unsigned int n_splits, const unsigned int* __restrict__ T, unsigned int sh,
    unsigned int head, unsigned int split, unsigned int qi, unsigned int j0, unsigned int j1)
{
    unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned int kvh = head / (nh / nkv);
    u64 kvd = (u64)nkv * D;
    unsigned int per = (D + 31) / 32;

    float q[MAXV], acc[MAXV];
    float scale = rsqrtf((float)D);
    const float* qp = Q + ((u64)qi * nh + head) * D;
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) {
        unsigned int d = v * 32 + lane;
        q[v] = (v < per && d < D) ? qp[d] * scale : 0.0f;
        acc[v] = 0.0f;
    }
    float m = NEG_INF, l = 0.0f;

    for (unsigned int j = j0 + warp; j < j1; j += 8) {
        const kv_t* kp = K + kv_row(T, sh, j) * kvd + (u64)kvh * D;
        float s = 0.0f;
        #pragma unroll
        for (unsigned int v = 0; v < MAXV; v++) {
            unsigned int d = v * 32 + lane;
            if (v < per && d < D) s += q[v] * kv1(kp + d);
        }
        s = warp_sum(s);
        float m2 = fmaxf(m, s);
        float c = expf(m - m2), p = expf(s - m2);
        l = l * c + p;
        const kv_t* vp = V + kv_row(T, sh, j) * kvd + (u64)kvh * D;
        #pragma unroll
        for (unsigned int v = 0; v < MAXV; v++) {
            unsigned int d = v * 32 + lane;
            if (v < per && d < D) acc[v] = acc[v] * c + p * kv1(vp + d);
        }
        m = m2;
    }

    // Merge the 8 warps.
    __shared__ float sm[8], sl[8];
    __shared__ float sacc[8 * 256];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) {
        unsigned int d = v * 32 + lane;
        if (v < per && d < D) sacc[warp * D + d] = acc[v];
    }
    __syncthreads();
    if (warp != 0) return;

    float M = NEG_INF;
    for (int g = 0; g < 8; g++) M = fmaxf(M, sm[g]);
    float L = 0.0f;
    float out[MAXV];
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) out[v] = 0.0f;
    for (int g = 0; g < 8; g++) {
        float c = (sm[g] == NEG_INF) ? 0.0f : expf(sm[g] - M);
        L += sl[g] * c;
        #pragma unroll
        for (unsigned int v = 0; v < MAXV; v++) {
            unsigned int d = v * 32 + lane;
            if (v < per && d < D) out[v] += sacc[g * D + d] * c;
        }
    }
    float* pp = P + (((u64)qi * nh + head) * n_splits + split) * (D + 2);
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) {
        unsigned int d = v * 32 + lane;
        if (v < per && d < D) pp[2 + d] = out[v];
    }
    if (lane == 0) { pp[0] = M; pp[1] = L; }
}

extern "C" __global__ void attn_partial(
    const float* __restrict__ Q, const kv_t* __restrict__ K, const kv_t* __restrict__ V,
    float* __restrict__ P,   // [q_len, nh, n_splits, D + 2]
    unsigned int cache_start, unsigned int q_len, unsigned int nh, unsigned int nkv,
    unsigned int D, unsigned int n_splits, unsigned int split_len, unsigned int window,
    unsigned int base, unsigned int bidir, const unsigned int* __restrict__ T, unsigned int sh)
{
    unsigned int split = blockIdx.y, qi = blockIdx.z;
    unsigned int qpos = cache_start + qi + 1;
    unsigned int attend = bidir ? cache_start + q_len : qpos;
    unsigned int lo = (window > 0 && qpos > window) ? qpos - window : 0;
    unsigned int s0 = base + split * split_len;
    unsigned int j0 = max(s0, lo);
    unsigned int j1 = min(s0 + split_len, attend);
    attn_partial_one(Q, K, V, P, nh, nkv, D, n_splits, T, sh, blockIdx.x, split, qi, j0, j1);
}

// One query each from several sequences (batched decode): query `qi` is at position `RP[qi]`
// of the cache whose block table starts at `T + RT[qi]`, and sees keys `0 ..= RP[qi]`. Each
// splits its keys as a decode of that sequence alone would; splits past its own count are
// empty (attn_combine skips them), so its output is that decode's to the bit. `n_splits`:
// the most any query has (P's stride).
extern "C" __global__ void attn_partial_rows(
    const float* __restrict__ Q, const kv_t* __restrict__ K, const kv_t* __restrict__ V,
    float* __restrict__ P, unsigned int nh, unsigned int nkv, unsigned int D,
    unsigned int n_splits, const unsigned int* __restrict__ T, unsigned int sh,
    const unsigned int* __restrict__ RP, const unsigned int* __restrict__ RT)
{
    unsigned int split = blockIdx.y, qi = blockIdx.z;
    unsigned int span = RP[qi] + 1;
    unsigned int mine = min(max((span + ATTN_SPLIT_KEYS - 1) / ATTN_SPLIT_KEYS, 1u),
                            (unsigned int)ATTN_MAX_SPLITS);
    unsigned int split_len = (span + mine - 1) / mine;
    unsigned int s0 = split * split_len;
    unsigned int j0 = split < mine ? s0 : 0;
    unsigned int j1 = split < mine ? min(s0 + split_len, span) : 0;
    attn_partial_one(Q, K, V, P, nh, nkv, D, n_splits, T + RT[qi], sh, blockIdx.x, split, qi,
                     j0, j1);
}

extern "C" __global__ void attn_combine(
    const float* __restrict__ P, float* __restrict__ O,
    unsigned int nh, unsigned int D, unsigned int n_splits)
{
    unsigned int head = blockIdx.x, qi = blockIdx.y;
    const float* base = P + ((u64)qi * nh + head) * n_splits * (D + 2);
    float M = NEG_INF;
    for (unsigned int s = 0; s < n_splits; s++) M = fmaxf(M, base[s * (D + 2)]);
    float L = 0.0f;
    for (unsigned int s = 0; s < n_splits; s++) {
        float m = base[s * (D + 2)];
        if (m != NEG_INF) L += base[s * (D + 2) + 1] * expf(m - M);
    }
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float o = 0.0f;
        for (unsigned int s = 0; s < n_splits; s++) {
            float m = base[s * (D + 2)];
            if (m != NEG_INF) o += base[s * (D + 2) + 2 + d] * expf(m - M);
        }
        O[((u64)qi * nh + head) * D + d] = o / L;
    }
}

#define BK 16

extern "C" __global__ void attn_prefill(
    const float* __restrict__ Q, const kv_t* __restrict__ K, const kv_t* __restrict__ V,
    float* __restrict__ O, unsigned int cache_start, unsigned int q_len, unsigned int nh,
    unsigned int nkv, unsigned int D, unsigned int window, unsigned int bidir, unsigned int BQ,
    const unsigned int* __restrict__ T, unsigned int sh)
{
    __shared__ float Qs[32 * 129];        // [BQ][D + 1]; BQ * (D + 1) <= 4128
    __shared__ float KVs[BK * 257];       // [BK][D + 1], K then V
    __shared__ float Ps[32 * BK];         // [BQ][BK] scores, then probabilities
    __shared__ float m_s[32], l_s[32], a_s[32];

    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.y, q0 = blockIdx.x * BQ;
    unsigned int kvh = head / (nh / nkv);
    u64 qs = (u64)nh * D, kvd = (u64)nkv * D;
    unsigned int ld = D + 1;
    unsigned int total = cache_start + q_len;
    float scale = rsqrtf((float)D);

    for (unsigned int i = tid; i < BQ * D; i += 256) {
        unsigned int r = i / D, d = i % D;
        Qs[r * ld + d] = (q0 + r < q_len) ? Q[(u64)(q0 + r) * qs + (u64)head * D + d] * scale : 0.0f;
    }
    if (tid < BQ) { m_s[tid] = NEG_INF; l_s[tid] = 0.0f; }

    // Keys any row of this block can see.
    unsigned int r_last = min(q0 + BQ, q_len) - 1;
    unsigned int k_end = bidir ? total : cache_start + r_last + 1;
    unsigned int first = cache_start + q0 + 1;
    unsigned int k_begin = (window > 0 && first > window) ? first - window : 0;

    // Accumulator ownership: row tid / tpr, dims tid % tpr + tpr * i.
    unsigned int tpr = 256 / BQ;
    unsigned int ar = tid / tpr, ad = tid % tpr;
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    unsigned int per_score = BQ * BK / 256;
    __syncthreads();

    for (unsigned int j0 = k_begin; j0 < k_end; j0 += BK) {
        for (unsigned int i = tid; i < BK * D; i += 256) {
            unsigned int c = i / D, d = i % D, j = j0 + c;
            KVs[c * ld + d] = (j < total) ? kv1(K + kv_row(T, sh, j) * kvd + (u64)kvh * D + d) : 0.0f;
        }
        __syncthreads();
        for (unsigned int t = 0; t < per_score; t++) {
            unsigned int e = tid + 256 * t;
            unsigned int r = e / BK, c = e % BK;
            unsigned int qi = q0 + r, j = j0 + c;
            unsigned int qpos = cache_start + qi + 1;
            unsigned int attend = bidir ? total : qpos;
            unsigned int lo = (window > 0 && qpos > window) ? qpos - window : 0;
            float s = NEG_INF;
            if (qi < q_len && j < attend && j >= lo) {
                s = 0.0f;
                const float* qr = Qs + r * ld;
                const float* kr = KVs + c * ld;
                for (unsigned int d = 0; d < D; d++) s += qr[d] * kr[d];
            }
            Ps[r * BK + c] = s;
        }
        __syncthreads();
        if (tid < BQ) {
            float* pr = Ps + tid * BK;
            float m = m_s[tid], bm = NEG_INF;
            for (int c = 0; c < BK; c++) bm = fmaxf(bm, pr[c]);
            float mn = fmaxf(m, bm);
            float alpha = (m == NEG_INF) ? 0.0f : expf(m - mn);
            float ps = 0.0f;
            for (int c = 0; c < BK; c++) {
                float p = (pr[c] == NEG_INF) ? 0.0f : expf(pr[c] - mn);
                pr[c] = p;
                ps += p;
            }
            l_s[tid] = l_s[tid] * alpha + ps;
            m_s[tid] = mn;
            a_s[tid] = alpha;
        }
        for (unsigned int i = tid; i < BK * D; i += 256) {
            unsigned int c = i / D, d = i % D, j = j0 + c;
            // Overwrites K: every score read of K finished before the barrier above.
            KVs[c * ld + d] = (j < total) ? kv1(V + kv_row(T, sh, j) * kvd + (u64)kvh * D + d) : 0.0f;
        }
        __syncthreads();
        float alpha = a_s[ar];
        const float* pr = Ps + ar * BK;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned int d = ad + tpr * i;
            if (d < D) {
                float o = acc[i] * alpha;
                for (int c = 0; c < BK; c++) o += pr[c] * KVs[c * ld + d];
                acc[i] = o;
            }
        }
        __syncthreads();
    }

    unsigned int qi = q0 + ar;
    if (qi >= q_len) return;
    float l = l_s[ar];
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    float* op = O + (u64)qi * qs + (u64)head * D;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned int d = ad + tpr * i;
        if (d < D) op[d] = acc[i] * inv;
    }
}

#undef BK

// Split-KV attention for a few queries (speculative-decoding verify, short incremental
// prefills): block (kv head, split, row group). A block's rows are (query, head) pairs that
// share one KV head (row f -> query f / gqa, head kvh * gqa + f % gqa), so each K/V row of the
// split is read once for every query and every head of the GQA group. Keys come in tiles of
// BK, row-major in shared memory (stride D + 4 keeps float4 accesses conflict-free), with the
// next tile's K and V prefetched into registers while the current one is used: scores with
// one key per lane, an online softmax per row in registers (warp w owns rows w*RW ..), then
// P·V with lanes over head dims. Writes unnormalized partials in `attn_partial`'s layout for
// `attn_combine`. Needs D % 4 == 0.
template <int R, int BK, int DMAX>
__device__ __forceinline__ void attn_multi_body(
    const float* __restrict__ Q, const kv_t* __restrict__ K, const kv_t* __restrict__ V,
    float* __restrict__ P, unsigned int cache_start, unsigned int q_len, unsigned int nh,
    unsigned int nkv, unsigned int D, unsigned int n_splits, unsigned int split_len,
    unsigned int window, unsigned int base, unsigned int bidir,
    const unsigned int* __restrict__ T, unsigned int sh, float* Qs, float* KVs, float* Ps)
{
    constexpr int RW = R / 8;              // rows per warp
    constexpr int SEGS = 32 / BK;          // key segments per warp
    constexpr int RPT = RW / SEGS;         // rows per lane in the score phase
    constexpr int NV = DMAX / 32;          // head dims per lane in the PV phase
    constexpr int PS = R + 4;              // Ps row stride
    constexpr int NL = BK * DMAX / 4 / 256;  // float4 loads per thread per tile

    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned int kvh = blockIdx.x, split = blockIdx.y, f0 = blockIdx.z * R;
    unsigned int gqa = nh / nkv, rows = q_len * gqa;
    unsigned int LD = D + 4, D4 = D / 4;
    u64 kvd = (u64)nkv * D;
    unsigned int total = cache_start + q_len;
    float scale = rsqrtf((float)D);

    for (unsigned int i = tid; i < R * D4; i += 256) {
        unsigned int r = i / D4, d = (i % D4) * 4, f = f0 + r;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (f < rows) {
            unsigned int qi = f / gqa, head = kvh * gqa + f % gqa;
            v = *(const float4*)(Q + ((u64)qi * nh + head) * D + d);
            v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        }
        *(float4*)(Qs + r * LD + d) = v;
    }

    // Keys any row of the block can see, clipped to this split.
    unsigned int f_last = min(f0 + R, rows) - 1;
    unsigned int first = cache_start + f0 / gqa + 1;
    unsigned int lo_blk = (window > 0 && first > window) ? first - window : 0;
    unsigned int hi_blk = bidir ? total : cache_start + f_last / gqa + 1;
    unsigned int s0 = base + split * split_len;
    unsigned int jb = max(s0, lo_blk), je = min(s0 + split_len, hi_blk);

    unsigned int c = lane % BK, seg = lane / BK;
    unsigned int r0 = warp * RW + seg * RPT;
    unsigned int lo[RPT], hi[RPT];
    float m[RPT], l[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; i++) {
        unsigned int f = f0 + r0 + i;
        lo[i] = 0; hi[i] = 0;
        if (f < rows) {
            unsigned int qpos = cache_start + f / gqa + 1;
            hi[i] = bidir ? total : qpos;
            lo[i] = (window > 0 && qpos > window) ? qpos - window : 0;
        }
        m[i] = NEG_INF; l[i] = 0.0f;
    }
    float acc[RW][NV];
    #pragma unroll
    for (int i = 0; i < RW; i++)
        #pragma unroll
        for (int v = 0; v < NV; v++) acc[i][v] = 0.0f;

    float4 kreg[NL], vreg[NL];
    auto fetch = [&](float4* reg, const kv_t* src, unsigned int j0) {
        #pragma unroll
        for (int t = 0; t < NL; t++) {
            unsigned int i = tid + 256 * t, cc = i / D4, d = (i % D4) * 4, j = j0 + cc;
            reg[t] = (cc < BK && j < je) ? kv4(src + kv_row(T, sh, j) * kvd + (u64)kvh * D + d)
                                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto stage = [&](const float4* reg) {
        #pragma unroll
        for (int t = 0; t < NL; t++) {
            unsigned int i = tid + 256 * t, cc = i / D4, d = (i % D4) * 4;
            if (cc < BK) *(float4*)(KVs + cc * LD + d) = reg[t];
        }
    };
    if (jb < je) { fetch(kreg, K, jb); fetch(vreg, V, jb); }
    __syncthreads();

    for (unsigned int j0 = jb; j0 < je; j0 += BK) {
        bool more = j0 + BK < je;
        stage(kreg);
        __syncthreads();
        if (more) fetch(kreg, K, j0 + BK);

        float s[RPT];
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] = 0.0f;
        const float* kr = KVs + c * LD;
        for (unsigned int d = 0; d < D; d += 4) {
            float4 k4 = *(const float4*)(kr + d);
            #pragma unroll
            for (int i = 0; i < RPT; i++) {
                float4 q4 = *(const float4*)(Qs + (r0 + i) * LD + d);
                s[i] += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
        }
        unsigned int j = j0 + c;
        float alpha[RPT];
        #pragma unroll
        for (int i = 0; i < RPT; i++) {
            if (!(j < je && j >= lo[i] && j < hi[i])) s[i] = NEG_INF;
            float mx = s[i];
            #pragma unroll
            for (int o = BK / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            float mn = fmaxf(m[i], mx);
            float p = (s[i] == NEG_INF) ? 0.0f : __expf(s[i] - mn);
            alpha[i] = (m[i] == NEG_INF) ? 0.0f : __expf(m[i] - mn);
            float ps = p;
            #pragma unroll
            for (int o = BK / 2; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
            l[i] = l[i] * alpha[i] + ps;
            m[i] = mn;
            Ps[c * PS + r0 + i] = p;
        }
        __syncthreads();   // every warp is done with K
        stage(vreg);
        __syncthreads();
        if (more) fetch(vreg, V, j0 + BK);

        #pragma unroll
        for (int i = 0; i < RW; i++) {
            float a = __shfl_sync(0xffffffffu, alpha[i % RPT], (i / RPT) * BK);
            #pragma unroll
            for (int v = 0; v < NV; v++) acc[i][v] *= a;
        }
        const float* pw = Ps + warp * RW;
        #pragma unroll 4
        for (int cc = 0; cc < BK; cc++) {
            float p[RW];
            #pragma unroll
            for (int i = 0; i < RW; i++) p[i] = pw[cc * PS + i];
            const float* vr = KVs + cc * LD;
            #pragma unroll
            for (int v = 0; v < NV; v++) {
                unsigned int d = v * 32 + lane;
                float x = (d < D) ? vr[d] : 0.0f;
                #pragma unroll
                for (int i = 0; i < RW; i++) acc[i][v] += p[i] * x;
            }
        }
        __syncthreads();   // before the next tile overwrites V and Ps
    }

    #pragma unroll
    for (int i = 0; i < RW; i++) {
        unsigned int src = (i / RPT) * BK;
        float mi = __shfl_sync(0xffffffffu, m[i % RPT], src);
        float li = __shfl_sync(0xffffffffu, l[i % RPT], src);
        unsigned int f = f0 + warp * RW + i;
        if (f >= rows) continue;
        unsigned int qi = f / gqa, head = kvh * gqa + f % gqa;
        float* pp = P + (((u64)qi * nh + head) * n_splits + split) * (D + 2);
        #pragma unroll
        for (int v = 0; v < NV; v++) {
            unsigned int d = v * 32 + lane;
            if (d < D) pp[2 + d] = acc[i][v];
        }
        if (lane == 0) { pp[0] = mi; pp[1] = li; }
    }
}

#define ATTN_MULTI(NAME, R, BK, DMAX)                                                         \
extern "C" __global__ void __launch_bounds__(256) NAME(                                      \
    const float* __restrict__ Q, const kv_t* __restrict__ K, const kv_t* __restrict__ V,      \
    float* __restrict__ P, unsigned int cache_start, unsigned int q_len, unsigned int nh,     \
    unsigned int nkv, unsigned int D, unsigned int n_splits, unsigned int split_len,          \
    unsigned int window, unsigned int base, unsigned int bidir,                               \
    const unsigned int* __restrict__ T, unsigned int sh)                                      \
{                                                                                             \
    __shared__ __align__(16) float Qs[R * (DMAX + 4)];                                        \
    __shared__ __align__(16) float KVs[BK * (DMAX + 4)];                                      \
    __shared__ __align__(16) float Ps[BK * (R + 4)];                                          \
    attn_multi_body<R, BK, DMAX>(Q, K, V, P, cache_start, q_len, nh, nkv, D, n_splits,        \
                                 split_len, window, base, bidir, T, sh, Qs, KVs, Ps);         \
}

ATTN_MULTI(attn_multi_r8, 8, 32, 128)
ATTN_MULTI(attn_multi_r16, 16, 32, 128)
ATTN_MULTI(attn_multi_r32, 32, 32, 128)
ATTN_MULTI(attn_multi_d256, 16, 16, 256)
"#
);

/// `gemm_tc_q4_m{16,32}`: `y[r, n] = sum_k x[r, k] * W[n, k]` for 9..=32 rows off MLX 4-bit
/// weights on tensor cores (TF32 `mma.sync.m16n8k8`, f32 accumulate; sm_80+). A block (8
/// warps) takes 128 output columns and a K range: each 32-wide K stage it widens its columns'
/// weights into shared memory (s * q + b) and stages the rows' activations, and each warp runs
/// the 16- or 32-row by 16-column tile. Rows past M are zero. A row's sums go in an order fixed
/// by K and the split alone (never by M), and a K split (gridDim.y > 1) writes its own [M, N]
/// slice for `sum_splits`. Needs K % 32 == 0, group % 32 == 0, k_split % 32 == 0.
/// Launch: grid (ceil(N / 128), splits), block 256.
pub const TC_GEMM_CUDA: &str = with_common!(
    r#"
__device__ __forceinline__ float tc_nib(unsigned int q, int e) {
    return __int_as_float(((q >> (4 * e)) & 0xfu) | 0x4B000000u) - 8388608.0f;
}
__device__ __forceinline__ unsigned int tf32(float f) {
    unsigned int r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(f));
    return r;
}
template <int MT>
__device__ __forceinline__ void gemm_tc_q4_body(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group,
    unsigned int k_split)
{
    __shared__ float Ws[128][36];
    __shared__ float Xs[MT * 16][36];
    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    unsigned int n0 = blockIdx.x * 128;
    unsigned int kb = blockIdx.y * k_split, ke = min(kb + k_split, K);
    Y += (u64)blockIdx.y * M * N;
    unsigned int words = K / 8, G = K / group;
    float acc[MT][2][4];
    #pragma unroll
    for (int a = 0; a < MT; a++)
        #pragma unroll
        for (int b = 0; b < 2; b++)
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.0f;
    for (unsigned int k0 = kb; k0 < ke; k0 += 32) {
        __syncthreads();
        // 128 columns x 4 words: two words a thread, widened to f32.
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int w = tid + h * 256, col = w >> 2, wi = w & 3, n = n0 + col;
            float* dst = &Ws[col][wi * 8];
            if (n < N) {
                unsigned int q = Wq[(u64)n * words + k0 / 8 + wi];
                u64 gi = (u64)n * G + (k0 + wi * 8) / group;
                float s = bf(S[gi]), b = bf(B[gi]);
                #pragma unroll
                for (int e = 0; e < 8; e++) dst[e] = fmaf(tc_nib(q, e), s, b);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) dst[e] = 0.0f;
            }
        }
        for (unsigned int i = tid; i < MT * 16 * 32; i += 256) {
            unsigned int r = i >> 5, kk = i & 31;
            Xs[r][kk] = r < M ? X[(u64)r * K + k0 + kk] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < 32; ks += 8) {
            unsigned int a[MT][4], bb[2][2];
            #pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                a[mt][0] = tf32(Xs[mt * 16 + g][ks + t]);
                a[mt][1] = tf32(Xs[mt * 16 + g + 8][ks + t]);
                a[mt][2] = tf32(Xs[mt * 16 + g][ks + t + 4]);
                a[mt][3] = tf32(Xs[mt * 16 + g + 8][ks + t + 4]);
            }
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                unsigned int col = warp * 16 + nt * 8 + g;
                bb[nt][0] = tf32(Ws[col][ks + t]);
                bb[nt][1] = tf32(Ws[col][ks + t + 4]);
            }
            #pragma unroll
            for (int mt = 0; mt < MT; mt++)
                #pragma unroll
                for (int nt = 0; nt < 2; nt++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]),
                          "+f"(acc[mt][nt][3])
                        : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]),
                          "r"(bb[nt][0]), "r"(bb[nt][1]));
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
        #pragma unroll
        for (int nt = 0; nt < 2; nt++)
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                unsigned int r = mt * 16 + g + (c >> 1) * 8;
                unsigned int n = n0 + warp * 16 + nt * 8 + 2 * t + (c & 1);
                if (r < M && n < N) Y[(u64)r * N + n] = acc[mt][nt][c];
            }
}
extern "C" __global__ void __launch_bounds__(256) gemm_tc_q4_m16(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group,
    unsigned int k_split)
{ gemm_tc_q4_body<1>(X, Wq, S, B, Y, M, K, N, group, k_split); }
extern "C" __global__ void __launch_bounds__(256) gemm_tc_q4_m32(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group,
    unsigned int k_split)
{ gemm_tc_q4_body<2>(X, Wq, S, B, Y, M, K, N, group, k_split); }
"#
);
/// `gemm_f16_q4`: `y[r, n] = sum_k x[r, k] * W[n, k]` for prefill-sized M off MLX 4-bit
/// weights on f16 tensor cores (`mma.sync.m16n8k16`, f32 accumulate; sm_80+). A block (8
/// warps, 2 x 4) takes a 128 x 128 output tile, each warp 64 x 32. Per 32-wide K chunk the
/// rows' activations are rounded to f16 and the columns' weights widened (s * q + b, in f32,
/// then rounded) into padded shared tiles (row stride 40 halves: conflict-free `ldmatrix`),
/// double-buffered, the next chunk's global loads in registers while this chunk's mma issue.
/// No dequantized weight ever reaches global memory. A row's sums go in K order alone (never
/// by M). Needs K % 32 == 0, group % 32 == 0. Launch: grid (ceil(N / 128), ceil(M / 128)),
/// block 256.
pub const F16_GEMM_CUDA: &str = with_common!(
    r#"
#define FG_LD 40
__device__ __forceinline__ unsigned int fg_h2(float lo, float hi) {
    unsigned int d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
__device__ __forceinline__ float fg_nib(unsigned int q, int e) {
    return __int_as_float(((q >> (4 * e)) & 0xfu) | 0x4B000000u) - 8388608.0f;
}
__device__ __forceinline__ void fg_ldsm4(unsigned int* r, const unsigned short* p) {
    unsigned int a = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
extern "C" __global__ void __launch_bounds__(256) gemm_f16_q4(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group)
{
    __shared__ __align__(16) unsigned short As[2][128 * FG_LD];
    __shared__ __align__(16) unsigned short Ws[2][128 * FG_LD];
    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned int wm = warp >> 2, wn = warp & 3;
    unsigned int n0 = blockIdx.x * 128, m0 = blockIdx.y * 128;
    unsigned int words = K / 8, G = K / group;
    // staging assignments: A 128 rows x 8 float4 (4 a thread), W 128 cols x 4 words (2 a thread)
    float4 xa[4];
    unsigned int wq[2];
    float ws[2], wb[2];
    auto load = [&](unsigned int k0) {
        #pragma unroll
        for (int h = 0; h < 4; h++) {
            unsigned int i = tid + h * 256, r = i >> 3, c4 = i & 7, row = m0 + r;
            xa[h] = row < M ? *(const float4*)&X[(u64)row * K + k0 + c4 * 4]
                            : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int i = tid + h * 256, col = i >> 2, wi = i & 3, n = n0 + col;
            if (n < N) {
                wq[h] = Wq[(u64)n * words + k0 / 8 + wi];
                u64 gi = (u64)n * G + (k0 + wi * 8) / group;
                ws[h] = bf(S[gi]);
                wb[h] = bf(B[gi]);
            } else {
                wq[h] = 0; ws[h] = 0.f; wb[h] = 0.f;
            }
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int h = 0; h < 4; h++) {
            unsigned int i = tid + h * 256, r = i >> 3, c4 = i & 7;
            uint2 v = make_uint2(fg_h2(xa[h].x, xa[h].y), fg_h2(xa[h].z, xa[h].w));
            *(uint2*)&As[buf][r * FG_LD + c4 * 4] = v;
        }
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int i = tid + h * 256, col = i >> 2, wi = i & 3;
            float s = ws[h], b = wb[h];
            unsigned int q = wq[h];
            uint4 v;
            v.x = fg_h2(fmaf(fg_nib(q, 0), s, b), fmaf(fg_nib(q, 1), s, b));
            v.y = fg_h2(fmaf(fg_nib(q, 2), s, b), fmaf(fg_nib(q, 3), s, b));
            v.z = fg_h2(fmaf(fg_nib(q, 4), s, b), fmaf(fg_nib(q, 5), s, b));
            v.w = fg_h2(fmaf(fg_nib(q, 6), s, b), fmaf(fg_nib(q, 7), s, b));
            *(uint4*)&Ws[buf][col * FG_LD + wi * 8] = v;
        }
    };
    float acc[4][4][4];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int b = 0; b < 4; b++)
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.0f;
    load(0);
    store(0);
    __syncthreads();
    unsigned int chunks = K / 32;
    // ldmatrix lane addresses: A rows lane%16, k half lane/16; B cols (lane&7)+8*(lane>>4),
    // k half (lane>>3)&1
    unsigned int ar = wm * 64 + (lane & 15), ak = (lane >> 4) * 8;
    unsigned int bc = wn * 32 + (lane & 7) + ((lane >> 4) << 3), bk = ((lane >> 3) & 1) * 8;
    for (unsigned int c = 0; c < chunks; c++) {
        int buf = c & 1;
        if (c + 1 < chunks) load((c + 1) * 32);
        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            unsigned int a[4][4], b[4][2];
            #pragma unroll
            for (int mt = 0; mt < 4; mt++)
                fg_ldsm4(a[mt], &As[buf][(ar + mt * 16) * FG_LD + ks + ak]);
            #pragma unroll
            for (int np = 0; np < 2; np++) {
                unsigned int r[4];
                fg_ldsm4(r, &Ws[buf][(bc + np * 16) * FG_LD + ks + bk]);
                b[np * 2][0] = r[0]; b[np * 2][1] = r[1];
                b[np * 2 + 1][0] = r[2]; b[np * 2 + 1][1] = r[3];
            }
            #pragma unroll
            for (int mt = 0; mt < 4; mt++)
                #pragma unroll
                for (int nt = 0; nt < 4; nt++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]),
                          "+f"(acc[mt][nt][3])
                        : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]),
                          "r"(b[nt][0]), "r"(b[nt][1]));
        }
        if (c + 1 < chunks) store(buf ^ 1);
        __syncthreads();
    }
    unsigned int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                unsigned int r = m0 + wm * 64 + mt * 16 + g + h * 8;
                unsigned int n = n0 + wn * 32 + nt * 8 + 2 * t;
                if (r >= M) continue;
                if (n + 1 < N)
                    *(float2*)&Y[(u64)r * N + n] = make_float2(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
                else if (n < N)
                    Y[(u64)r * N + n] = acc[mt][nt][2 * h];
            }
}
"#
);
/// `top_k_rows`: the K largest of each row of X `[rows, n]`, a block (1024 threads) a row.
/// Floats map to order-preserving u32 keys; four 8-bit radix rounds find the K-th largest key
/// T; then each thread, over its own contiguous range, keeps the keys above T and the first
/// ties at T, placed by a block-wide prefix count, so the kept set (and its order after the
/// final sort: value descending, index ascending) never depends on thread timing.
/// Out `[rows, 2K]`: K values, then K indices as float bits. Needs K <= 64 and K <= n.
pub const TOPK_CUDA: &str = r#"
#define TK_T 1024
__device__ __forceinline__ unsigned int tk_key(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// Exclusive prefix sum of v over the block (TK_T threads); returns this thread's offset and
// the total in *tot.
__device__ __forceinline__ unsigned int tk_scan(unsigned int v, unsigned int* sh, unsigned int* tot) {
    unsigned int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    unsigned int x = v;
    for (int o = 1; o < 32; o <<= 1) { unsigned int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        unsigned int s = sh[lane];
        for (int o = 1; o < 32; o <<= 1) { unsigned int y = __shfl_up_sync(0xffffffffu, s, o); if (lane >= o) s += y; }
        sh[lane] = s;
    }
    __syncthreads();
    unsigned int before = (w ? sh[w - 1] : 0) + x - v;
    *tot = sh[31];
    __syncthreads();
    return before;
}
extern "C" __global__ void __launch_bounds__(TK_T) top_k_rows(
    const float* __restrict__ X, float* __restrict__ Out, unsigned int n, unsigned int K)
{
    __shared__ unsigned int hist[256];
    __shared__ unsigned int sh[32];
    __shared__ unsigned int sel[2];
    __shared__ unsigned int ck[64], ci[64];
    const float* x = X + (unsigned long long)blockIdx.x * n;
    unsigned int tid = threadIdx.x;
    unsigned int per = (n + TK_T - 1) / TK_T, b = tid * per, e = min(b + per, n);
    unsigned int prefix = 0, pmask = 0, want = K;   // want: how many at-or-above still needed
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned int i = tid; i < 256; i += TK_T) hist[i] = 0;
        __syncthreads();
        for (unsigned int i = tid; i < n; i += TK_T) {   // coalesced; order doesn't matter here
            unsigned int key = tk_key(x[i]);
            if ((key & pmask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int acc = 0, d = 255;
            for (;; d--) {
                if (acc + hist[d] >= want || d == 0) break;
                acc += hist[d];
            }
            sel[0] = d; sel[1] = want - acc;
        }
        __syncthreads();
        prefix |= sel[0] << shift; pmask |= 255u << shift; want = sel[1];
        __syncthreads();
    }
    // prefix = T; keys above T: K - want of them; then the first `want` ties by index.
    unsigned int above = 0, ties = 0;
    for (unsigned int i = b; i < e; i++) {
        unsigned int key = tk_key(x[i]);
        above += key > prefix; ties += key == prefix;
    }
    unsigned int tot, oa = tk_scan(above, sh, &tot);
    unsigned int ot = tk_scan(ties, sh, &tot);
    unsigned int na = K - want;
    for (unsigned int i = b; i < e; i++) {
        unsigned int key = tk_key(x[i]);
        if (key > prefix) { ck[oa] = key; ci[oa] = i; oa++; }
        else if (key == prefix) { if (ot < want) { ck[na + ot] = key; ci[na + ot] = i; } ot++; }
    }
    __syncthreads();
    if (tid == 0) {
        for (unsigned int a = 1; a < K; a++) {
            unsigned int k0 = ck[a], i0 = ci[a]; int j = a - 1;
            while (j >= 0 && (ck[j] < k0 || (ck[j] == k0 && ci[j] > i0))) { ck[j + 1] = ck[j]; ci[j + 1] = ci[j]; j--; }
            ck[j + 1] = k0; ci[j + 1] = i0;
        }
    }
    __syncthreads();
    float* o = Out + (unsigned long long)blockIdx.x * 2 * K;
    for (unsigned int a = tid; a < K; a += TK_T) {
        unsigned int key = ck[a];
        unsigned int u = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
        o[a] = __uint_as_float(u);
        o[K + a] = __uint_as_float(ci[a]);
    }
}
"#;
