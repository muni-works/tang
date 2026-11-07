//! The chunked schedule of a delta-rule scan and its readout, on CUDA
//! tensor cores (bf16 `wmma`, f32 accumulation).
//!
//! [`TensorGraph::chunked_scan_plan`] recognizes `o = scan_readout(s, q)`
//! over `s = delta_scan(k, v, α, β)` ([`DeltaForm`](super::DeltaForm)) and
//! [`ChunkedScan::kernels`] lowers it to the chunked form of the affine
//! scan (Mamba-2's SSD, and the WY/UT form of the gated delta rule):
//!
//! 1. **Within a chunk** of `C = 64` steps, in parallel over every chunk
//!    and head. With `γ` the running sum of `log α` in the chunk, the
//!    product of the chunk's `A_t = α_t(I − β_t k_t k_tᵀ)` factors applied
//!    to the incoming state is `e^{γ}(… − W·S)`, with `W = T·(β e^{γ} k)`,
//!    `U = T·(β v)` and `T = (I + strict_tril(β k kᵀ ⊙ e^{γᵢ−γⱼ}))⁻¹`
//!    (one unit-triangular solve instead of `C` rank-1 updates). The Gram
//!    matrices `k kᵀ`, `q kᵀ` and the products `T·(…)` are tensor-core
//!    matmuls. Without the delta term `T = I` and `W` vanishes (SSD,
//!    linear attention).
//! 2. **Across chunks**, a sequential scan of the `n × m` state only, per
//!    head and column block: `V' = U − W·S`, `o = (q e^{γ})·S +
//!    tril(q kᵀ ⊙ e^{γᵢ−γⱼ})·V'`, `S ← e^{γ_C} S + (k e^{γ_C−γ})ᵀ·V'` —
//!    four matmuls a chunk, the readout fused, the state in registers.
//!
//! Buffers: `q`, `k` `[H, T, n]`, `v` `[H, T, m]` bf16, `α`, `β` `[H, T]`
//! f32, the output `[H, T, m]` bf16. `T` a multiple of 64; `n`, `m`
//! multiples of 16 (n ≤ 128, m ≤ 128 for shared memory). Checked against
//! the f64 interpreter by `scan_bench` in tang-bench.
use super::{MonoidKind, Op, ScanKind, TensorGraph, TensorId};

/// Steps per chunk.
pub const CHUNK: usize = 64;

/// A recognized delta-rule scan with its readout (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkedScan {
    pub heads: usize,
    pub t: usize,
    /// State rows (`k`, `q` width).
    pub n: usize,
    /// State columns (`v` width).
    pub m: usize,
    pub delta: bool,
    pub scale: f64,
    pub q: TensorId,
    pub k: TensorId,
    pub v: TensorId,
    pub alpha: Option<TensorId>,
    pub beta: Option<TensorId>,
}

/// One launch of the chunked schedule.
#[derive(Clone, Debug)]
pub struct ScanLaunch {
    /// CUDA source; entry point [`CUDA_ENTRY`](super::cuda::CUDA_ENTRY).
    pub source: String,
    pub grid: (u32, u32, u32),
    pub block: u32,
    /// Dynamic shared memory (bytes; also `TANG_SMEM` in the source).
    pub smem: u32,
}

const INCLUDES: &str = "#include <mma.h>\n#include <cuda_bf16.h>\n";

const PRELUDE: &str = r#"using namespace nvcuda;
typedef __nv_bfloat16 bf;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, bf, wmma::row_major> FA;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, bf, wmma::col_major> FAc;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, bf, wmma::row_major> FBr;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, bf, wmma::col_major> FBc;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> FC;
#define NT (T / C)
#define LDK (DK + 8)
#define LDV (DV + 8)
#define LDF (C + 4)
#define LDT (C + 8)
"#;

/// Kernel 1: per chunk and head, `γ` (per row), `γ_C`, and
/// `W = T·diag(β e^{γ})·k`, `U = T·diag(β)·v`. Operands come straight
/// from global memory into fragments; shared memory holds only the
/// `C × C` f32 work (two blocks an SM). Grid `(T/C, H)`, 256 threads.
const PREP: &str = r#"
// Unit lower-triangular inverse of the diagonal block at b0 of I + A
// (A in `sa`), written to `tm`; lane `cl` computes column b0 + cl.
__device__ __forceinline__ void diag_inv(const float* sa, float* tm, int b0, int cl) {
    float col[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) col[r] = r == cl ? 1.0f : 0.0f;
    #pragma unroll
    for (int rr = 1; rr < 16; rr++) {
        const float* ar = sa + (b0 + rr) * LDF + b0;
        float a0 = 0.0f, a1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < rr; j += 2) {
            a0 = fmaf(ar[j], col[j], a0);
            if (j + 1 < rr) a1 = fmaf(ar[j + 1], col[j + 1], a1);
        }
        if (cl < rr) col[rr] = -(a0 + a1);
    }
    #pragma unroll
    for (int r = 0; r < 16; r++) tm[(b0 + r) * LDF + b0 + cl] = col[r];
}

// Block column J of T below the diagonal: T_iJ = −T_ii·Σ_{J≤k<i} A_ik·T_kJ,
// lane `cl` holding column 16J + cl of T in registers.
template <int J>
__device__ __forceinline__ void tcol(const float* sa, float* tm, int cl) {
    float col[C - 16 * J];
    #pragma unroll
    for (int r = 0; r < 16; r++) col[r] = tm[(16 * J + r) * LDF + 16 * J + cl];
    #pragma unroll
    for (int i = J + 1; i < C / 16; i++) {
        float x[16];
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            const float* ar = sa + (16 * i + r) * LDF;
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll
            for (int k = 16 * J; k < 16 * i; k += 2) {
                a0 = fmaf(ar[k], col[k - 16 * J], a0);
                a1 = fmaf(ar[k + 1], col[k + 1 - 16 * J], a1);
            }
            x[r] = a0 + a1;
        }
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            const float* tr = tm + (16 * i + r) * LDF + 16 * i;
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k <= r; k++) acc = fmaf(tr[k], x[k], acc);
            col[16 * (i - J) + r] = -acc;
        }
    }
    #pragma unroll
    for (int r = 16; r < C - 16 * J; r++) tm[(16 * J + r) * LDF + 16 * J + cl] = col[r];
}

extern "C" __global__ void __launch_bounds__(256) main0(
    const bf* __restrict__ K, const bf* __restrict__ V,
    const float* __restrict__ AL, const float* __restrict__ BE,
    bf* __restrict__ Wo, bf* __restrict__ Uo, float* __restrict__ GR, float* __restrict__ GC) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* sa = (float*)smem;      // [C][LDF]  A, then bf16 T' / T'' and staging
    float* tm = sa + C * LDF;      // [C][LDF]  T
    float* gam = tm + C * LDF;     // [C]
    float* bet = gam + C;          // [C]
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t row0 = (size_t)h * T + (size_t)c * C;
    if (tid < C) {
#if GATE
        gam[tid] = logf(AL[row0 + tid]);
#else
        gam[tid] = 0.0f;
#endif
#if BETA
        bet[tid] = BE[row0 + tid];
#else
        bet[tid] = 1.0f;
#endif
    }
    __syncthreads();
    if (warp == 0) {
        const float a = gam[2 * lane], b = a + gam[2 * lane + 1];
        float s = b;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += n;
        }
        gam[2 * lane] = s - b + a;
        gam[2 * lane + 1] = s;
        GR[row0 + 2 * lane] = s - b + a;
        GR[row0 + 2 * lane + 1] = s;
        if (lane == 31) GC[h * NT + c] = s;
    }
#if DELTA
    // k kᵀ from k staged (bf16) over A's space, into T's space as f32
    bf* kst = (bf*)sa;
    for (int i = tid; i < C * DK / 8; i += 256) {
        const int r = i / (DK / 8), x = (i % (DK / 8)) * 8;
        *(uint4*)&kst[r * LDK + x] = *(const uint4*)&K[(row0 + r) * DK + x];
    }
    __syncthreads();
    for (int tt = warp; tt < 16; tt += 8) {
        const int tr = tt >> 2, tc = tt & 3;
        FC acc;
        wmma::fill_fragment(acc, 0.0f);
        if (tc <= tr) {
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBc b;
                wmma::load_matrix_sync(a, kst + tr * 16 * LDK + e, LDK);
                wmma::load_matrix_sync(b, kst + tc * 16 * LDK + e, LDK);
                wmma::mma_sync(acc, a, b, acc);
            }
        }
        wmma::store_matrix_sync(tm + tr * 16 * LDF + tc * 16, acc, LDF, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        sa[r * LDF + j] = j < r ? bet[r] * tm[r * LDF + j] * __expf(gam[r] - gam[j]) : 0.0f;
    }
    __syncthreads();
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        tm[r * LDF + j] = r == j ? 1.0f : 0.0f;
    }
    __syncthreads();
#ifndef XNOSOLVE
    // T = (I + A)⁻¹ in 16×16 blocks, columns in registers (lane = column)
    if (warp < C / 16 && lane < 16) diag_inv(sa, tm, warp * 16, lane);
    __syncthreads();
    if (lane < 16) {
        if (warp == 0) tcol<0>(sa, tm, lane);
        else if (warp == 1) tcol<1>(sa, tm, lane);
        else if (warp == 2) tcol<2>(sa, tm, lane);
    }
    __syncthreads();
#endif
    // T'' = T·diag(β) as a bf16 operand over A's space (A is done), then
    // W = T''·(e^γ k) and U = T''·v with k, v staged over T's space
    bf* t2 = (bf*)sa;
    float* stg = (float*)(t2 + C * LDT) + warp * 256;
    bf* ost = (bf*)tm;
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        t2[r * LDT + j] = __float2bfloat16(tm[r * LDF + j] * bet[j]);
    }
    __syncthreads();
    auto tri = [&](int ld, int ncols, bf* out) {
        for (int tt = warp; tt < 4 * (ncols / 16); tt += 8) {
            const int tr = tt & 3, tcol = tt >> 2;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int e = 0; e <= tr * 16; e += 16) {
                FA a;
                FBr b;
                wmma::load_matrix_sync(a, t2 + tr * 16 * LDT + e, LDT);
                wmma::load_matrix_sync(b, ost + e * ld + tcol * 16, ld);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 128; e += 32) {
                const int rr = e >> 3, cc = (e & 7) * 2;
                __nv_bfloat162 v2 = __floats2bfloat162_rn(stg[rr * 16 + cc], stg[rr * 16 + cc + 1]);
                *(__nv_bfloat162*)&out[(row0 + tr * 16 + rr) * ncols + tcol * 16 + cc] = v2;
            }
            __syncwarp();
        }
    };
    for (int i = tid; i < C * DK / 2; i += 256) {
        const int r = i / (DK / 2), x = (i % (DK / 2)) * 2;
        const __nv_bfloat162 v = *(const __nv_bfloat162*)&K[(row0 + r) * DK + x];
        const float g = __expf(gam[r]);
        *(__nv_bfloat162*)&ost[r * LDK + x] = __floats2bfloat162_rn(__low2float(v) * g, __high2float(v) * g);
    }
    __syncthreads();
#ifndef XNOTRI
    tri(LDK, DK, Wo);
#endif
    __syncthreads();
    for (int i = tid; i < C * DV / 8; i += 256) {
        const int r = i / (DV / 8), x = (i % (DV / 8)) * 8;
        *(uint4*)&ost[r * LDV + x] = *(const uint4*)&V[(row0 + r) * DV + x];
    }
    __syncthreads();
#ifndef XNOTRI
    tri(LDV, DV, Uo);
#endif
#else
    __syncthreads();
    for (int i = tid; i < C * DV / 2; i += 256) {
        const int r = i / (DV / 2), x = (i % (DV / 2)) * 2;
        const __nv_bfloat162 v = *(const __nv_bfloat162*)&V[(row0 + r) * DV + x];
        *(__nv_bfloat162*)&Uo[(row0 + r) * DV + x] =
            __floats2bfloat162_rn(__low2float(v) * bet[r], __high2float(v) * bet[r]);
    }
#endif
}
"#;

/// Kernel 2: the inter-chunk state scan. Grid `(DV/BV, H)`, 128
/// threads; warp `w` holds state rows `w·DK/4…` of the `DK × BV` block in
/// accumulator fragments. Per chunk: write the incoming state `S_c`
/// (bf16), `V' = U − W·S`, then `S ← e^{γ_C}·S + kᵀ·(e^{γ_C−γ} V')`. `W`
/// and `k` fragments load from global (L2); shared memory holds the
/// state operand and `V'` only, so every block is resident at once.
const STATE: &str = r#"
#define LDB (BV + 8)
#define NTC (BV / 16)
extern "C" __global__ void __launch_bounds__(128) main0(
    const bf* __restrict__ Wo, const bf* __restrict__ Uo, const bf* __restrict__ K,
    const float* __restrict__ GR, const float* __restrict__ GC, bf* __restrict__ Hs, bf* __restrict__ Vn) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf* sb = (bf*)smem;                // [DK][LDB]
    bf* vn = sb + DK * LDB;            // [C][LDB]  e^{γ_C−γ}·V'
    float* gm = (float*)(vn + C * LDB);  // [C] e^{γ_C−γ}
    float* stg = gm + C + (threadIdx.x >> 5) * 256;
    const int vb = blockIdx.x * BV, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int RT = DK / 64;
    FC st[RT][NTC];
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < NTC; j++) wmma::fill_fragment(st[i][j], 0.0f);
    // this chunk's W row tile (warp's rows) and k column tiles, prefetched
    // a chunk ahead into registers
    FA wf[DK / 16];
    FAc kf[RT][C / 16];
    float gcn = 0.0f, grn = 0.0f;
    __nv_bfloat162 uf[NTC][4];
    auto fetch = [&](int c, FA* w, FAc (*k)[C / 16], float& g1, float& g2, __nv_bfloat162 (*u)[4]) {
        const size_t r0 = (size_t)h * T + (size_t)c * C;
        #pragma unroll
        for (int j = 0; j < NTC; j++)
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const int e = lane + 32 * q;
                u[j][q] = *(const __nv_bfloat162*)&Uo[(r0 + warp * 16 + (e >> 3)) * DV + vb + j * 16 + (e & 7) * 2];
            }
#if DELTA
        #pragma unroll
        for (int e = 0; e < DK / 16; e++) wmma::load_matrix_sync(w[e], Wo + (r0 + warp * 16) * DK + e * 16, DK);
#endif
        #pragma unroll
        for (int i = 0; i < RT; i++)
            #pragma unroll
            for (int e = 0; e < C / 16; e++)
                wmma::load_matrix_sync(k[i][e], K + (r0 + e * 16) * DK + (warp * RT + i) * 16, DK);
        g1 = GC[h * NT + c];
        g2 = tid < C ? GR[r0 + tid] : 0.0f;
    };
    fetch(0, wf, kf, gcn, grn, uf);
    for (int c = 0; c < NT; c++) {
        const size_t row0 = (size_t)h * T + (size_t)c * C;
        const float glast = gcn;
        const float gc = __expf(glast);
        if (tid < C) gm[tid] = __expf(glast - grn);
        FA wn[DK / 16];
        FAc kn[RT][C / 16];
        __nv_bfloat162 un[NTC][4];
        if (c + 1 < NT) fetch(c + 1, wn, kn, gcn, grn, un);
        bf* hs = Hs + ((size_t)h * NT + c) * DK * DV;
        #pragma unroll
        for (int i = 0; i < RT; i++)
            #pragma unroll
            for (int j = 0; j < NTC; j++) {
                wmma::store_matrix_sync(stg, st[i][j], 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = lane; e < 128; e += 32) {
                    const int r = (warp * RT + i) * 16 + (e >> 3), x = j * 16 + (e & 7) * 2;
                    const __nv_bfloat162 v = __floats2bfloat162_rn(stg[(e >> 3) * 16 + (e & 7) * 2], stg[(e >> 3) * 16 + (e & 7) * 2 + 1]);
                    *(__nv_bfloat162*)&sb[r * LDB + x] = v;
                    *(__nv_bfloat162*)&hs[(size_t)r * DV + vb + x] = v;
                }
                __syncwarp();
            }
        __syncthreads();
        #pragma unroll
        for (int tcol = 0; tcol < NTC; tcol++) {
            const int tr = warp;
#if DELTA
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FBr b;
                wmma::load_matrix_sync(b, sb + e * LDB + tcol * 16, LDB);
                wmma::mma_sync(acc, wf[e / 16], b, acc);
            }
            wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
#endif
            __syncwarp();
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const int e = lane + 32 * q;
                const int rr = e >> 3, cc = (e & 7) * 2;
                const int r = tr * 16 + rr, x = tcol * 16 + cc;
                const __nv_bfloat162 u = uf[tcol][q];
#if DELTA
                const float v0 = __low2float(u) - stg[rr * 16 + cc], v1 = __high2float(u) - stg[rr * 16 + cc + 1];
                const __nv_bfloat162 v = __floats2bfloat162_rn(v0, v1);
                *(__nv_bfloat162*)&Vn[(row0 + r) * DV + vb + x] = v;
#else
                const __nv_bfloat162 v = u;
#endif
                const float g = gm[r];
                *(__nv_bfloat162*)&vn[r * LDB + x] = __floats2bfloat162_rn(__low2float(v) * g, __high2float(v) * g);
            }
            __syncwarp();
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < RT; i++)
            #pragma unroll
            for (int j = 0; j < NTC; j++) {
                #pragma unroll
                for (int x = 0; x < st[i][j].num_elements; x++) st[i][j].x[x] *= gc;
                #pragma unroll
                for (int e = 0; e < C; e += 16) {
                    FBr b;
                    wmma::load_matrix_sync(b, vn + e * LDB + j * 16, LDB);
                    wmma::mma_sync(st[i][j], kf[i][e / 16], b, st[i][j]);
                }
            }
        if (c + 1 < NT) {
            #pragma unroll
            for (int e = 0; e < DK / 16; e++) wf[e] = wn[e];
            #pragma unroll
            for (int j = 0; j < NTC; j++)
                #pragma unroll
                for (int q = 0; q < 4; q++) uf[j][q] = un[j][q];
            #pragma unroll
            for (int i = 0; i < RT; i++)
                #pragma unroll
                for (int e = 0; e < C / 16; e++) kf[i][e] = kn[i][e];
        }
        __syncthreads();
    }
}
"#;

/// Kernel 3: the readout, in parallel over chunks: `o = scale·(e^{γ}
/// q·S_c + tril(q kᵀ ⊙ e^{γᵢ−γⱼ})·V')`. Grid `(T/C, H)`, 256 threads;
/// `q` staged in shared memory (it feeds every tile), `k`, `S_c`, `V'`
/// fragments from global; `q kᵀ` recomputed here.
const OUT: &str = r#"
extern "C" __global__ void __launch_bounds__(256) main0(
    const bf* __restrict__ Q, const bf* __restrict__ K, const float* __restrict__ GR,
    const bf* __restrict__ Hs, const bf* __restrict__ Vn, bf* __restrict__ O) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf* qs = (bf*)smem;                       // [C][LDK]
    bf* ab = qs + C * LDK;                    // [C][LDT]
    float* gm = (float*)(ab + C * LDT);       // [C]
    float* stg = gm + C + (threadIdx.x >> 5) * 512;
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t row0 = (size_t)h * T + (size_t)c * C;
    const bf* hs = Hs + ((size_t)h * NT + c) * DK * DV;
    for (int i = tid; i < C * DK / 8; i += 256) {
        const int r = i / (DK / 8), x = (i % (DK / 8)) * 8;
        *(uint4*)&qs[r * LDK + x] = *(const uint4*)&Q[(row0 + r) * DK + x];
    }
    if (tid < C) gm[tid] = GR[row0 + tid];
    __syncthreads();
    // masked q kᵀ: the 10 lower tiles over 8 warps, each straight to bf16
    for (int tt = warp; tt < 16; tt += 8) {
        const int tr = tt >> 2, tc = tt & 3;
        if (tc <= tr) {
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBc b;
                wmma::load_matrix_sync(a, qs + tr * 16 * LDK + e, LDK);
                wmma::load_matrix_sync(b, K + (row0 + tc * 16) * DK + e, DK);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = tr * 16 + e / 16, j = tc * 16 + e % 16;
                ab[r * LDT + j] = __float2bfloat16(j <= r ? stg[e] * __expf(gm[r] - gm[j]) : 0.0f);
            }
            __syncwarp();
        } else {
            for (int e = lane; e < 256; e += 32)
                ab[(tr * 16 + e / 16) * LDT + tc * 16 + e % 16] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    for (int tt = warp; tt < 4 * (DV / 16); tt += 8) {
        const int tr = tt & 3, tcol = tt >> 2;
        FC acc, acc2;
        wmma::fill_fragment(acc, 0.0f);
        wmma::fill_fragment(acc2, 0.0f);
        #pragma unroll
        for (int e = 0; e < DK; e += 16) {
            FA a;
            FBr b;
            wmma::load_matrix_sync(a, qs + tr * 16 * LDK + e, LDK);
            wmma::load_matrix_sync(b, hs + (size_t)e * DV + tcol * 16, DV);
            wmma::mma_sync(acc, a, b, acc);
        }
        for (int e = 0; e <= tr * 16; e += 16) {
            FA a;
            FBr b;
            wmma::load_matrix_sync(a, ab + tr * 16 * LDT + e, LDT);
            wmma::load_matrix_sync(b, Vn + (row0 + e) * DV + tcol * 16, DV);
            wmma::mma_sync(acc2, a, b, acc2);
        }
        wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(stg + 256, acc2, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 128; e += 32) {
            const int rr = e >> 3, cc = (e & 7) * 2, r = tr * 16 + rr;
            const float g = __expf(gm[r]);
            const float o0 = (float)SCALE * (g * stg[rr * 16 + cc] + stg[256 + rr * 16 + cc]);
            const float o1 = (float)SCALE * (g * stg[rr * 16 + cc + 1] + stg[256 + rr * 16 + cc + 1]);
            *(__nv_bfloat162*)&O[(row0 + r) * DV + tcol * 16 + cc] = __floats2bfloat162_rn(o0, o1);
        }
        __syncwarp();
    }
}
"#;

impl ChunkedScan {
    fn defines(&self) -> String {
        format!(
            "#define T {}\n#define C {CHUNK}\n#define DK {}\n#define DV {}\n#define DELTA {}\n#define GATE {}\n#define BETA {}\n#define SCALE {:e}\n",
            self.t,
            self.n,
            self.m,
            self.delta as u8,
            self.alpha.is_some() as u8,
            self.beta.is_some() as u8,
            self.scale
        )
    }

    /// Scratch buffers between the kernels, as element counts: bf16 `W`,
    /// `U`, the per-chunk states `[H, T/C, n, m]` and `V'` (unused without
    /// the delta term, where `V' = U`); then f32 `γ` per row and `γ_C`
    /// per chunk.
    pub fn scratch(&self) -> [usize; 6] {
        let rows = self.heads * self.t;
        [
            rows * self.n,
            rows * self.m,
            self.heads * (self.t / CHUNK) * self.n * self.m,
            rows * self.m,
            rows,
            self.heads * (self.t / CHUNK),
        ]
    }

    /// The three launches: per-chunk transform, state scan, readout.
    /// `bv` is the state column block of the scan (16, 32 or 64).
    /// Arguments, in order — prep: `k v α β | W U γ γ_C`; state: `W U k γ
    /// γ_C | H V'`; readout: `q k γ H V' | o` (`V'` is `U` without the
    /// delta term).
    pub fn kernels(&self, bv: usize) -> [ScanLaunch; 3] {
        assert!(self.t % CHUNK == 0, "chunked scan: T = {} not a multiple of {CHUNK}", self.t);
        assert!(self.n % 64 == 0 && self.n <= 128, "chunked scan: n = {}", self.n);
        assert!(
            self.m % 64 == 0 && self.m % bv == 0 && bv % 16 == 0,
            "chunked scan: m = {}, bv {bv}",
            self.m
        );
        let d = self.defines();
        let (c, n) = (CHUNK, self.n);
        let smem1 = 2 * c * (c + 4) * 4 + 2 * c * 4;
        let smem2 = (n * (bv + 8) + c * (bv + 8)) * 2 + c * 4 + 4 * 256 * 4;
        let smem3 = (c * (n + 8) + c * (c + 8)) * 2 + c * 4 + 8 * 512 * 4;
        let src = |smem: usize, extra: &str, body: &str| {
            let x = std::env::var("TANG_SCAN_DEFS").unwrap_or_default().replace(';', "\n");
            format!("#define TANG_SMEM {smem}\n{INCLUDES}{d}{x}\n{extra}{PRELUDE}{body}")
        };
        [
            ScanLaunch {
                source: src(smem1, "", PREP),
                grid: ((self.t / c) as u32, self.heads as u32, 1),
                block: 256,
                smem: smem1 as u32,
            },
            ScanLaunch {
                source: src(smem2, &format!("#define BV {bv}\n"), STATE),
                grid: ((self.m / bv) as u32, self.heads as u32, 1),
                block: 128,
                smem: smem2 as u32,
            },
            ScanLaunch {
                source: src(smem3, "", OUT),
                grid: ((self.t / c) as u32, self.heads as u32, 1),
                block: 256,
                smem: smem3 as u32,
            },
        ]
    }
}

impl TensorGraph {
    /// The chunked plan of `o` when it is `scan_readout(delta_scan(…),
    /// q, scale)` over graph inputs laid out `[H, T, ·]` (the scan dim
    /// 1). `None` otherwise: the scan then runs as an ordinary scan
    /// kernel.
    pub fn chunked_scan_plan(&self, o: TensorId) -> Option<ChunkedScan> {
        let c = self.comp_at(o.index())?;
        let m = c.monoid.as_ref()?;
        if m.kind() != MonoidKind::Sum || c.args.len() != 2 || c.red.len() != 1 {
            return None;
        }
        let (s, q) = (c.args[0].src, c.args[1].src);
        let sc = self.scan_at(s)?;
        let ScanKind::Affine { form: Some(f) } = &sc.kind else {
            return None;
        };
        if sc.exclusive || sc.reverse || sc.dim != 1 {
            return None;
        }
        // the readout's body is scale·S·q
        let probe: Vec<f64> = c.body.eval_many(&c.body_outs, &[1.0, 1.0]);
        let probe2: Vec<f64> = c.body.eval_many(&c.body_outs, &[2.0, 3.0]);
        let scale = probe[0];
        if (probe2[0] - 6.0 * scale).abs() > 1e-12 * scale.abs().max(1.0) {
            return None;
        }
        // views: S[h, t, i, c] and q[h, t, i] over space [h, t, c, i]
        let pick = |v: &super::View, want: &[Option<usize>]| *v == super::View::pick(4, want);
        if !pick(&c.args[0].view, &[Some(0), Some(1), Some(3), Some(2)])
            || !pick(&c.args[1].view, &[Some(0), Some(1), Some(3)])
        {
            return None;
        }
        let input = |t: TensorId| matches!(self.nodes[t.index()].op, Op::Input(_));
        let k = sc.inputs[f.k];
        let v = sc.inputs[f.v];
        let alpha = f.alpha.map(|i| sc.inputs[i]);
        let beta = f.beta.map(|i| sc.inputs[i]);
        if ![q, k, v].into_iter().chain(alpha).chain(beta).all(input) {
            return None;
        }
        let dims = |t: TensorId| super::fixed(self.shape(t));
        let (sk, sv) = (dims(k), dims(v));
        if sk.len() != 3 || dims(q) != sk {
            return None;
        }
        Some(ChunkedScan {
            heads: sk[0],
            t: sk[1],
            n: sk[2],
            m: sv[2],
            delta: f.delta,
            scale,
            q,
            k,
            v,
            alpha,
            beta,
        })
    }
}
