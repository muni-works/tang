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
    bf* __restrict__ Wo, bf* __restrict__ Uo, float* __restrict__ GR, float* __restrict__ GC,
    float* __restrict__ TS) {
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
#if SAVE_T
    // T for the backward ([H, T/C, C, C] f32)
    for (int i = tid; i < C * C; i += 256)
        TS[((size_t)h * NT + c) * C * C + i] = tm[(i / C) * LDF + i % C];
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

/// Backward kernel 1: `A_qk = scale·tril(q kᵀ ⊙ e^{γᵢ−γⱼ})` per chunk
/// (bf16 `[H, T, C]`), for the reverse state scan. Grid `(T/C, H)`, 128
/// threads.
const BWD_AQK: &str = r#"
extern "C" __global__ void __launch_bounds__(128) main0(
    const bf* __restrict__ Q, const bf* __restrict__ K, const float* __restrict__ GR, bf* __restrict__ Ao) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* gm = (float*)smem;
    float* stg = gm + C + (threadIdx.x >> 5) * 256;
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t row0 = (size_t)h * T + (size_t)c * C;
    if (tid < C) gm[tid] = GR[row0 + tid];
    __syncthreads();
    for (int tt = warp; tt < 16; tt += 4) {
        const int tr = tt >> 2, tc = tt & 3;
        FC acc;
        wmma::fill_fragment(acc, 0.0f);
        if (tc <= tr) {
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBc b;
                wmma::load_matrix_sync(a, Q + (row0 + tr * 16) * DK + e, DK);
                wmma::load_matrix_sync(b, K + (row0 + tc * 16) * DK + e, DK);
                wmma::mma_sync(acc, a, b, acc);
            }
        }
        wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = tr * 16 + e / 16, j = tc * 16 + e % 16;
            Ao[(row0 + r) * C + j] = __float2bfloat16(j <= r ? (float)SCALE * stg[e] * __expf(gm[r] - gm[j]) : 0.0f);
        }
        __syncwarp();
    }
}
"#;

/// Backward kernel 2: the reverse scan of the state cotangent with the
/// transposed chunk map. Per chunk, from the last: write `dS_end` (bf16),
/// `dV' = e^{γ_C−γ}·(k·dS) + A_qkᵀ·dO`, then `dS ← e^{γ_C}·dS − Wᵀ·dV' +
/// qᵀ·(scale·e^{γ} dO)`. Grid `(DV/BV, H)`, 128 threads.
const BWD_STATE: &str = r#"
#define LDB (BV + 8)
#define NTC (BV / 16)
extern "C" __global__ void __launch_bounds__(128) main0(
    const bf* __restrict__ Q, const bf* __restrict__ K, const bf* __restrict__ Wo,
    const bf* __restrict__ Ao, const bf* __restrict__ DO, const float* __restrict__ GR,
    const float* __restrict__ GC, bf* __restrict__ DHs, bf* __restrict__ DVn) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf* sb = (bf*)smem;                // [DK][LDB]  dS
    bf* vng = sb + DK * LDB;           // [C][LDB]   −dV'
    bf* dog = vng + C * LDB;           // [C][LDB]   scale·e^γ·dO
    float* gm = (float*)(dog + C * LDB);   // [C] γ
    float* stg = gm + C + (threadIdx.x >> 5) * 512;
    const int vb = blockIdx.x * BV, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int RT = DK / 64;
    FC st[RT][NTC];
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < NTC; j++) wmma::fill_fragment(st[i][j], 0.0f);
    for (int c = NT - 1; c >= 0; c--) {
        const size_t row0 = (size_t)h * T + (size_t)c * C;
        const float glast = GC[h * NT + c];
        if (tid < C) gm[tid] = GR[row0 + tid];
        bf* dhs = DHs + ((size_t)h * NT + c) * DK * DV;
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
                    *(__nv_bfloat162*)&dhs[(size_t)r * DV + vb + x] = v;
                }
                __syncwarp();
            }
        __syncthreads();
        #pragma unroll
        for (int tcol = 0; tcol < NTC; tcol++) {
            const int tr = warp;
            FC a1, a2;
            wmma::fill_fragment(a1, 0.0f);
            wmma::fill_fragment(a2, 0.0f);
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBr b;
                wmma::load_matrix_sync(a, K + (row0 + tr * 16) * DK + e, DK);
                wmma::load_matrix_sync(b, sb + e * LDB + tcol * 16, LDB);
                wmma::mma_sync(a1, a, b, a1);
            }
            // A_qkᵀ is upper: rows tr need columns e ≥ tr of A_qk
            for (int e = tr * 16; e < C; e += 16) {
                FAc a;
                FBr b;
                wmma::load_matrix_sync(a, Ao + (row0 + e) * C + tr * 16, C);
                wmma::load_matrix_sync(b, DO + (row0 + e) * DV + vb + tcol * 16, DV);
                wmma::mma_sync(a2, a, b, a2);
            }
            wmma::store_matrix_sync(stg, a1, 16, wmma::mem_row_major);
            wmma::store_matrix_sync(stg + 256, a2, 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const int e = lane + 32 * q;
                const int rr = e >> 3, cc = (e & 7) * 2, r = tr * 16 + rr, x = tcol * 16 + cc;
                const float g = __expf(glast - gm[r]);
                const float d0 = g * stg[rr * 16 + cc] + stg[256 + rr * 16 + cc];
                const float d1 = g * stg[rr * 16 + cc + 1] + stg[256 + rr * 16 + cc + 1];
                *(__nv_bfloat162*)&DVn[(row0 + r) * DV + vb + x] = __floats2bfloat162_rn(d0, d1);
                *(__nv_bfloat162*)&vng[r * LDB + x] = __floats2bfloat162_rn(-d0, -d1);
                const __nv_bfloat162 dd = *(const __nv_bfloat162*)&DO[(row0 + r) * DV + vb + x];
                const float sg = (float)SCALE * __expf(gm[r]);
                *(__nv_bfloat162*)&dog[r * LDB + x] = __floats2bfloat162_rn(__low2float(dd) * sg, __high2float(dd) * sg);
            }
            __syncwarp();
        }
        __syncthreads();
        const float gc = __expf(glast);
        #pragma unroll
        for (int i = 0; i < RT; i++)
            #pragma unroll
            for (int j = 0; j < NTC; j++) {
                #pragma unroll
                for (int x = 0; x < st[i][j].num_elements; x++) st[i][j].x[x] *= gc;
                #pragma unroll
                for (int e = 0; e < C; e += 16) {
                    FAc a;
                    FBr b;
#if DELTA
                    wmma::load_matrix_sync(a, Wo + (row0 + e) * DK + (warp * RT + i) * 16, DK);
                    wmma::load_matrix_sync(b, vng + e * LDB + j * 16, LDB);
                    wmma::mma_sync(st[i][j], a, b, st[i][j]);
#endif
                    wmma::load_matrix_sync(a, Q + (row0 + e) * DK + (warp * RT + i) * 16, DK);
                    wmma::load_matrix_sync(b, dog + e * LDB + j * 16, LDB);
                    wmma::mma_sync(st[i][j], a, b, st[i][j]);
                }
            }
        __syncthreads();
    }
}
"#;

/// Backward kernel 3: per chunk, everything else. With `S₀` (forward
/// states), `dS_end`, `dV'`, `dO`, `V'`, `W` and the saved `T`:
/// `dW = −dV'·S₀ᵀ`, `dT = (dV'·vᵀ)diag(β) + (dW·kᵀ)diag(β e^γ)`,
/// `dA = −Tᵀ·dT·Tᵀ`, then `dq`, `dk`, `dv`, `dβ` and `dα` through the
/// masked Gram matrices and the decays (`dγ` summed back to `d log α`).
/// Grid `(T/C, H)`, 256 threads. Requires `DK == DV == 128`.
const BWD_GRAD: &str = r#"
// Sum over the 16 lanes of a half warp (all lanes get it).
__device__ __forceinline__ float hsum16(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}
extern "C" __global__ void __launch_bounds__(256) main0(
    const bf* __restrict__ Q, const bf* __restrict__ K, const bf* __restrict__ V,
    const float* __restrict__ AL, const float* __restrict__ BE, const float* __restrict__ GR,
    const float* __restrict__ GC, const float* __restrict__ TS, const bf* __restrict__ Hs,
    const bf* __restrict__ DHs, const bf* __restrict__ DO, const bf* __restrict__ Vn,
    const bf* __restrict__ DVn, bf* __restrict__ DQ, bf* __restrict__ DK_,
    bf* __restrict__ DV_, float* __restrict__ DA, float* __restrict__ DB) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* f1 = (float*)smem;                    // [C][LDF]  T, then q kᵀ; then staging
    float* f2 = f1 + C * LDF;                    // [C][LDF]  dT, dA, B, dA_qk; then staging
    float* xr = f2 + C * LDF;                    // [C][C] f32 (X = dT·Tᵀ, then k kᵀ) / 2·[C][LDT] bf16
    bf* bsb = (bf*)xr;                           // [C][LDT]  B + Bᵀ   (after k kᵀ)
    bf* g1b = bsb + C * LDT;                     // [C][LDT]  scale·dA_qk ⊙ E
    bf* tb = (bf*)(xr + C * LDT);                // [C][LDT]  T (bf16)
    bf* dwb = tb + C * LDT;                      // [C][LDK]  dW
    float* gam = (float*)(dwb + C * LDK);        // [C]
    float* bet = gam + C;                        // [C]
    float* dgam = bet + C;                       // [C]
    float* dbet = dgam + C;                      // [C]
    float* red = dbet + C;                       // [2]: ⟨dS_end, S₀⟩, Σ R
    float* stg1 = red + 4 + (threadIdx.x >> 5) * 256;  // [8][256] output staging
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t row0 = (size_t)h * T + (size_t)c * C;
    const bf* s0 = Hs + ((size_t)h * NT + c) * DK * DV;
    const bf* dse = DHs + ((size_t)h * NT + c) * DK * DV;
    const float glast = GC[h * NT + c];
    if (tid < C) {
        gam[tid] = GR[row0 + tid];
#if BETA
        bet[tid] = BE[row0 + tid];
#else
        bet[tid] = 1.0f;
#endif
        dgam[tid] = 0.0f;
        dbet[tid] = 0.0f;
    }
    if (tid < 2) red[tid] = 0.0f;
#if DELTA
    for (int i = tid; i < C * C; i += 256) {
        const float t = TS[((size_t)h * NT + c) * C * C + i];
        f1[(i / C) * LDF + i % C] = t;
        tb[(i / C) * LDT + i % C] = __float2bfloat16(t);
    }
#endif
    __syncthreads();
    // ⟨dS_end, S₀⟩ (for d γ_C)
    {
        float acc = 0.0f;
        for (int i = tid; i < DK * DV / 2; i += 256) {
            const __nv_bfloat162 a = ((const __nv_bfloat162*)dse)[i], b = ((const __nv_bfloat162*)s0)[i];
            acc = fmaf(__low2float(a), __low2float(b), fmaf(__high2float(a), __high2float(b), acc));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) atomicAdd(&red[0], acc);
    }
    float* stg = xr + warp * 256;
#if DELTA
#ifndef XNOCC
    // dW = −dV'·S₀ᵀ  [C × DK]; staging over xr (free until X)
    for (int tt = warp; tt < 4 * (DK / 16); tt += 8) {
        const int tr = tt & 3, tcol = tt >> 2;
        FC acc;
        wmma::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int e = 0; e < DV; e += 16) {
            FA a;
            FBc b;
            wmma::load_matrix_sync(a, DVn + (row0 + tr * 16) * DV + e, DV);
            wmma::load_matrix_sync(b, s0 + (size_t)(tcol * 16) * DV + e, DV);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32)
            dwb[(tr * 16 + e / 16) * LDK + tcol * 16 + e % 16] = __float2bfloat16(-stg[e]);
        __syncwarp();
    }
    __syncthreads();
    // dT = (dV'·vᵀ)·diag(β) + (dW·kᵀ)·diag(β e^γ)  → f2
    for (int tt = warp; tt < 16; tt += 8) {
        const int tr = tt >> 2, tc = tt & 3;
        FC a1, a2;
        wmma::fill_fragment(a1, 0.0f);
        wmma::fill_fragment(a2, 0.0f);
        #pragma unroll
        for (int e = 0; e < DV; e += 16) {
            FA a;
            FBc b;
            wmma::load_matrix_sync(a, DVn + (row0 + tr * 16) * DV + e, DV);
            wmma::load_matrix_sync(b, V + (row0 + tc * 16) * DV + e, DV);
            wmma::mma_sync(a1, a, b, a1);
        }
        #pragma unroll
        for (int e = 0; e < DK; e += 16) {
            FA a;
            FBc b;
            wmma::load_matrix_sync(a, dwb + tr * 16 * LDK + e, LDK);
            wmma::load_matrix_sync(b, K + (row0 + tc * 16) * DK + e, DK);
            wmma::mma_sync(a2, a, b, a2);
        }
        wmma::store_matrix_sync(stg, a1, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = tr * 16 + e / 16, j = tc * 16 + e % 16;
            f2[r * LDF + j] = bet[j] * stg[e];
        }
        __syncwarp();
        wmma::store_matrix_sync(stg, a2, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = tr * 16 + e / 16, j = tc * 16 + e % 16;
            f2[r * LDF + j] += bet[j] * __expf(gam[j]) * stg[e];
        }
        __syncwarp();
    }
    __syncthreads();
    // X = dT·Tᵀ, then dA = −Tᵀ·X (strict lower), on tensor cores: dT and X
    // as bf16 operands over xr
    {
        bf* dtb = (bf*)xr;
        bf* xb = dtb + C * LDT;
        for (int i = tid; i < C * C; i += 256) dtb[(i / C) * LDT + i % C] = __float2bfloat16(f2[(i / C) * LDF + i % C]);
        __syncthreads();
        for (int tt = warp; tt < 16; tt += 8) {
            const int tr = tt >> 2, tc = tt & 3;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            // Tᵀ[l][j] = T[j][l], nonzero for l ≤ j: l blocks ≤ tc
            for (int e = 0; e <= tc * 16; e += 16) {
                FA a;
                FBc b;
                wmma::load_matrix_sync(a, dtb + tr * 16 * LDT + e, LDT);
                wmma::load_matrix_sync(b, tb + tc * 16 * LDT + e, LDT);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stg1, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) xb[(tr * 16 + e / 16) * LDT + tc * 16 + e % 16] = __float2bfloat16(stg1[e]);
            __syncwarp();
        }
        __syncthreads();
        for (int tt = warp; tt < 16; tt += 8) {
            const int tr = tt >> 2, tc = tt & 3;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            if (tc <= tr) {
                // Tᵀ[r][k] = T[k][r], nonzero for k ≥ r: k blocks ≥ tr
                for (int e = tr * 16; e < C; e += 16) {
                    FAc a;
                    FBr b;
                    wmma::load_matrix_sync(a, tb + e * LDT + tr * 16, LDT);
                    wmma::load_matrix_sync(b, xb + e * LDT + tc * 16, LDT);
                    wmma::mma_sync(acc, a, b, acc);
                }
            }
            wmma::store_matrix_sync(stg1, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = tr * 16 + e / 16, j = tc * 16 + e % 16;
                f2[r * LDF + j] = j < r ? -stg1[e] : 0.0f;
            }
            __syncwarp();
        }
        __syncthreads();
    }
#endif
    // q kᵀ → f1, k kᵀ → xr (f32)
    for (int tt = warp; tt < 32; tt += 8) {
        const int which = tt >> 4, tr = (tt >> 2) & 3, tc = tt & 3;
        FC acc;
        wmma::fill_fragment(acc, 0.0f);
        if (tc <= tr) {
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBc b;
                wmma::load_matrix_sync(a, (which ? K : Q) + (row0 + tr * 16) * DK + e, DK);
                wmma::load_matrix_sync(b, K + (row0 + tc * 16) * DK + e, DK);
                wmma::mma_sync(acc, a, b, acc);
            }
        }
        if (which) wmma::store_matrix_sync(xr + tr * 16 * C + tc * 16, acc, C, wmma::mem_row_major);
        else wmma::store_matrix_sync(f1 + tr * 16 * LDF + tc * 16, acc, LDF, wmma::mem_row_major);
    }
    __syncthreads();
#if DELTA
    // D = dA (strict lower): dβ, Q = D ⊙ A into dγ, B = D·β_i·E in place
    // row sums (thread r) and column sums (thread C + j) of Q = D ⊙ A, and dβ
    if (tid < C) {
        const int r = tid;
        float sb = 0.0f, sq = 0.0f;
        for (int j = 0; j < r; j++) {
            const float t = f2[r * LDF + j] * xr[r * C + j] * __expf(gam[r] - gam[j]);
            sb += t;
            sq += t * bet[r];
        }
        dbet[r] += sb;
        dgam[r] += sq;
    } else if (tid < 2 * C) {
        const int j = tid - C;
        float sq = 0.0f;
        for (int r = j + 1; r < C; r++)
            sq += f2[r * LDF + j] * xr[r * C + j] * __expf(gam[r] - gam[j]) * bet[r];
        atomicAdd(&dgam[j], -sq);
    }
    __syncthreads();
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        f2[r * LDF + j] = j < r ? f2[r * LDF + j] * bet[r] * __expf(gam[r] - gam[j]) : 0.0f;
    }
    __syncthreads();
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        bsb[r * LDT + j] = __float2bfloat16(f2[r * LDF + j] + f2[j * LDF + r]);
    }
#else
    for (int i = tid; i < C * C; i += 256) bsb[(i / C) * LDT + i % C] = __float2bfloat16(0.0f);
#endif
    __syncthreads();
    // dA_qk = dO·V'ᵀ (lower incl tiles) → f2
    {
        const bf* vp = Vn;  // V' (U without the delta term)
        for (int tt = warp; tt < 16; tt += 8) {
            const int tr = tt >> 2, tc = tt & 3;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            if (tc <= tr) {
                #pragma unroll
                for (int e = 0; e < DV; e += 16) {
                    FA a;
                    FBc b;
                    wmma::load_matrix_sync(a, DO + (row0 + tr * 16) * DV + e, DV);
                    wmma::load_matrix_sync(b, vp + (row0 + tc * 16) * DV + e, DV);
                    wmma::mma_sync(acc, a, b, acc);
                }
            }
            wmma::store_matrix_sync(f2 + tr * 16 * LDF + tc * 16, acc, LDF, wmma::mem_row_major);
        }
    }
    __syncthreads();
    // P = dA_qk ⊙ A_qk into dγ; G1 = scale·dA_qk ⊙ E (incl lower)
    if (tid < C) {
        const int r = tid;
        float sp = 0.0f;
        for (int j = 0; j <= r; j++) sp += f2[r * LDF + j] * f1[r * LDF + j] * __expf(gam[r] - gam[j]);
        atomicAdd(&dgam[r], (float)SCALE * sp);
    } else if (tid < 2 * C) {
        const int j = tid - C;
        float sp = 0.0f;
        for (int r = j; r < C; r++) sp += f2[r * LDF + j] * f1[r * LDF + j] * __expf(gam[r] - gam[j]);
        atomicAdd(&dgam[j], -(float)SCALE * sp);
    }
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        g1b[r * LDT + j] = __float2bfloat16(j <= r ? (float)SCALE * f2[r * LDF + j] * __expf(gam[r] - gam[j]) : 0.0f);
    }
    __syncthreads();
#endif
    // outputs: q, k staged over f1/f2 (dead now), one accumulator staged at a time
    bf* qs = (bf*)f1;
    bf* ks = (bf*)f2;
    for (int i = tid; i < C * DK / 8; i += 256) {
        const int r = i / (DK / 8), x = (i % (DK / 8)) * 8;
        *(uint4*)&qs[r * LDK + x] = *(const uint4*)&Q[(row0 + r) * DK + x];
        *(uint4*)&ks[r * LDK + x] = *(const uint4*)&K[(row0 + r) * DK + x];
    }
    __syncthreads();
    stg = stg1;
#ifndef XNOOUT
    // dk = G1ᵀ·q + (B+Bᵀ)·k + β e^γ·Tᵀ·dW + e^{γ_C−γ}·V'·dS_endᵀ   [C × DK]
    for (int tt = warp; tt < 4 * (DK / 16); tt += 8) {
        const int tr = tt & 3, tcol = tt >> 2;
        FC am, akb, akg;
        wmma::fill_fragment(am, 0.0f);
        wmma::fill_fragment(akb, 0.0f);
        wmma::fill_fragment(akg, 0.0f);
        #pragma unroll
        for (int e = 0; e < C; e += 16) {
            FAc a;
            FBr b;
            if (e >= tr * 16) {
                wmma::load_matrix_sync(a, g1b + e * LDT + tr * 16, LDT);
                wmma::load_matrix_sync(b, qs + e * LDK + tcol * 16, LDK);
                wmma::mma_sync(am, a, b, am);
            }
            FA a2;
            wmma::load_matrix_sync(a2, bsb + tr * 16 * LDT + e, LDT);
            wmma::load_matrix_sync(b, ks + e * LDK + tcol * 16, LDK);
            wmma::mma_sync(am, a2, b, am);
#if DELTA
            if (e >= tr * 16) {
                wmma::load_matrix_sync(a, tb + e * LDT + tr * 16, LDT);
                wmma::load_matrix_sync(b, dwb + e * LDK + tcol * 16, LDK);
                wmma::mma_sync(akb, a, b, akb);
            }
#endif
        }
        #pragma unroll
        for (int e = 0; e < DV; e += 16) {
            FA a;
            FBc b;
            wmma::load_matrix_sync(a, Vn + (row0 + tr * 16) * DV + e, DV);
            wmma::load_matrix_sync(b, dse + (size_t)(tcol * 16) * DV + e, DV);
            wmma::mma_sync(akg, a, b, akg);
        }
        float acc[8];
        wmma::store_matrix_sync(stg, am, 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int q = 0; q < 8; q++) acc[q] = stg[lane + 32 * q];
        __syncwarp();
        wmma::store_matrix_sync(stg, akb, 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const int e = lane + 32 * q;
            const int r = tr * 16 + e / 16, x = tcol * 16 + e % 16;
            const float eg = __expf(gam[r]), kb = stg[e];
            acc[q] += bet[r] * eg * kb;
            const float bk = hsum16(eg * kb * __bfloat162float(ks[r * LDK + x]));
            if ((lane & 15) == 0) {
                atomicAdd(&dbet[r], bk);
                atomicAdd(&dgam[r], bet[r] * bk);
            }
        }
        __syncwarp();
        wmma::store_matrix_sync(stg, akg, 16, wmma::mem_row_major);
        __syncwarp();
        float rsum = 0.0f;
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const int e = lane + 32 * q;
            const int r = tr * 16 + e / 16, x = tcol * 16 + e % 16;
            const float ec = __expf(glast - gam[r]), kg = stg[e];
            DK_[(row0 + r) * DK + x] = __float2bfloat16(acc[q] + ec * kg);
            const float rk = hsum16(ec * kg * __bfloat162float(ks[r * LDK + x]));
            if ((lane & 15) == 0) {
                atomicAdd(&dgam[r], -rk);
                rsum += rk;
            }
        }
        if ((lane & 15) == 0) atomicAdd(&red[1], rsum);
        __syncwarp();
    }
    // dq = scale·e^γ·dO·S₀ᵀ + G1·k   [C × DK]
    for (int tt = warp; tt < 4 * (DK / 16); tt += 8) {
        const int tr = tt & 3, tcol = tt >> 2;
        FC aq, ag;
        wmma::fill_fragment(aq, 0.0f);
        wmma::fill_fragment(ag, 0.0f);
        #pragma unroll
        for (int e = 0; e < DV; e += 16) {
            FA a;
            FBc b;
            wmma::load_matrix_sync(a, DO + (row0 + tr * 16) * DV + e, DV);
            wmma::load_matrix_sync(b, s0 + (size_t)(tcol * 16) * DV + e, DV);
            wmma::mma_sync(aq, a, b, aq);
        }
        for (int e = 0; e <= tr * 16; e += 16) {
            FA a;
            FBr b;
            wmma::load_matrix_sync(a, g1b + tr * 16 * LDT + e, LDT);
            wmma::load_matrix_sync(b, ks + e * LDK + tcol * 16, LDK);
            wmma::mma_sync(ag, a, b, ag);
        }
        float acc[8];
        wmma::store_matrix_sync(stg, ag, 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int q = 0; q < 8; q++) acc[q] = stg[lane + 32 * q];
        __syncwarp();
        wmma::store_matrix_sync(stg, aq, 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const int e = lane + 32 * q;
            const int r = tr * 16 + e / 16, x = tcol * 16 + e % 16;
            const float sg = (float)SCALE * __expf(gam[r]);
            DQ[(row0 + r) * DK + x] = __float2bfloat16(sg * stg[e] + acc[q]);
            const float t = hsum16(sg * stg[e] * __bfloat162float(qs[r * LDK + x]));
            if ((lane & 15) == 0) atomicAdd(&dgam[r], t);
        }
        __syncwarp();
    }
    // dv = β·Tᵀ·dV'   [C × DV]  (no delta: dv = β·dV')
    for (int tt = warp; tt < 4 * (DV / 16); tt += 8) {
        const int tr = tt & 3, tcol = tt >> 2;
        FC av;
        wmma::fill_fragment(av, 0.0f);
#if DELTA
        #pragma unroll
        for (int e = 0; e < C; e += 16) {
            if (e + 15 < tr * 16) continue;
            FAc a;
            FBr b;
            wmma::load_matrix_sync(a, tb + e * LDT + tr * 16, LDT);
            wmma::load_matrix_sync(b, DVn + (row0 + e) * DV + tcol * 16, DV);
            wmma::mma_sync(av, a, b, av);
        }
        wmma::store_matrix_sync(stg, av, 16, wmma::mem_row_major);
        __syncwarp();
#endif
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const int e = lane + 32 * q;
            const int r = tr * 16 + e / 16, x = tcol * 16 + e % 16;
#if DELTA
            const float dvb = stg[e];
#else
            const float dvb = __bfloat162float(DVn[(row0 + r) * DV + x]);
#endif
            DV_[(row0 + r) * DV + x] = __float2bfloat16(bet[r] * dvb);
            const float t = hsum16(dvb * __bfloat162float(V[(row0 + r) * DV + x]));
            if ((lane & 15) == 0) atomicAdd(&dbet[r], t);
        }
        __syncwarp();
    }
#endif
    __syncthreads();
    // d γ_C, then d log α = reverse running sum of dγ over the chunk
    if (warp == 0) {
        if (lane == 0) dgam[C - 1] += __expf(glast) * red[0] + red[1];
        __syncwarp();
        const float a = dgam[C - 1 - 2 * lane], b = dgam[C - 2 - 2 * lane];
        float s = a + b;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += n;
        }
        const float ex = s - a - b;
        // rows C−1−2·lane (gets ex + a) and C−2−2·lane (gets s)
        const int r1 = C - 1 - 2 * lane, r2 = C - 2 - 2 * lane;
#if GATE
        DA[row0 + r1] = (ex + a) / AL[row0 + r1];
        DA[row0 + r2] = s / AL[row0 + r2];
#endif
#if BETA
        DB[row0 + r1] = dbet[r1];
        DB[row0 + r2] = dbet[r2];
#endif
    }
}
"#;

impl ChunkedScan {
    fn defines(&self, save_t: bool) -> String {
        format!(
            "#define T {}\n#define C {CHUNK}\n#define DK {}\n#define DV {}\n#define DELTA {}\n#define GATE {}\n#define BETA {}\n#define SCALE {:e}\n#define SAVE_T {}\n",
            self.t,
            self.n,
            self.m,
            self.delta as u8,
            self.alpha.is_some() as u8,
            self.beta.is_some() as u8,
            self.scale,
            save_t as u8
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
        self.kernels_saving(bv, false)
    }

    /// [`kernels`](Self::kernels), the per-chunk kernel also writing `T`
    /// (f32 `[H, T/C, C, C]`, its last argument) for the backward.
    pub fn kernels_saving(&self, bv: usize, save_t: bool) -> [ScanLaunch; 3] {
        assert!(self.t % CHUNK == 0, "chunked scan: T = {} not a multiple of {CHUNK}", self.t);
        assert!(self.n % 64 == 0 && self.n <= 128, "chunked scan: n = {}", self.n);
        assert!(
            self.m % 64 == 0 && self.m % bv == 0 && bv % 16 == 0,
            "chunked scan: m = {}, bv {bv}",
            self.m
        );
        let d = self.defines(save_t);
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

impl ChunkedScan {
    /// Scratch of the backward, as element counts: f32 `T` `[H, T/C, C,
    /// C]` (written by [`kernels_saving`](Self::kernels_saving)), bf16
    /// `A_qk` `[H, T, C]`, bf16 `dS_end` per chunk `[H, T/C, n, m]`, bf16
    /// `dV'` `[H, T, m]`.
    pub fn backward_scratch(&self) -> [usize; 4] {
        let (rows, chunks) = (self.heads * self.t, self.heads * (self.t / CHUNK));
        [chunks * CHUNK * CHUNK, rows * CHUNK, chunks * self.n * self.m, rows * self.m]
    }

    /// The backward: the reverse scan of the state cotangent (derived:
    /// the transposed chunk map) and the per-chunk gradients. Arguments —
    /// `A_qk`: `q k γ | Aqk`; state: `q k W Aqk dO γ γ_C | dS_end dV'`;
    /// grad: `q k v α β γ γ_C T S dS_end dO V' dV' | dq dk dv dα dβ`
    /// (bf16 `dq dk dv`, f32 `dα dβ`; `V'` is `U` without the delta term). Needs the
    /// forward's `W`, states `S`, `V'`, `γ`, `γ_C`, and `T` from
    /// [`kernels_saving`](Self::kernels_saving).
    pub fn backward_kernels(&self, bv: usize) -> [ScanLaunch; 3] {
        assert!(self.n == 128 && self.m == 128, "chunked scan backward: n = m = 128 only");
        let d = self.defines(false);
        let c = CHUNK;
        let ldb = bv + 8;
        let smem0 = c * 4 + 4 * 256 * 4;
        let smem1 = (self.n * ldb + 2 * c * ldb) * 2 + c * 4 + 4 * 512 * 4;
        let smem2 = 2 * c * (c + 4) * 4 + c * (c + 8) * 4 + c * (c + 8) * 2 + c * (self.n + 8) * 2 + 4 * c * 4 + 16 + 8 * 256 * 4;
        let src = |smem: usize, extra: &str, body: &str| {
            let x = std::env::var("TANG_SCAN_DEFS").unwrap_or_default().replace(';', "\n");
            format!("#define TANG_SMEM {smem}\n{INCLUDES}{d}{x}\n{extra}{PRELUDE}{body}")
        };
        let chunks = ((self.t / c) as u32, self.heads as u32, 1);
        [
            ScanLaunch { source: src(smem0, "", BWD_AQK), grid: chunks, block: 128, smem: smem0 as u32 },
            ScanLaunch {
                source: src(smem1, &format!("#define BV {bv}\n"), BWD_STATE),
                grid: ((self.m / bv) as u32, self.heads as u32, 1),
                block: 128,
                smem: smem1 as u32,
            },
            ScanLaunch { source: src(smem2, "", BWD_GRAD), grid: chunks, block: 256, smem: smem2 as u32 },
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
