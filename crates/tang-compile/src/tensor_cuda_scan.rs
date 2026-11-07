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
    pub grid: (u32, u32),
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

/// Kernel 1: per chunk and head, `W`, `U`, `q e^{γ}`, `k e^{γ_C−γ}`, the
/// masked `q kᵀ` and `γ_C`. Grid `(T/C, H)`, 256 threads.
const PREP: &str = r#"
extern "C" __global__ void __launch_bounds__(256) main0(
    const bf* __restrict__ Q, const bf* __restrict__ K, const bf* __restrict__ V,
    const float* __restrict__ AL, const float* __restrict__ BE,
    bf* __restrict__ Wo, bf* __restrict__ Uo, bf* __restrict__ Qo, bf* __restrict__ Ko,
    bf* __restrict__ Ao, float* __restrict__ GC) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf* ks = (bf*)smem;
    bf* qs = ks + C * LDK;
    bf* vs = qs + C * LDK;
    float* sa = (float*)(vs + C * LDV);
    float* tm = sa + C * LDF;
    float* gam = tm + C * LDF;
    float* bet = gam + C;
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t row0 = (size_t)h * T + (size_t)c * C;
    for (int i = tid; i < C * DK / 8; i += 256) {
        const int r = i / (DK / 8), x = (i % (DK / 8)) * 8;
        *(uint4*)&ks[r * LDK + x] = *(const uint4*)&K[(row0 + r) * DK + x];
        *(uint4*)&qs[r * LDK + x] = *(const uint4*)&Q[(row0 + r) * DK + x];
    }
    for (int i = tid; i < C * DV / 8; i += 256) {
        const int r = i / (DV / 8), x = (i % (DV / 8)) * 8;
        *(uint4*)&vs[r * LDV + x] = *(const uint4*)&V[(row0 + r) * DV + x];
    }
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
    // γ: inclusive running sum of log α over the chunk (one warp, two per lane)
    if (warp == 0) {
        float a = gam[2 * lane], b = a + gam[2 * lane + 1];
        float s = b;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += n;
        }
        const float ex = s - b;
        gam[2 * lane] = ex + a;
        gam[2 * lane + 1] = s;
        if (lane == 31) GC[h * NT + c] = s;
    }
    __syncthreads();
    // Gram matrix out[i][j] = Σ_e A[i][e]·ks[j][e] (64×64, two tiles a warp)
    auto gram = [&](const bf* A, float* out) {
        for (int tt = warp; tt < 16; tt += 8) {
            const int tr = tt >> 2, tc = tt & 3;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBc b;
                wmma::load_matrix_sync(a, A + tr * 16 * LDK + e, LDK);
                wmma::load_matrix_sync(b, ks + tc * 16 * LDK + e, LDK);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(out + tr * 16 * LDF + tc * 16, acc, LDF, wmma::mem_row_major);
        }
    };
#if DELTA
    gram(ks, sa);
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        tm[r * LDF + j] = r == j ? 1.0f : 0.0f;
    }
    __syncthreads();
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        sa[r * LDF + j] = j < r ? bet[r] * sa[r * LDF + j] * __expf(gam[r] - gam[j]) : 0.0f;
    }
    __syncthreads();
    // T = (I + A)⁻¹, unit lower triangular: row r = e_r − Σ_{j<r} A[r][j]·T[j]
    for (int r = 1; r < C; r++) {
        if (tid < r) {
            float acc = 0.0f;
            for (int j = tid; j < r; j++) acc = fmaf(sa[r * LDF + j], tm[j * LDF + tid], acc);
            tm[r * LDF + tid] = -acc;
        }
        __syncthreads();
    }
#endif
    gram(qs, sa);
    __syncthreads();
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        const float v = j <= r ? sa[r * LDF + j] * __expf(gam[r] - gam[j]) * (float)SCALE : 0.0f;
        Ao[(row0 + r) * C + j] = __float2bfloat16(v);
    }
    for (int i = tid; i < C * DK; i += 256) {
        const int r = i / DK, x = i % DK;
        Qo[(row0 + r) * DK + x] = __float2bfloat16(__bfloat162float(qs[r * LDK + x]) * __expf(gam[r]) * (float)SCALE);
        Ko[(row0 + r) * DK + x] = __float2bfloat16(__bfloat162float(ks[r * LDK + x]) * __expf(gam[C - 1] - gam[r]));
    }
#if DELTA
    __syncthreads();
    bf* tb = qs;
    for (int i = tid; i < C * DK; i += 256) {
        const int r = i / DK, x = i % DK;
        ks[r * LDK + x] = __float2bfloat16(__bfloat162float(ks[r * LDK + x]) * bet[r] * __expf(gam[r]));
    }
    for (int i = tid; i < C * DV; i += 256) {
        const int r = i / DV, x = i % DV;
        vs[r * LDV + x] = __float2bfloat16(__bfloat162float(vs[r * LDV + x]) * bet[r]);
    }
    for (int i = tid; i < C * C; i += 256) {
        const int r = i / C, j = i % C;
        tb[r * LDT + j] = __float2bfloat16(tm[r * LDF + j]);
    }
    __syncthreads();
    float* stg = sa + warp * 256;
    auto tri = [&](const bf* B, int ldb, int ncols, bf* out) {
        for (int tt = warp; tt < 4 * (ncols / 16); tt += 8) {
            const int tr = tt & 3, tcol = tt >> 2;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int e = 0; e <= tr * 16; e += 16) {
                FA a;
                FBr b;
                wmma::load_matrix_sync(a, tb + tr * 16 * LDT + e, LDT);
                wmma::load_matrix_sync(b, B + e * ldb + tcol * 16, ldb);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                out[(row0 + tr * 16 + e / 16) * ncols + tcol * 16 + e % 16] = __float2bfloat16(stg[e]);
            __syncwarp();
        }
    };
    tri(ks, LDK, DK, Wo);
    tri(vs, LDV, DV, Uo);
#else
    for (int i = tid; i < C * DV; i += 256) {
        const int r = i / DV, x = i % DV;
        Uo[(row0 + r) * DV + x] = __float2bfloat16(__bfloat162float(vs[r * LDV + x]) * bet[r]);
    }
#endif
}
"#;

/// Kernel 2: the inter-chunk state scan with the readout fused. Grid
/// `(DV/BV, H)`, 128 threads; warp `w` holds state rows `32w..32w+32`
/// of the `DK × BV` block in accumulator fragments.
const STATE: &str = r#"
#define LDB (BV + 8)
#define NTC (BV / 16)
extern "C" __global__ void __launch_bounds__(128) main0(
    const bf* __restrict__ Wo, const bf* __restrict__ Uo, const bf* __restrict__ Qo,
    const bf* __restrict__ Ko, const bf* __restrict__ Ao, const float* __restrict__ GC,
    bf* __restrict__ O) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf* qc = (bf*)smem;
    bf* kc = qc + C * LDK;
    bf* wc = kc + C * LDK;
    bf* uc = wc + C * LDK;
    bf* ac = uc + C * LDB;
    bf* sb = ac + C * LDT;
    bf* vn = sb + DK * LDB;
    float* stg0 = (float*)(vn + C * LDB);
    const int vb = blockIdx.x * BV, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float* stg = stg0 + warp * 256;
    constexpr int RT = DK / 64;  // state row tiles a warp (DK / 16 over 4 warps)
    FC st[RT][NTC];
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < NTC; j++) wmma::fill_fragment(st[i][j], 0.0f);
    for (int c = 0; c < NT; c++) {
        const size_t row0 = (size_t)h * T + (size_t)c * C;
        for (int i = tid; i < C * DK / 8; i += 128) {
            const int r = i / (DK / 8), x = (i % (DK / 8)) * 8;
            *(uint4*)&qc[r * LDK + x] = *(const uint4*)&Qo[(row0 + r) * DK + x];
            *(uint4*)&kc[r * LDK + x] = *(const uint4*)&Ko[(row0 + r) * DK + x];
#if DELTA
            *(uint4*)&wc[r * LDK + x] = *(const uint4*)&Wo[(row0 + r) * DK + x];
#endif
        }
        for (int i = tid; i < C * BV / 8; i += 128) {
            const int r = i / (BV / 8), x = (i % (BV / 8)) * 8;
            *(uint4*)&uc[r * LDB + x] = *(const uint4*)&Uo[(row0 + r) * DV + vb + x];
        }
        for (int i = tid; i < C * C / 8; i += 128) {
            const int r = i / (C / 8), x = (i % (C / 8)) * 8;
            *(uint4*)&ac[r * LDT + x] = *(const uint4*)&Ao[(row0 + r) * C + x];
        }
        const float gc = __expf(GC[h * NT + c]);
        // the state as a bf16 operand
        #pragma unroll
        for (int i = 0; i < RT; i++)
            #pragma unroll
            for (int j = 0; j < NTC; j++) {
                wmma::store_matrix_sync(stg, st[i][j], 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = lane; e < 256; e += 32)
                    sb[((warp * RT + i) * 16 + e / 16) * LDB + j * 16 + e % 16] = __float2bfloat16(stg[e]);
                __syncwarp();
            }
        __syncthreads();
        // V' = U − W·S
        const bf* vp = uc;
#if DELTA
        for (int tt = warp; tt < 4 * NTC; tt += 4) {
            const int tr = tt & 3, tcol = tt >> 2;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBr b;
                wmma::load_matrix_sync(a, wc + tr * 16 * LDK + e, LDK);
                wmma::load_matrix_sync(b, sb + e * LDB + tcol * 16, LDB);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = tr * 16 + e / 16, x = tcol * 16 + e % 16;
                vn[r * LDB + x] = __float2bfloat16(__bfloat162float(uc[r * LDB + x]) - stg[e]);
            }
            __syncwarp();
        }
        __syncthreads();
        vp = vn;
#endif
        // o = (q e^γ)·S + tril(q kᵀ e^{γᵢ−γⱼ})·V'
        for (int tt = warp; tt < 4 * NTC; tt += 4) {
            const int tr = tt & 3, tcol = tt >> 2;
            FC acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int e = 0; e < DK; e += 16) {
                FA a;
                FBr b;
                wmma::load_matrix_sync(a, qc + tr * 16 * LDK + e, LDK);
                wmma::load_matrix_sync(b, sb + e * LDB + tcol * 16, LDB);
                wmma::mma_sync(acc, a, b, acc);
            }
            for (int e = 0; e <= tr * 16; e += 16) {
                FA a;
                FBr b;
                wmma::load_matrix_sync(a, ac + tr * 16 * LDT + e, LDT);
                wmma::load_matrix_sync(b, vp + e * LDB + tcol * 16, LDB);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stg, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                O[(row0 + tr * 16 + e / 16) * DV + vb + tcol * 16 + e % 16] = __float2bfloat16(stg[e]);
            __syncwarp();
        }
        // S ← e^{γ_C}·S + (k e^{γ_C−γ})ᵀ·V'
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
                    wmma::load_matrix_sync(a, kc + e * LDK + (warp * RT + i) * 16, LDK);
                    wmma::load_matrix_sync(b, vp + e * LDB + j * 16, LDB);
                    wmma::mma_sync(st[i][j], a, b, st[i][j]);
                }
            }
        __syncthreads();
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

    /// Scratch buffers between the two kernels, in kernel-1 output order
    /// (`W`, `U`, `q e^γ`, `k e^{γ_C−γ}`, `A_qk` bf16; `γ_C` f32), as
    /// element counts.
    pub fn scratch(&self) -> [usize; 6] {
        let rows = self.heads * self.t;
        [
            rows * self.n,
            rows * self.m,
            rows * self.n,
            rows * self.n,
            rows * CHUNK,
            self.heads * (self.t / CHUNK),
        ]
    }

    /// The two launches. `bv` is the state column block of kernel 2 (16,
    /// 32 or 64; more blocks, less reuse).
    pub fn kernels(&self, bv: usize) -> [ScanLaunch; 2] {
        assert!(self.t % CHUNK == 0, "chunked scan: T = {} not a multiple of {CHUNK}", self.t);
        assert!(self.n % 64 == 0 && self.n <= 128, "chunked scan: n = {}", self.n);
        assert!(self.m % bv == 0 && self.m <= 128 && bv % 16 == 0, "chunked scan: m = {}, bv {bv}", self.m);
        let d = self.defines();
        let (c, n, m) = (CHUNK, self.n, self.m);
        let smem1 = (2 * c * (n + 8) + c * (m + 8)) * 2 + 2 * c * (c + 4) * 4 + 2 * c * 4;
        let ldb = bv + 8;
        let smem2 = (3 * c * (n + 8) + c * ldb + c * (c + 8) + n * ldb + c * ldb) * 2 + 4 * 256 * 4;
        let prep = format!("#define TANG_SMEM {smem1}\n{INCLUDES}{d}{PRELUDE}{PREP}");
        let state = format!("#define TANG_SMEM {smem2}\n{INCLUDES}{d}#define BV {bv}\n{PRELUDE}{STATE}");
        [
            ScanLaunch {
                source: prep,
                grid: ((self.t / c) as u32, self.heads as u32),
                block: 256,
                smem: smem1 as u32,
            },
            ScanLaunch {
                source: state,
                grid: ((m / bv) as u32, self.heads as u32),
                block: 128,
                smem: smem2 as u32,
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
