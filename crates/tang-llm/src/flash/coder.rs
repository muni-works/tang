//! The Qwen3-Coder-Next / Qwen3-Next decode engine on CUDA (GGUF architecture `qwen3next`).
//!
//! The same window model as the Flash-Next [`Engine`](super::engine::Engine): `t` = 1..=8
//! consecutive tokens at positions `pos0..pos0 + t`, plain windows commit every token's GDN
//! recurrence in the window, verify windows only read it and [`Coder::commit`] replays the kept
//! prefix. What differs is the model:
//!
//! - one residual stream with pre-norms (`r += mixer(rmsnorm(r))`, `r += moe(rmsnorm(r))`)
//!   instead of hyper-connections;
//! - full-attention layers are gated GQA over every earlier cell (no indexer), split-K;
//! - every routed expert is VRAM-resident in its GGUF format (an 80 GB card holds the
//!   UD-Q4_K_XL file), so there is no doorbell, no CPU miss path and no PLE;
//! - GDN v heads are grouped over k heads (`h / r`) and gated with SiLU (the flash module
//!   compiled for [`Geometry`] with `gdn_grouped`).
//!
//! Dense weights stay in their GGUF types through the native GEMV (`kernels::GEMV_SRC`,
//! int8 activations), so the engine reads exactly the bytes `flash-ref` and llama.cpp read.
//!
//! Per layer, on the stream:
//!
//! ```text
//! rmsnorm → stacked input GEMV → GDN conv + step | attention prep + split-K + gated merge
//! → output GEMV → r += mix, rmsnorm → router GEMV (+ shared gate row) → top-10
//! → int8 → routed gate|up → silu·up → int8 → routed down ‖ shared expert → r += Σ w·y + g·shared
//! ```

use super::coder_kernels::{self as ck, Shapes};
use super::engine::{Probe, WinStats};
use super::kernels;
use super::reference::{Arch, Hparams};
use crate::gguf::{self, GgmlType, Gguf, TensorInfo};
use anyhow::{anyhow, ensure, Context, Result};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::time::Instant;
use tang_compute::cuda::CudaBuffer;
use tang_compute::flash::{self as fl, Geometry};
use tang_compute::{ComputeDevice, CudaComputeDevice};
use tang_moe::arena::{ArenaOptions, HostArena};
use tang_moe::gpu::{self, Gpu, Graph, Module, Stream};

type B = CudaBuffer;
type Fun = cudarc::driver::sys::CUfunction;
/// A window's logits as they are computed: `(pos0, t, [t][vocab])`.
pub type LogitsCb<'a> = &'a mut dyn FnMut(usize, usize, &[f32]);

/// Widest window.
pub const MAX_T: usize = fl::MAX_T;
/// Sequences one engine holds at once (slot 0 is the one the single-sequence API uses).
pub const MAX_SLOTS: usize = 128;
/// u32s of the per-row record staged with each window: positions, slots, then segments.
/// u32s of the per-row record per window row (positions, slots, segments).
const ROWS_PER_T: usize = 6;
const ROWS_U32: usize = ROWS_PER_T * MAX_T;

/// Options of a [`Coder`] load.
#[derive(Clone, Debug)]
pub struct CoderOpts {
    pub max_ctx: usize,
    /// An MTP sidecar GGUF (`scripts/qwen3next_mtp_sidecar.py`): its one MTP layer drafts the
    /// next tokens. Default: `TANG_CODER_MTP`.
    pub mtp: Option<std::path::PathBuf>,
}

impl Default for CoderOpts {
    fn default() -> Self {
        CoderOpts {
            max_ctx: 8192,
            mtp: None,
        }
    }
}

/// `TANG_CODER_GEMV=v1`: dense GEMVs on `fe_gemv8` (two rows a warp) instead of the K-split
/// `cx_gemvk` (A/B).
/// Streaming multiprocessors on device 0 (sizes the FA2 key splits).
fn sm_count() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let mut v = 0i32;
        unsafe {
            cudarc::driver::sys::cuDeviceGetAttribute(
                &mut v,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                0,
            )
        };
        if v > 0 {
            v as usize
        } else {
            132
        }
    })
}

fn env_is(key: &str, val: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v == val)
}

/// Kernel variants for A/B on one build (the defaults are the measured winners; the losers are
/// in `docs/speed-log.md` under Dead ends): `TANG_CODER_KQ` warps a row in the K-split GEMV
/// (1, 2, 4 or 8; default 2), `TANG_CODER_FASTK` the K-quant expert dot (0: 8 weights a lane,
/// 1: 32 a lane, 2: also two down rows a warp; default 2), `TANG_CODER_SHPAR=1` the shared
/// expert on a parallel graph branch.
#[derive(Clone, Copy, Debug)]
struct Variants {
    kq: usize,
    fastk: i32,
    shared_par: bool,
    /// Grouped expert kernels (each distinct expert read once a window), `TANG_CODER_GROUPED=1`.
    /// Bitwise the same outputs as the per-pair ones. Off by default: at decode widths (≤ 8 rows ×
    /// top-10 of 512 experts) pairs rarely share an expert, and on an H200 the plan and the
    /// serialized groups cost more than they save (8 streams: 10.98 ms a step per-pair, 12.49
    /// grouped). They're for wide windows (prefill), where many rows share experts.
    grouped: bool,
    /// Per-pair expert kernels: 0 stage rows in shared memory (`cx_moe_gu` / `cx_moe_down`),
    /// 2 or 4 read them from global memory with that many rows a warp (`_d2` / `_d4`), 8
    /// double-buffer them with cp.async (`_db`); `usize::MAX` (the default) picks by window:
    /// `_d2` up to 4 rows, `_d4` past that. All bitwise equal. On an H200 (UD-Q4_K_XL, 1k
    /// context), step ms at 1/2/4/8 streams: staged 4.08/5.03/6.92/11.00, d2 3.82/4.58/6.09/9.41,
    /// d4 4.04/4.84/6.15/9.12, db 4.63/5.52/7.34/10.57.
    /// `TANG_CODER_MOE`. Bitwise the same outputs.
    moe: usize,
    /// SiLU(gate) * up and its int8 made inside the direct down kernels (`_d2f` / `_d4f`) instead
    /// of a `cx_silu_q` launch a layer; `TANG_CODER_FUSE_SILU=0` keeps the launch. Bitwise equal.
    fuse_silu: bool,
    /// Wide windows run the experts as a tensor-core GEMM (`cx_moe_gemm_*`) instead of the
    /// grouped decode kernels; `TANG_CODER_WGEMM=0` for those (bitwise narrow).
    wgemm: bool,
    /// Which GEMMs (`TANG_CODER_WGEMM=gu` / `down`; anything else: both). For localizing.
    wgemm_gu: bool,
    wgemm_down: bool,
    /// Wide windows run attention as the FA2-style tensor-core kernel (`cx_attn_fa2`, then
    /// `cx_attn_merge` over its key splits) instead of split-K part/merge;
    /// the default (H200: 1.66× at 30k, 2.15× at 54k); `TANG_CODER_ATTN=splitk` for split-K.
    /// Not bitwise: gated by `flash-wide-check`'s greedy + KL bar.
    attn_fa2: bool,
    /// Batch decode windows (past `MAX_T` rows) run the dense layers and experts as the
    /// tensor-core GEMMs too (`TANG_CODER_BGEMM=1`). Not bitwise; off by default.
    bgemm: bool,
    /// Batch windows run the streaming grouped expert kernels (`cx_moe_*_s`);
    /// `TANG_CODER_MOE_STREAM=0` for the 16-rows-a-block grouped ones. Bitwise equal.
    stream: bool,
    /// Batch windows run the experts on int8 tensor cores (`cx_moe_*_m`; not bitwise, gated
    /// by batch-check's tolerance mode): `TANG_CODER_MOE_MMA=down|gu|all|0` (default down; gate|up is no faster).
    mma_gu: bool,
    mma_down: bool,
    /// Batch windows run Q8X dense matrices on int8 tensor cores (`cx_mma_q8x`; not bitwise),
    /// `TANG_CODER_DMMA=0` for the K-split GEMVs.
    dmma: bool,
    /// Narrow windows of 2..8 rows run the streaming grouped expert kernels
    /// (`TANG_CODER_NSTREAM=1`). Bitwise.
    nstream: bool,
}

impl Default for Variants {
    fn default() -> Self {
        Variants {
            kq: 2,
            fastk: 2,
            shared_par: false,
            grouped: false,
            moe: usize::MAX,
            fuse_silu: true,
            wgemm: true,
            wgemm_gu: true,
            wgemm_down: true,
            attn_fa2: false,
            bgemm: false,
            stream: true,
            mma_gu: true,
            mma_down: true,
            dmma: true,
            nstream: false,
        }
    }
}

impl Variants {
    fn from_env() -> Self {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<i64>().ok());
        let d = Variants::default();
        Variants {
            kq: num("TANG_CODER_KQ").map_or(d.kq, |v| v as usize),
            fastk: num("TANG_CODER_FASTK").map_or(d.fastk, |v| v as i32),
            shared_par: env_is("TANG_CODER_SHPAR", "1"),
            grouped: env_is("TANG_CODER_GROUPED", "1"),
            moe: num("TANG_CODER_MOE").map_or(d.moe, |v| v as usize),
            fuse_silu: !env_is("TANG_CODER_FUSE_SILU", "0"),
            wgemm: !env_is("TANG_CODER_WGEMM", "0"),
            wgemm_gu: !env_is("TANG_CODER_WGEMM", "down"),
            wgemm_down: !env_is("TANG_CODER_WGEMM", "gu"),
            attn_fa2: !env_is("TANG_CODER_ATTN", "splitk"),
            bgemm: env_is("TANG_CODER_BGEMM", "1"),
            stream: !env_is("TANG_CODER_MOE_STREAM", "0"),
            dmma: !env_is("TANG_CODER_DMMA", "0"),
            nstream: env_is("TANG_CODER_NSTREAM", "1"),
            mma_gu: {
                let m = std::env::var("TANG_CODER_MOE_MMA").unwrap_or_else(|_| "down".into());
                m == "all" || m == "gu"
            },
            mma_down: {
                let m = std::env::var("TANG_CODER_MOE_MMA").unwrap_or_else(|_| "down".into());
                m == "all" || m == "down"
            },
        }
    }
}

fn gemv_v1() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TANG_CODER_GEMV").is_ok_and(|v| v == "v1"))
}

/// The ggml type id the native GEMV decodes, if it does.
fn native_id(ty: GgmlType) -> Option<u32> {
    use GgmlType::*;
    Some(match ty {
        F32 => 0,
        Q4_0 => 2,
        Q5_0 => 6,
        Q8_0 => 8,
        Q3K => 11,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        Iq4Nl => 20,
        Iq4Xs => 23,
        Bf16 => 30,
        Q2_0 => 42,
        _ => return None,
    })
}

/// Q8_0 blocks (f16 scale, 32 int8 codes) of `v` (`v.len() % 32 == 0`).
fn quantize_q8_0(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() / 32 * 34);
    for b in v.chunks(32) {
        let amax = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&gguf::f32_to_f16(d).to_le_bytes());
        for &x in b {
            let q = if d == 0.0 {
                0
            } else {
                (x / d).round().clamp(-127.0, 127.0) as i8
            };
            out.push(q as u8);
        }
    }
    out
}

/// A dense weight for the native GEMV: one or more GGUF tensors stacked by rows.
/// The MTP layer's index in [`Coder::lay`] (it lives outside `layers`).
const MTP_L: usize = usize::MAX;

/// The MTP layer's input (`cx_mtp_in`: `[rmsnorm(emb) * we | rmsnorm(h) * wh]` per row, grid
/// `(t, 2)`) and its pick (`cx_mtp_pick`: argmax and its softmax probability per row of logits,
/// grid `t`).
const MTP_SRC: &str = r#"
#define MTP_ALT 8
__device__ float mtp_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    if (threadIdx.x < 32)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    return red[0];
}

extern "C" __global__ void __launch_bounds__(1024) cx_mtp_in(const float* emb, const float* h,
        const float* we, const float* wh, float eps, int n, float* out) {
    __shared__ float red[32];
    const int c = blockIdx.x, half = blockIdx.y;
    const float* src = (half ? h : emb) + (size_t)c * n;
    const float* w = half ? wh : we;
    float* dst = out + (size_t)c * 2 * n + (size_t)half * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += src[i] * src[i];
    const float r = rsqrtf(mtp_block_sum(ss, red) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i] * r * w[i];
}

__device__ void mtp_argmax(const float* row, int v, int skip, float* bm, int* bi) {
    float m = -3.0e38f;
    int mi = 0;
    for (int i = threadIdx.x; i < v; i += blockDim.x)
        if (i != skip && row[i] > m) { m = row[i]; mi = i; }
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, m, o);
        const int oi = __shfl_xor_sync(0xffffffffu, mi, o);
        if (om > m || (om == m && oi < mi)) { m = om; mi = oi; }
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) { bm[threadIdx.x >> 5] = m; bi[threadIdx.x >> 5] = mi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (blockDim.x >> 5); w++)
            if (bm[w] > bm[0] || (bm[w] == bm[0] && bi[w] < bi[0])) { bm[0] = bm[w]; bi[0] = bi[w]; }
    }
    __syncthreads();
}

// Per row: the argmax and its probability, and the runner-up (`ids[MAX_T + row]`,
// `prob[MAX_T + row]`).
extern "C" __global__ void __launch_bounds__(1024) cx_mtp_pick(const float* lg, int v,
        unsigned* ids, float* prob) {
    __shared__ float red[32];
    __shared__ float bm[32];
    __shared__ int bi[32];
    const float* row = lg + (size_t)blockIdx.x * v;
    mtp_argmax(row, v, -1, bm, bi);
    const float top = bm[0];
    const int i1 = bi[0];
    __syncthreads();
    mtp_argmax(row, v, i1, bm, bi);
    const float second = bm[0];
    const int i2 = bi[0];
    float se = 0.f;
    for (int i = threadIdx.x; i < v; i += blockDim.x) se += __expf(row[i] - top);
    se = mtp_block_sum(se, red);
    if (threadIdx.x == 0) {
        ids[blockIdx.x] = (unsigned)i1;
        prob[blockIdx.x] = 1.f / se;
        ids[MTP_ALT + blockIdx.x] = (unsigned)i2;
        prob[MTP_ALT + blockIdx.x] = __expf(second - top) / se;
    }
}
"#;

/// The multi-token-prediction layer (an MTP sidecar): a gated-attention MoE layer whose input
/// is `fc([rmsnorm(embed(next token)) | rmsnorm(final hidden)])`, its output normed and put
/// through the model's head. Run after each window on the window's final hidden rows, then
/// chained on its own output, it drafts the next tokens.
struct Mtp {
    layer: Layer,
    fc: Dense,
    enorm: B,
    hnorm: B,
    onorm: B,
    /// Final hidden rows (normed): the last window's, then the MTP layer's own last row.
    mh: B,
    emb: B,
    cat: B,
    ids: B,
    prob: B,
    f_in: Fun,
    f_pick: Fun,
    _m: Module,
}

struct Dense {
    _w: Vec<B>,
    segs: B,
    nseg: i32,
    rows: usize,
    k: usize,
    row16: i32,
    /// Weight bytes in VRAM.
    bytes: usize,
    /// Whether the wide-window GEMM (`cx_gemm_dense`) takes it: every segment a type it
    /// decodes, 16-row aligned.
    gemm_ok: bool,
    /// Every segment Q8X (type 108), 16-row aligned, k a multiple of 256: batch windows can
    /// run it on int8 tensor cores (`cx_mma_q8x`).
    q8x_ok: bool,
    /// With `q8x_ok`: the Q8X segments' own table (segments, count, rows) and the other
    /// segments as a second matrix.
    mma: Option<(B, i32, usize)>,
    rest: Option<Box<Dense>>,
    /// Only F32 / BF16 segments (the router and the shared expert's gate): batch windows can run
    /// `cx_router_b` (`TANG_CODER_ROUTERB=1`; slower than the K-split GEMV so far: 1.89 vs
    /// 1.64 ms a step at B=64).
    float1: bool,
}

/// A routed-expert tensor (`[n_expert][rows][k]`, GGUF type) in VRAM, as the kernels take it.
#[repr(C)]
#[derive(Clone, Copy)]
struct ExW {
    base: u64,
    estride: u64,
    ty: i32,
    row_bytes: i32,
}

struct Experts {
    _w: B,
    w: ExW,
}

struct GdnL {
    conv: B,
    dt: B,
    a: B,
    norm: B,
    state: B,
    hist: B,
    /// The window's stacked projection and conv output, kept for a later commit.
    proj: B,
    /// Every slot's `state` and `hist` addresses (u64 each; slot 0: the two above).
    stab: B,
    htab: B,
}

struct AttnL {
    qn: B,
    kn: B,
    k: B,
    v: B,
    proj: B,
    /// Every slot's K and V cache addresses (u64 each; slot 0: `k` and `v`).
    ktab: B,
    vtab: B,
}

struct Layer {
    attn_norm: B,
    post_norm: B,
    w_in: Dense,
    w_out: Dense,
    router: Dense,
    sh_gate: Experts,
    sh_up: Experts,
    sh_down: Experts,
    /// Shared-memory 16-byte units a staged expert row needs (gate|up, down).
    r16_gu: i32,
    r16_down: i32,
    gate: Experts,
    up: Experts,
    down: Experts,
    gdn: Option<GdnL>,
    attn: Option<AttnL>,
}

struct Kern {
    _mods: Vec<Module>,
    q8: Fun,
    gemv8: [Fun; MAX_T],
    gemvk: [Fun; MAX_T],
    gemvk8: [Fun; MAX_T],
    gemvk2: [Fun; MAX_T],
    gemvk1: [Fun; MAX_T],
    /// Batch modules only (null elsewhere): `cx_gemvkb{Q}_t{T}`, by Q (1, 2, 4, 8) and T
    /// (8, 16, 32), over rows of a wider window.
    gemvkb: [[Fun; 3]; 4],
    norm_q8: Fun,
    route: Fun,
    moe_gu: Fun,
    silu_q: Fun,
    moe_down: Fun,
    moe_plan: Fun,
    moe_gu_d: [Fun; 2],
    moe_down_d: [Fun; 2],
    moe_down_df: [Fun; 2],
    moe_gu_db: Fun,
    moe_down_db: Fun,
    moe_gu_g: Fun,
    moe_down_g: Fun,
    /// Streaming grouped experts (batch windows): `cx_moe_gu_s` / `cx_moe_down_s`.
    moe_gu_s: Fun,
    moe_down_s: Fun,
    /// Int8 tensor-core grouped experts (batch windows, not bitwise): `cx_moe_gu_m` / `_down_m`.
    moe_gu_m: Fun,
    moe_down_m: Fun,
    /// Int8 tensor-core dense Q8X (batch windows): the GEMM and its split reduction.
    mma_q8x: Fun,
    mma_q8x_red: Fun,
    router_b: Fun,
    /// Tensor-core grouped expert GEMM (wide windows).
    moe_gemm_gu: Fun,
    moe_gemm_down: Fun,
    gemm_dense: Fun,
    moe_add_norm: Fun,
    prep: Fun,
    part: Fun,
    merge: Fun,
    fa2: Fun,
    gdn_step: Fun,
    gdn_commit: Fun,
    gdn_step_seg: Fun,
    gdn_commit_seg: Fun,
    copy: Fun,
    argmax1: Fun,
    argmax2: Fun,
}

impl Kern {
    /// The Coder kernels (`cm`'s) from another build of the Coder source (the wide prefill
    /// module, with more rows a window), the flash and engine kernels shared with `self`.
    fn with_coder_module(&self, cm: Module) -> Result<Kern> {
        let f = |n: &str| cm.func(n).map_err(|e| anyhow!("{n}: {e}"));
        let k = Kern {
            q8: f("fe_q8")?,
            gemv8: self.gemv8,
            gemvk: self.gemvk,
            gemvk1: self.gemvk1,
            gemvk2: self.gemvk2,
            gemvk8: self.gemvk8,
            gemvkb: {
                let mut a = [[std::ptr::null_mut(); 3]; 4];
                for (qi, q) in [1, 2, 4, 8].into_iter().enumerate() {
                    for (ti, t) in [8, 16, 32].into_iter().enumerate() {
                        a[qi][ti] = f(&format!("cx_gemvkb{q}_t{t}"))?;
                    }
                }
                a
            },
            norm_q8: f("cx_norm_q8")?,
            route: f("cx_route")?,
            moe_gu: f("cx_moe_gu")?,
            silu_q: f("cx_silu_q")?,
            moe_down: f("cx_moe_down")?,
            moe_plan: f("cx_moe_plan_w")?,
            moe_gu_d: [f("cx_moe_gu_d2")?, f("cx_moe_gu_d4")?],
            moe_down_d: [f("cx_moe_down_d2")?, f("cx_moe_down_d4")?],
            moe_down_df: [f("cx_moe_down_d2f")?, f("cx_moe_down_d4f")?],
            moe_gu_db: f("cx_moe_gu_db")?,
            moe_down_db: f("cx_moe_down_db")?,
            moe_gu_g: f("cx_moe_gu_g")?,
            moe_down_g: f("cx_moe_down_g")?,
            moe_gu_s: f("cx_moe_gu_s")?,
            moe_down_s: f("cx_moe_down_s")?,
            moe_gu_m: f("cx_moe_gu_m")?,
            moe_down_m: f("cx_moe_down_m")?,
            mma_q8x: f("cx_mma_q8x")?,
            mma_q8x_red: f("cx_mma_q8x_red")?,
            router_b: f("cx_router_b")?,
            moe_gemm_gu: f("cx_moe_gemm_gu")?,
            moe_gemm_down: f("cx_moe_gemm_down")?,
            gemm_dense: f("cx_gemm_dense")?,
            moe_add_norm: f("cx_moe_add_norm")?,
            prep: f("cx_attn_prep")?,
            part: f("cx_attn_part")?,
            merge: f("cx_attn_merge")?,
            fa2: f("cx_attn_fa2")?,
            gdn_step: self.gdn_step,
            gdn_commit: self.gdn_commit,
            gdn_step_seg: self.gdn_step_seg,
            gdn_commit_seg: self.gdn_commit_seg,
            copy: self.copy,
            argmax1: self.argmax1,
            argmax2: self.argmax2,
            _mods: vec![cm],
        };
        k.set_smem()?;
        Ok(k)
    }

    /// The engine kernels compiled on `gpu`'s device for `rows`-wide windows: the Coder
    /// module (GEMV/MoE/attention), the flash module's GDN kernels and the shared engine
    /// module (copy/argmax). Every device of a layer split compiles its own handle set
    /// (module handles are per context).
    fn build_on(gpu: &Gpu, sh: &Shapes, geo: &Geometry, rows: usize) -> Result<Kern> {
        let src = format!("{}{}{}", ck::prelude(sh, rows), kernels::GEMV_SRC, ck::SRC);
        let cm = gpu
            .module(&src)
            .map_err(|e| anyhow!("coder kernels: {e}"))?;
        let fm = gpu
            .module(&format!("#define GDN_SLOTS 1\n{}", geo.cuda_source()))
            .map_err(|e| anyhow!("flash kernels: {e}"))?;
        let em = gpu.module(kernels::SRC).map_err(|e| anyhow!("{e}"))?;
        let k = Kern::from_modules(&cm, &fm, &em)?;
        // Keep the modules alive for as long as the handles: dropping a CudaModule unloads
        // the cubin and every function handle in it dies with it.
        let mut k = k;
        k._mods = vec![cm, fm, em];
        Ok(k)
    }
}

impl Kern {
    /// The `Kern` for compiled modules `cm` (Coder), `fm` (flash GDN) and `em` (engine):
    /// every function handle, and the big-smem attributes set.
    fn from_modules(cm: &Module, fm: &Module, em: &Module) -> Result<Kern> {
        let f = |m: &Module, n: &str| m.func(n).map_err(|e| anyhow!("{n}: {e}"));
        let mut gemv8 = [std::ptr::null_mut(); MAX_T];
        for (i, slot) in gemv8.iter_mut().enumerate() {
            *slot = f(cm, &format!("fe_gemv8_t{}", i + 1))?;
            unsafe {
                gpu::check(
                    cudarc::driver::sys::cuFuncSetAttribute(
                        *slot,
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        99 * 1024,
                    ),
                    "cuFuncSetAttribute",
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }
        let mut gemvk = [std::ptr::null_mut(); MAX_T];
        for (i, slot) in gemvk.iter_mut().enumerate() {
            *slot = f(cm, &format!("cx_gemvk_t{}", i + 1))?;
        }
        let sized = |names: [String; MAX_T]| -> Result<[Fun; MAX_T]> {
            let mut a: [Fun; MAX_T] = [std::ptr::null_mut(); MAX_T];
            for (i, name) in names.iter().enumerate() {
                a[i] = f(cm, name)?;
            }
            Ok(a)
        };
        let k = Kern {
            q8: f(cm, "fe_q8")?,
            gemv8,
            gemvk,
            gemvk1: sized(std::array::from_fn(|i| format!("cx_gemvk1_t{}", i + 1)))?,
            gemvk2: sized(std::array::from_fn(|i| format!("cx_gemvk2_t{}", i + 1)))?,
            gemvk8: sized(std::array::from_fn(|i| format!("cx_gemvk8_t{}", i + 1)))?,
            gemvkb: [[std::ptr::null_mut(); 3]; 4],
            norm_q8: f(cm, "cx_norm_q8")?,
            route: f(cm, "cx_route")?,
            moe_gu: f(cm, "cx_moe_gu")?,
            silu_q: f(cm, "cx_silu_q")?,
            moe_down: f(cm, "cx_moe_down")?,
            moe_plan: f(cm, "cx_moe_plan")?,
            moe_gu_d: [f(cm, "cx_moe_gu_d2")?, f(cm, "cx_moe_gu_d4")?],
            moe_down_d: [f(cm, "cx_moe_down_d2")?, f(cm, "cx_moe_down_d4")?],
            moe_down_df: [f(cm, "cx_moe_down_d2f")?, f(cm, "cx_moe_down_d4f")?],
            moe_gu_db: f(cm, "cx_moe_gu_db")?,
            moe_down_db: f(cm, "cx_moe_down_db")?,
            moe_gu_g: f(cm, "cx_moe_gu_g")?,
            moe_down_g: f(cm, "cx_moe_down_g")?,
            moe_gu_s: f(cm, "cx_moe_gu_s")?,
            moe_down_s: f(cm, "cx_moe_down_s")?,
            moe_gu_m: f(cm, "cx_moe_gu_m")?,
            moe_down_m: f(cm, "cx_moe_down_m")?,
            mma_q8x: f(cm, "cx_mma_q8x")?,
            mma_q8x_red: f(cm, "cx_mma_q8x_red")?,
            router_b: f(cm, "cx_router_b")?,
            moe_gemm_gu: f(cm, "cx_moe_gemm_gu")?,
            moe_gemm_down: f(cm, "cx_moe_gemm_down")?,
            gemm_dense: f(cm, "cx_gemm_dense")?,
            moe_add_norm: f(cm, "cx_moe_add_norm")?,
            prep: f(cm, "cx_attn_prep")?,
            part: f(cm, "cx_attn_part")?,
            merge: f(cm, "cx_attn_merge")?,
            fa2: f(cm, "cx_attn_fa2")?,
            gdn_step: f(fm, "fl_gdn_conv_step")?,
            gdn_commit: f(fm, "fl_gdn_conv_commit")?,
            gdn_step_seg: f(fm, "fl_gdn_conv_step_seg")?,
            gdn_commit_seg: f(fm, "fl_gdn_conv_commit_seg")?,
            copy: f(em, "fe_copy")?,
            argmax1: f(em, "fe_argmax1")?,
            argmax2: f(em, "fe_argmax2")?,
            _mods: vec![],
        };
        k.set_smem()?;
        Ok(k)
    }

    /// Up to 200 KB of staged expert rows, capped at what the device allows a block (99 KB on
    /// sm_86/89, 227 KB on sm_90), so the engine also loads on smaller GPUs.
    fn set_smem(&self) -> Result<()> {
        let smem_max = {
            let mut v = 0i32;
            unsafe {
                cudarc::driver::sys::cuDeviceGetAttribute(
                    &mut v,
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    0,
                )
            };
            if v > 0 {
                v.min(200 * 1024)
            } else {
                99 * 1024
            }
        };
        for f in [
            self.moe_gu,
            self.moe_down,
            self.moe_gu_g,
            self.moe_down_g,
            self.moe_gu_s,
            self.moe_down_s,
            self.moe_gu_m,
            self.moe_down_m,
            self.moe_gu_db,
            self.moe_down_db,
            self.fa2,
        ] {
            unsafe {
                gpu::check(
                    cudarc::driver::sys::cuFuncSetAttribute(
                        f,
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem_max,
                    ),
                    "cuFuncSetAttribute",
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }
        Ok(())
    }
}

/// Per-window scratch, sized for `MAX_T`.
struct Scratch {
    ctl: B,
    /// Per-row record (`ROWS_U32`): row positions, row slots, then `[start, len, slot, keep]`
    /// per segment (a slot's consecutive rows).
    rows: B,
    r: B,
    x: B,
    xq8: B,
    mxq: B,
    mix: B,
    y: B,
    q: B,
    part: B,
    attn: B,
    lg: B,
    ids: B,
    w: B,
    egu: B,
    hq: B,
    /// Grouped-expert plans (`cx_moe_plan`): one for the main stream, one for the side stream.
    plan: B,
    parts: B,
    moe_y: B,
    head: B,
    out_ids: B,
    amax: B,
}

impl Scratch {
    /// Window scratch for windows of up to `rows` rows (`MAX_T`, or the wide prefill width),
    /// attention partials for `attn_rows` of them at a time.
    #[allow(clippy::too_many_arguments)]
    fn new(
        dev: &CudaComputeDevice,
        hp: &Hparams,
        geo: &Geometry,
        kmax: usize,
        max_ctx: usize,
        rows: usize,
        attn_rows: usize,
        head_rows: usize,
    ) -> Self {
        let np = rows * (hp.n_expert_used + 1);
        let nchunk_max = max_ctx / ck::attn_chunk();
        let z = |n: usize| dev.alloc_f32(n);
        let xq_words = |k: usize| (rows * k + rows * k / 32 * 4).div_ceil(4);
        let n = hp.n_embd;
        Scratch {
            ctl: z(8),
            rows: z(ROWS_PER_T * rows),
            r: z(rows * n),
            x: z(rows * n),
            // The dense GEMVs' own int8 staging: at most MAX_T rows a call, whatever the window.
            xq8: z((MAX_T * kmax + MAX_T * kmax / 32 * 4).div_ceil(4)),
            mxq: z(xq_words(n)),
            mix: z(rows * n),
            y: z(rows * geo.gdn_v()),
            q: z(rows * hp.n_head * hp.head_dim),
            part: z(attn_rows
                * hp.n_head_kv
                * nchunk_max
                * (hp.n_head / hp.n_head_kv)
                * (hp.head_dim + 2)),
            attn: z(rows * hp.n_head * hp.head_dim),
            lg: z(rows * (hp.n_expert + 1)),
            ids: z(rows * hp.n_expert_used),
            w: z(rows * hp.n_expert_used),
            egu: z(np * 2 * hp.n_ff_exp),
            plan: z(2 * (2 + 2 * np)),
            hq: z((np * hp.n_ff_exp + np * hp.n_ff_exp / 32 * 4).div_ceil(4)),
            parts: z(np * n),
            moe_y: z(rows * n),
            head: z(head_rows * hp.n_vocab),
            out_ids: z(head_rows),
            amax: z(head_rows * 64 * 2),
        }
    }
}

/// Pinned staging: control words and embeddings in, sampled ids out.
struct Io {
    arena: HostArena,
}

impl Io {
    const CTL: usize = 0;
    const CCTL: usize = 64;
    const IDS: usize = 128;
    const EMB: usize = 256;
    /// The MTP pick's probabilities (`2 * MAX_T` f32), past the control words' slack.
    const MTP_PROB: usize = 192;

    /// Bytes for windows of up to `rows` rows: control words, embeddings, the row record.
    fn bytes(hidden: usize, rows: usize) -> usize {
        Self::rows(hidden, rows) + ROWS_PER_T * rows * 4 + rows * 4
    }
    /// Where a window's sampled ids land: `IDS` for narrow windows (up to `MAX_T`), past the
    /// row record for wider ones (the slack before `EMB` holds 16 words).
    fn ids_at(hidden: usize, rows: usize) -> usize {
        if rows <= MAX_T {
            Self::IDS
        } else {
            Self::rows(hidden, rows) + ROWS_PER_T * rows * 4
        }
    }
    /// Where the per-row record sits (after the embeddings).
    fn rows(hidden: usize, rows: usize) -> usize {
        Self::EMB + rows * hidden * 4
    }
    fn ptr(&self, off: usize) -> *mut u8 {
        unsafe { self.arena.as_ptr().add(off) }
    }
    #[allow(clippy::mut_from_ref)]
    fn f32s(&self, off: usize, n: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(off) as *mut f32, n) }
    }
    #[allow(clippy::mut_from_ref)]
    fn u32s(&self, off: usize, n: usize) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(off) as *mut u32, n) }
    }
}

/// Wide prefill windows (up to `rows` tokens of one sequence a window): the Coder kernels built
/// for that many rows, their scratch and pinned staging, and one projection buffer the GDN and
/// attention layers share (prefill windows never verify, so no layer needs its own kept).
struct Wide {
    rows: usize,
    s: Scratch,
    k: Kern,
    io: Io,
    proj: B,
    /// int8 codes and scales of a dense GEMM's f32 input (`XQ8_T` = rows layout).
    xq: B,
    /// Split-K partials of the batch module's dense MMA (`DMMA_PART` f32; empty elsewhere).
    part: B,
}

/// Attention partials are computed this many rows of a wide window at a time (their scratch
/// grows with rows × context).
const WIDE_ATTN_ROWS: usize = 32;

/// Batch modules: an expert's pairs (the shared expert's above all: every row's) run as groups
/// of at most this many (`CX_GCHUNK`), so no expert is one long serial group.
const BATCH_GCHUNK: usize = 8;

/// The batch module's dense-MMA split-K partials (f32 values: splits x rows x tokens).
const DMMA_PART: usize = 1 << 22;

/// The streaming grouped expert kernels' shape (`CX_SST`, `CX_SNR_GU`, `CX_SNR_DOWN`): ring
/// stages a warp, rows a warp for gate|up and for down.
/// `TANG_CODER_SST` / `TANG_CODER_SNR_GU` / `TANG_CODER_SNR_DOWN` override them (sweeps).
fn stream_shape() -> (usize, usize, usize) {
    let s = stream_shape4();
    (s.0, s.1, s.2)
}

/// [`stream_shape`] and the down kernel's ring depth (`TANG_CODER_SST_DOWN`).
fn stream_sst_down() -> usize {
    stream_shape4().3
}

fn stream_shape4() -> (usize, usize, usize, usize) {
    static S: std::sync::OnceLock<(usize, usize, usize, usize)> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let n = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        (
            n("TANG_CODER_SST", 1),
            n("TANG_CODER_SNR_GU", 8),
            n("TANG_CODER_SNR_DOWN", 16),
            n("TANG_CODER_SST_DOWN", 2),
        )
    })
}

/// A sequence past slot 0: its own recurrent state, conv history and K/V caches (per layer:
/// `gdn` for GDN layers, `kv` for attention layers), and its tokens.
struct Slot {
    gdn: Vec<Option<(B, B)>>,
    kv: Vec<Option<(B, B)>>,
    /// The MTP layer's K/V (when one is loaded).
    mtp_kv: Option<(B, B)>,
    tokens: Vec<u32>,
}

/// The second GPU of a two-GPU layer split (`TANG_GPUS=2`; see [`super::split`]): layers
/// `[split, L)` of `layers`, the head, the output norm and any MTP layer live here, with
/// their own kernels, scratch, pinned staging, streams and captured graphs. `None` means
/// one GPU — every code path below then runs exactly as before, on `self.dev`.
struct Peer {
    /// GPU 1 as a compute device (its own pool/context; allocations live on it).
    dev: CudaComputeDevice,
    ctx: super::split::Ctx,
    layers: Vec<Layer>,
    out_norm: B,
    head: Dense,
    rope: (B, B),
    s: Scratch,
    io: Io,
    k: Kern,
    side: Stream,
    ev_fork: gpu::Event,
    ev_join: gpu::Event,
    wide: Option<Wide>,
    graphs: HashMap<(usize, bool, usize, usize, bool), Graph>,
    /// GPU 1's commit graphs, by window width.
    commit_graphs: Vec<Option<Graph>>,
    /// GPU 1's slots: their `gdn`/`kv` entries, per GPU 1 layer.
    slots: Vec<Option<(Vec<Option<(B, B)>>, Vec<Option<(B, B)>>)>>,
}

/// One GPU's share of a window: the device, stream, kernels, scratch, staging, layers and
/// side stream the enqueue functions need — GPU 0's own (`self.…`) or the peer's. With no
/// split there is only stage 0, and it reads exactly the old fields.
struct Stage<'a> {
    dev: &'a CudaComputeDevice,
    /// The launch stream (GPU 0: the legacy `dev` stream wrapped in `Stream`).
    stream: &'a Stream,
    k: &'a Kern,
    s: &'a Scratch,
    /// The narrow set, even in a wide window: `gemv`'s per-row GEMVs (`MAX_T` rows at a
    /// time) run the narrow kernels on the narrow `xq8`.
    nk: &'a Kern,
    ns: &'a Scratch,
    io: &'a Io,
    layers: &'a [Layer],
    side: &'a Stream,
    #[allow(dead_code)]
    wide: Option<&'a Wide>,
}

pub struct Coder {
    pub dev: CudaComputeDevice,
    gpu: Gpu,
    /// Top-k / top-p / presence sampling in the head (`enable_sampler`); `None`: Gumbel-max.
    sampler: Option<Box<super::sampler::Sampler>>,
    /// Per-row sampling in slot windows (`enable_row_sampling`): each slot its own settings.
    row_sampler: Option<Box<super::sampler::RowSampler>>,
    slot_samp: Vec<super::sampler::RowSampling>,
    /// The next window's per-row sampler words (`stage_row_words`); empty: every row greedy.
    row_words: Vec<u32>,
    pub hp: Hparams,
    g: Gguf,
    geo: Geometry,
    eps: f32,
    layers: Vec<Layer>,
    out_norm: B,
    head: Dense,
    rope: (B, B),
    s: Scratch,
    io: Io,
    k: Kern,
    stream: ManuallyDrop<Stream>,
    /// The shared expert's branch (`shared_par`), and its fork/join events.
    side: Stream,
    ev_fork: gpu::Event,
    ev_join: gpu::Event,
    /// Window graphs by (t, verify, attention chunks, segments).
    graphs: HashMap<(usize, bool, usize, usize, bool), Graph>,
    /// The staged window is greedy (temperature 0, no presence penalty): the head takes the
    /// plain argmax even with the sampler enabled (its top-1 is the argmax then, ties to the
    /// lower id either way). Part of the graph key.
    greedy: std::cell::Cell<bool>,
    commit_graphs: Vec<Option<Graph>>,
    max_ctx: usize,
    counter: u32,
    pub tokens: Vec<u32>,
    /// Sequences past slot 0: their state buffers and tokens (`None`: free).
    slots: Vec<Option<Slot>>,
    /// Wide prefill (`enable_wide`), and whether the window being enqueued is a wide one (the
    /// window functions then read `wide`'s scratch, kernels and staging), with its first position.
    wide: Option<Wide>,
    in_wide: std::cell::Cell<bool>,
    wide_pos0: std::cell::Cell<usize>,
    /// Batched decode past `MAX_T` rows (`enable_batch`): the Coder kernels built for that
    /// many rows, with their scratch (head included) and staging; `in_batch` while such a
    /// window is enqueued, `last_batch` when the latest window was one (for `logits`).
    batch: Option<Wide>,
    in_batch: std::cell::Cell<bool>,
    last_batch: bool,
    /// The Coder kernels' shapes (for building the wide module).
    shapes: Shapes,
    pub seed: u32,
    pub temperature: f32,
    pub use_graphs: bool,
    pub last: WinStats,
    pub last_commit_ms: f64,
    pub load_report: String,
    /// `TANG_CODER_PROF=1` (eager windows): GPU time per stage, by name, summed.
    prof: Option<std::cell::RefCell<Prof>>,
    /// Stage sweep: enqueue only this stage's launches (`stage_sweep`).
    only: std::cell::Cell<Option<&'static str>>,
    var: Variants,
    mtp: Option<Mtp>,
    /// MTP pass graphs by (rows, attention chunks).
    mtp_graphs: HashMap<(usize, usize), Graph>,
    /// Draft with the MTP layer after each window (when loaded and greedy).
    pub use_mtp: bool,
    /// Chain length (`TANG_CODER_MTP_K`, default 2).
    pub mtp_steps: usize,
    /// The latest MTP chain: draft tokens and their probabilities, for the next verify.
    pub mtp_last: Vec<(u32, f32)>,
    /// The MTP head's runner-up for `mtp_last[0]`'s position (a second tree branch).
    pub mtp_alt: Option<(u32, f32)>,
    /// Each slot's latest MTP chain (batched decode; slot 0's is `mtp_last`).
    slot_chain: Vec<Vec<(u32, f32)>>,
    /// How many sequence slots fit in VRAM (set at the first `alloc_slot`); 0: not yet known.
    slot_cap: usize,
    /// Per-segment commit graphs after a batched verify window, by segment count.
    commit_seg_graphs: Vec<Option<Graph>>,
    pub last_mtp_ms: f64,
    /// The second GPU of the layer split (`None`: one GPU, everything on `self.dev`).
    peer: Option<Box<Peer>>,
}

#[derive(Default)]
struct Prof {
    marks: Vec<(&'static str, gpu::Event)>,
    total: std::collections::BTreeMap<&'static str, (f64, usize)>,
}

impl Coder {
    /// Load a `qwen3next` GGUF (first shard) onto the GPU.
    pub fn load(path: &Path, opts: CoderOpts) -> Result<Self> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let hp = Hparams::from_gguf(&g)?;
        ensure!(hp.arch == Arch::Qwen3Next, "not a qwen3next GGUF");
        ensure!(
            hp.ssm_d_state == 128 && hp.ssm_d_conv == 4,
            "GDN state {} / conv {}: the kernels are built for 128 / 4",
            hp.ssm_d_state,
            hp.ssm_d_conv
        );
        ensure!(
            hp.head_dim == 256 && hp.n_rot == 64 && hp.n_head % hp.n_head_kv == 0,
            "attention head_dim {} rot {}: the kernels are built for 256 / 64",
            hp.head_dim,
            hp.n_rot
        );
        ensure!(
            hp.n_embd.is_multiple_of(256) && hp.n_ff_exp.is_multiple_of(256),
            "widths must be multiples of 256"
        );
        let sh = Shapes {
            hidden: hp.n_embd,
            n_head: hp.n_head,
            n_kv: hp.n_head_kv,
            head_dim: hp.head_dim,
            n_rot: hp.n_rot,
            ff: hp.n_ff_exp,
            topk: hp.n_expert_used,
            n_expert: hp.n_expert,
        };
        // The flash module's GDN kernels for this model's heads; the rest of its geometry is
        // Flash-Next's (unused here, kept so every kernel of the module still compiles).
        let geo = Geometry {
            gdn_hk: hp.ssm_k_heads,
            gdn_hv: hp.ssm_v_heads,
            gdn_d: hp.ssm_d_state,
            gdn_grouped: true,
            ..Geometry::FLASH_NEXT
        };
        let max_ctx = opts.max_ctx.next_multiple_of(ck::attn_chunk());
        let eps = hp.eps;

        // Where a two-GPU split cuts the layer list (`TANG_SPLIT_AT`, or the balanced point).
        // One GPU loads every layer.
        let split_at = if super::split::gpus() == 2 && hp.n_layer > 1 {
            split_point(&g, &hp)
        } else {
            hp.n_layer
        };

        let dev = CudaComputeDevice::new().map_err(|e| anyhow!("CUDA: {e:?}"))?;
        let gpu = Gpu {
            ctx: dev.cuda_context().clone(),
        };
        gpu.bind().map_err(|e| anyhow!("{e}"))?;
        let (free0, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;

        // ---- weights (a layer split loads [0, k) here; [k, L) on GPU 1 below)
        let t1 = Instant::now();
        let info = |name: &str| -> Result<TensorInfo> { Ok(g.info(name)?.clone()) };
        let vecf = |name: &str| -> Result<B> { Ok(dev.upload_f32(&g.tensor_f32(name)?)) };
        let mut expert_bytes = 0usize;
        let mut layers: Vec<Layer> = Vec::with_capacity(hp.n_layer);
        let (hk, hv) = (hp.ssm_k_heads, hp.ssm_v_heads);
        let r = hv / hk;
        // ssm_ba rows per k head g: [b of v heads g·r..g·r + r | a of the same]; the flash
        // kernels read [a (hv) | b (hv)] in v-head order.
        let ba_perm: Vec<usize> = (0..2 * hv)
            .map(|i| {
                let (is_b, h) = (i >= hv, i % hv);
                let (gk, j) = (h / r, h % r);
                gk * 2 * r + if is_b { j } else { r + j }
            })
            .collect();
        for l in 0..split_at {
            let layer = load_layer(
                &dev,
                &g,
                &hp,
                &geo,
                &format!("blk.{l}."),
                hp.is_recurrent[l],
                &ba_perm,
                max_ctx,
            )?;
            for e in [&layer.gate, &layer.up, &layer.down] {
                expert_bytes += e.w.estride as usize * hp.n_expert;
            }
            layers.push(layer);
            if l % 8 == 7 {
                eprintln!(
                    "coder: {} of {} layers in VRAM ({:.0} s)",
                    l + 1,
                    hp.n_layer,
                    t1.elapsed().as_secs_f64()
                );
            }
        }
        let ff_sh = hp.n_ff_exp;
        let out_norm = vecf("output_norm.weight")?;
        let head_name = if g.get("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let head = upload_dense(&dev, &g, &[(info(head_name)?, None)])?;
        ensure!(head.rows == hp.n_vocab, "head rows {}", head.rows);
        let rope = {
            let (c, s) = fl::rope_table(max_ctx, hp.rope_base as f64);
            (dev.upload_f32(&c), dev.upload_f32(&s))
        };
        let mtp_path = opts
            .mtp
            .clone()
            .or_else(|| std::env::var_os("TANG_CODER_MTP").map(Into::into));
        let mtp_parts = match &mtp_path {
            Some(p) => {
                let mg = Gguf::open(p).with_context(|| format!("MTP sidecar {}", p.display()))?;
                let layer = load_layer(&dev, &mg, &hp, &geo, "mtp.", false, &ba_perm, max_ctx)?;
                let fc = upload_dense(&dev, &mg, &[(mg.info("mtp.fc.weight")?.clone(), None)])?;
                ensure!(
                    fc.rows == hp.n_embd && fc.k == 2 * hp.n_embd,
                    "MTP fc {}x{}",
                    fc.rows,
                    fc.k
                );
                let v = |name: &str| -> Result<B> { Ok(dev.upload_f32(&mg.tensor_f32(name)?)) };
                Some((
                    layer,
                    fc,
                    v("mtp.enorm.weight")?,
                    v("mtp.hnorm.weight")?,
                    v("mtp.norm.weight")?,
                ))
            }
            None => None,
        };

        // ---- scratch
        let n = hp.n_embd;
        let kmax = [2 * n, geo.gdn_v(), hp.n_head * hp.head_dim, ff_sh]
            .into_iter()
            .max()
            .unwrap();
        let s = Scratch::new(&dev, &hp, &geo, kmax, max_ctx, MAX_T, MAX_T, MAX_T);
        dev.sync();
        let t_weights = t1.elapsed().as_secs_f64();
        let (free1, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;

        // ---- kernels
        let t2 = Instant::now();
        let k = Kern::build_on(&gpu, &sh, &geo, MAX_T)?;
        let mtp = match mtp_parts {
            Some((layer, fc, enorm, hnorm, onorm)) => {
                let m = gpu
                    .module(MTP_SRC)
                    .map_err(|e| anyhow!("MTP kernels: {e}"))?;
                Some(Mtp {
                    layer,
                    fc,
                    enorm,
                    hnorm,
                    onorm,
                    mh: dev.alloc_f32(MAX_T * n),
                    emb: dev.alloc_f32(MAX_T * n),
                    cat: dev.alloc_f32(MAX_T * 2 * n),
                    ids: dev.alloc_f32(2 * MAX_T),
                    prob: dev.alloc_f32(2 * MAX_T),
                    f_in: m.func("cx_mtp_in").map_err(|e| anyhow!("{e}"))?,
                    f_pick: m.func("cx_mtp_pick").map_err(|e| anyhow!("{e}"))?,
                    _m: m,
                })
            }
            None => None,
        };
        let t_kernels = t2.elapsed().as_secs_f64();

        let mut io = Io {
            arena: HostArena::new(
                Io::bytes(n, MAX_T),
                ArenaOptions {
                    try_hugetlb: false,
                    thp: false,
                },
            )?,
        };
        io.arena
            .register(&gpu.ctx, &[])
            .map_err(|e| anyhow!("{e}"))?;
        let stream = ManuallyDrop::new(Stream(dev.cu_stream()));

        // ---- the optional second GPU (`TANG_GPUS=2`): layers [split_at, L), head, out_norm
        // and rope loaded on device 1, with their own kernels/scratch/staging.
        let (peer, peer_report) = if super::split::gpus() == 2 && hp.n_layer > 1 {
            let (peer, report) = build_peer(
                &g, &gpu, &hp, &geo, &sh, max_ctx, &mtp_path, split_at, &ba_perm,
            )?;
            (Some(peer), report)
        } else {
            (None, String::new())
        };

        let load_report = format!(
            "coder: {} layers ({} GDN, {} attention), {} experts top-{}, hidden {}, vocab {}; \
             {:.1} GB VRAM ({:.1} GB experts), weights {:.0} s, kernels {:.0} s, total {:.0} s{}",
            hp.n_layer,
            hp.is_recurrent.iter().filter(|&&x| x).count(),
            hp.is_recurrent.iter().filter(|&&x| !x).count(),
            hp.n_expert,
            hp.n_expert_used,
            n,
            hp.n_vocab,
            (free0 - free1) as f64 / 1e9,
            expert_bytes as f64 / 1e9,
            t_weights,
            t_kernels,
            t0.elapsed().as_secs_f64(),
            peer_report
        );
        let c = Coder {
            dev,
            gpu,
            sampler: None,
            row_sampler: None,
            slot_samp: vec![super::sampler::RowSampling::greedy(); MAX_SLOTS],
            row_words: Vec::new(),
            hp,
            g,
            geo,
            eps,
            layers,
            out_norm,
            head,
            rope,
            s,
            io,
            k,
            stream,
            side: Stream::new().map_err(|e| anyhow!("{e}"))?,
            ev_fork: gpu::Event::new(false).map_err(|e| anyhow!("{e}"))?,
            ev_join: gpu::Event::new(false).map_err(|e| anyhow!("{e}"))?,
            graphs: HashMap::new(),
            commit_graphs: (0..=MAX_T).map(|_| None).collect(),
            max_ctx,
            counter: 0,
            tokens: Vec::new(),
            greedy: std::cell::Cell::new(true),
            slots: (0..MAX_SLOTS).map(|_| None).collect(),
            wide: None,
            in_wide: std::cell::Cell::new(false),
            wide_pos0: std::cell::Cell::new(0),
            batch: None,
            in_batch: std::cell::Cell::new(false),
            last_batch: false,
            shapes: sh.clone(),
            seed: 0,
            temperature: 0.0,
            use_graphs: true,
            last: WinStats::default(),
            last_commit_ms: 0.0,
            load_report,
            prof: std::env::var("TANG_CODER_PROF")
                .is_ok_and(|v| v == "1")
                .then(|| std::cell::RefCell::new(Prof::default())),
            only: std::cell::Cell::new(None),
            var: Variants::from_env(),
            use_mtp: mtp.is_some(),
            mtp,
            mtp_graphs: HashMap::new(),
            mtp_steps: std::env::var("TANG_CODER_MTP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2)
                .clamp(1, MAX_T - 1),
            mtp_last: Vec::new(),
            mtp_alt: None,
            slot_chain: vec![Vec::new(); MAX_SLOTS],
            slot_cap: 0,
            commit_seg_graphs: (0..=MAX_T).map(|_| None).collect(),
            last_mtp_ms: 0.0,
            peer,
        };
        let mut c = c;
        c.write_slot_tables();
        Ok(c)
    }

    // ---- sequence slots (several sequences in one window)

    /// Rewrite every layer's slot tables from the live slots (slot 0: the layer's own buffers;
    /// a free slot: 0), and sync. Once, at load; slots coming and going update their own
    /// entries ([`write_slot_entry`](Self::write_slot_entry)).
    fn write_slot_tables(&mut self) {
        for w in 0..self.n_stages() {
            self.write_slot_tables_on(w);
        }
        // The MTP layer: slot 0 its own buffers, the others their `mtp_kv`.
        if self.mtp.is_some() {
            let mut ta = [0u64; MAX_SLOTS];
            let mut tb = [0u64; MAX_SLOTS];
            if let Some(at) = self.mtp.as_ref().and_then(|m| m.layer.attn.as_ref()) {
                ta[0] = self.a(&at.k);
                tb[0] = self.a(&at.v);
            }
            for (i, sl) in self.slots.iter().enumerate().skip(1) {
                if let Some((a, b)) = sl.as_ref().and_then(|s| s.mtp_kv.as_ref()) {
                    ta[i] = self.a(a);
                    tb[i] = self.a(b);
                }
            }
            let words = |t: &[u64; MAX_SLOTS]| -> Vec<f32> {
                t.iter()
                    .flat_map(|&v| [f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32)])
                    .collect()
            };
            let (wa, wb) = (words(&ta), words(&tb));
            if let Some(at) = self.mtp.as_mut().and_then(|m| m.layer.attn.as_mut()) {
                self.dev.upload_into_f32(&mut at.ktab, &wa);
                self.dev.upload_into_f32(&mut at.vtab, &wb);
            }
        }
        self.dev.sync();
    }

    /// Slot `i`'s entry in every table (stage layers and MTP) after it was allocated or freed:
    /// two f32 words per table at `2 * i`, the same values a full
    /// [`write_slot_tables`](Self::write_slot_tables) would write there (0 for a free slot),
    /// the other entries untouched. No sync on GPU 0: the uploads are on the engine's stream,
    /// which every kernel reading the tables runs on (or forks from by event, the side
    /// stream), so they're ordered before the next window. The peer stage's kernels run on
    /// its own `Ctx` stream, not its device's upload stream, so that one is synced.
    fn write_slot_entry(&mut self, i: usize) {
        let words = |v: u64| [f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32)];
        for w in 0..self.n_stages() {
            let nl = self.stage_view(w).layers.len();
            let mut entries = Vec::with_capacity(nl);
            for l in 0..nl {
                let layer = &self.stage_view(w).layers[l];
                if layer.gdn.is_none() && layer.attn.is_none() {
                    continue;
                }
                let pair = if w == 0 {
                    self.slots[i].as_ref().and_then(|sl| {
                        if layer.gdn.is_some() {
                            sl.gdn[l].as_ref()
                        } else {
                            sl.kv[l].as_ref()
                        }
                    })
                } else {
                    let p = self.peer.as_ref().expect("no peer");
                    p.slots[i].as_ref().and_then(|(gdn, kv)| {
                        if layer.gdn.is_some() {
                            gdn[l].as_ref()
                        } else {
                            kv[l].as_ref()
                        }
                    })
                };
                let (a, b) = pair.map_or((0, 0), |(a, b)| (self.a_on(w, a), self.a_on(w, b)));
                entries.push((l, words(a), words(b)));
            }
            let (dev, layers) = if w == 0 {
                (&self.dev, &mut self.layers)
            } else {
                let p = self.peer.as_deref_mut().expect("no peer");
                (&p.dev, &mut p.layers)
            };
            for (l, wa, wb) in entries {
                let layer = &mut layers[l];
                let (da, db) = match (&mut layer.gdn, &mut layer.attn) {
                    (Some(g), _) => (&mut g.stab, &mut g.htab),
                    (_, Some(at)) => (&mut at.ktab, &mut at.vtab),
                    _ => continue,
                };
                dev.upload_into_f32_at(da, 2 * i, &wa);
                dev.upload_into_f32_at(db, 2 * i, &wb);
            }
            if w > 0 {
                dev.sync();
            }
        }
        if self.mtp.is_some() {
            let (a, b) = self.slots[i]
                .as_ref()
                .and_then(|s| s.mtp_kv.as_ref())
                .map_or((0, 0), |(a, b)| (self.a(a), self.a(b)));
            if let Some(at) = self.mtp.as_mut().and_then(|m| m.layer.attn.as_mut()) {
                self.dev.upload_into_f32_at(&mut at.ktab, 2 * i, &words(a));
                self.dev.upload_into_f32_at(&mut at.vtab, 2 * i, &words(b));
            }
        }
        // TANG_SLOT_TABLE_CHECK=1: the tables must equal a full rewrite's, bit for bit.
        if std::env::var("TANG_SLOT_TABLE_CHECK").is_ok_and(|v| v == "1") {
            let before = self.slot_table_words();
            self.write_slot_tables();
            assert!(
                before == self.slot_table_words(),
                "slot {i}: the per-entry tables differ from a full rewrite"
            );
        }
    }

    /// Every slot table's words (stage layers, then MTP), read back (for the check above).
    fn slot_table_words(&self) -> Vec<u32> {
        self.dev.sync();
        let mut out = Vec::new();
        for w in 0..self.n_stages() {
            let dev = if w == 0 {
                &self.dev
            } else {
                &self.peer.as_ref().expect("no peer").dev
            };
            dev.sync();
            for layer in self.stage_view(w).layers {
                let (a, b) = match (&layer.gdn, &layer.attn) {
                    (Some(g), _) => (&g.stab, &g.htab),
                    (_, Some(at)) => (&at.ktab, &at.vtab),
                    _ => continue,
                };
                out.extend(dev.download(a).iter().map(|x| x.to_bits()));
                out.extend(dev.download(b).iter().map(|x| x.to_bits()));
            }
        }
        if let Some(at) = self.mtp.as_ref().and_then(|m| m.layer.attn.as_ref()) {
            out.extend(self.dev.download(&at.ktab).iter().map(|x| x.to_bits()));
            out.extend(self.dev.download(&at.vtab).iter().map(|x| x.to_bits()));
        }
        out
    }

    /// [`write_slot_tables`](Self::write_slot_tables) for stage `w`'s layers only.
    fn write_slot_tables_on(&mut self, w: usize) {
        let nl = self.stage_view(w).layers.len();
        let mut tables = Vec::with_capacity(nl);
        for l in 0..nl {
            let mut ta = [0u64; MAX_SLOTS];
            let mut tb = [0u64; MAX_SLOTS];
            let layer = &self.stage_view(w).layers[l];
            let (a0, b0) = match (&layer.gdn, &layer.attn) {
                (Some(g), _) => (self.a_on(w, &g.state), self.a_on(w, &g.hist)),
                (_, Some(at)) => (self.a_on(w, &at.k), self.a_on(w, &at.v)),
                _ => continue,
            };
            ta[0] = a0;
            tb[0] = b0;
            for (i, sl) in self.slots.iter().enumerate().skip(1) {
                if sl.is_none() {
                    continue;
                }
                let (gdn, kv) = if w == 0 {
                    (&sl.as_ref().unwrap().gdn, &sl.as_ref().unwrap().kv)
                } else {
                    let p = self.peer.as_ref().expect("no peer");
                    let (a, b) = p.slots[i].as_ref().expect("peer slot");
                    (a, b)
                };
                let pair = if layer.gdn.is_some() { &gdn[l] } else { &kv[l] };
                if let Some((a, b)) = pair {
                    ta[i] = self.a_on(w, a);
                    tb[i] = self.a_on(w, b);
                }
            }
            tables.push((l, ta, tb));
        }
        let words = |t: &[u64; MAX_SLOTS]| -> Vec<f32> {
            t.iter()
                .flat_map(|&v| [f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32)])
                .collect()
        };
        if w == 0 {
            for (l, ta, tb) in tables {
                let (wa, wb) = (words(&ta), words(&tb));
                let layer = &mut self.layers[l];
                let (da, db) = match (&mut layer.gdn, &mut layer.attn) {
                    (Some(g), _) => (&mut g.stab, &mut g.htab),
                    (_, Some(at)) => (&mut at.ktab, &mut at.vtab),
                    _ => continue,
                };
                self.dev.upload_into_f32(da, &wa);
                self.dev.upload_into_f32(db, &wb);
            }
        } else {
            let p = self.peer.as_deref_mut().expect("no peer");
            for (l, ta, tb) in tables {
                let (wa, wb) = (words(&ta), words(&tb));
                let layer = &mut p.layers[l];
                let (da, db) = match (&mut layer.gdn, &mut layer.attn) {
                    (Some(g), _) => (&mut g.stab, &mut g.htab),
                    (_, Some(at)) => (&mut at.ktab, &mut at.vtab),
                    _ => continue,
                };
                p.dev.upload_into_f32(da, &wa);
                p.dev.upload_into_f32(db, &wb);
            }
        }
    }

    /// Switch the per-pair expert kernels (`TANG_CODER_MOE`, `TANG_CODER_FUSE_SILU`)
    /// without reloading; drops the captured window graphs, which hold the old kernels.
    pub fn set_moe_variant(&mut self, v: usize, fuse_silu: bool) {
        self.var.moe = v;
        self.var.fuse_silu = fuse_silu;
        self.graphs.clear();
        self.commit_graphs.iter_mut().for_each(|g| *g = None);
    }

    /// Device bytes one sequence slot takes (its GDN states, K/V and MTP K/V).
    pub fn slot_bytes(&self) -> usize {
        let kv = self.max_ctx * self.hp.n_head_kv * self.hp.head_dim * 2 * 2;
        let gdn = (self.geo.gdn_hv * 128 * 128 + 3 * self.geo.gdn_conv()) * 4;
        let per_layer: usize = self
            .layers
            .iter()
            .map(|l| usize::from(l.gdn.is_some()) * gdn + usize::from(l.attn.is_some()) * kv)
            .sum();
        per_layer + usize::from(self.mtp.is_some()) * kv
    }

    /// Sequence slots that fit in VRAM now: recounted from free memory (plus what the live
    /// slots already hold), and kept as the cap [`alloc_slot`](Self::alloc_slot) enforces.
    /// `flash serve` calls it at startup, before and after building its windows, to clamp
    /// `--batch`.
    pub fn fit_slots(&mut self) -> usize {
        // A slot's state is ~0.85 GB at a 32k context (K/V of every attention layer at
        // `max_ctx`, plus the GDN states): MAX_SLOTS of them don't fit. Cap the count at
        // what's free, less headroom for graphs and later scratch, so a server parking
        // finished slots evicts one instead of failing an allocation mid-step.
        let headroom = std::env::var("TANG_SLOT_HEADROOM_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(4.0);
        let per = self.slot_bytes().max(1);
        let live = self.slots.iter().skip(1).filter(|s| s.is_some()).count();
        let free = cudarc::driver::result::mem_get_info()
            .map_or(usize::MAX, |m| m.0.saturating_add(live * per));
        self.slot_cap =
            crate::batch::slots_that_fit(free, (headroom * 1e9) as usize, per, MAX_SLOTS);
        eprintln!(
            "coder: {} sequence slots fit ({:.2} GB each, {:.1} GB free, {headroom} GB headroom)",
            self.slot_cap,
            per as f64 / 1e9,
            free as f64 / 1e9
        );
        self.slot_cap
    }

    /// A free sequence slot (1..MAX_SLOTS), its state zeroed and no tokens.
    pub fn alloc_slot(&mut self) -> Result<usize> {
        if self.slot_cap == 0 {
            self.fit_slots();
        }
        let live = self.slots.iter().skip(1).filter(|s| s.is_some()).count();
        ensure!(live < self.slot_cap, "all {} slots that fit in VRAM in use", self.slot_cap);
        let i = (1..MAX_SLOTS)
            .find(|&i| self.slots[i].is_none())
            .ok_or_else(|| anyhow!("all {} slots in use", MAX_SLOTS))?;
        let (hv, conv, kv) = (
            self.geo.gdn_hv,
            self.geo.gdn_conv(),
            self.max_ctx * self.hp.n_head_kv * self.hp.head_dim,
        );
        let mut sl = Slot {
            gdn: Vec::with_capacity(self.layers.len()),
            kv: Vec::with_capacity(self.layers.len()),
            mtp_kv: self
                .mtp
                .as_ref()
                .map(|_| (self.dev.alloc_bf16(kv), self.dev.alloc_bf16(kv))),
            tokens: Vec::new(),
        };
        for layer in &self.layers {
            sl.gdn.push(layer.gdn.as_ref().map(|_| {
                (
                    self.dev.alloc_f32(hv * 128 * 128),
                    self.dev.alloc_f32(3 * conv),
                )
            }));
            sl.kv.push(
                layer
                    .attn
                    .as_ref()
                    .map(|_| (self.dev.alloc_bf16(kv), self.dev.alloc_bf16(kv))),
            );
        }
        self.slots[i] = Some(sl);
        // The peer's layers get their own state for the slot.
        if let Some(p) = self.peer.as_deref_mut() {
            let mut gdn = Vec::with_capacity(p.layers.len());
            let mut kvv = Vec::with_capacity(p.layers.len());
            for layer in &p.layers {
                gdn.push(
                    layer
                        .gdn
                        .as_ref()
                        .map(|_| (p.dev.alloc_f32(hv * 128 * 128), p.dev.alloc_f32(3 * conv))),
                );
                kvv.push(
                    layer
                        .attn
                        .as_ref()
                        .map(|_| (p.dev.alloc_bf16(kv), p.dev.alloc_bf16(kv))),
                );
            }
            p.slots[i] = Some((gdn, kvv));
        }
        self.slot_chain[i].clear();
        self.write_slot_entry(i);
        Ok(i)
    }

    /// Release slot `i` (not 0) and its memory, on every stage.
    pub fn free_slot(&mut self, i: usize) {
        if i > 0 && i < MAX_SLOTS && self.slots[i].take().is_some() {
            self.slot_samp[i] = super::sampler::RowSampling::greedy();
            if let Some(p) = self.peer.as_deref_mut() {
                p.slots[i] = None;
            }
            self.write_slot_entry(i);
        }
    }

    /// Slot `i`'s tokens (slot 0: [`tokens`](Self::tokens)).
    pub fn slot_tokens(&self, i: usize) -> &[u32] {
        if i == 0 {
            &self.tokens
        } else {
            self.slots[i].as_ref().map_or(&[], |s| &s.tokens)
        }
    }

    fn slot_tokens_mut(&mut self, i: usize) -> Result<&mut Vec<u32>> {
        if i == 0 {
            return Ok(&mut self.tokens);
        }
        self.slots
            .get_mut(i)
            .and_then(|s| s.as_mut())
            .map(|s| &mut s.tokens)
            .ok_or_else(|| anyhow!("slot {i} isn't allocated"))
    }

    /// Stage a window of rows `(slot, position, token)` in segments `[start, len, slot, keep]`.
    fn stage_rows(&mut self, rows: &[(usize, usize, u32)], segs: &[[u32; 4]]) -> Result<()> {
        let t = rows.len();
        let tw = self.tw();
        ensure!((1..=tw).contains(&t) && segs.len() <= tw);
        let ctl = self.iox().u32s(Io::CTL, 5);
        ctl[0] = rows[0].1 as u32;
        ctl[1] = t as u32;
        ctl[2] = self.counter;
        ctl[3] = self.seed;
        ctl[4] = self.temperature.to_bits();
        let n = self.hp.n_embd;
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.iox().f32s(Io::EMB, t * n);
        for (i, r) in rows.iter().enumerate() {
            let row = self.g.rows(&emb_t, r.2 as usize, 1)?;
            emb[i * n..(i + 1) * n].copy_from_slice(&row);
        }
        let rec = self.iox().u32s(Io::rows(n, tw), ROWS_PER_T * tw);
        rec.fill(0);
        for (i, r) in rows.iter().enumerate() {
            rec[i] = r.1 as u32;
            rec[tw + i] = r.0 as u32;
        }
        for (b, sg) in segs.iter().enumerate() {
            rec[2 * tw + 4 * b..2 * tw + 4 * b + 4].copy_from_slice(sg);
        }
        self.mirror_staging(tw, t);
        Ok(())
    }

    /// Copy this window's staged control words, rows and embeddings into GPU 1's own pinned
    /// staging (a split runs the same window on both, each from its own arena). Host-side
    /// only, a few KiB.
    fn mirror_staging(&mut self, tw: usize, t: usize) {
        let Some(p) = self.peer.as_deref_mut() else {
            return;
        };
        let n = self.hp.n_embd;
        let bytes = |off: usize, len: usize| unsafe {
            std::slice::from_raw_parts(self.io.arena.as_ptr().add(off), len)
        };
        let ctl = bytes(Io::CTL, 5 * 4);
        let rows = bytes(Io::rows(n, tw), ROWS_PER_T * tw * 4);
        let emb = bytes(Io::EMB, t * n * 4);
        let pio = &mut p.io;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctl.as_ptr(),
                pio.arena.as_ptr().add(Io::CTL) as *mut u8,
                ctl.len(),
            );
            std::ptr::copy_nonoverlapping(
                rows.as_ptr(),
                pio.arena.as_ptr().add(Io::rows(n, tw)) as *mut u8,
                rows.len(),
            );
            std::ptr::copy_nonoverlapping(
                emb.as_ptr(),
                pio.arena.as_ptr().add(Io::EMB) as *mut u8,
                emb.len(),
            );
        }
    }

    /// Feed `ids` to slot `slot` (windows of up to `MAX_T` of its rows) and return the next
    /// token. Greedy; the sampler isn't per slot yet.
    pub fn prefill_slot(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prompt");
        ensure!(self.sampler.is_none(), "slot windows sample per row, not with the engine sampler");
        let mut pos = self.slot_tokens(slot).len();
        self.slot_tokens_mut(slot)?.extend_from_slice(ids);
        let end = pos + ids.len();
        ensure!(end <= self.max_ctx, "context {end} > max {}", self.max_ctx);
        let mut next = 0;
        while pos < end {
            let t = (end - pos).min(MAX_T);
            let toks = &self.slot_tokens(slot)[pos..pos + t];
            let rows: Vec<(usize, usize, u32)> = toks
                .iter()
                .enumerate()
                .map(|(i, &k)| (slot, pos + i, k))
                .collect();
            let w0 = Instant::now();
            self.stage_rows(&rows, &[[0, t as u32, slot as u32, t as u32]])?;
            if pos + t == end {
                self.stage_row_words(&[(t - 1, slot, end - 1)]);
            }
            let nchunk = self.nchunk(pos + t);
            let st = WinStats {
                t,
                ..Default::default()
            };
            next = self.run_staged(st, w0, t, false, nchunk, 1, None)?[t - 1];
            if self.mtp_on() {
                let toks = self.slot_tokens(slot);
                let mut nx = toks[pos + 1..pos + t].to_vec();
                nx.push(if pos + t == end { next } else { toks[pos + t] });
                let c = self.mtp_groups(&[(slot, pos, 0, &nx)], pos + t == end)?;
                if pos + t == end {
                    self.slot_chain[slot] = c[0].0.clone();
                }
            }
            if (pos + t) % 1024 < t && std::env::var("TANG_WIDE_TIMES").is_ok_and(|v| v == "1") {
                eprintln!("narrow prefill at {}", pos + t);
            }
            pos += t;
        }
        Ok(next)
    }

    /// Slot `slot`'s latest MTP chain (draft tokens and probabilities), for its next verify.
    pub fn slot_drafts(&self, slot: usize) -> &[(u32, f32)] {
        if slot == 0 {
            &self.mtp_last
        } else {
            &self.slot_chain[slot]
        }
    }

    /// One verify window for several sequences: `(slot, token, drafts)` per sequence, distinct
    /// slots, `1 + drafts` rows each, `MAX_T` rows in all. Feeds each slot its token and
    /// drafts, keeps each slot's accepted drafts (its recurrent state committed per segment),
    /// and returns each sequence's kept tokens (accepted drafts' successors plus the first
    /// correction), as a lone `verify` of that sequence would. With an MTP layer, also drafts
    /// each slot's next chain (`slot_drafts`).
    pub fn verify_batch(&mut self, items: &[(usize, u32, Vec<u32>)]) -> Result<Vec<Vec<u32>>> {
        let b = items.len();
        let t: usize = items.iter().map(|x| 1 + x.2.len()).sum();
        ensure!(b >= 1 && t <= MAX_T, "{b} sequences, {t} rows");
        ensure!(self.sampler.is_none(), "slot windows sample per row, not with the engine sampler");
        ensure!(
            items.iter().all(|x| x.2.is_empty() || !self.slot_samp[x.0].sampled()),
            "drafts are verified greedily: a sampled slot takes no drafts"
        );
        if items.iter().all(|x| x.2.is_empty()) {
            let rows: Vec<(usize, u32)> = items.iter().map(|x| (x.0, x.1)).collect();
            return Ok(self
                .decode_batch(&rows)?
                .into_iter()
                .map(|k| vec![k])
                .collect());
        }
        let mut staged = Vec::with_capacity(t);
        let mut segs = Vec::with_capacity(b);
        let mut pos0 = Vec::with_capacity(b);
        let mut end = 0;
        for (i, (slot, cur, drafts)) in items.iter().enumerate() {
            ensure!(
                items[..i].iter().all(|r| r.0 != *slot),
                "slot {slot} twice in one step"
            );
            let toks = self.slot_tokens_mut(*slot)?;
            let pos = toks.len();
            toks.push(*cur);
            toks.extend_from_slice(drafts);
            let len = 1 + drafts.len();
            end = end.max(pos + len);
            segs.push([staged.len() as u32, len as u32, *slot as u32, len as u32]);
            staged.push((*slot, pos, *cur));
            for (j, &d) in drafts.iter().enumerate() {
                staged.push((*slot, pos + 1 + j, d));
            }
            pos0.push(pos);
        }
        ensure!(end <= self.max_ctx, "context {end} > max {}", self.max_ctx);
        let w0 = Instant::now();
        self.stage_rows(&staged, &segs)?;
        let rw: Vec<(usize, usize, usize)> = items
            .iter()
            .enumerate()
            .filter(|(_, x)| x.2.is_empty())
            .map(|(i, x)| (segs[i][0] as usize, x.0, pos0[i]))
            .collect();
        self.stage_row_words(&rw);
        let nchunk = self.nchunk(end);
        let st = WinStats {
            t,
            ..Default::default()
        };
        let targets = self.run_staged(st, w0, t, true, nchunk, b, None)?;
        // Each segment keeps its accepted drafts' rows: stage the counts, commit them.
        let mut kept = Vec::with_capacity(b);
        let rec = self.io.u32s(Io::rows(self.hp.n_embd, MAX_T), ROWS_U32);
        for (i, (_, _, drafts)) in items.iter().enumerate() {
            let r0 = segs[i][0] as usize;
            let mut n = 1;
            while n <= drafts.len() && drafts[n - 1] == targets[r0 + n - 1] {
                n += 1;
            }
            rec[2 * MAX_T + 4 * i + 3] = n as u32;
            kept.push(targets[r0..r0 + n].to_vec());
        }
        self.commit_batch(b)?;
        // The commit reads the staged row record when it runs: let it, before restaging.
        self.sync_all();
        for (i, (slot, _, _)) in items.iter().enumerate() {
            let p = pos0[i] + kept[i].len();
            self.slot_tokens_mut(*slot)?.truncate(p);
        }
        if self.mtp_on() {
            let groups: Vec<(usize, usize, usize, &[u32])> = items
                .iter()
                .enumerate()
                .map(|(i, x)| (x.0, pos0[i], segs[i][0] as usize, &kept[i][..]))
                .collect();
            let chains = self.mtp_groups(&groups, true)?;
            for (x, c) in items.iter().zip(chains) {
                self.slot_chain[x.0] = c.0;
            }
        }
        Ok(kept)
    }

    /// After a batched verify window of `nseg` segments whose keep counts are staged: commit
    /// each segment's kept rows of recurrent state and conv history.
    fn commit_batch(&mut self, nseg: usize) -> Result<()> {
        if !self.use_graphs {
            self.enqueue_commit_seg(nseg);
            return Ok(());
        }
        if self.peer.is_none() {
            if self.commit_seg_graphs[nseg].is_none() {
                self.enqueue_commit_seg(nseg);
                self.dev.sync();
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let me = self as *const Self;
                let gr = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }.enqueue_commit_seg(nseg);
                    Ok(())
                })
                .map_err(|e| anyhow!("capture commit nseg={nseg}: {e}"))?;
                self.commit_seg_graphs[nseg] = Some(gr);
                return Ok(());
            }
            return self.commit_seg_graphs[nseg]
                .as_ref()
                .unwrap()
                .launch(&self.stream)
                .map_err(|e| anyhow!("{e}"));
        }
        // A split: per-stage graphs (GPU 1's seg graphs in the peer).
        let need = self.peer.as_ref().expect("no peer").commit_graphs[nseg].is_none();
        if need {
            self.enqueue_commit_seg(nseg);
            self.sync_all();
            let me = self as *const Self;
            {
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let g0 = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }.enqueue_commit_seg_stage(0, nseg);
                    Ok(())
                })
                .map_err(|e| anyhow!("capture commit nseg={nseg} stage 0: {e}"))?;
                self.commit_seg_graphs[nseg] = Some(g0);
            }
            let peer = self.peer.as_mut().expect("no peer");
            peer.ctx.gpu.bind().map_err(|e| anyhow!("{e}"))?;
            let g1 = Graph::capture(&peer.ctx.stream, |_| {
                // SAFETY: `self` is not otherwise touched during capture.
                unsafe { &*me }.enqueue_commit_seg_stage(1, nseg);
                Ok(())
            })
            .map_err(|e| anyhow!("capture commit nseg={nseg} stage 1: {e}"))?;
            self.peer.as_mut().expect("no peer").commit_graphs[nseg] = Some(g1);
            self.gpu.bind().map_err(|e| anyhow!("{e}"))?;
            return Ok(());
        }
        self.commit_seg_graphs[nseg]
            .as_ref()
            .unwrap()
            .launch(&self.stream)
            .map_err(|e| anyhow!("{e}"))?;
        let peer = self.peer.as_ref().expect("no peer");
        peer.commit_graphs[nseg]
            .as_ref()
            .unwrap()
            .launch(&peer.ctx.stream)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Stage `w`'s half of a segment commit (graph capture).
    fn enqueue_commit_seg_stage(&self, w: usize, nseg: usize) {
        let v = self.stage_view(w);
        self.kcopy_on(
            w,
            v.dev.buffer_addr(&v.s.rows),
            self.io_dev_on(w, Io::rows(self.hp.n_embd, MAX_T)),
            ROWS_U32 * 4,
        );
        for layer in v.layers {
            if let Some(g) = &layer.gdn {
                self.gdn_step_seg(w, g, layer.w_in.rows, nseg, true);
                self.gdn_commit_seg(w, g, layer.w_in.rows, nseg);
            }
        }
    }

    fn enqueue_commit_seg(&self, nseg: usize) {
        for w in 0..self.n_stages() {
            let st = self.stage_view(w);
            let (rows_addr, io_rows) = (
                st.dev.buffer_addr(&st.s.rows),
                self.io_dev_on(w, Io::rows(self.hp.n_embd, MAX_T)),
            );
            self.kcopy_on(w, rows_addr, io_rows, ROWS_U32 * 4);
            for layer in st.layers {
                if let Some(g) = &layer.gdn {
                    self.gdn_step_seg(w, g, layer.w_in.rows, nseg, true);
                    self.gdn_commit_seg(w, g, layer.w_in.rows, nseg);
                }
            }
        }
    }

    /// How many stages the layer list is split over (1, or 2 with `TANG_GPUS=2`).
    fn n_stages(&self) -> usize {
        1 + usize::from(self.peer.is_some())
    }

    /// Synchronise every stage's device.
    fn sync_all(&self) {
        self.dev.sync();
        if let Some(p) = &self.peer {
            p.ctx.sync().expect("GPU 1 sync");
        }
    }

    /// The last stage writes the window's sampled ids (and the MTP's ids/probabilities)
    /// into its own pinned staging; bring them back into GPU 0's host arena, which the
    /// host reads from. Host memcpy, `t * 4` bytes (and `2 * MAX_T` words for the MTP).
    fn collect_out(&mut self, t: usize, mtp: bool) {
        let Some(p) = self.peer.as_deref_mut() else {
            return;
        };
        let (src, dst) = (p.io.arena.as_ptr(), self.io.arena.as_ptr());
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.add(Io::IDS) as *const u8,
                dst.add(Io::IDS) as *mut u8,
                t * 4,
            );
            if mtp {
                std::ptr::copy_nonoverlapping(
                    src.add(Io::IDS) as *const u8,
                    dst.add(Io::IDS) as *mut u8,
                    2 * MAX_T * 4,
                );
                std::ptr::copy_nonoverlapping(
                    src.add(Io::MTP_PROB) as *const u8,
                    dst.add(Io::MTP_PROB) as *mut u8,
                    2 * MAX_T * 4,
                );
            }
        }
    }

    /// Build the batch decode path: decode steps of up to `rows` sequences (32 or 64; past
    /// `MAX_T`, [`decode_batch`](Self::decode_batch) runs them as one window). Compiles the
    /// Coder kernels for that many rows (~1 min) and allocates their scratch, attention
    /// partials and head for every row. One GPU only.
    pub fn enable_batch(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows % 32 == 0 && (32..=256).contains(&rows),
            "batch windows of {rows} rows (a multiple of 32)"
        );
        ensure!(self.peer.is_none(), "batch windows need one GPU");
        if self.batch.as_ref().is_some_and(|w| w.rows == rows) {
            return Ok(());
        }
        self.batch = None;
        self.graphs.retain(|k, _| k.0 <= MAX_T);
        let src = format!(
            "{}#define CX_GCHUNK {BATCH_GCHUNK}\n#define CX_SST_DOWN {}\n#define CX_SST {}\n#define CX_SNR_GU {}\n#define CX_SNR_DOWN {}\n{}{}{}",
            std::env::var("TANG_CODER_BATCH_DEFS")
                .unwrap_or_default()
                .split(',')
                .filter(|d| !d.is_empty())
                .map(|d| format!("#define {}\n", d.replace('=', " ")))
                .collect::<String>(),
            stream_sst_down(),
            stream_shape().0,
            stream_shape().1,
            stream_shape().2,
            ck::prelude(&self.shapes, rows),
            kernels::GEMV_SRC,
            ck::SRC
        );
        let cm = self
            .gpu
            .module(&src)
            .map_err(|e| anyhow!("coder kernels ({rows} batch rows): {e}"))?;
        let k = self.k.with_coder_module(cm)?;
        let n = self.hp.n_embd;
        let s = Scratch::new(
            &self.dev,
            &self.hp,
            &self.geo,
            n,
            self.max_ctx,
            rows,
            rows,
            rows,
        );
        let mut io = Io {
            arena: HostArena::new(
                Io::bytes(n, rows),
                ArenaOptions {
                    try_hugetlb: false,
                    thp: false,
                },
            )?,
        };
        io.arena
            .register(&self.gpu.ctx, &[])
            .map_err(|e| anyhow!("{e}"))?;
        let pr = self.layers.iter().map(|l| l.w_in.rows).max().unwrap_or(0);
        let proj = self.dev.alloc_f32(rows * pr);
        let kmax = self
            .layers
            .iter()
            .map(|l| l.w_out.k.max(l.w_in.k))
            .max()
            .unwrap_or(n);
        let xq = self
            .dev
            .alloc_f32((rows * kmax + rows * kmax / 32 * 4).div_ceil(4));
        self.dev.sync();
        self.batch = Some(Wide {
            rows,
            s,
            k,
            io,
            proj,
            xq,
            part: self.dev.alloc_f32(DMMA_PART),
        });
        Ok(())
    }

    /// The batch windows' width ([`enable_batch`](Self::enable_batch); 0: none).
    pub fn batch_rows(&self) -> usize {
        self.batch.as_ref().map_or(0, |w| w.rows)
    }

    /// Build the wide prefill path: windows of up to `rows` tokens (16..=512; 256 is the
    /// target). Compiles the Coder kernels for that many rows (~1 min) and allocates their
    /// scratch. [`prefill_wide`](Self::prefill_wide) uses it.
    pub fn enable_wide(&mut self, rows: usize) -> Result<()> {
        ensure!(
            (2 * MAX_T..=4096).contains(&rows),
            "wide windows of {rows} rows"
        );
        if self.wide.as_ref().is_some_and(|w| w.rows == rows) {
            return Ok(());
        }
        self.wide = None;
        if let Some(p) = self.peer.as_deref_mut() {
            p.wide = None;
        }
        let src = format!(
            "{}{}{}",
            ck::prelude(&self.shapes, rows),
            kernels::GEMV_SRC,
            ck::SRC
        );
        let cm = self
            .gpu
            .module(&src)
            .map_err(|e| anyhow!("coder kernels ({rows} rows): {e}"))?;
        let k = self.k.with_coder_module(cm)?;
        let n = self.hp.n_embd;
        let s = Scratch::new(
            &self.dev,
            &self.hp,
            &self.geo,
            n,
            self.max_ctx,
            rows,
            WIDE_ATTN_ROWS,
            MAX_T,
        );
        let mut io = Io {
            arena: HostArena::new(
                Io::bytes(n, rows),
                ArenaOptions {
                    try_hugetlb: false,
                    thp: false,
                },
            )?,
        };
        io.arena
            .register(&self.gpu.ctx, &[])
            .map_err(|e| anyhow!("{e}"))?;
        let pr = self.layers.iter().map(|l| l.w_in.rows).max().unwrap_or(0);
        let proj = self.dev.alloc_f32(rows * pr);
        let kmax = self
            .layers
            .iter()
            .map(|l| l.w_out.k.max(l.w_in.k))
            .max()
            .unwrap_or(n);
        let xq = self
            .dev
            .alloc_f32((rows * kmax + rows * kmax / 32 * 4).div_ceil(4));
        self.dev.sync();
        self.wide = Some(Wide {
            rows,
            s,
            k,
            io,
            proj,
            xq,
            part: self.dev.alloc_f32(1),
        });
        // The peer's own wide set: the same kernels compiled on its context, its scratch
        // and staging, sized by its layers.
        if let Some(p) = self.peer.as_deref_mut() {
            let cm = p
                .ctx
                .gpu
                .module(&src)
                .map_err(|e| anyhow!("coder kernels ({rows} rows, GPU 1): {e}"))?;
            let pk = p.k.with_coder_module(cm)?;
            let ps = Scratch::new(
                &p.dev,
                &self.hp,
                &self.geo,
                n,
                self.max_ctx,
                rows,
                WIDE_ATTN_ROWS,
                MAX_T,
            );
            let mut pio = Io {
                arena: HostArena::new(
                    Io::bytes(n, rows),
                    ArenaOptions {
                        try_hugetlb: false,
                        thp: false,
                    },
                )?,
            };
            pio.arena
                .register(&p.ctx.gpu.ctx, &[])
                .map_err(|e| anyhow!("{e}"))?;
            let prj = p
                .dev
                .alloc_f32(rows * p.layers.iter().map(|l| l.w_in.rows).max().unwrap_or(0));
            let pkm = p
                .layers
                .iter()
                .map(|l| l.w_out.k.max(l.w_in.k))
                .max()
                .unwrap_or(n);
            let pxq = p
                .dev
                .alloc_f32((rows * pkm + rows * pkm / 32 * 4).div_ceil(4));
            p.dev.sync();
            p.wide = Some(Wide {
                rows,
                s: ps,
                k: pk,
                io: pio,
                proj: prj,
                xq: pxq,
                part: p.dev.alloc_f32(1),
            });
            p.ctx.sync()?;
        }
        Ok(())
    }

    /// One wide window (staged by `stage_rows` with `in_wide` set): every layer over its `t`
    /// rows. No head: prefill doesn't need these rows' logits.
    fn enqueue_wide(&self, t: usize) {
        self.enqueue_inputs(t);
        self.norm_q8(0, None, &self.layers[0].attn_norm, t);
        for w in 0..self.n_stages() {
            let st = self.stage_view(w);
            for l in 0..st.layers.len() {
                let layer = &st.layers[l];
                self.enqueue_mixer(w, l, t, false, 0, 1);
                self.norm_q8(w, Some(st.dev.buffer_addr(&st.s.mix)), &layer.post_norm, t);
                let next = if l + 1 < st.layers.len() {
                    &st.layers[l + 1].attn_norm
                } else if w + 1 < self.n_stages() {
                    // The next stage's first norm, read over the peer link.
                    &self.stage_view(w + 1).layers[0].attn_norm
                } else if self.peer.is_some() {
                    &self.peer.as_ref().unwrap().out_norm
                } else {
                    &self.out_norm
                };
                self.enqueue_moe(w, l, t, next, false);
            }
            if w + 1 < self.n_stages() {
                self.enqueue_boundary(w, t);
            }
        }
    }

    /// Between stage `w` and `w + 1`: record the last layer's state, wait on the next stage,
    /// copy `x`, `mxq` and `r` over, so stage `w + 1` starts exactly where `w`'s last
    /// `moe_add_norm` left `x`/`mxq`/`r`. Layer indices in `enqueue_moe`'s "next" are the
    /// *next stage's* first layer, read over the peer link.
    fn enqueue_boundary(&self, w: usize, t: usize) {
        let (a, b) = (self.stage_view(w), self.stage_view(w + 1));
        // Order: stage w+1's stream waits for stage w's work before it reads the copies.
        let done = gpu::Event::new(false).expect("event");
        done.record(a.stream).expect("record");
        b.stream.wait(&done).expect("wait");
        let n = self.hp.n_embd;
        // x, then r (t rows of n f32), then mxq (t rows of n + n/8 bytes codes+scales).
        // Peer copies are enqueued on the *destination* device's stream (the destination
        // context must be current), after its wait above.
        let peer = &self.peer.as_ref().expect("no peer").ctx;
        let xq_bytes = t * (n + n / 32 * 4);
        let tw = self.tw();
        for (dst, src, bytes) in [
            (
                b.dev.buffer_addr(&b.s.x),
                a.dev.buffer_addr(&a.s.x),
                t * n * 4,
            ),
            (
                b.dev.buffer_addr(&b.s.r),
                a.dev.buffer_addr(&a.s.r),
                t * n * 4,
            ),
            (
                b.dev.buffer_addr(&b.s.mxq),
                a.dev.buffer_addr(&a.s.mxq),
                xq_bytes,
            ),
            (b.dev.buffer_addr(&b.s.ctl), a.dev.buffer_addr(&a.s.ctl), 32),
            (
                b.dev.buffer_addr(&b.s.rows),
                a.dev.buffer_addr(&a.s.rows),
                ROWS_PER_T * tw * 4,
            ),
        ] {
            // SAFETY: plain peer copy of `bytes` from GPU 0's buffer to GPU 1's, on GPU 1's
            // stream, after the wait above (so after GPU 0's last layer has written them).
            let r = unsafe {
                cudarc::driver::sys::cuMemcpyPeerAsync(
                    dst,
                    peer.gpu.ctx.cu_ctx(),
                    src,
                    a.dev.cuda_context().cu_ctx(),
                    bytes,
                    peer.stream.0,
                )
            };
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!("boundary copy: {r:?}");
            }
        }
    }

    /// [`prefill_slot`](Self::prefill_slot) in wide windows ([`enable_wide`](Self::enable_wide);
    /// without it, the same as `prefill_slot`): every token but the last in windows of up to
    /// the wide width, then the last as a normal window, which gives the next token (and the
    /// logits) exactly as `prefill_slot` would. The state and K/V it leaves are bitwise
    /// `prefill_slot`'s: every row's arithmetic is the narrow kernels'.
    /// The wide windows' width ([`enable_wide`](Self::enable_wide); 0: none).
    pub fn wide_rows(&self) -> usize {
        self.wide.as_ref().map_or(0, |w| w.rows)
    }

    pub fn prefill_wide(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        let Some(rows) = self.wide.as_ref().map(|w| w.rows) else {
            return self.prefill_slot(slot, ids);
        };
        ensure!(!ids.is_empty(), "empty prompt");
        if ids.len() == 1 {
            return self.prefill_slot(slot, ids);
        }
        let body = &ids[..ids.len() - 1];
        let mut pos = self.slot_tokens(slot).len();
        ensure!(
            pos + ids.len() <= self.max_ctx,
            "context {} > max {}",
            pos + ids.len(),
            self.max_ctx
        );
        self.slot_tokens_mut(slot)?.extend_from_slice(body);
        let mut at = 0;
        while at < body.len() {
            let t = (body.len() - at).min(rows);
            let rws: Vec<(usize, usize, u32)> = body[at..at + t]
                .iter()
                .enumerate()
                .map(|(i, &k)| (slot, pos + i, k))
                .collect();
            let w0 = Instant::now();
            self.in_wide.set(true);
            self.wide_pos0.set(pos);
            let r = self
                .stage_rows(&rws, &[[0, t as u32, slot as u32, t as u32]])
                .map(|_| self.enqueue_wide(t));
            // The pinned staging is reused by the next window.
            self.sync_all();
            self.prof_collect();
            self.in_wide.set(false);
            r?;
            if std::env::var("TANG_WIDE_TIMES").is_ok_and(|v| v == "1") {
                eprintln!(
                    "wide window at {pos}+{t}: {:.1} ms",
                    w0.elapsed().as_secs_f64() * 1e3
                );
            }
            pos += t;
            at += t;
        }
        self.prefill_slot(slot, &ids[ids.len() - 1..])
    }

    /// One decode step for several sequences at once: `(slot, token)` per row, distinct
    /// slots, at most `MAX_T`. Feeds each slot its token and returns each row's next token
    /// (greedy). A slot's result is the one a window of that slot alone would give.
    pub fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        let b = rows.len();
        let cap = MAX_T.max(self.batch_rows());
        ensure!((1..=cap).contains(&b), "{b} rows (at most {cap})");
        ensure!(self.sampler.is_none(), "slot windows sample per row, not with the engine sampler");
        // 1..=MAX_T rows: the narrow module, always (`crate::batch::batch_module` says why);
        // concurrency dropping to 1 under `serve --batch 64` lands here.
        let wide = crate::batch::batch_module(b, MAX_T, self.batch_rows());
        let mut staged = Vec::with_capacity(b);
        let mut segs = Vec::with_capacity(b);
        let mut end = 0;
        for (i, &(slot, tok)) in rows.iter().enumerate() {
            ensure!(
                rows[..i].iter().all(|r| r.0 != slot),
                "slot {slot} twice in one step"
            );
            let toks = self.slot_tokens_mut(slot)?;
            let pos = toks.len();
            toks.push(tok);
            end = end.max(pos + 1);
            staged.push((slot, pos, tok));
            segs.push([i as u32, 1, slot as u32, 1]);
        }
        ensure!(end <= self.max_ctx, "context {end} > max {}", self.max_ctx);
        let rw: Vec<(usize, usize, usize)> =
            staged.iter().enumerate().map(|(i, r)| (i, r.0, r.1)).collect();
        self.stage_row_words(&rw);
        let w0 = Instant::now();
        let nchunk = self.nchunk(end);
        let st = WinStats {
            t: b,
            ..Default::default()
        };
        // Past MAX_T rows: one window of the batch module (its kernels, scratch and staging).
        self.in_batch.set(wide);
        let out = self
            .stage_rows(&staged, &segs)
            .and_then(|_| self.run_staged(st, w0, b, false, nchunk, b, None));
        self.in_batch.set(false);
        let out = out?;
        if self.mtp_on() && !wide {
            let nexts: Vec<[u32; 1]> = out.iter().map(|&k| [k]).collect();
            let groups: Vec<(usize, usize, usize, &[u32])> = staged
                .iter()
                .enumerate()
                .map(|(i, r)| (r.0, r.1, i, &nexts[i][..]))
                .collect();
            let chains = self.mtp_groups(&groups, true)?;
            for (r, c) in staged.iter().zip(chains) {
                self.slot_chain[r.0] = c.0;
            }
        }
        Ok(out)
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            if let Some(g) = l.gdn.as_mut() {
                self.dev.zero_buffer(&mut g.state);
                self.dev.zero_buffer(&mut g.hist);
            }
        }
        if let Some(p) = self.peer.as_deref_mut() {
            for l in &mut p.layers {
                if let Some(g) = l.gdn.as_mut() {
                    p.dev.zero_buffer(&mut g.state);
                    p.dev.zero_buffer(&mut g.hist);
                }
            }
            // GPU 1's slot states (its `slots` hold the buffers; slot 0's are the layers' own).
            for sl in p.slots.iter_mut().skip(1).flatten() {
                for (state, hist) in sl.0.iter_mut().flatten() {
                    p.dev.zero_buffer(state);
                    p.dev.zero_buffer(hist);
                }
            }
            p.ctx.sync().expect("GPU 1 sync");
        }
        // GPU 0's slot states.
        for sl in self.slots.iter_mut().skip(1).flatten() {
            for (state, hist) in sl.gdn.iter_mut().flatten() {
                self.dev.zero_buffer(state);
                self.dev.zero_buffer(hist);
            }
        }
        self.dev.sync();
        self.tokens.clear();
    }

    // ---- launch helpers

    /// Stage `which` (0: GPU 0, 1: the peer): its device, stream, kernels, scratch, staging
    /// and layers. The wide-window variants (`sc`, `kn`, `iox`, `tw`, `proj`) resolve within
    /// the stage.
    fn stage_view(&self, which: usize) -> Stage<'_> {
        let in_wide = self.in_wide.get();
        if which == 0 {
            // The wide set's kernels during a wide window (the batch set's during a batch
            // window): the narrow ones are compiled for `MAX_T` rows and overrun the wider
            // scratch.
            let ws = self.wide_on();
            Stage {
                dev: &self.dev,
                stream: &self.stream,
                k: ws.map_or(&self.k, |w| &w.k),
                s: ws.map_or(&self.s, |w| &w.s),
                io: ws.map_or(&self.io, |w| &w.io),
                nk: &self.k,
                ns: &self.s,
                layers: &self.layers,
                side: &self.side,
                wide: ws,
            }
        } else {
            let p = self.peer.as_ref().expect("no peer stage");
            Stage {
                dev: &p.dev,
                stream: &p.ctx.stream,
                k: if in_wide {
                    p.wide.as_ref().map(|w| &w.k).unwrap_or(&p.k)
                } else {
                    &p.k
                },
                s: if in_wide {
                    p.wide.as_ref().map(|w| &w.s).unwrap_or(&p.s)
                } else {
                    &p.s
                },
                io: if in_wide {
                    p.wide.as_ref().map(|w| &w.io).unwrap_or(&p.io)
                } else {
                    &p.io
                },
                nk: &p.k,
                ns: &p.s,
                layers: &p.layers,
                side: &p.side,
                wide: p.wide.as_ref().filter(|_| in_wide),
            }
        }
    }

    /// Bound pairs (per stage): which device and Gpu the kernels/launches belong to.
    fn a(&self, b: &B) -> u64 {
        self.dev.buffer_addr(b)
    }

    /// Stage `which`'s address of `b`.
    fn a_on(&self, which: usize, b: &B) -> u64 {
        if which == 0 {
            self.dev.buffer_addr(b)
        } else {
            self.peer.as_ref().expect("no peer").dev.buffer_addr(b)
        }
    }

    /// The active window's scratch, Coder kernels, staging and row capacity: the wide set while
    /// a wide prefill window is being enqueued, else the normal one.
    fn wide_on(&self) -> Option<&Wide> {
        if self.in_batch.get() {
            return self.batch.as_ref();
        }
        self.wide.as_ref().filter(|_| self.in_wide.get())
    }
    fn iox(&self) -> &Io {
        self.wide_on().map_or(&self.io, |w| &w.io)
    }
    fn tw(&self) -> usize {
        self.wide_on().map_or(MAX_T, |w| w.rows)
    }

    /// A launch with dynamic shared memory on any stream.
    unsafe fn launch_on(
        &self,
        stream: &Stream,
        f: Fun,
        grid: (u32, u32, u32),
        block: u32,
        smem: u32,
        args: &mut [*mut std::ffi::c_void],
    ) {
        if self.skip(self.kname(f)) {
            return;
        }
        gpu::launch(f, grid, (block, 1, 1), smem, stream, args).expect("launch");
        if self.prof.is_some() && std::ptr::eq(stream, &*self.stream) {
            self.mark(self.kname(f));
        }
    }

    /// Stage sweep: is `name` filtered out?
    fn skip(&self, name: &'static str) -> bool {
        self.only.get().is_some_and(|o| o != name)
    }

    fn kname(&self, f: Fun) -> &'static str {
        let n = self.kname_in(&self.k, f);
        match (n, self.wide_on()) {
            ("other", Some(w)) => self.kname_in(&w.k, f),
            _ => n,
        }
    }

    fn kname_in(&self, k: &Kern, f: Fun) -> &'static str {
        [
            (k.norm_q8, "norm_q8"),
            (k.route, "route"),
            (k.moe_gu, "moe_gu"),
            (k.silu_q, "silu_q"),
            (k.moe_down, "moe_down"),
            (k.moe_gu_d[0], "moe_gu"),
            (k.moe_gu_d[1], "moe_gu"),
            (k.moe_down_d[0], "moe_down"),
            (k.moe_down_d[1], "moe_down"),
            (k.moe_down_df[0], "moe_down"),
            (k.moe_down_df[1], "moe_down"),
            (k.moe_gu_db, "moe_gu"),
            (k.moe_down_db, "moe_down"),
            (k.moe_plan, "route"),
            (k.moe_gu_g, "moe_gu"),
            (k.moe_down_g, "moe_down"),
            (k.moe_gu_s, "moe_gu"),
            (k.moe_down_s, "moe_down"),
            (k.moe_gu_m, "moe_gu"),
            (k.moe_down_m, "moe_down"),
            (k.moe_gemm_gu, "moe_gu"),
            (k.moe_gemm_down, "moe_down"),
            (k.moe_add_norm, "moe_add_norm"),
            (k.prep, "attn_prep"),
            (k.part, "attn_part"),
            (k.merge, "attn_merge"),
            (k.fa2, "attn_fa2"),
            (k.gdn_step, "gdn_conv_step"),
            (k.gdn_commit, "gdn_commit"),
            (k.gdn_step_seg, "gdn_conv_step"),
            (k.gdn_commit_seg, "gdn_commit"),
            (k.copy, "copy"),
            (k.argmax1, "argmax"),
            (k.argmax2, "argmax"),
            (k.q8, "moe_q8"),
        ]
        .into_iter()
        .find(|(g, _)| *g == f)
        .map_or("other", |(_, n)| n)
    }

    /// Profiling: everything enqueued since the previous mark is charged to `name`.
    fn mark(&self, name: &'static str) {
        if let Some(p) = &self.prof {
            let ev = gpu::Event::new(true).expect("event");
            ev.record(&self.stream).expect("record");
            p.borrow_mut().marks.push((name, ev));
        }
    }

    /// Fold the window's marks into the totals.
    fn prof_collect(&self) {
        let Some(p) = &self.prof else { return };
        let mut p = p.borrow_mut();
        let marks = std::mem::take(&mut p.marks);
        for w in marks.windows(2) {
            let ms = w[1].1.since(&w[0].1).unwrap_or(0.0) as f64;
            let e = p.total.entry(w[1].0).or_default();
            e.0 += ms;
            e.1 += 1;
        }
    }

    /// Forget the profile so far.
    pub fn prof_reset(&self) {
        if let Some(p) = &self.prof {
            let mut p = p.borrow_mut();
            p.marks.clear();
            p.total.clear();
        }
    }

    /// The profile so far, by stage (`TANG_CODER_PROF=1`).
    pub fn prof_report(&self) -> Option<String> {
        let p = self.prof.as_ref()?.borrow();
        let tot: f64 = p.total.values().map(|v| v.0).sum();
        let mut v: Vec<_> = p.total.iter().collect();
        v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        Some(
            v.iter()
                .map(|(k, (ms, n))| {
                    format!("  {k:14} {ms:9.2} ms  x{n:<6} {:5.1}%\n", 100.0 * ms / tot)
                })
                .collect(),
        )
    }

    /// Stage `w`'s device pointer for pinned-staging byte `off` (GPU 1 maps the same host
    /// arena through its own context).
    fn io_dev_on(&self, w: usize, off: usize) -> u64 {
        let st = self.stage_view(w);
        if w == 0 {
            return st.io.arena.device_ptr(off).expect("io arena is mapped");
        }
        // The peer's own registered arena (registered from its context in `build_peer`).
        st.io.arena.device_ptr(off).expect("io arena is mapped")
    }

    /// `bytes` (a multiple of 4) from `src` to `dst` with the copy kernel on stage `w`'s
    /// stream (device or mapped host).
    fn kcopy_on(&self, w: usize, dst: u64, src: u64, bytes: usize) {
        let st = self.stage_view(w);
        let n = (bytes / 4) as i32;
        unsafe {
            self.launch_on(
                st.stream,
                st.k.copy,
                ((n as u32).div_ceil(256).max(1), 1, 1),
                256,
                0,
                tang_moe::args![dst, src, n],
            )
        };
    }

    /// `out[t][off + row] = W · x[t]` for every row of `w` (f32 `x`, `w.k` wide), outputs
    /// `ostride` apart; `x`'s int8 codes are made here (fe_q8) unless `xq` already holds them.
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        &self,
        wi: usize,
        name: &'static str,
        w: &Dense,
        x: u64,
        xq: Option<u64>,
        out: u64,
        t: usize,
        ostride: usize,
    ) {
        if self.skip(name) {
            return;
        }
        let v = self.stage_view(wi);
        // Wide windows: the tensor-core GEMM over every row (TANG_CODER_WGEMM=0: the GEMVs below).
        if t > MAX_T
            && self.var.wgemm
            && w.gemm_ok
            && (!self.in_batch.get() || self.var.bgemm)
        {
            if let Some(wd) = v.wide {
                let xq = match xq {
                    Some(q) => q,
                    None => {
                        let q = v.dev.buffer_addr(&wd.xq);
                        let ki = w.k as i32;
                        unsafe {
                            gpu::launch(
                                wd.k.q8,
                                ((w.k / 32) as u32, t as u32, 1),
                                (32, 1, 1),
                                0,
                                v.stream,
                                tang_moe::args![x, q, ki],
                            )
                            .expect("fe_q8");
                        }
                        q
                    }
                };
                let (sp, ns, ki, os, ti) = (
                    v.dev.buffer_addr(&w.segs),
                    w.nseg,
                    w.k as i32,
                    ostride as i32,
                    t as i32,
                );
                unsafe {
                    gpu::launch(
                        wd.k.gemm_dense,
                        (w.rows.div_ceil(128) as u32, t.div_ceil(64) as u32, 1),
                        (256, 1, 1),
                        0,
                        v.stream,
                        tang_moe::args![sp, ns, xq, ki, out, os, ti],
                    )
                    .expect("cx_gemm_dense");
                }
                // Debug: the GEMVs into a scratch, compared (first 2 rows' outputs of each kind).
                if env_is("TANG_CODER_WGEMM_CHECK", "1") {
                    let tmp = v.dev.alloc_f32(t * ostride);
                    let tp = v.dev.buffer_addr(&tmp);
                    for r0 in (0..t).step_by(MAX_T) {
                        let c = MAX_T.min(t - r0);
                        self.gemv_narrow(
                            wi,
                            w,
                            x + (r0 * w.k * 4) as u64,
                            tp + (r0 * ostride * 4) as u64,
                            c,
                            ostride,
                        );
                    }
                    v.dev.sync();
                    let n_out = t * ostride;
                    let mut a = vec![0f32; n_out];
                    unsafe {
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            a.as_mut_ptr() as *mut std::ffi::c_void,
                            out,
                            n_out * 4,
                        );
                    }
                    let b = v.dev.download(&tmp);
                    let (mut worst, mut sum, mut cnt) = (0f64, 0f64, 0usize);
                    for tt in 0..t {
                        for r in 0..w.rows {
                            let i = tt * ostride + r;
                            let (p, q) = (a[i] as f64, b[i] as f64);
                            worst = worst.max((p - q).abs() / q.abs().max(1e-3));
                            sum += (p - q).abs();
                            cnt += 1;
                        }
                    }
                    eprintln!(
                        "WGEMM check dense {name}: {cnt} values, mean |diff| {:.3e}, worst rel {worst:.3e}",
                        sum / cnt as f64
                    );
                }
                self.mark(name);
                return;
            }
        }
        // Wide windows: the same GEMV over MAX_T rows at a time, each chunk's int8 made here
        // (fe_q8's rule, the one norm_q8 applies), so every row's sum is the one a narrow window
        // gives it.
        if t > MAX_T
            && self.in_batch.get()
            && self.var.dmma
            && w.float1
            && env_is("TANG_CODER_ROUTERB", "1")
            && w.k.is_multiple_of(128)
            && t <= 64
            && (w.k / 128) * w.rows * t <= DMMA_PART
        {
            if let Some(wd) = v.wide {
                let ns = w.nseg;
                let (sp, ki, ti, part, rt, si, os) = (
                    v.dev.buffer_addr(&w.segs),
                    w.k as i32,
                    t as i32,
                    v.dev.buffer_addr(&wd.part),
                    w.rows as i32,
                    (w.k / 128) as i32,
                    ostride as i32,
                );
                unsafe {
                    gpu::launch(
                        wd.k.router_b,
                        (w.rows.div_ceil(16) as u32, (w.k / 128) as u32, 1),
                        (256, 1, 1),
                        0,
                        v.stream,
                        tang_moe::args![sp, ns, x, ki, ti, part, rt],
                    )
                    .expect("cx_router_b");
                    gpu::launch(
                        wd.k.mma_q8x_red,
                        (w.rows.div_ceil(128) as u32, t as u32, 1),
                        (128, 1, 1),
                        0,
                        v.stream,
                        tang_moe::args![sp, ns, part, si, rt, ti, out, os],
                    )
                    .expect("cx_mma_q8x_red");
                }
                self.mark(name);
                return;
            }
        }
        if t > MAX_T && self.in_batch.get() && self.var.dmma && w.q8x_ok && t <= 64 {
            if let Some(wd) = v.wide {
                self.gemm_q8x(wi, wd, w, x, xq, out, t, ostride);
                self.mark(name);
                return;
            }
        }
        if t > MAX_T && self.in_batch.get() && w.k.is_multiple_of(self.var.kq * 256) && !gemv_v1() {
            if let Some(wd) = v.wide {
                self.gemv_batch(wi, wd, w, x, xq, out, t, ostride);
                self.mark(name);
                return;
            }
        }
        if t > MAX_T {
            for r0 in (0..t).step_by(MAX_T) {
                let c = MAX_T.min(t - r0);
                self.gemv(
                    wi,
                    name,
                    w,
                    x + (r0 * w.k * 4) as u64,
                    None,
                    out + (r0 * ostride * 4) as u64,
                    c,
                    ostride,
                );
            }
            return;
        }
        let (sp, ns, os, r16, ki) = (
            v.dev.buffer_addr(&w.segs),
            w.nseg,
            ostride as i32,
            w.row16,
            w.k as i32,
        );
        let xq = match xq {
            Some(q) => q,
            None => {
                let q = v.dev.buffer_addr(&v.ns.xq8);
                unsafe {
                    gpu::launch(
                        v.nk.q8,
                        ((w.k / 32) as u32, t as u32, 1),
                        (32, 1, 1),
                        0,
                        v.stream,
                        tang_moe::args![x, q, ki],
                    )
                    .expect("fe_q8");
                }
                q
            }
        };
        let kq = self.var.kq;
        if w.k.is_multiple_of(kq * 256) && !gemv_v1() {
            let f = match kq {
                8 => v.nk.gemvk8[t - 1],
                2 => v.nk.gemvk2[t - 1],
                1 => v.nk.gemvk1[t - 1],
                _ => v.nk.gemvk[t - 1],
            };
            unsafe {
                gpu::launch(
                    f,
                    (w.rows.div_ceil(8 / kq) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    v.stream,
                    tang_moe::args![sp, ns, x, xq, ki, out, os],
                )
                .expect("cx_gemvk")
            };
        } else {
            unsafe {
                gpu::launch(
                    v.nk.gemv8[t - 1],
                    (w.rows.div_ceil(8) as u32, 1, 1),
                    (128, 1, 1),
                    (8 * 16 * r16) as u32,
                    v.stream,
                    tang_moe::args![sp, ns, x, xq, ki, out, os, r16],
                )
                .expect("fe_gemv8")
            };
        }
        self.mark(name);
    }

    /// A batch window's dense Q8X matrix on int8 tensor cores (`cx_mma_q8x`), all `t` rows in
    /// one pass; K split so a small matrix still fills the GPU (partials summed in order).
    #[allow(clippy::too_many_arguments)]
    fn gemm_q8x(
        &self,
        wi: usize,
        wd: &Wide,
        w: &Dense,
        x: u64,
        xq: Option<u64>,
        out: u64,
        t: usize,
        ostride: usize,
    ) {
        let v = self.stage_view(wi);
        let k = w.k;
        let ki = k as i32;
        let xq = match xq {
            Some(q) => q,
            None => {
                let q = v.dev.buffer_addr(&wd.xq);
                unsafe {
                    gpu::launch(
                        wd.k.q8,
                        ((k / 32) as u32, t as u32, 1),
                        (32, 1, 1),
                        0,
                        v.stream,
                        tang_moe::args![x, q, ki],
                    )
                    .expect("fe_q8");
                }
                q
            }
        };
        let (mseg, mn, mrows) = w.mma.as_ref().expect("q8x segments");
        let tiles = mrows / 16;
        let mut splits = 1usize;
        while tiles * splits < 2048
            && (k / 256).is_multiple_of(splits * 2)
            && splits * 2 * mrows * t <= DMMA_PART
        {
            splits *= 2;
        }
        let (sp, ns, os, ti, si, rt) = (
            v.dev.buffer_addr(mseg),
            *mn,
            ostride as i32,
            t as i32,
            splits as i32,
            *mrows as i32,
        );
        let part = v.dev.buffer_addr(&wd.part);
        unsafe {
            gpu::launch(
                wd.k.mma_q8x,
                (mrows.div_ceil(64) as u32, splits as u32, 1),
                (128, 1, 1),
                0,
                v.stream,
                tang_moe::args![sp, ns, xq, ki, out, os, ti, si, part, rt, x],
            )
            .expect("cx_mma_q8x");
            if splits > 1 {
                gpu::launch(
                    wd.k.mma_q8x_red,
                    (mrows.div_ceil(128) as u32, t as u32, 1),
                    (128, 1, 1),
                    0,
                    v.stream,
                    tang_moe::args![sp, ns, part, si, rt, ti, out, os],
                )
                .expect("cx_mma_q8x_red");
            }
        }
        if let Some(r) = &w.rest {
            if r.k.is_multiple_of(self.var.kq * 256) {
                self.gemv_batch(wi, wd, r, x, Some(xq), out, t, ostride);
            }
        }
    }

    /// A batch window's GEMV: the batch module's K-split kernels over up to 32 rows a launch,
    /// reading the module's int8 activations in place (`xq`, or `x` quantized here into the
    /// module's staging). Each row's sum is the narrow `cx_gemvk`'s at the same Q, bitwise.
    #[allow(clippy::too_many_arguments)]
    fn gemv_batch(
        &self,
        wi: usize,
        wd: &Wide,
        w: &Dense,
        x: u64,
        xq: Option<u64>,
        out: u64,
        t: usize,
        ostride: usize,
    ) {
        let v = self.stage_view(wi);
        let k = w.k;
        let ki = k as i32;
        let xq = match xq {
            Some(q) => q,
            None => {
                let q = v.dev.buffer_addr(&wd.xq);
                unsafe {
                    gpu::launch(
                        wd.k.q8,
                        ((k / 32) as u32, t as u32, 1),
                        (32, 1, 1),
                        0,
                        v.stream,
                        tang_moe::args![x, q, ki],
                    )
                    .expect("fe_q8");
                }
                q
            }
        };
        let kq = self.var.kq;
        let qi = match kq {
            1 => 0,
            2 => 1,
            8 => 3,
            _ => 2,
        };
        let (sp, ns, os) = (v.dev.buffer_addr(&w.segs), w.nseg, ostride as i32);
        let xd0 = xq + (wd.rows * k) as u64;
        let mut r0 = 0;
        while r0 < t {
            let c = (t - r0).min(32);
            let (ti, tt) = match c {
                0..=8 => (0, 8),
                9..=16 => (1, 16),
                _ => (2, 32),
            };
            // Rows past `t` in the last launch read stale activations and write rows of
            // `out` no one reads (the scratch holds `wd.rows` rows, a multiple of 32).
            let (xr, qr, dr, orow) = (
                x + (r0 * k * 4) as u64,
                xq + (r0 * k) as u64,
                xd0 + (r0 * (k / 32) * 4) as u64,
                out + (r0 * ostride * 4) as u64,
            );
            unsafe {
                gpu::launch(
                    wd.k.gemvkb[qi][ti],
                    (w.rows.div_ceil(8 / kq) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    v.stream,
                    tang_moe::args![sp, ns, xr, qr, dr, ki, orow, os],
                )
                .expect("cx_gemvkb")
            };
            r0 += tt;
        }
    }

    /// Stage `w`'s scratch (the wide set's during a wide window): the per-stage resolution
    /// of the old `sc()`.
    fn sc_on(&self, w: usize) -> &Scratch {
        self.stage_view(w).s
    }

    /// Stage `w`'s kernels.
    fn kn_on(&self, w: usize) -> &Kern {
        self.stage_view(w).k
    }

    /// Stage `w`'s stream.
    fn stream_on(&self, w: usize) -> &Stream {
        self.stage_view(w).stream
    }

    /// Stage `w`'s ротable: the projection buffer of the stage's layers, or the wide
    /// window's shared one.
    fn proj_on(&self, w: usize, b: &B) -> u64 {
        let v = self.stage_view(w);
        match v.wide {
            Some(wd) => v.dev.buffer_addr(&wd.proj),
            None => v.dev.buffer_addr(b),
        }
    }

    /// Stage `w`'s rope tables.
    fn rope_on(&self, w: usize) -> &(B, B) {
        if w == 0 {
            &self.rope
        } else {
            &self.peer.as_ref().expect("no peer").rope
        }
    }

    /// The narrow GEMV for `t <= MAX_T` rows (debug comparisons).
    fn gemv_narrow(&self, wi: usize, w: &Dense, x: u64, out: u64, t: usize, ostride: usize) {
        let wide = self.in_wide.replace(false);
        self.gemv(wi, "check", w, x, None, out, t, ostride);
        self.in_wide.set(wide);
    }

    /// `r += y` (when given), then `s.x = rmsnorm(r) * w` and its int8 codes in `s.mxq`.
    fn norm_q8(&self, w_stage: usize, add: Option<u64>, w: &B, t: usize) {
        let st = self.stage_view(w_stage);
        let (r, y, wp, x, xq, n, eps, ad) = (
            st.dev.buffer_addr(&st.s.r),
            add.unwrap_or(0),
            st.dev.buffer_addr(w),
            st.dev.buffer_addr(&st.s.x),
            st.dev.buffer_addr(&st.s.mxq),
            self.hp.n_embd as i32,
            self.eps,
            add.is_some() as i32,
        );
        unsafe {
            self.launch_on(
                st.stream,
                st.k.norm_q8,
                (t as u32, 1, 1),
                1024,
                0,
                tang_moe::args![r, y, wp, x, xq, n, eps, ad],
            )
        };
    }

    /// Split-K chunks a window at `pos0 + t` runs over: a power-of-two bucket of the context,
    /// so a window graph is reused until the context doubles.
    fn nchunk(&self, end: usize) -> usize {
        let bucket = end.next_power_of_two().max(4 * ck::attn_chunk());
        bucket.min(self.max_ctx) / ck::attn_chunk()
    }

    /// Key splits for the FA2 attention of a wide window of `t` rows, or `None` for split-K
    /// (not a wide window, `TANG_CODER_ATTN` not `fa2`, or partials that wouldn't fit).
    fn fa2_splits(&self, _w: usize, t: usize) -> Option<usize> {
        if !(self.in_wide.get() && self.var.attn_fa2) {
            return None;
        }
        let hp = &self.hp;
        // The partials buffer holds WIDE_ATTN_ROWS rows × nchunk_max splits (Scratch::new).
        let cap = WIDE_ATTN_ROWS * (self.max_ctx / ck::attn_chunk()) / t;
        let tiles = (t * (hp.n_head / hp.n_head_kv)).div_ceil(ck::FA2_BM) * hp.n_head_kv;
        let ns = ck::fa2_splits(tiles, self.wide_pos0.get() + t, sm_count(), cap);
        (ns > 0).then_some(ns)
    }

    // ---- the window

    fn enqueue_inputs(&self, t: usize) {
        self.mark("start");
        let n_rows = ROWS_PER_T * self.tw();
        let s = self.stage_view(0).s;
        self.kcopy_on(
            0,
            self.dev.buffer_addr(&s.ctl),
            self.io_dev_on(0, Io::CTL),
            32,
        );
        self.kcopy_on(
            0,
            self.dev.buffer_addr(&s.rows),
            self.io_dev_on(0, Io::rows(self.hp.n_embd, self.tw())),
            n_rows * 4,
        );
        self.kcopy_on(
            0,
            self.dev.buffer_addr(&s.r),
            self.io_dev_on(0, Io::EMB),
            t * self.hp.n_embd * 4,
        );
    }

    /// Stage `w`'s layer `l`, or the MTP layer for `MTP_L`.
    fn lay(&self, w: usize, l: usize) -> &Layer {
        if l == MTP_L {
            return &self.mtp.as_ref().expect("MTP layer").layer;
        }
        let v = self.stage_view(w);
        &v.layers[l]
    }

    /// Layer `l` of stage `w`'s mixer (input `s.x` / `s.mxq`) into `s.mix`.
    fn enqueue_mixer(
        &self,
        w: usize,
        l: usize,
        t: usize,
        verify: bool,
        nchunk: usize,
        nseg: usize,
    ) {
        let layer = self.lay(w, l);
        let s = self.sc_on(w);
        let (x, xq) = (self.a_on(w, &s.x), Some(self.a_on(w, &s.mxq)));
        let eps = self.eps;
        let n = self.hp.n_embd;
        if let Some(g) = &layer.gdn {
            let stride = layer.w_in.rows;
            self.gemv(
                w,
                "gdn_in",
                &layer.w_in,
                x,
                xq,
                self.proj_on(w, &g.proj),
                t,
                stride,
            );
            self.gdn_step_seg(w, g, stride, nseg, !verify);
            if !verify {
                self.gdn_commit_seg(w, g, stride, nseg);
            }
            self.gemv(
                w,
                "gdn_out",
                &layer.w_out,
                self.a_on(w, &s.y),
                None,
                self.a_on(w, &s.mix),
                t,
                n,
            );
        }
        if let Some(at) = &layer.attn {
            let stride = layer.w_in.rows;
            self.gemv(
                w,
                "attn_in",
                &layer.w_in,
                x,
                xq,
                self.proj_on(w, &at.proj),
                t,
                stride,
            );
            let hp = &self.hp;
            let (proj, st, qn, kn, rc, rs, q, kt, vt, rows) = (
                self.proj_on(w, &at.proj),
                stride as i32,
                self.a_on(w, &at.qn),
                self.a_on(w, &at.kn),
                self.a_on(w, &self.rope_on(w).0),
                self.a_on(w, &self.rope_on(w).1),
                self.a_on(w, &s.q),
                self.a_on(w, &at.ktab),
                self.a_on(w, &at.vtab),
                self.a_on(w, &s.rows),
            );
            let (part, attn) = (self.a_on(w, &s.part), self.a_on(w, &s.attn));
            unsafe {
                self.launch_on(
                    self.stream_on(w),
                    self.kn_on(w).prep,
                    (t as u32, (hp.n_head + hp.n_head_kv) as u32, 1),
                    hp.head_dim as u32,
                    0,
                    tang_moe::args![proj, st, qn, kn, rc, rs, rows, q, kt, vt, eps],
                );
            }
            // Partials and merge a slice of rows at a time in wide windows (the partials buffer
            // holds WIDE_ATTN_ROWS rows), each slice over the chunks its last row reaches.
            let slice = if self.in_wide.get() {
                WIDE_ATTN_ROWS
            } else {
                t
            };
            let qrow = (hp.n_head * hp.head_dim * 4) as u64;
            if let Some(ns) = self.fa2_splits(w, t) {
                let g = hp.n_head / hp.n_head_kv;
                let tiles = (t * g).div_ceil(ck::FA2_BM);
                unsafe {
                    self.launch_on(
                        self.stream_on(w),
                        self.kn_on(w).fa2,
                        (tiles as u32, hp.n_head_kv as u32, ns as u32),
                        128,
                        ck::fa2_smem(hp.head_dim) as u32,
                        tang_moe::args![q, kt, vt, rows, t as i32, part, ns as i32],
                    );
                    self.launch_on(
                        self.stream_on(w),
                        self.kn_on(w).merge,
                        (t as u32, hp.n_head as u32, 1),
                        hp.head_dim as u32,
                        0,
                        tang_moe::args![part, ns as i32, proj, st, attn],
                    );
                }
                self.gemv(
                    w,
                    "attn_out",
                    &layer.w_out,
                    attn,
                    None,
                    self.a_on(w, &s.mix),
                    t,
                    n,
                );
                return;
            }
            for r0 in (0..t).step_by(slice) {
                let c = slice.min(t - r0);
                let nc = if self.in_wide.get() {
                    self.nchunk(self.wide_pos0.get() + r0 + c)
                } else {
                    nchunk
                } as i32;
                let (q0, rows0, proj0, attn0) = (
                    q + r0 as u64 * qrow,
                    rows + (r0 * 4) as u64,
                    proj + (r0 * stride * 4) as u64,
                    attn + r0 as u64 * qrow,
                );
                unsafe {
                    self.launch_on(
                        self.stream_on(w),
                        self.kn_on(w).part,
                        (c as u32, hp.n_head_kv as u32, nc as u32),
                        128,
                        0,
                        tang_moe::args![q0, kt, vt, rows0, part, nc],
                    );
                    self.launch_on(
                        self.stream_on(w),
                        self.kn_on(w).merge,
                        (c as u32, hp.n_head as u32, 1),
                        hp.head_dim as u32,
                        0,
                        tang_moe::args![part, nc, proj0, st, attn0],
                    );
                }
            }
            self.gemv(
                w,
                "attn_out",
                &layer.w_out,
                attn,
                None,
                self.a_on(w, &s.mix),
                t,
                n,
            );
        }
    }

    /// The GDN conv and recurrence in one launch (`commit`: write the state for `win[1]`
    /// tokens), from the stacked projection and the conv history.
    fn gdn_step(&self, w: usize, g: &GdnL, stride: usize, t: usize, commit: bool) {
        let v = self.stage_view(w);
        let (st, p, sd, dt, sa, nw, y, tu, win, cm, eps, yq, quant, h0, wc) = (
            v.dev.buffer_addr(&g.state),
            v.dev.buffer_addr(&g.proj),
            stride as u32,
            v.dev.buffer_addr(&g.dt),
            v.dev.buffer_addr(&g.a),
            v.dev.buffer_addr(&g.norm),
            v.dev.buffer_addr(&v.s.y),
            t as u32,
            v.dev.buffer_addr(&v.s.ctl),
            commit as u32,
            self.eps,
            0u64,
            0u32,
            v.dev.buffer_addr(&g.hist),
            v.dev.buffer_addr(&g.conv),
        );
        unsafe {
            self.launch_on(
                v.stream,
                v.k.gdn_step,
                (self.geo.gdn_hv as u32, 1, 1),
                512,
                0,
                tang_moe::args![st, p, sd, dt, sa, nw, y, tu, win, cm, eps, yq, quant, h0, wc],
            )
        };
    }

    fn gdn_commit(&self, w: usize, g: &GdnL, stride: usize, t: usize) {
        let v = self.stage_view(w);
        let (h0, p, sd, win, tu) = (
            v.dev.buffer_addr(&g.hist),
            v.dev.buffer_addr(&g.proj),
            stride as u32,
            v.dev.buffer_addr(&v.s.ctl),
            t as u32,
        );
        unsafe {
            self.launch_on(
                v.stream,
                v.k.gdn_commit,
                (self.geo.gdn_conv().div_ceil(256) as u32, 1, 1),
                256,
                0,
                tang_moe::args![h0, p, sd, win, tu],
            )
        };
    }

    /// [`gdn_step`](Self::gdn_step) for every segment of the window (`s.rows`): each segment's
    /// rows run against its slot's state and history.
    fn gdn_step_seg(&self, w: usize, g: &GdnL, stride: usize, nseg: usize, commit: bool) {
        let v = self.stage_view(w);
        let (stab, htab, p, sd, dt, sa, nw, y, seg, cm, eps, wc) = (
            v.dev.buffer_addr(&g.stab),
            v.dev.buffer_addr(&g.htab),
            self.proj_on(w, &g.proj),
            stride as u32,
            v.dev.buffer_addr(&g.dt),
            v.dev.buffer_addr(&g.a),
            v.dev.buffer_addr(&g.norm),
            v.dev.buffer_addr(&v.s.y),
            v.dev.buffer_addr(&v.s.rows) + (2 * self.tw() * 4) as u64,
            commit as u32,
            self.eps,
            v.dev.buffer_addr(&g.conv),
        );
        unsafe {
            self.launch_on(
                v.stream,
                v.k.gdn_step_seg,
                (self.geo.gdn_hv as u32, nseg as u32, 1),
                512,
                0,
                tang_moe::args![stab, htab, p, sd, dt, sa, nw, y, seg, cm, eps, wc],
            )
        };
    }

    fn gdn_commit_seg(&self, w: usize, g: &GdnL, stride: usize, nseg: usize) {
        let v = self.stage_view(w);
        let (htab, p, sd, seg) = (
            v.dev.buffer_addr(&g.htab),
            self.proj_on(w, &g.proj),
            stride as u32,
            v.dev.buffer_addr(&v.s.rows) + (2 * self.tw() * 4) as u64,
        );
        unsafe {
            self.launch_on(
                v.stream,
                v.k.gdn_commit_seg,
                (self.geo.gdn_conv().div_ceil(256) as u32, nseg as u32, 1),
                256,
                0,
                tang_moe::args![htab, p, sd, seg],
            )
        };
    }

    /// Layer `l`'s MoE (input `s.x` / `s.mxq`): router, top-k, routed and shared experts in the
    /// same launches, then `r +=` the result and the next norm (`next`: the next layer's
    /// `attn_norm`, or `output_norm`) into `s.x` / `s.mxq`. `s.moe_y` gets the MoE output when
    /// `keep_y`.
    fn enqueue_moe(&self, wi: usize, l: usize, t: usize, next: &B, keep_y: bool) {
        let layer = self.lay(wi, l);
        let v = self.stage_view(wi);
        let (s, hp) = (&v.s, &self.hp);
        let (ne, kk) = (hp.n_expert, hp.n_expert_used);
        let (x, mxq) = (v.dev.buffer_addr(&s.x), v.dev.buffer_addr(&s.mxq));
        let ls = ne + 1;
        let (lg, lsi, ids, w, scale) = (
            v.dev.buffer_addr(&s.lg),
            ls as i32,
            v.dev.buffer_addr(&s.ids),
            v.dev.buffer_addr(&s.w),
            hp.expert_weights_scale,
        );
        let (r, yo, wn, xo, eps, ti) = (
            v.dev.buffer_addr(&s.r),
            if keep_y {
                v.dev.buffer_addr(&s.moe_y)
            } else {
                0u64
            },
            v.dev.buffer_addr(next),
            x,
            self.eps,
            t as i32,
        );
        let routed = t * kk;
        if self.var.shared_par {
            // The shared expert doesn't depend on the routing: fork it onto the side stream
            // before the router, join before the combine.
            self.fork(wi);
            self.experts(wi, layer, t, routed, t, v.side);
            self.gemv(wi, "router", &layer.router, x, Some(mxq), lg, t, ls);
            unsafe {
                self.launch_on(
                    v.stream,
                    v.k.route,
                    (t as u32, 1, 1),
                    32,
                    0,
                    tang_moe::args![lg, lsi, ids, w, scale],
                );
            }
            self.experts(wi, layer, t, 0, routed, v.stream);
            self.join(wi);
        } else {
            self.gemv(wi, "router", &layer.router, x, Some(mxq), lg, t, ls);
            unsafe {
                self.launch_on(
                    v.stream,
                    v.k.route,
                    (t as u32, 1, 1),
                    32,
                    0,
                    tang_moe::args![lg, lsi, ids, w, scale],
                );
            }
            self.experts(wi, layer, t, 0, routed + t, v.stream);
        }
        let parts = v.dev.buffer_addr(&s.parts);
        unsafe {
            self.launch_on(
                v.stream,
                v.k.moe_add_norm,
                (t as u32, 1, 1),
                1024,
                0,
                tang_moe::args![r, parts, w, lg, lsi, ti, yo, wn, xo, mxq, eps],
            );
        }
    }

    /// Expert pairs `pbase..pbase + n` (routed pairs first, then the shared expert per token):
    /// gate|up rows, SiLU+int8, down rows, on `stream`.
    fn experts(&self, wi: usize, layer: &Layer, t: usize, pbase: usize, n: usize, stream: &Stream) {
        if n == 0 {
            return;
        }
        let v = self.stage_view(wi);
        let (s, hp) = (&v.s, &self.hp);
        let (nh, ff) = (hp.n_embd, hp.n_ff_exp);
        let (gw, uw, dw) = (layer.gate.w, layer.up.w, layer.down.w);
        let (sg, su, sd) = (layer.sh_gate.w, layer.sh_up.w, layer.sh_down.w);
        let (ids, mxq, egu, hq, parts) = (
            v.dev.buffer_addr(&s.ids),
            v.dev.buffer_addr(&s.mxq),
            v.dev.buffer_addr(&s.egu),
            v.dev.buffer_addr(&s.hq),
            v.dev.buffer_addr(&s.parts),
        );
        let (ti, r16g, r16d, fastk, pb) = (
            t as i32,
            layer.r16_gu,
            layer.r16_down,
            self.var.fastk,
            pbase as i32,
        );
        let np = n as u32;
        let many = self.in_wide.get() || self.in_batch.get();
        // Narrow windows of 2..8 rows on the streaming grouped kernels (`TANG_CODER_NSTREAM=1`):
        // every group is at most 8 pairs there, as the kernels need. Bitwise.
        let nstream = !many && self.var.nstream && t >= 2;
        // The streaming kernels' shape: the batch module's (sweepable), else the source defaults.
        let (sst, snr_gu, snr_down, sst_down) = if self.in_batch.get() {
            let s = stream_shape();
            (s.0, s.1, s.2, stream_sst_down())
        } else {
            (1, 8, 16, 2)
        };
        if self.var.grouped || many || nstream {
            // One plan region per stream (the side stream's call is the one past the routed pairs).
            let pw = 2 + 2 * self.tw() * (hp.n_expert_used + 1);
            let plan = v.dev.buffer_addr(&s.plan) + if pbase > 0 { (pw * 4) as u64 } else { 0 };
            let ni = n as i32;
            // Batch modules split an expert's pairs into groups of `BATCH_GCHUNK`.
            let groups = if self.in_batch.get() {
                n.min(hp.n_expert + 1 + n / BATCH_GCHUNK)
            } else {
                n.min(hp.n_expert + 1)
            } as u32;
            unsafe {
                self.launch_on(
                    stream,
                    v.k.moe_plan,
                    (1, 1, 1),
                    if many { 1024 } else { 128 },
                    0,
                    tang_moe::args![ids, ti, pb, ni, plan],
                );
                // Wide windows: the tensor-core GEMM, 128 rows × 64 of a group's tokens a block.
                let gemm = (self.in_wide.get() || (self.in_batch.get() && self.var.bgemm))
                    && self.var.wgemm;
                let tiles = t.div_ceil(64) as u32;
                if gemm && self.var.wgemm_gu {
                    self.launch_on(
                        stream,
                        v.k.moe_gemm_gu,
                        ((2 * ff / 128) as u32, groups, tiles),
                        256,
                        0,
                        tang_moe::args![gw, uw, sg, su, ids, mxq, egu, ti, plan],
                    );
                    // Debug: the grouped decode kernel into `parts` (free until the down
                    // projection), then compare.
                    if env_is("TANG_CODER_WGEMM_CHECK", "1") {
                        let refb = parts;
                        self.launch_on(
                            stream,
                            v.k.moe_gu_g,
                            ((2 * ff / 16) as u32, groups, 1),
                            256,
                            (16 * 16 * r16g) as u32,
                            tang_moe::args![gw, uw, sg, su, ids, mxq, refb, ti, r16g, fastk, plan],
                        );
                        v.dev.sync();
                        let a = v.dev.download(&s.egu);
                        let b = v.dev.download(&s.parts);
                        let rows = 2 * ff;
                        let (mut worst, mut wi, mut sum, mut cnt) = (0f64, 0usize, 0f64, 0usize);
                        for i in 0..n * rows {
                            let (x, y) = (a[i] as f64, b[i] as f64);
                            let e = (x - y).abs() / y.abs().max(1e-3);
                            sum += (x - y).abs();
                            cnt += 1;
                            if e > worst {
                                worst = e;
                                wi = i;
                            }
                        }
                        eprintln!(
                            "WGEMM check gu: {cnt} values, mean |diff| {:.3e}, worst rel {worst:.3e} at pair {} row {} (gemm {} vs ref {})",
                            sum / cnt as f64,
                            wi / rows,
                            wi % rows,
                            a[wi],
                            b[wi]
                        );
                    }
                } else if self.in_batch.get() && self.var.mma_gu {
                    self.launch_on(
                        stream,
                        v.k.moe_gu_m,
                        ((2 * ff / 128) as u32, groups, 1),
                        256,
                        (BATCH_GCHUNK * (nh + nh / 8 + nh / 8)) as u32,
                        tang_moe::args![gw, uw, sg, su, ids, mxq, egu, ti, r16g, fastk, plan],
                    );
                } else if (self.in_batch.get() && self.var.stream) || nstream {
                    self.launch_on(
                        stream,
                        v.k.moe_gu_s,
                        ((2 * ff / (8 * snr_gu)) as u32, groups, 1),
                        256,
                        (8 * sst * 2 * 16 * r16g as usize + BATCH_GCHUNK * (nh + nh / 8)) as u32,
                        tang_moe::args![gw, uw, sg, su, ids, mxq, egu, ti, r16g, fastk, plan],
                    );
                } else {
                    self.launch_on(
                        stream,
                        v.k.moe_gu_g,
                        ((2 * ff / 16) as u32, groups, 1),
                        256,
                        (16 * 16 * r16g) as u32,
                        tang_moe::args![gw, uw, sg, su, ids, mxq, egu, ti, r16g, fastk, plan],
                    );
                }
                self.launch_on(
                    stream,
                    v.k.silu_q,
                    ((ff / 32) as u32, np, 1),
                    32,
                    0,
                    tang_moe::args![egu, hq, pb],
                );
                if gemm && self.var.wgemm_down {
                    self.launch_on(
                        stream,
                        v.k.moe_gemm_down,
                        ((nh / 128) as u32, groups, tiles),
                        256,
                        0,
                        tang_moe::args![dw, sd, ids, hq, parts, ti, plan],
                    );
                } else if self.in_batch.get() && self.var.mma_down {
                    self.launch_on(
                        stream,
                        v.k.moe_down_m,
                        ((nh / 128) as u32, groups, 1),
                        256,
                        (BATCH_GCHUNK * (ff + ff / 8 + ff / 8)) as u32,
                        tang_moe::args![dw, sd, ids, hq, parts, ti, r16d, fastk, plan],
                    );
                    if env_is("TANG_CODER_MMA_CHECK", "1") {
                        let tmp = v.dev.alloc_f32(n * nh);
                        let tp = v.dev.buffer_addr(&tmp);
                        self.launch_on(
                            stream,
                            v.k.moe_down_s,
                            ((nh / (8 * stream_shape().2)) as u32, groups, 1),
                            256,
                            (8 * stream_sst_down() * 2 * 16 * r16d as usize + BATCH_GCHUNK * (ff + ff / 8)) as u32,
                            tang_moe::args![dw, sd, ids, hq, tp, ti, r16d, fastk, plan],
                        );
                        v.dev.sync();
                        let a = v.dev.download(&s.parts);
                        let b = v.dev.download(&tmp);
                        let ids_h = v.dev.download(&s.ids);
                        let mut bad = 0usize;
                        let mut shown = 0;
                        let mut worst = 0f64;
                        for p in 0..n {
                            for r in 0..nh {
                                let (x, y) = (a[p * nh + r] as f64, b[p * nh + r] as f64);
                                let e = (x - y).abs() / y.abs().max(1e-2);
                                worst = worst.max(e);
                                if e > 1e-2 {
                                    bad += 1;
                                    if shown < 12 {
                                        shown += 1;
                                        let ex = if p < t * hp.n_expert_used { ids_h[p].to_bits() as i64 } else { -1 };
                                        eprintln!("  mma down pair {p} (expert {ex}) row {r}: mma {x} ref {y}");
                                    }
                                }
                            }
                        }
                        eprintln!("MMA check down: {bad} of {} differ >1%, worst rel {worst:.3e}", n * nh);
                    }
                } else if (self.in_batch.get() && self.var.stream) || nstream {
                    self.launch_on(
                        stream,
                        v.k.moe_down_s,
                        ((nh / (8 * snr_down)) as u32, groups, 1),
                        256,
                        (8 * sst_down * 2 * 16 * r16d as usize + BATCH_GCHUNK * (ff + ff / 8)) as u32,
                        tang_moe::args![dw, sd, ids, hq, parts, ti, r16d, fastk, plan],
                    );
                } else {
                    self.launch_on(
                        stream,
                        v.k.moe_down_g,
                        ((nh / 16) as u32, groups, 1),
                        256,
                        (16 * 16 * r16d) as u32,
                        tang_moe::args![dw, sd, ids, hq, parts, ti, r16d, fastk, plan],
                    );
                }
            }
            return;
        }
        // Staged (8 warps × 2 rows, rows in shared memory) or direct (8 warps × R rows).
        let moe = match self.var.moe {
            usize::MAX if t <= 4 => 2,
            usize::MAX => 4,
            v => v,
        };
        let (gu_f, down_f, rows, smem_g, smem_d) = match moe {
            2 => (v.k.moe_gu_d[0], v.k.moe_down_d[0], 16, 0, 0),
            4 => (v.k.moe_gu_d[1], v.k.moe_down_d[1], 32, 0, 0),
            // Double-buffered: 8 warps × 2 rows × 4 groups a block, two buffers.
            8 => (
                v.k.moe_gu_db,
                v.k.moe_down_db,
                64,
                (2 * 16 * 16 * r16g) as u32,
                (2 * 16 * 16 * r16d) as u32,
            ),
            _ => (
                v.k.moe_gu,
                v.k.moe_down,
                16,
                (16 * 16 * r16g) as u32,
                (16 * 16 * r16d) as u32,
            ),
        };
        unsafe {
            self.launch_on(
                stream,
                gu_f,
                ((2 * ff / rows) as u32, np, 1),
                256,
                smem_g,
                tang_moe::args![gw, uw, sg, su, ids, mxq, egu, ti, r16g, fastk, pb],
            );
            // The direct down kernels can make SiLU(gate) * up's int8 themselves.
            if self.var.fuse_silu && (moe == 2 || moe == 4) {
                let df = v.k.moe_down_df[usize::from(moe == 4)];
                self.launch_on(
                    stream,
                    df,
                    ((nh / rows) as u32, np, 1),
                    256,
                    0,
                    tang_moe::args![dw, sd, ids, egu, parts, ti, r16d, fastk, pb],
                );
                return;
            }
            self.launch_on(
                stream,
                v.k.silu_q,
                ((ff / 32) as u32, np, 1),
                32,
                0,
                tang_moe::args![egu, hq, pb],
            );
            self.launch_on(
                stream,
                down_f,
                ((nh / rows) as u32, np, 1),
                256,
                smem_d,
                tang_moe::args![dw, sd, ids, hq, parts, ti, r16d, fastk, pb],
            );
        }
    }

    /// Side stream: start after everything enqueued so far on stage `w`'s main stream.
    fn fork(&self, w: usize) {
        if w == 0 {
            self.ev_fork.record(&self.stream).expect("record");
            self.side.wait(&self.ev_fork).expect("wait");
        } else {
            let p = self.peer.as_ref().expect("no peer");
            p.ev_fork.record(&p.ctx.stream).expect("record");
            p.side.wait(&p.ev_fork).expect("wait");
        }
    }

    /// Main stream: wait for everything enqueued so far on the side stream of stage `w`.
    fn join(&self, w: usize) {
        if w == 0 {
            self.ev_join.record(&self.side).expect("record");
            self.stream.wait(&self.ev_join).expect("wait");
        } else {
            let p = self.peer.as_ref().expect("no peer");
            p.ev_join.record(&p.side).expect("record");
            p.ctx.stream.wait(&p.ev_join).expect("wait");
        }
    }

    /// The head runs on the last stage (where the final `x` is produced); its sampled ids go
    /// to that stage's view of the shared pinned staging.
    fn enqueue_head(&self, t: usize) {
        let last = self.n_stages() - 1;
        let v = self.stage_view(last);
        let head = if last == 0 {
            &self.head
        } else {
            &self.peer.as_ref().expect("no peer").head
        };
        let voc = self.hp.n_vocab;
        self.gemv(
            last,
            "head",
            head,
            v.dev.buffer_addr(&v.s.x),
            Some(v.dev.buffer_addr(&v.s.mxq)),
            v.dev.buffer_addr(&v.s.head),
            t,
            voc,
        );
        let (lg, ids, n, part, np, ctl) = (
            v.dev.buffer_addr(&v.s.head),
            v.dev.buffer_addr(&v.s.out_ids),
            voc as i32,
            v.dev.buffer_addr(&v.s.amax),
            64i32,
            v.dev.buffer_addr(&v.s.ctl),
        );
        if let Some(sp) = self.sampler.as_ref().filter(|_| !self.greedy.get()) {
            if !self.skip("argmax") {
                sp.enqueue(
                    v.dev,
                    v.stream,
                    &self.head_buf(last),
                    &v.s.ctl,
                    &v.s.out_ids,
                    t,
                );
            }
            self.kcopy_on(last, self.io_dev_on(last, Io::ids_at(self.hp.n_embd, self.tw())), ids, t * 4);
            return;
        }
        unsafe {
            self.launch_on(
                v.stream,
                v.k.argmax1,
                (64, t as u32, 1),
                1024,
                0,
                tang_moe::args![lg, n, part, ctl],
            );
            self.launch_on(
                v.stream,
                v.k.argmax2,
                (t as u32, 1, 1),
                32,
                0,
                tang_moe::args![part, np, ids],
            );
        }
        if let Some(rs) = self.row_sampler.as_ref().filter(|_| !self.skip("argmax")) {
            rs.enqueue(v.dev, v.stream, &self.head_buf(last), &v.s.out_ids, t);
        }
        self.kcopy_on(last, self.io_dev_on(last, Io::ids_at(self.hp.n_embd, self.tw())), ids, t * 4);
    }

    /// The head buffer of stage `w`'s device — the sampler needs it by reference.
    fn head_buf(&self, w: usize) -> &B {
        &self.stage_view(w).s.head
    }

    /// Everything a window enqueues; with `probe`, eager, syncing after each layer to read its
    /// last token's tensors.
    fn enqueue_window(
        &self,
        t: usize,
        verify: bool,
        nchunk: usize,
        nseg: usize,
        mut probe: Option<&mut Probe>,
    ) -> Result<()> {
        self.enqueue_inputs(t);
        let n = self.hp.n_embd;
        self.norm_q8(0, None, &self.layers[0].attn_norm, t);
        for w in 0..self.n_stages() {
            let v = self.stage_view(w);
            let last = |dev: &CudaComputeDevice, b: &B| -> Vec<f32> {
                dev.download(b)[(t - 1) * n..t * n].to_vec()
            };
            let nl = v.layers.len();
            for l in 0..nl {
                let layer = &v.layers[l];
                self.enqueue_mixer(w, l, t, verify, nchunk, nseg);
                self.norm_q8(w, Some(v.dev.buffer_addr(&v.s.mix)), &layer.post_norm, t);
                if let Some(p) = probe.as_deref_mut() {
                    v.dev.sync();
                    p.mixer_out.push(last(v.dev, &v.s.mix));
                    p.post_moe.push(last(v.dev, &v.s.r));
                }
                let next = if l + 1 < nl {
                    &v.layers[l + 1].attn_norm
                } else if w + 1 < self.n_stages() {
                    // The next stage's first `attn_norm`, read over the peer link by this
                    // stage's `moe_add_norm`.
                    &self.stage_view(w + 1).layers[0].attn_norm
                } else if self.peer.is_some() {
                    &self.peer.as_ref().unwrap().out_norm
                } else {
                    &self.out_norm
                };
                self.enqueue_moe(w, l, t, next, probe.is_some());
                if let Some(p) = probe.as_deref_mut() {
                    v.dev.sync();
                    p.moe_out.push(last(v.dev, &v.s.moe_y));
                    p.post_moe.push(last(v.dev, &v.s.r));
                    let k = self.hp.n_expert_used;
                    let ids = v.dev.download(&v.s.ids);
                    p.router_ids.push(
                        ids[(t - 1) * k..t * k]
                            .iter()
                            .map(|x| x.to_bits())
                            .collect(),
                    );
                    p.qsa_sel.push(None);
                }
            }
            if w + 1 < self.n_stages() {
                self.enqueue_boundary(w, t);
            }
        }
        self.enqueue_head(t);
        self.enqueue_mtp_in(t)?;
        Ok(())
    }

    /// The MTP layer on `t` staged rows: input from `m.emb` (staged) and `m.mh`, attention over
    /// `nchunk` chunks, then the head; ids and probabilities to the pinned `IDS` words. Its rows'
    /// normed outputs stay in `s.x` (a chained step's input, gathered by the host).
    fn enqueue_mtp(&self, t: usize, nchunk: usize) {
        let m = self.mtp.as_ref().expect("MTP layer");
        let (s, n) = (&self.s, self.hp.n_embd);
        self.kcopy_on(
            0,
            self.dev.buffer_addr(&s.rows),
            self.io_dev_on(0, Io::rows(n, MAX_T)),
            ROWS_U32 * 4,
        );
        self.kcopy_on(
            0,
            self.dev.buffer_addr(&m.emb),
            self.io_dev_on(0, Io::EMB),
            t * n * 4,
        );
        // The MTP layer stays on GPU 0: its buffers were allocated there at load, its input
        // `m.mh` was copied from the last stage's `x`, its output ids go to the shared staging.
        let (emb, mh, we, wh, eps, ni, cat) = (
            self.dev.buffer_addr(&m.emb),
            self.dev.buffer_addr(&m.mh),
            self.dev.buffer_addr(&m.enorm),
            self.dev.buffer_addr(&m.hnorm),
            self.eps,
            n as i32,
            self.dev.buffer_addr(&m.cat),
        );
        unsafe {
            self.launch_on(
                &self.stream,
                m.f_in,
                (t as u32, 2, 1),
                1024,
                0,
                tang_moe::args![emb, mh, we, wh, eps, ni, cat],
            )
        };
        self.gemv(
            0,
            "mtp_fc",
            &m.fc,
            cat,
            None,
            self.dev.buffer_addr(&s.r),
            t,
            n,
        );
        self.norm_q8(0, None, &m.layer.attn_norm, t);
        self.enqueue_mixer(0, MTP_L, t, true, nchunk, 1);
        self.norm_q8(0, Some(self.dev.buffer_addr(&s.mix)), &m.layer.post_norm, t);
        self.enqueue_moe(0, MTP_L, t, &m.onorm, false);
        let v = self.hp.n_vocab;
        self.gemv(
            0,
            "mtp_head",
            &self.head,
            self.dev.buffer_addr(&s.x),
            Some(self.dev.buffer_addr(&s.mxq)),
            self.dev.buffer_addr(&s.head),
            t,
            v,
        );
        let (lg, vi, ids, prob) = (
            self.dev.buffer_addr(&s.head),
            v as i32,
            self.dev.buffer_addr(&m.ids),
            self.dev.buffer_addr(&m.prob),
        );
        unsafe {
            self.launch_on(
                &self.stream,
                m.f_pick,
                (t as u32, 1, 1),
                1024,
                0,
                tang_moe::args![lg, vi, ids, prob],
            )
        };
        self.kcopy_on(0, self.io_dev_on(0, Io::IDS), ids, 2 * MAX_T * 4);
        self.kcopy_on(0, self.io_dev_on(0, Io::MTP_PROB), prob, 2 * MAX_T * 4);
    }

    fn enqueue_commit(&self, t: usize) {
        for w in 0..self.n_stages() {
            let v = self.stage_view(w);
            self.kcopy_on(
                w,
                v.dev.buffer_addr(&v.s.ctl),
                self.io_dev_on(w, Io::CCTL),
                32,
            );
            for layer in v.layers {
                if let Some(g) = &layer.gdn {
                    self.gdn_step(w, g, layer.w_in.rows, t, true);
                    self.gdn_commit(w, g, layer.w_in.rows, t);
                }
            }
        }
    }

    // ---- host side

    fn stage(&mut self, pos0: usize, t: usize) -> Result<()> {
        let ctl = self.io.u32s(Io::CTL, 5);
        ctl[0] = pos0 as u32;
        ctl[1] = t as u32;
        ctl[2] = self.counter;
        ctl[3] = self.seed;
        ctl[4] = self.temperature.to_bits();
        if let Some(sp) = self.sampler.as_mut() {
            sp.stage(&self.dev, &self.tokens, pos0, t)?;
        }
        let n = self.hp.n_embd;
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.io.f32s(Io::EMB, t * n);
        for i in 0..t {
            let row = self.g.rows(&emb_t, self.tokens[pos0 + i] as usize, 1)?;
            emb[i * n..(i + 1) * n].copy_from_slice(&row);
        }
        // One sequence (slot 0): rows at pos0.., one segment keeping all of them.
        let rows = self.io.u32s(Io::rows(n, MAX_T), ROWS_U32);
        rows.fill(0);
        for i in 0..t {
            rows[i] = (pos0 + i) as u32;
        }
        rows[2 * MAX_T..2 * MAX_T + 4].copy_from_slice(&[0, t as u32, 0, t as u32]);
        self.mirror_staging(MAX_T, t);
        Ok(())
    }

    /// The MTP layer on rows `(slot, pos, next)`: each takes the final hidden state `m.mh` holds
    /// (in row order) for position `pos` of `slot` and the token `next` at `pos + 1`, and
    /// guesses the token at `pos + 2`. Returns each row's guess and its probability.
    fn mtp_pass(&mut self, rows: &[(usize, usize, u32)]) -> Result<Vec<(u32, f32)>> {
        let (t, n) = (rows.len(), self.hp.n_embd);
        let end = rows.iter().map(|r| r.1 + 2).max().unwrap_or(0);
        ensure!((1..=MAX_T).contains(&t) && end <= self.max_ctx);
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.io.f32s(Io::EMB, t * n);
        for (i, r) in rows.iter().enumerate() {
            let row = self.g.rows(&emb_t, r.2 as usize, 1)?;
            emb[i * n..(i + 1) * n].copy_from_slice(&row);
        }
        let rec = self.io.u32s(Io::rows(n, MAX_T), ROWS_U32);
        rec.fill(0);
        for (i, r) in rows.iter().enumerate() {
            rec[i] = (r.1 + 1) as u32;
            rec[MAX_T + i] = r.0 as u32;
        }
        let nchunk = self.nchunk(end);
        if self.use_graphs && self.prof.is_none() {
            let key = (t, nchunk);
            if let Some(g) = self.mtp_graphs.get(&key) {
                g.launch(&self.stream).map_err(|e| anyhow!("{e}"))?;
            } else {
                self.enqueue_mtp(t, nchunk);
                self.dev.sync();
                let out = self.mtp_read(t);
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let me = self as *const Self;
                let gr = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }.enqueue_mtp(t, nchunk);
                    Ok(())
                })
                .map_err(|e| anyhow!("capture MTP T={t}: {e}"))?;
                self.mtp_graphs.insert(key, gr);
                return Ok(out);
            }
        } else {
            self.enqueue_mtp(t, nchunk);
        }
        self.dev.sync();
        Ok(self.mtp_read(t))
    }

    /// Copy rows `from[i]` of `src` (`n_embd` f32 each) to rows `i` of `m.mh`, through `m.cat`.
    fn mtp_gather(&self, src: u64, from: &[usize]) {
        let m = self.mtp.as_ref().expect("MTP layer");
        let (mh, cat, b) = (self.a(&m.mh), self.a(&m.cat), self.hp.n_embd * 4);
        if from.iter().enumerate().all(|(i, &f)| i == f) && src == mh {
            return;
        }
        for (i, &f) in from.iter().enumerate() {
            self.kcopy_on(0, cat + (i * b) as u64, src + (f * b) as u64, b);
        }
        self.kcopy_on(0, mh, cat, from.len() * b);
    }

    /// The MTP layer after a window, for groups `(slot, pos, r0, next)`: positions
    /// `pos..pos + next.len()` of `slot` were window rows `r0..` (their final hidden rows in
    /// `m.mh`), and `next[i]` followed position `pos + i`. Fills the MTP K/V for those
    /// positions; with `chain`, drafts `mtp_steps` tokens past each group's end and returns
    /// them (with the first draft's runner-up).
    #[allow(clippy::type_complexity)]
    fn mtp_groups(
        &mut self,
        groups: &[(usize, usize, usize, &[u32])],
        chain: bool,
    ) -> Result<Vec<(Vec<(u32, f32)>, Option<(u32, f32)>)>> {
        let mut out = vec![(Vec::new(), None); groups.len()];
        let total: usize = groups.iter().map(|g| g.3.len()).sum();
        let end = groups.iter().map(|g| g.1 + g.3.len()).max().unwrap_or(0);
        if total == 0 || total > MAX_T || end + self.mtp_steps + 1 >= self.max_ctx {
            return Ok(out);
        }
        let from: Vec<usize> = groups.iter().flat_map(|g| g.2..g.2 + g.3.len()).collect();
        let mh = self.a(&self.mtp.as_ref().unwrap().mh);
        self.mtp_gather(mh, &from);
        let rows: Vec<(usize, usize, u32)> = groups
            .iter()
            .flat_map(|g| g.3.iter().enumerate().map(move |(i, &k)| (g.0, g.1 + i, k)))
            .collect();
        let first = self.mtp_pass(&rows)?;
        if !chain {
            return Ok(out);
        }
        let mut last = Vec::with_capacity(groups.len());
        let mut r = 0;
        for (gi, g) in groups.iter().enumerate() {
            r += g.3.len();
            last.push(r - 1);
            out[gi].0.push(first[r - 1]);
            out[gi].1 = Some(self.mtp_read_alt(r - 1));
        }
        for j in 1..self.mtp_steps {
            // With a split, the chained step's source hidden rows are GPU 1's `s.x`; bring
            // them over before the gather (a few rows of `n` f32).
            if self.peer.is_some() {
                let lastv = self.stage_view(self.n_stages() - 1);
                let n = self.hp.n_embd;
                let done = gpu::Event::new(false).expect("event");
                done.record(lastv.stream).expect("record");
                self.stream.wait(&done).expect("wait");
                let r = unsafe {
                    cudarc::driver::sys::cuMemcpyPeerAsync(
                        self.dev.buffer_addr(&self.s.x),
                        self.dev.cuda_context().cu_ctx(),
                        lastv.dev.buffer_addr(&lastv.s.x),
                        lastv.dev.cuda_context().cu_ctx(),
                        MAX_T * n * 4,
                        self.stream.0,
                    )
                };
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("mtp chain boundary copy: {r:?}"));
                }
                self.dev.sync();
            }
            self.mtp_gather(self.a(&self.s.x), &last);
            let rows: Vec<(usize, usize, u32)> = groups
                .iter()
                .zip(&out)
                .map(|(g, o)| (g.0, g.1 + g.3.len() + j - 1, o.0[j - 1].0))
                .collect();
            let d = self.mtp_pass(&rows)?;
            for (o, &x) in out.iter_mut().zip(&d) {
                o.0.push(x);
            }
            last = (0..groups.len()).collect();
        }
        Ok(out)
    }

    fn mtp_read(&self, t: usize) -> Vec<(u32, f32)> {
        let ids = self.io.u32s(Io::IDS, t);
        let p = self.io.f32s(Io::MTP_PROB, t);
        ids.iter().zip(p.iter()).map(|(&d, &p)| (d, p)).collect()
    }

    /// The last MTP pass's runner-up for row `r`.
    fn mtp_read_alt(&self, r: usize) -> (u32, f32) {
        (
            self.io.u32s(Io::IDS, 2 * MAX_T)[MAX_T + r],
            self.io.f32s(Io::MTP_PROB, 2 * MAX_T)[MAX_T + r],
        )
    }

    /// After a window over positions `pos..pos + next.len()` (whose final hidden rows `m.mh`
    /// holds), feed the MTP layer the tokens that followed them (`next`), so its K/V cache
    /// covers those positions; with `chain`, then draft `mtp_steps` tokens past the last one
    /// into `mtp_last` (and the first draft's runner-up into `mtp_alt`).
    fn mtp_after(&mut self, pos: usize, next: &[u32], chain: bool) -> Result<()> {
        self.mtp_after_rows(pos, 0, next, chain)
    }

    /// [`mtp_after`](Self::mtp_after) for a window whose kept rows start at row `r0` (a tree
    /// window keeps one branch's rows): positions `pos..pos + next.len()` are window rows
    /// `r0..r0 + next.len()`. Drafts a chain into `mtp_last` / `mtp_alt`.
    pub fn mtp_after_window(&mut self, pos: usize, r0: usize, next: &[u32]) -> Result<()> {
        self.mtp_after_rows(pos, r0, next, true)
    }

    fn mtp_after_rows(&mut self, pos: usize, r0: usize, next: &[u32], chain: bool) -> Result<()> {
        self.mtp_last.clear();
        self.mtp_alt = None;
        if !self.mtp_on() || next.is_empty() {
            return Ok(());
        }
        ensure!(r0 + next.len() <= MAX_T, "rows {r0}+{}", next.len());
        let t0 = Instant::now();
        let mut g = self.mtp_groups(&[(0, pos, r0, next)], chain)?;
        let (c, alt) = g.pop().unwrap();
        self.mtp_last = c;
        self.mtp_alt = alt;
        self.last_mtp_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// MTP drafting applies: a layer is loaded, `use_mtp`, and the windows are greedy.
    fn mtp_on(&self) -> bool {
        self.mtp.is_some() && self.use_mtp && self.greedy.get()
    }

    /// After a [`window`](Self::window) over `pos0..pos0 + next.len()`: feed the MTP layer
    /// the tokens that followed each row, drafting a chain when it reaches the sequence's end.
    pub fn mtp_feed(&mut self, pos0: usize, next: &[u32]) -> Result<()> {
        let end = pos0 + next.len() >= self.tokens.len();
        self.mtp_after(pos0, next, end)
    }

    pub fn window(&mut self, pos0: usize, t: usize, probe: Option<&mut Probe>) -> Result<Vec<u32>> {
        self.window_mode(pos0, t, false, probe)
    }

    /// [`window`](Self::window) as a verify window (`verify`): the GDN state is only read; call
    /// [`commit`](Self::commit) with the number of tokens kept before the next window.
    pub fn window_mode(
        &mut self,
        pos0: usize,
        t: usize,
        verify: bool,
        probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        ensure!((1..=MAX_T).contains(&t) && pos0 + t <= self.tokens.len());
        ensure!(
            pos0 + t <= self.max_ctx,
            "context {} > max {}",
            pos0 + t,
            self.max_ctx
        );
        let w0 = Instant::now();
        let mut st = WinStats {
            t,
            ..Default::default()
        };
        self.stage(pos0, t)?;
        st.host_prep_ms = w0.elapsed().as_secs_f64() * 1e3;
        let nchunk = self.nchunk(pos0 + t);
        self.run_staged(st, w0, t, verify, nchunk, 1, probe)
    }

    /// Run the window `stage` / `stage_rows` staged: `t` rows in `nseg` segments, attention over
    /// `nchunk` chunks. A graph per (t, verify, nchunk, nseg), captured on first use.
    #[allow(clippy::too_many_arguments)]
    fn run_staged(
        &mut self,
        st: WinStats,
        w0: Instant,
        t: usize,
        verify: bool,
        nchunk: usize,
        nseg: usize,
        mut probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        // The batch module's kernels assume more than MAX_T rows (see decode_batch): refuse a
        // short window there instead of faulting the device.
        ensure!(
            !self.in_batch.get() || t > MAX_T,
            "a {t}-row window in the batch module (it takes {}..={} rows)",
            MAX_T + 1,
            self.batch_rows()
        );
        self.last_batch = self.in_batch.get();
        if let Some(rs) = self.row_sampler.as_ref() {
            // Every window stages its rows' words (none sampled unless a slot path set them).
            let mut w = std::mem::take(&mut self.row_words);
            w.resize(t * super::sampler::ROW_WORDS, 0);
            rs.stage(&self.dev, &w)?;
        }
        self.row_words.clear();
        self.greedy.set(
            self.temperature == 0.0 && self.sampler.as_ref().is_none_or(|s| s.presence == 0.0),
        );
        let key = (t, verify, nchunk, nseg, self.greedy.get());
        let split = self.peer.is_some();
        if probe.is_none() && self.use_graphs {
            if !split && !self.graphs.contains_key(&key) {
                // First use: run eagerly (loads every kernel), then capture.
                self.enqueue_window(t, verify, nchunk, nseg, None)?;
                self.dev.sync();
                let ids = self.iox().u32s(Io::ids_at(self.hp.n_embd, self.tw()), t).to_vec();
                // Undo nothing: the eager run *was* this window. Capture for next time without
                // running it (capture records, it doesn't execute).
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let me = self as *const Self;
                let gr = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }
                        .enqueue_window(t, verify, nchunk, nseg, None)
                        .map_err(|e| tang_moe::gpu::Error(format!("{e}")))
                })
                .map_err(|e| anyhow!("capture T={t}: {e}"))?;
                self.graphs.insert(key, gr);
                return Ok(self.finish(st, w0, ids));
            }
            if split {
                // Two devices: a graph per stage, the boundary copies between the launches.
                // First use on a stage: run the whole window eagerly (which loads every
                // kernel on both), then capture each stage once.
                if !self.peer.as_ref().unwrap().graphs.contains_key(&key) {
                    self.enqueue_window(t, verify, nchunk, nseg, None)?;
                    self.sync_all();
                    self.collect_out(t, false);
                    let ids = self.iox().u32s(Io::ids_at(self.hp.n_embd, self.tw()), t).to_vec();
                    self.capture_stages(t, verify, nchunk, nseg)?;
                    return Ok(self.finish(st, w0, ids));
                }
                let p = self.peer.as_ref().unwrap();
                self.graphs[&key]
                    .launch(&self.stream)
                    .map_err(|e| anyhow!("{e}"))?;
                self.enqueue_boundary(0, t);
                p.graphs[&key]
                    .launch(&p.ctx.stream)
                    .map_err(|e| anyhow!("{e}"))?;
                self.enqueue_mtp_in(t)?;
            } else {
                self.graphs[&key]
                    .launch(&self.stream)
                    .map_err(|e| anyhow!("{e}"))?;
            }
        } else {
            self.enqueue_window(t, verify, nchunk, nseg, probe.as_deref_mut())?;
        }
        self.sync_all();
        self.collect_out(t, false);
        self.prof_collect();
        let ids = self.iox().u32s(Io::ids_at(self.hp.n_embd, self.tw()), t).to_vec();
        if let Some(p) = probe {
            let n = self.hp.n_embd;
            let last = self.stage_view(self.n_stages() - 1);
            p.final_x = last.dev.download(&last.s.x)[(t - 1) * n..t * n].to_vec();
        }
        Ok(self.finish(st, w0, ids))
    }

    /// Capture each stage's half of the window by (temporarily) masking the other stage out:
    /// `enqueue_window` with `only_stage` set enqueues just that stage's layers (the
    /// boundary event/copy between the stages stays outside the graphs).
    fn capture_stages(&mut self, t: usize, verify: bool, nchunk: usize, nseg: usize) -> Result<()> {
        let key = (t, verify, nchunk, nseg, self.greedy.get());
        let me = self as *const Self;
        {
            let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
            let g0 = Graph::capture(&stream, |_| {
                // SAFETY: `self` is not otherwise touched during capture.
                unsafe { &*me }
                    .enqueue_stage(0, t, verify, nchunk, nseg)
                    .map_err(|e| tang_moe::gpu::Error(format!("{e}")))
            })
            .map_err(|e| anyhow!("capture T={t} stage 0: {e}"))?;
            self.graphs.insert(key.clone(), g0);
        }
        {
            let p = self.peer.as_ref().expect("no peer");
            p.ctx.gpu.bind().map_err(|e| anyhow!("{e}"))?;
            let me = self as *const Self;
            let g1 = Graph::capture(&p.ctx.stream, |_| {
                // SAFETY: `self` is not otherwise touched during capture.
                unsafe { &*me }
                    .enqueue_stage(1, t, verify, nchunk, nseg)
                    .map_err(|e| tang_moe::gpu::Error(format!("{e}")))
            })
            .map_err(|e| anyhow!("capture T={t} stage 1: {e}"))?;
            self.peer.as_mut().unwrap().graphs.insert(key, g1);
        }
        self.gpu.bind().map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// Stage `w`'s half of a window (graph capture): its layers, and — for the last stage —
    /// the head and the MTP input copy. `enqueue_window` is the both-stages version.
    fn enqueue_stage(
        &self,
        w: usize,
        t: usize,
        verify: bool,
        nchunk: usize,
        nseg: usize,
    ) -> Result<()> {
        if w == 0 {
            self.enqueue_inputs(t);
            self.norm_q8(0, None, &self.layers[0].attn_norm, t);
        }
        let last_stage = self.n_stages() - 1;
        let v = self.stage_view(w);
        for l in 0..v.layers.len() {
            let layer = &v.layers[l];
            self.enqueue_mixer(w, l, t, verify, nchunk, nseg);
            self.norm_q8(w, Some(v.dev.buffer_addr(&v.s.mix)), &layer.post_norm, t);
            let next = if l + 1 < v.layers.len() {
                &v.layers[l + 1].attn_norm
            } else if w + 1 < self.n_stages() {
                &self.stage_view(w + 1).layers[0].attn_norm
            } else {
                &self.peer.as_ref().expect("split").out_norm
            };
            self.enqueue_moe(w, l, t, next, false);
        }
        if w == last_stage {
            self.enqueue_head(t);
            // The MTP input copy is *not* here: a capture on one device cannot order the
            // other device's work, and with a split `m.mh` (GPU 0) reads GPU 1's `x`. The
            // eager `enqueue_window` enqueues it with its own stream ordering; the graph
            // path (`run_staged`) does it between the two launches.
        }
        Ok(())
    }

    /// The MTP input copy for a window whose head ran on the last stage: `m.mh ← x_last`.
    /// Eager, ordered after GPU 1's work (`enqueue_window` calls it; the graph path calls
    /// it between the stage launches). With no split it's a plain kcopy on GPU 0.
    fn enqueue_mtp_in(&self, t: usize) -> Result<()> {
        let Some(m) = &self.mtp else {
            return Ok(());
        };
        if t > MAX_T {
            // Batch windows don't draft (`m.mh` holds `MAX_T` rows).
            return Ok(());
        }
        let last = self.n_stages() - 1;
        let v = self.stage_view(last);
        let n = self.hp.n_embd;
        if last == 0 {
            self.kcopy_on(
                0,
                self.dev.buffer_addr(&m.mh),
                self.dev.buffer_addr(&v.s.x),
                t * n * 4,
            );
            return Ok(());
        }
        let done = gpu::Event::new(false).expect("event");
        done.record(v.stream).expect("record");
        self.stream.wait(&done).expect("wait");
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                self.dev.buffer_addr(&m.mh),
                self.dev.cuda_context().cu_ctx(),
                v.dev.buffer_addr(&v.s.x),
                v.dev.cuda_context().cu_ctx(),
                t * n * 4,
                self.stream.0,
            )
        };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("mtp boundary copy: {r:?}"));
        }
        Ok(())
    }

    fn finish(&mut self, mut st: WinStats, w0: Instant, ids: Vec<u32>) -> Vec<u32> {
        st.routed = st.t * self.hp.n_expert_used * self.hp.n_layer;
        st.wall_ms = w0.elapsed().as_secs_f64() * 1e3;
        self.counter = self.counter.wrapping_add(1);
        self.last = st;
        ids
    }

    /// After a verify window of `t`: keep the first `n_keep` tokens' recurrent state.
    pub fn commit(&mut self, t: usize, n_keep: usize) -> Result<()> {
        ensure!((1..=t).contains(&n_keep));
        let c = self.io.u32s(Io::CCTL, 2);
        c[0] = 0;
        c[1] = n_keep as u32;
        if let Some(p) = self.peer.as_deref_mut() {
            // The commit replays the same window segment on both devices.
            let pc = p.io.u32s(Io::CCTL, 2);
            let (a, b) = (c[0], c[1]);
            pc[0] = a;
            pc[1] = b;
        }
        if !self.use_graphs {
            self.enqueue_commit(t);
            return Ok(());
        }
        if self.commit_graphs[t].is_none() {
            self.enqueue_commit(t);
            self.sync_all();
            if self.peer.is_none() {
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let me = self as *const Self;
                let gr = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }.enqueue_commit(t);
                    Ok(())
                })
                .map_err(|e| anyhow!("capture commit T={t}: {e}"))?;
                self.commit_graphs[t] = Some(gr);
                return Ok(());
            }
            // A split: capture each stage's commit on its own device.
            let me = self as *const Self;
            {
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let g0 = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }.enqueue_commit_stage(0, t);
                    Ok(())
                })
                .map_err(|e| anyhow!("capture commit T={t} stage 0: {e}"))?;
                self.commit_graphs[t] = Some(g0);
            }
            let peer = self.peer.as_mut().expect("no peer");
            peer.ctx.gpu.bind().map_err(|e| anyhow!("{e}"))?;
            let g1 = Graph::capture(&peer.ctx.stream, |_| {
                // SAFETY: `self` is not otherwise touched during capture.
                unsafe { &*me }.enqueue_commit_stage(1, t);
                Ok(())
            })
            .map_err(|e| anyhow!("capture commit T={t} stage 1: {e}"))?;
            self.peer.as_mut().unwrap().commit_graphs[t] = Some(g1);
            self.gpu.bind().map_err(|e| anyhow!("{e}"))?;
            return Ok(());
        }
        self.commit_graphs[t]
            .as_ref()
            .unwrap()
            .launch(&self.stream)
            .map_err(|e| anyhow!("{e}"))?;
        if let Some(peer) = self.peer.as_ref() {
            peer.commit_graphs[t]
                .as_ref()
                .expect("peer commit graph")
                .launch(&peer.ctx.stream)
                .map_err(|e| anyhow!("{e}"))?;
        }
        Ok(())
    }

    /// Stage `w`'s half of a commit (graph capture): CCTL in, then its GDN layers' replay.
    fn enqueue_commit_stage(&self, w: usize, t: usize) {
        let v = self.stage_view(w);
        self.kcopy_on(
            w,
            v.dev.buffer_addr(&v.s.ctl),
            self.io_dev_on(w, Io::CCTL),
            32,
        );
        for layer in v.layers {
            if let Some(g) = &layer.gdn {
                self.gdn_step(w, g, layer.w_in.rows, t, true);
                self.gdn_commit(w, g, layer.w_in.rows, t);
            }
        }
    }

    /// Download the last window's logits (`[t][vocab]`).
    pub fn logits(&self, t: usize) -> Vec<f32> {
        let v = self.hp.n_vocab;
        if self.last_batch {
            if let Some(b) = &self.batch {
                return self.dev.download(&b.s.head)[..t * v].to_vec();
            }
        }
        let last = self.stage_view(self.n_stages() - 1);
        last.dev.download(&last.s.head)[..t * v].to_vec()
    }

    /// Feed `ids` as a prompt (windows of up to `MAX_T`), then return the next token.
    pub fn prefill(
        &mut self,
        ids: &[u32],
        chunk: usize,
        mut logits_cb: Option<LogitsCb<'_>>,
    ) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prompt");
        let mut pos = self.tokens.len();
        self.tokens.extend_from_slice(ids);
        let mut next = 0;
        while pos < self.tokens.len() {
            let t = chunk.min(self.tokens.len() - pos).clamp(1, MAX_T);
            let out = self.window(pos, t, None)?;
            if let Some(cb) = logits_cb.as_deref_mut() {
                cb(pos, t, &self.logits(t));
            }
            next = out[t - 1];
            if self.mtp.is_some() {
                let mut nx = self.tokens[pos + 1..pos + t].to_vec();
                let last = pos + t == self.tokens.len();
                nx.push(if last { next } else { self.tokens[pos + t] });
                self.mtp_after(pos, &nx, last)?;
            }
            pos += t;
        }
        // Profile decode only.
        if let Some(p) = &self.prof {
            p.borrow_mut().total.clear();
        }
        Ok(next)
    }

    /// Run `cur` and `drafts` as one verify window; returns the kept tokens (the accepted
    /// drafts' successors plus the first correction), and commits them.
    pub fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        let pos = self.tokens.len();
        let t = 1 + drafts.len();
        ensure!(t <= MAX_T, "{} drafts", drafts.len());
        self.tokens.push(cur);
        self.tokens.extend_from_slice(drafts);
        if t == 1 {
            let next = self.window(pos, 1, None)?[0];
            self.mtp_after(pos, &[next], true)?;
            return Ok(vec![next]);
        }
        let targets = self.window_mode(pos, t, true, None)?;
        let mut n = 1;
        while n < t && drafts[n - 1] == targets[n - 1] {
            n += 1;
        }
        let tc = Instant::now();
        self.commit(t, n)?;
        self.tokens.truncate(pos + n);
        self.last_commit_ms = tc.elapsed().as_secs_f64() * 1e3;
        self.mtp_after(pos, &targets[..n], true)?;
        Ok(targets[..n].to_vec())
    }

    /// Append `tok` and run it as a T=1 window; returns the next token.
    pub fn step(&mut self, tok: u32) -> Result<u32> {
        self.tokens.push(tok);
        let pos = self.tokens.len() - 1;
        let next = self.window(pos, 1, None)?[0];
        self.mtp_after(pos, &[next], true)?;
        Ok(next)
    }

    /// An MTP layer is loaded (`CoderOpts::mtp`).
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn gguf(&self) -> &Gguf {
        &self.g
    }

    pub fn vocab(&self) -> Result<Vec<String>> {
        let v = self.g.meta("tokenizer.ggml.tokens")?;
        let a = v.as_array().context("tokens")?;
        Ok(a.iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect())
    }

    /// Time each stage of a T=1 window alone: a graph of that stage's launches for every layer
    /// (the window with every other launch filtered out), replayed back to back. Returns
    /// (stage, µs a step, bytes a step), the whole window first. Run after a decode (the
    /// routing buffers then hold real expert ids).
    pub fn stage_sweep(&mut self) -> Result<Vec<(&'static str, f64, f64)>> {
        ensure!(
            self.peer.is_none(),
            "stage_sweep is single-GPU (TANG_GPUS=1)"
        );
        ensure!(!self.tokens.is_empty(), "decode something first");
        let pos0 = self.tokens.len() - 1;
        self.stage(pos0, 1)?;
        let nchunk = self.nchunk(pos0 + 1);
        self.enqueue_window(1, false, nchunk, 1, None)?;
        self.dev.sync();
        let hp = &self.hp;
        let lb = |f: &dyn Fn(&Layer) -> f64| -> f64 { self.layers.iter().map(f).sum() };
        let ex = |e: &Experts, n: usize| e.w.estride as f64 * n as f64;
        let k = hp.n_expert_used;
        let stages: Vec<(&'static str, f64)> = vec![
            ("", self.step_bytes(pos0 + 1)),
            (
                "gdn_in",
                lb(&|l| {
                    if l.gdn.is_some() {
                        dense_bytes(&l.w_in)
                    } else {
                        0.0
                    }
                }),
            ),
            ("gdn_conv_step", 0.0),
            ("gdn_commit", 0.0),
            (
                "gdn_out",
                lb(&|l| {
                    if l.gdn.is_some() {
                        dense_bytes(&l.w_out)
                    } else {
                        0.0
                    }
                }),
            ),
            (
                "attn_in",
                lb(&|l| {
                    if l.attn.is_some() {
                        dense_bytes(&l.w_in)
                    } else {
                        0.0
                    }
                }),
            ),
            ("attn_prep", 0.0),
            (
                "attn_part",
                lb(&|l| {
                    if l.attn.is_some() {
                        (pos0 * hp.n_head_kv * hp.head_dim * 4) as f64
                    } else {
                        0.0
                    }
                }),
            ),
            ("attn_merge", 0.0),
            (
                "attn_out",
                lb(&|l| {
                    if l.attn.is_some() {
                        dense_bytes(&l.w_out)
                    } else {
                        0.0
                    }
                }),
            ),
            ("norm_q8", 0.0),
            ("router", lb(&|l| dense_bytes(&l.router))),
            ("route", 0.0),
            (
                "moe_gu",
                lb(&|l| ex(&l.gate, k) + ex(&l.up, k) + ex(&l.sh_gate, 1) + ex(&l.sh_up, 1)),
            ),
            ("silu_q", 0.0),
            ("moe_down", lb(&|l| ex(&l.down, k) + ex(&l.sh_down, 1))),
            ("moe_add_norm", 0.0),
            ("head", dense_bytes(&self.head)),
            ("argmax", 0.0),
        ];
        const REPS: usize = 4;
        const RUNS: usize = 10;
        let mut out = Vec::new();
        for (name, bytes) in stages {
            self.only
                .set(if name.is_empty() { None } else { Some(name) });
            let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
            let me = self as *const Self;
            let gr = Graph::capture(&stream, |_| {
                for _ in 0..REPS {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }
                        .enqueue_window(1, false, nchunk, 1, None)
                        .map_err(|e| tang_moe::gpu::Error(format!("{e}")))?;
                }
                Ok(())
            });
            self.only.set(None);
            let gr = gr.map_err(|e| anyhow!("capture {name}: {e}"))?;
            gr.launch(&self.stream).map_err(|e| anyhow!("{e}"))?;
            let (e0, e1) = (
                gpu::Event::new(true).map_err(|e| anyhow!("{e}"))?,
                gpu::Event::new(true).map_err(|e| anyhow!("{e}"))?,
            );
            e0.record(&self.stream).map_err(|e| anyhow!("{e}"))?;
            for _ in 0..RUNS {
                gr.launch(&self.stream).map_err(|e| anyhow!("{e}"))?;
            }
            e1.record(&self.stream).map_err(|e| anyhow!("{e}"))?;
            e1.sync().map_err(|e| anyhow!("{e}"))?;
            let us = e1.since(&e0).map_err(|e| anyhow!("{e}"))? as f64 * 1e3 / (REPS * RUNS) as f64;
            out.push((if name.is_empty() { "window" } else { name }, us, bytes));
        }
        Ok(out)
    }

    /// Switch kernel variants (`kq=N`, `fastk=N`, `shpar=1`, comma-separated; empty for the
    /// defaults) and drop the captured graphs.
    pub fn set_variants(&mut self, spec: &str) -> Result<()> {
        let mut v = Variants::default();
        for kv in spec.split(',').filter(|x| !x.is_empty()) {
            match kv.split_once('=') {
                Some(("kq", n)) => v.kq = n.parse()?,
                Some(("fastk", n)) => v.fastk = n.parse()?,
                Some(("shpar", n)) => v.shared_par = n == "1",
                Some(("nstream", n)) => v.nstream = n == "1",
                _ => anyhow::bail!("unknown variant {kv}"),
            }
        }
        self.var = v;
        self.dev.sync();
        self.graphs.clear();
        Ok(())
    }

    /// Bytes one decode step reads from VRAM at context `ctx`: every dense weight, the
    /// routed experts a token uses (top-k per layer, at the layer's average expert size), and
    /// the attention cache. The speed-of-light denominator.
    pub fn step_bytes(&self, ctx: usize) -> f64 {
        let hp = &self.hp;
        let mut b = 0f64;
        for l in &self.layers {
            for d in [&l.w_in, &l.w_out, &l.router] {
                b += dense_bytes(d);
            }
            for e in [&l.sh_gate, &l.sh_up, &l.sh_down] {
                b += e.w.estride as f64;
            }
            for e in [&l.gate, &l.up, &l.down] {
                b += e.w.estride as f64 * hp.n_expert_used as f64;
            }
            if l.attn.is_some() {
                b += (ctx * hp.n_head_kv * hp.head_dim * 2 * 2) as f64;
            }
            if l.gdn.is_some() {
                // the recurrent state, read and written
                b += (2 * self.geo.gdn_hv * 128 * 128 * 4) as f64;
            }
        }
        b + dense_bytes(&self.head)
    }
}

/// Every layer's expert-weight bytes (the part of a layer that can land on either GPU; the
/// router/norm/attention weights are uniform enough across layers not to matter, and the
/// K/V + GDN state is allocated on whichever device owns the layer either way).
fn layer_vram(g: &Gguf, hp: &Hparams) -> Vec<usize> {
    (0..hp.n_layer)
        .map(|l| {
            let pre = format!("blk.{l}.");
            ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .iter()
                .map(|t| {
                    g.get(&format!("{pre}{t}.weight"))
                        .map_or(0, |t| t.nbytes as usize)
                })
                .sum::<usize>()
        })
        .collect()
}

/// The first layer of GPU 1 in a `TANG_GPUS=2` load: `TANG_SPLIT_AT` overrides; otherwise
/// the cut that best balances the two devices' weight bytes (the embedding and head, which
/// stay on GPU 0, are a rounding error next to the experts).
fn split_point(g: &Gguf, hp: &Hparams) -> usize {
    let cap = hp.n_layer.saturating_sub(1);
    if let Ok(k) = std::env::var("TANG_SPLIT_AT") {
        return k
            .parse::<usize>()
            .unwrap_or_else(|_| {
                eprintln!("TANG_SPLIT_AT: a layer index; using the balanced split");
                balanced(g, hp)
            })
            .min(cap);
    }
    balanced(g, hp).min(cap)
}

/// The layer index whose prefix of weight bytes is closest to half the total (ties: the
/// smaller side; `n_layer / 2` when the tensor table can't be read).
fn balanced(g: &Gguf, hp: &Hparams) -> usize {
    let bytes = layer_vram(g, hp);
    let total: usize = bytes.iter().sum();
    let (mut best, mut best_err) = (hp.n_layer / 2, usize::MAX);
    for k in 0..bytes.len() {
        let a: usize = bytes[..=k].iter().sum();
        let err = a.abs_diff(total / 2);
        if err < best_err {
            (best, best_err) = (k, err);
        }
    }
    best + 1 // layers [0, best] on GPU 0
}

/// GPU 1's half of a two-GPU load: layers `[split_at, L)` with their own kernels, scratch,
/// pinned staging, side stream, the output norm, head, rope, and (when requested) the MTP
/// parts. Returns the peer, the layer split, and its share of the load report.
#[allow(clippy::too_many_arguments)]
fn build_peer(
    g: &Gguf,
    gpu0: &Gpu,
    hp: &Hparams,
    geo: &Geometry,
    sh: &Shapes,
    max_ctx: usize,
    mtp_path: &Option<std::path::PathBuf>,
    split: usize,
    ba_perm: &[usize],
) -> Result<(Box<Peer>, String)> {
    let split_at = split;
    let ctx = super::split::Ctx::for_device(1)?;
    let peer_ok = ctx
        .enable_peer(gpu0)
        .map_err(|e| anyhow!("peer access: {e}"))?;
    if !peer_ok {
        eprintln!("coder: GPU 1 has no peer access; boundary copies stage through host");
    }
    let (free0, _) = ctx.gpu.mem_info().map_err(|e| anyhow!("GPU 1: {e}"))?;
    let dev = CudaComputeDevice::on(1).map_err(|e| anyhow!("GPU 1: {e:?}"))?;
    let mut layers = Vec::with_capacity(hp.n_layer - split_at);
    for l in split_at..hp.n_layer {
        layers.push(load_layer(
            &dev,
            g,
            hp,
            geo,
            &format!("blk.{l}."),
            hp.is_recurrent[l],
            ba_perm,
            max_ctx,
        )?);
    }
    let ff_sh = hp.n_ff_exp;
    let kmax = [2 * hp.n_embd, geo.gdn_v(), hp.n_head * hp.head_dim, ff_sh]
        .into_iter()
        .max()
        .unwrap();
    let s = Scratch::new(&dev, hp, geo, kmax, max_ctx, MAX_T, MAX_T, MAX_T);
    let k = Kern::build_on(&ctx.gpu, sh, geo, MAX_T)?;
    let head_name = if g.get("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let head = upload_dense(&dev, g, &[(g.info(head_name)?.clone(), None)])?;
    let out_norm = dev.upload_f32(&g.tensor_f32("output_norm.weight")?);
    let rope = {
        let (c, s) = fl::rope_table(max_ctx, hp.rope_base as f64);
        (dev.upload_f32(&c), dev.upload_f32(&s))
    };
    let mut io = Io {
        arena: HostArena::new(
            Io::bytes(hp.n_embd, MAX_T),
            ArenaOptions {
                try_hugetlb: false,
                thp: false,
            },
        )?,
    };
    io.arena
        .register(&ctx.gpu.ctx, &[])
        .map_err(|e| anyhow!("{e}"))?;
    dev.sync();
    let (free1, _) = ctx.gpu.mem_info()?;
    let spent = (free0 - free1) as f64 / 1e9;
    let _ = mtp_path; // the MTP layer stays on GPU 0 (it reads GPU 0's pinned staging)
    let report = format!(
        "; GPU 1: layers {split_at}..{} ({:.1} GB), peer split at {split_at}",
        hp.n_layer, spent,
    );
    let peer = Peer {
        dev,
        ctx,
        layers,
        out_norm,
        head,
        rope,
        s,
        io,
        k,
        side: Stream::new().map_err(|e| anyhow!("{e}"))?,
        ev_fork: gpu::Event::new(false).map_err(|e| anyhow!("{e}"))?,
        ev_join: gpu::Event::new(false).map_err(|e| anyhow!("{e}"))?,
        wide: None,
        graphs: HashMap::new(),
        commit_graphs: (0..=MAX_T).map(|_| None).collect(),
        slots: (0..MAX_SLOTS).map(|_| None).collect(),
    };
    Ok((Box::new(peer), report))
}

/// One decoder layer's weights and state buffers from `g`, its tensors named `{pre}…` (a GDN
/// layer when `rec`, else gated attention), with a K/V cache of `max_ctx`.
#[allow(clippy::too_many_arguments)]
fn load_layer(
    dev: &CudaComputeDevice,
    g: &Gguf,
    hp: &Hparams,
    geo: &Geometry,
    pre: &str,
    rec: bool,
    ba_perm: &[usize],
    max_ctx: usize,
) -> Result<Layer> {
    let info = |name: &str| -> Result<TensorInfo> { Ok(g.info(name)?.clone()) };
    let vecf = |name: &str| -> Result<B> { Ok(dev.upload_f32(&g.tensor_f32(name)?)) };
    let hv = hp.ssm_v_heads;
    let t = |s: &str| format!("{pre}{s}");
    let (w_in, gdn, attn) = if rec {
        let w_in = upload_dense(
            &dev,
            &g,
            &[
                (info(&t("attn_qkv.weight"))?, None),
                (info(&t("attn_gate.weight"))?, None),
                (info(&t("ssm_ba.weight"))?, Some(&ba_perm[..])),
            ],
        )?;
        ensure!(
            w_in.rows == geo.gdn_b() + hv,
            "{pre} GDN input rows {}",
            w_in.rows
        );
        let gd = GdnL {
            conv: vecf(&t("ssm_conv1d.weight"))?,
            dt: vecf(&t("ssm_dt.bias"))?,
            a: vecf(&t("ssm_a"))?,
            norm: vecf(&t("ssm_norm.weight"))?,
            state: dev.alloc_f32(hv * 128 * 128),
            hist: dev.alloc_f32(3 * geo.gdn_conv()),
            proj: dev.alloc_f32(MAX_T * w_in.rows),
            stab: dev.alloc_f32(2 * MAX_SLOTS),
            htab: dev.alloc_f32(2 * MAX_SLOTS),
        };
        (w_in, Some(gd), None)
    } else {
        let w_in = upload_dense(
            &dev,
            &g,
            &[
                (info(&t("attn_q.weight"))?, None),
                (info(&t("attn_k.weight"))?, None),
                (info(&t("attn_v.weight"))?, None),
            ],
        )?;
        ensure!(
            w_in.rows == (2 * hp.n_head + 2 * hp.n_head_kv) * hp.head_dim,
            "{pre} attention input rows {}",
            w_in.rows
        );
        let at = AttnL {
            qn: vecf(&t("attn_q_norm.weight"))?,
            kn: vecf(&t("attn_k_norm.weight"))?,
            k: dev.alloc_bf16(max_ctx * hp.n_head_kv * hp.head_dim),
            v: dev.alloc_bf16(max_ctx * hp.n_head_kv * hp.head_dim),
            proj: dev.alloc_f32(MAX_T * w_in.rows),
            ktab: dev.alloc_f32(2 * MAX_SLOTS),
            vtab: dev.alloc_f32(2 * MAX_SLOTS),
        };
        (w_in, None, Some(at))
    };
    let w_out = upload_dense(
        &dev,
        &g,
        &[(
            info(&t(if rec {
                "ssm_out.weight"
            } else {
                "attn_output.weight"
            }))?,
            None,
        )],
    )?;
    let router = upload_dense(
        &dev,
        &g,
        &[
            (info(&t("ffn_gate_inp.weight"))?, None),
            (info(&t("ffn_gate_inp_shexp.weight"))?, None),
        ],
    )?;
    ensure!(router.rows == hp.n_expert + 1, "{pre} router rows");
    let sh_gate = upload_experts(&dev, &g, &info(&t("ffn_gate_shexp.weight"))?)?;
    let sh_up = upload_experts(&dev, &g, &info(&t("ffn_up_shexp.weight"))?)?;
    let sh_down = upload_experts(&dev, &g, &info(&t("ffn_down_shexp.weight"))?)?;
    ensure!(
        sh_gate.w.estride as usize / sh_gate.w.row_bytes as usize == hp.n_ff_exp,
        "{pre}: the shared expert's FF must equal the routed experts' ({})",
        hp.n_ff_exp
    );
    let gate = upload_experts(&dev, &g, &info(&t("ffn_gate_exps.weight"))?)?;
    let up = upload_experts(&dev, &g, &info(&t("ffn_up_exps.weight"))?)?;
    let down = upload_experts(&dev, &g, &info(&t("ffn_down_exps.weight"))?)?;
    Ok(Layer {
        attn_norm: vecf(&t("attn_norm.weight"))?,
        post_norm: vecf(&t("post_attention_norm.weight"))?,
        w_in,
        w_out,
        router,
        r16_gu: [&gate, &up, &sh_gate, &sh_up]
            .iter()
            .map(|e| (e.w.row_bytes + 15) / 16 + 1)
            .max()
            .unwrap(),
        r16_down: [&down, &sh_down]
            .iter()
            .map(|e| (e.w.row_bytes + 15) / 16 + 1)
            .max()
            .unwrap(),
        sh_gate,
        sh_up,
        sh_down,
        gate,
        up,
        down,
        gdn,
        attn,
    })
}

fn dense_bytes(d: &Dense) -> f64 {
    d.bytes as f64
}

/// Stack `parts` (tensor, optional row permutation: output row `i` is input row `perm[i]`) by
/// rows into one native-GEMV weight. Types the GEMV doesn't decode are dequantized to f32.
fn upload_dense(
    dev: &CudaComputeDevice,
    g: &Gguf,
    parts: &[(TensorInfo, Option<&[usize]>)],
) -> Result<Dense> {
    let k = parts[0].0.row_len();
    let mut ws = Vec::new();
    let mut words = Vec::new();
    let mut segs16 = Vec::new();
    let mut off = 0usize;
    let mut total = 0usize;
    for (t, perm) in parts {
        ensure!(
            t.row_len() == k,
            "{}: {} wide, want {k}",
            t.name,
            t.row_len()
        );
        let rows = t.n_rows();
        let (ty, rb, mut bytes) = match native_id(t.ty) {
            Some(id) => (id, t.row_bytes()?, g.bytes(t).to_vec()),
            None => {
                let v = g.dequantize(t)?;
                (0, k * 4, v.iter().flat_map(|x| x.to_le_bytes()).collect())
            }
        };
        if let Some(p) = perm {
            ensure!(
                p.len() == rows,
                "{}: permutation of {} rows",
                t.name,
                p.len()
            );
            let src = bytes.clone();
            for (i, &j) in p.iter().enumerate() {
                bytes[i * rb..(i + 1) * rb].copy_from_slice(&src[j * rb..(j + 1) * rb]);
            }
        }
        // TANG_CODER_ROUTER16=1: F32 rows (the router) as bf16, half the bytes.
        let (ty, rb, bytes) = if ty == 0 && env_is("TANG_CODER_ROUTER16", "1") {
            let b16: Vec<u8> = bytes
                .chunks(4)
                .flat_map(|c| {
                    let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]).to_bits();
                    (((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16).to_le_bytes()
                })
                .collect();
            (30, k * 2, b16)
        } else if ty == 8 && k.is_multiple_of(1024) && !env_is("TANG_CODER_Q8X", "0") {
            // Q8X (type 108, default; TANG_CODER_Q8X=0 keeps Q8_0): [k codes][k / 32 f16 scales] a
            // row, lossless.
            let nb = k / 32;
            let mut out = Vec::with_capacity(rows * (k + 2 * nb));
            for r in 0..rows {
                let row = &bytes[r * rb..(r + 1) * rb];
                for b in 0..nb {
                    out.extend_from_slice(&row[b * 34 + 2..b * 34 + 34]);
                }
                for b in 0..nb {
                    out.extend_from_slice(&row[b * 34..b * 34 + 2]);
                }
            }
            (108, k + 2 * nb, out)
        } else {
            (ty, rb, bytes)
        };
        let mut bytes = bytes;
        // 16 bytes of slack: the GEMV stages rows with aligned 16-byte loads.
        total += rows * rb;
        bytes.extend_from_slice(&[0u8; 16]);
        let w = dev.upload_bytes(&bytes);
        let p = dev.buffer_addr(&w);
        words.extend_from_slice(&[
            p as u32,
            (p >> 32) as u32,
            ty,
            rows as u32,
            rb as u32,
            off as u32,
        ]);
        if ty != 0 && ty != 30 && ty != 108 {
            segs16.push(rb);
        }
        ws.push(w);
        off += rows;
    }
    let rb = segs16.into_iter().max().unwrap_or(0);
    const GEMM_TYPES: [u32; 11] = [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 108];
    let gemm_ok = k.is_multiple_of(32)
        && words
            .chunks(6)
            .all(|w| GEMM_TYPES.contains(&w[2]) && (w[3] as usize).is_multiple_of(16));
    // Batch windows: the Q8X segments on tensor cores (their own segment table), any others
    // (GDN's Q4_K ssm_ba, a BF16 layer) through the K-split GEMV as a second matrix.
    let (w108, wrest): (Vec<&[u32]>, Vec<&[u32]>) = words.chunks(6).partition(|w| w[2] == 108);
    let q8x_ok = k.is_multiple_of(256)
        && !w108.is_empty()
        && w108.iter().all(|w| (w[3] as usize).is_multiple_of(16));
    let mma = q8x_ok.then(|| {
        let ws: Vec<u32> = w108.iter().flat_map(|w| w.iter().copied()).collect();
        let rows: usize = w108.iter().map(|w| w[3] as usize).sum();
        (dev.upload_u32(&ws), w108.len() as i32, rows)
    });
    let rest = (q8x_ok && !wrest.is_empty()).then(|| {
        let ws: Vec<u32> = wrest.iter().flat_map(|w| w.iter().copied()).collect();
        Box::new(Dense {
            _w: Vec::new(),
            segs: dev.upload_u32(&ws),
            nseg: wrest.len() as i32,
            rows: wrest.iter().map(|w| w[3] as usize).sum(),
            k,
            row16: ((rb + 15).div_ceil(16) + 1) as i32,
            bytes: 0,
            gemm_ok: false,
            q8x_ok: false,
            mma: None,
            rest: None,
            float1: false,
        })
    });
    let float1 = words.chunks(6).all(|w| w[2] == 0 || w[2] == 30);
    Ok(Dense {
        gemm_ok,
        q8x_ok,
        mma,
        rest,
        float1,
        segs: dev.upload_u32(&words),
        nseg: parts.len() as i32,
        rows: off,
        k,
        row16: ((rb + 15).div_ceil(16) + 1) as i32,
        bytes: total,
        _w: ws,
    })
}

/// A `[n_expert][rows][k]` expert tensor into VRAM in its GGUF type (Q8_0 when the expert
/// kernels don't decode it).
fn upload_experts(dev: &CudaComputeDevice, g: &Gguf, t: &TensorInfo) -> Result<Experts> {
    ensure!(
        t.dims.len() == 3 || t.dims.len() == 2,
        "{}: not an expert stack or matrix",
        t.name
    );
    let (k, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let ne = t.dims.get(2).map_or(1, |&d| d as usize);
    let (ty, rb, mut bytes) = match native_id(t.ty).filter(|&id| id != 0 && id != 30) {
        Some(id) => (id, t.row_bytes()?, g.bytes(t).to_vec()),
        None => {
            let mut out = Vec::with_capacity(ne * rows * k / 32 * 34);
            for e in 0..ne {
                let v = if ne == 1 {
                    g.dequantize(t)?
                } else {
                    g.expert(t, e)?
                };
                out.extend_from_slice(&quantize_q8_0(&v));
            }
            (8, k / 32 * 34, out)
        }
    };
    ensure!(
        bytes.len() == ne * rows * rb,
        "{}: {} bytes, want {}",
        t.name,
        bytes.len(),
        ne * rows * rb
    );
    bytes.extend_from_slice(&[0u8; 16]);
    let w = dev.upload_bytes(&bytes);
    let base = dev.buffer_addr(&w);
    if t.ty != GgmlType::from_id(ty) {
        eprintln!("coder: {} requantized to Q8_0", t.name);
    }
    Ok(Experts {
        w: ExW {
            base,
            estride: (rows * rb) as u64,
            ty: ty as i32,
            row_bytes: rb as i32,
        },
        _w: w,
    })
}

/// Is the GGUF at `path` a `qwen3next` model (this engine) rather than Flash-Next?
pub fn is_qwen3next(path: &Path) -> Result<bool> {
    let g = Gguf::open(path)?;
    Ok(g.meta_str("general.architecture")
        .is_ok_and(|a| a == "qwen3next"))
}

// ---- serving: states, sampler, warm-up

impl Coder {
    /// Copy the running state out: every GDN layer's recurrent state and conv history (the
    /// attention K/V is positional, see [`save_positional`](Self::save_positional)).
    pub fn save_running(&mut self) -> Result<super::engine::RunningState> {
        self.save_running_at(0)
    }

    /// Slot `slot`'s GDN state and conv history buffers, per GDN layer.
    fn running_bufs(&self, slot: usize) -> Result<Vec<(&B, &B)>> {
        if slot == 0 {
            return Ok(self
                .layers
                .iter()
                .filter_map(|l| l.gdn.as_ref())
                .map(|g| (&g.state, &g.hist))
                .collect());
        }
        let sl = self.slots[slot]
            .as_ref()
            .ok_or_else(|| anyhow!("slot {slot} isn't allocated"))?;
        Ok(sl
            .gdn
            .iter()
            .filter_map(|x| x.as_ref())
            .map(|(a, b)| (a, b))
            .collect())
    }

    /// Slot `slot`'s recurrent state (after its tokens), to resume it later in any slot.
    pub fn save_running_at(&mut self, slot: usize) -> Result<super::engine::RunningState> {
        self.sync_all();
        let mut parts = Vec::new();
        for (st, hist) in self.running_bufs(slot)? {
            parts.push(self.dev.download(st));
            parts.push(self.dev.download(hist));
        }
        if let Some(p) = self.peer.as_deref_mut() {
            let bufs: Vec<(&B, &B)> = if slot == 0 {
                p.layers
                    .iter()
                    .filter_map(|l| l.gdn.as_ref())
                    .map(|g| (&g.state, &g.hist))
                    .collect()
            } else {
                p.slots[slot]
                    .as_ref()
                    .ok_or_else(|| anyhow!("slot {slot} isn't allocated"))?
                    .0
                    .iter()
                    .filter_map(|x| x.as_ref())
                    .map(|(a, b)| (a, b))
                    .collect()
            };
            for (st, hist) in bufs {
                parts.push(p.dev.download(st));
                parts.push(p.dev.download(hist));
            }
        }
        let n = parts.len() / 2;
        parts.push(Vec::new());
        super::engine::RunningState::from_parts(self.slot_tokens(slot).len(), n, parts)
    }

    /// Resume at `s.pos` with `prefix` (the `s.pos` tokens `s` was saved after).
    pub fn restore_running(
        &mut self,
        s: &super::engine::RunningState,
        prefix: &[u32],
    ) -> Result<()> {
        self.restore_running_at(0, s, prefix)
    }

    /// [`restore_running`](Self::restore_running) into slot `slot`.
    pub fn restore_running_at(
        &mut self,
        slot: usize,
        s: &super::engine::RunningState,
        prefix: &[u32],
    ) -> Result<()> {
        ensure!(
            prefix.len() == s.pos,
            "prefix of {} tokens for a state at {}",
            prefix.len(),
            s.pos
        );
        let parts = s.parts();
        let bufs: Vec<(u64, u64)> = self
            .running_bufs(slot)?
            .into_iter()
            .map(|(a, b)| (self.a(a), self.a(b)))
            .collect();
        // A saved state covers both stages' GDN layers (GPU 0's first, GPU 1's after).
        let total_gdn = bufs.len()
            + self
                .peer
                .as_ref()
                .map_or(0, |p| p.layers.iter().filter(|l| l.gdn.is_some()).count());
        ensure!(
            s.gdn_layers() == total_gdn && parts.len() == 2 * total_gdn + 1,
            "state is for another model"
        );
        let n0 = bufs.len();
        self.dev.sync();
        for (i, (st, hist)) in bufs.into_iter().enumerate() {
            for (addr, host) in [(st, parts[2 * i]), (hist, parts[2 * i + 1])] {
                htod(addr, bytemuck_f32(host))?;
            }
        }
        if let Some(p) = self.peer.as_deref_mut() {
            let bufs1: Vec<(&B, &B)> = if slot == 0 {
                p.layers
                    .iter()
                    .filter_map(|l| l.gdn.as_ref())
                    .map(|g| (&g.state, &g.hist))
                    .collect()
            } else {
                p.slots[slot]
                    .as_ref()
                    .ok_or_else(|| anyhow!("slot {slot} isn't allocated"))?
                    .0
                    .iter()
                    .filter_map(|x| x.as_ref())
                    .map(|(a, b)| (a, b))
                    .collect()
            };
            for (i, (st, hist)) in bufs1.into_iter().enumerate() {
                let (a, b) = (p.dev.buffer_addr(st), p.dev.buffer_addr(hist));
                for (addr, host) in [(a, parts[2 * (n0 + i)]), (b, parts[2 * (n0 + i) + 1])] {
                    htod(addr, bytemuck_f32(host))?;
                }
            }
            p.ctx.sync().expect("GPU 1 sync");
        }
        self.dev.sync();
        let toks = self.slot_tokens_mut(slot)?;
        toks.clear();
        toks.extend_from_slice(prefix);
        Ok(())
    }

    /// The positional buffers (attention K and V per layer, and the MTP layer's) of slot
    /// `slot`: device address, bytes a position.
    fn positional_bufs_at(&self, slot: usize) -> Result<Vec<(u64, usize)>> {
        let row = self.hp.n_head_kv * self.hp.head_dim * 2;
        if slot == 0 {
            return Ok(self
                .layers
                .iter()
                .chain(self.mtp.iter().map(|m| &m.layer))
                .filter_map(|l| l.attn.as_ref())
                .flat_map(|a| [(self.a(&a.k), row), (self.a(&a.v), row)])
                .collect());
        }
        let sl = self.slots[slot]
            .as_ref()
            .ok_or_else(|| anyhow!("slot {slot} isn't allocated"))?;
        Ok(sl
            .kv
            .iter()
            .filter_map(|x| x.as_ref())
            .chain(sl.mtp_kv.iter())
            .flat_map(|(k, v)| [(self.a(k), row), (self.a(v), row)])
            .collect())
    }

    /// Copy positions `from..to` of the positional state out.
    pub fn save_positional(&mut self, from: usize, to: usize) -> Result<super::engine::Positional> {
        self.save_positional_at(0, from, to)
    }

    /// [`save_positional`](Self::save_positional) of slot `slot`.
    pub fn save_positional_at(
        &mut self,
        slot: usize,
        from: usize,
        to: usize,
    ) -> Result<super::engine::Positional> {
        ensure!(from <= to && to <= self.max_ctx);
        self.sync_all();
        let mut bufs = Vec::new();
        for (addr, row) in self.positional_bufs_at(slot)? {
            let mut host = vec![0u8; (to - from) * row];
            if !host.is_empty() {
                let r = unsafe {
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        host.as_mut_ptr() as *mut _,
                        addr + (from * row) as u64,
                        host.len(),
                    )
                };
                ensure!(
                    r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "cuMemcpyDtoH: {r:?}"
                );
            }
            bufs.push(host);
        }
        if let Some(p) = self.peer.as_deref_mut() {
            let row = self.hp.n_head_kv * self.hp.head_dim * 2;
            let bufs1: Vec<(&B, &B)> = if slot == 0 {
                p.layers
                    .iter()
                    .filter_map(|l| l.attn.as_ref())
                    .flat_map(|a| [(&a.k, &a.v)])
                    .collect()
            } else {
                p.slots[slot]
                    .as_ref()
                    .ok_or_else(|| anyhow!("slot {slot} isn't allocated"))?
                    .1
                    .iter()
                    .filter_map(|x| x.as_ref())
                    .flat_map(|(k, v)| [(k, v)])
                    .collect()
            };
            for (k, v) in bufs1 {
                for b in [k, v] {
                    let addr = p.dev.buffer_addr(b);
                    let mut host = vec![0u8; (to - from) * row];
                    if !host.is_empty() {
                        let r = unsafe {
                            cudarc::driver::sys::cuMemcpyDtoH_v2(
                                host.as_mut_ptr() as *mut _,
                                addr + (from * row) as u64,
                                host.len(),
                            )
                        };
                        ensure!(
                            r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                            "cuMemcpyDtoH: {r:?}"
                        );
                    }
                    bufs.push(host);
                }
            }
        }
        Ok(super::engine::Positional::from_parts(from, to, bufs))
    }

    /// Write positions `from.max(p.from)..p.to` of `p` back.
    pub fn restore_positional(&mut self, p: &super::engine::Positional, from: usize) -> Result<()> {
        self.restore_positional_at(0, p, from)
    }

    /// [`restore_positional`](Self::restore_positional) into slot `slot`.
    pub fn restore_positional_at(
        &mut self,
        slot: usize,
        p: &super::engine::Positional,
        from: usize,
    ) -> Result<()> {
        let bufs = self.positional_bufs_at(slot)?;
        // GPU 0's K/V first, then GPU 1's (the order `save_positional_at` wrote).
        let mut n0 = bufs.len();
        if let Some(p) = self.peer.as_ref() {
            n0 += if slot == 0 {
                2 * p.layers.iter().filter(|l| l.attn.is_some()).count()
            } else {
                p.slots[slot]
                    .as_ref()
                    .ok_or_else(|| anyhow!("slot {slot} isn't allocated"))?
                    .1
                    .iter()
                    .filter(|x| x.is_some())
                    .count()
                    * 2
            };
        }
        ensure!(
            n0 == p.parts().len(),
            "positional state is for another model"
        );
        self.sync_all();
        let from = from.max(p.from);
        let parts = p.parts();
        let mut i = 0;
        for ((addr, row), host) in bufs.into_iter().zip(parts) {
            i += 1;
            ensure!(
                host.len() == (p.to - p.from) * row,
                "positional state size differs"
            );
            if p.to > from {
                htod(
                    addr + (from * row) as u64,
                    &host[(from - p.from) * row..(p.to - p.from) * row],
                )?;
            }
        }
        if let Some(peer) = self.peer.as_deref_mut() {
            let row = self.hp.n_head_kv * self.hp.head_dim * 2;
            let bufs1: Vec<(&B, &B)> = if slot == 0 {
                peer.layers
                    .iter()
                    .filter_map(|l| l.attn.as_ref())
                    .flat_map(|a| [(&a.k, &a.v)])
                    .collect()
            } else {
                peer.slots[slot]
                    .as_ref()
                    .ok_or_else(|| anyhow!("slot {slot} isn't allocated"))?
                    .1
                    .iter()
                    .filter_map(|x| x.as_ref())
                    .flat_map(|(k, v)| [(k, v)])
                    .collect()
            };
            for (k, v) in bufs1 {
                for b in [k, v] {
                    let addr = peer.dev.buffer_addr(b);
                    let host = &parts[i];
                    i += 1;
                    ensure!(
                        host.len() == (p.to - p.from) * row,
                        "positional state size differs"
                    );
                    if p.to > from {
                        htod(
                            addr + (from * row) as u64,
                            &host[(from - p.from) * row..(p.to - p.from) * row],
                        )?;
                    }
                }
            }
            peer.ctx.sync().expect("GPU 1 sync");
        }
        Ok(())
    }

    /// Sample in the head with top-k / top-p / a presence penalty (`flash::sampler`) instead of
    /// the Gumbel-max over the whole vocabulary. Window graphs are recaptured.
    pub fn enable_sampler(&mut self) -> Result<()> {
        if self.peer.is_some() {
            // The sampler's kernels are compiled on GPU 0's context; on a split the head
            // runs on GPU 1, where those handles are invalid. Sampling stays greedy until
            // the sampler is per-stage; generation is otherwise unaffected.
            eprintln!("coder: sampling with TANG_GPUS=2 stays greedy for now");
            return Ok(());
        }
        if self.sampler.is_none() {
            self.sampler = Some(Box::new(super::sampler::Sampler::new(
                &self.gpu,
                &self.dev,
                self.hp.n_vocab,
            )?));
            self.dev.sync();
            self.graphs.clear();
        }
        Ok(())
    }

    pub fn sampler_mut(&mut self) -> Option<&mut super::sampler::Sampler> {
        self.sampler.as_deref_mut()
    }

    /// Per-row sampling in slot windows (`prefill_slot`, `decode_batch`, `verify_batch` rows
    /// without drafts): each slot samples with its own [`set_slot_sampling`](Self::set_slot_sampling)
    /// settings; greedy rows keep the argmax bit for bit. One GPU. Window graphs are recaptured.
    pub fn enable_row_sampling(&mut self) -> Result<()> {
        ensure!(self.peer.is_none(), "per-row sampling needs one GPU");
        ensure!(self.sampler.is_none(), "per-row sampling replaces the engine sampler");
        if self.row_sampler.is_none() {
            let rows = MAX_T.max(256);
            self.row_sampler = Some(Box::new(super::sampler::RowSampler::new(
                &self.gpu,
                &self.dev,
                self.hp.n_vocab,
                rows,
                MAX_SLOTS,
            )?));
            self.graphs.clear();
        }
        Ok(())
    }

    /// Slot `slot`'s sampling from now on, its output history (presence penalty) cleared.
    pub fn set_slot_sampling(&mut self, slot: usize, s: super::sampler::RowSampling) -> Result<()> {
        ensure!(slot < MAX_SLOTS, "slot {slot}");
        if s.sampled() {
            let rs = self
                .row_sampler
                .as_ref()
                .ok_or_else(|| anyhow!("per-row sampling is off (enable_row_sampling)"))?;
            rs.clear_slot(&self.dev, slot)?;
        }
        self.slot_samp[slot] = s;
        Ok(())
    }

    /// The next window's sampled rows: `(row, slot, position)` for each row whose pick is a
    /// slot's next token.
    fn stage_row_words(&mut self, rows: &[(usize, usize, usize)]) {
        self.row_words.clear();
        if self.row_sampler.is_none() {
            return;
        }
        let w = super::sampler::ROW_WORDS;
        for &(r, slot, pos) in rows {
            let s = self.slot_samp[slot];
            if !s.sampled() {
                continue;
            }
            if self.row_words.len() < (r + 1) * w {
                self.row_words.resize((r + 1) * w, 0);
            }
            self.row_words[r * w..(r + 1) * w].copy_from_slice(&s.words(slot, pos));
        }
    }

    /// A new generation's output starts at position `pos` (penalties count output from there).
    pub fn sampler_begin(&mut self, pos: usize) -> Result<()> {
        match self.sampler.as_mut() {
            Some(s) => s.begin(&self.dev, pos),
            None => Ok(()),
        }
    }

    /// Capture every window width's graphs (plain and verify) and the commits on a throwaway
    /// sequence, then reset.
    pub fn warm(&mut self) -> Result<()> {
        self.reset();
        let ids: Vec<u32> = (0..16).map(|i| 1000 + i).collect();
        let mut cur = self.prefill(&ids, MAX_T, None)?;
        for t in 1..=MAX_T {
            let d: Vec<u32> = (0..t - 1).map(|i| 7 + i as u32).collect();
            cur = *self.verify(cur, &d)?.last().unwrap();
        }
        self.reset();
        Ok(())
    }
}

/// Copy `src` from the host to device address `addr`.
fn htod(addr: u64, src: &[u8]) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    let r =
        unsafe { cudarc::driver::sys::cuMemcpyHtoD_v2(addr, src.as_ptr() as *const _, src.len()) };
    ensure!(
        r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "cuMemcpyHtoD: {r:?}"
    );
    Ok(())
}

/// `v`'s bytes.
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
