//! Device side of the port (M2): weights uploaded in their checkpoint format, and launchers for
//! [`SRC`]'s kernels. Correctness first; every launcher is tested against its CPU twin in
//! [`super::forward`].
//!
//! Pod or mew (`flock ~/tang-gpu.lock` around GPU runs):
//! `DS41_TINY=/tmp/ds41-tiny TANG_REQUIRE_CUDA=1 cargo test --release --no-default-features
//! --features cuda --lib flash::deepseek::gpu`

use super::st::{Dtype, St};
use anyhow::{bail, Context, Result};
use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

/// kernels.cu, then the device-side selection (one module, so the latter can use the former's
/// helpers).
pub const SRC: &str = concat!(
    include_str!("kernels.cu"),
    include_str!("kernels_select.cu")
);

/// Threads per block of the selection kernels (`DS_SEL_THREADS` in kernels_select.cu).
const SEL: (u32, u32, u32) = (512, 1, 1);

type Func = cudarc::driver::sys::CUfunction;

/// `struct DsW` in kernels.cu (8 + 8 + 4 + 4 + 4, padded to 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DsW {
    pub w: u64,
    pub s: u64,
    pub rows: i32,
    pub cols: i32,
    pub fmt: i32,
    _pad: i32,
}

/// A weight on the device and its kernel view.
pub struct DevW {
    _w: DevBuf,
    _s: Option<DevBuf>,
    pub v: DsW,
}

impl DevW {
    /// Upload `name` from the checkpoint as stored: f32, bf16, fp8 with 32 × 32 block scales,
    /// or I8-packed fp4 with per-row × 32 scales.
    pub fn upload(st: &St, name: &str) -> Result<Self> {
        let info = st.info(name)?.clone();
        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            _ => (info.shape[0], info.shape[1..].iter().product()),
        };
        let scale = || -> Result<DevBuf> {
            let sname = format!(
                "{}scale",
                name.strip_suffix("weight")
                    .context("quantized non-weight")?
            );
            Ok(DevBuf::from_slice(st.bytes(&sname)?)?)
        };
        let (fmt, cols, s) = match info.dtype {
            Dtype::F32 => (0, cols, None),
            Dtype::BF16 => (1, cols, None),
            Dtype::F8E4M3 => {
                let ss = st
                    .info(&format!("{}scale", name.strip_suffix("weight").unwrap()))?
                    .shape
                    .clone();
                if ss != [rows.div_ceil(32), cols.div_ceil(32)] {
                    bail!("{name}: per-row fp8 scales {ss:?} are not a GEMM weight");
                }
                (2, cols, Some(scale()?))
            }
            Dtype::I8 | Dtype::U8 => (3, cols * 2, Some(scale()?)),
            d => bail!("{name}: {d:?} is not a weight format"),
        };
        let w = DevBuf::from_slice(st.bytes(name)?)?;
        let v = DsW {
            w: w.ptr,
            s: s.as_ref().map_or(0, |b| b.ptr),
            rows: rows as i32,
            cols: cols as i32,
            fmt,
            _pad: 0,
        };
        Ok(DevW { _w: w, _s: s, v })
    }
}

pub struct Kernels {
    _m: Module,
    linear: Func,
    fp8_qdq: Func,
    fp4_qdq: Func,
    rms: Func,
    rope: Func,
    sparse_attn: Func,
    sinkhorn: Func,
    /// `ds_lin_f{fmt}_t{tm}_g{gu}`, by (fmt - 2, log2 tm, gu).
    fast: [[[Func; 2]; 4]; 2],
    /// `ds_ks_f{fmt}_g{gu}`, the tensor-core K-split GEMV, by (fmt - 2, gu).
    ks: [[Func; 2]; 2],
    /// `ds_ks2_f2` / `ds_ks2_f3`: two K-split GEMVs on one activation
    ks2: [Func; 2],
    /// `ds_moe_{gu,down}_f{fmt}_t{tm}`, the grouped MoE projections, by (fmt - 2, log2 tm,
    /// down).
    moe: [[[Func; 2]; 4]; 2],
    /// `ds_moe_ks_{gu,down}_f{fmt}`, the grouped MoE on the K-split path, by (fmt - 2, down).
    moe_ks: [[Func; 2]; 2],
    /// `ds_moe_ks_{gu,down}_f{fmt}_s{sfmt}`: [`Kernels::moe_ks`] with the shared expert (format
    /// sfmt) as one more slot, by (fmt - 2, sfmt - 2, down).
    moe_ks_sh: [[[Func; 2]; 2]; 2],
    /// `ds_moe_kp_{gu,down}_f{fmt}_s{sfmt}`: the persistent one-wave form of `moe_ks_sh`
    /// (`DS_MOE_KP=0` keeps the slot grid), same index.
    moe_kp: Option<[[[Func; 2]; 2]; 2]>,
    /// `ds_wo_ks_f{fmt}`, wo_a's grouped K-split GEMV, by fmt - 2, and its quantized activation.
    wo_ks: [Func; 2],
    wo_x: DevBuf,
    attn_qk: Func,
    attn_soft: Func,
    attn_pv: Func,
    /// Scratch for the tiled attention (partial scores, probabilities, denominators).
    scratch: DevBuf,
    /// K-split partials across blocks [SB][t][rows (× 2 for GU)] and per-row-group arrival
    /// counters (left at zero by the last block). One launch at a time per `Kernels`.
    ks_part: DevBuf,
    ks_cnt: DevBuf,
    /// `ds_gemv_bf16_r1`'s scratch: per-chunk partials, and row-group arrival counters (zero)
    gemv_part: DevBuf,
    gemv_cnt: DevBuf,
    /// The quantized activation for the K-split GEMV: e4m3 codes [t][K], then scales.
    ks_x: DevBuf,
    /// The quantized activation of a many-row (t > 8) K-split GEMV, grown on demand.
    ks_big: std::sync::Mutex<Option<(usize, DevBuf)>>,
    ks_quant: Func,
    /// `ds_moe_gate`'s block arrival counter (left at zero by the last block).
    gate_cnt: DevBuf,
    /// `DS_MOE_KS_ITEMS` / `DS_MOE_KS_S`: row groups a grouped K-split block strides and K
    /// slices a block (0 = auto), for sweeping the grid on a new GPU.
    moe_items: usize,
    moe_s: usize,
    /// SM count and the opt-in dynamic shared memory limit per block.
    sms: usize,
    smem_max: usize,
    /// `DS_EXACT=1`: the bit-exact kernels (`ds_lin_*`, `ds_sparse_attn`) instead of the
    /// K-split GEMV and the tiled attention.
    pub exact: bool,
    /// The tiled attention's softmax fused into its PV kernel (`ds_attn_spv`, 2 launches;
    /// `DS_ATTN_FUSE=1`).
    pub attn_fuse: bool,
    /// the split-K decode attention (`ds_attn_fd`); `DS_ATTN_FD=0` keeps the 3-pass tiled one
    pub attn_fd: bool,
    /// `DS_ATTN_FD_S`: key splits per row of `ds_attn_fd` (0: about a wave of SMs)
    attn_fd_s: usize,
    /// `ds_attn_fdm` (tensor cores) for `ds_attn_fd`; `DS_ATTN_FDM=0` keeps the CUDA-core one
    pub attn_fdm: bool,
    /// Every other kernel, by name, for [`Kernels::go`].
    more: std::collections::HashMap<&'static str, Func>,
}

/// The kernels launched through [`Kernels::go`].
const MORE: &[&str] = &[
    "ds_qkv_post",
    "ds_comp_pool",
    "ds_ring_last2",
    "ds_comp_post",
    "ds_attn_spv",
    "ds_attn_fd",
    "ds_attn_fdm",
    "ds_attn_fd_comb",
    "ds_copy",
    "ds_copy_many",
    "ds_engram_gather",
    "ds_hc_scale",
    "ds_hc_sinkhorn",
    "ds_hc_pre",
    "ds_hc_post",
    "ds_hc_post_mix",
    "ds_hc_mix_ks",
    "ds_hc_fused_ks",
    "ds_hc_stage",
    "ds_embed",
    "ds_mean_hc",
    "ds_swiglu",
    "ds_router",
    "ds_linear_t8",
    "ds_gemv_bf16_r1",
    "ds_head_bf16",
    "ds_argmax_rows",
    "ds_gather_rows",
    "ds_scatter_add",
    "ds_add",
    "ds_add_round",
    "ds_scale_round",
    "ds_round",
    "ds_pool",
    "ds_index_score",
    "ds_wo_a",
    "ds_perm3",
    "ds_engram_gate",
    "ds_win_idx",
    "ds_idx_cat",
    "ds_gather_csr",
    "ds_scatter_csr",
    "ds_linear_csr",
    "ds_mask_scores",
    "ds_candidates",
    "ds_topk",
    "ds_route",
    "ds_fill_i32",
    "ds_copy_u8",
    "ds_spark_pick",
    "ds_spark_sample",
    "ds_spec_dist",
    "ds_spec_walk",
    "ds_moe_plan",
    "ds_moe_reduce",
    "ds_moe_reduce_sh",
    "ds_route_plan_q",
    "ds_moe_gate",
    "ds_moe_plan_tiles",
    "ds_kst_f2_g0",
    "ds_kst_f3_g0",
    "ds_kst_f2_g1",
    "ds_kst_f3_g1",
    "ds_moe_tgu_f2",
    "ds_moe_tgu_f3",
    "ds_moe_tdown_f2",
    "ds_moe_tdown_f3",
    "ds_moe_ks_tgu_f2",
    "ds_moe_ks_tgu_f3",
    "ds_moe_ks_tdown_f2",
    "ds_moe_ks_tdown_f3",
];

/// Floats of attention scratch: partial scores [t][d/64][h][k], then P [t][h][k] and den
/// [t][h]. Larger problems fall back to `ds_sparse_attn`.
const ATTN_SCRATCH: usize = 6 << 20;

/// A rope fused into the tiled attention ([`Kernels::sparse_attn_r`]): row r's position is
/// `P[0] + add + r * stride` (`dg_rope`'s `At::Dev` with no fixed row), table `tab`, the last
/// `rd` of each head.
/// The attention's index rows built inside `ds_attn_fdm` ([`Kernels::attn_fd_win_ok`]): the
/// window columns as `dg_win_idx` `mode` (1 decode, 3 verify window) at `P[0]` (the rope's
/// `p`), then the top-k rows `src` [t][kc] (`ds_idx_cat`'s).
#[derive(Clone, Copy, Debug)]
pub struct WinIdx {
    pub mode: i32,
    pub win: i32,
    pub src: u64,
    pub kc: i32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AttnRope {
    pub tab: u64,
    pub p: u64,
    pub rd: i32,
    pub add: i32,
    pub stride: i32,
    /// q too (else o only: q came rope'd)
    pub q: bool,
    /// also quantize o into wo_ks's activation buffer (`ds_attn_fd` paths; then
    /// [`Kernels::wo_ks_pre`] skips its quantize)
    pub wo_quant: bool,
}

/// Floats of cross-block K-split partials, and row-group counters.
const KS_PART: usize = 1 << 21;
const KS_CNT: usize = 1 << 13;
/// [`Kernels::gemv_bf16`]'s partials (floats) and row groups.
const GEMV_PART: usize = 1 << 20;
const GEMV_CNT: usize = 1 << 12;
/// Bytes for the quantized activation: 8 tokens × 16384 columns, codes and scales.
const KS_X: usize = 8 * 16384 * 2;
/// Codes, then scales, of wo_a's activation: 8 tokens × 32768 (64 heads × 512) columns.
const WO_X_COLS: usize = 32768;
const WO_X: usize = 8 * WO_X_COLS + 8 * WO_X_COLS / 8;
/// Row groups each grouped K-split block takes (blockIdx.x strides them).
const MOE_KS_ITEMS: usize = 2;

fn env_usize(k: &str) -> Option<usize> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

/// `__CUDA_ARCH__` the module was built for.
fn arch(m: &Module) -> Result<i32> {
    let f = m.func("ds_arch")?;
    let out = DevBuf::alloc(4)?;
    let s = Stream::blocking()?;
    unsafe { gpu::launch(f, (1, 1, 1), (1, 1, 1), 0, &s, tang_moe::args![out.ptr])? };
    s.sync()?;
    Ok(out.read::<i32>(0, 1)?[0])
}

fn lin_table(m: &Module, pre: &str) -> Result<[[[Func; 2]; 4]; 2]> {
    let mut a = [[[std::ptr::null_mut(); 2]; 4]; 2];
    for (fi, af) in a.iter_mut().enumerate() {
        for (ti, at) in af.iter_mut().enumerate() {
            for (gi, ag) in at.iter_mut().enumerate() {
                *ag = m.func(&format!("{pre}_f{}_t{}_g{gi}", fi + 2, 1 << ti))?;
            }
        }
    }
    Ok(a)
}

fn moe_table(m: &Module) -> Result<[[[Func; 2]; 4]; 2]> {
    let mut a = [[[std::ptr::null_mut(); 2]; 4]; 2];
    for (fi, af) in a.iter_mut().enumerate() {
        for (ti, at) in af.iter_mut().enumerate() {
            for (di, ad) in at.iter_mut().enumerate() {
                let kind = ["gu", "down"][di];
                *ad = m.func(&format!("ds_moe_{kind}_f{}_t{}", fi + 2, 1 << ti))?;
            }
        }
    }
    Ok(a)
}

/// The K split for `tiles` 16-row warp tiles with `groups` 32-column groups: S slices within a
/// block of 8 warps (RW = 8 / S tiles a block) and SB blocks along K, aiming at ≥ 2 blocks of 8
/// warps per SM while keeping ≥ 4 groups a warp. Returns (S, row-group blocks, SB).
/// Work items target `DS_KS_TGT` (default 6) × SMs. 2 is faster on H100 (wq_b 24.5 -> 23.8,
/// wo_b 36.4 -> 32.6 us at t = 6) but changes the partial-sum order, and greedy acceptance fell.
fn ks_split(tiles: usize, groups: usize, sms: usize) -> (usize, usize, usize) {
    static TGT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let tgt = *TGT.get_or_init(|| env_usize("DS_KS_TGT").unwrap_or(6).max(1));
    let target = tgt * sms;
    let mut sp = 1;
    while sp < 8 && tiles.div_ceil(8 / sp) < target && groups >= sp * 2 * 4 {
        sp *= 2;
    }
    let br = tiles.div_ceil(8 / sp).max(1);
    let mut sb = 1;
    while br * sb < target && groups >= (sb + 1) * sp * 4 && sb < 16 {
        sb += 1;
    }
    (sp, br, sb)
}

/// The `ds_lin_*` / `ds_moe_*` row-tile template index for `t` <= 8 rows.
fn tm_index(t: usize) -> Result<usize> {
    Ok(match t {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        5..=8 => 3,
        _ => bail!("tm_index: {t} rows"),
    })
}

pub(crate) fn blocks(n: usize, per: usize) -> (u32, u32, u32) {
    // at least one block: every kernel bounds-checks, and a zero grid is a launch error
    (n.div_ceil(per).max(1) as u32, 1, 1)
}

impl Kernels {
    pub fn new(g: &Gpu) -> Result<Self> {
        // ds_ks's L2 prefetch prologue only pays off under PDL (kernels.cu ks_pf);
        // DS_KS_DEFS="NAME=v,...": extra #defines ahead of the source (kernel sweeps)
        let mut defs = String::new();
        if super::pdl::on() {
            defs.push_str("#define DS_PDL_PF 1\n");
        }
        if let Ok(v) = std::env::var("DS_KS_DEFS") {
            for (n, x) in v.split(',').filter_map(|d| d.split_once('=')) {
                defs.push_str(&format!("#define {n} {x}\n"));
            }
        }
        let m = if defs.is_empty() {
            g.module(SRC)?
        } else {
            g.module(&format!("{defs}{SRC}"))?
        };
        Ok(Kernels {
            linear: m.func("ds_linear")?,
            fp8_qdq: m.func("ds_fp8_qdq")?,
            fp4_qdq: m.func("ds_fp4_qdq")?,
            rms: m.func("ds_rms")?,
            rope: m.func("ds_rope")?,
            sparse_attn: m.func("ds_sparse_attn")?,
            sinkhorn: m.func("ds_sinkhorn")?,
            fast: lin_table(&m, "ds_lin")?,
            moe: moe_table(&m)?,
            moe_ks: [
                [m.func("ds_moe_ks_gu_f2")?, m.func("ds_moe_ks_down_f2")?],
                [m.func("ds_moe_ks_gu_f3")?, m.func("ds_moe_ks_down_f3")?],
            ],
            moe_ks_sh: {
                let mut a = [[[std::ptr::null_mut(); 2]; 2]; 2];
                for (f, af) in a.iter_mut().enumerate() {
                    for (sf, asf) in af.iter_mut().enumerate() {
                        for (di, kind) in ["gu", "down"].iter().enumerate() {
                            asf[di] =
                                m.func(&format!("ds_moe_ks_{kind}_f{}_s{}", f + 2, sf + 2))?;
                        }
                    }
                }
                a
            },
            moe_kp: if std::env::var("DS_MOE_KP").is_ok_and(|v| v == "0") {
                None
            } else {
                let mut a = [[[std::ptr::null_mut(); 2]; 2]; 2];
                for (f, af) in a.iter_mut().enumerate() {
                    for (sf, asf) in af.iter_mut().enumerate() {
                        for (di, kind) in ["gu", "down"].iter().enumerate() {
                            asf[di] =
                                m.func(&format!("ds_moe_kp_{kind}_f{}_s{}", f + 2, sf + 2))?;
                        }
                    }
                }
                Some(a)
            },
            wo_ks: [m.func("ds_wo_ks_f2")?, m.func("ds_wo_ks_f3")?],
            ks2: [m.func("ds_ks2_f2")?, m.func("ds_ks2_f3")?],
            wo_x: DevBuf::alloc(WO_X)?,
            ks: [
                [m.func("ds_ks_f2_g0")?, m.func("ds_ks_f2_g1")?],
                [m.func("ds_ks_f3_g0")?, m.func("ds_ks_f3_g1")?],
            ],
            attn_qk: m.func("ds_attn_qk")?,
            attn_soft: m.func("ds_attn_soft")?,
            attn_pv: m.func("ds_attn_pv")?,
            scratch: DevBuf::alloc(ATTN_SCRATCH * 4)?,
            ks_part: DevBuf::alloc(KS_PART * 4)?,
            ks_cnt: DevBuf::from_slice(&vec![0i32; KS_CNT])?,
            gemv_part: DevBuf::alloc(GEMV_PART * 4)?,
            gemv_cnt: DevBuf::from_slice(&vec![0i32; GEMV_CNT])?,
            ks_x: DevBuf::alloc(KS_X)?,
            ks_big: std::sync::Mutex::new(None),
            ks_quant: m.func("ds_ks_quant")?,
            gate_cnt: DevBuf::from_slice(&[0i32])?,
            moe_items: env_usize("DS_MOE_KS_ITEMS").unwrap_or(MOE_KS_ITEMS).max(1),
            moe_s: env_usize("DS_MOE_KS_S")
                .filter(|s| [1, 2, 4, 8].contains(s))
                .unwrap_or(0),
            sms: g.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )? as usize,
            smem_max: {
                let mx = g.attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                )?;
                for n in [
                    "ds_ks_f2_g0",
                    "ds_ks_f2_g1",
                    "ds_ks2_f2",
                    "ds_ks2_f3",
                    "ds_ks_f3_g0",
                    "ds_ks_f3_g1",
                    "ds_kst_f2_g0",
                    "ds_kst_f3_g0",
                    "ds_kst_f2_g1",
                    "ds_kst_f3_g1",
                    "ds_moe_ks_gu_f2",
                    "ds_moe_ks_gu_f3",
                    "ds_moe_ks_down_f2",
                    "ds_moe_ks_down_f3",
                    "ds_wo_ks_f2",
                    "ds_wo_ks_f3",
                    "ds_moe_ks_tgu_f2",
                    "ds_moe_ks_tgu_f3",
                    "ds_moe_ks_tdown_f2",
                    "ds_moe_ks_tdown_f3",
                    "ds_attn_fd",
                    "ds_attn_fdm",
                ]
                .map(String::from)
                .into_iter()
                // the shared-expert-folded grouped kernels: past 48 KB at t 8 over d 5120
                .chain((0..8).map(|i| {
                    let kind = ["gu", "down"][i & 1];
                    format!("ds_moe_ks_{kind}_f{}_s{}", 2 + (i >> 1 & 1), 2 + (i >> 2))
                }))
                .chain((0..8).map(|i| {
                    let kind = ["gu", "down"][i & 1];
                    format!("ds_moe_kp_{kind}_f{}_s{}", 2 + (i >> 1 & 1), 2 + (i >> 2))
                })) {
                    let f = m.func(&n)?;
                    // the most dynamic memory the kernel's own static shared memory leaves
                    let mut st = 0i32;
                    let r = unsafe {
                        cudarc::driver::sys::cuFuncGetAttribute(
                            &mut st,
                            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            f,
                        )
                    };
                    gpu::check(r, "cuFuncGetAttribute")?;
                    let r = unsafe {
                        cudarc::driver::sys::cuFuncSetAttribute(
                            f,
                            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            mx - st.max(4352), // static: red + last
                        )
                    };
                    gpu::check(r, "cuFuncSetAttribute")?;
                }
                (mx - 4352) as usize
            },
            // the mma path needs sm_80 (an arch-less NVRTC build may target less)
            exact: std::env::var("DS_EXACT").is_ok_and(|v| v == "1") || arch(&m)? < 800,
            // slower than soft + pv on a 3090 (213 vs 164 us at t 6, k 646): opt-in
            attn_fuse: std::env::var("DS_ATTN_FUSE").is_ok_and(|v| v == "1"),
            attn_fd: std::env::var("DS_ATTN_FD").map_or(true, |v| v != "0"),
            attn_fdm: std::env::var("DS_ATTN_FDM").map_or(true, |v| v != "0"),
            attn_fd_s: std::env::var("DS_ATTN_FD_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            more: MORE
                .iter()
                .map(|&n| Ok((n, m.func(n)?)))
                .collect::<Result<_>>()?,
            _m: m,
        })
    }

    /// Launch kernel `name` (one of [`MORE`]).
    ///
    /// # Safety
    /// As [`gpu::launch`]: `args` must match the kernel's parameters.
    pub unsafe fn go(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        s: &Stream,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let f = *self
            .more
            .get(name)
            .with_context(|| format!("no kernel {name}"))?;
        Ok(super::obs::launch(f, grid, block, smem, s, args)?)
    }

    /// In place: -inf past each row's causal prefix (`causal`, ratio `r`) and where the u8
    /// candidate mask `cand` (0 for none) is 0. As forward.rs `indexer`.
    #[allow(clippy::too_many_arguments)]
    pub fn mask_scores(
        &self,
        s: &Stream,
        score: u64,
        cand: u64,
        t: usize,
        width: usize,
        causal: bool,
        r: usize,
    ) -> Result<()> {
        let (ti, wi, ci, ri) = (t as i32, width as i32, causal as i32, r as i32);
        unsafe {
            self.go(
                "ds_mask_scores",
                blocks(t * width, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![score, cand, ti, wi, ci, ri],
            )
        }
    }

    /// Candidate blocks into the u8 mask `cand` [t][width]. As [`super::forward::Model::candidates`].
    #[allow(clippy::too_many_arguments)]
    pub fn candidates(
        &self,
        s: &Stream,
        score: u64,
        cand: u64,
        t: usize,
        width: usize,
        causal: bool,
        r: usize,
        bs: usize,
        kb: usize,
    ) -> Result<()> {
        let nb = width.div_ceil(bs);
        let sc = DevBuf::alloc(t.max(1) * nb.max(1) * 4)?;
        let sel = DevBuf::alloc(t.max(1) * nb.max(1) * 4)?;
        let (ti, wi, ci, ri, bsi, kbi) = (
            t as i32,
            width as i32,
            causal as i32,
            r as i32,
            bs as i32,
            kb as i32,
        );
        unsafe {
            self.go(
                "ds_candidates",
                (t.max(1) as u32, 1, 1),
                SEL,
                0,
                s,
                tang_moe::args![score, sc.ptr, sel.ptr, cand, ti, wi, ci, ri, bsi, kbi],
            )?
        };
        // scratch drops here: drain first
        s.sync()?;
        Ok(())
    }

    /// Top-`k` per row of `score` [t][width] (ties to the lowest index), ascending, into
    /// `out[ti * stride ..]` as `idx + offset`, or -1 past the row's prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn topk(
        &self,
        s: &Stream,
        score: u64,
        out: u64,
        t: usize,
        width: usize,
        k: usize,
        stride: usize,
        offset: usize,
        causal: bool,
        r: usize,
    ) -> Result<()> {
        let scratch = DevBuf::alloc(t.max(1) * k.max(1) * 4)?;
        let (ti, wi, ki, si, oi, ci, ri) = (
            t as i32,
            width as i32,
            k as i32,
            stride as i32,
            offset as i32,
            causal as i32,
            r as i32,
        );
        unsafe {
            self.go(
                "ds_topk",
                (t.max(1) as u32, 1, 1),
                SEL,
                0,
                s,
                tang_moe::args![score, scratch.ptr, out, ti, wi, ki, si, oi, ci, ri],
            )?
        };
        s.sync()?;
        Ok(())
    }

    /// The MoE router: raw gate logits [t][ne] (f32) and bias [ne] -> ids/weights [t][k]
    /// (sqrt-softplus scores, top-k of score + bias ties to the lower id, weights normalized
    /// when `norm` and scaled by `scale`).
    #[allow(clippy::too_many_arguments)]
    pub fn router(
        &self,
        s: &Stream,
        raw: u64,
        bias: u64,
        ids: u64,
        wts: u64,
        t: usize,
        ne: usize,
        k: usize,
        norm: bool,
        scale: f32,
    ) -> Result<()> {
        let (ti, nei, ki, ni) = (t as i32, ne as i32, k as i32, norm as i32);
        unsafe {
            self.go(
                "ds_router",
                (t as u32, 1, 1),
                (128, 1, 1),
                (ne * 5) as u32,
                s,
                tang_moe::args![raw, bias, ids, wts, ti, nei, ki, ni, scale],
            )
        }
    }

    /// Whether [`Kernels::moe_gate`] takes this shape: `t` <= 8, `k` <= 16, `ne` <= 512.
    pub fn moe_gate_ok(w: &DsW, t: usize, ne: usize, k: usize) -> bool {
        (1..=8).contains(&t) && (1..=16).contains(&k) && ne <= 512 && w.rows as usize == ne
    }

    /// The decode router in one launch: the gate GEMV `raw` [t][ne] = `x` · `w`ᵀ (bit-identical
    /// to [`Kernels::linear`] unrounded), [`Kernels::router`] and [`Kernels::route`] (bit for
    /// bit), and with `elist` != 0 `ds_moe_plan`'s list of `cap` slots over `mask`. See
    /// [`Kernels::moe_gate_ok`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate(
        &self,
        s: &Stream,
        w: &DsW,
        x: u64,
        raw: u64,
        bias: u64,
        ids: u64,
        wts: u64,
        csr: (u64, u64, u64, u64),
        plan: (u64, u64, usize),
        t: usize,
        k: usize,
        norm: bool,
        scale: f32,
    ) -> Result<()> {
        let ne = w.rows as usize;
        if !Self::moe_gate_ok(w, t, ne, k) {
            bail!("moe_gate: t {t} ne {ne} k {k}");
        }
        let (counts, off, rows, rw) = csr;
        let (mask, elist, cap) = plan;
        let (wv, ci, ti, nei, ki, ni) =
            (*w, cap as i32, t as i32, ne as i32, k as i32, norm as i32);
        let cnt = self.gate_cnt.ptr;
        unsafe {
            self.go(
                "ds_moe_gate",
                (ne.div_ceil(4) as u32, 1, 1),
                (128, 1, 1),
                0,
                s,
                tang_moe::args![
                    wv, x, raw, bias, ids, wts, counts, off, rows, rw, mask, elist, ci, ti, nei,
                    ki, ni, scale, cnt
                ],
            )
        }
    }

    /// Router ids/weights [t][k] to per-expert CSR: counts [ne], off [ne + 1], rows and
    /// route weights [t * k], rows ascending within an expert.
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        s: &Stream,
        ids: u64,
        wts: u64,
        counts: u64,
        off: u64,
        rows: u64,
        rw: u64,
        t: usize,
        k: usize,
        ne: usize,
    ) -> Result<()> {
        if ne > 512 {
            bail!("route: ne {ne} > 512");
        }
        let (ti, ki, nei) = (t as i32, k as i32, ne as i32);
        unsafe {
            self.go(
                "ds_route",
                (1, 1, 1),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![ids, wts, counts, off, rows, rw, ti, ki, nei],
            )
        }
    }

    /// [`Kernels::route`], `ds_moe_plan` (`elist` [cap] over `mask`) and the K-split quantization
    /// of `x` [t][d] that [`Kernels::moe_gu_sh`] then takes as is (`pre`), in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn route_plan_q(
        &self,
        s: &Stream,
        (ids, wts): (u64, u64),
        (counts, off, rows, rw): (u64, u64, u64, u64),
        (mask, elist, cap): (u64, u64, usize),
        x: u64,
        (t, k, ne, d): (usize, usize, usize, usize),
    ) -> Result<()> {
        let (codes, scales) = (self.ks_x.ptr, self.ks_x.ptr + (8 * 16384) as u64);
        let ng = t * d / 32;
        let (ti, ki, nei, ci, ngi) = (t as i32, k as i32, ne as i32, cap as i32, ng as i32);
        unsafe {
            self.go(
                "ds_route_plan_q",
                ((1 + ng.div_ceil(8)) as u32, 1, 1),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![
                    ids, wts, counts, off, rows, rw, ti, ki, nei, mask, elist, ci, x, codes,
                    scales, ngi
                ],
            )
        }
    }

    /// `y = x · wᵀ` for `t` rows (`x` already activation-quantized when the weight is).
    pub fn linear(&self, s: &Stream, w: &DsW, x: u64, y: u64, t: usize, round: bool) -> Result<()> {
        let (wv, ti, r) = (*w, t as i32, round as i32);
        if (2..=8).contains(&t) {
            return unsafe {
                self.go(
                    "ds_linear_t8",
                    blocks(w.rows as usize * 32, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![wv, x, y, ti, r],
                )
            };
        }
        unsafe {
            super::obs::launch(
                self.linear,
                blocks(t * w.rows as usize * 32, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![wv, x, y, ti, r],
            )?
        }
        Ok(())
    }

    /// The vocab head, `y` [t][rows] = `x` [t][cols] · `w`ᵀ (no rounding): `ds_head_bf16` for
    /// bf16 weights in 8-row chunks, so each row's logits are the same at every `t`;
    /// [`Kernels::linear`] otherwise.
    pub fn head(&self, s: &Stream, w: &DsW, x: u64, y: u64, t: usize) -> Result<()> {
        if w.fmt != 1 || w.cols % 8 != 0 || x % 16 != 0 {
            return self.linear(s, w, x, y, t, false);
        }
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        for i0 in (0..t).step_by(8) {
            let ti = (t - i0).min(8) as i32;
            let (xp, yp) = (x + (i0 * cols * 4) as u64, y + (i0 * rows * 4) as u64);
            unsafe {
                self.go(
                    "ds_head_bf16",
                    blocks(rows.div_ceil(4) * 32, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![w.w, w.rows, w.cols, xp, yp, ti],
                )?
            };
        }
        Ok(())
    }

    /// `y = x · wᵀ` for bf16 `w` and t <= 8 (no rounding) on `ds_gemv_bf16_r1`, a warp per row,
    /// 16-byte weight loads: [`Kernels::linear`]'s values up to summation order (relaxed).
    /// Falls back to [`Kernels::linear`] off its shapes or under `DS_EXACT`.
    pub fn gemv_bf16(&self, s: &Stream, w: &DsW, x: u64, y: u64, t: usize) -> Result<()> {
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        if self.exact
            || w.fmt != 1
            || cols % 1024 != 0
            || x % 16 != 0
            || !(1..=8).contains(&t)
            || rows.div_ceil(8) > GEMV_CNT
            || cols / 1024 * t * rows > GEMV_PART
        {
            return self.linear(s, w, x, y, t, false);
        }
        let ti = t as i32;
        let (part, cnt) = (self.gemv_part.ptr, self.gemv_cnt.ptr);
        unsafe {
            self.go(
                "ds_gemv_bf16_r1",
                (rows.div_ceil(8) as u32, (cols / 1024) as u32, 1),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![w.w, w.rows, w.cols, x, y, ti, part, cnt],
            )
        }
    }

    /// Whether [`Kernels::linear_fast`] takes `w`: fp8 or fp4 with whole 32-column groups.
    pub fn fast_ok(w: &DsW) -> bool {
        w.fmt >= 2 && w.cols % 32 == 0 && w.cols > 0
    }

    /// `linear` for fp8/fp4 weights ([`Kernels::fast_ok`]), bit-identical to [`Kernels::linear`]
    /// (after [`Kernels::fp8_qdq`] when `qdq`), at near memory bandwidth for small `t`: weights
    /// are read once per 8 token rows. With `w3` (same shape and format as `w`), the fused
    /// SwiGLU: `y` = `ds_swiglu(bf(x·w1ᵀ), bf(x·w3ᵀ), rw, lim)`, `rw` per-token weights or 0.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_fast(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        x: u64,
        y: u64,
        t: usize,
        round: bool,
        qdq: bool,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        if !Self::fast_ok(w) {
            bail!("linear_fast: fmt {} cols {}", w.fmt, w.cols);
        }
        if let Some(w3) = w3 {
            if (w3.fmt, w3.rows, w3.cols) != (w.fmt, w.rows, w.cols) {
                bail!("linear_fast: w1/w3 mismatch");
            }
        }
        if !self.exact {
            return self.linear_ks(s, w, w3, x, y, t, round, qdq, rw, lim);
        }
        self.linear_exact(s, w, w3, x, y, t, round, qdq, rw, lim)
    }

    /// The bit-exact `ds_lin_*` kernels behind [`Kernels::linear_fast`] under `DS_EXACT=1`.
    #[allow(clippy::too_many_arguments)]
    fn linear_exact(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        x: u64,
        y: u64,
        t: usize,
        round: bool,
        qdq: bool,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        let gu = w3.is_some() as usize;
        let per_warp = if gu == 1 { 2 } else { 4 };
        let grid = blocks(rows.div_ceil(per_warp), 4);
        let (r, q) = (round as i32, qdq as i32);
        let mut t0 = 0;
        while t0 < t {
            let n = (t - t0).min(8);
            let tm = match n {
                1 => 0,
                2 => 1,
                3 | 4 => 2,
                _ => 3,
            };
            let f = self.fast[w.fmt as usize - 2][tm][gu];
            let (wv, w3v) = (*w, *w3.unwrap_or(w));
            let (xo, yo, ni) = (
                x + (t0 * cols * 4) as u64,
                y + (t0 * rows * 4) as u64,
                n as i32,
            );
            let rwo = if rw == 0 { 0 } else { rw + (t0 * 4) as u64 };
            unsafe {
                super::obs::launch(
                    f,
                    grid,
                    (128, 1, 1),
                    0,
                    s,
                    tang_moe::args![wv, w3v, xo, yo, ni, r, q, rwo, lim],
                )?
            }
            t0 += n;
        }
        Ok(())
    }

    /// The K-split GEMV behind [`Kernels::linear_fast`] (unless `DS_EXACT=1`): same contract,
    /// outputs within one bf16 ulp of the exact kernels in rare elements.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_ks(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        x: u64,
        y: u64,
        t: usize,
        round: bool,
        qdq: bool,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        if !Self::fast_ok(w) {
            bail!("linear_ks: fmt {} cols {}", w.fmt, w.cols);
        }
        if let Some(w3) = w3 {
            if (w3.fmt, w3.rows, w3.cols) != (w.fmt, w.rows, w.cols) {
                bail!("linear_ks: w1/w3 mismatch");
            }
        }
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        let pitch = cols.div_ceil(128) * 128 + 32;
        let nch = cols / 32;
        if 8 * pitch + 8 * nch * 4 > self.smem_max || cols > 16384 {
            // too wide to stage: the exact kernels
            return self.linear_exact(s, w, w3, x, y, t, round, qdq, rw, lim);
        }
        let _ = qdq; // quantized either way (idempotent on a quantized activation)
        if t > 8 {
            return self.linear_kst(s, w, w3, x, y, t, round as i32, rw, lim);
        }
        let (codes, scales) = (self.ks_x.ptr, self.ks_x.ptr + (8 * 16384) as u64);
        let mut t0 = 0;
        while t0 < t {
            let n = (t - t0).min(8);
            let ng = (n * nch) as i32;
            let xo = x + (t0 * cols * 4) as u64;
            unsafe {
                super::obs::launch(
                    self.ks_quant,
                    blocks(n * nch * 32, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![xo, codes, scales, ng],
                )?
            }
            let yo = y + (t0 * rows * 4) as u64;
            let rwo = if rw == 0 { 0 } else { rw + (t0 * 4) as u64 };
            self.ks_gemv(s, w, w3, codes, scales, yo, n, round, rwo, lim)?;
            t0 += n;
        }
        Ok(())
    }

    /// Whether [`Kernels::linear_ks`] runs `w` on the K-split GEMV (rather than falling back to
    /// the exact kernels), so its activation needs no separate `fp8_qdq`.
    pub fn ks_ok(&self, w: &DsW) -> bool {
        if self.exact || !Self::fast_ok(w) {
            return false;
        }
        let cols = w.cols as usize;
        let pitch = cols.div_ceil(128) * 128 + 32;
        8 * pitch + 8 * (cols / 32) * 4 <= self.smem_max && cols <= 16384
    }

    /// `ks_x`'s (codes, scales): [`Kernels::linear_ks`]'s quantized activation.
    pub fn ks_x_ptrs(&self) -> (u64, u64) {
        (self.ks_x.ptr, self.ks_x.ptr + (8 * 16384) as u64)
    }

    /// `wo_x`'s (codes, scales): a second quantized-activation buffer (8 × 32768).
    pub fn wo_x_ptrs(&self) -> (u64, u64) {
        (self.wo_x.ptr, self.wo_x.ptr + (8 * WO_X_COLS) as u64)
    }

    /// `ds_ks_quant` of `n` values at `x` (a multiple of 32) into (codes, scales).
    pub fn ks_quant_to(&self, s: &Stream, x: u64, codes: u64, scales: u64, n: usize) -> Result<()> {
        let ng = (n / 32) as i32;
        unsafe {
            super::obs::launch(
                self.ks_quant,
                blocks(n, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![x, codes, scales, ng],
            )?;
        }
        Ok(())
    }

    /// The K-split GEMV of [`Kernels::linear_ks`] on an already quantized activation (codes
    /// [n][cols], scales [n][cols / 32]), `n` <= 8 tokens. [`Kernels::ks_ok`] first.
    #[allow(clippy::too_many_arguments)]
    pub fn ks_gemv(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        codes: u64,
        scales: u64,
        y: u64,
        n: usize,
        round: bool,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        let gu = w3.is_some() as usize;
        let out = if gu == 1 { 8 } else { 16 };
        let pitch = cols.div_ceil(128) * 128 + 32;
        let nch = cols / 32;
        let (sp, br, mut sb) = ks_split(rows.div_ceil(out), nch, self.sms);
        if sb * n * rows * (1 + gu) > KS_PART || br > KS_CNT {
            sb = 1;
        }
        let f = self.ks[w.fmt as usize - 2][gu];
        let smem = n * pitch + n * nch * 4;
        let mut per_sm = 0i32;
        unsafe {
            gpu::check(
                cudarc::driver::sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut per_sm,
                    f,
                    256,
                    smem,
                ),
                "occupancy",
            )?
        };
        let grid = (br * sb).min(self.sms * per_sm.max(1) as usize);
        let (wv, w3v) = (*w, *w3.unwrap_or(w));
        let (ni, r) = (n as i32, round as i32);
        let (si, sbi, bri) = (sp as i32, sb as i32, br as i32);
        let (part, cnt) = (self.ks_part.ptr, self.ks_cnt.ptr);
        unsafe {
            super::obs::launch(
                f,
                (grid as u32, 1, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![wv, w3v, codes, scales, y, ni, r, rw, lim, si, sbi, bri, part, cnt],
            )?;
        }
        Ok(())
    }

    /// [`Kernels::ks_gemv`] of `w` into `y` and `wb` into `yb` on the same quantized activation
    /// in one launch (`ds_ks2_*`), each bit for bit its own `ks_gemv` (no `w3`, no `rw`).
    /// Falls back to two launches when the formats or columns differ or the scratch is short.
    #[allow(clippy::too_many_arguments)]
    pub fn ks_gemv2(
        &self,
        s: &Stream,
        w: &DsW,
        wb: &DsW,
        codes: u64,
        scales: u64,
        y: u64,
        yb: u64,
        n: usize,
        round: bool,
    ) -> Result<()> {
        let split = |w: &DsW| {
            let (rows, nch) = (w.rows as usize, w.cols as usize / 32);
            let (sp, br, mut sb) = ks_split(rows.div_ceil(16), nch, self.sms);
            if sb * n * rows > KS_PART || br > KS_CNT {
                sb = 1;
            }
            (sp, br, sb)
        };
        let (a, b) = (split(w), split(wb));
        let pa = a.2 * n * w.rows as usize;
        if w.fmt != wb.fmt
            || w.cols != wb.cols
            || std::env::var("DS_KS2").is_ok_and(|v| v == "0")
            || pa + b.2 * n * wb.rows as usize > KS_PART
            || a.1 + b.1 > KS_CNT
        {
            self.ks_gemv(s, w, None, codes, scales, y, n, round, 0, 0.0)?;
            return self.ks_gemv(s, wb, None, codes, scales, yb, n, round, 0, 0.0);
        }
        let f = self.ks2[w.fmt as usize - 2];
        let cols = w.cols as usize;
        let pitch = cols.div_ceil(128) * 128 + 32;
        let smem = n * pitch + n * (cols / 32) * 4;
        let mut per_sm = 0i32;
        unsafe {
            gpu::check(
                cudarc::driver::sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut per_sm,
                    f,
                    256,
                    smem,
                ),
                "occupancy",
            )?
        };
        let items = a.1 * a.2 + b.1 * b.2;
        let grid = items.min(self.sms * per_sm.max(1) as usize);
        let (ni, r) = (n as i32, round as i32);
        let (sa, sba, bra) = (a.0 as i32, a.2 as i32, a.1 as i32);
        let (sb_, sbb, brb) = (b.0 as i32, b.2 as i32, b.1 as i32);
        let (part, cnt) = (self.ks_part.ptr, self.ks_cnt.ptr);
        let (partb, cntb) = (part + (pa * 4) as u64, cnt + (a.1 * 4) as u64);
        let (wv, wbv) = (*w, *wb);
        unsafe {
            super::obs::launch(
                f,
                (grid as u32, 1, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![
                    wv, wbv, codes, scales, y, yb, ni, r, sa, sba, bra, sb_, sbb, brb, part, partb,
                    cnt, cntb
                ],
            )?;
        }
        Ok(())
    }

    /// Whether [`Kernels::wo_ks`] takes `t` tokens of `g` groups × `w.cols` columns.
    pub fn wo_ks_ok(&self, w: &DsW, t: usize, g: usize, ol: usize) -> bool {
        let gd = w.cols as usize;
        !self.exact
            && Self::fast_ok(w)
            && ol % 32 == 0
            && (1..=8).contains(&t)
            && g * gd <= WO_X_COLS
            && t * (gd.div_ceil(128) * 128 + 32) + t * (gd / 32) * 4 <= self.smem_max
    }

    /// wo_a, block-diagonal over `g` groups of `ol` rows (`w` all `g * ol` rows × gd cols),
    /// on the K-split path in two launches (quantize + one grouped GEMV, see `ds_wo_ks`): `x`
    /// [t][g * gd] (rope'd, unquantized: the quantize is ds_fp8_qdq's) into `y` [t][g * ol],
    /// bf16-rounded. Relaxed bar (the K-split linears'); [`Kernels::wo_ks_ok`] first.
    #[allow(clippy::too_many_arguments)]
    pub fn wo_ks(
        &self,
        s: &Stream,
        w: &DsW,
        x: u64,
        y: u64,
        t: usize,
        g: usize,
        ol: usize,
    ) -> Result<()> {
        self.wo_ks_inner(s, w, x, y, t, g, ol, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn wo_ks_inner(
        &self,
        s: &Stream,
        w: &DsW,
        x: u64,
        y: u64,
        t: usize,
        g: usize,
        ol: usize,
        quant: bool,
    ) -> Result<()> {
        if !self.wo_ks_ok(w, t, g, ol) {
            bail!("wo_ks: shape t {t} g {g} ol {ol} cols {}", w.cols);
        }
        let gd = w.cols as usize;
        let (codes, scales) = (self.wo_x.ptr, self.wo_x.ptr + (8 * WO_X_COLS) as u64);
        let ng = (t * g * gd / 32) as i32;
        if quant {
            unsafe {
                super::obs::launch(
                    self.ks_quant,
                    blocks(t * g * gd, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![x, codes, scales, ng],
                )?
            }
        }
        let (sp, nbr, gx) = self.moe_ks_grid(ol, 16, gd);
        let (ti, oli, si, nbri) = (t as i32, ol as i32, sp as i32, nbr as i32);
        let smem = t * (gd.div_ceil(128) * 128 + 32) + t * (gd / 32) * 4;
        unsafe {
            super::obs::launch(
                self.wo_ks[w.fmt as usize - 2],
                (gx as u32, g as u32, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![*w, codes, scales, y, ti, oli, si, nbri],
            )?;
        }
        Ok(())
    }

    /// [`Kernels::wo_ks`] on an activation already quantized into its buffer (by the
    /// attention's merge, [`AttnRope::wo_quant`]): the grouped GEMV alone.
    pub fn wo_ks_pre(
        &self,
        s: &Stream,
        w: &DsW,
        y: u64,
        t: usize,
        g: usize,
        ol: usize,
    ) -> Result<()> {
        self.wo_ks_inner(s, w, 0, y, t, g, ol, false)
    }

    /// [`Kernels::linear_ks`] for t > 8 rows in one launch (`ds_kst_*`): the activation
    /// quantized once, a block per (row group, 8-row chunk), chunks adjacent so each row
    /// group's weights come from L2 after the first chunk.
    #[allow(clippy::too_many_arguments)]
    fn linear_kst(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        x: u64,
        y: u64,
        t: usize,
        r: i32,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        let gu = w3.is_some() as usize;
        let out = if gu == 1 { 8 } else { 16 };
        let nch = cols / 32;
        let need = t * cols + t * nch * 4;
        let mut big = self.ks_big.lock().unwrap();
        if big.as_ref().is_none_or(|(n, _)| *n < need) {
            s.sync()?; // the old buffer may be in flight
            *big = Some((need, DevBuf::alloc(need)?));
        }
        let buf = big.as_ref().unwrap().1.ptr;
        let (codes, scales) = (buf, buf + (t * cols) as u64);
        let ng = (t * nch) as i32;
        unsafe {
            super::obs::launch(
                self.ks_quant,
                blocks(t * nch * 32, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![x, codes, scales, ng],
            )?
        }
        // S = 1: a block's 8 warps are 8 row tiles over all of K (most weight reuse per
        // staged chunk); the many chunks give the parallelism
        let sp = 1usize;
        let nbr = rows.div_ceil(out).div_ceil(8 / sp);
        let nc = t.div_ceil(8);
        let smem = 8 * (cols.div_ceil(128) * 128 + 32) + 8 * nch * 4;
        let (wv, w3v) = (*w, *w3.unwrap_or(w));
        let (ti, si, nbri) = (t as i32, sp as i32, nbr as i32);
        let name = [
            ["ds_kst_f2_g0", "ds_kst_f2_g1"],
            ["ds_kst_f3_g0", "ds_kst_f3_g1"],
        ][w.fmt as usize - 2][gu];
        unsafe {
            self.go(
                name,
                ((nbr * nc) as u32, 1, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![wv, w3v, codes, scales, y, ti, r, rw, lim, si, nbri],
            )
        }
    }

    /// Whether the grouped K-split kernels take `t` <= 8 tokens per expert over `cols`
    /// columns, with `n` activation rows to quantize into `ks_x`.
    fn moe_ks_ok(&self, t: usize, cols: usize, n: usize) -> bool {
        let pitch = cols.div_ceil(128) * 128 + 32;
        (1..=8).contains(&t)
            && cols % 32 == 0
            && cols > 0
            && n * cols <= 8 * 16384
            && t * pitch + t * (cols / 32) * 4 <= self.smem_max
    }

    /// Whether [`Kernels::moe_gu`] / [`Kernels::moe_down`] take the K-split path for `t`
    /// tokens an expert over `cols`, `n` activation rows (they then quantize the activation
    /// themselves: no fp8 qdq pass needed before).
    pub fn moe_ks_takes(&self, t: usize, cols: usize, n: usize) -> bool {
        !self.exact && self.moe_ks_ok(t, cols, n)
    }

    /// Whether [`Kernels::linear_fast`] of `w` runs the K-split GEMV, which quantizes its
    /// activation itself (so the caller's fp8 qdq pass is redundant).
    pub fn ks_takes(&self, w: &DsW) -> bool {
        let cols = w.cols as usize;
        let pitch = cols.div_ceil(128) * 128 + 32;
        !self.exact
            && Self::fast_ok(w)
            && 8 * pitch + 8 * (cols / 32) * 4 <= self.smem_max
            && cols <= 16384
    }

    /// `ds_ks_quant` of `n` rows × `cols` at `x` into `ks_x`: (codes, scales).
    fn ks_quantize(&self, s: &Stream, x: u64, n: usize, cols: usize) -> Result<(u64, u64)> {
        let (codes, scales) = (self.ks_x.ptr, self.ks_x.ptr + (8 * 16384) as u64);
        let ng = (n * cols / 32) as i32;
        unsafe {
            super::obs::launch(
                self.ks_quant,
                blocks(n * cols, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![x, codes, scales, ng],
            )?
        }
        Ok((codes, scales))
    }

    /// The grouped K-split shape for one expert's `rows` × `cols` (`out` rows a warp tile):
    /// S K-slices a block (≥ 8 groups a warp), its nbr row groups, and the blocks per slot,
    /// each striding over ~4 row groups so the staged activation amortizes over weights.
    fn moe_ks_grid(&self, rows: usize, out: usize, cols: usize) -> (usize, usize, usize) {
        let nch = cols / 32;
        let mut sp = 1;
        while sp < 8 && nch >= sp * 2 * 8 {
            sp *= 2;
        }
        if self.moe_s > 0 {
            sp = self.moe_s;
        }
        let nbr = rows.div_ceil(out).div_ceil(8 / sp).max(1);
        (sp, nbr, nbr.div_ceil(self.moe_items))
    }

    /// Grouped MoE gate/up + SwiGLU for every routed expert of a layer in one launch (see
    /// `DS_MOE` in kernels.cu): `w1s`/`w3s` device arrays of [`DsW`] per expert (format
    /// `fmt`, `inter` rows), `elist` [slots] from `ds_moe_plan`, the CSR of [`Kernels::route`],
    /// `xq` the fp8-quantized activation [t][d], into `h` [t * k][inter] at CSR positions.
    /// Bit-identical to the per-expert `ds_linear_csr` + `ds_swiglu` loop. `t` <= 8.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gu(
        &self,
        s: &Stream,
        fmt: i32,
        w1s: u64,
        w3s: u64,
        elist: u64,
        off: u64,
        rows: u64,
        rw: u64,
        xq: u64,
        h: u64,
        t: usize,
        slots: usize,
        inter: usize,
        d: usize,
        lim: f32,
    ) -> Result<()> {
        if !self.exact && self.moe_ks_ok(t, d, t) {
            let f = self.moe_ks[fmt as usize - 2][0];
            let (codes, scales) = self.ks_quantize(s, xq, t, d)?;
            let (sp, nbr, gx) = self.moe_ks_grid(inter, 8, d);
            let (si, nbri) = (sp as i32, nbr as i32);
            unsafe {
                super::obs::launch(
                    f,
                    (gx as u32, slots.max(1) as u32, 1),
                    (256, 1, 1),
                    (t * (d.div_ceil(128) * 128 + 32) + t * (d / 32) * 4) as u32,
                    s,
                    tang_moe::args![
                        w1s, w3s, elist, off, rows, rw, codes, scales, h, lim, si, nbri
                    ],
                )?
            }
            return Ok(());
        }
        let f = self.moe[fmt as usize - 2][tm_index(t)?][0];
        let grid = (inter.div_ceil(2).div_ceil(4) as u32, slots.max(1) as u32, 1);
        unsafe {
            super::obs::launch(
                f,
                grid,
                (128, 1, 1),
                0,
                s,
                tang_moe::args![w1s, w3s, elist, off, rows, rw, xq, h, lim],
            )?
        }
        Ok(())
    }

    /// Grouped MoE down projection: `hq` [t * k][inter] (quantized `h`) to per-slot partials
    /// `part` [t * k][d], bf16-rounded, for `ds_moe_reduce`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down(
        &self,
        s: &Stream,
        fmt: i32,
        w2s: u64,
        elist: u64,
        off: u64,
        hq: u64,
        part: u64,
        t: usize,
        slots: usize,
        d: usize,
        inter: usize,
        topk: usize,
    ) -> Result<()> {
        let n = t * topk;
        if !self.exact && self.moe_ks_ok(t, inter, n) {
            let f = self.moe_ks[fmt as usize - 2][1];
            let (codes, scales) = self.ks_quantize(s, hq, n, inter)?;
            let (sp, nbr, gx) = self.moe_ks_grid(d, 16, inter);
            let (si, nbri) = (sp as i32, nbr as i32);
            unsafe {
                super::obs::launch(
                    f,
                    (gx as u32, slots.max(1) as u32, 1),
                    (256, 1, 1),
                    (t * (inter.div_ceil(128) * 128 + 32) + t * (inter / 32) * 4) as u32,
                    s,
                    tang_moe::args![w2s, elist, off, codes, scales, part, si, nbri],
                )?
            }
            return Ok(());
        }
        let f = self.moe[fmt as usize - 2][tm_index(t)?][1];
        let grid = (d.div_ceil(4).div_ceil(4) as u32, slots.max(1) as u32, 1);
        unsafe {
            super::obs::launch(
                f,
                grid,
                (128, 1, 1),
                0,
                s,
                tang_moe::args![w2s, elist, off, hq, part],
            )?
        }
        Ok(())
    }

    /// Whether [`Kernels::moe_gu_sh`] / [`Kernels::moe_down_sh`] take `t` tokens with the
    /// shared expert `sh` (w1, w3, w2) folded in as one more slot of a layer whose routed
    /// experts are `fmt`, `inter` × `d`: the K-split path only (never under `DS_EXACT`), the
    /// shared expert of the routed shape, and `t * (k + 1)` hidden rows fitting `ks_x`.
    pub fn moe_sh_ok(
        &self,
        sh: [&DsW; 3],
        fmt: i32,
        t: usize,
        k: usize,
        d: usize,
        inter: usize,
    ) -> bool {
        let [a, b, c] = sh;
        let n = t * (k + 1);
        !self.exact
            && (2..=3).contains(&fmt)
            && Self::fast_ok(a)
            && Self::fast_ok(c)
            && (a.rows as usize, a.cols as usize) == (inter, d)
            && (b.fmt, b.rows, b.cols) == (a.fmt, a.rows, a.cols)
            && (c.fmt, c.rows as usize, c.cols as usize) == (a.fmt, d, inter)
            && self.moe_ks_ok(t, d, t)
            && self.moe_ks_ok(t, inter, n)
    }

    /// The persistent MoE launch's grid: as many blocks as fit on the GPU at once.
    fn kp_grid(&self, f: Func, smem: usize) -> Result<u32> {
        let mut per_sm = 0i32;
        unsafe {
            gpu::check(
                cudarc::driver::sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut per_sm,
                    f,
                    256,
                    smem,
                ),
                "occupancy",
            )?
        };
        Ok((self.sms * per_sm.max(1) as usize) as u32)
    }

    /// [`Kernels::moe_gu`] on the K-split path with the shared expert (`s1`, `s3`) as one more
    /// slot when `on` (every token at weight 1, into `h` rows `n = t * k` ..): one launch for
    /// the routed and shared gate/up. See [`Kernels::moe_sh_ok`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gu_sh(
        &self,
        s: &Stream,
        fmt: i32,
        (w1s, w3s, elist): (u64, u64, u64),
        (off, rows, rw): (u64, u64, u64),
        (s1, s3, on): (&DsW, &DsW, bool),
        xq: u64,
        h: u64,
        (t, n, slots, inter, d): (usize, usize, usize, usize, usize),
        lim: f32,
    ) -> Result<()> {
        let f = self.moe_ks_sh[fmt as usize - 2][s1.fmt as usize - 2][0];
        // xq == 0: route_plan_q already quantized the activation into ks_x
        let (codes, scales) = if xq == 0 {
            (self.ks_x.ptr, self.ks_x.ptr + (8 * 16384) as u64)
        } else {
            self.ks_quantize(s, xq, t, d)?
        };
        let (sp, nbr, gx) = self.moe_ks_grid(inter, 8, d);
        let (si, nbri, ts, ni, s1, s3) = (
            sp as i32,
            nbr as i32,
            (t * on as usize) as i32,
            n as i32,
            *s1,
            *s3,
        );
        let gy = slots + on as usize;
        let smem = t * (d.div_ceil(128) * 128 + 32) + t * (d / 32) * 4;
        if let (Some(kp), true) = (&self.moe_kp, slots <= 64) {
            let f = kp[fmt as usize - 2][s1.fmt as usize - 2][0];
            let (g, ci) = (self.kp_grid(f, smem)?, slots as i32);
            unsafe {
                super::obs::launch(
                    f,
                    (g, 1, 1),
                    (256, 1, 1),
                    smem as u32,
                    s,
                    tang_moe::args![
                        w1s, w3s, elist, off, rows, rw, codes, scales, h, lim, si, nbri, s1, s3,
                        ts, ni, ci
                    ],
                )?
            }
            return Ok(());
        }
        unsafe {
            super::obs::launch(
                f,
                (gx as u32, gy.max(1) as u32, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![
                    w1s, w3s, elist, off, rows, rw, codes, scales, h, lim, si, nbri, s1, s3, ts, ni
                ],
            )?
        }
        Ok(())
    }

    /// [`Kernels::moe_down`] on the K-split path with the shared expert's `s2` as one more
    /// slot when `on`: `hq` rows `n = t * k` .. `n + t` (quantized with the routed rows) into
    /// `part` rows `n` .., for `ds_moe_reduce_sh`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_sh(
        &self,
        s: &Stream,
        fmt: i32,
        (w2s, elist, off): (u64, u64, u64),
        (s2, on): (&DsW, bool),
        hq: u64,
        part: u64,
        (t, n, slots, d, inter): (usize, usize, usize, usize, usize),
    ) -> Result<()> {
        let f = self.moe_ks_sh[fmt as usize - 2][s2.fmt as usize - 2][1];
        let rows = n + if on { t } else { 0 };
        let (codes, scales) = self.ks_quantize(s, hq, rows, inter)?;
        let (sp, nbr, gx) = self.moe_ks_grid(d, 16, inter);
        let (si, nbri, ts, ni, s2) = (
            sp as i32,
            nbr as i32,
            (t * on as usize) as i32,
            n as i32,
            *s2,
        );
        let gy = slots + on as usize;
        let smem = t * (inter.div_ceil(128) * 128 + 32) + t * (inter / 32) * 4;
        if let (Some(kp), true) = (&self.moe_kp, slots <= 64) {
            let f = kp[fmt as usize - 2][s2.fmt as usize - 2][1];
            let (g, ci) = (self.kp_grid(f, smem)?, slots as i32);
            unsafe {
                super::obs::launch(
                    f,
                    (g, 1, 1),
                    (256, 1, 1),
                    smem as u32,
                    s,
                    tang_moe::args![w2s, elist, off, codes, scales, part, si, nbri, s2, ts, ni, ci],
                )?
            }
            return Ok(());
        }
        unsafe {
            super::obs::launch(
                f,
                (gx as u32, gy.max(1) as u32, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![w2s, elist, off, codes, scales, part, si, nbri, s2, ts, ni],
            )?
        }
        Ok(())
    }

    /// Upper bound on the 8-row tiles of `slots` expert segments holding `n` rows in all.
    pub fn moe_tile_cap(n: usize, slots: usize) -> usize {
        n / 8 + slots
    }

    /// Whether [`Kernels::moe_tiled`] takes the K-split kernels over `d` / `inter`: they
    /// quantize their activations themselves, so no fp8 qdq pass is needed before.
    pub fn moe_tiled_ks(&self, d: usize, inter: usize) -> bool {
        let fits = |cols: usize| {
            let pitch = cols.div_ceil(128) * 128 + 32;
            cols % 32 == 0 && 8 * pitch + 8 * (cols / 32) * 4 <= self.smem_max
        };
        !self.exact && fits(d) && fits(inter)
    }

    /// The grouped MoE for any rows per expert (prefill): `ds_moe_plan_tiles` of `elist` into
    /// `tiles` [`Kernels::moe_tile_cap`], then gate/up + SwiGLU into `h` [n][inter], its fp8
    /// quantization, and down into `part` [n][d] (`n` = t * topk), per 8-row tile of each
    /// segment. `codes` holds n * max(d, inter) bytes and `scales` n * max(d, inter) / 32
    /// floats (the K-split activation). The K-split kernels unless exact (or too wide to
    /// stage); per row, the t <= 8 kernels' results.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tiled(
        &self,
        s: &Stream,
        fmt: i32,
        (w1s, w3s, w2s): (u64, u64, u64),
        (elist, slots, tiles): (u64, usize, u64),
        (off, rows, rw): (u64, u64, u64),
        (xq, h, part): (u64, u64, u64),
        (codes, scales): (u64, u64),
        (t, topk, d, inter): (usize, usize, usize, usize),
        lim: f32,
    ) -> Result<()> {
        let n = t * topk;
        let cap = Self::moe_tile_cap(n, slots);
        if cap > 65535 {
            bail!("moe_tiled: {cap} tiles");
        }
        let (si, ci) = (slots as i32, cap as i32);
        unsafe {
            self.go(
                "ds_moe_plan_tiles",
                (1, 1, 1),
                (32, 1, 1),
                0,
                s,
                tang_moe::args![elist, off, tiles, si, ci],
            )?
        };
        let f = fmt;
        let ks = self.moe_tiled_ks(d, inter);
        let quant = |x: u64, rows: usize, cols: usize| -> Result<()> {
            let ng = (rows * cols / 32) as i32;
            unsafe {
                super::obs::launch(
                    self.ks_quant,
                    blocks(rows * cols, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![x, codes, scales, ng],
                )?
            }
            Ok(())
        };
        let smem = |cols: usize| (8 * (cols.div_ceil(128) * 128 + 32) + 8 * (cols / 32) * 4) as u32;
        if ks {
            quant(xq, t, d)?;
            let (sp, nbr, gx) = self.moe_ks_grid(inter, 8, d);
            let (spi, nbri) = (sp as i32, nbr as i32);
            unsafe {
                self.go(
                    ["ds_moe_ks_tgu_f2", "ds_moe_ks_tgu_f3"][f as usize - 2],
                    (gx as u32, cap as u32, 1),
                    (256, 1, 1),
                    smem(d),
                    s,
                    tang_moe::args![
                        w1s, w3s, tiles, off, rows, rw, codes, scales, h, lim, spi, nbri
                    ],
                )?
            };
        } else {
            unsafe {
                self.go(
                    ["ds_moe_tgu_f2", "ds_moe_tgu_f3"][f as usize - 2],
                    (inter.div_ceil(2).div_ceil(4) as u32, cap as u32, 1),
                    (128, 1, 1),
                    0,
                    s,
                    tang_moe::args![w1s, w3s, tiles, off, rows, rw, xq, h, lim],
                )?
            };
        }
        // h stays fp8-quantized (the contract: down's input, read back by callers)
        self.fp8_qdq(s, h, n * inter)?;
        if ks {
            quant(h, n, inter)?;
            let (sp, nbr, gx) = self.moe_ks_grid(d, 16, inter);
            let (spi, nbri) = (sp as i32, nbr as i32);
            unsafe {
                self.go(
                    ["ds_moe_ks_tdown_f2", "ds_moe_ks_tdown_f3"][f as usize - 2],
                    (gx as u32, cap as u32, 1),
                    (256, 1, 1),
                    smem(inter),
                    s,
                    tang_moe::args![w2s, tiles, off, codes, scales, part, spi, nbri],
                )?
            };
        } else {
            unsafe {
                self.go(
                    ["ds_moe_tdown_f2", "ds_moe_tdown_f3"][f as usize - 2],
                    (d.div_ceil(4).div_ceil(4) as u32, cap as u32, 1),
                    (128, 1, 1),
                    0,
                    s,
                    tang_moe::args![w2s, tiles, off, h, part],
                )?
            };
        }
        Ok(())
    }

    pub fn fp8_qdq(&self, s: &Stream, v: u64, n: usize) -> Result<()> {
        let g = (n / 32) as i32;
        unsafe {
            super::obs::launch(
                self.fp8_qdq,
                blocks(n, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![v, g],
            )?
        }
        Ok(())
    }

    pub fn fp4_qdq(
        &self,
        s: &Stream,
        v: u64,
        n: usize,
        block: usize,
        e4m3_scale: bool,
    ) -> Result<()> {
        let (g, b, e) = ((n / block) as i32, block as i32, e4m3_scale as i32);
        unsafe {
            super::obs::launch(
                self.fp4_qdq,
                blocks(n / block, 128),
                (128, 1, 1),
                0,
                s,
                tang_moe::args![v, g, b, e],
            )?
        }
        Ok(())
    }

    pub fn rms(&self, s: &Stream, x: u64, w: &DsW, rows: usize, eps: f32) -> Result<()> {
        let (wv, n) = (*w, w.cols);
        unsafe {
            super::obs::launch(
                self.rms,
                (rows as u32, 1, 1),
                (1024, 1, 1),
                0,
                s,
                tang_moe::args![x, wv, n, eps],
            )?
        }
        Ok(())
    }

    /// Rotate the last `rd` of each `d`-wide vector, row `r` at `pos0 + r * pstride` (or
    /// `fixed` for every row).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        s: &Stream,
        v: u64,
        tab: u64,
        rows: usize,
        per_row: usize,
        d: usize,
        rd: usize,
        pos0: usize,
        pstride: usize,
        fixed: Option<usize>,
        inverse: bool,
    ) -> Result<()> {
        let n = rows * per_row * rd / 2;
        let (r, pr, di, rdi, p0, ps, fx, inv) = (
            rows as i32,
            per_row as i32,
            d as i32,
            rd as i32,
            pos0 as i32,
            pstride as i32,
            fixed.map_or(-1, |f| f as i32),
            inverse as i32,
        );
        unsafe {
            super::obs::launch(
                self.rope,
                blocks(n, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![v, tab, r, pr, di, rdi, p0, ps, fx, inv],
            )?
        }
        Ok(())
    }

    /// The split-K decode attention (`ds_attn_fd` + `ds_attn_fd_comb`), in passes of rows
    /// whose partials fit the scratch; splits per row to fill about two waves of SMs.
    #[allow(clippy::too_many_arguments)]
    fn attn_fd(
        &self,
        s: &Stream,
        q: u64,
        kv: u64,
        sp: i32,
        kv1: u64,
        sink: u64,
        idxs: u64,
        o: u64,
        t: usize,
        h: usize,
        d: usize,
        k: usize,
        scale: f32,
        rope: Option<AttnRope>,
        widx: Option<WinIdx>,
    ) -> Result<()> {
        let r = rope.unwrap_or_default();
        let (pp, rd, ps) = (r.p, r.rd, r.stride);
        if widx.is_some() {
            anyhow::ensure!(
                self.attn_fdm && pp != 0,
                "attn_fd: in-kernel indices want ds_attn_fdm and P"
            );
        }
        let (wm, ww, wsrc, wkc) = widx.map_or((0, 0, 0, 0), |w| (w.mode, w.win, w.src, w.kc));
        let qtab = if r.q { r.tab } else { 0 };
        let ntile = k.div_ceil(16);
        let hg = h / 64;
        let (fname, smem) = if self.attn_fdm {
            (
                "ds_attn_fdm",
                (64 * 257 + 16 * 257 + 64 * 17 + 64 * 9 + 3 * 64 + 1024) * 4,
            )
        } else {
            (
                "ds_attn_fd",
                (64 * 257 + 16 * 257 + 16 * 64 + 3 * 64 + 16) * 4,
            )
        };
        let mut t0 = 0;
        while t0 < t {
            let n0 = (t - t0).min(64);
            let sp_n = match self.attn_fd_s {
                0 => self.sms.div_ceil(n0 * hg),
                v => v,
            }
            .clamp(1, ntile);
            let n = n0.min(ATTN_SCRATCH / (sp_n * h * (d + 2))).max(1);
            let ns = sp_n as i32;
            let part = self.scratch.ptr;
            let ml = part + (n * sp_n * h * d * 4) as u64;
            let add = r.add + t0 as i32 * r.stride;
            let (wsrc_o, r0) = (wsrc + (t0 * wkc as usize * 4) as u64, t0 as i32);
            let (qo, io, oo) = (
                q + (t0 * h * d * 4) as u64,
                idxs + (t0 * k * 4) as u64,
                o + (t0 * h * d * 4) as u64,
            );
            let (hi, di, ki) = (h as i32, d as i32, k as i32);
            let (qc, qsc) = match r.wo_quant {
                true => (
                    self.wo_x.ptr + (t0 * h * d) as u64,
                    self.wo_x.ptr + (8 * WO_X_COLS + t0 * h * d / 8) as u64,
                ),
                false => (0, 0),
            };
            unsafe {
                super::obs::launch(
                    self.more[fname],
                    ((n * hg * sp_n) as u32, 1, 1),
                    (256, 1, 1),
                    smem as u32,
                    s,
                    tang_moe::args![
                        qo, kv, sp, kv1, io, part, ml, hi, di, ki, scale, ns, qtab, pp, rd, add,
                        ps, wm, ww, wsrc_o, wkc, r0
                    ],
                )?;
                super::obs::launch(
                    self.more["ds_attn_fd_comb"],
                    ((n * h) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![
                        part, ml, sink, oo, hi, di, ns, r.tab, pp, rd, add, ps, qc, qsc
                    ],
                )?;
            }
            t0 += n;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// The three tiled-attention launches over `t` rows (t * per-row scratch must fit).
    #[allow(clippy::too_many_arguments)]
    fn attn_tiled(
        &self,
        s: &Stream,
        q: u64,
        kv: u64,
        sp: i32,
        kv1: u64,
        sink: u64,
        idxs: u64,
        o: u64,
        t: usize,
        h: usize,
        d: usize,
        k: usize,
        scale: f32,
        rope: Option<AttnRope>,
        r0: usize,
    ) -> Result<()> {
        let (hi, di, ki) = (h as i32, d as i32, k as i32);
        let r = rope.unwrap_or_default();
        let (tab, pp, rd, ps) = (r.tab, r.p, r.rd, r.stride);
        let add = r.add + r0 as i32 * r.stride;
        let qtab = if r.q { tab } else { 0 };
        let part = self.scratch.ptr;
        let pr = part + (t * h * k * (d / 64) * 4) as u64;
        let den = pr + (t * h * k * 4) as u64;
        let ti = t as i32;
        unsafe {
            super::obs::launch(
                self.attn_qk,
                ((t * (h / 64) * k.div_ceil(32) * (d / 64)) as u32, 1, 1),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![q, kv, sp, kv1, idxs, part, hi, di, ki, qtab, pp, rd, add, ps],
            )?;
            if self.attn_fuse {
                // soft + pv in one launch: dsb slices a block, as few as keep ~all SMs busy
                let nds = d / 64;
                let mut dsb = nds;
                while dsb > 1 && t * (h / 8) * (nds / dsb) < self.sms {
                    dsb -= 1;
                    while nds % dsb != 0 {
                        dsb -= 1;
                    }
                }
                let dsbi = dsb as i32;
                let f = self.more["ds_attn_spv"];
                super::obs::launch(
                    f,
                    ((t * (h / 8) * (nds / dsb)) as u32, 1, 1),
                    (256, 1, 1),
                    (9 * k * 4) as u32,
                    s,
                    tang_moe::args![
                        part, sink, kv, sp, kv1, idxs, o, hi, di, ki, scale, dsbi, tab, pp, rd,
                        add, ps
                    ],
                )?;
                return Ok(());
            }
            super::obs::launch(
                self.attn_soft,
                blocks(t * h * 32, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![part, sink, idxs, pr, den, ti, hi, di, ki, scale],
            )?;
            super::obs::launch(
                self.attn_pv,
                ((t * (h / 8) * (d / 64)) as u32, 1, 1),
                (256, 1, 1),
                (9 * k * 4) as u32,
                s,
                tang_moe::args![pr, den, kv, sp, kv1, idxs, o, hi, di, ki, tab, pp, rd, add, ps],
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        s: &Stream,
        q: u64,
        kv: u64,
        split: usize,
        kv1: u64,
        sink: u64,
        idxs: u64,
        o: u64,
        t: usize,
        h: usize,
        d: usize,
        k: usize,
        scale: f32,
    ) -> Result<()> {
        self.sparse_attn_r(s, q, kv, split, kv1, sink, idxs, o, t, h, d, k, scale, None)
    }

    /// Whether [`Kernels::attn_fd_win`] applies: `ds_attn_fdm` building its own index rows.
    pub fn attn_fd_win_ok(&self, h: usize, d: usize, k: usize) -> bool {
        self.attn_tiled_ok(h, d, k) && self.attn_fd && self.attn_fdm && d == 512 && h % 64 == 0
    }

    /// [`Kernels::sparse_attn_r`] on `ds_attn_fdm` with the index rows built in the kernel
    /// (no `idxs`): one launch fewer. [`Kernels::attn_fd_win_ok`] first; `rope.p` is `P`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_fd_win(
        &self,
        s: &Stream,
        q: u64,
        kv: u64,
        split: usize,
        kv1: u64,
        sink: u64,
        o: u64,
        t: usize,
        h: usize,
        d: usize,
        k: usize,
        scale: f32,
        rope: AttnRope,
        widx: WinIdx,
    ) -> Result<()> {
        anyhow::ensure!(
            self.attn_fd_win_ok(h, d, k),
            "attn_fd_win: shape h {h} d {d} k {k}"
        );
        let sp = split.min(i32::MAX as usize) as i32;
        self.attn_fd(
            s,
            q,
            kv,
            sp,
            kv1,
            sink,
            0,
            o,
            t,
            h,
            d,
            k,
            scale,
            Some(rope),
            Some(widx),
        )
    }

    /// Whether [`Kernels::sparse_attn`] takes the tiled path (and so [`AttnRope`]).
    pub fn attn_tiled_ok(&self, h: usize, d: usize, k: usize) -> bool {
        let per_row = h * k * (d / 64 + 1) + h;
        !self.exact && h % 64 == 0 && d % 64 == 0 && k > 0 && k <= 960 && per_row <= ATTN_SCRATCH
    }

    /// [`Kernels::sparse_attn`], with `rope` (tiled path only, [`Kernels::attn_tiled_ok`]):
    /// q not yet rope'd (rotated as it loads) and o inverse-rope'd as it stores, both bit
    /// for bit `dg_rope`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn_r(
        &self,
        s: &Stream,
        q: u64,
        kv: u64,
        split: usize,
        kv1: u64,
        sink: u64,
        idxs: u64,
        o: u64,
        t: usize,
        h: usize,
        d: usize,
        k: usize,
        scale: f32,
        rope: Option<AttnRope>,
    ) -> Result<()> {
        let (hi, di, ki) = (h as i32, d as i32, k as i32);
        let sp = split.min(i32::MAX as usize) as i32;
        // rows per pass that fit the scratch (prefill: many passes, each row independent)
        let per_row = h * k * (d / 64 + 1) + h;
        // ds_attn_fdm holds a split's indices (<= 1024) in shared memory: k <= 960 here
        if self.attn_tiled_ok(h, d, k) && self.attn_fd && d == 512 && h % 64 == 0 {
            return self.attn_fd(
                s, q, kv, sp, kv1, sink, idxs, o, t, h, d, k, scale, rope, None,
            );
        }
        if self.attn_tiled_ok(h, d, k) {
            let tc = ATTN_SCRATCH / per_row;
            let mut t0 = 0;
            while t0 < t {
                let n = (t - t0).min(tc);
                let (qo, io, oo) = (
                    q + (t0 * h * d * 4) as u64,
                    idxs + (t0 * k * 4) as u64,
                    o + (t0 * h * d * 4) as u64,
                );
                self.attn_tiled(
                    s, qo, kv, sp, kv1, sink, io, oo, n, h, d, k, scale, rope, t0,
                )?;
                t0 += n;
            }
            return Ok(());
        }
        anyhow::ensure!(
            rope.is_none(),
            "sparse_attn: a fused rope wants the tiled path"
        );
        let threads = d.clamp(32, 1024) as u32;
        unsafe {
            super::obs::launch(
                self.sparse_attn,
                ((t * h) as u32, 1, 1),
                (threads, 1, 1),
                (k * 4) as u32,
                s,
                tang_moe::args![q, kv, sp, kv1, sink, idxs, o, hi, di, ki, scale],
            )?
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sinkhorn(
        &self,
        s: &Stream,
        mixes: u64,
        scale: u64,
        base: u64,
        pre: u64,
        post: u64,
        comb: u64,
        t: usize,
        hc: usize,
        iters: usize,
        eps: f32,
    ) -> Result<()> {
        let (ti, hci, it) = (t as i32, hc as i32, iters as i32);
        unsafe {
            super::obs::launch(
                self.sinkhorn,
                blocks(t, 64),
                (64, 1, 1),
                0,
                s,
                tang_moe::args![mixes, scale, base, pre, post, comb, ti, hci, it, eps],
            )?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::{self, tests::tiny, Model};
    use super::super::quant::{bf, fp4_qdq, fp8_qdq};
    use super::*;
    use std::path::Path;

    /// Deterministic bf16-valued test data.
    fn data(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                bf(((x >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * amp)
            })
            .collect()
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let m = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / m
    }

    fn setup() -> Option<(Gpu, Kernels, Stream, Model)> {
        let g = Gpu::for_test()?;
        let dir = Path::new(&tiny()).join("hf");
        if !dir.join("model.safetensors").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return None;
        }
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        Some((g, k, s, Model::open(&dir).unwrap()))
    }

    #[test]
    fn quant_kernels_are_bit_exact() {
        let Some((_g, k, s, _m)) = setup() else {
            return;
        };
        let x = data(32 * 64, 1, 3.0);
        let dev = DevBuf::from_slice(&x).unwrap();
        k.fp8_qdq(&s, dev.ptr, x.len()).unwrap();
        s.sync().unwrap();
        let mut want = x.clone();
        fp8_qdq(&mut want, 32);
        assert_eq!(dev.read::<f32>(0, x.len()).unwrap(), want, "fp8");
        for (block, e4) in [(32, false), (16, true)] {
            let dev = DevBuf::from_slice(&x).unwrap();
            k.fp4_qdq(&s, dev.ptr, x.len(), block, e4).unwrap();
            s.sync().unwrap();
            let mut want = x.clone();
            fp4_qdq(&mut want, block, e4);
            assert_eq!(
                dev.read::<f32>(0, x.len()).unwrap(),
                want,
                "fp4 block {block}"
            );
        }
    }

    /// Every weight format against `Model::linear` (which quantizes the activation itself).
    #[test]
    fn linear_matches_cpu() {
        let Some((_g, k, s, m)) = setup() else { return };
        let t = 5;
        for name in [
            "layers.0.attn.wq_b.weight",           // fp8
            "layers.0.ffn.experts.3.w2.weight",    // fp4
            "layers.2.attn.compressor.wkv.weight", // bf16
            "layers.0.ffn.gate.weight",            // bf16 router
            "layers.1.engram.wkv.weight",
            "layers.0.attn.wo_b.weight",
            "layers.0.ffn.shared_experts.w2.weight",
            "mtp.0.main_proj.weight",
        ] {
            let w = DevW::upload(&m.st, name).unwrap();
            let cols = w.v.cols as usize;
            let rows = w.v.rows as usize;
            let x = data(t * cols, 7, 2.0);
            let want = m.linear(&x, t, name).unwrap();
            let dx = DevBuf::from_slice(&x).unwrap();
            if w.v.fmt >= 2 {
                k.fp8_qdq(&s, dx.ptr, x.len()).unwrap();
            }
            let dy = DevBuf::alloc((t * rows) * 4).unwrap();
            k.linear(&s, &w.v, dx.ptr, dy.ptr, t, true).unwrap();
            s.sync().unwrap();
            let got = dy.read::<f32>(0, t * rows).unwrap();
            // the t <= 8 kernel is bit-identical to one row at a time
            for ti in 0..t {
                k.linear(
                    &s,
                    &w.v,
                    dx.ptr + (ti * cols * 4) as u64,
                    dy.ptr + (ti * rows * 4) as u64,
                    1,
                    true,
                )
                .unwrap();
            }
            s.sync().unwrap();
            let one = dy.read::<f32>(0, t * rows).unwrap();
            assert!(
                got.iter()
                    .zip(&one)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: t8 != t1"
            );
            let r = rel(&got, &want);
            eprintln!("{name}: rel {r:.2e}");
            assert!(r < 1e-2, "{name}: rel {r}");
        }
    }

    #[test]
    fn rms_rope_attn_sinkhorn_match_cpu() {
        let Some((_g, k, s, m)) = setup() else { return };
        let c = m.cfg.clone();
        let (t, h, d, rd) = (6, c.n_heads, c.head_dim, c.rope_head_dim);

        // rms
        let x = data(t * c.dim, 3, 4.0);
        let mut want = x.clone();
        m.rms(&mut want, "layers.0.attn_norm.weight").unwrap();
        let dx = DevBuf::from_slice(&x).unwrap();
        let nw = DevW::upload(&m.st, "layers.0.attn_norm.weight").unwrap();
        k.rms(&s, dx.ptr, &nw.v, t, c.norm_eps).unwrap();
        s.sync().unwrap();
        let r = rel(&dx.read::<f32>(0, x.len()).unwrap(), &want);
        assert!(r < 1e-2, "rms rel {r}");

        // rope, forward and inverse, at an offset
        let tab = &m.freqs[2];
        let flat: Vec<f32> = tab.iter().flat_map(|&(a, b)| [a, b]).collect();
        let dtab = DevBuf::from_slice(&flat).unwrap();
        let q = data(t * h * d, 4, 2.0);
        for inverse in [false, true] {
            let mut want = q.clone();
            forward::rope(&mut want, d, rd, h, tab, |i| 9 + i, inverse);
            let dq = DevBuf::from_slice(&q).unwrap();
            k.rope(&s, dq.ptr, dtab.ptr, t, h, d, rd, 9, 1, None, inverse)
                .unwrap();
            s.sync().unwrap();
            let r = rel(&dq.read::<f32>(0, q.len()).unwrap(), &want);
            assert!(r < 1e-2, "rope inverse={inverse} rel {r}");
        }

        // sparse attention with -1 holes and a sink
        let n = 20;
        let kv = data(n * d, 5, 1.0);
        let sink = data(h, 6, 1.0);
        let idxs: Vec<Vec<i32>> = (0..t)
            .map(|ti| {
                (0..8)
                    .map(|j| {
                        if (ti + j) % 5 == 0 {
                            -1
                        } else {
                            ((ti * 3 + j * 2) % n) as i32
                        }
                    })
                    .collect()
            })
            .collect();
        let want = forward::sparse_attn(&q, h, d, &kv, &sink, &idxs, (d as f32).powf(-0.5));
        let (dq, dkv, ds) = (
            DevBuf::from_slice(&q).unwrap(),
            DevBuf::from_slice(&kv).unwrap(),
            DevBuf::from_slice(&sink).unwrap(),
        );
        let flat_idx: Vec<i32> = idxs.iter().flatten().copied().collect();
        let di = DevBuf::from_slice(&flat_idx).unwrap();
        let o = DevBuf::alloc((t * h * d) * 4).unwrap();
        k.sparse_attn(
            &s,
            dq.ptr,
            dkv.ptr,
            usize::MAX,
            0,
            ds.ptr,
            di.ptr,
            o.ptr,
            t,
            h,
            d,
            8,
            (d as f32).powf(-0.5),
        )
        .unwrap();
        s.sync().unwrap();
        let r = rel(&o.read::<f32>(0, t * h * d).unwrap(), &want);
        assert!(r < 1e-2, "sparse_attn rel {r}");

        // sinkhorn
        let hc = c.hc_mult;
        let mix = (2 + hc) * hc;
        let mixes = data(t * mix, 8, 2.0);
        let scale = vec![0.7f32, 1.1, 0.9];
        let base = data(mix, 9, 0.5);
        let (dm, dsc, db) = (
            DevBuf::from_slice(&mixes).unwrap(),
            DevBuf::from_slice(&scale).unwrap(),
            DevBuf::from_slice(&base).unwrap(),
        );
        let (pre, post, comb) = (
            DevBuf::alloc((t * hc) * 4).unwrap(),
            DevBuf::alloc((t * hc) * 4).unwrap(),
            DevBuf::alloc((t * hc * hc) * 4).unwrap(),
        );
        k.sinkhorn(
            &s,
            dm.ptr,
            dsc.ptr,
            db.ptr,
            pre.ptr,
            post.ptr,
            comb.ptr,
            t,
            hc,
            c.hc_sinkhorn_iters,
            c.hc_eps,
        )
        .unwrap();
        s.sync().unwrap();
        let (gp, gq, gc) = (
            pre.read::<f32>(0, t * hc).unwrap(),
            post.read::<f32>(0, t * hc).unwrap(),
            comb.read::<f32>(0, t * hc * hc).unwrap(),
        );
        for ti in 0..t {
            let (wp, wq, wc) = forward::sinkhorn(
                &mixes[ti * mix..(ti + 1) * mix],
                &scale,
                &base,
                hc,
                c.hc_sinkhorn_iters,
                c.hc_eps,
            );
            assert!(rel(&gp[ti * hc..(ti + 1) * hc], &wp) < 1e-5);
            assert!(rel(&gq[ti * hc..(ti + 1) * hc], &wq) < 1e-5);
            assert!(
                rel(&gc[ti * hc * hc..(ti + 1) * hc * hc], &wc) < 1e-5,
                "sinkhorn comb"
            );
        }
    }

    /// Activations with group magnitudes spread over a few binades, so the fused activation
    /// quant sees varied scales.
    fn acts(n: usize, seed: u64) -> Vec<f32> {
        let mut x = data(n, seed, 2.0);
        for (g, c) in x.chunks_mut(32).enumerate() {
            let m = 2f32.powi((g % 7) as i32 - 3);
            for v in c {
                *v = bf(*v * m);
            }
        }
        x
    }

    /// The slow path: fp8_qdq on a copy, then ds_linear (bf16 out).
    fn slow(k: &Kernels, s: &Stream, w: &DsW, x: &[f32], t: usize) -> Vec<f32> {
        let dx = DevBuf::from_slice(x).unwrap();
        k.fp8_qdq(s, dx.ptr, x.len()).unwrap();
        let dy = DevBuf::alloc(t * w.rows as usize * 4).unwrap();
        k.linear(s, w, dx.ptr, dy.ptr, t, true).unwrap();
        s.sync().unwrap();
        dy.read::<f32>(0, t * w.rows as usize).unwrap()
    }

    /// The fast-path bar against the exact kernels: bit for bit under `DS_EXACT=1`; otherwise
    /// ≥ 99.5% of elements equal and every other one a neighbouring bf16 value.
    fn check(k: &Kernels, got: &[f32], want: &[f32], what: &str) {
        if k.exact {
            assert_eq!(got, want, "{what}");
            return;
        }
        assert_eq!(got.len(), want.len());
        let mut diff = 0;
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            diff += 1;
            let (ua, ub) = ((a.to_bits() >> 16) as i32, (b.to_bits() >> 16) as i32);
            let near = (a.signum() == b.signum() && (ua - ub).abs() == 1) || (a == 0.0 && b == 0.0);
            assert!(near, "{what}: [{i}] {a} vs {b} is not a bf16 neighbour");
        }
        assert!(
            diff as f64 <= 0.005 * got.len() as f64,
            "{what}: {diff} of {} differ",
            got.len()
        );
    }

    /// The fast GEMV against the slow path, bit for bit, for every fp8/fp4 weight shape in
    /// the fixture, with the activation quant fused and not, over decode and verify widths
    /// (and t > 8, which runs in launches of 8).
    #[test]
    fn fast_linear_matches_slow() {
        let Some((_g, k, s, m)) = setup() else { return };
        let mut seen = std::collections::HashSet::new();
        let mut names: Vec<&str> = m.st.names().filter(|n| n.ends_with("weight")).collect();
        names.sort();
        let mut checked = 0;
        for name in names {
            let info = m.st.info(name).unwrap();
            if !matches!(info.dtype, Dtype::F8E4M3 | Dtype::I8 | Dtype::U8) || info.shape.len() != 2
            {
                continue;
            }
            if !seen.insert((info.dtype as u8, info.shape.clone())) {
                continue;
            }
            let Ok(w) = DevW::upload(&m.st, name) else {
                continue;
            };
            assert!(Kernels::fast_ok(&w.v), "{name}");
            let (rows, cols) = (w.v.rows as usize, w.v.cols as usize);
            for t in [1, 2, 3, 4, 5, 6, 8, 11, 64, 137] {
                let x = acts(t * cols, 11 + t as u64);
                let want = slow(&k, &s, &w.v, &x, t);
                let dy = DevBuf::alloc(t * rows * 4).unwrap();
                // fused qdq
                let dx = DevBuf::from_slice(&x).unwrap();
                k.linear_fast(&s, &w.v, None, dx.ptr, dy.ptr, t, true, true, 0, 0.0)
                    .unwrap();
                s.sync().unwrap();
                check(
                    &k,
                    &dy.read::<f32>(0, t * rows).unwrap(),
                    &want,
                    &format!("{name} t {t} fused"),
                );
                // pre-quantized
                k.fp8_qdq(&s, dx.ptr, x.len()).unwrap();
                k.linear_fast(&s, &w.v, None, dx.ptr, dy.ptr, t, true, false, 0, 0.0)
                    .unwrap();
                s.sync().unwrap();
                check(
                    &k,
                    &dy.read::<f32>(0, t * rows).unwrap(),
                    &want,
                    &format!("{name} t {t}"),
                );
                // unrounded f32 against unrounded ds_linear
                let dq = DevBuf::from_slice(&x).unwrap();
                k.fp8_qdq(&s, dq.ptr, x.len()).unwrap();
                k.linear(&s, &w.v, dq.ptr, dy.ptr, t, false).unwrap();
                s.sync().unwrap();
                let wf = dy.read::<f32>(0, t * rows).unwrap();
                k.linear_fast(&s, &w.v, None, dq.ptr, dy.ptr, t, false, false, 0, 0.0)
                    .unwrap();
                s.sync().unwrap();
                let r = rel(&dy.read::<f32>(0, t * rows).unwrap(), &wf);
                assert!(r < 1e-5, "{name} t {t}: f32 rel {r:e}");
            }
            eprintln!("{name} fmt {} [{rows}, {cols}]: exact", w.v.fmt);
            checked += 1;
        }
        assert!(checked >= 8, "only {checked} shapes");
    }

    /// The fused w1/w3 GEMV + SwiGLU against two slow GEMVs and ds_swiglu, bit for bit.
    #[test]
    fn fast_swiglu_matches_slow() {
        let Some((_g, k, s, m)) = setup() else { return };
        let lim = m.cfg.swiglu_limit;
        for p in ["layers.0.ffn.experts.0", "layers.0.ffn.shared_experts"] {
            let w1 = DevW::upload(&m.st, &format!("{p}.w1.weight")).unwrap();
            let w3 = DevW::upload(&m.st, &format!("{p}.w3.weight")).unwrap();
            let (inter, cols) = (w1.v.rows as usize, w1.v.cols as usize);
            for t in [1, 2, 5, 6, 8, 11, 64, 137] {
                for routed in [false, true] {
                    let x = acts(t * cols, 21 + t as u64);
                    let g = DevBuf::from_slice(&slow(&k, &s, &w1.v, &x, t)).unwrap();
                    let u = DevBuf::from_slice(&slow(&k, &s, &w3.v, &x, t)).unwrap();
                    let rw = DevBuf::from_slice(&data(t, 5, 1.5)).unwrap();
                    let rwp = if routed { rw.ptr } else { 0 };
                    let h = DevBuf::alloc(t * inter * 4).unwrap();
                    let (nn, ii) = ((t * inter) as i64, inter as i32);
                    unsafe {
                        k.go(
                            "ds_swiglu",
                            blocks(t * inter, 256),
                            (256, 1, 1),
                            0,
                            &s,
                            tang_moe::args![g.ptr, u.ptr, rwp, h.ptr, nn, ii, lim],
                        )
                        .unwrap()
                    };
                    s.sync().unwrap();
                    let want = h.read::<f32>(0, t * inter).unwrap();
                    let dx = DevBuf::from_slice(&x).unwrap();
                    k.linear_fast(
                        &s,
                        &w1.v,
                        Some(&w3.v),
                        dx.ptr,
                        h.ptr,
                        t,
                        true,
                        true,
                        rwp,
                        lim,
                    )
                    .unwrap();
                    s.sync().unwrap();
                    check(
                        &k,
                        &h.read::<f32>(0, t * inter).unwrap(),
                        &want,
                        &format!("{p} t {t} routed {routed}"),
                    );
                }
            }
            eprintln!("{p} fmt {}: swiglu exact", w1.v.fmt);
        }
    }

    /// [`check`]'s bar for random weights, where a sum can cancel to far below its terms:
    /// ≤ 0.5% of elements differ, each a bf16 neighbour or within 2^-8 of the output's scale
    /// (the K-split's reduction order against the exact kernels').
    fn check_abs(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len());
        let m = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        let mut diff = 0;
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            diff += 1;
            let (ua, ub) = ((a.to_bits() >> 16) as i32, (b.to_bits() >> 16) as i32);
            let near =
                (a.signum() == b.signum() && (ua - ub).abs() == 1) || (a - b).abs() <= m / 256.0;
            assert!(near, "{what}: [{i}] {a} vs {b} (scale {m})");
        }
        assert!(
            diff as f64 <= 0.005 * got.len() as f64,
            "{what}: {diff} of {} differ",
            got.len()
        );
    }

    /// Random fp8 (fmt 2) or fp4 (fmt 3) weight [rows, cols], kept alive in `keep`.
    fn rand_w(fmt: i32, rows: usize, cols: usize, seed: u64, keep: &mut Vec<DevBuf>) -> DsW {
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut rb = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect()
        };
        let mut wb = rb(if fmt == 3 {
            rows * cols / 2
        } else {
            rows * cols
        });
        if fmt == 2 {
            for b in &mut wb {
                if *b & 0x7f == 0x7f {
                    *b = 0x10; // no NaN codes
                }
            }
        }
        let ns = if fmt == 3 {
            rows * cols / 32
        } else {
            rows.div_ceil(32) * cols / 32
        };
        let sb: Vec<u8> = rb(ns).iter().map(|b| 118 + b % 8).collect();
        let (w, sc) = (
            DevBuf::from_slice(&wb).unwrap(),
            DevBuf::from_slice(&sb).unwrap(),
        );
        let v = DsW {
            w: w.ptr,
            s: sc.ptr,
            rows: rows as i32,
            cols: cols as i32,
            fmt,
            _pad: 0,
        };
        keep.push(w);
        keep.push(sc);
        v
    }

    /// The grouped K-split MoE kernels against the grouped exact ones (bit-identical to the
    /// per-expert loop), h and the down partials, under the fast-path bar, for fp8 and fp4,
    /// decode and verify widths, distinct and shared experts.
    #[test]
    fn moe_grouped_ks_matches_exact() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        if k.exact {
            eprintln!("exact build: nothing to compare");
            return;
        }
        let s = Stream::blocking().unwrap();
        let (ne, topk, d, inter) = (32usize, 4usize, 1024usize, 512usize);
        for fmt in [2, 3] {
            let mut keep = Vec::new();
            let (mut w1, mut w3, mut w2) = (vec![], vec![], vec![]);
            for e in 0..ne as u64 {
                w1.push(rand_w(fmt, inter, d, 3 * e + 1, &mut keep));
                w3.push(rand_w(fmt, inter, d, 3 * e + 2, &mut keep));
                w2.push(rand_w(fmt, d, inter, 3 * e + 3, &mut keep));
            }
            let (w1, w3, w2) = (
                DevBuf::from_slice(&w1).unwrap(),
                DevBuf::from_slice(&w3).unwrap(),
                DevBuf::from_slice(&w2).unwrap(),
            );
            let mask = DevBuf::from_slice(&vec![1i32; ne]).unwrap();
            for t in [1, 2, 3, 6, 8] {
                for shared in [false, true] {
                    let n = t * topk;
                    let ids: Vec<i32> = (0..n)
                        .map(|i| {
                            let (ti, j) = (i / topk, i % topk);
                            if shared {
                                (j * 5 + 1) as i32
                            } else {
                                ((ti * 7 + j * 3 + 2) % ne) as i32
                            }
                        })
                        .collect();
                    let wts = DevBuf::from_slice(&data(n, 9 + t as u64, 1.0)).unwrap();
                    let ids = DevBuf::from_slice(&ids).unwrap();
                    let (counts, off) = (
                        DevBuf::alloc(ne * 4).unwrap(),
                        DevBuf::alloc((ne + 1) * 4).unwrap(),
                    );
                    let (rows, rw) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
                    k.route(
                        &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
                    )
                    .unwrap();
                    let x = DevBuf::from_slice(&data(t * d, 3 + t as u64, 2.0)).unwrap();
                    k.fp8_qdq(&s, x.ptr, t * d).unwrap();
                    let slots = n.min(ne);
                    let elist = DevBuf::alloc(slots * 4).unwrap();
                    let (nei, ci) = (ne as i32, slots as i32);
                    unsafe {
                        k.go(
                            "ds_moe_plan",
                            (1, 1, 1),
                            (32, 1, 1),
                            0,
                            &s,
                            tang_moe::args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
                        )
                        .unwrap();
                    }
                    let mut gu = |exact: bool| -> Vec<f32> {
                        k.exact = exact;
                        let h = DevBuf::from_slice(&vec![0f32; n * inter]).unwrap();
                        k.moe_gu(
                            &s, fmt, w1.ptr, w3.ptr, elist.ptr, off.ptr, rows.ptr, rw.ptr, x.ptr,
                            h.ptr, t, slots, inter, d, 10.0,
                        )
                        .unwrap();
                        s.sync().unwrap();
                        h.read::<f32>(0, n * inter).unwrap()
                    };
                    let (he, hk) = (gu(true), gu(false));
                    // down from the same (exact, quantized) h on both paths
                    let hx = DevBuf::from_slice(&he).unwrap();
                    k.fp8_qdq(&s, hx.ptr, n * inter).unwrap();
                    let mut down = |exact: bool| -> Vec<f32> {
                        k.exact = exact;
                        let part = DevBuf::from_slice(&vec![0f32; n * d]).unwrap();
                        k.moe_down(
                            &s, fmt, w2.ptr, elist.ptr, off.ptr, hx.ptr, part.ptr, t, slots, d,
                            inter, topk,
                        )
                        .unwrap();
                        s.sync().unwrap();
                        part.read::<f32>(0, n * d).unwrap()
                    };
                    let (pe, pk) = (down(true), down(false));
                    k.exact = false;
                    let what = format!("fmt {fmt} t {t} shared {shared}");
                    check_abs(&hk, &he, &format!("{what} gu"));
                    check_abs(&pk, &pe, &format!("{what} down"));
                    assert!(he.iter().any(|&v| v != 0.0) && pe.iter().any(|&v| v != 0.0));
                }
            }
            eprintln!("fmt {fmt}: grouped K-split within the fast-path bar");
        }
    }

    /// The 8-row tiled grouped MoE (prefill, t > 8, many rows per expert) against a
    /// per-expert loop of the exact `linear_fast` over each CSR segment: bit for bit under
    /// the exact kernels, within the K-split bar otherwise.
    #[test]
    fn moe_tiled_matches_loop() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (ne, topk, d, inter) = (16usize, 4usize, 512usize, 256usize);
        let ks_avail = !k.exact;
        for fmt in [2, 3] {
            let mut keep = Vec::new();
            let (mut w1, mut w3, mut w2) = (vec![], vec![], vec![]);
            for e in 0..ne as u64 {
                w1.push(rand_w(fmt, inter, d, 3 * e + 1, &mut keep));
                w3.push(rand_w(fmt, inter, d, 3 * e + 2, &mut keep));
                w2.push(rand_w(fmt, d, inter, 3 * e + 3, &mut keep));
            }
            let (w1d, w3d, w2d) = (
                DevBuf::from_slice(&w1).unwrap(),
                DevBuf::from_slice(&w3).unwrap(),
                DevBuf::from_slice(&w2).unwrap(),
            );
            let mask = DevBuf::from_slice(&vec![1i32; ne]).unwrap();
            for t in [9, 64, 200, 512] {
                let n = t * topk;
                // skewed: expert 0 takes every token, the rest spread unevenly
                let ids: Vec<i32> = (0..n)
                    .map(|i| {
                        let (ti, j) = (i / topk, i % topk);
                        if j == 0 {
                            0
                        } else {
                            (1 + (ti * ti + 3 * j) % (ne - 1)) as i32
                        }
                    })
                    .collect();
                // distinct ids per token
                let ids: Vec<i32> = ids
                    .chunks(topk)
                    .flat_map(|c| {
                        let mut v: Vec<i32> = vec![];
                        for &e in c {
                            let mut e = e;
                            while v.contains(&e) {
                                e = 1 + e % (ne as i32 - 1);
                            }
                            v.push(e);
                        }
                        v
                    })
                    .collect();
                let wts = DevBuf::from_slice(&data(n, 9 + t as u64, 1.0)).unwrap();
                let idd = DevBuf::from_slice(&ids).unwrap();
                let (counts, off) = (
                    DevBuf::alloc(ne * 4).unwrap(),
                    DevBuf::alloc((ne + 1) * 4).unwrap(),
                );
                let (rows, rw) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
                k.route(
                    &s, idd.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
                )
                .unwrap();
                let x = DevBuf::from_slice(&data(t * d, 3 + t as u64, 2.0)).unwrap();
                k.fp8_qdq(&s, x.ptr, t * d).unwrap();
                let slots = n.min(ne);
                let elist = DevBuf::alloc(slots * 4).unwrap();
                let (nei, ci) = (ne as i32, slots as i32);
                unsafe {
                    k.go(
                        "ds_moe_plan",
                        (1, 1, 1),
                        (32, 1, 1),
                        0,
                        &s,
                        tang_moe::args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
                    )
                    .unwrap();
                }
                s.sync().unwrap();
                let offh = off.read::<i32>(0, ne + 1).unwrap();
                let rowsh = rows.read::<i32>(0, n).unwrap();
                let rwh = rw.read::<f32>(0, n).unwrap();
                let xh = x.read::<f32>(0, t * d).unwrap();
                let tiles = DevBuf::alloc(Kernels::moe_tile_cap(n, slots) * 4).unwrap();
                let wide = n * d.max(inter);
                let (codes, scales) = (
                    DevBuf::alloc(wide).unwrap(),
                    DevBuf::alloc(wide / 32 * 4).unwrap(),
                );
                let mut tiled = |exact: bool| -> (Vec<f32>, Vec<f32>) {
                    k.exact = exact;
                    let h = DevBuf::from_slice(&vec![0f32; n * inter]).unwrap();
                    let part = DevBuf::from_slice(&vec![0f32; n * d]).unwrap();
                    k.moe_tiled(
                        &s,
                        fmt,
                        (w1d.ptr, w3d.ptr, w2d.ptr),
                        (elist.ptr, slots, tiles.ptr),
                        (off.ptr, rows.ptr, rw.ptr),
                        (x.ptr, h.ptr, part.ptr),
                        (codes.ptr, scales.ptr),
                        (t, topk, d, inter),
                        10.0,
                    )
                    .unwrap();
                    s.sync().unwrap();
                    (
                        h.read::<f32>(0, n * inter).unwrap(),
                        part.read::<f32>(0, n * d).unwrap(),
                    )
                };
                let (he, pe) = tiled(true);
                let (hk, pk) = if ks_avail {
                    tiled(false)
                } else {
                    (he.clone(), pe.clone())
                };
                k.exact = true;
                // the loop: gate/up from x, down from a given (quantized) h
                let gu_ref = || -> Vec<f32> {
                    let mut out = vec![0f32; n * inter];
                    for e in 0..ne {
                        let (o, m) = (offh[e] as usize, (offh[e + 1] - offh[e]) as usize);
                        if m == 0 {
                            continue;
                        }
                        let xs: Vec<f32> = (0..m)
                            .flat_map(|j| {
                                let r = rowsh[o + j] as usize;
                                xh[r * d..(r + 1) * d].to_vec()
                            })
                            .collect();
                        let (xd, rwd) = (
                            DevBuf::from_slice(&xs).unwrap(),
                            DevBuf::from_slice(&rwh[o..o + m]).unwrap(),
                        );
                        let y = DevBuf::alloc(m * inter * 4).unwrap();
                        k.linear_fast(
                            &s,
                            &w1[e],
                            Some(&w3[e]),
                            xd.ptr,
                            y.ptr,
                            m,
                            true,
                            false,
                            rwd.ptr,
                            10.0,
                        )
                        .unwrap();
                        s.sync().unwrap();
                        out[o * inter..(o + m) * inter]
                            .copy_from_slice(&y.read::<f32>(0, m * inter).unwrap());
                    }
                    out
                };
                let down_ref = |h: &[f32]| -> Vec<f32> {
                    let hq = DevBuf::from_slice(h).unwrap();
                    k.fp8_qdq(&s, hq.ptr, n * inter).unwrap();
                    let mut out = vec![0f32; n * d];
                    for e in 0..ne {
                        let (o, m) = (offh[e] as usize, (offh[e + 1] - offh[e]) as usize);
                        if m == 0 {
                            continue;
                        }
                        let y = DevBuf::alloc(m * d * 4).unwrap();
                        let xo = hq.ptr + (o * inter * 4) as u64;
                        k.linear_fast(&s, &w2[e], None, xo, y.ptr, m, true, false, 0, 0.0)
                            .unwrap();
                        s.sync().unwrap();
                        out[o * d..(o + m) * d].copy_from_slice(&y.read::<f32>(0, m * d).unwrap());
                    }
                    out
                };
                // moe_tiled leaves h fp8-quantized (down's input)
                let hr = {
                    let b = DevBuf::from_slice(&gu_ref()).unwrap();
                    k.fp8_qdq(&s, b.ptr, n * inter).unwrap();
                    s.sync().unwrap();
                    b.read::<f32>(0, n * inter).unwrap()
                };
                let what = format!("fmt {fmt} t {t}");
                let bits = |a: &[f32], b: &[f32], w: &str| {
                    let bad: Vec<usize> = (0..a.len())
                        .filter(|&i| a[i].to_bits() != b[i].to_bits())
                        .collect();
                    assert!(
                        bad.is_empty(),
                        "{w}: {} of {} differ, first {:?} ({} vs {})",
                        bad.len(),
                        a.len(),
                        bad.first(),
                        a[bad[0]],
                        b[bad[0]]
                    );
                };
                bits(&he, &hr, &format!("{what} exact gu"));
                bits(&pe, &down_ref(&he), &format!("{what} exact down"));
                if ks_avail {
                    check_abs(&hk, &hr, &format!("{what} ks gu"));
                    check_abs(&pk, &down_ref(&hk), &format!("{what} ks down"));
                }
                assert!(he.iter().any(|&v| v != 0.0) && pe.iter().any(|&v| v != 0.0));
                k.exact = !ks_avail;
            }
            eprintln!("fmt {fmt}: tiled grouped MoE matches the loop");
        }
    }

    /// Prefill at real shapes, t = 512 (manual: `--ignored --nocapture prefill_bench`): one
    /// layer's routed MoE by the tiled grouped kernels (384 experts, top-6, d 5120, inter
    /// 2304, fp4; 16 distinct weight sets aliased across the experts), and the t > 8 K-split
    /// GEMV in one launch against 8-row launches (the previous path) for wq_b / wo_b shapes.
    #[test]
    #[ignore]
    fn prefill_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        if k.exact {
            return;
        }
        let s = Stream::blocking().unwrap();
        let t: usize = std::env::var("DS_BENCH_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let time = |f: &mut dyn FnMut()| -> f64 {
            f();
            s.sync().unwrap();
            let iters = 10;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            s.sync().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64 * 1e6
        };
        // linears
        let mut keep = Vec::new();
        for (name, rows, cols, fmt) in [("wq_b", 32768, 1024, 2), ("wo_b", 5120, 8192, 2)] {
            let w = rand_w(fmt, rows, cols, 7, &mut keep);
            let x = DevBuf::from_slice(&data(t * cols, 5, 2.0)).unwrap();
            let y = DevBuf::alloc(t * rows * 4).unwrap();
            let one = time(&mut || {
                k.linear_ks(&s, &w, None, x.ptr, y.ptr, t, true, false, 0, 0.0)
                    .unwrap()
            });
            let chunks = time(&mut || {
                for t0 in (0..t).step_by(8) {
                    let n = (t - t0).min(8);
                    let (xo, yo) = (
                        x.ptr + (t0 * cols * 4) as u64,
                        y.ptr + (t0 * rows * 4) as u64,
                    );
                    k.linear_ks(&s, &w, None, xo, yo, n, true, false, 0, 0.0)
                        .unwrap();
                }
            });
            let fl = 2.0 * (t * rows * cols) as f64;
            eprintln!(
                "{name} [{rows}x{cols}] t {t}: one launch {one:.0} us ({:.1} TFLOP/s), 8-row launches {chunks:.0} us",
                fl / one / 1e6
            );
        }
        // MoE
        let (ne, topk, d, inter, fmt) = (384usize, 6usize, 5120usize, 2304usize, 3);
        let (mut w1, mut w3, mut w2) = (vec![], vec![], vec![]);
        for e in 0..16u64 {
            w1.push(rand_w(fmt, inter, d, 3 * e + 1, &mut keep));
            w3.push(rand_w(fmt, inter, d, 3 * e + 2, &mut keep));
            w2.push(rand_w(fmt, d, inter, 3 * e + 3, &mut keep));
        }
        let pick = |v: &Vec<DsW>| (0..ne).map(|e| v[e % 16]).collect::<Vec<_>>();
        let (w1d, w3d, w2d) = (
            DevBuf::from_slice(&pick(&w1)).unwrap(),
            DevBuf::from_slice(&pick(&w3)).unwrap(),
            DevBuf::from_slice(&pick(&w2)).unwrap(),
        );
        let n = t * topk;
        let mut st = 0x2545F4914F6CDD1Du64;
        let ids: Vec<i32> = (0..t)
            .flat_map(|_| {
                let mut v: Vec<i32> = vec![];
                while v.len() < topk {
                    st ^= st << 13;
                    st ^= st >> 7;
                    st ^= st << 17;
                    let e = (st % ne as u64) as i32;
                    if !v.contains(&e) {
                        v.push(e);
                    }
                }
                v
            })
            .collect();
        let idd = DevBuf::from_slice(&ids).unwrap();
        let wts = DevBuf::from_slice(&data(n, 9, 1.0)).unwrap();
        let (counts, off) = (
            DevBuf::alloc(ne * 4).unwrap(),
            DevBuf::alloc((ne + 1) * 4).unwrap(),
        );
        let (rows, rw) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
        k.route(
            &s, idd.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
        )
        .unwrap();
        let mask = DevBuf::from_slice(&vec![1i32; ne]).unwrap();
        let slots = n.min(ne);
        let elist = DevBuf::alloc(slots * 4).unwrap();
        let (nei, ci) = (ne as i32, slots as i32);
        unsafe {
            k.go(
                "ds_moe_plan",
                (1, 1, 1),
                (32, 1, 1),
                0,
                &s,
                tang_moe::args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
            )
            .unwrap();
        }
        let x = DevBuf::from_slice(&data(t * d, 3, 2.0)).unwrap();
        k.fp8_qdq(&s, x.ptr, t * d).unwrap();
        let tiles = DevBuf::alloc(Kernels::moe_tile_cap(n, slots) * 4).unwrap();
        let wide = n * d.max(inter);
        let (codes, scales) = (
            DevBuf::alloc(wide).unwrap(),
            DevBuf::alloc(wide / 32 * 4).unwrap(),
        );
        let (h, part) = (
            DevBuf::alloc(n * inter * 4).unwrap(),
            DevBuf::alloc(n * d * 4).unwrap(),
        );
        let us = time(&mut || {
            k.moe_tiled(
                &s,
                fmt,
                (w1d.ptr, w3d.ptr, w2d.ptr),
                (elist.ptr, slots, tiles.ptr),
                (off.ptr, rows.ptr, rw.ptr),
                (x.ptr, h.ptr, part.ptr),
                (codes.ptr, scales.ptr),
                (t, topk, d, inter),
                10.0,
            )
            .unwrap()
        });
        let fl = 2.0 * (n * 3 * d * inter) as f64;
        eprintln!(
            "moe tiled t {t} ({n} pairs): {us:.0} us/layer ({:.1} TFLOP/s), 4 launches",
            fl / us / 1e6
        );
    }

    /// `gemv_bf16` against the exact `linear` at the router's real shape: relaxed order only.
    #[test]
    fn router_gemv_matches_linear() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (ne, d) = (384usize, 5120usize);
        let wb: Vec<u16> = data(ne * d, 5, 0.05)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let wdev = DevBuf::from_slice(&wb).unwrap();
        let w = DsW {
            w: wdev.ptr,
            s: 0,
            rows: ne as i32,
            cols: d as i32,
            fmt: 1,
            _pad: 0,
        };
        for t in [1usize, 6, 8] {
            let x = DevBuf::from_slice(&data(t * d, 7, 1.0)).unwrap();
            let (a, b) = (
                DevBuf::alloc(t * ne * 4).unwrap(),
                DevBuf::alloc(t * ne * 4).unwrap(),
            );
            k.linear(&s, &w, x.ptr, a.ptr, t, false).unwrap();
            k.gemv_bf16(&s, &w, x.ptr, b.ptr, t).unwrap();
            s.sync().unwrap();
            let (a, b) = (
                a.read::<f32>(0, t * ne).unwrap(),
                b.read::<f32>(0, t * ne).unwrap(),
            );
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
            let err = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            assert!(
                err <= 1e-5 * scale.max(1.0),
                "t {t}: err {err} scale {scale}"
            );
        }
    }

    /// One layer's routed MoE by the grouped kernels at the real shapes (384 experts, top-6,
    /// d = 5120, inter = 2304, fp4) with random weights: plan, GU, h quant, down, reduce
    /// (manual: `--ignored --nocapture moe_grouped_bench`). Bytes = the live experts' weights
    /// and scales.
    #[test]
    #[ignore]
    fn moe_grouped_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let peak: f64 = std::env::var("DS_PEAK_GBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(936.0);
        let (ne, topk, d, inter) = (384usize, 6usize, 5120usize, 2304usize);
        let rand_bytes = |n: usize, seed: u64| -> Vec<u8> {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect()
        };
        let mut keep = Vec::new();
        let mut mk = |rows: usize, cols: usize, seed: u64| -> (DsW, usize) {
            let wb = rand_bytes(rows * cols / 2, seed);
            let sb: Vec<u8> = rand_bytes(rows * cols / 32, seed + 1)
                .iter()
                .map(|b| 118 + b % 8)
                .collect();
            let (w, sc) = (
                DevBuf::from_slice(&wb).unwrap(),
                DevBuf::from_slice(&sb).unwrap(),
            );
            let v = DsW {
                w: w.ptr,
                s: sc.ptr,
                rows: rows as i32,
                cols: cols as i32,
                fmt: 3,
                _pad: 0,
            };
            keep.push((w, sc));
            (v, wb.len() + sb.len())
        };
        // 36 distinct experts with weights (every t = 6 slot distinct: the worst case)
        let used: Vec<usize> = (0..36).map(|i| (i * 67 + 5) % ne).collect();
        let (mut w1, mut w3, mut w2) = (
            vec![DsW::default(); ne],
            vec![DsW::default(); ne],
            vec![DsW::default(); ne],
        );
        let mut per = 0;
        for (i, &e) in used.iter().enumerate() {
            let a = mk(inter, d, 10 * i as u64 + 1);
            let b = mk(inter, d, 10 * i as u64 + 2);
            let c = mk(d, inter, 10 * i as u64 + 3);
            (w1[e], w3[e], w2[e]) = (a.0, b.0, c.0);
            per = a.1 + b.1 + c.1;
        }
        let (w1, w3, w2) = (
            DevBuf::from_slice(&w1).unwrap(),
            DevBuf::from_slice(&w3).unwrap(),
            DevBuf::from_slice(&w2).unwrap(),
        );
        let mask = DevBuf::from_slice(&vec![1i32; ne]).unwrap();
        eprintln!(
            "\n| t | routing | live experts | exact µs | exact GB/s | K-split µs | K-split GB/s | % of {peak} |\n|---|---|---|---|---|---|---|---|"
        );
        for (t, label) in [(1, "6 experts"), (6, "36 distinct"), (6, "6 shared")] {
            let n = t * topk;
            let ids: Vec<i32> = (0..n)
                .map(|i| {
                    if label == "6 shared" {
                        used[i % topk] as i32
                    } else {
                        used[i] as i32
                    }
                })
                .collect();
            let live = if label == "36 distinct" { n } else { topk };
            let wts = DevBuf::from_slice(&vec![0.2f32; n]).unwrap();
            let ids = DevBuf::from_slice(&ids).unwrap();
            let (counts, off) = (
                DevBuf::alloc(ne * 4).unwrap(),
                DevBuf::alloc((ne + 1) * 4).unwrap(),
            );
            let (rows, rw) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
            k.route(
                &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
            )
            .unwrap();
            let x = DevBuf::from_slice(&data(t * d, 3, 2.0)).unwrap();
            k.fp8_qdq(&s, x.ptr, t * d).unwrap();
            let (elist, h, part, y) = (
                DevBuf::alloc(n * 4).unwrap(),
                DevBuf::alloc(n * inter * 4).unwrap(),
                DevBuf::alloc(n * d * 4).unwrap(),
                DevBuf::alloc(t * d * 4).unwrap(),
            );
            let slots = n;
            let mut run = |exact: bool| {
                k.exact = exact;
                let (nei, ci) = (ne as i32, slots as i32);
                let (ti, ki, di) = (t as i32, topk as i32, d as i32);
                unsafe {
                    k.go(
                        "ds_moe_plan",
                        (1, 1, 1),
                        (32, 1, 1),
                        0,
                        &s,
                        tang_moe::args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
                    )
                    .unwrap();
                }
                k.moe_gu(
                    &s, 3, w1.ptr, w3.ptr, elist.ptr, off.ptr, rows.ptr, rw.ptr, x.ptr, h.ptr, t,
                    slots, inter, d, 10.0,
                )
                .unwrap();
                if !k.moe_ks_takes(t, inter, n) {
                    k.fp8_qdq(&s, h.ptr, n * inter).unwrap();
                }
                k.moe_down(
                    &s, 3, w2.ptr, elist.ptr, off.ptr, h.ptr, part.ptr, t, slots, d, inter, topk,
                )
                .unwrap();
                unsafe {
                    k.go(
                        "ds_moe_reduce",
                        blocks(t * d, 256),
                        (256, 1, 1),
                        0,
                        &s,
                        tang_moe::args![
                            y.ptr, part.ptr, ids.ptr, off.ptr, rows.ptr, mask.ptr, ti, ki, di
                        ],
                    )
                    .unwrap();
                }
            };
            let mut time = |exact: bool| {
                run(exact);
                s.sync().unwrap();
                let iters = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    run(exact);
                }
                s.sync().unwrap();
                t0.elapsed().as_secs_f64() / iters as f64
            };
            let (te, tk) = (time(true), time(false));
            let (ge, gk) = (
                (live * per) as f64 / te / 1e9,
                (live * per) as f64 / tk / 1e9,
            );
            eprintln!(
                "| {t} | {label} | {live} | {:.0} | {ge:.0} | {:.0} | {gk:.0} | {:.0}% |",
                te * 1e6,
                tk * 1e6,
                100.0 * gk / peak
            );
        }
    }

    /// Bandwidth on the real model's shapes with random weights (manual: `--ignored
    /// --nocapture fast_linear_bench`). Bytes = weights + scales read once per launch.
    /// The vocab head at the real shape (129,280 x 5120 bf16, t = 1 and the 6-token verify),
    /// CUDA-event timed, plus a bit-identity check of the R-rows-per-warp kernel against
    /// `ds_linear_t8` (`cargo test ... --ignored --nocapture head_bench`).
    #[test]
    #[ignore]
    fn head_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (rows, cols) = (129_280usize, 5120usize);
        let wb: Vec<u16> = data(rows * cols, 7, 0.05)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let wd = DevBuf::from_slice(&wb).unwrap();
        let w = DsW {
            w: wd.ptr,
            s: 0,
            rows: rows as i32,
            cols: cols as i32,
            fmt: 1,
            _pad: 0,
        };
        let gb = (rows * cols * 2) as f64 / 1e9;
        let ev = |f: &dyn Fn()| -> f64 {
            f();
            s.sync().unwrap();
            let (a, b) = (
                gpu::Event::new(true).unwrap(),
                gpu::Event::new(true).unwrap(),
            );
            let n = 20;
            a.record(&s).unwrap();
            for _ in 0..n {
                f();
            }
            b.record(&s).unwrap();
            b.sync().unwrap();
            b.since(&a).unwrap() as f64 * 1e3 / n as f64
        };
        for t in [1usize, 2, 6, 8] {
            let x = DevBuf::from_slice(&data(t * cols, 3, 2.0)).unwrap();
            let (y0, y1) = (
                DevBuf::alloc(t * rows * 4).unwrap(),
                DevBuf::alloc(t * rows * 4).unwrap(),
            );
            let base = ev(&|| k.linear(&s, &w, x.ptr, y0.ptr, t, false).unwrap());
            let rr = ev(&|| k.head(&s, &w, x.ptr, y1.ptr, t).unwrap());
            // each row of the t-row head equals that row run alone
            let y2 = DevBuf::alloc(rows * 4).unwrap();
            let mut same = true;
            let all = y1.read::<f32>(0, t * rows).unwrap();
            for i in 0..t {
                k.head(&s, &w, x.ptr + (i * cols * 4) as u64, y2.ptr, 1)
                    .unwrap();
                let one = y2.read::<f32>(0, rows).unwrap();
                same &= one
                    .iter()
                    .zip(&all[i * rows..])
                    .all(|(p, q)| p.to_bits() == q.to_bits());
            }
            let base_v = y0.read::<f32>(0, t * rows).unwrap();
            let md = base_v
                .iter()
                .zip(&all)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "head t={t}: linear {base:.0} us ({:.0} GB/s), ds_head_bf16 {rr:.0} us ({:.0} GB/s), rows t-independent {same}, max |diff| vs linear {md:.1e}",
                gb / base * 1e6,
                gb / rr * 1e6
            );
        }
    }

    #[test]
    #[ignore]
    fn fast_linear_bench() {
        let Some(_g) = Gpu::for_test() else { return };
        _g.bind().unwrap();
        let mut k = Kernels::new(&_g).unwrap();
        let s = Stream::blocking().unwrap();
        let peak: f64 = std::env::var("DS_PEAK_GBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(936.0);
        let rand_bytes = |n: usize, seed: u64| -> Vec<u8> {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect()
        };
        let mk = |fmt: i32, rows: usize, cols: usize, seed: u64| -> (DevBuf, DevBuf, DsW, usize) {
            let mut wb = rand_bytes(
                if fmt == 3 {
                    rows * cols / 2
                } else {
                    rows * cols
                },
                seed,
            );
            if fmt == 2 {
                for b in &mut wb {
                    if *b & 0x7f == 0x7f {
                        *b = 0x10; // no NaN codes
                    }
                }
            }
            let ns = if fmt == 3 {
                rows * cols / 32
            } else {
                rows.div_ceil(32) * cols / 32
            };
            let sb: Vec<u8> = rand_bytes(ns, seed + 1)
                .iter()
                .map(|b| 118 + b % 8)
                .collect();
            let (w, sc) = (
                DevBuf::from_slice(&wb).unwrap(),
                DevBuf::from_slice(&sb).unwrap(),
            );
            let v = DsW {
                w: w.ptr,
                s: sc.ptr,
                rows: rows as i32,
                cols: cols as i32,
                fmt,
                _pad: 0,
            };
            (w, sc, v, wb.len() + sb.len())
        };
        // Host-timed loop of 50 calls: includes launch overhead (~2 µs a launch here, see the
        // rope row), which a CUDA graph mostly removes.
        let time = |f: &dyn Fn()| -> f64 {
            f();
            s.sync().unwrap();
            let n = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            s.sync().unwrap();
            t0.elapsed().as_secs_f64() / n as f64
        };
        // Weights cycle through copies totalling > 4× L2 (6 MB on the 3090), so every launch
        // reads DRAM. Bytes = weights + scales.
        let copies = |bytes: usize| (24usize << 20).div_ceil(bytes).max(1);
        {
            // calibration: what a plain copy reaches here (read + write bytes)
            let n = 32usize << 20;
            let (a, b) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
            let nn = n as i64;
            let tc = time(&|| unsafe {
                k.go(
                    "ds_copy",
                    blocks(n, 256),
                    (256, 1, 1),
                    0,
                    &s,
                    tang_moe::args![b.ptr, a.ptr, nn],
                )
                .unwrap()
            });
            eprintln!(
                "calibration: ds_copy 128 MB -> {:.0} GB/s",
                2.0 * (n * 4) as f64 / tc / 1e9
            );
        }
        let kinds: &[(&str, i32, usize, usize, bool)] = &[
            ("wq_a", 2, 1280, 5120, false),
            ("wq_b", 2, 32768, 1280, false),
            ("wkv", 2, 512, 5120, false),
            ("wo_a", 2, 8192, 4096, false),
            ("wo_b", 2, 5120, 8192, false),
            ("expert w1", 3, 2304, 5120, false),
            ("expert w2", 3, 5120, 2304, false),
            ("expert w1+w3", 3, 2304, 5120, true),
        ];
        eprintln!(
            "\n| weight | fmt | [rows, cols] | t | exact µs | exact GB/s | K-split µs | K-split GB/s | % of {peak} |"
        );
        eprintln!("|---|---|---|---|---|---|---|---|---|");
        // t = 1 K-split µs per kind, for the per-token estimate
        let mut t1 = std::collections::HashMap::new();
        for &(label, fmt, rows, cols, gu) in kinds {
            let one = mk(fmt, rows, cols, 1).3 * if gu { 2 } else { 1 };
            let nc = copies(one);
            let ws: Vec<_> = (0..nc * if gu { 2 } else { 1 })
                .map(|i| mk(fmt, rows, cols, rows as u64 + cols as u64 + i as u64))
                .collect();
            for t in [1, 2, 6, 8] {
                let x = DevBuf::from_slice(&data(t * cols, 3, 2.0)).unwrap();
                let y = DevBuf::alloc(t * rows * 4).unwrap();
                let i = std::cell::Cell::new(0usize);
                let mut run = |exact: bool| {
                    k.exact = exact;
                    let f = || {
                        let c = i.get() % nc;
                        i.set(i.get() + 1);
                        let (w, w3) = if gu {
                            (&ws[2 * c].2, Some(&ws[2 * c + 1].2))
                        } else {
                            (&ws[c].2, None)
                        };
                        k.linear_fast(&s, w, w3, x.ptr, y.ptr, t, true, true, 0, 10.0)
                            .unwrap()
                    };
                    time(&f)
                };
                let (te, tk) = (run(true), run(false));
                k.exact = false;
                let (ge, gk) = (one as f64 / te / 1e9, one as f64 / tk / 1e9);
                eprintln!(
                    "| {label} | {} | [{}{rows}, {cols}] | {t} | {:.1} | {ge:.0} | {:.1} | {gk:.0} | {:.0}% |",
                    if fmt == 2 { "fp8" } else { "fp4" },
                    if gu { "2×" } else { "" },
                    te * 1e6,
                    tk * 1e6,
                    100.0 * gk / peak
                );
                if t == 1 {
                    t1.insert(label, (tk, one));
                }
            }
        }

        // The small ops at d = 5120, hc = 4, 64 heads × 512 (t = 1, and 6).
        eprintln!("\n| op | t | µs |\n|---|---|---|");
        let d = 5120;
        let nw = DevBuf::from_slice(&vec![0x3f80u16; d]).unwrap();
        let nv = DsW {
            w: nw.ptr,
            s: 0,
            rows: 1,
            cols: d as i32,
            fmt: 1,
            _pad: 0,
        };
        let tab: Vec<f32> = (0..4096 * 32)
            .flat_map(|i| [(i as f32).cos(), (i as f32).sin()])
            .collect();
        let dtab = DevBuf::from_slice(&tab).unwrap();
        let mut small = 0.0;
        for t in [1, 6] {
            let xb = DevBuf::from_slice(&data(t * 4 * d, 3, 2.0)).unwrap();
            let pre = DevBuf::from_slice(&data(t * 4, 4, 1.0)).unwrap();
            let comb = DevBuf::from_slice(&data(t * 16, 5, 1.0)).unwrap();
            let hb = DevBuf::from_slice(&data(t * d, 6, 1.0)).unwrap();
            let out = DevBuf::alloc(t * 4 * d * 4).unwrap();
            let qb = DevBuf::from_slice(&data(t * 64 * 512, 7, 1.0)).unwrap();
            let (ti, hc, di) = (t as i32, 4i32, d as i32);
            let ops: Vec<(&str, Box<dyn Fn()>)> = vec![
                (
                    "rms [t, 5120]",
                    Box::new(|| k.rms(&s, xb.ptr, &nv, t, 1e-6).unwrap()),
                ),
                (
                    "fp8_qdq [t, 5120]",
                    Box::new(|| k.fp8_qdq(&s, xb.ptr, t * d).unwrap()),
                ),
                (
                    "rope q [t, 64, 512] rd 64",
                    Box::new(|| {
                        k.rope(&s, qb.ptr, dtab.ptr, t, 64, 512, 64, 9, 1, None, false)
                            .unwrap()
                    }),
                ),
                (
                    "hc_pre [t, 4, 5120]",
                    Box::new(|| unsafe {
                        k.go(
                            "ds_hc_pre",
                            blocks(t * d, 256),
                            (256, 1, 1),
                            0,
                            &s,
                            tang_moe::args![xb.ptr, pre.ptr, out.ptr, ti, hc, di],
                        )
                        .unwrap()
                    }),
                ),
                (
                    "hc_post [t, 4, 5120]",
                    Box::new(|| unsafe {
                        k.go(
                            "ds_hc_post",
                            blocks(t * 4 * d, 256),
                            (256, 1, 1),
                            0,
                            &s,
                            tang_moe::args![
                                hb.ptr, 0u64, xb.ptr, pre.ptr, comb.ptr, out.ptr, ti, hc, di
                            ],
                        )
                        .unwrap()
                    }),
                ),
            ];
            for (name, f) in &ops {
                let us = time(f.as_ref()) * 1e6;
                if t == 1 {
                    small += us;
                }
                eprintln!("| {name} | {t} | {us:.1} |");
            }
        }

        // Per-token estimate at the real shapes, t = 1: 40 layers × (dense attention
        // projections + 6 routed + 1 shared expert), the GEMVs at their measured K-split time.
        let l = 40.0;
        let dense = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"];
        let (mut us, mut bytes) = (0.0, 0.0);
        for n in dense {
            let (tk, b) = t1[n];
            us += tk * 1e6;
            bytes += b as f64;
        }
        for n in ["expert w1+w3", "expert w2"] {
            let (tk, b) = t1[n];
            us += 7.0 * tk * 1e6;
            bytes += 7.0 * b as f64;
        }
        // the bf16 router [384, 5120] isn't on the fast path; count its bytes at peak
        let router = 384.0 * 5120.0 * 2.0;
        let (us_l, by_l) = (us + router / peak / 1e3, bytes + router);
        let gb = l * by_l / 1e9;
        eprintln!(
            "\nper layer: {:.1} MB, {:.1} µs GEMV (+ {small:.1} µs of small ops, launched ~4× a layer)",
            by_l / 1e6,
            us_l
        );
        eprintln!(
            "per token (40 layers, no head/embed/MTP): {gb:.2} GB; GEMV-only measured {:.0} tok/s",
            1e6 / (l * us_l)
        );
        for (hw, bw) in [("3090", peak), ("H200", 4800.0), ("B300", 8000.0)] {
            eprintln!(
                "{hw}: memory-bound ceiling {:.0} tok/s at 100% of {bw} GB/s; {:.0} tok/s at the measured {:.0}%",
                bw / gb,
                bw / gb * (by_l / us_l / 1e3) / peak,
                100.0 * (by_l / us_l / 1e3) / peak
            );
        }

        // attention at the real head shapes
        eprintln!("\n| attention h 64 d 512 | t | k | exact µs | tiled µs | tiled kv GB/s |\n|---|---|---|---|---|---|");
        for (t, kk) in [(1, 640), (6, 640), (5, 640)] {
            let [q, kv, sink, ix] = attn_case(t, kk);
            let o = DevBuf::alloc(t * 64 * 512 * 4).unwrap();
            let mut run = |exact: bool| {
                k.exact = exact;
                time(&|| {
                    k.sparse_attn(
                        &s,
                        q.ptr,
                        kv.ptr,
                        usize::MAX,
                        0,
                        sink.ptr,
                        ix.ptr,
                        o.ptr,
                        t,
                        64,
                        512,
                        kk,
                        0.04,
                    )
                    .unwrap()
                })
            };
            let (te, tt) = (run(true), run(false));
            k.exact = false;
            eprintln!(
                "| | {t} | {kk} | {:.1} | {:.1} | {:.0} |",
                te * 1e6,
                tt * 1e6,
                (t * kk * 512 * 4) as f64 / tt / 1e9
            );
        }
    }

    /// Real-model attention shapes (64 heads × 512, 1 kv head) with `k` gathered keys, -1
    /// holes and a sink; returns (q, kv, sink, idxs) on the device.
    fn attn_case(t: usize, k: usize) -> [DevBuf; 4] {
        let (h, d, n) = (64, 512, 4096);
        let idxs: Vec<i32> = (0..t * k)
            .map(|i| {
                if i % 37 == 5 {
                    -1
                } else {
                    ((i * 7919 + 13) % n) as i32
                }
            })
            .collect();
        [
            DevBuf::from_slice(&data(t * h * d, 31, 2.0)).unwrap(),
            DevBuf::from_slice(&data(n * d, 32, 1.0)).unwrap(),
            DevBuf::from_slice(&data(h, 33, 1.0)).unwrap(),
            DevBuf::from_slice(&idxs).unwrap(),
        ]
    }

    /// The tiled attention against `ds_sparse_attn` (DS_EXACT=1) at the real head shapes.
    #[test]
    fn tiled_attn_matches_exact() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (h, d) = (64, 512);
        for (t, kk) in [(1, 640), (6, 640), (5, 100), (2, 960)] {
            let [q, kv, sink, ix] = attn_case(t, kk);
            let o = DevBuf::alloc(t * h * d * 4).unwrap();
            // the tiled run reads its keys split in two (window rows, then the rest)
            let run = |k: &Kernels, split: usize| {
                k.sparse_attn(
                    &s,
                    q.ptr,
                    kv.ptr,
                    split,
                    kv.ptr + (split.min(4096) * d * 4) as u64,
                    sink.ptr,
                    ix.ptr,
                    o.ptr,
                    t,
                    h,
                    d,
                    kk,
                    (d as f32).powf(-0.5),
                )
                .unwrap();
                s.sync().unwrap();
                o.read::<f32>(0, t * h * d).unwrap()
            };
            k.exact = true;
            let want = run(&k, usize::MAX);
            k.exact = false;
            k.attn_fd = false;
            k.attn_fuse = false;
            let three = run(&k, 134);
            k.attn_fuse = true;
            let got = run(&k, 134);
            k.attn_fd = true;
            let fd = run(&k, 134);
            // the fused soft+pv is the 3-kernel path bit for bit
            assert!(
                got.iter()
                    .zip(&three)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "attn t {t} k {kk}: ds_attn_spv != soft + pv"
            );
            // Reordered sums flip bf(p) of a few keys by an ulp, and outputs that cancel to
            // near zero then miss by more than an ulp of themselves: ≥ 99% equal, normwise within
            // two ulps of the largest output.
            let (mut eq, mut ulp1) = (0, 0);
            for (a, b) in got.iter().zip(&want) {
                let du = ((a.to_bits() >> 16) as i32 - (b.to_bits() >> 16) as i32).abs();
                eq += (du == 0) as usize;
                ulp1 += (du == 1) as usize;
            }
            let r = rel(&got, &want);
            eprintln!(
                "attn t {t} k {kk}: {:.2}% equal, {:.2}% 1-ulp, normwise rel {r:.2e}",
                100.0 * eq as f64 / got.len() as f64,
                100.0 * ulp1 as f64 / got.len() as f64
            );
            assert!(r < 8e-3, "attn t {t} k {kk}: rel {r}");
            assert!(
                eq as f64 >= 0.9 * got.len() as f64,
                "attn t {t} k {kk}: {eq} equal"
            );
            // the split-K one: p unrounded (no global max to round against), so the outputs that
            // cancel to near zero miss by a few of their own ulps: the same normwise error as
            // the tiled path, ≥ 85% within an ulp
            let eq = fd
                .iter()
                .zip(&want)
                .filter(|(a, b)| {
                    ((a.to_bits() >> 16) as i32 - (b.to_bits() >> 16) as i32).abs() <= 1
                })
                .count();
            let r = rel(&fd, &want);
            eprintln!(
                "attn fd t {t} k {kk}: {:.2}% within 1 ulp, normwise rel {r:.2e}",
                100.0 * eq as f64 / fd.len() as f64
            );
            assert!(r < 8e-3, "attn fd t {t} k {kk}: rel {r}");
            assert!(
                eq as f64 >= 0.85 * fd.len() as f64,
                "attn fd t {t} k {kk}: {eq} within 1 ulp"
            );
        }
    }

    /// The rope fused into the tiled attention (`sparse_attn_r`) is ds_rope on q, the
    /// attention, then ds_rope inverse on o, bit for bit, on both tiled paths.
    #[test]
    fn tiled_attn_rope_fused_matches() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (h, d, rd, p0) = (64, 512, 64, 37usize);
        let tabv: Vec<f32> = (0..256 * rd / 2)
            .flat_map(|e| {
                let (pos, j) = ((e / (rd / 2)) as f32, (e % (rd / 2)) as f32);
                let th = pos * 10000f32.powf(-2.0 * j / rd as f32);
                [th.cos(), th.sin()]
            })
            .collect();
        let tab = DevBuf::from_slice(&tabv).unwrap();
        let pbuf = DevBuf::from_slice(&[p0 as i32]).unwrap();
        for (t, kk, add, st) in [
            (1, 640, 0, 1),
            (6, 640, 0, 1),
            (5, 100, 3, 1),
            (2, 960, 1, 2),
        ] {
            for fuse in [false, true] {
                k.attn_fuse = fuse;
                let [q, kv, sink, ix] = attn_case(t, kk);
                let o = DevBuf::alloc(t * h * d * 4).unwrap();
                let scale = (d as f32).powf(-0.5);
                let split = 134;
                let kv1 = kv.ptr + (split * d * 4) as u64;
                let q0 = q.read::<f32>(0, t * h * d).unwrap();
                let rope = AttnRope {
                    tab: tab.ptr,
                    p: pbuf.ptr,
                    rd: rd as i32,
                    add: add as i32,
                    stride: st as i32,
                    q: true,
                    wo_quant: false,
                };
                k.sparse_attn_r(
                    &s,
                    q.ptr,
                    kv.ptr,
                    split,
                    kv1,
                    sink.ptr,
                    ix.ptr,
                    o.ptr,
                    t,
                    h,
                    d,
                    kk,
                    scale,
                    Some(rope),
                )
                .unwrap();
                s.sync().unwrap();
                let got = o.read::<f32>(0, t * h * d).unwrap();
                // q untouched by the fused path
                assert_eq!(q.read::<f32>(0, t * h * d).unwrap(), q0);
                k.rope(&s, q.ptr, tab.ptr, t, h, d, rd, p0 + add, st, None, false)
                    .unwrap();
                k.sparse_attn(
                    &s, q.ptr, kv.ptr, split, kv1, sink.ptr, ix.ptr, o.ptr, t, h, d, kk, scale,
                )
                .unwrap();
                k.rope(&s, o.ptr, tab.ptr, t, h, d, rd, p0 + add, st, None, true)
                    .unwrap();
                s.sync().unwrap();
                let want = o.read::<f32>(0, t * h * d).unwrap();
                let bad = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(bad, 0, "t {t} k {kk} fuse {fuse}: {bad} differ");
            }
        }
    }

    /// `ds_attn_fdm` building its own index rows (`attn_fd_win`) is `sparse_attn_r` on the
    /// rows dg_win_idx + ds_idx_cat build, bit for bit (decode and verify-window modes).
    #[test]
    fn attn_fd_win_matches_rows() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (h, d, rd, win, kc) = (64, 512, 64, 128usize, 512usize);
        if !k.attn_fd_win_ok(h, d, win + kc) {
            return;
        }
        let tabv: Vec<f32> = (0..2048 * rd / 2)
            .flat_map(|e| {
                let (pos, j) = ((e / (rd / 2)) as f32, (e % (rd / 2)) as f32);
                let th = pos * 10000f32.powf(-2.0 * j / rd as f32);
                [th.cos(), th.sin()]
            })
            .collect();
        let tab = DevBuf::from_slice(&tabv).unwrap();
        for (mode, t, sp, kc) in [
            (1, 1, 37usize, kc),
            (1, 1, 300, 0),
            (3, 6, 20, kc),
            (3, 5, 900, kc),
        ] {
            let k_all = win + kc;
            let [q, kv, sink, _] = attn_case(t, k_all);
            let pbuf = DevBuf::from_slice(&[sp as i32]).unwrap();
            let src: Vec<i32> = (0..t * kc.max(1))
                .map(|i| ((i * 31 + 7) % 3000) as i32 + 200)
                .collect();
            let srcb = DevBuf::from_slice(&src).unwrap();
            // the rows on the host, as dg_win_idx (modes 1 / 3) then ds_idx_cat
            let mut ix = vec![0i32; t * k_all];
            for r in 0..t {
                for j in 0..win {
                    ix[r * k_all + j] = if mode == 1 {
                        let i = (sp % win + 1 + j) % win;
                        if i > sp {
                            -1
                        } else {
                            i as i32
                        }
                    } else {
                        let qq = sp + r;
                        let cnt = win.min(qq + 1);
                        let pos = qq + 1 - cnt + j;
                        if j >= cnt {
                            -1
                        } else if pos < sp {
                            (pos % win) as i32
                        } else {
                            (win + pos - sp) as i32
                        }
                    };
                }
                for j in 0..kc {
                    ix[r * k_all + win + j] = src[r * kc + j];
                }
            }
            let ixb = DevBuf::from_slice(&ix).unwrap();
            let split = if mode == 1 { win } else { win + t };
            let kv1 = kv.ptr + (split * d * 4) as u64;
            let rope = AttnRope {
                tab: tab.ptr,
                p: pbuf.ptr,
                rd: rd as i32,
                add: 0,
                stride: 1,
                q: true,
                wo_quant: false,
            };
            let scale = (d as f32).powf(-0.5);
            let o = DevBuf::alloc(t * h * d * 4).unwrap();
            k.sparse_attn_r(
                &s,
                q.ptr,
                kv.ptr,
                split,
                kv1,
                sink.ptr,
                ixb.ptr,
                o.ptr,
                t,
                h,
                d,
                k_all,
                scale,
                Some(rope),
            )
            .unwrap();
            s.sync().unwrap();
            let want = o.read::<f32>(0, t * h * d).unwrap();
            let w = WinIdx {
                mode,
                win: win as i32,
                src: srcb.ptr,
                kc: kc as i32,
            };
            k.attn_fd_win(
                &s, q.ptr, kv.ptr, split, kv1, sink.ptr, o.ptr, t, h, d, k_all, scale, rope, w,
            )
            .unwrap();
            s.sync().unwrap();
            let got = o.read::<f32>(0, t * h * d).unwrap();
            let bad = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(bad, 0, "mode {mode} t {t} sp {sp} kc {kc}: {bad} differ");
            // the merge's quantize of o (wo_quant) is ds_ks_quant's, bit for bit
            let rq = AttnRope {
                wo_quant: true,
                ..rope
            };
            k.attn_fd_win(
                &s, q.ptr, kv.ptr, split, kv1, sink.ptr, o.ptr, t, h, d, k_all, scale, rq, w,
            )
            .unwrap();
            s.sync().unwrap();
            let n = t * h * d;
            let got_c = k.wo_x.read::<u8>(0, n).unwrap();
            let got_s = k.wo_x.read::<f32>(8 * WO_X_COLS, n / 32).unwrap();
            let qb = DevBuf::alloc(n + n / 8).unwrap();
            k.ks_quant_to(&s, o.ptr, qb.ptr, qb.ptr + n as u64, n)
                .unwrap();
            s.sync().unwrap();
            assert_eq!(got_c, qb.read::<u8>(0, n).unwrap(), "codes");
            let ws = qb.read::<f32>(n, n / 32).unwrap();
            assert!(
                got_s
                    .iter()
                    .zip(&ws)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "scales"
            );
        }
    }

    /// `gemv_bf16` (a warp per row) against `linear` (ds_linear_t8 / ds_linear) on bf16 weights:
    /// summation order only. With `DS_BENCH_GEMV=1` also times both at the compressor shapes.
    #[test]
    fn gemv_bf16_matches_linear() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        for (rows, cols) in [(1024usize, 5120usize), (512, 4096), (64, 5120), (100, 2048)] {
            let wb: Vec<u16> = data(rows * cols, 7, 0.05)
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect();
            let wbuf = DevBuf::from_slice(&wb).unwrap();
            let w = DsW {
                w: wbuf.ptr,
                s: 0,
                rows: rows as i32,
                cols: cols as i32,
                fmt: 1,
                _pad: 0,
            };
            for t in [1usize, 6, 8] {
                let x = DevBuf::from_slice(&data(t * cols, 3, 2.0)).unwrap();
                let (y0, y1) = (
                    DevBuf::alloc(t * rows * 4).unwrap(),
                    DevBuf::alloc(t * rows * 4).unwrap(),
                );
                k.linear(&s, &w, x.ptr, y0.ptr, t, false).unwrap();
                k.gemv_bf16(&s, &w, x.ptr, y1.ptr, t).unwrap();
                s.sync().unwrap();
                let (a, b) = (
                    y0.read::<f32>(0, t * rows).unwrap(),
                    y1.read::<f32>(0, t * rows).unwrap(),
                );
                let r = rel(&b, &a);
                assert!(r < 1e-5, "rows {rows} cols {cols} t {t}: rel {r}");
                if std::env::var("DS_BENCH_GEMV").is_ok_and(|v| v == "1") {
                    let ev = |f: &dyn Fn()| {
                        f();
                        s.sync().unwrap();
                        let (e0, e1) = (
                            gpu::Event::new(true).unwrap(),
                            gpu::Event::new(true).unwrap(),
                        );
                        e0.record(&s).unwrap();
                        for _ in 0..50 {
                            f();
                        }
                        e1.record(&s).unwrap();
                        e1.sync().unwrap();
                        e1.since(&e0).unwrap() as f64 * 1e3 / 50.0
                    };
                    let u0 = ev(&|| k.linear(&s, &w, x.ptr, y0.ptr, t, false).unwrap());
                    let u1 = ev(&|| k.gemv_bf16(&s, &w, x.ptr, y1.ptr, t).unwrap());
                    let gb = (rows * cols * 2) as f64 / 1e3;
                    eprintln!(
                        "gemv_bf16 rows {rows} cols {cols} t {t}: linear {u0:.1} us ({:.0} GB/s), r1 {u1:.1} us ({:.0} GB/s)",
                        gb / u0,
                        gb / u1
                    );
                }
            }
        }
    }

    /// Engram's row gather: today's host decode + upload against `ds_engram_gather` reading
    /// a host-registered table over UVA (bit for bit), timed; `DS_EG_GB` sizes the table
    /// (default 8). `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn engram_gather_uva() {
        use cudarc::driver::sys;
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let gb: usize = std::env::var("DS_EG_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let ehd = 256usize;
        let nrow = (gb << 30) / ehd;
        let mut w = vec![0u8; nrow * ehd];
        for (i, x) in w.iter_mut().enumerate() {
            *x = ((i as u64).wrapping_mul(2654435761) >> 7) as u8;
        }
        let sc: Vec<u8> = (0..nrow * ehd / 32).map(|i| 110 + (i % 30) as u8).collect();
        let reg = |p: *const u8, n: usize| -> u64 {
            let mut d = 0u64;
            unsafe {
                gpu::check(
                    sys::cuMemHostRegister_v2(
                        p as *mut _,
                        n,
                        sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_READ_ONLY,
                    ),
                    "cuMemHostRegister",
                )
                .unwrap();
                gpu::check(
                    sys::cuMemHostGetDevicePointer_v2(&mut d, p as *mut _, 0),
                    "cuMemHostGetDevicePointer",
                )
                .unwrap();
            }
            d
        };
        let (dw, ds) = (reg(w.as_ptr(), w.len()), reg(sc.as_ptr(), sc.len()));
        let lut = DevBuf::from_slice(&super::super::quant::e4m3_lut()[..]).unwrap();
        let lut_h = super::super::quant::e4m3_lut();
        let n = 336usize;
        let mut seed = 99u64;
        for rep in 0..5 {
            let rows: Vec<i64> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 17) as usize % nrow) as i64
                })
                .collect();
            // today's path: host decode (table-driven), then the upload
            let t0 = std::time::Instant::now();
            let mut host = Vec::with_capacity(n * ehd);
            for &r in &rows {
                let r = r as usize;
                let (wr, sr) = (&w[r * ehd..(r + 1) * ehd], &sc[r * (ehd / 32)..]);
                for (g, ch) in wr.chunks(32).enumerate() {
                    let f = super::super::quant::e8m0_to_f32(sr[g]);
                    host.extend(ch.iter().map(|&b| lut_h[b as usize] * f));
                }
            }
            let hb = DevBuf::from_slice(&host).unwrap();
            s.sync().unwrap();
            let t_host = t0.elapsed().as_secs_f64() * 1e6;
            // the device gather: the row indices up, one kernel
            let out = DevBuf::alloc(n * ehd * 4).unwrap();
            let t1 = std::time::Instant::now();
            let rb = DevBuf::from_slice(&rows).unwrap();
            let (ni, ei) = (n as i32, ehd as i32);
            unsafe {
                k.go(
                    "ds_engram_gather",
                    (n as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &s,
                    tang_moe::args![lut.ptr, dw, ds, rb.ptr, ni, ei, out.ptr],
                )
                .unwrap();
            }
            s.sync().unwrap();
            let t_dev = t1.elapsed().as_secs_f64() * 1e6;
            let (a, b) = (
                hb.read::<f32>(0, n * ehd).unwrap(),
                out.read::<f32>(0, n * ehd).unwrap(),
            );
            assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
            eprintln!("rep {rep}: {n} rows of {gb} GB: host gather + upload {t_host:.0} us, device gather (UVA) {t_dev:.0} us");
        }
        unsafe {
            sys::cuMemHostUnregister(w.as_ptr() as *mut _);
            sys::cuMemHostUnregister(sc.as_ptr() as *mut _);
        }
    }

    /// `ks_gemv2` (two weights, one launch) is two `ks_gemv`s bit for bit, at the wq_a / wkv
    /// shapes; with `DS_BENCH_GEMV=1` also times both.
    #[test]
    fn ks_gemv2_matches_two() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let cols = 4096usize;
        let mk = |rows: usize, seed: u64| {
            let codes: Vec<u8> = (0..rows * cols)
                .map(|i| ((i as u64 * 2654435761 + seed) % 255) as u8 & 0x7e)
                .collect();
            let sc: Vec<u8> = (0..(rows / 32) * (cols / 32))
                .map(|i| 120 + (i % 7) as u8)
                .collect();
            let (wb, sb) = (
                DevBuf::from_slice(&codes).unwrap(),
                DevBuf::from_slice(&sc).unwrap(),
            );
            let w = DsW {
                w: wb.ptr,
                s: sb.ptr,
                rows: rows as i32,
                cols: cols as i32,
                fmt: 2,
                _pad: 0,
            };
            (w, wb, sb)
        };
        let (wa, _a0, _a1) = mk(1024, 1);
        let (wk, _b0, _b1) = mk(512, 2);
        if !k.ks_ok(&wa) || !k.ks_ok(&wk) {
            return;
        }
        for t in [1usize, 6, 7, 8] {
            let x = DevBuf::from_slice(&data(t * cols, 3, 2.0)).unwrap();
            let (xc, xs) = k.ks_x_ptrs();
            k.ks_quant_to(&s, x.ptr, xc, xs, t * cols).unwrap();
            let ys: Vec<DevBuf> = (0..4)
                .map(|i| DevBuf::alloc(t * if i % 2 == 0 { 1024 } else { 512 } * 4).unwrap())
                .collect();
            k.ks_gemv(&s, &wa, None, xc, xs, ys[0].ptr, t, true, 0, 0.0)
                .unwrap();
            k.ks_gemv(&s, &wk, None, xc, xs, ys[1].ptr, t, true, 0, 0.0)
                .unwrap();
            k.ks_gemv2(&s, &wa, &wk, xc, xs, ys[2].ptr, ys[3].ptr, t, true)
                .unwrap();
            s.sync().unwrap();
            for (i, n) in [(0usize, 1024usize), (1, 512)] {
                let (a, b) = (
                    ys[i].read::<f32>(0, t * n).unwrap(),
                    ys[i + 2].read::<f32>(0, t * n).unwrap(),
                );
                assert!(
                    a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
                    "t {t} out {i}"
                );
            }
            if std::env::var("DS_BENCH_GEMV").is_ok_and(|v| v == "1") {
                let ev = |f: &dyn Fn()| {
                    f();
                    s.sync().unwrap();
                    let (e0, e1) = (
                        gpu::Event::new(true).unwrap(),
                        gpu::Event::new(true).unwrap(),
                    );
                    e0.record(&s).unwrap();
                    for _ in 0..50 {
                        f();
                    }
                    e1.record(&s).unwrap();
                    e1.sync().unwrap();
                    e1.since(&e0).unwrap() as f64 * 1e3 / 50.0
                };
                let u0 = ev(&|| {
                    k.ks_gemv(&s, &wa, None, xc, xs, ys[0].ptr, t, true, 0, 0.0)
                        .unwrap();
                    k.ks_gemv(&s, &wk, None, xc, xs, ys[1].ptr, t, true, 0, 0.0)
                        .unwrap();
                });
                let u1 = ev(&|| {
                    k.ks_gemv2(&s, &wa, &wk, xc, xs, ys[2].ptr, ys[3].ptr, t, true)
                        .unwrap()
                });
                eprintln!("ks_gemv2 t {t}: two launches {u0:.1} us, one {u1:.1} us");
            }
        }
    }

    /// Scores with heavy ties (few levels, ±0, -inf) so tie-breaking is exercised.
    fn tied(n: usize, seed: u64) -> Vec<f32> {
        data(n, seed, 1.0)
            .into_iter()
            .enumerate()
            .map(|(i, x)| match i % 11 {
                0 => f32::NEG_INFINITY,
                1 => -0.0,
                2 => 0.0,
                _ => (x * 4.0).round() / 4.0,
            })
            .collect()
    }

    fn lens(t: usize, width: usize, causal: bool, r: usize) -> Vec<usize> {
        (0..t)
            .map(|ti| if causal { (ti + 1) / r } else { width })
            .collect()
    }

    /// Logits shaped like a model's: bf16 noise of amplitude `amp`, `peaks` boosted tokens.
    fn peaky(v: usize, seed: u64, amp: f32, peaks: usize) -> Vec<f32> {
        let mut l = data(v, seed, amp);
        for j in 0..peaks {
            let i = (seed as usize * 7919 + j * 104729) % v;
            l[i] = amp * 3.0 + (j % 4) as f32 * 0.25;
        }
        l
    }

    /// `ds_spark_sample` against `SpecSampler::sample` on the real vocab size: the same draw
    /// and RNG state, and the kept distribution it leaves for the device verify equal to
    /// `SpecSampler::dist` (ids exactly, probabilities to f64 rounding). Covers the fast
    /// bucketed path and the full-vocab fallback (a flat row's nucleus exceeds the cap).
    #[test]
    fn spark_sample_matches_host() {
        use super::super::forward::SpecSampler;
        use tang_moe::args;
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let v = 129280usize;
        let cap = 2048usize;
        let (ids, w) = (DevBuf::zeroed(8).unwrap(), DevBuf::alloc(v * 8).unwrap());
        let (qi, qp, qn) = (
            DevBuf::alloc(cap * 4).unwrap(),
            DevBuf::alloc(cap * 8).unwrap(),
            DevBuf::zeroed(4).unwrap(),
        );
        let (mut fast, mut full) = (0, 0);
        let (mut ms, mut fms) = (0.0, 0.0);
        for trial in 0..120u64 {
            let (temp, top_p) =
                [(0.6f32, 0.95f32), (1.0, 0.9), (0.3, 0.99), (0.6, 1.0)][trial as usize % 4];
            let amp = [1.0f32, 8.0, 16.0, 30.0][(trial / 4) as usize % 4];
            let l = peaky(v, trial + 1, amp, 1 + (trial as usize % 40));
            let row = DevBuf::from_slice(&l).unwrap();
            let rng = DevBuf::from_slice(&[trial * 977 + 5]).unwrap();
            let (vi, slot, null, ri) = (v as i32, 1i32, 0u64, 0i32);
            let t = std::time::Instant::now();
            unsafe {
                k.go(
                    "ds_spark_sample",
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    &s,
                    args![
                        row.ptr, vi, ids.ptr, slot, null, ri, null, temp, top_p, rng.ptr, w.ptr,
                        null, qi.ptr, qp.ptr, qn.ptr
                    ],
                )
                .unwrap()
            };
            s.sync().unwrap();
            let kt = t.elapsed().as_secs_f64() * 1e3;
            ms += kt;
            let mut host = SpecSampler::new(temp, top_p, trial * 977 + 5);
            let want = host.sample(&l);
            let got = ids.read::<i32>(4, 1).unwrap()[0] as usize;
            assert_eq!(got, want, "trial {trial} amp {amp} T {temp} p {top_p}");
            assert_eq!(rng.read::<u64>(0, 1).unwrap()[0], host.rng, "trial {trial}");
            let n = qn.read::<i32>(0, 1).unwrap()[0] as usize;
            if n == 0 {
                full += 1;
                continue;
            }
            fast += 1;
            fms += kt;
            let d = SpecSampler::new(temp, top_p, 0).dist(&l);
            assert_eq!(n, d.len(), "trial {trial} kept");
            let (gi, gp) = (qi.read::<i32>(0, n).unwrap(), qp.read::<f64>(0, n).unwrap());
            for (j, &(i, p)) in d.iter().enumerate() {
                assert_eq!(gi[j] as usize, i, "trial {trial} entry {j}");
                assert!(
                    (gp[j] - p).abs() <= 1e-12 * p.max(1e-300),
                    "trial {trial} p {j}"
                );
            }
        }
        eprintln!(
            "spark_sample: {fast} fast ({:.3} ms/draw), {full} full-vocab ({:.3} ms/draw)",
            fms / fast.max(1) as f64,
            (ms - fms) / full.max(1) as f64,
        );
        assert!(fast > 40 && full > 0, "{fast} fast {full} full");
    }

    /// The device speculative-sampling walk (`ds_spec_dist` + `ds_spec_walk`) against the
    /// host's `verify_rows` with a `SpecSampler`: drafts drawn on the device from Q rows,
    /// target rows P near Q (so walks accept, reject and reach the bonus row), and also
    /// forced wrong drafts. Same accepted count, tokens and RNG state; and over many walks
    /// the emitted first token's frequencies match a plain draw from P (chi-square).
    #[test]
    fn spec_walk_matches_host() {
        use super::super::forward::{verify_rows, SpecSampler};
        use tang_moe::args;
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (v, blk, cap) = (129280usize, 5usize, 2048usize);
        let n = blk + 1;
        let null = 0u64;
        let (qi, qp, qn) = (
            DevBuf::alloc(blk * cap * 4).unwrap(),
            DevBuf::alloc(blk * cap * 8).unwrap(),
            DevBuf::zeroed(blk * 4).unwrap(),
        );
        let (pi, pp, pn) = (
            DevBuf::alloc(n * cap * 4).unwrap(),
            DevBuf::alloc(n * cap * 8).unwrap(),
            DevBuf::zeroed(n * 4).unwrap(),
        );
        let (w, pw) = (
            DevBuf::alloc(v * 8).unwrap(),
            DevBuf::alloc(n * v * 8).unwrap(),
        );
        let out = DevBuf::alloc((n + 4) * 4).unwrap();
        let dids = DevBuf::zeroed((n + 1) * 4).unwrap();
        let (mut walks, mut acc, mut ms) = (0, 0, 0.0);
        for trial in 0..60u64 {
            let (temp, top_p) = [(0.6f32, 0.95f32), (1.0, 0.9)][trial as usize % 2];
            let amp = [12.0f32, 20.0][(trial / 2) as usize % 2];
            let peaks = 2 + trial as usize % 12;
            // Q rows, and P = Q plus a perturbation
            let q: Vec<f32> = (0..blk)
                .flat_map(|r| peaky(v, trial * 31 + r as u64 + 1, amp, peaks))
                .collect();
            let p: Vec<f32> = (0..n)
                .flat_map(|r| {
                    let base = if r < blk {
                        q[r * v..(r + 1) * v].to_vec()
                    } else {
                        peaky(v, trial * 31 + 99, amp, peaks)
                    };
                    let e = data(v, trial * 131 + r as u64, 1.5);
                    base.iter().zip(e).map(|(a, b)| a + b).collect::<Vec<_>>()
                })
                .collect();
            let dq = DevBuf::from_slice(&q).unwrap();
            let dp = DevBuf::from_slice(&p).unwrap();
            let seed = trial * 7 + 3;
            let rng = DevBuf::from_slice(&[seed]).unwrap();
            let vi = v as i32;
            for r in 0..blk {
                let (row, slot, ri) = (dq.ptr + (r * v * 4) as u64, r as i32 + 1, 0i32);
                unsafe {
                    k.go(
                        "ds_spark_sample",
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        &s,
                        args![
                            row, vi, dids.ptr, slot, null, ri, null, temp, top_p, rng.ptr, w.ptr,
                            null, qi.ptr, qp.ptr, qn.ptr
                        ],
                    )
                    .unwrap()
                };
            }
            s.sync().unwrap();
            let mut host = SpecSampler::new(temp, top_p, seed);
            let mut ids: Vec<usize> = vec![7];
            for r in 0..blk {
                ids.push(host.sample(&q[r * v..(r + 1) * v]));
            }
            let got: Vec<usize> = dids
                .read::<i32>(4, blk)
                .unwrap()
                .iter()
                .map(|&x| x as usize)
                .collect();
            assert_eq!(got, ids[1..], "trial {trial} drafts");
            assert_eq!(rng.read::<u64>(0, 1).unwrap()[0], host.rng);
            for force in [false, true] {
                let mut ids = ids.clone();
                if force {
                    ids[1 + trial as usize % blk] = (trial as usize * 4099) % v;
                }
                let r0 = host.rng;
                let mut h = SpecSampler::new(temp, top_p, r0);
                let (wk, wt) = verify_rows(&p, &q, &ids, v, &mut h);
                let di =
                    DevBuf::from_slice(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>()).unwrap();
                let ni = n as i32;
                let t = std::time::Instant::now();
                unsafe {
                    k.go(
                        "ds_spec_dist",
                        (n as u32, 1, 1),
                        (1024, 1, 1),
                        0,
                        &s,
                        args![dp.ptr, vi, temp, top_p, pw.ptr, pi.ptr, pp.ptr, pn.ptr],
                    )
                    .unwrap();
                    k.go(
                        "ds_spec_walk",
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        &s,
                        args![
                            di.ptr, ni, pi.ptr, pp.ptr, pn.ptr, qi.ptr, qp.ptr, qn.ptr, r0, out.ptr
                        ],
                    )
                    .unwrap();
                }
                s.sync().unwrap();
                ms += t.elapsed().as_secs_f64() * 1e3;
                let o = out.read::<i32>(0, n + 4).unwrap();
                assert_eq!(
                    o[0],
                    1,
                    "trial {trial}: no device walk, pn {:?} qn {:?}",
                    pn.read::<i32>(0, n).unwrap(),
                    qn.read::<i32>(0, blk).unwrap()
                );
                let gk = o[1] as usize;
                let gt: Vec<usize> = o[2..3 + gk].iter().map(|&x| x as usize).collect();
                let grng = o[2 + n] as u32 as u64 | (o[3 + n] as u32 as u64) << 32;
                assert_eq!((gk, &gt), (wk, &wt), "trial {trial} force {force}");
                assert_eq!(grng, h.rng, "trial {trial} force {force} rng");
                walks += 1;
                acc += gk;
            }
        }
        eprintln!(
            "spec_walk: {walks} walks, {acc} accepted, {:.3} ms/walk (dist + walk)",
            ms / walks as f64
        );
        assert!(acc > 0);
    }

    #[test]
    fn select_kernels_match_cpu() {
        let Some((_g, k, s, m)) = setup() else { return };
        // mask + top-k
        for (t, width, kk, causal, r, off) in [
            (1, 1, 1, false, 1, 0),
            (3, 7, 6, false, 1, 8),
            (16, 8, 6, true, 2, 16),
            (5, 600, 64, true, 1, 3),
            (2, 5000, 512, false, 4, 128),
            (1, 16384, 512, false, 1, 128),
        ] {
            let x = tied(t * width, (width + kk) as u64);
            let ls = lens(t, width, causal, r);
            let mut want_sc = x.clone();
            if causal {
                for ti in 0..t {
                    for j in ls[ti]..width {
                        want_sc[ti * width + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let dx = DevBuf::from_slice(&x).unwrap();
            k.mask_scores(&s, dx.ptr, 0, t, width, causal, r).unwrap();
            let out = DevBuf::alloc(t * kk * 4).unwrap();
            k.topk(&s, dx.ptr, out.ptr, t, width, kk, kk, off, causal, r)
                .unwrap();
            s.sync().unwrap();
            let got_sc = dx.read::<f32>(0, t * width).unwrap();
            assert!(
                got_sc
                    .iter()
                    .zip(&want_sc)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "mask t{t} w{width}"
            );
            let got = out.read::<i32>(0, t * kk).unwrap();
            for ti in 0..t {
                let mut sel = forward::topk_stable(&want_sc[ti * width..(ti + 1) * width], kk);
                sel.sort();
                let want: Vec<i32> = sel
                    .into_iter()
                    .map(|i| if i < ls[ti] { (i + off) as i32 } else { -1 })
                    .collect();
                assert_eq!(
                    &got[ti * kk..(ti + 1) * kk],
                    &want[..],
                    "topk t{t} w{width} row {ti}"
                );
            }
        }
        // candidate blocks, then the candidate mask
        let (bs, kb) = (m.cfg.candidate_block_size, m.cfg.candidate_topk_blocks);
        for (t, width, causal, r) in [
            (16, 16, true, 1),
            (1, 17, false, 1),
            (12, 6, true, 2),
            (1, 3000, false, 1),
        ] {
            let ls = lens(t, width, causal, r);
            let mut x = tied(t * width, width as u64);
            if causal {
                for ti in 0..t {
                    for j in ls[ti]..width {
                        x[ti * width + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let want = m.candidates(&x, width, &ls);
            let dx = DevBuf::from_slice(&x).unwrap();
            let cand = DevBuf::alloc(t * width).unwrap();
            k.candidates(&s, dx.ptr, cand.ptr, t, width, causal, r, bs, kb)
                .unwrap();
            let got = cand.read::<u8>(0, t * width).unwrap();
            let flat: Vec<u8> = want.iter().flatten().map(|&b| b as u8).collect();
            assert_eq!(got, flat, "candidates t{t} w{width}");
            let y = tied(t * width, 99);
            let dy = DevBuf::from_slice(&y).unwrap();
            k.mask_scores(&s, dy.ptr, cand.ptr, t, width, false, r)
                .unwrap();
            s.sync().unwrap();
            let got = dy.read::<f32>(0, t * width).unwrap();
            for (i, (&g, &v)) in got.iter().zip(&y).enumerate() {
                let w = if flat[i] == 1 { v } else { f32::NEG_INFINITY };
                assert_eq!(g.to_bits(), w.to_bits(), "cand mask {i}");
            }
        }
        // routing lists
        for (t, kk, ne) in [(1, 2, 8), (37, 6, 64), (300, 8, 256)] {
            let mut ids = Vec::new();
            let mut wts = Vec::new();
            for ti in 0..t {
                // distinct per token: 3 is coprime to every ne here
                let used: Vec<usize> = (0..kk).map(|j| (ti * 7 + j * 3) % ne).collect();
                for (j, &e) in used.iter().enumerate() {
                    ids.push(e as i32);
                    wts.push((ti * 10 + j) as f32);
                }
            }
            let (di, dw) = (
                DevBuf::from_slice(&ids).unwrap(),
                DevBuf::from_slice(&wts).unwrap(),
            );
            let counts = DevBuf::alloc(ne * 4).unwrap();
            let off = DevBuf::alloc((ne + 1) * 4).unwrap();
            let rows = DevBuf::alloc(t * kk * 4).unwrap();
            let rw = DevBuf::alloc(t * kk * 4).unwrap();
            k.route(
                &s, di.ptr, dw.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, kk, ne,
            )
            .unwrap();
            s.sync().unwrap();
            let mut routes: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ne];
            for ti in 0..t {
                for j in 0..kk {
                    routes[ids[ti * kk + j] as usize].push((ti as i32, wts[ti * kk + j]));
                }
            }
            let (gc, go, gr, gw) = (
                counts.read::<i32>(0, ne).unwrap(),
                off.read::<i32>(0, ne + 1).unwrap(),
                rows.read::<i32>(0, t * kk).unwrap(),
                rw.read::<f32>(0, t * kk).unwrap(),
            );
            for e in 0..ne {
                assert_eq!(gc[e] as usize, routes[e].len());
                let o = go[e] as usize;
                assert_eq!(go[e + 1] as usize, o + routes[e].len());
                for (q, &(r, w)) in routes[e].iter().enumerate() {
                    assert_eq!((gr[o + q], gw[o + q]), (r, w), "route e{e}");
                }
            }
        }
    }

    /// The MoE router on the real shape (manual: `--ignored --nocapture router_bench`):
    /// bf16 gate [384, 5120] GEMV, `ds_router`, `route`, each timed alone.
    #[test]
    #[ignore]
    fn router_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (ne, topk, d, t) = (384usize, 6usize, 5120usize, 6usize);
        let wb: Vec<u16> = data(ne * d, 5, 0.05)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let wdev = DevBuf::from_slice(&wb).unwrap();
        let w = DsW {
            w: wdev.ptr,
            s: 0,
            rows: ne as i32,
            cols: d as i32,
            fmt: 1,
            _pad: 0,
        };
        let x = DevBuf::from_slice(&data(t * d, 7, 1.0)).unwrap();
        let bias = DevBuf::from_slice(&data(ne, 9, 0.1)).unwrap();
        let raw = DevBuf::alloc(t * ne * 4).unwrap();
        let (ids, wts) = (
            DevBuf::alloc(t * topk * 4).unwrap(),
            DevBuf::alloc(t * topk * 4).unwrap(),
        );
        let (counts, off) = (
            DevBuf::alloc(ne * 4).unwrap(),
            DevBuf::alloc((ne + 1) * 4).unwrap(),
        );
        let (rows, rw) = (
            DevBuf::alloc(t * topk * 4).unwrap(),
            DevBuf::alloc(t * topk * 4).unwrap(),
        );
        let time = |name: &str, f: &dyn Fn()| {
            f();
            s.sync().unwrap();
            let iters = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            s.sync().unwrap();
            let us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
            eprintln!("router_bench {name}: {us:.1} us");
        };
        time("gate", &|| {
            k.linear(&s, &w, x.ptr, raw.ptr, t, false).unwrap();
        });
        time("gate_gemv", &|| {
            k.gemv_bf16(&s, &w, x.ptr, raw.ptr, t).unwrap();
        });
        time("ds_router", &|| {
            k.router(
                &s, raw.ptr, bias.ptr, ids.ptr, wts.ptr, t, ne, topk, true, 1.5,
            )
            .unwrap();
        });
        time("route", &|| {
            k.route(
                &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
            )
            .unwrap();
        });
        time("gate+router+route (3 launches)", &|| {
            k.linear(&s, &w, x.ptr, raw.ptr, t, false).unwrap();
            k.router(
                &s, raw.ptr, bias.ptr, ids.ptr, wts.ptr, t, ne, topk, true, 1.5,
            )
            .unwrap();
            k.route(
                &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
            )
            .unwrap();
        });
        let mask = DevBuf::from_slice(&vec![1i32; ne]).unwrap();
        let elist = DevBuf::alloc(t * topk * 4).unwrap();
        time("ds_moe_plan", &|| unsafe {
            let (nei, ci) = (ne as i32, (t * topk) as i32);
            k.go(
                "ds_moe_plan",
                (1, 1, 1),
                (32, 1, 1),
                0,
                &s,
                tang_moe::args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
            )
            .unwrap();
        });
        let rpq = || {
            k.route_plan_q(
                &s,
                (ids.ptr, wts.ptr),
                (counts.ptr, off.ptr, rows.ptr, rw.ptr),
                (mask.ptr, elist.ptr, t * topk),
                x.ptr,
                (t, topk, ne, d),
            )
            .unwrap();
        };
        time("route_plan_q", &rpq);
        time("gemv + router + route_plan_q (default path)", &|| {
            k.gemv_bf16(&s, &w, x.ptr, raw.ptr, t).unwrap();
            k.router(
                &s, raw.ptr, bias.ptr, ids.ptr, wts.ptr, t, ne, topk, true, 1.5,
            )
            .unwrap();
            rpq();
        });
        time("ds_moe_gate (fused, + plan)", &|| {
            k.moe_gate(
                &s,
                &w,
                x.ptr,
                raw.ptr,
                bias.ptr,
                ids.ptr,
                wts.ptr,
                (counts.ptr, off.ptr, rows.ptr, rw.ptr),
                (mask.ptr, elist.ptr, t * topk),
                t,
                topk,
                true,
                1.5,
            )
            .unwrap();
        });
    }

    /// `ds_moe_gate` (gate GEMV + router + route + plan in one launch) against the separate
    /// `linear`, `ds_router`, `ds_route` and `ds_moe_plan` launches, bit for bit.
    #[test]
    fn moe_gate_matches_separate() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let d = 512usize;
        for (t, ne, kk, norm, coarse) in [
            (1, 8, 1, true, false),
            (1, 384, 6, true, false),
            (6, 384, 6, true, true),
            (6, 384, 6, true, false),
            (5, 256, 8, false, true),
            (8, 512, 16, true, false),
            (3, 100, 4, true, true),
        ] {
            let mut wf = data(ne * d, (ne * 3 + t) as u64, 0.05);
            if coarse {
                // many equal logits: rows share weights
                for e in 0..ne {
                    let src = (e % 7) * d;
                    for c in 0..d {
                        wf[e * d + c] = wf[src + c];
                    }
                }
            }
            let wb: Vec<u16> = wf.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let wdev = DevBuf::from_slice(&wb).unwrap();
            let w = DsW {
                w: wdev.ptr,
                s: 0,
                rows: ne as i32,
                cols: d as i32,
                fmt: 1,
                _pad: 0,
            };
            let x = DevBuf::from_slice(&data(t * d, 7 + t as u64, 1.0)).unwrap();
            let mut bias = data(ne, 11, 0.5);
            if coarse {
                bias.iter_mut().for_each(|v| *v = (*v * 4.0).round() / 4.0);
            }
            let bias = DevBuf::from_slice(&bias).unwrap();
            let mask: Vec<i32> = (0..ne).map(|e| (e % 3 != 1) as i32).collect();
            let mask = DevBuf::from_slice(&mask).unwrap();
            let n = t * kk;
            let cap = n.min(ne);
            let bufs = || {
                (
                    DevBuf::alloc(t * ne * 4).unwrap(),
                    DevBuf::alloc(n * 4).unwrap(),
                    DevBuf::alloc(n * 4).unwrap(),
                    DevBuf::alloc(ne * 4).unwrap(),
                    DevBuf::alloc((ne + 1) * 4).unwrap(),
                    DevBuf::alloc(n * 4).unwrap(),
                    DevBuf::alloc(n * 4).unwrap(),
                    DevBuf::alloc(cap * 4).unwrap(),
                )
            };
            let a = bufs();
            k.linear(&s, &w, x.ptr, a.0.ptr, t, false).unwrap();
            k.router(
                &s, a.0.ptr, bias.ptr, a.1.ptr, a.2.ptr, t, ne, kk, norm, 1.5,
            )
            .unwrap();
            k.route(
                &s, a.1.ptr, a.2.ptr, a.3.ptr, a.4.ptr, a.5.ptr, a.6.ptr, t, kk, ne,
            )
            .unwrap();
            let (nei, ci) = (ne as i32, cap as i32);
            unsafe {
                k.go(
                    "ds_moe_plan",
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    &s,
                    tang_moe::args![a.3.ptr, mask.ptr, a.7.ptr, nei, ci],
                )
                .unwrap();
            }
            let b = bufs();
            for _ in 0..2 {
                // twice: the arrival counter resets itself
                k.moe_gate(
                    &s,
                    &w,
                    x.ptr,
                    b.0.ptr,
                    bias.ptr,
                    b.1.ptr,
                    b.2.ptr,
                    (b.3.ptr, b.4.ptr, b.5.ptr, b.6.ptr),
                    (mask.ptr, b.7.ptr, cap),
                    t,
                    kk,
                    norm,
                    1.5,
                )
                .unwrap();
            }
            s.sync().unwrap();
            let what = format!("t{t} ne{ne} k{kk}");
            let u = |x: &DevBuf, n: usize| x.read::<u32>(0, n).unwrap();
            assert_eq!(u(&a.0, t * ne), u(&b.0, t * ne), "{what} raw");
            assert_eq!(u(&a.1, n), u(&b.1, n), "{what} ids");
            assert_eq!(u(&a.2, n), u(&b.2, n), "{what} wts");
            assert_eq!(u(&a.3, ne), u(&b.3, ne), "{what} counts");
            assert_eq!(u(&a.4, ne + 1), u(&b.4, ne + 1), "{what} off");
            assert_eq!(u(&a.5, n), u(&b.5, n), "{what} rows");
            assert_eq!(u(&a.6, n), u(&b.6, n), "{what} rw");
            assert_eq!(u(&a.7, cap), u(&b.7, cap), "{what} elist");
        }
    }

    /// The shared expert MLP at the real shape (fp8 w1/w3 [2304, 5120], w2 [5120, 2304], t 6)
    /// through `linear_fast`, as `mlp` runs it (manual: `--ignored --nocapture
    /// shared_expert_bench`): with and without the redundant fp8 qdq passes.
    #[test]
    #[ignore]
    fn shared_expert_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (d, inter) = (5120usize, 2304usize);
        let mut keep = Vec::new();
        let w1 = rand_w(2, inter, d, 1, &mut keep);
        let w3 = rand_w(2, inter, d, 2, &mut keep);
        let w2 = rand_w(2, d, inter, 3, &mut keep);
        let bytes = (2 * inter * d + d * inter) as f64 * (1.0 + 1.0 / 1024.0);
        for t in [1usize, 6] {
            let x = DevBuf::from_slice(&data(t * d, 3, 1.0)).unwrap();
            let xq = DevBuf::alloc(t * d * 4).unwrap();
            let h = DevBuf::alloc(t * inter * 4).unwrap();
            let hq = DevBuf::alloc(t * inter * 4).unwrap();
            let y = DevBuf::alloc(t * d * 4).unwrap();
            let ev = |f: &dyn Fn()| -> f64 {
                f();
                s.sync().unwrap();
                let n = 200;
                let (a, b) = (
                    gpu::Event::new(true).unwrap(),
                    gpu::Event::new(true).unwrap(),
                );
                a.record(&s).unwrap();
                for _ in 0..n {
                    f();
                }
                b.record(&s).unwrap();
                s.sync().unwrap();
                b.since(&a).unwrap() as f64 * 1e3 / n as f64
            };
            let old = ev(&|| {
                let nn = (t * d) as i64;
                unsafe {
                    k.go(
                        "ds_copy",
                        blocks(t * d, 256),
                        (256, 1, 1),
                        0,
                        &s,
                        tang_moe::args![xq.ptr, x.ptr, nn],
                    )
                    .unwrap()
                };
                k.fp8_qdq(&s, xq.ptr, t * d).unwrap();
                k.linear_fast(&s, &w1, Some(&w3), xq.ptr, h.ptr, t, true, false, 0, 10.0)
                    .unwrap();
                let nn = (t * inter) as i64;
                unsafe {
                    k.go(
                        "ds_copy",
                        blocks(t * inter, 256),
                        (256, 1, 1),
                        0,
                        &s,
                        tang_moe::args![hq.ptr, h.ptr, nn],
                    )
                    .unwrap()
                };
                k.fp8_qdq(&s, hq.ptr, t * inter).unwrap();
                k.linear_fast(&s, &w2, None, hq.ptr, y.ptr, t, true, false, 0, 0.0)
                    .unwrap();
            });
            let new = ev(&|| {
                k.linear_fast(&s, &w1, Some(&w3), x.ptr, h.ptr, t, true, false, 0, 10.0)
                    .unwrap();
                k.linear_fast(&s, &w2, None, h.ptr, y.ptr, t, true, false, 0, 0.0)
                    .unwrap();
            });
            eprintln!(
                "shared_expert_bench t {t}: with qdq passes {old:.1} us, without {new:.1} us ({:.0} GB/s)",
                bytes / new / 1e3
            );
        }
    }

    /// `ds_router` against the CPU router math on exact ties (coarse raw values shared by
    /// many experts, so the order rests on the lower-id rule) and on random logits.
    #[test]
    fn router_matches_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        for (t, ne, kk, norm, coarse) in [
            (1, 8, 1, true, false),
            (6, 384, 6, true, true),
            (6, 384, 6, true, false),
            (5, 256, 8, false, true),
            (40, 1000, 16, true, true),
        ] {
            let mut raw = data(t * ne, (ne + kk) as u64, 3.0);
            let mut bias = data(ne, 11, 0.5);
            if coarse {
                raw.iter_mut().for_each(|v| *v = (*v * 2.0).round() / 2.0);
                bias.iter_mut().for_each(|v| *v = (*v * 4.0).round() / 4.0);
            }
            let (dr, db) = (
                DevBuf::from_slice(&raw).unwrap(),
                DevBuf::from_slice(&bias).unwrap(),
            );
            let (ids, wts) = (
                DevBuf::alloc(t * kk * 4).unwrap(),
                DevBuf::alloc(t * kk * 4).unwrap(),
            );
            k.router(&s, dr.ptr, db.ptr, ids.ptr, wts.ptr, t, ne, kk, norm, 1.5)
                .unwrap();
            s.sync().unwrap();
            let (gi, gw) = (
                ids.read::<i32>(0, t * kk).unwrap(),
                wts.read::<f32>(0, t * kk).unwrap(),
            );
            for ti in 0..t {
                let row: Vec<f32> = raw[ti * ne..(ti + 1) * ne]
                    .iter()
                    .map(|&x| forward::softplus(x).sqrt())
                    .collect();
                let biased: Vec<f32> = row.iter().zip(&bias).map(|(a, b)| a + b).collect();
                let sel = forward::topk_stable(&biased, kk);
                let want: Vec<i32> = sel.iter().map(|&e| e as i32).collect();
                assert_eq!(
                    &gi[ti * kk..(ti + 1) * kk],
                    &want[..],
                    "ids t{t} ne{ne} row {ti}"
                );
                let mut w: Vec<f32> = sel.iter().map(|&e| row[e]).collect();
                if norm && kk > 1 {
                    let sm: f32 = w.iter().sum::<f32>() + 1e-20;
                    w.iter_mut().for_each(|v| *v /= sm);
                }
                for (j, &wv) in w.iter().enumerate() {
                    let gv = gw[ti * kk + j];
                    assert!(
                        (gv - wv * 1.5).abs() <= 1e-5 * (1.0 + gv.abs()),
                        "wts row {ti}"
                    );
                }
            }
        }
    }

    /// The attention block of a verify step at the real shapes (t 6, 64 heads × 512, window
    /// 128 + top-k 512, dim 5120, pos ~2000): wo_a, sparse attention (the decode kernel and
    /// the tiled one), the indexer's score + top-k. `-- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn attn_block_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let gk = super::super::graph::GKernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let ev = |f: &dyn Fn()| -> f64 {
            f();
            s.sync().unwrap();
            let n = 50;
            let (a, b) = (
                gpu::Event::new(true).unwrap(),
                gpu::Event::new(true).unwrap(),
            );
            a.record(&s).unwrap();
            for _ in 0..n {
                f();
            }
            b.record(&s).unwrap();
            s.sync().unwrap();
            b.since(&a).unwrap() as f64 * 1e3 / n as f64
        };
        let (t, h, d, g8, ol) = (6usize, 64usize, 512usize, 8usize, 1024usize);
        // wo_a: fp8 [8 × 1024, 4096]
        let (rows, cols) = (g8 * ol, h * d / g8);
        let mut wb: Vec<u8> = (0..rows * cols)
            .map(|i| (i * 2654435761 >> 13) as u8)
            .collect();
        wb.iter_mut()
            .filter(|b| **b & 0x7f == 0x7f)
            .for_each(|b| *b = 0x10);
        let sb: Vec<u8> = (0..rows.div_ceil(32) * cols / 32)
            .map(|i| 118 + (i % 8) as u8)
            .collect();
        let (w, sc) = (
            DevBuf::from_slice(&wb).unwrap(),
            DevBuf::from_slice(&sb).unwrap(),
        );
        let wv = DsW {
            w: w.ptr,
            s: sc.ptr,
            rows: rows as i32,
            cols: cols as i32,
            fmt: 2,
            _pad: 0,
        };
        let o = DevBuf::from_slice(&data(t * h * d, 3, 1.0)).unwrap();
        let mid = DevBuf::alloc(t * rows * 4).unwrap();
        let (ti, gi, oli) = (t as i32, g8 as i32, ol as i32);
        let us = ev(&|| unsafe {
            k.go(
                "ds_wo_a",
                ((g8 * ol / 16) as u32, 1, 1),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![o.ptr, wv, mid.ptr, ti, gi, oli],
            )
            .unwrap()
        });
        eprintln!("attn_block_bench wo_a exact t {t}: {us:.1} us");
        // the K-split path as engine.rs `wo_a`: group-major copy, a GEMV a group, back
        let og = DevBuf::alloc(t * h * d * 4).unwrap();
        let yg = DevBuf::alloc(t * rows * 4).unwrap();
        let perm = |src: u64, dst: u64, a: usize, b: usize, c: usize| unsafe {
            let (ai, bi, ci) = (a as i32, b as i32, c as i32);
            k.go(
                "ds_perm3",
                blocks(a * b * c, 256),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![src, dst, ai, bi, ci],
            )
            .unwrap()
        };
        let us = ev(&|| {
            perm(o.ptr, og.ptr, t, g8, cols);
            for gi in 0..g8 {
                let mut v = wv;
                v.rows = ol as i32;
                v.w += (gi * ol * cols) as u64;
                v.s += ((gi * ol / 32) * (cols / 32)) as u64;
                k.linear_fast(
                    &s,
                    &v,
                    None,
                    og.ptr + (gi * t * cols * 4) as u64,
                    yg.ptr + (gi * t * ol * 4) as u64,
                    t,
                    true,
                    false,
                    0,
                    0.0,
                )
                .unwrap();
            }
            perm(yg.ptr, mid.ptr, g8, t, ol);
        });
        eprintln!("attn_block_bench wo_a K-split t {t}: {us:.1} us");
        let want = mid.read::<f32>(0, t * rows).unwrap();
        let mid2 = DevBuf::alloc(t * rows * 4).unwrap();
        let us = ev(&|| k.wo_ks(&s, &wv, o.ptr, mid2.ptr, t, g8, ol).unwrap());
        eprintln!("attn_block_bench wo_a grouped K-split (2 launches) t {t}: {us:.1} us");
        let got = mid2.read::<f32>(0, t * rows).unwrap();
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| (*a - *b).abs() > 1e-2 * (1.0 + b.abs()))
            .count();
        assert_eq!(bad, 0, "grouped wo_a vs per-group K-split");
        // q + window kv: wq_a 5120 -> 1280, wkv 5120 -> 512, wq_b 1280 -> 64 × 512 (fp8)
        let fp8w = |rows: usize, cols: usize, seed: usize| {
            let mut wb: Vec<u8> = (0..rows * cols)
                .map(|i| ((i + seed) * 2654435761 >> 13) as u8)
                .collect();
            wb.iter_mut()
                .filter(|b| **b & 0x7f == 0x7f)
                .for_each(|b| *b = 0x10);
            let sb: Vec<u8> = (0..rows.div_ceil(32) * cols / 32)
                .map(|i| 118 + ((i + seed) % 8) as u8)
                .collect();
            let (w, sc) = (
                DevBuf::from_slice(&wb).unwrap(),
                DevBuf::from_slice(&sb).unwrap(),
            );
            let v = DsW {
                w: w.ptr,
                s: sc.ptr,
                rows: rows as i32,
                cols: cols as i32,
                fmt: 2,
                _pad: 0,
            };
            (w, sc, v)
        };
        let (dm, ql, rd, win) = (5120usize, 1280usize, 64usize, 128usize);
        let (_a0, _a1, wqa) = fp8w(ql, dm, 1);
        let (_b0, _b1, wkv) = fp8w(d, dm, 2);
        let (_c0, _c1, wqb) = fp8w(h * d, ql, 3);
        let f32w = |n: usize, seed: u64| {
            let b = DevBuf::from_slice(&data(n, seed, 1.0)).unwrap();
            let v = DsW {
                w: b.ptr,
                s: 0,
                rows: 1,
                cols: n as i32,
                fmt: 0,
                _pad: 0,
            };
            (b, v)
        };
        let ((_qn, qn), (_kn, kn)) = (f32w(ql, 12), f32w(d, 13));
        let pos0 = 2000usize;
        let tabv: Vec<f32> = (0..4096 * rd)
            .map(|i| {
                let (p, j) = ((i / rd) as f32, (i % rd) / 2);
                let a = p * 10000f32.powf(-(j as f32) / (rd / 2) as f32);
                if i % 2 == 0 {
                    a.cos()
                } else {
                    a.sin()
                }
            })
            .collect();
        let tab = DevBuf::from_slice(&tabv).unwrap();
        let pdev = DevBuf::from_slice(&[pos0 as i32]).unwrap();
        let x = DevBuf::from_slice(&data(t * dm, 14, 1.0)).unwrap();
        let xq = DevBuf::alloc(t * dm * 4).unwrap();
        let (qr, kvb, qb) = (
            DevBuf::alloc(t * ql * 4).unwrap(),
            DevBuf::alloc(t * d * 4).unwrap(),
            DevBuf::alloc(t * h * d * 4).unwrap(),
        );
        let qrq = DevBuf::alloc(t * ql * 4).unwrap();
        let ring = DevBuf::from_slice(&data(win * d, 15, 1.0)).unwrap();
        let all = DevBuf::alloc((win + t) * d * 4).unwrap();
        let cp = |dst: u64, src: u64, n: usize| unsafe {
            let nn = n as i64;
            k.go(
                "ds_copy",
                blocks(n, 256),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![dst, src, nn],
            )
            .unwrap()
        };
        // before: lin = copy + fp8_qdq + ks_quant + GEMV, each norm/rope/qdq its own launch
        let lin_old = |w: &DsW, x: u64, xq: u64, y: u64, n: usize| {
            cp(xq, x, t * n);
            k.fp8_qdq(&s, xq, t * n).unwrap();
            k.linear_fast(&s, w, None, xq, y, t, true, false, 0, 0.0)
                .unwrap();
        };
        let q_old = || {
            lin_old(&wqa, x.ptr, xq.ptr, qr.ptr, dm);
            k.rms(&s, qr.ptr, &qn, t, 1e-6).unwrap();
            lin_old(&wqb, qr.ptr, qrq.ptr, qb.ptr, ql);
            k.rope(&s, qb.ptr, tab.ptr, t, h, d, rd, pos0, 1, None, false)
                .unwrap();
        };
        let kv_old = || {
            lin_old(&wkv, x.ptr, xq.ptr, kvb.ptr, dm);
            k.rms(&s, kvb.ptr, &kn, t, 1e-6).unwrap();
            k.rope(&s, kvb.ptr, tab.ptr, t, 1, d, rd, pos0, 1, None, false)
                .unwrap();
            k.fp8_qdq(&s, kvb.ptr, t * d).unwrap();
            cp(all.ptr, ring.ptr, win * d);
            cp(all.ptr + (win * d * 4) as u64, kvb.ptr, t * d);
            cp(ring.ptr + ((pos0 % win) * d * 4) as u64, kvb.ptr, t * d); // ring_write
        };
        let us = ev(&q_old);
        eprintln!("attn_block_bench q before (8 launches) t {t}: {us:.1} us");
        let us = ev(&kv_old);
        eprintln!("attn_block_bench window_kv before (10 launches) t {t}: {us:.1} us");
        s.sync().unwrap();
        let ring0 = DevBuf::from_slice(&ring.read::<f32>(0, win * d).unwrap()).unwrap();
        q_old();
        kv_old();
        s.sync().unwrap();
        let (kv_want, all_want) = (
            kvb.read::<f32>(0, t * d).unwrap(),
            all.read::<f32>(0, (win + t) * d).unwrap(),
        );
        let qr_want = qr.read::<f32>(0, t * ql).unwrap();
        let (xc, xs) = k.ks_x_ptrs();
        let (qc, qs) = k.wo_x_ptrs();
        let fused = || {
            k.ks_quant_to(&s, x.ptr, xc, xs, t * dm).unwrap();
            k.ks_gemv(&s, &wqa, None, xc, xs, qr.ptr, t, true, 0, 0.0)
                .unwrap();
            k.ks_gemv(&s, &wkv, None, xc, xs, kvb.ptr, t, true, 0, 0.0)
                .unwrap();
            cp(all.ptr, ring.ptr, win * d);
            let (rdi, wi, eps) = (rd as i32, win as i32, 1e-6f32);
            unsafe {
                k.go(
                    "ds_qkv_post",
                    (t as u32, 2, 1),
                    (1024, 1, 1),
                    0,
                    &s,
                    tang_moe::args![
                        qr.ptr, qn, qc, qs, kvb.ptr, kn, tab.ptr, rdi, pdev.ptr, ring.ptr, wi,
                        all.ptr, eps
                    ],
                )
                .unwrap()
            };
            k.ks_gemv(&s, &wqb, None, qc, qs, qb.ptr, t, true, 0, 0.0)
                .unwrap();
            k.rope(&s, qb.ptr, tab.ptr, t, h, d, rd, pos0, 1, None, false)
                .unwrap();
        };
        let us = ev(&fused);
        eprintln!("attn_block_bench q + window_kv fused (7 launches) t {t}: {us:.1} us");
        // one more from the same ring as the reference run: kv, qr and the window bit-equal
        cp(ring.ptr, ring0.ptr, win * d);
        fused();
        s.sync().unwrap();
        assert!(kvb.read::<f32>(0, t * d).unwrap() == kv_want, "fused kv");
        assert!(qr.read::<f32>(0, t * ql).unwrap() == qr_want, "fused qr");
        assert!(
            all.read::<f32>(0, (win + t) * d).unwrap() == all_want,
            "fused window"
        );
        // sparse attention: window 128 + 6 chunk rows, then 512 compressed rows
        let (win, kc) = (128 + t, 512usize);
        let kk = win + kc;
        let q = DevBuf::from_slice(&data(t * h * d, 5, 1.0)).unwrap();
        let kv0 = DevBuf::from_slice(&data(win * d, 6, 1.0)).unwrap();
        let kv1 = DevBuf::from_slice(&data(2048 * d, 7, 1.0)).unwrap();
        let sink = DevBuf::from_slice(&data(h, 8, 1.0)).unwrap();
        let ix: Vec<i32> = (0..t * kk)
            .map(|i| {
                let j = i % kk;
                if j < win {
                    j as i32
                } else {
                    (win + (j * 7 + i / kk) % 500) as i32
                }
            })
            .collect();
        let ix = DevBuf::from_slice(&ix).unwrap();
        let ob = DevBuf::alloc(t * h * d * 4).unwrap();
        let (hi, di, ki, split, scale) = (h as i32, d as i32, kk as i32, win as i32, 0.04f32);
        let us = ev(&|| unsafe {
            gk.go(
                "dg_sparse_attn",
                ((t * h) as u32, 1, 1),
                (512, 1, 1),
                (kk * 4) as u32,
                &s,
                tang_moe::args![
                    q.ptr, kv0.ptr, split, kv1.ptr, sink.ptr, ix.ptr, ob.ptr, hi, di, ki, scale
                ],
            )
            .unwrap()
        });
        eprintln!("attn_block_bench dg_sparse_attn t {t} k {kk}: {us:.1} us");
        k.exact = false;
        let us = ev(&|| {
            k.sparse_attn(
                &s, q.ptr, kv0.ptr, win, kv1.ptr, sink.ptr, ix.ptr, ob.ptr, t, h, d, kk, scale,
            )
            .unwrap()
        });
        eprintln!("attn_block_bench split-K ds_attn_fd (2 launches) t {t} k {kk}: {us:.1} us");
        k.attn_fd = false;
        k.attn_fuse = false;
        let us = ev(&|| {
            k.sparse_attn(
                &s, q.ptr, kv0.ptr, win, kv1.ptr, sink.ptr, ix.ptr, ob.ptr, t, h, d, kk, scale,
            )
            .unwrap()
        });
        eprintln!("attn_block_bench tiled sparse_attn (3 launches) t {t} k {kk}: {us:.1} us");
        k.attn_fuse = true;
        let us = ev(&|| {
            k.sparse_attn(
                &s, q.ptr, kv0.ptr, win, kv1.ptr, sink.ptr, ix.ptr, ob.ptr, t, h, d, kk, scale,
            )
            .unwrap()
        });
        eprintln!("attn_block_bench tiled sparse_attn fused soft+pv (2 launches) t {t} k {kk}: {us:.1} us");
        // indexer at pos 2000, ratio 4: 64 heads × 128, width 500, top-512
        let (nh, ihd, r, wmax, kmax) = (64usize, 128usize, 4usize, 4096usize, 512usize);
        let qi = DevBuf::from_slice(&data(t * nh * ihd, 9, 1.0)).unwrap();
        let ik = DevBuf::from_slice(&data(wmax * ihd, 10, 1.0)).unwrap();
        let wts = DevBuf::from_slice(&data(t * nh, 11, 1.0)).unwrap();
        let score = DevBuf::alloc(t * wmax * 4).unwrap();
        let sel = DevBuf::alloc(t * kmax * 4).unwrap();
        let out = DevBuf::alloc(t * kmax * 4).unwrap();
        let pp = DevBuf::from_slice(&[2000i32]).unwrap();
        let (ri, nhi, ihdi, kmi, offi) = (r as i32, nh as i32, ihd as i32, kmax as i32, win as i32);
        let wi = wmax as i32;
        let us = ev(&|| unsafe {
            let sg = blocks(wmax * 32, 256);
            gk.go(
                "dg_index_score",
                (sg.0, t as u32, 1),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![
                    qi.ptr, ik.ptr, wts.ptr, score.ptr, pp.ptr, 0i32, ri, nhi, ihdi, wi
                ],
            )
            .unwrap();
            gk.go(
                "dg_topk",
                (t as u32, 1, 1),
                super::super::graph::SEL,
                0,
                &s,
                tang_moe::args![score.ptr, sel.ptr, out.ptr, pp.ptr, 0i32, ri, kmi, offi, wi],
            )
            .unwrap();
        });
        eprintln!("attn_block_bench indexer score+topk t {t} batched (per layer): {us:.1} us");
    }

    /// The grouped K-split wo_a (`Kernels::wo_ks`, quantize + one launch) against the exact
    /// `ds_wo_a` on the same qdq'd activation, fp8 and fp4: the K-split bar.
    #[test]
    fn wo_ks_matches_exact() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        if k.exact {
            return;
        }
        let s = Stream::blocking().unwrap();
        let (gn, ol, gd) = (3usize, 64usize, 256usize);
        let rows = gn * ol;
        for fmt in [2i32, 3] {
            for t in [1usize, 6, 8] {
                let wb: Vec<u8> = (0..rows * gd / if fmt == 3 { 2 } else { 1 })
                    .map(|i| {
                        let b = (i * 2654435761 >> 11) as u8;
                        if fmt == 2 && b & 0x7f == 0x7f {
                            0x08
                        } else {
                            b
                        }
                    })
                    .collect();
                let ns = if fmt == 2 {
                    rows.div_ceil(32) * gd / 32
                } else {
                    rows * gd / 32
                };
                let sb: Vec<u8> = (0..ns).map(|i| 120 + (i % 9) as u8).collect();
                let (dw, ds) = (
                    DevBuf::from_slice(&wb).unwrap(),
                    DevBuf::from_slice(&sb).unwrap(),
                );
                let w = DsW {
                    w: dw.ptr,
                    s: ds.ptr,
                    rows: rows as i32,
                    cols: gd as i32,
                    fmt,
                    _pad: 0,
                };
                let o = DevBuf::from_slice(&data(t * gn * gd, 3, 2.0)).unwrap();
                k.fp8_qdq(&s, o.ptr, t * gn * gd).unwrap();
                let (a, b) = (
                    DevBuf::alloc(t * rows * 4).unwrap(),
                    DevBuf::alloc(t * rows * 4).unwrap(),
                );
                let (ti, gi, oli) = (t as i32, gn as i32, ol as i32);
                unsafe {
                    k.go(
                        "ds_wo_a",
                        ((gn * ol / 16) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        &s,
                        tang_moe::args![o.ptr, w, a.ptr, ti, gi, oli],
                    )
                    .unwrap()
                };
                assert!(k.wo_ks_ok(&w, t, gn, ol));
                k.wo_ks(&s, &w, o.ptr, b.ptr, t, gn, ol).unwrap();
                s.sync().unwrap();
                let (want, got) = (
                    a.read::<f32>(0, t * rows).unwrap(),
                    b.read::<f32>(0, t * rows).unwrap(),
                );
                for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-2 * (1.0 + y.abs()),
                        "fmt {fmt} t {t} @{i}: {x} vs {y}"
                    );
                }
            }
        }
    }

    /// `ds_wo_a`'s fp8 fast path against its generic (wget) path on the same weights
    /// dequantized to f32: identical sums, so bit-equal. t 9 crosses the 8-token pass.
    #[test]
    fn wo_a_fp8_matches_f32() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (t, gn, ol, gd) = (9usize, 2usize, 48usize, 96usize);
        let rows = gn * ol;
        let mut wb: Vec<u8> = (0..rows * gd)
            .map(|i| (i * 2654435761 >> 11) as u8)
            .collect();
        wb.iter_mut()
            .filter(|b| **b & 0x7f == 0x7f)
            .for_each(|b| *b = 0x08);
        let nsc = gd.div_ceil(32);
        let sb: Vec<u8> = (0..rows.div_ceil(32) * nsc)
            .map(|i| 120 + (i % 9) as u8)
            .collect();
        let wf: Vec<f32> = (0..rows * gd)
            .map(|i| {
                let (r, c) = (i / gd, i % gd);
                super::super::quant::e4m3_to_f32(wb[i])
                    * super::super::quant::e8m0_to_f32(sb[(r / 32) * nsc + c / 32])
            })
            .collect();
        let (dw, ds, df) = (
            DevBuf::from_slice(&wb).unwrap(),
            DevBuf::from_slice(&sb).unwrap(),
            DevBuf::from_slice(&wf).unwrap(),
        );
        let o = DevBuf::from_slice(&data(t * gn * gd, 3, 2.0)).unwrap();
        let run = |w: DsW| {
            let mid = DevBuf::alloc(t * rows * 4).unwrap();
            let (ti, gi, oli) = (t as i32, gn as i32, ol as i32);
            unsafe {
                k.go(
                    "ds_wo_a",
                    ((gn * ol / 16) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &s,
                    tang_moe::args![o.ptr, w, mid.ptr, ti, gi, oli],
                )
                .unwrap()
            };
            s.sync().unwrap();
            mid.read::<f32>(0, t * rows).unwrap()
        };
        let (rw, cw) = (rows as i32, gd as i32);
        let got = run(DsW {
            w: dw.ptr,
            s: ds.ptr,
            rows: rw,
            cols: cw,
            fmt: 2,
            _pad: 0,
        });
        let want = run(DsW {
            w: df.ptr,
            s: 0,
            rows: rw,
            cols: cw,
            fmt: 0,
            _pad: 0,
        });
        // a CPU check of the sum too (any order: within an ulp or so of bf16)
        for ti in 0..t {
            for gi in 0..gn {
                for r in 0..ol {
                    let row = gi * ol + r;
                    let ov =
                        &data(t * gn * gd, 3, 2.0)[(ti * gn + gi) * gd..(ti * gn + gi + 1) * gd];
                    let cpu: f32 = ov
                        .iter()
                        .zip(&wf[row * gd..(row + 1) * gd])
                        .map(|(a, b)| a * bf(*b))
                        .sum();
                    let gv = got[(ti * gn + gi) * ol + r];
                    assert!(
                        (gv - cpu).abs() <= 2e-2 * (1.0 + cpu.abs()),
                        "t {ti} g {gi} r {r}: {gv} vs {cpu}"
                    );
                }
            }
        }
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "fp8 path != f32 path"
        );
    }

    /// The fused mHC glue of one layer (ds_hc_stage, ds_hc_post_mix, ds_hc_post with the
    /// shared add) against the unfused kernels, bit for bit, at the real shapes.
    fn hc_glue(bench: bool) {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::new().unwrap();
        let (hc, d) = (4usize, 5120usize);
        let mix = (2 + hc) * hc;
        let up = |v: &[f32]| DevBuf::from_slice(v).unwrap();
        let fns: Vec<DevBuf> = (0..2)
            .map(|i| up(&data(mix * hc * d, 20 + i, 0.02)))
            .collect();
        let fw = |i: usize| DsW {
            w: fns[i].ptr,
            s: 0,
            rows: mix as i32,
            cols: (hc * d) as i32,
            fmt: 0,
            _pad: 0,
        };
        let scale = up(&[0.5, 0.25, 0.125]);
        let base = up(&data(mix, 31, 0.5));
        let nw = DevBuf::from_slice(
            &data(d, 33, 1.0)
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let nv = DsW {
            w: nw.ptr,
            s: 0,
            rows: 1,
            cols: d as i32,
            fmt: 1,
            _pad: 0,
        };
        // DS_BENCH_HC_IT: the Sinkhorn iterations (the model's 20), to time that part
        let it: i32 = std::env::var("DS_BENCH_HC_IT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        let (hci, di, eps, neps) = (hc as i32, d as i32, 1e-6f32, 1e-6f32);
        // DS_BENCH_HC_STUB: ds_hc_fused_ks's phases stubbed out (timing only, see the kernel)
        let stub: i32 = std::env::var("DS_BENCH_HC_STUB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        for t in [1usize, 6, 13] {
            let ti = t as i32;
            let x = up(&data(t * hc * d, 1, 1.0));
            let pre0 = up(&data(t * hc, 2, 1.0));
            let ha = up(&data(t * d, 3, 1.0));
            let y = up(&data(t * d, 4, 1.0));
            let sh = up(&data(t * d, 5, 1.0));
            let a = |n: usize| DevBuf::alloc(n * 4).unwrap();
            // [mixA, preA, postA, combA, hA, x2, mixF, preF, postF, combF, hF, x3]
            let sizes = [
                mix,
                hc,
                hc,
                hc * hc,
                d,
                hc * d,
                mix,
                hc,
                hc,
                hc * hc,
                d,
                hc * d,
            ];
            let mk = || sizes.iter().map(|&n| a(t * n)).collect::<Vec<_>>();
            let (o, f) = (mk(), mk());
            let y2 = a(t * d);
            let old = |s: &Stream| -> Result<()> {
                let b = &o;
                let nn = (t * d) as i64;
                unsafe {
                    k.go(
                        "ds_copy",
                        blocks(t * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![y2.ptr, y.ptr, nn],
                    )?;
                }
                k.linear(s, &fw(0), x.ptr, b[0].ptr, t, false)?;
                let n = (hc * d) as i32;
                unsafe {
                    k.go(
                        "ds_hc_sinkhorn",
                        (ti as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            x.ptr, b[0].ptr, n, neps, scale.ptr, base.ptr, b[1].ptr, b[2].ptr,
                            b[3].ptr, hci, it, eps
                        ],
                    )?;
                    k.go(
                        "ds_hc_pre",
                        blocks(t * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![x.ptr, pre0.ptr, b[4].ptr, ti, hci, di],
                    )?;
                }
                k.rms(s, b[4].ptr, &nv, t, neps)?;
                unsafe {
                    k.go(
                        "ds_hc_post",
                        blocks(t * hc * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            ha.ptr, 0u64, x.ptr, b[2].ptr, b[3].ptr, b[5].ptr, ti, hci, di
                        ],
                    )?;
                }
                k.linear(s, &fw(1), b[5].ptr, b[6].ptr, t, false)?;
                unsafe {
                    k.go(
                        "ds_hc_sinkhorn",
                        (ti as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            b[5].ptr, b[6].ptr, n, neps, scale.ptr, base.ptr, b[7].ptr, b[8].ptr,
                            b[9].ptr, hci, it, eps
                        ],
                    )?;
                    k.go(
                        "ds_hc_pre",
                        blocks(t * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![b[5].ptr, b[1].ptr, b[10].ptr, ti, hci, di],
                    )?;
                }
                k.rms(s, b[10].ptr, &nv, t, neps)?;
                let nn = (t * d) as i64;
                unsafe {
                    k.go(
                        "ds_add_round",
                        blocks(t * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![y2.ptr, sh.ptr, nn],
                    )?;
                    k.go(
                        "ds_hc_post",
                        blocks(t * hc * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            y2.ptr, 0u64, b[5].ptr, b[8].ptr, b[9].ptr, b[11].ptr, ti, hci, di
                        ],
                    )?;
                }
                Ok(())
            };
            let new = |s: &Stream| -> Result<()> {
                let b = &f;
                let (w0, w1) = (nv, fw(1));
                let wa = fw(0);
                unsafe {
                    k.go(
                        "ds_hc_post_mix",
                        (mix as u32, t.div_ceil(8) as u32, 1),
                        (32 * t.min(8) as u32, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            0u64, 0u64, x.ptr, 0u64, 0u64, x.ptr, wa, b[0].ptr, ti, hci, di
                        ],
                    )?;
                    k.go(
                        "ds_hc_stage",
                        (ti as u32, 1, 1),
                        (1024, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            x.ptr, b[0].ptr, neps, scale.ptr, base.ptr, b[1].ptr, b[2].ptr,
                            b[3].ptr, hci, it, eps, pre0.ptr, b[4].ptr, w0, di, neps, 1i32
                        ],
                    )?;
                    k.go(
                        "ds_hc_post_mix",
                        (mix as u32, t.div_ceil(8) as u32, 1),
                        (32 * t.min(8) as u32, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            ha.ptr, 0u64, x.ptr, b[2].ptr, b[3].ptr, b[5].ptr, w1, b[6].ptr, ti,
                            hci, di
                        ],
                    )?;
                    k.go(
                        "ds_hc_stage",
                        (ti as u32, 1, 1),
                        (1024, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            b[5].ptr, b[6].ptr, neps, scale.ptr, base.ptr, b[7].ptr, b[8].ptr,
                            b[9].ptr, hci, it, eps, b[1].ptr, b[10].ptr, w0, di, neps, 1i32
                        ],
                    )?;
                    k.go(
                        "ds_hc_post",
                        blocks(t * hc * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            y.ptr, sh.ptr, b[5].ptr, b[8].ptr, b[9].ptr, b[11].ptr, ti, hci, di
                        ],
                    )?;
                }
                Ok(())
            };
            // the default path: ds_hc_mix_ks (K-split partials) into ds_hc_stage
            let g3 = mk();
            let npk = (hc * d).div_ceil(256);
            let (pa, pf) = (a(t * npk * mix), a(t * npk * mix));
            let ks = |s: &Stream| -> Result<()> {
                let b = &g3;
                let (w0, w1, wa) = (nv, fw(1), fw(0));
                let (gx, gy, np) = (npk as u32, t.div_ceil(8) as u32, npk as i32);
                unsafe {
                    k.go(
                        "ds_hc_mix_ks",
                        (gx, gy, 1),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            0u64, 0u64, x.ptr, 0u64, 0u64, x.ptr, wa, pa.ptr, ti, hci, di
                        ],
                    )?;
                    k.go(
                        "ds_hc_stage",
                        (ti as u32, 1, 1),
                        (1024, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            x.ptr, pa.ptr, neps, scale.ptr, base.ptr, b[1].ptr, b[2].ptr, b[3].ptr,
                            hci, it, eps, pre0.ptr, b[4].ptr, w0, di, neps, np
                        ],
                    )?;
                    k.go(
                        "ds_hc_mix_ks",
                        (gx, gy, 1),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            ha.ptr, 0u64, x.ptr, b[2].ptr, b[3].ptr, b[5].ptr, w1, pf.ptr, ti, hci,
                            di
                        ],
                    )?;
                    k.go(
                        "ds_hc_stage",
                        (ti as u32, 1, 1),
                        (1024, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            b[5].ptr, pf.ptr, neps, scale.ptr, base.ptr, b[7].ptr, b[8].ptr,
                            b[9].ptr, hci, it, eps, b[1].ptr, b[10].ptr, w0, di, neps, np
                        ],
                    )?;
                    k.go(
                        "ds_hc_post",
                        blocks(t * hc * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            y.ptr, sh.ptr, b[5].ptr, b[8].ptr, b[9].ptr, b[11].ptr, ti, hci, di
                        ],
                    )?;
                }
                Ok(())
            };
            // ds_hc_fused_ks: one launch per stage (in the model the closing post-mix also
            // carries the next block's attention stage: 2 launches/layer)
            let g4 = mk();
            let gxf = d.div_ceil(256 / hc);
            let (qa, qf) = (a(t * gxf * (mix + 2)), a(t * gxf * (mix + 2)));
            let cnt = DevBuf::from_slice(&[0i32; 1024]).unwrap();
            let fk = |s: &Stream| -> Result<()> {
                let b = &g4;
                let (w0, w1, wa) = (nv, fw(1), fw(0));
                let g = (gxf as u32, t.div_ceil(8) as u32, 1);
                unsafe {
                    k.go(
                        "ds_hc_fused_ks",
                        g,
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            0u64, 0u64, x.ptr, 0u64, 0u64, x.ptr, wa, qa.ptr, pre0.ptr, b[4].ptr,
                            w0, scale.ptr, base.ptr, b[1].ptr, b[2].ptr, b[3].ptr, cnt.ptr, ti,
                            hci, di, it, eps, neps, neps, stub
                        ],
                    )?;
                    k.go(
                        "ds_hc_fused_ks",
                        g,
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            ha.ptr, 0u64, x.ptr, b[2].ptr, b[3].ptr, b[5].ptr, w1, qf.ptr,
                            b[1].ptr, b[10].ptr, w0, scale.ptr, base.ptr, b[7].ptr, b[8].ptr,
                            b[9].ptr, cnt.ptr, ti, hci, di, it, eps, neps, neps, stub
                        ],
                    )?;
                    k.go(
                        "ds_hc_post",
                        blocks(t * hc * d, 256),
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            y.ptr, sh.ptr, b[5].ptr, b[8].ptr, b[9].ptr, b[11].ptr, ti, hci, di
                        ],
                    )?;
                }
                Ok(())
            };
            // one ds_hc_fused_ks launch alone (the first stage's), for the phase timings
            let fk1 = |s: &Stream| -> Result<()> {
                let b = &g4;
                let (w0, wa) = (nv, fw(0));
                let g = (gxf as u32, t.div_ceil(8) as u32, 1);
                unsafe {
                    k.go(
                        "ds_hc_fused_ks",
                        g,
                        (256, 1, 1),
                        0,
                        s,
                        tang_moe::args![
                            0u64, 0u64, x.ptr, 0u64, 0u64, x.ptr, wa, qa.ptr, pre0.ptr, b[4].ptr,
                            w0, scale.ptr, base.ptr, b[1].ptr, b[2].ptr, b[3].ptr, cnt.ptr, ti,
                            hci, di, it, eps, neps, neps, stub
                        ],
                    )
                }
            };
            old(&s).unwrap();
            new(&s).unwrap();
            ks(&s).unwrap();
            // ds_hc_fused_ks's grid barrier wants one row of blocks (t <= 8)
            let one = t <= 8;
            if one {
                fk(&s).unwrap();
            }
            s.sync().unwrap();
            // the relaxed bar against the exact glue: every element within ~1 bf16 ulp
            // (the mixes themselves are partials on this path, not compared)
            for (i, (&n, gb)) in sizes
                .iter()
                .enumerate()
                .flat_map(|(i, n)| [(i, (n, &g3)), (i, (n, &g4))])
            {
                if (!one && std::ptr::eq(gb, &g4)) || stub != 0 {
                    continue;
                }
                if i == 0 || i == 6 {
                    continue;
                }
                let (u, v) = (
                    o[i].read::<f32>(0, t * n).unwrap(),
                    gb[i].read::<f32>(0, t * n).unwrap(),
                );
                // 1 bf16 ulp of the element, floored at the buffer's scale (cancellation)
                let fl = u.iter().fold(0f32, |m, a| m.max(a.abs())) / 4096.0;
                let (mut bad, mut ne) = (0, 0);
                for (a, b) in u.iter().zip(&v) {
                    ne += (a.to_bits() != b.to_bits()) as usize;
                    bad += ((a - b).abs() > a.abs().max(b.abs()) / 128.0 + fl) as usize;
                }
                assert_eq!(bad, 0, "ks t {t} buffer {i}: {bad} of {} past 1 ulp", t * n);
                if bench {
                    eprintln!("ks t {t} buffer {i}: {ne} of {} not bit-equal", t * n);
                }
            }
            for (i, &n) in sizes.iter().enumerate() {
                let (u, v) = (
                    o[i].read::<f32>(0, t * n).unwrap(),
                    f[i].read::<f32>(0, t * n).unwrap(),
                );
                let bad = u
                    .iter()
                    .zip(&v)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(bad, 0, "t {t} buffer {i}: {bad} of {} differ", t * n);
            }
            if !bench {
                continue;
            }
            // 40 layers of glue per graph, replayed; CUDA events around the replays
            for (name, body) in [
                (
                    "unfused (11 launches/layer)",
                    &old as &dyn Fn(&Stream) -> Result<()>,
                ),
                ("fused (5 launches/layer)", &new),
                ("ks (5 launches/layer)", &ks),
                ("fused ks (3 launches/layer)", &fk),
                ("fused ks, 1 stage launch alone", &fk1),
            ] {
                if !one && name.starts_with("fused ks") {
                    continue;
                }
                let gr = gpu::Graph::capture(&s, |s| {
                    for _ in 0..40 {
                        body(s).unwrap();
                    }
                    Ok(())
                })
                .unwrap();
                gr.launch(&s).unwrap();
                let (e0, e1) = (
                    gpu::Event::new(true).unwrap(),
                    gpu::Event::new(true).unwrap(),
                );
                let n = 20;
                e0.record(&s).unwrap();
                for _ in 0..n {
                    gr.launch(&s).unwrap();
                }
                e1.record(&s).unwrap();
                e1.sync().unwrap();
                let us = e1.since(&e0).unwrap() as f64 * 1e3 / n as f64;
                eprintln!(
                    "hc_glue t {t} {name}: {us:.0} us / 40 layers, {:.1} us / layer",
                    us / 40.0
                );
                // the same launches direct (no graph), each wrapped in a timing-event pair as
                // the obs op table's spans are: what the direct-path table charges them
                let evs: Vec<_> = (0..80).map(|_| gpu::Event::new(true).unwrap()).collect();
                body(&s).unwrap();
                s.sync().unwrap();
                let mut tot = 0.0;
                for _ in 0..n {
                    for l in 0..40 {
                        evs[2 * l].record(&s).unwrap();
                        body(&s).unwrap();
                        evs[2 * l + 1].record(&s).unwrap();
                    }
                    s.sync().unwrap();
                    for l in 0..40 {
                        tot += evs[2 * l + 1].since(&evs[2 * l]).unwrap() as f64 * 1e3;
                    }
                }
                eprintln!(
                    "hc_glue t {t} {name}, direct + span events: {:.1} us / layer",
                    tot / (n * 40) as f64
                );
                // and with L2 flushed before each layer (in the model the MoE's ~100 MB of
                // expert weights run between stages)
                let fl = 32usize << 20;
                let (fa, fb) = (
                    DevBuf::alloc(fl * 4).unwrap(),
                    DevBuf::alloc(fl * 4).unwrap(),
                );
                let mut tot = 0.0;
                for _ in 0..4 {
                    for l in 0..40 {
                        unsafe {
                            k.go(
                                "ds_copy",
                                blocks(fl, 256),
                                (256, 1, 1),
                                0,
                                &s,
                                tang_moe::args![fa.ptr, fb.ptr, fl as i64],
                            )
                            .unwrap()
                        };
                        evs[2 * l].record(&s).unwrap();
                        body(&s).unwrap();
                        evs[2 * l + 1].record(&s).unwrap();
                    }
                    s.sync().unwrap();
                    for l in 0..40 {
                        tot += evs[2 * l + 1].since(&evs[2 * l]).unwrap() as f64 * 1e3;
                    }
                }
                eprintln!(
                    "hc_glue t {t} {name}, direct + span events, cold L2: {:.1} us / layer",
                    tot / (4 * 40) as f64
                );
            }
        }
    }

    /// Also the default (K-split) glue against the exact one, to the relaxed bar.
    #[test]
    fn hc_glue_fused_exact() {
        hc_glue(false);
    }

    #[test]
    #[ignore]
    fn hc_glue_bench() {
        hc_glue(true);
    }
}
