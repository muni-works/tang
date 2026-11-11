//! GPU launcher for the GLM-5 MoE kernels ([`super::SRC`]): weights uploaded in their storage
//! format (GGUF bytes as-is, or f32), f32 activations at the interface.
//!
//! Two paths, picked per [`Kernels`] (`TANG_GLM5_MOE=fast` | anything else = naive):
//! - naive: the reference kernels (one thread per output); the parity baseline.
//! - fast (`g5f_*` in moe_cuda.cuh): warp-per-row dequant-reuse GEMVs with int8 activations
//!   (dp4a), one grouped gate|up and one grouped down(+combine) launch for all of a token's
//!   experts + the shared expert, warp router/select, scratch reused across calls.
//!
//! Pod parity test (1×H200 or any CUDA card):
//! `TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm --features cuda --lib flash::glm5::moe`

use super::{MoeParams, W};
use crate::gguf::{GgmlType, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::sys as cu;
use std::cell::RefCell;
use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

type Func = cudarc::driver::sys::CUfunction;

/// `struct G5W` in moe_cuda.cuh (same field order and widths: 8 + 8 + 4 + 4 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct G5W {
    pub base: u64,
    pub estride: u64,
    pub row_bytes: u32,
    pub ty: u32,
}

/// No weight (`base == 0`): the kernels' "no shared expert".
const NO_W: G5W = G5W {
    base: 0,
    estride: 0,
    row_bytes: 0,
    ty: 0,
};

/// Scratch slots of [`Pool`].
const P_X: usize = 0;
const P_Y: usize = 1;
const P_XQ: usize = 2;
const P_XD: usize = 3;
const P_SC: usize = 4;
const P_IDS: usize = 5;
const P_WTS: usize = 6;
const P_H: usize = 7;
const P_HQ: usize = 8;
const P_HD: usize = 9;
const P_XQ2: usize = 10;
const P_HQ2: usize = 11;
const P_YP: usize = 12;
const P_CNT: usize = 13;
const P_N: usize = 14;

/// Device scratch reused across calls (grown on demand), plus the stream `ffn_host` runs on.
#[derive(Default)]
struct Pool {
    bufs: Vec<Option<DevBuf>>,
    stream: Option<Stream>,
    /// dynamic-smem opt-in already set on g5f_gu / g5f_down (bytes)
    smem_set: [u32; 5],
    /// u32s of P_CNT known zero (g5f_route resets its counters on exit)
    cnt_zero: usize,
}

impl Pool {
    fn buf(&mut self, slot: usize, bytes: usize) -> Result<&DevBuf> {
        if self.bufs.len() < P_N {
            self.bufs.resize_with(P_N, || None);
        }
        let b = &mut self.bufs[slot];
        if b.as_ref().is_none_or(|b| b.len < bytes) {
            *b = None; // free before allocating the larger one
            *b = Some(DevBuf::alloc(bytes.max(256))?);
        }
        Ok(b.as_ref().unwrap())
    }

    fn get(&mut self, slot: usize, bytes: usize) -> Result<u64> {
        Ok(self.buf(slot, bytes)?.ptr)
    }
}

pub struct Kernels {
    _m: Module,
    logits: Func,
    select: Func,
    gu: Func,
    down: Func,
    combine: Func,
    f_q8: Func,
    f_router: Func,
    f_select: Func,
    f_gu: Func,
    f_down: Func,
    f_down_split: Func,
    f_sum: Func,
    f_route: Func,
    f_gun: Func,
    f_down_splitn: Func,
    f_gux: Func,
    f_gu4: Func,
    f_down4: Func,
    /// Round-4 gate|up / down (`g5f_gu4` / `g5f_down4`, `TANG_GLM5_MOE_K4=1`, default OFF: no
    /// end-to-end gain on the B300 and logit KL 2.2e-3 > the 1e-3 bar, see NOTES):
    /// activation-resident lane-contiguous rows, NOT bit-identical to `g5f_gux` /
    /// `g5f_down_splitn` (summation order); `TANG_GLM5_MOE_K4=0` restores those.
    pub k4: bool,
    /// Rows a block of `g5f_gu4` / `g5f_down4` (`TANG_GLM5_MOE_K4R` / `_K4DR`).
    k4r: usize,
    k4dr: usize,
    k4nr: usize,
    k4dnr: usize,
    k4ns: usize,
    /// Rows a block in `g5f_gux` (`TANG_GLM5_MOE_GUX`, 1..16, default 4; 0 = `g5f_gun` / `g5f_gu`).
    pub gux: usize,
    /// Rows a warp in `g5f_gun` (`TANG_GLM5_MOE_GUR`; 0 = `g5f_gu`, one row a warp).
    pub gur: usize,
    /// Rows a warp in `g5f_down_splitn` (`TANG_GLM5_MOE_DNR`, even; 0 = `g5f_down_split`, two).
    pub dnr: usize,
    /// q8(x) + router + select as one launch (`g5f_route`; `TANG_GLM5_MOE_ROUTE1=0`: three).
    pub route1: bool,
    /// The fast path (`TANG_GLM5_MOE=fast`). Tests flip it directly.
    pub fast: bool,
    /// Two-level int8 activations (codes + residual codes; `TANG_GLM5_MOE_ACT=q8` turns it
    /// off): ~1e-5 activation error instead of ~1e-2, 2–4 more dp4a per 8 weights.
    pub act2: bool,
    /// Stage weight rows in shared memory (`TANG_GLM5_MOE_STAGE=0`: read them from global).
    pub stage: bool,
    /// Routed layers' down as one warp per (slot, row pair) + `g5f_sum`
    /// (`TANG_GLM5_MOE_DOWN_SPLIT`, default on); off = `g5f_down` with `down_sw` warps a pair.
    pub down_split: bool,
    /// Warps sharing one down row pair on routed layers (1, 2, 4; `TANG_GLM5_MOE_DOWN_SW`).
    pub down_sw: usize,
    /// NaN-fill h / y before each fast call (`TANG_GLM5_MOE_DEBUG=1`, unless `..._POISON=0`).
    pub poison: bool,
    smem_optin: u32,
    pool: RefCell<Pool>,
}

impl Drop for Kernels {
    fn drop(&mut self) {
        // device buffers + the stream go before the module (both before the context, which
        // the owner drops after us)
        let p = self.pool.get_mut();
        p.bufs.clear();
        p.stream = None;
    }
}

impl Kernels {
    pub fn new(g: &Gpu) -> Result<Self> {
        let env_on = |k: &str, d: bool| match std::env::var(k) {
            Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
            Err(_) => d,
        };
        // TANG_GLM5_MOE_MINB: __launch_bounds__ min blocks for g5f_gu / g5f_down. 6 (<= 80 regs)
        // measured best on the H200 l4 T1: 1 -> 126 us (86 / 139 regs), 4 -> 115, 6 -> 110
        let minb = std::env::var("TANG_GLM5_MOE_MINB")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8);
        // TANG_GLM5_MOE_DOT32=0: the 8-weight Q4_K / Q5_K row decode instead of the 32-chunk one
        let dot32 = i32::from(env_on("TANG_GLM5_MOE_DOT32", true));
        // TANG_GLM5_MOE_UNROLL: unroll of the 32-chunk dot loops (more loads in flight per lane)
        let unroll = std::env::var("TANG_GLM5_MOE_UNROLL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1);
        // TANG_GLM5_MOE_FUSE2=0: gate / up (and the down row pair) as two separate row loops
        let fuse2 = i32::from(env_on("TANG_GLM5_MOE_FUSE2", true));
        // TANG_GLM5_MOE_GUR / _DNR: rows a warp in the routed gate|up / split down (0 = the
        // one-row g5f_gu / two-row g5f_down_split); same sums, so the same h / y bit for bit
        // TANG_GLM5_MOE_PF (mask, default 2): L2 prefetch of a warp's whole rows before its dot,
        // 1 = gate|up, 2 = down (no-op on the math; B300 l11 Q6_K down 52 -> 44 us, gate|up no gain)
        let pf = std::env::var("TANG_GLM5_MOE_PF")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(2);
        let rows = |key: &str, d: usize| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(d)
        };
        let gur = rows("TANG_GLM5_MOE_GUR", 1).min(8);
        let dnr = match rows("TANG_GLM5_MOE_DNR", 2).min(16) {
            1 => 2,
            n => n & !1,
        };
        // TANG_GLM5_MOE_GUX: rows a block of the activation-resident gate|up (0 = off)
        let gux = rows("TANG_GLM5_MOE_GUX", 4).min(16);
        // TANG_GLM5_MOE_SEL2=0: the previous top-k select (smem scan, per-round global stores)
        // (2, default: the select rounds as warp reductions — same choices, same float ops)
        let sel2 = std::env::var("TANG_GLM5_MOE_SEL2")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(2);
        // TANG_GLM5_MOE_LDU: unroll of the 8-weight row loop (Q8_0 shared / dense rows; same order)
        let ldu = rows("TANG_GLM5_MOE_LDU", 4).min(16);
        // TANG_GLM5_MOE_K4R / _K4NR / _K4DR / _K4DNR / _K4MINB: g5f_gu4 / g5f_down4 rows a block,
        // rows in flight a thread, min blocks per SM
        let k4r = rows("TANG_GLM5_MOE_K4R", 8).clamp(1, 64);
        let k4nr = rows("TANG_GLM5_MOE_K4NR", 4).clamp(1, 4);
        let k4dr = rows("TANG_GLM5_MOE_K4DR", 8).clamp(1, 64);
        let k4dnr = rows("TANG_GLM5_MOE_K4DNR", 4).clamp(1, 8);
        let k4minb = rows("TANG_GLM5_MOE_K4MINB", 4).clamp(1, 16);
        let k4ns = rows("TANG_GLM5_MOE_K4NS", 3).clamp(2, 8);
        let rrel = i32::from(env_on("TANG_GLM5_MOE_ROUTE_REL", true));
        let src = format!(
            "#define G5F_K4_R {k4r}\n#define G5F_K4_NR {k4nr}\n#define G5F_K4_DR {k4dr}\n#define G5F_K4_DNR {k4dnr}\n#define G5F_K4_MINB {k4minb}\n#define G5F_K4_NS {k4ns}\n#define G5F_GUX_R {}\n#define G5F_SEL2 {sel2}\n#define G5F_ROUTE_REL {rrel}\n#define G5F_LDU {ldu}\n#define G5F_MINB {minb}\n#define G5F_DOT32 {dot32}\n#define G5F_UNROLL {unroll}\n#define G5F_FUSE2 {fuse2}\n#define G5F_GUR {}\n#define G5F_DNR {}\n#define G5F_PF {pf}\n{}",
            gux.max(1),
            gur.max(1),
            dnr.max(2),
            super::SRC
        );
        let m = g.module(&src)?;
        let debug = std::env::var("TANG_GLM5_MOE_DEBUG").is_ok_and(|v| v == "1");
        let smem_optin = g
            .attribute(
                cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .unwrap_or(48 * 1024) as u32;
        Ok(Kernels {
            logits: m.func("g5_router_logits")?,
            select: m.func("g5_router_select")?,
            gu: m.func("g5_expert_gu")?,
            down: m.func("g5_expert_down")?,
            combine: m.func("g5_combine")?,
            f_q8: m.func("g5f_q8")?,
            f_router: m.func("g5f_router")?,
            f_select: m.func("g5f_select")?,
            f_gu: m.func("g5f_gu")?,
            f_down: m.func("g5f_down")?,
            f_down_split: m.func("g5f_down_split")?,
            f_sum: m.func("g5f_sum")?,
            f_route: m.func("g5f_route")?,
            f_gun: m.func("g5f_gun")?,
            f_down_splitn: m.func("g5f_down_splitn")?,
            f_gux: m.func("g5f_gux")?,
            f_gu4: m.func("g5f_gu4")?,
            f_down4: m.func("g5f_down4")?,
            k4: env_on("TANG_GLM5_MOE_K4", false),
            k4r,
            k4dr,
            k4nr,
            k4dnr,
            k4ns,
            gux,
            gur,
            dnr,
            route1: env_on("TANG_GLM5_MOE_ROUTE1", true),
            fast: std::env::var("TANG_GLM5_MOE").is_ok_and(|v| v == "fast"),
            stage: env_on("TANG_GLM5_MOE_STAGE", false),
            down_split: env_on("TANG_GLM5_MOE_DOWN_SPLIT", true),
            down_sw: match std::env::var("TANG_GLM5_MOE_DOWN_SW").ok().as_deref() {
                Some("1") => 1,
                Some("2") => 2,
                _ => 4,
            },
            act2: !std::env::var("TANG_GLM5_MOE_ACT").is_ok_and(|v| v == "q8"),
            poison: debug && env_on("TANG_GLM5_MOE_DEBUG_POISON", true),
            smem_optin,
            pool: RefCell::new(Pool::default()),
            _m: m,
        })
    }

    /// Registers per thread of (g5f_gu, g5f_down) — bench diagnostics.
    pub fn regs(&self) -> (i32, i32) {
        let r = |f: Func| {
            let mut v = 0i32;
            unsafe {
                cu::cuFuncGetAttribute(
                    &mut v,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
                    f,
                );
            }
            v
        };
        (r(self.f_gu), r(self.f_down))
    }

    /// The fast path's routing from the last MoE call (`[t][k]` ids, weights): the debug
    /// shadow's tap. `None` if the last call was not a fast MoE call of this shape.
    pub fn last_routing(&self, t: usize, k: usize) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        let p = self.pool.borrow();
        let (Some(Some(i)), Some(Some(w))) = (p.bufs.get(P_IDS), p.bufs.get(P_WTS)) else {
            return Ok(None);
        };
        if i.len < t * k * 4 || w.len < t * k * 4 {
            return Ok(None);
        }
        Ok(Some((i.read::<u32>(0, t * k)?, w.read::<f32>(0, t * k)?)))
    }
}

/// One weight on the device, with its kernel view.
pub struct DevW {
    pub buf: DevBuf,
    pub w: G5W,
}

fn ggml_id(t: &TensorInfo) -> Result<u32> {
    Ok(match t.ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q5_0 => 6,
        GgmlType::Q8_0 => 8,
        GgmlType::Q4K => 12,
        GgmlType::Q5K => 13,
        GgmlType::Q6K => 14,
        other => bail!(
            "{}: {} has no decoder in moe_cuda.cuh",
            t.name,
            other.name()
        ),
    })
}

/// Upload `w`, which holds `n_mat` matrices of `rows × cols`.
pub fn upload(w: &W, n_mat: usize, rows: usize, cols: usize) -> Result<DevW> {
    match *w {
        W::F32(v) => {
            ensure!(
                v.len() == n_mat * rows * cols,
                "f32 weight: {} values, want {n_mat}x{rows}x{cols}",
                v.len()
            );
            let buf = DevBuf::from_slice(v)?;
            let w = G5W {
                base: buf.ptr,
                estride: (rows * cols * 4) as u64,
                row_bytes: (cols * 4) as u32,
                ty: 0,
            };
            Ok(DevW { buf, w })
        }
        W::Gguf(g, t) => {
            ensure!(
                t.dims[0] as usize == cols && t.dims.get(1).copied().unwrap_or(1) as usize == rows,
                "{}: dims {:?}, want [{cols}, {rows}, ..]",
                t.name,
                t.dims
            );
            let rb = t.row_bytes()?;
            let buf = DevBuf::from_slice(g.bytes(t))?;
            let w = G5W {
                base: buf.ptr,
                estride: (rows * rb) as u64,
                row_bytes: rb as u32,
                ty: ggml_id(t)?,
            };
            Ok(DevW { buf, w })
        }
    }
}

/// Repacked row layouts (moe_cuda.cuh `G5_Q8R` / `G5_Q6R`): the same values with every 32-weight
/// quant payload 16-byte aligned, so `g5f_lane_dot32r` streams them as 16-byte loads.
pub const G5_Q8R: u32 = 108;
pub const G5_Q6R: u32 = 114;

/// One GGUF row of `k` weights (ggml type `ty`: 8 Q8_0 or 14 Q6_K) in its repacked layout,
/// appended to `out` (padded to `repacked_row_bytes`).
fn repack_row(ty: u32, row: &[u8], k: usize, out: &mut Vec<u8>) {
    let start = out.len();
    match ty {
        8 => {
            let nb = k / 32;
            for b in 0..nb {
                out.extend_from_slice(&row[b * 34 + 2..b * 34 + 34]);
            }
            for b in 0..nb {
                out.extend_from_slice(&row[b * 34..b * 34 + 2]);
            }
        }
        14 => {
            let nb = k / 256;
            for b in 0..nb {
                out.extend_from_slice(&row[b * 210..b * 210 + 192]);
            }
            for b in 0..nb {
                out.extend_from_slice(&row[b * 210 + 192..b * 210 + 208]);
            }
            for b in 0..nb {
                out.extend_from_slice(&row[b * 210 + 208..b * 210 + 210]);
            }
        }
        _ => unreachable!(),
    }
    out.resize(start + repacked_row_bytes(ty, k), 0);
}

fn repacked_row_bytes(ty: u32, k: usize) -> usize {
    let raw = if ty == 8 { k / 32 * 34 } else { k / 256 * 210 };
    raw.next_multiple_of(16)
}

/// `TANG_GLM5_MOE_REPACK`: repack the FFN's Q8_0 / Q6_K weights at upload. Default: on with the
/// round-4 kernels (`TANG_GLM5_MOE_K4=1`; g5f_gu4 / g5f_down4 take Q8_0 / Q6_K only
/// repacked), off otherwise (the B300 A/B with the round-2 kernels was 35.02 off
/// vs 33.51 on).
fn repack_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("TANG_GLM5_MOE_REPACK") {
        Ok(v) => v == "1",
        Err(_) => std::env::var("TANG_GLM5_MOE_K4").is_ok_and(|v| v == "1"),
    })
}

/// [`upload`] for the MoE / dense FFN weights: Q8_0 (k % 32 == 0) and Q6_K (k % 256 == 0) are
/// repacked to [`G5_Q8R`] / [`G5_Q6R`] (when [`repack_on`]); everything else as `upload`.
pub fn upload_ffn(w: &W, n_mat: usize, rows: usize, cols: usize) -> Result<DevW> {
    let W::Gguf(g, t) = *w else {
        return upload(w, n_mat, rows, cols);
    };
    let ty = ggml_id(t)?;
    let ok = match ty {
        8 => cols % 32 == 0,
        14 => cols % 256 == 0,
        _ => false,
    };
    if !ok || !repack_on() {
        return upload(w, n_mat, rows, cols);
    }
    ensure!(
        t.dims[0] as usize == cols && t.dims.get(1).copied().unwrap_or(1) as usize == rows,
        "{}: dims {:?}, want [{cols}, {rows}, ..]",
        t.name,
        t.dims
    );
    let rb = t.row_bytes()?;
    let src = g.bytes(t);
    ensure!(
        src.len() == n_mat * rows * rb,
        "{}: {} bytes, want {n_mat}x{rows}x{rb}",
        t.name,
        src.len()
    );
    let nrb = repacked_row_bytes(ty, cols);
    let mut out = Vec::with_capacity(n_mat * rows * nrb);
    for row in src.chunks_exact(rb) {
        repack_row(ty, row, cols, &mut out);
    }
    let buf = DevBuf::from_slice(&out)?;
    let w = G5W {
        base: buf.ptr,
        estride: (rows * nrb) as u64,
        row_bytes: nrb as u32,
        ty: if ty == 8 { G5_Q8R } else { G5_Q6R },
    };
    Ok(DevW { buf, w })
}

/// A layer's FFN on the device: geometry (weight handles cleared) plus uploaded weights.
pub struct DevMoe {
    pub p: MoeParams<'static>,
    pub router: Option<DevW>,
    pub bias: Option<DevBuf>,
    pub exps: Option<(DevW, DevW, DevW)>,
    pub sh: Option<(DevW, DevW, DevW)>,
    pub dense: Option<(DevW, DevW, DevW)>,
}

type Mlp = (DevW, DevW, DevW);

fn up3(
    g: Option<W>,
    u: Option<W>,
    d: Option<W>,
    n: usize,
    ff: usize,
    hid: usize,
) -> Result<Option<Mlp>> {
    let (Some(g), Some(u), Some(d)) = (g, u, d) else {
        return Ok(None);
    };
    Ok(Some((
        upload_ffn(&g, n, ff, hid)?,
        upload_ffn(&u, n, ff, hid)?,
        upload_ffn(&d, n, hid, ff)?,
    )))
}

impl DevMoe {
    pub fn new(p: &MoeParams) -> Result<Self> {
        let (hid, e) = (p.hidden, p.n_expert);
        let bias = match p.bias {
            Some(b) => Some(DevBuf::from_slice(&b.vec(e)?)?),
            None => None,
        };
        let geom = MoeParams {
            hidden: p.hidden,
            n_expert: p.n_expert,
            top_k: p.top_k,
            n_group: p.n_group,
            topk_group: p.topk_group,
            ff_exp: p.ff_exp,
            ff_shexp: p.ff_shexp,
            ff_dense: p.ff_dense,
            scale: p.scale,
            norm_topk_prob: p.norm_topk_prob,
            limit_exp: p.limit_exp,
            limit_shexp: p.limit_shexp,
            ..Default::default()
        };
        Ok(DevMoe {
            p: geom,
            router: p.router.map(|w| upload(&w, 1, e, hid)).transpose()?,
            bias,
            exps: up3(p.gate_exps, p.up_exps, p.down_exps, e, p.ff_exp, hid)?,
            sh: up3(p.sh_gate, p.sh_up, p.sh_down, 1, p.ff_shexp, hid)?,
            dense: up3(p.dense_gate, p.dense_up, p.dense_down, 1, p.ff_dense, hid)?,
        })
    }

    /// Weight bytes one token's FFN call reads: top_k experts' gate + up + down rows, the
    /// shared expert, the router (or the dense MLP). The bandwidth floor's numerator.
    pub fn active_bytes(&self) -> usize {
        let p = &self.p;
        let mlp = |m: &Mlp, ff: usize| {
            ff * (m.0.w.row_bytes as usize + m.1.w.row_bytes as usize)
                + p.hidden * m.2.w.row_bytes as usize
        };
        if let Some(d) = &self.dense {
            return mlp(d, p.ff_dense);
        }
        let r = self
            .router
            .as_ref()
            .map_or(0, |w| p.n_expert * w.w.row_bytes as usize);
        let e = self.exps.as_ref().map_or(0, |m| p.top_k * mlp(m, p.ff_exp));
        let s = self.sh.as_ref().map_or(0, |m| mlp(m, p.ff_shexp));
        r + e + s
    }
}

const BLOCK: (u32, u32, u32) = (128, 1, 1);

fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(128) as u32, 1, 1)
}

/// `out = MLP(x)` per token on matrix 0 of `m` (shared expert or dense FFN).
#[allow(clippy::too_many_arguments)]
fn mlp(
    k: &Kernels,
    s: &Stream,
    m: &Mlp,
    ff: usize,
    hid: usize,
    lim: f32,
    x: u64,
    out: u64,
    t_len: usize,
) -> Result<()> {
    let h = DevBuf::alloc(t_len * ff * 4)?;
    let (null, one, zero) = (0u64, 1i32, 0i32);
    let (t, ffi, hidi) = (t_len as i32, ff as i32, hid as i32);
    let (gw, uw, dw, hp) = (m.0.w, m.1.w, m.2.w, h.ptr);
    unsafe {
        // TANG_GLM5_TRACE: matrix 0's weight bytes (estride = one matrix)
        gpu::trace::bytes(gw.estride + uw.estride);
        gpu::launch(
            k.gu,
            grid(t_len * ff),
            BLOCK,
            0,
            s,
            tang_moe::args![gw, uw, null, one, zero, x, hp, t, ffi, hidi, lim],
        )?;
        gpu::trace::bytes(dw.estride);
        gpu::launch(
            k.down,
            grid(t_len * hid),
            BLOCK,
            0,
            s,
            tang_moe::args![dw, null, zero, hp, out, t, ffi, hidi],
        )?;
    }
    s.sync()?;
    Ok(())
}

/// `y = ffn(x)` for `t_len` tokens (dense or MoE); `x`, `y` are device `[t_len][hidden]` f32.
/// Synchronous (returns after the stream drains). Fast or naive per `k.fast`.
pub fn ffn(k: &Kernels, s: &Stream, m: &DevMoe, x: u64, y: u64, t_len: usize) -> Result<()> {
    if k.fast {
        let mut pool = k.pool.borrow_mut();
        ffn_fast(k, &mut pool, s, m, x, y, t_len)?;
        s.sync()?;
        return Ok(());
    }
    ffn_naive(k, s, m, x, y, t_len)
}

/// The fast path's `ffn` without the sync: queued on `s`, `x` / `y` device-resident, so a
/// caller that keeps the residual stream on the device (or captures a graph) pays no host
/// round trip for the layer. Errors when `k.fast` is off (naive syncs internally).
/// Debug: the routed expert ids of the last fast-path FFN (`n` u32s from the route
/// scratch; the caller syncs its stream first). `None` before any routed call.
pub fn last_route_ids(k: &Kernels, n: usize) -> Result<Option<Vec<u32>>> {
    let pool = k.pool.borrow();
    let Some(Some(b)) = pool.bufs.get(P_IDS) else {
        return Ok(None);
    };
    if b.len < n * 4 {
        return Ok(None);
    }
    Ok(Some(b.read::<u32>(0, n)?))
}

pub fn ffn_async(k: &Kernels, s: &Stream, m: &DevMoe, x: u64, y: u64, t_len: usize) -> Result<()> {
    ensure!(k.fast, "ffn_async: fast path only (TANG_GLM5_MOE=fast)");
    let mut pool = k.pool.borrow_mut();
    ffn_fast(k, &mut pool, s, m, x, y, t_len)
}

fn ffn_naive(k: &Kernels, s: &Stream, m: &DevMoe, x: u64, y: u64, t_len: usize) -> Result<()> {
    let p = &m.p;
    let hid = p.hidden;
    use super::super::trace::span;
    if let Some(d) = &m.dense {
        let _tr = span("dense");
        return mlp(k, s, d, p.ff_dense, hid, p.limit_shexp, x, y, t_len);
    }
    let router = m.router.as_ref().context("no router")?;
    let (gw, uw, dw) = m
        .exps
        .as_ref()
        .map(|(g, u, d)| (g.w, u.w, d.w))
        .context("no routed experts")?;
    let np = t_len * p.top_k;
    let sc = DevBuf::alloc(t_len * p.n_expert * 4)?;
    let ids = DevBuf::alloc(np * 4)?;
    let wts = DevBuf::alloc(np * 4)?;
    let h = DevBuf::alloc(np * p.ff_exp * 4)?;
    let yp = DevBuf::alloc(np * hid * 4)?;
    let ysh = match &m.sh {
        Some(sh) => {
            let _tr = span("moe.shared");
            let b = DevBuf::alloc(t_len * hid * 4)?;
            mlp(k, s, sh, p.ff_shexp, hid, p.limit_shexp, x, b.ptr, t_len)?;
            Some(b)
        }
        None => None,
    };
    let (t, ne, kk, hidi, ff, npi) = (
        t_len as i32,
        p.n_expert as i32,
        p.top_k as i32,
        hid as i32,
        p.ff_exp as i32,
        np as i32,
    );
    let (ng, tg, norm, scale) = (
        p.n_group as i32,
        p.topk_group as i32,
        p.norm_topk_prob as i32,
        p.scale,
    );
    let bias = m.bias.as_ref().map_or(0u64, |b| b.ptr);
    let yshp = ysh.as_ref().map_or(0u64, |b| b.ptr);
    let (rw, scp, idp, wtp, hp, ypp, lim) = (
        router.w,
        sc.ptr,
        ids.ptr,
        wts.ptr,
        h.ptr,
        yp.ptr,
        p.limit_exp,
    );
    // the experts a window can touch: every routed pair, at most every expert (an upper
    // bound on the bytes for prefill; exact for decode's distinct top-k)
    let n_touch = np.min(p.n_expert) as u64;
    unsafe {
        let tr = span("moe.router");
        gpu::trace::bytes(rw.estride);
        gpu::launch(
            k.logits,
            grid(t_len * p.n_expert),
            BLOCK,
            0,
            s,
            tang_moe::args![rw, x, scp, t, ne, hidi],
        )?;
        gpu::launch(
            k.select,
            grid(t_len),
            BLOCK,
            0,
            s,
            tang_moe::args![scp, bias, idp, wtp, t, ne, kk, ng, tg, norm, scale],
        )?;
        drop(tr);
        let tr = span("moe.experts");
        gpu::trace::bytes(n_touch * (gw.estride + uw.estride));
        gpu::launch(
            k.gu,
            grid(np * p.ff_exp),
            BLOCK,
            0,
            s,
            tang_moe::args![gw, uw, idp, kk, ne, x, hp, npi, ff, hidi, lim],
        )?;
        gpu::trace::bytes(n_touch * dw.estride);
        gpu::launch(
            k.down,
            grid(np * hid),
            BLOCK,
            0,
            s,
            tang_moe::args![dw, idp, ne, hp, ypp, npi, ff, hidi],
        )?;
        drop(tr);
        let _tr = span("moe.combine");
        gpu::launch(
            k.combine,
            grid(t_len * hid),
            BLOCK,
            0,
            s,
            tang_moe::args![ypp, wtp, yshp, y, t, kk, hidi],
        )?;
    }
    let _tr = span("sync");
    s.sync()?;
    Ok(())
}

/// Staging rows of `ws` (the widest row decides): uint4s per warp row, padded for the
/// aligned-down start and `g5f_rd64`'s 4-byte over-read.
fn row16(ws: &[G5W]) -> usize {
    ws.iter()
        .filter(|w| w.base != 0)
        .map(|w| (w.row_bytes as usize + 30) / 16 + 1)
        .max()
        .unwrap_or(0)
}

/// Dynamic smem for a 4-warp block staging two rows a warp, or `(0, 0)` (no staging) when it
/// exceeds the device's opt-in; sets the function's opt-in once when it passes 48 KB.
fn smem_for(
    k: &Kernels,
    pool: &mut Pool,
    which: usize,
    f: Func,
    r16: usize,
    pairs: usize,
) -> Result<(i32, u32)> {
    if !k.stage || r16 == 0 {
        return Ok((0, 0));
    }
    let bytes = (8 * pairs * r16 * 16) as u32;
    if bytes + 512 > k.smem_optin {
        return Ok((0, 0));
    }
    if bytes > 48 * 1024 && pool.smem_set[which] < bytes {
        gpu::check(
            unsafe {
                cu::cuFuncSetAttribute(
                    f,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
            },
            "cuFuncSetAttribute(g5f smem)",
        )?;
        pool.smem_set[which] = bytes;
    }
    Ok((r16 as i32, bytes))
}

/// Dynamic shared memory of g5f_gu4 / g5f_down4 (moe_cuda.cuh `g5f_k4_smem`): the cp.async ring
/// of the largest K4-eligible view (`(view, k)`), opted in past 48 KB. `m` matrices a batch.
fn k4_smem(
    k: &Kernels,
    pool: &mut Pool,
    which: usize,
    f: Func,
    views: &[(G5W, usize)],
    m: usize,
    nr: usize,
) -> Result<u32> {
    // gate|up (m == 2) keeps Q8_0r rows on the g5f_gux row code (moe_cuda.cuh g5f_k4_ok)
    let ok = |w: &G5W, kk: usize| {
        let nb = kk / 256;
        w.base != 0
            && (w.base | w.estride | w.row_bytes as u64) % 16 == 0
            && kk % 256 == 0
            && (1..=16).contains(&nb)
            && nb.is_power_of_two()
            && (matches!(w.ty, 12 | 13 | 114) || (w.ty == 108 && m == 1))
    };
    let bytes = views
        .iter()
        .filter(|(w, kk)| ok(w, *kk))
        .map(|(w, kk)| (k.k4ns * m * (4096 / kk) * nr * w.row_bytes as usize) as u32)
        .max()
        .unwrap_or(0);
    ensure!(
        bytes <= k.smem_optin,
        "g5f K4: {bytes} B of shared memory > opt-in {}",
        k.smem_optin
    );
    if bytes > 48 * 1024 && pool.smem_set[which] < bytes {
        gpu::check(
            unsafe {
                cu::cuFuncSetAttribute(
                    f,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
            },
            "cuFuncSetAttribute(g5f K4 smem)",
        )?;
        pool.smem_set[which] = bytes;
    }
    Ok(bytes)
}

fn nan_fill(ptr: u64, n_f32: usize, s: &Stream) -> Result<()> {
    Ok(gpu::check(
        unsafe { cu::cuMemsetD32Async(ptr, 0x7fc0_0000, n_f32, s.0) },
        "cuMemsetD32Async(NaN)",
    )?)
}

/// The fast path: q8(x) → [router → select] → grouped gate|up → q8(h) → grouped down + combine.
/// Async on `s` (no sync, no host copies: routing stays on the device); callers sync.
#[allow(clippy::too_many_arguments)]
fn ffn_fast(
    k: &Kernels,
    pool: &mut Pool,
    s: &Stream,
    m: &DevMoe,
    x: u64,
    y: u64,
    t_len: usize,
) -> Result<()> {
    let p = &m.p;
    let hid = p.hidden;
    let (gate, up, down, ff, lim, nr, routed) = match (&m.dense, &m.exps) {
        (Some(d), _) => (
            d.0.w,
            d.1.w,
            d.2.w,
            p.ff_dense,
            p.limit_shexp,
            1usize,
            false,
        ),
        (None, Some(e)) => (e.0.w, e.1.w, e.2.w, p.ff_exp, p.limit_exp, p.top_k, true),
        (None, None) => bail!("no dense FFN and no routed experts"),
    };
    let sh = if routed { m.sh.as_ref() } else { None };
    let (sg, su, sd, ffs) = sh.map_or((NO_W, NO_W, NO_W, 0), |s| (s.0.w, s.1.w, s.2.w, p.ff_shexp));
    ensure!(
        hid % 32 == 0 && ff % 32 == 0 && ffs % 32 == 0 && nr <= 32,
        "fast MoE: hid {hid}, ff {ff}, ff_sh {ffs} must be multiples of 32, top_k {nr} <= 32"
    );
    let slots = nr + usize::from(sh.is_some());
    let hs = ff.max(ffs);
    let t = t_len;
    let ti = t as i32;
    let (hidi, ffi, ffsi, hsi) = (hid as i32, ff as i32, ffs as i32, hs as i32);
    let blk = (128u32, 1u32, 1u32);

    // x -> int8 codes + per-32 scales
    let xq = pool.get(P_XQ, t * hid)?;
    let xd = pool.get(P_XD, t * hid / 32 * 4)?;
    let xq2 = if k.act2 {
        pool.get(P_XQ2, t * hid)?
    } else {
        0u64
    };
    let fused = routed
        && k.route1
        && m.router.as_ref().is_some_and(|r| r.w.ty == 0)
        && hid % 8 == 0
        && p.n_group <= 1
        && p.n_expert <= 1024
        && p.n_expert >= hid / 32;
    if !fused {
        unsafe {
            gpu::launch(
                k.f_q8,
                (hid.div_ceil(128) as u32, t as u32, 1),
                blk,
                0,
                s,
                tang_moe::args![x, xq, xq2, xd, hidi],
            )?;
        }
    }

    // routing
    let (idp, wtp, kk, ne) = if routed {
        let router = m.router.as_ref().context("no router")?;
        let ne = p.n_expert;
        let sc = pool.get(P_SC, t * ne * 4)?;
        let idp = pool.get(P_IDS, t * nr * 4)?;
        let wtp = pool.get(P_WTS, t * nr * 4)?;
        let (rw, nei) = (router.w, ne as i32);
        let bias = m.bias.as_ref().map_or(0u64, |b| b.ptr);
        let (kki, norm, scale) = (nr as i32, p.norm_topk_prob as i32, p.scale);
        if fused {
            if pool.cnt_zero < t {
                let c = pool.get(P_CNT, t * 4)?;
                gpu::check(
                    unsafe { cu::cuMemsetD32Async(c, 0, t, s.0) },
                    "cuMemsetD32Async(cnt)",
                )?;
                pool.cnt_zero = t;
            }
            let cnt = pool.get(P_CNT, t * 4)?;
            unsafe {
                gpu::launch(
                    k.f_route,
                    (ne as u32, t as u32, 1),
                    blk,
                    0,
                    s,
                    tang_moe::args![
                        rw, x, sc, nei, hidi, xq, xq2, xd, bias, idp, wtp, kki, norm, scale, cnt
                    ],
                )?;
            }
        } else {
            unsafe {
                if rw.ty == 0 && hid % 8 == 0 {
                    gpu::launch(
                        k.f_router,
                        (ne as u32, t as u32, 1),
                        blk,
                        0,
                        s,
                        tang_moe::args![rw, x, sc, nei, hidi],
                    )?;
                } else {
                    gpu::launch(
                        k.logits,
                        grid(t * ne),
                        BLOCK,
                        0,
                        s,
                        tang_moe::args![rw, x, sc, ti, nei, hidi],
                    )?;
                }
                if p.n_group <= 1 && ne <= 1024 {
                    gpu::launch(
                        k.f_select,
                        (t as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        s,
                        tang_moe::args![sc, bias, idp, wtp, nei, kki, norm, scale],
                    )?;
                } else {
                    let (ng, tg) = (p.n_group as i32, p.topk_group as i32);
                    gpu::launch(
                        k.select,
                        grid(t),
                        BLOCK,
                        0,
                        s,
                        tang_moe::args![sc, bias, idp, wtp, ti, nei, kki, ng, tg, norm, scale],
                    )?;
                }
            }
        }
        (idp, wtp, nr as i32, ne as i32)
    } else {
        (0u64, 0u64, 1i32, 1i32)
    };

    // grouped gate|up -> h, q8(h)
    let nh = t * slots * hs;
    let h = pool.get(P_H, nh * 4)?;
    let hq = pool.get(P_HQ, nh)?;
    let hd = pool.get(P_HD, nh / 32 * 4)?;
    let hq2 = if k.act2 { pool.get(P_HQ2, nh)? } else { 0u64 };
    if k.poison {
        nan_fill(h, nh, s)?;
        nan_fill(y, t * hid, s)?;
    }
    let (r16g, smg) = smem_for(k, pool, 0, k.f_gu, row16(&[gate, up, sg, su]), 1)?;
    let split = k.down_split && slots >= 4;
    let (r16d, smd) = if split {
        smem_for(k, pool, 2, k.f_down_split, row16(&[down, sd]), 1)?
    } else {
        smem_for(k, pool, 1, k.f_down, row16(&[down, sd]), 1)?
    };
    let lims = p.limit_shexp;
    // warps sharing a down row pair: the routed layers' 9 slots spread over a block's 4 warps
    // (TANG_GLM5_MOE_DOWN_SW overrides: 1, 2 or 4); dense has one slot, so one warp a pair
    let sw = if slots >= 4 { k.down_sw } else { 1 };
    let swi = sw as i32;
    // weight bytes the two grouped launches stream (per token; T tokens re-read them only
    // when they route to different experts — this is the T=1 figure x T, an upper bound)
    let rb = |w: G5W| w.row_bytes as u64;
    let (nr64, t64) = (nr as u64, t as u64);
    let gu_b = t64 * (nr64 * ff as u64 * (rb(gate) + rb(up)) + ffs as u64 * (rb(sg) + rb(su)));
    let dn_b = t64 * hid as u64 * (nr64 * rb(down) + rb(sd));
    unsafe {
        gpu::trace::bytes(gu_b);
        if k.k4 && smg == 0 {
            let sm4 = k4_smem(
                k,
                pool,
                3,
                k.f_gu4,
                &[(gate, hid), (up, hid), (sg, hid), (su, hid)],
                2,
                k.k4nr,
            )?;
            gpu::launch(
                k.f_gu4,
                (hs.div_ceil(k.k4r) as u32, slots as u32, t as u32),
                blk,
                sm4,
                s,
                tang_moe::args![
                    gate, up, sg, su, idp, kk, ne, x, xq, xq2, xd, h, hsi, ffi, ffsi, hidi, lim,
                    lims
                ],
            )?;
        } else if k.gux > 0 && smg == 0 {
            gpu::launch(
                k.f_gux,
                (hs.div_ceil(k.gux) as u32, slots as u32, t as u32),
                blk,
                0,
                s,
                tang_moe::args![
                    gate, up, sg, su, idp, kk, ne, x, xq, xq2, xd, h, hsi, ffi, ffsi, hidi, lim,
                    lims
                ],
            )?;
        } else if k.gur > 0 && smg == 0 {
            gpu::launch(
                k.f_gun,
                (hs.div_ceil(4 * k.gur) as u32, slots as u32, t as u32),
                blk,
                0,
                s,
                tang_moe::args![
                    gate, up, sg, su, idp, kk, ne, x, xq, xq2, xd, h, hsi, ffi, ffsi, hidi, lim,
                    lims
                ],
            )?;
        } else {
            gpu::launch(
                k.f_gu,
                (hs.div_ceil(4) as u32, slots as u32, t as u32),
                blk,
                smg,
                s,
                tang_moe::args![
                    gate, up, sg, su, idp, kk, ne, x, xq, xq2, xd, h, hsi, ffi, ffsi, hidi, lim,
                    lims, r16g
                ],
            )?;
        }
        gpu::launch(
            k.f_q8,
            (hs.div_ceil(128) as u32, (t * slots) as u32, 1),
            blk,
            0,
            s,
            tang_moe::args![h, hq, hq2, hd, hsi],
        )?;
        gpu::trace::bytes(dn_b);
        if split {
            let yp = pool.get(P_YP, t * slots * hid * 4)?;
            if k.k4 && smd == 0 {
                let sm4 = k4_smem(k, pool, 4, k.f_down4, &[(down, ff), (sd, ffs)], 1, k.k4dnr)?;
                gpu::launch(
                    k.f_down4,
                    (hid.div_ceil(k.k4dr) as u32, slots as u32, t as u32),
                    blk,
                    sm4,
                    s,
                    tang_moe::args![
                        down, sd, idp, wtp, kk, ne, h, hq, hq2, hd, hsi, yp, ffi, ffsi, hidi
                    ],
                )?;
            } else if k.dnr > 0 && smd == 0 {
                gpu::launch(
                    k.f_down_splitn,
                    (hid.div_ceil(4 * k.dnr) as u32, slots as u32, t as u32),
                    blk,
                    0,
                    s,
                    tang_moe::args![
                        down, sd, idp, wtp, kk, ne, h, hq, hq2, hd, hsi, yp, ffi, ffsi, hidi
                    ],
                )?;
            } else {
                gpu::launch(
                    k.f_down_split,
                    (hid.div_ceil(8) as u32, slots as u32, t as u32),
                    blk,
                    smd,
                    s,
                    tang_moe::args![
                        down, sd, idp, wtp, kk, ne, h, hq, hq2, hd, hsi, yp, ffi, ffsi, hidi, r16d
                    ],
                )?;
            }
            let si = slots as i32;
            gpu::launch(
                k.f_sum,
                (hid.div_ceil(128) as u32, t as u32, 1),
                blk,
                0,
                s,
                tang_moe::args![yp, y, si, hidi],
            )?;
        } else {
            gpu::launch(
                k.f_down,
                (hid.div_ceil(8 / sw) as u32, t as u32, 1),
                blk,
                smd,
                s,
                tang_moe::args![
                    down, sd, idp, wtp, kk, ne, h, hq, hq2, hd, hsi, y, ffi, ffsi, hidi, r16d, swi
                ],
            )?;
        }
    }
    Ok(())
}

/// Convenience for tests and W1's first wiring: host `x` in, host `y` out. The fast path
/// reuses its device x / y and stream across calls; the naive path allocates per call (as
/// it always has — part of the baseline it measures).
pub fn ffn_host(k: &Kernels, m: &DevMoe, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
    ensure!(x.len() == t_len * m.p.hidden, "ffn_host: x size");
    if k.fast {
        let mut pool = k.pool.borrow_mut();
        let s = match pool.stream.take() {
            Some(s) => s,
            None => Stream::blocking()?,
        };
        let r = (|| {
            let xp = {
                let b = pool.buf(P_X, x.len() * 4)?;
                b.write(0, x)?;
                b.ptr
            };
            let yp = pool.get(P_Y, x.len() * 4)?;
            ffn_fast(k, &mut pool, &s, m, xp, yp, t_len)?;
            s.sync()?;
            Ok(pool.buf(P_Y, x.len() * 4)?.read::<f32>(0, x.len())?)
        })();
        pool.stream = Some(s);
        return r;
    }
    let s = Stream::blocking()?;
    let xd = DevBuf::from_slice(x)?;
    let yd = DevBuf::zeroed(x.len() * 4)?;
    ffn_naive(k, &s, m, xd.ptr, yd.ptr, t_len)?;
    Ok(yd.read::<f32>(0, x.len())?)
}

/// The KDA projection kernels ([`super::K5_SRC`] = the MoE source + `kda_k5.cuh`, so
/// `g5_dot`/`g5_deq` are in scope): `gemv` (one thread per output), `gate` (the forget
/// gate's finish: dt_bias + the safe-gate/softplus form), `beta` (the β GEMV + sigmoid),
/// `norm` (the gated output norm). Compiled once per arch; one instance serves every
/// KDA layer.
pub struct Kernels5 {
    _m: Module,
    pub gemv: Func,
    pub gemv8: Func,
    pub q8: Func,
    pub gate: Func,
    pub beta: Func,
    pub norm: Func,
    /// The fast path's plain GEMV (`g5f_gemv` + `g5f_q8`), for the KDA projections.
    pub fast: FastGemv,
    /// `k5_qgemv9` / `k5_qgemv17`: the multi-matrix T=1 Q8_0 GEMV (kda_k5.cuh).
    pub qgemv9: Func,
    pub qgemv17: Func,
    /// `k5_qgemv_q4k5` / `k5_qgemv_q4k9`: the same launch for Q4_K rows of 4096 / 8192.
    pub qgemv_q4k5: Func,
    pub qgemv_q4k9: Func,
}

impl Kernels5 {
    pub fn new(g: &Gpu) -> Result<Self> {
        let m = g.module(super::K5_SRC)?;
        Ok(Kernels5 {
            fast: FastGemv::new(g, &m)?,
            gemv: m.func("k5_gemv")?,
            gemv8: m.func("k5_gemv8")?,
            q8: m.func("k5_q8")?,
            gate: m.func("k5_gate_finish")?,
            beta: m.func("k5_beta")?,
            norm: m.func("k5_gated_norm")?,
            qgemv9: m.func("k5_qgemv9")?,
            qgemv17: m.func("k5_qgemv17")?,
            qgemv_q4k5: m.func("k5_qgemv_q4k5")?,
            qgemv_q4k9: m.func("k5_qgemv_q4k9")?,
            _m: m,
        })
    }
}

/// Activations for [`FastGemv::gemv`], device pointers, `xs` elements per token: `xq` / `xq2`
/// / `xd` = [`FastGemv::q8`] of x (`xq2 = 0`: one-level codes, ~1e-2 activation error instead
/// of ~1e-5; k5_q8's codes + scales also fit: `xq2 = 0`, `xd` = its scales, `xs` = COLS).
/// `xf` = x in f32, read only by F32 / F16 rows (0 otherwise).
#[derive(Clone, Copy, Debug, Default)]
pub struct Act {
    pub xf: u64,
    pub xq: u64,
    pub xq2: u64,
    pub xd: u64,
    pub xs: usize,
}

/// The plain GEMV on the fast MoE path's dequant-reuse row kernel (`g5f_gemv`): two rows a
/// warp staged in shared memory (cp.async), int8 activations and `__dp4a` for quantized
/// rows, any G5W type; a token tile of 8 reads W once per 8 tokens. Launches are async on
/// the caller's stream (no sync).
pub struct FastGemv {
    pub gemv: Func,
    pub q8: Func,
    /// Stage rows in shared memory (`TANG_GLM5_MOE_STAGE=0`: read them from global).
    pub stage: bool,
    smem_optin: u32,
}

impl FastGemv {
    pub fn new(g: &Gpu, m: &Module) -> Result<Self> {
        Ok(FastGemv {
            gemv: m.func("g5f_gemv")?,
            q8: m.func("g5f_q8")?,
            stage: !std::env::var("TANG_GLM5_MOE_STAGE").is_ok_and(|v| v == "0"),
            smem_optin: g
                .attribute(
                    cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                )
                .unwrap_or(48 * 1024) as u32,
        })
    }

    /// `x [t][xs]` f32 → `xq [t][xs]` codes, `xd [t][xs/32]` scales, `xq2` residual codes
    /// (0 = skip). Quantises the first `k` of each token (`k % 32 == 0`, `k <= xs`).
    pub fn q8(
        &self,
        s: &Stream,
        x: u64,
        xq: u64,
        xq2: u64,
        xd: u64,
        k: usize,
        t: usize,
    ) -> Result<()> {
        ensure!(k % 32 == 0, "g5f_q8: k {k} not a multiple of 32");
        let ki = k as i32;
        unsafe {
            gpu::launch(
                self.q8,
                (k.div_ceil(128) as u32, t as u32, 1),
                (128, 1, 1),
                0,
                s,
                tang_moe::args![x, xq, xq2, xd, ki],
            )?;
        }
        Ok(())
    }

    /// `y[t * ystride + r] = (row r of matrix e of w) · x_t` for `r < rows`, `t < t_len`;
    /// `cols` = the row length (`% 8 == 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        s: &Stream,
        w: G5W,
        e: u32,
        a: Act,
        y: u64,
        ystride: usize,
        t_len: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        ensure!(
            cols % 8 == 0 && a.xs >= cols && a.xs % 32 == 0,
            "g5f_gemv: cols {cols}, xs {}",
            a.xs
        );
        ensure!(
            if w.ty <= 1 {
                a.xf != 0
            } else {
                a.xq != 0 && a.xd != 0
            },
            "g5f_gemv: ggml type {} needs {} activations",
            w.ty,
            if w.ty <= 1 { "f32" } else { "int8" }
        );
        // staging: two rows a warp, 4 warps (row16 / smem_for's rule), only up to 48 KB a
        // block: past that occupancy drops and staging loses (H200, o_proj 4096x8192 Q8_0,
        // 70 KB: 35.8 us staged vs 23.8 us from global; q/k/v 8192x4096, 35 KB: 23.4 vs 25.9)
        let r16 = if self.stage { row16(&[w]) } else { 0 };
        let bytes = (8 * r16 * 16) as u32;
        let cap = self.smem_optin.min(48 * 1024);
        let (r16, bytes) = if r16 == 0 || bytes > cap {
            (0, 0)
        } else {
            (r16, bytes)
        };
        let (xf, xq, xq2, xd) = (a.xf, a.xq, a.xq2, a.xd);
        let (xs, ys, ti, ri, ci, r16i) = (
            a.xs as i32,
            ystride as i32,
            t_len as i32,
            rows as i32,
            cols as i32,
            r16 as i32,
        );
        unsafe {
            gpu::launch(
                self.gemv,
                (rows.div_ceil(8) as u32, t_len.div_ceil(8) as u32, 1),
                (128, 1, 1),
                bytes,
                s,
                tang_moe::args![w, e, xf, xq, xq2, xd, xs, y, ys, ti, ri, ci, r16i],
            )?;
        }
        Ok(())
    }
}
