//! The KDA layers resident on the GPU: W4's conv + delta-rule core ([`DevKda`]) plus the
//! projections, gates and gated output norm as device kernels — the engine's wiring of
//! everything W4 built.
//!
//! Before this module the engine's `kda_gpu` (engine.rs) staged every weight per window
//! (`DevBuf::from_slice` of `w.q`/`k`/`v`/… on the host) and ran the projections as the
//! CPU f64 `matmul` — ~8 GFLOP of scalar host work per 64-token window per layer, 34
//! layers: the 22 s/window profile. The fix is the DevMoe shape (W5):
//!
//! - weights uploaded ONCE at engine load, in their GGUF storage format (Q8_0 stays Q8_0;
//!   [`super::moe::gpu::upload`] + the `G5W` view, dequantised element-wise in the
//!   kernel) — 67 MB/layer instead of 268 MB f32;
//! - one [`DevLayer`] per KDA layer holding the conv history + recurrent state + scratch
//!   between windows on its own persistent stream;
//! - per window the host traffic is `x` in, the core I/O W4's launchers define, and `y`
//!   out — no weight uploads, no per-window allocations.
//!
//! Kernels (K5_SRC, appended to [`super::moe::SRC`] so W5's dequant table is reused):
//! `k5_gemv` (one thread per output, G5W row dot, destination base + stride so the q/k/v
//! projections write the stacked q|k|v layout directly), `k5_gate_finish` (dt_bias + the
//! safe-gate/softplus form), `k5_beta` (the β GEMV + sigmoid), `k5_gated_norm` (the host
//! `rms_norm_mul`'s f64 mean-of-squares exactly, times `o_norm` and `sigmoid(g2)`).
//!
//! State ownership: the device holds the live state. The engine's host
//! `(ConvState, KdaState)` stays authoritative only across a CPU-fallback window
//! (`TANG_GLM5_GPU_KDA=0`, a layer past [`Self::max_t`]): [`Self::read_back`] pulls the
//! device state before the CPU runs, and [`Self::seed`] pushes it back before the GPU
//! resumes (the engine tracks this with a per-layer `host_fresh` flag). Decode also
//! updates the host conv copy as a side effect (W4's design); the recurrent state does
//! not. `read_back`/`seed` are cheap (288 KiB + 4 MiB) and only run on the fallback path.
//!
//! Later (FAST, after the gates): keep qkv/g/beta on the device across the core call
//! (extends DevKda's launchers to take device pointers — kills the per-chunk host
//! round-trip); tensor-core GEMV for prefill; CUDA graphs.
#![allow(clippy::too_many_arguments)]
use super::kda::{ConvState, KdaConfig, KdaState};

/// warps per `k5_gemv8` block (the kernel's K5_WARPS; one row a warp).
const K5_WARPS: usize = 4;
use anyhow::{ensure, Context, Result};
use tang_moe::gpu::{DevBuf, Gpu, Stream};
use tang_moe::{args, gpu::launch};

use super::kda_cuda::cuda::{DevKda, Kernels};

/// The fused + graph-captured decode step (TANG_GLM5_KDA_FUSED / TANG_GLM5_KDA_GRAPH).
#[path = "kda_decode.rs"]
pub mod decode;

/// The KDA projection kernels, appended to [`super::moe::SRC`] (the G5W dequant table is
/// defined there). NAIVE ON PURPOSE (GATES.md): one thread per output value, plain loops,
/// f32 accumulate. All layouts match the host 1:1 so parity is a value compare:
///
/// - `k5_gemv(W, E, X, Y, YSTRIDE, T, ROWS, COLS)`: `y[t*ystride + r] = dot(W row r of
///   matrix e, x[t])` — the host `matmul`'s f32 form (the host accumulates f64; parity
///   comes from the engine test's 1e-3 bar). `ystride = ROWS` is the plain layout; the
///   q/k/v projections write the stacked q|k|v layout directly (`ystride = 3·d_inner`,
///   base offsets `q=0, k=d_inner, v=2·d_inner` — the host's interleave loop, gone).
///   Grid `ceil(T*ROWS/128)`, block 128.
/// - `k5_gate_finish(G1, DT, RATE, NH, HD, LB, T)`: in place on the forget gate's raw
///   GEMV output `g1[t][nh*hd]`: `+ dt_bias[chan]`, then `lb·σ(rate[h]·pre)` when
///   `lb >= 0` (GLM-5.3-Flash passes −5.0), else the Qwen3-Next softplus form — the
///   caller passes a NaN for `None` (safe: a real lower bound is −5, never NaN).
/// - `k5_beta(BP, X, BETA, T, HID, NH)`: `beta[t][h] = σ(dot(b_proj row h, x[t]))`.
/// - `k5_gated_norm(O, ON, G2, Y, NH, HD, EPS, T)`: the host's `rms_norm_mul` (f64 sum
///   of squares over the head, the exact form) times `o_norm[j]`, then
///   `y[t][c] = that · σ(g2[t][c])`, where `g2 = g_b(g_a(x))` was produced by two
///   ordinary `k5_gemv` launches into the `g2` scratch (`g_a`: `[t][hd]`, then `g_b`
///   in place on a separate buffer, so no read/write race).

/// Everything of one KDA layer on one GPU, built once at engine load.
pub struct DevLayer {
    /// The conv + delta-rule core (W4's `DevKda`): state lives on device between windows.
    core: DevKda,
    /// Projection weights in GGUF storage format: q/k/v `[d_inner][hidden]`,
    /// o_proj `[hidden][d_inner]`.
    wq: super::moe::gpu::DevW,
    wk: super::moe::gpu::DevW,
    wv: super::moe::gpu::DevW,
    wo: super::moe::gpu::DevW,
    /// The gates' GEMV weights: f_a `[hd][hidden]`, f_b `[d_inner][hd]`,
    /// b_proj `[nh][hidden]`, g_a `[hd][hidden]`, g_b `[d_inner][hd]`.
    fa: super::moe::gpu::DevW,
    fb: super::moe::gpu::DevW,
    bproj: super::moe::gpu::DevW,
    ga: super::moe::gpu::DevW,
    gb: super::moe::gpu::DevW,
    /// Small exact vectors: dt_bias `[d_inner]`, decay_rate `[nh]` (= `−ssm_a`),
    /// o_norm `[hd]`.
    dt: DevBuf,
    rate: DevBuf,
    on: DevBuf,
    /// Per-window scratch (device, persistent): `x` `[t][hidden]`, the stacked qkv
    /// `[t][3·d_inner]`, the forget gate's raw GEMV `g1` `[t][d_inner]`, the `f_b` GEMV
    /// output `g` (separate from `g1`: the GEMV writes a buffer it does not read),
    /// `beta` `[t][nh]`, the core output `o` `[t][d_inner]`, `g2` `[t][d_inner]`, the
    /// gated-norm output, and the layer output `y` `[t][hidden]`.
    xb: DevBuf,
    qkvb: DevBuf,
    g1b: DevBuf,
    gb_out: DevBuf,
    bb: DevBuf,
    ob: DevBuf,
    g2a: DevBuf,
    g2b: DevBuf,
    yb: DevBuf,
    outb: DevBuf,
    /// k5d_pret's per-row scratch [max_t][4][d_inner] (q, k, v, expf(gate)).
    rsc: DevBuf,
    /// int8 activations for `k5_gemv8` (codes + scales; see the xq8 field).
    xq8: DevBuf,
    /// The layer's persistent stream (reads host-staged buffers, hence blocking).
    s: Stream,
    kers: Kernels,
    /// The projection kernels (`K5_SRC` compiled per arch; the G5W dequant table comes
    /// from the MoE source it is appended to).
    gv: super::moe::gpu::Kernels5,
    cfg: KdaConfig,
    /// The layer's conv weights on the host (for the <64-token CPU tail path).
    conv_w: Vec<f32>,
    hidden: usize,
    d_inner: usize,
    nh: usize,
    hd: usize,
    max_t: usize,
    /// `TANG_GLM5_KDA_PF2=1` (read at load): the device-resident core — conv in place on
    /// `qkvb`, the parallel chunked prefill ([`super::kda_cuda::PF2_SRC`]) for any `t`
    /// (no CPU tail), the decode step from device pointers; qkv/g/beta never visit the
    /// host. Off: the original host-staged path (64-chunks on the device + CPU tail).
    pf2: bool,
    /// `TANG_GLM5_KDA_DEBUG` / `TANG_KERNEL_VERIFY`: when the engine sets a probe for
    /// this window, [`Self::window`] poisons its scratch with NaN first (per the probe) and
    /// leaves every intermediate in the probe's taps for the shadow check
    /// ([`super::kda_debug`], names in [`super::kda_debug::tap`]). `None`: zero cost.
    pub probe: Option<tang_verify::Probe>,
    /// The fused decode kernels + captured graph ([`decode`]); built at load when the env
    /// selects a fused mode, else `None` (the naive path).
    fast: Option<decode::Fast>,
    /// `TANG_GLM5_GEMV8=1` at construction: the q/k/v projections on k5_gemv8 (a field,
    /// not an env read per window, so a test can A/B one layer in-process).
    pub gemv8: bool,
    /// `TANG_GLM5_KDA_G5F` at construction (default 0 = off): x's projections (q|k|v,
    /// f_a, g_a) and o_proj on the fast row kernels instead of k5_gemv8 / k5_gemv, on
    /// BOTH the naive and the fused path (the per-matrix rule: one kernel per matrix).
    /// 1 = `g5f_gemv` (one launch per matrix, k5_q8's codes); 2 = `k5_qgemv` (q|k|v|f_a|
    /// g_a in one launch, T = 1, Q8_0 or Q4_K — any other window or type takes 1).
    /// Takes precedence over `gemv8` for those six matrices.
    pub g5f: u8,
    /// k5_qgemv segment tables (G5F=2, built at load when every matrix is Q8_0 (or every one Q4_K) with
    /// 16-byte-aligned rows): x's five matrices, o_proj.
    segx: Option<DevBuf>,
    sego: Option<DevBuf>,
}

/// The G5W descriptor of one `KdaLayerW` matrix, from the GGUF's raw tensor (the weights
/// stay quantised on the device; the kernel dequantises element-wise). `rows × cols` in
/// HF orientation (`[out][in]`), which for GGUF ne dims `[cols, rows]` is
/// `rows = n_rows()`, `cols = row_len()`.
fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(128) as u32, 1, 1)
}

fn dev_w(g: &crate::gguf::Gguf, name: &str) -> Result<super::moe::gpu::DevW> {
    let t = g.info(name).with_context(|| format!("k5 {name}"))?;
    super::moe::gpu::upload(&super::moe::W::Gguf(g, t), 1, t.n_rows(), t.row_len())
}

impl DevLayer {
    /// Build from the GGUF directly (the device reads its own bytes; the CPU loader's
    /// dequantised copy stays for the fallback path and the parity tests).
    /// `conv`/`state` seed the device (the engine passes the zeros it would give the CPU
    /// path at load). `g` must already be bound to this device.
    #[allow(clippy::too_many_arguments)]
    pub fn new_gguf(
        g: &Gpu,
        gguf: &crate::gguf::Gguf,
        l: usize,
        cfg: &KdaConfig,
        hidden: usize,
        conv: &ConvState,
        state: &KdaState,
        max_t: usize,
        slots: usize,
    ) -> Result<Self> {
        let (nh, hd) = (cfg.heads, cfg.head_dim);
        let d_inner = nh * hd;
        let t = |s: &str| format!("blk.{l}.{s}");
        let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("k5 upload: {e}");
        let kers = super::kda_cuda::cuda::load(g, cfg)?;
        let gv = super::moe::gpu::Kernels5::new(g)?;
        let upf = |name: &str| -> Result<DevBuf> {
            let ti = gguf.info(&t(name))?;
            ensure!(
                ti.ty == crate::gguf::GgmlType::F32,
                "{}: want F32 for the small vectors",
                t(name)
            );
            let v = gguf.tensor_f32(&t(name))?;
            DevBuf::from_slice(&v).map_err(gerr)
        };
        let conv_w = {
            // the converter splits HF's fused qkv conv per q|k|v; the CPU loader stacks
            // them in q,k,v order — the core kernel indexes the same stack
            let mut c = Vec::with_capacity(3 * d_inner * cfg.d_conv);
            for n in [
                "ssm_conv1d_q.weight",
                "ssm_conv1d_k.weight",
                "ssm_conv1d_v.weight",
            ] {
                c.extend_from_slice(&gguf.tensor_f32(&t(n))?);
            }
            c
        };
        let rate = {
            // GGUF `ssm_a` = `-exp(A_log)`; the gate's `rate` = `exp(A_log)` = `−ssm_a`
            let r = gguf.tensor_f32(&t("ssm_a"))?;
            let neg: Vec<f32> = r.iter().map(|&a| -a).collect();
            DevBuf::from_slice(&neg).map_err(gerr)
        };
        let slots = conv.0.len() / (cfg.conv_dim() * (cfg.d_conv - 1));
        let pf2 = std::env::var("TANG_GLM5_KDA_PF2").is_ok_and(|v| v == "1");
        let mut core = DevKda::new(g, &kers, cfg, &conv_w, conv, state, slots)?;
        if pf2 {
            core.enable_pf2(max_t)?;
        }
        Self {
            core,
            wq: dev_w(gguf, &t("attn_q.weight"))?,
            wk: dev_w(gguf, &t("attn_k.weight"))?,
            wv: dev_w(gguf, &t("attn_v.weight"))?,
            wo: dev_w(gguf, &t("attn_output.weight"))?,
            fa: dev_w(gguf, &t("ssm_f_a.weight"))?,
            fb: dev_w(gguf, &t("ssm_f_b.weight"))?,
            bproj: dev_w(gguf, &t("ssm_beta.weight"))?,
            ga: dev_w(gguf, &t("ssm_g_a.weight"))?,
            gb: dev_w(gguf, &t("ssm_g_b.weight"))?,
            dt: upf("ssm_dt.bias")?,
            rate: rate?,
            on: upf("ssm_norm.weight")?,
            xb: DevBuf::zeroed_elems::<f32>(max_t * hidden).map_err(gerr)?,
            qkvb: DevBuf::zeroed_elems::<f32>(max_t * 3 * d_inner).map_err(gerr)?,
            g1b: DevBuf::zeroed_elems::<f32>(max_t * hd).map_err(gerr)?,
            gb_out: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            bb: DevBuf::zeroed_elems::<f32>(max_t * nh).map_err(gerr)?,
            ob: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            g2a: DevBuf::zeroed_elems::<f32>(max_t * hd).map_err(gerr)?,
            g2b: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            yb: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            outb: DevBuf::zeroed_elems::<f32>(max_t * hidden).map_err(gerr)?,
            rsc: DevBuf::zeroed_elems::<f32>(max_t * 4 * d_inner).map_err(gerr)?,
            // TANG_GLM5_GEMV8: int8 codes [t][hidden] + f32 per-32 scales [t][hidden/32]
            // (k5_q8's layout; the codes+scales in one buffer, scales after T*hidden codes)
            xq8: {
                let w = hidden.max(d_inner); // o_proj quantises yb (d_inner wide)
                DevBuf::zeroed_elems::<u8>(max_t * w + max_t * w.div_ceil(32) * 4).map_err(gerr)?
            },
            s: Stream::blocking().map_err(gerr)?,
            kers,
            gv,
            cfg: *cfg,
            conv_w,
            hidden,
            d_inner,
            nh,
            hd,
            max_t,
            pf2,
            probe: None,
            fast: {
                let mode = decode::Mode::from_env();
                decode::log_load(mode);
                match mode {
                    decode::Mode::Naive => None,
                    _ => Some(decode::Fast::new(g, cfg)?),
                }
            },
            gemv8: std::env::var("TANG_GLM5_GEMV8").is_ok_and(|v| v == "1"),
            g5f: std::env::var("TANG_GLM5_KDA_G5F")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            segx: None,
            sego: None,
        }
        .with_seg_tables()
    }

    /// Build the k5_qgemv segment tables (only with `g5f == 2`; `None` = fall back to
    /// g5f_gemv). Each entry: base, y, row_bytes | rows << 32 (the kernel's K5Seg).
    fn with_seg_tables(mut self) -> Result<Self> {
        if self.g5f < 2 {
            return Ok(self);
        }
        let f = std::mem::size_of::<f32>() as u64;
        let di = self.d_inner as u64;
        let ok = |ws: &[&super::moe::gpu::DevW]| {
            let rb = ws[0].w.row_bytes;
            // Q8_0 (k5_qgemv9/17) or Q4_K (k5_qgemv_q4k5/9), one type per table
            let cap = match ws[0].w.ty {
                8 => 17,
                12 => 9,
                _ => 0,
            } * 32
                * 16;
            ws.iter().all(|w| {
                w.w.ty == ws[0].w.ty
                    && w.w.base % 16 == 0
                    && w.w.row_bytes == rb
                    && rb % 16 == 0
                    && rb <= cap
            })
        };
        let tab = |e: &[(&super::moe::gpu::DevW, u64, usize)]| -> Result<DevBuf> {
            let mut v: Vec<u64> = Vec::new();
            for (w, y, rows) in e {
                v.extend_from_slice(&[w.w.base, *y, w.w.row_bytes as u64 | ((*rows as u64) << 32)]);
            }
            DevBuf::from_slice(&v).map_err(|e| anyhow::anyhow!("k5_qgemv table: {e}"))
        };
        if ok(&[&self.wq, &self.wk, &self.wv, &self.fa, &self.ga]) {
            let (q, hd) = (self.qkvb.ptr, self.hd);
            self.segx = Some(tab(&[
                (&self.wq, q, self.d_inner),
                (&self.wk, q + di * f, self.d_inner),
                (&self.wv, q + 2 * di * f, self.d_inner),
                (&self.fa, self.g1b.ptr, hd),
                (&self.ga, self.g2a.ptr, hd),
            ])?);
        }
        if ok(&[&self.wo]) {
            self.sego = Some(tab(&[(&self.wo, self.outb.ptr, self.hidden)])?);
        }
        Ok(self)
    }

    /// k5_qgemv over a segment table (T = 1): `nseg` matrices of `rows_total` rows of
    /// `rb` bytes, all reading x's k5_q8 codes. Warps a block: as many staged rows as fit
    /// 48 KB, at most 8.
    unsafe fn qgemv_launch(
        &self,
        s: &Stream,
        tab: &DevBuf,
        nseg: usize,
        rb: u32,
        rows_total: usize,
        bytes: u64,
        codes: u64,
        scales: u64,
        cols: usize,
    ) -> Result<()> {
        let nw = (49152 / rb as usize).clamp(1, 8);
        // a Q4_K row is cols/256 super-blocks of 144 B (a Q8_0 row: cols/32 blocks of 34 B)
        let f = match (rb as usize == cols / 256 * 144, rb as usize <= 9 * 32 * 16) {
            (true, true) if rb as usize <= 5 * 32 * 16 => self.gv.qgemv_q4k5,
            (true, _) => self.gv.qgemv_q4k9,
            (false, true) => self.gv.qgemv9,
            (false, false) => self.gv.qgemv17,
        };
        // the Q8_0 kernels stage nothing (kda_k5.cuh): 4 warps a block, no shared memory
        let (nw, smem) = if rb as usize == cols / 256 * 144 {
            (nw, nw as u32 * rb)
        } else {
            (4, 0)
        };
        // two rows a warp on k5_qgemv9 (bit 31 of nseg; every segment's rows are even here:
        // di, hd), same per-row chains: B300 23.4 -> 22.1 us. Not on k5_qgemv17 (o_proj:
        // 11.8 -> 12.7 us, 2 x 81 words a lane). TANG_GLM5_KDA_QR2=0: one row a warp.
        static QR2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let qr2 = smem == 0
            && rb as usize <= 9 * 32 * 16
            && *QR2.get_or_init(|| !std::env::var("TANG_GLM5_KDA_QR2").is_ok_and(|v| v == "0"));
        let (rows_total, nseg) = if qr2 {
            (rows_total.div_ceil(2), nseg as u32 | 0x8000_0000)
        } else {
            (rows_total, nseg as u32)
        };
        tang_moe::gpu::trace::bytes(bytes);
        unsafe {
            launch(
                f,
                (rows_total.div_ceil(nw) as u32, 1, 1),
                ((32 * nw) as u32, 1, 1),
                smem,
                s,
                args![tab.ptr, nseg, codes, scales, cols as u32],
            )?;
        }
        Ok(())
    }

    /// TANG_GLM5_KDA_G5F: quantise x (k5_q8 into xq8), then q|k|v → qkvb, f_a → g1b,
    /// g_a → g2a on the fast kernels. `false` (nothing launched) when the flag is off.
    unsafe fn g5f_x(&self, s: &Stream, t: usize) -> Result<bool> {
        if self.g5f == 0 {
            return Ok(false);
        }
        let (di, hd, hidden) = (self.d_inner, self.hd, self.hidden);
        let f = std::mem::size_of::<f32>() as u64;
        let codes = self.xq8.ptr;
        let scales = codes + (self.max_t * hidden) as u64;
        unsafe {
            launch(
                self.gv.q8,
                ((hidden / 32) as u32, t as u32, 1),
                (32, 1, 1),
                0,
                s,
                args![self.xb.ptr, codes, hidden as u32, self.max_t as u32],
            )?;
            if let (1, Some(tab)) = (t, self.segx.as_ref()) {
                let bytes = [&self.wq, &self.wk, &self.wv, &self.fa, &self.ga]
                    .iter()
                    .map(|w| w.w.estride)
                    .sum();
                return self
                    .qgemv_launch(
                        s,
                        tab,
                        5,
                        self.wq.w.row_bytes,
                        3 * di + 2 * hd,
                        bytes,
                        codes,
                        scales,
                        hidden,
                    )
                    .map(|_| true);
            }
            let a = super::moe::gpu::Act {
                xf: self.xb.ptr,
                xq: codes,
                xq2: 0,
                xd: scales,
                xs: hidden,
            };
            let q = self.qkvb.ptr;
            for (w, y, ys, rows) in [
                (&self.wq, q, 3 * di, di),
                (&self.wk, q + di as u64 * f, 3 * di, di),
                (&self.wv, q + 2 * di as u64 * f, 3 * di, di),
                (&self.fa, self.g1b.ptr, hd, hd),
                (&self.ga, self.g2a.ptr, hd, hd),
            ] {
                tang_moe::gpu::trace::bytes(w.w.estride);
                self.gv.fast.gemv(s, w.w, 0, a, y, ys, t, rows, hidden)?;
            }
        }
        Ok(true)
    }

    /// TANG_GLM5_KDA_G5F: o_proj — quantise yb (d_inner wide) into xq8, then wo → outb.
    /// `false` when the flag is off.
    unsafe fn g5f_o(&self, s: &Stream, t: usize) -> Result<bool> {
        if self.g5f == 0 {
            return Ok(false);
        }
        let (di, hidden) = (self.d_inner, self.hidden);
        let codes = self.xq8.ptr;
        let scales = codes + (self.max_t * di) as u64;
        unsafe {
            launch(
                self.gv.q8,
                ((di / 32) as u32, t as u32, 1),
                (32, 1, 1),
                0,
                s,
                args![self.yb.ptr, codes, di as u32, self.max_t as u32],
            )?;
            if let (1, Some(tab)) = (t, self.sego.as_ref()) {
                return self
                    .qgemv_launch(
                        s,
                        tab,
                        1,
                        self.wo.w.row_bytes,
                        hidden,
                        self.wo.w.estride,
                        codes,
                        scales,
                        di,
                    )
                    .map(|_| true);
            }
            let a = super::moe::gpu::Act {
                xf: self.yb.ptr,
                xq: codes,
                xq2: 0,
                xd: scales,
                xs: di,
            };
            tang_moe::gpu::trace::bytes(self.wo.w.estride);
            self.gv
                .fast
                .gemv(s, self.wo.w, 0, a, self.outb.ptr, hidden, t, hidden, di)?;
        }
        Ok(true)
    }

    /// Pull the device's conv + recurrent state into the host copies (before a CPU
    /// fallback window on this layer; cheap, 288 KiB + 4 MiB). Slot 0 (the single-sequence
    /// callers: tests, the bench).
    pub fn read_back(&self, conv: &mut ConvState, state: &mut KdaState) -> Result<()> {
        self.read_back_at(0, conv, state)
    }

    /// Slot `slot`'s device KDA state regions (conv history, recurrent state) as
    /// `(device ptr, bytes)`.
    pub fn state_regions(&self, slot: usize) -> [(u64, usize); 2] {
        self.core.slot_state_regions(slot)
    }

    /// [`Self::read_back`] of sequence `slot`'s state.
    pub fn read_back_at(
        &self,
        slot: usize,
        conv: &mut ConvState,
        state: &mut KdaState,
    ) -> Result<()> {
        self.s.sync()?;
        self.core.read_conv_at(slot, conv)?;
        self.core.read_state_at(slot, state)?;
        Ok(())
    }

    /// Push the host copies back onto the device (after a CPU fallback window; the device
    /// takes over from the host's values). Slot 0.
    pub fn seed(&self, conv: &ConvState, state: &KdaState) -> Result<()> {
        self.seed_at(0, conv, state)
    }

    /// [`Self::seed`] of sequence `slot`'s state.
    pub fn seed_at(&self, slot: usize, conv: &ConvState, state: &KdaState) -> Result<()> {
        self.core.seed_at(slot, conv, state)
    }

    /// Run one window (`x` `[t][hidden]`, `t ≤ max_t`) through the layer. The host's
    /// `conv` is updated in place (W4's launchers keep the host conv copy fresh); the
    /// recurrent state lives only on the device except across the CPU tail, which reads
    /// it back and seeds it forward.

    /// `y[t*ystride + r] = W row r · x[t]` on k5_gemv8, for a Q8_0 `w` with `cols`-wide
    /// rows; `xq8` holds x's k5_q8 codes (scales at `max_t * cols` bytes after them).
    /// One row per warp (K5_WARPS warps a block); dynamic smem = one staged row per warp
    /// plus one granule (an unaligned row spans row16 + 1 granules).
    unsafe fn gemv8_launch(
        &self,
        w: &super::moe::gpu::DevW,
        xq8: u64,
        cols: usize,
        y: u64,
        ystride: usize,
        rows: usize,
        t: usize,
    ) -> Result<()> {
        unsafe { self.gemv8_launch_on(&self.s, w, xq8, cols, y, ystride, rows, t) }
    }

    /// [`Self::gemv8_launch`] on stream `s` (the fused decode enqueues on its caller's
    /// stream, which a whole-step graph capture records).
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemv8_launch_on(
        &self,
        s: &Stream,
        w: &super::moe::gpu::DevW,
        xq8: u64,
        cols: usize,
        y: u64,
        ystride: usize,
        rows: usize,
        t: usize,
    ) -> Result<()> {
        let scales = xq8 + (self.max_t * cols) as u64;
        let smem = (K5_WARPS * ((w.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32;
        // TANG_GLM5_TRACE: the weight bytes (the stream a decode GEMV is bound by)
        tang_moe::gpu::trace::bytes(w.w.estride);
        unsafe {
            launch(
                self.gv.gemv8,
                (rows.div_ceil(K5_WARPS) as u32, t as u32, 1),
                (128, 1, 1),
                smem,
                s,
                args![
                    w.w,
                    0u32,
                    xq8,
                    scales,
                    y,
                    ystride as u32,
                    t as u32,
                    rows as u32,
                    cols as u32
                ],
            )?;
        }
        Ok(())
    }

    pub fn window(&mut self, x: &[f32], t: usize, conv: &mut ConvState) -> Result<Vec<f32>> {
        self.window_at(0, x, t, conv)
    }

    /// [`Self::window`] for sequence `slot` (its device conv + recurrent state, one slice
    /// of the layer's `[slots][...]` state buffers). The projection scratch is shared: one
    /// window at a time runs per layer, so no slot partitioning is needed there.
    pub fn window_at(
        &mut self,
        slot: usize,
        x: &[f32],
        t: usize,
        conv: &mut ConvState,
    ) -> Result<Vec<f32>> {
        self.window_impl(&[slot], x, t, conv)
    }

    /// Whether the device-resident core is on (`TANG_GLM5_KDA_PF2=1` at load): the
    /// batched decode rows need it.
    pub fn pf2_on(&self) -> bool {
        self.pf2
    }

    /// The batched decode window: row `i` of `x` (`[t][hidden]`, `t = slots.len()`) is ONE
    /// token of sequence `slots[i]`. The projections, gates, gated norm and o_proj run once
    /// over all `t` rows (the weight reads amortise across the batch); only the core — the
    /// conv + delta-rule step — runs per row, on that row's slot state. Needs the
    /// device-resident core (`TANG_GLM5_KDA_PF2=1`); the slots must be distinct.
    pub fn window_rows(&mut self, slots: &[usize], x: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            self.pf2,
            "batched KDA rows need the device-resident core (TANG_GLM5_KDA_PF2=1)"
        );
        ensure!(!slots.is_empty(), "empty batch");
        let mut dummy = ConvState(Vec::new());
        self.window_impl(slots, x, slots.len(), &mut dummy)
    }

    /// `slots.len() == 1`: a window of `t` consecutive tokens of `slots[0]`;
    /// `slots.len() == t > 1`: one decode row per slot ([`Self::window_rows`]).
    fn window_impl(
        &mut self,
        slots: &[usize],
        x: &[f32],
        t: usize,
        conv: &mut ConvState,
    ) -> Result<Vec<f32>> {
        ensure!(
            slots.len() == 1 || slots.len() == t,
            "k5 rows: {} slots for {t} rows",
            slots.len()
        );
        let rows = slots.len() > 1;
        let slot = slots[0];
        ensure!(x.len() == t * self.hidden, "k5 layer input");
        ensure!(t <= self.max_t, "k5 window {t} > {max}", max = self.max_t);
        ensure!(
            !rows || self.probe.is_none(),
            "k5 rows: no shadow check on batched rows"
        );
        if t == 1 && self.fast.is_some() {
            let mode = decode::Mode::from_env();
            if mode != decode::Mode::Naive {
                return self.decode_fast_at(slot, x, conv, mode);
            }
        }
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        let lb = match self.cfg.gate_lower_bound {
            Some(v) => v,
            None => f32::NAN, // the kernel's softplus-form sentinel
        };
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the layer input, matching kda::kda_layer's `kda cpu x` line: if x already
            // differs, the divergence is upstream of KDA (not a staging bug)
            eprintln!("kda gpu x[0..8]: {:?}", &x[..8.min(x.len())]);
        }
        use super::trace::span;
        use tang_moe::gpu::trace::bytes as tr_bytes;
        let tr = span("proj");
        self.xb.write(0, x)?;
        let mut tp = self.probe.take();
        if let Some(p) = &tp {
            // NaN-fill every output scratch this window writes (whole buffers): a region
            // no kernel touches then reads NaN in the shadow check, not a plausible 0.0.
            // Includes the core's own scratch (conv output, core output) and PF2's
            // workspace (the pf_state temporaries).
            p.poison(&[
                &self.qkvb,
                &self.g1b,
                &self.gb_out,
                &self.bb,
                &self.g2a,
                &self.g2b,
                &self.yb,
                &self.outb,
                &self.ob,
                &self.core.yb,
                &self.core.ob,
                &self.core.ws,
            ])?;
        }
        let prof = std::env::var("TANG_GLM5_PROF").is_ok_and(|v| v == "1");
        let t_win = std::time::Instant::now();
        // projections → the stacked q|k|v layout (base offsets q=0, k=di, v=2·di FLOATS).
        // The device pointer is a BYTE address: the offsets must be scaled by
        // size_of::<f32>(). Unscaled (the GLM-5.3 staging bug, Oct 2026) k landed di/4 and
        // v di/2 floats into the q block — q corrupted past 2048, half of k and all of v
        // never written (zeros on a fresh buffer → the core output was exactly 0).
        let qkv = self.qkvb.ptr;
        let f = std::mem::size_of::<f32>() as u64;
        // TANG_GLM5_GEMV8=1: the dequant-reuse GEMV (k5_gemv8, docs/models/glm5-speed-plan.md
        // step 1) for the Q8_0 projections. x is quantised once per window (k5_q8), then
        // every GEMV reads the int8 codes; one row per warp, the whole 8.7 KB row in
        // flight through shared memory. The gates' small GEMVs (f_a: 128 outs, beta: 64)
        // stay on k5_gemv — too few rows to amortise the staging.
        // gemv8 is Q8_0-only (its dp4a reads 34-byte Q8_0 blocks): any other weight type —
        // e.g. the f32 tiny fixture — must take k5_gemv, or the kernel reads f32 rows as
        // Q8_0 blocks (NaN logits on the first tiny run)
        // PER MATRIX: gemv8 only takes Q8_0 rows. An all-or-nothing guard turned gemv8 off
        // for the whole layer when one matrix wasn't Q8_0 (the tiny fixture's f_a/g_a are
        // F32), which made the in-process A/B test compare naive against naive.
        let g8 = |w: &super::moe::gpu::DevW| self.gemv8 && w.w.ty == 8;
        let gemv8 = g8(&self.wq) && g8(&self.wk) && g8(&self.wv); // the x pre-pass + q/k/v
        unsafe {
            let g5 = self.g5f_x(&self.s, t)?;
            if g5 {
                // TANG_GLM5_KDA_G5F: q|k|v, f_a, g_a all written above
            } else if gemv8 {
                launch(
                    self.gv.q8,
                    ((hidden / 32) as u32, t as u32, 1),
                    (32, 1, 1),
                    0,
                    &self.s,
                    args![self.xb.ptr, self.xq8.ptr, hidden as u32, self.max_t as u32], // MAXT: the codes array's row cap (scales follow it), NOT t
                )?;
                for (w, off) in [
                    (&self.wq, 0u64),
                    (&self.wk, di as u64 * f),
                    (&self.wv, 2 * di as u64 * f),
                ] {
                    self.gemv8_launch(w, self.xq8.ptr, hidden, qkv + off, 3 * di, di, t)?;
                }
            } else {
                for (w, off) in [
                    (&self.wq, 0u64),
                    (&self.wk, di as u64 * f),
                    (&self.wv, 2 * di as u64 * f),
                ] {
                    tr_bytes(w.w.estride);
                    launch(
                        self.gv.gemv,
                        grid(t * di),
                        (128, 1, 1),
                        0,
                        &self.s,
                        args![
                            w.w,
                            0u32,
                            self.xb.ptr,
                            qkv + off,
                            (3 * di) as u32,
                            t as u32,
                            di as u32,
                            hidden as u32
                        ],
                    )?;
                }
            }
            drop(tr);
            let tr = span("gates");
            // forget gate: f_a → g1, f_b: g1 → g (out of place), finish in place on g
            if g5 {
                // f_a went with q/k/v (g5f_x)
            } else if gemv8 && g8(&self.fa) {
                // x's int8 codes are still in xq8 from the q/k/v pre-pass
                self.gemv8_launch(&self.fa, self.xq8.ptr, hidden, self.g1b.ptr, hd, hd, t)?;
            } else {
                tr_bytes(self.fa.w.estride);
                launch(
                    self.gv.gemv,
                    grid(t * hd),
                    (128, 1, 1),
                    0,
                    &self.s,
                    args![
                        self.fa.w,
                        0u32,
                        self.xb.ptr,
                        self.g1b.ptr,
                        hd as u32,
                        t as u32,
                        hd as u32,
                        hidden as u32
                    ],
                )?;
            }
            tr_bytes(self.fb.w.estride);
            launch(
                self.gv.gemv,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![
                    self.fb.w,
                    0u32,
                    self.g1b.ptr,
                    self.gb_out.ptr,
                    di as u32,
                    t as u32,
                    di as u32,
                    hd as u32
                ],
            )?;
            launch(
                self.gv.gate,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![
                    self.gb_out.ptr,
                    self.dt.ptr,
                    self.rate.ptr,
                    nh as u32,
                    hd as u32,
                    lb,
                    t as u32
                ],
            )?;
            // beta gate
            tr_bytes(self.bproj.w.estride);
            launch(
                self.gv.beta,
                // one WARP per (t, h), K5_WARPS warps a 128-thread block: ceil(t·nh / 4)
                // BLOCKS. (`grid(n)` makes ceil(n/128) blocks — handed the block count it
                // launched 1/128 of them, so beta was computed for 4 (t,h) per 128 and the
                // rest of `bb` kept stale values: 158 top-1 flips GPU vs CPU on tiny.)
                ((t * nh).div_ceil(K5_WARPS) as u32, 1, 1),
                (128, 1, 1),
                0,
                &self.s,
                args![
                    self.bproj.w,
                    self.xb.ptr,
                    self.bb.ptr,
                    t as u32,
                    hidden as u32,
                    nh as u32
                ],
            )?;
            drop(tr);
        }
        let tr = span("core");
        if self.pf2 {
            if prof {
                self.s.sync()?;
                eprintln!(
                    "glm5: prof kda proj: {:.2} ms",
                    t_win.elapsed().as_secs_f32() * 1e3
                );
            }
            let t_core = std::time::Instant::now();
            if rows {
                self.core_rows(slots)?;
            } else {
                self.core_pf2(slot, t, conv, &mut tp)?;
            }
            if prof {
                self.s.sync()?;
                eprintln!(
                    "glm5: prof kda core: {:.2} ms",
                    t_core.elapsed().as_secs_f32() * 1e3
                );
            }
            drop(tr);
            return self.finish(t, None, tp);
        }
        self.s.sync()?;
        if prof {
            eprintln!(
                "glm5: prof kda proj: {:.2} ms",
                t_win.elapsed().as_secs_f32() * 1e3
            );
        }
        let t_core = std::time::Instant::now();
        ensure!(
            !rows,
            "k5 rows: the host-staged core runs one sequence (TANG_GLM5_KDA_PF2=1)"
        );
        // the core (W4's launchers stage their inputs from host): qkv, g, beta
        let qkv_h = self.qkvb.read::<f32>(0, t * 3 * di)?;
        let g_h = self.gb_out.read::<f32>(0, t * di)?;
        let beta_h = self.bb.read::<f32>(0, t * nh)?;
        if let Some(p) = tp.as_mut() {
            use super::kda_debug::tap;
            p.put(tap::QKV, qkv_h.clone());
            p.put(tap::G, g_h.clone());
            p.put(tap::BETA, beta_h.clone());
            // the device conv path's output is only observable per 64-chunk (core.yb),
            // filled below; decode fuses the conv into the step kernel (no conv tap)
        }
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the staging probe (matches kda::kda_layer's CPU dump; layout/units equal):
            // qkv[0..16] (token 0), g[0..8], beta[0..4], the device state's first 8,
            // and the host conv copy's first 4
            let st_h = self.core.st.read::<f32>(0, 8).unwrap_or_else(|e| {
                // surface read errors loudly: a silent empty vec here looks like a zero
                // state and wasted a diagnosis cycle (every state line printed 0.0)
                eprintln!("kda gpu state READ ERROR: {e}");
                Vec::new()
            });
            eprintln!(
                "kda gpu qkv[0..16]: {:?}\n kda gpu qkv_k[0..8]: {:?}\n kda gpu qkv_v[0..8]: {:?}\n kda gpu g[0..8]: {:?}\n kda gpu beta[0..4]: {:?}\n kda gpu state[0..8]: {:?}\n kda gpu conv_host[0..4]: {:?}",
                &qkv_h[..16.min(qkv_h.len())],
                &qkv_h[di..di + 8.min(qkv_h.len() - di)],
                &qkv_h[2 * di..2 * di + 8.min(qkv_h.len() - 2 * di)],
                &g_h[..8.min(g_h.len())],
                &beta_h[..4.min(beta_h.len())],
                &st_h,
                &conv.0[..4.min(conv.0.len())],
            );
        }
        // W4's core decode/prefill index state.head(cfg, slot, h) on the host even when the
        // device owns the state — hand it a correctly-sized zero buffer so the indexing is
        // in-bounds (the values are ignored: the device-side state is authoritative).
        let mut dummy_state = KdaState(vec![
            0f32;
            self.cfg.heads * self.cfg.head_dim * self.cfg.head_dim
        ]);
        let mut o = vec![0f32; t * di];
        if t == 1 {
            o.copy_from_slice(&self.core.decode(
                &self.kers,
                &self.cfg,
                &qkv_h,
                &g_h,
                &beta_h,
                conv,
                &mut dummy_state,
                1,
            )?);
        } else {
            let mut off = 0usize;
            while off + 64 <= t {
                let o64 = self.core.prefill(
                    &self.kers,
                    &self.cfg,
                    &qkv_h[off * 3 * di..(off + 64) * 3 * di],
                    &g_h[off * di..(off + 64) * di],
                    &beta_h[off * nh..(off + 64) * nh],
                    conv,
                    &mut dummy_state,
                    1,
                    64,
                )?;
                o[off * di..(off + 64) * di].copy_from_slice(&o64);
                if let Some(p) = tp.as_mut() {
                    p.read_at(
                        super::kda_debug::tap::CONV,
                        off * 3 * di,
                        &self.core.yb,
                        64 * 3 * di,
                    )?;
                }
                off += 64;
            }
            // a tail shorter than one chunk: the CPU core on the device's state —
            // read it back, run the CPU chunk, seed the result forward (tails < 64 only)
            if off < t {
                let rem = t - off;
                let (mut hconv, mut hstate) = (
                    conv.clone(),
                    KdaState(vec![
                        0f32;
                        self.cfg.heads * self.cfg.head_dim * self.cfg.head_dim
                    ]),
                );
                self.read_back_at(slot, &mut hconv, &mut hstate)?;
                let yc = super::kda::conv1d_prefill(
                    &qkv_h[off * 3 * di..t * 3 * di],
                    &self.conv_w,
                    &mut hconv,
                    &self.cfg,
                    1,
                    rem,
                )?;
                let oc = super::kda::kda_chunk(
                    &yc,
                    &g_h[off * di..t * di],
                    &beta_h[off * nh..t * nh],
                    &mut hstate,
                    &self.cfg,
                    1,
                    rem,
                    64,
                )?;
                o[off * di..t * di].copy_from_slice(&oc);
                if let Some(p) = tp.as_mut() {
                    p.taps
                        .put_at(super::kda_debug::tap::CONV, off * 3 * di, &yc);
                }
                self.seed_at(slot, &hconv, &hstate)?;
                *conv = hconv;
            }
        }
        self.ob.write(0, &o)?;
        if prof {
            eprintln!(
                "glm5: prof kda core: {:.2} ms",
                t_core.elapsed().as_secs_f32() * 1e3
            );
        }
        drop(tr);
        self.finish(t, Some(o), tp)
    }

    /// The per-row core of a batched decode window: row `i`'s qkv / g / beta (already
    /// projected for all rows) step slot `slots[i]`'s conv + recurrent state, writing row
    /// `i` of `ob`. ONE `kda_step` launch for all rows + one batched conv commit when the
    /// rows are in slot order (the usual case: `window_batch` walks slots 0..n).
    fn core_rows(&mut self, slots: &[usize]) -> Result<()> {
        let (nh, di) = (self.nh, self.d_inner);
        let f = std::mem::size_of::<f32>() as u64;
        // kda_step indexes the state by `blockIdx.y` — which is the ROW index, so it hits
        // slot i's state only when row i IS slot i. Otherwise the per-row fallback.
        if slots.iter().enumerate().all(|(i, &s)| s == i) {
            self.core.decode_dev_rows(
                &self.kers,
                &self.s,
                slots,
                self.qkvb.ptr,
                self.gb_out.ptr,
                self.bb.ptr,
                self.ob.ptr,
            )?;
            return Ok(());
        }
        for (i, &slot) in slots.iter().enumerate() {
            let i = i as u64;
            self.core.decode_dev(
                &self.kers,
                &self.s,
                slot,
                self.qkvb.ptr + i * (3 * di) as u64 * f,
                self.gb_out.ptr + i * di as u64 * f,
                self.bb.ptr + i * nh as u64 * f,
                self.ob.ptr + i * di as u64 * f,
            )?;
        }
        Ok(())
    }

    /// The PF2 core on the device: qkv (pre-conv, in `qkvb`), g (`gb_out`) and beta (`bb`)
    /// straight from the projection kernels into the core, the core output into `ob` —
    /// all on the layer stream, no host round trip. Under the shadow check the inputs and
    /// the conv output (in place in `qkvb`) are read back as taps, and the host conv copy
    /// is mirrored from the device (this path doesn't maintain it: the device is
    /// authoritative and `read_back` syncs it before any CPU use).
    fn core_pf2(
        &mut self,
        slot: usize,
        t: usize,
        conv: &mut ConvState,
        tp: &mut Option<tang_verify::Probe>,
    ) -> Result<()> {
        let (nh, di) = (self.nh, self.d_inner);
        if let Some(p) = tp.as_mut() {
            use super::kda_debug::tap;
            self.s.sync()?;
            p.read(tap::QKV, &self.qkvb, t * 3 * di)?;
            p.read(tap::G, &self.gb_out, t * di)?;
            p.read(tap::BETA, &self.bb, t * nh)?;
        }
        // TANG_GLM5_KDA_WIN_STEP=N: windows of t <= N run the core as t recurrent decode
        // steps (the t=1 kernel, token by token) instead of the chunked prefill form, so a
        // short verify window (MTP) computes exactly what t single decode steps would.
        // Unset, it defaults to `win_step_default()` (MTP depth + 1 under --mtp, else 0).
        let win_step = super::mtp::win_step();
        if t == 1 || t <= win_step {
            let f = std::mem::size_of::<f32>() as u64;
            for i in 0..t as u64 {
                self.core.decode_dev(
                    &self.kers,
                    &self.s,
                    slot,
                    self.qkvb.ptr + i * (3 * di) as u64 * f,
                    self.gb_out.ptr + i * di as u64 * f,
                    self.bb.ptr + i * nh as u64 * f,
                    self.ob.ptr + i * di as u64 * f,
                )?;
            }
        } else {
            self.core.prefill_dev(
                &self.kers,
                &self.s,
                slot,
                self.qkvb.ptr,
                3 * di,
                self.gb_out.ptr,
                di,
                self.bb.ptr,
                nh,
                self.ob.ptr,
                t,
            )?;
        }
        if let Some(p) = tp.as_mut() {
            use super::kda_debug::tap;
            self.s.sync()?;
            // the conv ran in place on qkvb — the post-conv qkv IS the conv output tap
            if t > 1 {
                p.read_at(tap::CONV, 0, &self.qkvb, t * 3 * di)?;
            }
            self.core.read_conv_at(slot, conv)?;
        }
        Ok(())
    }

    /// The gated output norm + o_proj from the core output in `ob` (`o`: the host copy
    /// when the host-staged path produced it), then the taps.
    fn finish(
        &mut self,
        t: usize,
        o: Option<Vec<f32>>,
        tp: Option<tang_verify::Probe>,
    ) -> Result<Vec<f32>> {
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        // same per-matrix rule as window(): g_a reuses x's codes from the q/k/v pre-pass,
        // which ran only if q/k/v all took gemv8; o_proj quantises its own input
        let g8 = |w: &super::moe::gpu::DevW| self.gemv8 && w.w.ty == 8;
        let gemv8 = g8(&self.wq) && g8(&self.wk) && g8(&self.wv);
        let prof = std::env::var("TANG_GLM5_PROF").is_ok_and(|v| v == "1");
        let t_out = std::time::Instant::now();
        // gated output norm (device) then o_proj
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the core output pre-norm (the CPU tail path's `o`): y-divergence with a
            // matching o pins the bug to the gated norm or the o_proj GEMV
            let o = match &o {
                Some(o) => o.clone(),
                None => self.ob.read::<f32>(0, t * di)?,
            };
            eprintln!("kda gpu o[0..8]: {:?}", &o[..8.min(o.len())]);
        }
        use super::trace::span;
        use tang_moe::gpu::trace::bytes as tr_bytes;
        let tr = span("out_gate");
        unsafe {
            // g2 = g_b(g_a(x)): g_a → g2a, g_b: g2a → g2b (out of place)
            if self.g5f > 0 {
                // g_a was written into g2a by g5f_x in window() (x's codes are gone by now)
            } else if gemv8 && g8(&self.ga) {
                // x's codes are still in xq8 (nothing has re-quantised since the pre-pass;
                // the o_proj pass below is the first to overwrite them)
                self.gemv8_launch(&self.ga, self.xq8.ptr, hidden, self.g2a.ptr, hd, hd, t)?;
            } else {
                tr_bytes(self.ga.w.estride);
                launch(
                    self.gv.gemv,
                    grid(t * hd),
                    (128, 1, 1),
                    0,
                    &self.s,
                    args![
                        self.ga.w,
                        0u32,
                        self.xb.ptr,
                        self.g2a.ptr,
                        hd as u32,
                        t as u32,
                        hd as u32,
                        hidden as u32
                    ],
                )?;
            }
            tr_bytes(self.gb.w.estride);
            launch(
                self.gv.gemv,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![
                    self.gb.w,
                    0u32,
                    self.g2a.ptr,
                    self.g2b.ptr,
                    di as u32,
                    t as u32,
                    di as u32,
                    hd as u32
                ],
            )?;
            launch(
                self.gv.norm,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![
                    self.ob.ptr,
                    self.on.ptr,
                    self.g2b.ptr,
                    self.yb.ptr,
                    nh as u32,
                    hd as u32,
                    self.cfg.eps,
                    t as u32
                ],
            )?;
            drop(tr);
            let _tr = span("o_proj");
            if self.g5f_o(&self.s, t)? {
            } else if g8(&self.wo) {
                // o_proj reads yb (d_inner wide): quantise it, then the dequant-reuse GEMV
                launch(
                    self.gv.q8,
                    ((di / 32) as u32, t as u32, 1),
                    (32, 1, 1),
                    0,
                    &self.s,
                    args![self.yb.ptr, self.xq8.ptr, di as u32, self.max_t as u32],
                )?;
                self.gemv8_launch(&self.wo, self.xq8.ptr, di, self.outb.ptr, hidden, hidden, t)?;
            } else {
                tr_bytes(self.wo.w.estride);
                launch(
                    self.gv.gemv,
                    grid(t * hidden),
                    (128, 1, 1),
                    0,
                    &self.s,
                    args![
                        self.wo.w,
                        0u32,
                        self.yb.ptr,
                        self.outb.ptr,
                        hidden as u32,
                        t as u32,
                        hidden as u32,
                        di as u32
                    ],
                )?;
            }
        }
        // the layer's one wait: the stream drains here (async kernels' wall time lands in
        // this span; their GPU time is on the spans that launched them)
        let tr = span("sync+read");
        self.s.sync()?;
        if prof {
            eprintln!(
                "glm5: prof kda out: {:.2} ms",
                t_out.elapsed().as_secs_f32() * 1e3
            );
        }
        let out = self.outb.read::<f32>(0, t * hidden)?;
        drop(tr);
        if let Some(mut p) = tp {
            use super::kda_debug::tap;
            // the host-staged path hands back the core output it read; PF2 leaves it in
            // `ob` (device-resident) — read it here for the tap either way
            let o = match o {
                Some(o) => o,
                None => self.ob.read::<f32>(0, t * di)?,
            };
            p.put(tap::O, o);
            p.read(tap::G2, &self.g2b, t * di)?;
            p.read(tap::Y_NORM, &self.yb, t * di)?;
            p.put(tap::OUT, out.clone());
            self.probe = Some(p);
        }
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the layer OUTPUT (post gated-norm + o_proj): pins the divergence to the
            // core (ob/yb) or the output path (norm/wo) when x already matches
            eprintln!("kda gpu y[0..8]: {:?}", &out[..8.min(out.len())]);
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::super::kda::{ConvState, KdaState};
    use super::DevLayer;

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut n) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += ((x - y) as f64).powi(2);
            n += (*y as f64).powi(2);
        }
        if a.iter().any(|x| !x.is_finite()) {
            return f64::INFINITY;
        }
        (d / n.max(1e-30)).sqrt()
    }

    /// Real GLM-5.3 KDA layer 0 (`TANG_GLM5_BENCH_GGUF` = the first shard), one resident
    /// `DevLayer` per path: the original host-staged core (64-chunks + CPU tail) vs PF2.
    /// Times whole windows (proj + core + out, `TANG_GLM5_PROF=1` prints the split) at
    /// 20 / 512 / 2048 tokens, checks PF2's layer output against the original path's at
    /// every length, and both against the CPU `kda_layer` at 20 (the oracle). Each
    /// window starts from the state the previous one left (a running prefill).
    /// Ignored: `TANG_GLM5_BENCH_GGUF=… cargo test … -- --ignored glm5_kda_layer_bench_real`.
    #[test]
    #[ignore]
    fn glm5_kda_layer_bench_real() {
        let _env = crate::flash::glm5::env_write();
        let Ok(path) = std::env::var("TANG_GLM5_BENCH_GGUF") else {
            eprintln!("SKIP: TANG_GLM5_BENCH_GGUF unset");
            return;
        };
        let g = tang_moe::gpu::Gpu::for_test().expect("CUDA GPU");
        g.bind().unwrap();
        let r = super::super::forward::Glm5Ref::open(std::path::Path::new(&path)).unwrap();
        let l = 0usize;
        let (w, cfg) = r.kda_weights(l).unwrap();
        let hidden = r.hp.n_embd;
        let max_t = 2048usize;
        let mk = |pf2: bool| {
            // DevLayer reads the flag at load
            std::env::set_var("TANG_GLM5_KDA_PF2", if pf2 { "1" } else { "0" });
            let d = DevLayer::new_gguf(
                &g,
                &r.g,
                l,
                &cfg,
                hidden,
                &ConvState::zeros(&cfg, 1),
                &KdaState::zeros(&cfg, 1),
                max_t,
                1,
            )
            .unwrap();
            std::env::remove_var("TANG_GLM5_KDA_PF2");
            d
        };
        let (mut old, mut new) = (mk(false), mk(true));
        let (mut c_old, mut c_new) = (ConvState::zeros(&cfg, 1), ConvState::zeros(&cfg, 1));
        let (mut c_cpu, mut s_cpu) = (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1));
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for t in [20usize, 512, 2048] {
            // a normed-hidden-like input: uniform with unit RMS
            let x: Vec<f32> = (0..t * hidden)
                .map(|_| {
                    seed ^= seed >> 12;
                    seed ^= seed << 25;
                    seed ^= seed >> 27;
                    ((seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
                        - 0.5)
                        * 3.46
                })
                .collect();
            let t0 = std::time::Instant::now();
            let y_old = old.window(&x, t, &mut c_old).unwrap();
            let ms_old = t0.elapsed().as_secs_f64() * 1e3;
            // PF2: best of 3 from the same state (read back, re-seed): the first window
            // pays the module's first-launch costs
            let (mut c0, mut s0) = (c_new.clone(), KdaState::zeros(&cfg, 1));
            new.read_back(&mut c0, &mut s0).unwrap();
            let mut best = f64::MAX;
            let mut y_new = Vec::new();
            for _ in 0..3 {
                new.seed(&c0, &s0).unwrap();
                let t1 = std::time::Instant::now();
                y_new = new.window(&x, t, &mut c_new).unwrap();
                best = best.min(t1.elapsed().as_secs_f64() * 1e3);
            }
            let e = rel(&y_new, &y_old);
            // the states the two paths carry forward
            let (mut co, mut so) = (c_old.clone(), KdaState::zeros(&cfg, 1));
            old.read_back(&mut co, &mut so).unwrap();
            let (mut cn, mut sn) = (c_new.clone(), KdaState::zeros(&cfg, 1));
            new.read_back(&mut cn, &mut sn).unwrap();
            let es = rel(&sn.0, &so.0);
            let ec = rel(&cn.0, &co.0);
            println!(
                "kda layer {l} T={t}: window old {ms_old:.1} ms, pf2 {best:.2} ms ({:.0}x); x34 layers: old {:.0} ms, pf2 {:.0} ms; pf2 vs old: out rel {e:.2e} state {es:.2e} conv {ec:.2e}",
                ms_old / best,
                ms_old * 34.0,
                best * 34.0
            );
            assert!(
                e < 1e-3 && es < 1e-3 && ec < 1e-6,
                "pf2 diverges from the original path at T={t}"
            );
            if t == 20 {
                let y_cpu =
                    super::super::kda::kda_layer(&x, &w, &cfg, &mut c_cpu, &mut s_cpu, 1, t)
                        .unwrap();
                let (ecpu, escpu) = (rel(&y_new, &y_cpu), rel(&sn.0, &s_cpu.0));
                println!("kda layer {l} T={t}: pf2 vs CPU kda_layer: out rel {ecpu:.2e} state {escpu:.2e}");
                assert!(ecpu < 1e-3 && escpu < 1e-3, "pf2 vs CPU at T={t}");
            }
        }
    }
}

/// k5_gemv8 vs k5_gemv on one real-shaped layer of the tiny GLM-5 fixture, in-process:
/// the same x through the same DevLayer with the projection kernel flipped, the device
/// taps compared in full. Isolates the kernel from the engine (the gemv8 bring-up's first
/// three failures — no smem, misaligned u64, missing __syncwarp + overlapping staging,
/// and the k5_q8 MAXT argument — would each have shown here as a q/k/v divergence).
#[cfg(all(test, feature = "cuda"))]
mod gemv8_tests {
    use super::*;
    use std::path::Path;

    /// section `sec` (0=q,1=k,2=v) of a stacked [t][3·di] tap (the old kda_debug helper,
    /// which the tang-verify port replaced with the Spec's stacked sections)
    fn section(v: &[f32], t: usize, di: usize, sec: usize) -> Vec<f32> {
        if v.len() < t * 3 * di {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(t * di);
        for ti in 0..t {
            let b = ti * 3 * di + sec * di;
            out.extend_from_slice(&v[b..b + di]);
        }
        out
    }

    /// relative L2 (the old kda_debug::stat's headline number)
    fn stat(gpu: &[f32], cpu: &[f32], _tol: f64) -> (f64, f64, usize) {
        if gpu.is_empty() || cpu.is_empty() {
            return (0.0, 0.0, 1); // missing tap: counts as nonfinite so the gate fails
        }
        let (mut d2, mut c2, mut nf) = (0f64, 0f64, 0usize);
        for (&g, &c) in gpu.iter().zip(cpu) {
            if !g.is_finite() {
                nf += 1;
                continue;
            }
            d2 += ((g as f64) - (c as f64)).powi(2);
            c2 += (c as f64).powi(2);
        }
        let n = gpu.len().min(cpu.len()) as f64;
        let rms = (c2 / n).sqrt();
        let _ = (rms, _tol);
        // (rel L2, abs L2, NaN/Inf count): the gate is NaN-free + rel under the band, as
        // the harness's Stat::bad — an element-wise miss count is always > 0 under int8
        // activation noise (4 of 163840 on the fixture) and must not fail the test
        (d2.sqrt() / c2.max(1e-30).sqrt(), d2.sqrt(), nf)
    }

    // the Q8_0 fixture: gemv8 only takes Q8_0 rows (the f32 fixture routes to k5_gemv)
    const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-q8_0.gguf";

    #[test]
    fn gemv8_matches_gemv_on_tiny_layer() {
        let _env = crate::flash::glm5::env_read();
        let required = std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1");
        let Some(g) = Gpu::for_test() else {
            assert!(!required, "no CUDA device but TANG_REQUIRE_CUDA=1");
            return;
        };
        if !Path::new(TINY).exists() {
            assert!(!required, "tiny fixture absent but TANG_REQUIRE_CUDA=1");
            return;
        }
        let r = super::super::forward::Glm5Ref::open(Path::new(TINY)).unwrap();
        let hp = &r.hp;
        let l = (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let conv = ConvState::zeros(&cfg, 1);
        let state = KdaState::zeros(&cfg, 1);
        g.bind().unwrap();
        let mut layer =
            DevLayer::new_gguf(&g, &r.g, l, &cfg, hp.n_embd, &conv, &state, 64, 1).unwrap();
        let t = 20usize;
        let mut seed = 7u64;
        let x: Vec<f32> = (0..t * hp.n_embd)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        let run = |layer: &mut DevLayer, on: bool| {
            layer.gemv8 = on;
            layer.probe = Some(tang_verify::Probe::new(true));
            let mut c = ConvState::zeros(&cfg, 1);
            layer.seed(&c, &KdaState::zeros(&cfg, 1)).unwrap();
            let y = layer.window(&x, t, &mut c).unwrap();
            (y, layer.probe.take().unwrap().taps)
        };
        let tap = |t: &tang_verify::Taps, n: &str| t.get(n).unwrap_or(&[]).to_vec();
        assert!(
            [&layer.wq, &layer.wk, &layer.wv]
                .iter()
                .all(|w| w.w.ty == 8),
            "fixture q/k/v must be Q8_0 for this test to exercise k5_gemv8"
        );
        let (y0, t0) = run(&mut layer, false);
        let (y1, t1) = run(&mut layer, true);
        let di = cfg.heads * cfg.head_dim;
        for (sec, name) in [(0, "q"), (1, "k"), (2, "v")] {
            let a = section(&tap(&t0, super::super::kda_debug::tap::QKV), t, di, sec);
            let b = section(&tap(&t1, super::super::kda_debug::tap::QKV), t, di, sec);
            let s = stat(&b, &a, 1e-2);
            eprintln!(
                "qkv.{name}: rel {:.3e} l2 {:.3e} nonfinite {}",
                s.0, s.1, s.2
            );
            // int8 activations vs f32: the fe_gemv8 band, not bitwise
            assert!(
                s.2 == 0 && s.0 < 1e-2,
                "gemv8 qkv.{name} diverges: rel {:.3e}",
                s.0
            );
        }
        // f_a (gate) and g_a (g2) now on gemv8 too; o_proj covers the layer output
        for (name, a, b) in [
            (
                "g",
                tap(&t0, super::super::kda_debug::tap::G),
                tap(&t1, super::super::kda_debug::tap::G),
            ),
            (
                "g2",
                tap(&t0, super::super::kda_debug::tap::G2),
                tap(&t1, super::super::kda_debug::tap::G2),
            ),
        ] {
            let s = stat(&b, &a, 1e-2);
            eprintln!("{name}: rel {:.3e} l2 {:.3e} nonfinite {}", s.0, s.1, s.2);
            assert!(
                s.2 == 0 && s.0 < 1e-2,
                "gemv8 {name} diverges: rel {:.3e}",
                s.0
            );
        }
        let s = stat(&y1, &y0, 1e-2);
        eprintln!("layer out: rel {:.3e} max {:.3e}", s.0, s.1);
        // a hollow A/B (both runs on the naive kernel) gives EXACTLY 0 — int8 activations
        // always leave some noise, so 0 means gemv8 never ran
        let sq = stat(
            &section(&tap(&t1, super::super::kda_debug::tap::QKV), t, di, 0),
            &section(&tap(&t0, super::super::kda_debug::tap::QKV), t, di, 0),
            1e-2,
        );
        assert!(
            sq.0 > 0.0,
            "A/B is hollow: gemv8 produced bit-identical q (did it run?)"
        );
        // relative, like the taps above. s.1 is an ABSOLUTE l2: it grew 1.1e-2 -> 1.5e-1 when
        // k5_beta began computing every (t, h) (output scale, not error; rel stayed 9.2e-3)
        assert!(
            s.2 == 0 && s.0 < 1e-2,
            "gemv8 layer output diverges: rel {:.3e}",
            s.0
        );
    }
}
