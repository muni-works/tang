//! The fused KDA decode step + its CUDA graph (docs/models/glm5-speed-plan.md step 2).
//!
//! A child module of [`super`] (`kda_dev`) so it reads [`DevLayer`]'s fields directly and
//! keeps its diff to three lines there. At T = 1 the naive [`DevLayer::window`] issues 13
//! launches on two streams with three host round-trips between them (qkv/g/beta read back
//! and re-staged for W4's `DevKda::decode`, the core output `o` read back and re-staged
//! for the norm, the 4 MiB dummy state zeroed per call, three stream syncs). Here:
//!
//! 1. `k5d_proj`  — q | k | v, f_a, beta (+σ), g_a: every GEMV over `x`, one launch
//! 2. `k5d_gate2` — f_b(g1) + dt + gate form, and g_b(g2a): one launch
//! 3. `k5d_step`  — conv + delta rule + T=1 conv commit + gated norm: one launch
//! 4. `k5_gemv`   — o_proj (the existing kernel)
//!
//! all on the layer's stream, device pointers only — the host writes `x` before and reads
//! `out` after. That sequence has fixed pointers and shapes, so with
//! `TANG_GLM5_KDA_GRAPH=1` it is captured once (after one eager run) and replayed as a
//! single `cuGraphLaunch` per layer per token. With `TANG_GLM5_GEMV8=1` the q/k/v rows
//! go through `k5_q8` + `k5_gemv8` instead (kda_dev's Q8_0-only rule; the graph is keyed
//! on the choice).
//!
//! Env (default OFF until verified on the real model):
//! - `TANG_GLM5_KDA_FUSED=1` — the fused launches, eager
//! - `TANG_GLM5_KDA_GRAPH=1` — the fused launches, graph-captured (implies FUSED)
//!
//! Numerics: each fused kernel evaluates the naive kernel's expressions in the same
//! order, so the outputs equal the naive path's to the bit — with one deliberate
//! exception: `k5d_step` adds the `__syncthreads()` the naive `kda_step` lacks between the
//! q/k norm reductions and the in-place rescale (a cross-warp read-after-write race in
//! W4's kernel; benign whenever the 4 warps happen to finish the reduction together).
//!
//! Host conv copy: the naive decode shifts the engine's host `ConvState` as a side effect.
//! Nothing reads it while the device is fresh (`kda_mixer` reads back before any CPU
//! window), and the prefill-chunk path already leaves it stale — so the fused path
//! skips the shift (it would cost a qkv read-back) EXCEPT in a debug window, where the
//! shadow check compares it (`conv_host`).
#![allow(clippy::too_many_arguments)]
use super::super::kda::ConvState;
use super::DevLayer;
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use tang_moe::args;
use tang_moe::gpu::{launch, Event, Gpu, Graph, Module, Stream};

/// Which decode path a call takes (env-selected by [`Mode::from_env`]; tests pass it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Naive,
    Fused,
    Graph,
}

impl Mode {
    pub fn from_env() -> Self {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
        if on("TANG_GLM5_KDA_GRAPH") {
            Mode::Graph
        } else if on("TANG_GLM5_KDA_FUSED") {
            Mode::Fused
        } else {
            Mode::Naive
        }
    }

    fn name(self) -> &'static str {
        match self {
            Mode::Naive => "naive",
            Mode::Fused => "fused",
            Mode::Graph => "graph",
        }
    }
}

/// Once per process: the decode mode the KDA layers were built with (gate logs grep
/// `glm5 kda: decode mode`, so a run's path is auditable from its log).
pub fn log_load(mode: Mode) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("glm5 kda: decode mode {} (load)", mode.name()));
}

/// Once per process per event: the fused path's first eager step, first graph capture,
/// first graph replay — proof in the log that the path actually ran, not just loaded.
fn log_once(flag: &std::sync::atomic::AtomicBool, msg: impl FnOnce() -> String) {
    if !flag.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("{}", msg());
    }
}
static LOG_EAGER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOG_CAPTURE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOG_REPLAY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The fused kernels of one layer's device + the captured graph.
pub struct Fast {
    _m: Module,
    proj: sys::CUfunction,
    gate2: sys::CUfunction,
    step: sys::CUfunction,
    /// `k5d_tail`: beta + gate2 + step + o_proj's q8 in one launch (the G5F path).
    tail: sys::CUfunction,
    /// `k5_qgemvt9` / `k5_qgemvt17` (kda_k5.cuh): the t-row window GEMV (rows_enqueue).
    qgt9: sys::CUfunction,
    /// `k5d_tailt`: k5d_tail over a window's t rows in one launch (rows_fast).
    tailt: sys::CUfunction,
    /// `k5d_pret` / `k5d_stept`: the window tail split (rows_fast, TANG_GLM5_KDA_SPLIT).
    pret: sys::CUfunction,
    stept: sys::CUfunction,
    qgt17: sys::CUfunction,
    /// The captured decode step and the `TANG_GLM5_GEMV8` value it was captured with.
    graph: Option<(Graph, bool)>,
}

impl Fast {
    pub fn new(g: &Gpu, cfg: &super::KdaConfig) -> Result<Self> {
        // TANG_GLM5_KDA_NORM=exact: k5d_tail's serial f64 norm chain (default: f64 tree)
        let norm = if std::env::var("TANG_GLM5_KDA_NORM").is_ok_and(|v| v == "exact") {
            "#define K5D_NORM_EXACT 1"
        } else {
            ""
        };
        let src = format!(
            "{norm}\n{}\n{}\n{}",
            super::super::kda_cuda::prelude(cfg),
            super::super::moe::K5_SRC,
            include_str!("kda_decode.cuh")
        );
        let m = g
            .module(&src)
            .map_err(|e| anyhow::anyhow!("kda_decode nvrtc: {e}"))?;
        let f = |n: &str| {
            m.func(n)
                .map_err(|e| anyhow::anyhow!("kda_decode {n}: {e}"))
        };
        let tail = f("k5d_tail")?;
        // k5d_tail stages x (HID * 4 + 16 B, HID <= K5D_HIDMAX = 4096) in dynamic shared
        // memory on top of ~43 KB static: past the 48 KB default
        let r = unsafe {
            sys::cuFuncSetAttribute(
                tail,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                4096 * 4 + 16,
            )
        };
        anyhow::ensure!(
            r == sys::CUresult::CUDA_SUCCESS,
            "k5d_tail smem attribute: {r:?}"
        );
        let tailt = f("k5d_tailt")?;
        let r = unsafe {
            sys::cuFuncSetAttribute(
                tailt,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                4096 * 4 + 16,
            )
        };
        anyhow::ensure!(
            r == sys::CUresult::CUDA_SUCCESS,
            "k5d_tailt smem attribute: {r:?}"
        );
        let pret = f("k5d_pret")?;
        let r = unsafe {
            sys::cuFuncSetAttribute(
                pret,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                4096 * 4 + 16,
            )
        };
        anyhow::ensure!(
            r == sys::CUresult::CUDA_SUCCESS,
            "k5d_pret smem attribute: {r:?}"
        );
        Ok(Fast {
            proj: f("k5d_proj")?,
            gate2: f("k5d_gate2")?,
            step: f("k5d_step")?,
            tail,
            qgt9: f("k5_qgemvt9")?,
            tailt,
            pret,
            stept: f("k5d_stept")?,
            qgt17: f("k5_qgemvt17")?,
            graph: None,
            _m: m,
        })
    }
}

fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(128) as u32, 1, 1)
}

impl DevLayer {
    /// Enqueue the fused decode step on `s` (device pointers only: valid inside a capture).
    /// `ev` (eager only, the bench's per-stage breakdown): 5 events recorded before the
    /// projections and after each of proj / gate2 / step / o_proj.
    fn enqueue_decode(&self, s: &Stream, gemv8: bool, ev: Option<&[Event]>) -> Result<()> {
        let mark = |i: usize| -> Result<()> {
            if let Some(e) = ev {
                e[i].record(s).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            Ok(())
        };
        mark(0)?;
        self.enqueue_decode_at(s, 0, gemv8, ev)?;
        Ok(())
    }

    /// [`Self::enqueue_decode`] for sequence `slot`: only the step kernel's state pointers
    /// (h0 / st) move — everything else is per-window scratch.
    fn enqueue_decode_at(
        &self,
        s: &Stream,
        slot: usize,
        gemv8: bool,
        ev: Option<&[Event]>,
    ) -> Result<()> {
        let mark = |i: usize| -> Result<()> {
            if let Some(e) = ev {
                e[i].record(s).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            Ok(())
        };
        let fast = self
            .fast
            .as_ref()
            .context("kda fused decode: kernels not built")?;
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        let lb = self.cfg.gate_lower_bound.unwrap_or(f32::NAN);
        let f = std::mem::size_of::<f32>() as u64;
        let qkv = self.qkvb.ptr;
        let rows = 3 * di + 2 * hd + nh;
        // the per-matrix gemv8 rule (kda_dev window()): f_a / g_a go through k5_gemv8 when
        // Q8_0 — their f32-dequant g5_dot rounds differently, so k5d_proj must SKIP exactly
        // the groups gemv8 wrote (the forget gate amplifies any mismatch into the state)
        let g8f = gemv8 && self.fa.w.ty == 8;
        let g8ga = gemv8 && self.ga.w.ty == 8;
        let skip = (g8f as u32) | ((g8ga as u32) << 1);
        // this slot's conv history / recurrent state (the `[slot][...]` state layout)
        let ch_dim = 3 * di;
        let m = self.cfg.d_conv - 1;
        let h0 = self.core.h0.ptr + (slot * ch_dim * m) as u64 * f;
        let st = self.core.st.ptr + (slot * nh * hd * hd) as u64 * f;
        unsafe {
            if self.g5f > 0 {
                // TANG_GLM5_KDA_G5F: q|k|v, f_a, g_a on the fast kernels (kda_dev g5f_x, the
                // same launches as the naive window), beta on k5_beta as the naive path does —
                // k5d_proj would compute only beta, one THREAD per 4096-wide row (~1.3 ms)
                self.g5f_x(s, 1)?;
                if self.tail_ok() {
                    mark(1)?;
                    self.tail_launch(s, fast, h0, st, lb, 0)?;
                    mark(3)?;
                    match self.sego.as_ref() {
                        // k5d_tail wrote yb's k5_q8 codes: o_proj straight on k5_qgemv
                        Some(tab) => {
                            let codes = self.xq8.ptr;
                            self.qgemv_launch(
                                s,
                                tab,
                                1,
                                self.wo.w.row_bytes,
                                hidden,
                                self.wo.w.estride,
                                codes,
                                codes + (self.max_t * di) as u64,
                                di,
                            )?;
                        }
                        None => {
                            self.g5f_o(s, 1)?;
                        }
                    }
                    mark(4)?;
                    return Ok(());
                }
                launch(
                    self.gv.beta,
                    (nh.div_ceil(super::K5_WARPS) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    s,
                    args![
                        self.bproj.w,
                        self.xb.ptr,
                        self.bb.ptr,
                        1u32,
                        hidden as u32,
                        nh as u32
                    ],
                )?;
            } else {
                let r0 = if gemv8 {
                    // the dequant-reuse GEMV for q/k/v (kda_dev's TANG_GLM5_GEMV8 path, T = 1)
                    launch(
                        self.gv.q8,
                        ((hidden / 32) as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        s,
                        args![self.xb.ptr, self.xq8.ptr, hidden as u32, self.max_t as u32],
                    )?;
                    let codes = self.xq8.ptr;
                    let scales = codes + (self.max_t * hidden) as u64;
                    for (w, off) in [
                        (&self.wq, 0u64),
                        (&self.wk, di as u64 * f),
                        (&self.wv, 2 * di as u64 * f),
                    ] {
                        // the same dynamic smem as kda_dev's gemv8 launch (keep the two in step)
                        let smem = (super::K5_WARPS
                            * ((w.w.row_bytes as usize).div_ceil(16) + 1)
                            * 16) as u32;
                        launch(
                            self.gv.gemv8,
                            (di.div_ceil(super::K5_WARPS) as u32, 1, 1),
                            (128, 1, 1),
                            smem,
                            s,
                            args![
                                w.w,
                                0u32,
                                codes,
                                scales,
                                qkv + off,
                                (3 * di) as u32,
                                1u32,
                                di as u32,
                                hidden as u32
                            ],
                        )?;
                    }
                    // f_a / g_a read the SAME x codes — strictly AFTER k5_q8 above: before it,
                    // xq8 still holds the previous step's o_proj codes (yb, d_inner wide), a
                    // stale read that diverged every output on the first real-weights rerun
                    if g8f {
                        self.gemv8_launch_on(s, &self.fa, codes, hidden, self.g1b.ptr, hd, hd, 1)?;
                    }
                    if g8ga {
                        self.gemv8_launch_on(s, &self.ga, codes, hidden, self.g2a.ptr, hd, hd, 1)?;
                    }
                    3 * di
                } else {
                    0
                };
                mark(2)?;
                launch(
                    fast.proj,
                    grid(rows - r0),
                    (128, 1, 1),
                    0,
                    s,
                    args![
                        self.wq.w,
                        self.wk.w,
                        self.wv.w,
                        self.fa.w,
                        self.bproj.w,
                        self.ga.w,
                        self.xb.ptr,
                        qkv,
                        self.g1b.ptr,
                        self.bb.ptr,
                        self.g2a.ptr,
                        di as u32,
                        hd as u32,
                        nh as u32,
                        hidden as u32,
                        r0 as u32,
                        skip
                    ],
                )?;
            }
            mark(1)?;
            launch(
                fast.gate2,
                grid(2 * di),
                (128, 1, 1),
                0,
                s,
                args![
                    self.fb.w,
                    self.gb.w,
                    self.g1b.ptr,
                    self.g2a.ptr,
                    self.dt.ptr,
                    self.rate.ptr,
                    self.gb_out.ptr,
                    self.g2b.ptr,
                    nh as u32,
                    hd as u32,
                    lb
                ],
            )?;
            launch(
                fast.step,
                (nh as u32, 1, 1),
                (hd as u32, 1, 1),
                0,
                s,
                args![
                    qkv,
                    h0,
                    self.core.wconv.ptr,
                    self.gb_out.ptr,
                    self.bb.ptr,
                    st,
                    self.ob.ptr,
                    self.on.ptr,
                    self.g2b.ptr,
                    self.yb.ptr,
                    self.cfg.eps
                ],
            )?;
            mark(3)?;
            // o_proj: gemv8 quantises yb (d_inner wide) first — the SAME xq8 the q/k/v
            // pre-pass used (nothing reads x's codes after the q/k/v GEMVs, so overwriting
            // them here is safe; kda_dev's finish() does the same)
            let g8o = gemv8 && self.wo.w.ty == 8;
            if self.g5f_o(s, 1)? {
            } else if g8o {
                launch(
                    self.gv.q8,
                    ((di / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    s,
                    args![self.yb.ptr, self.xq8.ptr, di as u32, self.max_t as u32],
                )?;
                let codes = self.xq8.ptr;
                let scales = codes + (self.max_t * di) as u64;
                let smem = (super::K5_WARPS
                    * ((self.wo.w.row_bytes as usize).div_ceil(16) + 1)
                    * 16) as u32;
                launch(
                    self.gv.gemv8,
                    (hidden.div_ceil(super::K5_WARPS) as u32, 1, 1),
                    (128, 1, 1),
                    smem,
                    s,
                    args![
                        self.wo.w,
                        0u32,
                        codes,
                        scales,
                        self.outb.ptr,
                        hidden as u32,
                        1u32,
                        hidden as u32,
                        di as u32
                    ],
                )?;
            } else {
                launch(
                    self.gv.gemv,
                    grid(hidden),
                    (128, 1, 1),
                    0,
                    s,
                    args![
                        self.wo.w,
                        0u32,
                        self.yb.ptr,
                        self.outb.ptr,
                        hidden as u32,
                        1u32,
                        hidden as u32,
                        di as u32
                    ],
                )?;
            }
            mark(4)?;
        }
        Ok(())
    }

    /// Whether [`Self::decode_dev`] can run this layer: the fused kernels are built, the
    /// mode is Fused or Graph, and no shadow probe is attached.
    pub fn resident_ok(&self) -> bool {
        self.fast.is_some() && Mode::from_env() != Mode::Naive && self.probe.is_none()
    }

    /// The decode input buffer (`[hidden]` f32): the resident step's mHC pre writes here.
    pub fn x_ptr(&self) -> u64 {
        self.xb.ptr
    }

    /// [`Self::decode_dev`]'s kernels enqueued directly on `s` (no graph launch): the
    /// whole-step graph capture records them as its own nodes.
    pub fn decode_enqueue(&self, s: &Stream) -> Result<()> {
        ensure!(
            self.resident_ok(),
            "kda decode_enqueue: fused kernels off, naive mode, or a probe"
        );
        let g8 = |w: &super::super::moe::gpu::DevW| self.gemv8 && w.w.ty == 8;
        self.enqueue_decode(s, g8(&self.wq) && g8(&self.wk) && g8(&self.wv), None)
    }

    /// The decode output buffer (`[hidden]` f32), valid after [`Self::decode_dev`].
    pub fn y_ptr(&self) -> u64 {
        self.outb.ptr
    }

    /// The device-resident decode token (slot 0): `xb` already holds x (written on `s`),
    /// the fused step (Graph mode: the captured graph) is enqueued on `s`, `outb` holds y
    /// once `s` reaches it. No host copy, no sync. Graph mode captures on the layer's own
    /// stream after the first eager enqueue (capture only records).
    pub fn decode_dev(&mut self, s: &Stream) -> Result<()> {
        ensure!(
            self.resident_ok(),
            "kda decode_dev: fused kernels off, naive mode, or a probe"
        );
        let g8 = |w: &super::super::moe::gpu::DevW| self.gemv8 && w.w.ty == 8;
        let gemv8 = g8(&self.wq) && g8(&self.wk) && g8(&self.wv);
        let mode = Mode::from_env();
        if mode == Mode::Graph {
            if let Some((g, _)) = self
                .fast
                .as_ref()
                .and_then(|f| f.graph.as_ref())
                .filter(|g| g.1 == gemv8)
            {
                g.launch(s)
                    .map_err(|e| anyhow::anyhow!("kda graph launch: {e}"))?;
                return Ok(());
            }
            self.enqueue_decode(s, gemv8, None)?;
            if let Some(f) = self.fast.as_mut() {
                f.graph = None;
            }
            let me = &*self;
            let gr = Graph::capture(&me.s, |cs| {
                me.enqueue_decode(cs, gemv8, None)
                    .map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))
            })
            .map_err(|e| anyhow::anyhow!("kda decode capture: {e}"))?;
            self.fast.as_mut().unwrap().graph = Some((gr, gemv8));
            log_once(&LOG_CAPTURE, || {
                format!("glm5 kda: decode mode graph captured (resident, gemv8={gemv8})")
            });
            return Ok(());
        }
        self.enqueue_decode(s, gemv8, None)
    }

    /// One decode token through the fused step (`mode` Fused or Graph). Same contract as
    /// [`DevLayer::window`] at `t = 1`, including the shadow-check probe (the poisoned
    /// buffers get their taps read out of the device, `out`/`conv_host` put on the host).
    pub(super) fn decode_fast(
        &mut self,
        x: &[f32],
        conv: &mut ConvState,
        mode: Mode,
    ) -> Result<Vec<f32>> {
        self.decode_fast_at(0, x, conv, mode)
    }

    /// [`Self::decode_fast`] for sequence `slot`. The fused launches take the state
    /// pointers directly (`enqueue_decode`'s `self.core.h0.ptr` etc. are slot-0 bases), so
    /// a slot adds its offsets to the step kernel's arguments — and only the EAGER path can
    /// do that: a captured graph freezes the pointers, so Graph mode runs slot 0 only.
    pub(super) fn decode_fast_at(
        &mut self,
        slot: usize,
        x: &[f32],
        conv: &mut ConvState,
        mode: Mode,
    ) -> Result<Vec<f32>> {
        ensure!(x.len() == self.hidden, "k5 decode input");
        ensure!(mode != Mode::Naive, "decode_fast: naive mode");
        let (nh, di, hidden) = (self.nh, self.d_inner, self.hidden);
        // kda_dev's per-matrix rule (the x pre-pass): gemv8 only when q/k/v are all Q8_0
        // (the kernel reads 34-byte Q8_0 blocks; the f32 tiny fixture must take k5_gemv)
        let g8 = |w: &super::super::moe::gpu::DevW| self.gemv8 && w.w.ty == 8;
        let gemv8 = g8(&self.wq) && g8(&self.wk) && g8(&self.wv);
        // the core's prefill path uses its own stream: make sure it has drained (it syncs
        // at the end of every call, so this is free; it keeps the ordering explicit)
        self.core.s.sync()?;
        // synchronous HtoD on the legacy stream; self.s is BLOCKING, so the launches (or
        // the graph) below are ordered after it
        self.xb.write(0, x)?;
        let mut tp = self.probe.take();
        if let Some(p) = &tp {
            // NaN-fill every output scratch this window writes (whole buffers): a region
            // no kernel touches then reads NaN in the shadow check, not a plausible 0.0.
            // (The PF2 workspace `core.ws` is prefill-only, not touched at t = 1.)
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
            ])?;
        }
        if slot == 0 {
            match mode {
                Mode::Graph
                    if self
                        .fast
                        .as_ref()
                        .and_then(|f| f.graph.as_ref())
                        .is_some_and(|g| g.1 == gemv8) =>
                {
                    let f = self.fast.as_ref().unwrap();
                    f.graph
                        .as_ref()
                        .unwrap()
                        .0
                        .launch(&self.s)
                        .map_err(|e| anyhow::anyhow!("kda graph launch: {e}"))?;
                    log_once(&LOG_REPLAY, || {
                        format!("glm5 kda: decode mode graph engaged (first replay, gemv8={gemv8})")
                    });
                }
                Mode::Graph => {
                    // first token (or GEMV8 toggled): run eagerly, then capture for the next
                    // tokens (capture records, it does not execute)
                    self.enqueue_decode(&self.s, gemv8, None)?;
                    self.s.sync()?;
                    if let Some(f) = self.fast.as_mut() {
                        f.graph = None;
                    }
                    let me = &*self;
                    let gr = Graph::capture(&me.s, |s| {
                        me.enqueue_decode(s, gemv8, None)
                            .map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))
                    })
                    .map_err(|e| anyhow::anyhow!("kda decode capture: {e}"))?;
                    self.fast.as_mut().unwrap().graph = Some((gr, gemv8));
                    log_once(&LOG_CAPTURE, || {
                        format!("glm5 kda: decode mode graph captured (first decode ran eager, gemv8={gemv8})")
                    });
                }
                _ => {
                    self.enqueue_decode(&self.s, gemv8, None)?;
                    log_once(&LOG_EAGER, || {
                        format!(
                            "glm5 kda: decode mode fused engaged (first eager step, gemv8={gemv8})"
                        )
                    });
                }
            }
        } else {
            ensure!(mode == Mode::Fused, "kda fused decode: slot {slot} needs eager mode (the captured graph has slot 0's pointers frozen)");
            self.enqueue_decode_at(&self.s, slot, gemv8, None)?;
        }
        self.s.sync()?;
        let out = self.outb.read::<f32>(0, hidden)?;
        let Some(mut p) = tp else {
            // no probe: keep the pre-probe behavior — the host conv copy is NOT shifted
            // (that would cost a qkv read-back per token per layer; nothing reads it while
            // the device is fresh — `read_back` syncs it before any CPU use)
            return Ok(out);
        };
        use super::super::kda_debug::tap;
        // the host conv shift (DevKda::decode's side effect), from qkv PRE-conv (the same
        // qkv the step kernel consumed)
        let qkv_h = self.qkvb.read::<f32>(0, 3 * di)?;
        let m = self.cfg.d_conv - 1;
        let ch = 3 * di;
        let mut next = conv.0.clone();
        for c in 0..ch {
            for s2 in 0..m {
                next[c * m + s2] = if s2 + 1 < m {
                    conv.0[c * m + s2 + 1]
                } else {
                    qkv_h[c]
                };
            }
        }
        *conv = ConvState(next);
        p.put(tap::QKV, qkv_h);
        p.put(tap::G, self.gb_out.read::<f32>(0, di)?);
        p.put(tap::BETA, self.bb.read::<f32>(0, nh)?);
        p.put(tap::O, self.ob.read::<f32>(0, di)?);
        p.put(tap::G2, self.g2b.read::<f32>(0, di)?);
        p.put(tap::Y_NORM, self.yb.read::<f32>(0, di)?);
        p.put(tap::OUT, out.clone());
        self.probe = Some(p);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::kda::{ConvState, KdaConfig, KdaState};
    use super::*;
    use std::path::Path;

    const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf";

    fn poison(d: &DevLayer) {
        let (di, hd, nh) = (d.d_inner, d.hd, d.nh);
        let nan = |n: usize| vec![f32::NAN; n];
        for (b, n) in [
            (&d.qkvb, 3 * di),
            (&d.g1b, hd),
            (&d.gb_out, di),
            (&d.bb, nh),
            (&d.g2a, hd),
            (&d.g2b, di),
            (&d.ob, di),
            (&d.yb, di),
            (&d.outb, d.hidden),
        ] {
            b.write(0, &nan(n)).unwrap();
        }
    }

    /// The fused decode (eager and graph-replayed) equals the naive decode on the tiny
    /// model's first KDA layer: 6 tokens of carried state, output and device state
    /// compared every step. Bit-equality is expected (same expressions) up to the naive
    /// step kernel's sync race, so the bar is 1e-5 relative.
    #[test]
    fn kda_fused_decode_matches_naive() {
        let _env = crate::flash::glm5::env_read();
        let required = std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1");
        if !Path::new(TINY).exists() {
            assert!(
                !required,
                "TANG_REQUIRE_CUDA=1 but the tiny fixture is absent"
            );
            eprintln!("SKIP kda_fused_decode_matches_naive: no fixture");
            return;
        }
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let gguf = crate::gguf::Gguf::open(Path::new(TINY)).unwrap();
        let hp = super::super::super::Hparams::from_gguf(&gguf).unwrap();
        let l = (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let n = hp.n_embd;
        // `a` has no fused kernels, so its window() is the naive path whatever the env says
        let mk = |fast: bool| {
            let mut d = DevLayer::new_gguf(
                &g,
                &gguf,
                l,
                &cfg,
                n,
                &ConvState::zeros(&cfg, 1),
                &KdaState::zeros(&cfg, 1),
                8,
                1,
            )
            .unwrap();
            d.fast = fast.then(|| Fast::new(&g, &cfg).unwrap());
            d
        };
        let (mut a, mut b, mut c) = (mk(false), mk(true), mk(true));
        let mut seed = 7u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let (mut ca, mut cb, mut cc) = (
            ConvState::zeros(&cfg, 1),
            ConvState::zeros(&cfg, 1),
            ConvState::zeros(&cfg, 1),
        );
        let rel = |p: &[f32], q: &[f32]| {
            let num: f64 = p.iter().zip(q).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
            let den: f64 = q.iter().map(|y| (*y as f64).powi(2)).sum();
            (num / den.max(1e-30)).sqrt()
        };
        for step in 0..6 {
            let x: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let ya = a.window(&x, 1, &mut ca).unwrap();
            // NaN-poison every buffer the fused step writes before each call: a graph node
            // that didn't replay (or a stale pointer) then reads NaN, not last step's values
            for d in [&b, &c] {
                poison(d);
            }
            let yb = b.decode_fast(&x, &mut cb, Mode::Fused).unwrap();
            let yc = c.decode_fast(&x, &mut cc, Mode::Graph).unwrap();
            let (mut sa, mut sb, mut sc) = (
                KdaState::zeros(&cfg, 1),
                KdaState::zeros(&cfg, 1),
                KdaState::zeros(&cfg, 1),
            );
            let (mut ha, mut hb, mut hc) = (ca.clone(), cb.clone(), cc.clone());
            a.read_back(&mut ha, &mut sa).unwrap();
            b.read_back(&mut hb, &mut sb).unwrap();
            c.read_back(&mut hc, &mut sc).unwrap();
            let (r_fb, r_gr) = (rel(&yb, &ya), rel(&yc, &ya));
            let (s_fb, s_gr) = (rel(&sb.0, &sa.0), rel(&sc.0, &sa.0));
            let (h_fb, h_gr) = (rel(&hb.0, &ha.0), rel(&hc.0, &ha.0));
            let bits = ya == yb && ya == yc;
            eprintln!(
                "step {step}: out rel fused {r_fb:.2e} graph {r_gr:.2e}; state {s_fb:.2e} {s_gr:.2e}; conv {h_fb:.2e} {h_gr:.2e}; bit-equal {bits}"
            );
            assert!(
                ya.iter().all(|v| v.is_finite()) && ya.iter().any(|&v| v != 0.0),
                "naive output degenerate"
            );
            for r in [r_fb, r_gr, s_fb, s_gr, h_fb, h_gr] {
                assert!(
                    r <= 1e-5,
                    "fused decode diverges from naive: {r:.3e} at step {step}"
                );
            }
        }
        assert!(
            c.fast.as_ref().unwrap().graph.is_some(),
            "graph never captured"
        );
    }

    /// REAL WEIGHTS, one KDA layer (TANG_GLM5_REAL_GGUF = shard 1; TANG_GLM5_BENCH_LAYER,
    /// default the first KDA layer): naive vs fused vs graph decode on the same random
    /// token stream — per-step bit-equality of the outputs and the final device state,
    /// then wall-clock µs per decode step of each (host-timed, x in → y out, i.e. what
    /// the engine's `kda mixer` stage pays per layer). Skips without the env var.
    #[test]
    fn kda_decode_real_bench() {
        let _env = crate::flash::glm5::env_read();
        let Ok(real) = std::env::var("TANG_GLM5_REAL_GGUF") else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let g = Gpu::for_test().expect("GPU");
        g.bind().unwrap();
        let gguf = crate::gguf::Gguf::open(Path::new(&real)).unwrap();
        let hp = super::super::super::Hparams::from_gguf(&gguf).unwrap();
        let l = std::env::var("TANG_GLM5_BENCH_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap());
        assert!(hp.is_kda[l], "layer {l} is not KDA");
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let n = hp.n_embd;
        let mk = |fast: bool| {
            let mut d = DevLayer::new_gguf(
                &g,
                &gguf,
                l,
                &cfg,
                n,
                &ConvState::zeros(&cfg, 1),
                &KdaState::zeros(&cfg, 1),
                64,
                1,
            )
            .unwrap();
            d.fast = fast.then(|| Fast::new(&g, &cfg).unwrap());
            d
        };
        let mut seed = 11u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let steps = 64usize;
        let xs: Vec<Vec<f32>> = (0..steps)
            .map(|_| (0..n).map(|_| rnd()).collect())
            .collect();
        let mut outs: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut states = Vec::new();
        for mode in [Mode::Naive, Mode::Fused, Mode::Graph] {
            let mut d = mk(mode != Mode::Naive);
            let mut conv = ConvState::zeros(&cfg, 1);
            let mut ys = Vec::with_capacity(steps);
            // 8 warm-up steps (the graph captures on its first), then the timed rest
            let mut t0 = std::time::Instant::now();
            for (i, x) in xs.iter().enumerate() {
                if i == 8 {
                    t0 = std::time::Instant::now();
                }
                ys.push(match mode {
                    Mode::Naive => d.window(x, 1, &mut conv).unwrap(),
                    m => d.decode_fast(x, &mut conv, m).unwrap(),
                });
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / (steps - 8) as f64;
            eprintln!(
                "kda decode layer {l} {mode:?}: {us:.0} us/step (real weights, {} steps timed)",
                steps - 8
            );
            let mut st = KdaState::zeros(&cfg, 1);
            let mut hc = conv.clone();
            d.read_back(&mut hc, &mut st).unwrap();
            outs.push(ys);
            states.push((hc, st));
        }
        {
            // per-stage GPU time of the fused step (eager, events between the launches)
            let d = mk(true);
            let ev: Vec<Event> = (0..5).map(|_| Event::new(true).unwrap()).collect();
            let gemv8 = d.gemv8 && [&d.wq, &d.wk, &d.wv].iter().all(|w| w.w.ty == 8);
            let mut acc = [0f32; 4];
            for x in &xs {
                d.xb.write(0, x).unwrap();
                d.enqueue_decode(&d.s, gemv8, Some(&ev)).unwrap();
                d.s.sync().unwrap();
                for i in 0..4 {
                    acc[i] += ev[i + 1].since(&ev[i]).unwrap();
                }
            }
            let k = xs.len() as f32;
            eprintln!(
                "kda decode layer {l} fused stages (GPU us/step, gemv8 {gemv8}): proj {:.0}, gate2 {:.0}, step {:.0}, o_proj {:.0}",
                acc[0] * 1e3 / k, acc[1] * 1e3 / k, acc[2] * 1e3 / k, acc[3] * 1e3 / k
            );
        }
        for (mi, name) in [(1, "fused"), (2, "graph")] {
            let diff = outs[mi]
                .iter()
                .zip(&outs[0])
                .flat_map(|(a, b)| a.iter().zip(b))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let sdiff = states[mi]
                .1
                 .0
                .iter()
                .zip(&states[0].1 .0)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let cdiff = states[mi]
                .0
                 .0
                .iter()
                .zip(&states[0].0 .0)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "{name} vs naive: {diff} output values differ (bits), state {sdiff}, conv {cdiff}"
            );
            assert_eq!(
                (diff, sdiff, cdiff),
                (0, 0, 0),
                "{name} decode is not bit-equal to naive on real weights"
            );
        }
    }
    /// REAL WEIGHTS (TANG_GLM5_REAL_GGUF = shard 1), TANG_GLM5_KDA_G5F: (a) per-matrix
    /// kernel time (CUDA events, kernel only, T = 1, median of 50) and HBM GB/s of
    /// k5_gemv8, g5f_gemv (q8 codes) and k5_qgemv for q/k/v/o/f_a/g_a plus the merged
    /// q|k|v|f_a|g_a launch; (b) parity: 32 decode steps through the naive path with
    /// G5F=0 (gemv8) vs 1 vs 2 — layer output and final state within the int8 band
    /// (rel 1e-2); (c) fused and graph vs naive under G5F=1/2 bit-equal (per-matrix rule);
    /// (d) fused per-stage µs per G5F mode. Skips without the env var.
    #[test]
    fn kda_g5f_real_bench() {
        let _env = crate::flash::glm5::env_write();
        let Ok(real) = std::env::var("TANG_GLM5_REAL_GGUF") else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let g = Gpu::for_test().expect("GPU");
        g.bind().unwrap();
        let gguf = crate::gguf::Gguf::open(Path::new(&real)).unwrap();
        let hp = super::super::super::Hparams::from_gguf(&gguf).unwrap();
        let l = std::env::var("TANG_GLM5_BENCH_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap());
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let n = hp.n_embd;
        let mk = |g5f: u8, fast: bool| {
            std::env::set_var("TANG_GLM5_KDA_G5F", g5f.to_string());
            std::env::set_var("TANG_GLM5_GEMV8", "1");
            let mut d = DevLayer::new_gguf(
                &g,
                &gguf,
                l,
                &cfg,
                n,
                &ConvState::zeros(&cfg, 1),
                &KdaState::zeros(&cfg, 1),
                64,
                1,
            )
            .unwrap();
            std::env::remove_var("TANG_GLM5_KDA_G5F");
            d.fast = fast.then(|| Fast::new(&g, &cfg).unwrap());
            d
        };
        let mut seed = 11u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let xs: Vec<Vec<f32>> = (0..32).map(|_| (0..n).map(|_| rnd()).collect()).collect();

        // (a) per-matrix kernel time
        let d = mk(2, false);
        assert!(
            d.segx.is_some() && d.sego.is_some(),
            "k5_qgemv tables not built (non-Q8_0 or unaligned weights)"
        );
        d.xb.write(0, &xs[0]).unwrap();
        d.yb.write(
            0,
            &xs[0]
                .iter()
                .cycle()
                .take(d.d_inner)
                .copied()
                .collect::<Vec<f32>>(),
        )
        .unwrap();
        let (e0, e1) = (Event::new(true).unwrap(), Event::new(true).unwrap());
        let time = |f: &dyn Fn()| -> f32 {
            for _ in 0..5 {
                f();
            }
            let mut v: Vec<f32> = (0..50)
                .map(|_| {
                    e0.record(&d.s).unwrap();
                    f();
                    e1.record(&d.s).unwrap();
                    d.s.sync().unwrap();
                    e1.since(&e0).unwrap() * 1e3
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[25]
        };
        let (di, hd, hid) = (d.d_inner, d.hd, d.hidden);
        let mats: [(
            &str,
            &crate::flash::glm5::moe::gpu::DevW,
            usize,
            usize,
            u64,
            u64,
        ); 6] = [
            ("q", &d.wq, di, hid, d.xb.ptr, d.qkvb.ptr),
            ("k", &d.wk, di, hid, d.xb.ptr, d.qkvb.ptr),
            ("v", &d.wv, di, hid, d.xb.ptr, d.qkvb.ptr),
            ("f_a", &d.fa, hd, hid, d.xb.ptr, d.g1b.ptr),
            ("g_a", &d.ga, hd, hid, d.xb.ptr, d.g2a.ptr),
            ("o", &d.wo, hid, di, d.yb.ptr, d.outb.ptr),
        ];
        let codes = d.xq8.ptr;
        let mut tot = [0f32; 3];
        let mut to = 0f32;
        let mut tabs = Vec::new();
        for (name, w, rows, cols, x, y) in mats {
            unsafe {
                launch(
                    d.gv.q8,
                    ((cols / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &d.s,
                    args![x, codes, cols as u32, d.max_t as u32],
                )
                .unwrap();
            }
            let scales = codes + (d.max_t * cols) as u64;
            let t8 = time(&|| unsafe { d.gemv8_launch(w, codes, cols, y, rows, rows, 1).unwrap() });
            let a = crate::flash::glm5::moe::gpu::Act {
                xf: x,
                xq: codes,
                xq2: 0,
                xd: scales,
                xs: cols,
            };
            let tf = time(&|| {
                d.gv.fast
                    .gemv(&d.s, w.w, 0, a, y, rows, 1, rows, cols)
                    .unwrap()
            });
            let tab = tang_moe::gpu::DevBuf::from_slice(&[
                w.w.base,
                y,
                w.w.row_bytes as u64 | ((rows as u64) << 32),
            ])
            .unwrap();
            let tq = time(&|| unsafe {
                d.qgemv_launch(&d.s, &tab, 1, w.w.row_bytes, rows, 0, codes, scales, cols)
                    .unwrap()
            });
            tabs.push(tab);
            let mb = w.w.estride as f64 / 1e6;
            eprintln!(
                "g5f bench {name:>3} {rows}x{cols} ({mb:.1} MB): k5_gemv8 {t8:.1} us ({:.2} TB/s), g5f_gemv {tf:.1} us ({:.2} TB/s), k5_qgemv {tq:.1} us ({:.2} TB/s)",
                mb / t8 as f64, mb / tf as f64, mb / tq as f64
            );
            if name == "o" {
                to = tq;
            }
            for (i, v) in [t8, tf, tq].into_iter().enumerate() {
                tot[i] += v;
            }
        }
        unsafe {
            launch(
                d.gv.q8,
                ((hid / 32) as u32, 1, 1),
                (32, 1, 1),
                0,
                &d.s,
                args![d.xb.ptr, codes, hid as u32, d.max_t as u32],
            )
            .unwrap();
        }
        let mb5: f64 = [&d.wq, &d.wk, &d.wv, &d.fa, &d.ga]
            .iter()
            .map(|w| w.w.estride as f64)
            .sum::<f64>()
            / 1e6;
        let tm = time(&|| unsafe {
            d.qgemv_launch(
                &d.s,
                d.segx.as_ref().unwrap(),
                5,
                d.wq.w.row_bytes,
                3 * di + 2 * hd,
                0,
                codes,
                codes + (d.max_t * hid) as u64,
                hid,
            )
            .unwrap()
        });
        eprintln!(
            "g5f bench merged q|k|v|f_a|g_a ({mb5:.1} MB): k5_qgemv {tm:.1} us ({:.2} TB/s)",
            mb5 / tm as f64
        );
        eprintln!(
            "g5f bench per layer (6 matrices, separate launches): k5_gemv8 {:.1} us, g5f_gemv {:.1} us, k5_qgemv {:.1} us; merged x-launch + o: {:.1} us",
            tot[0], tot[1], tot[2], tm + to
        );
        drop(tabs);
        drop(d);

        // (b) naive-path parity across modes, (c) fused/graph bit-equality per mode
        let run = |g5f: u8, mode: Mode| {
            let mut d = mk(g5f, mode != Mode::Naive);
            let mut conv = ConvState::zeros(&cfg, 1);
            let mut ys = Vec::new();
            let t0 = std::time::Instant::now();
            for x in &xs {
                ys.push(match mode {
                    Mode::Naive => d.window(x, 1, &mut conv).unwrap(),
                    m => d.decode_fast(x, &mut conv, m).unwrap(),
                });
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / xs.len() as f64;
            let mut st = KdaState::zeros(&cfg, 1);
            let mut hc = conv.clone();
            d.read_back(&mut hc, &mut st).unwrap();
            (ys.concat(), st, us)
        };
        let base = run(0, Mode::Naive);
        for g5f in [1u8, 2] {
            let nv = run(g5f, Mode::Naive);
            let (eo, es) = (rel64(&nv.0, &base.0), rel64(&nv.1 .0, &base.1 .0));
            eprintln!("g5f parity G5F={g5f} naive vs gemv8 naive: out rel {eo:.3e}, state rel {es:.3e} ({:.0} vs {:.0} us/step host)", nv.2, base.2);
            assert!(
                eo < 1e-2 && es < 1e-2,
                "G5F={g5f} diverges from the gemv8 path"
            );
            // G5F=1 is EXPECTED bit-identical: g5f_gemv's Q8_0 lane dot on k5_q8's codes walks
            // the same e0 = lane*8 + 256i steps with the same fmaf and xor reduce as k5_gemv8
            // (measured: rel 0 on real layer 0). k5_qgemv sums 32 per dp4a chain: must differ.
            if g5f == 2 {
                assert!(eo > 0.0, "G5F=2 bit-identical to gemv8: did k5_qgemv run?");
            }
            for m in [Mode::Fused, Mode::Graph] {
                let fv = run(g5f, m);
                let diff =
                    fv.0.iter()
                        .zip(&nv.0)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                let sdiff =
                    fv.1 .0
                        .iter()
                        .zip(&nv.1 .0)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                eprintln!("g5f parity G5F={g5f} {m:?} vs naive: {diff} outputs, {sdiff} state values differ (bits); {:.0} us/step host", fv.2);
                assert_eq!(
                    (diff, sdiff),
                    (0, 0),
                    "G5F={g5f} {m:?} not bit-equal to naive"
                );
            }
        }
        // (d) fused per-stage GPU time per mode
        for g5f in [0u8, 1, 2] {
            let d = mk(g5f, true);
            let ev: Vec<Event> = (0..5).map(|_| Event::new(true).unwrap()).collect();
            let gemv8 = d.gemv8 && [&d.wq, &d.wk, &d.wv].iter().all(|w| w.w.ty == 8);
            let mut acc = [0f32; 4];
            for x in &xs {
                d.xb.write(0, x).unwrap();
                d.enqueue_decode(&d.s, gemv8, Some(&ev)).unwrap();
                d.s.sync().unwrap();
                for i in 0..4 {
                    acc[i] += ev[i + 1].since(&ev[i]).unwrap();
                }
            }
            let k = xs.len() as f32;
            eprintln!(
                "g5f fused stages G5F={g5f} (GPU us/step): proj {:.1}, gate2 {:.1}, step {:.1}, o_proj {:.1}, total {:.1}",
                acc[0] * 1e3 / k, acc[1] * 1e3 / k, acc[2] * 1e3 / k, acc[3] * 1e3 / k, acc.iter().sum::<f32>() * 1e3 / k
            );
        }
    }

    /// REAL WEIGHTS of any projection type (TANG_GLM5_REAL_GGUF = shard 1 of e.g. the
    /// requantized Q4_K / Q5_K KDA GGUF, tools/glm5-kda-requant.py): per-matrix kernel µs
    /// (CUDA events, T = 1, median of 50) of g5f_gemv and — when the tables build (Q8_0 or
    /// Q4_K) — k5_qgemv, each checked against the f32 dequant · x on the CPU (rel < 2e-2:
    /// the int8-activation band); then G5F=1 vs 2 through 32 naive decode steps. Skips
    /// without the env var.
    #[test]
    fn kda_quant_real_bench() {
        let _env = crate::flash::glm5::env_write();
        let Ok(real) = std::env::var("TANG_GLM5_REAL_GGUF") else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let g = Gpu::for_test().expect("GPU");
        g.bind().unwrap();
        let gguf = crate::gguf::Gguf::open(Path::new(&real)).unwrap();
        let hp = super::super::super::Hparams::from_gguf(&gguf).unwrap();
        let l = (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let n = hp.n_embd;
        let mk = |g5f: u8| {
            std::env::set_var("TANG_GLM5_KDA_G5F", g5f.to_string());
            let d = DevLayer::new_gguf(
                &g,
                &gguf,
                l,
                &cfg,
                n,
                &ConvState::zeros(&cfg, 1),
                &KdaState::zeros(&cfg, 1),
                64,
                1,
            )
            .unwrap();
            std::env::remove_var("TANG_GLM5_KDA_G5F");
            d
        };
        let mut seed = 11u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let xs: Vec<Vec<f32>> = (0..32).map(|_| (0..n).map(|_| rnd()).collect()).collect();
        let d = mk(2);
        let qg = d.segx.is_some() && d.sego.is_some();
        let (di, hd, hid) = (d.d_inner, d.hd, d.hidden);
        let yo: Vec<f32> = xs[0].iter().cycle().take(di).copied().collect();
        d.xb.write(0, &xs[0]).unwrap();
        d.yb.write(0, &yo).unwrap();
        let (e0, e1) = (Event::new(true).unwrap(), Event::new(true).unwrap());
        let time = |f: &dyn Fn()| -> f32 {
            for _ in 0..5 {
                f();
            }
            let mut v: Vec<f32> = (0..50)
                .map(|_| {
                    e0.record(&d.s).unwrap();
                    f();
                    e1.record(&d.s).unwrap();
                    d.s.sync().unwrap();
                    e1.since(&e0).unwrap() * 1e3
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[25]
        };
        let t = |s: &str| format!("blk.{l}.{s}");
        let mats = [
            ("q", &d.wq, "attn_q.weight", di, hid, d.xb.ptr, &xs[0]),
            ("k", &d.wk, "attn_k.weight", di, hid, d.xb.ptr, &xs[0]),
            ("v", &d.wv, "attn_v.weight", di, hid, d.xb.ptr, &xs[0]),
            ("f_a", &d.fa, "ssm_f_a.weight", hd, hid, d.xb.ptr, &xs[0]),
            ("g_a", &d.ga, "ssm_g_a.weight", hd, hid, d.xb.ptr, &xs[0]),
            ("o", &d.wo, "attn_output.weight", hid, di, d.yb.ptr, &yo),
        ];
        let codes = d.xq8.ptr;
        let (mut tf_sum, mut tq_sum, mut to) = (0f32, 0f32, 0f32);
        let ybuf = tang_moe::gpu::DevBuf::zeroed_elems::<f32>(di.max(hid)).unwrap();
        for (name, w, tn, rows, cols, x, xh) in mats {
            let info = gguf.info(&t(tn)).unwrap();
            let wf = gguf.dequantize(info).unwrap();
            let want: Vec<f32> = (0..rows)
                .map(|r| {
                    wf[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(xh.iter())
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum::<f64>() as f32
                })
                .collect();
            unsafe {
                launch(
                    d.gv.q8,
                    ((cols / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &d.s,
                    args![x, codes, cols as u32, d.max_t as u32],
                )
                .unwrap();
            }
            let scales = codes + (d.max_t * cols) as u64;
            let a = crate::flash::glm5::moe::gpu::Act {
                xf: x,
                xq: codes,
                xq2: 0,
                xd: scales,
                xs: cols,
            };
            let tf = time(&|| {
                d.gv.fast
                    .gemv(&d.s, w.w, 0, a, ybuf.ptr, rows, 1, rows, cols)
                    .unwrap()
            });
            d.s.sync().unwrap();
            let ef = rel64(&ybuf.read::<f32>(0, rows).unwrap(), &want);
            let mb = w.w.estride as f64 / 1e6;
            let mut line = format!(
                "quant bench {name:>3} {} {rows}x{cols} ({mb:.1} MB): g5f_gemv {tf:.1} us ({:.2} TB/s, rel {ef:.2e})",
                info.ty.name(), mb / tf as f64
            );
            assert!(ef < 2e-2, "{name}: g5f_gemv rel {ef} vs the f32 dequant");
            tf_sum += tf;
            if qg {
                let tab = tang_moe::gpu::DevBuf::from_slice(&[
                    w.w.base,
                    ybuf.ptr,
                    w.w.row_bytes as u64 | ((rows as u64) << 32),
                ])
                .unwrap();
                let tq = time(&|| unsafe {
                    d.qgemv_launch(&d.s, &tab, 1, w.w.row_bytes, rows, 0, codes, scales, cols)
                        .unwrap()
                });
                d.s.sync().unwrap();
                let eq = rel64(&ybuf.read::<f32>(0, rows).unwrap(), &want);
                line += &format!(
                    ", k5_qgemv {tq:.1} us ({:.2} TB/s, rel {eq:.2e})",
                    mb / tq as f64
                );
                assert!(eq < 2e-2, "{name}: k5_qgemv rel {eq} vs the f32 dequant");
                tq_sum += tq;
                if name == "o" {
                    to = tq;
                }
            }
            eprintln!("{line}");
        }
        if qg {
            unsafe {
                launch(
                    d.gv.q8,
                    ((hid / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &d.s,
                    args![d.xb.ptr, codes, hid as u32, d.max_t as u32],
                )
                .unwrap();
            }
            let mb5: f64 = [&d.wq, &d.wk, &d.wv, &d.fa, &d.ga]
                .iter()
                .map(|w| w.w.estride as f64)
                .sum::<f64>()
                / 1e6;
            let tm = time(&|| unsafe {
                d.qgemv_launch(
                    &d.s,
                    d.segx.as_ref().unwrap(),
                    5,
                    d.wq.w.row_bytes,
                    3 * di + 2 * hd,
                    0,
                    codes,
                    codes + (d.max_t * hid) as u64,
                    hid,
                )
                .unwrap()
            });
            eprintln!(
                "quant bench merged q|k|v|f_a|g_a ({mb5:.1} MB): k5_qgemv {tm:.1} us ({:.2} TB/s)",
                mb5 / tm as f64
            );
            eprintln!("quant bench per layer: g5f_gemv {tf_sum:.1} us, k5_qgemv {tq_sum:.1} us; merged + o {:.1} us", tm + to);
        } else {
            eprintln!(
                "quant bench per layer: g5f_gemv {tf_sum:.1} us (no k5_qgemv tables for this type)"
            );
        }
        drop(d);
        let run = |g5f: u8| {
            let mut d = mk(g5f);
            let mut conv = ConvState::zeros(&cfg, 1);
            let ys: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| d.window(x, 1, &mut conv).unwrap())
                .collect();
            let mut st = KdaState::zeros(&cfg, 1);
            let mut hc = conv.clone();
            d.read_back(&mut hc, &mut st).unwrap();
            (ys.concat(), st)
        };
        let (a, b) = (run(1), run(2));
        let (eo, es) = (rel64(&b.0, &a.0), rel64(&b.1 .0, &a.1 .0));
        eprintln!(
            "quant parity G5F=2 vs G5F=1 naive, 32 steps: out rel {eo:.3e}, state rel {es:.3e}"
        );
        assert!(eo < 1e-2 && es < 1e-2, "G5F=2 diverges from G5F=1");
    }

    fn rel64(a: &[f32], b: &[f32]) -> f64 {
        if a.iter().any(|x| !x.is_finite()) {
            return f64::INFINITY;
        }
        let (mut d, mut n) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += ((x - y) as f64).powi(2);
            n += (*y as f64).powi(2);
        }
        (d / n.max(1e-30)).sqrt()
    }
}

/// `TANG_GLM5_KDA_TAIL` (default 1): `0` keeps the G5F fused step on the separate
/// k5_beta / k5d_gate2 / k5d_step / k5_q8 launches (the A/B control for `k5d_tail`).
fn tail_flag() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("TANG_GLM5_KDA_TAIL").map_or(true, |v| v != "0"))
}

/// `TANG_GLM5_KDA_ROWS_FAST=0`: rows_enqueue keeps the generic t-row kernels.
/// `TANG_GLM5_KDA_TAILT=0`: rows_fast runs k5d_tail once per row instead of k5d_tailt.
/// `TANG_GLM5_KDA_SPLIT=0`: rows_fast runs k5d_tailt (round 4) instead of k5d_pret + k5d_stept.
fn split_flag() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("TANG_GLM5_KDA_SPLIT").map_or(true, |v| v != "0"))
}

fn tailt_flag() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("TANG_GLM5_KDA_TAILT").map_or(true, |v| v != "0"))
}

fn rows_fast() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("TANG_GLM5_KDA_ROWS_FAST").map_or(true, |v| v != "0"))
}

impl DevLayer {
    /// [`Self::rows_enqueue`] on the tuned t=1 kernels, bit-equal per row to t serial
    /// decode steps (NOTES 2026-10-28 kda4 round 3): k5_q8 over the t rows of x, ONE
    /// k5_qgemvt9 (q|k|v|f_a|g_a, the weight stream read once for all t rows), then per row
    /// k5d_tail (gates, beta, conv, the recurrent step, norm, yb's q8 at row i; the state
    /// checkpoint after each row but the last), then ONE k5_qgemvt17 for o_proj.
    /// The t-row fast path applies (resident.rs window_resident routes a t>1 KDA item
    /// through [`Self::rows_enqueue`] when this holds).
    pub fn rows_fast_ok(&self, t: usize) -> bool {
        (2..=4).contains(&t)
            && rows_fast()
            && self.tail_ok()
            && self.segx.is_some()
            && self.sego.is_some()
    }

    unsafe fn rows_fast(&self, s: &Stream, t: usize, ck: Option<(u64, usize)>) -> Result<()> {
        let fast = self.fast.as_ref().context("kda rows: kernels not built")?;
        let (hd, di, hidden) = (self.hd, self.d_inner, self.hidden);
        let lb = self.cfg.gate_lower_bound.unwrap_or(f32::NAN);
        let codes = self.xq8.ptr;
        let regs = self.core.slot_state_regions(0);
        unsafe {
            launch(
                self.gv.q8,
                ((hidden / 32) as u32, t as u32, 1),
                (32, 1, 1),
                0,
                s,
                args![self.xb.ptr, codes, hidden as u32, self.max_t as u32],
            )?;
            let rows = 3 * di + 2 * hd;
            launch(
                if self.wq.w.row_bytes as usize <= 9 * 32 * 16 {
                    fast.qgt9
                } else {
                    fast.qgt17
                },
                (rows.div_ceil(4) as u32, 1, 1),
                (128, 1, 1),
                0,
                s,
                args![
                    self.segx.as_ref().unwrap().ptr,
                    5u32,
                    codes,
                    codes + (self.max_t * hidden) as u64,
                    hidden as u32,
                    3u32,
                    (3 * di) as u32,
                    t as u32
                ],
            )?;
            if split_flag() {
                // heads x rows pre-pass (state independent), then the serial state rows
                let (ckp, ckf) = ck.map_or((0u64, 0u64), |(b, st)| (b, (st / 4) as u64));
                launch(
                    fast.pret,
                    (self.nh as u32, t as u32, 1),
                    (self.hd as u32, 1, 1),
                    (self.hidden * 4 + 16) as u32,
                    s,
                    args![
                        self.fb.w,
                        self.gb.w,
                        self.bproj.w,
                        self.xb.ptr,
                        self.g1b.ptr,
                        self.g2a.ptr,
                        self.dt.ptr,
                        self.rate.ptr,
                        lb,
                        self.qkvb.ptr,
                        regs[0].0,
                        self.core.wconv.ptr,
                        regs[1].0,
                        self.on.ptr,
                        self.cfg.eps,
                        self.bb.ptr,
                        self.gb_out.ptr,
                        self.g2b.ptr,
                        self.rsc.ptr,
                        self.hidden as u32,
                        t as u32,
                        ckp,
                        ckf
                    ],
                )?;
                launch(
                    fast.stept,
                    (self.nh as u32, 1, 1),
                    (self.hd as u32, 1, 1),
                    0,
                    s,
                    args![
                        self.qkvb.ptr,
                        regs[0].0,
                        regs[1].0,
                        self.on.ptr,
                        self.cfg.eps,
                        self.bb.ptr,
                        self.g2b.ptr,
                        self.rsc.ptr,
                        self.ob.ptr,
                        self.yb.ptr,
                        self.xq8.ptr,
                        self.max_t as u32,
                        t as u32,
                        ckp,
                        ckf,
                        (regs[0].1 / 4) as u64
                    ],
                )?;
            } else if tailt_flag() {
                // one launch for the t rows; the checkpoints are written from registers
                let (ckp, ckf) = ck.map_or((0u64, 0u64), |(b, st)| (b, (st / 4) as u64));
                launch(
                    fast.tailt,
                    (self.nh as u32, 1, 1),
                    (self.hd as u32, 1, 1),
                    (self.hidden * 4 + 16) as u32,
                    s,
                    args![
                        self.fb.w,
                        self.gb.w,
                        self.bproj.w,
                        self.xb.ptr,
                        self.g1b.ptr,
                        self.g2a.ptr,
                        self.dt.ptr,
                        self.rate.ptr,
                        lb,
                        self.qkvb.ptr,
                        regs[0].0,
                        self.core.wconv.ptr,
                        regs[1].0,
                        self.on.ptr,
                        self.cfg.eps,
                        self.bb.ptr,
                        self.gb_out.ptr,
                        self.g2b.ptr,
                        self.ob.ptr,
                        self.yb.ptr,
                        self.xq8.ptr,
                        self.max_t as u32,
                        self.hidden as u32,
                        t as u32,
                        ckp,
                        ckf,
                        (regs[0].1 / 4) as u64
                    ],
                )?;
            }
            for i in 0..if tailt_flag() || split_flag() { 0 } else { t } {
                self.tail_launch(s, fast, self.core.h0.ptr, self.core.st.ptr, lb, i)?;
                if let (Some((base, stride)), true) = (ck, i + 1 < t) {
                    let b = base + (i * stride) as u64;
                    tang_moe::gpu::copy_d2d_async(b, regs[0].0, regs[0].1, s)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    tang_moe::gpu::copy_d2d_async(b + regs[0].1 as u64, regs[1].0, regs[1].1, s)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                }
            }
            launch(
                if self.wo.w.row_bytes as usize <= 9 * 32 * 16 {
                    fast.qgt9
                } else {
                    fast.qgt17
                },
                (hidden.div_ceil(4) as u32, 1, 1),
                (128, 1, 1),
                0,
                s,
                args![
                    self.sego.as_ref().unwrap().ptr,
                    1u32,
                    codes,
                    codes + (self.max_t * di) as u64,
                    di as u32,
                    0u32,
                    0u32,
                    t as u32
                ],
            )?;
        }
        Ok(())
    }

    /// `k5d_tail`'s preconditions (kda_decode.cuh): f_b / g_b / b_proj Q8_0 with hd-wide
    /// f_b / g_b rows, hd % 32 == 0, hidden <= K5D_HIDMAX and a multiple of 256.
    fn tail_ok(&self) -> bool {
        let q8 = |w: &super::super::moe::gpu::DevW| w.w.ty == 8;
        tail_flag()
            && self.fast.is_some()
            && q8(&self.fb)
            && q8(&self.gb)
            && q8(&self.bproj)
            && self.hd % 32 == 0
            && self.fb.w.row_bytes as usize == self.hd / 32 * 34
            && self.gb.w.row_bytes as usize == self.hd / 32 * 34
            && self.hidden <= 4096
            && self.hidden % 256 == 0
    }

    /// One `k5d_tail` launch: grid nh, block hd. `h0` / `st` = the slot's conv history /
    /// state bases. Writes bb, gb_out, g2b, ob, yb and yb's k5_q8 codes into xq8.
    /// `i` = the row of a t-row window (0 for a decode step): every per-row input / output
    /// pointer moves by row i, the state (h0 / st) does not (rows run serially).
    unsafe fn tail_launch(
        &self,
        s: &Stream,
        fast: &Fast,
        h0: u64,
        st: u64,
        lb: f32,
        i: usize,
    ) -> Result<()> {
        let f = 4u64;
        let (i, di, hd, nh) = (
            i as u64,
            self.d_inner as u64,
            self.hd as u64,
            self.nh as u64,
        );
        unsafe {
            launch(
                fast.tail,
                (self.nh as u32, 1, 1),
                (self.hd as u32, 1, 1),
                (self.hidden * 4 + 16) as u32,
                s,
                args![
                    self.fb.w,
                    self.gb.w,
                    self.bproj.w,
                    self.xb.ptr + i * self.hidden as u64 * f,
                    self.g1b.ptr + i * hd * f,
                    self.g2a.ptr + i * hd * f,
                    self.dt.ptr,
                    self.rate.ptr,
                    lb,
                    self.qkvb.ptr + i * 3 * di * f,
                    h0,
                    self.core.wconv.ptr,
                    st,
                    self.on.ptr,
                    self.cfg.eps,
                    self.bb.ptr + i * nh * f,
                    self.gb_out.ptr + i * di * f,
                    self.g2b.ptr + i * di * f,
                    self.ob.ptr + i * di * f,
                    self.yb.ptr + i * di * f,
                    self.xq8.ptr,
                    self.max_t as u32,
                    self.hidden as u32,
                    i as u32
                ],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tail_tests {
    use super::super::super::kda::{ConvState, KdaConfig, KdaState};
    use super::*;
    use std::path::Path;

    /// REAL WEIGHTS (TANG_GLM5_REAL_GGUF, run with TANG_GLM5_KDA_G5F=2): (a) the G5F fused
    /// step kernel by kernel (CUDA events around each launch, median of 56 steps) on the
    /// old launches (k5_q8 x, k5_qgemv q|k|v|f_a|g_a, k5_beta, k5d_gate2, k5d_step, k5_q8 yb,
    /// k5_qgemv o) and on the k5d_tail form (k5_q8 x, k5_qgemv, k5d_tail, k5_qgemv o);
    /// (b) parity, 64 steps: the tail form (eager and graph via `decode_fast`) vs the old
    /// launches — bit-equal output, state and conv.
    #[test]
    fn kda_tail_real_bench() {
        let _env = crate::flash::glm5::env_read();
        let Ok(real) = std::env::var("TANG_GLM5_REAL_GGUF") else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let g = Gpu::for_test().expect("GPU");
        g.bind().unwrap();
        let gguf = crate::gguf::Gguf::open(Path::new(&real)).unwrap();
        let hp = super::super::super::Hparams::from_gguf(&gguf).unwrap();
        let l = (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let n = hp.n_embd;
        let mk = || {
            let mut d = DevLayer::new_gguf(
                &g,
                &gguf,
                l,
                &cfg,
                n,
                &ConvState::zeros(&cfg, 1),
                &KdaState::zeros(&cfg, 1),
                64,
                1,
            )
            .unwrap();
            d.fast = Some(Fast::new(&g, &cfg).unwrap());
            d
        };
        let mut seed = 11u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let steps = 64usize;
        let xs: Vec<Vec<f32>> = (0..steps)
            .map(|_| (0..n).map(|_| rnd()).collect())
            .collect();
        let d0 = mk();
        assert!(
            d0.g5f == 2 && d0.segx.is_some() && d0.sego.is_some(),
            "run with TANG_GLM5_KDA_G5F=2 (Q8_0 weights)"
        );
        assert!(
            d0.tail_ok(),
            "k5d_tail preconditions fail on the real layer"
        );
        let (nh, hd, di, hidden) = (d0.nh, d0.hd, d0.d_inner, d0.hidden);
        eprintln!(
            "layer {l}: nh {nh} hd {hd} d_inner {di} hidden {hidden}; state {:.2} MiB/layer",
            (nh * hd * hd * 4) as f64 / 1048576.0
        );
        let lb = d0.cfg.gate_lower_bound.unwrap_or(f32::NAN);
        drop(d0);
        // the launches, one closure per kernel (index into the timing table)
        type K<'a> = Box<dyn Fn(&DevLayer, &Stream) + 'a>;
        let q8x: K = Box::new(|d, s| unsafe {
            launch(
                d.gv.q8,
                ((d.hidden / 32) as u32, 1, 1),
                (32, 1, 1),
                0,
                s,
                args![d.xb.ptr, d.xq8.ptr, d.hidden as u32, d.max_t as u32],
            )
            .unwrap();
        });
        let qx: K = Box::new(|d, s| unsafe {
            let c = d.xq8.ptr;
            let bytes = [&d.wq, &d.wk, &d.wv, &d.fa, &d.ga]
                .iter()
                .map(|w| w.w.estride)
                .sum();
            d.qgemv_launch(
                s,
                d.segx.as_ref().unwrap(),
                5,
                d.wq.w.row_bytes,
                3 * d.d_inner + 2 * d.hd,
                bytes,
                c,
                c + (d.max_t * d.hidden) as u64,
                d.hidden,
            )
            .unwrap();
        });
        let beta: K = Box::new(|d, s| unsafe {
            launch(
                d.gv.beta,
                (d.nh.div_ceil(super::super::K5_WARPS) as u32, 1, 1),
                (128, 1, 1),
                0,
                s,
                args![
                    d.bproj.w,
                    d.xb.ptr,
                    d.bb.ptr,
                    1u32,
                    d.hidden as u32,
                    d.nh as u32
                ],
            )
            .unwrap();
        });
        let gate2: K = Box::new(move |d, s| unsafe {
            launch(
                d.fast.as_ref().unwrap().gate2,
                grid(2 * d.d_inner),
                (128, 1, 1),
                0,
                s,
                args![
                    d.fb.w,
                    d.gb.w,
                    d.g1b.ptr,
                    d.g2a.ptr,
                    d.dt.ptr,
                    d.rate.ptr,
                    d.gb_out.ptr,
                    d.g2b.ptr,
                    d.nh as u32,
                    d.hd as u32,
                    lb
                ],
            )
            .unwrap();
        });
        let step: K = Box::new(|d, s| unsafe {
            launch(
                d.fast.as_ref().unwrap().step,
                (d.nh as u32, 1, 1),
                (d.hd as u32, 1, 1),
                0,
                s,
                args![
                    d.qkvb.ptr,
                    d.core.h0.ptr,
                    d.core.wconv.ptr,
                    d.gb_out.ptr,
                    d.bb.ptr,
                    d.core.st.ptr,
                    d.ob.ptr,
                    d.on.ptr,
                    d.g2b.ptr,
                    d.yb.ptr,
                    d.cfg.eps
                ],
            )
            .unwrap();
        });
        let q8y: K = Box::new(|d, s| unsafe {
            launch(
                d.gv.q8,
                ((d.d_inner / 32) as u32, 1, 1),
                (32, 1, 1),
                0,
                s,
                args![d.yb.ptr, d.xq8.ptr, d.d_inner as u32, d.max_t as u32],
            )
            .unwrap();
        });
        let qo: K = Box::new(|d, s| unsafe {
            let c = d.xq8.ptr;
            d.qgemv_launch(
                s,
                d.sego.as_ref().unwrap(),
                1,
                d.wo.w.row_bytes,
                d.hidden,
                d.wo.w.estride,
                c,
                c + (d.max_t * d.d_inner) as u64,
                d.d_inner,
            )
            .unwrap();
        });
        let tail: K = Box::new(move |d, s| unsafe {
            d.tail_launch(
                s,
                d.fast.as_ref().unwrap(),
                d.core.h0.ptr,
                d.core.st.ptr,
                lb,
                0,
            )
            .unwrap();
        });
        let old: Vec<(&str, &K)> = vec![
            ("k5_q8 x", &q8x),
            ("k5_qgemv q|k|v|f_a|g_a", &qx),
            ("k5_beta", &beta),
            ("k5d_gate2", &gate2),
            ("k5d_step", &step),
            ("k5_q8 yb", &q8y),
            ("k5_qgemv o", &qo),
        ];
        let new: Vec<(&str, &K)> = vec![
            ("k5_q8 x", &q8x),
            ("k5_qgemv q|k|v|f_a|g_a", &qx),
            ("k5d_tail", &tail),
            ("k5_qgemv o", &qo),
        ];
        let mut outs = Vec::new();
        let mut states = Vec::new();
        let mut taps: Vec<[Vec<f32>; 5]> = Vec::new();
        for (form, seq) in [("old", &old), ("tail", &new)] {
            let d = mk();
            let ev: Vec<Event> = (0..=seq.len()).map(|_| Event::new(true).unwrap()).collect();
            let mut per: Vec<Vec<f32>> = vec![Vec::new(); seq.len() + 1];
            let mut ys = Vec::new();
            for (i, x) in xs.iter().enumerate() {
                d.xb.write(0, x).unwrap();
                ev[0].record(&d.s).unwrap();
                for (j, (_, k)) in seq.iter().enumerate() {
                    k(&d, &d.s);
                    ev[j + 1].record(&d.s).unwrap();
                }
                d.s.sync().unwrap();
                ys.push(d.outb.read::<f32>(0, hidden).unwrap());
                if i == 0 {
                    taps.push([
                        d.bb.read::<f32>(0, nh).unwrap(),
                        d.gb_out.read::<f32>(0, di).unwrap(),
                        d.g2b.read::<f32>(0, di).unwrap(),
                        d.ob.read::<f32>(0, di).unwrap(),
                        d.yb.read::<f32>(0, di).unwrap(),
                    ]);
                }
                if i >= 8 {
                    for j in 0..seq.len() {
                        per[j].push(ev[j + 1].since(&ev[j]).unwrap() * 1e3);
                    }
                    per[seq.len()].push(ev[seq.len()].since(&ev[0]).unwrap() * 1e3);
                }
            }
            let med = |v: &mut Vec<f32>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            for (j, (name, _)) in seq.iter().enumerate() {
                eprintln!("tail bench {form}: {name:<24} {:7.1} us", med(&mut per[j]));
            }
            eprintln!(
                "tail bench {form}: {:<24} {:7.1} us (events, back to back, one stream)",
                "TOTAL step",
                med(&mut per[seq.len()])
            );
            let mut conv = ConvState::zeros(&cfg, 1);
            let mut st = KdaState::zeros(&cfg, 1);
            d.read_back(&mut conv, &mut st).unwrap();
            outs.push(ys);
            states.push((conv, st));
        }
        for (ti, name) in ["beta", "g", "g2", "o", "y"].iter().enumerate() {
            let a = &taps[0][ti];
            let b = &taps[1][ti];
            let n = a
                .iter()
                .zip(b)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let m = a
                .iter()
                .zip(b)
                .map(|(a, b)| ((a - b).abs() / a.abs().max(1e-30)) as f64)
                .fold(0.0, f64::max);
            eprintln!(
                "tail tap step 0 {name}: {n}/{} differ, max rel {m:.2e}",
                a.len()
            );
        }
        // the production entry: decode_fast eager and graph (tail on unless TANG_GLM5_KDA_TAIL=0)
        for mode in [Mode::Fused, Mode::Graph] {
            let mut d = mk();
            let mut conv = ConvState::zeros(&cfg, 1);
            let mut ys = Vec::new();
            let t0 = std::time::Instant::now();
            for x in &xs {
                ys.push(d.decode_fast(x, &mut conv, mode).unwrap());
            }
            eprintln!("tail bench decode_fast {mode:?}: {:.0} us/step host wall (64 steps, incl. capture)", t0.elapsed().as_secs_f64() * 1e6 / steps as f64);
            let mut hc = conv.clone();
            let mut st = KdaState::zeros(&cfg, 1);
            d.read_back(&mut hc, &mut st).unwrap();
            outs.push(ys);
            states.push((hc, st));
        }
        for (mi, name) in [
            (1, "tail eager"),
            (2, "decode_fast fused"),
            (3, "decode_fast graph"),
        ] {
            let diff = outs[mi]
                .iter()
                .zip(&outs[0])
                .flat_map(|(a, b)| a.iter().zip(b))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let sdiff = states[mi]
                .1
                 .0
                .iter()
                .zip(&states[0].1 .0)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let cdiff = states[mi]
                .0
                 .0
                .iter()
                .zip(&states[0].0 .0)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!("{name} vs old launches: {diff} output values differ (bits), state {sdiff}, conv {cdiff}");
            assert_eq!(
                (diff, sdiff, cdiff),
                (0, 0, 0),
                "{name} is not bit-equal to the old launches"
            );
        }
    }
}

impl DevLayer {
    /// The resident MTP window's batched KDA layer (`TANG_GLM5_RES_KDA_ROWS`, default on):
    /// `xb` holds `t` consecutive rows of slot 0. The projections, gates, gated norm and
    /// o_proj run ONCE over the t rows (each weight read once); only the recurrent core —
    /// conv + delta rule, the t=1 step kernel — runs row by row on slot 0's state (the
    /// WIN_STEP form of `core_pf2`). After row `i < t - 1` the slot's state is copied to
    /// `ck.0 + i * ck.1` (the window's rewind checkpoints). Everything on `s`, device
    /// pointers only (capturable); `outb` holds y `[t][hidden]`. Not bit-identical to the
    /// t=1 decode step (batched GEMVs; the windiff note in NOTES.md).
    pub fn rows_enqueue(&self, s: &Stream, t: usize, ck: Option<(u64, usize)>) -> Result<()> {
        ensure!(
            self.pf2 && t >= 1 && t <= self.max_t,
            "kda rows_enqueue: pf2 {} t {t}",
            self.pf2
        );
        if self.rows_fast_ok(t) {
            return unsafe { self.rows_fast(s, t, ck) };
        }
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        let lb = self.cfg.gate_lower_bound.unwrap_or(f32::NAN);
        let f = std::mem::size_of::<f32>() as u64;
        let qkv = self.qkvb.ptr;
        let g8 = |w: &super::super::moe::gpu::DevW| self.gemv8 && w.w.ty == 8;
        let gemv8 = g8(&self.wq) && g8(&self.wk) && g8(&self.wv);
        unsafe {
            let g5 = self.g5f_x(s, t)?;
            if !g5 {
                if gemv8 {
                    launch(
                        self.gv.q8,
                        ((hidden / 32) as u32, t as u32, 1),
                        (32, 1, 1),
                        0,
                        s,
                        args![self.xb.ptr, self.xq8.ptr, hidden as u32, self.max_t as u32],
                    )?;
                    for (w, off) in [
                        (&self.wq, 0u64),
                        (&self.wk, di as u64 * f),
                        (&self.wv, 2 * di as u64 * f),
                    ] {
                        self.gemv8_launch_on(s, w, self.xq8.ptr, hidden, qkv + off, 3 * di, di, t)?;
                    }
                } else {
                    for (w, off) in [
                        (&self.wq, 0u64),
                        (&self.wk, di as u64 * f),
                        (&self.wv, 2 * di as u64 * f),
                    ] {
                        launch(
                            self.gv.gemv,
                            grid(t * di),
                            (128, 1, 1),
                            0,
                            s,
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
                if gemv8 && g8(&self.fa) {
                    self.gemv8_launch_on(
                        s,
                        &self.fa,
                        self.xq8.ptr,
                        hidden,
                        self.g1b.ptr,
                        hd,
                        hd,
                        t,
                    )?;
                } else {
                    launch(
                        self.gv.gemv,
                        grid(t * hd),
                        (128, 1, 1),
                        0,
                        s,
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
            }
            launch(
                self.gv.gemv,
                grid(t * di),
                (128, 1, 1),
                0,
                s,
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
                s,
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
            launch(
                self.gv.beta,
                ((t * nh).div_ceil(super::K5_WARPS) as u32, 1, 1),
                (128, 1, 1),
                0,
                s,
                args![
                    self.bproj.w,
                    self.xb.ptr,
                    self.bb.ptr,
                    t as u32,
                    hidden as u32,
                    nh as u32
                ],
            )?;
            // g_a (unless g5f_x wrote it): before o_proj re-quantises xq8
            if !g5 {
                if gemv8 && g8(&self.ga) {
                    self.gemv8_launch_on(
                        s,
                        &self.ga,
                        self.xq8.ptr,
                        hidden,
                        self.g2a.ptr,
                        hd,
                        hd,
                        t,
                    )?;
                } else {
                    launch(
                        self.gv.gemv,
                        grid(t * hd),
                        (128, 1, 1),
                        0,
                        s,
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
            }
        }
        // the core, row by row on slot 0
        let regs = self.core.slot_state_regions(0);
        for i in 0..t as u64 {
            self.core.decode_dev(
                &self.kers,
                s,
                0,
                qkv + i * (3 * di) as u64 * f,
                self.gb_out.ptr + i * di as u64 * f,
                self.bb.ptr + i * nh as u64 * f,
                self.ob.ptr + i * di as u64 * f,
            )?;
            if let (Some((base, stride)), true) = (ck, (i as usize) + 1 < t) {
                let b = base + i * stride as u64;
                tang_moe::gpu::copy_d2d_async(b, regs[0].0, regs[0].1, s)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                tang_moe::gpu::copy_d2d_async(b + regs[0].1 as u64, regs[1].0, regs[1].1, s)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
        }
        unsafe {
            launch(
                self.gv.gemv,
                grid(t * di),
                (128, 1, 1),
                0,
                s,
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
                s,
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
            if self.g5f_o(s, t)? {
            } else if g8(&self.wo) {
                launch(
                    self.gv.q8,
                    ((di / 32) as u32, t as u32, 1),
                    (32, 1, 1),
                    0,
                    s,
                    args![self.yb.ptr, self.xq8.ptr, di as u32, self.max_t as u32],
                )?;
                self.gemv8_launch_on(
                    s,
                    &self.wo,
                    self.xq8.ptr,
                    di,
                    self.outb.ptr,
                    hidden,
                    hidden,
                    t,
                )?;
            } else {
                launch(
                    self.gv.gemv,
                    grid(t * hidden),
                    (128, 1, 1),
                    0,
                    s,
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
        Ok(())
    }
}
