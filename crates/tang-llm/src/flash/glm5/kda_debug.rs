//! Shadow-check debugging for the GPU-resident KDA layers (`TANG_GLM5_KDA_DEBUG=1`, or
//! `TANG_KERNEL_VERIFY=1` / `=kda`), on the [`tang_verify`] harness.
//!
//! The staging bisect (Oct 2026) chased a GPU-vs-CPU divergence through four pod runs
//! with `TANG_GLM5_KDA_DUMP`'s first-8-value prints, paired ACROSS runs and machines — and
//! got stuck on token 0, whose core output is ~1e-7 noise on both paths. This replaces that
//! with an in-process check that answers "which kernel is wrong" in one run:
//!
//! - every resident window of a checked layer is re-run on the CPU (`kda::kda_layer_taps`)
//!   from the SAME `x` the GPU got, so upstream divergence can't leak in;
//! - the CPU shadow starts from the device's own state (resync after every window, default
//!   on), so each window tests ONE step, not accumulated drift;
//! - every intermediate is compared in full (all tokens, all channels): q/k/v
//!   projections, gates, conv output, core output, g2, gated norm, o_proj, plus the
//!   device's recurrent + conv state against the shadow's and the host conv copy;
//! - device scratch is poisoned with NaN before each window (`..._POISON`, default on),
//!   so an output region a kernel never wrote reads NaN instead of a plausible 0.0;
//! - the harness attributes each failing window (`kdadbg VERDICT`, the first one also as
//!   `kdadbg FIRST DIVERGENCE`): provenance (whose values sit in the failing tap) and
//!   unwritten/zero anomalies outrank pipeline order, and q|k|v are declared as sections
//!   of one buffer so a misplaced write is named with its actual offset. Full tensors of
//!   failing windows go to `..._DIR`.
//!
//! Env (all optional; each also reads `TANG_KERNEL_VERIFY_{KNOB}`):
//! - `TANG_GLM5_KDA_DEBUG=1`           enable
//! - `TANG_GLM5_KDA_DEBUG_LAYERS=0,1`  only these (global) layer indices (default: all KDA)
//! - `TANG_GLM5_KDA_DEBUG_WINDOWS=4`   only the first N windows per layer (default: all)
//! - `TANG_GLM5_KDA_DEBUG_TOL=1e-3`    relative-L2 bar per tensor
//! - `TANG_GLM5_KDA_DEBUG_RESYNC=0`    let the shadow carry its own state (tests drift)
//! - `TANG_GLM5_KDA_DEBUG_POISON=0`    don't NaN-fill device scratch
//! - `TANG_GLM5_KDA_DEBUG_DIR=/path`   write `w{W}_l{L}_{tap}.{gpu,cpu}.f32` (raw LE f32)
//!   for the first window checked and the first `..._DUMPS` (default 3) failing windows
//!
//! Output lines all start with `kdadbg` (grep them out of the run log).
use super::kda::{ConvState, KdaConfig, KdaLayerW, KdaState};
use anyhow::Result;
use tang_verify::{Layout, Probe, Spec, Tap, Taps, Verifier};

/// The adapter's legacy switch (and knob prefix).
pub const LEGACY: &str = "TANG_GLM5_KDA_DEBUG";

/// Whether device code should poison/tap: `TANG_GLM5_KDA_DEBUG=1` and poison not
/// explicitly off (the fused decode path's cheap per-call check; the harness's config
/// reads the same knob).
pub fn poison_on() -> bool {
    let flag = |k: &str, default: bool| match std::env::var(k) {
        Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => default,
    };
    std::env::var(LEGACY).is_ok_and(|v| v == "1") && flag("TANG_GLM5_KDA_DEBUG_POISON", true)
}

/// Tap names both paths fill (host layouts, shared): `qkv_pre`/`conv_out` `[t][3·di]`
/// (q|k|v stacked per token), `g`/`o`/`g2`/`y_norm` `[t][di]`, `beta` `[t][nh]`, `out`
/// `[t][hidden]`. A tap a path can't observe (the decode step fuses the conv) is left out
/// and skipped.
pub mod tap {
    pub const QKV: &str = "qkv_pre";
    pub const CONV: &str = "conv_out";
    pub const G: &str = "g";
    pub const BETA: &str = "beta";
    pub const O: &str = "o";
    pub const G2: &str = "g2";
    pub const Y_NORM: &str = "y_norm";
    pub const OUT: &str = "out";
}

/// The KDA layer's taps in pipeline order: what each reads and which kernel writes it.
pub fn spec(kcfg: &KdaConfig, hidden: usize) -> Spec {
    let (nh, hd) = (kcfg.heads, kcfg.head_dim);
    let di = nh * hd;
    let inner = Layout::new().dim("tok", 0).dim("h", nh).dim("d", hd);
    let state = Layout::new().dim("h", 0).dim("key", hd).dim("val", hd);
    let conv = Layout::new()
        .labeled("sec", &["q", "k", "v"])
        .dim("h", nh)
        .dim("d", hd)
        .dim("tap", kcfg.d_conv - 1);
    let qkv = ["qkv.q", "qkv.k", "qkv.v"];
    let mut s = Spec::new("kda")
        .tap(Tap::new("state.in", state.clone()).what("device state going in vs the shadow's"))
        .stacked(
            tap::QKV,
            &[
                ("qkv.q", di, "k5_gemv attn_q (device)"),
                ("qkv.k", di, "k5_gemv attn_k (device)"),
                ("qkv.v", di, "k5_gemv attn_v (device)"),
            ],
            inner.clone(),
            &[],
        )
        .tap(Tap::new(tap::G, inner.clone()).what("k5_gemv f_a, f_b + k5_gate_finish (device)"))
        .tap(
            Tap::new(tap::BETA, Layout::new().dim("tok", 0).dim("h", nh)).what("k5_beta (device)"),
        );
    for (sec, (name, input)) in ["conv.q", "conv.k", "conv.v"]
        .into_iter()
        .zip(qkv)
        .enumerate()
    {
        s.push(
            Tap::new(name, inner.clone())
                .after(&[input])
                .what("conv1d (device kda_conv / CPU tail conv1d_prefill)")
                .within(tap::CONV, sec * di, di, 3 * di),
        );
    }
    s.tap(
        Tap::new(tap::O, inner.clone())
            .after(&["qkv.q", "qkv.k", "qkv.v", "g", "beta", "state.in"])
            .what("core: kda_step (decode) / kda_prefill (64-chunks) / CPU kda_chunk (tail)"),
    )
    .tap(Tap::new(tap::G2, inner.clone()).what("k5_gemv g_a, g_b (device)"))
    .tap(
        Tap::new(tap::Y_NORM, inner)
            .after(&["o", "g2"])
            .what("k5_gated_norm (device)"),
    )
    .tap(
        Tap::new(tap::OUT, Layout::new().dim("tok", 0).dim("c", hidden))
            .after(&["y_norm"])
            .what("k5_gemv attn_output (device)"),
    )
    .tap(
        Tap::new("state", state)
            .after(&["o"])
            .what("core state update (device st / tail seed)"),
    )
    .tap(
        Tap::new("conv_state", conv.clone())
            .after(&qkv)
            .what("conv commit (device h0) vs host shift"),
    )
    .tap(
        Tap::new("conv_host", conv)
            .after(&qkv)
            .what("host conv copy (decode side effect / tail)"),
    )
}

/// One shadow per KDA layer: the CPU's copy of its conv + recurrent state.
pub struct Shadow {
    pub v: Verifier,
    states: Vec<Option<(ConvState, KdaState)>>,
}

impl Shadow {
    pub fn from_env(n_layer: usize) -> Option<Self> {
        let v = Verifier::from_env("kda", Some(LEGACY))?
            .with_tag("kdadbg")
            .with_unit("l");
        let c = &v.cfg;
        v.log(format_args!(
            "ON: layers {:?} windows {:?} tol {:.1e} resync {} poison {} dir {:?}",
            c.units, c.windows, c.tol, c.resync, c.poison, c.dir
        ));
        Some(Self {
            v,
            states: (0..n_layer).map(|_| None).collect(),
        })
    }

    /// Whether layer `l`'s next window is checked.
    pub fn wants(&self, l: usize) -> bool {
        self.v.wants(l)
    }

    /// The device-side probe for a checked window (poisons per the config).
    pub fn probe(&self) -> Probe {
        Probe::new(self.v.cfg.poison)
    }

    /// Seed the shadow from the engine's host state before the layer's first checked
    /// window (later windows resync from the device, or carry their own state).
    pub fn prime(&mut self, l: usize, conv: &ConvState, state: &KdaState) {
        if self.states[l].is_none() {
            self.states[l] = Some((conv.clone(), state.clone()));
        }
    }

    /// Compare one window. `gpu` are the device taps, `host_conv` the engine's host conv
    /// copy after the window, `dev_conv`/`dev_state` the device's state read back after.
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        &mut self,
        l: usize,
        x: &[f32],
        t: usize,
        w: &KdaLayerW,
        kcfg: &KdaConfig,
        mut gpu: Taps,
        host_conv: &ConvState,
        dev_conv: &ConvState,
        dev_state: &KdaState,
        state_in_dev: Option<&KdaState>,
    ) -> Result<()> {
        let win = self.v.next_window(l);
        let Some((sc, ss)) = self.states[l].as_mut() else {
            self.v.log(format_args!(
                "w{win} l{l}: no shadow state (prime not called) — skipped"
            ));
            return Ok(());
        };
        let mut cpu = Taps::new();
        if let Some(din) = state_in_dev {
            gpu.put("state.in", din.0.clone());
            cpu.put("state.in", ss.0.clone());
        }
        let _ = super::kda::kda_layer_taps(x, w, kcfg, sc, ss, 1, t, Some(&mut cpu))?;
        // state after: device vs shadow; conv: device h0 vs shadow vs the host copy
        gpu.put("state", dev_state.0.clone());
        cpu.put("state", ss.0.clone());
        gpu.put("conv_state", dev_conv.0.clone());
        cpu.put("conv_state", sc.0.clone());
        gpu.put("conv_host", host_conv.0.clone());
        cpu.put("conv_host", sc.0.clone());
        let xs = tang_verify::stat(x, x, 0.0);
        self.v.log(format_args!(
            "w{win} l{l} t{t} x |x| {:.4e} nonfinite {} (same input both paths)",
            xs.norm_cpu, xs.nonfinite_cpu
        ));
        gpu.put("x", x.to_vec());
        self.v
            .check(l, &format!("t{t}"), &spec(kcfg, w.hidden), &gpu, &cpu);
        if self.v.cfg.resync {
            // the next window tests one step from the device's own state
            *sc = dev_conv.clone();
            *ss = dev_state.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_sections_and_locate() {
        let kcfg = KdaConfig {
            heads: 2,
            head_dim: 4,
            d_conv: 4,
            gate_lower_bound: None,
            eps: 1e-6,
        };
        let s = spec(&kcfg, 8);
        // t=2, di=8: [q k v] per token
        let mut taps = Taps::new();
        taps.put(tap::QKV, (0..48).map(|i| i as f32).collect());
        let k = s.get("qkv.k").unwrap().resolve(&taps).unwrap();
        assert_eq!(&k[..2], &[8.0, 9.0]);
        assert_eq!(k[8], 32.0);
        assert_eq!(s.get("o").unwrap().layout.locate(13), "tok1 h1 d1");
        assert_eq!(
            s.get("state").unwrap().layout.locate(16 + 4 * 2 + 3),
            "h1 key2 val3"
        );
        assert_eq!(
            s.get("conv_state")
                .unwrap()
                .layout
                .locate(8 * 3 + 3 * 2 + 1),
            "k h0 d2 tap1"
        );
    }
}
