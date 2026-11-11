//! Shadow check for the GPU FFN / MoE path (`TANG_GLM5_MOE_DEBUG=1`), the
//! `TANG_GLM5_KDA_DEBUG` pattern ([`super::kda_debug`]) applied to the expert path.
//!
//! Every GPU FFN call of a checked layer is re-run on the CPU from the SAME `x` the GPU got:
//! - MoE layers: the GPU's routing (fast path only: ids / weights read back) against the CPU
//!   `moe::route` — per token, same expert SET and the weights' max error; then the experts
//!   are re-run on the CPU with the GPU's OWN routing (`moe::experts` + `moe::shared_expert`),
//!   so a near-tie in the router can't masquerade as an expert-kernel bug, and the full
//!   `[t][hidden]` output is compared. Without a GPU routing (naive path) the CPU runs the
//!   whole `moe_layer`.
//! - dense layers: the engine's CPU dense FFN.
//! - with poisoning (default on), the fast path NaN-fills h and y before its launches, so a
//!   region no kernel wrote is NaN in the comparison, not a plausible 0.0.
//!
//! Env (all optional):
//! - `TANG_GLM5_MOE_DEBUG=1`            enable
//! - `TANG_GLM5_MOE_DEBUG_LAYERS=0,3`   only these layers (default: all GPU FFN layers)
//! - `TANG_GLM5_MOE_DEBUG_WINDOWS=4`    only the first N calls per layer (default: all)
//! - `TANG_GLM5_MOE_DEBUG_TOL=2e-2`     relative-L2 bar for y (int8 activations: ~1e-2)
//! - `TANG_GLM5_MOE_DEBUG_POISON=0`     don't NaN-fill (read by moe::gpu::Kernels)
//!
//! Output lines start with `moedbg`.
use super::moe::{self, MoeParams};
use anyhow::Result;
use tang_verify::{stat, Stat};

pub struct Config {
    pub layers: Option<Vec<usize>>,
    pub windows: Option<usize>,
    pub tol: f64,
}

pub struct Shadow {
    pub cfg: Config,
    calls: Vec<usize>,
    checked: usize,
    failed: usize,
    route_bad: usize,
    first: Option<String>,
    worst: Vec<(f64, String)>,
}

impl Shadow {
    pub fn from_env(n_layer: usize) -> Option<Self> {
        if !std::env::var("TANG_GLM5_MOE_DEBUG").is_ok_and(|v| v == "1") {
            return None;
        }
        let cfg = Config {
            layers: std::env::var("TANG_GLM5_MOE_DEBUG_LAYERS")
                .ok()
                .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect()),
            windows: std::env::var("TANG_GLM5_MOE_DEBUG_WINDOWS")
                .ok()
                .and_then(|s| s.parse().ok()),
            tol: std::env::var("TANG_GLM5_MOE_DEBUG_TOL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2e-2),
        };
        eprintln!(
            "moedbg ON: layers {:?} windows {:?} tol {:.1e}",
            cfg.layers, cfg.windows, cfg.tol
        );
        Some(Self {
            cfg,
            calls: vec![0; n_layer],
            checked: 0,
            failed: 0,
            route_bad: 0,
            first: None,
            worst: Vec::new(),
        })
    }

    pub fn wants(&self, l: usize) -> bool {
        self.cfg.layers.as_ref().is_none_or(|v| v.contains(&l))
            && self.cfg.windows.is_none_or(|n| self.calls[l] < n)
    }

    fn line(&self, w: usize, l: usize, t: usize, name: &str, s: &Stat) -> String {
        format!(
            "moedbg w{w} l{l} t{t} {name:<8} {} rel {:.3e} max {:.3e} @tok{} c{} |gpu| {:.4e} |cpu| {:.4e} cos {:.7} nonfinite {}/{} bad {}/{}",
            if s.bad(self.cfg.tol) { "BAD" } else { "ok " },
            s.rel_l2,
            s.max_abs,
            s.max_idx / (s.n / t.max(1)).max(1),
            s.max_idx % (s.n / t.max(1)).max(1),
            s.norm_gpu,
            s.norm_cpu,
            s.cos,
            s.nonfinite_gpu,
            s.nonfinite_cpu,
            s.bad_elems,
            s.n
        )
    }

    /// Compare one GPU FFN call. `cpu_dense` computes the engine's CPU dense FFN (dense
    /// layers); `gpu_route` is the fast path's `[t][k]` routing, if it has one.
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        &mut self,
        l: usize,
        p: &MoeParams,
        x: &[f32],
        t: usize,
        y_gpu: &[f32],
        gpu_route: Option<(Vec<u32>, Vec<f32>)>,
        cpu_dense: &dyn Fn() -> Result<Vec<f32>>,
    ) -> Result<()> {
        let w = self.calls[l];
        self.calls[l] += 1;
        self.checked += 1;
        let tol = self.cfg.tol;
        let t0 = std::time::Instant::now();
        let mut bad = false;
        let y_cpu = if p.is_dense() {
            cpu_dense()?
        } else {
            match gpu_route {
                Some((gi, gw)) => {
                    let (ci, cw) = moe::route(p, x, t)?;
                    let k = p.top_k;
                    let (mut set_bad, mut werr) = (0usize, 0f32);
                    for tt in 0..t {
                        let mut a: Vec<(u32, f32)> = gi[tt * k..(tt + 1) * k]
                            .iter()
                            .copied()
                            .zip(gw[tt * k..(tt + 1) * k].iter().copied())
                            .collect();
                        let mut b: Vec<(u32, f32)> = ci[tt * k..(tt + 1) * k]
                            .iter()
                            .copied()
                            .zip(cw[tt * k..(tt + 1) * k].iter().copied())
                            .collect();
                        a.sort_by_key(|v| v.0);
                        b.sort_by_key(|v| v.0);
                        if a.iter().map(|v| v.0).ne(b.iter().map(|v| v.0)) {
                            set_bad += 1;
                            eprintln!(
                                "moedbg w{w} l{l} tok{tt} ROUTE gpu {:?} cpu {:?}",
                                &gi[tt * k..(tt + 1) * k],
                                &ci[tt * k..(tt + 1) * k]
                            );
                        } else {
                            for (u, v) in a.iter().zip(&b) {
                                werr = werr.max((u.1 - v.1).abs());
                            }
                        }
                    }
                    eprintln!(
                        "moedbg w{w} l{l} t{t} route    {} sets differ {set_bad}/{t}, weights max abs err {werr:.3e}",
                        if set_bad == 0 && werr <= 1e-4 { "ok " } else { "BAD" }
                    );
                    if set_bad > 0 || werr > 1e-4 {
                        self.route_bad += 1;
                        bad = true;
                    }
                    // the experts on the GPU's own routing
                    let mut y = moe::experts(p, x, t, &gi, &gw)?;
                    for (o, s) in y.iter_mut().zip(moe::shared_expert(p, x, t)?) {
                        *o += s;
                    }
                    y
                }
                None => moe::moe_layer(p, x, t)?,
            }
        };
        let s = stat(y_gpu, &y_cpu, tol);
        eprintln!(
            "{} (cpu {:.0} ms)",
            self.line(w, l, t, if p.is_dense() { "dense.y" } else { "moe.y" }, &s),
            t0.elapsed().as_secs_f64() * 1e3
        );
        if s.rel_l2.is_finite() {
            self.worst.push((s.rel_l2, format!("w{w} l{l} t{t}")));
        } else {
            self.worst.push((f64::INFINITY, format!("w{w} l{l} t{t}")));
        }
        if s.bad(tol) {
            bad = true;
        }
        if bad {
            self.failed += 1;
            if self.first.is_none() {
                let msg = format!("w{w} l{l} t{t}");
                eprintln!("moedbg FIRST DIVERGENCE {msg}");
                self.first = Some(msg);
            }
        }
        Ok(())
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        self.worst
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "moedbg SUMMARY: {} calls checked, {} failed ({} routing); first: {}",
            self.checked,
            self.failed,
            self.route_bad,
            self.first.as_deref().unwrap_or("none (all within tol)")
        );
        for (r, what) in self.worst.iter().take(10) {
            eprintln!("moedbg SUMMARY worst rel {r:.3e} {what}");
        }
    }
}
