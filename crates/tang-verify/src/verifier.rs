//! The session: compare windows, attribute, dump, count, summarise.

use crate::attribute::{attribute, Row, Verdict};
use crate::{stat, Config, Layout, Spec, Stat, Taps};
use std::fmt::Write as _;

/// The result of one checked window.
#[derive(Clone, Debug)]
pub struct Report {
    pub window: usize,
    pub rows: Vec<Row>,
    /// `Some` iff a tap failed.
    pub verdict: Option<Verdict>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.verdict.is_none()
    }
}

/// One kernel family's verification session. Construct it only when verification is on
/// ([`Verifier::from_env`] returns `None` otherwise), so an off run never touches it.
/// Prints `{tag} SUMMARY: N windows checked, M failed; first: …` on drop.
pub struct Verifier {
    pub cfg: Config,
    pub kernel: String,
    /// Line prefix (`kdadbg` for the KDA port; default `verify[{kernel}]`).
    pub tag: String,
    /// Unit prefix in lines (`l` = layer): `w3 l12 …`.
    pub unit: &'static str,
    /// Print every row (`true`, default) or only failing windows' rows.
    pub verbose: bool,
    windows: Vec<usize>,
    checked: usize,
    failed: usize,
    dumped: usize,
    first: Option<String>,
    worst: Vec<(f64, String)>,
}

impl Verifier {
    pub fn new(kernel: &str, cfg: Config) -> Self {
        if let Some(d) = &cfg.dir {
            let _ = std::fs::create_dir_all(d);
        }
        Self {
            cfg,
            kernel: kernel.into(),
            tag: format!("verify[{kernel}]"),
            unit: "u",
            verbose: true,
            windows: Vec::new(),
            checked: 0,
            failed: 0,
            dumped: 0,
            first: None,
            worst: Vec::new(),
        }
    }

    /// On when `TANG_KERNEL_VERIFY` names `kernel` (or is `1`), or `{legacy}=1`.
    pub fn from_env(kernel: &str, legacy: Option<&str>) -> Option<Self> {
        Config::from_env(kernel, legacy).map(|c| Self::new(kernel, c))
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tag = tag.into();
        self
    }

    pub fn with_unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }

    pub fn checked(&self) -> usize {
        self.checked
    }

    pub fn failed(&self) -> usize {
        self.failed
    }

    pub fn first(&self) -> Option<&str> {
        self.first.as_deref()
    }

    /// Print a tagged line.
    pub fn log(&self, msg: impl std::fmt::Display) {
        eprintln!("{} {msg}", self.tag);
    }

    /// Whether unit `u`'s next window is checked (the `LAYERS` / `WINDOWS` filters).
    pub fn wants(&self, u: usize) -> bool {
        self.cfg.units.as_ref().is_none_or(|v| v.contains(&u))
            && self
                .cfg
                .windows
                .is_none_or(|n| self.windows.get(u).copied().unwrap_or(0) < n)
    }

    /// The index the next window of unit `u` will get.
    pub fn next_window(&self, u: usize) -> usize {
        self.windows.get(u).copied().unwrap_or(0)
    }

    /// Compare every tap of `spec` present on both sides, print, attribute, dump.
    /// `ctx` is extra context for the lines (e.g. `t20`).
    pub fn check(&mut self, u: usize, ctx: &str, spec: &Spec, gpu: &Taps, cpu: &Taps) -> Report {
        if self.windows.len() <= u {
            self.windows.resize(u + 1, 0);
        }
        let win = self.windows[u];
        self.windows[u] += 1;
        self.checked += 1;
        let tol = self.cfg.tol;
        let rows: Vec<Row> = spec
            .taps
            .iter()
            .filter_map(|t| {
                let (g, c) = (t.resolve(gpu)?, t.resolve(cpu)?);
                let s = stat(&g, &c, tol);
                Some(Row {
                    name: t.name,
                    bad: s.bad(tol),
                    stat: s,
                    layout: t.layout.clone(),
                })
            })
            .collect();
        let any_bad = rows.iter().any(|r| r.bad);
        let ctx = if ctx.is_empty() {
            String::new()
        } else {
            format!(" {ctx}")
        };
        let head = format!("w{win} {}{u}{ctx}", self.unit);
        if self.verbose || any_bad {
            for r in &rows {
                self.log(format_args!("{head} {}", fmt_row(r, tol)));
            }
        }
        for r in &rows {
            if r.stat.rel_l2.is_finite() {
                self.worst
                    .push((r.stat.rel_l2, format!("w{win} {}{u} {}", self.unit, r.name)));
            }
        }
        let verdict = any_bad.then(|| attribute(spec, &rows, gpu, cpu, tol, self.cfg.poison));
        if let Some(v) = &verdict {
            self.failed += 1;
            let msg = format!("{head}: {}", v.message());
            self.log(format_args!("VERDICT {msg}"));
            if self.failed <= self.cfg.dumps.max(1) {
                for e in &v.evidence {
                    self.log(format_args!("  evidence {head}: {e}"));
                }
            }
            if self.first.is_none() {
                self.log(format_args!("FIRST DIVERGENCE {msg}"));
                self.first = Some(msg);
            }
        }
        let dump = self.checked == 1 || (any_bad && self.dumped < self.cfg.dumps);
        if let (true, Some(dir)) = (dump, self.cfg.dir.clone()) {
            if any_bad {
                self.dumped += 1;
            }
            for (side, taps) in [("gpu", gpu), ("cpu", cpu)] {
                for (name, v) in taps.iter() {
                    write_f32(
                        &dir.join(format!("w{win}_{}{u}_{name}.{side}.f32", self.unit)),
                        v,
                    );
                }
            }
            self.log(format_args!("{head}: tensors written to {}", dir.display()));
        }
        Report {
            window: win,
            rows,
            verdict,
        }
    }

    pub fn summary(&self) -> Vec<String> {
        let mut w = self.worst.clone();
        w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = vec![format!(
            "SUMMARY: {} windows checked, {} failed; first: {}",
            self.checked,
            self.failed,
            self.first.as_deref().unwrap_or("none (all within tol)")
        )];
        out.extend(
            w.iter()
                .take(10)
                .map(|(r, what)| format!("SUMMARY worst rel {r:.3e} {what}")),
        );
        out
    }
}

impl Drop for Verifier {
    fn drop(&mut self) {
        for l in self.summary() {
            self.log(l);
        }
    }
}

/// The row format (unchanged from the KDA shadow check, so old logs diff cleanly).
pub fn fmt_row(r: &Row, tol: f64) -> String {
    fmt_stat(r.name, &r.stat, &r.layout, tol)
}

pub fn fmt_stat(name: &str, s: &Stat, l: &Layout, tol: f64) -> String {
    let mut line = String::new();
    let _ = write!(
        line,
        "{:<11} {} rel {:.3e} max {:.3e} @{} |gpu| {:.4e} |cpu| {:.4e} cos {:.6} zeros {}/{} nonfinite {}/{} bad {}/{}",
        name,
        if s.bad(tol) { "BAD" } else { "ok " },
        s.rel_l2,
        s.max_abs,
        l.locate(s.max_idx),
        s.norm_gpu,
        s.norm_cpu,
        s.cos,
        s.zeros_gpu,
        s.zeros_cpu,
        s.nonfinite_gpu,
        s.nonfinite_cpu,
        s.bad_elems,
        s.n,
    );
    if s.len_mismatch {
        line.push_str(" LEN-MISMATCH");
    }
    line
}

/// Raw little-endian f32.
pub fn write_f32(p: &std::path::Path, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    if let Err(e) = std::fs::write(p, b) {
        eprintln!("tang-verify: write {}: {e}", p.display());
    }
}
