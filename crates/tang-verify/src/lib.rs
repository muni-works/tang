//! tang-verify: every GPU kernel checked against its CPU oracle, at write time.
//!
//! The method (see `docs/kernel-verify.md` for the four GLM-5.3 bugs it was built from):
//!
//! - **shadow**: run the CPU reference on the SAME input the device got, from the device's
//!   OWN state, and compare every intermediate in full ([`Stat`]: rel-L2, max abs with a
//!   decoded location, cosine, exact-zero and NaN counts);
//! - **poison**: fill device outputs with NaN before the kernel runs ([`Probe`]), so an
//!   element no kernel wrote reads NaN, never a plausible 0.0;
//! - **attribute**: name the first divergent kernel ([`attribute`]): provenance search
//!   ("these values belong to a sibling") and unwritten/zero anomalies outrank plain
//!   pipeline order;
//! - **never skip silently**: under `TANG_REQUIRE_CUDA=1` a missing GPU is a failure
//!   ([`skip_or_fail`], [`kernel_test!`]).
//!
//! Switches: `TANG_KERNEL_VERIFY=1` (all kernels) or `=kda,mhc` (these), plus any
//! per-kernel legacy switch an adapter passes ([`Config::from_env`]). Off, nothing here
//! runs: adapters hold an `Option<Verifier>` that is `None`, and hot paths that need a
//! check use [`on`] (one relaxed atomic load).
//!
//! A kernel author's three pieces: a [`Spec`] (taps in pipeline order, their
//! [`Layout`]s, inputs and writers), the device side filling a [`Probe`] (poisoned
//! outputs, named read-backs), and the CPU reference filling a [`Taps`]. Then
//! [`Verifier::check`] in an engine, or [`Cases`] in a test.
pub mod attribute;
mod layout;
mod spec;
mod stat;
pub mod testing;
mod verifier;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub mod device;

pub use attribute::{attribute, Cause, Row, Verdict};
pub use layout::{Dim, Layout};
pub use spec::{tap, Region, Spec, Tap, Taps};
pub use stat::{stat, Stat};
pub use testing::{require_cuda, skip_or_fail, CaseResult, Cases};
pub use verifier::{fmt_stat, write_f32, Report, Verifier};

use std::path::PathBuf;
use std::sync::OnceLock;

/// The global switch's env var.
pub const ENV: &str = "TANG_KERNEL_VERIFY";

/// Whether `TANG_KERNEL_VERIFY` turns ANY kernel on (cached: read once per process).
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var(ENV).is_ok_and(|v| !matches!(v.as_str(), "" | "0" | "false")))
}

/// Whether `kernel` is selected: `TANG_KERNEL_VERIFY=1|all` or a comma list naming it,
/// or the adapter's legacy switch `{legacy}=1`. Not cached (call once at setup).
pub fn enabled(kernel: &str, legacy: Option<&str>) -> bool {
    if legacy.is_some_and(|p| std::env::var(p).is_ok_and(|v| v == "1")) {
        return true;
    }
    match std::env::var(ENV) {
        Ok(v) => match v.as_str() {
            "" | "0" | "false" => false,
            "1" | "all" | "*" | "true" => true,
            list => list.split(',').any(|k| k.trim() == kernel),
        },
        Err(_) => false,
    }
}

/// A knob: `{legacy}_{name}` first (an adapter's old env var), then
/// `TANG_KERNEL_VERIFY_{name}`.
pub fn knob(legacy: Option<&str>, name: &str) -> Option<String> {
    legacy
        .and_then(|p| std::env::var(format!("{p}_{name}")).ok())
        .or_else(|| std::env::var(format!("{ENV}_{name}")).ok())
}

fn flag(v: Option<String>, default: bool) -> bool {
    v.map_or(default, |v| !(v == "0" || v.eq_ignore_ascii_case("false")))
}

/// What a [`Verifier`] checks and how strictly.
#[derive(Clone, Debug)]
pub struct Config {
    /// Only these units (e.g. layer indices); `None` = all. Knob `LAYERS`.
    pub units: Option<Vec<usize>>,
    /// Only the first N windows per unit. Knob `WINDOWS`.
    pub windows: Option<usize>,
    /// Relative-L2 bar per tensor. Knob `TOL` (default 1e-3).
    pub tol: f64,
    /// The shadow restarts from the device's state each window (tests one step, not
    /// drift). Knob `RESYNC` (default on). Stateful adapters honour it.
    pub resync: bool,
    /// NaN-fill device outputs before each window. Knob `POISON` (default on).
    pub poison: bool,
    /// Raw f32 dumps of the first window and the first `dumps` failing ones. Knob `DIR`.
    pub dir: Option<PathBuf>,
    /// Knob `DUMPS` (default 3).
    pub dumps: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            units: None,
            windows: None,
            tol: 1e-3,
            resync: true,
            poison: true,
            dir: None,
            dumps: 3,
        }
    }
}

impl Config {
    /// `None` unless [`enabled`]. Knobs per [`knob`]; a `DIR` from the global knob gets a
    /// per-kernel subdirectory so kernels can't overwrite each other's dumps.
    pub fn from_env(kernel: &str, legacy: Option<&str>) -> Option<Self> {
        if !enabled(kernel, legacy) {
            return None;
        }
        let d = Self::default();
        fn parse<T: std::str::FromStr>(legacy: Option<&str>, n: &str) -> Option<T> {
            knob(legacy, n).and_then(|s| s.trim().parse().ok())
        }
        let dir = legacy
            .and_then(|p| std::env::var(format!("{p}_DIR")).ok().map(PathBuf::from))
            .or_else(|| {
                std::env::var(format!("{ENV}_DIR"))
                    .ok()
                    .map(|s| PathBuf::from(s).join(kernel))
            });
        Some(Self {
            units: knob(legacy, "LAYERS")
                .or_else(|| knob(legacy, "UNITS"))
                .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect()),
            windows: parse(legacy, "WINDOWS"),
            tol: parse(legacy, "TOL").unwrap_or(d.tol),
            resync: flag(knob(legacy, "RESYNC"), d.resync),
            poison: flag(knob(legacy, "POISON"), d.poison),
            dir,
            dumps: parse(legacy, "DUMPS").unwrap_or(d.dumps),
        })
    }
}

/// The device side of one checked run: whether to poison, and the named read-backs.
/// The engine hands a `Probe` to the kernel wrapper for a checked window and takes the
/// [`Taps`] back; with `cuda`, [`Probe::poison`] / [`Probe::read`] / [`Probe::output`]
/// work on `tang_moe::gpu::DevBuf`s.
#[derive(Clone, Debug, Default)]
pub struct Probe {
    pub poison: bool,
    pub taps: Taps,
}

impl Probe {
    pub fn new(poison: bool) -> Self {
        Self {
            poison,
            taps: Taps::new(),
        }
    }

    /// Record a host-side value (a tap the device path computed on the host).
    pub fn put(&mut self, name: impl Into<String>, v: Vec<f32>) {
        self.taps.put(name, v);
    }
}
