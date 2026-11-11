//! Kernel tests: shapes × dtypes × poison × compare vs the CPU reference, failing (never
//! skipping) under `TANG_REQUIRE_CUDA=1`.
//!
//! ```ignore
//! tang_verify::kernel_test!(k5_gemv_matches_cpu, {
//!     tang_verify::Cases::new("k5_gemv")
//!         .shapes(&[(1, 64, 32), (20, 256, 96)])
//!         .dtypes(&["f32", "q8_0"])
//!         .run(|&(t, rows, cols), &dt, p| {
//!             let y = p.output(t * rows)?;           // NaN-poisoned
//!             /* upload, launch into y */
//!             p.read("y", &y, t * rows)?;
//!             let mut cpu = tang_verify::Taps::new();
//!             cpu.put("y", reference(t, rows, cols, dt));
//!             let lay = tang_verify::Layout::new().dim("tok", 0).dim("r", rows);
//!             Ok((tang_verify::Spec::new("k5_gemv").tap(tang_verify::Tap::new("y", lay)), cpu))
//!         })
//! });
//! ```
use crate::{Config, Probe, Spec, Taps, Verifier};
use std::fmt::Debug;

/// `TANG_REQUIRE_CUDA=1`: GPU tests must run (CI on a GPU box, pods).
pub fn require_cuda() -> bool {
    std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1")
}

/// A GPU test that can't run: a FAILURE under `TANG_REQUIRE_CUDA=1`, else a loud banner.
/// A silent pass looks green in CI; it is how the GLM-5.3 GPU bugs survived a review.
pub fn skip_or_fail(reason: &str) {
    let msg = format!("SKIP: {reason} (set TANG_REQUIRE_CUDA=1 to make this a failure)");
    if require_cuda() {
        panic!("{msg} — but TANG_REQUIRE_CUDA=1");
    }
    eprintln!("==================== {msg} ====================");
}

pub type CaseResult = std::result::Result<(Spec, Taps), Box<dyn std::error::Error + Send + Sync>>;

/// A test's case grid.
pub struct Cases<S, D> {
    pub kernel: String,
    pub shapes: Vec<S>,
    pub dtypes: Vec<D>,
    pub cfg: Config,
}

impl<S: Debug + Clone> Cases<S, ()> {
    /// `kernel`'s cases, poison on, tol 1e-3, dumps to `TANG_KERNEL_VERIFY_DIR/{kernel}`
    /// when set.
    pub fn new(kernel: &str) -> Self {
        let dir = std::env::var(format!("{}_DIR", crate::ENV))
            .ok()
            .map(|d| std::path::PathBuf::from(d).join(kernel));
        Self {
            kernel: kernel.into(),
            shapes: Vec::new(),
            dtypes: vec![()],
            cfg: Config {
                dir,
                ..Config::default()
            },
        }
    }
}

impl<S: Debug + Clone, D: Debug + Clone> Cases<S, D> {
    pub fn shapes(mut self, s: &[S]) -> Self {
        self.shapes = s.to_vec();
        self
    }

    pub fn dtypes<D2: Debug + Clone>(self, d: &[D2]) -> Cases<S, D2> {
        Cases {
            kernel: self.kernel,
            shapes: self.shapes,
            dtypes: d.to_vec(),
            cfg: self.cfg,
        }
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.cfg.tol = tol;
        self
    }

    /// Run every shape × dtype with a poisoned [`Probe`]; `f` launches the kernel(s),
    /// reads the outputs into the probe, and returns the [`Spec`] + the CPU reference's
    /// [`Taps`]. Returns the failures (empty = all clean); errors count as failures.
    /// Doesn't need a GPU itself: [`Self::run`] is the GPU entry point.
    pub fn check(&self, mut f: impl FnMut(&S, &D, &mut Probe) -> CaseResult) -> Vec<String> {
        assert!(!self.shapes.is_empty(), "{}: no shapes", self.kernel);
        let mut v = Verifier::new(&self.kernel, self.cfg.clone()).with_unit("case");
        v.verbose = false;
        let mut fails = Vec::new();
        let mut case = 0;
        for s in &self.shapes {
            for d in &self.dtypes {
                let mut p = Probe::new(self.cfg.poison);
                let what = format!("shape {s:?} dtype {d:?}");
                match f(s, d, &mut p) {
                    Ok((spec, cpu)) => {
                        v.log(format_args!("case{case}: {what}"));
                        let r = v.check(case, &what, &spec, &p.taps, &cpu);
                        if let Some(ver) = r.verdict {
                            fails.push(format!("{what}: {}", ver.message()));
                        }
                    }
                    Err(e) => fails.push(format!("{what}: error: {e}")),
                }
                case += 1;
            }
        }
        fails
    }

    /// [`Self::check`] on the GPU: binds device 0 (or [`skip_or_fail`]s), runs the grid,
    /// panics listing every failing case.
    #[cfg(feature = "cuda")]
    pub fn run(self, f: impl FnMut(&S, &D, &mut Probe) -> CaseResult) {
        let Some(_g) = crate::device::gpu_or_skip(&self.kernel) else {
            return;
        };
        let n = self.shapes.len() * self.dtypes.len();
        let fails = self.check(f);
        assert!(
            fails.is_empty(),
            "{}: {}/{n} cases failed:\n  {}",
            self.kernel,
            fails.len(),
            fails.join("\n  ")
        );
    }
}

/// A GPU kernel test: `kernel_test!(name, { Cases::new(..)...run(..) })`. With the
/// INVOKING crate's `cuda` feature the body runs; without it the test
/// [`skip_or_fail`]s (so `TANG_REQUIRE_CUDA=1` on a non-CUDA build fails, not passes).
#[macro_export]
macro_rules! kernel_test {
    ($(#[$m:meta])* $name:ident, $body:block) => {
        $(#[$m])*
        #[test]
        fn $name() {
            #[cfg(feature = "cuda")]
            $body
            #[cfg(not(feature = "cuda"))]
            $crate::skip_or_fail(concat!(stringify!($name), ": built without the `cuda` feature"));
        }
    };
}
