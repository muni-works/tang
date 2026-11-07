//! Adaptive-damping Levenberg-Marquardt over a [`LeastSquares`] system.
//!
//! Moved from vcad's sketch-constraint solver
//! (`vcad-kernel-constraints/src/solver.rs`), where it drives both the
//! constraint solver and differentiable design. Unlike
//! [`crate::LevenbergMarquardt`], which takes dense residual and Jacobian
//! closures, the driver here only sees the normal-equation pieces
//! (`JᵀJ`, `Jᵀr`), so a system can assemble them from a sparse Jacobian,
//! and it reports why it stopped.
//!
//! Each iteration solves `(JᵀJ + λI) δ = −Jᵀr` by LU, projects `x + δ` into
//! the feasible set, and accepts the step only if the squared residual norm
//! drops. Accepting shrinks λ (`lambda_decrease`, floored at `min_lambda`);
//! rejecting or hitting a singular system grows it (`lambda_increase`), and λ
//! passing `max_lambda` ends the run.


use alloc::vec::Vec;
use tang_la::{DMat, DVec};

/// A least-squares system [`levenberg_marquardt`] can minimize.
///
/// Implementors expose the normal-equation pieces (`JᵀJ`, `Jᵀr`) and the
/// squared residual norm; the driver never touches residuals directly.
/// `project` is the box-bounds hook; the default no-op is unconstrained.
pub trait LeastSquares {
    /// Number of free parameters (the dimension of the step vector).
    fn num_params(&self) -> usize;

    /// Compute `(JᵀJ, Jᵀr)` at `params`: `JᵀJ` is `n×n`, `Jᵀr` has length `n`.
    fn eval_jtj_jtr(&self, params: &[f64]) -> (DMat<f64>, Vec<f64>);

    /// Sum of squared residuals at `params`.
    fn residual_norm_squared(&self, params: &[f64]) -> f64;

    /// Project a candidate parameter vector into the feasible set (e.g. box
    /// bounds). Default is identity.
    fn project(&self, params: &mut [f64]) {
        let _ = params;
    }
}

/// Configuration for [`levenberg_marquardt`].
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// Maximum number of iterations before stopping.
    pub max_iterations: usize,
    /// Convergence tolerance on the residual norm (not squared).
    pub tolerance: f64,
    /// Initial damping factor λ.
    pub initial_lambda: f64,
    /// Factor applied to λ when a step is rejected.
    pub lambda_increase: f64,
    /// Factor applied to λ when a step is accepted.
    pub lambda_decrease: f64,
    /// Floor on λ.
    pub min_lambda: f64,
    /// λ above this ends the run.
    pub max_lambda: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
            initial_lambda: 1e-3,
            lambda_increase: 10.0,
            lambda_decrease: 0.1,
            min_lambda: 1e-12,
            max_lambda: 1e12,
        }
    }
}

/// Why [`levenberg_marquardt`] stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LmStatus {
    /// Residual norm fell below `tolerance`.
    Converged,
    /// Reached `max_iterations`.
    MaxIterations,
    /// λ passed `max_lambda` after rejected steps.
    LambdaOverflow,
    /// The residual set is empty. Never produced by the driver; callers with
    /// an empty problem (e.g. a sketch with no constraints) report it.
    NoConstraints,
    /// No parameters to optimize.
    NoParameters,
    /// λ passed `max_lambda` while the damped system stayed singular.
    SingularMatrix,
}

/// Result of [`levenberg_marquardt`].
#[derive(Debug, Clone)]
pub struct LmResult {
    /// Final parameter values.
    pub parameters: Vec<f64>,
    /// Final residual norm (not squared).
    pub residual_norm: f64,
    /// Iterations performed.
    pub iterations: usize,
    /// Whether the residual norm fell below tolerance.
    pub converged: bool,
    /// Why the run stopped.
    pub status: LmStatus,
}

/// Run Levenberg-Marquardt on `system`, mutating `params` in place.
pub fn levenberg_marquardt<Sys: LeastSquares + ?Sized>(
    system: &Sys,
    params: &mut [f64],
    config: &LmConfig,
) -> LmResult {
    let done = |params: &[f64], norm_sq: f64, iterations, converged, status| LmResult {
        parameters: params.to_vec(),
        residual_norm: libm_sqrt(norm_sq),
        iterations,
        converged,
        status,
    };
    if params.is_empty() {
        let n = system.residual_norm_squared(params);
        return done(params, n, 0, false, LmStatus::NoParameters);
    }

    let mut lambda = config.initial_lambda;
    let mut current_norm_sq = system.residual_norm_squared(params);

    for iteration in 0..config.max_iterations {
        if libm_sqrt(current_norm_sq) < config.tolerance {
            return done(params, current_norm_sq, iteration, true, LmStatus::Converged);
        }

        let (jtj, jtr_vec) = system.eval_jtj_jtr(params);
        let jtr = DVec::from_vec(jtr_vec);

        match try_step(system, params, &jtj, &jtr, lambda) {
            Step::Accepted {
                new_params,
                new_norm_sq,
            } => {
                params.copy_from_slice(&new_params);
                current_norm_sq = new_norm_sq;
                lambda = (lambda * config.lambda_decrease).max(config.min_lambda);
            }
            Step::Rejected => {
                lambda *= config.lambda_increase;
                if lambda > config.max_lambda {
                    let s = LmStatus::LambdaOverflow;
                    return done(params, current_norm_sq, iteration, false, s);
                }
            }
            Step::Singular => {
                lambda *= config.lambda_increase;
                if lambda > config.max_lambda {
                    let s = LmStatus::SingularMatrix;
                    return done(params, current_norm_sq, iteration, false, s);
                }
            }
        }
    }

    let n = config.max_iterations;
    done(params, current_norm_sq, n, false, LmStatus::MaxIterations)
}

enum Step {
    Accepted { new_params: Vec<f64>, new_norm_sq: f64 },
    Rejected,
    Singular,
}

fn try_step<Sys: LeastSquares + ?Sized>(
    system: &Sys,
    params: &[f64],
    jtj: &DMat<f64>,
    jtr: &DVec<f64>,
    lambda: f64,
) -> Step {
    let n = jtj.nrows();
    let mut a = jtj.clone();
    for i in 0..n {
        a[(i, i)] += lambda;
    }
    let Some(delta) = a.lu().solve(&(-jtr)) else {
        return Step::Singular;
    };
    let mut new_params: Vec<f64> = params.iter().enumerate().map(|(i, &p)| p + delta[i]).collect();
    system.project(&mut new_params);

    let new_norm_sq = system.residual_norm_squared(&new_params);
    let old_norm_sq = system.residual_norm_squared(params);
    if new_norm_sq < old_norm_sq {
        Step::Accepted {
            new_params,
            new_norm_sq,
        }
    } else {
        Step::Rejected
    }
}

fn libm_sqrt(x: f64) -> f64 {
    <f64 as tang::Scalar>::sqrt(x)
}


#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Rosenbrock as residuals r = (1 − x, 10(y − x²)).
    struct Rosen;
    impl LeastSquares for Rosen {
        fn num_params(&self) -> usize {
            2
        }
        fn eval_jtj_jtr(&self, p: &[f64]) -> (DMat<f64>, Vec<f64>) {
            let (x, y) = (p[0], p[1]);
            let r = [1.0 - x, 10.0 * (y - x * x)];
            let j = [[-1.0, 0.0], [-20.0 * x, 10.0]];
            let mut jtj = DMat::zeros(2, 2);
            let mut jtr = vec![0.0; 2];
            for a in 0..2 {
                for b in 0..2 {
                    jtj[(a, b)] = (0..2).map(|k| j[k][a] * j[k][b]).sum();
                }
                jtr[a] = (0..2).map(|k| j[k][a] * r[k]).sum();
            }
            (jtj, jtr)
        }
        fn residual_norm_squared(&self, p: &[f64]) -> f64 {
            let (x, y) = (p[0], p[1]);
            (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
        }
    }

    #[test]
    fn rosenbrock_converges() {
        let mut p = vec![-1.2, 1.0];
        let r = levenberg_marquardt(&Rosen, &mut p, &LmConfig::default());
        assert_eq!(r.status, LmStatus::Converged);
        assert!((p[0] - 1.0).abs() < 1e-8 && (p[1] - 1.0).abs() < 1e-8);
    }

    struct Boxed;
    impl LeastSquares for Boxed {
        fn num_params(&self) -> usize {
            1
        }
        fn eval_jtj_jtr(&self, p: &[f64]) -> (DMat<f64>, Vec<f64>) {
            let mut m = DMat::zeros(1, 1);
            m[(0, 0)] = 1.0;
            (m, vec![p[0] - 5.0])
        }
        fn residual_norm_squared(&self, p: &[f64]) -> f64 {
            (p[0] - 5.0) * (p[0] - 5.0)
        }
        fn project(&self, p: &mut [f64]) {
            p[0] = p[0].min(2.0);
        }
    }

    #[test]
    fn projection_holds_bound() {
        let mut p = vec![0.0];
        let r = levenberg_marquardt(&Boxed, &mut p, &LmConfig::default());
        assert!(!r.converged);
        assert_eq!(p[0], 2.0);
    }
}
