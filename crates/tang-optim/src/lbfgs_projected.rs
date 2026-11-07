//! Projected two-loop L-BFGS with Armijo backtracking.
//!
//! Moved from vcad's differentiable kernel
//! (`vcad-kernel-diff/src/lbfgs.rs`). [`crate::Lbfgs`] is a fixed-learning-
//! rate stepper; [`minimize_lbfgs`] is a full minimizer that owns the line
//! search, the box projection and the curvature memory.
//!
//! - **Box projection.** Iterates are clamped into `bounds`; the Armijo test
//!   measures sufficient decrease along the *projected* displacement
//!   `trial − x`, so a step flattened against a bound is measured honestly.
//! - **Failed evaluations are failed steps.** The objective returns
//!   `Result`. An error at a trial point shrinks the step like a
//!   non-decreasing value would; an error at the starting point is returned.
//!   vcad uses this for topology changes in a frozen-plan seam.
//! - **Guarded memory.** A pair `(s, y)` is stored only when
//!   `s·y > ε‖s‖‖y‖`, and not when the line search had to shrink past a
//!   failed evaluation to reach it (the pair would straddle two smooth
//!   branches). A non-descent direction or a failed line search clears the
//!   memory and falls back to projected steepest descent.

use alloc::vec;
use alloc::vec::Vec;

/// Options for [`minimize_lbfgs`].
#[derive(Debug, Clone)]
pub struct LbfgsOptions {
    /// Maximum accepted iterations.
    pub max_iters: usize,
    /// Trial length of the first (steepest-descent) line search. Once
    /// curvature exists the natural quasi-Newton trial length is 1.
    pub initial_step: f64,
    /// Line-search steps below this length end the search.
    pub min_step: f64,
    /// Gradient ∞-norm below which the run stops.
    pub grad_tol: f64,
    /// Per-parameter inclusive bounds; parameters past the end are unbounded.
    pub bounds: Vec<(f64, f64)>,
    /// Curvature memory depth.
    pub memory: usize,
    /// Armijo sufficient-decrease coefficient.
    pub armijo_c1: f64,
    /// Backtracking contraction per trial.
    pub backtrack: f64,
    /// Relative floor on `s·y` for storing a pair.
    pub curvature_eps: f64,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            max_iters: 50,
            initial_step: 1.0,
            min_step: 1e-8,
            grad_tol: 1e-8,
            bounds: Vec::new(),
            memory: 8,
            armijo_c1: 1e-4,
            backtrack: 0.5,
            curvature_eps: 1e-12,
        }
    }
}

/// Why [`minimize_lbfgs`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbfgsStop {
    /// Gradient ∞-norm fell below `grad_tol`.
    GradientConverged,
    /// No decreasing step above `min_step` even from steepest descent (a
    /// minimum, a bound, or a wall of failed evaluations).
    StepConverged,
    /// Iteration budget exhausted.
    MaxIters,
}

/// One accepted iterate.
#[derive(Debug, Clone)]
pub struct LbfgsIterate {
    /// Parameters.
    pub x: Vec<f64>,
    /// Objective value.
    pub value: f64,
    /// Gradient.
    pub gradient: Vec<f64>,
}

/// Result of [`minimize_lbfgs`].
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Final parameters.
    pub x: Vec<f64>,
    /// Final objective value.
    pub value: f64,
    /// Every accepted iterate, starting point first.
    pub history: Vec<LbfgsIterate>,
    /// Why the run stopped.
    pub stop: LbfgsStop,
}

/// Clamp `x` into `bounds` (entries past `bounds.len()` are unbounded).
pub fn project_box(x: &mut [f64], bounds: &[(f64, f64)]) {
    for (k, t) in x.iter_mut().enumerate() {
        if let Some(&(lo, hi)) = bounds.get(k) {
            *t = t.clamp(lo, hi);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Pair {
    s: Vec<f64>,
    y: Vec<f64>,
    rho: f64,
}

/// `d = −H·g` from the two-loop recursion over `history` (oldest first),
/// scaled by `γ = (s·y)/(y·y)` of the newest pair. Empty history gives `−g`.
fn two_loop_direction(g: &[f64], history: &[Pair]) -> Vec<f64> {
    let mut q = g.to_vec();
    let mut alpha = vec![0.0; history.len()];
    for (i, pair) in history.iter().enumerate().rev() {
        let a = pair.rho * dot(&pair.s, &q);
        alpha[i] = a;
        for (qj, yj) in q.iter_mut().zip(&pair.y) {
            *qj -= a * yj;
        }
    }
    let gamma = match history.last() {
        Some(p) => {
            let yy = dot(&p.y, &p.y);
            if yy > 0.0 {
                dot(&p.s, &p.y) / yy
            } else {
                1.0
            }
        }
        None => 1.0,
    };
    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();
    for (i, pair) in history.iter().enumerate() {
        let beta = pair.rho * dot(&pair.y, &r);
        let coeff = alpha[i] - beta;
        for (ri, si) in r.iter_mut().zip(&pair.s) {
            *ri += coeff * si;
        }
    }
    r.iter().map(|ri| -ri).collect()
}

/// Minimize `f` from `x0` by projected L-BFGS. `f` returns `(value,
/// gradient)` or an error; see the module docs for how errors are treated.
pub fn minimize_lbfgs<E>(
    mut f: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>,
    x0: &[f64],
    options: &LbfgsOptions,
) -> Result<LbfgsResult, E> {
    let mut x = x0.to_vec();
    project_box(&mut x, &options.bounds);

    let (mut value, mut gradient) = f(&x)?;
    let mut history: Vec<Pair> = Vec::with_capacity(options.memory);
    let mut records = vec![LbfgsIterate {
        x: x.clone(),
        value,
        gradient: gradient.clone(),
    }];
    let grad_inf = |g: &[f64]| g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let finish = |x, value, records, stop| LbfgsResult {
        x,
        value,
        history: records,
        stop,
    };

    for _ in 0..options.max_iters {
        if grad_inf(&gradient) < options.grad_tol {
            return Ok(finish(x, value, records, LbfgsStop::GradientConverged));
        }

        let mut direction = two_loop_direction(&gradient, &history);
        if dot(&gradient, &direction) >= 0.0 {
            history.clear();
            direction = gradient.iter().map(|g| -g).collect();
        }

        let mut step = if history.is_empty() {
            options.initial_step
        } else {
            1.0
        };

        let mut hit_failure = false;
        let accepted = loop {
            let mut trial: Vec<f64> = x.iter().zip(&direction).map(|(t, d)| t + step * d).collect();
            project_box(&mut trial, &options.bounds);
            let displacement: Vec<f64> = trial.iter().zip(&x).map(|(a, b)| a - b).collect();
            let moved = displacement.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if moved > 0.0 {
                let expected = options.armijo_c1 * dot(&gradient, &displacement);
                match f(&trial) {
                    Ok((jt, gt)) if jt <= value + expected => {
                        break Some((trial, jt, gt, displacement));
                    }
                    Ok(_) => {}
                    Err(_) => hit_failure = true,
                }
            }
            step *= options.backtrack;
            if step < options.min_step {
                break None;
            }
        };

        let Some((t, j, g, s)) = accepted else {
            if history.is_empty() {
                return Ok(finish(x, value, records, LbfgsStop::StepConverged));
            }
            history.clear();
            continue;
        };
        let y: Vec<f64> = g.iter().zip(&gradient).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        let sn = <f64 as tang::Scalar>::sqrt(dot(&s, &s));
        let yn = <f64 as tang::Scalar>::sqrt(dot(&y, &y));
        if !hit_failure && sy > options.curvature_eps * sn * yn && options.memory > 0 {
            if history.len() == options.memory {
                history.remove(0);
            }
            history.push(Pair { rho: 1.0 / sy, s, y });
        }
        x = t;
        value = j;
        gradient = g;
        records.push(LbfgsIterate {
            x: x.clone(),
            value,
            gradient: gradient.clone(),
        });
    }

    Ok(finish(x, value, records, LbfgsStop::MaxIters))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosen(p: &[f64]) -> Result<(f64, Vec<f64>), ()> {
        let (x, y) = (p[0], p[1]);
        let v = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
        let g = vec![-2.0 * (1.0 - x) - 400.0 * x * (y - x * x), 200.0 * (y - x * x)];
        Ok((v, g))
    }

    #[test]
    fn rosenbrock() {
        let opts = LbfgsOptions {
            max_iters: 200,
            initial_step: 1e-3,
            ..Default::default()
        };
        let r = minimize_lbfgs(rosen, &[-1.2, 1.0], &opts).unwrap();
        assert_eq!(r.stop, LbfgsStop::GradientConverged);
        assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bounded_stops_on_bound() {
        let opts = LbfgsOptions {
            bounds: vec![(-5.0, 0.5), (-5.0, 5.0)],
            max_iters: 200,
            initial_step: 1e-3,
            ..Default::default()
        };
        let r = minimize_lbfgs(rosen, &[-1.2, 1.0], &opts).unwrap();
        assert!((r.x[0] - 0.5).abs() < 1e-9 && (r.x[1] - 0.25).abs() < 1e-4);
    }

    #[test]
    fn failures_shrink_the_step() {
        // f = (x-3)², but evaluations past x > 1 fail: stops at the wall.
        let f = |p: &[f64]| {
            if p[0] > 1.0 {
                Err(())
            } else {
                Ok(((p[0] - 3.0).powi(2), vec![2.0 * (p[0] - 3.0)]))
            }
        };
        let r = minimize_lbfgs(f, &[0.0], &LbfgsOptions::default()).unwrap();
        assert!(r.x[0] <= 1.0 && r.x[0] > 0.9);
        assert!(minimize_lbfgs(f, &[2.0], &LbfgsOptions::default()).is_err());
    }
}
