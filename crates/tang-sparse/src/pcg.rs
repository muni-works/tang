//! Preconditioned conjugate gradients for symmetric positive-definite
//! systems, with Jacobi and IC(0) preconditioners.
//!
//! [`pcg`] is operator-generic: the matrix is any `apply(x, y)` computing
//! `y = A·x`, so matrix-free operators (an FEA stiffness applied element by
//! element, a voxel conduction stencil) and [`CsrMatrix`] share one solver.
//! The iteration is the textbook one, and its convergence test is the
//! relative residual `‖r‖/‖b‖ ≤ tol`. It never fails: the
//! [`PcgSolution::status`] says whether it converged, ran out of
//! iterations, or broke down (`dᵀAd ≤ 0`, a non-SPD operator or a
//! numerically exhausted direction), and callers decide what that means.
//!
//! The tensor-IR form of the same algorithm (SpMV as a gather
//! comprehension) lives in `tang-compile`'s `sparse` module; this is the
//! native executor its `LinearSolve` operator node runs forward.

use alloc::vec;
use alloc::vec::Vec;

use crate::CsrMatrix;

/// Stopping rule for [`pcg`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgOptions {
    /// Relative residual `‖r‖/‖b‖` at or below which the solve stops.
    pub tol: f64,
    /// Iteration budget.
    pub max_iters: usize,
}

impl Default for PcgOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_iters: 10_000,
        }
    }
}

/// How a [`pcg`] run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcgStatus {
    /// `‖r‖/‖b‖ ≤ tol` (including a zero right-hand side).
    Converged,
    /// `max_iters` reached first.
    MaxIters,
    /// `dᵀAd ≤ 0` at some iteration; `x` is the last iterate.
    Breakdown,
}

/// Result of [`pcg`].
#[derive(Debug, Clone)]
pub struct PcgSolution {
    pub x: Vec<f64>,
    /// Iterations taken (matrix applications after the initial residual).
    pub iterations: usize,
    /// Final `‖r‖/‖b‖` (0 for a zero right-hand side).
    pub residual_rel: f64,
    pub status: PcgStatus,
}

impl PcgSolution {
    pub fn converged(&self) -> bool {
        self.status == PcgStatus::Converged
    }
}

/// `z = M⁻¹·r`.
pub trait Preconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]);
}

/// No preconditioning: `z = r`.
pub struct Identity;

impl Preconditioner for Identity {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
    }
}

/// Diagonal scaling `z = r / diag`. Entries with a zero diagonal map to 0,
/// which is how masked (pinned, fixed) unknowns stay out of the iteration.
pub struct Jacobi {
    pub diag: Vec<f64>,
}

impl Jacobi {
    pub fn new(diag: Vec<f64>) -> Self {
        Self { diag }
    }

    pub fn from_csr(a: &CsrMatrix<f64>) -> Self {
        Self::new(csr_diagonal(a))
    }
}

impl Preconditioner for Jacobi {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        for ((z, &r), &d) in z.iter_mut().zip(r).zip(&self.diag) {
            *z = if d != 0.0 { r / d } else { 0.0 };
        }
    }
}

/// The diagonal of a square CSR matrix (0 where it has no stored entry).
pub fn csr_diagonal(a: &CsrMatrix<f64>) -> Vec<f64> {
    (0..a.nrows)
        .map(|i| {
            let (s, e) = (a.row_ptrs[i], a.row_ptrs[i + 1]);
            (s..e)
                .find(|&k| a.col_indices[k] == i)
                .map_or(0.0, |k| a.values[k])
        })
        .collect()
}

/// `y = A·x` on slices.
pub fn csr_apply(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for (i, yi) in y.iter_mut().enumerate().take(a.nrows) {
        let mut s = 0.0;
        for k in a.row_ptrs[i]..a.row_ptrs[i + 1] {
            s += a.values[k] * x[a.col_indices[k]];
        }
        *yi = s;
    }
}

/// Incomplete Cholesky with zero fill: `A ≈ L·Lᵀ` with `L` restricted to
/// the lower-triangular pattern of `A`. Applying it is a forward then a
/// backward triangular solve.
pub struct Ic0 {
    /// `L` row by row (CSR, lower triangle, diagonal stored last in each row).
    pub l: CsrMatrix<f64>,
}

/// Why [`Ic0::new`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ic0Error {
    /// A row has no diagonal entry.
    MissingDiagonal(usize),
    /// The pivot of this row went non-positive: IC(0) does not exist for
    /// this matrix (it is not an M-matrix or is badly scaled).
    NonPositivePivot(usize),
}

impl Ic0 {
    /// Factor the lower triangle of a symmetric CSR matrix whose rows have
    /// sorted column indices.
    pub fn new(a: &CsrMatrix<f64>) -> Result<Self, Ic0Error> {
        let n = a.nrows;
        let mut row_ptrs = vec![0usize; n + 1];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            let mut has_diag = false;
            for k in a.row_ptrs[i]..a.row_ptrs[i + 1] {
                let j = a.col_indices[k];
                if j < i {
                    cols.push(j);
                    vals.push(a.values[k]);
                } else if j == i {
                    has_diag = true;
                }
            }
            if !has_diag {
                return Err(Ic0Error::MissingDiagonal(i));
            }
            cols.push(i);
            vals.push(0.0);
            row_ptrs[i + 1] = cols.len();
        }
        let diag_a = csr_diagonal(a);
        // Row-by-row (left-looking) IC(0): for each i and stored j < i,
        //   L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]
        //   L[i,i] = sqrt(A[i,i] − Σ_{k<i} L[i,k]²)
        // with the sums restricted to the pattern (merge of sorted rows).
        for i in 0..n {
            let (si, ei) = (row_ptrs[i], row_ptrs[i + 1] - 1);
            for p in si..ei {
                let j = cols[p];
                let (sj, ej) = (row_ptrs[j], row_ptrs[j + 1] - 1);
                let (mut a_, mut b_) = (si, sj);
                let mut s = 0.0;
                while a_ < p && b_ < ej {
                    match cols[a_].cmp(&cols[b_]) {
                        core::cmp::Ordering::Less => a_ += 1,
                        core::cmp::Ordering::Greater => b_ += 1,
                        core::cmp::Ordering::Equal => {
                            s += vals[a_] * vals[b_];
                            a_ += 1;
                            b_ += 1;
                        }
                    }
                }
                vals[p] = (vals[p] - s) / vals[ej];
            }
            let s: f64 = vals[si..ei].iter().map(|v| v * v).sum();
            let piv = diag_a[i] - s;
            if piv <= 0.0 || piv.is_nan() {
                return Err(Ic0Error::NonPositivePivot(i));
            }
            vals[ei] = <f64 as tang::Scalar>::sqrt(piv);
        }
        Ok(Self {
            l: CsrMatrix {
                nrows: n,
                ncols: n,
                row_ptrs,
                col_indices: cols,
                values: vals,
            },
        })
    }
}

impl Preconditioner for Ic0 {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let l = &self.l;
        let n = l.nrows;
        // L·w = r
        for i in 0..n {
            let (s, e) = (l.row_ptrs[i], l.row_ptrs[i + 1] - 1);
            let mut acc = r[i];
            for k in s..e {
                acc -= l.values[k] * z[l.col_indices[k]];
            }
            z[i] = acc / l.values[e];
        }
        // Lᵀ·z = w, column-oriented over the rows of L.
        for i in (0..n).rev() {
            let e = l.row_ptrs[i + 1] - 1;
            z[i] /= l.values[e];
            let zi = z[i];
            for k in l.row_ptrs[i]..e {
                z[l.col_indices[k]] -= l.values[k] * zi;
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

fn norm(a: &[f64]) -> f64 {
    <f64 as tang::Scalar>::sqrt(dot(a, a))
}

/// Solve `A·x = b` from `x0`. See the module docs.
///
/// A zero `b` returns `x = 0` (not `x0`) in zero iterations.
pub fn pcg(
    mut apply: impl FnMut(&[f64], &mut [f64]),
    precond: &impl Preconditioner,
    b: &[f64],
    x0: &[f64],
    opts: &PcgOptions,
) -> PcgSolution {
    let n = b.len();
    let b_norm = norm(b);
    if b_norm == 0.0 {
        return PcgSolution {
            x: vec![0.0; n],
            iterations: 0,
            residual_rel: 0.0,
            status: PcgStatus::Converged,
        };
    }
    let mut x = x0.to_vec();
    let mut r = vec![0.0; n];
    apply(&x, &mut r);
    for (r, &b) in r.iter_mut().zip(b) {
        *r = b - *r;
    }
    let mut residual_rel = norm(&r) / b_norm;
    let done = |x, iterations, residual_rel, status| PcgSolution {
        x,
        iterations,
        residual_rel,
        status,
    };
    if residual_rel <= opts.tol {
        return done(x, 0, residual_rel, PcgStatus::Converged);
    }
    let mut z = vec![0.0; n];
    precond.apply(&r, &mut z);
    let mut d = z.clone();
    let mut rz = dot(&r, &z);
    let mut ad = vec![0.0; n];
    for iter in 1..=opts.max_iters {
        apply(&d, &mut ad);
        let dad = dot(&d, &ad);
        if dad <= 0.0 || dad.is_nan() {
            return done(x, iter, residual_rel, PcgStatus::Breakdown);
        }
        let alpha = rz / dad;
        for i in 0..n {
            x[i] += alpha * d[i];
            r[i] -= alpha * ad[i];
        }
        residual_rel = norm(&r) / b_norm;
        if residual_rel <= opts.tol {
            return done(x, iter, residual_rel, PcgStatus::Converged);
        }
        precond.apply(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            d[i] = z[i] + beta * d[i];
        }
    }
    done(x, opts.max_iters, residual_rel, PcgStatus::MaxIters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CooMatrix;

    /// 2D 5-point Laplacian plus a shift, n×n grid.
    fn laplace(n: usize, shift: f64) -> CsrMatrix<f64> {
        let mut c = CooMatrix::new(n * n, n * n);
        for j in 0..n {
            for i in 0..n {
                let p = j * n + i;
                c.push(p, p, 4.0 + shift);
                if i > 0 {
                    c.push(p, p - 1, -1.0);
                }
                if i + 1 < n {
                    c.push(p, p + 1, -1.0);
                }
                if j > 0 {
                    c.push(p, p - n, -1.0);
                }
                if j + 1 < n {
                    c.push(p, p + n, -1.0);
                }
            }
        }
        c.to_csr()
    }

    fn check(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let mut ax = vec![0.0; b.len()];
        csr_apply(a, x, &mut ax);
        let r: f64 = ax.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
        <f64 as tang::Scalar>::sqrt(r) / norm(b)
    }

    #[test]
    fn jacobi_and_ic0_solve() {
        let a = laplace(20, 0.01);
        let b: Vec<f64> = (0..400).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let opts = PcgOptions {
            tol: 1e-12,
            max_iters: 2000,
        };
        let x0 = vec![0.0; 400];
        let j = pcg(
            |x, y| csr_apply(&a, x, y),
            &Jacobi::from_csr(&a),
            &b,
            &x0,
            &opts,
        );
        let ic = Ic0::new(&a).unwrap();
        let k = pcg(|x, y| csr_apply(&a, x, y), &ic, &b, &x0, &opts);
        assert!(j.converged() && k.converged());
        assert!(check(&a, &j.x, &b) < 1e-11 && check(&a, &k.x, &b) < 1e-11);
        // IC(0) earns its keep on a Laplacian.
        assert!(
            k.iterations * 2 < j.iterations,
            "{} vs {}",
            k.iterations,
            j.iterations
        );
    }

    #[test]
    fn ic0_is_exact_on_tridiagonal() {
        // No fill on a tridiagonal matrix: IC(0) is the exact Cholesky.
        let n = 30;
        let mut c = CooMatrix::new(n, n);
        for i in 0..n {
            c.push(i, i, 2.5);
            if i > 0 {
                c.push(i, i - 1, -1.0);
                c.push(i - 1, i, -1.0);
            }
        }
        let a = c.to_csr();
        let b: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let s = pcg(
            |x, y| csr_apply(&a, x, y),
            &Ic0::new(&a).unwrap(),
            &b,
            &vec![0.0; n],
            &PcgOptions::default(),
        );
        assert!(s.converged() && s.iterations <= 1);
    }

    #[test]
    fn zero_rhs_and_breakdown() {
        let a = laplace(4, 0.0);
        let s = pcg(
            |x, y| csr_apply(&a, x, y),
            &Identity,
            &[0.0; 16],
            &[1.0; 16],
            &PcgOptions::default(),
        );
        assert_eq!(
            (s.status, s.iterations, s.x[0]),
            (PcgStatus::Converged, 0, 0.0)
        );
        let neg = |x: &[f64], y: &mut [f64]| y.iter_mut().zip(x).for_each(|(y, x)| *y = -x);
        let s = pcg(neg, &Identity, &[1.0; 4], &[0.0; 4], &PcgOptions::default());
        assert_eq!(s.status, PcgStatus::Breakdown);
    }
}
