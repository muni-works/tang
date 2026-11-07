//! Sparse linear algebra in the tensor IR, and the first operator nodes.
//!
//! - **SpMV is a gather comprehension.** A matrix in ELL layout (every row
//!   padded to `width` slots, padding has value 0 and column 0) multiplies
//!   a vector as `y[i] = Σ_s vals[i,s] · x[cols[i,s]]`: the `x` read is a
//!   gathered view addressed by the sibling `cols` arg. See [`spmv`].
//! - **PCG is a tensor-IR program.** Each step of preconditioned conjugate
//!   gradients (apply, update, precondition, new direction) is a small
//!   graph of comprehensions; the host keeps only the scalar recurrence
//!   (α, β, the stopping test). Jacobi is a map; IC(0) is a pair of
//!   level-scheduled triangular solves, one comprehension per level. See
//!   [`PcgProgram`]. It reproduces [`tang_sparse::pcg`] to round-off.
//! - **[`LinearSolve`]** is the first [`Operator`]: forward is the native
//!   [`tang_sparse::pcg`] on CSR values; its adjoint is, by the implicit
//!   function theorem, one more solve with `Aᵀ` (`λ = A⁻ᵀ·x̄`, `b̄ = λ`) and
//!   the outer-product cotangent `Ā[r,c] = −λ[r]·x[c]` on the pattern, a
//!   gather comprehension.
//! - **[`IndexedGather`]** is `out[i] = src[index[i]]` with a known inverse
//!   map. A gathered view differentiates to a dense gated scatter (see
//!   `tensor_grad.rs`), quadratic in size; with the inverse lists in hand
//!   the adjoint is a gather instead, linear. Assembly (element → node,
//!   parameter → element) uses it.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tang::Scalar;
use tang_expr::ExprId;
use tang_sparse::{CsrMatrix, Ic0, Jacobi, PcgOptions, PcgSolution, PcgStatus, Preconditioner};

use super::{err, Arg, Dim, Monoid, Operator, ShapeError, TensorGraph, TensorId, View};

// ---------------------------------------------------------------------------
// ELL layout and the SpMV comprehension
// ---------------------------------------------------------------------------

/// A CSR pattern laid out as ELL: `width` slots per row, row-major.
#[derive(Clone, Debug)]
pub struct Ell {
    pub n: usize,
    pub width: usize,
    /// Column of each slot (`n · width`; padding reads column 0).
    pub cols: Vec<usize>,
    /// CSR entry `k` lives in slot `slot[k]`.
    pub slot: Vec<usize>,
}

impl Ell {
    pub fn from_csr(a: &CsrMatrix<f64>) -> Self {
        let n = a.nrows;
        let width = (0..n)
            .map(|i| a.row_ptrs[i + 1] - a.row_ptrs[i])
            .max()
            .unwrap_or(0)
            .max(1);
        let mut cols = vec![0; n * width];
        let mut slot = vec![0; a.nnz()];
        for i in 0..n {
            for (s, k) in (a.row_ptrs[i]..a.row_ptrs[i + 1]).enumerate() {
                cols[i * width + s] = a.col_indices[k];
                slot[k] = i * width + s;
            }
        }
        Self {
            n,
            width,
            cols,
            slot,
        }
    }

    /// CSR values scattered into ELL slots (padding 0).
    pub fn values(&self, csr_values: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0; self.n * self.width];
        for (k, &s) in self.slot.iter().enumerate() {
            v[s] = csr_values[k];
        }
        v
    }

    /// The column table as f64 index values, for a graph input.
    pub fn cols_f64(&self) -> Vec<f64> {
        self.cols.iter().map(|&c| c as f64).collect()
    }
}

/// `y[i] = Σ_s vals[i,s] · x[cols[i,s]]` with `vals`, `cols` shaped
/// `[n, width]` and `x` shaped `[m]`.
pub fn spmv(
    g: &mut TensorGraph,
    vals: TensorId,
    cols: TensorId,
    x: TensorId,
) -> Result<TensorId, ShapeError> {
    let s = g.shape(vals).to_vec();
    if s.len() != 2 || g.shape(cols) != s || g.shape(x).len() != 1 {
        return err(format!(
            "spmv: vals {s:?}, cols {:?}, x {:?}",
            g.shape(cols),
            g.shape(x)
        ));
    }
    let args = vec![
        Arg {
            src: vals,
            view: View::identity(2, 2),
        },
        Arg {
            src: cols,
            view: View::identity(2, 2),
        },
        Arg {
            src: x,
            view: View::pick(2, &[None]).gather(0, 1, 1),
        },
    ];
    g.comp(&[s[0]], &[s[1]], args, Some(Monoid::sum()), |v| {
        vec![v[0] * v[2]]
    })
}

/// `Σ_i a[i]·b[i]` as a `[1]` tensor.
pub fn dot(g: &mut TensorGraph, a: TensorId, b: TensorId) -> Result<TensorId, ShapeError> {
    let n = g.shape(a)[0];
    let args = vec![
        Arg {
            src: a,
            view: View::pick(2, &[Some(1)]),
        },
        Arg {
            src: b,
            view: View::pick(2, &[Some(1)]),
        },
    ];
    g.comp(&[Dim::Fixed(1)], &[n], args, Some(Monoid::sum()), |v| vec![v[0] * v[1]])
}

/// `a + s·b` with `s` a `[1]` tensor broadcast over `a`'s `[n]`.
fn axpy(
    g: &mut TensorGraph,
    a: TensorId,
    s: TensorId,
    b: TensorId,
    sign: f64,
) -> Result<TensorId, ShapeError> {
    let n = g.shape(a)[0];
    let args = vec![
        Arg {
            src: a,
            view: View::identity(1, 1),
        },
        Arg {
            src: s,
            view: View::pick(1, &[None]),
        },
        Arg {
            src: b,
            view: View::identity(1, 1),
        },
    ];
    g.comp(&[n], &[], args, None, |v| {
        let sb = v[1] * v[2];
        vec![if sign > 0.0 { v[0] + sb } else { v[0] - sb }]
    })
}

// ---------------------------------------------------------------------------
// Preconditioners as tensor-IR programs
// ---------------------------------------------------------------------------

/// How [`PcgProgram`] (and [`LinearSolve`]) precondition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precond {
    Jacobi,
    Ic0,
}

/// IC(0) factor as ELL tables plus a level schedule for each triangle.
struct Ic0Tables {
    /// Strict lower part of `L` (`[n, wl]`), its columns, and the diagonal.
    lower: (Vec<f64>, Vec<f64>, usize),
    /// Strict upper part of `Lᵀ` (`[n, wu]`) and its columns.
    upper: (Vec<f64>, Vec<f64>, usize),
    diag: Vec<f64>,
    /// Level of each row in the forward (lower) and backward (upper) sweep.
    lower_levels: Vec<f64>,
    upper_levels: Vec<f64>,
    n_lower: usize,
    n_upper: usize,
}

fn ic0_tables(ic: &Ic0) -> Ic0Tables {
    let l = &ic.l;
    let n = l.nrows;
    let diag: Vec<f64> = (0..n).map(|i| l.values[l.row_ptrs[i + 1] - 1]).collect();
    // strict lower rows
    let lrows: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|i| {
            (l.row_ptrs[i]..l.row_ptrs[i + 1] - 1)
                .map(|k| (l.col_indices[k], l.values[k]))
                .collect()
        })
        .collect();
    // strict upper rows of Lᵀ: row j holds (i, L[i,j]) for i > j
    let mut urows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, r) in lrows.iter().enumerate() {
        for &(j, v) in r {
            urows[j].push((i, v));
        }
    }
    let ell = |rows: &Vec<Vec<(usize, f64)>>| {
        let w = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
        let mut v = vec![0.0; n * w];
        let mut c = vec![0.0; n * w];
        for (i, r) in rows.iter().enumerate() {
            for (s, &(j, x)) in r.iter().enumerate() {
                v[i * w + s] = x;
                c[i * w + s] = j as f64;
            }
        }
        (v, c, w)
    };
    let mut lower_levels = vec![0usize; n];
    for i in 0..n {
        lower_levels[i] = lrows[i].iter().map(|&(j, _)| lower_levels[j] + 1).max().unwrap_or(0);
    }
    let mut upper_levels = vec![0usize; n];
    for i in (0..n).rev() {
        upper_levels[i] = urows[i].iter().map(|&(j, _)| upper_levels[j] + 1).max().unwrap_or(0);
    }
    Ic0Tables {
        lower: ell(&lrows),
        upper: ell(&urows),
        diag,
        n_lower: lower_levels.iter().max().map_or(0, |m| m + 1),
        n_upper: upper_levels.iter().max().map_or(0, |m| m + 1),
        lower_levels: lower_levels.iter().map(|&l| l as f64).collect(),
        upper_levels: upper_levels.iter().map(|&l| l as f64).collect(),
    }
}

/// One level of a triangular solve: rows at `level` become
/// `(rhs[i] − Σ_s T[i,s]·z[cols[i,s]]) / diag[i]`, every other row keeps
/// `z[i]`.
#[allow(clippy::too_many_arguments)]
fn tri_level(
    g: &mut TensorGraph,
    z: TensorId,
    rhs: TensorId,
    t: TensorId,
    cols: TensorId,
    diag: TensorId,
    levels: TensorId,
    level: usize,
) -> Result<TensorId, ShapeError> {
    let off = spmv(g, t, cols, z)?;
    let lv = level as f64;
    g.map(&[z, rhs, off, diag, levels], move |v| {
        let here = v[4] - ExprId::from_f64(lv);
        let hit = ExprId::from_f64(0.5) - here * here;
        <ExprId as Scalar>::select(hit, (v[1] - v[2]) / v[3], v[0])
    })
}

// ---------------------------------------------------------------------------
// PCG as a tensor-IR program
// ---------------------------------------------------------------------------

/// Preconditioned CG whose every vector operation is a comprehension
/// evaluated by the reference interpreter. Built once per matrix pattern
/// and preconditioner; [`PcgProgram::solve`] runs the host recurrence.
pub struct PcgProgram {
    /// Graph inputs, in order: ELL values, ELL cols, then the step's own.
    apply: (TensorGraph, [TensorId; 2]),
    update: (TensorGraph, [TensorId; 3]),
    precond: (TensorGraph, [TensorId; 2]),
    direction: (TensorGraph, TensorId),
    /// Fixed inputs of the apply graph (values, cols).
    a_inputs: [Vec<f64>; 2],
    /// Fixed inputs of the preconditioner graph.
    m_inputs: Vec<Vec<f64>>,
}

impl PcgProgram {
    /// Programs for `A` (CSR, sorted rows) with `precond`.
    pub fn new(a: &CsrMatrix<f64>, precond: Precond) -> Result<Self, String> {
        let n = a.nrows;
        let ell = Ell::from_csr(a);
        let w = ell.width;

        // ad = A·d, dad = d·ad
        let apply = {
            let mut g = TensorGraph::new();
            let v = g.input(&[n, w]);
            let c = g.input(&[n, w]);
            let d = g.input(&[n]);
            let ad = spmv(&mut g, v, c, d).map_err(|e| e.0)?;
            let dad = dot(&mut g, d, ad).map_err(|e| e.0)?;
            (g, [ad, dad])
        };
        // x += α·d; r −= α·ad; rr = r·r
        let update = {
            let mut g = TensorGraph::new();
            let x = g.input(&[n]);
            let r = g.input(&[n]);
            let d = g.input(&[n]);
            let ad = g.input(&[n]);
            let alpha = g.input(&[1]);
            let x2 = axpy(&mut g, x, alpha, d, 1.0).map_err(|e| e.0)?;
            let r2 = axpy(&mut g, r, alpha, ad, -1.0).map_err(|e| e.0)?;
            let rr = dot(&mut g, r2, r2).map_err(|e| e.0)?;
            (g, [x2, r2, rr])
        };
        // z = M⁻¹·r, rz = r·z
        let (precond, m_inputs) = {
            let mut g = TensorGraph::new();
            let r = g.input(&[n]);
            let (z, inputs) = match precond {
                Precond::Jacobi => {
                    let d = g.input(&[n]);
                    let z = g
                        .map(&[r, d], |v| {
                            <ExprId as Scalar>::select(v[1] * v[1], v[0] / v[1], ExprId::from_f64(0.0))
                        })
                        .map_err(|e| e.0)?;
                    (z, vec![tang_sparse::csr_diagonal(a)])
                }
                Precond::Ic0 => {
                    let ic = Ic0::new(a).map_err(|e| format!("{e:?}"))?;
                    let t = ic0_tables(&ic);
                    let (lv, lc, wl) = t.lower;
                    let (uv, uc, wu) = t.upper;
                    let li = (g.input(&[n, wl]), g.input(&[n, wl]));
                    let ui = (g.input(&[n, wu]), g.input(&[n, wu]));
                    let dg = g.input(&[n]);
                    let ll = g.input(&[n]);
                    let ul = g.input(&[n]);
                    // L·w = r, level by level, starting from w = 0
                    let mut z = g.map(&[r], |_| ExprId::from_f64(0.0)).map_err(|e| e.0)?;
                    for lev in 0..t.n_lower {
                        z = tri_level(&mut g, z, r, li.0, li.1, dg, ll, lev).map_err(|e| e.0)?;
                    }
                    // Lᵀ·z = w
                    let w_ = z;
                    let mut z = g.map(&[r], |_| ExprId::from_f64(0.0)).map_err(|e| e.0)?;
                    for lev in 0..t.n_upper {
                        z = tri_level(&mut g, z, w_, ui.0, ui.1, dg, ul, lev).map_err(|e| e.0)?;
                    }
                    (z, vec![lv, lc, uv, uc, t.diag, t.lower_levels, t.upper_levels])
                }
            };
            let rz = dot(&mut g, r, z).map_err(|e| e.0)?;
            ((g, [z, rz]), inputs)
        };
        // d = z + β·d
        let direction = {
            let mut g = TensorGraph::new();
            let z = g.input(&[n]);
            let d = g.input(&[n]);
            let beta = g.input(&[1]);
            let d2 = axpy(&mut g, z, beta, d, 1.0).map_err(|e| e.0)?;
            (g, d2)
        };
        Ok(Self {
            apply,
            update,
            precond,
            direction,
            a_inputs: [ell.values(&a.values), ell.cols_f64()],
            m_inputs,
        })
    }

    fn run_apply(&self, d: &[f64]) -> (Vec<f64>, f64) {
        let ins = [self.a_inputs[0].clone(), self.a_inputs[1].clone(), d.to_vec()];
        let mut o = self.apply.0.eval(&ins, &self.apply.1);
        let dad = o[1][0];
        (o.swap_remove(0), dad)
    }

    fn run_precond(&self, r: &[f64]) -> (Vec<f64>, f64) {
        let mut ins = vec![r.to_vec()];
        ins.extend(self.m_inputs.iter().cloned());
        let mut o = self.precond.0.eval(&ins, &self.precond.1);
        let rz = o[1][0];
        (o.swap_remove(0), rz)
    }

    /// Solve `A·x = b` from `x0`; same recurrence and stopping rule as
    /// [`tang_sparse::pcg`].
    pub fn solve(&self, b: &[f64], x0: &[f64], opts: &PcgOptions) -> PcgSolution {
        let n = b.len();
        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let done = |x, iterations, residual_rel, status| PcgSolution {
            x,
            iterations,
            residual_rel,
            status,
        };
        if b_norm == 0.0 {
            return done(vec![0.0; n], 0, 0.0, PcgStatus::Converged);
        }
        let mut x = x0.to_vec();
        let (ax, _) = self.run_apply(&x);
        let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let mut residual_rel = r.iter().map(|v| v * v).sum::<f64>().sqrt() / b_norm;
        if residual_rel <= opts.tol {
            return done(x, 0, residual_rel, PcgStatus::Converged);
        }
        let (z, mut rz) = self.run_precond(&r);
        let mut d = z;
        for iter in 1..=opts.max_iters {
            let (ad, dad) = self.run_apply(&d);
            if dad <= 0.0 || dad.is_nan() {
                return done(x, iter, residual_rel, PcgStatus::Breakdown);
            }
            let alpha = rz / dad;
            let mut o = self.update.0.eval(
                &[x, r, d.clone(), ad, vec![alpha]],
                &self.update.1,
            );
            residual_rel = o[2][0].sqrt() / b_norm;
            r = o.swap_remove(1);
            x = o.swap_remove(0);
            if residual_rel <= opts.tol {
                return done(x, iter, residual_rel, PcgStatus::Converged);
            }
            let (z, rz_new) = self.run_precond(&r);
            let beta = rz_new / rz;
            rz = rz_new;
            d = self
                .direction
                .0
                .eval(&[z, d, vec![beta]], &[self.direction.1])
                .swap_remove(0);
        }
        done(x, opts.max_iters, residual_rel, PcgStatus::MaxIters)
    }
}

// ---------------------------------------------------------------------------
// Operator nodes
// ---------------------------------------------------------------------------

/// A fixed square sparsity pattern (CSR, sorted rows).
#[derive(Clone, Debug)]
pub struct Pattern {
    pub n: usize,
    pub row_ptrs: Vec<usize>,
    pub cols: Vec<usize>,
}

impl Pattern {
    pub fn nnz(&self) -> usize {
        self.cols.len()
    }

    /// Row of each stored entry.
    pub fn rows(&self) -> Vec<usize> {
        (0..self.n)
            .flat_map(|i| std::iter::repeat_n(i, self.row_ptrs[i + 1] - self.row_ptrs[i]))
            .collect()
    }

    pub fn csr(&self, values: &[f64]) -> CsrMatrix<f64> {
        CsrMatrix {
            nrows: self.n,
            ncols: self.n,
            row_ptrs: self.row_ptrs.clone(),
            col_indices: self.cols.clone(),
            values: values.to_vec(),
        }
    }
}

/// One forward solve as [`LinearSolve`] recorded it.
#[derive(Clone, Debug)]
pub struct SolveRecord {
    /// `true` for the adjoint (`Aᵀ`) solve.
    pub transpose: bool,
    pub iterations: usize,
    pub residual_rel: f64,
    pub status: PcgStatus,
    /// Wall time of factorization plus iteration.
    pub time: Duration,
}

/// `x = A⁻¹·b` (or `A⁻ᵀ·b`) as an operator node; see the module docs.
///
/// Inputs: `[values (nnz), b (n), rows (nnz), cols (nnz)]`, where `rows`
/// and `cols` are the pattern as index values (graph inputs the adjoint
/// gathers through; they receive no cotangent). Output: `x (n)`.
///
/// Forward never panics on a hard system: it returns the last iterate and
/// appends a [`SolveRecord`] (shared with the adjoint node) for callers to
/// check with [`LinearSolve::records`].
pub struct LinearSolve {
    pub pattern: Arc<Pattern>,
    pub precond: Precond,
    pub opts: PcgOptions,
    pub transpose: bool,
    records: Arc<Mutex<Vec<SolveRecord>>>,
}

impl LinearSolve {
    pub fn new(pattern: Arc<Pattern>, precond: Precond, opts: PcgOptions) -> Self {
        Self {
            pattern,
            precond,
            opts,
            transpose: false,
            records: Arc::default(),
        }
    }

    /// Every solve this node and its adjoints have run, in order.
    pub fn records(&self) -> Vec<SolveRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Append the node: `x = A⁻¹·b`.
    pub fn apply(
        self: &Arc<Self>,
        g: &mut TensorGraph,
        values: TensorId,
        b: TensorId,
        rows: TensorId,
        cols: TensorId,
    ) -> Result<TensorId, ShapeError> {
        let (n, nnz) = (self.pattern.n, self.pattern.nnz());
        if g.shape(values) != [nnz] || g.shape(b) != [n] {
            return err(format!(
                "LinearSolve: values {:?} (want [{nnz}]), b {:?} (want [{n}])",
                g.shape(values),
                g.shape(b)
            ));
        }
        Ok(g.operator(self.clone(), &[values, b, rows, cols], &[n]))
    }
}

impl Operator for LinearSolve {
    fn name(&self) -> &str {
        if self.transpose {
            "LinearSolveT"
        } else {
            "LinearSolve"
        }
    }

    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        let t0 = Instant::now();
        let mut a = self.pattern.csr(inputs[0]);
        if self.transpose {
            a = a.transpose();
        }
        let b = inputs[1];
        let x0 = vec![0.0; b.len()];
        let apply = |x: &[f64], y: &mut [f64]| tang_sparse::csr_apply(&a, x, y);
        let sol = match self.precond {
            Precond::Jacobi => tang_sparse::pcg(apply, &Jacobi::from_csr(&a), b, &x0, &self.opts),
            Precond::Ic0 => match Ic0::new(&a) {
                Ok(ic) => tang_sparse::pcg(apply, &ic, b, &x0, &self.opts),
                // IC(0) does not exist for this matrix: fall back to Jacobi
                Err(_) => tang_sparse::pcg(apply, &Jacobi::from_csr(&a), b, &x0, &self.opts),
            },
        };
        self.records.lock().unwrap().push(SolveRecord {
            transpose: self.transpose,
            iterations: sol.iterations,
            residual_rel: sol.residual_rel,
            status: sol.status,
            time: t0.elapsed(),
        });
        sol.x
    }

    fn backward(
        &self,
        g: &mut TensorGraph,
        inputs: &[TensorId],
        output: TensorId,
        cot: TensorId,
    ) -> Result<Vec<Option<TensorId>>, ShapeError> {
        let (values, rows, cols) = (inputs[0], inputs[2], inputs[3]);
        // λ = A⁻ᵀ·x̄ (for the transposed node, A⁻¹·x̄)
        let adj = Arc::new(LinearSolve {
            pattern: self.pattern.clone(),
            precond: self.precond,
            opts: self.opts,
            transpose: !self.transpose,
            records: self.records.clone(),
        });
        let lam = g.operator(adj, &[values, cot, rows, cols], &[self.pattern.n]);
        // Ā[r,c] = −λ[r]·x[c]   (transposed node: −λ[c]·x[r])
        let (li, xi) = if self.transpose { (3, 2) } else { (2, 3) };
        let nnz = self.pattern.nnz();
        let args = vec![
            Arg {
                src: inputs[2],
                view: View::identity(1, 1),
            },
            Arg {
                src: inputs[3],
                view: View::identity(1, 1),
            },
            Arg {
                src: lam,
                view: View::pick(1, &[None]).gather(0, li - 2, 1),
            },
            Arg {
                src: output,
                view: View::pick(1, &[None]).gather(0, xi - 2, 1),
            },
        ];
        let dvalues = g.comp(&[nnz], &[], args, None, |v| vec![-(v[2] * v[3])])?;
        Ok(vec![Some(dvalues), Some(lam), None, None])
    }
}

/// `out[i…] = src[index[i…]]` for a 1-D `src`, with the inverse lists that
/// make the adjoint a gather. `index` entries `< 0` read 0 (padding).
///
/// Inputs: `[src (m), index (shape), inverse (m, w)]`, where row `j` of
/// `inverse` lists the flat output positions that read `src[j]`, padded
/// with −1. Output: `index`'s shape. Build the index/inverse pair with
/// [`IndexedGather::tables`].
pub struct IndexedGather;

impl IndexedGather {
    /// `(inverse, width)` for `index` over a source of length `m`.
    pub fn tables(index: &[i64], m: usize) -> (Vec<f64>, usize) {
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (p, &j) in index.iter().enumerate() {
            if j >= 0 {
                lists[j as usize].push(p);
            }
        }
        let w = lists.iter().map(|l| l.len()).max().unwrap_or(0).max(1);
        let mut inv = vec![-1.0; m * w];
        for (j, l) in lists.iter().enumerate() {
            for (s, &p) in l.iter().enumerate() {
                inv[j * w + s] = p as f64;
            }
        }
        (inv, w)
    }

    /// Append the node.
    pub fn apply(
        g: &mut TensorGraph,
        src: TensorId,
        index: TensorId,
        inverse: TensorId,
    ) -> Result<TensorId, ShapeError> {
        let m = g.shape(src).to_vec();
        let inv = g.shape(inverse).to_vec();
        if m.len() != 1 || inv.len() != 2 || inv[0] != m[0] {
            return err(format!("IndexedGather: src {m:?}, inverse {inv:?}"));
        }
        let shape = g.shape(index).to_vec();
        Ok(g.operator(Arc::new(IndexedGather), &[src, index, inverse], &shape))
    }
}

impl Operator for IndexedGather {
    fn name(&self) -> &str {
        "IndexedGather"
    }

    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        let src = inputs[0];
        inputs[1]
            .iter()
            .map(|&j| if j < 0.0 { 0.0 } else { src[j.round() as usize] })
            .collect()
    }

    fn backward(
        &self,
        g: &mut TensorGraph,
        inputs: &[TensorId],
        _output: TensorId,
        cot: TensorId,
    ) -> Result<Vec<Option<TensorId>>, ShapeError> {
        let inv = g.shape(inputs[2]).to_vec();
        // src̄[j] = Σ_s cot[inverse[j,s]], padding (−1) read as 0
        let flat = g.reshape(cot, &[super::numel(g.shape(cot))])?;
        let mut view = View::pick(2, &[None]).gather(0, 1, 1);
        view.guard = true;
        let args = vec![
            Arg {
                src: flat,
                view,
            },
            Arg {
                src: inputs[2],
                view: View::identity(2, 2),
            },
        ];
        let d = g.comp(&[inv[0]], &[inv[1]], args, Some(Monoid::sum()), |v| vec![v[0]])?;
        Ok(vec![Some(d), None, None])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_sparse::CooMatrix;

    fn laplace(n: usize, shift: f64) -> CsrMatrix<f64> {
        let mut c = CooMatrix::new(n * n, n * n);
        for j in 0..n {
            for i in 0..n {
                let p = j * n + i;
                c.push(p, p, 4.0 + shift + 0.01 * (p % 5) as f64);
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

    fn rhs(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect()
    }

    #[test]
    fn spmv_matches_csr() {
        let a = laplace(6, 0.1);
        let ell = Ell::from_csr(&a);
        let mut g = TensorGraph::new();
        let v = g.input(&[ell.n, ell.width]);
        let c = g.input(&[ell.n, ell.width]);
        let x = g.input(&[ell.n]);
        let y = spmv(&mut g, v, c, x).unwrap();
        let xs = rhs(36);
        let got = g.eval(&[ell.values(&a.values), ell.cols_f64(), xs.clone()], &[y]);
        let mut want = vec![0.0; 36];
        tang_sparse::csr_apply(&a, &xs, &mut want);
        assert_eq!(got[0], want);
    }

    #[test]
    fn ir_pcg_matches_native() {
        let a = laplace(8, 0.05);
        let b = rhs(64);
        let opts = PcgOptions {
            tol: 1e-12,
            max_iters: 500,
        };
        let x0 = vec![0.0; 64];
        for p in [Precond::Jacobi, Precond::Ic0] {
            let prog = PcgProgram::new(&a, p).unwrap();
            let ir = prog.solve(&b, &x0, &opts);
            let apply = |x: &[f64], y: &mut [f64]| tang_sparse::csr_apply(&a, x, y);
            let native = match p {
                Precond::Jacobi => tang_sparse::pcg(apply, &Jacobi::from_csr(&a), &b, &x0, &opts),
                Precond::Ic0 => tang_sparse::pcg(apply, &Ic0::new(&a).unwrap(), &b, &x0, &opts),
            };
            assert!(ir.converged() && native.converged(), "{p:?}");
            assert_eq!(ir.iterations, native.iterations, "{p:?}");
            let err = ir
                .x
                .iter()
                .zip(&native.x)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(err < 1e-12, "{p:?}: {err}");
        }
    }

    #[test]
    fn ir_ic0_apply_matches_native() {
        let a = laplace(5, 0.2);
        let ic = Ic0::new(&a).unwrap();
        let prog = PcgProgram::new(&a, Precond::Ic0).unwrap();
        let r = rhs(25);
        let (z, _) = prog.run_precond(&r);
        let mut want = vec![0.0; 25];
        ic.apply(&r, &mut want);
        let err = z.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1e-14, "{err}");
    }

    /// J = Σ_i w_i·x_i with A(θ)·x = b(θ), A's values and b graph inputs:
    /// the operator adjoint against central differences.
    #[test]
    fn linear_solve_gradient_matches_fd() {
        let a = laplace(4, 0.3);
        let n = a.nrows;
        let pattern = Arc::new(Pattern {
            n,
            row_ptrs: a.row_ptrs.clone(),
            cols: a.col_indices.clone(),
        });
        let nnz = pattern.nnz();
        let opts = PcgOptions {
            tol: 1e-14,
            max_iters: 1000,
        };
        for precond in [Precond::Jacobi, Precond::Ic0] {
            let ls = Arc::new(LinearSolve::new(pattern.clone(), precond, opts));
            let mut g = TensorGraph::new();
            let vals = g.input(&[nnz]);
            let b = g.input(&[n]);
            let rows = g.input(&[nnz]);
            let cols = g.input(&[nnz]);
            let w = g.input(&[n]);
            let x = ls.apply(&mut g, vals, b, rows, cols).unwrap();
            let wx = g.map(&[w, x], |v| v[0] * v[1]).unwrap();
            let j = g.reduce(wx, 0, Monoid::sum()).unwrap();
            let grads = g.backward(j, &[vals, b]).unwrap();
            let rows_f: Vec<f64> = pattern.rows().iter().map(|&r| r as f64).collect();
            let cols_f: Vec<f64> = pattern.cols.iter().map(|&c| c as f64).collect();
            let wv: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
            let inputs = |v: &[f64], bb: &[f64]| {
                vec![v.to_vec(), bb.to_vec(), rows_f.clone(), cols_f.clone(), wv.clone()]
            };
            let b0 = rhs(n);
            let out = g.eval(&inputs(&a.values, &b0), &[grads[0], grads[1]]);
            let jv = |v: &[f64], bb: &[f64]| g.eval(&inputs(v, bb), &[j])[0][0];
            let h = 1e-6;
            for k in [0, 3, nnz / 2, nnz - 1] {
                let (mut p, mut m) = (a.values.clone(), a.values.clone());
                p[k] += h;
                m[k] -= h;
                let fd = (jv(&p, &b0) - jv(&m, &b0)) / (2.0 * h);
                assert!((fd - out[0][k]).abs() < 1e-6 * (1.0 + fd.abs()), "{precond:?} A[{k}]");
            }
            for i in [0, n - 1] {
                let (mut p, mut m) = (b0.clone(), b0.clone());
                p[i] += h;
                m[i] -= h;
                let fd = (jv(&a.values, &p) - jv(&a.values, &m)) / (2.0 * h);
                assert!((fd - out[1][i]).abs() < 1e-6 * (1.0 + fd.abs()), "{precond:?} b[{i}]");
            }
            assert!(ls.records().iter().any(|r| r.transpose));
        }
    }

    #[test]
    fn indexed_gather_adjoint_is_a_gather() {
        let index: Vec<i64> = vec![2, 0, -1, 2, 1, 2];
        let (inv, w) = IndexedGather::tables(&index, 3);
        let mut g = TensorGraph::new();
        let src = g.input(&[3]);
        let ix = g.input(&[2, 3]);
        let iv = g.input(&[3, w]);
        let out = IndexedGather::apply(&mut g, src, ix, iv).unwrap();
        let wts = g.input(&[2, 3]);
        let p = g.map(&[out, wts], |v| v[0] * v[1]).unwrap();
        let s1 = g.reduce(p, 1, Monoid::sum()).unwrap();
        let j = g.reduce(s1, 0, Monoid::sum()).unwrap();
        let gs = g.backward(j, &[src]).unwrap();
        let ins = vec![
            vec![10.0, 20.0, 30.0],
            index.iter().map(|&i| i as f64).collect(),
            inv,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ];
        let o = g.eval(&ins, &[out, gs[0]]);
        assert_eq!(o[0], vec![30.0, 10.0, 0.0, 30.0, 20.0, 30.0]);
        assert_eq!(o[1], vec![2.0, 5.0, 1.0 + 4.0 + 6.0]);
    }
}
