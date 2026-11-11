//! Scans: inclusive or exclusive prefix combines over one dim.
//!
//! A [`Scan`] node is `out[…, t, …] = e[…, 0, …] ⊕ … ⊕ e[…, t, …]` for an
//! associative `⊕`. Two kinds of monoid:
//!
//! - **[`ScanKind::Monoid`]**: a tuple monoid ([`Monoid`], the same type a
//!   reduction uses), one input tensor per state slot. Cumulative sums,
//!   running maxima, the scalar affine monoid `(a, b)` of a gated
//!   recurrence.
//! - **[`ScanKind::Affine`]**: the matrix affine monoid. Inputs `A
//!   [P…, n, n]`, `U [P…, n, m]`; the state is a matrix and
//!   `S_t = A_t·S_{t−1} + U_t` (`S_{−1} = 0`). Its combine, `(A₂, U₂) ∘
//!   (A₁, U₁) = (A₂A₁, A₂U₁ + U₂)`, is associative, so this is a scan, and
//!   it is the state recurrence of every linear-attention-like layer:
//!   linear attention (`A = I`), Mamba-2's SSD (`A = αI`), DeltaNet and
//!   the gated delta rule (`A = α(I − β·k·kᵀ)`), with `U = β·k·vᵀ`.
//!
//! **Backward.** The cotangent of a scan is a reverse scan with the
//! transposed combine. For the affine monoid that is exact and closed:
//! `λ_t = G_t + A_{t+1}ᵀ·λ_{t+1}` (an affine scan the other way), then
//! `∂U_t = λ_t` and `∂A_t = λ_t·S_{t−1}ᵀ`. A tuple monoid reduces to the
//! same thing: `λ_t = G_t + J_{t+1}ᵀ·λ_{t+1}` with `J` the Jacobian of the
//! combine in its accumulator, which is an affine scan over `w×w`
//! matrices built from `tang-expr` derivatives of the combine, then
//! `∂e_t = J^eᵀ_t·λ_t`. Every derivative is derived from the monoid; no
//! scan has a hand-written adjoint.
//!
//! **Chunked schedule.** An affine scan built by
//! [`TensorGraph::delta_scan`] remembers its structure ([`DeltaForm`]),
//! and a readout of it ([`TensorGraph::scan_readout`]) lowers on CUDA to
//! the chunked (SSD / chunked-delta-rule) form: inside a chunk the scan
//! is a handful of small matmuls on tensor cores (the WY/UT transform
//! turns the product of `I − β·k·kᵀ` factors into one triangular solve),
//! and only the `n×m` state crosses chunk boundaries, through a
//! sequential scan over chunks. See `tensor_cuda_scan.rs`.

use std::fmt::Write;

use super::wgsl::{load, store, WgslKernel};
use super::*;

/// Which inputs of an affine [`Scan`] its `A` and `U` were built from:
/// `A_t = α_t·(I − [delta]·β_t·k_t·k_tᵀ)`, `U_t = β_t·k_t·v_tᵀ`, with
/// `α`, `β` (shape `[P…]`) 1 when absent. Indices into
/// [`Scan::inputs`]; `k` is `[P…, n]`, `v` is `[P…, m]`. The interpreter
/// and the backward use `A` and `U`; the chunked schedule uses the form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaForm {
    pub alpha: Option<usize>,
    pub beta: Option<usize>,
    pub k: usize,
    pub v: usize,
    pub delta: bool,
}

#[derive(Clone)]
pub enum ScanKind {
    /// A tuple monoid; input `s` is state slot `s`.
    Monoid(Monoid),
    /// The matrix affine monoid; inputs `[A, U, …]` (anything after `U`
    /// is only referenced by `form`).
    ///
    /// With `bands`, `A` is banded and stored `[P…, n, k]`: entry
    /// `A[…, i, b]` is the dense `A[…, i, i + bands[b]]` (offsets strictly
    /// increasing; entries whose column falls outside `0..n` are ignored),
    /// so a step costs `O(n·k)` instead of `O(n²)`. A CTC lattice is
    /// `bands = [-2, -1, 0]`, a monotonic alignment `[-1, 0]`.
    Affine {
        form: Option<DeltaForm>,
        ring: Semiring,
        bands: Option<Vec<i64>>,
    },
}

/// The scalar semiring an affine scan's `A_t ⊗ S_{t−1} ⊕ U_t` is taken in.
///
/// - [`Real`](Semiring::Real): `(+, ×)`, the ordinary matrix recurrence.
/// - [`Log`](Semiring::Log): `(logsumexp, +)`. With `A` a log transition
///   matrix the state is the forward variable of an HMM / CTC lattice, and
///   the last state is `log Z`: the sum over all paths.
/// - [`MaxPlus`](Semiring::MaxPlus): `(max, +)`, the tropical semiring.
///   Same lattice, best path instead of all paths (Viterbi, MAS, DTW).
///
/// **Backward.** `∂S_t[i]/∂S_{t−1}[j]` is `J_t[i,j] = exp(A_t[i,j] +
/// S_{t−1}[j] − S_t[i])` in the log semiring (each row a distribution over
/// predecessors) and the argmax indicator in max-plus. The cotangent
/// recurrence `λ_t = G_t + J_{t+1}ᵀ·λ_{t+1}` is therefore an ordinary
/// *real* affine scan the other way: the backward pass of the forward
/// algorithm is the backward algorithm, and `∂ log Z/∂A` are the edge
/// posteriors (max-plus: the best path's edges). Derived for vector
/// states (`m = 1`), inclusive scans; other shapes are rejected when the
/// scan is built. Reverse scans are supported.
///
/// **Max-plus subgradient.** On a tie the whole gradient goes to one
/// predecessor: the one with the smallest index `j`, then `U` (as if it
/// were index `n`), matching [`Monoid::argmax`] and top-k. The result is
/// the gradient of the max-plus scan along the first-maximum path, an
/// element of the subdifferential, and does not depend on evaluation
/// order.
///
/// **Log combine.** A row `⊕_j (A[i,j] + S[j]) ⊕ U[i]` is computed in the
/// online-softmax form (`m' = max(m, x)`, `l' = l·exp(m − m') +
/// exp(x − m')`, result `m + log l`): the same stabilized logsumexp
/// [`Monoid::online_softmax`] that [`TensorGraph::stabilize_softmax`]
/// rewrites softmax denominators to, with one `log` per row.
///
/// The semiring zero is [`Semiring::NEG`] (a large finite negative, not
/// `−∞`, so `f32` kernels and `0·∞` never meet); use it for forbidden
/// transitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Semiring {
    #[default]
    Real,
    Log,
    MaxPlus,
}

impl Semiring {
    /// The additive identity of the log and max-plus semirings.
    pub const NEG: f64 = -1e30;

    pub fn zero(self) -> f64 {
        match self {
            Semiring::Real => 0.0,
            _ => Self::NEG,
        }
    }

    /// `u ⊕ ⊕ᵢ (aᵢ ⊗ sᵢ)`, the log semiring in the online-softmax form.
    fn row(self, u: f64, terms: impl Iterator<Item = (f64, f64)>) -> f64 {
        match self {
            Semiring::Real => terms.fold(u, |acc, (a, s)| acc + a * s),
            Semiring::MaxPlus => terms.fold(u, |acc, (a, s)| acc.max(a + s)),
            Semiring::Log => {
                let (m, l) = terms.fold((u, 1.0), |(m, l), (a, s)| {
                    let x = a + s;
                    let m2 = m.max(x);
                    (m2, l * (m - m2).exp() + (x - m2).exp())
                });
                m + l.ln()
            }
        }
    }
}

/// A prefix scan along point dim `dim` (see the module docs).
///
/// `reverse` scans from the end: a suffix product in the original order,
/// `out_t = e_t ⊕ e_{t+1} ⊕ …` (affine: `S_t = A_t·S_{t+1} + U_t`).
/// `exclusive` shifts by one: `out_t` is the state before `e_t` (the
/// identity, or `0`, at the first step).
#[derive(Clone)]
pub struct Scan {
    pub inputs: Vec<TensorId>,
    pub dim: usize,
    pub exclusive: bool,
    pub reverse: bool,
    pub kind: ScanKind,
}

impl Scan {
    /// The reference interpreter: `ins` row-major with `shapes`.
    pub(super) fn eval(&self, ins: &[&[f64]], shapes: &[Vec<usize>], _out: &[usize]) -> Vec<f64> {
        let ps: &[usize] = match &self.kind {
            ScanKind::Monoid(_) => &shapes[0],
            ScanKind::Affine { .. } => &shapes[0][..shapes[0].len() - 2],
        };
        let tn = ps[self.dim];
        let outer: usize = ps[..self.dim].iter().product();
        let inner: usize = ps[self.dim + 1..].iter().product();
        let npts = outer * tn * inner;
        let order = |step: usize| if self.reverse { tn - 1 - step } else { step };
        match &self.kind {
            ScanKind::Monoid(m) => {
                let w = m.width;
                let mut out = vec![0.0; npts * w];
                let mut vars = vec![0.0; 2 * w];
                for o in 0..outer {
                    for i in 0..inner {
                        let mut s = m.identity.clone();
                        for step in 0..tn {
                            let p = (o * tn + order(step)) * inner + i;
                            if self.exclusive {
                                out[p * w..(p + 1) * w].copy_from_slice(&s);
                            }
                            // forward: s ⊕ e; reverse: e ⊕ s
                            for k in 0..w {
                                let e = ins[k][p];
                                if self.reverse {
                                    vars[k] = e;
                                    vars[w + k] = s[k];
                                } else {
                                    vars[k] = s[k];
                                    vars[w + k] = e;
                                }
                            }
                            s = m.combine.eval_many(&m.outs, &vars);
                            if !self.exclusive {
                                out[p * w..(p + 1) * w].copy_from_slice(&s);
                            }
                        }
                    }
                }
                out
            }
            ScanKind::Affine { ring, bands, .. } => {
                let n = shapes[1][shapes[1].len() - 2];
                let k = shapes[0][shapes[0].len() - 1];
                let m = shapes[1][shapes[1].len() - 1];
                let mut out = vec![0.0; npts * n * m];
                let mut s = vec![0.0; n * m];
                let mut nx = vec![0.0; n * m];
                for o in 0..outer {
                    for i in 0..inner {
                        s.iter_mut().for_each(|x| *x = ring.zero());
                        for step in 0..tn {
                            let p = (o * tn + order(step)) * inner + i;
                            if self.exclusive {
                                out[p * n * m..(p + 1) * n * m].copy_from_slice(&s);
                            }
                            let a = &ins[0][p * n * k..(p + 1) * n * k];
                            let u = &ins[1][p * n * m..(p + 1) * n * m];
                            for r in 0..n {
                                for c in 0..m {
                                    // (A[r, j], S[j]) for every stored entry in range
                                    let terms = (0..k).filter_map(|b| {
                                        let j = match bands {
                                            Some(o) => r as i64 + o[b],
                                            None => b as i64,
                                        };
                                        let j = usize::try_from(j).ok().filter(|&j| j < n)?;
                                        Some((a[r * k + b], s[j * m + c]))
                                    });
                                    nx[r * m + c] = ring.row(u[r * m + c], terms);
                                }
                            }
                            std::mem::swap(&mut s, &mut nx);
                            if !self.exclusive {
                                out[p * n * m..(p + 1) * n * m].copy_from_slice(&s);
                            }
                        }
                    }
                }
                out
            }
        }
    }

    /// The point shape (the elements' shape, without the state dims).
    fn points(&self, g: &TensorGraph) -> Vec<Dim> {
        let s = g.shape(self.inputs[0]);
        match &self.kind {
            ScanKind::Monoid(_) => s.to_vec(),
            ScanKind::Affine { .. } => s[..s.len() - 2].to_vec(),
        }
    }
}

/// A plain (unguarded unless `guard`) view from forms.
fn view(forms: Vec<Form>, guard: bool) -> View {
    View {
        forms,
        guard,
        gather: None,
    }
}

/// `unit(rank, d)` shifted by `off`.
fn shifted(rank: usize, d: usize, off: i64) -> Form {
    let mut f = Form::unit(rank, d);
    f.offset = off;
    f
}

impl TensorGraph {
    fn push_scan(&mut self, sc: Scan, shape: Vec<Dim>) -> TensorId {
        self.nodes.push(TNode {
            op: Op::Scan(sc),
            shape,
            inline: false,
            split: 0,
            tile: (0, 0, 0),
            reg: (1, 1),
            fragment: false,
            int_mma: None,
            subgroup: false,
            frag_cfg: [0; 4],
            host: None,
            segment: None,
            sorted: false,
            detach: false,
            precision: Precision::default(),
        });
        TensorId(self.nodes.len() as u32 - 1)
    }

    /// Which nodes a live scan reads. A scan kernel reads each input as a
    /// buffer in the layout the scan was built with, so these stay
    /// materialized and keep their dim order.
    pub(super) fn scan_inputs(&self, live: &[bool]) -> Vec<bool> {
        let mut read = vec![false; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            if let (true, Op::Scan(s)) = (live[i], &node.op) {
                for t in &s.inputs {
                    read[t.n()] = true;
                }
            }
        }
        read
    }

    /// The [`Scan`] at `t`, if it is one.
    pub fn scan_at(&self, t: TensorId) -> Option<&Scan> {
        match &self.nodes[t.n()].op {
            Op::Scan(s) => Some(s),
            _ => None,
        }
    }

    /// A tuple-monoid scan of `elems` (one tensor per state slot, all the
    /// same shape) along `dim`. The output is the elements' shape, plus a
    /// trailing dim of the monoid's width when it is wider than 1.
    pub fn scan(
        &mut self,
        elems: &[TensorId],
        dim: usize,
        monoid: Monoid,
        exclusive: bool,
        reverse: bool,
    ) -> Result<TensorId, ShapeError> {
        if elems.len() != monoid.width {
            return err(format!(
                "scan: {} elements for a monoid of width {}",
                elems.len(),
                monoid.width
            ));
        }
        let ps = self.shape(elems[0]).to_vec();
        if elems.iter().any(|e| self.shape(*e) != ps) {
            return err("scan: elements differ in shape");
        }
        if dim >= ps.len() {
            return err(format!("scan dim {dim} of {ps:?}"));
        }
        let mut shape = ps;
        if monoid.width > 1 {
            shape.push(Dim::Fixed(monoid.width));
        }
        Ok(self.push_scan(
            Scan {
                inputs: elems.to_vec(),
                dim,
                exclusive,
                reverse,
                kind: ScanKind::Monoid(monoid),
            },
            shape,
        ))
    }

    /// The matrix affine scan `S_t = A_t·S_{t−1} + U_t` along `dim` of
    /// `A [P…, n, n]`, `U [P…, n, m]`; the output is `U`'s shape.
    pub fn linear_scan(
        &mut self,
        a: TensorId,
        u: TensorId,
        dim: usize,
        exclusive: bool,
        reverse: bool,
    ) -> Result<TensorId, ShapeError> {
        self.affine_scan(
            vec![a, u],
            dim,
            exclusive,
            reverse,
            None,
            Semiring::Real,
            None,
        )
    }

    /// [`linear_scan`](Self::linear_scan) in another [`Semiring`]:
    /// `S_t = A_t ⊗ S_{t−1} ⊕ U_t`, `S_{−1}` the semiring zero. In the log
    /// semiring this is the forward algorithm; in max-plus, Viterbi.
    ///
    /// Log and max-plus scans need a vector state (`U [P…, n, 1]`) and
    /// `exclusive = false`, the shapes their backward is derived for.
    pub fn semiring_scan(
        &mut self,
        a: TensorId,
        u: TensorId,
        dim: usize,
        exclusive: bool,
        reverse: bool,
        ring: Semiring,
    ) -> Result<TensorId, ShapeError> {
        self.affine_scan(vec![a, u], dim, exclusive, reverse, None, ring, None)
    }

    /// [`semiring_scan`](Self::semiring_scan) with a banded transition:
    /// `A [P…, n, k]` holds `A[…, i, b] = A_dense[…, i, i + bands[b]]`
    /// (see [`ScanKind::Affine`]), `U [P…, n, 1]`, inclusive. Any
    /// semiring, `Real` included; one step is `O(n·k)`.
    pub fn semiring_scan_banded(
        &mut self,
        a: TensorId,
        u: TensorId,
        bands: &[i64],
        dim: usize,
        reverse: bool,
        ring: Semiring,
    ) -> Result<TensorId, ShapeError> {
        self.affine_scan(
            vec![a, u],
            dim,
            false,
            reverse,
            None,
            ring,
            Some(bands.to_vec()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn affine_scan(
        &mut self,
        inputs: Vec<TensorId>,
        dim: usize,
        exclusive: bool,
        reverse: bool,
        form: Option<DeltaForm>,
        ring: Semiring,
        bands: Option<Vec<i64>>,
    ) -> Result<TensorId, ShapeError> {
        let (sa, su) = (
            self.shape(inputs[0]).to_vec(),
            self.shape(inputs[1]).to_vec(),
        );
        let r = su.len();
        // A's last dim: n dense, k = bands.len() banded
        let cols = match &bands {
            Some(o) => Dim::Fixed(o.len()),
            None => su.get(r.wrapping_sub(2)).copied().unwrap_or(Dim::Fixed(0)),
        };
        let ok = r >= 3
            && sa.len() == r
            && sa[..r - 2] == su[..r - 2]
            && sa[r - 2] == su[r - 2]
            && sa[r - 1] == cols
            && dim < r - 2;
        if !ok {
            return err(format!("linear_scan: A {sa:?}, U {su:?}, dim {dim}"));
        }
        if let Some(o) = &bands {
            if o.is_empty() || o.windows(2).any(|w| w[0] >= w[1]) {
                return err(format!(
                    "banded scan: offsets {o:?} must be strictly increasing"
                ));
            }
        }
        // the shapes the non-dense-real backward is derived for
        if (ring != Semiring::Real || bands.is_some()) && (su[r - 1] != Dim::Fixed(1) || exclusive)
        {
            return err(format!(
                "{ring:?} scan{}: needs a vector state (U [.., n, 1]) and an inclusive scan, \
                 got U {su:?}, exclusive {exclusive}",
                if bands.is_some() { " (banded)" } else { "" }
            ));
        }
        Ok(self.push_scan(
            Scan {
                inputs,
                dim,
                exclusive,
                reverse,
                kind: ScanKind::Affine { form, ring, bands },
            },
            su,
        ))
    }

    /// The delta-rule state scan along `dim`: `S_t = α_t·(I −
    /// [delta]·β_t·k_t·k_tᵀ)·S_{t−1} + β_t·k_t·v_tᵀ` for `k [P…, n]`, `v
    /// [P…, m]`, `α`, `β` `[P…]` (1 when `None`). `A` and `U` are built
    /// as comprehensions, so the interpreter, the backward and every
    /// backend see an ordinary affine scan; the scan also records the
    /// [`DeltaForm`] for the chunked schedule.
    ///
    /// - linear attention: `alpha: None, beta: None, delta: false`
    /// - Mamba-2 SSD: `α = exp(Δ·A)`, `β = Δ`, `k = B`, `v = x`, no delta
    /// - gated delta rule (Qwen3-Next GDN): `α = exp(g)`, `β`, `delta: true`
    pub fn delta_scan(
        &mut self,
        k: TensorId,
        v: TensorId,
        alpha: Option<TensorId>,
        beta: Option<TensorId>,
        dim: usize,
        delta: bool,
    ) -> Result<TensorId, ShapeError> {
        let sk = self.shape(k).to_vec();
        let sv = self.shape(v).to_vec();
        let p = sk.len() - 1;
        let ps = sk[..p].to_vec();
        if sv[..sv.len() - 1] != ps[..] {
            return err(format!("delta_scan: k {sk:?}, v {sv:?}"));
        }
        for x in [alpha, beta].into_iter().flatten() {
            if self.shape(x) != &ps[..] {
                return err(format!("delta_scan: gate {:?} vs {ps:?}", self.shape(x)));
            }
        }
        let (n, m) = (sk[p], sv[p]);
        let pts: Vec<Option<usize>> = (0..p).map(Some).collect();
        let pick = |extra: &[Option<usize>]| {
            let mut v = pts.clone();
            v.extend_from_slice(extra);
            View::pick(p + 2, &v)
        };
        // A[P…, i, j] = α·(δᵢⱼ − [delta]·β·kᵢ·kⱼ)
        let ii = self.iota(&[n, n], 0);
        let jj = self.iota(&[n, n], 1);
        let mut args = vec![
            Arg {
                src: ii,
                view: View::pick(p + 2, &[Some(p), Some(p + 1)]),
            },
            Arg {
                src: jj,
                view: View::pick(p + 2, &[Some(p), Some(p + 1)]),
            },
            Arg {
                src: k,
                view: pick(&[Some(p)]),
            },
            Arg {
                src: k,
                view: pick(&[Some(p + 1)]),
            },
        ];
        let ia = alpha.map(|a| {
            args.push(Arg {
                src: a,
                view: pick(&[]),
            });
            args.len() - 1
        });
        let ib = beta.map(|b| {
            args.push(Arg {
                src: b,
                view: pick(&[]),
            });
            args.len() - 1
        });
        let mut ashape = ps.clone();
        ashape.extend([n, n]);
        let a = self.comp(&ashape, &[], args, None, |x| {
            let d = x[0] - x[1];
            let half = ExprId::from_f64(0.5);
            let c = half - d * d;
            let eye = tang_expr::with_graph(|g| g.select(c, ExprId::ONE, ExprId::ZERO));
            let mut v = if delta {
                let kk = x[2] * x[3];
                let bkk = ib.map_or(kk, |b| x[b] * kk);
                eye - bkk
            } else {
                eye
            };
            if let Some(a) = ia {
                v = x[a] * v;
            }
            vec![v]
        })?;
        // U[P…, i, c] = β·kᵢ·v_c
        let mut uargs = vec![
            Arg {
                src: k,
                view: pick(&[Some(p)]),
            },
            Arg {
                src: v,
                view: pick(&[Some(p + 1)]),
            },
        ];
        if let Some(b) = beta {
            uargs.push(Arg {
                src: b,
                view: pick(&[]),
            });
        }
        let mut ushape = ps.clone();
        ushape.extend([n, m]);
        let has_b = beta.is_some();
        let u = self.comp(&ushape, &[], uargs, None, |x| {
            let kv = x[0] * x[1];
            vec![if has_b { x[2] * kv } else { kv }]
        })?;
        let mut inputs = vec![a, u, k, v];
        let alpha_i = alpha.map(|t| {
            inputs.push(t);
            inputs.len() - 1
        });
        let beta_i = beta.map(|t| {
            inputs.push(t);
            inputs.len() - 1
        });
        let form = DeltaForm {
            alpha: alpha_i,
            beta: beta_i,
            k: 2,
            v: 3,
            delta,
        };
        self.affine_scan(inputs, dim, false, false, Some(form), Semiring::Real, None)
    }

    /// `o[P…, c] = scale · Σᵢ S[P…, i, c]·q[P…, i]`: the output of a
    /// linear-attention-like layer from its state scan `s [P…, n, m]` and
    /// queries `q [P…, n]`.
    pub fn scan_readout(
        &mut self,
        s: TensorId,
        q: TensorId,
        scale: f64,
    ) -> Result<TensorId, ShapeError> {
        let ss = self.shape(s).to_vec();
        let r = ss.len();
        let p = r - 2;
        let mut out = ss[..p].to_vec();
        out.push(ss[r - 1]);
        let pts: Vec<Option<usize>> = (0..p).map(Some).collect();
        // space [P…, c, i]
        let mut sp = pts.clone();
        sp.extend([Some(p + 1), Some(p)]);
        let mut qp = pts.clone();
        qp.push(Some(p + 1));
        self.comp(
            &out,
            &[ss[p]],
            vec![
                Arg {
                    src: s,
                    view: View::pick(p + 2, &sp),
                },
                Arg {
                    src: q,
                    view: View::pick(p + 2, &qp),
                },
            ],
            Some(Monoid::sum()),
            move |x| vec![ExprId::from_f64(scale) * x[0] * x[1]],
        )
    }

    /// The input cotangents of scan node `j` given its output cotangent
    /// `dy` (see the module docs), appended to the graph.
    pub(super) fn scan_backward(
        &mut self,
        j: usize,
        dy: TensorId,
    ) -> Result<Vec<Option<TensorId>>, ShapeError> {
        let sc = self.scan_at(TensorId(j as u32)).unwrap().clone();
        let ps = sc.points(self);
        let q = ps.len();
        let dim = sc.dim;
        // the step the recurrence takes: S_t depends on S_{t−d}
        let d: i64 = if sc.reverse { -1 } else { 1 };
        let out = TensorId(j as u32);
        match &sc.kind {
            ScanKind::Affine { ring, bands, .. } if *ring != Semiring::Real || bands.is_some() => {
                self.semiring_backward(&sc, *ring, bands.clone(), out, dy)
            }
            ScanKind::Affine { .. } => {
                let (a, sa) = (sc.inputs[0], self.shape(sc.inputs[0]).to_vec());
                let su = self.shape(sc.inputs[1]).to_vec();
                let r = q + 2;
                // A'_t = A_{t+d}ᵀ (0 past the end)
                let mut f: Vec<Form> = (0..q).map(|x| Form::unit(r, x)).collect();
                f[dim] = shifted(r, dim, d);
                f.push(Form::unit(r, q + 1));
                f.push(Form::unit(r, q));
                let at = self.comp(
                    &sa,
                    &[],
                    vec![Arg {
                        src: a,
                        view: view(f, true),
                    }],
                    None,
                    |x| vec![x[0]],
                )?;
                // the cotangent of the internal (inclusive) state
                let gint = if sc.exclusive {
                    let mut f: Vec<Form> = (0..r).map(|x| Form::unit(r, x)).collect();
                    f[dim] = shifted(r, dim, d);
                    self.comp(
                        &su,
                        &[],
                        vec![Arg {
                            src: dy,
                            view: view(f, true),
                        }],
                        None,
                        |x| vec![x[0]],
                    )?
                } else {
                    dy
                };
                let lam = self.linear_scan(at, gint, dim, false, !sc.reverse)?;
                // S_{t−d}: the exclusive output itself, or the inclusive one shifted
                let sprev = if sc.exclusive {
                    out
                } else {
                    let mut f: Vec<Form> = (0..r).map(|x| Form::unit(r, x)).collect();
                    f[dim] = shifted(r, dim, -d);
                    self.comp(
                        &su,
                        &[],
                        vec![Arg {
                            src: out,
                            view: view(f, true),
                        }],
                        None,
                        |x| vec![x[0]],
                    )?
                };
                // ∂A[P…, i, j] = Σ_c λ[P…, i, c]·S_prev[P…, j, c]; space [P…, i, j, c]
                let r4 = q + 3;
                let mut lf: Vec<Form> = (0..q).map(|x| Form::unit(r4, x)).collect();
                lf.extend([Form::unit(r4, q), Form::unit(r4, q + 2)]);
                let mut sf: Vec<Form> = (0..q).map(|x| Form::unit(r4, x)).collect();
                sf.extend([Form::unit(r4, q + 1), Form::unit(r4, q + 2)]);
                let ga = self.comp(
                    &sa,
                    &[su[q + 1]],
                    vec![
                        Arg {
                            src: lam,
                            view: view(lf, false),
                        },
                        Arg {
                            src: sprev,
                            view: view(sf, false),
                        },
                    ],
                    Some(Monoid::sum()),
                    |x| vec![x[0] * x[1]],
                )?;
                let mut gs = vec![Some(ga), Some(lam)];
                gs.resize(sc.inputs.len(), None);
                Ok(gs)
            }
            ScanKind::Monoid(m) => {
                let w = m.width;
                let tn = ps[dim]
                    .fixed()
                    .ok_or_else(|| ShapeError("scan backward needs a bound scan dim".into()))?;
                // Jacobians of F(p, e) (= p ⊕ e forward, e ⊕ p reverse) in p and e
                let mut g2 = ExprGraph::new();
                let pv: Vec<ExprId> = (0..w).map(|i| g2.var(i as u16)).collect();
                let ev: Vec<ExprId> = (0..w).map(|i| g2.var((w + i) as u16)).collect();
                let vars: Vec<ExprId> = if sc.reverse {
                    ev.iter().chain(&pv).copied().collect()
                } else {
                    pv.iter().chain(&ev).copied().collect()
                };
                let fo = replay(&m.combine, &m.outs, &vars, &mut g2);
                let mut jp = Vec::with_capacity(w * w);
                let mut je = Vec::with_capacity(w * w);
                for &fa in &fo {
                    for b in 0..w {
                        jp.push(g2.diff(fa, b as u16));
                    }
                }
                for &fa in &fo {
                    for b in 0..w {
                        je.push(g2.diff(fa, (w + b) as u16));
                    }
                }
                // args: prev state slots (raw), element slots, the scan coordinate
                let mut args = Vec::new();
                for s in 0..w {
                    let mut f: Vec<Form> = (0..q).map(|x| Form::unit(q, x)).collect();
                    if !sc.exclusive {
                        f[dim] = shifted(q, dim, -d);
                    }
                    if w > 1 {
                        f.push(Form::constant(q, s as i64));
                    }
                    args.push(Arg {
                        src: out,
                        view: view(f, !sc.exclusive),
                    });
                }
                for &e in &sc.inputs {
                    args.push(Arg {
                        src: e,
                        view: View::identity(q, q),
                    });
                }
                let iota = self.iota(&ps, dim);
                args.push(Arg {
                    src: iota,
                    view: View::identity(q, q),
                });
                let id = m.identity.clone();
                let (excl, rev) = (sc.exclusive, sc.reverse);
                let jac = |g: &mut TensorGraph, outs: Vec<ExprId>| {
                    let mut shape = ps.clone();
                    shape.push(Dim::Fixed(w * w));
                    let g2 = g2.clone();
                    let id = id.clone();
                    let flat = g.comp_tuple(&shape, &[], args.clone(), None, move |x| {
                        // boundary: the first step has the identity as its prev
                        let t = x[2 * w];
                        let c = if rev {
                            t - ExprId::from_f64(tn as f64 - 1.5)
                        } else {
                            ExprId::from_f64(0.5) - t
                        };
                        let mut vars: Vec<ExprId> = (0..w)
                            .map(|s| {
                                if excl {
                                    x[s]
                                } else {
                                    let idv = ExprId::from_f64(id[s]);
                                    tang_expr::with_graph(|gr| gr.select(c, idv, x[s]))
                                }
                            })
                            .collect();
                        vars.extend((0..w).map(|s| x[w + s]));
                        tang_expr::with_graph(|gr| replay(&g2, &outs, &vars, gr))
                    })?;
                    let mut shape = ps.clone();
                    shape.extend([Dim::Fixed(w), Dim::Fixed(w)]);
                    g.reshape(flat, &shape)
                };
                let jpt = jac(self, jp)?;
                let jet = jac(self, je)?;
                // A'_t[b, a] = Jp_{t+d}[a, b]
                let r = q + 2;
                let mut f: Vec<Form> = (0..q).map(|x| Form::unit(r, x)).collect();
                f[dim] = shifted(r, dim, d);
                f.extend([Form::unit(r, q + 1), Form::unit(r, q)]);
                let mut sa = ps.clone();
                sa.extend([Dim::Fixed(w), Dim::Fixed(w)]);
                let at = self.comp(
                    &sa,
                    &[],
                    vec![Arg {
                        src: jpt,
                        view: view(f, true),
                    }],
                    None,
                    |x| vec![x[0]],
                )?;
                // U' = the internal state's cotangent as [P…, w, 1]
                let mut f: Vec<Form> = (0..q).map(|x| Form::unit(r, x)).collect();
                if sc.exclusive {
                    f[dim] = shifted(r, dim, d);
                }
                if w > 1 {
                    f.push(Form::unit(r, q));
                }
                let mut su = ps.clone();
                su.extend([Dim::Fixed(w), Dim::Fixed(1)]);
                let gint = self.comp(
                    &su,
                    &[],
                    vec![Arg {
                        src: dy,
                        view: view(f, sc.exclusive),
                    }],
                    None,
                    |x| vec![x[0]],
                )?;
                let lam = self.linear_scan(at, gint, dim, false, !sc.reverse)?;
                // ∂e_b = Σ_a Je[P…, a, b]·λ[P…, a, 0]; space [P…, a]
                let r1 = q + 1;
                let mut gs = Vec::with_capacity(w);
                for b in 0..w {
                    let mut jf: Vec<Form> = (0..q).map(|x| Form::unit(r1, x)).collect();
                    jf.extend([Form::unit(r1, q), Form::constant(r1, b as i64)]);
                    let mut lf: Vec<Form> = (0..q).map(|x| Form::unit(r1, x)).collect();
                    lf.extend([Form::unit(r1, q), Form::constant(r1, 0)]);
                    let ge = self.comp(
                        &ps,
                        &[Dim::Fixed(w)],
                        vec![
                            Arg {
                                src: jet,
                                view: view(jf, false),
                            },
                            Arg {
                                src: lam,
                                view: view(lf, false),
                            },
                        ],
                        Some(Monoid::sum()),
                        |x| vec![x[0] * x[1]],
                    )?;
                    gs.push(Some(ge));
                }
                Ok(gs)
            }
        }
    }

    /// The backward of a log / max-plus affine scan, or of a banded one in
    /// any semiring (see [`Semiring`]): build the predecessor weights
    /// `J[i, b]` (`∂S_t[i]/∂S_{t−1}[j]`, `j` the column of entry `b`), run
    /// the real scan `λ_t = G_t + J_{t+1}ᵀ·λ_{t+1}` (banded with the
    /// mirrored offsets when `A` is), then `∂A`, `∂U` from `λ` and `J`.
    fn semiring_backward(
        &mut self,
        sc: &Scan,
        ring: Semiring,
        bands: Option<Vec<i64>>,
        out: TensorId,
        dy: TensorId,
    ) -> Result<Vec<Option<TensorId>>, ShapeError> {
        let ps = sc.points(self);
        let q = ps.len();
        let dim = sc.dim;
        let d: i64 = if sc.reverse { -1 } else { 1 };
        let (a, u) = (sc.inputs[0], sc.inputs[1]);
        let (sa, su) = (self.shape(a).to_vec(), self.shape(u).to_vec());
        if su[q + 1] != Dim::Fixed(1) || sc.exclusive {
            return err("semiring scan backward: needs a vector state (m = 1), inclusive");
        }
        let tn = ps[dim]
            .fixed()
            .ok_or_else(|| ShapeError("scan backward needs a bound scan dim".into()))?;
        let n = su[q]
            .fixed()
            .ok_or_else(|| ShapeError("scan backward needs a bound state dim".into()))?;
        let k = sa[q + 1].fixed().unwrap_or(n);
        let r = q + 2;
        let pts = |shift: i64| -> Vec<Form> {
            let mut f: Vec<Form> = (0..q).map(|x| Form::unit(r, x)).collect();
            f[dim] = shifted(r, dim, shift);
            f
        };
        let at = |mut f: Vec<Form>, i: Form, b: Form| {
            f.extend([i, b]);
            f
        };
        let c = |v: f64| ExprId::from_f64(v);
        // `x == v` for small integers, as a select condition (> 0 when equal)
        let is = move |x: ExprId, v: f64| c(0.5) - (x - c(v)) * (x - c(v));
        let iota_t = self.iota(&ps, dim);
        let ii = self.iota(&sa, q);
        let bb = self.iota(&sa, q + 1);
        let rev = sc.reverse;
        // Candidate args over space (P…, i, b): A, S_t[i], t, i, b, then the
        // predecessor S_{t−d}[j]: one read (dense, j = b) or one per band.
        let mut args = vec![
            Arg {
                src: a,
                view: View::identity(r, r),
            },
            Arg {
                src: out,
                view: view(at(pts(0), Form::unit(r, q), Form::constant(r, 0)), false),
            },
            Arg {
                src: iota_t,
                view: view(pts(0), false),
            },
            Arg {
                src: ii,
                view: View::identity(r, r),
            },
            Arg {
                src: bb,
                view: View::identity(r, r),
            },
        ];
        let offs: Vec<i64> = bands.clone().unwrap_or_default();
        match &bands {
            None => args.push(Arg {
                src: out,
                view: view(
                    at(pts(-d), Form::unit(r, q + 1), Form::constant(r, 0)),
                    true,
                ),
            }),
            Some(o) => args.extend(o.iter().map(|&off| Arg {
                src: out,
                view: view(at(pts(-d), shifted(r, q, off), Form::constant(r, 0)), true),
            })),
        }
        let base = args.len();
        // (ok ∈ {0, 1}, S_{t−d}[j], A + S_{t−d}[j]) from the candidate args
        let cand = {
            let offs = offs.clone();
            move |x: &[ExprId]| -> (ExprId, ExprId, ExprId) {
                // the first step's predecessor is the semiring zero: no weight
                let first = if rev {
                    x[2] - c(tn as f64 - 1.5)
                } else {
                    c(0.5) - x[2]
                };
                let (sp, inr) = if offs.is_empty() {
                    (x[5], c(1.0))
                } else {
                    let (mut sp, mut j) = (x[5], x[3] + c(offs[0] as f64));
                    for (b, &off) in offs.iter().enumerate().skip(1) {
                        sp = ExprId::select(is(x[4], b as f64), x[5 + b], sp);
                        j = ExprId::select(is(x[4], b as f64), x[3] + c(off as f64), j);
                    }
                    // 0 ≤ j < n
                    let inr = (j + c(0.5)) * (c(n as f64 - 0.5) - j);
                    (sp, inr)
                };
                let ok = ExprId::select(
                    first,
                    ExprId::ZERO,
                    ExprId::select(inr, ExprId::ONE, ExprId::ZERO),
                );
                (ok, sp, x[0] + sp)
            }
        };
        // max-plus: the winning entry per (P…, i), ties to the smaller
        // column, then U (index k) only if strictly larger
        let winner = if ring == Semiring::MaxPlus {
            let mut out_shape = sa.clone();
            out_shape[q + 1] = Dim::Fixed(2);
            let cand = cand.clone();
            let best = self.comp_tuple(
                &out_shape[..q + 2],
                &[sa[q + 1]],
                // reduce over b: the same views (the tuple dim is no space dim)
                args.clone(),
                Some(Monoid::argmax()),
                move |x| {
                    let (ok, _, v) = cand(x);
                    vec![x[4], ExprId::select(ok - c(0.5), v, c(-(f32::MAX as f64)))]
                },
            )?;
            let slot = |s: i64| {
                let mut f: Vec<Form> = (0..q + 1).map(|x| Form::unit(q + 1, x)).collect();
                f.push(Form::constant(q + 1, s));
                view(f, false)
            };
            let mut uf: Vec<Form> = (0..q + 1).map(|x| Form::unit(q + 1, x)).collect();
            uf.push(Form::constant(q + 1, 0));
            Some(self.comp(
                &su[..q + 1],
                &[] as &[Dim],
                vec![
                    Arg {
                        src: best,
                        view: slot(0),
                    },
                    Arg {
                        src: best,
                        view: slot(1),
                    },
                    Arg {
                        src: u,
                        view: view(uf, false),
                    },
                ],
                None,
                move |x| vec![ExprId::select(x[2] - x[1], c(k as f64), x[0])],
            )?)
        } else {
            None
        };
        // J[P…, i, b]
        let mut jargs = args.clone();
        if let Some(w) = winner {
            let f: Vec<Form> = (0..q + 1).map(|x| Form::unit(r, x)).collect();
            jargs.push(Arg {
                src: w,
                view: view(f, false),
            });
        }
        let jac = {
            let cand = cand.clone();
            self.comp(&sa, &[] as &[Dim], jargs, None, move |x| {
                let (ok, _, v) = cand(x);
                let w = match ring {
                    Semiring::Real => x[0],
                    Semiring::Log => (v - x[1]).exp(),
                    Semiring::MaxPlus => {
                        ExprId::select(is(x[base] - x[4], 0.0), ExprId::ONE, ExprId::ZERO)
                    }
                };
                vec![ExprId::select(ok - c(0.5), w, ExprId::ZERO)]
            })?
        };
        // A'_t = J_{t+d}ᵀ (0 past the end), then the real scan the other way
        let lam = match &bands {
            None => {
                let at_t = self.comp(
                    &sa,
                    &[] as &[Dim],
                    vec![Arg {
                        src: jac,
                        view: view(at(pts(d), Form::unit(r, q + 1), Form::unit(r, q)), true),
                    }],
                    None,
                    |x| vec![x[0]],
                )?;
                self.linear_scan(at_t, dy, dim, false, !rev)?
            }
            Some(o) => {
                // band b' of the transpose is band k−1−b mirrored:
                // A'[j, b'] = J_{t+d}[j − o[b], b]
                let mirror: Vec<i64> = o.iter().rev().map(|x| -x).collect();
                let mut targs = vec![Arg {
                    src: bb,
                    view: View::identity(r, r),
                }];
                targs.extend(o.iter().enumerate().map(|(b, &off)| Arg {
                    src: jac,
                    view: view(
                        at(pts(d), shifted(r, q, -off), Form::constant(r, b as i64)),
                        true,
                    ),
                }));
                let at_t = self.comp(&sa, &[] as &[Dim], targs, None, move |x| {
                    // x[1 + b] is band b, which sits at b' = k−1−b
                    let mut v = x[1];
                    for b in 1..k {
                        v = ExprId::select(is(x[0], (k - 1 - b) as f64), x[1 + b], v);
                    }
                    vec![v]
                })?;
                self.semiring_scan_banded(at_t, dy, &mirror, dim, !rev, Semiring::Real)?
            }
        };
        // ∂U = λ ⊙ w_U, w_U the weight of the U branch
        let gu = match (ring, winner) {
            (Semiring::Real, _) => lam,
            (Semiring::Log, _) => self.map(&[lam, u, out], |x| x[0] * (x[1] - x[2]).exp())?,
            (_, w) => {
                let w = w.expect("max-plus winner");
                let ws = self.reshape(w, &su)?;
                self.map(&[lam, ws], move |x| {
                    x[0] * ExprId::select(is(x[1], k as f64), ExprId::ONE, ExprId::ZERO)
                })?
            }
        };
        // ∂A[P…, i, b] = λ[i]·(real: ok·S_{t−d}[j]; else J[i, b])
        let mut gargs = args.clone();
        gargs.push(Arg {
            src: lam,
            view: view(at(pts(0), Form::unit(r, q), Form::constant(r, 0)), false),
        });
        gargs.push(Arg {
            src: jac,
            view: View::identity(r, r),
        });
        let nl = gargs.len();
        let ga = self.comp(&sa, &[] as &[Dim], gargs, None, move |x| {
            let w = match ring {
                Semiring::Real => {
                    let (ok, sp, _) = cand(x);
                    ExprId::select(ok - c(0.5), sp, ExprId::ZERO)
                }
                _ => x[nl - 1],
            };
            vec![x[nl - 2] * w]
        })?;
        Ok(vec![Some(ga), Some(gu)])
    }

    /// The WGSL kernel of scan node `j`: one thread per scanned line
    /// (and per state column, for the affine monoid) walking the scan
    /// dim sequentially. Correct for any monoid on every backend; the
    /// chunked schedule is the fast path for delta-rule scans on CUDA.
    pub(super) fn scan_kernel(&self, j: usize) -> WgslKernel {
        let sc = self.scan_at(TensorId(j as u32)).unwrap();
        let ps = fixed(&sc.points(self));
        let tn = ps[sc.dim];
        let outer: usize = ps[..sc.dim].iter().product();
        let inner: usize = ps[sc.dim + 1..].iter().product();
        let mut reads: Vec<usize> = Vec::new();
        for t in &sc.inputs {
            let s = self.storage(*t).n();
            if !reads.contains(&s) {
                reads.push(s);
            }
        }
        let bind = |t: TensorId| format!("b{}", self.storage(t).n());
        let fmt = |t: TensorId| self.nodes[self.storage(t).n()].precision.storage;
        let out_f = self.nodes[j].precision.storage;
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile from a TensorGraph (scan)"
        )
        .unwrap();
        writeln!(
            s,
            "struct Params {{ count: u32, _p1: u32, _p2: u32, _p3: u32 }}"
        )
        .unwrap();
        for (b, &r) in reads.iter().enumerate() {
            writeln!(
                s,
                "@group(0) @binding({b}) var<storage, read> b{r}: array<{}>;",
                super::wgsl::elem_ty(self.nodes[r].precision.storage)
            )
            .unwrap();
        }
        writeln!(
            s,
            "@group(0) @binding({}) var<storage, read_write> out: array<{}>;",
            reads.len(),
            super::wgsl::out_ty(out_f)
        )
        .unwrap();
        writeln!(
            s,
            "@group(0) @binding({}) var<uniform> params: Params;\n",
            reads.len() + 1
        )
        .unwrap();
        let step = |s: &mut String| {
            if sc.reverse {
                writeln!(s, "        let t = {}u - st;", tn - 1).unwrap();
            } else {
                writeln!(s, "        let t = st;").unwrap();
            }
            writeln!(s, "        let p = (po * {tn}u + t) * {inner}u + pi;").unwrap();
        };
        let lines;
        match &sc.kind {
            ScanKind::Monoid(m) => {
                let w = m.width;
                super::wgsl::emit_combine(&mut s, j, m, Format::F32);
                lines = outer * inner;
                writeln!(s, "@compute @workgroup_size(64)").unwrap();
                writeln!(
                    s,
                    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{"
                )
                .unwrap();
                writeln!(s, "    let g = gid.x + gid.y * {}u;", 65535 * 64).unwrap();
                writeln!(s, "    if (g >= params.count) {{ return; }}").unwrap();
                writeln!(s, "    let po = g / {inner}u;").unwrap();
                writeln!(s, "    let pi = g % {inner}u;").unwrap();
                let ids = m
                    .identity
                    .iter()
                    .map(|v| super::wgsl::lit(*v))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(s, "    var acc = array<f32, {w}>({ids});").unwrap();
                writeln!(s, "    for (var st: u32 = 0u; st < {tn}u; st = st + 1u) {{").unwrap();
                step(&mut s);
                let el = sc
                    .inputs
                    .iter()
                    .map(|t| load(fmt(*t), &bind(*t), "p"))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(s, "        let e = array<f32, {w}>({el});").unwrap();
                let put = |s: &mut String| {
                    for k in 0..w {
                        let st = store(out_f, &format!("p * {w}u + {k}u"), &format!("acc[{k}]"));
                        writeln!(s, "        {st}").unwrap();
                    }
                };
                if sc.exclusive {
                    put(&mut s);
                }
                if sc.reverse {
                    writeln!(s, "        acc = combine_{j}(e, acc);").unwrap();
                } else {
                    writeln!(s, "        acc = combine_{j}(acc, e);").unwrap();
                }
                if !sc.exclusive {
                    put(&mut s);
                }
                writeln!(s, "    }}").unwrap();
                writeln!(s, "}}").unwrap();
            }
            ScanKind::Affine { ring, bands, .. } => {
                let su = fixed(self.shape(sc.inputs[1]));
                let (n, m) = (su[su.len() - 2], su[su.len() - 1]);
                let zero = match ring {
                    Semiring::Real => "0.0",
                    _ => "-1e30",
                };
                lines = outer * inner * m;
                let (ta, tu) = (sc.inputs[0], sc.inputs[1]);
                writeln!(s, "@compute @workgroup_size(64)").unwrap();
                writeln!(
                    s,
                    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{"
                )
                .unwrap();
                writeln!(s, "    let g = gid.x + gid.y * {}u;", 65535 * 64).unwrap();
                writeln!(s, "    if (g >= params.count) {{ return; }}").unwrap();
                writeln!(s, "    let col = g % {m}u;").unwrap();
                writeln!(s, "    let line = g / {m}u;").unwrap();
                writeln!(s, "    let po = line / {inner}u;").unwrap();
                writeln!(s, "    let pi = line % {inner}u;").unwrap();
                writeln!(s, "    var sv: array<f32, {n}>;").unwrap();
                writeln!(s, "    var nv: array<f32, {n}>;").unwrap();
                writeln!(
                    s,
                    "    for (var i: u32 = 0u; i < {n}u; i = i + 1u) {{ sv[i] = {zero}; }}"
                )
                .unwrap();
                writeln!(s, "    for (var st: u32 = 0u; st < {tn}u; st = st + 1u) {{").unwrap();
                step(&mut s);
                let put = |s: &mut String| {
                    writeln!(s, "        for (var i: u32 = 0u; i < {n}u; i = i + 1u) {{").unwrap();
                    let st = store(out_f, &format!("(p * {n}u + i) * {m}u + col"), "sv[i]");
                    writeln!(s, "            {st}").unwrap();
                    writeln!(s, "        }}").unwrap();
                };
                if sc.exclusive {
                    put(&mut s);
                }
                writeln!(s, "        for (var i: u32 = 0u; i < {n}u; i = i + 1u) {{").unwrap();
                let u = load(fmt(tu), &bind(tu), &format!("(p * {n}u + i) * {m}u + col"));
                writeln!(s, "            var acc: f32 = {u};").unwrap();
                if *ring == Semiring::Log {
                    // online logsumexp: acc is the running max, l the scaled sum
                    writeln!(s, "            var l: f32 = 1.0;").unwrap();
                }
                // acc ⊕= a ⊗ sv[j]
                let combine = |s: &mut String, a: &str, j: &str, pad: &str| match ring {
                    Semiring::Real => writeln!(s, "{pad}acc = acc + {a} * sv[{j}];").unwrap(),
                    Semiring::Log => {
                        writeln!(s, "{pad}let x = {a} + sv[{j}];").unwrap();
                        writeln!(s, "{pad}let mx = max(acc, x);").unwrap();
                        writeln!(s, "{pad}l = l * exp(acc - mx) + exp(x - mx);").unwrap();
                        writeln!(s, "{pad}acc = mx;").unwrap();
                    }
                    Semiring::MaxPlus => {
                        writeln!(s, "{pad}acc = max(acc, {a} + sv[{j}]);").unwrap()
                    }
                };
                match bands {
                    None => {
                        writeln!(
                            s,
                            "            for (var k: u32 = 0u; k < {n}u; k = k + 1u) {{"
                        )
                        .unwrap();
                        let a = load(fmt(ta), &bind(ta), &format!("(p * {n}u + i) * {n}u + k"));
                        combine(&mut s, &a, "k", "                ");
                        writeln!(s, "            }}").unwrap();
                    }
                    Some(o) => {
                        let kb = o.len();
                        for (b, off) in o.iter().enumerate() {
                            writeln!(s, "            {{").unwrap();
                            writeln!(s, "                let j = i32(i) + ({off});").unwrap();
                            writeln!(s, "                if (j >= 0 && j < {n}) {{").unwrap();
                            let a = load(
                                fmt(ta),
                                &bind(ta),
                                &format!("(p * {n}u + i) * {kb}u + {b}u"),
                            );
                            combine(&mut s, &a, "u32(j)", "                    ");
                            writeln!(s, "                }}").unwrap();
                            writeln!(s, "            }}").unwrap();
                        }
                    }
                }
                if *ring == Semiring::Log {
                    writeln!(s, "            acc = acc + log(l);").unwrap();
                }
                writeln!(s, "            nv[i] = acc;").unwrap();
                writeln!(s, "        }}").unwrap();
                writeln!(
                    s,
                    "        for (var i: u32 = 0u; i < {n}u; i = i + 1u) {{ sv[i] = nv[i]; }}"
                )
                .unwrap();
                if !sc.exclusive {
                    put(&mut s);
                }
                writeln!(s, "    }}").unwrap();
                writeln!(s, "}}").unwrap();
            }
        }
        WgslKernel {
            source: s,
            node: TensorId(j as u32),
            reads: reads.into_iter().map(|r| TensorId(r as u32)).collect(),
            points: lines as u32,
            count: lines.div_ceil(64) as u32 * 64,
            workgroup_size: 64,
            writes: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "[{i}] {x} vs {y}");
        }
    }

    fn ramp(n: usize, scale: f64, off: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919 + 13) % 97) as f64 * scale + off)
            .collect()
    }

    /// Scalar affine monoid `(a, b)`: `h_t = a_t·h_{t−1} + b_t`, not
    /// commutative.
    fn affine1() -> Monoid {
        Monoid::build(2, vec![1.0, 0.0], |p, e| {
            vec![p[0] * e[0], e[0] * p[1] + e[1]]
        })
    }

    /// Finite-difference check of `∂(Σ r·y)/∂x_k` for every input `k`.
    fn check_grads(g: &TensorGraph, y: TensorId, inputs: &[Vec<f64>], tol: f64) {
        let mut g = g.clone();
        let ny = numel(g.shape(y));
        let r = g.input(&[ny]);
        let ys = g.shape(y).to_vec();
        let rr = g.reshape(r, &ys).unwrap();
        let prod = g.map(&[y, rr], |v| v[0] * v[1]).unwrap();
        let flat = g.reshape(prod, &[ny]).unwrap();
        let loss = g
            .comp(
                &[1usize],
                &[ny],
                vec![Arg {
                    src: flat,
                    view: View::pick(2, &[Some(1)]),
                }],
                Some(Monoid::sum()),
                |x| vec![x[0]],
            )
            .unwrap();
        let wrt: Vec<TensorId> = (0..inputs.len())
            .map(|k| {
                g.nodes
                    .iter()
                    .position(|n| matches!(n.op, Op::Input(i) if i == k))
                    .map(|p| TensorId(p as u32))
                    .unwrap()
            })
            .collect();
        let grads = g.backward(loss, &wrt).unwrap();
        let mut ins = inputs.to_vec();
        ins.push(ramp(ny, 0.02, -0.9));
        let gv = g.eval(&ins, &grads);
        let h = 1e-6;
        for k in 0..inputs.len() {
            for i in 0..ins[k].len() {
                let mut p = ins.clone();
                p[k][i] += h;
                let lp = g.eval(&p, &[loss])[0][0];
                p[k][i] -= 2.0 * h;
                let lm = g.eval(&p, &[loss])[0][0];
                let fd = (lp - lm) / (2.0 * h);
                let an = gv[k][i];
                assert!(
                    (fd - an).abs() <= tol * (1.0 + fd.abs()),
                    "input {k}[{i}]: fd {fd} vs backward {an}"
                );
            }
        }
    }

    #[test]
    fn cumsum_all_variants() {
        let x = ramp(2 * 5 * 3, 0.1, -2.0);
        for (excl, rev) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut g = TensorGraph::new();
            let xi = g.input(&[2, 5, 3]);
            let y = g.scan(&[xi], 1, Monoid::sum(), excl, rev).unwrap();
            let got = &g.eval(std::slice::from_ref(&x), &[y])[0];
            let mut want = vec![0.0; x.len()];
            for o in 0..2 {
                for i in 0..3 {
                    for t in 0..5 {
                        let range: Vec<usize> = match (excl, rev) {
                            (false, false) => (0..=t).collect(),
                            (true, false) => (0..t).collect(),
                            (false, true) => (t..5).collect(),
                            (true, true) => (t + 1..5).collect(),
                        };
                        want[(o * 5 + t) * 3 + i] =
                            range.iter().map(|&s| x[(o * 5 + s) * 3 + i]).sum();
                    }
                }
            }
            close(got, &want, 1e-12);
        }
    }

    #[test]
    fn noncommutative_monoid_is_the_recurrence() {
        let (tn, c) = (7, 2);
        let a = ramp(tn * c, 0.01, 0.5);
        let b = ramp(tn * c, 0.03, -1.0);
        for rev in [false, true] {
            let mut g = TensorGraph::new();
            let ai = g.input(&[tn, c]);
            let bi = g.input(&[tn, c]);
            let y = g.scan(&[ai, bi], 0, affine1(), false, rev).unwrap();
            let got = &g.eval(&[a.clone(), b.clone()], &[y])[0];
            // forward: e_0 ⊕ … ⊕ e_t; reverse: e_t ⊕ … ⊕ e_{T−1}, both applied
            // left to right as h ← a·h + b from h = 0
            for ch in 0..c {
                for t in 0..tn {
                    let range = if rev { t..tn } else { 0..t + 1 };
                    let h = range.fold(0.0, |h, s| a[s * c + ch] * h + b[s * c + ch]);
                    let p = t * c + ch;
                    assert!((got[p * 2 + 1] - h).abs() < 1e-12, "rev {rev} t {t}");
                }
            }
        }
    }

    /// A map read by a scan and, transposed, by a reduction: `relayout`
    /// would lay it out for the reduction, but a scan reads its inputs as
    /// buffers in the layout it was built with.
    #[test]
    fn relayout_keeps_a_scan_inputs_layout() {
        let mut g = TensorGraph::new();
        let x = g.input(&[3usize, 5]);
        let m = g.map(&[x], |v| v[0] * v[0] + v[0]).unwrap();
        let s = g.scan(&[m], 1, Monoid::sum(), false, false).unwrap();
        let t = g.transpose(m, &[1, 0]).unwrap();
        let r = g.reduce(t, 0, Monoid::sum()).unwrap();
        let ins = vec![ramp(15, 0.1, -0.4)];
        let want = g.eval(&ins, &[s, r]);
        let (rg, ro) = g.relayout(&[s, r]);
        assert_eq!(rg.shape(m), g.shape(m));
        assert_eq!(rg.eval(&ins, &ro), want);
    }

    /// A map read only by a scan stays materialized: the scan kernel reads
    /// a buffer, it cannot recompute the map.
    #[test]
    fn schedules_materialize_a_scan_input() {
        let mut g = TensorGraph::new();
        let x = g.input(&[3usize, 5]);
        let m = g.map(&[x], |v| v[0] * v[0] + v[0]).unwrap();
        let s = g.scan(&[m], 1, Monoid::sum(), false, false).unwrap();
        g.schedule_auto(&[s]);
        assert!(!g.nodes()[m.n()].inline);
        let p = g.to_wgsl(&[s]);
        assert!(p.kernels.len() >= 2, "the map gets its own kernel");
    }

    #[test]
    fn monoid_scan_backward() {
        let (tn, c) = (6, 2);
        let ins = vec![ramp(tn * c, 0.01, 0.5), ramp(tn * c, 0.03, -1.0)];
        for (excl, rev) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut g = TensorGraph::new();
            let ai = g.input(&[tn, c]);
            let bi = g.input(&[tn, c]);
            let y = g.scan(&[ai, bi], 0, affine1(), excl, rev).unwrap();
            check_grads(&g, y, &ins, 1e-5);
        }
        // a running max (piecewise linear, no ties in the data)
        let mut g = TensorGraph::new();
        let xi = g.input(&[2, 9]);
        let y = g.scan(&[xi], 1, Monoid::max(), false, false).unwrap();
        let x: Vec<f64> = (0..18)
            .map(|i| ((i * 37) % 19) as f64 * 0.1 + i as f64 * 1e-3)
            .collect();
        check_grads(&g, y, &[x], 1e-5);
    }

    fn rand_like(n: usize, seed: u64, scale: f64) -> Vec<f64> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    #[test]
    fn affine_scan_backward() {
        let (h, tn, n, m) = (2, 5, 3, 2);
        let a = rand_like(h * tn * n * n, 1, 0.6);
        let u = rand_like(h * tn * n * m, 2, 1.0);
        for (excl, rev) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut g = TensorGraph::new();
            let ai = g.input(&[h, tn, n, n]);
            let ui = g.input(&[h, tn, n, m]);
            let s = g.linear_scan(ai, ui, 1, excl, rev).unwrap();
            check_grads(&g, s, &[a.clone(), u.clone()], 1e-5);
        }
    }

    /// The gated delta rule, token by token.
    #[allow(clippy::too_many_arguments)]
    fn gdn_reference(
        h: usize,
        tn: usize,
        n: usize,
        m: usize,
        q: &[f64],
        k: &[f64],
        v: &[f64],
        al: &[f64],
        be: &[f64],
        scale: f64,
    ) -> Vec<f64> {
        let mut o = vec![0.0; h * tn * m];
        for hh in 0..h {
            let mut s = vec![0.0; n * m];
            for t in 0..tn {
                let pt = hh * tn + t;
                let kt = &k[pt * n..(pt + 1) * n];
                let (a, b) = (al[pt], be[pt]);
                // S = a·S;  S += β·k·(v − Sᵀk)ᵀ
                s.iter_mut().for_each(|x| *x *= a);
                for c in 0..m {
                    let sk: f64 = (0..n).map(|i| s[i * m + c] * kt[i]).sum();
                    let d = b * (v[pt * m + c] - sk);
                    for i in 0..n {
                        s[i * m + c] += kt[i] * d;
                    }
                }
                for c in 0..m {
                    o[pt * m + c] =
                        scale * (0..n).map(|i| s[i * m + c] * q[pt * n + i]).sum::<f64>();
                }
            }
        }
        o
    }

    fn gdn_graph(h: usize, tn: usize, n: usize, m: usize) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let q = g.input(&[h, tn, n]);
        let k = g.input(&[h, tn, n]);
        let v = g.input(&[h, tn, m]);
        let al = g.input(&[h, tn]);
        let be = g.input(&[h, tn]);
        let s = g.delta_scan(k, v, Some(al), Some(be), 1, true).unwrap();
        let o = g.scan_readout(s, q, 1.0 / (n as f64).sqrt()).unwrap();
        (g, o)
    }

    #[test]
    fn delta_scan_is_the_gated_delta_rule() {
        let (h, tn, n, m) = (2, 9, 4, 3);
        let q = rand_like(h * tn * n, 3, 1.0);
        let k = rand_like(h * tn * n, 4, 0.5);
        let v = rand_like(h * tn * m, 5, 1.0);
        let al: Vec<f64> = rand_like(h * tn, 6, 0.1).iter().map(|x| 0.9 + x).collect();
        let be: Vec<f64> = rand_like(h * tn, 7, 0.3).iter().map(|x| 0.5 + x).collect();
        let (g, o) = gdn_graph(h, tn, n, m);
        let got = &g.eval(
            &[q.clone(), k.clone(), v.clone(), al.clone(), be.clone()],
            &[o],
        )[0];
        let want = gdn_reference(h, tn, n, m, &q, &k, &v, &al, &be, 1.0 / (n as f64).sqrt());
        close(got, &want, 1e-12);
    }

    #[test]
    fn delta_scan_backward() {
        let (h, tn, n, m) = (1, 5, 3, 2);
        let ins = vec![
            rand_like(h * tn * n, 3, 1.0),
            rand_like(h * tn * n, 4, 0.5),
            rand_like(h * tn * m, 5, 1.0),
            rand_like(h * tn, 6, 0.1).iter().map(|x| 0.9 + x).collect(),
            rand_like(h * tn, 7, 0.3).iter().map(|x| 0.5 + x).collect(),
        ];
        let (g, o) = gdn_graph(h, tn, n, m);
        check_grads(&g, o, &ins, 1e-5);
    }

    fn naga_ok(src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
    }

    #[test]
    fn scans_lower_to_valid_wgsl() {
        let (h, tn, n, m) = (2, 9, 4, 3);
        let (g, o) = gdn_graph(h, tn, n, m);
        let p = g.to_wgsl(&[o]);
        assert!(p.kernels.iter().any(|k| k.source.contains("(scan)")));
        for k in &p.kernels {
            naga_ok(&k.source);
        }
        for (excl, rev) in [(false, false), (true, true)] {
            let mut g = TensorGraph::new();
            let ai = g.input(&[7, 2]);
            let bi = g.input(&[7, 2]);
            let y = g.scan(&[ai, bi], 0, affine1(), excl, rev).unwrap();
            for k in &g.to_wgsl(&[y]).kernels {
                naga_ok(&k.source);
            }
        }
    }

    #[test]
    fn semiring_scan_backward() {
        let (h, tn, n) = (2, 5, 3);
        let a = rand_like(h * tn * n * n, 3, 1.0);
        let u = rand_like(h * tn * n, 4, 1.0);
        for ring in [Semiring::Log, Semiring::MaxPlus] {
            for rev in [false, true] {
                let mut g = TensorGraph::new();
                let ai = g.input(&[h, tn, n, n]);
                let ui = g.input(&[h, tn, n, 1]);
                let s = g.semiring_scan(ai, ui, 1, false, rev, ring).unwrap();
                check_grads(&g, s, &[a.clone(), u.clone()], 1e-5);
            }
        }
    }

    /// The monotonic-alignment lattice of `lp [B, T, N]` (frame t emits
    /// token i with log-prob `lp[b,t,i]`; a path starts at token 0 and
    /// stays or advances by one per frame) as a semiring scan, read out at
    /// the last token of the last frame: `[B]`.
    fn alignment(
        g: &mut TensorGraph,
        lp: TensorId,
        b: usize,
        tn: usize,
        n: usize,
        ring: Semiring,
    ) -> TensorId {
        let neg = Semiring::NEG;
        let ii = g.iota(&[n, n], 0);
        let jj = g.iota(&[n, n], 1);
        let a = g
            .comp(
                &[b, tn, n, n],
                &[],
                vec![
                    Arg {
                        src: lp,
                        view: View::pick(4, &[Some(0), Some(1), Some(2)]),
                    },
                    Arg {
                        src: ii,
                        view: View::pick(4, &[Some(2), Some(3)]),
                    },
                    Arg {
                        src: jj,
                        view: View::pick(4, &[Some(2), Some(3)]),
                    },
                ],
                None,
                move |x| {
                    // on the band when i − j ∈ {0, 1}
                    let d = x[1] - x[2];
                    let on = (d + ExprId::from_f64(0.5)) * (ExprId::from_f64(1.5) - d);
                    vec![ExprId::select(on, x[0], ExprId::from_f64(neg))]
                },
            )
            .unwrap();
        let tt = g.iota(&[tn, n], 0);
        let ti = g.iota(&[tn, n], 1);
        let u = g
            .comp(
                &[b, tn, n, 1],
                &[],
                vec![
                    Arg {
                        src: lp,
                        view: View::pick(4, &[Some(0), Some(1), Some(2)]),
                    },
                    Arg {
                        src: tt,
                        view: View::pick(4, &[Some(1), Some(2)]),
                    },
                    Arg {
                        src: ti,
                        view: View::pick(4, &[Some(1), Some(2)]),
                    },
                ],
                None,
                move |x| {
                    let h = ExprId::from_f64(0.5);
                    let ng = ExprId::from_f64(neg);
                    vec![ExprId::select(
                        h - x[1],
                        ExprId::select(h - x[2], x[0], ng),
                        ng,
                    )]
                },
            )
            .unwrap();
        let s = g.semiring_scan(a, u, 1, false, false, ring).unwrap();
        g.comp(
            &[b],
            &[],
            vec![Arg {
                src: s,
                view: view(
                    vec![
                        Form::unit(1, 0),
                        Form::constant(1, tn as i64 - 1),
                        Form::constant(1, n as i64 - 1),
                        Form::constant(1, 0),
                    ],
                    false,
                ),
            }],
            None,
            |x| vec![x[0]],
        )
        .unwrap()
    }

    fn lse(a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        a.max(b) + (-(a - b).abs()).exp().ln_1p()
    }

    /// Host forward–backward over the same paths: `(log Z, γ[t·n + i])`.
    fn forward_backward(lp: &[f64], n: usize, t: usize) -> (f64, Vec<f64>) {
        let ninf = f64::NEG_INFINITY;
        let mut al = vec![ninf; t * n];
        al[0] = lp[0];
        for f in 1..t {
            for i in 0..n {
                let stay = al[(f - 1) * n + i];
                let adv = if i > 0 { al[(f - 1) * n + i - 1] } else { ninf };
                al[f * n + i] = lse(stay, adv) + lp[f * n + i];
            }
        }
        let mut be = vec![ninf; t * n];
        be[(t - 1) * n + n - 1] = 0.0;
        for f in (0..t - 1).rev() {
            for i in 0..n {
                let stay = be[(f + 1) * n + i] + lp[(f + 1) * n + i];
                let adv = if i + 1 < n {
                    be[(f + 1) * n + i + 1] + lp[(f + 1) * n + i + 1]
                } else {
                    ninf
                };
                be[f * n + i] = lse(stay, adv);
            }
        }
        let z = al[(t - 1) * n + n - 1];
        let ga = (0..t * n).map(|k| (al[k] + be[k] - z).exp()).collect();
        (z, ga)
    }

    /// Host Viterbi over the same paths: `(best score, token per frame)`.
    fn viterbi(lp: &[f64], n: usize, t: usize) -> (f64, Vec<usize>) {
        let ninf = f64::NEG_INFINITY;
        let mut v = vec![ninf; t * n];
        v[0] = lp[0];
        for f in 1..t {
            for i in 0..n {
                let stay = v[(f - 1) * n + i];
                let adv = if i > 0 { v[(f - 1) * n + i - 1] } else { ninf };
                v[f * n + i] = stay.max(adv) + lp[f * n + i];
            }
        }
        let mut path = vec![0; t];
        let mut i = n - 1;
        for f in (0..t).rev() {
            path[f] = i;
            if f > 0 && i > 0 && v[(f - 1) * n + i - 1] > v[(f - 1) * n + i] {
                i -= 1;
            }
        }
        (v[(t - 1) * n + n - 1], path)
    }

    /// The forward-sum loss and MAS are one scan in two semirings, and
    /// their gradients are the posteriors and the best path.
    #[test]
    fn forward_sum_and_mas_are_semiring_scans() {
        let (b, tn, n) = (3, 12, 5);
        let lp = rand_like(b * tn * n, 7, 3.0);
        for ring in [Semiring::Log, Semiring::MaxPlus] {
            let mut g = TensorGraph::new();
            let x = g.input(&[b, tn, n]);
            let z = alignment(&mut g, x, b, tn, n, ring);
            let dx = g.backward(z, &[x]).unwrap()[0];
            let o = g.eval(std::slice::from_ref(&lp), &[z, dx]);
            for bb in 0..b {
                let row = &lp[bb * tn * n..(bb + 1) * tn * n];
                let got = &o[1][bb * tn * n..(bb + 1) * tn * n];
                if ring == Semiring::Log {
                    let (zr, ga) = forward_backward(row, n, tn);
                    close(&[o[0][bb]], &[zr], 1e-12);
                    close(got, &ga, 1e-10);
                } else {
                    let (zr, path) = viterbi(row, n, tn);
                    close(&[o[0][bb]], &[zr], 1e-12);
                    let hot: Vec<f64> = (0..tn * n)
                        .map(|k| (path[k / n] == k % n) as u8 as f64)
                        .collect();
                    close(got, &hot, 0.0);
                }
            }
        }
    }

    #[test]
    fn semiring_scans_lower_to_valid_wgsl() {
        for ring in [Semiring::Log, Semiring::MaxPlus] {
            let mut g = TensorGraph::new();
            let x = g.input(&[2, 9, 4]);
            let z = alignment(&mut g, x, 2, 9, 4, ring);
            let dx = g.backward(z, &[x]).unwrap()[0];
            let p = g.to_wgsl(&[z, dx]);
            assert!(p.kernels.iter().any(|k| k.source.contains("(scan)")));
            for k in &p.kernels {
                naga_ok(&k.source);
            }
        }
    }

    /// The dense `[.., n, n]` of a banded `[.., n, k]` (the semiring zero
    /// off the band).
    fn densify(ab: &[f64], pts: usize, n: usize, bands: &[i64], zero: f64) -> Vec<f64> {
        let k = bands.len();
        let mut a = vec![zero; pts * n * n];
        for p in 0..pts {
            for i in 0..n {
                for (b, &o) in bands.iter().enumerate() {
                    let j = i as i64 + o;
                    if (0..n as i64).contains(&j) {
                        a[(p * n + i) * n + j as usize] = ab[(p * n + i) * k + b];
                    }
                }
            }
        }
        a
    }

    /// A banded scan is the dense scan of its densified `A`, forward and
    /// backward, in every semiring and direction; and its own backward
    /// matches finite differences.
    #[test]
    fn banded_scan_is_the_dense_scan() {
        let (h, tn, n) = (2, 6, 5);
        for bands in [vec![-1i64, 0], vec![-2, -1, 0], vec![-2, 0, 1]] {
            let k = bands.len();
            let ab = rand_like(h * tn * n * k, 11, 1.0);
            let u = rand_like(h * tn * n, 12, 1.0);
            for ring in [Semiring::Real, Semiring::Log, Semiring::MaxPlus] {
                let ad = densify(&ab, h * tn, n, &bands, ring.zero());
                for rev in [false, true] {
                    let mut g = TensorGraph::new();
                    let ai = g.input(&[h, tn, n, k]);
                    let ui = g.input(&[h, tn, n, 1]);
                    let s = g
                        .semiring_scan_banded(ai, ui, &bands, 1, rev, ring)
                        .unwrap();
                    let gr = g.backward(s, &[ai, ui]).unwrap();
                    let ob = g.eval(&[ab.clone(), u.clone()], &[s, gr[0], gr[1]]);

                    let mut g = TensorGraph::new();
                    let ai = g.input(&[h, tn, n, n]);
                    let ui = g.input(&[h, tn, n, 1]);
                    let s = g.semiring_scan(ai, ui, 1, false, rev, ring).unwrap();
                    let gr = g.backward(s, &[ai, ui]).unwrap();
                    let od = g.eval(&[ad.clone(), u.clone()], &[s, gr[0], gr[1]]);

                    close(&ob[0], &od[0], 1e-12);
                    close(&ob[2], &od[2], 1e-10);
                    // ∂A on the band (off it the real ring has a gradient too)
                    let on = densify(&vec![1.0; ab.len()], h * tn, n, &bands, 0.0);
                    let ga = densify(&ob[1], h * tn, n, &bands, 0.0);
                    let gd: Vec<f64> = od[1].iter().zip(&on).map(|(g, m)| g * m).collect();
                    close(&ga, &gd, 1e-10);

                    let mut g = TensorGraph::new();
                    let ai = g.input(&[h, tn, n, k]);
                    let ui = g.input(&[h, tn, n, 1]);
                    let s = g
                        .semiring_scan_banded(ai, ui, &bands, 1, rev, ring)
                        .unwrap();
                    check_grads(&g, s, &[ab.clone(), u.clone()], 1e-5);
                }
            }
        }
    }

    /// Max-plus ties: the gradient goes to the smallest predecessor index,
    /// and to a predecessor before `U`.
    #[test]
    fn max_plus_ties_go_to_the_smaller_index() {
        let neg = Semiring::NEG;
        // two steps, n = 2. Step 0: S = U = [0, 0]. Step 1: A = 0, so both
        // predecessors tie for both rows; row 1's U ties them too.
        let a = vec![0.0; 2 * 2 * 2];
        let u = vec![0.0, 0.0, neg, 0.0];
        for banded in [false, true] {
            let mut g = TensorGraph::new();
            let (ai, s) = if banded {
                // bands [-1, 0, 1] cover the dense 2×2 (column j = i + o)
                let ab: Vec<f64> = vec![0.0; 2 * 2 * 3];
                let ai = g.input(&[2, 2, 3]);
                let ui = g.input(&[2, 2, 1]);
                let s = g
                    .semiring_scan_banded(ai, ui, &[-1, 0, 1], 0, false, Semiring::MaxPlus)
                    .unwrap();
                (ai, (s, ui, ab))
            } else {
                let ai = g.input(&[2, 2, 2]);
                let ui = g.input(&[2, 2, 1]);
                let s = g
                    .semiring_scan(ai, ui, 0, false, false, Semiring::MaxPlus)
                    .unwrap();
                (ai, (s, ui, a.clone()))
            };
            let (s, ui, av) = s;
            let gr = g.backward(s, &[ai, ui]).unwrap();
            let o = g.eval(&[av, u.clone()], &[s, gr[0], gr[1]]);
            close(&o[0], &[0.0, 0.0, 0.0, 0.0], 0.0);
            // ∂A at step 1, as dense [i][j]: every row picks j = 0
            let ga = if banded {
                densify(&o[1], 2, 2, &[-1, 0, 1], 0.0)
            } else {
                o[1].clone()
            };
            close(&ga[4..], &[1.0, 0.0, 1.0, 0.0], 0.0);
            // ∂U: step 0 gets both rows' cotangent on j = 0 (plus its own),
            // step 1's U lost every tie
            close(&o[2], &[3.0, 1.0, 0.0, 0.0], 0.0);
        }
    }

    #[test]
    fn semiring_scans_reject_unsupported_shapes() {
        let mut g = TensorGraph::new();
        let a = g.input(&[4, 3, 3]);
        let u2 = g.input(&[4, 3, 2]);
        let u1 = g.input(&[4, 3, 1]);
        for ring in [Semiring::Log, Semiring::MaxPlus] {
            let e = g.semiring_scan(a, u2, 0, false, false, ring).unwrap_err();
            assert!(e.0.contains("vector state"), "{}", e.0);
            let e = g.semiring_scan(a, u1, 0, true, false, ring).unwrap_err();
            assert!(e.0.contains("inclusive"), "{}", e.0);
        }
        // real dense keeps m > 1 and exclusive
        assert!(g
            .semiring_scan(a, u2, 0, true, true, Semiring::Real)
            .is_ok());
        let ab = g.input(&[4, 3, 2]);
        assert!(g
            .semiring_scan_banded(ab, u2, &[-1, 0], 0, false, Semiring::Real)
            .is_err());
        assert!(g
            .semiring_scan_banded(ab, u1, &[0, -1], 0, false, Semiring::Log)
            .is_err());
        assert!(g
            .semiring_scan_banded(ab, u1, &[-1, 0, 1], 0, false, Semiring::Log)
            .is_err());
        assert!(g
            .semiring_scan_banded(ab, u1, &[-1, 0], 0, false, Semiring::Log)
            .is_ok());
    }

    fn fnv(s: &str) -> u64 {
        s.bytes().fold(0xcbf29ce484222325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        })
    }

    /// The real ring emits exactly what it did before semirings existed:
    /// the gated-delta graph, forward and backward, in WGSL and CUDA.
    /// Hashes taken on `origin/main` at e889ba87.
    #[test]
    fn real_scan_codegen_is_unchanged() {
        let (h, tn, n, m) = (2, 9, 4, 3);
        let (mut g, o) = gdn_graph(h, tn, n, m);
        let ins: Vec<TensorId> = (0..5).map(TensorId).collect();
        let mut outs = vec![o];
        outs.extend(g.backward(o, &ins).unwrap());
        let cat = |k: &[WgslKernel]| {
            k.iter()
                .map(|k| k.source.clone())
                .collect::<Vec<_>>()
                .join("\n//----\n")
        };
        assert_eq!(fnv(&cat(&g.to_wgsl(&outs).kernels)), 0xa708c4381058eff8);
        assert_eq!(fnv(&cat(&g.to_cuda(&outs).kernels)), 0x13f2b5c9a82d1b89);
        assert!(g.chunked_scan_plan(o).is_some());
    }

    /// Banded alignment: the same lattice as [`alignment`] with `A` stored
    /// `[B, T, N, 2]`, bands `[-1, 0]` (advance, stay).
    fn alignment_banded(
        g: &mut TensorGraph,
        lp: TensorId,
        b: usize,
        tn: usize,
        n: usize,
        ring: Semiring,
    ) -> TensorId {
        let neg = Semiring::NEG;
        let a = g
            .comp(
                &[b, tn, n, 2],
                &[],
                vec![Arg {
                    src: lp,
                    view: View::pick(4, &[Some(0), Some(1), Some(2)]),
                }],
                None,
                |x| vec![x[0]],
            )
            .unwrap();
        let tt = g.iota(&[tn, n], 0);
        let ti = g.iota(&[tn, n], 1);
        let u = g
            .comp(
                &[b, tn, n, 1],
                &[],
                vec![
                    Arg {
                        src: lp,
                        view: View::pick(4, &[Some(0), Some(1), Some(2)]),
                    },
                    Arg {
                        src: tt,
                        view: View::pick(4, &[Some(1), Some(2)]),
                    },
                    Arg {
                        src: ti,
                        view: View::pick(4, &[Some(1), Some(2)]),
                    },
                ],
                None,
                move |x| {
                    let h = ExprId::from_f64(0.5);
                    let ng = ExprId::from_f64(neg);
                    vec![ExprId::select(
                        h - x[1],
                        ExprId::select(h - x[2], x[0], ng),
                        ng,
                    )]
                },
            )
            .unwrap();
        let s = g
            .semiring_scan_banded(a, u, &[-1, 0], 1, false, ring)
            .unwrap();
        g.comp(
            &[b],
            &[],
            vec![Arg {
                src: s,
                view: view(
                    vec![
                        Form::unit(1, 0),
                        Form::constant(1, tn as i64 - 1),
                        Form::constant(1, n as i64 - 1),
                        Form::constant(1, 0),
                    ],
                    false,
                ),
            }],
            None,
            |x| vec![x[0]],
        )
        .unwrap()
    }

    #[test]
    fn banded_forward_sum_and_mas() {
        let (b, tn, n) = (3, 12, 5);
        let lp = rand_like(b * tn * n, 7, 3.0);
        for ring in [Semiring::Log, Semiring::MaxPlus] {
            let mut g = TensorGraph::new();
            let x = g.input(&[b, tn, n]);
            let z = alignment_banded(&mut g, x, b, tn, n, ring);
            let dx = g.backward(z, &[x]).unwrap()[0];
            let o = g.eval(std::slice::from_ref(&lp), &[z, dx]);
            for bb in 0..b {
                let row = &lp[bb * tn * n..(bb + 1) * tn * n];
                let got = &o[1][bb * tn * n..(bb + 1) * tn * n];
                if ring == Semiring::Log {
                    let (zr, ga) = forward_backward(row, n, tn);
                    close(&[o[0][bb]], &[zr], 1e-12);
                    close(got, &ga, 1e-10);
                } else {
                    let (zr, path) = viterbi(row, n, tn);
                    close(&[o[0][bb]], &[zr], 1e-12);
                    let hot: Vec<f64> = (0..tn * n)
                        .map(|k| (path[k / n] == k % n) as u8 as f64)
                        .collect();
                    close(got, &hot, 0.0);
                }
            }
            for k in &g.to_wgsl(&[z, dx]).kernels {
                naga_ok(&k.source);
            }
        }
    }
}
