//! Tensor-level IR: views + one compute op.
//!
//! Everything above the scalar [`tang_expr`] graph is built from two ideas:
//!
//! - A [`View`] is an affine map from a comprehension's iteration space to a
//!   source tensor's coordinates. Broadcast, transpose, slice and stride are
//!   all views. They compose and never touch memory.
//! - A [`Comp`] (comprehension) is the only compute op:
//!
//!   ```text
//!   out[i…] = ⊕_{k…} body(x₀[view₀(i,k)], x₁[view₁(i,k)], …)
//!   ```
//!
//!   `body` is a scalar `tang_expr` graph and `⊕` is a [`Monoid`] whose
//!   combine is also a scalar graph. A map is a comp with no reduce dims;
//!   sum/max are comps with a monoid; matmul is `Σ_k a[i,k]·b[k,j]`.
//!
//! A comp may produce several values per point: its trailing `tuple` out
//! dims are filled by the body rather than iterated. That is how monoids
//! carry tuple state — online softmax is `(max, sum, acc…)` — while
//! consumers still address the result with ordinary views.
//!
//! Passes:
//! - [`TensorGraph::substitute_maps`] — the single fusion rule. A map is a
//!   pure function of its inputs, so it is inlined into every consumer's
//!   body and its views are composed with the consumer's.
//! - [`TensorGraph::stabilize_softmax`] — rewrites `max_j s` + `Σ_j g·exp(s−max)`
//!   into one reduction with the online-softmax monoid.
//! - [`TensorGraph::schedule_inline`] — a schedule decision: a single-use
//!   reduction is recomputed inside its consumer when that costs no more
//!   than materializing it.
//!
//! Scope of this sketch: fixed shapes, f64, a reference interpreter that
//! respects materialization decisions, no autodiff/codegen/sharding yet.

use std::collections::HashMap;

use tang::Scalar;
use tang_expr::{trace, ExprGraph, ExprId, Node};

/// Plain C for CPUs (`to_c`): comps lowered directly to loops.
#[path = "tensor_c.rs"]
pub mod c;
#[path = "tensor_cuda.rs"]
pub mod cuda;
#[path = "tensor_cuda_gemv.rs"]
pub mod cuda_gemv;
/// The int8 tensor-core lowering of [`IntMma`] and the Q8x32 store.
#[path = "tensor_cuda_int8.rs"]
mod cuda_int8;
#[path = "tensor_cuda_mega.rs"]
pub mod cuda_mega;
/// A whole program as one persistent CUDA kernel (no kernel boundaries).
#[path = "tensor_cuda_scan.rs"]
pub mod cuda_scan;
/// Algorithm discovery: online, scan and chunked lifts.
#[path = "tensor_discover.rs"]
pub mod discover;
#[path = "tensor_discover_catalog.rs"]
pub mod discover_catalog;
#[cfg(test)]
#[path = "tensor_gather_tests.rs"]
mod gather_tests;
/// Vulkan decode lowering: the decode kernels as GLSL (SPIR-V passthrough).
#[path = "tensor_glsl_gemv.rs"]
pub mod glsl_gemv;
#[path = "tensor_grad.rs"]
mod grad;
#[path = "tensor_io.rs"]
pub mod io;
pub use io::{IoReport, KernelIo};
#[path = "tensor_io_lp.rs"]
pub mod io_lp;
#[path = "tensor_memplan.rs"]
pub mod memplan;
#[path = "tensor_modq.rs"]
pub mod modq;
#[path = "tensor_msl.rs"]
pub mod msl;
/// #34's f32 simdgroup-matrix tile (`to_msl_mma`, opt-in via TANG_NATIVE)
/// and the affine/bilinear analysis the CUDA wmma lowering shares.
#[path = "tensor_msl_gemv.rs"]
mod msl_gemv;
#[path = "tensor_msl_mma.rs"]
mod msl_mma;
/// OpenCL C (Adreno and other mobile GPUs): the WGSL kernels transpiled.
#[path = "tensor_opencl.rs"]
pub mod opencl;
#[path = "tensor_opencl_gemv.rs"]
pub mod opencl_gemv;
#[path = "tensor_place.rs"]
pub mod place;
pub use place::{Boundary, MemorySchedule, Place, Placed};
#[path = "tensor_precision.rs"]
pub mod precision;
/// The declarative schedule search space and the autotuner's cost model.
#[path = "tensor_scan.rs"]
pub mod scan;
pub use scan::{DeltaForm, Scan, ScanKind};
#[path = "tensor_space.rs"]
pub mod space;
/// Sparse SpMV/PCG as tensor-IR programs and the `LinearSolve` /
/// `IndexedGather` operator nodes.
#[path = "tensor_sparse.rs"]
pub mod sparse;
#[path = "tensor_tile.rs"]
mod tile;
#[path = "tensor_wgsl.rs"]
pub mod wgsl;
pub use grad::BackwardOptions;
pub use precision::{ErrorBudget, ErrorStats, Format, Precision};
pub use wgsl::WgslCaps;

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// `offset + Σ coefs[d] · idx[d]` over an iteration space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Form {
    pub coefs: Vec<i64>,
    pub offset: i64,
}

impl Form {
    pub fn zero(rank: usize) -> Self {
        Self {
            coefs: vec![0; rank],
            offset: 0,
        }
    }

    fn scaled(&self, c: i64) -> Form {
        Form {
            coefs: self.coefs.iter().map(|x| x * c).collect(),
            offset: self.offset * c,
        }
    }

    pub fn unit(rank: usize, d: usize) -> Self {
        let mut f = Self::zero(rank);
        f.coefs[d] = 1;
        f
    }

    pub fn constant(rank: usize, c: i64) -> Self {
        let mut f = Self::zero(rank);
        f.offset = c;
        f
    }

    fn eval(&self, idx: &[usize]) -> i64 {
        self.offset
            + self
                .coefs
                .iter()
                .zip(idx)
                .map(|(c, i)| c * *i as i64)
                .sum::<i64>()
    }

    /// Substitute each coordinate `d` of this form by `inner[d]`.
    fn compose(&self, inner: &[Form], rank: usize) -> Form {
        let mut out = Form::zero(rank);
        out.offset = self.offset;
        for (c, f) in self.coefs.iter().zip(inner) {
            out.offset += c * f.offset;
            for (o, i) in out.coefs.iter_mut().zip(&f.coefs) {
                *o += c * i;
            }
        }
        out
    }

    fn range(&self, space: &[usize]) -> (i64, i64) {
        let (mut lo, mut hi) = (self.offset, self.offset);
        for (c, n) in self.coefs.iter().zip(space) {
            let extent = c * (*n as i64 - 1);
            if extent < 0 {
                lo += extent
            } else {
                hi += extent
            }
        }
        (lo, hi)
    }

    /// `Some(d)` if this form is exactly coordinate `d`.
    fn as_unit(&self) -> Option<usize> {
        if self.offset != 0 || self.coefs.iter().any(|c| *c != 0 && *c != 1) {
            return None;
        }
        let mut it = self.coefs.iter().enumerate().filter(|(_, c)| **c == 1);
        match (it.next(), it.next()) {
            (Some((d, _)), None) => Some(d),
            _ => None,
        }
    }
}

/// One form per source dimension, over the consumer's iteration space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View {
    pub forms: Vec<Form>,
    /// Reads outside the source return 0 instead of being a shape error:
    /// zero padding as a view. The guard travels with the leaf read through
    /// composition, and a gradient through it scatters only in range.
    pub guard: bool,
    /// Data-dependent addressing: source coordinate `d` is
    /// `forms[d] + gather.coefs[d] · round(args[gather.arg])`, the value of a
    /// sibling arg of the same comp (an integer-valued index tensor read
    /// through its own affine view). `None` for a plain affine view.
    pub gather: Option<Gather>,
}

/// The data-dependent term of a [`View`]: the index is the value of arg
/// `arg` of the comp that owns the view (at the same point), scaled per
/// source dim by `coefs`. Referencing a sibling arg rather than a tensor
/// keeps dependency tracking and node renumbering free: the index tensor is
/// an ordinary arg. The index arg's own view must be affine (no nesting).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Gather {
    pub arg: usize,
    pub coefs: Vec<i64>,
}

impl View {
    /// Source dim `d` reads iteration coordinate `picks[d]`, or is
    /// broadcast (constant 0) when `None`.
    pub fn pick(rank: usize, picks: &[Option<usize>]) -> Self {
        let forms = picks
            .iter()
            .map(|p| p.map_or_else(|| Form::zero(rank), |d| Form::unit(rank, d)))
            .collect();
        Self {
            forms,
            guard: false,
            gather: None,
        }
    }

    /// Source dim `dim` (of `src_rank`) reads `scale · index + forms[dim]`
    /// where the index is the value of arg `arg`; the other forms are kept.
    pub fn gather(mut self, dim: usize, arg: usize, scale: i64) -> Self {
        let mut coefs = vec![0; self.forms.len()];
        coefs[dim] = scale;
        self.gather = Some(Gather { arg, coefs });
        self
    }

    /// Whether this view addresses its source through an index arg.
    pub fn is_gather(&self) -> bool {
        self.gather.is_some()
    }

    /// Identity over the first `src_rank` coordinates of a `rank`-dim space.
    pub fn identity(rank: usize, src_rank: usize) -> Self {
        Self::pick(rank, &(0..src_rank).map(Some).collect::<Vec<_>>())
    }

    fn rank(&self) -> usize {
        self.forms.first().map_or(0, |f| f.coefs.len())
    }

    /// Signed source coordinates; `index` is the gather arg's value (ignored
    /// for an affine view).
    fn coords_signed(&self, idx: &[usize], index: f64) -> Vec<i64> {
        let g = index.round() as i64;
        self.forms
            .iter()
            .enumerate()
            .map(|(d, f)| f.eval(idx) + self.gather.as_ref().map_or(0, |ga| ga.coefs[d] * g))
            .collect()
    }

    /// Coordinates in `shape`, or `None` when a guarded read falls outside.
    /// An unguarded read outside the source is a caller bug (affine views
    /// are range-checked at construction; gathered ones cannot be).
    fn coords_in(&self, idx: &[usize], shape: &[usize], index: f64) -> Option<Vec<usize>> {
        let cs = self.coords_signed(idx, index);
        let inside = cs
            .iter()
            .zip(shape)
            .all(|(&c, &n)| (0..n as i64).contains(&c));
        if !inside {
            assert!(self.guard, "unguarded read {cs:?} outside source {shape:?}");
            return None;
        }
        Some(cs.into_iter().map(|c| c as usize).collect())
    }

    /// A view that may read outside its source; out-of-range reads are 0.
    pub fn guarded(mut self) -> Self {
        self.guard = true;
        self
    }

    /// `self ∘ inner`: this view is over a space whose coordinates are
    /// given by `inner` over a `rank`-dim space.
    ///
    /// A gather on either side carries through, its `arg` still indexing
    /// the args of the comp that owned it: the caller remaps it (see
    /// [`splice_arg`]). An inner gather scales through `self`'s forms (a
    /// data-dependent coordinate is still added linearly); both sides
    /// gathering would nest one index read inside another and panics —
    /// callers check [`View::is_gather`] first.
    fn compose(&self, inner: &View, rank: usize) -> View {
        let gather = match (&self.gather, &inner.gather) {
            (None, None) => None,
            (Some(g), None) => Some(g.clone()),
            (None, Some(gi)) => Some(Gather {
                arg: gi.arg,
                coefs: self
                    .forms
                    .iter()
                    .map(|f| f.coefs.iter().zip(&gi.coefs).map(|(a, b)| a * b).sum())
                    .collect(),
            }),
            (Some(_), Some(_)) => panic!("nested gather in view composition"),
        };
        View {
            // one side is always a plain view: either a guarded leaf read
            // pushed through an unguarded consumer, or the reverse
            guard: self.guard || inner.guard,
            forms: self
                .forms
                .iter()
                .map(|f| f.compose(&inner.forms, rank))
                .collect(),
            gather,
        }
    }

    /// This view reads a tensor of shape `dst` that is a row-major reshape
    /// of one of shape `src`. When every `src` dim is a contiguous product
    /// of `dst` dims (a merge, seen from `dst`), the read is affine in the
    /// source's coordinates and this returns the equivalent view into
    /// `src`. A reshape that splits a source dim across destination dims
    /// (`[s, d] -> [s*d]`) is not affine (it needs `/` and `%`): `None`.
    pub fn through_reshape(&self, dst: &[Dim], src: &[Dim]) -> Option<View> {
        // a guard bounds the reshape's coords, not the source's; a gather
        // would need its coefs re-strided too (not done)
        if self.guard || self.is_gather() {
            return None;
        }
        let dst: Vec<usize> = dst.iter().map(|d| d.fixed()).collect::<Option<_>>()?;
        let src: Vec<usize> = src.iter().map(|d| d.fixed()).collect::<Option<_>>()?;
        let rank = self.rank();
        let mut forms = Vec::with_capacity(src.len());
        let mut d = 0;
        for &n in &src {
            // consume dst dims d.. until their product is n
            let start = d;
            let mut prod = 1;
            while prod < n {
                prod *= *dst.get(d)?;
                d += 1;
            }
            if prod != n {
                return None;
            }
            // src coord = Σ stride_k · dst_coord_k over the run
            let mut f = Form::zero(rank);
            let mut stride = 1i64;
            for k in (start..d).rev() {
                let scaled = self.forms[k].scaled(stride);
                for (o, i) in f.coefs.iter_mut().zip(&scaled.coefs) {
                    *o += i;
                }
                f.offset += scaled.offset;
                stride *= dst[k] as i64;
            }
            forms.push(f);
        }
        (d == dst.len()).then_some(View {
            forms,
            guard: false,
            gather: None,
        })
    }

    /// Drop dependence on size-1 coordinates so equal views compare equal.
    fn normalized(&self, space: &[Dim]) -> View {
        let mut v = self.clone();
        for f in &mut v.forms {
            for (c, &n) in f.coefs.iter_mut().zip(space) {
                if n == Dim::Fixed(1) {
                    *c = 0;
                }
            }
        }
        v
    }

    fn depends_on(&self, d: usize) -> bool {
        self.forms.iter().any(|f| f.coefs[d] != 0)
    }
}

// ---------------------------------------------------------------------------
// Monoids
// ---------------------------------------------------------------------------

/// Reduction state of `width` values. `combine` has vars `0..w` for the
/// accumulator and `w..2w` for the incoming element, and `w` outputs.
#[derive(Clone)]
pub struct Monoid {
    pub width: usize,
    pub identity: Vec<f64>,
    pub combine: ExprGraph,
    pub outs: Vec<ExprId>,
    /// Structure of a discovered online lift (state `(R₁, Σ D)`), from
    /// which the backward derives an elementwise saved-state gradient
    /// instead of the generic prefix/suffix-scan one.
    pub saved: Option<std::sync::Arc<discover::SavedBwd>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonoidKind {
    Sum,
    Max,
    OnlineSoftmax,
    Other,
}

impl Monoid {
    fn build(
        width: usize,
        identity: Vec<f64>,
        f: impl FnOnce(&[ExprId], &[ExprId]) -> Vec<ExprId>,
    ) -> Self {
        let (combine, outs) = trace(|| {
            let acc: Vec<ExprId> = (0..width).map(|i| ExprId::var(i as u16)).collect();
            let el: Vec<ExprId> = (0..width)
                .map(|i| ExprId::var((width + i) as u16))
                .collect();
            f(&acc, &el)
        });
        Self {
            width,
            identity,
            combine,
            outs,
            saved: None,
        }
    }

    pub fn sum() -> Self {
        Self::build(1, vec![0.0], |a, e| vec![a[0] + e[0]])
    }

    pub fn max() -> Self {
        Self::build(1, vec![f64::NEG_INFINITY], |a, e| vec![max(a[0], e[0])])
    }

    /// State `(i, v)`, element `(j, w)`: the larger value with its index,
    /// the smaller index on a tie (so the result is the first maximum,
    /// whatever the combine order). Identity `(0, -f32::MAX)`.
    pub fn argmax() -> Self {
        Self::build(2, vec![0.0, -(f32::MAX as f64)], |a, e| {
            let (i, v, j, w) = (a[0], a[1], e[0], e[1]);
            let d = w - v;
            let sel = |c, t, f| tang_expr::with_graph(|g| g.select(c, t, f));
            let tie = sel(i - j, j, i);
            vec![sel(d, j, sel(-d, i, tie)), sel(d, w, v)]
        })
    }

    /// The `k` best `(index, value)` pairs, best first: state and element
    /// are both `(i₀, v₀, …, iₖ₋₁, vₖ₋₁)` sorted by value descending, the
    /// smaller index first on a tie (the order k argmax rounds pick in, so
    /// the result is the same whatever the combine order). An element of
    /// one pair is `(j, w)` followed by `k − 1` identity pairs
    /// `(0, -f32::MAX)`. The combine is the merge-path rule: the r-th best
    /// of two sorted lists `A`, `B` is the best of `Aᵣ`, `Bᵣ` and
    /// `min(Aᵢ₋₁, Bᵣ₋ᵢ)` for `i` in `1..=r` (k²/2 compare-selects, no
    /// sort). One reduction replaces k argmax rounds and their masks.
    pub fn topk(k: usize) -> Self {
        assert!(k >= 1, "top-k of k >= 1");
        let mut id = Vec::with_capacity(2 * k);
        for _ in 0..k {
            id.push(0.0);
            id.push(-(f32::MAX as f64));
        }
        Self::build(2 * k, id, |a, e| {
            let sel = |c, t, f| tang_expr::with_graph(|g| g.select(c, t, f));
            // > 0 when x ranks above y (larger value, then smaller index)
            let better = |x: (ExprId, ExprId), y: (ExprId, ExprId)| {
                let d = x.1 - y.1;
                sel(d, ExprId::ONE, sel(-d, -ExprId::ONE, y.0 - x.0))
            };
            let pick = |c: ExprId, x: (ExprId, ExprId), y: (ExprId, ExprId)| {
                (sel(c, x.0, y.0), sel(c, x.1, y.1))
            };
            let hi = |x, y| pick(better(x, y), x, y);
            let lo = |x, y| pick(better(x, y), y, x);
            let av: Vec<(ExprId, ExprId)> = (0..k).map(|r| (a[2 * r], a[2 * r + 1])).collect();
            let bv: Vec<(ExprId, ExprId)> = (0..k).map(|r| (e[2 * r], e[2 * r + 1])).collect();
            let mut out = Vec::with_capacity(2 * k);
            for r in 0..k {
                let mut best = hi(av[r], bv[r]);
                for i in 1..=r {
                    best = hi(best, lo(av[i - 1], bv[r - i]));
                }
                out.push(best.0);
                out.push(best.1);
            }
            out
        })
    }

    /// State `(m, acc₁…accₖ)`, element `(s, g₁…gₖ)`:
    /// `m' = max(m, s)`, `accᵢ' = accᵢ·exp(m−m') + gᵢ·exp(s−m')`.
    pub fn online_softmax(k: usize) -> Self {
        let mut id = vec![0.0; k + 1];
        id[0] = f64::NEG_INFINITY;
        Self::build(k + 1, id, |a, e| {
            let m = max(a[0], e[0]);
            let sa = (a[0] - m).exp();
            let se = (e[0] - m).exp();
            let mut out = vec![m];
            out.extend((1..=k).map(|i| a[i] * sa + e[i] * se));
            out
        })
    }

    pub fn kind(&self) -> MonoidKind {
        let n = self.combine.nodes_slice();
        if self.width == 1 {
            return match n[self.outs[0].index() as usize] {
                Node::Add(a, b) if is_var(n, a, 0) && is_var(n, b, 1) => MonoidKind::Sum,
                Node::Select(c, a, b)
                    if is_var(n, a, 0)
                        && is_var(n, b, 1)
                        && matches!(n[c.index() as usize], Node::Add(..)) =>
                {
                    MonoidKind::Max
                }
                _ => MonoidKind::Other,
            };
        }
        if self.identity[0] <= -1e30 && self.matches(&Monoid::online_softmax(self.width - 1)) {
            MonoidKind::OnlineSoftmax
        } else {
            MonoidKind::Other
        }
    }

    /// Same combine as `o` at a few probe points (a tuple monoid with a
    /// `−∞` first slot is not necessarily an online softmax: the
    /// discovered lifts have that shape too).
    fn matches(&self, o: &Monoid) -> bool {
        if o.width != self.width {
            return false;
        }
        let w = self.width;
        (0..3).all(|t| {
            let v: Vec<f64> = (0..2 * w)
                .map(|i| ((i * 7 + t * 13) % 11) as f64 * 0.37 - 1.3)
                .collect();
            let a = self.combine.eval_many::<f64>(&self.outs, &v);
            let b = o.combine.eval_many::<f64>(&o.outs, &v);
            a.iter()
                .zip(&b)
                .all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + y.abs()))
        })
    }
}

/// `max(a, b)` via select, so it is exact at `-inf` (the `Scalar` impl
/// uses `sqrt((a-b)²)`, which is NaN there).
fn max(a: ExprId, b: ExprId) -> ExprId {
    let d = a - b;
    tang_expr::with_graph(|g| g.select(d, a, b))
}

fn is_var(nodes: &[Node], id: ExprId, v: u16) -> bool {
    nodes[id.index() as usize] == Node::Var(v)
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(u32);

impl TensorId {
    fn n(self) -> usize {
        self.0 as usize
    }

    /// Index into [`TensorGraph::nodes`].
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Handle for node `index` (no validation).
    pub fn from_index(index: usize) -> Self {
        TensorId(index as u32)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Arg {
    pub src: TensorId,
    pub view: View,
}

#[derive(Clone)]
pub struct Comp {
    /// Output shape. The last `tuple` dims are produced per point.
    pub out: Vec<Dim>,
    /// Trailing out dims filled by the body (0 for ordinary comps).
    pub tuple: usize,
    pub red: Vec<Dim>,
    pub args: Vec<Arg>,
    pub body: ExprGraph,
    pub body_outs: Vec<ExprId>,
    pub monoid: Option<Monoid>,
}

impl Comp {
    fn width(&self) -> usize {
        self.body_outs.len()
    }

    /// Iterated out dims (excluding tuple dims).
    fn points(&self) -> &[Dim] {
        &self.out[..self.out.len() - self.tuple]
    }

    fn rank(&self) -> usize {
        self.points().len() + self.red.len()
    }

    fn space(&self) -> Vec<Dim> {
        self.points().iter().chain(&self.red).copied().collect()
    }

    /// Whether any arg addresses its source through an index arg. Passes
    /// that drop, reorder or split args leave such comps alone.
    pub fn has_gather(&self) -> bool {
        self.args.iter().any(|a| a.view.is_gather())
    }

    /// Args read as gather indices (by some other arg's view).
    fn index_args(&self) -> Vec<usize> {
        self.args
            .iter()
            .filter_map(|a| a.view.gather.as_ref().map(|g| g.arg))
            .collect()
    }
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Op {
    Input(usize),
    /// The coordinate along `dim` at every point, as a value. Lets a body
    /// see where it is: causal masks, positional tables, slice gradients.
    Iota(usize),
    /// The same bytes as `src` (row-major) under this node's shape. Costs
    /// nothing: readers address the source buffer. Its source is always
    /// materialized, so strided and offset views over a reshape stay
    /// affine (and `backward` can invert them as pick views).
    Reshape(TensorId),
    Comp(Comp),
    /// An opaque operator: native code forward, a user-supplied adjoint
    /// backward. See [`Operator`].
    Operator(OperatorNode),
    /// A prefix scan over one dim with an associative combine: a tuple
    /// monoid or the matrix affine monoid. See [`Scan`].
    Scan(Scan),
}

/// An opaque operator node: whole-tensor native code with its own
/// adjoint. Solves, simulations and other steps that are not
/// comprehensions (a sparse linear solve, a rollout) enter the graph as
/// one of these; everything around them stays comprehensions.
///
/// The reference interpreter materializes every input, calls
/// [`forward`](Operator::forward) once, and stores the result like any
/// other buffer. [`TensorGraph::backward`] calls
/// [`backward`](Operator::backward), which appends the adjoint to the
/// graph: comprehensions, further operator nodes, or both (`LinearSolve`'s
/// adjoint is a transposed solve followed by an outer-product
/// comprehension). Device backends do not lower operator nodes; a graph
/// containing one runs on the interpreter or splits around it.
pub trait Operator: Send + Sync + std::panic::RefUnwindSafe {
    /// Short name for debugging and graph dumps.
    fn name(&self) -> &str;

    /// The output, row-major, from the inputs (row-major, in
    /// [`OperatorNode::inputs`] order).
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64>;

    /// Append the vector-Jacobian product to `g`: given the forward
    /// node's inputs, its output and the output cotangent `cot`, return
    /// each input's cotangent (`None` for inputs that are not
    /// differentiable, such as an index pattern).
    fn backward(
        &self,
        g: &mut TensorGraph,
        inputs: &[TensorId],
        output: TensorId,
        cot: TensorId,
    ) -> Result<Vec<Option<TensorId>>, ShapeError>;
}

/// [`Op::Operator`]: the operator and the tensors it reads.
#[derive(Clone)]
pub struct OperatorNode {
    pub op: std::sync::Arc<dyn Operator>,
    pub inputs: Vec<TensorId>,
}

/// [`TNode::segment`]: reduce dim `red` (index into the comp's reduce dims)
/// only contributes where `key[r] == point coord `point``, `key` a sorted
/// 1-D tensor read by one of the comp's args as `key[r]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub red: usize,
    pub key: TensorId,
    pub point: usize,
}

#[derive(Clone)]
pub struct TNode {
    pub op: Op,
    pub shape: Vec<Dim>,
    /// Schedule: recompute inside consumers instead of writing a buffer.
    pub inline: bool,
    /// Schedule: split the reduction across a workgroup of this many
    /// threads, tree-combined with the monoid (0 = one thread per point).
    pub split: u32,
    /// Schedule: `(threads over last point dim, threads over second-to-last,
    /// reduce chunk)`. Reads shared along a tile axis are staged chunk by
    /// chunk in workgroup memory.
    pub tile: (u32, u32, u32),
    /// Schedule: points per thread along `(x, y)` within a tile — the
    /// register tile. `(1, 1)` is one point per thread.
    pub reg: (u32, u32),
    /// Schedule: lower this tiled contraction onto matrix-unit fragments
    /// (Metal `simdgroup_matrix`) on backends that have them. Only legal
    /// for an f16 × f16 → f32 contraction; see
    /// [`TensorGraph::fragment_plan`]. Backends without fragments ignore it.
    pub fragment: bool,
    /// Schedule: lower this contraction onto integer tensor-core fragments
    /// (CUDA `mma.sync` m16n8k32 s8) with its weight operand dequantized
    /// in registers from a block format. See [`IntMma`]. Backends without
    /// the lowering ignore it (and then reject the block formats).
    pub int_mma: Option<IntMma>,
    /// A fact about the data, not a schedule: this 1-D tensor is
    /// nondecreasing (e.g. MoE pairs sorted by expert). Set by
    /// [`TensorGraph::set_sorted`]; the backward uses it to give gathered
    /// contractions a [`Segment`].
    pub sorted: bool,
    /// A fact about differentiation, not a schedule: this node is
    /// [`detach`](TensorGraph::detach)ed — an identity map whose
    /// cotangent is zero. Rewrite passes keep it a node of its own (they
    /// never splice it into its readers, nor merge it with an
    /// undetached twin); schedules may still inline it, so it costs
    /// nothing at run time.
    pub detach: bool,
    /// The reduction's contributing range along one reduce dim is the run
    /// of a sorted key equal to a point coordinate (a segmented sum). Set by
    /// the backward of a gather through a [`sorted`](Self::sorted) index;
    /// the body keeps its `[key == coord]` gate, so lowerings that ignore
    /// it stay correct and the WGSL point loop binary-searches the run.
    pub segment: Option<Segment>,
    /// Schedule: combine a `split` reduction's per-thread partials with
    /// subgroup operations (WGSL `subgroupAdd`/`subgroupShuffleXor`, MSL
    /// `simd_sum`/`simd_shuffle_xor`) instead of a workgroup-memory tree:
    /// one barrier instead of log₂(split). Honoured where the target has
    /// subgroups ([`WgslCaps::subgroups`](crate::tensor::WgslCaps); always
    /// on Metal), ignored elsewhere.
    pub subgroup: bool,
    /// Schedule: geometry of a fragment-lowered attention kernel —
    /// `[query rows per SIMD group / 8, keys per tile / 8, value columns
    /// per threadgroup / 8, SIMD groups per threadgroup]`; 0 = the
    /// backend's default. Ignored unless `fragment` is set and the node
    /// has an attention plan.
    pub frag_cfg: [u8; 4],
    /// Schedule: number formats — storage of a materialized buffer (or
    /// input) and the running accumulator of a reduction. Default all
    /// f32. See [`Precision`].
    pub precision: Precision,
}

/// The integer-MMA schedule of one contraction
/// `out[p…, r] = Σ_k W[idx(p), r, k] · X[p…, k]`: `W` a graph input stored
/// Q4_K / Q5_K (its first dim optionally gathered by a per-point index), `X`
/// a buffer stored [`Format::Q8x32`]. A warp owns 16 rows `r` (the MMA's M)
/// and 8 consecutive points `p` (its N); consecutive points that gather the
/// same `W` slice share each A fragment, so a point dim sorted by the index
/// reads every slice once a tile. The integer dot of a 32-column step is
/// scaled once in float (weight `d · scale − dmin · min`, activation `d`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntMma {
    /// How the activation operand reaches the B fragment.
    pub stage: Stage,
    /// Warps that split a 16-row tile's reduction (1, 2, 4 or 8), their
    /// partial sums added through shared memory.
    pub ksplit: u32,
    /// [`Stage::Gemv`] only: consecutive positions that share one read of
    /// the weights (each block dots its weight rows with up to this many
    /// activation rows; 0 or 1: one). Needs an ungathered weight.
    pub pos_tile: u32,
}

/// Activation staging for [`IntMma`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Fragment order: each lane's B fragment is 8 consecutive columns, one
    /// 8-byte load from global, no shared memory; the A fragment takes the
    /// same permutation of the 32 columns (a dot product does not care).
    Fragment,
    /// The MMA's own k order: two 4-byte loads a lane (columns `4·tig..`
    /// and `16 + 4·tig..`), on both operands.
    Mma,
    /// No MMA: the narrow schedule for few points a weight slice (decode,
    /// batch 1). Each output `(p, r)` is a dot of one weight row with one
    /// activation row on `dp4a`, the row's half-super-blocks spread over
    /// a group of lanes (as many as `k / 128`, at most 32) and summed by
    /// shuffles; a warp holds `32 / lanes` rows. `ksplit` is ignored.
    Gemv,
}

/// Reduce dims the tiled emitter handles: one chunked axis (the largest)
/// plus small inner axes looped inside each chunk step — a conv's
/// `(c, k)`, a polyphase upsample's `(j, c)`. The inner product bounds
/// how much each staged region grows.
pub(crate) fn tileable_red(red: &[Dim]) -> bool {
    let Some(r) = red.iter().map(|d| d.fixed()).collect::<Option<Vec<_>>>() else {
        return false;
    };
    let Some(&big) = r.iter().max() else {
        return false;
    };
    r.iter().product::<usize>() / big.max(1) <= 32
}

/// The point dims a tile maps to its `(x, y)` axes. Lanes (x): the
/// innermost dim of at least 16 (falling back to the innermost non-unit
/// dim), so a narrow trailing dim — a conv weight gradient's taps `k` in
/// `dw[o, c, k]`, a polyphase upsample's phase `r` — becomes a uniform
/// (per-workgroup) coordinate instead of a 7-wide lane axis. Rows (y):
/// the largest other non-unit dim (ties: the innermost), so a layout
/// chosen for the consumer's reads (`relayout`) still tiles well.
pub(crate) fn tile_axes(pts: &[usize]) -> (Option<usize>, Option<usize>) {
    let x = (0..pts.len())
        .rev()
        .find(|&d| pts[d] >= 16)
        .or_else(|| (0..pts.len()).rev().find(|&d| pts[d] > 1));
    (x, x.and_then(|dx| tile::largest_other(pts, dx)))
}

/// How `merge_reduce_dims` rewrites one arg: `None` leaves it, `Some((d0,
/// shape))` reads a reshape of its source with dims `d0, d0 + 1` merged.
type MergePlan = Option<(usize, Vec<Dim>)>;

/// `(split, tile, reg)` — one thread mapping for a comp.
pub type ScheduleCandidate = (u32, (u32, u32, u32), (u32, u32));

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeError(pub String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "shape error: {}", self.0)
    }
}

impl std::error::Error for ShapeError {}

pub(super) fn err<T>(msg: impl Into<String>) -> Result<T, ShapeError> {
    Err(ShapeError(msg.into()))
}

/// A tensor dimension: a fixed size or a named symbol (`"seq"`, `"batch"`).
///
/// Symbolic dims flow through construction, rewriting and autodiff.
/// Anything that needs numbers — the interpreter, schedules, lowering —
/// works on a graph after [`TensorGraph::bind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Fixed(usize),
    Sym(&'static str),
}

impl Dim {
    pub fn fixed(self) -> Option<usize> {
        match self {
            Dim::Fixed(n) => Some(n),
            Dim::Sym(_) => None,
        }
    }

    /// The size, panicking on an unbound symbol.
    pub fn n(self) -> usize {
        match self {
            Dim::Fixed(n) => n,
            Dim::Sym(name) => panic!("dim `{name}` is unbound; call TensorGraph::bind first"),
        }
    }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Dim::Fixed(n)
    }
}

impl PartialEq<usize> for Dim {
    fn eq(&self, other: &usize) -> bool {
        *self == Dim::Fixed(*other)
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{n}"),
            Dim::Sym(s) => write!(f, "{s}"),
        }
    }
}

/// Concrete sizes of a bound shape.
pub fn fixed(shape: &[Dim]) -> Vec<usize> {
    shape.iter().map(|d| d.n()).collect()
}

/// Number of elements of a bound shape.
pub fn numel(shape: &[Dim]) -> usize {
    shape.iter().map(|d| d.n()).product()
}

fn dims<D: Into<Dim> + Copy>(d: &[D]) -> Vec<Dim> {
    d.iter().map(|&x| x.into()).collect()
}

#[derive(Clone, Default)]
pub struct TensorGraph {
    nodes: Vec<TNode>,
    n_inputs: usize,
}

impl TensorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn nodes(&self) -> &[TNode] {
        &self.nodes
    }

    /// Set a node's thread-mapping schedule (`split`, `tile`), e.g. from
    /// an autotuner. Materialization (`inline`) is left alone: it changes
    /// which kernels exist, not how one runs.
    pub fn set_schedule(
        &mut self,
        t: TensorId,
        split: u32,
        tile: (u32, u32, u32),
        reg: (u32, u32),
    ) {
        self.nodes[t.n()].split = split;
        self.nodes[t.n()].tile = tile;
        self.nodes[t.n()].reg = reg;
    }

    /// Schedule: recompute `t` inside its consumers (`true`) or write it
    /// to a buffer. `schedule_inline` decides this by rule; this is the
    /// manual override, e.g. to fuse a reduction under its epilogue map.
    pub fn set_subgroup(&mut self, t: TensorId, on: bool) {
        self.nodes[t.n()].subgroup = on;
    }

    /// Assert that 1-D `t` is nondecreasing (data, e.g. keys sorted by an
    /// earlier comp). Gradients of contractions gathered through it then
    /// reduce over runs found by binary search instead of the whole dim.
    pub fn set_sorted(&mut self, t: TensorId, sorted: bool) {
        self.nodes[t.n()].sorted = sorted;
    }

    pub fn segment(&self, t: TensorId) -> Option<Segment> {
        self.nodes[t.n()].segment
    }

    pub fn set_inline(&mut self, t: TensorId, inline: bool) {
        self.nodes[t.n()].inline = inline;
    }

    /// Schedule: the number formats of `t` — `storage` for its buffer
    /// (inputs and materialized comps; ignored when inlined) and `accum`
    /// for its running reduction state. Precision is a schedule, not a
    /// type: the graph stays math, and [`TensorGraph::eval`] ignores it.
    /// [`TensorGraph::eval_emulated`] and the WGSL lowering honour it.
    pub fn set_precision(&mut self, t: TensorId, p: Precision) {
        self.nodes[t.n()].precision = p;
    }

    /// Schedule: run contraction `t` on integer MMA fragments
    /// ([`IntMma`]); `None` clears it. The CUDA lowering checks the pattern
    /// and the operand formats and panics when they do not fit.
    pub fn set_int_mma(&mut self, t: TensorId, s: Option<IntMma>) {
        self.nodes[t.n()].int_mma = s;
    }

    pub fn precision(&self, t: TensorId) -> Precision {
        self.nodes[t.n()].precision
    }

    /// Thread mappings worth trying for a materialized comp on a GPU:
    /// one thread per point, a workgroup split (reductions), and rows ×
    /// lanes tiles (single-axis reductions with a lane dim).
    pub fn schedule_candidates(&self, t: TensorId) -> Vec<ScheduleCandidate> {
        let mut out = vec![(0, (0, 0, 0), (1, 1))];
        let Some(c) = self.tiled_reduction(t.n()).and_then(|r| self.comp_at(r)) else {
            return out;
        };
        // a tree-combine of a wide tuple through workgroup memory is never
        // the answer (59 ms for attention's 66-wide state on a 3090)
        if c.width() <= 8 {
            out.push((256, (0, 0, 0), (1, 1)));
        }
        // tiles stage affine regions of their operands; a gathered read
        // has no region to stage (one thread per point, or a split, only)
        let root = self.comp_at(t.n()).unwrap();
        if c.has_gather() || root.has_gather() {
            return out;
        }
        if tileable_red(&c.red) {
            let pts = fixed(c.points());
            let (ax, ay) = tile_axes(&pts);
            let lanes = ax.map_or(1, |d| pts[d]);
            let rows = ay.is_some();
            if (16..=128).contains(&lanes) && rows {
                out.push((0, (lanes as u32, (256 / lanes) as u32, 32), (1, 1)));
            }
            if lanes >= 64 && !rows {
                out.push((0, (lanes.min(256) as u32, 1, 32), (1, 1)));
            }
            if lanes >= 16 && rows {
                // the classic shared-memory 2-D tile (matmul-shaped comps),
                // one point per thread and with 2×2 / 4×4 register tiles
                out.push((0, (16, 16, 32), (1, 1)));
                out.push((0, (16, 16, 32), (2, 2)));
                out.push((0, (16, 16, 32), (4, 4)));
            }
        }
        out
    }

    pub fn shape(&self, t: TensorId) -> &[Dim] {
        &self.nodes[t.n()].shape
    }

    pub fn input<D: Into<Dim> + Copy>(&mut self, shape: &[D]) -> TensorId {
        let i = self.n_inputs;
        self.n_inputs += 1;
        self.nodes.push(TNode {
            op: Op::Input(i),
            shape: dims(shape),
            inline: false,
            split: 0,
            tile: (0, 0, 0),
            reg: (1, 1),
            fragment: false,
            int_mma: None,
            segment: None,
            sorted: false,
            detach: false,
            subgroup: false,
            frag_cfg: [0; 4],
            precision: Precision::default(),
        });
        TensorId(self.nodes.len() as u32 - 1)
    }

    /// `out[i…] = i[dim]` as a float. No buffer: reads compute the
    /// coordinate through the consumer's view.
    pub fn iota<D: Into<Dim> + Copy>(&mut self, shape: &[D], dim: usize) -> TensorId {
        self.nodes.push(TNode {
            op: Op::Iota(dim),
            shape: dims(shape),
            inline: false,
            split: 0,
            tile: (0, 0, 0),
            reg: (1, 1),
            fragment: false,
            int_mma: None,
            segment: None,
            sorted: false,
            detach: false,
            subgroup: false,
            frag_cfg: [0; 4],
            precision: Precision::default(),
        });
        TensorId(self.nodes.len() as u32 - 1)
    }

    /// `x` viewed row-major under `shape` (same number of elements).
    pub fn reshape<D: Into<Dim> + Copy>(
        &mut self,
        x: TensorId,
        shape: &[D],
    ) -> Result<TensorId, ShapeError> {
        let shape = dims(shape);
        let src = self.shape(x).to_vec();
        let same = match (
            src.iter().map(|d| d.fixed()).collect::<Option<Vec<_>>>(),
            shape.iter().map(|d| d.fixed()).collect::<Option<Vec<_>>>(),
        ) {
            (Some(a), Some(b)) => a.iter().product::<usize>() == b.iter().product::<usize>(),
            _ => true, // symbolic: trusted until bind
        };
        if !same {
            return err(format!("reshape {src:?} -> {shape:?}"));
        }
        self.nodes.push(TNode {
            op: Op::Reshape(x),
            shape,
            inline: false,
            split: 0,
            tile: (0, 0, 0),
            reg: (1, 1),
            fragment: false,
            int_mma: None,
            segment: None,
            sorted: false,
            detach: false,
            subgroup: false,
            frag_cfg: [0; 4],
            precision: Precision::default(),
        });
        Ok(TensorId(self.nodes.len() as u32 - 1))
    }

    /// Append an [`Operator`] node reading `inputs`, with output `shape`.
    pub fn operator<D: Into<Dim> + Copy>(
        &mut self,
        op: std::sync::Arc<dyn Operator>,
        inputs: &[TensorId],
        shape: &[D],
    ) -> TensorId {
        self.nodes.push(TNode {
            op: Op::Operator(OperatorNode {
                op,
                inputs: inputs.to_vec(),
            }),
            shape: dims(shape),
            inline: false,
            split: 0,
            tile: (0, 0, 0),
            reg: (1, 1),
            fragment: false,
            int_mma: None,
            segment: None,
            sorted: false,
            detach: false,
            subgroup: false,
            frag_cfg: [0; 4],
            precision: Precision::default(),
        });
        TensorId(self.nodes.len() as u32 - 1)
    }

    /// The node whose buffer holds `t`'s bytes: `t` itself, or the source
    /// behind a chain of reshapes.
    pub fn storage(&self, t: TensorId) -> TensorId {
        match self.nodes[t.n()].op {
            Op::Reshape(src) => self.storage(src),
            _ => t,
        }
    }

    /// Nodes `i` reads.
    pub(super) fn deps(&self, i: usize) -> Vec<usize> {
        match &self.nodes[i].op {
            Op::Comp(c) => c.args.iter().map(|a| a.src.n()).collect(),
            Op::Reshape(src) => vec![src.n()],
            Op::Operator(o) => o.inputs.iter().map(|t| t.n()).collect(),
            Op::Scan(sc) => sc.inputs.iter().map(|t| t.n()).collect(),
            Op::Input(_) | Op::Iota(_) => Vec::new(),
        }
    }

    /// The general op. `f` receives one scalar per arg and returns the
    /// per-point values (`monoid.width` of them when reducing).
    ///
    /// `f` runs inside a fresh `tang_expr` trace: every literal
    /// (`ExprId::from_f64`) and operation must be created *inside* the
    /// closure. An `ExprId` made outside refers to some other graph and
    /// silently means the wrong node.
    pub fn comp<D: Into<Dim> + Copy>(
        &mut self,
        out: &[D],
        red: &[D],
        args: Vec<Arg>,
        monoid: Option<Monoid>,
        f: impl FnOnce(&[ExprId]) -> Vec<ExprId>,
    ) -> Result<TensorId, ShapeError> {
        let (body, body_outs) = trace(|| {
            let vars: Vec<ExprId> = (0..args.len()).map(|i| ExprId::var(i as u16)).collect();
            f(&vars)
        });
        self.push_comp(Comp {
            out: dims(out),
            tuple: 0,
            red: dims(red),
            args,
            body,
            body_outs,
            monoid,
        })
    }

    /// [`comp`](Self::comp) whose body produces a tuple: the output's
    /// last dim (of size the body's width) holds a point's values — the
    /// state of a tuple monoid such as [`Monoid::argmax`].
    pub fn comp_tuple<D: Into<Dim> + Copy>(
        &mut self,
        out: &[D],
        red: &[D],
        args: Vec<Arg>,
        monoid: Option<Monoid>,
        f: impl FnOnce(&[ExprId]) -> Vec<ExprId>,
    ) -> Result<TensorId, ShapeError> {
        let (body, body_outs) = trace(|| {
            let vars: Vec<ExprId> = (0..args.len()).map(|i| ExprId::var(i as u16)).collect();
            f(&vars)
        });
        self.push_comp(Comp {
            out: dims(out),
            tuple: 1,
            red: dims(red),
            args,
            body,
            body_outs,
            monoid,
        })
    }

    fn push_comp(&mut self, comp: Comp) -> Result<TensorId, ShapeError> {
        let space = comp.space();
        for (i, a) in comp.args.iter().enumerate() {
            let src = self.shape(a.src);
            if a.view.forms.len() != src.len() || a.view.rank() != space.len() {
                return err(format!(
                    "arg {i}: view has {} forms of rank {}, source {src:?} in space {space:?}",
                    a.view.forms.len(),
                    a.view.rank()
                ));
            }
            if let Some(g) = &a.view.gather {
                let ok = g.coefs.len() == src.len()
                    && comp
                        .args
                        .get(g.arg)
                        .is_some_and(|ia| ia.view.gather.is_none());
                if !ok {
                    return err(format!(
                        "arg {i}: gather through arg {} must name an affine sibling, {} coefs",
                        g.arg,
                        src.len()
                    ));
                }
            }
            let bound: Option<Vec<usize>> = space.iter().map(|d| d.fixed()).collect();
            if let (Some(space), false) = (bound, a.view.guard) {
                let gathered = |d: usize| a.view.gather.as_ref().is_some_and(|g| g.coefs[d] != 0);
                for (d, (f, &n)) in a.view.forms.iter().zip(src).enumerate() {
                    // a data-dependent coordinate cannot be range-checked
                    if gathered(d) {
                        continue;
                    }
                    let (lo, hi) = f.range(&space);
                    if lo < 0 || n.fixed().is_some_and(|n| hi >= n as i64) {
                        return err(format!("arg {i}: view reaches {lo}..={hi}, dim is {n}"));
                    }
                }
            }
        }
        let width = numel(&comp.out[comp.out.len() - comp.tuple..]); // tuple dims are always fixed
        if comp.width() != width {
            return err(format!(
                "body produces {} values, tuple dims need {width}",
                comp.width()
            ));
        }
        match &comp.monoid {
            Some(m) if m.width != width => {
                return err(format!(
                    "monoid width {} but body produces {width}",
                    m.width
                ))
            }
            None if !comp.red.is_empty() => return err("reduce dims without a monoid"),
            _ => {}
        }
        self.nodes.push(TNode {
            shape: comp.out.clone(),
            inline: false,
            split: 0,
            tile: (0, 0, 0),
            reg: (1, 1),
            fragment: false,
            int_mma: None,
            segment: None,
            sorted: false,
            detach: false,
            subgroup: false,
            frag_cfg: [0; 4],
            precision: Precision::default(),
            op: Op::Comp(comp),
        });
        Ok(TensorId(self.nodes.len() as u32 - 1))
    }

    // -- conveniences, all sugar over `comp` ---------------------------------

    /// Elementwise over equally-shaped args.
    pub fn map(
        &mut self,
        args: &[TensorId],
        f: impl FnOnce(&[ExprId]) -> ExprId,
    ) -> Result<TensorId, ShapeError> {
        let Some(&first) = args.first() else {
            return err("map needs at least one arg");
        };
        let shape = self.shape(first).to_vec();
        if let Some(a) = args.iter().find(|a| self.shape(**a) != shape) {
            return err(format!(
                "map args differ: {shape:?} vs {:?}",
                self.shape(*a)
            ));
        }
        let r = shape.len();
        let args = args
            .iter()
            .map(|&src| Arg {
                src,
                view: View::identity(r, r),
            })
            .collect();
        self.comp(&shape, &[], args, None, |v| vec![f(v)])
    }

    /// `x`, as a constant of [`backward`](Self::backward): the same
    /// values, no cotangent flows through.
    ///
    /// **Implicit differentiation.** Let `λ*` solve `F(λ, θ) = 0`, found
    /// by any iteration, and `H = ∂F/∂λ`. Take one more Newton step from
    /// the detached solution with a detached matrix,
    ///
    /// `λ' = detach(λ*) − detach(H)⁻¹·F(detach(λ*), θ)`.
    ///
    /// Its value is `λ*` (the residual is zero) and its derivative is
    /// `−H⁻¹·∂F/∂θ`: the implicit function theorem. So the adjoint of a
    /// converged solve is one linear solve with the same matrix, whatever
    /// the iteration count, and nothing of the iteration is kept for the
    /// backward. Exact only at convergence: the gradient's error is of the
    /// order of the residual.
    ///
    /// **Representation.** An identity map with [`TNode::detach`] set.
    /// The mark is on the node, so it survives every rewrite pass
    /// (`substitute_maps`, `eliminate_common`, `merge_siblings`,
    /// `relayout`, `eliminate_dead`, the schedules): passes never splice a
    /// detached map into its readers or merge it with an undetached twin.
    /// Schedules may inline it, so it costs no buffer or kernel.
    ///
    /// **Higher order.** `detach` is a constant for every `backward`, so
    /// derivatives of any order through it are zero: a gradient graph that
    /// reads a detached node (the cotangent of a body that read it) treats
    /// it as a constant again, and the vjp-of-vjp through it is zero —
    /// `stop_gradient` semantics, not the implicit function theorem
    /// applied twice. For second-order implicit derivatives, detach
    /// inside the step you differentiate twice. tang-compile has no
    /// forward-mode (jvp) pass; there the same rule (zero tangent) would
    /// apply.
    pub fn detach(&mut self, x: TensorId) -> Result<TensorId, ShapeError> {
        let y = self.map(&[x], |v| v[0])?;
        self.nodes[y.n()].detach = true;
        Ok(y)
    }

    /// Reduce `axis` of `x` with `m`; the axis is kept as size 1 so the
    /// result broadcasts back against `x`.
    pub fn reduce(&mut self, x: TensorId, axis: usize, m: Monoid) -> Result<TensorId, ShapeError> {
        let shape = self.shape(x).to_vec();
        if axis >= shape.len() {
            return err(format!("axis {axis} out of range for {shape:?}"));
        }
        let mut out = shape.clone();
        out[axis] = Dim::Fixed(1);
        let red = [shape[axis]];
        let r = shape.len();
        // iteration space = out dims ++ [axis]; source dim `axis` reads the red coord
        let picks: Vec<Option<usize>> = (0..r)
            .map(|d| Some(if d == axis { r } else { d }))
            .collect();
        let args = vec![Arg {
            src: x,
            view: View::pick(r + 1, &picks),
        }];
        self.comp(&out, &red, args, Some(m), |v| vec![v[0]])
    }

    /// Broadcast `x` (with size-1 dims) up to `shape`, as a view-only map.
    pub fn broadcast<D: Into<Dim> + Copy>(
        &mut self,
        x: TensorId,
        shape: &[D],
    ) -> Result<TensorId, ShapeError> {
        let shape = &dims(shape);
        let src = self.shape(x).to_vec();
        if src.len() != shape.len() {
            return err(format!("broadcast {src:?} -> {shape:?}: rank differs"));
        }
        let r = shape.len();
        let picks: Vec<Option<usize>> = src
            .iter()
            .zip(shape)
            .enumerate()
            .map(|(d, (&s, &t))| match (s == t, s == Dim::Fixed(1)) {
                (true, _) => Ok(Some(d)),
                (false, true) => Ok(None),
                _ => err(format!("broadcast {src:?} -> {shape:?}")),
            })
            .collect::<Result<_, _>>()?;
        let args = vec![Arg {
            src: x,
            view: View::pick(r, &picks),
        }];
        self.comp(shape, &[], args, None, |v| vec![v[0]])
    }

    pub fn transpose(&mut self, x: TensorId, perm: &[usize]) -> Result<TensorId, ShapeError> {
        let src = self.shape(x).to_vec();
        let r = src.len();
        let mut inv = vec![None; r];
        for (o, &p) in perm.iter().enumerate() {
            if p >= r || inv[p].replace(o).is_some() {
                return err(format!("transpose {src:?} by {perm:?}"));
            }
        }
        if perm.len() != r {
            return err(format!("transpose {src:?} by {perm:?}"));
        }
        let shape: Vec<Dim> = perm.iter().map(|&p| src[p]).collect();
        let args = vec![Arg {
            src: x,
            view: View::pick(r, &inv),
        }];
        self.comp(&shape, &[], args, None, |v| vec![v[0]])
    }

    /// `[m,k] x [k,n] -> [m,n]`.
    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, ShapeError> {
        let (sa, sb) = (self.shape(a).to_vec(), self.shape(b).to_vec());
        if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
            return err(format!("matmul {sa:?} x {sb:?}"));
        }
        let args = vec![
            Arg {
                src: a,
                view: View::pick(3, &[Some(0), Some(2)]),
            },
            Arg {
                src: b,
                view: View::pick(3, &[Some(2), Some(1)]),
            },
        ];
        self.comp(&[sa[0], sb[1]], &[sa[1]], args, Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
    }

    /// `out[i…, j…] = src[…, round(index[i…]), …]`: dim `dim` of `src`
    /// addressed through `index` (an integer-valued tensor), the other
    /// source dims kept in order after the index dims. An embedding
    /// lookup is `gather(table, 0, ids)`. Sugar over a comp with a
    /// gathered view; `guard` makes out-of-range indices (a `-1` pad) read
    /// 0 instead of being undefined.
    pub fn gather(
        &mut self,
        src: TensorId,
        dim: usize,
        index: TensorId,
        guard: bool,
    ) -> Result<TensorId, ShapeError> {
        let s = self.shape(src).to_vec();
        let ix = self.shape(index).to_vec();
        if dim >= s.len() {
            return err(format!("gather dim {dim} of {s:?}"));
        }
        let k = ix.len();
        let rest: Vec<usize> = (0..s.len()).filter(|&d| d != dim).collect();
        let out: Vec<Dim> = ix
            .iter()
            .copied()
            .chain(rest.iter().map(|&d| s[d]))
            .collect();
        let rank = out.len();
        let mut picks: Vec<Option<usize>> = vec![None; s.len()];
        for (j, &d) in rest.iter().enumerate() {
            picks[d] = Some(k + j);
        }
        let mut view = View::pick(rank, &picks).gather(dim, 1, 1);
        view.guard = guard;
        let args = vec![
            Arg { src, view },
            Arg {
                src: index,
                view: View::identity(rank, k),
            },
        ];
        self.comp(&out, &[], args, None, |v| vec![v[0]])
    }

    // -- analysis -----------------------------------------------------------

    pub fn comp_at(&self, i: usize) -> Option<&Comp> {
        match &self.nodes[i].op {
            Op::Comp(c) => Some(c),
            Op::Input(_) | Op::Iota(_) | Op::Reshape(_) | Op::Operator(_) | Op::Scan(_) => None,
        }
    }

    fn live(&self, outputs: &[TensorId]) -> Vec<bool> {
        let mut live = vec![false; self.nodes.len()];
        let mut stack: Vec<usize> = outputs.iter().map(|t| t.n()).collect();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut live[i], true) {
                continue;
            }
            stack.extend(self.deps(i));
        }
        live
    }

    /// Live consumers per node (an output counts once).
    fn use_counts(&self, outputs: &[TensorId], live: &[bool]) -> Vec<usize> {
        let mut uses = vec![0; self.nodes.len()];
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            let mut deps = self.deps(i);
            deps.sort_unstable();
            deps.dedup();
            for d in deps {
                uses[d] += 1;
            }
        }
        for o in outputs {
            uses[o.n()] += 1;
        }
        uses
    }

    /// How a map reads a reduction point-for-point, when it does: the
    /// reduction's point coordinates as affine forms over the map's
    /// points (shared by every arg that reads it), with each arg's view
    /// into the tuple dims a constant. Such a map is the reduction's
    /// epilogue: every value it needs is one thread's final accumulator.
    ///
    /// A state slot that does not vary along a point dim (the row sum of
    /// an online softmax along its lanes) may be read at any coordinate
    /// of that dim: every thread's accumulator holds the same value.
    pub(super) fn epilogue_read(&self, c: &Comp, p: usize) -> Option<Vec<Form>> {
        let pc = self.comp_at(p)?;
        let pk = pc.points().len();
        let varies = self.slot_varies(pc);
        let mut points: Vec<Option<Form>> = vec![None; pk];
        let mut fallback: Option<Vec<Form>> = None;
        for a in c.args.iter().filter(|a| a.src.n() == p) {
            // a guarded read is 0 outside: not the accumulator; a gathered
            // one reads some other point's accumulator
            if a.view.guard || a.view.is_gather() {
                return None;
            }
            let (pt, state) = a.view.forms.split_at(pk);
            if state.iter().any(|f| f.coefs.iter().any(|&x| x != 0)) {
                return None;
            }
            let slot = self.epilogue_slot(a);
            for d in (0..pk).filter(|&d| varies[slot][d]) {
                match &points[d] {
                    Some(q) if *q != pt[d] => return None,
                    Some(_) => {}
                    None => points[d] = Some(pt[d].clone()),
                }
            }
            fallback.get_or_insert_with(|| pt.to_vec());
        }
        let fallback = fallback?;
        Some(
            points
                .into_iter()
                .zip(fallback)
                .map(|(p, f)| p.unwrap_or(f))
                .collect(),
        )
    }

    /// `[slot][dim]`: whether state slot `slot` of reduction `c` can vary
    /// along point dim `dim`. A slot varies along a dim when, through the
    /// monoid's combine, it depends on an element value whose body reads
    /// an arg along that dim.
    fn slot_varies(&self, c: &Comp) -> Vec<Vec<bool>> {
        let w = c.width();
        let pk = c.points().len();
        let var_dims = |g: &ExprGraph, out: ExprId, f: &mut dyn FnMut(u16)| {
            for k in g.live_set(&[out]) {
                if let Node::Var(v) = g.nodes_slice()[k] {
                    f(v);
                }
            }
        };
        // element slot e varies along d when body out e reads an arg along d
        let mut elem = vec![vec![false; pk]; w];
        for (e, &out) in c.body_outs.iter().enumerate() {
            var_dims(&c.body, out, &mut |v| {
                for (d, varies) in elem[e].iter_mut().enumerate() {
                    *varies |= c.args[v as usize].view.depends_on(d);
                }
            });
        }
        let Some(m) = &c.monoid else { return elem };
        // state slot k depends on element slots and other state slots
        let deps: Vec<(Vec<usize>, Vec<usize>)> = m
            .outs
            .iter()
            .map(|&out| {
                let (mut st, mut el) = (Vec::new(), Vec::new());
                var_dims(&m.combine, out, &mut |v| {
                    if (v as usize) < w {
                        st.push(v as usize)
                    } else {
                        el.push(v as usize - w)
                    }
                });
                (st, el)
            })
            .collect();
        let mut varies = vec![vec![false; pk]; w];
        loop {
            let mut changed = false;
            for k in 0..w {
                for d in 0..pk {
                    let v = deps[k].1.iter().any(|&e| elem[e][d])
                        || deps[k].0.iter().any(|&j| varies[j][d]);
                    if v && !varies[k][d] {
                        varies[k][d] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                return varies;
            }
        }
    }

    /// Constant tuple slot an epilogue arg reads from its reduction.
    pub(super) fn epilogue_slot(&self, a: &Arg) -> usize {
        let pc = self.comp_at(a.src.n()).unwrap();
        let pk = pc.points().len();
        a.view.forms[pk..]
            .iter()
            .zip(&pc.out[pk..])
            .fold(0usize, |acc, (f, d)| acc * d.n() + f.offset as usize)
    }

    /// The reduction a tiled schedule of `root` iterates: `root` itself,
    /// or the single-use single-axis reduction inlined into it when `root`
    /// is a map reading it point-for-point (its epilogue: bias, activation).
    pub fn tiled_reduction(&self, root: usize) -> Option<usize> {
        let c = self.comp_at(root)?;
        if c.monoid.is_some() {
            return Some(root);
        }
        let mut found = None;
        for a in &c.args {
            let p = a.src.n();
            let Some(pc) = self.comp_at(p) else { continue };
            if self.nodes[p].inline && pc.monoid.is_some() {
                let ok = (pc.red.len() == 1 || tileable_red(&pc.red))
                    && (found.is_none() || found == Some(p))
                    && self.epilogue_read(c, p).is_some();
                if !ok {
                    return None;
                }
                found = Some(p);
            }
        }
        found
    }

    /// Live, non-inlined comps: the buffers a backend would allocate.
    pub fn materialized(&self, outputs: &[TensorId]) -> Vec<(TensorId, Vec<Dim>)> {
        let live = self.live(outputs);
        (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline && self.comp_at(i).is_some())
            .map(|i| (TensorId(i as u32), self.nodes[i].shape.clone()))
            .collect()
    }

    pub fn count_comps(&self, outputs: &[TensorId]) -> usize {
        let live = self.live(outputs);
        (0..self.nodes.len())
            .filter(|&i| live[i] && self.comp_at(i).is_some())
            .count()
    }

    /// Monoid kinds of all live reductions.
    pub fn reduction_kinds(&self, outputs: &[TensorId]) -> Vec<MonoidKind> {
        let live = self.live(outputs);
        (0..self.nodes.len())
            .filter(|&i| live[i])
            .filter_map(|i| self.comp_at(i)?.monoid.as_ref().map(Monoid::kind))
            .collect()
    }

    // -- passes ---------------------------------------------------------------

    /// Fuse maps (comps without reduction) into their consumers by
    /// splicing their bodies and composing views. Dead nodes are dropped.
    ///
    /// A map is substituted when every consumer evaluates it at most once
    /// per element, or when it is cheap enough to recompute freely: one
    /// buffer read and no transcendental. Anything else that a reduction
    /// would re-evaluate once per output column — a normalization, a
    /// residual add, an activation with an `exp` — stays materialized,
    /// which is what a hand-written pipeline does too. Rewrites that need
    /// the fully spliced expression (`stabilize_softmax`) splice on their
    /// own, so run them first.
    pub fn substitute_maps(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        self.substitute_maps_with(outputs, true)
    }

    /// [`substitute_maps`](Self::substitute_maps) with `cost_aware =
    /// false` substituting every map unconditionally.
    pub fn substitute_maps_with(
        &self,
        outputs: &[TensorId],
        cost_aware: bool,
    ) -> (TensorGraph, Vec<TensorId>) {
        let n = self.nodes.len();
        let live = self.live(outputs);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        // a map read through a reshape stays materialized unless every
        // such reshape is affine (see `View::through_reshape`), in which
        // case the reshape dissolves into the reader's view
        let reshaped: Vec<bool> = {
            let mut r = vec![false; n];
            for i in 0..n {
                if let Op::Reshape(src) = self.nodes[i].op {
                    let rank = self.nodes[i].shape.len();
                    let affine = View::identity(rank, rank)
                        .through_reshape(&self.nodes[i].shape, &self.nodes[src.n()].shape)
                        .is_some();
                    if !affine || is_output(i) {
                        r[src.n()] = true;
                    }
                }
            }
            r
        };
        // recompute factor per producer: max over consumers of
        // consumer space / producer points (unknown when symbolic)
        let mut recompute = vec![Some(0usize); n];
        for i in (0..n).filter(|&i| live[i]) {
            if let Some(c) = self.comp_at(i) {
                let space: Option<usize> = c.space().iter().map(|d| d.fixed()).product();
                for a in &c.args {
                    let p = a.src.n();
                    let pts: Option<usize> =
                        self.nodes[p].shape.iter().map(|d| d.fixed()).product();
                    recompute[p] = match (recompute[p], space, pts) {
                        (Some(r), Some(s), Some(q)) if q > 0 => Some(r.max(s.div_ceil(q))),
                        _ => None,
                    };
                }
            }
        }
        let cheap = |c: &Comp| {
            let reads = c
                .args
                .iter()
                .filter(|a| !matches!(self.nodes[a.src.n()].op, Op::Iota(_)))
                .count();
            let live = c.body.live_set(&c.body_outs);
            let nodes = c.body.nodes_slice();
            let transcendental = live.iter().any(|&k| {
                matches!(
                    nodes[k],
                    Node::Exp2(_) | Node::Log2(_) | Node::Sqrt(_) | Node::Sin(_) | Node::Atan2(..)
                )
            });
            reads <= 1 && !transcendental
        };
        // an operator node reads whole buffers: its inputs stay materialized
        let mut op_read = vec![false; n];
        for i in (0..n).filter(|&i| live[i]) {
            if let Op::Operator(o) = &self.nodes[i].op {
                for t in &o.inputs {
                    op_read[t.n()] = true;
                }
            }
            if let Op::Scan(sc) = &self.nodes[i].op {
                for t in &sc.inputs {
                    op_read[t.n()] = true;
                }
            }
        }
        // a map read through a guarded view must stay materialized
        let mut guarded = vec![false; n];
        for i in (0..n).filter(|&i| live[i]) {
            if let Some(c) = self.comp_at(i) {
                for a in c.args.iter().filter(|a| a.view.guard) {
                    guarded[a.src.n()] = true;
                }
            }
        }
        let candidate: Vec<bool> = (0..n)
            .map(|i| {
                self.comp_at(i).is_some_and(|c| {
                    c.red.is_empty()
                        && c.tuple == 0
                        && !is_output(i)
                        && !reshaped[i]
                        && !guarded[i]
                        && !op_read[i]
                        && (!cost_aware || recompute[i] == Some(1) || cheap(c))
                })
            })
            .collect();
        let pinned = self.gather_pins(&live, &candidate);
        // a detached map is never spliced (its reader would differentiate
        // through it); where it would have been, it is scheduled inline
        // instead, so it still costs no buffer
        let spliceable: Vec<bool> = (0..n).map(|i| candidate[i] && !pinned[i]).collect();
        let inline: Vec<bool> = (0..n)
            .map(|i| spliceable[i] && !self.nodes[i].detach)
            .collect();

        let mut ng = TensorGraph {
            nodes: Vec::new(),
            n_inputs: self.n_inputs,
        };
        let mut remap: Vec<Option<usize>> = vec![None; n];
        // an affine reshape of an inlined map dissolves into its readers
        let dissolved = |i: usize| match self.nodes[i].op {
            Op::Reshape(src) => inline[src.n()],
            _ => false,
        };
        for i in (0..n).filter(|&i| live[i] && !inline[i] && !dissolved(i)) {
            let node = &self.nodes[i];
            let op = match &node.op {
                Op::Input(k) => Op::Input(*k),
                Op::Iota(d) => Op::Iota(*d),
                Op::Reshape(src) => Op::Reshape(TensorId(remap[src.n()].unwrap() as u32)),
                Op::Operator(o) => Op::Operator(OperatorNode {
                    op: o.op.clone(),
                    inputs: o
                        .inputs
                        .iter()
                        .map(|t| TensorId(remap[t.n()].unwrap() as u32))
                        .collect(),
                }),
                Op::Scan(sc) => Op::Scan(Scan {
                    inputs: sc
                        .inputs
                        .iter()
                        .map(|t| TensorId(remap[t.n()].unwrap() as u32))
                        .collect(),
                    ..sc.clone()
                }),
                Op::Comp(c) => {
                    let rank = c.rank();
                    let mut body = ExprGraph::new();
                    let mut args: Vec<Arg> = Vec::new();
                    let vars = self.splice_args(c, rank, &inline, &remap, &mut body, &mut args);
                    let body_outs = replay(&c.body, &c.body_outs, &vars, &mut body);
                    Op::Comp(Comp {
                        out: c.out.clone(),
                        tuple: c.tuple,
                        red: c.red.clone(),
                        args,
                        body,
                        body_outs,
                        monoid: c.monoid.clone(),
                    })
                }
            };
            ng.nodes.push(TNode {
                op,
                shape: node.shape.clone(),
                inline: node.inline || spliceable[i],
                split: node.split,
                tile: node.tile,
                reg: node.reg,
                fragment: node.fragment,
                int_mma: node.int_mma,
                segment: node.segment,
                sorted: node.sorted,
                detach: node.detach,
                subgroup: node.subgroup,
                frag_cfg: node.frag_cfg,
                precision: node.precision,
            });
            remap[i] = Some(ng.nodes.len() - 1);
        }
        let outs = outputs
            .iter()
            .map(|o| TensorId(remap[o.n()].unwrap() as u32))
            .collect();
        (ng, outs)
    }

    /// `comp_at(i)` with every producer map spliced in, as a standalone
    /// `Comp`. Used by rewrites that match on expressions regardless of
    /// where the materialization boundaries ended up.
    fn effective(&self, i: usize) -> Option<Comp> {
        let c = self.comp_at(i)?;
        let n = self.nodes.len();
        let candidate: Vec<bool> = (0..n)
            .map(|j| {
                j != i
                    && !self.nodes[j].detach
                    && self
                        .comp_at(j)
                        .is_some_and(|c| c.red.is_empty() && c.tuple == 0)
            })
            .collect();
        let pinned = self.gather_pins(&vec![true; n], &candidate);
        let inline: Vec<bool> = (0..n).map(|j| candidate[j] && !pinned[j]).collect();
        let remap: Vec<Option<usize>> = (0..n).map(Some).collect();
        let rank = c.rank();
        let mut body = ExprGraph::new();
        let mut args: Vec<Arg> = Vec::new();
        let vars = self.splice_args(c, rank, &inline, &remap, &mut body, &mut args);
        let body_outs = replay(&c.body, &c.body_outs, &vars, &mut body);
        Some(Comp {
            out: c.out.clone(),
            tuple: c.tuple,
            red: c.red.clone(),
            args,
            body,
            body_outs,
            monoid: c.monoid.clone(),
        })
    }

    /// Maps that must stay materialized because of gathers, given the
    /// substitution candidates:
    ///
    /// - an index tensor (read as some gather's index) is a leaf: the
    ///   address needs a value in memory, not an expression;
    /// - a map that itself gathers, when it is read through a gathered
    ///   view (directly, or through a chain of candidate maps): inlining it
    ///   would make an index read data-dependent, a nested gather.
    ///
    /// Everything else fuses: an affine map read through a gather has the
    /// gather pushed into its args, and a gathering map read affinely
    /// carries its gather (and its index arg) into the reader.
    fn gather_pins(&self, live: &[bool], candidate: &[bool]) -> Vec<bool> {
        let n = self.nodes.len();
        let mut pinned = vec![false; n];
        let mut gathered_read = vec![false; n];
        // readers sit after their producers: one descending sweep
        for r in (0..n).rev().filter(|&r| live[r]) {
            let Some(c) = self.comp_at(r) else { continue };
            for &k in &c.index_args() {
                pinned[c.args[k].src.n()] = true;
            }
            for a in &c.args {
                if a.view.is_gather() || (candidate[r] && gathered_read[r]) {
                    gathered_read[a.src.n()] = true;
                }
            }
        }
        for p in 0..n {
            if gathered_read[p] && self.comp_at(p).is_some_and(Comp::has_gather) {
                pinned[p] = true;
            }
        }
        pinned
    }

    /// Splice every arg of `c` (see [`splice`](Self::splice)): index args
    /// first, as leaves, so gathered args can name their new positions.
    fn splice_args(
        &self,
        c: &Comp,
        rank: usize,
        inline: &[bool],
        remap: &[Option<usize>],
        body: &mut ExprGraph,
        args: &mut Vec<Arg>,
    ) -> Vec<ExprId> {
        let resolved = self.resolve_gathers(&c.args, |a| a.clone(), inline, remap, args);
        resolved
            .iter()
            .map(|a| self.splice(a, rank, inline, remap, body, args))
            .collect()
    }

    /// `own` (one comp's args) with each view mapped by `view_of` and every
    /// gather's `arg` renumbered into `args`, where the index args are
    /// interned as leaves (they are pinned: never inlined).
    fn resolve_gathers(
        &self,
        own: &[Arg],
        view_of: impl Fn(&Arg) -> Arg,
        inline: &[bool],
        remap: &[Option<usize>],
        args: &mut Vec<Arg>,
    ) -> Vec<Arg> {
        let mut leaf_of: HashMap<usize, usize> = HashMap::new();
        for a in own {
            let Some(g) = &a.view.gather else { continue };
            if leaf_of.contains_key(&g.arg) {
                continue;
            }
            let ia = view_of(&own[g.arg]);
            assert!(!inline[ia.src.n()], "gather index tensors are pinned");
            let leaf = Arg {
                src: TensorId(remap[ia.src.n()].unwrap() as u32),
                view: ia.view,
            };
            leaf_of.insert(g.arg, intern_arg(args, leaf));
        }
        own.iter()
            .map(|a| {
                let mut m = view_of(a);
                // only `a`'s own gather is renumbered here; one inherited
                // through `view_of` (the reader's) is already resolved
                if let (Some(g), Some(own_g)) = (&mut m.view.gather, &a.view.gather) {
                    g.arg = leaf_of[&own_g.arg];
                }
                m
            })
            .collect()
    }

    /// Expression for `arg` inside a consumer of iteration rank `rank`:
    /// either a leaf var or the producer map's body, recursively. A gather
    /// in `arg`'s view already names its index arg's position in `args`.
    fn splice(
        &self,
        arg: &Arg,
        rank: usize,
        inline: &[bool],
        remap: &[Option<usize>],
        body: &mut ExprGraph,
        args: &mut Vec<Arg>,
    ) -> ExprId {
        let p = arg.src.n();
        // an affine reshape of an inlineable map: read the map directly
        // (not under a guard: its bounds are the reshape's shape)
        if let Op::Reshape(src) = self.nodes[p].op {
            if inline[src.n()] && !arg.view.guard {
                if let Some(view) = arg
                    .view
                    .through_reshape(&self.nodes[p].shape, &self.nodes[src.n()].shape)
                {
                    let through = Arg { src, view };
                    return self.splice(&through, rank, inline, remap, body, args);
                }
            }
        }
        // a guarded read of a map is 0 outside, not the map of 0: keep it a leaf
        if !inline[p] || arg.view.guard {
            let leaf = Arg {
                src: TensorId(remap[p].unwrap() as u32),
                view: arg.view.clone(),
            };
            return body.var(intern_arg(args, leaf) as u16);
        }
        let c = self.comp_at(p).unwrap();
        // the reader's view pushed into each of the map's args; the map's
        // own gathers (only when the read is affine: see `gather_pins`)
        // renumbered onto its index args, interned as leaves
        let composed = self.resolve_gathers(
            &c.args,
            |a| Arg {
                src: a.src,
                view: a.view.compose(&arg.view, rank),
            },
            inline,
            remap,
            args,
        );
        let vars: Vec<ExprId> = composed
            .iter()
            .map(|a| self.splice(a, rank, inline, remap, body, args))
            .collect();
        replay(&c.body, &c.body_outs, &vars, body)[0]
    }

    /// Canonical form of the expression `out` of `body` over `args`, with
    /// views over `space`: leaves renumbered in order of first use and
    /// size-1 coordinates dropped, so equal computations compare equal.
    fn canon(body: &ExprGraph, out: ExprId, args: &[Arg], space: &[Dim]) -> Canon {
        let mut leaves: Vec<Arg> = Vec::new();
        let mut g = ExprGraph::new();
        let vars: Vec<ExprId> = args
            .iter()
            .enumerate()
            .map(|(k, a)| {
                if !vars_reachable(body, out).contains(&(k as u16)) {
                    return ExprId::ZERO;
                }
                let leaf = Arg {
                    src: a.src,
                    view: a.view.normalized(space),
                };
                g.var(intern_arg(&mut leaves, leaf) as u16)
            })
            .collect();
        let out = replay(body, &[out], &vars, &mut g)[0];
        Canon {
            leaves,
            nodes: g.nodes_slice()[..=out.index() as usize].to_vec(),
        }
    }

    /// Does `m_arg` (as read by `c`) hold `max` over `c`'s reduce coord of
    /// exactly the expression `s` of `c`'s body? Returns `s`'s canonical form.
    fn is_row_max_of(&self, c: &Comp, s: ExprId, m_arg: &Arg) -> Option<Canon> {
        let mc = &self.effective(m_arg.src.n())?;
        // gathers name arg positions, which differ between `c` and `mc`;
        // the softmax rewrite (which re-interns args) leaves them alone
        if mc.has_gather() || c.has_gather() || m_arg.view.is_gather() {
            return None;
        }
        let n_out = c.points().len();
        let structural = mc.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Max)
            && mc.tuple == 0
            && mc.red.len() == 1
            && c.red.len() == 1
            && !m_arg.view.depends_on(n_out);
        if !structural {
            return None;
        }
        // Extend c's view of m with "m's reduce coord = c's reduce coord",
        // push m's args through it, and compare m's body with `s`.
        let mut ext = m_arg.view.clone();
        ext.forms.push(Form::unit(c.rank(), n_out));
        let space = c.space();
        let m_args: Vec<Arg> = mc
            .args
            .iter()
            .map(|a| Arg {
                src: a.src,
                view: a.view.compose(&ext, c.rank()),
            })
            .collect();
        let via_m = Self::canon(&mc.body, mc.body_outs[0], &m_args, &space);
        let direct = Self::canon(&c.body, s, &c.args, &space);
        (via_m == direct).then_some(direct)
    }

    /// Rewrite `m = max_j s` and every `Σ_j g·exp(s − m)` over the same
    /// `s` into a single online-softmax reduction. Run after
    /// [`substitute_maps`](Self::substitute_maps) so `exp(s − m)` is visible
    /// in reduction bodies.
    ///
    /// Members may have extra out dims beyond `m`'s (e.g. `Σ_j p[i,j]·v[j,c]`
    /// has a `c`); those are unrolled into the fused comp's tuple dim.
    pub fn stabilize_softmax(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        struct Member {
            node: usize,
            m_view: View,
            s: ExprId,
            g: Option<ExprId>,
        }
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        // (max node, canonical s) -> members
        let mut groups: HashMap<(usize, Canon), Vec<Member>> = HashMap::new();
        for i in 0..self.nodes.len() {
            let Some(c) = self.effective(i) else { continue };
            let c = &c;
            let sum = c.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Sum);
            if !sum || c.red.len() != 1 || c.tuple != 0 || is_output(i) {
                continue;
            }
            let Some((s, m_var, g)) = match_exp_diff(&c.body, c.body_outs[0]) else {
                continue;
            };
            let m_arg = &c.args[m_var as usize];
            if let Some(canon) = self.is_row_max_of(c, s, m_arg) {
                groups
                    .entry((m_arg.src.n(), canon))
                    .or_default()
                    .push(Member {
                        node: i,
                        m_view: m_arg.view.clone(),
                        s,
                        g,
                    });
            }
        }
        if groups.is_empty() {
            return (self.clone(), outputs.to_vec());
        }

        let mut g2 = self.clone();
        // member node -> (fused node, projection from member out coords to fused coords)
        let mut redirect: HashMap<usize, (usize, View)> = HashMap::new();
        for ((m_node, _), members) in groups {
            let mc = &self.effective(m_node).unwrap();
            let m_out = mc.out.len();

            // Extra out dims of members (e.g. the value column of Σ_j p·v)
            // become point dims of the fused comp, so a lane per column
            // can share the reduction. All members must agree on them.
            let member_extra: Vec<(Vec<Option<usize>>, Vec<usize>)> = members
                .iter()
                .map(|mem| {
                    let c = self.effective(mem.node).unwrap();
                    let n_out = c.points().len();
                    let reads: Vec<Option<usize>> = (0..n_out)
                        .map(|d| mem.m_view.forms.iter().position(|f| f.as_unit() == Some(d)))
                        .collect();
                    // size-1 dims carry no lanes; they are neither read nor extra
                    let extra: Vec<usize> = (0..n_out)
                        .filter(|&d| reads[d].is_none() && c.out[d] != Dim::Fixed(1))
                        .collect();
                    (reads, extra)
                })
                .collect();
            let extra_shape: Vec<Dim> = member_extra
                .iter()
                .zip(&members)
                .map(|((_, extra), mem)| {
                    let c = self.effective(mem.node).unwrap();
                    extra.iter().map(|&d| c.out[d]).collect::<Vec<Dim>>()
                })
                .max_by_key(|v| v.len())
                .unwrap_or_default();
            let agree = member_extra.iter().zip(&members).all(|((_, extra), mem)| {
                let c = self.effective(mem.node).unwrap();
                let mine: Vec<Dim> = extra.iter().map(|&d| c.out[d]).collect();
                mine.is_empty() || mine == extra_shape
            });
            if !agree {
                continue;
            }
            let e = extra_shape.len();
            let fused_rank = m_out + e + 1;

            let mut body = ExprGraph::new();
            let mut args: Vec<Arg> = Vec::new();
            // slot 0: s, from the max's own body, re-indexed into the fused space
            let m_ext = View {
                guard: false,
                gather: None,
                forms: (0..=m_out)
                    .map(|d| Form::unit(fused_rank, if d == m_out { m_out + e } else { d }))
                    .collect(),
            };
            let m_vars: Vec<ExprId> = mc
                .args
                .iter()
                .map(|a| {
                    let leaf = Arg {
                        src: a.src,
                        view: a.view.compose(&m_ext, fused_rank),
                    };
                    body.var(intern_arg(&mut args, leaf) as u16)
                })
                .collect();
            let mut outs = replay(&mc.body, &mc.body_outs, &m_vars, &mut body);
            let mut projections = Vec::new();

            for (mem, (reads, extra)) in members.iter().zip(&member_extra) {
                let c = &self.effective(mem.node).unwrap();
                let n_out = c.points().len();
                let slot = outs.len() as i64;
                let inv = View {
                    guard: false,
                    gather: None,
                    forms: (0..c.rank())
                        .map(|d| match reads.get(d) {
                            Some(Some(j)) => Form::unit(fused_rank, *j),
                            // lane dim, or a size-1 dim that is always 0
                            Some(None) => match extra.iter().position(|x| *x == d) {
                                Some(k) => Form::unit(fused_rank, m_out + k),
                                None => Form::zero(fused_rank),
                            },
                            None => Form::unit(fused_rank, m_out + e),
                        })
                        .collect(),
                };
                let _ = mem.s;
                let used = mem.g.map_or_else(Vec::new, |g| vars_reachable(&c.body, g));
                let vars: Vec<ExprId> = c
                    .args
                    .iter()
                    .enumerate()
                    .map(|(k, a)| {
                        if !used.contains(&(k as u16)) {
                            return ExprId::ZERO;
                        }
                        let leaf = Arg {
                            src: a.src,
                            view: a.view.compose(&inv, fused_rank),
                        };
                        body.var(intern_arg(&mut args, leaf) as u16)
                    })
                    .collect();
                outs.push(match mem.g {
                    Some(g) => replay(&c.body, &[g], &vars, &mut body)[0],
                    None => body.lit(1.0),
                });

                // member[o] == fused[m_view(o), extra coords (0 if absent), slot]
                let mut proj: Vec<Form> = mem
                    .m_view
                    .forms
                    .iter()
                    .map(|f| Form {
                        coefs: f.coefs[..n_out].to_vec(),
                        offset: f.offset,
                    })
                    .collect();
                for k in 0..e {
                    proj.push(match extra.get(k) {
                        Some(&d) => Form::unit(n_out, d),
                        None => Form::zero(n_out),
                    });
                }
                proj.push(Form::constant(n_out, slot));
                projections.push((
                    mem.node,
                    View {
                        guard: false,
                        gather: None,
                        forms: proj,
                    },
                ));
            }

            let width = outs.len();
            let mut out = mc.out.clone();
            out.extend(extra_shape.iter().copied());
            out.push(Dim::Fixed(width));
            let fused = g2
                .push_comp(Comp {
                    out,
                    tuple: 1,
                    red: mc.red.clone(),
                    args,
                    body,
                    body_outs: outs,
                    monoid: Some(Monoid::online_softmax(width - 1)),
                })
                .expect("fused softmax comp is well-formed");
            for (node, proj) in projections {
                redirect.insert(node, (fused.n(), proj));
            }
            // The online state's slot 0 is the same max, bit for bit (max
            // is exact in any order), so the max's other readers (the
            // backward's `exp(s − m)`) read it there instead of a second
            // reduction over `s`.
            if !is_output(m_node) {
                let proj = View {
                    guard: false,
                    gather: None,
                    forms: (0..m_out)
                        .map(|d| Form::unit(m_out, d))
                        .chain((0..=e).map(|_| Form::zero(m_out)))
                        .collect(),
                };
                redirect.entry(m_node).or_insert((fused.n(), proj));
            }
        }
        if redirect.is_empty() {
            return (self.clone(), outputs.to_vec());
        }

        for node in &mut g2.nodes {
            let Op::Comp(c) = &mut node.op else { continue };
            let rank = c.rank();
            for a in &mut c.args {
                if let Some((fused, proj)) = redirect.get(&a.src.n()) {
                    a.view = proj.compose(&a.view, rank);
                    a.src = TensorId(*fused as u32);
                }
            }
        }
        g2.eliminate_dead(outputs)
    }

    /// Live nodes in dependency order (post-order DFS from the outputs).
    /// Live inputs come first, in their original order, so a graph's
    /// input ids survive every pass.
    fn topo(&self, outputs: &[TensorId]) -> Vec<usize> {
        let live = self.live(outputs);
        let mut order: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| live[i] && matches!(self.nodes[i].op, Op::Input(_)))
            .collect();
        let mut seen = vec![false; self.nodes.len()];
        for &i in &order {
            seen[i] = true;
        }
        let mut stack: Vec<(usize, bool)> = outputs.iter().map(|o| (o.n(), false)).collect();
        while let Some((i, done)) = stack.pop() {
            if done {
                order.push(i);
                continue;
            }
            if std::mem::replace(&mut seen[i], true) {
                continue;
            }
            stack.push((i, true));
            stack.extend(self.deps(i).into_iter().map(|d| (d, false)));
        }
        order
    }

    /// Drop dead nodes and renumber in dependency order.
    pub fn eliminate_dead(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        let mut remap = vec![None; self.nodes.len()];
        let mut ng = TensorGraph {
            nodes: Vec::new(),
            n_inputs: self.n_inputs,
        };
        for i in self.topo(outputs) {
            let mut node = self.nodes[i].clone();
            match &mut node.op {
                Op::Comp(c) => {
                    for a in &mut c.args {
                        a.src = TensorId(remap[a.src.n()].unwrap() as u32);
                    }
                }
                Op::Reshape(src) => *src = TensorId(remap[src.n()].unwrap() as u32),
                Op::Operator(o) => {
                    for t in &mut o.inputs {
                        *t = TensorId(remap[t.n()].unwrap() as u32);
                    }
                }
                Op::Scan(sc) => {
                    for t in &mut sc.inputs {
                        *t = TensorId(remap[t.n()].unwrap() as u32);
                    }
                }
                Op::Input(_) | Op::Iota(_) => {}
            }
            ng.nodes.push(node);
            remap[i] = Some(ng.nodes.len() - 1);
        }
        let outs = outputs
            .iter()
            .map(|o| TensorId(remap[o.n()].unwrap() as u32))
            .collect();
        (ng, outs)
    }

    /// Schedule: recompute a single-use reduction inside its consumer when
    /// the consumer's iteration space is no larger than the producer's
    /// point count, i.e. inlining evaluates each point at most once.
    pub fn schedule_inline(&mut self, outputs: &[TensorId]) {
        self.schedule_inline_up_to(outputs, 1);
    }

    /// Like [`schedule_inline`](Self::schedule_inline) but allows the
    /// consumer to evaluate the producer up to `factor` times per point.
    /// Worth it when the consumer's extra points are lanes that a tiled
    /// schedule computes the shared part for cooperatively.
    pub fn schedule_inline_up_to(&mut self, outputs: &[TensorId], factor: usize) {
        let live = self.live(outputs);
        let uses = self.use_counts(outputs, &live);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let n = self.nodes.len();
        let mut consumer_space = vec![0usize; n];
        for i in (0..n).filter(|&i| live[i]) {
            if let Some(c) = self.comp_at(i) {
                for a in &c.args {
                    consumer_space[a.src.n()] = numel(&c.space());
                }
            }
        }
        let reshaped: Vec<bool> = {
            let mut r = vec![false; n];
            for i in 0..n {
                if let Op::Reshape(src) = self.nodes[i].op {
                    r[src.n()] = true;
                }
            }
            r
        };
        // A reduction may be absorbed by a reduction, or by a map that
        // reads it point-for-point (the map becomes the reduction's
        // epilogue — bias, activation — and the tiled schedules see
        // through it via `tiled_reduction`). Any other map consumer would
        // hide the reduction from tiling, so it stays materialized.
        let mut consumer_reduces = vec![false; n];
        for i in (0..n).filter(|&i| live[i]) {
            if let Some(c) = self.comp_at(i) {
                let mut reductions: Vec<usize> = c
                    .args
                    .iter()
                    .map(|a| a.src.n())
                    .filter(|&p| self.comp_at(p).is_some_and(|pc| pc.monoid.is_some()))
                    .collect();
                reductions.dedup();
                let epilogue_ok = c.monoid.is_none() && reductions.len() == 1;
                for a in &c.args {
                    let p = a.src.n();
                    let as_epilogue = epilogue_ok
                        && self
                            .comp_at(p)
                            .is_some_and(|pc| pc.red.len() == 1 || tileable_red(&pc.red))
                        && self.epilogue_read(c, p).is_some();
                    // a fold of partial folds with the same monoid (a
                    // split-K combine) inlined is the split undone
                    let refold =
                        match (&c.monoid, self.comp_at(p).and_then(|pc| pc.monoid.as_ref())) {
                            (Some(a), Some(b)) => a.width == b.width && a.kind() == b.kind(),
                            _ => false,
                        };
                    consumer_reduces[p] = (c.monoid.is_some() && !refold) || as_epilogue;
                }
            }
        }
        for i in 0..n {
            let Some(c) = self.comp_at(i) else { continue };
            if live[i]
                && uses[i] == 1
                && !is_output(i)
                && !reshaped[i]
                && (c.monoid.is_none() || consumer_reduces[i])
                && consumer_space[i] <= factor * numel(c.points())
            {
                self.nodes[i].inline = true;
            }
        }
    }

    /// Schedule: every materialized reduction with at least `min_red`
    /// elements per point is split across a workgroup of `threads`.
    /// The monoid's combine is associative, so this is always valid —
    /// including for tuple-state monoids like online softmax.
    pub fn schedule_split(&mut self, outputs: &[TensorId], threads: u32, min_red: usize) {
        let live = self.live(outputs);
        let targets: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline)
            .filter(|&i| {
                self.comp_at(i)
                    .is_some_and(|c| c.monoid.is_some() && numel(&c.red) >= min_red)
            })
            .collect();
        for i in targets {
            self.nodes[i].split = threads;
        }
    }

    /// Schedule: materialized single-axis reductions with at least
    /// `min_red` elements per point get `points` threads per workgroup
    /// (one point each) and stage workgroup-invariant reads in chunks of
    /// `chunk` reduce indices. This is the structure of flash attention
    /// and of a shared-memory matmul, derived from the views alone.
    pub fn schedule_tile(&mut self, outputs: &[TensorId], points: u32, chunk: u32, min_red: usize) {
        self.schedule_tile_2d(outputs, points, 1, chunk, min_red);
    }

    /// Like [`schedule_tile`](Self::schedule_tile) with a `bx × by` block
    /// of points per workgroup over the last two point dims. Reads that
    /// vary along only one tile axis (matmul's `a[i,k]`, `b[k,j]`) are
    /// staged once per workgroup instead of once per thread.
    pub fn schedule_tile_2d(
        &mut self,
        outputs: &[TensorId],
        bx: u32,
        by: u32,
        chunk: u32,
        min_red: usize,
    ) {
        let live = self.live(outputs);
        let targets: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline && self.nodes[i].split == 0)
            .filter(|&i| {
                self.comp_at(i).is_some_and(|c| {
                    c.monoid.is_some()
                        && tileable_red(&c.red)
                        && c.red.iter().map(|d| d.n()).max().unwrap_or(0) >= min_red
                        && (by == 1 || c.points().len() >= 2)
                })
            })
            .collect();
        for i in targets {
            self.nodes[i].tile = (bx, by, chunk);
        }
    }

    /// Replace symbolic dims by sizes. Every `Sym` in the graph must be
    /// named; the result is a concrete graph that can be scheduled,
    /// evaluated and lowered.
    pub fn bind(&self, bindings: &[(&str, usize)]) -> Result<TensorGraph, ShapeError> {
        let sub = |d: &Dim| match d {
            Dim::Fixed(n) => Ok(Dim::Fixed(*n)),
            Dim::Sym(name) => bindings
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| Dim::Fixed(*v))
                .ok_or_else(|| ShapeError(format!("bind: no value for dim `{name}`"))),
        };
        let mut g = self.clone();
        for node in &mut g.nodes {
            node.shape = node.shape.iter().map(sub).collect::<Result<_, _>>()?;
            if let Op::Comp(c) = &mut node.op {
                c.out = c.out.iter().map(sub).collect::<Result<_, _>>()?;
                c.red = c.red.iter().map(sub).collect::<Result<_, _>>()?;
            }
        }
        Ok(g)
    }

    /// `true` if no dim is symbolic.
    pub fn is_bound(&self) -> bool {
        self.nodes.iter().all(|n| {
            let shape_ok = n.shape.iter().all(|d| d.fixed().is_some());
            let comp_ok = match &n.op {
                Op::Comp(c) => c.red.iter().all(|d| d.fixed().is_some()),
                Op::Input(_) | Op::Iota(_) | Op::Reshape(_) | Op::Operator(_) | Op::Scan(_) => true,
            };
            shape_ok && comp_ok
        })
    }

    /// Default thread mapping for a bound graph, from the measured rules
    /// of thumb in `BENCHMARKS.md`: a reduction whose points end in a lane
    /// axis (attention's value columns) is tiled rows × lanes with shared
    /// work staged cooperatively; a reduction with few output points is
    /// split across a workgroup; one with many points is already parallel
    /// enough one-thread-per-point.
    /// `autotune` over candidate graphs still decides which *algebra* to
    /// run (e.g. materialized scores vs online softmax).
    pub fn schedule_auto(&mut self, outputs: &[TensorId]) {
        const MANY_POINTS: usize = 1 << 14;
        const LANES: usize = 16;
        self.schedule_inline(outputs);
        let live = self.live(outputs);
        let reductions: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline)
            .filter(|&i| {
                self.tiled_reduction(i)
                    .and_then(|r| self.comp_at(r))
                    .is_some_and(|c| numel(&c.red) >= 16)
            })
            .collect();
        let uses = self.use_counts(outputs, &live);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        // a map every one of whose consumers reduces may be recomputed in
        // each of them (flash backward recomputes dS in dQ and dK alike)
        let mut consumers_reduce = vec![true; self.nodes.len()];
        for j in (0..self.nodes.len()).filter(|&j| live[j]) {
            match &self.nodes[j].op {
                Op::Comp(c) => {
                    for a in &c.args {
                        if c.monoid.is_none() {
                            consumers_reduce[a.src.n()] = false;
                        }
                    }
                }
                Op::Reshape(src) => consumers_reduce[src.n()] = false,
                _ => {}
            }
        }
        // ... and only when it is bigger than what they produce, so writing
        // it out would cost more traffic than recomputing it
        let mut consumer_points = vec![0usize; self.nodes.len()];
        for j in (0..self.nodes.len()).filter(|&j| live[j]) {
            if let Some(c) = self.comp_at(j) {
                let pts = numel(c.points());
                for a in &c.args {
                    consumer_points[a.src.n()] = consumer_points[a.src.n()].max(pts);
                }
            }
        }
        let recomputable: Vec<bool> = (0..self.nodes.len())
            .map(|p| {
                uses[p] == 1
                    || (consumers_reduce[p]
                        && self.comp_at(p).is_some_and(|pc| {
                            pc.red.is_empty() && numel(pc.points()) > consumer_points[p]
                        }))
            })
            .collect();
        for i in reductions {
            // schedule by the reduction's shape, even when `i` is its epilogue
            let c = self.comp_at(self.tiled_reduction(i).unwrap()).unwrap();
            // tiles stage affine operand regions: a gathered read gets the
            // split (or one thread per point), never a tile
            if c.has_gather() || self.comp_at(i).unwrap().has_gather() {
                if numel(c.points()) < MANY_POINTS && c.width() <= 8 {
                    self.nodes[i].split = 256;
                }
                continue;
            }
            let pts = fixed(c.points());
            let (lane_dim, row_dim) = tile_axes(&pts);
            let lanes = lane_dim.map_or(1, |d| pts[d]);
            let rows_n = row_dim.map_or(1, |d| pts[d]);
            let rows = row_dim.is_some();
            // Producers the lane axis does not depend on: a tile computes
            // them once per workgroup instead of once per lane.
            let shared: Vec<usize> = lane_dim
                .map(|lane| {
                    c.args
                        .iter()
                        .filter(|a| !a.view.depends_on(lane) && recomputable[a.src.n()])
                        .map(|a| a.src.n())
                        .filter(|&p| !is_output(p) && self.comp_at(p).is_some())
                        .collect()
                })
                .unwrap_or_default();
            let by = (256 / lanes).max(1) as u32;
            if c.red.len() == 1 && (LANES..=256).contains(&lanes) && rows && !shared.is_empty() {
                // Rows × lanes per workgroup with the shared sub-computation
                // staged cooperatively. Without one (a plain matmul) the
                // tile has nothing to share and one thread per point wins.
                self.nodes[i].tile = (lanes as u32, by, 32);
                for p in shared {
                    self.nodes[p].inline = true;
                }
            } else if tileable_red(&c.red) && lanes >= 16 && rows && {
                // reuse: some operand is shared along the lanes and some
                // along the rows — otherwise a tile only adds staging
                let row_dim = lane_dim.and_then(|dx| tile::largest_other(&pts, dx));
                let real = |a: &&Arg| !matches!(self.nodes[a.src.n()].op, Op::Iota(_));
                let lane = lane_dim.unwrap();
                c.args.iter().filter(real).any(|a| !a.view.depends_on(lane))
                    && row_dim
                        .is_some_and(|r| c.args.iter().filter(real).any(|a| !a.view.depends_on(r)))
            } {
                // matmul-shaped: the 16×16 shared-memory tile, with a 4×4
                // register tile once both axes are wide enough to fill it
                // (won on an M4 and a 3090 alike; see BENCHMARKS.md)
                let reg = if lanes >= 64 && rows_n >= 64 {
                    (4, 4)
                } else {
                    (1, 1)
                };
                self.nodes[i].tile = (16, 16, 32);
                self.nodes[i].reg = reg;
            } else if pts.iter().product::<usize>() < MANY_POINTS {
                self.nodes[i].split = 256;
                // subgroup combine where the target has it (ignored elsewhere)
                self.nodes[i].subgroup = true;
            }
            if self.nodes[i].tile.0 > 0 {
                self.unfuse_heavy_epilogue(i);
            }
        }
    }

    /// Schedule: nest short contractions in the tiles that read them.
    /// A sum of products over one reduce dim of at most 128 (`S = Q·Kᵀ`
    /// over the head dim, `dP = dO·Vᵀ`) whose every reader is a tile, or
    /// a map recomputed in tiles, is inlined: a tile lowering that can
    /// run it as a small GEMM per staged block (CUDA `mma`) recomputes it
    /// instead of reading it back. The scalar tiled lowering (WGSL, MSL)
    /// computes it as a dot per element of the shared comp it feeds, once
    /// per chunk into workgroup memory; see
    /// [`schedule_nested_contractions_scalar`](Self::schedule_nested_contractions_scalar).
    /// Applied by the GPU lowerings, after [`schedule_auto`](Self::schedule_auto).
    pub fn schedule_nested_contractions(&mut self, outputs: &[TensorId]) {
        self.nest_contractions(outputs, false);
    }

    /// [`schedule_nested_contractions`](Self::schedule_nested_contractions)
    /// for the scalar tiled lowering (WGSL, MSL): also nests through a
    /// tile's epilogue-inlined reduction, and reshapes each tile that
    /// recomputes a nested contraction to more rows per workgroup.
    pub fn schedule_nested_contractions_scalar(&mut self, outputs: &[TensorId]) {
        self.nest_contractions(outputs, true);
    }

    fn nest_contractions(&mut self, outputs: &[TensorId], scalar: bool) {
        let mut nested: Vec<usize> = Vec::new();
        let live = self.live(outputs);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let n = self.nodes.len();
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in (0..n).filter(|&j| live[j]) {
            for d in self.deps(j) {
                readers[d].push(j);
            }
        }
        // The scalar lowering also reads through a tile's own reduction,
        // inlined under its epilogue map (flash backward's dK = (Σ dS·Q)·c).
        let mut tile = vec![false; n];
        for j in (0..n).filter(|&j| live[j] && self.nodes[j].tile.0 > 0) {
            tile[j] = true;
            if let Some(r) = self.tiled_reduction(j).filter(|_| scalar) {
                tile[r] = true;
            }
        }
        // a map that is inline and read only by tiles (or such maps)
        let mut tiled_map = vec![false; n];
        for j in (0..n).rev().filter(|&j| live[j]) {
            let map = self.comp_at(j).is_some_and(|c| c.red.is_empty());
            tiled_map[j] = map
                && self.nodes[j].inline
                && !readers[j].is_empty()
                && readers[j].iter().all(|&r| tile[r] || tiled_map[r]);
        }
        for i in (0..n).filter(|&i| live[i] && !is_output(i)) {
            let Some(c) = self.comp_at(i) else { continue };
            let sum = c.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Sum);
            let short = c.red.len() == 1 && c.red[0].fixed().is_some_and(|r| r <= 128);
            if !sum || !short || c.tuple != 0 || c.width() != 1 || readers[i].is_empty() {
                continue;
            }
            let direct = |r: usize| tile[r];
            if !readers[i].iter().all(|&r| tiled_map[r] || direct(r)) {
                continue;
            }
            // only plain sums of two-leaf products nest as GEMMs
            if bilinear_terms(&c.body, c.body_outs[0]).is_none()
                || c.args.iter().any(|a| {
                    self.comp_at(a.src.n())
                        .is_some_and(|_| self.nodes[a.src.n()].inline)
                })
            {
                continue;
            }
            self.nodes[i].inline = true;
            self.nodes[i].tile = (0, 0, 0);
            self.nodes[i].split = 0;
            nested.push(i);
        }
        if !scalar {
            return;
        }
        // A tile that recomputes a nested contraction computes it per
        // chunk for each of its rows: give it rows enough to fill the
        // workgroup (16 rows × 16 threads of 4 lanes) rather than 4 rows
        // of 64 lanes, where most threads would idle through the dots.
        let tiles: Vec<usize> = (0..n)
            .filter(|&t| live[t] && self.nodes[t].tile.0 > 0)
            .collect();
        for t in tiles {
            let (bx, by, _) = self.nodes[t].tile;
            let (rx, ry) = self.nodes[t].reg;
            let (lanes, rows) = (bx * rx.max(1), by * ry.max(1));
            if rows >= 16 || lanes < 64 || lanes % 4 != 0 {
                continue;
            }
            if !self.inline_closure(t).iter().any(|j| nested.contains(j)) {
                continue;
            }
            self.nodes[t].tile = (lanes / 4, 256 / (lanes / 4), 16);
            self.nodes[t].reg = (4, 1);
        }
    }

    /// Schedule heuristic for a tiled epilogue map `m` (see
    /// [`tiled_reduction`](Self::tiled_reduction)): measured on an M4, a
    /// GELU epilogue (exp2 + recip per point, once per register-tile
    /// copy) made ConvNeXt's F×C matmul slower fused (0.60 vs 0.56 ms
    /// block) while snake/bias/residual epilogues win. Under a register
    /// tile, an epilogue with exp/log hands the tile back to its
    /// reduction and runs as its own map. Returns whether it unfused.
    pub fn unfuse_heavy_epilogue(&mut self, m: usize) -> bool {
        let Some(r) = self.tiled_reduction(m).filter(|&r| r != m) else {
            return false;
        };
        if self.nodes[m].reg == (1, 1) {
            return false;
        }
        let heavy = self.comp_at(m).is_some_and(|c| {
            c.body
                .nodes_slice()
                .iter()
                .any(|n| matches!(n, Node::Exp2(_) | Node::Log2(_)))
        });
        if !heavy {
            return false;
        }
        self.nodes[r].inline = false;
        self.nodes[r].tile = self.nodes[m].tile;
        self.nodes[r].reg = self.nodes[m].reg;
        self.nodes[m].tile = (0, 0, 0);
        self.nodes[m].reg = (1, 1);
        true
    }

    /// Common-subexpression elimination over comps: two live comps with
    /// the same args, views, body, reduce dims and monoid are one comp.
    /// Backward graphs produce many such twins (the same strided read of
    /// a weight for several gradients).
    pub fn eliminate_common(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        let n = self.nodes.len();
        let mut g = self.clone();
        let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut redirect: Vec<usize> = (0..n).collect();
        for i in 0..n {
            // rewrite args through earlier redirects first
            if let Op::Comp(c) = &mut g.nodes[i].op {
                for a in &mut c.args {
                    a.src = TensorId(redirect[a.src.n()] as u32);
                }
            } else if let Op::Reshape(src) = &mut g.nodes[i].op {
                *src = TensorId(redirect[src.n()] as u32);
            } else if let Op::Operator(o) = &mut g.nodes[i].op {
                for t in &mut o.inputs {
                    *t = TensorId(redirect[t.n()] as u32);
                }
            } else if let Op::Scan(sc) = &mut g.nodes[i].op {
                for t in &mut sc.inputs {
                    *t = TensorId(redirect[t.n()] as u32);
                }
            }
            let Op::Comp(c) = &g.nodes[i].op else {
                continue;
            };
            let key = format!(
                "{}|{:?}|{:?}|{}|{:?}|{:?}|{:?}|{:?}",
                g.nodes[i].detach,
                c.out,
                c.red,
                c.tuple,
                c.args
                    .iter()
                    .map(|a| (
                        a.src.n(),
                        a.view
                            .forms
                            .iter()
                            .map(|f| (f.coefs.clone(), f.offset))
                            .collect::<Vec<_>>(),
                        a.view.guard,
                        &a.view.gather,
                    ))
                    .collect::<Vec<_>>(),
                c.body.nodes_slice(),
                c.body_outs,
                c.monoid.as_ref().map(|m| (
                    m.identity.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    m.combine.nodes_slice().to_vec(),
                    m.outs.clone()
                ))
            )
            .into_bytes();
            match seen.get(&key) {
                Some(&first) => redirect[i] = first,
                None => {
                    seen.insert(key, i);
                }
            }
        }
        let outs: Vec<TensorId> = outputs
            .iter()
            .map(|o| TensorId(redirect[o.n()] as u32))
            .collect();
        g.eliminate_dead(&outs)
    }

    /// Layout: a materialized comp read by exactly one consumer, through
    /// one pick view, gets its point dims reordered to the consumer's
    /// liking — the dims the consumer iterates as points first (in its
    /// order), then size-1 dims, then the dims it reduces over (in its
    /// order). Its reduce dims are then adjacent in the buffer, so
    /// `merge_reduce_dims` can fold them, and the tile schedules still
    /// find their row axis (the largest remaining dim) wherever it is.
    /// Same bytes computed, different order; nothing else changes.
    pub fn relayout(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        let mut g = self.clone();
        let live = self.live(outputs);
        let n = self.nodes.len();
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        // (consumer, arg index) pairs per producer; a reshape reader
        // pins the layout
        let mut readers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut pinned = vec![false; n];
        for i in (0..n).filter(|&i| live[i]) {
            match &self.nodes[i].op {
                Op::Comp(c) => {
                    for (k, a) in c.args.iter().enumerate() {
                        readers[a.src.n()].push((i, k));
                    }
                }
                Op::Reshape(src) => pinned[src.n()] = true,
                _ => {}
            }
        }
        // which consumer dim each producer point dim is picked from, when
        // the view is a pick
        let picks_of = |view: &View, kp: usize| -> Option<Vec<Option<usize>>> {
            // a gathered read addresses the producer by data: its layout
            // is pinned (the gather's coefs name source dims)
            if view.is_gather() {
                return None;
            }
            view.forms[..kp]
                .iter()
                .map(|f| {
                    if f.coefs.iter().all(|&c| c == 0) {
                        return Some(None);
                    }
                    let d = (0..f.coefs.len()).find(|&d| f.coefs[d] != 0)?;
                    (f.coefs[d] == 1
                        && f.offset == 0
                        && f.coefs.iter().filter(|&&c| c != 0).count() == 1)
                        .then_some(Some(d))
                })
                .collect()
        };
        for p in
            (0..n).filter(|&p| live[p] && !is_output(p) && !pinned[p] && !readers[p].is_empty())
        {
            let Some(pc) = self.comp_at(p) else { continue };
            let kp = pc.points().len();
            // every reader must read by a pick (so its forms can follow a
            // permutation); the layout follows the reader that reduces
            // over the most of the producer's dims
            let mut best: Option<(usize, Vec<Option<usize>>, usize)> = None;
            let mut all_picks = true;
            for &(ci, k) in &readers[p] {
                let cc = self.comp_at(ci).unwrap();
                let Some(picks) = picks_of(&cc.args[k].view, kp) else {
                    all_picks = false;
                    break;
                };
                let kc = cc.points().len();
                let n_red = picks.iter().filter(|s| s.is_some_and(|s| s >= kc)).count();
                if best.as_ref().is_none_or(|b| n_red > b.0) {
                    best = Some((n_red, picks, kc));
                }
            }
            if !all_picks {
                continue;
            }
            let Some((_, picks, kc)) = best else { continue };
            let mut order: Vec<usize> = (0..kp)
                .filter(|&d| picks[d].is_some_and(|s| s < kc))
                .collect();
            order.sort_by_key(|&d| picks[d]);
            let mut units: Vec<usize> = (0..kp).filter(|&d| picks[d].is_none()).collect();
            let mut reds: Vec<usize> = (0..kp)
                .filter(|&d| picks[d].is_some_and(|s| s >= kc))
                .collect();
            reds.sort_by_key(|&d| picks[d]);
            order.append(&mut units);
            order.append(&mut reds);
            if order.iter().enumerate().all(|(i, &d)| i == d) {
                continue;
            }
            // producer: permute out (points) and every arg's point columns
            let Op::Comp(pm) = &mut g.nodes[p].op else {
                unreachable!()
            };
            let out_points: Vec<Dim> = order.iter().map(|&d| pm.out[d]).collect();
            for (i, d) in out_points.into_iter().enumerate() {
                pm.out[i] = d;
            }
            for a in &mut pm.args {
                for f in &mut a.view.forms {
                    let cols: Vec<i64> = order.iter().map(|&d| f.coefs[d]).collect();
                    f.coefs[..kp].copy_from_slice(&cols);
                }
            }
            g.nodes[p].shape = pm.out.clone();
            // consumers: their forms into the producer follow the new order
            for &(ci, k) in &readers[p] {
                let Op::Comp(cm) = &mut g.nodes[ci].op else {
                    unreachable!()
                };
                let forms: Vec<Form> = order
                    .iter()
                    .map(|&d| cm.args[k].view.forms[d].clone())
                    .chain(cm.args[k].view.forms[kp..].iter().cloned())
                    .collect();
                cm.args[k].view = View {
                    forms,
                    guard: cm.args[k].view.guard,
                    gather: None,
                };
            }
        }
        (g, outputs.to_vec())
    }

    /// A reduction over several dims with a small one among them
    /// (≤ `SMALL` elements) is two reductions: the inner one over the
    /// rest, keeping the small dim as a point dim, and an outer one
    /// folding the small dim with the same monoid (states combine like
    /// elements — the property the split schedule relies on). The inner
    /// reduction then has one reduce dim and tiles; the outer one is tiny.
    /// `Σ_{r,i} dS[i,g,r,j]·q[i,g,r,d]` (dK for grouped heads) becomes
    /// `Σ_r (Σ_i …)`. Sibling reductions exposed by then are merged
    /// ([`merge_siblings`](Self::merge_siblings)).
    pub fn split_reduce_dims(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        const SMALL: usize = 16;
        let mut g = self.clone();
        let live = self.live(outputs);
        let mut redirect: Vec<usize> = (0..self.nodes.len()).collect();
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            let Some(m) = c.monoid.clone() else { continue };
            if c.red.len() < 2 || c.tuple > 1 {
                continue;
            }
            let reds: Option<Vec<usize>> = c.red.iter().map(|d| d.fixed()).collect();
            let Some(reds) = reds else { continue };
            // the smallest reduce dim, if small
            let Some((f, &nf)) = reds.iter().enumerate().min_by_key(|(_, n)| **n) else {
                continue;
            };
            if nf > SMALL {
                continue;
            }
            let k = c.points().len();
            let rank = c.rank();
            let w = c.width();
            // inner space: the small dim first (so the original points keep
            // their lane and row axes), the points, the other reduce dims
            let order: Vec<usize> = [k + f]
                .into_iter()
                .chain(0..k)
                .chain((0..reds.len()).filter(|&r| r != f).map(|r| k + r))
                .collect();
            let inner_args: Vec<Arg> = c
                .args
                .iter()
                .map(|a| Arg {
                    src: a.src,
                    view: View {
                        forms: a
                            .view
                            .forms
                            .iter()
                            .map(|fm| Form {
                                coefs: order.iter().map(|&d| fm.coefs[d]).collect(),
                                offset: fm.offset,
                            })
                            .collect(),
                        guard: a.view.guard,
                        gather: a.view.gather.clone(),
                    },
                })
                .collect();
            let mut inner_out: Vec<Dim> = vec![Dim::Fixed(nf)];
            inner_out.extend_from_slice(c.points());
            inner_out.extend_from_slice(&c.out[k..]);
            let inner = Comp {
                out: inner_out.clone(),
                tuple: c.tuple,
                red: (0..reds.len())
                    .filter(|&r| r != f)
                    .map(|r| c.red[r])
                    .collect(),
                args: inner_args,
                body: c.body.clone(),
                body_outs: c.body_outs.clone(),
                monoid: Some(m.clone()),
            };
            let inner_id = g.nodes.len();
            g.nodes.push(TNode {
                shape: inner_out,
                inline: false,
                split: 0,
                tile: (0, 0, 0),
                reg: (1, 1),
                fragment: false,
                int_mma: None,
                segment: None,
                sorted: false,
                detach: false,
                subgroup: false,
                frag_cfg: [0; 4],
                precision: Precision::default(),
                op: Op::Comp(inner),
            });
            // outer: a map folding the nf partial states with the monoid's
            // combine, unrolled — a map, so it never absorbs the inner
            // reduction back into a naive loop
            let args: Vec<Arg> = (0..nf)
                .flat_map(|fi| {
                    (0..w).map(move |slot| {
                        let mut forms: Vec<Form> = vec![Form::constant(k, fi as i64)];
                        forms.extend((0..k).map(|d| Form::unit(k, d)));
                        if c.tuple == 1 {
                            forms.push(Form::constant(k, slot as i64));
                        }
                        Arg {
                            src: TensorId(inner_id as u32),
                            view: View {
                                forms,
                                guard: false,
                                gather: None,
                            },
                        }
                    })
                })
                .collect();
            let mut body = ExprGraph::new();
            let vars: Vec<ExprId> = (0..nf * w).map(|v| body.var(v as u16)).collect();
            let mut acc: Vec<ExprId> = vars[..w].to_vec();
            for fi in 1..nf {
                let mut ins = acc.clone();
                ins.extend_from_slice(&vars[fi * w..(fi + 1) * w]);
                acc = replay(&m.combine, &m.outs, &ins, &mut body);
            }
            let outer = Comp {
                out: c.out.clone(),
                tuple: c.tuple,
                red: vec![],
                args,
                body,
                body_outs: acc,
                monoid: None,
            };
            let _ = rank;
            g.nodes[i].op = Op::Comp(outer);
            redirect[i] = i;
        }
        let _ = redirect;
        // the layout passes and the split expose sibling reductions that
        // `merge_siblings` could not see before them (the two halves of
        // dQ, both reading dS): merge them now
        let (g, outputs) = g.eliminate_dead(outputs);
        let (g, outputs) = g.hoist_invariant_factors(&outputs);
        g.merge_siblings(&outputs)
    }

    /// Distributivity for sums: `Σ_k f(i)·g(i, k)` is `f(i) · Σ_k g(i, k)`,
    /// so factors of a sum's body that do not read the reduce coords move
    /// to a map over the reduction (`Σ_c P̃·dO·V` becomes `P̃ · Σ_c dO·V`,
    /// a plain bilinear contraction a tile can run as a GEMM). A sum over
    /// size-1 dims only is a map. Maps exposed this way are substituted
    /// into their readers again.
    pub fn hoist_invariant_factors(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        let mut g = self.clone();
        let live = self.live(outputs);
        let mut changed = false;
        // args no body output reads (the product rule's dropped factor)
        // are reads all the same for a kernel: drop them
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            if c.has_gather() {
                continue;
            }
            let mut used = vec![false; c.args.len()];
            for &o in &c.body_outs {
                for v in vars_reachable(&c.body, o) {
                    used[v as usize] = true;
                }
            }
            if used.iter().all(|&u| u) {
                continue;
            }
            let mut body = ExprGraph::new();
            let mut args = Vec::new();
            let vars: Vec<ExprId> = c
                .args
                .iter()
                .zip(&used)
                .map(|(a, &u)| {
                    if !u {
                        return ExprId::ZERO;
                    }
                    args.push(a.clone());
                    body.var((args.len() - 1) as u16)
                })
                .collect();
            let body_outs = replay(&c.body, &c.body_outs, &vars, &mut body);
            let body_outs = body_outs.into_iter().map(|o| body.simplify(o)).collect();
            let mut c = c.clone();
            c.args = args;
            c.body = body;
            c.body_outs = body_outs;
            g.nodes[i].op = Op::Comp(c);
            changed = true;
        }
        let pruned = g.clone();
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            let Some(c) = pruned.comp_at(i) else { continue };
            let sum = c.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Sum);
            if !sum || c.tuple != 0 || c.width() != 1 || c.red.is_empty() || c.has_gather() {
                continue;
            }
            let k = c.points().len();
            let rank = c.rank();
            // the args' views with the reduce coords dropped (valid where
            // they do not read them)
            let point_view = |v: &View| View {
                guard: v.guard,
                gather: v.gather.clone(),
                forms: v
                    .forms
                    .iter()
                    .map(|f| Form {
                        coefs: f.coefs[..k].to_vec(),
                        offset: f.offset,
                    })
                    .collect(),
            };
            if c.red.iter().all(|d| *d == Dim::Fixed(1)) {
                let mut m = c.clone();
                for a in &mut m.args {
                    a.view = point_view(&a.view);
                }
                m.red.clear();
                m.monoid = None;
                g.nodes[i].op = Op::Comp(m);
                changed = true;
                continue;
            }
            let nodes = c.body.nodes_slice();
            let mut factors = Vec::new();
            let mut stack = vec![c.body_outs[0]];
            while let Some(e) = stack.pop() {
                match nodes[e.index() as usize] {
                    Node::Mul(a, b) => stack.extend([a, b]),
                    _ => factors.push(e),
                }
            }
            let invariant = |e: ExprId| {
                vars_reachable(&c.body, e).iter().all(|&v| {
                    c.args[v as usize]
                        .view
                        .forms
                        .iter()
                        .all(|f| f.coefs[k..].iter().all(|&x| x == 0))
                })
            };
            let (inv, var): (Vec<ExprId>, Vec<ExprId>) =
                factors.iter().partition(|&&e| invariant(e));
            let real = inv
                .iter()
                .any(|&e| nodes[e.index() as usize].as_f64().is_none());
            if !real || var.is_empty() {
                continue;
            }
            let build = |fs: &[ExprId],
                         body: &mut ExprGraph,
                         args: &mut Vec<Arg>,
                         view: &dyn Fn(&View) -> View| {
                let vars: Vec<ExprId> = c
                    .args
                    .iter()
                    .enumerate()
                    .map(|(v, a)| {
                        let used = fs
                            .iter()
                            .any(|&e| vars_reachable(&c.body, e).contains(&(v as u16)));
                        if !used {
                            return ExprId::ZERO;
                        }
                        let leaf = Arg {
                            src: a.src,
                            view: view(&a.view),
                        };
                        body.var(intern_arg(args, leaf) as u16)
                    })
                    .collect();
                let vals = replay(&c.body, fs, &vars, body);
                vals.into_iter()
                    .reduce(|x, y| body.mul(x, y))
                    .expect("factors")
            };
            let mut rbody = ExprGraph::new();
            let mut rargs = Vec::new();
            let rout = build(&var, &mut rbody, &mut rargs, &|v: &View| v.clone());
            let inner = Comp {
                out: c.out.clone(),
                tuple: 0,
                red: c.red.clone(),
                args: rargs,
                body: rbody,
                body_outs: vec![rout],
                monoid: c.monoid.clone(),
            };
            let inner_id = g.nodes.len();
            let mut node = self.nodes[i].clone();
            node.op = Op::Comp(inner);
            g.nodes.push(node);
            let mut mbody = ExprGraph::new();
            let mut margs = vec![Arg {
                src: TensorId(inner_id as u32),
                view: View::identity(k, k),
            }];
            let r = mbody.var(0);
            let f = build(&inv, &mut mbody, &mut margs, &point_view);
            let out = mbody.mul(r, f);
            let _ = rank;
            g.nodes[i].op = Op::Comp(Comp {
                out: c.out.clone(),
                tuple: 0,
                red: vec![],
                args: margs,
                body: mbody,
                body_outs: vec![out],
                monoid: None,
            });
            changed = true;
        }
        if !changed {
            return (self.clone(), outputs.to_vec());
        }
        let (g, outs) = g.eliminate_dead(outputs);
        g.substitute_maps(&outs)
    }

    /// Two adjacent reduce dims that every arg reads either not at all
    /// or as a pair of adjacent source dims (pure picks, the inner one
    /// of the inner dim's size) are one dim: the arg reads a reshape of
    /// its source with those two dims merged — zero-cost, same bytes —
    /// and `Σ_{g,r,d} q[s,g,r,d]·w[m,g,r,d]` becomes `Σ_k q[s,k]·w[m,k]`,
    /// which is what the matmul schedules and the tensor-core lowerings
    /// look for. Repeated until nothing merges.
    pub fn merge_reduce_dims(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        let mut g = self.clone();
        let live = self.live(outputs);
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            while let Some(c) = g.comp_at(i) {
                let k = c.points().len();
                // the innermost mergeable pair, if any
                let found = (2..=c.red.len()).rev().find_map(|d| {
                    let (outer, inner) = (k + d - 2, k + d - 1);
                    let n_inner = c.red[d - 1].fixed()?;
                    let plan: Option<Vec<MergePlan>> = c
                        .args
                        .iter()
                        .map(|a| {
                            // a guarded read's bounds are per source dim
                            let reads = a.view.depends_on(outer) || a.view.depends_on(inner);
                            if (a.view.guard || a.view.is_gather()) && reads {
                                return None;
                            }
                            // a map over a contraction just as big (P̃ =
                            // exp(S − m)) is recomputed by its readers,
                            // the contraction nested in their tiles; a
                            // reshape would pin it to memory
                            if reads && g.maps_contraction(a.src.n()) {
                                return None;
                            }
                            g.merge_plan(a, outer, inner, n_inner)
                        })
                        .collect();
                    plan.map(|p| (d, p))
                });
                let Some((d, plan)) = found else { break };
                let (outer, inner) = (k + d - 2, k + d - 1);
                let c = c.clone();
                let mut args = Vec::with_capacity(c.args.len());
                for (a, p) in c.args.iter().zip(plan) {
                    let (src, forms) = match p {
                        None => (a.src, a.view.forms.clone()),
                        Some((d0, shape)) => {
                            let src = g.reshape(a.src, &shape).unwrap();
                            let mut forms = a.view.forms.clone();
                            // dims d0, d0+1 become one dim picking `outer`
                            forms[d0] = Form::unit(forms[d0].coefs.len(), outer);
                            forms.remove(d0 + 1);
                            (src, forms)
                        }
                    };
                    let forms = forms
                        .into_iter()
                        .map(|mut f| {
                            f.coefs.remove(inner);
                            f
                        })
                        .collect();
                    args.push(Arg {
                        src,
                        view: View {
                            forms,
                            guard: a.view.guard,
                            gather: a.view.gather.clone(),
                        },
                    });
                }
                let n_inner = c.red[d - 1].fixed().unwrap();
                let n_outer = c.red[d - 2].fixed().unwrap_or(1);
                let Op::Comp(cm) = &mut g.nodes[i].op else {
                    unreachable!()
                };
                cm.args = args;
                cm.red[d - 2] = Dim::Fixed(n_outer * n_inner);
                cm.red.remove(d - 1);
            }
        }
        (g, outputs.to_vec())
    }

    /// Whether node `p` is a map that reads a sum-reduction with at least
    /// as many points as itself (`exp(S − m)` over `S = Q·Kᵀ`).
    fn maps_contraction(&self, p: usize) -> bool {
        let Some(pc) = self.comp_at(p) else {
            return false;
        };
        pc.red.is_empty()
            && pc.args.iter().any(|a| {
                self.comp_at(a.src.n()).is_some_and(|r| {
                    r.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Sum)
                        && !r.red.is_empty()
                        && numel(r.points()) >= numel(pc.points())
                })
            })
    }

    /// How `a` would read with space dims `outer`/`inner` merged: `None`
    /// if it cannot; `Some(None)` if it reads neither; `Some((d0, shape))`
    /// if source dims `d0`, `d0+1` pick `outer`, `inner` and `shape` is
    /// the source's shape with the two merged.
    #[allow(clippy::type_complexity)]
    fn merge_plan(
        &self,
        a: &Arg,
        outer: usize,
        inner: usize,
        n_inner: usize,
    ) -> Option<Option<(usize, Vec<Dim>)>> {
        let forms = &a.view.forms;
        let touches: Vec<usize> = (0..forms.len())
            .filter(|&d| forms[d].coefs[outer] != 0 || forms[d].coefs[inner] != 0)
            .collect();
        if touches.is_empty() {
            return Some(None);
        }
        let [d0, d1] = touches[..] else { return None };
        if d1 != d0 + 1 {
            return None;
        }
        let rank = forms[d0].coefs.len();
        if forms[d0] != Form::unit(rank, outer) || forms[d1] != Form::unit(rank, inner) {
            return None;
        }
        let shape = &self.nodes[a.src.n()].shape;
        if shape[d1].fixed()? != n_inner {
            return None;
        }
        let mut merged = shape.clone();
        merged[d0] = Dim::Fixed(shape[d0].fixed()? * n_inner);
        merged.remove(d1);
        Some(Some((d0, merged)))
    }

    /// Whether node `a` reads `b`, directly or transitively.
    fn depends_on(&self, a: usize, b: usize) -> bool {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = self.deps(a);
        while let Some(i) = stack.pop() {
            if i == b {
                return true;
            }
            if !std::mem::replace(&mut seen[i], true) {
                stack.extend(self.deps(i));
            }
        }
        false
    }

    /// Reads of comp `i` that it recomputes per element: through its
    /// map args (not outputs, so `schedule_auto` may inline them), the
    /// `(source, view over i's space)` of every comp read that is larger
    /// than `i`'s points — an n×n score tile read by an n×d gradient.
    fn recomputed_reads(&self, i: usize, is_output: &dyn Fn(usize) -> bool) -> Vec<(usize, View)> {
        let Some(c) = self.comp_at(i) else {
            return Vec::new();
        };
        let rank = c.rank();
        let pts = numel(c.points());
        let mut out: Vec<(usize, View)> = Vec::new();
        // (source, view, reached through a recomputed map)
        let mut stack: Vec<(usize, View, bool)> = c
            .args
            .iter()
            .map(|a| (a.src.n(), a.view.clone(), false))
            .collect();
        while let Some((p, v, via)) = stack.pop() {
            let Some(pc) = self.comp_at(p) else { continue };
            if numel(&self.nodes[p].shape) <= pts {
                continue;
            }
            let map = pc.monoid.is_none() && pc.red.is_empty() && pc.tuple == 0 && !is_output(p);
            if map {
                for a in &pc.args {
                    stack.push((a.src.n(), a.view.compose(&v, rank), true));
                }
            }
            if (map || via) && !out.contains(&(p, v.clone())) {
                out.push((p, v));
            }
        }
        out
    }

    /// Sibling reductions — the same points, reduce dims and monoid,
    /// each one value per point, read by exactly the same consumers —
    /// become one comp producing all of them (the product monoid, one
    /// tuple dim). A map over the siblings then reads a single reduction
    /// and is its epilogue: the gate and up projections of a SwiGLU MLP
    /// and their activation are one kernel, with one pass over the
    /// shared operand.
    pub fn merge_siblings(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        const MAX_GROUP: usize = 4;
        let n = self.nodes.len();
        let live = self.live(outputs);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in (0..n).filter(|&i| live[i]) {
            for d in self.deps(i) {
                if consumers[d].last() != Some(&i) {
                    consumers[d].push(i);
                }
            }
        }
        // a big map that only reductions read (and that is bigger than
        // any of them): `schedule_auto` recomputes it inside each reader,
        // so siblings reading it share that recomputation once merged
        let recomputed = |p: usize| {
            self.comp_at(p).is_some_and(|pc| {
                pc.monoid.is_none()
                    && pc.red.is_empty()
                    && !is_output(p)
                    && consumers[p].iter().all(|&r| {
                        self.comp_at(r).is_some_and(|rc| {
                            rc.monoid.is_some() && numel(rc.points()) < numel(pc.points())
                        })
                    })
            })
        };
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        // members that fit neither rule above, by shape key: they may
        // still share a recomputed read (see `recomputed_reads`)
        let mut loose: Vec<(String, usize)> = Vec::new();
        for i in (0..n).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            let Some(m) = &c.monoid else { continue };
            if c.tuple != 0 || c.width() != 1 || c.has_gather() {
                continue;
            }
            let readers = &consumers[i];
            let base = format!(
                "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
                self.nodes[i].precision,
                c.out,
                c.red,
                m.identity.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                m.combine.nodes_slice(),
                m.outs,
            );
            // the same readers, or a shared recomputed operand (then a
            // reader that is not a comp, a reshape, reads the member's
            // slot of the merged reduction through a copy)
            let shared = c.args.iter().map(|a| a.src.n()).find(|&p| recomputed(p));
            if is_output(i)
                || readers.is_empty()
                || (shared.is_none() && readers.iter().any(|&r| self.comp_at(r).is_none()))
            {
                loose.push((base, i));
                continue;
            }
            let key = format!(
                "{base}|{}",
                match shared {
                    Some(p) => format!("via {p}"),
                    None => format!("{readers:?}"),
                }
            );
            let g = groups.entry(key.clone()).or_insert_with(|| {
                order.push(key);
                Vec::new()
            });
            g.push(i);
        }
        // Groups of one may still merge with a loose member, and loose
        // members with each other, when they recompute a common read: the
        // flash backward's dV = Σᵢ P̃ᵀ·dO and dK = Σᵢ dSᵀ·Q both evaluate
        // S = Q·Kᵀ (and exp(S − m)) at the same (j, i). Merged, one tile
        // recomputes that tile once for both.
        for key in &order {
            if let [i] = groups[key][..] {
                let c = self.comp_at(i).unwrap();
                let m = c.monoid.as_ref().unwrap();
                let base = format!(
                    "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
                    self.nodes[i].precision,
                    c.out,
                    c.red,
                    m.identity.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    m.combine.nodes_slice(),
                    m.outs,
                );
                loose.push((base, i));
            }
        }
        loose.sort_by_key(|&(_, i)| i);
        let reads: HashMap<usize, Vec<(usize, View)>> = loose
            .iter()
            .map(|&(_, i)| (i, self.recomputed_reads(i, &is_output)))
            .collect();
        let mut taken: Vec<bool> = vec![false; loose.len()];
        for a in 0..loose.len() {
            if taken[a] || reads[&loose[a].1].is_empty() {
                continue;
            }
            let mut members = vec![loose[a].1];
            for b in a + 1..loose.len() {
                if taken[b] || loose[b].0 != loose[a].0 || members.len() == MAX_GROUP {
                    continue;
                }
                let rb = &reads[&loose[b].1];
                let independent = members
                    .iter()
                    .all(|&m| !self.depends_on(m, loose[b].1) && !self.depends_on(loose[b].1, m));
                if independent && reads[&loose[a].1].iter().any(|r| rb.contains(r)) {
                    taken[b] = true;
                    members.push(loose[b].1);
                }
            }
            if members.len() < 2 {
                continue;
            }
            taken[a] = true;
            // drop the singleton groups these came from
            for key in &order {
                let g = groups.get_mut(key).unwrap();
                if g.len() == 1 && members.contains(&g[0]) {
                    g.clear();
                }
            }
            let key = format!("shared {members:?}");
            order.push(key.clone());
            groups.insert(key, members);
        }
        let mut g = self.clone();
        // member -> (merged node, slot)
        let mut slot_of: Vec<Option<(usize, usize)>> = vec![None; n];
        for key in &order {
            for members in groups[key].chunks(MAX_GROUP).filter(|m| m.len() > 1) {
                let first = self.comp_at(members[0]).unwrap();
                let k = members.len();
                let mut args: Vec<Arg> = Vec::new();
                let mut body = ExprGraph::new();
                let mut body_outs = Vec::with_capacity(k);
                let mut combine = ExprGraph::new();
                let acc: Vec<ExprId> = (0..k).map(|j| combine.var(j as u16)).collect();
                let el: Vec<ExprId> = (0..k).map(|j| combine.var((k + j) as u16)).collect();
                let mut outs = Vec::with_capacity(k);
                let mut identity = Vec::with_capacity(k);
                for (j, &i) in members.iter().enumerate() {
                    let c = self.comp_at(i).unwrap();
                    let vars: Vec<ExprId> = c
                        .args
                        .iter()
                        .map(|a| body.var(intern_arg(&mut args, a.clone()) as u16))
                        .collect();
                    body_outs.push(replay(&c.body, &c.body_outs, &vars, &mut body)[0]);
                    let m = c.monoid.as_ref().unwrap();
                    outs.push(replay(&m.combine, &m.outs, &[acc[j], el[j]], &mut combine)[0]);
                    identity.push(m.identity[0]);
                }
                let mut out = first.out.clone();
                out.push(Dim::Fixed(k));
                let merged = g.nodes.len();
                g.nodes.push(TNode {
                    shape: out.clone(),
                    // siblings group by precision: all members share it
                    precision: self.nodes[members[0]].precision,
                    fragment: false,
                    int_mma: None,
                    segment: None,
                    sorted: false,
                    detach: false,
                    subgroup: false,
                    frag_cfg: [0; 4],
                    inline: false,
                    split: 0,
                    tile: (0, 0, 0),
                    reg: (1, 1),
                    op: Op::Comp(Comp {
                        out,
                        tuple: 1,
                        red: first.red.clone(),
                        args,
                        body,
                        body_outs,
                        monoid: Some(Monoid {
                            width: k,
                            identity,
                            combine,
                            outs,
                            saved: None,
                        }),
                    }),
                });
                for (j, &i) in members.iter().enumerate() {
                    slot_of[i] = Some((merged, j));
                    if !is_output(i) && consumers[i].iter().all(|&r| self.comp_at(r).is_some()) {
                        continue;
                    }
                    let c = self.comp_at(i).unwrap();
                    let rank = c.out.len();
                    let mut view = View::identity(rank, rank);
                    view.forms.push(Form::constant(rank, j as i64));
                    let mut body = ExprGraph::new();
                    let v = body.var(0);
                    g.nodes[i].op = Op::Comp(Comp {
                        out: c.out.clone(),
                        tuple: 0,
                        red: Vec::new(),
                        args: vec![Arg {
                            src: TensorId(merged as u32),
                            view,
                        }],
                        body,
                        body_outs: vec![v],
                        monoid: None,
                    });
                }
            }
        }
        for i in 0..n {
            if let Op::Comp(c) = &mut g.nodes[i].op {
                let rank = c.rank();
                for a in &mut c.args {
                    if slot_of[i].is_some_and(|(m, _)| m == a.src.n()) {
                        continue;
                    }
                    if let Some((merged, slot)) = slot_of[a.src.n()] {
                        a.src = TensorId(merged as u32);
                        a.view.forms.push(Form::constant(rank, slot as i64));
                    }
                }
            }
        }
        g.eliminate_dead(outputs)
    }

    /// Schedule (split-K): cut the (outermost) reduce axis of comp `t`
    /// (length `K`) into `parts` contiguous chunks computed in parallel, then fold
    /// the chunks' partial states with the monoid's own combine. `t` is
    /// rewritten in place to the combine, so every reader is unchanged:
    ///
    /// - partial `[parts, points…, tuple…]`: the original body and monoid
    ///   over `K / parts` keys, the chunk index a new (leading) point axis,
    ///   so one launch has `parts`× the points — the parallelism a long
    ///   reduction over few points (batch-1 decode attention, a long-K
    ///   GEMV/GEMM) lacks;
    /// - combine: a reduction over the chunk axis whose body reads one
    ///   partial state and whose monoid is `t`'s. That is legal for any
    ///   monoid because partial states and elements are the same width
    ///   and the combine is associative: for a sum it adds, for online
    ///   softmax it is the rescale-and-add of `(m, l, acc)`.
    ///
    /// Nothing here knows about attention. Returns the rewritten graph
    /// (topologically ordered) and `outputs` in it.
    pub fn split_reduction(
        &self,
        outputs: &[TensorId],
        t: TensorId,
        parts: usize,
    ) -> Result<(TensorGraph, Vec<TensorId>), ShapeError> {
        let mut g = self.clone();
        g.split_reduction_in_place(t, parts)?;
        Ok(g.eliminate_dead(outputs))
    }

    /// [`TensorGraph::split_reduction`] without the reordering: the
    /// partial is pushed after `t`, which now reads it.
    fn split_reduction_in_place(
        &mut self,
        t: TensorId,
        parts: usize,
    ) -> Result<TensorId, ShapeError> {
        let Some(c) = self.comp_at(t.n()).cloned() else {
            return err("split_reduction: not a comp");
        };
        let Some(m) = c.monoid.clone() else {
            return err("split_reduction: not a reduction");
        };
        // several reduce axes (a grouped 4-bit GEMV's groups × 64): the
        // outermost one is cut, the inner ones stay in the partial
        if c.red.is_empty() || c.tuple > 1 {
            return err("split_reduction: a reduce axis, at most one tuple dim");
        }
        let total = c.red[0].n();
        if parts < 2 || total % parts != 0 {
            return err(format!(
                "split_reduction: {total} keys do not split into {parts}"
            ));
        }
        let per = total / parts;
        let k = c.points().len();
        let w = c.width();
        // partial space (p, points…, r): the original reduce coordinate is
        // p·per + r
        let args = c
            .args
            .iter()
            .map(|a| Arg {
                src: a.src,
                view: View {
                    forms: a
                        .view
                        .forms
                        .iter()
                        .map(|f| Form {
                            coefs: std::iter::once(f.coefs[k] * per as i64)
                                .chain(f.coefs.iter().copied())
                                .collect(),
                            offset: f.offset,
                        })
                        .collect(),
                    guard: a.view.guard,
                    gather: a.view.gather.clone(),
                },
            })
            .collect();
        let mut out: Vec<Dim> = vec![Dim::Fixed(parts)];
        out.extend_from_slice(&c.out);
        let mut red = vec![Dim::Fixed(per)];
        red.extend_from_slice(&c.red[1..]);
        let partial = self.push_comp(Comp {
            out,
            tuple: c.tuple,
            red,
            args,
            body: c.body.clone(),
            body_outs: c.body_outs.clone(),
            monoid: Some(m.clone()),
        })?;
        self.nodes[partial.n()].precision = self.nodes[t.n()].precision;
        // combine space (points…, p): read slot s of partial state p
        let args = (0..w)
            .map(|s| {
                let mut forms = vec![Form::unit(k + 1, k)];
                forms.extend((0..k).map(|d| Form::unit(k + 1, d)));
                if c.tuple == 1 {
                    forms.push(Form::constant(k + 1, s as i64));
                }
                Arg {
                    src: partial,
                    view: View {
                        forms,
                        guard: false,
                        gather: None,
                    },
                }
            })
            .collect();
        let mut body = ExprGraph::new();
        let body_outs: Vec<ExprId> = (0..w).map(|v| body.var(v as u16)).collect();
        let node = &mut self.nodes[t.n()];
        node.op = Op::Comp(Comp {
            out: c.out.clone(),
            tuple: c.tuple,
            red: vec![Dim::Fixed(parts)],
            args,
            body,
            body_outs,
            monoid: Some(m),
        });
        node.inline = false;
        node.split = 0;
        node.tile = (0, 0, 0);
        node.reg = (1, 1);
        node.fragment = false;
        Ok(partial)
    }

    /// Split-K factors worth trying for comp `t`: powers of two that
    /// divide its reduce axis, leave at least `min_chunk` keys per chunk,
    /// and do not take the grid past `max_points` (points × parts). `[]`
    /// when `t` is not a single-axis reduction. An autotuner times each
    /// (and the unsplit kernel) per sequence length.
    pub fn split_k_candidates(
        &self,
        t: TensorId,
        min_chunk: usize,
        max_points: usize,
    ) -> Vec<usize> {
        let Some(c) = self.comp_at(t.n()) else {
            return vec![];
        };
        if c.monoid.is_none() || c.red.len() != 1 || c.tuple > 1 {
            return vec![];
        }
        let (Some(total), Some(pts)) = (
            c.red[0].fixed(),
            c.points()
                .iter()
                .map(|d| d.fixed())
                .product::<Option<usize>>(),
        ) else {
            return vec![];
        };
        (1..)
            .map(|e| 1usize << e)
            .take_while(|&p| total / p >= min_chunk && pts * p <= max_points)
            .filter(|&p| total % p == 0)
            .collect()
    }

    /// Split-K by rule, for when nothing is tuned: every live single-axis
    /// reduction of fewer than `target_points` points over a reduce axis of
    /// at least 1024 is split into the smallest power of
    /// two that brings points × parts to `target_points`, keeping at least
    /// 64 keys per chunk. Decode attention and long-K GEMVs qualify; a
    /// prefill or a square matmul has the points already. The autotuned
    /// path times [`split_k_candidates`](Self::split_k_candidates) instead.
    pub fn schedule_split_k(
        &self,
        outputs: &[TensorId],
        target_points: usize,
    ) -> (TensorGraph, Vec<TensorId>) {
        let mut g = self.clone();
        let live = self.live(outputs);
        let mut any = false;
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            let (Some(red), Some(pts)) = (
                c.red.first().and_then(|d| d.fixed()),
                c.points()
                    .iter()
                    .map(|d| d.fixed())
                    .product::<Option<usize>>(),
            ) else {
                continue;
            };
            if c.red.len() != 1 || pts >= target_points || red < 1024 {
                continue;
            }
            let pick = self
                .split_k_candidates(TensorId(i as u32), 64, usize::MAX)
                .into_iter()
                .find(|&p| pts * p >= target_points)
                .or_else(|| {
                    self.split_k_candidates(TensorId(i as u32), 64, usize::MAX)
                        .pop()
                });
            if let Some(p) = pick {
                any |= g.split_reduction_in_place(TensorId(i as u32), p).is_ok();
            }
        }
        if !any {
            return (g, outputs.to_vec());
        }
        g.eliminate_dead(outputs)
    }

    /// Placement: split the (single) reduce axis of comp `t` into `parts`
    /// partial reductions over contiguous slices — each a comp that can
    /// live on its own device — plus a combine that folds the partials
    /// with the monoid. Returns `(graph, partials, combined)`; the
    /// combined node is interchangeable with `t`. Tiling a reduce axis
    /// across threads and sharding it across devices are the same
    /// transform; this is the latter with the all-reduce written as a
    /// comprehension over the partial results.
    pub fn shard_reduction(
        &self,
        t: TensorId,
        parts: usize,
    ) -> Result<(TensorGraph, Vec<TensorId>, TensorId), ShapeError> {
        let Some(c) = self.comp_at(t.n()) else {
            return err("shard_reduction: not a comp");
        };
        let Some(m) = &c.monoid else {
            return err("shard_reduction: not a reduction");
        };
        if c.red.len() != 1 {
            return err("shard_reduction: one reduce axis only");
        }
        let total = c.red[0].n();
        if parts == 0 || total % parts != 0 {
            return err(format!(
                "shard_reduction: {total} keys do not split into {parts}"
            ));
        }
        let per = total / parts;
        let red_coord = c.points().len();
        let mut g = self.clone();
        let partials: Vec<TensorId> = (0..parts)
            .map(|p| {
                let shift = (p * per) as i64;
                let args = c
                    .args
                    .iter()
                    .map(|a| Arg {
                        src: a.src,
                        view: View {
                            guard: false,
                            gather: a.view.gather.clone(),
                            forms: a
                                .view
                                .forms
                                .iter()
                                .map(|f| Form {
                                    coefs: f.coefs.clone(),
                                    offset: f.offset + f.coefs[red_coord] * shift,
                                })
                                .collect(),
                        },
                    })
                    .collect();
                let mut out = c.out.clone();
                if c.tuple == 0 {
                    out.push(Dim::Fixed(1));
                }
                g.push_comp(Comp {
                    out,
                    tuple: 1,
                    red: vec![Dim::Fixed(per)],
                    args,
                    body: c.body.clone(),
                    body_outs: c.body_outs.clone(),
                    monoid: Some(m.clone()),
                })
            })
            .collect::<Result<_, _>>()?;
        let acc = g.fold_partials(&partials, m)?;
        // present the result in `t`'s own shape
        let combined = if c.tuple == 0 {
            g.reshape(acc, &c.out.clone())?
        } else {
            acc
        };
        Ok((g, partials, combined))
    }

    /// The all-reduce as a comprehension: fold tuple-valued partial
    /// results `[points…, width]` with the monoid's combine, slot by slot.
    /// The partials may be inputs (results shipped from other devices).
    pub fn fold_partials(
        &mut self,
        partials: &[TensorId],
        m: &Monoid,
    ) -> Result<TensorId, ShapeError> {
        let Some(&first) = partials.first() else {
            return err("fold_partials: no partials");
        };
        let shape = self.shape(first).to_vec();
        let w = m.width;
        if shape.last() != Some(&Dim::Fixed(w)) {
            return err(format!(
                "fold_partials: shape {shape:?} does not end in width {w}"
            ));
        }
        let rank = shape.len() - 1;
        let slot_view = |s: usize| View {
            guard: false,
            gather: None,
            forms: (0..rank)
                .map(|d| Form::unit(rank, d))
                .chain([Form::constant(rank, s as i64)])
                .collect(),
        };
        let mut acc = first;
        for &p in &partials[1..] {
            let mut args = Vec::new();
            for s in 0..w {
                args.push(Arg {
                    src: acc,
                    view: slot_view(s),
                });
            }
            for s in 0..w {
                args.push(Arg {
                    src: p,
                    view: slot_view(s),
                });
            }
            let mut body = ExprGraph::new();
            let vars: Vec<ExprId> = (0..2 * w).map(|i| body.var(i as u16)).collect();
            let outs = replay(&m.combine, &m.outs, &vars, &mut body);
            acc = self.push_comp(Comp {
                out: shape.clone(),
                tuple: 1,
                red: vec![],
                args,
                body,
                body_outs: outs,
                monoid: None,
            })?;
        }
        Ok(acc)
    }

    // -- reference interpreter -------------------------------------------------

    /// Evaluate `outputs`. Materialized nodes are computed once into
    /// buffers; inlined nodes are recomputed at every point they are read.
    pub fn eval(&self, inputs: &[Vec<f64>], outputs: &[TensorId]) -> Vec<Vec<f64>> {
        self.eval_with(inputs, outputs, false)
    }

    /// [`TensorGraph::eval`] under the graph's precision schedule: every
    /// stored value (inputs, materialized buffers) is rounded to its
    /// node's storage format, and a reduction's running state is rounded
    /// to its accum format after every combine step (round to nearest
    /// even, subnormals, overflow to ∞). Body arithmetic stays f64, so
    /// this predicts what a schedule costs in accuracy without a GPU.
    pub fn eval_emulated(&self, inputs: &[Vec<f64>], outputs: &[TensorId]) -> Vec<Vec<f64>> {
        self.eval_with(inputs, outputs, true)
    }

    fn eval_with(&self, inputs: &[Vec<f64>], outputs: &[TensorId], emulate: bool) -> Vec<Vec<f64>> {
        let live = self.live(outputs);
        // block-scaled inputs: the scales depend on the whole buffer
        let scaled: Vec<Vec<f64>>;
        let inputs = if emulate
            && self
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::Input(_)) && n.precision.storage.is_fp8())
        {
            let mut v = inputs.to_vec();
            for n in &self.nodes {
                if let Op::Input(k) = n.op {
                    if n.precision.storage.is_fp8() {
                        n.precision.storage.round_buf(&mut v[k]);
                    }
                }
            }
            scaled = v;
            &scaled[..]
        } else {
            inputs
        };
        let mut ev = Evaluator {
            g: self,
            emulate,
            inputs,
            bufs: vec![Vec::new(); self.nodes.len()],
            compiled: self
                .nodes
                .iter()
                .map(|n| match &n.op {
                    Op::Comp(c) => Some(Compiled {
                        body: Prog::new(&c.body, &c.body_outs),
                        combine: c.monoid.as_ref().map(|m| Prog::new(&m.combine, &m.outs)),
                        direct: c
                            .args
                            .iter()
                            .map(|a| direct_read(self, a, c.rank(), emulate))
                            .collect(),
                    }),
                    Op::Input(_) | Op::Iota(_) | Op::Reshape(_) | Op::Operator(_) | Op::Scan(_) => {
                        None
                    }
                })
                .collect(),
        };
        for i in (0..self.nodes.len()).filter(|&i| live[i] && !self.nodes[i].inline) {
            if let Op::Operator(o) = &self.nodes[i].op {
                let ins: Vec<std::borrow::Cow<[f64]>> =
                    o.inputs.iter().map(|t| ev.tensor_ref(*t)).collect();
                let refs: Vec<&[f64]> = ins.iter().map(|v| &v[..]).collect();
                let out = o.op.forward(&refs);
                assert_eq!(
                    out.len(),
                    numel(&self.nodes[i].shape),
                    "operator `{}` returned the wrong number of values",
                    o.op.name()
                );
                ev.bufs[i] = out;
                continue;
            }
            if let Op::Scan(sc) = &self.nodes[i].op {
                let ins: Vec<std::borrow::Cow<[f64]>> =
                    sc.inputs.iter().map(|t| ev.tensor_ref(*t)).collect();
                let refs: Vec<&[f64]> = ins.iter().map(|v| &v[..]).collect();
                let shapes: Vec<Vec<usize>> =
                    sc.inputs.iter().map(|t| fixed(self.shape(*t))).collect();
                ev.bufs[i] = sc.eval(&refs, &shapes, &fixed(&self.nodes[i].shape));
                continue;
            }
            if let Some(c) = self.comp_at(i) {
                let w = c.width();
                let pts = fixed(c.points());
                let n = numel(c.points());
                let mut buf = vec![0.0; n * w];
                // points are independent: evaluate them on all cores (the
                // result does not depend on the split)
                // a thread per ~32k scalar ops of work: spawning costs tens
                // of microseconds, so small or cheap comps use fewer threads
                let per_elem = ev.compiled[i]
                    .as_ref()
                    .map_or(1, |k| k.body.ops.len() + c.args.len());
                let work = n
                    .saturating_mul(numel(&c.red).max(1))
                    .saturating_mul(per_elem);
                let threads = std::thread::available_parallelism()
                    .map_or(1, |t| t.get())
                    .min(work.div_ceil(1 << 15))
                    .min(n)
                    .max(1);
                let per = n.div_ceil(threads);
                let ev_ref = &ev;
                if threads == 1 {
                    // one chunk: run it here. This is also the only path on targets that
                    // cannot spawn threads (wasm32), where available_parallelism errs to 1.
                    let mut sc = Scratch::default();
                    let mut idx = vec![0; pts.len()];
                    for p in 0..n {
                        if p > 0 {
                            odometer(&mut idx, &pts);
                        }
                        ev_ref.point(i, &idx, &mut buf[p * w..(p + 1) * w], &mut sc);
                    }
                } else {
                    std::thread::scope(|sc| {
                        for (t, chunk) in buf.chunks_mut(per * w).enumerate() {
                            let pts = &pts;
                            sc.spawn(move || {
                                let start = t * per;
                                let mut scr = Scratch::default();
                                let mut idx = vec![0; pts.len()];
                                let mut rest = start;
                                for d in (0..pts.len()).rev() {
                                    idx[d] = rest % pts[d];
                                    rest /= pts[d];
                                }
                                for p in 0..chunk.len() / w {
                                    if p > 0 {
                                        odometer(&mut idx, pts);
                                    }
                                    ev_ref.point(i, &idx, &mut chunk[p * w..(p + 1) * w], &mut scr);
                                }
                            });
                        }
                    });
                }
                if emulate {
                    self.nodes[i].precision.storage.round_buf(&mut buf);
                    self.nodes[i].precision.storage.round_blocks(&mut buf);
                }
                ev.bufs[i] = buf;
            }
        }
        outputs.iter().map(|o| ev.tensor(*o)).collect()
    }
}

/// `out` as `Σ_t coef_t · var a_t · var b_t`, or `None` when it is not a
/// sum of two-var products (literal factors and negation allowed).
pub(crate) fn bilinear_terms(g: &ExprGraph, out: ExprId) -> Option<Vec<(f64, u16, u16)>> {
    let nodes = g.nodes_slice();
    fn term(nodes: &[Node], e: ExprId, coef: f64, vars: &mut Vec<u16>) -> Option<f64> {
        match nodes[e.index() as usize] {
            Node::Var(v) => {
                vars.push(v);
                Some(coef)
            }
            Node::Lit(_) => Some(coef * nodes[e.index() as usize].as_f64()?),
            Node::Neg(a) => term(nodes, a, -coef, vars),
            Node::Mul(a, b) => {
                let c = term(nodes, a, coef, vars)?;
                term(nodes, b, c, vars)
            }
            _ => None,
        }
    }
    fn sum(nodes: &[Node], e: ExprId, sign: f64, out: &mut Vec<(f64, u16, u16)>) -> Option<()> {
        match nodes[e.index() as usize] {
            Node::Add(a, b) => {
                sum(nodes, a, sign, out)?;
                sum(nodes, b, sign, out)
            }
            Node::Neg(a) => sum(nodes, a, -sign, out),
            _ => {
                let mut vars = Vec::new();
                let c = term(nodes, e, sign, &mut vars)?;
                if c == 0.0 {
                    return Some(());
                }
                let [a, b] = vars[..] else { return None };
                out.push((c, a, b));
                Some(())
            }
        }
    }
    let mut terms = Vec::new();
    sum(nodes, out, 1.0, &mut terms)?;
    (!terms.is_empty()).then_some(terms)
}

fn intern_arg(args: &mut Vec<Arg>, a: Arg) -> usize {
    args.iter().position(|x| *x == a).unwrap_or_else(|| {
        args.push(a);
        args.len() - 1
    })
}

/// A scalar body as a flat program over dense slots, evaluated into a
/// caller-owned scratch buffer (no allocation per call). Same arithmetic,
/// node for node, as `tang_expr`'s `compile_many`.
struct Prog {
    ops: Vec<POp>,
    outs: Vec<u32>,
}

#[derive(Clone, Copy)]
enum POp {
    Var(u16),
    Lit(f64),
    Add(u32, u32),
    Mul(u32, u32),
    Neg(u32),
    Recip(u32),
    Sqrt(u32),
    Sin(u32),
    Atan2(u32, u32),
    Exp2(u32),
    Log2(u32),
    Select(u32, u32, u32),
}

impl Prog {
    fn new(g: &ExprGraph, outs: &[ExprId]) -> Self {
        let nodes = g.nodes_slice();
        let n = outs
            .iter()
            .map(|o| o.index() as usize + 1)
            .max()
            .unwrap_or(0);
        let mut live = vec![false; n];
        let mut stack: Vec<usize> = outs.iter().map(|o| o.index() as usize).collect();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut live[i], true) {
                continue;
            }
            let kids: &[ExprId] = match &nodes[i] {
                Node::Var(_) | Node::Lit(_) => &[],
                Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => &[*a, *b],
                Node::Neg(a)
                | Node::Recip(a)
                | Node::Sqrt(a)
                | Node::Sin(a)
                | Node::Exp2(a)
                | Node::Log2(a) => std::slice::from_ref(a),
                Node::Select(c, a, b) => &[*c, *a, *b],
            };
            stack.extend(kids.iter().map(|k| k.index() as usize));
        }
        let mut slot = vec![0u32; n];
        let mut ops = Vec::new();
        for i in (0..n).filter(|&i| live[i]) {
            let m = |e: &ExprId| slot[e.index() as usize];
            let op = match &nodes[i] {
                Node::Var(v) => POp::Var(*v),
                Node::Lit(bits) => POp::Lit(f64::from_bits(*bits)),
                Node::Add(a, b) => POp::Add(m(a), m(b)),
                Node::Mul(a, b) => POp::Mul(m(a), m(b)),
                Node::Neg(a) => POp::Neg(m(a)),
                Node::Recip(a) => POp::Recip(m(a)),
                Node::Sqrt(a) => POp::Sqrt(m(a)),
                Node::Sin(a) => POp::Sin(m(a)),
                Node::Atan2(y, x) => POp::Atan2(m(y), m(x)),
                Node::Exp2(a) => POp::Exp2(m(a)),
                Node::Log2(a) => POp::Log2(m(a)),
                Node::Select(c, a, b) => POp::Select(m(c), m(a), m(b)),
            };
            slot[i] = ops.len() as u32;
            ops.push(op);
        }
        Self {
            outs: outs.iter().map(|o| slot[o.index() as usize]).collect(),
            ops,
        }
    }

    #[inline]
    fn eval(&self, inputs: &[f64], outputs: &mut [f64], v: &mut Vec<f64>) {
        v.clear();
        for op in &self.ops {
            let x = match *op {
                POp::Var(i) => inputs[i as usize],
                POp::Lit(x) => x,
                POp::Add(a, b) => v[a as usize] + v[b as usize],
                POp::Mul(a, b) => v[a as usize] * v[b as usize],
                POp::Neg(a) => -v[a as usize],
                POp::Recip(a) => 1.0 / v[a as usize],
                POp::Sqrt(a) => v[a as usize].sqrt(),
                POp::Sin(a) => v[a as usize].sin(),
                POp::Atan2(y, x) => v[y as usize].atan2(v[x as usize]),
                POp::Exp2(a) => v[a as usize].exp2(),
                POp::Log2(a) => v[a as usize].log2(),
                POp::Select(c, a, b) => {
                    if v[c as usize] > 0.0 {
                        v[a as usize]
                    } else {
                        v[b as usize]
                    }
                }
            };
            v.push(x);
        }
        for (o, &s) in outputs.iter_mut().zip(&self.outs) {
            *o = v[s as usize];
        }
    }
}

/// Per-thread buffers for [`Evaluator::point`].
#[derive(Default)]
struct Scratch {
    arg_vals: Vec<f64>,
    idx: Vec<usize>,
    elem: Vec<f64>,
    pair: Vec<f64>,
    prog: Vec<f64>,
    cprog: Vec<f64>,
}

struct Compiled {
    body: Prog,
    combine: Option<Prog>,
    /// Per arg: a precomputed flat-address read, when the arg reads a
    /// plain buffer (see [`direct_read`]).
    direct: Vec<Option<Direct>>,
}

/// Where a [`Direct`] read finds its buffer.
#[derive(Clone, Copy)]
enum DirectSrc {
    Input(usize),
    Buf(usize),
}

/// An arg read as `buf[base + Σ_k lin[k]·idx[k] + gather·round(index)]`:
/// the composition of its affine view with the source's row-major strides.
struct Direct {
    src: DirectSrc,
    base: i64,
    lin: Vec<i64>,
    gather: i64,
    /// `Some` for a guarded view: reads outside any source dim are 0.
    guard: Option<Vec<GuardDim>>,
}

/// The [`Direct`] plan for `a`, or `None` when the read needs the general
/// path: a guarded view, a source that is not a plain one-value-per-point
/// buffer (an iota, a reshape, an inlined or tuple comp), or emulation
/// (which rounds input reads).
fn direct_read(g: &TensorGraph, a: &Arg, rank: usize, emulate: bool) -> Option<Direct> {
    if emulate {
        return None;
    }
    // a reshape reads its storage's bytes row-major under its own shape
    let node = &g.nodes[a.src.n()];
    let store = g.storage(a.src);
    let snode = &g.nodes[store.n()];
    let src = match &snode.op {
        Op::Input(k) => DirectSrc::Input(*k),
        Op::Operator(_) | Op::Scan(_) => DirectSrc::Buf(store.n()),
        Op::Comp(c) if !snode.inline && c.tuple == 0 && c.width() == 1 => DirectSrc::Buf(store.n()),
        _ => return None,
    };
    let shape = node
        .shape
        .iter()
        .map(|d| d.fixed())
        .collect::<Option<Vec<_>>>()?;
    let mut strides = vec![0i64; shape.len()];
    let mut st = 1i64;
    for d in (0..shape.len()).rev() {
        strides[d] = st;
        st *= shape[d] as i64;
    }
    let mut base = 0;
    let mut lin = vec![0i64; rank];
    for (d, f) in a.view.forms.iter().enumerate() {
        base += strides[d] * f.offset;
        for (k, &c) in f.coefs.iter().enumerate() {
            lin[k] += strides[d] * c;
        }
    }
    let gather = a.view.gather.as_ref().map_or(0, |ga| {
        ga.coefs.iter().zip(&strides).map(|(c, s)| c * s).sum()
    });
    // a guarded view checks each source coordinate against its extent
    let guard = a.view.guard.then(|| {
        a.view
            .forms
            .iter()
            .enumerate()
            .map(|(d, f)| GuardDim {
                offset: f.offset,
                coefs: f.coefs.clone(),
                gather: a.view.gather.as_ref().map_or(0, |ga| ga.coefs[d]),
                extent: shape[d] as i64,
            })
            .collect()
    });
    Some(Direct {
        src,
        base,
        lin,
        gather,
        guard,
    })
}

/// One source dim of a guarded [`Direct`] read.
struct GuardDim {
    offset: i64,
    coefs: Vec<i64>,
    gather: i64,
    extent: i64,
}

struct Evaluator<'a> {
    g: &'a TensorGraph,
    /// Round stored values and accumulators per the precision schedule.
    emulate: bool,
    inputs: &'a [Vec<f64>],
    bufs: Vec<Vec<f64>>,
    compiled: Vec<Option<Compiled>>,
}

impl Evaluator<'_> {
    /// All of `t`, row-major: a copy of its buffer when it has a plain
    /// one, else read point by point.
    fn tensor(&self, t: TensorId) -> Vec<f64> {
        self.tensor_ref(t).into_owned()
    }

    /// [`tensor`](Self::tensor), borrowing the buffer when there is one.
    fn tensor_ref(&self, t: TensorId) -> std::borrow::Cow<'_, [f64]> {
        let node = &self.g.nodes[t.n()];
        let store = self.g.storage(t);
        let snode = &self.g.nodes[store.n()];
        match &snode.op {
            Op::Input(k) if !self.emulate => return self.inputs[*k][..].into(),
            Op::Operator(_) | Op::Scan(_) => return self.bufs[store.n()][..].into(),
            Op::Comp(c) if !snode.inline && c.tuple == 0 && c.width() == 1 => {
                return self.bufs[store.n()][..].into()
            }
            _ => {}
        }
        indices(&fixed(&node.shape))
            .map(|idx| self.read(t, &idx))
            .collect::<Vec<_>>()
            .into()
    }

    /// Value of `t` at full-shape coordinates `idx`.
    fn read(&self, t: TensorId, idx: &[usize]) -> f64 {
        let node = &self.g.nodes[t.n()];
        match &node.op {
            Op::Input(k) => {
                let v = self.inputs[*k][flat_index(&fixed(&node.shape), idx)];
                // block-scaled inputs are rounded whole up front
                if self.emulate && !node.precision.storage.is_fp8() {
                    node.precision.storage.round(v)
                } else {
                    v
                }
            }
            Op::Iota(d) => idx[*d] as f64,
            Op::Operator(_) | Op::Scan(_) => self.bufs[t.n()][flat_index(&fixed(&node.shape), idx)],
            Op::Reshape(src) => {
                let flat = flat_index(&fixed(&node.shape), idx);
                let src_shape = fixed(&self.g.nodes[src.n()].shape);
                let mut coords = vec![0; src_shape.len()];
                let mut p = flat;
                for d in (0..src_shape.len()).rev() {
                    coords[d] = p % src_shape[d];
                    p /= src_shape[d];
                }
                self.read(*src, &coords)
            }
            Op::Comp(c) => {
                let (pt, tup) = idx.split_at(c.points().len());
                let slot = flat_index(&fixed(&c.out[c.points().len()..]), tup);
                if node.inline {
                    let mut vals = vec![0.0; c.width()];
                    self.point(t.n(), pt, &mut vals, &mut Scratch::default());
                    return vals[slot];
                }
                let w = c.width();
                self.bufs[t.n()][flat_index(&fixed(c.points()), pt) * w + slot]
            }
        }
    }

    fn point(&self, i: usize, out_idx: &[usize], vals: &mut [f64], sc: &mut Scratch) {
        let c = self.g.comp_at(i).unwrap();
        let k = self.compiled[i].as_ref().unwrap();
        let w = c.width();
        let Scratch {
            arg_vals,
            idx,
            elem,
            pair,
            prog,
            cprog,
        } = sc;
        arg_vals.clear();
        arg_vals.resize(c.args.len(), 0.0);
        idx.clear();
        idx.extend_from_slice(out_idx);
        let base = idx.len();
        idx.extend(std::iter::repeat_n(0, c.red.len()));
        let mut body_at = |idx: &[usize], elem: &mut [f64]| {
            // affine args first: gathered args read their index from them
            for gathered in [false, true] {
                for (j, a) in c.args.iter().enumerate() {
                    if a.view.is_gather() != gathered {
                        continue;
                    }
                    let index = a.view.gather.as_ref().map_or(0.0, |g| arg_vals[g.arg]);
                    if let Some(d) = &k.direct[j] {
                        let gi = index.round() as i64;
                        if let Some(dims) = &d.guard {
                            let outside = dims.iter().any(|gd| {
                                let mut c = gd.offset + gd.gather * gi;
                                for (l, &x) in gd.coefs.iter().zip(idx) {
                                    c += l * x as i64;
                                }
                                c < 0 || c >= gd.extent
                            });
                            if outside {
                                arg_vals[j] = 0.0;
                                continue;
                            }
                        }
                        let mut at = d.base + d.gather * gi;
                        for (l, &x) in d.lin.iter().zip(idx) {
                            at += l * x as i64;
                        }
                        let buf = match d.src {
                            DirectSrc::Input(n) => &self.inputs[n],
                            DirectSrc::Buf(n) => &self.bufs[n],
                        };
                        arg_vals[j] = buf[at as usize];
                        continue;
                    }
                    let shape = fixed(&self.g.nodes[a.src.n()].shape);
                    arg_vals[j] = match a.view.coords_in(idx, &shape, index) {
                        Some(at) => self.read(a.src, &at),
                        None => 0.0,
                    };
                }
            }
            k.body.eval(arg_vals, elem, prog);
        };
        let Some(m) = &c.monoid else {
            body_at(idx, vals);
            return;
        };
        vals.copy_from_slice(&m.identity);
        let combine = k.combine.as_ref().unwrap();
        elem.clear();
        elem.resize(w, 0.0);
        pair.clear();
        pair.resize(2 * w, 0.0);
        // softmax weights rounded to the operand format (an online-softmax
        // reduction on a tile that feeds exp(s − m) to the tensor cores):
        // the element's weight is rounded before it scales g₁…gₖ
        let operand = self.g.nodes[i].precision.operand;
        let round_weights =
            self.emulate && operand != Format::F32 && m.kind() == MonoidKind::OnlineSoftmax;
        let red = fixed(&c.red);
        let n_red: usize = red.iter().product();
        for step in 0..n_red {
            if step > 0 {
                // odometer over the reduce dims, last fastest
                for d in (0..red.len()).rev() {
                    idx[base + d] += 1;
                    if idx[base + d] < red[d] {
                        break;
                    }
                    idx[base + d] = 0;
                }
            }
            body_at(idx, elem);
            if round_weights {
                // m' = max(m, s); accᵢ' = accᵢ·exp(m − m') + gᵢ·round(exp(s − m'))
                let mx = vals[0].max(elem[0]);
                if mx == f64::NEG_INFINITY {
                    continue;
                }
                let sa = if vals[0] == f64::NEG_INFINITY {
                    0.0
                } else {
                    (vals[0] - mx).exp()
                };
                let se = precision::round_weight(operand, (elem[0] - mx).exp());
                vals[0] = mx;
                for t in 1..w {
                    vals[t] = vals[t] * sa + elem[t] * se;
                }
                if self.emulate {
                    let f = self.g.nodes[i].precision.accum;
                    vals.iter_mut().for_each(|v| *v = f.round(*v));
                }
                continue;
            }
            pair[..w].copy_from_slice(vals);
            pair[w..].copy_from_slice(elem);
            combine.eval(pair, vals, cprog);
            if self.emulate {
                let f = self.g.nodes[i].precision.accum;
                vals.iter_mut().for_each(|v| *v = f.round(*v));
            }
        }
    }
}

/// Advance row-major coordinates `idx` in `shape` by one (last dim fastest).
fn odometer(idx: &mut [usize], shape: &[usize]) {
    for d in (0..shape.len()).rev() {
        idx[d] += 1;
        if idx[d] < shape[d] {
            return;
        }
        idx[d] = 0;
    }
}

fn flat_index(shape: &[usize], idx: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |acc, (i, n)| acc * n + i)
}

fn indices(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    (0..shape.iter().product::<usize>()).map(move |mut p| {
        let mut idx = vec![0; shape.len()];
        for d in (0..shape.len()).rev() {
            idx[d] = p % shape[d];
            p /= shape[d];
        }
        idx
    })
}

/// Copy the subgraph under `outs` into `dst`, substituting `Var(i)` with `vars[i]`.
pub(super) fn replay(
    src: &ExprGraph,
    outs: &[ExprId],
    vars: &[ExprId],
    dst: &mut ExprGraph,
) -> Vec<ExprId> {
    let nodes = src.nodes_slice();
    let last = outs.iter().map(|o| o.index()).max().unwrap_or(0) as usize;
    let mut map: Vec<ExprId> = Vec::with_capacity(last + 1);
    for node in &nodes[..=last] {
        let m = |e: &ExprId| map[e.index() as usize];
        let id = match node {
            Node::Var(v) => vars[*v as usize],
            Node::Lit(bits) => dst.lit(f64::from_bits(*bits)),
            Node::Add(a, b) => dst.add(m(a), m(b)),
            Node::Mul(a, b) => dst.mul(m(a), m(b)),
            Node::Neg(a) => dst.neg(m(a)),
            Node::Recip(a) => dst.recip(m(a)),
            Node::Sqrt(a) => dst.sqrt(m(a)),
            Node::Sin(a) => dst.sin(m(a)),
            Node::Atan2(a, b) => dst.atan2(m(a), m(b)),
            Node::Exp2(a) => dst.exp2(m(a)),
            Node::Log2(a) => dst.log2(m(a)),
            Node::Select(c, a, b) => dst.select(m(c), m(a), m(b)),
        };
        map.push(id);
    }
    outs.iter().map(|o| map[o.index() as usize]).collect()
}

/// Input vars reachable from `out`.
pub(super) fn vars_reachable(g: &ExprGraph, out: ExprId) -> Vec<u16> {
    let live = g.live_set(&[out]);
    let nodes = g.nodes_slice();
    live.iter()
        .filter_map(|&i| match nodes[i] {
            Node::Var(v) => Some(v),
            _ => None,
        })
        .collect()
}

/// Canonical expression for structural comparison (see [`TensorGraph::canon`]).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Canon {
    leaves: Vec<Arg>,
    nodes: Vec<Node>,
}

/// Match `g · exp(s − m)` (or just `exp(s − m)`) where `m` is a var and
/// `s` is any expression. `exp(x)` lowers to `exp2(x · log2e)` and
/// `a − b` to `add(a, neg(b))`.
fn match_exp_diff(g: &ExprGraph, out: ExprId) -> Option<(ExprId, u16, Option<ExprId>)> {
    let n = g.nodes_slice();
    let at = |e: ExprId| n[e.index() as usize];
    let exp_diff = |e: ExprId| -> Option<(ExprId, u16)> {
        let Node::Exp2(x) = at(e) else { return None };
        let Node::Mul(d, l) = at(x) else { return None };
        if at(l) != Node::lit(std::f64::consts::LOG2_E) {
            return None;
        }
        let Node::Add(s, nb) = at(d) else { return None };
        let Node::Neg(b) = at(nb) else { return None };
        match at(b) {
            Node::Var(m) => Some((s, m)),
            _ => None,
        }
    };
    if let Some((s, m)) = exp_diff(out) {
        return Some((s, m, None));
    }
    let Node::Mul(a, b) = at(out) else {
        return None;
    };
    exp_diff(a)
        .map(|(s, m)| (s, m, Some(b)))
        .or_else(|| exp_diff(b).map(|(s, m)| (s, m, Some(a))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-9 * (1.0 + y.abs()), "{x} vs {y}");
        }
    }

    fn ramp(n: usize, scale: f64, off: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919) % 97) as f64 * scale + off)
            .collect()
    }

    fn rmsnorm(g: &mut TensorGraph, x: TensorId) -> TensorId {
        let shape = g.shape(x).to_vec();
        let dim = shape[1].n();
        let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
        let sum = g.reduce(sq, 1, Monoid::sum()).unwrap();
        let inv = g
            .map(&[sum], |v| {
                ExprId::from_f64(1.0)
                    / (v[0] / ExprId::from_f64(dim as f64) + ExprId::from_f64(1e-6)).sqrt()
            })
            .unwrap();
        let inv = g.broadcast(inv, &shape).unwrap();
        g.map(&[x, inv], |v| v[0] * v[1]).unwrap()
    }

    #[test]
    fn rmsnorm_matches_reference_and_fuses_to_two_comps() {
        let (rows, dim) = (3, 8);
        let mut g = TensorGraph::new();
        let x = g.input(&[rows, dim]);
        let y = rmsnorm(&mut g, x);
        let data = ramp(rows * dim, 0.1, -1.0);

        let mut want = vec![0.0; rows * dim];
        for r in 0..rows {
            let row = &data[r * dim..(r + 1) * dim];
            let rms = (row.iter().map(|v| v * v).sum::<f64>() / dim as f64 + 1e-6).sqrt();
            for c in 0..dim {
                want[r * dim + c] = row[c] / rms;
            }
        }
        close(&g.eval(std::slice::from_ref(&data), &[y])[0], &want);

        // sq folds into the reduction; the broadcast folds into the output
        // map; inv (a sqrt, recomputed dim× if inlined) stays materialized
        let (fg, outs) = g.substitute_maps(&[y]);
        assert_eq!(fg.count_comps(&outs), 3);
        close(&fg.eval(std::slice::from_ref(&data), &outs)[0], &want);
        // unconditional substitution folds everything into two comps
        let (fg, outs) = g.substitute_maps_with(&[y], false);
        assert_eq!(fg.count_comps(&outs), 2);
        close(&fg.eval(&[data], &outs)[0], &want);
    }

    #[test]
    fn matmul_and_transpose() {
        let mut g = TensorGraph::new();
        let a = g.input(&[2, 3]);
        let b = g.input(&[2, 3]);
        let bt = g.transpose(b, &[1, 0]).unwrap();
        let c = g.matmul(a, bt).unwrap();
        assert_eq!(g.shape(c), &[2, 2]);
        let av = vec![1., 2., 3., 4., 5., 6.];
        let bv = vec![1., 0., 1., 0., 1., 0.];
        close(
            &g.eval(&[av.clone(), bv.clone()], &[c])[0],
            &[4., 2., 10., 5.],
        );

        // the transpose is a view: after substitution it is gone
        let (fg, outs) = g.substitute_maps(&[c]);
        assert_eq!(fg.count_comps(&outs), 1);
        close(&fg.eval(&[av, bv], &outs)[0], &[4., 2., 10., 5.]);
    }

    #[test]
    fn shape_errors() {
        let mut g = TensorGraph::new();
        let a = g.input(&[2, 3]);
        let b = g.input(&[4, 3]);
        assert!(g.matmul(a, b).is_err());
        assert!(g.map(&[a, b], |v| v[0] + v[1]).is_err());
        assert!(g.broadcast(a, &[2, 4]).is_err());
        assert!(g.transpose(a, &[0, 0]).is_err());
        assert!(g.reduce(a, 2, Monoid::sum()).is_err());
        let out_of_bounds = Arg {
            src: a,
            view: View::pick(2, &[Some(1), Some(0)]),
        };
        assert!(g
            .comp(&[2, 3], &[], vec![out_of_bounds], None, |v| vec![v[0]])
            .is_err());
    }

    /// Sharding the key axis of the flash-fused attention across two
    /// "devices" and folding the partial (m, l, acc) states with the
    /// online-softmax combine reproduces the unsharded result exactly.
    #[test]
    fn sharded_online_softmax_is_the_all_reduce() {
        let (n, d, dv) = (8, 4, 4);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let y = naive_attention(&mut g, q, k, v, false);
        let (g, outs) = g.stabilize_softmax(&[y]);
        let (g, _outs) = g.substitute_maps(&outs);
        let fused = (0..g.nodes().len())
            .find(|&i| matches!(&g.nodes()[i].op, Op::Comp(c) if c.tuple == 1))
            .unwrap();
        let fused = TensorId::from_index(fused);
        let (sg, partials, combined) = g.shard_reduction(fused, 2).unwrap();
        assert_eq!(partials.len(), 2);
        let inputs = [
            ramp(n * d, 0.05, -1.0),
            ramp(n * d, 0.07, 0.3),
            ramp(n * dv, 0.1, -2.0),
        ];
        let want = g.eval(&inputs, &[fused]).remove(0);
        let got = sg.eval(&inputs, &[combined]).remove(0);
        close(&got, &want);
        // and a plain sum, which must shard too
        let mut g = TensorGraph::new();
        let x = g.input(&[3, 6]);
        let s = g.reduce(x, 1, Monoid::sum()).unwrap();
        let (sg, _, c) = g.shard_reduction(s, 3).unwrap();
        let data = ramp(18, 0.5, -1.0);
        close(
            &sg.eval(std::slice::from_ref(&data), &[c])[0],
            &g.eval(std::slice::from_ref(&data), &[s])[0],
        );
    }

    /// Split-K, one launch: decode attention (1 query row, 64 keys), a
    /// long-K matmul and an argmax all split into 2…16 chunks and fold
    /// back with their own monoid; readers of the split node are untouched.
    #[test]
    fn split_reduction_folds_partials_with_the_monoid() {
        let (nk, d, dv) = (64, 4, 4);
        let mut g = TensorGraph::new();
        let q = g.input(&[1, d]);
        let k = g.input(&[nk, d]);
        let v = g.input(&[nk, dv]);
        let y = naive_attention(&mut g, q, k, v, false);
        let (g, outs) = g.stabilize_softmax(&[y]);
        let (g, outs) = g.substitute_maps(&outs);
        let fused = (0..g.nodes().len())
            .find(|&i| matches!(&g.nodes()[i].op, Op::Comp(c) if c.tuple == 1))
            .map(TensorId::from_index)
            .unwrap();
        let inputs = [
            ramp(d, 0.3, -1.0),
            ramp(nk * d, 0.07, -2.0),
            ramp(nk * dv, 0.1, -2.0),
        ];
        let want = g.eval(&inputs, &outs).remove(0);
        assert_eq!(g.split_k_candidates(fused, 8, 1 << 20), vec![2, 4, 8]);
        for p in [2, 4, 8, 16] {
            let (sg, so) = g.split_reduction(&outs, fused, p).unwrap();
            assert_eq!(sg.count_comps(&so), g.count_comps(&outs) + 1);
            close(&sg.eval(&inputs, &so).remove(0), &want);
        }
        assert!(g.split_reduction(&outs, fused, 3).is_err());

        // a long-K matmul: [2, 96] · [96, 3]
        let mut g = TensorGraph::new();
        let a = g.input(&[2, 96]);
        let b = g.input(&[96, 3]);
        let c = g.matmul(a, b).unwrap();
        let data = [ramp(192, 0.01, -1.0), ramp(288, 0.02, -2.0)];
        let want = g.eval(&data, &[c]).remove(0);
        for p in [2, 3, 8, 32] {
            let (sg, so) = g.split_reduction(&[c], c, p).unwrap();
            close(&sg.eval(&data, &so).remove(0), &want);
        }

        // argmax: a tuple monoid whose tie rule must survive the split
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 16]);
        let i = g.iota(&[2, 16], 1);
        let rank = 2;
        let am = g
            .comp_tuple(
                &[2, 2],
                &[16],
                vec![
                    Arg {
                        src: i,
                        view: View::pick(rank, &[Some(0), Some(1)]),
                    },
                    Arg {
                        src: x,
                        view: View::pick(rank, &[Some(0), Some(1)]),
                    },
                ],
                Some(Monoid::argmax()),
                |v| vec![v[0], v[1]],
            )
            .unwrap();
        let data = vec![(0..32).map(|j| ((j * 7) % 5) as f64).collect::<Vec<_>>()];
        let want = g.eval(&data, &[am]).remove(0);
        let (sg, so) = g.split_reduction(&[am], am, 4).unwrap();
        close(&sg.eval(&data, &so).remove(0), &want);

        // top-k: k argmax rounds in one tuple reduction, ties to the smaller
        // index, the same result split or not
        for k in [1usize, 3, 5] {
            let mut g = TensorGraph::new();
            let x = g.input(&[2, 16]);
            let i = g.iota(&[2, 16], 1);
            let tk = g
                .comp_tuple(
                    &[2, 2 * k],
                    &[16],
                    vec![
                        Arg {
                            src: i,
                            view: View::pick(2, &[Some(0), Some(1)]),
                        },
                        Arg {
                            src: x,
                            view: View::pick(2, &[Some(0), Some(1)]),
                        },
                    ],
                    Some(Monoid::topk(k)),
                    |v| {
                        let mut o = vec![v[0], v[1]];
                        for _ in 1..k {
                            o.push(ExprId::ZERO);
                            o.push(ExprId::from_f64(-(f32::MAX as f64)));
                        }
                        o
                    },
                )
                .unwrap();
            let data = vec![(0..32)
                .map(|j| ((j * 7) % 5) as f64 + 0.01 * (j % 3) as f64)
                .collect::<Vec<_>>()];
            let got = g.eval(&data, &[tk]).remove(0);
            for row in 0..2 {
                let mut idx: Vec<usize> = (0..16).collect();
                let v = &data[0][row * 16..row * 16 + 16];
                idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
                for r in 0..k {
                    assert_eq!(
                        got[row * 2 * k + 2 * r] as usize,
                        idx[r],
                        "k {k} row {row} rank {r}"
                    );
                    assert_eq!(got[row * 2 * k + 2 * r + 1], v[idx[r]]);
                }
            }
            let (sg, so) = g.split_reduction(&[tk], tk, 4).unwrap();
            close(&sg.eval(&data, &so).remove(0), &got);
        }

        // the rule: a 2-point, 4096-long sum splits to fill 64 points
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 4096]);
        let s = g.reduce(x, 1, Monoid::sum()).unwrap();
        let (sg, so) = g.schedule_split_k(&[s], 64);
        assert_eq!(sg.count_comps(&so), 2);
        let data = vec![ramp(8192, 0.001, -1.0)];
        close(&sg.eval(&data, &so)[0], &g.eval(&data, &[s])[0]);
        // ... and leaves a reduction with enough points alone
        let (sg, so) = g.schedule_split_k(&[s], 2);
        assert_eq!(sg.count_comps(&so), 1);
    }

    #[test]
    fn maps_substitute_through_affine_reshapes() {
        // m = 2x read as [4, 2, 4] by a comp: the reshape is a merge seen
        // from the reader, so m folds into the reader and no buffer remains
        let mut g = TensorGraph::new();
        let x = g.input(&[4, 8]);
        let m = g.map(&[x], |v| v[0] * ExprId::from_f64(2.0)).unwrap();
        let r = g.reshape(m, &[4, 2, 4]).unwrap();
        let y = g
            .comp(
                &[4, 2],
                &[4],
                vec![Arg {
                    src: r,
                    view: View::pick(3, &[Some(0), Some(1), Some(2)]),
                }],
                Some(Monoid::sum()),
                |v| vec![v[0]],
            )
            .unwrap();
        let data = ramp(32, 0.25, -1.0);
        let want = g.eval(std::slice::from_ref(&data), &[y]).remove(0);
        let (fg, outs) = g.substitute_maps(&[y]);
        assert_eq!(fg.count_comps(&outs), 1);
        close(&fg.eval(std::slice::from_ref(&data), &outs)[0], &want);

        // a splitting reshape ([4, 8] -> [32]) is not affine: m stays
        let mut g = TensorGraph::new();
        let x = g.input(&[4, 8]);
        let m = g.map(&[x], |v| v[0] * ExprId::from_f64(2.0)).unwrap();
        let r = g.reshape(m, &[32]).unwrap();
        let y = g.reduce(r, 0, Monoid::sum()).unwrap();
        let (fg, outs) = g.substitute_maps(&[y]);
        assert_eq!(fg.count_comps(&outs), 2);
        close(
            &fg.eval(std::slice::from_ref(&data), &outs)[0],
            &g.eval(std::slice::from_ref(&data), &[y])[0],
        );
    }

    #[test]
    fn relayout_orders_a_producer_for_its_consumer() {
        // p[g, s, d] = x[s, g, d] * 2 (materialized by the reduce that
        // reads it); y[s, m] = Σ_{g,d} p[g, s, d] · w[m, g, d]. After
        // relayout p is [s, g, d] and the reduce dims merge into one.
        let (sq, groups, hd, dm) = (8, 3, 4, 16);
        let mut g = TensorGraph::new();
        let x = g.input(&[sq, groups, hd]);
        let w = g.input(&[dm, groups, hd]);
        let p = g
            .comp(
                &[groups, sq, hd],
                &[],
                vec![Arg {
                    src: x,
                    view: View::pick(3, &[Some(1), Some(0), Some(2)]),
                }],
                None,
                |v| vec![v[0] * ExprId::from_f64(2.0)],
            )
            .unwrap();
        let y = g
            .comp(
                &[sq, dm],
                &[groups, hd],
                vec![
                    Arg {
                        src: p,
                        view: View::pick(4, &[Some(2), Some(0), Some(3)]),
                    },
                    Arg {
                        src: w,
                        view: View::pick(4, &[Some(1), Some(2), Some(3)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let inputs = [
            ramp(sq * groups * hd, 0.01, -0.3),
            ramp(dm * groups * hd, 0.02, 0.1),
        ];
        let want = g.eval(&inputs, &[y]).remove(0);
        let (rg, outs) = g.relayout(&[y]);
        assert_eq!(
            rg.nodes()[p.n()].shape,
            vec![Dim::Fixed(sq), Dim::Fixed(groups), Dim::Fixed(hd)]
        );
        close(&rg.eval(&inputs, &outs)[0], &want);
        let (mg, outs) = rg.merge_reduce_dims(&outs);
        assert_eq!(
            mg.comp_at(outs[0].n()).unwrap().red,
            vec![Dim::Fixed(groups * hd)]
        );
        close(&mg.eval(&inputs, &outs)[0], &want);
    }

    #[test]
    fn split_reduce_dims_factors_a_small_dim() {
        // y[j, d] = Σ_{r, i} a[i, r, j] · b[i, r, d], r small
        let (n, r, dd) = (32, 3, 8);
        let mut g = TensorGraph::new();
        let a = g.input(&[n, r, n]);
        let b = g.input(&[n, r, dd]);
        let y = g
            .comp(
                &[n, dd],
                &[r, n],
                vec![
                    Arg {
                        src: a,
                        view: View::pick(4, &[Some(3), Some(2), Some(0)]),
                    },
                    Arg {
                        src: b,
                        view: View::pick(4, &[Some(3), Some(2), Some(1)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let inputs = [ramp(n * r * n, 0.01, -0.3), ramp(n * r * dd, 0.02, 0.1)];
        let want = g.eval(&inputs, &[y]).remove(0);
        let (sg, outs) = g.split_reduce_dims(&[y]);
        let c = sg.comp_at(outs[0].n()).unwrap();
        assert!(c.red.is_empty() && c.args.len() == r);
        let inner = sg.comp_at(c.args[0].src.n()).unwrap();
        assert_eq!(inner.red, vec![Dim::Fixed(n)]);
        close(&sg.eval(&inputs, &outs)[0], &want);
    }

    #[test]
    fn merge_reduce_dims_makes_grouped_projection_a_matmul() {
        // out[s, m] = Σ_{g,r,d} att[s, g, r, d] · wo[m, g, r, d]: three
        // reduce dims, contiguous in both operands -> one of size 24
        let (sq, groups, per, hd, dm) = (8, 2, 3, 4, 16);
        let mut g = TensorGraph::new();
        let att = g.input(&[sq, groups, per, hd]);
        let wo = g.input(&[dm, groups, per, hd]);
        let r = 5;
        let y = g
            .comp(
                &[sq, dm],
                &[groups, per, hd],
                vec![
                    Arg {
                        src: att,
                        view: View::pick(r, &[Some(0), Some(2), Some(3), Some(4)]),
                    },
                    Arg {
                        src: wo,
                        view: View::pick(r, &[Some(1), Some(2), Some(3), Some(4)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let inputs = [
            ramp(sq * groups * per * hd, 0.01, -0.3),
            ramp(dm * groups * per * hd, 0.02, 0.1),
        ];
        let want = g.eval(&inputs, &[y]).remove(0);
        let (mg, outs) = g.merge_reduce_dims(&[y]);
        let c = mg.comp_at(outs[0].n()).unwrap();
        assert_eq!(c.red, vec![Dim::Fixed(groups * per * hd)]);
        assert_eq!(c.args[0].view.rank(), 3);
        close(&mg.eval(&inputs, &outs)[0], &want);
    }

    #[test]
    fn merge_siblings_fuses_gate_and_up_with_their_activation() {
        // swiglu: silu(x·wg) * (x·wu) — two reductions over the same
        // space with the same consumer become one, and the activation
        // is that reduction's epilogue: one kernel
        let mut g = TensorGraph::new();
        let x = g.input(&[64, 64]);
        let wg = g.input(&[64, 128]);
        let wu = g.input(&[64, 128]);
        let gate = g.matmul(x, wg).unwrap();
        let up = g.matmul(x, wu).unwrap();
        let y = g
            .map(&[gate, up], |v| {
                let one = ExprId::from_f64(1.0);
                v[0] / (one + (-v[0]).exp()) * v[1]
            })
            .unwrap();
        let inputs = [
            ramp(64 * 64, 0.01, -0.5),
            ramp(64 * 128, 0.003, -0.2),
            ramp(64 * 128, -0.002, 0.3),
        ];
        let want = g.eval(&inputs, &[y]).remove(0);
        let (mg, outs) = g.merge_siblings(&[y]);
        assert_eq!(mg.count_comps(&outs), 2);
        close(&mg.eval(&inputs, &outs)[0], &want);
        let mut mg = mg;
        let mut fused = mg.clone();
        fused.schedule_inline(&outs);
        assert_eq!(fused.materialized(&outs).len(), 1);
        assert!(fused
            .tiled_reduction(outs[0].n())
            .is_some_and(|r| r != outs[0].n()));
        mg.schedule_auto(&outs);
        // under the 4×4 register tile the SiLU (exp) epilogue is handed
        // back to its own map (`unfuse_heavy_epilogue`): two kernels
        assert_eq!(mg.materialized(&outs).len(), 2);
        let r = mg.tiled_reduction(outs[0].n());
        assert_eq!(r, None);
        close(&mg.eval(&inputs, &outs)[0], &want);

        // siblings with different consumers stay apart: merging would
        // cost each of them its own epilogue
        let mut g = TensorGraph::new();
        let x = g.input(&[64, 64]);
        let wg = g.input(&[64, 128]);
        let wu = g.input(&[64, 128]);
        let gate = g.matmul(x, wg).unwrap();
        let up = g.matmul(x, wu).unwrap();
        let a = g.map(&[gate], |v| v[0] * ExprId::from_f64(2.0)).unwrap();
        let b = g.map(&[up], |v| v[0] * ExprId::from_f64(3.0)).unwrap();
        let (mg, outs) = g.merge_siblings(&[a, b]);
        assert_eq!(mg.count_comps(&outs), 4);
    }

    #[test]
    fn eliminate_common_merges_twins() {
        let mut g = TensorGraph::new();
        let x = g.input(&[4, 4]);
        let a = g.map(&[x], |v| v[0] * v[0]).unwrap();
        let b = g.map(&[x], |v| v[0] * v[0]).unwrap(); // twin of a
        let c = g.map(&[x], |v| v[0] + v[0]).unwrap(); // different body
        let y = g.map(&[a, b, c], |v| v[0] + v[1] + v[2]).unwrap();
        let data = ramp(16, 0.3, -1.0);
        let want = g.eval(std::slice::from_ref(&data), &[y]).remove(0);
        let (cg, outs) = g.eliminate_common(&[y]);
        assert_eq!(cg.count_comps(&outs), 3);
        close(&cg.eval(std::slice::from_ref(&data), &outs)[0], &want);
    }

    #[test]
    fn iota_is_the_coordinate_through_any_view() {
        let mut g = TensorGraph::new();
        let j = g.iota(&[2, 3], 1);
        let t = g.transpose(j, &[1, 0]).unwrap(); // t[a, b] = j[b, a] = a
        let got = g.eval(&[], &[t]).remove(0);
        assert_eq!(got, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let (fg, outs) = g.substitute_maps(&[t]);
        assert_eq!(fg.eval(&[], &outs)[0], got);
    }

    #[test]
    fn reshape_is_a_free_alias() {
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 6]);
        let r = g.reshape(x, &[3, 4]).unwrap();
        // strided pick view over the reshape: r[a, 2b] for a<3, b<2
        let picked = g
            .comp(
                &[3, 2],
                &[],
                vec![Arg {
                    src: r,
                    view: View {
                        guard: false,
                        gather: None,
                        forms: vec![
                            Form::unit(2, 0),
                            Form {
                                coefs: vec![0, 2],
                                offset: 1,
                            },
                        ],
                    },
                }],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let got = g.eval(std::slice::from_ref(&data), &[picked]).remove(0);
        assert_eq!(got, vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
        assert!(g.reshape(x, &[5]).is_err());
        // the reshape has no buffer: nothing to materialize but the comp
        assert_eq!(g.materialized(&[picked]).len(), 1);
        let (fg, outs) = fg_subst(&g, &[picked]);
        assert_eq!(fg.eval(std::slice::from_ref(&data), &outs)[0], got);
    }

    fn fg_subst(g: &TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        g.substitute_maps(outs)
    }

    #[test]
    fn monoid_kinds() {
        assert_eq!(Monoid::sum().kind(), MonoidKind::Sum);
        assert_eq!(Monoid::max().kind(), MonoidKind::Max);
        assert_eq!(Monoid::online_softmax(2).kind(), MonoidKind::OnlineSoftmax);
    }

    /// `[n, n]` additive causal mask built in-graph from coordinates:
    /// 0 where `j <= i`, a large negative where `j > i`.
    fn causal_mask(g: &mut TensorGraph, n: Dim) -> TensorId {
        let i = g.iota(&[n, n], 0);
        let j = g.iota(&[n, n], 1);
        g.map(&[i, j], |v| {
            let above = v[1] - v[0]; // > 0 when j > i
            tang_expr::with_graph(|eg| {
                let neg = eg.lit(-1e30);
                eg.select(above, neg, ExprId::ZERO)
            })
        })
        .unwrap()
    }

    /// softmax(q·kᵀ [+ causal mask])·v with every intermediate written out.
    fn naive_attention(
        g: &mut TensorGraph,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        causal: bool,
    ) -> TensorId {
        let n = g.shape(q)[0];
        let nk = g.shape(k)[0];
        let dv = g.shape(v)[1];
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let mut s = g.matmul(q, kt).unwrap(); // [n, n]
        if causal {
            let mask = causal_mask(g, n);
            s = g.map(&[s, mask], |v| v[0] + v[1]).unwrap();
        }
        let m = g.reduce(s, 1, Monoid::max()).unwrap(); // [n, 1]
        let mb = g.broadcast(m, &[n, nk]).unwrap();
        let p = g.map(&[s, mb], |v| (v[0] - v[1]).exp()).unwrap(); // [n, n]
        let l = g.reduce(p, 1, Monoid::sum()).unwrap(); // [n, 1]
        let o = g.matmul(p, v).unwrap(); // [n, dv]
        let lb = g.broadcast(l, &[n, dv]).unwrap();
        g.map(&[o, lb], |v| v[0] / v[1]).unwrap()
    }

    /// One symbolic graph serves every sequence length: rewrite and
    /// differentiate with `n` unbound, bind per size, match fixed builds.
    #[test]
    fn symbolic_seq_len_binds_to_any_size() {
        let (d, dv) = (4, 4);
        let n = Dim::Sym("n");
        let mut g = TensorGraph::new();
        let q = g.input(&[n, Dim::Fixed(d)]);
        let k = g.input(&[n, Dim::Fixed(d)]);
        let v = g.input(&[n, Dim::Fixed(dv)]);
        let y = naive_attention(&mut g, q, k, v, true);
        assert!(!g.is_bound());
        let grads = g.backward(y, &[q]).unwrap();
        let (sg, outs) = g.substitute_maps(&[y, grads[0]]);
        let (sg, outs) = sg.stabilize_softmax(&outs);
        assert!(sg
            .reduction_kinds(&outs)
            .contains(&MonoidKind::OnlineSoftmax));
        assert!(sg.bind(&[("m", 3)]).is_err(), "wrong symbol must fail");

        for size in [5usize, 16] {
            let mut bound = sg.bind(&[("n", size)]).unwrap();
            assert!(bound.is_bound());
            bound.schedule_inline_up_to(&outs, dv);
            let inputs = [
                ramp(size * d, 0.05, -1.0),
                ramp(size * d, 0.07, 0.3),
                ramp(size * dv, 0.1, -2.0),
            ];
            let got = bound.eval(&inputs, &outs);

            let mut fg = TensorGraph::new();
            let fq = fg.input(&[size, d]);
            let fk = fg.input(&[size, d]);
            let fv = fg.input(&[size, dv]);
            let fy = naive_attention(&mut fg, fq, fk, fv, true);
            let fgr = fg.backward(fy, &[fq]).unwrap();
            let want = fg.eval(&inputs, &[fy, fgr[0]]);
            close(&got[0], &want[0]);
            close(&got[1], &want[1]);
        }
    }

    #[test]
    fn schedule_auto_splits_narrow_reductions_only() {
        // 64 rows x 2048: few points, wide reduce -> split
        let mut g = TensorGraph::new();
        let x = g.input(&[64, 2048]);
        let y = g.reduce(x, 1, Monoid::sum()).unwrap();
        g.schedule_auto(&[y]);
        assert_eq!(g.nodes()[y.n()].split, 256);

        // 1M points x 8: plenty of threads -> one thread per point
        let mut g = TensorGraph::new();
        let x = g.input(&[1 << 20, 8]);
        let y = g.reduce(x, 1, Monoid::sum()).unwrap();
        g.schedule_auto(&[y]);
        assert_eq!(g.nodes()[y.n()].split, 0);
        assert_eq!(g.nodes()[y.n()].tile, (0, 0, 0));

        // a plain matmul: 16×16 shared-memory tile, 4×4 register tile when wide
        let mut g = TensorGraph::new();
        let a = g.input(&[256, 128]);
        let b = g.input(&[128, 64]);
        let c = g.matmul(a, b).unwrap();
        g.schedule_auto(&[c]);
        assert_eq!(g.nodes()[c.n()].tile, (16, 16, 32));
        assert_eq!(g.nodes()[c.n()].reg, (4, 4));
        assert_eq!(g.nodes()[c.n()].split, 0);

        // attention after the softmax rewrite: scores are lane-invariant ->
        // tiled, scores inlined, the normalizer is the tile's epilogue
        let (n, d, dv) = (64, 16, 32);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let y = naive_attention(&mut g, q, k, v, false);
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        g.schedule_auto(&outs);
        let fused = g
            .nodes()
            .iter()
            .position(|nd| matches!(&nd.op, Op::Comp(c) if c.tuple == 1))
            .unwrap();
        assert!(g.nodes()[fused].inline);
        assert_eq!(g.nodes()[outs[0].n()].tile, (dv as u32, 8, 32));
        assert_eq!(g.tiled_reduction(outs[0].n()), Some(fused));
        assert_eq!(g.materialized(&outs).len(), 1);
    }

    #[test]
    fn unbound_graph_cannot_be_scheduled_or_evaluated() {
        let mut g = TensorGraph::new();
        let x = g.input(&[Dim::Sym("n"), Dim::Fixed(2)]);
        let y = g.reduce(x, 1, Monoid::sum()).unwrap();
        let r = std::panic::catch_unwind(|| g.eval(&[vec![0.0; 4]], &[y]));
        assert!(r.is_err());
    }

    #[test]
    fn attention_is_derived_not_written() {
        check_attention_derivation(false);
    }

    /// With a mask, `s` in `exp(s − max_j s)` is `scores + mask`, not a
    /// bare arg: the matcher must compare expressions, not vars.
    #[test]
    fn causal_attention_is_derived_too() {
        check_attention_derivation(true);
    }

    fn check_attention_derivation(causal: bool) {
        let (n, d, dv) = (16, 4, 4);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let y = naive_attention(&mut g, q, k, v, causal);

        let inputs = [
            ramp(n * d, 0.05, -1.0),
            ramp(n * d, 0.07, 0.3),
            ramp(n * dv, 0.1, -2.0),
        ];
        let want = g.eval(&inputs, &[y]).remove(0);
        if causal {
            // row 0 attends only to itself: y[0] == v[0]
            close(&want[..dv], &inputs[2][..dv]);
        }

        // naive: scores and p are materialized n×n
        assert!(g.materialized(&[y]).iter().any(|(_, s)| numel(s) == n * n));

        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        g.schedule_inline_up_to(&outs, dv);

        // flash: one online-softmax reduction, q·k recomputed inside it,
        // and nothing n×n ever hits memory.
        let kinds = g.reduction_kinds(&outs);
        assert_eq!(
            kinds
                .iter()
                .filter(|k| **k == MonoidKind::OnlineSoftmax)
                .count(),
            1,
            "{kinds:?}"
        );
        assert!(!kinds.contains(&MonoidKind::Max), "max_j s should be gone");
        let mats = g.materialized(&outs);
        assert!(
            mats.iter().all(|(_, s)| numel(s) < n * n),
            "materialized: {mats:?}"
        );
        // scores (inlined), online reduce [n, 1, dv, 3] (inlined into its
        // normalizing epilogue), final divide: one kernel
        assert_eq!(g.count_comps(&outs), 3);
        assert_eq!(mats.len(), 1);

        close(&g.eval(&inputs, &outs)[0], &want);
    }

    /// `y[o, t] = Σ_{c,k} w[o,c,k] · x[c, t + d·k − p]` with zero padding.
    fn conv1d(g: &mut TensorGraph, x: TensorId, w: TensorId, dil: i64, pad: i64) -> TensorId {
        let (ci, t) = (g.shape(x)[0].n(), g.shape(x)[1].n());
        let (co, k) = (g.shape(w)[0].n(), g.shape(w)[2].n());
        // space: (o, t, c, k)
        let xv = View {
            gather: None,
            forms: vec![
                Form::unit(4, 2),
                Form {
                    coefs: vec![0, 1, 0, dil],
                    offset: -pad,
                },
            ],
            guard: true,
        };
        let wv = View::pick(4, &[Some(0), Some(2), Some(3)]);
        g.comp(
            &[co, t],
            &[ci, k],
            vec![Arg { src: x, view: xv }, Arg { src: w, view: wv }],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d_ref(
        x: &[f64],
        w: &[f64],
        ci: usize,
        co: usize,
        t: usize,
        k: usize,
        dil: i64,
        pad: i64,
    ) -> Vec<f64> {
        let mut y = vec![0.0; co * t];
        for o in 0..co {
            for tt in 0..t {
                for c in 0..ci {
                    for kk in 0..k {
                        let s = tt as i64 + dil * kk as i64 - pad;
                        if (0..t as i64).contains(&s) {
                            y[o * t + tt] += w[(o * ci + c) * k + kk] * x[c * t + s as usize];
                        }
                    }
                }
            }
        }
        y
    }

    #[test]
    fn schedule_auto_tiles_multi_dim_reductions() {
        // direct conv: red (c, k) -> chunked c, inner k; 2-D register tile
        let mut g = TensorGraph::new();
        let x = g.input(&[64, 200]);
        let w = g.input(&[96, 64, 7]);
        let y = conv1d(&mut g, x, w, 3, 9);
        g.schedule_auto(&[y]);
        assert_eq!(g.nodes()[y.n()].tile, (16, 16, 32));
        assert_eq!(g.nodes()[y.n()].reg, (4, 4));
        // too many inner elements per chunk step: not tiled
        assert!(!tileable_red(&dims(&[64, 9, 9])));
        assert!(tileable_red(&dims(&[512, 7])));
        // a narrow trailing point dim is uniform, not the lane axis
        assert_eq!(tile_axes(&[512, 512, 7]), (Some(1), Some(0)));
        assert_eq!(tile_axes(&[256, 2000, 10]), (Some(1), Some(0)));
        assert_eq!(tile_axes(&[1, 8]), (Some(1), None));
    }

    #[test]
    fn windowed_staging_shrinks_conv_regions() {
        // x[c, t + 3k − 9] over a 64-lane tile: one 64 + 3·6 = 82 wide
        // window per staged channel instead of 7 copies of 64
        let mut g = TensorGraph::new();
        let x = g.input(&[512, 2000]);
        let w = g.input(&[512, 512, 7]);
        let y = conv1d(&mut g, x, w, 3, 9);
        g.schedule_auto(&[y]);
        let src = &g.to_wgsl(&[y]).kernels[0].source;
        if std::env::var("TANG_DUMP").is_ok() {
            eprintln!("{src}");
        }
        let sizes: Vec<usize> = src
            .lines()
            .filter_map(|l| l.strip_prefix("var<workgroup> stage_"))
            .map(|l| {
                let n = l.rsplit("array<f32, ").next().unwrap();
                n.trim_end_matches(">;").parse().unwrap()
            })
            .collect();
        assert_eq!(sizes.len(), 2);
        assert!(sizes.iter().sum::<usize>() <= 4096);
        let chunk = sizes[1] / (64 * 7);
        assert!(chunk > 4, "chunk should grow past 4: {sizes:?}");
        assert_eq!(sizes[0], 82 * chunk, "x should be staged as one window");
    }

    #[test]
    fn oversized_staging_falls_back_untiled() {
        // r0 = c, inner k = 300: w[o, c, k] stages 16 rows × 300 taps per
        // chunk step, 4800 words over the 16 KiB limit even at chunk 1
        let mut g = TensorGraph::new();
        let x = g.input(&[600, 512]);
        let w = g.input(&[64, 600, 300]);
        let y = conv1d(&mut g, x, w, 1, 150);
        g.set_schedule(y, 0, (16, 16, 32), (1, 1));
        let k = &g.to_wgsl(&[y]).kernels[0];
        assert!(
            !k.source.contains("var<workgroup>"),
            "should not stage:\n{}",
            k.source
        );
        assert_eq!(k.workgroup_size, 256);
        let m = &g.to_msl(&[y]).kernels[0];
        assert_eq!(m.threadgroup_bytes, 0);

        // the same schedule with few taps still tiles, within the limit
        let mut g = TensorGraph::new();
        let x = g.input(&[600, 512]);
        let w = g.input(&[64, 600, 7]);
        let y = conv1d(&mut g, x, w, 1, 3);
        g.set_schedule(y, 0, (16, 16, 32), (1, 1));
        let m = &g.to_msl(&[y]).kernels[0];
        assert!(m.threadgroup_bytes > 0);
        assert!(m.threadgroup_bytes <= 4 * tile::MAX_STAGE_WORDS);
    }

    #[test]
    fn guarded_view_is_zero_padding() {
        let (ci, co, t, k, dil) = (3, 2, 7, 3, 2);
        let pad = dil * (k as i64 - 1) / 2;
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, k]);
        let y = conv1d(&mut g, x, w, dil, pad);
        let (xv, wv) = (ramp(ci * t, 0.3, -1.0), ramp(co * ci * k, 0.2, -0.4));
        let want = conv1d_ref(&xv, &wv, ci, co, t, k, dil, pad);
        close(&g.eval(&[xv.clone(), wv.clone()], &[y])[0], &want);

        // the same read unguarded is still a shape error
        let mut g2 = TensorGraph::new();
        let x2 = g2.input(&[ci, t]);
        let w2 = g2.input(&[co, ci, k]);
        let mut bad = View::pick(4, &[Some(2), Some(1)]);
        bad.forms[1].coefs[3] = dil;
        bad.forms[1].offset = -pad;
        let r = g2.comp(
            &[co, t],
            &[ci, k],
            vec![
                Arg {
                    src: x2,
                    view: bad.clone(),
                },
                Arg {
                    src: w2,
                    view: View::pick(4, &[Some(0), Some(2), Some(3)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        );
        assert!(r.is_err());
    }

    #[test]
    fn guarded_read_of_a_map_stays_materialized() {
        // y = conv(snake(x)): snake(0) ≠ 0 in general, so the map is not
        // spliced through the guarded read (it would pad with f(0))
        let (ci, co, t, k) = (2, 2, 6, 3);
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, k]);
        let shifted = g.map(&[x], |v| v[0] + ExprId::from_f64(1.0)).unwrap();
        let y = conv1d(&mut g, shifted, w, 1, 1);
        let (xv, wv) = (ramp(ci * t, 0.3, -1.0), ramp(co * ci * k, 0.2, -0.4));
        let xs: Vec<f64> = xv.iter().map(|v| v + 1.0).collect();
        let want = conv1d_ref(&xs, &wv, ci, co, t, k, 1, 1);
        let (fg, outs) = g.substitute_maps_with(&[y], false);
        assert_eq!(fg.count_comps(&outs), 2);
        close(&fg.eval(&[xv, wv], &outs)[0], &want);
    }

    #[test]
    fn guarded_reads_compose_through_spliced_maps() {
        // pad is a map with a guarded read; its consumer reads it plainly,
        // so it splices away and the guard follows the leaf
        let t = 5;
        let mut g = TensorGraph::new();
        let x = g.input(&[t]);
        let pv = View {
            gather: None,
            forms: vec![Form {
                coefs: vec![1],
                offset: -2,
            }],
            guard: true,
        };
        let padded = g
            .comp(&[t + 4], &[], vec![Arg { src: x, view: pv }], None, |v| {
                vec![v[0]]
            })
            .unwrap();
        let y = g
            .map(&[padded], |v| v[0] * v[0] + ExprId::from_f64(1.0))
            .unwrap();
        let xv = ramp(t, 0.3, -1.0);
        let want: Vec<f64> = (0..t + 4)
            .map(|i| {
                let v = if (2..t + 2).contains(&i) {
                    xv[i - 2]
                } else {
                    0.0
                };
                v * v + 1.0
            })
            .collect();
        let (fg, outs) = g.substitute_maps_with(&[y], false);
        assert_eq!(fg.count_comps(&outs), 1);
        close(&fg.eval(&[xv], &outs)[0], &want);
    }
}
