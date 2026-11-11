//! Decode as the finite difference of prefill.
//!
//! A model author writes one graph: the full-sequence forward over a
//! symbolic sequence dim `T` (prefill). [`derive_step`] derives from it
//! the step graph that appends `k` tokens to a sequence already `p0`
//! long — `k = 1` is decode, `k > 1` a draft-verify window — together
//! with the set of nodes whose old rows the step reads: the KV cache.
//!
//! **Modes.** Every `T`-axis of every node the step needs is either
//! *new* (extent `k`, relative coordinate `t'`, absolute position
//! `p0 + t'`) or *full* (extent `capacity`, absolute coordinate). Output
//! rows are new. A comp's point axes take the modes demanded of it, its
//! reduce dims over `T` are full, and each arg's `T`-axes take the mode
//! of the coordinate that reads them (views over `T` must be plain unit
//! reads). The body is copied unchanged: an [`Op::Iota`] read along a new
//! axis becomes `iota + p0`, so a causal mask written over coordinates
//! (`j > i`) becomes `j > p0 + t'` — which hides both the future and the
//! unfilled tail of a fixed-capacity cache.
//!
//! **The cache.** A node with one `T`-axis demanded full is not
//! recomputed: its new rows are computed and written into a cache input
//! of `capacity` rows at `p0..p0+k` (`p0` a one-element runtime input),
//! in the in-place form `ir_qwen`'s hand-written step uses. Those nodes
//! are the cache: for attention, the K and V nodes as the author wrote
//! them (derive on the graph as built to keep that cut; after
//! `substitute_maps` a position map may be inlined and the cache taken
//! before it).
//!
//! **Causality.** The derivation computes only new rows, which is right
//! iff growing `T` leaves old rows unchanged. Views are unit reads, so
//! the only way a row can see a later one is a reduction over `T`; each
//! such reduction's contribution must be `select(cond, x, identity)`,
//! `cond` affine in iota reads and changing sign exactly between
//! `j = i` and `j = i + 1`. The masked elements are then exactly the
//! monoid's identity whatever `x` is (NaN in an unfilled cache row
//! included). An additive mask (`s + (−1e30)` before `exp`) is refused:
//! `0·NaN` would leak.
//!
//! **Carried states** (`tensor_step_scan.rs`): a scan along `T` (monoid,
//! or real dense affine: linear attention, the gated delta rule, KDA)
//! caches its state at `p0 − 1`; a reduction over `T` into no `T` axis
//! caches its running monoid state. A cached row may not depend on one.
//!
//! **Options** ([`derive_step_with`]): [`Rows::Last`] computes the outputs'
//! last new row only; [`StepOptions::batch`] names a batch dim of
//! independent sessions, each with its own `p0`.
//!
//! **Rings** (`tensor_step_ring.rs`): a reduction masked to a window
//! `t − W < j ≤ t`, and a guarded read a bounded way back along `T` (a
//! causal conv), keep the last few positions instead of `capacity` rows.
//!
//! Not handled (an error, not a wrong graph): unguarded shifted or strided
//! reads along `T`, position tables indexed by `T` through a fixed axis (write the
//! lookup as a masked sum over the table, as `ir_qwen` does), gathers
//! and guards over `T`, reshapes that move `T` off the outer dim,
//! operator nodes, exclusive/reverse/log/max-plus/banded scans along `T`,
//! and outputs with several `T`-axes.

use std::collections::HashMap;

use tang::Scalar;
use tang_expr::{ExprGraph, ExprId, Node};

use super::{err, Arg, Comp, Dim, Form, Op, ShapeError, TensorGraph, TensorId, View};

/// A prefill graph with a batch dim of independent sessions in front.
#[path = "tensor_step_batch.rs"]
mod batch;
/// Carried states: scans along the sequence and reductions over it into no
/// sequence axis (the cache is the monoid / scan state).
#[path = "tensor_step_scan.rs"]
mod scan;
pub use batch::batch_graph;

#[path = "tensor_step_ring.rs"]
mod ring;
pub use ring::seed_cache;
#[cfg(test)]
#[path = "tensor_step_state_tests.rs"]
mod state_tests;
// derive-gather: gathers along the sequence (top-k attention)
#[path = "tensor_step_gather.rs"]
mod gather;
#[cfg(test)]
#[path = "tensor_step_gather_tests.rs"]
mod gather_tests;

/// What the step derivation is asked for.
#[derive(Clone, Copy, Debug)]
pub struct StepSpec {
    /// The symbolic sequence dim of the prefill graph.
    pub seq: &'static str,
    /// Tokens appended per step (`k`).
    pub new: usize,
    /// Cache capacity in positions (`p0 + k ≤ capacity`).
    pub capacity: usize,
}

/// Which rows of the outputs the step computes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rows {
    /// All `k` new rows (a draft verify needs every row's logits).
    #[default]
    All,
    /// Only the last new row: what comes after the window. Nodes that
    /// only feed it (the LM head, the last layer's MLP) run for one row.
    Last,
}

/// Options of [`derive_step_with`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StepOptions {
    pub rows: Rows,
    /// A symbolic batch dim of the prefill graph (independent sessions):
    /// the step's position input is then `[batch]`, one `p0` a session,
    /// read through each comp's batch coordinate. Sessions at different
    /// lengths step together.
    pub batch: Option<&'static str>,
}

/// Where a step-graph input comes from, in the step graph's input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepInput {
    /// `[1]` (or `[batch]`): `p0`, the number of positions already in
    /// the caches.
    Pos,
    /// Prefill input `i`, only the `k` new rows.
    New(usize),
    /// Prefill input `i` unchanged (it has no `T`-axis: weights).
    Same(usize),
    /// Cache slot `c` (an index into [`StepGraph::caches`]), `capacity` rows.
    Cache(usize),
}

/// One cached node.
#[derive(Clone, Debug)]
pub struct CacheSlot {
    /// The node of the prefill graph whose rows are cached.
    pub node: TensorId,
    /// Its `T`-axis.
    pub axis: usize,
    /// The step graph's cache input (`capacity` rows on `axis`).
    pub input: TensorId,
    /// The updated cache (rows `p0..p0+k` written): a step output, to be
    /// stored back into `input` (in place on a device).
    pub updated: TensorId,
    /// A carried state rather than rows: the monoid state of a reduction
    /// over the sequence (`node`'s own shape), or a scan's state at the
    /// last position (`node`'s shape without its scan dim `axis`). Empty
    /// sequence: the monoid identity (zero for an affine scan).
    pub state: bool,
    /// A ring cache (`tensor_step_ring.rs`): only the last `S` positions,
    /// row `s` holding position `p0 − S + s` (NaN before position 0),
    /// instead of `capacity` rows. `None`: rows by absolute position.
    pub ring: Option<usize>,
}

/// The derived step.
pub struct StepGraph {
    pub graph: TensorGraph,
    pub inputs: Vec<StepInput>,
    /// The prefill outputs' new rows, in the order asked for.
    pub outputs: Vec<TensorId>,
    pub caches: Vec<CacheSlot>,
    pub pos: TensorId,
    /// Prefill node → its step node holding the new rows (every derived
    /// node whose sequence axes are all new, and the sequence-free ones).
    pub rows: HashMap<TensorId, TensorId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Mode {
    New,
    Full,
    /// The last new row only (extent 1, position `p0 + k − 1`).
    Last,
    /// Ring caches: the last `S` old positions then the new ones (extent
    /// `S + k`, row `s` at position `p0 − S + s`); see `tensor_step_ring.rs`.
    Win(usize),
}

type Modes = Vec<Option<Mode>>;

/// Derive the `T → T + k` step of a causal prefill graph, as built or after
/// rewrite passes. `[T, T]` nodes (scores) are computed for the new rows
/// against every cache row; only single-`T` nodes are cached.
pub fn derive_step(
    g: &TensorGraph,
    outputs: &[TensorId],
    spec: &StepSpec,
) -> Result<StepGraph, ShapeError> {
    derive_step_with(g, outputs, spec, &StepOptions::default())
}

/// [`derive_step`] with [`StepOptions`]: last-row outputs, batched sessions.
pub fn derive_step_with(
    g: &TensorGraph,
    outputs: &[TensorId],
    spec: &StepSpec,
    opts: &StepOptions,
) -> Result<StepGraph, ShapeError> {
    // The derivation recurses once per node along the longest dependency
    // chain (GLM tiny overflowed a 2 MB test thread). It runs on its own
    // thread with a stack sized for any model: an explicit worklist would
    // repeat every op's demand rules (which modes each arg is read in) in
    // a second place.
    const STACK: usize = 1 << 30; // reserved, not committed, until used
    std::thread::scope(|sc| {
        std::thread::Builder::new()
            .name("derive_step".into())
            .stack_size(STACK)
            .spawn_scoped(sc, || derive_on_this_thread(g, outputs, spec, opts))
            .expect("spawn the derive_step thread")
            .join()
            .unwrap_or_else(|e| std::panic::resume_unwind(e))
    })
}

fn derive_on_this_thread(
    g: &TensorGraph,
    outputs: &[TensorId],
    spec: &StepSpec,
    opts: &StepOptions,
) -> Result<StepGraph, ShapeError> {
    // last-row outputs: a node whose new rows feed a cache anyway is
    // sliced from them rather than computed again for the last row
    let need_new = match opts.rows {
        Rows::All => None,
        Rows::Last => {
            let all = derive_on_this_thread(
                g,
                outputs,
                spec,
                &StepOptions {
                    rows: Rows::All,
                    ..*opts
                },
            )?;
            let mut need = vec![false; g.nodes.len()];
            let mut stack: Vec<usize> = all.caches.iter().map(|c| c.node.n()).collect();
            while let Some(i) = stack.pop() {
                if !std::mem::replace(&mut need[i], true) {
                    stack.extend(g.deps(i));
                }
            }
            Some(need)
        }
    };
    let mut d = Deriver {
        g,
        spec: *spec,
        opts: *opts,
        need_new,
        sg: TensorGraph::new(),
        inputs: Vec::new(),
        pos: TensorId(0),
        memo: HashMap::new(),
        inputs_memo: HashMap::new(),
        caches: Vec::new(),
        cache_of: HashMap::new(),
        ring_of: HashMap::new(),
    };
    d.pos = match opts.batch {
        Some(b) => d.sg.input(&[Dim::Sym(b)]),
        None => d.sg.input(&[1usize]),
    };
    d.inputs.push(StepInput::Pos);
    let mut outs = Vec::new();
    for &o in outputs {
        let tax = d.t_axes(o.n());
        if tax.len() > 1 {
            return err(format!(
                "step: output {o:?} has {} sequence axes",
                tax.len()
            ));
        }
        let mode = match opts.rows {
            Rows::All => Mode::New,
            Rows::Last => Mode::Last,
        };
        let modes = d.modes_with(o.n(), mode);
        outs.push(d.get(o.n(), modes)?);
    }
    // a cached row must not depend on a running total (a reduction over
    // the sequence into no sequence axis): every row reads the total over
    // all rows, which changes with every new one, so the old rows' cached
    // values would go stale. A scan's output is per row (row t reads the
    // state at t, final once computed), so rows built from it cache fine.
    let seq = Dim::Sym(spec.seq);
    for r in d.caches.iter().filter(|c| !c.state) {
        for st in d
            .caches
            .iter()
            .filter(|c| c.state && !g.shape(c.node).contains(&seq))
        {
            if g.depends_on(r.node.n(), st.node.n()) {
                return err(format!(
                    "step: cached rows of {:?} depend on the running state {:?}; \
                     old rows would change as the sequence grows",
                    r.node, st.node
                ));
            }
        }
    }
    let rows = d
        .memo
        .iter()
        .filter(|((_, m), _)| m.iter().all(|x| x.is_none() || *x == Some(Mode::New)))
        .map(|((n, _), &t)| (TensorId(*n as u32), t))
        .collect();
    Ok(StepGraph {
        graph: d.sg,
        inputs: d.inputs,
        outputs: outs,
        caches: d.caches,
        pos: d.pos,
        rows,
    })
}

struct Deriver<'a> {
    g: &'a TensorGraph,
    spec: StepSpec,
    opts: StepOptions,
    /// [`Rows::Last`]: the nodes whose new rows a cache needs.
    need_new: Option<Vec<bool>>,
    sg: TensorGraph,
    inputs: Vec<StepInput>,
    pos: TensorId,
    memo: HashMap<(usize, Modes), TensorId>,
    inputs_memo: HashMap<(usize, Modes), TensorId>,
    caches: Vec<CacheSlot>,
    cache_of: HashMap<usize, usize>,
    /// Ring caches by (node, `S`).
    ring_of: HashMap<(usize, usize), usize>,
}

impl Deriver<'_> {
    fn is_seq(&self, d: Dim) -> bool {
        d == Dim::Sym(self.spec.seq)
    }

    fn t_axes(&self, n: usize) -> Vec<usize> {
        let shape = &self.g.nodes[n].shape;
        (0..shape.len())
            .filter(|&a| self.is_seq(shape[a]))
            .collect()
    }

    /// Every `T`-axis of node `n` in mode `m`.
    fn modes_with(&self, n: usize, m: Mode) -> Modes {
        let shape = &self.g.nodes[n].shape;
        shape.iter().map(|&d| self.is_seq(d).then_some(m)).collect()
    }

    fn extent(&self, m: Mode) -> Dim {
        Dim::Fixed(match m {
            Mode::New => self.spec.new,
            Mode::Full => self.spec.capacity,
            Mode::Last => 1,
            Mode::Win(s) => s + self.spec.new,
        })
    }

    fn step_shape(&self, shape: &[Dim], modes: &Modes) -> Vec<Dim> {
        shape
            .iter()
            .zip(modes)
            .map(|(&d, m)| m.map_or(d, |m| self.extent(m)))
            .collect()
    }

    /// Node `n` with its `T`-axes in `modes`.
    fn get(&mut self, n: usize, modes: Modes) -> Result<TensorId, ShapeError> {
        if let Some(&t) = self.memo.get(&(n, modes.clone())) {
            return Ok(t);
        }
        let tax = self.t_axes(n);
        let full_single = tax.len() == 1 && modes[tax[0]] == Some(Mode::Full);
        let g = self.g;
        let node = &g.nodes[n];
        let last_of_new = tax.len() == 1
            && modes[tax[0]] == Some(Mode::Last)
            && !matches!(node.op, Op::Iota(_))
            && self.need_new.as_ref().is_some_and(|need| need[n]);
        if last_of_new {
            let new = self.get(n, self.modes_with(n, Mode::New))?;
            let t = self.last_row(new, tax[0], true)?;
            self.memo.insert((n, modes), t);
            return Ok(t);
        }
        let t = match &node.op {
            Op::Iota(dim) => self.iota(n, *dim, &modes)?,
            Op::Input(_) | Op::Comp(_) if full_single => self.cache(n, tax[0])?,
            // ring caches (tensor_step_ring.rs)
            Op::Input(_) | Op::Comp(_)
                if tax.len() == 1 && matches!(modes[tax[0]], Some(Mode::Win(_))) =>
            {
                let Some(Mode::Win(s)) = modes[tax[0]] else {
                    unreachable!()
                };
                self.ring(n, tax[0], s)?
            }
            // the last row of the new rows (one input, read once)
            Op::Input(_) if tax.len() == 1 && modes[tax[0]] == Some(Mode::Last) => {
                let new = self.get(n, self.modes_with(n, Mode::New))?;
                self.last_row(new, tax[0], true)?
            }
            Op::Input(i) => {
                let i = *i;
                if tax.len() > 1 {
                    return err(format!("step: input {i} has several sequence axes"));
                }
                let key = (i, modes.clone());
                if let Some(&t) = self.inputs_memo.get(&key) {
                    t
                } else {
                    let shape = self.step_shape(&node.shape, &modes);
                    let t = self.sg.input(&shape);
                    self.inputs.push(if tax.is_empty() {
                        StepInput::Same(i)
                    } else {
                        StepInput::New(i)
                    });
                    self.inputs_memo.insert(key, t);
                    t
                }
            }
            Op::Comp(c) => self.comp(n, c, &modes)?,
            Op::Reshape(src) => {
                let src = *src;
                let stax = self.t_axes(src.n());
                let smodes = match (tax.as_slice(), stax.as_slice()) {
                    ([], []) => vec![None; g.nodes[src.n()].shape.len()],
                    // the sequence at the same place on both sides behind the
                    // same outer dims (none, or a batch dim): row-major, its
                    // rows are the same bytes whatever the inner dims
                    ([a], [sa]) if a == sa && node.shape[..*a] == g.nodes[src.n()].shape[..*a] => {
                        let mut m = vec![None; g.nodes[src.n()].shape.len()];
                        m[*a] = modes[*a];
                        m
                    }
                    _ => return err("step: reshape that moves the sequence dim"),
                };
                let s = self.get(src.n(), smodes)?;
                let shape = self.step_shape(&node.shape, &modes);
                self.sg.reshape(s, &shape)?
            }
            Op::Scan(sc) => self.scan(n, sc, &modes)?,
            Op::Operator(_) => return err("step: operator nodes are not derived yet"),
        };
        self.memo.insert((n, modes), t);
        Ok(t)
    }

    /// An iota in the step: a new-mode axis counts from `p0`.
    fn iota(&mut self, n: usize, dim: usize, modes: &Modes) -> Result<TensorId, ShapeError> {
        let shape = self.step_shape(&self.g.nodes[n].shape, modes);
        let it = self.sg.iota(&shape, dim);
        let off = match modes[dim] {
            Some(Mode::New) => 0.0,
            Some(Mode::Last) => (self.spec.new - 1) as f64,
            // ring caches (tensor_step_ring.rs)
            Some(Mode::Win(s)) => return self.win_iota(it, &shape, s),
            _ => return Ok(it),
        };
        if self.opts.batch.is_some() {
            return err(
                "step: a position read outside a comp (batched sessions need its batch coordinate)",
            );
        }
        let r = shape.len();
        let args = vec![
            Arg {
                src: it,
                view: View::identity(r, r),
            },
            Arg {
                src: self.pos,
                view: View::pick(r, &[None]),
            },
        ];
        self.sg.comp(&shape, &[], args, None, move |v| {
            vec![v[0] + v[1] + ExprId::from_f64(off)]
        })
    }

    /// How a comp over `space` reads `p0`: the one position, or (batched)
    /// the position of the space's batch coordinate.
    fn pos_arg(&self, space: &[Dim]) -> Result<Arg, ShapeError> {
        let r = space.len();
        let view = match self.opts.batch {
            None => View::pick(r, &[None]),
            Some(b) => match space.iter().position(|&d| d == Dim::Sym(b)) {
                Some(bc) => View::pick(r, &[Some(bc)]),
                None => {
                    return err(format!(
                        "step: a position read in a space {space:?} without the batch dim `{b}`"
                    ))
                }
            },
        };
        Ok(Arg {
            src: self.pos,
            view,
        })
    }

    /// Row `k − 1` of `src` along `axis` (kept as an extent-1 axis, or
    /// dropped).
    fn last_row(&mut self, src: TensorId, axis: usize, keep: bool) -> Result<TensorId, ShapeError> {
        let shape = self.sg.shape(src).to_vec();
        let mut out = shape.clone();
        if keep {
            out[axis] = Dim::Fixed(1);
        } else {
            out.remove(axis);
        }
        let r = out.len();
        let last = (self.spec.new - 1) as i64;
        let forms = (0..shape.len())
            .map(|d| {
                if d == axis {
                    Form::constant(r, last)
                } else {
                    Form::unit(r, if keep || d < axis { d } else { d - 1 })
                }
            })
            .collect();
        let view = View {
            forms,
            guard: false,
            gather: None,
        };
        self.sg
            .comp(&out, &[], vec![Arg { src, view }], None, |v| vec![v[0]])
    }

    /// Node `n` (one `T`-axis, `axis`) read at old rows: its new rows
    /// written into a cache input at `p0..p0 + k`, in the form the decode
    /// lowerings run as an in-place update: `select(p0 ≤ j < p0 + k, new,
    /// old)`, where `new` is the one new row (`k = 1`) or a sum over the
    /// new rows with one live term (`Σ_t [t = j − p0]·new[t]`).
    fn cache(&mut self, n: usize, axis: usize) -> Result<TensorId, ShapeError> {
        if let Some(&c) = self.cache_of.get(&n) {
            return Ok(self.caches[c].updated);
        }
        let new = self.get(n, self.modes_with(n, Mode::New))?;
        let full = self.modes_with(n, Mode::Full);
        let shape = self.step_shape(&self.g.nodes[n].shape, &full);
        let slot = self.caches.len();
        let input = self.sg.input(&shape);
        self.inputs.push(StepInput::Cache(slot));
        let r = shape.len();
        let k = self.spec.new;
        let mut sp = shape.clone();
        sp.push(Dim::Fixed(k));
        let (pos_r, pos_r1) = (self.pos_arg(&shape)?, self.pos_arg(&sp)?);
        let row = if k == 1 {
            let mut v = View::identity(r, r);
            v.forms[axis] = Form::zero(r);
            Arg { src: new, view: v }
        } else {
            // space: cache coords ++ (t)
            let mut v = View::identity(r + 1, r);
            v.forms[axis] = Form::unit(r + 1, r);
            let (jt, tt) = (self.sg.iota(&sp, axis), self.sg.iota(&sp, r));
            let gathered = self.sg.comp(
                &shape,
                &[Dim::Fixed(k)],
                vec![
                    Arg { src: new, view: v },
                    Arg {
                        src: jt,
                        view: View::identity(r + 1, r + 1),
                    },
                    Arg {
                        src: tt,
                        view: View::identity(r + 1, r + 1),
                    },
                    pos_r1,
                ],
                Some(super::Monoid::sum()),
                // new row t lands at cache row j = p0 + t
                |v| vec![pick(v[2], v[1] - v[3], v[0])],
            )?;
            Arg {
                src: gathered,
                view: View::identity(r, r),
            }
        };
        let iota = self.sg.iota(&shape, axis);
        let args = vec![
            row,
            Arg {
                src: input,
                view: View::identity(r, r),
            },
            Arg {
                src: iota,
                view: View::identity(r, r),
            },
            pos_r,
        ];
        let kf = k as f64;
        let updated = self.sg.comp(&shape, &[], args, None, |v| {
            let (new, old, j, p0) = (v[0], v[1], v[2], v[3]);
            let half = ExprId::from_f64(0.5);
            let below = p0 + ExprId::from_f64(kf) - j - half; // j < p0 + k
            let above = j - p0 + half; // j >= p0
            vec![tang_expr::with_graph(|eg| {
                let inner = eg.select(above, new, old);
                eg.select(below, inner, old)
            })]
        })?;
        self.caches.push(CacheSlot {
            node: TensorId(n as u32),
            axis,
            input,
            updated,
            state: false,
            ring: None,
        });
        self.cache_of.insert(n, slot);
        Ok(updated)
    }

    fn comp(&mut self, n: usize, c: &Comp, modes: &Modes) -> Result<TensorId, ShapeError> {
        let pts = c.points().len();
        // the mode of every iteration coordinate (None: not over T)
        let mut coord: Modes = modes[..pts].to_vec();
        let running =
            !modes[..pts].iter().any(|m| m.is_some()) && c.red.iter().any(|&d| self.is_seq(d));
        // a reduction over the sequence into no sequence axis is a running
        // state: the step reduces the new rows only and combines them into
        // the cached total
        let red_mode = if running { Mode::New } else { Mode::Full };
        coord.extend(c.red.iter().map(|&d| self.is_seq(d).then_some(red_mode)));
        let seq_red: Vec<usize> = (pts..coord.len()).filter(|&d| coord[d].is_some()).collect();
        if !seq_red.is_empty() && !running {
            let seq_pts: Vec<usize> = (0..pts).filter(|&d| coord[d].is_some()).collect();
            for &j in &seq_red {
                if !seq_pts.iter().any(|&i| self.causal_mask(c, i, j)) {
                    return err(format!(
                        "step: node {n} reduces over the sequence without a causal mask; \
                         its old rows would change as the sequence grows"
                    ));
                }
            }
            // ring caches (tensor_step_ring.rs): window-masked reduce coords
            self.windowed(c, &mut coord, pts);
        }
        let mut args = Vec::with_capacity(c.args.len());
        for (ai, a) in c.args.iter().enumerate() {
            let src_shape = self.g.nodes[a.src.n()].shape.clone();
            let src_seq = src_shape.iter().any(|&d| self.is_seq(d));
            let touches_seq = |f: &Form| {
                f.coefs
                    .iter()
                    .enumerate()
                    .any(|(d, &k)| k != 0 && coord[d].is_some())
            };
            // derive-gather: a gather over the sequence reads the source's cache
            // when its index is provably causal (tensor_step_gather.rs)
            if a.view.is_gather() && (src_seq || a.view.forms.iter().any(touches_seq)) {
                let smodes = self.seq_gather(n, ai, c, &coord)?;
                let src = self.get(a.src.n(), smodes)?;
                args.push(Arg {
                    src,
                    view: a.view.clone(),
                });
                continue;
            }
            if a.view.guard && src_seq {
                // ring caches (tensor_step_ring.rs): a read a bounded way back
                let Some((smodes, view)) = self.shifted(c, a, &coord) else {
                    return err(format!(
                        "step: node {n} arg {ai} is a guarded read over the sequence"
                    ));
                };
                let src = self.get(a.src.n(), smodes)?;
                args.push(Arg { src, view });
                continue;
            }
            let mut smodes = Modes::with_capacity(src_shape.len());
            for (sa, f) in a.view.forms.iter().enumerate() {
                if self.is_seq(src_shape[sa]) {
                    match f.as_unit().and_then(|d| coord[d]) {
                        Some(m) => smodes.push(Some(m)),
                        None => {
                            return err(format!(
                                "step: node {n} arg {ai} reads sequence axis {sa} through a \
                                 shifted or non-sequence coordinate"
                            ))
                        }
                    }
                } else {
                    if touches_seq(f) {
                        return err(format!(
                            "step: node {n} arg {ai} indexes a fixed axis by a sequence \
                             coordinate (a position table)"
                        ));
                    }
                    smodes.push(None);
                }
            }
            // batched: a position iota becomes a map over this comp's space,
            // where the batch coordinate picks the session's `p0`
            if let (Some(_), Op::Iota(dim)) = (self.opts.batch, &self.g.nodes[a.src.n()].op) {
                let off = match smodes[*dim] {
                    Some(Mode::New) => Some(0.0),
                    Some(Mode::Last) => Some((self.spec.new - 1) as f64),
                    // ring caches (tensor_step_ring.rs)
                    Some(Mode::Win(s)) => Some(-(s as f64)),
                    _ => None,
                };
                let win = matches!(smodes[*dim], Some(Mode::Win(_)));
                if let Some(off) = off {
                    let ishape = self.step_shape(&src_shape, &smodes);
                    let it = self.sg.iota(&ishape, *dim);
                    let space: Vec<Dim> = c
                        .space()
                        .iter()
                        .zip(&coord)
                        .map(|(&d, m)| m.map_or(d, |m| self.extent(m)))
                        .collect();
                    let r = space.len();
                    let pos = self.pos_arg(&space)?;
                    let map = self.sg.comp(
                        &space,
                        &[],
                        vec![
                            Arg {
                                src: it,
                                view: a.view.clone(),
                            },
                            pos,
                        ],
                        None,
                        move |v| {
                            let p = v[0] + v[1] + ExprId::from_f64(off);
                            vec![if win { ring::before_start(p) } else { p }]
                        },
                    )?;
                    args.push(Arg {
                        src: map,
                        view: View::identity(r, r),
                    });
                    continue;
                }
            }
            let src = self.get(a.src.n(), smodes)?;
            args.push(Arg {
                src,
                view: a.view.clone(),
            });
        }
        let out: Vec<Dim> = c
            .out
            .iter()
            .enumerate()
            .map(|(d, &x)| match modes.get(d).copied().flatten() {
                Some(m) if d < pts => self.extent(m),
                _ => x,
            })
            .collect();
        let red: Vec<Dim> = c
            .red
            .iter()
            .enumerate()
            .map(|(r, &d)| match coord[pts + r] {
                Some(m) => self.extent(m),
                None => d,
            })
            .collect();
        let t = self.sg.push_comp(Comp {
            out: out.clone(),
            tuple: c.tuple,
            red,
            args,
            body: c.body.clone(),
            body_outs: c.body_outs.clone(),
            monoid: c.monoid.clone(),
        })?;
        if !running {
            return Ok(t);
        }
        // total = cached total ⊕ the new rows' partial
        let m = c.monoid.clone().expect("a reduction has a monoid");
        let input = self.sg.input(&out);
        self.inputs.push(StepInput::Cache(self.caches.len()));
        let picks: Vec<Option<usize>> = (0..pts).map(Some).collect();
        let total = self.combine(&m, &out, input, &picks, t)?;
        self.state_slot(n, 0, input, total);
        Ok(total)
    }

    /// Whether every value `c`'s body contributes is `select(cond, x, id)`
    /// with `id` the monoid's identity and `cond` affine in iota reads,
    /// flipping sign exactly between reduce coord `j = i` and `j = i + 1`
    /// (point coord `i`): a causal mask whose masked elements are exactly
    /// the identity, whatever `x` is there (an unfilled cache row may hold
    /// NaN). A body that is a read of a materialized map is looked through.
    /// For a max-like slot (identity −∞ or ≤ −1e30) any literal ≤ −1e30
    /// counts: it is absorbed by the unmasked diagonal.
    fn causal_mask(&self, c: &Comp, i: usize, j: usize) -> bool {
        let Some(m) = &c.monoid else {
            return false;
        };
        let args: Vec<(TensorId, View)> = c.args.iter().map(|a| (a.src, a.view.clone())).collect();
        c.body_outs
            .iter()
            .enumerate()
            .all(|(k, &out)| self.masked(&c.body, &args, c.rank(), out, m.identity[k], i, j))
    }

    #[allow(clippy::too_many_arguments)]
    fn masked(
        &self,
        body: &ExprGraph,
        args: &[(TensorId, View)],
        rank: usize,
        e: ExprId,
        id: f64,
        i: usize,
        j: usize,
    ) -> bool {
        let nodes = body.nodes_slice();
        match nodes[e.index() as usize] {
            Node::Select(cond, _, f) => {
                let Some(l) = nodes[f.index() as usize].as_f64() else {
                    return false;
                };
                let identity = l == id || (id <= -1e30 && l <= -1e30);
                let Some((coefs, konst)) = self.affine(body, args, rank, cond) else {
                    return false;
                };
                let only_ij = coefs
                    .iter()
                    .enumerate()
                    .all(|(d, &k)| k == 0.0 || d == i || d == j);
                // value at (i, j = i + δ) is coefs[j]·δ + konst; unmasked
                // (cond > 0) at δ = 0, masked at δ = 1
                let at = |delta: f64| coefs[j] * delta + konst > 0.0;
                identity
                    && only_ij
                    && coefs[j] != 0.0
                    && coefs[i] == -coefs[j]
                    && at(0.0)
                    && !at(1.0)
            }
            Node::Var(v) => {
                let Some((src, view)) = args.get(v as usize) else {
                    return false;
                };
                let Op::Comp(m) = &self.g.nodes[src.n()].op else {
                    return false;
                };
                if !m.red.is_empty() || m.tuple != 0 || view.is_gather() || m.has_gather() {
                    return false;
                }
                let inner: Vec<(TensorId, View)> = m
                    .args
                    .iter()
                    .map(|a| (a.src, a.view.compose(view, rank)))
                    .collect();
                self.masked(&m.body, &inner, rank, m.body_outs[0], id, i, j)
            }
            _ => false,
        }
    }

    /// `e` of `body` (vars: `args`, views over a `rank`-dim space) as
    /// `Σ coefs[d]·x_d + konst`, when it is built from literals and iota
    /// reads with `+`, `−` and scaling.
    fn affine(
        &self,
        body: &ExprGraph,
        args: &[(TensorId, View)],
        rank: usize,
        e: ExprId,
    ) -> Option<(Vec<f64>, f64)> {
        let nodes = body.nodes_slice();
        let lit = |e: ExprId| nodes[e.index() as usize].as_f64();
        let rec = |e: ExprId| self.affine(body, args, rank, e);
        match nodes[e.index() as usize] {
            Node::Lit(b) => Some((vec![0.0; rank], f64::from_bits(b))),
            Node::Var(v) => {
                let (src, view) = args.get(v as usize)?;
                match self.g.nodes[src.n()].op {
                    Op::Iota(dim) if !view.is_gather() => {
                        let f = &view.forms[dim];
                        Some((f.coefs.iter().map(|&k| k as f64).collect(), f.offset as f64))
                    }
                    _ => None,
                }
            }
            Node::Add(x, y) => {
                let (mut a, ka) = rec(x)?;
                let (b, kb) = rec(y)?;
                a.iter_mut().zip(&b).for_each(|(p, q)| *p += q);
                Some((a, ka + kb))
            }
            Node::Neg(x) => {
                let (a, k) = rec(x)?;
                Some((a.iter().map(|v| -v).collect(), -k))
            }
            Node::Mul(x, y) => {
                let (s, e) = match (lit(x), lit(y)) {
                    (Some(s), _) => (s, y),
                    (_, Some(s)) => (s, x),
                    _ => return None,
                };
                let (a, k) = rec(e)?;
                Some((a.iter().map(|v| v * s).collect(), k * s))
            }
            _ => None,
        }
    }
}

/// `x` where the integer-valued `a == b`, else 0 (two half-planes; the
/// form `ir_qwen`'s gathers use, which the decode lowerings run as a
/// one-iteration loop).
fn pick(a: ExprId, b: ExprId, x: ExprId) -> ExprId {
    let half = ExprId::from_f64(0.5);
    let (lo, hi) = (a - b + half, b - a + half);
    tang_expr::with_graph(|eg| {
        let z = eg.lit(0.0);
        let inner = eg.select(hi, x, z);
        eg.select(lo, inner, z)
    })
}

#[cfg(test)]
mod tests {
    use super::super::{Monoid, ScanKind};
    use super::*;

    const T: Dim = Dim::Sym("T");

    /// Deterministic values in about [-scale, scale].
    fn rand(n: usize, seed: u64, scale: f64) -> Vec<f64> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn lit(v: f64) -> ExprId {
        ExprId::from_f64(v)
    }

    /// `[T, T]` additive causal mask: 0 where `j <= i`, −1e30 above.
    fn causal_mask(g: &mut TensorGraph) -> TensorId {
        let i = g.iota(&[T, T], 0);
        let j = g.iota(&[T, T], 1);
        g.map(&[i, j], |v| {
            let above = v[1] - v[0];
            tang_expr::with_graph(|eg| {
                let neg = eg.lit(-1e30);
                eg.select(above, neg, ExprId::ZERO)
            })
        })
        .unwrap()
    }

    /// A rotary-like position map on `[T, d]`: `x · cos(0.3·t·(c+1))`.
    fn position(g: &mut TensorGraph, x: TensorId, d: usize) -> TensorId {
        let t = g.iota(&[T, Dim::Fixed(d)], 0);
        let c = g.iota(&[T, Dim::Fixed(d)], 1);
        g.map(&[x, t, c], |v| {
            v[0] * (lit(0.3) * v[1] * (v[2] + lit(1.0))).cos()
        })
        .unwrap()
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Mask {
        /// No mask: non-causal attention.
        None,
        /// `s + select(j > i, −1e30, 0)` before max/exp: the textbook form.
        Additive,
        /// Every reduction over keys contributes `select(i − j + ½, x, id)`.
        Select,
    }

    /// Single-head attention `x → softmax(q·kᵀ/√d)·v · Wo`, causal per `mask`.
    fn attention(g: &mut TensorGraph, x: TensorId, dm: usize, dh: usize, mask: Mask) -> TensorId {
        let wq = g.input(&[dm, dh]);
        let wk = g.input(&[dm, dh]);
        let wv = g.input(&[dm, dh]);
        let wo = g.input(&[dh, dm]);
        let q = g.matmul(x, wq).unwrap();
        let k = g.matmul(x, wk).unwrap();
        let v = g.matmul(x, wv).unwrap();
        let q = position(g, q, dh);
        let k = position(g, k, dh);
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap();
        let scale = 1.0 / (dh as f64).sqrt();
        if mask == Mask::Select {
            let o = select_softmax_v(g, s, v, dh, scale);
            return g.matmul(o, wo).unwrap();
        }
        let mut s = g.map(&[s], |v| v[0] * lit(scale)).unwrap();
        if mask == Mask::Additive {
            let mask = causal_mask(g);
            s = g.map(&[s, mask], |v| v[0] + v[1]).unwrap();
        }
        let m = g.reduce(s, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[T, T]).unwrap();
        let p = g.map(&[s, mb], |v| (v[0] - v[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let o = g.matmul(p, v).unwrap();
        let lb = g.broadcast(l, &[T, Dim::Fixed(dh)]).unwrap();
        let o = g.map(&[o, lb], |v| v[0] / v[1]).unwrap();
        g.matmul(o, wo).unwrap()
    }

    /// `softmax(s·scale)·v` with every reduction over keys `j` masked as
    /// `select(t − j + ½, x, identity)`.
    fn select_softmax_v(
        g: &mut TensorGraph,
        s: TensorId,
        v: TensorId,
        dh: usize,
        scale: f64,
    ) -> TensorId {
        let a = |src, picks: &[Option<usize>], r| Arg {
            src,
            view: View::pick(r, picks),
        };
        let (it, ij) = (g.iota(&[T, T], 0), g.iota(&[T, T], 1));
        let vis = |t: ExprId, j: ExprId| t - j + lit(0.5);
        let sel = |c, x, f| tang_expr::with_graph(|eg| eg.select(c, x, f));
        let m = g
            .comp(
                &[T],
                &[T],
                vec![
                    a(s, &[Some(0), Some(1)], 2),
                    a(it, &[Some(0), Some(1)], 2),
                    a(ij, &[Some(0), Some(1)], 2),
                ],
                Some(Monoid::max()),
                |v| vec![sel(vis(v[1], v[2]), v[0] * lit(scale), lit(-1e30))],
            )
            .unwrap();
        let l = g
            .comp(
                &[T],
                &[T],
                vec![
                    a(s, &[Some(0), Some(1)], 2),
                    a(m, &[Some(0)], 2),
                    a(it, &[Some(0), Some(1)], 2),
                    a(ij, &[Some(0), Some(1)], 2),
                ],
                Some(Monoid::sum()),
                |v| {
                    vec![sel(
                        vis(v[2], v[3]),
                        (v[0] * lit(scale) - v[1]).exp(),
                        lit(0.0),
                    )]
                },
            )
            .unwrap();
        let sp = [T, Dim::Fixed(dh), T];
        let (it3, ij3) = (g.iota(&sp, 0), g.iota(&sp, 2));
        g.comp(
            &[T, Dim::Fixed(dh)],
            &[T],
            vec![
                a(s, &[Some(0), Some(2)], 3),
                a(m, &[Some(0)], 3),
                a(l, &[Some(0)], 3),
                a(v, &[Some(2), Some(1)], 3),
                a(it3, &[Some(0), Some(1), Some(2)], 3),
                a(ij3, &[Some(0), Some(1), Some(2)], 3),
            ],
            Some(Monoid::sum()),
            |v| {
                let pv = (v[0] * lit(scale) - v[1]).exp() * v[3] / v[2];
                vec![sel(vis(v[4], v[5]), pv, lit(0.0))]
            },
        )
        .unwrap()
    }

    fn rmsnorm(g: &mut TensorGraph, x: TensorId, dm: usize) -> TensorId {
        let w = g.input(&[1, dm]);
        let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
        let ms = g.reduce(sq, 1, Monoid::sum()).unwrap();
        let msb = g.broadcast(ms, &[T, Dim::Fixed(dm)]).unwrap();
        let wb = g.broadcast(w, &[T, Dim::Fixed(dm)]).unwrap();
        let inv = 1.0 / dm as f64;
        g.map(&[x, msb, wb], |v| {
            v[0] * (v[1] * lit(inv) + lit(1e-6)).sqrt().recip() * v[2]
        })
        .unwrap()
    }

    fn mlp(g: &mut TensorGraph, x: TensorId, dm: usize, df: usize) -> TensorId {
        let w1 = g.input(&[dm, df]);
        let w2 = g.input(&[df, dm]);
        let h = g.matmul(x, w1).unwrap();
        let a = g.map(&[h], |v| v[0] / (lit(1.0) + (-v[0]).exp())).unwrap();
        g.matmul(a, w2).unwrap()
    }

    /// `layers` pre-norm blocks: `x += attn(norm(x)); x += mlp(norm(x))`.
    /// With `layers == 0`, one bare attention layer.
    fn model(layers: usize, mask: Mask) -> (TensorGraph, TensorId) {
        let (dm, dh, df) = (6, 4, 8);
        let mut g = TensorGraph::new();
        let x = g.input(&[T, Dim::Fixed(dm)]);
        if layers == 0 {
            let y = attention(&mut g, x, dm, dh, mask);
            return (g, y);
        }
        let mut h = x;
        for _ in 0..layers {
            let n = rmsnorm(&mut g, h, dm);
            let a = attention(&mut g, n, dm, dh, mask);
            h = g.map(&[h, a], |v| v[0] + v[1]).unwrap();
            let n = rmsnorm(&mut g, h, dm);
            let f = mlp(&mut g, n, dm, df);
            h = g.map(&[h, f], |v| v[0] + v[1]).unwrap();
        }
        (g, h)
    }

    /// Input values for a sequence of `tmax` rows (rows of `T`-inputs are
    /// a prefix of these).
    fn data(g: &TensorGraph, tmax: usize) -> Vec<(Vec<Dim>, Vec<f64>)> {
        let mut ins: Vec<(usize, Vec<Dim>)> = g
            .nodes()
            .iter()
            .filter_map(|nd| match nd.op {
                Op::Input(i) => Some((i, nd.shape.clone())),
                _ => None,
            })
            .collect();
        ins.sort_by_key(|(i, _)| *i);
        assert!(ins.iter().enumerate().all(|(k, (i, _))| k == *i));
        ins.into_iter()
            .map(|(i, shape)| {
                let n: usize = shape
                    .iter()
                    .map(|&d| if d == T { tmax } else { d.n() })
                    .product();
                (shape, rand(n, 17 + i as u64, 0.8))
            })
            .collect()
    }

    /// Rows `r0..r1` along axis `axis` of a row-major `shape` buffer.
    fn rows(buf: &[f64], shape: &[usize], axis: usize, r0: usize, r1: usize) -> Vec<f64> {
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out = Vec::new();
        for o in 0..outer {
            for r in r0..r1 {
                let base = (o * shape[axis] + r) * inner;
                out.extend_from_slice(&buf[base..base + inner]);
            }
        }
        out
    }

    fn concrete(shape: &[Dim], t: usize) -> Vec<usize> {
        shape
            .iter()
            .map(|&d| if d == T { t } else { d.n() })
            .collect()
    }

    fn prefill_inputs(data: &[(Vec<Dim>, Vec<f64>)], tmax: usize, t: usize) -> Vec<Vec<f64>> {
        data.iter()
            .map(|(shape, v)| match shape.iter().position(|&d| d == T) {
                Some(a) => rows(v, &concrete(shape, tmax), a, 0, t),
                None => v.clone(),
            })
            .collect()
    }

    /// Worst `|a − b| / (1 + |b|)`; a NaN or infinity anywhere is ∞ (a
    /// plain `f64::max` fold would skip NaN).
    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let e = (x - y).abs() / (1.0 + y.abs());
                if e.is_finite() {
                    e
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0, f64::max)
    }

    /// Prefill to `p0`, then run derived steps of the given sizes, each
    /// checked against a prefill of the grown sequence: the new output
    /// rows, and every cache's filled rows. Returns the worst error.
    fn check_steps(g: &TensorGraph, outs: &[TensorId], p0: usize, ks: &[usize]) -> f64 {
        check_steps_with(g, outs, p0, ks, (0, 0), Rows::All)
    }

    fn check_steps_skewed(
        g: &TensorGraph,
        outs: &[TensorId],
        p0: usize,
        ks: &[usize],
        skew: usize,
    ) -> f64 {
        check_steps_with(g, outs, p0, ks, (skew, 0), Rows::All)
    }

    /// A carried state's value in a prefill of `t` rows: a scan's row
    /// `t − 1`, or a reduction's whole value.
    fn state_of(g: &TensorGraph, c: &CacheSlot, v: &[f64], t: usize) -> Vec<f64> {
        if g.shape(c.node).contains(&T) {
            rows(v, &concrete(g.shape(c.node), t), c.axis, t - 1, t)
        } else {
            v.to_vec()
        }
    }

    /// [`check_steps`] feeding the step `p0 + skew.0` as its position and
    /// carried states from a prefill of `p0 − skew.1` rows (negative
    /// controls when nonzero), and outputs for `rows`.
    pub(super) fn check_steps_with(
        g: &TensorGraph,
        outs: &[TensorId],
        p0: usize,
        ks: &[usize],
        skew: (usize, usize),
        rows_of: Rows,
    ) -> f64 {
        let cap = 16;
        let tmax = p0 + ks.iter().sum::<usize>();
        let data = data(g, tmax);
        let mut worst = 0.0f64;
        let mut pos = p0;
        // cache buffers keyed by prefill node, `cap` rows; unfilled rows hold
        // NaN, which the derived masks must keep out
        let mut caches: HashMap<TensorId, Vec<f64>> = HashMap::new();
        for &k in ks {
            let spec = StepSpec {
                seq: "T",
                new: k,
                capacity: cap,
            };
            let opts = StepOptions {
                rows: rows_of,
                batch: None,
            };
            let step = derive_step_with(g, outs, &spec, &opts).unwrap();
            let cache_nodes: Vec<TensorId> = step.caches.iter().map(|c| c.node).collect();
            if caches.is_empty() {
                let pg = g.bind(&[("T", pos)]).unwrap();
                let vals = pg.eval(&prefill_inputs(&data, tmax, pos), &cache_nodes);
                let sp = pos - skew.1;
                let svals = g
                    .bind(&[("T", sp)])
                    .unwrap()
                    .eval(&prefill_inputs(&data, tmax, sp), &cache_nodes);
                for ((c, v), sv) in step.caches.iter().zip(vals).zip(svals) {
                    if c.state {
                        caches.insert(c.node, state_of(g, c, &sv, sp));
                        continue;
                    }
                    let shape = concrete(g.shape(c.node), pos);
                    let cshape = concrete(g.shape(c.node), cap);
                    let mut buf = vec![f64::NAN; cshape.iter().product()];
                    // scatter rows 0..pos into the capacity buffer
                    let inner: usize = shape[c.axis + 1..].iter().product();
                    let outer: usize = shape[..c.axis].iter().product();
                    for o in 0..outer {
                        for r in 0..pos {
                            let src = (o * pos + r) * inner;
                            let dst = (o * cap + r) * inner;
                            buf[dst..dst + inner].copy_from_slice(&v[src..src + inner]);
                        }
                    }
                    caches.insert(c.node, buf);
                }
            }
            let inputs: Vec<Vec<f64>> = step
                .inputs
                .iter()
                .map(|si| match *si {
                    StepInput::Pos => vec![(pos + skew.0) as f64],
                    StepInput::Same(i) => data[i].1.clone(),
                    StepInput::New(i) => {
                        let (shape, v) = &data[i];
                        let a = shape.iter().position(|&d| d == T).unwrap();
                        rows(v, &concrete(shape, tmax), a, pos, pos + k)
                    }
                    StepInput::Cache(c) => caches[&step.caches[c].node].clone(),
                })
                .collect();
            let mut want_ids: Vec<TensorId> = step.outputs.clone();
            want_ids.extend(step.caches.iter().map(|c| c.updated));
            let got = step.graph.eval(&inputs, &want_ids);

            let grown = pos + k;
            let fg = g.bind(&[("T", grown)]).unwrap();
            let mut ref_ids = outs.to_vec();
            ref_ids.extend(&cache_nodes);
            let want = fg.eval(&prefill_inputs(&data, tmax, grown), &ref_ids);
            for (o, (gv, wv)) in outs.iter().zip(got.iter().zip(&want)) {
                let shape = concrete(g.shape(*o), grown);
                let e = match g.shape(*o).iter().position(|&d| d == T) {
                    Some(a) => {
                        let r0 = if rows_of == Rows::Last {
                            grown - 1
                        } else {
                            pos
                        };
                        max_err(gv, &rows(wv, &shape, a, r0, grown))
                    }
                    // a running total: the whole value
                    None => max_err(gv, wv),
                };
                worst = worst.max(e);
            }
            for (ci, c) in step.caches.iter().enumerate() {
                let gv = &got[outs.len() + ci];
                if c.state {
                    let e = max_err(gv, &state_of(g, c, &want[outs.len() + ci], grown));
                    worst = worst.max(e);
                    caches.insert(c.node, gv.clone());
                    continue;
                }
                let cshape = concrete(g.shape(c.node), cap);
                let wshape = concrete(g.shape(c.node), grown);
                let wv = &want[outs.len() + ci];
                let e = max_err(
                    &rows(gv, &cshape, c.axis, 0, grown),
                    &rows(wv, &wshape, c.axis, 0, grown),
                );
                worst = worst.max(e);
                caches.insert(c.node, gv.clone());
            }
            pos = grown;
        }
        worst
    }

    fn describe(g: &TensorGraph, step: &StepGraph) -> Vec<String> {
        step.caches
            .iter()
            .map(|c| {
                let what = match &g.nodes()[c.node.n()].op {
                    Op::Comp(cm) if !cm.red.is_empty() => format!("reduction over {:?}", cm.red),
                    Op::Comp(cm) => {
                        let sin = cm
                            .body
                            .nodes_slice()
                            .iter()
                            .any(|n| matches!(n, Node::Sin(_)));
                        format!(
                            "map{}",
                            if sin {
                                " with sin/cos (position map)"
                            } else {
                                ""
                            }
                        )
                    }
                    Op::Input(i) => format!("input {i}"),
                    _ => "other".into(),
                };
                format!("{:?} {:?} ({what})", c.node, g.shape(c.node))
            })
            .collect()
    }

    /// How the graph is prepared before deriving.
    #[derive(Clone, Copy, Debug)]
    enum Prep {
        /// As built.
        Raw,
        /// After `substitute_maps`.
        Fused,
        /// After `substitute_maps` and `stabilize_softmax`.
        Online,
    }

    fn prepared(layers: usize, mask: Mask, prep: Prep) -> (TensorGraph, Vec<TensorId>) {
        let (g, y) = model(layers, mask);
        match prep {
            Prep::Raw => (g, vec![y]),
            Prep::Fused => g.substitute_maps(&[y]),
            Prep::Online => {
                let (g, outs) = g.substitute_maps(&[y]);
                g.stabilize_softmax(&outs)
            }
        }
    }

    fn spec(k: usize) -> StepSpec {
        StepSpec {
            seq: "T",
            new: k,
            capacity: 16,
        }
    }

    #[test]
    fn attention_step_matches_prefill() {
        for prep in [Prep::Raw, Prep::Fused, Prep::Online] {
            let (g, outs) = prepared(0, Mask::Select, prep);
            let step = derive_step(&g, &outs, &spec(1)).unwrap();
            let caches = describe(&g, &step);
            eprintln!("attention ({prep:?}) caches: {caches:#?}");
            // K and V, [T, dh]: nothing else is read at old rows
            assert_eq!(caches.len(), 2, "{caches:?}");
            // (as built, K is cached as the author's transposed `kᵀ`)
            assert!(step.caches.iter().all(|c| {
                let sh = g.shape(c.node);
                sh.len() == 2 && sh.contains(&T) && sh.contains(&Dim::Fixed(4))
            }));
            for k in [1usize, 4] {
                let e = check_steps(&g, &outs, 5, &[k]);
                eprintln!("attention {prep:?} k={k}: max rel err {e:.2e}");
                assert!(e < 1e-12, "k={k}: {e}");
            }
        }
    }

    #[test]
    fn chained_steps_feed_the_cache_back() {
        // prefill 3, verify a 4-token window, then decode one at a time
        let (g, outs) = prepared(0, Mask::Select, Prep::Raw);
        let e = check_steps(&g, &outs, 3, &[4, 1, 1, 2]);
        eprintln!("chained: max rel err {e:.2e}");
        assert!(e < 1e-12, "{e}");
    }

    #[test]
    fn two_layer_stack_caches_k_and_v_per_layer() {
        for prep in [Prep::Raw, Prep::Fused, Prep::Online] {
            let (g, outs) = prepared(2, Mask::Select, prep);
            let step = derive_step(&g, &outs, &spec(1)).unwrap();
            let caches = describe(&g, &step);
            eprintln!("2 layers ({prep:?}) caches: {caches:#?}");
            assert_eq!(caches.len(), 4, "{caches:?}");
            for ks in [&[1usize][..], &[4], &[4, 1, 1]] {
                let e = check_steps(&g, &outs, 5, ks);
                eprintln!("2 layers {prep:?} ks={ks:?}: max rel err {e:.2e}");
                assert!(e < 1e-12, "{ks:?}: {e}");
            }
        }
    }

    /// The comparison can fail: a step told the wrong position writes the
    /// cache rows one off and masks one key too many.
    #[test]
    fn wrong_position_is_caught() {
        let (g, outs) = prepared(0, Mask::Select, Prep::Raw);
        let e = check_steps_skewed(&g, &outs, 5, &[1], 1);
        eprintln!("skewed position: max rel err {e:.2e}");
        assert!(e > 1e-3, "{e}");
    }

    /// Without a mask, or with an additive one (whose `0·NaN` would read
    /// an unfilled cache row), there is no step.
    #[test]
    fn unmasked_and_additive_attention_have_no_step() {
        for mask in [Mask::None, Mask::Additive] {
            for prep in [Prep::Raw, Prep::Fused] {
                let (g, outs) = prepared(0, mask, prep);
                let r = derive_step(&g, &outs, &spec(1));
                let msg = r.err().expect("must be refused").0;
                assert!(msg.contains("causal"), "{mask:?} {prep:?}: {msg}");
            }
        }
    }
    fn sigmoid(x: ExprId) -> ExprId {
        lit(1.0) / (lit(1.0) + (-x).exp())
    }

    /// A linear-attention-like layer over `x [T, dm]`: `delta_scan` state
    /// (plain linear attention, or the gated delta rule with L2-normed
    /// keys, `α = σ(x·wa)`, `β = σ(x·wb)`), its readout, an output
    /// projection.
    fn scan_layer(g: &mut TensorGraph, x: TensorId, dm: usize, n: usize, delta: bool) -> TensorId {
        let (wq, wk, wv, wo) = (
            g.input(&[dm, n]),
            g.input(&[dm, n]),
            g.input(&[dm, n]),
            g.input(&[n, dm]),
        );
        let q = g.matmul(x, wq).unwrap();
        let k = g.matmul(x, wk).unwrap();
        let v = g.matmul(x, wv).unwrap();
        let k = g
            .map(&[k], |v| lit(2.0) * sigmoid(v[0]) - lit(1.0))
            .unwrap();
        let s = if delta {
            let sq = g.map(&[k], |v| v[0] * v[0]).unwrap();
            let ss = g.reduce(sq, 1, Monoid::sum()).unwrap();
            let ssb = g.broadcast(ss, &[T, Dim::Fixed(n)]).unwrap();
            let kn = g
                .map(&[k, ssb], |v| v[0] * (v[1] + lit(1e-6)).sqrt().recip())
                .unwrap();
            let (wa, wb) = (g.input(&[dm, 1]), g.input(&[dm, 1]));
            let a = g.matmul(x, wa).unwrap();
            let b = g.matmul(x, wb).unwrap();
            let a = g.map(&[a], |v| sigmoid(v[0] + lit(2.0))).unwrap();
            let b = g.map(&[b], |v| sigmoid(v[0])).unwrap();
            let a = g.reshape(a, &[T]).unwrap();
            let b = g.reshape(b, &[T]).unwrap();
            g.delta_scan(kn, v, Some(a), Some(b), 0, true).unwrap()
        } else {
            g.delta_scan(k, v, None, None, 0, false).unwrap()
        };
        let o = g.scan_readout(s, q, 1.0 / (n as f64).sqrt()).unwrap();
        g.matmul(o, wo).unwrap()
    }

    /// Attention, linear attention and a gated-delta-rule layer as
    /// residual blocks, then `y = h + cumsum_t(h)/(t + 1)` (a monoid scan
    /// and a position), and `pooled = Σ_t y` (a reduction over the
    /// sequence into no sequence axis). Outputs `[y, pooled]`.
    fn hybrid() -> (TensorGraph, Vec<TensorId>) {
        let (dm, dh, n) = (6, 4, 4);
        let mut g = TensorGraph::new();
        let x = g.input(&[T, Dim::Fixed(dm)]);
        let a = attention(&mut g, x, dm, dh, Mask::Select);
        let h = g.map(&[x, a], |v| v[0] + v[1]).unwrap();
        let l = scan_layer(&mut g, h, dm, n, false);
        let h = g.map(&[h, l], |v| v[0] + v[1]).unwrap();
        let d = scan_layer(&mut g, h, dm, n, true);
        let h = g.map(&[h, d], |v| v[0] + v[1]).unwrap();
        let cs = g.scan(&[h], 0, Monoid::sum(), false, false).unwrap();
        let t = g.iota(&[T, Dim::Fixed(dm)], 0);
        let y = g
            .map(&[h, cs, t], |v| v[0] + v[1] / (v[2] + lit(1.0)))
            .unwrap();
        let pooled = g.reduce(y, 0, Monoid::sum()).unwrap();
        (g, vec![y, pooled])
    }

    /// Scan states and a running total are carried: K/V rows (NaN tail),
    /// the linear-attention and delta-rule states, the cumsum state and the
    /// pooled total; steps k = 1, 4, chained, all rows or the last.
    #[test]
    fn running_states_match_prefill() {
        let (g, outs) = hybrid();
        let step = derive_step(&g, &outs, &spec(1)).unwrap();
        let kinds: Vec<(bool, Vec<Dim>)> = step
            .caches
            .iter()
            .map(|c| (c.state, g.shape(c.node).to_vec()))
            .collect();
        eprintln!("hybrid caches (state, prefill shape): {kinds:?}");
        assert_eq!(step.caches.iter().filter(|c| !c.state).count(), 2); // K, V
        assert_eq!(step.caches.iter().filter(|c| c.state).count(), 4);
        for ks in [&[1usize][..], &[4], &[4, 1, 2]] {
            for rows in [Rows::All, Rows::Last] {
                let e = check_steps_with(&g, &outs, 5, ks, (0, 0), rows);
                eprintln!("hybrid ks={ks:?} {rows:?}: max rel err {e:.2e}");
                assert!(e < 1e-12, "{ks:?} {rows:?}: {e}");
            }
        }
        // negative control: states from a prefill one row short
        let e = check_steps_with(&g, &outs, 5, &[1], (0, 1), Rows::All);
        eprintln!("hybrid, stale states: max rel err {e:.2e}");
        assert!(e > 1e-3, "{e}");
    }

    /// The delta-rule step without `A`: the hybrid's step (whose
    /// gated-delta scan carries a [`DeltaForm`](super::super::DeltaForm))
    /// against the same step derived with the form stripped (the dense
    /// `Σ_j A[i, j]·S[j, c]`), on the same inputs: equal outputs and
    /// states, and at k = 1 no reduction left whose output is a whole
    /// `[n, m]` state.
    #[test]
    fn delta_step_factors_out_the_dense_product() {
        let (g, outs) = hybrid();
        let mut dense = g.clone();
        let mut forms = 0;
        for nd in &mut dense.nodes {
            if let Op::Scan(sc) = &mut nd.op {
                if let ScanKind::Affine { form, .. } = &mut sc.kind {
                    forms += usize::from(form.take().is_some());
                }
            }
        }
        assert_eq!(forms, 2, "linear attention and the delta rule");
        // reductions with an [n, m] = [4, 4] output: the dense AS
        let state_reductions = |st: &StepGraph| {
            (0..st.graph.nodes().len())
                .filter(|&i| {
                    st.graph.comp_at(i).is_some_and(|c| {
                        !c.red.is_empty() && c.points().ends_with(&[Dim::Fixed(4), Dim::Fixed(4)])
                    })
                })
                .count()
        };
        for k in [1usize, 4] {
            let fs = derive_step(&g, &outs, &spec(k)).unwrap();
            let ds = derive_step(&dense, &outs, &spec(k)).unwrap();
            assert_eq!(fs.inputs, ds.inputs);
            let shapes: Vec<Vec<usize>> = (0..fs.inputs.len())
                .map(|j| {
                    let nd = fs
                        .graph
                        .nodes()
                        .iter()
                        .find(|nd| matches!(nd.op, Op::Input(i) if i == j))
                        .unwrap();
                    nd.shape.iter().map(|d| d.n()).collect()
                })
                .collect();
            let inputs: Vec<Vec<f64>> = fs
                .inputs
                .iter()
                .zip(&shapes)
                .enumerate()
                .map(|(j, (si, sh))| match si {
                    StepInput::Pos => vec![5.0],
                    _ => rand(sh.iter().product(), 101 + j as u64, 0.8),
                })
                .collect();
            let ids = |st: &StepGraph| {
                let mut v = st.outputs.clone();
                v.extend(st.caches.iter().map(|c| c.updated));
                v
            };
            let a = fs.graph.eval(&inputs, &ids(&fs));
            let b = ds.graph.eval(&inputs, &ids(&ds));
            let worst = a
                .iter()
                .zip(&b)
                .map(|(x, y)| max_err(x, y))
                .fold(0.0, f64::max);
            let (nf, nd) = (state_reductions(&fs), state_reductions(&ds));
            eprintln!(
                "k={k}: factored vs dense delta step max rel err {worst:.2e}; \
                 [n, m] reductions {nf} vs {nd}"
            );
            assert!(worst < 1e-12, "k={k}: {worst}");
            // (k > 1 scans its rows with `linear_scan`, which has its own
            // [n, m] products; the decode step k = 1 has none)
            if k == 1 {
                assert_eq!(nf, 0);
                assert!(nd >= 2, "{nd}");
            }
        }
    }

    /// A reverse scan reads the future; a cached row computed from a
    /// running total would go stale. Both are refused.
    #[test]
    fn anticausal_states_are_refused() {
        let mut g = TensorGraph::new();
        let x = g.input(&[T, Dim::Fixed(3)]);
        let r = g.scan(&[x], 0, Monoid::sum(), false, true).unwrap();
        let msg = derive_step(&g, &[r], &spec(1)).err().unwrap().0;
        assert!(msg.contains("revers"), "{msg}");

        // attention whose keys read the running total
        let mut g = TensorGraph::new();
        let x = g.input(&[T, Dim::Fixed(4)]);
        let tot = g.reduce(x, 0, Monoid::sum()).unwrap();
        let totb = g.broadcast(tot, &[T, Dim::Fixed(4)]).unwrap();
        let xk = g.map(&[x, totb], |v| v[0] + v[1]).unwrap();
        let wq = g.input(&[4, 4]);
        let q = g.matmul(x, wq).unwrap();
        let kt = g.transpose(xk, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap();
        let y = select_softmax_v(&mut g, s, x, 4, 0.5);
        let msg = derive_step(&g, &[y], &spec(1)).err().unwrap().0;
        assert!(msg.contains("running state"), "{msg}");
    }
}
