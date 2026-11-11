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
//! Not handled (an error, not a wrong graph): shifted or strided reads
//! along `T` (convolutions over time would need a ring of the last few
//! rows), position tables indexed by `T` through a fixed axis (write the
//! lookup as a masked sum over the table, as `ir_qwen` does), gathers
//! and guards over `T`, reshapes that move `T` off the outer dim,
//! operators/scans touching `T`, outputs
//! with several `T`-axes, and reductions over `T` whose output has no
//! `T` (a running state: the monoid state itself would be the cache).

use std::collections::HashMap;

use tang::Scalar;
use tang_expr::{ExprGraph, ExprId, Node};

use super::{
    err, Arg, Comp, Dim, Form, Monoid, Op, Scan, ScanKind, Semiring, ShapeError, TensorGraph,
    TensorId, View,
};

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
    // last-row outputs: a node whose new rows feed a cache anyway is
    // sliced from them rather than computed again for the last row
    let need_new = match opts.rows {
        Rows::All => None,
        Rows::Last => {
            let all = derive_step_with(
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
    // a cached row must not depend on a carried state: the state changes
    // with every new row, so the old rows' cached values would go stale
    for r in d.caches.iter().filter(|c| !c.state) {
        for st in d.caches.iter().filter(|c| c.state) {
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
                    // the sequence is the outer dim on both sides: row-major,
                    // its rows are the same bytes whatever the inner dims
                    ([0], [0]) => {
                        let mut m = vec![None; g.nodes[src.n()].shape.len()];
                        m[0] = modes[0];
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

    /// `out[p…] = m.combine(acc[…], el[p…])` over `shape` (a trailing tuple
    /// dim when the monoid is wider than 1): `acc_picks` maps `acc`'s dims
    /// (without the tuple dim) to point coordinates.
    fn combine(
        &mut self,
        m: &Monoid,
        shape: &[Dim],
        acc: TensorId,
        acc_picks: &[Option<usize>],
        el: TensorId,
    ) -> Result<TensorId, ShapeError> {
        let w = m.width;
        let tuple = usize::from(w > 1);
        let pts = shape.len() - tuple;
        let read = |src: TensorId, picks: &[Option<usize>], slot: usize| {
            let mut v = View::pick(pts, picks);
            if w > 1 {
                v.forms.push(Form::constant(pts, slot as i64));
            }
            Arg { src, view: v }
        };
        let el_picks: Vec<Option<usize>> = (0..pts).map(Some).collect();
        let mut args: Vec<Arg> = (0..w).map(|s| read(acc, acc_picks, s)).collect();
        args.extend((0..w).map(|s| read(el, &el_picks, s)));
        self.sg.push_comp(Comp {
            out: shape.to_vec(),
            tuple,
            red: Vec::new(),
            args,
            body: m.combine.clone(),
            body_outs: m.outs.clone(),
            monoid: None,
        })
    }

    fn state_slot(&mut self, n: usize, axis: usize, input: TensorId, updated: TensorId) {
        let slot = self.caches.len();
        self.caches.push(CacheSlot {
            node: TensorId(n as u32),
            axis,
            input,
            updated,
            state: true,
        });
        self.cache_of.insert(n, slot);
    }

    /// A scan along the sequence: the scan of the new rows, started from
    /// the cached state at `p0 − 1`; the updated state is its last row.
    fn scan(&mut self, n: usize, sc: &Scan, modes: &Modes) -> Result<TensorId, ShapeError> {
        let shape = self.g.nodes[n].shape.clone();
        let tax = self.t_axes(n);
        if tax.is_empty() {
            // no sequence: the same scan over the same (sequence-free) inputs
            let ins = sc
                .inputs
                .iter()
                .map(|t| self.get(t.n(), vec![None; self.g.nodes[t.n()].shape.len()]))
                .collect::<Result<Vec<_>, _>>()?;
            return match &sc.kind {
                ScanKind::Monoid(m) => {
                    self.sg
                        .scan(&ins, sc.dim, m.clone(), sc.exclusive, sc.reverse)
                }
                ScanKind::Affine {
                    form: None,
                    ring: Semiring::Real,
                    bands: None,
                } => self
                    .sg
                    .linear_scan(ins[0], ins[1], sc.dim, sc.exclusive, sc.reverse),
                _ => err("step: this scan kind is not derived yet"),
            };
        }
        if tax != [sc.dim] {
            return err(format!(
                "step: scan node {n} runs along dim {} but its sequence axes are {tax:?}",
                sc.dim
            ));
        }
        if sc.exclusive || sc.reverse {
            return err(format!(
                "step: scan node {n} is exclusive or reversed (a reverse scan reads the future)"
            ));
        }
        match modes[sc.dim] {
            Some(Mode::New) => {}
            Some(Mode::Last) => {
                let new = self.get(n, self.modes_with(n, Mode::New))?;
                return self.last_row(new, sc.dim, true);
            }
            _ => {
                return err(format!(
                    "step: scan node {n} is read at old rows (cache the state, not the rows)"
                ))
            }
        }
        let dim = sc.dim;
        let k = self.spec.new;
        let step_shape = self.step_shape(&shape, modes);
        // state: the output's shape without the scan dim
        let mut st_shape = step_shape.clone();
        st_shape.remove(dim);
        let new_in = |d: &mut Self, t: TensorId| {
            let m = d.modes_with(t.n(), Mode::New);
            d.get(t.n(), m)
        };
        match &sc.kind {
            ScanKind::Monoid(m) => {
                let ins = sc
                    .inputs
                    .iter()
                    .map(|&t| new_in(self, t))
                    .collect::<Result<Vec<_>, _>>()?;
                let part = self.sg.scan(&ins, dim, m.clone(), false, false)?;
                let input = self.sg.input(&st_shape);
                self.inputs.push(StepInput::Cache(self.caches.len()));
                let pts = step_shape.len() - usize::from(m.width > 1);
                let acc_picks: Vec<Option<usize>> =
                    (0..pts).filter(|&d| d != dim).map(Some).collect();
                let out = self.combine(m, &step_shape, input, &acc_picks, part)?;
                let updated = self.last_row(out, dim, false)?;
                self.state_slot(n, dim, input, updated);
                Ok(out)
            }
            ScanKind::Affine {
                ring: Semiring::Real,
                bands: None,
                ..
            } => {
                // S_t = A_t·S_{t−1} + U_t from S_{p0−1}: fold the cached
                // state into the first new row, U'_0 = A_0·S + U_0
                let a = new_in(self, sc.inputs[0])?;
                let u = new_in(self, sc.inputs[1])?;
                let input = self.sg.input(&st_shape);
                self.inputs.push(StepInput::Cache(self.caches.len()));
                let r = step_shape.len(); // [P…, n, m]
                let nn = step_shape[r - 2];
                // AS[P…, i, c] = Σ_j A[P…, i, j]·S[P…\dim, j, c]: space (P…, i, c, j)
                let mut a_picks: Vec<Option<usize>> = (0..r - 1).map(Some).collect();
                a_picks.push(Some(r));
                let mut s_picks: Vec<Option<usize>> =
                    (0..r - 2).filter(|&d| d != dim).map(Some).collect();
                s_picks.extend([Some(r), Some(r - 1)]);
                let as_ = self.sg.comp(
                    &step_shape,
                    &[nn],
                    vec![
                        Arg {
                            src: a,
                            view: View::pick(r + 1, &a_picks),
                        },
                        Arg {
                            src: input,
                            view: View::pick(r + 1, &s_picks),
                        },
                    ],
                    Some(Monoid::sum()),
                    |v| vec![v[0] * v[1]],
                )?;
                let t = self.sg.iota(&step_shape, dim);
                let u0 = self.sg.comp(
                    &step_shape,
                    &[],
                    vec![
                        Arg {
                            src: as_,
                            view: View::identity(r, r),
                        },
                        Arg {
                            src: u,
                            view: View::identity(r, r),
                        },
                        Arg {
                            src: t,
                            view: View::identity(r, r),
                        },
                    ],
                    None,
                    |v| {
                        let first = ExprId::from_f64(0.5) - v[2]; // t' == 0
                        let folded = v[0] + v[1];
                        vec![tang_expr::with_graph(|eg| eg.select(first, folded, v[1]))]
                    },
                )?;
                // (the chunked schedule's DeltaForm no longer describes U')
                let out = self.sg.linear_scan(a, u0, dim, false, false)?;
                let updated = self.last_row(out, dim, false)?;
                self.state_slot(n, dim, input, updated);
                let _ = k;
                Ok(out)
            }
            _ => err(format!(
                "step: scan node {n} is a log/max-plus or banded scan (not derived yet)"
            )),
        }
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
            if a.view.is_gather() && (src_seq || a.view.forms.iter().any(touches_seq)) {
                return err(format!("step: node {n} arg {ai} gathers over the sequence"));
            }
            if a.view.guard && src_seq {
                return err(format!(
                    "step: node {n} arg {ai} is a guarded read over the sequence"
                ));
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
                    _ => None,
                };
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
                        move |v| vec![v[0] + v[1] + ExprId::from_f64(off)],
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
            .map(|&d| {
                if self.is_seq(d) {
                    self.extent(red_mode)
                } else {
                    d
                }
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
    use super::super::Monoid;
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
        check_steps_skewed(g, outs, p0, ks, 0)
    }

    /// [`check_steps`] feeding the step `p0 + skew` as its position (a
    /// negative control when `skew != 0`).
    fn check_steps_skewed(
        g: &TensorGraph,
        outs: &[TensorId],
        p0: usize,
        ks: &[usize],
        skew: usize,
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
            let step = derive_step(g, outs, &spec).unwrap();
            let cache_nodes: Vec<TensorId> = step.caches.iter().map(|c| c.node).collect();
            if caches.is_empty() {
                let pg = g.bind(&[("T", pos)]).unwrap();
                let vals = pg.eval(&prefill_inputs(&data, tmax, pos), &cache_nodes);
                for (c, v) in step.caches.iter().zip(vals) {
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
                    StepInput::Pos => vec![(pos + skew) as f64],
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
                let a = g.shape(*o).iter().position(|&d| d == T).unwrap();
                let e = max_err(gv, &rows(wv, &shape, a, pos, grown));
                worst = worst.max(e);
            }
            for (ci, c) in step.caches.iter().enumerate() {
                let cshape = concrete(g.shape(c.node), cap);
                let gv = &got[outs.len() + ci];
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
}
