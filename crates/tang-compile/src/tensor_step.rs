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
//! of `capacity` rows at `p0..p0+k` (a map with a guarded gather through
//! `p0`, which is a one-element runtime input). Those nodes are the
//! cache; for attention, K and V (before any position map the rewrite
//! passes inlined into the scores).
//!
//! **Causality.** The derivation computes only new rows, which is right
//! iff growing `T` leaves old rows unchanged. Views are unit reads, so
//! the only way a row can see a later one is a reduction over `T`; each
//! such reduction must contain a `select` whose condition is affine in
//! iota reads and changes sign exactly between `j = i` and `j = i + 1`
//! (a causal mask). That the masked branch contributes the monoid's
//! identity (`exp(−1e30 − m) = 0`, or `select(c, x, id)`) is not proven
//! here; the interpreter tests check it.
//!
//! Not handled (an error, not a wrong graph): shifted or strided reads
//! along `T` (convolutions over time would need a ring of the last few
//! rows), position tables indexed by `T` through a fixed axis, gathers
//! and guards over `T`, reshapes/operators/scans touching `T`, outputs
//! with several `T`-axes, and reductions over `T` whose output has no
//! `T` (a running state: the monoid state itself would be the cache).

use std::collections::HashMap;

use tang::Scalar;
use tang_expr::{ExprGraph, ExprId, Node};

use super::{err, Arg, Comp, Dim, Form, Op, ShapeError, TensorGraph, TensorId, View};

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

/// Where a step-graph input comes from, in the step graph's input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepInput {
    /// `[1]`: `p0`, the number of positions already in the caches.
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
}

/// The derived step.
pub struct StepGraph {
    pub graph: TensorGraph,
    pub inputs: Vec<StepInput>,
    /// The prefill outputs' new rows, in the order asked for.
    pub outputs: Vec<TensorId>,
    pub caches: Vec<CacheSlot>,
    pub pos: TensorId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Mode {
    New,
    Full,
}

type Modes = Vec<Option<Mode>>;

/// Derive the `T → T + k` step of a causal prefill graph. Run the rewrite
/// passes first (`substitute_maps`, optionally `stabilize_softmax`): the
/// derivation works on materialized nodes, so `[T, T]` maps such as a
/// mask or probabilities should already be inlined into their readers.
pub fn derive_step(
    g: &TensorGraph,
    outputs: &[TensorId],
    spec: &StepSpec,
) -> Result<StepGraph, ShapeError> {
    let mut d = Deriver {
        g,
        spec: *spec,
        sg: TensorGraph::new(),
        inputs: Vec::new(),
        pos: TensorId(0),
        memo: HashMap::new(),
        inputs_memo: HashMap::new(),
        caches: Vec::new(),
        cache_of: HashMap::new(),
    };
    d.pos = d.sg.input(&[1usize]);
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
        let modes = d.modes_with(o.n(), Mode::New);
        outs.push(d.get(o.n(), modes)?);
    }
    Ok(StepGraph {
        graph: d.sg,
        inputs: d.inputs,
        outputs: outs,
        caches: d.caches,
        pos: d.pos,
    })
}

struct Deriver<'a> {
    g: &'a TensorGraph,
    spec: StepSpec,
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
        let t = match &node.op {
            Op::Iota(dim) => self.iota(n, *dim, &modes)?,
            Op::Input(_) | Op::Comp(_) if full_single => self.cache(n, tax[0])?,
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
                if !tax.is_empty() || !self.t_axes(src.n()).is_empty() {
                    return err("step: reshape over the sequence dim");
                }
                let s = self.get(src.n(), vec![None; self.g.nodes[src.n()].shape.len()])?;
                let shape = node.shape.clone();
                self.sg.reshape(s, &shape)?
            }
            Op::Operator(_) | Op::Scan(_) => {
                return err("step: operator and scan nodes are not derived yet")
            }
        };
        self.memo.insert((n, modes), t);
        Ok(t)
    }

    /// An iota in the step: a new-mode axis counts from `p0`.
    fn iota(&mut self, n: usize, dim: usize, modes: &Modes) -> Result<TensorId, ShapeError> {
        let shape = self.step_shape(&self.g.nodes[n].shape, modes);
        let it = self.sg.iota(&shape, dim);
        if modes[dim] != Some(Mode::New) {
            return Ok(it);
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
        self.sg.comp(&shape, &[], args, None, |v| vec![v[0] + v[1]])
    }

    /// Node `n` (one `T`-axis, `axis`) read at old rows: its new rows
    /// written into a cache input.
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
        let iota = self.sg.iota(&shape, axis);
        let k = self.spec.new as f64;
        // new[j − p0] (0 outside the new rows), through a gather on `p0`
        let gathered = View::identity(r, r).gather(axis, 2, -1).guarded();
        let args = vec![
            Arg {
                src: input,
                view: View::identity(r, r),
            },
            Arg {
                src: new,
                view: gathered,
            },
            Arg {
                src: self.pos,
                view: View::pick(r, &[None]),
            },
            Arg {
                src: iota,
                view: View::identity(r, r),
            },
        ];
        let updated = self.sg.comp(&shape, &[], args, None, |v| {
            let (old, new, p0, j) = (v[0], v[1], v[2], v[3]);
            let half = ExprId::from_f64(0.5);
            let lo = j - p0 + half; // j >= p0
            let hi = p0 + ExprId::from_f64(k) - j - half; // j < p0 + k
            vec![tang_expr::with_graph(|eg| {
                let inner = eg.select(hi, new, old);
                eg.select(lo, inner, old)
            })]
        })?;
        self.caches.push(CacheSlot {
            node: TensorId(n as u32),
            axis,
            input,
            updated,
        });
        self.cache_of.insert(n, slot);
        Ok(updated)
    }

    fn comp(&mut self, n: usize, c: &Comp, modes: &Modes) -> Result<TensorId, ShapeError> {
        let pts = c.points().len();
        // the mode of every iteration coordinate (None: not over T)
        let mut coord: Modes = modes[..pts].to_vec();
        coord.extend(c.red.iter().map(|&d| self.is_seq(d).then_some(Mode::Full)));
        let seq_red: Vec<usize> = (pts..coord.len()).filter(|&d| coord[d].is_some()).collect();
        if !seq_red.is_empty() {
            let seq_pts: Vec<usize> = (0..pts).filter(|&d| coord[d].is_some()).collect();
            if seq_pts.is_empty() {
                return err(format!(
                    "step: node {n} reduces over the sequence into no sequence axis \
                     (a running state; not derived yet)"
                ));
            }
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
                    self.extent(Mode::Full)
                } else {
                    d
                }
            })
            .collect();
        self.sg.push_comp(Comp {
            out,
            tuple: c.tuple,
            red,
            args,
            body: c.body.clone(),
            body_outs: c.body_outs.clone(),
            monoid: c.monoid.clone(),
        })
    }

    /// Whether `c`'s body (or the body of a materialized map it reads,
    /// recursively) has a `select` whose condition is affine in iota reads
    /// and flips sign exactly between reduce coord `j = i` and `j = i + 1`
    /// (point coord `i`): a causal mask.
    fn causal_mask(&self, c: &Comp, i: usize, j: usize) -> bool {
        let rank = c.rank();
        let mut stack: Vec<(&ExprGraph, Vec<(TensorId, View)>)> = vec![(
            &c.body,
            c.args.iter().map(|a| (a.src, a.view.clone())).collect(),
        )];
        while let Some((body, args)) = stack.pop() {
            for nd in body.nodes_slice() {
                let Node::Select(cond, _, _) = *nd else {
                    continue;
                };
                let Some((coefs, konst)) = self.affine(body, &args, rank, cond) else {
                    continue;
                };
                let only_ij = coefs
                    .iter()
                    .enumerate()
                    .all(|(d, &k)| k == 0.0 || d == i || d == j);
                // value at (i, j = i + δ) is coefs[j]·δ + konst
                let at = |delta: f64| coefs[j] * delta + konst > 0.0;
                if only_ij && coefs[j] != 0.0 && coefs[i] == -coefs[j] && at(0.0) != at(1.0) {
                    return true;
                }
            }
            // look inside materialized maps read here, views composed
            for (src, view) in &args {
                if let Op::Comp(m) = &self.g.nodes[src.n()].op {
                    if m.red.is_empty() && m.tuple == 0 && !view.is_gather() {
                        let inner = m
                            .args
                            .iter()
                            .filter(|a| !a.view.is_gather())
                            .map(|a| (a.src, a.view.compose(view, rank)))
                            .collect::<Vec<_>>();
                        if inner.len() == m.args.len() {
                            stack.push((&m.body, inner));
                        }
                    }
                }
            }
        }
        false
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

    /// Single-head attention `x → softmax(q·kᵀ/√d [+ mask])·v · Wo`.
    fn attention(g: &mut TensorGraph, x: TensorId, dm: usize, dh: usize, causal: bool) -> TensorId {
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
        let mut s = g.map(&[s], |v| v[0] * lit(scale)).unwrap();
        if causal {
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
    fn model(layers: usize, causal: bool) -> (TensorGraph, TensorId) {
        let (dm, dh, df) = (6, 4, 8);
        let mut g = TensorGraph::new();
        let x = g.input(&[T, Dim::Fixed(dm)]);
        if layers == 0 {
            let y = attention(&mut g, x, dm, dh, causal);
            return (g, y);
        }
        let mut h = x;
        for _ in 0..layers {
            let n = rmsnorm(&mut g, h, dm);
            let a = attention(&mut g, n, dm, dh, causal);
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

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
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
        // junk, which the derived mask must hide
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
                    let mut buf = vec![7.0; cshape.iter().product()];
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

    fn prepared(layers: usize, causal: bool, stable: bool) -> (TensorGraph, Vec<TensorId>) {
        let (g, y) = model(layers, causal);
        let (g, outs) = g.substitute_maps(&[y]);
        if stable {
            g.stabilize_softmax(&outs)
        } else {
            (g, outs)
        }
    }

    #[test]
    fn attention_step_matches_prefill() {
        for stable in [false, true] {
            let (g, outs) = prepared(0, true, stable);
            let step = derive_step(
                &g,
                &outs,
                &StepSpec {
                    seq: "T",
                    new: 1,
                    capacity: 16,
                },
            )
            .unwrap();
            let caches = describe(&g, &step);
            eprintln!("attention (online softmax: {stable}) caches: {caches:#?}");
            // K and V, [T, dh]: nothing else is read at old rows
            assert_eq!(caches.len(), 2, "{caches:?}");
            assert!(step
                .caches
                .iter()
                .all(|c| g.shape(c.node) == [T, Dim::Fixed(4)]));
            for k in [1usize, 4] {
                let e = check_steps(&g, &outs, 5, &[k]);
                eprintln!("attention stable={stable} k={k}: max rel err {e:.2e}");
                assert!(e < 1e-12, "k={k}: {e}");
            }
        }
    }

    #[test]
    fn chained_steps_feed_the_cache_back() {
        // prefill 3, verify a 4-token window, then decode one at a time
        let (g, outs) = prepared(0, true, false);
        let e = check_steps(&g, &outs, 3, &[4, 1, 1, 2]);
        eprintln!("chained: max rel err {e:.2e}");
        assert!(e < 1e-12, "{e}");
    }

    #[test]
    fn two_layer_stack_caches_k_and_v_per_layer() {
        for stable in [false, true] {
            let (g, outs) = prepared(2, true, stable);
            let step = derive_step(
                &g,
                &outs,
                &StepSpec {
                    seq: "T",
                    new: 1,
                    capacity: 16,
                },
            )
            .unwrap();
            let caches = describe(&g, &step);
            eprintln!("2 layers (online softmax: {stable}) caches: {caches:#?}");
            assert_eq!(caches.len(), 4, "{caches:?}");
            for ks in [&[1usize][..], &[4], &[4, 1, 1]] {
                let e = check_steps(&g, &outs, 5, ks);
                eprintln!("2 layers stable={stable} ks={ks:?}: max rel err {e:.2e}");
                assert!(e < 1e-12, "{ks:?}: {e}");
            }
        }
    }

    /// The comparison can fail: a step told the wrong position writes the
    /// cache rows one off and masks one key too many.
    #[test]
    fn wrong_position_is_caught() {
        let (g, outs) = prepared(0, true, false);
        let e = check_steps_skewed(&g, &outs, 5, &[1], 1);
        eprintln!("skewed position: max rel err {e:.2e}");
        assert!(e > 1e-3, "{e}");
    }

    #[test]
    fn non_causal_attention_has_no_step() {
        let (g, outs) = prepared(0, false, false);
        let r = derive_step(
            &g,
            &outs,
            &StepSpec {
                seq: "T",
                new: 1,
                capacity: 16,
            },
        );
        let msg = r.err().expect("non-causal attention must be refused").0;
        assert!(msg.contains("causal"), "{msg}");
    }
}
