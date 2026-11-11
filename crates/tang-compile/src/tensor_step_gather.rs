//! Gathers along the sequence in [`derive_step`](super::derive_step): top-k
//! (sparse) attention that reads only the picked rows.
//!
//! A comp arg may gather a source's sequence axis when all of this holds
//! (anything else is refused, as before):
//!
//! - the gathered source axis is a sequence axis, read *only* through the
//!   index (`forms[sa]` the constant 0, `gather.coefs` 1 there and 0
//!   elsewhere), and the source has no other sequence axis;
//! - the index arg reads an index tensor `X` whose one sequence axis is read
//!   by a unit form of a sequence point coordinate `i` of the comp;
//! - `X` is provably a *causal index*: every value of row `r` lies in
//!   `[0, r]`. Accepted: `X` a [`Monoid::topk`] reduction (the arg reading
//!   index slots only: even slot forms) or a map, whose index-slot body is
//!   built from the literal 0, iota reads of `i`, and iota reads of a reduce
//!   coordinate `j` under `select(cond, ·, ·)` whose `cond` is a causal
//!   mask `j ≤ i` (affine in iotas, flipping between `j = i` and `i + 1`).
//!   The top-k identity's index is 0.
//!
//! The source is then demanded at full rows, i.e. read from its cache (a
//! single-`T` node is cached as usual), and the gather view is copied
//! unchanged: in the step the index tensor holds absolute positions (iotas
//! count from `p0`), every one `≤ p0 + t'`, so the gather only reads filled
//! cache rows — never an unfilled (NaN) one. Batched sessions compose: the
//! batch axis is an ordinary non-sequence axis of both the index and the
//! source.
//!
//! Data-dependent masks need no new rule: a reduction over the sequence is
//! accepted when its body is `select(causal, select(data, x, id), id)` (the
//! data mask ANDed with a positional causal mask at the top), the existing
//! recognizer. A data mask alone, however it was computed, is refused.

use tang_expr::{ExprGraph, ExprId, Node};

use super::super::{Comp, Form, Monoid, Op, ShapeError, TensorId, View};
use super::{err, Deriver, Mode, Modes};

impl Deriver<'_> {
    /// The source modes of gather arg `ai` of comp node `n` (iteration
    /// coordinate modes `coord`, `pts` point coordinates), or why not.
    pub(super) fn seq_gather(
        &self,
        n: usize,
        ai: usize,
        c: &Comp,
        coord: &Modes,
    ) -> Result<Modes, ShapeError> {
        let a = &c.args[ai];
        let pts = c.points().len();
        let g = a.view.gather.as_ref().expect("a gather arg");
        let src_shape = &self.g.nodes[a.src.n()].shape;
        let no = |why: &str| {
            err(format!(
                "step: node {n} arg {ai} gathers over the sequence: {why}"
            ))
        };
        let touches_seq = |f: &Form| {
            f.coefs
                .iter()
                .enumerate()
                .any(|(d, &k)| k != 0 && coord[d].is_some())
        };
        let gathered: Vec<usize> = (0..g.coefs.len()).filter(|&d| g.coefs[d] != 0).collect();
        let [sa] = gathered[..] else {
            return no("the index addresses several source axes");
        };
        if !self.is_seq(src_shape[sa]) || g.coefs[sa] != 1 {
            return no("the index addresses a fixed axis (or is scaled)");
        }
        let f = &a.view.forms[sa];
        if f.offset != 0 || f.coefs.iter().any(|&k| k != 0) {
            return no("the gathered axis has an affine part besides the index");
        }
        let mut smodes = Modes::with_capacity(src_shape.len());
        for (d, f) in a.view.forms.iter().enumerate() {
            if d == sa {
                smodes.push(Some(Mode::Full));
            } else if self.is_seq(src_shape[d]) {
                return no("the source has a second sequence axis");
            } else if touches_seq(f) {
                return no("a fixed source axis is indexed by a sequence coordinate");
            } else {
                smodes.push(None);
            }
        }
        if a.view.guard {
            return no("the read is guarded");
        }
        // the index tensor and the point coordinate that reads its row
        let ix = &c.args[g.arg];
        let x = ix.src.n();
        let xt = self.t_axes(x);
        let [xa] = xt[..] else {
            return no("the index tensor does not have exactly one sequence axis");
        };
        let row = ix.view.forms[xa].as_unit();
        if !row.is_some_and(|i| i < pts && coord[i].is_some()) {
            return no("the index row is not read at a sequence point coordinate");
        }
        if ix
            .view
            .forms
            .iter()
            .enumerate()
            .any(|(d, f)| d != xa && touches_seq(f))
        {
            return no("the index tensor is read at a shifted sequence coordinate");
        }
        if !self.causal_index(x, &ix.view) {
            return no(
                "its index is not provably causal (a top-k or map whose index is j under \
                 select(causal(j ≤ i), …) or 0)",
            );
        }
        Ok(smodes)
    }

    /// Whether every value of node `x` that `view` reads lies in `[0, r]`
    /// for row `r` (see the module doc).
    fn causal_index(&self, x: usize, view: &View) -> bool {
        let Op::Comp(xc) = &self.g.nodes[x].op else {
            return false;
        };
        if xc.has_gather() {
            return false;
        }
        let xa = self.t_axes(x)[0];
        let rank = xc.rank();
        let outs: Vec<usize> = match (&xc.monoid, xc.tuple) {
            (None, 0) => vec![0],
            (Some(m), 1) if is_topk(m) => {
                // the tuple axis must be read at even (index) slots only
                let f = &view.forms[xc.out.len() - 1];
                if f.offset % 2 != 0 || f.coefs.iter().any(|k| k % 2 != 0) {
                    return false;
                }
                (0..xc.body_outs.len()).step_by(2).collect()
            }
            _ => return false,
        };
        let args: Vec<(TensorId, View)> = xc.args.iter().map(|a| (a.src, a.view.clone())).collect();
        outs.iter()
            .all(|&o| self.bounded(&xc.body, &args, rank, xc.body_outs[o], xa, &mut Vec::new()))
    }

    /// `e ∈ [0, x_i]` given `under` (coordinates proven `≤ x_i`).
    fn bounded(
        &self,
        body: &ExprGraph,
        args: &[(TensorId, View)],
        rank: usize,
        e: ExprId,
        i: usize,
        under: &mut Vec<usize>,
    ) -> bool {
        let nodes = body.nodes_slice();
        match nodes[e.index() as usize] {
            Node::Lit(b) => f64::from_bits(b) == 0.0,
            Node::Var(v) => {
                let Some((src, view)) = args.get(v as usize) else {
                    return false;
                };
                let Op::Iota(dim) = self.g.nodes[src.n()].op else {
                    return false;
                };
                !view.is_gather()
                    && view.forms[dim]
                        .as_unit()
                        .is_some_and(|d| d == i || under.contains(&d))
            }
            Node::Select(cond, t, f) => {
                let causal =
                    (0..rank).find(|&j| j != i && self.causal_cond(body, args, rank, cond, i, j));
                let n0 = under.len();
                if let Some(j) = causal {
                    under.push(j);
                }
                let ok_t = self.bounded(body, args, rank, t, i, under);
                under.truncate(n0);
                ok_t && self.bounded(body, args, rank, f, i, under)
            }
            _ => false,
        }
    }

    /// `cond > 0` iff `j ≤ i` (affine in iota reads of `i` and `j`).
    fn causal_cond(
        &self,
        body: &ExprGraph,
        args: &[(TensorId, View)],
        rank: usize,
        cond: ExprId,
        i: usize,
        j: usize,
    ) -> bool {
        let Some((coefs, konst)) = self.affine(body, args, rank, cond) else {
            return false;
        };
        let only_ij = coefs
            .iter()
            .enumerate()
            .all(|(d, &k)| k == 0.0 || d == i || d == j);
        let at = |delta: f64| coefs[j] * delta + konst > 0.0;
        only_ij && coefs[j] != 0.0 && coefs[i] == -coefs[j] && at(0.0) && !at(1.0)
    }
}

/// `m` is [`Monoid::topk`] of some `k` (pairs `(index, value)`).
fn is_topk(m: &Monoid) -> bool {
    if m.width < 2 || m.width % 2 != 0 {
        return false;
    }
    let t = Monoid::topk(m.width / 2);
    m.identity == t.identity
        && m.outs == t.outs
        && m.combine.nodes_slice() == t.combine.nodes_slice()
}
