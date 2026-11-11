//! Contractions with many rows as tensor-core matmuls in the CUDA decode
//! lowering.
//!
//! A derived step that appends `k > 1` rows has contractions with `k`
//! (times heads) rows: attention's `Σ_j p·v` over the cache is a matmul
//! per head, but its body (`select(visible, exp(s − m)·v / l, 0)`) is not
//! a product of two reads, so the MMA tile does not take it and it runs
//! as a warp reduction per output. [`TensorGraph::split_bilinear`]
//! materializes the factor the column operand does not touch: when the
//! body is linear in one leaf read `B` and the rest of it does not depend
//! on some point dims `B` does (the columns), `A = body|_{B=1}` becomes
//! a map over the other points and the reduce dim, and the contraction
//! `Σ A·B`, which [`cuda_mma_kernel`](TensorGraph::cuda_mma_kernel)
//! lowers.

use super::{fixed, Arg, Comp, Form, MonoidKind, Op, TensorGraph, TensorId, View};
use tang_expr::{ExprGraph, ExprId, Node};

/// Rows (output points the column operand does not vary over) below which
/// a contraction stays a warp reduction (decode GEMVs).
pub(super) const MMA_ROWS: usize = 64;
/// Reduce length below which a contraction stays a warp reduction.
pub(super) const MMA_RED: usize = 256;

/// The dims a read depends on: nonzero coefficients over the comp space.
fn dims_of(v: &View, rank: usize) -> Vec<bool> {
    let mut d = vec![false; rank];
    for f in &v.forms {
        for (i, &c) in f.coefs.iter().enumerate() {
            if c != 0 {
                d[i] = true;
            }
        }
    }
    d
}

/// `g` with `Var(var)` replaced by `1`, rebuilt from `out`.
fn with_one(g: &ExprGraph, out: ExprId, var: u16) -> (ExprGraph, ExprId) {
    let mut ng = ExprGraph::new();
    let nodes = g.nodes_slice();
    let mut map: Vec<ExprId> = Vec::with_capacity(nodes.len());
    for nd in nodes {
        let m = |e: &ExprId| map[e.index() as usize];
        let id = match nd {
            Node::Var(p) if *p == var => ExprId::ONE,
            Node::Var(p) => ng.var(*p),
            Node::Lit(b) => ng.lit(f64::from_bits(*b)),
            Node::Add(a, b) => ng.add(m(a), m(b)),
            Node::Mul(a, b) => ng.mul(m(a), m(b)),
            Node::Neg(a) => ng.neg(m(a)),
            Node::Recip(a) => ng.recip(m(a)),
            Node::Sqrt(a) => ng.sqrt(m(a)),
            Node::Sin(a) => ng.sin(m(a)),
            Node::Atan2(a, b) => ng.atan2(m(a), m(b)),
            Node::Exp2(a) => ng.exp2(m(a)),
            Node::Log2(a) => ng.log2(m(a)),
            Node::Floor(a) => ng.floor(m(a)),
            Node::Select(c, a, b) => ng.select(m(c), m(a), m(b)),
        };
        map.push(id);
    }
    let o = map[out.index() as usize];
    (ng, o)
}

impl TensorGraph {
    /// Whether `root` is a contraction worth a tensor-core tile in a decode
    /// program: a sum over one reduce dim of at least [`MMA_RED`] with at
    /// least [`MMA_ROWS`] rows.
    pub(super) fn mma_candidate(&self, root: usize) -> bool {
        self.mma_candidate_min(root, MMA_ROWS, MMA_RED)
    }

    fn mma_candidate_min(&self, root: usize, min_rows: usize, min_red: usize) -> bool {
        let Some(c) = self.comp_at(root) else {
            return false;
        };
        c.monoid.as_ref().map(|m| m.kind()) == Some(MonoidKind::Sum)
            && c.red.len() == 1
            && c.tuple == 0
            && c.red[0].fixed().is_some_and(|r| r >= min_red)
            && c.points().iter().all(|d| d.fixed().is_some())
            && fixed(c.points()).iter().product::<usize>() >= min_rows * 8
    }

    /// The graph with each [`mma_candidate`](Self::mma_candidate) whose
    /// body is linear in one leaf read split into a materialized factor and
    /// a plain contraction (see the module docs); `None` when nothing
    /// splits.
    pub(super) fn split_bilinear(&self, outputs: &[TensorId]) -> Option<TensorGraph> {
        self.split_bilinear_min(outputs, MMA_ROWS, MMA_RED)
    }

    /// [`split_bilinear`](Self::split_bilinear) with the size thresholds
    /// given (tests split tiny graphs).
    #[doc(hidden)]
    pub fn split_bilinear_min(
        &self,
        outputs: &[TensorId],
        min_rows: usize,
        min_red: usize,
    ) -> Option<TensorGraph> {
        let live = self.live(outputs);
        let mut out: Option<TensorGraph> = None;
        for r in (0..self.nodes.len()).filter(|&r| live[r]) {
            if !self.mma_candidate_min(r, min_rows, min_red)
                || self.nodes[r].int_mma.is_some()
                || self.nodes[r].segment.is_some()
                || self.nodes[r].inline
            {
                continue;
            }
            let c = self.comp_at(r).unwrap();
            if c.args.iter().any(|a| a.view.gather.is_some() || a.view.guard) {
                continue;
            }
            // already a product of two reads
            let e = c.body_outs[0];
            let nodes = c.body.nodes_slice();
            if let Node::Mul(a, b) = nodes[e.index() as usize] {
                let leaf = |x: ExprId| {
                    matches!(nodes[x.index() as usize], Node::Var(_) | Node::Lit(_))
                };
                if leaf(a) && leaf(b) {
                    continue;
                }
            }
            let np = c.points().len();
            let rank = np + 1;
            let used = super::vars_reachable(&c.body, e);
            let deps: Vec<Vec<bool>> = c
                .args
                .iter()
                .map(|a| match self.nodes[a.src.n()].op {
                    // an iota varies along its own axis only
                    Op::Iota(ax) => {
                        let mut d = vec![false; rank];
                        for (i, &k) in a.view.forms[ax].coefs.iter().enumerate() {
                            d[i] = k != 0;
                        }
                        d
                    }
                    _ => dims_of(&a.view, rank),
                })
                .collect();

            if let Some((bv, cols)) = used.iter().find_map(|&b| {
                let b = b as usize;
                // columns: point dims only B reads
                let cols: Vec<bool> = (0..np)
                    .map(|d| {
                        deps[b][d]
                            && used
                                .iter()
                                .all(|&o| o as usize == b || !deps[o as usize][d])
                    })
                    .collect();
                // B is a plain (non-comp-inlined) buffer read reading the
                // reduce dim, and the body is linear in it
                let leaf = !matches!(self.nodes[c.args[b].src.n()].op, Op::Comp(_))
                    || !self.nodes[c.args[b].src.n()].inline;
                // linear in B at 64 pseudo-random points (any mask open at
                // some of them)
                let mut seed = 0x9e37_79b9_u64;
                let mut rnd = || {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 11) as f64 / (1u64 << 53) as f64) * 16.0 - 8.0
                };
                let (mut linear, mut live) = (true, false);
                for _ in 0..64 {
                    let mut x: Vec<f64> = (0..c.args.len())
                        .map(|_| {
                            let v = rnd().round() / 2.0;
                            if v == 0.0 { 0.5 } else { v }
                        })
                        .collect();
                    x[b] = 1.0;
                    let one = c.body.eval_many(&[e], &x)[0];
                    x[b] = 2.5;
                    let two = c.body.eval_many(&[e], &x)[0];
                    x[b] = 0.0;
                    let zero = c.body.eval_many(&[e], &x)[0];
                    linear &= one.is_finite()
                        && (two - 2.5 * one).abs() <= 1e-9 * one.abs().max(1.0)
                        && zero == 0.0;
                    live |= one != 0.0;
                }
                let linear = linear && live;
                (leaf && deps[b][np] && cols.iter().any(|&x| x) && linear)
                    .then_some((b, cols))
            }) {
                // A: space (points without the columns, j), body with B = 1
                let keep: Vec<usize> = (0..np).filter(|&d| !cols[d]).collect();
                let rows: usize = keep.iter().map(|&d| c.points()[d].n()).product();
                if rows < min_rows {
                    continue;
                }
                let mut ashape: Vec<_> = keep.iter().map(|&d| c.points()[d]).collect();
                ashape.push(c.red[0]);
                let ar = ashape.len();
                let remap = |v: &View| -> View {
                    View {
                        forms: v
                            .forms
                            .iter()
                            .map(|f| Form {
                                coefs: keep
                                    .iter()
                                    .map(|&d| f.coefs[d])
                                    .chain([f.coefs[np]])
                                    .collect(),
                                offset: f.offset,
                            })
                            .collect(),
                        guard: v.guard,
                        gather: None,
                    }
                };
                let (body, ae) = with_one(&c.body, e, bv as u16);
                let aargs: Vec<Arg> = c
                    .args
                    .iter()
                    .enumerate()
                    .map(|(i, a)| Arg {
                        src: a.src,
                        view: if i == bv {
                            // unused in A: any in-range read
                            View {
                                forms: a.view.forms.iter().map(|_| Form::zero(ar)).collect(),
                                guard: false,
                                gather: None,
                            }
                        } else {
                            remap(&a.view)
                        },
                    })
                    .collect();
                let g = out.get_or_insert_with(|| self.clone());
                let Ok(a) = g.push_comp(Comp {
                    out: ashape,
                    tuple: 0,
                    red: vec![],
                    args: aargs,
                    body,
                    body_outs: vec![ae],
                    monoid: None,
                }) else {
                    continue;
                };
                g.nodes[a.n()].precision = self.nodes[r].precision;
                g.nodes[a.n()].inline = false;
                // R = Σ_j A·B
                let mut av: Vec<Form> = keep.iter().map(|&d| Form::unit(rank, d)).collect();
                av.push(Form::unit(rank, np));
                let pts = c.points().to_vec();
                let red = c.red.clone();
                let bsrc = c.args[bv].clone();
                let Ok(nr) = g.comp(
                    &pts,
                    &red,
                    vec![
                        Arg {
                            src: a,
                            view: View {
                                forms: av,
                                guard: false,
                                gather: None,
                            },
                        },
                        bsrc,
                    ],
                    c.monoid.clone(),
                    |x| vec![x[0] * x[1]],
                ) else {
                    continue;
                };
                g.nodes[r].op = g.nodes[nr.n()].op.clone();
            }
        }
        out
    }
}

impl TensorGraph {
    /// The live graph renumbered in dependency order, and for each new
    /// node its id here (a split appends factors after their readers;
    /// the lowerings assume a node's args come before it).
    pub(super) fn renumber(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<usize>) {
        let order = self.topo(outputs);
        let mut remap = vec![usize::MAX; self.nodes.len()];
        for (new, &old) in order.iter().enumerate() {
            remap[old] = new;
        }
        let m = |t: TensorId| TensorId(remap[t.n()] as u32);
        let mut ng = TensorGraph {
            nodes: Vec::with_capacity(order.len()),
            n_inputs: self.n_inputs,
        };
        for &i in &order {
            let mut node = self.nodes[i].clone();
            match &mut node.op {
                Op::Comp(c) => c.args.iter_mut().for_each(|a| a.src = m(a.src)),
                Op::Reshape(src) => *src = m(*src),
                Op::Operator(o) => o.inputs.iter_mut().for_each(|t| *t = m(*t)),
                Op::Scan(sc) => sc.inputs.iter_mut().for_each(|t| *t = m(*t)),
                Op::Input(_) | Op::Iota(_) => {}
            }
            if let Some(sg) = &mut node.segment {
                sg.key = m(sg.key);
            }
            ng.nodes.push(node);
        }
        (ng, order)
    }
}
