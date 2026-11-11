//! Schedule GGUF block-format weights onto the int8 GEMV ([`IntMma`],
//! `tensor_cuda_int8.rs`) in any graph: every contraction `Σ_k W[(i,) r,
//! k]·X[…, k]` whose weight is a graph input stored Q4_K / Q5_K / Q8_0
//! reads a Q8x32 copy of its activation and runs as the int8 GEMV
//! ([`Stage::Gemv`]: decode, few positions). A block-format input that any
//! other kernel reads is demoted to f32 (the host uploads its dequantized
//! values, `tensor::gguf::dequantize`: the same values).

use std::collections::HashMap;

use tang_expr::Node;

use super::{
    Arg, Comp, Format, IntMma, MonoidKind, Op, Precision, Stage, TensorGraph, TensorId, View,
};

/// What [`TensorGraph::schedule_block_weights`] did.
#[derive(Debug, Default)]
pub struct BlockSchedule {
    /// Contractions now on the int8 GEMV.
    pub gemvs: usize,
    /// The Q8x32 activation copies. Keep them materialized through later
    /// passes (pass them as outputs to `substitute_maps`): an inlined copy
    /// is no Q8x32 buffer.
    pub copies: Vec<TensorId>,
    /// Block-format inputs some other kernel reads: now f32 storage.
    pub demoted: Vec<TensorId>,
}

fn int8_format(f: Format) -> bool {
    matches!(f, Format::Q4K | Format::Q5K | Format::Q8_0)
}

impl TensorGraph {
    /// The weight arg and activation arg of comp `i` when it is an int8
    /// GEMV candidate.
    fn int8_candidate(&self, i: usize) -> Option<(usize, usize)> {
        let c = self.comp_at(i)?;
        if c.tuple != 0
            || c.red.len() != 1
            || c.monoid.as_ref().map(|m| m.kind()) != Some(MonoidKind::Sum)
        {
            return None;
        }
        let pts: Vec<usize> = c
            .points()
            .iter()
            .map(|d| d.fixed())
            .collect::<Option<_>>()?;
        let k = c.red[0].fixed()?;
        let (np, kd) = (pts.len(), pts.len());
        if np < 2 || !pts[np - 1].is_multiple_of(16) || !k.is_multiple_of(256) {
            return None;
        }
        // the body: the product of two args
        let nodes = c.body.nodes_slice();
        let Node::Mul(x, y) = nodes[c.body_outs[0].index() as usize] else {
            return None;
        };
        let var = |e: tang_expr::ExprId| match nodes[e.index() as usize] {
            Node::Var(v) => Some(v as usize),
            _ => None,
        };
        let (a, b) = (var(x)?, var(y)?);
        let is_w = |j: usize| {
            matches!(self.nodes[c.args[j].src.n()].op, Op::Input(_))
                && int8_format(self.nodes[c.args[j].src.n()].precision.storage)
        };
        let (wi, xi) = match (is_w(a), is_w(b)) {
            (true, false) => (a, b),
            (false, true) => (b, a),
            _ => return None,
        };
        // every other arg is the weight's gather index
        let index = c.args[wi].view.gather.as_ref().map(|g| g.arg);
        if (0..c.args.len()).any(|j| j != wi && j != xi && Some(j) != index) {
            return None;
        }
        let (w, xa) = (&c.args[wi].view, &c.args[xi].view);
        let rank = np + 1;
        let wn = w.forms.len();
        let unit = |f: &super::Form, d: usize| *f == super::Form::unit(rank, d);
        if wn < 2 || !unit(&w.forms[wn - 1], kd) || !unit(&w.forms[wn - 2], np - 1) {
            return None;
        }
        if w.forms[..wn - 2]
            .iter()
            .any(|f| f.coefs.iter().any(|&q| q != 0))
        {
            return None;
        }
        // X: k along its last dim, contiguous; the copy is Q8x32 in 32s
        let xs = self.nodes[c.args[xi].src.n()].shape.clone();
        let xl = xs.last()?.fixed()?;
        if xa.is_gather() || !unit(xa.forms.last()?, kd) || xl != k || xl % 32 != 0 {
            return None;
        }
        if xa.forms[..xa.forms.len() - 1]
            .iter()
            .any(|f| f.coefs[kd] != 0 || f.coefs[np - 1] != 0)
        {
            return None;
        }
        Some((wi, xi))
    }

    /// See the module docs. Returns the rescheduled graph, `outputs`' ids
    /// in it, and what was done.
    pub fn schedule_block_weights(
        &self,
        outputs: &[TensorId],
    ) -> (TensorGraph, Vec<TensorId>, BlockSchedule) {
        let mut g = self.clone();
        let live = self.live(outputs);
        let n = self.nodes.len();
        let cand: Vec<Option<(usize, usize)>> = (0..n)
            .map(|i| {
                if live[i] {
                    self.int8_candidate(i)
                } else {
                    None
                }
            })
            .collect();
        // a block input every live reader of which is a candidate weight read
        let mut keep: HashMap<usize, bool> = HashMap::new();
        for i in (0..n).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else {
                for d in self.deps(i) {
                    keep.insert(d, false);
                }
                continue;
            };
            for (j, a) in c.args.iter().enumerate() {
                let s = a.src.n();
                if !int8_format(self.nodes[s].precision.storage) {
                    continue;
                }
                let ok = cand[i].is_some_and(|(wi, _)| wi == j);
                let e = keep.entry(s).or_insert(true);
                *e &= ok;
            }
        }
        let mut out = BlockSchedule::default();
        for i in 0..n {
            if int8_format(self.nodes[i].precision.storage)
                && !keep.get(&i).copied().unwrap_or(false)
            {
                g.nodes[i].precision = Precision::default();
                out.demoted.push(TensorId(i as u32));
            }
        }
        // rebuild in order, each copy just before its first reader (passes
        // assume a node's args come before it)
        let mut ng = TensorGraph::new();
        ng.n_inputs = g.n_inputs;
        let mut map: Vec<TensorId> = Vec::with_capacity(n);
        let mut copies: HashMap<usize, TensorId> = HashMap::new();
        for (i, &ci) in cand.iter().enumerate() {
            let sched = ci.filter(|&(wi, _)| {
                let w = self.comp_at(i).unwrap().args[wi].src.n();
                keep.get(&w).copied().unwrap_or(false)
            });
            let mut node = g.nodes[i].clone();
            node.op = match &g.nodes[i].op {
                Op::Comp(c) => {
                    let mut c = c.clone();
                    for a in &mut c.args {
                        a.src = map[a.src.n()];
                    }
                    if let Some((_, xi)) = sched {
                        let x = self.comp_at(i).unwrap().args[xi].src.n();
                        let xq = *copies.entry(x).or_insert_with(|| {
                            let shape = g.nodes[x].shape.clone();
                            let r = shape.len();
                            let mut body = tang_expr::ExprGraph::new();
                            let v = body.var(0);
                            let t = ng
                                .push_comp(Comp {
                                    out: shape,
                                    tuple: 0,
                                    red: Vec::new(),
                                    args: vec![Arg {
                                        src: map[x],
                                        view: View::identity(r, r),
                                    }],
                                    body,
                                    body_outs: vec![v],
                                    monoid: None,
                                })
                                .expect("a copy of the activation");
                            ng.nodes[t.n()].precision = Precision::storage(Format::Q8x32);
                            out.copies.push(t);
                            t
                        });
                        c.args[xi].src = xq;
                        node.int_mma = Some(IntMma {
                            stage: Stage::Gemv,
                            ksplit: 1,
                            pos_tile: 1,
                        });
                        out.gemvs += 1;
                    }
                    Op::Comp(c)
                }
                Op::Reshape(src) => Op::Reshape(map[src.n()]),
                Op::Scan(sc) => {
                    let mut sc = sc.clone();
                    for t in &mut sc.inputs {
                        *t = map[t.n()];
                    }
                    Op::Scan(sc)
                }
                Op::Operator(o) => {
                    let mut o = o.clone();
                    for t in &mut o.inputs {
                        *t = map[t.n()];
                    }
                    Op::Operator(o)
                }
                op @ (Op::Input(_) | Op::Iota(_)) => op.clone(),
            };
            if let Some(sg) = &mut node.segment {
                sg.key = map[sg.key.n()];
            }
            ng.nodes.push(node);
            map.push(TensorId((ng.nodes.len() - 1) as u32));
        }
        out.demoted = out.demoted.iter().map(|t| map[t.n()]).collect();
        (ng, outputs.iter().map(|o| map[o.n()]).collect(), out)
    }
}
