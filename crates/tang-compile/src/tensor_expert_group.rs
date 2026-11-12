//! Expert grouping for batched decode: a contraction whose weight slice is
//! picked per position by an index (a routed expert's rows, `W[idx(p), r,
//! k]`), over enough positions (streams × routed slots), is rewritten to
//! run over the positions sorted by that index, so positions that share a
//! slice are consecutive and share one weight read (the int8 tensor-core
//! schedule's runs, [`BlockOpts::mma_min`](super::int8_sched::BlockOpts)):
//!
//! ```text
//! key[p]   = idx(p)·P + p            (p flattened row-major over the positions)
//! rank[p]  = #{p' : key[p'] < key[p]}
//! sidx[q]  = idx at rank q,  srow[q] = X's row at rank q
//! C'[q, r] = Σ_k W[sidx[q], r, k] · Xrows[srow[q], k]
//! C[p, r]  = C'[rank[p], r]          (a gathered read, inlined into readers)
//! ```
//!
//! All ordinary comps (`O(P²)` compares for the rank and the two picks, a
//! few hundred thousand at `P = 640`). The keys are distinct, so the sort is
//! a permutation and the result is the same sums as before, term for term.

use super::{Arg, Dim, Form, Gather, Monoid, MonoidKind, Op, TensorGraph, TensorId, View};
use tang::Scalar;
use tang_expr::{ExprId, Node};

/// What [`TensorGraph::schedule_expert_groups`] did.
#[derive(Debug, Default)]
pub struct ExpertGroups {
    /// Contractions now over sorted positions: `(sorted comp, rank)`.
    pub grouped: Vec<(TensorId, TensorId)>,
}

/// The parts of a candidate comp.
struct Cand {
    wi: usize,
    xi: usize,
    ii: usize,
    /// Point extents; positions are all but the last (the weight row).
    pts: Vec<usize>,
    k: usize,
}

impl TensorGraph {
    fn expert_candidate(&self, i: usize, min_positions: usize, blocks_only: bool) -> Option<Cand> {
        let c = self.comp_at(i)?;
        if c.tuple != 0
            || c.red.len() != 1
            || c.args.len() != 3
            || c.monoid.as_ref().map(|m| m.kind()) != Some(MonoidKind::Sum)
            || self.nodes[i].int_mma.is_some()
        {
            return None;
        }
        let pts: Vec<usize> = c.out.iter().map(|d| d.fixed()).collect::<Option<_>>()?;
        let k = c.red[0].fixed()?;
        let np = pts.len();
        let m = np.checked_sub(1)?;
        let npos: usize = pts[..m].iter().product();
        if m == 0 || npos < min_positions.max(2) {
            return None;
        }
        let nodes = c.body.nodes_slice();
        let Node::Mul(x, y) = nodes[c.body_outs[0].index() as usize] else {
            return None;
        };
        let var = |e: ExprId| match nodes[e.index() as usize] {
            Node::Var(v) => Some(v as usize),
            _ => None,
        };
        let (a, b) = (var(x)?, var(y)?);
        let (wi, xi) = match (c.args[a].view.is_gather(), c.args[b].view.is_gather()) {
            (true, false) => (a, b),
            (false, true) => (b, a),
            _ => return None,
        };
        if !matches!(self.nodes[c.args[wi].src.n()].op, Op::Input(_)) {
            return None;
        }
        if blocks_only && !self.nodes[c.args[wi].src.n()].precision.storage.is_block() {
            return None;
        }
        let ii = c.args[wi].view.gather.as_ref()?.arg;
        if ii == xi || ii == wi {
            return None;
        }
        let rank = np + 1;
        let (row, kd) = (m, np);
        let only_pos = |f: &Form| f.coefs[row] == 0 && f.coefs[kd] == 0;
        // W: [.., r, k]; leading dims constant but for the gather
        let w = &c.args[wi].view;
        let wn = w.forms.len();
        let mut rowf = Form::unit(rank, row);
        rowf.offset = w.forms.get(wn.checked_sub(2)?)?.offset;
        if wn < 2
            || w.guard
            || w.forms[wn - 1] != Form::unit(rank, kd)
            || w.forms[wn - 2] != rowf
            || !w.forms[..wn - 2]
                .iter()
                .all(|f| f.coefs.iter().all(|&q| q == 0))
            || w.gather.as_ref()?.coefs[wn - 2..].iter().any(|&q| q != 0)
        {
            return None;
        }
        // the index: per position
        let iv = &c.args[ii].view;
        if iv.is_gather() || iv.guard || !iv.forms.iter().all(only_pos) {
            return None;
        }
        // X: [.., k] with k unit, the other dims per position only
        let xv = &c.args[xi].view;
        let xs: Vec<usize> = self.nodes[c.args[xi].src.n()]
            .shape
            .iter()
            .map(|d| d.fixed())
            .collect::<Option<_>>()?;
        let xn = xv.forms.len();
        let mut kf = Form::unit(rank, kd);
        kf.offset = xv.forms[xn - 1].offset;
        if xv.guard
            || xv.forms[xn - 1] != kf
            || !xv.forms[..xn - 1].iter().all(only_pos)
            || xs.len() != xn
        {
            return None;
        }
        Some(Cand { wi, xi, ii, pts, k })
    }

    /// See the module docs: every contraction with a gathered weight over
    /// at least `min_positions` positions, rewritten over positions sorted
    /// by its index. `blocks_only`: only block-format weights (the int8
    /// schedule's runs); an f32 weight grouped goes to the generic lowering,
    /// slower than its ungrouped gather (coder-next blk.46's Q6_K down,
    /// demoted to f32: 2.7 ms at B = 64). Returns the graph, `outputs`' ids
    /// in it, and what was done.
    pub fn schedule_expert_groups(
        &self,
        outputs: &[TensorId],
        min_positions: usize,
        blocks_only: bool,
    ) -> (TensorGraph, Vec<TensorId>, ExpertGroups) {
        let live = self.live(outputs);
        let n = self.nodes.len();
        let mut ng = TensorGraph::new();
        ng.n_inputs = self.n_inputs;
        let mut map: Vec<TensorId> = Vec::with_capacity(n);
        let mut out = ExpertGroups::default();
        for i in 0..n {
            if live[i] {
                if let Some(cd) = self.expert_candidate(i, min_positions, blocks_only) {
                    let t = self.group_one(&mut ng, &map, i, &cd, &mut out);
                    map.push(t);
                    continue;
                }
            }
            let mut node = self.nodes[i].clone();
            node.op = match &self.nodes[i].op {
                Op::Comp(c) => {
                    let mut c = c.clone();
                    for a in &mut c.args {
                        a.src = map[a.src.n()];
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
        (ng, outputs.iter().map(|o| map[o.n()]).collect(), out)
    }

    fn group_one(
        &self,
        ng: &mut TensorGraph,
        map: &[TensorId],
        i: usize,
        cd: &Cand,
        out: &mut ExpertGroups,
    ) -> TensorId {
        let c = self.comp_at(i).unwrap();
        let np = cd.pts.len();
        let m = np - 1;
        let pos = &cd.pts[..m];
        let npos: usize = pos.iter().product();
        let rows = cd.pts[m];
        // row-major strides of the positions
        let strides: Vec<i64> = (0..m)
            .map(|d| pos[d + 1..].iter().product::<usize>() as i64)
            .collect();
        let iotas: Vec<TensorId> = (0..m).map(|d| ng.iota(pos, d)).collect();
        let ident = View::identity(m, m);
        let a = |src: TensorId, view: View| Arg { src, view };
        // a form over the comp's space restricted to the positions
        let on_pos = |f: &Form| Form {
            coefs: f.coefs[..m].to_vec(),
            offset: f.offset,
        };
        // the index per position
        let iv = &c.args[cd.ii].view;
        let idx = ng
            .comp(
                pos,
                &[] as &[usize],
                vec![a(
                    map[c.args[cd.ii].src.n()],
                    View {
                        forms: iv.forms.iter().map(on_pos).collect(),
                        guard: false,
                        gather: None,
                    },
                )],
                None,
                |v| vec![v[0]],
            )
            .expect("expert index");
        // key = idx·P + flat(p)
        let mut kargs = vec![a(idx, ident.clone())];
        kargs.extend(iotas.iter().map(|&t| a(t, ident.clone())));
        let st = strides.clone();
        let key = ng
            .comp(pos, &[] as &[usize], kargs, None, move |v| {
                let mut e = v[0] * ExprId::from_f64(npos as f64);
                for (d, &s) in st.iter().enumerate() {
                    e += v[1 + d] * ExprId::from_f64(s as f64);
                }
                vec![e]
            })
            .expect("sort key");
        // rank[p] = #{p' : key[p'] < key[p]}, space [p | p']
        let rank = ng
            .comp(
                pos,
                pos,
                vec![
                    a(
                        key,
                        View::pick(2 * m, &(0..m).map(Some).collect::<Vec<_>>()),
                    ),
                    a(
                        key,
                        View::pick(2 * m, &(m..2 * m).map(Some).collect::<Vec<_>>()),
                    ),
                ],
                Some(Monoid::sum()),
                |v| {
                    let d = v[0] - v[1];
                    vec![tang_expr::with_graph(|eg| {
                        eg.select(d, ExprId::ONE, ExprId::ZERO)
                    })]
                },
            )
            .expect("sort rank");
        let iq = ng.iota(&[npos], 0);
        // the value at rank q: Σ_p [rank[p] = q]·val[p], space [q | p]
        let at_rank = |ng: &mut TensorGraph, val: TensorId| {
            let rp: Vec<Option<usize>> = (1..=m).map(Some).collect();
            ng.comp(
                &[npos],
                pos,
                vec![
                    a(rank, View::pick(1 + m, &rp)),
                    a(iq, View::pick(1 + m, &[Some(0)])),
                    a(val, View::pick(1 + m, &rp)),
                ],
                Some(Monoid::sum()),
                |v| {
                    let (up, down) = (v[0] - v[1], v[1] - v[0]);
                    let eq = tang_expr::with_graph(|eg| {
                        let inner = eg.select(down, ExprId::ZERO, ExprId::ONE);
                        eg.select(up, ExprId::ZERO, inner)
                    });
                    vec![eq * v[2]]
                },
            )
            .expect("sorted pick")
        };
        let sidx = at_rank(ng, idx);
        // X as rows: [Πxs[..-1], xs[-1]] and the row each position reads
        let xv = &c.args[cd.xi].view;
        let xsrc = c.args[cd.xi].src.n();
        let xs: Vec<usize> = self.nodes[xsrc]
            .shape
            .iter()
            .map(|d| d.fixed().unwrap())
            .collect();
        let xn = xs.len();
        let xrows: usize = xs[..xn - 1].iter().product();
        let xw = xs[xn - 1];
        let xstr: Vec<i64> = (0..xn - 1)
            .map(|d| xs[d + 1..xn - 1].iter().product::<usize>() as i64)
            .collect();
        let xforms: Vec<Form> = xv.forms[..xn - 1].iter().map(on_pos).collect();
        let xrow = ng
            .comp(
                pos,
                &[] as &[usize],
                iotas.iter().map(|&t| a(t, ident.clone())).collect(),
                None,
                move |v| {
                    let mut konst = 0i64;
                    let mut e = ExprId::ZERO;
                    for (f, &s) in xforms.iter().zip(&xstr) {
                        konst += s * f.offset;
                        for (j, &q) in f.coefs.iter().enumerate() {
                            if q != 0 {
                                e += v[j] * ExprId::from_f64((s * q) as f64);
                            }
                        }
                    }
                    vec![e + ExprId::from_f64(konst as f64)]
                },
            )
            .expect("activation row");
        let srow = at_rank(ng, xrow);
        let xflat = ng
            .reshape(map[xsrc], &[Dim::Fixed(xrows), Dim::Fixed(xw)])
            .expect("activation rows");
        // C'[q, r | k]
        let w = &c.args[cd.wi].view;
        let wn = w.forms.len();
        let mut wforms: Vec<Form> = w.forms[..wn - 2]
            .iter()
            .map(|f| Form {
                coefs: vec![0; 3],
                offset: f.offset,
            })
            .collect();
        wforms.push(Form {
            coefs: vec![0, 1, 0],
            offset: w.forms[wn - 2].offset,
        });
        wforms.push(Form::unit(3, 2));
        let wview = View {
            forms: wforms,
            guard: false,
            gather: Some(Gather {
                arg: 2,
                coefs: w.gather.as_ref().unwrap().coefs.clone(),
            }),
        };
        let xview = View {
            forms: vec![
                Form::zero(3),
                Form {
                    coefs: vec![0, 0, 1],
                    offset: xv.forms[xn - 1].offset,
                },
            ],
            guard: false,
            gather: Some(Gather {
                arg: 3,
                coefs: vec![1, 0],
            }),
        };
        let sorted = ng
            .comp(
                &[npos, rows],
                &[cd.k],
                vec![
                    a(map[c.args[cd.wi].src.n()], wview),
                    a(xflat, xview),
                    a(sidx, View::pick(3, &[Some(0)])),
                    a(srow, View::pick(3, &[Some(0)])),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .expect("grouped contraction");
        ng.nodes[sorted.n()].precision = self.nodes[i].precision;
        // C[p, r] = C'[rank[p], r]
        let f0 = Form::zero(np);
        let unsort = ng
            .comp(
                &cd.pts,
                &[] as &[usize],
                vec![
                    a(
                        sorted,
                        View {
                            forms: vec![f0, Form::unit(np, m)],
                            guard: false,
                            gather: Some(Gather {
                                arg: 1,
                                coefs: vec![1, 0],
                            }),
                        },
                    ),
                    a(rank, View::identity(np, m)),
                ],
                None,
                |v| vec![v[0]],
            )
            .expect("unsort");
        out.grouped.push((sorted, rank));
        unsort
    }
}
