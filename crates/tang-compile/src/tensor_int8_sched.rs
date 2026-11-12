//! Schedule GGUF block-format weights onto the int8 GEMV ([`IntMma`],
//! `tensor_cuda_int8.rs`) in any graph: every contraction `Σ_k W[(i,) r,
//! k]·X[…, k]` whose weight is a graph input stored Q4_K / Q5_K / Q8_0 (or
//! DeepSeek's Fp8B32 / Mxfp4) reads a Q8x32 copy of its activation and runs
//! as the int8 GEMV ([`Stage::Gemv`]: decode, few positions). A weight's
//! rows may be picked by a position coordinate (a block-diagonal weight's
//! group) and the activation may be a 32-aligned slice of a wider row. A block-format input that any
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

/// How [`TensorGraph::schedule_block_weights_with`] stages a contraction
/// over `P` positions (the point dims before the weight row: streams ×
/// rows × routed slots). The default is the decode schedule everywhere
/// (the dp4a GEMV, one position a weight read).
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockOpts {
    /// An ungathered weight over `2 ≤ P < mma_min` positions: the GEMV
    /// with up to this many positions on one weight read (0 or 1: off).
    pub pos_tile: usize,
    /// From this many positions (0: never) an ungathered weight, or a
    /// gathered one whose positions are grouped by weight slice (the
    /// activation gathered too, `schedule_expert_groups`), runs on the
    /// int8 tensor cores: one weight read per run of 8 positions.
    pub mma_min: usize,
    /// Warps splitting a tile's `k` on the tensor cores (0: 2).
    pub ksplit: u32,
}

impl BlockOpts {
    /// The batched-decode schedule: `TANG_BATCH_PT` (8),
    /// `TANG_BATCH_MMA_MIN` (16), `TANG_BATCH_KSPLIT` (2).
    pub fn batched() -> Self {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Self {
            pos_tile: env("TANG_BATCH_PT", 8),
            mma_min: env("TANG_BATCH_MMA_MIN", 16),
            ksplit: env("TANG_BATCH_KSPLIT", 2) as u32,
        }
    }

    fn stage(&self, g: &TensorGraph, i: usize, (wi, xi): (usize, usize)) -> IntMma {
        let gemv = IntMma {
            stage: Stage::Gemv,
            ksplit: 1,
            pos_tile: 1,
        };
        let c = g.comp_at(i).unwrap();
        let pts: Vec<usize> = c.points().iter().map(|d| d.fixed().unwrap()).collect();
        let npos: usize = pts[..pts.len() - 1].iter().product();
        let wf = g.nodes[c.args[wi].src.n()].precision.storage;
        if npos <= 1 || matches!(wf, Format::Fp8B32 | Format::Mxfp4) {
            return gemv;
        }
        let mma = IntMma {
            stage: Stage::Mma,
            ksplit: if self.ksplit == 0 { 2 } else { self.ksplit },
            pos_tile: 8,
        };
        let big = self.mma_min > 0 && npos >= self.mma_min;
        match (c.args[wi].view.is_gather(), c.args[xi].view.is_gather()) {
            (true, true) if big => IntMma { pos_tile: 1, ..mma },
            (true, _) => gemv,
            (false, _) if big => mma,
            (false, _) if self.pos_tile > 1 => IntMma {
                pos_tile: npos.min(self.pos_tile) as u32,
                ..gemv
            },
            _ => gemv,
        }
    }
}

fn int8_format(f: Format) -> bool {
    matches!(
        f,
        Format::Q4K | Format::Q5K | Format::Q8_0 | Format::Fp8B32 | Format::Mxfp4
    )
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
        // every other arg is the weight's gather index or the
        // activation's (rows picked by position: expert-grouped positions,
        // `schedule_expert_groups`)
        let index = c.args[wi].view.gather.as_ref().map(|g| g.arg);
        let xindex = c.args[xi].view.gather.as_ref().map(|g| g.arg);
        if (0..c.args.len()).any(|j| j != wi && j != xi && Some(j) != index && Some(j) != xindex) {
            return None;
        }
        let (w, xa) = (&c.args[wi].view, &c.args[xi].view);
        let rank = np + 1;
        let wn = w.forms.len();
        let unit = |f: &super::Form, d: usize| *f == super::Form::unit(rank, d);
        // `d` with coefficient 1 plus position terms only (a block-diagonal
        // weight's group, a grouped activation's slice)
        let on = |f: &super::Form, d: usize| {
            f.coefs[d] == 1 && (np - 1..=kd).all(|j| j == d || f.coefs[j] == 0)
        };
        let free = |f: &super::Form| f.coefs[np - 1] == 0 && f.coefs[kd] == 0;
        let wl = self.nodes[c.args[wi].src.n()].shape.last()?.fixed()?;
        if wn < 2 || !unit(&w.forms[wn - 1], kd) || !on(&w.forms[wn - 2], np - 1) || wl != k {
            return None;
        }
        if !w.forms[..wn - 2].iter().all(free) {
            return None;
        }
        // X: k along its last dim, contiguous, a slice starting at a 32;
        // the copy is Q8x32 in 32s
        let xs = self.nodes[c.args[xi].src.n()].shape.clone();
        let xl = xs.last()?.fixed()?;
        let xf = xa.forms.last()?;
        // a gathered activation picks whole rows: never along k
        let x_gather_ok = xa.gather.as_ref().is_none_or(|g| {
            g.coefs.last() == Some(&0)
                && c.args[g.arg].view.gather.is_none()
                && c.args[g.arg]
                    .view
                    .forms
                    .iter()
                    .all(|f| f.coefs[kd] == 0 && f.coefs[np - 1] == 0)
        });
        if !x_gather_ok
            || !on(xf, kd)
            || xf.offset % 32 != 0
            || xf
                .coefs
                .iter()
                .enumerate()
                .any(|(j, &q)| j != kd && q % 32 != 0)
            || xl % 32 != 0
        {
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
        self.schedule_block_weights_opt(outputs, None)
    }

    /// [`schedule_block_weights`](Self::schedule_block_weights) with a
    /// stage per contraction by its number of positions ([`BlockOpts`]).
    pub fn schedule_block_weights_with(
        &self,
        outputs: &[TensorId],
        bo: &BlockOpts,
    ) -> (TensorGraph, Vec<TensorId>, BlockSchedule) {
        self.schedule_block_weights_opt(outputs, Some(bo))
    }

    fn schedule_block_weights_opt(
        &self,
        outputs: &[TensorId],
        bo: Option<&BlockOpts>,
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
                        node.int_mma = Some(match bo {
                            Some(bo) => bo.stage(self, i, sched.unwrap()),
                            None => many_positions(self, i),
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

/// The schedule of int8 contraction `i`: the decode GEMV for few
/// positions; for a many-row step (`TANG_INT8_ROWS`, default 16, positions
/// or more) the MMA tile (`TANG_INT8_STAGE` `gemv` / `mma` / `fragment` / `tile`,
/// `TANG_INT8_KSPLIT`, `TANG_INT8_POS_TILE` override).
fn many_positions(g: &TensorGraph, i: usize) -> IntMma {
    let env = |k: &str| std::env::var(k).ok();
    let num = |k: &str, d: u32| env(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let pts = super::fixed(g.comp_at(i).unwrap().points());
    let positions: usize = pts[..pts.len() - 1].iter().product();
    let decode = IntMma {
        stage: Stage::Gemv,
        ksplit: 1,
        pos_tile: 1,
    };
    if positions < num("TANG_INT8_ROWS", 16) as usize {
        return decode;
    }
    // the tensor-core tile: 1.78k vs 0.84k tok/s for gemv on coder-next
    // chunked prefill (k = 256, H200)
    let stage = match env("TANG_INT8_STAGE").as_deref() {
        Some("gemv") => Stage::Gemv,
        Some("fragment") => Stage::Fragment,
        Some("tile") => Stage::Tile,
        _ => Stage::Mma,
    };
    IntMma {
        stage,
        ksplit: num("TANG_INT8_KSPLIT", 1),
        pos_tile: num("TANG_INT8_POS_TILE", 1),
    }
}
