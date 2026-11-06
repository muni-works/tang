//! Memory placement as a schedule axis.
//!
//! Every intermediate of a graph lives somewhere: written to device memory
//! (`Hbm`), kept in cache by residency hints (`L2`), staged per tile in
//! workgroup memory (`Smem`), carried in registers as running reduction
//! state (`Reg`), or never stored at all (`Recompute`). And every edge
//! between two comprehensions is either a kernel boundary (`Split`) or not
//! (`Fuse`).
//!
//! The algebra-preserving rewrites tang already has are what particular
//! settings of this axis produce:
//!
//! | setting | rewrite |
//! |---|---|
//! | normalizer = `Reg` | [`stabilize_softmax`](TensorGraph::stabilize_softmax): `max` and `Σ exp` become one online-softmax state, carried in registers by the consumer reduction |
//! | pointwise = `Recompute` | [`substitute_maps`](TensorGraph::substitute_maps) + inlining: a map between reductions is re-evaluated in every reader instead of written |
//! | contraction = `Smem` | [`schedule_nested_contractions_scalar`](TensorGraph::schedule_nested_contractions_scalar): a short contraction (`S = QKᵀ`) is computed per staged chunk into workgroup memory |
//! | boundary = `Fuse` | [`merge_siblings`](TensorGraph::merge_siblings) + [`schedule_auto`](TensorGraph::schedule_auto)'s producer/epilogue fusion |
//!
//! So the named attention algorithms are points on one axis over one graph:
//! naive (everything `Hbm`, every edge `Split`), activation checkpointing
//! (probabilities recomputed), FlashAttention-2 (scores in `Smem`, softmax
//! state in `Reg`, fused). FA3 is FA2's placement plus asynchronous
//! staging (TMA / wgmma), which is a lowering of `Smem`, not a new
//! placement; a whole-layer megakernel is `boundary = Fuse` at every edge
//! of the layer, including across the matmuls around attention.
//!
//! Roles are found structurally, not by name: a *contraction* is a short
//! sum of products (reduce dim ≤ 128), a *pointwise* intermediate is a map
//! with no reduce dims, a *normalizer* is a reduction another reduction's
//! body subtracts or divides by (softmax's max and Σ). Per-intermediate
//! overrides (`overrides`, ids of the scheduled graph) refine a role-level
//! setting, which is what an autotuner searches.

use super::*;

/// Where one intermediate lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Place {
    /// Written to and read back from device memory.
    Hbm,
    /// Device memory with a cache-residency hint. No backend here exposes
    /// residency control (Hopper's access-policy window would), so it
    /// lowers as `Hbm` today; kept so the axis is complete.
    L2,
    /// Staged per tile in workgroup / shared memory.
    Smem,
    /// Carried in registers as running reduction state.
    Reg,
    /// Never stored: re-evaluated in each reader.
    Recompute,
}

/// What happens at an edge between two comprehensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Boundary {
    /// A kernel boundary: the producer finishes before the consumer runs.
    Split,
    /// Producer and consumer (or two siblings over the same space) share a kernel.
    Fuse,
}

/// One point on the memory-placement axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemorySchedule {
    /// Short contractions read by tiles (`S = QKᵀ`, `dP = dO·Vᵀ`).
    pub contraction: Place,
    /// Maps between reductions (`P = exp(S − m)`).
    pub pointwise: Place,
    /// Normalizer reductions (softmax's max and Σ exp).
    pub normalizer: Place,
    /// Every comp edge.
    pub boundary: Boundary,
    /// Per-intermediate refinements, ids of the scheduled graph.
    pub overrides: Vec<(TensorId, Place)>,
}

impl MemorySchedule {
    /// Everything in device memory, every edge a kernel boundary.
    pub fn naive() -> Self {
        Self {
            contraction: Place::Hbm,
            pointwise: Place::Hbm,
            normalizer: Place::Hbm,
            boundary: Boundary::Split,
            overrides: Vec::new(),
        }
    }

    /// Activation checkpointing: maps between reductions (attention's
    /// probabilities) are recomputed by every reader, the forward and the
    /// backward alike, rather than stored.
    pub fn checkpoint() -> Self {
        Self {
            pointwise: Place::Recompute,
            ..Self::naive()
        }
    }

    /// FlashAttention-2: scores staged per tile in workgroup memory,
    /// softmax state carried in registers, probabilities recomputed, fused.
    pub fn flash() -> Self {
        Self {
            contraction: Place::Smem,
            pointwise: Place::Recompute,
            normalizer: Place::Reg,
            boundary: Boundary::Fuse,
            overrides: Vec::new(),
        }
    }

    /// The named points, in dial order.
    pub fn named() -> Vec<(&'static str, MemorySchedule)> {
        vec![
            ("naive", Self::naive()),
            ("checkpoint", Self::checkpoint()),
            ("flash", Self::flash()),
        ]
    }

    /// Role-level settings worth searching. Combinations whose rewrites
    /// would not change anything (`contraction = Smem` without fusion has
    /// no tile to nest in) are left out.
    pub fn candidates() -> Vec<MemorySchedule> {
        let mut out = Vec::new();
        for boundary in [Boundary::Split, Boundary::Fuse] {
            for normalizer in [Place::Hbm, Place::Reg] {
                for pointwise in [Place::Hbm, Place::Recompute] {
                    let cs: &[Place] = if boundary == Boundary::Fuse {
                        &[Place::Hbm, Place::Smem]
                    } else {
                        &[Place::Hbm]
                    };
                    for &contraction in cs {
                        out.push(MemorySchedule {
                            contraction,
                            pointwise,
                            normalizer,
                            boundary,
                            overrides: Vec::new(),
                        });
                    }
                }
            }
        }
        out
    }
}

/// A graph scheduled under a [`MemorySchedule`]: lower with
/// `to_wgsl_with(&outputs, nest)` (or `to_msl_with`).
pub struct Placed {
    pub graph: TensorGraph,
    pub outputs: Vec<TensorId>,
    /// Lower with short contractions nested in tiles (`contraction = Smem`).
    pub nest: bool,
}

impl TensorGraph {
    /// Apply a memory placement: run exactly the rewrites its settings
    /// imply and mark each intermediate's materialization. With
    /// [`MemorySchedule::flash`] this is the training pass pipeline
    /// (`stabilize_softmax` … `schedule_auto`), with
    /// [`MemorySchedule::naive`] it is the graph as written.
    pub fn place(&self, outputs: &[TensorId], s: &MemorySchedule) -> Placed {
        let mut g = self.clone();
        let mut outs = outputs.to_vec();
        let mut step = |f: &dyn Fn(&TensorGraph, &[TensorId]) -> (TensorGraph, Vec<TensorId>)| {
            let (ng, no) = f(&g, &outs);
            g = ng;
            outs = no;
        };
        if s.normalizer == Place::Reg {
            step(&|g, o| g.stabilize_softmax(o));
        }
        if s.pointwise == Place::Recompute {
            step(&|g, o| g.substitute_maps(o));
        }
        // algebra-neutral clean-ups any placement may run
        if s != &MemorySchedule::naive() {
            step(&|g, o| g.eliminate_common(o));
        }
        if s.boundary == Boundary::Fuse {
            step(&|g, o| g.merge_siblings(o));
            step(&|g, o| g.relayout(o));
            step(&|g, o| g.merge_reduce_dims(o));
            step(&|g, o| g.split_reduce_dims(o));
        }
        let live = g.live(&outs);
        let is_output = |i: usize| outs.iter().any(|o| o.n() == i);
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); g.nodes.len()];
        for j in (0..g.nodes.len()).filter(|&j| live[j]) {
            for d in g.deps(j) {
                readers[d].push(j);
            }
        }
        match s.boundary {
            Boundary::Fuse => g.schedule_auto(&outs),
            Boundary::Split => {
                // one kernel per comp, except maps placed `Recompute`,
                // evaluated in every reader that reduces (a map read by a
                // map would just move the buffer one node down)
                for i in (0..g.nodes.len()).filter(|&i| live[i] && !is_output(i)) {
                    let Some(c) = g.comp_at(i) else { continue };
                    let map = c.red.is_empty();
                    let reduced_by_all = !readers[i].is_empty()
                        && readers[i]
                            .iter()
                            .all(|&r| g.comp_at(r).is_some_and(|rc| rc.monoid.is_some()));
                    g.nodes[i].inline =
                        map && s.pointwise == Place::Recompute && reduced_by_all;
                }
            }
        }
        if s.pointwise == Place::Hbm {
            for i in (0..g.nodes.len()).filter(|&i| live[i]) {
                if g.comp_at(i).is_some_and(|c| c.red.is_empty())
                    && g.tiled_reduction(i).is_none_or(|r| r == i)
                {
                    g.nodes[i].inline = false;
                }
            }
        }
        for &(t, p) in &s.overrides {
            if t.n() < g.nodes.len() && !is_output(t.n()) && g.comp_at(t.n()).is_some() {
                g.nodes[t.n()].inline = matches!(p, Place::Recompute | Place::Reg | Place::Smem);
            }
        }
        Placed {
            graph: g,
            outputs: outs,
            nest: s.contraction == Place::Smem,
        }
    }

    /// Where each live intermediate of a scheduled graph ended up: a
    /// buffer (`Hbm`), recomputed in its readers, or nested per tile.
    /// Inputs and outputs are left out.
    pub fn placements(&self, outputs: &[TensorId]) -> Vec<(TensorId, Place)> {
        let live = self.live(outputs);
        (0..self.nodes.len())
            .filter(|&i| live[i] && !outputs.iter().any(|o| o.n() == i))
            .filter_map(|i| {
                let c = self.comp_at(i)?;
                let p = if !self.nodes[i].inline {
                    Place::Hbm
                } else if c.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::OnlineSoftmax) {
                    Place::Reg
                } else {
                    Place::Recompute
                };
                Some((TensorId(i as u32), p))
            })
            .collect()
    }

    /// Per-intermediate memory refinements worth trying around a role-level
    /// placement: each live non-output comp flipped between a buffer and
    /// recomputation. Pair with [`schedule_candidates`](Self::schedule_candidates)
    /// (how a kernel runs) in an autotuner.
    pub fn placement_candidates(&self, outputs: &[TensorId]) -> Vec<(TensorId, Place)> {
        self.placements(outputs)
            .into_iter()
            .map(|(t, p)| {
                let flip = if p == Place::Hbm { Place::Recompute } else { Place::Hbm };
                (t, flip)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_expr::ExprId;

    /// O = softmax(QKᵀ)V with a causal mask.
    fn attention(n: usize, d: usize) -> (TensorGraph, Vec<TensorId>) {
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, d]);
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap();
        let ii = g.iota(&[n, n], 0);
        let jj = g.iota(&[n, n], 1);
        let s = g
            .map(&[s, ii, jj], |x| {
                let above = x[2] - x[1]; // > 0 when j > i
                let bias = tang_expr::with_graph(|eg| {
                    let neg = eg.lit(-1e30);
                    eg.select(above, neg, ExprId::ZERO)
                });
                x[0] + bias
            })
            .unwrap();
        let m = g.reduce(s, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g
            .map(&[s, mb], |x| {
                let t = x[0] - x[1];
                tang_expr::with_graph(|eg| {
                    let c = eg.lit(std::f64::consts::LOG2_E);
                    let u = eg.mul(t, c);
                    eg.exp2(u)
                })
            })
            .unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let o = g.matmul(p, v).unwrap();
        let lb = g.broadcast(l, &[n, d]).unwrap();
        let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
        (g, vec![y])
    }

    fn nxn(p: &Placed, n: usize) -> usize {
        p.graph
            .to_wgsl_with(&p.outputs, p.nest)
            .buffers
            .iter()
            .filter(|b| b.input.is_none() && b.len >= n * n)
            .count()
    }

    #[test]
    fn named_points_agree_and_move_nxn_intermediates_out_of_memory() {
        let (n, d) = (32, 8);
        let (g, outs) = attention(n, d);
        let mut rng = 1u64;
        let mut r = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
        };
        let inputs: Vec<Vec<f64>> = (0..3).map(|_| (0..n * d).map(|_| r()).collect()).collect();
        let want = g.eval(&inputs, &outs);
        let mut big = Vec::new();
        for (name, s) in MemorySchedule::named() {
            let p = g.place(&outs, &s);
            let got = p.graph.eval(&inputs, &p.outputs);
            let err = got[0]
                .iter()
                .zip(&want[0])
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-9, "{name}: {err}");
            big.push(nxn(&p, n));
        }
        // naive writes S, masked S and P; checkpointing drops P; flash none
        assert!(big[0] > big[1], "{big:?}");
        assert_eq!(big[2], 0, "{big:?}");
    }

    #[test]
    fn flash_is_the_training_pipeline() {
        let (g, outs) = attention(64, 16);
        let p = g.place(&outs, &MemorySchedule::flash());
        let (a, ao) = g.stabilize_softmax(&outs);
        let (a, ao) = a.substitute_maps(&ao);
        let (a, ao) = a.eliminate_common(&ao);
        let (a, ao) = a.merge_siblings(&ao);
        let (a, ao) = a.relayout(&ao);
        let (a, ao) = a.merge_reduce_dims(&ao);
        let (mut a, ao) = a.split_reduce_dims(&ao);
        a.schedule_auto(&ao);
        assert_eq!(a.count_comps(&ao), p.graph.count_comps(&p.outputs));
        assert_eq!(
            a.to_wgsl_with(&ao, true).kernels.len(),
            p.graph.to_wgsl_with(&p.outputs, p.nest).kernels.len()
        );
    }

    #[test]
    fn candidates_cover_the_named_points() {
        let c = MemorySchedule::candidates();
        for (_, s) in MemorySchedule::named() {
            assert!(c.contains(&s));
        }
    }
}
