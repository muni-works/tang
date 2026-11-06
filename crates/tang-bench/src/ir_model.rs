//! Whole-model compilation: `layers` stacked blocks, an MSE loss, the
//! full backward and an AdamW update of every weight as ONE
//! `TensorGraph`, lowered to one program and run as one submission with
//! a liveness-planned memory layout (`TensorGraph::plan_memory`).
//!
//! Input order: `x`, `cos`, `sin`, then per layer the block's inputs in
//! `ir_block::INPUTS` order (minus `x`, `cos`, `sin`), then `target`,
//! then `(m, v)` per trained weight, then `bc = [1 − β1ᵗ, 1 − β2ᵗ]`.
use tang::Scalar;
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{numel, Arg, Form, Monoid, Op, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use super::ir_block::{block_into, BlockDims};

#[derive(Clone, Copy, Debug)]
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// One trained weight and its optimizer state, before and after the step.
#[derive(Clone, Copy, Debug)]
pub struct ParamState {
    pub p: TensorId,
    pub m: TensorId,
    pub v: TensorId,
    pub p2: TensorId,
    pub m2: TensorId,
    pub v2: TensorId,
}

pub struct TrainStep {
    pub graph: TensorGraph,
    pub loss: TensorId,
    /// The last block's output (the stacked forward).
    pub hidden: TensorId,
    /// Every block's output (the residual stream between blocks), then
    /// the loss gradient w.r.t. each.
    pub boundaries: Vec<TensorId>,
    pub params: Vec<ParamState>,
    /// Initial value of every graph input, by input index.
    pub inputs: Vec<Vec<f32>>,
}

impl TrainStep {
    /// `[loss, p2…, m2…, v2…]`.
    pub fn outputs(&self) -> Vec<TensorId> {
        let mut o = vec![self.loss];
        o.extend(self.params.iter().map(|s| s.p2));
        o.extend(self.params.iter().map(|s| s.m2));
        o.extend(self.params.iter().map(|s| s.v2));
        o
    }

    /// Graph input index of `t`.
    pub fn input_index(g: &TensorGraph, t: TensorId) -> usize {
        match g.nodes()[t.index()].op {
            Op::Input(k) => k,
            _ => panic!("not an input"),
        }
    }
}

fn rand(n: usize, scale: f32, rng: &mut super::Rng) -> Vec<f32> {
    (0..n).map(|_| rng.f32() * scale).collect()
}

/// `layers` blocks, loss `mean((h − target)²)`, backward w.r.t. every
/// projection weight, and an AdamW step for each.
pub fn train_step(dims: &BlockDims, layers: usize, adam: AdamW) -> TrainStep {
    let mut g = TensorGraph::new();
    let mut rng = super::Rng::new(11);
    let (seq, d) = (dims.seq, dims.d_model);
    let half = dims.head_dim / 2;
    let x = g.input(&[seq, d]);
    let cos = g.input(&[seq, half]);
    let sin = g.input(&[seq, half]);
    let mut weights = Vec::new();
    let mut h = x;
    let mut boundaries = Vec::new();
    for _ in 0..layers {
        let (nodes, w) = block_into(&mut g, h, Some((cos, sin)), dims);
        weights.extend(w);
        h = nodes.out;
        boundaries.push(h);
    }
    let shape = g.shape(h).to_vec();
    let target = g.input(&shape);
    let n = numel(&shape) as f64;
    let sq = g
        .map(&[h, target], |v| (v[0] - v[1]) * (v[0] - v[1]))
        .unwrap();
    let flat = g.reshape(sq, &[numel(&shape)]).unwrap();
    let sum = g.reduce(flat, 0, Monoid::sum()).unwrap();
    let loss = g.map(&[sum], |v| v[0] / ExprId::from_f64(n)).unwrap();
    // the residual stream's gradients come along, so they can be barriers
    let wrt: Vec<TensorId> = weights.iter().chain(&boundaries).copied().collect();
    let mut grads = g.backward(loss, &wrt).unwrap();
    let dboundaries = grads.split_off(weights.len());
    boundaries.extend(dboundaries);

    let states: Vec<(TensorId, TensorId)> = weights
        .iter()
        .map(|&w| {
            let s = g.shape(w).to_vec();
            (g.input(&s), g.input(&s))
        })
        .collect();
    let bc = g.input(&[2]);
    let AdamW {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    } = adam;
    let mut params = Vec::new();
    for ((&p, &gr), &(m, v)) in weights.iter().zip(&grads).zip(&states) {
        let shape = g.shape(p).to_vec();
        let rank = shape.len();
        let id = |t: TensorId| Arg {
            src: t,
            view: View::identity(rank, rank),
        };
        let bcv = |k: i64| Arg {
            src: bc,
            view: View {
                gather: None,
                guard: false,
                forms: vec![Form {
                    coefs: vec![0; rank],
                    offset: k,
                }],
            },
        };
        let m2 = g
            .map(&[m, gr], |x| {
                x[0] * ExprId::from_f64(beta1) + x[1] * ExprId::from_f64(1.0 - beta1)
            })
            .unwrap();
        let v2 = g
            .map(&[v, gr], |x| {
                x[0] * ExprId::from_f64(beta2) + x[1] * x[1] * ExprId::from_f64(1.0 - beta2)
            })
            .unwrap();
        let p2 = g
            .comp(
                &shape,
                &[],
                vec![id(p), id(m2), id(v2), bcv(0), bcv(1)],
                None,
                |x| {
                    let m_hat = x[1] / x[3];
                    let v_hat = x[2] / x[4];
                    let u = m_hat / (v_hat.sqrt() + ExprId::from_f64(eps));
                    vec![x[0] - ExprId::from_f64(lr) * (u + ExprId::from_f64(weight_decay) * x[0])]
                },
            )
            .unwrap();
        params.push(ParamState {
            p,
            m,
            v,
            p2,
            m2,
            v2,
        });
    }

    // initial values, by input index
    let n_inputs = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input(_)))
        .count();
    let mut inputs: Vec<Vec<f32>> = vec![Vec::new(); n_inputs];
    for (i, node) in g.nodes().iter().enumerate() {
        let Op::Input(k) = node.op else { continue };
        let len = numel(&node.shape);
        let t = TensorId::from_index(i);
        inputs[k] = if t == x || t == target {
            rand(len, 1.0, &mut rng)
        } else if t == cos || t == sin {
            (0..len)
                .map(|j| {
                    let (pos, i) = (j / half, j % half);
                    let th = pos as f64 / 10000f64.powf(2.0 * i as f64 / dims.head_dim as f64);
                    (if t == cos { th.cos() } else { th.sin() }) as f32
                })
                .collect()
        } else if t == bc {
            vec![(1.0 - beta1) as f32, (1.0 - beta2) as f32]
        } else if let Some(w) = weights.iter().find(|&&w| w == t) {
            let fan_in = g.shape(*w)[1].n() as f32;
            rand(len, 1.0 / fan_in.sqrt(), &mut rng)
        } else {
            // biases (unused without `bias`) and optimizer state
            vec![0.0; len]
        };
    }
    TrainStep {
        graph: g,
        loss,
        hidden: h,
        boundaries,
        params,
        inputs,
    }
}

/// The block's pass pipeline (`ir_block::compiled_block`) over any graph.
pub fn compile(g: &TensorGraph, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    compile_with_barriers(g, outputs, &[])
}

/// [`compile`] keeping `barriers` materialized: the passes never splice
/// an output, so the barriers are passed as extra outputs and dropped
/// from the returned list.
///
/// Why: across blocks the residual stream is a chain of maps
/// (`h₁ = h₀ + attn₀ + ffn₀`, …) each read point-for-point, which
/// `substitute_maps` happily splices, so every reader of `h_N` recomputes
/// `x + Σ` over all earlier blocks: O(N) buffer reads per kernel, O(N²)
/// overall, and more storage bindings than wgpu allows (16) from N = 2.
/// Materializing each block's output bounds the chain to one block.
pub fn compile_with_barriers(
    g: &TensorGraph,
    outputs: &[TensorId],
    barriers: &[TensorId],
) -> (TensorGraph, Vec<TensorId>) {
    let all: Vec<TensorId> = outputs.iter().chain(barriers).copied().collect();
    let (g, outs) = g.stabilize_softmax(&all);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, mut outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    outs.truncate(outputs.len());
    (g, outs)
}

/// The graph's node for input `k`.
pub fn input_node(g: &TensorGraph, k: usize) -> TensorId {
    let i = g
        .nodes()
        .iter()
        .position(|n| matches!(n.op, Op::Input(j) if j == k))
        .expect("input");
    TensorId::from_index(i)
}

/// In-place offers for a compiled step: each `p2/m2/v2` into its input.
/// `outs` is `[loss, p2…, m2…, v2…]` as returned by [`compile`], `orig`
/// the step it was compiled from.
pub fn donations(
    g: &TensorGraph,
    orig: &TrainStep,
    outs: &[TensorId],
) -> Vec<(TensorId, TensorId)> {
    let n = orig.params.len();
    let og = &orig.graph;
    let mut d = Vec::new();
    for (i, s) in orig.params.iter().enumerate() {
        for (j, src) in [s.p, s.m, s.v].into_iter().enumerate() {
            let k = TrainStep::input_index(og, src);
            d.push((outs[1 + j * n + i], input_node(g, k)));
        }
    }
    d
}

/// Kernel sources with node numbers renamed in order of first
/// appearance, so the same kernel in different layers is one string
/// (one compile).
pub fn canonical_source(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut names: Vec<String> = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_digit()
            && i > 0
            && (bytes[i - 1].is_ascii_alphabetic() || bytes[i - 1] == b'_')
        {
            // identifier with a numeric suffix: find the identifier start
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let num = &src[i..j];
            let k = match names.iter().position(|n| n == num) {
                Some(k) => k,
                None => {
                    names.push(num.to_string());
                    names.len() - 1
                }
            };
            out.push_str(&format!("{k}"));
            i = j;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Sizes and counts for a report line.
pub struct ProgramStats {
    pub kernels: usize,
    pub unique_kernels: usize,
    /// distinct sources as emitted: what a source-keyed cache compiles
    pub distinct_sources: usize,
    pub buffers: usize,
}

pub fn program_stats(prog: &WgslProgram) -> ProgramStats {
    let mut uniq: Vec<String> = prog
        .kernels
        .iter()
        .map(|k| canonical_source(&k.source))
        .collect();
    uniq.sort();
    uniq.dedup();
    let distinct: std::collections::HashSet<&str> =
        prog.kernels.iter().map(|k| k.source.as_str()).collect();
    ProgramStats {
        kernels: prog.kernels.len(),
        unique_kernels: uniq.len(),
        distinct_sources: distinct.len(),
        buffers: prog.buffers.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_compile::tensor::memplan::MemoryPlan;

    fn tiny() -> BlockDims {
        BlockDims {
            seq: 4,
            kv_len: 4,
            rope: true,
            bias: false,
            d_model: 8,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 4,
            ff_dim: 6,
        }
    }

    /// Two stacked blocks plus AdamW: the compiled graph computes the same
    /// updated weights as the uncompiled one, and the plan finds every
    /// in-place update.
    #[test]
    fn two_layer_step_compiles_and_plans() {
        let step = train_step(&tiny(), 2, AdamW::default());
        let outs = step.outputs();
        let inputs: Vec<Vec<f64>> = step
            .inputs
            .iter()
            .map(|v| v.iter().map(|&x| x as f64).collect())
            .collect();
        let want = step.graph.eval(&inputs, &outs);
        let (g, couts) = compile(&step.graph, &outs);
        let got = g.eval(&inputs, &couts);
        for (a, b) in want.iter().zip(&got) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= 1e-6 * (1.0 + x.abs()), "{x} vs {y}");
            }
        }
        // the update moved the weights
        let p0 = &inputs[TrainStep::input_index(&step.graph, step.params[0].p)];
        assert!(p0.iter().zip(&want[1]).any(|(a, b)| a != b));
        let prog = g.to_wgsl(&couts);
        let plan = g.plan_memory(&prog, &couts, &donations(&g, &step, &couts));
        assert_eq!(plan.inplace.len(), 3 * step.params.len());
        assert!(plan.stats.planned_bytes < MemoryPlan::naive(&prog).stats.planned_bytes);
    }

    /// Layers are the same kernels at other node numbers: emitted sources
    /// are canonical, so 30 stacked layers have no more distinct sources
    /// than 4 (one compile per distinct kernel, not per layer).
    #[test]
    fn distinct_sources_do_not_grow_with_depth() {
        let distinct = |layers| {
            let step = train_step(&tiny(), layers, AdamW::default());
            let outs = step.outputs();
            let (g, couts) = compile_with_barriers(&step.graph, &outs, &step.boundaries);
            let s = program_stats(&g.to_wgsl(&couts));
            assert!(s.kernels > 6 * layers);
            let c = program_stats(&g.to_cuda(&couts));
            (s.distinct_sources, c.distinct_sources)
        };
        let (d4, d30) = (distinct(4), distinct(30));
        assert_eq!(
            d4, d30,
            "distinct (wgsl, cuda) sources: 4 layers {d4:?}, 30 layers {d30:?}"
        );
    }

    #[test]
    fn canonical_source_renames_numbers() {
        assert_eq!(
            canonical_source("b12 point_12(b7) 3u"),
            canonical_source("b40 point_40(b9) 3u")
        );
    }
}
