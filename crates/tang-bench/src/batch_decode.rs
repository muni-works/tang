//! Batched decode: `B` independent streams, each at its own length, stepped
//! by one derived step (`batch_graph` + `StepOptions::batch`), lowered to
//! CUDA. Checked on the CPU (`cuda_cpu`) against `B` single-stream steps.

use std::collections::HashMap;

use tang_compile::tensor::cuda_gemv::{DecodeOpts, DecodeProgram};
use tang_compile::tensor::step::{
    batch_graph, derive_step_with, seed_cache, Rows, StepGraph, StepInput, StepOptions,
};
use tang_compile::tensor::{Dim, TensorGraph, TensorId};

use crate::derive_all::*;

/// The batched step of `m` (`B` streams, `k = 1`, last row only).
pub fn batched_step(m: &Model) -> (TensorGraph, StepGraph) {
    let (bg, bouts) = batch_graph(&m.g, &[m.logits], "T", "B").unwrap();
    let st = derive_step_with(
        &bg,
        &bouts,
        &spec(m, 1),
        &StepOptions {
            rows: Rows::Last,
            batch: Some("B"),
        },
    )
    .unwrap_or_else(|e| panic!("{} batched: {}", m.name, e.0));
    (bg, st)
}

/// One stream's tokens and its caches after a prefill of `p` tokens on the
/// interpreter, in the batched step's cache layout (`B = 1`).
pub fn seed_stream(
    m: &Model,
    bg: &TensorGraph,
    st: &StepGraph,
    w: &HashMap<String, Vec<f64>>,
    toks: &[f64],
    p: usize,
) -> Vec<Vec<f64>> {
    let ins: Vec<Vec<f64>> = m
        .inputs
        .iter()
        .map(|(n, _)| {
            if n == "tokens" {
                toks[..p].to_vec()
            } else {
                w[n].clone()
            }
        })
        .collect();
    let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
    let v = bg.bind(&[("B", 1), ("T", p)]).unwrap().eval(&ins, &nodes);
    st.caches
        .iter()
        .zip(v)
        .map(|(c, v)| {
            let sh: Vec<usize> = bg
                .shape(c.node)
                .iter()
                .map(|&d| match d {
                    Dim::Sym("T") => p,
                    Dim::Sym("B") => 1,
                    d => d.n(),
                })
                .collect();
            if !c.state {
                return seed_cache(c, &sh, &v, p, m.cap);
            }
            if !bg.shape(c.node).contains(&Dim::Sym("T")) {
                return v;
            }
            let outer: usize = sh[..c.axis].iter().product();
            let inner: usize = sh[c.axis + 1..].iter().product();
            (0..outer)
                .flat_map(|o| {
                    let s0 = (o * p + p - 1) * inner;
                    v[s0..s0 + inner].to_vec()
                })
                .collect()
        })
        .collect()
}

/// The step's inputs for streams `sess` (positions `pos`, caches per stream).
pub fn step_inputs(
    m: &Model,
    st: &StepGraph,
    w: &HashMap<String, Vec<f64>>,
    toks: &[Vec<f64>],
    pos: &[usize],
    caches: &[Vec<Vec<f64>>],
    sess: &[usize],
) -> Vec<Vec<f64>> {
    st.inputs
        .iter()
        .map(|&x| match x {
            StepInput::Pos => sess.iter().map(|&b| pos[b] as f64).collect(),
            StepInput::New(_) => sess.iter().map(|&b| toks[b][pos[b]]).collect(),
            StepInput::Same(i) => w[&m.inputs[i].0].clone(),
            StepInput::Cache(c) => sess.iter().flat_map(|&b| caches[b][c].clone()).collect(),
        })
        .collect()
}

/// The batched step bound to `nb` streams, routed experts grouped
/// (`schedule_expert_groups`) from `group` positions (0: not), simplified
/// and lowered. Returns the program, its outputs, and the contractions
/// grouped.
pub fn lower(
    st: &StepGraph,
    nb: usize,
    opts: &DecodeOpts,
    group: usize,
) -> (DecodeProgram, Vec<TensorId>, usize) {
    let mut ids = st.outputs.clone();
    ids.extend(st.caches.iter().map(|c| c.updated));
    let g = st.graph.bind(&[("B", nb)]).unwrap();
    let (g, ids, ng) = if group > 0 {
        let (g, ids, eg) = g.schedule_expert_groups(&ids, group);
        (g, ids, eg.grouped.len())
    } else {
        (g, ids, 0)
    };
    let (g, keep) = g.substitute_maps(&ids);
    let (g, keep) = g.eliminate_common(&keep);
    let dp = g.to_cuda_decode(&keep, opts);
    (dp, keep, ng)
}

/// Shapes of [`int8_case`].
#[derive(Clone, Copy, Debug)]
pub struct Int8Dims {
    /// The routed weights' format: Q4_K or Q5_K.
    pub q5: bool,
    pub ne: usize,
    pub rows: usize,
    pub k: usize,
    pub nb: usize,
    pub ns: usize,
}

impl Int8Dims {
    pub fn tiny() -> Self {
        Self {
            q5: false,
            ne: 6,
            rows: 32,
            k: 512,
            nb: 5,
            ns: 2,
        }
    }
    /// coder-next's routed gate (512 experts, 512 rows, d = 2048, top-10)
    /// and a dense `[512, 2048]` Q8_0 weight, at `nb` streams.
    pub fn coder(nb: usize) -> Self {
        Self {
            q5: false,
            ne: 512,
            rows: 512,
            k: 2048,
            nb,
            ns: 10,
        }
    }
    /// coder-next's routed down projection (Q5_K, 2048 rows, k = 512).
    pub fn coder_down(nb: usize) -> Self {
        Self {
            q5: true,
            ne: 512,
            rows: 2048,
            k: 512,
            nb,
            ns: 10,
        }
    }
}

/// A routed contraction `y[b, s, r] = Σ_k W[idx[b, s], r, k]·X[b, k]` (Q4_K
/// experts, grouped by expert from `group` positions) and a dense one
/// `z[b, r] = Σ_k V[r, k]·X[b, k]` (Q8_0), staged by `bo`, lowered; with
/// inputs and the f64 references (dequantized weights times the
/// Q8x32-rounded activations).
pub struct Int8Case {
    pub program: DecodeProgram,
    pub outs: Vec<TensorId>,
    pub inputs: Vec<Vec<f32>>,
    pub want: [Vec<f64>; 2],
    pub stages: Vec<String>,
}

impl Int8Case {
    /// `(routed, dense)` rel errs of outputs `got`.
    pub fn errors(&self, got: &[Vec<f32>]) -> (f64, f64) {
        let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        (
            rel(&f(&got[0]), &self.want[0]),
            rel(&f(&got[1]), &self.want[1]),
        )
    }
}

pub fn int8_case(
    d: &Int8Dims,
    bo: &tang_compile::tensor::int8_sched::BlockOpts,
    group: usize,
) -> Int8Case {
    use tang_compile::tensor::{gguf, Arg, Format, Monoid, Precision, View};
    let Int8Dims {
        q5,
        ne,
        rows,
        k,
        nb,
        ns,
    } = *d;
    let (rf, rb) = if q5 {
        (Format::Q5K, 176)
    } else {
        (Format::Q4K, 144)
    };
    let mut seed = 0x2545F491u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut w4 = Vec::new();
    for _ in 0..ne * rows * k / 256 {
        let mut b = vec![0u8; rb];
        b[..2].copy_from_slice(&0x2000u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x1c00u16.to_le_bytes());
        for x in &mut b[4..] {
            *x = rnd() as u8;
        }
        w4.extend(b);
    }
    let mut w8 = Vec::new();
    for _ in 0..rows * k / 32 {
        let mut b = vec![0u8; 34];
        b[..2].copy_from_slice(&0x2400u16.to_le_bytes());
        for x in &mut b[2..] {
            *x = rnd() as u8;
        }
        w8.extend(b);
    }
    let deq = |f: Format, b: &[u8], n: usize| {
        let mut o = vec![0f32; n];
        gguf::dequantize(f, b, &mut o).unwrap();
        o.into_iter().map(|v| v as f64).collect::<Vec<f64>>()
    };
    let d4 = deq(rf, &w4, ne * rows * k);
    let d8 = deq(Format::Q8_0, &w8, rows * k);
    // routed ids: some experts repeat across streams, distinct within one
    let idx: Vec<usize> = (0..nb * ns)
        .map(|p| ((p % ns) * 7 + (p / ns) * 3 * usize::from(p % 3 == 0)) % ne)
        .collect();
    let xs: Vec<f64> = (0..nb * k)
        .map(|i| ((i as f64) * 0.37).sin() * 0.8 + if i % 97 == 0 { 3.0 } else { 0.0 })
        .collect();
    let mut xq = xs.clone();
    Format::Q8x32.round_blocks(&mut xq);
    let f = Dim::Fixed;
    let mut g = TensorGraph::new();
    let ix = g.input(&[f(nb), f(ns)]);
    let x = g.input(&[f(nb), f(k)]);
    let w = g.input(&[f(ne), f(rows), f(k)]);
    let v = g.input(&[f(rows), f(k)]);
    let a = |src, view| Arg { src, view };
    let y = g
        .comp(
            &[f(nb), f(ns), f(rows)],
            &[f(k)],
            vec![
                a(ix, View::pick(4, &[Some(0), Some(1)])),
                a(x, View::pick(4, &[Some(0), Some(3)])),
                a(w, View::pick(4, &[None, Some(2), Some(3)]).gather(0, 0, 1)),
            ],
            Some(Monoid::sum()),
            |v| vec![v[1] * v[2]],
        )
        .unwrap();
    let z = g
        .comp(
            &[f(nb), f(rows)],
            &[f(k)],
            vec![
                a(x, View::pick(3, &[Some(0), Some(2)])),
                a(v, View::pick(3, &[Some(1), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    g.set_precision(w, Precision::storage(rf));
    g.set_precision(v, Precision::storage(Format::Q8_0));
    let (g, outs) = if group > 0 {
        let (g, o, _) = g.schedule_expert_groups(&[y, z], group);
        (g, o)
    } else {
        (g, vec![y, z])
    };
    let (g, outs, sched) = g.schedule_block_weights_with(&outs, bo);
    assert_eq!(sched.gemvs, 2, "both contractions on the int8 GEMV");
    let mut keep = outs.clone();
    keep.extend(&sched.copies);
    let (g, keep) = g.substitute_maps(&keep);
    let stages = g
        .nodes()
        .iter()
        .filter_map(|n| n.int_mma.map(|m| format!("{:?}/pt{}", m.stage, m.pos_tile)))
        .collect();
    let program = g.to_cuda_decode(&keep[..2], &Default::default());
    let inputs: Vec<Vec<f32>> = vec![
        idx.iter().map(|&e| e as f32).collect(),
        xs.iter().map(|&v| v as f32).collect(),
        crate::ir_ds_real::words_of(&w4),
        crate::ir_ds_real::words_of(&w8),
    ];
    let mut wy = vec![0.0; nb * ns * rows];
    for p in 0..nb * ns {
        for r in 0..rows {
            wy[p * rows + r] = (0..k)
                .map(|c| d4[(idx[p] * rows + r) * k + c] * xq[(p / ns) * k + c])
                .sum();
        }
    }
    let mut wz = vec![0.0; nb * rows];
    for b in 0..nb {
        for r in 0..rows {
            wz[b * rows + r] = (0..k).map(|c| d8[r * k + c] * xq[b * k + c]).sum();
        }
    }
    Int8Case {
        program,
        outs: keep[..2].to_vec(),
        inputs,
        want: [wy, wz],
        stages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streams at lengths `ps` stepped once together: the CUDA program run
    /// on the CPU against each stream's interpreter step alone (logits and
    /// every updated cache). Returns the worst rel err.
    pub(crate) fn batched_on_cpu(m: &Model, ps: &[usize], opts: &DecodeOpts, group: usize) -> f64 {
        let (bg, st) = batched_step(m);
        let w = weights(m);
        let nb = ps.len();
        let vocab = vocab(m);
        let toks: Vec<Vec<f64>> = (0..nb)
            .map(|b| {
                (0..m.cap)
                    .map(|i| ((37 * i + 5 + 11 * b) % vocab) as f64)
                    .collect()
            })
            .collect();
        let caches: Vec<Vec<Vec<f64>>> = ps
            .iter()
            .enumerate()
            .map(|(b, &p)| seed_stream(m, &bg, &st, &w, &toks[b], p))
            .collect();
        let mut ids = st.outputs.clone();
        ids.extend(st.caches.iter().map(|c| c.updated));
        let one = st.graph.bind(&[("B", 1)]).unwrap();
        let alone: Vec<Vec<Vec<f64>>> = (0..nb)
            .map(|b| one.eval(&step_inputs(m, &st, &w, &toks, ps, &caches, &[b]), &ids))
            .collect();
        let (dp, outs, ngroup) = lower(&st, nb, opts, group);
        let all: Vec<usize> = (0..nb).collect();
        let words: Vec<Vec<f32>> = step_inputs(m, &st, &w, &toks, ps, &caches, &all)
            .iter()
            .map(|v| {
                v.iter()
                    .map(|&x| if x.is_nan() { 0.0 } else { x as f32 })
                    .collect()
            })
            .collect();
        let got = crate::cuda_cpu::run(&dp, &words, &outs);
        let mut worst = 0.0f64;
        for (oi, g) in got.iter().enumerate() {
            let per = g.len() / nb;
            for b in 0..nb {
                let (gk, wk): (Vec<f64>, Vec<f64>) = g[b * per..(b + 1) * per]
                    .iter()
                    .zip(&alone[b][oi])
                    .filter(|(_, w)| !w.is_nan())
                    .map(|(a, w)| (*a as f64, *w))
                    .unzip();
                let e = rel(&gk, &wk);
                if e > 1e-4 {
                    eprintln!("{} output {oi} stream {b}: rel err {e:.2e}", m.name);
                }
                worst = worst.max(e);
            }
        }
        eprintln!(
            "{} batched B={nb} at lengths {ps:?} ({ngroup} expert-grouped contractions): {} \
             CUDA kernels on the CPU vs {nb} single-stream interpreter steps: rel err {worst:.2e}",
            m.name,
            dp.program.kernels.len()
        );
        worst
    }

    #[test]
    fn qwen3_batched_step_runs_on_the_cpu() {
        let e = batched_on_cpu(&qwen(), &[3, 9, 5, 1], &DecodeOpts::default(), 0);
        assert!(e < 1e-4, "{e}");
    }

    #[test]
    fn coder_batched_step_runs_on_the_cpu() {
        let e = batched_on_cpu(&coder(), &[3, 9, 5, 1], &DecodeOpts::default(), 0);
        assert!(e < 1e-4, "{e}");
    }

    /// The tiny coder's routed experts (6 experts, top-2) over B = 5
    /// streams grouped by expert: 10 (stream, slot) positions sorted per
    /// contraction, the same step.
    #[test]
    fn coder_batched_expert_groups_run_on_the_cpu() {
        let e = batched_on_cpu(&coder(), &[3, 9, 5, 1, 7], &DecodeOpts::default(), 2);
        assert!(e < 1e-4, "{e}");
    }

    /// Grouping on the interpreter: bit for bit the ungrouped step (the
    /// sort is a permutation; every sum keeps its terms and order).
    #[test]
    fn expert_groups_are_a_permutation() {
        let m = coder();
        let (bg, st) = batched_step(&m);
        let w = weights(&m);
        let ps = [3usize, 9, 5, 1, 7, 2];
        let nb = ps.len();
        let toks: Vec<Vec<f64>> = (0..nb)
            .map(|b| {
                (0..m.cap)
                    .map(|i| ((13 * i + 3 + 7 * b) % vocab(&m)) as f64)
                    .collect()
            })
            .collect();
        let caches: Vec<Vec<Vec<f64>>> = ps
            .iter()
            .enumerate()
            .map(|(b, &p)| seed_stream(&m, &bg, &st, &w, &toks[b], p))
            .collect();
        let mut ids = st.outputs.clone();
        ids.extend(st.caches.iter().map(|c| c.updated));
        let g = st.graph.bind(&[("B", nb)]).unwrap();
        let all: Vec<usize> = (0..nb).collect();
        let ins = step_inputs(&m, &st, &w, &toks, &ps, &caches, &all);
        let want = g.eval(&ins, &ids);
        let (gg, gids, eg) = g.schedule_expert_groups(&ids, 2);
        assert!(!eg.grouped.is_empty(), "nothing grouped");
        let got = gg.eval(&ins, &gids);
        for (a, b) in got.iter().zip(&want) {
            let same = a
                .iter()
                .zip(b)
                .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
            assert!(same, "grouped step differs");
        }
        eprintln!(
            "tiny coder B={nb}: {} routed contractions grouped by expert; step bit for bit",
            eg.grouped.len()
        );
    }

    #[test]
    fn block_weights_at_batch_run_on_the_cpu() {
        use tang_compile::tensor::int8_sched::BlockOpts;
        let bo = BlockOpts {
            pos_tile: 8,
            mma_min: 0,
            ksplit: 0,
        };
        let c = int8_case(&Int8Dims::tiny(), &bo, 2);
        let got = crate::cuda_cpu::run(&c.program, &c.inputs, &c.outs);
        let (ey, ez) = c.errors(&got);
        eprintln!(
            "int8 at B=5: routed Q4_K grouped by expert rel err {ey:.2e}; dense Q8_0 \
             5 positions a weight read rel err {ez:.2e}"
        );
        assert!(ey < 1e-5 && ez < 1e-5, "{ey} {ez}");
    }

    /// The real-size lowering (`lower_real_step_with`) of a coder step with
    /// int8-sized dims (d = 256, 8 experts top-4, Q4_K experts) at B = 4:
    /// routed experts grouped by expert against ungrouped, both run on the
    /// CPU. The grouping is a permutation of the same dot products.
    #[test]
    fn grouped_real_lowering_matches_ungrouped() {
        grouped_real(8, 4, 4);
    }

    /// coder-next's routing width: 512 experts, top-10.
    #[test]
    fn grouped_real_lowering_matches_ungrouped_512() {
        grouped_real(512, 10, 1);
        grouped_real(512, 10, 4);
    }

    fn grouped_real(n_expert: usize, n_used: usize, nb: usize) {
        use crate::ir_coder_prefill::*;
        use crate::ir_coder_real::*;
        use tang_compile::tensor::int8_sched::BlockOpts;
        use tang_llm::gguf::GgmlType;
        let dims = CoderDims {
            d_model: 256,
            vocab: 64,
            recurrent: vec![true, false],
            eps: 1e-6,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 64,
            n_rot: 32,
            rope_base: 1e4,
            ssm_d_conv: 4,
            ssm_d_state: 32,
            ssm_k_heads: 2,
            ssm_v_heads: 4,
            n_expert,
            n_expert_used: n_used,
            n_ff_exp: 256,
            n_ff_shexp: 256,
        };
        let pg = prefill_graph(&dims);
        let tys: Vec<Option<GgmlType>> = pg
            .inputs
            .iter()
            .map(|i| {
                if i.name == "tokens" {
                    None
                } else if i.name.contains("_exps") {
                    Some(GgmlType::Q4K)
                } else {
                    Some(GgmlType::F32)
                }
            })
            .collect();
        let st = derived_step_batched(&pg, 8).unwrap();
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // inputs in the step's order, per input: (f32 values | raw block words)
        let mut vals: Vec<Vec<f32>> = Vec::new();
        for si in &st.inputs {
            let v = match *si {
                StepInput::Pos => (0..nb).map(|b| b as f32).collect(),
                StepInput::New(_) => (0..nb).map(|b| (5 + 9 * b) as f32).collect(),
                StepInput::Cache(c) => {
                    let n: usize = st
                        .graph
                        .shape(st.caches[c].input)
                        .iter()
                        .map(|d| match d {
                            Dim::Sym(_) => nb,
                            d => d.n(),
                        })
                        .product();
                    vec![0.0; n]
                }
                StepInput::Same(i) => {
                    let inp = &pg.inputs[i];
                    let n: usize = inp.shape.iter().map(|d| d.n()).product();
                    if tys[i] == Some(GgmlType::Q4K) {
                        let mut b = Vec::new();
                        for _ in 0..n / 256 {
                            let mut blk = vec![0u8; 144];
                            blk[..2].copy_from_slice(&0x2000u16.to_le_bytes());
                            blk[2..4].copy_from_slice(&0x1c00u16.to_le_bytes());
                            for x in &mut blk[4..] {
                                *x = rnd() as u8;
                            }
                            b.extend(blk);
                        }
                        crate::ir_ds_real::words_of(&b)
                    } else if inp.name.contains("norm") {
                        (0..n)
                            .map(|_| 1.0 + ((rnd() % 1000) as f32 - 500.0) * 1e-4)
                            .collect()
                    } else {
                        (0..n)
                            .map(|_| ((rnd() % 2001) as f32 - 1000.0) * 1e-4)
                            .collect()
                    }
                }
            };
            vals.push(v);
        }
        let run = |group: usize| {
            let low = lower_real_step_with(
                &pg,
                &st,
                &tys,
                &RealOpts {
                    batch: Some(nb),
                    group,
                    block: Some(BlockOpts::default()),
                },
            );
            crate::cuda_cpu::run(&low.program, &vals, &low.outs[..1]).remove(0)
        };
        let a = run(0);
        let b = run(2);
        let ad: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let bd: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        let e = rel(&bd, &ad);
        eprintln!("real lowering B={nb}: grouped vs ungrouped logits rel err {e:.2e}");
        assert!(e < 1e-5, "{e}");
    }
}
