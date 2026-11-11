//! Multi-GPU decode from the derived step graph (`tang_compile::tensor::shard`):
//! DeepSeek V4.1 and Qwen3-Coder-Next steps split across `N` ranks —
//! experts by expert parallelism, dense and attention weights by tensor
//! parallelism, all-reduce / all-gather nodes between — checked against the
//! unsharded step on the interpreter with the ranks as threads, and the
//! real-size DeepSeek plan's communication per step.
use std::collections::HashSet;

use tang_compile::tensor::shard::{Analysis, CollKind, ShardPlan, WeightPlace};
use tang_compile::tensor::step::{derive_step, StepGraph, StepInput};
use tang_compile::tensor::TensorId;

use crate::derive_all::{self, Model};
use crate::ir_model::input_node;

/// The step's weight inputs (node ids) and their prefill names.
pub fn weight_inputs(m: &Model, st: &StepGraph) -> Vec<(TensorId, String)> {
    st.inputs
        .iter()
        .enumerate()
        .filter_map(|(k, si)| match si {
            StepInput::Same(i) => Some((input_node(&st.graph, k), m.inputs[*i].0.clone())),
            _ => None,
        })
        .collect()
}

/// Bytes per element a decode step reads, by name (DeepSeek's native
/// formats, as `scripts/ds-roofline.py`): fp4 experts, bf16 head and
/// embeddings, f32 mHC and norms, fp8 everything else.
pub fn native_bytes(name: &str) -> f64 {
    if name.contains("experts.") && !name.contains("shared") {
        0.5 + 1.0 / 32.0
    } else if name.contains("head") || name.contains("embed") {
        2.0
    } else if name.contains("hc_")
        || name.contains("norm")
        || name.contains("bias")
        || name.contains("sink")
    {
        4.0
    } else {
        1.0 + 1.0 / 1024.0
    }
}

pub struct Summary {
    pub experts: usize,
    pub splits: usize,
    pub all_reduce: usize,
    pub all_gather: usize,
}

pub fn summary(plan: &ShardPlan, an: &Analysis) -> Summary {
    Summary {
        experts: plan
            .inputs
            .values()
            .filter(|p| **p == WeightPlace::Experts)
            .count(),
        splits: plan
            .inputs
            .values()
            .filter(|p| matches!(p, WeightPlace::Split(_)))
            .count(),
        all_reduce: an.count(|k| *k == CollKind::AllReduce),
        all_gather: an.count(|k| matches!(k, CollKind::AllGather(_))),
    }
}

pub fn ds_model(dm: crate::ir_ds_prefill::DsDims) -> Model {
    let pg = crate::ir_ds_prefill::prefill_graph(&dm);
    Model {
        name: "deepseek-v4.1",
        inputs: pg
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.shape.clone()))
            .collect(),
        logits: pg.logits,
        named: pg.named.clone(),
        g: pg.g,
        cap: derive_all::CAP,
    }
}

pub fn coder_model(cd: crate::ir_coder_prefill::CoderDims) -> Model {
    let pg = crate::ir_coder_prefill::prefill_graph(&cd);
    Model {
        name: "coder-next",
        inputs: pg
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.shape.clone()))
            .collect(),
        logits: pg.logits,
        named: vec![],
        g: pg.g,
        cap: derive_all::CAP,
    }
}

/// DeepSeek tiny, widened so 8 ranks divide the experts and heads.
pub fn ds_wide() -> crate::ir_ds_prefill::DsDims {
    crate::ir_ds_prefill::DsDims {
        n_heads: 8,
        o_groups: 8,
        n_experts: 16,
        n_active: 3,
        moe_inter: 16,
        index_n_heads: 8,
        ..crate::ir_ds_prefill::DsDims::tiny()
    }
}

pub fn coder_wide() -> crate::ir_coder_prefill::CoderDims {
    crate::ir_coder_prefill::CoderDims {
        n_head: 8,
        n_head_kv: 8,
        n_expert: 16,
        n_expert_used: 3,
        n_ff_exp: 16,
        n_ff_shexp: 32,
        ssm_k_heads: 8,
        ssm_v_heads: 8,
        ..crate::ir_coder_prefill::CoderDims::tiny()
    }
}

/// One step case on the interpreter: the derived step of `k` tokens at
/// position `p0` after a prefill (caches seeded from it), its inputs, the
/// outputs checked (logits then updated caches) and their values.
pub struct Case {
    pub st: StepGraph,
    pub ins: Vec<Vec<f64>>,
    pub outs: Vec<TensorId>,
    pub want: Vec<Vec<f64>>,
    pub weights: HashSet<TensorId>,
    /// `(cache input, index into outs of its update)`.
    pub caches: Vec<(TensorId, usize)>,
}

pub fn case(m: &Model, k: usize, p0: usize) -> Case {
    use crate::derive_all::{seed, spec, vocab, weights};
    let w = weights(m);
    let toks: Vec<f64> = (0..p0 + k)
        .map(|i| ((37 * i + 5) % vocab(m)) as f64)
        .collect();
    let full_ins = |t: &[f64]| -> Vec<Vec<f64>> {
        m.inputs
            .iter()
            .map(|(n, _)| {
                if n == "tokens" {
                    t.to_vec()
                } else {
                    w[n].clone()
                }
            })
            .collect()
    };
    let st = derive_step(&m.g, &[m.logits], &spec(m, k)).expect("derives");
    let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
    let fg = m.g.bind(&[("T", p0)]).unwrap();
    let caches: Vec<Vec<f64>> = st
        .caches
        .iter()
        .zip(fg.eval(&full_ins(&toks[..p0]), &nodes))
        .map(|(c, v)| seed(m, c, v, p0))
        .collect();
    let ins: Vec<Vec<f64>> = st
        .inputs
        .iter()
        .map(|&x| match x {
            StepInput::Pos => vec![p0 as f64],
            StepInput::New(_) => toks[p0..p0 + k].to_vec(),
            StepInput::Same(i) => w[&m.inputs[i].0].clone(),
            StepInput::Cache(c) => caches[c].clone(),
        })
        .collect();
    let mut outs = st.outputs.clone();
    outs.extend(st.caches.iter().map(|c| c.updated));
    let want = st.graph.eval(&ins, &outs);
    let weights: HashSet<TensorId> = weight_inputs(m, &st).into_iter().map(|(t, _)| t).collect();
    let caches = cache_pairs(&st);
    Case {
        st,
        ins,
        outs,
        want,
        weights,
        caches,
    }
}

/// Each cache's input and the index of its update in `outputs ++ updated`.
pub fn cache_pairs(st: &StepGraph) -> Vec<(TensorId, usize)> {
    st.caches
        .iter()
        .enumerate()
        .map(|(c, s)| (s.input, st.outputs.len() + c))
        .collect()
}

/// Worst `|a − b| / max|b|` over the finite reference entries (unwritten
/// cache rows are NaN in both).
pub fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b
        .iter()
        .filter(|x| x.is_finite())
        .fold(1e-300f64, |s, x| s.max(x.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| match (x.is_nan(), y.is_nan()) {
            (true, true) => 0.0,
            (false, false) => (x - y).abs() / scale,
            _ => f64::INFINITY,
        })
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derive_all::spec;
    use tang_compile::tensor::shard::{
        analyze, plan_auto, simulate, simulate_segmented, step_cost, CostModel,
    };

    /// Step `k` tokens at `p0` unsharded and on `N` simulated ranks under the
    /// eager plan (every split the rules take) and the 8×H200 plan, whole
    /// and cut into segments at the collectives.
    fn check(m: &Model, k: usize, worlds: &[usize]) {
        let Case {
            st,
            ins,
            outs,
            want,
            weights: ws,
            caches,
        } = case(m, k, 5);
        assert!(want[0].iter().all(|x| x.is_finite()));
        let one = |_: TensorId| 1.0;
        let dense = |n: usize| {
            let an = analyze(
                &st.graph,
                &outs,
                &ShardPlan {
                    world: n,
                    ..Default::default()
                },
            );
            step_cost(&st.graph, &outs, &an, &ws, &one, &CostModel::h200()).weight_bytes
        };
        for &n in worlds {
            for (label, cm) in [("eager", CostModel::eager()), ("8xH200", CostModel::h200())] {
                let plan = plan_auto(&st.graph, &outs, &ws, &caches, n, &one, &cm);
                let an = analyze(&st.graph, &outs, &plan);
                let got = simulate(&st.graph, &outs, &an, &ws, &ins);
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| rel(a, b))
                    .fold(0.0, f64::max);
                let seg = simulate_segmented(&st.graph, &outs, &an, &ws, &ins);
                let serr = seg
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| rel(a, b))
                    .fold(0.0, f64::max);
                let s = summary(&plan, &an);
                let c = step_cost(&st.graph, &outs, &an, &ws, &one, &CostModel::h200());
                eprintln!(
                    "{} k={k} N={n} {label:6}: {} expert / {} split weights, {} all-reduce + {} all-gather, \
                     {:.0} of {:.0} dense elems a rank; max rel err {err:.2e} (segmented {serr:.2e})",
                    m.name,
                    s.experts,
                    s.splits,
                    s.all_reduce,
                    s.all_gather,
                    c.weight_bytes,
                    dense(n),
                );
                assert!(err < 1e-10, "{} N={n} {label}: {err:e}", m.name);
                assert!(serr < 1e-10, "{} N={n} {label} segmented: {serr:e}", m.name);
                assert!(s.experts >= 3, "{} N={n}: experts not sharded", m.name);
                if label == "eager" {
                    assert!(s.splits > 0, "{} N={n}: no tensor-parallel split", m.name);
                }
            }
        }
    }

    #[test]
    fn ds_tiny_sharded_matches() {
        let m = ds_model(ds_wide());
        check(&m, 1, &[2, 4, 8]);
        check(&m, 2, &[8]);
    }

    #[test]
    fn coder_tiny_sharded_matches() {
        let m = coder_model(coder_wide());
        check(&m, 1, &[2, 4, 8]);
        check(&m, 2, &[8]);
    }

    /// GDN under TP (joint moves in `plan_auto`): the scans run split by head and their state caches
    /// stay split per rank (no gather of the state each step).
    #[test]
    fn coder_gdn_scans_split_by_head() {
        use tang_compile::tensor::shard::Place;
        use tang_compile::tensor::Op;
        let m = coder_model(coder_wide());
        // two rows: the step keeps a real scan (one row lowers to plain comps)
        let Case {
            st,
            ins,
            outs,
            want,
            weights: ws,
            caches,
        } = case(&m, 2, 5);
        for n in [2usize, 4, 8] {
            let plan = plan_auto(
                &st.graph,
                &outs,
                &ws,
                &caches,
                n,
                &|_| 1.0,
                &CostModel::eager(),
            );
            let an = analyze(&st.graph, &outs, &plan);
            let scans: Vec<bool> = st
                .graph
                .nodes()
                .iter()
                .enumerate()
                .filter(|(_, nd)| matches!(nd.op, Op::Scan(_)))
                .map(|(i, _)| matches!(an.place[i], Place::S(_)))
                .collect();
            let local = an.out_local.iter().filter(|l| l.is_some()).count();
            eprintln!(
                "coder N={n}: {}/{} scans split, {local} caches kept split",
                scans.iter().filter(|&&b| b).count(),
                scans.len()
            );
            assert!(
                !scans.is_empty() && scans.iter().any(|&b| b),
                "N={n}: no scan split"
            );
            assert!(local > 0, "N={n}: no cache kept split");
            let got = simulate(&st.graph, &outs, &an, &ws, &ins);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| rel(a, b))
                .fold(0.0, f64::max);
            eprintln!("  rel err {err:.2e}");
            assert!(err < 1e-9, "N={n}: sharded GDN diverges");
        }
    }

    /// The check notices a wrong program: the same plan with every
    /// all-reduce dropped (partial sums left per rank) diverges.
    #[test]
    fn negative_control_dropped_all_reduce_fails() {
        for m in [ds_model(ds_wide()), coder_model(coder_wide())] {
            let Case {
                st,
                ins,
                outs,
                want,
                weights: ws,
                caches,
            } = case(&m, 1, 5);
            let plan = plan_auto(
                &st.graph,
                &outs,
                &ws,
                &caches,
                4,
                &|_| 1.0,
                &CostModel::eager(),
            );
            let mut an = analyze(&st.graph, &outs, &plan);
            for c in an.conv.iter_mut().flatten().chain(an.out_conv.iter_mut()) {
                if *c == Some(CollKind::AllReduce) {
                    *c = None;
                }
            }
            let got = std::panic::catch_unwind(|| simulate(&st.graph, &outs, &an, &ws, &ins));
            let e = got.map_or(f64::INFINITY, |g| rel(&g[0], &want[0]));
            eprintln!("{}: all-reduces dropped: logits rel err {e:.2e}", m.name);
            assert!(
                e > 1e-3,
                "{}: dropping the all-reduces went unnoticed",
                m.name
            );
        }
    }

    /// Real-size DeepSeek-V4.1-Flash (40 layers, config from `DS41_DIR`):
    /// the plan `plan_auto` derives for 8×H200 and 4×H200, its collectives
    /// and bytes per decode step, and the roofline time it implies. Shapes
    /// only: no weights are touched.
    #[test]
    #[ignore]
    fn ds_real_plan_and_comm() {
        let dir = crate::ir_ds_real::ds_dir();
        let cfg =
            crate::ir_ds_real::load_config(std::path::Path::new(&dir)).expect("DS41_DIR config");
        let layers: Vec<usize> = (0..cfg["n_layers"].as_u64().unwrap() as usize).collect();
        let vocab = cfg["vocab_size"].as_u64().unwrap() as usize;
        let dm = crate::ir_ds_real::dims_from_config(&cfg, &layers, vocab, 0, 4096);
        // every index source has its own indexer (24..36 keyed off layer
        // 20's compressed latent); ~6 MB fp8 a layer, replicated
        eprintln!(
            "index sources {:?}, KV sources {:?}",
            dm.index_source_layers, dm.kv_source_layers
        );
        let (l, inter, d) = (dm.n_layers, dm.moe_inter, dm.dim);
        let t0 = std::time::Instant::now();
        let m = Model {
            cap: 4096,
            ..ds_model(dm.clone())
        };
        let st = derive_step(&m.g, &[m.logits], &spec(&m, 1)).expect("derives");
        eprintln!(
            "real DS: {} step nodes, derived in {:.1?}",
            st.graph.nodes().len(),
            t0.elapsed()
        );
        let mut outs = st.outputs.clone();
        outs.extend(st.caches.iter().map(|c| c.updated));
        let wi = weight_inputs(&m, &st);
        let ws: HashSet<TensorId> = wi.iter().map(|(t, _)| *t).collect();
        let names: std::collections::HashMap<TensorId, String> = wi.iter().cloned().collect();
        let elem = |t: TensorId| native_bytes(&names[&t]);
        for (n, bw, label) in [(8usize, 4.8e12, "8xH200"), (4, 4.8e12, "4xH200")] {
            let t0 = std::time::Instant::now();
            let cm = CostModel::h200();
            let plan = plan_auto(
                &st.graph,
                &outs,
                &ws,
                &crate::ir_shard::cache_pairs(&st),
                n,
                &elem,
                &cm,
            );
            let an = analyze(&st.graph, &outs, &plan);
            let c = step_cost(&st.graph, &outs, &an, &ws, &elem, &cm);
            let s = summary(&plan, &an);
            let mut split_names: Vec<String> = plan
                .inputs
                .iter()
                .filter_map(|(t, p)| match p {
                    WeightPlace::Split(dd) => Some(format!(
                        "{}[{dd}]",
                        names[t].split('.').skip(2).collect::<Vec<_>>().join(".")
                    )),
                    _ => None,
                })
                .collect();
            split_names.sort();
            split_names.dedup();
            // routed experts: the busiest rank streams 1.4x the mean of its picks
            // (`scripts/ds-roofline.py`), at least one expert a layer
            let exp1 = 3.0 * inter as f64 * d as f64 * native_bytes("experts.w1");
            let picks = dm.n_active as f64;
            let rank_exp = l as f64 * (1.4 * picks / n as f64).max(1.0) * exp1;
            let t_w = c.weight_bytes / bw;
            let t_e = rank_exp / bw;
            let t_c = c.coll_bytes / cm.link + c.n_coll as f64 * cm.latency;
            let t = t_w + t_e + t_c;
            eprintln!("--- real DS, {label}: planned in {:.1?}", t0.elapsed());
            eprintln!(
                "  {} expert-parallel inputs, {} tensor-parallel splits: {}",
                s.experts,
                s.splits,
                split_names.join(" ")
            );
            eprintln!(
                "  per step: {} collectives ({} all-reduce, {} all-gather) = {:.1} per layer; {:.1} KB on the wire per rank",
                c.n_coll,
                s.all_reduce,
                s.all_gather,
                c.n_coll as f64 / l as f64,
                c.coll_bytes / 1e3
            );
            let mut sizes: Vec<(usize, CollKind)> =
                an.collectives.iter().map(|c| (c.numel, c.kind)).collect();
            sizes.sort_by_key(|x| (x.0, format!("{:?}", x.1)));
            let mut hist: Vec<(usize, CollKind, usize)> = Vec::new();
            for (e, k) in sizes {
                match hist.last_mut() {
                    Some(h) if h.0 == e && h.1 == k => h.2 += 1,
                    _ => hist.push((e, k, 1)),
                }
            }
            for (e, k, cnt) in hist {
                eprintln!(
                    "    {cnt:4} x {k:?} of {e} f32 ({:.1} KB)",
                    e as f64 * 4.0 / 1e3
                );
            }
            eprintln!(
                "  rank reads: dense {:.2} GB + experts {:.2} GB; time {:.0} us weights + {:.0} us experts + {:.0} us collectives = {:.0} us -> {:.0} tok/s plain",
                c.weight_bytes / 1e9,
                rank_exp / 1e9,
                t_w * 1e6,
                t_e * 1e6,
                t_c * 1e6,
                t * 1e6,
                1.0 / t
            );
        }
    }
}
