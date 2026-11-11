//! The derived decode step split across GPUs (`tang_compile::tensor::shard`):
//! one thread and CUDA device per rank, each running its rank program cut
//! at the collectives into segments (one `Program` per segment), the
//! collectives on the host between them (read back, sum / concatenate in
//! rank order, write back: `ep.rs`'s `Backend::Host`, correct and slow —
//! NCCL on the device buffers is the next step). Compares rank 0's logits
//! and caches with the unsharded step on the interpreter.
//!
//! ```text
//! MGPU_MODEL=ds-tiny|coder-tiny|ds-real  MGPU_WORLD=8  ds_multi_gpu
//! ```
//!
//! `ds-real` reads `DS41_DIR` / `DS_REAL_LAYERS` like `ds_real_gpu` (run
//! `ir_ds_real::tests::real_layers_match_reference` first for the routed
//! expert tables). `MGPU_PLAN=eager` takes every split the rules allow
//! (default: the 8×H200 cost model). Exits 1 on a mismatch over 1e-3.
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use tang_bench::ir_shard::{case, coder_model, coder_wide, ds_model, ds_wide, native_bytes, rel, summary, Case};
use tang_bench::qwen_session::lower;
use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_compile::tensor::shard::{
    analyze, input_shapes, plan_auto, rank_program, segments, CollKind, Comm, CostModel, Feed, HostComm,
};
use tang_compile::tensor::TensorId;
use tang_gpu::program::{Program, Runtime};

fn real_case() -> (Case, HashMap<TensorId, String>) {
    use tang_bench::ir_ds_prefill::{derived_step, prefill_graph, DsDims, SEQ};
    use tang_bench::ir_ds_real::*;
    use tang_compile::tensor::step::{seed_cache, StepInput};
    use tang_compile::tensor::Dim;
    let run = RealRun::from_env().unwrap_or_else(|| panic!("no {}/inference_config.json", ds_dir()));
    let t_pre = 3;
    let toks = run.rows_of(&prompt(t_pre + 1));
    let used = run.read_routed(toks.len()).expect("routed tables (run real_layers_match_reference)");
    let rows = used.iter().map(|u| u.len()).max().unwrap();
    // a world-divisible row count: pad the compact expert tables
    let world: usize = std::env::var("MGPU_WORLD").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let rows = rows.div_ceil(world) * world;
    let dm: DsDims = dims_from_config(&run.cfg, &run.layer_ids, run.vocab_ids.len(), rows, 8);
    let pg = prefill_graph(&dm);
    let mut w = run.dense_inputs(&pg.inputs, &dm).unwrap();
    for (l, u) in used.iter().enumerate() {
        run.expert_tables(l, u, rows, &dm, &mut w);
    }
    let names: Vec<String> = pg.inputs.iter().map(|s| s.name.clone()).collect();
    let mut base: Vec<Vec<f64>> =
        pg.inputs.iter().map(|s| if s.name == "tokens" { toks[..t_pre].to_vec() } else { w.remove(&s.name).unwrap() }).collect();
    let st = derived_step(&pg, &dm, 1).unwrap();
    let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
    let pre = pg.g.bind(&[(SEQ, t_pre)]).unwrap().eval(&base, &nodes);
    let caches: Vec<Vec<f64>> = nodes
        .iter()
        .zip(pre)
        .zip(&st.caches)
        .map(|((&n, v), c)| {
            let shape: Vec<usize> =
                pg.g.shape(n).iter().map(|d| if *d == Dim::Sym(SEQ) { t_pre } else { d.n() }).collect();
            seed_cache(c, &shape, &v, t_pre, dm.ctx)
        })
        .collect();
    let mut wnames = HashMap::new();
    let ins: Vec<Vec<f64>> = st
        .inputs
        .iter()
        .enumerate()
        .map(|(k, &x)| match x {
            StepInput::Pos => vec![t_pre as f64],
            StepInput::New(_) => toks[t_pre..t_pre + 1].to_vec(),
            StepInput::Same(i) => {
                wnames.insert(tang_bench::ir_model::input_node(&st.graph, k), names[i].clone());
                std::mem::take(&mut base[i])
            }
            StepInput::Cache(c) => caches[c].clone(),
        })
        .collect();
    let mut outs = st.outputs.clone();
    outs.extend(st.caches.iter().map(|c| c.updated));
    let want = st.graph.eval(&ins, &outs);
    let weights: HashSet<TensorId> = wnames.keys().copied().collect();
    (Case { st, ins, outs, want, weights }, wnames)
}

fn main() {
    let model = std::env::var("MGPU_MODEL").unwrap_or_else(|_| "ds-tiny".into());
    let world: usize = std::env::var("MGPU_WORLD").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let eager = std::env::var("MGPU_PLAN").as_deref() == Ok("eager");
    let t0 = Instant::now();
    let (c, names) = match model.as_str() {
        "ds-tiny" => (case(&ds_model(ds_wide()), 1, 5), HashMap::new()),
        "coder-tiny" => (case(&coder_model(coder_wide()), 1, 5), HashMap::new()),
        "ds-real" => real_case(),
        m => panic!("MGPU_MODEL {m}?"),
    };
    eprintln!("{model}: step + interpreter reference in {:.1?}", t0.elapsed());
    let elem = |t: TensorId| names.get(&t).map_or(1.0, |n| native_bytes(n));
    let cm = if eager { CostModel::eager() } else { CostModel::h200() };
    let plan = plan_auto(&c.st.graph, &c.outs, &c.weights, world, &elem, &cm);
    let an = analyze(&c.st.graph, &c.outs, &plan);
    let s = summary(&plan, &an);
    eprintln!(
        "plan for {world} ranks: {} expert-parallel, {} tensor-parallel inputs; {} all-reduce + {} all-gather a step",
        s.experts, s.splits, s.all_reduce, s.all_gather
    );
    let comm = HostComm::new(world);
    let shapes = input_shapes(&c.st.graph);
    let results: Vec<(Vec<Vec<f64>>, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..world)
            .map(|r| {
                let (c, an, comm, shapes) = (&c, &an, comm.clone(), &shapes);
                sc.spawn(move || {
                    let rt = Runtime::cuda_on(r).unwrap_or_else(|| panic!("no CUDA device {r}"));
                    let p = rank_program(&c.st.graph, &c.outs, an, &c.weights, r, comm.clone());
                    let ins = p.local_inputs(&c.ins, shapes);
                    let sg = segments(&p);
                    // one Program per segment, its fed values written each step
                    let progs: Vec<Option<(Program, Vec<TensorId>)>> = sg
                        .segments
                        .iter()
                        .map(|seg| {
                            if seg.outputs.is_empty() {
                                return None;
                            }
                            let (g, outs) = seg.graph.substitute_maps(&seg.outputs);
                            let (g, outs) = g.eliminate_common(&outs);
                            let dp = lower(&rt, &g, &outs, &DecodeOpts::default());
                            let init: Vec<Vec<f32>> = seg
                                .feeds
                                .iter()
                                .enumerate()
                                .map(|(k, f)| match f {
                                    Feed::Input(i) => ins[*i].iter().map(|&x| x as f32).collect(),
                                    Feed::Value(_) => vec![0.0; seg.graph.shape(tang_bench::ir_model::input_node(&seg.graph, k)).iter().map(|d| d.n()).product()],
                                })
                                .collect();
                            let refs: Vec<&[f32]> = init.iter().map(|v| v.as_slice()).collect();
                            Some((Program::new(&rt, dp.program, &refs), outs))
                        })
                        .collect();
                    let step = || -> Vec<Vec<f64>> {
                        let mut vals: HashMap<usize, Vec<f64>> =
                            sg.input_nodes.iter().map(|&(i, k)| (i, ins[k].clone())).collect();
                        for (si, seg) in sg.segments.iter().enumerate() {
                            for b in &sg.before[si] {
                                let x = &vals[&b.src];
                                let y = match b.kind {
                                    CollKind::AllReduce => comm.all_reduce(r, x),
                                    CollKind::AllGather(_) => comm.all_gather(r, x, b.outer),
                                };
                                vals.insert(b.dst, y);
                            }
                            let Some((prog, outs)) = &progs[si] else { continue };
                            for (k, f) in seg.feeds.iter().enumerate() {
                                if let Feed::Value(n) = f {
                                    let v: Vec<f32> = vals[n].iter().map(|&x| x as f32).collect();
                                    prog.write(&rt, prog.input(k), &v);
                                }
                            }
                            prog.run(&rt);
                            for (d, o) in seg.defines.iter().zip(outs) {
                                vals.insert(*d, prog.read(&rt, *o).iter().map(|&x| x as f64).collect());
                            }
                        }
                        sg.outputs.iter().map(|o| vals[o].clone()).collect()
                    };
                    let out = step();
                    let t = Instant::now();
                    let iters = 5;
                    for _ in 0..iters {
                        step();
                    }
                    (out, t.elapsed().as_secs_f64() * 1e3 / iters as f64)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let worst = results[0].0.iter().zip(&c.want).map(|(a, b)| rel(a, b)).fold(0.0, f64::max);
    let ok = worst < 1e-3;
    println!(
        "{model} on {world} GPUs: rank 0 vs unsharded interpreter rel err {worst:.2e}, {:.2} ms a step \
         (host collectives, f32 weights): {}",
        results[0].1,
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}
