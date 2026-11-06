//! Is the autotuner trustworthy? For matmul and attention forward and
//! backward at several sizes:
//!
//! 1. tune exhaustively (every distinct candidate timed, GPU clock) —
//!    the reference, and the cost model's training log;
//! 2. tune 5 more times from scratch with the cost model pruning
//!    (trained leave-one-problem-out: never on the problem being tuned);
//! 3. re-time the exhaustive pick against each rerun's pick, whole
//!    programs, interleaved.
//!
//! Prints whether the 5 picks agree and each one's gap to the exhaustive
//! best. `cargo run --release -p tang-bench [--features cuda] --example
//! autotune_stability [quick]` (`TANG_BACKEND=wgpu|cuda` picks the runtime).
use tang_bench::ir_gpu::naive_attention_heads;
use tang_bench::Rng;
use tang_compile::tensor::space::CostModel;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_gpu::autotune::{autotune, load_samples, program_samples, Stats, TuneOptions};
use tang_gpu::program::{Program, Runtime};

fn passes(g: TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, o) = g.stabilize_softmax(outs);
    let (g, o) = g.substitute_maps(&o);
    let (g, o) = g.eliminate_common(&o);
    let (g, o) = g.merge_siblings(&o);
    let (g, o) = g.relayout(&o);
    let (g, o) = g.merge_reduce_dims(&o);
    let (mut g, o) = g.split_reduce_dims(&o);
    g.schedule_auto(&o);
    (g, o)
}

fn matmul(m: usize, k: usize, n: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let a = g.input(&[m, k]);
    let b = g.input(&[k, n]);
    let c = g.matmul(a, b).unwrap();
    passes(g, &[c])
}

fn attn_fwd(h: usize, n: usize, d: usize) -> (TensorGraph, Vec<TensorId>) {
    let (g, y) = naive_attention_heads(h, 1, n, d, d, true);
    passes(g, &[y])
}

fn attn_bwd(h: usize, n: usize, d: usize) -> (TensorGraph, Vec<TensorId>) {
    let (mut g, y) = naive_attention_heads(h, 1, n, d, d, true);
    let inputs: Vec<TensorId> = (0..3).map(TensorId::from_index).collect();
    let dy = g.input(g.shape(y).to_vec().as_slice());
    let l = g.map(&[y, dy], |x| x[0] * x[1]).unwrap();
    let grads = g.backward(l, &inputs).unwrap();
    passes(g, &grads)
}

fn hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let rt = Runtime::select().expect("gpu");
    let reruns: usize = std::env::var("RERUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    eprintln!("device: {}", rt.name());
    let dir = std::env::temp_dir().join(format!("tang-autotune-stability-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    type Build = Box<dyn Fn() -> (TensorGraph, Vec<TensorId>)>;
    let mut problems: Vec<(String, Build)> = Vec::new();
    let mm: &[usize] = if quick {
        &[256, 512]
    } else {
        &[256, 512, 1024, 2048]
    };
    for &n in mm {
        problems.push((format!("matmul {n}"), Box::new(move || matmul(n, n, n))));
    }
    if !quick {
        problems.push((
            "matmul 4096x64x1024".into(),
            Box::new(|| matmul(4096, 64, 1024)),
        ));
    }
    let fw: &[usize] = if quick { &[256] } else { &[512, 1024, 2048] };
    for &n in fw {
        problems.push((
            format!("attn fwd h8 n{n} d64"),
            Box::new(move || attn_fwd(8, n, 64)),
        ));
    }
    let bw: &[usize] = if quick { &[128] } else { &[256, 512, 1024] };
    for &n in bw {
        problems.push((
            format!("attn bwd h4 n{n} d64"),
            Box::new(move || attn_bwd(4, n, 64)),
        ));
    }

    let mut rng = Rng::new(7);
    type Built = (String, TensorGraph, Vec<TensorId>, Vec<Vec<f32>>);
    let built: Vec<Built> = problems
        .iter()
        .map(|(name, b)| {
            let (g, o) = b();
            let prog = rt.lower(&g, &o);
            let n_in = prog.buffers.iter().filter(|b| b.input.is_some()).count();
            let inputs = (0..n_in)
                .map(|k| {
                    let len = prog
                        .buffers
                        .iter()
                        .find(|b| b.input == Some(k))
                        .unwrap()
                        .len;
                    (0..len).map(|_| rng.f32() - 0.5).collect()
                })
                .collect();
            (name.clone(), g, o, inputs)
        })
        .collect();

    // 1. exhaustive
    let mut exhaustive = Vec::new();
    for (name, g, o, inputs) in &built {
        let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let opts = TuneOptions {
            exhaustive: true,
            dir: Some(dir.clone()),
            group: hash(name),
            ..Default::default()
        };
        let t = std::time::Instant::now();
        let (tg, rep) = autotune(&rt, g.clone(), o, &ins, &opts);
        let timed: usize = rep.iter().map(|k| k.n_timed).sum();
        let wall = rep.iter().any(|k| k.wall_clock);
        eprintln!(
            "{name}: exhaustive {} kernels, {timed} timed, {:.1}s{}",
            rep.len(),
            t.elapsed().as_secs_f64(),
            if wall { " (HOST CLOCK)" } else { "" }
        );
        exhaustive.push((tg, timed));
    }
    let log = load_samples(&dir.join(format!("{}.log", slug(rt.name()))));
    eprintln!("cost-model log: {} samples", log.len());

    // 2. reruns with the model, 3. verification
    println!("| problem | kernels | timed exh. | timed w/ model | stable 5/5 | gap to exhaustive (max over reruns) | auto→tuned |");
    println!("|---|---|---|---|---|---|---|");
    for ((name, g, o, inputs), (best_g, timed_ex)) in built.iter().zip(&exhaustive) {
        let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let train: Vec<_> = log
            .iter()
            .filter(|s| s.group != hash(name))
            .cloned()
            .collect();
        let model = CostModel::train(&train);
        let mut picks: Vec<(TensorGraph, Vec<String>)> = Vec::new();
        let mut timed_model = 0;
        for _ in 0..reruns {
            let opts = TuneOptions {
                model: model.clone(),
                dir: None,
                ..Default::default()
            };
            let (tg, rep) = autotune(&rt, g.clone(), o, &ins, &opts);
            timed_model = rep.iter().map(|k| k.n_timed).sum();
            let enc: Vec<String> = rep.iter().map(|k| k.picked.encode(&opts.space)).collect();
            picks.push((tg, enc));
        }
        let stable = picks.iter().all(|p| p.1 == picks[0].1);
        // whole programs, interleaved: exhaustive best, schedule_auto, each rerun
        let mut graphs: Vec<&TensorGraph> = vec![best_g, g];
        graphs.extend(picks.iter().map(|p| &p.0));
        let progs: Vec<Program> = graphs
            .iter()
            .map(|gr| {
                let p = Program::compile(&rt, gr, o, &ins);
                p.run(&rt);
                p
            })
            .collect();
        // paired rounds: every program once per round, scored by its
        // ratio to the exhaustive pick in that round (drift cancels)
        let mut samples = vec![Vec::new(); progs.len()];
        let mut ratios = vec![Vec::new(); progs.len()];
        for _ in 0..12 {
            let f: Vec<f64> = progs
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let s = program_samples(&rt, p, 2, 10);
                    let f = Stats::of(&s).fast;
                    samples[i].extend(s);
                    f
                })
                .collect();
            for i in 0..progs.len() {
                ratios[i].push(f[i] / f[0]);
            }
        }
        let st: Vec<Stats> = samples.iter().map(|s| Stats::of(s)).collect();
        let rel: Vec<f64> = ratios.iter().map(|r| Stats::of(r).median).collect();
        // gap to the fastest program measured here: the exhaustive pick
        // or, where it lost the re-timing, the best rerun pick
        let best = rel[2..].iter().cloned().fold(1.0, f64::min);
        let gaps: Vec<f64> = rel[2..].iter().map(|r| r / best - 1.0).collect();
        let worst = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "| {name} | {} | {timed_ex} | {timed_model} | {} | {:+.1}% (exh. {:.3} ms ±{:.1}%) | {:.3} → {:.3} ms |",
            picks[0].1.len(),
            if stable { "yes" } else { "NO" },
            worst * 100.0,
            st[0].fast,
            st[0].noise * 100.0,
            st[0].fast * rel[1],
            st[0].fast * rel[2],
        );
        if !stable {
            for (r, p) in picks.iter().enumerate() {
                eprintln!("  rerun {r}: {:?}", p.1);
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
