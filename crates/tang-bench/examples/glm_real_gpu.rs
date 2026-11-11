//! GLM-5.3-Flash at real size from its derived graph, on the GPU, with the
//! sampler in the step: the real-dims graph of the GGUF subset's layers
//! (default checkpoint layers 0 and 3; `tang_bench::ir_glm_real`), real
//! weights dequantized to f32, `tang_bench::ir_sample` appended before
//! `derive_step`, so the step's outputs are the logits and the next token.
//!
//! Default (compact experts): prefills `GLM_REAL_T` tokens (default 3) on
//! the interpreter with the experts the run routes to, seeds the caches,
//! runs the derived one-token step on the device and checks it against the
//! interpreter's step: logits and caches (f32 device vs f64), the greedy
//! token (bit-identical to the first argmax of the device's own logits,
//! and the interpreter's token), and a seeded sampled token (same seed,
//! same token, run to run).
//!
//! `--decode N` (needs every expert on the device: `expert_rows = 0`,
//! ~29 GB f32 per MoE layer, an 80 GB card for the default subset): greedy
//! decode of N tokens from the prompt where the device feeds its sampled
//! token back to its own token input (`Program::copy`) and the host reads
//! no logits, only the 4-byte token id; then the hand engine (`Glm5Ref`)
//! runs prompt + generated tokens and its argmax over the kept vocab rows
//! must give the same tokens. `--temp T --top-p P --top-k K --seed S`
//! samples instead (checked for run-to-run reproducibility).
//!
//! ```text
//! GLM_REAL_GGUF=/workspace/glm53-l0_3.gguf glm_real_gpu
//! GLM_REAL_GGUF=/workspace/glm53-l0_3.gguf glm_real_gpu --decode 32
//! ```
//!
//! Exits 1 on any disagreement.
use std::time::Instant;

use tang_bench::ir_glm_prefill::{
    add_sampler, prefill_graph, seed_caches, try_step, try_step_sampled, GlmDims, SEQ,
};
use tang_bench::ir_glm_real::*;
use tang_bench::ir_sample::SampleSpec;
use tang_bench::qwen_session::{lower, runtime};
use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_compile::tensor::step::{StepGraph, StepInput};
use tang_compile::tensor::TensorId;
use tang_gpu::program::Program;
use tang_gpu::program::Runtime;

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (x - y).abs() / scale;
            if d.is_nan() {
                f64::INFINITY
            } else {
                d
            }
        })
        .fold(0.0, f64::max)
}

fn first_argmax(r: &[f32]) -> usize {
    (0..r.len())
        .max_by(|&a, &b| r[a].total_cmp(&r[b]).then(b.cmp(&a)))
        .unwrap()
}

/// The step lowered for the device; outputs `[logits, token, caches…]`.
fn build(rt: &Runtime, st: &StepGraph, inputs: &[Vec<f32>]) -> (Program, Vec<TensorId>) {
    let mut ids = st.outputs.clone();
    ids.extend(st.caches.iter().map(|c| c.updated));
    let (g, outs) = st.graph.substitute_maps(&ids);
    let (g, outs) = g.eliminate_common(&outs);
    let dp = lower(rt, &g, &outs, &DecodeOpts::default());
    eprintln!(
        "lowered: {} kernels ({} native, {} top-k block select)",
        dp.program.kernels.len(),
        dp.native.iter().filter(|n| **n).count(),
        dp.program
            .kernels
            .iter()
            .filter(|k| k.source.contains("block select"))
            .count()
    );
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    (Program::new(rt, dp.program, &refs), outs)
}

fn opt<T: std::str::FromStr>(args: &[String], k: &str) -> Option<T> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run = RealRun::from_env().unwrap_or_else(|| panic!("no {}", gguf_path()));
    let rt = runtime();
    eprintln!("runtime: {}", rt.name());
    let ok = match opt::<usize>(&args, "--decode") {
        Some(n) => decode(&run, &rt, &args, n),
        None => check_step(&run, &rt),
    };
    if !ok {
        std::process::exit(1);
    }
}

/// One step on the device against the interpreter (compact experts).
fn check_step(run: &RealRun, rt: &Runtime) -> bool {
    let t_pre: usize = std::env::var("GLM_REAL_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let toks = run.rows_of(&prompt(t_pre + 1));
    let v = run.vocab_ids.len();
    let cap = 8.max(t_pre + 1);
    let t0 = Instant::now();
    let d0 = dims_from_hparams(&run.hp, v, 1);
    let mut w = run.dense_inputs(&prefill_graph(&d0).inputs).unwrap();
    let used = run.route(&toks, &d0, &mut w).unwrap();
    let rows = used.iter().map(|u| u.len()).max().unwrap().max(1);
    let d = GlmDims {
        expert_rows: rows,
        ..d0
    };
    let mut pg = prefill_graph(&d);
    let token = add_sampler(&mut pg, SampleSpec::default());
    for (l, u) in used.iter().enumerate() {
        if !u.is_empty() {
            run.expert_tables(l, u, rows, &d, &mut w).unwrap();
        }
    }
    let greedy = [0.0, 1.0, 0.0, 0.0];
    let mut base: Vec<Vec<f64>> = pg
        .inputs
        .iter()
        .map(|s| match s.name.as_str() {
            "tokens" => toks[..t_pre].to_vec(),
            "sample.params" => greedy.to_vec(),
            n => w.remove(n).unwrap(),
        })
        .collect();
    drop(w);
    eprintln!("weights + routing: {:.1?}", t0.elapsed());

    let t0 = Instant::now();
    let st = try_step_sampled(&pg, token, 1, cap).unwrap();
    let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
    let pre = pg.g.bind(&[(SEQ, t_pre)]).unwrap().eval(&base, &nodes);
    let caches = seed_caches(&pg, &st, &pre, t_pre, cap);
    let step_ins: Vec<Vec<f64>> = st
        .inputs
        .iter()
        .map(|&x| match x {
            StepInput::Pos => vec![t_pre as f64],
            StepInput::New(_) => toks[t_pre..t_pre + 1].to_vec(),
            StepInput::Same(i) => std::mem::take(&mut base[i]),
            StepInput::Cache(c) => caches[c].clone(),
        })
        .collect();
    let mut ids = st.outputs.clone();
    ids.extend(st.caches.iter().map(|c| c.updated));
    let want = st.graph.eval(&step_ins, &ids);
    eprintln!("CPU prefill + interpreter step: {:.1?}", t0.elapsed());

    let t0 = Instant::now();
    let words: Vec<Vec<f32>> = step_ins
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();
    drop(step_ins);
    let (prog, outs) = build(rt, &st, &words);
    eprintln!("{}: lowered + uploaded in {:.1?}", rt.name(), t0.elapsed());
    prog.run(rt);
    let mut worst = 0.0f64;
    for (j, (o, wv)) in outs.iter().zip(&want).enumerate() {
        if j == 1 {
            continue; // the token, below
        }
        let got: Vec<f64> = prog.read(rt, *o).iter().map(|&x| x as f64).collect();
        let (gk, wk): (Vec<f64>, Vec<f64>) = got
            .iter()
            .zip(wv)
            .filter(|(_, w)| !w.is_nan())
            .map(|(a, b)| (*a, *b))
            .unzip();
        worst = worst.max(rel(&gk, &wk));
    }
    let logits = prog.read(rt, outs[0]);
    let tok = prog.read(rt, outs[1])[0] as usize;
    let greedy_ok = tok == first_argmax(&logits);
    let same_as_cpu = tok == want[1][0] as usize;
    // sampled: same seed twice, then another seed
    let k_params = st
        .inputs
        .iter()
        .position(|&si| si == StepInput::Same(pg.input("sample.params")))
        .unwrap();
    let mut draws = Vec::new();
    for seed in [7.0f32, 7.0, 8.0] {
        prog.write(rt, prog.input(k_params), &[0.8, 0.95, 40.0, seed]);
        prog.run(rt);
        draws.push(prog.read(rt, outs[1])[0] as usize);
    }
    let repro = draws[0] == draws[1];
    let ms = prog.time(rt, 20);
    // the same step derived without the sampler: its logits, bit for bit
    let plain = try_step(&pg, 1, cap).unwrap();
    let pwords: Vec<Vec<f32>> = plain
        .inputs
        .iter()
        .map(|&si| {
            let k = st
                .inputs
                .iter()
                .position(|&x| x == si)
                .expect("an input of the sampled step");
            if si == StepInput::Same(pg.input("sample.params")) {
                greedy.iter().map(|&x| x as f32).collect()
            } else {
                words[k].clone()
            }
        })
        .collect();
    let (p0, pouts) = build(rt, &plain, &pwords);
    drop(pwords);
    p0.run(rt);
    let plain_logits = p0.read(rt, pouts[0]);
    let same_bits = plain_logits.len() == logits.len()
        && plain_logits
            .iter()
            .zip(&logits)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let ms0 = p0.time(rt, 20);
    let pass = worst < 1e-3 && greedy_ok && same_as_cpu && repro && same_bits;
    println!(
        "glm-5.3 real size, checkpoint layers {:?}, step at position {t_pre}: GPU vs interpreter rel err {worst:.2e}; \
         greedy token {tok} (first argmax of the device logits: {greedy_ok}; interpreter's: {same_as_cpu}); \
         sampled (T 0.8, top-p 0.95, top-k 40) seeds 7, 7, 8: {draws:?}; logits bit-identical to the step without \
         the sampler: {same_bits}; {ms:.3} ms device with the sampler, {ms0:.3} ms without: {}",
        run.layer_ids,
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// Device decode with the token fed back on the device; checked against
/// the hand engine's greedy argmax over the kept vocab rows.
fn decode(run: &RealRun, rt: &Runtime, args: &[String], n: usize) -> bool {
    let temp: f32 = opt(args, "--temp").unwrap_or(0.0);
    let top_p: f32 = opt(args, "--top-p").unwrap_or(1.0);
    let top_k: f32 = opt(args, "--top-k").unwrap_or(0.0);
    let seed: f32 = opt(args, "--seed").unwrap_or(1.0);
    let prompt_ids = prompt(5);
    let cap = (prompt_ids.len() + n).next_power_of_two().max(16);
    let v = run.vocab_ids.len();
    let d = dims_from_hparams(&run.hp, v, 0);
    let mut pg = prefill_graph(&d);
    let token = add_sampler(&mut pg, SampleSpec::default());
    let st = try_step_sampled(&pg, token, 1, cap).unwrap();
    let t0 = Instant::now();
    // inputs: weights from the GGUF in f32 (all experts), empty caches
    let params = [temp, top_p, top_k, seed];
    let words: Vec<Vec<f32>> = st
        .inputs
        .iter()
        .map(|&si| match si {
            StepInput::Pos => vec![0.0],
            StepInput::New(_) => vec![0.0],
            StepInput::Same(i) => match pg.inputs[i].name.as_str() {
                "sample.params" => params.to_vec(),
                nm => run.input32(nm).unwrap(),
            },
            StepInput::Cache(c) => {
                let len: usize = st
                    .graph
                    .shape(st.caches[c].input)
                    .iter()
                    .map(|d| d.n())
                    .product();
                if st.caches[c].state {
                    vec![0.0; len]
                } else {
                    vec![f32::NAN; len]
                }
            }
        })
        .collect();
    eprintln!(
        "weights (all {} experts per MoE layer, f32): {:.1} GB in {:.1?}",
        d.n_expert,
        words.iter().map(|w| w.len()).sum::<usize>() as f64 * 4e-9,
        t0.elapsed()
    );
    let (prog, outs) = build(rt, &st, &words);
    drop(words);
    let pos_in = st.inputs.iter().position(|s| *s == StepInput::Pos).unwrap();
    let tok_in = st
        .inputs
        .iter()
        .position(|s| matches!(s, StepInput::New(_)))
        .unwrap();
    let cache_ins: Vec<(usize, TensorId)> = (0..st.caches.len())
        .map(|c| {
            let k = st
                .inputs
                .iter()
                .position(|s| *s == StepInput::Cache(c))
                .unwrap();
            (k, outs[2 + c])
        })
        .collect();
    let step = |pos: usize| {
        prog.write(rt, prog.input(pos_in), &[pos as f32]);
        prog.run(rt);
        for &(k, upd) in &cache_ins {
            prog.copy(rt, prog.input(k), &prog, upd);
        }
    };
    let rows = run.rows_of(&prompt_ids);
    let gen = |label: &str| -> Vec<usize> {
        let t0 = Instant::now();
        for (p, &r) in rows.iter().enumerate() {
            prog.write(rt, prog.input(tok_in), &[r as f32]);
            step(p);
        }
        let mut out = Vec::new();
        let t1 = Instant::now();
        for p in rows.len()..rows.len() + n {
            // the sampled token becomes the next input on the device
            prog.copy(rt, prog.input(tok_in), &prog, outs[1]);
            out.push(prog.read(rt, outs[1])[0] as usize);
            if out.len() == n {
                break;
            }
            step(p);
        }
        eprintln!(
            "{label}: prompt {} tokens in {:.2?}; {} tokens in {:.2?} ({:.1} tok/s, 4-byte id read per token, no logits)",
            rows.len(),
            t1 - t0,
            out.len(),
            t1.elapsed(),
            out.len() as f64 / t1.elapsed().as_secs_f64()
        );
        out
    };
    let a = gen("run 1");
    // reset caches for a second run: rows to NaN, states to zero
    for c in 0..st.caches.len() {
        let len: usize = st
            .graph
            .shape(st.caches[c].input)
            .iter()
            .map(|d| d.n())
            .product();
        let fill = if st.caches[c].state { 0.0 } else { f32::NAN };
        prog.write(rt, prog.input(cache_ins[c].0), &vec![fill; len]);
    }
    let b = gen("run 2");
    let ids: Vec<usize> = a.iter().map(|&r| run.vocab_ids[r]).collect();
    println!("tokens (ids): {ids:?}");
    let repro = a == b;
    if temp > 0.0 {
        println!(
            "sampled decode (T {temp}, top-p {top_p}, top-k {top_k}, seed {seed}): run to run {}",
            if repro {
                "identical: PASS"
            } else {
                "DIFFERENT: FAIL"
            }
        );
        return repro;
    }
    // the hand engine over prompt + generated, argmax over the kept rows
    let t0 = Instant::now();
    let mut seq = prompt_ids.clone();
    seq.extend(&ids);
    let want = run.reference(&seq);
    let want_rows: Vec<usize> = (prompt_ids.len() - 1..seq.len() - 1)
        .map(|t| {
            let r: Vec<f32> = want[t * v..(t + 1) * v].iter().map(|&x| x as f32).collect();
            first_argmax(&r)
        })
        .collect();
    let agree = want_rows.iter().zip(&a).take_while(|(x, y)| x == y).count();
    let pass = agree == n && repro;
    println!(
        "greedy device decode vs hand engine (Glm5Ref, {:.1?}): first {agree}/{n} tokens agree; run to run {}: {}",
        t0.elapsed(),
        if repro { "identical" } else { "different" },
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}
