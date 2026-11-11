//! Qwen3-Coder-Next decode from its *derived* step (`ir_coder_real`): the
//! prefill graph written once, `derive_step` for the T → T + 1 step, the
//! GGUF's weights uploaded in their own formats (the int8 GEMV reads
//! Q4_K / Q5_K / Q8_0 as the file stores them; the few tensors no int8
//! GEMV reads go up dequantized), lowered by `to_cuda_decode`.
//!
//! ```text
//! coder_derived <file.gguf> --ids-file F [-n 64] [--ctx 4096] [--top K] [--logprobs OUT.jsonl]
//!               [--out IDS] [--bench L1,L2,…]
//! ```
//!
//! The prompt runs through decode steps (one token a step), then `-n`
//! greedy tokens. `--logprobs` writes one JSON line per position from the
//! last prompt token on, `{"pos": i, "top": [[id, logprob], …]}`, the
//! format `tang-llm flash-ref` prints. `--bench` times one step (device
//! ms, best of 3×20 replays) at each context length.
//!
//! Caches: full-capacity K/V rows update in place (`DecodeOpts::inplace`);
//! the GDN conv rings and states are written to fresh buffers and copied
//! back after each step (device copies).
use std::path::Path;
use std::time::Instant;

use tang_bench::ir_coder_prefill::{derived_step, prefill_graph};
use tang_bench::ir_coder_real::*;
use tang_bench::qwen_session::{runtime, slot_plan};
use tang_compile::tensor::step::StepInput;
use tang_gpu::program::Program;
use tang_llm::gguf::{GgmlType, Gguf};

fn words_of_bytes(b: &[u8]) -> Vec<f32> {
    let mut w: Vec<f32> = b
        .chunks(4)
        .map(|c| {
            let mut a = [0u8; 4];
            a[..c.len()].copy_from_slice(c);
            f32::from_le_bytes(a)
        })
        .collect();
    w.shrink_to_fit();
    w
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let path = args.first().cloned().unwrap_or_else(gguf_path);
    let ids: Vec<usize> = std::fs::read_to_string(opt("--ids-file").expect("--ids-file"))
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect();
    let gen: usize = opt("-n").map_or(64, |v| v.parse().unwrap());
    let ctx: usize = opt("--ctx").map_or(4096, |v| v.parse().unwrap());
    let top: usize = opt("--top").map_or(5, |v| v.parse().unwrap());

    let t0 = Instant::now();
    let g = Gguf::open(Path::new(&path)).unwrap();
    let dims = dims_from_gguf(&g).unwrap();
    let pg = prefill_graph(&dims);
    let tys = bind(&pg, &g).unwrap();
    let st = derived_step(&pg, 1, ctx).unwrap();
    let low = lower_real_step(&pg, &st, &tys);
    let native = low.program.native.iter().filter(|n| **n).count();
    eprintln!(
        "derived step: {} kernels ({native} native, {} int8 GEMVs), {} caches; f32 uploads {:?}; built in {:.1}s",
        low.program.program.kernels.len(),
        low.sched.gemvs,
        st.caches.len(),
        low.f32_inputs,
        t0.elapsed().as_secs_f64()
    );
    let rt = runtime();
    eprintln!("runtime: {}", rt.name());
    let plan = slot_plan(&low.program.program, &low.outs, &low.inplace);
    let empty: Vec<&[f32]> = vec![&[]; st.inputs.len()];
    let RealStep {
        program,
        outs,
        storage,
        graph,
        ..
    } = low;
    let kinfo: Vec<(String, bool, String)> = program
        .program
        .kernels
        .iter()
        .zip(&program.native)
        .map(|(k, &nat)| {
            let nd = &graph.nodes()[k.node.index()];
            let what = match graph.comp_at(k.node.index()) {
                Some(c) => format!(
                    "{:?} red {:?} gathers {} int8 {}",
                    nd.shape,
                    c.red,
                    c.args.iter().filter(|a| a.view.is_gather()).count(),
                    nd.int_mma.is_some()
                ),
                None => format!("{:?} (not a comp) gathers 0", nd.shape),
            };
            (what, nat, k.source.lines().next().unwrap_or("").to_string())
        })
        .collect();
    let mut prog = Program::with_plan(&rt, program.program, &empty, &plan);

    // weights, streamed from the mapping one tensor at a time
    let t1 = Instant::now();
    let mut bytes_up = 0usize;
    for (k, si) in st.inputs.iter().enumerate() {
        let StepInput::Same(i) = *si else { continue };
        let t = g.info(&pg.inputs[i].name).unwrap();
        let words = match (storage[k], t.ty) {
            (Some(_), _) | (None, GgmlType::F32) => words_of_bytes(g.bytes(t)),
            (None, _) => g.dequantize(t).unwrap(),
        };
        bytes_up += words.len() * 4;
        prog.write(&rt, prog.input(k), &words);
    }
    eprintln!(
        "weights uploaded: {:.1} GB in {:.1}s",
        bytes_up as f64 / 1e9,
        t1.elapsed().as_secs_f64()
    );
    // one CUDA graph a step (writes and cache copies stay outside it)
    if !args.iter().any(|a| a == "--no-graph") {
        prog.capture(&rt);
    }
    let pos_in = st.inputs.iter().position(|s| *s == StepInput::Pos).unwrap();
    let tok_in = st
        .inputs
        .iter()
        .position(|s| matches!(s, StepInput::New(_)))
        .unwrap();
    // (input, updated) of the caches written to fresh buffers
    let swaps: Vec<(usize, tang_compile::tensor::TensorId)> = st
        .caches
        .iter()
        .enumerate()
        .filter(|(_, c)| c.state || c.ring.is_some())
        .map(|(c, _)| {
            let k = st
                .inputs
                .iter()
                .position(|s| *s == StepInput::Cache(c))
                .unwrap();
            (k, outs[1 + c])
        })
        .collect();
    let vocab = dims.vocab;
    let step = |tok: usize, pos: usize| {
        prog.write(&rt, prog.input(pos_in), &[pos as f32]);
        prog.write(&rt, prog.input(tok_in), &[tok as f32]);
        prog.run(&rt);
        for &(k, upd) in &swaps {
            prog.copy(&rt, prog.input(k), &prog, upd);
        }
    };
    let logprobs = |l: &[f32]| -> Vec<(usize, f64)> {
        let m = l.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x)) as f64;
        let z: f64 = l.iter().map(|&x| (x as f64 - m).exp()).sum();
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        idx.into_iter()
            .take(top)
            .map(|i| (i, l[i] as f64 - m - z.ln()))
            .collect()
    };
    let mut lines = Vec::new();
    let t2 = Instant::now();
    for (p, &tok) in ids.iter().enumerate() {
        step(tok, p);
    }
    let prompt_s = t2.elapsed().as_secs_f64();
    let mut out = Vec::new();
    let t3 = Instant::now();
    for pos in (ids.len()..).take(gen) {
        let l = prog.read(&rt, outs[0]);
        assert_eq!(l.len(), vocab);
        let lp = logprobs(&l);
        lines.push(format!(
            "{{\"pos\": {}, \"top\": [{}]}}",
            pos - 1,
            lp.iter()
                .map(|(i, v)| format!("[{i}, {v:.6}]"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let next = lp[0].0;
        out.push(next);
        if pos >= ctx {
            break;
        }
        step(next, pos);
    }
    let dec_s = t3.elapsed().as_secs_f64();
    println!(
        "prompt {} tokens in {prompt_s:.2}s (decode steps); decode {} tokens in {dec_s:.2}s = {:.1} tok/s end to end",
        ids.len(),
        out.len(),
        out.len() as f64 / dec_s
    );
    println!(
        "ids: {}",
        out.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if let Some(p) = opt("--out") {
        std::fs::write(
            p,
            out.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )
        .unwrap();
    }
    if let Some(p) = opt("--logprobs") {
        std::fs::write(p, lines.join("\n") + "\n").unwrap();
    }
    if args.iter().any(|a| a == "--profile") {
        // every kernel timed alone at the last position, then grouped
        let mut t: Vec<(f64, usize)> = prog
            .order()
            .iter()
            .map(|&i| (prog.time_kernel(&rt, i, 5), i))
            .collect();
        let total: f64 = t.iter().map(|x| x.0).sum();
        t.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("profile: {} kernels, sum {total:.3} ms", t.len());
        for (ms, i) in t.iter().take(25) {
            let (what, nat, head) = &kinfo[*i];
            println!("  {ms:8.4} ms  k{i:<5} native {nat:<5} {what}  | {head}");
        }
        // by shape: the same op in every layer, summed
        let mut sh: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
        for (ms, i) in &t {
            let e = sh.entry(kinfo[*i].0.as_str()).or_default();
            e.0 += 1;
            e.1 += ms;
        }
        let mut sh: Vec<_> = sh.into_iter().collect();
        sh.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        for (k, (n, ms)) in sh.iter().take(30) {
            println!("  shape {ms:8.3} ms over {n:5} kernels ({:6.1} µs each): {k}", 1e3 * ms / *n as f64);
        }
        // by kind: (native, int8, first line)
        let mut by: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
        for (ms, i) in &t {
            let (what, nat, head) = &kinfo[*i];
            let key = format!(
                "native={nat} int8={} gather={} {head}",
                what.contains("int8 true"),
                !what.contains("gathers 0")
            );
            let e = by.entry(key).or_default();
            e.0 += 1;
            e.1 += ms;
        }
        let mut by: Vec<_> = by.into_iter().collect();
        by.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        for (k, (n, ms)) in by.iter().take(12) {
            println!("  group {ms:8.3} ms over {n:5} kernels: {k}");
        }
    }
    if let Some(ls) = opt("--bench") {
        for l in ls.split(',').map(|v| v.parse::<usize>().unwrap()) {
            let p = l.min(ctx - 1);
            prog.write(&rt, prog.input(pos_in), &[p as f32]);
            prog.write(&rt, prog.input(tok_in), &[ids[0] as f32]);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                best = best.min(prog.time(&rt, 20));
            }
            println!(
                "derived decode step at len {l}: {best:.3} ms device ({:.1} tok/s)",
                1e3 / best
            );
        }
    }
}
