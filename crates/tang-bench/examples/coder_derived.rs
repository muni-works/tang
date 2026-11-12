//! Qwen3-Coder-Next decode from its *derived* step (`ir_coder_real`): the
//! prefill graph written once, `derive_step` for the T → T + 1 step, the
//! GGUF's weights uploaded in their own formats (the int8 GEMV reads
//! Q4_K / Q5_K / Q8_0 as the file stores them; the few tensors no int8
//! GEMV reads go up dequantized), lowered by `to_cuda_decode`.
//!
//! ```text
//! coder_derived <file.gguf> --ids-file F [-n 64] [--ctx 4096] [--top K] [--logprobs OUT.jsonl]
//!               [--out IDS] [--bench L1,L2,…] [--chunk K [--group N]]
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

use tang_bench::ir_coder_prefill::{
    derived_step, derived_step_batched_k, derived_step_rows, prefill_graph,
};
use tang_bench::ir_coder_real::*;
use tang_bench::qwen_session::{runtime, slot_plan};
use tang_compile::tensor::step::StepInput;
use tang_gpu::program::Program;
use tang_llm::gguf::Gguf;

macro_rules! profile {
    ($prog:expr, $rt:expr, $kinfo:expr) => {{
        let prog = &$prog;
        let rt = &$rt;
        let kinfo = &$kinfo;
        // every kernel timed alone at the last position, then grouped
        let mut t: Vec<(f64, usize)> = prog
            .order()
            .iter()
            .map(|&i| (prog.time_kernel(rt, i, 5), i))
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
            println!(
                "  shape {ms:8.3} ms over {n:5} kernels ({:6.1} µs each): {k}",
                1e3 * ms / *n as f64
            );
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
    }};
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
    // --chunk K: the derived step appends K tokens; the prompt runs in
    // K-token chunks (chunked prefill), timed, and the run stops there
    let chunk: usize = opt("--chunk").map_or(1, |v| v.parse().unwrap());

    let t0 = Instant::now();
    let g = Gguf::open(Path::new(&path)).unwrap();
    let dims = dims_from_gguf(&g).unwrap();
    let pg = prefill_graph(&dims);
    let tys = bind(&pg, &g).unwrap();
    // a prefill chunk needs only its last row's logits (all rows for --check)
    // --batch B (with --chunk K): packed prefill, B prompts (the ids file
    // rotated by b·997 tokens for prompt b) appending K tokens each per step
    let batch: usize = opt("--batch").map_or(1, |v| v.parse().unwrap());
    let st = if batch > 1 {
        derived_step_batched_k(&pg, chunk, ctx, tang_compile::tensor::step::Rows::Last).unwrap()
    } else if chunk > 1 && opt("--check").is_none() {
        derived_step_rows(&pg, chunk, ctx, tang_compile::tensor::step::Rows::Last).unwrap()
    } else {
        derived_step(&pg, chunk, ctx).unwrap()
    };
    // the device first: its tuning cache (tune/<gpu>.tsv) steers the lowering
    let rt = runtime();
    let tuned = tang_compile::tensor::tune::load_for_gpu(rt.name());
    eprintln!("runtime: {}; tuning cache: {tuned} entries", rt.name());
    // --group N: routed experts sorted by expert over the chunk's
    // (token, slot) positions once there are N of them (one weight read
    // per expert run: the grouped expert GEMM of layer-wise prefill)
    let group: usize = opt("--group").map_or(0, |v| v.parse().unwrap());
    let low = lower_real_step_with(
        &pg,
        &st,
        &tys,
        &RealOpts {
            group,
            batch: (batch > 1).then_some(batch),
            ..Default::default()
        },
    );
    let native = low.program.native.iter().filter(|n| **n).count();
    eprintln!(
        "derived step: {} kernels ({native} native, {} int8 GEMVs), {} caches; f32 uploads {:?}; built in {:.1}s",
        low.program.program.kernels.len(),
        low.sched.gemvs,
        st.caches.len(),
        low.f32_inputs,
        t0.elapsed().as_secs_f64()
    );
    // (dumped residual buffers stay live to the end, out of slot reuse)
    let mut plan_outs = low.outs.clone();
    if opt("--dump-resid").is_some() {
        plan_outs.extend(residual_stream(&low.graph, &pg));
    }
    let plan = slot_plan(&low.program.program, &plan_outs, &low.inplace);
    let empty: Vec<&[f32]> = vec![&[]; st.inputs.len()];
    // --dump-resid FILE: every residual-stream buffer after the prompt
    // (chunk mode) or after each prompt step (decode mode), f32 LE
    let resid = residual_stream(&low.graph, &pg);
    let dump = opt("--dump-resid");
    let mut dumped: Vec<f32> = Vec::new();
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
            let head = k.source.lines().next().unwrap_or("").to_string();
            // nodes the lowering added (a split factor) have no entry here
            let Some(nd) = graph.nodes().get(k.node.index()) else {
                return ("(lowering node) gathers 0".to_string(), nat, head);
            };
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
            (what, nat, head)
        })
        .collect();
    let mut prog = Program::with_plan(&rt, program.program, &empty, &plan);

    // weights, streamed from the mapping one tensor at a time
    let t1 = Instant::now();
    let mut bytes_up = 0usize;
    for (k, si) in st.inputs.iter().enumerate() {
        let StepInput::Same(i) = *si else { continue };
        let t = g.info(&pg.inputs[i].name).unwrap();
        let words = tang_bench::ir_coder_real::encode(&g, t, storage[k]).unwrap();
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
    if batch > 1 {
        let n = ids.len() / chunk * chunk;
        let prompts: Vec<Vec<f32>> = (0..batch)
            .map(|b| {
                (0..n)
                    .map(|i| ids[(i + b * 997) % ids.len()] as f32)
                    .collect()
            })
            .collect();
        let t = Instant::now();
        for c in 0..n / chunk {
            prog.write(&rt, prog.input(pos_in), &vec![(c * chunk) as f32; batch]);
            let toks: Vec<f32> = prompts
                .iter()
                .flat_map(|p| p[c * chunk..(c + 1) * chunk].to_vec())
                .collect();
            prog.write(&rt, prog.input(tok_in), &toks);
            prog.run(&rt);
            for &(k, upd) in &swaps {
                prog.copy(&rt, prog.input(k), &prog, upd);
            }
        }
        let l = prog.read(&rt, outs[0]);
        let s = t.elapsed().as_secs_f64();
        for b in 0..batch {
            let row = &l[l.len() / batch * (b + 1) - vocab..l.len() / batch * (b + 1)];
            println!("prompt {b} last row top: {:?}", logprobs(row));
        }
        println!(
            "packed prefill: {batch} prompts x {n} tokens in chunks of {chunk}: {:.1} ms = {:.1} tok/s",
            s * 1e3,
            (batch * n) as f64 / s
        );
        if args.iter().any(|a| a == "--profile") {
            profile!(prog, rt, kinfo);
        }
        return;
    }
    if chunk > 1 {
        let n = ids.len() / chunk * chunk;
        let toks: Vec<f32> = ids[..n].iter().map(|&t| t as f32).collect();
        // no warm-up chunk: a run updates caches in place, and the timed
        // run must start from the empty session
        let check_from: Option<usize> = opt("--check").map(|v| v.parse().unwrap());
        let (mut agree, mut total) = (0usize, 0usize);
        let t = Instant::now();
        for c in 0..n / chunk {
            prog.write(&rt, prog.input(pos_in), &[(c * chunk) as f32]);
            prog.write(&rt, prog.input(tok_in), &toks[c * chunk..(c + 1) * chunk]);
            prog.run(&rt);
            for &(k, upd) in &swaps {
                prog.copy(&rt, prog.input(k), &prog, upd);
            }
            // --check: each row's argmax against the next prompt token
            // (teacher-forced; the prompt a greedy k = 1 continuation)
            if check_from.is_some() {
                let l = prog.read(&rt, outs[0]);
                for r in 0..chunk {
                    let p = c * chunk + r;
                    if p + 1 < n && p + 1 >= check_from.unwrap() {
                        let row = &l[r * vocab..(r + 1) * vocab];
                        let am = (0..vocab).fold(0, |b, i| if row[i] > row[b] { i } else { b });
                        agree += usize::from(am == ids[p + 1]);
                        total += 1;
                    }
                }
            }
        }
        if total > 0 {
            println!("teacher-forced argmax = next token: {agree}/{total}");
        }
        let l = prog.read(&rt, outs[0]);
        let s = t.elapsed().as_secs_f64();
        if let Some(p) = &dump {
            for &r in &resid {
                dumped.extend(prog.read(&rt, r));
            }
            let b: Vec<u8> = dumped.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(p, b).unwrap();
            eprintln!("dumped {} residual buffers", resid.len());
        }
        println!("last row top: {:?}", logprobs(&l[l.len() - vocab..]));
        println!(
            "chunked prefill: {n} tokens in chunks of {chunk}: {:.1} ms = {:.1} tok/s ({:.2} ms a chunk)",
            s * 1e3,
            n as f64 / s,
            s * 1e3 / (n / chunk) as f64
        );
        if args.iter().any(|a| a == "--profile") {
            profile!(prog, rt, kinfo);
        }
        return;
    }
    let mut lines = Vec::new();
    let t2 = Instant::now();
    for (p, &tok) in ids.iter().enumerate() {
        step(tok, p);
        if dump.is_some() {
            for &r in &resid {
                dumped.extend(prog.read(&rt, r));
            }
        }
    }
    if let Some(p) = &dump {
        let b: Vec<u8> = dumped.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(p, b).unwrap();
        eprintln!("dumped {} residual buffers a step", resid.len());
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
        profile!(prog, rt, kinfo);
    }
    // --tune: the autotuner over this step on the live buffers (after the
    // generation: variant steps write the shared caches), cache written to
    // TANG_TUNE_OUT or tune/<gpu>.tsv
    #[cfg(feature = "cuda")]
    if args.iter().any(|a| a == "--tune") {
        use tang_bench::autotune;
        use tang_gpu::program::Shared;
        // positions: the --bench lengths (default 128,4096)
        let pos: Vec<usize> = opt("--bench")
            .unwrap_or_else(|| "128,4096".into())
            .split(',')
            .map(|v| v.parse::<usize>().unwrap().min(ctx - 1))
            .collect();
        prog.write(&rt, prog.input(tok_in), &[ids[0] as f32]);
        let set_pos = |q: usize| prog.write(&rt, prog.input(pos_in), &[q as f32]);
        let reps: usize = std::env::var("TUNE_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let t = autotune::tune_step(
            &rt,
            "coder-next",
            &[("TANG_STAGED_MAP", &["1"]), ("TANG_COKERNEL", &["1"])],
            || {
                let low = lower_real_step(&pg, &st, &tys);
                let plan = slot_plan(&low.program.program, &low.outs, &low.inplace);
                let shared: Vec<Shared> = (0..st.inputs.len())
                    .map(|k| Shared {
                        input: k,
                        from: &prog,
                        tensor: prog.input(k),
                    })
                    .collect();
                Program::with_shared(&rt, low.program.program, &empty, &plan, &shared)
            },
            reps,
            &pos,
            &set_pos,
            &|p: &Program| p.read(&rt, outs[0]),
        );
        autotune::report(&t);
        let path = std::env::var("TANG_TUNE_OUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| tang_compile::tensor::tune::repo_file(rt.name()));
        autotune::write_cache(&path, rt.name(), "coder-next", &t);
        println!("tuning cache written: {}", path.display());
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
