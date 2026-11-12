//! Qwen3-Coder-Next batched decode from its derived step: `batch_graph`
//! adds a dim of `B` independent streams to the prefill graph,
//! `derive_step_with` steps them together (one `p0` a stream), routed
//! experts are grouped by expert across streams (`schedule_expert_groups`)
//! and the int8 contractions are staged by their positions (`BlockOpts`:
//! pos-tiled dp4a GEMV at small B, int8 tensor cores from
//! `TANG_BATCH_MMA_MIN` positions).
//!
//! ```text
//! coder_batched <file.gguf> --batch B [--ctx 2048] [--lens 128,1024] [--group G]
//!               [--check-ids F --check-steps N --check-out F | --check-in F]
//! ```
//!
//! `--lens`: device ms per step (best of 3×20 replays) with the streams at
//! ragged lengths around each `L` (stream `b` at `L·(1/2 + b/B)`, capped by
//! the context), and the aggregate tok/s `B / step`.
//!
//! The check: stream `s` of `B` is fed `ids[i] + 7·s` (mod vocab) for
//! `--check-steps` steps from position 0 (caches zeroed), teacher-forced;
//! `--check-out` writes each stream's last logits, `--check-in` compares
//! with a file written by a run at another `B` (`B = 1` runs the streams
//! one after another): top-1 agreement and the worst rel err.
use std::path::Path;
use std::time::Instant;

use tang_bench::ir_coder_prefill::{derived_step_batched, prefill_graph};
use tang_bench::ir_coder_real::*;
use tang_bench::qwen_session::{runtime, slot_plan};
use tang_compile::tensor::int8_sched::BlockOpts;
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
    let nb: usize = opt("--batch").map_or(1, |v| v.parse().unwrap());
    let ctx: usize = opt("--ctx").map_or(2048, |v| v.parse().unwrap());
    let group: usize = opt("--group").map_or_else(
        || {
            std::env::var("TANG_BATCH_GROUP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64)
        },
        |v| v.parse().unwrap(),
    );
    // streams processed together (the check at B = 1 runs S streams one at a time)
    let streams: usize = opt("--streams").map_or(nb, |v| v.parse().unwrap());

    let t0 = Instant::now();
    let g = Gguf::open(Path::new(&path)).unwrap();
    let dims = dims_from_gguf(&g).unwrap();
    let pg = prefill_graph(&dims);
    let tys = bind(&pg, &g).unwrap();
    let st = derived_step_batched(&pg, ctx).unwrap();
    let ro = RealOpts {
        batch: Some(nb),
        group,
        block: Some(BlockOpts::batched()),
        // `TANG_Q6K_Q8=0`: keep Q6_K tensors as f32
        q6k_as_q8_0: std::env::var("TANG_Q6K_Q8").as_deref() != Ok("0"),
        // `TANG_KV_F16=1`: K/V caches stored f16
        kv_f16: std::env::var("TANG_KV_F16").as_deref() == Ok("1"),
    };
    let low = lower_real_step_with(&pg, &st, &tys, &ro);
    let native = low.program.native.iter().filter(|n| **n).count();
    let stages = {
        let mut m = std::collections::BTreeMap::<String, usize>::new();
        for n in low.graph.nodes() {
            if let Some(s) = n.int_mma {
                *m.entry(format!("{:?}/pt{}", s.stage, s.pos_tile))
                    .or_default() += 1;
            }
        }
        m
    };
    eprintln!(
        "batched step B={nb}: {} kernels ({native} native, {} int8 contractions {stages:?}), {} caches; built in {:.1}s",
        low.program.program.kernels.len(),
        low.sched.gemvs,
        st.caches.len(),
        t0.elapsed().as_secs_f64()
    );
    // element counts of the step inputs
    let input_len: Vec<usize> = (0..st.inputs.len())
        .map(|k| {
            let id = (0..low.graph.nodes().len())
                .find(|&i| {
                    matches!(low.graph.nodes()[i].op, tang_compile::tensor::Op::Input(j) if j == k)
                })
                .unwrap();
            low.graph.nodes()[id]
                .shape
                .iter()
                .map(|d| d.fixed().unwrap())
                .product()
        })
        .collect();
    let rt = runtime();
    let plan = slot_plan(&low.program.program, &low.outs, &low.inplace);
    let empty: Vec<&[f32]> = vec![&[]; st.inputs.len()];
    let RealStep {
        program,
        outs,
        storage,
        graph,
        ..
    } = low;
    let kinfo: Vec<(String, String)> = program
        .program
        .kernels
        .iter()
        .map(|k| {
            // (the lowering may add nodes past the returned graph)
            let Some(nd) = graph.nodes().get(k.node.index()) else {
                let head: String = k
                    .source
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(60)
                    .collect();
                return (format!("n{} (lowering-added)", k.node.index()), head);
            };
            let what = match graph.comp_at(k.node.index()) {
                Some(c) => format!(
                    "{:?} red {:?} gathers {} int8 {:?}",
                    nd.shape,
                    c.red,
                    c.args.iter().filter(|a| a.view.is_gather()).count(),
                    nd.int_mma.map(|m| (m.stage, m.pos_tile))
                ),
                None => format!("{:?} (not a comp)", nd.shape),
            };
            (
                what,
                k.source
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(60)
                    .collect(),
            )
        })
        .collect();
    // `--no-plan`: every buffer its own allocation (no aliasing)
    let mut prog = if args.iter().any(|a| a == "--no-plan") {
        Program::new(&rt, program.program, &empty)
    } else {
        Program::with_plan(&rt, program.program, &empty, &plan)
    };
    let t1 = Instant::now();
    let mut bytes_up = 0usize;
    for (k, si) in st.inputs.iter().enumerate() {
        let StepInput::Same(i) = *si else { continue };
        let t = g.info(&pg.inputs[i].name).unwrap();
        let words = match (storage[k], t.ty) {
            // a Q6_K tensor stored Q8_0 (`RealOpts::q6k_as_q8_0`)
            (Some(tang_compile::tensor::Format::Q8_0), GgmlType::Q6K) => {
                words_of_bytes(&q8_0_bytes(&g.dequantize(t).unwrap()))
            }
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
    if !args.iter().any(|a| a == "--no-graph") {
        prog.capture(&rt);
    }
    let pos_in = st.inputs.iter().position(|s| *s == StepInput::Pos).unwrap();
    let tok_in = st
        .inputs
        .iter()
        .position(|s| matches!(s, StepInput::New(_)))
        .unwrap();
    let cache_ins: Vec<usize> = (0..st.inputs.len())
        .filter(|&k| matches!(st.inputs[k], StepInput::Cache(_)))
        .collect();
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
    let step = |toks: &[f32], pos: &[f32]| {
        prog.write(&rt, prog.input(pos_in), pos);
        prog.write(&rt, prog.input(tok_in), toks);
        prog.run(&rt);
        for &(k, upd) in &swaps {
            prog.copy(&rt, prog.input(k), &prog, upd);
        }
    };

    if let Some(f) = opt("--check-ids") {
        let ids: Vec<usize> = std::fs::read_to_string(f)
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let steps: usize = opt("--check-steps").map_or(ids.len(), |v| v.parse().unwrap());
        let mut last: Vec<Vec<f32>> = vec![Vec::new(); streams];
        for g0 in (0..streams).step_by(nb) {
            for &k in &cache_ins {
                prog.write(&rt, prog.input(k), &vec![0.0; input_len[k]]);
            }
            for i in 0..steps {
                let toks: Vec<f32> = (0..nb)
                    .map(|b| ((ids[i % ids.len()] + 7 * (g0 + b)) % vocab) as f32)
                    .collect();
                step(&toks, &vec![i as f32; nb]);
            }
            let l = prog.read(&rt, outs[0]);
            for b in 0..nb.min(streams - g0) {
                last[g0 + b] = l[b * vocab..(b + 1) * vocab].to_vec();
            }
        }
        let top1 = |l: &[f32]| {
            (0..l.len())
                .max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a)))
                .unwrap()
        };
        if let Some(p) = opt("--check-out") {
            let s: Vec<String> = last
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|v| format!("{v:e}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            std::fs::write(p, s.join("\n") + "\n").unwrap();
        }
        if let Some(p) = opt("--check-in") {
            let want: Vec<Vec<f32>> = std::fs::read_to_string(p)
                .unwrap()
                .lines()
                .map(|l| l.split(' ').map(|v| v.parse().unwrap()).collect())
                .collect();
            let (mut agree, mut worst) = (0, 0.0f64);
            let mut per = Vec::new();
            for (a, w) in last.iter().zip(&want) {
                agree += usize::from(top1(a) == top1(w));
                let scale = w.iter().fold(1e-30f64, |m, &x| m.max(x.abs() as f64));
                let e = a
                    .iter()
                    .zip(w)
                    .map(|(&x, &y)| ((x - y) as f64).abs())
                    .fold(0.0, f64::max)
                    / scale;
                worst = worst.max(e);
                per.push(format!("{e:.1e}"));
            }
            println!(
                "check B={nb}: {streams} streams x {steps} teacher-forced steps vs the reference \
                 run: top-1 agree {agree}/{streams}, worst logit rel err {worst:.2e}; per stream [{}]",
                per.join(" ")
            );
        }
    }

    // greedy generation: stream s's prompt is `--gen-ids` with every id
    // shifted by 7·s (mod vocab); all streams step together from position 0,
    // then `--gen` greedy tokens each. Writes `D/ids-s.txt` per stream and the
    // top-5 logprobs `D/s.jsonl` (the race's format, from the last prompt
    // token on) for the first `--gen-logprobs` streams.
    if let Some(f) = opt("--gen-ids") {
        let base: Vec<usize> = std::fs::read_to_string(f)
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let n: usize = opt("--gen").map_or(64, |v| v.parse().unwrap());
        let dir = opt("--gen-dir").unwrap_or_else(|| ".".into());
        let nlp: usize = opt("--gen-logprobs").map_or(2, |v| v.parse().unwrap());
        std::fs::create_dir_all(&dir).unwrap();
        for &k in &cache_ins {
            prog.write(&rt, prog.input(k), &vec![0.0; input_len[k]]);
        }
        let prompt =
            |s: usize| -> Vec<usize> { base.iter().map(|&t| (t + 7 * s) % vocab).collect() };
        let prompts: Vec<Vec<usize>> = (0..nb).map(prompt).collect();
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); nb];
        let mut lines: Vec<Vec<String>> = vec![Vec::new(); nb.min(nlp)];
        let lp = base.len();
        let t = Instant::now();
        let mut next: Vec<usize> = prompts.iter().map(|p| p[0]).collect();
        for i in 0..lp + n - 1 {
            let toks: Vec<f32> = (0..nb)
                .map(|b| (if i < lp { prompts[b][i] } else { next[b] }) as f32)
                .collect();
            step(&toks, &vec![i as f32; nb]);
            if i + 1 < lp {
                continue;
            }
            let l = prog.read(&rt, outs[0]);
            for b in 0..nb {
                let row = &l[b * vocab..(b + 1) * vocab];
                let m = row.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x)) as f64;
                let z: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
                let mut idx: Vec<usize> = (0..vocab).collect();
                idx.sort_by(|&a, &c| row[c].partial_cmp(&row[a]).unwrap().then(a.cmp(&c)));
                next[b] = idx[0];
                out[b].push(idx[0]);
                if b < lines.len() {
                    let top: Vec<String> = idx[..5]
                        .iter()
                        .map(|&j| format!("[{j}, {:.6}]", row[j] as f64 - m - z.ln()))
                        .collect();
                    lines[b].push(format!("{{\"pos\": {i}, \"top\": [{}]}}", top.join(", ")));
                }
            }
        }
        let secs = t.elapsed().as_secs_f64();
        for b in 0..nb {
            let s = |v: &[usize]| {
                v.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            // (newline-terminated: `cat prompt ids` must not join two ids)
            std::fs::write(format!("{dir}/prompt-{b}.txt"), s(&prompts[b]) + "\n").unwrap();
            std::fs::write(format!("{dir}/ids-{b}.txt"), s(&out[b]) + "\n").unwrap();
        }
        for (b, l) in lines.iter().enumerate() {
            std::fs::write(format!("{dir}/{b}.jsonl"), l.join("\n") + "\n").unwrap();
        }
        println!(
            "gen B={nb}: {} prompt + {n} greedy tokens a stream in {secs:.2}s end to end ({:.1} tok/s aggregate incl. host sampling)",
            lp,
            (nb * (lp + n)) as f64 / secs
        );
    }

    if let Some(ls) = opt("--lens") {
        for l in ls.split(',').map(|v| v.parse::<usize>().unwrap()) {
            let pos: Vec<f32> = (0..nb)
                .map(|b| {
                    let p = l as f64 * (0.5 + b as f64 / nb as f64);
                    (p as usize).clamp(1, ctx - 1) as f32
                })
                .collect();
            let toks: Vec<f32> = (0..nb).map(|b| ((17 + 7 * b) % vocab) as f32).collect();
            prog.write(&rt, prog.input(pos_in), &pos);
            prog.write(&rt, prog.input(tok_in), &toks);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                best = best.min(prog.time(&rt, 20));
            }
            if args.iter().any(|a| a == "--profile") {
                let mut t: Vec<(f64, usize)> = prog
                    .order()
                    .iter()
                    .map(|&i| (prog.time_kernel(&rt, i, 5), i))
                    .collect();
                let total: f64 = t.iter().map(|x| x.0).sum();
                t.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                println!(
                    "profile B={nb} len {l}: {} kernels, sum {total:.3} ms",
                    t.len()
                );
                for (ms, i) in t.iter().take(15) {
                    println!("  {ms:8.4} ms  k{i:<5} {} | {}", kinfo[*i].0, kinfo[*i].1);
                }
                let mut by: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
                for (ms, i) in &t {
                    // group by kind: int8 stage, or shape rank + first line
                    let w = &kinfo[*i].0;
                    let key = if let Some(p) = w.find("int8 Some") {
                        w[p..].to_string()
                    } else {
                        format!(
                            "{} {}",
                            if w.contains("gathers 0") {
                                "plain"
                            } else {
                                "gather"
                            },
                            kinfo[*i].1
                        )
                    };
                    let e = by.entry(key).or_default();
                    e.0 += 1;
                    e.1 += ms;
                }
                let mut by: Vec<_> = by.into_iter().collect();
                by.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
                for (k, (n, ms)) in by.iter().take(14) {
                    println!("  group {ms:8.3} ms over {n:5} kernels: {k}");
                }
            }
            println!(
                "batched decode B={nb} ragged around len {l}: {best:.3} ms device a step = {:.1} tok/s aggregate ({:.1} tok/s a stream)",
                nb as f64 * 1e3 / best,
                1e3 / best
            );
        }
    }
}
