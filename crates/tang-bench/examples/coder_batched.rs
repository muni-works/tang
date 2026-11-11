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
                .unwrap_or(32)
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
        block: BlockOpts::batched(),
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
        ..
    } = low;
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
            println!(
                "batched decode B={nb} ragged around len {l}: {best:.3} ms device a step = {:.1} tok/s aggregate ({:.1} tok/s a stream)",
                nb as f64 * 1e3 / best,
                1e3 / best
            );
        }
    }
}
