//! DeepSeek-V4.1-Flash at real size from its derived graph, on the GPU:
//! the real-dims graph of `DS_REAL_LAYERS` checkpoint layers (default 0
//! and 2; `tang_bench::ir_ds_real`), real weights dequantized to f32, the
//! routed experts the CPU check recorded (`routed-*.txt`, written by
//! `ir_ds_real::tests::real_layers_match_reference`) in compact tables.
//! Prefills `DS_REAL_T` tokens (default 3) on the interpreter, seeds the
//! caches, runs the derived one-token step on the device and compares its
//! logits and updated caches with the interpreter's step (f64).
//!
//! `DS_NATIVE=1`: the weights in the checkpoint's formats
//! (`ir_ds_real::lower_native_step`: dense fp8 as `Fp8B32`, routed experts
//! as `Mxfp4`, the int8 GEMV with Q8x32 activations), checked against the
//! interpreter with that Q8x32 rounding emulated (and reported against the
//! pure step). Otherwise every weight is uploaded dequantized to f32.
//! `DS_TIME_ITERS` (default 20): timed steps after the check.
//!
//! Chunked device prefill: `DS_STEP_K=k` derives the step appending `k`
//! tokens (tok/s = k / step time). `DS_ALL_EXPERTS=1` keeps all routed
//! experts in the tables (no `routed-*.txt` needed; real expert bytes).
//! `DS_NO_CHECK=1` skips the interpreter step and only times the device.
//!
//! ```text
//! DS41_DIR=/workspace/ds41-l02 DS_NATIVE=1 ds_real_gpu
//! ```
//!
//! Exits 1 when the device disagrees by more than 1e-3 of the output scale
//! (native: 2e-2 of it, or a different top-1 logit).
use std::time::Instant;

use tang_bench::ir_ds_prefill::{derived_step, prefill_graph, DsDims, SEQ};
use tang_bench::ir_ds_real::*;
use tang_bench::qwen_session::{lower, runtime};
use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_compile::tensor::step::{seed_cache, StepInput};
use tang_compile::tensor::{Dim, TensorId};
use tang_gpu::program::Program;

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

fn main() {
    let run =
        RealRun::from_env().unwrap_or_else(|| panic!("no {}/inference_config.json", ds_dir()));
    let t_pre: usize = std::env::var("DS_REAL_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let k: usize = std::env::var("DS_STEP_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let check = std::env::var("DS_NO_CHECK").as_deref() != Ok("1");
    let toks = run.rows_of(&prompt(t_pre + k));
    let all = std::env::var("DS_ALL_EXPERTS").as_deref() == Ok("1");
    let n_exp = run.cfg["n_routed_experts"].as_u64().unwrap_or(384) as usize;
    let used = if all {
        Ok(vec![(0..n_exp).collect::<Vec<_>>(); run.layer_ids.len()])
    } else {
        run.read_routed(toks.len())
    };
    let used = used.unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run ir_ds_real::tests::real_layers_match_reference first)",
            run.routed_path(toks.len()).display()
        )
    });
    let rows = used.iter().map(|u| u.len()).max().unwrap();
    let mut dm: DsDims = dims_from_config(
        &run.cfg,
        &run.layer_ids,
        run.vocab_ids.len(),
        rows,
        8.max(t_pre + k),
    );
    clamp_index_topk(&mut dm);
    let t0 = Instant::now();
    let pg = prefill_graph(&dm);
    let mut w = run.dense_inputs(&pg.inputs, &dm).unwrap();
    for (l, u) in used.iter().enumerate() {
        run.expert_tables(l, u, rows, &dm, &mut w);
    }
    let mut base: Vec<Vec<f64>> = pg
        .inputs
        .iter()
        .map(|s| {
            if s.name == "tokens" {
                toks[..t_pre].to_vec()
            } else {
                w.remove(&s.name).unwrap()
            }
        })
        .collect();
    eprintln!("weights loaded: {:.1?}", t0.elapsed());

    let t0 = Instant::now();
    let st = derived_step(&pg, &dm, k).unwrap();
    let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
    let pre = pg.g.bind(&[(SEQ, t_pre)]).unwrap().eval(&base, &nodes);
    let caches: Vec<Vec<f64>> = nodes
        .iter()
        .zip(pre)
        .zip(&st.caches)
        .map(|((&n, v), c)| {
            let shape: Vec<usize> =
                pg.g.shape(n)
                    .iter()
                    .map(|d| if *d == Dim::Sym(SEQ) { t_pre } else { d.n() })
                    .collect();
            seed_cache(c, &shape, &v, t_pre, dm.ctx)
        })
        .collect();
    let step_ins: Vec<Vec<f64>> = st
        .inputs
        .iter()
        .map(|&x| match x {
            StepInput::Pos => vec![t_pre as f64],
            StepInput::New(_) => toks[t_pre..t_pre + k].to_vec(),
            StepInput::Same(i) => std::mem::take(&mut base[i]),
            StepInput::Cache(c) => caches[c].clone(),
        })
        .collect();
    let mut ids = st.outputs.clone();
    ids.extend(st.caches.iter().map(|c| c.updated));
    let want = if check {
        st.graph.eval(&step_ins, &ids)
    } else {
        Vec::new()
    };
    eprintln!("CPU prefill + interpreter step: {:.1?}", t0.elapsed());

    let rt = runtime();
    let iters: usize = std::env::var("DS_TIME_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    if std::env::var("DS_NATIVE").as_deref() == Ok("1") {
        let t0 = Instant::now();
        let ns = lower_native_step(&pg, &st, &|n| native_format_of(&run, n));
        let emu = if check {
            ns.graph.eval_emulated(&step_ins, &ns.outs)
        } else {
            Vec::new()
        };
        let words = run
            .native_words(&pg, &st, &ns, &used, rows, &step_ins)
            .unwrap();
        let bytes: usize = words.iter().map(|w| 4 * w.len()).sum();
        let nk = ns.program.program.kernels.len();
        let native = ns.program.native.iter().filter(|n| **n).count();
        let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
        let prog = Program::new(&rt, ns.program.program, &refs);
        eprintln!(
            "{}: native lowering ({} int8 GEMVs, demoted {:?}) + upload of {:.2} GB in {:.1?}",
            rt.name(),
            ns.sched.gemvs,
            ns.demoted,
            bytes as f64 / 1e9,
            t0.elapsed()
        );
        prog.run(&rt);
        if !check {
            let ms = prog.time(&rt, iters);
            println!(
                "deepseek-v4.1 real size, layers {:?}, {k}-token step at position {t_pre} \
                 ({} expert rows): {nk} kernels ({native} native), {ms:.3} ms device \
                 (native fp8 / fp4), {:.1} tok/s (no check)",
                run.layer_ids,
                rows,
                k as f64 * 1e3 / ms
            );
            return;
        }
        let (mut worst, mut worst_pure) = (0.0f64, 0.0f64);
        let mut top1 = true;
        let argmax = |r: &[f64]| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
        for (j, ((o, e), p)) in ns.outs.iter().zip(&emu).zip(&want).enumerate() {
            let got: Vec<f64> = prog.read(&rt, *o).iter().map(|&x| x as f64).collect();
            let keep = |x: &[f64]| -> Vec<f64> {
                x.iter()
                    .zip(p)
                    .filter(|(_, w)| !w.is_nan())
                    .map(|(a, _)| *a)
                    .collect()
            };
            let (g, e, p) = (keep(&got), keep(e), keep(p));
            if j == 0 {
                let last = g.len() - dm.vocab;
                top1 = argmax(&g[last..]) == argmax(&p[last..]);
            }
            worst = worst.max(rel(&g, &e));
            worst_pure = worst_pure.max(rel(&g, &p));
        }
        let ms = prog.time(&rt, iters);
        // Q8x32 codes flip where the device's f32 activation and the
        // interpreter's f64 one straddle a rounding boundary: noise of the
        // quantization's own size (emulated vs pure ~9e-3), so the same gate
        // as native_step_matches_interpreter: 2e-2 and the same top-1.
        let ok = worst < 2e-2 && top1;
        println!(
            "deepseek-v4.1 real size, layers {:?}, step at position {t_pre}: {nk} kernels ({native} native), \
             GPU vs emulated interpreter rel err {worst:.2e} (vs pure f64 step {worst_pure:.2e}), \
             {ms:.3} ms device (native fp8 / fp4 weights, {} layers): {}",
            run.layer_ids,
            dm.n_layers,
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            std::process::exit(1);
        }
        return;
    }
    let t0 = Instant::now();
    let (g, outs) = st.graph.substitute_maps(&ids);
    let (g, outs) = g.eliminate_common(&outs);
    let dp = lower(&rt, &g, &outs, &DecodeOpts::default());
    let native = dp.native.iter().filter(|n| **n).count();
    let nk = dp.program.kernels.len();
    let words: Vec<Vec<f32>> = step_ins
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
    let prog = Program::new(&rt, dp.program, &refs);
    eprintln!("{}: lowered + uploaded in {:.1?}", rt.name(), t0.elapsed());
    prog.run(&rt);
    let mut worst = 0.0f64;
    for (o, wv) in outs.iter().zip(&want) {
        let got: Vec<f64> = prog.read(&rt, *o).iter().map(|&x| x as f64).collect();
        let (gk, wk): (Vec<f64>, Vec<f64>) = got
            .iter()
            .zip(wv)
            .filter(|(_, w)| !w.is_nan())
            .map(|(a, b)| (*a, *b))
            .unzip();
        worst = worst.max(rel(&gk, &wk));
    }
    let ms = prog.time(&rt, iters);
    let ok = worst < 1e-3;
    println!(
        "deepseek-v4.1 real size, layers {:?}, step at position {t_pre}: {nk} kernels ({native} native), \
         GPU vs interpreter rel err {worst:.2e}, {ms:.3} ms device (f32 weights, {} layers): {}",
        run.layer_ids,
        dm.n_layers,
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}
