//! The M4's gap to MPS on the fragment workloads, before and after the
//! autotuner: attention forward (f16 q/k/v, f32 accumulate, simdgroup
//! fragments) and f16 matmul on fragments, at the shapes of
//! `scripts/torch_lowering_caps.py`.
//!
//! before = the fragment kernel at its fixed default geometry (#155);
//! after  = `autotune` over the standard space (fragment geometry,
//!          subgroups, thread mappings), lowered with the device's caps.
//!
//! Two timings each: the GPU clock (5th percentile of paired rounds) and
//! the PyTorch script's method (30 calls, one sync, best of 5 rounds —
//! host clock, launch cost included), which is the column to compare
//! with MPS.
//!
//!   cargo run --release -p tang-bench --example mps_gap
use std::time::Instant;
use tang_bench::ir_gpu::{flash_attention_heads_fragment, matmul_candidates};
use tang_compile::tensor::{Format, Precision, TensorGraph, TensorId};
use tang_gpu::autotune::{autotune, program_samples, Stats, TuneOptions};
use tang_gpu::program::{Program, Runtime};

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// 30 runs, one sync, best of 5 (the torch script's method), ms. The
/// programs are [`Program::capture`]d: a run is one command buffer.
fn wall(rt: &Runtime, p: &Program) -> f64 {
    for _ in 0..3 {
        p.run(rt);
    }
    rt.sync();
    (0..5)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..30 {
                p.run(rt);
            }
            rt.sync();
            t.elapsed().as_secs_f64() * 1e3 / 30.0
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let rt = Runtime::wgpu().expect("wgpu");
    eprintln!("device: {}", rt.name());
    type Problem = (String, TensorGraph, TensorId, Vec<Vec<f32>>, f64);
    let mut problems: Vec<Problem> = Vec::new();
    let d = 64;
    for (h, n, causal) in [
        (1usize, 1024usize, false),
        (8, 1024, false),
        (8, 2048, false),
        (8, 2048, true),
        (8, 4096, false),
    ] {
        let (g, y) = flash_attention_heads_fragment(h, 1, n, d, d, causal);
        let ins = vec![rand(h * n * d, 1), rand(h * n * d, 2), rand(h * n * d, 3)];
        let flops = 4.0 * (h * n * n * d) as f64 / if causal { 2.0 } else { 1.0 };
        problems.push((
            format!("attn h{h} n{n}{}", if causal { " causal" } else { "" }),
            g,
            y,
            ins,
            flops,
        ));
    }
    for (m, k, n) in [
        (1024usize, 1024usize, 1024usize),
        (2048, 2048, 2048),
        (512, 576, 1536),
    ] {
        let (mut g, c) = matmul_candidates(m, k, n)
            .into_iter()
            .find(|(nm, ..)| *nm == "auto")
            .map(|(_, g, c)| (g, c))
            .unwrap();
        let f16 = Precision {
            storage: Format::F16,
            accum: Format::F32,
        };
        g.set_precision(TensorId::from_index(0), f16);
        g.set_precision(TensorId::from_index(1), f16);
        g.set_fragment(c, true);
        problems.push((
            format!("matmul {m}x{k}x{n}"),
            g,
            c,
            vec![rand(m * k, 1), rand(k * n, 2)],
            2.0 * (m * k * n) as f64,
        ));
    }
    println!("| workload | before GPU ms | after GPU ms | GPU speedup | before (torch method) | after (torch method) | after TFLOP/s | picked |");
    println!("|---|---|---|---|---|---|---|---|");
    for (name, g, y, inputs, flops) in &problems {
        let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let opts = TuneOptions {
            dir: None,
            ..Default::default()
        };
        let t = Instant::now();
        let (tuned, rep) = autotune(&rt, g.clone(), &[*y], &ins, &opts);
        let picked: Vec<String> = rep.iter().map(|k| k.picked.encode(&opts.space)).collect();
        eprintln!(
            "{name}: tuned in {:.1}s: {picked:?}",
            t.elapsed().as_secs_f64()
        );
        let progs: Vec<Program> = [g, &tuned]
            .iter()
            .map(|gr| {
                let mut p = Program::compile(&rt, gr, &[*y], &ins);
                // one pre-recorded command buffer per run, as MPS caches
                // its pipelines and encoders
                p.capture(&rt);
                p.run(&rt);
                p
            })
            .collect();
        let (a, b) = (progs[0].read(&rt, *y), progs[1].read(&rt, *y));
        let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, z)| (x - z).abs())
            .fold(0f32, f32::max);
        assert!(
            diff <= 2e-2 * scale + 1e-3,
            "{name}: tuned output differs by {diff}"
        );
        // paired rounds on the GPU clock
        let mut r = vec![Vec::new(); 2];
        let mut fast = vec![Vec::new(); 2];
        for _ in 0..10 {
            let f: Vec<f64> = progs
                .iter()
                .map(|p| Stats::of(&program_samples(&rt, p, 2, 10)).fast)
                .collect();
            for i in 0..2 {
                fast[i].push(f[i]);
                r[i].push(f[i] / f[0]);
            }
        }
        let before = Stats::of(&fast[0]).median;
        let ratio = Stats::of(&r[1]).median;
        let after = before * ratio;
        let mut w = [f64::INFINITY; 2];
        for _ in 0..3 {
            for i in 0..2 {
                w[i] = w[i].min(wall(&rt, &progs[i]));
            }
        }
        println!(
            "| {name} | {before:.3} | {after:.3} | {:.2}× | {:.3} | {:.3} | {:.1} | {} |",
            1.0 / ratio,
            w[0],
            w[1],
            flops / w[1] / 1e9,
            picked
                .join(" / ")
                .replace(";fragment=1", "")
                .replace("split=0;tile=0,0,0;reg=1,1;", "")
        );
    }
}
