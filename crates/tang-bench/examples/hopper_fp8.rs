//! FP8 matmul on Hopper: `C = A · Bᵀ` with A `[m, k]` and B `[n, k]`
//! stored block-scaled FP8 (`Format::E4M3(128)`: one f32 scale per 128
//! consecutive elements along k, i.e. 1×128 blocks), f32 accumulation,
//! bf16 output. Every shape in `SM90_FP8_SHAPES`, plus the bf16 `wgmma`
//! tiles on the same problem for reference.
//!
//! Error, on 256 sampled entries:
//! - `vs f64`: the GPU result against the exact product of the f32 inputs;
//! - `predicted`: the emulator (`eval_emulated`'s rounding: the inputs
//!   block-quantized, the products in f64) against the same exact product,
//!   computed *before* anything runs, and checked against the
//!   `ErrorBudget` (`TANG_BUDGET`, default 5e-2) — a schedule over budget
//!   is not run;
//! - `vs emu`: the GPU result against the emulator (what is left is f32
//!   accumulation order and the bf16 output rounding).
//!
//! The first (smallest) case also runs the whole graph through
//! `TensorGraph::precision_error` / `eval_emulated` and compares every entry.
//!
//!     cargo run --release -p tang-bench --features cuda --example hopper_fp8
//!     MM_CASES=4096x4096x4096 MM_FMT=e5m2 ...
//!
//! `scripts/hopper_attention.py fp8mm` times cuBLASLt FP8 (`torch._scaled_mm`)
//! on the same shapes.
use tang_compile::tensor::cuda::{MMA_SHAPES, SM90_FP8_SHAPES, SM90_MMA_SHAPES};
use tang_compile::tensor::{ErrorBudget, ErrorStats, Format, Precision, TensorGraph};
use tang_gpu::program::{Program, Runtime};

fn main() {
    let dump: Option<usize> = std::env::var("MM_DUMP").ok().and_then(|v| v.parse().ok());
    let rt = if dump.is_none() {
        Some(Runtime::select().expect("gpu"))
    } else {
        None
    };
    let fp8 = match std::env::var("MM_FMT").ok().as_deref() {
        Some("e5m2") => Format::E5M2(128),
        Some("e4m3t") => Format::E4M3(0), // per-tensor scale
        _ => Format::E4M3(128),
    };
    let budget = ErrorBudget::rel(
        std::env::var("TANG_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5e-2),
    );
    let cases: Vec<(usize, usize, usize)> = match std::env::var("MM_CASES") {
        Ok(v) => v
            .split(',')
            .map(|c| {
                let d: Vec<usize> = c.split('x').map(|x| x.parse().unwrap()).collect();
                (d[0], d[1], d[2])
            })
            .collect(),
        Err(_) => vec![
            (512, 1024, 384),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 2048, 8192),
            (16384, 8192, 2048),
        ],
    };
    let mut rng = tang_bench::Rng::new(5);
    for (ci, &(m, k, n)) in cases.iter().enumerate() {
        let build = |f: Format| {
            let mut g = TensorGraph::new();
            let a = g.input(&[m, k]);
            let b = g.input(&[n, k]);
            let bt = g.transpose(b, &[1, 0]).unwrap();
            let c = g.matmul(a, bt).unwrap();
            let (mut g, outs) = g.substitute_maps(&[c]);
            g.schedule_auto(&outs);
            let c = outs[0];
            g.set_precision(a, Precision::storage(f));
            g.set_precision(b, Precision::storage(f));
            // MM_OUT=f32: f32 output (the GPU vs emulator comparison is then
            // down to accumulation order)
            let out_f = if std::env::var("MM_OUT").ok().as_deref() == Some("f32") {
                Format::F32
            } else {
                Format::BF16
            };
            // MM_ACC=fast: accept tensor-core accumulation over the whole
            // reduction (`Format::Fp8Accum`; tensorwise scales only)
            let accum = if f.is_fp8() && std::env::var("MM_ACC").ok().as_deref() == Some("fast") {
                Format::Fp8Accum
            } else {
                Format::F32
            };
            g.set_precision(c, Precision { storage: out_f, accum, operand: Format::F32 });
            (g, a, b, c)
        };
        let (g, a, b, c) = build(fp8);
        if let Some(v) = dump {
            print!("{}", g.to_cuda_variant(&[c], Some(v)).kernels[0].source);
            return;
        }
        let rt = rt.as_ref().unwrap();
        // N(0,1)-ish entries with a per-row magnitude spread of 2^±4, so the
        // block scales matter
        let mut gen = |rows: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(rows * k);
            for _ in 0..rows {
                let mag = 2f32.powi((rng.f32() * 8.0) as i32 - 4);
                for _ in 0..k {
                    v.push((rng.f32() + rng.f32() + rng.f32() - 1.5) * 2.0 * mag);
                }
            }
            v
        };
        let av = gen(m);
        let bv = gen(n);
        let a64: Vec<f64> = av.iter().map(|&x| x as f64).collect();
        let b64: Vec<f64> = bv.iter().map(|&x| x as f64).collect();
        // the emulator's view of the inputs (eval_emulated rounds them the
        // same way)
        let (mut aq, mut bq) = (a64.clone(), b64.clone());
        g.precision(a).storage.round_buf(&mut aq);
        g.precision(b).storage.round_buf(&mut bq);
        let samples: Vec<(usize, usize)> = (0..256usize)
            .map(|s| ((s * 7919) % m, (s * 104729 + s / 7) % n))
            .collect();
        let dot = |x: &[f64], y: &[f64], i: usize, j: usize| -> f64 {
            (0..k).map(|kk| x[i * k + kk] * y[j * k + kk]).sum()
        };
        let exact: Vec<f64> = samples.iter().map(|&(i, j)| dot(&a64, &b64, i, j)).collect();
        // the emulator's accumulation: the running sum rounded to the
        // reduction's accum format after every addition
        let acc_f = g.precision(c).accum;
        let emu_dot = |i: usize, j: usize, f: &dyn Fn(f64) -> f64| -> f64 {
            (0..k).fold(0.0, |s, kk| f(s + aq[i * k + kk] * bq[j * k + kk]))
        };
        let emu: Vec<f64> = samples
            .iter()
            .map(|&(i, j)| emu_dot(i, j, &|x| acc_f.round(x)))
            .collect();
        let predicted = ErrorStats::compare(&[emu.clone()], &[exact.clone()]);
        let flop = 2.0 * (m * k * n) as f64;
        println!(
            "case {m}x{k}x{n} {fp8:?}: predicted rel_err {:.2e} (max_abs {:.2e}) — {}",
            predicted.max_rel,
            predicted.max_abs,
            if budget.admits(&predicted) { "within budget" } else { "OVER budget, skipped" }
        );
        if !budget.admits(&predicted) {
            continue;
        }
        if ci == 0 && m * k * n <= 1 << 28 {
            // whole-graph emulation through the IR, against the GPU
            let p = Program::new(rt, g.to_cuda_variant(&[c], Some(0)), &[&av, &bv]);
            p.run(rt);
            let got: Vec<f64> = p.read(rt, c).iter().map(|&x| x as f64).collect();
            let inputs = [a64.clone(), b64.clone()];
            let e_pred = g.precision_error(&inputs, &[c]);
            let emu_full = g.eval_emulated(&inputs, &[c]);
            let want_full = g.eval(&inputs, &[c]);
            let e_gpu = ErrorStats::compare(&[got.clone()], &want_full);
            let e_gpu_emu = ErrorStats::compare(&[got], &emu_full);
            println!(
                "  full graph: precision_error (emulated vs f64) max_rel {:.2e} mean_rel {:.2e}; GPU vs f64 max_rel {:.2e} mean_rel {:.2e}; GPU vs emulated max_rel {:.2e}",
                e_pred.max_rel, e_pred.mean_rel, e_gpu.max_rel, e_gpu.mean_rel, e_gpu_emu.max_rel
            );
        }
        let mut best = (f64::INFINITY, 0usize);
        let only: Option<usize> = std::env::var("MM_VARIANT").ok().and_then(|v| v.parse().ok());
        for v in (0..SM90_FP8_SHAPES.len()).filter(|v| only.is_none_or(|o| o == *v)) {
            let prog = g.to_cuda_variant(&[c], Some(v));
            let src = prog.kernels[0].source.clone();
            if !src.contains("FP8") {
                continue;
            }
            let p = Program::new(rt, prog, &[&av, &bv]);
            p.run(rt);
            let got = p.read(rt, c);
            let gs: Vec<f64> = samples.iter().map(|&(i, j)| got[i * n + j] as f64).collect();
            let e64 = ErrorStats::compare(&[gs.clone()], &[exact.clone()]);
            let eemu = ErrorStats::compare(&[gs.clone()], &[emu.clone()]);
            if acc_f == Format::Fp8Accum && v == 0 {
                // calibration: which truncation width matches the GPU
                for bits in [10, 11, 12, 13, 14, 15, 23] {
                    let tr = |x: f64| {
                        let x = x as f32 as f64;
                        if x == 0.0 {
                            return x;
                        }
                        let e = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
                        let q = 2f64.powi(e - bits);
                        (x / q).trunc() * q
                    };
                    let e: Vec<f64> = samples.iter().map(|&(i, j)| emu_dot(i, j, &tr)).collect();
                    let vs_gpu = ErrorStats::compare(&[gs.clone()], &[e.clone()]);
                    let vs_64 = ErrorStats::compare(&[e], &[exact.clone()]);
                    println!(
                        "    truncate to {bits} bits: emulated vs f64 {:.2e} (mean {:.2e}), GPU vs emulated {:.2e} (mean {:.2e})",
                        vs_64.max_rel, vs_64.mean_rel, vs_gpu.max_rel, vs_gpu.mean_rel
                    );
                }
            }
            let t = p.time_kernel(rt, 0, 20);
            let sh = SM90_FP8_SHAPES[v];
            println!(
                "  fp8 {}x{} s{}: {:.4} ms {:.1} TFLOP/s  vs f64 {:.2e}  vs emu {:.2e}",
                sh.bm,
                sh.bn,
                sh.stages,
                t,
                flop / t / 1e9,
                e64.max_rel,
                eemu.max_rel
            );
            if t < best.0 {
                best = (t, v);
            }
        }
        if best.0.is_finite() {
            println!(
                "tang fp8 {m}x{k}x{n}: {:.4} ms {:.1} TFLOP/s (variant {})",
                best.0,
                flop / best.0 / 1e9,
                best.1
            );
        }
        if std::env::var("MM_BF16").ok().as_deref() != Some("0") {
            let (gb, _, _, cb) = build(Format::BF16);
            let mut bb = f64::INFINITY;
            for v in MMA_SHAPES.len()..MMA_SHAPES.len() + SM90_MMA_SHAPES.len() {
                let prog = gb.to_cuda_variant(&[cb], Some(v));
                if !prog.kernels[0].source.contains("wgmma) tile") {
                    continue;
                }
                let p = Program::new(rt, prog, &[&av, &bv]);
                p.run(rt);
                bb = bb.min(p.time_kernel(rt, 0, 20));
            }
            if bb.is_finite() {
                println!("tang bf16 {m}x{k}x{n}: {bb:.4} ms {:.1} TFLOP/s", flop / bb / 1e9);
            }
        }
    }
}
