//! FP8 attention forward on Hopper: the attention written naively in
//! tensor IR with q `[bh, 1, n, d]`, k `[bh, n, d]` and v stored transposed
//! `[bh, d, n]` (FP8 `wgmma` reads K-major operands only, and for P·V the
//! reduction runs over keys), all three `Format::E4M3(128)` (one scale per
//! 128 elements along the contiguous dim: per row of Q and K, per 128 keys
//! of each V column), the online-softmax reduction's `operand` format
//! E4M3 (the softmax weights P are rounded to e4m3 for the tensor cores),
//! bf16 output.
//!
//! Error against the f64 attention of the f32 inputs, on two heads:
//! - `predicted`: the emulator's semantics (inputs block-quantized, the
//!   softmax weights rounded to e4m3, everything else f64), computed before
//!   anything runs and checked against `TANG_BUDGET` (default 5e-2);
//! - `vs f64` and `vs emu`: the GPU result.
//! The first case also runs `precision_error` / `eval_emulated` on a small
//! graph (bh=2, n=256) and compares the GPU with it entry by entry.
//!
//!     cargo run --release -p tang-bench --features cuda --example hopper_fp8_attn
//!     ATT_CASES=4096x128,8192x128 ATT_CAUSAL=1 ...
//!
//! Every case has 16k tokens (bh = 16384·16 / n / ... see `cases`); the
//! FLOP count is FlashAttention's 4·n²·d per head (half when causal).
//! `scripts/hopper_attention.py fp8attn` times FA3 FP8 on the same shapes.
use tang_compile::tensor::cuda::SM90_FP8_FLASH_SHAPES;
use tang_compile::tensor::{
    Arg, ErrorBudget, ErrorStats, Format, Monoid, MonoidKind, Op, Precision, TensorGraph,
    TensorId, View,
};
use tang::Scalar;
use tang_expr::ExprId;
use tang_gpu::program::{Program, Runtime};

fn attention_vt(bh: usize, n: usize, d: usize, causal: bool) -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let q = g.input(&[bh, 1, n, d]);
    let k = g.input(&[bh, n, d]);
    let vt = g.input(&[bh, d, n]);
    let scale = 1.0 / (d as f64).sqrt();
    let qs = g.map(&[q], |x| x[0] * ExprId::from_f64(scale)).unwrap();
    let s = g
        .comp(
            &[bh, 1, n, n],
            &[d],
            vec![
                Arg {
                    src: qs,
                    view: View::pick(5, &[Some(0), Some(1), Some(2), Some(4)]),
                },
                Arg {
                    src: k,
                    view: View::pick(5, &[Some(0), Some(3), Some(4)]),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[1]],
        )
        .unwrap();
    let s = if causal {
        let i = g.iota(&[bh, 1, n, n], 2);
        let j = g.iota(&[bh, 1, n, n], 3);
        g.map(&[s, i, j], |x| {
            let above = x[2] - x[1];
            let m = tang_expr::with_graph(|eg| {
                let neg = eg.lit(-1e30);
                eg.select(above, neg, ExprId::ZERO)
            });
            x[0] + m
        })
        .unwrap()
    } else {
        s
    };
    let m = g.reduce(s, 3, Monoid::max()).unwrap();
    let mb = g.broadcast(m, &[bh, 1, n, n]).unwrap();
    let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
    let l = g.reduce(p, 3, Monoid::sum()).unwrap();
    // space [h, g, i, c | j]; v read transposed: vt[h, c, j]
    let o = g
        .comp(
            &[bh, 1, n, d],
            &[n],
            vec![
                Arg {
                    src: p,
                    view: View::pick(5, &[Some(0), Some(1), Some(2), Some(4)]),
                },
                Arg {
                    src: vt,
                    view: View::pick(5, &[Some(0), Some(3), Some(4)]),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[1]],
        )
        .unwrap();
    let lb = g.broadcast(l, &[bh, 1, n, d]).unwrap();
    let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
    (g, y)
}

fn passes(g: TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, outs) = g.stabilize_softmax(outs);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}

/// The FP8 schedule: inputs 0..3 in `fmt`, the output in `out`, the
/// online-softmax reductions' weights rounded to E4M3.
fn fp8_schedule(g: &mut TensorGraph, outs: &[TensorId], fmt: Format, out: Format) {
    for i in 0..g.nodes().len() {
        let t = TensorId::from_index(i);
        let n = &g.nodes()[i];
        match &n.op {
            Op::Input(_) => g.set_precision(t, Precision::storage(fmt)),
            Op::Comp(c)
                if c.monoid.as_ref().is_some_and(|m| m.kind() == MonoidKind::OnlineSoftmax) =>
            {
                let mut p = g.precision(t);
                p.operand = Format::E4M3(0);
                g.set_precision(t, p);
            }
            _ => {}
        }
    }
    for &o in outs {
        g.set_precision(o, Precision::storage(out));
    }
}

/// One head's attention in f64: `round_p` rounds each softmax weight
/// (relative to the running max, as `eval_emulated` does) before it
/// multiplies v and joins the row sum.
fn head(q: &[f64], k: &[f64], vt: &[f64], n: usize, d: usize, causal: bool, round_p: Option<Format>) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut o = vec![0.0; n * d];
    for i in 0..n {
        let (mut m, mut l) = (f64::NEG_INFINITY, 0.0);
        let mut acc = vec![0.0; d];
        for j in 0..n {
            let mut sv: f64 = (0..d).map(|c| q[i * d + c] * k[j * d + c]).sum::<f64>() * scale;
            if causal && j > i {
                sv -= 1e30;
            }
            let mx = m.max(sv);
            let sa = if m == f64::NEG_INFINITY { 0.0 } else { (m - mx).exp() };
            let mut se = (sv - mx).exp();
            if let Some(f) = round_p {
                se = f.round(se);
            }
            l = l * sa + se;
            for c in 0..d {
                acc[c] = acc[c] * sa + se * vt[c * n + j];
            }
            m = mx;
        }
        for c in 0..d {
            o[i * d + c] = acc[c] / l;
        }
    }
    o
}

fn main() {
    let fmt = match std::env::var("ATT_FMT").ok().as_deref() {
        Some("e4m3t") => Format::E4M3(0),
        Some("e5m2") => Format::E5M2(128),
        _ => Format::E4M3(128),
    };
    let causal = std::env::var("ATT_CAUSAL").ok().as_deref() == Some("1");
    // ATT_DUMP=v: print variant v's kernel for a small case (no GPU)
    if let Some(v) = std::env::var("ATT_DUMP").ok().and_then(|v| v.parse().ok()) {
        let (g, y) = attention_vt(2, 256, 128, causal);
        let (mut g, outs) = passes(g, &[y]);
        fp8_schedule(&mut g, &outs, fmt, Format::BF16);
        for k in g.to_cuda_variant(&outs, Some(v)).kernels {
            println!("{}", k.source);
        }
        return;
    }
    let rt = Runtime::select().expect("gpu");
    let budget = ErrorBudget::rel(
        std::env::var("TANG_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5e-2),
    );
    let cases: Vec<(usize, usize)> = match std::env::var("ATT_CASES") {
        Ok(v) => v
            .split(',')
            .map(|c| {
                let x: Vec<usize> = c.split('x').map(|x| x.parse().unwrap()).collect();
                (x[0], x[1])
            })
            .collect(),
        Err(_) => vec![(1024, 128), (2048, 128), (4096, 128), (8192, 128), (16384, 128)],
    };
    let mut rng = tang_bench::Rng::new(11);
    let mut gauss = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| (rng.f32() + rng.f32() + rng.f32() + rng.f32() - 2.0) * 1.7)
            .collect()
    };

    // whole-graph emulation through the IR, on a small case
    {
        let (bh, n, d) = (2, 256, 128);
        let (g, y) = attention_vt(bh, n, d, causal);
        let (mut g, outs) = passes(g, &[y]);
        fp8_schedule(&mut g, &outs, fmt, Format::F32);
        let ins: Vec<Vec<f32>> = vec![gauss(bh * n * d), gauss(bh * n * d), gauss(bh * n * d)];
        let ins64: Vec<Vec<f64>> = ins.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
        let e_pred = g.precision_error(&ins64, &outs);
        let emu = g.eval_emulated(&ins64, &outs);
        let want = g.eval(&ins64, &outs);
        let refs: Vec<&[f32]> = ins.iter().map(|v| v.as_slice()).collect();
        let p = Program::new(&rt, g.to_cuda_variant(&outs, Some(0)), &refs);
        p.run(&rt);
        let got: Vec<f64> = p.read(&rt, outs[0]).iter().map(|&x| x as f64).collect();
        let e_gpu = ErrorStats::compare(&[got.clone()], &want);
        let e_emu = ErrorStats::compare(&[got], &emu);
        println!(
            "small bh={bh} n={n} d={d} {fmt:?} causal={causal}: precision_error (emulated vs f64) max_rel {:.2e} mean_rel {:.2e}; GPU vs f64 max_rel {:.2e} mean_rel {:.2e}; GPU vs emulated max_rel {:.2e} mean_rel {:.2e}",
            e_pred.max_rel, e_pred.mean_rel, e_gpu.max_rel, e_gpu.mean_rel, e_emu.max_rel, e_emu.mean_rel
        );
    }

    for (n, d) in cases {
        let bh = 16384 / n * 16; // 16k tokens, 16 heads of 128 (2048 hidden)
        let (g, y) = attention_vt(bh, n, d, causal);
        let (mut g, outs) = passes(g, &[y]);
        fp8_schedule(&mut g, &outs, fmt, Format::BF16);
        let q = gauss(bh * n * d);
        let k = gauss(bh * n * d);
        let vt = gauss(bh * n * d);
        // the emulator's view of the inputs, and two heads in f64
        let round = |x: &[f32]| -> (Vec<f64>, Vec<f64>) {
            let full: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let mut r = full.clone();
            fmt.round_buf(&mut r);
            (full, r)
        };
        let ((q64, qq), (k64, kq), (v64, vq)) = (round(&q), round(&k), round(&vt));
        let heads = [0, bh - 1];
        let per = n * d;
        let sl = |x: &[f64], h: usize| x[h * per..(h + 1) * per].to_vec();
        let mut exact = Vec::new();
        let mut emu = Vec::new();
        for &h in &heads {
            exact.push(head(&sl(&q64, h), &sl(&k64, h), &sl(&v64, h), n, d, causal, None));
            emu.push(head(&sl(&qq, h), &sl(&kq, h), &sl(&vq, h), n, d, causal, Some(Format::E4M3(0))));
        }
        let predicted = ErrorStats::compare(&emu, &exact);
        let frac = if causal { 0.5 } else { 1.0 };
        let flop = 4.0 * (bh * n * n * d) as f64 * frac;
        println!(
            "case bh={bh} n={n} d={d} {fmt:?} causal={causal}: predicted max_rel {:.2e} mean_rel {:.2e} — {}",
            predicted.max_rel,
            predicted.mean_rel,
            if budget.admits(&predicted) { "within budget" } else { "OVER budget, skipped" }
        );
        if !budget.admits(&predicted) {
            continue;
        }
        let mut best = (f64::INFINITY, 0usize);
        for v in 0..SM90_FP8_FLASH_SHAPES.len() {
            let prog = g.to_cuda_variant(&outs, Some(v));
            if !prog.kernels.iter().any(|k| k.source.contains("flash attention on wgmma (FP8)")) {
                continue;
            }
            let p = Program::new(&rt, prog, &[&q, &k, &vt]);
            p.run(&rt);
            let got = p.read(&rt, outs[0]);
            let gh: Vec<Vec<f64>> = heads
                .iter()
                .map(|&h| got[h * per..(h + 1) * per].iter().map(|&x| x as f64).collect())
                .collect();
            let e64 = ErrorStats::compare(&gh, &exact);
            let eemu = ErrorStats::compare(&gh, &emu);
            rt.sync();
            let t = (0..5).map(|_| p.time(&rt, 20)).fold(f64::INFINITY, f64::min);
            let fs = SM90_FP8_FLASH_SHAPES[v];
            println!(
                "  fp8 flash {}x{} s{}: {:.3} ms {:.1} TFLOP/s  vs f64 max {:.2e} mean {:.2e}  vs emu max {:.2e} mean {:.2e}",
                64 * fs.sm90.unwrap().consumers,
                fs.bj,
                fs.stages,
                t,
                flop / t / 1e9,
                e64.max_rel,
                e64.mean_rel,
                eemu.max_rel,
                eemu.mean_rel
            );
            if t < best.0 {
                best = (t, v);
            }
        }
        if best.0.is_finite() {
            println!(
                "tang fp8 attn fwd bh={bh} n={n} d={d} causal={causal}: {:.3} ms {:.1} TFLOP/s (variant {})",
                best.0,
                flop / best.0 / 1e9,
                best.1
            );
        }
    }
}
