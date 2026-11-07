//! The fused attention backward (`tensor_cuda_attn_bwd`) against the
//! unfused program (`TANG_NO_ATTN_BWD=1`) on random inputs: dQ, dK, dV
//! agreement and both programs' times.
//!
//!     cargo run --release -p tang-bench --features cuda --example attn_bwd_check
use tang_bench::ir_gpu::naive_attention_heads;
use tang_compile::tensor::{Format, Monoid, Precision, TensorGraph, TensorId};
use tang_gpu::program::{Program, Runtime};

fn graph(
    bh: usize,
    n: usize,
    d: usize,
    causal: bool,
) -> (TensorGraph, Vec<TensorId>, Vec<TensorId>) {
    let (mut g, y) = naive_attention_heads(bh, 1, n, d, d, causal);
    let mut ins: Vec<TensorId> = (0..3).map(TensorId::from_index).collect();
    let dy = g.input(&[bh, 1, n, d]);
    let prod = g.map(&[y, dy], |x| x[0] * x[1]).unwrap();
    let flat = g.reshape(prod, &[bh * n * d]).unwrap();
    let loss = g.reduce(flat, 0, Monoid::sum()).unwrap();
    let grads = g.backward(loss, &ins).unwrap();
    let (g, outs) = g.stabilize_softmax(&grads);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    ins.push(dy);
    for &t in ins.iter().chain(&outs) {
        g.set_precision(t, Precision::storage(Format::BF16));
    }
    (g, ins, outs)
}

fn main() {
    let rt = Runtime::cuda().expect("cuda");
    let cases: Vec<(usize, usize, usize, bool)> = match std::env::var("CASES") {
        Ok(v) => v
            .split(',')
            .map(|c| {
                let x: Vec<usize> = c.split('x').map(|t| t.parse().unwrap()).collect();
                (x[0], x[1], x[2], x[3] != 0)
            })
            .collect(),
        Err(_) => vec![
            (8, 256, 64, true),
            (8, 256, 64, false),
            (4, 512, 128, true),
            (4, 256, 128, false),
        ],
    };
    let variant: Option<usize> = std::env::var("V")
        .ok()
        .and_then(|v| v.parse().ok())
        .or(Some(0));
    let mut rng = tang_bench::Rng::new(3);
    for (bh, n, d, causal) in cases {
        let (g, _ins, outs) = graph(bh, n, d, causal);
        let len = bh * n * d;
        let inputs: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..len).map(|_| rng.f32() * 2.0 - 1.0).collect())
            .collect();
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut res = Vec::new();
        for fused in [true, false] {
            std::env::set_var("TANG_NO_ATTN_BWD", if fused { "0" } else { "1" });
            let sm: u32 = std::env::var("SM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(match &rt {
                    Runtime::Cuda(d) => d
                        .arch
                        .trim_start_matches("sm_")
                        .trim_end_matches('a')
                        .parse()
                        .unwrap_or(0),
                    #[allow(unreachable_patterns)]
                    _ => 0,
                });
            let prog =
                tang_compile::tensor::cuda::with_cuda_sm(sm, || g.to_cuda_variant(&outs, variant));
            let fk = prog
                .kernels
                .iter()
                .any(|k| k.source.contains("attention backward"));
            if std::env::var_os("DUMP").is_some() && fused {
                for k in &prog.kernels {
                    if k.source.contains("attention backward") {
                        eprintln!("{}", k.source);
                    }
                }
            }
            let p = Program::new(&rt, prog, &refs);
            p.run(&rt);
            rt.sync();
            let got: Vec<Vec<f32>> = outs.iter().map(|&t| p.read(&rt, t)).collect();
            if std::env::var_os("BREAK").is_some() {
                for (i, k) in p.program.kernels.iter().enumerate() {
                    let head = k.source.lines().next().unwrap_or("");
                    eprintln!(
                        "  [{}] kernel {i}: {:.4} ms  {head}",
                        if fused { "fused" } else { "unfused" },
                        p.time_kernel(&rt, i, 10)
                    );
                }
            }
            let ms = (0..3)
                .map(|_| p.time(&rt, 10))
                .fold(f64::INFINITY, f64::min);
            res.push((fk, got, ms, p.program.kernels.len()));
        }
        let names = ["dq", "dk", "dv"];
        let mut line = format!("bh={bh} n={n} d={d} causal={causal}: fused={} ({} kernels) {:.3} ms vs unfused {:.3} ms ({} kernels);", res[0].0, res[0].3, res[0].2, res[1].2, res[1].3);
        for (w, nm) in names.iter().enumerate() {
            let (a, b) = (&res[0].1[w], &res[1].1[w]);
            let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let err = a
                .iter()
                .zip(b)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            line.push_str(&format!(" {nm} rel {:.2e}", err / scale));
        }
        println!("{line}");
    }
}
