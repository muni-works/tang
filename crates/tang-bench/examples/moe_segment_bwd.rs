//! The expert-weight gradient of a gathered MoE contraction on the GPU:
//! `y[p, r] = Σ_f W[key[p], r, f] · h[p, f]` with the pairs sorted by
//! expert, `dW[e, r, f] = Σ_p [key[p] == e] · dy[p, r] · h[p, f]`. With
//! `key` marked sorted the backward reduces over each expert's run of
//! pairs (binary-searched) instead of all `p`; `TANG_NO_SEGMENT=1` runs the
//! dense gated sum for comparison. Checks the GPU against a plain-Rust dW.
//!
//! `cargo run --release -p tang-bench --example moe_segment_bwd`
//! (`MOE_SEG=e,p,r,f`, default 64,640,256,128: 64 tokens × top-10).

use std::time::Instant;
use tang_compile::tensor::{Arg, Monoid, TensorGraph, View};
use tang_gpu::program::{Program, Runtime};

fn main() {
    let d: Vec<usize> = std::env::var("MOE_SEG")
        .unwrap_or_else(|_| "64,640,256,128".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let (e, p, r, f) = (d[0], d[1], d[2], d[3]);
    let mut g = TensorGraph::new();
    let w = g.input(&[e, r, f]);
    let h = g.input(&[p, f]);
    let key = g.input(&[p]);
    let dy = g.input(&[p, r]);
    g.set_sorted(key, std::env::var("MOE_SEG_UNSORTED").is_err());
    let a = |src, view| Arg { src, view };
    let y = g
        .comp(
            &[p, r],
            &[f],
            vec![
                a(w, View::pick(3, &[None, Some(1), Some(2)]).gather(0, 2, 1)),
                a(h, View::pick(3, &[Some(0), Some(2)])),
                a(key, View::pick(3, &[Some(0)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let prod = g.map(&[y, dy], |x| x[0] * x[1]).unwrap();
    let r1 = g.reduce(prod, 1, Monoid::sum()).unwrap();
    let loss = g.reduce(r1, 0, Monoid::sum()).unwrap();
    let dw = g.backward(loss, &[w]).unwrap()[0];
    println!("segment: {:?}", g.segment(dw));

    // pairs sorted by a skewed routing (some experts empty)
    let mut keys: Vec<f32> = (0..p).map(|q| ((q * q * 7 + q * 3) % e) as f32).collect();
    keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fill = |n: usize, s: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 2654435761 + s) % 1000) as f32 / 500.0) - 1.0)
            .collect()
    };
    let (wv, hv, dyv) = (fill(e * r * f, 1), fill(p * f, 2), fill(p * r, 3));
    let mut want = vec![0.0f64; e * r * f];
    for (q, &k) in keys.iter().enumerate() {
        for i in 0..r {
            for j in 0..f {
                want[(k as usize * r + i) * f + j] += dyv[q * r + i] as f64 * hv[q * f + j] as f64;
            }
        }
    }
    let rt = Runtime::select().expect("gpu");
    println!(
        "backend: {:?} — {}; E {e} P {p} R {r} F {f}",
        rt.kind(),
        rt.name()
    );
    let exe = Program::compile(&rt, &g, &[dw], &[&wv, &hv, &keys, &dyv]);
    exe.run(&rt);
    let got = exe.read(&rt, dw);
    let err = got
        .iter()
        .zip(&want)
        .map(|(&a, &b)| (a as f64 - b).abs() / (1.0 + b.abs()))
        .fold(0.0, f64::max);
    let iters = 20;
    let t = Instant::now();
    for _ in 0..iters {
        exe.run(&rt);
    }
    let _ = exe.read(&rt, dw);
    println!(
        "dW max rel err {err:.2e}; {:.3} ms a run ({} kernels)",
        t.elapsed().as_secs_f64() * 1e3 / iters as f64,
        exe.program.kernels.len()
    );
}
