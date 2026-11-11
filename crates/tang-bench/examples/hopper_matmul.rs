//! bf16 matmul tiles on Hopper: every `mma.sync` shape (`MMA_SHAPES`)
//! against every `wgmma` shape (`SM90_MMA_SHAPES`), bf16 operands and
//! output, fp32 accumulation. Both B layouts: `[k, n]` (an activation)
//! and `[n, k]` (a weight, `x · Wᵀ`). Fastest single launch per family
//! (`Program::time_kernel`, best of 20), checked against an f64 product
//! on sampled entries. `scripts/hopper_attention.py matmul` times cuBLAS
//! (torch bf16 matmul) on the same shapes.
//!
//!     cargo run --release -p tang-bench --features cuda --example hopper_matmul
use tang_compile::tensor::cuda::{MMA_SHAPES, SM90_MMA_SHAPES};
use tang_compile::tensor::{Format, Precision, TensorGraph};
use tang_gpu::program::{Program, Runtime};

fn main() {
    // MM_DUMP=v: print variant v's kernel for the first case (no GPU)
    let dump: Option<usize> = std::env::var("MM_DUMP").ok().and_then(|v| v.parse().ok());
    let rt = if dump.is_none() {
        Some(Runtime::select().expect("gpu"))
    } else {
        None
    };
    let cases: Vec<(usize, usize, usize)> = match std::env::var("MM_CASES") {
        Ok(v) => v
            .split(',')
            .map(|c| {
                let d: Vec<usize> = c.split('x').map(|x| x.parse().unwrap()).collect();
                (d[0], d[1], d[2])
            })
            .collect(),
        Err(_) => vec![
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 2048, 2048),
            (16384, 2048, 8192),
            (16384, 8192, 2048),
        ],
    };
    let mut rng = tang_bench::Rng::new(5);
    for (m, k, n) in cases {
        for weight in [false, true] {
            let mut g = TensorGraph::new();
            let a = g.input(&[m, k]);
            let bshape = if weight { [n, k] } else { [k, n] };
            let b = g.input(&bshape);
            let bt = if weight {
                g.transpose(b, &[1, 0]).unwrap()
            } else {
                b
            };
            let c = g.matmul(a, bt).unwrap();
            let (mut g, outs) = g.substitute_maps(&[c]);
            g.schedule_auto(&outs);
            let c = outs[0];
            g.set_precision(a, Precision::storage(Format::BF16));
            g.set_precision(b, Precision::storage(Format::BF16));
            g.set_precision(c, Precision::storage(Format::BF16));
            if let Some(v) = dump {
                print!("{}", g.to_cuda_variant(&[c], Some(v)).kernels[0].source);
                return;
            }
            let rt = rt.as_ref().unwrap();
            let av: Vec<f32> = (0..m * k).map(|_| rng.f32() - 0.5).collect();
            let bv: Vec<f32> = (0..k * n).map(|_| rng.f32() - 0.5).collect();
            let bf = |x: f32| {
                f32::from_bits(((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) << 16)
                    as f64
            };
            let flop = 2.0 * (m * k * n) as f64;
            let mut line = format!(
                "tang matmul {m}x{k}x{n} b={}:",
                if weight { "[n,k]" } else { "[k,n]" }
            );
            for (fam, range) in [
                ("mma.sync", 0..MMA_SHAPES.len()),
                (
                    "wgmma",
                    MMA_SHAPES.len()..MMA_SHAPES.len() + SM90_MMA_SHAPES.len(),
                ),
            ] {
                let mut best = (f64::INFINITY, 0usize, 0.0f64);
                for v in range {
                    let p = g.to_cuda_variant(&[c], Some(v));
                    let src = &p.kernels[0].source;
                    let is_wg = src.contains("wgmma) tile");
                    if (fam == "wgmma") != is_wg || !src.contains("mma (") {
                        continue;
                    }
                    let p = Program::new(rt, p, &[&av, &bv]);
                    p.run(rt);
                    let got = p.read(rt, c);
                    // 64 sampled entries against f64
                    let mut err: f64 = 0.0;
                    let mut scale: f64 = 0.0;
                    for s in 0..64usize {
                        let (i, j) = ((s * 7919) % m, (s * 104729) % n);
                        let mut want = 0.0f64;
                        for kk in 0..k {
                            let bb = if weight {
                                bv[j * k + kk]
                            } else {
                                bv[kk * n + j]
                            };
                            want += bf(av[i * k + kk]) * bf(bb);
                        }
                        err = err.max((got[i * n + j] as f64 - want).abs());
                        scale = scale.max(want.abs());
                    }
                    let t = p.time_kernel(rt, 0, 20);
                    if t < best.0 {
                        best = (t, v, err / scale);
                    }
                }
                if best.0.is_finite() {
                    line.push_str(&format!(
                        "  {fam} {:.4} ms {:.1} TFLOP/s (variant {}, rel_err {:.1e})",
                        best.0,
                        flop / best.0 / 1e9,
                        best.1,
                        best.2
                    ));
                } else {
                    line.push_str(&format!("  {fam} n/a"));
                }
            }
            println!("{line}");
        }
    }
}
