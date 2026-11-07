//! A/B for windowed fragment staging: the conv-shaped ops of
//! `kokoro_ops` on f16 fragments, windowed (default) vs implicit-GEMM
//! gather (`TANG_MSL_NO_WINDOW`), interleaved so machine load hits both.
//! `cargo run --release -p tang-bench --features metal --example conv_frag`
#[cfg(feature = "metal")]
fn main() {
    use tang_bench::ir_msl::{Executable, MslDevice};
    use tang_compile::tensor::{Arg, Form, Format, Monoid, Precision, TensorGraph, TensorId, View};
    let dev = MslDevice::new().expect("metal");
    let mut rng = tang_bench::Rng::new(1);
    let f = |rank: usize, terms: &[(usize, i64)], off: i64| {
        let mut x = Form::zero(rank);
        for &(d, c) in terms {
            x.coefs[d] += c;
        }
        x.offset = off;
        x
    };
    let mut cases: Vec<(String, TensorGraph, TensorId, f64)> = Vec::new();
    // a: conv1d 512→512 k7 d3 T2000
    {
        let (c, t, k, d) = (512usize, 2000usize, 7usize, 3i64);
        let mut g = TensorGraph::new();
        let x = g.input(&[c, t]);
        let w = g.input(&[c, c, k]);
        let xv = View {
            forms: vec![f(4, &[(2, 1)], 0), f(4, &[(1, 1), (3, d)], -d * 3)],
            guard: true,
            gather: None,
        };
        let y = g
            .comp(
                &[c, t],
                &[c, k],
                vec![
                    Arg { src: x, view: xv },
                    Arg {
                        src: w,
                        view: View::pick(4, &[Some(0), Some(2), Some(3)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        cases.push(("a conv k7 d3".into(), g, y, 2.0 * (c * c * k * t) as f64));
    }
    // b: polyphase upsample 512→256 s10, T2000
    {
        let (ci, co, t, s) = (512usize, 256usize, 2000usize, 10usize);
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, 2 * s]);
        let xv = View {
            forms: vec![f(5, &[(4, 1)], 0), f(5, &[(1, 1), (3, -1)], 0)],
            guard: true,
            gather: None,
        };
        let wv = View {
            forms: vec![
                f(5, &[(0, 1)], 0),
                f(5, &[(4, 1)], 0),
                f(5, &[(2, 1), (3, s as i64)], 0),
            ],
            guard: false,
            gather: None,
        };
        let y = g
            .comp(
                &[co, t, s],
                &[2, ci],
                vec![Arg { src: x, view: xv }, Arg { src: w, view: wv }],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        cases.push((
            "b upsample s10".into(),
            g,
            y,
            2.0 * (co * ci * 2 * t * s) as f64,
        ));
    }
    for (name, mut g, y, flops) in cases {
        for i in 0..2 {
            g.set_precision(TensorId::from_index(i), Precision::storage(Format::F16));
        }
        g.set_fragment(y, true);
        let inputs: Vec<Vec<f32>> = (0..2)
            .map(|i| {
                let n: usize = g
                    .shape(TensorId::from_index(i))
                    .iter()
                    .map(|d| d.n())
                    .product();
                (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
            })
            .collect();
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut best = [f64::INFINITY; 2];
        let mut outs = [Vec::new(), Vec::new()];
        for rep in 0..4 {
            for (m, b) in best.iter_mut().enumerate() {
                if m == 1 {
                    std::env::set_var("TANG_MSL_NO_WINDOW", "1");
                } else {
                    std::env::remove_var("TANG_MSL_NO_WINDOW");
                }
                let exe = Executable::new(&dev, g.to_msl(&[y]), &refs);
                exe.run(&dev);
                if rep == 0 {
                    outs[m] = exe.read(y);
                }
                *b = b.min(exe.time(&dev, 50));
            }
        }
        std::env::remove_var("TANG_MSL_NO_WINDOW");
        let diff = outs[0]
            .iter()
            .zip(&outs[1])
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "{name:<16} window {:.3} ms ({:.1} TFLOP/s)  gather {:.3} ms  max|Δ| {diff:.1e}",
            best[0],
            flops / best[0] * 1e-9,
            best[1]
        );
    }
    // a fwd+bwd with a given dy: every contraction on fragments, per kernel
    {
        let (c, t, k, d) = (512usize, 2000usize, 7usize, 3i64);
        let mut g = TensorGraph::new();
        let x = g.input(&[c, t]);
        let w = g.input(&[c, c, k]);
        let gy = g.input(&[c, t]);
        let xv = View {
            forms: vec![f(4, &[(2, 1)], 0), f(4, &[(1, 1), (3, d)], -d * 3)],
            guard: true,
            gather: None,
        };
        let y = g
            .comp(
                &[c, t],
                &[c, k],
                vec![
                    Arg { src: x, view: xv },
                    Arg {
                        src: w,
                        view: View::pick(4, &[Some(0), Some(2), Some(3)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let l = g.map(&[y, gy], |v| v[0] * v[1]).unwrap();
        let grads = g.backward(l, &[x, w]).unwrap();
        let (g, outs) = g.substitute_maps(&[y, grads[0], grads[1]]);
        let (mut g, outs) = g.eliminate_common(&outs);
        g.schedule_auto(&outs);
        for t in tang_bench::ir_msl::fragment_candidates(&g, &outs) {
            for o in g.fragment_operands(t).unwrap() {
                g.set_precision(o, Precision::storage(Format::F16));
            }
            g.set_fragment(t, true);
        }
        let inputs: Vec<Vec<f32>> = [c * t, c * c * k, c * t]
            .iter()
            .map(|&n| (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect())
            .collect();
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let exe = Executable::new(&dev, g.to_msl(&outs), &refs);
        let total = (0..4)
            .map(|_| exe.time(&dev, 50))
            .fold(f64::INFINITY, f64::min);
        println!("a fwd+bwd, dy given, all on fragments: {total:.3} ms");
        for (i, kk) in exe.program.kernels.iter().enumerate() {
            let ms = (0..4)
                .map(|_| exe.time_kernels(&dev, i..i + 1, 50))
                .fold(f64::INFINITY, f64::min);
            let what = if outs[0] == kk.node {
                "y"
            } else if outs[1] == kk.node {
                "dx"
            } else if outs[2] == kk.node {
                "dw"
            } else {
                "other"
            };
            println!(
                "    {what:<5} node {:>3} {ms:.3} ms  fragment {} windowed {}",
                kk.node.index(),
                kk.fragment,
                kk.source.contains("windowed operand")
            );
        }
    }
}
#[cfg(not(feature = "metal"))]
fn main() {}
