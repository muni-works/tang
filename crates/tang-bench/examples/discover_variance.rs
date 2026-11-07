//! GPU timing of the discovered one-pass softmax-weighted variance
//! `Var_p[v]` (`p = softmax(s)` per row, `s, v [rows, n]`) against the
//! two-pass definition written with stock reductions (max, sums, maps),
//! forward and (`BWD=1`) backward of `Σ Var`. Every graph gets
//! `schedule_auto`; both backwards are derived by `TensorGraph::backward`.
use std::time::Instant;
use tang_bench::ir_gpu::{program_for, Executable};
use tang_compile::tensor::discover::{lit, online_lift};
use tang_compile::tensor::discover_catalog::online_catalog;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_gpu::{GpuDevice, KernelCache};

/// Two-pass with stock ops: m, Z, μ, then Σ p(v−μ)²/Z.
fn two_pass(rows: usize, n: usize) -> (TensorGraph, TensorId, [TensorId; 2]) {
    use tang::Scalar;
    let mut g = TensorGraph::new();
    let s = g.input(&[rows, n]);
    let v = g.input(&[rows, n]);
    let m = g.reduce(s, 1, Monoid::max()).unwrap();
    let mb = g.broadcast(m, &[rows, n]).unwrap();
    let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
    let z = g.reduce(p, 1, Monoid::sum()).unwrap();
    let pv = g.map(&[p, v], |x| x[0] * x[1]).unwrap();
    let spv = g.reduce(pv, 1, Monoid::sum()).unwrap();
    let mu = g.map(&[spv, z], |x| x[0] / x[1]).unwrap();
    let mub = g.broadcast(mu, &[rows, n]).unwrap();
    let d = g
        .map(&[p, v, mub], |x| x[0] * (x[1] - x[2]) * (x[1] - x[2]))
        .unwrap();
    let sd = g.reduce(d, 1, Monoid::sum()).unwrap();
    let var = g.map(&[sd, z], |x| x[0] / x[1]).unwrap();
    (g, var, [s, v])
}

/// The discovered monoid, then Var = S₀/Z.
fn one_pass(rows: usize, n: usize) -> (TensorGraph, TensorId, [TensorId; 2]) {
    let case = online_catalog()
        .into_iter()
        .find(|c| c.spec.name.starts_with("softmax-weighted variance"))
        .unwrap();
    let lift = online_lift(&case.spec).unwrap_or_else(|e| panic!("{e:?}"));
    let mut g = TensorGraph::new();
    let s = g.input(&[rows, n]);
    let v = g.input(&[rows, n]);
    let pick = View::pick(2, &[Some(0), Some(1)]);
    let el = lift.element.clone();
    let st = g
        .comp_tuple(
            &[rows, lift.monoid.width],
            &[n],
            vec![
                Arg {
                    src: s,
                    view: pick.clone(),
                },
                Arg { src: v, view: pick },
            ],
            Some(lift.monoid.clone()),
            move |x| el(x),
        )
        .unwrap();
    let slot = |j: i64| View {
        forms: vec![Form::unit(1, 0), Form::constant(1, j)],
        guard: false,
        gather: None,
    };
    // state (m, Z, Σpv, S₀, S₁, S₂); readout row 0 of the lift is S₀
    let ro = lift.readout[0].clone();
    let var = g
        .comp(
            &[rows],
            &[] as &[usize],
            (0..6)
                .map(|j| Arg {
                    src: st,
                    view: slot(j),
                })
                .collect(),
            None,
            move |x| {
                let mut acc = lit(0.0);
                for (b, &sj) in ro.iter().zip(&x[3..]) {
                    if *b != 0.0 {
                        acc = acc + lit(*b) * sj;
                    }
                }
                vec![acc / x[1]]
            },
        )
        .unwrap();
    (g, var, [s, v])
}

fn main() {
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let bwd = std::env::var("BWD").is_ok();
    println!(
        "{} — {}",
        device.name,
        if bwd { "backward (ds, dv)" } else { "forward" }
    );
    for (rows, n) in [(4096usize, 4096usize), (1024, 16384), (16384, 1024)] {
        let mut rng = tang_bench::Rng::new(3);
        let s: Vec<f32> = (0..rows * n).map(|_| 6.0 * (rng.f32() - 0.5)).collect();
        let v: Vec<f32> = (0..rows * n).map(|_| 4.0 + rng.f32()).collect();
        let mut res = vec![];
        for build in [two_pass, one_pass] {
            let (mut g, y, xs) = build(rows, n);
            let outs: Vec<TensorId> = if bwd {
                g.backward(y, &xs).unwrap()
            } else {
                vec![y]
            };
            let (mut g, outs) = g.eliminate_dead(&outs);
            g.schedule_auto(&outs);
            let exe = Executable::new(&device, program_for(&device, &g, &outs), &[&s, &v]);
            for _ in 0..3 {
                exe.run(&device, &mut cache);
            }
            cache.flush(&device);
            let _ = exe.read(&device, outs[0]);
            let t = Instant::now();
            for _ in 0..30 {
                exe.run(&device, &mut cache);
            }
            cache.flush(&device);
            let vals: Vec<Vec<f32>> = outs.iter().map(|&o| exe.read(&device, o)).collect();
            res.push((t.elapsed().as_secs_f64() * 1e3 / 30.0, vals));
        }
        // error relative to each output's max magnitude
        let mut err = 0f32;
        for (a, b) in res[0].1.iter().zip(&res[1].1) {
            let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
            for (x, y) in a.iter().zip(b) {
                err = err.max((x - y).abs() / scale);
            }
        }
        println!(
            "rows {rows} n {n}: two-pass {:.3} ms, one-pass {:.3} ms ({:.2}x), max diff {err:.1e} of max|out|",
            res[0].0,
            res[1].0,
            res[0].0 / res[1].0
        );
    }
}
