//! A top-k MoE FFN layer as one tensor-IR graph, the expert weights read
//! through gather views: router GEMM → top-k (k rounds of argmax) →
//! softmax over the selected logits → SwiGLU expert GEMMs reading
//! `W[sel[t, k]]` → weighted combine. Token-major (decode-style): each
//! (token, slot) addresses its expert's matrices by data, no host routing.
//!
//! Checked against the f64 interpreter (small config) and a plain-Rust
//! reference, then timed per kernel on the selected GPU runtime (CUDA with
//! `--features cuda` on NVIDIA, else wgpu/Metal). The baseline is the same
//! expert GEMMs over weights the host gathered contiguous (`[T, K, …]`,
//! affine views): same FLOPs and bytes, so the difference is the cost of
//! the data-dependent address.
//!
//! `cargo run --release -p tang-bench --example moe_gather [--features cuda]`
//! `MOE_DIMS=T,D,E,K,F` overrides the timed config (default 8,2048,128,10,512:
//! Qwen3-Next-80B-A3B's hidden / expert-ffn / top-k with 128 of its 512 experts).

use tang::Scalar;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;
use tang_gpu::program::{Program, Runtime};

#[derive(Clone, Copy, Debug)]
struct Dims {
    t: usize,
    d: usize,
    e: usize,
    k: usize,
    f: usize,
}

fn eq(a: ExprId, b: ExprId) -> ExprId {
    let (up, down) = (a - b, b - a);
    tang_expr::with_graph(|g| {
        let inner = g.select(down, ExprId::ZERO, ExprId::ONE);
        g.select(up, ExprId::ZERO, inner)
    })
}

/// Slot `j` of a `[T, w]` tuple output, read over a space whose coord 0 is `t`.
fn slot(rank: usize, j: i64) -> View {
    View {
        forms: vec![Form::unit(rank, 0), Form::constant(rank, j)],
        guard: false,
        gather: None,
    }
}

struct Moe {
    g: TensorGraph,
    out: TensorId,
    sel: TensorId,
}

/// Inputs: x[T,D], wr[D,E], w1[E,F,D] (gate), w3[E,F,D] (up), w2[E,D,F] (down).
fn moe_graph(m: Dims) -> Moe {
    let Dims { t, d, e, k, f } = m;
    let mut g = TensorGraph::new();
    let x = g.input(&[t, d]);
    let wr = g.input(&[d, e]);
    let w1 = g.input(&[e, f, d]);
    let w3 = g.input(&[e, f, d]);
    let w2 = g.input(&[e, d, f]);
    let logits = g.matmul(x, wr).unwrap();
    let iota_e = g.iota(&[t, e], 1);
    // top-k: round j takes the argmax of the logits with rounds < j masked
    let mut picks: Vec<TensorId> = Vec::new(); // each [T, 2] = (index, logit)
    for _ in 0..k {
        let mut args = vec![
            Arg {
                src: logits,
                view: View::identity(2, 2),
            },
            Arg {
                src: iota_e,
                view: View::identity(2, 2),
            },
        ];
        args.extend(picks.iter().map(|&p| Arg {
            src: p,
            view: slot(2, 0),
        }));
        let n_prev = picks.len();
        let masked = g
            .comp(&[t, e], &[], args, None, |v| {
                let mut hit = ExprId::ZERO;
                for j in 0..n_prev {
                    hit = hit + eq(v[1], v[2 + j]);
                }
                vec![v[0] - hit * ExprId::from_f64(1e9)]
            })
            .unwrap();
        let am = g
            .comp_tuple(
                &[t, 2],
                &[e],
                vec![
                    Arg {
                        src: iota_e,
                        view: View::identity(2, 2),
                    },
                    Arg {
                        src: masked,
                        view: View::identity(2, 2),
                    },
                ],
                Some(Monoid::argmax()),
                |v| vec![v[0], v[1]],
            )
            .unwrap();
        picks.push(am);
    }
    // sel[t, k] and the renormalized softmax weights wt[t, k]
    let iota_k = g.iota(&[t, k], 1);
    let stack = |g: &mut TensorGraph, s: i64, weights: bool| {
        let mut args = vec![Arg {
            src: iota_k,
            view: View::identity(2, 2),
        }];
        args.extend(picks.iter().map(|&p| Arg {
            src: p,
            view: slot(2, s),
        }));
        g.comp(&[t, k], &[], args, None, |v| {
            let mut acc = ExprId::ZERO;
            let mut norm = ExprId::ZERO;
            for j in 0..k {
                let val = if weights {
                    let ex = (v[1 + j] - v[1]).exp(); // v[1] is the max
                    norm = norm + ex;
                    ex
                } else {
                    v[1 + j]
                };
                acc = acc + eq(v[0], ExprId::from_f64(j as f64)) * val;
            }
            vec![if weights { acc / norm } else { acc }]
        })
        .unwrap()
    };
    let sel = stack(&mut g, 0, false);
    let wt = stack(&mut g, 1, true);
    let (act, y) = expert_ffn(&mut g, m, x, sel, [w1, w3, w2], true);
    let _ = act;
    let out = combine(&mut g, m, wt, y);
    Moe { g, out, sel }
}

/// `gate/up[t,k,f] = Σ_d x[t,d]·W[sel[t,k],f,d]`, SwiGLU, then
/// `y[t,k,d] = Σ_f act[t,k,f]·W2[sel[t,k],d,f]`. With `gathered = false`
/// the weights are `[T,K,…]` inputs read affinely and `sel` is unused.
fn expert_ffn(
    g: &mut TensorGraph,
    m: Dims,
    x: TensorId,
    sel: TensorId,
    [w1, w3, w2]: [TensorId; 3],
    gathered: bool,
) -> (TensorId, TensorId) {
    let Dims { t, d, k, f, .. } = m;
    // weight view over a [t, k, i, j] space reading W[·, i, j]
    let wview = || {
        if gathered {
            View::pick(4, &[None, Some(2), Some(3)]).gather(0, 2, 1)
        } else {
            View::identity(4, 4)
        }
    };
    let sel_arg = Arg {
        src: sel,
        view: View::pick(4, &[Some(0), Some(1)]),
    };
    let proj = |g: &mut TensorGraph, w: TensorId| {
        let mut args = vec![
            Arg {
                src: x,
                view: View::pick(4, &[Some(0), Some(3)]),
            },
            Arg {
                src: w,
                view: wview(),
            },
        ];
        if gathered {
            args.push(sel_arg.clone());
        }
        g.comp(&[t, k, f], &[d], args, Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
        .unwrap()
    };
    let gate = proj(g, w1);
    let up = proj(g, w3);
    let act = g
        .map(&[gate, up], |v| {
            let sig = ExprId::ONE / (ExprId::ONE + (-v[0]).exp());
            v[0] * sig * v[1]
        })
        .unwrap();
    let mut args = vec![
        Arg {
            src: act,
            view: View::pick(4, &[Some(0), Some(1), Some(3)]),
        },
        Arg {
            src: w2,
            view: wview(),
        },
    ];
    if gathered {
        args.push(sel_arg);
    }
    let y = g
        .comp(&[t, k, d], &[f], args, Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
        .unwrap();
    (act, y)
}

/// `out[t,d] = Σ_k wt[t,k]·y[t,k,d]`.
fn combine(g: &mut TensorGraph, m: Dims, wt: TensorId, y: TensorId) -> TensorId {
    let args = vec![
        Arg {
            src: wt,
            view: View::pick(3, &[Some(0), Some(2)]),
        },
        Arg {
            src: y,
            view: View::pick(3, &[Some(0), Some(2), Some(1)]),
        },
    ];
    g.comp(&[m.t, m.d], &[m.k], args, Some(Monoid::sum()), |v| {
        vec![v[0] * v[1]]
    })
    .unwrap()
}

/// Expert half only, weights pre-gathered by the host: inputs
/// x[T,D], wt[T,K], w1g[T,K,F,D], w3g[T,K,F,D], w2g[T,K,D,F].
fn baseline_graph(m: Dims) -> (TensorGraph, TensorId) {
    let Dims { t, d, k, f, .. } = m;
    let mut g = TensorGraph::new();
    let x = g.input(&[t, d]);
    let wt = g.input(&[t, k]);
    let w1 = g.input(&[t, k, f, d]);
    let w3 = g.input(&[t, k, f, d]);
    let w2 = g.input(&[t, k, d, f]);
    let (_, y) = expert_ffn(&mut g, m, x, x, [w1, w3, w2], false);
    let out = combine(&mut g, m, wt, y);
    (g, out)
}

/// Plain-Rust f64 reference: (out[T,D], sel[T,K]).
fn reference(m: Dims, w: &[Vec<f32>]) -> (Vec<f64>, Vec<usize>) {
    let Dims { t, d, e, k, f } = m;
    let (x, wr, w1, w3, w2) = (&w[0], &w[1], &w[2], &w[3], &w[4]);
    let mut out = vec![0.0; t * d];
    let mut sels = Vec::new();
    for ti in 0..t {
        let logits: Vec<f64> = (0..e)
            .map(|ei| {
                (0..d)
                    .map(|di| x[ti * d + di] as f64 * wr[di * e + ei] as f64)
                    .sum()
            })
            .collect();
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let top = &order[..k];
        let ex: Vec<f64> = top
            .iter()
            .map(|&i| (logits[i] - logits[top[0]]).exp())
            .collect();
        let norm: f64 = ex.iter().sum();
        for (j, &ei) in top.iter().enumerate() {
            sels.push(ei);
            let mut act = vec![0.0; f];
            for (fi, a) in act.iter_mut().enumerate() {
                let (mut gt, mut up) = (0.0, 0.0);
                for di in 0..d {
                    let xv = x[ti * d + di] as f64;
                    gt += xv * w1[(ei * f + fi) * d + di] as f64;
                    up += xv * w3[(ei * f + fi) * d + di] as f64;
                }
                *a = gt / (1.0 + (-gt).exp()) * up;
            }
            for di in 0..d {
                let yv: f64 = (0..f)
                    .map(|fi| act[fi] * w2[(ei * d + di) * f + fi] as f64)
                    .sum();
                out[ti * d + di] += ex[j] / norm * yv;
            }
        }
    }
    (out, sels)
}

fn max_rel(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - w).abs() / scale)
        .fold(0.0, f64::max)
}

fn weights(m: Dims, rng: &mut tang_bench::Rng) -> Vec<Vec<f32>> {
    let Dims { t, d, e, f, .. } = m;
    let mut rand =
        |n: usize, s: f32| -> Vec<f32> { (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * s).collect() };
    let sd = 1.0 / (d as f32).sqrt();
    let sf = 1.0 / (f as f32).sqrt();
    vec![
        rand(t * d, 1.0),
        rand(d * e, sd),
        rand(e * f * d, sd),
        rand(e * f * d, sd),
        rand(e * d * f, sf),
    ]
}

fn optimized(g: &TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (mut h, o) = g.substitute_maps(outs);
    h.schedule_auto(&o);
    (h, o)
}

fn main() {
    let rt = Runtime::select().expect("gpu");
    println!("backend: {:?} — {}", rt.kind(), rt.name());
    let mut rng = tang_bench::Rng::new(11);

    // 1. small config: IR graph (f64 interpreter, before and after the
    //    passes) and GPU against the plain-Rust reference
    let small = Dims {
        t: 4,
        d: 64,
        e: 16,
        k: 4,
        f: 32,
    };
    let w = weights(small, &mut rng);
    let moe = moe_graph(small);
    let (want, want_sel) = reference(small, &w);
    let w64: Vec<Vec<f64>> = w
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();
    let ev = moe.g.eval(&w64, &[moe.out, moe.sel]);
    let sel_ok = ev[1]
        .iter()
        .map(|&s| s as usize)
        .eq(want_sel.iter().copied());
    let err_ir: f64 = ev[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!(
        "small {small:?}: interpreter vs reference max |Δ| {err_ir:.2e}, top-k match {sel_ok}"
    );
    assert!(sel_ok && err_ir < 1e-9);
    let (h, outs) = optimized(&moe.g, &[moe.out, moe.sel]);
    let evh = h.eval(&w64, &outs);
    let err_pass: f64 = evh[0]
        .iter()
        .zip(&ev[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!(
        "  after substitute_maps + schedule_auto: {} comps -> {} kernels, max |Δ| {err_pass:.2e}",
        moe.g.count_comps(&[moe.out, moe.sel]),
        h.materialized(&outs).len()
    );
    assert!(err_pass < 1e-9);
    let refs: Vec<&[f32]> = w.iter().map(|v| v.as_slice()).collect();
    let exe = Program::compile(&rt, &h, &outs, &refs);
    exe.run(&rt);
    let got = exe.read(&rt, outs[0]);
    let gsel = exe.read(&rt, outs[1]);
    let gpu_sel_ok = gsel
        .iter()
        .map(|&s| s as usize)
        .eq(want_sel.iter().copied());
    let e_gpu = max_rel(&got, &want);
    println!("  gpu vs reference: max rel err {e_gpu:.2e}, top-k match {gpu_sel_ok}");
    // CUDA runs the router GEMM as a TF32 `mma.sync` tile (10-bit mantissa)
    let tol = match rt.kind() {
        tang_gpu::program::BackendKind::Cuda => 2e-3,
        _ => 1e-4,
    };
    assert!(gpu_sel_ok && e_gpu < tol);

    // 2. timed config
    let m = match std::env::var("MOE_DIMS") {
        Ok(s) => {
            let v: Vec<usize> = s.split(',').map(|x| x.parse().unwrap()).collect();
            Dims {
                t: v[0],
                d: v[1],
                e: v[2],
                k: v[3],
                f: v[4],
            }
        }
        Err(_) => Dims {
            t: 8,
            d: 2048,
            e: 128,
            k: 10,
            f: 512,
        },
    };
    let w = weights(m, &mut rng);
    let moe = moe_graph(m);
    let (want, want_sel) = reference(m, &w);
    let (h, outs) = optimized(&moe.g, &[moe.out, moe.sel]);
    let refs: Vec<&[f32]> = w.iter().map(|v| v.as_slice()).collect();
    let exe = Program::compile(&rt, &h, &outs, &refs);
    exe.run(&rt);
    let got = exe.read(&rt, outs[0]);
    let gsel = exe.read(&rt, outs[1]);
    let sel_ok = gsel
        .iter()
        .map(|&s| s as usize)
        .eq(want_sel.iter().copied());
    println!(
        "\ntimed {m:?}: gpu vs reference max rel err {:.2e}, top-k match {sel_ok}",
        max_rel(&got, &want)
    );
    let iters = 50;
    let total = exe.time(&rt, iters);
    println!(
        "  whole layer: {total:.3} ms ({} kernels)",
        exe.program.kernels.len()
    );
    let describe = |h: &TensorGraph, node: TensorId| -> String {
        let c = h.comp_at(node.index()).unwrap();
        let sched = &h.nodes()[node.index()];
        format!(
            "out {:?} red {:?}{}{}",
            c.out.iter().map(|d| d.n()).collect::<Vec<_>>(),
            c.red.iter().map(|d| d.n()).collect::<Vec<_>>(),
            if c.has_gather() { " gather" } else { "" },
            if sched.split > 0 {
                format!(" split {}", sched.split)
            } else if sched.tile.0 > 0 {
                format!(" tile {:?} reg {:?}", sched.tile, sched.reg)
            } else {
                String::new()
            }
        )
    };
    // the expert half: every kernel from the first gathered one on
    // (kernels are in dependency order; router and top-k come first)
    let first_expert = exe
        .program
        .kernels
        .iter()
        .position(|k| h.comp_at(k.node.index()).unwrap().has_gather())
        .expect("a gathered kernel");
    let mut expert_ms = 0.0;
    for (i, kern) in exe.program.kernels.iter().enumerate() {
        let ms = exe.time_kernel(&rt, i, iters);
        if i >= first_expert {
            expert_ms += ms;
        }
        println!("    k{i}: {ms:.3} ms  {}", describe(&h, kern.node));
    }
    println!("  expert GEMMs + combine (gathered): {expert_ms:.3} ms");

    // baseline: same expert math over host-gathered contiguous weights
    let Dims { t, d, k, f, e: _ } = m;
    let (bg, bout) = baseline_graph(m);
    let (bh, bouts) = optimized(&bg, &[bout]);
    let mut wt = vec![0f32; t * k];
    let mut w1g = Vec::with_capacity(t * k * f * d);
    let mut w3g = Vec::with_capacity(t * k * f * d);
    let mut w2g = Vec::with_capacity(t * k * d * f);
    {
        // weights from the reference's selection; wt from the gpu run of
        // the full layer is not exposed, so recompute it here
        let (x, wr) = (&w[0], &w[1]);
        for ti in 0..t {
            let logit = |ei: usize| -> f64 {
                (0..d)
                    .map(|di| x[ti * d + di] as f64 * wr[di * m.e + ei] as f64)
                    .sum()
            };
            let top = &want_sel[ti * k..(ti + 1) * k];
            let ex: Vec<f64> = top
                .iter()
                .map(|&i| (logit(i) - logit(top[0])).exp())
                .collect();
            let norm: f64 = ex.iter().sum();
            for (j, &ei) in top.iter().enumerate() {
                wt[ti * k + j] = (ex[j] / norm) as f32;
                w1g.extend_from_slice(&w[2][ei * f * d..(ei + 1) * f * d]);
                w3g.extend_from_slice(&w[3][ei * f * d..(ei + 1) * f * d]);
                w2g.extend_from_slice(&w[4][ei * d * f..(ei + 1) * d * f]);
            }
        }
    }
    let bexe = Program::compile(&rt, &bh, &bouts, &[&w[0], &wt, &w1g, &w3g, &w2g]);
    bexe.run(&rt);
    println!(
        "  baseline (pre-gathered, affine) vs reference: max rel err {:.2e}",
        max_rel(&bexe.read(&rt, bouts[0]), &want)
    );
    let mut base_ms = 0.0;
    for (i, kern) in bexe.program.kernels.iter().enumerate() {
        let ms = bexe.time_kernel(&rt, i, iters);
        base_ms += ms;
        println!("    b{i}: {ms:.3} ms  {}", describe(&bh, kern.node));
    }
    println!("  expert GEMMs + combine (pre-gathered): {base_ms:.3} ms");
    // and with the gathered layer's schedule (one thread per point): the
    // like-for-like cost of the data-dependent address
    let mut flat = bh.clone();
    for i in 0..flat.nodes().len() {
        flat.set_schedule(TensorId::from_index(i), 0, (0, 0, 0), (1, 1));
    }
    let fexe = Program::compile(&rt, &flat, &bouts, &[&w[0], &wt, &w1g, &w3g, &w2g]);
    let flat_ms: f64 = (0..fexe.program.kernels.len())
        .map(|i| fexe.time_kernel(&rt, i, iters))
        .sum();
    println!("  expert GEMMs + combine (pre-gathered, per-point schedule): {flat_ms:.3} ms");
    // bytes the expert GEMMs must read at least once: 3 matrices per (t, k)
    let bytes = (t * k * 3 * f * d * 4) as f64;
    println!(
        "  weight traffic {:.1} MB: gathered {:.0} GB/s, pre-gathered {:.0} GB/s (tiled), {:.0} GB/s (per point)",
        bytes / 1e6,
        bytes / expert_ms / 1e6,
        bytes / base_ms / 1e6,
        bytes / flat_ms / 1e6
    );
}
