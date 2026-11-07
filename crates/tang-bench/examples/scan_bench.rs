//! The chunked delta-rule scan on CUDA, checked against the f64
//! interpreter and timed.
//!
//!   scan_bench MODE H T [N M BV]
//!
//! MODE is `gdn` (gated delta rule, Qwen3-Next), `ssd` (Mamba-2: gate
//! and step size, no delta) or `lin` (plain linear attention). The graph
//! is written in tensor IR (`delta_scan` + `scan_readout`), recognized by
//! `chunked_scan_plan` and lowered to two kernels. First a check at
//! H = 1, T = 128 against `TensorGraph::eval` on the same bf16-rounded
//! inputs, then the median time of 20 runs at H × T. One line of output:
//! `mode H T n m bv ms rel_err`.
use tang_compile::tensor::precision::{bf16_bits, bf16_from_bits};
use tang_compile::tensor::{cuda_scan::ChunkedScan, TensorGraph, TensorId};
use tang_gpu::cuda::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use tang_gpu::cuda::CudaDevice;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Gdn,
    Ssd,
    Lin,
}

struct Data {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn bf(x: f64) -> f64 {
    bf16_from_bits(bf16_bits(x))
}

fn data(mode: Mode, h: usize, t: usize, n: usize, m: usize) -> Data {
    let mut r = rng(7);
    let mut gauss = move || {
        let (a, b) = (r().max(1e-12), r());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    };
    let mut rows = |w: usize, norm: bool| {
        let mut x: Vec<f64> = (0..h * t * w).map(|_| gauss()).collect();
        if norm {
            for row in x.chunks_mut(w) {
                let s = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                row.iter_mut().for_each(|v| *v /= s);
            }
        }
        x.into_iter().map(bf).collect::<Vec<_>>()
    };
    let norm = mode != Mode::Ssd;
    let q = rows(n, norm);
    let k = rows(n, norm);
    let v = rows(m, false);
    let mut u = rng(11);
    let (alpha, beta): (Vec<f64>, Vec<f64>) = (0..h * t)
        .map(|_| match mode {
            // α = exp(g), g = −softplus-like in (−0.5, 0); β = sigmoid
            Mode::Gdn => ((-0.5 * u()).exp(), 1.0 / (1.0 + (-(4.0 * u() - 2.0)).exp())),
            // α = exp(Δ·A), A = −1, Δ ∈ (0.001, 0.1); β = Δ
            Mode::Ssd => {
                let dt = 0.001 + 0.099 * u();
                ((-dt).exp(), dt)
            }
            Mode::Lin => (1.0, 1.0),
        })
        .unzip();
    Data {
        q,
        k,
        v,
        alpha: alpha.into_iter().map(|x| x as f32 as f64).collect(),
        beta: beta.into_iter().map(|x| x as f32 as f64).collect(),
    }
}

fn graph(mode: Mode, h: usize, t: usize, n: usize, m: usize) -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let q = g.input(&[h, t, n]);
    let k = g.input(&[h, t, n]);
    let v = g.input(&[h, t, m]);
    let al = g.input(&[h, t]);
    let be = g.input(&[h, t]);
    let (a, b, delta) = match mode {
        Mode::Gdn => (Some(al), Some(be), true),
        Mode::Ssd => (Some(al), Some(be), false),
        Mode::Lin => (None, None, false),
    };
    let s = g.delta_scan(k, v, a, b, 1, delta).unwrap();
    let scale = if mode == Mode::Ssd { 1.0 } else { 1.0 / (n as f64).sqrt() };
    let o = g.scan_readout(s, q, scale).unwrap();
    (g, o)
}

struct Run {
    out: CudaSlice<u16>,
    launches: Vec<(tang_gpu::cuda::cudarc::driver::CudaFunction, LaunchConfig)>,
    ins: Vec<CudaSlice<u16>>,
    gates: Vec<CudaSlice<f32>>,
    scratch16: Vec<CudaSlice<u16>>,
    delta: bool,
    gf: Vec<CudaSlice<f32>>,
    /// backward: T (f32), A_qk, dS_end, dV' (bf16), dO (bf16)
    ts: CudaSlice<f32>,
    b16: Vec<CudaSlice<u16>>,
    dout: CudaSlice<u16>,
    /// dq dk dv (bf16), dα dβ (f32)
    g16: Vec<CudaSlice<u16>>,
    grads: Vec<CudaSlice<f32>>,
}

fn setup(dev: &CudaDevice, plan: &ChunkedScan, d: &Data, bv: usize, dout: Option<&[f64]>) -> Run {
    let s = &dev.stream;
    let up16 = |x: &[f64]| s.memcpy_stod(&x.iter().map(|&v| bf16_bits(v)).collect::<Vec<_>>()).unwrap();
    let up32 = |x: &[f64]| s.memcpy_stod(&x.iter().map(|&v| v as f32).collect::<Vec<_>>()).unwrap();
    let ins = vec![up16(&d.q), up16(&d.k), up16(&d.v)];
    let gates = vec![up32(&d.alpha), up32(&d.beta)];
    let sz = plan.scratch();
    let scratch16 = sz[..4].iter().map(|&n| s.alloc_zeros::<u16>(n).unwrap()).collect();
    let gf = sz[4..].iter().map(|&n| s.alloc_zeros::<f32>(n).unwrap()).collect();
    let out = s.alloc_zeros::<u16>(plan.heads * plan.t * plan.m).unwrap();
    let bwd = dout.is_some();
    let bs = plan.backward_scratch();
    let ts = s.alloc_zeros::<f32>(if bwd { bs[0] } else { 1 }).unwrap();
    let b16 = if bwd { bs[1..].iter().map(|&n| s.alloc_zeros::<u16>(n).unwrap()).collect() } else { Vec::new() };
    let rows = plan.heads * plan.t;
    let dout = up16(dout.unwrap_or(&[0.0]));
    let (g16, grads) = if bwd {
        (
            [rows * plan.n, rows * plan.n, rows * plan.m].iter().map(|&n| s.alloc_zeros::<u16>(n).unwrap()).collect(),
            [rows, rows].iter().map(|&n| s.alloc_zeros::<f32>(n).unwrap()).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let mut ks: Vec<_> = plan.kernels_saving(bv, bwd).into_iter().collect();
    if bwd {
        ks.extend(plan.backward_kernels(bv));
    }
    let launches = ks
        .into_iter()
        .map(|l| {
            let f = dev.function(&l.source);
            let cfg = LaunchConfig {
                grid_dim: l.grid,
                block_dim: (l.block, 1, 1),
                shared_mem_bytes: l.smem,
            };
            (f, cfg)
        })
        .collect();
    Run {
        out,
        launches,
        ins,
        gates,
        scratch16,
        delta: plan.delta,
        gf,
        ts,
        b16,
        dout,
        g16,
        grads,
    }
}

fn launch(dev: &CudaDevice, r: &Run) {
    for k in 0..r.launches.len() {
        launch_k(dev, r, k);
    }
}

/// Launch `k` with its arguments (see `ChunkedScan::kernels`).
fn launch_k(dev: &CudaDevice, r: &Run, k: usize) {
    let s = &dev.stream;
    let (f, cfg) = &r.launches[k];
    let sc = &r.scratch16;
    // V' is U itself without the delta term
    let vn = if r.delta { &sc[3] } else { &sc[1] };
    let mut b = s.launch_builder(f);
    let (q, kk, v) = (&r.ins[0], &r.ins[1], &r.ins[2]);
    let (gr, gc) = (&r.gf[0], &r.gf[1]);
    match k {
        0 => {
            b.arg(kk).arg(v).arg(&r.gates[0]).arg(&r.gates[1]);
            b.arg(&sc[0]).arg(&sc[1]).arg(gr).arg(gc).arg(&r.ts);
        }
        1 => {
            b.arg(&sc[0]).arg(&sc[1]).arg(kk).arg(gr).arg(gc).arg(&sc[2]).arg(&sc[3]);
        }
        2 => {
            b.arg(q).arg(kk).arg(gr).arg(&sc[2]).arg(vn).arg(&r.out);
        }
        3 => {
            b.arg(q).arg(kk).arg(gr).arg(&r.b16[0]);
        }
        4 => {
            b.arg(q).arg(kk).arg(&sc[0]).arg(&r.b16[0]).arg(&r.dout).arg(gr).arg(gc);
            b.arg(&r.b16[1]).arg(&r.b16[2]);
        }
        _ => {
            b.arg(q).arg(kk).arg(v).arg(&r.gates[0]).arg(&r.gates[1]).arg(gr).arg(gc);
            b.arg(&r.ts).arg(&sc[2]).arg(&r.b16[1]).arg(&r.dout).arg(vn).arg(&r.b16[2]);
            for x in &r.g16 {
                b.arg(x);
            }
            for x in &r.grads {
                b.arg(x);
            }
        }
    }
    unsafe { b.launch(*cfg) }.expect("launch");
}

/// Mean ms of `f` over 20 back-to-back runs (one sync at the end, so
/// host overhead is not counted), the best of 5 such batches.
fn median(dev: &CudaDevice, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..5 {
        dev.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            f();
        }
        dev.sync();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3 / 20.0);
    }
    best
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = match a.get(1).map(String::as_str) {
        Some("gdn") => Mode::Gdn,
        Some("ssd") => Mode::Ssd,
        Some("lin") => Mode::Lin,
        _ => panic!("usage: scan_bench gdn|ssd|lin H T [N M BV]"),
    };
    let num = |i: usize, d: usize| a.get(i).map_or(d, |s| s.parse().unwrap());
    let (h, t, n, m, bv) = (num(2, 32), num(3, 4096), num(4, 128), num(5, 128), num(6, 32));
    let dev = CudaDevice::new(0).expect("cuda device");
    if a.iter().any(|x| x == "--bwd") {
        return backward(&dev, mode, h, t, n, m, bv);
    }

    // check against the interpreter
    let (ch, ct) = (1, 128);
    let (g, o) = graph(mode, ch, ct, n, m);
    let plan = g.chunked_scan_plan(o).expect("chunked plan");
    let d = data(mode, ch, ct, n, m);
    let want = &g.eval(
        &[d.q.clone(), d.k.clone(), d.v.clone(), d.alpha.clone(), d.beta.clone()],
        &[o],
    )[0];
    let r = setup(&dev, &plan, &d, bv, None);
    launch(&dev, &r);
    dev.sync();
    let got: Vec<f64> = dev
        .stream
        .memcpy_dtov(&r.out)
        .unwrap()
        .into_iter()
        .map(bf16_from_bits)
        .collect();
    let (mut e, mut mx) = (0f64, 0f64);
    for (x, y) in got.iter().zip(want) {
        e = e.max((x - y).abs());
        mx = mx.max(y.abs());
    }
    let rel = e / mx;

    // time
    let (g, o) = graph(mode, h, t, n, m);
    let plan = g.chunked_scan_plan(o).expect("chunked plan");
    let d = data(mode, h, t, n, m);
    let r = setup(&dev, &plan, &d, bv, None);
    for _ in 0..3 {
        launch(&dev, &r);
    }
    dev.sync();
    let total = median(&dev, || launch(&dev, &r));
    let k1 = median(&dev, || launch_k(&dev, &r, 0));
    let k2 = median(&dev, || launch_k(&dev, &r, 1));
    let k3 = median(&dev, || launch_k(&dev, &r, 2));
    let name = match mode {
        Mode::Gdn => "gdn",
        Mode::Ssd => "ssd",
        Mode::Lin => "lin",
    };
    println!(
        "{name} H={h} T={t} n={n} m={m} bv={bv} {total:.3} ms (chunk {k1:.3} + state {k2:.3} + out {k3:.3}) rel_err={rel:.2e} ({})",
        dev.name
    );
}

/// `--bwd`: gradients of `Σ o ⊙ r` against the interpreter's derived
/// backward at H = 1, T = 128, then forward + backward timed at H × T.
fn backward(dev: &CudaDevice, mode: Mode, h: usize, t: usize, n: usize, m: usize, bv: usize) {
    let (ch, ct) = (1, 128);
    let (mut g, o) = graph(mode, ch, ct, n, m);
    let plan = g.chunked_scan_plan(o).expect("chunked plan");
    let d = data(mode, ch, ct, n, m);
    let rr: Vec<f64> = {
        let mut u = rng(5);
        (0..ch * ct * m).map(|_| bf(u() - 0.5)).collect()
    };
    // loss = Σ o ⊙ r
    let ri = g.input(&[ch, ct, m]);
    let prod = g.map(&[o, ri], |x| x[0] * x[1]).unwrap();
    let flat = g.reshape(prod, &[ch * ct * m]).unwrap();
    let loss = g
        .comp(
            &[1usize],
            &[ch * ct * m],
            vec![tang_compile::tensor::Arg {
                src: flat,
                view: tang_compile::tensor::View::pick(2, &[Some(1)]),
            }],
            Some(tang_compile::tensor::Monoid::sum()),
            |x| vec![x[0]],
        )
        .unwrap();
    let wrt: Vec<TensorId> = (0..5).map(TensorId::from_index).collect();
    let grads = g.backward(loss, &wrt).unwrap();
    let want = g.eval(
        &[d.q.clone(), d.k.clone(), d.v.clone(), d.alpha.clone(), d.beta.clone(), rr.clone()],
        &grads,
    );
    let r = setup(dev, &plan, &d, bv, Some(&rr));
    launch(dev, &r);
    dev.sync();
    let names = ["dq", "dk", "dv", "dα", "dβ"];
    let used = match mode {
        Mode::Lin => 3,
        _ => 5,
    };
    let mut errs = Vec::new();
    for i in 0..used {
        let got: Vec<f64> = if i < 3 {
            dev.stream.memcpy_dtov(&r.g16[i]).unwrap().into_iter().map(bf16_from_bits).collect()
        } else {
            dev.stream.memcpy_dtov(&r.grads[i - 3]).unwrap().into_iter().map(|x| x as f64).collect()
        };
        let (mut e, mut mx) = (0f64, 0f64);
        for (x, y) in got.iter().zip(&want[i]) {
            e = e.max((x - y).abs());
            mx = mx.max(y.abs());
        }
        errs.push(format!("{}={:.1e}", names[i], e / mx));
    }

    let (g, o) = graph(mode, h, t, n, m);
    let plan = g.chunked_scan_plan(o).expect("chunked plan");
    let d = data(mode, h, t, n, m);
    let rr: Vec<f64> = {
        let mut u = rng(5);
        (0..h * t * m).map(|_| bf(u() - 0.5)).collect()
    };
    let r = setup(dev, &plan, &d, bv, Some(&rr));
    for _ in 0..3 {
        launch(dev, &r);
    }
    dev.sync();
    let total = median(dev, || launch(dev, &r));
    let fwd = median(dev, || (0..3).for_each(|k| launch_k(dev, &r, k)));
    let per: Vec<String> = (3..6).map(|k| format!("{:.3}", median(dev, || launch_k(dev, &r, k)))).collect();
    let name = match mode {
        Mode::Gdn => "gdn",
        Mode::Ssd => "ssd",
        Mode::Lin => "lin",
    };
    println!(
        "{name} H={h} T={t} n={n} m={m} bv={bv} fwd+bwd {total:.3} ms (fwd {fwd:.3}, bwd aqk/state/grad {}) rel_err {} ({})",
        per.join("/"),
        errs.join(" "),
        dev.name
    );
}
