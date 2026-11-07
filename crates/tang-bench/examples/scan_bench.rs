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
    gc: CudaSlice<f32>,
}

fn setup(dev: &CudaDevice, plan: &ChunkedScan, d: &Data, bv: usize) -> Run {
    let s = &dev.stream;
    let up16 = |x: &[f64]| s.memcpy_stod(&x.iter().map(|&v| bf16_bits(v)).collect::<Vec<_>>()).unwrap();
    let up32 = |x: &[f64]| s.memcpy_stod(&x.iter().map(|&v| v as f32).collect::<Vec<_>>()).unwrap();
    let ins = vec![up16(&d.q), up16(&d.k), up16(&d.v)];
    let gates = vec![up32(&d.alpha), up32(&d.beta)];
    let sz = plan.scratch();
    let scratch16 = sz[..5].iter().map(|&n| s.alloc_zeros::<u16>(n).unwrap()).collect();
    let gc = s.alloc_zeros::<f32>(sz[5]).unwrap();
    let out = s.alloc_zeros::<u16>(plan.heads * plan.t * plan.m).unwrap();
    let launches = plan
        .kernels(bv)
        .into_iter()
        .map(|l| {
            let f = dev.function(&l.source);
            let cfg = LaunchConfig {
                grid_dim: (l.grid.0, l.grid.1, 1),
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
        gc,
    }
}

fn launch(dev: &CudaDevice, r: &Run) {
    launch_k(dev, r, 0);
    launch_k(dev, r, 1);
}

fn launch_k(dev: &CudaDevice, r: &Run, k: usize) {
    let s = &dev.stream;
    if k == 1 {
        let (f, cfg) = &r.launches[1];
        let mut b = s.launch_builder(f);
        for x in &r.scratch16 {
            b.arg(x);
        }
        b.arg(&r.gc);
        b.arg(&r.out);
        unsafe { b.launch(*cfg) }.expect("state launch");
        return;
    }
    let (f, cfg) = &r.launches[0];
    let mut b = s.launch_builder(f);
    for x in &r.ins {
        b.arg(x);
    }
    for x in &r.gates {
        b.arg(x);
    }
    for x in &r.scratch16 {
        b.arg(x);
    }
    b.arg(&r.gc);
    unsafe { b.launch(*cfg) }.expect("prep launch");
}

/// Median ms of 20 runs of `f`.
fn median(dev: &CudaDevice, mut f: impl FnMut()) -> f64 {
    let mut times = Vec::new();
    for _ in 0..20 {
        let t0 = std::time::Instant::now();
        f();
        dev.sync();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[10]
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

    // check against the interpreter
    let (ch, ct) = (1, 128);
    let (g, o) = graph(mode, ch, ct, n, m);
    let plan = g.chunked_scan_plan(o).expect("chunked plan");
    let d = data(mode, ch, ct, n, m);
    let want = &g.eval(
        &[d.q.clone(), d.k.clone(), d.v.clone(), d.alpha.clone(), d.beta.clone()],
        &[o],
    )[0];
    let r = setup(&dev, &plan, &d, bv);
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
    let r = setup(&dev, &plan, &d, bv);
    for _ in 0..3 {
        launch(&dev, &r);
    }
    dev.sync();
    let total = median(&dev, || launch(&dev, &r));
    let k1 = median(&dev, || launch_k(&dev, &r, 0));
    let k2 = median(&dev, || launch_k(&dev, &r, 1));
    let name = match mode {
        Mode::Gdn => "gdn",
        Mode::Ssd => "ssd",
        Mode::Lin => "lin",
    };
    println!(
        "{name} H={h} T={t} n={n} m={m} bv={bv} {total:.3} ms (chunk {k1:.3} + state {k2:.3}) rel_err={rel:.2e} ({})",
        dev.name
    );
}
