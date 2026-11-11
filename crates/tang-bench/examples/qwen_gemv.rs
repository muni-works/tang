//! A 4-bit GEMV (Qwen3-8B's q|k|v shape by default) through the decode
//! lowering, timed alone: `copies` independent GEMVs over one weight in
//! one program (one compute pass), so launch and submit costs amortize.
//!
//! ```text
//! qwen_gemv [rows] [cols] [copies]     (env: TANG_DECODE_ROWS / _WARPS / _UNROLL)
//! ```
//! `GEMV_SOURCE=file` replaces the kernel source (for hand experiments);
//! `GEMV_DUMP=file` writes it.
use tang_bench::ir_qwen::GROUP;
use tang_compile::tensor::cuda_gemv::{DecodeOpts, Packing};
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, View};
use tang_gpu::program::{BackendKind, Program, Runtime};

fn arg(src: tang_compile::tensor::TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View {
            gather: None,
            guard: false,
            forms,
        },
    }
}

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    let rows = a.first().copied().unwrap_or(6144);
    let cols = a.get(1).copied().unwrap_or(4096);
    let copies = a.get(2).copied().unwrap_or(16);
    let groups = cols / GROUP;
    let mut g = TensorGraph::new();
    let q = g.input(&[rows, groups, GROUP]);
    let s = g.input(&[rows, groups]);
    let b = g.input(&[rows, groups]);
    let mut outs = Vec::new();
    let mut xs = Vec::new();
    for _ in 0..copies {
        let x = g.input(&[cols]);
        xs.push(x);
        let u = |d| Form::unit(4, d);
        let dq = g
            .comp(
                &[rows, groups, GROUP],
                &[],
                vec![
                    arg(q, (0..3).map(|d| Form::unit(3, d)).collect()),
                    arg(s, vec![Form::unit(3, 0), Form::unit(3, 1)]),
                    arg(b, vec![Form::unit(3, 0), Form::unit(3, 1)]),
                ],
                None,
                |v| vec![v[0] * v[1] + v[2]],
            )
            .unwrap();
        let mut xf = Form::zero(4);
        xf.coefs[2] = GROUP as i64;
        xf.coefs[3] = 1;
        let y = g
            .comp(
                &[1, rows],
                &[groups, GROUP],
                vec![arg(x, vec![xf]), arg(dq, vec![u(1), u(2), u(3)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        outs.push(y);
    }
    let (g, outs) = g.substitute_maps(&outs);
    let env = |k: &str| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let opts = DecodeOpts {
        packed: vec![(0, Packing::U4), (1, Packing::Bf16), (2, Packing::Bf16)],
        rows: env("TANG_DECODE_ROWS"),
        warps: env("TANG_DECODE_WARPS"),
        unroll: env("TANG_DECODE_UNROLL"),
        ..Default::default()
    };
    let rt = Runtime::wgpu().expect("gpu");
    let mut p = match rt.kind() {
        BackendKind::Wgpu => g.to_msl_decode(&outs, &opts).program,
        BackendKind::Cuda => g.to_cuda_decode(&outs, &opts).program,
        k => unimplemented!("qwen_gemv: no decode lowering for {k:?}"),
    };
    if let Ok(f) = std::env::var("GEMV_DUMP") {
        std::fs::write(f, &p.kernels[0].source).unwrap();
    }
    if let Ok(f) = std::env::var("GEMV_SOURCE") {
        let src = std::fs::read_to_string(f).unwrap();
        for k in &mut p.kernels {
            k.source = src.clone();
        }
    }
    let words = |n: usize, p: Packing| p.words(n);
    let qw: Vec<f32> = (0..words(rows * cols, Packing::U4))
        .map(|i| f32::from_bits((i as u32).wrapping_mul(2654435761)))
        .collect();
    let sw = vec![f32::from_bits(0x3c003c00); words(rows * groups, Packing::Bf16)];
    let x = vec![0.5f32; cols];
    let mut inputs: Vec<&[f32]> = vec![&qw, &sw, &sw];
    for _ in 0..copies {
        inputs.push(&x);
    }
    let mut prog = Program::new(&rt, p, &inputs);
    prog.capture(&rt);
    let bytes = (rows * cols / 2 + 4 * rows * groups) as f64;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        best = best.min(prog.time(&rt, 10) / copies as f64);
    }
    println!(
        "gemv {rows}x{cols}: {:.1} us per GEMV, {:.0} GB/s (weights + scales)",
        best * 1e3,
        bytes / (best * 1e-3) / 1e9
    );
}
