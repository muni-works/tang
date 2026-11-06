//! The derived attention forward + backward (`tang-derive-wasm`'s presets,
//! through the training pass pipeline) on the GPU, with the short
//! contractions S = Q·Kᵀ and dP = dO·Vᵀ materialized (as scheduled) and
//! nested in the dQ/dK/dV tiles (`to_wgsl` / `to_msl` default). Whole
//! program, mean over launches after warm-up, best of three. Gradients are
//! checked against the f64 interpreter at n ≤ 1024.
//!
//!   cargo run --release -p tang-bench --example attention_backward            # wgpu (Metal / Vulkan)
//!   cargo run --release -p tang-bench --features metal --example attention_backward   # + native MSL
use tang_bench::ir_gpu::{time_program, Executable};
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_derive_wasm::{derive_graph, naive_graph, parse_variant};
use tang_gpu::{GpuDevice, KernelCache};

const ITERS: usize = 20;

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn rel_err(got: &[Vec<f32>], want: &[Vec<f64>]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let scale = w.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
            g.iter().zip(w).fold(0.0f64, |m, (&a, &b)| m.max((a as f64 - b).abs())) / scale
        })
        .fold(0.0, f64::max)
}

fn nxn(p: &tang_compile::tensor::wgsl::WgslProgram, n: usize) -> usize {
    p.buffers.iter().filter(|b| b.input.is_none() && b.len >= n * n).count()
}

fn main() {
    let device = GpuDevice::new_sync().expect("GPU");
    let mut cache = KernelCache::new();
    #[cfg(feature = "metal")]
    let msl = tang_bench::ir_msl::MslDevice::new();
    let d = 64;
    let presets = [("causal", "s", "j <= i"), ("bidirectional", "s", "")];
    println!("| preset | n | backend | materialized ms (n×n) | nested ms (n×n) | speedup | max rel err |");
    println!("|---|---|---|---|---|---|---|");
    let sizes: Vec<usize> = std::env::var("N")
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![512, 1024, 2048, 4096]);
    for n in sizes {
        for (name, score, mask) in presets {
            let v = parse_variant(score, mask).unwrap();
            let (g, outs) = naive_graph(&v, n, d, true).unwrap();
            let inputs: Vec<Vec<f32>> = (0..4).map(|k| rand(n * d, 7 + k)).collect();
            let want = (n <= 1024).then(|| {
                let f: Vec<Vec<f64>> = inputs
                    .iter()
                    .map(|v| v.iter().map(|&x| x as f64).collect())
                    .collect();
                g.eval(&f, &outs)
            });
            let (dg, douts) = derive_graph(&g, &outs);
            let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
            let wgpu = |dg: &TensorGraph, nested: bool, cache: &mut KernelCache| {
                let p = dg.to_wgsl_with(&douts, nested);
                let big = nxn(&p, n);
                let exe = Executable::new(&device, p, &ins);
                let ms = (0..3)
                    .map(|_| time_program(&device, cache, &exe, douts[0], ITERS))
                    .fold(f64::INFINITY, f64::min);
                let got: Vec<Vec<f32>> = douts.iter().map(|o| exe.read(&device, *o)).collect();
                (ms, big, got)
            };
            let (ms0, big0, got0) = wgpu(&dg, false, &mut cache);
            let (ms1, big1, got1) = wgpu(&dg, true, &mut cache);
            let err = want
                .as_ref()
                .map_or("—".into(), |w| format!("{:.1e} / {:.1e}", rel_err(&got0, w), rel_err(&got1, w)));
            println!(
                "| {name} | {n} | wgpu | {ms0:.3} ({big0}) | {ms1:.3} ({big1}) | {:.2}× | {err} |",
                ms0 / ms1
            );
            #[cfg(feature = "metal")]
            if let Some(dev) = &msl {
                use tang_bench::ir_msl::Executable as MslExe;
                use tang_compile::tensor::msl::FragmentTarget;
                let run = |nested: bool| {
                    let p = dg.to_msl_with(&douts, &FragmentTarget::APPLE, nested);
                    let big = p.buffers.iter().filter(|b| b.input.is_none() && b.len >= n * n).count();
                    let names: Vec<usize> = p.kernels.iter().map(|k| k.node.index()).collect();
                    let exe = MslExe::new(dev, p, &ins);
                    if std::env::var("PER_KERNEL").is_ok() {
                        for (i, node) in names.iter().enumerate() {
                            let ms = exe.time_kernels(dev, i..i + 1, ITERS);
                            eprintln!("  nested={nested} kernel {i} (node {node}): {ms:.3} ms");
                        }
                    }
                    let ms = (0..3).map(|_| exe.time(dev, ITERS)).fold(f64::INFINITY, f64::min);
                    exe.run(dev);
                    let got: Vec<Vec<f32>> = douts.iter().map(|o| exe.read(*o)).collect();
                    (ms, big, got)
                };
                let (ms0, big0, got0) = run(false);
                let (ms1, big1, got1) = run(true);
                let err = want.as_ref().map_or("—".into(), |w| {
                    format!("{:.1e} / {:.1e}", rel_err(&got0, w), rel_err(&got1, w))
                });
                println!(
                    "| {name} | {n} | msl | {ms0:.3} ({big0}) | {ms1:.3} ({big1}) | {:.2}× | {err} |",
                    ms0 / ms1
                );
            }
            let _: &[TensorId] = &douts;
        }
    }
}
