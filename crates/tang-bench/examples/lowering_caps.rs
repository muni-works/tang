//! A/B of the WGSL lowering's device caps (`WgslCaps`: subgroup split
//! combine, native `enable f16`) on the workloads they touch: split row
//! reductions, f16-stored matmul, attention forward and forward+backward.
//! Each pair runs interleaved (portable, caps, portable, …), best of
//! `ROUNDS`, kernel-only (`ITERS` launches, one sync), so drift on a
//! shared GPU hits both sides alike.
//!
//!   cargo run --release -p tang-bench --example lowering_caps
use std::time::Instant;
use tang_bench::ir_gpu::{
    flash_attention_heads_auto, flash_attention_heads_fragment, matmul_candidates,
    naive_attention_heads, Executable,
};
use tang_compile::tensor::{Format, Monoid, Precision, TensorGraph, TensorId, WgslCaps};
use tang_gpu::{GpuDevice, KernelCache};

const ITERS: usize = 30;
const ROUNDS: usize = 5;

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn time(device: &GpuDevice, cache: &mut KernelCache, exe: &Executable, out: TensorId) -> f64 {
    let t = Instant::now();
    for _ in 0..ITERS {
        exe.run(device, cache);
    }
    cache.flush(device);
    let _ = exe.read(device, out);
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

const F16: Precision = Precision {
    storage: Format::F16,
    accum: Format::F32,
};

/// Best-of-ROUNDS ms for `g` lowered with each caps, interleaved; also
/// the max abs difference between the two outputs.
fn ab(
    device: &GpuDevice,
    cache: &mut KernelCache,
    g: &TensorGraph,
    out: TensorId,
    inputs: &[&[f32]],
    caps: &[WgslCaps],
) -> (Vec<f64>, f32) {
    let exes: Vec<Executable> = caps
        .iter()
        .map(|&c| Executable::new(device, g.to_wgsl_caps(&[out], c), inputs))
        .collect();
    for e in &exes {
        e.run(device, cache);
    }
    cache.flush(device);
    let outs: Vec<Vec<f32>> = exes.iter().map(|e| e.read(device, out)).collect();
    let diff = outs[0]
        .iter()
        .zip(outs.last().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mut best = vec![f64::INFINITY; exes.len()];
    for _ in 0..ROUNDS {
        for (b, e) in best.iter_mut().zip(&exes) {
            *b = b.min(time(device, cache, e, out));
        }
    }
    (best, diff)
}

fn main() {
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let full = device.wgsl_caps();
    let none = WgslCaps::default();
    let sg_only = WgslCaps { f16: false, ..full };
    println!(
        "{} — caps {full:?}; best of {ROUNDS}×{ITERS} launches, ms",
        device.name
    );
    println!(
        "{:<44} {:>10} {:>10} {:>8} {:>10}",
        "workload", "portable", "caps", "speedup", "max|diff|"
    );
    let row = |name: &str, t: &[f64], d: f32| {
        println!(
            "{name:<44} {:>10.3} {:>10.3} {:>7.2}x {:>10.2e}",
            t[0],
            t[1],
            t[0] / t[1],
            d
        );
    };

    let only = std::env::var("ONLY").unwrap_or_default();
    if only.is_empty() || only == "caps" {
        // split row reductions (few rows, long reduce): the subgroup combine
        for (rows, len) in [(64usize, 65536usize), (512, 8192)] {
            let x = rand(rows * len, 1);
            for (mname, m) in [
                ("sum", Monoid::sum()),
                ("max", Monoid::max()),
                ("argmax", Monoid::argmax()),
            ] {
                let mut g = TensorGraph::new();
                let xi = g.input(&[rows, len]);
                let w = m.width;
                let y = if w == 1 {
                    g.reduce(xi, 1, m).unwrap()
                } else {
                    use tang_compile::tensor::{Arg, View};
                    g.comp_tuple(
                        &[rows, w],
                        &[len],
                        vec![Arg {
                            src: xi,
                            view: View::identity(2, 2),
                        }],
                        Some(m),
                        |v| vec![v[0], v[0]],
                    )
                    .unwrap()
                };
                g.set_schedule(y, 256, (0, 0, 0), (1, 1));
                g.set_subgroup(y, true);
                let (t, d) = ab(&device, &mut cache, &g, y, &[&x], &[none, sg_only]);
                row(&format!("{mname} [{rows},{len}] split256"), &t, d);
            }
        }

        // matmul under schedule_auto, f32 and f16 storage (f32 accumulate)
        for (m, k, n) in [
            (1024usize, 1024usize, 1024usize),
            (512, 576, 1536),
            (2048, 2048, 2048),
        ] {
            let (a, b) = (rand(m * k, 1), rand(k * n, 2));
            let (mut g, c) = matmul_candidates(m, k, n)
                .into_iter()
                .find(|(nm, ..)| *nm == "auto")
                .map(|(_, g, c)| (g, c))
                .unwrap();
            let (t, d) = ab(&device, &mut cache, &g, c, &[&a, &b], &[none, full]);
            row(&format!("matmul {m}x{k}x{n} f32"), &t, d);
            for i in 0..2 {
                let id = TensorId::from_index(i);
                g.set_precision(id, F16);
            }
            let (t, d) = ab(&device, &mut cache, &g, c, &[&a, &b], &[none, full]);
            row(&format!("matmul {m}x{k}x{n} f16 in"), &t, d);
        }

        // attention forward (schedule_auto), heads h, d = 64
        for (h, n, causal) in [
            (1usize, 1024usize, false),
            (8, 1024, false),
            (8, 2048, true),
        ] {
            let d = 64;
            let (q, k, v) = (rand(h * n * d, 1), rand(h * n * d, 2), rand(h * n * d, 3));
            let (mut g, y) = flash_attention_heads_auto(h, 1, n, d, d, causal);
            let (t, df) = ab(&device, &mut cache, &g, y, &[&q, &k, &v], &[none, full]);
            row(&format!("attn fwd h{h} n{n} causal={causal} f32"), &t, df);
            for i in 0..3 {
                g.set_precision(TensorId::from_index(i), F16);
            }
            let (t, df) = ab(&device, &mut cache, &g, y, &[&q, &k, &v], &[none, full]);
            row(
                &format!("attn fwd h{h} n{n} causal={causal} f16 in"),
                &t,
                df,
            );
        }
    }
    // wgpu on Metal: `fragment` nodes spliced in as native simdgroup-matrix
    // kernels through MSL passthrough (f16 q/k/v, f32 accumulate), vs the
    // f32 WGSL schedule
    if device.msl_passthrough && (only.is_empty() || only == "frag") {
        for (h, n, causal) in [
            (1usize, 1024usize, false),
            (8, 1024, false),
            (8, 2048, false),
            (8, 2048, true),
            (8, 4096, false),
        ] {
            let d = 64;
            let (q, k, v) = (rand(h * n * d, 1), rand(h * n * d, 2), rand(h * n * d, 3));
            let (ga, y) = flash_attention_heads_auto(h, 1, n, d, d, causal);
            let (gf, yf) = flash_attention_heads_fragment(h, 1, n, d, d, causal);
            let ea = Executable::new(&device, ga.to_wgsl_caps(&[y], full), &[&q, &k, &v]);
            let ef = Executable::new(&device, gf.to_msl_mma_caps(&[yf], full), &[&q, &k, &v]);
            for e in [&ea, &ef] {
                e.run(&device, &mut cache);
            }
            cache.flush(&device);
            let diff = ea
                .read(&device, y)
                .iter()
                .zip(ef.read(&device, yf))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mut best = [f64::INFINITY; 2];
            for _ in 0..ROUNDS {
                best[0] = best[0].min(time(&device, &mut cache, &ea, y));
                best[1] = best[1].min(time(&device, &mut cache, &ef, yf));
            }
            let flops = 4.0 * (h * n * n * d) as f64 / if causal { 2.0 } else { 1.0 };
            row(
                &format!(
                    "attn h{h} n{n} c={causal} wgsl-f32 | frag {:.1}TF",
                    flops / best[1] / 1e9
                ),
                &best,
                diff,
            );
        }
        for (m, k, n) in [
            (1024usize, 1024usize, 1024usize),
            (2048, 2048, 2048),
            (512, 576, 1536),
        ] {
            let (a, b) = (rand(m * k, 1), rand(k * n, 2));
            let (ga, c) = matmul_candidates(m, k, n)
                .into_iter()
                .find(|(nm, ..)| *nm == "auto")
                .map(|(_, g, c)| (g, c))
                .unwrap();
            let mut gf = ga.clone();
            gf.set_precision(TensorId::from_index(0), F16);
            gf.set_precision(TensorId::from_index(1), F16);
            gf.set_fragment(c, true);
            let ea = Executable::new(&device, ga.to_wgsl_caps(&[c], full), &[&a, &b]);
            let ef = Executable::new(&device, gf.to_msl_mma_caps(&[c], full), &[&a, &b]);
            let spliced = ef
                .program
                .kernels
                .iter()
                .any(|k| k.source.contains("simdgroup_half8x8"));
            for e in [&ea, &ef] {
                e.run(&device, &mut cache);
            }
            cache.flush(&device);
            let diff = ea
                .read(&device, c)
                .iter()
                .zip(ef.read(&device, c))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mut best = [f64::INFINITY; 2];
            for _ in 0..ROUNDS {
                best[0] = best[0].min(time(&device, &mut cache, &ea, c));
                best[1] = best[1].min(time(&device, &mut cache, &ef, c));
            }
            let flops = 2.0 * (m * k * n) as f64;
            row(
                &format!(
                    "matmul {m}x{k}x{n} wgsl-f32 | frag{} {:.1}TF",
                    if spliced { "" } else { "(none)" },
                    flops / best[1] / 1e9
                ),
                &best,
                diff,
            );
        }
    }

    if !(only.is_empty() || only == "bwd") {
        return;
    }
    // attention forward + backward: loss = Σ attention(q,k,v)·dy, grads
    // wrt q, k, v, through the standard pass pipeline + schedule_auto
    let bw: Vec<(usize, usize)> = std::env::var("BWD")
        .map(|v| {
            v.split(';')
                .map(|p| {
                    let (a, b) = p.split_once(',').unwrap();
                    (a.parse().unwrap(), b.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or(vec![(1, 512), (8, 1024)]);
    for (h, n) in bw {
        let d = 64;
        let (mut g, y) = naive_attention_heads(h, 1, n, d, d, false);
        let dy = g.input(&[h, 1, n, d]);
        let prod = g.map(&[y, dy], |v| v[0] * v[1]).unwrap();
        let mut l = prod;
        // reduce the trailing axis until one value is left (reduce keeps
        // a unit dim, so stop on the element count, not the rank)
        while g
            .shape(l)
            .iter()
            .map(|d| d.fixed().unwrap())
            .product::<usize>()
            > 1
        {
            let ax = g
                .shape(l)
                .iter()
                .rposition(|d| d.fixed() != Some(1))
                .unwrap();
            l = g.reduce(l, ax, Monoid::sum()).unwrap();
        }
        let grads = g
            .backward(
                l,
                &[
                    TensorId::from_index(0),
                    TensorId::from_index(1),
                    TensorId::from_index(2),
                ],
            )
            .unwrap();
        let (g, outs) = g.stabilize_softmax(&grads);
        let (g, outs) = g.substitute_maps(&outs);
        let (g, outs) = g.eliminate_common(&outs);
        let (g, outs) = g.merge_siblings(&outs);
        let (g, outs) = g.relayout(&outs);
        let (g, outs) = g.merge_reduce_dims(&outs);
        let (mut g, outs) = g.split_reduce_dims(&outs);
        g.schedule_auto(&outs);
        eprintln!("bwd graph h{h} n{n}: {} nodes", g.nodes().len());
        let ins: Vec<Vec<f32>> = (0..4).map(|s| rand(h * n * d, s + 1)).collect();
        let refs: Vec<&[f32]> = ins.iter().map(|v| v.as_slice()).collect();
        // time the whole program, read one gradient
        let progs = [none, full].map(|c| g.to_wgsl_caps(&outs, c));
        let exes: Vec<Executable> = progs
            .into_iter()
            .map(|p| Executable::new(&device, p, &refs))
            .collect();
        let mut best = [f64::INFINITY; 2];
        for _ in 0..ROUNDS {
            for (b, e) in best.iter_mut().zip(&exes) {
                *b = b.min(time(&device, &mut cache, e, outs[0]));
            }
        }
        let diff = exes[0]
            .read(&device, outs[0])
            .iter()
            .zip(exes[1].read(&device, outs[0]))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        row(
            &format!(
                "attn fwd+bwd h{h} n{n} f32 ({} kernels)",
                exes[0].program.kernels.len()
            ),
            &best,
            diff,
        );
    }
}
