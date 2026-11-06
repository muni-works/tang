//! Kernel-only timings (N launches, one sync, divide) for the shapes in
//! `torch_bench.py`, so derived kernels can be compared with PyTorch
//! without the submit+readback floor the criterion benches include.
use std::time::Instant;
use tang_bench::ir_block::{train_loss_graph, BlockDims};
use tang_bench::ir_block_gpu::{hand_forward, ir_executable, params};
use tang_bench::ir_gpu::{
    adam_step, attention_candidates, matmul_candidates, program_for, Executable,
};
use tang_compile::tensor::TensorId;
use tang_gpu::{kv_attention_fused, GpuBuffer, GpuDevice, GpuTensor, KernelCache};

const ITERS: usize = 50;

fn time_exe(device: &GpuDevice, cache: &mut KernelCache, exe: &Executable, out: TensorId) -> f64 {
    for _ in 0..5 {
        exe.run(device, cache);
    }
    cache.flush(device);
    let _ = exe.read(device, out);
    let t = Instant::now();
    for _ in 0..ITERS {
        exe.run(device, cache);
    }
    cache.flush(device);
    let _ = exe.read(device, out);
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn time_hand(
    device: &GpuDevice,
    cache: &mut KernelCache,
    mut f: impl FnMut(&mut KernelCache) -> GpuTensor,
) -> f64 {
    for _ in 0..5 {
        let _ = f(cache);
    }
    cache.flush(device);
    let t = Instant::now();
    let mut last = None;
    for _ in 0..ITERS {
        last = Some(f(cache));
    }
    cache.flush(device);
    let _ = last.unwrap().to_vec_flushed(device, cache);
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn main() {
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    println!(
        "{} — kernel-only, {ITERS} launches per sample, {:?}",
        device.name,
        device.wgsl_caps()
    );

    // matmul
    for (m, k, n) in [
        (1024usize, 1024usize, 1024usize),
        (128, 576, 1536),
        (512, 576, 1536),
    ] {
        let (a, b) = (rand(m * k, 1), rand(k * n, 2));
        let (at, bt) = (
            GpuTensor::from_slice(&device, &a, &[m, k]),
            GpuTensor::from_slice(&device, &b, &[k, n]),
        );
        let hand = time_hand(&device, &mut cache, |c| {
            tang_gpu::matmul::matmul(&device, c, &at, &bt)
        });
        let mut line = format!("matmul {m}x{k}x{n}: hand {hand:.3} ms");
        for (name, g, c) in matmul_candidates(m, k, n) {
            if name == "split" {
                continue;
            }
            let exe = Executable::new(&device, program_for(&device, &g, &[c]), &[&a, &b]);
            line.push_str(&format!(
                "  {name} {:.3}",
                time_exe(&device, &mut cache, &exe, c)
            ));
            if name == "auto" && device.msl_passthrough {
                let exe = Executable::new(&device, g.to_msl_mma(&[c]), &[&a, &b]);
                line.push_str(&format!(
                    "  mma {:.3}",
                    time_exe(&device, &mut cache, &exe, c)
                ));
            }
        }
        println!("{line}");
    }
    // attention, single head d=64, causal
    let d = 64;
    for n in [512usize, 1024] {
        let (q, k, v) = (rand(n * d, 1), rand(n * d, 2), rand(n * d, 3));
        let (qt, kt, vt) = (
            GpuTensor::from_slice(&device, &q, &[n, d]),
            GpuTensor::from_slice(&device, &k, &[n, d]),
            GpuTensor::from_slice(&device, &v, &[n, d]),
        );
        let hand = time_hand(&device, &mut cache, |c| {
            kv_attention_fused(&device, c, &qt, &kt, &vt, n, n, 1, 1, d, 0)
        });
        let mut line = format!("attention n={n}: hand {hand:.3} ms");
        for (name, g, y) in attention_candidates(n, d, d) {
            if name == "flash_split" {
                continue;
            }
            let exe = Executable::new(&device, program_for(&device, &g, &[y]), &[&q, &k, &v]);
            line.push_str(&format!(
                "  {name} {:.3}",
                time_exe(&device, &mut cache, &exe, y)
            ));
        }
        println!("{line}");
    }
    // block fwd and fwd+bwd
    for seq in [128usize, 512] {
        let dims = BlockDims {
            seq,
            kv_len: seq,
            rope: true,
            bias: true,
            d_model: 576,
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
            ff_dim: 1536,
        };
        let x = rand(seq * 576, 7);
        let input = GpuTensor::from_slice(&device, &x, &[seq, 576]);
        let p = params(&device, &mut cache, dims, x);
        let hand = time_hand(&device, &mut cache, |c| {
            hand_forward(&device, c, &p, &input)
        });
        let (exe, y) = ir_executable(&device, &p, true);
        let ir = time_exe(&device, &mut cache, &exe, y);
        let (g, outs, _) = train_loss_graph(&p.dims);
        let (g, outs) = g.stabilize_softmax(&outs);
        let (g, outs) = g.substitute_maps(&outs);
        let (g, outs) = g.eliminate_common(&outs);
        let (g, outs) = g.merge_siblings(&outs);
        let (g, outs) = g.relayout(&outs);
        let (g, outs) = g.merge_reduce_dims(&outs);
        let (mut g, outs) = g.split_reduce_dims(&outs);
        g.schedule_auto(&outs);
        let target = vec![0.0f32; seq * 576];
        let mut inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
        inputs.push(&target);
        let train = Executable::new(&device, program_for(&device, &g, &outs), &inputs);
        let bwd = time_exe(&device, &mut cache, &train, outs[0]);
        println!("block seq={seq}: fwd hand {hand:.3} ms  fwd ir_auto {ir:.3} ms  fwd+bwd ir_auto {bwd:.3} ms ({} kernels)", train.program.kernels.len());
    }
    // adam 4M
    let n = 4_194_304;
    let (p, g, m, v) = (rand(n, 1), rand(n, 2), rand(n, 3), rand(n, 4));
    let (pb, gb, mb, vb) = (
        GpuBuffer::from_slice(&device, &p),
        GpuBuffer::from_slice(&device, &g),
        GpuBuffer::from_slice(&device, &m),
        GpuBuffer::from_slice(&device, &v),
    );
    let t = Instant::now();
    for _ in 0..ITERS {
        cache.dispatch_adam(
            &device, &pb, &gb, &mb, &vb, n as u32, 1e-3, 0.9, 0.999, 1e-8, 0.19, 0.002,
        );
    }
    cache.flush(&device);
    let _ = pb.to_vec_sync(&device);
    let hand = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
    let (graph, outs) = adam_step(n, 1e-3, 0.9, 0.999, 1e-8);
    let bc = [0.19f32, 0.002];
    let exe = Executable::new(&device, graph.to_wgsl(&outs), &[&p, &g, &m, &v, &bc]);
    println!(
        "adam 4M: hand {hand:.3} ms  ir {:.3} ms",
        time_exe(&device, &mut cache, &exe, outs[2])
    );
}
