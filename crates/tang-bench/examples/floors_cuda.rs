//! Tuned flash attention and matmul tile programs for the traffic-floor tables: the program
//! is tuned first, then run inside one `cuProfilerStart` / `Stop` window, so
//! `ncu --profile-from-start off` sees only the kernels the timing below measures. Per-kernel
//! times come from the same program, so an idle GPU gives clean times and a shared one, bytes.
//!
//!   floors_cuda flash_fwd|flash_bwd|matmul1024|matmul512 [n] [d]
use tang_compile::tensor::{MemorySchedule, TensorGraph, TensorId};
use tang_derive_wasm::{naive_graph, parse_variant};
use tang_gpu::program::{Program, Runtime};

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let what = a.first().map_or("flash_fwd", String::as_str);
    let n: usize = a.get(1).map_or(4096, |s| s.parse().unwrap());
    let d: usize = a.get(2).map_or(64, |s| s.parse().unwrap());
    let rt = Runtime::cuda().expect("cuda");
    let (g, outs, inputs, nest): (TensorGraph, Vec<TensorId>, Vec<Vec<f32>>, bool) = match what {
        "flash_fwd" | "flash_bwd" => {
            let bwd = what == "flash_bwd";
            let v = parse_variant("s", "j <= i").unwrap();
            let (g, outs) = naive_graph(&v, n, d, bwd).unwrap();
            let pl = g.place(&outs, &MemorySchedule::flash());
            let ins = (0..if bwd { 4 } else { 3 }).map(|k| rand(n * d, 7 + k)).collect();
            (pl.graph, pl.outputs, ins, pl.nest)
        }
        "matmul1024" | "matmul512" => {
            let (m, k, r) = if what == "matmul1024" { (1024, 1024, 1024) } else { (512, 576, 1536) };
            let mut g = TensorGraph::new();
            let x = g.input(&[m, k]);
            let y = g.input(&[k, r]);
            let c = g.matmul(x, y).unwrap();
            g.schedule_auto(&[c]);
            (g, vec![c], vec![rand(m * k, 1), rand(k * r, 2)], true)
        }
        _ => panic!("unknown workload {what}"),
    };
    let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let (p, _) = Program::tuned_placed(&rt, &g, &outs, &ins, nest);
    p.run(&rt);
    rt.sync();
    println!("{what} n={n} d={d}: program {:.4} ms, {} kernels", p.time(&rt, 20), p.program.kernels.len());
    for (i, k) in p.program.kernels.iter().enumerate() {
        println!(
            "  kernel {i}: {:.4} ms count {} block {} {}",
            p.time_kernel(&rt, i, 20),
            k.count,
            k.workgroup_size,
            k.source.lines().next().unwrap_or("").chars().take(60).collect::<String>()
        );
    }
    rt.sync();
    rt.profiler(true);
    p.run(&rt);
    rt.sync();
    rt.profiler(false);
}
