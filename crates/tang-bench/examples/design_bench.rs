//! Times the design-space candidates (`tang_derive_wasm::design`) through
//! `Program::tuned`, forward and forward+backward (q, k, v grads).
//! `cargo run --release -p tang-bench --features cuda --example design_bench -- [n,..] [d]`
//! CSV: name,norm,n,d,pass,ms,kernels,nxn
fn main() {
    use tang_derive_wasm::design::{design_graph, gated, Norm};
    use tang_derive_wasm::{derive_graph, parse_variant};
    use tang_gpu::program::{Program, Runtime};
    let args: Vec<String> = std::env::args().collect();
    let ns: Vec<usize> = args.get(1).map_or(vec![1024, 2048, 4096], |s| {
        s.split(',').map(|x| x.parse().unwrap()).collect()
    });
    let d: usize = args.get(2).map_or(64, |s| s.parse().unwrap());
    let rt = Runtime::select().expect("gpu");
    eprintln!("backend: {:?} — {}", rt.kind(), rt.name());
    let cands = [
        ("softmax", "s", Norm::Softmax),
        ("forget", "s + u - w", Norm::Softmax),
        ("softcap_forget", "30*tanh(s/30) + u - w", Norm::Softmax),
        ("logdist", "s - log(1 + i - j)", Norm::Softmax),
        ("sigmoid", "s - log(n)", Norm::Sigmoid),
        ("forget_sigmoid", "s + u - w - log(n)", Norm::Sigmoid),
    ];
    let cache = std::path::PathBuf::from("design-tuning.txt");
    println!("name,norm,n,d,pass,ms,kernels,nxn");
    for &n in &ns {
        for (name, score, norm) in cands {
            for bwd in [false, true] {
                let v = parse_variant(score, "j <= i").unwrap();
                let (g, outs) = design_graph(&v, norm, n, d, bwd, false).unwrap();
                let (dg, douts) = derive_graph(&g, &outs);
                let mut rng = tang_bench::Rng::new(7);
                let mut ins: Vec<Vec<f32>> = (0..3)
                    .map(|_| (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect())
                    .collect();
                if gated(&v) {
                    // cumulative log forget gate: decreasing, ~ -0.05 per step
                    ins.push((0..n).map(|t| -0.05 * t as f32).collect());
                }
                if bwd {
                    ins.push((0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect());
                }
                let refs: Vec<&[f32]> = ins.iter().map(|x| x.as_slice()).collect();
                let pass = if bwd { "fwdbwd" } else { "fwd" };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let (exe, _) = Program::tuned(&rt, &dg, &douts, &refs, Some(&cache));
                    exe.run(&rt);
                    let ms = exe.time(&rt, 20);
                    let nxn = exe
                        .program
                        .buffers
                        .iter()
                        .filter(|b| b.input.is_none() && b.len >= n * n)
                        .count();
                    (ms, exe.program.kernels.len(), nxn)
                }));
                match r {
                    Ok((ms, k, x)) => {
                        println!("{name},{},{n},{d},{pass},{ms:.4},{k},{x}", norm.name())
                    }
                    Err(_) => println!("{name},{},{n},{d},{pass},FAIL,,", norm.name()),
                }
            }
        }
    }
}
