//! tang's derived attention variants on CUDA, for the "Against Flashlight /
//! EqiForge / FlexAttention" table in BENCHMARKS.md.
//!
//! Each variant is built exactly as the post's lab builds it
//! (`tang_derive_wasm::naive_graph` + `derive_graph`), lowered with
//! `Program::tuned` (tensor-core tiles vs transpiled WGSL, per kernel) and
//! timed. Inputs are written to `$OUT/{tag}.in.bin` and outputs to
//! `$OUT/{tag}.out.bin` (raw f32) so `scripts/attn_variants_bench.py` checks
//! them against a torch f64 reference and runs the baselines on the same data.
//!
//! `cargo run --release -p tang-bench --features cuda --example attn_variants -- OUT [n,..] [d,..]`
//! Prints CSV: variant,d,n,pass,ms,kernels,nxn_buffers,native_kept,native_total
fn main() {
    use tang_derive_wasm::{derive_graph, naive_graph, parse_variant};
    use tang_gpu::program::{Program, Runtime};

    let args: Vec<String> = std::env::args().collect();
    let out = std::path::PathBuf::from(args.get(1).cloned().unwrap_or("attn_out".into()));
    std::fs::create_dir_all(&out).unwrap();
    let list = |i: usize, def: &[usize]| -> Vec<usize> {
        args.get(i).map_or(def.to_vec(), |s| s.split(',').map(|x| x.parse().unwrap()).collect())
    };
    let ns = list(2, &[1024, 2048, 4096, 8192]);
    let ds = list(3, &[64, 128]);
    let only = std::env::var("VARIANTS").ok();
    let passes: Vec<bool> = match std::env::var("PASSES").as_deref() {
        Ok("fwd") => vec![false],
        Ok("bwd") => vec![true],
        _ => vec![false, true],
    };
    let rt = Runtime::select().expect("gpu");
    eprintln!("backend: {:?} — {}", rt.kind(), rt.name());
    let presets = [
        ("causal", "s", "j <= i"),
        ("sliding128", "s", "j <= i && i - j < 128"),
        ("alibi", "s - 0.0625*(i - j)", "j <= i"),
        ("softcap", "30*tanh(s/30)", "j <= i"),
        ("prefix64", "s", "j <= i || j < 64"),
        ("bidirectional", "s", ""),
    ];
    let cache = out.join("tang-cuda-tuning.txt");
    println!("variant,d,n,pass,ms,kernels,nxn_buffers,native_kept,native_total");
    for &d in &ds {
        for &n in &ns {
            for (name, score, mask) in presets {
                if only.as_deref().is_some_and(|o| !o.split(',').any(|x| x == name)) {
                    continue;
                }
                for &bwd in &passes {
                    let pass = if bwd { "fwdbwd" } else { "fwd" };
                    let tag = format!("{name}_d{d}_n{n}_{pass}");
                    let mut rng = tang_bench::Rng::new(1234);
                    let k = if bwd { 4 } else { 3 };
                    let ins: Vec<Vec<f32>> =
                        (0..k).map(|_| (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect()).collect();
                    let v = parse_variant(score, mask).unwrap();
                    let (g, outs) = naive_graph(&v, n, d, bwd).unwrap();
                    let (dg, douts) = derive_graph(&g, &outs);
                    let refs: Vec<&[f32]> = ins.iter().map(|x| x.as_slice()).collect();
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let (exe, rep) = Program::tuned(&rt, &dg, &douts, &refs, Some(&cache));
                        exe.run(&rt);
                        let ms = exe.time(&rt, if n >= 4096 { 10 } else { 30 });
                        let res: Vec<f32> = douts.iter().flat_map(|&t| exe.read(&rt, t)).collect();
                        let nxn = exe.program.buffers.iter().filter(|b| b.input.is_none() && b.len >= n * n).count();
                        (ms, exe.program.kernels.len(), nxn, rep, res)
                    }));
                    let Ok((ms, nk, nxn, rep, res)) = res else {
                        println!("{name},{d},{n},{pass},FAIL,,,,");
                        continue;
                    };
                    let w = |p: std::path::PathBuf, v: &[f32]| {
                        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                        std::fs::write(p, b).unwrap();
                    };
                    w(out.join(format!("{tag}.in.bin")), &ins.concat());
                    w(out.join(format!("{tag}.out.bin")), &res);
                    println!("{name},{d},{n},{pass},{ms:.4},{nk},{nxn},{},{}", rep.native_kept, rep.native_total);
                }
            }
        }
    }
}
