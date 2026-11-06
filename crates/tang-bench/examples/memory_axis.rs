//! The memory-placement axis on causal attention: the named points
//! (naive, checkpointing, FlashAttention-2-like) and the autotuner's pick,
//! each derived from the same graph by `TensorGraph::place` and timed on
//! the GPU through wgpu (Metal on an M4, Vulkan on a 3090). One JSON object
//! per line:
//!
//!   {point, device, op, n, d, ms, hbm_bytes, kernels, nxn_buffers, correct, check, note}
//!
//! `hbm_bytes` is the compulsory-traffic model: every kernel reads each
//! buffer it reads once and writes its own once (tiles' re-reads of staged
//! operands are not counted). `correct` checks against the f64 interpreter
//! at n ≤ 1024 and against the verified flash point's output above that.
//! Points that cannot run here are emitted with `ms: null` and a reason.
//!
//!   N=1024,4096,16384 D=64,128 OPS=fwd,fwdbwd DEVICE=m4 \
//!     cargo run --release -p tang-bench --example memory_axis
use tang_bench::ir_gpu::{time_program, Executable};
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{MemorySchedule, TensorGraph, TensorId};
use tang_derive_wasm::{naive_graph, parse_variant};
use tang_gpu::{GpuDevice, KernelCache};

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn env_list(k: &str, default: &str) -> Vec<String> {
    std::env::var(k)
        .unwrap_or(default.into())
        .split(',')
        .map(String::from)
        .collect()
}

fn hbm_bytes(p: &WgslProgram) -> u64 {
    let len = |t: TensorId| {
        let s = p.storage(t);
        p.buffers
            .iter()
            .find(|b| b.node == s)
            .map_or(0, |b| b.len as u64 * 4)
    };
    p.kernels
        .iter()
        .map(|k| {
            let mut r: Vec<TensorId> = k.reads.iter().map(|&t| p.storage(t)).collect();
            r.sort_by_key(|t| t.index());
            r.dedup();
            r.iter().map(|&t| len(t)).sum::<u64>() + len(k.node)
        })
        .sum()
}

fn alloc_bytes(p: &WgslProgram) -> u64 {
    p.buffers.iter().map(|b| b.len as u64 * 4).sum()
}

fn max_rel(got: &[Vec<f32>], want: &[Vec<f64>]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let scale = w.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
            g.iter()
                .zip(w)
                .fold(0.0f64, |m, (&a, &b)| m.max((a as f64 - b).abs()))
                / scale
        })
        .fold(0.0, f64::max)
}

struct Run {
    ms: f64,
    hbm: u64,
    kernels: usize,
    nxn: usize,
    out: Vec<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn run(
    device: &GpuDevice,
    cache: &mut KernelCache,
    g: &TensorGraph,
    outs: &[TensorId],
    nest: bool,
    ins: &[&[f32]],
    n: usize,
    iters: usize,
    budget: u64,
) -> Result<Run, String> {
    let p = g.to_wgsl_with(outs, nest);
    let bytes = alloc_bytes(&p);
    if bytes > budget {
        return Err(format!(
            "needs {:.1} GB of buffers, over this harness's {:.1} GB budget",
            bytes as f64 / 1e9,
            budget as f64 / 1e9
        ));
    }
    let big = p
        .buffers
        .iter()
        .filter(|b| b.input.is_none() && b.len >= n * n)
        .count();
    let hbm = hbm_bytes(&p);
    let kernels = p.kernels.len();
    let exe = Executable::new(device, p, ins);
    let ms = (0..3)
        .map(|_| time_program(device, cache, &exe, outs[0], iters))
        .fold(f64::INFINITY, f64::min);
    let out = outs.iter().map(|o| exe.read(device, *o)).collect();
    Ok(Run {
        ms,
        hbm,
        kernels,
        nxn: big,
        out,
    })
}

fn main() {
    let device = GpuDevice::new_sync().expect("GPU");
    let mut cache = KernelCache::new();
    let dev_name = std::env::var("DEVICE").unwrap_or("gpu".into());
    let budget: u64 = std::env::var("BUDGET_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(16e9 as u64, |g| (g * 1e9) as u64);
    let ns: Vec<usize> = env_list("N", "1024,4096")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let ds: Vec<usize> = env_list("D", "64")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let ops = env_list("OPS", "fwd,fwdbwd");
    let tune = std::env::var("NO_TUNE").is_err();
    let v = parse_variant("s", "j <= i").unwrap();
    let emit = |point: &str,
                op: &str,
                n: usize,
                d: usize,
                r: Result<(&Run, Option<bool>, &str), String>| {
        match r {
            Ok((r, ok, check)) => println!(
                "{{\"point\":\"{point}\",\"device\":\"{dev_name}\",\"op\":\"{op}\",\"n\":{n},\"d\":{d},\"ms\":{:.4},\"hbm_bytes\":{},\"kernels\":{},\"nxn_buffers\":{},\"correct\":{},\"check\":\"{check}\",\"note\":null}}",
                r.ms, r.hbm, r.kernels, r.nxn,
                ok.map_or("null".into(), |b| b.to_string())
            ),
            Err(why) => println!(
                "{{\"point\":\"{point}\",\"device\":\"{dev_name}\",\"op\":\"{op}\",\"n\":{n},\"d\":{d},\"ms\":null,\"hbm_bytes\":null,\"kernels\":null,\"nxn_buffers\":null,\"correct\":null,\"check\":null,\"note\":\"{}\"}}",
                why.replace('"', "'")
            ),
        }
    };
    for &d in &ds {
        for &n in &ns {
            for op in &ops {
                let bwd = op == "fwdbwd";
                let iters = if n >= 16384 {
                    3
                } else if n >= 4096 {
                    5
                } else {
                    20
                };
                let (g, outs) = naive_graph(&v, n, d, bwd).unwrap();
                let nin = if bwd { 4 } else { 3 };
                let inputs: Vec<Vec<f32>> = (0..nin).map(|k| rand(n * d, 7 + k as u64)).collect();
                let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
                let want = (n <= 1024).then(|| {
                    let f: Vec<Vec<f64>> = inputs
                        .iter()
                        .map(|v| v.iter().map(|&x| x as f64).collect())
                        .collect();
                    g.eval(&f, &outs)
                });
                // tolerance: f32 accumulation over n terms
                let tol = 2e-3;
                // flash first: it is the cross-check above n = 1024
                let fl = g.place(&outs, &MemorySchedule::flash());
                let flash = run(
                    &device,
                    &mut cache,
                    &fl.graph,
                    &fl.outputs,
                    fl.nest,
                    &ins,
                    n,
                    iters,
                    budget,
                );
                let reference: Option<Vec<Vec<f64>>> = want.clone().or_else(|| {
                    flash.as_ref().ok().map(|r| {
                        r.out
                            .iter()
                            .map(|o| o.iter().map(|&x| x as f64).collect())
                            .collect()
                    })
                });
                let check = if want.is_some() { "f64" } else { "vs-flash" };
                let verdict = |r: &Run| reference.as_ref().map(|w| max_rel(&r.out, w) < tol);
                for (name, s) in MemorySchedule::named() {
                    let r = if name == "flash" {
                        flash
                            .as_ref()
                            .map_err(|e| e.clone())
                            .map(|r| (r, if want.is_some() { verdict(r) } else { None }))
                    } else {
                        let p = g.place(&outs, &s);
                        match run(
                            &device, &mut cache, &p.graph, &p.outputs, p.nest, &ins, n, iters,
                            budget,
                        ) {
                            Ok(r) => {
                                let ok = verdict(&r);
                                emit(name, op, n, d, Ok((&r, ok, check)));
                                continue;
                            }
                            Err(e) => Err(e),
                        }
                    };
                    let chk = if name == "flash" && want.is_none() {
                        "self (reference)"
                    } else {
                        check
                    };
                    emit(name, op, n, d, r.map(|(r, ok)| (r, ok, chk)));
                }
                emit("fa3", op, n, d, Err("FA3 = FA2 placement + async TMA/wgmma staging; needs Hopper (sm_90). Not this device; tang's wgmma lowering is on an unmerged branch".into()));
                emit("megakernel", op, n, d, Err("not yet: tang has no whole-layer single-kernel lowering (boundary=Fuse across the projections) in the tensor IR".into()));
                if !tune {
                    continue;
                }
                // autotune: every role-level placement, then each kernel's
                // thread mappings on the winner (one kernel at a time)
                let mut best: Option<(f64, TensorGraph, Vec<TensorId>, bool, MemorySchedule)> =
                    None;
                let titers = iters.min(5);
                for s in MemorySchedule::candidates() {
                    let p = g.place(&outs, &s);
                    if let Ok(r) = run(
                        &device, &mut cache, &p.graph, &p.outputs, p.nest, &ins, n, titers, budget,
                    ) {
                        if verdict(&r) != Some(false) && best.as_ref().is_none_or(|b| r.ms < b.0) {
                            best = Some((r.ms, p.graph, p.outputs, p.nest, s));
                        }
                    }
                }
                let Some((mut bms, mut bg, bo, nest, bs)) = best else {
                    emit("autotune", op, n, d, Err("no candidate ran".into()));
                    continue;
                };
                let prog = bg.to_wgsl_with(&bo, nest);
                let nodes: Vec<TensorId> = prog.kernels.iter().map(|k| k.node).collect();
                for t in nodes {
                    if t.index() >= bg.nodes().len() {
                        continue;
                    }
                    for (split, tile, reg) in bg.schedule_candidates(t) {
                        let mut c = bg.clone();
                        c.set_schedule(t, split, tile, reg);
                        if let Ok(r) =
                            run(&device, &mut cache, &c, &bo, nest, &ins, n, titers, budget)
                        {
                            if verdict(&r) != Some(false) && r.ms < bms {
                                bms = r.ms;
                                bg = c;
                            }
                        }
                    }
                }
                eprintln!("autotune {op} n={n} d={d}: {bs:?}");
                let r = run(&device, &mut cache, &bg, &bo, nest, &ins, n, iters, budget);
                let tag = format!(
                    "autotune:{:?}/{:?}/{:?}/{:?}",
                    bs.contraction, bs.pointwise, bs.normalizer, bs.boundary
                );
                emit(
                    &tag,
                    op,
                    n,
                    d,
                    r.as_ref()
                        .map_err(|e| e.clone())
                        .map(|r| (r, verdict(r), check)),
                );
            }
        }
    }
}
