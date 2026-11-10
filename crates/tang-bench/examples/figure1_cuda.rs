//! Figure 1 of the tang paper on the CUDA path: one causal attention graph
//! under four memory schedules (naive, checkpoint, flash, searched), every
//! point checked against the f64 interpreter at the timed size. A placement
//! with `nest` false stays one in the program (`to_cuda_placed`); each
//! schedule takes the per-kernel fastest lowering (`tuned_placed`).
//!
//! JSON lines on stdout: {point, op, n, d, ms, ms_min, ms_max, hbm_bytes, kernels, max_rel_err,
//! correct, tol}; per-kernel times on stderr. `searched` keeps the fastest correct
//! role-level placement, then each kernel's thread mappings.
//!
//!   N=4096 D=64 OPS=fwd,fwdbwd REF_DIR=/tmp/ref cargo run --release -p tang-bench \
//!     --features cuda --example figure1_cuda
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{MemorySchedule, TensorGraph, TensorId};
use tang_derive_wasm::{naive_graph, parse_variant};
use tang_gpu::program::{Program, Runtime};

const TOL: f64 = 2e-3; // one tolerance for every point: f32 accumulation over n terms

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// Compulsory-traffic model: every kernel reads each buffer it reads once and
/// writes its own once.
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

fn reference(
    g: &TensorGraph,
    outs: &[TensorId],
    inputs: &[Vec<f32>],
    tag: &str,
) -> Vec<Vec<f64>> {
    let path = std::env::var("REF_DIR")
        .ok()
        .map(|d| std::path::PathBuf::from(d).join(format!("{tag}.f64")));
    if let Some(b) = path.as_ref().and_then(|p| std::fs::read(p).ok()) {
        let all: Vec<f64> = b
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let per = all.len() / outs.len();
        eprintln!("reference {tag}: read from disk");
        return all.chunks(per).map(<[f64]>::to_vec).collect();
    }
    let t0 = std::time::Instant::now();
    let f: Vec<Vec<f64>> = inputs
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();
    let want = g.eval(&f, outs);
    eprintln!(
        "reference {tag}: f64 interpreter, {:.0} s",
        t0.elapsed().as_secs_f64()
    );
    if let Some(p) = &path {
        let b: Vec<u8> = want
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let _ = std::fs::write(p, b);
    }
    want
}

struct Point {
    ms: f64,
    lo: f64,
    hi: f64,
    hbm: u64,
    kernels: usize,
    err: f64,
}

fn measure(
    rt: &Runtime,
    p: &Program,
    outs: &[TensorId],
    want: &[Vec<f64>],
    iters: usize,
    name: &str,
) -> Point {
    p.run(rt);
    rt.sync();
    let mut t: Vec<f64> = (0..7).map(|_| p.time(rt, iters)).collect();
    t.sort_by(f64::total_cmp);
    p.run(rt);
    rt.sync();
    let got: Vec<Vec<f32>> = outs.iter().map(|o| p.read(rt, *o)).collect();
    for (o, (g, w)) in got.iter().zip(want).enumerate() {
        eprintln!("  {name} output {o}: max rel err {:.3e}", max_rel(std::slice::from_ref(g), std::slice::from_ref(w)));
    }
    for (i, k) in p.program.kernels.iter().enumerate() {
        eprintln!(
            "  {name} kernel {i}: {:.4} ms {}",
            p.time_kernel(rt, i, iters.max(5)),
            k.source.lines().next().unwrap_or("").chars().take(70).collect::<String>()
        );
    }
    Point {
        ms: t[3],
        lo: t[0],
        hi: t[6],
        hbm: hbm_bytes(&p.program),
        kernels: p.program.kernels.len(),
        err: max_rel(&got, want),
    }
}

fn main() {
    let rt = Runtime::cuda().expect("cuda");
    let list = |k: &str, d: &str| -> Vec<String> {
        std::env::var(k).unwrap_or(d.into()).split(',').map(String::from).collect()
    };
    let n: usize = list("N", "4096")[0].parse().unwrap();
    let d: usize = list("D", "64")[0].parse().unwrap();
    let v = parse_variant("s", "j <= i").unwrap();
    for op in list("OPS", "fwd,fwdbwd") {
        let bwd = op == "fwdbwd";
        let (g, outs) = naive_graph(&v, n, d, bwd).unwrap();
        let nin = if bwd { 4 } else { 3 };
        let inputs: Vec<Vec<f32>> = (0..nin).map(|k| rand(n * d, 7 + k as u64)).collect();
        let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let want = reference(&g, &outs, &inputs, &format!("{op}_{n}_{d}"));
        let iters = 5;
        let emit = |point: &str, r: &Point| {
            println!(
                "{{\"point\":\"{point}\",\"op\":\"{op}\",\"n\":{n},\"d\":{d},\"ms\":{:.4},\"ms_min\":{:.4},\"ms_max\":{:.4},\"hbm_bytes\":{},\"kernels\":{},\"max_rel_err\":{:.3e},\"correct\":{},\"tol\":{TOL:e}}}",
                r.ms, r.lo, r.hi, r.hbm, r.kernels, r.err, r.err < TOL
            );
        };
        for (name, s) in MemorySchedule::named() {
            let pl = g.place(&outs, &s);
            let (p, _) = Program::tuned_placed(&rt, &pl.graph, &pl.outputs, &ins, pl.nest);
            if let Ok(dir) = std::env::var("FIG_DUMP") {
                for (i, k) in p.program.kernels.iter().enumerate() {
                    let f = format!("{dir}/{name}_{op}_k{i:02}.cu");
                    std::fs::write(f, &k.source).unwrap();
                }
                for t in pl.graph.placements(&pl.outputs) {
                    eprintln!("  {name} placement {t:?}");
                }
            }
            let r = measure(&rt, &p, &pl.outputs, &want, iters, name);
            emit(name, &r);
        }
        // flash with no tensor cores (every kernel transpiled, f32 arithmetic): separates
        // the schedule from the TF32 rounding of the mma tiles
        let pl = g.place(&outs, &MemorySchedule::flash());
        let p = Program::new(&rt, pl.graph.to_cuda_placed(&pl.outputs, pl.nest, None), &ins);
        let r = measure(&rt, &p, &pl.outputs, &want, iters, "flash_fp32");
        emit("flash_fp32", &r);
        // searched: every role-level placement, each with its per-kernel lowering, then
        // every kernel's thread mappings on the winner. Only correct candidates can win.
        let mut best: Option<(f64, TensorGraph, Vec<TensorId>, bool, MemorySchedule)> = None;
        for s in MemorySchedule::candidates() {
            let pl = g.place(&outs, &s);
            let (p, _) = Program::tuned_placed(&rt, &pl.graph, &pl.outputs, &ins, pl.nest);
            p.run(&rt);
            rt.sync();
            let got: Vec<Vec<f32>> = pl.outputs.iter().map(|o| p.read(&rt, *o)).collect();
            let err = max_rel(&got, &want);
            let ms = (0..3).map(|_| p.time(&rt, iters)).fold(f64::INFINITY, f64::min);
            eprintln!("candidate {s:?}: {ms:.3} ms, err {err:.2e}");
            if err < TOL && best.as_ref().is_none_or(|b| ms < b.0) {
                best = Some((ms, pl.graph, pl.outputs, pl.nest, s));
            }
        }
        let (mut bms, mut bg, bo, nest, bs) = best.expect("no candidate correct");
        let nodes: Vec<TensorId> = bg
            .to_cuda_placed(&bo, nest, Some(0))
            .kernels
            .iter()
            .map(|k| k.node)
            .collect();
        for t in nodes {
            if t.index() >= bg.nodes().len() {
                continue;
            }
            for (split, tile, reg) in bg.schedule_candidates(t) {
                let mut c = bg.clone();
                c.set_schedule(t, split, tile, reg);
                let (p, _) = Program::tuned_placed(&rt, &c, &bo, &ins, nest);
                p.run(&rt);
                rt.sync();
                let got: Vec<Vec<f32>> = bo.iter().map(|o| p.read(&rt, *o)).collect();
                let ms = (0..3).map(|_| p.time(&rt, iters)).fold(f64::INFINITY, f64::min);
                if max_rel(&got, &want) < TOL && ms < bms {
                    bms = ms;
                    bg = c;
                }
            }
        }
        eprintln!("searched {op}: {bs:?}");
        let (p, _) = Program::tuned_placed(&rt, &bg, &bo, &ins, nest);
        let r = measure(&rt, &p, &bo, &want, iters, "searched");
        emit("searched", &r);
    }
}
