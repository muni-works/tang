//! Whole-model compilation spike: `layers` SmolLM2-135M-like blocks, an
//! MSE loss, the full backward and AdamW on every projection weight as
//! one `TensorGraph`, one program, one submission per step.
//!
//! Reports kernels per step, cross-layer fusion (stacked forward vs
//! `layers ×` one block), memory (one buffer per node vs liveness-planned
//! slots vs planned + in-place updates) and step time.
//!
//! Usage: `whole_model [layers] [seq] [wgpu|cuda|stats]`. `stats` builds
//! and plans without a GPU. `cuda` needs `--features cuda`.
use std::time::Instant;

use tang_bench::ir_block::{compiled_block, BlockDims};
use tang_bench::ir_model::{
    compile, compile_with_barriers, donations, program_stats, train_step, AdamW, TrainStep,
};
use tang_compile::tensor::memplan::MemoryPlan;
use tang_compile::tensor::wgsl::WgslProgram;
#[cfg(feature = "cuda")]
use tang_compile::tensor::Format;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_gpu::program::{Program, Runtime};

/// FNV-1a over the f32 bits of every output, to compare runs bitwise.
fn bits_hash(outs: impl Iterator<Item = Vec<f32>>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in outs {
        for x in v {
            for b in x.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

fn mb(b: usize) -> f64 {
    b as f64 / (1 << 20) as f64
}

fn dims(seq: usize) -> BlockDims {
    BlockDims {
        seq,
        kv_len: seq,
        rope: true,
        bias: false,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    }
}

/// Forward kernels: `layers` stacked blocks vs one block.
fn fusion_report(seq: usize, layers: usize) {
    let d = dims(seq);
    let (g1, y1) = compiled_block(&d);
    let k1 = g1.to_wgsl(&[y1]).kernels.len();
    let step = train_step(&d, layers, AdamW::default());
    let h = step.hidden;
    let (gs, ys) = compile(&step.graph, &[h]);
    let ks = gs.to_wgsl(&ys).kernels.len();
    let (gb, yb) = compile_with_barriers(&step.graph, &[h], &step.boundaries);
    let kb = gb.to_wgsl(&yb).kernels.len();
    println!("forward kernels with block outputs materialized: {kb}");
    println!(
        "forward kernels: one block {k1}, {layers} stacked {ks} (vs {} = {layers} × {k1}): {} saved across block boundaries",
        layers * k1,
        (layers * k1) as i64 - ks as i64
    );
}

struct Built {
    step: TrainStep,
    g: TensorGraph,
    outs: Vec<TensorId>,
    prog: WgslProgram,
    naive: MemoryPlan,
    planned: MemoryPlan,
    inplace: MemoryPlan,
}

fn build(seq: usize, layers: usize, cuda: bool) -> Built {
    let t = Instant::now();
    let step = train_step(&dims(seq), layers, AdamW::default());
    let outs = step.outputs();
    let t_graph = t.elapsed();
    let t = Instant::now();
    let barriers = if std::env::var("NOBARRIER").is_ok() {
        vec![]
    } else {
        step.boundaries.clone()
    };
    let (mut g, couts) = compile_with_barriers(&step.graph, &outs, &barriers);
    // TANG_PRECISION=<file>: a precision schedule from `prec-tune`
    if let Ok(path) = std::env::var("TANG_PRECISION") {
        let text = std::fs::read_to_string(&path).expect("precision file");
        tang_bench::ir_precision::read_assignment(&mut g, &text);
        println!("  precision: {} buffers from {path}", text.lines().count());
    }
    // TANG_PRECISION_TILES=f16|bf16: the precision search's first move
    // alone — every tensor-core tile's operands in half, graph inputs
    // (weights, AdamW state, data) kept f32
    if let Ok(f) = std::env::var("TANG_PRECISION_TILES") {
        use tang_bench::ir_precision;
        let f = match f.as_str() {
            "bf16" => tang_compile::tensor::Format::BF16,
            _ => tang_compile::tensor::Format::F16,
        };
        let inputs: Vec<TensorId> = g
            .to_cuda(&couts)
            .buffers
            .iter()
            .filter(|b| b.input.is_some())
            .map(|b| b.node)
            .collect();
        if let Some(m) = ir_precision::moves(&g, &couts, &inputs)
            .into_iter()
            .find(|m| m.what.starts_with("all"))
        {
            println!(
                "  precision: {} ({} buffers) in {f:?}",
                m.what,
                m.buffers.len()
            );
            g = ir_precision::apply(&g, &m, f);
        }
    }
    let t_passes = t.elapsed();
    let t = Instant::now();
    let prog = if cuda {
        #[cfg(feature = "cuda")]
        {
            g.to_cuda_with(&couts, true)
        }
        #[cfg(not(feature = "cuda"))]
        {
            unreachable!()
        }
    } else {
        g.to_wgsl(&couts)
    };
    let t_lower = t.elapsed();
    let t = Instant::now();
    let naive = MemoryPlan::naive(&prog);
    let planned = g.plan_memory(&prog, &couts, &[]);
    let inplace = g.plan_memory(&prog, &couts, &donations(&g, &step, &couts));
    let t_plan = t.elapsed();
    let s = program_stats(&prog);
    if let Some(k) = prog.kernels.iter().max_by_key(|k| k.reads.len()) {
        let c = g.comp_at(k.node.index()).unwrap();
        println!(
            "  widest kernel: node {:?} reads {} buffers (shape {:?}, red {:?}, {} args)",
            k.node,
            k.reads.len(),
            g.shape(k.node),
            c.red,
            c.args.len()
        );
        if std::env::var("WIDE").is_ok() {
            for a in &c.args {
                let s = a.src.index();
                eprintln!(
                    "    arg {:?} shape {:?} red {:?} readers-of-it {} forms {:?}",
                    a.src,
                    g.shape(a.src),
                    g.comp_at(s).map(|c| c.red.clone()),
                    prog.kernels
                        .iter()
                        .filter(|k| k.reads.contains(&a.src))
                        .count(),
                    a.view
                        .forms
                        .iter()
                        .map(|f| (f.coefs.clone(), f.offset))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    println!(
        "layers {layers} seq {seq}: {} graph nodes -> {} compiled; {} kernels ({} distinct sources, {} up to renaming), {} buffers",
        step.graph.nodes().len(),
        g.nodes().len(),
        s.kernels,
        s.distinct_sources,
        s.unique_kernels,
        s.buffers
    );
    println!(
        "  build {:.2?} passes {:.2?} lower {:.2?} plan {:.2?}",
        t_graph, t_passes, t_lower, t_plan
    );
    let p = naive.stats.persistent_bytes;
    println!(
        "  memory: persistent {:.1} MB (params, m, v, data); intermediates naive {:.1} MB, planned {:.1} MB ({} slots), planned+in-place {:.1} MB ({} in-place updates, live peak {:.1} MB)",
        mb(p),
        mb(naive.stats.planned_bytes),
        mb(planned.stats.planned_bytes),
        planned.slot_words.len() - planned.slot_input.iter().filter(|s| s.is_some()).count(),
        mb(inplace.stats.planned_bytes),
        inplace.inplace.len(),
        mb(planned.stats.live_peak_bytes),
    );
    println!(
        "  peak total: naive {:.1} MB, planned {:.1} MB, planned+in-place {:.1} MB",
        mb(naive.total_bytes()),
        mb(planned.total_bytes()),
        mb(inplace.total_bytes()),
    );
    Built {
        step,
        g,
        outs: couts,
        prog,
        naive,
        planned,
        inplace,
    }
}

fn clone_prog(p: &WgslProgram) -> WgslProgram {
    use tang_compile::tensor::wgsl::{BufferSpec, WgslKernel};
    WgslProgram {
        kernels: p
            .kernels
            .iter()
            .map(|k| WgslKernel {
                source: k.source.clone(),
                node: k.node,
                reads: k.reads.clone(),
                points: k.points,
                count: k.count,
                workgroup_size: k.workgroup_size,
                writes: Vec::new(),
            })
            .collect(),
        buffers: p
            .buffers
            .iter()
            .map(|b| BufferSpec {
                node: b.node,
                len: b.len,
                format: b.format,
                input: b.input,
            })
            .collect(),
        aliases: p.aliases.clone(),
    }
}

fn planned(rt: &Runtime, prog: &WgslProgram, plan: &MemoryPlan, inputs: &[Vec<f32>]) -> Program {
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    Program::planned(rt, clone_prog(prog), &refs, plan)
}

fn run_wgpu(b: &Built, layers: usize, seq: usize) {
    let rt = Runtime::wgpu().expect("gpu");
    let time = |exe: &Program| {
        let mut best = f64::INFINITY;
        exe.run(&rt);
        let _ = exe.read(&rt, b.outs[0]);
        for _ in 0..3 {
            let t = Instant::now();
            for _ in 0..3 {
                exe.run(&rt);
            }
            let _ = exe.read(&rt, b.outs[0]);
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / 3.0);
        }
        best
    };
    let t = Instant::now();
    let naive = planned(&rt, &b.prog, &b.naive, &b.step.inputs);
    naive.run(&rt);
    let loss_n = naive.read(&rt, b.outs[0]);
    let p_n = naive.read(&rt, b.outs[1]);
    println!("  wgpu first run (shader compiles) {:.2?}", t.elapsed());
    let planned_exe = planned(&rt, &b.prog, &b.planned, &b.step.inputs);
    planned_exe.run(&rt);
    let loss_p = planned_exe.read(&rt, b.outs[0]);
    println!(
        "  output bits hash {:016x}",
        bits_hash(b.outs.iter().map(|&t| planned_exe.read(&rt, t)))
    );
    let p_p = planned_exe.read(&rt, b.outs[1]);
    let dp = p_n
        .iter()
        .zip(&p_p)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  loss naive {} planned {} (max |Δp'| {dp:e})",
        loss_n[0], loss_p[0]
    );
    if std::env::var("CPU").is_ok() {
        let inputs: Vec<Vec<f64>> = b
            .step
            .inputs
            .iter()
            .map(|v| v.iter().map(|&x| x as f64).collect())
            .collect();
        let want = b.g.eval(&inputs, &b.outs[..2]);
        let dp = want[1]
            .iter()
            .zip(&p_p)
            .map(|(a, b)| (a - *b as f64).abs())
            .fold(0.0, f64::max);
        println!("  cpu loss {} (max |Δp'| gpu vs cpu {dp:e})", want[0][0]);
    }
    let tn = time(&naive);
    let tp = time(&planned_exe);
    drop(naive);
    drop(planned_exe);
    // one block's step, timed alone
    let one = build(seq, 1, false);
    let e1 = planned(&rt, &one.prog, &one.planned, &one.step.inputs);
    let mut best = f64::INFINITY;
    e1.run(&rt);
    let _ = e1.read(&rt, one.outs[0]);
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..5 {
            e1.run(&rt);
        }
        let _ = e1.read(&rt, one.outs[0]);
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / 5.0);
    }
    println!(
        "  wgpu step: naive {tn:.2} ms, planned {tp:.2} ms; one block's step {best:.2} ms × {layers} = {:.2} ms",
        best * layers as f64
    );
}

#[cfg(feature = "cuda")]
fn run_cuda(b: &Built, layers: usize, seq: usize) {
    // own stream, event tracking off: what CUDA Graph capture needs
    let rt = Runtime::cuda_capturable().expect("cuda");
    // TANG_TUNE: every kernel at its fastest lowering (`Program::tuned`:
    // tensor-core tile shapes against transpiled WGSL). The kernels and
    // buffers are the same in every variant, so the memory plans hold.
    let tuned;
    let prog = if std::env::var_os("TANG_TUNE").is_some() {
        let refs: Vec<&[f32]> = b.step.inputs.iter().map(|v| v.as_slice()).collect();
        let tag = std::env::var("TANG_PRECISION_TILES").unwrap_or_else(|_| "f32".into());
        let path = format!(
            "{}/target/tang-tuning-whole-model-{layers}-{seq}-{tag}.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let (p, rep) = Program::tuned(&rt, &b.g, &b.outs, &refs, Some(std::path::Path::new(&path)));
        println!(
            "  tuned: {} of {} native kernels kept{}",
            rep.native_kept,
            rep.native_total,
            if rep.from_cache { " (cached)" } else { "" }
        );
        tuned = p.program;
        &tuned
    } else {
        &b.prog
    };
    let t = Instant::now();
    let mut naive = planned(&rt, prog, &b.naive, &b.step.inputs);
    println!(
        "  cuda: {} kernels, {} distinct sources compiled in {:.2?}",
        b.prog.kernels.len(),
        program_stats(prog).distinct_sources,
        t.elapsed()
    );
    naive.run(&rt);
    println!(
        "  naive output bits hash {:016x}",
        bits_hash(b.outs.iter().map(|&t| naive.read(&rt, t)))
    );
    let loss_n = naive.read(&rt, b.outs[0]);
    let p_n = naive.read(&rt, b.outs[1]);
    let tn = naive.time(&rt, 5);
    naive.capture(&rt);
    let tng = naive.time(&rt, 5);
    drop(naive);
    let mut inplace = planned(&rt, prog, &b.inplace, &b.step.inputs);
    inplace.run(&rt);
    let loss_i = inplace.read(&rt, b.outs[0]);
    println!(
        "  output bits hash {:016x}",
        bits_hash(b.outs.iter().map(|&t| inplace.read(&rt, t)))
    );
    let p_i = inplace.read(&rt, b.outs[1]);
    // TANG_REF=<file>: error of every output against the f64 reference
    // (from `prec-ref`)
    if let Ok(path) = std::env::var("TANG_REF") {
        let want = tang_bench::ir_precision::read_f64s(&path);
        let got: Vec<Vec<f64>> = b
            .outs
            .iter()
            .map(|&t| inplace.read(&rt, t).iter().map(|&x| x as f64).collect())
            .collect();
        let e = tang_compile::tensor::ErrorStats::compare(&got, &want);
        let loss = tang_compile::tensor::ErrorStats::compare(&got[..1], &want[..1]);
        println!(
            "  error vs f64: max rel {:.2e} (mean {:.2e}) over all outputs; loss rel {:.2e}",
            e.max_rel, e.mean_rel, loss.max_rel
        );
    }
    {
        // the first replay of a captured step from fresh state
        let mut fresh = planned(&rt, prog, &b.inplace, &b.step.inputs);
        fresh.capture(&rt);
        fresh.run(&rt);
        println!(
            "  graph output bits hash {:016x}",
            bits_hash(b.outs.iter().map(|&t| fresh.read(&rt, t)))
        );
    }
    let dp = p_n
        .iter()
        .zip(&p_i)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  loss naive {} planned+in-place {} (max |Δp'| {dp:e})",
        loss_n[0], loss_i[0]
    );
    let ti = inplace.time(&rt, 5);
    inplace.capture(&rt);
    let tig = inplace.time(&rt, 5);
    drop(inplace);
    let one = build(seq, 1, true);
    let mut e1 = planned(&rt, &one.prog, &one.inplace, &one.step.inputs);
    let t1 = e1.time(&rt, 10);
    e1.capture(&rt);
    let t1g = e1.time(&rt, 10);
    println!(
        "  cuda step: naive {tn:.2} ms (graph {tng:.2}), planned+in-place {ti:.2} ms (graph {tig:.2}); one block's step {t1:.2} ms (graph {t1g:.2}) × {layers} = {:.2} ms",
        t1 * layers as f64
    );
    let _ = &b.g;
}

/// The program's graph inputs (weights, AdamW state, data).
#[cfg(feature = "cuda")]
fn graph_inputs(p: &WgslProgram) -> Vec<TensorId> {
    p.buffers
        .iter()
        .filter(|b| b.input.is_some())
        .map(|b| b.node)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layers: usize = args.get(1).map_or(4, |s| s.parse().unwrap());
    let seq: usize = args.get(2).map_or(128, |s| s.parse().unwrap());
    let mode = args.get(3).map_or("wgpu", |s| s.as_str());
    if std::env::var("FUSION").is_ok() {
        fusion_report(seq, layers);
    }
    let b = build(seq, layers, mode.starts_with("cuda"));
    match mode {
        "wgpu" => run_wgpu(&b, layers, seq),
        // nvrtc only (no GPU): fill the PTX cache outside the GPU lock
        #[cfg(feature = "cuda")]
        "cuda-compile" => {
            let t = Instant::now();
            let sources = |p: &WgslProgram| {
                p.kernels
                    .iter()
                    .map(|k| k.source.clone())
                    .collect::<Vec<_>>()
            };
            let (mut a, c) = (sources(&b.prog), sources(&build(seq, 1, true).prog));
            // with TANG_TUNE, every lowering `Program::tuned` will time
            if std::env::var_os("TANG_TUNE").is_some() {
                for v in std::iter::once(None)
                    .chain((0..tang_compile::tensor::cuda::MMA_SHAPES.len()).map(Some))
                {
                    a.extend(sources(&b.g.to_cuda_variant(&b.outs, v)));
                }
            }
            tang_gpu::cuda::precompile(a.iter().chain(&c).map(|s| s.as_str()), "sm_86");
            println!("  nvrtc {:.2?}", t.elapsed());
        }
        #[cfg(feature = "cuda")]
        "cuda" => run_cuda(&b, layers, seq),
        // precision search, phase 1 (CPU): the f64 reference of every
        // output, and nvrtc for the search's candidates
        #[cfg(feature = "cuda")]
        "prec-ref" => {
            use tang_bench::ir_precision;
            let path = args.get(4).expect("reference path");
            let t = Instant::now();
            let inputs: Vec<Vec<f64>> = b
                .step
                .inputs
                .iter()
                .map(|v| v.iter().map(|&x| x as f64).collect())
                .collect();
            ir_precision::write_f64s(path, &b.g.eval(&inputs, &b.outs));
            println!("  reference in {:.1?}", t.elapsed());
            let t = Instant::now();
            let srcs = ir_precision::candidate_sources(
                &b.g,
                &b.outs,
                &graph_inputs(&b.prog),
                &[Format::F16, Format::BF16],
            );
            tang_gpu::cuda::precompile(srcs.iter().map(|s| s.as_str()), "sm_86");
            println!(
                "  {} candidate sources, nvrtc {:.1?}",
                srcs.len(),
                t.elapsed()
            );
        }
        // phase 2 (GPU): the search; graph inputs (weights, AdamW state,
        // data) stay f32. Writes the schedule to `<reference>.prec`
        #[cfg(feature = "cuda")]
        "prec-tune" => {
            use tang_bench::ir_precision;
            let path = args.get(4).expect("reference path");
            let want = ir_precision::read_f64s(path);
            let rt = Runtime::select().expect("gpu");
            let refs: Vec<&[f32]> = b.step.inputs.iter().map(|v| v.as_slice()).collect();
            let budget = tang_compile::tensor::ErrorBudget::rel(
                std::env::var("TANG_BUDGET")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1e-2),
            );
            let r = ir_precision::tune_precision(
                &rt,
                b.g.clone(),
                &b.outs,
                &refs,
                &want,
                budget,
                &graph_inputs(&b.prog),
                &[Format::F16, Format::BF16],
                min_gain(),
                5,
            );
            for t in &r.trials {
                println!(
                    "  {:<48} rel {:.2e} {} {}",
                    t.what,
                    t.error.max_rel,
                    t.ms.zip(t.vs)
                        .map_or("over budget".to_string(), |(m, r)| format!(
                            "{m:.3} ms ({r:.3}x current)"
                        )),
                    if t.kept { "kept" } else { "" }
                );
            }
            println!(
                "  f32 {:.3} ms (rel {:.2e}) -> {:.3} ms (rel {:.2e}), {} buffers changed",
                r.base_ms,
                r.base_error.max_rel,
                r.ms,
                r.error.max_rel,
                r.assignment.len()
            );
            std::fs::write(
                format!("{path}.prec"),
                ir_precision::write_assignment(&r.assignment),
            )
            .unwrap();
        }
        _ => {}
    }
}

/// `TANG_MIN_GAIN`: the fraction a precision move must save (default 0.01).
fn min_gain() -> f64 {
    std::env::var("TANG_MIN_GAIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01)
}
