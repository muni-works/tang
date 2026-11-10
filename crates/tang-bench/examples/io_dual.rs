//! Primal bound vs dual tile vs the tile tang picks.
//!
//! For matmul and attention forward, run tang's pass pipeline, read the
//! HBL LP of the main kernel off its views (`TensorGraph::kernel_lp`),
//! solve primal and dual (`tensor::io_lp`), and compare the dual's tile
//! with (a) `schedule_auto`'s pick and (b) with `TUNE=1`, the GPU
//! autotuner's pick on this machine (`tang_gpu::autotune`, M4 via wgpu
//! here). `2-level dual` is the tile with the output held in registers
//! (the schedule space's limit, `SCHEDULE_REG_TILE`; the sweep shows other
//! capacities up to the register file) and the inputs staged in shared memory
//! (`Lp::tile_levels`). Traffic in MB at f32; `floor` is the composite time floor
//! `max(bytes/BW, flops/peak, launches×latency)` with its binding term.
//!
//!   cargo run --release -p tang-bench --example io_dual
//!   TUNE=1 cargo run --release -p tang-bench --example io_dual       # M4
//!   RACE=1 TUNE=3090 cargo run --release -p tang-bench --features cuda --example io_dual   # time every candidate
//!   TUNE=3090 cargo run --release -p tang-bench --example io_dual    # on a 3090
use tang_bench::ir_gpu::naive_attention_heads;
use tang_compile::tensor::io_lp::{Lp, Machine, TimeFloor};
use tang_compile::tensor::{TensorGraph, TensorId};

fn passes(g: TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, o) = g.stabilize_softmax(outs);
    let (g, o) = g.substitute_maps(&o);
    let (g, o) = g.eliminate_common(&o);
    let (g, o) = g.merge_siblings(&o);
    let (g, o) = g.relayout(&o);
    let (g, o) = g.merge_reduce_dims(&o);
    let (mut g, o) = g.split_reduce_dims(&o);
    g.schedule_auto(&o);
    (g, o)
}

/// Traffic of a box tiling with the best innermost coordinate: arrays
/// that do not depend on it stay resident across it.
fn tiled_traffic(lp: &Lp, tile: &[usize]) -> f64 {
    let sizes: Vec<f64> = lp
        .projs
        .iter()
        .map(|&p| {
            (0..lp.extents.len())
                .filter(|q| p & (1 << q) != 0)
                .map(|q| lp.extents[q] as f64)
                .product()
        })
        .collect();
    (0..lp.extents.len())
        .filter(|&q| tile[q] < lp.extents[q])
        .map(|inner| {
            lp.projs
                .iter()
                .zip(&sizes)
                .map(|(&p, &n)| {
                    let resident = p & (1 << inner) == 0;
                    n * (0..lp.extents.len())
                        .filter(|q| p & (1 << q) == 0 && !(resident && *q == inner))
                        .map(|q| lp.extents[q].div_ceil(tile[q].max(1)) as f64)
                        .product::<f64>()
                })
                .sum::<f64>()
        })
        .fold(lp.tiled_traffic(tile, &sizes), f64::min)
}

/// Kernel tile of tang's schedule mapped onto the LP's coordinates:
/// points from `io_report`, reduce/nested coords staged in full chunks.
fn tang_tile(
    g: &TensorGraph,
    outs: &[TensorId],
    k: TensorId,
    lp: &Lp,
    nest: bool,
) -> (Vec<usize>, f64, usize) {
    let mut g = g.clone();
    if nest {
        g.schedule_nested_contractions_scalar(outs);
    }
    let r = g.io_report(outs, 0.0);
    let kio = r
        .kernels
        .iter()
        .find(|x| x.node == k)
        .unwrap_or(&r.kernels[0]);
    let mut t: Vec<usize> = lp.extents.clone();
    for (q, &v) in kio.tile.iter().enumerate() {
        t[q] = v;
    }
    // the kernel's own reduce dims are staged in chunks of `tile.2`
    let chunk = g.nodes()[k.index()].tile.2 as usize;
    let np = kio.tile.len();
    if chunk > 0 && np < t.len() {
        t[np] = chunk.min(t[np]);
    }
    (t, kio.words(), r.kernels.len())
}

fn main_kernel(g: &TensorGraph, outs: &[TensorId]) -> TensorId {
    let r = g.io_report(outs, 0.0);
    r.kernels
        .iter()
        .max_by(|a, b| a.words().partial_cmp(&b.words()).unwrap())
        .unwrap()
        .node
}

fn tune(g: &TensorGraph, outs: &[TensorId]) -> Option<TensorGraph> {
    if std::env::var("TUNE").is_err() {
        return None;
    }
    use tang_gpu::autotune::{autotune, TuneOptions};
    use tang_gpu::program::Runtime;
    let rt = Runtime::select().ok()?;
    let prog = rt.lower(g, outs);
    let mut rng = tang_bench::Rng::new(7);
    let inputs: Vec<Vec<f32>> = (0..prog.buffers.iter().filter(|b| b.input.is_some()).count())
        .map(|k| {
            let len = prog
                .buffers
                .iter()
                .find(|b| b.input == Some(k))
                .unwrap()
                .len;
            (0..len).map(|_| rng.f32() - 0.5).collect()
        })
        .collect();
    let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let opts = TuneOptions {
        dir: None,
        ..Default::default()
    };
    Some(autotune(&rt, g.clone(), outs, &ins, &opts).0)
}

/// `RACE=1`: time every candidate of the main kernel's schedule space on
/// this GPU (whole program, best of 5×20 runs) and print the best time per
/// LP tile, so the dual tiles can be compared with what wins on hardware.
/// Launch shape of the lowered main kernel: (grid, block, regs/thread,
/// smem words). Registers are counted from the source: `REG_BASE` for
/// indices and addresses (assumed, not measured) plus every
/// function-scope `var` word (accumulator arrays count their length).
fn launch_shape(src: &str, count: u32, block: u32) -> (usize, usize, usize, f64) {
    const REG_BASE: usize = 24;
    let words = |t: &str| -> f64 {
        t.split(|c: char| !c.is_ascii_digit())
            .filter_map(|n| n.parse::<f64>().ok())
            .filter(|&n| n > 0.0 && t.contains("array"))
            .product::<f64>()
            .max(1.0)
    };
    let mut smem = 0.0;
    let mut regs = REG_BASE as f64;
    for l in src.lines().map(str::trim) {
        // CUDA lowering: dynamic smem in bytes, register arrays `float a[..][..];`
        if let Some(r) = l.strip_prefix("#define TANG_SMEM ") {
            smem += r.trim().parse::<f64>().unwrap_or(0.0) / 4.0;
        } else if l.starts_with("float ") && l.contains('[') && l.ends_with("];") {
            regs += l
                .split('[')
                .skip(1)
                .filter_map(|d| d.split(']').next()?.trim().parse::<f64>().ok())
                .product::<f64>();
        } else if let Some(r) = l.strip_prefix("var<workgroup>") {
            let ty = r.split_once(':').map(|x| x.1).unwrap_or("");
            // drop the f32/f16 width digits before multiplying
            smem += words(&ty.replace("f32", "").replace("f16", "").replace("u32", ""));
        } else if let Some(r) = l.strip_prefix("var ") {
            let ty = r.split_once(':').map(|x| x.1).unwrap_or("");
            if ty.contains("array") {
                regs += words(&ty.replace("f32", "").replace("u32", ""));
            } else {
                regs += 1.0;
            }
        }
    }
    let block = block.max(1) as usize;
    (
        (count as usize).div_ceil(block),
        block,
        (regs as usize).min(255),
        smem,
    )
}

/// Read bandwidth by working-set size, bytes/s, from the probes
/// `scripts/review/cachebw.cu` (mew 3090) and `cachebw.swift` (M4 Max);
/// independent of every kernel raced here. `(size bytes, bandwidth)`.
fn cache_table(m: &Machine) -> &'static [(f64, f64)] {
    const MB: f64 = 1048576.0;
    match m.name {
        "3090" => &[
            (1.0 * MB, 3446e9),
            (2.0 * MB, 4403e9),
            (4.0 * MB, 5734e9),
            (8.0 * MB, 4914e9),
            (32.0 * MB, 2080e9),
            (128.0 * MB, 894e9),
            (f64::INFINITY, 890e9),
        ],
        _ => &[
            (1.0 * MB, 2138e9),
            (4.0 * MB, 2213e9),
            (8.0 * MB, 2157e9),
            (16.0 * MB, 2153e9),
            (32.0 * MB, 1496e9),
            (128.0 * MB, 512e9),
            (f64::INFINITY, 498e9),
        ],
    }
}

/// Highest probed bandwidth at any size ≥ the working set: what traffic
/// that misses the workgroup's fast memory can at best be served at.
fn cache_bw(m: &Machine, ws_bytes: f64) -> f64 {
    cache_table(m)
        .iter()
        .filter(|e| e.0 >= ws_bytes)
        .map(|e| e.1)
        .fold(0.0, f64::max)
}

fn race(
    g: &TensorGraph,
    outs: &[TensorId],
    lp: &Lp,
    nest: bool,
    names: &[String],
    flops: f64,
    m: &Machine,
) {
    if std::env::var("RACE").is_err() {
        return;
    }
    use tang_compile::tensor::space::Space;
    use tang_gpu::program::Runtime;
    let Ok(rt) = Runtime::select() else {
        println!("  race: no GPU runtime");
        return;
    };
    use tang_gpu::program::Program;
    let base = rt.lower(g, outs);
    let mut rng = tang_bench::Rng::new(7);
    let inputs: Vec<Vec<f32>> = (0..base.buffers.iter().filter(|b| b.input.is_some()).count())
        .map(|i| {
            let len = base
                .buffers
                .iter()
                .find(|b| b.input == Some(i))
                .unwrap()
                .len;
            (0..len).map(|_| rng.f32() - 0.5).collect()
        })
        .collect();
    let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let k = main_kernel(g, outs);
    let space = Space::standard();
    // (tile, measured ms, encoding, floor s, binding term)
    let mut best: Vec<(Vec<usize>, f64, String, f64, &'static str, String)> = Vec::new();
    for c in space.candidates(g, k) {
        let mut gc = g.clone();
        space.apply(&mut gc, k, &c);
        let kk = main_kernel(&gc, outs);
        let low = rt.lower(&gc, outs);
        let Some(lk) = low.kernels.iter().find(|x| x.node == kk) else {
            continue;
        };
        if std::env::var("DUMP").is_ok() {
            eprintln!("--- {}\n{}", c.encode(&space), lk.source);
        }
        let (grid, block, regs, smem) = launch_shape(&lk.source, lk.count, lk.workgroup_size);
        let Ok(prog) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Program::compile(&rt, &gc, outs, &ins)
        })) else {
            continue;
        };
        let Ok(ms) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prog.time(&rt, 20)))
        else {
            continue;
        };
        let (mut t, _, _) = tang_tile(&gc, outs, kk, lp, nest);
        // nothing staged: io_report's tile is the workgroup's point range,
        // but no operand of it is resident, so reuse is per point
        if smem == 0.0 {
            t = vec![1; t.len()];
        }
        if std::env::var("DBG").is_ok() {
            eprintln!("{} -> {:?} {}", c.encode(&space), t, tiled_traffic(lp, &t));
        }
        let sk = tang_compile::tensor::io_lp::StepKernel {
            bytes: tiled_traffic(lp, &t) * 4.0,
            flops,
            grid,
            block,
            regs,
            smem_words: smem,
            issue: Default::default(),
        };
        // LP traffic leaves shared memory and is served by L2/SLC at the
        // probed rate for this working set; DRAM sees at least the
        // compulsory bytes (every array once)
        let comp = compulsory(lp) * 4.0;
        let mc = Machine {
            dram_bw: cache_bw(m, comp),
            ..*m
        };
        let (fl, why) = sk.floor(&mc);
        let dram = m.launch + m.mem_latency + comp / m.dram_bw;
        let (fl, why) = if dram > fl { (dram, "dram") } else { (fl, why) };
        // keep every candidate: the model ranks schedules, not LP tiles
        best.push((
            t,
            ms,
            c.encode(&space),
            fl,
            why,
            format!(
                "grid {grid} block {block} regs {regs} smem {smem} waves {:.1}",
                sk.waves(m)
            ),
        ));
    }
    best.retain(|b| b.1.is_finite());
    // faster than its floor: either a wrong kernel (RACE does not check
    // outputs) or the floor is not a floor; kept and counted, not dropped
    let viol: Vec<String> = best
        .iter()
        .filter(|b| b.1 * 1e-3 < b.3)
        .map(|b| format!("{:.3}<{:.3} ms ({})", b.1, b.3 * 1e3, b.2))
        .collect();
    println!("  floor violations: {}/{}", viol.len(), best.len());
    for v in viol.iter().take(4) {
        println!("    {v}");
    }
    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    // the model's pick: argmin floor, ties (within 0.1%) to lower traffic
    let pick = (0..best.len()).min_by(|&i, &j| {
        let (a, b) = (&best[i], &best[j]);
        if (a.3 - b.3).abs() <= 1e-3 * a.3.min(b.3) {
            tiled_traffic(lp, &a.0)
                .partial_cmp(&tiled_traffic(lp, &b.0))
                .unwrap()
        } else {
            a.3.partial_cmp(&b.3).unwrap()
        }
    });
    if let Some(p) = pick {
        let b = &best[p];
        println!(
            "  time-floor pick: rank {}/{}  {:.3} ms = {:.3}× winner  floor {:.3} ms ({})  ({})",
            p + 1,
            best.len(),
            b.1,
            b.1 / best[0].1,
            b.3 * 1e3,
            b.4,
            b.2
        );
        println!("    pick: {}", b.5);
    }
    for (t, ms, enc, fl, why, occ) in best.iter().take(12) {
        println!(
            "  race [{}]  {:.3} ms  floor {:.3} ms ({why})  LP traffic {:.2} MB  [{occ}]  ({enc})",
            names
                .iter()
                .zip(t)
                .map(|(n, v)| format!("{n}={v}"))
                .collect::<Vec<_>>()
                .join(" "),
            ms,
            fl * 1e3,
            mb(tiled_traffic(lp, t))
        );
    }
}

fn compulsory(lp: &Lp) -> f64 {
    lp.projs
        .iter()
        .map(|&p| {
            (0..lp.extents.len())
                .filter(|q| p & (1 << q) != 0)
                .map(|q| lp.extents[q] as f64)
                .product::<f64>()
        })
        .sum()
}

fn mb(w: f64) -> f64 {
    w * 4.0 / 1e6
}

fn report(name: &str, g: &TensorGraph, outs: &[TensorId], nest: bool, flops: f64, m: &Machine) {
    let k = main_kernel(g, outs);
    let Some((lp, names, cov)) = g.kernel_lp(k) else {
        println!("{name}: no LP");
        return;
    };
    let mem = m.mem_words;
    let Some(s) = lp.solve(2.0 * mem) else {
        println!("## {name}: LP infeasible: {lp:?} {names:?}");
        return;
    };
    let bound = lp.bound(mem).unwrap();
    let dual = lp.tile(mem).unwrap();
    let dual_traffic = tiled_traffic(&lp, &dual);
    use tang_compile::tensor::io_lp::SCHEDULE_REG_TILE;
    let levels = lp.tile_levels(SCHEDULE_REG_TILE, mem);
    for rw in [
        1024.0,
        2048.0,
        4096.0,
        8192.0,
        16384.0,
        32768.0,
        m.reg_words,
    ] {
        if let Some(l) = lp.tile_levels(rw, mem) {
            println!(
                "  regs {:>6} words{}: [{}] {:.2} MB",
                rw,
                if rw == m.reg_words {
                    " (register file)"
                } else {
                    ""
                },
                names
                    .iter()
                    .zip(&l.tile)
                    .map(|(n, v)| format!("{n}={v}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                l.traffic * 4.0 / 1e6
            );
        }
    }
    let (auto_t, auto_staged, launches) = tang_tile(g, outs, k, &lp, nest);
    let fmt = |t: &[usize]| {
        names
            .iter()
            .zip(t)
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!("## {name} on {} (M = {mem} words)", m.name);
    println!(
        "  LP: coords [{}], arrays {:?}, atomic {:#b}; skipped: {} gathered, {} non-projective",
        names.join(","),
        lp.projs
            .iter()
            .map(|p| format!("{p:#b}"))
            .collect::<Vec<_>>(),
        lp.atomic,
        cov.gathered.len(),
        cov.non_projective.len()
    );
    println!(
        "  primal s={:?} t={:?}  dual x={:?}  σ={:.4}  gap={:.1e}",
        s.s.iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        s.t.iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        s.x.iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        s.sigma,
        s.gap
    );
    println!("  bound {:.2} MB", mb(bound));
    println!(
        "  dual tile    [{}]  traffic {:.2} MB ({:.2}× bound)",
        fmt(&dual),
        mb(dual_traffic),
        dual_traffic / bound.max(1.0)
    );
    if let Some(lt) = &levels {
        println!(
            "  2-level dual [{}]  traffic {:.2} MB ({:.2}× bound at regs+smem {:.2} MB); in registers: {}",
            fmt(&lt.tile),
            mb(lt.traffic),
            lt.traffic / lp.bound(SCHEDULE_REG_TILE + mem).unwrap().max(1.0),
            mb(lp.bound(SCHEDULE_REG_TILE + mem).unwrap()),
            (0..lp.projs.len())
                .filter(|j| lt.in_regs & (1 << j) != 0)
                .map(|j| format!("{:#b}", lp.projs[j]))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!(
        "  schedule_auto[{}]  LP traffic {:.2} MB, staged {:.2} MB ({:.2}× bound), {launches} kernel(s)",
        fmt(&auto_t),
        mb(tiled_traffic(&lp, &auto_t)),
        mb(auto_staged),
        auto_staged / bound.max(1.0)
    );
    if let Some(tg) = tune(g, outs) {
        let kk = main_kernel(&tg, outs);
        let (tt, ts, tl) = tang_tile(&tg, outs, kk, &lp, nest);
        println!(
            "  autotuned    [{}]  LP traffic {:.2} MB, staged {:.2} MB ({:.2}× bound), {tl} kernel(s)",
            fmt(&tt),
            mb(tiled_traffic(&lp, &tt)),
            mb(ts),
            ts / bound.max(1.0)
        );
    }
    race(g, outs, &lp, nest, &names, flops, m);
    let at_bound = TimeFloor::new(m, bound * 4.0, flops, 1);
    let at_auto = TimeFloor::new(m, auto_staged * 4.0, flops, launches);
    println!(
        "  floor at bound {:.3} ms ({}); at schedule_auto's traffic {:.3} ms ({})",
        at_bound.seconds() * 1e3,
        at_bound.binding(),
        at_auto.seconds() * 1e3,
        at_auto.binding()
    );
}

fn main() {
    let machines = if std::env::var("TUNE").is_ok_and(|v| v == "3090") {
        vec![Machine::RTX3090]
    } else if std::env::var("TUNE").is_ok() {
        vec![Machine::M4_MAX]
    } else {
        vec![Machine::RTX3090, Machine::M4]
    };
    for m in &machines {
        for &n in &[1024usize, 4096] {
            let mut g = TensorGraph::new();
            let a = g.input(&[n, n]);
            let b = g.input(&[n, n]);
            let c = g.matmul(a, b).unwrap();
            let (g, o) = passes(g, &[c]);
            report(
                &format!("matmul {n}³"),
                &g,
                &o,
                false,
                2.0 * (n * n * n) as f64,
                m,
            );
        }
        for &n in &[1024usize, 4096] {
            let (g, y) = naive_attention_heads(8, 1, n, 64, 64, true);
            let (g, o) = passes(g, &[y]);
            let flops = 8.0 * 4.0 * (n * n * 64) as f64 / 2.0;
            report(
                &format!("attention fwd 8 heads N={n} d=64 causal"),
                &g,
                &o,
                true,
                flops,
                m,
            );
        }
    }
}
