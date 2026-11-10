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
        vec![Machine::M4]
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
