//! Derived flash attention, forward and backward, on CUDA in bf16: the
//! attention written naively in tensor IR (`naive_attention_heads`), the
//! pass pipeline, `backward` for the gradients, and `Program::tuned`
//! picking a lowering per kernel (on Hopper the `SM90_FLASH_SHAPES`
//! wgmma variants join the `mma.sync` ones). Inputs and the reference
//! come from `scripts/hopper_attention.py gen`, so FlashAttention-3,
//! PyTorch SDPA and tang run on the same tensors.
//!
//!   hopper_attention DIR
//!
//! DIR holds `meta.txt` (`bh n d causal ref_heads...`), `q.bin`, `k.bin`,
//! `v.bin`, `do.bin` (bf16, `[bh, n, d]`) and the f64 reference for the
//! listed heads as f32: `o_ref.bin`, `dq_ref.bin`, `dk_ref.bin`,
//! `dv_ref.bin` (`[ref_heads, n, d]`). Prints one line per direction:
//! time (best of `ATT_SAMPLES` program runs of 20), TFLOP/s, and the
//! relative error on the reference heads. `TANG_VARIANT=v` forces one
//! lowering variant instead of tuning (debugging).
use tang_bench::ir_gpu::naive_attention_heads;
use tang_compile::tensor::{Format, Monoid, Precision, TensorGraph, TensorId};
use tang_gpu::program::{Program, Runtime};

fn bf16_file(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn f32_file(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// max |got − want| / max |want| over the reference heads.
fn rel_err(got: &[f32], want: &[f32], heads: &[usize], per_head: usize) -> f64 {
    let (mut e, mut m) = (0f64, 0f64);
    for (r, &h) in heads.iter().enumerate() {
        for i in 0..per_head {
            let w = want[r * per_head + i] as f64;
            let g = got[h * per_head + i] as f64;
            e = e.max((g - w).abs());
            m = m.max(w.abs());
        }
    }
    e / m
}

fn passes(g: TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, outs) = g.stabilize_softmax(outs);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    tile_reductions(&mut g);
    (g, outs)
}

fn bf16(g: &mut TensorGraph, ts: &[TensorId]) {
    for &t in ts {
        g.set_precision(t, Precision::storage(Format::BF16));
    }
}

fn build(rt: &Runtime, g: &TensorGraph, outs: &[TensorId], inputs: &[&[f32]]) -> Program {
    match std::env::var("TANG_VARIANT")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(v) => {
            let p = g.to_cuda_variant(outs, Some(v));
            if std::env::var_os("TANG_DUMP").is_some() {
                for k in &p.kernels {
                    eprintln!("// ---- kernel {}\n{}", k.node.index(), k.source);
                }
            }
            Program::new(rt, p, inputs)
        }
        None => {
            let (p, rep) = Program::tuned(rt, g, outs, inputs, None);
            eprintln!(
                "  tuned: {} of {} native kernels kept",
                rep.native_kept, rep.native_total
            );
            p
        }
    }
}

fn best_ms(rt: &Runtime, p: &Program) -> f64 {
    if std::env::var_os("TANG_BREAKDOWN").is_some() {
        for (i, k) in p.program.kernels.iter().enumerate() {
            let head = k.source.lines().next().unwrap_or("");
            eprintln!("  kernel {i}: {:.4} ms  {head}", p.time_kernel(rt, i, 10));
        }
    }
    let samples: usize = std::env::var("ATT_SAMPLES").map_or(5, |s| s.parse().unwrap());
    p.run(rt);
    rt.sync();
    (0..samples)
        .map(|_| p.time(rt, 20))
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("DIR"));
    let meta = std::fs::read_to_string(dir.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let (bh, n, d, causal) = (m[0], m[1], m[2], m[3] != 0);
    let heads = &m[4..];
    let per_head = n * d;
    let rt = Runtime::cuda().expect("cuda");
    let q = bf16_file(&dir.join("q.bin"));
    let k = bf16_file(&dir.join("k.bin"));
    let v = bf16_file(&dir.join("v.bin"));
    let d_o = bf16_file(&dir.join("do.bin"));
    // causal attention does half the work
    let frac = if causal { 0.5 } else { 1.0 };
    let fwd_flop = 4.0 * (bh * n * n * d) as f64 * frac;
    let only = std::env::var("ATT_ONLY").unwrap_or_default();

    if only != "bwd" {
        let (g, y) = naive_attention_heads(bh, 1, n, d, d, causal);
        let (mut g, outs) = passes(g, &[y]);
        let ins: Vec<TensorId> = (0..3).map(TensorId::from_index).collect();
        bf16(&mut g, &ins);
        bf16(&mut g, &outs);
        let p = build(&rt, &g, &outs, &[&q, &k, &v]);
        let ms = best_ms(&rt, &p);
        let o = p.read(&rt, outs[0]);
        let err = rel_err(&o, &f32_file(&dir.join("o_ref.bin")), heads, per_head);
        println!(
            "tang fwd bh={bh} n={n} d={d} causal={causal}: {ms:.3} ms {:.1} TFLOP/s rel_err {err:.2e} ({} kernels)",
            fwd_flop / ms / 1e9,
            p.program.kernels.len()
        );
    }
    if only != "fwd" {
        let (mut g, y) = naive_attention_heads(bh, 1, n, d, d, causal);
        let qkv: Vec<TensorId> = (0..3).map(TensorId::from_index).collect();
        let dy = g.input(&[bh, 1, n, d]);
        let prod = g.map(&[y, dy], |x| x[0] * x[1]).unwrap();
        let flat = g.reshape(prod, &[bh * n * d]).unwrap();
        let loss = g.reduce(flat, 0, Monoid::sum()).unwrap();
        let grads = g.backward(loss, &qkv).unwrap();
        let (mut g, outs) = passes(g, &grads);
        let mut ins = qkv.clone();
        ins.push(dy);
        bf16(&mut g, &ins);
        bf16(&mut g, &outs);
        let p = build(&rt, &g, &outs, &[&q, &k, &v, &d_o]);
        let ms = best_ms(&rt, &p);
        let names = ["dq", "dk", "dv"];
        let errs: Vec<String> = outs
            .iter()
            .zip(names)
            .map(|(&t, nm)| {
                let got = p.read(&rt, t);
                let want = f32_file(&dir.join(format!("{nm}_ref.bin")));
                format!("{nm} {:.2e}", rel_err(&got, &want, heads, per_head))
            })
            .collect();
        // the backward's own work is 2.5× the forward's (FlashAttention's
        // convention); this program also recomputes the forward state
        println!(
            "tang fwd+bwd-program bh={bh} n={n} d={d} causal={causal}: {ms:.3} ms bwd-equiv {:.1} TFLOP/s rel_err {} ({} kernels)",
            2.5 * fwd_flop / ms / 1e9,
            errs.join(" "),
            p.program.kernels.len()
        );
    }
}

/// Every reduction `schedule_auto` left untiled (the attention backward's,
/// whose score has two readers) gets the 16×16 tile, so the CUDA lowering
/// can try its tensor-core tiles on it (`TANG_NO_TILE_ALL` skips this).
fn tile_reductions(g: &mut tang_compile::tensor::TensorGraph) {
    if std::env::var_os("TANG_NO_TILE_ALL").is_some() {
        return;
    }
    for i in 0..g.nodes().len() {
        let n = &g.nodes()[i];
        let Some(c) = g.comp_at(i) else { continue };
        let red: usize = c.red.iter().map(|r| r.n()).product();
        if !n.inline && !c.red.is_empty() && red >= 16 && n.tile.0 == 0 && n.split == 0 {
            g.set_schedule(
                tang_compile::tensor::TensorId::from_index(i),
                0,
                (16, 16, 32),
                (1, 1),
            );
        }
    }
    // a tiled epilogue that only copies its reduction into another layout
    // (`[1, h, n, d]` -> `[h, n, d]`): the reduction becomes its own tile,
    // written in its own layout, and the copy a separate map
    use tang_compile::tensor::TensorId;
    for m in 0..g.nodes().len() {
        if g.nodes()[m].tile.0 == 0 {
            continue;
        }
        let Some(r) = g.tiled_reduction(m).filter(|&r| r != m) else {
            continue;
        };
        let copy = g.comp_at(m).is_some_and(|c| {
            c.args.len() == 1
                && c.body_outs.len() == 1
                && matches!(
                    c.body.nodes_slice()[c.body_outs[0].index() as usize],
                    tang_expr::Node::Var(0)
                )
                && g.nodes()[m].shape.len() != g.nodes()[r].shape.len()
        });
        if copy {
            g.set_inline(TensorId::from_index(r), false);
            g.set_schedule(TensorId::from_index(r), 0, (16, 16, 32), (1, 1));
            g.set_schedule(TensorId::from_index(m), 0, (0, 0, 0), (1, 1));
        }
    }
}
