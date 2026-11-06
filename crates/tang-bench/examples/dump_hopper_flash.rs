//! Print the CUDA sources of the derived bf16 attention (forward, or with
//! `bwd` the gradient program) in lowering variant `v`, no device needed:
//!
//!   dump_hopper_flash BH N D CAUSAL V [bwd]
use tang_bench::ir_gpu::naive_attention_heads;
use tang_compile::tensor::{Format, Monoid, Precision, TensorId};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (bh, n, d, causal, v) = (p(0), p(1), p(2), p(3) != 0, p(4));
    let (mut g, y) = naive_attention_heads(bh, 1, n, d, d, causal);
    let mut ins: Vec<TensorId> = (0..3).map(TensorId::from_index).collect();
    let outs = if a.get(5).map(String::as_str) == Some("bwd") {
        let dy = g.input(&[bh, 1, n, d]);
        let prod = g.map(&[y, dy], |x| x[0] * x[1]).unwrap();
        let flat = g.reshape(prod, &[bh * n * d]).unwrap();
        let loss = g.reduce(flat, 0, Monoid::sum()).unwrap();
        let grads = g.backward(loss, &ins).unwrap();
        ins.push(dy);
        grads
    } else {
        vec![y]
    };
    let (g, outs) = g.stabilize_softmax(&outs);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    tile_reductions(&mut g);
    for &t in ins.iter().chain(&outs) {
        g.set_precision(t, Precision::storage(Format::BF16));
    }
    for k in &g.to_cuda_variant(&outs, Some(v)).kernels {
        println!("// ==== kernel node {}\n{}", k.node.index(), k.source);
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
#[allow(dead_code)]
pub fn report(g: &tang_compile::tensor::TensorGraph) {
    for i in 0..g.nodes().len() {
        if let Some(c) = g.comp_at(i) {
            let n = &g.nodes()[i];
            eprintln!(
                "node {i} shape {:?} red {:?} inline {} tile {:?} split {} args {:?}",
                n.shape,
                c.red,
                n.inline,
                n.tile,
                n.split,
                c.args.iter().map(|a| a.src.index()).collect::<Vec<_>>()
            );
        }
    }
}
