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
    for &t in ins.iter().chain(&outs) {
        g.set_precision(t, Precision::storage(Format::BF16));
    }
    for k in &g.to_cuda_variant(&outs, Some(v)).kernels {
        println!("// ==== kernel node {}\n{}", k.node.index(), k.source);
    }
}
