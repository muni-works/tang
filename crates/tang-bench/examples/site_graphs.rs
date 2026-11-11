//! Dump the transformer block's kernel graph for the website: the block as
//! written (one kernel per comp) and after the compiler pipeline, for a
//! prefill shape and a Qwen3-8B decode shape. Writes JSON to stdout.
//!
//! cargo run --release -p tang-bench --example site_graphs > site/graphs.json
use std::collections::HashMap;

use tang_bench::ir_block::{compiled_block, transformer_block, BlockDims, INPUTS};
use tang_compile::tensor::{MonoidKind, Op, TensorGraph, TensorId};

fn label(g: &TensorGraph, t: TensorId) -> String {
    match &g.nodes()[t.index()].op {
        Op::Comp(c) => match c.monoid.as_ref().map(|m| m.kind()) {
            Some(MonoidKind::OnlineSoftmax) => "flash attention".into(),
            Some(MonoidKind::Max) => "max".into(),
            Some(MonoidKind::Sum) if c.args.len() >= 2 => "matmul".into(),
            Some(MonoidKind::Sum) => "sum".into(),
            Some(MonoidKind::Other) => "reduce".into(),
            None => "map".into(),
        },
        Op::Input(i) => INPUTS.get(*i).copied().unwrap_or("cache").into(),
        Op::Iota(_) => "iota".into(),
        Op::Reshape(_) => "reshape".into(),
        Op::Operator(_) => "operator".into(),
        Op::Scan(_) => "scan".into(),
    }
}

/// Buffers a kernel reads, looking through inlined comps and reshapes;
/// also counts the comps fused into it.
fn rank(l: &str) -> u8 {
    match l {
        "flash attention" => 5,
        "matmul" => 4,
        "reduce" => 3,
        "max" | "sum" => 2,
        _ => 1,
    }
}

fn reads(
    g: &TensorGraph,
    t: TensorId,
    top: bool,
    out: &mut Vec<TensorId>,
    fused: &mut usize,
    best: &mut String,
) {
    let node = &g.nodes()[t.index()];
    match &node.op {
        Op::Comp(c) if top || node.inline => {
            if !top {
                *fused += 1;
            }
            let l = label(g, t);
            if rank(&l) > rank(best) {
                *best = l;
            }
            for a in &c.args {
                reads(g, a.src, false, out, fused, best);
            }
        }
        Op::Reshape(src) => reads(g, *src, false, out, fused, best),
        Op::Iota(_) => {}
        _ => {
            if !out.contains(&t) {
                out.push(t);
            }
        }
    }
}

fn dump(g: &TensorGraph, y: TensorId) -> String {
    let prog = g.to_wgsl(&[y]);
    let kernel_ids: Vec<TensorId> = prog.kernels.iter().map(|k| k.node).collect();
    let mut ids: HashMap<TensorId, usize> = HashMap::new();
    let mut nodes = Vec::new();
    let mut emit = |t: TensorId, kind: &str, extra: String, nodes: &mut Vec<String>| {
        let i = nodes.len();
        let shape: Vec<String> = g.shape(t).iter().map(|d| d.n().to_string()).collect();
        nodes.push(format!(
            r#"{{"label":"{}","kind":"{}","shape":"{}"{}}}"#,
            label(g, t),
            kind,
            shape.join("×"),
            extra
        ));
        i
    };
    let mut edges_of = Vec::new();
    for &k in &kernel_ids {
        let mut r = Vec::new();
        let mut fused = 0;
        let mut best = String::from("map");
        reads(g, k, true, &mut r, &mut fused, &mut best);
        edges_of.push((k, r, fused, best));
    }
    for (_, r, _, _) in &edges_of {
        for &s in r {
            if !ids.contains_key(&s) && !kernel_ids.contains(&s) {
                let i = emit(s, "input", String::new(), &mut nodes);
                ids.insert(s, i);
            }
        }
    }
    for (k, r, fused, best) in &edges_of {
        let args: Vec<String> = r
            .iter()
            .filter_map(|s| ids.get(s))
            .map(|i| i.to_string())
            .collect();
        let extra = format!(
            r#","op":"{}","fused":{},"args":[{}]"#,
            best,
            fused,
            args.join(",")
        );
        let i = emit(*k, "kernel", extra, &mut nodes);
        ids.insert(*k, i);
    }
    format!(
        r#"{{"kernels":{},"nodes":[{}],"output":{}}}"#,
        kernel_ids.len(),
        nodes.join(","),
        ids[&y]
    )
}

fn main() {
    let prefill = BlockDims {
        seq: 128,
        kv_len: 128,
        rope: true,
        bias: true,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    };
    let decode = BlockDims {
        seq: 1,
        kv_len: 4096,
        rope: true,
        bias: false,
        d_model: 4096,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        ff_dim: 12288,
    };
    let (gw, yw) = transformer_block(&prefill);
    let (gc, yc) = compiled_block(&prefill);
    let (gd, yd) = compiled_block(&decode);
    println!(
        r#"{{"block_written":{},"block_compiled":{},"decode_compiled":{}}}"#,
        dump(&gw, yw),
        dump(&gc, yc),
        dump(&gd, yd)
    );
}
