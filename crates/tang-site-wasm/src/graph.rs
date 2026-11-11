//! Live graph views for the site: trace a demo's own `S: Scalar` code with
//! `ExprId`, then serve the node list and per-node values to the page.

use core::cell::RefCell;
use std::collections::HashMap;

use tang::Scalar;
use tang_expr::{graph::ExprGraph, node::ExprId, node::Node, trace};
use wasm_bindgen::prelude::*;

struct Traced {
    graph: ExprGraph,
    outputs: Vec<ExprId>,
    live: Vec<ExprId>,
}

thread_local! {
    static TRACED: RefCell<HashMap<String, Traced>> = RefCell::new(HashMap::new());
}

fn build(kind: &str) -> Result<(ExprGraph, Vec<ExprId>), String> {
    match kind {
        "dual" => {
            let (mut g, f) = trace(|| {
                let x = ExprId::var(0);
                x * x.sin()
            });
            let d = g.diff(f, 0);
            let d = g.simplify(d);
            Ok((g, vec![f, d]))
        }
        "orbit" => {
            let (g, (ax, ay)) = trace(|| crate::gravity(ExprId::var(0), ExprId::var(1)));
            Ok((g, vec![ax, ay]))
        }
        "arm" => {
            let (g, (x, y)) = trace(|| crate::arm_tip(ExprId::var(0), ExprId::var(1)));
            Ok((g, vec![x, y]))
        }
        k => match k.strip_prefix("expr:") {
            Some(src) => crate::kernel::parse_graph(src, true).map(|(g, o, _)| (g, o)),
            None => Err(format!("unknown graph {k}")),
        },
    }
}

fn with_traced<R>(kind: &str, f: impl FnOnce(&Traced) -> R) -> Result<R, String> {
    TRACED.with(|t| {
        let mut t = t.borrow_mut();
        if !t.contains_key(kind) {
            let (graph, outputs) = build(kind)?;
            let mut live: Vec<usize> = graph.live_set(&outputs).into_iter().collect();
            live.sort_unstable();
            let live = live
                .into_iter()
                .map(|i| ExprId::from_index(i as u32))
                .collect();
            if t.len() > 32 {
                t.clear();
            }
            t.insert(
                kind.to_string(),
                Traced {
                    graph,
                    outputs,
                    live,
                },
            );
        }
        Ok(f(&t[kind]))
    })
}

fn op(node: &Node) -> (String, Vec<ExprId>) {
    match *node {
        Node::Var(n) => (
            ["x", "y", "z", "w"]
                .get(n as usize)
                .unwrap_or(&"v")
                .to_string(),
            vec![],
        ),
        Node::Lit(_) => {
            let v = node.as_f64().unwrap_or(0.0);
            (format!("{}", (v * 1e4).round() / 1e4), vec![])
        }
        Node::Add(a, b) => ("+".into(), vec![a, b]),
        Node::Mul(a, b) => ("×".into(), vec![a, b]),
        Node::Neg(a) => ("−".into(), vec![a]),
        Node::Recip(a) => ("1/".into(), vec![a]),
        Node::Sqrt(a) => ("√".into(), vec![a]),
        Node::Sin(a) => ("sin".into(), vec![a]),
        Node::Floor(a) => ("⌊⌋".into(), vec![a]),
        Node::Atan2(a, b) => ("atan2".into(), vec![a, b]),
        Node::Exp2(a) => ("2^".into(), vec![a]),
        Node::Log2(a) => ("log₂".into(), vec![a]),
        Node::Select(c, a, b) => ("sel".into(), vec![c, a, b]),
    }
}

/// The live part of a demo's graph, as JSON:
/// `{"nodes":[{"op":"×","args":[0,1],"lit":false}],"outputs":[5,9]}` with
/// indices into `nodes`, which are in topological order.
#[wasm_bindgen]
pub fn graph_layout(kind: &str) -> String {
    let r = with_traced(kind, |t| {
        let pos: HashMap<ExprId, usize> = t.live.iter().enumerate().map(|(i, e)| (*e, i)).collect();
        let nodes: Vec<String> = t
            .live
            .iter()
            .map(|e| {
                let n = t.graph.node(*e);
                let (name, args) = op(&n);
                let args: Vec<String> = args
                    .iter()
                    .filter_map(|a| pos.get(a))
                    .map(|i| i.to_string())
                    .collect();
                let kind = match n {
                    Node::Var(_) => "var",
                    Node::Lit(_) => "lit",
                    _ => "op",
                };
                format!(
                    r#"{{"op":"{}","args":[{}],"kind":"{}","id":{}}}"#,
                    name,
                    args.join(","),
                    kind,
                    e.index()
                )
            })
            .collect();
        let outs: Vec<String> = t
            .outputs
            .iter()
            .filter_map(|o| pos.get(o))
            .map(|i| i.to_string())
            .collect();
        format!(
            r#"{{"nodes":[{}],"outputs":[{}]}}"#,
            nodes.join(","),
            outs.join(",")
        )
    });
    r.unwrap_or_else(|e| format!(r#"{{"error":"{e}"}}"#))
}

/// Every live node's value for these inputs, in `graph_layout` order.
#[wasm_bindgen]
pub fn graph_values(kind: &str, inputs: &[f64]) -> Box<[f64]> {
    with_traced(kind, |t| t.graph.eval_many::<f64>(&t.live, inputs))
        .map(|v| v.into_boxed_slice())
        .unwrap_or_default()
}
