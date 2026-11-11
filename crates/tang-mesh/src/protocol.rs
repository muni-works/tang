//! Wire protocol types for distributed expression graph execution.
//!
//! `WireNode` mirrors `tang_expr::Node` with serde derives, keeping tang-expr
//! dependency-free. `WireGraph` is the serializable expression graph sent over
//! the wire.

use serde::{Deserialize, Serialize};
use tang_expr::node::{ExprId, Node};
use tang_expr::ExprGraph;

/// Protocol version. Incremented on breaking wire format changes.
pub const PROTOCOL_VERSION: u32 = 2;

/// A node in the wire-format expression graph.
///
/// Mirrors `tang_expr::Node` exactly — same 11 variants, same semantics.
/// Indices are u32 (matching ExprId internals).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum WireNode {
    Var(u16),
    Lit(u64),
    Add(u32, u32),
    Mul(u32, u32),
    Neg(u32),
    Recip(u32),
    Sqrt(u32),
    Sin(u32),
    Atan2(u32, u32),
    Exp2(u32),
    Log2(u32),
    Select(u32, u32, u32),
    /// Appended last so existing variant tags stay put on the wire.
    Floor(u32),
}

impl WireNode {
    /// Convert from tang-expr `Node`.
    pub fn from_node(node: Node) -> Self {
        match node {
            Node::Var(n) => WireNode::Var(n),
            Node::Lit(bits) => WireNode::Lit(bits),
            Node::Add(a, b) => WireNode::Add(a.index(), b.index()),
            Node::Mul(a, b) => WireNode::Mul(a.index(), b.index()),
            Node::Neg(a) => WireNode::Neg(a.index()),
            Node::Recip(a) => WireNode::Recip(a.index()),
            Node::Sqrt(a) => WireNode::Sqrt(a.index()),
            Node::Sin(a) => WireNode::Sin(a.index()),
            Node::Floor(a) => WireNode::Floor(a.index()),
            Node::Atan2(y, x) => WireNode::Atan2(y.index(), x.index()),
            Node::Exp2(a) => WireNode::Exp2(a.index()),
            Node::Log2(a) => WireNode::Log2(a.index()),
            Node::Select(c, a, b) => WireNode::Select(c.index(), a.index(), b.index()),
        }
    }

    /// Convert to tang-expr `Node`.
    pub fn to_node(self) -> Node {
        match self {
            WireNode::Var(n) => Node::Var(n),
            WireNode::Lit(bits) => Node::Lit(bits),
            WireNode::Add(a, b) => Node::Add(ExprId::from_index(a), ExprId::from_index(b)),
            WireNode::Mul(a, b) => Node::Mul(ExprId::from_index(a), ExprId::from_index(b)),
            WireNode::Neg(a) => Node::Neg(ExprId::from_index(a)),
            WireNode::Recip(a) => Node::Recip(ExprId::from_index(a)),
            WireNode::Sqrt(a) => Node::Sqrt(ExprId::from_index(a)),
            WireNode::Sin(a) => Node::Sin(ExprId::from_index(a)),
            WireNode::Floor(a) => Node::Floor(ExprId::from_index(a)),
            WireNode::Atan2(y, x) => Node::Atan2(ExprId::from_index(y), ExprId::from_index(x)),
            WireNode::Exp2(a) => Node::Exp2(ExprId::from_index(a)),
            WireNode::Log2(a) => Node::Log2(ExprId::from_index(a)),
            WireNode::Select(c, a, b) => Node::Select(
                ExprId::from_index(c),
                ExprId::from_index(a),
                ExprId::from_index(b),
            ),
        }
    }
}

/// A serializable expression graph for wire transmission.
///
/// Contains all nodes, output indices, and input count — everything a worker
/// needs to compile a WGSL kernel locally.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireGraph {
    /// Protocol version for forward compatibility.
    pub version: u32,
    /// All nodes in topological order (index = ExprId).
    pub nodes: Vec<WireNode>,
    /// Output node indices.
    pub outputs: Vec<u32>,
    /// Number of input variables.
    pub n_inputs: u16,
}

impl WireGraph {
    /// Convert from an `ExprGraph` with specified outputs and input count.
    pub fn from_expr_graph(graph: &ExprGraph, outputs: &[ExprId], n_inputs: u16) -> Self {
        let nodes = graph
            .nodes_slice()
            .iter()
            .map(|n| WireNode::from_node(*n))
            .collect();
        let outputs = outputs.iter().map(|e| e.index()).collect();
        Self {
            version: PROTOCOL_VERSION,
            nodes,
            outputs,
            n_inputs,
        }
    }

    /// Reconstruct an `ExprGraph` and output `ExprId`s from a wire graph.
    ///
    /// Rebuilds the graph through the public API (preserving interning).
    pub fn to_expr_graph(&self) -> (ExprGraph, Vec<ExprId>) {
        let mut graph = ExprGraph::new();

        // The pre-populated nodes (ZERO=0, ONE=1, TWO=2) are already in the
        // new graph. We need to map wire indices to new ExprIds.
        // Since ExprGraph interns, inserting the same node returns the same id.
        let mut id_map: Vec<ExprId> = Vec::with_capacity(self.nodes.len());

        for wire_node in &self.nodes {
            let id = match *wire_node {
                WireNode::Var(n) => graph.var(n),
                WireNode::Lit(bits) => graph.lit(f64::from_bits(bits)),
                WireNode::Add(a, b) => graph.add(id_map[a as usize], id_map[b as usize]),
                WireNode::Mul(a, b) => graph.mul(id_map[a as usize], id_map[b as usize]),
                WireNode::Neg(a) => graph.neg(id_map[a as usize]),
                WireNode::Recip(a) => graph.recip(id_map[a as usize]),
                WireNode::Sqrt(a) => graph.sqrt(id_map[a as usize]),
                WireNode::Sin(a) => graph.sin(id_map[a as usize]),
                WireNode::Floor(a) => graph.floor(id_map[a as usize]),
                WireNode::Atan2(y, x) => graph.atan2(id_map[y as usize], id_map[x as usize]),
                WireNode::Exp2(a) => graph.exp2(id_map[a as usize]),
                WireNode::Log2(a) => graph.log2(id_map[a as usize]),
                WireNode::Select(c, a, b) => {
                    graph.select(id_map[c as usize], id_map[a as usize], id_map[b as usize])
                }
            };
            id_map.push(id);
        }

        let outputs = self
            .outputs
            .iter()
            .map(|&idx| id_map[idx as usize])
            .collect();

        (graph, outputs)
    }

    /// Generate WGSL directly from the wire graph (convenience).
    pub fn to_wgsl(&self) -> tang_expr::wgsl::WgslKernel {
        let (graph, outputs) = self.to_expr_graph();
        graph.to_wgsl(&outputs, self.n_inputs as usize)
    }

    /// Serialize to postcard bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize from postcard bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_expr::ExprGraph;

    #[test]
    fn roundtrip_simple() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        let prod = g.mul(sum, x);

        let wire = WireGraph::from_expr_graph(&g, &[prod], 2);
        assert_eq!(wire.version, PROTOCOL_VERSION);
        assert_eq!(wire.n_inputs, 2);
        assert_eq!(wire.outputs.len(), 1);

        let (g2, outputs2) = wire.to_expr_graph();
        let result: f64 = g2.eval(outputs2[0], &[3.0, 4.0]);
        let expected: f64 = g.eval(prod, &[3.0, 4.0]);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn roundtrip_all_ops() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);

        let a = g.add(x, y);
        let b = g.mul(x, y);
        let c = g.neg(a);
        let d = g.recip(b);
        let e = g.sqrt(x);
        let f = g.sin(y);
        let h = g.atan2(x, y);
        let i = g.exp2(x);
        let j = g.log2(y);

        let outputs = vec![a, b, c, d, e, f, h, i, j];
        let wire = WireGraph::from_expr_graph(&g, &outputs, 2);
        let (g2, outputs2) = wire.to_expr_graph();

        let inputs = [2.0_f64, 3.0];
        let orig: Vec<f64> = g.eval_many(&outputs, &inputs);
        let reconst: Vec<f64> = g2.eval_many(&outputs2, &inputs);

        for (a, b) in orig.iter().zip(reconst.iter()) {
            assert!((a - b).abs() < 1e-10, "mismatch: orig={a}, reconst={b}");
        }
    }

    #[test]
    fn roundtrip_wgsl_identical() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let xx = g.mul(x, x);
        let yy = g.mul(y, y);
        let sum = g.add(xx, yy);
        let dist = g.sqrt(sum);

        let wgsl_orig = g.to_wgsl(&[dist], 2);
        let wire = WireGraph::from_expr_graph(&g, &[dist], 2);
        let wgsl_wire = wire.to_wgsl();

        assert_eq!(wgsl_orig.source, wgsl_wire.source);
        assert_eq!(wgsl_orig.n_inputs, wgsl_wire.n_inputs);
        assert_eq!(wgsl_orig.n_outputs, wgsl_wire.n_outputs);
    }

    #[test]
    fn postcard_serde_roundtrip() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);

        let wire = WireGraph::from_expr_graph(&g, &[sum], 2);
        let bytes = wire.to_bytes().unwrap();
        let wire2 = WireGraph::from_bytes(&bytes).unwrap();

        assert_eq!(wire, wire2);
    }

    #[test]
    fn roundtrip_select() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let a = g.lit(3.0);
        let b = g.lit(7.0);
        let s = g.select(x, a, b);

        let wire = WireGraph::from_expr_graph(&g, &[s], 1);
        let (g2, outputs2) = wire.to_expr_graph();

        // cond > 0 → a
        let r1: f64 = g2.eval(outputs2[0], &[1.0]);
        assert!((r1 - 3.0).abs() < 1e-10);
        // cond <= 0 → b
        let r2: f64 = g2.eval(outputs2[0], &[-1.0]);
        assert!((r2 - 7.0).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn wire_node_conversions() {
        let e = ExprId::from_index;
        let test_cases = vec![
            Node::Var(42),
            Node::Lit(3.14_f64.to_bits()),
            Node::Add(e(0), e(1)),
            Node::Mul(e(2), e(3)),
            Node::Neg(e(4)),
            Node::Recip(e(5)),
            Node::Sqrt(e(6)),
            Node::Sin(e(7)),
            Node::Atan2(e(8), e(9)),
            Node::Exp2(e(10)),
            Node::Log2(e(11)),
            Node::Select(e(0), e(1), e(2)),
        ];

        for node in test_cases {
            let wire = WireNode::from_node(node);
            let back = wire.to_node();
            assert_eq!(node, back);
        }
    }
}
