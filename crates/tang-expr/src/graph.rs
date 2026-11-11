//! Expression graph with structural interning (automatic CSE).

use std::collections::HashMap;

use crate::node::{ExprId, Node};

/// Arena-based expression graph with structural interning.
///
/// Identical subexpressions always return the same `ExprId` — this gives
/// automatic common subexpression elimination (CSE) for free.
#[derive(Clone)]
pub struct ExprGraph {
    nodes: Vec<Node>,
    intern: HashMap<Node, ExprId>,
}

impl ExprGraph {
    /// Create a new graph pre-populated with ZERO, ONE, TWO.
    pub fn new() -> Self {
        let mut g = Self {
            nodes: Vec::new(),
            intern: HashMap::new(),
        };
        // Index 0 = ZERO
        let z = g.insert(Node::lit(0.0));
        debug_assert_eq!(z, ExprId::ZERO);
        // Index 1 = ONE
        let o = g.insert(Node::lit(1.0));
        debug_assert_eq!(o, ExprId::ONE);
        // Index 2 = TWO
        let t = g.insert(Node::lit(2.0));
        debug_assert_eq!(t, ExprId::TWO);
        g
    }

    /// Total number of nodes in the graph.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty (it never is after construction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look up the node for an ExprId.
    #[inline]
    pub fn node(&self, id: ExprId) -> Node {
        self.nodes[id.0 as usize]
    }

    /// Read-only access to the node arena for serialization.
    #[inline]
    pub fn nodes_slice(&self) -> &[Node] {
        &self.nodes
    }

    /// Internal: insert a node, returning its interned ExprId.
    fn insert(&mut self, node: Node) -> ExprId {
        if let Some(&id) = self.intern.get(&node) {
            return id;
        }
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.intern.insert(node, id);
        id
    }

    /// Create a variable node.
    #[inline]
    pub fn var(&mut self, n: u16) -> ExprId {
        self.insert(Node::Var(n))
    }

    /// Create a literal node.
    #[inline]
    pub fn lit(&mut self, v: f64) -> ExprId {
        self.insert(Node::lit(v))
    }

    /// Add two expressions.
    #[inline]
    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.insert(Node::Add(a, b))
    }

    /// Multiply two expressions.
    #[inline]
    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.insert(Node::Mul(a, b))
    }

    /// Negate an expression.
    #[inline]
    pub fn neg(&mut self, a: ExprId) -> ExprId {
        self.insert(Node::Neg(a))
    }

    /// Reciprocal (1/x).
    #[inline]
    pub fn recip(&mut self, a: ExprId) -> ExprId {
        self.insert(Node::Recip(a))
    }

    /// Square root.
    #[inline]
    pub fn sqrt(&mut self, a: ExprId) -> ExprId {
        self.insert(Node::Sqrt(a))
    }

    /// Sine.
    #[inline]
    pub fn sin(&mut self, a: ExprId) -> ExprId {
        self.insert(Node::Sin(a))
    }

    /// atan2(y, x).
    #[inline]
    pub fn atan2(&mut self, y: ExprId, x: ExprId) -> ExprId {
        self.insert(Node::Atan2(y, x))
    }

    /// Base-2 exponential.
    #[inline]
    pub fn exp2(&mut self, a: ExprId) -> ExprId {
        self.insert(Node::Exp2(a))
    }

    /// Base-2 logarithm.
    #[inline]
    pub fn log2(&mut self, a: ExprId) -> ExprId {
        self.insert(Node::Log2(a))
    }

    /// Floor (largest integer <= x). Folds literals.
    #[inline]
    pub fn floor(&mut self, a: ExprId) -> ExprId {
        if let Some(v) = self.node(a).as_f64() {
            return self.lit(v.floor());
        }
        self.insert(Node::Floor(a))
    }

    /// Integer `a mod b` for integer-valued `a` and a positive integer `b`
    /// (floored, so `-1 mod 4 == 3`): `a - b * floor((a + 0.5) / b)`. The
    /// half offset keeps the quotient at least `0.5 / b` from an integer, so
    /// rounding in `1 / b` (and f32 kernels) cannot flip the floor.
    pub fn imod(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let half = self.lit(0.5);
        let ah = self.add(a, half);
        let rb = self.recip(b);
        let q = self.mul(ah, rb);
        let fq = self.floor(q);
        let bq = self.mul(b, fq);
        let nbq = self.neg(bq);
        self.add(a, nbq)
    }

    /// Branchless select: returns `a` if `cond > 0`, else `b`.
    #[inline]
    pub fn select(&mut self, cond: ExprId, a: ExprId, b: ExprId) -> ExprId {
        self.insert(Node::Select(cond, a, b))
    }
}

impl Default for ExprGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_populated() {
        let g = ExprGraph::new();
        assert_eq!(g.node(ExprId::ZERO).as_f64(), Some(0.0));
        assert_eq!(g.node(ExprId::ONE).as_f64(), Some(1.0));
        assert_eq!(g.node(ExprId::TWO).as_f64(), Some(2.0));
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn interning() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let x2 = g.var(0);
        assert_eq!(x, x2);

        let a = g.add(x, ExprId::ONE);
        let a2 = g.add(x, ExprId::ONE);
        assert_eq!(a, a2);
    }

    #[test]
    fn lit_nan_distinct() {
        let mut g = ExprGraph::new();
        // NaN bits are deterministic for the same f64::NAN
        let a = g.lit(f64::NAN);
        let b = g.lit(f64::NAN);
        assert_eq!(a, b); // same bits → same id
    }
}
