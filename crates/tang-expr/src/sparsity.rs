//! Sparsity analysis via dependency bitmasks.

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

impl ExprGraph {
    /// Compute a bitmask of which `Var(n)` indices appear in `expr`.
    ///
    /// Bit `n` is set if `Var(n)` is reachable from `expr`.
    /// Supports up to 64 variables.
    pub fn deps(&self, expr: ExprId) -> u64 {
        let n = expr.0 as usize + 1;
        let mut masks = vec![0u64; n];

        for i in 0..n {
            let m = match self.node(ExprId(i as u32)) {
                Node::Var(idx) => {
                    assert!(idx < 64, "deps() supports at most 64 variables");
                    1u64 << idx
                }
                Node::Lit(_) => 0,
                Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => {
                    masks[a.0 as usize] | masks[b.0 as usize]
                }
                Node::Neg(a)
                | Node::Recip(a)
                | Node::Sqrt(a)
                | Node::Sin(a)
                | Node::Floor(a)
                | Node::Exp2(a)
                | Node::Log2(a) => masks[a.0 as usize],
                Node::Select(c, a, b) => {
                    masks[c.0 as usize] | masks[a.0 as usize] | masks[b.0 as usize]
                }
            };
            masks[i] = m;
        }

        masks[expr.0 as usize]
    }

    /// Compute the Jacobian sparsity pattern.
    ///
    /// Returns one `u64` bitmask per output expression. Bit `j` of `result[i]`
    /// is set if `outputs[i]` depends on `Var(j)`.
    pub fn jacobian_sparsity(&self, outputs: &[ExprId], n_vars: usize) -> Vec<u64> {
        if outputs.is_empty() {
            return Vec::new();
        }

        let max_id = outputs.iter().map(|e| e.0).max().unwrap() as usize;
        let n = max_id + 1;
        let mut masks = vec![0u64; n];

        for i in 0..n {
            let m = match self.node(ExprId(i as u32)) {
                Node::Var(idx) => {
                    if (idx as usize) < n_vars {
                        1u64 << idx
                    } else {
                        0
                    }
                }
                Node::Lit(_) => 0,
                Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => {
                    masks[a.0 as usize] | masks[b.0 as usize]
                }
                Node::Neg(a)
                | Node::Recip(a)
                | Node::Sqrt(a)
                | Node::Sin(a)
                | Node::Floor(a)
                | Node::Exp2(a)
                | Node::Log2(a) => masks[a.0 as usize],
                Node::Select(c, a, b) => {
                    masks[c.0 as usize] | masks[a.0 as usize] | masks[b.0 as usize]
                }
            };
            masks[i] = m;
        }

        outputs.iter().map(|e| masks[e.0 as usize]).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ExprGraph;

    #[test]
    fn deps_var() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        assert_eq!(g.deps(x), 0b1);
        let y = g.var(1);
        assert_eq!(g.deps(y), 0b10);
    }

    #[test]
    fn deps_lit() {
        let mut g = ExprGraph::new();
        let c = g.lit(42.0);
        assert_eq!(g.deps(c), 0);
    }

    #[test]
    fn deps_add() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        assert_eq!(g.deps(sum), 0b11);
    }

    #[test]
    fn deps_dot_product() {
        let mut g = ExprGraph::new();
        let x0 = g.var(0);
        let x1 = g.var(1);
        let x2 = g.var(2);
        let x3 = g.var(3);
        let x4 = g.var(4);
        let x5 = g.var(5);

        let t0 = g.mul(x0, x3);
        let t1 = g.mul(x1, x4);
        let t2 = g.mul(x2, x5);
        let s = g.add(t0, t1);
        let dot = g.add(s, t2);

        assert_eq!(g.deps(dot), 0b111111);
    }

    #[test]
    fn jacobian_sparsity_basic() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let z = g.var(2);

        let f0 = g.add(x, y); // depends on x0, x1
        let f1 = g.mul(y, z); // depends on x1, x2
        let f2 = g.sin(x); // depends on x0

        let sparsity = g.jacobian_sparsity(&[f0, f1, f2], 3);
        assert_eq!(sparsity[0], 0b011); // f0 depends on x0, x1
        assert_eq!(sparsity[1], 0b110); // f1 depends on x1, x2
        assert_eq!(sparsity[2], 0b001); // f2 depends on x0
    }
}
