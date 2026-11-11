//! Pattern-matched simplification rules.

use std::collections::HashMap;

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

impl ExprGraph {
    /// Simplify an expression by applying rewrite rules to fixpoint.
    ///
    /// Bottom-up: simplify children first, then match parent. Iterates
    /// until no more changes occur.
    pub fn simplify(&mut self, expr: ExprId) -> ExprId {
        let mut memo = HashMap::new();
        self.simplify_inner(expr, &mut memo)
    }

    fn simplify_inner(&mut self, expr: ExprId, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
        if let Some(&cached) = memo.get(&expr) {
            return cached;
        }

        // First, simplify children
        let simplified_children = match self.node(expr) {
            Node::Var(_) | Node::Lit(_) => expr,
            Node::Add(a, b) => {
                let sa = self.simplify_inner(a, memo);
                let sb = self.simplify_inner(b, memo);
                self.add(sa, sb)
            }
            Node::Mul(a, b) => {
                let sa = self.simplify_inner(a, memo);
                let sb = self.simplify_inner(b, memo);
                self.mul(sa, sb)
            }
            Node::Neg(a) => {
                let sa = self.simplify_inner(a, memo);
                self.neg(sa)
            }
            Node::Recip(a) => {
                let sa = self.simplify_inner(a, memo);
                self.recip(sa)
            }
            Node::Sqrt(a) => {
                let sa = self.simplify_inner(a, memo);
                self.sqrt(sa)
            }
            Node::Sin(a) => {
                let sa = self.simplify_inner(a, memo);
                self.sin(sa)
            }
            Node::Floor(a) => {
                let sa = self.simplify_inner(a, memo);
                self.floor(sa)
            }
            Node::Atan2(y, x) => {
                let sy = self.simplify_inner(y, memo);
                let sx = self.simplify_inner(x, memo);
                self.atan2(sy, sx)
            }
            Node::Exp2(a) => {
                let sa = self.simplify_inner(a, memo);
                self.exp2(sa)
            }
            Node::Log2(a) => {
                let sa = self.simplify_inner(a, memo);
                self.log2(sa)
            }
            Node::Select(c, a, b) => {
                let sc = self.simplify_inner(c, memo);
                let sa = self.simplify_inner(a, memo);
                let sb = self.simplify_inner(b, memo);
                self.select(sc, sa, sb)
            }
        };

        // Now apply rewrite rules on the node with simplified children
        let result = self.rewrite(simplified_children);

        // If rewrite changed something, simplify again (fixpoint)
        let final_result = if result != simplified_children {
            self.simplify_inner(result, memo)
        } else {
            result
        };

        memo.insert(expr, final_result);
        final_result
    }

    /// Apply one round of rewrite rules.
    fn rewrite(&mut self, expr: ExprId) -> ExprId {
        match self.node(expr) {
            // --- Identity / Annihilation ---

            // Add(x, ZERO) → x
            Node::Add(a, b) if b == ExprId::ZERO => a,
            // Add(ZERO, x) → x
            Node::Add(a, b) if a == ExprId::ZERO => b,

            // Mul(x, ONE) → x
            Node::Mul(a, b) if b == ExprId::ONE => a,
            // Mul(ONE, x) → x
            Node::Mul(a, b) if a == ExprId::ONE => b,
            // Mul(x, ZERO) → ZERO
            Node::Mul(_, b) if b == ExprId::ZERO => ExprId::ZERO,
            // Mul(ZERO, x) → ZERO
            Node::Mul(a, _) if a == ExprId::ZERO => ExprId::ZERO,

            // Neg(Neg(x)) → x
            Node::Neg(a) => match self.node(a) {
                Node::Neg(inner) => inner,
                // Neg(ZERO) → ZERO
                _ if a == ExprId::ZERO => ExprId::ZERO,
                // Constant folding: Neg(Lit(v)) → Lit(-v)
                Node::Lit(bits) => {
                    let v = f64::from_bits(bits);
                    self.lit(-v)
                }
                _ => expr,
            },

            // Recip(Recip(x)) → x
            Node::Recip(a) => match self.node(a) {
                Node::Recip(inner) => inner,
                // Constant folding: Recip(Lit(v)) → Lit(1/v)
                Node::Lit(bits) => {
                    let v = f64::from_bits(bits);
                    self.lit(1.0 / v)
                }
                _ => expr,
            },

            // --- Cancellation ---

            // Add(x, Neg(x)) → ZERO
            Node::Add(a, b) => {
                if let Node::Neg(inner) = self.node(b) {
                    if inner == a {
                        return ExprId::ZERO;
                    }
                }
                if let Node::Neg(inner) = self.node(a) {
                    if inner == b {
                        return ExprId::ZERO;
                    }
                }
                // Constant folding: Add(Lit(a), Lit(b)) → Lit(a+b)
                if let (Some(va), Some(vb)) = (self.node(a).as_f64(), self.node(b).as_f64()) {
                    return self.lit(va + vb);
                }
                expr
            }

            Node::Mul(a, b) => {
                // Mul(x, Recip(x)) → ONE
                if let Node::Recip(inner) = self.node(b) {
                    if inner == a {
                        return ExprId::ONE;
                    }
                }
                if let Node::Recip(inner) = self.node(a) {
                    if inner == b {
                        return ExprId::ONE;
                    }
                }
                // Constant folding: Mul(Lit(a), Lit(b)) → Lit(a*b)
                if let (Some(va), Some(vb)) = (self.node(a).as_f64(), self.node(b).as_f64()) {
                    return self.lit(va * vb);
                }
                expr
            }

            // Constant folding for unary ops
            Node::Sqrt(a) => {
                if let Some(v) = self.node(a).as_f64() {
                    self.lit(v.sqrt())
                } else {
                    expr
                }
            }
            Node::Sin(a) => {
                if let Some(v) = self.node(a).as_f64() {
                    self.lit(v.sin())
                } else {
                    expr
                }
            }
            Node::Floor(a) => {
                if let Some(v) = self.node(a).as_f64() {
                    self.lit(v.floor())
                } else {
                    expr
                }
            }
            Node::Exp2(a) => {
                if let Some(v) = self.node(a).as_f64() {
                    self.lit(v.exp2())
                } else {
                    expr
                }
            }
            Node::Log2(a) => {
                if let Some(v) = self.node(a).as_f64() {
                    self.lit(v.log2())
                } else {
                    expr
                }
            }

            // Select constant folding
            Node::Select(c, a, b) => {
                if let Some(vc) = self.node(c).as_f64() {
                    if vc > 0.0 {
                        a
                    } else {
                        b
                    }
                } else {
                    expr
                }
            }

            _ => expr,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ExprGraph;
    use crate::node::ExprId;

    #[test]
    fn simplify_add_zero() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let sum = g.add(x, ExprId::ZERO);
        let s = g.simplify(sum);
        assert_eq!(s, x);

        let sum2 = g.add(ExprId::ZERO, x);
        let s2 = g.simplify(sum2);
        assert_eq!(s2, x);
    }

    #[test]
    fn simplify_mul_one() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let prod = g.mul(x, ExprId::ONE);
        let s = g.simplify(prod);
        assert_eq!(s, x);
    }

    #[test]
    fn simplify_mul_zero() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let prod = g.mul(x, ExprId::ZERO);
        let s = g.simplify(prod);
        assert_eq!(s, ExprId::ZERO);
    }

    #[test]
    fn simplify_neg_neg() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let nn = g.neg(x);
        let nnn = g.neg(nn);
        let s = g.simplify(nnn);
        assert_eq!(s, x);
    }

    #[test]
    fn simplify_recip_recip() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let r = g.recip(x);
        let rr = g.recip(r);
        let s = g.simplify(rr);
        assert_eq!(s, x);
    }

    #[test]
    fn simplify_cancel_add_neg() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let nx = g.neg(x);
        let sum = g.add(x, nx);
        let s = g.simplify(sum);
        assert_eq!(s, ExprId::ZERO);
    }

    #[test]
    fn simplify_cancel_mul_recip() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let rx = g.recip(x);
        let prod = g.mul(x, rx);
        let s = g.simplify(prod);
        assert_eq!(s, ExprId::ONE);
    }

    #[test]
    fn simplify_constant_fold_add() {
        let mut g = ExprGraph::new();
        let a = g.lit(3.0);
        let b = g.lit(4.0);
        let sum = g.add(a, b);
        let s = g.simplify(sum);
        let result: f64 = g.eval(s, &[]);
        assert!((result - 7.0).abs() < 1e-10);
    }

    #[test]
    fn simplify_constant_fold_mul() {
        let mut g = ExprGraph::new();
        let a = g.lit(3.0);
        let b = g.lit(4.0);
        let prod = g.mul(a, b);
        let s = g.simplify(prod);
        let result: f64 = g.eval(s, &[]);
        assert!((result - 12.0).abs() < 1e-10);
    }

    #[test]
    fn simplify_neg_zero() {
        let mut g = ExprGraph::new();
        let nz = g.neg(ExprId::ZERO);
        let s = g.simplify(nz);
        // Neg(Lit(0)) → Lit(-0) which is 0.0 in bits check
        // Actually -0.0 has different bits than 0.0, so this creates a new lit.
        // But functionally it's still zero. Let's just verify the value.
        let result: f64 = g.eval(s, &[]);
        assert!(result == 0.0);
    }

    #[test]
    fn simplify_derivative() {
        // d/dx (x*x) = 2x after simplification
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let xx = g.mul(x, x);
        let d = g.diff(xx, 0);

        // Before simplification, d = Add(Mul(ONE, x), Mul(x, ONE))
        // After: Add(x, x)
        let s = g.simplify(d);
        let result: f64 = g.eval(s, &[5.0]);
        assert!((result - 10.0).abs() < 1e-10);
    }
}
