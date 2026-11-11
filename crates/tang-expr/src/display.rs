//! Pretty-printing for expressions.

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

impl ExprGraph {
    /// Format an expression as a human-readable string.
    pub fn fmt_expr(&self, expr: ExprId) -> String {
        match self.node(expr) {
            Node::Var(n) => format!("x{n}"),
            Node::Lit(bits) => {
                let v = f64::from_bits(bits);
                if v == 0.0 {
                    "0".to_string()
                } else if v == 1.0 {
                    "1".to_string()
                } else if v == 2.0 {
                    "2".to_string()
                } else if v == -1.0 {
                    "-1".to_string()
                } else {
                    format!("{v}")
                }
            }
            Node::Add(a, b) => {
                format!("({} + {})", self.fmt_expr(a), self.fmt_expr(b))
            }
            Node::Mul(a, b) => {
                format!("({} * {})", self.fmt_expr(a), self.fmt_expr(b))
            }
            Node::Neg(a) => format!("(-{})", self.fmt_expr(a)),
            Node::Recip(a) => format!("(1 / {})", self.fmt_expr(a)),
            Node::Sqrt(a) => format!("sqrt({})", self.fmt_expr(a)),
            Node::Sin(a) => format!("sin({})", self.fmt_expr(a)),
            Node::Floor(a) => format!("floor({})", self.fmt_expr(a)),
            Node::Atan2(y, x) => {
                format!("atan2({}, {})", self.fmt_expr(y), self.fmt_expr(x))
            }
            Node::Exp2(a) => format!("exp2({})", self.fmt_expr(a)),
            Node::Log2(a) => format!("log2({})", self.fmt_expr(a)),
            Node::Select(c, a, b) => {
                format!(
                    "select({}, {}, {})",
                    self.fmt_expr(c),
                    self.fmt_expr(a),
                    self.fmt_expr(b)
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ExprGraph;
    use crate::node::ExprId;

    #[test]
    fn display_simple() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        assert_eq!(g.fmt_expr(sum), "(x0 + x1)");

        let prod = g.mul(x, y);
        assert_eq!(g.fmt_expr(prod), "(x0 * x1)");

        let s = g.sin(x);
        assert_eq!(g.fmt_expr(s), "sin(x0)");
    }

    #[test]
    fn display_constants() {
        let g = ExprGraph::new();
        assert_eq!(g.fmt_expr(ExprId::ZERO), "0");
        assert_eq!(g.fmt_expr(ExprId::ONE), "1");
        assert_eq!(g.fmt_expr(ExprId::TWO), "2");
    }
}
