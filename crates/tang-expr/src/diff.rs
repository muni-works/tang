//! Symbolic differentiation.

use std::collections::HashMap;

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

impl ExprGraph {
    /// Differentiate `expr` with respect to `Var(var)`.
    ///
    /// Returns a new ExprId in the same graph. Uses memoization to avoid
    /// recomputing derivatives of shared subexpressions.
    pub fn diff(&mut self, expr: ExprId, var: u16) -> ExprId {
        let mut memo = HashMap::new();
        self.diff_inner(expr, var, &mut memo)
    }

    fn diff_inner(
        &mut self,
        expr: ExprId,
        var: u16,
        memo: &mut HashMap<(ExprId, u16), ExprId>,
    ) -> ExprId {
        if let Some(&cached) = memo.get(&(expr, var)) {
            return cached;
        }

        let result = match self.node(expr) {
            Node::Var(n) => {
                if n == var {
                    ExprId::ONE
                } else {
                    ExprId::ZERO
                }
            }
            Node::Lit(_) => ExprId::ZERO,

            Node::Add(a, b) => {
                // d(a + b) = da + db
                let da = self.diff_inner(a, var, memo);
                let db = self.diff_inner(b, var, memo);
                self.add(da, db)
            }

            Node::Mul(a, b) => {
                // d(a * b) = da*b + a*db (product rule)
                let da = self.diff_inner(a, var, memo);
                let db = self.diff_inner(b, var, memo);
                let t1 = self.mul(da, b);
                let t2 = self.mul(a, db);
                self.add(t1, t2)
            }

            Node::Neg(a) => {
                // d(-a) = -da
                let da = self.diff_inner(a, var, memo);
                self.neg(da)
            }

            Node::Recip(a) => {
                // d(1/a) = -da / a^2
                let da = self.diff_inner(a, var, memo);
                let a_sq = self.mul(a, a);
                let r = self.recip(a_sq);
                let t = self.mul(da, r);
                self.neg(t)
            }

            Node::Sqrt(a) => {
                // d(sqrt(a)) = da / (2 * sqrt(a))
                let da = self.diff_inner(a, var, memo);
                let sq = self.sqrt(a);
                let two_sq = self.mul(ExprId::TWO, sq);
                let r = self.recip(two_sq);
                self.mul(da, r)
            }

            Node::Sin(a) => {
                // d(sin(a)) = cos(a) * da
                // cos(a) = sin(a + PI/2)
                let da = self.diff_inner(a, var, memo);
                let half_pi = self.lit(std::f64::consts::FRAC_PI_2);
                let shifted = self.add(a, half_pi);
                let cos_a = self.sin(shifted);
                self.mul(cos_a, da)
            }

            // floor is piecewise constant: d(floor(a)) = 0.
            Node::Floor(_) => ExprId::ZERO,

            Node::Atan2(y, x) => {
                // d(atan2(y, x)) = (x*dy - y*dx) / (x^2 + y^2)
                let dy = self.diff_inner(y, var, memo);
                let dx = self.diff_inner(x, var, memo);
                let x_dy = self.mul(x, dy);
                let y_dx = self.mul(y, dx);
                let neg_y_dx = self.neg(y_dx);
                let numer = self.add(x_dy, neg_y_dx);
                let xx = self.mul(x, x);
                let yy = self.mul(y, y);
                let denom = self.add(xx, yy);
                let r = self.recip(denom);
                self.mul(numer, r)
            }

            Node::Exp2(a) => {
                // d(2^a) = ln(2) * 2^a * da
                let da = self.diff_inner(a, var, memo);
                let ln2 = self.lit(std::f64::consts::LN_2);
                let exp2_a = self.exp2(a);
                let t = self.mul(ln2, exp2_a);
                self.mul(t, da)
            }

            Node::Log2(a) => {
                // d(log2(a)) = da / (ln(2) * a)
                let da = self.diff_inner(a, var, memo);
                let ln2 = self.lit(std::f64::consts::LN_2);
                let ln2_a = self.mul(ln2, a);
                let r = self.recip(ln2_a);
                self.mul(da, r)
            }

            Node::Select(c, a, b) => {
                // Straight-through: condition doesn't contribute gradient
                let da = self.diff_inner(a, var, memo);
                let db = self.diff_inner(b, var, memo);
                self.select(c, da, db)
            }
        };

        memo.insert((expr, var), result);
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ExprGraph;
    use crate::node::ExprId;

    #[test]
    fn diff_constant() {
        let mut g = ExprGraph::new();
        let c = g.lit(5.0);
        let dc = g.diff(c, 0);
        assert_eq!(dc, ExprId::ZERO);
    }

    #[test]
    fn diff_var_self() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let dx = g.diff(x, 0);
        assert_eq!(dx, ExprId::ONE);
    }

    #[test]
    fn diff_var_other() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let dx = g.diff(x, 1);
        assert_eq!(dx, ExprId::ZERO);
    }

    #[test]
    fn diff_add() {
        // d/dx (x + 3) = 1
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let c = g.lit(3.0);
        let sum = g.add(x, c);
        let d = g.diff(sum, 0);
        // d = Add(ONE, ZERO)
        let result: f64 = g.eval(d, &[99.0]); // value doesn't matter
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn diff_mul_product_rule() {
        // d/dx (x * x) = 2x
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let xx = g.mul(x, x);
        let d = g.diff(xx, 0);
        // At x=3, d/dx x^2 = 6
        let result: f64 = g.eval(d, &[3.0]);
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn diff_sin() {
        // d/dx sin(x) = cos(x)
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let s = g.sin(x);
        let ds = g.diff(s, 0);
        // At x=0, cos(0) = 1
        let result: f64 = g.eval(ds, &[0.0]);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn diff_chain_rule() {
        // d/dx sin(x^2) = 2x * cos(x^2)
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let xx = g.mul(x, x);
        let s = g.sin(xx);
        let ds = g.diff(s, 0);
        // At x=1: 2*1*cos(1)
        let expected = 2.0 * 1.0_f64.cos();
        let result: f64 = g.eval(ds, &[1.0]);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn diff_sqrt() {
        // d/dx sqrt(x) = 1 / (2*sqrt(x))
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let sq = g.sqrt(x);
        let d = g.diff(sq, 0);
        // At x=4: 1/(2*2) = 0.25
        let result: f64 = g.eval(d, &[4.0]);
        assert!((result - 0.25).abs() < 1e-10);
    }

    #[test]
    fn diff_recip() {
        // d/dx (1/x) = -1/x^2
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let r = g.recip(x);
        let d = g.diff(r, 0);
        // At x=2: -1/4 = -0.25
        let result: f64 = g.eval(d, &[2.0]);
        assert!((result - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn diff_memoization() {
        // Shared subexpression: d/dx (x*x + x*x)
        // Should reuse derivative of x*x
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let xx = g.mul(x, x);
        let sum = g.add(xx, xx);
        let d = g.diff(sum, 0);
        // d/dx (2x^2) = 4x, at x=3 → 12
        let result: f64 = g.eval(d, &[3.0]);
        assert!((result - 12.0).abs() < 1e-10);
    }

    #[test]
    fn diff_select() {
        // d/dx select(x, x*x, x+1) at x=2 (cond>0 → d/dx x² = 2x = 4)
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let xx = g.mul(x, x);
        let xp1 = g.add(x, ExprId::ONE);
        let s = g.select(x, xx, xp1);
        let ds = g.diff(s, 0);
        let result: f64 = g.eval(ds, &[2.0]);
        assert!((result - 4.0).abs() < 1e-10);

        // At x=-1 (cond<=0 → d/dx (x+1) = 1)
        let result2: f64 = g.eval(ds, &[-1.0]);
        assert!((result2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn diff_dot_product() {
        // f = x0*x3 + x1*x4 + x2*x5  (dot product)
        // df/dx0 = x3
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
        let s01 = g.add(t0, t1);
        let dot = g.add(s01, t2);

        let d0 = g.diff(dot, 0);
        // df/dx0 = x3, at inputs [1,2,3,4,5,6] → 4
        let result: f64 = g.eval(d0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!((result - 4.0).abs() < 1e-10);
    }
}
