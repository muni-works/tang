//! Generic evaluation of expression graphs.

use tang::Scalar;

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

impl ExprGraph {
    /// Evaluate an expression with concrete scalar inputs.
    ///
    /// `inputs[n]` provides the value for `Var(n)`. Walks the graph in
    /// topological order (which is just index order, since children are
    /// always created before parents).
    pub fn eval<S: Scalar>(&self, expr: ExprId, inputs: &[S]) -> S {
        let n = expr.0 as usize + 1;
        let mut vals: Vec<S> = Vec::with_capacity(n);

        for i in 0..n {
            let v = match self.node(ExprId(i as u32)) {
                Node::Var(idx) => inputs[idx as usize],
                Node::Lit(bits) => S::from_f64(f64::from_bits(bits)),
                Node::Add(a, b) => vals[a.0 as usize] + vals[b.0 as usize],
                Node::Mul(a, b) => vals[a.0 as usize] * vals[b.0 as usize],
                Node::Neg(a) => -vals[a.0 as usize],
                Node::Recip(a) => vals[a.0 as usize].recip(),
                Node::Sqrt(a) => vals[a.0 as usize].sqrt(),
                Node::Sin(a) => vals[a.0 as usize].sin(),
                Node::Floor(a) => vals[a.0 as usize].floor(),
                Node::Atan2(y, x) => vals[y.0 as usize].atan2(vals[x.0 as usize]),
                Node::Exp2(a) => {
                    // exp2(x) = 2^x = exp(x * ln(2))
                    let x = vals[a.0 as usize];
                    (x * S::from_f64(std::f64::consts::LN_2)).exp()
                }
                Node::Log2(a) => {
                    // log2(x) = ln(x) / ln(2)
                    let x = vals[a.0 as usize];
                    x.ln() * S::from_f64(std::f64::consts::LOG2_E)
                }
                Node::Select(c, a, b) => {
                    S::select(vals[c.0 as usize], vals[a.0 as usize], vals[b.0 as usize])
                }
            };
            vals.push(v);
        }

        vals[expr.0 as usize]
    }

    /// Evaluate multiple output expressions, sharing intermediate values.
    pub fn eval_many<S: Scalar>(&self, exprs: &[ExprId], inputs: &[S]) -> Vec<S> {
        if exprs.is_empty() {
            return Vec::new();
        }
        let max_id = exprs.iter().map(|e| e.0).max().unwrap() as usize;
        let n = max_id + 1;
        let mut vals: Vec<S> = Vec::with_capacity(n);

        for i in 0..n {
            let v = match self.node(ExprId(i as u32)) {
                Node::Var(idx) => inputs[idx as usize],
                Node::Lit(bits) => S::from_f64(f64::from_bits(bits)),
                Node::Add(a, b) => vals[a.0 as usize] + vals[b.0 as usize],
                Node::Mul(a, b) => vals[a.0 as usize] * vals[b.0 as usize],
                Node::Neg(a) => -vals[a.0 as usize],
                Node::Recip(a) => vals[a.0 as usize].recip(),
                Node::Sqrt(a) => vals[a.0 as usize].sqrt(),
                Node::Sin(a) => vals[a.0 as usize].sin(),
                Node::Floor(a) => vals[a.0 as usize].floor(),
                Node::Atan2(y, x) => vals[y.0 as usize].atan2(vals[x.0 as usize]),
                Node::Exp2(a) => {
                    let x = vals[a.0 as usize];
                    (x * S::from_f64(std::f64::consts::LN_2)).exp()
                }
                Node::Log2(a) => {
                    let x = vals[a.0 as usize];
                    x.ln() * S::from_f64(std::f64::consts::LOG2_E)
                }
                Node::Select(c, a, b) => {
                    S::select(vals[c.0 as usize], vals[a.0 as usize], vals[b.0 as usize])
                }
            };
            vals.push(v);
        }

        exprs.iter().map(|e| vals[e.0 as usize]).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ExprGraph;

    #[test]
    fn eval_add_lits() {
        let mut g = ExprGraph::new();
        let a = g.lit(3.0);
        let b = g.lit(4.0);
        let sum = g.add(a, b);
        let result: f64 = g.eval(sum, &[]);
        assert!((result - 7.0).abs() < 1e-10);
    }

    #[test]
    fn eval_with_vars() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        let prod = g.mul(sum, x);
        // (x + y) * x at x=3, y=4 = 7 * 3 = 21
        let result: f64 = g.eval(prod, &[3.0, 4.0]);
        assert!((result - 21.0).abs() < 1e-10);
    }

    #[test]
    fn eval_sqrt() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let sq = g.sqrt(x);
        let result: f64 = g.eval(sq, &[9.0]);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eval_sin() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let s = g.sin(x);
        let result: f64 = g.eval(s, &[std::f64::consts::FRAC_PI_2]);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eval_select_positive_cond() {
        let mut g = ExprGraph::new();
        let cond = g.lit(1.0);
        let a = g.lit(3.0);
        let b = g.lit(7.0);
        let s = g.select(cond, a, b);
        let result: f64 = g.eval(s, &[]);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eval_select_negative_cond() {
        let mut g = ExprGraph::new();
        let cond = g.lit(-1.0);
        let a = g.lit(3.0);
        let b = g.lit(7.0);
        let s = g.select(cond, a, b);
        let result: f64 = g.eval(s, &[]);
        assert!((result - 7.0).abs() < 1e-10);
    }

    #[test]
    fn eval_select_zero_cond() {
        // cond == 0 should select b (not > 0)
        let mut g = ExprGraph::new();
        let cond = g.lit(0.0);
        let a = g.lit(3.0);
        let b = g.lit(7.0);
        let s = g.select(cond, a, b);
        let result: f64 = g.eval(s, &[]);
        assert!((result - 7.0).abs() < 1e-10);
    }

    #[test]
    fn eval_many_outputs() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        let prod = g.mul(x, y);
        let results: Vec<f64> = g.eval_many(&[sum, prod], &[3.0, 4.0]);
        assert!((results[0] - 7.0).abs() < 1e-10);
        assert!((results[1] - 12.0).abs() < 1e-10);
    }
}
