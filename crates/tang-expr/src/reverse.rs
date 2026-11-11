//! Reverse-mode differentiation: vector-Jacobian products as graphs.

use std::collections::HashMap;

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

impl ExprGraph {
    /// The vector-Jacobian product `seedᵀ ∂outputs/∂vars` as new
    /// expressions in this graph: entry `k` is `Σ_i seeds[i] ∂outputs[i]/∂Var(k)`
    /// for `k < n_vars`. One reverse sweep over the live nodes, so the cost
    /// is a small multiple of evaluating `outputs` whatever `n_vars` is
    /// (forward [`diff`](Self::diff) is one sweep per variable). Seeds are
    /// expressions too (typically fresh `Var`s past the inputs), so one
    /// compiled graph serves every seed. `Select` passes gradient to the
    /// taken branch only, as `diff` does.
    pub fn vjp(&mut self, outputs: &[ExprId], seeds: &[ExprId], n_vars: usize) -> Vec<ExprId> {
        assert_eq!(outputs.len(), seeds.len(), "one seed per output");
        let live = self.live_set(outputs);
        let mut order: Vec<usize> = live.into_iter().collect();
        // children are created before their parents: descending index is a
        // reverse topological order
        order.sort_unstable_by(|a, b| b.cmp(a));
        let mut adj: HashMap<usize, ExprId> = HashMap::new();
        for (&o, &s) in outputs.iter().zip(seeds) {
            self.accumulate(&mut adj, o, s);
        }
        let mut grad = vec![ExprId::ZERO; n_vars];
        for i in order {
            let Some(&g) = adj.get(&i) else { continue };
            let id = ExprId(i as u32);
            match self.node(id) {
                Node::Var(n) => {
                    if (n as usize) < n_vars {
                        grad[n as usize] = g;
                    }
                }
                Node::Lit(_) => {}
                Node::Add(a, b) => {
                    self.accumulate(&mut adj, a, g);
                    self.accumulate(&mut adj, b, g);
                }
                Node::Mul(a, b) => {
                    let ga = self.mul(g, b);
                    let gb = self.mul(g, a);
                    self.accumulate(&mut adj, a, ga);
                    self.accumulate(&mut adj, b, gb);
                }
                Node::Neg(a) => {
                    let ga = self.neg(g);
                    self.accumulate(&mut adj, a, ga);
                }
                Node::Recip(a) => {
                    // d(1/a) = -(1/a)² da
                    let rr = self.mul(id, id);
                    let t = self.mul(g, rr);
                    let ga = self.neg(t);
                    self.accumulate(&mut adj, a, ga);
                }
                Node::Sqrt(a) => {
                    // d√a = da / (2√a)
                    let two = self.mul(ExprId::TWO, id);
                    let r = self.recip(two);
                    let ga = self.mul(g, r);
                    self.accumulate(&mut adj, a, ga);
                }
                Node::Sin(a) => {
                    let half_pi = self.lit(std::f64::consts::FRAC_PI_2);
                    let shifted = self.add(a, half_pi);
                    let cos = self.sin(shifted);
                    let ga = self.mul(g, cos);
                    self.accumulate(&mut adj, a, ga);
                }
                // floor is piecewise constant: no gradient flows through it.
                Node::Floor(_) => {}
                Node::Atan2(y, x) => {
                    // ∂/∂y = x / (x² + y²), ∂/∂x = -y / (x² + y²)
                    let xx = self.mul(x, x);
                    let yy = self.mul(y, y);
                    let den = self.add(xx, yy);
                    let r = self.recip(den);
                    let gr = self.mul(g, r);
                    let gy = self.mul(gr, x);
                    let t = self.mul(gr, y);
                    let gx = self.neg(t);
                    self.accumulate(&mut adj, y, gy);
                    self.accumulate(&mut adj, x, gx);
                }
                Node::Exp2(a) => {
                    let ln2 = self.lit(std::f64::consts::LN_2);
                    let t = self.mul(ln2, id);
                    let ga = self.mul(g, t);
                    self.accumulate(&mut adj, a, ga);
                }
                Node::Log2(a) => {
                    let ln2 = self.lit(std::f64::consts::LN_2);
                    let t = self.mul(ln2, a);
                    let r = self.recip(t);
                    let ga = self.mul(g, r);
                    self.accumulate(&mut adj, a, ga);
                }
                Node::Select(c, a, b) => {
                    let ga = self.select(c, g, ExprId::ZERO);
                    let gb = self.select(c, ExprId::ZERO, g);
                    self.accumulate(&mut adj, a, ga);
                    self.accumulate(&mut adj, b, gb);
                }
            }
        }
        grad
    }

    fn accumulate(&mut self, adj: &mut HashMap<usize, ExprId>, at: ExprId, g: ExprId) {
        let i = at.0 as usize;
        let sum = match adj.get(&i) {
            Some(&prev) => self.add(prev, g),
            None => g,
        };
        adj.insert(i, sum);
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ExprGraph;

    /// Every op, against forward `diff`, at a few points.
    #[test]
    fn vjp_matches_forward_diff() {
        let mut g = ExprGraph::new();
        let (x, y, z) = (g.var(0), g.var(1), g.var(2));
        let xy = g.mul(x, y);
        let s = g.sin(xy);
        let r = g.recip(z);
        let sq = g.sqrt(z);
        let at = g.atan2(y, x);
        let e = g.exp2(x);
        let l = g.log2(z);
        let n = g.neg(at);
        let sel = g.select(x, s, r);
        let t1 = g.add(sel, sq);
        let t2 = g.mul(t1, n);
        let t3 = g.add(e, l);
        let o0 = g.add(t2, t3);
        let o1 = g.mul(xy, xy);
        let outs = [o0, o1];
        let seeds = [g.var(3), g.var(4)];
        let grad = g.vjp(&outs, &seeds, 3);
        let fwd: Vec<_> = (0..3u16).map(|k| [g.diff(o0, k), g.diff(o1, k)]).collect();
        for pt in [[0.7, -0.4, 2.3, 1.5, -0.25], [-1.2, 0.9, 0.6, -0.5, 2.0]] {
            for k in 0..3 {
                let want: f64 =
                    pt[3] * g.eval::<f64>(fwd[k][0], &pt) + pt[4] * g.eval::<f64>(fwd[k][1], &pt);
                let got: f64 = g.eval(grad[k], &pt);
                assert!(
                    (got - want).abs() < 1e-12 * want.abs().max(1.0),
                    "var {k}: {got} vs {want}"
                );
            }
        }
    }
}
