//! Compile expression graphs to optimized Rust closures.

use std::collections::HashSet;

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

/// Compiled single-output expression closure.
pub type CompiledExpr = Box<dyn Fn(&[f64]) -> f64>;

/// Compiled multi-output expression closure.
pub type CompiledMany = Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>;

impl ExprGraph {
    /// Compile a single expression to a closure `&[f64] -> f64`.
    ///
    /// Shared subexpressions (from interning) are computed once.
    /// Dead nodes (not reachable from output) are skipped.
    pub fn compile(&self, expr: ExprId) -> CompiledExpr {
        let live = self.live_set(&[expr]);
        let nodes = self.collect_eval_order(&live, expr.0 as usize + 1);
        let out_idx = expr.0 as usize;

        Box::new(move |inputs: &[f64]| {
            let mut vals = vec![0.0f64; out_idx + 1];
            for &(i, ref node) in &nodes {
                vals[i] = eval_node(node, &vals, inputs);
            }
            vals[out_idx]
        })
    }

    /// Compile multiple output expressions to a single closure.
    ///
    /// Writes results into the output slice.
    pub fn compile_many(&self, exprs: &[ExprId]) -> CompiledMany {
        if exprs.is_empty() {
            return Box::new(|_, _| {});
        }

        let live = self.live_set(exprs);
        let max_id = exprs.iter().map(|e| e.0).max().unwrap() as usize;
        let nodes = self.collect_eval_order(&live, max_id + 1);
        let out_indices: Vec<usize> = exprs.iter().map(|e| e.0 as usize).collect();

        Box::new(move |inputs: &[f64], outputs: &mut [f64]| {
            let mut vals = vec![0.0f64; max_id + 1];
            for &(i, ref node) in &nodes {
                vals[i] = eval_node(node, &vals, inputs);
            }
            for (k, &idx) in out_indices.iter().enumerate() {
                outputs[k] = vals[idx];
            }
        })
    }

    /// Find all node indices reachable from the given outputs.
    pub fn live_set(&self, outputs: &[ExprId]) -> HashSet<usize> {
        let mut live = HashSet::new();
        let mut stack: Vec<usize> = outputs.iter().map(|e| e.0 as usize).collect();
        while let Some(i) = stack.pop() {
            if !live.insert(i) {
                continue;
            }
            match self.node(ExprId(i as u32)) {
                Node::Var(_) | Node::Lit(_) => {}
                Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => {
                    stack.push(a.0 as usize);
                    stack.push(b.0 as usize);
                }
                Node::Neg(a)
                | Node::Recip(a)
                | Node::Sqrt(a)
                | Node::Sin(a)
                | Node::Floor(a)
                | Node::Exp2(a)
                | Node::Log2(a) => {
                    stack.push(a.0 as usize);
                }
                Node::Select(c, a, b) => {
                    stack.push(c.0 as usize);
                    stack.push(a.0 as usize);
                    stack.push(b.0 as usize);
                }
            }
        }
        live
    }

    /// Collect (index, node) pairs in topological order, only for live nodes.
    fn collect_eval_order(&self, live: &HashSet<usize>, count: usize) -> Vec<(usize, Node)> {
        (0..count)
            .filter(|i| live.contains(i))
            .map(|i| (i, self.node(ExprId(i as u32))))
            .collect()
    }
}

#[inline]
fn eval_node(node: &Node, vals: &[f64], inputs: &[f64]) -> f64 {
    match *node {
        Node::Var(idx) => inputs[idx as usize],
        Node::Lit(bits) => f64::from_bits(bits),
        Node::Add(a, b) => vals[a.0 as usize] + vals[b.0 as usize],
        Node::Mul(a, b) => vals[a.0 as usize] * vals[b.0 as usize],
        Node::Neg(a) => -vals[a.0 as usize],
        Node::Recip(a) => 1.0 / vals[a.0 as usize],
        Node::Sqrt(a) => vals[a.0 as usize].sqrt(),
        Node::Sin(a) => vals[a.0 as usize].sin(),
        Node::Floor(a) => vals[a.0 as usize].floor(),
        Node::Atan2(y, x) => vals[y.0 as usize].atan2(vals[x.0 as usize]),
        Node::Exp2(a) => vals[a.0 as usize].exp2(),
        Node::Log2(a) => vals[a.0 as usize].log2(),
        Node::Select(c, a, b) => {
            if vals[c.0 as usize] > 0.0 {
                vals[a.0 as usize]
            } else {
                vals[b.0 as usize]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ExprGraph;

    #[test]
    fn compile_add_lits() {
        let mut g = ExprGraph::new();
        let a = g.lit(3.0);
        let b = g.lit(4.0);
        let sum = g.add(a, b);
        let f = g.compile(sum);
        assert!((f(&[]) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn compile_with_vars() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        let prod = g.mul(sum, x);
        let f = g.compile(prod);
        // (3 + 4) * 3 = 21
        assert!((f(&[3.0, 4.0]) - 21.0).abs() < 1e-10);
    }

    #[test]
    fn compile_sin() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let s = g.sin(x);
        let f = g.compile(s);
        assert!((f(&[std::f64::consts::FRAC_PI_2]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn compile_many_outputs() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        let prod = g.mul(x, y);
        let f = g.compile_many(&[sum, prod]);
        let mut out = [0.0; 2];
        f(&[3.0, 4.0], &mut out);
        assert!((out[0] - 7.0).abs() < 1e-10);
        assert!((out[1] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn compile_dead_code_elimination() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let _dead = g.sin(x); // not used in output
        let result = g.mul(x, x);
        let f = g.compile(result);
        assert!((f(&[5.0]) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn compile_matches_eval() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let xx = g.mul(x, x);
        let yy = g.mul(y, y);
        let sum = g.add(xx, yy);
        let dist = g.sqrt(sum);

        let inputs = [3.0, 4.0];
        let eval_result: f64 = g.eval(dist, &inputs);
        let f = g.compile(dist);
        let compile_result = f(&inputs);
        assert!((eval_result - compile_result).abs() < 1e-10);
        assert!((compile_result - 5.0).abs() < 1e-10);
    }
}
