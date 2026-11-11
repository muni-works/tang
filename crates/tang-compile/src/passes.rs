//! Optimization passes for expression graphs.

use tang_expr::{ExprGraph, ExprId, Node};

/// A single optimization pass over an expression graph.
pub trait OptimizationPass {
    /// Name of this pass (for logging).
    fn name(&self) -> &str;

    /// Run the pass. Returns a new graph and remapped output expressions.
    fn run(&self, graph: &ExprGraph, outputs: &[ExprId]) -> (ExprGraph, Vec<ExprId>);
}

/// Manages a pipeline of optimization passes.
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the pipeline.
    pub fn add(&mut self, pass: impl OptimizationPass + 'static) {
        self.passes.push(Box::new(pass));
    }

    /// Run all passes in sequence.
    pub fn run(&self, graph: &ExprGraph, outputs: &[ExprId]) -> (ExprGraph, Vec<ExprId>) {
        let mut g = graph.clone();
        let mut outs = outputs.to_vec();

        for pass in &self.passes {
            let (new_g, new_outs) = pass.run(&g, &outs);
            g = new_g;
            outs = new_outs;
        }

        (g, outs)
    }

    /// Number of registered passes.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Constant folding: evaluate subexpressions that depend only on literals.
pub struct FoldConstants;

impl OptimizationPass for FoldConstants {
    fn name(&self) -> &str {
        "FoldConstants"
    }

    fn run(&self, graph: &ExprGraph, outputs: &[ExprId]) -> (ExprGraph, Vec<ExprId>) {
        let nodes = graph.nodes_slice();
        let n = nodes.len();

        // Determine which nodes are constant (all inputs are literals/constants)
        let mut is_const = vec![false; n];
        let mut const_val = vec![0.0f64; n];

        for i in 0..n {
            match nodes[i] {
                Node::Lit(bits) => {
                    is_const[i] = true;
                    const_val[i] = f64::from_bits(bits);
                }
                Node::Var(_) => {
                    is_const[i] = false;
                }
                Node::Add(a, b) => {
                    let ai = a.index() as usize;
                    let bi = b.index() as usize;
                    if is_const[ai] && is_const[bi] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai] + const_val[bi];
                    }
                }
                Node::Mul(a, b) => {
                    let ai = a.index() as usize;
                    let bi = b.index() as usize;
                    if is_const[ai] && is_const[bi] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai] * const_val[bi];
                    }
                }
                Node::Neg(a) => {
                    let ai = a.index() as usize;
                    if is_const[ai] {
                        is_const[i] = true;
                        const_val[i] = -const_val[ai];
                    }
                }
                Node::Recip(a) => {
                    let ai = a.index() as usize;
                    if is_const[ai] {
                        is_const[i] = true;
                        const_val[i] = 1.0 / const_val[ai];
                    }
                }
                Node::Sqrt(a) => {
                    let ai = a.index() as usize;
                    if is_const[ai] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai].sqrt();
                    }
                }
                Node::Sin(a) => {
                    let ai = a.index() as usize;
                    if is_const[ai] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai].sin();
                    }
                }
                Node::Floor(a) => {
                    let ai = a.index() as usize;
                    if is_const[ai] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai].floor();
                    }
                }
                Node::Exp2(a) => {
                    let ai = a.index() as usize;
                    if is_const[ai] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai].exp2();
                    }
                }
                Node::Log2(a) => {
                    let ai = a.index() as usize;
                    if is_const[ai] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai].log2();
                    }
                }
                Node::Atan2(a, b) => {
                    let ai = a.index() as usize;
                    let bi = b.index() as usize;
                    if is_const[ai] && is_const[bi] {
                        is_const[i] = true;
                        const_val[i] = const_val[ai].atan2(const_val[bi]);
                    }
                }
                Node::Select(c, a, b) => {
                    let ci = c.index() as usize;
                    let ai = a.index() as usize;
                    let bi = b.index() as usize;
                    if is_const[ci] && is_const[ai] && is_const[bi] {
                        is_const[i] = true;
                        const_val[i] = if const_val[ci] > 0.0 {
                            const_val[ai]
                        } else {
                            const_val[bi]
                        };
                    }
                }
            }
        }

        // Rebuild graph, replacing constant subtrees with literals
        let mut new_g = ExprGraph::new();
        let mut remap = vec![ExprId::from_index(0); n];

        for i in 0..n {
            if is_const[i] {
                remap[i] = new_g.lit(const_val[i]);
            } else {
                remap[i] = match nodes[i] {
                    Node::Lit(bits) => new_g.lit(f64::from_bits(bits)),
                    Node::Var(v) => new_g.var(v),
                    Node::Add(a, b) => {
                        new_g.add(remap[a.index() as usize], remap[b.index() as usize])
                    }
                    Node::Mul(a, b) => {
                        new_g.mul(remap[a.index() as usize], remap[b.index() as usize])
                    }
                    Node::Neg(a) => new_g.neg(remap[a.index() as usize]),
                    Node::Recip(a) => new_g.recip(remap[a.index() as usize]),
                    Node::Sqrt(a) => new_g.sqrt(remap[a.index() as usize]),
                    Node::Sin(a) => new_g.sin(remap[a.index() as usize]),
                    Node::Floor(a) => new_g.floor(remap[a.index() as usize]),
                    Node::Exp2(a) => new_g.exp2(remap[a.index() as usize]),
                    Node::Log2(a) => new_g.log2(remap[a.index() as usize]),
                    Node::Atan2(a, b) => {
                        new_g.atan2(remap[a.index() as usize], remap[b.index() as usize])
                    }
                    Node::Select(c, a, b) => new_g.select(
                        remap[c.index() as usize],
                        remap[a.index() as usize],
                        remap[b.index() as usize],
                    ),
                };
            }
        }

        let new_outputs: Vec<ExprId> = outputs.iter().map(|o| remap[o.index() as usize]).collect();

        (new_g, new_outputs)
    }
}

/// Dead code elimination: remove nodes not reachable from outputs.
pub struct EliminateDeadCode;

impl OptimizationPass for EliminateDeadCode {
    fn name(&self) -> &str {
        "EliminateDeadCode"
    }

    fn run(&self, graph: &ExprGraph, outputs: &[ExprId]) -> (ExprGraph, Vec<ExprId>) {
        let nodes = graph.nodes_slice();
        let n = nodes.len();

        // Mark live nodes (backwards from outputs)
        let mut live = vec![false; n];
        let mut stack: Vec<usize> = outputs.iter().map(|o| o.index() as usize).collect();

        while let Some(idx) = stack.pop() {
            if live[idx] {
                continue;
            }
            live[idx] = true;
            match nodes[idx] {
                Node::Lit(_) | Node::Var(_) => {}
                Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => {
                    stack.push(a.index() as usize);
                    stack.push(b.index() as usize);
                }
                Node::Neg(a)
                | Node::Recip(a)
                | Node::Sqrt(a)
                | Node::Sin(a)
                | Node::Floor(a)
                | Node::Exp2(a)
                | Node::Log2(a) => {
                    stack.push(a.index() as usize);
                }
                Node::Select(c, a, b) => {
                    stack.push(c.index() as usize);
                    stack.push(a.index() as usize);
                    stack.push(b.index() as usize);
                }
            }
        }

        // Rebuild with only live nodes
        let mut new_g = ExprGraph::new();
        let mut remap = vec![ExprId::from_index(0); n];

        for i in 0..n {
            if !live[i] {
                continue;
            }
            remap[i] = match nodes[i] {
                Node::Lit(bits) => new_g.lit(f64::from_bits(bits)),
                Node::Var(v) => new_g.var(v),
                Node::Add(a, b) => new_g.add(remap[a.index() as usize], remap[b.index() as usize]),
                Node::Mul(a, b) => new_g.mul(remap[a.index() as usize], remap[b.index() as usize]),
                Node::Neg(a) => new_g.neg(remap[a.index() as usize]),
                Node::Recip(a) => new_g.recip(remap[a.index() as usize]),
                Node::Sqrt(a) => new_g.sqrt(remap[a.index() as usize]),
                Node::Sin(a) => new_g.sin(remap[a.index() as usize]),
                Node::Floor(a) => new_g.floor(remap[a.index() as usize]),
                Node::Exp2(a) => new_g.exp2(remap[a.index() as usize]),
                Node::Log2(a) => new_g.log2(remap[a.index() as usize]),
                Node::Atan2(a, b) => {
                    new_g.atan2(remap[a.index() as usize], remap[b.index() as usize])
                }
                Node::Select(c, a, b) => new_g.select(
                    remap[c.index() as usize],
                    remap[a.index() as usize],
                    remap[b.index() as usize],
                ),
            };
        }

        let new_outputs: Vec<ExprId> = outputs.iter().map(|o| remap[o.index() as usize]).collect();

        (new_g, new_outputs)
    }
}

/// Algebraic simplification using tang-expr's built-in simplifier.
pub struct SimplifyAlgebraic;

impl OptimizationPass for SimplifyAlgebraic {
    fn name(&self) -> &str {
        "SimplifyAlgebraic"
    }

    fn run(&self, graph: &ExprGraph, outputs: &[ExprId]) -> (ExprGraph, Vec<ExprId>) {
        let mut g = graph.clone();
        let simplified: Vec<ExprId> = outputs.iter().map(|&o| g.simplify(o)).collect();
        (g, simplified)
    }
}

/// Fuse sequential element-wise operations into groups.
///
/// Identifies chains of unary/binary element-wise ops that can be fused
/// into a single kernel pass. This is a metadata pass — it marks fusion
/// groups but doesn't generate code.
pub struct FuseElementwise;

impl OptimizationPass for FuseElementwise {
    fn name(&self) -> &str {
        "FuseElementwise"
    }

    fn run(&self, graph: &ExprGraph, outputs: &[ExprId]) -> (ExprGraph, Vec<ExprId>) {
        // For now, this is a no-op pass that preserves the graph.
        // In a full implementation, it would annotate fusion groups
        // that the WGSL codegen can emit as single kernels.
        //
        // The actual fusion happens at the WGSL level via tang_expr::to_wgsl()
        // which already emits fused element-wise chains as single kernels.
        (graph.clone(), outputs.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_constants_basic() {
        let mut g = ExprGraph::new();
        let a = g.lit(3.0);
        let b = g.lit(4.0);
        let sum = g.add(a, b);
        let x = g.var(0);
        let result = g.mul(sum, x); // (3+4) * x -> 7 * x

        let pass = FoldConstants;
        let (new_g, new_outs) = pass.run(&g, &[result]);

        // The folded graph should evaluate correctly
        let val = new_g.eval(new_outs[0], &[2.0_f64]);
        assert!((val - 14.0).abs() < 1e-10);
    }

    #[test]
    fn fold_constants_nested() {
        let mut g = ExprGraph::new();
        let a = g.lit(2.0);
        let b = g.lit(3.0);
        let c = g.lit(5.0);
        let ab = g.mul(a, b); // 6
        let abc = g.add(ab, c); // 11

        let pass = FoldConstants;
        let (new_g, new_outs) = pass.run(&g, &[abc]);

        let val: f64 = new_g.eval(new_outs[0], &[]);
        assert!((val - 11.0).abs() < 1e-10);

        // The folded graph should just be a literal
        assert!(matches!(new_g.node(new_outs[0]), Node::Lit(_)));
    }

    #[test]
    fn dce_removes_dead_nodes() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let _dead = g.mul(x, y); // dead — not in outputs
        let one = g.lit(1.0);
        let live = g.add(x, one);

        let pass = EliminateDeadCode;
        let (new_g, new_outs) = pass.run(&g, &[live]);

        // New graph should be smaller (no dead mul node, no y)
        assert!(new_g.len() < g.len());

        let val = new_g.eval(new_outs[0], &[5.0_f64]);
        assert!((val - 6.0).abs() < 1e-10);
    }

    #[test]
    fn simplify_algebraic() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let zero = g.lit(0.0);
        let result = g.add(x, zero); // x + 0 -> x

        let pass = SimplifyAlgebraic;
        let (new_g, new_outs) = pass.run(&g, &[result]);

        // After simplification, x + 0 should just be x
        let val = new_g.eval(new_outs[0], &[42.0_f64]);
        assert!((val - 42.0).abs() < 1e-10);
    }

    #[test]
    fn pass_manager_pipeline() {
        let mut g = ExprGraph::new();
        let a = g.lit(2.0);
        let b = g.lit(3.0);
        let ab = g.mul(a, b); // 6, foldable
        let x = g.var(0);
        let _dead = g.sin(x); // dead
        let result = g.add(ab, x); // 6 + x

        let mut pm = PassManager::new();
        pm.add(FoldConstants);
        pm.add(EliminateDeadCode);

        let (new_g, new_outs) = pm.run(&g, &[result]);

        let val = new_g.eval(new_outs[0], &[4.0_f64]);
        assert!((val - 10.0).abs() < 1e-10);

        // Should be smaller than original
        assert!(new_g.len() < g.len());
    }
}
