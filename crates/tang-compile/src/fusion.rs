//! Operator fusion planning for tensor computation graphs.
//!
//! Groups sequential element-wise operations into fused kernels that
//! execute in a single GPU dispatch, reducing memory bandwidth.

use std::collections::HashSet;
use tang_expr::{ExprGraph, ExprId, Node};

/// A group of fused operations that can execute as a single kernel.
#[derive(Clone, Debug)]
pub struct FusionGroup {
    /// Unique group ID.
    pub id: usize,
    /// Node indices in this group (topologically sorted).
    pub nodes: Vec<u32>,
    /// External inputs to this group (produced outside the group).
    pub inputs: Vec<ExprId>,
    /// Outputs consumed outside this group.
    pub outputs: Vec<ExprId>,
}

/// A fused kernel ready for code generation.
#[derive(Clone, Debug)]
pub struct FusedKernel {
    /// The subgraph for this kernel.
    pub group: FusionGroup,
    /// Estimated FLOPs for this kernel.
    pub flops: usize,
    /// Whether this kernel is memory-bound (low arithmetic intensity).
    pub memory_bound: bool,
}

/// Plans operator fusion for an expression graph.
pub struct FusionPlanner;

impl FusionPlanner {
    /// Identify fusible groups of element-wise operations.
    ///
    /// Returns groups of nodes that can be fused into single kernels.
    /// Non-element-wise ops (like MatMul via Var nodes) act as barriers.
    pub fn plan(graph: &ExprGraph, outputs: &[ExprId]) -> Vec<FusionGroup> {
        let nodes = graph.nodes_slice();
        let n = nodes.len();

        // Find live nodes
        let mut live = vec![false; n];
        let mut stack: Vec<usize> = outputs.iter().map(|o| o.index() as usize).collect();
        while let Some(idx) = stack.pop() {
            if live[idx] {
                continue;
            }
            live[idx] = true;
            Self::visit_children(&nodes[idx], &mut stack);
        }

        // Count uses of each node
        let mut use_count = vec![0u32; n];
        for i in 0..n {
            if !live[i] {
                continue;
            }
            Self::for_each_child(&nodes[i], |child_idx| {
                use_count[child_idx] += 1;
            });
        }
        for o in outputs {
            use_count[o.index() as usize] += 1;
        }

        // Assign nodes to fusion groups.
        // Element-wise ops with single-use inputs can be fused with their producer.
        let mut group_of = vec![usize::MAX; n];
        let mut groups: Vec<Vec<u32>> = Vec::new();

        for i in 0..n {
            if !live[i] {
                continue;
            }

            if Self::is_elementwise(&nodes[i]) {
                // Try to merge into an existing group from an input
                let mut merged = false;
                Self::for_each_child(&nodes[i], |child_idx| {
                    if !merged
                        && group_of[child_idx] != usize::MAX
                        && use_count[child_idx] == 1
                        && Self::is_elementwise(&nodes[child_idx])
                    {
                        let gid = group_of[child_idx];
                        group_of[i] = gid;
                        groups[gid].push(i as u32);
                        merged = true;
                    }
                });

                if !merged {
                    let gid = groups.len();
                    group_of[i] = gid;
                    groups.push(vec![i as u32]);
                }
            }
            // Non-elementwise nodes (Var, Lit) are not grouped
        }

        // Build FusionGroup structs
        let mut result = Vec::new();
        for (gid, node_indices) in groups.into_iter().enumerate() {
            if node_indices.len() < 2 {
                continue; // Only fuse groups with 2+ ops
            }

            let group_set: HashSet<u32> = node_indices.iter().copied().collect();

            // Find external inputs
            let mut inputs = Vec::new();
            let mut seen_inputs = HashSet::new();
            for &ni in &node_indices {
                Self::for_each_child(&nodes[ni as usize], |child_idx| {
                    if !group_set.contains(&(child_idx as u32)) && seen_inputs.insert(child_idx) {
                        inputs.push(ExprId::from_index(child_idx as u32));
                    }
                });
            }

            // Find outputs (used outside group or in final outputs)
            let output_set: HashSet<u32> = outputs.iter().map(|o| o.index()).collect();
            let mut group_outputs = Vec::new();
            for &ni in &node_indices {
                let is_output = output_set.contains(&ni);
                let used_outside = Self::is_used_outside(ni, &group_set, nodes, n, &live);
                if is_output || used_outside {
                    group_outputs.push(ExprId::from_index(ni));
                }
            }

            result.push(FusionGroup {
                id: gid,
                nodes: node_indices,
                inputs,
                outputs: group_outputs,
            });
        }

        result
    }

    fn is_elementwise(node: &Node) -> bool {
        matches!(
            node,
            Node::Add(_, _)
                | Node::Mul(_, _)
                | Node::Neg(_)
                | Node::Recip(_)
                | Node::Sqrt(_)
                | Node::Sin(_)
                | Node::Floor(_)
                | Node::Exp2(_)
                | Node::Log2(_)
                | Node::Atan2(_, _)
                | Node::Select(_, _, _)
        )
    }

    fn visit_children(node: &Node, stack: &mut Vec<usize>) {
        Self::for_each_child(node, |idx| stack.push(idx));
    }

    fn for_each_child(node: &Node, mut f: impl FnMut(usize)) {
        match node {
            Node::Lit(_) | Node::Var(_) => {}
            Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => {
                f(a.index() as usize);
                f(b.index() as usize);
            }
            Node::Neg(a)
            | Node::Recip(a)
            | Node::Sqrt(a)
            | Node::Sin(a)
            | Node::Floor(a)
            | Node::Exp2(a)
            | Node::Log2(a) => {
                f(a.index() as usize);
            }
            Node::Select(c, a, b) => {
                f(c.index() as usize);
                f(a.index() as usize);
                f(b.index() as usize);
            }
        }
    }

    fn is_used_outside(
        node_idx: u32,
        group: &HashSet<u32>,
        nodes: &[Node],
        n: usize,
        live: &[bool],
    ) -> bool {
        let target = ExprId::from_index(node_idx);
        for i in 0..n {
            if !live[i] || group.contains(&(i as u32)) {
                continue;
            }
            let mut uses_target = false;
            Self::for_each_child(&nodes[i], |child_idx| {
                if ExprId::from_index(child_idx as u32) == target {
                    uses_target = true;
                }
            });
            if uses_target {
                return true;
            }
        }
        false
    }
}

impl FusedKernel {
    /// Create from a fusion group with estimated cost.
    pub fn from_group(group: FusionGroup) -> Self {
        let flops = group.nodes.len(); // 1 FLOP per node (rough estimate)
        let memory_bound = group.nodes.len() < 4; // small chains are memory-bound
        Self {
            group,
            flops,
            memory_bound,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_chain() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let a = g.neg(x); // elementwise
        let b = g.neg(a); // elementwise, single-use chain
        let c = g.neg(b); // elementwise, single-use chain

        let groups = FusionPlanner::plan(&g, &[c]);

        // Should fuse neg(neg(neg(x))) into one group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].nodes.len(), 3);
    }

    #[test]
    fn no_fusion_for_single_ops() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let a = g.neg(x);

        let groups = FusionPlanner::plan(&g, &[a]);
        // Single op — no fusion needed
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn diamond_pattern() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let a = g.neg(x); // used by both b and c
        let b = g.neg(a);
        let c = g.neg(a); // multi-use of `a` prevents fusion with b
        let d = g.add(b, c);

        let groups = FusionPlanner::plan(&g, &[d]);

        // a is multi-use so acts as a barrier
        // b+d and c can potentially be fused in smaller groups
        // The exact grouping depends on traversal order, but we should get some fusion
        assert!(!groups.is_empty());
    }

    #[test]
    fn fused_kernel_from_group() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let a = g.neg(x);
        let b = g.neg(a);
        let c = g.neg(b);

        let groups = FusionPlanner::plan(&g, &[c]);
        assert_eq!(groups.len(), 1);

        let kernel = FusedKernel::from_group(groups.into_iter().next().unwrap());
        assert_eq!(kernel.flops, 3);
    }
}
