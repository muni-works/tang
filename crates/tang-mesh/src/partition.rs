//! Graph partitioning: cut expression graphs for distribution across nodes.
//!
//! The expression graph enables partitioning at arbitrary cut points,
//! not just layer boundaries. This finds minimum-cut partitions that
//! minimize activation surface area crossing device boundaries
//! (inspired by Alpa's inter-operator partitioning pass).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::mesh::{Mesh, NodeId};
use crate::placement::Strategy;
use crate::protocol::{WireGraph, WireNode};

/// A subgraph assigned to a specific node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphPartition {
    /// Target node for this partition.
    pub node: NodeId,
    /// The subgraph to execute.
    pub graph: WireGraph,
    /// Inputs received from other partitions: (source_node, var_indices).
    pub ingress: Vec<(NodeId, Vec<u32>)>,
    /// Outputs sent to other partitions: (dest_node, output_indices).
    pub egress: Vec<(NodeId, Vec<u32>)>,
}

/// Partition an expression graph across a mesh.
///
/// For `DataParallel`, the full graph goes to every node.
/// For `Pipeline`, the graph is split into sequential stages.
pub fn partition(
    graph: &WireGraph,
    mesh: &Mesh,
    strategy: &Strategy,
) -> Result<Vec<GraphPartition>, crate::error::MeshError> {
    match strategy {
        Strategy::DataParallel => partition_data_parallel(graph, mesh),
        Strategy::Pipeline { stage_assignments } => {
            partition_pipeline(graph, mesh, stage_assignments)
        }
        Strategy::TensorParallel { split_axis: _ } => {
            // Tensor parallelism requires splitting individual operations.
            // For now, fall back to data parallel.
            partition_data_parallel(graph, mesh)
        }
    }
}

/// Data-parallel partitioning: full graph on every node.
fn partition_data_parallel(
    graph: &WireGraph,
    mesh: &Mesh,
) -> Result<Vec<GraphPartition>, crate::error::MeshError> {
    Ok(mesh
        .nodes()
        .iter()
        .map(|node| GraphPartition {
            node: node.id,
            graph: graph.clone(),
            ingress: Vec::new(),
            egress: Vec::new(),
        })
        .collect())
}

/// Pipeline partitioning: split graph into sequential stages.
///
/// Each stage assignment maps a node to a set of node indices in the graph.
/// The partitioner extracts subgraphs and wires up ingress/egress.
fn partition_pipeline(
    graph: &WireGraph,
    _mesh: &Mesh,
    stage_assignments: &[(NodeId, Vec<u32>)],
) -> Result<Vec<GraphPartition>, crate::error::MeshError> {
    if stage_assignments.is_empty() {
        return Err(crate::error::MeshError::PartitionFailed(
            "no stage assignments".into(),
        ));
    }

    let mut partitions = Vec::new();

    for (stage_idx, (target_node, node_indices)) in stage_assignments.iter().enumerate() {
        let node_set: HashSet<u32> = node_indices.iter().copied().collect();

        // Find which inputs this subgraph needs from outside its node set
        let mut external_inputs: HashMap<u32, HashSet<NodeId>> = HashMap::new();
        let mut subgraph_nodes = Vec::new();

        for &idx in node_indices {
            if (idx as usize) >= graph.nodes.len() {
                continue;
            }
            let wire_node = graph.nodes[idx as usize];
            subgraph_nodes.push((idx, wire_node));

            // Check if any operands reference nodes outside this partition
            for dep in wire_node_deps(wire_node) {
                if !node_set.contains(&dep) {
                    // Find which stage owns this dependency
                    for (other_node, other_indices) in stage_assignments {
                        if other_indices.contains(&dep) {
                            external_inputs.entry(dep).or_default().insert(*other_node);
                        }
                    }
                }
            }
        }

        // Build the subgraph: remap indices to be contiguous
        let mut old_to_new: HashMap<u32, u32> = HashMap::new();
        let mut new_nodes = Vec::new();
        let mut n_vars = 0u16;

        // External inputs become Var nodes in the subgraph
        let external_deps: Vec<u32> = external_inputs.keys().copied().collect();
        for &ext_idx in &external_deps {
            let var_idx = n_vars;
            old_to_new.insert(ext_idx, new_nodes.len() as u32);
            new_nodes.push(WireNode::Var(var_idx));
            n_vars += 1;
        }

        // Add original Var nodes (shifted)
        for &idx in node_indices {
            if (idx as usize) >= graph.nodes.len() {
                continue;
            }
            if old_to_new.contains_key(&idx) {
                continue;
            }
            let wire_node = graph.nodes[idx as usize];
            let new_idx = new_nodes.len() as u32;
            old_to_new.insert(idx, new_idx);

            match wire_node {
                WireNode::Var(_v) => {
                    new_nodes.push(WireNode::Var(n_vars));
                    n_vars += 1;
                }
                _ => {
                    // Remap operand indices
                    new_nodes.push(remap_wire_node(wire_node, &old_to_new));
                }
            }
        }

        // Determine outputs: nodes referenced by later stages or in graph.outputs
        let output_set: HashSet<u32> = graph.outputs.iter().copied().collect();
        let mut subgraph_outputs = Vec::new();

        for &idx in node_indices {
            if output_set.contains(&idx) {
                if let Some(&new_idx) = old_to_new.get(&idx) {
                    subgraph_outputs.push(new_idx);
                }
            }
        }

        // Also mark nodes that later stages depend on as outputs
        for (later_idx, (_, later_indices)) in stage_assignments.iter().enumerate() {
            if later_idx <= stage_idx {
                continue;
            }
            for &later_node_idx in later_indices {
                if (later_node_idx as usize) >= graph.nodes.len() {
                    continue;
                }
                for dep in wire_node_deps(graph.nodes[later_node_idx as usize]) {
                    if node_set.contains(&dep) {
                        if let Some(&new_idx) = old_to_new.get(&dep) {
                            if !subgraph_outputs.contains(&new_idx) {
                                subgraph_outputs.push(new_idx);
                            }
                        }
                    }
                }
            }
        }

        if subgraph_outputs.is_empty() && !node_indices.is_empty() {
            // Default: last node is the output
            if let Some(&last) = node_indices.last() {
                if let Some(&new_idx) = old_to_new.get(&last) {
                    subgraph_outputs.push(new_idx);
                }
            }
        }

        // Build ingress/egress
        let ingress: Vec<(NodeId, Vec<u32>)> = external_inputs
            .iter()
            .map(|(&dep_idx, sources)| {
                let source = *sources.iter().next().unwrap();
                (source, vec![dep_idx])
            })
            .collect();

        // Egress: find which later stages need our outputs
        let mut egress: Vec<(NodeId, Vec<u32>)> = Vec::new();
        for (later_node, later_indices) in stage_assignments.iter().skip(stage_idx + 1) {
            let needed: Vec<u32> = later_indices
                .iter()
                .filter(|&&li| {
                    if (li as usize) >= graph.nodes.len() {
                        return false;
                    }
                    wire_node_deps(graph.nodes[li as usize])
                        .into_iter()
                        .any(|dep| node_set.contains(&dep))
                })
                .copied()
                .collect();
            if !needed.is_empty() {
                egress.push((*later_node, needed));
            }
        }

        let sub_wire = WireGraph {
            version: graph.version,
            nodes: new_nodes,
            outputs: subgraph_outputs,
            n_inputs: n_vars,
        };

        partitions.push(GraphPartition {
            node: *target_node,
            graph: sub_wire,
            ingress,
            egress,
        });
    }

    Ok(partitions)
}

/// Get direct dependencies of a WireNode (operand indices).
fn wire_node_deps(node: WireNode) -> Vec<u32> {
    match node {
        WireNode::Var(_) | WireNode::Lit(_) => vec![],
        WireNode::Add(a, b) | WireNode::Mul(a, b) | WireNode::Atan2(a, b) => vec![a, b],
        WireNode::Neg(a)
        | WireNode::Recip(a)
        | WireNode::Sqrt(a)
        | WireNode::Sin(a)
        | WireNode::Floor(a)
        | WireNode::Exp2(a)
        | WireNode::Log2(a) => vec![a],
        WireNode::Select(c, a, b) => vec![c, a, b],
    }
}

/// Remap operand indices in a WireNode using an old→new index map.
fn remap_wire_node(node: WireNode, map: &HashMap<u32, u32>) -> WireNode {
    let remap = |idx: u32| -> u32 { *map.get(&idx).unwrap_or(&idx) };
    match node {
        WireNode::Var(n) => WireNode::Var(n),
        WireNode::Lit(bits) => WireNode::Lit(bits),
        WireNode::Add(a, b) => WireNode::Add(remap(a), remap(b)),
        WireNode::Mul(a, b) => WireNode::Mul(remap(a), remap(b)),
        WireNode::Neg(a) => WireNode::Neg(remap(a)),
        WireNode::Recip(a) => WireNode::Recip(remap(a)),
        WireNode::Sqrt(a) => WireNode::Sqrt(remap(a)),
        WireNode::Sin(a) => WireNode::Sin(remap(a)),
        WireNode::Floor(a) => WireNode::Floor(remap(a)),
        WireNode::Atan2(y, x) => WireNode::Atan2(remap(y), remap(x)),
        WireNode::Exp2(a) => WireNode::Exp2(remap(a)),
        WireNode::Log2(a) => WireNode::Log2(remap(a)),
        WireNode::Select(c, a, b) => WireNode::Select(remap(c), remap(a), remap(b)),
    }
}

/// Auto-partition a graph across mesh nodes.
///
/// Simple heuristic: divide nodes roughly equally among mesh nodes
/// in topological order (since expression graph indices are topological).
pub fn auto_partition(
    graph: &WireGraph,
    mesh: &Mesh,
) -> Result<Vec<GraphPartition>, crate::error::MeshError> {
    let n_nodes = graph.nodes.len();
    let n_workers = mesh.len();

    if n_workers == 0 {
        return Err(crate::error::MeshError::NoWorkers);
    }

    if n_workers == 1 {
        // Single node: no partitioning needed
        return partition_data_parallel(graph, mesh);
    }

    let chunk_size = n_nodes.div_ceil(n_workers);
    let stage_assignments: Vec<(NodeId, Vec<u32>)> = mesh
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(n_nodes);
            let indices: Vec<u32> = (start as u32..end as u32).collect();
            (node.id, indices)
        })
        .collect();

    partition_pipeline(graph, mesh, &stage_assignments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::PROTOCOL_VERSION;

    fn simple_graph() -> WireGraph {
        // x0 + x1, then * x0
        // Indices: 0=Lit(0.0), 1=Lit(1.0), 2=Lit(2.0), 3=Var(0), 4=Var(1), 5=Add(3,4), 6=Mul(5,3)
        WireGraph {
            version: PROTOCOL_VERSION,
            nodes: vec![
                WireNode::Lit(0.0_f64.to_bits()),
                WireNode::Lit(1.0_f64.to_bits()),
                WireNode::Lit(2.0_f64.to_bits()),
                WireNode::Var(0),
                WireNode::Var(1),
                WireNode::Add(3, 4),
                WireNode::Mul(5, 3),
            ],
            outputs: vec![6],
            n_inputs: 2,
        }
    }

    #[test]
    fn data_parallel_gives_full_graph_to_each_node() {
        let graph = simple_graph();

        let parts = partition_data_parallel(&graph, &mock_mesh(3)).unwrap();
        assert_eq!(parts.len(), 3);
        for p in &parts {
            assert_eq!(p.graph.nodes.len(), graph.nodes.len());
            assert!(p.ingress.is_empty());
            assert!(p.egress.is_empty());
        }
    }

    #[test]
    fn wire_node_deps_coverage() {
        assert!(wire_node_deps(WireNode::Var(0)).is_empty());
        assert!(wire_node_deps(WireNode::Lit(0)).is_empty());
        assert_eq!(wire_node_deps(WireNode::Add(1, 2)), vec![1, 2]);
        assert_eq!(wire_node_deps(WireNode::Neg(5)), vec![5]);
        assert_eq!(wire_node_deps(WireNode::Atan2(3, 4)), vec![3, 4]);
    }

    /// Create a mock mesh with N nodes (no real transport).
    fn mock_mesh(n: usize) -> Mesh {
        // We need a way to create a mesh without transport for testing.
        // For now, use a helper that creates nodes with dummy data.
        crate::mesh::Mesh::mock(n)
    }
}
