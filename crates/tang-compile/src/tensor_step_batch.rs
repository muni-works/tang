//! [`batch_graph`]: a prefill graph written for one sequence, with a
//! leading batch dim of independent sessions added to every node that
//! carries the sequence, so [`derive_step_with`](super::derive_step_with)
//! with [`StepOptions::batch`](super::StepOptions) steps sessions at
//! different lengths together.

use super::super::{err, Arg, Comp, Dim, Form, Op, ShapeError, TensorGraph, TensorId, View};
use super::super::{Scan, ScanKind, Segment};

/// `g` with a leading dim `batch` on every node that has the sequence dim
/// `seq` or depends on one that does (weights and iotas stay as they are).
/// Each comp of such a node gets the batch coordinate in front of its
/// space; reads of batched sources pick it, every other coordinate shifts
/// by one. Returns the graph and `outputs`' new ids. Inputs keep their
/// order: a sequence input becomes `[batch, …]`. A scan keeps its
/// `DeltaForm`.
pub fn batch_graph(
    g: &TensorGraph,
    outputs: &[TensorId],
    seq: &'static str,
    batch: &'static str,
) -> Result<(TensorGraph, Vec<TensorId>), ShapeError> {
    let s = Dim::Sym(seq);
    let bd = Dim::Sym(batch);
    let n = g.nodes.len();
    let mut ng = TensorGraph::new();
    let mut map: Vec<TensorId> = Vec::with_capacity(n);
    let mut batched = vec![false; n];
    // a form over the old space as a form over the batched one
    let shift = |f: &Form| {
        let mut c = vec![0];
        c.extend_from_slice(&f.coefs);
        Form {
            coefs: c,
            offset: f.offset,
        }
    };
    for i in 0..n {
        let node = &g.nodes[i];
        let deps = g.deps(i);
        let b = !matches!(node.op, Op::Iota(_))
            && (node.shape.contains(&s) || deps.iter().any(|&d| batched[d]));
        batched[i] = b;
        let with_b = |shape: &[Dim]| -> Vec<Dim> {
            let mut v = vec![bd];
            v.extend_from_slice(shape);
            v
        };
        let t = match &node.op {
            Op::Input(_) => {
                if b {
                    ng.input(&with_b(&node.shape))
                } else {
                    ng.input(&node.shape)
                }
            }
            Op::Iota(d) => ng.iota(&node.shape, *d),
            Op::Reshape(src) => {
                if b != batched[src.n()] {
                    return err(format!("batch_graph: reshape {i} batches its source"));
                }
                let shape = if b {
                    with_b(&node.shape)
                } else {
                    node.shape.clone()
                };
                ng.reshape(map[src.n()], &shape)?
            }
            Op::Comp(c) => {
                if !b {
                    let args = c
                        .args
                        .iter()
                        .map(|a| Arg {
                            src: map[a.src.n()],
                            view: a.view.clone(),
                        })
                        .collect();
                    ng.push_comp(Comp { args, ..c.clone() })?
                } else {
                    let rank = c.rank() + 1;
                    let args = c
                        .args
                        .iter()
                        .map(|a| {
                            let mut forms: Vec<Form> = a.view.forms.iter().map(shift).collect();
                            let mut gather = a.view.gather.clone();
                            if batched[a.src.n()] {
                                forms.insert(0, Form::unit(rank, 0));
                                if let Some(ga) = &mut gather {
                                    ga.coefs.insert(0, 0);
                                }
                            }
                            Arg {
                                src: map[a.src.n()],
                                view: View {
                                    forms,
                                    guard: a.view.guard,
                                    gather,
                                },
                            }
                        })
                        .collect();
                    ng.push_comp(Comp {
                        out: with_b(&c.out),
                        args,
                        ..c.clone()
                    })?
                }
            }
            Op::Scan(sc) => {
                let ins: Vec<TensorId> = sc.inputs.iter().map(|t| map[t.n()]).collect();
                let dim = sc.dim + usize::from(b);
                if b && sc.inputs.iter().any(|t| !batched[t.n()]) {
                    return err(format!("batch_graph: scan {i} mixes batched inputs"));
                }
                let Scan {
                    exclusive, reverse, ..
                } = *sc;
                match &sc.kind {
                    ScanKind::Monoid(m) => ng.scan(&ins, dim, m.clone(), exclusive, reverse)?,
                    // the delta rule's structure travels with it (its
                    // inputs keep their order): the step folds the cached
                    // state in `O(n·m)` instead of the dense `A·S`
                    ScanKind::Affine {
                        ring,
                        bands: None,
                        form,
                    } => ng.affine_scan(
                        ins.clone(),
                        dim,
                        exclusive,
                        reverse,
                        form.clone(),
                        *ring,
                        None,
                    )?,
                    ScanKind::Affine {
                        ring,
                        bands: Some(bands),
                        ..
                    } => ng.semiring_scan_banded(ins[0], ins[1], bands, dim, reverse, *ring)?,
                    ScanKind::Delta(_) | ScanKind::Attend(_) => {
                        return err(format!("batch_graph: scan {i} is a fused step"))
                    }
                }
            }
            Op::Operator(_) => return err("batch_graph: operator nodes are not batched"),
        };
        // the schedule and data facts travel with the node
        let mut tn = node.clone();
        tn.op = ng.nodes[t.n()].op.clone();
        tn.shape = ng.nodes[t.n()].shape.clone();
        tn.segment = tn.segment.map(|sg| Segment {
            key: map[sg.key.n()],
            ..sg
        });
        ng.nodes[t.n()] = tn;
        map.push(t);
    }
    Ok((ng, outputs.iter().map(|o| map[o.n()]).collect()))
}
