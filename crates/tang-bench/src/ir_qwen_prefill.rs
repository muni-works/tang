//! Qwen3 written once, as a full-sequence (prefill) graph over a symbolic
//! sequence dim `T`, with its decode step and KV cache derived by
//! [`tang_compile::tensor::step::derive_step`] instead of written by hand
//! as [`super::step_graph`] is.
//!
//! The math is [`super::step_graph`]'s, term for term (4-bit projections,
//! QK-norm, rotate-half rope from the `rope` table, GQA, SwiGLU), with the
//! sequence where the hand graph has its step: token `t` is at position
//! `t`, attention runs over keys `j < T` with every reduction masked as
//! `select(t − j + ½, x, identity)`, and nothing refers to a cache. The K
//! and V nodes are laid out as the hand caches are (`[kvh, T, 2, hd/2]`,
//! `[kvh, hd, T]`), so the derived step reads and writes the same buffers.
use tang::Scalar;
use tang_compile::tensor::cuda_gemv::Packing;
use tang_compile::tensor::step::{
    derive_step_with, Rows, StepGraph, StepInput, StepOptions, StepSpec,
};
use tang_compile::tensor::{Dim, Form, Monoid, Op, TensorGraph, TensorId};
use tang_expr::ExprId;

use super::{arg, form, ident, lit, pick, select, unit, DecodeGraph, InputSpec, QwenDims, GROUP};

/// The symbolic sequence dim.
pub const SEQ: &str = "T";
const T: Dim = Dim::Sym(SEQ);

/// One prefill input: name (as [`super::step_graph`] names it, plus
/// `tokens`), shape (with `T`), packing.
#[derive(Clone, Debug)]
pub struct PrefillInput {
    pub name: String,
    pub shape: Vec<Dim>,
    pub packing: Option<Packing>,
}

pub struct PrefillGraph {
    pub g: TensorGraph,
    pub inputs: Vec<PrefillInput>,
    /// `[T, vocab]`: every token's logits.
    pub logits: TensorId,
    /// Per layer: the roped, QK-normed keys `[kvh, T, 2, hd/2]`.
    pub k: Vec<TensorId>,
    /// Per layer: the values `[kvh, hd, T]`.
    pub v: Vec<TensorId>,
    /// The residual stream after each block (kept materialized, as
    /// [`super::DecodeGraph::keep`]).
    pub keep: Vec<TensorId>,
}

impl PrefillGraph {
    pub fn input(&self, name: &str) -> usize {
        self.inputs
            .iter()
            .position(|s| s.name == name)
            .unwrap_or_else(|| panic!("no input {name}"))
    }
}

struct B {
    g: TensorGraph,
    inputs: Vec<PrefillInput>,
}

impl B {
    fn input(&mut self, name: String, shape: &[Dim], packing: Option<Packing>) -> TensorId {
        self.inputs.push(PrefillInput {
            name,
            shape: shape.to_vec(),
            packing,
        });
        self.g.input(shape)
    }

    fn fixed(&mut self, name: String, shape: &[usize], packing: Option<Packing>) -> TensorId {
        let shape: Vec<Dim> = shape.iter().map(|&n| Dim::Fixed(n)).collect();
        self.input(name, &shape, packing)
    }

    fn q4(&mut self, name: &str, rows: usize, cols: usize) -> [TensorId; 3] {
        let groups = cols / GROUP;
        [
            self.fixed(
                format!("{name}.q"),
                &[rows, groups, GROUP],
                Some(Packing::U4),
            ),
            self.fixed(format!("{name}.s"), &[rows, groups], Some(Packing::Bf16)),
            self.fixed(format!("{name}.b"), &[rows, groups], Some(Packing::Bf16)),
        ]
    }

    /// `y[t, n] = Σ_k x[t, k]·(q[n,k]·s[n,k/64] + b[n,k/64])`.
    fn linear(&mut self, x: TensorId, w: [TensorId; 3]) -> TensorId {
        let shape = self.g.shape(w[0]).to_vec();
        let (rows, groups) = (shape[0].n(), shape[1].n());
        let dq = self
            .g
            .comp(
                &[rows, groups, GROUP],
                &[],
                vec![
                    arg(w[0], ident(3)),
                    arg(w[1], vec![unit(3, 0), unit(3, 1)]),
                    arg(w[2], vec![unit(3, 0), unit(3, 1)]),
                ],
                None,
                |v| vec![v[0] * v[1] + v[2]],
            )
            .unwrap();
        // space (t, n, g, j)
        let xa = arg(
            x,
            vec![unit(4, 0), form(4, &[(2, GROUP as i64), (3, 1)], 0)],
        );
        self.g
            .comp(
                &[T, Dim::Fixed(rows)],
                &[Dim::Fixed(groups), Dim::Fixed(GROUP)],
                vec![xa, arg(dq, vec![unit(4, 1), unit(4, 2), unit(4, 3)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    fn rms_norm(&mut self, x: TensorId, n: usize, w: TensorId, eps: f64) -> TensorId {
        let ss = self
            .g
            .comp(
                &[T],
                &[Dim::Fixed(n)],
                vec![arg(x, ident(2))],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        self.g
            .comp(
                &[T, Dim::Fixed(n)],
                &[],
                vec![
                    arg(x, ident(2)),
                    arg(ss, vec![unit(2, 0)]),
                    arg(w, vec![unit(2, 1)]),
                ],
                None,
                |v| {
                    let inv = ExprId::from_f64(1.0)
                        / (v[1] / ExprId::from_f64(n as f64) + ExprId::from_f64(eps)).sqrt();
                    vec![v[0] * inv * v[2]]
                },
            )
            .unwrap()
    }

    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.g.map(&[a, b], |v| v[0] + v[1]).unwrap()
    }
}

fn dims_of(shape: &[usize]) -> Vec<Dim> {
    shape.iter().map(|&n| Dim::Fixed(n)).collect()
}

/// `shape` with `T` in front: `[T, shape…]`.
fn with_t(shape: &[usize]) -> Vec<Dim> {
    std::iter::once(T).chain(dims_of(shape)).collect()
}

/// The full-sequence graph of [`super::step_graph`]'s model.
pub fn prefill_graph(dims: &QwenDims) -> PrefillGraph {
    let QwenDims {
        d_model: d,
        heads,
        kv_heads: kvh,
        head_dim: hd,
        ff,
        vocab,
        layers,
        ctx,
        eps,
        vocab_chunks,
        prefill_q4,
        ..
    } = *dims;
    assert!(
        vocab_chunks == 1 && !prefill_q4,
        "prefill graph: one vocabulary block, f32 prefill"
    );
    let half = hd / 2;
    let per = heads / kvh;
    let groups = d / GROUP;
    let q_rows = heads * hd;
    let k_off = q_rows as i64;
    let v_off = (q_rows + kvh * hd) as i64;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut b = B {
        g: TensorGraph::new(),
        inputs: Vec::new(),
    };
    let tokens = b.input("tokens".into(), &[T], None);

    // x[t, g, j] = Σ_v pick(v, tok_t, E[v, g, j])
    let ew = b.q4("embed", vocab, d);
    let v_iota = b.g.iota(&with_t(&[groups, GROUP, vocab]), 3);
    let x3 =
        b.g.comp(
            &with_t(&[groups, GROUP]),
            &[Dim::Fixed(vocab)],
            vec![
                arg(ew[0], vec![unit(4, 3), unit(4, 1), unit(4, 2)]),
                arg(ew[1], vec![unit(4, 3), unit(4, 1)]),
                arg(ew[2], vec![unit(4, 3), unit(4, 1)]),
                arg(tokens, vec![unit(4, 0)]),
                arg(v_iota, ident(4)),
            ],
            Some(Monoid::sum()),
            |v| vec![pick(v[4], v[3], v[0] * v[1] + v[2])],
        )
        .unwrap();
    let mut x = b.g.reshape(x3, &with_t(&[d])).unwrap();
    // rope angles of position t from the [cos | sin][ctx, half] table
    let rope = b.fixed("rope".into(), &[2, ctx, half], None);
    let rsp = with_t(&[2, half, ctx]);
    let (rp, rt) = (b.g.iota(&rsp, 3), b.g.iota(&rsp, 0));
    let cs =
        b.g.comp(
            &with_t(&[2, half]),
            &[Dim::Fixed(ctx)],
            vec![
                arg(rope, vec![unit(4, 1), unit(4, 3), unit(4, 2)]),
                arg(rp, ident(4)),
                arg(rt, ident(4)),
            ],
            Some(Monoid::sum()),
            |v| vec![pick(v[1], v[2], v[0])],
        )
        .unwrap();
    let cs = b.g.reshape(cs, &with_t(&[hd])).unwrap();

    let mut ks = Vec::new();
    let mut vs = Vec::new();
    let mut keep = Vec::new();
    for l in 0..layers {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        let qkv_w = b.q4(&p("qkv"), dims.qkv_rows(), d);
        let o_w = b.q4(&p("o"), d, q_rows);
        let gate_w = b.q4(&p("gate"), ff, d);
        let up_w = b.q4(&p("up"), ff, d);
        let down_w = b.q4(&p("down"), d, ff);
        let ln1 = b.fixed(p("ln1"), &[d], None);
        let ln2 = b.fixed(p("ln2"), &[d], None);
        let qn = b.fixed(p("q_norm"), &[hd], None);
        let kn = b.fixed(p("k_norm"), &[hd], None);

        let normed = b.rms_norm(x, d, ln1, eps);
        let qkv = b.linear(normed, qkv_w);
        let nh = heads + kvh;
        let hss =
            b.g.comp(
                &with_t(&[nh]),
                &[Dim::Fixed(hd)],
                vec![arg(
                    qkv,
                    vec![unit(3, 0), form(3, &[(1, hd as i64), (2, 1)], 0)],
                )],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        // QK-norm then rotate-half rope (as `step_graph`), with the token
        // coordinate `t` anywhere in `out`
        #[allow(clippy::too_many_arguments)]
        let roped = |b: &mut B,
                     base: i64,
                     out: &[Dim],
                     t: usize,
                     head: Form,
                     h: usize,
                     i: usize,
                     ss_head: Form,
                     w: TensorId,
                     mul: f64| {
            let rank = out.len();
            let mut self_f = head.clone();
            self_f.coefs[h] += half as i64;
            self_f.coefs[i] += 1;
            self_f.offset += base;
            let mut other = head.clone();
            other.coefs[h] -= half as i64;
            other.coefs[i] += 1;
            other.offset += base + half as i64;
            let w_self = form(rank, &[(h, half as i64), (i, 1)], 0);
            let w_other = form(rank, &[(h, -(half as i64)), (i, 1)], half as i64);
            let iota = b.g.iota(out, h);
            let mut sin_f = unit(rank, i);
            sin_f.offset += half as i64;
            b.g.comp(
                out,
                &[],
                vec![
                    arg(qkv, vec![unit(rank, t), self_f]),
                    arg(qkv, vec![unit(rank, t), other]),
                    arg(w, vec![w_self]),
                    arg(w, vec![w_other]),
                    arg(hss, vec![unit(rank, t), ss_head]),
                    arg(cs, vec![unit(rank, t), unit(rank, i)]),
                    arg(cs, vec![unit(rank, t), sin_f]),
                    arg(iota, ident(rank)),
                ],
                None,
                |v| {
                    let inv = ExprId::from_f64(1.0)
                        / (v[4] / ExprId::from_f64(hd as f64) + ExprId::from_f64(eps)).sqrt();
                    let sign = v[7] * ExprId::from_f64(2.0) - ExprId::from_f64(1.0);
                    let rot = v[0] * v[2] * v[5] + sign * v[1] * v[3] * v[6];
                    vec![rot * inv * ExprId::from_f64(mul)]
                },
            )
            .unwrap()
        };
        // q: [T, kvh, per, 2, half], head kvh·per + r, scaled by 1/√hd
        let q_rot = roped(
            &mut b,
            0,
            &with_t(&[kvh, per, 2, half]),
            0,
            form(5, &[(1, (per * hd) as i64), (2, hd as i64)], 0),
            3,
            4,
            form(5, &[(1, per as i64), (2, 1)], 0),
            qn,
            scale,
        );
        // k: [kvh, T, 2, half] — the hand K cache's layout
        let k_rot = roped(
            &mut b,
            k_off,
            &[Dim::Fixed(kvh), T, Dim::Fixed(2), Dim::Fixed(half)],
            1,
            form(4, &[(0, hd as i64)], 0),
            2,
            3,
            form(4, &[(0, 1)], heads as i64),
            kn,
            1.0,
        );
        // v: [kvh, hd, T] — the hand V cache's layout
        let v =
            b.g.comp(
                &[Dim::Fixed(kvh), Dim::Fixed(hd), T],
                &[],
                vec![arg(
                    qkv,
                    vec![unit(3, 2), form(3, &[(0, hd as i64), (1, 1)], v_off)],
                )],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        ks.push(k_rot);
        vs.push(v);

        let visible = |t: ExprId, j: ExprId| t - j + lit(0.5);
        // scores[t, kvh, r, j] over (h, i): space (t, kvh, r, j, h, i)
        let sp = [
            T,
            Dim::Fixed(kvh),
            Dim::Fixed(per),
            T,
            Dim::Fixed(2),
            Dim::Fixed(half),
        ];
        let (ts, js) = (b.g.iota(&sp, 0), b.g.iota(&sp, 3));
        let scores =
            b.g.comp(
                &sp[..4],
                &sp[4..],
                vec![
                    arg(
                        q_rot,
                        vec![unit(6, 0), unit(6, 1), unit(6, 2), unit(6, 4), unit(6, 5)],
                    ),
                    arg(k_rot, vec![unit(6, 1), unit(6, 3), unit(6, 4), unit(6, 5)]),
                    arg(ts, ident(6)),
                    arg(js, ident(6)),
                ],
                Some(Monoid::sum()),
                |v| vec![select(visible(v[2], v[3]), v[0] * v[1], lit(0.0))],
            )
            .unwrap();
        // softmax statistics over j: space (t, kvh, r, j)
        let sp = [T, Dim::Fixed(kvh), Dim::Fixed(per), T];
        let (tr, jr) = (b.g.iota(&sp, 0), b.g.iota(&sp, 3));
        let m =
            b.g.comp(
                &sp[..3],
                &[T],
                vec![arg(scores, ident(4)), arg(tr, ident(4)), arg(jr, ident(4))],
                Some(Monoid::max()),
                |v| vec![select(visible(v[1], v[2]), v[0], lit(-1e30))],
            )
            .unwrap();
        let lsum =
            b.g.comp(
                &sp[..3],
                &[T],
                vec![
                    arg(scores, ident(4)),
                    arg(tr, ident(4)),
                    arg(jr, ident(4)),
                    arg(m, vec![unit(4, 0), unit(4, 1), unit(4, 2)]),
                ],
                Some(Monoid::sum()),
                |v| vec![select(visible(v[1], v[2]), (v[0] - v[3]).exp(), lit(0.0))],
            )
            .unwrap();
        // attn[t, kvh, r, c] = Σ_j p·v / l: space (t, kvh, r, c, j)
        let sp = [T, Dim::Fixed(kvh), Dim::Fixed(per), Dim::Fixed(hd), T];
        let (to, jo) = (b.g.iota(&sp, 0), b.g.iota(&sp, 4));
        let mlv = vec![unit(5, 0), unit(5, 1), unit(5, 2)];
        let attn =
            b.g.comp(
                &sp[..4],
                &[T],
                vec![
                    arg(scores, vec![unit(5, 0), unit(5, 1), unit(5, 2), unit(5, 4)]),
                    arg(m, mlv.clone()),
                    arg(lsum, mlv),
                    arg(v, vec![unit(5, 1), unit(5, 3), unit(5, 4)]),
                    arg(to, ident(5)),
                    arg(jo, ident(5)),
                ],
                Some(Monoid::sum()),
                |v| {
                    let pv = (v[0] - v[1]).exp() * v[3] / v[2];
                    vec![select(visible(v[4], v[5]), pv, lit(0.0))]
                },
            )
            .unwrap();
        let attn = b.g.reshape(attn, &with_t(&[q_rows])).unwrap();
        let proj = b.linear(attn, o_w);
        let res1 = b.add(x, proj);
        let normed2 = b.rms_norm(res1, d, ln2, eps);
        let gate = b.linear(normed2, gate_w);
        let up = b.linear(normed2, up_w);
        let act =
            b.g.map(&[gate, up], |a| {
                a[0] / (ExprId::from_f64(1.0) + (-a[0]).exp()) * a[1]
            })
            .unwrap();
        let down = b.linear(act, down_w);
        let res2 = b.add(res1, down);
        keep.extend([res1, res2]);
        x = res2;
    }
    let norm = b.fixed("model.norm".into(), &[d], None);
    let lm = b.q4("lm_head", vocab, d);
    let normed = b.rms_norm(x, d, norm, eps);
    let logits = b.linear(normed, lm);
    PrefillGraph {
        g: b.g,
        inputs: b.inputs,
        logits,
        k: ks,
        v: vs,
        keep,
    }
}

/// The derived step of `pg` appending `k` tokens, against caches of
/// `dims.ctx` positions: outputs `[logits, updated caches…]`.
pub fn derived_step(pg: &PrefillGraph, dims: &QwenDims, k: usize) -> StepGraph {
    derived_step_rows(pg, dims, k, Rows::All)
}

/// [`derived_step`] with only the last new row's logits ([`Rows::Last`]):
/// the LM head and the last layer's MLP run for one row, as in the hand
/// prefill step.
pub fn derived_step_rows(pg: &PrefillGraph, dims: &QwenDims, k: usize, rows: Rows) -> StepGraph {
    derive_step_with(
        &pg.g,
        &[pg.logits],
        &StepSpec {
            seq: SEQ,
            new: k,
            capacity: dims.ctx,
        },
        &StepOptions { rows, batch: None },
    )
    .expect("the Qwen prefill graph has a step")
}

/// Name of step input `si` (`pos`, a prefill input's name, or the hand
/// graph's cache name for a cache slot).
pub fn step_input_name(pg: &PrefillGraph, step: &StepGraph, si: StepInput) -> String {
    match si {
        StepInput::Pos => "pos".into(),
        StepInput::New(i) | StepInput::Same(i) => pg.inputs[i].name.clone(),
        StepInput::Cache(c) => {
            let node = step.caches[c].node;
            if let Some(l) = pg.k.iter().position(|&t| t == node) {
                format!("model.layers.{l}.k_cache")
            } else if let Some(l) = pg.v.iter().position(|&t| t == node) {
                format!("model.layers.{l}.v_cache")
            } else {
                panic!("cache slot {c} is neither a K nor a V node: {node:?}")
            }
        }
    }
}

/// The derived step as a [`DecodeGraph`], so the hand graph's passes
/// (`compiled_decode`) and lowering options (`decode_opts`) apply as they
/// are: outputs `[logits, k_cache'₀, v_cache'₀, …]` (no `state`: the
/// step's position is the `pos` input), inputs named as the hand graph
/// names them, packed as it packs them.
pub fn as_decode_graph(pg: &PrefillGraph, step: &StepGraph, dims: &QwenDims) -> DecodeGraph {
    let inputs = step
        .inputs
        .iter()
        .enumerate()
        .map(|(k, &si)| {
            let shape = step
                .graph
                .nodes()
                .iter()
                .find(|n| matches!(n.op, Op::Input(i) if i == k))
                .map(|n| n.shape.iter().map(|d| d.n()).collect())
                .unwrap();
            let packing = match si {
                StepInput::New(i) | StepInput::Same(i) => pg.inputs[i].packing,
                StepInput::Cache(_) => dims.kv_f16.then_some(Packing::F16),
                StepInput::Pos => None,
            };
            InputSpec {
                name: step_input_name(pg, step, si),
                shape,
                packing,
            }
        })
        .collect();
    // cache outputs in the hand graph's order: k₀, v₀, k₁, v₁, …
    let mut caches: Vec<(String, TensorId)> = step
        .caches
        .iter()
        .enumerate()
        .map(|(c, slot)| (step_input_name(pg, step, StepInput::Cache(c)), slot.updated))
        .collect();
    let order = |n: &str| {
        let l: usize = n.split('.').nth(2).unwrap().parse().unwrap();
        2 * l + usize::from(n.ends_with("v_cache"))
    };
    caches.sort_by_key(|(n, _)| order(n));
    let mut outputs = step.outputs.clone();
    outputs.extend(caches.iter().map(|(_, t)| *t));
    // tokens per step (the logits may be the last row only)
    let tokens = step
        .inputs
        .iter()
        .position(|si| matches!(si, StepInput::New(i) if pg.inputs[*i].name == "tokens"))
        .expect("a step reads its tokens");
    let seq = step
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Input(i) if i == tokens))
        .map(|n| n.shape[0].n())
        .unwrap();
    let mut keep: Vec<TensorId> = pg
        .keep
        .iter()
        .filter_map(|t| step.rows.get(t).copied())
        .collect();
    // (a short window, `seq ≤ window_gemv_max()`, is lowered like decode:
    // the dequantization stays inlined into the GEMV that loads the packed
    // words, each word read once for all the window's rows)
    if seq > super::window_gemv_max() {
        // a window reads every weight `seq` times: its dequantization is
        // materialized (stored f16 by `compiled_decode`), as `step_graph`
        // writes it for seq > 1, and `merge_reduce_dims_scaled` reads it
        // as a flat `k`: each projection a tensor-core tile
        let g = &step.graph;
        let u4: Vec<usize> = step
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, si)| {
                matches!(si, StepInput::Same(i) if pg.inputs[*i].packing == Some(Packing::U4))
            })
            .map(|(k, _)| k)
            .collect();
        // (only a weight read by more than one row: the LM head of a
        // last-row window is a GEMV over the packed words, as in the hand
        // step, not a 1.2 GB f16 copy a step)
        let wide_reader = |n: usize| {
            (0..g.nodes().len()).any(|r| {
                g.comp_at(r).is_some_and(|c| {
                    c.args.iter().any(|a| a.src.index() == n)
                        && c.out.first().and_then(|d| d.fixed()).unwrap_or(1) > 1
                })
            })
        };
        keep.extend((0..g.nodes().len()).filter_map(|n| {
            let c = g.comp_at(n)?;
            let reads_u4 = c
                .args
                .iter()
                .any(|a| matches!(g.nodes()[a.src.index()].op, Op::Input(i) if u4.contains(&i)));
            (c.monoid.is_none() && reads_u4 && wide_reader(n)).then(|| TensorId::from_index(n))
        }));
    }
    DecodeGraph {
        g: step.graph.clone(),
        seq,
        kctx: dims.ctx,
        outputs,
        keep,
        logit_chunks: Vec::new(),
        inputs,
        q4: false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{compiled_decode, decode_graph, decode_opts, rope_table, step_graph};
    use super::*;
    use std::collections::HashMap;

    fn tiny() -> QwenDims {
        QwenDims {
            d_model: 128,
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
            ff: 192,
            vocab: 256,
            layers: 2,
            ctx: 16,
            eps: 1e-6,
            theta: 1e4,
            kv_f16: false,
            vocab_chunks: 1,
            prefill_q4: false,
        }
    }

    /// Weights by name (the hand graph's names), deterministic per name.
    fn weights(pg: &PrefillGraph, dims: &QwenDims) -> HashMap<String, Vec<f64>> {
        pg.inputs
            .iter()
            .filter(|s| s.name != "tokens")
            .map(|s| {
                let n: usize = s.shape.iter().map(|d| d.n()).product();
                let mut seed = s.name.bytes().fold(1469598103934665603u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(1099511628211)
                });
                let mut rnd = move || {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed >> 11) as f64 / (1u64 << 53) as f64
                };
                let v: Vec<f64> = if s.name == "rope" {
                    rope_table(dims).iter().map(|&x| x as f64).collect()
                } else {
                    match s.packing {
                        Some(Packing::U4) => (0..n).map(|_| (rnd() * 15.999).floor()).collect(),
                        Some(_) if s.name.ends_with(".s") => {
                            (0..n).map(|_| 0.01 + 0.02 * rnd()).collect()
                        }
                        Some(_) => (0..n).map(|_| -0.1 * (1.0 + 0.1 * rnd())).collect(),
                        None => (0..n).map(|_| 1.0 + 0.2 * rnd()).collect(),
                    }
                };
                (s.name.clone(), v)
            })
            .collect()
    }

    fn tokens(n: usize, vocab: usize) -> Vec<f64> {
        (0..n).map(|i| ((37 * i + 5) % vocab) as f64).collect()
    }

    /// Prefill inputs for the first `t` tokens.
    fn prefill_inputs(
        pg: &PrefillGraph,
        w: &HashMap<String, Vec<f64>>,
        toks: &[f64],
    ) -> Vec<Vec<f64>> {
        pg.inputs
            .iter()
            .map(|s| {
                if s.name == "tokens" {
                    toks.to_vec()
                } else {
                    w[&s.name].clone()
                }
            })
            .collect()
    }

    /// `src` (`[outer, t, inner]` along `axis` of `shape` with `t` rows)
    /// into a `cap`-row buffer, the rest NaN.
    fn pad_rows(src: &[f64], shape: &[usize], axis: usize, cap: usize) -> Vec<f64> {
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let t = shape[axis];
        let mut out = vec![f64::NAN; outer * cap * inner];
        for o in 0..outer {
            for r in 0..t {
                let s = (o * t + r) * inner;
                let d = (o * cap + r) * inner;
                out[d..d + inner].copy_from_slice(&src[s..s + inner]);
            }
        }
        out
    }

    /// Rows `r0..r1` along `axis`.
    fn rows(buf: &[f64], shape: &[usize], axis: usize, r0: usize, r1: usize) -> Vec<f64> {
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out = Vec::new();
        for o in 0..outer {
            for r in r0..r1 {
                let base = (o * shape[axis] + r) * inner;
                out.extend_from_slice(&buf[base..base + inner]);
            }
        }
        out
    }

    /// Worst `|a − b| / max|b|`; NaN only where both are NaN, else ∞.
    fn rel(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let scale = b
            .iter()
            .filter(|x| x.is_finite())
            .fold(0.0f64, |m, x| m.max(x.abs()))
            .max(1e-300);
        a.iter()
            .zip(b)
            .map(|(x, y)| match (x.is_nan(), y.is_nan()) {
                (true, true) => 0.0,
                (false, false) if (x - y).is_finite() => (x - y).abs() / scale,
                _ => f64::INFINITY,
            })
            .fold(0.0, f64::max)
    }

    fn shape_at(pg: &PrefillGraph, t: TensorId, rows: usize) -> Vec<usize> {
        pg.g.shape(t)
            .iter()
            .map(|&d| if d == T { rows } else { d.n() })
            .collect()
    }

    fn t_axis(pg: &PrefillGraph, t: TensorId) -> usize {
        pg.g.shape(t).iter().position(|&d| d == T).unwrap()
    }

    /// Cache buffers (by hand-graph name) after a prefill of `toks`.
    fn prefilled_caches(
        pg: &PrefillGraph,
        dims: &QwenDims,
        w: &HashMap<String, Vec<f64>>,
        toks: &[f64],
    ) -> HashMap<String, Vec<f64>> {
        let p0 = toks.len();
        let g = pg.g.bind(&[(SEQ, p0)]).unwrap();
        let nodes: Vec<TensorId> = pg.k.iter().chain(&pg.v).copied().collect();
        let vals = g.eval(&prefill_inputs(pg, w, toks), &nodes);
        let mut out = HashMap::new();
        for (i, (&t, v)) in nodes.iter().zip(vals).enumerate() {
            let l = i % dims.layers;
            let name = if i < dims.layers {
                format!("model.layers.{l}.k_cache")
            } else {
                format!("model.layers.{l}.v_cache")
            };
            out.insert(
                name,
                pad_rows(&v, &shape_at(pg, t, p0), t_axis(pg, t), dims.ctx),
            );
        }
        out
    }

    fn step_inputs(
        pg: &PrefillGraph,
        step: &StepGraph,
        w: &HashMap<String, Vec<f64>>,
        caches: &HashMap<String, Vec<f64>>,
        new_toks: &[f64],
        p0: usize,
    ) -> Vec<Vec<f64>> {
        step.inputs
            .iter()
            .map(|&si| match si {
                StepInput::Pos => vec![p0 as f64],
                StepInput::New(_) => new_toks.to_vec(),
                StepInput::Same(_) => w[&step_input_name(pg, step, si)].clone(),
                StepInput::Cache(_) => caches[&step_input_name(pg, step, si)].clone(),
            })
            .collect()
    }

    /// The derived cache is exactly the hand graph's: per layer the roped,
    /// QK-normed K and the raw V, in the hand caches' layouts.
    #[test]
    fn derived_cache_is_the_hand_cache() {
        let dims = tiny();
        let pg = prefill_graph(&dims);
        let step = derived_step(&pg, &dims, 1);
        let names: Vec<String> = step
            .caches
            .iter()
            .enumerate()
            .map(|(c, _)| step_input_name(&pg, &step, StepInput::Cache(c)))
            .collect();
        eprintln!("derived cache slots: {names:?}");
        assert_eq!(step.caches.len(), 2 * dims.layers);
        let hand = decode_graph(&dims);
        for (c, slot) in step.caches.iter().enumerate() {
            let shape: Vec<usize> = step.graph.shape(slot.input).iter().map(|d| d.n()).collect();
            let h = &hand.inputs[hand.input(&names[c])];
            assert_eq!(shape, h.shape, "{}", names[c]);
        }
    }

    /// Prefill to `p0`, then derived steps (NaN in unfilled cache rows),
    /// each against a prefill of the grown sequence: the new rows' logits
    /// and every filled cache row.
    #[test]
    fn derived_step_matches_prefill() {
        let dims = tiny();
        let pg = prefill_graph(&dims);
        let w = weights(&pg, &dims);
        let all = tokens(dims.ctx, dims.vocab);
        for ks in [&[1usize][..], &[4], &[4, 1, 1]] {
            let mut p0 = 5;
            let mut caches = prefilled_caches(&pg, &dims, &w, &all[..p0]);
            let mut worst = 0.0f64;
            for &k in ks {
                let step = derived_step(&pg, &dims, k);
                let inputs = step_inputs(&pg, &step, &w, &caches, &all[p0..p0 + k], p0);
                let mut ids = step.outputs.clone();
                ids.extend(step.caches.iter().map(|c| c.updated));
                let got = step.graph.eval(&inputs, &ids);
                assert!(got[0].iter().all(|x| x.is_finite()), "logits not finite");

                let grown = p0 + k;
                let fg = pg.g.bind(&[(SEQ, grown)]).unwrap();
                let mut rids = vec![pg.logits];
                rids.extend(step.caches.iter().map(|c| c.node));
                let want = fg.eval(&prefill_inputs(&pg, &w, &all[..grown]), &rids);
                let lw = rows(&want[0], &shape_at(&pg, pg.logits, grown), 0, p0, grown);
                worst = worst.max(rel(&got[0], &lw));
                for (c, slot) in step.caches.iter().enumerate() {
                    let ax = slot.axis;
                    let g_rows = rows(
                        &got[1 + c],
                        &shape_at(&pg, slot.node, dims.ctx),
                        ax,
                        0,
                        grown,
                    );
                    let w_rows = rows(&want[1 + c], &shape_at(&pg, slot.node, grown), ax, 0, grown);
                    worst = worst.max(rel(&g_rows, &w_rows));
                    caches.insert(
                        step_input_name(&pg, &step, StepInput::Cache(c)),
                        got[1 + c].clone(),
                    );
                }
                p0 = grown;
            }
            eprintln!("qwen tiny: derived steps {ks:?} from 5 vs prefill: max rel err {worst:.2e}");
            assert!(worst < 1e-12, "{ks:?}: {worst:e}");
        }
    }

    /// The derived step against the hand-written PR #56 graphs on the same
    /// weights and caches: decode (`decode_graph`, k = 1) and a 4-token
    /// prefill chunk (`step_graph(dims, 4, ctx)`) continuing at `p0`.
    /// Logits (the hand graphs give the last token's) and the whole
    /// updated cache buffers, NaN tail included.
    #[test]
    fn derived_step_matches_hand_step() {
        let dims = tiny();
        let pg = prefill_graph(&dims);
        let w = weights(&pg, &dims);
        let all = tokens(dims.ctx, dims.vocab);
        let p0 = 5;
        let caches = prefilled_caches(&pg, &dims, &w, &all[..p0]);
        for k in [1usize, 4] {
            let step = derived_step(&pg, &dims, k);
            let inputs = step_inputs(&pg, &step, &w, &caches, &all[p0..p0 + k], p0);
            let mut ids = step.outputs.clone();
            ids.extend(step.caches.iter().map(|c| c.updated));
            let got = step.graph.eval(&inputs, &ids);
            assert!(got[0].iter().all(|x| x.is_finite()), "logits not finite");

            let hand = if k == 1 {
                decode_graph(&dims)
            } else {
                step_graph(&dims, k, dims.ctx)
            };
            let hin: Vec<Vec<f64>> = hand
                .inputs
                .iter()
                .map(|s| match s.name.as_str() {
                    "state" => vec![all[p0], p0 as f64],
                    "chunk" => all[p0..p0 + k].iter().copied().chain([p0 as f64]).collect(),
                    n if n.ends_with("cache") => caches[n].clone(),
                    n => w[n].clone(),
                })
                .collect();
            let hres = hand.g.eval(&hin, &hand.outputs);
            let vocab = dims.vocab;
            let last = &got[0][(k - 1) * vocab..k * vocab];
            let mut worst = rel(last, &hres[0]);
            for (c, _) in step.caches.iter().enumerate() {
                let name = step_input_name(&pg, &step, StepInput::Cache(c));
                let l: usize = name.split('.').nth(2).unwrap().parse().unwrap();
                let hi = if name.ends_with("k_cache") {
                    2 + 2 * l
                } else {
                    3 + 2 * l
                };
                worst = worst.max(rel(&got[1 + c], &hres[hi]));
            }
            eprintln!("qwen tiny: derived k={k} vs hand step at p0={p0}: max rel err {worst:.2e}");
            assert!(worst < 1e-12, "k={k}: {worst:e}");
        }
    }

    /// The derived step survives the decode passes (interpreter: same
    /// values) and lowers through `to_cuda_decode` to CUDA source, kernel
    /// for kernel like the hand graph.
    #[test]
    fn derived_step_lowers_to_cuda() {
        let dims = tiny();
        let pg = prefill_graph(&dims);
        let w = weights(&pg, &dims);
        let all = tokens(dims.ctx, dims.vocab);
        let p0 = 5;
        let caches = prefilled_caches(&pg, &dims, &w, &all[..p0]);
        for k in [1usize, 4] {
            let step = derived_step(&pg, &dims, k);
            let dg = as_decode_graph(&pg, &step, &dims);
            let (g, outs) = compiled_decode(&dg, true);
            let opts = decode_opts(&dg, &outs);
            let inputs = step_inputs(&pg, &step, &w, &caches, &all[p0..p0 + k], p0);
            let n = 1 + step.caches.len();
            let want = step.graph.eval(&inputs, &dg.outputs);
            let got = g.eval(&inputs, &outs[..n]);
            let e = want
                .iter()
                .zip(&got)
                .map(|(a, b)| rel(b, a))
                .fold(0.0, f64::max);
            let p = g.to_cuda_decode(&outs, &opts);
            let native = p.native.iter().filter(|x| **x).count();
            for (kern, nat) in p.program.kernels.iter().zip(&p.native) {
                if !nat {
                    let nd = &g.nodes()[kern.node.index()];
                    let what = match g.comp_at(kern.node.index()) {
                        Some(c) => format!(
                            "red {:?}, monoid {:?}, {} args, reads {:?}",
                            c.red,
                            c.monoid.as_ref().map(|m| m.kind()),
                            c.args.len(),
                            kern.reads
                                .iter()
                                .map(|r| (
                                    g.nodes()[r.index()].shape.clone(),
                                    matches!(
                                        g.nodes()[r.index()].op,
                                        tang_compile::tensor::Op::Input(_)
                                    )
                                ))
                                .collect::<Vec<_>>()
                        ),
                        None => "not a comp".into(),
                    };
                    eprintln!(
                        "k={k} transpiled: node {:?} {:?} {what}",
                        kern.node, nd.shape
                    );
                }
            }
            let inplace = p
                .program
                .kernels
                .iter()
                .filter(|k| k.source.contains("in-place"))
                .count();
            let mma = p
                .program
                .kernels
                .iter()
                .filter(|k| k.source.contains("mma"))
                .count();
            let hand = if k == 1 {
                decode_graph(&dims)
            } else {
                step_graph(&dims, k, dims.ctx)
            };
            let (hg, houts) = compiled_decode(&hand, true);
            let hp = hg.to_cuda_decode(&houts, &decode_opts(&hand, &houts));
            let hnative = hp.native.iter().filter(|x| **x).count();
            let hinplace = hp
                .program
                .kernels
                .iter()
                .filter(|k| k.source.contains("in-place"))
                .count();
            let hmma = hp
                .program
                .kernels
                .iter()
                .filter(|k| k.source.contains("mma"))
                .count();
            let same = shared_sources(&p, &hp);
            eprintln!(
                "k={k}: passes vs derived max rel err {e:.2e}; CUDA kernels derived {} ({native} native, {inplace} in-place, {mma} mma) vs hand {} ({hnative} native, {hinplace} in-place, {hmma} mma); {same} derived sources byte-identical to a hand kernel",
                p.native.len(),
                hp.native.len()
            );
            assert!(e < 1e-12, "passes changed values: {e:e}");
            // every cache update is an in-place predicated kernel
            assert_eq!(inplace, 2 * dims.layers, "in-place kernels");
            if k == 1 {
                // decode: every kernel has the native decode lowering
                assert_eq!(native, p.native.len(), "transpiled decode kernels");
            } else if k <= super::super::window_gemv_max() {
                // a short window is lowered like decode: no tiles, and no
                // kernel writes a dequantized weight (the GEMVs read the
                // packed words, the window's rows in registers)
                assert_eq!(mma, 0, "mma tiles in a short window");
                assert!(
                    !p.program
                        .kernels
                        .iter()
                        .any(|kn| kn.source.contains("map over packed")),
                    "a short window materializes a dequantized weight"
                );
            } else {
                // a window: every projection a tensor-core tile, as the hand
                // prefill's, plus the LM head (logits of all k rows, which a
                // draft verify needs; the hand step computes the last row's)
                let shapes: Vec<_> = p
                    .program
                    .kernels
                    .iter()
                    .filter(|kn| kn.source.contains("mma"))
                    .map(|kn| g.nodes()[kn.node.index()].shape.clone())
                    .collect();
                let hshapes: Vec<_> = hp
                    .program
                    .kernels
                    .iter()
                    .filter(|kn| kn.source.contains("mma"))
                    .map(|kn| hg.nodes()[kn.node.index()].shape.clone())
                    .collect();
                let mut want = hshapes.clone();
                want.push(vec![Dim::Fixed(k), Dim::Fixed(dims.vocab)]);
                assert_eq!(shapes, want, "mma tiles");
            }
        }
    }

    /// A 4-token window with only the last row's logits: equal to the
    /// prefill's last row and to the hand prefill step's logits, with as
    /// many kernels as the hand step and a subset of its mma tiles.
    #[test]
    fn last_row_window_matches_hand() {
        let dims = tiny();
        let pg = prefill_graph(&dims);
        let w = weights(&pg, &dims);
        let all = tokens(dims.ctx, dims.vocab);
        let (p0, k) = (5, 4);
        let caches = prefilled_caches(&pg, &dims, &w, &all[..p0]);
        let step = derived_step_rows(&pg, &dims, k, Rows::Last);
        assert_eq!(step.graph.shape(step.outputs[0])[0].n(), 1);
        let inputs = step_inputs(&pg, &step, &w, &caches, &all[p0..p0 + k], p0);
        let mut ids = step.outputs.clone();
        ids.extend(step.caches.iter().map(|c| c.updated));
        let got = step.graph.eval(&inputs, &ids);
        let fg = pg.g.bind(&[(SEQ, p0 + k)]).unwrap();
        let want = fg.eval(&prefill_inputs(&pg, &w, &all[..p0 + k]), &[pg.logits]);
        let lw = rows(
            &want[0],
            &shape_at(&pg, pg.logits, p0 + k),
            0,
            p0 + k - 1,
            p0 + k,
        );
        let e = rel(&got[0], &lw);
        let hand = step_graph(&dims, k, dims.ctx);
        let hin: Vec<Vec<f64>> = hand
            .inputs
            .iter()
            .map(|s| match s.name.as_str() {
                "chunk" => all[p0..p0 + k].iter().copied().chain([p0 as f64]).collect(),
                "state" => vec![0.0, 0.0],
                n if n.ends_with("cache") => caches[n].clone(),
                n => w[n].clone(),
            })
            .collect();
        let hres = hand.g.eval(&hin, &hand.outputs[..1]);
        let eh = rel(&got[0], &hres[0]);
        let dg = as_decode_graph(&pg, &step, &dims);
        assert_eq!(dg.seq, k);
        let (g, outs) = compiled_decode(&dg, true);
        let p = g.to_cuda_decode(&outs, &decode_opts(&dg, &outs));
        let (hg, houts) = compiled_decode(&hand, true);
        let hp = hg.to_cuda_decode(&houts, &decode_opts(&hand, &houts));
        let tiles = |p: &tang_compile::tensor::cuda_gemv::DecodeProgram, g: &TensorGraph| {
            p.program
                .kernels
                .iter()
                .filter(|kn| kn.source.contains("mma"))
                .map(|kn| g.nodes()[kn.node.index()].shape.clone())
                .collect::<Vec<_>>()
        };
        eprintln!(
            "last-row window k={k}: vs prefill {e:.2e}, vs hand {eh:.2e}; kernels {} ({} mma) vs hand {} ({} mma)",
            p.program.kernels.len(),
            tiles(&p, &g).len(),
            hp.program.kernels.len(),
            tiles(&hp, &hg).len()
        );
        assert!(e < 1e-12 && eh < 1e-12, "{e:e} {eh:e}");
        // the hand step's tiles, less the last layer's o, gate|up and down:
        // those run for the last row only here (the hand step computes all
        // k rows and keeps the last); no LM-head tile either
        let ht = tiles(&hp, &hg);
        if k <= super::super::window_gemv_max() {
            // (lowered like decode: no tiles)
            assert!(tiles(&p, &g).is_empty());
        } else {
            assert_eq!(tiles(&p, &g), ht[..ht.len() - 3]);
        }
    }

    /// Derived kernels whose (canonical) source is one of the hand
    /// program's, counted with multiplicity.
    fn shared_sources(
        a: &tang_compile::tensor::cuda_gemv::DecodeProgram,
        b: &tang_compile::tensor::cuda_gemv::DecodeProgram,
    ) -> usize {
        let mut pool: HashMap<&str, usize> = HashMap::new();
        for k in &b.program.kernels {
            *pool.entry(k.source.as_str()).or_default() += 1;
        }
        a.program
            .kernels
            .iter()
            .filter(|k| {
                pool.get_mut(k.source.as_str()).is_some_and(|n| {
                    *n > 0 && {
                        *n -= 1;
                        true
                    }
                })
            })
            .count()
    }

    /// Qwen3-8B shapes (36 layers, ctx 4352, f16 caches): the derived
    /// decode step lowers to CUDA source like the hand one. No weights,
    /// no device. `cargo test -p tang-bench --lib -- --ignored qwen3_8b`
    #[test]
    #[ignore]
    fn derived_step_lowers_qwen3_8b_shapes() {
        let dims = QwenDims::qwen3_8b(4352);
        let t0 = std::time::Instant::now();
        let pg = prefill_graph(&dims);
        let step = derived_step(&pg, &dims, 1);
        let t_derive = t0.elapsed();
        assert_eq!(step.caches.len(), 2 * dims.layers);
        let dg = as_decode_graph(&pg, &step, &dims);
        let (g, outs) = compiled_decode(&dg, true);
        let p = g.to_cuda_decode(&outs, &decode_opts(&dg, &outs));
        let t_all = t0.elapsed();
        let hand = decode_graph(&dims);
        let (hg, houts) = compiled_decode(&hand, true);
        let hp = hg.to_cuda_decode(&houts, &decode_opts(&hand, &houts));
        let count = |p: &tang_compile::tensor::cuda_gemv::DecodeProgram| {
            (p.native.len(), p.native.iter().filter(|x| **x).count())
        };
        let same = shared_sources(&p, &hp);
        eprintln!(
            "qwen3-8b decode: derive {t_derive:.2?}, derive+passes+codegen {t_all:.2?}; kernels derived {:?} vs hand {:?} (total, native); {same} derived kernel sources byte-identical to a hand kernel",
            count(&p),
            count(&hp)
        );
        assert_eq!(p.native.iter().filter(|x| !**x).count(), 0);
        // a 4-token window, last row only, against the hand prefill step
        let mma = |p: &tang_compile::tensor::cuda_gemv::DecodeProgram| {
            p.program
                .kernels
                .iter()
                .filter(|kn| kn.source.contains("mma"))
                .count()
        };
        let step = derived_step_rows(&pg, &dims, 4, Rows::Last);
        let dg = as_decode_graph(&pg, &step, &dims);
        let (g, outs) = compiled_decode(&dg, true);
        let p = g.to_cuda_decode(&outs, &decode_opts(&dg, &outs));
        let hand = step_graph(&dims, 4, dims.ctx);
        let (hg, houts) = compiled_decode(&hand, true);
        let hp = hg.to_cuda_decode(&houts, &decode_opts(&hand, &houts));
        eprintln!(
            "qwen3-8b window 4, last row: kernels derived {} ({} mma) vs hand {} ({} mma)",
            p.program.kernels.len(),
            mma(&p),
            hp.program.kernels.len(),
            mma(&hp)
        );
        assert_eq!(mma(&p), mma(&hp) - 3);
    }

    /// Every CUDA kernel of the derived steps (tiny k = 1 and 4, Qwen3-8B
    /// decode) compiled to PTX by nvrtc: no device needed, only libnvrtc.
    /// `cargo test -p tang-bench --features cuda --lib -- --ignored nvrtc`
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn derived_step_kernels_compile_with_nvrtc() {
        let arch = "sm_86";
        let mut progs = Vec::new();
        for (dims, k, rows) in [
            (tiny(), 1, Rows::All),
            (tiny(), 4, Rows::All),
            (tiny(), 4, Rows::Last),
            (QwenDims::qwen3_8b(4352), 1, Rows::All),
            (QwenDims::qwen3_8b(4352), 4, Rows::All),
            (QwenDims::qwen3_8b(4352), 4, Rows::Last),
        ] {
            let pg = prefill_graph(&dims);
            let step = derived_step_rows(&pg, &dims, k, rows);
            let dg = as_decode_graph(&pg, &step, &dims);
            let (g, outs) = compiled_decode(&dg, true);
            let p = g.to_cuda_decode(&outs, &decode_opts(&dg, &outs));
            progs.push((dims.layers, k, rows, p));
        }
        for (layers, k, rows, p) in &progs {
            let mut distinct: Vec<&str> = p
                .program
                .kernels
                .iter()
                .map(|k| k.source.as_str())
                .collect();
            distinct.sort_unstable();
            distinct.dedup();
            for src in &distinct {
                let ptx = tang_gpu::cuda::ptx(src, arch);
                assert!(ptx.contains(".entry"), "no entry point");
            }
            let mma = p
                .program
                .kernels
                .iter()
                .filter(|kn| kn.source.contains("mma"))
                .count();
            eprintln!(
                "{layers} layers, k={k} ({rows:?} rows): {} kernels ({} distinct sources, {mma} mma) compiled to {arch} PTX",
                p.program.kernels.len(),
                distinct.len()
            );
        }
    }

    /// A fingerprint of the hand graphs' CUDA sources (tiny and Qwen3-8B
    /// shapes; decode, 6- and 512-token prefill, OpenCL q4 prefill), to
    /// show a pass change leaves them byte-identical.
    #[test]
    #[ignore]
    fn hand_sources_fingerprint() {
        use std::hash::{Hash, Hasher};
        let q4 = QwenDims {
            prefill_q4: true,
            ff: 40 * GROUP,
            ..tiny()
        };
        for (name, dims, seq) in [
            ("tiny decode", tiny(), 1),
            ("tiny prefill 6", tiny(), 6),
            ("8b decode", QwenDims::qwen3_8b(4352), 1),
            ("8b prefill 512", QwenDims::qwen3_8b(4352), 512),
            ("tiny q4 prefill 6 (opencl)", q4, 6),
        ] {
            let dg = step_graph(&dims, seq, dims.ctx);
            let (g, outs) = compiled_decode(&dg, !dims.prefill_q4);
            let opts = decode_opts(&dg, &outs);
            let p = if dims.prefill_q4 {
                g.to_opencl_decode(&outs, &opts)
            } else {
                g.to_cuda_decode(&outs, &opts)
            };
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for k in &p.program.kernels {
                k.source.hash(&mut h);
            }
            eprintln!(
                "fingerprint {name}: {} kernels, {:016x}",
                p.program.kernels.len(),
                h.finish()
            );
        }
    }
}
