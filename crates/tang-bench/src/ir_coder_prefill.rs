//! Qwen3-Coder-Next (GGUF architecture `qwen3next`) written once, as a
//! full-sequence (prefill) graph over a symbolic sequence dim `T`.
//!
//! The math is `tang_llm::flash::reference`'s `forward_plain` term for
//! term — the plain f32 CPU forward every coder-next kernel is checked
//! against — and the tests run both on the same random weights (a tiny
//! GGUF written in the test):
//!
//! ```text
//! r = embed(tokens)
//! for l:  r += mixer_l(rmsnorm(r)·attn_norm)          mixer: GDN or gated attention
//!         r += moe_l(rmsnorm(r)·post_attention_norm)
//! logits = output · (rmsnorm(r)·output_norm)
//! ```
//!
//! - **GDN** (gated delta net): `qkv`, `z`, fused `ba` projections; a
//!   causal depthwise conv of `d_conv` taps over time (zero history) then
//!   SiLU; q, k L2-normed per head (eps on the squared norm); v head
//!   `(g, j)` reads k head `g` (grouped); decay `α = exp(softplus(a + dt)·A)`,
//!   `β = sigmoid(b)`; the state is [`TensorGraph::delta_scan`] along `T`
//!   (`S_t = α(I − β k kᵀ) S_{t−1} + β k vᵀ`) read out by
//!   [`TensorGraph::scan_readout`] with q scaled by `1/√S`; then
//!   `rmsnorm(o)·ssm_norm·silu(z)` per head and `ssm_out`.
//! - **Attention**: q and its gate interleaved per head (`[q | gate]`),
//!   QK rmsnorm, NeoX rope on the first `n_rot` dims of each head (partial
//!   rotary), GQA, causal softmax, output × `sigmoid(gate)`, `attn_output`.
//!   The rotation partner is a guarded read at `i ± n_rot/2` on the head
//!   axis (a plain read along `T`); position comes from an iota (the
//!   derived step's `iota + p0`).
//! - **MoE**: router softmax over all experts, top-k by one
//!   [`Monoid::topk`] reduction, weights renormalized with the `2⁻¹⁴`
//!   clamp, each picked expert's SwiGLU read through a gathered view of
//!   the fused `[n_expert, …]` weights (index = the top-k slot), plus the
//!   sigmoid-gated shared expert.
//! - **Embedding**: a gathered view of the table by token id.
//!
//! Every reduction over `T` (attention) is masked as `select(t − j + ½, x,
//! identity)`. The GDN conv is a guarded shifted read along `T` and the
//! GDN state a scan along `T`: [`derive_step`] refuses both today (see
//! the `derive_step_*` tests and the gap list in the PR).
//!
//! Inputs are named and laid out as the GGUF tensors are (row-major, the
//! reverse of ggml's `ne` order: a `[cols, rows]` ggml matrix is
//! `[rows, cols]` here), so a GGUF's f32 data feeds the graph as is.
use tang::Scalar;
use tang_compile::tensor::step::{derive_step, StepGraph, StepSpec};
use tang_compile::tensor::{Arg, Dim, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

/// The symbolic sequence dim.
pub const SEQ: &str = "T";
const T: Dim = Dim::Sym(SEQ);

/// Model geometry, as `tang_llm::flash::reference::Hparams` reads it from
/// `qwen3next.*` GGUF metadata.
#[derive(Clone, Debug)]
pub struct CoderDims {
    pub d_model: usize,
    pub vocab: usize,
    /// Per layer: GDN (`true`) or full attention (`false`).
    pub recurrent: Vec<bool>,
    pub eps: f64,
    // attention
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f64,
    // GDN
    pub ssm_d_conv: usize,
    pub ssm_d_state: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    // MoE
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
}

impl CoderDims {
    /// Three layers (GDN, attention, GDN), d = 32, 6 experts top-2,
    /// partial rotary (4 of 8 head dims), GDN 2 k heads × 2 v heads each.
    pub fn tiny() -> Self {
        Self {
            d_model: 32,
            vocab: 64,
            recurrent: vec![true, false, true],
            eps: 1e-6,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 8,
            n_rot: 4,
            rope_base: 1e4,
            ssm_d_conv: 4,
            ssm_d_state: 8,
            ssm_k_heads: 2,
            ssm_v_heads: 4,
            n_expert: 6,
            n_expert_used: 2,
            n_ff_exp: 16,
            n_ff_shexp: 24,
        }
    }

    /// GDN conv channels: `[q | k | v]`.
    pub fn conv_channels(&self) -> usize {
        2 * self.ssm_d_state * self.ssm_k_heads + self.ssm_d_state * self.ssm_v_heads
    }
}

/// One graph input: GGUF tensor name (or `tokens`), row-major shape.
#[derive(Clone, Debug)]
pub struct CoderInput {
    pub name: String,
    pub shape: Vec<Dim>,
}

pub struct CoderPrefill {
    pub g: TensorGraph,
    pub inputs: Vec<CoderInput>,
    /// `[T, vocab]`: every position's logits.
    pub logits: TensorId,
    /// Per attention layer: the normed, roped keys `[T, kv heads, hd]`.
    pub k: Vec<TensorId>,
    /// Per attention layer: the values `[T, kv heads · hd]`.
    pub v: Vec<TensorId>,
    /// Per GDN layer: the state scan `[T, k heads, v per k, S, S]`.
    pub gdn_state: Vec<TensorId>,
}

impl CoderPrefill {
    pub fn input(&self, name: &str) -> usize {
        self.inputs
            .iter()
            .position(|s| s.name == name)
            .unwrap_or_else(|| panic!("no input {name}"))
    }
}

fn unit(rank: usize, d: usize) -> Form {
    Form::unit(rank, d)
}

fn form(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::constant(rank, offset);
    for &(d, c) in terms {
        f.coefs[d] += c;
    }
    f
}

fn ident(rank: usize) -> Vec<Form> {
    (0..rank).map(|d| unit(rank, d)).collect()
}

fn arg(src: TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View {
            gather: None,
            guard: false,
            forms,
        },
    }
}

fn select(c: ExprId, t: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(c, t, f))
}

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

fn sigmoid(x: ExprId) -> ExprId {
    lit(1.0) / (lit(1.0) + (-x).exp())
}

fn silu(x: ExprId) -> ExprId {
    x * sigmoid(x)
}

/// ggml's softplus: `log(1 + eˣ)`, passed through above 20.
fn softplus(x: ExprId) -> ExprId {
    select(x - lit(20.0), x, (lit(1.0) + x.exp()).ln())
}

struct B {
    g: TensorGraph,
    inputs: Vec<CoderInput>,
    eps: f64,
}

impl B {
    fn input(&mut self, name: String, shape: &[Dim]) -> TensorId {
        self.inputs.push(CoderInput {
            name,
            shape: shape.to_vec(),
        });
        self.g.input(shape)
    }

    fn fixed(&mut self, name: String, shape: &[usize]) -> TensorId {
        let shape: Vec<Dim> = shape.iter().map(|&n| Dim::Fixed(n)).collect();
        self.input(name, &shape)
    }

    /// `y[t, r] = Σ_c x[t, c]·w[r, c]`.
    fn linear(&mut self, x: TensorId, w: TensorId) -> TensorId {
        let ws = self.g.shape(w).to_vec();
        let (rows, cols) = (ws[0], ws[1]);
        self.g
            .comp(
                &[T, rows],
                &[cols],
                vec![
                    arg(x, vec![unit(3, 0), unit(3, 2)]),
                    arg(w, vec![unit(3, 1), unit(3, 2)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    /// `x / rms(x) · w` over the last dim of `[T, n]`.
    fn rms_norm(&mut self, x: TensorId, w: TensorId) -> TensorId {
        let n = self.g.shape(x)[1].n();
        let eps = self.eps;
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
                    let inv = lit(1.0) / (v[1] / lit(n as f64) + lit(eps)).sqrt();
                    vec![v[0] * inv * v[2]]
                },
            )
            .unwrap()
    }

    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.g.map(&[a, b], |v| v[0] + v[1]).unwrap()
    }

    /// `Σ_{i<n} src[t, base + h·stride + i]²` for each of `heads` heads.
    fn head_ss(
        &mut self,
        src: TensorId,
        heads: usize,
        n: usize,
        stride: usize,
        base: i64,
    ) -> TensorId {
        self.g
            .comp(
                &[T, Dim::Fixed(heads)],
                &[Dim::Fixed(n)],
                vec![arg(
                    src,
                    vec![unit(3, 0), form(3, &[(1, stride as i64), (2, 1)], base)],
                )],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap()
    }
}

/// The full-sequence graph of coder-next at `dims`.
pub fn prefill_graph(dims: &CoderDims) -> CoderPrefill {
    let d = dims.d_model;
    let mut b = B {
        g: TensorGraph::new(),
        inputs: Vec::new(),
        eps: dims.eps,
    };
    let tokens = b.input("tokens".into(), &[T]);
    let embd = b.fixed("token_embd.weight".into(), &[dims.vocab, d]);
    let mut x = b.g.gather(embd, 0, tokens, false).unwrap();
    let (mut ks, mut vs, mut states) = (Vec::new(), Vec::new(), Vec::new());
    for (l, &rec) in dims.recurrent.iter().enumerate() {
        let p = |s: &str| format!("blk.{l}.{s}");
        let attn_norm = b.fixed(p("attn_norm.weight"), &[d]);
        let normed = b.rms_norm(x, attn_norm);
        let y = if rec {
            let (y, s) = gdn(&mut b, dims, l, normed);
            states.push(s);
            y
        } else {
            let (y, k, v) = attention(&mut b, dims, l, normed);
            ks.push(k);
            vs.push(v);
            y
        };
        x = b.add(x, y);
        let post_norm = b.fixed(p("post_attention_norm.weight"), &[d]);
        let normed = b.rms_norm(x, post_norm);
        let y = moe(&mut b, dims, l, normed);
        x = b.add(x, y);
    }
    let out_norm = b.fixed("output_norm.weight".into(), &[d]);
    let head = b.fixed("output.weight".into(), &[dims.vocab, d]);
    let normed = b.rms_norm(x, out_norm);
    let logits = b.linear(normed, head);
    CoderPrefill {
        g: b.g,
        inputs: b.inputs,
        logits,
        k: ks,
        v: vs,
        gdn_state: states,
    }
}

/// Gated DeltaNet over the sequence from zero state: `(output [T, d],
/// state scan)`.
fn gdn(b: &mut B, dims: &CoderDims, l: usize, x: TensorId) -> (TensorId, TensorId) {
    let p = |s: &str| format!("blk.{l}.{s}");
    let d = dims.d_model;
    let (s, hk, hv, dc) = (
        dims.ssm_d_state,
        dims.ssm_k_heads,
        dims.ssm_v_heads,
        dims.ssm_d_conv,
    );
    let r = hv / hk;
    let kd = s * hk;
    let vd = s * hv;
    let ch = dims.conv_channels();
    let eps = dims.eps;
    let w_qkv = b.fixed(p("attn_qkv.weight"), &[ch, d]);
    let w_z = b.fixed(p("attn_gate.weight"), &[vd, d]);
    let w_ba = b.fixed(p("ssm_ba.weight"), &[2 * hv, d]);
    // ggml [d_conv, ch]: tap i of channel c at c·d_conv + i
    let conv = b.fixed(p("ssm_conv1d.weight"), &[ch, dc]);
    let dt = b.fixed(p("ssm_dt.bias"), &[hv]);
    let a_log = b.fixed(p("ssm_a"), &[hv]); // already −exp(A_log)
    let norm = b.fixed(p("ssm_norm.weight"), &[s]);
    let w_out = b.fixed(p("ssm_out.weight"), &[d, vd]);

    let qkv = b.linear(x, w_qkv);
    let z = b.linear(x, w_z);
    let ba = b.linear(x, w_ba);
    // causal depthwise conv, tap i reaching d_conv − 1 − i back, zero
    // history: a guarded read at t + i − (d_conv − 1). Space (t, c, i).
    let shifted = Arg {
        src: qkv,
        view: View {
            gather: None,
            guard: true,
            forms: vec![form(3, &[(0, 1), (2, 1)], -(dc as i64 - 1)), unit(3, 1)],
        },
    };
    let conv_acc =
        b.g.comp(
            &[T, Dim::Fixed(ch)],
            &[Dim::Fixed(dc)],
            vec![shifted, arg(conv, vec![unit(3, 1), unit(3, 2)])],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let h = b.g.map(&[conv_acc], |v| silu(v[0])).unwrap();
    // q and k L2 norms per head: [T, 2·hk] (q heads, then k heads)
    let qk_ss = b.head_ss(h, 2 * hk, s, s, 0);
    // [T, hk, r, s]: v head (g, j) = g·r + j reads k head g
    let pv = [T, Dim::Fixed(hk), Dim::Fixed(r), Dim::Fixed(s)];
    let l2 = |b: &mut B, base: i64, ss_off: i64| {
        b.g.comp(
            &pv,
            &[],
            vec![
                arg(h, vec![unit(4, 0), form(4, &[(1, s as i64), (3, 1)], base)]),
                arg(qk_ss, vec![unit(4, 0), form(4, &[(1, 1)], ss_off)]),
            ],
            None,
            |v| vec![v[0] / (v[1] + lit(eps)).sqrt()],
        )
        .unwrap()
    };
    let q = l2(b, 0, 0);
    let k = l2(b, kd as i64, hk as i64);
    let v =
        b.g.comp(
            &pv,
            &[],
            vec![arg(
                h,
                vec![
                    unit(4, 0),
                    form(
                        4,
                        &[(1, (r * s) as i64), (2, s as i64), (3, 1)],
                        2 * kd as i64,
                    ),
                ],
            )],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    // fused ba: k head g's row is [b of its r v heads | a of the same]
    let pg = [T, Dim::Fixed(hk), Dim::Fixed(r)];
    let vh = form(3, &[(1, r as i64), (2, 1)], 0);
    let alpha =
        b.g.comp(
            &pg,
            &[],
            vec![
                arg(
                    ba,
                    vec![unit(3, 0), form(3, &[(1, 2 * r as i64), (2, 1)], r as i64)],
                ),
                arg(dt, vec![vh.clone()]),
                arg(a_log, vec![vh]),
            ],
            None,
            |v| vec![(softplus(v[0] + v[1]) * v[2]).exp()],
        )
        .unwrap();
    let beta =
        b.g.comp(
            &pg,
            &[],
            vec![arg(
                ba,
                vec![unit(3, 0), form(3, &[(1, 2 * r as i64), (2, 1)], 0)],
            )],
            None,
            |v| vec![sigmoid(v[0])],
        )
        .unwrap();
    let state =
        b.g.delta_scan(k, v, Some(alpha), Some(beta), 0, true)
            .unwrap();
    let o = b.g.scan_readout(state, q, 1.0 / (s as f64).sqrt()).unwrap();
    // rmsnorm(o)·ssm_norm·silu(z) per v head
    let o_ss =
        b.g.comp(
            &pg,
            &[Dim::Fixed(s)],
            vec![arg(o, ident(4))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[0]],
        )
        .unwrap();
    let y =
        b.g.comp(
            &pv,
            &[],
            vec![
                arg(o, ident(4)),
                arg(o_ss, vec![unit(4, 0), unit(4, 1), unit(4, 2)]),
                arg(norm, vec![unit(4, 3)]),
                arg(
                    z,
                    vec![
                        unit(4, 0),
                        form(4, &[(1, (r * s) as i64), (2, s as i64), (3, 1)], 0),
                    ],
                ),
            ],
            None,
            |v| {
                let inv = lit(1.0) / (v[1] / lit(s as f64) + lit(eps)).sqrt();
                vec![v[0] * inv * v[2] * silu(v[3])]
            },
        )
        .unwrap();
    let y = b.g.reshape(y, &[T, Dim::Fixed(vd)]).unwrap();
    (b.linear(y, w_out), state)
}

/// Gated causal GQA: `(output [T, d], keys [T, kvh, hd], values [T, kvh·hd])`.
fn attention(b: &mut B, dims: &CoderDims, l: usize, x: TensorId) -> (TensorId, TensorId, TensorId) {
    let p = |s: &str| format!("blk.{l}.{s}");
    let d = dims.d_model;
    let (nh, kvh, hd) = (dims.n_head, dims.n_head_kv, dims.head_dim);
    let per = nh / kvh;
    let w_q = b.fixed(p("attn_q.weight"), &[nh * 2 * hd, d]);
    let w_k = b.fixed(p("attn_k.weight"), &[kvh * hd, d]);
    let w_v = b.fixed(p("attn_v.weight"), &[kvh * hd, d]);
    let qn = b.fixed(p("attn_q_norm.weight"), &[hd]);
    let kn = b.fixed(p("attn_k_norm.weight"), &[hd]);
    let w_o = b.fixed(p("attn_output.weight"), &[d, nh * hd]);

    let q_full = b.linear(x, w_q); // [T, nh·(q hd | gate hd)]
    let k_lin = b.linear(x, w_k);
    let v = b.linear(x, w_v);
    let q_ss = b.head_ss(q_full, nh, hd, 2 * hd, 0);
    let k_ss = b.head_ss(k_lin, kvh, hd, hd, 0);
    let q = rope_normed(b, dims, q_full, q_ss, qn, nh, 2 * hd);
    let k = rope_normed(b, dims, k_lin, k_ss, kn, kvh, hd);

    let visible = |t: ExprId, j: ExprId| t - j + lit(0.5);
    let scale = 1.0 / (hd as f64).sqrt();
    // scores[t, g, r, j] over c: space (t, g, r, j, c). Unmasked: the
    // statistics and p·v below mask every read, so the scores are a plain
    // q·kᵀ (a tensor-core matmul in a many-row step)
    let sp = [T, Dim::Fixed(kvh), Dim::Fixed(per), T, Dim::Fixed(hd)];
    let scores =
        b.g.comp(
            &sp[..4],
            &sp[4..],
            vec![
                arg(
                    q,
                    vec![
                        unit(5, 0),
                        form(5, &[(1, per as i64), (2, 1)], 0),
                        unit(5, 4),
                    ],
                ),
                arg(k, vec![unit(5, 3), unit(5, 1), unit(5, 4)]),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1] * lit(scale)],
        )
        .unwrap();
    // softmax statistics over j: space (t, g, r, j)
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
    // attn[t, g, r, c] = Σ_j p·v / l: space (t, g, r, c, j)
    let sp = [T, Dim::Fixed(kvh), Dim::Fixed(per), Dim::Fixed(hd), T];
    // `TANG_CODER_VT=1`: V read through a transposed copy `[kvh·hd, T]`, so
    // the derived step caches it with positions innermost and P·V streams
    // each head dim's cached values contiguously (batched decode at long
    // context: P·V was ~half the step, its V loads strided by kvh·hd)
    let vt_arg = if std::env::var("TANG_CODER_VT").as_deref() == Ok("1") {
        let vt =
            b.g.comp(
                &[Dim::Fixed(kvh * hd), T],
                &[],
                vec![arg(v, vec![unit(2, 1), unit(2, 0)])],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        arg(vt, vec![form(5, &[(1, hd as i64), (3, 1)], 0), unit(5, 4)])
    } else {
        arg(v, vec![unit(5, 4), form(5, &[(1, hd as i64), (3, 1)], 0)])
    };
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
                vt_arg.clone(),
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
    // × sigmoid(gate): head g·per + r's second half
    let gated =
        b.g.comp(
            &sp[..4],
            &[],
            vec![
                arg(attn, ident(4)),
                arg(
                    q_full,
                    vec![
                        unit(4, 0),
                        form(
                            4,
                            &[(1, (per * 2 * hd) as i64), (2, 2 * hd as i64), (3, 1)],
                            hd as i64,
                        ),
                    ],
                ),
            ],
            None,
            |v| vec![v[0] * sigmoid(v[1])],
        )
        .unwrap();
    let gated = b.g.reshape(gated, &[T, Dim::Fixed(nh * hd)]).unwrap();
    (b.linear(gated, w_o), k, v)
}

/// `[T, heads, hd]`: each head of `src` (head `h` at `h·stride`) rmsnormed
/// with `w`, then NeoX rope on dims `0..n_rot` (pairs `i`, `i + n_rot/2`,
/// angle `t·base^(−2i/n_rot)`), identity above. The partner of dim `i` is a
/// guarded read of the normed head at `i ± n_rot/2` (a shift on the head
/// axis, a plain read along `T`); position is an iota.
fn rope_normed(
    b: &mut B,
    dims: &CoderDims,
    src: TensorId,
    ss: TensorId,
    w: TensorId,
    heads: usize,
    stride: usize,
) -> TensorId {
    let hd = dims.head_dim;
    let (n_rot, half) = (dims.n_rot as f64, dims.n_rot / 2);
    let (eps, ln_base) = (dims.eps, dims.rope_base.ln());
    let sp = [T, Dim::Fixed(heads), Dim::Fixed(hd)];
    let xn =
        b.g.comp(
            &sp,
            &[],
            vec![
                arg(
                    src,
                    vec![unit(3, 0), form(3, &[(1, stride as i64), (2, 1)], 0)],
                ),
                arg(ss, vec![unit(3, 0), unit(3, 1)]),
                arg(w, vec![unit(3, 2)]),
            ],
            None,
            move |v| {
                let inv = lit(1.0) / (v[1] / lit(hd as f64) + lit(eps)).sqrt();
                vec![v[0] * inv * v[2]]
            },
        )
        .unwrap();
    let shifted = |o: i64| {
        let mut a = arg(xn, vec![unit(3, 0), unit(3, 1), form(3, &[(2, 1)], o)]);
        a.view.guard = true;
        a
    };
    let (ti, ii) = (b.g.iota(&sp, 0), b.g.iota(&sp, 2));
    let h = half as f64;
    b.g.comp(
        &sp,
        &[],
        vec![
            arg(xn, ident(3)),
            shifted(half as i64),
            shifted(-(half as i64)),
            arg(ti, ident(3)),
            arg(ii, ident(3)),
        ],
        None,
        move |v| {
            let (x, up, dn, t, i) = (v[0], v[1], v[2], v[3], v[4]);
            let lower = lit(h - 0.5) - i; // i < half
            let f = select(lower, i, i - lit(h));
            let theta = t * (lit(-2.0 * ln_base / n_rot) * f).exp();
            let (c, s) = (theta.cos(), theta.sin());
            let rot = select(lower, x * c - up * s, x * c + dn * s);
            vec![select(lit(n_rot - 0.5) - i, rot, x)]
        },
    )
    .unwrap()
}

/// Routed experts (softmax, top-k, renormalized) plus the gated shared
/// expert: `[T, d]`.
fn moe(b: &mut B, dims: &CoderDims, l: usize, x: TensorId) -> TensorId {
    let p = |s: &str| format!("blk.{l}.{s}");
    let d = dims.d_model;
    let (ne, k, ff, ffs) = (
        dims.n_expert,
        dims.n_expert_used,
        dims.n_ff_exp,
        dims.n_ff_shexp,
    );
    let router = b.fixed(p("ffn_gate_inp.weight"), &[ne, d]);
    let w_gate = b.fixed(p("ffn_gate_exps.weight"), &[ne, ff, d]);
    let w_up = b.fixed(p("ffn_up_exps.weight"), &[ne, ff, d]);
    let w_down = b.fixed(p("ffn_down_exps.weight"), &[ne, d, ff]);
    let sh_inp = b.fixed(p("ffn_gate_inp_shexp.weight"), &[d]);
    let sh_gate = b.fixed(p("ffn_gate_shexp.weight"), &[ffs, d]);
    let sh_up = b.fixed(p("ffn_up_shexp.weight"), &[ffs, d]);
    let sh_down = b.fixed(p("ffn_down_shexp.weight"), &[d, ffs]);

    let logits = b.linear(x, router);
    let e = Dim::Fixed(ne);
    // top-k over the logits (TANG_ROUTER_LOGITS=0: over the softmax): the
    // order is the same and the renormalized weights need no softmax
    // statistics, p_i / Σ_topk p_j = exp(l_i - l_1) / Σ_topk exp(l_j - l_1).
    // Exact up to rounding: a top-k holds at least k/ne of the mass
    // (10/512 here), far over the 2⁻¹⁴ clamp, which never applies
    let by_logit = std::env::var("TANG_ROUTER_LOGITS").map_or(true, |v| v != "0");
    let score = if by_logit {
        logits
    } else {
        let mx =
            b.g.comp(
                &[T],
                &[e],
                vec![arg(logits, ident(2))],
                Some(Monoid::max()),
                |v| vec![v[0]],
            )
            .unwrap();
        let den =
            b.g.comp(
                &[T],
                &[e],
                vec![arg(logits, ident(2)), arg(mx, vec![unit(2, 0)])],
                Some(Monoid::sum()),
                |v| vec![(v[0] - v[1]).exp()],
            )
            .unwrap();
        b.g.comp(
            &[T, e],
            &[],
            vec![
                arg(logits, ident(2)),
                arg(mx, vec![unit(2, 0)]),
                arg(den, vec![unit(2, 0)]),
            ],
            None,
            |v| vec![(v[0] - v[1]).exp() / v[2]],
        )
        .unwrap()
    };
    // top-k as (expert, p) pairs, best first, ties to the lower expert
    let ei = b.g.iota(&[T, e], 1);
    let topk =
        b.g.comp_tuple(
            &[T, Dim::Fixed(2 * k)],
            &[e],
            vec![arg(ei, ident(2)), arg(score, ident(2))],
            Some(Monoid::topk(k)),
            |v| {
                let mut o = vec![v[0], v[1]];
                for _ in 1..k {
                    o.push(lit(0.0));
                    o.push(lit(-(f32::MAX as f64)));
                }
                o
            },
        )
        .unwrap();
    let wsum =
        b.g.comp(
            &[T],
            &[Dim::Fixed(k)],
            vec![
                arg(topk, vec![unit(2, 0), form(2, &[(1, 2)], 1)]),
                arg(topk, vec![unit(2, 0), form(2, &[], 1)]),
            ],
            Some(Monoid::sum()),
            move |v| vec![if by_logit { (v[0] - v[1]).exp() } else { v[0] }],
        )
        .unwrap();
    // each picked expert's gate/up row block, gathered by its id:
    // space (t, slot, f, n)
    let expert_in = |b: &mut B, w: TensorId| {
        b.g.comp(
            &[T, Dim::Fixed(k), Dim::Fixed(ff)],
            &[Dim::Fixed(d)],
            vec![
                arg(topk, vec![unit(4, 0), form(4, &[(1, 2)], 0)]),
                arg(x, vec![unit(4, 0), unit(4, 3)]),
                Arg {
                    src: w,
                    view: View::pick(4, &[None, Some(2), Some(3)]).gather(0, 0, 1),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[1] * v[2]],
        )
        .unwrap()
    };
    let g_h = expert_in(b, w_gate);
    let u_h = expert_in(b, w_up);
    let h = b.g.map(&[g_h, u_h], |v| silu(v[0]) * v[1]).unwrap();
    // each picked expert's down projection: space (t, slot, n, f) (a
    // gathered GEMV, the int8 GEMV's shape), then Σ_slot w_slot / Σw · dn
    let dn =
        b.g.comp(
            &[T, Dim::Fixed(k), Dim::Fixed(d)],
            &[Dim::Fixed(ff)],
            vec![
                arg(topk, vec![unit(4, 0), form(4, &[(1, 2)], 0)]),
                arg(h, vec![unit(4, 0), unit(4, 1), unit(4, 3)]),
                Arg {
                    src: w_down,
                    view: View::pick(4, &[None, Some(2), Some(3)]).gather(0, 0, 1),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[1] * v[2]],
        )
        .unwrap();
    // space (t, n, slot)
    let routed =
        b.g.comp(
            &[T, Dim::Fixed(d)],
            &[Dim::Fixed(k)],
            vec![
                arg(topk, vec![unit(3, 0), form(3, &[(2, 2)], 1)]),
                arg(wsum, vec![unit(3, 0)]),
                arg(dn, vec![unit(3, 0), unit(3, 2), unit(3, 1)]),
                arg(topk, vec![unit(3, 0), form(3, &[], 1)]),
            ],
            Some(Monoid::sum()),
            move |v| {
                if by_logit {
                    // Σ ≥ 1 (the top logit's own term)
                    return vec![(v[0] - v[3]).exp() / v[1] * v[2]];
                }
                let clamp = lit(6.103515625e-5);
                let ws = select(v[1] - clamp, v[1], clamp);
                vec![v[0] / ws * v[2]]
            },
        )
        .unwrap();
    let sg =
        b.g.comp(
            &[T],
            &[Dim::Fixed(d)],
            vec![arg(x, ident(2)), arg(sh_inp, vec![unit(2, 1)])],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let sgv = b.linear(x, sh_gate);
    let suv = b.linear(x, sh_up);
    let sh = b.g.map(&[sgv, suv], |v| silu(v[0]) * v[1]).unwrap();
    let sh = b.linear(sh, sh_down);
    b.g.comp(
        &[T, Dim::Fixed(d)],
        &[],
        vec![
            arg(routed, ident(2)),
            arg(sg, vec![unit(2, 0)]),
            arg(sh, ident(2)),
        ],
        None,
        |v| vec![v[0] + sigmoid(v[1]) * v[2]],
    )
    .unwrap()
}

/// [`derived_step`] with only the rows `rows` of the logits (a prefill
/// chunk needs the last).
pub fn derived_step_rows(
    pg: &CoderPrefill,
    k: usize,
    capacity: usize,
    rows: tang_compile::tensor::step::Rows,
) -> Result<StepGraph, String> {
    tang_compile::tensor::step::derive_step_with(
        &pg.g,
        &[pg.logits],
        &StepSpec {
            seq: SEQ,
            new: k,
            capacity,
        },
        &tang_compile::tensor::step::StepOptions { rows, batch: None },
    )
    .map_err(|e| e.0)
}

/// The derived step of `pg` appending `k` tokens against `capacity`-row
/// caches (outputs `[logits]`), or derive_step's refusal.
pub fn derived_step(pg: &CoderPrefill, k: usize, capacity: usize) -> Result<StepGraph, String> {
    derive_step(
        &pg.g,
        &[pg.logits],
        &StepSpec {
            seq: SEQ,
            new: k,
            capacity,
        },
    )
    .map_err(|e| e.0)
}

/// The batched `T → T + 1` step: [`batch_graph`] adds a leading dim `B` of
/// independent streams, each at its own length (its `p0` an element of the
/// `[B]` position input).
///
/// [`batch_graph`]: tang_compile::tensor::step::batch_graph
pub fn derived_step_batched(pg: &CoderPrefill, capacity: usize) -> Result<StepGraph, String> {
    derived_step_batched_k(pg, 1, capacity, tang_compile::tensor::step::Rows::All)
}

/// [`derived_step_batched`] appending `k` tokens to each stream: packed
/// prefill, `B` prompts' chunks in one step.
pub fn derived_step_batched_k(
    pg: &CoderPrefill,
    k: usize,
    capacity: usize,
    rows: tang_compile::tensor::step::Rows,
) -> Result<StepGraph, String> {
    use tang_compile::tensor::step::{batch_graph, derive_step_with, StepOptions};
    let (bg, outs) = batch_graph(&pg.g, &[pg.logits], SEQ, "B").map_err(|e| e.0)?;
    derive_step_with(
        &bg,
        &outs,
        &StepSpec {
            seq: SEQ,
            new: k,
            capacity,
        },
        &StepOptions {
            rows,
            batch: Some("B"),
        },
    )
    .map_err(|e| e.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tang_compile::tensor::step::StepInput;
    use tang_llm::flash::reference::FlashRef;

    /// A GGUF tensor: name, ggml dims (`ne`, fastest first), f32 data.
    struct GT {
        name: String,
        ne: Vec<u64>,
        data: Vec<f32>,
    }

    enum Meta {
        U32(u32),
        F32(f32),
        Str(String),
        U32s(Vec<u32>),
    }

    /// A single-file GGUF v3, every tensor f32.
    fn write_gguf(path: &Path, meta: &[(String, Meta)], tensors: &[GT]) {
        const ALIGN: usize = 32;
        let mut h = Vec::new();
        let s = |h: &mut Vec<u8>, x: &str| {
            h.extend_from_slice(&(x.len() as u64).to_le_bytes());
            h.extend_from_slice(x.as_bytes());
        };
        h.extend_from_slice(b"GGUF");
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        h.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        for (k, v) in meta {
            s(&mut h, k);
            match v {
                Meta::U32(x) => {
                    h.extend_from_slice(&4u32.to_le_bytes());
                    h.extend_from_slice(&x.to_le_bytes());
                }
                Meta::F32(x) => {
                    h.extend_from_slice(&6u32.to_le_bytes());
                    h.extend_from_slice(&x.to_le_bytes());
                }
                Meta::Str(x) => {
                    h.extend_from_slice(&8u32.to_le_bytes());
                    s(&mut h, x);
                }
                Meta::U32s(xs) => {
                    h.extend_from_slice(&9u32.to_le_bytes());
                    h.extend_from_slice(&4u32.to_le_bytes());
                    h.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs {
                        h.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        let mut off = 0usize;
        for t in tensors {
            s(&mut h, &t.name);
            h.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
            for &n in &t.ne {
                h.extend_from_slice(&n.to_le_bytes());
            }
            h.extend_from_slice(&0u32.to_le_bytes()); // F32
            h.extend_from_slice(&(off as u64).to_le_bytes());
            off += (t.data.len() * 4).div_ceil(ALIGN) * ALIGN;
        }
        while h.len() % ALIGN != 0 {
            h.push(0);
        }
        for t in tensors {
            let start = h.len();
            for v in &t.data {
                h.extend_from_slice(&v.to_le_bytes());
            }
            h.resize(start + (t.data.len() * 4).div_ceil(ALIGN) * ALIGN, 0);
        }
        std::fs::write(path, &h).unwrap();
    }

    fn meta(dims: &CoderDims) -> Vec<(String, Meta)> {
        let k = |s: &str| format!("qwen3next.{s}");
        let u = |s: &str, v: usize| (k(s), Meta::U32(v as u32));
        vec![
            ("general.architecture".into(), Meta::Str("qwen3next".into())),
            u("block_count", dims.recurrent.len()),
            u("embedding_length", dims.d_model),
            (
                k("attention.layer_norm_rms_epsilon"),
                Meta::F32(dims.eps as f32),
            ),
            u("attention.head_count", dims.n_head),
            u("attention.head_count_kv", dims.n_head_kv),
            u("attention.key_length", dims.head_dim),
            u("rope.dimension_count", dims.n_rot),
            (k("rope.freq_base"), Meta::F32(dims.rope_base as f32)),
            u("ssm.conv_kernel", dims.ssm_d_conv),
            u("ssm.state_size", dims.ssm_d_state),
            u("ssm.group_count", dims.ssm_k_heads),
            u("ssm.time_step_rank", dims.ssm_v_heads),
            u("expert_count", dims.n_expert),
            u("expert_used_count", dims.n_expert_used),
            u("expert_feed_forward_length", dims.n_ff_exp),
            (
                k("attention.recurrent_layers"),
                Meta::U32s(dims.recurrent.iter().map(|&r| r as u32).collect()),
            ),
        ]
    }

    /// Deterministic f32 weights per tensor name, scaled so activations
    /// stay O(1): matrices ±1/√cols, norms 1 ± 0.2, `ssm_a` negative.
    fn weights(pg: &CoderPrefill, seed: u64) -> Vec<GT> {
        pg.inputs
            .iter()
            .filter(|s| s.name != "tokens")
            .map(|s| {
                let shape: Vec<usize> = s.shape.iter().map(|d| d.n()).collect();
                let n: usize = shape.iter().product();
                let mut st = s.name.bytes().fold(1469598103934665603u64 ^ seed, |h, b| {
                    (h ^ b as u64).wrapping_mul(1099511628211)
                });
                let mut rnd = move || {
                    st = st
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
                };
                let name = s.name.as_str();
                let cols = *shape.last().unwrap();
                let data: Vec<f32> = (0..n)
                    .map(|_| {
                        let u = rnd();
                        (if name.ends_with("norm.weight") {
                            1.0 + 0.2 * u
                        } else if name.ends_with("ssm_a") {
                            -(0.5 + 0.4 * u).exp()
                        } else if name.ends_with("ssm_dt.bias") {
                            0.5 * u
                        } else if name.ends_with("ssm_conv1d.weight") {
                            0.6 * u
                        } else if name == "token_embd.weight" {
                            u
                        } else if name.ends_with("ffn_gate_inp.weight") {
                            3.0 * u / (cols as f64).sqrt()
                        } else {
                            1.5 * u / (cols as f64).sqrt()
                        }) as f32
                    })
                    .collect();
                GT {
                    name: s.name.clone(),
                    ne: shape.iter().rev().map(|&x| x as u64).collect(),
                    data,
                }
            })
            .collect()
    }

    fn tokens(n: usize, vocab: usize) -> Vec<u32> {
        (0..n).map(|i| ((37 * i + 5) % vocab) as u32).collect()
    }

    fn inputs(pg: &CoderPrefill, w: &[GT], toks: &[u32]) -> Vec<Vec<f64>> {
        pg.inputs
            .iter()
            .map(|s| {
                if s.name == "tokens" {
                    toks.iter().map(|&t| t as f64).collect()
                } else {
                    let t = w.iter().find(|t| t.name == s.name).unwrap();
                    t.data.iter().map(|&x| x as f64).collect()
                }
            })
            .collect()
    }

    /// Worst `|a − b| / max|b|`.
    fn rel(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let e = (x - y).abs() / scale;
                if e.is_nan() {
                    f64::INFINITY
                } else {
                    e
                }
            })
            .fold(0.0, f64::max)
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tang-coder-ir-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("coder-tiny.gguf")
    }

    /// The reference's logits (`[T, vocab]`, f32 → f64) for `toks`.
    fn reference_logits(path: &Path, toks: &[u32]) -> Vec<f64> {
        let mut r = FlashRef::open(path).unwrap();
        r.logits = Some(std::sync::Mutex::new(Vec::new()));
        r.forward(toks, 0, 1, None).unwrap();
        let lg = r.logits.take().unwrap().into_inner().unwrap();
        lg.iter().map(|&x| x as f64).collect()
    }

    fn ir_logits(pg: &CoderPrefill, w: &[GT], toks: &[u32]) -> Vec<f64> {
        let g = pg.g.bind(&[(SEQ, toks.len())]).unwrap();
        g.eval(&inputs(pg, w, toks), &[pg.logits]).remove(0)
    }

    /// Distinct experts the IR's router picks over `toks`, per MoE layer.
    fn experts_used(pg: &CoderPrefill, w: &[GT], toks: &[u32], dims: &CoderDims) -> Vec<usize> {
        // the top-k comps are the tuple comps (one per layer), in order
        let g = pg.g.bind(&[(SEQ, toks.len())]).unwrap();
        let tk: Vec<TensorId> = (0..g.nodes().len())
            .filter(|&n| g.comp_at(n).is_some_and(|c| c.tuple == 1))
            .map(TensorId::from_index)
            .collect();
        let vals = g.eval(&inputs(pg, w, toks), &tk);
        let k = dims.n_expert_used;
        vals.iter()
            .map(|v| {
                let mut ids: Vec<usize> = v
                    .chunks(2 * k)
                    .flat_map(|row| (0..k).map(move |r| row[2 * r] as usize))
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                ids.len()
            })
            .collect()
    }

    /// The prefill graph against `tang_llm`'s qwen3next CPU reference
    /// (`FlashRef::forward_plain`), same f32 weights, every position's
    /// logits, at several sequence lengths.
    #[test]
    fn prefill_matches_flash_reference() {
        let dims = CoderDims::tiny();
        let pg = prefill_graph(&dims);
        let w = weights(&pg, 0);
        let path = scratch("ref");
        write_gguf(&path, &meta(&dims), &w);
        let all = tokens(16, dims.vocab);
        let mut worst = 0.0f64;
        for t in [1usize, 2, 5, 9, 16] {
            let toks = &all[..t];
            let want = reference_logits(&path, toks);
            let got = ir_logits(&pg, &w, toks);
            assert!(
                got.iter().all(|x| x.is_finite()),
                "T={t}: IR logits not finite"
            );
            let e = rel(&got, &want);
            let used = experts_used(&pg, &w, toks, &dims);
            let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            eprintln!(
                "coder tiny T={t:2}: IR (f64) vs FlashRef (f32) logits max rel err {e:.2e} \
                 (max |logit| {scale:.2}); distinct experts routed per layer {used:?}"
            );
            worst = worst.max(e);
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        eprintln!("coder tiny: worst max rel err over T in {{1,2,5,9,16}}: {worst:.2e}");
        assert!(worst < 1e-4, "IR vs reference: {worst:e}");
    }

    /// Negative controls: one perturbed weight per block kind, fed to the
    /// IR only, must move the logits far past the tolerance — the check
    /// sees the GDN conv, the GDN decay, the attention QK norm, the
    /// routed experts and the shared expert.
    #[test]
    fn negative_controls_fail() {
        let dims = CoderDims::tiny();
        let pg = prefill_graph(&dims);
        let w = weights(&pg, 0);
        let path = scratch("neg");
        write_gguf(&path, &meta(&dims), &w);
        let toks = tokens(9, dims.vocab);
        let want = reference_logits(&path, &toks);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        let base = rel(&ir_logits(&pg, &w, &toks), &want);
        for (name, what) in [
            ("blk.0.ssm_conv1d.weight", "one GDN conv tap +0.5"),
            ("blk.2.ssm_a", "one GDN decay ×0.5"),
            ("blk.1.attn_q_norm.weight", "attention q-norm dim 5 ×1.5"),
            ("blk.1.ffn_down_exps.weight", "routed experts' down ×1.1"),
            (
                "blk.0.ffn_gate_inp_shexp.weight",
                "shared-expert gate dim 0 +1",
            ),
        ] {
            let mut bad: Vec<GT> = w
                .iter()
                .map(|t| GT {
                    name: t.name.clone(),
                    ne: t.ne.clone(),
                    data: t.data.clone(),
                })
                .collect();
            let t = bad.iter_mut().find(|t| t.name == name).unwrap();
            match name {
                n if n.ends_with("conv1d.weight") => t.data[3] += 0.5,
                n if n.ends_with("ssm_a") => t.data[1] *= 0.5,
                n if n.ends_with("q_norm.weight") => t.data[5] *= 1.5,
                n if n.ends_with("down_exps.weight") => t.data.iter_mut().for_each(|x| *x *= 1.1),
                _ => t.data[0] += 1.0,
            }
            let e = rel(&ir_logits(&pg, &bad, &toks), &want);
            eprintln!("negative control ({what}): max rel err {e:.2e} (unperturbed {base:.2e})");
            assert!(e > 1e-3, "{what}: the check did not notice ({e:e})");
        }
    }

    /// The full graph derives: the GDN state scan is a carried state and
    /// the GDN conv's guarded read back along `T` a ring cache (values:
    /// `derive_all::tests::coder_next_derives`). The conv alone too.
    #[test]
    fn derive_step_takes_gdn() {
        let pg = prefill_graph(&CoderDims::tiny());
        let st = derived_step(&pg, 1, 16).expect("derives");
        eprintln!("derive_step on coder tiny: {} caches", st.caches.len());
        assert!(st.caches.iter().any(|c| c.state));
        assert!(st.caches.iter().any(|c| c.ring.is_some()));
        // the conv by itself, written as in `gdn`
        let (ch, dc) = (6usize, 4usize);
        let mut g = TensorGraph::new();
        let x = g.input(&[T, Dim::Fixed(ch)]);
        let w = g.input(&[ch, dc]);
        let shifted = Arg {
            src: x,
            view: View {
                gather: None,
                guard: true,
                forms: vec![form(3, &[(0, 1), (2, 1)], -(dc as i64 - 1)), unit(3, 1)],
            },
        };
        let acc = g
            .comp(
                &[T, Dim::Fixed(ch)],
                &[Dim::Fixed(dc)],
                vec![shifted, arg(w, vec![unit(3, 1), unit(3, 2)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let spec = StepSpec {
            seq: SEQ,
            new: 1,
            capacity: 16,
        };
        let st = derive_step(&g, &[acc], &spec).expect("derives");
        assert_eq!(st.caches.len(), 1);
        assert_eq!(st.caches[0].ring, Some(dc - 1), "a ring of d_conv − 1 rows");
    }

    /// With attention layers only, the same builder derives: gated
    /// attention, partial rope as a rotation matrix, and gathered top-k
    /// MoE give a step whose logits and caches match the prefill grown by
    /// the new tokens.
    #[test]
    fn derive_step_attention_and_moe() {
        let dims = CoderDims {
            recurrent: vec![false, false],
            ..CoderDims::tiny()
        };
        let cap = 12;
        let pg = prefill_graph(&dims);
        let w = weights(&pg, 0);
        let all = tokens(cap, dims.vocab);
        for (p0, k) in [(5usize, 1usize), (5, 3)] {
            let step = derived_step(&pg, k, cap).expect("attention + MoE derives");
            assert_eq!(
                step.caches.len(),
                2 * dims.recurrent.len(),
                "a K and a V per layer"
            );
            // caches from a prefill of p0, NaN past p0
            let g0 = pg.g.bind(&[(SEQ, p0)]).unwrap();
            let nodes: Vec<TensorId> = step.caches.iter().map(|c| c.node).collect();
            let vals = g0.eval(&inputs(&pg, &w, &all[..p0]), &nodes);
            let caches: Vec<Vec<f64>> = vals
                .iter()
                .map(|v| {
                    assert_eq!(v.len() % p0, 0);
                    let row = v.len() / p0; // T is the outer axis of K and V
                    let mut c = v.clone();
                    c.resize(cap * row, f64::NAN);
                    c
                })
                .collect();
            let full = inputs(&pg, &w, &all[..p0 + k]);
            let sin: Vec<Vec<f64>> = step
                .inputs
                .iter()
                .map(|&si| match si {
                    StepInput::Pos => vec![p0 as f64],
                    StepInput::New(i) => {
                        assert_eq!(pg.inputs[i].name, "tokens");
                        full[i][p0..p0 + k].to_vec()
                    }
                    StepInput::Same(i) => full[i].clone(),
                    StepInput::Cache(c) => caches[c].clone(),
                })
                .collect();
            let got = step.graph.eval(&sin, &step.outputs).remove(0);
            let want = ir_logits(&pg, &w, &all[..p0 + k]);
            let e = rel(&got, &want[p0 * dims.vocab..]);
            eprintln!(
                "coder attn-only: derived step k={k} at p0={p0} vs prefill: max rel err {e:.2e}"
            );
            assert!(e < 1e-12, "k={k}: {e:e}");
        }
    }
}
