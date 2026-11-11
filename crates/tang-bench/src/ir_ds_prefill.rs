//! DeepSeek-V4.1-Flash written once, as a full-sequence (prefill) graph over
//! a symbolic sequence dim `T`, at a tiny config, to see how far
//! [`derive_step`] carries past plain GQA (`ir_qwen_prefill`).
//!
//! The math is `tang-llm/src/flash/deepseek/forward.rs`'s prefill (`sp = 0`),
//! term for term, in f64 and without its bf16 / fp8 / fp4 rounding:
//!
//! - **mHC**: the residual is `hc` streams; each sublayer's mixes come from
//!   an RMS-scaled projection of the whole stream, through sigmoid and a
//!   Sinkhorn-normalized `hc × hc` combine (fixed iterations, unrolled).
//!   The attention collapse uses the previous block's FFN pre-mix (one-hot
//!   for block 0), the FFN collapse the same block's attention pre-mix.
//! - **Attention**: low-rank q (`wq_a`, `q_norm`, `wq_b`), one shared KV
//!   head (`wkv`, `kv_norm`; K = V), interleaved-pair RoPE on the last
//!   `rope_head_dim` of every head (written as RoPE on all pairs with zero
//!   frequency on the no-rope pairs), a sliding window, a learned per-head
//!   sink in the softmax denominator, inverse RoPE on the output, and the
//!   grouped low-rank output (`wo_a` block-diagonal over `o_groups`, `wo_b`).
//! - **Compressed KV + indexer** (layers with `compress_ratio = r > 0`):
//!   the compressor pools each group of `r` tokens with a per-channel
//!   softmax gate, RMS-norms and ropes it at the group's first position;
//!   the indexer scores every complete group with `Σ_h relu(q_h·k)·w_h` and
//!   each query attends to its `index_topk` best groups on top of its
//!   window. Group `g` lives at row `t = g·r + r − 1` of a `T`-long node
//!   (the row that completes it), so everything stays over `T`; rows that
//!   complete no group are masked (`(t + 1) mod r ≠ 0`, with `floor`).
//!   Non-source layers read the source layer's compressed KV
//!   and top-k, as the hand path's `Shared` does.
//! - **MoE**: `sqrt(softplus(gate·x))` scores, top-k on `score + bias`
//!   ([`Monoid::topk`]), weights renormalized and scaled, SwiGLU with the
//!   `swiglu_limit` clamp, plus a shared expert. Routed experts are computed
//!   densely and weighted by membership (no dispatch).
//!
//! Every reduction over `T` is masked `select(t − j + ½, x, identity)` so
//! [`derive_step`] can take the step. Not written: Engram, the DSpark
//! drafter, the two-level candidate top-k, and the quantization.
use tang::Scalar;
use tang_compile::tensor::step::{derive_step, StepGraph, StepSpec};
use tang_compile::tensor::{Arg, Dim, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

/// The symbolic sequence dim.
pub const SEQ: &str = "T";
const T: Dim = Dim::Sym(SEQ);

fn fx(n: usize) -> Dim {
    Dim::Fixed(n)
}

fn unit(rank: usize, d: usize) -> Form {
    Form::unit(rank, d)
}

fn konst(rank: usize, c: i64) -> Form {
    Form::constant(rank, c)
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

/// `x` where `a == b` (integers as floats), else 0.
fn pick(a: ExprId, b: ExprId, x: ExprId) -> ExprId {
    let z = lit(0.0);
    select(a - b + lit(0.5), select(b - a + lit(0.5), x, z), z)
}

fn sigmoid(x: ExprId) -> ExprId {
    lit(1.0) / (lit(1.0) + (-x).exp())
}

/// The masked-out value of a max (absorbed by the always-visible diagonal).
const NEG: f64 = -1e30;

/// The tiny config: the real attention form, a few experts.
#[derive(Clone, Debug)]
pub struct DsDims {
    pub vocab: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window: usize,
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub n_experts: usize,
    pub n_active: usize,
    pub moe_inter: usize,
    pub route_scale: f64,
    pub norm_topk_prob: bool,
    pub swiglu_limit: f64,
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub eps: f64,
    /// Cache capacity for the derived step.
    pub ctx: usize,
    /// Rows of the bound expert tensors when they hold only some experts
    /// (a CPU check at real size can't hold all 384 in f64): each MoE
    /// layer then reads `ffn.expert_slot[n_experts]`, expert id → row.
    /// 0: the tensors hold every expert, no slot map.
    pub expert_rows: usize,
}

impl DsDims {
    /// Three layers: window-only, a compressed-KV + indexer source, and a
    /// compressed layer reading the source's caches and top-k.
    pub fn tiny() -> Self {
        DsDims {
            vocab: 48,
            dim: 32,
            n_layers: 3,
            n_heads: 4,
            head_dim: 16,
            rope_head_dim: 8,
            q_lora_rank: 24,
            o_groups: 2,
            o_lora_rank: 8,
            window: 4,
            compress_ratios: vec![0, 2, 2],
            kv_source_layers: vec![1],
            index_source_layers: vec![1],
            index_n_heads: 2,
            index_head_dim: 8,
            index_topk: 2,
            n_experts: 4,
            n_active: 2,
            moe_inter: 16,
            route_scale: 1.5,
            norm_topk_prob: true,
            swiglu_limit: 1.0,
            hc: 2,
            sinkhorn_iters: 3,
            hc_eps: 1e-6,
            eps: 1e-6,
            ctx: 16,
            expert_rows: 0,
        }
    }

    pub fn mix(&self) -> usize {
        (2 + self.hc) * self.hc
    }

    /// The layer whose compressed KV / top-k layer `l` reads (the last source at or below it).
    fn source(&self, list: &[usize], l: usize) -> usize {
        *list
            .iter()
            .filter(|&&s| s <= l)
            .max()
            .expect("a compressed layer below its first source")
    }
}

#[derive(Clone, Debug)]
pub struct PrefillInput {
    pub name: String,
    pub shape: Vec<Dim>,
}

pub struct PrefillGraph {
    pub g: TensorGraph,
    pub inputs: Vec<PrefillInput>,
    /// `[T, vocab]`.
    pub logits: TensorId,
    /// Named intermediates (for cache naming and debugging): `layers.{l}.{what}`.
    pub named: Vec<(String, TensorId)>,
}

impl PrefillGraph {
    pub fn name_of(&self, t: TensorId) -> Option<&str> {
        self.named
            .iter()
            .find(|(_, n)| *n == t)
            .map(|(s, _)| s.as_str())
    }
}

struct B {
    g: TensorGraph,
    inputs: Vec<PrefillInput>,
    named: Vec<(String, TensorId)>,
    eps: f64,
}

impl B {
    fn input(&mut self, name: String, shape: &[Dim]) -> TensorId {
        self.inputs.push(PrefillInput {
            name,
            shape: shape.to_vec(),
        });
        self.g.input(shape)
    }

    fn w(&mut self, name: String, shape: &[usize]) -> TensorId {
        let s: Vec<Dim> = shape.iter().map(|&n| fx(n)).collect();
        self.input(name, &s)
    }

    fn name(&mut self, s: String, t: TensorId) {
        self.named.push((s, t));
    }

    fn n(&self, t: TensorId, d: usize) -> usize {
        self.g.shape(t)[d].n()
    }

    /// `y[t, n] = Σ_k x[t, k]·w[n, k]`.
    fn linear(&mut self, x: TensorId, w: TensorId) -> TensorId {
        let (n, k) = (self.n(w, 0), self.n(w, 1));
        self.g
            .comp(
                &[T, fx(n)],
                &[fx(k)],
                vec![
                    arg(x, vec![unit(3, 0), unit(3, 2)]),
                    arg(w, vec![unit(3, 1), unit(3, 2)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    fn rms(&mut self, x: TensorId, w: TensorId) -> TensorId {
        let n = self.n(w, 0);
        let eps = self.eps;
        let ss = self
            .g
            .comp(
                &[T],
                &[fx(n)],
                vec![arg(x, ident(2))],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        self.g
            .comp(
                &[T, fx(n)],
                &[],
                vec![
                    arg(x, ident(2)),
                    arg(ss, vec![unit(2, 0)]),
                    arg(w, vec![unit(2, 1)]),
                ],
                None,
                |v| vec![v[0] * v[2] / (v[1] / lit(n as f64) + lit(eps)).sqrt()],
            )
            .unwrap()
    }

    /// Interleaved-pair RoPE of every `hd`-wide head of `x` (`[T, heads·hd]`)
    /// at position `t + off`; `fr` (`[hd/2]`) is zero on the no-rope pairs.
    fn rope(
        &mut self,
        x: TensorId,
        heads: usize,
        hd: usize,
        fr: TensorId,
        off: i64,
        inv: bool,
    ) -> TensorId {
        let sp = [T, fx(heads), fx(hd / 2), fx(2)];
        let (ti, par) = (self.g.iota(&sp, 0), self.g.iota(&sp, 3));
        let pair = |o| arg(x, vec![unit(4, 0), form(4, &[(1, hd as i64), (2, 2)], o)]);
        let sgn = if inv { -1.0 } else { 1.0 };
        let r = self
            .g
            .comp(
                &sp,
                &[],
                vec![
                    pair(0),
                    pair(1),
                    arg(fr, vec![unit(4, 2)]),
                    arg(ti, ident(4)),
                    arg(par, ident(4)),
                ],
                None,
                |v| {
                    let ang = (v[3] + lit(off as f64)) * v[2];
                    let (c, s) = (ang.cos(), ang.sin() * lit(sgn));
                    let even = v[0] * c - v[1] * s;
                    let odd = v[0] * s + v[1] * c;
                    vec![select(v[4] - lit(0.5), odd, even)]
                },
            )
            .unwrap();
        self.g.reshape(r, &[T, fx(heads * hd)]).unwrap()
    }

    /// `[T, a, b]` reduced over axis 2 (`row`) or 1 (`col`) with `m`, then
    /// `f(x, r)` elementwise: one Sinkhorn half-step or a softmax piece.
    fn hc_norm(
        &mut self,
        c: TensorId,
        hc: usize,
        rows: bool,
        f: impl Fn(ExprId, ExprId) -> ExprId,
    ) -> TensorId {
        // reduce: space (t, keep, red); row: c[t, keep, red], col: c[t, red, keep]
        let rd = if rows {
            vec![unit(3, 0), unit(3, 1), unit(3, 2)]
        } else {
            vec![unit(3, 0), unit(3, 2), unit(3, 1)]
        };
        let s = self
            .g
            .comp(
                &[T, fx(hc)],
                &[fx(hc)],
                vec![arg(c, rd)],
                Some(Monoid::sum()),
                |v| vec![v[0]],
            )
            .unwrap();
        let sv = if rows {
            vec![unit(3, 0), unit(3, 1)]
        } else {
            vec![unit(3, 0), unit(3, 2)]
        };
        self.g
            .comp(
                &[T, fx(hc), fx(hc)],
                &[],
                vec![arg(c, ident(3)), arg(s, sv)],
                None,
                |v| vec![f(v[0], v[1])],
            )
            .unwrap()
    }

    /// The mHC coefficients of stream `x` (`[T, hc·d]`): pre `[T, hc]`,
    /// post `[T, hc]`, comb `[T, hc(j), hc(k)]`.
    fn hc_mixes(&mut self, p: &str, kind: &str, x: TensorId, dm: &DsDims) -> [TensorId; 3] {
        let (hc, d, mix) = (dm.hc, dm.dim, dm.mix());
        let (eps, he) = (dm.eps, dm.hc_eps);
        let f = self.w(format!("{p}.hc_{kind}_fn"), &[mix, hc * d]);
        let scale = self.w(format!("{p}.hc_{kind}_scale"), &[3]);
        let base = self.w(format!("{p}.hc_{kind}_base"), &[mix]);
        let mixes = self.linear(x, f);
        let ss = self
            .g
            .comp(
                &[T],
                &[fx(hc * d)],
                vec![arg(x, ident(2))],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        let n = (hc * d) as f64;
        let m = self
            .g
            .comp(
                &[T, fx(mix)],
                &[],
                vec![arg(mixes, ident(2)), arg(ss, vec![unit(2, 0)])],
                None,
                |v| vec![v[0] / (v[1] / lit(n) + lit(eps)).sqrt()],
            )
            .unwrap();
        let gate = |b: &mut B, off: usize, si: i64, post: bool| {
            b.g.comp(
                &[T, fx(hc)],
                &[],
                vec![
                    arg(m, vec![unit(2, 0), form(2, &[(1, 1)], off as i64)]),
                    arg(scale, vec![konst(2, si)]),
                    arg(base, vec![form(2, &[(1, 1)], off as i64)]),
                ],
                None,
                |v| {
                    let s = sigmoid(v[0] * v[1] + v[2]);
                    vec![if post { lit(2.0) * s } else { s + lit(he) }]
                },
            )
            .unwrap()
        };
        let pre = gate(self, 0, 0, false);
        let post = gate(self, hc, 1, true);
        let cl = self
            .g
            .comp(
                &[T, fx(hc), fx(hc)],
                &[],
                vec![
                    arg(
                        m,
                        vec![
                            unit(3, 0),
                            form(3, &[(1, hc as i64), (2, 1)], 2 * hc as i64),
                        ],
                    ),
                    arg(scale, vec![konst(3, 2)]),
                    arg(
                        base,
                        vec![form(3, &[(1, hc as i64), (2, 1)], 2 * hc as i64)],
                    ),
                ],
                None,
                |v| vec![v[0] * v[1] + v[2]],
            )
            .unwrap();
        // row softmax (+eps); the logits are O(1), so no max shift is needed
        // for f64, but keep the reference's form: exp(x − max)
        let mx = self
            .g
            .comp(
                &[T, fx(hc)],
                &[fx(hc)],
                vec![arg(cl, ident(3))],
                Some(Monoid::max()),
                |v| vec![v[0]],
            )
            .unwrap();
        let e = self
            .g
            .comp(
                &[T, fx(hc), fx(hc)],
                &[],
                vec![arg(cl, ident(3)), arg(mx, vec![unit(3, 0), unit(3, 1)])],
                None,
                |v| vec![(v[0] - v[1]).exp()],
            )
            .unwrap();
        let mut comb = self.hc_norm(e, hc, true, |x, s| x / s + lit(he));
        comb = self.hc_norm(comb, hc, false, |x, s| x / (s + lit(he)));
        for _ in 1..dm.sinkhorn_iters {
            comb = self.hc_norm(comb, hc, true, |x, s| x / (s + lit(he)));
            comb = self.hc_norm(comb, hc, false, |x, s| x / (s + lit(he)));
        }
        [pre, post, comb]
    }

    /// `out[t, i] = Σ_c pre[t, c]·x[t, c·d + i]`.
    fn hc_pre(&mut self, x: TensorId, pre: TensorId, hc: usize, d: usize) -> TensorId {
        self.g
            .comp(
                &[T, fx(d)],
                &[fx(hc)],
                vec![
                    arg(x, vec![unit(3, 0), form(3, &[(2, d as i64), (1, 1)], 0)]),
                    arg(pre, vec![unit(3, 0), unit(3, 2)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    /// `out[t, k, i] = post[t, k]·h[t, i] + Σ_j comb[t, j, k]·res[t, j·d + i]`.
    fn hc_post(
        &mut self,
        h: TensorId,
        res: TensorId,
        post: TensorId,
        comb: TensorId,
        hc: usize,
        d: usize,
    ) -> TensorId {
        let acc = self
            .g
            .comp(
                &[T, fx(hc), fx(d)],
                &[fx(hc)],
                vec![
                    arg(comb, vec![unit(4, 0), unit(4, 3), unit(4, 1)]),
                    arg(res, vec![unit(4, 0), form(4, &[(3, d as i64), (2, 1)], 0)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let out = self
            .g
            .comp(
                &[T, fx(hc), fx(d)],
                &[],
                vec![
                    arg(post, vec![unit(3, 0), unit(3, 1)]),
                    arg(h, vec![unit(3, 0), unit(3, 2)]),
                    arg(acc, ident(3)),
                ],
                None,
                |v| vec![v[0] * v[1] + v[2]],
            )
            .unwrap();
        self.g.reshape(out, &[T, fx(hc * d)]).unwrap()
    }

    /// Masked reduction over keys `j` of a `[T, H, T]` score node: `f(s, extra…)`
    /// where `vis(t, j, extra…) > 0`, the monoid identity elsewhere.
    #[allow(clippy::too_many_arguments)]
    fn attn_reduce(
        &mut self,
        out: &[Dim],
        red_j: usize,
        args: Vec<Arg>,
        m: Monoid,
        id: f64,
        vis: impl Fn(ExprId, ExprId, &[ExprId]) -> ExprId,
        f: impl Fn(&[ExprId]) -> ExprId,
    ) -> TensorId {
        let mut sp = out.to_vec();
        sp.push(T);
        let rank = sp.len();
        assert_eq!(red_j, rank - 1);
        let (ti, ji) = (self.g.iota(&sp, 0), self.g.iota(&sp, rank - 1));
        let na = args.len();
        let mut args = args;
        args.push(arg(ti, ident(rank)));
        args.push(arg(ji, ident(rank)));
        self.g
            .comp(out, &[T], args, Some(m), |v| {
                let (t, j) = (v[na], v[na + 1]);
                vec![select(
                    t - j + lit(0.5),
                    select(vis(t, j, v), f(v), lit(id)),
                    lit(id),
                )]
            })
            .unwrap()
    }
}

/// The full-sequence graph.
pub fn prefill_graph(dm: &DsDims) -> PrefillGraph {
    let (d, h, hd, hc) = (dm.dim, dm.n_heads, dm.head_dim, dm.hc);
    let (ql, g, ol) = (dm.q_lora_rank, dm.o_groups, dm.o_lora_rank);
    let (nh, ihd, kk) = (dm.index_n_heads, dm.index_head_dim, dm.index_topk);
    let (ne, ka, inter) = (dm.n_experts, dm.n_active, dm.moe_inter);
    let win = dm.window as f64;
    let mut b = B {
        g: TensorGraph::new(),
        inputs: Vec::new(),
        named: Vec::new(),
        eps: dm.eps,
    };
    let tokens = b.input("tokens".into(), &[T]);
    let emb = b.w("embed.weight".into(), &[dm.vocab, d]);
    // the embedding row by token id, copied to every stream
    let e = b.g.gather(emb, 0, tokens, false).unwrap();
    let x3 =
        b.g.comp(
            &[T, fx(hc), fx(d)],
            &[],
            vec![arg(e, vec![unit(3, 0), unit(3, 2)])],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let mut x = b.g.reshape(x3, &[T, fx(hc * d)]).unwrap();
    let ci = b.g.iota(&[T, fx(hc)], 1);
    let mut pre_mix =
        b.g.comp(&[T, fx(hc)], &[], vec![arg(ci, ident(2))], None, |v| {
            vec![select(lit(0.5) - v[0], lit(1.0), lit(0.0))]
        })
        .unwrap();

    // per layer: (compressed kv [T, hd], top-k [T, 2k]) of the sources
    let mut ckv_of: Vec<Option<TensorId>> = vec![None; dm.n_layers];
    let mut top_of: Vec<Option<TensorId>> = vec![None; dm.n_layers];
    for l in 0..dm.n_layers {
        let p = format!("layers.{l}");
        let r = dm.compress_ratios[l];
        let fr = b.w(format!("{p}.freqs"), &[hd / 2]);
        // ---- attention sublayer
        let [apre, apost, acomb] = b.hc_mixes(&p, "attn", x, dm);
        let hx = b.hc_pre(x, pre_mix, hc, d);
        let an = b.w(format!("{p}.attn_norm.weight"), &[d]);
        let xn = b.rms(hx, an);

        let wq_a = b.w(format!("{p}.attn.wq_a.weight"), &[ql, d]);
        let q_norm = b.w(format!("{p}.attn.q_norm.weight"), &[ql]);
        let wq_b = b.w(format!("{p}.attn.wq_b.weight"), &[h * hd, ql]);
        let wkv = b.w(format!("{p}.attn.wkv.weight"), &[hd, d]);
        let kv_norm = b.w(format!("{p}.attn.kv_norm.weight"), &[hd]);
        let sink = b.w(format!("{p}.attn.attn_sink"), &[h]);
        let wo_a = b.w(format!("{p}.attn.wo_a.weight"), &[g * ol, h * hd / g]);
        let wo_b = b.w(format!("{p}.attn.wo_b.weight"), &[d, g * ol]);

        let qr0 = b.linear(xn, wq_a);
        let qr = b.rms(qr0, q_norm);
        let q0 = b.linear(qr, wq_b);
        let q = b.rope(q0, h, hd, fr, 0, false);
        let kv0 = b.linear(xn, wkv);
        let kv1 = b.rms(kv0, kv_norm);
        let kv = b.rope(kv1, 1, hd, fr, 0, false);
        b.name(format!("{p}.window_kv"), kv);

        if r > 0 {
            let rf = r as f64;
            if dm.kv_source_layers.contains(&l) {
                let cwkv = b.w(format!("{p}.attn.compressor.wkv.weight"), &[hd, d]);
                let cwg = b.w(format!("{p}.attn.compressor.wgate.weight"), &[hd, d]);
                let cnorm = b.w(format!("{p}.attn.compressor.norm.weight"), &[hd]);
                let kvp = b.linear(xn, cwkv);
                let scg = b.linear(xn, cwg);
                b.name(format!("{p}.compressor_kv"), kvp);
                b.name(format!("{p}.compressor_gate"), scg);
                // pooled at row t over its group j ∈ (t − r, t]
                let in_group = move |t: ExprId, j: ExprId, _: &[ExprId]| j - t + lit(rf - 0.5);
                let gsp = [T, fx(hd)];
                let gj = |rank: usize| vec![unit(rank, rank - 1), unit(rank, 1)];
                let mx = b.attn_reduce(
                    &gsp,
                    2,
                    vec![arg(scg, gj(3))],
                    Monoid::max(),
                    NEG,
                    in_group,
                    |v| v[0],
                );
                let mxa = arg(mx, vec![unit(3, 0), unit(3, 1)]);
                let num = b.attn_reduce(
                    &gsp,
                    2,
                    vec![arg(scg, gj(3)), arg(kvp, gj(3)), mxa.clone()],
                    Monoid::sum(),
                    0.0,
                    in_group,
                    |v| v[1] * (v[0] - v[2]).exp(),
                );
                let den = b.attn_reduce(
                    &gsp,
                    2,
                    vec![arg(scg, gj(3)), mxa],
                    Monoid::sum(),
                    0.0,
                    in_group,
                    |v| (v[0] - v[1]).exp(),
                );
                let pooled = b.g.map(&[num, den], |v| v[0] / v[1]).unwrap();
                let lat = b.rms(pooled, cnorm);
                let ckv = b.rope(lat, 1, hd, fr, -(r as i64 - 1), false);
                b.name(format!("{p}.compressed_kv"), ckv);
                ckv_of[l] = Some(ckv);

                if dm.index_source_layers.contains(&l) {
                    let fri = b.w(format!("{p}.attn.indexer.freqs"), &[ihd / 2]);
                    let wk = b.w(format!("{p}.attn.indexer.wk.weight"), &[ihd, hd]);
                    let knorm = b.w(format!("{p}.attn.indexer.k_norm.weight"), &[ihd]);
                    let wqb = b.w(format!("{p}.attn.indexer.wq_b.weight"), &[nh * ihd, ql]);
                    let wproj = b.w(format!("{p}.attn.indexer.weights_proj.weight"), &[nh, d]);
                    let k0 = b.linear(lat, wk);
                    let k1 = b.rms(k0, knorm);
                    let ik = b.rope(k1, 1, ihd, fri, -(r as i64 - 1), false);
                    b.name(format!("{p}.index_k"), ik);
                    let iq0 = b.linear(qr, wqb);
                    let iq = b.rope(iq0, nh, ihd, fri, 0, false);
                    let ws = (ihd as f64).powf(-0.5) * (nh as f64).powf(-0.5);
                    let wts0 = b.linear(xn, wproj);
                    let wts = b.g.map(&[wts0], |v| v[0] * lit(ws)).unwrap();
                    // dots[t, j, h] over c
                    let dots =
                        b.g.comp(
                            &[T, T, fx(nh)],
                            &[fx(ihd)],
                            vec![
                                arg(iq, vec![unit(4, 0), form(4, &[(2, ihd as i64), (3, 1)], 0)]),
                                arg(ik, vec![unit(4, 1), unit(4, 3)]),
                            ],
                            Some(Monoid::sum()),
                            |v| vec![v[0] * v[1]],
                        )
                        .unwrap();
                    let score =
                        b.g.comp(
                            &[T, T],
                            &[fx(nh)],
                            vec![arg(dots, ident(3)), arg(wts, vec![unit(3, 0), unit(3, 2)])],
                            Some(Monoid::sum()),
                            |v| vec![select(v[0], v[0], lit(0.0)) * v[1]],
                        )
                        .unwrap();
                    // top-k over the complete groups j ≤ t
                    let tsp = [T, T];
                    let (tt, tj) = (b.g.iota(&tsp, 0), b.g.iota(&tsp, 1));
                    let fmax = -(f32::MAX as f64);
                    let top =
                        b.g.comp_tuple(
                            &[T, fx(2 * kk)],
                            &[T],
                            vec![arg(score, ident(2)), arg(tt, ident(2)), arg(tj, ident(2))],
                            Some(Monoid::topk(kk)),
                            |v| {
                                let (s, t, j) = (v[0], v[1], v[2]);
                                let causal = t - j + lit(0.5);
                                // group end: (j + 1) mod r == 0
                                let m = j + lit(1.0);
                                let q = ((m + lit(0.5)) * lit(1.0 / rf)).floor();
                                let end = lit(0.5) - (m - lit(rf) * q);
                                let mut o = vec![
                                    select(causal, j, lit(0.0)),
                                    select(causal, select(end, s, lit(fmax)), lit(fmax)),
                                ];
                                for _ in 1..kk {
                                    o.push(select(causal, lit(0.0), lit(0.0)));
                                    o.push(select(causal, lit(fmax), lit(fmax)));
                                }
                                o
                            },
                        )
                        .unwrap();
                    b.name(format!("{p}.topk"), top);
                    top_of[l] = Some(top);
                }
            }
        }

        // scores [T, H, T] (window keys, then compressed rows)
        let scale = (hd as f64).powf(-0.5);
        let scores_vs = |b: &mut B, keys: TensorId| {
            b.g.comp(
                &[T, fx(h), T],
                &[fx(hd)],
                vec![
                    arg(q, vec![unit(4, 0), form(4, &[(1, hd as i64), (3, 1)], 0)]),
                    arg(keys, vec![unit(4, 2), unit(4, 3)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1] * lit(scale)],
            )
            .unwrap()
        };
        let sw = scores_vs(&mut b, kv);
        let in_win = move |t: ExprId, j: ExprId, _: &[ExprId]| j - t + lit(win - 0.5);
        let comp_src = (r > 0).then(|| {
            let c = ckv_of[dm.source(&dm.kv_source_layers, l)].unwrap();
            let tp = top_of[dm.source(&dm.index_source_layers, l)].unwrap();
            (c, tp)
        });
        let hsp = [T, fx(h)];
        let s3 = |src| arg(src, ident(3));
        let mut ms =
            vec![b.attn_reduce(&hsp, 2, vec![s3(sw)], Monoid::max(), NEG, in_win, |v| v[0])];
        // the compressed rows query t picked, read by index (a gather over
        // the sequence whose index is the causal top-k): slot s of row t is
        // `(idx, val)` at `tp[t, 2s]`, `tp[t, 2s + 1]`, real iff val > −1e29
        let kp = fx(kk);
        let slot = |tp: TensorId, r: usize, sd: usize, off: i64| {
            arg(tp, vec![unit(r, 0), form(r, &[(sd, 2)], off)])
        };
        // the source row picked by arg `ix`, other source dims `rest`
        let gathered = |src: TensorId, r: usize, c: usize, ix: usize| Arg {
            src,
            view: View {
                forms: vec![konst(r, 0), unit(r, c)],
                guard: false,
                gather: None,
            }
            .gather(0, ix, 1),
        };
        // sg[t, h, s] = q[t, h]·c[idx(t, s)]·scale: space (t, h, s, c)
        let sg = comp_src.map(|(c, tp)| {
            let sg =
                b.g.comp(
                    &[T, fx(h), kp],
                    &[fx(hd)],
                    vec![
                        arg(q, vec![unit(4, 0), form(4, &[(1, hd as i64), (3, 1)], 0)]),
                        gathered(c, 4, 3, 2),
                        slot(tp, 4, 2, 0),
                    ],
                    Some(Monoid::sum()),
                    |v| vec![v[0] * v[1] * lit(scale)],
                )
                .unwrap();
            (sg, c, tp)
        });
        if let Some((sg, _, tp)) = sg {
            ms.push(
                b.g.comp(
                    &hsp,
                    &[kp],
                    vec![s3(sg), slot(tp, 3, 2, 1)],
                    Some(Monoid::max()),
                    |v| vec![select(v[1] + lit(1e29), v[0], lit(NEG))],
                )
                .unwrap(),
            );
        }
        // the stabilizer: max of the visible scores and the sink
        let mut margs: Vec<Arg> = ms.iter().map(|&m| arg(m, ident(2))).collect();
        margs.push(arg(sink, vec![unit(2, 1)]));
        let mm =
            b.g.comp(&hsp, &[], margs, None, |v| {
                vec![v.iter().skip(1).fold(v[0], |a, &x| a.max(x))]
            })
            .unwrap();
        let m3 = || arg(mm, vec![unit(3, 0), unit(3, 1)]);
        let mut dens = vec![b.attn_reduce(
            &hsp,
            2,
            vec![s3(sw), m3()],
            Monoid::sum(),
            0.0,
            in_win,
            |v| (v[0] - v[1]).exp(),
        )];
        // o[t, h, c] over j: space (t, h, c, j)
        let osp = [T, fx(h), fx(hd)];
        let s4 = |src| arg(src, vec![unit(4, 0), unit(4, 1), unit(4, 3)]);
        let m4 = || arg(mm, vec![unit(4, 0), unit(4, 1)]);
        let k4 = |src| arg(src, vec![unit(4, 3), unit(4, 2)]);
        let mut nums = vec![b.attn_reduce(
            &osp,
            3,
            vec![s4(sw), m4(), k4(kv)],
            Monoid::sum(),
            0.0,
            in_win,
            |v| (v[0] - v[1]).exp() * v[2],
        )];
        if let Some((sg, c, tp)) = sg {
            dens.push(
                b.g.comp(
                    &hsp,
                    &[kp],
                    vec![s3(sg), m3(), slot(tp, 3, 2, 1)],
                    Some(Monoid::sum()),
                    |v| vec![select(v[2] + lit(1e29), (v[0] - v[1]).exp(), lit(0.0))],
                )
                .unwrap(),
            );
            // o[t, h, c] over s: space (t, h, c, s); arg 3 the index
            nums.push(
                b.g.comp(
                    &osp,
                    &[kp],
                    vec![
                        s4(sg),
                        m4(),
                        gathered(c, 4, 2, 3),
                        slot(tp, 4, 3, 0),
                        slot(tp, 4, 3, 1),
                    ],
                    Some(Monoid::sum()),
                    |v| {
                        vec![select(
                            v[4] + lit(1e29),
                            (v[0] - v[1]).exp() * v[2],
                            lit(0.0),
                        )]
                    },
                )
                .unwrap(),
            );
        }
        let nd = nums.len();
        let mut oargs: Vec<Arg> = nums.iter().map(|&n| arg(n, ident(3))).collect();
        oargs.extend(dens.iter().map(|&n| arg(n, vec![unit(3, 0), unit(3, 1)])));
        oargs.push(arg(mm, vec![unit(3, 0), unit(3, 1)]));
        oargs.push(arg(sink, vec![unit(3, 1)]));
        let o =
            b.g.comp(&osp, &[], oargs, None, |v| {
                let num = v[1..nd].iter().fold(v[0], |a, &x| a + x);
                let (m, s) = (v[2 * nd], v[2 * nd + 1]);
                let den = v[nd + 1..2 * nd].iter().fold(v[nd], |a, &x| a + x) + (s - m).exp();
                vec![num / den]
            })
            .unwrap();
        let o = b.g.reshape(o, &[T, fx(h * hd)]).unwrap();
        let o = b.rope(o, h, hd, fr, 0, true);
        let gd = h * hd / g;
        let mid =
            b.g.comp(
                &[T, fx(g), fx(ol)],
                &[fx(gd)],
                vec![
                    arg(o, vec![unit(4, 0), form(4, &[(1, gd as i64), (3, 1)], 0)]),
                    arg(
                        wo_a,
                        vec![form(4, &[(1, ol as i64), (2, 1)], 0), unit(4, 3)],
                    ),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let mid = b.g.reshape(mid, &[T, fx(g * ol)]).unwrap();
        let attn = b.linear(mid, wo_b);
        x = b.hc_post(attn, x, apost, acomb, hc, d);

        // ---- MoE sublayer
        let [fpre, fpost, fcomb] = b.hc_mixes(&p, "ffn", x, dm);
        let hx = b.hc_pre(x, apre, hc, d);
        let fnw = b.w(format!("{p}.ffn_norm.weight"), &[d]);
        let xn = b.rms(hx, fnw);
        let gate = b.w(format!("{p}.ffn.gate.weight"), &[ne, d]);
        let bias = b.w(format!("{p}.ffn.gate.bias"), &[ne]);
        let rows = if dm.expert_rows > 0 {
            dm.expert_rows
        } else {
            ne
        };
        let w1 = b.w(format!("{p}.ffn.experts.w1"), &[rows, inter, d]);
        let w3 = b.w(format!("{p}.ffn.experts.w3"), &[rows, inter, d]);
        let w2 = b.w(format!("{p}.ffn.experts.w2"), &[rows, d, inter]);
        let slot_map = (dm.expert_rows > 0).then(|| b.w(format!("{p}.ffn.expert_slot"), &[ne]));
        let s1 = b.w(format!("{p}.ffn.shared_experts.w1.weight"), &[inter, d]);
        let s3w = b.w(format!("{p}.ffn.shared_experts.w3.weight"), &[inter, d]);
        let s2 = b.w(format!("{p}.ffn.shared_experts.w2.weight"), &[d, inter]);
        let lg = b.linear(xn, gate);
        let sc =
            b.g.map(&[lg], |v| {
                let sp = select(v[0] - lit(20.0), v[0], (lit(1.0) + v[0].exp()).ln());
                sp.sqrt()
            })
            .unwrap();
        let ei = b.g.iota(&[T, fx(ne)], 1);
        let fmax = -(f32::MAX as f64);
        let etop =
            b.g.comp_tuple(
                &[T, fx(2 * ka)],
                &[fx(ne)],
                vec![
                    arg(sc, ident(2)),
                    arg(bias, vec![unit(2, 1)]),
                    arg(ei, ident(2)),
                ],
                Some(Monoid::topk(ka)),
                |v| {
                    let mut o = vec![v[2], v[0] + v[1]];
                    for _ in 1..ka {
                        o.push(lit(0.0));
                        o.push(lit(fmax));
                    }
                    o
                },
            )
            .unwrap();
        b.name(format!("{p}.expert_topk"), etop);
        // the picked expert ids `[T, ka]` (slot s at etop[t, 2s])
        let ids =
            |rank: usize, sd: usize| arg(etop, vec![unit(rank, 0), form(rank, &[(sd, 2)], 0)]);
        // the row of the expert tensors slot s reads
        let erow = slot_map.map(|m| {
            b.g.comp(
                &[T, fx(ka)],
                &[],
                vec![
                    ids(2, 1),
                    Arg {
                        src: m,
                        view: View::pick(2, &[None]).gather(0, 0, 1),
                    },
                ],
                None,
                |v| vec![v[1]],
            )
            .unwrap()
        });
        let row_ix = |rank: usize, sd: usize| match erow {
            Some(e) => arg(e, vec![unit(rank, 0), unit(rank, sd)]),
            None => ids(rank, sd),
        };
        // routing weights: the unbiased score of each pick (a one-hot sum
        // over the experts: a gather of the `[T, ne]` scores would be a
        // gather of a sequence-bearing tensor), renormalized, scaled
        let ei3 = b.g.iota(&[T, fx(ka), fx(ne)], 2);
        let wpick =
            b.g.comp(
                &[T, fx(ka)],
                &[fx(ne)],
                vec![
                    ids(3, 1),
                    arg(sc, vec![unit(3, 0), unit(3, 2)]),
                    arg(ei3, ident(3)),
                ],
                Some(Monoid::sum()),
                |v| vec![pick(v[0], v[2], v[1])],
            )
            .unwrap();
        let (norm, rs) = (dm.norm_topk_prob && ka > 1, dm.route_scale);
        let wsum =
            b.g.comp(
                &[T],
                &[fx(ka)],
                vec![arg(wpick, ident(2))],
                Some(Monoid::sum()),
                |v| vec![v[0]],
            )
            .unwrap();
        let wt =
            b.g.comp(
                &[T, fx(ka)],
                &[],
                vec![arg(wpick, ident(2)), arg(wsum, vec![unit(2, 0)])],
                None,
                |v| {
                    vec![
                        if norm {
                            v[0] / (v[1] + lit(1e-20))
                        } else {
                            v[0]
                        } * lit(rs),
                    ]
                },
            )
            .unwrap();
        let lim = dm.swiglu_limit;
        let swiglu = move |g: ExprId, u: ExprId| -> ExprId {
            let (g, u) = if lim > 0.0 {
                let gl = select(g - lit(lim), lit(lim), g);
                let ul = select(u - lit(lim), lit(lim), select(-u - lit(lim), lit(-lim), u));
                (gl, ul)
            } else {
                (g, u)
            };
            g * sigmoid(g) * u
        };
        // each picked expert's gate/up rows, gathered by its row: space (t, slot, f, n)
        let up = |b: &mut B, w: TensorId| {
            b.g.comp(
                &[T, fx(ka), fx(inter)],
                &[fx(d)],
                vec![
                    row_ix(4, 1),
                    arg(xn, vec![unit(4, 0), unit(4, 3)]),
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
        let (eg, eu) = (up(&mut b, w1), up(&mut b, w3));
        let eh =
            b.g.comp(
                &[T, fx(ka), fx(inter)],
                &[],
                vec![
                    arg(eg, ident(3)),
                    arg(eu, ident(3)),
                    arg(wt, vec![unit(3, 0), unit(3, 1)]),
                ],
                None,
                |v| vec![v[2] * swiglu(v[0], v[1])],
            )
            .unwrap();
        // each pick's down projection: space (t, slot, n, f)
        let dn =
            b.g.comp(
                &[T, fx(ka), fx(d)],
                &[fx(inter)],
                vec![
                    row_ix(4, 1),
                    arg(eh, vec![unit(4, 0), unit(4, 1), unit(4, 3)]),
                    Arg {
                        src: w2,
                        view: View::pick(4, &[None, Some(2), Some(3)]).gather(0, 0, 1),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[1] * v[2]],
            )
            .unwrap();
        let routed =
            b.g.comp(
                &[T, fx(d)],
                &[fx(ka)],
                vec![arg(dn, vec![unit(3, 0), unit(3, 2), unit(3, 1)])],
                Some(Monoid::sum()),
                |v| vec![v[0]],
            )
            .unwrap();
        let sg = b.linear(xn, s1);
        let su = b.linear(xn, s3w);
        let sh = b.g.map(&[sg, su], |v| swiglu(v[0], v[1])).unwrap();
        let shared = b.linear(sh, s2);
        let moe = b.g.map(&[routed, shared], |v| v[0] + v[1]).unwrap();
        x = b.hc_post(moe, x, fpost, fcomb, hc, d);
        b.name(format!("{p}.stream"), x);
        // the next block's attention collapses with this block's ffn pre-mix
        pre_mix = fpre;
    }
    let hx = b.hc_pre(x, pre_mix, hc, d);
    let nw = b.w("norm.weight".into(), &[d]);
    let xn = b.rms(hx, nw);
    let head = b.w("head.weight".into(), &[dm.vocab, d]);
    let logits = b.linear(xn, head);
    PrefillGraph {
        g: b.g,
        inputs: b.inputs,
        logits,
        named: b.named,
    }
}

/// The derived step of `pg` appending `k` tokens against caches of `dm.ctx`
/// positions: outputs `[logits]`, caches in `caches`.
pub fn derived_step(pg: &PrefillGraph, dm: &DsDims, k: usize) -> Result<StepGraph, String> {
    derive_step(
        &pg.g,
        &[pg.logits],
        &StepSpec {
            seq: SEQ,
            new: k,
            capacity: dm.ctx,
        },
    )
    .map_err(|e| format!("{e:?}"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;
    use tang_compile::tensor::step::{seed_cache, StepInput};

    pub(crate) type Weights = HashMap<String, Vec<f64>>;

    fn weights(pg: &PrefillGraph, dm: &DsDims) -> Weights {
        let rd = dm.rope_head_dim;
        pg.inputs
            .iter()
            .filter(|s| s.name != "tokens")
            .map(|s| {
                let shape: Vec<usize> = s.shape.iter().map(|d| d.n()).collect();
                let n: usize = shape.iter().product();
                let mut seed = s.name.bytes().fold(1469598103934665603u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(1099511628211)
                });
                let mut rnd = move || {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed >> 11) as f64 / (1u64 << 53) as f64
                };
                let nm = s.name.as_str();
                let v: Vec<f64> = if nm.ends_with("freqs") {
                    let nope = n - rd / 2;
                    (0..n)
                        .map(|j| {
                            if j < nope {
                                0.0
                            } else {
                                100f64.powf(-2.0 * (j - nope) as f64 / rd as f64)
                            }
                        })
                        .collect()
                } else if nm.contains("norm") {
                    (0..n).map(|_| 1.0 + 0.4 * (rnd() - 0.5)).collect()
                } else if nm.ends_with("attn_sink") {
                    (0..n).map(|_| 2.0 * rnd() - 1.0).collect()
                } else if nm.ends_with("_scale") {
                    (0..n).map(|_| 0.5 + rnd()).collect()
                } else if nm.ends_with("_base") || nm.ends_with("gate.bias") {
                    (0..n).map(|_| 0.5 * (rnd() - 0.5)).collect()
                } else {
                    let cols = *shape.last().unwrap() as f64;
                    (0..n)
                        .map(|_| 1.5 * (2.0 * rnd() - 1.0) / cols.sqrt())
                        .collect()
                };
                (s.name.clone(), v)
            })
            .collect()
    }

    fn tokens(n: usize, vocab: usize) -> Vec<f64> {
        (0..n).map(|i| ((37 * i + 5) % vocab) as f64).collect()
    }

    fn inputs(pg: &PrefillGraph, w: &Weights, toks: &[f64]) -> Vec<Vec<f64>> {
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

    pub(crate) fn rel(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = (x - y).abs() / scale;
                if d.is_nan() {
                    f64::INFINITY
                } else {
                    d
                }
            })
            .fold(0.0, f64::max)
    }

    // ---- the reference: forward.rs's prefill (sp = 0), in f64, no rounding

    /// Knobs for negative controls.
    #[derive(Clone, Copy, Default)]
    pub(crate) struct Bugs {
        window_plus_one: bool,
        drop_sink: bool,
        no_compressed: bool,
    }

    /// Routed expert `e` of layer `l` as `(w1, w3, w2)`, when the weights
    /// don't hold the fused expert tensors (real size: loaded per pick).
    pub(crate) type LoadExpert<'a> = &'a dyn Fn(usize, usize) -> (Vec<f64>, Vec<f64>, Vec<f64>);

    pub(crate) struct Ref<'a> {
        pub(crate) dm: &'a DsDims,
        pub(crate) w: &'a Weights,
        pub(crate) bugs: Bugs,
        pub(crate) load: Option<LoadExpert<'a>>,
    }

    fn mm(x: &[f64], t: usize, w: &[f64], n: usize, k: usize) -> Vec<f64> {
        assert_eq!(x.len(), t * k);
        let mut y = vec![0.0; t * n];
        for ti in 0..t {
            for ni in 0..n {
                y[ti * n + ni] = (0..k).map(|ki| x[ti * k + ki] * w[ni * k + ki]).sum();
            }
        }
        y
    }

    fn sig(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn topk_stable(v: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(k);
        idx
    }

    /// forward.rs `rope`: the last `rd` of every `d`-wide vector, adjacent pairs.
    fn rope(
        v: &mut [f64],
        d: usize,
        rd: usize,
        per_row: usize,
        fr: &[f64],
        pos: impl Fn(usize) -> f64,
        inv: bool,
    ) {
        let tail_fr = &fr[fr.len() - rd / 2..];
        for (r, row) in v.chunks_mut(d * per_row).enumerate() {
            for vec in row.chunks_mut(d) {
                let tail = &mut vec[d - rd..];
                for (j, &f) in tail_fr.iter().enumerate() {
                    let a = pos(r) * f;
                    let (c, s) = (a.cos(), if inv { -a.sin() } else { a.sin() });
                    let (x, y) = (tail[2 * j], tail[2 * j + 1]);
                    tail[2 * j] = x * c - y * s;
                    tail[2 * j + 1] = x * s + y * c;
                }
            }
        }
    }

    impl Ref<'_> {
        fn w(&self, n: &str) -> &[f64] {
            self.w.get(n).unwrap_or_else(|| panic!("no weight {n}"))
        }

        fn linear(&self, x: &[f64], t: usize, n: &str, rows: usize) -> Vec<f64> {
            let k = x.len() / t;
            mm(x, t, self.w(n), rows, k)
        }

        fn rms(&self, x: &mut [f64], n: &str) {
            let g = self.w(n);
            let k = g.len();
            for row in x.chunks_mut(k) {
                let ms = row.iter().map(|v| v * v).sum::<f64>() / k as f64;
                let r = 1.0 / (ms + self.dm.eps).sqrt();
                for (v, &gi) in row.iter_mut().zip(g) {
                    *v = gi * (*v * r);
                }
            }
        }

        fn sinkhorn(
            &self,
            m: &[f64],
            scale: &[f64],
            base: &[f64],
        ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let (hc, eps) = (self.dm.hc, self.dm.hc_eps);
            let pre = (0..hc)
                .map(|j| sig(m[j] * scale[0] + base[j]) + eps)
                .collect();
            let post = (0..hc)
                .map(|j| 2.0 * sig(m[hc + j] * scale[1] + base[hc + j]))
                .collect();
            let mut comb: Vec<f64> = (0..hc * hc)
                .map(|i| m[2 * hc + i] * scale[2] + base[2 * hc + i])
                .collect();
            for row in comb.chunks_mut(hc) {
                let mx = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut s = 0.0;
                for x in row.iter_mut() {
                    *x = (*x - mx).exp();
                    s += *x;
                }
                for x in row.iter_mut() {
                    *x = *x / s + eps;
                }
            }
            let colnorm = |comb: &mut [f64]| {
                for k in 0..hc {
                    let s: f64 = (0..hc).map(|j| comb[j * hc + k]).sum();
                    for j in 0..hc {
                        comb[j * hc + k] /= s + eps;
                    }
                }
            };
            colnorm(&mut comb);
            for _ in 1..self.dm.sinkhorn_iters {
                for row in comb.chunks_mut(hc) {
                    let s: f64 = row.iter().sum();
                    for x in row.iter_mut() {
                        *x /= s + eps;
                    }
                }
                colnorm(&mut comb);
            }
            (pre, post, comb)
        }

        #[allow(clippy::type_complexity)]
        fn hc_mixes(&self, p: &str, kind: &str, x: &[f64]) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
            let (hc, d, mix) = (self.dm.hc, self.dm.dim, self.dm.mix());
            let t = x.len() / (hc * d);
            let mixes = self.linear(x, t, &format!("{p}.hc_{kind}_fn"), mix);
            let (scale, base) = (
                self.w(&format!("{p}.hc_{kind}_scale")),
                self.w(&format!("{p}.hc_{kind}_base")),
            );
            (0..t)
                .map(|ti| {
                    let xr = &x[ti * hc * d..(ti + 1) * hc * d];
                    let rs = 1.0
                        / (xr.iter().map(|v| v * v).sum::<f64>() / (hc * d) as f64 + self.dm.eps)
                            .sqrt();
                    let m: Vec<f64> = mixes[ti * mix..(ti + 1) * mix]
                        .iter()
                        .map(|v| v * rs)
                        .collect();
                    self.sinkhorn(&m, scale, base)
                })
                .collect()
        }

        fn hc_pre(&self, x: &[f64], pre: &[Vec<f64>]) -> Vec<f64> {
            let (hc, d) = (self.dm.hc, self.dm.dim);
            let mut out = vec![0.0; pre.len() * d];
            for (ti, pm) in pre.iter().enumerate() {
                for (ci, &w) in pm.iter().enumerate() {
                    for i in 0..d {
                        out[ti * d + i] += w * x[(ti * hc + ci) * d + i];
                    }
                }
            }
            out
        }

        #[allow(clippy::type_complexity)]
        fn hc_post(
            &self,
            h: &[f64],
            res: &[f64],
            mixes: &[(Vec<f64>, Vec<f64>, Vec<f64>)],
        ) -> Vec<f64> {
            let (hc, d) = (self.dm.hc, self.dm.dim);
            let mut out = vec![0.0; res.len()];
            for (ti, (_, post, comb)) in mixes.iter().enumerate() {
                for k in 0..hc {
                    for i in 0..d {
                        let mut acc = 0.0;
                        for j in 0..hc {
                            acc += comb[j * hc + k] * res[(ti * hc + j) * d + i];
                        }
                        out[(ti * hc + k) * d + i] = post[k] * h[ti * d + i] + acc;
                    }
                }
            }
            out
        }

        /// Returns the attention output; fills `ckv` / `topk` for sources.
        fn attention(
            &self,
            l: usize,
            x: &[f64],
            ckv: &mut HashMap<usize, Vec<f64>>,
            topk: &mut HashMap<usize, Vec<Vec<i64>>>,
        ) -> Vec<f64> {
            let dm = self.dm;
            let p = format!("layers.{l}");
            let (h, hd, rd, d) = (dm.n_heads, dm.head_dim, dm.rope_head_dim, dm.dim);
            let r = dm.compress_ratios[l];
            let t = x.len() / d;
            let fr = self.w(&format!("{p}.freqs")).to_vec();
            let mut qr = self.linear(x, t, &format!("{p}.attn.wq_a.weight"), dm.q_lora_rank);
            self.rms(&mut qr, &format!("{p}.attn.q_norm.weight"));
            let mut q = self.linear(&qr, t, &format!("{p}.attn.wq_b.weight"), h * hd);
            rope(&mut q, hd, rd, h, &fr, |i| i as f64, false);
            let mut kv = self.linear(x, t, &format!("{p}.attn.wkv.weight"), hd);
            self.rms(&mut kv, &format!("{p}.attn.kv_norm.weight"));
            rope(&mut kv, hd, rd, 1, &fr, |i| i as f64, false);
            let win = dm.window + self.bugs.window_plus_one as usize;
            let mut idxs: Vec<Vec<i64>> = (0..t)
                .map(|e| {
                    ((e + 1).saturating_sub(win)..=e)
                        .map(|j| j as i64)
                        .collect()
                })
                .collect();
            if r > 0 && !self.bugs.no_compressed {
                let offset = t as i64;
                if dm.kv_source_layers.contains(&l) {
                    let ng = t / r;
                    let kvp = self.linear(x, t, &format!("{p}.attn.compressor.wkv.weight"), hd);
                    let sc = self.linear(x, t, &format!("{p}.attn.compressor.wgate.weight"), hd);
                    let mut lat = Vec::with_capacity(ng * hd);
                    for gi in 0..ng {
                        for ch in 0..hd {
                            let rows = gi * r..(gi + 1) * r;
                            let mx = rows
                                .clone()
                                .map(|j| sc[j * hd + ch])
                                .fold(f64::NEG_INFINITY, f64::max);
                            let s: f64 = rows.clone().map(|j| (sc[j * hd + ch] - mx).exp()).sum();
                            lat.push(
                                rows.map(|j| kvp[j * hd + ch] * (sc[j * hd + ch] - mx).exp() / s)
                                    .sum(),
                            );
                        }
                    }
                    self.rms(&mut lat, &format!("{p}.attn.compressor.norm.weight"));
                    if dm.index_source_layers.contains(&l) {
                        let sel = if ng == 0 {
                            vec![Vec::new(); t]
                        } else {
                            self.indexer(l, x, &qr, &lat)
                        };
                        topk.insert(l, sel);
                    }
                    rope(&mut lat, hd, rd, 1, &fr, |j| (j * r) as f64, false);
                    ckv.insert(l, lat);
                }
                let c = &ckv[&dm.source(&dm.kv_source_layers, l)];
                kv.extend_from_slice(c);
                let tk = &topk[&dm.source(&dm.index_source_layers, l)];
                for (a, b) in idxs.iter_mut().zip(tk) {
                    a.extend(b.iter().filter(|&&i| i >= 0).map(|&i| i + offset));
                }
            }
            // sparse_attn with the sink
            let sink = self.w(&format!("{p}.attn.attn_sink"));
            let scale = (hd as f64).powf(-0.5);
            let mut o = vec![0.0; t * h * hd];
            for ti in 0..t {
                for hi in 0..h {
                    let qv = &q[(ti * h + hi) * hd..(ti * h + hi + 1) * hd];
                    let s: Vec<f64> = idxs[ti]
                        .iter()
                        .map(|&r| {
                            qv.iter()
                                .zip(&kv[r as usize * hd..(r as usize + 1) * hd])
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    let mx = s
                        .iter()
                        .fold(f64::NEG_INFINITY, |m, &x| m.max(x))
                        .max(-1e30);
                    let pr: Vec<f64> = s.iter().map(|&x| (x - mx).exp()).collect();
                    let sk = if self.bugs.drop_sink {
                        0.0
                    } else {
                        (sink[hi] - mx).exp()
                    };
                    let denom = pr.iter().sum::<f64>() + sk;
                    let out = &mut o[(ti * h + hi) * hd..(ti * h + hi + 1) * hd];
                    for (&r, &pi) in idxs[ti].iter().zip(&pr) {
                        for (ov, &k) in out
                            .iter_mut()
                            .zip(&kv[r as usize * hd..(r as usize + 1) * hd])
                        {
                            *ov += pi * k;
                        }
                    }
                    for ov in out.iter_mut() {
                        *ov /= denom;
                    }
                }
            }
            rope(&mut o, hd, rd, h, &fr, |i| i as f64, true);
            let (g, ol) = (dm.o_groups, dm.o_lora_rank);
            let gd = h * hd / g;
            let wo_a = self.w(&format!("{p}.attn.wo_a.weight"));
            let mut mid = vec![0.0; t * g * ol];
            for ti in 0..t {
                for gi in 0..g {
                    let ov = &o[(ti * g + gi) * gd..(ti * g + gi + 1) * gd];
                    for ri in 0..ol {
                        let wr = &wo_a[(gi * ol + ri) * gd..(gi * ol + ri + 1) * gd];
                        mid[(ti * g + gi) * ol + ri] = ov.iter().zip(wr).map(|(a, b)| a * b).sum();
                    }
                }
            }
            self.linear(&mid, t, &format!("{p}.attn.wo_b.weight"), d)
        }

        fn indexer(&self, l: usize, x: &[f64], qr: &[f64], lat: &[f64]) -> Vec<Vec<i64>> {
            let dm = self.dm;
            let p = format!("layers.{l}");
            let (r, rd, nh, ihd, hd) = (
                dm.compress_ratios[l],
                dm.rope_head_dim,
                dm.index_n_heads,
                dm.index_head_dim,
                dm.head_dim,
            );
            let t = x.len() / dm.dim;
            let n = lat.len() / hd;
            let fri = self.w(&format!("{p}.attn.indexer.freqs")).to_vec();
            let mut k = self.linear(lat, n, &format!("{p}.attn.indexer.wk.weight"), ihd);
            self.rms(&mut k, &format!("{p}.attn.indexer.k_norm.weight"));
            rope(&mut k, ihd, rd, 1, &fri, |j| (j * r) as f64, false);
            let mut q = self.linear(qr, t, &format!("{p}.attn.indexer.wq_b.weight"), nh * ihd);
            rope(&mut q, ihd, rd, nh, &fri, |i| i as f64, false);
            let ws = (ihd as f64).powf(-0.5) * (nh as f64).powf(-0.5);
            let wts: Vec<f64> = self
                .linear(x, t, &format!("{p}.attn.indexer.weights_proj.weight"), nh)
                .iter()
                .map(|v| v * ws)
                .collect();
            let width = t / r;
            let mut score = vec![0.0; t * width];
            for ti in 0..t {
                for j in 0..width {
                    let kj = &k[j * ihd..(j + 1) * ihd];
                    let mut s = 0.0;
                    for hi in 0..nh {
                        let qv = &q[(ti * nh + hi) * ihd..(ti * nh + hi + 1) * ihd];
                        let dd: f64 = qv.iter().zip(kj).map(|(a, b)| a * b).sum();
                        s += dd.max(0.0) * wts[ti * nh + hi];
                    }
                    score[ti * width + j] = s;
                }
            }
            let lens: Vec<usize> = (0..t).map(|ti| (ti + 1) / r).collect();
            for ti in 0..t {
                for j in lens[ti]..width {
                    score[ti * width + j] = f64::NEG_INFINITY;
                }
            }
            let kk = dm.index_topk.min(width);
            (0..t)
                .map(|ti| {
                    let mut sel = topk_stable(&score[ti * width..(ti + 1) * width], kk);
                    sel.sort();
                    sel.into_iter()
                        .map(|i| if i < lens[ti] { i as i64 } else { -1 })
                        .collect()
                })
                .collect()
        }

        fn expert(
            &self,
            x: &[f64],
            t: usize,
            w1: &[f64],
            w3: &[f64],
            w2: &[f64],
            wts: Option<&[f64]>,
        ) -> Vec<f64> {
            let (d, inter, lim) = (self.dm.dim, self.dm.moe_inter, self.dm.swiglu_limit);
            let g = mm(x, t, w1, inter, d);
            let u = mm(x, t, w3, inter, d);
            let hh: Vec<f64> = g
                .iter()
                .zip(&u)
                .enumerate()
                .map(|(i, (&g, &u))| {
                    let (g, u) = if lim > 0.0 {
                        (g.min(lim), u.clamp(-lim, lim))
                    } else {
                        (g, u)
                    };
                    let v = g / (1.0 + (-g).exp()) * u;
                    match wts {
                        Some(w) => w[i / inter] * v,
                        None => v,
                    }
                })
                .collect();
            mm(&hh, t, w2, d, inter)
        }

        fn moe(&self, l: usize, x: &[f64]) -> Vec<f64> {
            let dm = self.dm;
            let p = format!("layers.{l}");
            let (ne, k, d, inter) = (dm.n_experts, dm.n_active, dm.dim, dm.moe_inter);
            let t = x.len() / d;
            let sc: Vec<f64> = self
                .linear(x, t, &format!("{p}.ffn.gate.weight"), ne)
                .into_iter()
                .map(|s| (if s > 20.0 { s } else { s.exp().ln_1p() }).sqrt())
                .collect();
            let bias = self.w(&format!("{p}.ffn.gate.bias"));
            let fused = |k: &str| -> &[f64] {
                if self.load.is_some() {
                    &[]
                } else {
                    self.w(&format!("{p}.ffn.experts.{k}"))
                }
            };
            let (w1, w3, w2) = (fused("w1"), fused("w3"), fused("w2"));
            let slot = self.w.get(&format!("{p}.ffn.expert_slot"));
            let mut y = vec![0.0; t * d];
            for ti in 0..t {
                let row = &sc[ti * ne..(ti + 1) * ne];
                let biased: Vec<f64> = row.iter().zip(bias).map(|(a, b)| a + b).collect();
                let sel = topk_stable(&biased, k);
                let mut wv: Vec<f64> = sel.iter().map(|&e| row[e]).collect();
                if dm.norm_topk_prob && k > 1 {
                    let s: f64 = wv.iter().sum::<f64>() + 1e-20;
                    wv.iter_mut().for_each(|v| *v /= s);
                }
                let xr = &x[ti * d..(ti + 1) * d];
                for (e, wv) in sel.into_iter().zip(wv) {
                    let ws = [wv * dm.route_scale];
                    let o = if let Some(load) = self.load {
                        let (a, b, c) = load(l, e);
                        self.expert(xr, 1, &a, &b, &c, Some(&ws))
                    } else {
                        let e = slot.map_or(e, |m| m[e] as usize);
                        let (a, b) = (e * inter * d, (e + 1) * inter * d);
                        self.expert(
                            xr,
                            1,
                            &w1[a..b],
                            &w3[a..b],
                            &w2[e * d * inter..(e + 1) * d * inter],
                            Some(&ws),
                        )
                    };
                    for i in 0..d {
                        y[ti * d + i] += o[i];
                    }
                }
            }
            let sh = self.expert(
                x,
                t,
                self.w(&format!("{p}.ffn.shared_experts.w1.weight")),
                self.w(&format!("{p}.ffn.shared_experts.w3.weight")),
                self.w(&format!("{p}.ffn.shared_experts.w2.weight")),
                None,
            );
            y.iter().zip(&sh).map(|(a, b)| a + b).collect()
        }

        pub(crate) fn forward(&self, toks: &[f64]) -> Vec<f64> {
            let dm = self.dm;
            let (hc, d, t) = (dm.hc, dm.dim, toks.len());
            let e = self.w("embed.weight");
            let mut h = Vec::with_capacity(t * hc * d);
            for &id in toks {
                for _ in 0..hc {
                    h.extend_from_slice(&e[id as usize * d..(id as usize + 1) * d]);
                }
            }
            let mut pre_mix: Vec<Vec<f64>> = (0..t)
                .map(|_| (0..hc).map(|i| (i == 0) as u8 as f64).collect())
                .collect();
            let (mut ckv, mut topk) = (HashMap::new(), HashMap::new());
            for l in 0..dm.n_layers {
                let p = format!("layers.{l}");
                let am = self.hc_mixes(&p, "attn", &h);
                let mut x = self.hc_pre(&h, &pre_mix);
                self.rms(&mut x, &format!("{p}.attn_norm.weight"));
                let a = self.attention(l, &x, &mut ckv, &mut topk);
                h = self.hc_post(&a, &h, &am);
                let fm = self.hc_mixes(&p, "ffn", &h);
                let a_pre: Vec<Vec<f64>> = am.iter().map(|m| m.0.clone()).collect();
                let mut x = self.hc_pre(&h, &a_pre);
                self.rms(&mut x, &format!("{p}.ffn_norm.weight"));
                let y = self.moe(l, &x);
                h = self.hc_post(&y, &h, &fm);
                pre_mix = fm.into_iter().map(|m| m.0).collect();
            }
            let mut x = self.hc_pre(&h, &pre_mix);
            self.rms(&mut x, "norm.weight");
            self.linear(&x, t, "head.weight", dm.vocab)
        }
    }

    const LENS: [usize; 6] = [1, 2, 3, 6, 9, 13];

    /// The prefill graph on the f64 interpreter against the direct f64
    /// transcription of forward.rs, every position's logits, several T.
    #[test]
    fn prefill_matches_reference() {
        let dm = DsDims::tiny();
        let pg = prefill_graph(&dm);
        let w = weights(&pg, &dm);
        let all = tokens(dm.ctx, dm.vocab);
        let mut worst = 0.0f64;
        for &t in &LENS {
            let g = pg.g.bind(&[(SEQ, t)]).unwrap();
            let got = g.eval(&inputs(&pg, &w, &all[..t]), &[pg.logits]);
            let want = Ref {
                dm: &dm,
                w: &w,
                bugs: Bugs::default(),
                load: None,
            }
            .forward(&all[..t]);
            let e = rel(&got[0], &want);
            eprintln!("ds tiny T={t}: max rel err {e:.2e}");
            worst = worst.max(e);
        }
        assert!(worst < 1e-10, "{worst:e}");
    }

    /// The same comparison against deliberately wrong references must fail.
    #[test]
    fn negative_controls_fail() {
        let dm = DsDims::tiny();
        let pg = prefill_graph(&dm);
        let w = weights(&pg, &dm);
        let t = 9;
        let all = tokens(t, dm.vocab);
        let g = pg.g.bind(&[(SEQ, t)]).unwrap();
        let got = g.eval(&inputs(&pg, &w, &all), &[pg.logits]);
        for (name, bugs) in [
            (
                "window + 1",
                Bugs {
                    window_plus_one: true,
                    ..Default::default()
                },
            ),
            (
                "no sink",
                Bugs {
                    drop_sink: true,
                    ..Default::default()
                },
            ),
            (
                "no compressed rows",
                Bugs {
                    no_compressed: true,
                    ..Default::default()
                },
            ),
        ] {
            let want = Ref {
                dm: &dm,
                w: &w,
                bugs,
                load: None,
            }
            .forward(&all);
            let e = rel(&got[0], &want);
            eprintln!("negative control ({name}): max rel err {e:.2e}");
            assert!(e > 1e-3, "{name}: {e:e} — the test can't tell");
        }
    }

    /// A cache slot's node by name (a named reshape counts for its source).
    fn cache_name(pg: &PrefillGraph, step: &StepGraph, c: usize) -> String {
        let n = step.caches[c].node;
        let via = pg.named.iter().find(|(_, t)| {
            *t == n || matches!(pg.g.nodes()[t.index()].op, tang_compile::tensor::Op::Reshape(s) if s == n)
        });
        via.map_or_else(|| format!("{n:?}"), |(s, _)| s.clone())
    }

    fn shape_at(pg: &PrefillGraph, t: TensorId, rows: usize) -> Vec<usize> {
        pg.g.shape(t)
            .iter()
            .map(|&d| if d == T { rows } else { d.n() })
            .collect()
    }

    /// What derive_step makes of the DS graph: the cache it picks, and its
    /// steps against prefills of the grown sequence.
    #[test]
    fn derived_step_matches_prefill() {
        let dm = DsDims::tiny();
        let pg = prefill_graph(&dm);
        let step = match derived_step(&pg, &dm, 1) {
            Ok(s) => s,
            Err(e) => panic!("derive_step refused: {e}"),
        };
        for (c, slot) in step.caches.iter().enumerate() {
            eprintln!(
                "cache slot {c}: {} axis {} shape {:?}",
                cache_name(&pg, &step, c),
                slot.axis,
                step.graph.shape(slot.input)
            );
        }
        let w = weights(&pg, &dm);
        let all = tokens(dm.ctx, dm.vocab);
        for ks in [&[1usize][..], &[4], &[4, 1, 1, 2]] {
            let mut p0 = 5;
            let fg = pg.g.bind(&[(SEQ, p0)]).unwrap();
            let nodes: Vec<TensorId> = step.caches.iter().map(|c| c.node).collect();
            let vals = fg.eval(&inputs(&pg, &w, &all[..p0]), &nodes);
            let mut caches: Vec<Vec<f64>> = nodes
                .iter()
                .zip(vals)
                .zip(&step.caches)
                .map(|((&n, v), c)| seed_cache(c, &shape_at(&pg, n, p0), &v, p0, dm.ctx))
                .collect();
            let mut worst = 0.0f64;
            for &k in ks {
                let st = derived_step(&pg, &dm, k).unwrap();
                assert_eq!(st.caches.len(), caches.len());
                let ins: Vec<Vec<f64>> = st
                    .inputs
                    .iter()
                    .map(|&si| match si {
                        StepInput::Pos => vec![p0 as f64],
                        StepInput::New(_) => all[p0..p0 + k].to_vec(),
                        StepInput::Same(i) => w[&pg.inputs[i].name].clone(),
                        StepInput::Cache(c) => caches[c].clone(),
                    })
                    .collect();
                let mut ids = st.outputs.clone();
                ids.extend(st.caches.iter().map(|c| c.updated));
                let got = st.graph.eval(&ins, &ids);
                assert!(got[0].iter().all(|x| x.is_finite()), "logits not finite");
                let grown = p0 + k;
                let fg = pg.g.bind(&[(SEQ, grown)]).unwrap();
                let want = fg.eval(&inputs(&pg, &w, &all[..grown]), &[pg.logits]);
                let v = dm.vocab;
                worst = worst.max(rel(&got[0], &want[0][p0 * v..grown * v]));
                for c in 0..caches.len() {
                    caches[c] = got[1 + c].clone();
                }
                p0 = grown;
            }
            eprintln!("ds tiny: derived steps {ks:?} from 5 vs prefill: max rel err {worst:.2e}");
            assert!(worst < 1e-10, "{ks:?}: {worst:e}");
        }
    }

    /// The derived step's top-k attention reads the picked compressed rows
    /// by gather (`index_topk` of them), not every cache row.
    #[test]
    fn derived_step_attention_reads_k_rows() {
        use tang_compile::tensor::Op;
        let dm = DsDims::tiny();
        let pg = prefill_graph(&dm);
        let st = derived_step(&pg, &dm, 1).unwrap();
        let ckv: Vec<TensorId> = st
            .caches
            .iter()
            .filter(|c| {
                cache_name(
                    &pg,
                    &st,
                    st.caches.iter().position(|x| x.node == c.node).unwrap(),
                )
                .ends_with("compressed_kv")
            })
            .map(|c| c.updated)
            .collect();
        let mut gathers = 0;
        for nd in st.graph.nodes() {
            let Op::Comp(c) = &nd.op else { continue };
            for a in c.args.iter().filter(|a| a.view.is_gather()) {
                gathers += 1;
                eprintln!(
                    "step gather: out {:?} red {:?} source shape {:?}",
                    c.out,
                    c.red,
                    st.graph.shape(a.src)
                );
            }
        }
        // the compressed rows (the cache or a reshape of it) are read only
        // through gathers
        let mut srcs: Vec<usize> = ckv.iter().map(|t| t.index()).collect();
        for (i, nd) in st.graph.nodes().iter().enumerate() {
            if matches!(nd.op, Op::Reshape(r) if srcs.contains(&r.index())) {
                srcs.push(i);
            }
        }
        let dense_reads = st
            .graph
            .nodes()
            .iter()
            .filter_map(|nd| match &nd.op {
                Op::Comp(c) => Some(c),
                _ => None,
            })
            .filter(|c| {
                c.args
                    .iter()
                    .any(|a| !a.view.is_gather() && srcs.contains(&a.src.index()))
            })
            .count();
        eprintln!("ds step: {gathers} gather reads; reductions over the cache reading compressed_kv: {dense_reads}");
        assert!(gathers >= 2);
        assert_eq!(dense_reads, 0);
    }

    /// CUDA lowering of the derived step (source only, no device): the
    /// kernels that read compressed_kv by gather, and their loops.
    #[test]
    fn derived_step_cuda_kernels() {
        use tang_compile::tensor::Op;
        let dm = DsDims::tiny();
        let pg = prefill_graph(&dm);
        let st = derived_step(&pg, &dm, 1).unwrap();
        let mut ids = st.outputs.clone();
        ids.extend(st.caches.iter().map(|c| c.updated));
        let prog = st.graph.to_cuda(&ids);
        eprintln!("ds step cuda: {} kernels", prog.kernels.len());
        for kn in &prog.kernels {
            let Op::Comp(c) = &st.graph.nodes()[kn.node.index()].op else {
                continue;
            };
            if !c.has_gather() {
                continue;
            }
            let loops: Vec<&str> = kn
                .source
                .lines()
                .filter(|l| l.trim_start().starts_with("for"))
                .map(str::trim)
                .collect();
            eprintln!(
                "gather kernel node {:?}: out {:?} red {:?} points {}; loops {loops:?}",
                kn.node, c.out, c.red, kn.points
            );
        }
    }
}
