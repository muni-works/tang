//! GLM-5.3 (GGUF arch `glm5-next`, tang-llm's `flash::glm5`) written once,
//! as a full-sequence (prefill) graph over a symbolic sequence dim `T`, to
//! see how far [`derive_step`] carries from Qwen3 to GLM.
//!
//! The math is the hand path's CPU reference, term for term
//! (`tang_llm::flash::glm5::{mhc, mla, kda, moe}` composed as
//! `forward::Glm5Ref::forward` composes them):
//!
//! - **mHC**: 4 residual streams `[T, hc, n]`; per block an unweighted flat
//!   RMSNorm, the 24 mixes, `pre`/`post`/`comb` (softmax + Sinkhorn, 1 column
//!   pass then `iters − 1` × (row, column)), collapse, and the post combine.
//! - **DSA** (MLA without rope): `q_a → RMSNorm → q_b`, the latent
//!   `kv_a → RMSNorm` (`[T, kv_lora]`), per-head `k_b`/`v_b` decompression,
//!   written either absorbed (scores against the latent, `W_v` after the
//!   weighted latent sum) or expanded (per-head K and V nodes); and the
//!   k-pool indexer: LayerNorm keys, gate-softmax pooled keys of complete
//!   pools of `kpool` tokens, relu head-weighted scores, top `topk/kpool`
//!   pools (ties to the lower index) plus the query's incomplete tail pool.
//!   With `sparse: false` the indexer is left out (dense causal attention,
//!   exact while `T ≤ topk`, the hand path's own fast path).
//! - **KDA** (Kimi delta attention): stacked q|k|v projections, the depthwise
//!   causal conv (`d_conv` taps, zero history) + SiLU, l2-normed q/k, the
//!   per-channel safe gate `lb·σ(rate·(f_b f_a x + dt))`, `β = σ(b x)`, the
//!   delta rule `S_t = (I − β k kᵀ)·diag(e^g)·S_{t−1} + β k vᵀ` as a matrix
//!   affine [`TensorGraph::linear_scan`], the readout, and the gated RMSNorm.
//! - **FFN**: dense clamped SwiGLU, or the MoE: sigmoid router, selection on
//!   score + bias (top-k by rank, ties to the lower id), weights the unbiased
//!   scores normalized × scale, every routed expert evaluated and weighted
//!   (zero weight off the top-k), plus the ungated shared expert.
//! - **Head**: unweighted stream mean, RMSNorm, lm head.
//!
//! Weights are plain f32 matrices (no GGUF quant formats; see the gap list
//! in the PR). Top-k selection (indexer pools, MoE experts) is a rank count:
//! `rank(x) = #{y : y > x or (y = x and id(y) < id(x))}`, `selected ⇔ rank
//! < k` — a masked-sum comprehension, no new IR. Modular position facts
//! (pool ends, the tail pool) use `floor`: [`idiv`].
use tang::Scalar;
use tang_compile::tensor::step::{derive_step, StepGraph, StepSpec};
use tang_compile::tensor::{Arg, Dim, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

/// The symbolic sequence dim.
pub const SEQ: &str = "T";
const T: Dim = Dim::Sym(SEQ);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    /// MLA-NoPE attention with the k-pool indexer.
    Dsa,
    /// Kimi delta attention.
    Kda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffn {
    Dense,
    Moe,
}

/// GLM-5.3 geometry (field names follow `flash::glm5::Hparams`).
#[derive(Clone, Debug)]
pub struct GlmDims {
    pub vocab: usize,
    pub n: usize,
    /// mHC streams (the hand mHC is fixed at 4).
    pub hc: usize,
    pub sinkhorn: usize,
    pub hc_eps: f64,
    pub eps: f64,
    pub layers: Vec<(Mixer, Ffn)>,
    // DSA
    pub heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub topk: usize,
    pub kpool: usize,
    /// Scores against the latent (`true`) or against per-head expanded K/V.
    pub absorbed: bool,
    /// The indexer's top-k pool selection (`false`: dense causal).
    pub sparse: bool,
    // KDA
    pub kda_heads: usize,
    pub kda_hd: usize,
    pub d_conv: usize,
    pub gate_lb: f64,
    // FFN
    pub ff_dense: usize,
    pub n_expert: usize,
    pub top_k: usize,
    pub ff_exp: usize,
    pub ff_sh: usize,
    pub route_scale: f64,
    /// SwiGLU clamp (routed, shared and dense alike, as in the real GGUF).
    pub limit: f64,
}

impl GlmDims {
    /// Three layers covering every block kind: DSA + dense FFN, KDA + MoE,
    /// DSA + MoE. `topk = 4`, `kpool = 2`: from `T = 5` the indexer drops keys.
    pub fn tiny() -> Self {
        Self {
            vocab: 24,
            n: 16,
            hc: 4,
            sinkhorn: 20,
            hc_eps: 1e-6,
            eps: 1e-5,
            layers: vec![
                (Mixer::Dsa, Ffn::Dense),
                (Mixer::Kda, Ffn::Moe),
                (Mixer::Dsa, Ffn::Moe),
            ],
            heads: 2,
            dk: 8,
            dv: 8,
            q_lora: 8,
            kv_lora: 6,
            idx_heads: 2,
            idx_dim: 4,
            topk: 4,
            kpool: 2,
            absorbed: true,
            sparse: true,
            kda_heads: 2,
            kda_hd: 4,
            d_conv: 4,
            gate_lb: -5.0,
            ff_dense: 12,
            n_expert: 4,
            top_k: 2,
            ff_exp: 6,
            ff_sh: 6,
            route_scale: 2.5,
            limit: 1.0,
        }
    }
}

/// One prefill input: name (`blk.{l}.…` as the GGUF names them, plus
/// `tokens`) and shape.
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
    /// Named intermediate nodes (latent, expanded K/V, indexer keys, KDA
    /// state…), to say which ones a derived step caches.
    pub named: Vec<(String, TensorId)>,
}

impl PrefillGraph {
    pub fn input(&self, name: &str) -> usize {
        self.inputs
            .iter()
            .position(|s| s.name == name)
            .unwrap_or_else(|| panic!("no input {name}"))
    }

    pub fn name_of(&self, t: TensorId) -> Option<&str> {
        self.named
            .iter()
            .find(|(_, n)| *n == t)
            .map(|(s, _)| s.as_str())
    }
}

fn u(rank: usize, d: usize) -> Form {
    Form::unit(rank, d)
}

fn fm(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::constant(rank, offset);
    for &(d, c) in terms {
        f.coefs[d] += c;
    }
    f
}

fn c(rank: usize, v: i64) -> Form {
    Form::constant(rank, v)
}

fn id(rank: usize) -> Vec<Form> {
    (0..rank).map(|d| u(rank, d)).collect()
}

fn a(src: TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View {
            gather: None,
            guard: false,
            forms,
        },
    }
}

/// A read that may fall outside `src` (reads 0 there).
fn ag(src: TensorId, forms: Vec<Form>) -> Arg {
    let mut x = a(src, forms);
    x.view.guard = true;
    x
}

fn select(cond: ExprId, t: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(cond, t, f))
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

/// `x` where `a == b` (integers as floats), else 0.
fn pick(a: ExprId, b: ExprId, x: ExprId) -> ExprId {
    let z = lit(0.0);
    select(a - b + lit(0.5), select(b - a + lit(0.5), x, z), z)
}

/// `⌊x / k⌋` for integer-valued `x` and a positive integer `k`; the half
/// offset keeps the quotient off integers, so rounding in `1/k` (or f32)
/// cannot flip it.
fn idiv(x: ExprId, k: f64) -> ExprId {
    ((x + lit(0.5)) * lit(1.0 / k)).floor()
}

/// 1 when `(y, iy)` ranks above `(x, ix)`: `y > x`, or a tie and `iy < ix`.
fn above(y: ExprId, x: ExprId, iy: ExprId, ix: ExprId) -> ExprId {
    select(
        y - x,
        lit(1.0),
        select(x - y, lit(0.0), select(ix - iy, lit(1.0), lit(0.0))),
    )
}

fn fx(n: usize) -> Dim {
    Dim::Fixed(n)
}

struct B {
    g: TensorGraph,
    inputs: Vec<PrefillInput>,
    named: Vec<(String, TensorId)>,
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
        let shape: Vec<Dim> = shape.iter().map(|&n| fx(n)).collect();
        self.input(name, &shape)
    }

    fn name(&mut self, s: String, t: TensorId) {
        self.named.push((s, t));
    }

    fn comp(
        &mut self,
        out: &[Dim],
        red: &[Dim],
        args: Vec<Arg>,
        m: Option<Monoid>,
        f: impl Fn(&[ExprId]) -> ExprId,
    ) -> TensorId {
        self.g
            .comp(out, red, args, m, |v| vec![f(v)])
            .unwrap_or_else(|e| panic!("comp {out:?} / {red:?}: {e:?}"))
    }

    fn sum(
        &mut self,
        out: &[Dim],
        red: &[Dim],
        args: Vec<Arg>,
        f: impl Fn(&[ExprId]) -> ExprId,
    ) -> TensorId {
        self.comp(out, red, args, Some(Monoid::sum()), f)
    }

    fn map(&mut self, out: &[Dim], args: Vec<Arg>, f: impl Fn(&[ExprId]) -> ExprId) -> TensorId {
        self.comp(out, &[], args, None, f)
    }

    /// `[T, out] = x [T, in] · Wᵀ`, `W [out, in]`.
    fn linear(&mut self, x: TensorId, w: TensorId) -> TensorId {
        let s = self.g.shape(w).to_vec();
        self.sum(
            &[T, s[0]],
            &[s[1]],
            vec![a(x, vec![u(3, 0), u(3, 2)]), a(w, vec![u(3, 1), u(3, 2)])],
            |v| v[0] * v[1],
        )
    }

    /// RMSNorm over the last dim of `[T, n]`, times `w` (or not).
    fn rms(&mut self, x: TensorId, w: Option<TensorId>, eps: f64) -> TensorId {
        let n = self.g.shape(x)[1].n();
        let ss = self.sum(&[T], &[fx(n)], vec![a(x, id(2))], |v| v[0] * v[0]);
        let mut args = vec![a(x, id(2)), a(ss, vec![u(2, 0)])];
        if let Some(w) = w {
            args.push(a(w, vec![u(2, 1)]));
        }
        let has_w = w.is_some();
        self.map(&[T, fx(n)], args, move |v| {
            let inv = lit(1.0) / (v[1] / lit(n as f64) + lit(eps)).sqrt();
            if has_w {
                v[0] * inv * v[2]
            } else {
                v[0] * inv
            }
        })
    }

    fn add(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.g.map(&[x, y], |v| v[0] + v[1]).unwrap()
    }

    /// Clamped SwiGLU MLP: `down · (silu(min(g, L)) · clamp(u, ±L))`.
    fn mlp(
        &mut self,
        x: TensorId,
        gate: TensorId,
        up: TensorId,
        down: TensorId,
        lim: f64,
    ) -> TensorId {
        let g = self.linear(x, gate);
        let up = self.linear(x, up);
        let ff = self.g.shape(gate)[0];
        let h = self.map(&[T, ff], vec![a(g, id(2)), a(up, id(2))], move |v| {
            silu(v[0].min(lit(lim))) * v[1].max(lit(-lim)).min(lit(lim))
        });
        self.linear(h, down)
    }
}

/// mHC pre: the block input `[T, n]`, `post [T, hc]`, `comb [T, src, dst]`.
fn mhc_pre(b: &mut B, d: &GlmDims, r: TensorId, p: &str) -> (TensorId, TensorId, TensorId) {
    let (hc, n) = (d.hc, d.n);
    let mix = (2 + hc) * hc;
    let f = b.w(format!("{p}_fn"), &[mix, hc * n]);
    let base = b.w(format!("{p}_base"), &[mix]);
    let scale = b.w(format!("{p}_scale"), &[3]);
    let (rms_eps, eps) = (d.eps, d.hc_eps);
    let flat = (hc * n) as f64;
    let ss = b.sum(&[T], &[fx(hc), fx(n)], vec![a(r, id(3))], |v| v[0] * v[0]);
    let mixes = b.sum(
        &[T, fx(mix)],
        &[fx(hc), fx(n)],
        vec![
            a(f, vec![u(4, 1), fm(4, &[(2, n as i64), (3, 1)], 0)]),
            a(r, vec![u(4, 0), u(4, 2), u(4, 3)]),
            a(ss, vec![u(4, 0)]),
        ],
        move |v| v[0] * v[1] / (v[2] / lit(flat) + lit(rms_eps)).sqrt(),
    );
    let pre = b.map(
        &[T, fx(hc)],
        vec![
            a(mixes, id(2)),
            a(scale, vec![c(2, 0)]),
            a(base, vec![u(2, 1)]),
        ],
        move |v| sigmoid(v[0] * v[1] + v[2]) + lit(eps),
    );
    let h = hc as i64;
    let post = b.map(
        &[T, fx(hc)],
        vec![
            a(mixes, vec![u(2, 0), fm(2, &[(1, 1)], h)]),
            a(scale, vec![c(2, 1)]),
            a(base, vec![fm(2, &[(1, 1)], h)]),
        ],
        |v| lit(2.0) * sigmoid(v[0] * v[1] + v[2]),
    );
    let sd = [T, fx(hc), fx(hc)];
    let ez = b.map(
        &sd,
        vec![
            a(mixes, vec![u(3, 0), fm(3, &[(1, h), (2, 1)], 2 * h)]),
            a(scale, vec![c(3, 2)]),
            a(base, vec![fm(3, &[(1, h), (2, 1)], 2 * h)]),
        ],
        |v| (v[0] * v[1] + v[2]).exp(),
    );
    let z = b.sum(&[T, fx(hc)], &[fx(hc)], vec![a(ez, id(3))], |v| v[0]);
    let mut comb = b.map(
        &sd,
        vec![a(ez, id(3)), a(z, vec![u(3, 0), u(3, 1)])],
        move |v| v[0] / v[1] + lit(eps),
    );
    // Sinkhorn: column (over src), then (row, column) × (iters − 1)
    let col = |b: &mut B, cm: TensorId| {
        let zc = b.sum(
            &[T, fx(hc)],
            &[fx(hc)],
            vec![a(cm, vec![u(3, 0), u(3, 2), u(3, 1)])],
            |v| v[0],
        );
        b.map(
            &sd,
            vec![a(cm, id(3)), a(zc, vec![u(3, 0), u(3, 2)])],
            move |v| v[0] / (v[1] + lit(eps)),
        )
    };
    let row = |b: &mut B, cm: TensorId| {
        let zr = b.sum(&[T, fx(hc)], &[fx(hc)], vec![a(cm, id(3))], |v| v[0]);
        b.map(
            &sd,
            vec![a(cm, id(3)), a(zr, vec![u(3, 0), u(3, 1)])],
            move |v| v[0] / (v[1] + lit(eps)),
        )
    };
    comb = col(b, comb);
    for _ in 1..d.sinkhorn {
        comb = row(b, comb);
        comb = col(b, comb);
    }
    let x = b.sum(
        &[T, fx(n)],
        &[fx(hc)],
        vec![
            a(pre, vec![u(3, 0), u(3, 2)]),
            a(r, vec![u(3, 0), u(3, 2), u(3, 1)]),
        ],
        |v| v[0] * v[1],
    );
    (x, post, comb)
}

/// mHC post: `r'[t, d] = post[t, d]·y[t] + Σ_s comb[t, s, d]·r[t, s]`.
fn mhc_post(
    b: &mut B,
    d: &GlmDims,
    r: TensorId,
    y: TensorId,
    post: TensorId,
    comb: TensorId,
) -> TensorId {
    let (hc, n) = (d.hc, d.n);
    let mixed = b.sum(
        &[T, fx(hc), fx(n)],
        &[fx(hc)],
        vec![
            a(comb, vec![u(4, 0), u(4, 3), u(4, 1)]),
            a(r, vec![u(4, 0), u(4, 3), u(4, 2)]),
        ],
        |v| v[0] * v[1],
    );
    b.map(
        &[T, fx(hc), fx(n)],
        vec![
            a(mixed, id(3)),
            a(post, vec![u(3, 0), u(3, 1)]),
            a(y, vec![u(3, 0), u(3, 2)]),
        ],
        |v| v[0] + v[1] * v[2],
    )
}

/// The indexer's key mask `[T(query), T(key)]`, 1 where selected.
fn indexer_mask(b: &mut B, d: &GlmDims, p: &str, x: TensorId, qr: TensorId) -> TensorId {
    let (n, ql, ih, idd, kp) = (d.n, d.q_lora, d.idx_heads, d.idx_dim, d.kpool);
    let wk = b.w(format!("{p}.indexer.attn_k"), &[idd, n]);
    let knw = b.w(format!("{p}.indexer.k_norm.weight"), &[idd]);
    let knb = b.w(format!("{p}.indexer.k_norm.bias"), &[idd]);
    let wqb = b.w(format!("{p}.indexer.attn_q_b"), &[ih * idd, ql]);
    let proj = b.w(format!("{p}.indexer.proj"), &[ih, n]);
    let gate_w = b.w(format!("{p}.indexer_compressor_gate"), &[idd, n]);
    let ape = b.w(format!("{p}.indexer_compressor_ape"), &[kp, idd]);
    // keys: LayerNorm(wk·x) (eps 1e-6, weight and bias)
    let kr = b.linear(x, wk);
    let dn = idd as f64;
    let mean = b.sum(&[T], &[fx(idd)], vec![a(kr, id(2))], move |v| {
        v[0] / lit(dn)
    });
    let var = b.sum(
        &[T],
        &[fx(idd)],
        vec![a(kr, id(2)), a(mean, vec![u(2, 0)])],
        move |v| (v[0] - v[1]) * (v[0] - v[1]) / lit(dn),
    );
    let ik = b.map(
        &[T, fx(idd)],
        vec![
            a(kr, id(2)),
            a(mean, vec![u(2, 0)]),
            a(var, vec![u(2, 0)]),
            a(knw, vec![u(2, 1)]),
            a(knb, vec![u(2, 1)]),
        ],
        |v| (v[0] - v[1]) / (v[2] + lit(1e-6)).sqrt() * v[3] + v[4],
    );
    let gate = b.linear(x, gate_w);
    b.name(format!("{p}.indexer.k"), ik);
    b.name(format!("{p}.indexer.gate"), gate);
    // the pool ending at token e: slots e − kp + 1 + j, gate-softmax over j
    let back = 1 - kp as i64;
    let lg = b.map(
        &[T, fx(kp), fx(idd)],
        vec![
            ag(gate, vec![fm(3, &[(0, 1), (1, 1)], back), u(3, 2)]),
            a(ape, vec![u(3, 1), u(3, 2)]),
        ],
        |v| v[0] + v[1],
    );
    let gmax = b.comp(
        &[T, fx(idd)],
        &[fx(kp)],
        vec![a(lg, vec![u(3, 0), u(3, 2), u(3, 1)])],
        Some(Monoid::max()),
        |v| v[0],
    );
    let gz = b.sum(
        &[T, fx(idd)],
        &[fx(kp)],
        vec![
            a(lg, vec![u(3, 0), u(3, 2), u(3, 1)]),
            a(gmax, vec![u(3, 0), u(3, 1)]),
        ],
        |v| (v[0] - v[1]).exp(),
    );
    let pk = b.sum(
        &[T, fx(idd)],
        &[fx(kp)],
        vec![
            a(lg, vec![u(3, 0), u(3, 2), u(3, 1)]),
            a(gmax, vec![u(3, 0), u(3, 1)]),
            a(gz, vec![u(3, 0), u(3, 1)]),
            ag(ik, vec![fm(3, &[(0, 1), (2, 1)], back), u(3, 1)]),
        ],
        |v| (v[0] - v[1]).exp() / v[2] * v[3],
    );
    b.name(format!("{p}.indexer.pool_key"), pk);
    // pe[e] = 1 iff e + 1 is a multiple of kp
    let sp = [T, T];
    let ie = b.g.iota(&[T], 0);
    let kpf = kp as f64;
    let pe = b.map(&[T], vec![a(ie, id(1))], move |v| {
        let e1 = v[0] + lit(1.0);
        pick(e1, lit(kpf) * idiv(e1, kpf), lit(1.0))
    });
    // scores[i, e] = Σ_h w_h(x_i)/√ih · relu(iq_h(i)·pk(e)/√idd)
    let iq = b.linear(qr, wqb);
    let hw = b.linear(x, proj);
    let dots = b.sum(
        &[T, T, fx(ih)],
        &[fx(idd)],
        vec![
            a(iq, vec![u(4, 0), fm(4, &[(2, idd as i64), (3, 1)], 0)]),
            a(pk, vec![u(4, 1), u(4, 3)]),
        ],
        |v| v[0] * v[1],
    );
    let (hs, ss) = (1.0 / (ih as f64).sqrt(), 1.0 / (idd as f64).sqrt());
    let score = b.sum(
        &[T, T],
        &[fx(ih)],
        vec![a(hw, vec![u(3, 0), u(3, 2)]), a(dots, id(3))],
        move |v| v[0] * lit(hs) * (v[1] * lit(ss)).max(lit(0.0)),
    );
    // rank[i, e] among the pools visible to i (complete, end ≤ i)
    let sp3 = [T, T, T];
    let (i3, e3, f3) = (b.g.iota(&sp3, 0), b.g.iota(&sp3, 1), b.g.iota(&sp3, 2));
    let rank = b.sum(
        &[T, T],
        &[T],
        vec![
            a(score, vec![u(3, 0), u(3, 2)]),
            a(score, vec![u(3, 0), u(3, 1)]),
            a(pe, vec![u(3, 2)]),
            a(i3, id(3)),
            a(e3, id(3)),
            a(f3, id(3)),
        ],
        |v| {
            let cand = select(v[3] - v[5] + lit(0.5), v[2], lit(0.0));
            cand * above(v[0], v[1], v[5], v[4])
        },
    );
    let (i2, e2) = (b.g.iota(&sp, 0), b.g.iota(&sp, 1));
    let nsel = (d.topk / kp) as f64;
    let sel = b.map(
        &sp,
        vec![
            a(rank, id(2)),
            a(pe, vec![u(2, 1)]),
            a(i2, id(2)),
            a(e2, id(2)),
        ],
        move |v| {
            let cand = select(v[2] - v[3] + lit(0.5), v[1], lit(0.0));
            cand * select(lit(nsel) - v[0] - lit(0.5), lit(1.0), lit(0.0))
        },
    );
    // key j is in a selected pool: Σ_e sel[i, e]·[e − kp < j ≤ e]
    let (ij, jj, ej) = (b.g.iota(&sp3, 0), b.g.iota(&sp3, 1), b.g.iota(&sp3, 2));
    let _ = ij;
    let pooled = b.sum(
        &sp,
        &[T],
        vec![a(sel, vec![u(3, 0), u(3, 2)]), a(jj, id(3)), a(ej, id(3))],
        move |v| {
            select(
                v[2] - v[1] + lit(0.5),
                select(v[1] - v[2] + lit(kpf) - lit(0.5), v[0], lit(0.0)),
                lit(0.0),
            )
        },
    );
    // the tail: j ≤ i with no pool end in [j, i], i.e. j in i's pool
    // (⌊(i + 1)/kp⌋ = ⌊j/kp⌋)
    let mask = b.map(
        &sp,
        vec![a(pooled, id(2)), a(i2, id(2)), a(e2, id(2))],
        move |v| {
            let tail = select(
                v[1] - v[2] + lit(0.5),
                pick(idiv(v[1] + lit(1.0), kpf), idiv(v[2], kpf), lit(1.0)),
                lit(0.0),
            );
            v[0] + tail
        },
    );
    b.name(format!("{p}.indexer.mask"), mask);
    mask
}

/// The DSA mixer (MLA-NoPE + indexer) of block input `x` (post `attn_norm`).
fn dsa(b: &mut B, d: &GlmDims, l: usize, x: TensorId) -> TensorId {
    let p = format!("blk.{l}");
    let (n, h, dk, dv, ql, kl) = (d.n, d.heads, d.dk, d.dv, d.q_lora, d.kv_lora);
    let q_a = b.w(format!("{p}.attn_q_a"), &[ql, n]);
    let q_a_norm = b.w(format!("{p}.attn_q_a_norm"), &[ql]);
    let q_b = b.w(format!("{p}.attn_q_b"), &[h * dk, ql]);
    let kv_a = b.w(format!("{p}.attn_kv_a_mqa"), &[kl, n]);
    let kv_a_norm = b.w(format!("{p}.attn_kv_a_norm"), &[kl]);
    let k_b = b.w(format!("{p}.attn_k_b"), &[h, kl, dk]);
    let v_b = b.w(format!("{p}.attn_v_b"), &[h, dv, kl]);
    let o = b.w(format!("{p}.attn_output"), &[n, h * dv]);
    let qa = b.linear(x, q_a);
    let qr = b.rms(qa, Some(q_a_norm), d.eps);
    let la = b.linear(x, kv_a);
    let lat = b.rms(la, Some(kv_a_norm), d.eps);
    b.name(format!("{p}.latent"), lat);
    let q = b.linear(qr, q_b);
    let mask = d.sparse.then(|| indexer_mask(b, d, &p, x, qr));
    let scale = 1.0 / (dk as f64).sqrt();
    let (hh, dki) = (h, dk as i64);
    // scores [T(i), H, T(j)]
    let (sc, vx, absorbed_q) = if d.absorbed {
        // q'_h = W_k_hᵀ·q_h: [T, H, kl]
        let qa = b.sum(
            &[T, fx(hh), fx(kl)],
            &[fx(dk)],
            vec![
                a(q, vec![u(4, 0), fm(4, &[(1, dki), (3, 1)], 0)]),
                a(k_b, vec![u(4, 1), u(4, 2), u(4, 3)]),
            ],
            |v| v[0] * v[1],
        );
        let sc = b.sum(
            &[T, fx(hh), T],
            &[fx(kl)],
            vec![
                a(qa, vec![u(4, 0), u(4, 1), u(4, 3)]),
                a(lat, vec![u(4, 2), u(4, 3)]),
            ],
            move |v| v[0] * v[1] * lit(scale),
        );
        (sc, None, true)
    } else {
        let kx = b.sum(
            &[fx(hh), T, fx(dk)],
            &[fx(kl)],
            vec![
                a(lat, vec![u(4, 1), u(4, 3)]),
                a(k_b, vec![u(4, 0), u(4, 3), u(4, 2)]),
            ],
            |v| v[0] * v[1],
        );
        let vx = b.sum(
            &[fx(hh), fx(dv), T],
            &[fx(kl)],
            vec![
                a(v_b, vec![u(4, 0), u(4, 1), u(4, 3)]),
                a(lat, vec![u(4, 2), u(4, 3)]),
            ],
            |v| v[0] * v[1],
        );
        b.name(format!("{p}.k_expanded"), kx);
        b.name(format!("{p}.v_expanded"), vx);
        let sc = b.sum(
            &[T, fx(hh), T],
            &[fx(dk)],
            vec![
                a(q, vec![u(4, 0), fm(4, &[(1, dki), (3, 1)], 0)]),
                a(kx, vec![u(4, 1), u(4, 2), u(4, 3)]),
            ],
            move |v| v[0] * v[1] * lit(scale),
        );
        (sc, Some(vx), false)
    };
    // the visibility condition of key j for query i (> 0 visible)
    let vis_args = |b: &mut B, sp: &[Dim], qi: usize, kj: usize| -> Vec<Arg> {
        let r = sp.len();
        let (ti, tj) = (b.g.iota(sp, qi), b.g.iota(sp, kj));
        let mut v = vec![a(ti, id(r)), a(tj, id(r))];
        if let Some(m) = mask {
            v.push(a(m, vec![u(r, qi), u(r, kj)]));
        }
        v
    };
    let sparse = mask.is_some();
    let cond = move |v: &[ExprId], at: usize| {
        if sparse {
            v[at + 2] - lit(0.5)
        } else {
            v[at] - v[at + 1] + lit(0.5)
        }
    };
    // softmax statistics over j: space (i, h, j)
    let sp = [T, fx(hh), T];
    let mut args = vec![a(sc, id(3))];
    args.extend(vis_args(b, &sp, 0, 2));
    let m = b.comp(&sp[..2], &[T], args, Some(Monoid::max()), move |v| {
        select(cond(v, 1), v[0], lit(-1e30))
    });
    let mut args = vec![a(sc, id(3)), a(m, vec![u(3, 0), u(3, 1)])];
    args.extend(vis_args(b, &sp, 0, 2));
    let lsum = b.sum(&sp[..2], &[T], args, move |v| {
        select(cond(v, 2), (v[0] - v[1]).exp(), lit(0.0))
    });
    let attn = if absorbed_q {
        // weighted latent Σ_j p·lat[j]: space (i, h, l, j), then W_v
        let sp = [T, fx(hh), fx(kl), T];
        let ml = vec![u(4, 0), u(4, 1)];
        let mut args = vec![
            a(sc, vec![u(4, 0), u(4, 1), u(4, 3)]),
            a(m, ml.clone()),
            a(lsum, ml),
            a(lat, vec![u(4, 3), u(4, 2)]),
        ];
        args.extend(vis_args(b, &sp, 0, 3));
        let ol = b.sum(&sp[..3], &[T], args, move |v| {
            select(cond(v, 4), (v[0] - v[1]).exp() / v[2] * v[3], lit(0.0))
        });
        b.sum(
            &[T, fx(hh), fx(dv)],
            &[fx(kl)],
            vec![
                a(v_b, vec![u(4, 1), u(4, 2), u(4, 3)]),
                a(ol, vec![u(4, 0), u(4, 1), u(4, 3)]),
            ],
            |v| v[0] * v[1],
        )
    } else {
        let vx = vx.unwrap();
        let sp = [T, fx(hh), fx(dv), T];
        let ml = vec![u(4, 0), u(4, 1)];
        let mut args = vec![
            a(sc, vec![u(4, 0), u(4, 1), u(4, 3)]),
            a(m, ml.clone()),
            a(lsum, ml),
            a(vx, vec![u(4, 1), u(4, 2), u(4, 3)]),
        ];
        args.extend(vis_args(b, &sp, 0, 3));
        b.sum(&sp[..3], &[T], args, move |v| {
            select(cond(v, 4), (v[0] - v[1]).exp() / v[2] * v[3], lit(0.0))
        })
    };
    let attn = b.g.reshape(attn, &[T, fx(h * dv)]).unwrap();
    b.linear(attn, o)
}

/// The KDA mixer of block input `x` (post `attn_norm`).
fn kda(b: &mut B, d: &GlmDims, l: usize, x: TensorId) -> TensorId {
    let p = format!("blk.{l}");
    let (n, h, hd, kc) = (d.n, d.kda_heads, d.kda_hd, d.d_conv);
    let ch = 3 * h * hd;
    let wqkv = b.w(format!("{p}.attn_qkv"), &[ch, n]);
    let conv = b.w(format!("{p}.ssm_conv1d"), &[ch, kc]);
    let f_a = b.w(format!("{p}.ssm_f_a"), &[hd, n]);
    let f_b = b.w(format!("{p}.ssm_f_b"), &[h * hd, hd]);
    let dt = b.w(format!("{p}.ssm_dt"), &[h * hd]);
    let rate = b.w(format!("{p}.ssm_decay_rate"), &[h]);
    let bp = b.w(format!("{p}.ssm_beta"), &[h, n]);
    let g_a = b.w(format!("{p}.ssm_g_a"), &[hd, n]);
    let g_b = b.w(format!("{p}.ssm_g_b"), &[h * hd, hd]);
    let o_norm = b.w(format!("{p}.ssm_norm"), &[hd]);
    let o_proj = b.w(format!("{p}.attn_out"), &[n, h * hd]);
    let qkv = b.linear(x, wqkv);
    b.name(format!("{p}.conv_in"), qkv);
    // depthwise causal conv, zero history: Σ_i w[c, i]·qkv[t − (kc − 1) + i, c]
    let cv = b.sum(
        &[T, fx(ch)],
        &[fx(kc)],
        vec![
            ag(qkv, vec![fm(3, &[(0, 1), (2, 1)], 1 - kc as i64), u(3, 1)]),
            a(conv, vec![u(3, 1), u(3, 2)]),
        ],
        |v| v[0] * v[1],
    );
    let cv = b.map(&[T, fx(ch)], vec![a(cv, id(2))], |v| silu(v[0]));
    let hdi = hd as i64;
    // q, k: l2-normed per head (eps 1e-6), q / √hd
    let normed = |b: &mut B, off: i64, mul: f64| {
        let ss = b.sum(
            &[T, fx(h)],
            &[fx(hd)],
            vec![a(cv, vec![u(3, 0), fm(3, &[(1, hdi), (2, 1)], off)])],
            |v| v[0] * v[0],
        );
        b.map(
            &[T, fx(h), fx(hd)],
            vec![
                a(cv, vec![u(3, 0), fm(3, &[(1, hdi), (2, 1)], off)]),
                a(ss, vec![u(3, 0), u(3, 1)]),
            ],
            move |v| v[0] / (v[1] + lit(1e-6)).sqrt() * lit(mul),
        )
    };
    let q = normed(b, 0, 1.0 / (hd as f64).sqrt());
    let k = normed(b, (h * hd) as i64, 1.0);
    let v = b.map(
        &[T, fx(h), fx(hd)],
        vec![a(
            cv,
            vec![u(3, 0), fm(3, &[(1, hdi), (2, 1)], 2 * (h * hd) as i64)],
        )],
        |v| v[0],
    );
    // gates
    let fa = b.linear(x, f_a);
    let fb = b.linear(fa, f_b);
    let lb = d.gate_lb;
    let g = b.map(
        &[T, fx(h), fx(hd)],
        vec![
            a(fb, vec![u(3, 0), fm(3, &[(1, hdi), (2, 1)], 0)]),
            a(dt, vec![fm(3, &[(1, hdi), (2, 1)], 0)]),
            a(rate, vec![u(3, 1)]),
        ],
        move |v| lit(lb) * sigmoid(v[2] * (v[0] + v[1])),
    );
    let braw = b.linear(x, bp);
    let beta = b.map(&[T, fx(h)], vec![a(braw, id(2))], |v| sigmoid(v[0]));
    // S_t = (I − β k kᵀ)·diag(e^g)·S_{t−1} + β k vᵀ over [key, value]
    let sp = [T, fx(h), fx(hd), fx(hd)];
    let (ia, ic) = (b.g.iota(&sp, 2), b.g.iota(&sp, 3));
    let am = b.map(
        &sp,
        vec![
            a(beta, vec![u(4, 0), u(4, 1)]),
            a(k, vec![u(4, 0), u(4, 1), u(4, 2)]),
            a(k, vec![u(4, 0), u(4, 1), u(4, 3)]),
            a(g, vec![u(4, 0), u(4, 1), u(4, 3)]),
            a(ia, id(4)),
            a(ic, id(4)),
        ],
        |v| (pick(v[4], v[5], lit(1.0)) - v[0] * v[1] * v[2]) * v[3].exp(),
    );
    let um = b.map(
        &sp,
        vec![
            a(beta, vec![u(4, 0), u(4, 1)]),
            a(k, vec![u(4, 0), u(4, 1), u(4, 2)]),
            a(v, vec![u(4, 0), u(4, 1), u(4, 3)]),
        ],
        |v| v[0] * v[1] * v[2],
    );
    let s = b.g.linear_scan(am, um, 0, false, false).unwrap();
    b.name(format!("{p}.kda_state"), s);
    let o = b.g.scan_readout(s, q, 1.0).unwrap();
    // gated RMSNorm per head: rms(o)·o_norm·σ(g_b g_a x)
    let os = b.sum(&[T, fx(h)], &[fx(hd)], vec![a(o, id(3))], |v| v[0] * v[0]);
    let ga = b.linear(x, g_a);
    let g2 = b.linear(ga, g_b);
    let (eps, hdf) = (d.eps, hd as f64);
    let y = b.map(
        &[T, fx(h), fx(hd)],
        vec![
            a(o, id(3)),
            a(os, vec![u(3, 0), u(3, 1)]),
            a(o_norm, vec![u(3, 2)]),
            a(g2, vec![u(3, 0), fm(3, &[(1, hdi), (2, 1)], 0)]),
        ],
        move |v| v[0] / (v[1] / lit(hdf) + lit(eps)).sqrt() * v[2] * sigmoid(v[3]),
    );
    let y = b.g.reshape(y, &[T, fx(h * hd)]).unwrap();
    b.linear(y, o_proj)
}

/// The MoE FFN: sigmoid router with selection bias, top-k by rank, every
/// expert weighted (zero off the top-k), plus the shared expert.
fn moe(b: &mut B, d: &GlmDims, l: usize, x: TensorId) -> TensorId {
    let p = format!("blk.{l}");
    let (n, e, ff, ffs) = (d.n, d.n_expert, d.ff_exp, d.ff_sh);
    let router = b.w(format!("{p}.ffn_gate_inp"), &[e, n]);
    let bias = b.w(format!("{p}.exp_probs_b"), &[e]);
    let gate = b.w(format!("{p}.ffn_gate_exps"), &[e, ff, n]);
    let up = b.w(format!("{p}.ffn_up_exps"), &[e, ff, n]);
    let down = b.w(format!("{p}.ffn_down_exps"), &[e, n, ff]);
    let sg = b.w(format!("{p}.ffn_gate_shexp"), &[ffs, n]);
    let su = b.w(format!("{p}.ffn_up_shexp"), &[ffs, n]);
    let sd = b.w(format!("{p}.ffn_down_shexp"), &[n, ffs]);
    let logits = b.linear(x, router);
    let s = b.map(&[T, fx(e)], vec![a(logits, id(2))], |v| sigmoid(v[0]));
    let sp = [T, fx(e), fx(e)];
    let (ie, ie2) = (b.g.iota(&sp, 1), b.g.iota(&sp, 2));
    let rank = b.sum(
        &[T, fx(e)],
        &[fx(e)],
        vec![
            a(s, vec![u(3, 0), u(3, 2)]),
            a(bias, vec![u(3, 2)]),
            a(s, vec![u(3, 0), u(3, 1)]),
            a(bias, vec![u(3, 1)]),
            a(ie2, id(3)),
            a(ie, id(3)),
        ],
        |v| above(v[0] + v[1], v[2] + v[3], v[4], v[5]),
    );
    let k = d.top_k as f64;
    let ws = b.map(&[T, fx(e)], vec![a(s, id(2)), a(rank, id(2))], move |v| {
        select(lit(k) - v[1] - lit(0.5), v[0], lit(0.0))
    });
    let wsum = b.sum(&[T], &[fx(e)], vec![a(ws, id(2))], |v| v[0]);
    let scale = d.route_scale;
    let wt = b.map(
        &[T, fx(e)],
        vec![a(ws, id(2)), a(wsum, vec![u(2, 0)])],
        move |v| v[0] / (v[1] + lit(1e-20)) * lit(scale),
    );
    let proj = |b: &mut B, w: TensorId| {
        b.sum(
            &[T, fx(e), fx(ff)],
            &[fx(n)],
            vec![
                a(w, vec![u(4, 1), u(4, 2), u(4, 3)]),
                a(x, vec![u(4, 0), u(4, 3)]),
            ],
            |v| v[0] * v[1],
        )
    };
    let eg = proj(b, gate);
    let eu = proj(b, up);
    let lim = d.limit;
    let hidden = b.map(
        &[T, fx(e), fx(ff)],
        vec![a(eg, id(3)), a(eu, id(3))],
        move |v| silu(v[0].min(lit(lim))) * v[1].max(lit(-lim)).min(lit(lim)),
    );
    let routed = b.sum(
        &[T, fx(n)],
        &[fx(e), fx(ff)],
        vec![
            a(wt, vec![u(4, 0), u(4, 2)]),
            a(down, vec![u(4, 2), u(4, 1), u(4, 3)]),
            a(hidden, vec![u(4, 0), u(4, 2), u(4, 3)]),
        ],
        |v| v[0] * v[1] * v[2],
    );
    let shared = b.mlp(x, sg, su, sd, lim);
    b.add(routed, shared)
}

/// The full-sequence graph of GLM-5.3 at `d`.
pub fn prefill_graph(d: &GlmDims) -> PrefillGraph {
    let (n, hc, vocab) = (d.n, d.hc, d.vocab);
    let mut b = B {
        g: TensorGraph::new(),
        inputs: Vec::new(),
        named: Vec::new(),
    };
    let tokens = b.input("tokens".into(), &[T]);
    let embd = b.w("token_embd".into(), &[vocab, n]);
    let vi = b.g.iota(&[T, fx(n), fx(vocab)], 2);
    let x = b.sum(
        &[T, fx(n)],
        &[fx(vocab)],
        vec![
            a(embd, vec![u(3, 2), u(3, 1)]),
            a(tokens, vec![u(3, 0)]),
            a(vi, id(3)),
        ],
        |v| pick(v[2], v[1], v[0]),
    );
    // every stream starts as the embedding
    let mut r = b.map(
        &[T, fx(hc), fx(n)],
        vec![a(x, vec![u(3, 0), u(3, 2)])],
        |v| v[0],
    );
    for (l, &(mixer, ffn)) in d.layers.iter().enumerate() {
        let p = format!("blk.{l}");
        let (xi, post, comb) = mhc_pre(&mut b, d, r, &format!("{p}.hc_attn"));
        let an = b.w(format!("{p}.attn_norm"), &[n]);
        let xn = b.rms(xi, Some(an), d.eps);
        let y = match mixer {
            Mixer::Dsa => dsa(&mut b, d, l, xn),
            Mixer::Kda => kda(&mut b, d, l, xn),
        };
        r = mhc_post(&mut b, d, r, y, post, comb);
        let (xi, post, comb) = mhc_pre(&mut b, d, r, &format!("{p}.hc_ffn"));
        let fnw = b.w(format!("{p}.ffn_norm"), &[n]);
        let xn = b.rms(xi, Some(fnw), d.eps);
        let y = match ffn {
            Ffn::Dense => {
                let g = b.w(format!("{p}.ffn_gate"), &[d.ff_dense, n]);
                let up = b.w(format!("{p}.ffn_up"), &[d.ff_dense, n]);
                let dn = b.w(format!("{p}.ffn_down"), &[n, d.ff_dense]);
                b.mlp(xn, g, up, dn, d.limit)
            }
            Ffn::Moe => moe(&mut b, d, l, xn),
        };
        r = mhc_post(&mut b, d, r, y, post, comb);
        b.name(format!("{p}.residual"), r);
    }
    let hf = hc as f64;
    let h = b.sum(
        &[T, fx(n)],
        &[fx(hc)],
        vec![a(r, vec![u(3, 0), u(3, 2), u(3, 1)])],
        move |v| v[0] / lit(hf),
    );
    let on = b.w("output_norm".into(), &[n]);
    let hn = b.rms(h, Some(on), d.eps);
    let lm = b.w("output".into(), &[vocab, n]);
    let logits = b.linear(hn, lm);
    PrefillGraph {
        g: b.g,
        inputs: b.inputs,
        logits,
        named: b.named,
    }
}

/// [`derive_step`] on `pg`'s logits, appending `k` tokens to caches of
/// `capacity` positions.
pub fn try_step(pg: &PrefillGraph, k: usize, capacity: usize) -> Result<StepGraph, String> {
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

#[cfg(test)]
#[path = "ir_glm_prefill_tests.rs"]
mod tests;
