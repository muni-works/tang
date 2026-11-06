//! Qwen3 as one tensor-IR graph per step: decode (one token) and
//! prefill (`seq` tokens continuing the cache from any position).
//!
//! A step runs every layer against KV caches of fixed capacity `ctx` and
//! ends with the greedy next token. Everything a decode step needs is on
//! the device: a `state` input `(token, position)` holds the token to
//! feed and where it goes, the step looks up its embedding and rope
//! angles there, and writes `(argmax, position + 1)` back over `state`.
//! The host loop reads one token id per step and writes nothing. A
//! prefill step reads its tokens and start position `p0` from a `chunk`
//! input, appends their K/V rows to the caches, attends causally over
//! the cache's `p0` older rows and its own, and writes `state` likewise.
//!
//! Weights are MLX 4-bit: per projection a `[rows, cols/64, 64]` tensor
//! of nibbles (logical values 0..=15) and bf16 `[rows, cols/64]` scales
//! and biases. The dequantization `q·s + b` is a map the matmul reads: in
//! decode it is inlined and evaluated where the packed words are loaded
//! (`to_cuda_decode` / `to_msl_decode`, `Packing::U4`); in prefill, where
//! each weight is read by `seq` tokens, it stays materialized and the
//! matmul is a tensor-core (simdgroup-matrix) tile over it — except on
//! OpenCL ([`QwenDims::prefill_q4`]), where the prefill's register-block
//! kernel dequantizes at load as decode does and no f32 copy is written.
//!
//! **Data-dependent reads are bounded sums.** The IR's views are affine,
//! so a read at a runtime index — the embedding row of a token, the rope
//! angles of a position, the chunk token that cache row `j` receives — is
//! written as a sum with one live term: `Σ_v select(v == tok, E[v], 0)`,
//! the equality as two half-planes
//! `select(tok − v + ½, select(v − tok + ½, …, 0), 0)`. The value is
//! exact (one term plus zeros), the
//! interpreter runs it as written, and the decode lowering turns the two
//! predicates into loop bounds, so the loop has one iteration.
//!
//! Attention masks key positions past each query's position with
//! `select(pos − j + ½, …, identity)` in every reduction over the cache
//! axis, which the lowering also turns into loop bounds: a step reads
//! `len` rows of cache, not `ctx`. The cache updates are maps
//! `select(len − seq ≤ j < len, new, old)` that overwrite the cache in
//! place.
use std::collections::HashMap;
use tang::Scalar;
use tang_compile::tensor::cuda_gemv::{DecodeOpts, Packing};

use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

/// Quantization group: one scale and bias per 64 weights of a row.
pub const GROUP: usize = 64;

#[derive(Clone, Debug)]
pub struct QwenDims {
    pub d_model: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub vocab: usize,
    pub layers: usize,
    /// KV cache capacity (positions).
    pub ctx: usize,
    pub eps: f64,
    pub theta: f64,
    /// KV caches stored as f16 (`Packing::F16`) instead of f32.
    pub kv_f16: bool,
    /// The embedding and LM head as this many row blocks of the
    /// vocabulary, each its own input: a GPU's largest storage binding can
    /// be smaller than one 4-bit `[vocab, d]` matrix (311 MB for
    /// Qwen3-8B; Vulkan on the Adreno binds at most 128 MiB). 1 is whole.
    pub vocab_chunks: usize,
    /// Prefill matmuls read the 4-bit weights as decode does (the
    /// dequantization inlined where the packed words are loaded) instead of
    /// over a materialized f32 copy per pass: the OpenCL prefill's
    /// register-block kernel over packed weights
    /// (`opencl_mma_kernel_q4`). Off: the f32 maps (every other backend).
    pub prefill_q4: bool,
}

impl QwenDims {
    /// Qwen3-8B (`config.json`), with a cache of `ctx` positions.
    pub fn qwen3_8b(ctx: usize) -> Self {
        Self {
            d_model: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            ff: 12288,
            vocab: 151936,
            layers: 36,
            ctx,
            eps: 1e-6,
            theta: 1e6,
            kv_f16: true,
            vocab_chunks: 1,
            prefill_q4: false,
        }
    }

    /// Rows of the fused q|k|v projection.
    pub fn qkv_rows(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }
}

/// One graph input: name (MLX tensor name stem), logical shape, packing.
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub packing: Option<Packing>,
}

/// Inputs per layer, in order.
pub const LAYER_INPUTS: [&str; 21] = [
    "qkv.q", "qkv.s", "qkv.b", "o.q", "o.s", "o.b", "gate.q", "gate.s", "gate.b", "up.q", "up.s",
    "up.b", "down.q", "down.s", "down.b", "ln1", "ln2", "q_norm", "k_norm", "k_cache", "v_cache",
];

pub struct DecodeGraph {
    pub g: TensorGraph,
    /// Tokens per step.
    pub seq: usize,
    /// Cache rows attention reads (`ctx` for decode; a prefill reads only
    /// the first `kctx`, which must cover its positions).
    pub kctx: usize,
    /// `[logits, state', k_cache'₀, v_cache'₀, k_cache'₁, …]`: the last
    /// token's logits, then the in-place outputs.
    pub outputs: Vec<TensorId>,
    /// The residual stream after each attention and MLP block, kept
    /// materialized: a residual read once per element would otherwise be
    /// substituted into its readers, and the stream would become a sum
    /// over every earlier layer.
    pub keep: Vec<TensorId>,
    /// Logits of vocabulary blocks 1.. when `vocab_chunks > 1` (`outputs[0]`
    /// holds block 0): the full logits are their concatenation.
    pub logit_chunks: Vec<TensorId>,
    pub inputs: Vec<InputSpec>,
    /// A prefill over the packed weights ([`QwenDims::prefill_q4`]).
    pub q4: bool,
}

impl DecodeGraph {
    /// Index of the graph input called `name`.
    pub fn input(&self, name: &str) -> usize {
        self.inputs
            .iter()
            .position(|s| s.name == name)
            .unwrap_or_else(|| panic!("no input {name}"))
    }

    /// `(output, input)` pairs written in place: `state` and the caches.
    pub fn inplace(&self) -> Vec<(TensorId, usize)> {
        let mut v = vec![(self.outputs[1], self.input("state"))];
        for l in 0..(self.outputs.len() - 2) / 2 {
            v.push((
                self.outputs[2 + 2 * l],
                self.input(&format!("model.layers.{l}.k_cache")),
            ));
            v.push((
                self.outputs[3 + 2 * l],
                self.input(&format!("model.layers.{l}.v_cache")),
            ));
        }
        v
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

/// `x` where `a == b` (integers as floats), else 0: the one live term of
/// a gather written as a sum.
fn pick(a: ExprId, b: ExprId, x: ExprId) -> ExprId {
    let z = lit(0.0);
    select(a - b + lit(0.5), select(b - a + lit(0.5), x, z), z)
}

/// Rows of a per-token tensor: a `[seq, n]` comp, or the flat `step`
/// input at `off` with row stride `stride`.
#[derive(Clone, Copy)]
struct Rows {
    src: TensorId,
    off: i64,
    stride: i64,
}

struct Builder {
    g: TensorGraph,
    inputs: Vec<InputSpec>,
    /// [`QwenDims::prefill_q4`]
    q4: bool,
}

impl Builder {
    fn input(&mut self, name: String, shape: &[usize], packing: Option<Packing>) -> TensorId {
        self.inputs.push(InputSpec {
            name,
            shape: shape.to_vec(),
            packing,
        });
        self.g.input(shape)
    }

    /// Read row `t` element `k` of `x` (forms over the comp's space).
    fn row(&self, x: Rows, t: Form, k: Form) -> Arg {
        if self.g.shape(x.src).len() == 1 {
            let mut f = k;
            f.offset += x.off;
            for (c, tc) in f.coefs.iter_mut().zip(&t.coefs) {
                *c += x.stride * tc;
            }
            f.offset += x.stride * t.offset;
            arg(x.src, vec![f])
        } else {
            let mut k = k;
            k.offset += x.off;
            arg(x.src, vec![t, k])
        }
    }

    /// A 4-bit projection `[rows, cols]`: nibbles, scales, biases.
    /// [`Self::q4`] as `chunks` row blocks, named `{name}.{c}` when split.
    fn q4_chunks(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        chunks: usize,
    ) -> Vec<[TensorId; 3]> {
        if chunks == 1 {
            return vec![self.q4(name, rows, cols)];
        }
        (0..chunks)
            .map(|c| self.q4(&format!("{name}.{c}"), rows / chunks, cols))
            .collect()
    }

    fn q4(&mut self, name: &str, rows: usize, cols: usize) -> [TensorId; 3] {
        let groups = cols / GROUP;
        [
            self.input(
                format!("{name}.q"),
                &[rows, groups, GROUP],
                Some(Packing::U4),
            ),
            self.input(format!("{name}.s"), &[rows, groups], Some(Packing::Bf16)),
            self.input(format!("{name}.b"), &[rows, groups], Some(Packing::Bf16)),
        ]
    }

    /// `y[t, n] = Σ_k x[t, k]·(q[n,k]·s[n,k/64] + b[n,k/64])` over `seq`
    /// rows of `x` starting at row `t0`.
    fn linear(&mut self, x: Rows, seq: usize, t0: i64, w: [TensorId; 3]) -> TensorId {
        let shape = self.g.shape(w[0]).to_vec();
        let (rows, groups) = (shape[0].n(), shape[1].n());
        // dequantization: a map over the nibbles the matmul reads
        let r3 = 3;
        let dq = self
            .g
            .comp(
                &[rows, groups, GROUP],
                &[],
                vec![
                    arg(w[0], ident(3)),
                    arg(w[1], vec![unit(r3, 0), unit(r3, 1)]),
                    arg(w[2], vec![unit(r3, 0), unit(r3, 1)]),
                ],
                None,
                |v| vec![v[0] * v[1] + v[2]],
            )
            .unwrap();
        if seq > 1 && !self.q4 {
            // many rows: the weight is read by every one, so the
            // dequantized weight is materialized (seq-fold recompute
            // otherwise) and the matmul runs over one flat k — a
            // tensor-core tile
            let k = groups * GROUP;
            let dq = self.g.reshape(dq, &[rows, k]).unwrap();
            let xa = self.row(x, form(3, &[(0, 1)], t0), unit(3, 2));
            return self
                .g
                .comp(
                    &[seq, rows],
                    &[k],
                    vec![xa, arg(dq, vec![unit(3, 1), unit(3, 2)])],
                    Some(Monoid::sum()),
                    |v| vec![v[0] * v[1]],
                )
                .unwrap();
        }
        // space (t, n, g, j)
        let r4 = 4;
        let xa = self.row(
            x,
            form(r4, &[(0, 1)], t0),
            form(r4, &[(2, GROUP as i64), (3, 1)], 0),
        );
        self.g
            .comp(
                &[seq, rows],
                &[groups, GROUP],
                vec![xa, arg(dq, vec![unit(r4, 1), unit(r4, 2), unit(r4, 3)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    /// RMSNorm of `seq` rows of `x` (from row `t0`, `n` wide) with weight `w`.
    fn rms_norm(
        &mut self,
        x: Rows,
        seq: usize,
        t0: i64,
        n: usize,
        w: TensorId,
        eps: f64,
    ) -> TensorId {
        let xa = self.row(x, form(2, &[(0, 1)], t0), unit(2, 1));
        let ss = self
            .g
            .comp(&[seq], &[n], vec![xa], Some(Monoid::sum()), |v| {
                vec![v[0] * v[0]]
            })
            .unwrap();
        let xa = self.row(x, form(2, &[(0, 1)], t0), unit(2, 1));
        self.g
            .comp(
                &[seq, n],
                &[],
                vec![xa, arg(ss, vec![unit(2, 0)]), arg(w, vec![unit(2, 1)])],
                None,
                |v| {
                    let inv = ExprId::from_f64(1.0)
                        / (v[1] / ExprId::from_f64(n as f64) + ExprId::from_f64(eps)).sqrt();
                    vec![v[0] * inv * v[2]]
                },
            )
            .unwrap()
    }

    /// `a[t, i] + b[t, i]` over `[seq, n]`.
    fn add(&mut self, a: Rows, b: TensorId, seq: usize, n: usize) -> TensorId {
        let aa = self.row(a, unit(2, 0), unit(2, 1));
        self.g
            .comp(&[seq, n], &[], vec![aa, arg(b, ident(2))], None, |v| {
                vec![v[0] + v[1]]
            })
            .unwrap()
    }
}

/// The decode step (one token, at the position `state` holds).
pub fn decode_graph(dims: &QwenDims) -> DecodeGraph {
    step_graph(dims, 1, dims.ctx)
}

/// A step over `seq` tokens. `seq == 1`: decode — token and position
/// from `state`. `seq > 1`: prefill — tokens and start position `p0`
/// from `chunk` (`[tok₀ … tok_{seq−1}, p0]`), positions `p0..p0 + seq`;
/// attention reads cache rows `0..kctx`, so `p0 + seq ≤ kctx ≤ ctx`.
pub fn step_graph(dims: &QwenDims, seq: usize, kctx: usize) -> DecodeGraph {
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
        vocab_chunks >= 1 && vocab % vocab_chunks == 0,
        "vocab_chunks must divide the vocabulary"
    );
    let vc = vocab / vocab_chunks;
    assert!(
        seq <= kctx && kctx <= ctx,
        "a step's positions must fit the window"
    );
    let half = hd / 2;
    let per = heads / kvh;
    let groups = d / GROUP;
    let mut b = Builder {
        g: TensorGraph::new(),
        inputs: Vec::new(),
        q4: prefill_q4,
    };
    let state = b.input("state".into(), &[2], None);
    let chunk = (seq > 1).then(|| b.input("chunk".into(), &[seq + 1], None));
    // `len`: cache rows filled after the step; token t sits at
    // position len − seq + t. Decode: state's position + 1; prefill:
    // p0 + seq.
    let (len_src, len_off, len_add) = match chunk {
        Some(c) => (c, seq as i64, seq as f64),
        None => (state, 1, 1.0),
    };
    let len_at = |rank: usize| arg(len_src, vec![Form::constant(rank, len_off)]);
    let lenv = |v: ExprId| v + lit(len_add);
    let seq_f = seq as f64;
    // token t's id (`t`: its row form)
    let tok_at = |rank: usize, t: Form| match chunk {
        Some(c) => arg(c, vec![t]),
        None => arg(state, vec![Form::constant(rank, 0)]),
    };

    // x[t, g, j] = Σ_v pick(v, tok_t, E[v, g, j]): the embedding row
    // (summed over vocabulary blocks when the table is split)
    let mut parts = Vec::with_capacity(vocab_chunks);
    for (c, ew) in b
        .q4_chunks("embed", vocab, d, vocab_chunks)
        .into_iter()
        .enumerate()
    {
        let esp = [seq, groups, GROUP, vc];
        let v_iota = b.g.iota(&esp, 3);
        // a plain number: literals belong to the closure's own trace
        let off = (c * vc) as f64;
        let xc =
            b.g.comp(
                &[seq, groups, GROUP],
                &[vc],
                vec![
                    arg(ew[0], vec![unit(4, 3), unit(4, 1), unit(4, 2)]),
                    arg(ew[1], vec![unit(4, 3), unit(4, 1)]),
                    arg(ew[2], vec![unit(4, 3), unit(4, 1)]),
                    tok_at(4, unit(4, 0)),
                    arg(v_iota, ident(4)),
                ],
                Some(Monoid::sum()),
                // `pick(row, token - off)`, not `pick(row + off, token)`: the
                // decode lowering reads a pick of the reduce index as loop
                // bounds (one row, or none in another block) only in this form
                move |v| vec![pick(v[4], v[3] + lit(-off), v[0] * v[1] + v[2])],
            )
            .unwrap();
        parts.push(xc);
    }
    let x3 = if parts.len() == 1 {
        parts[0]
    } else {
        b.g.comp(
            &[seq, groups, GROUP],
            &[],
            parts.iter().map(|&p| arg(p, ident(3))).collect(),
            None,
            |v| vec![v.iter().skip(1).fold(v[0], |a, &x| a + x)],
        )
        .unwrap()
    };
    let x0 = b.g.reshape(x3, &[seq, d]).unwrap();
    // rope angles of each token's position, from a [cos | sin][ctx, half] table
    let rope = b.input("rope".into(), &[2, ctx, half], None);
    let rsp = [seq, 2, half, ctx];
    let (rp, rt) = (b.g.iota(&rsp, 3), b.g.iota(&rsp, 0));
    let cs =
        b.g.comp(
            &[seq, 2, half],
            &[ctx],
            vec![
                arg(rope, vec![unit(4, 1), unit(4, 3), unit(4, 2)]),
                arg(rp, ident(4)),
                arg(rt, ident(4)),
                len_at(4),
            ],
            Some(Monoid::sum()),
            |v| {
                let pos = lenv(v[3]) - lit(seq_f) + v[2];
                vec![pick(v[1], pos, v[0])]
            },
        )
        .unwrap();
    let cs = b.g.reshape(cs, &[seq, hd]).unwrap();
    let cos_rows = Rows {
        src: cs,
        off: 0,
        stride: hd as i64,
    };
    let sin_rows = Rows {
        src: cs,
        off: half as i64,
        stride: hd as i64,
    };
    let q_rows = heads * hd;
    let k_off = q_rows as i64;
    let v_off = (q_rows + kvh * hd) as i64;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut outputs = Vec::new();
    let mut keep = Vec::new();
    // each block's embedding gather its own kernel: merged into one
    // multi-output kernel, the gathers lose their runtime row bounds and
    // read every row of every block
    if vocab_chunks > 1 {
        keep.extend(&parts);
    }
    let mut x = Rows {
        src: x0,
        off: 0,
        stride: d as i64,
    };
    for l in 0..layers {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        let qkv_w = b.q4(&p("qkv"), dims.qkv_rows(), d);
        let o_w = b.q4(&p("o"), d, q_rows);
        let gate_w = b.q4(&p("gate"), ff, d);
        let up_w = b.q4(&p("up"), ff, d);
        let down_w = b.q4(&p("down"), d, ff);
        let ln1 = b.input(p("ln1"), &[d], None);
        let ln2 = b.input(p("ln2"), &[d], None);
        let qn = b.input(p("q_norm"), &[hd], None);
        let kn = b.input(p("k_norm"), &[hd], None);
        let kv = dims.kv_f16.then_some(Packing::F16);
        let k_cache = b.input(p("k_cache"), &[kvh, ctx, 2, half], kv);
        let v_cache = b.input(p("v_cache"), &[kvh, hd, ctx], kv);

        let normed = b.rms_norm(x, seq, 0, d, ln1, eps);
        let normed = Rows {
            src: normed,
            off: 0,
            stride: d as i64,
        };
        let qkv = b.linear(normed, seq, 0, qkv_w);
        // per-head Σ x² of q and k heads: [seq, heads + kvh]
        let nh = heads + kvh;
        let hss =
            b.g.comp(
                &[seq, nh],
                &[hd],
                vec![arg(
                    qkv,
                    vec![unit(3, 0), form(3, &[(1, hd as i64), (2, 1)], 0)],
                )],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        // QK-norm then rotate-half rope over (t, head…, half h, i):
        // element 64h+i pairs with 64(1−h)+i, the sin term's sign is 2h−1
        #[allow(clippy::too_many_arguments)]
        let roped = |b: &mut Builder,
                     base: i64,
                     out: &[usize],
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
            let cos = b.row(cos_rows, unit(rank, 0), unit(rank, i));
            let sin = b.row(sin_rows, unit(rank, 0), unit(rank, i));
            b.g.comp(
                out,
                &[],
                vec![
                    arg(qkv, vec![unit(rank, 0), self_f]),
                    arg(qkv, vec![unit(rank, 0), other]),
                    arg(w, vec![w_self]),
                    arg(w, vec![w_other]),
                    arg(hss, vec![unit(rank, 0), ss_head]),
                    cos,
                    sin,
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
        // q: [seq, kvh, per, 2, half], head kvh·per + r, scaled by 1/√hd
        let q_rot = roped(
            &mut b,
            0,
            &[seq, kvh, per, 2, half],
            form(5, &[(1, (per * hd) as i64), (2, hd as i64)], 0),
            3,
            4,
            form(5, &[(1, per as i64), (2, 1)], 0),
            qn,
            scale,
        );
        // k: [seq, kvh, 2, half]
        let k_rot = roped(
            &mut b,
            k_off,
            &[seq, kvh, 2, half],
            form(4, &[(1, hd as i64)], 0),
            2,
            3,
            form(4, &[(1, 1)], heads as i64),
            kn,
            1.0,
        );
        // cache rows len−seq ≤ j < len, in place
        let in_step = |len: ExprId, j: ExprId, new: ExprId, old: ExprId| {
            let le = len - lit(1.0) - j + lit(0.5);
            let ge = j - len + lit(seq_f) + lit(0.5);
            select(le, select(ge, new, old), old)
        };
        // the token row j receives: decode, the one token; prefill, token
        // j − p0 — a runtime offset, so a sum over the chunk with one live
        // term (`pick`) that the lowering runs as a one-iteration loop
        let kc_shape = [kvh, ctx, 2, half];
        let vc_shape = [kvh, hd, ctx];
        let (k_val, v_val) = if seq == 1 {
            (
                arg(
                    k_rot,
                    vec![Form::zero(4), unit(4, 0), unit(4, 2), unit(4, 3)],
                ),
                arg(
                    qkv,
                    vec![Form::zero(3), form(3, &[(0, hd as i64), (1, 1)], v_off)],
                ),
            )
        } else {
            let sp = [kvh, ctx, 2, half, seq];
            let (j, t) = (b.g.iota(&sp, 1), b.g.iota(&sp, 4));
            let kp =
                b.g.comp(
                    &kc_shape,
                    &[seq],
                    vec![
                        arg(k_rot, vec![unit(5, 4), unit(5, 0), unit(5, 2), unit(5, 3)]),
                        arg(j, ident(5)),
                        arg(t, ident(5)),
                        len_at(5),
                    ],
                    Some(Monoid::sum()),
                    |v| {
                        let p0 = lenv(v[3]) - lit(seq_f);
                        vec![pick(v[2], v[1] - p0, v[0])]
                    },
                )
                .unwrap();
            let sp = [kvh, hd, ctx, seq];
            let (j, t) = (b.g.iota(&sp, 2), b.g.iota(&sp, 3));
            let vp =
                b.g.comp(
                    &vc_shape,
                    &[seq],
                    vec![
                        arg(
                            qkv,
                            vec![unit(4, 3), form(4, &[(0, hd as i64), (1, 1)], v_off)],
                        ),
                        arg(j, ident(4)),
                        arg(t, ident(4)),
                        len_at(4),
                    ],
                    Some(Monoid::sum()),
                    |v| {
                        let p0 = lenv(v[3]) - lit(seq_f);
                        vec![pick(v[2], v[1] - p0, v[0])]
                    },
                )
                .unwrap();
            (arg(kp, ident(4)), arg(vp, ident(3)))
        };
        let j_iota = b.g.iota(&kc_shape, 1);
        let k_new =
            b.g.comp(
                &kc_shape,
                &[],
                vec![
                    k_val,
                    arg(k_cache, ident(4)),
                    arg(j_iota, ident(4)),
                    len_at(4),
                ],
                None,
                |v| vec![in_step(lenv(v[3]), v[2], v[0], v[1])],
            )
            .unwrap();
        let jv_iota = b.g.iota(&vc_shape, 2);
        let v_new =
            b.g.comp(
                &vc_shape,
                &[],
                vec![
                    v_val,
                    arg(v_cache, ident(3)),
                    arg(jv_iota, ident(3)),
                    len_at(3),
                ],
                None,
                |v| vec![in_step(lenv(v[3]), v[2], v[0], v[1])],
            )
            .unwrap();
        outputs.push(k_new);
        outputs.push(v_new);

        // causal: key j is visible to token t at position len − seq + t
        let visible = |len: ExprId, t: ExprId, j: ExprId| lenv(len) - lit(seq_f) + t - j + lit(0.5);
        // scores[t, kvh, r, j] over (h, i): space (t, kvh, r, j, h, i)
        let sp = [seq, kvh, per, kctx, 2, half];
        let (ts, js) = (b.g.iota(&sp, 0), b.g.iota(&sp, 3));
        let scores =
            b.g.comp(
                &[seq, kvh, per, kctx],
                &[2, half],
                vec![
                    arg(
                        q_rot,
                        vec![unit(6, 0), unit(6, 1), unit(6, 2), unit(6, 4), unit(6, 5)],
                    ),
                    arg(k_new, vec![unit(6, 1), unit(6, 3), unit(6, 4), unit(6, 5)]),
                    arg(ts, ident(6)),
                    arg(js, ident(6)),
                    len_at(6),
                ],
                Some(Monoid::sum()),
                |v| {
                    let c = visible(v[4], v[2], v[3]);
                    vec![select(c, v[0] * v[1], lit(0.0))]
                },
            )
            .unwrap();
        // softmax statistics over j: space (t, kvh, r, j)
        let sp = [seq, kvh, per, kctx];
        let (tr, jr) = (b.g.iota(&sp, 0), b.g.iota(&sp, 3));
        let s_args = |m: Option<TensorId>| {
            let mut a = vec![
                arg(scores, ident(4)),
                arg(tr, ident(4)),
                arg(jr, ident(4)),
                len_at(4),
            ];
            if let Some(m) = m {
                a.push(arg(m, vec![unit(4, 0), unit(4, 1), unit(4, 2)]));
            }
            a
        };
        let m =
            b.g.comp(
                &[seq, kvh, per],
                &[kctx],
                s_args(None),
                Some(Monoid::max()),
                |v| {
                    let c = visible(v[3], v[1], v[2]);
                    vec![select(c, v[0], lit(-1e30))]
                },
            )
            .unwrap();
        let lsum =
            b.g.comp(
                &[seq, kvh, per],
                &[kctx],
                s_args(Some(m)),
                Some(Monoid::sum()),
                |v| {
                    let c = visible(v[3], v[1], v[2]);
                    vec![select(c, (v[0] - v[4]).exp(), lit(0.0))]
                },
            )
            .unwrap();
        // attn[t, kvh, r, c] = Σ_j p·v / l: space (t, kvh, r, c, j)
        let sp = [seq, kvh, per, hd, kctx];
        let (to, jo) = (b.g.iota(&sp, 0), b.g.iota(&sp, 4));
        let mlv = vec![unit(5, 0), unit(5, 1), unit(5, 2)];
        let attn =
            b.g.comp(
                &[seq, kvh, per, hd],
                &[kctx],
                vec![
                    arg(scores, vec![unit(5, 0), unit(5, 1), unit(5, 2), unit(5, 4)]),
                    arg(m, mlv.clone()),
                    arg(lsum, mlv),
                    arg(v_new, vec![unit(5, 1), unit(5, 3), unit(5, 4)]),
                    arg(to, ident(5)),
                    arg(jo, ident(5)),
                    len_at(5),
                ],
                Some(Monoid::sum()),
                |v| {
                    let c = visible(v[6], v[4], v[5]);
                    let pv = (v[0] - v[1]).exp() * v[3] / v[2];
                    vec![select(c, pv, lit(0.0))]
                },
            )
            .unwrap();
        let attn = b.g.reshape(attn, &[seq, q_rows]).unwrap();
        let attn = Rows {
            src: attn,
            off: 0,
            stride: q_rows as i64,
        };
        let proj = b.linear(attn, seq, 0, o_w);
        let res1 = b.add(x, proj, seq, d);
        let res1_rows = Rows {
            src: res1,
            off: 0,
            stride: d as i64,
        };
        let normed2 = b.rms_norm(res1_rows, seq, 0, d, ln2, eps);
        let normed2 = Rows {
            src: normed2,
            off: 0,
            stride: d as i64,
        };
        let gate = b.linear(normed2, seq, 0, gate_w);
        let up = b.linear(normed2, seq, 0, up_w);
        let act =
            b.g.map(&[gate, up], |a| {
                a[0] / (ExprId::from_f64(1.0) + (-a[0]).exp()) * a[1]
            })
            .unwrap();
        let act = Rows {
            src: act,
            off: 0,
            stride: ff as i64,
        };
        let down = b.linear(act, seq, 0, down_w);
        let res2 = b.add(res1_rows, down, seq, d);
        keep.extend([res1, res2]);
        x = Rows {
            src: res2,
            off: 0,
            stride: d as i64,
        };
    }
    // logits of the last token only
    let norm = b.input("model.norm".into(), &[d], None);
    let lms = b.q4_chunks("lm_head", vocab, d, vocab_chunks);
    let normed = b.rms_norm(x, 1, seq as i64 - 1, d, norm, eps);
    let normed = Rows {
        src: normed,
        off: 0,
        stride: d as i64,
    };
    let logits: Vec<TensorId> = lms.iter().map(|&lm| b.linear(normed, 1, 0, lm)).collect();
    // greedy: (index, value) of the first maximum, per block then across
    // blocks (in order, so ties still go to the first)
    let vi = b.g.iota(&[vc], 0);
    let ams: Vec<TensorId> = logits
        .iter()
        .enumerate()
        .map(|(c, &l)| {
            let off = (c * vc) as f64;
            b.g.comp_tuple(
                &[2],
                &[vc],
                vec![arg(l, vec![Form::zero(1), unit(1, 0)]), arg(vi, ident(1))],
                Some(Monoid::argmax()),
                move |v| vec![v[1] + lit(off), v[0]],
            )
            .unwrap()
        })
        .collect();
    let am = if ams.len() == 1 {
        ams[0]
    } else {
        // [block, (index, value)], then the argmax over blocks
        let k = ams.len();
        let bi = b.g.iota(&[k, 2], 0);
        let mut args: Vec<_> = ams.iter().map(|&a| arg(a, vec![unit(2, 1)])).collect();
        args.push(arg(bi, ident(2)));
        let stacked =
            b.g.comp(&[k, 2], &[], args, None, move |v| {
                let c = v[k];
                let mut acc = lit(0.0);
                for (j, &x) in v[..k].iter().enumerate() {
                    acc += pick(c, lit(j as f64), x);
                }
                vec![acc]
            })
            .unwrap();
        b.g.comp_tuple(
            &[2],
            &[k],
            vec![
                arg(stacked, vec![unit(1, 0), Form::constant(1, 0)]),
                arg(stacked, vec![unit(1, 0), Form::constant(1, 1)]),
            ],
            Some(Monoid::argmax()),
            |v| vec![v[0], v[1]],
        )
        .unwrap()
    };
    // state ← (argmax, len): the next step's token and its position
    let si = b.g.iota(&[2], 0);
    let state_new =
        b.g.comp(
            &[2],
            &[],
            vec![
                arg(am, vec![Form::constant(1, 0)]),
                len_at(1),
                arg(state, ident(1)),
                arg(si, ident(1)),
            ],
            None,
            |v| {
                let val = select(lit(0.5) - v[3], v[0], lenv(v[1]));
                vec![select(lit(1.5) - v[3], val, v[2])]
            },
        )
        .unwrap();
    outputs.insert(0, state_new);
    outputs.insert(0, logits[0]);
    let logit_chunks = logits[1..].to_vec();
    DecodeGraph {
        g: b.g,
        seq,
        kctx,
        outputs,
        keep,
        logit_chunks,
        inputs: b.inputs,
        q4: seq > 1 && prefill_q4,
    }
}

/// The passes: maps fused, siblings (gate/up) merged, epilogues
/// inlined; for prefill (`seq > 1`) the reduce dims of the matmuls
/// merged, so they are tensor-core tiles over the dequantized weights,
/// stored f16 when `half_tiles` (CUDA's m16n8k16 tiles; Metal's
/// simdgroup tile stages f32). Returns the graph and its outputs:
/// `dg.outputs` (same order), then `dg.keep`, then `dg.logit_chunks`.
pub fn compiled_decode(dg: &DecodeGraph, half_tiles: bool) -> (TensorGraph, Vec<TensorId>) {
    let all: Vec<TensorId> = dg
        .outputs
        .iter()
        .chain(&dg.keep)
        .chain(&dg.logit_chunks)
        .copied()
        .collect();
    let (g, outs) = dg.g.substitute_maps(&all);
    let (g, outs) = g.eliminate_common(&outs);
    // gate|up as one tuple-valued matmul: CUDA's tiles take it; Metal's
    // simdgroup tile takes one output, so a prefill there keeps them apart
    let (g, outs) = if dg.seq > 1 && !half_tiles {
        (g, outs)
    } else {
        g.merge_siblings(&outs)
    };
    let (mut g, mut outs) = if dg.seq > 1 {
        g.merge_reduce_dims(&outs)
    } else {
        (g, outs)
    };
    // TANG_SPLIT_GEMV=P: decode reductions over at least TANG_SPLIT_MIN_K
    // keys (default 8192: the down projection) split-K into P chunks
    let parts: usize = std::env::var("TANG_SPLIT_GEMV")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if dg.seq == 1 && parts > 1 {
        let min_k: usize = std::env::var("TANG_SPLIT_MIN_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        // (the graph is dead-code free here and after each split)
        let live = |g: &TensorGraph, _outs: &[TensorId]| {
            (0..g.nodes().len())
                .filter(|&i| {
                    g.comp_at(i).is_some_and(|c| {
                        c.monoid.is_some()
                            && c.tuple <= 1
                            && c.red.iter().map(|d| d.n()).product::<usize>() >= min_k
                            && c.red[0].n() % parts == 0
                            && c.red[0].n() / parts >= 2
                    })
                })
                .collect::<Vec<_>>()
        };
        if std::env::var_os("TANG_SPLIT_DEBUG").is_some() {
            for i in 0..g.nodes().len() {
                if let Some(c) = g.comp_at(i).filter(|c| c.monoid.is_some()) {
                    let red: Vec<usize> = c.red.iter().map(|d| d.n()).collect();
                    eprintln!(
                        "red node {i}: shape {:?} red {red:?} tuple {}",
                        g.nodes()[i].shape,
                        c.tuple
                    );
                }
            }
        }
        let todo = live(&g, &outs).len();
        for _ in 0..todo {
            let Some(&t) = live(&g, &outs).first() else {
                break;
            };
            match g.split_reduction(&outs, TensorId::from_index(t), parts) {
                Ok((g2, o2)) => {
                    g = g2;
                    outs = o2;
                }
                Err(_) => break,
            }
        }
    }
    g.schedule_inline(&outs);
    g.schedule_epilogues(&outs);
    // the norms feeding a 4-bit GEMV and the roped key of a cache update
    // are recomputed where they are read
    let u4: Vec<usize> = dg
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.packing == Some(Packing::U4))
        .map(|(k, _)| k)
        .collect();
    let caches = outs[1..dg.outputs.len()].to_vec();
    g.schedule_decode_maps(&outs, &u4, &caches);
    if dg.q4 {
        // a prefill over packed weights: the dequantization maps are
        // evaluated where the matmul loads the packed words, and the
        // activations the matmuls read stay materialized (a norm would be
        // recomputed per register block and k step otherwise)
        use tang_compile::tensor::Op;
        let n = g.nodes().len();
        let u4n: Vec<usize> = (0..n)
            .filter(|&i| matches!(g.nodes()[i].op, Op::Input(k) if u4.contains(&k)))
            .collect();
        let reads_any = |i: usize, set: &[usize]| {
            g.comp_at(i)
                .is_some_and(|c| c.args.iter().any(|a| set.contains(&a.src.index())))
        };
        let dq: Vec<usize> = (0..n)
            .filter(|&i| g.comp_at(i).is_some_and(|c| c.monoid.is_none()) && reads_any(i, &u4n))
            .collect();
        let mm: Vec<usize> = (0..n)
            .filter(|&i| {
                g.comp_at(i)
                    .is_some_and(|c| c.monoid.is_some() && c.red.len() == 2)
                    && (reads_any(i, &dq) || reads_any(i, &u4n))
            })
            .collect();
        let read_by_mm = |i: usize| mm.iter().any(|&r| reads_any(r, &[i]));
        let undo: Vec<usize> = (0..n)
            .filter(|&i| {
                g.nodes()[i].inline
                    && g.comp_at(i).is_some_and(|c| c.monoid.is_none())
                    && !dq.contains(&i)
                    && read_by_mm(i)
            })
            .collect();
        let inl: Vec<usize> = dq.iter().copied().filter(|&i| read_by_mm(i)).collect();
        for i in undo {
            g.set_inline(TensorId::from_index(i), false);
        }
        for i in inl {
            g.set_inline(TensorId::from_index(i), true);
        }
    }
    if dg.seq > 1 && half_tiles && std::env::var_os("TANG_PREFILL_TF32").is_none() {
        // prefill: the dequantized weights stored f16, so the matmuls run
        // on f16 tensor cores (the f32 activations are converted while
        // staging)
        use tang_compile::tensor::{Format, Op, Precision};
        let u4_nodes: Vec<usize> = (0..g.nodes().len())
            .filter(|&i| matches!(g.nodes()[i].op, Op::Input(k) if u4.contains(&k)))
            .collect();
        for i in 0..g.nodes().len() {
            let n = &g.nodes()[i];
            let is_dq = !n.inline
                && g.comp_at(i).is_some_and(|c| {
                    c.monoid.is_none() && c.args.iter().any(|a| u4_nodes.contains(&a.src.index()))
                });
            if is_dq {
                g.set_precision(TensorId::from_index(i), Precision::storage(Format::F16));
            }
        }
    }
    (g, outs)
}

/// Packed inputs and the in-place updates (`state`, the caches) for
/// `to_cuda_decode` / `to_msl_decode`; `outputs` from [`compiled_decode`].
pub fn decode_opts(dg: &DecodeGraph, outputs: &[TensorId]) -> DecodeOpts {
    let packed = dg
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(k, s)| s.packing.map(|p| (k, p)))
        .collect();
    let inplace = dg
        .inplace()
        .into_iter()
        .map(|(t, k)| {
            let i = dg.outputs.iter().position(|&o| o == t).unwrap();
            (outputs[i], k)
        })
        .collect();
    DecodeOpts {
        packed,
        inplace,
        mma: dg.seq > 1,
        ..Default::default()
    }
}

/// Slots for `program`'s buffers in kernel order: inputs keep their own,
/// an in-place output shares its donor's (`inplace`: output tensor,
/// input index), every other buffer takes a free slot of exactly its
/// size, else the smallest free one that fits, else a new one. Same-role
/// buffers of successive layers have the same size, so a step needs about
/// one layer's worth of intermediates. Returns `(slot, slot_words)`.
pub fn plan_slots(
    program: &WgslProgram,
    outputs: &[TensorId],
    inplace: &[(TensorId, usize)],
) -> (Vec<usize>, Vec<usize>) {
    let nb = program.buffers.len();
    let buf_of = |t: TensorId| {
        let t = program.storage(t);
        program.buffers.iter().position(|b| b.node == t).unwrap()
    };
    let words = |b: usize| {
        let spec = &program.buffers[b];
        match spec.format {
            tang_compile::tensor::Format::F32 => spec.len,
            _ => spec.len.div_ceil(2),
        }
        .max(1)
    };
    let mut last = vec![0usize; nb];
    for (p, k) in program.kernels.iter().enumerate() {
        for &t in k.reads.iter().chain([&k.node]) {
            last[buf_of(t)] = p;
        }
    }
    for &t in outputs {
        last[buf_of(t)] = usize::MAX;
    }
    let mut slot = vec![usize::MAX; nb];
    let mut slot_words: Vec<usize> = Vec::new();
    for (b, spec) in program.buffers.iter().enumerate() {
        if spec.input.is_some() {
            slot[b] = slot_words.len();
            slot_words.push(words(b));
        }
    }
    for &(o, k) in inplace {
        let d = program
            .buffers
            .iter()
            .position(|b| b.input == Some(k))
            .unwrap();
        slot[buf_of(o)] = slot[d];
        last[buf_of(o)] = usize::MAX;
        last[d] = usize::MAX;
    }
    let mut free: Vec<usize> = Vec::new();
    for (p, k) in program.kernels.iter().enumerate() {
        let b = buf_of(k.node);
        if slot[b] == usize::MAX {
            let w = words(b);
            // an exact fit, else a new slot when the best fit would waste
            // more than half of it (it is someone else's exact fit later)
            let pick = free.iter().position(|&s| slot_words[s] == w).or_else(|| {
                free.iter()
                    .enumerate()
                    .filter(|(_, &s)| slot_words[s] >= w && slot_words[s] <= 2 * w)
                    .min_by_key(|(_, &s)| slot_words[s])
                    .map(|(i, _)| i)
            });
            slot[b] = match pick {
                Some(i) => free.swap_remove(i),
                None => {
                    slot_words.push(w);
                    slot_words.len() - 1
                }
            };
        }
        // buffers whose last use is this kernel free their slots
        let mut dying: Vec<usize> = k
            .reads
            .iter()
            .chain([&k.node])
            .map(|&t| buf_of(t))
            .filter(|&x| last[x] == p && program.buffers[x].input.is_none())
            .collect();
        dying.sort_unstable();
        dying.dedup();
        for x in dying {
            free.push(slot[x]);
        }
    }
    (slot, slot_words)
}

/// Kernels reordered so that a kernel reading only graph inputs (a
/// weight's dequantization) runs just before its first reader: with
/// a liveness memory plan, one layer's f32 weights are live at a time.
pub fn lazy_order(program: &mut WgslProgram) {
    let n = program.kernels.len();
    let producer: HashMap<TensorId, usize> = program
        .kernels
        .iter()
        .enumerate()
        .map(|(i, k)| (program.storage(k.node), i))
        .collect();
    let deps: Vec<Vec<usize>> = program
        .kernels
        .iter()
        .map(|k| {
            k.reads
                .iter()
                .filter_map(|t| producer.get(&program.storage(*t)).copied())
                .collect()
        })
        .collect();
    let leaf: Vec<bool> = deps.iter().map(|d| d.is_empty()).collect();
    let mut order = Vec::with_capacity(n);
    let mut placed = vec![false; n];
    for i in 0..n {
        if leaf[i] {
            continue;
        }
        for &d in &deps[i] {
            if leaf[d] && !placed[d] {
                placed[d] = true;
                order.push(d);
            }
        }
        placed[i] = true;
        order.push(i);
    }
    order.extend((0..n).filter(|&i| !placed[i]));
    let mut ks: Vec<Option<_>> = program.kernels.drain(..).map(Some).collect();
    program.kernels = order.iter().map(|&i| ks[i].take().unwrap()).collect();
}

/// `cos`/`sin` of rope at `pos` (`head_dim/2` each), from f64 angles.
pub fn rope_angles(dims: &QwenDims, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dims.head_dim / 2;
    (0..half)
        .map(|i| {
            let inv = dims.theta.powf(-(2.0 * i as f64) / dims.head_dim as f64);
            let a = pos as f64 * inv;
            (a.cos() as f32, a.sin() as f32)
        })
        .unzip()
}

/// The `rope` input: `[cos | sin][ctx][head_dim/2]`.
pub fn rope_table(dims: &QwenDims) -> Vec<f32> {
    let rows: Vec<_> = (0..dims.ctx).map(|p| rope_angles(dims, p)).collect();
    let mut t: Vec<f32> = rows.iter().flat_map(|(c, _)| c.iter().copied()).collect();
    t.extend(rows.iter().flat_map(|(_, s)| s.iter().copied()));
    t
}

/// The `chunk` input of a prefill: its tokens, then the position of the
/// first.
pub fn chunk_input(tokens: &[usize], p0: usize) -> Vec<f32> {
    tokens.iter().chain([&p0]).map(|&t| t as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The embedding and LM head split into vocabulary blocks compute what
    /// the whole matrices do: same full logits and next token, given the
    /// same weights cut into row blocks.
    #[test]
    fn vocab_chunks_match_whole() {
        let whole = tiny();
        let split = QwenDims {
            vocab_chunks: 2,
            ..whole
        };
        let (gw, gs) = (decode_graph(&whole), decode_graph(&split));
        let mut seed = 7u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let inputs_w: Vec<Vec<f64>> = gw
            .inputs
            .iter()
            .map(|s| {
                let n: usize = s.shape.iter().product();
                match s.name.as_str() {
                    "state" => vec![37.0, 3.0],
                    "rope" => (0..n).map(|i| ((i % 7) as f64 * 0.3).cos()).collect(),
                    _ if s.name.ends_with("cache") => (0..n).map(|_| rnd() - 0.5).collect(),
                    _ => match s.packing {
                        Some(Packing::U4) => (0..n).map(|_| (rnd() * 15.999).floor()).collect(),
                        Some(_) => (0..n).map(|_| 0.01 + 0.02 * rnd()).collect(),
                        None => (0..n).map(|_| 1.0 + 0.2 * rnd()).collect(),
                    },
                }
            })
            .collect();
        // block c of `embed.q` (etc.) is rows c·V/2.. of the whole tensor
        let inputs_s: Vec<Vec<f64>> = gs
            .inputs
            .iter()
            .map(|s| {
                for base in ["embed", "lm_head"] {
                    if let Some(rest) = s.name.strip_prefix(&format!("{base}.")) {
                        if let Some((c, part)) = rest.split_once('.') {
                            let c: usize = c.parse().unwrap();
                            let w = &inputs_w[gw.input(&format!("{base}.{part}"))];
                            let per = w.len() / split.vocab_chunks;
                            return w[c * per..(c + 1) * per].to_vec();
                        }
                    }
                }
                inputs_w[gw.input(&s.name)].clone()
            })
            .collect();
        let rw = gw.g.eval(&inputs_w, &gw.outputs[..2]);
        let mut outs = vec![gs.outputs[0], gs.outputs[1]];
        outs.extend(&gs.logit_chunks);
        let rs = gs.g.eval(&inputs_s, &outs);
        let mut logits = rs[0].clone();
        for c in &rs[2..] {
            logits.extend(c);
        }
        assert_eq!(logits.len(), whole.vocab);
        let scale = rw[0].iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let err = logits
            .iter()
            .zip(&rw[0])
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(err <= 1e-9 * scale, "logits differ: {err:e} of {scale:e}");
        assert_eq!(rs[1], rw[1], "next token and position");
    }

    #[test]
    fn decode_graph_lowers() {
        let dims = tiny();
        let dg = decode_graph(&dims);
        let (g, outs) = compiled_decode(&dg, true);
        let opts = decode_opts(&dg, &outs);
        for p in [
            g.to_cuda_decode(&outs, &opts),
            g.to_msl_decode(&outs, &opts),
            g.to_glsl_decode(&outs, &opts),
        ] {
            let transpiled = p.native.iter().filter(|n| !**n).count();
            assert!(p.native.iter().filter(|n| **n).count() >= 10 * dims.layers);
            assert!(transpiled <= dims.layers + 2, "{transpiled} transpiled");
            // packed weights are sized in words
            let k = dg.input("model.layers.0.qkv.q");
            let q = p
                .program
                .buffers
                .iter()
                .find(|b| b.input == Some(k))
                .unwrap();
            assert_eq!(q.len, dims.qkv_rows() * dims.d_model / 8);
        }
    }

    #[test]
    fn prefill_graph_lowers() {
        let dims = tiny();
        let dg = step_graph(&dims, 6, dims.ctx);
        let (g, outs) = compiled_decode(&dg, true);
        let opts = decode_opts(&dg, &outs);
        let p = g.to_cuda_decode(&outs, &opts);
        let mma = p
            .program
            .kernels
            .iter()
            .filter(|k| k.source.contains("mma"))
            .count();
        // qkv (fused rows), o, gate|up (merged siblings), down per layer
        assert_eq!(mma, 4 * dims.layers, "{mma} mma tiles");
        // the cache updates are the one-iteration gathers, in place
        let inplace = p
            .program
            .kernels
            .iter()
            .filter(|k| k.source.contains("in-place"))
            .count();
        assert_eq!(inplace, 2 * dims.layers + 1);
    }

    /// Kernels of the OpenCL lowering of a prefill step whose first line
    /// names them.
    fn opencl_prefill_heads(dims: &QwenDims, seq: usize) -> Vec<String> {
        let dg = step_graph(dims, seq, dims.ctx);
        let (g, outs) = compiled_decode(&dg, false);
        let opts = decode_opts(&dg, &outs);
        g.to_opencl_decode(&outs, &opts)
            .program
            .kernels
            .iter()
            .map(|k| k.source.lines().next().unwrap_or("").to_string())
            .collect()
    }

    /// Every projection of an OpenCL prefill (q|k|v, o, gate, up, down per
    /// layer) is a register-block kernel (`opencl_mma_kernel`): both
    /// operands contiguous along k — the activations `[token][k]`, the
    /// weights `[row][k]`. The LM head is the last token's GEMV.
    #[test]
    fn opencl_prefill_projections_are_register_blocks() {
        let dims = tiny();
        let heads = opencl_prefill_heads(&dims, 6);
        let mm = heads
            .iter()
            .filter(|h| h.starts_with("// tang-compile: OpenCL C (native)") && h.contains("matmul"))
            .count();
        assert_eq!(mm, 5 * dims.layers, "{heads:#?}");
    }

    /// With `prefill_q4` every projection is the register block over the
    /// packed weights, and no kernel materializes a dequantized weight.
    #[test]
    fn opencl_prefill_q4_reads_packed_weights() {
        // ff of 40 groups: the down projection's reduce dims [40, 64] are
        // past `tiled_reduction`'s cap (Qwen3-1.7B's are [96, 64])
        let dims = QwenDims {
            prefill_q4: true,
            ff: 40 * GROUP,
            ..tiny()
        };
        let heads = opencl_prefill_heads(&dims, 6);
        let q4 = heads
            .iter()
            .filter(|h| h.contains("matmul over 4-bit weights"))
            .count();
        assert_eq!(q4, 5 * dims.layers, "{heads:#?}");
        assert!(
            !heads.iter().any(|h| h.contains("map over packed inputs")),
            "{heads:#?}"
        );
    }

    /// `TANG_CL_MM_KIND=stage`: every projection is the staged kernel.
    #[test]
    fn opencl_prefill_staged_kernel_covers_projections() {
        let dims = QwenDims {
            prefill_q4: true,
            ff: 40 * GROUP,
            ..tiny()
        };
        std::env::set_var("TANG_CL_MM_KIND", "stage");
        let heads = opencl_prefill_heads(&dims, 12);
        std::env::remove_var("TANG_CL_MM_KIND");
        let staged = heads.iter().filter(|h| h.contains("staged")).count();
        assert_eq!(staged, 5 * dims.layers, "{heads:#?}");
    }
}
