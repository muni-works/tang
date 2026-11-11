//! The truth track: a slow, plain f32 CPU forward of Qwen3.8-Flash-Next straight from its GGUF.
//!
//! Every fast kernel is checked against this, so it is written for reading, not speed: each block is
//! the formula in `docs/strata.md` (as corrected in `docs/flash-next-tensors.md`), every weight is
//! dequantised to f32, every activation stays f32, and reductions that decide discrete things
//! (router softmax, logprobs) run in f64. The one concession to time is that a whole sequence goes
//! through a layer at once, so each dense weight is dequantised once per layer, and the matmuls and
//! independent heads run on rayon.
//!
//! What it computes for a token sequence `t[0..T]` starting at position 0:
//!
//! ```text
//! R[c] = embed(t) for the 4 hyper-connection streams
//! for l in 0..48:
//!   if l is the PLE layer (1):  R = ple(R)
//!   x, inj = hc_read(R, hc_attn_*);  R = hc_write(R, gdn(x) or qsa(x), inj)
//!   x, inj = hc_read(R, hc_ffn_*);   R = hc_write(R, moe(x), inj)
//! logits = output @ hc_read(R, output_hc_*)
//! ```
//!
//! State is what a decode would carry: GDN conv history (3 × 10240) and recurrent state
//! (48 × 128 × 128) start at zero; the PLE conv history (9 × 10240) starts at zero; QSA attends over
//! every earlier cell of the sequence, through the indexer's block selection.
//!
//! The same file also runs GGUF architecture `qwen3next` (Qwen3-Next-80B-A3B, Qwen3-Coder-Next):
//! one residual stream with pre-norms instead of hyper-connections, gated attention with no
//! indexer, no PLE ([`Arch::Qwen3Next`], `FlashRef::forward_plain`). Its parity against Hugging
//! Face is `scripts/qwen3next_tiny.py`.
//!
//! Routed experts are dequantised lazily, one (layer, expert) at a time, only for the experts some
//! token picked. N-gram table rows are read with `pread` at their offsets (16 per token).

use super::requant::{self, DensePolicy, Format};
use crate::gguf::{f16_to_f32, f32_to_f16, GgmlType, Gguf, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `LLAMA_TOKEN_NULL`: a predecessor before the start of the sequence.
const TOKEN_NULL: i64 = -1;

/// Which GGUF architecture the file is: the block wiring and a few conventions differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// Qwen3.8-Flash-Next: hyper-connections, QSA, PLE; GDN v head `h` reads k head `h % hk`,
    /// output gate sigmoid, `ssm_alpha` / `ssm_beta` separate.
    Qwen4Exp,
    /// Qwen3-Next and Qwen3-Coder-Next: one residual stream with pre-norms (`attn_norm`,
    /// `post_attention_norm`, `output_norm`, all with the +1 folded in by the converter), dense
    /// gated attention, GDN v head `h` reads k head `h / (hv / hk)` (grouped), output gate SiLU,
    /// `ssm_ba` fused per k head as `[b × r | a × r]`.
    Qwen3Next,
}

/// The model geometry, from GGUF metadata (`qwen4exp.*` or `qwen3next.*`).
#[derive(Clone, Debug)]
pub struct Hparams {
    pub arch: Arch,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// Hyper-connection streams and bottleneck rank.
    pub hc: usize,
    pub hc_lr: usize,
    // QSA
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub rope_sections: [usize; 4],
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    /// Cells per indexer block (`attention.compress_ratios` of the QSA layers).
    pub kpool: usize,
    // GDN
    pub ssm_d_conv: usize,
    pub ssm_d_state: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    // MoE
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub expert_weights_scale: f32,
    /// Per layer: true for recurrent (GDN) mixers, false for QSA.
    pub is_recurrent: Vec<bool>,
    pub ple: Option<PleParams>,
}

/// The n-gram ("PLE") block's constants.
#[derive(Clone, Debug)]
pub struct PleParams {
    pub layer: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: usize,
    pub eos: i64,
    pub head_dim: usize,
    pub multipliers: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub head_vocab: Vec<u64>,
}

impl PleParams {
    pub fn n_heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }

    /// The table rows for the token at `i` of `tokens`, exactly as llama.cpp's
    /// `llm_graph_input_qwen4exp_ple::set_input`: the window is the token and its predecessors
    /// (newest first); a missing predecessor or an EOS anywhere in it replaces that slot *and every
    /// older one* with EOS, but the token's own EOS doesn't cut. Head `h` of n-gram `n` takes
    /// `XOR_j(ctx[j] * mult[j]) % vocab[h] + offset[h]` over the first `n` slots.
    pub fn rows(&self, tokens: &[u32], i: usize) -> Vec<u64> {
        let mut ctx = vec![tokens[i] as i64; self.ngram];
        let mut cut = false;
        for s in 1..self.ngram {
            let t = if cut || i < s {
                TOKEN_NULL
            } else {
                tokens[i - s] as i64
            };
            cut = cut || t < 0 || t == self.eos;
            ctx[s] = if cut { self.eos } else { t };
        }
        let mut out = Vec::with_capacity(self.n_heads());
        for n in 2..=self.ngram {
            let mixed = ctx[..n]
                .iter()
                .zip(&self.multipliers)
                .map(|(&c, &m)| (c as u64).wrapping_mul(m))
                .fold(0u64, |acc, v| acc ^ v);
            for g in 0..self.heads_per_ngram {
                let h = (n - 2) * self.heads_per_ngram + g;
                out.push(mixed % self.head_vocab[h] + self.head_offsets[h]);
            }
        }
        out
    }
}

impl Hparams {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        if arch == "qwen3next" {
            return Self::from_gguf_qwen3next(g);
        }
        ensure!(
            arch == "qwen4exp",
            "architecture {arch}, want qwen4exp or qwen3next"
        );
        let k = |s: &str| format!("qwen4exp.{s}");
        let u = |s: &str| g.meta_u64(&k(s)).map(|v| v as usize);
        let n_layer = u("block_count")?;
        let n_embd = u("embedding_length")?;
        let n_vocab = g.info("token_embd.weight")?.dims[1] as usize;
        let eps = g.meta_f64(&k("attention.layer_norm_rms_epsilon"))? as f32;

        let ratios = g.meta_u64s(&k("attention.compress_ratios"))?;
        let ratios: Vec<u64> = if ratios.len() == 1 {
            vec![ratios[0]; n_layer]
        } else {
            ratios
        };
        ensure!(
            ratios.len() >= n_layer,
            "compress_ratios has {} entries",
            ratios.len()
        );
        let mut kpool = 0;
        for &r in &ratios[..n_layer] {
            if r != 0 {
                ensure!(
                    kpool == 0 || kpool == r as usize,
                    "QSA layers with different compress ratios"
                );
                kpool = r as usize;
            }
        }
        let is_recurrent: Vec<bool> = match g.meta.get(&k("attention.recurrent_layers")) {
            Some(v) => {
                let a = v
                    .as_array()
                    .context("attention.recurrent_layers is not an array")?;
                a.iter()
                    .take(n_layer)
                    .map(|x| x.as_u64().unwrap_or(0) != 0)
                    .collect()
            }
            None => {
                let every = g
                    .meta
                    .get(&k("full_attention_interval"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as usize;
                ensure!(every > 0, "full_attention_interval is 0");
                (0..n_layer).map(|i| (i + 1) % every != 0).collect()
            }
        };
        for (i, &r) in is_recurrent.iter().enumerate() {
            ensure!(r || ratios[i] > 0, "layer {i} is full attention without a compress ratio (dense attention isn't implemented)");
        }
        let secs = g.meta_u64s(&k("rope.dimension_sections"))?;
        let mut rope_sections = [0usize; 4];
        for (i, s) in secs.iter().take(4).enumerate() {
            rope_sections[i] = *s as usize;
        }

        let ple = match g.meta.get(&k("ple.layers")) {
            Some(v) => {
                let layers = v.as_array().context("ple.layers")?;
                ensure!(
                    layers.len() == 1,
                    "{} PLE layers (one is supported)",
                    layers.len()
                );
                let layer = layers[0].as_u64().context("ple.layers")? as usize;
                ensure!(
                    layer < n_layer && is_recurrent[layer],
                    "PLE layer {layer} must be a GDN layer"
                );
                let p = PleParams {
                    layer,
                    ngram: u("ple.ngram_size")?,
                    heads_per_ngram: u("ple.heads_per_ngram")?,
                    conv_kernel: u("ple.conv_kernel")?,
                    eos: g.meta_u64(&k("ple.eos_token_id"))? as i64,
                    head_dim: u("embedding_length_per_layer_input")?,
                    multipliers: g.meta_u64s(&k("ple.layer_multipliers"))?,
                    head_offsets: g.meta_u64s(&k("ple.head_offsets"))?,
                    head_vocab: g.meta_u64s(&k("ple.head_vocab_sizes"))?,
                };
                ensure!(
                    p.ngram >= 2 && p.multipliers.len() >= p.ngram,
                    "PLE n-gram constants"
                );
                ensure!(
                    p.head_offsets.len() >= p.n_heads() && p.head_vocab.len() >= p.n_heads(),
                    "PLE head constants"
                );
                ensure!(
                    p.n_heads() * p.head_dim == n_embd,
                    "PLE heads × head_dim != n_embd"
                );
                Some(p)
            }
            None => None,
        };

        Ok(Self {
            arch: Arch::Qwen4Exp,
            n_embd,
            n_layer,
            n_vocab,
            eps,
            hc: u("hyper_connection.count")?,
            hc_lr: u("hyper_connection.low_rank")?,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: g.meta_f64(&k("rope.freq_base"))? as f32,
            rope_sections,
            idx_heads: u("attention.indexer.head_count")?,
            idx_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            kpool,
            ssm_d_conv: u("ssm.conv_kernel")?,
            ssm_d_state: u("ssm.state_size")?,
            ssm_k_heads: u("ssm.group_count")?,
            ssm_v_heads: u("ssm.time_step_rank")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            expert_weights_scale: g
                .meta
                .get(&k("expert_weights_scale"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            is_recurrent,
            ple,
        })
    }
}

impl Hparams {
    /// `qwen3next` (Qwen3-Next-80B-A3B, Qwen3-Coder-Next): no hyper-connections (`hc` = 1), no
    /// indexer (`idx_heads` = 0: every full-attention layer attends to every earlier cell), no PLE.
    fn from_gguf_qwen3next(g: &Gguf) -> Result<Self> {
        let k = |s: &str| format!("qwen3next.{s}");
        let u = |s: &str| g.meta_u64(&k(s)).map(|v| v as usize);
        let n_layer = u("block_count")?;
        let is_recurrent: Vec<bool> = match g.meta.get(&k("attention.recurrent_layers")) {
            Some(v) => v
                .as_array()
                .context("attention.recurrent_layers is not an array")?
                .iter()
                .take(n_layer)
                .map(|x| x.as_u64().unwrap_or(0) != 0)
                .collect(),
            None => {
                let every = g
                    .meta
                    .get(&k("full_attention_interval"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as usize;
                ensure!(every > 0, "full_attention_interval is 0");
                (0..n_layer).map(|i| (i + 1) % every != 0).collect()
            }
        };
        let hp = Self {
            arch: Arch::Qwen3Next,
            n_embd: u("embedding_length")?,
            n_layer,
            n_vocab: g.info("token_embd.weight")?.dims[1] as usize,
            eps: g.meta_f64(&k("attention.layer_norm_rms_epsilon"))? as f32,
            hc: 1,
            hc_lr: 0,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: g.meta_f64(&k("rope.freq_base"))? as f32,
            rope_sections: [0; 4],
            idx_heads: 0,
            idx_dim: 0,
            idx_top_k: 0,
            kpool: 0,
            ssm_d_conv: u("ssm.conv_kernel")?,
            ssm_d_state: u("ssm.state_size")?,
            ssm_k_heads: u("ssm.group_count")?,
            ssm_v_heads: u("ssm.time_step_rank")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            expert_weights_scale: g
                .meta
                .get(&k("expert_weights_scale"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            is_recurrent,
            ple: None,
        };
        ensure!(
            hp.ssm_v_heads.is_multiple_of(hp.ssm_k_heads),
            "GDN v heads {} not a multiple of k heads {}",
            hp.ssm_v_heads,
            hp.ssm_k_heads
        );
        ensure!(
            hp.n_head.is_multiple_of(hp.n_head_kv) && hp.n_rot <= hp.head_dim,
            "attention head shapes"
        );
        Ok(hp)
    }

    /// The k head GDN value head `vh` reads.
    pub fn gdn_k_head(&self, vh: usize) -> usize {
        match self.arch {
            Arch::Qwen4Exp => vh % self.ssm_k_heads,
            Arch::Qwen3Next => vh / (self.ssm_v_heads / self.ssm_k_heads),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Small numeric pieces
// ---------------------------------------------------------------------------------------------

/// How a matmul's input is rounded before the dot products.
///
/// The truth is [`Act::F32`]. The others reproduce what ggml-cpu does to the *activation* for a
/// weight of a given type (its `vec_dot_type`), so `--llama-numerics` can show how much of the
/// distance to llama.cpp is llama.cpp's own rounding: Q8_0 (blocks of 32, `d = amax/127` stored as
/// f16, round half to even) for Q2_0/Q4_0/Q5_0/Q8_0/IQ4_NL; Q8_K (blocks of 256, `d = max/-127` in
/// f32, round half to even) for the K-quants and the other i-quants; BF16 for BF16 weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    F32,
    Q8_0,
    Q8K,
    Bf16,
    /// The fast kernels' contract (`tang_compute::flash::QAct`): per 32-element chunk
    /// `d = amax / 127` in f32, `q = clamp(round_half_away(x / d), ±127)`.
    Int8,
}

impl Act {
    pub fn for_weight(ty: GgmlType) -> Act {
        use GgmlType::*;
        match ty {
            Q2_0 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Iq4Nl => Act::Q8_0,
            Q2K | Q3K | Q4K | Q5K | Q6K | Iq2Xxs | Iq2Xs | Iq2S | Iq3Xxs | Iq3S | Iq1S | Iq1M
            | Iq4Xs => Act::Q8K,
            Bf16 => Act::Bf16,
            _ => Act::F32,
        }
    }

    /// Round `x` (one or more whole rows) in place to what the dot product would see.
    pub fn round(self, x: &mut [f32]) {
        match self {
            Act::F32 => {}
            Act::Bf16 => x.iter_mut().for_each(|v| *v = round_bf16(*v)),
            Act::Int8 => {
                for b in x.chunks_mut(32) {
                    let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let d = amax / 127.0;
                    for v in b.iter_mut() {
                        *v = if d == 0.0 {
                            0.0
                        } else {
                            (*v / d).round().clamp(-127.0, 127.0) * d
                        };
                    }
                }
            }
            Act::Q8_0 => {
                for b in x.chunks_mut(32) {
                    let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let dh = f16_to_f32(f32_to_f16(d));
                    for v in b.iter_mut() {
                        *v = (*v * id).round_ties_even() * dh;
                    }
                }
            }
            Act::Q8K => {
                for b in x.chunks_mut(256) {
                    let (mut amax, mut max) = (0f32, 0f32);
                    for &v in b.iter() {
                        if v.abs() > amax {
                            amax = v.abs();
                            max = v;
                        }
                    }
                    if amax == 0.0 {
                        b.iter_mut().for_each(|v| *v = 0.0);
                        continue;
                    }
                    let iscale = -127.0 / max;
                    let d = 1.0 / iscale;
                    for v in b.iter_mut() {
                        *v = (*v * iscale).round_ties_even().min(127.0) * d;
                    }
                }
            }
        }
    }
}

/// f32 -> bf16 -> f32, round to nearest even (ggml's `ggml_fp32_to_bf16`).
fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let u = x.to_bits();
    let r = (u + (0x7fff + ((u >> 16) & 1))) & 0xffff_0000;
    f32::from_bits(r)
}

/// f32 -> f16 -> f32.
fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

/// A dequantised `y = W x` matrix: `rows` outputs, each `cols` contiguous inputs.
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    /// Rounding applied to each input row first ([`Act::F32`]: none).
    pub act: Act,
}

impl Mat {
    /// Dequantize `t`, then round-trip it through `fmt` (the dense requant study).
    fn from_info(g: &Gguf, t: &TensorInfo, fmt: Format, act: Act) -> Result<Self> {
        ensure!(t.dims.len() <= 2, "{}: not a matrix ({:?})", t.name, t.dims);
        let cols = t.dims[0] as usize;
        let rows = if t.dims.len() == 2 {
            t.dims[1] as usize
        } else {
            1
        };
        let mut data = g.dequantize(t)?;
        requant::apply(fmt, &mut data, cols);
        Ok(Self {
            rows,
            cols,
            data,
            act,
        })
    }

    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// `Y[t] = W X[t]` for `n` inputs packed `X[t * cols ..]`; output `Y[t * rows ..]`.
    /// Parallel over blocks of output rows unless `serial`; the result doesn't depend on which.
    pub fn apply(&self, x: &[f32], n: usize, serial: bool) -> Vec<f32> {
        assert_eq!(x.len(), n * self.cols, "matmul input");
        let rounded;
        let x = if self.act == Act::F32 {
            x
        } else {
            let mut r = x.to_vec();
            for row in r.chunks_mut(self.cols) {
                self.act.round(row);
            }
            rounded = r;
            &rounded[..]
        };
        // Tile: a block of output rows against a block of inputs, so both stay in cache.
        const RB: usize = 16;
        const TB: usize = 32;
        let work = |r0: usize, out: &mut [f32]| {
            // `out` holds rows r0..r0+RB of every input, stored [input][RB]
            let r1 = (r0 + RB).min(self.rows);
            for t0 in (0..n).step_by(TB) {
                let t1 = (t0 + TB).min(n);
                for r in r0..r1 {
                    let w = self.row(r);
                    for t in t0..t1 {
                        out[t * RB + (r - r0)] = dot(w, &x[t * self.cols..(t + 1) * self.cols]);
                    }
                }
            }
        };
        let blocks = self.rows.div_ceil(RB);
        let mut tmp = vec![0f32; blocks * n * RB];
        if serial {
            for (b, out) in tmp.chunks_mut(n * RB).enumerate() {
                work(b * RB, out);
            }
        } else {
            tmp.par_chunks_mut(n * RB)
                .enumerate()
                .for_each(|(b, out)| work(b * RB, out));
        }
        let mut y = vec![0f32; n * self.rows];
        for (b, out) in tmp.chunks(n * RB).enumerate() {
            let r0 = b * RB;
            let r1 = (r0 + RB).min(self.rows);
            for t in 0..n {
                y[t * self.rows + r0..t * self.rows + r1]
                    .copy_from_slice(&out[t * RB..t * RB + (r1 - r0)]);
            }
        }
        y
    }
}

/// f32 dot product with eight independent accumulators (the order is fixed, so it's
/// deterministic run to run).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ggml's softplus: `log(1 + e^x)`, passed through above 20.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `1 / sqrt(mean(x²) + eps)`, the sum in f64 as ggml does.
fn inv_rms(x: &[f32], eps: f32) -> f32 {
    let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    1.0 / ((s / x.len() as f64) as f32 + eps).sqrt()
}

/// In place: `x = x / rms(x) * w`.
pub(crate) fn rms_norm_mul(x: &mut [f32], w: &[f32], eps: f32) {
    let s = inv_rms(x, eps);
    for (v, g) in x.iter_mut().zip(w) {
        *v = *v * s * g;
    }
}

/// In place: `x / sqrt(Σx² + eps)` — the GDN q/k norm, eps on the *squared* norm.
fn l2_norm(x: &mut [f32], eps: f32) {
    let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let r = 1.0 / ((s as f32) + eps).sqrt();
    for v in x {
        *v *= r;
    }
}

/// NeoX rope on the first `n_rot` dims of one head (pairs `i`, `i + n_rot/2`), at `pos`.
/// The model's rope is interleaved M-RoPE with sections [11, 11, 10, 0]; for text every section
/// gets the same position, so it is exactly this.
pub(crate) fn rope_neox(x: &mut [f32], pos: usize, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos as f32 * base.powf(-2.0 * i as f32 / n_rot as f32);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

/// Intermediate tensors of the last position, as raw little-endian files plus `index.json`.
pub struct Dump {
    dir: PathBuf,
    entries: Vec<serde_json::Value>,
}

impl Dump {
    pub fn new(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            entries: Vec::new(),
        })
    }

    pub(crate) fn f32(&mut self, name: &str, shape: &[usize], v: &[f32]) -> Result<()> {
        ensure!(
            shape.iter().product::<usize>() == v.len(),
            "{name}: shape {shape:?} vs {} values",
            v.len()
        );
        let file = format!("{name}.f32");
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(self.dir.join(&file), bytes)?;
        self.entries
            .push(serde_json::json!({"name": name, "file": file, "dtype": "f32", "shape": shape}));
        Ok(())
    }

    pub(crate) fn u32(&mut self, name: &str, v: &[u32]) -> Result<()> {
        let file = format!("{name}.u32");
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(self.dir.join(&file), bytes)?;
        self.entries.push(
            serde_json::json!({"name": name, "file": file, "dtype": "u32", "shape": [v.len()]}),
        );
        Ok(())
    }

    pub(crate) fn finish(&self, tokens: &[u32], position: usize) -> Result<()> {
        let idx = serde_json::json!({
            "model": "qwen4exp",
            "position": position,
            "tokens": tokens,
            "layout": "little-endian raw arrays; residual streams are [hc=4][n_embd=2560], stream-major",
            "tensors": self.entries,
        });
        std::fs::write(
            self.dir.join("index.json"),
            serde_json::to_string_pretty(&idx)?,
        )?;
        Ok(())
    }
}

/// The reference model: an open GGUF and its geometry. Weights are dequantised as each layer
/// runs and dropped after.
pub struct FlashRef {
    pub g: Gguf,
    pub hp: Hparams,
    /// Print per-layer timings to stderr.
    pub verbose: bool,
    /// Round activations the way ggml-cpu does (see [`Act`]) and keep QSA K/V, Q and the
    /// indexer cache in f16, as llama.cpp's default f16 cache and flash attention do. Not the
    /// truth: a way to measure how much of the distance to llama.cpp is llama.cpp's rounding.
    pub emulate_llama: bool,
    /// Ablation: QSA attends to every earlier cell, skipping the indexer's selection.
    pub qsa_dense: bool,
    /// What the dense weights are requantized to first ([`DensePolicy::F32`]: nothing).
    pub dense: DensePolicy,
    /// The fast kernels' activation contract ([`Act::Int8`]) on every matmul whose weight isn't
    /// kept as bf16/f32.
    pub act_int8: bool,
    /// Hyper-connection weight study: the format the HC matrices named in `hc_parts` (`down`,
    /// `up`, `inject` of the layers, `out` for `output_hc_*`) are round-tripped through, and
    /// whether their GEMVs then take int8 activations (the bf16 contract is f32).
    pub hc_fmt: Format,
    pub hc_parts: Vec<String>,
    /// Per-part formats (`--hc-as down=q5,up=fp8,...`), checked before `hc_fmt`.
    pub hc_mix: Vec<(String, Format)>,
    pub hc_act_int8: bool,
    /// Write the one-layer-ahead expert prediction study here (see [`lookahead`]).
    pub lookahead: Option<PathBuf>,
    /// When set, every output position's full f32 logits are appended here (`[n_out][n_vocab]`).
    pub logits: Option<std::sync::Mutex<Vec<f32>>>,
}

/// Predicted experts kept per method in the lookahead study.
pub const LA_PRED: usize = 32;

/// Layer `l − 1`'s state for predicting layer `l`'s experts.
struct LaPrev {
    x: Vec<f32>,
    r_mid: Vec<f32>,
    r_post: Vec<f32>,
    mean_delta: Vec<f32>,
}

/// Which HC matrix `name` is, for the HC precision study.
fn hc_part(name: &str) -> Option<&'static str> {
    if name.starts_with("output_hc_") {
        return (name.ends_with("down.weight") || name.ends_with("up.weight")).then_some("out");
    }
    if !name.starts_with("blk.") || !name.contains(".hc_") {
        return None;
    }
    if name.ends_with("_down.weight") {
        Some("down")
    } else if name.ends_with("_up.weight") {
        Some("up")
    } else if name.ends_with("_inject.weight") {
        Some("inject")
    } else {
        None
    }
}

/// One position's top-k next-token log-probabilities.
pub struct TopK {
    pub pos: usize,
    pub top: Vec<(u32, f64)>,
}

impl FlashRef {
    pub fn open(path: &Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        let hp = Hparams::from_gguf(&g)?;
        Ok(Self {
            g,
            hp,
            verbose: false,
            emulate_llama: false,
            qsa_dense: false,
            dense: DensePolicy::f32(),
            act_int8: false,
            hc_fmt: Format::Keep,
            hc_mix: Vec::new(),
            hc_parts: ["down", "up", "inject", "out"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            hc_act_int8: false,
            lookahead: None,
            logits: None,
        })
    }

    fn t(&self, l: usize, name: &str) -> String {
        format!("blk.{l}.{name}")
    }

    fn mat(&self, l: usize, name: &str) -> Result<Mat> {
        self.load(&self.t(l, name))
    }

    pub(crate) fn load(&self, name: &str) -> Result<Mat> {
        let t = self.g.info(name)?;
        if let Some(part) = hc_part(name) {
            let act = if self.hc_act_int8 {
                Act::Int8
            } else {
                Act::F32
            };
            if let Some((_, f)) = self.hc_mix.iter().find(|(p, _)| p == part) {
                return Mat::from_info(&self.g, t, *f, act);
            }
            if self.hc_fmt != Format::Keep && self.hc_parts.iter().any(|p| p == part) {
                let act = if self.hc_act_int8 {
                    Act::Int8
                } else {
                    Act::F32
                };
                return Mat::from_info(&self.g, t, self.hc_fmt, act);
            }
        }
        let fmt = if t.dims.len() == 2 {
            self.dense.format(name, t.ty)
        } else {
            Format::Keep
        };
        Mat::from_info(&self.g, t, fmt, self.act_for(t, fmt))
    }

    /// The input rounding a matmul against `t` (stored as `fmt`) gets.
    fn act_for(&self, t: &TensorInfo, fmt: Format) -> Act {
        let float =
            fmt == Format::Keep && matches!(t.ty, GgmlType::Bf16 | GgmlType::F32 | GgmlType::F16);
        if self.act_int8 {
            if float {
                Act::F32
            } else {
                Act::Int8
            }
        } else if self.emulate_llama {
            Act::for_weight(t.ty)
        } else {
            Act::F32
        }
    }

    pub(crate) fn vec(&self, name: &str) -> Result<Vec<f32>> {
        self.g.tensor_f32(name)
    }

    /// Run `tokens` (positions 0..T) and return the top-`k` next-token logprobs for each position
    /// in `out_from..T`. With `dump`, intermediate tensors of the last position are written.
    pub fn forward(
        &self,
        tokens: &[u32],
        out_from: usize,
        k: usize,
        dump: Option<&mut Dump>,
    ) -> Result<Vec<TopK>> {
        Ok(self.forward_res(tokens, out_from, k, dump)?.0)
    }

    /// [`forward`](Self::forward), also returning the final 4-stream residual of every position
    /// (`[T][hc][n_embd]`, before the output hyper-connection read): what the MTP layer reads.
    pub fn forward_res(
        &self,
        tokens: &[u32],
        out_from: usize,
        k: usize,
        mut dump: Option<&mut Dump>,
    ) -> Result<(Vec<TopK>, Vec<f32>)> {
        if self.hp.arch == Arch::Qwen3Next {
            return self.forward_plain(tokens, out_from, k, dump);
        }
        let hp = &self.hp;
        let (n, hc, t_len) = (hp.n_embd, hp.hc, tokens.len());
        ensure!(t_len > 0, "no tokens");
        ensure!(out_from < t_len, "no output positions");
        let last = t_len - 1;
        let start = Instant::now();

        // ---- embedding, copied into every stream
        let embd = self.g.info("token_embd.weight")?;
        let mut r = vec![0f32; t_len * hc * n];
        for (t, &tok) in tokens.iter().enumerate() {
            ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
            let e = self.g.rows(embd, tok as usize, 1)?;
            for c in 0..hc {
                r[(t * hc + c) * n..(t * hc + c + 1) * n].copy_from_slice(&e);
            }
            if t == last {
                if let Some(d) = dump.as_deref_mut() {
                    d.f32("embd", &[n], &e)?;
                }
            }
        }

        // one-layer-ahead expert prediction study: the previous layer's inputs, and the records
        let mut la_prev: Option<LaPrev> = None;
        let mut la_out: Vec<u16> = Vec::new();
        for l in 0..hp.n_layer {
            let lt = Instant::now();
            if hp.ple.as_ref().is_some_and(|p| p.layer == l) {
                self.ple_block(l, tokens, &mut r, dump.as_deref_mut())?;
            }
            let (x, inj) = self.hc_read(&r, t_len, &self.t(l, "hc_attn_"), true)?;
            let y = if hp.is_recurrent[l] {
                self.gdn(l, &x, t_len, dump.as_deref_mut())?
            } else {
                self.qsa(l, &x, t_len, dump.as_deref_mut())?
            };
            hc_write(&mut r, &y, &inj, t_len, hc, n);
            if let Some(d) = dump.as_deref_mut() {
                d.f32(
                    &format!("L{l:02}.mixer_out"),
                    &[n],
                    &y[last * n..(last + 1) * n],
                )?;
                d.f32(
                    &format!("L{l:02}.post_mixer"),
                    &[hc, n],
                    &r[last * hc * n..(last + 1) * hc * n],
                )?;
            }
            let r_mid = self.lookahead.is_some().then(|| r.clone());
            let (x, inj) = self.hc_read(&r, t_len, &self.t(l, "hc_ffn_"), true)?;
            let y = self.moe(l, &x, t_len, dump.as_deref_mut())?;
            hc_write(&mut r, &y, &inj, t_len, hc, n);
            if let Some(r_mid) = r_mid {
                la_out.extend(self.lookahead_layer(l, la_prev.as_ref(), &x, t_len)?);
                let mut mean_delta = vec![0f32; hc * n];
                for t in 0..t_len {
                    for (i, m) in mean_delta.iter_mut().enumerate() {
                        *m += (r[t * hc * n + i] - r_mid[t * hc * n + i]) / t_len as f32;
                    }
                }
                la_prev = Some(LaPrev {
                    x: x.clone(),
                    r_mid,
                    r_post: r.clone(),
                    mean_delta,
                });
            }
            if let Some(d) = dump.as_deref_mut() {
                d.f32(
                    &format!("L{l:02}.moe_out"),
                    &[n],
                    &y[last * n..(last + 1) * n],
                )?;
                d.f32(
                    &format!("L{l:02}.post_moe"),
                    &[hc, n],
                    &r[last * hc * n..(last + 1) * hc * n],
                )?;
            }
            if self.verbose {
                eprintln!(
                    "layer {l:2} {} {:6.1}s (total {:6.1}s)",
                    if hp.is_recurrent[l] { "gdn" } else { "qsa" },
                    lt.elapsed().as_secs_f64(),
                    start.elapsed().as_secs_f64()
                );
            }
        }

        if let Some(p) = &self.lookahead {
            let hdr = format!(
                "{{\"layers\":{},\"tokens\":{t_len},\"k\":{},\"pred\":{LA_PRED},\"methods\":[\"x_l\",\"r_mid\",\"r_mid+mean_moe\",\"r_post\"]}}\n",
                hp.n_layer, hp.n_expert_used
            );
            let mut b = hdr.into_bytes();
            b.extend(la_out.iter().flat_map(|v| v.to_le_bytes()));
            std::fs::write(p, b).with_context(|| format!("writing {}", p.display()))?;
        }

        // ---- head: the final hyper-connection read is the output norm
        let rows: Vec<f32> = r[out_from * hc * n..].to_vec();
        let n_out = t_len - out_from;
        let (x, _) = self.hc_read(&rows, n_out, "output_hc_", false)?;
        let head = self.load("output.weight")?;
        ensure!(head.cols == n, "output.weight is {} wide", head.cols);
        let mut tops = Vec::with_capacity(n_out);
        // a few positions at a time, so T × 248k logits never live at once
        for c0 in (0..n_out).step_by(64) {
            let c1 = (c0 + 64).min(n_out);
            let logits = head.apply(&x[c0 * n..c1 * n], c1 - c0, false);
            if let Some(sink) = &self.logits {
                sink.lock().unwrap().extend_from_slice(&logits);
            }
            for (j, lg) in logits.chunks(head.rows).enumerate() {
                let pos = out_from + c0 + j;
                tops.push(TopK {
                    pos,
                    top: top_logprobs(lg, k),
                });
                if pos == last {
                    if let Some(d) = dump.as_deref_mut() {
                        d.f32("final_x", &[n], &x[(c0 + j) * n..(c0 + j + 1) * n])?;
                        d.f32("logits", &[lg.len()], lg)?;
                    }
                }
            }
        }
        if let Some(d) = dump {
            d.finish(tokens, last)?;
        }
        if self.verbose {
            eprintln!(
                "forward of {t_len} tokens: {:.1}s",
                start.elapsed().as_secs_f64()
            );
        }
        Ok((tops, r))
    }

    /// [`forward_res`](Self::forward_res) for one residual stream with pre-norms (`qwen3next`):
    ///
    /// ```text
    /// r = embed(t)
    /// for l: r += mixer(rmsnorm(r) * attn_norm);  r += moe(rmsnorm(r) * post_attention_norm)
    /// logits = output @ (rmsnorm(r) * output_norm)
    /// ```
    fn forward_plain(
        &self,
        tokens: &[u32],
        out_from: usize,
        k: usize,
        mut dump: Option<&mut Dump>,
    ) -> Result<(Vec<TopK>, Vec<f32>)> {
        let hp = &self.hp;
        let (n, t_len) = (hp.n_embd, tokens.len());
        ensure!(t_len > 0, "no tokens");
        ensure!(out_from < t_len, "no output positions");
        let last = t_len - 1;
        let start = Instant::now();
        let embd = self.g.info("token_embd.weight")?;
        let mut r = vec![0f32; t_len * n];
        for (t, &tok) in tokens.iter().enumerate() {
            ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
            r[t * n..(t + 1) * n].copy_from_slice(&self.g.rows(embd, tok as usize, 1)?);
        }
        if let Some(d) = dump.as_deref_mut() {
            d.f32("embd", &[n], &r[last * n..])?;
        }
        let norm = |r: &[f32], w: &[f32]| -> Vec<f32> {
            let mut x = r.to_vec();
            x.par_chunks_mut(n).for_each(|s| rms_norm_mul(s, w, hp.eps));
            x
        };
        for l in 0..hp.n_layer {
            let lt = Instant::now();
            let x = norm(&r, &self.vec(&self.t(l, "attn_norm.weight"))?);
            let y = if hp.is_recurrent[l] {
                self.gdn(l, &x, t_len, dump.as_deref_mut())?
            } else {
                self.qsa(l, &x, t_len, dump.as_deref_mut())?
            };
            r.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            if let Some(d) = dump.as_deref_mut() {
                d.f32(&format!("L{l:02}.mixer_out"), &[n], &y[last * n..])?;
                d.f32(&format!("L{l:02}.post_mixer"), &[n], &r[last * n..])?;
            }
            let x = norm(&r, &self.vec(&self.t(l, "post_attention_norm.weight"))?);
            let y = self.moe(l, &x, t_len, dump.as_deref_mut())?;
            r.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            if let Some(d) = dump.as_deref_mut() {
                d.f32(&format!("L{l:02}.moe_out"), &[n], &y[last * n..])?;
                d.f32(&format!("L{l:02}.post_moe"), &[n], &r[last * n..])?;
            }
            if self.verbose {
                eprintln!(
                    "layer {l:2} {} {:6.1}s (total {:6.1}s)",
                    if hp.is_recurrent[l] { "gdn" } else { "attn" },
                    lt.elapsed().as_secs_f64(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
        let x = norm(&r[out_from * n..], &self.vec("output_norm.weight")?);
        let head = self.load("output.weight")?;
        ensure!(head.cols == n, "output.weight is {} wide", head.cols);
        let n_out = t_len - out_from;
        let mut tops = Vec::with_capacity(n_out);
        for c0 in (0..n_out).step_by(64) {
            let c1 = (c0 + 64).min(n_out);
            let logits = head.apply(&x[c0 * n..c1 * n], c1 - c0, false);
            if let Some(sink) = &self.logits {
                sink.lock().unwrap().extend_from_slice(&logits);
            }
            for (j, lg) in logits.chunks(head.rows).enumerate() {
                let pos = out_from + c0 + j;
                tops.push(TopK {
                    pos,
                    top: top_logprobs(lg, k),
                });
                if let Some(d) = dump.as_deref_mut() {
                    if pos == last {
                        d.f32("final_x", &[n], &x[(c0 + j) * n..(c0 + j + 1) * n])?;
                        d.f32("logits", &[lg.len()], lg)?;
                    }
                }
            }
        }
        if let Some(d) = dump {
            d.finish(tokens, last)?;
        }
        Ok((tops, r))
    }

    /// One layer's records for the lookahead study: per token, layer `l`'s true top-k (by router
    /// logit), then the top-[`LA_PRED`] that layer `l`'s router gives on four stand-ins built from
    /// layer `l − 1` (all `u16::MAX` for layer 0):
    ///
    /// 0. `x_{l-1}`, layer `l−1`'s MoE input, straight into router `l`;
    /// 1. `hc_read_l(R_mid_{l-1})`, the residual after layer `l−1`'s mixer, before its MoE;
    /// 2. `hc_read_l(R_mid_{l-1} + mean_t ΔR_moe_{l-1})`, plus the sequence-mean MoE write
    ///    standing in for this token's ("quasi-hidden state", arXiv 2603.19289);
    /// 3. `hc_read_l(R_post_{l-1})`, the residual after layer `l−1`'s MoE (available one step
    ///    later: during layer `l`'s mixer).
    fn lookahead_layer(
        &self,
        l: usize,
        prev: Option<&LaPrev>,
        x: &[f32],
        t_len: usize,
    ) -> Result<Vec<u16>> {
        let hp = &self.hp;
        let k = hp.n_expert_used;
        let router = self.mat(l, "ffn_gate_inp.weight")?;
        let rank = |lg: &[f32], m: usize| -> Vec<u16> {
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]).then(a.cmp(&b)));
            idx[..m].iter().map(|&e| e as u16).collect()
        };
        let truth = router.apply(x, t_len, false);
        let ne = hp.n_expert;
        let mut preds: Vec<Vec<f32>> = Vec::new();
        if let Some(p) = prev {
            let prefix = self.t(l, "hc_ffn_");
            preds.push(router.apply(&p.x, t_len, false));
            let (xb, _) = self.hc_read(&p.r_mid, t_len, &prefix, false)?;
            preds.push(router.apply(&xb, t_len, false));
            let w = p.mean_delta.len();
            let rc: Vec<f32> = p
                .r_mid
                .iter()
                .enumerate()
                .map(|(i, v)| v + p.mean_delta[i % w])
                .collect();
            let (xc, _) = self.hc_read(&rc, t_len, &prefix, false)?;
            preds.push(router.apply(&xc, t_len, false));
            let (xd, _) = self.hc_read(&p.r_post, t_len, &prefix, false)?;
            preds.push(router.apply(&xd, t_len, false));
        }
        let mut out = Vec::with_capacity(t_len * (k + 4 * LA_PRED));
        for t in 0..t_len {
            out.extend(rank(&truth[t * ne..(t + 1) * ne], k));
            for m in 0..4 {
                match preds.get(m) {
                    Some(lg) => out.extend(rank(&lg[t * ne..(t + 1) * ne], LA_PRED)),
                    None => out.extend(std::iter::repeat_n(u16::MAX, LA_PRED)),
                }
            }
        }
        Ok(out)
    }

    /// `build_hc_mix`: collapse the 4 streams to the one vector a block reads, and (when `inject`)
    /// the per-stream write-back logits.
    ///
    /// `xn[c] = rmsnorm(R[c]) * w_norm[c]` (w stored as 1 + w); `gate = sigmoid(up @ silu(down @ xn / hc))`;
    /// `x = mean_c(xn[c] * gate[c])`; `inj = inject @ xn`.
    pub(crate) fn hc_read(
        &self,
        r: &[f32],
        t_len: usize,
        prefix: &str,
        inject: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (n, hc, eps) = (self.hp.n_embd, self.hp.hc, self.hp.eps);
        let w_norm = self.vec(&format!("{prefix}norm.weight"))?;
        let down = self.load(&format!("{prefix}down.weight"))?;
        let up = self.load(&format!("{prefix}up.weight"))?;
        ensure!(
            w_norm.len() == hc * n && down.cols == hc * n && up.rows == hc * n,
            "{prefix}: hc shapes"
        );
        let mut xn = r.to_vec();
        xn.par_chunks_mut(n).enumerate().for_each(|(i, s)| {
            let c = i % hc;
            rms_norm_mul(s, &w_norm[c * n..(c + 1) * n], eps);
        });
        let mut lo = down.apply(&xn, t_len, false);
        for v in &mut lo {
            *v = silu(*v / hc as f32);
        }
        let gate = up.apply(&lo, t_len, false);
        let mut x = vec![0f32; t_len * n];
        for t in 0..t_len {
            for c in 0..hc {
                let base = (t * hc + c) * n;
                for i in 0..n {
                    x[t * n + i] += xn[base + i] * sigmoid(gate[base + i]);
                }
            }
            for v in &mut x[t * n..(t + 1) * n] {
                *v /= hc as f32;
            }
        }
        let inj = if inject {
            let w = self.load(&format!("{prefix}inject.weight"))?;
            ensure!(
                w.rows == hc && w.cols == hc * n,
                "{prefix}inject: {}×{}",
                w.rows,
                w.cols
            );
            w.apply(&xn, t_len, false)
        } else {
            Vec::new()
        };
        Ok((x, inj))
    }

    /// Gated DeltaNet over the whole sequence from zero state.
    fn gdn(&self, l: usize, x: &[f32], t_len: usize, dump: Option<&mut Dump>) -> Result<Vec<f32>> {
        let hp = &self.hp;
        let (s, hk, hv, eps) = (hp.ssm_d_state, hp.ssm_k_heads, hp.ssm_v_heads, hp.eps);
        let kd = s * hk;
        let ch = 2 * kd + s * hv;
        let d_conv = hp.ssm_d_conv;
        let w_qkv = self.mat(l, "attn_qkv.weight")?;
        let w_z = self.mat(l, "attn_gate.weight")?;
        let conv = self.vec(&self.t(l, "ssm_conv1d.weight"))?; // [d_conv, ch]: tap i of channel c at c*d_conv + i
        let dt = self.vec(&self.t(l, "ssm_dt.bias"))?;
        let a = self.vec(&self.t(l, "ssm_a"))?; // already -exp(A_log)
        let norm = self.vec(&self.t(l, "ssm_norm.weight"))?;
        ensure!(
            w_qkv.rows == ch && conv.len() == ch * d_conv && norm.len() == s,
            "blk.{l} GDN shapes"
        );

        let qkv = w_qkv.apply(x, t_len, false);
        let z = w_z.apply(x, t_len, false);
        let (alpha, beta) = self.gdn_alpha_beta(l, x, t_len)?;

        // causal conv over [3 zero columns | qkv] (tap 0 is the oldest), SiLU on all channels
        let cols: Vec<Vec<f32>> = (0..ch)
            .into_par_iter()
            .map(|c| {
                let w = &conv[c * d_conv..(c + 1) * d_conv];
                (0..t_len)
                    .map(|t| {
                        let mut acc = 0f32;
                        for (i, wi) in w.iter().enumerate() {
                            let back = d_conv - 1 - i;
                            if t >= back {
                                acc += qkv[(t - back) * ch + c] * wi;
                            }
                        }
                        silu(acc)
                    })
                    .collect()
            })
            .collect();
        let mut h = vec![0f32; t_len * ch];
        for (c, col) in cols.into_iter().enumerate() {
            for (t, v) in col.into_iter().enumerate() {
                h[t * ch + c] = v;
            }
        }
        // q, k L2-normed per head; q also / sqrt(S)
        let qscale = 1.0 / (s as f32).sqrt();
        h.par_chunks_mut(ch).for_each(|row| {
            for head in 0..2 * hk {
                l2_norm(&mut row[head * s..(head + 1) * s], eps);
            }
            for v in &mut row[..kd] {
                *v *= qscale;
            }
        });

        // the recurrence, one value head at a time (heads are independent)
        let results: Vec<(Vec<f32>, f32)> = (0..hv)
            .into_par_iter()
            .map(|vh| {
                let kh = hp.gdn_k_head(vh);
                let mut st = vec![0f32; s * s]; // st[a * s + b]: a = key dim, b = value dim
                let mut o = vec![0f32; t_len * s];
                let mut sk = vec![0f32; s];
                for t in 0..t_len {
                    let row = &h[t * ch..(t + 1) * ch];
                    let q = &row[kh * s..(kh + 1) * s];
                    let k = &row[kd + kh * s..kd + (kh + 1) * s];
                    let v = &row[2 * kd + vh * s..2 * kd + (vh + 1) * s];
                    let g = softplus(alpha[t * hv + vh] + dt[vh]) * a[vh];
                    let decay = g.exp();
                    let b = sigmoid(beta[t * hv + vh]);
                    // decay first, then the delta update, then read the updated state
                    for e in st.iter_mut() {
                        *e *= decay;
                    }
                    sk.iter_mut().for_each(|e| *e = 0.0);
                    for (ai, &ka) in k.iter().enumerate() {
                        for (acc, &sv) in sk.iter_mut().zip(&st[ai * s..(ai + 1) * s]) {
                            *acc += sv * ka;
                        }
                    }
                    let d: Vec<f32> = v.iter().zip(&sk).map(|(&vv, &kk)| (vv - kk) * b).collect();
                    for (ai, &ka) in k.iter().enumerate() {
                        for (sv, &dv) in st[ai * s..(ai + 1) * s].iter_mut().zip(&d) {
                            *sv += ka * dv;
                        }
                    }
                    let ot = &mut o[t * s..(t + 1) * s];
                    for (ai, &qa) in q.iter().enumerate() {
                        for (acc, &sv) in ot.iter_mut().zip(&st[ai * s..(ai + 1) * s]) {
                            *acc += sv * qa;
                        }
                    }
                }
                let norm2: f64 = st.iter().map(|&e| (e as f64) * (e as f64)).sum();
                (o, norm2.sqrt() as f32)
            })
            .collect();

        // y = rmsnorm(o) * ssm_norm * sigmoid(z), per head
        let vd = s * hv;
        let mut y = vec![0f32; t_len * vd];
        for (vh, (o, _)) in results.iter().enumerate() {
            for t in 0..t_len {
                let dst = &mut y[t * vd + vh * s..t * vd + (vh + 1) * s];
                dst.copy_from_slice(&o[t * s..(t + 1) * s]);
                rms_norm_mul(dst, &norm, eps);
                let gate = if hp.arch == Arch::Qwen3Next {
                    silu
                } else {
                    sigmoid
                };
                for (j, v) in dst.iter_mut().enumerate() {
                    *v *= gate(z[t * vd + vh * s + j]);
                }
            }
        }
        if let Some(d) = dump {
            let norms: Vec<f32> = results.iter().map(|(_, nn)| *nn).collect();
            d.f32(&format!("L{l:02}.gdn_state_norm"), &[hv], &norms)?;
        }
        let w_o = self.mat(l, "ssm_out.weight")?;
        Ok(w_o.apply(&y, t_len, false))
    }

    /// The GDN decay-gate and beta projections, `[T][hv]` each (`a`, `b` before softplus and
    /// sigmoid): `ssm_alpha` and `ssm_beta`, or `qwen3next`'s fused `ssm_ba`, whose row for k head
    /// `g` is `[b of v heads g·r .. g·r + r | a of the same]`, `r = hv / hk`.
    fn gdn_alpha_beta(&self, l: usize, x: &[f32], t_len: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let hp = &self.hp;
        if self.g.info(&self.t(l, "ssm_ba.weight")).is_err() {
            let a = self.mat(l, "ssm_alpha.weight")?.apply(x, t_len, false);
            let b = self.mat(l, "ssm_beta.weight")?.apply(x, t_len, false);
            return Ok((a, b));
        }
        let (hk, hv) = (hp.ssm_k_heads, hp.ssm_v_heads);
        let r = hv / hk;
        let ba = self.mat(l, "ssm_ba.weight")?;
        ensure!(ba.rows == 2 * hv, "blk.{l}.ssm_ba has {} rows", ba.rows);
        let y = ba.apply(x, t_len, false);
        let (mut a, mut b) = (vec![0f32; t_len * hv], vec![0f32; t_len * hv]);
        for t in 0..t_len {
            for g in 0..hk {
                for j in 0..r {
                    b[t * hv + g * r + j] = y[t * 2 * hv + g * 2 * r + j];
                    a[t * hv + g * r + j] = y[t * 2 * hv + g * 2 * r + r + j];
                }
            }
        }
        Ok((a, b))
    }

    /// QSA: gated GQA over the cells the indexer selects (all of them until a token sees more
    /// than `top_k / kpool` complete blocks), causal, positions 0..T.
    fn qsa(&self, l: usize, x: &[f32], t_len: usize, dump: Option<&mut Dump>) -> Result<Vec<f32>> {
        let hp = &self.hp;
        let (nh, nkv, hd, eps) = (hp.n_head, hp.n_head_kv, hp.head_dim, hp.eps);
        let (ih, idd, kp) = (hp.idx_heads, hp.idx_dim, hp.kpool);
        let (n_rot, base) = (hp.n_rot, hp.rope_base);
        let q_full = self.mat(l, "attn_q.weight")?.apply(x, t_len, false); // [T][nh][q hd | gate hd]
        let mut kc = self.mat(l, "attn_k.weight")?.apply(x, t_len, false); // [T][nkv][hd]
        let vc = self.mat(l, "attn_v.weight")?.apply(x, t_len, false);
        let qn = self.vec(&self.t(l, "attn_q_norm.weight"))?;
        let kn = self.vec(&self.t(l, "attn_k_norm.weight"))?;
        ensure!(
            q_full.len() == t_len * nh * 2 * hd && kc.len() == t_len * nkv * hd,
            "blk.{l} QSA shapes"
        );
        // no indexer (qwen3next): plain causal gated attention over every earlier cell
        let has_idx = ih > 0;
        let (mut k_raw, mut qi, iqn, ikn) = if has_idx {
            (
                self.mat(l, "indexer.k_proj.weight")?.apply(x, t_len, false), // [T][idd]
                self.mat(l, "indexer.q_proj.weight")?.apply(x, t_len, false), // [T][ih][idd]
                self.vec(&self.t(l, "indexer.q_norm.weight"))?,
                self.vec(&self.t(l, "indexer.k_norm.weight"))?,
            )
        } else {
            Default::default()
        };
        ensure!(
            qi.len() == t_len * ih * idd && k_raw.len() == t_len * idd,
            "blk.{l} indexer shapes"
        );

        // q is the first half of each head's [q | gate]; q, k: rmsnorm * w then rope
        let mut q = vec![0f32; t_len * nh * hd];
        let mut gate = vec![0f32; t_len * nh * hd];
        for t in 0..t_len {
            for h in 0..nh {
                let src = &q_full[(t * nh + h) * 2 * hd..(t * nh + h + 1) * 2 * hd];
                let dq = &mut q[(t * nh + h) * hd..(t * nh + h + 1) * hd];
                dq.copy_from_slice(&src[..hd]);
                rms_norm_mul(dq, &qn, eps);
                rope_neox(dq, t, n_rot, base);
                gate[(t * nh + h) * hd..(t * nh + h + 1) * hd].copy_from_slice(&src[hd..]);
            }
            for g in 0..nkv {
                let dk = &mut kc[(t * nkv + g) * hd..(t * nkv + g + 1) * hd];
                rms_norm_mul(dk, &kn, eps);
                rope_neox(dk, t, n_rot, base);
            }
            for h in 0..ih {
                let d = &mut qi[(t * ih + h) * idd..(t * ih + h + 1) * idd];
                rms_norm_mul(d, &iqn, eps);
                rope_neox(d, t, n_rot, base);
            }
        }
        let mut vc = vc;
        if self.emulate_llama {
            // llama.cpp's default caches are f16 (QSA K/V and the indexer's raw|pooled rows), and
            // CPU flash attention converts Q to the K type
            for buf in [&mut q, &mut kc, &mut vc, &mut k_raw] {
                buf.iter_mut().for_each(|v| *v = round_f16(*v));
            }
        }
        // pooled indexer keys: mean of the block's raw keys, *then* norm, then rope at its first cell
        let n_blocks = if has_idx { t_len / kp } else { 0 };
        let mut pooled = vec![0f32; n_blocks * idd];
        for b in 0..n_blocks {
            let p = &mut pooled[b * idd..(b + 1) * idd];
            for j in 0..kp {
                for (o, v) in p
                    .iter_mut()
                    .zip(&k_raw[(b * kp + j) * idd..(b * kp + j + 1) * idd])
                {
                    *o += v;
                }
            }
            for o in p.iter_mut() {
                *o /= kp as f32;
            }
            rms_norm_mul(p, &ikn, eps);
            rope_neox(p, b * kp, n_rot, base);
            if self.emulate_llama {
                p.iter_mut().for_each(|v| *v = round_f16(*v));
            }
        }

        let max_blocks = if has_idx { hp.idx_top_k / kp } else { 0 };
        let iscale = 1.0 / (idd as f32).sqrt();
        let kq_scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let last = t_len - 1;
        let per_token: Vec<(Vec<f32>, Option<Vec<u32>>)> = (0..t_len)
            .into_par_iter()
            .map(|t| {
                // the blocks complete at this token, then the tail cells after them (itself included)
                let nv = if has_idx { (t + 1) / kp } else { 0 };
                let mut blocks: Vec<usize> = (0..nv).collect();
                if nv > max_blocks && !self.qsa_dense {
                    // score = Σ_heads relu(q_h · pooled_b) / sqrt(idx_dim)
                    let qt = &qi[t * ih * idd..(t + 1) * ih * idd];
                    let scores: Vec<f32> = (0..nv)
                        .map(|b| {
                            let kb = &pooled[b * idd..(b + 1) * idd];
                            (0..ih)
                                .map(|h| dot(&qt[h * idd..(h + 1) * idd], kb).max(0.0) * iscale)
                                .sum()
                        })
                        .collect();
                    // highest first; ties to the lower block (ggml's top_k leaves ties unspecified)
                    blocks.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                    blocks.truncate(max_blocks);
                    blocks.sort_unstable();
                }
                let mut cells: Vec<usize> =
                    blocks.iter().flat_map(|&b| b * kp..(b + 1) * kp).collect();
                cells.extend(nv * kp..=t);
                let mut out = vec![0f32; nh * hd];
                let mut w = vec![0f32; cells.len()];
                for h in 0..nh {
                    let g = h / group; // kv head = q head / 12
                    let qh = &q[(t * nh + h) * hd..(t * nh + h + 1) * hd];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, &c) in cells.iter().enumerate() {
                        w[j] = dot(qh, &kc[(c * nkv + g) * hd..(c * nkv + g + 1) * hd]) * kq_scale;
                        mx = mx.max(w[j]);
                    }
                    let mut sum = 0f64;
                    for v in w.iter_mut() {
                        *v = (*v - mx).exp();
                        sum += *v as f64;
                    }
                    let oh = &mut out[h * hd..(h + 1) * hd];
                    for (j, &c) in cells.iter().enumerate() {
                        let p = (w[j] as f64 / sum) as f32;
                        for (o, v) in oh
                            .iter_mut()
                            .zip(&vc[(c * nkv + g) * hd..(c * nkv + g + 1) * hd])
                        {
                            *o += p * v;
                        }
                    }
                    let gh = &gate[(t * nh + h) * hd..(t * nh + h + 1) * hd];
                    for (o, gv) in oh.iter_mut().zip(gh) {
                        *o *= sigmoid(*gv);
                    }
                }
                let sel = (t == last).then(|| cells.iter().map(|&c| c as u32).collect());
                (out, sel)
            })
            .collect();
        let mut attn = vec![0f32; t_len * nh * hd];
        let mut sel_last = None;
        for (t, (o, sel)) in per_token.into_iter().enumerate() {
            attn[t * nh * hd..(t + 1) * nh * hd].copy_from_slice(&o);
            if sel.is_some() {
                sel_last = sel;
            }
        }
        if let Some(d) = dump.filter(|_| has_idx) {
            d.u32(
                &format!("L{l:02}.qsa_selected"),
                &sel_last.unwrap_or_default(),
            )?;
            // the last token's indexer score of every complete block, selected or not
            let nv = t_len / kp;
            let qt = &qi[last * ih * idd..(last + 1) * ih * idd];
            let scores: Vec<f32> = (0..nv)
                .map(|b| {
                    let kb = &pooled[b * idd..(b + 1) * idd];
                    (0..ih)
                        .map(|h| dot(&qt[h * idd..(h + 1) * idd], kb).max(0.0) * iscale)
                        .sum()
                })
                .collect();
            d.f32(&format!("L{l:02}.qsa_scores"), &[nv], &scores)?;
            d.f32(&format!("L{l:02}.qsa_q_idx"), &[ih, idd], qt)?;
            if nv > 0 {
                d.f32(
                    &format!("L{l:02}.qsa_pooled"),
                    &[nv, idd],
                    &pooled[..nv * idd],
                )?;
            }
        }
        let w_o = self.mat(l, "attn_output.weight")?;
        Ok(w_o.apply(&attn, t_len, false))
    }

    /// MoE: softmax over all experts, top-k, renormalised with the 2^-14 clamp; plus the
    /// sigmoid-gated shared expert.
    pub(crate) fn moe(
        &self,
        l: usize,
        x: &[f32],
        t_len: usize,
        dump: Option<&mut Dump>,
    ) -> Result<Vec<f32>> {
        let hp = &self.hp;
        let (n, ne, k) = (hp.n_embd, hp.n_expert, hp.n_expert_used);
        let router = self.mat(l, "ffn_gate_inp.weight")?;
        ensure!(
            router.rows == ne && router.cols == n,
            "blk.{l} router shape"
        );
        let logits = router.apply(x, t_len, false);

        // routing: per token, (expert, weight) in rank order
        let mut routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let lg = &logits[t * ne..(t + 1) * ne];
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let ex: Vec<f64> = lg.iter().map(|&v| (v as f64 - mx).exp()).collect();
            let sum: f64 = ex.iter().sum();
            let p: Vec<f64> = ex.iter().map(|v| v / sum).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
            idx.truncate(k);
            let wsum: f64 = idx.iter().map(|&e| p[e]).sum::<f64>().max(6.103515625e-5);
            let mut r: Vec<(usize, f32)> = idx.iter().map(|&e| (e, (p[e] / wsum) as f32)).collect();
            if hp.expert_weights_scale != 0.0 && hp.expert_weights_scale != 1.0 {
                for e in &mut r {
                    e.1 *= hp.expert_weights_scale;
                }
            }
            routes.push(r);
        }

        // group tokens by expert: expert -> [(token, rank)]
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
        for (t, r) in routes.iter().enumerate() {
            for (rank, &(e, _)) in r.iter().enumerate() {
                by_expert[e].push((t, rank));
            }
        }
        let gate_t = self.g.info(&self.t(l, "ffn_gate_exps.weight"))?;
        let up_t = self.g.info(&self.t(l, "ffn_up_exps.weight"))?;
        let down_t = self.g.info(&self.t(l, "ffn_down_exps.weight"))?;
        let ff = hp.n_ff_exp;
        ensure!(
            gate_t.dims == [n as u64, ff as u64, ne as u64]
                && down_t.dims == [ff as u64, n as u64, ne as u64],
            "blk.{l} expert shapes"
        );

        // each used expert: dequantise it, run its tokens, hand back (token, rank, y)
        type ExpertOut = Vec<(usize, usize, Vec<f32>)>;
        let used: Vec<usize> = (0..ne).filter(|&e| !by_expert[e].is_empty()).collect();
        let outs: Vec<Result<ExpertOut>> = used
            .par_iter()
            .map(|&e| {
                let toks = &by_expert[e];
                let xe: Vec<f32> = toks
                    .iter()
                    .flat_map(|&(t, _)| x[t * n..(t + 1) * n].iter().copied())
                    .collect();
                let m = toks.len();
                let gate = Mat {
                    rows: ff,
                    cols: n,
                    data: self.g.expert(gate_t, e)?,
                    act: self.act_for(gate_t, Format::Keep),
                };
                let up = Mat {
                    rows: ff,
                    cols: n,
                    data: self.g.expert(up_t, e)?,
                    act: self.act_for(up_t, Format::Keep),
                };
                let down = Mat {
                    rows: n,
                    cols: ff,
                    data: self.g.expert(down_t, e)?,
                    act: self.act_for(down_t, Format::Keep),
                };
                let gv = gate.apply(&xe, m, true);
                let uv = up.apply(&xe, m, true);
                let hv: Vec<f32> = gv.iter().zip(&uv).map(|(&g, &u)| silu(g) * u).collect();
                let y = down.apply(&hv, m, true);
                Ok(toks
                    .iter()
                    .enumerate()
                    .map(|(i, &(t, rank))| (t, rank, y[i * n..(i + 1) * n].to_vec()))
                    .collect())
            })
            .collect();
        let mut slots: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); k]; t_len];
        for o in outs {
            for (t, rank, y) in o? {
                slots[t][rank] = y;
            }
        }
        // shared expert, gated per token by sigmoid(w · x)
        let sg_t = self.g.info(&self.t(l, "ffn_gate_inp_shexp.weight"))?;
        let sg = self.g.dequantize(sg_t)?;
        let sh_gate = self.mat(l, "ffn_gate_shexp.weight")?.apply(x, t_len, false);
        let sh_up = self.mat(l, "ffn_up_shexp.weight")?.apply(x, t_len, false);
        let sh_h: Vec<f32> = sh_gate
            .iter()
            .zip(&sh_up)
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        let sh = self
            .mat(l, "ffn_down_shexp.weight")?
            .apply(&sh_h, t_len, false);

        // combine in rank order, so the sum is the same however the experts were scheduled
        let mut y = vec![0f32; t_len * n];
        for t in 0..t_len {
            let yt = &mut y[t * n..(t + 1) * n];
            for (rank, &(_, w)) in routes[t].iter().enumerate() {
                for (o, v) in yt.iter_mut().zip(&slots[t][rank]) {
                    *o += w * v;
                }
            }
            let mut xt = x[t * n..(t + 1) * n].to_vec();
            self.act_for(sg_t, Format::Keep).round(&mut xt);
            let gsh = sigmoid(dot(&sg, &xt));
            for (o, v) in yt.iter_mut().zip(&sh[t * n..(t + 1) * n]) {
                *o += gsh * v;
            }
        }
        if let Some(d) = dump {
            let last = &routes[t_len - 1];
            d.u32(
                &format!("L{l:02}.router_ids"),
                &last.iter().map(|&(e, _)| e as u32).collect::<Vec<_>>(),
            )?;
            d.f32(
                &format!("L{l:02}.router_w"),
                &[k],
                &last.iter().map(|&(_, w)| w).collect::<Vec<_>>(),
            )?;
        }
        Ok(y)
    }

    /// The n-gram embedding block (`build_ple`), applied to the residual stack before layer `l`'s
    /// mixer read.
    ///
    /// `key = gnorm(ple_key @ e)`, `q = gnorm(R)` (per stream), `s[c] = <key[c], q[c]> / sqrt(n)`,
    /// `gate[c] = sigmoid(sign(s) sqrt(max(|s|, 1e-6)))`, `gated[c] = (ple_value @ e) * gate[c]`;
    /// `R += gated + silu(conv(gnorm(gated)))`, the conv depthwise, 4 taps, dilation 3, causal.
    fn ple_block(
        &self,
        l: usize,
        tokens: &[u32],
        r: &mut [f32],
        dump: Option<&mut Dump>,
    ) -> Result<()> {
        let hp = &self.hp;
        let p = hp.ple.as_ref().context("no PLE constants")?;
        let (n, hc, eps, t_len) = (hp.n_embd, hp.hc, hp.eps, tokens.len());
        let hcd = n * hc;
        let table = self.g.info("per_layer_token_embd.weight")?;
        ensure!(
            table.row_len() == p.head_dim,
            "PLE table rows are {} wide",
            table.row_len()
        );

        // gather 16 rows per token, flattened head-slowest
        let mut emb = vec![0f32; t_len * n];
        let mut all_rows = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let rows = p.rows(tokens, t);
            for (h, &row) in rows.iter().enumerate() {
                self.g.read_row(
                    table,
                    row as usize,
                    &mut emb[t * n + h * p.head_dim..t * n + (h + 1) * p.head_dim],
                )?;
            }
            all_rows.push(rows);
        }
        let key = self.mat(l, "ple_key.weight")?.apply(&emb, t_len, false); // [T][hc*n]
        let value = self.mat(l, "ple_value.weight")?.apply(&emb, t_len, false); // [T][n]
        let nk = self.vec(&self.t(l, "ple_norm_key.weight"))?;
        let nq = self.vec(&self.t(l, "ple_norm_query.weight"))?;
        let nc = self.vec(&self.t(l, "ple_norm_conv.weight"))?;
        let conv = self.vec(&self.t(l, "ple_conv1d.weight"))?; // [kern, hc*n]: tap k of channel c at k + kern*c
        let kern = p.conv_kernel;
        let dil = p.ngram;
        ensure!(
            key.len() == t_len * hcd && value.len() == t_len * n && conv.len() == kern * hcd,
            "PLE shapes"
        );

        let grouped = |v: &mut [f32], w: &[f32]| {
            for c in 0..hc {
                rms_norm_mul(&mut v[c * n..(c + 1) * n], &w[c * n..(c + 1) * n], eps);
            }
        };
        let mut gated = vec![0f32; t_len * hcd];
        let mut normalized = vec![0f32; t_len * hcd];
        let mut gates = vec![0f32; t_len * hc];
        for t in 0..t_len {
            let mut k = key[t * hcd..(t + 1) * hcd].to_vec();
            grouped(&mut k, &nk);
            let mut q = r[t * hcd..(t + 1) * hcd].to_vec();
            grouped(&mut q, &nq);
            for c in 0..hc {
                let s = dot(&k[c * n..(c + 1) * n], &q[c * n..(c + 1) * n]) / (n as f32).sqrt();
                // ggml_sgn(0) is 0, so a zero score gates at exactly one half
                let sgn = if s > 0.0 {
                    1.0
                } else if s < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let gte = sigmoid(sgn * s.abs().max(1e-6).sqrt());
                gates[t * hc + c] = gte;
                for i in 0..n {
                    gated[t * hcd + c * n + i] = value[t * n + i] * gte;
                }
            }
            let nm = &mut normalized[t * hcd..(t + 1) * hcd];
            nm.copy_from_slice(&gated[t * hcd..(t + 1) * hcd]);
            grouped(nm, &nc);
        }
        // causal depthwise conv over the normalized rows, tap k reaching (kern-1-k)*dil back, zero history
        for t in 0..t_len {
            for ch in 0..hcd {
                let mut acc = 0f32;
                for kk in 0..kern {
                    let back = (kern - 1 - kk) * dil;
                    if t >= back {
                        acc += conv[kk + kern * ch] * normalized[(t - back) * hcd + ch];
                    }
                }
                r[t * hcd + ch] += gated[t * hcd + ch] + silu(acc);
            }
        }
        if let Some(d) = dump {
            let last = t_len - 1;
            d.u32(
                "ple.rows",
                &all_rows[last].iter().map(|&x| x as u32).collect::<Vec<_>>(),
            )?;
            d.f32("ple.emb", &[n], &emb[last * n..(last + 1) * n])?;
            d.f32("ple.gate", &[hc], &gates[last * hc..(last + 1) * hc])?;
            d.f32("ple.out", &[hc, n], &r[last * hcd..(last + 1) * hcd])?;
        }
        Ok(())
    }
}

/// `build_hc_combine`: every stream adds the block output, scaled by `2 * sigmoid(inj[c] / hc)`.
pub(crate) fn hc_write(r: &mut [f32], y: &[f32], inj: &[f32], t_len: usize, hc: usize, n: usize) {
    for t in 0..t_len {
        for c in 0..hc {
            let w = 2.0 * sigmoid(inj[t * hc + c] / hc as f32);
            let dst = &mut r[(t * hc + c) * n..(t * hc + c + 1) * n];
            for (o, v) in dst.iter_mut().zip(&y[t * n..(t + 1) * n]) {
                *o += v * w;
            }
        }
    }
}

/// The `k` largest log-softmax entries, f64, descending (ties to the lower id).
pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = mx
        + logits
            .iter()
            .map(|&v| (v as f64 - mx).exp())
            .sum::<f64>()
            .ln();
    let k = k.clamp(1, logits.len());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let cmp = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
    idx.select_nth_unstable_by(k - 1, cmp);
    idx.truncate(k);
    idx.sort_by(cmp);
    idx.into_iter()
        .map(|i| (i as u32, logits[i] as f64 - lse))
        .collect()
}

/// `tang-llm flash-ref <gguf-first-shard> <token ids...> [--ids-file F] [--last N] [--top K]
/// [--dump DIR] [--llama-numerics] [-v]`: run the reference and print one JSON object per output
/// position, `{"pos": i, "top": [[id, logprob], ...]}` (the distribution of the token *after*
/// position i). `--llama-numerics` rounds activations and the QSA caches the way llama.cpp's CPU
/// backend does (see [`Act`]); without it everything is f32. `--qsa-dense` is an ablation: QSA
/// attends to every cell, as if the indexer selected everything. `--dense-as` requantizes dense
/// weights first (see [`requant`]); `--act-int8` applies the fast kernels' int8 activation
/// contract to every matmul whose weight isn't bf16/f32.
pub fn cli(args: &[String]) -> Result<()> {
    let usage =
        "usage: flash-ref <gguf> <ids...> [--ids-file F] [--last N] [--top K] [--dump DIR] [--logits FILE] [-v]";
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context(usage)?);
    let mut ids: Vec<u32> = Vec::new();
    let mut last: Option<usize> = None;
    let mut top = 10;
    let mut dump_dir: Option<PathBuf> = None;
    let mut verbose = false;
    let mut llama_numerics = false;
    let mut qsa_dense = false;
    let mut act_int8 = false;
    let mut hc_fmt = Format::Keep;
    let mut hc_mix = Vec::new();
    let mut hc_parts: Vec<String> = ["down", "up", "inject", "out"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut hc_act_int8 = false;
    let mut lookahead = None;
    let mut dense = DensePolicy::f32();
    let mut logits_out: Option<PathBuf> = None;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--last" => last = Some(it.next().context("--last N")?.parse().context("--last N")?),
            "--top" => top = it.next().context("--top K")?.parse().context("--top K")?,
            "--dump" => dump_dir = Some(PathBuf::from(it.next().context("--dump DIR")?)),
            "--ids-file" => {
                let f = it.next().context("--ids-file F")?;
                let s = std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?;
                for w in s.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']') {
                    if !w.is_empty() {
                        ids.push(
                            w.parse()
                                .with_context(|| format!("{f}: bad token id {w:?}"))?,
                        );
                    }
                }
            }
            "-v" | "--verbose" => verbose = true,
            "--llama-numerics" => llama_numerics = true,
            "--qsa-dense" => qsa_dense = true,
            "--act-int8" => act_int8 = true,
            "--hc-as" => {
                let v = it.next().context("--hc-as FORMAT | part=format,...")?;
                if v.contains('=') {
                    for kv in v.split(',') {
                        let (p, f) = kv.split_once('=').context("--hc-as part=format")?;
                        hc_mix.push((p.to_string(), Format::parse_hc(f)?));
                    }
                } else {
                    hc_fmt = Format::parse_hc(v)?;
                }
            }
            "--hc-parts" => {
                hc_parts = it
                    .next()
                    .context("--hc-parts down,up,inject,out")?
                    .split(',')
                    .map(String::from)
                    .collect()
            }
            "--hc-act-int8" => hc_act_int8 = true,
            "--lookahead" => {
                lookahead = Some(PathBuf::from(it.next().context("--lookahead FILE")?))
            }
            "--dense-as" => dense = DensePolicy::parse(it.next().context("--dense-as POLICY")?)?,
            "--logits" => logits_out = Some(PathBuf::from(it.next().context("--logits FILE")?)),
            s => ids.push(
                s.parse()
                    .with_context(|| format!("bad token id {s:?}; {usage}"))?,
            ),
        }
    }
    if ids.is_empty() {
        bail!("no token ids; {usage}");
    }
    let mut m = FlashRef::open(&path)?;
    m.verbose = verbose;
    m.emulate_llama = llama_numerics;
    m.qsa_dense = qsa_dense;
    m.act_int8 = act_int8;
    m.hc_fmt = hc_fmt;
    m.hc_mix = hc_mix;
    m.hc_parts = hc_parts;
    m.hc_act_int8 = hc_act_int8;
    m.lookahead = lookahead;
    m.dense = dense;
    m.logits = logits_out.as_ref().map(|_| Default::default());
    let from = ids.len() - last.unwrap_or(ids.len()).clamp(1, ids.len());
    let mut dump = dump_dir.as_deref().map(Dump::new).transpose()?;
    let tops = m.forward(&ids, from, top, dump.as_mut())?;
    if let (Some(p), Some(l)) = (&logits_out, &m.logits) {
        let l = l.lock().unwrap();
        std::fs::write(
            p,
            l.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .with_context(|| format!("writing {}", p.display()))?;
    }
    let mut out = String::new();
    for t in tops {
        let top: Vec<String> = t
            .top
            .iter()
            .map(|(i, lp)| format!("[{i},{lp:.6}]"))
            .collect();
        writeln!(out, "{{\"pos\":{},\"top\":[{}]}}", t.pos, top.join(","))?;
    }
    print!("{out}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ple_params() -> PleParams {
        PleParams {
            layer: 1,
            ngram: 3,
            heads_per_ngram: 8,
            conv_kernel: 4,
            eos: 248044,
            head_dim: 160,
            multipliers: vec![23703573157769, 20109073645365, 8052911324071],
            head_offsets: (0..16).map(|h| h * 20_000_100).collect(),
            head_vocab: (0..16).map(|h| 20_000_003 + 10 * h).collect(),
        }
    }

    #[test]
    fn ple_hash_window() {
        let p = ple_params();
        // first token: both predecessors missing -> EOS
        let r0 = p.rows(&[5], 0);
        let mixed2 = 5u64.wrapping_mul(p.multipliers[0]) ^ 248044u64.wrapping_mul(p.multipliers[1]);
        assert_eq!(r0[0], mixed2 % p.head_vocab[0]);
        // an EOS predecessor cuts everything older; the token's own EOS doesn't
        let a = p.rows(&[9, 248044, 7], 2);
        let b = p.rows(&[1, 248044, 7], 2);
        assert_eq!(a, b, "older context is cut at the EOS");
        let c = p.rows(&[3, 4, 248044], 2);
        let d = p.rows(&[3, 5, 248044], 2);
        assert_ne!(c, d, "own EOS keeps its context");
        // token id 0 is a real token, not a missing one
        assert_ne!(p.rows(&[0, 7], 1), p.rows(&[7], 0));
        // XOR, not sum: equal products cancel
        let mut q = p.clone();
        q.multipliers = vec![3, 3, 3];
        let r = q.rows(&[11, 11], 1);
        assert_eq!(r[0], q.head_offsets[0]);
    }

    #[test]
    fn rope_is_a_rotation() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
        let n0: f32 = x[..64].iter().map(|v| v * v).sum();
        let tail = x[64..].to_vec();
        rope_neox(&mut x, 1234, 64, 1e7);
        let n1: f32 = x[..64].iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4);
        assert_eq!(&x[64..], &tail[..]);
        let mut y = vec![1.0f32; 64];
        rope_neox(&mut y, 0, 64, 1e7);
        assert_eq!(y, vec![1.0f32; 64]);
    }

    #[test]
    fn l2_norm_eps_on_squared_norm() {
        let mut x = vec![3.0f32, 4.0];
        l2_norm(&mut x, 0.0);
        assert!((x[0] - 0.6).abs() < 1e-6 && (x[1] - 0.8).abs() < 1e-6);
        let mut y = vec![0.0f32; 4];
        y[0] = 1e-4;
        l2_norm(&mut y, 1e-6);
        // sqrt(1e-8 + 1e-6), not sqrt(mean + eps)
        assert!((y[0] - 1e-4 / (1e-8f32 + 1e-6).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn matmul_tiles_match_naive() {
        let rows = 37;
        let cols = 29;
        let n = 70;
        let w = Mat {
            rows,
            cols,
            data: (0..rows * cols)
                .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
                .collect(),
            act: Act::F32,
        };
        let x: Vec<f32> = (0..n * cols)
            .map(|i| ((i * 104729) % 89) as f32 / 44.0 - 1.0)
            .collect();
        let y = w.apply(&x, n, false);
        let ys = w.apply(&x, n, true);
        for t in 0..n {
            for r in 0..rows {
                let want: f32 = (0..cols)
                    .map(|c| w.data[r * cols + c] * x[t * cols + c])
                    .sum();
                assert!((y[t * rows + r] - want).abs() < 1e-4);
                assert_eq!(y[t * rows + r], ys[t * rows + r]);
            }
        }
    }

    #[test]
    fn activation_rounding() {
        // Q8_0: the block's largest magnitude lands on ±127 steps of an f16 scale
        let mut x: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.013).collect();
        let orig = x.clone();
        Act::Q8_0.round(&mut x);
        for (b, ob) in x.chunks(32).zip(orig.chunks(32)) {
            let amax = ob.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = f16_to_f32(f32_to_f16(amax / 127.0));
            for (v, o) in b.iter().zip(ob) {
                let q = v / d;
                assert!((q - q.round()).abs() < 1e-3 && q.abs() <= 127.5, "{v} {o}");
                assert!((v - o).abs() <= d * 0.5 + amax * 1e-3);
            }
        }
        // Q8_K: scale from the signed max, so it maps to exactly -127
        let mut y: Vec<f32> = (0..256)
            .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
            .collect();
        y[7] = -3.0;
        Act::Q8K.round(&mut y);
        assert_eq!(y[7], -3.0);
        // BF16: round to nearest even on the 16 dropped bits
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0); // tie -> even
        assert_eq!(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert_eq!(round_bf16(1.0 + 1.0 / 128.0), 1.0 + 1.0 / 128.0);
    }

    #[test]
    fn top_logprobs_sorted_and_normalised() {
        let lg = vec![0.0f32, 2.0, 1.0, 2.0, -1.0];
        let t = top_logprobs(&lg, 3);
        assert_eq!(t.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 3, 2]);
        let z: f64 = lg.iter().map(|&v| (v as f64).exp()).sum();
        assert!((t[0].1 - (2.0f64 - z.ln())).abs() < 1e-9);
    }

    fn gdn_hparams(arch: Arch, hk: usize, hv: usize) -> Hparams {
        Hparams {
            arch,
            n_embd: 0,
            n_layer: 0,
            n_vocab: 0,
            eps: 1e-6,
            hc: 1,
            hc_lr: 0,
            n_head: 0,
            n_head_kv: 0,
            head_dim: 0,
            n_rot: 0,
            rope_base: 0.0,
            rope_sections: [0; 4],
            idx_heads: 0,
            idx_dim: 0,
            idx_top_k: 0,
            kpool: 0,
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_k_heads: hk,
            ssm_v_heads: hv,
            n_expert: 0,
            n_expert_used: 0,
            n_ff_exp: 0,
            expert_weights_scale: 0.0,
            is_recurrent: Vec::new(),
            ple: None,
        }
    }

    /// qwen4exp tiles v heads over k heads (`h % hk`); qwen3next groups them (`h / r`). Mixing
    /// the two up is silent and fatal (the tiny-model check gives KL 31 with the wrong one).
    #[test]
    fn gdn_v_to_k_head_order() {
        let flash = gdn_hparams(Arch::Qwen4Exp, 16, 48);
        assert_eq!(
            [0, 15, 16, 17, 47].map(|h| flash.gdn_k_head(h)),
            [0, 15, 0, 1, 15]
        );
        let next = gdn_hparams(Arch::Qwen3Next, 16, 32);
        assert_eq!(
            [0, 1, 2, 30, 31].map(|h| next.gdn_k_head(h)),
            [0, 0, 1, 15, 15]
        );
    }
}
