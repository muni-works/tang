//! Inkling 2 Mini (GGUF architecture `inkling2_loop`): Thinking Machines Lab's looped
//! transformer. Geometry from GGUF metadata, a tensor inventory, and the CPU f32 reference
//! forward ([`forward`]) the fast path is checked against.
//!
//! What makes it different from every other arch here:
//!
//! - **Looped core.** 24 physical layers, but the middle 12 (`loop.start..loop.end`) run
//!   `loop.count` times, so a token passes 48 layers. Each repeat shares the weights and mixes
//!   the previous pass's activations in through four low-rank "control" gates (Q, K, V, FFN).
//! - **No RoPE.** Positions come from a learned relative bias table (`attn_rel_bias`,
//!   `[relative_length, extent]`) gathered by the distance between the query and each KV cell,
//!   plus a log-scaled multiplier on Q and R for the full-attention layers.
//! - **Hybrid attention.** Most layers are sliding-window (1024); every sixth layer is global.
//!   Global layers have separate Q/K/V input norms and input-dependent conv taps.
//! - **Causal short convolutions** (kernel 4) on K, V, the attention output and the FFN output,
//!   each with its own recurrent state.
//! - **Block attention residuals.** Every `attn_res.block_size` (6) layers start a new residual
//!   stream; the others mix all earlier block starts with a learned query.
//! - **SITU MoE.** 128 routed experts, top-8, sigmoid gating, `situ(g,u) = β·tanh(g/β)·σ(g) ·
//!   λ·tanh(u/λ)`, weights normalized then scaled by `expert_weights_scale`.
//!
//! The reference for all of it is llama.cpp's `src/models/inkling2.cpp` and `inkling2-loop.cpp`.

pub mod forward;
#[cfg(test)]
mod tests;

use crate::gguf::{Gguf, TensorInfo, Value};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// `blk.{l}.{name}`.
pub fn t(l: usize, name: &str) -> String {
    format!("blk.{l}.{name}")
}

/// The model geometry, from GGUF metadata keys `inkling2_loop.*`.
#[derive(Clone, Debug)]
pub struct Hparams {
    /// Padded vocabulary (`vocab_size`); the embedding and output rows go past the tokenizer.
    pub n_vocab: usize,
    /// The real vocabulary (`vocab_size_unpadded`): logits past this are dropped.
    pub n_vocab_unpadded: usize,
    pub n_embd: usize,
    /// Physical layers with their own weights (24).
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub d_head: usize,
    pub eps: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    /// `expert_weights_scale` (8.0): the routing weights are normalized then scaled by this.
    pub expert_weights_scale: f32,
    /// SITU: the gate's soft-clamp (`situ_beta`, 4.0).
    pub situ_beta: f32,
    /// SITU: the up branch's soft-clamp (`situ_linear_beta`, 25.0); 0 disables it.
    pub situ_linear_beta: f32,
    /// Per layer: sliding-window attention (else global).
    pub is_swa: Vec<bool>,
    /// `attention.sliding_window` (1024).
    pub n_swa: usize,
    /// Short convolution kernel (`shortconv.l_cache`, 4).
    pub n_conv: usize,
    /// `attn_res.block_size` (6): every this many layers starts a new residual stream.
    pub attn_res_block: usize,
    /// `attention.relative_length` (16): rows of the relative bias table.
    pub n_rel: usize,
    /// `attention.relative_extent` (1024): distances at or past this share the last column.
    pub n_rel_extent: usize,
    /// `attention.qk_clip` (16√2): rows of Q, K and R are scaled to an L2 norm of at most this.
    pub qk_clip: f32,
    /// `attention.relative_clip` (8.0): as `qk_clip`, for R.
    pub rel_clip: f32,
    /// `attention.log_scale_floor` (8192) and `attention.log_scale_alpha` (0.1): full-attention
    /// Q and R are multiplied by `1 + alpha·log(max(1, (pos+1)/floor))`.
    pub log_scale_floor: usize,
    pub log_scale_alpha: f32,
    /// `final_logit_softcapping` (30.0): `cap·tanh(logits/cap)`; 0 disables it.
    pub final_softcap: f32,
    /// The looped core: layers `[loop_start, loop_end)` run `n_passes` times (`loop.*`).
    pub loop_start: usize,
    pub loop_end: usize,
    pub n_passes: usize,
    /// `loop.gate_rank` (64): the control gates' low rank.
    pub gate_rank: usize,
}

impl Hparams {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        ensure!(
            arch == "inkling2_loop",
            "architecture {arch}, want inkling2_loop"
        );
        let k = |s: &str| format!("inkling2_loop.{s}");
        let u = |s: &str| g.meta_u64(&k(s)).map(|v| v as usize);
        let f = |s: &str| g.meta_f64(&k(s)).map(|v| v as f32);

        let n_layer = u("block_count")?;
        let n_embd = u("embedding_length")?;
        // head_count / head_count_kv are per-layer arrays (the global layers may differ); this
        // port assumes one geometry for all of them, which is what the released checkpoint has.
        let heads = g.meta_u64s(&k("attention.head_count"))?;
        let kv_heads = g.meta_u64s(&k("attention.head_count_kv"))?;
        ensure!(!heads.is_empty() && heads.len() == kv_heads.len(), "head arrays");
        ensure!(
            heads.iter().all(|&h| h == heads[0]) && kv_heads.iter().all(|&h| h == kv_heads[0]),
            "per-layer head counts differ ({heads:?} / {kv_heads:?}); this port assumes one geometry"
        );
        let n_head = heads[0] as usize;
        let n_head_kv = kv_heads[0] as usize;
        ensure!(n_head_kv > 0 && n_head.is_multiple_of(n_head_kv), "head counts");

        let pattern = g.meta_u64s(&k("attention.sliding_window_pattern"))?;
        ensure!(
            pattern.len() == n_layer,
            "sliding_window_pattern has {} entries, want {n_layer}",
            pattern.len()
        );
        let is_swa: Vec<bool> = pattern.iter().map(|&x| x != 0).collect();

        Ok(Self {
            n_vocab: u("vocab_size")?,
            n_vocab_unpadded: u("vocab_size_unpadded")?,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            d_head: u("attention.key_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            expert_weights_scale: f("expert_weights_scale")?,
            situ_beta: f("activation.situ_beta")?,
            situ_linear_beta: f("activation.situ_linear_beta")?,
            is_swa,
            n_swa: u("attention.sliding_window")?,
            n_conv: u("shortconv.l_cache")?,
            attn_res_block: u("attn_res.block_size")?,
            n_rel: u("attention.relative_length")?,
            n_rel_extent: u("attention.relative_extent")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_scale_floor: u("attention.log_scale_floor")?,
            log_scale_alpha: f("attention.log_scale_alpha")?,
            final_softcap: g
                .meta_f64(&k("final_logit_softcapping"))
                .unwrap_or(0.0) as f32,
            loop_start: u("loop.start")?,
            loop_end: u("loop.end")?,
            n_passes: u("loop.count")?,
            gate_rank: u("loop.gate_rank")?,
        })
    }

    /// Validate what the forward assumes, so a mismatched checkpoint fails at open.
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.n_vocab_unpadded <= self.n_vocab && self.n_vocab_unpadded > 0,
            "unpadded vocabulary {} of {}",
            self.n_vocab_unpadded,
            self.n_vocab
        );
        ensure!(
            self.n_rel > 0 && self.n_rel_extent > 0 && self.n_swa > 0,
            "relative/ sliding-window geometry"
        );
        ensure!(
            self.n_conv >= 2 && self.n_conv <= 64,
            "convolution kernel {}",
            self.n_conv
        );
        ensure!(
            self.n_swa <= self.n_rel_extent,
            "sliding window {} is wider than the relative bias extent {} (the padded column \
             this port relies on would not exist)",
            self.n_swa,
            self.n_rel_extent
        );
        ensure!(
            self.attn_res_block > 0,
            "attn_res.block_size must be positive"
        );
        ensure!(
            self.loop_start < self.loop_end && self.loop_end <= self.n_layer,
            "loop core [{}, {}) of {} layers",
            self.loop_start,
            self.loop_end,
            self.n_layer
        );
        ensure!(
            self.loop_start.is_multiple_of(self.attn_res_block)
                && self.loop_end.is_multiple_of(self.attn_res_block),
            "loop core [{}, {}) is not on {} layer block boundaries",
            self.loop_start,
            self.loop_end,
            self.attn_res_block
        );
        ensure!(
            self.n_passes >= 2 && self.gate_rank > 0,
            "{} passes at rank {}",
            self.n_passes,
            self.gate_rank
        );
        ensure!(
            self.n_expert_used > 0 && self.n_expert_used <= self.n_expert,
            "{} of {} experts",
            self.n_expert_used,
            self.n_expert
        );
        ensure!(
            self.d_head * self.n_head_kv > 0,
            "attention dimensions"
        );
        for (name, v) in [
            ("eps", self.eps),
            ("qk_clip", self.qk_clip),
            ("rel_clip", self.rel_clip),
            ("situ_beta", self.situ_beta),
            ("situ_linear_beta", self.situ_linear_beta),
        ] {
            ensure!(
                v.is_finite() && v > 0.0,
                "{name} must be finite and positive, got {v}"
            );
        }
        ensure!(
            self.log_scale_alpha.is_finite() && self.log_scale_alpha >= 0.0,
            "log_scale_alpha {}",
            self.log_scale_alpha
        );
        ensure!(self.log_scale_floor > 0, "log_scale_floor must be positive");
        Ok(())
    }

    /// The physical layer a cache slot's weights come from: slots past `n_layer` replay the
    /// looped core (`llama_model_inkling2_loop::layer_phys`).
    pub fn layer_phys(&self, il: usize) -> usize {
        if il < self.n_layer {
            return il;
        }
        self.loop_start + (il - self.n_layer) % (self.loop_end - self.loop_start)
    }

    /// The cache slots one token visits: `n_layer` plus the core once more per extra pass.
    pub fn n_slot(&self) -> usize {
        self.n_layer + (self.n_passes - 1) * (self.loop_end - self.loop_start)
    }

    /// The number of residual streams in flight at layer `l`: every `attn_res_block` layers
    /// pushes one.
    pub fn n_block(&self, l: usize) -> usize {
        l / self.attn_res_block + 1
    }
}

/// A plain-text inventory of an inkling2_loop GGUF: metadata, tensor patterns, byte totals.
pub fn inventory(g: &Gguf) -> Result<String> {
    let hp = Hparams::from_gguf(g)?;
    let mut s = String::new();
    writeln!(
        s,
        "inkling2_loop: {} layers ({} local, {} global), core [{}, {}) × {}, hidden {}, \
         vocab {} ({} unpadded)",
        hp.n_layer,
        hp.is_swa.iter().filter(|&&b| b).count(),
        hp.is_swa.iter().filter(|&&b| !b).count(),
        hp.loop_start,
        hp.loop_end,
        hp.n_passes,
        hp.n_embd,
        hp.n_vocab,
        hp.n_vocab_unpadded
    )?;
    writeln!(s, "\n## metadata ({} keys)", g.meta.len())?;
    for (k, v) in &g.meta {
        writeln!(s, "{k} = {}", v.summary())?;
    }
    #[derive(Default)]
    struct Row {
        count: usize,
        dims: BTreeMap<String, usize>,
        types: BTreeMap<String, usize>,
        bytes: u64,
    }
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    let mut total = 0u64;
    for ti in &g.tensors {
        let r = rows.entry(super::name_pattern(&ti.name)).or_default();
        r.count += 1;
        *r.dims.entry(format!("{:?}", ti.dims)).or_default() += 1;
        *r.types.entry(ti.ty.name()).or_default() += 1;
        r.bytes += ti.nbytes;
        total += ti.nbytes;
    }
    writeln!(s, "\n## tensors ({})", g.tensors.len())?;
    writeln!(s, "pattern | count | dims | types | bytes")?;
    for (p, r) in &rows {
        writeln!(
            s,
            "{p} | {} | {} | {} | {}",
            r.count,
            r.dims
                .iter()
                .map(|(d, n)| if *n > 1 {
                    format!("{d}×{n}")
                } else {
                    d.clone()
                })
                .collect::<Vec<_>>()
                .join(" "),
            r.types
                .iter()
                .map(|(d, n)| format!("{d}×{n}"))
                .collect::<Vec<_>>()
                .join(" "),
            r.bytes
        )?;
    }
    writeln!(
        s,
        "total: {total} ({:.3} GiB)",
        total as f64 / (1u64 << 30) as f64
    )?;
    Ok(s)
}

/// The tensors of one physical layer, checked for the shapes the forward indexes.
pub fn layer_tensors(g: &Gguf, hp: &Hparams, l: usize) -> Result<()> {
    let need = |name: &str| -> Result<&TensorInfo> {
        g.info(&t(l, name))
            .with_context(|| format!("layer {l} is missing {name}"))
    };
    let (e, k) = (hp.n_embd as u64, hp.n_expert as u64);
    let n_ff = 2 * hp.n_ff_exp as u64;
    let kv = (hp.d_head * hp.n_head_kv) as u64;
    let q = (hp.d_head * hp.n_head) as u64;
    let want = |ti: &TensorInfo, dims: &[u64]| -> Result<()> {
        ensure!(
            ti.dims == dims,
            "{}: dims {:?}, want {dims:?}",
            ti.name,
            ti.dims
        );
        Ok(())
    };
    want(need("attn_q.weight")?, &[e, q])?;
    want(need("attn_k.weight")?, &[e, kv])?;
    want(need("attn_v.weight")?, &[e, kv])?;
    want(need("attn_output.weight")?, &[q, e])?;
    want(need("attn_gate.weight")?, &[e, q])?;
    want(need("attn_rel.weight")?, &[e, hp.n_rel as u64 * hp.n_head as u64])?;
    want(
        need("attn_rel_bias.weight")?,
        &[hp.n_rel as u64, hp.n_rel_extent as u64],
    )?;
    want(
        need("attn_rel_clip.weight")?,
        &[hp.n_rel as u64, hp.n_head as u64],
    )?;
    want(need("attn_q_clip.weight")?, &[hp.d_head as u64, hp.n_head as u64])?;
    want(need("attn_sink_proj.weight")?, &[e, hp.n_head as u64])?;
    want(need("attn_k_conv.weight")?, &[hp.n_conv as u64, kv])?;
    want(need("attn_v_conv.weight")?, &[hp.n_conv as u64, kv])?;
    want(need("attn_post_conv.weight")?, &[hp.n_conv as u64, e])?;
    want(need("ffn_post_conv.weight")?, &[hp.n_conv as u64, e])?;
    want(need("ffn_norm.weight")?, &[e])?;
    want(need("ffn_gate_inp.weight")?, &[e, k])?;
    want(need("ffn_gate_up_exps.weight")?, &[e, n_ff, k])?;
    want(need("ffn_down_exps.weight")?, &[hp.n_ff_exp as u64, e, k])?;
    want(need("ffn_res_query.weight")?, &[e])?;
    if hp.is_swa[l] {
        want(need("attn_norm.weight")?, &[e])?;
    } else {
        want(need("attn_q_inp_norm.weight")?, &[e])?;
        want(need("attn_k_inp_norm.weight")?, &[e])?;
        want(need("attn_v_inp_norm.weight")?, &[e])?;
        want(
            need("attn_k_conv_proj.weight")?,
            &[e, hp.n_conv as u64 * hp.n_head_kv as u64],
        )?;
        want(
            need("attn_v_conv_proj.weight")?,
            &[e, hp.n_conv as u64 * hp.n_head_kv as u64],
        )?;
    }
    if l > 0 {
        // A one-column query (the local layers') is stored flat (`[e]`) or as `[e, 1]`,
        // depending on the converter; a three-column one is always 2-D.
        let nq = if hp.is_swa[l] { 1 } else { 3 };
        let ti = need("attn_res_query.weight")?;
        let ok = ti.dims == [e, nq] || (nq == 1 && ti.dims == [e]);
        ensure!(
            ok,
            "{}: dims {:?}, want [e, {nq}]{}",
            ti.name,
            ti.dims,
            if nq == 1 { " or [e]" } else { "" }
        );
    }
    // The looped core's control gates.
    if (hp.loop_start..hp.loop_end).contains(&l) {
        want(need("loop_msg_query.weight")?, &[e, 4])?;
        for c in ["q", "k", "v", "ffn"] {
            want(need(&format!("loop_{c}_ctrl_down.weight"))?, &[2 * e, hp.gate_rank as u64])?;
            want(need(&format!("loop_{c}_ctrl_gain.weight"))?, &[hp.gate_rank as u64, e])?;
            want(need(&format!("loop_{c}_ctrl_forget.weight"))?, &[hp.gate_rank as u64, e])?;
        }
    }
    Ok(())
}

/// Every tensor the forward reads, at the shape it reads it. `Err` names the first mismatch.
pub fn check_tensors(g: &Gguf, hp: &Hparams) -> Result<()> {
    let e = hp.n_embd as u64;
    let want = |name: &str, dims: &[u64]| -> Result<()> {
        let ti = g.info(name)?;
        ensure!(
            ti.dims == dims,
            "{name}: dims {:?}, want {dims:?}",
            ti.dims
        );
        Ok(())
    };
    want("token_embd.weight", &[e, hp.n_vocab as u64])?;
    want("token_embd_norm.weight", &[e])?;
    want("output.weight", &[e, hp.n_vocab as u64])?;
    want("output_norm.weight", &[e])?;
    want("output_res_query.weight", &[e])?;
    for l in 0..hp.n_layer {
        layer_tensors(g, hp, l)?;
    }
    Ok(())
}

/// `Value` arrays of a metadata key as `f32`s, for the metadata table's float arrays.
#[allow(dead_code)]
pub(crate) fn meta_f32s(g: &Gguf, key: &str) -> Option<Vec<f32>> {
    match g.meta.get(key)? {
        Value::Array(a) => Some(a.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect()),
        _ => None,
    }
}
