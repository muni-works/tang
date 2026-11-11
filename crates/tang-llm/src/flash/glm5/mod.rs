//! GLM-5.3-Flash (GGUF architecture `glm5-next`): one file per owner.
//!
//! - [`gguf`]/[`forward`] (this file's siblings, W1): geometry from GGUF metadata, the tensor
//!   inventory, and the CPU f32 reference forward that calls the component modules.
//! - [`mhc`] (W2): mHC Sinkhorn hyper-connections.
//! - [`mla`] (W3): the 11 NoPE-MLA DSA layers with the pooled indexer.
//! - [`kda`] (W4): the 34 Kimi-delta-attention layers.
//! - [`moe`] (W5): the sigmoid/noaux_tc router, routed + shared experts, dense lead-in FFN.

pub mod dump;
pub mod engine;
pub mod forward;
pub mod gguf;
pub mod kda;
#[cfg(feature = "cuda")]
pub mod kda_cuda;
pub mod kda_debug;
#[cfg(feature = "cuda")]
pub mod kda_dev;
pub mod mhc;
#[cfg(feature = "cuda")]
pub mod mhc_fused;
pub mod mla;
#[cfg(feature = "cuda")]
pub mod mla_debug;
#[cfg(feature = "cuda")]
pub mod mla_dev;
pub mod moe;
#[cfg(feature = "cuda")]
pub mod moe_debug;
pub mod mtp;
#[cfg(test)]
mod mtp_tests;
#[cfg(feature = "cuda")]
pub mod serve_glue;
pub mod trace;

use crate::gguf::Gguf;
use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The model geometry, from GGUF metadata keys `glm5-next.*` (the writer is
/// llama.cpp's `conversion/glm.py`, `Glm5NextModel.set_gguf_parameters`).
///
/// Field names follow the Hugging Face config where a direct equivalent exists (`kv_lora_rank`,
/// `q_lora_rank`, `n_expert`, ...), and the llama.cpp hparam otherwise.
#[derive(Clone, Debug)]
pub struct Hparams {
    pub n_vocab: usize,
    pub n_embd: usize,
    /// Trunk layers (45), excluding the MTP/NextN layer.
    pub n_layer: usize,
    /// `first_k_dense_replace` (3): layers below this have a dense FFN instead of MoE.
    pub n_dense_lead: usize,
    /// MoE layers also have `n_expert_shared` shared experts (1), always run.
    pub n_expert_shared: usize,
    /// The dense lead-in FFN width (`intermediate_size`, 12288).
    pub n_ff: usize,
    /// `attention.head_count` (64).
    pub n_head: usize,
    /// The DSA qk head dim (`attention.key_length_mla`, `qk_nope_head_dim` = 256).
    pub d_qk: usize,
    /// The DSA v head dim (`attention.value_length_mla`, `v_head_dim` = 256).
    pub d_v: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    /// KDA (`kda.head_dim`, 128) × `attention.head_count` (64) = d_inner.
    pub kda_head_dim: usize,
    /// `ssm.conv_kernel` (4).
    pub ssm_d_conv: usize,
    /// Per-layer gate lower bound of the KDA decay gate (−5.0), or `None` for softplus.
    pub kda_gate_lower_bound: Option<f32>,
    /// `attention.indexer.head_count` (32).
    pub idx_heads: usize,
    /// `attention.indexer.key_length` (128).
    pub idx_dim: usize,
    /// `attention.indexer.top_k` (2048).
    pub idx_top_k: usize,
    /// `attention.indexer.kpool` (4).
    pub kpool: usize,
    /// `attention.indexer.kpool_select_tail`.
    pub kpool_select_tail: bool,
    /// Per layer: `true` for KDA (linear attention), `false` for DSA (MLA). Length `n_layer`.
    pub is_kda: Vec<bool>,
    /// Per layer: `true` when the layer runs its own indexer (its top-k selection is shared with
    /// any following `false` layers). From `attention.indexer.types`, default all `true`.
    pub indexer_full: Vec<bool>,
    /// mHC (`hyper_connection.count`, 4).
    pub hc: usize,
    /// mHC Sinkhorn iterations (`hyper_connection.sinkhorn_iterations`, 20).
    pub hc_sinkhorn_iters: usize,
    /// mHC epsilon (`hyper_connection.epsilon`, 1e-6).
    pub hc_eps: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    /// `expert_weights_scale` (`routed_scaling_factor`, 2.5).
    pub expert_weights_scale: f32,
    /// `expert_weights_norm` (`norm_topk_prob`).
    pub expert_weights_norm: bool,
    /// Swiglu clamp per layer (from the `swiglu_clamp_exp` array, or `None`: unclamped).
    pub swiglu_limit: Vec<Option<f32>>,
    /// RMSNorm epsilon (`attention.layer_norm_rms_epsilon`, 1e-5).
    pub eps: f32,
    /// NextN/MTP layers after the trunk (`nextn_predict_layers`, 1).
    pub n_nextn: usize,
}

impl Hparams {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        ensure!(arch == "glm5-next", "architecture {arch}, want glm5-next");
        let k = |s: &str| format!("glm5-next.{s}");
        let u = |s: &str| g.meta_u64(&k(s)).map(|v| v as usize);
        let f = |s: &str| g.meta_f64(&k(s)).map(|v| v as f32);

        // block_count covers trunk + NextN layers; the trunk is what we loop over.
        let n_total = u("block_count")?;
        let n_nextn = match g.meta.get(&k("nextn_predict_layers")) {
            Some(v) => v.as_u64().unwrap_or(0) as usize,
            None => 0,
        };
        ensure!(
            n_total > n_nextn,
            "block_count {n_total} <= nextn layers {n_nextn}"
        );
        let n_layer = n_total - n_nextn;
        // The converter pads attention.head_count_kv to block_count with 1s for the NextN
        // layer; a layer is KDA exactly when its kv head count is 0 (llama.cpp is_recr).
        let kv_heads = g.meta_u64s(&k("attention.head_count_kv"))?;
        ensure!(
            kv_heads.len() >= n_total,
            "attention.head_count_kv has {} entries, want {n_total}",
            kv_heads.len()
        );
        let is_kda: Vec<bool> = kv_heads[..n_layer].iter().map(|&h| h == 0).collect();
        ensure!(
            is_kda.iter().any(|b| !b),
            "no DSA layers: not a glm5-next model?"
        );

        let indexer_full: Vec<bool> = match g.meta.get(&k("attention.indexer.types")) {
            Some(v) => {
                let a = v
                    .as_array()
                    .context("attention.indexer.types is not an array")?;
                // one entry per *trunk* layer
                ensure!(
                    a.len() >= n_layer,
                    "indexer.types has {} entries, want {n_layer}",
                    a.len()
                );
                a[..n_layer]
                    .iter()
                    .map(|x| x.as_u64().unwrap_or(0) != 0)
                    .collect()
            }
            None => vec![true; n_layer],
        };

        let swiglu_limit: Vec<Option<f32>> = {
            let raw = g
                .meta
                .get(&k("swiglu_clamp_exp"))
                .map(|v| -> Result<Vec<f64>> {
                    match v {
                        crate::gguf::Value::Array(a) => a
                            .iter()
                            .map(|x| x.as_f64().context("swiglu_clamp_exp entry"))
                            .collect(),
                        other => Ok(vec![other.as_f64().context("swiglu_clamp_exp")?]),
                    }
                })
                .transpose()?;
            match raw {
                Some(v) if v.len() == 1 => vec![Some(v[0] as f32); n_layer],
                Some(v) if v.len() >= n_layer => {
                    v[..n_layer].iter().map(|&x| Some(x as f32)).collect()
                }
                Some(v) => bail!("swiglu_clamp_exp has {} entries", v.len()),
                None => vec![None; n_layer],
            }
        };

        Ok(Self {
            n_vocab: g.info("token_embd.weight")?.dims[1] as usize,
            n_embd: u("embedding_length")?,
            n_layer,
            n_dense_lead: g
                .meta
                .get(&k("leading_dense_block_count"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            n_expert_shared: u("expert_shared_count")?,
            n_ff: u("feed_forward_length")?,
            n_head: u("attention.head_count")?,
            d_qk: u("attention.key_length_mla")?,
            d_v: u("attention.value_length_mla")?,
            q_lora_rank: u("attention.q_lora_rank")?,
            kv_lora_rank: u("attention.kv_lora_rank")?,
            kda_head_dim: u("kda.head_dim")?,
            ssm_d_conv: u("ssm.conv_kernel")?,
            kda_gate_lower_bound: match g.meta.get(&k("kda.gate_lower_bound")) {
                Some(v) => v.as_f64().map(|x| x as f32),
                None => None,
            },
            idx_heads: u("attention.indexer.head_count")?,
            idx_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            kpool: u("attention.indexer.kpool")?,
            kpool_select_tail: match g.meta.get(&k("attention.indexer.kpool_select_tail")) {
                Some(v) => v.as_u64().unwrap_or(0) != 0,
                None => false,
            },
            is_kda,
            indexer_full,
            hc: u("hyper_connection.count")?,
            hc_sinkhorn_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            expert_weights_scale: g
                .meta
                .get(&k("expert_weights_scale"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            expert_weights_norm: matches!(
                g.meta.get(&k("expert_weights_norm")),
                Some(crate::gguf::Value::Bool(true))
            ),
            swiglu_limit,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_nextn,
        })
    }

    /// The mHC weight vector width: `(2 + hc) * hc` per block (pre | post | comb).
    pub fn hc_mix_dim(&self) -> usize {
        (2 + self.hc) * self.hc
    }

    /// True when layer `l` is dense-FFN (the lead-in), false for MoE.
    pub fn is_dense(&self, l: usize) -> bool {
        l < self.n_dense_lead
    }

    /// The swiglu clamp for layer `l`'s routed experts.
    pub fn clamp_exp(&self, l: usize) -> Option<f32> {
        self.swiglu_limit.get(l).copied().flatten()
    }

    /// The swiglu clamp for layer `l`'s shared expert / dense FFN (`swiglu_clamp_shexp`,
    /// defaults to the routed clamp).
    pub fn clamp_shexp(&self, l: usize) -> Option<f32> {
        self.clamp_exp(l)
    }
}

/// mHC stream count (`hyper_connection.count`).
pub const HC_STREAMS: usize = 4;
/// mHC Sinkhorn iterations (`hyper_connection.sinkhorn_iterations`).
pub const SINKHORN_ITERS: u32 = 20;
/// mHC epsilon (`hyper_connection.epsilon`).
pub const HC_EPS: f32 = 1e-6;

/// `blk.{l}.{name}`.
pub fn t(l: usize, name: &str) -> String {
    format!("blk.{l}.{name}")
}

/// A plain-text inventory of a glm5-next GGUF: metadata, tensor patterns, byte totals.
pub fn inventory(g: &Gguf) -> Result<String> {
    let hp = Hparams::from_gguf(g)?;
    let mut s = String::new();
    writeln!(
        s,
        "glm5-next: {} layers ({} KDA + {} DSA), hidden {}, vocab {}",
        hp.n_layer,
        hp.is_kda.iter().filter(|&&b| b).count(),
        hp.is_kda.iter().filter(|&&b| !b).count(),
        hp.n_embd,
        hp.n_vocab
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
    for t in &g.tensors {
        let r = rows.entry(super::name_pattern(&t.name)).or_default();
        r.count += 1;
        *r.dims.entry(format!("{:?}", t.dims)).or_default() += 1;
        *r.types.entry(t.ty.name()).or_default() += 1;
        r.bytes += t.nbytes;
        total += t.nbytes;
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

#[cfg(test)]
pub(crate) mod synth {
    //! A GGUF file writer for tests: build a tiny glm5-next header + tensor data, open it with
    //! [`Gguf`], never touch the network.

    use super::*;
    use crate::gguf::Value;
    use std::path::Path;

    /// One tensor to write: name, dims, row-major f32 data.
    pub struct T<'a> {
        pub name: &'a str,
        pub dims: &'a [u64],
        pub data: &'a [f32],
    }

    fn put_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn put_str(b: &mut Vec<u8>, s: &str) {
        put_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }

    /// Write a single-shard GGUF v3 file with the given metadata and f32 tensors.
    /// Returns the path.
    pub fn write_gguf(path: &Path, meta: &[(&str, Value)], tensors: &[T]) -> Result<()> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        put_u32(&mut b, 3);
        put_u64(&mut b, tensors.len() as u64);
        put_u64(&mut b, meta.len() as u64);
        for (k, v) in meta {
            put_str(&mut b, k);
            put_u32(&mut b, value_ty(v));
            put_value(&mut b, v);
        }
        let mut data = Vec::new();
        for t in tensors {
            let n: u64 = t.dims.iter().product();
            ensure!(t.data.len() == n as usize, "{}: data length", t.name);
            put_str(&mut b, t.name);
            put_u32(&mut b, t.dims.len() as u32);
            for &d in t.dims {
                put_u64(&mut b, d);
            }
            put_u32(&mut b, 0); // F32
            put_u64(&mut b, data.len() as u64); // relative to the aligned data start
            for v in t.data {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&data);
        std::fs::write(path, b)?;
        Ok(())
    }

    fn value_ty(v: &Value) -> u32 {
        match v {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    fn put_value(b: &mut Vec<u8>, v: &Value) {
        match v {
            Value::U8(v) => b.push(*v),
            Value::I8(v) => b.push(*v as u8),
            Value::U16(v) => b.extend_from_slice(&v.to_le_bytes()),
            Value::I16(v) => b.extend_from_slice(&v.to_le_bytes()),
            Value::U32(v) => b.extend_from_slice(&v.to_le_bytes()),
            Value::I32(v) => b.extend_from_slice(&v.to_le_bytes()),
            Value::F32(v) => b.extend_from_slice(&v.to_le_bytes()),
            Value::Bool(v) => b.push(*v as u8),
            Value::Str(s) => put_str(b, s),
            Value::Array(a) => {
                // The parser reads the element type only for arrays; assume homogeneous.
                put_u32(b, a.first().map(value_ty).unwrap_or(4));
                put_u64(b, a.len() as u64);
                for x in a {
                    put_value(b, x);
                }
            }
            Value::U64(v) => b.extend_from_slice(&v.to_le_bytes()),
            Value::I64(v) => b.extend_from_slice(&v.to_le_bytes()),
            Value::F64(v) => b.extend_from_slice(&v.to_le_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_mix_dim_is_24() {
        let hp = Hparams {
            n_vocab: 4,
            n_embd: 2,
            n_layer: 1,
            n_dense_lead: 1,
            n_expert_shared: 1,
            n_ff: 2,
            n_head: 1,
            d_qk: 1,
            d_v: 1,
            q_lora_rank: 1,
            kv_lora_rank: 1,
            kda_head_dim: 1,
            ssm_d_conv: 4,
            kda_gate_lower_bound: None,
            idx_heads: 1,
            idx_dim: 1,
            idx_top_k: 4,
            kpool: 4,
            kpool_select_tail: false,
            is_kda: vec![true],
            indexer_full: vec![true],
            hc: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            n_expert: 2,
            n_expert_used: 2,
            n_ff_exp: 2,
            expert_weights_scale: 2.5,
            expert_weights_norm: true,
            swiglu_limit: vec![None],
            eps: 1e-5,
            n_nextn: 0,
        };
        assert_eq!(hp.hc_mix_dim(), 24);
        assert!(hp.is_dense(0));
    }
}

/// The process env (and the trace's global switch) as seen by the glm5 tests. The harness
/// runs tests in parallel and the engine reads `TANG_GLM5_*` at load and per call, so a
/// `set_var` in one test leaks into another's run. Every glm5 test holds one guard for its
/// whole body: [`env_write`] if it sets / removes env vars (or toggles the trace), else
/// [`env_read`]. Readers still run in parallel with each other. One guard per test: never
/// take a second one inside (no re-entrancy).
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Exclusive: this test mutates process env (poison-tolerant: a failed test must not fail
/// every later one).
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn env_write() -> std::sync::RwLockWriteGuard<'static, ()> {
    ENV_LOCK.write().unwrap_or_else(|e| e.into_inner())
}

/// Shared: this test only reads the ambient env.
#[cfg(test)]
pub(crate) fn env_read() -> std::sync::RwLockReadGuard<'static, ()> {
    ENV_LOCK.read().unwrap_or_else(|e| e.into_inner())
}
