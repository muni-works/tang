//! DeepSeek-V4.1-Flash (`deepseek_v41`): one file per owner.
//!
//! - [`st`]: a safetensors reader that knows `F8_E4M3`, `F8_E8M0` and the `I8`-packed fp4
//!   experts (the workspace's safetensors 0.4 rejects these headers).
//! - [`quant`]: the number formats, and the fp8 / fp4 quant-dequant steps the forward applies to
//!   activations.
//! - [`gpu`] (cuda): weights on the device in their checkpoint format and the kernels
//!   (`kernels.cu`), each tested against its CPU twin.
//! - [`engine`] (cuda): the forward on the device, the CPU reference's control flow over
//!   [`gpu`]'s kernels; selection steps (top-k, routing lists) still on the host.
//! - [`ep`] (cuda): expert parallelism, one [`engine::GpuModel`] per GPU with routed experts
//!   sharded by id and the MoE combined by all-reduce (`docs/deepseek-ep.md`).
//! - [`serve`] (cuda): the OpenAI-compatible backend (`tang-llm deepseek-serve <dir>`): chat
//!   template, tokenizer, DSML tool calls.
//! - [`tcx`]: CPU twins of the TCX grouped MoE kernels (`DS_MOE_TCX=1`).
//! - [`forward`]: the CPU f32 reference forward, a transcription of `scripts/deepseek_ref.py`
//!   (itself DeepSeek's `inference/model.py` in plain torch), checked against its dumps.
//!
//! See `docs/deepseek-v41-plan.md` for the architecture and the port's milestones.

#[cfg(feature = "cuda")]
pub mod engine;
#[cfg(feature = "cuda")]
pub mod ep;
pub mod forward;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod mega;
#[cfg(feature = "cuda")]
pub mod obs;
#[cfg(feature = "cuda")]
pub mod pdl;
pub mod quant;
#[cfg(feature = "cuda")]
pub mod serve;
pub mod st;
pub mod tcx;

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// `inference/config.json`, field for field (the keys DeepSeek's `ModelArgs` takes).
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_mtp_layers: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    #[serde(default = "yes")]
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub norm_eps: f32,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub original_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub compress_rope_theta: f32,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer: i64,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram_layer_ids: Vec<usize>,
    pub engram_max_ngram_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: usize,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    #[serde(default)]
    pub dspark_n_routed_experts: usize,
    #[serde(default)]
    pub dspark_n_activated_experts: usize,
    /// Not in the released config (a runtime limit in DeepSeek's code); sizes the caches.
    /// `DS41_CTX` overrides it.
    #[serde(default = "max_seq")]
    pub max_seq_len: usize,
    /// Runtime, not in the checkpoint: drafts a greedy speculative step verifies at most
    /// (`DS_DRAFT_N`, else `DS_DRAFT_N_GREEDY`, else the trained block). Below the trained
    /// block the drafter still runs it and the window keeps the first `draft_n` drafts; above
    /// it the drafter's block grows to `draft_n` (more noise tokens than trained).
    #[serde(skip)]
    pub draft_n: usize,
    /// As `draft_n` for sampled steps (`DS_DRAFT_N`, else `DS_DRAFT_N_SAMPLED`, else the
    /// trained block). The draft graphs are keyed by sampler, so each mode has its own.
    #[serde(skip)]
    pub draft_n_sampled: usize,
    /// The checkpoint's `dspark_block_size`; [`Config::load`] raises `dspark_block_size` to
    /// the largest block either mode drafts (it sizes windows and caches).
    #[serde(skip)]
    pub trained_block: usize,
    /// Runtime: `DS_DRAFT_CONF`, the confidence early stop. The window keeps the longest
    /// draft prefix whose product of sigmoid(confidence) is >= this (0 = off).
    #[serde(skip)]
    pub draft_conf: f32,
}

fn yes() -> bool {
    true
}

fn max_seq() -> usize {
    4096
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let s =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut cfg: Self = serde_json::from_str(&s)?;
        if let Some(n) = std::env::var("DS41_CTX").ok().and_then(|v| v.parse().ok()) {
            cfg.max_seq_len = n;
        }
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        let tb = cfg.dspark_block_size;
        let n = |k: &str| {
            env("DS_DRAFT_N")
                .or_else(|| env(k))
                .map_or(tb, |n| (n as usize).max(1))
        };
        (cfg.trained_block, cfg.draft_n, cfg.draft_n_sampled) =
            (tb, n("DS_DRAFT_N_GREEDY"), n("DS_DRAFT_N_SAMPLED"));
        cfg.dspark_block_size = tb.max(cfg.draft_n).max(cfg.draft_n_sampled);
        cfg.draft_conf = env("DS_DRAFT_CONF").unwrap_or(0.0) as f32;
        Ok(cfg)
    }

    /// The drafter's block for a greedy or sampled step: the trained block, or the step's
    /// draft length when that is longer.
    pub fn draft_block(&self, greedy: bool) -> usize {
        self.trained_block.max(self.n_for(greedy))
    }

    fn n_for(&self, greedy: bool) -> usize {
        if greedy {
            self.draft_n
        } else {
            self.draft_n_sampled
        }
    }

    /// How many of a draft's drafts (confidences `conf`, one per draft) the verify window
    /// of a greedy or sampled step takes: at most its draft length, cut at the first draft where the running product
    /// of sigmoid(conf) falls below [`Config::draft_conf`] (at least one draft is kept).
    pub fn draft_keep(&self, conf: &[f32], greedy: bool) -> usize {
        let n = self.n_for(greedy).min(conf.len());
        if self.draft_conf <= 0.0 {
            return n;
        }
        let mut p = 1f64;
        for (i, &c) in conf[..n].iter().enumerate() {
            p /= 1.0 + (-(c as f64)).exp();
            if i > 0 && p < self.draft_conf as f64 {
                return i;
            }
        }
        n
    }

    /// Routed experts and how many a token picks, for layer `lid` (DSpark stages have their own).
    pub fn moe(&self, lid: usize) -> (usize, usize) {
        if lid < self.n_layers {
            return (self.n_routed_experts, self.n_activated_experts);
        }
        let n = if self.dspark_n_routed_experts > 0 {
            self.dspark_n_routed_experts
        } else {
            self.n_routed_experts
        };
        let k = if self.dspark_n_activated_experts > 0 {
            self.dspark_n_activated_experts
        } else {
            self.n_activated_experts
        };
        (n, k)
    }

    pub fn is_kv_source(&self, lid: usize) -> bool {
        lid < self.n_layers && self.kv_source_layers.contains(&lid)
    }

    pub fn is_index_source(&self, lid: usize) -> bool {
        lid < self.n_layers && self.index_source_layers.contains(&lid)
    }
}

/// Engram's hash constants: the compressed token map (from the tokenizer) and numpy-drawn
/// multipliers, precomputed in Python (`scripts/deepseek_tiny.py` writes `engram.json`).
#[derive(Clone, Debug, Deserialize)]
pub struct EngramMeta {
    pub token_map: Vec<i64>,
    pub pad: i64,
    /// [engram layer][n-gram position]
    pub multipliers: Vec<Vec<i64>>,
    /// [engram layer][n-gram size - 2][head]
    pub primes: Vec<Vec<Vec<i64>>>,
    /// [engram layer][hash column]
    pub offsets: Vec<Vec<i64>>,
}

impl EngramMeta {
    pub fn load(path: &Path) -> Result<Self> {
        let s =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&s)?)
    }
}
