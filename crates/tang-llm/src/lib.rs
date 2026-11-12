//! LLM inference on tang-compute.
//!
//! Loads Qwen3-family safetensors checkpoints straight from a Hugging Face snapshot directory
//! and runs them on any [`ComputeDevice`](tang_compute::ComputeDevice) (Metal on Apple Silicon),
//! with a device-resident KV cache, the checkpoint's own chat template, and an
//! OpenAI-compatible server.

pub mod advertise;
pub mod auth;
pub mod batch;
pub mod blocks;
pub mod chat;
pub mod config;
pub mod dit;
pub mod draft;
pub mod draft_model;
pub mod engine;
pub mod ext;
pub mod flash;
pub mod gguf;
pub mod image_pipeline;
pub mod image_preview;
pub mod image_server;
pub mod kvcomp;
pub mod kvstore;
pub mod model;
pub mod model_id;
pub mod node;
pub mod pair;
pub mod pull;
pub mod queue;
pub mod qwen3vl;
pub mod retention;
pub mod sample;
pub mod scheduler;
pub mod server;
pub mod slots;
pub mod special;
pub mod step_cost;
mod structured;
pub mod telemetry;
pub mod vae;
pub mod vision;
pub mod weights;

pub use config::Config;
pub use engine::{Engine, Request};
pub use model::{Cache, Dtype, Model};

use std::path::PathBuf;

/// A model directory, or a Hugging Face repo id (`Qwen/Qwen3-4B`) already in the local HF cache.
pub fn resolve_model(spec: &str) -> anyhow::Result<PathBuf> {
    let p = PathBuf::from(spec);
    if p.join("config.json").exists() {
        return Ok(p);
    }
    let snaps = hf_hub_dir()
        .join(format!("models--{}", spec.replace('/', "--")))
        .join("snapshots");
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
        .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    dirs.retain(|d: &PathBuf| d.join("config.json").exists());
    dirs.sort_by_key(|d| std::fs::metadata(d).and_then(|m| m.modified()).ok());
    dirs.pop().ok_or_else(|| {
        anyhow::anyhow!(
            "{spec}: not a model directory, and not in the Hugging Face cache ({})",
            snaps.display()
        )
    })
}

/// The Hugging Face hub cache: `HF_HUB_CACHE`, `$HF_HOME/hub`, or `~/.cache/huggingface/hub`.
pub fn hf_hub_dir() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })
        .unwrap_or_default()
}
