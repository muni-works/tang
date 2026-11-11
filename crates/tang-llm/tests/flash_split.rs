//! The two-GPU layer split (`TANG_GPUS=2`) against the single-GPU engine: the same
//! `qwen3next` GGUF loaded twice (once per mode), prefill and decode on both, logits
//! compared bitwise — the same kernels on the same inputs, only the executing device
//! differs, so any difference is a wiring bug (a missed boundary copy, a wrong stream
//! order, stale staging).
//!
//! Needs a 2-GPU CUDA pod and `TANG_LLM_TEST_GGUF`: the first shard of a `qwen3next`
//! GGUF (e.g. Qwen3-Coder-Next UD-Q4_K_XL). Skipped without the env var (panics when
//! `TANG_REQUIRE_CUDA=1`). On the pod (see `docs/models/glm5-split-plan.md`):
//!
//! ```text
//! TANG_LLM_TEST_GGUF=/root/Qwen3-Coder-Next-480B-A35B-Instruct-UD-Q4_K_XL-00001-of-00003.gguf \
//! TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm \
//!     --no-default-features --features cuda --test flash_split -- --nocapture
//! ```

#![cfg(feature = "cuda")]

use anyhow::{ensure, Result};
use tang_llm::flash::coder::{Coder, CoderOpts};

/// Decode steps checked after the prompt.
const STEPS: usize = 8;
/// The prompt: 64 fixed token ids (arbitrary but nonzero, distinct from decode steps').
const PROMPT: std::ops::Range<u32> = 1000..1064;

fn gguf() -> Option<std::path::PathBuf> {
    match std::env::var("TANG_LLM_TEST_GGUF") {
        Ok(p) => Some(p.into()),
        Err(_) => {
            eprintln!("TANG_LLM_TEST_GGUF not set: skipping");
            None
        }
    }
}

fn has_cuda() -> bool {
    tang_moe::gpu::Gpu::for_test().is_some()
}

/// Prompt then `STEPS` decode steps; each step's logits (after its token) recorded.
fn run(e: &mut Coder) -> Result<Vec<Vec<f32>>> {
    let ids: Vec<u32> = PROMPT.collect();
    let mut next = e.prefill(&ids, 8, None)?;
    let mut rows = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        next = e.step(next)?;
        rows.push(e.logits(1));
    }
    Ok(rows)
}

#[test]
fn split_matches_single_gpu() -> Result<()> {
    let Some(path) = gguf() else {
        return Ok(());
    };
    ensure!(has_cuda(), "no CUDA device");

    std::env::set_var("TANG_GPUS", "1");
    let mut one = Coder::load(&path, CoderOpts::default())?;
    eprintln!("single: {}", one.load_report);
    let l1 = run(&mut one)?;
    ensure!(!l1.is_empty() && !l1[0].is_empty(), "empty logits");

    std::env::set_var("TANG_GPUS", "2");
    let mut two = Coder::load(&path, CoderOpts::default())?;
    eprintln!("split: {}", two.load_report);
    let l2 = run(&mut two)?;

    ensure!(l1.len() == l2.len(), "{} vs {} rows", l1.len(), l2.len());
    let mut worst = 0f64;
    for (i, (a, b)) in l1.iter().zip(&l2).enumerate() {
        ensure!(a.len() == b.len(), "vocab differs at step {i}");
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs() as f64;
            if d > worst {
                worst = d;
                eprintln!("first difference: step {i} token {j}: {x} vs {y}");
            }
            ensure!(d == 0.0, "step {i} token {j}: |Δlogit| {d:.3e}");
        }
    }
    eprintln!(
        "split parity: {} steps of {} logits, all bitwise equal",
        l1.len(),
        l1[0].len()
    );
    Ok(())
}
