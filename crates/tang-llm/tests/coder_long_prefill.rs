//! A long prompt through the Coder's wide prefill (`prefill_wide`) against the narrow one
//! (`prefill_slot`, windows of `MAX_T`): same engine, same tokens, then decode steps fed the
//! narrow run's greedy tokens. Next tokens must match and each step's KL(narrow || wide)
//! must stay under 1e-3 (`flash-wide-check`'s bar). The prompt is several wide windows long,
//! the shape that crashed dart in serve (`fe_q8: CUDA_ERROR_ILLEGAL_ADDRESS` on the first
//! 3,318-token prompt) when stage 0's view kept the narrow kernels while it pointed at the
//! wide scratch.
//!
//! Needs a CUDA GPU and `TANG_LLM_TEST_GGUF`: the first shard of a `qwen3next` GGUF (e.g.
//! Qwen3-Coder-Next UD-Q4_K_XL). Skipped without the env var:
//!
//! ```text
//! TANG_LLM_TEST_GGUF=/root/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
//! cargo test --release -p tang-llm \
//!     --no-default-features --features cuda --test coder_long_prefill -- --nocapture
//! ```

#![cfg(feature = "cuda")]

use anyhow::{ensure, Result};
use tang_llm::flash::coder::{Coder, CoderOpts};

/// Prompt length: the crashing serve prompt's.
const LEN: usize = 3318;
/// Wide window rows (serve's default).
const ROWS: usize = 256;
/// The prompt text, chat-encoded then cycled.
const PROMPT: &str = "def quicksort(xs): return xs if len(xs) < 2 else";
/// Greedy decode steps compared after each prefill.
const STEPS: usize = 8;

fn gguf() -> Option<std::path::PathBuf> {
    match std::env::var("TANG_LLM_TEST_GGUF") {
        Ok(p) => Some(p.into()),
        Err(_) => {
            eprintln!("TANG_LLM_TEST_GGUF not set: skipping");
            None
        }
    }
}

/// `STEPS` decode steps from `next`: the tokens picked and each step's logits. `forced`
/// feeds these tokens instead of the picks.
fn decode(
    e: &mut Coder,
    slot: usize,
    mut next: u32,
    forced: Option<&[u32]>,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let (mut toks, mut lg) = (Vec::with_capacity(STEPS), Vec::with_capacity(STEPS));
    for i in 0..STEPS {
        let got = e.decode_batch(&[(slot, next)])?[0];
        lg.push(e.logits(1));
        toks.push(got);
        next = forced.map_or(got, |f| f[i]);
    }
    Ok((toks, lg))
}

/// KL(p || q) of two logit rows, in f64.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let ls = |l: &[f32]| -> Vec<f64> {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
        l.iter().map(|&v| v as f64 - m - z).collect()
    };
    let (a, b) = (ls(p), ls(q));
    a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum()
}

#[test]
fn wide_prefill_of_thousands_of_tokens_matches_narrow() -> Result<()> {
    let Some(path) = gguf() else {
        return Ok(());
    };
    ensure!(tang_moe::gpu::Gpu::for_test().is_some(), "no CUDA device");

    let opts = CoderOpts {
        max_ctx: (LEN + STEPS + 8).next_multiple_of(1024),
        ..CoderOpts::default()
    };
    let mut e = Coder::load(&path, opts)?;
    ensure!(e.temperature == 0.0, "greedy only");
    e.enable_wide(ROWS)?;
    // A real prompt cycled to `LEN` tokens, as `flash-wide-check` builds it (random ids put
    // the model in a degenerate loop where even a correct wide prefill misses the bar).
    let base = tang_llm::flash::tokenize::encode_chat(&path, PROMPT, None)?;
    let ids: Vec<u32> = base.iter().cycle().take(LEN).copied().collect();

    let s1 = e.alloc_slot()?;
    let n1 = e.prefill_slot(s1, &ids)?;
    let (narrow, l1) = decode(&mut e, s1, n1, None)?;
    e.free_slot(s1);

    let s2 = e.alloc_slot()?;
    let n2 = e.prefill_wide(s2, &ids)?;
    let (wide, l2) = decode(&mut e, s2, n1, Some(&narrow))?;
    e.free_slot(s2);

    let worst = l1
        .iter()
        .zip(&l2)
        .map(|(a, b)| kl(a, b))
        .fold(0f64, f64::max);
    eprintln!("narrow {n1} {narrow:?}\nwide   {n2} {wide:?}\nmax KL {worst:.2e}");
    ensure!(n1 == n2, "first token: narrow {n1} vs wide {n2}");
    ensure!(worst <= 1e-3, "KL(narrow || wide) {worst:.2e} > 1e-3");
    Ok(())
}
