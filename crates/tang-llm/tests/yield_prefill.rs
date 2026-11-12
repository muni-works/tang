//! A background request's prefill gives way between chunks: the run stops at a chunk boundary
//! with its prefilled blocks kept, an interactive request (another conversation) runs in
//! between, and the background request then goes on from where it stopped and generates
//! exactly what it would have without the interruption.
//!
//! Needs `TANG_LLM_TEST_MODEL` (e.g. `mlx-community/Qwen3-4B-4bit`); skipped without it. Run
//! with `--release`.
#![cfg(any(feature = "metal", feature = "cuda"))]

use serde_json::json;
use tang_compute::ComputeDevice;
use tang_llm::engine::{Control, Engine, Outcome, Progress, Request};
use tang_llm::sample::Sampling;
use tang_llm::Dtype;

fn model() -> Option<std::path::PathBuf> {
    let Ok(spec) = std::env::var("TANG_LLM_TEST_MODEL") else {
        eprintln!("TANG_LLM_TEST_MODEL not set: skipping");
        return None;
    };
    Some(tang_llm::resolve_model(&spec).expect("TANG_LLM_TEST_MODEL"))
}

fn device() -> impl ComputeDevice {
    #[cfg(feature = "metal")]
    return tang_compute::MetalDevice::new().expect("Metal");
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return tang_compute::CudaComputeDevice::new().expect("CUDA");
}

fn req(user: &str, key: &str, max: usize) -> Request {
    Request {
        messages: json!([{ "role": "user", "content": user }]),
        images: vec![],
        tools: None,
        response_schema: None,
        think: Some(false),
        thinking_budget: None,
        temperature_set: true,
        top_k_set: false,
        top_p_set: false,
        presence_penalty: None,
        sampling: Sampling {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens: Some(max),
        stop: vec![],
        cache_key: Some(key.into()),
        tenant: None,
        prefill_only: false,
    }
}

fn long_text() -> String {
    (0..3000)
        .map(|i| ["alpha", "beta", "gamma", "delta", "kappa"][(i * 7 + i / 3) % 5])
        .collect::<Vec<_>>()
        .join(" ")
        + "\n\nHow many words above are \"gamma\"? Guess."
}

/// Gives way after `chunks` prefill chunks; records progress.
struct YieldAfter {
    chunks: usize,
    asked: usize,
    last: Progress,
}

impl Control for YieldAfter {
    fn keep_prefilling(&mut self) -> bool {
        self.asked += 1;
        self.asked < self.chunks
    }

    fn progress(&mut self, p: Progress) {
        self.last = p;
    }
}

#[test]
fn a_yielded_prefill_goes_on_where_it_stopped() {
    let Some(dir) = model() else { return };
    let mut e = Engine::load(device(), &dir, 8192, Dtype::Bf16).unwrap();
    e.set_slots(2);
    let bg = req(&long_text(), "background", 24);

    // Uninterrupted, for reference.
    let (_, want) = e.complete(&bg, |_| true).unwrap();
    let want_tokens = e.last_tokens().1.to_vec();
    assert!(want.prompt_tokens > 3 * 512, "{}", want.prompt_tokens);
    e.reset();

    // Interrupted after two chunks.
    let mut ctl = YieldAfter {
        chunks: 2,
        asked: 0,
        last: Progress::default(),
    };
    let out = e.complete_with(&bg, None, |_| true, &mut ctl).unwrap();
    let Outcome::Yielded {
        prompt_tokens,
        cached_tokens,
        prefilled,
        ..
    } = out
    else {
        panic!("expected a yield, got {out:?}");
    };
    assert_eq!(prompt_tokens, want.prompt_tokens);
    assert_eq!(cached_tokens, 0);
    assert_eq!(prefilled, 2 * 512);
    assert_eq!(ctl.last.prefilled, prefilled);
    assert_eq!(ctl.last.generated, 0);

    // An interactive request from another conversation runs in between.
    let (_, u) = e
        .complete(&req("Say hi.", "interactive", 4), |_| true)
        .unwrap();
    assert!(u.completion_tokens > 0);

    // The background request goes on from its prefilled blocks.
    let (_, got) = e.complete(&bg, |_| true).unwrap();
    assert_eq!(got.cached_tokens, prefilled, "picks up where it stopped");
    assert_eq!(e.last_tokens().1, &want_tokens[..], "same output");
}

#[test]
fn a_yielded_prefill_survives_losing_its_slot() {
    let Some(dir) = model() else { return };
    let mut e = Engine::load(device(), &dir, 8192, Dtype::Bf16).unwrap();
    // One slot: the interactive request takes the background one's cache; its sealed blocks
    // stay in the pool.
    e.set_slots(1);
    let bg = req(&long_text(), "background", 8);
    let mut ctl = YieldAfter {
        chunks: 3,
        asked: 0,
        last: Progress::default(),
    };
    let out = e.complete_with(&bg, None, |_| true, &mut ctl).unwrap();
    assert!(
        matches!(
            out,
            Outcome::Yielded {
                prefilled: 1536,
                ..
            }
        ),
        "{out:?}"
    );
    e.complete(&req("Say hi.", "interactive", 4), |_| true)
        .unwrap();
    let (_, got) = e.complete(&bg, |_| true).unwrap();
    // Whole blocks of what it prefilled come back from the pool.
    assert_eq!(got.cached_tokens, 1536 / 256 * 256);
}
