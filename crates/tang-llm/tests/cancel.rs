//! A run whose client goes away stops at the next decode step, and the engine serves the next
//! request from what it left in the cache.
//!
//! Needs `TANG_LLM_TEST_MODEL` (e.g. `mlx-community/Qwen3-4B-4bit`); skipped without it. Run
//! with `--release`.
#![cfg(any(feature = "metal", feature = "cuda"))]

use serde_json::json;
use tang_compute::ComputeDevice;
use tang_llm::engine::{Control, Engine, Finish, Outcome, Progress, Request};
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

fn req(user: &str, max: usize) -> Request {
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
        cache_key: Some("cancel".into()),
        tenant: None,
        prefill_only: false,
    }
}

/// Cancels once `after` tokens are generated.
struct GoAway {
    after: usize,
    generated: usize,
}

impl Control for GoAway {
    fn progress(&mut self, p: Progress) {
        self.generated = p.generated;
    }
    fn cancelled(&mut self) -> bool {
        self.generated >= self.after
    }
}

#[test]
fn a_cancelled_run_stops_early_and_the_engine_goes_on() {
    let Some(dir) = model() else { return };
    let mut e = Engine::load(device(), &dir, 4096, Dtype::Bf16).unwrap();
    let r = req("Write a long story about a lighthouse.", 400);
    let mut ctl = GoAway {
        after: 5,
        generated: 0,
    };
    let Outcome::Done(finish, usage) = e.complete_with(&r, None, |_| true, &mut ctl).unwrap()
    else {
        panic!("yielded")
    };
    assert_eq!(finish, Finish::Cancelled);
    assert_eq!(usage.completion_tokens, 5);
    // The same conversation again: it reuses the prompt and runs to its limit.
    let (finish, again) = e
        .complete(&req("Write a long story about a lighthouse.", 20), |_| true)
        .unwrap();
    assert_eq!(finish, Finish::Length);
    assert_eq!(again.completion_tokens, 20);
    assert!(again.cached_tokens > 0, "{again:?}");
}
