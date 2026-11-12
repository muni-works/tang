//! KV blocks shared between conversations: a second conversation (another
//! `prompt_cache_key`) whose prompt starts like the first's reuses the first's blocks instead of
//! prefilling them, and generates exactly what it would have alone; the block pool stays within
//! its memory budget; and blocks saved to disk come back after a restart, for any conversation
//! that shares them.
//!
//! Needs `TANG_LLM_TEST_MODEL` (e.g. `mlx-community/Qwen3-4B-4bit`); skipped without it. Run
//! with `--release`.
#![cfg(any(feature = "metal", feature = "cuda"))]

use serde_json::json;
use tang_compute::ComputeDevice;
use tang_llm::blocks::BLOCK;
use tang_llm::engine::{Engine, Request, Usage};
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

/// Deterministic filler text of about `words` words.
fn filler(words: usize, salt: usize) -> String {
    const W: [&str; 16] = [
        "kernel",
        "buffer",
        "token",
        "cache",
        "block",
        "layer",
        "query",
        "value",
        "shared",
        "prefix",
        "engine",
        "model",
        "budget",
        "memory",
        "attention",
        "position",
    ];
    (0..words)
        .map(|i| W[(i * 7 + salt * 3 + i / 5) % W.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

fn req(system: &str, user: &str, key: &str, max: usize) -> Request {
    Request {
        messages: json!([
            { "role": "system", "content": system },
            { "role": "user", "content": user },
        ]),
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

/// Usage and completion token ids.
fn run<D: ComputeDevice>(e: &mut Engine<D>, r: &Request) -> (Usage, Vec<u32>) {
    let (_, usage) = e.complete(r, |_| true).unwrap();
    (usage, e.last_tokens().1.to_vec())
}

/// A system prompt the chat template turns into a shared prefix of between 8 and 9 blocks.
fn system_prompt<D: ComputeDevice>(e: &Engine<D>) -> String {
    let mut words = 1000;
    loop {
        let s = filler(words, 1);
        let n = e.count_tokens(&s).unwrap();
        if n > 8 * BLOCK + 20 {
            assert!(n < 8 * BLOCK + 150, "filler overshot: {n} tokens");
            return s;
        }
        words += 25;
    }
}

#[test]
fn a_second_conversation_reuses_the_first_ones_prefix() {
    let Some(dir) = model() else { return };
    let mut e = Engine::load(device(), &dir, 8192, Dtype::Bf16).unwrap();
    e.set_slots(4);
    let system = system_prompt(&e);
    let a = req(&system, "Name three colours.", "conv-a", 24);
    let b = req(&system, "Name three animals, briefly.", "conv-b", 24);

    let (ua, _) = run(&mut e, &a);
    assert_eq!(ua.cached_tokens, 0);
    assert!(
        ua.prompt_tokens + 24 < 9 * BLOCK,
        "a must not seal a ninth block"
    );
    // b shares a's first 8 blocks and prefills only the rest.
    let (ub, shared_out) = run(&mut e, &b);
    eprintln!(
        "a: {} prompt tokens; b: {} prompt tokens, {} cached",
        ua.prompt_tokens, ub.prompt_tokens, ub.cached_tokens
    );
    assert_eq!(
        ub.cached_tokens,
        8 * BLOCK,
        "b should start from a's 8 sealed blocks"
    );
    let stats = e.kv_stats();
    // a's 8 shared blocks plus each conversation's own tail.
    assert_eq!(stats.used, 8 + 1 + 1, "{stats:?}");

    // b alone, from nothing: the same tokens (the shared prefix is a multiple of the prefill
    // chunk, so the blocks were computed exactly as b would have).
    e.reset();
    let (ub2, alone_out) = run(
        &mut e,
        &req(&system, "Name three animals, briefly.", "conv-c", 24),
    );
    assert_eq!(ub2.cached_tokens, 0);
    assert_eq!(
        shared_out, alone_out,
        "shared-prefix output differs from prefilling alone"
    );
}

#[test]
fn a_shared_block_reused_part_way_is_copied() {
    let Some(dir) = model() else { return };
    let mut e = Engine::load(device(), &dir, 8192, Dtype::Bf16).unwrap();
    e.set_slots(4);
    let system = system_prompt(&e);
    // a runs long enough to seal its ninth block, which b shares only the start of.
    let a = req(
        &system,
        "Write a long story about a lighthouse.",
        "conv-a",
        300,
    );
    let b = req(&system, "Name three animals, briefly.", "conv-b", 24);
    let (ua, _) = run(&mut e, &a);
    assert!(ua.prompt_tokens + ua.completion_tokens >= 9 * BLOCK);
    let (ub, shared_out) = run(&mut e, &b);
    let common = ub.cached_tokens;
    eprintln!("b: {} prompt tokens, {common} cached", ub.prompt_tokens);
    assert!(common > 8 * BLOCK && common < ub.prompt_tokens);
    e.reset();
    let (_, alone_out) = run(
        &mut e,
        &req(&system, "Name three animals, briefly.", "conv-c", 24),
    );
    // Not a chunk boundary: rows can differ from a fresh prefill by rounding, so this reports
    // rather than asserts.
    eprintln!(
        "part-way share: output {} prefilling alone",
        if shared_out == alone_out {
            "matches"
        } else {
            "differs from"
        }
    );
}

#[test]
fn the_pool_stays_within_its_budget() {
    let Some(dir) = model() else { return };
    let mut e = Engine::load(device(), &dir, 8192, Dtype::Bf16).unwrap();
    let block_bytes = e.model.cache_bytes(BLOCK);
    let max = 12;
    e.set_slots(4);
    e.set_kv_budget(Some(max * block_bytes));
    // Six conversations of ~3.5 blocks each: more than the budget holds.
    let mut first = None;
    for i in 0..6 {
        let system = filler(560, 10 + i);
        let r = req(&system, "Say ok.", &format!("conv-{i}"), 8);
        let (u, _) = run(&mut e, &r);
        first.get_or_insert(r);
        let s = e.kv_stats();
        eprintln!(
            "conversation {i}: {} prompt tokens, pool {s:?}",
            u.prompt_tokens
        );
        assert!(s.capacity <= max, "pool grew past its budget: {s:?}");
        assert!(e.kv_bytes() <= max * block_bytes);
    }
    // The first conversation's blocks were evicted to make room: it prefills again.
    let (u, _) = run(&mut e, &first.unwrap());
    assert_eq!(
        u.cached_tokens, 0,
        "the oldest blocks should have been evicted"
    );
}

#[test]
fn blocks_on_disk_survive_a_restart() {
    let Some(dir) = model() else { return };
    let disk = std::env::temp_dir().join(format!("tang-shared-prefix-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&disk);
    let load = || {
        let mut e = Engine::load(device(), &dir, 8192, Dtype::Bf16).unwrap();
        e.set_slots(4);
        e.set_kv_store(disk.clone(), 1 << 32).unwrap();
        e
    };
    let mut e = load();
    let system = system_prompt(&e);
    let a = req(&system, "Name three colours.", "conv-a", 24);
    run(&mut e, &a);
    e.save(Some("conv-a"));
    // The same turn again, warm in memory: what a restored conversation must reproduce.
    let (warm, warm_out) = run(&mut e, &a);
    assert_eq!(warm.cached_tokens, warm.prompt_tokens - 1);
    e.flush_kv_store();
    drop(e);

    // A new engine: conv-a's blocks and tail come back from disk.
    let mut e = load();
    let (u, out) = run(&mut e, &a);
    assert_eq!(u.cached_tokens, u.prompt_tokens - 1, "conv-a from disk");
    assert_eq!(out, warm_out, "restored conversation output differs");
    drop(e);

    // Another conversation sharing the prefix finds conv-a's blocks on disk.
    let mut e = load();
    let (u, _) = run(
        &mut e,
        &req(&system, "Name three animals, briefly.", "conv-b", 24),
    );
    assert_eq!(
        u.cached_tokens,
        8 * BLOCK,
        "conv-b from conv-a's blocks on disk"
    );
    let _ = std::fs::remove_dir_all(&disk);
}
