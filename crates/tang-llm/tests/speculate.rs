//! Speculative decoding must not change outputs: the same requests with drafting off and on,
//! greedy and seeded sampling, with thinking budgets, stop strings and length limits.
//!
//! Needs `TANG_LLM_TEST_MODEL` (e.g. `Qwen/Qwen3-0.6B`); skipped without it. Run with
//! `--release`.
//!
//! Exactness has two layers. The algorithm is exact: every token is sampled from the target's
//! logits in order with the same sampler state, and drafts only skip forwards. The arithmetic
//! is exact only where a k-token forward's rows are bit-identical to one-token forwards: on
//! Metal with bf16 weights that holds for k <= 4, so `bitwise_*` caps drafts at 3 there and
//! demands identical bytes. With longer drafts (and on other kernels), rows differ by ~1e-5 in
//! the logits, which can flip a near-tie; `default_drafts_match` checks the full-length
//! drafts anyway (they matched on Metal and CUDA when written).
//!
//! Both run with f32 KV caches. With bf16 caches (the default) the rows drift further apart:
//! a key the wider batch computes ~1e-6 differently can round to the neighbouring bf16 value,
//! which moves later logits by ~1e-3 (Qwen3-4B, k <= 4), enough to flip a near-tie now and
//! then (2 of these 8 requests). Both outputs are the model's; they just aren't the same bytes.
#![cfg(any(feature = "metal", feature = "cuda"))]

use serde_json::json;
use tang_compute::ComputeDevice;
use tang_llm::chat::Piece;
use tang_llm::draft::{DraftConfig, COST_METAL};
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

const CODE: &str = r#"use std::collections::HashMap;

/// Count how often each word appears, ignoring case and punctuation.
pub fn word_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let word: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if word.is_empty() {
            continue;
        }
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

/// The `n` most common words, most common first (ties broken alphabetically).
pub fn top_words(text: &str, n: usize) -> Vec<(String, usize)> {
    let mut all: Vec<(String, usize)> = word_counts(text).into_iter().collect();
    all.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    all.truncate(n);
    all
}
"#;

fn requests() -> Vec<(&'static str, Request)> {
    let greedy = Sampling {
        temperature: 0.0,
        ..Default::default()
    };
    let seeded = Sampling {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.95,
        seed: 42,
    };
    let req = |msg: String,
               think: bool,
               budget: Option<usize>,
               s: &Sampling,
               max: usize,
               stop: &[&str]| Request {
        messages: json!([{ "role": "user", "content": msg }]),
        images: vec![],
        tools: None,
        response_schema: None,
        think: Some(think),
        thinking_budget: budget,
        sampling: s.clone(),
        temperature_set: true,
        top_k_set: true,
        top_p_set: true,
        presence_penalty: None,
        max_tokens: Some(max),
        stop: stop.iter().map(|s| s.to_string()).collect(),
        cache_key: None,
        tenant: None,
        prefill_only: false,
    };
    let edit = format!(
        "Here is src/words.rs:\n\n```rust\n{CODE}```\n\nRename `word_counts` to `count_words` everywhere and output the whole updated file in a rust code block, nothing else."
    );
    vec![
        (
            "edit/greedy",
            req(edit.clone(), false, None, &greedy, 400, &[]),
        ),
        (
            "edit/seeded",
            req(edit.clone(), false, None, &seeded, 400, &[]),
        ),
        (
            "chat/greedy",
            req(
                "Write a short poem about a lighthouse keeper.".into(),
                false,
                None,
                &greedy,
                120,
                &[],
            ),
        ),
        (
            "chat/seeded",
            req(
                "Explain what a suffix automaton is in three sentences.".into(),
                false,
                None,
                &seeded,
                120,
                &[],
            ),
        ),
        (
            "think+budget/greedy",
            req(edit.clone(), true, Some(40), &greedy, 200, &[]),
        ),
        (
            "think+budget/seeded",
            req(
                "What is 17 * 23? Answer briefly.".into(),
                true,
                Some(24),
                &seeded,
                120,
                &[],
            ),
        ),
        (
            "stop/greedy",
            req(edit.clone(), false, None, &greedy, 300, &["top_words"]),
        ),
        (
            "short-limit/greedy",
            req(edit, false, None, &greedy, 37, &[]),
        ),
    ]
}

fn run<D: ComputeDevice>(e: &mut Engine<D>, r: &Request) -> (String, Usage, Vec<u32>) {
    e.reset();
    let mut out = String::new();
    let (finish, usage) = e
        .complete(r, |p| {
            match p {
                Piece::Reasoning(t) => out.push_str(&format!("[r]{t}")),
                Piece::Text(t) => out.push_str(&t),
                Piece::ToolCall { name, arguments } => {
                    out.push_str(&format!("[call {name} {arguments}]"))
                }
            }
            true
        })
        .unwrap();
    out.push_str(&format!("\n[finish {finish:?}]"));
    (out, usage, e.cached_tokens().to_vec())
}

fn device() -> impl ComputeDevice {
    #[cfg(feature = "metal")]
    return tang_compute::MetalDevice::new().expect("Metal");
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return tang_compute::CudaComputeDevice::new().expect("CUDA");
}

/// Off vs on for every request: (name, identical, off, on, usage on, caches identical).
fn compare(cfg: DraftConfig) -> Vec<(String, bool, String, String, Usage, bool)> {
    let dir = model().unwrap();
    let mut e = Engine::load(device(), &dir, 4096, Dtype::Bf16).unwrap();
    // f32 KV caches (see the module docs).
    e.set_kv_f32(true);
    let mut rows = Vec::new();
    for (name, r) in requests() {
        e.set_speculation(None);
        let (off, _, cache_off) = run(&mut e, &r);
        e.set_speculation(Some(cfg.clone()));
        let (on, u, cache_on) = run(&mut e, &r);
        eprintln!(
            "{name}: identical={} completion={} drafts {}/{} accepted",
            off == on,
            u.completion_tokens,
            u.accepted_tokens,
            u.draft_tokens
        );
        // Rolled back correctly: both caches hold the prompt and the generated tokens (the
        // last token may or may not have been fed).
        let (a, b) = (&cache_off, &cache_on);
        let cache_ok = (a.starts_with(b) || b.starts_with(a)) && a.len().abs_diff(b.len()) <= 1;
        rows.push((name.to_string(), off == on, off, on, u, cache_ok));
    }
    rows
}

fn first_diff(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

#[test]
fn bitwise_exact_with_short_drafts() {
    if model().is_none() {
        return;
    }
    let mut cfg = DraftConfig::new(COST_METAL);
    cfg.max_draft = 3;
    cfg.global_tokens = 0;
    let rows = compare(cfg);
    for (name, same, off, on, _, cache_same) in &rows {
        assert!(
            same,
            "{name}: outputs differ at byte {}\noff: {off}\non:  {on}",
            first_diff(off, on)
        );
        assert!(cache_same, "{name}: the KV cache holds different tokens");
    }
    let edit = &rows[0].4;
    assert!(
        edit.accepted_tokens > 50,
        "the edit task should accept many drafts: {edit:?}"
    );
}

#[test]
fn default_drafts_match() {
    if model().is_none() {
        return;
    }
    let rows = compare(DraftConfig::new(COST_METAL));
    let mut bad = Vec::new();
    for (name, same, off, on, _, cache_same) in &rows {
        if !same || !cache_same {
            let i = first_diff(off, on);
            eprintln!(
                "{name}: differs at byte {i} of {}: off {:?} / on {:?}",
                off.len(),
                &off[i.saturating_sub(40)..(i + 40).min(off.len())],
                &on[i.saturating_sub(40)..(i + 40).min(on.len())]
            );
            bad.push(name.clone());
        }
    }
    assert!(bad.is_empty(), "outputs differ: {bad:?}");
}

/// Every request decoded together: with `stagger`, half join first and the rest a few steps
/// later (so they prefill while the first half decodes). Outputs as `run` formats them.
fn run_batch<D: ComputeDevice>(e: &mut Engine<D>, reqs: &[Request], stagger: bool) -> Vec<String> {
    use tang_llm::engine::{Control, Lane, Outcome, Stream};
    e.reset();
    let mut outs = vec![String::new(); reqs.len()];
    let mut streams: Vec<Option<Stream<_>>> = (0..reqs.len()).map(|_| None).collect();
    let mut finished = vec![false; reqs.len()];
    let first = if stagger { reqs.len() / 2 } else { reqs.len() };
    for i in 0..first {
        streams[i] = Some(e.admit(&reqs[i], None).unwrap());
    }
    let mut steps = 0;
    while finished.iter().any(|f| !f) {
        steps += 1;
        if steps == 4 {
            for i in first..reqs.len() {
                streams[i] = Some(e.admit(&reqs[i], None).unwrap());
            }
        }
        let mut ons: Vec<Box<dyn FnMut(Piece) -> bool>> = Vec::new();
        let mut refs: Vec<&mut Stream<_>> = Vec::new();
        let mut live: Vec<usize> = Vec::new();
        for (i, (out, s)) in outs.iter_mut().zip(streams.iter_mut()).enumerate() {
            let Some(s) = s else { continue };
            live.push(i);
            refs.push(s);
            ons.push(Box::new(move |p| {
                match p {
                    Piece::Reasoning(t) => out.push_str(&format!("[r]{t}")),
                    Piece::Text(t) => out.push_str(&t),
                    Piece::ToolCall { name, arguments } => {
                        out.push_str(&format!("[call {name} {arguments}]"))
                    }
                }
                true
            }));
        }
        let mut ctls = vec![(); live.len()];
        let mut lanes: Vec<Lane<_>> = refs
            .iter_mut()
            .zip(ons.iter_mut())
            .zip(ctls.iter_mut())
            .map(|((s, on), ctl)| Lane {
                stream: &mut **s,
                on: &mut **on,
                ctl: ctl as &mut dyn Control,
            })
            .collect();
        e.step(&mut lanes).unwrap();
        drop(lanes);
        drop(refs);
        drop(ons);
        for &i in &live {
            if streams[i].as_ref().unwrap().done() {
                let s = streams[i].take().unwrap();
                match e.retire(s).unwrap() {
                    Outcome::Done(finish, _) => outs[i].push_str(&format!("\n[finish {finish:?}]")),
                    o => panic!("{o:?}"),
                }
                finished[i] = true;
            }
        }
        assert!(steps < 100_000);
    }
    outs
}

/// Batched decoding (and batched speculation, and prefill joining a decoding batch) gives
/// each request what it gets alone.
#[test]
fn batched_matches_alone() {
    let Some(dir) = model() else { return };
    let mut e = Engine::load(device(), &dir, 4096, Dtype::Bf16).unwrap();
    e.set_kv_f32(true);
    e.set_slots(8);
    let reqs: Vec<(&str, Request)> = requests();
    let names: Vec<&str> = reqs.iter().map(|r| r.0).collect();
    let reqs: Vec<Request> = reqs.into_iter().map(|r| r.1).collect();
    let mut short = DraftConfig::new(COST_METAL);
    short.max_draft = 3;
    short.global_tokens = 0;
    let mut bad = Vec::new();
    for (label, cfg) in [("plain", None), ("speculating", Some(short))] {
        e.set_speculation(cfg.clone());
        let alone: Vec<String> = reqs.iter().map(|r| run(&mut e, r).0).collect();
        for stagger in [false, true] {
            e.set_speculation(cfg.clone());
            let together = run_batch(&mut e, &reqs, stagger);
            for (i, (a, b)) in alone.iter().zip(&together).enumerate() {
                let same = a == b;
                eprintln!("{label} stagger={stagger} {}: identical={same}", names[i]);
                if !same {
                    let j = first_diff(a, b);
                    eprintln!(
                        "  differs at byte {j}: alone {:?} / batched {:?}",
                        &a[j.saturating_sub(30)..(j + 30).min(a.len())],
                        &b[j.saturating_sub(30)..(j + 30).min(b.len())]
                    );
                    bad.push(format!("{label}/{stagger}/{}", names[i]));
                }
            }
        }
    }
    assert!(bad.is_empty(), "batched outputs differ: {bad:?}");
}

/// A Qwen3-VL checkpoint (`TANG_LLM_TEST_VL_MODEL`, e.g. the tiny one `qwen3vl_tiny.py`
/// writes, or Qwen3-VL-8B-Instruct); the image tests are skipped without it.
fn vl_model() -> Option<std::path::PathBuf> {
    let Ok(spec) = std::env::var("TANG_LLM_TEST_VL_MODEL") else {
        eprintln!("TANG_LLM_TEST_VL_MODEL not set: skipping");
        return None;
    };
    Some(tang_llm::resolve_model(&spec).expect("TANG_LLM_TEST_VL_MODEL"))
}

/// A small synthetic picture as a PNG data URL.
fn png(w: u32, h: u32, seed: u32) -> String {
    use base64::Engine as _;
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        let v = (x * 7 + y * 13 + seed * 31) % 256;
        if (x / 24 + y / 24 + seed).is_multiple_of(2) {
            image::Rgb([v as u8, 90, 200 - (v as u8) / 2])
        } else {
            image::Rgb([30, (v as u8) / 2 + 60, 40])
        }
    });
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    let data = base64::engine::general_purpose::STANDARD.encode(out.into_inner());
    format!("data:image/png;base64,{data}")
}

fn vl_requests() -> Vec<(&'static str, Request)> {
    let body = |content: serde_json::Value, extra: serde_json::Value| {
        let mut b = json!({"messages": [{"role": "user", "content": content}], "max_tokens": 80});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        tang_llm::server::parse(&b).unwrap()
    };
    let img = |w, h, s| json!({"type": "image_url", "image_url": {"url": png(w, h, s)}});
    let text = |t: &str| json!({"type": "text", "text": t});
    let greedy = json!({"temperature": 0.0});
    let seeded = json!({"temperature": 0.7, "top_k": 20, "top_p": 0.95, "seed": 42});
    vec![
        (
            "image/greedy",
            body(
                json!([img(330, 200, 0), text("Describe this view.")]),
                greedy.clone(),
            ),
        ),
        (
            "two-images/seeded",
            body(
                json!([
                    img(260, 300, 1),
                    text("and"),
                    img(300, 260, 2),
                    text("What changed?")
                ]),
                seeded,
            ),
        ),
        (
            "text-only/greedy",
            body(json!("Name three CAD file formats."), greedy),
        ),
    ]
}

/// With images in the prompt (M-RoPE rows in the cache, DeepStack), speculative decoding and
/// batching still give each request exactly what plain decoding gives it alone.
#[test]
fn images_drafts_and_batching_match() {
    let Some(dir) = vl_model() else { return };
    let mut e = Engine::load(device(), &dir, 4096, Dtype::Bf16).unwrap();
    e.set_kv_f32(true);
    e.set_slots(8);
    let reqs = vl_requests();
    let mut short = DraftConfig::new(COST_METAL);
    short.max_draft = 3;
    short.global_tokens = 0;
    let mut bad = Vec::new();
    let mut alone = Vec::new();
    for (name, r) in &reqs {
        e.set_speculation(None);
        let (off, _, _) = run(&mut e, r);
        e.set_speculation(Some(short.clone()));
        let (on, u, _) = run(&mut e, r);
        eprintln!(
            "{name}: identical={} completion={} drafts {}/{} accepted",
            off == on,
            u.completion_tokens,
            u.accepted_tokens,
            u.draft_tokens
        );
        if off != on {
            bad.push(format!("drafts/{name} at byte {}", first_diff(&off, &on)));
        }
        alone.push(off);
    }
    let rs: Vec<Request> = vl_requests().into_iter().map(|r| r.1).collect();
    for spec in [None, Some(short)] {
        for stagger in [false, true] {
            e.set_speculation(spec.clone());
            let together = run_batch(&mut e, &rs, stagger);
            for (i, (a, b)) in alone.iter().zip(&together).enumerate() {
                if a != b {
                    bad.push(format!(
                        "batch/{}/{stagger}/{} at byte {}",
                        spec.is_some(),
                        reqs[i].0,
                        first_diff(a, b)
                    ));
                }
            }
        }
    }
    assert!(bad.is_empty(), "outputs differ: {bad:?}");
}
