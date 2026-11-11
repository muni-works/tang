//! `tang-llm serve <first shard.gguf> --mtp <mtp.gguf>`: the Flash-Next engine behind the
//! OpenAI-compatible server (`crate::server`), for agents like frog. The shared server keeps
//! priority queue ordering and `/node` rates; this fixed GGUF backend does not swap models,
//! yield a background prefill, or report dense-engine KV blocks.
//!
//! - **Prompt.** The GGUF's own chat template. An assistant message this server generated is
//!   spliced back in as the exact tokens it generated (reasoning included, as the template's
//!   `preserve_thinking` wants), found by its visible text and tool calls; so a follow-up's
//!   prompt extends the live sequence token for token.
//! - **Prefix reuse.** The engine keeps the live sequence. A prompt that extends it prefills
//!   only the new tokens; otherwise the longest saved running state (`RunningState`, ~118 MB
//!   in host RAM) that the prompt extends is restored: taken at the end of the system prompt
//!   (pinned, and saved to disk so a restart doesn't pay for it again), at each prompt's last
//!   turn boundary, and at the end of each request.
//! - **Decode.** MTP drafts, priced per width like `flash-generate --draft mtp`, verified with
//!   exact match, so the output is the engine's sampling at `Philox(seed, position)` whatever
//!   the drafts. Temperature only (no top-k/top-p); greedy unless the request names one.
//! - **Tool calls.** Qwen XML (`chat::parse_xml_call`), typed by the request's schemas.

#[path = "serve_batch.rs"]
mod batch;

use super::engine::{Engine, Opts, Positional, RunningState};
use super::tokenize::FlashTokenizer;
use crate::chat::{Chunk, Detok, Parser, Piece, Tags};
use crate::engine::{Control, Finish, Outcome, Request, ThinkBudget, Usage, WRAP_UP};
use crate::server::Backend;
use crate::special::Shield;
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tang_compute::flash::MAX_T;

/// What sampling a request gets for the parameters it doesn't name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplingMode {
    /// Temperature 0 (greedy); top-k/top-p/penalty only when asked for.
    Greedy,
    /// The model card's: thinking temperature 1.0, top-p 0.95, top-k 20; not thinking 0.7,
    /// 0.8, 20 and a presence penalty of 1.5.
    ModelCard,
}

pub struct Settings {
    pub sampling: SamplingMode,
    pub gguf: PathBuf,
    pub mtp: Option<PathBuf>,
    pub ctx: usize,
    /// Temperature for requests that don't name one (default 0: greedy).
    pub temp: f32,
    /// Saved states kept in RAM (besides the pinned system-prompt ones), and their total size.
    pub snapshots: usize,
    pub snapshot_bytes: usize,
    /// Where pinned (system prompt) states are saved; `None` keeps them in RAM only.
    pub disk: Option<PathBuf>,
    /// Write each request's prompt and output ids here (`req-N.ids`, `req-N.out`).
    pub dump: Option<PathBuf>,
    /// Chat completion bodies (JSON files) to prefill at start, so their tools block and
    /// system prompt states are saved before the first real request.
    pub warm: Vec<PathBuf>,
    /// Suffix drafts (the n-gram drafter over the sequence) for an engine without MTP.
    pub suffix: bool,
    /// Learned pre-warm: each distinct tools block a client sends is recorded here as a minimal
    /// request body (`tools-<hash>.json`), and every body here is prefilled at start, so a
    /// client's first turn after a restart reuses its tools block.
    pub warm_dir: Option<PathBuf>,
    /// Continuous batching (`serve_batch.rs`): up to this many requests decode together, on an
    /// engine with sequence slots (qwen3next). 0 or 1: one request at a time.
    pub batch: usize,
    /// Seed the expert cache from this routing profile. Without one, `routing.bin` in `disk`
    /// (written after every request) seeds it.
    pub profile: Option<PathBuf>,
}

/// A request's (temperature, top-k, top-p, presence penalty): what it names, the mode's
/// defaults for the rest.
fn request_sampling(mode: SamplingMode, temp0: f32, req: &Request) -> (f32, usize, f32, f32) {
    let thinking = req.think != Some(false);
    let card = match (mode, thinking) {
        (SamplingMode::Greedy, _) => (temp0, 0usize, 1.0f32, 0.0f32),
        (SamplingMode::ModelCard, true) => (1.0, 20, 0.95, 0.0),
        (SamplingMode::ModelCard, false) => (0.7, 20, 0.8, 1.5),
    };
    let temp = if req.temperature_set {
        req.sampling.temperature.max(0.0)
    } else {
        card.0
    };
    let top_k = if req.top_k_set {
        req.sampling.top_k
    } else {
        card.1
    };
    let top_p = if req.top_p_set {
        req.sampling.top_p
    } else {
        card.2
    };
    (temp, top_k, top_p, req.presence_penalty.unwrap_or(card.3))
}

/// What the server needs from an engine: the Flash-Next [`Engine`] or the qwen3next
/// [`Coder`](super::coder::Coder).
pub trait ServeEngine {
    fn tokens(&self) -> &Vec<u32>;
    fn tokens_mut(&mut self) -> &mut Vec<u32>;
    fn reset(&mut self);
    fn gguf(&self) -> &crate::gguf::Gguf;
    fn last(&self) -> super::engine::WinStats;
    fn save_running(&mut self) -> Result<RunningState>;
    fn restore_running(&mut self, s: &RunningState, prefix: &[u32]) -> Result<()>;
    fn save_positional(&mut self, from: usize, to: usize) -> Result<Positional>;
    fn restore_positional(&mut self, p: &Positional, from: usize) -> Result<()>;
    fn prefill(&mut self, ids: &[u32], chunk: usize) -> Result<u32>;
    fn window(&mut self, pos0: usize, t: usize) -> Result<Vec<u32>>;
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>>;
    /// The last window's logits, `[t][vocab]`.
    fn logits(&self, t: usize) -> Vec<f32>;
    fn set_sampling(&mut self, temperature: f32, seed: u32);
    fn sampler_mut(&mut self) -> Option<&mut super::sampler::Sampler>;
    fn sampler_begin(&mut self, pos: usize) -> Result<()>;
    /// Routing counts per expert key (the Flash-Next expert cache's profile).
    fn routing(&mut self) -> Option<&mut Vec<u32>> {
        None
    }
    fn save_routing(&self, _path: &Path) -> Result<()> {
        Ok(())
    }
    /// Wide prefill windows (`Engine::prefill_wide`).
    fn has_wide(&self) -> bool {
        false
    }
    fn prefill_wide(&mut self, _pos0: usize, _t: usize) -> Result<super::engine::WinStats> {
        bail!("no wide prefill")
    }
    fn use_mtp(&self) -> bool {
        false
    }
    fn mtp_last(&self) -> &[(u32, f32)] {
        &[]
    }
    fn mtp_draft(&mut self, _pos0: usize, _next: &[u32]) -> Result<()> {
        bail!("no MTP")
    }
    fn last_mtp_ms(&self) -> f64 {
        0.0
    }
    /// Sequence slots for batching (`--batch`); 1: none.
    fn max_slots(&self) -> usize {
        1
    }
    fn slot_tokens(&self, _slot: usize) -> &[u32] {
        &[]
    }
    fn alloc_slot(&mut self) -> Result<usize> {
        bail!("this engine has no sequence slots")
    }
    /// Sequence slots that fit in VRAM at the configured context (see `Coder::fit_slots`).
    fn fit_slots(&mut self) -> usize {
        self.max_slots().saturating_sub(1).max(1)
    }
    fn free_slot(&mut self, _slot: usize) {}
    fn prefill_slot(&mut self, _slot: usize, _ids: &[u32]) -> Result<u32> {
        bail!("this engine has no sequence slots")
    }
    fn decode_batch(&mut self, _rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        bail!("this engine has no sequence slots")
    }
    /// Per-row sampling in slot windows: turn it on (startup, `--batch`).
    fn enable_row_sampling(&mut self) -> Result<()> {
        Ok(())
    }
    /// Slot `slot`'s sampling for its next request. Engines without per-row sampling refuse a
    /// sampled one rather than run it greedy.
    fn set_slot_sampling(&mut self, _slot: usize, s: super::sampler::RowSampling) -> Result<()> {
        ensure!(
            !s.sampled(),
            "this engine's batched windows are greedy: temperature {} isn't supported with --batch",
            s.temperature
        );
        Ok(())
    }
    /// Wide prefill windows for sequence slots: build them (`rows` tokens a window).
    fn enable_wide_rows(&mut self, _rows: usize) -> Result<()> {
        bail!("this engine has no wide slot prefill")
    }
    /// The wide windows' width (0: none).
    fn wide_rows(&self) -> usize {
        0
    }
    /// Batch decode windows past `MAX_T` rows: build them (`rows` streams a window).
    fn enable_batch_rows(&mut self, _rows: usize) -> Result<()> {
        bail!("this engine has no batch windows")
    }
    /// The batch decode windows' width (0: none; decode then runs `MAX_T` rows a window).
    fn batch_rows(&self) -> usize {
        0
    }
    /// [`prefill_slot`](Self::prefill_slot) through wide windows.
    fn prefill_slot_wide(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        self.prefill_slot(slot, ids)
    }
    /// [`save_running`](Self::save_running) of a sequence slot (0: the single sequence).
    fn save_running_at(&mut self, slot: usize) -> Result<RunningState> {
        ensure!(slot == 0, "this engine has no sequence slots");
        self.save_running()
    }
    fn restore_running_at(&mut self, slot: usize, s: &RunningState, prefix: &[u32]) -> Result<()> {
        ensure!(slot == 0, "this engine has no sequence slots");
        self.restore_running(s, prefix)
    }
    fn save_positional_at(&mut self, slot: usize, from: usize, to: usize) -> Result<Positional> {
        ensure!(slot == 0, "this engine has no sequence slots");
        self.save_positional(from, to)
    }
    fn restore_positional_at(&mut self, slot: usize, p: &Positional, from: usize) -> Result<()> {
        ensure!(slot == 0, "this engine has no sequence slots");
        self.restore_positional(p, from)
    }
    /// One verify window for several slots: `(slot, token, drafts)` each; returns each one's
    /// kept tokens (`Coder::verify_batch`).
    fn verify_batch(&mut self, _items: &[(usize, u32, Vec<u32>)]) -> Result<Vec<Vec<u32>>> {
        bail!("this engine has no sequence slots")
    }
    /// A slot's latest MTP chain (tokens and the head's probabilities), for its next verify.
    fn slot_drafts(&self, _slot: usize) -> Vec<(u32, f32)> {
        Vec::new()
    }
    /// Run the MTP head after slot windows (`Coder::slot_mtp`).
    fn set_slot_mtp(&mut self, _on: bool) {}
    /// Fill the MTP K/V over wide prefill windows (`Coder::mtp_fill`).
    fn set_mtp_fill(&mut self, _on: bool) {}
    /// Logit rows' width (0: this engine has no structured output).
    fn vocab_size(&self) -> usize {
        0
    }
    /// A no-draft verify of `cur` that also returns the row's logits, for structured output.
    fn step_logits(&mut self, _cur: u32) -> Result<(u32, Vec<f32>)> {
        bail!("structured output needs the qwen3next engine")
    }
    /// `decode_batch` that also returns the window's logits (`[rows][vocab]`).
    fn decode_batch_logits(&mut self, _rows: &[(usize, u32)]) -> Result<(Vec<u32>, Vec<f32>)> {
        bail!("structured output needs the qwen3next engine")
    }
    /// Verify-window cost by width relative to one token, for pricing suffix drafts.
    fn draft_cost(&self) -> Vec<(usize, f32)> {
        vec![(1, 1.0), (2, 1.12), (4, 1.35), (8, 1.9)]
    }
}

impl ServeEngine for Engine {
    fn vocab_size(&self) -> usize {
        self.hp.n_vocab
    }
    fn tokens(&self) -> &Vec<u32> {
        &self.tokens
    }
    fn tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.tokens
    }
    fn reset(&mut self) {
        Engine::reset(self)
    }
    fn gguf(&self) -> &crate::gguf::Gguf {
        Engine::gguf(self)
    }
    fn last(&self) -> super::engine::WinStats {
        self.last
    }
    fn save_running(&mut self) -> Result<RunningState> {
        Engine::save_running(self)
    }
    fn restore_running(&mut self, s: &RunningState, prefix: &[u32]) -> Result<()> {
        Engine::restore_running(self, s, prefix)
    }
    fn save_positional(&mut self, from: usize, to: usize) -> Result<Positional> {
        Engine::save_positional(self, from, to)
    }
    fn restore_positional(&mut self, p: &Positional, from: usize) -> Result<()> {
        Engine::restore_positional(self, p, from)
    }
    fn prefill(&mut self, ids: &[u32], chunk: usize) -> Result<u32> {
        Engine::prefill(self, ids, chunk, None)
    }
    fn window(&mut self, pos0: usize, t: usize) -> Result<Vec<u32>> {
        Engine::window(self, pos0, t, None)
    }
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        Engine::verify(self, cur, drafts)
    }
    fn logits(&self, t: usize) -> Vec<f32> {
        Engine::logits(self, t)
    }
    fn set_sampling(&mut self, temperature: f32, seed: u32) {
        self.temperature = temperature;
        self.seed = seed;
    }
    fn sampler_mut(&mut self) -> Option<&mut super::sampler::Sampler> {
        Engine::sampler_mut(self)
    }
    fn sampler_begin(&mut self, pos: usize) -> Result<()> {
        Engine::sampler_begin(self, pos)
    }
    fn routing(&mut self) -> Option<&mut Vec<u32>> {
        Some(&mut self.routing)
    }
    fn save_routing(&self, path: &Path) -> Result<()> {
        Engine::save_routing(self, path)
    }
    fn has_wide(&self) -> bool {
        true
    }
    fn prefill_wide(&mut self, pos0: usize, t: usize) -> Result<super::engine::WinStats> {
        Engine::prefill_wide(self, pos0, t)
    }
    fn use_mtp(&self) -> bool {
        self.use_mtp && self.has_mtp()
    }
    fn mtp_last(&self) -> &[(u32, f32)] {
        &self.mtp_last
    }
    fn mtp_draft(&mut self, pos0: usize, next: &[u32]) -> Result<()> {
        self.mtp_last = Engine::mtp_draft(self, pos0, next)?;
        Ok(())
    }
    fn last_mtp_ms(&self) -> f64 {
        self.last_mtp_ms
    }
}

impl ServeEngine for super::coder::Coder {
    fn vocab_size(&self) -> usize {
        self.hp.n_vocab
    }
    fn step_logits(&mut self, cur: u32) -> Result<(u32, Vec<f32>)> {
        super::coder::Coder::step_logits(self, cur)
    }
    fn decode_batch_logits(&mut self, rows: &[(usize, u32)]) -> Result<(Vec<u32>, Vec<f32>)> {
        let (out, logits) = super::coder::Coder::decode_batch_keep(self, rows, true)?;
        Ok((out, logits.unwrap_or_default()))
    }
    fn tokens(&self) -> &Vec<u32> {
        &self.tokens
    }
    fn tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.tokens
    }
    fn reset(&mut self) {
        super::coder::Coder::reset(self)
    }
    fn gguf(&self) -> &crate::gguf::Gguf {
        super::coder::Coder::gguf(self)
    }
    fn last(&self) -> super::engine::WinStats {
        self.last
    }
    fn save_running(&mut self) -> Result<RunningState> {
        super::coder::Coder::save_running(self)
    }
    fn restore_running(&mut self, s: &RunningState, prefix: &[u32]) -> Result<()> {
        super::coder::Coder::restore_running(self, s, prefix)
    }
    fn save_positional(&mut self, from: usize, to: usize) -> Result<Positional> {
        super::coder::Coder::save_positional(self, from, to)
    }
    fn restore_positional(&mut self, p: &Positional, from: usize) -> Result<()> {
        super::coder::Coder::restore_positional(self, p, from)
    }
    fn prefill(&mut self, ids: &[u32], chunk: usize) -> Result<u32> {
        super::coder::Coder::prefill(self, ids, chunk, None)
    }
    fn window(&mut self, pos0: usize, t: usize) -> Result<Vec<u32>> {
        super::coder::Coder::window(self, pos0, t, None)
    }
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        super::coder::Coder::verify(self, cur, drafts)
    }
    fn logits(&self, t: usize) -> Vec<f32> {
        super::coder::Coder::logits(self, t)
    }
    fn set_sampling(&mut self, temperature: f32, seed: u32) {
        self.temperature = temperature;
        self.seed = seed;
    }
    fn sampler_mut(&mut self) -> Option<&mut super::sampler::Sampler> {
        super::coder::Coder::sampler_mut(self)
    }
    fn sampler_begin(&mut self, pos: usize) -> Result<()> {
        super::coder::Coder::sampler_begin(self, pos)
    }
    fn max_slots(&self) -> usize {
        super::coder::MAX_SLOTS
    }
    fn slot_tokens(&self, slot: usize) -> &[u32] {
        super::coder::Coder::slot_tokens(self, slot)
    }
    fn alloc_slot(&mut self) -> Result<usize> {
        super::coder::Coder::alloc_slot(self)
    }
    fn fit_slots(&mut self) -> usize {
        super::coder::Coder::fit_slots(self)
    }
    fn free_slot(&mut self, slot: usize) {
        super::coder::Coder::free_slot(self, slot)
    }
    fn prefill_slot(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        super::coder::Coder::prefill_slot(self, slot, ids)
    }
    fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        super::coder::Coder::decode_batch(self, rows)
    }
    fn enable_row_sampling(&mut self) -> Result<()> {
        super::coder::Coder::enable_row_sampling(self)
    }
    fn set_slot_sampling(&mut self, slot: usize, s: super::sampler::RowSampling) -> Result<()> {
        super::coder::Coder::set_slot_sampling(self, slot, s)
    }
    fn enable_batch_rows(&mut self, rows: usize) -> Result<()> {
        super::coder::Coder::enable_batch(self, rows)
    }
    fn batch_rows(&self) -> usize {
        super::coder::Coder::batch_rows(self)
    }
    fn enable_wide_rows(&mut self, rows: usize) -> Result<()> {
        super::coder::Coder::enable_wide(self, rows)
    }
    fn wide_rows(&self) -> usize {
        super::coder::Coder::wide_rows(self)
    }
    fn prefill_slot_wide(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        super::coder::Coder::prefill_wide(self, slot, ids)
    }
    fn save_running_at(&mut self, slot: usize) -> Result<RunningState> {
        super::coder::Coder::save_running_at(self, slot)
    }
    fn restore_running_at(&mut self, slot: usize, s: &RunningState, prefix: &[u32]) -> Result<()> {
        super::coder::Coder::restore_running_at(self, slot, s, prefix)
    }
    fn save_positional_at(&mut self, slot: usize, from: usize, to: usize) -> Result<Positional> {
        super::coder::Coder::save_positional_at(self, slot, from, to)
    }
    fn restore_positional_at(&mut self, slot: usize, p: &Positional, from: usize) -> Result<()> {
        super::coder::Coder::restore_positional_at(self, slot, p, from)
    }
    fn verify_batch(&mut self, items: &[(usize, u32, Vec<u32>)]) -> Result<Vec<Vec<u32>>> {
        super::coder::Coder::verify_batch(self, items)
    }
    fn slot_drafts(&self, slot: usize) -> Vec<(u32, f32)> {
        super::coder::Coder::slot_drafts(self, slot).to_vec()
    }
    fn set_slot_mtp(&mut self, on: bool) {
        self.slot_mtp = on;
    }
    fn set_mtp_fill(&mut self, on: bool) {
        self.mtp_fill = on;
    }
    fn use_mtp(&self) -> bool {
        self.use_mtp
    }
    fn mtp_last(&self) -> &[(u32, f32)] {
        &self.mtp_last
    }
    fn mtp_draft(&mut self, pos0: usize, next: &[u32]) -> Result<()> {
        super::coder::Coder::mtp_feed(self, pos0, next)
    }
    fn last_mtp_ms(&self) -> f64 {
        self.last_mtp_ms
    }
    fn draft_cost(&self) -> Vec<(usize, f32)> {
        // H200, UD-Q4_K_XL: a T=8 verify window is ~3.3x a T=1 one (each token routes to its
        // own experts), so long drafts only pay when they're very likely.
        vec![(1, 1.0), (2, 1.35), (4, 1.9), (8, 3.3)]
    }
}

/// A saved running state and the tokens it follows.
struct Snap {
    tokens: Vec<u32>,
    state: Arc<RunningState>,
    /// Positions `0..tokens.len()` of the positional state (another sequence may have
    /// overwritten them on the device since), in segments shared with the states it extends.
    pos: Vec<Arc<Positional>>,
    pinned: bool,
    used: u64,
    /// Last used, by the wall clock (see [`crate::retention`]).
    at: std::time::SystemTime,
}

/// A completion this server generated, to splice back into later prompts.
struct Memo {
    content: String,
    calls: Vec<(String, Value)>,
    /// The generation prompt's tail after `<|im_start|>assistant\n` (`<think>\n`, or the
    /// closed empty block with thinking off).
    tail: String,
    ids: Vec<u32>,
    at: std::time::SystemTime,
}

/// Draft pricing (the `flash-generate --draft mtp` policy): expected tokens per ms of window
/// (EMA per width) and drafting, with acceptance calibrated by draft-probability bucket.
struct Policy {
    wcost: Vec<f64>,
    mtp_cost: f64,
    calib: Vec<(f64, f64)>,
    think_room: usize,
    gate: Option<f32>,
}

impl Policy {
    fn new() -> Self {
        Policy {
            wcost: vec![10.6, 10.6, 12.9, 15.0, 17.1, 19.2, 21.3, 23.4, 25.5],
            mtp_cost: 2.5,
            calib: (0..10)
                .map(|b| {
                    let p = (b as f64 + 0.5) / 10.0;
                    let a = if p >= 0.9 {
                        0.985
                    } else if p >= 0.5 {
                        0.6 + (p - 0.5) * 0.8
                    } else {
                        0.3 + p * 0.6
                    };
                    (4.0 * a, 4.0)
                })
                .collect(),
            think_room: std::env::var("TANG_FLASH_THINK_STEPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(MAX_T),
            gate: std::env::var("TANG_FLASH_GATE")
                .ok()
                .and_then(|v| v.parse().ok()),
        }
    }

    /// Drafts (and their probabilities) from the MTP chain, at most `room`.
    fn choose(&self, chain: &[(u32, f32)], room: usize) -> (Vec<u32>, Vec<f32>) {
        let n = match self.gate {
            Some(g) => chain.iter().take(room).take_while(|x| x.1 >= g).count(),
            None => {
                let mut best = (0usize, 1.0 / (self.wcost[1] + self.mtp_cost));
                let (mut cum, mut exp) = (1.0f64, 1.0f64);
                for (k, &(_, p)) in chain.iter().enumerate().take(room) {
                    let b = ((p * 10.0) as usize).min(9);
                    cum *= self.calib[b].0 / self.calib[b].1;
                    exp += cum;
                    let rate = exp / (self.wcost[k + 2] + self.mtp_cost);
                    if rate > best.1 {
                        best = (k + 1, rate);
                    }
                }
                best.0
            }
        };
        (
            chain.iter().take(n).map(|x| x.0).collect(),
            chain.iter().take(n).map(|x| x.1).collect(),
        )
    }

    fn observe(&mut self, probs: &[f32], accepted: usize, width: usize, wall_ms: f64, mtp_ms: f64) {
        for (j, &p) in probs.iter().enumerate().take(accepted + 1) {
            let b = ((p * 10.0) as usize).min(9);
            self.calib[b].1 += 1.0;
            if j < accepted {
                self.calib[b].0 += 1.0;
            }
        }
        self.wcost[width] = 0.9 * self.wcost[width] + 0.1 * wall_ms;
        self.mtp_cost = 0.9 * self.mtp_cost + 0.1 * mtp_ms;
    }
}

/// What one request did, for the server log.
#[derive(Default)]
struct Line {
    prompt: usize,
    reused: usize,
    prefill_ms: f64,
    restore_ms: f64,
    snap_ms: f64,
    ttft_ms: f64,
    decode: usize,
    decode_s: f64,
    windows: usize,
    drafted: usize,
    accepted: usize,
    calls: usize,
    finish: &'static str,
    spliced: usize,
    sampling: String,
}

pub struct FlashServe {
    gguf: PathBuf,
    e: Box<dyn ServeEngine>,
    /// Suffix drafts (the n-gram drafter over the sequence) when the engine has no MTP: its
    /// config, the completions it also matches against (across requests), and its hit rates.
    suffix: bool,
    dcfg: crate::draft::DraftConfig,
    dglobal: crate::draft::Global,
    dcalib: crate::draft::Calibration,
    /// See [`Settings::warm_dir`].
    warm_dir: Option<PathBuf>,
    /// See [`Settings::batch`] (1: off), and the engine's sequence slots, parked or running.
    batch: usize,
    /// `TANG_SERVE_DETERMINISTIC=1`: every window on the narrow (bitwise) path, at most
    /// `MAX_T` rows, so a stream's greedy output doesn't depend on what it's batched with.
    deterministic: bool,
    /// Speculative decoding under batching: the draft budget's table (`crate::batch::spec`).
    spec: crate::batch::spec::Tuner,
    /// Spec-decode counters since the last summary line (steps, extra rows, drafted,
    /// accepted, tokens, ms) and when it was printed.
    spec_log: ([f64; 6], Instant),
    slots: Vec<batch::SlotInfo>,
    /// The batch prefill scheduler's account and knobs (see `batch_step`).
    sched: crate::batch::prefill::Sched,
    prefill_policy: Option<crate::batch::prefill::Policy>,
    tok: FlashTokenizer,
    /// Keeps request text out of the special tokens (see [`Shield`]).
    shield: Shield,
    /// Tag tokens the output parser splits on.
    tags: Tags,
    ctx: usize,
    eos: Vec<u32>,
    /// Structured output's grammar compiler over this vocabulary (built on first use).
    structured: Option<llguidance::ParserFactory>,
    think: (u32, u32),
    im_start: u32,
    wrap_up: Vec<u32>,
    snaps: Vec<Snap>,
    max_snaps: usize,
    max_snap_bytes: usize,
    memo: VecDeque<Memo>,
    policy: Policy,
    clock: u64,
    /// When the live sequence last ran (for the TTL).
    live_at: std::time::SystemTime,
    temp: f32,
    sampling: SamplingMode,
    /// Widest prefill window (`TANG_FLASH_WIDE`, default 64; 0: decode-width windows only).
    wide: usize,
    disk: Option<PathBuf>,
    dump: Option<PathBuf>,
    /// Lengths of a running state's parts, to check states read from disk.
    shape: Vec<usize>,
    gdn_layers: usize,
    requests: usize,
    line: Line,
    drafts_mode: &'static str,
}

/// Tool-call arguments as objects (the template wants mappings; OpenAI clients send strings).
fn args_as_objects(msgs: &mut Value) {
    for m in msgs.as_array_mut().into_iter().flatten() {
        for c in m["tool_calls"].as_array_mut().into_iter().flatten() {
            let f = if c["function"].is_object() {
                &mut c["function"]
            } else {
                &mut *c
            };
            if let Some(a) = f["arguments"].as_str() {
                f["arguments"] = serde_json::from_str(a)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| json!({}));
            }
        }
    }
}

const MARK: &str = "\u{F8FF}tang-splice\u{F8FF}";

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn hash(ids: &[u32]) -> u64 {
    // FNV-1a: stable across runs (the disk cache is keyed by it).
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &i in ids {
        for b in i.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

impl FlashServe {
    /// `want` streams at once, clamped to the slots that fit in VRAM at `--ctx` (one spare,
    /// see [`crate::batch::clamp_batch`]) and to `ceiling` (what the windows were built for);
    /// warns when it cuts, unless `quiet` (the provisional count before the windows exist).
    fn clamp_batch(&mut self, want: usize, ceiling: usize, quiet: bool) -> usize {
        let cap = self.e.fit_slots();
        let n = crate::batch::clamp_batch(want, self.e.max_slots(), cap).min(ceiling);
        if n < want && !quiet {
            eprintln!(
                "tang-llm: warning: --batch {want} clamped to {n}: {cap} sequence slots fit in \
                 free VRAM at --ctx {} after the batch and wide windows \
                 (TANG_SLOT_HEADROOM_GB sets the headroom)",
                self.ctx
            );
        }
        n
    }

    fn request_sampling(&self, req: &Request) -> (f32, usize, f32, f32) {
        request_sampling(self.sampling, self.temp, req)
    }

    pub fn load(s: &Settings) -> Result<Self> {
        let t = Instant::now();
        let saved = s.disk.as_ref().map(|d| d.join(ROUTING));
        let profile = s
            .profile
            .clone()
            .or_else(|| saved.clone().filter(|p| p.exists()));
        let opts = Opts {
            max_ctx: s.ctx,
            mtp: s.mtp.clone(),
            profile: profile.clone(),
            ..Opts::default()
        };
        let mut e: Box<dyn ServeEngine> = if super::coder::is_qwen3next(&s.gguf)? {
            let mut c = super::coder::Coder::load(
                &s.gguf,
                super::coder::CoderOpts {
                    max_ctx: s.ctx,
                    // `--mtp` (the qwen3next path used to drop it: no MTP head, suffix drafts).
                    mtp: s.mtp.clone(),
                },
            )?;
            eprintln!("{}", c.load_report);
            c.use_graphs = true;
            if s.batch <= 1 {
                c.enable_sampler()?;
            }
            c.warm()?;
            Box::new(c)
        } else if super::glm5::engine::is_glm5(&s.gguf) {
            // GLM-5.3-Flash (W6's serve glue): greedy, N sequence slots
            // (TANG_GLM5_SLOTS, default 8) when --batch asks for them.
            let slots = s
                .batch
                .max(1)
                .max(super::glm5::serve_glue::slots_from_env());
            let g = super::glm5::serve_glue::Glm5Serve::load_slots(&s.gguf, slots)?;
            Box::new(g)
        } else {
            let mut e = Engine::load(&s.gguf, opts)?;
            eprintln!("{}", e.load_report);
            e.use_graphs = true;
            e.use_mtp = e.has_mtp();
            e.enable_sampler()?;
            e.warm()?;
            // Keep counting from the saved profile, not from the warm-up's routing.
            e.routing.fill(0);
            if let Some(p) = profile.as_ref().filter(|p| Some(*p) == saved.as_ref()) {
                let b = std::fs::read(p)?;
                if b.len() == e.routing.len() * 4 {
                    for (r, c) in e.routing.iter_mut().zip(b.chunks(4)) {
                        *r = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
            }
            if let Some(p) = &profile {
                eprintln!("tang-llm: expert cache seeded from {}", p.display());
            }
            Box::new(e)
        };
        let tok = FlashTokenizer::from_gguf(e.gguf())?;
        let id = |p: &str| tok.token_id(p).with_context(|| format!("no {p} token"));
        let mut eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|p| tok.token_id(p))
            .collect();
        if let Ok(n) = e.gguf().meta_u64("tokenizer.ggml.eos_token_id") {
            eos.push(n as u32);
        }
        eos.sort_unstable();
        eos.dedup();
        let think = (id("<think>")?, id("</think>")?);
        let mut wrap_up = tok.encode(WRAP_UP)?;
        wrap_up.push(think.1);
        wrap_up.extend(tok.encode("\n\n")?);
        let shape0 = e.save_running()?;
        let shape = shape0.parts().iter().map(|p| p.len()).collect();
        let gdn_layers = shape0.gdn_layers();
        let state_mb = shape0.bytes() as f64 / 1e6;
        drop(shape0);
        if let Some(d) = &s.disk {
            std::fs::create_dir_all(d)?;
        }
        if let Some(d) = &s.dump {
            std::fs::create_dir_all(d)?;
        }
        eprintln!(
            "tang-llm: flash engine ready in {:.1} s: ctx {}, drafts {}, running state {:.0} MB, up to {} saved states{}",
            t.elapsed().as_secs_f64(),
            s.ctx,
            if e.use_mtp() {
                "mtp"
            } else if s.suffix {
                "suffix"
            } else {
                "none"
            },
            state_mb,
            s.snapshots,
            s.disk.as_ref().map_or(String::new(), |d| format!(", system prompts saved in {}", d.display()))
        );
        let mut me = FlashServe {
            gguf: s.gguf.clone(),
            drafts_mode: if e.use_mtp() {
                "mtp"
            } else if s.suffix {
                "suffix"
            } else {
                "none"
            },
            // Suffix drafts need an engine whose verify can rewind cheaply; the glm5 glue
            // cannot yet, and one-sequence serving has nothing to draft against anyway.
            suffix: s.suffix && !e.use_mtp() && e.max_slots() > 1,
            dcfg: crate::draft::DraftConfig::new(&e.draft_cost()),
            dglobal: crate::draft::Global::new(1 << 20, None),
            dcalib: crate::draft::Calibration::default(),
            warm_dir: None,
            batch: 1,
            deterministic: std::env::var("TANG_SERVE_DETERMINISTIC").is_ok_and(|v| v == "1"),
            spec: crate::batch::spec::Tuner::new(crate::batch::spec::Mode::from_env()),
            spec_log: ([0.0; 6], Instant::now()),
            slots: Vec::new(),
            sched: Default::default(),
            prefill_policy: None,
            e,
            im_start: id("<|im_start|>")?,
            shield: Shield::new(&tok.tok),
            tags: Tags::new(|t| tok.token_id(t)),
            tok,
            ctx: s.ctx,
            eos,
            structured: None,
            think,
            wrap_up,
            snaps: Vec::new(),
            max_snaps: s.snapshots,
            max_snap_bytes: s.snapshot_bytes,
            memo: VecDeque::new(),
            policy: Policy::new(),
            clock: 0,
            live_at: crate::retention::now(),
            temp: s.temp,
            sampling: s.sampling,
            wide: std::env::var("TANG_FLASH_WIDE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            disk: s.disk.clone(),
            dump: s.dump.clone(),
            shape,
            gdn_layers,
            requests: 0,
            line: Line::default(),
        };
        // Capture the wide prefill graphs (64, 32, 16) now, not in the first request.
        if me.wide > 0 {
            let t = Instant::now();
            let ids: Vec<u32> = (0..64 + 32 + 16 + 1).map(|i| 1000 + i as u32).collect();
            for _ in 0..2 {
                me.e.reset();
                prefill_span(&mut *me.e, &ids, MAX_T, usize::MAX, me.wide, true)?;
            }
            me.e.reset();
            eprintln!(
                "tang-llm: wide prefill graphs (up to T={}) captured in {:.1} s",
                me.wide,
                t.elapsed().as_secs_f64()
            );
        }
        let mut warm = s.warm.clone();
        if let Some(d) = &s.warm_dir {
            std::fs::create_dir_all(d)?;
            let mut v: Vec<PathBuf> = std::fs::read_dir(d)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect();
            v.sort();
            warm.extend(v);
        }
        me.warm_dir = s.warm_dir.clone();
        if s.batch > 1 {
            ensure!(
                me.e.max_slots() > 1,
                "--batch needs an engine with sequence slots (qwen3next)"
            );
            // Provisional (sizes the batch windows); the count that sticks is taken once the
            // windows exist, from the request, so it clamps (and warns) once.
            me.batch = me.clamp_batch(s.batch, usize::MAX, true);
            me.e.enable_row_sampling()?;
            eprintln!(
                "tang-llm: continuous batching: up to {} requests at once (per-row sampling{})",
                me.batch,
                if me.e.use_mtp() {
                    ", MTP drafts"
                } else {
                    ", no drafts"
                }
            );
            if me.deterministic {
                eprintln!(
                    "tang-llm: TANG_SERVE_DETERMINISTIC=1: windows of at most {MAX_T} rows \
                     (bitwise; greedy output independent of the batch)"
                );
            }
            if !me.e.use_mtp() {
                me.spec.mode = crate::batch::spec::Mode::Off;
            }
            if me.e.use_mtp() {
                eprintln!(
                    "tang-llm: speculative decoding under batching: {:?} (TANG_SERVE_SPEC)",
                    me.spec.mode
                );
            }
            // Past MAX_T streams, or MTP drafts on top of the streams: batch windows, up to
            // TANG_SERVE_BATCH_ROWS rows (default --batch rounded up to 32; the drafts get
            // the rows the streams leave).
            let spec_rows = me.spec.mode != crate::batch::spec::Mode::Off;
            if (me.batch > MAX_T || spec_rows) && !me.deterministic {
                let rows = std::env::var("TANG_SERVE_BATCH_ROWS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(me.batch)
                    .max(me.batch)
                    .next_multiple_of(32)
                    .clamp(32, 256);
                let t = Instant::now();
                match me.e.enable_batch_rows(rows) {
                    Ok(()) => eprintln!(
                        "tang-llm: batch decode windows of {rows} rows built in {:.1} s",
                        t.elapsed().as_secs_f64()
                    ),
                    Err(e) => eprintln!("tang-llm: no batch windows ({e}); {MAX_T} rows a window"),
                }
            }
            // Wide prefill windows for the slots (TANG_SERVE_WIDE rows, default 4096; 0: off).
            let rows: usize = std::env::var("TANG_SERVE_WIDE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            if rows > 0 {
                let t = Instant::now();
                match me.e.enable_wide_rows(rows) {
                    Ok(()) => eprintln!(
                        "tang-llm: wide prefill windows of {rows} rows built in {:.1} s",
                        t.elapsed().as_secs_f64()
                    ),
                    Err(e) => eprintln!("tang-llm: no wide prefill ({e})"),
                }
            }
            // The windows took memory: count again before any slot exists, from `--batch`
            // (not the provisional count, whose own spare slot would be taken off twice).
            me.batch = me.clamp_batch(s.batch, me.batch, false);
        } else if !me.e.use_mtp() {
            // One stream: wide prefill windows for long prompts too (TANG_SERVE_WIDE rows).
            let rows: usize = std::env::var("TANG_SERVE_WIDE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            if rows > 0 {
                let t = Instant::now();
                match me.e.enable_wide_rows(rows) {
                    Ok(()) => eprintln!(
                        "tang-llm: wide prefill windows of {rows} rows built in {:.1} s",
                        t.elapsed().as_secs_f64()
                    ),
                    Err(e) => eprintln!("tang-llm: no wide prefill ({e})"),
                }
            }
        }
        for f in &warm {
            let t = Instant::now();
            let body: Value = serde_json::from_str(&std::fs::read_to_string(f)?)
                .with_context(|| format!("reading {}", f.display()))?;
            let mut req = crate::server::parse(&body).map_err(|e| anyhow::anyhow!(e))?;
            req.prefill_only = true;
            me.generate(&req, &mut |_| true, &mut ())?;
            eprintln!(
                "tang-llm: warmed {} ({} tokens, {} reused) in {:.1} s",
                f.display(),
                me.line.prompt,
                me.line.reused,
                t.elapsed().as_secs_f64()
            );
        }
        Ok(me)
    }

    /// The template's messages: tool-call arguments as objects (the template wants mappings),
    /// and assistant messages this server generated replaced by a marker (returned with the
    /// memo they match, in order).
    fn messages(&self, req: &Request) -> (Value, Vec<usize>) {
        let mut msgs = req.messages.clone();
        args_as_objects(&mut msgs);
        let mut spliced = Vec::new();
        let splice = std::env::var("TANG_FLASH_SPLICE").map_or(true, |v| v != "0");
        for m in msgs.as_array_mut().into_iter().flatten() {
            if m["role"] != "assistant" {
                continue;
            }
            let mut calls: Vec<(String, Value)> = Vec::new();
            for c in m["tool_calls"].as_array().into_iter().flatten() {
                let f = if c["function"].is_object() {
                    &c["function"]
                } else {
                    c
                };
                calls.push((
                    f["name"].as_str().unwrap_or_default().to_string(),
                    f["arguments"].clone(),
                ));
            }
            if !splice
                || m["reasoning_content"]
                    .as_str()
                    .is_some_and(|r| !r.trim().is_empty())
            {
                continue;
            }
            let content = m["content"].as_str().unwrap_or_default().trim().to_string();
            // The newest match: the same visible reply can come from more than one generation.
            let hit = self
                .memo
                .iter()
                .rposition(|x| x.content.trim() == content && x.calls == calls);
            if let Some(i) = hit {
                m["content"] = json!(MARK);
                if let Some(o) = m.as_object_mut() {
                    o.remove("tool_calls");
                    o.remove("reasoning_content");
                }
                spliced.push(i);
            }
        }
        (msgs, spliced)
    }

    /// The template, with the system block always rendered as with thinking on: this template
    /// puts its reasoning instructions at the very top of the system prompt, so a client that
    /// switches thinking per turn (frog does) would otherwise invalidate every cached token.
    /// Thinking off then only closes the generation prompt's `<think>` block, as the template
    /// does (`TANG_FLASH_STABLE_SYSTEM=0`: the template exactly).
    /// Request text is protected (see [`Shield`]): encode the result with [`Self::encode`].
    fn render(&self, msgs: &Value, req: &Request) -> Result<String> {
        let msgs = &self.shield.protect(msgs);
        let tools = req.tools.as_ref().map(|t| self.shield.protect(t));
        let stable = std::env::var("TANG_FLASH_STABLE_SYSTEM").map_or(true, |v| v != "0");
        // Structured output answers with the JSON value alone: no reasoning block.
        let think = if req.response_schema.is_some() {
            Some(false)
        } else {
            req.think
        };
        if !stable || think != Some(false) {
            return self.tok.render(msgs, tools.as_ref(), think);
        }
        let text = self.tok.render(msgs, tools.as_ref(), Some(true))?;
        match text.strip_suffix("<think>\n") {
            Some(head) => Ok(format!("{head}<think>\n\n</think>\n\n")),
            None => self.tok.render(msgs, tools.as_ref(), think),
        }
    }

    /// The prompt's ids, the rendered text, and how many assistant turns were spliced.
    fn prompt(&self, req: &Request) -> Result<(Vec<u32>, String, usize)> {
        let (msgs, spliced) = self.messages(req);
        let text = self.render(&msgs, req)?;
        let mut ids = Vec::new();
        let mut at = 0;
        let mut n = 0;
        let mut ok = true;
        let mut pieces = Vec::new();
        for (k, (i, _)) in text.match_indices(MARK).enumerate() {
            let Some(&mi) = spliced.get(k) else {
                ok = false;
                break;
            };
            let head = &text[at..i];
            let open = "<|im_start|>assistant\n";
            let head = if let Some(h) = head
                .strip_suffix("<think>\n\n</think>\n\n")
                .filter(|h| h.ends_with(open))
            {
                h
            } else if head.ends_with(open) {
                head
            } else {
                ok = false;
                break;
            };
            let m = &self.memo[mi];
            pieces.push((format!("{head}{}", m.tail), mi));
            at = i + MARK.len();
            n += 1;
        }
        if !ok || n != spliced.len() {
            // The template didn't render the markers where expected: no splicing.
            let mut msgs = req.messages.clone();
            args_as_objects(&mut msgs);
            let text = self.render(&msgs, req)?;
            eprintln!(
                "tang-llm: splicing skipped (template rendered generated turns unexpectedly)"
            );
            return Ok((self.encode(&text)?, text, 0));
        }
        for (piece, mi) in pieces {
            ids.extend(self.encode(&piece)?);
            ids.extend_from_slice(&self.memo[mi].ids);
        }
        ids.extend(self.encode(&text[at..])?);
        Ok((ids, text, n))
    }

    /// Rendered (protected) prompt text to ids: the template's special tokens by id, request
    /// text as plain tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.shield.encode(&self.tok.tok, text)
    }

    /// Keep a copy of the running state after `tokens` (the engine's sequence now).
    fn snapshot(&mut self, pinned: bool) -> Result<()> {
        self.snapshot_at(0, pinned)
    }

    /// Save sequence slot `slot`'s state (after its tokens) as a reusable prefix state.
    pub(super) fn snapshot_at(&mut self, slot: usize, pinned: bool) -> Result<()> {
        let toks: Vec<u32> = if slot == 0 {
            self.e.tokens().clone()
        } else {
            self.e.slot_tokens(slot).to_vec()
        };
        self.clock += 1;
        let clock = self.clock;
        if let Some(s) = self.snaps.iter_mut().find(|s| s.tokens == toks) {
            s.used = clock;
            s.at = crate::retention::now();
            s.pinned |= pinned;
            return Ok(());
        }
        let t = Instant::now();
        let state = Arc::new(self.e.save_running_at(slot)?);
        let t_run = t.elapsed().as_secs_f64() * 1e3;
        let n = toks.len();
        // Only the positions past the longest saved state this one extends are copied.
        let base = (0..self.snaps.len())
            .filter(|&i| toks.starts_with(&self.snaps[i].tokens))
            .max_by_key(|&i| self.snaps[i].tokens.len());
        let (mut pos, from) = match base {
            Some(i) => (self.snaps[i].pos.clone(), self.snaps[i].tokens.len()),
            None => (Vec::new(), 0),
        };
        if from < n {
            pos.push(Arc::new(self.e.save_positional_at(slot, from, n)?));
        }
        let t_pos = t.elapsed().as_secs_f64() * 1e3;
        let tokens = toks;
        if pinned && self.disk_path(&tokens).is_some_and(|p| !p.exists()) {
            // From the host copies just made (the segments cover 0..n), on the writer thread.
            self.save_disk(&tokens, state.clone(), pos.clone());
        }
        let t_disk = t.elapsed().as_secs_f64() * 1e3;
        self.line.snap_ms += t.elapsed().as_secs_f64() * 1e3;
        self.snaps.push(Snap {
            tokens,
            state,
            pos,
            pinned,
            used: clock,
            at: crate::retention::now(),
        });
        let size = |snaps: &[Snap]| {
            let mut seen = std::collections::HashSet::new();
            snaps
                .iter()
                .map(|s| {
                    s.state.bytes()
                        + s.pos
                            .iter()
                            .filter(|p| seen.insert(Arc::as_ptr(p)))
                            .map(|p| p.bytes())
                            .sum::<usize>()
                })
                .sum::<usize>()
        };
        while self.snaps.len() > 1 && size(&self.snaps) > self.max_snap_bytes {
            let lru = (0..self.snaps.len())
                .min_by_key(|&i| (self.snaps[i].pinned, self.snaps[i].used))
                .unwrap();
            self.snaps.swap_remove(lru);
        }
        // Least recently used first; pinned ones have their own (small) quota.
        for (want_pinned, cap) in [(false, self.max_snaps), (true, 4)] {
            while self
                .snaps
                .iter()
                .filter(|s| s.pinned == want_pinned)
                .count()
                > cap
            {
                let lru = (0..self.snaps.len())
                    .filter(|&i| self.snaps[i].pinned == want_pinned)
                    .min_by_key(|&i| self.snaps[i].used)
                    .unwrap();
                self.snaps.swap_remove(lru);
            }
        }
        if std::env::var("TANG_SNAP_TIMES").is_ok_and(|v| v == "1") {
            eprintln!(
                "snapshot {n} tok pinned {pinned}: running {t_run:.0} ms, positional {:.0}, disk {:.0}, evict {:.0}",
                t_pos - t_run,
                t_disk - t_pos,
                t.elapsed().as_secs_f64() * 1e3 - t_disk
            );
        }
        Ok(())
    }

    /// Drop saved states, remembered replies and the live sequence unused for the TTL.
    fn expire_memory(&mut self) {
        let ttl = crate::retention::ttl();
        let stale = |at| crate::retention::expired(at, ttl);
        self.snaps.retain(|s| !stale(s.at));
        self.memo.retain(|m| !stale(m.at));
        if !self.e.tokens().is_empty() && stale(self.live_at) {
            self.e.reset();
        }
    }

    /// [`Self::expire_memory`], and delete saved states on disk unused for the TTL.
    fn expire(&mut self) {
        let ttl = crate::retention::ttl();
        if ttl.is_none() {
            return;
        }
        self.expire_memory();
        let Some(dir) = &self.disk else { return };
        for ent in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = ent.path();
            if p.extension().is_some_and(|x| x == "state") && crate::retention::stale_file(&p, ttl)
            {
                let _ = std::fs::remove_file(&p);
            }
        }
    }

    fn disk_path(&self, ids: &[u32]) -> Option<PathBuf> {
        Some(
            self.disk
                .as_ref()?
                .join(format!("{}-{:016x}.state", ids.len(), hash(ids))),
        )
    }

    /// Write the routing counts to `disk` (seeds the next load's expert cache). Halves them
    /// before they can overflow, so the profile also follows recent use.
    fn save_routing(&mut self) {
        let Some(d) = &self.disk else { return };
        if let Some(r) = self.e.routing() {
            if r.iter().any(|&c| c > 1 << 30) {
                r.iter_mut().for_each(|c| *c /= 2);
            }
        }
        let tmp = d.join(format!("{ROUTING}.tmp"));
        if let Err(err) = self
            .e
            .save_routing(&tmp)
            .and_then(|_| Ok(std::fs::rename(&tmp, d.join(ROUTING))?))
        {
            eprintln!("tang-llm: saving routing counts: {err:#}");
        }
    }

    /// Write a pinned state to disk: positions `0..ids.len()` are `pos`'s segments in order.
    /// Serialized and written on a thread (the file appears complete or not at all), so the
    /// serve loop only pays the device copies.
    fn save_disk(&self, ids: &[u32], s: Arc<RunningState>, pos: Vec<Arc<Positional>>) {
        let Some(path) = self.disk_path(ids) else {
            return;
        };
        if path.exists() {
            crate::retention::touch(&path);
            return;
        }
        let ids = ids.to_vec();
        let budget = std::env::var("TANG_FLASH_DISK_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(8e9, |g| g * 1e9) as u64;
        std::thread::spawn(move || {
            let mut bytes: Vec<u8> = Vec::with_capacity(s.bytes() + 4 * ids.len() + 64);
            bytes.extend_from_slice(b"TANGRS02");
            bytes.extend_from_slice(&(ids.len() as u64).to_le_bytes());
            for &i in &ids {
                bytes.extend_from_slice(&i.to_le_bytes());
            }
            let parts = s.parts();
            bytes.extend_from_slice(&(parts.len() as u64).to_le_bytes());
            for p in parts {
                bytes.extend_from_slice(&(p.len() as u64).to_le_bytes());
                // One copy of the f32s' little-endian bytes (a per-value loop here cost ~100 ms a
                // save on the serve thread, in every pinned prompt's TTFT).
                if cfg!(target_endian = "little") {
                    // SAFETY: `p` is a valid `[f32]`; any byte view of it is valid `u8`s.
                    bytes.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(
                            p.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(p),
                        )
                    });
                } else {
                    for &x in p {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
            // Each buffer's rows over 0..n: its part of every segment, in order.
            let nbuf = pos.first().map_or(0, |p| p.parts().len());
            bytes.extend_from_slice(&(nbuf as u64).to_le_bytes());
            for b in 0..nbuf {
                let len: usize = pos.iter().map(|p| p.parts()[b].len()).sum();
                bytes.extend_from_slice(&(len as u64).to_le_bytes());
                for p in &pos {
                    bytes.extend_from_slice(&p.parts()[b]);
                }
            }
            let tmp = path.with_extension("tmp");
            let r = std::fs::File::create(&tmp)
                .and_then(|mut f| f.write_all(&bytes))
                .and_then(|_| std::fs::rename(&tmp, &path));
            if let Err(e) = r {
                eprintln!("tang-llm: saving {}: {e}", path.display());
            }
            // Keep the directory under its budget, oldest first.
            let Some(dir) = path.parent() else { return };
            let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "state"))
                .filter_map(|e| {
                    let m = e.metadata().ok()?;
                    Some((m.modified().ok()?, m.len(), e.path()))
                })
                .collect();
            files.sort();
            let mut total: u64 = files.iter().map(|f| f.1).sum();
            for (_, len, p) in files {
                if total <= budget {
                    break;
                }
                let _ = std::fs::remove_file(&p);
                total -= len;
            }
        });
    }

    /// The longest state on disk that `ids[..=limit]` extends.
    fn load_disk(&self, ids: &[u32], limit: usize) -> Option<Snap> {
        let dir = self.disk.as_ref()?;
        let mut best: Option<(usize, PathBuf)> = None;
        for ent in std::fs::read_dir(dir).ok()?.flatten() {
            let name = ent.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".state") else {
                continue;
            };
            let Some((n, h)) = stem.split_once('-') else {
                continue;
            };
            let (Ok(n), Ok(h)) = (n.parse::<usize>(), u64::from_str_radix(h, 16)) else {
                continue;
            };
            if crate::retention::stale_file(&ent.path(), crate::retention::ttl()) {
                let _ = std::fs::remove_file(ent.path());
                continue;
            }
            if n <= limit && n > best.as_ref().map_or(0, |b| b.0) && hash(&ids[..n]) == h {
                best = Some((n, ent.path()));
            }
        }
        let (n, path) = best?;
        crate::retention::touch(&path);
        let read = || -> Result<Snap> {
            let mut b = Vec::new();
            std::fs::File::open(&path)?.read_to_end(&mut b)?;
            let mut at = 8usize;
            fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8]> {
                let s = b.get(*at..*at + n).context("state file is short")?;
                *at += n;
                Ok(s)
            }
            let word = |b: &[u8], at: &mut usize| -> Result<usize> {
                Ok(u64::from_le_bytes(take(b, at, 8)?.try_into()?) as usize)
            };
            ensure!(b.get(..8) == Some(&b"TANGRS02"[..]), "not a state file");
            let len = word(&b, &mut at)?;
            ensure!(len == n, "length");
            let tokens: Vec<u32> = take(&b, &mut at, 4 * len)?
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ensure!(tokens == ids[..n], "tokens differ");
            ensure!(word(&b, &mut at)? == self.shape.len(), "parts");
            let mut parts = Vec::with_capacity(self.shape.len());
            for &want in &self.shape {
                let k = word(&b, &mut at)?;
                ensure!(k == want, "part size");
                parts.push(
                    take(&b, &mut at, 4 * k)?
                        .chunks(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                );
            }
            let np = word(&b, &mut at)?;
            let mut pos = Vec::with_capacity(np);
            for _ in 0..np {
                let k = word(&b, &mut at)?;
                pos.push(take(&b, &mut at, k)?.to_vec());
            }
            Ok(Snap {
                tokens,
                state: Arc::new(RunningState::from_parts(n, self.gdn_layers, parts)?),
                pos: vec![Arc::new(Positional::from_parts(0, n, pos))],
                pinned: true,
                used: 0,
                at: crate::retention::now(),
            })
        };
        match read() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("tang-llm: reading {}: {e:#}", path.display());
                None
            }
        }
    }

    /// Bring the engine to a prefix of `ids` (live sequence, a saved state, or empty), prefill
    /// the rest (saving states at the system prompt's end and the last turn boundary), and
    /// return (tokens reused, the sampled next token).
    fn prefill(&mut self, ids: &[u32], text: &str) -> Result<(usize, u32)> {
        let limit = ids.len() - 1;
        let live = self.e.tokens().len();
        let live_ok = live <= limit && lcp(self.e.tokens(), ids) == live;
        let best = (0..self.snaps.len())
            .filter(|&i| {
                self.snaps[i].tokens.len() <= limit && ids.starts_with(&self.snaps[i].tokens)
            })
            .max_by_key(|&i| self.snaps[i].tokens.len());
        let snap_len = best.map_or(0, |i| self.snaps[i].tokens.len());
        let mut start = if live_ok { live } else { 0 };
        let t = Instant::now();
        if snap_len > start {
            let i = best.unwrap();
            self.clock += 1;
            self.snaps[i].used = self.clock;
            self.snaps[i].at = crate::retention::now();
            if self.snaps[i].pinned {
                if let Some(p) = self.disk_path(&self.snaps[i].tokens) {
                    crate::retention::touch(&p);
                }
            }
            let valid = lcp(self.e.tokens(), &self.snaps[i].tokens);
            for p in &self.snaps[i].pos {
                self.e.restore_positional(p, valid)?;
            }
            let (tokens, state) = (&self.snaps[i].tokens, &self.snaps[i].state);
            self.e.restore_running(state, tokens)?;
            start = snap_len;
        } else if !live_ok {
            match self.load_disk(ids, limit).filter(|s| s.tokens.len() > 0) {
                Some(s) => {
                    let valid = lcp(self.e.tokens(), &s.tokens);
                    for p in &s.pos {
                        self.e.restore_positional(p, valid)?;
                    }
                    self.e.restore_running(&s.state, &s.tokens)?;
                    start = s.tokens.len();
                    eprintln!("tang-llm: read the running state at {start} from disk");
                    self.snaps.push(s);
                }
                None => self.e.reset(),
            }
        }
        self.line.restore_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut marks = self.marks(ids, text);
        marks.retain(|&(p, _)| p > start);
        // The MTP only over the prompt's last positions (`TANG_FLASH_MTP_PREFILL`, default 512).
        let mtp_from = ids.len().saturating_sub(
            std::env::var("TANG_FLASH_MTP_PREFILL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
        );
        let log = std::env::var("TANG_FLASH_SPAN_LOG").is_ok_and(|v| v == "1");
        let span_line = |a: usize, b: usize, ms: f64, st: &[super::engine::WinStats], mtp: f64| {
            let (wide, narrow): (Vec<_>, Vec<_>) = st.iter().partition(|s| s.t > MAX_T);
            let sum = |v: &[&super::engine::WinStats], f: fn(&super::engine::WinStats) -> f64| {
                v.iter().map(|s| f(s)).sum::<f64>()
            };
            eprintln!(
                "  span {a}..{b} ({} tok) {ms:.0} ms = {:.0} tok/s | wide {} win {:.0} ms (cpu {:.0}, swaps {:.0}) | narrow {} win {:.0} ms (cpu {:.0}) | mtp {mtp:.0} ms",
                b - a,
                (b - a) as f64 / ms * 1e3,
                wide.len(),
                sum(&wide, |s| s.wall_ms),
                sum(&wide, |s| s.cpu_ms),
                sum(&wide, |s| s.swaps as f64),
                narrow.len(),
                sum(&narrow, |s| s.wall_ms),
                sum(&narrow, |s| s.cpu_ms),
            );
        };
        let mut a = start;
        for (p, pinned) in marks {
            if p > a {
                let t = Instant::now();
                let (_, st, mtp) =
                    prefill_span(&mut *self.e, &ids[a..p], MAX_T, mtp_from, self.wide, false)?;
                if log {
                    span_line(a, p, t.elapsed().as_secs_f64() * 1e3, &st, mtp);
                }
                a = p;
            }
            let t = Instant::now();
            self.snapshot(pinned)?;
            if log {
                eprintln!(
                    "  snapshot at {a} (pinned {pinned}) {:.0} ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        let t = Instant::now();
        let (next, st, mtp) =
            prefill_span(&mut *self.e, &ids[a..], MAX_T, mtp_from, self.wide, true)?;
        if log {
            span_line(a, ids.len(), t.elapsed().as_secs_f64() * 1e3, &st, mtp);
        }
        Ok((start, next))
    }

    fn generate(
        &mut self,
        req: &Request,
        on: &mut dyn FnMut(Piece) -> bool,
        ctl: &mut dyn Control,
    ) -> Result<(Finish, Usage)> {
        let t0 = Instant::now();
        self.line = Line::default();
        let (ids, text, spliced) = self.prompt(req)?;
        self.line.spliced = spliced;
        ensure!(
            ids.len() + 1 < self.ctx,
            "prompt is {} tokens; the context window is {}",
            ids.len(),
            self.ctx
        );
        let limit = req
            .max_tokens
            .unwrap_or(usize::MAX)
            .min(self.ctx - ids.len() - 1)
            .max(1);
        let (temp, top_k, top_p, presence) = self.request_sampling(req);
        self.e.set_sampling(temp, req.sampling.seed as u32);
        if let Some(sp) = self.e.sampler_mut() {
            // top-k 0 (off) keeps the 64 best: the sampler's bound.
            sp.top_k = if top_k == 0 {
                super::sampler::MAX_K
            } else {
                top_k
            };
            sp.top_p = if top_p > 0.0 && top_p <= 1.0 {
                top_p
            } else {
                1.0
            };
            sp.presence = presence;
        }
        self.e.sampler_begin(ids.len())?;
        self.line.sampling = format!("T {temp} k {top_k} p {top_p} pres {presence}");
        self.requests += 1;
        let tag = self.requests;
        if let Some(d) = &self.dump {
            std::fs::write(
                d.join(format!("req-{tag}.ids")),
                ids.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )?;
        }
        // Structured output: the prompt's last token is decoded like an output step, so every
        // output token (the first too) is checked against the grammar with its row's logits.
        let mut constraint = self.constraint(req)?;
        let hs = crate::structured::HostSampling {
            temperature: temp,
            top_k,
            top_p,
            seed: req.sampling.seed,
        };
        let (reused, cur) = match constraint.as_mut() {
            Some(c) => {
                ensure!(ids.len() >= 2, "prompt too short for structured output");
                let n = ids.len() - 1;
                let (reused, _) = self.prefill(&ids[..n], &text)?;
                let (proposed, mut lg) = self.e.step_logits(ids[n])?;
                (reused, c.choose(proposed, &mut lg, &hs, ids.len() as u64)?)
            }
            None => self.prefill(&ids, &text)?,
        };
        let prefill_s = t0.elapsed().as_secs_f64();
        self.line.prompt = ids.len();
        self.line.reused = reused;
        self.line.prefill_ms = prefill_s * 1e3;
        let usage =
            |out: usize, reasoning: usize, decode_s: f64, drafted: usize, accepted: usize| Usage {
                prompt_tokens: ids.len(),
                cached_tokens: reused,
                completion_tokens: out,
                reasoning_tokens: reasoning,
                prefill_tok_s: (ids.len() - reused) as f64 / prefill_s.max(1e-9),
                decode_tok_s: out as f64 / decode_s.max(1e-9),
                draft_tokens: drafted,
                accepted_tokens: accepted,
                ..Default::default()
            };
        if req.prefill_only {
            self.line.finish = "prefill";
            return Ok((Finish::Stop, usage(0, 0, 0.0, 0, 0)));
        }

        let gen_at = text
            .rfind("<|im_start|>assistant\n")
            .map_or(text.len(), |i| i + "<|im_start|>assistant\n".len());
        let tail = text[gen_at..].to_string();
        let opened = text.trim_end().ends_with("<think>");
        let mut parser = Parser::new(opened).with_tools(req.tools.as_ref());
        let mut think = ThinkBudget::new(Some(self.think), req.thinking_budget, opened);
        let mut in_think = opened;
        let mut spent = 0usize;
        let mut out: Vec<u32> = Vec::new();
        let mut detok = Detok::default();
        let mut text_out = String::new();
        let (mut content, mut calls) = (String::new(), Vec::new());
        let mut finish = Finish::Length;
        let (mut windows, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
        let mut first = None;
        let t1 = Instant::now();
        let mut pending = vec![cur];
        let mut sess = self.suffix.then(|| {
            let cost: Vec<f32> = (0..=MAX_T).map(|k| self.dcfg.cost(k.max(1))).collect();
            crate::draft::Session::new(&self.dcfg, &self.dglobal, self.dcalib.clone(), cost, &ids)
        });
        // Deliver pieces; false when the client went away.
        let mut deliver = |pieces: Vec<Piece>,
                           content: &mut String,
                           calls: &mut Vec<(String, Value)>,
                           first: &mut Option<f64>|
         -> bool {
            for p in pieces {
                if first.is_none() {
                    *first = Some(t0.elapsed().as_secs_f64() * 1e3);
                }
                match &p {
                    Piece::Text(t) => content.push_str(t),
                    Piece::ToolCall { name, arguments } => {
                        calls.push((name.clone(), arguments.clone()))
                    }
                    Piece::Reasoning(_) => {}
                }
                if !on(p) {
                    return false;
                }
            }
            true
        };
        'gen: loop {
            // The prefill can't stop part way; decode stops between windows.
            if ctl.cancelled() {
                finish = Finish::Cancelled;
                break 'gen;
            }
            let mut wrap = false;
            for &tok in &pending {
                if self.eos.contains(&tok) {
                    finish = Finish::Stop;
                    break 'gen;
                }
                out.push(tok);
                if let Some(c) = constraint.as_mut() {
                    c.accept(tok)?;
                }
                if let Some(ss) = sess.as_mut() {
                    ss.push(tok);
                }
                if tok == self.think.0 {
                    in_think = true;
                } else if tok == self.think.1 {
                    in_think = false;
                } else if in_think {
                    spent += 1;
                }
                wrap |= think.step(tok);
                // Tag tokens come out as tags, everything else as literal text (see `Parser`).
                for chunk in detok.push(&out, &self.tags, |ids| self.tok.decode(ids))? {
                    let (new, tag) = match chunk {
                        Chunk::Text(s) => (s, None),
                        Chunk::Tag(t) => (t.to_string(), Some(t)),
                    };
                    text_out.push_str(&new);
                    let stop_at = req
                        .stop
                        .iter()
                        .filter_map(|s| text_out.find(s.as_str()))
                        .min();
                    let new = match stop_at {
                        Some(i) => new[..new.len().saturating_sub(text_out.len() - i)].to_string(),
                        None => new,
                    };
                    // Constrained output is the grammar's JSON verbatim: tool-call or think
                    // markers inside a string value are data, not structure.
                    let pieces = match tag {
                        _ if constraint.is_some() => vec![Piece::Text(new)],
                        Some(t) if new == t => parser.push_tag(t),
                        _ => parser.push_text(&new),
                    };
                    if !deliver(pieces, &mut content, &mut calls, &mut first) {
                        finish = Finish::Cancelled;
                        break 'gen;
                    }
                    if stop_at.is_some() {
                        finish = Finish::Stop;
                        break 'gen;
                    }
                }
                // The JSON value is complete: nothing but an end of turn may follow.
                if constraint.as_ref().is_some_and(|c| c.finished()) {
                    finish = Finish::Stop;
                    break 'gen;
                }
                if out.len() >= limit {
                    break 'gen;
                }
            }
            let last = *pending.last().unwrap();
            // Structured output: one token a window, no drafts (an accepted draft is committed
            // before the grammar could reject it), checked against the grammar.
            if let Some(c) = constraint.as_mut() {
                let (proposed, mut lg) = self.e.step_logits(last)?;
                let pos = (ids.len() + out.len()) as u64;
                pending = vec![c.choose(proposed, &mut lg, &hs, pos)?];
                windows += 1;
                continue;
            }
            if wrap && out.len() + self.wrap_up.len() < limit {
                // Out of thinking budget: the wrap-up and `</think>` as if the model wrote them.
                think.closed(self.wrap_up.len());
                for &w in &self.wrap_up.clone() {
                    out.push(w);
                    if let Some(ss) = sess.as_mut() {
                        ss.push(w);
                    }
                    for chunk in detok.push(&out, &self.tags, |ids| self.tok.decode(ids))? {
                        text_out.push_str(match &chunk {
                            Chunk::Text(s) => s,
                            Chunk::Tag(t) => t,
                        });
                        let pieces = parser.push_chunk(&chunk);
                        if !deliver(pieces, &mut content, &mut calls, &mut first) {
                            finish = Finish::Cancelled;
                            break 'gen;
                        }
                    }
                }
                in_think = false;
                let mut feed = vec![last];
                feed.extend_from_slice(&self.wrap_up);
                let next = self.e.prefill(&feed, MAX_T)?;
                windows += feed.len().div_ceil(MAX_T);
                pending = vec![next];
                continue;
            }
            // Drafts: none past the token limit, the thinking budget, or an end of turn (so the
            // sequence never runs past where the reply ends).
            let mut room = (MAX_T - 1).min(limit - out.len() - 1);
            if in_think {
                room = room.min(self.policy.think_room);
                if let (Some(b), false) = (req.thinking_budget, think.hit) {
                    room = room.min(b.saturating_sub(spent + 1));
                }
            }
            let proposal = sess.as_ref().map(|ss| ss.propose(room));
            let (mut drafts, mut probs) = match &proposal {
                Some(d) => (d.tokens.clone(), Vec::new()),
                None => self.policy.choose(self.e.mtp_last(), room),
            };
            if let Some(k) = drafts.iter().position(|d| self.eos.contains(d)) {
                drafts.truncate(k);
                probs.truncate(k);
            }
            let kept = self.e.verify(last, &drafts)?;
            if self.e.use_mtp() {
                let wall = self.e.last().wall_ms;
                let mtp = self.e.last_mtp_ms();
                self.policy
                    .observe(&probs, kept.len() - 1, drafts.len() + 1, wall, mtp);
            }
            if let (Some(ss), Some(d)) = (sess.as_mut(), &proposal) {
                let acc = kept.len() - 1;
                for j in 0..drafts.len().min(acc + 1) {
                    ss.calib.observe(d.match_len, d.shares[j], j < acc);
                }
            }
            windows += 1;
            drafted += drafts.len();
            accepted += kept.len() - 1;
            pending = kept;
        }
        if finish != Finish::Cancelled {
            deliver(parser.finish(), &mut content, &mut calls, &mut first);
            if !calls.is_empty() && finish == Finish::Stop {
                finish = Finish::ToolCalls;
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        if let Some(ss) = sess {
            self.dcalib = ss.calib.clone();
            drop(ss);
            self.dglobal.push(&out);
        }
        // A reply that ended on its own can be spliced into the next prompt.
        if matches!(finish, Finish::Stop | Finish::ToolCalls)
            && req.stop.iter().all(|s| !text_out.contains(s.as_str()))
        {
            self.memo.push_back(Memo {
                content: content.clone(),
                calls: calls.clone(),
                tail,
                ids: out.clone(),
                at: crate::retention::now(),
            });
            while self.memo.len() > 256 {
                self.memo.pop_front();
            }
        }
        if let Some(d) = &self.dump {
            std::fs::write(
                d.join(format!("req-{tag}.out")),
                out.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )?;
        }
        self.line.ttft_ms = first.unwrap_or(t0.elapsed().as_secs_f64() * 1e3);
        self.line.decode = out.len();
        self.line.decode_s = decode_s;
        self.line.windows = windows;
        self.line.drafted = drafted;
        self.line.accepted = accepted;
        self.line.calls = calls.len();
        self.line.finish = match finish {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
            Finish::Cancelled => "cancelled",
        };
        Ok((
            finish,
            usage(out.len(), think.reasoning, decode_s, drafted, accepted),
        ))
    }
}

impl FlashServe {
    fn run(
        &mut self,
        req: &Request,
        on: &mut dyn FnMut(Piece) -> bool,
        ctl: &mut dyn Control,
    ) -> Result<(Finish, Usage)> {
        self.expire_memory();
        let r = self.generate(req, on, ctl);
        self.live_at = crate::retention::now();
        if r.is_ok() {
            self.record_warm(req);
        }
        if r.is_err() {
            // Whatever the engine holds now is unknown: start over next time.
            self.e.reset();
        }
        self.save_routing();
        r
    }
}

impl FlashServe {
    /// Where to save prefix states in prompt `ids` (rendered from `text`), in order, with
    /// whether to pin each: the end of the tools block, the end of the system prompt (the second
    /// `<|im_start|>`) and the last turn boundary (the last one, which opens the generation
    /// prompt).
    /// `req`'s structured-output constraint, if it has a schema.
    pub(super) fn constraint(
        &mut self,
        req: &Request,
    ) -> Result<Option<crate::structured::Constraint>> {
        let Some(schema) = &req.response_schema else {
            return Ok(None);
        };
        ensure!(
            req.stop.is_empty(),
            "stop strings cannot be combined with structured output"
        );
        ensure!(
            self.e.vocab_size() > 0,
            "structured output needs the qwen3next engine"
        );
        if self.structured.is_none() {
            let eos = *self.eos.first().context("model has no EOS token")?;
            self.structured = Some(crate::structured::model_factory(
                &self.tok.tok,
                self.e.vocab_size(),
                eos,
            )?);
        }
        Ok(Some(crate::structured::Constraint::new(
            self.structured.as_ref().unwrap(),
            schema.clone(),
            self.eos.clone(),
        )?))
    }

    pub(super) fn marks(&self, ids: &[u32], text: &str) -> Vec<(usize, bool)> {
        let starts: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == self.im_start)
            .map(|(i, _)| i)
            .collect();
        let mut marks: Vec<(usize, bool)> = Vec::new();
        // The end of the tools block, which the template puts before the system text (that
        // has the working directory in it): the same for every session of a client.
        if let Some(i) = text.find("</IMPORTANT>").filter(|&i| i < 200_000) {
            if let Ok(head) = self.encode(&text[..i + "</IMPORTANT>".len()]) {
                if head.len() > 1 && ids.starts_with(&head[..head.len() - 1]) {
                    // The last token may merge with what follows; stop one short.
                    marks.push((head.len() - 1, true));
                }
            }
        }
        if let Some(&p) = starts.get(1) {
            marks.push((p, true));
        }
        if let Some(&p) = starts.last() {
            marks.push((p, false));
        }
        marks.sort_by_key(|m| m.0);
        marks.retain(|&(p, _)| p > 0 && p < ids.len());
        marks.dedup_by_key(|m| m.0);
        marks
    }

    /// Record `req`'s tools block for the learned pre-warm (see [`Settings::warm_dir`]): a
    /// body with just its tools and a placeholder turn, once per distinct tool set.
    fn record_warm(&self, req: &Request) {
        let (Some(d), Some(tools)) = (&self.warm_dir, req.tools.as_ref()) else {
            return;
        };
        if !tools.as_array().is_some_and(|a| !a.is_empty()) {
            return;
        }
        let text = tools.to_string();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in text.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        let p = d.join(format!("tools-{h:016x}.json"));
        if p.exists() {
            return;
        }
        // At most 32 learned tool sets.
        let n = std::fs::read_dir(d).map_or(0, |r| r.count());
        if n >= 32 {
            return;
        }
        let body = json!({
            "messages": [{"role": "system", "content": "."}, {"role": "user", "content": "."}],
            "tools": tools,
            "max_tokens": 1,
        });
        match std::fs::write(&p, body.to_string()) {
            Ok(()) => eprintln!(
                "tang-llm: learned a new tools block for the pre-warm: {}",
                p.display()
            ),
            Err(e) => eprintln!("tang-llm: recording {}: {e}", p.display()),
        }
    }
}

impl Backend for FlashServe {
    fn model_path(&self, _spec: &str) -> Result<PathBuf> {
        Ok(self.gguf.clone())
    }

    fn complete(
        &mut self,
        req: &Request,
        on: &mut dyn FnMut(Piece) -> bool,
    ) -> Result<(Finish, Usage)> {
        self.run(req, on, &mut ())
    }

    fn complete_with(
        &mut self,
        req: &Request,
        _prompt: Option<(String, Vec<u32>)>,
        on: &mut dyn FnMut(Piece) -> bool,
        ctl: &mut dyn Control,
    ) -> Result<Outcome> {
        self.run(req, on, ctl).map(|(f, u)| Outcome::Done(f, u))
    }

    fn context_window(&self) -> usize {
        self.ctx
    }

    fn max_batch(&self) -> usize {
        self.batch
    }

    fn can_admit(&self, key: Option<&str>) -> bool {
        self.batch > 1 && self.batch_can_admit(key)
    }

    fn admit(
        &mut self,
        req: &Request,
        _prompt: Option<(String, Vec<u32>)>,
    ) -> Result<crate::server::Inflight> {
        self.batch_admit(req)
    }

    fn step(&mut self, lanes: &mut [crate::server::DynLane<'_>]) -> Result<()> {
        self.batch_step(lanes)
    }

    fn retire(&mut self, s: crate::server::Inflight) -> Result<Outcome> {
        self.batch_retire(s)
    }

    fn stream_state(&self, s: &crate::server::Inflight) -> (bool, bool) {
        FlashServe::batch_state(s)
    }

    fn expire(&mut self) {
        FlashServe::expire(self)
    }

    fn after(&mut self, _req: &Request, result: &Result<(Finish, Usage)>) {
        // A batched request logged its own line when it retired (`batch req:`); `line` and
        // the slot-0 snapshot below belong to the one-stream path (`line.snap_ms` only ever
        // summed the batch's prefix-state saves, which is what the phantom `req N: prompt 0
        // ... state saves 7000 ms` after every batched completion showed).
        if self.batch > 1 {
            if let Err(e) = result {
                eprintln!("tang-llm: req {}: error: {e:#}", self.requests);
            }
            return;
        }
        let l = &self.line;
        match result {
            Ok(_) => eprintln!(
                "tang-llm: req {}: prompt {} (reused {}, prefilled {} in {:.0} ms = {:.0} tok/s; restore {:.0} ms, state saves {:.0} ms; {} turns spliced) | ttft {:.0} ms | decode {} tokens in {:.2} s = {:.1} tok/s, {} windows, {:.2} tokens/window, drafts {} {}/{} accepted | {} tool calls, {} | {}",
                self.requests,
                l.prompt,
                l.reused,
                l.prompt - l.reused,
                l.prefill_ms,
                (l.prompt - l.reused) as f64 / (l.prefill_ms / 1e3).max(1e-9),
                l.restore_ms,
                l.snap_ms,
                l.spliced,
                l.ttft_ms,
                l.decode,
                l.decode_s,
                l.decode as f64 / l.decode_s.max(1e-9),
                l.windows,
                l.decode as f64 / l.windows.max(1) as f64,
                self.drafts_mode,
                l.accepted,
                l.drafted,
                l.calls,
                l.finish,
                l.sampling
            ),
            Err(e) => eprintln!("tang-llm: req {}: error: {e:#}", self.requests),
        }
        // Keep the end of this conversation, in case another one runs next.
        if result.is_ok() && !self.e.tokens().is_empty() {
            let t = Instant::now();
            if let Err(e) = self.snapshot(false) {
                eprintln!("tang-llm: saving the running state: {e:#}");
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if ms > 50.0 {
                eprintln!(
                    "tang-llm: saved the running state at {} in {ms:.0} ms",
                    self.e.tokens().len()
                );
            }
        }
    }
}

/// `tang-llm flash-resume-test <gguf> --ids-file F [--at P] [--mtp M]`: is resuming from a saved
/// running state bitwise the same as never having left? Prefills `ids[..P]`, saves the state,
/// runs unrelated tokens, restores, prefills `ids[P..]`, and compares every logit of the suffix
/// with a run that prefilled `ids[..P]` then `ids[P..]` straight through, and with one that
/// prefilled all of `ids` in one go (window boundaries differ there).
pub fn resume_test(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut at, mut mtp) = (Vec::new(), None, None);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--ids-file" => {
                ids = std::fs::read_to_string(val()?)?
                    .split(|c: char| c.is_whitespace() || c == ',')
                    .filter(|w| !w.is_empty())
                    .map(|w| w.parse::<u32>())
                    .collect::<Result<_, _>>()?
            }
            "--at" => at = Some(val()?.parse::<usize>()?),
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            s => bail!("unknown argument {s}"),
        }
    }
    ensure!(ids.len() > 16, "need a prompt of more than 16 ids");
    let p = at.unwrap_or(ids.len() * 2 / 3 + 3);
    let opts = Opts {
        max_ctx: (ids.len() + 256).next_multiple_of(1024),
        mtp,
        ..Opts::default()
    };
    let mut e = Engine::load(&path, opts)?;
    e.use_graphs = true;
    e.warm()?;
    e.use_mtp = e.has_mtp();
    let v = e.hp.n_vocab;
    let run = |e: &mut Engine, segs: &[&[u32]], from: usize| -> Result<(Vec<f32>, u32)> {
        let mut rows = vec![0f32; (ids.len() - from) * v];
        let mut next = 0;
        for seg in segs {
            let mut cb = |pos0: usize, t: usize, lg: &[f32]| {
                for i in 0..t {
                    if pos0 + i >= from {
                        rows[(pos0 + i - from) * v..(pos0 + i - from + 1) * v]
                            .copy_from_slice(&lg[i * v..(i + 1) * v]);
                    }
                }
            };
            next = e.prefill(seg, MAX_T, Some(&mut cb))?;
        }
        Ok((rows, next))
    };
    let bits = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    };
    // Straight through, split at p.
    e.reset();
    let (split, n_split) = run(&mut e, &[&ids[..p], &ids[p..]], p)?;
    // All at once.
    e.reset();
    let (whole, n_whole) = run(&mut e, &[&ids], p)?;
    // Saved at p, disturbed, restored.
    e.reset();
    e.prefill(&ids[..p], MAX_T, None)?;
    let t = Instant::now();
    let st = e.save_running()?;
    let pos = e.save_positional(0, p)?;
    let save_ms = t.elapsed().as_secs_f64() * 1e3;
    let junk: Vec<u32> = (0..53).map(|i| 1000 + 37 * i).collect();
    e.prefill(&junk, MAX_T, None)?;
    let t = Instant::now();
    e.restore_running(&st, &ids[..p])?;
    let restore_ms = t.elapsed().as_secs_f64() * 1e3;
    let (resumed, n_resumed) = run(&mut e, &[&ids[p..]], p)?;
    // And from a fresh engine state (reset, then restore), as after another conversation.
    e.reset();
    e.prefill(&junk, MAX_T, None)?;
    let valid = lcp(&e.tokens, &ids[..p]);
    e.restore_positional(&pos, valid)?;
    e.restore_running(&st, &ids[..p])?;
    let (resumed2, _) = run(&mut e, &[&ids[p..]], p)?;
    println!(
        "resume test: {} ids, state saved at {p} ({:.0} MB + {:.0} MB positional, save {save_ms:.0} ms, restore {restore_ms:.0} ms)",
        ids.len(),
        st.bytes() as f64 / 1e6,
        pos.bytes() as f64 / 1e6
    );
    println!(
        "  resumed vs straight (same windows): {} of {} logits differ; next token {} vs {}",
        bits(&resumed, &split),
        split.len(),
        n_resumed,
        n_split
    );
    println!(
        "  resumed after other prefills vs straight: {} of {} differ",
        bits(&resumed2, &split),
        split.len()
    );
    println!("  split at {p} vs one prefill (different window boundaries): {} of {} differ; next {} vs {}", bits(&split, &whole), split.len(), n_split, n_whole);
    let ok = bits(&resumed, &split) == 0 && bits(&resumed2, &split) == 0;
    println!(
        "resume test: {}",
        if ok { "PASS (bitwise)" } else { "FAIL" }
    );
    Ok(())
}

/// The routing-count profile in the state directory.
const ROUTING: &str = "routing.bin";

/// `TANG_FLASH_STATE_DIR` or `~/.cache/tang/flash-serve/<model>`.
pub fn default_disk(gguf: &Path) -> Option<PathBuf> {
    match std::env::var("TANG_FLASH_STATE_DIR").as_deref() {
        Ok("0") | Ok("") => None,
        Ok(d) => Some(PathBuf::from(d)),
        Err(_) => {
            let name = gguf.file_stem()?.to_string_lossy().to_string();
            Some(
                PathBuf::from(std::env::var_os("HOME")?)
                    .join(".cache/tang/flash-serve")
                    .join(name),
            )
        }
    }
}

/// Prefill `ids` after the engine's sequence in windows of `chunk`, the MTP run only over
/// windows that reach position `mtp_from` (0: all). Returns the sampled next token, each
/// window's stats and the MTP time. The MTP's K/V for earlier positions stays unwritten: that
/// only changes its drafts' acceptance, never the output (measured: no change in tokens/window
/// at 2K and 10K with the last 512 positions).
pub fn prefill_windows<E: ServeEngine + ?Sized>(
    e: &mut E,
    ids: &[u32],
    chunk: usize,
    mtp_from: usize,
) -> Result<(u32, Vec<super::engine::WinStats>, f64)> {
    prefill_span(e, ids, chunk, mtp_from, 0, true)
}

/// [`prefill_windows`] with wide prefill windows (`Engine::prefill_wide`, widths 16/32/64 up
/// to `wide`; 0: none) for everything but the last `MAX_T` tokens when `last` (that window
/// runs the head), or for all of it otherwise (the result's next token is then meaningless).
pub fn prefill_span<E: ServeEngine + ?Sized>(
    e: &mut E,
    ids: &[u32],
    chunk: usize,
    mtp_from: usize,
    wide: usize,
    last: bool,
) -> Result<(u32, Vec<super::engine::WinStats>, f64)> {
    ensure!(!ids.is_empty(), "empty prompt");
    let pos_start = e.tokens().len();
    e.tokens_mut().extend_from_slice(ids);
    let end = e.tokens().len();
    let wide = if e.has_wide() { wide } else { 0 };
    let (mut pos, mut next, mut stats, mut mtp_ms) = (pos_start, 0, Vec::new(), 0.0);
    while pos < end {
        let left = end - pos;
        let room = if last { left.saturating_sub(1) } else { left };
        if let Some(&w) = [64usize, 32, 16].iter().find(|&&w| w <= wide && w <= room) {
            stats.push(e.prefill_wide(pos, w)?);
            pos += w;
            continue;
        }
        let t = chunk.min(left).clamp(1, MAX_T);
        let out = e.window(pos, t)?;
        stats.push(e.last());
        if e.use_mtp() && pos + t > mtp_from {
            let nexts: Vec<u32> = (0..t)
                .map(|i| e.tokens().get(pos + i + 1).copied().unwrap_or(out[t - 1]))
                .collect();
            let t0 = Instant::now();
            e.mtp_draft(pos, &nexts)?;
            mtp_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        next = out[t - 1];
        pos += t;
    }
    Ok((next, stats, mtp_ms))
}

/// `tang-llm flash-prefill-bench <gguf> --ids-file F [--n 2048,10240] [--mtp M] [--caps 0,16,88]
/// [--tails 0,512]`: prefill tok/s and the per-window split for the served prefill and its
/// variants (MTP over the last N positions only, GPU-streamed misses via `pcie_cap`), each
/// checked bitwise against the served path's last logits, with draft acceptance over 128
/// decoded tokens after it.
pub fn prefill_bench(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut ns, mut mtp) = (Vec::<u32>::new(), vec![2048usize, 10240], None);
    let (mut caps, mut tails) = (vec![16usize, 88], vec![0usize, 512]);
    let mut quick = false;
    let mut chunks: Vec<usize> = Vec::new();
    let mut wides: Vec<usize> = Vec::new();
    let list =
        |s: &str| -> Result<Vec<usize>> { s.split(',').map(|x| Ok(x.parse::<usize>()?)).collect() };
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--ids-file" => {
                for f in val()?.split(',') {
                    ids.extend(
                        std::fs::read_to_string(f)?
                            .split(|c: char| c.is_whitespace() || c == ',')
                            .filter(|w| !w.is_empty())
                            .map(|w| w.parse::<u32>())
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
            }
            "--n" => ns = list(val()?)?,
            "--caps" => caps = list(val()?)?,
            "--tails" => tails = list(val()?)?,
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            "--quick" => quick = true,
            "--chunks" => chunks = list(val()?)?,
            "--wide" => wides = list(val()?)?,
            s => bail!("unknown argument {s}"),
        }
    }
    if quick {
        caps.clear();
        tails = vec![512];
    }
    ensure!(!ids.is_empty(), "--ids-file");
    let max_n = *ns.iter().max().unwrap();
    let base = ids.clone();
    while ids.len() < max_n {
        ids.extend_from_slice(&base);
    }
    let opts = Opts {
        max_ctx: (max_n + 1024).next_multiple_of(1024),
        mtp,
        split: true,
        ..Opts::default()
    };
    let mut e = Engine::load(&path, opts)?;
    eprintln!("{}", e.load_report);
    e.use_graphs = true;
    e.warm()?;
    let has_mtp = e.has_mtp();
    let v = e.hp.n_vocab;
    let default_cap = e.pcie_cap;
    for &n in &ns {
        let p = &ids[..n];
        let mut configs: Vec<(String, usize, usize, usize)> = vec![(
            "served (mtp all, cap default)".into(),
            usize::MAX,
            default_cap,
            MAX_T,
        )];
        for &w in &wides {
            configs.push((
                format!("wide {w}, mtp last 512"),
                512,
                default_cap,
                1000 + w,
            ));
        }
        for &c in &chunks {
            configs.push((format!("chunk {c}, mtp last 512"), 512, default_cap, c));
        }
        for &t in &tails {
            configs.push((format!("mtp last {t}"), t, default_cap, MAX_T));
        }
        for &c in &caps {
            configs.push((format!("mtp last 512, pcie_cap {c}"), 512, c, MAX_T));
        }
        if !quick {
            configs.push(("served again".into(), usize::MAX, default_cap, MAX_T));
        }
        let mut want: Option<(u32, Vec<f32>)> = None;
        println!("prefill {n} tokens:");
        for (name, tail, cap, chunk) in configs {
            e.reset();
            e.use_mtp = has_mtp;
            if e.pcie_cap != cap {
                e.pcie_cap = cap;
                for t in 1..=MAX_T {
                    e.drop_window_graph(t, false);
                    e.drop_window_graph(t, true);
                }
            }
            let t0 = Instant::now();
            let (next, st, mtp_ms) = if chunk > 1000 {
                prefill_span(&mut e, p, MAX_T, n.saturating_sub(tail), chunk - 1000, true)?
            } else {
                prefill_windows(&mut e, p, chunk, n.saturating_sub(tail))?
            };
            let secs = t0.elapsed().as_secs_f64();
            let last_t = st.last().map_or(1, |s| s.t);
            let lg = e.logits(last_t)[(last_t - 1) * v..].to_vec();
            let same = match &want {
                None => {
                    want = Some((next, lg));
                    "reference".to_string()
                }
                Some((wn, wl)) => {
                    let d = wl
                        .iter()
                        .zip(&lg)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    format!(
                        "{} (next {} vs {}, {d} logits differ)",
                        if d == 0 && *wn == next {
                            "SAME"
                        } else {
                            "DIFFERS"
                        },
                        next,
                        wn
                    )
                }
            };
            // Decode 128 tokens with greedy MTP chains, for the drafts' acceptance.
            let (mut cur, mut toks, mut wins) = (next, 0usize, 0usize);
            let t1 = Instant::now();
            while toks < 128 && has_mtp {
                let d: Vec<u32> = e.mtp_last.iter().take(MAX_T - 1).map(|x| x.0).collect();
                let kept = e.verify(cur, &d)?;
                toks += kept.len();
                wins += 1;
                cur = *kept.last().unwrap();
            }
            let dsecs = t1.elapsed().as_secs_f64();
            let wide: Vec<&super::engine::WinStats> = st.iter().filter(|s| s.t > MAX_T).collect();
            if !wide.is_empty() {
                let n = wide.len() as f64;
                let ms = wide.iter().map(|s| s.wall_ms).sum::<f64>() / n;
                let pcie = wide.iter().map(|s| s.pcie).sum::<usize>() as f64 / n;
                println!(
                    "    wide windows: {} of T={}, {ms:.1} ms each, {:.0} distinct experts, {pcie:.0} over PCIe ({:.2} GB) per window",
                    wide.len(),
                    wide[0].t,
                    wide.iter().map(|s| s.distinct).sum::<usize>() as f64 / n,
                    pcie * tang_compute::flash::ExpertBlob::BYTES as f64 / 1e9
                );
            }
            let w = st.len().max(1) as f64;
            let m = |f: fn(&super::engine::WinStats) -> f64| st.iter().map(f).sum::<f64>() / w;
            let c = |f: fn(&super::engine::WinStats) -> usize| {
                st.iter().map(f).sum::<usize>() as f64 / w
            };
            println!(
                "  {name:<34} {:7.0} tok/s ({:.2} s) | window {:.2} ms: GPU {:.2}, wait plan {:.2}, wait CPU rows {:.2}, host CPU experts {:.2}, plan {:.2}, MTP {:.2} | experts/window {:.0} distinct, {:.1} CPU, {:.1} PCIe, swaps {:.1} | {} | decode 128: {:.2} tok/window, {:.0} tok/s",
                n as f64 / secs,
                secs,
                m(|s| s.wall_ms),
                m(|s| s.gpu_ms),
                m(|s| s.gpu_wait_a_ms),
                m(|s| s.gpu_wait_b_ms),
                m(|s| s.cpu_ms),
                m(|s| s.plan_ms),
                mtp_ms / w,
                c(|s| s.distinct),
                c(|s| s.missed),
                c(|s| s.pcie),
                c(|s| s.swaps),
                same,
                toks as f64 / wins.max(1) as f64,
                toks as f64 / dsecs.max(1e-9),
            );
        }
    }
    Ok(())
}

/// Decode `n` tokens after `ids` from a fresh sequence, with MTP chains as drafts or none.
fn decode_n(e: &mut Engine, ids: &[u32], n: usize, drafts: bool) -> Result<Vec<u32>> {
    e.reset();
    e.sampler_begin(ids.len())?;
    let mut cur = e.prefill(ids, MAX_T, None)?;
    let mut out = vec![cur];
    while out.len() < n {
        let d: Vec<u32> = if drafts {
            e.mtp_last
                .iter()
                .take((MAX_T - 1).min(n - out.len()))
                .map(|x| x.0)
                .collect()
        } else {
            Vec::new()
        };
        let kept = e.verify(cur, &d)?;
        out.extend_from_slice(&kept);
        cur = *kept.last().unwrap();
    }
    out.truncate(n);
    Ok(out)
}

/// `tang-llm flash-sampler-test <gguf> --ids-file F --mtp M [-n N]`: the head sampler against
/// the plain argmax (greedy, top-k 1: identical), and with and without drafts at the model
/// card's settings (identical: drafts never change a sample).
pub fn sampler_test(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut mtp, mut n) = (Vec::<u32>::new(), None, 96usize);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--ids-file" => {
                ids = std::fs::read_to_string(val()?)?
                    .split_whitespace()
                    .map(|w| w.parse::<u32>())
                    .collect::<Result<_, _>>()?
            }
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            "-n" => n = val()?.parse()?,
            s => bail!("unknown argument {s}"),
        }
    }
    let opts = Opts {
        max_ctx: (ids.len() + n + 64).next_multiple_of(1024),
        mtp,
        ..Opts::default()
    };
    let mut e = Engine::load(&path, opts)?;
    e.use_graphs = true;
    e.use_mtp = e.has_mtp();
    e.warm()?;
    let base = decode_n(&mut e, &ids, n, true)?;
    e.enable_sampler()?;
    e.warm()?;
    let mut ok = true;
    let mut check = |name: &str, a: &[u32], b: &[u32]| {
        let same = a == b;
        ok &= same;
        println!(
            "  {name}: {}{}",
            if same { "SAME" } else { "DIFFERS" },
            a.iter()
                .zip(b)
                .position(|(x, y)| x != y)
                .map_or(String::new(), |i| format!(" at {i}"))
        );
    };
    let set = |e: &mut Engine, temp: f32, k: usize, p: f32, pres: f32| {
        e.temperature = temp;
        e.seed = 1234;
        let s = e.sampler_mut().unwrap();
        s.top_k = k;
        s.top_p = p;
        s.presence = pres;
    };
    println!("sampler test, {} prompt ids, {n} tokens:", ids.len());
    set(&mut e, 0.0, 1, 1.0, 0.0);
    let g = decode_n(&mut e, &ids, n, true)?;
    check("greedy, sampler top-k 1 vs argmax", &g, &base);
    for (name, t, k, p, pres) in [
        ("thinking card (T 1.0, k 20, p 0.95)", 1.0, 20, 0.95, 0.0),
        (
            "answer card (T 0.7, k 20, p 0.8, presence 1.5)",
            0.7,
            20,
            0.8,
            1.5,
        ),
        ("greedy + presence 1.5", 0.0, 20, 1.0, 1.5),
    ] {
        set(&mut e, t, k, p, pres);
        let a = decode_n(&mut e, &ids, n, false)?;
        let b = decode_n(&mut e, &ids, n, true)?;
        check(&format!("{name}: drafts vs none"), &b, &a);
        let c = decode_n(&mut e, &ids, n, true)?;
        check(&format!("{name}: repeat"), &c, &a);
        let differs = a.iter().zip(&base).filter(|(x, y)| x != y).count();
        println!(
            "    differs from greedy in {differs} of {n}; first {:?}",
            &a[..12.min(a.len())]
        );
    }
    println!("sampler test: {}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}

/// `tang-llm flash-serve-replay <gguf> --mtp M [--wide W] [--state-dir D|none] [--fresh] [--profile P] [--dump-routing P] [--dump-logits P] IDS...`:
/// the served prefill (`FlashServe::prefill`, saved states and all) over recorded prompts
/// (`--dump-ids` files), each after dropping the live sequence (`--fresh`: and every saved
/// state), with per-span timing (`TANG_FLASH_SPAN_LOG=1` is set).
pub fn serve_replay(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let gguf = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut mtp, mut disk, mut files, mut fresh) = (None, None, Vec::new(), false);
    let mut wide: Option<usize> = None;
    let mut keep = usize::MAX;
    let (mut profile, mut dump_routing, mut dump_logits) = (None, None, None);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            "--wide" => wide = Some(val()?.parse()?),
            "--state-dir" => {
                let v = val()?;
                disk = (v != "none").then(|| PathBuf::from(v));
            }
            "--fresh" => fresh = true,
            "--keep-upto" => keep = val()?.parse()?,
            "--profile" => profile = Some(PathBuf::from(val()?)),
            "--dump-routing" => dump_routing = Some(PathBuf::from(val()?)),
            // The head buffer's bits after each prompt (`P.N`), to compare variants with `cmp`.
            "--dump-logits" => dump_logits = Some(PathBuf::from(val()?)),
            f => files.push(PathBuf::from(f)),
        }
    }
    if let Some(w) = wide {
        std::env::set_var("TANG_FLASH_WIDE", w.to_string());
    }
    std::env::set_var("TANG_FLASH_SPAN_LOG", "1");
    let s = Settings {
        sampling: SamplingMode::Greedy,
        gguf,
        mtp,
        ctx: 32_768,
        temp: 0.0,
        snapshots: 6,
        snapshot_bytes: 12_000_000_000,
        disk,
        dump: None,
        warm: Vec::new(),
        suffix: false,
        warm_dir: None,
        batch: 0,
        profile,
    };
    let mut me = FlashServe::load(&s)?;
    for f in &files {
        let ids: Vec<u32> = std::fs::read_to_string(f)?
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|w| !w.is_empty())
            .map(|w| w.parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;
        let text = me.tok.decode(&ids)?;
        me.e.reset();
        if fresh {
            me.snaps.clear();
        }
        me.snaps.retain(|s| s.tokens.len() <= keep);
        me.line = Line::default();
        let t = Instant::now();
        let (reused, next) = me.prefill(&ids, &text)?;
        let secs = t.elapsed().as_secs_f64();
        if let Some(p) = &dump_logits {
            let b: Vec<u8> =
                me.e.logits(MAX_T)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
            let n = files.iter().position(|g| g == f).unwrap_or(0);
            std::fs::write(p.with_extension(n.to_string()), b)?;
        }
        println!(
            "{}: {} tokens, {reused} reused, {:.2} s = {:.0} tok/s (snapshots {:.0} ms, restore {:.0} ms), next {next}",
            f.display(),
            ids.len(),
            secs,
            (ids.len() - reused) as f64 / secs,
            me.line.snap_ms,
            me.line.restore_ms
        );
    }
    if let Some(p) = &dump_routing {
        me.e.save_routing(p)?;
    }
    Ok(())
}
