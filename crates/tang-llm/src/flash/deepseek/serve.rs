//! DeepSeek-V4.1-Flash behind the OpenAI-compatible server ([`crate::server`]): the chat
//! template, `tokenizer.json`, the device forward, and DeepSeek's DSML tool calls.
//!
//! Deliberately minimal, like GLM's first serve glue:
//! - One sequence at a time. Each request prefills its prompt from position 0 in chunks of
//!   `DS_PREFILL_CHUNK` tokens (1024) and then decodes; there is no prefix reuse across
//!   requests yet.
//! - Requests decode speculatively with DSpark (`DS_SPEC=0` turns it off): each forward
//!   verifies the sampled token and its drafts as one window ([`forward::generate`]);
//!   greedy drafts [`SERVE_DRAFT_N`] a window, sampled [`SERVE_DRAFT_N_SAMPLED`] (`DS_DRAFT_N`
//!   sets both), `DS_DRAFT_CONF` cuts it at the
//!   confidence head ([`Config::draft_keep`]). Greedy
//!   gives the same tokens as plain greedy decoding; sampled requests use speculative
//!   (rejection) sampling, the same distribution as plain temperature/top-p sampling
//!   ([`SpecSampler`]). Sampled spec needs the engine's `decode_spec` to take a
//!   [`forward::SpecPick`] ([`SPEC_SAMPLING`]); until then sampled requests decode one token
//!   at a time.
//! - Sampling on the host from the last logits: greedy at temperature 0, else top-p.
//!
//! The reply streams as reasoning (inside `<think>`), text, and tool calls. DeepSeek writes a
//! tool call as
//! `<｜DSML｜ calls><｜DSML｜ invoke name="N"><｜DSML｜ parameter name="K" string="true">V
//! </｜DSML｜ parameter>…</｜DSML｜ invoke></｜DSML｜ calls>`, where `｜DSML｜` is one token;
//! [`Dsml`] turns that into [`Piece::ToolCall`]s.

use super::engine::GpuModel;
use super::forward::{self, Decoder, SpecPick, SpecSampler, SpecStep};
use super::Config;
use crate::chat::{Chunk, Detok, Parser, Piece, Tags, Template};
use crate::engine::Outcome;
use crate::engine::{Finish, Request, Usage};
use crate::server::{Backend, DynLane, Inflight};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

const DSML: &str = "｜DSML｜";

/// One GPU, or expert-parallel across several (`--devices`).
pub enum Engine {
    Single(GpuModel),
    #[cfg(feature = "ep")]
    Ep(super::ep::EpGroup),
}

/// The served greedy draft length (`DS_DRAFT_N_GREEDY`; `DS_DRAFT_N` overrides both modes):
/// one past the trained block of 5.
/// Measured on 8x H100 (1017 weights, 200 tokens, greedy): n 4 / 5 / 6 / 7 / 8 = 136 / 149 /
/// 158 / 151 / 96 tok/s, 3.8 / 4.2 / 4.9 / 4.9 / 5.0 tokens a step (t 9 leaves the 8-row
/// kernels).
pub const SERVE_DRAFT_N: usize = 6;

/// As [`SERVE_DRAFT_N`] for sampled requests (`DS_DRAFT_N_SAMPLED`): the trained 5. At n 6
/// sampled T=0.6 averaged ~141 tok/s against ~147 at n 5 (the 6th sampled draft lands less).
pub const SERVE_DRAFT_N_SAMPLED: usize = 5;

impl Engine {
    /// One GPU, or expert-parallel over `devices` (more than one; needs the `ep` feature).
    pub fn open(dir: &Path, devices: &[usize]) -> Result<Self> {
        // every rank's Config::load reads it
        for (k, n) in [
            ("DS_DRAFT_N_GREEDY", SERVE_DRAFT_N),
            ("DS_DRAFT_N_SAMPLED", SERVE_DRAFT_N_SAMPLED),
        ] {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, n.to_string());
            }
        }
        if devices.len() <= 1 {
            return Ok(Engine::Single(GpuModel::open_on(
                dir,
                devices.first().copied().unwrap_or(0),
            )?));
        }
        #[cfg(feature = "ep")]
        {
            Ok(Engine::Ep(super::ep::EpGroup::open(
                dir,
                devices,
                super::ep::Backend::Nccl,
            )?))
        }
        #[cfg(not(feature = "ep"))]
        bail!("--devices needs a build with --features ep")
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Engine::Single(m) => m.reset(),
            #[cfg(feature = "ep")]
            Engine::Ep(g) => g.reset(),
        }
    }
}

/// Prefill `ids` from position `sp` through `forward` at most [`prefill_chunk`] tokens at a
/// time, returning the last chunk's outputs. A step's scratch grows with its token count
/// (the indexer scores every position against every compressed slot), so a 10k-token prompt
/// in one forward runs out of device memory where ten 1k-token ones don't; the chunks produce
/// the same state (`gpu_tiny_chunk_and_rewind_exact`).
fn feed_chunked(
    ids: &[usize],
    sp: usize,
    mut forward: impl FnMut(&[usize], usize) -> Result<(Vec<f32>, Vec<f32>)>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let chunk = prefill_chunk();
    if ids.len() <= chunk {
        return forward(ids, sp);
    }
    let mut out = None;
    for (i, c) in ids.chunks(chunk).enumerate() {
        out = Some(forward(c, sp + i * chunk)?);
    }
    Ok(out.unwrap())
}

/// `DS_SERVE_PROFILE`, when set to anything but "0": "direct" or the graph path.
fn serve_profile() -> Option<String> {
    std::env::var("DS_SERVE_PROFILE")
        .ok()
        .filter(|v| !v.is_empty() && v != "0")
}

/// Tokens per prefill forward (`DS_PREFILL_CHUNK`, default 1024).
fn prefill_chunk() -> usize {
    std::env::var("DS_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1024)
}

impl Decoder for Engine {
    fn config(&self) -> &Config {
        match self {
            Engine::Single(m) => &m.cpu.cfg,
            #[cfg(feature = "ep")]
            Engine::Ep(g) => g.config(),
        }
    }
    fn feed(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        feed_chunked(ids, sp, |ids, sp| match self {
            Engine::Single(m) => m.forward(ids, sp, None),
            #[cfg(feature = "ep")]
            Engine::Ep(g) => g.forward(ids, sp),
        })
    }
    fn seed_drafter(&mut self, main: &[f32]) -> Result<()> {
        match self {
            Engine::Single(m) => m.forward_spec(0, main, 0).map(|_| ()),
            #[cfg(feature = "ep")]
            Engine::Ep(g) => g.forward_spec(0, main, 0).map(|_| ()),
        }
    }
    fn spec_ok(&self, greedy: bool) -> bool {
        greedy || SPEC_SAMPLING
    }
    fn spec_step(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep> {
        match self {
            Engine::Single(m) => m.decode_spec(token, main, sp, &mut |_, _| {}, pick),
            #[cfg(feature = "ep")]
            Engine::Ep(g) => {
                // the ranks each run the sampler, so they need its concrete state
                let sampler = pick
                    .as_sampler()
                    .ok_or_else(|| anyhow::anyhow!("EP spec needs a SpecSampler"))?;
                g.decode_spec(token, main, sp, sampler)
            }
        }
    }
}

pub struct DsServe {
    m: Engine,
    tok: tokenizers::Tokenizer,
    tpl: Template,
    tags: Tags,
    eos: Vec<u32>,
    ctx: usize,
    /// Continuous batching (`DS_SERVE_SLOTS`, default 1 = one request at a time): which
    /// sequence slots a running request holds.
    busy: Vec<bool>,
}

/// The served sequence slots (`DS_SERVE_SLOTS`; 1 = no batching).
fn serve_slots() -> usize {
    std::env::var("DS_SERVE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1)
}

/// The positions each extra slot holds (`DS_SLOT_CTX`, default 32768, at most the model's
/// context; slot 0 keeps the full context).
fn slot_ctx(max: usize) -> usize {
    std::env::var("DS_SLOT_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32768)
        .min(max)
}

/// One sequence of the slot batcher ([`slots_step`]): token ids only, greedy. The text side
/// (detokenizer, parser, stop strings) is the caller's.
pub(crate) struct SlotSeq {
    pub slot: usize,
    prompt: Vec<usize>,
    /// Positions fed to the slot (prompt chunks while prefilling, then decode tokens).
    pos: usize,
    /// The token to feed at `pos` in the next decode step.
    feed: Option<usize>,
    /// Tokens produced so far, and the limit.
    pub n: usize,
    max_new: usize,
    /// Set once the sequence ends; the caller may set it too (a stop string, a cancel).
    pub done: Option<Finish>,
}

impl SlotSeq {
    pub fn new(slot: usize, prompt: Vec<usize>, max_new: usize) -> Self {
        Self {
            slot,
            prompt,
            pos: 0,
            feed: None,
            n: 0,
            max_new,
            done: None,
        }
    }

    /// Still prefilling (prompt tokens left to feed).
    pub fn prefilling(&self) -> bool {
        self.pos < self.prompt.len() && self.done.is_none()
    }

    /// The model's next token `cand` (as `forward::generate`): EOS ends it unproduced; else
    /// it is produced, and fed next step unless the limit or the slot's cap is reached.
    fn take(&mut self, cand: usize, eos: &[usize], cap: usize, out: &mut Vec<usize>) {
        if eos.contains(&cand) {
            self.done = Some(Finish::Stop);
            return;
        }
        self.n += 1;
        out.push(cand);
        if self.n >= self.max_new || self.pos + 1 >= cap {
            self.done = Some(Finish::Length);
        } else {
            self.feed = Some(cand);
        }
    }
}

/// Prompt chunks one batcher step feeds, over all prefilling sequences
/// (`DS_SERVE_PREFILL_CHUNKS`, default 1; 0 = every prompt whole in its first step). Each
/// chunk is a [`prefill_chunk`]-token forward, so a long prompt delays the running streams'
/// next token by at most this many chunks per step.
fn prefill_chunks_per_step() -> usize {
    std::env::var("DS_SERVE_PREFILL_CHUNKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

/// One batcher step over `seqs` (distinct slots): prompt chunks for the prefilling
/// sequences, oldest (lowest index) first, at most [`prefill_chunks_per_step`] of them
/// (each the chunk the single-request path would feed there, so the slot ends in the same
/// state), then every running sequence decodes one token in ONE
/// [`GpuModel::forward_rows`] step. Returns the tokens each sequence produced this step
/// (index-aligned with `seqs`). Greedy.
pub(crate) fn slots_step(
    m: &mut GpuModel,
    seqs: &mut [&mut SlotSeq],
    eos: &[usize],
) -> Result<Vec<Vec<usize>>> {
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); seqs.len()];
    let (chunk, budget) = (prefill_chunk(), prefill_chunks_per_step());
    let mut left = if budget == 0 { usize::MAX } else { budget };
    for (i, q) in seqs.iter_mut().enumerate() {
        while left > 0 && q.prefilling() {
            m.select_slot(q.slot)?;
            if q.pos == 0 {
                m.reset()?;
            }
            let end = (q.pos + chunk).min(q.prompt.len());
            let (lg, _) = m.forward(&q.prompt[q.pos..end], q.pos, None)?;
            q.pos = end;
            left = left.saturating_sub(1);
            if !q.prefilling() {
                let cap = m.slot_cap(q.slot);
                q.take(forward::argmax(&lg), eos, cap, &mut out[i]);
            }
        }
    }
    let live: Vec<usize> = (0..seqs.len())
        .filter(|&i| seqs[i].done.is_none() && seqs[i].feed.is_some() && out[i].is_empty())
        .collect();
    if live.is_empty() {
        return Ok(out);
    }
    let rows: Vec<(usize, usize, usize)> = live
        .iter()
        .map(|&i| (seqs[i].slot, seqs[i].feed.unwrap(), seqs[i].pos))
        .collect();
    let next = m.forward_rows(&rows)?;
    for (j, &i) in live.iter().enumerate() {
        let q = &mut *seqs[i];
        q.feed = None;
        q.pos += 1;
        let cap = m.slot_cap(q.slot);
        q.take(next[j], eos, cap, &mut out[i]);
    }
    Ok(out)
}

/// A request in the batch ([`Backend::admit`]): its sequence and its text pipeline.
struct DsStream {
    seq: SlotSeq,
    think: bool,
    out: Vec<u32>,
    detok: Detok,
    parser: Parser,
    dsml: Dsml,
    called: bool,
    stop: Vec<String>,
    usage: Usage,
    t0: Instant,
    t1: Option<Instant>,
}

impl DsServe {
    /// `dir`: the checkpoint with `tokenizer.json`, `tokenizer_config.json`,
    /// `chat_template.jinja`, `inference/config.json` and `engram.json`.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_on(dir, &[])
    }

    /// `devices`: GPU ordinals for expert parallelism (needs the `ep` feature); empty or one
    /// means a single GPU.
    pub fn load_on(dir: &Path, devices: &[usize]) -> Result<Self> {
        let t = Instant::now();
        // `DS_SERVE_PROFILE=1` prints deepseek-profile's table after every request (`=direct`
        // forces the direct path so the steps' interiors are timed; see docs/observability.md)
        if let Some(mode) = serve_profile() {
            super::obs::set(2, 1, mode == "direct");
            super::obs::quiet(true);
            super::obs::collect(true);
        }
        let mut m = Engine::open(dir, devices)?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
        let tpl = Template::load(dir)?;
        let tags = Tags::new(|s| tok.token_to_id(s));
        let eos = ["<｜end▁of▁sentence｜>"]
            .iter()
            .filter_map(|s| tok.token_to_id(s))
            .collect::<Vec<_>>();
        if eos.is_empty() {
            bail!("tokenizer has no <｜end▁of▁sentence｜>");
        }
        let ctx = m.config().max_seq_len;
        let how = match &m {
            Engine::Single(g) => format!(
                "1 GPU, expert slots {}",
                if g.expert_slots == 0 {
                    "all".to_string()
                } else {
                    g.expert_slots.to_string()
                }
            ),
            #[cfg(feature = "ep")]
            Engine::Ep(g) => format!("expert parallel over {} GPUs (NCCL)", g.world()),
        };
        eprintln!(
            "tang-llm: deepseek ready in {:.1} s: ctx {ctx}, {how}",
            t.elapsed().as_secs_f64()
        );
        let slots = serve_slots();
        if slots > 1 {
            match &mut m {
                Engine::Single(g) => {
                    let cap = slot_ctx(ctx);
                    g.add_slots(slots, cap)?;
                    eprintln!("tang-llm: deepseek batching: {slots} slots, {cap} positions each");
                }
                #[cfg(feature = "ep")]
                Engine::Ep(_) => bail!("DS_SERVE_SLOTS: expert parallel doesn't batch yet"),
            }
        }
        Ok(Self {
            m,
            tok,
            tpl,
            tags,
            eos,
            ctx,
            busy: vec![false; slots],
        })
    }
}

/// Whether the GPU `decode_spec` takes a [`SpecPick`] (speculative sampling). The port: give
/// `GpuModel::forward_spec` a `draw: &mut dyn FnMut(&[f32]) -> usize` for the Markov loop
/// (as `Model::forward_spec_with`), and `GpuModel::decode_spec` a trailing
/// `pick: &mut dyn SpecPick` used as `Model::decode_spec` does (`forward::verify_rows`); then
/// flip this and call it below.
const SPEC_SAMPLING: bool = true;

impl Decoder for GpuModel {
    fn config(&self) -> &Config {
        &self.cpu.cfg
    }
    fn feed(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        feed_chunked(ids, sp, |ids, sp| self.forward(ids, sp, None))
    }
    fn seed_drafter(&mut self, main: &[f32]) -> Result<()> {
        self.forward_spec(0, main, 0).map(|_| ())
    }
    fn spec_ok(&self, greedy: bool) -> bool {
        greedy || SPEC_SAMPLING
    }
    fn spec_step(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep> {
        self.decode_spec(token, main, sp, &mut |_, _| {}, pick)
    }
}

/// Pulls DSML tool calls out of the text stream; everything else passes through. Holds back
/// text that could still turn into `<｜DSML｜ calls>`.
#[derive(Default)]
pub struct Dsml {
    buf: String,
    in_calls: bool,
}

impl Dsml {
    fn open() -> String {
        format!("<{DSML} calls>")
    }

    fn close() -> String {
        format!("</{DSML} calls>")
    }

    pub fn push(&mut self, s: &str) -> Vec<Piece> {
        self.buf.push_str(s);
        let mut out = Vec::new();
        loop {
            if self.in_calls {
                let Some(end) = self.buf.find(&Self::close()) else {
                    break;
                };
                out.extend(parse_calls(&self.buf[..end]));
                self.buf.drain(..end + Self::close().len());
                self.in_calls = false;
                continue;
            }
            let open = Self::open();
            if let Some(at) = self.buf.find(&open) {
                let before = self.buf[..at].trim_end().to_string();
                if !before.is_empty() {
                    out.push(Piece::Text(before));
                }
                self.buf.drain(..at + open.len());
                self.in_calls = true;
                continue;
            }
            // flush all but a tail that may be the start of the marker
            let keep = (1..open.len().min(self.buf.len()) + 1)
                .rev()
                .find(|&n| {
                    self.buf.is_char_boundary(self.buf.len() - n)
                        && open.starts_with(&self.buf[self.buf.len() - n..])
                })
                .unwrap_or(0);
            // and the whitespace before it: the template puts "\n\n" ahead of a calls block,
            // which a tool call drops
            let cut = self.buf[..self.buf.len() - keep].trim_end().len();
            if cut > 0 {
                out.push(Piece::Text(self.buf[..cut].to_string()));
                self.buf.drain(..cut);
            }
            break;
        }
        out
    }

    pub fn finish(&mut self) -> Vec<Piece> {
        let rest = std::mem::take(&mut self.buf);
        if self.in_calls {
            // an unterminated block: parse what's there
            return parse_calls(&rest);
        }
        if rest.is_empty() {
            Vec::new()
        } else {
            vec![Piece::Text(rest)]
        }
    }
}

/// The invokes inside a `calls` block.
fn parse_calls(body: &str) -> Vec<Piece> {
    let inv_open = format!("<{DSML} invoke name=\"");
    let inv_close = format!("</{DSML} invoke>");
    let par_open = format!("<{DSML} parameter name=\"");
    let par_close = format!("</{DSML} parameter>");
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(i) = rest.find(&inv_open) {
        rest = &rest[i + inv_open.len()..];
        let Some(q) = rest.find('"') else { break };
        let name = rest[..q].to_string();
        let end = rest.find(&inv_close).unwrap_or(rest.len());
        let mut inv = &rest[q..end];
        let mut args = serde_json::Map::new();
        while let Some(j) = inv.find(&par_open) {
            inv = &inv[j + par_open.len()..];
            let Some(q) = inv.find('"') else { break };
            let key = inv[..q].to_string();
            let Some(gt) = inv.find('>') else { break };
            let is_string = inv[q..gt].contains("string=\"true\"");
            let vend = inv[gt + 1..]
                .find(&par_close)
                .map_or(inv.len(), |e| gt + 1 + e);
            let raw = &inv[gt + 1..vend];
            let v = if is_string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
            };
            args.insert(key, v);
            inv = &inv[vend.min(inv.len())..];
        }
        out.push(Piece::ToolCall {
            name,
            arguments: Value::Object(args),
        });
        rest = &rest[end.min(rest.len())..];
    }
    out
}

impl Backend for DsServe {
    fn complete(
        &mut self,
        req: &Request,
        on: &mut dyn FnMut(Piece) -> bool,
    ) -> Result<(Finish, Usage)> {
        // the lab switch for `DS_SERVE_PROFILE`: the file /tmp/ds-profile-direct existing
        // runs this request on the direct path (step interiors timed), else the graph path
        let prof_direct = serve_profile().is_some() && {
            let d = std::path::Path::new("/tmp/ds-profile-direct").exists();
            super::obs::set(2, 1, d);
            d
        };
        let think = req.think.unwrap_or(true);
        let prompt = self
            .tpl
            .render(&req.messages, req.tools.as_ref(), Some(think))?;
        let enc = self
            .tok
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids: Vec<usize> = enc.get_ids().iter().map(|&i| i as usize).collect();
        let max_new = req
            .max_tokens
            .unwrap_or(self.ctx)
            .min(self.ctx.saturating_sub(ids.len()));
        if ids.is_empty() || max_new == 0 {
            bail!(
                "prompt of {} tokens leaves no room in a {}-token context",
                ids.len(),
                self.ctx
            );
        }
        let mut usage = Usage::default();
        self.m.reset()?;
        let mut pick = SpecSampler::new(
            req.sampling.temperature,
            req.sampling.top_p,
            req.sampling.seed ^ 0x5eed,
        );
        let (mut detok, mut parser, mut dsml) =
            (Detok::default(), Parser::new(think), Dsml::default());
        let mut out: Vec<u32> = Vec::new();
        let mut called = false;
        let mut emit = |pieces: Vec<Piece>, dsml: &mut Dsml, called: &mut bool| -> bool {
            for p in pieces {
                let ps = match p {
                    Piece::Text(s) => dsml.push(&s),
                    other => vec![other],
                };
                for p in ps {
                    *called |= matches!(p, Piece::ToolCall { .. });
                    if !on(p) {
                        return false;
                    }
                }
            }
            true
        };
        let spec = std::env::var("DS_SPEC").map_or(true, |v| v != "0");
        let eos: Vec<usize> = self.eos.iter().map(|&e| e as usize).collect();
        let (tok, tags, stop) = (&self.tok, &self.tags, &req.stop);
        let stop_toks = stop.iter().map(|s| s.len()).max().unwrap_or(0) + 4;
        let decode = |ids: &[u32]| -> Result<String> {
            tok.decode(ids, false)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))
        };
        let mut err = None;
        let mut finish = forward::generate(
            &mut self.m,
            &ids,
            max_new,
            &mut pick,
            spec,
            &eos,
            &mut usage,
            &mut |next| {
                out.push(next as u32);
                let chunks = match detok.push(&out, tags, decode) {
                    Ok(c) => c,
                    Err(e) => {
                        err = Some(e);
                        return Some(Finish::Cancelled);
                    }
                };
                for ch in chunks {
                    let pieces = match ch {
                        Chunk::Text(s) => parser.push_text(&s),
                        Chunk::Tag(t) => parser.push_tag(t),
                    };
                    if !emit(pieces, &mut dsml, &mut called) {
                        return Some(Finish::Cancelled);
                    }
                }
                if !stop.is_empty() {
                    // every (byte-level BPE) token is at least one byte of the text, so the
                    // last `longest stop + 4` tokens hold any stop the text ends with (the
                    // 4 cover a split UTF-8 char at the cut); not the whole output each token
                    match decode(&out[out.len().saturating_sub(stop_toks)..]) {
                        Ok(text) if stop.iter().any(|s| text.ends_with(s.as_str())) => {
                            return Some(Finish::Stop)
                        }
                        Ok(_) => {}
                        Err(e) => {
                            err = Some(e);
                            return Some(Finish::Cancelled);
                        }
                    }
                }
                None
            },
        )
        .context("decode")?;
        if let Some(e) = err {
            return Err(e);
        }
        let tail = parser.finish();
        emit(tail, &mut dsml, &mut called);
        for p in dsml.finish() {
            called |= matches!(p, Piece::ToolCall { .. });
            on(p);
        }
        if called && finish == Finish::Stop {
            finish = Finish::ToolCalls;
        }
        if serve_profile().is_some() {
            let recs = super::obs::take();
            super::obs::collect(true);
            let o = ProfileOpts {
                steps: 0,
                prompt_tokens: ids.len(),
                direct: prof_direct,
                graph_debug: false,
                spec: spec,
            };
            eprintln!("{}", profile_table(&recs, &o, 0.0, &usage));
        }
        Ok((finish, usage))
    }

    fn context_window(&self) -> usize {
        self.ctx
    }

    fn max_batch(&self) -> usize {
        self.busy.len()
    }

    fn can_admit(&self, _key: Option<&str>) -> bool {
        self.busy.len() > 1 && self.busy.iter().any(|b| !b)
    }

    fn admit(&mut self, req: &Request, _prompt: Option<(String, Vec<u32>)>) -> Result<Inflight> {
        let Engine::Single(g) = &self.m else {
            bail!("deepseek batching needs one GPU");
        };
        if req.sampling.temperature > 0.0 {
            bail!("deepseek batching is greedy only for now (temperature 0)");
        }
        let slot = self
            .busy
            .iter()
            .position(|b| !b)
            .context("no free sequence slot")?;
        let think = req.think.unwrap_or(true);
        let prompt = self
            .tpl
            .render(&req.messages, req.tools.as_ref(), Some(think))?;
        let enc = self
            .tok
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids: Vec<usize> = enc.get_ids().iter().map(|&i| i as usize).collect();
        let cap = g.slot_cap(slot);
        let max_new = req
            .max_tokens
            .unwrap_or(cap)
            .min(cap.saturating_sub(ids.len()));
        if ids.is_empty() || max_new == 0 {
            bail!(
                "prompt of {} tokens leaves no room in a {cap}-token slot",
                ids.len()
            );
        }
        self.busy[slot] = true;
        Ok(Box::new(DsStream {
            seq: SlotSeq::new(slot, ids, max_new),
            think,
            out: Vec::new(),
            detok: Detok::default(),
            parser: Parser::new(think),
            dsml: Dsml::default(),
            called: false,
            stop: req.stop.clone(),
            usage: Usage::default(),
            t0: Instant::now(),
            t1: None,
        }))
    }

    fn step(&mut self, lanes: &mut [DynLane<'_>]) -> Result<()> {
        let Engine::Single(g) = &mut self.m else {
            bail!("deepseek batching needs one GPU");
        };
        let eos: Vec<usize> = self.eos.iter().map(|&e| e as usize).collect();
        let mut streams: Vec<&mut DsStream> = Vec::with_capacity(lanes.len());
        let mut ons: Vec<&mut dyn FnMut(Piece) -> bool> = Vec::with_capacity(lanes.len());
        for l in lanes.iter_mut() {
            let s: &mut DsStream = l
                .stream
                .downcast_mut()
                .context("a stream this backend admitted")?;
            if s.seq.done.is_none() && l.ctl.cancelled() {
                s.seq.done = Some(Finish::Cancelled);
            }
            streams.push(s);
            ons.push(&mut *l.on);
        }
        let shared = streams
            .iter()
            .filter(|s| s.seq.done.is_none() && !s.seq.prefilling())
            .count();
        let active: Vec<bool> = streams.iter().map(|s| s.seq.done.is_none()).collect();
        let was_prefill: Vec<bool> = streams.iter().map(|s| s.seq.prefilling()).collect();
        let toks = {
            let mut seqs: Vec<&mut SlotSeq> = streams
                .iter_mut()
                .filter(|s| s.seq.done.is_none())
                .map(|s| &mut s.seq)
                .collect();
            slots_step(g, &mut seqs, &eos)?
        };
        let (tok, tags) = (&self.tok, &self.tags);
        let decode = |ids: &[u32]| -> Result<String> {
            tok.decode(ids, false)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))
        };
        let mut k = 0;
        for (li, (s, on)) in streams.iter_mut().zip(ons.iter_mut()).enumerate() {
            if active[li] {
                let (produced, pre) = (&toks[k], was_prefill[li]);
                k += 1;
                if pre {
                    s.usage.prefill_chunks += 1;
                    if !s.seq.prefilling() {
                        s.usage.prompt_tokens = s.seq.prompt.len();
                        s.usage.prefill_tok_s =
                            s.seq.prompt.len() as f64 / s.t0.elapsed().as_secs_f64().max(1e-9);
                        s.t1 = Some(Instant::now());
                    }
                } else {
                    s.usage.decode_steps += 1;
                    s.usage.shared_streams += shared.saturating_sub(1);
                }
                for &t in produced {
                    s.usage.completion_tokens += 1;
                    s.out.push(t as u32);
                    let chunks = s.detok.push(&s.out, tags, decode)?;
                    let mut pieces = Vec::new();
                    for ch in chunks {
                        pieces.extend(match ch {
                            Chunk::Text(x) => s.parser.push_text(&x),
                            Chunk::Tag(tg) => s.parser.push_tag(tg),
                        });
                    }
                    if !emit_pieces(pieces, &mut s.dsml, &mut s.called, &mut **on) {
                        s.seq.done = Some(Finish::Cancelled);
                    }
                    if s.seq.done.is_none() && !s.stop.is_empty() {
                        let n = s.stop.iter().map(|x| x.len()).max().unwrap_or(0) + 4;
                        let text = decode(&s.out[s.out.len().saturating_sub(n)..])?;
                        if s.stop.iter().any(|x| text.ends_with(x.as_str())) {
                            s.seq.done = Some(Finish::Stop);
                        }
                    }
                }
            }
            if s.seq.done.is_some() && s.t1.is_some() {
                // the tails, once
                let tail = s.parser.finish();
                emit_pieces(tail, &mut s.dsml, &mut s.called, &mut **on);
                for p in s.dsml.finish() {
                    s.called |= matches!(p, Piece::ToolCall { .. });
                    on(p);
                }
                if s.called && s.seq.done == Some(Finish::Stop) {
                    s.seq.done = Some(Finish::ToolCalls);
                }
                let t1 = s.t1.take().unwrap();
                s.usage.decode_tok_s =
                    s.usage.completion_tokens as f64 / t1.elapsed().as_secs_f64().max(1e-9);
            }
        }
        Ok(())
    }

    fn retire(&mut self, s: Inflight) -> Result<Outcome> {
        let s: Box<DsStream> = s
            .downcast()
            .map_err(|_| anyhow::anyhow!("a stream this backend admitted"))?;
        if let Some(b) = self.busy.get_mut(s.seq.slot) {
            *b = false;
        }
        let _ = s.think;
        Ok(Outcome::Done(
            s.seq.done.unwrap_or(Finish::Cancelled),
            s.usage,
        ))
    }

    fn stream_state(&self, s: &Inflight) -> (bool, bool) {
        match s.downcast_ref::<DsStream>() {
            Some(s) => (s.seq.done.is_some(), s.seq.prefilling()),
            None => (true, false),
        }
    }
}

/// Text pieces to `on`, DSML tool calls parsed out of the text (as `complete`'s `emit`);
/// false once `on` refuses one (the client went away).
fn emit_pieces(
    pieces: Vec<Piece>,
    dsml: &mut Dsml,
    called: &mut bool,
    on: &mut dyn FnMut(Piece) -> bool,
) -> bool {
    for p in pieces {
        let ps = match p {
            Piece::Text(s) => dsml.push(&s),
            other => vec![other],
        };
        for p in ps {
            *called |= matches!(p, Piece::ToolCall { .. });
            if !on(p) {
                return false;
            }
        }
    }
    true
}

/// `tang-llm deepseek-profile`'s settings.
#[derive(Clone, Debug)]
pub struct ProfileOpts {
    /// Decode / verify forwards after the prefill.
    pub steps: usize,
    pub prompt_tokens: usize,
    /// Force the direct path (`DS_OBS_DIRECT=1`) so decode steps' interiors are timed.
    pub direct: bool,
    /// Print each change of a step kind's path (graph capture / replay / direct).
    pub graph_debug: bool,
    /// Speculative decoding with DSpark (as `serve`; `DS_SPEC=0` turns it off).
    pub spec: bool,
}

/// Load the model, prefill a synthetic prompt and run `steps` decode or verify forwards
/// (greedy) with the step timeline at `DS_OBS=2` on every step; the table as text.
pub fn profile(dir: &Path, devices: &[usize], o: &ProfileOpts) -> Result<String> {
    use super::obs;
    obs::set(2, 1, o.direct);
    obs::quiet(true);
    let t0 = Instant::now();
    let mut m = Engine::open(dir, devices)?;
    let load_s = t0.elapsed().as_secs_f64();
    let vocab = m.config().vocab_size;
    let blk = m.config().dspark_block_size;
    let ids: Vec<usize> = (0..o.prompt_tokens.max(1))
        .map(|i| (i * 7919 + 13) % vocab)
        .collect();
    obs::collect(true);
    let mut usage = Usage::default();
    let mut pick = SpecSampler::new(0.0, 1.0, 0);
    let steps = o.steps;
    forward::generate(
        &mut m,
        &ids,
        steps * (blk + 2) + 1,
        &mut pick,
        o.spec,
        &[],
        &mut usage,
        &mut |_| (obs::kept(&["decode", "verify"]) >= steps).then_some(Finish::Length),
    )?;
    let recs = obs::take();
    obs::collect(false);
    crate::telemetry::ds_flush_orphans();
    Ok(profile_table(&recs, o, load_s, &usage))
}

fn profile_table(recs: &[super::obs::StepRec], o: &ProfileOpts, load_s: f64, u: &Usage) -> String {
    use super::obs;
    use std::fmt::Write as _;
    let mut s = String::new();
    let r0: Vec<&obs::StepRec> = recs.iter().filter(|r| r.rank == 0).collect();
    let _ = writeln!(
        s,
        "deepseek-profile: load {load_s:.1} s, prompt {} tokens, {} forwards, {} tokens out, drafts {}/{} accepted{}",
        u.prompt_tokens,
        r0.len(),
        u.completion_tokens,
        u.accepted_tokens,
        u.draft_tokens,
        if o.direct { ", direct path (DS_OBS_DIRECT)" } else { "" }
    );
    let _ = writeln!(
        s,
        "\n{:>4} {:<8} {:>3} {:>6} {:<14} {:>10} {:>10} {:>10} {:>8} {:>5} {:>10} {:>10} {:>6} {:>6}",
        "step", "kind", "t", "sp", "path", "wall_ms", "gpu_ms", "host_ms", "launches", "syncs",
        "d2h", "h2d", "allocs", "expup"
    );
    for r in &r0 {
        let c = &r.counts;
        let path = format!("{}{}", r.path, if r.captured { "+capture" } else { "" });
        let _ = writeln!(
            s,
            "{:>4} {:<8} {:>3} {:>6} {:<14} {:>10.3} {:>10.3} {:>10.3} {:>8} {:>5} {:>10} {:>10} {:>6} {:>6}",
            r.step, r.kind, r.t, r.sp, path, r.wall_ms, r.gpu_ms, r.host_ms, c.launches,
            c.syncs, c.d2h_bytes, c.h2d_bytes, c.allocs, c.expert_uploads
        );
    }
    if o.graph_debug {
        let _ = writeln!(s, "\ngraph transitions:");
        let mut last: std::collections::HashMap<(&str, usize), (&str, bool)> = Default::default();
        for r in &r0 {
            let now = (r.path, r.captured);
            let prev = last.insert((r.kind, r.t), now);
            if prev != Some(now) {
                let _ = writeln!(
                    s,
                    "  step {} {} t={}: {} -> {}{} (captures {}, replays {})",
                    r.step,
                    r.kind,
                    r.t,
                    prev.map_or("start".to_string(), |p| p.0.to_string()),
                    r.path,
                    if r.captured { " (captured)" } else { "" },
                    r.counts.captures,
                    r.counts.replays
                );
            }
        }
    }
    // per kind: means over its steps after the first (the first uploads weights and sizes
    // the arena)
    for kind in ["prefill", "draft", "decode", "verify"] {
        let all: Vec<&&obs::StepRec> = r0.iter().filter(|r| r.kind == kind).collect();
        if all.is_empty() {
            continue;
        }
        let warm: Vec<&&obs::StepRec> = if all.len() > 1 {
            all[1..].to_vec()
        } else {
            all.clone()
        };
        let n = warm.len() as f64;
        let mean = |f: &dyn Fn(&obs::StepRec) -> f64| warm.iter().map(|r| f(r)).sum::<f64>() / n;
        let gpu = mean(&|r| r.gpu_ms);
        let _ = writeln!(
            s,
            "\n{kind}: {} steps (means over {}): wall {:.3} ms, gpu {:.3} ms, host {:.3} ms, launches {:.0}, syncs {:.1}, reduce_wait {:.3} ms",
            all.len(),
            warm.len(),
            mean(&|r| r.wall_ms),
            gpu,
            mean(&|r| r.host_ms),
            mean(&|r| r.counts.launches as f64),
            mean(&|r| r.counts.syncs as f64),
            mean(&|r| r.reduce_wait_ms),
        );
        let inner: Vec<&&&obs::StepRec> = warm.iter().filter(|r| !r.spans.is_empty()).collect();
        if inner.is_empty() {
            let _ = writeln!(
                s,
                "  (no interior spans: graph steps are opaque; rerun with --direct)"
            );
            continue;
        }
        let ni = inner.len() as f64;
        let mut ops: Vec<(String, f64, f64)> = Vec::new();
        for r in &inner {
            for (name, ms, cnt) in obs::breakdown(r) {
                match ops.iter_mut().find(|(n, _, _)| *n == name) {
                    Some(e) => {
                        e.1 += ms;
                        e.2 += cnt as f64;
                    }
                    None => ops.push((name, ms, cnt as f64)),
                }
            }
        }
        let gi = inner.iter().map(|r| r.gpu_ms).sum::<f64>() / ni;
        let _ = writeln!(
            s,
            "  {:<28} {:>11} {:>7} {:>7} {:>11}",
            "op", "ms/step", "%step", "n/step", "ms/each"
        );
        for (name, ms, cnt) in ops {
            let per = ms / ni;
            let _ = writeln!(
                s,
                "  {:<28} {:>11.4} {:>6.1}% {:>7.1} {:>11.4}",
                name,
                per,
                100.0 * per / gi.max(1e-9),
                cnt / ni,
                ms / cnt.max(1.0)
            );
        }
        let cov = inner.iter().map(|r| obs::covered_ms(r)).sum::<f64>() / ni;
        let _ = writeln!(
            s,
            "  covered by embed+layers+head: {:.1}% of {:.3} ms gpu",
            100.0 * cov / gi.max(1e-9),
            gi
        );
    }
    let others: Vec<&obs::StepRec> = recs.iter().filter(|r| r.rank != 0).collect();
    if !others.is_empty() {
        // per rank and step kind, so a slow rank shows (rank 0 also reads back and samples);
        // the span sums are the top-level blocks of the layers, where the rank recorded any
        let _ = writeln!(
            s,
            "\nother ranks (means over each kind's steps after the first): gpu | reduce_wait | launches | attn hc moe (ms)"
        );
        let mut ranks: Vec<usize> = others.iter().map(|r| r.rank).collect();
        ranks.sort();
        ranks.dedup();
        for rk in ranks {
            for kind in ["prefill", "draft", "decode", "verify"] {
                let rs: Vec<&&obs::StepRec> = others
                    .iter()
                    .filter(|r| r.rank == rk && r.kind == kind)
                    .collect();
                let rs: Vec<&&obs::StepRec> = if rs.len() > 1 { rs[1..].to_vec() } else { rs };
                if rs.is_empty() {
                    continue;
                }
                let n = rs.len() as f64;
                let mean =
                    |f: &dyn Fn(&obs::StepRec) -> f64| rs.iter().map(|r| f(r)).sum::<f64>() / n;
                let span = |name: &str| {
                    mean(&|r| {
                        r.spans
                            .iter()
                            .filter(|p| p.name == name)
                            .map(|p| p.gpu_ms)
                            .sum::<f64>()
                    })
                };
                let _ = writeln!(
                    s,
                    "  rank {rk} {kind:7} {:3} steps: {:8.3} | {:7.3} | {:6.0} | {:7.3} {:7.3} {:7.3}",
                    rs.len(),
                    mean(&|r| r.gpu_ms),
                    mean(&|r| r.reduce_wait_ms),
                    mean(&|r| r.counts.launches as f64),
                    span("attn"),
                    span("hc"),
                    span("moe"),
                );
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slot batcher against the single-request path: sequences that arrive at different
    /// steps, with different prompts and limits, batched by [`slots_step`] (prefill on the
    /// slot, then shared [`GpuModel::forward_rows`] steps), produce exactly the tokens
    /// `forward::generate` gives each alone (greedy, no drafts), and end the same way.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_tiny_slots_step_matches_generate() {
        use super::super::forward::tests::tiny;
        if tang_moe::gpu::Gpu::for_test().is_none()
            || !Path::new(&tiny()).join("ref/meta.json").exists()
        {
            eprintln!("skip: no GPU or no {}", tiny());
            return;
        }
        let dir = Path::new(&tiny()).join("hf");
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).unwrap(),
        )
        .unwrap();
        let ids: Vec<usize> = meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        // prompts span several chunks, fed one chunk per batcher step (the lone runs chunk
        // the same way through feed_chunked)
        std::env::set_var("DS_PREFILL_CHUNK", "4");
        std::env::set_var("DS_SERVE_PREFILL_CHUNKS", "1");
        let mut a = GpuModel::open(&dir).unwrap();
        a.hc_exact = true;
        let v = a.cpu.cfg.vocab_size;
        // (prompt, max_new, arrival step); 11 sequences: past the 8-row chunk
        let reqs: Vec<(Vec<usize>, usize, usize)> = (0..11)
            .map(|k| {
                let mut p = ids.clone();
                if k > 0 {
                    let j = p.len() - 1 - (k % 3);
                    p[j] = (p[j] + 7919 * k) % v;
                    p.truncate(p.len() - k % 2);
                }
                (p, 6 + k % 5, k / 2)
            })
            .collect();
        // a token the lone runs produce mid-way, as EOS for every run: some end with Stop
        let mut want: Vec<(Vec<usize>, Finish)> = Vec::new();
        let mut eos = Vec::new();
        for pass in 0..2 {
            want.clear();
            for (p, max_new, _) in &reqs {
                a.reset().unwrap();
                let mut got = Vec::new();
                let mut pick = SpecSampler::new(0.0, 1.0, 1);
                let mut u = Usage::default();
                let f = forward::generate(
                    &mut a,
                    p,
                    *max_new,
                    &mut pick,
                    false,
                    &eos,
                    &mut u,
                    &mut |t| {
                        got.push(t);
                        None
                    },
                )
                .unwrap();
                want.push((got, f));
            }
            if pass == 0 {
                eos = vec![want[1].0[3]];
            }
        }
        assert!(
            want.iter().any(|w| w.1 == Finish::Stop),
            "no run hit the EOS"
        );
        assert!(
            want.iter().any(|w| w.1 == Finish::Length),
            "no run hit its limit"
        );
        drop(a);
        let mut m = GpuModel::open(&dir).unwrap();
        m.hc_exact = true;
        let ns = 6; // fewer slots than requests: finished ones free theirs
        m.add_slots(ns, m.cpu.cfg.max_seq_len.min(512)).unwrap();
        let mut free: Vec<usize> = (0..ns).collect();
        let mut queue: Vec<usize> = (0..reqs.len()).collect();
        let mut running: Vec<(usize, SlotSeq)> = Vec::new();
        let mut got: Vec<Vec<usize>> = vec![Vec::new(); reqs.len()];
        let mut fin: Vec<Option<Finish>> = vec![None; reqs.len()];
        let mut step = 0;
        while !(queue.is_empty() && running.is_empty()) {
            queue.retain(|&r| {
                if reqs[r].2 <= step && !free.is_empty() {
                    let sl = free.remove(0);
                    running.push((r, SlotSeq::new(sl, reqs[r].0.clone(), reqs[r].1)));
                    false
                } else {
                    true
                }
            });
            let mut seqs: Vec<&mut SlotSeq> = running.iter_mut().map(|x| &mut x.1).collect();
            let out = slots_step(&mut m, &mut seqs, &eos).unwrap();
            for (j, (r, _)) in running.iter().enumerate() {
                got[*r].extend(&out[j]);
            }
            running.retain(|(r, q)| match q.done {
                Some(f) => {
                    fin[*r] = Some(f);
                    free.push(q.slot);
                    false
                }
                None => true,
            });
            step += 1;
            assert!(step < 200, "batcher did not finish");
        }
        std::env::remove_var("DS_PREFILL_CHUNK");
        std::env::remove_var("DS_SERVE_PREFILL_CHUNKS");
        for r in 0..reqs.len() {
            eprintln!(
                "req {r}: batched {:?} {:?} lone {:?} {:?}",
                got[r], fin[r], want[r].0, want[r].1
            );
            assert_eq!(got[r], want[r].0, "req {r} tokens");
            assert_eq!(fin[r], Some(want[r].1), "req {r} finish");
        }
    }

    #[test]
    fn dsml_tool_calls_parse_across_chunks() {
        let text = format!(
            "Let me look.\n\n<{DSML} calls>\n<{DSML} invoke name=\"read\">\n<{DSML} parameter name=\"path\" string=\"true\">src/main.rs</{DSML} parameter>\n<{DSML} parameter name=\"limit\" string=\"false\">40</{DSML} parameter>\n</{DSML} invoke>\n<{DSML} invoke name=\"ls\">\n</{DSML} invoke>\n</{DSML} calls>"
        );
        // feed it a few characters at a time, as a token stream would
        let mut d = Dsml::default();
        let mut got = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for c in chars.chunks(3) {
            got.extend(d.push(&c.iter().collect::<String>()));
        }
        got.extend(d.finish());
        let text_out: String = got
            .iter()
            .filter_map(|p| {
                if let Piece::Text(s) = p {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text_out, "Let me look.");
        let calls: Vec<_> = got
            .iter()
            .filter(|p| matches!(p, Piece::ToolCall { .. }))
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0],
            &Piece::ToolCall {
                name: "read".into(),
                arguments: serde_json::json!({"path": "src/main.rs", "limit": 40})
            }
        );
        assert_eq!(
            calls[1],
            &Piece::ToolCall {
                name: "ls".into(),
                arguments: serde_json::json!({})
            }
        );
    }

    #[test]
    fn plain_text_passes_through() {
        let mut d = Dsml::default();
        let mut got = d.push("a < b and <｜");
        got.extend(d.push("x"));
        got.extend(d.finish());
        let s: String = got
            .iter()
            .filter_map(|p| {
                if let Piece::Text(s) = p {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(s, "a < b and <｜x");
    }

    #[test]
    fn sampling_is_greedy_at_zero_and_seeded_otherwise() {
        let l = vec![0.1, 2.0, 1.9, -1.0];
        assert_eq!(SpecSampler::new(0.0, 1.0, 1).sample(&l), 1);
        let (mut a, mut b) = (
            SpecSampler::new(1.0, 0.95, 7),
            SpecSampler::new(1.0, 0.95, 7),
        );
        let xs: Vec<usize> = (0..20).map(|_| a.sample(&l)).collect();
        let ys: Vec<usize> = (0..20).map(|_| b.sample(&l)).collect();
        assert_eq!(xs, ys);
        assert!(xs.iter().all(|&i| i < 3), "top-p drops the tail");
    }
}
