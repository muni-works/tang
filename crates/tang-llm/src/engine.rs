//! A loaded model plus tokenizer and chat template: turns chat requests into streamed pieces,
//! reusing the KV cache for whatever prefix the new prompt shares with the previous one.

use crate::chat::{Chunk, Detok, Parser, Piece, Tags, Template};
use crate::draft::{Calibration, CostModel, DraftConfig, Drafter, Global};
use crate::model::{Cache, Dtype, Model};
use crate::sample::{Sampler, Sampling};
use crate::special::Shield;
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tang_compute::ComputeDevice;
use tokenizers::Tokenizer;

/// Prefill in chunks to bound scratch memory.
pub const PREFILL_CHUNK: usize = 512;

/// Fewest tokens a prefill chunk sized to a time budget takes: past the decode rows' share of
/// a step's matmuls (`SHORT_ROWS` in model.rs), so a chunk gets its own.
const MIN_CHUNK: usize = 9;

/// A step's default time budget, while anything decodes: 20 tokens a second for each stream.
pub const STEP_BUDGET: f64 = 0.050;

/// Draft only while at most this many streams decode in a step. Above it verify rows cost
/// more than the forwards they save. Measured with zest-harness on an H100 (Qwen3-8B-4bit,
/// total tok/s, drafting at most at 2 / 4 / always): 4 streams 408 / 304 / 296, 8 streams
/// 542 / 500 / 337, 16 streams 600 / 595 / 314, while at 1-2 streams drafting still pays
/// (1 stream: 174 off vs 320-350 on). Overridden by `TANG_SPEC_MAX_STREAMS`; drafting never
/// changes outputs, only which forwards run.
pub const SPEC_MAX_STREAMS: usize = 2;

/// Fewest leading positions of another conversation's block worth copying (fewer are just
/// prefilled: a chat template's first few tokens are the same everywhere).
const MIN_PARTIAL: usize = 32;

/// Where an image goes in a flattened message; expanded to the model's image tokens. Not an
/// added token itself, so it survives [`Shield::protect`] (a user can type it, but only as many
/// times as there are images, and the images are theirs).
pub const IMAGE_MARKER: &str = "\u{F8FF}tang-image\u{F8FF}";

pub struct Request {
    pub messages: Value,
    /// Encoded images (PNG, JPEG, ...), in the order their markers appear.
    pub images: Vec<Vec<u8>>,
    pub tools: Option<Value>,
    /// JSON schema enforced during sampling; reasoning and drafting are disabled.
    pub response_schema: Option<Value>,
    pub think: Option<bool>,
    /// Most tokens to spend inside `<think>` before the reasoning is wrapped up (see
    /// [`ThinkBudget`]); `None` is unlimited.
    pub thinking_budget: Option<usize>,
    pub sampling: Sampling,
    /// The request named a temperature (`sampling.temperature` is otherwise the default).
    pub temperature_set: bool,
    /// Which of `sampling.top_k` / `top_p` the request named, and its presence penalty.
    pub top_k_set: bool,
    pub top_p_set: bool,
    pub presence_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stop: Vec<String>,
    /// The conversation this continues (`prompt_cache_key`), so it runs in that
    /// conversation's KV slot.
    pub cache_key: Option<String>,
    /// Who sent it (OpenAI's `user` or `safety_identifier`; the gateway sets it to the account):
    /// prefix states past the shared part of a prompt are reused only within a
    /// tenant unless `TANG_SERVE_SHARE=all` (see `flash::serve::Share`).
    pub tenant: Option<String>,
    /// Only prefill (warm the cache for a request that will follow); nothing is generated.
    pub prefill_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Finish {
    Stop,
    Length,
    ToolCalls,
    Cancelled,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// Of the completion, tokens inside `<think>` (including any wrap-up the budget injected).
    pub reasoning_tokens: usize,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    /// Speculative decoding: draft tokens verified, and how many of them were kept.
    pub draft_tokens: usize,
    pub accepted_tokens: usize,
    /// Prefill chunks this run fed, decode forwards (verify windows) it took, and summed over
    /// those forwards how many other streams shared each one.
    pub prefill_chunks: usize,
    pub decode_steps: usize,
    pub shared_streams: usize,
}

/// What the last [`Engine::step`] ran: its forward's decode rows and streams, prefill tokens
/// and chunks, and seconds (all zero when it ran no forward).
#[derive(Debug, Clone, Copy, Default)]
pub struct StepStats {
    pub decode_rows: usize,
    pub decode_streams: usize,
    pub prefill_tokens: usize,
    pub prefill_chunks: usize,
    pub secs: f64,
}

/// How a run went: finished, or stopped part way through its prefill because its
/// [`Control`] asked it to yield. A yielded run's prefilled blocks are sealed and its cache
/// kept, so running the same request again picks up where it stopped (as any prompt that
/// shares a cached prefix does).
#[derive(Debug, Clone)]
pub enum Outcome {
    Done(Finish, Usage),
    Yielded {
        prompt_tokens: usize,
        /// Positions reused from caches before this run's prefill.
        cached_tokens: usize,
        /// Positions this run prefilled before yielding.
        prefilled: usize,
        /// Seconds those took.
        secs: f64,
    },
}

/// Where a running request is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Progress {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    /// Positions prefilled so far (past the cached ones).
    pub prefilled: usize,
    /// Tokens generated so far.
    pub generated: usize,
}

/// What the caller of [`Engine::complete_with`] gets to say while a request runs.
pub trait Control {
    /// Asked between prefill chunks (never before the first): false stops the prefill there,
    /// and the run returns [`Outcome::Yielded`].
    fn keep_prefilling(&mut self) -> bool {
        true
    }

    /// After each prefill chunk and each generated token.
    fn progress(&mut self, _p: Progress) {}

    /// Asked between prefill chunks and before each decode step: true stops the run there
    /// with [`Finish::Cancelled`] (the client went away). What's in the cache stays usable.
    fn cancelled(&mut self) -> bool {
        false
    }
}

/// Never yields; ignores progress.
impl Control for () {}

/// The chat template and tokenizer, shareable across threads: turns a request into its
/// prompt's token ids, as the engine will see them (to count a waiting request's prompt).
#[derive(Clone)]
pub struct Prompter {
    tok: Arc<Tokenizer>,
    template: Arc<Template>,
    shield: Arc<Shield>,
    /// How images become placeholder tokens, if the model sees images.
    image_tokens: Option<ImageSlots>,
}

/// How an image is laid out in the prompt.
#[derive(Clone, Copy, Debug)]
pub enum ImageSlots {
    /// Gemma 3: a fixed number of soft tokens.
    Gemma(usize),
    /// Qwen3-VL: one `<|image_pad|>` per merged patch, by the image's size.
    Qwen(crate::qwen3vl::Sizing),
}

impl ImageSlots {
    /// The prompt text an image (encoded bytes) stands for, as the reference processor
    /// writes it.
    pub fn block(&self, bytes: &[u8]) -> Result<String> {
        Ok(match self {
            Self::Gemma(per) => format!(
                "\n\n<start_of_image>{}<end_of_image>\n\n",
                "<image_soft_token>".repeat(*per)
            ),
            Self::Qwen(sizing) => {
                let (h, w) = crate::qwen3vl::image_size(bytes)?;
                format!(
                    "<|vision_start|>{}<|vision_end|>",
                    "<|image_pad|>".repeat(sizing.tokens(h, w)?)
                )
            }
        })
    }
}

impl Prompter {
    /// The rendered prompt and its token ids. Request text (messages, tool results, tool-call
    /// arguments, tool definitions) is only ever plain tokens; the template's own special
    /// tokens are the only ones (see [`Shield`]).
    pub fn prompt(&self, req: &Request) -> Result<(String, Vec<u32>)> {
        let tools = req.tools.as_ref().map(|t| self.shield.protect(t));
        let mut prompt = self.template.render(
            &self.shield.protect(&req.messages),
            tools.as_ref(),
            if req.response_schema.is_some() {
                Some(false)
            } else {
                req.think
            },
        )?;
        if !req.images.is_empty() {
            let slots = self
                .image_tokens
                .ok_or_else(|| anyhow!("this model can't see images"))?;
            anyhow::ensure!(
                prompt.matches(IMAGE_MARKER).count() == req.images.len(),
                "the chat template dropped some images"
            );
            // As the processor does: each marker becomes a block of image tokens.
            let mut out = String::with_capacity(prompt.len());
            let mut rest = prompt.as_str();
            for img in &req.images {
                let (before, after) = rest.split_once(IMAGE_MARKER).expect("counted above");
                out.push_str(before);
                out.push_str(&slots.block(img)?);
                rest = after;
            }
            out.push_str(rest);
            prompt = out;
        }
        let ids = self.shield.encode(&self.tok, &prompt)?;
        Ok((self.shield.reveal(&prompt), ids))
    }
}

/// Speculative decoding state that outlives a request.
pub struct Speculation {
    pub cfg: DraftConfig,
    pub global: Global,
    pub calib: Calibration,
    pub cost: CostModel,
}

pub struct Engine<D: ComputeDevice> {
    pub model: Model<D>,
    tok: Arc<Tokenizer>,
    template: Arc<Template>,
    shield: Arc<Shield>,
    /// Tag tokens the output parser splits on.
    tags: Tags,
    cache: Cache<D::Buffer>,
    /// Images in the cache: where each image's tokens start, and a hash of the image, so a
    /// shared prefix is only reused when the images in it are the same.
    cache_images: Vec<(usize, u64)>,
    /// The active slot's conversation, and when it last ran (see [`crate::slots`]).
    cache_key: Option<String>,
    cache_used: u64,
    /// When the active slot was last used, by the wall clock (for the TTL).
    cache_at: std::time::SystemTime,
    /// Other conversations' caches, swapped in when a request names them.
    parked: Vec<Slot<D::Buffer>>,
    /// Most caches to keep, the active one included. Each is allocated on first use.
    slots: usize,
    /// Most device memory all caches together may hold; least recently used parked caches
    /// are dropped to stay under it.
    kv_budget: Option<usize>,
    /// Conversations' caches on disk, by `prompt_cache_key`.
    store: Option<crate::kvstore::Writer>,
    clock: u64,
    eos: Vec<u32>,
    structured_factory: Option<llguidance::ParserFactory>,
    /// `<think>` and `</think>`, when the model has them.
    think_tags: Option<(u32, u32)>,
    /// What a spent thinking budget appends: a wrap-up sentence, `</think>`, a blank line.
    wrap_up: Vec<u32>,
    /// Suffix drafting with verification, when on.
    spec: Option<Speculation>,
    /// A small model drafting when suffix drafting has nothing (needs `spec` on).
    drafter: Option<crate::draft_model::DraftModel<D>>,
    /// The last request's prompt and completion tokens (for offline drafting studies).
    last: (Vec<u32>, Vec<u32>),
    /// Decode forwards so far by width: (count, seconds).
    pub forwards: Vec<(usize, f64)>,
    /// The last [`step`](Self::step)'s shape and time.
    pub last_step: StepStats,
    /// Streams admitted and not yet retired, and the conversations they run in.
    out: usize,
    running: Vec<String>,
    max_batch: usize,
    /// How long steps take, fitted as they run; and how long one may take while streams
    /// decode (prefill chunks are sized to fit), if limited.
    step_cost: crate::step_cost::StepCost,
    step_budget: Option<f64>,
    /// Draft only while at most this many streams decode (see [`SPEC_MAX_STREAMS`]), and how
    /// many decode in the current step.
    spec_max_streams: usize,
    decoding: usize,
    /// Sample from device-cut top candidates where a sampler allows (`TANG_TOP_SAMPLING=0`
    /// turns it off: whole rows come back as before).
    top_sampling: bool,
    /// Finished streams' outputs, for the global draft automaton once no stream drafts.
    global_pending: Vec<Vec<u32>>,
}

/// A request in flight, from [`Engine::admit`] to [`Engine::retire`]: its slot's KV cache
/// (checked out of the engine while it runs) and everything its decoding carries between
/// steps.
pub struct Stream<B> {
    cache: Cache<B>,
    /// Images in `cache`, its conversation, and when its slot was last picked.
    images: Vec<(usize, u64)>,
    key: Option<String>,
    used: u64,
    ids: Vec<u32>,
    progress: Progress,
    /// Positions reused from caches.
    reuse: usize,
    limit: usize,
    stop: Vec<String>,
    prefill_only: bool,
    phase: Phase,
    error: Option<anyhow::Error>,
    started: Instant,
    prefill_s: f64,
    decode_from: Option<Instant>,
    constraint: Option<llguidance::Matcher>,
    sampler: Sampler,
    parser: Parser,
    think: ThinkBudget,
    out: Vec<u32>,
    /// Incremental detokenization, tag tokens kept as tags.
    detok: Detok,
    text: String,
    finish: Finish,
    tool_calls: usize,
    drafter: Option<Drafter>,
    /// Logits rows from the last forward: row r follows the fed tokens and `draft[..r]`.
    rows: crate::model::Logits,
    draft: crate::draft::Draft,
    row: usize,
    /// Draft tokens in the cache that aren't accepted yet.
    pending: usize,
    drafted: usize,
    accepted: usize,
    /// Its solo decode forwards by width: (width, seconds).
    timed: Vec<(usize, f64)>,
    /// Prefill chunks fed, decode forwards, and other streams sharing those forwards.
    chunks: usize,
    verifies: usize,
    shared: usize,
    /// What the next forward feeds: new tokens, then (decoding) the draft.
    fed: Vec<u32>,
    /// Of `fed`, the new tokens (the rest is draft).
    fed_new: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Prefill,
    Decode,
    Done,
    Yielded,
}

/// A stream in a [`Engine::step`], with where its pieces go and who controls it.
pub struct Lane<'a, B> {
    pub stream: &'a mut Stream<B>,
    pub on: &'a mut dyn FnMut(Piece) -> bool,
    pub ctl: &'a mut dyn Control,
}

impl<B> Stream<B> {
    /// Finished, yielded or failed: ready for [`Engine::retire`].
    pub fn done(&self) -> bool {
        matches!(self.phase, Phase::Done | Phase::Yielded)
    }

    /// Still prefilling its prompt.
    pub fn prefilling(&self) -> bool {
        self.phase == Phase::Prefill
    }

    pub fn progress(&self) -> Progress {
        self.progress
    }

    /// The conversation it runs in (`prompt_cache_key`).
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    fn fail(&mut self, e: anyhow::Error) {
        self.error = Some(e);
        self.phase = Phase::Done;
    }

    fn end_prefill(&mut self) {
        self.prefill_s = self.started.elapsed().as_secs_f64();
    }

    /// The prompt is in the cache; `logits` are its last position's.
    fn prefilled(&mut self, logits: crate::model::Logits) {
        self.end_prefill();
        if self.prefill_only {
            self.finish = Finish::Stop;
            self.phase = Phase::Done;
            return;
        }
        self.rows = logits;
        self.row = 0;
        self.phase = Phase::Decode;
        self.decode_from = Some(Instant::now());
    }

    /// Hand `pieces` to `on`; false once it refuses one (the client went away).
    fn deliver(&mut self, pieces: Vec<Piece>, on: &mut dyn FnMut(Piece) -> bool) -> bool {
        for p in pieces {
            if matches!(p, Piece::ToolCall { .. }) {
                self.tool_calls += 1;
            }
            if !on(p) {
                return false;
            }
        }
        true
    }
}

/// A KV cache parked while another conversation runs.
struct Slot<B> {
    cache: Cache<B>,
    images: Vec<(usize, u64)>,
    key: Option<String>,
    used: u64,
    /// Last used, by the wall clock (see [`crate::retention`]).
    at: std::time::SystemTime,
}

/// Qwen3's suggested way to end reasoning early: say so, then close the block.
pub const WRAP_UP: &str =
    "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n";

/// Caps reasoning at a token budget. Fed each generated token, it tracks whether decoding is
/// inside `<think>` and how much it has spent there, and says when the budget runs out so the
/// engine can append the wrap-up (through `</think>`) as if the model had written it.
#[derive(Debug)]
pub struct ThinkBudget {
    tags: Option<(u32, u32)>,
    budget: Option<usize>,
    inside: bool,
    spent: usize,
    /// Tokens inside `<think>` so far, the tags and any wrap-up included.
    pub reasoning: usize,
    /// The budget ran out and the reasoning was closed.
    pub hit: bool,
}

impl ThinkBudget {
    /// `inside`: the prompt already opened `<think>`. Without tags nothing is ever cut.
    pub fn new(tags: Option<(u32, u32)>, budget: Option<usize>, inside: bool) -> Self {
        Self {
            tags,
            budget,
            inside: inside && tags.is_some(),
            spent: 0,
            reasoning: 0,
            hit: false,
        }
    }

    /// Note a generated token. True when the reasoning must be closed now.
    pub fn step(&mut self, tok: u32) -> bool {
        let Some((open, close)) = self.tags else {
            return false;
        };
        if tok == close {
            self.reasoning += self.inside as usize;
            self.inside = false;
            return false;
        }
        if tok == open {
            self.inside = true;
        } else if self.inside {
            self.spent += 1;
        }
        if !self.inside {
            return false;
        }
        self.reasoning += 1;
        self.budget.is_some_and(|b| !self.hit && self.spent >= b)
    }

    /// The engine appended `n` tokens of wrap-up, ending with `</think>`.
    pub fn closed(&mut self, n: usize) {
        self.reasoning += n;
        self.inside = false;
        self.hit = true;
    }
}

impl<D: ComputeDevice> Engine<D> {
    pub fn load(dev: D, dir: &Path, max_ctx: usize, dtype: Dtype) -> Result<Self> {
        let model = Model::load(dev, dir, max_ctx, dtype)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let template = Template::load(dir)?;
        let mut eos = model.cfg.eos();
        // generation_config.json often lists more stop ids (e.g. <|im_end|> and <|endoftext|>).
        let gen: Option<Value> = std::fs::read(dir.join("generation_config.json"))
            .ok()
            .and_then(|g| serde_json::from_slice(&g).ok());
        if let Some(g) = gen {
            match &g["eos_token_id"] {
                Value::Number(n) => eos.extend(n.as_u64().map(|n| n as u32)),
                Value::Array(a) => {
                    eos.extend(a.iter().filter_map(|n| n.as_u64().map(|n| n as u32)))
                }
                _ => {}
            }
        }
        // gpt-oss (harmony) ends a turn with `<|return|>` and stops for a tool at `<|call|>`.
        for t in [
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<|return|>",
            "<|call|>",
        ] {
            eos.extend(tok.token_to_id(t));
        }
        eos.sort_unstable();
        eos.dedup();
        let cache = model.new_cache();
        let think_tags = tok.token_to_id("<think>").zip(tok.token_to_id("</think>"));
        let mut wrap_up = Vec::new();
        if let Some((_, close)) = think_tags {
            let enc = |s: &str| -> Result<Vec<u32>> {
                Ok(tok
                    .encode(s, false)
                    .map_err(|e| anyhow!("tokenize: {e}"))?
                    .get_ids()
                    .to_vec())
            };
            wrap_up = enc(WRAP_UP)?;
            wrap_up.push(close);
            wrap_up.extend(enc("\n\n")?);
        }
        Ok(Self {
            model,
            shield: Arc::new(Shield::new(&tok)),
            tags: Tags::new(|t| tok.token_to_id(t)),
            tok: Arc::new(tok),
            template: Arc::new(template),
            cache,
            cache_images: Vec::new(),
            cache_key: None,
            tenant: None,
            cache_used: 0,
            cache_at: crate::retention::now(),
            parked: Vec::new(),
            slots: 1,
            kv_budget: None,
            store: None,
            clock: 0,
            eos,
            structured_factory: None,
            think_tags,
            wrap_up,
            spec: None,
            drafter: None,
            last: (Vec::new(), Vec::new()),
            forwards: vec![(0, 0.0); 65],
            last_step: StepStats::default(),
            out: 0,
            running: Vec::new(),
            max_batch: usize::MAX,
            step_cost: crate::step_cost::StepCost::default(),
            step_budget: Some(STEP_BUDGET),
            spec_max_streams: std::env::var("TANG_SPEC_MAX_STREAMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SPEC_MAX_STREAMS),
            decoding: 0,
            top_sampling: std::env::var("TANG_TOP_SAMPLING").map_or(true, |v| v != "0"),
            global_pending: Vec::new(),
        })
    }

    /// Turn speculative decoding on (with these settings) or off. Outputs don't change.
    pub fn set_speculation(&mut self, cfg: Option<DraftConfig>) {
        self.spec = cfg.map(|cfg| Speculation {
            global: Global::new(cfg.global_tokens, cfg.store.clone()),
            calib: Calibration::default(),
            cost: CostModel::new(cfg.max_draft + 1),
            cfg,
        });
    }

    /// Draft with a small model when suffix drafting finds nothing (speculation must be on).
    pub fn set_draft_model(&mut self, d: Option<crate::draft_model::DraftModel<D>>) {
        self.drafter = d;
    }

    pub fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }

    pub fn speculation(&self) -> Option<&Speculation> {
        self.spec.as_ref()
    }

    /// Take the speculation state out (decoding plainly until it's put back), keeping what it
    /// has learned.
    pub fn take_speculation(&mut self) -> Option<Speculation> {
        self.spec.take()
    }

    pub fn put_speculation(&mut self, spec: Option<Speculation>) {
        self.spec = spec;
    }

    /// The last request's prompt and completion tokens.
    pub fn last_tokens(&self) -> (&[u32], &[u32]) {
        (&self.last.0, &self.last.1)
    }

    /// Forget every KV cache, shared blocks included (the next request prefills from
    /// scratch).
    pub fn reset(&mut self) {
        self.cache.truncate(0);
        self.cache_images.clear();
        for s in &mut self.parked {
            s.cache.truncate(0);
            s.images.clear();
        }
        crate::blocks::lock(self.model.pool()).forget();
    }

    /// Keep KV caches in f32 instead of bf16 (or back). Drops every cache.
    pub fn set_kv_f32(&mut self, f32: bool) {
        self.model.set_kv_f32(f32);
        self.cache = self.model.new_cache();
        self.cache_images.clear();
        self.cache_key = None;
        self.parked.clear();
        self.apply_budget();
    }

    /// Keep up to `n` conversations' KV caches (at least 1). Without a memory budget, the
    /// caches may hold up to `n` full contexts between them.
    pub fn set_slots(&mut self, n: usize) {
        self.slots = n.max(1);
        self.parked.truncate(self.slots - 1);
        self.apply_budget();
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Run at most `n` requests at once (at least 1; never more than the slots).
    pub fn set_max_batch(&mut self, n: usize) {
        self.max_batch = n.max(1);
    }

    /// Limit a step to `secs` while streams decode (prefill chunks shrink to fit), or don't
    /// (`None`: whole chunks always).
    /// Draft only while at most `n` streams decode in a step (`usize::MAX`: always).
    pub fn set_spec_max_streams(&mut self, n: usize) {
        self.spec_max_streams = n;
    }

    pub fn set_step_budget(&mut self, secs: Option<f64>) {
        self.step_budget = secs;
    }

    /// The fitted step-time model.
    pub fn step_cost(&self) -> &crate::step_cost::StepCost {
        &self.step_cost
    }

    /// Time a few steps of different shapes (decode at 1-8 rows, prefill chunks at a few sizes
    /// and prefix lengths, mixed) on scratch caches, and make the fit from them the prior.
    /// A few seconds for a mid-sized model (most of it the 1k-token prefix).
    pub fn calibrate(&mut self) -> Result<()> {
        use crate::model::Segment;
        use crate::step_cost::Shape;
        let m = &self.model;
        let ids: Vec<u32> = (0..1400).map(|i| (100 + i % 1000) as u32).collect();
        let mut rows: Vec<Cache<D::Buffer>> = (0..8).map(|_| m.new_cache()).collect();
        for c in rows.iter_mut() {
            m.forward(&ids[..64], c, false)?;
        }
        let mut long = m.new_cache();
        let one = [7u32];
        // (decode rows, chunk tokens, prefix): both chunk regimes, two prefix lengths (the
        // prefix terms extrapolate; real steps refine them).
        let plan: &[(usize, usize, usize)] = &[
            (1, 0, 0),
            (2, 0, 0),
            (4, 0, 0),
            (8, 0, 0),
            (0, 16, 0),
            (4, 32, 0),
            (1, 64, 0),
            (8, 192, 0),
            (2, 12, 1024),
            (0, 32, 1024),
            (4, 96, 1024),
            (1, 256, 1024),
            (8, 9, 1024),
        ];
        for warm in [true, false] {
            for &(r, q, c) in plan {
                if warm && c > 0 {
                    continue;
                }
                while long.len < c {
                    let k = long.len;
                    m.forward(&ids[k..(k + 512).min(c)], &mut long, false)?;
                }
                long.truncate(c);
                let mut segs: Vec<_> = rows[..r]
                    .iter_mut()
                    .map(|c| Segment {
                        tokens: &one,
                        cache: c,
                        all: false,
                    })
                    .collect();
                if q > 0 {
                    segs.push(Segment {
                        tokens: &ids[c..c + q],
                        cache: &mut long,
                        all: false,
                    });
                }
                let t = Instant::now();
                m.forward_batch(&mut segs)?;
                let secs = t.elapsed().as_secs_f64();
                drop(segs);
                for c in rows.iter_mut() {
                    c.truncate(64);
                }
                if !warm {
                    let s = Shape {
                        decode_rows: r,
                        ..Shape::default()
                    };
                    let s = if q > 0 { s.chunk(q, c) } else { s };
                    self.step_cost.observe(&s, secs);
                }
            }
        }
        self.step_cost.settle();
        Ok(())
    }

    /// Most requests that run at once: one per slot, up to the cap.
    pub fn max_batch(&self) -> usize {
        self.slots.min(self.max_batch)
    }

    /// Cap the device memory all KV caches may hold together (caches grow with their
    /// conversations, so this, not the slot count, is usually what limits them).
    pub fn set_kv_budget(&mut self, bytes: Option<usize>) {
        self.kv_budget = bytes;
        self.apply_budget();
    }

    /// The pool's block budget: the memory budget, or else the slots' full contexts.
    fn apply_budget(&mut self) {
        let max_ctx = self.model.max_ctx();
        let bytes = self
            .kv_budget
            .unwrap_or_else(|| self.slots * self.model.cache_bytes(max_ctx));
        self.model.set_kv_budget(Some(bytes));
    }

    /// Keep conversations' caches on disk in `dir` (up to `budget` bytes), so one that lost its
    /// slot, or outlived the server, is read back instead of prefilled.
    pub fn set_kv_store(&mut self, dir: std::path::PathBuf, budget: u64) -> Result<()> {
        self.store = Some(crate::kvstore::Writer::new(crate::kvstore::Store::new(
            dir,
            budget,
            self.model.row_elems(),
        )?));
        Ok(())
    }

    /// Wait until saves already made are on disk.
    pub fn flush_kv_store(&self) {
        if let Some(s) = &self.store {
            s.flush();
        }
    }

    /// Save `key`'s cache to disk (after a request; the reply has gone out): its sealed blocks
    /// that aren't there yet, and its tail.
    pub fn save(&mut self, key: Option<&str>) {
        use crate::blocks::{lock, BLOCK};
        let (Some(key), Some(store)) = (key, &self.store) else {
            return;
        };
        // Only the conversation that just ran, and not with images (they aren't in tokens).
        if self.cache_key.as_deref() != Some(key) || !self.cache_images.is_empty() {
            return;
        }
        let full = self.cache.len / BLOCK;
        let tokens = &self.cache.tokens;
        // Read back from the device here; the disk writes happen on the writer's thread.
        let shared = self.model.pool().clone();
        let mut pool = lock(&shared);
        for (i, &b) in self.cache.blocks()[..full].iter().enumerate() {
            let Some(h) = pool.hash(b).filter(|_| !pool.on_disk(b)) else {
                continue;
            };
            let rows = pool.read_block(&self.model.dev, b, 0, BLOCK);
            store.write_block(h, tokens[i * BLOCK..(i + 1) * BLOCK].to_vec(), rows);
            pool.set_on_disk(b);
        }
        drop(pool);
        let start = full * BLOCK;
        if store
            .store()
            .tail(key)
            .is_some_and(|(t, s)| s == start && t == *tokens)
        {
            return;
        }
        let rows = self.model.read_rows(&self.cache, start, tokens.len());
        store.write_tail(key, tokens.clone(), start, rows);
    }

    /// Read back from disk what it has of this prompt beyond what the cache holds: whole blocks
    /// by hash (whichever conversation wrote them), then `key`'s tail.
    fn restore(&mut self, key: Option<&str>, ids: &[u32]) {
        use crate::blocks::{chain_all, lock, BLOCK, SEED};
        let Some(store) = &self.store else {
            return;
        };
        let store = store.store();
        // At least one prompt token is still fed, for the logits.
        let limit = ids.len() - 1;
        let have = shared_len(&self.cache.tokens, ids);
        let t = Instant::now();
        let hashes = chain_all(&ids[..limit / BLOCK * BLOCK]);
        let mut found = Vec::new();
        for (j, &h) in hashes.iter().enumerate().skip(have / BLOCK) {
            match store.read_block(h, &ids[j * BLOCK..(j + 1) * BLOCK]) {
                Ok(Some(rows)) => found.push(rows),
                Ok(None) => break,
                Err(e) => {
                    eprintln!("tang-llm: reading a KV block: {e:#}");
                    break;
                }
            }
        }
        let from = have / BLOCK * BLOCK;
        if !found.is_empty() {
            // Whole blocks from `from` (a partial one held is replaced by the full one).
            self.cache.truncate(from);
            let shared = self.model.pool().clone();
            let mut pool = lock(&shared);
            for (j, rows) in (from / BLOCK..).zip(found) {
                let toks = &ids[j * BLOCK..(j + 1) * BLOCK];
                let parent = if j == 0 { SEED } else { hashes[j - 1] };
                let b = pool.alloc(&self.model.dev);
                pool.write_block(&self.model.dev, b, 0, &rows);
                let b = pool.seal(b, hashes[j], parent, toks);
                pool.set_on_disk(b);
                drop(pool);
                self.cache.push_block(b, toks);
                pool = lock(&shared);
            }
        }
        // Then the conversation's tail, if it continues past what's held: it starts on a block
        // boundary the cache has reached.
        if let Some((toks, start)) = key.and_then(|k| store.tail(k)) {
            let end = if start <= self.cache.len && toks.get(..start) == ids.get(..start) {
                start + shared_len(&toks[start..], &ids[start..limit.max(start)])
            } else {
                0
            };
            if end > self.cache.len {
                match store.read_tail(key.unwrap(), end - start) {
                    Ok(rows) => {
                        self.cache.truncate(start);
                        self.model
                            .write_rows(&mut self.cache, &ids[start..end], &rows);
                    }
                    Err(e) => eprintln!("tang-llm: reading a KV tail: {e:#}"),
                }
            }
        }
        if self.cache.len > have {
            let read = self.cache.len - have;
            self.cache_images.retain(|&(s, _)| s < from);
            eprintln!(
                "tang-llm: read {read} positions from disk in {:.0} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// Device memory the KV block pool holds now.
    pub fn kv_bytes(&self) -> usize {
        let pool = crate::blocks::lock(self.model.pool());
        pool.stats().capacity * pool.block_bytes()
    }

    /// Drop conversations (parked, or the idle active one) unused for the TTL.
    fn expire_slots(&mut self) -> bool {
        let ttl = crate::retention::ttl();
        let stale = |at| crate::retention::expired(at, ttl);
        let before = self.parked.len();
        self.parked.retain(|s| !stale(s.at));
        let mut gone = self.parked.len() != before;
        if stale(self.cache_at) && (self.cache.len > 0 || self.cache_key.is_some()) {
            self.cache.truncate(0);
            self.cache_images.clear();
            self.cache_key = None;
            self.last = Default::default();
            gone = true;
        }
        gone
    }

    /// Drop everything derived from prompts that has gone unused for the TTL (see
    /// [`crate::retention`]): conversations, cached blocks, files in the disk tier, completions
    /// in the draft store, the draft model's cache. Running requests keep theirs.
    pub fn expire(&mut self) {
        let ttl = crate::retention::ttl();
        if ttl.is_none() {
            return;
        }
        let idle = self.out == 0 && crate::retention::expired(self.cache_at, ttl);
        self.expire_slots();
        crate::blocks::lock(self.model.pool()).expire();
        if let Some(s) = &self.store {
            s.expire();
        }
        if let Some(s) = &mut self.spec {
            s.global.expire();
        }
        if idle {
            self.global_pending.clear();
            if let Some(d) = &mut self.drafter {
                d.forget();
            }
        }
    }

    /// What the KV block pool holds, in blocks.
    pub fn kv_stats(&self) -> crate::blocks::Stats {
        crate::blocks::lock(self.model.pool()).stats()
    }

    /// Drop least recently used parked conversations until the active one can grow to `rows`
    /// positions within the budget (their sealed blocks stay cached while there's room).
    fn make_room(&mut self, rows: usize) {
        use crate::blocks::BLOCK;
        // Blocks past what it holds now, plus a copy of its last one if that's shared.
        let need = (rows.min(self.model.max_ctx()).div_ceil(BLOCK) + 1)
            .saturating_sub(self.cache.len / BLOCK);
        while crate::blocks::lock(self.model.pool()).available() < need {
            let Some(lru) = (0..self.parked.len()).min_by_key(|&i| self.parked[i].used) else {
                return;
            };
            self.parked.swap_remove(lru);
        }
    }

    /// Take the longest prefix of `ids` (up to `limit` positions) that sealed blocks in the
    /// pool hold, when that beats what the active cache already shares with `ids`: its full
    /// blocks are attached as they are, and a block sharing only some leading positions is
    /// copied that far. Returns the positions attached (0 if the cache was kept).
    fn attach(&mut self, ids: &[u32], limit: usize) -> usize {
        use crate::blocks::{chain, lock, BLOCK, SEED};
        let own = shared_len(&self.cache.tokens, ids);
        let shared = self.model.pool().clone();
        let mut pool = lock(&shared);
        let (mut found, mut parent) = (Vec::new(), SEED);
        while (found.len() + 1) * BLOCK <= limit {
            let t = &ids[found.len() * BLOCK..(found.len() + 1) * BLOCK];
            let h = chain(parent, t);
            let Some(b) = pool.find(h, t) else { break };
            found.push(b);
            parent = h;
        }
        let at = found.len() * BLOCK;
        // A block after the last match that shares some leading positions.
        let rest = &ids[at..limit.min(at + BLOCK)];
        let partial = pool
            .children(parent)
            .into_iter()
            .map(|(b, t)| (b, shared_len(t, rest)))
            .max_by_key(|&(_, n)| n)
            .filter(|&(_, n)| n >= MIN_PARTIAL);
        let total = at + partial.map_or(0, |p| p.1);
        if total <= own {
            return 0;
        }
        // Held before anything is allocated, so eviction can't take them.
        if let Some((b, _)) = partial {
            pool.retain(b);
        }
        drop(pool);
        self.cache.attach(&found, ids);
        self.cache_images.clear();
        if let Some((b, n)) = partial {
            let mut pool = lock(&shared);
            let fresh = pool.alloc(&self.model.dev);
            pool.copy_rows(&self.model.dev, b, fresh, n);
            pool.release(b);
            drop(pool);
            self.cache.push_block(fresh, &ids[at..at + n]);
        }
        total
    }

    /// Make the slot for this request the active one.
    fn select(&mut self, key: Option<&str>, ids: &[u32]) {
        use crate::slots::{pick, Pick, View};
        fn view<'a, B>(key: &'a Option<String>, c: &'a Cache<B>, used: u64) -> View<'a> {
            View {
                key: key.as_deref(),
                tokens: &c.tokens,
                used,
            }
        }
        // Conversations past the TTL aren't picked up again.
        self.expire_slots();
        let parked: Vec<View> = self
            .parked
            .iter()
            .map(|s| view(&s.key, &s.cache, s.used))
            .collect();
        let active = view(&self.cache_key, &self.cache, self.cache_used);
        let room = self.parked.len() + 1 + self.out < self.slots;
        let i = match pick(active, &parked, key, ids, room) {
            Pick::Active => None,
            Pick::Parked(i) => Some(i),
            Pick::Fresh => {
                self.parked.push(Slot {
                    cache: self.model.new_cache(),
                    images: Vec::new(),
                    key: None,
                    used: 0,
                    at: crate::retention::now(),
                });
                Some(self.parked.len() - 1)
            }
        };
        if let Some(i) = i {
            let s = &mut self.parked[i];
            std::mem::swap(&mut self.cache, &mut s.cache);
            std::mem::swap(&mut self.cache_images, &mut s.images);
            std::mem::swap(&mut self.cache_key, &mut s.key);
            std::mem::swap(&mut self.cache_used, &mut s.used);
            std::mem::swap(&mut self.cache_at, &mut s.at);
        }
        if let Some(k) = key {
            self.cache_key = Some(k.to_string());
        }
        self.clock += 1;
        self.cache_used = self.clock;
        self.cache_at = crate::retention::now();
    }

    /// The template and tokenizer, to count prompts elsewhere.
    pub fn prompter(&self) -> Prompter {
        Prompter {
            tok: self.tok.clone(),
            template: self.template.clone(),
            shield: self.shield.clone(),
            image_tokens: match (&self.model.vision, &self.model.qwen_vision) {
                (Some(v), _) => Some(ImageSlots::Gemma(v.tokens)),
                (None, Some(q)) => Some(ImageSlots::Qwen(q.sizing)),
                _ => None,
            },
        }
    }

    /// The sealed KV blocks in memory, by id (see [`crate::blocks::chain`]).
    pub fn block_hashes(&self) -> Vec<u64> {
        crate::blocks::lock(self.model.pool()).hashes()
    }

    /// Each conversation's `prompt_cache_key` and how many positions its cache holds (the
    /// active one first; keyless caches are left out).
    pub fn cached_keys(&self) -> Vec<(String, usize)> {
        let active = std::iter::once((&self.cache_key, &self.cache));
        active
            .chain(self.parked.iter().map(|s| (&s.key, &s.cache)))
            .filter_map(|(k, c)| Some((k.clone()?, c.len)))
            .collect()
    }

    /// Where KV blocks are kept on disk (`<dir>/blocks/<hash>.kvb`), if they are.
    pub fn kv_store_dir(&self) -> Option<&Path> {
        self.store.as_ref().map(|s| s.store().dir())
    }

    /// Tokens whose keys and values are in the cache.
    pub fn cached_tokens(&self) -> &[u32] {
        &self.cache.tokens
    }

    pub fn context_window(&self) -> usize {
        self.model.max_ctx()
    }

    /// Tokens `text` takes as plain text (as a message's content would).
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.shield.plain(&self.tok, text)?.len())
    }

    /// Generate a reply. `on` gets each piece as it's decoded and returns false to stop.
    pub fn complete(
        &mut self,
        req: &Request,
        on: impl FnMut(Piece) -> bool,
    ) -> Result<(Finish, Usage)> {
        match self.complete_with(req, None, on, &mut ())? {
            Outcome::Done(finish, usage) => Ok((finish, usage)),
            Outcome::Yielded { .. } => unreachable!("() never yields"),
        }
    }

    /// [`complete`](Self::complete), with `ctl` told the request's progress and asked between
    /// prefill chunks whether to go on (a prompt with images is prefilled in one go).
    /// `prompt`: the request's prompt and token ids if already made (by this model's
    /// [`Prompter`]). This is [`admit`](Self::admit), [`step`](Self::step) on a batch of
    /// one until it's done, then [`retire`](Self::retire).
    pub fn complete_with(
        &mut self,
        req: &Request,
        prompt: Option<(String, Vec<u32>)>,
        mut on: impl FnMut(Piece) -> bool,
        ctl: &mut dyn Control,
    ) -> Result<Outcome> {
        let mut st = self.admit(req, prompt)?;
        while !st.done() {
            let mut lanes = [Lane {
                stream: &mut st,
                on: &mut on,
                ctl: &mut *ctl,
            }];
            if let Err(e) = self.step(&mut lanes) {
                self.retire(st).ok();
                return Err(e);
            }
        }
        self.retire(st)
    }

    /// Whether a request in conversation `key` can start now: a slot is free and that
    /// conversation isn't already running (two streams can't share a cache).
    pub fn can_admit(&self, key: Option<&str>) -> bool {
        self.out < self.max_batch() && key.is_none_or(|k| !self.running.iter().any(|r| r == k))
    }

    /// Requests running (admitted and not yet retired).
    pub fn running(&self) -> usize {
        self.out
    }

    /// Whether the KV pool has room for another stream of `rows` positions without evicting
    /// a running one's blocks (parked caches may go).
    pub fn kv_room(&self, rows: usize) -> bool {
        use crate::blocks::{lock, BLOCK};
        let pool = lock(self.model.pool());
        let parked: usize = self.parked.iter().map(|s| s.cache.blocks().len()).sum();
        let active = self.cache.blocks().len();
        pool.available() + parked + active > rows.min(self.model.max_ctx()).div_ceil(BLOCK)
    }

    /// Start a request: render its prompt, pick its slot (see [`crate::slots`]), reuse what
    /// caches and the disk hold of it, and take the slot's cache into the returned stream,
    /// which [`step`](Self::step) then prefills and decodes. Check [`can_admit`](Self::can_admit)
    /// first. A prompt with images is prefilled here, in one go.
    pub fn admit(
        &mut self,
        req: &Request,
        prompt: Option<(String, Vec<u32>)>,
    ) -> Result<Stream<D::Buffer>> {
        anyhow::ensure!(
            self.can_admit(req.cache_key.as_deref()),
            "no free slot for this request"
        );
        let constraint = if let Some(schema) = &req.response_schema {
            anyhow::ensure!(
                req.stop.is_empty(),
                "stop strings cannot be combined with structured output"
            );
            if self.structured_factory.is_none() {
                self.structured_factory = Some(crate::structured::model_factory(
                    &self.tok,
                    self.model.cfg.vocab_size,
                    *self
                        .eos
                        .first()
                        .ok_or_else(|| anyhow!("model has no EOS token"))?,
                )?);
            }
            Some(crate::structured::matcher(
                self.structured_factory.as_ref().unwrap(),
                schema.clone(),
            )?)
        } else {
            None
        };
        let (prompt, ids) = match prompt {
            Some(p) => p,
            None => self.prompter().prompt(req)?,
        };
        // Where each image's tokens start, with a hash of the image.
        let img_tok = self.model.cfg.image_token;
        let mut runs: Vec<(usize, u64)> = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            if !req.images.is_empty() && Some(t) == img_tok && (i == 0 || ids[i - 1] != t) {
                let bytes = &req.images[runs.len().min(req.images.len().saturating_sub(1))];
                runs.push((i, hash(bytes)));
            }
        }
        let ctx = self.model.max_ctx();
        anyhow::ensure!(
            ids.len() < ctx,
            "prompt is {} tokens; the context window is {ctx}",
            ids.len()
        );
        let limit = req.max_tokens.unwrap_or(usize::MAX).min(ctx - ids.len());

        self.select(req.cache_key.as_deref(), &ids);
        // Another conversation's blocks, if they hold more of this prompt (at least one token
        // is still fed, for the logits; nothing from an image on).
        let first_image = runs.first().map_or(ids.len(), |r| r.0);
        let attached = self.attach(&ids, (ids.len() - 1).min(first_image));
        if attached > 0 {
            eprintln!("tang-llm: {attached} positions from shared blocks");
        }
        // What's past the shared prefix won't be reused.
        let keep = shared_len(&self.cache.tokens, &ids);
        self.cache.truncate(keep);
        // Room for the prompt and a typical reply; a longer one grows past the budget a little.
        self.make_room(ids.len() + limit.min(4096));
        if req.images.is_empty() {
            self.restore(req.cache_key.as_deref(), &ids);
        }
        // Reuse the shared prefix (at least one token must be fed to get logits).
        let shared = shared_len(&self.cache.tokens, &ids);
        // Only as far as the images match, and never into the middle of an image's tokens.
        let differ = self
            .cache_images
            .iter()
            .zip(&runs)
            .find(|(a, b)| a != b)
            .map(|(a, b)| a.0.min(b.0))
            .or_else(|| runs.get(self.cache_images.len()).map(|r| r.0))
            .unwrap_or(usize::MAX);
        let mut reuse = shared.min(ids.len() - 1).min(differ);
        let run_len = |s: usize| ids[s..].iter().take_while(|&&t| Some(t) == img_tok).count();
        if let Some(&(start, _)) = runs
            .iter()
            .find(|(s, _)| *s < reuse && reuse < s + run_len(*s))
        {
            reuse = start;
        }
        self.cache.truncate(reuse);

        // The slot's cache goes with the stream; an empty one holds its place.
        let cache = std::mem::replace(&mut self.cache, self.model.new_cache());
        let images = std::mem::take(&mut self.cache_images);
        let key = self.cache_key.take();
        let used = self.cache_used;
        self.parked
            .retain(|s| s.key.is_some() || !s.cache.tokens.is_empty());
        self.out += 1;
        if let Some(k) = &key {
            self.running.push(k.clone());
        }

        let opened = prompt.trim_end().ends_with("<think>");
        let drafter = self
            .spec
            .as_ref()
            .filter(|_| constraint.is_none())
            .map(|s| {
                let cost = s.cost.table(&s.cfg, s.cfg.max_draft + 1);
                Drafter::new(&s.cfg, &s.global, s.calib.clone(), cost, &ids)
            });
        let mut st = Stream {
            cache,
            images,
            key,
            used,
            progress: Progress {
                prompt_tokens: ids.len(),
                cached_tokens: reuse,
                prefilled: 0,
                generated: 0,
            },
            reuse,
            limit,
            stop: req.stop.clone(),
            prefill_only: req.prefill_only,
            phase: Phase::Prefill,
            error: None,
            started: Instant::now(),
            prefill_s: 0.0,
            decode_from: None,
            constraint,
            sampler: Sampler::new(req.sampling.clone()),
            parser: Parser::for_tags(&self.tags, opened),
            think: ThinkBudget::new(self.think_tags, req.thinking_budget, opened),
            out: Vec::new(),
            detok: Detok::default(),
            text: String::new(),
            finish: Finish::Length,
            tool_calls: 0,
            drafter,
            rows: Default::default(),
            draft: crate::draft::Draft::default(),
            row: 0,
            pending: 0,
            drafted: 0,
            accepted: 0,
            timed: Vec::new(),
            chunks: 0,
            verifies: 0,
            shared: 0,
            fed: Vec::new(),
            fed_new: 0,
            ids,
        };
        if !runs.is_empty() {
            let mut feats = Vec::new();
            for (i, &(start, _)) in runs.iter().enumerate() {
                if start >= reuse {
                    feats.push(self.model.encode_image_bytes(&req.images[i])?);
                }
            }
            let logits =
                match self
                    .model
                    .forward_images(&st.ids[reuse..], &feats, &mut st.cache, false)
                {
                    Ok(l) => l,
                    Err(e) => {
                        self.retire(st).ok();
                        return Err(e);
                    }
                };
            st.progress.prefilled = st.ids.len() - reuse;
            st.images = runs;
            st.prefilled(crate::model::Logits::Full(logits));
        }
        Ok(st)
    }

    /// One forward pass for every stream in `lanes`, and what follows it: each decoding
    /// stream samples from its last logits (accepting what it can of its draft) and feeds its
    /// next token and draft; each prefilling stream feeds its next chunk of prompt, within
    /// `budget` tokens across the prefills (decoding rows go first; prefills are served in
    /// lane order). Pieces go to each lane's `on`; its `ctl` is told progress and asked whether
    /// to go on. A stream that finishes, yields or fails is [`done`](Stream::done); hand it to
    /// [`retire`](Self::retire). An error here is the forward's, and every stream in it is
    /// lost (retire them all).
    pub fn step(&mut self, lanes: &mut [Lane<'_, D::Buffer>]) -> Result<()> {
        self.last_step = StepStats::default();
        // Each stream's segment, if it has one this step: (tokens fed, all rows).
        let mut segs: Vec<Option<bool>> = Vec::with_capacity(lanes.len());
        let mut decode_rows = 0;
        self.decoding = lanes
            .iter()
            .filter(|l| l.stream.phase == Phase::Decode)
            .count();
        for lane in lanes.iter_mut() {
            let want = match lane.stream.phase {
                Phase::Decode => match self.advance(lane) {
                    Ok(w) => w,
                    Err(e) => {
                        lane.stream.fail(e);
                        None
                    }
                },
                _ => None,
            };
            if want.is_some() {
                decode_rows += lane.stream.fed.len();
            }
            segs.push(want);
        }
        // Prefill chunks fill what's left of the step's time budget, in lane order, each as
        // many tokens as the fitted step cost says fit (see `crate::step_cost`). With nothing
        // decoding (or no budget) a chunk is a whole PREFILL_CHUNK, as a prompt alone gets.
        let mut shape = crate::step_cost::Shape {
            decode_rows,
            ..Default::default()
        };
        let mut left = match self.step_budget {
            Some(b) if decode_rows > 0 => Some(b - self.step_cost.predict(&shape)),
            _ => None,
        };
        let mut first = true;
        for (lane, seg) in lanes.iter_mut().zip(segs.iter_mut()) {
            let st = &mut *lane.stream;
            if st.phase != Phase::Prefill {
                continue;
            }
            if st.progress.prefilled > 0 {
                if lane.ctl.cancelled() {
                    // Seal what's done (see `retire`); nothing past an image here.
                    st.images.clear();
                    st.end_prefill();
                    st.finish = Finish::Cancelled;
                    st.phase = Phase::Done;
                    continue;
                }
                if !lane.ctl.keep_prefilling() {
                    st.images.clear();
                    st.end_prefill();
                    st.phase = Phase::Yielded;
                    continue;
                }
            }
            let from = st.reuse + st.progress.prefilled;
            let rest = st.ids.len() - from;
            let c = st.cache.len;
            let n = match left {
                None => rest.min(PREFILL_CHUNK),
                Some(secs) => {
                    let fit = self.step_cost.fit_chunk(secs, c, PREFILL_CHUNK.min(rest));
                    // The first prefilling stream always moves, by at least MIN_CHUNK.
                    let n = if first { fit.max(MIN_CHUNK) } else { fit };
                    if n < MIN_CHUNK.min(rest) {
                        continue;
                    }
                    n.min(rest)
                }
            };
            first = false;
            if let Some(secs) = left.as_mut() {
                let one = crate::step_cost::Shape::default().chunk(n, c);
                *secs -= self.step_cost.predict(&one) - self.step_cost.coef()[0];
            }
            shape = shape.chunk(n, c);
            st.fed.clear();
            st.fed.extend_from_slice(&st.ids[from..from + n]);
            *seg = Some(false);
        }
        // Rows a stream's sampler can draw from their top candidates are cut on the device
        // (see `Logits::Top`); a grammar needs the whole row to mask. A lone stream samples
        // its one row on the host as fast (measured on an H100), and either way picks the same
        // token, so only batches cut.
        let batched = segs.iter().filter(|s| s.is_some()).count() > 1;
        let top: Vec<bool> = lanes
            .iter()
            .zip(&segs)
            .filter(|(_, seg)| seg.is_some())
            .map(|(lane, _)| {
                let st = &lane.stream;
                self.top_sampling
                    && batched
                    && st.constraint.is_none()
                    && st.sampler.top_needed(crate::model::TOP_LOGITS).is_some()
            })
            .collect();
        let mut idx = Vec::new();
        let mut parts = Vec::new();
        for (i, (lane, seg)) in lanes.iter_mut().zip(&segs).enumerate() {
            if let Some(all) = *seg {
                let st = &mut *lane.stream;
                idx.push(i);
                parts.push(crate::model::Segment {
                    tokens: &st.fed,
                    cache: &mut st.cache,
                    all,
                });
            }
        }
        if parts.is_empty() {
            return Ok(());
        }
        let solo = parts.len() == 1;
        let tf = Instant::now();
        let logits = self.model.forward_batch_top(&mut parts, &top)?;
        let secs = tf.elapsed().as_secs_f64();
        self.step_cost.observe(&shape, secs);
        drop(parts);
        let others = idx.len() - 1;
        let mut stats = StepStats {
            decode_rows,
            secs,
            ..StepStats::default()
        };
        let vocab = self.model.cfg.vocab_size;
        for (i, rows) in idx.into_iter().zip(logits) {
            let lane = &mut lanes[i];
            let st = &mut *lane.stream;
            match st.phase {
                Phase::Prefill => {
                    st.chunks += 1;
                    stats.prefill_chunks += 1;
                    stats.prefill_tokens += st.fed.len();
                    st.progress.prefilled += st.fed.len();
                    lane.ctl.progress(st.progress);
                    if st.reuse + st.progress.prefilled == st.ids.len() {
                        st.prefilled(rows);
                    }
                }
                Phase::Decode => {
                    let (m, k) = (st.fed_new, st.fed.len() - st.fed_new);
                    st.verifies += 1;
                    st.shared += others;
                    stats.decode_streams += 1;
                    if m == 1 && solo {
                        st.timed.push((1 + k, secs));
                    }
                    st.rows = rows;
                    if k > 0 {
                        st.rows.skip_rows(m - 1, vocab);
                    }
                    st.row = 0;
                    st.pending = k;
                    st.drafted += k;
                }
                _ => {}
            }
        }
        self.last_step = stats;
        crate::telemetry::step(&stats);
        Ok(())
    }

    /// Sample a decoding stream forward from the logits it has until it needs a forward:
    /// returns `Some(all)` with its next segment in `fed`, or `None` once it's done.
    fn advance(&mut self, lane: &mut Lane<'_, D::Buffer>) -> Result<Option<bool>> {
        let st = &mut *lane.stream;
        let vocab = self.model.cfg.vocab_size;
        let max_ctx = self.model.max_ctx();
        let spec = self.spec.as_ref();
        // Every token is still sampled from the target's logits in order, with the same
        // sampler state, so the output is what plain decoding would produce; a draft token
        // only saves a forward when it equals the sampled token.
        'decode: loop {
            if st.out.len() >= st.limit {
                break;
            }
            if lane.ctl.cancelled() {
                st.finish = Finish::Cancelled;
                break;
            }
            let next = match &mut st.rows {
                crate::model::Logits::Full(rows) => {
                    let logits = &mut rows[st.row * vocab..(st.row + 1) * vocab];
                    if let Some(grammar) = st.constraint.as_mut() {
                        crate::structured::mask(grammar, logits, &self.eos)?;
                    }
                    st.sampler.sample(logits)
                }
                crate::model::Logits::Top { vals, idx } => {
                    let k = crate::model::TOP_LOGITS.min(vocab);
                    let r = st.row * k..(st.row + 1) * k;
                    st.sampler.sample_top(&vals[r.clone()], &idx[r])
                }
            };
            let row = st.row;
            let hit = st.draft.tokens.get(row) == Some(&next);
            if row < st.draft.tokens.len() && !st.draft.from_model {
                if let Some(d) = st.drafter.as_mut() {
                    d.calib
                        .observe(st.draft.match_len, st.draft.shares[row], hit);
                }
            }
            if hit {
                st.accepted += 1;
                st.pending -= 1;
            }
            if self.eos.contains(&next) {
                st.finish = Finish::Stop;
                break;
            }
            if let Some(grammar) = st.constraint.as_mut() {
                grammar.consume_token(next)?;
            }
            st.out.push(next);
            st.progress.generated = st.out.len();
            lane.ctl.progress(st.progress);
            // What the next forward must feed (a hit is already in the cache).
            st.fed.clear();
            if !hit {
                st.fed.push(next);
            }
            // Out of thinking budget: write the wrap-up and `</think>` for the model (into the
            // KV cache like its own tokens, and streamed as reasoning), then let it answer.
            let wrap = st.think.step(next) && st.out.len() + self.wrap_up.len() < st.limit;
            if wrap {
                st.out.extend(&self.wrap_up);
                st.fed.extend(&self.wrap_up);
                st.think.closed(self.wrap_up.len());
            }
            if let (Some(d), Some(s)) = (st.drafter.as_mut(), spec) {
                let n = st.out.len() - if wrap { self.wrap_up.len() } else { 0 };
                for &tok in &st.out[n - 1..] {
                    d.push(&s.cfg, &s.global, tok);
                }
            }
            // Tag tokens come out as tags, everything else as literal text (see `Parser`).
            for chunk in st.detok.push(&st.out, &self.tags, |ids| self.decode(ids))? {
                let (new, tag) = match chunk {
                    Chunk::Text(s) => (s, None),
                    Chunk::Tag(t) => (t.to_string(), Some(t)),
                };
                st.text.push_str(&new);
                let stop_at = st
                    .stop
                    .iter()
                    .filter_map(|s| st.text.find(s.as_str()))
                    .min();
                let new = match stop_at {
                    Some(i) => {
                        let keep = new.len().saturating_sub(st.text.len() - i);
                        new[..keep].to_string()
                    }
                    None => new,
                };
                // Constrained output is the grammar's JSON verbatim: tool-call or think
                // markers inside a string value are data, not structure.
                let pieces = match tag {
                    _ if st.constraint.is_some() => vec![Piece::Text(new)],
                    Some(t) if new == t => st.parser.push_tag(t),
                    _ => st.parser.push_text(&new),
                };
                if !st.deliver(pieces, &mut *lane.on) {
                    st.finish = Finish::Cancelled;
                    break 'decode;
                }
                if stop_at.is_some() {
                    st.finish = Finish::Stop;
                    break 'decode;
                }
            }
            if st.fed.is_empty() {
                // The draft was right: the next position's logits are already here.
                st.row += 1;
                continue;
            }
            if st.out.len() >= st.limit {
                break;
            }
            // Drop the rejected rest of the draft from the cache.
            st.cache.truncate(st.cache.len - st.pending);
            st.pending = 0;
            // A new draft, unless the wrap-up was just forced in (don't draft across it) or
            // the batch is too wide for verify rows to pay.
            let wide = self.decoding > self.spec_max_streams;
            st.draft = match (st.drafter.as_ref(), spec) {
                (Some(d), Some(s)) if !wrap && !wide => {
                    let room = (st.limit - st.out.len())
                        .min(max_ctx.saturating_sub(st.cache.len + st.fed.len()));
                    let p = d.propose(&s.cfg, &s.global, room);
                    match self.drafter.as_mut() {
                        Some(m) if p.tokens.is_empty() => crate::draft::Draft {
                            tokens: m.propose(&st.ids, &st.out, room, |k| d.cost(k))?,
                            from_model: true,
                            ..Default::default()
                        },
                        _ => p,
                    }
                }
                _ => crate::draft::Draft::default(),
            };
            st.fed_new = st.fed.len();
            st.fed.extend(&st.draft.tokens);
            return Ok(Some(!st.draft.tokens.is_empty()));
        }
        // Done: leave only accepted tokens in the cache.
        st.cache.truncate(st.cache.len - st.pending);
        st.pending = 0;
        if st.finish != Finish::Cancelled {
            let rest = st.parser.finish();
            st.deliver(rest, &mut *lane.on);
            if st.tool_calls > 0 && st.finish == Finish::Stop {
                st.finish = Finish::ToolCalls;
            }
        }
        st.phase = Phase::Done;
        Ok(None)
    }

    /// Finish with a stream that's done (or abandon one that isn't): seal its blocks so
    /// other conversations can share them, put its cache back as the active slot, fold what
    /// its drafting learned into the engine's, and say how it went.
    pub fn retire(&mut self, mut st: Stream<D::Buffer>) -> Result<Outcome> {
        st.cache.truncate(st.cache.len - st.pending);
        st.pending = 0;
        let upto = st.images.first().map_or(usize::MAX, |&(s, _)| s);
        st.cache.seal(upto);
        // Back as the active slot (the conversation that ran last, as `save` expects).
        self.out -= 1;
        if let Some(k) = &st.key {
            if let Some(i) = self.running.iter().position(|r| r == k) {
                self.running.swap_remove(i);
            }
        }
        let cache = std::mem::replace(&mut st.cache, self.model.new_cache());
        let prev = Slot {
            cache: std::mem::replace(&mut self.cache, cache),
            images: std::mem::replace(&mut self.cache_images, std::mem::take(&mut st.images)),
            key: std::mem::replace(&mut self.cache_key, st.key.take()),
            used: std::mem::replace(&mut self.cache_used, st.used),
            at: std::mem::replace(&mut self.cache_at, crate::retention::now()),
        };
        if prev.key.is_some() || !prev.cache.tokens.is_empty() {
            self.parked.push(prev);
        }
        while 1 + self.parked.len() + self.out > self.slots {
            let Some(lru) = (0..self.parked.len()).min_by_key(|&i| self.parked[i].used) else {
                break;
            };
            self.parked.swap_remove(lru);
        }
        if let Some(e) = st.error.take() {
            return Err(e);
        }
        if st.phase == Phase::Yielded {
            return Ok(Outcome::Yielded {
                prompt_tokens: st.ids.len(),
                cached_tokens: st.reuse,
                prefilled: st.progress.prefilled,
                secs: st.prefill_s,
            });
        }
        if st.phase == Phase::Prefill {
            st.end_prefill();
        }
        for &(w, secs) in &st.timed {
            if let Some(slot) = self.forwards.get_mut(w) {
                slot.0 += 1;
                slot.1 += secs;
            }
        }
        if let (Some(d), Some(spec)) = (st.drafter.take(), self.spec.as_mut()) {
            spec.calib = d.calib;
            for &(w, secs) in &st.timed {
                spec.cost.observe(w, secs);
            }
            // Into the global automaton once nothing is drafting from it.
            self.global_pending.push(st.out.clone());
        }
        if self.out == 0 {
            if let Some(spec) = self.spec.as_mut() {
                for out in self.global_pending.drain(..) {
                    spec.global.push(&out);
                }
            }
        }
        let decode_s = st.decode_from.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let prompt_tokens = st.ids.len();
        let usage = Usage {
            prompt_tokens,
            cached_tokens: st.reuse,
            completion_tokens: st.out.len(),
            reasoning_tokens: st.think.reasoning,
            prefill_tok_s: st.progress.prefilled as f64 / st.prefill_s.max(1e-9),
            decode_tok_s: if st.decode_from.is_some() {
                st.out.len() as f64 / decode_s.max(1e-9)
            } else {
                0.0
            },
            draft_tokens: st.drafted,
            accepted_tokens: st.accepted,
            prefill_chunks: st.chunks,
            decode_steps: st.verifies,
            shared_streams: st.shared,
        };
        self.last = (st.ids, st.out);
        Ok(Outcome::Done(st.finish, usage))
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok
            .decode(ids, false)
            .map_err(|e| anyhow!("detokenize: {e}"))
            .context("decode")
    }
}

/// How many leading tokens `a` and `b` share.
fn shared_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn hash(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::ThinkBudget;

    const OPEN: u32 = 1;
    const CLOSE: u32 = 2;

    /// Feed tokens; the index of the token after which the budget cut in, if it did.
    fn run(b: &mut ThinkBudget, toks: &[u32]) -> Option<usize> {
        for (i, &t) in toks.iter().enumerate() {
            if b.step(t) {
                b.closed(3);
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn cuts_after_budget_tokens_of_reasoning() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(3), false);
        // `<think>`, then three reasoning tokens: the third spends the budget.
        assert_eq!(run(&mut b, &[OPEN, 10, 11, 12, 13]), Some(3));
        assert!(b.hit);
        assert_eq!(b.reasoning, 4 + 3);
        // Answer tokens after the close don't count, and a second block isn't cut.
        assert_eq!(run(&mut b, &[20, 21, OPEN, 30, 31, 32, 33]), None);
    }

    #[test]
    fn reasoning_that_ends_in_time_is_left_alone() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(3), false);
        assert_eq!(run(&mut b, &[OPEN, 10, 11, CLOSE, 20, 21, 22, 23]), None);
        assert!(!b.hit);
        assert_eq!(b.reasoning, 4);
    }

    #[test]
    fn a_prompt_that_opened_think_counts_from_the_first_token() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(2), true);
        assert_eq!(run(&mut b, &[10, 11, 12]), Some(1));
    }

    #[test]
    fn zero_budget_closes_right_after_the_open_tag() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(0), false);
        assert_eq!(run(&mut b, &[5, OPEN, 10]), Some(1));
    }

    #[test]
    fn no_budget_or_no_tags_never_cuts() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), None, false);
        assert_eq!(run(&mut b, &[OPEN, 10, 11, 12, CLOSE]), None);
        assert_eq!(b.reasoning, 5);
        let mut b = ThinkBudget::new(None, Some(0), true);
        assert_eq!(run(&mut b, &[OPEN, 10, 11]), None);
        assert_eq!(b.reasoning, 0);
    }
}
