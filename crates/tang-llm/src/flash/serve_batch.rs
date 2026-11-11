//! Continuous batching for `flash serve --batch N` (qwen3next only, `docs/qwen3next-batching.md`):
//! the scheduler half. The engine half is the Coder's sequence slots (`alloc_slot`,
//! `prefill_slot`, `decode_batch`).
//!
//! - **Slots and prefix reuse.** A finished request's slot is parked with its tokens. A new
//!   request takes the parked slot whose tokens are a prefix of its prompt (an agent's next turn
//!   extends the conversation), so only the new tokens are prefilled; otherwise a fresh slot,
//!   evicting the least recently used parked one when all are taken. The recurrent state can't
//!   rewind, so a slot is only ever extended.
//! - **A step.** Prefilling streams advance by a chunk, then one batched decode window runs
//!   every decoding stream's next token. Which prompts, how many tokens and how GPU time is split
//!   is `crate::batch::prefill` (pure, tested on a simulated engine): shortest-first with aging
//!   and a deadline; prefill held to a share of GPU time (higher while decode rows are below the
//!   fill target), its chunk as long as the banked decode credit buys; knobs `TANG_SERVE_*`.
//! - **Sampling.** Each stream samples with its own temperature / top-k / top-p / seed /
//!   presence penalty (the engine's per-row sampler; greedy rows are the plain argmax). A
//!   sampled stream takes no MTP drafts (verification is greedy). An engine without per-row
//!   sampling refuses a sampled request instead of running it greedy.
//! - **Drafts under batching** (`crate::batch::spec`): every greedy stream's MTP chain, ranked
//!   across the window by cumulative probability (D-Cut); the window verifies the best ones in
//!   the rows the streams leave (batch module up to its width, else `MAX_T`), the budget picked
//!   from a measured step-cost table, none at full load. `TANG_SERVE_SPEC=off|auto|<rows>`.

use super::{FlashServe, Memo};
use crate::batch::prefill::{prefill_plan, Policy, Waiting};
use crate::chat::{Chunk, Detok, Parser, Piece};
use crate::engine::{Finish, Outcome, Request, Usage};
use crate::server::{DynLane, Inflight, StepFailure};
use anyhow::{bail, ensure, Result};
use serde_json::Value;
use std::time::Instant;
use tang_compute::flash::MAX_T;

/// A parked or running sequence slot.
pub(super) struct SlotInfo {
    pub id: usize,
    pub busy: bool,
    pub key: Option<String>,
    pub used: u64,
}

/// One request in the batch.
pub(super) struct Stream {
    slot: usize,
    key: Option<String>,
    ids: Vec<u32>,
    /// Prompt tokens already in the slot (reused or prefilled).
    done_prompt: usize,
    /// Where to save prefix states (`FlashServe::marks`) still ahead, and whether to pin each.
    marks: Vec<(usize, bool)>,
    reused: usize,
    /// The sampled token not yet fed to the slot.
    pending: Option<u32>,
    /// MTP drafts for the next window (the engine's chain for this slot, with the head's
    /// probabilities).
    drafts: Vec<(u32, f32)>,
    /// Draft rows verified and accepted, and verify windows run.
    drafted: usize,
    accepted: usize,
    windows: usize,
    out: Vec<u32>,
    limit: usize,
    stop: Vec<String>,
    parser: Parser,
    detok: Detok,
    text_out: String,
    content: String,
    calls: Vec<(String, Value)>,
    tail: String,
    finish: Option<Finish>,
    t0: Instant,
    first: Option<f64>,
    prefill_s: f64,
    /// Of `prefill_s`, saving prefix states at its marks.
    snap_s: f64,
    decode_from: Option<Instant>,
    prefill_only: bool,
    /// Sampled (temperature > 0): no drafts.
    sampled: bool,
    /// A step failed this stream: its slot's state is suspect, so retiring frees the slot
    /// instead of parking it for reuse.
    broken: bool,
    /// Host-side sampling for a constrained row whose device pick breaks the grammar.
    host: crate::structured::HostSampling,
    /// Structured output's grammar. Such a stream prefills all but the prompt's last token,
    /// which it then decodes like an output step, so every output token is checked with its
    /// row's logits; it never drafts.
    constraint: Option<crate::structured::Constraint>,
}

impl Stream {
    /// Where prefill stops: a constrained stream decodes the prompt's last token.
    fn prompt_end(&self) -> usize {
        self.ids.len() - self.constraint.is_some() as usize
    }
}

impl FlashServe {
    pub(super) fn batch_can_admit(&self, key: Option<&str>) -> bool {
        let busy = self.slots.iter().filter(|s| s.busy).count();
        busy < self.batch
            && !(key.is_some() && self.slots.iter().any(|s| s.busy && s.key.as_deref() == key))
    }

    /// A slot for prompt `ids`: a parked one whose tokens it extends (the longest), else a
    /// fresh one, evicting the least recently used parked slot when none is free.
    fn take_slot(&mut self, ids: &[u32], key: Option<&str>) -> Result<(usize, usize)> {
        self.clock += 1;
        let best = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.busy)
            .filter_map(|(i, s)| {
                let t = self.e.slot_tokens(s.id);
                (t.len() < ids.len() && ids.starts_with(t)).then_some((i, t.len()))
            })
            .max_by_key(|&(_, n)| n);
        let parked = best.map_or(0, |b| b.1);
        // A saved prefix state (the end of a tools block or system prompt, from any session,
        // in RAM or on disk) that reaches further: restore it into a fresh slot.
        let limit = ids.len() - 1;
        let mut snap = (0..self.snaps.len())
            .filter(|&i| {
                self.snaps[i].tokens.len() <= limit && ids.starts_with(&self.snaps[i].tokens)
            })
            .max_by_key(|&i| self.snaps[i].tokens.len());
        if snap.is_none() && parked == 0 {
            if let Some(sn) = self.load_disk(ids, limit).filter(|s| !s.tokens.is_empty()) {
                eprintln!(
                    "tang-llm: read the prefix state at {} from disk",
                    sn.tokens.len()
                );
                self.snaps.push(sn);
                snap = Some(self.snaps.len() - 1);
            }
        }
        let snap_len = snap.map_or(0, |i| self.snaps[i].tokens.len());
        if snap_len <= parked {
            if let Some((i, n)) = best {
                let s = &mut self.slots[i];
                s.busy = true;
                s.key = key.map(String::from);
                s.used = self.clock;
                return Ok((s.id, n));
            }
        }
        // No parked slot extends: say how close the nearest one came, so a client changing early tokens
        // (or a template re-rendering an earlier turn) shows up in the log.
        if let Some((id, common, len)) = self
            .slots
            .iter()
            .filter(|s| !s.busy)
            .map(|s| {
                let t = self.e.slot_tokens(s.id);
                let common = t.iter().zip(ids).take_while(|(a, b)| a == b).count();
                (s.id, common, t.len())
            })
            .max_by_key(|&(_, c, _)| c)
        {
            eprintln!(
                "tang-llm: prefix miss: prompt {} tokens; nearest parked slot {id} has {len}, \
                 diverging at token {common}",
                ids.len()
            );
        }
        let id = match self.e.alloc_slot() {
            Ok(id) => id,
            Err(_) => {
                // All allocated: drop the least recently used parked slot.
                let (i, _) = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| !s.busy)
                    .min_by_key(|(_, s)| s.used)
                    .ok_or_else(|| anyhow::anyhow!("every slot is running"))?;
                let old = self.slots.remove(i);
                self.e.free_slot(old.id);
                self.e.alloc_slot()?
            }
        };
        self.slots.push(SlotInfo {
            id,
            busy: true,
            key: key.map(String::from),
            used: self.clock,
        });
        if let Some(i) = snap.filter(|_| snap_len > parked) {
            self.snaps[i].used = self.clock;
            self.snaps[i].at = crate::retention::now();
            let (sn, e) = (&self.snaps[i], &mut self.e);
            let restored = sn
                .pos
                .iter()
                .try_for_each(|p| e.restore_positional_at(id, p, 0))
                .and_then(|()| e.restore_running_at(id, &sn.state, &sn.tokens));
            match restored {
                Ok(()) => return Ok((id, snap_len)),
                // A state from another configuration (say, saved without the MTP layer):
                // prefill instead.
                Err(e) => eprintln!("tang-llm: prefix state at {snap_len} not restored: {e}"),
            }
        }
        Ok((id, 0))
    }

    pub(super) fn batch_admit(&mut self, req: &Request) -> Result<Inflight> {
        self.expire_memory();
        let t0 = Instant::now();
        let (ids, text, _spliced) = self.prompt(req)?;
        self.record_warm(req);
        // Refused here, before a slot is taken (a 400 for the client); a prompt that fits
        // generates up to the window's end and stops there with `length`.
        let limit = crate::batch::token_limit(ids.len(), self.ctx, req.max_tokens)
            .map_err(|e| anyhow::anyhow!(e))?;
        let (temperature, top_k, top_p, presence) = self.request_sampling(req);
        let samp = crate::flash::sampler::RowSampling {
            temperature,
            top_k,
            top_p,
            seed: (req.sampling.seed ^ (req.sampling.seed >> 32)) as u32,
            presence,
        };
        let constraint = self.constraint(req)?;
        ensure!(
            constraint.is_none() || ids.len() >= 2,
            "prompt too short for structured output"
        );
        let host = crate::structured::HostSampling {
            temperature,
            top_k,
            top_p,
            seed: req.sampling.seed,
        };
        let (slot, reused) = self.take_slot(&ids, req.cache_key.as_deref())?;
        // An engine without per-row sampling refuses a sampled request: release the slot.
        if let Err(e) = self.e.set_slot_sampling(slot, samp) {
            if let Some(sl) = self.slots.iter_mut().find(|x| x.id == slot) {
                sl.busy = false;
            }
            return Err(e);
        }
        let mut marks = self.marks(&ids, &text);
        let end = ids.len() - constraint.is_some() as usize;
        marks.retain(|&(p, _)| p > reused && p <= end);
        let gen_at = text
            .rfind("<|im_start|>assistant\n")
            .map_or(text.len(), |i| i + "<|im_start|>assistant\n".len());
        let opened = text.trim_end().ends_with("<think>");
        self.requests += 1;
        // A constrained stream whose slot already holds all but the last prompt token decodes
        // straight away.
        let pending =
            (constraint.is_some() && reused == end && !req.prefill_only).then(|| ids[end]);
        let decode_from = pending.map(|_| Instant::now());
        let finish =
            (constraint.is_some() && reused == end && req.prefill_only).then_some(Finish::Stop);
        Ok(Box::new(Stream {
            slot,
            key: req.cache_key.clone(),
            done_prompt: reused,
            marks,
            reused,
            ids,
            pending,
            drafts: Vec::new(),
            drafted: 0,
            accepted: 0,
            windows: 0,
            out: Vec::new(),
            limit,
            stop: req.stop.clone(),
            parser: Parser::new(opened).with_tools(req.tools.as_ref()),
            detok: Detok::default(),
            text_out: String::new(),
            content: String::new(),
            calls: Vec::new(),
            tail: text[gen_at..].to_string(),
            finish,
            t0,
            first: None,
            prefill_s: 0.0,
            snap_s: 0.0,
            decode_from,
            prefill_only: req.prefill_only,
            sampled: samp.sampled(),
            broken: false,
            constraint,
            host,
        }))
    }

    /// Feed one sampled token to stream `s`'s output: end of turn, stop strings, the parser,
    /// delivery. Sets `s.finish` when the stream is done.
    fn batch_take(
        &self,
        s: &mut Stream,
        tok: u32,
        on: &mut dyn FnMut(Piece) -> bool,
    ) -> Result<()> {
        if self.eos.contains(&tok) {
            s.finish = Some(Finish::Stop);
            return Ok(());
        }
        s.out.push(tok);
        if let Some(c) = s.constraint.as_mut() {
            c.accept(tok)?;
        }
        for chunk in s
            .detok
            .push(&s.out, &self.tags, |ids| self.tok.decode(ids))?
        {
            let (new, tag) = match chunk {
                Chunk::Text(t) => (t, None),
                Chunk::Tag(t) => (t.to_string(), Some(t)),
            };
            s.text_out.push_str(&new);
            let stop_at = s
                .stop
                .iter()
                .filter_map(|x| s.text_out.find(x.as_str()))
                .min();
            let new = match stop_at {
                Some(i) => new[..new.len().saturating_sub(s.text_out.len() - i)].to_string(),
                None => new,
            };
            // Constrained output is the grammar's JSON verbatim: tool-call or think markers
            // inside a string value are data, not structure.
            let pieces = match tag {
                _ if s.constraint.is_some() => vec![Piece::Text(new)],
                Some(t) if new == t => s.parser.push_tag(t),
                _ => s.parser.push_text(&new),
            };
            for p in pieces {
                if s.first.is_none() {
                    s.first = Some(s.t0.elapsed().as_secs_f64() * 1e3);
                }
                match &p {
                    Piece::Text(t) => s.content.push_str(t),
                    Piece::ToolCall { name, arguments } => {
                        s.calls.push((name.clone(), arguments.clone()))
                    }
                    Piece::Reasoning(_) => {}
                }
                if !on(p) {
                    s.finish = Some(Finish::Cancelled);
                    return Ok(());
                }
            }
            if stop_at.is_some() {
                s.finish = Some(Finish::Stop);
                return Ok(());
            }
        }
        // The JSON value is complete: nothing but an end of turn may follow.
        if s.constraint.as_ref().is_some_and(|c| c.finished()) {
            s.finish = Some(Finish::Stop);
        } else if s.out.len() >= s.limit {
            s.finish = Some(Finish::Length);
        }
        Ok(())
    }

    /// One step for every lane; a failure names the lanes it hurt ([`StepFailure`]), whose
    /// streams are marked broken so retiring them drops their slots.
    pub(super) fn batch_step(&mut self, lanes: &mut [DynLane<'_>]) -> Result<()> {
        let result = self.batch_step_lanes(lanes);
        if let Some(f) = result
            .as_ref()
            .err()
            .and_then(|e| e.downcast_ref::<StepFailure>())
        {
            for &i in &f.lanes {
                if let Some(s) = lanes
                    .get_mut(i)
                    .and_then(|l| l.stream.downcast_mut::<Stream>())
                {
                    s.broken = true;
                    s.finish.get_or_insert(Finish::Cancelled);
                }
            }
        }
        result
    }

    fn batch_step_lanes(&mut self, lanes: &mut [DynLane<'_>]) -> Result<()> {
        let t_step = Instant::now();
        let prefill_tokens0: usize = lanes
            .iter()
            .map(|l| {
                let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
                s.done_prompt
            })
            .sum();
        // Cancellations first.
        for l in lanes.iter_mut() {
            let s = l.stream.downcast_mut::<Stream>().expect("a flash stream");
            if s.finish.is_none() && l.ctl.cancelled() {
                s.finish = Some(Finish::Cancelled);
            }
        }
        // Prefill, shortest remaining prompt first. With streams decoding, a short chunk and only
        // when decode has caught up on GPU time (prefill_debt), so both make progress.
        let decoding = lanes.iter().any(|l| {
            let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
            s.finish.is_none() && s.pending.is_some()
        });
        if !decoding {
            self.sched.idle();
        }
        let waiting: Vec<Waiting> = lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
                (s.finish.is_none() && s.done_prompt < s.prompt_end()).then(|| Waiting {
                    idx: i,
                    rem: s.prompt_end() - s.done_prompt,
                    age: s.t0.elapsed().as_secs_f64(),
                })
            })
            .collect();
        let wide = self.e.wide_rows();
        let policy = self
            .prefill_policy
            .get_or_insert_with(|| {
                let p = Policy::from_env(wide);
                eprintln!("tang-llm: batch prefill policy {p:?}");
                p
            })
            .clone();
        let plan = prefill_plan(&waiting, &policy, self.sched.budget(&policy, decoding));
        let oldest = waiting.iter().map(|w| w.age).reduce(f64::max);
        let n_chunks = plan.len();
        // The prompts' MTP K/V only pays if they'll draft: skip the fill with spec off, or
        // while every row of a decode window is taken (drafts need spare rows), as at C = 64.
        if self.e.use_mtp() && !plan.is_empty() {
            let live = lanes
                .iter()
                .filter(|l| l.stream.downcast_ref::<Stream>().unwrap().finish.is_none())
                .count();
            let width = crate::batch::window_width(MAX_T, self.e.batch_rows(), self.deterministic);
            let spec = self.spec.mode != crate::batch::spec::Mode::Off;
            self.e.set_mtp_fill(spec && live < width);
        }
        for (i, n) in plan {
            let l = &mut lanes[i];
            let s = l.stream.downcast_mut::<Stream>().expect("a flash stream");
            // Stop at the next prefix-state mark, to save the state there.
            let n = s
                .marks
                .first()
                .map_or(n, |&(m, _)| n.min(m - s.done_prompt));
            let t = Instant::now();
            let chunk = &s.ids[s.done_prompt..s.done_prompt + n];
            let next = if wide > 0 && n > 2 * MAX_T {
                self.e.prefill_slot_wide(s.slot, chunk)
            } else {
                self.e.prefill_slot(s.slot, chunk)
            }
            .map_err(|e| StepFailure::lanes(vec![i], e))?;
            let dt = t.elapsed().as_secs_f64();
            s.prefill_s += dt;
            self.sched.prefilled(n, dt, decoding);
            s.done_prompt += n;
            if let Some(&(_, pinned)) = s.marks.first().filter(|m| m.0 == s.done_prompt) {
                s.marks.remove(0);
                let t = Instant::now();
                let slot = s.slot;
                self.snapshot_at(slot, pinned)
                    .map_err(|e| StepFailure::lanes(vec![i], e))?;
                let s = lanes[i].stream.downcast_mut::<Stream>().unwrap();
                s.prefill_s += t.elapsed().as_secs_f64();
                s.snap_s += t.elapsed().as_secs_f64();
            }
            let l = &mut lanes[i];
            let s = l.stream.downcast_mut::<Stream>().expect("a flash stream");
            if s.done_prompt == s.prompt_end() && s.constraint.is_some() {
                if s.prefill_only {
                    s.finish = Some(Finish::Stop);
                } else {
                    // The last prompt token goes through the decode step (see `constraint`).
                    s.decode_from = Some(Instant::now());
                    s.pending = Some(s.ids[s.done_prompt]);
                }
            } else if s.done_prompt == s.ids.len() {
                if s.prefill_only {
                    s.finish = Some(Finish::Stop);
                } else {
                    s.decode_from = Some(Instant::now());
                    self.batch_take(s, next, l.on)
                        .map_err(|e| StepFailure::lanes(vec![i], e))?;
                    if s.finish.is_none() {
                        s.pending = Some(next);
                        s.drafts = self.e.slot_drafts(s.slot);
                    }
                }
            }
        }
        // Decode: every stream with a pending token, MAX_T rows a window (verify windows, with
        // drafts), or the batch module's width (plain decode, `enable_batch`).
        let ready: Vec<usize> = lanes
            .iter_mut()
            .enumerate()
            .filter_map(|(i, l)| {
                let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
                (s.finish.is_none() && s.pending.is_some()).then_some(i)
            })
            .collect();
        // Structured-output streams: one row each and no drafts, in windows of their own (their
        // logits are read before the MTP pass reuses the head buffer). The device's greedy
        // pick stands when the grammar allows it; otherwise the masked argmax of the row.
        let (constrained, ready): (Vec<usize>, Vec<usize>) = ready.into_iter().partition(|&i| {
            let s = lanes[i].stream.downcast_ref::<Stream>().unwrap();
            s.constraint.is_some()
        });
        let v = self.e.vocab_size();
        for group in constrained.chunks(MAX_T) {
            let rows: Vec<(usize, u32)> = group
                .iter()
                .map(|&i| {
                    let s = lanes[i].stream.downcast_ref::<Stream>().unwrap();
                    (s.slot, s.pending.unwrap())
                })
                .collect();
            let t = Instant::now();
            let (out, mut logits) = self
                .e
                .decode_batch_logits(&rows)
                .and_then(|(out, logits)| {
                    anyhow::ensure!(
                        out.len() == rows.len() && logits.len() >= rows.len() * v,
                        "decode window returned {} tokens, {} logits for {} rows",
                        out.len(),
                        logits.len(),
                        rows.len()
                    );
                    Ok((out, logits))
                })
                .map_err(|e| StepFailure::lanes(group.to_vec(), e))?;
            let share = policy.share_for(ready.len() + constrained.len(), self.batch, oldest);
            self.sched
                .decoded(t.elapsed().as_secs_f64(), share, &policy);
            for (k, &i) in group.iter().enumerate() {
                let l = &mut lanes[i];
                let s = l.stream.downcast_mut::<Stream>().unwrap();
                s.pending = None;
                s.windows += 1;
                let pos = (s.ids.len() + s.out.len()) as u64;
                let row = &mut logits[k * v..(k + 1) * v];
                let tok = s
                    .constraint
                    .as_mut()
                    .unwrap()
                    .choose(out[k], row, &s.host, pos)
                    .map_err(|e| StepFailure::lanes(vec![i], e))?;
                self.batch_take(s, tok, l.on)
                    .map_err(|e| StepFailure::lanes(vec![i], e))?;
                if s.finish.is_none() {
                    s.pending = Some(tok);
                }
            }
        }
        let mtp = self.e.use_mtp();
        // Streams per window: the batch module's width (spare rows go to drafts), or MAX_T
        // (no batch module, or TANG_SERVE_DETERMINISTIC=1).
        let width = crate::batch::window_width(MAX_T, self.e.batch_rows(), self.deterministic);
        for group in ready.chunks(width) {
            // Each stream's chain, cut at an end of turn and the token limit; sampled streams
            // draft nothing (verification is greedy).
            let chains: Vec<Vec<(u32, f32)>> = group
                .iter()
                .map(|&i| {
                    let s = lanes[i].stream.downcast_ref::<Stream>().unwrap();
                    if !mtp || s.sampled {
                        return Vec::new();
                    }
                    let room = s.limit.saturating_sub(s.out.len() + 1);
                    let mut d: Vec<(u32, f32)> = s.drafts.iter().take(room).copied().collect();
                    if let Some(k) = d.iter().position(|x| self.eos.contains(&x.0)) {
                        d.truncate(k);
                    }
                    d
                })
                .collect();
            // D-Cut over the window's spare rows, the budget from the measured table.
            let conf: Vec<Vec<f32>> = chains
                .iter()
                .map(|c| c.iter().map(|x| x.1).collect())
                .collect();
            // A window of sampled streams only: no drafts, no MTP pass.
            let greedy = group
                .iter()
                .any(|&i| !lanes[i].stream.downcast_ref::<Stream>().unwrap().sampled);
            let choice = if greedy {
                self.spec.choose(&conf, width.saturating_sub(group.len()))
            } else {
                crate::batch::spec::Choice {
                    drafts: vec![0; group.len()],
                    predicted: 0.0,
                    raw: 0.0,
                    mtp: false,
                }
            };
            self.e.set_slot_mtp(mtp && choice.mtp);
            let items: Vec<(usize, u32, Vec<u32>)> = group
                .iter()
                .zip(chains.iter().zip(&choice.drafts))
                .map(|(&i, (c, &g))| {
                    let s = lanes[i].stream.downcast_ref::<Stream>().unwrap();
                    (
                        s.slot,
                        s.pending.unwrap(),
                        c[..g].iter().map(|x| x.0).collect(),
                    )
                })
                .collect();
            let extra: usize = choice.drafts.iter().sum();
            let t = Instant::now();
            // A window's failure is its rows' (a slot past its context, say), unless it's a
            // device fault (`StepFailure::lanes` marks those fatal).
            let kept = if extra > 0 {
                self.e.verify_batch(&items)
            } else {
                let rows: Vec<(usize, u32)> = items.iter().map(|x| (x.0, x.1)).collect();
                self.e
                    .decode_batch(&rows)
                    .map(|k| k.into_iter().map(|k| vec![k]).collect())
            }
            .map_err(|e| StepFailure::lanes(group.to_vec(), e))?;
            let dt = t.elapsed().as_secs_f64();
            let share = policy.share_for(ready.len(), self.batch, oldest);
            self.sched.decoded(dt, share, &policy);
            let accepted: usize = kept.iter().map(|k| k.len() - 1).sum();
            if mtp {
                self.spec.observe(
                    group.len(),
                    extra,
                    choice.mtp,
                    dt * 1e3,
                    choice.raw,
                    accepted,
                );
                self.spec_note(group.len(), extra, accepted, dt);
            }
            for ((&i, toks), item) in group.iter().zip(&kept).zip(&items) {
                let l = &mut lanes[i];
                let s = l.stream.downcast_mut::<Stream>().unwrap();
                s.pending = None;
                s.windows += 1;
                s.drafted += item.2.len();
                s.accepted += toks.len() - 1;
                for &tok in toks {
                    self.batch_take(s, tok, l.on)
                        .map_err(|e| StepFailure::lanes(vec![i], e))?;
                    if s.finish.is_some() {
                        break;
                    }
                }
                if s.finish.is_none() {
                    s.pending = toks.last().copied();
                    s.drafts = if mtp {
                        self.e.slot_drafts(s.slot)
                    } else {
                        Vec::new()
                    };
                }
            }
        }
        // One step's shape, for the otel gauges/histograms (zero-sized without the feature).
        crate::telemetry::step(&crate::engine::StepStats {
            decode_rows: lanes
                .iter()
                .filter(|l| {
                    let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
                    s.windows > 0
                })
                .count(),
            decode_streams: lanes
                .iter()
                .filter(|l| {
                    let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
                    s.finish.is_none() && s.pending.is_some()
                })
                .count(),
            prefill_tokens: lanes
                .iter()
                .map(|l| {
                    let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
                    s.done_prompt
                })
                .sum::<usize>()
                .saturating_sub(prefill_tokens0),
            prefill_chunks: n_chunks,
            secs: t_step.elapsed().as_secs_f64(),
        });
        Ok(())
    }

    /// Count one window for the spec-decode summary (`tang-llm: spec:` every 30 s while
    /// windows run; every window with `TANG_SERVE_SPEC_LOG=1`).
    fn spec_note(&mut self, live: usize, extra: usize, accepted: usize, dt: f64) {
        let (c, at) = &mut self.spec_log;
        for (k, v) in [
            1.0,
            extra as f64,
            extra as f64,
            accepted as f64,
            (live + accepted) as f64,
            dt * 1e3,
        ]
        .into_iter()
        .enumerate()
        {
            c[k] += v;
        }
        if std::env::var("TANG_SERVE_SPEC_LOG").is_ok_and(|v| v == "1") {
            eprintln!(
                "tang-llm: spec window: {live} streams + {extra} drafts, {accepted} accepted, {:.1} ms, alpha {:.2}",
                dt * 1e3,
                self.spec.alpha
            );
        }
        if at.elapsed().as_secs_f64() >= 30.0 {
            eprintln!(
                "tang-llm: spec: {:.0} windows, {:.1} draft rows a window, {:.0}/{:.0} accepted ({:.0} %), {:.2} tokens a stream-window, {:.0} tok/s in windows, alpha {:.2}",
                c[0],
                c[1] / c[0].max(1.0),
                c[3],
                c[2],
                100.0 * c[3] / c[2].max(1.0),
                c[4] / (c[4] - c[3]).max(1.0),
                c[4] / (c[5] / 1e3).max(1e-9),
                self.spec.alpha
            );
            *c = [0.0; 6];
            *at = Instant::now();
        }
    }

    pub(super) fn batch_state(s: &Inflight) -> (bool, bool) {
        let s = s.downcast_ref::<Stream>().expect("a flash stream");
        (s.finish.is_some(), s.done_prompt < s.prompt_end())
    }

    pub(super) fn batch_retire(&mut self, s: Inflight) -> Result<Outcome> {
        let Ok(mut s) = s.downcast::<Stream>() else {
            bail!("not a flash stream")
        };
        let finish = s.finish.unwrap_or(Finish::Cancelled);
        let mut finish = finish;
        if finish != Finish::Cancelled {
            let _ = s.parser.finish();
            if !s.calls.is_empty() && finish == Finish::Stop {
                finish = Finish::ToolCalls;
            }
        }
        // Park the slot. The pending token (if any) was sampled but never fed: the slot's
        // tokens are the prompt plus the output before it.
        if s.broken {
            if let Some(k) = self.slots.iter().position(|x| x.id == s.slot) {
                self.slots.remove(k);
                self.e.free_slot(s.slot);
            }
        } else if let Some(sl) = self.slots.iter_mut().find(|x| x.id == s.slot) {
            sl.busy = false;
            sl.key = s.key.clone();
        }
        if matches!(finish, Finish::Stop | Finish::ToolCalls)
            && s.stop.iter().all(|x| !s.text_out.contains(x.as_str()))
        {
            self.memo.push_back(Memo {
                content: s.content.clone(),
                calls: s.calls.clone(),
                tail: s.tail.clone(),
                ids: s.out.clone(),
                at: crate::retention::now(),
            });
            while self.memo.len() > 256 {
                self.memo.pop_front();
            }
        }
        let decode_s = s.decode_from.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let prefilled = s.ids.len() - s.reused;
        eprintln!(
            "tang-llm: batch req: slot {} prompt {} (reused {}, prefilled {} in {:.0} ms, state saves {:.0} ms) | ttft {:.0} ms | decode {} tokens in {:.2} s = {:.1} tok/s, {} windows, drafts {}/{} accepted | {} tool calls",
            s.slot,
            s.ids.len(),
            s.reused,
            prefilled,
            s.prefill_s * 1e3,
            s.snap_s * 1e3,
            s.first.unwrap_or(0.0),
            s.out.len(),
            decode_s,
            s.out.len() as f64 / decode_s.max(1e-9),
            s.windows,
            s.accepted,
            s.drafted,
            s.calls.len()
        );
        Ok(Outcome::Done(
            finish,
            Usage {
                prompt_tokens: s.ids.len(),
                cached_tokens: s.reused,
                completion_tokens: s.out.len(),
                prefill_tok_s: prefilled as f64 / s.prefill_s.max(1e-9),
                decode_tok_s: s.out.len() as f64 / decode_s.max(1e-9),
                ..Default::default()
            },
        ))
    }
}
