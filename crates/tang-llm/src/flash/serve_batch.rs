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
//! - No drafts in batched windows yet (per-slot verify is engine work still to come).

use super::{FlashServe, Memo};
use crate::batch::prefill::{prefill_plan, Policy, Waiting};
use crate::chat::{Chunk, Detok, Parser, Piece};
use crate::engine::{Finish, Outcome, Request, Usage};
use crate::server::{DynLane, Inflight, StepFailure};
use anyhow::{bail, ensure, Result};
use serde_json::Value;
use std::time::Instant;
use tang_compute::flash::MAX_T;

/// Draft rows per stream in one window, for streams wanting `want[i]` drafts: one row each for
/// its token, and the window's spare rows (`MAX_T` minus the streams) shared out round-robin, so
/// a lone stream can draft its whole chain and a full window drafts nothing.
fn share_drafts(want: &[usize]) -> Vec<usize> {
    let mut give = vec![0; want.len()];
    let mut spare = MAX_T.saturating_sub(want.len());
    while spare > 0 {
        let mut moved = false;
        for (g, &w) in give.iter_mut().zip(want) {
            if spare > 0 && *g < w {
                *g += 1;
                spare -= 1;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    give
}

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
    /// MTP drafts for the next window (the engine's chain for this slot).
    drafts: Vec<u32>,
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
    decode_from: Option<Instant>,
    prefill_only: bool,
    /// Sampled (temperature > 0): no drafts.
    sampled: bool,
    /// A step failed this stream: its slot's state is suspect, so retiring frees the slot
    /// instead of parking it for reuse.
    broken: bool,
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
        ensure!(
            ids.len() + 1 < self.ctx,
            "prompt is {} tokens; the context window is {}",
            ids.len(),
            self.ctx
        );
        let (temperature, top_k, top_p, presence) = self.request_sampling(req);
        let samp = crate::flash::sampler::RowSampling {
            temperature,
            top_k,
            top_p,
            seed: (req.sampling.seed ^ (req.sampling.seed >> 32)) as u32,
            presence,
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
        marks.retain(|&(p, _)| p > reused);
        let limit = req
            .max_tokens
            .unwrap_or(usize::MAX)
            .min(self.ctx - ids.len() - 1)
            .max(1);
        let gen_at = text
            .rfind("<|im_start|>assistant\n")
            .map_or(text.len(), |i| i + "<|im_start|>assistant\n".len());
        let opened = text.trim_end().ends_with("<think>");
        self.requests += 1;
        Ok(Box::new(Stream {
            slot,
            key: req.cache_key.clone(),
            done_prompt: reused,
            marks,
            reused,
            ids,
            pending: None,
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
            finish: None,
            t0,
            first: None,
            prefill_s: 0.0,
            decode_from: None,
            prefill_only: req.prefill_only,
            sampled: samp.sampled(),
            broken: false,
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
            let pieces = match tag {
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
        if s.out.len() >= s.limit {
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
                (s.finish.is_none() && s.done_prompt < s.ids.len()).then(|| Waiting {
                    idx: i,
                    rem: s.ids.len() - s.done_prompt,
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
        let prompts_waiting = !waiting.is_empty();
        let n_chunks = plan.len();
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
            }
            let l = &mut lanes[i];
            let s = l.stream.downcast_mut::<Stream>().expect("a flash stream");
            if s.done_prompt == s.ids.len() {
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
        let mtp = self.e.use_mtp();
        let width = if mtp {
            MAX_T
        } else {
            MAX_T.max(self.e.batch_rows())
        };
        for group in ready.chunks(width) {
            // Drafts for each stream, cut at an end of turn and the token limit, then the
            // window's spare rows shared out.
            let want: Vec<Vec<u32>> = group
                .iter()
                .map(|&i| {
                    let s = lanes[i].stream.downcast_ref::<Stream>().unwrap();
                    if !mtp || s.sampled {
                        return Vec::new();
                    }
                    let room = s.limit.saturating_sub(s.out.len() + 1);
                    let mut d: Vec<u32> = s.drafts.iter().take(room).copied().collect();
                    if let Some(k) = d.iter().position(|x| self.eos.contains(x)) {
                        d.truncate(k);
                    }
                    d
                })
                .collect();
            let give = share_drafts(&want.iter().map(Vec::len).collect::<Vec<_>>());
            let items: Vec<(usize, u32, Vec<u32>)> = group
                .iter()
                .zip(want.iter().zip(&give))
                .map(|(&i, (d, &g))| {
                    let s = lanes[i].stream.downcast_ref::<Stream>().unwrap();
                    (s.slot, s.pending.unwrap(), d[..g].to_vec())
                })
                .collect();
            let t = Instant::now();
            // A window's failure is its rows' (a slot past its context, say), unless it's a
            // device fault (`StepFailure::lanes` marks those fatal).
            let kept = if mtp {
                self.e.verify_batch(&items)
            } else {
                let rows: Vec<(usize, u32)> = items.iter().map(|x| (x.0, x.1)).collect();
                self.e
                    .decode_batch(&rows)
                    .map(|k| k.into_iter().map(|k| vec![k]).collect())
            }
            .map_err(|e| StepFailure::lanes(group.to_vec(), e))?;
            let share = policy.share_for(ready.len(), self.batch, prompts_waiting);
            self.sched
                .decoded(t.elapsed().as_secs_f64(), share, &policy);
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

    pub(super) fn batch_state(s: &Inflight) -> (bool, bool) {
        let s = s.downcast_ref::<Stream>().expect("a flash stream");
        (s.finish.is_some(), s.done_prompt < s.ids.len())
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
            "tang-llm: batch req: slot {} prompt {} (reused {}, prefilled {} in {:.0} ms) | ttft {:.0} ms | decode {} tokens in {:.2} s = {:.1} tok/s, {} windows, drafts {}/{} accepted | {} tool calls",
            s.slot,
            s.ids.len(),
            s.reused,
            prefilled,
            s.prefill_s * 1e3,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spare_rows_go_to_drafts_round_robin() {
        assert_eq!(share_drafts(&[2]), vec![2]);
        assert_eq!(share_drafts(&[2, 2, 2]), vec![2, 2, 1]);
        assert_eq!(share_drafts(&[0, 3, 1]), vec![0, 3, 1]);
        assert_eq!(share_drafts(&[2; 8]), vec![0; 8]);
        assert_eq!(share_drafts(&[2; 6]), vec![1, 1, 0, 0, 0, 0]);
    }
}
