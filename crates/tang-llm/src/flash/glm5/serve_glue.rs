//! `ServeEngine` for GLM-5.3-Flash (W6): adapts W1's [`Glm5Engine`] — whose window takes the
//! ids and returns their logits — to the server's position/verify contract, without touching
//! the engine's math.
//!
//! - **Greedy only.** `set_sampling` is stored; the GPU head-sampler (a `Sampler`) is not
//!   built for this engine, so `sampler_mut()` is `None` and decode walks argmax.
//! - **Sequence slots** (`TANG_GLM5_SLOTS`, default 8; 1 = the pre-batching single
//!   sequence). Each slot owns the engine state a sequence accumulates: the KDA conv +
//!   recurrent state (host copies and the device's `[slots][...]` slices, 4 MiB + 288 KiB
//!   per layer per slot), the DSA latent + indexer caches (host mirrors + the `MlaCuda`
//!   LRU entry, id = slot), the residual streams and the logits (per window, no state) and
//!   the tokens. `alloc_slot` hands out a free slot; a finished request keeps it parked
//!   (`free_slot` drops the sequence) so its next turn extends the live caches instead of
//!   re-prefilling.
//! - **Snapshots carry the real state** (`save_running_at`/`restore_running_at`): the KDA
//!   state's 34 × 4 MiB + 288 KiB is readable from the device (`read_back_at`) and
//!   re-seedable, so a saved state restores without replaying the prefix (the old glue
//!   re-prefilled everything).
//! - **The batcher** (serve_batch.rs): `decode_batch` runs ONE engine window over every
//!   ready slot (`Glm5Engine::window_batch`): the per-token stages (embedding, mHC, KDA
//!   projections / o_proj, FFN, head) run once for all rows, only the KDA core and the DSA
//!   attention split per slot. `TANG_GLM5_BATCH_RR=1` runs one window per row instead.
//! - **No drafts**: `use_mtp()` is false and the server runs this engine with
//!   `suffix: false`, so `verify` always sees a single token and never needs to rewind.

use super::engine::{argmax, Glm5Engine};
use super::kda::{ConvState, KdaConfig, KdaState};
use crate::flash::engine::{Positional, RunningState, WinStats};
use crate::flash::serve::ServeEngine;
use anyhow::{ensure, Result};
use std::path::Path;
use std::time::Instant;

/// GLM-5.3-Flash behind the server's engine trait.
pub struct Glm5Serve {
    e: Glm5Engine,
    /// Sequence slots (`TANG_GLM5_SLOTS`; 1 = single sequence).
    slots: usize,
    /// The last window's logits per slot (`[t][n_vocab]`), for [`Glm5Serve::logits_at`]
    /// and slot 0's [`ServeEngine::logits`].
    last_logits: Vec<Vec<f32>>,
    last_window: WinStats,
}

/// The default slot count (`TANG_GLM5_SLOTS`).
pub fn slots_from_env() -> usize {
    std::env::var("TANG_GLM5_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1)
}

impl Glm5Serve {
    /// Open `path`, move the weights onto the GPU when built with `cuda` and a device is
    /// present, and warm the lazy per-layer weight caches with one throwaway window.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_slots(path, slots_from_env())
    }

    /// [`Self::load`] with `slots` sequence slots (`slots_from_env()`).
    pub fn load_slots(path: &Path, slots: usize) -> Result<Self> {
        let mut e = Glm5Engine::open(path)?;
        let moved = if slots > 1 {
            e.use_gpu_slots(slots)?
        } else {
            e.use_gpu()?
        };
        // Fault in the head and any cached weights once, then drop the throwaway state.
        let _ = e.prefill_slot(0, &[0], 1);
        e.reset();
        if moved && slots > 1 {
            eprintln!("glm5 serve: {slots} sequence slots (TANG_GLM5_SLOTS)");
        }
        Ok(Glm5Serve {
            e,
            slots,
            last_logits: (0..slots).map(|_| Vec::new()).collect(),
            last_window: WinStats::default(),
        })
    }

    /// Run `ids` through the engine as one window for `slot` (the engine appends them
    /// itself), cache the logits, and record the wall time.
    fn run_window(&mut self, slot: usize, ids: &[u32]) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let lg = self.e.window_slot(slot, ids)?;
        let secs = t0.elapsed().as_secs_f64();
        self.last_window = WinStats {
            t: ids.len(),
            wall_ms: secs * 1e3,
            ..WinStats::default()
        };
        crate::telemetry::glm5_window(ids.len(), secs);
        self.last_logits[slot] = lg;
        Ok(self.last_logits[slot].clone())
    }

    /// The KDA state parts of one slot's sequence, in layer order (conv history then
    /// recurrent state per KDA layer), read off the device when it holds them. The two
    /// trailing empty parts are the `RunningState` contract's qsa ring / PLE ring (this
    /// engine has none — `from_parts` splits them off by `gdn_layers`).
    fn kda_parts(&mut self, slot: usize) -> Result<Vec<Vec<f32>>> {
        let mut parts = Vec::new();
        let layers = self.e.m.hp.n_layer;
        for l in 0..layers {
            if !self.e.m.hp.is_kda[l] {
                continue;
            }
            let Some(st) = self.e.kda_state_at(slot, l) else {
                // no state yet: still emit zeros, so the parts line up by layer on restore
                let cfg = self.e.kda_config();
                parts.push(vec![0.0; cfg.conv_dim() * (cfg.d_conv - 1)]);
                parts.push(vec![0.0; cfg.heads * cfg.head_dim * cfg.head_dim]);
                continue;
            };
            parts.push(st.0 .0);
            parts.push(st.1 .0);
        }
        parts.push(Vec::new()); // qsa ring: none
        parts.push(Vec::new()); // PLE ring: none
        Ok(parts)
    }
}

impl ServeEngine for Glm5Serve {
    fn tokens(&self) -> &Vec<u32> {
        &self.e.tokens
    }

    fn tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.e.tokens
    }

    fn reset(&mut self) {
        self.e.reset();
        for lg in &mut self.last_logits {
            lg.clear();
        }
    }

    fn gguf(&self) -> &crate::gguf::Gguf {
        &self.e.m.g
    }

    fn last(&self) -> WinStats {
        self.last_window
    }

    fn save_running(&mut self) -> Result<RunningState> {
        self.save_running_at(0)
    }

    /// The KDA conv + recurrent state of sequence `slot` (the device's own values when it
    /// holds them), so a restore is exact — no prefix replay.
    fn save_running_at(&mut self, slot: usize) -> Result<RunningState> {
        ensure!(slot < self.slots, "slot {slot} >= {}", self.slots);
        let n = self.e.slot_tokens(slot).len();
        RunningState::from_parts(n, self.kda_layers(), self.kda_parts(slot)?)
    }

    fn restore_running(&mut self, s: &RunningState, prefix: &[u32]) -> Result<()> {
        self.restore_running_at(0, s, prefix)
    }

    /// Restore sequence `slot` to `s` (its KDA state in `parts`, tokens `prefix`): put the
    /// state back into the engine (the device re-seeds from it), truncate the tokens, and
    /// reset the DSA caches (the positional state is NOT saved — a restore re-derives it
    /// by re-prefilling the restored state's tail, which the caller does with the ids past
    /// the longest saved prefix; here the caller's `prefix` IS the whole prompt, so the
    /// DSA caches rebuild on the first window after the KDA restore).
    fn restore_running_at(&mut self, slot: usize, s: &RunningState, prefix: &[u32]) -> Result<()> {
        ensure!(prefix.len() == s.pos, "prefix/state length mismatch");
        ensure!(
            s.pos == self.e.slot_tokens(slot).len() || self.e.slot_tokens(slot).is_empty(),
            "slot {slot} holds a different sequence ({tok} tokens vs {pos})",
            tok = self.e.slot_tokens(slot).len(),
            pos = s.pos
        );
        self.e.restore_kda_parts(
            slot,
            &s.parts().iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        )?;
        self.e.slot_tokens_mut(slot).clear();
        self.e.slot_tokens_mut(slot).extend_from_slice(prefix);
        // the DSA caches are per-position and not saved: drop them; the caller prefills
        // the suffix (positions past `s.pos`) and the windows rebuild them
        self.e.reset_dsa(slot);
        self.last_logits[slot.min(self.slots - 1)].clear();
        Ok(())
    }

    fn save_positional(&mut self, from: usize, to: usize) -> Result<Positional> {
        Ok(Positional::from_parts(from, to, Vec::new()))
    }

    fn restore_positional(&mut self, _p: &Positional, _from: usize) -> Result<()> {
        Ok(())
    }

    fn save_positional_at(&mut self, _slot: usize, from: usize, to: usize) -> Result<Positional> {
        // the DSA caches (the positional state) are not saved: snapshots carry the KDA
        // state (the running part) only, so this is the empty positional for any slot
        Ok(Positional::from_parts(from, to, Vec::new()))
    }

    fn restore_positional_at(&mut self, _slot: usize, _p: &Positional, _from: usize) -> Result<()> {
        Ok(())
    }

    fn prefill(&mut self, ids: &[u32], chunk: usize) -> Result<u32> {
        {
            let _ = chunk;
            self.prefill_slot(0, ids)
        }
    }

    fn prefill_slot(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        // Appends (the trait contract; the caller has not extended `tokens`).
        let mut next = 0u32;
        for c in ids.chunks(512.max(1)) {
            let lg = self.run_window(slot, c)?;
            next = argmax(&lg[(c.len() - 1) * self.e.m.hp.n_vocab..]);
        }
        Ok(next)
    }

    fn window(&mut self, pos0: usize, t: usize) -> Result<Vec<u32>> {
        // The caller has already extended `tokens` with the window's ids; the engine's
        // `window` appends what it is given, so hand it exactly that slice.
        ensure!(pos0 + t <= self.e.tokens.len(), "window past the sequence");
        // The caller extended `tokens` past the window (prefill_span extends by the whole
        // span first); the engine appends what it runs and reads its position from the
        // token count — so cut back to `pos0`, run, and put the span's tail back.
        let ids: Vec<u32> = self.e.tokens[pos0..pos0 + t].to_vec();
        let tail: Vec<u32> = self.e.tokens[pos0 + t..].to_vec();
        self.e.tokens.truncate(pos0);
        let lg = self.run_window(0, &ids);
        self.e.tokens.extend_from_slice(&tail);
        let lg = lg?;
        Ok((0..t)
            .map(|i| argmax(&lg[i * self.e.m.hp.n_vocab..(i + 1) * self.e.m.hp.n_vocab]))
            .collect())
    }

    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        self.verify_slot(0, cur, drafts)
    }

    fn logits(&self, t: usize) -> Vec<f32> {
        self.logits_at(0, t)
    }

    fn set_sampling(&mut self, _temperature: f32, _seed: u32) {
        // Greedy only (see the module docs).
    }

    fn sampler_mut(&mut self) -> Option<&mut crate::flash::sampler::Sampler> {
        None
    }

    fn sampler_begin(&mut self, _pos: usize) -> Result<()> {
        Ok(())
    }

    fn max_slots(&self) -> usize {
        self.slots
    }

    fn slot_tokens(&self, slot: usize) -> &[u32] {
        self.e.slot_tokens(slot)
    }

    fn alloc_slot(&mut self) -> Result<usize> {
        // The engine keeps every slot alive from load (the device state is allocated once,
        // no per-slot alloc churn); hand out the first empty one, else the emptiest RESET
        // to fresh (the caller restores a snapshot into it or prefills the whole prompt —
        // the batcher only calls this when no parked slot's tokens matched, so dropping
        // the shortest sequence loses nothing reusable).
        let mut best = 0;
        let mut best_len = usize::MAX;
        for s in 0..self.slots {
            let n = self.e.slot_tokens(s).len();
            if n == 0 {
                return Ok(s);
            }
            if n < best_len {
                best_len = n;
                best = s;
            }
        }
        self.free_slot(best);
        Ok(best)
    }

    fn free_slot(&mut self, slot: usize) {
        self.e.reset_slot(slot);
        self.last_logits[slot.min(self.slots - 1)].clear();
    }

    fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        // ONE engine window over every ready slot (`Glm5Engine::window_batch`): the
        // per-token stages run once for all rows, only the KDA core / DSA attention split
        // per slot. Greedy argmax per row. `TANG_GLM5_BATCH_RR=1`: one window per row
        // instead (the round-robin A/B).
        let nv = self.e.m.hp.n_vocab;
        if std::env::var("TANG_GLM5_BATCH_RR").is_ok_and(|v| v == "1") {
            let mut out = Vec::with_capacity(rows.len());
            for &(slot, tok) in rows {
                let lg = self.run_window(slot, &[tok])?;
                out.push(argmax(&lg[(lg.len() - nv)..]));
            }
            return Ok(out);
        }
        let t0 = Instant::now();
        let lg = self.e.window_batch(rows)?;
        let secs = t0.elapsed().as_secs_f64();
        self.last_window = WinStats {
            t: rows.len(),
            wall_ms: secs * 1e3,
            ..WinStats::default()
        };
        crate::telemetry::glm5_window(rows.len(), secs);
        let mut out = Vec::with_capacity(rows.len());
        for (i, &(slot, _)) in rows.iter().enumerate() {
            let row = lg[i * nv..(i + 1) * nv].to_vec();
            out.push(argmax(&row));
            self.last_logits[slot] = row;
        }
        Ok(out)
    }

    fn verify_batch(&mut self, items: &[(usize, u32, Vec<u32>)]) -> Result<Vec<Vec<u32>>> {
        let mut out = Vec::with_capacity(items.len());
        for &(slot, cur, ref drafts) in items {
            out.push(self.verify_slot(slot, cur, drafts)?);
        }
        Ok(out)
    }
}

impl Glm5Serve {
    /// One verify window for `slot` (a single token: no drafts for this engine).
    fn verify_slot(&mut self, slot: usize, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        // Drafting is off for this engine (no MTP, no suffix drafts), so `drafts` is empty
        // and a verify is one decode step. A non-empty draft list would need a state
        // rewind: run the chain, keep the greedy prefix, replay the kept tokens
        // (correct, slow, and currently unreachable).
        // the engine appends what it runs (no push here: a pushed `cur` would be counted
        // twice and put the DSA caches one position out of step)
        let mut lg = self.run_window(slot, &[cur])?;
        let nv = self.e.m.hp.n_vocab;
        let mut kept = vec![argmax(&lg[(lg.len() - nv)..])];
        if drafts.is_empty() {
            return Ok(kept);
        }
        for (i, &d) in drafts.iter().enumerate() {
            lg = self.run_window(slot, &[d])?;
            let next = argmax(&lg[(lg.len() - nv)..]);
            if next != d || i + 1 == drafts.len() {
                break;
            }
            kept.push(next);
        }
        // Rewind: the engine state now covers the whole attempted chain; rebuild it for the
        // accepted prefix only.
        let toks = self.e.slot_tokens(slot);
        let keep_to = toks.len() - (drafts.len() + 1 - kept.len());
        let prefix: Vec<u32> = toks[..keep_to].to_vec();
        self.e.reset_slot(slot);
        if !prefix.is_empty() {
            self.e.prefill_slot(slot, &prefix, 512)?;
        }
        Ok(kept)
    }

    fn logits_at(&self, slot: usize, t: usize) -> Vec<f32> {
        let nv = self.e.m.hp.n_vocab;
        let lg = &self.last_logits[slot.min(self.slots - 1)];
        let start = lg.len().saturating_sub(t * nv);
        lg[start..].to_vec()
    }

    /// The KDA layer count (the state's `gdn_layers`).
    fn kda_layers(&self) -> usize {
        self.e.m.hp.is_kda.iter().filter(|&&k| k).count()
    }
}
