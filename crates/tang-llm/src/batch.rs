//! Who's in the batch: the rules the server's worker follows between decode steps (see
//! `docs/batching.md`). Pure, so they're tested without a model.
//!
//! Requests join while there's a free KV slot, their conversation isn't already running (it
//! has one cache) and the KV budget has room; the interactive lane goes first. Running streams
//! share each step's prefill budget in lane order: interactive before background, then by
//! arrival. A background stream that's still prefilling gives way when interactive work is
//! waiting that couldn't join (its slot is what it needs); one that's decoding keeps going.

use crate::queue::{Priority, Ticket};

#[path = "batch_prefill.rs"]
pub mod prefill;

/// What the rules need to know about the engine's slots right now.
#[derive(Debug, Clone, Copy)]
pub struct Room {
    /// Streams running.
    pub running: usize,
    /// Most streams that may run at once.
    pub max: usize,
    /// The KV pool can hold another stream's prompt and a typical reply.
    pub kv: bool,
}

/// Whether a waiting request in conversation `key` joins the batch now. `running_keys`: the
/// conversations running. An empty batch takes anything (it must make progress, and a prompt
/// too long for the KV budget grows past it as it always has).
pub fn joins(key: Option<&str>, running_keys: &[&str], room: Room) -> bool {
    if room.running == 0 {
        return true;
    }
    room.running < room.max && room.kv && key.is_none_or(|k| !running_keys.contains(&k))
}

/// The order streams get the prefill budget in: interactive first, then by arrival.
pub fn lane_order(tickets: &[&Ticket]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..tickets.len()).collect();
    idx.sort_by_key(|&i| (tickets[i].priority != Priority::Interactive, tickets[i].id));
    idx
}

/// Whether a running stream should stop prefilling now and give its slot back.
/// `blocked_interactive`: interactive work is waiting that couldn't join.
pub fn yields(priority: Priority, prefilling: bool, blocked_interactive: bool) -> bool {
    priority == Priority::Background && prefilling && blocked_interactive
}

/// Sequence slots that fit in `free` device bytes less `headroom`, `per` bytes each, at least
/// 1 and below `max_slots` (slot 0 is the engine's own).
pub fn slots_that_fit(free: usize, headroom: usize, per: usize, max_slots: usize) -> usize {
    (free.saturating_sub(headroom) / per.max(1)).clamp(1, max_slots.max(2) - 1)
}

/// `--batch` within the engine's slots (`max_slots`, slot 0 excluded) and below the VRAM cap
/// (`cap` slots fit; one stays spare so admitting can always evict a parked slot), at least 1.
pub fn clamp_batch(batch: usize, max_slots: usize, cap: usize) -> usize {
    batch
        .min(max_slots.saturating_sub(1))
        .min(cap.saturating_sub(1))
        .max(1)
}

/// Admission against the context window: a prompt of `prompt` tokens needs room for itself
/// and one generated token, else it's refused (the message says so, for a 400); otherwise
/// the most tokens it may generate: `max_tokens`, cut so the sequence ends inside `ctx`
/// (finish reason `length` there), at least one.
pub fn token_limit(prompt: usize, ctx: usize, max_tokens: Option<usize>) -> Result<usize, String> {
    if prompt + 1 >= ctx {
        return Err(format!(
            "context_length_exceeded: the prompt is {prompt} tokens; the context window is {ctx}"
        ));
    }
    Ok(max_tokens
        .unwrap_or(usize::MAX)
        .min(ctx - prompt - 1)
        .max(1))
}

/// A decode step of `rows` sequences runs in the batch module (`Coder::enable_batch`, built
/// for `batch_rows`, a multiple of 32) only past the narrow module's `narrow` rows; 1..=narrow
/// rows always take the narrow module, batch module or not. The batch module's dense paths
/// (K-split GEMV, Q8X MMA, router) are all gated on more than `narrow` rows, so a short window
/// there falls through to the narrow kernels on the batch module's scratch and staging layout
/// (int8 scales at `batch_rows * k`, not `narrow * k`), the likely cause of the
/// `CUDA_ERROR_ILLEGAL_ADDRESS` the dropped `batch_all` experiment (every step through the batch
/// module) hit on one-row steps.
pub fn batch_module(rows: usize, narrow: usize, batch_rows: usize) -> bool {
    rows > narrow && rows <= batch_rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn ticket(id: u64, priority: Priority) -> Ticket {
        Ticket {
            id,
            priority,
            prompt_tokens: None,
            since: Instant::now(),
            yields: 0,
        }
    }

    const ROOM: Room = Room {
        running: 2,
        max: 4,
        kv: true,
    };

    #[test]
    fn joins_while_there_is_a_slot_and_kv_room() {
        assert!(joins(None, &["a", "b"], ROOM));
        assert!(joins(Some("c"), &["a", "b"], ROOM));
        let full = Room { running: 4, ..ROOM };
        assert!(!joins(None, &[], full));
        let tight = Room { kv: false, ..ROOM };
        assert!(!joins(None, &[], tight));
    }

    #[test]
    fn a_conversation_runs_once_at_a_time() {
        assert!(!joins(Some("a"), &["a", "b"], ROOM));
    }

    #[test]
    fn an_empty_batch_takes_anything() {
        let empty = Room {
            running: 0,
            max: 1,
            kv: false,
        };
        assert!(joins(Some("a"), &[], empty));
    }

    #[test]
    fn interactive_streams_prefill_first_then_by_arrival() {
        let t = [
            ticket(1, Priority::Background),
            ticket(4, Priority::Interactive),
            ticket(2, Priority::Interactive),
            ticket(3, Priority::Background),
        ];
        let refs: Vec<&Ticket> = t.iter().collect();
        assert_eq!(lane_order(&refs), [2, 1, 0, 3]);
    }

    #[test]
    fn only_background_prefill_yields_and_only_to_blocked_interactive_work() {
        assert!(yields(Priority::Background, true, true));
        assert!(!yields(Priority::Background, false, true));
        assert!(!yields(Priority::Background, true, false));
        assert!(!yields(Priority::Interactive, true, true));
    }

    #[test]
    fn batch_stays_under_the_slots_that_fit() {
        const GB: usize = 1_000_000_000;
        // 0.85 GB a slot, 60 GB free, 4 GB headroom: 65 fit.
        let cap = super::slots_that_fit(60 * GB, 4 * GB, 850_000_000, 128);
        assert_eq!(cap, 65);
        assert_eq!(super::clamp_batch(64, 128, cap), 64);
        assert_eq!(super::clamp_batch(96, 128, cap), 64);
        // Less free memory than headroom: one slot, and batching is off.
        let cap = super::slots_that_fit(3 * GB, 4 * GB, 850_000_000, 128);
        assert_eq!(cap, 1);
        assert_eq!(super::clamp_batch(64, 128, cap), 1);
        // Lots of memory: the engine's slot count bounds it.
        let cap = super::slots_that_fit(usize::MAX, 0, 1, 128);
        assert_eq!(cap, 127);
        assert_eq!(super::clamp_batch(200, 128, cap), 126);
        assert_eq!(super::clamp_batch(8, 128, cap), 8);
    }

    /// `serve --batch 128 --ctx 131072` on an H200 (3.30 GB a slot, 4 GB headroom): 96.6 GB
    /// free before the windows, 93.8 GB after. The provisional count sizes the windows; the
    /// count that sticks comes from `--batch` against the second count, once (it was 128 → 27
    /// → 26 with two warnings, the second clamping the already clamped value).
    #[test]
    fn startup_clamps_once_from_the_request() {
        const GB: usize = 1_000_000_000;
        let per = 3_300_000_000;
        let before = super::slots_that_fit(96_600_000_000, 4 * GB, per, 128);
        let after = super::slots_that_fit(93_800_000_000, 4 * GB, per, 128);
        assert_eq!((before, after), (28, 27));
        let provisional = super::clamp_batch(128, 128, before);
        let fin = super::clamp_batch(128, 128, after).min(provisional);
        assert_eq!((provisional, fin), (27, 26));
        // Room to spare after the windows: unchanged, never above what the windows fit.
        let fin = super::clamp_batch(16, 128, after).min(super::clamp_batch(16, 128, before));
        assert_eq!(fin, 16);
        assert_eq!(super::clamp_batch(128, 128, 40).min(provisional), provisional);
    }

    #[test]
    fn prompts_past_the_context_are_refused_and_outputs_stop_inside_it() {
        assert!(super::token_limit(4096, 4096, None).is_err());
        assert!(super::token_limit(5000, 4096, Some(1)).is_err());
        assert!(super::token_limit(4095, 4096, None).is_err());
        assert_eq!(super::token_limit(4094, 4096, None), Ok(1));
        assert_eq!(super::token_limit(4000, 4096, Some(1000)), Ok(95));
        assert_eq!(super::token_limit(100, 4096, Some(256)), Ok(256));
        assert_eq!(super::token_limit(100, 4096, Some(0)), Ok(1));
        let e = super::token_limit(5000, 4096, None).unwrap_err();
        assert!(e.contains("5000") && e.contains("4096"), "{e}");
    }

    #[test]
    fn short_steps_never_take_the_batch_module() {
        // `serve --batch 64`: 64-row module, narrow 8.
        for rows in 1..=8 {
            assert!(!super::batch_module(rows, 8, 64), "{rows} rows");
        }
        for rows in 9..=64 {
            assert!(super::batch_module(rows, 8, 64), "{rows} rows");
        }
        assert!(!super::batch_module(65, 8, 64));
        // No batch module: never.
        assert!(!super::batch_module(9, 8, 0));
    }
}
