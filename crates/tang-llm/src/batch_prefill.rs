//! The batch prefill scheduler for `flash serve --batch N` (`flash/serve_batch.rs`): which
//! prompts prefill this step, how many tokens, and how GPU time is split with decode windows.
//! Pure, so it's tested on a simulated engine (`docs/qwen3next-batching.md`, "Scheduler for
//! B = 64").

/// The prefill scheduler's knobs (`TANG_SERVE_*`, read once at the first step; see
/// `docs/qwen3next-batching.md`, "Scheduler for B = 64"). Defaults depend on whether the engine has
/// wide prefill windows.
#[derive(Clone, Debug)]
pub struct Policy {
    /// Prompt tokens a step when nothing decodes (`TANG_SERVE_PREFILL_BUDGET`).
    pub full: usize,
    /// The smallest chunk while something decodes (`TANG_SERVE_PREFILL_SLICE`): a window's fixed
    /// cost is amortized over at least this many tokens.
    pub slice: usize,
    /// The largest chunk while something decodes (`TANG_SERVE_PREFILL_MAX_CHUNK`). Between
    /// `slice` and this, a chunk is as long as the decode credit banked buys.
    pub max_chunk: usize,
    /// Prefill's share of GPU time once decode rows are full (`TANG_SERVE_PREFILL_SHARE`).
    pub share: f64,
    /// Prefill's share while decode rows are below `fill` of the batch and prompts wait
    /// (`TANG_SERVE_PREFILL_FILL_SHARE`): admission fills decode rows before prefill is throttled.
    pub fill_share: f64,
    /// The fraction of the batch's rows counted as full (`TANG_SERVE_DECODE_FILL`).
    pub fill: f64,
    /// Priority a waiting prompt gains a second, in tokens of remaining prompt
    /// (`TANG_SERVE_AGE_TOKENS`): shortest-first with aging.
    pub age_tokens: f64,
    /// Seconds after which a prompt goes ahead of every younger one, oldest first
    /// (`TANG_SERVE_PREFILL_DEADLINE_S`).
    pub deadline: f64,
}

impl Policy {
    /// The defaults; `wide` is the engine's wide window width (0: none).
    pub fn defaults(wide: usize) -> Self {
        let (full, slice, max_chunk) = if wide > 0 {
            (2048.min(wide), 512.min(wide), 2048.min(wide))
        } else {
            (256, 64, 256)
        };
        Policy {
            full,
            slice,
            max_chunk,
            share: 0.5,
            fill_share: 0.85,
            fill: 0.8,
            age_tokens: 400.0,
            deadline: 20.0,
        }
    }

    /// [`defaults`](Self::defaults) with the `TANG_SERVE_*` overrides, clamped to sane ranges
    /// (chunks to the wide width when there is one).
    pub fn from_env(wide: usize) -> Self {
        Self::from_lookup(wide, |k| std::env::var(k).ok())
    }

    fn from_lookup(wide: usize, get: impl Fn(&str) -> Option<String>) -> Self {
        let mut p = Self::defaults(wide);
        let num = |k: &str| get(k).and_then(|v| v.trim().parse::<f64>().ok());
        let cap = if wide > 0 { wide } else { usize::MAX };
        if let Some(v) = num("TANG_SERVE_PREFILL_BUDGET") {
            p.full = (v as usize).clamp(1, cap);
        }
        if let Some(v) = num("TANG_SERVE_PREFILL_SLICE") {
            p.slice = (v as usize).clamp(1, cap);
        }
        if let Some(v) = num("TANG_SERVE_PREFILL_MAX_CHUNK") {
            p.max_chunk = (v as usize).clamp(1, cap);
        }
        p.max_chunk = p.max_chunk.max(p.slice);
        if let Some(v) = num("TANG_SERVE_PREFILL_SHARE") {
            p.share = v.clamp(0.05, 0.95);
        }
        if let Some(v) = num("TANG_SERVE_PREFILL_FILL_SHARE") {
            p.fill_share = v.clamp(0.05, 0.95);
        }
        if let Some(v) = num("TANG_SERVE_DECODE_FILL") {
            p.fill = v.clamp(0.0, 1.0);
        }
        if let Some(v) = num("TANG_SERVE_AGE_TOKENS") {
            p.age_tokens = v.max(0.0);
        }
        if let Some(v) = num("TANG_SERVE_PREFILL_DEADLINE_S") {
            p.deadline = if v > 0.0 { v } else { f64::INFINITY };
        }
        p
    }

    /// Prefill's GPU share with `rows` streams decoding in a batch of `batch`, prompts waiting.
    pub fn share_for(&self, rows: usize, batch: usize, prompts_waiting: bool) -> f64 {
        if prompts_waiting && (rows as f64) < self.fill * batch as f64 {
            self.fill_share
        } else {
            self.share
        }
    }
}

/// The scheduler's running account: `debt` is prefill seconds ahead of its share (negative:
/// decode credit banked, which buys a longer chunk), `rate` the measured prefill tokens a second.
#[derive(Clone, Debug)]
pub struct Sched {
    pub debt: f64,
    pub rate: f64,
}

impl Default for Sched {
    fn default() -> Self {
        // ~H200 wide prefill on Qwen3-Coder-Next (artifacts/gates-1104); measured from then on.
        Sched {
            debt: 0.0,
            rate: 1300.0,
        }
    }
}

impl Sched {
    /// Nothing decodes: no account to keep.
    pub fn idle(&mut self) {
        self.debt = 0.0;
    }

    /// A prefill chunk of `n` tokens took `dt` seconds.
    pub fn prefilled(&mut self, n: usize, dt: f64, decoding: bool) {
        if n >= 64 && dt > 0.0 {
            self.rate = 0.8 * self.rate + 0.2 * (n as f64 / dt);
        }
        if decoding {
            self.debt += dt;
        }
    }

    /// A decode window took `dt` seconds with prefill's share `share`: it pays off
    /// `dt * share / (1 - share)` of prefill, banking at most one `max_chunk` of credit.
    pub fn decoded(&mut self, dt: f64, share: f64, p: &Policy) {
        let cap = p.max_chunk as f64 / self.rate.max(1.0);
        self.debt = (self.debt - dt * share / (1.0 - share)).max(-cap);
    }

    /// This step's prompt tokens: the full budget when nothing decodes; none while prefill is
    /// ahead of its share; otherwise what the banked credit buys, between `slice` and
    /// `max_chunk`.
    pub fn budget(&self, p: &Policy, decoding: bool) -> usize {
        if !decoding {
            return p.full;
        }
        if self.debt > 0.0 {
            return 0;
        }
        let buys = (-self.debt * self.rate) as usize;
        buys.clamp(p.slice, p.max_chunk)
    }
}

/// A prompt still prefilling: lane `idx`, `rem` tokens to go, waiting `age` seconds.
#[derive(Clone, Copy, Debug)]
pub struct Waiting {
    pub idx: usize,
    pub rem: usize,
    pub age: f64,
}

/// The order prompts prefill in: past the deadline, oldest first; then shortest remaining minus
/// `age_tokens` a second waited (shortest-first with aging; a cached agent turn still jumps a
/// cold prompt, but a 6000-token prompt can't wait forever behind a stream of short ones).
pub fn prefill_order(waiting: &[Waiting], p: &Policy) -> Vec<usize> {
    let mut w = waiting.to_vec();
    let key = |x: &Waiting| -> (bool, f64) {
        if x.age >= p.deadline {
            (false, -x.age)
        } else {
            (true, x.rem as f64 - x.age * p.age_tokens)
        }
    };
    w.sort_by(|a, b| {
        key(a)
            .partial_cmp(&key(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.idx.cmp(&b.idx))
    });
    w.iter().map(|x| x.idx).collect()
}

/// This step's prefill chunks as `(lane, tokens)`: `budget` tokens in [`prefill_order`].
pub fn prefill_plan(waiting: &[Waiting], p: &Policy, budget: usize) -> Vec<(usize, usize)> {
    let mut budget = budget;
    let mut plan = Vec::new();
    for i in prefill_order(waiting, p) {
        if budget == 0 {
            break;
        }
        let rem = waiting.iter().find(|x| x.idx == i).unwrap().rem;
        let n = rem.min(budget);
        plan.push((i, n));
        budget -= n;
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old policy (before 2026-11-04) in the new terms: shortest first, no aging, a fixed
    /// 256-token wide slice, prefill held to half the GPU.
    fn old_wide() -> Policy {
        Policy {
            full: 2048,
            slice: 256,
            max_chunk: 256,
            share: 0.5,
            fill_share: 0.5,
            fill: 0.8,
            age_tokens: 0.0,
            deadline: f64::INFINITY,
        }
    }

    /// What one simulated load run gives.
    #[derive(Debug)]
    struct Run {
        agg: f64,
        ttft_p50: f64,
        ttft_p95: f64,
        stream_tps_p50: f64,
        rows: f64,
    }

    /// A closed-loop load on a simulated engine with `batch_step`'s accounting: `c` clients,
    /// each sending its next request when the last finishes (all cold at t = 0, like
    /// `artifacts/lt/load.py`), prompts 200–6000 tokens and 256–1024 out from a fixed LCG.
    /// Costs from the 2026-11-04 H200 logs: prefill `1 / 1350` s a token (wide windows), a
    /// decode window `10 ms + 0.53 ms` a row (44 ms at 64 rows).
    fn simulate(p: &Policy, c: usize, batch: usize, requests: usize) -> Run {
        const PF: f64 = 1.0 / 1350.0;
        let window = |rows: usize| 0.010 + 0.00053 * rows as f64;
        let mut seed = 0x2545_f491u64;
        let mut rnd = move |lo: usize, hi: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((seed >> 33) as usize) % (hi - lo + 1)
        };
        // Per request: (arrival, prompt left, out left, out total, first token time).
        struct R {
            t0: f64,
            rem: usize,
            out: usize,
            total: usize,
            first: Option<f64>,
            end: Option<f64>,
        }
        let mut reqs: Vec<R> = Vec::new();
        let mut live: Vec<usize> = Vec::new();
        let mut sent = 0usize;
        let (mut clock, mut sched, mut out_toks) = (0.0f64, Sched::default(), 0usize);
        let (mut windows, mut row_sum) = (0usize, 0usize);
        let mut new_req = |reqs: &mut Vec<R>, t: f64| {
            let total = rnd(256, 1024);
            reqs.push(R {
                t0: t,
                rem: rnd(200, 6000),
                out: total,
                total,
                first: None,
                end: None,
            });
            reqs.len() - 1
        };
        while sent < c.min(requests) {
            live.push(new_req(&mut reqs, 0.0));
            sent += 1;
        }
        // Admission: at most `batch` live at once (the server's queue holds the rest).
        while !live.is_empty() {
            let active: Vec<usize> = live.iter().copied().take(batch).collect();
            let decoding = active.iter().any(|&i| reqs[i].rem == 0);
            if !decoding {
                sched.idle();
            }
            let waiting: Vec<Waiting> = active
                .iter()
                .filter(|&&i| reqs[i].rem > 0)
                .map(|&i| Waiting {
                    idx: i,
                    rem: reqs[i].rem,
                    age: clock - reqs[i].t0,
                })
                .collect();
            let prompts_waiting = !waiting.is_empty();
            for (i, n) in prefill_plan(&waiting, p, sched.budget(p, decoding)) {
                let dt = 0.004 + n as f64 * PF;
                clock += dt;
                sched.prefilled(n, dt, decoding);
                reqs[i].rem -= n;
                if reqs[i].rem == 0 {
                    reqs[i].first = Some(clock);
                    reqs[i].out -= 1;
                    out_toks += 1;
                }
            }
            let rows: Vec<usize> = active
                .iter()
                .copied()
                .filter(|&i| {
                    reqs[i].rem == 0 && reqs[i].out > 0 && reqs[i].first.unwrap() < clock + 1e-12
                })
                .collect();
            if !rows.is_empty() {
                let dt = window(rows.len());
                clock += dt;
                sched.decoded(dt, p.share_for(rows.len(), batch, prompts_waiting), p);
                windows += 1;
                row_sum += rows.len();
                for &i in &rows {
                    reqs[i].out -= 1;
                    out_toks += 1;
                }
            }
            let mut k = 0;
            while k < live.len() {
                let i = live[k];
                if reqs[i].rem == 0 && reqs[i].out == 0 {
                    reqs[i].end = Some(clock);
                    live.remove(k);
                    if sent < requests {
                        live.push(new_req(&mut reqs, clock));
                        sent += 1;
                    }
                } else {
                    k += 1;
                }
            }
            assert!(clock < 1e5, "the load never finished");
        }
        let mut ttft: Vec<f64> = reqs.iter().map(|r| r.first.unwrap() - r.t0).collect();
        ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut tps: Vec<f64> = reqs
            .iter()
            .map(|r| (r.total - 1) as f64 / (r.end.unwrap() - r.first.unwrap()).max(1e-9))
            .collect();
        tps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |v: &[f64], f: f64| v[((v.len() - 1) as f64 * f) as usize];
        Run {
            agg: out_toks as f64 / clock,
            ttft_p50: q(&ttft, 0.5),
            ttft_p95: q(&ttft, 0.95),
            stream_tps_p50: q(&tps, 0.5),
            rows: row_sum as f64 / windows.max(1) as f64,
        }
    }

    /// The simulated C = 64 load (3 requests a client): the new policy gets more prompts
    /// through (decode rows fuller, more output a second) and cuts the TTFT tail that
    /// shortest-first left to 6000-token prompts. The ceiling is prefill speed: at 1,350
    /// prompt tok/s and ~3,100-token prompts, no policy beats ~280 output tok/s.
    #[test]
    fn c64_more_throughput_and_a_shorter_ttft_tail() {
        let old = simulate(&old_wide(), 64, 64, 192);
        let new = simulate(&Policy::defaults(2048), 64, 64, 192);
        eprintln!("old {old:?}\nnew {new:?}");
        assert!(
            new.agg > 1.15 * old.agg,
            "agg {:.0} vs {:.0}",
            new.agg,
            old.agg
        );
        assert!(
            new.ttft_p95 < 0.8 * old.ttft_p95,
            "p95 {:.0} vs {:.0}",
            new.ttft_p95,
            old.ttft_p95
        );
        assert!(
            new.rows > old.rows,
            "rows {:.1} vs {:.1}",
            new.rows,
            old.rows
        );
        assert!(
            new.agg < 300.0,
            "faster than prefill allows: {:.0}",
            new.agg
        );
    }

    /// C = 8 (3 a client): no regression in aggregate or per-stream speed.
    #[test]
    fn c8_no_regression() {
        let old = simulate(&old_wide(), 8, 64, 24);
        let new = simulate(&Policy::defaults(2048), 8, 64, 24);
        eprintln!("old {old:?}\nnew {new:?}");
        assert!(
            new.agg >= 0.97 * old.agg,
            "agg {:.0} vs {:.0}",
            new.agg,
            old.agg
        );
        assert!(
            new.ttft_p50 <= old.ttft_p50,
            "p50 {:.1} vs {:.1}",
            new.ttft_p50,
            old.ttft_p50
        );
        assert!(
            new.ttft_p95 <= 1.05 * old.ttft_p95,
            "p95 {:.1} vs {:.1}",
            new.ttft_p95,
            old.ttft_p95
        );
        assert!(
            new.stream_tps_p50 > 0.5 * old.stream_tps_p50,
            "{new:?} vs {old:?}"
        );
    }

    #[test]
    #[ignore]
    fn sweep() {
        for fs in [0.5, 0.6, 0.7, 0.75, 0.85] {
            for sl in [256, 512] {
                let mut p = Policy::defaults(2048);
                p.fill_share = fs;
                p.slice = sl;
                let a = simulate(&p, 8, 64, 24);
                let b = simulate(&p, 64, 64, 192);
                eprintln!("fs {fs} slice {sl}: C8 agg {:.0} p95 {:.1} tps {:.1} | C64 agg {:.0} p50 {:.0} p95 {:.0} tps {:.1} rows {:.1}",
                    a.agg, a.ttft_p95, a.stream_tps_p50, b.agg, b.ttft_p50, b.ttft_p95, b.stream_tps_p50, b.rows);
            }
        }
    }

    #[test]
    fn aging_and_the_deadline_lift_old_long_prompts() {
        let p = Policy::defaults(2048);
        let w = |idx, rem, age| Waiting { idx, rem, age };
        // Fresh: shortest first (a cached agent turn jumps a cold prompt).
        assert_eq!(
            prefill_order(&[w(0, 6000, 0.0), w(1, 27, 0.0)], &p),
            vec![1, 0]
        );
        // 6000 tokens waiting 15 s (key 0) beats a fresh 200-token prompt.
        assert_eq!(
            prefill_order(&[w(0, 6000, 15.0), w(1, 200, 0.0)], &p),
            vec![0, 1]
        );
        // Past the deadline: ahead of everything, oldest first.
        assert_eq!(
            prefill_order(&[w(0, 27, 0.0), w(1, 6000, 21.0), w(2, 6000, 30.0)], &p),
            vec![2, 1, 0]
        );
        // Without aging (TANG_SERVE_AGE_TOKENS=0, no deadline) it's plain shortest-first.
        let mut q = p.clone();
        q.age_tokens = 0.0;
        q.deadline = f64::INFINITY;
        assert_eq!(
            prefill_order(&[w(0, 6000, 300.0), w(1, 200, 0.0)], &q),
            vec![1, 0]
        );
    }

    #[test]
    fn chunks_scale_with_banked_decode_credit() {
        let p = Policy::defaults(2048);
        let mut s = Sched::default();
        assert_eq!(
            s.budget(&p, false),
            2048,
            "nothing decoding: the full budget"
        );
        assert_eq!(s.budget(&p, true), 512, "no credit: the minimum slice");
        s.prefilled(512, 0.4, true);
        assert_eq!(s.budget(&p, true), 0, "prefill ahead of its share waits");
        // Five 40 ms windows at the fill share (0.85) pay 1.13 s: 0.4 s debt, 0.73 s banked,
        // which buys ~950 tokens.
        for _ in 0..5 {
            s.decoded(0.040, p.fill_share, &p);
        }
        let b = s.budget(&p, true);
        assert!((512..=2048).contains(&b) && b > 512, "{b}");
        // Credit is capped at one max chunk.
        for _ in 0..1000 {
            s.decoded(0.040, 0.95, &p);
        }
        assert_eq!(s.budget(&p, true), 2048);
        s.idle();
        assert_eq!(s.debt, 0.0);
    }

    #[test]
    fn decode_rows_fill_before_prefill_is_throttled() {
        let p = Policy::defaults(2048);
        assert_eq!(p.share_for(10, 64, true), p.fill_share);
        assert_eq!(p.share_for(52, 64, true), p.share);
        assert_eq!(p.share_for(10, 64, false), p.share);
    }

    #[test]
    fn knobs_come_from_the_environment() {
        let env = |k: &str| match k {
            "TANG_SERVE_PREFILL_SLICE" => Some("1024".into()),
            "TANG_SERVE_PREFILL_MAX_CHUNK" => Some("99999".into()),
            "TANG_SERVE_PREFILL_SHARE" => Some("0.7".into()),
            "TANG_SERVE_PREFILL_FILL_SHARE" => Some("2".into()),
            "TANG_SERVE_PREFILL_DEADLINE_S" => Some("0".into()),
            "TANG_SERVE_AGE_TOKENS" => Some("junk".into()),
            _ => None,
        };
        let p = Policy::from_lookup(2048, env);
        assert_eq!((p.slice, p.max_chunk), (1024, 2048));
        assert_eq!((p.share, p.fill_share), (0.7, 0.95));
        assert!(p.deadline.is_infinite());
        assert_eq!(p.age_tokens, 400.0);
        let n = Policy::from_lookup(0, |_| None);
        assert_eq!((n.full, n.slice, n.max_chunk), (256, 64, 256));
    }
}
