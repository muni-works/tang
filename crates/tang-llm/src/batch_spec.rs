//! Speculative decoding under batching (`flash serve --batch N` with an MTP head): which draft
//! rows a batch window verifies, and how many. Pure, so it's tested without a model.
//!
//! - **D-Cut ranking** (arXiv 2607.14647). Every greedy stream proposes its MTP chain with the
//!   head's probabilities. Position `k` of stream `i` scores `s_ik = c_i1 · … · c_ik` (the
//!   chance the whole prefix is accepted, as far as the head knows). All positions across the
//!   batch are ranked by score and the top `budget` are verified on top of the one row each
//!   stream always gets; scores fall along a chain, so each stream keeps a prefix.
//! - **Budget from a measured table** (TableTune-style, arXiv 2610.38968). A step of `live`
//!   streams and `extra` draft rows costs `C(live, extra, mtp)` ms, measured as the server runs
//!   (EMA per cell; a least-squares fit over every step, then a prior, for cells not yet seen).
//!   The expected tokens of a budget are `live + α · Σ top-budget scores`, `α` the live ratio of
//!   accepted drafts to the scores' prediction. The budget maximising tokens / ms wins, among
//!   `{0} ∪ {⌈ρ · live · k⌉ : ρ ∈ ¼, ½, ¾, 1}` capped by the window's spare rows. "Plain" (budget
//!   0 and no MTP pass at all: today's batched path) competes too; while it wins, the MTP head
//!   is run one step in `PROBE` to keep the confidence estimate fresh.

use std::collections::HashMap;

/// Consecutive runs of `lens` whose sums stay within `cap` (each run at least one item):
/// `(lo, hi)` index ranges, in order.
pub fn pack_rows(lens: &[usize], cap: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut lo = 0;
    while lo < lens.len() {
        let mut hi = lo;
        let mut rows = 0;
        while hi < lens.len() && (hi == lo || rows + lens[hi] <= cap) {
            rows += lens[hi];
            hi += 1;
        }
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// Cumulative-product scores of a chain's probabilities.
pub fn scores(conf: &[f32]) -> Vec<f64> {
    let mut p = 1.0f64;
    conf.iter()
        .map(|&c| {
            p *= f64::from(c.clamp(0.0, 1.0));
            p
        })
        .collect()
}

/// D-Cut: draft rows per stream for chains with probabilities `conf[i]`, verifying the
/// `budget` best positions across the batch. Ties go to the shallower position, then the
/// earlier stream, so every stream's pick is a prefix of its chain. Also returns the picked
/// scores' sum (the expected accepted drafts, before calibration).
pub fn dcut(conf: &[Vec<f32>], budget: usize) -> (Vec<usize>, f64) {
    let mut all: Vec<(f64, usize, usize)> = conf
        .iter()
        .enumerate()
        .flat_map(|(i, c)| scores(c).into_iter().enumerate().map(move |(k, s)| (s, k, i)))
        .collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut give = vec![0; conf.len()];
    let mut sum = 0.0;
    for &(s, _, i) in all.iter().take(budget) {
        give[i] += 1;
        sum += s;
    }
    (give, sum)
}

/// What the tuner may do (`TANG_SERVE_SPEC`: `off` (default), `auto`, or a fixed budget).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// No drafts and no MTP pass after batch windows: the plain batched path.
    Off,
    /// Always this many extra rows (capped by the window), MTP on.
    Fixed(usize),
    /// The table's pick.
    Auto,
}

impl Mode {
    pub fn from_env() -> Self {
        match std::env::var("TANG_SERVE_SPEC").ok().as_deref() {
            // Off by default: on the 2026-11-04 serve load test drafting under batching didn't
            // beat plain batch windows at any concurrency (docs/qwen3next-batching.md).
            Some("off") | Some("0") | None | Some("") => Mode::Off,
            Some("auto") => Mode::Auto,
            Some(n) => n.parse().map_or(Mode::Auto, Mode::Fixed),
        }
    }
}

/// One step's decision.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// Draft rows per stream this step.
    pub drafts: Vec<usize>,
    /// The predicted accepted drafts (calibrated), and the picked scores' sum (raw).
    pub predicted: f64,
    pub raw: f64,
    /// Run the MTP head after this window (chains for the next step).
    pub mtp: bool,
}

/// Steps between MTP probes while plain decoding wins.
pub const PROBE: u64 = 64;
const RATIOS: [f64; 4] = [0.25, 0.5, 0.75, 1.0];
/// Cell width (streams, extra rows).
const BUCKET: usize = 4;

#[derive(Clone, Copy, Default, Debug)]
struct Ema {
    ms: f64,
    n: u32,
}

/// The measured step-cost table and acceptance calibration.
#[derive(Debug)]
pub struct Tuner {
    pub mode: Mode,
    cells: HashMap<(usize, usize, bool), Ema>,
    /// Decayed normal equations of `ms ≈ a + b·rows + c·mtp_rows` over every step.
    xtx: [[f64; 3]; 3],
    xty: [f64; 3],
    /// Accepted drafts per predicted (EMA).
    pub alpha: f64,
    /// Mean score by chain depth (EMA), for when no chains are at hand.
    depth: Vec<f64>,
    steps: u64,
    /// The last step ran the MTP head (its chains are fresh).
    pub mtp_live: bool,
}

impl Tuner {
    pub fn new(mode: Mode) -> Self {
        Tuner {
            mode,
            cells: HashMap::new(),
            xtx: [[0.0; 3]; 3],
            xty: [0.0; 3],
            alpha: 1.0,
            depth: Vec::new(),
            steps: 0,
            mtp_live: false,
        }
    }

    fn key(live: usize, extra: usize, mtp: bool) -> (usize, usize, bool) {
        (live.div_ceil(BUCKET), extra.div_ceil(BUCKET), mtp)
    }

    /// Estimated ms of a step: the cell, else the fit, else a prior (H200 batch windows: 10 ms
    /// plus 0.53 ms a row; an MTP pass ~0.15 ms a row).
    pub fn cost(&self, live: usize, extra: usize, mtp: bool) -> f64 {
        if let Some(c) = self.cells.get(&Self::key(live, extra, mtp)).filter(|c| c.n >= 2) {
            return c.ms;
        }
        let x = [1.0, (live + extra) as f64, if mtp { (live + extra) as f64 } else { 0.0 }];
        if let Some(b) = solve3(self.xtx, self.xty) {
            let v = b[0] * x[0] + b[1] * x[1] + b[2] * x[2];
            if v.is_finite() && v > 0.0 {
                return v;
            }
        }
        10.0 + 0.53 * x[1] + 0.15 * x[2]
    }

    /// A step ran: `live` streams, `extra` draft rows, MTP or not, `ms` long; `predicted`
    /// accepted drafts (uncalibrated) and `accepted` actually.
    pub fn observe(&mut self, live: usize, extra: usize, mtp: bool, ms: f64, raw: f64, accepted: usize) {
        self.steps += 1;
        let c = self.cells.entry(Self::key(live, extra, mtp)).or_default();
        c.ms = if c.n == 0 { ms } else { 0.8 * c.ms + 0.2 * ms };
        c.n = c.n.saturating_add(1);
        let rows = (live + extra) as f64;
        let x = [1.0, rows, if mtp { rows } else { 0.0 }];
        for i in 0..3 {
            self.xty[i] = 0.995 * self.xty[i] + x[i] * ms;
            for j in 0..3 {
                self.xtx[i][j] = 0.995 * self.xtx[i][j] + x[i] * x[j];
            }
        }
        if raw > 0.0 {
            let r = (accepted as f64 / raw).clamp(0.0, 2.0);
            self.alpha = 0.9 * self.alpha + 0.1 * r;
        }
        self.mtp_live = mtp;
    }

    /// Fold the chains' scores into the per-depth means.
    fn learn_depth(&mut self, conf: &[Vec<f32>]) {
        for c in conf.iter().filter(|c| !c.is_empty()) {
            for (k, s) in scores(c).into_iter().enumerate() {
                if self.depth.len() <= k {
                    self.depth.push(s);
                } else {
                    self.depth[k] = 0.95 * self.depth[k] + 0.05 * s;
                }
            }
        }
    }

    /// This step's drafts for `live` streams with chains `conf` (empty for a stream that
    /// can't draft: sampled, or no fresh chain) and `room` spare rows in the window.
    pub fn choose(&mut self, conf: &[Vec<f32>], room: usize) -> Choice {
        let live = conf.len();
        let none = |mtp| Choice {
            drafts: vec![0; live],
            predicted: 0.0,
            raw: 0.0,
            mtp,
        };
        match self.mode {
            Mode::Off => return none(false),
            Mode::Fixed(k) => {
                let (d, s) = dcut(conf, k.min(room));
                return Choice {
                    drafts: d,
                    predicted: s * self.alpha,
                    raw: s,
                    mtp: true,
                };
            }
            Mode::Auto => {}
        }
        self.learn_depth(conf);
        let k = conf.iter().map(Vec::len).max().unwrap_or(0).max(self.depth.len());
        // With no fresh chains, judge from the typical scores (to decide whether to probe).
        let fresh = conf.iter().any(|c| !c.is_empty());
        let typical: Vec<Vec<f32>>;
        let est = if fresh {
            conf
        } else {
            let c: Vec<f32> = self
                .depth
                .iter()
                .scan(1.0f64, |prev, &s| {
                    let r = if *prev > 0.0 { s / *prev } else { 0.0 };
                    *prev = s;
                    Some(r as f32)
                })
                .collect();
            typical = vec![c; live];
            &typical
        };
        let plain = live as f64 / self.cost(live, 0, false);
        let mut best = (plain, 0usize, false);
        let mut cands: Vec<usize> = RATIOS
            .iter()
            .map(|r| ((r * (live * k) as f64).ceil() as usize).min(room))
            .collect();
        cands.push(0);
        cands.sort_unstable();
        cands.dedup();
        for &b in &cands {
            let (_, s) = dcut(est, b);
            let tps = (live as f64 + self.alpha * s) / self.cost(live, b, true);
            if tps > best.0 {
                best = (tps, b, true);
            }
        }
        let (_, b, mtp) = best;
        if !fresh {
            // Nothing to verify this step; run the head if drafting would pay, or to probe.
            let probe = self.depth.is_empty() || self.steps % PROBE == 0;
            return none(mtp || probe);
        }
        // Plain steps are only measured while plain decoding runs: while drafting wins, run
        // one plain step now and then (often until its cell has samples), so a wrong guess of
        // its cost (the prior, or a fit from drafting steps only) can't keep drafting going.
        let plain_n = self.cells.get(&Self::key(live, 0, false)).map_or(0, |c| c.n);
        let every = if plain_n < 2 { PROBE / 8 } else { PROBE };
        if b > 0 && self.steps % every == every / 2 {
            return none(false);
        }
        let probe = self.steps % PROBE == 0;
        // Acceptance is only measured while drafting: now and then, verify one draft a stream
        // anyway, so a low `alpha` (an earlier, unlucky load) can recover.
        let b = if b == 0 && probe { live.min(room) } else { b };
        if b == 0 {
            return none(mtp);
        }
        let (d, s) = dcut(conf, b);
        Choice {
            drafts: d,
            predicted: s * self.alpha,
            raw: s,
            mtp: true,
        }
    }
}

/// `a x = y` for 3×3 `a` (Gaussian elimination with pivoting); `None` when singular.
fn solve3(mut a: [[f64; 3]; 3], mut y: [f64; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-9 {
            return None;
        }
        a.swap(c, p);
        y.swap(c, p);
        for r in 0..3 {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in 0..3 {
                    a[r][k] -= f * a[c][k];
                }
                y[r] -= f * y[c];
            }
        }
    }
    Some([y[0] / a[0][0], y[1] / a[1][1], y[2] / a[2][2]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_consecutive_runs_under_the_cap() {
        assert_eq!(pack_rows(&[3, 3, 3], 8), vec![(0, 2), (2, 3)]);
        assert_eq!(pack_rows(&[8, 1, 7, 2], 8), vec![(0, 1), (1, 3), (3, 4)]);
        assert_eq!(pack_rows(&[], 8), vec![]);
        // An item over the cap still gets a run of its own.
        assert_eq!(pack_rows(&[9, 1], 8), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn dcut_takes_the_best_positions_across_the_batch_as_prefixes() {
        let conf = vec![vec![0.9, 0.9], vec![0.5, 0.99], vec![], vec![0.95, 0.1]];
        // Scores: a .9 .81 | b .5 .495 | d .95 .095.
        assert_eq!(dcut(&conf, 0).0, vec![0, 0, 0, 0]);
        assert_eq!(dcut(&conf, 1).0, vec![0, 0, 0, 1]);
        assert_eq!(dcut(&conf, 3).0, vec![2, 0, 0, 1]);
        assert_eq!(dcut(&conf, 5).0, vec![2, 2, 0, 1]);
        assert_eq!(dcut(&conf, 100).0, vec![2, 2, 0, 2]);
        let (_, s) = dcut(&conf, 3);
        assert!((s - (0.95 + 0.9 + 0.81)).abs() < 1e-6);
        // Equal scores: shallower first, then the earlier stream, so picks stay prefixes.
        let flat = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(dcut(&flat, 1).0, vec![1, 0]);
        assert_eq!(dcut(&flat, 3).0, vec![2, 1]);
    }

    #[test]
    fn off_and_fixed_modes() {
        let conf = vec![vec![0.9, 0.8]; 4];
        let mut t = Tuner::new(Mode::Off);
        let c = t.choose(&conf, 60);
        assert_eq!(c.drafts, vec![0; 4]);
        assert!(!c.mtp);
        let mut t = Tuner::new(Mode::Fixed(3));
        let c = t.choose(&conf, 60);
        assert_eq!(c.drafts.iter().sum::<usize>(), 3);
        assert!(c.mtp);
        // Capped by the window's room.
        let c = t.choose(&conf, 2);
        assert_eq!(c.drafts.iter().sum::<usize>(), 2);
    }

    /// A simulated engine: a step costs 10 ms + 0.53 ms a row (+0.15 ms a row with MTP), each
    /// draft accepted with its probability. The tuner learns the table and should draft at
    /// light load and stop at full load.
    #[test]
    fn auto_drafts_at_light_load_and_stops_when_rows_cost_more() {
        let cost = |live: usize, extra: usize, mtp: bool| {
            let rows = (live + extra) as f64;
            10.0 + 0.53 * rows + if mtp { 0.15 * rows } else { 0.0 }
        };
        for (live, room, drafts) in [(8, 56, true), (64, 0, false)] {
            let mut t = Tuner::new(Mode::Auto);
            let conf = vec![vec![0.8f32, 0.7]; live];
            let mut last = Choice {
                drafts: vec![],
                predicted: 0.0,
                raw: 0.0,
                mtp: false,
            };
            for _ in 0..200 {
                let c = if t.mtp_live { conf.clone() } else { vec![vec![]; live] };
                last = t.choose(&c, room);
                let extra: usize = last.drafts.iter().sum();
                let (_, raw) = dcut(&c, extra);
                t.observe(live, extra, last.mtp, cost(live, extra, last.mtp), raw, raw.round() as usize);
            }
            assert_eq!(last.drafts.iter().sum::<usize>() > 0, drafts, "live {live}: {last:?}");
        }
    }

    #[test]
    fn auto_stops_drafting_when_drafts_are_rejected() {
        let mut t = Tuner::new(Mode::Auto);
        let conf = vec![vec![0.9f32, 0.9]; 8];
        let mut extra = 0;
        for _ in 0..299 {
            let c = if t.mtp_live { conf.clone() } else { vec![vec![]; 8] };
            let ch = t.choose(&c, 56);
            extra = ch.drafts.iter().sum();
            let (_, raw) = dcut(&c, extra);
            // The head is overconfident: nothing is ever accepted.
            let rows = (8 + extra) as f64;
            t.observe(8, extra, ch.mtp, 10.0 + 0.53 * rows + if ch.mtp { 0.15 * rows } else { 0.0 }, raw, 0);
        }
        assert_eq!(extra, 0);
        assert!(t.alpha < 0.5, "{}", t.alpha);
    }

    /// Plain steps are cheaper than the prior says (and drafting steps carry an expensive MTP
    /// pass): the tuner has to measure plain decoding to find it wins, then stay there.
    #[test]
    fn auto_measures_plain_steps_and_keeps_them_when_cheaper() {
        let mut t = Tuner::new(Mode::Auto);
        let live = 8;
        let conf = vec![vec![0.95f32, 0.95]; live];
        let cost = |extra: usize, mtp: bool| {
            if mtp {
                12.0 + 0.6 * (live + extra) as f64
            } else {
                4.0 + 0.3 * live as f64
            }
        };
        let mut plain = 0;
        for step in 0..400 {
            let c = if t.mtp_live { conf.clone() } else { vec![vec![]; live] };
            let ch = t.choose(&c, 24);
            let extra: usize = ch.drafts.iter().sum();
            let (_, raw) = dcut(&c, extra);
            t.observe(live, extra, ch.mtp, cost(extra, ch.mtp), raw, raw.round() as usize);
            if step >= 300 && !ch.mtp {
                plain += 1;
            }
        }
        assert!(plain > 90, "plain steps in the last 100: {plain}");
    }

    #[test]
    fn the_fit_recovers_a_linear_cost() {
        let mut t = Tuner::new(Mode::Auto);
        for live in [9, 17, 30, 41, 64] {
            for extra in [0, 5, 13] {
                for mtp in [false, true] {
                    let rows = (live + extra) as f64;
                    t.observe(live, extra, mtp, 7.0 + 0.4 * rows + if mtp { 0.1 * rows } else { 0.0 }, 0.0, 0);
                }
            }
        }
        // An unseen cell comes from the fit.
        assert!((t.cost(50, 22, true) - (7.0 + 0.5 * 72.0)).abs() < 0.5);
    }
}
