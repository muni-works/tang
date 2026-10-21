//! Tree speculative decoding for the Coder engine: a *forest of chains* sharing the root.
//!
//! A verify window holds chains `[cur, d1, d2, …]` back to back (at most `MAX_T` rows). Every
//! chain repeats `cur`, so each chain is one GDN segment reading the committed state, and rows of
//! chain b keep their window keys at cache offset `off_b` so chains at the same depth don't
//! overwrite each other's K/V. See `docs/plans/2026-10-21-qwen3next-tree-spec.md`.
//!
//! This file is host-side only: the window layout, the accept rule and the shape choice.

/// Rows per window (the Coder engine's `MAX_T`; a test there checks they agree).
pub const MAX_T: usize = 8;

/// One row of a forest window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub tok: u32,
    /// Position (rope and causal extent): `pos0 + depth`.
    pub pos: usize,
    /// Cache offset for this row's window keys past the root (0 for chain 0).
    pub off: usize,
}

/// A forest window: chains laid out back to back.
#[derive(Clone, Debug, Default)]
pub struct Forest {
    /// Drafts per chain (without the root).
    pub chains: Vec<Vec<u32>>,
}

/// A GDN segment for one chain: (first row, rows).
pub type Seg = (usize, usize);

impl Forest {
    /// Drop empty and duplicate-first-token chains, then trim from the last chain until the
    /// rows fit `MAX_T`. Chain 0 is kept whole when it fits.
    pub fn new(mut chains: Vec<Vec<u32>>) -> Self {
        let mut seen = Vec::new();
        chains.retain(|c| {
            let keep = !c.is_empty() && !seen.contains(&c[0]);
            if keep {
                seen.push(c[0]);
            }
            keep
        });
        if chains.is_empty() {
            return Self { chains };
        }
        chains[0].truncate(MAX_T - 1);
        let mut used = 1 + chains[0].len();
        let mut out = vec![chains[0].clone()];
        for c in chains.into_iter().skip(1) {
            let room = MAX_T.saturating_sub(used + 1);
            if room == 0 {
                break;
            }
            let c: Vec<u32> = c.into_iter().take(room).collect();
            used += 1 + c.len();
            out.push(c);
        }
        Self { chains: out }
    }

    pub fn rows_len(&self) -> usize {
        if self.chains.is_empty() {
            1
        } else {
            self.chains.iter().map(|c| 1 + c.len()).sum()
        }
    }

    /// The window rows for root `cur` at `pos0`, and one segment per chain.
    pub fn layout(&self, cur: u32, pos0: usize) -> (Vec<Row>, Vec<Seg>) {
        if self.chains.is_empty() {
            let r = Row { tok: cur, pos: pos0, off: 0 };
            return (vec![r], vec![(0, 1)]);
        }
        let (mut rows, mut segs, mut off) = (Vec::new(), Vec::new(), 0);
        for c in &self.chains {
            segs.push((rows.len(), 1 + c.len()));
            rows.push(Row { tok: cur, pos: pos0, off: 0 });
            for (i, &t) in c.iter().enumerate() {
                rows.push(Row { tok: t, pos: pos0 + 1 + i, off });
            }
            off += c.len();
        }
        (rows, segs)
    }

    /// Given each row's target (argmax), the winning chain and its accepted draft count `m`;
    /// ties go to the lower chain. The tokens to emit are the winner's targets `0..=m`.
    pub fn accept(&self, targets: &[u32]) -> (usize, usize) {
        let (_, segs) = self.layout(0, 0);
        let mut best = (0, 0);
        for (b, &(r0, _)) in segs.iter().enumerate() {
            let Some(c) = self.chains.get(b) else { break };
            let m = c
                .iter()
                .enumerate()
                .take_while(|&(i, &d)| targets[r0 + i] == d)
                .count();
            if m > best.1 {
                best = (b, m);
            }
        }
        best
    }

    /// The emitted tokens for an accept result.
    pub fn emitted(&self, targets: &[u32], (b, m): (usize, usize)) -> Vec<u32> {
        let (_, segs) = self.layout(0, 0);
        let r0 = segs[b].0;
        targets[r0..=r0 + m].to_vec()
    }

    /// The cache offset of chain `b`'s window keys (what the accepted K/V move back from).
    pub fn off(&self, b: usize) -> usize {
        self.chains[..b].iter().map(|c| c.len()).sum()
    }
}

/// Expected tokens per step for `forest` with per-chain, per-depth acceptance estimates
/// (`p[b][i]`: probability draft i of chain b is accepted given its prefix was; chains'
/// first tokens are disjoint, so first-token events are exclusive).
pub fn expected(p: &[Vec<f32>]) -> f32 {
    1.0 + p
        .iter()
        .map(|c| {
            let (mut cum, mut s) = (1.0f32, 0.0f32);
            for &a in c {
                cum *= a;
                s += cum;
            }
            s
        })
        .sum::<f32>()
}

/// Pick per-chain lengths maximising expected tokens per unit cost, `cost[rows]` relative to a
/// T=1 step. `p` as in [`expected`], chains in priority order. Returns the kept length per
/// chain (0 drops the chain; later chains drop first).
pub fn choose(p: &[Vec<f32>], cost: &[f32]) -> Vec<usize> {
    let mut best = (vec![0; p.len()], 1.0 / cost[1]);
    let mut lens = vec![0usize; p.len()];
    fn rec(
        b: usize,
        used: usize,
        p: &[Vec<f32>],
        cost: &[f32],
        lens: &mut Vec<usize>,
        best: &mut (Vec<usize>, f32),
    ) {
        if b == p.len() {
            let rows = used.max(1);
            if rows < cost.len() {
                let q: Vec<Vec<f32>> = p.iter().zip(lens.iter()).map(|(c, &l)| c[..l].to_vec()).collect();
                let r = expected(&q) / cost[rows];
                if r > best.1 + 1e-6 {
                    *best = (lens.clone(), r);
                }
            }
            return;
        }
        // A chain after a dropped chain is pointless (priority order).
        let can = b == 0 || lens[b - 1] > 0;
        lens[b] = 0;
        rec(b + 1, used, p, cost, lens, best);
        if !can {
            return;
        }
        let base = if b == 0 { 1 } else { used + 1 };
        for l in 1..=p[b].len() {
            if base + l > MAX_T || base + l >= cost.len() {
                break;
            }
            lens[b] = l;
            rec(b + 1, base + l, p, cost, lens, best);
        }
        lens[b] = 0;
    }
    rec(0, 1, p, cost, &mut lens, &mut best);
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_offsets_and_segments() {
        let f = Forest::new(vec![vec![10, 11], vec![20, 21], vec![30]]);
        let (rows, segs) = f.layout(5, 100);
        assert_eq!(segs, vec![(0, 3), (3, 3), (6, 2)]);
        let pos: Vec<usize> = rows.iter().map(|r| r.pos).collect();
        assert_eq!(pos, vec![100, 101, 102, 100, 101, 102, 100, 101]);
        let off: Vec<usize> = rows.iter().map(|r| r.off).collect();
        assert_eq!(off, vec![0, 0, 0, 0, 2, 2, 0, 4]);
        // Window keys never collide: cache index pos + off is unique past the root.
        let mut idx: Vec<usize> = rows.iter().filter(|r| r.pos > 100).map(|r| r.pos + r.off).collect();
        idx.sort();
        idx.dedup();
        assert_eq!(idx.len(), 5);
        assert_eq!(f.off(2), 4);
    }

    #[test]
    fn new_dedupes_and_fits_max_t() {
        let f = Forest::new(vec![vec![1, 2, 3, 4, 5], vec![1, 9], vec![], vec![7, 8, 9, 10]]);
        assert_eq!(f.chains, vec![vec![1, 2, 3, 4, 5], vec![7]]);
        assert!(f.rows_len() <= MAX_T);
    }

    #[test]
    fn accept_picks_longest_then_lowest() {
        let f = Forest::new(vec![vec![10, 11], vec![20, 21, 22]]);
        // Targets per row: chain 0 rows 0..3, chain 1 rows 3..7.
        let t = [10, 99, 0, 20, 21, 50, 0];
        // Both chains can't be right at depth 1 in reality; accept just measures each.
        assert_eq!(f.accept(&t), (1, 2));
        assert_eq!(f.emitted(&t, (1, 2)), vec![20, 21, 50]);
        assert_eq!(f.accept(&[1, 2, 3, 4, 5, 6, 7]), (0, 0));
    }

    /// A toy deterministic "model": the next token is a hash of the whole prefix. Decoding with
    /// forests of random chains (one sometimes right) must equal plain greedy.
    #[test]
    fn forest_decode_equals_greedy() {
        fn next(prefix: &[u32]) -> u32 {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for &t in prefix {
                h = (h ^ t as u64).wrapping_mul(0x100_0000_01b3);
            }
            (h % 7) as u32
        }
        let prompt = vec![1u32, 2, 3];
        let n = 200;
        let mut greedy = prompt.clone();
        while greedy.len() < prompt.len() + n {
            greedy.push(next(&greedy));
        }
        let mut seq = prompt.clone();
        let mut rng = 12345u64;
        let mut rand = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut steps = 0;
        while seq.len() < prompt.len() + n {
            steps += 1;
            // Chains: maybe the oracle's next few tokens, mixed with junk, in a random slot.
            let k = 1 + (rand() % 3) as usize;
            let mut chains: Vec<Vec<u32>> = (0..k)
                .map(|_| (0..1 + rand() % 4).map(|_| (rand() % 7) as u32).collect())
                .collect();
            if rand() % 2 == 0 {
                let at = seq.len();
                let len = 1 + (rand() % 4) as usize;
                let good: Vec<u32> = greedy[at..(at + len).min(greedy.len())].to_vec();
                let slot = (rand() as usize) % (k + 1);
                chains.insert(slot.min(chains.len()), good);
            }
            let f = Forest::new(chains);
            // The window: each row's target from its own causal prefix (context + chain).
            let cur = seq[seq.len() - 1];
            let (rows, segs) = f.layout(cur, seq.len() - 1);
            let mut targets = vec![0; rows.len()];
            for (b, &(r0, len)) in segs.iter().enumerate() {
                let mut ctx = seq.clone();
                for i in 0..len {
                    if i > 0 {
                        ctx.push(f.chains[b][i - 1]);
                    }
                    targets[r0 + i] = next(&ctx);
                }
            }
            let acc = f.accept(&targets);
            seq.extend(f.emitted(&targets, acc));
        }
        seq.truncate(prompt.len() + n);
        assert_eq!(seq, greedy);
        assert!(steps < n, "no speedup at all: {steps} steps");
    }

    #[test]
    fn choose_prefers_branches_when_cheap_and_chain_when_good() {
        // Steep cost (today): short single chain.
        let steep: Vec<f32> = (0..=MAX_T).map(|t| 1.0 + 0.33 * (t.max(1) - 1) as f32).collect();
        let p = vec![vec![0.35, 0.6, 0.6, 0.6], vec![0.15, 0.6]];
        let c = choose(&p, &steep);
        assert_eq!(c[1], 0, "{c:?}");
        // Flat cost: the second chain earns its rows.
        let flat: Vec<f32> = (0..=MAX_T).map(|t| 1.0 + 0.05 * (t.max(1) - 1) as f32).collect();
        let c = choose(&p, &flat);
        assert!(c[0] >= 3 && c[1] >= 1, "{c:?}");
        // Worthless drafts: none.
        assert_eq!(choose(&[vec![0.01]], &steep), vec![0]);
    }
}
