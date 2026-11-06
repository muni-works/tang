//! Drafts for speculative decoding, SuffixDecoding-style: no draft model, just suffix matches
//! against text the model has already seen or written.
//!
//! Two token-level suffix automata answer "where did the last few tokens appear before, and
//! what came next most often?":
//! - one per request, over the prompt plus what's been generated so far (an agent's replies
//!   copy file contents, earlier edits and tool output from the prompt);
//! - a bounded global one over earlier completions from this server (optionally kept on disk
//!   across restarts), since agents repeat themselves across requests and sessions.
//!
//! A draft follows the most frequent continuation of the longest match. Its length adapts: each
//! drafted token gets an acceptance estimate (the continuation's frequency times a per-match-
//! length hit rate learned online), and the draft is cut where expected tokens per unit of
//! verify cost peaks, using the backend's measured cost of a k-token forward.
//!
//! Drafting never changes what's generated: the engine samples every position from the target
//! model and only keeps draft tokens that equal what it sampled.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Separates sequences in the global store; never drafted.
const SEP: u32 = u32::MAX;

/// Walk at most this far up suffix links when counting occurrences (bounds pathological runs
/// like a long stretch of one repeated token; counts near the root just saturate).
const COUNT_WALK: usize = 1024;

/// A fast hasher for the (state, token) transition table (splitmix64's finalizer, so both
/// halves of the key reach the low bits the table indexes by).
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = self.0.rotate_left(5) ^ n;
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// An edge in a state's child list.
#[derive(Clone, Copy)]
struct Edge {
    tok: u32,
    to: u32,
    /// Next edge of the same state (`u32::MAX`: none).
    next: u32,
}

const NONE: u32 = u32::MAX;

/// A suffix automaton over token ids with occurrence counts (how many times each state's
/// strings occur), so it gives both the longest earlier match and continuation frequencies.
pub struct Sam {
    len: Vec<u32>,
    link: Vec<u32>,
    cnt: Vec<u32>,
    first: Vec<u32>,
    edges: Vec<Edge>,
    /// (state << 32 | token) -> edge index.
    index: FxMap<u64, u32>,
    last: u32,
    /// Tokens added.
    pub size: usize,
}

impl Default for Sam {
    fn default() -> Self {
        Self::new()
    }
}

impl Sam {
    pub fn new() -> Self {
        Self {
            len: vec![0],
            link: vec![NONE],
            cnt: vec![0],
            first: vec![NONE],
            edges: Vec::new(),
            index: FxMap::default(),
            last: 0,
            size: 0,
        }
    }

    pub fn with_capacity(tokens: usize) -> Self {
        let mut s = Self::new();
        s.len.reserve(2 * tokens);
        s.link.reserve(2 * tokens);
        s.cnt.reserve(2 * tokens);
        s.first.reserve(2 * tokens);
        s.edges.reserve(3 * tokens);
        s.index.reserve(3 * tokens);
        s
    }

    fn key(s: u32, c: u32) -> u64 {
        (s as u64) << 32 | c as u64
    }

    fn go(&self, s: u32, c: u32) -> Option<u32> {
        self.index
            .get(&Self::key(s, c))
            .map(|&e| self.edges[e as usize].to)
    }

    fn set(&mut self, s: u32, c: u32, to: u32) {
        match self.index.get(&Self::key(s, c)) {
            Some(&e) => self.edges[e as usize].to = to,
            None => {
                let e = self.edges.len() as u32;
                self.edges.push(Edge {
                    tok: c,
                    to,
                    next: self.first[s as usize],
                });
                self.first[s as usize] = e;
                self.index.insert(Self::key(s, c), e);
            }
        }
    }

    fn new_state(&mut self, len: u32, link: u32, cnt: u32) -> u32 {
        self.len.push(len);
        self.link.push(link);
        self.cnt.push(cnt);
        self.first.push(NONE);
        (self.len.len() - 1) as u32
    }

    pub fn add(&mut self, c: u32) {
        self.size += 1;
        let cur = self.new_state(self.len[self.last as usize] + 1, 0, 0);
        let mut p = self.last;
        while p != NONE && self.go(p, c).is_none() {
            self.set(p, c, cur);
            p = self.link[p as usize];
        }
        if p != NONE {
            let q = self.go(p, c).unwrap();
            if self.len[p as usize] + 1 == self.len[q as usize] {
                self.link[cur as usize] = q;
            } else {
                let clone = self.new_state(
                    self.len[p as usize] + 1,
                    self.link[q as usize],
                    self.cnt[q as usize],
                );
                let mut e = self.first[q as usize];
                while e != NONE {
                    let Edge { tok, to, next } = self.edges[e as usize];
                    self.set(clone, tok, to);
                    e = next;
                }
                while p != NONE && self.go(p, c) == Some(q) {
                    self.set(p, c, clone);
                    p = self.link[p as usize];
                }
                self.link[q as usize] = clone;
                self.link[cur as usize] = clone;
            }
        }
        self.last = cur;
        // Every state on the suffix path now ends here too.
        let mut s = cur;
        for _ in 0..COUNT_WALK {
            if s == 0 || s == NONE {
                break;
            }
            self.cnt[s as usize] += 1;
            s = self.link[s as usize];
        }
    }

    pub fn extend(&mut self, toks: &[u32]) {
        for &t in toks {
            self.add(t);
        }
    }

    /// The longest suffix of everything added that also occurred earlier: (state, length),
    /// length capped at `depth`.
    pub fn self_match(&self, depth: usize) -> (u32, usize) {
        let s = if self.last == 0 {
            0
        } else {
            self.link[self.last as usize]
        };
        self.cap(s, self.len[s as usize] as usize, depth)
    }

    fn cap(&self, mut s: u32, n: usize, depth: usize) -> (u32, usize) {
        if n <= depth {
            return (s, n);
        }
        while self.len[self.link[s as usize] as usize] as usize >= depth {
            s = self.link[s as usize];
        }
        (s, depth)
    }

    /// The most frequent continuation of state `s`: (token, next state, its share).
    fn best_child(&self, s: u32) -> Option<(u32, u32, f32)> {
        let (mut total, mut best) = (0u64, None::<(u32, u32, u32)>);
        let mut e = self.first[s as usize];
        while e != NONE {
            let Edge { tok, to, next } = self.edges[e as usize];
            let c = self.cnt[to as usize].max(1);
            total += c as u64;
            // Ties go to the lower token id, so drafts are reproducible.
            if best.is_none_or(|(bt, _, bc)| c > bc || (c == bc && tok < bt)) {
                best = Some((tok, to, c));
            }
            e = next;
        }
        let (tok, to, c) = best?;
        (tok != SEP).then_some((tok, to, c as f32 / total as f32))
    }

    /// Every continuation of state `s` (not the separator), most frequent first (ties: lower
    /// token id): (token, next state, share).
    pub fn children(&self, s: u32) -> Vec<(u32, u32, f32)> {
        let (mut total, mut out) = (0u64, Vec::new());
        let mut e = self.first[s as usize];
        while e != NONE {
            let Edge { tok, to, next } = self.edges[e as usize];
            let c = self.cnt[to as usize].max(1);
            total += c as u64;
            if tok != SEP {
                out.push((tok, to, c));
            }
            e = next;
        }
        out.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        out.into_iter()
            .map(|(t, to, c)| (t, to, c as f32 / total as f32))
            .collect()
    }

    /// Follow the most frequent continuation from `s` for up to `max` tokens: each token with
    /// its conditional share among the state's continuations.
    pub fn continuation(&self, mut s: u32, max: usize) -> Vec<(u32, f32)> {
        let mut out = Vec::new();
        while out.len() < max {
            let Some((tok, to, p)) = self.best_child(s) else {
                break;
            };
            out.push((tok, p));
            s = to;
        }
        out
    }
}

/// Tracks the longest suffix of a token stream that occurs in a fixed automaton.
#[derive(Clone, Copy, Default)]
pub struct Matcher {
    s: u32,
    n: usize,
}

impl Matcher {
    pub fn feed(&mut self, sam: &Sam, c: u32, depth: usize) {
        let (mut s, mut n) = (self.s, self.n);
        while s != 0 && sam.go(s, c).is_none() {
            s = sam.link[s as usize];
            n = sam.len[s as usize] as usize;
        }
        match sam.go(s, c) {
            Some(t) => {
                s = t;
                n += 1;
            }
            None => {
                s = 0;
                n = 0;
            }
        }
        (self.s, self.n) = sam.cap(s, n, depth);
    }
}

/// A bounded store of earlier completions, as one automaton, optionally mirrored on disk.
/// With a TTL (see [`crate::retention`]), a completion is dropped (from memory and the file)
/// once it's that old.
pub struct Global {
    /// Each completion and when it was added (unix seconds).
    seqs: VecDeque<(u64, Vec<u32>)>,
    tokens: usize,
    cap: usize,
    pub sam: Sam,
    file: Option<PathBuf>,
    ttl: Option<std::time::Duration>,
}

/// Seconds since the epoch, by [`crate::retention::now`].
fn unix(t: std::time::SystemTime) -> u64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Global {
    /// `cap`: most tokens kept. `file`: where to keep them across restarts.
    pub fn new(cap: usize, file: Option<PathBuf>) -> Self {
        let mut g = Self {
            seqs: VecDeque::new(),
            tokens: 0,
            cap,
            sam: Sam::new(),
            file,
            ttl: crate::retention::ttl(),
        };
        if let Some(seqs) = g.file.as_ref().and_then(|f| read_seqs(f).ok()) {
            for s in seqs {
                g.tokens += s.1.len();
                g.seqs.push_back(s);
            }
            g.trim(cap);
            g.drop_expired();
            g.rebuild();
            // Compact the file to what's kept.
            g.save_all();
        }
        g
    }

    pub fn set_ttl(&mut self, ttl: Option<std::time::Duration>) {
        self.ttl = ttl;
    }

    /// Drop the oldest completions past the TTL; whether any went.
    fn drop_expired(&mut self) -> bool {
        let Some(ttl) = self.ttl else { return false };
        let before = self.seqs.len();
        let cut = unix(crate::retention::now()).saturating_sub(ttl.as_secs());
        while self.seqs.front().is_some_and(|s| s.0 < cut) {
            let (_, s) = self.seqs.pop_front().unwrap();
            self.tokens -= s.len();
        }
        self.seqs.len() != before
    }

    /// Drop completions past the TTL, rebuilding the automaton and rewriting the file (deleted
    /// when nothing's left). Returns whether any went.
    pub fn expire(&mut self) -> bool {
        if !self.drop_expired() {
            return false;
        }
        self.rebuild();
        self.save_all();
        true
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    fn trim(&mut self, to: usize) {
        while self.tokens > to {
            let Some((_, s)) = self.seqs.pop_front() else {
                break;
            };
            self.tokens -= s.len();
        }
    }

    fn rebuild(&mut self) {
        let mut sam = Sam::with_capacity(self.tokens + self.seqs.len());
        for (_, s) in &self.seqs {
            sam.extend(s);
            sam.add(SEP);
        }
        self.sam = sam;
    }

    /// Remember a completion. Past the cap the oldest quarter goes and the automaton is rebuilt.
    pub fn push(&mut self, seq: &[u32]) {
        if seq.len() < 2 || self.cap == 0 {
            return;
        }
        let seq = seq[seq.len().saturating_sub(self.cap)..].to_vec();
        self.sam.extend(&seq);
        self.sam.add(SEP);
        self.tokens += seq.len();
        let at = unix(crate::retention::now());
        if let Some(f) = &self.file {
            let _ = append_seq(f, at, &seq);
        }
        self.seqs.push_back((at, seq));
        if self.tokens > self.cap {
            self.trim(self.cap * 3 / 4);
            self.rebuild();
            self.save_all();
        }
    }

    fn save_all(&self) {
        let Some(f) = &self.file else { return };
        if self.seqs.is_empty() {
            let _ = std::fs::remove_file(f);
            return;
        }
        let tmp = f.with_extension("tmp");
        let ok = (|| -> std::io::Result<()> {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            w.write_all(&STORE_MAGIC.to_le_bytes())?;
            for (at, s) in &self.seqs {
                write_seq(&mut w, *at, s)?;
            }
            w.flush()?;
            std::fs::rename(&tmp, f)
        })();
        if ok.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// The store file's first word: records are (unix seconds u64, length u32, tokens). A file
/// without it is the older format, records (length, tokens), all dated by the file's
/// modification time.
const STORE_MAGIC: u32 = 0xffff_0002;

fn write_seq(w: &mut impl Write, at: u64, s: &[u32]) -> std::io::Result<()> {
    w.write_all(&at.to_le_bytes())?;
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    let bytes: Vec<u8> = s.iter().flat_map(|t| t.to_le_bytes()).collect();
    w.write_all(&bytes)
}

fn append_seq(f: &PathBuf, at: u64, s: &[u32]) -> std::io::Result<()> {
    if let Some(dir) = f.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(f)?;
    let mut buf = Vec::with_capacity(4 * s.len() + 16);
    if w.metadata()?.len() == 0 {
        buf.extend_from_slice(&STORE_MAGIC.to_le_bytes());
    }
    write_seq(&mut buf, at, s)?;
    w.write_all(&buf)
}

fn read_seqs(f: &PathBuf) -> std::io::Result<Vec<(u64, Vec<u32>)>> {
    let mut file = std::fs::File::open(f)?;
    let mtime = unix(file.metadata()?.modified()?);
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let word = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    let dated = bytes.len() >= 4 && word(0) == STORE_MAGIC;
    let (mut out, mut i) = (Vec::new(), if dated { 4 } else { 0 });
    let head = if dated { 12 } else { 4 };
    while i + head <= bytes.len() {
        let at = if dated {
            u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap())
        } else {
            mtime
        };
        let n = word(i + head - 4) as usize;
        i += head;
        if i + 4 * n > bytes.len() {
            break; // a torn write at the end
        }
        out.push((at, (0..n).map(|j| word(i + 4 * j)).collect()));
        i += 4 * n;
    }
    Ok(out)
}

/// Knobs for drafting.
#[derive(Debug, Clone)]
pub struct DraftConfig {
    /// Most draft tokens per verify step.
    pub max_draft: usize,
    /// Draft at most `alpha x match length` tokens.
    pub alpha: f32,
    /// Stop drafting once the chance that the whole draft so far is accepted drops below this.
    pub min_prob: f32,
    /// Longest suffix match considered.
    pub max_depth: usize,
    /// Shortest suffix match drafted from.
    pub min_match: usize,
    /// Tokens kept in the global store of earlier completions (0: none); about 110 bytes of
    /// automaton each.
    pub global_tokens: usize,
    /// Where the global store lives across restarts.
    pub store: Option<PathBuf>,
    /// Cost of a forward over k tokens relative to one, as (k, cost) points, interpolated.
    pub cost: Vec<(usize, f32)>,
}

/// Measured verify cost curves (median forward time over k tokens / one token, at 4K context;
/// crates/tang-llm `bench-verify`, 2026-09-24).
pub const COST_CUDA: &[(usize, f32)] = &[
    (1, 1.0),
    (2, 1.17),
    (4, 1.74),
    (8, 1.97),
    (16, 3.28),
    (32, 6.22),
];
/// Up to 8: Qwen3-32B-4bit, M4 Max, 512 context, with the small-row Q4 kernels (2026-10-17).
pub const COST_METAL: &[(usize, f32)] = &[
    (1, 1.0),
    (2, 1.14),
    (3, 1.83),
    (4, 2.2),
    (8, 2.33),
    (16, 2.86),
    (32, 4.63),
];

impl DraftConfig {
    pub fn new(cost: &[(usize, f32)]) -> Self {
        Self {
            // Forwards of up to 16 tokens stay on the small-batch kernels; 17 costs far more.
            max_draft: 15,
            alpha: 2.0,
            min_prob: 0.1,
            max_depth: 64,
            min_match: 1,
            global_tokens: 1 << 19,
            store: None,
            cost: cost.to_vec(),
        }
    }

    /// `TANG_DRAFT_*` overrides: MAX, ALPHA, MIN_PROB, GLOBAL (tokens), COST ("1:1,2:1.2,...").
    pub fn from_env(mut self) -> Self {
        let get = |k: &str| std::env::var(format!("TANG_DRAFT_{k}")).ok();
        if let Some(v) = get("MAX").and_then(|v| v.parse().ok()) {
            self.max_draft = v;
        }
        if let Some(v) = get("ALPHA").and_then(|v| v.parse().ok()) {
            self.alpha = v;
        }
        if let Some(v) = get("MIN_PROB").and_then(|v| v.parse().ok()) {
            self.min_prob = v;
        }
        if let Some(v) = get("MIN_MATCH").and_then(|v| v.parse().ok()) {
            self.min_match = v;
        }
        if let Some(v) = get("GLOBAL").and_then(|v| v.parse().ok()) {
            self.global_tokens = v;
        }
        if let Some(v) = get("COST") {
            let pts: Option<Vec<(usize, f32)>> = v
                .split(',')
                .map(|p| {
                    let (k, c) = p.split_once(':')?;
                    Some((k.trim().parse().ok()?, c.trim().parse().ok()?))
                })
                .collect();
            if let Some(p) = pts.filter(|p| !p.is_empty()) {
                self.cost = p;
            }
        }
        self
    }

    /// Relative cost of a forward over `k` tokens.
    pub fn cost(&self, k: usize) -> f32 {
        let pts = &self.cost;
        if k <= pts[0].0 {
            return pts[0].1;
        }
        for w in pts.windows(2) {
            let ((k0, c0), (k1, c1)) = (w[0], w[1]);
            if k <= k1 {
                return c0 + (c1 - c0) * (k - k0) as f32 / (k1 - k0) as f32;
            }
        }
        // Past the table: extend the last segment's slope.
        let n = pts.len();
        let (k0, c0) = pts[n.saturating_sub(2)];
        let (k1, c1) = pts[n - 1];
        let slope = if k1 > k0 {
            (c1 - c0) / (k1 - k0) as f32
        } else {
            0.0
        };
        c1 + slope * (k - k1) as f32
    }
}

/// Match-length buckets for the learned hit rates.
const BUCKETS: [usize; 8] = [1, 2, 3, 5, 8, 16, 32, usize::MAX];

fn bucket(n: usize) -> usize {
    BUCKETS
        .iter()
        .position(|&b| n <= b)
        .unwrap_or(BUCKETS.len() - 1)
}

/// Per-token hit rates by match length: how often a drafted token (whose predecessors were all
/// accepted) is what the model produced, relative to the continuation's frequency share.
#[derive(Debug, Clone)]
pub struct Calibration {
    hits: [f32; BUCKETS.len()],
    tries: [f32; BUCKETS.len()],
}

/// Prior pseudo-observations per bucket.
const PRIOR: f32 = 20.0;
/// Forget old evidence slowly (per observation), so the rates track the workload.
const DECAY: f32 = 0.998;

impl Default for Calibration {
    fn default() -> Self {
        // Priors: short matches are often coincidental, long ones usually copies.
        let prior = [0.35, 0.45, 0.55, 0.65, 0.75, 0.8, 0.85, 0.9];
        Self {
            hits: prior.map(|p| p * PRIOR),
            tries: [PRIOR; BUCKETS.len()],
        }
    }
}

impl Calibration {
    pub fn rate(&self, n: usize) -> f32 {
        let b = bucket(n);
        self.hits[b] / self.tries[b]
    }

    /// A drafted token at match length `n`, with continuation share `share`, was tried: `hit`
    /// if accepted.
    pub fn observe(&mut self, n: usize, share: f32, hit: bool) {
        let b = bucket(n);
        self.hits[b] = self.hits[b] * DECAY + hit as u8 as f32;
        self.tries[b] = self.tries[b] * DECAY + share;
    }
}

/// Verify costs: the configured curve until this server has timed enough forwards of a width
/// (and of one token) to use its own measurements, which track the backend, the context
/// length and whatever else shares the GPU.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Mean seconds per forward, by width (tokens fed), as a moving average.
    secs: Vec<f32>,
    n: Vec<u32>,
}

/// Forwards of a width (and of width 1) before its measured cost is trusted.
const COST_MIN_N: u32 = 8;
const COST_EMA: f32 = 0.05;

impl CostModel {
    pub fn new(max_width: usize) -> Self {
        Self {
            secs: vec![0.0; max_width + 1],
            n: vec![0; max_width + 1],
        }
    }

    pub fn observe(&mut self, width: usize, secs: f64) {
        let Some(n) = self.n.get_mut(width) else {
            return;
        };
        *n += 1;
        let a = COST_EMA.max(1.0 / *n as f32);
        self.secs[width] += a * (secs as f32 - self.secs[width]);
    }

    /// Relative cost by width (index 0 unused), measured where there's enough data, the
    /// configured curve elsewhere; made non-decreasing.
    pub fn table(&self, cfg: &DraftConfig, max_width: usize) -> Vec<f32> {
        let mut t = vec![1.0f32; max_width + 1];
        let base = (self.n.get(1).copied().unwrap_or(0) >= COST_MIN_N).then(|| self.secs[1]);
        for (w, c) in t.iter_mut().enumerate().skip(2) {
            *c = match base {
                Some(b) if b > 0.0 && self.n.get(w).copied().unwrap_or(0) >= COST_MIN_N => {
                    self.secs[w] / b
                }
                _ => cfg.cost(w),
            };
        }
        for w in 2..t.len() {
            t[w] = t[w].max(t[w - 1]);
        }
        t
    }

    /// Mean measured seconds for a width, and how many forwards that's from.
    pub fn measured(&self, width: usize) -> (f32, u32) {
        (self.secs[width], self.n[width])
    }
}

/// A proposed draft.
#[derive(Debug, Clone, Default)]
pub struct Draft {
    pub tokens: Vec<u32>,
    /// Each token's continuation share (for calibration).
    pub shares: Vec<f32>,
    /// Match length it came from (for calibration).
    pub match_len: usize,
    /// From a draft model rather than a suffix match (not calibrated).
    pub from_model: bool,
}

/// The drafter for one request: the request's automaton plus a matcher into the global one.
/// It borrows the shared state it drafts from; a [`Drafter`] is the same without the borrow
/// (handed the config and global automaton each call), for a request that outlives one.
pub struct Session<'a> {
    cfg: &'a DraftConfig,
    global: &'a Global,
    d: Drafter,
}

impl<'a> Session<'a> {
    pub fn new(
        cfg: &'a DraftConfig,
        global: &'a Global,
        calib: Calibration,
        cost: Vec<f32>,
        prompt: &[u32],
    ) -> Self {
        Self {
            cfg,
            global,
            d: Drafter::new(cfg, global, calib, cost, prompt),
        }
    }

    /// A forward over `width` tokens relative to one (measured as decoding goes).
    pub fn cost(&self, width: usize) -> f32 {
        self.d.cost(width)
    }

    /// A token was generated (or injected).
    pub fn push(&mut self, t: u32) {
        self.d.push(self.cfg, self.global, t)
    }

    /// The draft to verify next, at most `room` tokens.
    pub fn propose(&self, room: usize) -> Draft {
        self.d.propose(self.cfg, self.global, room)
    }
}

impl std::ops::Deref for Session<'_> {
    type Target = Drafter;
    fn deref(&self) -> &Drafter {
        &self.d
    }
}

impl std::ops::DerefMut for Session<'_> {
    fn deref_mut(&mut self) -> &mut Drafter {
        &mut self.d
    }
}

/// A [`Session`]'s own state.
pub struct Drafter {
    /// The hit rates, updated as drafts are verified (the caller keeps them afterwards).
    pub calib: Calibration,
    /// Relative verify cost by forward width.
    cost: Vec<f32>,
    local: Sam,
    gm: Matcher,
}

impl Drafter {
    pub fn new(
        cfg: &DraftConfig,
        global: &Global,
        calib: Calibration,
        cost: Vec<f32>,
        prompt: &[u32],
    ) -> Self {
        let mut local = Sam::with_capacity(prompt.len() + 1024);
        local.extend(prompt);
        let mut gm = Matcher::default();
        for &t in &prompt[prompt.len().saturating_sub(cfg.max_depth)..] {
            gm.feed(&global.sam, t, cfg.max_depth);
        }
        Self {
            calib,
            cost,
            local,
            gm,
        }
    }

    /// A forward over `width` tokens relative to one (measured as decoding goes).
    pub fn cost(&self, width: usize) -> f32 {
        self.cost.get(width).copied().unwrap_or(f32::INFINITY)
    }

    /// A token was generated (or injected).
    pub fn push(&mut self, cfg: &DraftConfig, global: &Global, t: u32) {
        self.local.add(t);
        self.gm.feed(&global.sam, t, cfg.max_depth);
    }

    /// Branches for a draft tree: up to `k` chains of at most `room` tokens, each starting
    /// with a different token, from the local and global matches (most frequent first).
    /// Each chain is the child token followed by that child's greedy continuation.
    pub fn branches(&self, cfg: &DraftConfig, global: &Global, k: usize, room: usize) -> Vec<Draft> {
        let (ls, ln) = self.local.self_match(cfg.max_depth);
        let mut cands: Vec<(f32, Draft)> = Vec::new();
        for (sam, s, n) in [(&self.local, ls, ln), (&global.sam, self.gm.s, self.gm.n)] {
            if n == 0 || n < cfg.min_match || room == 0 {
                continue;
            }
            let rate = self.calib.rate(n);
            for (tok, to, share) in sam.children(s).into_iter().take(k) {
                let mut d = Draft {
                    tokens: vec![tok],
                    shares: vec![share],
                    match_len: n,
                    from_model: false,
                };
                for (t, sh) in sam.continuation(to, room - 1) {
                    d.tokens.push(t);
                    d.shares.push(sh);
                }
                cands.push((rate * share, d));
            }
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out: Vec<Draft> = Vec::new();
        for (_, d) in cands {
            if out.len() < k && !out.iter().any(|o| o.tokens[0] == d.tokens[0]) {
                out.push(d);
            }
        }
        out
    }

    /// The draft to verify next, at most `room` tokens: the best source's continuation, cut
    /// where expected tokens per verify cost peaks.
    pub fn propose(&self, cfg: &DraftConfig, global: &Global, room: usize) -> Draft {
        let room = room
            .min(cfg.max_draft)
            .min(self.cost.len().saturating_sub(2));
        if room == 0 {
            return Draft::default();
        }
        let (ls, ln) = self.local.self_match(cfg.max_depth);
        let mut best = (Draft::default(), 0.0f32);
        for (sam, s, n) in [(&self.local, ls, ln), (&global.sam, self.gm.s, self.gm.n)] {
            if n == 0 || n < cfg.min_match {
                continue;
            }
            let limit = room.min(((cfg.alpha * n as f32) as usize).max(1));
            let rate = self.calib.rate(n);
            // Expected tokens per step for each draft length; keep the best.
            let (mut p, mut sum) = (1.0f32, 0.0f32);
            let (mut len, mut score) = (0usize, 1.0 / cfg.cost(1));
            let cont = sam.continuation(s, limit);
            for (i, &(_, share)) in cont.iter().enumerate() {
                p *= (rate * share).min(1.0);
                if p < cfg.min_prob {
                    break;
                }
                sum += p;
                let sc = (1.0 + sum) / self.cost[i + 2];
                if sc > score {
                    (len, score) = (i + 1, sc);
                }
            }
            if len > 0 && score > best.1 {
                best = (
                    Draft {
                        tokens: cont[..len].iter().map(|c| c.0).collect(),
                        shares: cont[..len].iter().map(|c| c.1).collect(),
                        match_len: n,
                        from_model: false,
                    },
                    score,
                );
            }
        }
        best.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DraftConfig {
        let mut c = DraftConfig::new(COST_CUDA);
        c.global_tokens = 1000;
        c
    }

    #[test]
    fn sam_finds_longest_earlier_match_and_its_continuation() {
        let mut s = Sam::new();
        s.extend(&[1, 2, 3, 4, 5, 9, 1, 2, 3]);
        let (st, n) = s.self_match(64);
        assert_eq!(n, 3);
        let cont: Vec<u32> = s.continuation(st, 10).iter().map(|c| c.0).collect();
        assert_eq!(cont, vec![4, 5, 9, 1, 2, 3]);
    }

    #[test]
    fn continuation_prefers_the_more_frequent_branch() {
        let mut s = Sam::new();
        // "7 8" is followed by 1 twice and by 2 once.
        s.extend(&[9, 7, 8, 1, 5, 7, 8, 2, 6, 7, 8, 1, 3, 7, 8]);
        let (st, n) = s.self_match(64);
        assert!(n >= 2);
        let c = s.continuation(st, 1);
        assert_eq!(c[0].0, 1);
        assert!((c[0].1 - 2.0 / 3.0).abs() < 1e-6, "{c:?}");
    }

    #[test]
    fn matcher_tracks_a_stream_against_a_fixed_sam() {
        let mut s = Sam::new();
        s.extend(&[5, 6, 7, 8, 9]);
        let mut m = Matcher::default();
        for t in [1, 6, 7] {
            m.feed(&s, t, 64);
        }
        assert_eq!(m.n, 2);
        assert_eq!(s.continuation(m.s, 5)[0].0, 8);
        m.feed(&s, 3, 64);
        assert_eq!(m.n, 0);
    }

    #[test]
    fn children_are_every_branch_by_frequency() {
        let mut s = Sam::new();
        s.extend(&[9, 7, 8, 1, 5, 7, 8, 2, 6, 7, 8, 1, 3, 7, 8]);
        let (st, _) = s.self_match(64);
        let c: Vec<u32> = s.children(st).iter().map(|c| c.0).collect();
        assert_eq!(c, vec![1, 2]);
    }

    #[test]
    fn depth_caps_the_match() {
        let mut s = Sam::new();
        let seq: Vec<u32> = (0..50).chain(0..50).collect();
        s.extend(&seq);
        let (st, n) = s.self_match(8);
        assert_eq!(n, 8);
        // Still continues correctly from the capped state (the earlier copy's next token).
        assert_eq!(s.continuation(st, 1)[0].0, 0);
    }

    #[test]
    fn a_session_drafts_a_copy_from_the_prompt() {
        let cfg = cfg();
        let g = Global::new(0, None);
        let cal = Calibration::default();
        let code: Vec<u32> = (100..140).collect();
        let mut prompt = code.clone();
        prompt.extend([1, 2, 3]);
        let mut s = Session::new(
            &cfg,
            &g,
            cal.clone(),
            CostModel::new(17).table(&cfg, 17),
            &prompt,
        );
        for &t in &code[..12] {
            s.push(t);
        }
        let d = s.propose(16);
        assert!(!d.tokens.is_empty());
        assert_eq!(d.tokens, code[12..12 + d.tokens.len()].to_vec());
    }

    #[test]
    fn global_store_drafts_earlier_completions_and_stays_bounded() {
        let cfg = cfg();
        let mut g = Global::new(100, None);
        let old: Vec<u32> = (500..540).collect();
        g.push(&old);
        let cal = Calibration::default();
        let mut s = Session::new(
            &cfg,
            &g,
            cal.clone(),
            CostModel::new(17).table(&cfg, 17),
            &[1, 2, 3],
        );
        for &t in &old[..10] {
            s.push(t);
        }
        let d = s.propose(16);
        assert_eq!(d.tokens, old[10..10 + d.tokens.len()].to_vec());
        assert!(!d.tokens.is_empty());
        for i in 0..10 {
            g.push(&(i * 30..i * 30 + 30).collect::<Vec<u32>>());
        }
        assert!(g.tokens() <= 100);
    }

    #[test]
    fn global_store_survives_a_restart() {
        let dir = std::env::temp_dir().join(format!("tang-draft-test-{}", std::process::id()));
        let f = dir.join("store.tok");
        let _ = std::fs::remove_file(&f);
        let mut g = Global::new(1000, Some(f.clone()));
        g.push(&[1, 2, 3, 4]);
        g.push(&[5, 6, 7]);
        let g2 = Global::new(1000, Some(f.clone()));
        assert_eq!(g2.tokens(), 7);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn global_store_forgets_completions_past_the_ttl() {
        use crate::retention::advance;
        use std::time::Duration;
        let dir = std::env::temp_dir().join(format!("tang-draft-ttl-{}", std::process::id()));
        let f = dir.join("store.tok");
        let _ = std::fs::remove_file(&f);
        let mut g = Global::new(1000, Some(f.clone()));
        g.set_ttl(Some(Duration::from_secs(60)));
        g.push(&[1, 2, 3, 4]);
        advance(Duration::from_secs(40));
        g.push(&[5, 6, 7]);
        assert!(!g.expire());
        advance(Duration::from_secs(30));
        // The first is past it, in memory and on disk.
        assert!(g.expire());
        assert_eq!(g.tokens(), 3);
        assert_eq!(read_seqs(&f).unwrap().len(), 1);
        advance(Duration::from_secs(40));
        assert!(g.expire());
        assert_eq!(g.tokens(), 0);
        assert!(!f.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn global_store_reads_the_undated_format() {
        let dir = std::env::temp_dir().join(format!("tang-draft-old-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("store.tok");
        let old: Vec<u8> = [2u32, 8, 9].iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&f, old).unwrap();
        let g = Global::new(1000, Some(f.clone()));
        assert_eq!(g.tokens(), 2);
        // Rewritten dated.
        assert_eq!(read_seqs(&f).unwrap()[0].1, vec![8, 9]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_match_no_draft_and_cost_interpolates() {
        let cfg = cfg();
        let g = Global::new(0, None);
        let cal = Calibration::default();
        let s = Session::new(
            &cfg,
            &g,
            cal.clone(),
            CostModel::new(17).table(&cfg, 17),
            &[1, 2, 3, 4],
        );
        assert!(s.propose(16).tokens.is_empty());
        assert!((cfg.cost(3) - (1.17 + 1.74) / 2.0).abs() < 1e-5);
        assert!(cfg.cost(64) > cfg.cost(32));
    }
}

#[cfg(test)]
mod scale {
    /// `cargo test --release -p tang-llm --lib scale -- --ignored --nocapture`: build time and
    /// memory for a full global store.
    #[test]
    #[ignore]
    fn million_token_store() {
        let mut x = 0x2545_f491u64;
        // Text-like: a Zipfian-ish token stream with repeated phrases.
        let mut toks = Vec::with_capacity(1 << 20);
        while toks.len() < 1 << 20 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let r = (x % 1000) as usize;
            if r < 300 && toks.len() > 64 {
                let start = (x as usize / 1000) % (toks.len() - 32);
                let len = 4 + r % 28;
                let copy: Vec<u32> = toks[start..start + len].to_vec();
                toks.extend(copy);
            } else {
                toks.push((1.0 / ((x >> 20) as f64 / (1u64 << 44) as f64 + 1e-5)) as u32 % 150_000);
            }
        }
        let t = std::time::Instant::now();
        let mut s = super::Sam::with_capacity(toks.len());
        s.extend(&toks);
        let bytes = s.len.capacity() * 16 + s.edges.capacity() * 12 + s.index.capacity() * 13;
        eprintln!(
            "{} tokens: {} states, {} edges, built in {:.2}s, ~{} MB",
            toks.len(),
            s.len.len(),
            s.edges.len(),
            t.elapsed().as_secs_f64(),
            bytes >> 20
        );
    }
}
