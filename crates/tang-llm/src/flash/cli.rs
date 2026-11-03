//! `tang-llm flash-generate`, `flash-parity` and `flash-bench`: the CUDA engine from the
//! command line.

use super::coder::{self, Coder, CoderOpts};
use super::engine::{Engine, ExpertMode, Opts, Probe, WinStats};
use super::reference::{top_logprobs, Hparams};
use anyhow::{bail, ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tang_compute::flash::shape::*;
use tang_compute::flash::MAX_T;

struct Args {
    path: PathBuf,
    ids: Vec<u32>,
    prompt: Option<String>,
    n: usize,
    opts: Opts,
    eager: bool,
    chunk: usize,
    dump_routing: Option<PathBuf>,
    reference: Option<PathBuf>,
    dump: Option<PathBuf>,
    ctx: Option<usize>,
    quiet: bool,
    save: Option<PathBuf>,
    /// Drafts: none, suffix, wrong (forced-wrong tokens), oracle:FILE (a previous run's ids).
    draft: String,
    temp: f32,
    seed: u32,
    out: Option<PathBuf>,
    think: Option<bool>,
    dump_mtp_train: Option<PathBuf>,
    /// Memory bandwidth for `flash-bench`'s speed-of-light line (GB/s; default H100 SXM).
    bw: f64,
}

fn read_ids(f: &str) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?;
    s.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
        .filter(|w| !w.is_empty())
        .map(|w| w.parse().with_context(|| format!("{f}: bad id {w:?}")))
        .collect()
}

fn parse(args: &[String]) -> Result<Args> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let mut a = Args {
        path,
        ids: Vec::new(),
        prompt: None,
        n: 32,
        opts: Opts::default(),
        eager: false,
        chunk: 8,
        dump_routing: None,
        reference: None,
        dump: None,
        ctx: None,
        quiet: false,
        save: None,
        draft: "none".into(),
        temp: 0.0,
        seed: 0,
        out: None,
        think: None,
        dump_mtp_train: None,
        bw: 3350.0,
    };
    while let Some(x) = it.next() {
        let mut val = || it.next().with_context(|| format!("{x} needs a value"));
        match x.as_str() {
            "--prompt-ids" => {
                for w in val()?.split(|c: char| c.is_whitespace() || c == ',') {
                    if !w.is_empty() {
                        a.ids.push(w.parse()?);
                    }
                }
            }
            "--ids-file" => a.ids = read_ids(val()?)?,
            "--prompt" => a.prompt = Some(val()?.clone()),
            "--prompt-file" => a.prompt = Some(std::fs::read_to_string(val()?)?),
            "-n" => a.n = val()?.parse()?,
            "--max-ctx" => a.opts.max_ctx = val()?.parse()?,
            "--ctx" => a.ctx = Some(val()?.parse()?),
            "--slots" => a.opts.slots = Some(val()?.parse()?),
            "--reserve-mb" => a.opts.reserve_mb = val()?.parse()?,
            "--mtp-mb" => a.opts.mtp_mb = val()?.parse()?,
            "--profile" => a.opts.profile = Some(PathBuf::from(val()?)),
            "--dump-routing" => a.dump_routing = Some(PathBuf::from(val()?)),
            "--all-cpu" => a.opts.experts = ExpertMode::AllCpu,
            "--no-adapt" => a.opts.adapt = false,
            "--split" => a.opts.split = true,
            "--eager" => a.eager = true,
            "--chunk" => a.chunk = val()?.parse()?,
            "--ref" => a.reference = Some(PathBuf::from(val()?)),
            "--dump" => a.dump = Some(PathBuf::from(val()?)),
            "-q" => a.quiet = true,
            "--save" => a.save = Some(PathBuf::from(val()?)),
            "--draft" => a.draft = val()?.clone(),
            "--no-think" => a.think = Some(false),
            "--dump-mtp-train" => a.dump_mtp_train = Some(PathBuf::from(val()?)),
            "--mtp" => a.opts.mtp = Some(PathBuf::from(val()?)),
            "--temp" => a.temp = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--out" => a.out = Some(PathBuf::from(val()?)),
            "--bw" => a.bw = val()?.parse()?,
            s => bail!("unknown argument {s}"),
        }
    }
    Ok(a)
}

fn load(a: &Args) -> Result<Engine> {
    let mut e = Engine::load(&a.path, a.opts.clone())?;
    e.use_graphs = !a.eager;
    e.temperature = a.temp;
    e.seed = a.seed;
    eprintln!("{}", e.load_report);
    Ok(e)
}

/// What the decode, spec-test, tcheck and parity commands need from an engine: the Flash-Next
/// [`Engine`] or the qwen3next [`Coder`].
pub trait FlashModel {
    fn reset(&mut self);
    fn prefill(
        &mut self,
        ids: &[u32],
        chunk: usize,
        cb: Option<coder::LogitsCb<'_>>,
    ) -> Result<u32>;
    fn step(&mut self, tok: u32) -> Result<u32>;
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>>;
    fn window(&mut self, pos0: usize, t: usize, probe: Option<&mut Probe>) -> Result<Vec<u32>>;
    fn window_mode(
        &mut self,
        pos0: usize,
        t: usize,
        verify: bool,
        probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>>;
    fn logits(&self, t: usize) -> Vec<f32>;
    fn last(&self) -> WinStats;
    fn hp(&self) -> &Hparams;
    fn tokens_mut(&mut self) -> &mut Vec<u32>;
    fn set_sampling(&mut self, temperature: f32, seed: u32);
    fn vocab(&self) -> Result<Vec<String>>;
    fn last_commit_ms(&self) -> f64;
    fn op_tcheck(&mut self, _l: usize) -> Result<()> {
        Ok(())
    }
    fn has_mtp(&self) -> bool {
        false
    }
    fn set_use_mtp(&mut self, _on: bool) {}
    fn mtp_last(&self) -> Vec<(u32, f32)> {
        Vec::new()
    }
    fn last_mtp_ms(&self) -> f64 {
        0.0
    }
    fn last_mtp_gpu_ms(&self) -> f64 {
        0.0
    }
}

impl FlashModel for Engine {
    fn reset(&mut self) {
        Engine::reset(self)
    }
    fn prefill(
        &mut self,
        ids: &[u32],
        chunk: usize,
        cb: Option<coder::LogitsCb<'_>>,
    ) -> Result<u32> {
        Engine::prefill(self, ids, chunk, cb)
    }
    fn step(&mut self, tok: u32) -> Result<u32> {
        Engine::step(self, tok)
    }
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        Engine::verify(self, cur, drafts)
    }
    fn window(&mut self, pos0: usize, t: usize, probe: Option<&mut Probe>) -> Result<Vec<u32>> {
        Engine::window(self, pos0, t, probe)
    }
    fn window_mode(
        &mut self,
        pos0: usize,
        t: usize,
        verify: bool,
        probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        Engine::window_mode(self, pos0, t, verify, probe)
    }
    fn logits(&self, t: usize) -> Vec<f32> {
        Engine::logits(self, t)
    }
    fn last(&self) -> WinStats {
        self.last
    }
    fn hp(&self) -> &Hparams {
        &self.hp
    }
    fn tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.tokens
    }
    fn set_sampling(&mut self, temperature: f32, seed: u32) {
        self.temperature = temperature;
        self.seed = seed;
    }
    fn vocab(&self) -> Result<Vec<String>> {
        Engine::vocab(self)
    }
    fn last_commit_ms(&self) -> f64 {
        self.last_commit_ms
    }
    fn op_tcheck(&mut self, l: usize) -> Result<()> {
        Engine::op_tcheck(self, l)
    }
    fn has_mtp(&self) -> bool {
        Engine::has_mtp(self)
    }
    fn set_use_mtp(&mut self, on: bool) {
        self.use_mtp = on;
    }
    fn mtp_last(&self) -> Vec<(u32, f32)> {
        self.mtp_last.clone()
    }
    fn last_mtp_ms(&self) -> f64 {
        self.last_mtp_ms
    }
    fn last_mtp_gpu_ms(&self) -> f64 {
        self.last_mtp_gpu_ms
    }
}

impl FlashModel for Coder {
    fn reset(&mut self) {
        Coder::reset(self)
    }
    fn has_mtp(&self) -> bool {
        Coder::has_mtp(self)
    }
    fn set_use_mtp(&mut self, on: bool) {
        self.use_mtp = on && Coder::has_mtp(self);
    }
    fn mtp_last(&self) -> Vec<(u32, f32)> {
        self.mtp_last.clone()
    }
    fn last_mtp_ms(&self) -> f64 {
        self.last_mtp_ms
    }
    fn prefill(
        &mut self,
        ids: &[u32],
        chunk: usize,
        cb: Option<coder::LogitsCb<'_>>,
    ) -> Result<u32> {
        Coder::prefill(self, ids, chunk, cb)
    }
    fn step(&mut self, tok: u32) -> Result<u32> {
        Coder::step(self, tok)
    }
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        Coder::verify(self, cur, drafts)
    }
    fn window(&mut self, pos0: usize, t: usize, probe: Option<&mut Probe>) -> Result<Vec<u32>> {
        Coder::window(self, pos0, t, probe)
    }
    fn window_mode(
        &mut self,
        pos0: usize,
        t: usize,
        verify: bool,
        probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        Coder::window_mode(self, pos0, t, verify, probe)
    }
    fn logits(&self, t: usize) -> Vec<f32> {
        Coder::logits(self, t)
    }
    fn last(&self) -> WinStats {
        self.last
    }
    fn hp(&self) -> &Hparams {
        &self.hp
    }
    fn tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.tokens
    }
    fn set_sampling(&mut self, temperature: f32, seed: u32) {
        self.temperature = temperature;
        self.seed = seed;
    }
    fn vocab(&self) -> Result<Vec<String>> {
        Coder::vocab(self)
    }
    fn last_commit_ms(&self) -> f64 {
        self.last_commit_ms
    }
}

/// The engine for the GGUF's architecture: [`Coder`] for `qwen3next`, else [`Engine`].
fn load_model(a: &Args) -> Result<Box<dyn FlashModel>> {
    if coder::is_qwen3next(&a.path)? {
        return Ok(Box::new(load_coder(a)?));
    }
    Ok(Box::new(load(a)?))
}

fn load_coder(a: &Args) -> Result<Coder> {
    let mut e = Coder::load(
        &a.path,
        CoderOpts {
            max_ctx: a.opts.max_ctx,
            mtp: a.opts.mtp.clone(),
        },
    )?;
    e.use_graphs = !a.eager;
    e.temperature = a.temp;
    e.seed = a.seed;
    eprintln!("{}", e.load_report);
    Ok(e)
}

/// Byte-level BPE piece → text (GPT-2 byte map), for printing.
pub fn detok(vocab: &[String], ids: &[u32]) -> String {
    let mut inv = [0u8; 512];
    let mut n = 0u32;
    let mut map = std::collections::HashMap::new();
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            n += 1;
            255 + n
        };
        map.insert(char::from_u32(c).unwrap(), b as u8);
        if (c as usize) < inv.len() {
            inv[c as usize] = b as u8;
        }
    }
    let mut bytes = Vec::new();
    for &i in ids {
        let piece = vocab.get(i as usize).map(String::as_str).unwrap_or("");
        if piece.starts_with("<|") && piece.ends_with("|>") {
            bytes.extend_from_slice(piece.as_bytes());
            continue;
        }
        for ch in piece.chars() {
            match map.get(&ch) {
                Some(&b) => bytes.push(b),
                None => bytes.extend_from_slice(ch.to_string().as_bytes()),
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn prompt_ids(a: &Args) -> Result<Vec<u32>> {
    if let Some(p) = &a.prompt {
        return super::tokenize::encode_chat(&a.path, p, a.think);
    }
    ensure!(
        !a.ids.is_empty(),
        "no prompt (--prompt-ids, --ids-file or --prompt)"
    );
    Ok(a.ids.clone())
}

fn split_line(s: &WinStats) -> String {
    format!(
        "T={} wall {:.2} ms (host prep {:.2}) | GPU {:.2} ms, waiting for plan {:.2}, for CPU rows {:.2} | host plan {:.2}, CPU experts {:.2} | routed {} distinct {} missed {} ({:.1}% hit) swaps {}",
        s.t,
        s.wall_ms,
        s.host_prep_ms,
        s.gpu_ms,
        s.gpu_wait_a_ms,
        s.gpu_wait_b_ms,
        s.plan_ms,
        s.cpu_ms,
        s.routed,
        s.distinct,
        s.missed,
        100.0 * (1.0 - s.missed as f64 / s.distinct.max(1) as f64),
        s.swaps
    )
}

/// Decode `n` greedy tokens after `ids`; returns (generated, per-step stats, decode seconds).
fn decode(
    e: &mut dyn FlashModel,
    ids: &[u32],
    n: usize,
    chunk: usize,
) -> Result<(Vec<u32>, Vec<WinStats>, f64, f64)> {
    e.reset();
    let t0 = Instant::now();
    let mut next = e.prefill(ids, chunk, None)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    let mut out = vec![next];
    let mut stats = Vec::with_capacity(n);
    // The first T=1 window captures its graph; time from the second.
    next = e.step(next)?;
    out.push(next);
    stats.push(e.last());
    let t1 = Instant::now();
    for _ in 2..n {
        next = e.step(next)?;
        out.push(next);
        stats.push(e.last());
    }
    let secs = t1.elapsed().as_secs_f64();
    out.truncate(n);
    Ok((out, stats, prefill_s, secs))
}

/// Per-run speculation counters.
#[derive(Default, Debug, Clone)]
pub struct SpecStats {
    pub windows: usize,
    pub tokens: usize,
    /// Per draft position j: (proposed, accepted).
    pub by_pos: Vec<(usize, usize)>,
    /// Windows per width.
    pub widths: Vec<usize>,
    /// MTP time per window (ms, summed).
    pub mtp_ms: f64,
    pub commit_ms: f64,
    pub mtp_gpu_ms: f64,
    /// Windows that used the suffix drafter's proposal (hybrid).
    pub suffix_windows: usize,
    /// Per draft position, (proposed, accepted) inside `<think>` and after it.
    pub by_pos_think: Vec<(usize, usize)>,
    pub by_pos_answer: Vec<(usize, usize)>,
    /// Windows per width inside / after thinking.
    pub widths_think: Vec<usize>,
    pub widths_answer: Vec<usize>,
}

/// Decode `n` tokens after `ids` with verify windows and drafts from `kind` (`none`, `suffix`,
/// `wrong`, `oracle:FILE`). Returns (tokens, per-window stats, prefill s, decode s, spec stats).
fn decode_spec(
    e: &mut dyn FlashModel,
    ids: &[u32],
    n: usize,
    chunk: usize,
    kind: &str,
) -> Result<(Vec<u32>, Vec<WinStats>, f64, f64, SpecStats)> {
    decode_spec_options(e, ids, n, chunk, kind, None, None)
}
fn decode_spec_options(
    e: &mut dyn FlashModel,
    ids: &[u32],
    n: usize,
    chunk: usize,
    kind: &str,
    draft_limit: Option<usize>,
    stop_ids: Option<&[u32]>,
) -> Result<(Vec<u32>, Vec<WinStats>, f64, f64, SpecStats)> {
    use crate::draft::{Calibration, DraftConfig, Global, Session};
    e.reset();
    let is_mtp = kind == "mtp" || kind == "hybrid";
    e.set_use_mtp(is_mtp);
    ensure!(!is_mtp || e.has_mtp(), "--draft mtp needs --mtp FILE");
    let t0 = Instant::now();
    let mut cur = e.prefill(ids, chunk, None)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    let cfg = DraftConfig::new(&[(1, 1.0), (2, 1.12), (4, 1.35), (8, 1.9)]);
    let global = Global::new(0, None);
    let cost: Vec<f32> = (0..=MAX_T).map(|k| cfg.cost(k.max(1))).collect();
    let mut sess = Session::new(&cfg, &global, Calibration::default(), cost, ids);
    let oracle: Vec<u32> = match kind.strip_prefix("oracle:") {
        Some(f) => read_ids(f)?,
        None => match kind.strip_prefix("fixed:") {
            Some(r) => read_ids(r.split_once(':').context("fixed:K:FILE")?.1)?,
            None => Vec::new(),
        },
    };
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let vocab = e.hp().n_vocab.min(248_000) as u64;
    let mut out = vec![cur];
    sess.push(cur);
    let mut stats = Vec::new();
    let mut sp = SpecStats {
        by_pos: vec![(0, 0); MAX_T],
        widths: vec![0; MAX_T + 1],
        ..Default::default()
    };
    let mut started = None;
    // Window cost by width (ms, EMA; priors from flash-bench on mew) and the drafting cost.
    let mut wcost: Vec<f64> = vec![10.6, 10.6, 12.9, 15.0, 17.1, 19.2, 21.3, 23.4, 25.5];
    let mut mtp_cost = 2.5f64;
    // Acceptance calibration by draft-probability bucket: (accepted, tried) with a prior.
    let mut calib: Vec<(f64, f64)> = (0..10)
        .map(|b| {
            let p = (b as f64 + 0.5) / 10.0;
            let a = if p >= 0.9 {
                0.985
            } else if p >= 0.5 {
                0.6 + (p - 0.5) * 0.8
            } else {
                0.3 + p * 0.6
            };
            (4.0 * a, 4.0)
        })
        .collect();
    let mut mtp_probs: Vec<f32> = Vec::new();
    let fixed_gate = std::env::var("TANG_FLASH_GATE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    // Thinking spans: the prompt ends inside `<think>` (248068) until `</think>` (248069).
    let think_end = 248_069u32;
    let mut in_think = ids
        .iter()
        .rev()
        .take(8)
        .find(|&&x| x == 248_068 || x == think_end)
        == Some(&248_068);
    let think_room: usize = std::env::var("TANG_FLASH_THINK_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_T);
    // TANG_FLASH_STOP_IDS=a,b: stop once any of these tokens is generated (data generation).
    let stop: Vec<u32> = stop_ids.map(|ids| ids.to_vec()).unwrap_or_else(|| {
        std::env::var("TANG_FLASH_STOP_IDS")
            .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
            .unwrap_or_default()
    });
    let mut stop_seen = 0;
    while out.len() < n {
        if !stop.is_empty() {
            if out[stop_seen..].iter().any(|x| stop.contains(x)) {
                break;
            }
            stop_seen = out.len();
        }
        if out.len() >= 2 && started.is_none() {
            started = Some(Instant::now());
            stats.clear();
            sp = SpecStats {
                by_pos: vec![(0, 0); MAX_T],
                widths: vec![0; MAX_T + 1],
                ..Default::default()
            };
        }
        let mut room = (MAX_T - 1).min(n - out.len());
        if let Some(limit) = draft_limit {
            room = room.min(limit);
        }
        if in_think {
            room = room.min(think_room);
        }
        let (drafts, draft) = match kind {
            "suffix" => {
                let d = sess.propose(room);
                (d.tokens.clone(), Some(d))
            }
            "mtp" | "hybrid" => {
                // Price each draft length: expected tokens (1 + Σ cumulative acceptance) per
                // ms of window (measured EMA per width) plus drafting, keep the best. The
                // acceptance of a draft is estimated from its probability by an online
                // calibration (10 buckets, prior from the truth track's measurements).
                let mut best = (0usize, 1.0 / (wcost[1] + mtp_cost));
                let (mut cum, mut exp) = (1.0f64, 1.0f64);
                for (k, &(_, p)) in e.mtp_last().iter().enumerate().take(room) {
                    let b = ((p * 10.0) as usize).min(9);
                    cum *= calib[b].0 / calib[b].1;
                    exp += cum;
                    let rate = exp / (wcost[k + 2] + mtp_cost);
                    if rate > best.1 {
                        best = (k + 1, rate);
                    }
                }
                let d: Vec<u32> = e.mtp_last().iter().take(best.0).map(|x| x.0).collect();
                mtp_probs = e.mtp_last().iter().take(best.0).map(|x| x.1).collect();
                // Hybrid: take the suffix drafter's longer proposal when it agrees with the
                // MTP's first draft (a long earlier match: quoted text, code being repeated).
                let sd = if kind == "hybrid" {
                    sess.propose(room)
                } else {
                    Default::default()
                };
                if !sd.tokens.is_empty()
                    && e.mtp_last().first().is_some_and(|x| x.0 == sd.tokens[0])
                    && sd.tokens.len() > d.len()
                {
                    mtp_probs.clear();
                    sp.suffix_windows += 1;
                    (sd.tokens.clone(), Some(sd))
                } else {
                    (d, None)
                }
            }
            "wrong" => {
                // Forced-wrong: random tokens (and sometimes a true prefix from an oracle).
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let k = 1 + (rng % room.max(1) as u64) as usize;
                (
                    (0..k.min(room))
                        .map(|i| ((rng >> (8 * i)) % vocab) as u32)
                        .collect(),
                    None,
                )
            }
            _ if !oracle.is_empty() && kind.starts_with("fixed") => {
                // Fixed width: the true continuation, never corrupted (every window T = k + 1).
                let k = kind
                    .split(':')
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3usize);
                (
                    oracle
                        .iter()
                        .skip(out.len())
                        .take(room.min(k))
                        .copied()
                        .collect(),
                    None,
                )
            }
            _ if !oracle.is_empty() => {
                // Oracle: the true continuation, with one token corrupted now and then.
                let at = out.len();
                let mut d: Vec<u32> = oracle.iter().skip(at).take(room).copied().collect();
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if !d.is_empty() && rng % 3 == 0 {
                    let j = (rng >> 8) as usize % d.len();
                    d[j] = d[j].wrapping_add(1);
                }
                (d, None)
            }
            _ => (Vec::new(), None),
        };
        let drafts = match (kind, fixed_gate) {
            ("mtp" | "hybrid", Some(g)) => {
                // Fixed gate (TANG_FLASH_GATE=0.5): keep drafts while p >= g.
                let d: Vec<u32> = e
                    .mtp_last()
                    .iter()
                    .take_while(|x| x.1 >= g)
                    .take(room)
                    .map(|x| x.0)
                    .collect();
                mtp_probs = e.mtp_last().iter().take(d.len()).map(|x| x.1).collect();
                d
            }
            _ => drafts,
        };
        let kept = e.verify(cur, &drafts)?;
        let acc = kept.len() - 1;
        if is_mtp {
            for (j, &p) in mtp_probs.iter().enumerate().take(acc + 1) {
                let b = ((p * 10.0) as usize).min(9);
                calib[b].1 += 1.0;
                if j < acc {
                    calib[b].0 += 1.0;
                }
            }
            let w = drafts.len() + 1;
            wcost[w] = 0.9 * wcost[w] + 0.1 * e.last().wall_ms;
            mtp_cost = 0.9 * mtp_cost + 0.1 * e.last_mtp_ms();
        }
        for (j, slot) in sp.by_pos.iter_mut().enumerate().take(drafts.len()) {
            slot.0 += 1;
            if j < acc {
                slot.1 += 1;
            }
        }
        {
            let (bp, wd) = if in_think {
                (&mut sp.by_pos_think, &mut sp.widths_think)
            } else {
                (&mut sp.by_pos_answer, &mut sp.widths_answer)
            };
            bp.resize(MAX_T, (0, 0));
            wd.resize(MAX_T + 1, 0);
            wd[drafts.len() + 1] += 1;
            for (j, slot) in bp.iter_mut().enumerate().take(drafts.len()) {
                slot.0 += 1;
                if j < acc {
                    slot.1 += 1;
                }
            }
        }
        if let Some(d) = &draft {
            for j in 0..d.tokens.len().min(acc + 1) {
                sess.calib.observe(d.match_len, d.shares[j], j < acc);
            }
        }
        sp.windows += 1;
        sp.mtp_ms += e.last_mtp_ms();
        sp.commit_ms += e.last_commit_ms();
        sp.mtp_gpu_ms += e.last_mtp_gpu_ms();
        sp.widths[drafts.len() + 1] += 1;
        sp.tokens += kept.len();
        stats.push(e.last());
        for &k in &kept {
            sess.push(k);
            if k == think_end {
                in_think = false;
            }
        }
        out.extend_from_slice(&kept);
        cur = *kept.last().unwrap();
    }
    let secs = started.map_or(0.0, |s| s.elapsed().as_secs_f64());
    out.truncate(n);
    Ok((out, stats, prefill_s, secs, sp))
}

fn spec_report(sp: &SpecStats) -> String {
    let mut s = format!("suffix drafts in {} windows; ", sp.suffix_windows);
    s += &format!(
        "mtp {:.2} ms/window (waiting for the commit {:.2}, MTP graph on the GPU {:.2}); ",
        sp.mtp_ms / sp.windows.max(1) as f64,
        sp.commit_ms / sp.windows.max(1) as f64,
        sp.mtp_gpu_ms / sp.windows.max(1) as f64
    );
    s += &format!(
        "{} windows, {:.2} tokens/window; widths {:?}; accepted by draft position:",
        sp.windows,
        sp.tokens as f64 / sp.windows.max(1) as f64,
        sp.widths
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(w, c)| format!("T{w}:{c}"))
            .collect::<Vec<_>>()
    );
    for (j, &(p, a)) in sp.by_pos.iter().enumerate() {
        if p > 0 {
            s += &format!(
                " d{}={}/{} ({:.0}%)",
                j + 1,
                a,
                p,
                100.0 * a as f64 / p as f64
            );
        }
    }
    for (name, bp, wd) in [
        ("thinking", &sp.by_pos_think, &sp.widths_think),
        ("answer", &sp.by_pos_answer, &sp.widths_answer),
    ] {
        let w: usize = wd.iter().sum();
        if w == 0 {
            continue;
        }
        let acc: usize = bp.iter().map(|x| x.1).sum();
        s += &format!("\n                     {name}: {w} windows, {:.2} tokens/window, widths {:?}, accepted", (w + acc) as f64 / w as f64,
            wd.iter().enumerate().filter(|(_, &c)| c > 0).map(|(t, c)| format!("T{t}:{c}")).collect::<Vec<_>>());
        for (j, &(p, a)) in bp.iter().enumerate() {
            if p > 0 {
                s += &format!(" d{}={:.0}%", j + 1, 100.0 * a as f64 / p as f64);
            }
        }
    }
    s
}

fn mean(stats: &[WinStats]) -> WinStats {
    let n = stats.len().max(1) as f64;
    let mut m = WinStats {
        t: stats.first().map_or(1, |s| s.t),
        ..Default::default()
    };
    for s in stats {
        m.wall_ms += s.wall_ms / n;
        m.host_prep_ms += s.host_prep_ms / n;
        m.plan_ms += s.plan_ms / n;
        m.cpu_ms += s.cpu_ms / n;
        m.gpu_wait_a_ms += s.gpu_wait_a_ms / n;
        m.gpu_wait_b_ms += s.gpu_wait_b_ms / n;
        m.gpu_ms += s.gpu_ms / n;
        m.serve_ms += s.serve_ms / n;
        m.commit_host_ms += s.commit_host_ms / n;
        m.boundary_ms += s.boundary_ms / n;
        m.drain_ms += s.drain_ms / n;
        m.post_ms += s.post_ms / n;
        m.routed += s.routed;
        m.distinct += s.distinct;
        m.missed += s.missed;
        m.pcie += s.pcie;
        m.skipped += s.skipped;
        m.swaps += s.swaps;
    }
    m
}

pub fn generate(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let ids = prompt_ids(&a)?;
    let mut e = load(&a)?;
    // Warm-up: capture every graph (window sizes, commits, MTP cell counts) before timing.
    e.warm()?;
    // After the warm-up, whose throwaway sequence must not land in the training dump.
    if let Some(d) = &a.dump_mtp_train {
        e.dump_mtp_train(d)?;
    }
    let (out, stats, prefill_s, secs, sp) = decode_spec(&mut e, &ids, a.n, a.chunk, &a.draft)?;
    let vocab = e.vocab()?;
    println!(
        "prompt: {} tokens, prefill {:.2} s ({:.0} tok/s)",
        ids.len(),
        prefill_s,
        ids.len() as f64 / prefill_s
    );
    println!(
        "ids: {}",
        out.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if !a.quiet {
        let g = crate::gguf::Gguf::open(&a.path)?;
        match super::tokenize::FlashTokenizer::from_gguf(&g).and_then(|t| t.decode(&out)) {
            Ok(text) => println!("text: {text}"),
            Err(_) => println!("text: {}", detok(&vocab, &out)),
        }
    }
    println!(
        "decode: {} tokens in {:.3} s = {:.1} tok/s (drafts {}, temp {}, graphs {})",
        sp.tokens,
        secs,
        sp.tokens as f64 / secs,
        a.draft,
        a.temp,
        e.use_graphs
    );
    println!("speculation: {}", spec_report(&sp));
    println!("mean window: {}", split_line(&mean(&stats)));
    if let Some(cs) = e.cache_stats() {
        println!(
            "cache: hit rate {:.3} over the run ({} hits, {} misses, {} swaps)",
            cs.hit_rate(),
            cs.hits,
            cs.misses,
            cs.swaps
        );
    }
    let (h, r) = e.ngram_stats();
    println!("n-gram rows: {r} reads, {h} cache hits");
    if let Some(p) = &a.out {
        std::fs::write(
            p,
            out.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
    }
    if let Some(p) = &a.dump_routing {
        e.save_routing(p)?;
        println!("routing counts -> {}", p.display());
    }
    Ok(())
}

/// Compare the engine's logits with a `flash-ref` JSONL (top-K logprobs per position) over a
/// prompt, and, with `--dump DIR` (a `flash-ref --dump` of the same ids), the last position's
/// per-layer intermediates.
pub fn parity(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let ids = prompt_ids(&a)?;
    let mut e = load_model(&a)?;
    let e = &mut *e;
    e.reset();
    let refs: Vec<(usize, Vec<(u32, f64)>)> = match &a.reference {
        Some(p) => std::fs::read_to_string(p)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| -> Result<(usize, Vec<(u32, f64)>)> {
                let v: serde_json::Value = serde_json::from_str(l)?;
                let pos = v["pos"].as_u64().context("pos")? as usize;
                let top = v["top"]
                    .as_array()
                    .context("top")?
                    .iter()
                    .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_f64().unwrap()))
                    .collect();
                Ok((pos, top))
            })
            .collect::<Result<_>>()?,
        None => Vec::new(),
    };
    let want: std::collections::HashMap<usize, &Vec<(u32, f64)>> =
        refs.iter().map(|(p, t)| (*p, t)).collect();
    let (mut n, mut agree, mut kl_sum) = (0usize, 0usize, 0f64);
    let mut kls = Vec::new();
    let mut misses = Vec::new();
    let mut saved = String::new();
    let save = a.save.is_some();
    let mut cb = |pos0: usize, t: usize, logits: &[f32]| {
        let v = logits.len() / t;
        for i in 0..t {
            let pos = pos0 + i;
            if save {
                let top = top_logprobs(&logits[i * v..(i + 1) * v], 40);
                let s: Vec<String> = top.iter().map(|(i, lp)| format!("[{i},{lp:.6}]")).collect();
                saved += &format!("{{\"pos\":{pos},\"top\":[{}]}}\n", s.join(","));
            }
            let Some(r) = want.get(&pos) else { continue };
            let lg = &logits[i * v..(i + 1) * v];
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + lg.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
            let mine = top_logprobs(lg, 1)[0].0;
            let (mut kl, mut pr, mut qr) = (0f64, 0f64, 0f64);
            for &(id, lp) in r.iter() {
                let p = lp.exp();
                let lq = lg[id as usize] as f64 - lse;
                kl += p * (lp - lq);
                pr += p;
                qr += lq.exp();
            }
            let (pr, qr) = ((1.0 - pr).max(1e-12), (1.0 - qr).max(1e-12));
            kl += pr * (pr.ln() - qr.ln());
            n += 1;
            if mine == r[0].0 {
                agree += 1;
            } else {
                misses.push((pos, r[0].0, mine, r[0].1, r.get(1).map_or(0.0, |x| x.1)));
            }
            kl_sum += kl;
            kls.push(kl);
        }
    };
    let t0 = Instant::now();
    let probe_last = a.dump.is_some();
    let total = ids.len();
    let body = if probe_last { total - 1 } else { total };
    if body > 0 {
        e.prefill(&ids[..body], a.chunk, Some(&mut cb))?;
    }
    let mut probe = Probe::default();
    if probe_last {
        e.tokens_mut().push(ids[total - 1]);
        e.window(total - 1, 1, Some(&mut probe))?;
        cb(total - 1, 1, &e.logits(1));
    }
    let secs = t0.elapsed().as_secs_f64();
    if let Some(p) = &a.save {
        std::fs::write(p, &saved)?;
    }
    kls.sort_by(|a, b| a.total_cmp(b));
    if n > 0 {
        println!(
            "logits vs reference: {n} positions, top-1 agree {agree}/{n} ({:.1}%), KL mean {:.4} median {:.5} p99 {:.4} max {:.4}  ({:.1} s, chunk {})",
            100.0 * agree as f64 / n as f64,
            kl_sum / n as f64,
            kls[n / 2],
            kls[(n * 99 / 100).min(n - 1)],
            kls[n - 1],
            secs,
            a.chunk
        );
        for (pos, want, got, lp0, lp1) in misses.iter().take(12) {
            println!(
                "  top-1 differs at {pos}: ref {want} ({lp0:.3}, runner-up {lp1:.3}), engine {got}"
            );
        }
    }
    if let Some(dir) = &a.dump {
        compare_dump(e, dir, &probe)?;
    }
    Ok(())
}

fn read_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(dir.join(format!("{name}.f32")))?;
    Ok(b.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_u32(dir: &Path, name: &str) -> Result<Vec<u32>> {
    let b = std::fs::read(dir.join(format!("{name}.u32")))?;
    Ok(b.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

fn compare_dump(e: &dyn FlashModel, dir: &Path, p: &Probe) -> Result<()> {
    println!("per layer, last position (relative L2 error vs flash-ref; router overlap of 10; QSA selection):");
    for l in 0..e.hp().n_layer {
        let pm = read_f32(dir, &format!("L{l:02}.post_mixer"))?;
        let po = read_f32(dir, &format!("L{l:02}.post_moe"))?;
        let mo = read_f32(dir, &format!("L{l:02}.mixer_out"))?;
        let eo = read_f32(dir, &format!("L{l:02}.moe_out"))?;
        let rid = read_u32(dir, &format!("L{l:02}.router_ids"))?;
        let overlap = p.router_ids[l].iter().filter(|x| rid.contains(x)).count();
        let order = p.router_ids[l] == rid;
        let sel = match &p.qsa_sel[l] {
            Some(s) => {
                let want = read_u32(dir, &format!("L{l:02}.qsa_selected")).unwrap_or_default();
                let same = *s == want;
                let common = s.iter().filter(|c| want.binary_search(c).is_ok()).count();
                format!(
                    " qsa sel {} ({} of {} common)",
                    if same { "identical" } else { "DIFFERS" },
                    common,
                    want.len()
                )
            }
            None => String::new(),
        };
        println!(
            "  L{l:02}: mixer_out {:.4} post_mixer {:.4} moe_out {:.4} post_moe {:.4} router {overlap}/10{}{sel}",
            rel(&p.mixer_out[l], &mo),
            rel(&p.post_mixer[l], &pm),
            rel(&p.moe_out[l], &eo),
            rel(&p.post_moe[l], &po),
            if order { " (same order)" } else { "" },
        );
    }
    let fx = read_f32(dir, "final_x")?;
    println!("  final_x {:.4}", rel(&p.final_x, &fx));
    let lg = read_f32(dir, "logits")?;
    println!("  logits {:.4}", rel(&e.logits(1), &lg));
    Ok(())
}

/// Decode benchmark: prefill `--ctx` tokens of the prompt (cycled if short), then `-n` greedy
/// T=1 tokens; prints tok/s and the per-window split.
pub fn bench(args: &[String]) -> Result<()> {
    let mut a = parse(args)?;
    a.opts.split = true;
    let mut ids = prompt_ids(&a)?;
    if let Some(c) = a.ctx {
        let base = ids.clone();
        while ids.len() < c {
            ids.extend_from_slice(&base);
        }
        ids.truncate(c);
    }
    if a.opts.max_ctx < ids.len() + a.n + 8 {
        a.opts.max_ctx = (ids.len() + a.n + 8).next_multiple_of(1024);
    }
    if coder::is_qwen3next(&a.path)? {
        return bench_coder(&a, &ids);
    }
    let mut e = load(&a)?;
    // Warm-up: capture every graph (window sizes, commits, MTP cell counts) before timing.
    e.warm()?;
    let (out, stats, prefill_s, secs, sp) = decode_spec(&mut e, &ids, a.n, a.chunk, &a.draft)?;
    let timed = &stats[..];
    let m = mean(timed);
    let tps = sp.tokens as f64 / secs;
    let walls: Vec<f64> = {
        let mut w: Vec<f64> = timed.iter().map(|s| s.wall_ms).collect();
        w.sort_by(|a, b| a.total_cmp(b));
        w
    };
    println!(
        "flash-bench: context {} + {} new tokens, temp {}, drafts {}, graphs {}",
        ids.len(),
        a.n,
        a.temp,
        a.draft,
        e.use_graphs
    );
    println!(
        "  prefill            {:8.2} s ({:.0} tok/s, chunk {})",
        prefill_s,
        ids.len() as f64 / prefill_s,
        a.chunk
    );
    println!(
        "  decode             {:8.1} tok/s ({} tokens in {:.3} s)",
        tps, sp.tokens, secs
    );
    println!("  speculation        {}", spec_report(&sp));
    println!(
        "  window wall        {:8.3} ms mean, {:.3} median, {:.3} p90",
        m.wall_ms,
        walls[walls.len() / 2],
        walls[walls.len() * 9 / 10]
    );
    println!("  GPU (graph)        {:8.3} ms", m.gpu_ms);
    println!(
        "    waiting for plan {:8.3} ms  (handoff A)",
        m.gpu_wait_a_ms
    );
    println!(
        "    waiting for CPU  {:8.3} ms  (CPU-miss time exposed)",
        m.gpu_wait_b_ms
    );
    println!(
        "  host prep          {:8.3} ms  (embedding; n-gram rows are read during layer 0)",
        m.host_prep_ms
    );
    println!(
        "  host: launch..served {:6.3} ms, ..drained {:.3} ms, boundary+tables {:.3} ms",
        m.serve_ms, m.drain_ms, m.post_ms
    );
    println!("  host plan          {:8.3} ms", m.plan_ms);
    println!(
        "  after window: commit launch {:.3} ms, cache boundary + tables {:.3} ms",
        m.commit_host_ms, m.boundary_ms
    );
    println!(
        "  host CPU experts   {:8.3} ms ({:.1} of 48 layers/window without a doorbell wait)",
        m.cpu_ms,
        m.skipped as f64 / timed.len().max(1) as f64
    );
    println!(
        "  experts            {:.1} distinct/window, {:.2} on the CPU/window, {:.2} over PCIe/window, VRAM hit rate {:.3}, swaps {}",
        m.distinct as f64 / timed.len() as f64,
        m.missed as f64 / timed.len() as f64,
        m.pcie as f64 / timed.len() as f64,
        1.0 - (m.missed + m.pcie) as f64 / m.distinct.max(1) as f64,
        m.swaps
    );
    let (h, r) = e.ngram_stats();
    println!("  n-gram rows        {r} reads, {h} cache hits");
    println!(
        "  first tokens: {}",
        out.iter()
            .take(16)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if a.prompt.is_some() && !a.quiet {
        let g = crate::gguf::Gguf::open(&a.path)?;
        let t = super::tokenize::FlashTokenizer::from_gguf(&g)?;
        let text = t.decode(&out)?;
        println!(
            "  text: {}",
            text.chars()
                .take(600)
                .collect::<String>()
                .replace('\n', "\\n")
        );
    }
    if let Some(p) = &a.out {
        std::fs::write(
            p,
            out.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
    }
    if let Some(p) = &a.dump_routing {
        e.save_routing(p)?;
    }
    let _ = HIDDEN;
    Ok(())
}

/// `flash-bench` for a qwen3next model: decode tok/s at the prompt's context and its share of
/// the speed of light (bytes a step reads from VRAM over `--bw`).
fn bench_coder(a: &Args, ids: &[u32]) -> Result<()> {
    let mut e = load_coder(a)?;
    let (out, stats, prefill_s, secs, sp) = decode_spec(&mut e, ids, a.n, a.chunk, &a.draft)?;
    let m = mean(&stats);
    let tps = sp.tokens as f64 / secs;
    let mut walls: Vec<f64> = stats.iter().map(|s| s.wall_ms).collect();
    walls.sort_by(|a, b| a.total_cmp(b));
    let ctx = ids.len() + a.n / 2;
    let bytes = e.step_bytes(ctx);
    let sol = a.bw * 1e9 / bytes;
    // A T=1 window's tokens per second against the speed of light (drafts aside).
    let t1_tps = 1e3 / walls[walls.len() / 2];
    println!(
        "flash-bench (qwen3next): context {} + {} new tokens, temp {}, drafts {}, graphs {}",
        ids.len(),
        a.n,
        a.temp,
        a.draft,
        e.use_graphs
    );
    println!(
        "  prefill            {:8.2} s ({:.0} tok/s, chunk {})",
        prefill_s,
        ids.len() as f64 / prefill_s,
        a.chunk
    );
    println!(
        "  decode             {:8.1} tok/s ({} tokens in {:.3} s)",
        tps, sp.tokens, secs
    );
    println!("  speculation        {}", spec_report(&sp));
    println!(
        "  window wall        {:8.3} ms mean, {:.3} median, {:.3} p90",
        m.wall_ms,
        walls[walls.len() / 2],
        walls[walls.len() * 9 / 10]
    );
    println!(
        "  speed of light     {:8.1} tok/s ({:.2} GB a step at context {ctx}, {:.0} GB/s); \
         median window {:.1} tok/s = {:.1}% of it",
        sol,
        bytes / 1e9,
        a.bw,
        t1_tps,
        100.0 * t1_tps / sol
    );
    println!(
        "  first tokens: {}",
        out.iter()
            .take(16)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if std::env::var("TANG_CODER_STAGES").is_ok_and(|v| v == "1") {
        let st = e.stage_sweep()?;
        let win = st[0].1;
        println!("  stages (T=1 graph replay of each stage alone, µs a step; GB/s achieved):");
        for (name, us, bytes) in &st {
            println!(
                "    {name:14} {us:8.1} µs {:5.1}%  {}",
                100.0 * us / win,
                if *bytes > 0.0 {
                    format!("{:6.0} GB/s", bytes / us / 1e3)
                } else {
                    String::new()
                }
            );
        }
        let sum: f64 = st[1..].iter().map(|x| x.1).sum();
        println!("    (sum of stages {sum:.1} µs vs window {win:.1} µs)");
    }
    if let Some(r) = e.prof_report() {
        println!("  GPU time by stage (TANG_CODER_PROF, eager):\n{r}");
    }
    Ok(())
}

/// `flash-aa`: Artificial-Analysis-style decode speed for a qwen3next model with one load.
/// For each `--ids-file` prompt (comma-separated list), `-n` output tokens, drafts off, suffix
/// and (with `--mtp FILE`) MTP: output tok/s after the first token. Then, on the first prompt without drafts, each
/// kernel variant in `TANG_CODER_AB` (`;`-separated specs, see `Coder::set_variants`), and with
/// `TANG_CODER_STAGES=1` the stage sweep. `--out DIR` saves each run's ids.
pub fn aa(args: &[String]) -> Result<()> {
    let mut it: Vec<String> = Vec::new();
    let mut files = Vec::new();
    let mut k = 0;
    while k < args.len() {
        if args[k] == "--ids-files" {
            files = args[k + 1].split(',').map(String::from).collect::<Vec<_>>();
            k += 2;
        } else {
            it.push(args[k].clone());
            k += 1;
        }
    }
    let mut a = parse(&it)?;
    ensure!(!files.is_empty(), "--ids-files A,B,..");
    a.opts.max_ctx = a.opts.max_ctx.max(4096);
    let mut e = load_coder(&a)?;
    let mut row = |e: &mut Coder, name: &str, ids: &[u32], draft: &str| -> Result<f64> {
        let (out, _, prefill_s, secs, sp) = decode_spec(e, ids, a.n, a.chunk, draft)?;
        let tps = sp.tokens as f64 / secs;
        println!(
            "  {name:12} prompt {:5} drafts {draft:7} {tps:7.1} tok/s  ({} tokens/window, prefill {:.2} s)",
            ids.len(),
            format!("{:.2}", sp.tokens as f64 / sp.windows.max(1) as f64),
            prefill_s
        );
        if let Some(d) = &a.out {
            std::fs::create_dir_all(d)?;
            std::fs::write(
                d.join(format!("{name}.{draft}.out")),
                out.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )?;
        }
        Ok(tps)
    };
    println!("flash-aa: {} output tokens a prompt, single stream", a.n);
    let mut prompts = Vec::new();
    for f in &files {
        let name = Path::new(f)
            .file_stem()
            .map_or(f.clone(), |s| s.to_string_lossy().into_owned());
        prompts.push((name, read_ids(f)?));
    }
    let drafts: &[&str] = if e.has_mtp() {
        &["none", "suffix", "mtp"]
    } else {
        &["none", "suffix"]
    };
    for (name, ids) in &prompts {
        for &draft in drafts {
            row(&mut e, name, ids, draft)?;
        }
    }
    let ab = std::env::var("TANG_CODER_AB").unwrap_or_default();
    if !ab.is_empty() {
        println!("kernel variants (first prompt, no drafts):");
        for spec in std::iter::once("").chain(ab.split(';')) {
            e.set_variants(spec)?;
            let (name, ids) = &prompts[0];
            // one untimed run captures the graphs
            decode_spec(&mut e, ids, 16, a.chunk, "none")?;
            print!("  [{:>16}]", if spec.is_empty() { "default" } else { spec });
            row(&mut e, name, ids, "none")?;
        }
        e.set_variants("")?;
    }
    if std::env::var("TANG_CODER_STAGES").is_ok_and(|v| v == "1") {
        let (_, ids) = &prompts[0];
        decode_spec(&mut e, ids, 64, a.chunk, "none")?;
        let st = e.stage_sweep()?;
        let win = st[0].1;
        println!(
            "stages at context {} (T=1, each stage alone in a graph, µs a step):",
            ids.len() + 64
        );
        for (name, us, bytes) in &st {
            println!(
                "  {name:14} {us:8.1} µs {:5.1}%  {}",
                100.0 * us / win,
                if *bytes > 0.0 {
                    format!("{:6.0} GB/s", bytes / us / 1e3)
                } else {
                    String::new()
                }
            );
        }
        let sum: f64 = st[1..].iter().map(|x| x.1).sum();
        println!("  (sum of stages {sum:.1} µs vs window {win:.1} µs)");
    }
    Ok(())
}

/// `flash-spec-test`: with one load, decode `-n` tokens without drafts, then with suffix drafts,
/// forced-wrong drafts and an oracle (the no-draft output, corrupted now and then), greedy and
/// at `--temp` (default 0.8), and check every output equals the no-draft one.
pub fn spec_test(args: &[String]) -> Result<()> {
    let mut a = parse(args)?;
    let ids = prompt_ids(&a)?;
    if a.temp == 0.0 {
        a.temp = 0.8;
    }
    let mut e = load_model(&a)?;
    let e = &mut *e;
    let mut ok = true;
    for temp in [0.0, a.temp] {
        e.set_sampling(temp, a.seed);
        let (base, _, _, s0, sp0) = decode_spec(e, &ids, a.n, a.chunk, "none")?;
        let f = std::env::temp_dir().join(format!("flash-oracle-{}.ids", std::process::id()));
        std::fs::write(
            &f,
            base.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
        println!(
            "temp {temp}: none   {:.1} tok/s  first {:?}",
            sp0.tokens as f64 / s0,
            &base[..base.len().min(12)]
        );
        let mut kinds = vec![
            "suffix".to_string(),
            "wrong".to_string(),
            format!("oracle:{}", f.display()),
        ];
        if e.has_mtp() {
            kinds.insert(0, "mtp".into());
            kinds.insert(1, "hybrid".into());
        }
        for kind in kinds {
            let (out, stats, _, secs, sp) = decode_spec(e, &ids, a.n, a.chunk, &kind)?;
            let same = out == base;
            ok &= same;
            let first_diff = out.iter().zip(&base).position(|(x, y)| x != y);
            println!(
                "temp {temp}: {:<7} {} {:.1} tok/s, {} | mean window {:.2} ms{}",
                kind.split(':').next().unwrap(),
                if same { "SAME" } else { "DIFFERS" },
                sp.tokens as f64 / secs,
                spec_report(&sp),
                mean(&stats).wall_ms,
                first_diff.map_or(String::new(), |i| format!(" (first difference at {i})"))
            );
        }
        let _ = std::fs::remove_file(&f);
    }
    println!(
        "{}",
        if ok {
            "spec test: PASS"
        } else {
            "spec test: FAIL"
        }
    );
    Ok(())
}

/// `flash-wide-check` (qwen3next): wide prefill against normal prefill. The prompt cycled to
/// `--ctx` tokens (default 1000) is prefilled into one slot with `prefill_slot` (windows of
/// `MAX_T`) and into another with `prefill_wide` (windows of `TANG_CODER_WIDE` rows, default
/// 256), then each decodes `-n` greedy tokens: tokens and logits must be bitwise equal. Prints
/// both prefills' times (the wide one twice: the first run warms it).
pub fn wide_check(args: &[String]) -> Result<()> {
    let mut a = parse(args)?;
    let base = prompt_ids(&a)?;
    // TANG_WIDE_CTXS=1000,8000,...: several prompt lengths on one load.
    let lens: Vec<usize> = match std::env::var("TANG_WIDE_CTXS") {
        Ok(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => vec![a.ctx.unwrap_or(1000)],
    };
    let longest = lens.iter().copied().max().unwrap_or(1000);
    let steps = a.n.max(4);
    if a.opts.max_ctx < longest + steps + 8 {
        a.opts.max_ctx = (longest + steps + 8).next_multiple_of(1024);
    }
    let rows: usize = std::env::var("TANG_CODER_WIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut e = load_coder(&a)?;
    ensure!(e.temperature == 0.0, "wide-check is greedy");
    // Free-running greedy decode, or (`forced`) fed these tokens instead of its own picks.
    let decode = |e: &mut Coder,
                  slot: usize,
                  mut next: u32,
                  forced: Option<&[u32]>|
     -> Result<(Vec<u32>, Vec<f32>)> {
        let (mut toks, mut lg) = (Vec::new(), Vec::new());
        for i in 0..steps {
            let got = e.decode_batch(&[(slot, next)])?[0];
            lg.extend_from_slice(&e.logits(1));
            toks.push(got);
            next = forced.map_or(got, |f| f[i]);
        }
        Ok((toks, lg))
    };
    let t0 = std::time::Instant::now();
    e.enable_wide(rows)?;
    println!(
        "wide kernels ({rows} rows) built in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let mut all_ok = true;
    for len in lens {
        let ids: Vec<u32> = base.iter().cycle().take(len).copied().collect();
        let s1 = e.alloc_slot()?;
        let t0 = std::time::Instant::now();
        let n1 = e.prefill_slot(s1, &ids)?;
        let narrow = t0.elapsed().as_secs_f64();
        let (t1, l1) = decode(&mut e, s1, n1, None)?;
        e.free_slot(s1);
        let mut wide = Vec::new();
        let mut got = None;
        let mut forced = None;
        for _ in 0..2 {
            let s2 = e.alloc_slot()?;
            let t0 = std::time::Instant::now();
            let n2 = e.prefill_wide(s2, &ids)?;
            wide.push(t0.elapsed().as_secs_f64());
            if got.is_none() {
                got = Some((n2, decode(&mut e, s2, n2, None)?));
            } else {
                // Teacher-forced on the narrow run's tokens: the logits measure the state
                // (K/V, GDN) the wide prefill left against narrow's.
                forced = Some(decode(&mut e, s2, n1, Some(&t1))?.1);
            }
            e.free_slot(s2);
        }
        let (n2, (t2, l2)) = got.unwrap();
        if let Some(r) = e.prof_report() {
            println!("  GPU time by stage (TANG_CODER_PROF, narrow + wide prefills):\n{r}");
        }
        let bits = l1
            .iter()
            .zip(&l2)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let v = e.hp.n_vocab;
        let kl = |p: &[f32], q: &[f32]| -> f64 {
            let ls = |l: &[f32]| -> Vec<f64> {
                let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
                l.iter().map(|&x| x as f64 - z).collect()
            };
            (0..p.len() / v)
                .map(|r| {
                    let (a, b) = (ls(&p[r * v..(r + 1) * v]), ls(&q[r * v..(r + 1) * v]));
                    a.iter()
                        .zip(&b)
                        .map(|(x, y)| x.exp() * (x - y))
                        .sum::<f64>()
                })
                .fold(0.0, f64::max)
        };
        let max_kl = forced.as_ref().map_or(0.0, |f| kl(&l1, f));
        // Bitwise (grouped decode kernels), or the GEMM bar: same greedy tokens, KL <= 1e-3.
        let ok = n1 == n2 && t1 == t2 && (bits == 0 || max_kl <= 1e-3);
        println!("  teacher-forced max KL(narrow || wide) over {steps} steps: {max_kl:.2e}");
        println!(
            "prefill {len} tokens: narrow {:.2} s ({:.0} tok/s), wide {:.2} s cold / {:.2} s warm ({:.0} tok/s)",
            narrow,
            len as f64 / narrow,
            wide[0],
            wide[1],
            len as f64 / wide[1]
        );
        println!(
            "wide-check {len}: {} (next token {}, {steps} decoded tokens {}, logits differ in {bits} of {})",
            if !ok { "FAIL" } else if bits == 0 { "PASS (bitwise)" } else { "PASS (greedy equal, KL <= 1e-3)" },
            if n1 == n2 { "equal" } else { "DIFFERS" },
            if t1 == t2 { "equal" } else { "DIFFER" },
            l1.len()
        );
        all_ok &= ok;
    }
    ensure!(all_ok, "wide-check failed");
    Ok(())
}

/// `flash-batch-check` (qwen3next): is a sequence's output independent of the batch it decodes
/// in? Up to 8 sequences (the prompt truncated by different amounts, so their positions differ)
/// each decode `-n` greedy tokens alone, then all together in one batched step per token, with
/// the batch order reversed every other step. Tokens and logits must be bitwise equal. Then
/// decode speed at 1, 2, 4 and 8 sequences (tok/s aggregate and per stream).
pub fn batch_check(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let ids = prompt_ids(&a)?;
    ensure!(ids.len() > 32, "need a longer prompt");
    let mut e = load_coder(&a)?;
    ensure!(e.temperature == 0.0, "batch-check is greedy");
    let v = e.hp.n_vocab;
    let nb = super::coder::MAX_T.min(8);
    let seqs: Vec<Vec<u32>> = (0..nb).map(|i| ids[..ids.len() - 3 * i].to_vec()).collect();
    let steps = a.n.max(8);
    // TANG_CODER_MOE_SWEEP=0,2,4,8: repeat everything for each expert-kernel variant on one load.
    // Entries are a variant, with `n` for SiLU unfused (`TANG_CODER_FUSE_SILU=0`): "2,4,4n".
    let sweep: Vec<Option<(usize, bool)>> = match std::env::var("TANG_CODER_MOE_SWEEP") {
        Ok(s) => s
            .split(',')
            .filter_map(|x| {
                let x = x.trim();
                let (num, fuse) = x.strip_suffix('n').map_or((x, true), |n| (n, false));
                num.parse().ok().map(|v| Some((v, fuse)))
            })
            .collect(),
        Err(_) => vec![None],
    };
    let mut all_ok = true;
    for variant in sweep {
        if let Some((m, fuse)) = variant {
            e.set_moe_variant(m, fuse);
            println!("== TANG_CODER_MOE={m} fuse_silu={fuse}");
        }
        // Alone.
        let mut alone: Vec<(Vec<u32>, Vec<f32>)> = Vec::new();
        for s in &seqs {
            let slot = e.alloc_slot()?;
            let mut next = e.prefill_slot(slot, s)?;
            let (mut toks, mut lg) = (Vec::new(), Vec::new());
            for _ in 0..steps {
                let out = e.decode_batch(&[(slot, next)])?;
                lg.extend_from_slice(&e.logits(1));
                next = out[0];
                toks.push(next);
            }
            e.free_slot(slot);
            alone.push((toks, lg));
        }
        // Batched.
        let slots: Vec<usize> = (0..nb).map(|_| e.alloc_slot()).collect::<Result<_>>()?;
        let mut next: Vec<u32> = Vec::new();
        for (i, s) in seqs.iter().enumerate() {
            next.push(e.prefill_slot(slots[i], s)?);
        }
        let mut got: Vec<(Vec<u32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); nb];
        for step in 0..steps {
            let order: Vec<usize> = if step % 2 == 0 {
                (0..nb).collect()
            } else {
                (0..nb).rev().collect()
            };
            let rows: Vec<(usize, u32)> = order.iter().map(|&i| (slots[i], next[i])).collect();
            let out = e.decode_batch(&rows)?;
            let lg = e.logits(nb);
            for (r, &i) in order.iter().enumerate() {
                next[i] = out[r];
                got[i].0.push(out[r]);
                got[i].1.extend_from_slice(&lg[r * v..(r + 1) * v]);
            }
        }
        for &s in &slots {
            e.free_slot(s);
        }
        let mut ok = true;
        for i in 0..nb {
            let tok_eq = alone[i].0 == got[i].0;
            let bits = alone[i]
                .1
                .iter()
                .zip(&got[i].1)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!(
                "seq {i} (prompt {} tokens): tokens {} , logits differ in {bits} of {} entries",
                seqs[i].len(),
                if tok_eq { "identical" } else { "DIFFER" },
                steps * v
            );
            ok &= tok_eq && bits == 0;
        }
        // FNV-1a over every batched logit, to compare runs (e.g. TANG_CODER_GROUPED=0 against the
        // grouped default): equal checksums, equal bits.
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for (_, lg) in &got {
            for x in lg {
                h = (h ^ x.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        println!(
            "batch-check: {} (logits checksum {h:016x})",
            if ok { "PASS (bitwise)" } else { "FAIL" }
        );
        // With an MTP layer: up to 3 sequences in verify windows with their MTP drafts (the
        // window's spare rows shared out) must give the same tokens as each one alone.
        if e.has_mtp() {
            let nm = nb.min(3);
            let slots: Vec<usize> = (0..nm).map(|_| e.alloc_slot()).collect::<Result<_>>()?;
            let mut cur: Vec<u32> = Vec::new();
            for (i, &s) in slots.iter().enumerate() {
                cur.push(e.prefill_slot(s, &seqs[i])?);
            }
            let mut got: Vec<Vec<u32>> = vec![Vec::new(); nm];
            let (mut windows, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
            let t0 = std::time::Instant::now();
            while got.iter().any(|g| g.len() < steps) {
                let live: Vec<usize> = (0..nm).filter(|&i| got[i].len() < steps).collect();
                let spare = (super::coder::MAX_T - live.len()) / live.len();
                let items: Vec<(usize, u32, Vec<u32>)> = live
                    .iter()
                    .map(|&i| {
                        let d: Vec<u32> = e
                            .slot_drafts(slots[i])
                            .iter()
                            .take(spare.min(steps - got[i].len() - 1))
                            .map(|x| x.0)
                            .collect();
                        (slots[i], cur[i], d)
                    })
                    .collect();
                let kept = e.verify_batch(&items)?;
                windows += 1;
                for ((&i, k), it) in live.iter().zip(&kept).zip(&items) {
                    drafted += it.2.len();
                    accepted += k.len() - 1;
                    got[i].extend_from_slice(k);
                    cur[i] = *k.last().unwrap();
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            for &s in &slots {
                e.free_slot(s);
            }
            let mut mok = true;
            for i in 0..nm {
                let same = got[i][..steps] == alone[i].0[..steps];
                mok &= same;
                println!(
                    "mtp seq {i}: tokens {}",
                    if same { "identical" } else { "DIFFER" }
                );
            }
            println!(
                "batch-check mtp ({nm} sequences): {} | {windows} windows, drafts {accepted}/{drafted} accepted, {:.0} tok/s aggregate",
                if mok { "PASS" } else { "FAIL" },
                got.iter().map(|g| g.len()).sum::<usize>() as f64 / dt
            );
            ok &= mok;
        }
        // Speed.
        for b in [1usize, 2, 4, 8].into_iter().filter(|&b| b <= nb) {
            let slots: Vec<usize> = (0..b).map(|_| e.alloc_slot()).collect::<Result<_>>()?;
            let mut next: Vec<u32> = Vec::new();
            for (i, &s) in slots.iter().enumerate() {
                next.push(e.prefill_slot(s, &seqs[i])?);
            }
            let run = |e: &mut Coder, next: &mut Vec<u32>, n: usize| -> Result<()> {
                for _ in 0..n {
                    let rows: Vec<(usize, u32)> = slots
                        .iter()
                        .zip(next.iter())
                        .map(|(&s, &t)| (s, t))
                        .collect();
                    *next = e.decode_batch(&rows)?;
                }
                Ok(())
            };
            run(&mut e, &mut next, 4)?; // capture the graphs
            let t0 = std::time::Instant::now();
            run(&mut e, &mut next, steps)?;
            let dt = t0.elapsed().as_secs_f64();
            let agg = (b * steps) as f64 / dt;
            println!(
                "batch {b}: {:.2} ms a step, {agg:.0} tok/s aggregate, {:.0} tok/s a stream",
                dt * 1e3 / steps as f64,
                agg / b as f64
            );
            for &s in &slots {
                e.free_slot(s);
            }
        }
        all_ok &= ok;
    }
    ensure!(all_ok, "batch-check failed");
    Ok(())
}

/// `flash-tcheck`: is a token's output independent of the window it is computed in? After a
/// shared prefix, runs the next 8 tokens as eight T=1 windows, as one T=8 window, and as one
/// T=8 verify window, each with per-layer probes of the last token, and reports where they
/// first differ bitwise.
pub fn tcheck(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let ids = prompt_ids(&a)?;
    ensure!(ids.len() > 16, "need a longer prompt");
    let mut e = load_model(&a)?;
    let e = &mut *e;
    for l in [0usize, 3] {
        println!("layer {l} ops:");
        e.op_tcheck(l)?;
    }
    let p = ids.len() - 8;
    let v = e.hp().n_vocab;
    let mut runs: Vec<(String, Vec<f32>, Probe)> = Vec::new();
    for mode in ["t1", "t8", "t8-verify"] {
        e.reset();
        e.prefill(&ids[..p], 8, None)?;
        e.tokens_mut().extend_from_slice(&ids[p..]);
        let mut probe = Probe::default();
        let logits = match mode {
            "t1" => {
                for i in 0..7 {
                    e.window(p + i, 1, None)?;
                }
                e.window(p + 7, 1, Some(&mut probe))?;
                e.logits(1)
            }
            "t8" => {
                e.window(p, 8, Some(&mut probe))?;
                e.logits(8)[7 * v..].to_vec()
            }
            _ => {
                e.window_mode(p, 8, true, Some(&mut probe))?;
                e.logits(8)[7 * v..].to_vec()
            }
        };
        runs.push((mode.to_string(), logits, probe));
    }
    let bits = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    for i in 1..runs.len() {
        let (a0, b0) = (&runs[0], &runs[i]);
        println!(
            "{} vs {}: logits differ in {} of {} entries",
            a0.0,
            b0.0,
            bits(&a0.1, &b0.1),
            v
        );
        let mut first = None;
        for l in 0..e.hp().n_layer {
            let d = [
                ("mixer_out", bits(&a0.2.mixer_out[l], &b0.2.mixer_out[l])),
                ("post_mixer", bits(&a0.2.post_mixer[l], &b0.2.post_mixer[l])),
                ("moe_out", bits(&a0.2.moe_out[l], &b0.2.moe_out[l])),
            ];
            if d.iter().any(|x| x.1 > 0) && first.is_none() {
                first = Some(l);
                println!(
                    "  first difference at layer {l}: {d:?}, router ids {:?} vs {:?}",
                    a0.2.router_ids[l], b0.2.router_ids[l]
                );
            }
        }
        if first.is_none() {
            println!("  every probed layer bitwise equal");
        }
    }
    Ok(())
}

/// Reconstruct missing prompt residuals once per corpus, without changing generation dumps.
pub fn mtp_prefix(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == 5 && args[1] == "--data" && args[3] == "--out",
        "flash-mtp-prefix <main.gguf> --data DIR --out DIR"
    );
    let root = Path::new(&args[2]);
    let out = Path::new(&args[4]);
    ensure!(!out.exists(), "prefix output already exists");
    let a = parse(&[args[0].clone(), "--max-ctx".into(), "512".into()])?;
    let mut e = load(&a)?;
    e.warm()?;
    let g = crate::gguf::Gguf::open(&a.path)?;
    let tok = super::tokenize::FlashTokenizer::from_gguf(&g)?;
    let mut paths: Vec<_> = std::fs::read_dir(root)?
        .filter_map(|x| x.ok().map(|x| x.path()))
        .filter(|p| p.join("done").exists())
        .collect();
    paths.sort();
    ensure!(!paths.is_empty(), "no completed sequences");
    std::fs::create_dir(out)?;
    for p in paths {
        let prompt = std::fs::read_to_string(p.join("prompt.txt"))?;
        let ids = tok.encode(&tok.chat(&prompt, None)?)?;
        let pairs = std::fs::read(p.join("ids.u32"))?;
        ensure!(pairs.len() >= 12, "empty source ids");
        let first: [u32; 3] = std::array::from_fn(|i| {
            u32::from_le_bytes(pairs[i * 4..i * 4 + 4].try_into().unwrap())
        });
        ensure!(
            first[0] as usize + 1 == ids.len() && first[1] == *ids.last().unwrap(),
            "prompt tokenization changed for {}",
            p.display()
        );
        let dest = out.join(p.file_name().unwrap());
        e.reset();
        e.dump_mtp_train(&dest)?;
        e.dump_prompt_positions(true);
        let next = e.prefill(&ids, MAX_T, None)?;
        e.finish_mtp_train_dump()?;
        ensure!(
            next == first[2],
            "first generated token changed for {}",
            p.display()
        );
        let residual = std::fs::read(dest.join("h.f16"))?;
        let width = HC * HIDDEN * 2;
        let mut source = std::fs::File::open(p.join("h.f16"))?;
        let mut expected = vec![0u8; width];
        std::io::Read::read_exact(&mut source, &mut expected)?;
        ensure!(
            residual.len() == ids.len() * width && residual[residual.len() - width..] == expected,
            "last prompt residual is not bitwise equal for {}",
            p.display()
        );
        std::fs::write(dest.join("done"), b"")?;
        println!(
            "prefix {}: {} positions, overlap bitwise equal",
            p.file_name().unwrap().to_string_lossy(),
            ids.len()
        );
    }
    std::fs::write(out.join("done"), b"")?;
    Ok(())
}

/// Repeated serving cells and held-out acceptance, with one engine load per checkpoint.
pub fn bench_panel(args: &[String]) -> Result<()> {
    let mut regular = Vec::new();
    let (mut prompt_dir, mut heldout_dir, mut repeats) = (None, None, 3usize);
    let mut it = args.iter();
    while let Some(k) = it.next() {
        match k.as_str() {
            "--prompt-dir" => {
                prompt_dir = Some(PathBuf::from(
                    it.next().context("--prompt-dir needs a value")?,
                ))
            }
            "--heldout-dir" => {
                heldout_dir = Some(PathBuf::from(
                    it.next().context("--heldout-dir needs a value")?,
                ))
            }
            "--repeats" => repeats = it.next().context("--repeats needs a value")?.parse()?,
            _ => regular.push(k.clone()),
        }
    }
    ensure!(
        repeats > 0 && (prompt_dir.is_some() || heldout_dir.is_some()),
        "panel needs prompts and positive repeats"
    );
    let mut a = parse(&regular)?;
    a.opts.split = true;
    ensure!(a.opts.mtp.is_some(), "panel requires --mtp FILE");
    let g = crate::gguf::Gguf::open(&a.path)?;
    let tok = super::tokenize::FlashTokenizer::from_gguf(&g)?;
    let mut serving = Vec::new();
    if let Some(dir) = prompt_dir {
        for (name, think, kind, limit) in [
            ("code", false, "hybrid", 5),
            ("chat", false, "mtp", 3),
            ("code", true, "mtp", 3),
            ("chat", true, "mtp", 3),
        ] {
            let text = std::fs::read_to_string(dir.join(format!("{name}4k.txt")))?;
            let ids = tok.encode(&tok.chat(&text, Some(think))?)?;
            serving.push((
                format!("{name}/{}", if think { "on" } else { "off" }),
                ids,
                kind,
                limit,
            ));
        }
    }
    let mut heldout = Vec::new();
    if let Some(dir) = heldout_dir {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|x| x.ok().map(|x| x.path()))
            .filter(|p| p.join("done").exists())
            .collect();
        paths.sort();
        for p in paths {
            let name = p.file_name().unwrap().to_string_lossy();
            if name
                .strip_prefix('p')
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|n| n.is_multiple_of(10))
            {
                let text = std::fs::read_to_string(p.join("prompt.txt"))?;
                heldout.push((
                    name.to_string(),
                    tok.encode(&tok.chat(&text, None)?)?,
                    "mtp",
                    3,
                ));
            }
        }
    }
    let longest = serving
        .iter()
        .chain(&heldout)
        .map(|x| x.1.len())
        .max()
        .unwrap_or(0);
    ensure!(longest > 0, "panel has no eligible prompts");
    a.opts.max_ctx = a
        .opts
        .max_ctx
        .max((longest + a.n + 8).next_multiple_of(1024));
    eprintln!(
        "panel: max_ctx {}, fixed slots {:?}, repeats {}, held-out {}",
        a.opts.max_ctx,
        a.opts.slots,
        repeats,
        heldout.len()
    );
    let mut e = load(&a)?;
    for steps in [3, 5] {
        e.set_mtp_steps(steps)?;
        e.warm()?;
    }
    for (group, cases, rounds) in [("serving", serving, repeats), ("heldout", heldout, 1)] {
        for repeat in 0..rounds {
            for j in 0..cases.len() {
                let (name, ids, kind, limit) = &cases[(j + repeat) % cases.len()];
                e.set_mtp_steps(*limit)?;
                let stop = if group == "heldout" {
                    Some(&[248046, 248044][..])
                } else {
                    Some(&[][..])
                };
                let (mut out, _, prefill_s, secs, sp) =
                    decode_spec_options(&mut e, ids, a.n, a.chunk, kind, Some(*limit), stop)?;
                if group == "heldout" {
                    if let Some(i) = out.iter().position(|t| [248046, 248044].contains(t)) {
                        out.truncate(i + 1);
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":group,"case":name,"repeat":repeat,"prompt_tokens":ids.len(),"requested":a.n,"generated":out,"prefill_seconds":prefill_s,"seconds":secs,"tokens":sp.tokens,"windows":sp.windows,"tokens_per_second":sp.tokens as f64/secs,"tokens_per_window":sp.tokens as f64/sp.windows.max(1) as f64,"acceptance":sp.by_pos,"thinking_acceptance":sp.by_pos_think,"answer_acceptance":sp.by_pos_answer,"widths":sp.widths,"draft":kind,"draft_limit":limit})
                );
            }
        }
    }
    Ok(())
}

/// `flash-coder-dump <trunk.gguf> --mtp SIDECAR (--prompts FILE.jsonl | --prompt-dir DIR)
/// --out DIR [-n 2000] [--k 3] [--chunk 8] [--max-ctx 8192] [--stop 151645,151643]`:
/// MTP distillation data for the Coder engine. One load; per prompt `i` (chat template,
/// greedy, drafting with the sidecar's `k`-step chain), `OUT/data/pNNN` (generated rows:
/// `h.f16`, `ids.u32`, the engine's drafts `draft.u32`, `prompt.txt`, `done`) and
/// `OUT/prefix/pNNN` (every prompt row, for `flash-mtp-train --prefix-data`). Completed
/// prompts are skipped, so a killed run resumes. `flash-mtp-train` holds out `pNNN % 10 == 0`.
pub fn coder_dump(args: &[String]) -> Result<()> {
    let usage = "flash-coder-dump <trunk.gguf> --mtp SIDECAR (--prompts FILE.jsonl | --prompt-dir DIR) --out DIR [-n 2000] [--k 3] [--chunk 8] [--max-ctx 8192] [--stop 151645,151643] [--eval [--held-out]]";
    let mut it = args.iter();
    let trunk = PathBuf::from(it.next().context(usage)?);
    let (mut mtp, mut out, mut prompts) = (None, None, Vec::<String>::new());
    let (mut n, mut k, mut chunk, mut max_ctx) = (2000usize, 3usize, 8usize, 8192usize);
    let mut stop = vec![151_645u32, 151_643];
    let (mut eval, mut held_out) = (false, false);
    while let Some(x) = it.next() {
        let mut val = || it.next().with_context(|| format!("{x} needs a value"));
        match x.as_str() {
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            "--eval" => eval = true,
            "--held-out" => held_out = true,
            "--out" => out = Some(PathBuf::from(val()?)),
            "-n" => n = val()?.parse()?,
            "--k" => k = val()?.parse()?,
            "--chunk" => chunk = val()?.parse()?,
            "--max-ctx" => max_ctx = val()?.parse()?,
            "--stop" => {
                stop = val()?
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--prompts" => {
                for line in std::fs::read_to_string(val()?)?.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let v: serde_json::Value = serde_json::from_str(line)?;
                    prompts.push(match &v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v["prompt"]
                            .as_str()
                            .context("jsonl lines: a string or {\"prompt\": ...}")?
                            .to_string(),
                    });
                }
            }
            "--prompt-dir" => {
                let mut ps: Vec<_> = std::fs::read_dir(val()?)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.is_file())
                    .collect();
                ps.sort();
                for p in ps {
                    prompts.push(std::fs::read_to_string(p)?);
                }
            }
            _ => bail!("unknown argument {x}: {usage}"),
        }
    }
    let mtp = mtp.context(usage)?;
    let out = match out {
        Some(o) => o,
        None if eval => PathBuf::new(),
        None => bail!("{usage}"),
    };
    ensure!(!prompts.is_empty(), "no prompts: {usage}");
    ensure!((1..MAX_T).contains(&k), "--k must be 1..{}", MAX_T - 1);
    let mut e = Coder::load(
        &trunk,
        CoderOpts {
            max_ctx,
            mtp: Some(mtp),
        },
    )?;
    eprintln!("{}", e.load_report);
    e.temperature = 0.0;
    e.mtp_steps = k;
    e.warm()?;
    let g = crate::gguf::Gguf::open(&trunk)?;
    let tok = super::tokenize::FlashTokenizer::from_gguf(&g)?;
    let (mut rows, mut proposed, mut accepted) = (0usize, 0usize, 0usize);
    // Per draft depth: (proposed, accepted); windows and tokens for tokens/window.
    let mut by_depth = vec![(0usize, 0usize); k];
    let (mut windows, mut tokens) = (0usize, 0usize);
    for (i, prompt) in prompts.iter().enumerate() {
        let name = format!("p{i:03}");
        if held_out && i % 10 != 0 {
            continue;
        }
        let (data, prefix) = (out.join("data").join(&name), out.join("prefix").join(&name));
        if !eval && data.join("done").exists() && prefix.join("done").exists() {
            continue;
        }
        for d in [&data, &prefix] {
            if !eval && d.exists() {
                std::fs::remove_dir_all(d)?;
            }
        }
        let ids = tok.encode(&tok.chat(prompt, None)?)?;
        // The chain's K/V runs `k + 1` past the last row; keep it inside the context.
        let room = max_ctx.saturating_sub(ids.len() + 2 * MAX_T + k + 2);
        if room < 64 {
            eprintln!(
                "{name}: prompt of {} tokens too long for --max-ctx {max_ctx}; skipped",
                ids.len()
            );
            continue;
        }
        let limit = n.min(room);
        e.reset();
        e.use_mtp = true;
        if !eval {
            e.start_train_dump(super::coder_mtp::TrainDump::create(
                &data,
                &prefix,
                ids.len() - 1,
                &stop,
            )?);
        }
        let t0 = Instant::now();
        let mut cur = e.prefill(&ids, chunk, None)?;
        let mut gen = vec![cur];
        while gen.len() < limit && !stop.contains(&cur) {
            let drafts: Vec<u32> = e.mtp_last.iter().take(k).map(|d| d.0).collect();
            let kept = e.verify(cur, &drafts)?;
            proposed += drafts.len();
            accepted += kept.len() - 1;
            for (j, b) in by_depth.iter_mut().enumerate().take(drafts.len()) {
                b.0 += 1;
                b.1 += usize::from(kept.len() - 1 > j);
            }
            windows += 1;
            tokens += kept.len();
            cur = *kept.last().unwrap();
            gen.extend_from_slice(&kept);
        }
        let secs = t0.elapsed().as_secs_f64();
        if eval {
            eprintln!(
                "{name}: prompt {} + {} generated, {secs:.1} s; running draft acceptance {:.3}",
                ids.len(),
                gen.len(),
                accepted as f64 / proposed.max(1) as f64
            );
            continue;
        }
        let r = e.finish_train_dump()?;
        rows += r;
        std::fs::write(data.join("prompt.txt"), prompt)?;
        std::fs::write(
            data.join("gen.txt"),
            gen.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
        std::fs::write(prefix.join("done"), b"")?;
        std::fs::write(data.join("done"), b"")?;
        eprintln!(
            "{name}: prompt {} + {} generated, {r} rows, {secs:.1} s; running draft acceptance {:.3}",
            ids.len(),
            gen.len(),
            accepted as f64 / proposed.max(1) as f64
        );
    }
    // Per depth j: P(draft j accepted | drafts 1..j-1 accepted), and cumulative from depth 1.
    let summary = serde_json::json!({
        "k": k,
        "windows": windows,
        "tokens": tokens,
        "tokens_per_window": tokens as f64 / windows.max(1) as f64,
        "by_depth": by_depth.iter().enumerate().map(|(j, &(p, a))| serde_json::json!({
            "depth": j + 1,
            "proposed": p,
            "accepted": a,
            "cumulative": a as f64 / windows.max(1) as f64,
            "conditional": if j == 0 { a as f64 / p.max(1) as f64 }
                else { a as f64 / by_depth[j - 1].1.max(1) as f64 },
        })).collect::<Vec<_>>(),
    });
    println!("{summary}");
    eprintln!("dumped {rows} rows to {}", out.display());
    Ok(())
}

/// `flash-tree-check` (qwen3next): is tree verification lossless? Decodes `-n` greedy tokens
/// one at a time, then again with forests of draft chains (`flash::tree`): random junk chains,
/// with the true continuation (from the first run) put into a random chain half the time.
/// Tokens must be identical. Then speed on the prompt: none, suffix chain, suffix forest.
pub fn tree_check(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let mut e = load_coder(&a)?;
    ensure!(e.temperature == 0.0, "tree-check is greedy");
    // TANG_TREE_PROMPTS=DIR: every file in DIR (sorted) as a chat prompt, one load.
    let prompts: Vec<(String, Vec<u32>)> = match std::env::var("TANG_TREE_PROMPTS") {
        Ok(dir) => {
            let mut fs: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .map(|d| Ok(d?.path()))
                .collect::<Result<_>>()?;
            fs.retain(|f| !f.file_name().unwrap().to_string_lossy().starts_with('.'));
            fs.sort();
            fs.iter()
                .map(|f| {
                    let text = std::fs::read_to_string(f)?;
                    let ids = super::tokenize::encode_chat(&a.path, &text, a.think)?;
                    Ok((f.file_name().unwrap().to_string_lossy().into_owned(), ids))
                })
                .collect::<Result<_>>()?
        }
        Err(_) => vec![("prompt".into(), prompt_ids(&a)?)],
    };
    let mut ok = true;
    for (name, ids) in &prompts {
        println!("== {name} ({} prompt tokens)", ids.len());
        ok &= tree_check_one(&mut e, ids, a.n.max(16))?;
    }
    ensure!(ok, "tree-check failed");
    Ok(())
}

fn tree_check_one(e: &mut Coder, ids: &[u32], n: usize) -> Result<bool> {
    use super::tree::Forest;
    use crate::draft::{Calibration, DraftConfig, Global, Session, COST_CUDA};
    let ids = ids.to_vec();
    e.reset();
    let mut cur = e.prefill(&ids, MAX_T, None)?;
    let mut greedy = vec![cur];
    while greedy.len() < n {
        cur = e.step(cur)?;
        greedy.push(cur);
    }
    let vocab = e.hp.n_vocab as u64;
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut rand = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    e.reset();
    let mut out = vec![e.prefill(&ids, MAX_T, None)?];
    let (mut windows, mut moved) = (0usize, 0usize);
    while out.len() < n {
        let at = out.len();
        let k = 1 + (rand() % 3) as usize;
        let mut chains: Vec<Vec<u32>> = (0..k)
            .map(|_| (0..1 + rand() % 3).map(|_| (rand() % vocab) as u32).collect())
            .collect();
        if rand() % 2 == 0 && at < greedy.len() {
            let len = 1 + (rand() % 4) as usize;
            let good = greedy[at..(at + len).min(greedy.len())].to_vec();
            let slot = (rand() as usize) % (chains.len() + 1);
            chains.insert(slot, good);
        }
        let f = Forest::new(chains);
        let cur = *out.last().unwrap();
        let emitted = e.verify_forest(cur, &f)?;
        windows += 1;
        // Windows won by a chain past the first with drafts kept: K/V moved down.
        moved += (e.last_tree.0 > 0 && e.last_tree.1 > 0) as usize;
        out.extend(emitted);
    }
    out.truncate(n);
    let ok = out == greedy;
    let first_bad = out.iter().zip(&greedy).position(|(x, y)| x != y);
    println!(
        "tree-check: {} ({n} tokens in {windows} windows, {moved} won by a later chain; first \
         difference {first_bad:?})",
        if ok { "PASS (tokens identical)" } else { "FAIL" }
    );
    // Speed: none / suffix chain / suffix forest.
    let cfg = DraftConfig::new(COST_CUDA);
    let global = Global::new(0, None);
    let modes: &[&str] = if e.has_mtp() {
        e.use_mtp = true;
        &["none", "chain", "forest", "mtp", "mtp-forest"]
    } else {
        &["none", "chain", "forest"]
    };
    for &mode in modes {
        e.reset();
        let mut cur = e.prefill(&ids, MAX_T, None)?;
        let cost: Vec<f32> = (0..=MAX_T).map(|k| cfg.cost(k.max(1))).collect();
        let mut sess = Session::new(&cfg, &global, Calibration::default(), cost, &ids);
        sess.push(cur);
        let (mut got, mut steps) = (1usize, 0usize);
        let t0 = Instant::now();
        while got < n {
            let room = (MAX_T - 1).min(n - got);
            let chains: Vec<Vec<u32>> = match mode {
                "none" => Vec::new(),
                "chain" => vec![sess.propose(room).tokens],
                "mtp" => vec![e.mtp_last.iter().take(room).map(|x| x.0).collect()],
                "mtp-forest" => {
                    let mut c = vec![e.mtp_last.iter().take(room).map(|x| x.0).collect()];
                    c.extend(e.mtp_alt.map(|(t, _)| vec![t]));
                    c
                }
                _ => sess
                    .branches(&cfg, &global, 3, room)
                    .into_iter()
                    .map(|d| d.tokens)
                    .collect(),
            };
            let em = e.verify_forest(cur, &Forest::new(chains))?;
            for &t in &em {
                sess.push(t);
            }
            got += em.len();
            cur = *em.last().unwrap();
            steps += 1;
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "{mode:>6}: {:.0} tok/s, {:.2} tokens a window",
            got as f64 / dt,
            got as f64 / steps as f64
        );
    }
    Ok(ok)
}
