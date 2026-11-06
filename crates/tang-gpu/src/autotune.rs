//! Per-kernel schedule autotuning on any [`Runtime`], built to be trusted:
//!
//! - **GPU clocks.** Kernels are timed with timestamp queries (wgpu, on
//!   Metal and Vulkan), CUDA events or OpenCL profiling events
//!   ([`Program::kernel_samples`]), never host wall clock where the
//!   backend has a GPU timer. Each candidate gets warm-up launches, then
//!   `samples` timed ones; the 5th percentile is its time (a shared GPU
//!   only ever adds time), the MAD its noise.
//! - **Playoffs, not single shots.** Candidates within `playoff` of the
//!   best screening time are re-timed interleaved (round-robin, so thermal
//!   drift and other tenants hit all of them alike). A winner whose noise
//!   stays above `noise` after `max_samples` is rejected in favour of a
//!   clean contender. Contenders within `tie` of the best are equivalent;
//!   among them the one earliest in the space's order wins (the node's
//!   current schedule first), so reruns pick the same schedule.
//! - **A declared space.** Candidates come from a
//!   [`Space`](tang_compile::tensor::space::Space): knobs, values,
//!   constraints. New schedule fields register a knob and are searched
//!   with no change here. Candidates that lower to the same kernel source
//!   are timed once; ones that fail to lower, fail to run, or disagree
//!   with the baseline's output are dropped.
//! - **A cost model to prune.** With a [`CostModel`] trained on logged
//!   measurements only the most promising candidates (plus the current
//!   schedule) are timed; without one, all are.
//! - **A cache per device and kernel shape.** Keyed by the device name
//!   and the kernel's canonical source with its schedule neutral (shapes
//!   are literals in it), so a kernel tuned once on a chip is never
//!   re-timed there, in any graph.

use crate::program::{Program, Runtime};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use tang_compile::tensor::space::{features, Candidate, CostModel, Sample, Space};
use tang_compile::tensor::wgsl::canonical_source;
use tang_compile::tensor::{TensorGraph, TensorId};

/// How hard to look, and where to remember what was found.
#[derive(Clone)]
pub struct TuneOptions {
    pub space: Space,
    /// Untimed launches before each candidate's samples.
    pub warmup: usize,
    /// Timed launches per candidate while screening.
    pub samples: usize,
    /// Most timed launches any one candidate gets (playoff included).
    pub max_samples: usize,
    /// Relative noise (1.4826·MAD / median) above which a time is not
    /// trusted to pick a winner.
    pub noise: f64,
    /// Candidates within this fraction of the best are re-timed together.
    pub playoff: f64,
    /// Contenders within this fraction of the best are equivalent.
    pub tie: f64,
    /// Prune with this model (rank by predicted time, time the top ones).
    pub model: Option<CostModel>,
    /// With a model: candidates timed per kernel, at least.
    pub keep: usize,
    /// With a model: and at least this fraction of the candidates.
    pub keep_frac: f64,
    /// Ignore the model and the cache: time every candidate.
    pub exhaustive: bool,
    /// Directory of the per-device cache and measurement log
    /// (`None`: neither read nor written).
    pub dir: Option<PathBuf>,
    /// Tag for logged samples (which problem they came from).
    pub group: u64,
    pub verbose: bool,
}

impl Default for TuneOptions {
    fn default() -> Self {
        Self {
            space: Space::standard(),
            warmup: 3,
            samples: 15,
            max_samples: 60,
            noise: 0.05,
            playoff: 0.15,
            tie: 0.05,
            model: None,
            keep: 6,
            keep_frac: 0.25,
            exhaustive: false,
            dir: default_dir(),
            group: 0,
            verbose: std::env::var_os("TANG_TUNE_VERBOSE").is_some(),
        }
    }
}

/// `$TANG_TUNE_DIR`, else `~/.cache/tang/autotune`.
pub fn default_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("TANG_TUNE_DIR") {
        return Some(PathBuf::from(d));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/tang/autotune"))
}

/// A sample set's summary.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// 5th percentile: the decision statistic. A shared GPU (another
    /// process time-slicing in, the compositor) only ever adds time to a
    /// pass, so the low tail is what the kernel costs; the median moves
    /// with whoever else is running.
    pub fast: f64,
    pub median: f64,
    pub min: f64,
    /// 1.4826·MAD / median: a robust relative standard deviation.
    pub noise: f64,
    pub n: usize,
}

impl Stats {
    pub fn of(v: &[f64]) -> Self {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        let median = med(&s);
        let mut dev: Vec<f64> = s.iter().map(|x| (x - median).abs()).collect();
        dev.sort_by(f64::total_cmp);
        Self {
            fast: s[(s.len() - 1) / 20],
            median,
            min: s[0],
            noise: 1.4826 * med(&dev) / median.max(1e-12),
            n: s.len(),
        }
    }
}

fn med(s: &[f64]) -> f64 {
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Paired playoff rounds, at least.
const PLAYOFF_ROUNDS: usize = 7;

fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// Interquartile range over the median: how much a ratio moves round to
/// round.
fn spread(v: &[f64]) -> f64 {
    (quantile(v, 0.75) - quantile(v, 0.25)) / quantile(v, 0.5)
}

/// One candidate's fate.
#[derive(Clone, Debug)]
pub struct Measured {
    pub candidate: Candidate,
    pub stats: Option<Stats>,
    /// Why it was not timed (lowering, launch, wrong output, duplicate…).
    pub dropped: Option<String>,
}

/// What the tuner did for one kernel.
#[derive(Clone, Debug)]
pub struct KernelTune {
    pub node: TensorId,
    /// Cache key: device + neutral-schedule kernel source.
    pub key: u64,
    pub picked: Candidate,
    /// Time of the picked candidate (its playoff 5th percentile), ms.
    pub ms: f64,
    /// Time of the node's schedule before tuning, ms.
    pub base_ms: f64,
    pub from_cache: bool,
    /// Candidates in the space / distinct kernels / actually timed.
    pub n_space: usize,
    pub n_distinct: usize,
    pub n_timed: usize,
    /// Every candidate's outcome (screening, then playoff stats).
    pub measured: Vec<Measured>,
    /// The fastest candidate was rejected as too noisy.
    pub rejected_noisy: bool,
    /// Timed with the host clock (no GPU timer on this backend).
    pub wall_clock: bool,
}

/// Minimum GPU time of the warm-up before samples are kept, ms.
pub const WARM_MS: f64 = 20.0;
/// Minimum GPU time the kept samples span, ms.
pub const SPAN_MS: f64 = 10.0;

/// Samples with the zeros dropped: a Metal pass too short for the
/// counter's resolution reads begin = end. All zero: one `INFINITY`, so
/// an unmeasurable candidate never wins.
fn positive(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|&x| x > 0.0);
    if v.is_empty() {
        v.push(f64::INFINITY);
    }
    v
}

/// Probe the per-launch time, then time with warm-up and sample counts
/// scaled to [`WARM_MS`] / [`SPAN_MS`].
fn budgeted(
    f: impl Fn(usize, usize) -> Option<Vec<f64>>,
    warmup: usize,
    n: usize,
) -> Option<Vec<f64>> {
    let probe = Stats::of(&positive(f(warmup.max(1), 3)?))
        .median
        .clamp(1e-4, 1e3);
    let w = ((WARM_MS / probe).ceil() as usize).clamp(warmup, 400);
    let m = ((SPAN_MS / probe).ceil() as usize).clamp(n, 400);
    f(w, m)
}

/// GPU-clock samples of kernel `k` (host clock if the backend has none),
/// and whether they came from the host.
///
/// `warmup` and `n` are minimums: the warm-up runs for at least
/// [`WARM_MS`] and the samples span at least [`SPAN_MS`] of GPU time, so a
/// 50 µs kernel is timed at the clock a sustained load runs at (an M4's
/// GPU otherwise reports a different speed every run).
pub fn kernel_samples(
    rt: &Runtime,
    p: &Program,
    k: usize,
    warmup: usize,
    n: usize,
) -> (Vec<f64>, bool) {
    if let Some(v) = budgeted(|w, m| p.kernel_samples(rt, k, w, m), warmup, n) {
        return (positive(v), false);
    }
    for _ in 0..warmup {
        p.run_kernel(rt, k);
    }
    rt.sync();
    let v = (0..n)
        .map(|_| {
            let t = std::time::Instant::now();
            p.run_kernel(rt, k);
            rt.sync();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    (v, true)
}

/// GPU-clock samples of whole runs of `p` (host clock as a fallback).
pub fn program_samples(rt: &Runtime, p: &Program, warmup: usize, n: usize) -> Vec<f64> {
    if let Some(v) = budgeted(|w, m| p.program_samples(rt, w, m), warmup, n) {
        return positive(v);
    }
    for _ in 0..warmup {
        p.run(rt);
    }
    rt.sync();
    (0..n)
        .map(|_| {
            let t = std::time::Instant::now();
            p.run(rt);
            rt.sync();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect()
}

/// Pick the fastest of several whole graphs (algebra choices: fused vs
/// unfused, materialized vs recomputed) by interleaved GPU-clock 5th percentiles
/// of whole runs; ties within `tie` go to the earliest. Returns the index
/// and each graph's stats (`None`: failed to lower or run).
pub fn pick_graph(
    rt: &Runtime,
    graphs: &[(&TensorGraph, &[TensorId])],
    inputs: &[&[f32]],
    rounds: usize,
    per_round: usize,
    tie: f64,
) -> (usize, Vec<Option<Stats>>) {
    let progs: Vec<Option<Program>> = graphs
        .iter()
        .map(|(g, o)| {
            catch_unwind(AssertUnwindSafe(|| {
                let p = Program::compile(rt, g, o, inputs);
                p.run(rt);
                rt.sync();
                p
            }))
            .ok()
        })
        .collect();
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); progs.len()];
    for _ in 0..rounds {
        for (i, p) in progs.iter().enumerate() {
            if let Some(p) = p {
                samples[i].extend(program_samples(rt, p, 2, per_round));
            }
        }
    }
    let stats: Vec<Option<Stats>> = samples
        .iter()
        .map(|s| (!s.is_empty()).then(|| Stats::of(s)))
        .collect();
    let best = stats
        .iter()
        .flatten()
        .map(|s| s.fast)
        .fold(f64::INFINITY, f64::min);
    let pick = stats
        .iter()
        .position(|s| s.is_some_and(|s| s.fast <= best * (1.0 + tie)))
        .unwrap_or(0);
    (pick, stats)
}

/// The tuning cache and measurement log of one device.
pub struct TuneStore {
    cache_path: Option<PathBuf>,
    log_path: Option<PathBuf>,
    entries: HashMap<u64, String>,
}

fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

impl TuneStore {
    pub fn open(dir: Option<&Path>, device: &str) -> Self {
        let Some(dir) = dir else {
            return Self {
                cache_path: None,
                log_path: None,
                entries: HashMap::new(),
            };
        };
        let cache_path = dir.join(format!("{}.cache", slug(device)));
        let log_path = dir.join(format!("{}.log", slug(device)));
        let entries = std::fs::read_to_string(&cache_path)
            .map(|t| {
                t.lines()
                    .filter_map(|l| {
                        let mut it = l.split('\t');
                        let k = u64::from_str_radix(it.next()?, 16).ok()?;
                        Some((k, it.next()?.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            cache_path: Some(cache_path),
            log_path: Some(log_path),
            entries,
        }
    }

    pub fn get(&self, key: u64) -> Option<&str> {
        self.entries.get(&key).map(|s| s.as_str())
    }

    fn append(path: &Option<PathBuf>, text: &str) {
        let Some(p) = path else { return };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            let _ = f.write_all(text.as_bytes());
        }
    }

    pub fn put(&mut self, key: u64, cand: &str, ms: f64) {
        self.entries.insert(key, cand.to_string());
        Self::append(&self.cache_path, &format!("{key:016x}\t{cand}\t{ms:.6}\n"));
    }

    /// Log one kernel's measurements as cost-model samples: `group key
    /// target f0 f1 …`, target `ln(t / t_base)`.
    pub fn log(&self, group: u64, key: u64, rows: &[(f32, Vec<f32>)]) {
        let mut s = String::new();
        for (t, f) in rows {
            s.push_str(&format!("{group:016x}\t{key:016x}\t{t:.5}"));
            for x in f {
                s.push_str(&format!("\t{x:.4}"));
            }
            s.push('\n');
        }
        Self::append(&self.log_path, &s);
    }

    /// Every logged sample (all groups).
    pub fn samples(&self) -> Vec<Sample> {
        let Some(p) = &self.log_path else {
            return Vec::new();
        };
        load_samples(p)
    }
}

/// Samples from a measurement log written by [`TuneStore::log`].
pub fn load_samples(path: &Path) -> Vec<Sample> {
    std::fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter_map(|l| {
                    let mut it = l.split('\t');
                    let group = u64::from_str_radix(it.next()?, 16).ok()?;
                    let _key = it.next()?;
                    let target = it.next()?.parse().ok()?;
                    let features = it.map(|x| x.parse().ok()).collect::<Option<Vec<f32>>>()?;
                    Some(Sample {
                        features,
                        target,
                        group,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `node`'s kernel in a lowering, by node.
fn kernel_of(p: &tang_compile::tensor::wgsl::WgslProgram, node: TensorId) -> Option<usize> {
    p.kernels.iter().position(|k| k.node == node)
}

fn quiet<R>(f: impl FnOnce() -> R) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f)).ok();
    std::panic::set_hook(hook);
    r
}

fn close(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .all(|(x, y)| x.is_finite() == y.is_finite() && (x - y).abs() <= 2e-2 * scale + 1e-4)
}

/// Tune every kernel of `g` (in program order, each with its inputs
/// computed by the kernels before it under their tuned schedules).
pub fn autotune(
    rt: &Runtime,
    mut g: TensorGraph,
    outputs: &[TensorId],
    inputs: &[&[f32]],
    opts: &TuneOptions,
) -> (TensorGraph, Vec<KernelTune>) {
    let mut store = TuneStore::open(opts.dir.as_deref(), rt.name());
    let nodes: Vec<TensorId> = rt
        .lower(&g, outputs)
        .kernels
        .iter()
        .map(|k| k.node)
        .collect();
    let mut report = Vec::new();
    for node in nodes {
        if let Some(k) = tune_kernel(rt, &mut g, outputs, inputs, node, opts, &mut store) {
            report.push(k);
        }
    }
    (g, report)
}

/// Cache key of `node`'s kernel on this device: the device name and the
/// kernel's canonical source with every knob neutral.
pub fn kernel_key(
    rt: &Runtime,
    g: &TensorGraph,
    outputs: &[TensorId],
    node: TensorId,
    space: &Space,
) -> Option<u64> {
    let mut n = g.clone();
    space.apply(&mut n, node, &space.neutral(g, node));
    let p = quiet(|| rt.lower(&n, outputs))?;
    let k = kernel_of(&p, node)?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    rt.name().hash(&mut h);
    canonical_source(&p.kernels[k].source).hash(&mut h);
    Some(h.finish())
}

fn tune_kernel(
    rt: &Runtime,
    g: &mut TensorGraph,
    outputs: &[TensorId],
    inputs: &[&[f32]],
    node: TensorId,
    opts: &TuneOptions,
    store: &mut TuneStore,
) -> Option<KernelTune> {
    let space = &opts.space;
    let key = kernel_key(rt, g, outputs, node, space)?;
    let cands = space.candidates(g, node);
    let n_space = cands.len();
    if !opts.exhaustive {
        if let Some(c) = store
            .get(key)
            .and_then(|t| Candidate::decode(space, g, node, t))
        {
            space.apply(g, node, &c);
            return Some(KernelTune {
                node,
                key,
                picked: c,
                ms: f64::NAN,
                base_ms: f64::NAN,
                from_cache: true,
                n_space,
                n_distinct: 0,
                n_timed: 0,
                measured: Vec::new(),
                rejected_noisy: false,
                wall_clock: false,
            });
        }
    }
    // lower every candidate; keep one per distinct kernel source
    let mut measured: Vec<Measured> = Vec::new();
    let mut live: Vec<(usize, TensorGraph)> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for c in cands {
        let mut v = g.clone();
        space.apply(&mut v, node, &c);
        let src = quiet(|| rt.lower(&v, outputs))
            .and_then(|p| Some(p.kernels[kernel_of(&p, node)?].source.clone()));
        let dropped = match src {
            None => Some("does not lower".to_string()),
            Some(s) => match seen.get(&s) {
                Some(&j) => Some(format!("same kernel as #{j}")),
                None => {
                    seen.insert(s, measured.len());
                    live.push((measured.len(), v));
                    None
                }
            },
        };
        measured.push(Measured {
            candidate: c,
            stats: None,
            dropped,
        });
    }
    let n_distinct = live.len();
    if n_distinct <= 1 {
        return None; // nothing to choose
    }
    // prune with the model: the current schedule plus the top predictions
    if let (Some(m), false) = (&opts.model, opts.exhaustive) {
        let keep = opts
            .keep
            .max((opts.keep_frac * n_distinct as f64).ceil() as usize);
        if live.len() > keep && m.n_features == space.feature_names().len() {
            let mut scored: Vec<(f32, usize)> = live
                .iter()
                .enumerate()
                .skip(1)
                .map(|(j, (i, _))| {
                    (
                        m.predict(&features(space, g, node, &measured[*i].candidate)),
                        j,
                    )
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut keep_idx: Vec<usize> = std::iter::once(0)
                .chain(scored.iter().take(keep - 1).map(|s| s.1))
                .collect();
            keep_idx.sort();
            let mut kept = Vec::new();
            for (j, e) in live.into_iter().enumerate() {
                if keep_idx.contains(&j) {
                    kept.push(e);
                } else {
                    measured[e.0].dropped = Some("pruned by cost model".into());
                }
            }
            live = kept;
        }
    }
    // build, check against the baseline output, screen
    let build = |v: &TensorGraph| -> Option<(Program, usize)> {
        quiet(|| {
            let p = Program::compile(rt, v, outputs, inputs);
            let k = kernel_of(&p.program, node).expect("kernel");
            p.run(rt);
            rt.sync();
            (p, k)
        })
    };
    let mut reference: Option<Vec<f32>> = None;
    let mut wall = false;
    let mut screened: Vec<(usize, f64)> = Vec::new(); // (measured index, median)
    for (i, v) in &live {
        let Some((p, k)) = build(v) else {
            measured[*i].dropped = Some("does not run".into());
            continue;
        };
        let out = p.read(rt, node);
        match &reference {
            None => reference = Some(out),
            Some(r) if !close(&out, r) => {
                measured[*i].dropped = Some("wrong output".into());
                continue;
            }
            _ => {}
        }
        let (s, w) = kernel_samples(rt, &p, k, opts.warmup, opts.samples);
        wall |= w;
        let st = Stats::of(&s);
        measured[*i].stats = Some(st);
        screened.push((*i, st.fast));
    }
    if screened.is_empty() {
        return None;
    }
    let base_i = live[0].0;
    let screen_best = screened.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
    // playoff: the close ones, re-timed round-robin
    let mut contenders: Vec<usize> = screened
        .iter()
        .filter(|s| s.1 <= screen_best * (1.0 + opts.playoff))
        .map(|s| s.0)
        .collect();
    contenders.sort_by(|a, b| {
        measured[*a]
            .stats
            .unwrap()
            .fast
            .total_cmp(&measured[*b].stats.unwrap().fast)
    });
    contenders.truncate(8);
    contenders.sort();
    // Paired rounds: every round times each contender once, back to
    // back, and scores it by its ratio to that round's fastest. Drift
    // between rounds (clocks, other tenants) cancels; a contender's score
    // is its median ratio, its noise the spread of its ratios.
    let mut ratios: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut pool: HashMap<usize, Vec<f64>> = HashMap::new();
    let progs: Vec<(usize, Program, usize)> = if contenders.len() > 1 {
        contenders
            .iter()
            .filter_map(|&i| {
                let v = &live.iter().find(|(j, _)| *j == i)?.1;
                build(v).map(|(p, k)| (i, p, k))
            })
            .collect()
    } else {
        Vec::new()
    };
    let per = (opts.samples / 2).max(3);
    let mut round = |ratios: &mut HashMap<usize, Vec<f64>>| {
        let fasts: Vec<(usize, f64)> = progs
            .iter()
            .map(|(i, p, k)| {
                let s = kernel_samples(rt, p, *k, 2, per).0;
                let f = Stats::of(&s).fast;
                pool.entry(*i).or_default().extend(s);
                (*i, f)
            })
            .collect();
        let b = fasts.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        for (i, f) in fasts {
            ratios.entry(i).or_default().push(f / b);
        }
    };
    if progs.len() > 1 {
        let max_rounds = (opts.max_samples / per).max(PLAYOFF_ROUNDS);
        for r in 1..=max_rounds {
            round(&mut ratios);
            // more rounds while the leaders' order is unclear
            if r >= PLAYOFF_ROUNDS && ratios.values().all(|r| spread(r) <= opts.noise) {
                break;
            }
        }
    }
    // (pick, rejected_noisy, near misses: earlier candidates just outside
    // the tie band, whose verdict a few more rounds could flip)
    let decide = |ratios: &HashMap<usize, Vec<f64>>| -> (usize, bool, bool) {
        let score = |i: usize| ratios.get(&i).map_or(1.0, |r| quantile(r, 0.5));
        // noisy = an outlier among the contenders: on a shared GPU every
        // ratio moves; one that moves far more than the rest is not trusted
        let typical = {
            let v: Vec<f64> = ratios.values().map(|r| spread(r)).collect();
            if v.is_empty() {
                0.0
            } else {
                quantile(&v, 0.5)
            }
        };
        let noisy = |i: usize| {
            ratios
                .get(&i)
                .is_some_and(|r| spread(r) > opts.noise.max(2.0 * typical))
        };
        let clean: Vec<usize> = contenders.iter().copied().filter(|&i| !noisy(i)).collect();
        let fastest = contenders
            .iter()
            .copied()
            .min_by(|a, b| score(*a).total_cmp(&score(*b)))
            .unwrap();
        let pool_from = if clean.is_empty() {
            contenders.clone()
        } else {
            clean
        };
        let leader = *pool_from
            .iter()
            .min_by(|a, b| score(**a).total_cmp(&score(**b)))
            .unwrap();
        let best = score(leader);
        // Equivalent: within `tie`, or within what the playoff can
        // resolve (2 standard errors of the leader's median ratio, at
        // most 8%). The earliest in the space's order wins, so a rerun
        // that sees the same near-tie in another order picks the same.
        let se = ratios.get(&leader).map_or(0.0, |r| {
            // median's standard error from the IQR (σ ≈ IQR/1.35)
            1.25 * spread(r) / 1.35 / (r.len() as f64).sqrt()
        });
        let tie = opts.tie.max((2.0 * se).min(0.08));
        let pick = *pool_from
            .iter()
            .filter(|&&i| score(i) <= best * (1.0 + tie))
            .min()
            .unwrap();
        let near = pool_from
            .iter()
            .any(|&i| i < pick && score(i) <= best * (1.0 + 2.0 * tie));
        (pick, !pool_from.contains(&fastest), near)
    };
    let (mut pick, mut rejected_noisy, mut near) = decide(&ratios);
    // settle near misses with more rounds (twice at most)
    for _ in 0..2 {
        if !near {
            break;
        }
        for _ in 0..PLAYOFF_ROUNDS {
            round(&mut ratios);
        }
        (pick, rejected_noisy, near) = decide(&ratios);
    }
    for (i, s) in &pool {
        measured[*i].stats = Some(Stats::of(s));
    }
    let score = |i: usize| ratios.get(&i).map_or(1.0, |r| quantile(r, 0.5));
    let stat = |i: usize| measured[i].stats.unwrap();
    let picked = measured[pick].candidate.clone();
    space.apply(g, node, &picked);
    // remember
    let base_ms = measured[base_i].stats.map_or(f64::NAN, |s| s.fast);
    let base_screen = measured[base_i]
        .stats
        .map(|s| s.fast)
        .unwrap_or(screen_best);
    let rows: Vec<(f32, Vec<f32>)> = screened
        .iter()
        .map(|&(i, m)| {
            // screening medians: one clock epoch for all of them
            (
                (m / base_screen).ln() as f32,
                features(space, g, node, &measured[i].candidate),
            )
        })
        .collect();
    store.log(opts.group, key, &rows);
    store.put(key, &picked.encode(space), stat(pick).fast);
    if opts.verbose {
        for &i in &contenders {
            let st = stat(i);
            eprintln!(
                "  #{i:<4} {:<44} score {:.3} ±{:.1}% fast {:.4} med {:.4} noise {:.1}% n {}",
                measured[i].candidate.encode(space),
                score(i),
                ratios.get(&i).map_or(0.0, |r| spread(r) * 100.0),
                st.fast,
                st.median,
                st.noise * 100.0,
                st.n
            );
        }
        eprintln!(
            "autotune node {} : {} distinct of {}, timed {}, picked {} {:.4} ms (base {:.4}){}",
            node.index(),
            n_distinct,
            n_space,
            screened.len(),
            picked.encode(space),
            stat(pick).fast,
            base_ms,
            if rejected_noisy {
                " [fastest rejected: noisy]"
            } else {
                ""
            }
        );
    }
    Some(KernelTune {
        node,
        key,
        picked,
        ms: stat(pick).fast,
        base_ms,
        from_cache: false,
        n_space,
        n_distinct,
        n_timed: screened.len(),
        measured,
        rejected_noisy,
        wall_clock: wall,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_round_trips() {
        let dir = std::env::temp_dir().join(format!("tang-tune-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut st = TuneStore::open(Some(&dir), "Some GPU (Metal)");
            st.put(0xabc, "split=0;tile=16,16,32", 0.5);
            st.log(7, 0xabc, &[(-0.25, vec![1.0, 2.0]), (0.1, vec![3.0, 4.0])]);
        }
        let st = TuneStore::open(Some(&dir), "Some GPU (Metal)");
        assert_eq!(st.get(0xabc), Some("split=0;tile=16,16,32"));
        assert_eq!(TuneStore::open(Some(&dir), "Other GPU").get(0xabc), None);
        let s = st.samples();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].group, 7);
        assert!((s[0].target + 0.25).abs() < 1e-6);
        assert_eq!(s[1].features, vec![3.0, 4.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_samples_never_win() {
        assert_eq!(positive(vec![0.0, 0.0]), vec![f64::INFINITY]);
        assert_eq!(positive(vec![0.0, 2.0]), vec![2.0]);
    }

    #[test]
    fn stats_are_robust() {
        let s = Stats::of(&[1.0, 1.01, 0.99, 1.0, 50.0]);
        assert!((s.median - 1.0).abs() < 1e-9);
        assert!(s.noise < 0.02);
    }
}
