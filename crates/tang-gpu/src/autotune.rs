//! Per-kernel schedule autotuning on any [`Runtime`], built to be trusted:
//!
//! - **GPU clocks.** Kernels are timed with timestamp queries (wgpu, on
//!   Metal and Vulkan), CUDA events or OpenCL profiling events
//!   ([`Program::kernel_samples`]), never host wall clock where the
//!   backend has a GPU timer. Each candidate gets warm-up launches, then
//!   `samples` timed ones; the 20th percentile is its time (a shared GPU
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
            tie: 0.04,
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
    /// 20th percentile: the decision statistic. A shared GPU (another
    /// process, the compositor) only ever adds time, so a low quantile
    /// is what the kernel costs; the median drifts with contention.
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
            fast: s[(s.len() - 1) / 5],
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
    /// Time of the picked candidate (its playoff 20th percentile), ms.
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

/// Probe the per-launch time, then time with warm-up and sample counts
/// scaled to [`WARM_MS`] / [`SPAN_MS`].
fn budgeted(
    f: impl Fn(usize, usize) -> Option<Vec<f64>>,
    warmup: usize,
    n: usize,
) -> Option<Vec<f64>> {
    let probe = Stats::of(&f(warmup.max(1), 3)?).median.max(1e-4);
    let w = ((WARM_MS / probe).ceil() as usize).clamp(warmup, 4000);
    let m = ((SPAN_MS / probe).ceil() as usize).clamp(n, 1000);
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
        return (v, false);
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
        return v;
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
/// unfused, materialized vs recomputed) by interleaved GPU-clock 20th percentiles
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
    let mut pool: HashMap<usize, Vec<f64>> = HashMap::new();
    if contenders.len() > 1 {
        let progs: Vec<(usize, Program, usize)> = contenders
            .iter()
            .filter_map(|&i| {
                let v = &live.iter().find(|(j, _)| *j == i)?.1;
                build(v).map(|(p, k)| (i, p, k))
            })
            .collect();
        let rounds = 4;
        let per = (opts.samples / 2).max(3);
        for _ in 0..rounds {
            for (i, p, k) in &progs {
                pool.entry(*i)
                    .or_default()
                    .extend(kernel_samples(rt, p, *k, 2, per).0);
            }
        }
        // noisy contenders get more, up to max_samples
        for (i, p, k) in &progs {
            let s = pool.get_mut(i).unwrap();
            while Stats::of(s).noise > opts.noise && s.len() < opts.max_samples {
                s.extend(kernel_samples(rt, p, *k, 1, per).0);
            }
        }
        for (i, s) in &pool {
            measured[*i].stats = Some(Stats::of(s));
        }
    }
    let stat = |i: usize| measured[i].stats.unwrap();
    let clean: Vec<usize> = contenders
        .iter()
        .copied()
        .filter(|&i| stat(i).noise <= opts.noise)
        .collect();
    let fastest = contenders
        .iter()
        .copied()
        .min_by(|a, b| stat(*a).fast.total_cmp(&stat(*b).fast))
        .unwrap();
    let pool_from = if clean.is_empty() {
        contenders.clone()
    } else {
        clean
    };
    let rejected_noisy = !pool_from.contains(&fastest);
    let best = pool_from
        .iter()
        .map(|&i| stat(i).fast)
        .fold(f64::INFINITY, f64::min);
    let pick = *pool_from
        .iter()
        .filter(|&&i| stat(i).fast <= best * (1.0 + opts.tie))
        .min()
        .unwrap();
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
    fn stats_are_robust() {
        let s = Stats::of(&[1.0, 1.01, 0.99, 1.0, 50.0]);
        assert!((s.median - 1.0).abs() < 1e-9);
        assert!(s.noise < 0.02);
    }
}
