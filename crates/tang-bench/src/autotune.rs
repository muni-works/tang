//! The schedule autotuner over a lowered decode step, in three layers:
//!
//! 1. **Micro-benchmark**: every kernel of a variant program timed alone on
//!    the GPU ([`Program::time_kernel`]: CUDA events around single launches,
//!    the minimum of `5 * reps`), on the live model's buffers (each variant
//!    is built [`with_shared`](Program::with_shared) on the base step's
//!    weights and caches, so gathers and routing see real data).
//! 2. **Per-kernel selection**: the lowering reports which knobs each kernel
//!    consulted ([`tune::take_report`]); each knob of the grid is swept one at
//!    a time across every kernel that reads it, each kernel signature keeps
//!    the fastest legal value per knob (variants that fail to lower are
//!    skipped), the per-knob winners are combined and re-measured, and a
//!    signature keeps a choice only if it beats the default by `MIN_GAIN`.
//!    Step-level knobs (multi-output kernels, staged maps) change kernel
//!    boundaries, so they are judged by whole-step time instead.
//! 3. **Tuning cache**: the choices go to `tune/<gpu>.tsv`
//!    ([`tune`](tang_compile::tensor::tune)), merged with the entries already
//!    there, and every later lowering on that GPU reads them.
use std::collections::{BTreeMap, HashMap};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tang_compile::tensor::tune::{self, Choice};
use tang_gpu::program::{Program, Runtime};

/// The per-kernel grid: knob, its alternatives to the lowering's default.
pub const GRID: &[(&str, &[&str])] = &[
    ("TANG_GEMV_WARPS", &["2", "4", "16"]),
    ("TANG_GEMV_UPL", &["1", "4"]),
    ("TANG_GEMV_UNROLL", &["1"]),
    ("TANG_GEMV_Q8_UPL", &["2", "4"]),
    ("TANG_WIN_TILE", &["8", "16"]),
    ("TANG_WIN_STREAM", &["0"]),
    ("TANG_REG2", &["0"]),
    ("TANG_SUBWARP", &["0"]),
    ("TANG_SPLIT_FUSE", &["0"]),
    ("TANG_STAGED_MAP", &["0"]),
];

/// A choice must beat the default by this fraction to be kept (noise floor).
pub const MIN_GAIN: f64 = 0.03;

/// One signature's outcome.
#[derive(Clone, Debug)]
pub struct SigResult {
    pub sig: String,
    pub instances: usize,
    pub default_ms: f64,
    pub tuned_ms: f64,
    pub choice: Choice,
}

/// Whole-step and per-signature results of a tuning pass.
pub struct Tuned {
    pub step_choice: Choice,
    pub sigs: Vec<SigResult>,
    /// Kernel-sum ms (each kernel alone) before and after.
    pub sum_default: f64,
    pub sum_tuned: f64,
}

fn msg(s: &str) {
    eprintln!("[autotune] {s}");
}

/// Build under `force` (every kernel), run once so each kernel's inputs exist.
/// `None` when the variant does not lower (an illegal schedule).
fn build_forced<F: FnMut() -> Program>(
    rt: &Runtime,
    build: &mut F,
    force: Choice,
) -> Option<(Program, Vec<(usize, String, Vec<String>)>)> {
    tune::force_all(force);
    let _ = tune::take_report();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let got = catch_unwind(AssertUnwindSafe(&mut *build));
    std::panic::set_hook(prev);
    tune::force_all(Choice::new());
    let report = tune::take_report();
    let p = got.ok()?;
    p.run(rt);
    rt.sync();
    Some((p, report))
}

/// Up to `per` kernel indices per signature, the kernels of `filter`.
fn sample<'a>(
    report: &'a [(usize, String, Vec<String>)],
    per: usize,
    filter: impl Fn(&[String]) -> bool,
) -> BTreeMap<&'a str, Vec<usize>> {
    let mut m: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, s, ks) in report {
        if !filter(ks) {
            continue;
        }
        let v = m.entry(s.as_str()).or_default();
        if v.len() < per {
            v.push(*i);
        }
    }
    m
}

/// Per signature: mean ms of its sampled kernels at each position of `pos`
/// (`set_pos` writes the step's position input; context-dependent kernels
/// such as attention reductions must win at every length, not one).
fn time_sigs(
    rt: &Runtime,
    p: &Program,
    s: &BTreeMap<&str, Vec<usize>>,
    reps: usize,
    pos: &[usize],
    set_pos: &dyn Fn(usize),
) -> HashMap<String, Vec<f64>> {
    let mut m: HashMap<String, Vec<f64>> = HashMap::new();
    for &q in pos {
        set_pos(q);
        p.run(rt);
        rt.sync();
        for (k, v) in time_sigs_at(rt, p, s, reps) {
            m.entry(k).or_default().push(v);
        }
    }
    m
}

/// Better than `base` at every position, and by `MIN_GAIN` in total.
fn wins(t: &[f64], base: &[f64]) -> bool {
    let (a, b): (f64, f64) = (t.iter().sum(), base.iter().sum());
    t.len() == base.len() && t.iter().zip(base).all(|(x, y)| *x <= *y * 1.01) && a < b * (1.0 - MIN_GAIN)
}

fn time_sigs_at(rt: &Runtime, p: &Program, s: &BTreeMap<&str, Vec<usize>>, reps: usize) -> HashMap<String, f64> {
    s.iter()
        .map(|(sig, ks)| {
            let n = p.program.kernels.len();
            let t: Vec<f64> = ks
                .iter()
                .filter(|&&i| i < n)
                .map(|&i| p.time_kernel(rt, i, reps))
                .collect();
            (sig.to_string(), t.iter().sum::<f64>() / t.len().max(1) as f64)
        })
        .collect()
}

/// Whole-step device ms: best of 3 × `iters` replays.
pub fn step_ms(rt: &Runtime, p: &Program, iters: usize) -> f64 {
    (0..3).map(|_| p.time(rt, iters)).fold(f64::INFINITY, f64::min)
}

/// Tune one step. `build` lowers and allocates the step (sharing the live
/// model's buffers) under whatever knobs are forced at the time; `step_grid`
/// lists step-level knobs to judge by whole-step time (`step:<step_name>`).
/// Environment variables set by the caller still win everywhere.
pub fn tune_step<F: FnMut() -> Program>(
    rt: &Runtime,
    step_name: &str,
    step_grid: &[(&str, &[&str])],
    mut build: F,
    reps: usize,
    pos: &[usize],
    set_pos: &dyn Fn(usize),
) -> Tuned {
    tune::set_entries(HashMap::new());
    // step level: whole-step time per value, one knob at a time
    let mut step_choice = Choice::new();
    if !step_grid.is_empty() {
        let (mut p, _) = build_forced(rt, &mut build, Choice::new()).expect("default step lowers");
        p.capture(rt);
        let mut best: f64 = pos.iter().map(|&q| { set_pos(q); step_ms(rt, &p, 20) }).sum();
        drop(p);
        msg(&format!("step {step_name}: default {best:.3} ms"));
        for (k, vals) in step_grid {
            for v in *vals {
                let mut c = step_choice.clone();
                c.insert(k.to_string(), v.to_string());
                let Some((mut p, _)) = build_forced(rt, &mut build, c.clone()) else {
                    msg(&format!("step {k}={v}: does not lower"));
                    continue;
                };
                p.capture(rt);
                let ms: f64 = pos.iter().map(|&q| { set_pos(q); step_ms(rt, &p, 20) }).sum();
                msg(&format!("step {k}={v}: {ms:.3} ms"));
                if ms < best * (1.0 - MIN_GAIN / 3.0) {
                    best = ms;
                    step_choice = c;
                }
            }
        }
    }
    let with_step = |c: &Choice| {
        let mut c = c.clone();
        c.extend(step_choice.clone());
        c
    };
    // base: per-kernel times and what each kernel consults
    let (p0, r0) = build_forced(rt, &mut build, with_step(&Choice::new())).expect("base lowers");
    let all = sample(&r0, 3, |_| true);
    let t0 = time_sigs(rt, &p0, &all, reps, pos, set_pos);
    let instances: HashMap<&str, usize> = r0.iter().fold(HashMap::new(), |mut m, (_, s, _)| {
        *m.entry(s.as_str()).or_default() += 1;
        m
    });
    let consulted: std::collections::BTreeSet<&str> =
        r0.iter().flat_map(|(_, _, ks)| ks.iter().map(|k| k.as_str())).collect();
    msg(&format!(
        "{} kernels, {} signatures, knobs consulted: {:?}",
        r0.len(),
        all.len(),
        consulted
    ));
    drop(p0);
    // sweep: per knob, per value, the kernels that read it
    let mut best: HashMap<String, Choice> = HashMap::new();
    let mut best_ms: HashMap<String, BTreeMap<String, f64>> = HashMap::new();
    for (k, vals) in GRID {
        if !consulted.contains(k) || std::env::var(k).is_ok() {
            continue;
        }
        for v in *vals {
            let force = with_step(&[(k.to_string(), v.to_string())].into());
            let Some((p, r)) = build_forced(rt, &mut build, force) else {
                msg(&format!("{k}={v}: does not lower, skipped"));
                continue;
            };
            let readers = sample(&r, 3, |ks| ks.iter().any(|x| x == k));
            let t = time_sigs(rt, &p, &readers, reps, pos, set_pos);
            let mut won = 0;
            for (sig, tv) in t {
                let Some(base) = t0.get(&sig) else { continue };
                let ms: f64 = tv.iter().sum();
                let cur = best_ms.entry(sig.clone()).or_default();
                let prev = cur.get(*k).copied().unwrap_or(base.iter().sum());
                if ms < prev && wins(&tv, base) {
                    cur.insert(k.to_string(), ms);
                    best.entry(sig).or_default().insert(k.to_string(), v.to_string());
                    won += 1;
                }
            }
            msg(&format!("{k}={v}: {} kernels read it, better on {won} signatures", readers.len()));
        }
    }
    // verify the combined choices; drop the ones that don't hold up
    tune::set_entries(best.clone());
    let (p1, _) = build_forced(rt, &mut build, with_step(&Choice::new())).expect("tuned lowers");
    let tuned_sample: BTreeMap<&str, Vec<usize>> =
        all.iter().filter(|(s, _)| best.contains_key(**s)).map(|(s, v)| (*s, v.clone())).collect();
    let t1 = time_sigs(rt, &p1, &tuned_sample, reps, pos, set_pos);
    drop(p1);
    let mut sigs = Vec::new();
    let (mut sum0, mut sum1) = (0.0, 0.0);
    for (sig, _) in &all {
        let n = instances[sig] as f64;
        let d0 = &t0[*sig];
        let d: f64 = d0.iter().sum::<f64>() / d0.len() as f64;
        let (ms, choice) = match (best.get(*sig), t1.get(*sig)) {
            (Some(c), Some(v)) if wins(v, d0) => (v.iter().sum::<f64>() / v.len() as f64, c.clone()),
            _ => (d, Choice::new()),
        };
        sum0 += d * n;
        sum1 += ms * n;
        if !choice.is_empty() {
            sigs.push(SigResult {
                sig: sig.to_string(),
                instances: instances[sig],
                default_ms: d,
                tuned_ms: ms,
                choice,
            });
        }
    }
    sigs.sort_by(|a, b| {
        let g = |s: &SigResult| (s.default_ms - s.tuned_ms) * s.instances as f64;
        g(b).partial_cmp(&g(a)).unwrap()
    });
    let keep: HashMap<String, Choice> = sigs.iter().map(|s| (s.sig.clone(), s.choice.clone())).collect();
    let mut entries = keep;
    if !step_choice.is_empty() {
        entries.insert(format!("step:{step_name}"), step_choice.clone());
    }
    tune::set_entries(entries);
    Tuned {
        step_choice,
        sigs,
        sum_default: sum0,
        sum_tuned: sum1,
    }
}

/// Merge `t` into the cache file at `path` (other entries kept) and write it.
pub fn write_cache(path: &std::path::Path, gpu: &str, step_name: &str, t: &Tuned) {
    let old = std::fs::read_to_string(path).unwrap_or_default();
    let ours: std::collections::HashSet<&str> = t.sigs.iter().map(|s| s.sig.as_str()).collect();
    let step_key = format!("step:{step_name}");
    let mut lines: Vec<String> = old
        .lines()
        .filter(|l| {
            let sig = l.split('\t').next().unwrap_or("");
            !l.starts_with('#') && !l.trim().is_empty() && !ours.contains(sig) && sig != step_key
        })
        .map(|l| l.to_string())
        .collect();
    let kv = |c: &Choice| c.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",");
    if !t.step_choice.is_empty() {
        lines.push(format!("{step_key}\t{}\t\t\t{step_name} step", kv(&t.step_choice)));
    }
    for s in &t.sigs {
        lines.push(format!(
            "{}\t{}\t{:.5}\t{:.5}\t{step_name} x{}",
            s.sig,
            kv(&s.choice),
            s.tuned_ms,
            s.default_ms,
            s.instances
        ));
    }
    lines.sort();
    let head = format!(
        "# tang tuning cache for {gpu}: sig<TAB>knobs<TAB>tuned ms<TAB>default ms<TAB>note\n\
         # written by the autotuner (crates/tang-bench/src/autotune.rs); TANG_TUNE=0 ignores it\n"
    );
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(path, head + &lines.join("\n") + "\n").unwrap();
}

/// Print a short summary.
pub fn report(t: &Tuned) {
    println!(
        "autotune: kernel sum {:.3} -> {:.3} ms; step choice {:?}; {} signatures changed",
        t.sum_default,
        t.sum_tuned,
        t.step_choice,
        t.sigs.len()
    );
    for s in t.sigs.iter().take(15) {
        println!(
            "  {} x{:<3} {:.4} -> {:.4} ms  {:?}",
            s.sig, s.instances, s.default_ms, s.tuned_ms, s.choice
        );
    }
}
