//! `TANG_GLM5_TRACE=1`: where a GLM-5 window's time goes, per stage of every layer family.
//!
//! The GLM-5 twin of the Coder-Next tracing (`TANG_CODER_PROF` per-stage GPU events +
//! [`crate::telemetry`]'s otel spans). Every window builds a span tree:
//!
//! ```text
//! glm5.window (t, pos0, phase)
//! ├─ embed
//! ├─ layer (l, mixer)
//! │  ├─ mhc_pre.attn      ├─ kda ─ proj | gates | core | out_gate | o_proj
//! │  │                    └─ dsa ─ q_a | kv_append | idx_append | idx_select | q_b | attend | o_proj
//! │  ├─ mhc_post.attn
//! │  ├─ mhc_pre.ffn
//! │  ├─ ffn ─ moe.shared | moe.router | moe.experts | moe.combine  (or ffn.dense)
//! │  └─ mhc_post.ffn
//! └─ head ─ head.mean | head.norm | lm_head
//! ```
//!
//! Each span carries its host wall time, the GPU time of the kernels launched inside it
//! (two timing events per launch, [`tang_moe::gpu::trace`]), launch count, host→device and
//! device→host bytes and calls, the host time inside synchronous copies, and device
//! allocations. Kernels whose launch site declares the bytes it moves
//! ([`tang_moe::gpu::trace::bytes`]: the weight GEMVs, the experts, MLA's attend) also get
//! GB/s and the share of the device's memory bandwidth (SOL; `TANG_SOL_GBS` overrides).
//!
//! Output: [`report`] prints the end-of-run bottleneck table on stderr (the CLI calls it);
//! with the `otel` feature every window's tree also goes out through
//! [`crate::telemetry::glm5_tree`] (`tang-otel`: OTLP/JSON lines to `OTEL_FILE`).
//!
//! Off (the default) every hook is one relaxed atomic load. On, the cost is a few µs per
//! launch (two events + a capture check) and one event sync per launch at window end.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::{Instant, SystemTime};

static ON: AtomicBool = AtomicBool::new(false);

/// A copy of [`tang_moe::gpu::trace::Snap`] that exists without `cuda`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Io {
    pub launches: u64,
    pub h2d_bytes: u64,
    pub h2d_calls: u64,
    pub d2h_bytes: u64,
    pub d2h_calls: u64,
    pub copy_ns: u64,
    pub allocs: u64,
    pub alloc_ns: u64,
    pub syncs: u64,
    pub sync_ns: u64,
}

impl Io {
    fn now() -> Io {
        #[cfg(feature = "cuda")]
        {
            let s = tang_moe::gpu::trace::snap();
            Io {
                launches: s.launches,
                h2d_bytes: s.h2d_bytes,
                h2d_calls: s.h2d_calls,
                d2h_bytes: s.d2h_bytes,
                d2h_calls: s.d2h_calls,
                copy_ns: s.copy_ns,
                allocs: s.allocs,
                alloc_ns: s.alloc_ns,
                syncs: s.syncs,
                sync_ns: s.sync_ns,
            }
        }
        #[cfg(not(feature = "cuda"))]
        Io::default()
    }
    fn since(&self, e: &Io) -> Io {
        Io {
            launches: self.launches - e.launches,
            h2d_bytes: self.h2d_bytes - e.h2d_bytes,
            h2d_calls: self.h2d_calls - e.h2d_calls,
            d2h_bytes: self.d2h_bytes - e.d2h_bytes,
            d2h_calls: self.d2h_calls - e.d2h_calls,
            copy_ns: self.copy_ns - e.copy_ns,
            allocs: self.allocs - e.allocs,
            alloc_ns: self.alloc_ns - e.alloc_ns,
            syncs: self.syncs - e.syncs,
            sync_ns: self.sync_ns - e.sync_ns,
        }
    }
    fn add(&mut self, o: &Io) {
        self.launches += o.launches;
        self.h2d_bytes += o.h2d_bytes;
        self.h2d_calls += o.h2d_calls;
        self.d2h_bytes += o.d2h_bytes;
        self.d2h_calls += o.d2h_calls;
        self.copy_ns += o.copy_ns;
        self.allocs += o.allocs;
        self.alloc_ns += o.alloc_ns;
        self.syncs += o.syncs;
        self.sync_ns += o.sync_ns;
    }
}

/// One span of the current window.
struct Span {
    name: &'static str,
    parent: Option<usize>,
    /// The interned stage path ([`State::path_of`]; [`NO_PATH`] for the window/layer levels).
    pid: u32,
    layer: Option<usize>,
    t0: Instant,
    t1: Instant,
    io0: Io,
    io: Io,
    /// Inclusive GPU ms (own launches + descendants'), filled at window end.
    gpu_ms: f64,
    /// Inclusive declared bytes, and the GPU ms of exactly those launches.
    bytes: u64,
    bytes_ms: f64,
}

#[derive(Default, Clone)]
struct Agg {
    calls: u64,
    wall_ms: f64,
    gpu_ms: f64,
    bytes: u64,
    bytes_ms: f64,
    io: Io,
}

#[derive(Default, Clone)]
struct KAgg {
    launches: u64,
    gpu_ms: f64,
    bytes: u64,
    bytes_ms: f64,
}

#[derive(Default)]
struct Phase {
    windows: u64,
    tokens: u64,
    wall_ms: f64,
    gpu_ms: f64,
    io: Io,
    /// By interned stage path (below the layer level: `kda/proj`, `ffn/moe.experts`, ...).
    stages: Vec<Agg>,
    /// By (kernel, interned stage path). Names are looked up only at [`report`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    kernels: HashMap<(usize, u32), KAgg>,
}

const NO_PATH: u32 = u32::MAX;

struct State {
    spans: Vec<Span>,
    stack: Vec<usize>,
    /// `[decode, prefill]`.
    phases: [Phase; 2],
    base: (Instant, SystemTime),
    t_first: Option<Instant>,
    t_last: Instant,
    /// The memory bandwidth SOL is measured against, GB/s, and where it came from.
    sol: Option<(f64, String)>,
    /// Stage paths interned by (parent path, name): no allocation per span once warm.
    path_of: HashMap<(u32, &'static str), u32>,
    paths: Vec<String>,
}

thread_local! {
    static ST: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Is tracing on? (`TANG_GLM5_TRACE=1`, read by [`init`].)
#[inline(always)]
pub fn on() -> bool {
    ON.load(Relaxed)
}

/// Read `TANG_GLM5_TRACE` and start tracing on this thread (the engine's). Call before the
/// GPU loads, so kernel names are recorded. Returns whether it's on.
pub fn init() -> bool {
    let want = std::env::var("TANG_GLM5_TRACE").is_ok_and(|v| !v.is_empty() && v != "0");
    if !want {
        return false;
    }
    start();
    true
}

/// Start tracing regardless of the environment (tests, benches).
pub fn start() {
    #[cfg(feature = "cuda")]
    tang_moe::gpu::trace::enable(true);
    ST.with(|s| {
        *s.borrow_mut() = Some(State {
            spans: Vec::new(),
            stack: Vec::new(),
            phases: Default::default(),
            base: (Instant::now(), SystemTime::now()),
            t_first: None,
            t_last: Instant::now(),
            sol: None,
            path_of: HashMap::new(),
            paths: Vec::new(),
        })
    });
    ON.store(true, Relaxed);
}

/// Stop tracing (the collected totals stay for [`report`]).
pub fn stop() {
    ON.store(false, Relaxed);
    #[cfg(feature = "cuda")]
    tang_moe::gpu::trace::enable(false);
}

/// Name the bandwidth ceiling (GB/s) and its source (e.g. the device name).
pub fn set_sol(gbs: f64, what: String) {
    with(|s| s.sol = Some((gbs, what)));
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    ST.with(|s| s.borrow_mut().as_mut().map(f))
}

/// Closes its span on drop.
#[must_use]
pub struct Guard(bool);

impl Drop for Guard {
    #[inline(always)]
    fn drop(&mut self) {
        if self.0 {
            close();
        }
    }
}

/// Open a child span of the current one (no-op when off).
#[inline(always)]
pub fn span(name: &'static str) -> Guard {
    if !on() {
        return Guard(false);
    }
    open(name, None)
}

/// [`span`] with a layer index (the `layer` spans).
#[inline(always)]
pub fn span_l(name: &'static str, layer: usize) -> Guard {
    if !on() {
        return Guard(false);
    }
    open(name, Some(layer))
}

fn open(name: &'static str, layer: Option<usize>) -> Guard {
    let ok = with(|s| {
        let i = s.spans.len();
        let parent = s.stack.last().copied();
        // the window and layer levels carry no path; their children start one
        let pid = if name == "glm5.window" || name == "layer" {
            NO_PATH
        } else {
            let pp = parent.map_or(NO_PATH, |p| s.spans[p].pid);
            let next = s.paths.len() as u32;
            let id = *s.path_of.entry((pp, name)).or_insert(next);
            if id == next {
                let full = match pp {
                    NO_PATH => name.to_string(),
                    _ => format!("{}/{name}", s.paths[pp as usize]),
                };
                s.paths.push(full);
            }
            id
        };
        s.spans.push(Span {
            name,
            parent,
            pid,
            layer,
            t0: Instant::now(),
            t1: Instant::now(),
            io0: Io::now(),
            io: Io::default(),
            gpu_ms: 0.0,
            bytes: 0,
            bytes_ms: 0.0,
        });
        s.stack.push(i);
        #[cfg(feature = "cuda")]
        tang_moe::gpu::trace::tag(i as u32 + 1);
    })
    .is_some();
    Guard(ok)
}

fn close() {
    with(|s| {
        let Some(i) = s.stack.pop() else { return };
        let sp = &mut s.spans[i];
        sp.t1 = Instant::now();
        sp.io = Io::now().since(&sp.io0);
        #[cfg(feature = "cuda")]
        tang_moe::gpu::trace::tag(s.stack.last().map_or(0, |&p| p as u32 + 1));
    });
}

/// Open the window's root span. Pair with [`window_end`].
pub fn window_begin(t: usize, pos0: usize) -> Guard {
    if !on() {
        return Guard(false);
    }
    with(|s| {
        s.t_first.get_or_insert_with(Instant::now);
        s.spans.clear();
        s.stack.clear();
    });
    let _ = (t, pos0);
    open("glm5.window", None)
}

/// Close out the window: resolve its launches' GPU time, fold the tree into the run's
/// totals and export it (otel). `root` is [`window_begin`]'s guard.
pub fn window_end(root: Guard, t: usize, pos0: usize) {
    if !root.0 {
        return;
    }
    drop(root);
    #[cfg(feature = "cuda")]
    let launches = tang_moe::gpu::trace::drain();
    with(|s| {
        s.t_last = Instant::now();
        let n = s.spans.len();
        // own GPU time + declared bytes per span (tag = span index + 1)
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut own: Vec<(f64, u64, f64)> = vec![(0.0, 0, 0.0); n];
        #[cfg(feature = "cuda")]
        let mut kern: Vec<(usize, usize, f64, u64)> = Vec::with_capacity(launches.len());
        #[cfg(feature = "cuda")]
        for l in &launches {
            let Some(i) = (l.tag as usize).checked_sub(1).filter(|&i| i < n) else {
                continue;
            };
            let ms = l.gpu_ms.unwrap_or(0.0) as f64;
            own[i].0 += ms;
            if l.bytes > 0 {
                own[i].1 += l.bytes;
                own[i].2 += ms;
            }
            kern.push((i, l.func, ms, l.bytes));
        }
        // inclusive: children follow their parents in `spans`, so a reverse pass folds up
        for i in (0..n).rev() {
            let (g, b, bm) = own[i];
            s.spans[i].gpu_ms += g;
            s.spans[i].bytes += b;
            s.spans[i].bytes_ms += bm;
            if let Some(p) = s.spans[i].parent {
                let (g, b, bm) = (s.spans[i].gpu_ms, s.spans[i].bytes, s.spans[i].bytes_ms);
                s.spans[p].gpu_ms += g;
                s.spans[p].bytes += b;
                s.spans[p].bytes_ms += bm;
            }
        }
        let n_paths = s.paths.len();
        let ph = &mut s.phases[if t == 1 { 0 } else { 1 }];
        ph.stages.resize(n_paths, Agg::default());
        let root = &s.spans[0];
        ph.windows += 1;
        ph.tokens += t as u64;
        ph.wall_ms += ms(root.t0, root.t1);
        ph.gpu_ms += root.gpu_ms;
        ph.io.add(&root.io);
        for sp in s.spans.iter().skip(1) {
            if sp.pid == NO_PATH {
                continue;
            }
            let a = &mut ph.stages[sp.pid as usize];
            a.calls += 1;
            a.wall_ms += ms(sp.t0, sp.t1);
            a.gpu_ms += sp.gpu_ms;
            a.bytes += sp.bytes;
            a.bytes_ms += sp.bytes_ms;
            a.io.add(&sp.io);
        }
        #[cfg(feature = "cuda")]
        for (i, f, gms, b) in kern {
            let k = ph.kernels.entry((f, s.spans[i].pid)).or_default();
            k.launches += 1;
            k.gpu_ms += gms;
            if b > 0 {
                k.bytes += b;
                k.bytes_ms += gms;
            }
        }
        export(s, t, pos0);
    });
}

fn ms(a: Instant, b: Instant) -> f64 {
    b.duration_since(a).as_secs_f64() * 1e3
}

fn export(s: &State, t: usize, pos0: usize) {
    if !crate::telemetry::enabled() {
        return;
    }
    let at = |i: Instant| s.base.1 + i.duration_since(s.base.0);
    let tree: Vec<crate::telemetry::TreeSpan> = s
        .spans
        .iter()
        .enumerate()
        .map(|(i, sp)| {
            let mut ints = vec![
                ("gpu.launches", sp.io.launches as i64),
                ("h2d.bytes", sp.io.h2d_bytes as i64),
                ("h2d.calls", sp.io.h2d_calls as i64),
                ("d2h.bytes", sp.io.d2h_bytes as i64),
                ("d2h.calls", sp.io.d2h_calls as i64),
                ("gpu.allocs", sp.io.allocs as i64),
            ];
            let mut floats = vec![
                ("gpu.ms", sp.gpu_ms),
                ("copy.ms", sp.io.copy_ns as f64 / 1e6),
                ("alloc.ms", sp.io.alloc_ns as f64 / 1e6),
            ];
            if i == 0 {
                ints.push(("tokens", t as i64));
                ints.push(("pos0", pos0 as i64));
            } else if let Some(l) = sp.layer {
                ints.push(("layer", l as i64));
            }
            if sp.bytes > 0 && sp.bytes_ms > 0.0 {
                let gbs = sp.bytes as f64 / sp.bytes_ms / 1e6;
                floats.push(("gb_per_s", gbs));
                if let Some((sol, _)) = &s.sol {
                    floats.push(("sol.pct", 100.0 * gbs / sol));
                }
            }
            crate::telemetry::TreeSpan {
                name: sp.name,
                parent: sp.parent,
                start: at(sp.t0),
                end: at(sp.t1),
                ints,
                floats,
            }
        })
        .collect();
    crate::telemetry::glm5_tree(&tree, if t == 1 { "decode" } else { "prefill" });
}

/// The end-of-run bottleneck table (`None` when tracing never ran a window).
pub fn report() -> Option<String> {
    with(|s| render(s)).flatten()
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn render(s: &State) -> Option<String> {
    use std::fmt::Write;
    let first = s.t_first?;
    let mut o = String::new();
    let run_ms = ms(first, s.t_last);
    let (sol, sol_src) = s.sol.clone().unwrap_or((0.0, "unknown".into()));
    let all_gpu: f64 = s.phases.iter().map(|p| p.gpu_ms).sum();
    let _ =
        writeln!(
        o,
        "\n== glm5 trace (TANG_GLM5_TRACE) — {:.2} s traced, GPU busy {:.2} s ({:.0}%); SOL {} ==",
        run_ms / 1e3,
        all_gpu / 1e3,
        100.0 * all_gpu / run_ms.max(1e-9),
        if sol > 0.0 { format!("{sol:.0} GB/s ({sol_src})") } else { "unknown".into() },
    );
    for (pi, ph) in s.phases.iter().enumerate() {
        if ph.windows == 0 {
            continue;
        }
        // decode is shown per token (what tok/s is made of); prefill as totals
        // names for the interned paths and kernels (only here, off the per-window path)
        let stages: BTreeMap<String, Agg> = ph
            .stages
            .iter()
            .enumerate()
            .filter(|(_, a)| a.calls > 0)
            .map(|(i, a)| (s.paths[i].clone(), a.clone()))
            .collect();
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut kernels: BTreeMap<(String, String), KAgg> = BTreeMap::new();
        #[cfg(feature = "cuda")]
        for (&(f, pid), a) in &ph.kernels {
            let path = s.paths.get(pid as usize).cloned().unwrap_or_default();
            kernels.insert((tang_moe::gpu::trace::name(f), path), a.clone());
        }
        let (label, div) = if pi == 0 {
            ("decode, per token", ph.windows as f64)
        } else {
            ("prefill, totals", 1.0)
        };
        let _ = writeln!(
            o,
            "\n-- {label}: {} windows, {} tokens, wall {:.1} ms{}, GPU {:.1} ms ({:.0}% of wall), \
             {} launches, H2D {:.2} MB / {} calls, D2H {:.2} MB / {} calls, sync-copy {:.1} ms, \
             allocs {} ({:.1} ms), syncs {} ({:.1} ms)",
            ph.windows,
            ph.tokens,
            ph.wall_ms / div,
            if pi == 0 {
                format!(
                    " ({:.2} tok/s)",
                    1e3 * ph.windows as f64 / ph.wall_ms.max(1e-9)
                )
            } else {
                String::new()
            },
            ph.gpu_ms / div,
            100.0 * ph.gpu_ms / ph.wall_ms.max(1e-9),
            (ph.io.launches as f64 / div).round(),
            mb(ph.io.h2d_bytes) / div,
            (ph.io.h2d_calls as f64 / div).round(),
            mb(ph.io.d2h_bytes) / div,
            (ph.io.d2h_calls as f64 / div).round(),
            ph.io.copy_ns as f64 / 1e6 / div,
            (ph.io.allocs as f64 / div).round(),
            ph.io.alloc_ns as f64 / 1e6 / div,
            (ph.io.syncs as f64 / div).round(),
            ph.io.sync_ns as f64 / 1e6 / div,
        );
        let _ = writeln!(
            o,
            "{:<26} {:>6} {:>10} {:>6} {:>9} {:>7} {:>6} {:>6} {:>6} {:>9} {:>9} {:>8} {:>8} {:>8} {:>6}",
            "stage", "calls", "wall ms", "%wall", "gpu ms", "launch", "h2d#", "d2h#", "sync#", "H2D MB", "D2H MB", "copy ms",
            "alloc ms", "GB/s", "%SOL"
        );
        let mut rows: Vec<(&String, &Agg)> = stages.iter().collect();
        // tree order (paths sort parents before children), each depth by wall
        rows.sort_by(|a, b| {
            let (ka, kb) = (sort_key(&stages, a.0), sort_key(&stages, b.0));
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (p, a) in rows {
            let depth = p.matches('/').count();
            let leaf = p.rsplit('/').next().unwrap_or(p);
            let name = format!("{}{}", "  ".repeat(depth), leaf);
            let gbs = (a.bytes > 0 && a.bytes_ms > 0.0).then(|| a.bytes as f64 / a.bytes_ms / 1e6);
            let _ = writeln!(
                o,
                "{:<26} {:>6} {:>10.2} {:>5.1}% {:>9.2} {:>7.0} {:>6.0} {:>6.0} {:>6.0} {:>9.3} {:>9.3} {:>8.2} {:>8.2} {:>8} {:>6}",
                name,
                (a.calls as f64 / div).round(),
                a.wall_ms / div,
                100.0 * a.wall_ms / ph.wall_ms.max(1e-9),
                a.gpu_ms / div,
                a.io.launches as f64 / div,
                a.io.h2d_calls as f64 / div,
                a.io.d2h_calls as f64 / div,
                a.io.syncs as f64 / div,
                mb(a.io.h2d_bytes) / div,
                mb(a.io.d2h_bytes) / div,
                a.io.copy_ns as f64 / 1e6 / div,
                a.io.alloc_ns as f64 / 1e6 / div,
                gbs.map_or("-".into(), |g| format!("{g:.0}")),
                match gbs {
                    Some(g) if sol > 0.0 => format!("{:.1}", 100.0 * g / sol),
                    _ => "-".into(),
                },
            );
        }
        if !kernels.is_empty() {
            let _ = writeln!(o, "\nkernels by GPU time ({label}):");
            let _ = writeln!(
                o,
                "{:<24} {:<26} {:>7} {:>9} {:>8} {:>6} {:>8} {:>6}",
                "kernel", "stage", "launch", "gpu ms", "avg µs", "%gpu", "GB/s", "%SOL"
            );
            let mut ks: Vec<_> = kernels.iter().collect();
            ks.sort_by(|a, b| {
                b.1.gpu_ms
                    .partial_cmp(&a.1.gpu_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for ((k, st), a) in ks.iter().take(24) {
                let gbs =
                    (a.bytes > 0 && a.bytes_ms > 0.0).then(|| a.bytes as f64 / a.bytes_ms / 1e6);
                let _ = writeln!(
                    o,
                    "{:<24} {:<26} {:>7.0} {:>9.2} {:>8.1} {:>5.1}% {:>8} {:>6}",
                    k,
                    st,
                    a.launches as f64 / div,
                    a.gpu_ms / div,
                    1e3 * a.gpu_ms / a.launches.max(1) as f64,
                    100.0 * a.gpu_ms / ph.gpu_ms.max(1e-9),
                    gbs.map_or("-".into(), |g| format!("{g:.0}")),
                    match gbs {
                        Some(g) if sol > 0.0 => format!("{:.1}", 100.0 * g / sol),
                        _ => "-".into(),
                    },
                );
            }
        }
        // the ranking: top-level stages by wall, how much of it the GPU covers
        let _ = writeln!(
            o,
            "\nbottlenecks ({label}; top-level stages by wall, host = wall − GPU):"
        );
        let mut top: Vec<(&String, &Agg)> =
            stages.iter().filter(|(p, _)| !p.contains('/')).collect();
        top.sort_by(|a, b| {
            b.1.wall_ms
                .partial_cmp(&a.1.wall_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, (p, a)) in top.iter().take(8).enumerate() {
            let host = (a.wall_ms - a.gpu_ms).max(0.0);
            let _ = writeln!(
                o,
                "  {}. {:<16} {:>9.2} ms {:>5.1}%  gpu {:>8.2}  host {:>8.2}  copies {:.2} MB  allocs {:.0}",
                i + 1,
                p,
                a.wall_ms / div,
                100.0 * a.wall_ms / ph.wall_ms.max(1e-9),
                a.gpu_ms / div,
                host / div,
                mb(a.io.h2d_bytes + a.io.d2h_bytes) / div,
                a.io.allocs as f64 / div,
            );
        }
        let covered: f64 = top.iter().map(|(_, a)| a.wall_ms).sum();
        let _ = writeln!(
            o,
            "  (untraced glue between stages: {:.2} ms)",
            (ph.wall_ms - covered).max(0.0) / div
        );
    }
    Some(o)
}

/// Sort key that keeps each subtree under its parent: the chain of (−wall) of every
/// ancestor prefix, then the path itself.
fn sort_key(stages: &BTreeMap<String, Agg>, p: &str) -> (Vec<i64>, String) {
    let mut key = Vec::new();
    let mut acc = String::new();
    for (i, part) in p.split('/').enumerate() {
        if i > 0 {
            acc.push('/');
        }
        acc.push_str(part);
        let w = stages.get(&acc).map_or(0.0, |a| a.wall_ms);
        key.push(-(w * 1e3) as i64);
    }
    (key, p.to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    fn tree_folds_and_renders() {
        super::start();
        let root = super::window_begin(1, 0);
        {
            let _l = super::span_l("layer", 0);
            {
                let _k = super::span("kda");
                let _p = super::span("proj");
            }
            let _f = super::span("ffn");
        }
        {
            let _h = super::span("head");
        }
        super::window_end(root, 1, 0);
        let r = super::report().expect("a report");
        eprintln!("{r}");
        super::stop();
        assert!(r.contains("decode, per token"), "{r}");
        assert!(r.contains("  proj"), "{r}");
        assert!(r.contains("bottlenecks"), "{r}");
    }
}
