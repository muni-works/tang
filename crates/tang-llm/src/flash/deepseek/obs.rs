//! The step timeline: where a DeepSeek forward's time goes, without OpenTelemetry
//! (docs/observability.md, "DeepSeek").
//!
//! Each forward is a step (`prefill`, `decode`, `verify` or `draft`) timed two ways: CUDA
//! events recorded on the model's stream at its boundaries (read only once the stream has
//! drained, after the step's own read-back sync or lazily at the next step's start, so no
//! sync is added) and host wall-clock. On a direct step at `DS_OBS=2` the interior is spans
//! too, nested by name: `layer{i}` > `attn` > `q`, `window_kv`, `compress` (> `indexer`),
//! `sparse_attn`, `wo`; `moe` > `router`, `route`, `grouped_gu`, `grouped_down` (or the
//! per-expert `experts`), `reduce`, `shared`, `finish`; `hc`; `engram`; plus step-level
//! `embed`, `head`. A replayed (or captured) graph is one opaque launch: its interior is
//! never recorded (no events inside a capture), so `DS_OBS_DIRECT=1` forces the direct
//! path for a profiled run.
//!
//! Counters (kernel launches, host syncs, D2H / H2D bytes, device allocations, graph
//! captures / replays, expert tensors uploaded) are thread-local and always counted; a step
//! reports its deltas. Each EP rank runs on its own thread, so its counts are its own.
//!
//! - `DS_OBS=1`: a summary line per step on stderr (rank 0).
//! - `DS_OBS=2`: interior spans on every `DS_OBS_EVERY`-th step (default 50) and a per-op
//!   breakdown printed for it.
//! - `DS_OBS_FILE=path`: every observed step (all ranks) as a JSON line.
//! - With the `otel` feature and telemetry installed, every `DS_OBS_EVERY`-th step is
//!   observed even at `DS_OBS=0` and becomes a `forward` span under the request's `decode`.
//!
//! Off (`DS_OBS` unset, no telemetry): a step costs a few relaxed atomic loads and
//! thread-local increments.

use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Mutex, Once};
use std::time::{Instant, SystemTime};
use tang_moe::gpu::{self, Event, Stream};

static INIT: Once = Once::new();
static LEVEL: AtomicU8 = AtomicU8::new(0);
static EVERY: AtomicU64 = AtomicU64::new(50);
static DIRECT: AtomicBool = AtomicBool::new(false);
/// No stderr lines (the profile command prints its own table).
static QUIET: AtomicBool = AtomicBool::new(false);
static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);
/// Every observed step, when [`collect`] turned it on (the profile command).
static SINK: Mutex<Option<Vec<StepRec>>> = Mutex::new(None);

fn init() {
    INIT.call_once(|| {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok());
        LEVEL.store(num("DS_OBS").unwrap_or(0).min(2) as u8, Relaxed);
        EVERY.store(num("DS_OBS_EVERY").unwrap_or(50).max(1), Relaxed);
        DIRECT.store(
            std::env::var("DS_OBS_DIRECT").is_ok_and(|v| v == "1"),
            Relaxed,
        );
        if let Ok(p) = std::env::var("DS_OBS_FILE") {
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&p)
            {
                Ok(f) => *FILE.lock().unwrap() = Some(f),
                Err(e) => eprintln!("DS_OBS_FILE {p}: {e}"),
            }
        }
    });
}

/// `DS_OBS`: 0 off, 1 step lines, 2 interior spans every [`every`] steps.
pub fn level() -> u8 {
    init();
    LEVEL.load(Relaxed)
}

pub fn every() -> u64 {
    init();
    EVERY.load(Relaxed)
}

/// `DS_OBS_DIRECT=1`: decode steps skip CUDA graphs so their interior can be timed.
pub fn force_direct() -> bool {
    init();
    DIRECT.load(Relaxed)
}

/// Override the environment (the profile command, tests).
pub fn set(level: u8, every: u64, direct: bool) {
    init();
    LEVEL.store(level.min(2), Relaxed);
    EVERY.store(every.max(1), Relaxed);
    DIRECT.store(direct, Relaxed);
}

/// Keep stderr quiet whatever the level.
pub fn quiet(on: bool) {
    QUIET.store(on, Relaxed);
}

/// Keep every observed step (any thread) for [`take`].
pub fn collect(on: bool) {
    *SINK.lock().unwrap() = on.then(Vec::new);
}

/// Rank 0's kept steps of these kinds so far.
pub fn kept(kinds: &[&str]) -> usize {
    SINK.lock().unwrap().as_ref().map_or(0, |v| {
        v.iter()
            .filter(|r| r.rank == 0 && kinds.contains(&r.kind))
            .count()
    })
}

/// The steps kept since [`collect`].
pub fn take() -> Vec<StepRec> {
    SINK.lock()
        .unwrap()
        .as_mut()
        .map(std::mem::take)
        .unwrap_or_default()
}

// counters

/// Per-thread totals; a step reports its deltas.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct Counts {
    pub launches: u64,
    pub syncs: u64,
    pub d2h_bytes: u64,
    pub h2d_bytes: u64,
    pub allocs: u64,
    pub captures: u64,
    pub replays: u64,
    pub expert_uploads: u64,
}

impl Counts {
    fn minus(self, o: Counts) -> Counts {
        Counts {
            launches: self.launches - o.launches,
            syncs: self.syncs - o.syncs,
            d2h_bytes: self.d2h_bytes - o.d2h_bytes,
            h2d_bytes: self.h2d_bytes - o.h2d_bytes,
            allocs: self.allocs - o.allocs,
            captures: self.captures - o.captures,
            replays: self.replays - o.replays,
            expert_uploads: self.expert_uploads - o.expert_uploads,
        }
    }
}

thread_local! {
    static CNT: Cell<Counts> = const { Cell::new(Counts {
        launches: 0, syncs: 0, d2h_bytes: 0, h2d_bytes: 0, allocs: 0, captures: 0,
        replays: 0, expert_uploads: 0,
    }) };
}

#[inline(always)]
fn bump(f: impl FnOnce(&mut Counts)) {
    CNT.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    })
}

/// This thread's totals.
pub fn counts() -> Counts {
    CNT.with(|c| c.get())
}

#[inline(always)]
pub fn sync() {
    bump(|c| c.syncs += 1)
}
#[inline(always)]
pub fn d2h(bytes: usize) {
    bump(|c| c.d2h_bytes += bytes as u64)
}
#[inline(always)]
pub fn h2d(bytes: usize) {
    bump(|c| c.h2d_bytes += bytes as u64)
}
#[inline(always)]
pub fn alloc() {
    bump(|c| c.allocs += 1)
}
#[inline(always)]
pub fn capture() {
    bump(|c| c.captures += 1)
}
#[inline(always)]
pub fn replay() {
    bump(|c| c.replays += 1)
}
#[inline(always)]
pub fn expert_upload() {
    bump(|c| c.expert_uploads += 1)
}

/// [`gpu::launch`], counted.
///
/// # Safety
/// As [`gpu::launch`].
#[inline(always)]
pub unsafe fn launch(
    f: cudarc::driver::sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    s: &Stream,
    args: &mut [*mut c_void],
) -> gpu::Result<()> {
    bump(|c| c.launches += 1);
    // every kernel of these modules opens with DS_PDL_ENTRY (or ds_ks's prefetch + wait)
    let r = super::pdl::launch(f, grid, block, smem, s, args, super::pdl::on());
    if r.is_err() {
        eprintln!("tang-llm: launch failed: grid {grid:?} block {block:?} smem {smem}");
    }
    r
}

// records

/// One span of a step: `name` is its dotted path (`layer.attn.q`), `layer` the backbone
/// layer it is in (-1 outside one). `start_ms` is from the step's start event.
#[derive(Clone, Debug, Serialize)]
pub struct SpanRec {
    pub name: String,
    /// Index of the enclosing span in [`StepRec::spans`], -1 for the step.
    pub parent: i32,
    pub layer: i32,
    pub start_ms: f64,
    pub gpu_ms: f64,
    pub host_ms: f64,
}

/// One observed forward.
#[derive(Clone, Debug, Serialize)]
pub struct StepRec {
    /// prefill | decode | verify | draft
    pub kind: &'static str,
    pub t: usize,
    pub sp: usize,
    /// graph (captured or replayed) | direct
    pub path: &'static str,
    /// This step captured its graph (and then launched it).
    pub captured: bool,
    pub rank: usize,
    /// This model's forward count (steps, observed or not, from 0).
    pub step: u64,
    /// The EP group's command this step ran for (0 outside EP): equal on every rank.
    pub parent: u64,
    /// Host wall-clock from the step's start to its outputs read back.
    pub wall_ms: f64,
    /// Device time from the first to the last event of the step.
    pub gpu_ms: f64,
    /// `wall_ms - gpu_ms`: host time the device didn't overlap (hashing, staging, launch
    /// gaps, the read-back).
    pub host_ms: f64,
    /// Summed `reduce` spans (the EP all-reduce, enqueue to completion); direct path only.
    pub reduce_wait_ms: f64,
    pub arena_bytes: u64,
    pub counts: Counts,
    #[serde(skip)]
    pub start: SystemTime,
    pub spans: Vec<SpanRec>,
}

/// A speculative step: drafting, the verify window, the rewind (host wall-clock).
#[derive(Clone, Debug, Serialize)]
pub struct SpecRec {
    pub drafted: usize,
    pub accepted: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub rewind_ms: f64,
    pub rank: usize,
    #[serde(skip)]
    pub start: SystemTime,
}

// the tracer

struct Sp {
    name: &'static str,
    layer: i32,
    parent: i32,
    e0: usize,
    e1: usize,
    h0: Instant,
    host: f64,
}

struct Open {
    kind: &'static str,
    t: usize,
    sp: usize,
    path: &'static str,
    captured: bool,
    h0: Instant,
    start: SystemTime,
    c0: Counts,
    /// The counters at [`Obs::finish`]: a step emitted later must not take in what ran
    /// between its finish and the next step's start (the next step's leading sync, a
    /// verify's row reads, the rewind).
    c1: Option<Counts>,
    wall: Option<f64>,
    e1: Option<usize>,
    arena: u64,
}

#[derive(Default)]
struct St {
    pool: Vec<Event>,
    used: usize,
    spans: Vec<Sp>,
    stack: Vec<usize>,
    open: Option<Open>,
    layer: i32,
}

/// A model's tracer (one per [`super::engine::GpuModel`], on its thread).
#[derive(Default)]
pub struct Obs {
    /// Steps begun (observed or not).
    n: Cell<u64>,
    /// This step is observed / records its interior.
    on: Cell<bool>,
    interior: Cell<bool>,
    pub rank: Cell<usize>,
    pub parent: Cell<u64>,
    /// This model's (level, every, direct) instead of the process's (tests, benches).
    pub over: Cell<Option<(u8, u64, bool)>>,
    st: RefCell<St>,
}

pub const NONE: u32 = u32::MAX;

impl Obs {
    fn lvl(&self) -> u8 {
        self.over.get().map_or_else(level, |o| o.0)
    }

    fn ev(&self) -> u64 {
        self.over.get().map_or_else(every, |o| o.1.max(1))
    }

    /// Decode steps run directly (`DS_OBS_DIRECT`, or the override).
    pub fn direct(&self) -> bool {
        self.over.get().map_or_else(force_direct, |o| o.2)
    }

    fn event(st: &mut St, s: &Stream) -> Option<usize> {
        if st.used == st.pool.len() {
            st.pool.push(Event::new(true).ok()?);
        }
        let i = st.used;
        st.pool[i].record(s).ok()?;
        st.used += 1;
        Some(i)
    }

    /// Whether this step is observed.
    #[inline(always)]
    pub fn on(&self) -> bool {
        self.on.get()
    }

    /// Start a step. Call after the stream drained (the model's `begin_step`): it first
    /// finishes the last step if that was left pending.
    pub fn begin(&self, s: &Stream, kind: &'static str, t: usize, sp: usize) {
        self.flush();
        let n = self.n.get();
        self.n.set(n + 1);
        let lvl = self.lvl();
        let sampled = n % self.ev() == 0;
        let on = lvl > 0 || (sampled && crate::telemetry::ds_enabled());
        self.on.set(on);
        self.interior.set(false);
        if !on {
            return;
        }
        let mut st = self.st.borrow_mut();
        st.used = 0;
        st.spans.clear();
        st.stack.clear();
        st.layer = -1;
        let start = SystemTime::now();
        let h0 = Instant::now();
        let e0 = Self::event(&mut st, s);
        if e0.is_none() {
            self.on.set(false);
            return;
        }
        st.open = Some(Open {
            kind,
            t,
            sp,
            path: "direct",
            captured: false,
            h0,
            start,
            c0: counts(),
            c1: None,
            wall: None,
            e1: None,
            arena: 0,
        });
        // the step itself is span 0
        st.spans.push(Sp {
            name: "step",
            layer: -1,
            parent: -1,
            e0: 0,
            e1: 0,
            h0,
            host: 0.0,
        });
        self.interior.set(lvl >= 2 && sampled);
    }

    /// How the step runs. A graph's interior is not observable (and no event may be
    /// recorded into a capture).
    pub fn path(&self, graph: bool, captured: bool) {
        if !self.on.get() {
            return;
        }
        if graph {
            self.interior.set(false);
        }
        if let Some(o) = self.st.borrow_mut().open.as_mut() {
            o.path = if graph { "graph" } else { "direct" };
            o.captured = captured;
        }
    }

    /// Open a span (nested in the open one); [`NONE`] when the interior isn't recorded.
    #[inline(always)]
    pub fn span(&self, s: &Stream, name: &'static str, layer: Option<usize>) -> u32 {
        if !self.interior.get() {
            return NONE;
        }
        self.span_slow(s, name, layer)
    }

    fn span_slow(&self, s: &Stream, name: &'static str, layer: Option<usize>) -> u32 {
        let mut st = self.st.borrow_mut();
        let Some(e0) = Self::event(&mut st, s) else {
            return NONE;
        };
        if let Some(l) = layer {
            st.layer = l as i32;
        }
        let parent = st.stack.last().map_or(0, |&p| p as i32);
        let i = st.spans.len();
        let layer = st.layer;
        st.spans.push(Sp {
            name,
            layer,
            parent,
            e0,
            e1: usize::MAX,
            h0: Instant::now(),
            host: 0.0,
        });
        st.stack.push(i);
        i as u32
    }

    /// Close span `id` (and any left open inside it).
    #[inline(always)]
    pub fn end(&self, s: &Stream, id: u32) {
        if id == NONE {
            return;
        }
        let mut st = self.st.borrow_mut();
        let Some(e1) = Self::event(&mut st, s) else {
            return;
        };
        while let Some(top) = st.stack.pop() {
            st.spans[top].e1 = e1;
            st.spans[top].host = st.spans[top].h0.elapsed().as_secs_f64() * 1e3;
            if st.spans[top].name == "layer" {
                st.layer = -1;
            }
            if top == id as usize {
                break;
            }
        }
    }

    /// The step's device work is all enqueued.
    pub fn gpu_end(&self, s: &Stream, arena_bytes: usize) {
        if !self.on.get() {
            return;
        }
        let mut st = self.st.borrow_mut();
        let e1 = Self::event(&mut st, s);
        if let Some(o) = st.open.as_mut() {
            o.e1 = e1;
            o.arena = arena_bytes as u64;
        }
    }

    /// The step's outputs are on the host: stop its wall clock and emit it if its events
    /// completed (else at the next [`Self::begin`]).
    pub fn finish(&self) {
        if !self.on.get() {
            return;
        }
        if let Some(o) = self.st.borrow_mut().open.as_mut() {
            o.wall.get_or_insert(o.h0.elapsed().as_secs_f64() * 1e3);
            o.c1.get_or_insert_with(counts);
        }
        self.on.set(false);
        self.interior.set(false);
        self.flush();
    }

    /// Emit a finished step whose events have completed.
    fn flush(&self) {
        let rec = {
            let mut st = self.st.borrow_mut();
            let ready = match &st.open {
                Some(o) => o.wall.is_some() && o.e1.is_some_and(|e| st.pool[e].done()),
                None => false,
            };
            if !ready {
                // a step that never finished (an error) is dropped at the next begin
                if st.open.as_ref().is_some_and(|o| o.wall.is_none()) && !self.on.get() {
                    st.open = None;
                }
                return;
            }
            let o = st.open.take().unwrap();
            let e1 = o.e1.unwrap();
            let base = &st.pool[0];
            let ms = |a: usize, b: usize| -> f64 {
                if b == usize::MAX {
                    return 0.0;
                }
                st.pool[b].since(&st.pool[a]).unwrap_or(0.0) as f64
            };
            let gpu_ms = ms(0, e1);
            let wall = o.wall.unwrap();
            let mut names: Vec<String> = Vec::with_capacity(st.spans.len());
            let mut spans = Vec::new();
            for (i, sp) in st.spans.iter().enumerate() {
                let name = if i == 0 {
                    "step".to_string()
                } else if sp.parent <= 0 {
                    sp.name.to_string()
                } else {
                    format!("{}.{}", names[sp.parent as usize], sp.name)
                };
                if i > 0 {
                    spans.push(SpanRec {
                        name: name.clone(),
                        parent: sp.parent - 1,
                        layer: sp.layer,
                        start_ms: st.pool[sp.e0].since(base).unwrap_or(0.0) as f64,
                        gpu_ms: ms(sp.e0, sp.e1),
                        host_ms: sp.host,
                    });
                }
                names.push(name);
            }
            let reduce_wait_ms = spans
                .iter()
                .filter(|s| s.name.ends_with(".reduce"))
                .map(|s| s.gpu_ms)
                .sum();
            StepRec {
                kind: o.kind,
                t: o.t,
                sp: o.sp,
                path: o.path,
                captured: o.captured,
                rank: self.rank.get(),
                step: self.n.get() - 1,
                parent: self.parent.get(),
                wall_ms: wall,
                gpu_ms,
                host_ms: (wall - gpu_ms).max(0.0),
                reduce_wait_ms,
                arena_bytes: o.arena,
                counts: o.c1.unwrap_or_else(counts).minus(o.c0),
                start: o.start,
                spans,
            }
        };
        emit(rec);
    }
}

fn emit(rec: StepRec) {
    let lvl = level();
    if lvl >= 1 && rec.rank == 0 && !QUIET.load(Relaxed) {
        let c = &rec.counts;
        eprintln!(
            "step {} t={} sp={} path={}{} {:.2}ms gpu={:.2}ms host={:.2}ms launches={} syncs={} d2h={} h2d={} allocs={}{}",
            rec.kind,
            rec.t,
            rec.sp,
            rec.path,
            if rec.captured { "(capture)" } else { "" },
            rec.wall_ms,
            rec.gpu_ms,
            rec.host_ms,
            c.launches,
            c.syncs,
            c.d2h_bytes,
            c.h2d_bytes,
            c.allocs,
            if c.expert_uploads > 0 {
                format!(" expert_uploads={}", c.expert_uploads)
            } else {
                String::new()
            },
        );
        if lvl >= 2 && !rec.spans.is_empty() {
            for (name, ms, n) in breakdown(&rec) {
                eprintln!(
                    "  {name:<28} {ms:>9.3}ms {:>5.1}%  x{n}",
                    100.0 * ms / rec.gpu_ms.max(1e-9)
                );
            }
        }
    }
    if let Some(f) = FILE.lock().unwrap().as_mut() {
        if let Ok(s) = serde_json::to_string(&rec) {
            let _ = writeln!(f, "{s}");
        }
    }
    crate::telemetry::ds_forward(&rec);
    if let Some(v) = SINK.lock().unwrap().as_mut() {
        v.push(rec);
    }
}

/// A speculative step's timings.
pub fn spec(rec: SpecRec) {
    let lvl = level();
    if lvl >= 1 && rec.rank == 0 && !QUIET.load(Relaxed) {
        eprintln!(
            "spec drafted={} accepted={} draft={:.2}ms verify={:.2}ms rewind={:.2}ms",
            rec.drafted, rec.accepted, rec.draft_ms, rec.verify_ms, rec.rewind_ms
        );
    }
    if let Some(f) = FILE.lock().unwrap().as_mut() {
        if let Ok(s) = serde_json::to_string(&serde_json::json!({ "spec": &rec })) {
            let _ = writeln!(f, "{s}");
        }
    }
    crate::telemetry::ds_spec(&rec);
}

/// A step's spans summed by name across layers: (name, gpu ms, count), in first-seen
/// order.
pub fn breakdown(rec: &StepRec) -> Vec<(String, f64, usize)> {
    let mut out: Vec<(String, f64, usize)> = Vec::new();
    for s in &rec.spans {
        match out.iter_mut().find(|(n, _, _)| *n == s.name) {
            Some(e) => {
                e.1 += s.gpu_ms;
                e.2 += 1;
            }
            None => out.push((s.name.clone(), s.gpu_ms, 1)),
        }
    }
    out
}

/// The step's time covered by its top-level spans (`embed`, `layer`s, `head`): what the
/// interior accounts for.
pub fn covered_ms(rec: &StepRec) -> f64 {
    rec.spans
        .iter()
        .filter(|s| !s.name.contains('.'))
        .map(|s| s.gpu_ms)
        .sum()
}

/// The ops directly inside each layer (`layer.attn`, `layer.moe`, `layer.hc`,
/// `layer.engram`), summed.
pub fn layer_ops_ms(rec: &StepRec) -> f64 {
    rec.spans
        .iter()
        .filter(|s| s.name.starts_with("layer.") && s.name.matches('.').count() == 1)
        .map(|s| s.gpu_ms)
        .sum()
}
