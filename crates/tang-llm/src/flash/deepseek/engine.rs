//! The forward on the device (M2): [`super::forward`]'s control flow over [`super::gpu`]'s
//! kernels, so every step has a CPU twin and the fixture checks the whole.
//!
//! On the device: every GEMM, norm, RoPE, quant-dequant, attention, mHC, the router, expert
//! MLPs and Engram's gate, and selection (kernels_select.cu, kernels_graph.cu): the
//! indexer's masks, candidate blocks and top-k, the per-expert routing lists (CSR; experts
//! launch for every id and exit early when empty), and the attention index rows. On the
//! host: Engram's hashing and row gather (the tables stay mapped on the host: 101 GB each
//! for the real model), uploaded through pinned staging; DSpark's left-to-right Markov
//! sampling only for a non-[`SpecSampler`] pick (`forward_spec_with`); and, only when `DS_EXPERT_SLOTS` bounds the expert LRU, the routed-expert
//! counts. Weights upload lazily on first use, in their checkpoint format.
//!
//! Scratch and syncs ([`super::graph`]): a step's temporaries come from a per-model arena
//! released at the next step's start, so nothing is allocated or freed on the token path and
//! one stream orders all reuse. The host syncs only where it reads: the logits and DSpark
//! hiddens at the end of a step, `DS_TAP`, a forward hook, and the LRU's expert counts.
//!
//! CUDA graphs (`DS_GRAPH`, on by default): a decode step (`sp > 0`: t = 1, or a verify
//! window) is shape-stable. Everything that depends on the position reads it from a device
//! parameter (`P[0]` = the start position, copied in by the graph's first node) through
//! kernels_graph.cu; widths that grow with the position (the indexer's `(pos + 1) / r`,
//! candidate masks, top-k) are launched for their maximum (`max_seq_len / r`, `index_topk`)
//! and bounded on the device, and short top-k rows are padded with -1 holes, which the
//! attention skips exactly. So the step is captured once per t (after a warm-up run that
//! uploads the weights and sizes the arena) and replayed per token. The DSpark draft
//! (sp > 0, drawn on the device) is captured too, per (main rows, sampler). Not captured:
//! prefill, `forward_spec_with`, and steps with the expert LRU (`DS_EXPERT_SLOTS`), a host-side
//! expert-parallel `moe_reduce` (one that syncs: `reduce_graphable` unset), a hook,
//! `force` or `DS_TAP`; they run the same code directly. An NCCL `moe_reduce` only enqueues
//! (`reduce_graphable`), so it is captured as a graph node with the step.

use super::forward::{Model, SpecPick, SpecSampler, SpecStep};
use super::gpu::{blocks, AttnRope, DevW, DsW, Kernels, WinIdx};
use super::graph::{Arena, Buf, GKernels, Staging, Upload, SEL};
use super::quant::{e4m3_to_f32, e8m0_to_f32};
use anyhow::{bail, Context, Result};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::rc::Rc;
use tang_moe::args;
use tang_moe::gpu::{DevBuf, Event, Gpu, Graph, Stream};

#[path = "mega_engine.rs"]
mod mega_engine;

const B256: (u32, u32, u32) = (256, 1, 1);

/// Stands in for the start position in a decode step's host code, which must not depend on
/// it (the kernels read it from `P[0]`): any host use gives wrong answers the tests catch.
const DYN: usize = 1 << 40;

/// `DS_COMP_SEQ=1`: the per-position compressor loop instead of the batched `compress_rows`.
fn comp_batched() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("DS_COMP_SEQ").map_or(true, |v| v != "1"))
}

fn b(d: &DevBuf) -> Buf {
    Buf::from(d)
}

#[derive(Default)]
struct ExpertCache {
    map: HashMap<String, (Rc<DevW>, u64)>,
    tick: u64,
    misses: usize,
}

/// A layer's routed experts for the grouped MoE kernels: device arrays of [`DsW`] (w1, w3,
/// w2) and the ownership mask, indexed by expert; non-local experts are zeroed and masked off.
struct MoeTab {
    w1: DevBuf,
    w3: DevBuf,
    w2: DevBuf,
    mask: DevBuf,
    fmt: i32,
    inter: usize,
    local: usize,
    _keep: Vec<Rc<DevW>>,
}

/// Which attention a block runs.
enum Attn {
    Main,
    Spark(Buf, usize),
}

/// A draft's staged inputs: the target hiddens [t][targets][D], the block's ids, the
/// one-hot pre-mix, the sampler's RNG state.
struct SparkIn {
    mh: Buf,
    idb: Buf,
    oh: Buf,
    rb: Buf,
    t: usize,
}

/// The draft's read-back, in f32 words: [logits blk*V | hiddens blk*D | drafts blk+1 | RNG
/// state 2 (8-byte aligned)].
struct SparkPack {
    hid: usize,
    ids: usize,
    rng: usize,
    len: usize,
}

impl SparkPack {
    fn new(blk: usize, v: usize, d: usize) -> Self {
        let (hid, ids) = (blk * v, blk * v + blk * d);
        let rng = (ids + blk + 2) & !1;
        Self {
            hid,
            ids,
            rng,
            len: rng + 2,
        }
    }
}

/// Entries a row of a device-kept distribution (`DS_SS_CAP` in kernels_select.cu).
const SS_CAP: usize = 2048;

/// The device speculative sampler's buffers (fixed addresses: a captured draft writes the
/// Q rows): the draft's kept distributions Q [blk][SS_CAP] (ids i32, probs f64, count), the
/// verify window's P [blk + 1][...], P's f64 scratch [blk + 1][V], the window ids and the
/// walk's result [ok, k, tokens[blk + 1], rng lo, rng hi].
struct SpecDev {
    qi: DevBuf,
    qp: DevBuf,
    qn: DevBuf,
    pi: DevBuf,
    pp: DevBuf,
    pn: DevBuf,
    w: DevBuf,
    ids: DevBuf,
    out: DevBuf,
}

/// `DS_SPEC_HOST=1`: sampled verify walks on the host (the reference path).
fn host_walk() -> bool {
    std::env::var("DS_SPEC_HOST").is_ok_and(|v| v == "1")
}

impl SpecDev {
    fn new(blk: usize, v: usize) -> Result<Self> {
        let n = blk + 1;
        Ok(Self {
            qi: DevBuf::alloc(blk * SS_CAP * 4)?,
            qp: DevBuf::alloc(blk * SS_CAP * 8)?,
            qn: DevBuf::zeroed(blk * 4)?,
            pi: DevBuf::alloc(n * SS_CAP * 4)?,
            pp: DevBuf::alloc(n * SS_CAP * 8)?,
            pn: DevBuf::zeroed(n * 4)?,
            w: DevBuf::alloc(n * v * 8)?,
            ids: DevBuf::alloc(n * 4)?,
            out: DevBuf::alloc((n + 4) * 4)?,
        })
    }
}

/// A captured draft (see [`GpuModel::spark`]): its pack in the arena.
struct SparkGraph {
    g: Graph,
    pk: Buf,
    gen: (u64, u64),
}

/// How [`GpuModel::spark`] picks the drafts.
enum Draw<'a> {
    /// Argmax, on the device.
    Greedy,
    /// The sampler's draws, on the device; its RNG state advances as on the host.
    Sample(&'a mut SpecSampler),
    /// `draw(logits)` on the host.
    Host(&'a mut dyn FnMut(&[f32]) -> usize),
}

/// mHC coefficients on the device: pre [T][hc], post [T][hc], comb [T][hc][hc].
struct Mix {
    pre: Buf,
    post: Buf,
    comb: Buf,
}

/// `DS_OBS_HC_SPLIT=1`: the K-split block's three mHC spans under their own names (`hc.pend`
/// the carried attention stage, `hc.ffn`, `hc.next` the closing post-mix + next stage), else
/// all `hc` (the per-rank table reads that name).
fn hc_name(split: &'static str) -> &'static str {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("DS_OBS_HC_SPLIT").is_ok_and(|v| v == "1")) {
        split
    } else {
        "hc"
    }
}

/// The attention's key index rows: built ([`Buf`] [t][k]) or left to `ds_attn_fdm`.
#[derive(Clone, Copy)]
enum Ix {
    Rows(Buf),
    Win(WinIdx),
}

/// Where RoPE reads a row's position: from the host, or from `P[0] + add` on the device.
#[derive(Clone, Copy)]
enum At {
    Host {
        pos0: usize,
        stride: usize,
        fixed: Option<usize>,
    },
    /// Row r at `P[0] + add + r * stride`, or (`fixed_r` > 0) a compressed group's
    /// `P[0] + add + 1 - fixed_r`.
    Dev {
        add: usize,
        stride: usize,
        fixed_r: usize,
    },
}

/// A step's host inputs, staged.
struct Inputs {
    ids: Buf,
    one_hot: Buf,
    engram: Vec<Buf>,
    copies: Vec<Upload>,
}

/// What a step leaves for the host: last (or all) rows' logits, the DSpark target hiddens,
/// and the ratio-r compressors' partial-group rows (layer, row in the step, kv, score).
#[derive(Clone)]
struct StepOut {
    logits: Buf,
    rows: usize,
    /// i32 [rows]: each logits row's argmax (as [`super::forward::argmax`]).
    amax: Buf,
    mains: Vec<Buf>,
    jrows: Vec<(usize, usize, Buf, Buf)>,
}

struct Captured {
    g: Graph,
    out: StepOut,
    gen: (u64, u64),
}

/// Graph counters (tests and the benchmark).
#[derive(Default, Clone, Copy, Debug)]
pub struct GraphStats {
    pub direct: usize,
    pub captures: usize,
    pub replays: usize,
}

pub struct GpuModel {
    /// The CPU reference: config, the mapped checkpoint, Engram hashing, host-side selection.
    pub cpu: Model,
    w: RefCell<HashMap<String, Rc<DevW>>>,
    experts: RefCell<ExpertCache>,
    /// Routed-expert tensors kept on the device (`DS_EXPERT_SLOTS`; 0 = all).
    pub expert_slots: usize,
    /// The fast GEMV path (`ds_lin_*`: fused activation quant and SwiGLU), bit-identical to
    /// the naive `ds_linear` path kept for A/B (`DS_SLOW=1` turns it off).
    pub fast: bool,
    /// Capture decode steps as CUDA graphs (`DS_GRAPH=0` turns it off; see the module doc).
    pub graph: bool,
    pub graph_stats: GraphStats,
    tabs: Vec<DevBuf>,
    win: Vec<DevBuf>,
    ckv: HashMap<usize, DevBuf>,
    idxk: HashMap<usize, DevBuf>,
    cst: HashMap<usize, (DevBuf, DevBuf)>,
    qk: RefCell<HashMap<usize, Rc<DevBuf>>>,
    /// Set by [`GpuModel::block`]: `moe_finish` leaves the shared expert's output here,
    /// unsummed, for the post-mix to add (`ds_add_round` folded into `ds_hc_post`).
    defer_sh: Cell<bool>,
    sh: Cell<Option<Buf>>,
    /// 8 rows of f32 zeros: the `sh` of a folded shared expert (the post-mix's `bf(y + 0)`).
    zero_rows: DevBuf,
    /// The shared expert folded into the grouped K-split launches at t <= 8 (`DS_MOE_SH=0`
    /// keeps its own MLP; never under `DS_EXACT=1`).
    pub moe_sh: bool,
    /// The router's gate GEMV in `ds_linear_t8`'s bit-exact order (`DS_ROUTER_EXACT=1`; also
    /// under `DS_EXACT=1`); otherwise [`Kernels::gemv_bf16`].
    pub router_exact: bool,
    /// Set by the forward loop: the next block's prefix when it reads this block's stream
    /// as is, so the last post-mix also computes its attention mixes (`pend_mixes`).
    next_hc: Option<String>,
    pend_mixes: Option<(Buf, i32)>,
    /// As `pend_mixes` on the K-split path: the next block's whole attention stage.
    pend_stage: Option<(Mix, Buf)>,
    sh_ckv: usize,
    sh_idxk: usize,
    /// The last index source's selection: [t][k] positions (offset into the attention's
    /// keys), and k.
    topk: Option<(Buf, usize)>,
    /// The candidate mask: u8 [t][stride], and the stride.
    cand: Option<(Buf, usize)>,
    /// This step's device parameters: `P[0]` = its start position.
    p: Buf,
    /// This step's compressor rows (see [`StepOut`]).
    jrows: Vec<(usize, usize, Buf, Buf)>,
    arena: RefCell<Arena>,
    stage: RefCell<Staging>,
    graphs: HashMap<(usize, bool), Captured>,
    warm: HashMap<(usize, bool), (u64, u64)>,
    /// Captured drafts by (main rows, sampler (temperature, top-p) bits), and their warm-ups.
    spark_graphs: HashMap<(usize, Option<(u32, u32)>), SparkGraph>,
    spark_warm: HashMap<(usize, Option<(u32, u32)>), (u64, u64)>,
    /// Debugging: each backbone layer's input stream and pre-mix (as [`Model::force`]).
    pub force: Option<Vec<(Vec<f32>, Vec<f32>)>>,
    /// Intermediates read back by name when `DS_TAP` is set (as [`Model::taps`]).
    pub taps: RefCell<Vec<(String, Vec<f32>)>>,
    /// While a [`GpuSnapshot`] is open: whether it is, the steps run under it, and the
    /// compressors' partial-group writes (as [`Model::snapshot`]): lid, position, kv row,
    /// score row (arena views of the one step run under it).
    journal_on: bool,
    journal_steps: usize,
    journal: Vec<(usize, usize, Buf, Buf)>,
    snap: Option<(Vec<DevBuf>, HashMap<usize, (DevBuf, DevBuf)>)>,
    /// DSpark's Markov embedding table [V][rank] f32, for the device-side greedy draft.
    spark_tab: Option<DevBuf>,
    /// The device rejection sampler's buffers (see [`SpecDev`]), once a sampled draft ran.
    spec_dev: Option<SpecDev>,
    /// Whether the last draft left its kept distributions in `spec_dev` (a device sampled
    /// draft), so a verify may walk on the device.
    spec_q: bool,
    /// Sampled verify walks run on the device, and on the host (a fallback).
    pub spec_walks: (usize, usize),
    /// Expert parallelism ([`super::ep`]): the routed experts this rank runs. Others are
    /// skipped in `moe` and never uploaded. All of them when unsharded.
    pub owns_expert: Box<dyn Fn(usize) -> bool>,
    /// EP ranks (1 off EP); with `sh_split` each rank folds in its 1/world slice of the shared
    /// expert (gate/up rows, down's K) and the all-reduce sums the slices (relaxed order).
    pub ep_world: usize,
    /// `DS_MOE_SH_SPLIT=0`: the shared expert on rank 0 alone (exact).
    pub sh_split: bool,
    /// Expert parallelism: combines the routed partial sum `y` ([t][d] f32, the buffer and its
    /// element count) across ranks in place, before the shared expert is added. A host
    /// callback enqueueing on the stream: a step that has one is not captured (to capture
    /// it, the callback's NCCL all-reduce would have to be captured with the step).
    pub moe_reduce: Option<MoeReduce>,
    /// Grouped MoE (a fixed handful of launches per layer, whatever the expert count) for
    /// steps of <= 8 tokens with every expert resident; `DS_MOE_LOOP=1` keeps the per-expert
    /// loop. Bit-identical either way.
    pub moe_grouped: bool,
    /// The mHC mixes GEMMs bit-exact with the CPU (`ds_hc_post_mix`), as under `DS_EXACT=1`;
    /// otherwise K-split (`ds_hc_fused_ks` / `ds_hc_mix_ks`, the linears' relaxed bar).
    /// `DS_HC_EXACT=1` sets it.
    pub hc_exact: bool,
    moe_tabs: RefCell<HashMap<usize, Option<Rc<MoeTab>>>>,
    /// `moe_reduce` only enqueues on the stream (no host sync, no allocation), so a step
    /// that has one can be captured with it (NCCL's all-reduce becomes a graph node). Every
    /// rank then captures and replays its collectives in the same order.
    pub reduce_graphable: bool,
    /// Skip the logits/hiddens read-back (and its sync): `forward` returns empty vectors.
    /// For expert-parallel ranks whose output nobody reads.
    pub skip_readback: bool,
    /// The last step's output while its logits stay on the device ([`Self::verify_window`]).
    last: Option<StepOut>,
    /// The step timeline ([`super::obs`]; `DS_OBS`).
    pub obs: super::obs::Obs,
    /// The expert-parallel reduce's stream: a graphable `moe_reduce` forks onto it (`fork`,
    /// recorded on `s`) so the shared expert overlaps the all-reduce, and `s` joins it
    /// (`join`) before the sum. Captured graphs take the fork/join as edges.
    side: Stream,
    fork: Event,
    join: Event,
    /// `ds_hc_fused_ks`'s grid barrier: arrivals (zero between launches) and generation.
    hc_cnt: DevBuf,
    /// `DS_MEGA=1`: the dense-glue megakernel ([`super::mega`]) and the deferred mHC stage
    /// it takes over (see mega_engine.rs).
    mega: Option<super::mega::Mega>,
    pend_glue: Option<mega_engine::GluePend>,
    // last: dropped after every buffer and graph above (see `Drop`)
    k: Kernels,
    gk: GKernels,
    s: Stream,
    _g: Gpu,
}

impl Drop for GpuModel {
    /// Drain the stream and destroy the graphs before any memory they use is freed.
    fn drop(&mut self) {
        let _ = self.s.sync();
        let _ = self.side.sync();
        self.graphs.clear();
        self.spark_graphs.clear();
    }
}

/// As [`super::forward::Snapshot`], on the device: views of the model's one snapshot store
/// (a newer snapshot overwrites it).
pub struct GpuSnapshot {
    win: Vec<Buf>,
    cst: HashMap<usize, (Buf, Buf)>,
}

/// See [`GpuModel::moe_reduce`].
pub type MoeReduce = Box<dyn Fn(&DevBuf, usize, &Stream) -> Result<()>>;

impl GpuModel {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_on(dir, 0)
    }

    /// [`Self::open`] on device `ordinal`, binding its context to the calling thread (the
    /// model is not `Send`: it lives on that thread).
    pub fn open_on(dir: &Path, ordinal: usize) -> Result<Self> {
        let g = Gpu::at(ordinal)?;
        g.bind()?;
        let k = Kernels::new(&g)?;
        let gk = GKernels::new(&g)?;
        // blocking: weight uploads (pageable `cuMemcpyHtoD`) finish their DMA on the legacy
        // stream, which only a blocking stream orders behind
        let s = Stream::blocking()?;
        let cpu = Model::open(dir)?;
        let c = cpu.cfg.clone();
        let (hd, sl) = (c.head_dim, c.max_seq_len);
        let upf = |v: &[f32]| -> Result<DevBuf> { Ok(DevBuf::from_slice(v)?) };
        let tabs = cpu
            .freqs
            .iter()
            .map(|t| upf(&t.iter().flat_map(|&(a, b)| [a, b]).collect::<Vec<f32>>()))
            .collect::<Result<_>>()?;
        let n_all = c.n_layers + c.n_mtp_layers;
        let win = (0..n_all)
            .map(|_| upf(&vec![0f32; c.window_size * hd]))
            .collect::<Result<_>>()?;
        let mut ckv = HashMap::new();
        let mut idxk = HashMap::new();
        let mut cst = HashMap::new();
        for &l in &c.kv_source_layers {
            let r = c.compress_ratios[l];
            ckv.insert(l, upf(&vec![0f32; sl / r * hd])?);
            idxk.insert(l, upf(&vec![0f32; sl / r * c.index_head_dim])?);
            if r > 1 {
                cst.insert(
                    l,
                    (
                        upf(&vec![0f32; r * hd])?,
                        upf(&vec![f32::NEG_INFINITY; r * hd])?,
                    ),
                );
            }
        }
        Ok(Self {
            cpu,
            k,
            gk,
            s,
            side: Stream::new()?,
            fork: Event::new(false)?,
            join: Event::new(false)?,
            hc_cnt: DevBuf::from_slice(&[0i32; 2])?,
            mega: match super::mega::on() {
                true => Some(super::mega::Mega::new(&g)?),
                false => None,
            },
            pend_glue: None,
            _g: g,
            w: RefCell::new(HashMap::new()),
            experts: RefCell::new(ExpertCache::default()),
            // an expert MLP holds three tensors at once
            expert_slots: match std::env::var("DS_EXPERT_SLOTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                Some(0) | None => 0,
                Some(n) => n.max(3),
            },
            fast: std::env::var_os("DS_SLOW").is_none(),
            graph: std::env::var("DS_GRAPH").map_or(true, |v| v != "0"),
            graph_stats: GraphStats::default(),
            tabs,
            win,
            ckv,
            idxk,
            cst,
            qk: RefCell::new(HashMap::new()),
            defer_sh: Cell::new(false),
            sh: Cell::new(None),
            zero_rows: DevBuf::from_slice(&vec![0f32; 8 * c.dim])?,
            moe_sh: std::env::var("DS_MOE_SH").map_or(true, |v| v != "0"),
            ep_world: 1,
            sh_split: std::env::var("DS_MOE_SH_SPLIT").map_or(true, |v| v != "0"),
            router_exact: std::env::var("DS_ROUTER_EXACT").is_ok_and(|v| v == "1"),
            next_hc: None,
            pend_mixes: None,
            pend_stage: None,
            sh_ckv: 0,
            sh_idxk: 0,
            topk: None,
            cand: None,
            p: Buf { ptr: 0, len: 0 },
            jrows: Vec::new(),
            arena: RefCell::new(Arena::default()),
            stage: RefCell::new(Staging::default()),
            graphs: HashMap::new(),
            warm: HashMap::new(),
            spark_graphs: HashMap::new(),
            spark_warm: HashMap::new(),
            force: None,
            taps: RefCell::new(Vec::new()),
            journal_on: false,
            journal_steps: 0,
            journal: Vec::new(),
            snap: None,
            spark_tab: None,
            spec_dev: None,
            spec_q: false,
            spec_walks: (0, 0),
            owns_expert: Box::new(|_| true),
            moe_reduce: None,
            moe_grouped: std::env::var("DS_MOE_LOOP").map_or(true, |v| v != "1"),
            moe_tabs: RefCell::new(HashMap::new()),
            hc_exact: std::env::var("DS_HC_EXACT").is_ok_and(|v| v == "1"),
            reduce_graphable: false,
            skip_readback: false,
            last: None,
            obs: Default::default(),
        })
    }

    /// Forget the sequence: the next forward must start at position 0. The caches need no
    /// clearing (nothing past a sequence's own positions is ever read); the ratio-2
    /// compressors' partial groups do (in place: captured graphs hold their addresses).
    pub fn reset(&mut self) -> Result<()> {
        let hd = self.cpu.cfg.head_dim;
        for (&l, (a, bb)) in self.cst.iter() {
            let n = self.cpu.cfg.compress_ratios[l] * hd;
            self.fill(a.ptr, 0, n)?;
            self.fill(bb.ptr, f32::NEG_INFINITY.to_bits() as i32, n)?;
        }
        self.sh_ckv = 0;
        self.sh_idxk = 0;
        self.topk = None;
        self.cand = None;
        Ok(())
    }

    // scratch

    /// `n` uninitialized f32 (or i32) of this step's scratch.
    fn fbuf(&self, n: usize) -> Result<Buf> {
        self.arena.borrow_mut().alloc(n.max(1) * 4)
    }

    fn bytes(&self, n: usize) -> Result<Buf> {
        self.arena.borrow_mut().alloc(n.max(1))
    }

    /// `n` zeroed f32 of scratch (a kernel: `DevBuf::zeroed`'s memset is on the NULL stream).
    fn zbuf(&self, n: usize) -> Result<Buf> {
        let o = self.fbuf(n)?;
        self.fill(o.ptr, 0, n)?;
        Ok(o)
    }

    fn fill(&self, p: u64, v: i32, n: usize) -> Result<()> {
        let nn = n as i64;
        unsafe {
            self.k.go(
                "ds_fill_i32",
                blocks(n, 256),
                B256,
                0,
                &self.s,
                args![p, v, nn],
            )
        }
    }

    /// Stage `v` for upload: scratch on the device, filled by the returned copy.
    fn stage<T: Copy>(&self, v: &[T]) -> Result<(Buf, Upload)> {
        let n = std::mem::size_of_val(v);
        let d = self.arena.borrow_mut().alloc(n)?;
        let h = self.stage.borrow_mut().alloc(n)?;
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, h, n) };
        Ok((
            d,
            Upload {
                dst: d.ptr,
                src: h,
                n,
            },
        ))
    }

    /// `v` on the device now (a direct step's upload).
    fn up<T: Copy>(&self, v: &[T]) -> Result<Buf> {
        let (d, u) = self.stage(v)?;
        u.enqueue(&self.s)?;
        Ok(d)
    }

    /// Start a step: drain the stream (its staging may still be read), then release the
    /// last step's scratch and staging.
    fn begin_step(&mut self) -> Result<()> {
        self.sync()?;
        self.arena.borrow_mut().reset()?;
        self.stage.borrow_mut().reset()?;
        let g = self.gen();
        self.graphs.retain(|_, c| c.gen == g);
        self.spark_graphs.retain(|_, c| c.gen == g);
        Ok(())
    }

    /// Drain the stream (counted: [`super::obs`]).
    fn sync(&self) -> Result<()> {
        super::obs::sync();
        Ok(self.s.sync()?)
    }

    /// Open a timeline span ([`super::obs::Obs::span`]).
    #[inline(always)]
    fn span(&self, name: &'static str) -> u32 {
        self.obs.span(&self.s, name, None)
    }

    #[inline(always)]
    fn end(&self, id: u32) {
        self.obs.end(&self.s, id)
    }

    fn gen(&self) -> (u64, u64) {
        (self.arena.borrow().gen, self.stage.borrow().gen)
    }

    // weights

    fn dw(&self, name: &str) -> Result<Rc<DevW>> {
        if name.contains(".ffn.experts.") {
            return self.expert_w(name);
        }
        if let Some(w) = self.w.borrow().get(name) {
            return Ok(w.clone());
        }
        let w = Rc::new(DevW::upload(&self.cpu.st, name)?);
        self.w.borrow_mut().insert(name.to_string(), w.clone());
        Ok(w)
    }

    /// A routed expert's weight through the LRU of `expert_slots` tensors (unbounded when 0),
    /// uploaded from the mapped checkpoint on a miss. Evicting frees device memory a queued
    /// kernel may still read, so it drains the stream first (the LRU's only sync besides
    /// its counts read-back).
    fn expert_w(&self, name: &str) -> Result<Rc<DevW>> {
        let mut ex = self.experts.borrow_mut();
        ex.tick += 1;
        let tick = ex.tick;
        if let Some((w, t)) = ex.map.get_mut(name) {
            *t = tick;
            return Ok(w.clone());
        }
        if self.expert_slots > 0 && ex.map.len() >= self.expert_slots {
            let oldest = ex
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone())
                .unwrap();
            self.sync()?;
            ex.map.remove(&oldest);
        }
        ex.misses += 1;
        super::obs::expert_upload();
        let w = Rc::new(DevW::upload(&self.cpu.st, name)?);
        ex.map.insert(name.to_string(), (w.clone(), tick));
        Ok(w)
    }

    /// Routed-expert tensors uploaded so far (cache misses).
    pub fn expert_misses(&self) -> usize {
        self.experts.borrow().misses
    }

    fn tap(&self, name: &str, b: Buf, n: usize) -> Result<()> {
        if std::env::var_os("DS_TAP").is_some() {
            let v = self.read(b, n)?;
            self.taps.borrow_mut().push((name.to_string(), v));
        }
        Ok(())
    }

    fn read(&self, b: Buf, n: usize) -> Result<Vec<f32>> {
        self.sync()?;
        super::obs::d2h(n * 4);
        Ok(b.dev().read::<f32>(0, n)?)
    }

    fn copy(&self, dst: u64, src: u64, n: usize) -> Result<()> {
        let nn = n as i64;
        unsafe {
            self.k.go(
                "ds_copy",
                blocks(n, 256),
                B256,
                0,
                &self.s,
                args![dst, src, nn],
            )
        }
    }

    /// `linear`: quantized weights get an fp8-quantized copy of the activation; bf16 out.
    fn lin(&self, x: Buf, t: usize, name: &str) -> Result<Buf> {
        let w = self.dw(name)?;
        let (k, n) = (w.v.cols as usize, w.v.rows as usize);
        let y = self.fbuf(t * n)?;
        if self.fast && self.k.ks_ok(&w.v) {
            // the K-split GEMV quantizes the activation itself (bit-identical to ds_fp8_qdq)
            self.k
                .linear_ks(&self.s, &w.v, None, x.ptr, y.ptr, t, true, false, 0, 0.0)?;
        } else if self.fast && Kernels::fast_ok(&w.v) {
            let xq = self.qdq(x, t * k)?;
            self.k
                .linear_fast(&self.s, &w.v, None, xq.ptr, y.ptr, t, true, false, 0, 0.0)?;
        } else if w.v.fmt >= 2 {
            let xq = self.qdq(x, t * k)?;
            self.k.linear(&self.s, &w.v, xq.ptr, y.ptr, t, true)?;
        } else {
            self.k.linear(&self.s, &w.v, x.ptr, y.ptr, t, true)?;
        }
        Ok(y)
    }

    /// An f32 GEMM with no activation quantization or rounding (router, mHC, compressor > 1,
    /// the head).
    fn raw(&self, x: u64, t: usize, name: &str) -> Result<Buf> {
        let w = self.dw(name)?;
        let y = self.fbuf(t * w.v.rows as usize)?;
        self.k.linear(&self.s, &w.v, x, y.ptr, t, false)?;
        Ok(y)
    }

    /// [`GpuModel::raw`] for the decode step's small bf16 linears on [`Kernels::gemv_bf16`]
    /// (relaxed: summation order); `raw` itself under `DS_EXACT` or off the fast path.
    fn raw_bf(&self, x: u64, t: usize, name: &str) -> Result<Buf> {
        if !self.fast {
            return self.raw(x, t, name);
        }
        let w = self.dw(name)?;
        let y = self.fbuf(t * w.v.rows as usize)?;
        self.k.gemv_bf16(&self.s, &w.v, x, y.ptr, t)?;
        Ok(y)
    }

    fn rms(&self, x: Buf, rows: usize, name: &str) -> Result<()> {
        let w = self.dw(name)?;
        self.k
            .rms(&self.s, x.ptr, &w.v, rows, self.cpu.cfg.norm_eps)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        x: Buf,
        rows: usize,
        per_row: usize,
        d: usize,
        lid: usize,
        at: At,
        inverse: bool,
    ) -> Result<()> {
        let rd = self.cpu.cfg.rope_head_dim;
        let tab = self.tabs[lid].ptr;
        match at {
            At::Host {
                pos0,
                stride,
                fixed,
            } => self.k.rope(
                &self.s, x.ptr, tab, rows, per_row, d, rd, pos0, stride, fixed, inverse,
            ),
            At::Dev {
                add,
                stride,
                fixed_r,
            } => {
                let n = rows * per_row * rd / 2;
                let (r, pr, di, rdi, p, a, ps, fx, inv) = (
                    rows as i32,
                    per_row as i32,
                    d as i32,
                    rd as i32,
                    self.p.ptr,
                    add as i32,
                    stride as i32,
                    fixed_r as i32,
                    inverse as i32,
                );
                unsafe {
                    self.gk.go(
                        "dg_rope",
                        blocks(n, 256),
                        B256,
                        0,
                        &self.s,
                        args![x.ptr, tab, r, pr, di, rdi, p, a, ps, fx, inv],
                    )
                }
            }
        }
    }

    // attention

    fn attention(&mut self, p: &str, lid: usize, x: Buf, t: usize, sp: usize) -> Result<Buf> {
        if sp == 0 {
            self.attention_prefill(p, lid, x, t)
        } else {
            self.attention_dyn(p, lid, x, t)
        }
    }

    fn attention_prefill(&mut self, p: &str, lid: usize, x: Buf, t: usize) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (h, hd) = (c.n_heads, c.head_dim);
        let at0 = At::Host {
            pos0: 0,
            stride: 1,
            fixed: None,
        };
        let s = self.span("q");
        let qr = self.lin(x, t, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(qr, t, &format!("{p}.attn.q_norm.weight"))?;
        self.tap(&format!("{p}.qr"), qr, t * c.q_lora_rank)?;
        let q = self.lin(qr, t, &format!("{p}.attn.wq_b.weight"))?;
        self.tap(&format!("{p}.q0"), q, t * h * hd)?;
        self.rope(q, t, h, hd, lid, at0, false)?;
        self.tap(&format!("{p}.q"), q, t * h * hd)?;
        self.end(s);
        let s = self.span("window_kv");
        let kv = self.lin(x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(kv, t, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(kv, t, 1, hd, lid, at0, false)?;
        self.k.fp8_qdq(&self.s, kv.ptr, t * hd)?;
        self.ring_seed(lid, kv, t)?;
        self.tap(&format!("{p}.wkv"), kv, t * hd)?;
        self.end(s);
        let nw = t.min(c.window_size);
        let (mut kc, mut kv1) = (0, 0);
        if c.compress_ratios[lid] > 0 {
            let s = self.span("compress");
            let (src, k) = self.compress_kv_prefill(p, lid, x, qr, t)?;
            kc = k;
            kv1 = self.ckv[&src].ptr;
            self.end(s);
        }
        let s = self.span("sparse_attn");
        let idx = self.win_idx(t, nw + kc, nw, 0, 0, 0)?;
        if kc > 0 {
            let tk = self.topk.as_ref().context("no top-k")?.0;
            self.idx_cat(idx, tk, t, kc, nw + kc, nw)?;
        }
        self.end(s);
        self.attn_out(
            p,
            q,
            true,
            (kv, t, kv1),
            Ix::Rows(idx),
            t,
            nw + kc,
            lid,
            at0,
        )
    }

    /// The decode step's q and window kv on the K-split path, in 6 launches: x quantized once,
    /// the wq_a and wkv GEMVs, the pre-chunk ring copy (t > 1), `ds_qkv_post` (both rms, kv
    /// rope + qdq + ring/window writes, qr's codes), the wq_b GEMV and q's rope. As the unfused
    /// path bit for bit up to the K-split GEMVs. `None` when it does not apply (DS_EXACT, shapes).
    #[allow(clippy::type_complexity)]
    fn qkv_fused(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
    ) -> Result<Option<(Buf, Buf, Buf, usize, usize)>> {
        let c = &self.cpu.cfg;
        let (h, hd, win, rd, eps) = (
            c.n_heads,
            c.head_dim,
            c.window_size,
            c.rope_head_dim,
            c.norm_eps,
        );
        let wqa = self.dw(&format!("{p}.attn.wq_a.weight"))?.v;
        let wkv = self.dw(&format!("{p}.attn.wkv.weight"))?.v;
        let wqb = self.dw(&format!("{p}.attn.wq_b.weight"))?.v;
        let qn = self.dw(&format!("{p}.attn.q_norm.weight"))?.v;
        let kn = self.dw(&format!("{p}.attn.kv_norm.weight"))?.v;
        let ok = self.fast
            && (1..=8).contains(&t)
            && [wqa, wkv, wqb].iter().all(|w| self.k.ks_ok(w))
            && wqa.cols == wkv.cols
            && wkv.rows as usize == hd
            && wqb.cols == wqa.rows
            && wqb.rows as usize == h * hd
            && qn.cols == wqa.rows
            && kn.cols as usize == hd
            && wqa.rows % 32 == 0
            && hd % 32 == 0;
        if !ok {
            self.glue_flush()?;
            return Ok(None);
        }
        if let Some(r) = self.glue_qkv(p, lid, x, t, [wqa, wkv, wqb, qn, kn])? {
            return Ok(Some(r));
        }
        let (d, ql) = (wqa.cols as usize, wqa.rows as usize);
        let at = At::Dev {
            add: 0,
            stride: 1,
            fixed_r: 0,
        };
        let sp = self.span("q");
        let (xc, xs) = self.k.ks_x_ptrs();
        self.k.ks_quant_to(&self.s, x.ptr, xc, xs, t * d)?;
        let (qr, kv) = (self.fbuf(t * ql)?, self.fbuf(t * hd)?);
        // wq_a and wkv in one launch (ds_ks2), each bit for bit its own ks_gemv
        self.k
            .ks_gemv2(&self.s, &wqa, &wkv, xc, xs, qr.ptr, kv.ptr, t, true)?;
        self.end(sp);
        let sp = self.span("window_kv");
        let ring = b(&self.win[lid]);
        let (kv0, split, mode, allp) = if t > 1 {
            let all = self.fbuf((win + t) * hd)?;
            self.copy(all.ptr, ring.ptr, win * hd)?;
            (all, win + t, 3, all.ptr)
        } else {
            (ring, win, 1, 0)
        };
        let (qc, qs) = self.k.wo_x_ptrs();
        let (tab, rdi, pp, wi) = (self.tabs[lid].ptr, rd as i32, self.p.ptr, win as i32);
        unsafe {
            self.k.go(
                "ds_qkv_post",
                (t as u32, 2, 1),
                (1024, 1, 1),
                0,
                &self.s,
                args![qr.ptr, qn, qc, qs, kv.ptr, kn, tab, rdi, pp, ring.ptr, wi, allp, eps],
            )?
        };
        self.end(sp);
        let sp = self.span("q");
        let q = self.fbuf(t * h * hd)?;
        self.k
            .ks_gemv(&self.s, &wqb, None, qc, qs, q.ptr, t, true, 0, 0.0)?;
        // q's rope is attn_out's (fused into the tiled attention)
        self.end(sp);
        Ok(Some((qr, q, kv0, split, mode)))
    }

    /// The decode step's attention (t = 1, or a verify window), position-free on the host.
    fn attention_dyn(&mut self, p: &str, lid: usize, x: Buf, t: usize) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (h, hd, win) = (c.n_heads, c.head_dim, c.window_size);
        let at = At::Dev {
            add: 0,
            stride: 1,
            fixed_r: 0,
        };
        let (qr, q, kv0, split, mode) = match self.qkv_fused(p, lid, x, t)? {
            Some(v) => v,
            None => {
                let s = self.span("q");
                let qr = self.lin(x, t, &format!("{p}.attn.wq_a.weight"))?;
                self.rms(qr, t, &format!("{p}.attn.q_norm.weight"))?;
                let q = self.lin(qr, t, &format!("{p}.attn.wq_b.weight"))?;
                self.end(s);
                let s = self.span("window_kv");
                let kv = self.lin(x, t, &format!("{p}.attn.wkv.weight"))?;
                self.rms(kv, t, &format!("{p}.attn.kv_norm.weight"))?;
                self.rope(kv, t, 1, hd, lid, at, false)?;
                self.k.fp8_qdq(&self.s, kv.ptr, t * hd)?;
                let ring = b(&self.win[lid]);
                let (kv0, split, mode) = if t > 1 {
                    // a verify window (as forward.rs): the pre-chunk ring then the chunk's rows
                    let all = self.fbuf((win + t) * hd)?;
                    self.copy(all.ptr, ring.ptr, win * hd)?;
                    self.copy(all.ptr + (win * hd * 4) as u64, kv.ptr, t * hd)?;
                    self.ring_write(ring, kv, t, hd, win, 0)?;
                    (all, win + t, 3)
                } else {
                    self.ring_write(ring, kv, 1, hd, win, 0)?;
                    (ring, win, 1)
                };
                self.end(s);
                (qr, q, kv0, split, mode)
            }
        };
        let (mut kc, mut kv1) = (0, 0);
        if c.compress_ratios[lid] > 0 {
            let s = self.span("compress");
            let src = self.compress_dyn(p, lid, x, qr, t, split)?;
            kc = self.topk.as_ref().context("no top-k")?.1;
            kv1 = self.ckv[&src].ptr;
            self.end(s);
        }
        let (src, ki) = match kc {
            0 => (0, 0),
            _ => (self.topk.as_ref().unwrap().0.ptr, kc as i32),
        };
        let rd = c.rope_head_dim;
        if rd % 2 == 0 && rd <= hd && self.k.attn_fd_win_ok(h, hd, win + kc) {
            // ds_attn_fdm builds the index rows itself
            let w = WinIdx {
                mode: mode as i32,
                win: win as i32,
                src,
                kc: ki,
            };
            return self.attn_out(
                p,
                q,
                false,
                (kv0, split, kv1),
                Ix::Win(w),
                t,
                win + kc,
                lid,
                at,
            );
        }
        let s = self.span("sparse_attn");
        let idx = self.fbuf(t * (win + kc))?;
        let (ti, st, ni, mi, pp, wi) = (
            t as i32,
            (win + kc) as i32,
            win as i32,
            mode as i32,
            self.p.ptr,
            win as i32,
        );
        unsafe {
            self.gk.go(
                "dg_win_idx",
                blocks(t * (win + kc), 256),
                B256,
                0,
                &self.s,
                args![idx.ptr, ti, st, ni, mi, pp, wi, src, ki],
            )?
        };
        self.end(s);
        self.attn_out(
            p,
            q,
            false,
            (kv0, split, kv1),
            Ix::Rows(idx),
            t,
            win + kc,
            lid,
            at,
        )
    }

    fn idx_cat(&self, idx: Buf, src: Buf, t: usize, k: usize, st: usize, col: usize) -> Result<()> {
        let (ti, ki, sti, ci) = (t as i32, k as i32, st as i32, col as i32);
        unsafe {
            self.k.go(
                "ds_idx_cat",
                blocks(t * k, 256),
                B256,
                0,
                &self.s,
                args![idx.ptr, src.ptr, ti, ki, sti, ci],
            )
        }
    }

    /// `t` rows of `hd` into a ring of `win` slots at positions `P[0] + add + e`.
    fn ring_write(
        &self,
        ring: Buf,
        src: Buf,
        t: usize,
        hd: usize,
        win: usize,
        add: usize,
    ) -> Result<()> {
        let (ti, hdi, wi, pp, ai) = (t as i32, hd as i32, win as i32, self.p.ptr, add as i32);
        unsafe {
            self.gk.go(
                "dg_ring_write",
                blocks(t * hd, 256),
                B256,
                0,
                &self.s,
                args![ring.ptr, src.ptr, ti, hdi, wi, pp, ai],
            )
        }
    }

    /// One row of `n` into row `(P[0] + add) / r` of `dst`.
    fn row_write(&self, dst: u64, src: Buf, n: usize, add: usize, r: usize) -> Result<()> {
        let (ni, pp, ai, ri) = (n as i32, self.p.ptr, add as i32, r as i32);
        unsafe {
            self.gk.go(
                "dg_row_write",
                blocks(n, 256),
                B256,
                0,
                &self.s,
                args![dst, src.ptr, ni, pp, ai, ri],
            )
        }
    }

    /// Attention index rows [t][stride] with the window's `n` columns filled (`ds_win_idx`).
    fn win_idx(
        &self,
        t: usize,
        stride: usize,
        n: usize,
        mode: i32,
        sp: usize,
        bb: usize,
    ) -> Result<Buf> {
        let idx = self.fbuf(t.max(1) * stride.max(1))?;
        let (ti, si, ni, spi, wi, bi) = (
            t as i32,
            stride as i32,
            n as i32,
            sp as i32,
            self.cpu.cfg.window_size as i32,
            bb as i32,
        );
        unsafe {
            self.k.go(
                "ds_win_idx",
                blocks(t * n, 256),
                B256,
                0,
                &self.s,
                args![idx.ptr, ti, si, ni, mode, spi, wi, bi],
            )?
        };
        Ok(idx)
    }

    /// Sparse attention over keys `kv` = (rows below `split` here, the rest from `kv1`), then
    /// the output projection.
    #[allow(clippy::too_many_arguments)]
    fn attn_out(
        &self,
        p: &str,
        q: Buf,
        q_roped: bool,
        kv: (Buf, usize, u64),
        di: Ix,
        t: usize,
        k: usize,
        lid: usize,
        at: At,
    ) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (h, hd, g, ol) = (c.n_heads, c.head_dim, c.o_groups, c.o_lora_rank);
        let s = self.span("sparse_attn");
        let mut wo_pre = false;
        let sink = self.dw(&format!("{p}.attn.attn_sink"))?;
        let o = self.fbuf(t * h * hd)?;
        // the tiled kernels at the real shapes (see `Kernels::sparse_attn`), else the exact one
        let scale = (hd as f32).powf(-0.5);
        // q's rope and o's inverse rope fused into the tiled attention (bit for bit dg_rope)
        let rd = c.rope_head_dim;
        let rope = match at {
            At::Dev {
                add,
                stride,
                fixed_r: 0,
            } if rd % 2 == 0 && rd <= hd && self.k.attn_tiled_ok(h, hd, k) => Some(AttnRope {
                tab: self.tabs[lid].ptr,
                p: self.p.ptr,
                rd: rd as i32,
                add: add as i32,
                stride: stride as i32,
                q: !q_roped,
                wo_quant: false,
            }),
            _ => None,
        };
        if !q_roped && rope.is_none() {
            self.rope(q, t, h, hd, lid, at, false)?;
        }
        match (di, rope) {
            (Ix::Rows(di), _) => self.k.sparse_attn_r(
                &self.s, q.ptr, kv.0.ptr, kv.1, kv.2, sink.v.w, di.ptr, o.ptr, t, h, hd, k, scale,
                rope,
            )?,
            (Ix::Win(w), Some(mut r)) => {
                // the merge also quantizes o for wo_ks (one launch fewer)
                let wo_a = self.dw(&format!("{p}.attn.wo_a.weight"))?;
                r.wo_quant = self.fast && self.k.wo_ks_ok(&wo_a.v, t, g, ol);
                wo_pre = r.wo_quant;
                self.k.attn_fd_win(
                    &self.s, q.ptr, kv.0.ptr, kv.1, kv.2, sink.v.w, o.ptr, t, h, hd, k, scale, r, w,
                )?
            }
            (Ix::Win(_), None) => bail!("attn_out: in-kernel indices want the fused rope"),
        }
        self.tap(&format!("{p}.sa"), o, t * h * hd)?;
        self.end(s);
        let s = self.span("wo");
        if rope.is_none() {
            self.rope(o, t, h, hd, lid, at, true)?;
        }
        let wo_a = self.dw(&format!("{p}.attn.wo_a.weight"))?;
        let mid = self.fbuf(t * g * ol)?;
        if self.fast && self.k.wo_ks_ok(&wo_a.v, t, g, ol) {
            // the quantize is ds_ks_quant's (bit-identical to ds_fp8_qdq)
            if wo_pre {
                self.k.wo_ks_pre(&self.s, &wo_a.v, mid.ptr, t, g, ol)?;
            } else {
                self.k.wo_ks(&self.s, &wo_a.v, o.ptr, mid.ptr, t, g, ol)?;
            }
        } else {
            self.k.fp8_qdq(&self.s, o.ptr, t * h * hd)?;
            self.wo_a(o, &wo_a.v, mid, t, g, ol)?;
        }
        let y = self.lin(mid, t, &format!("{p}.attn.wo_b.weight"));
        self.end(s);
        y
    }

    /// wo_a, block-diagonal over `g` groups: one K-split GEMV a group over the group-major
    /// activations (the ks linears' relaxed bar), or under `DS_EXACT=1` the exact `ds_wo_a`.
    fn wo_a(&self, o: Buf, w: &DsW, mid: Buf, t: usize, g: usize, ol: usize) -> Result<()> {
        let gd = w.cols as usize;
        let sub = |gi: usize| {
            let mut v = *w;
            v.rows = ol as i32;
            if w.fmt == 2 {
                v.w += (gi * ol * gd) as u64;
                v.s += ((gi * ol / 32) * gd.div_ceil(32)) as u64;
            } else {
                v.w += (gi * ol * gd / 2) as u64;
                v.s += (gi * ol * gd / 32) as u64;
            }
            v
        };
        let ks = self.fast && !self.k.exact && Kernels::fast_ok(w) && ol % 32 == 0;
        if !ks {
            anyhow::ensure!(ol % 16 == 0, "ds_wo_a wants o_lora_rank % 16 == 0");
            let (ti, gi, oli) = (t as i32, g as i32, ol as i32);
            return unsafe {
                self.k.go(
                    "ds_wo_a",
                    ((g * ol / 16) as u32, 1, 1),
                    B256,
                    0,
                    &self.s,
                    args![o.ptr, *w, mid.ptr, ti, gi, oli],
                )
            };
        }
        let (og, yg) = (self.fbuf(t * g * gd)?, self.fbuf(t * g * ol)?);
        self.perm3(o, og, t, g, gd)?;
        for gi in 0..g {
            let (xo, yo) = (
                og.ptr + (gi * t * gd * 4) as u64,
                yg.ptr + (gi * t * ol * 4) as u64,
            );
            self.k
                .linear_fast(&self.s, &sub(gi), None, xo, yo, t, true, false, 0, 0.0)?;
        }
        self.perm3(yg, mid, g, t, ol)
    }

    fn perm3(&self, src: Buf, dst: Buf, na: usize, nb: usize, nc: usize) -> Result<()> {
        let (a, b2, c) = (na as i32, nb as i32, nc as i32);
        unsafe {
            self.k.go(
                "ds_perm3",
                blocks(na * nb * nc, 256),
                B256,
                0,
                &self.s,
                args![src.ptr, dst.ptr, a, b2, c],
            )
        }
    }

    fn ring_seed(&self, lid: usize, kv: Buf, t: usize) -> Result<()> {
        let (hd, win) = (self.cpu.cfg.head_dim, self.cpu.cfg.window_size);
        let ring = self.win[lid].ptr;
        if t <= win {
            return self.copy(ring, kv.ptr, t * hd);
        }
        let cut = t % win;
        let tail = kv.ptr + ((t - win) * hd * 4) as u64;
        self.copy(ring + (cut * hd * 4) as u64, tail, (win - cut) * hd)?;
        self.copy(ring, tail + ((win - cut) * hd * 4) as u64, cut * hd)
    }

    fn pool(&self, kvp: u64, scp: u64, groups: usize, r: usize) -> Result<Buf> {
        let hd = self.cpu.cfg.head_dim;
        let out = self.fbuf(groups * hd)?;
        let (gi, ri, hdi) = (groups as i32, r as i32, hd as i32);
        unsafe {
            self.k.go(
                "ds_pool",
                blocks(groups * hd, 256),
                B256,
                0,
                &self.s,
                args![kvp, scp, out.ptr, gi, ri, hdi],
            )?
        };
        let n = (groups * hd) as i64;
        unsafe {
            self.k.go(
                "ds_round",
                blocks(groups * hd, 256),
                B256,
                0,
                &self.s,
                args![out.ptr, n],
            )?
        };
        Ok(out)
    }

    /// Prefill: the source's new compressed latents (pre-RoPE) and how many.
    fn compressor_prefill(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
    ) -> Result<Option<(Buf, usize)>> {
        let c = self.cpu.cfg.clone();
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let norm = format!("{p}.attn.compressor.norm.weight");
        if r == 1 {
            let kv = self.lin(x, t, &format!("{p}.attn.compressor.wkv.weight"))?;
            self.rms(kv, t, &norm)?;
            return Ok(Some((kv, t)));
        }
        let kv = self.raw(x.ptr, t, &format!("{p}.attn.compressor.wkv.weight"))?;
        let sc = self.raw(x.ptr, t, &format!("{p}.attn.compressor.wgate.weight"))?;
        let (kst, sst) = {
            let (a, bb) = &self.cst[&lid];
            (a.ptr, bb.ptr)
        };
        let row = (hd * 4) as u64;
        if t < r {
            self.copy(kst, kv.ptr, t * hd)?;
            self.copy(sst, sc.ptr, t * hd)?;
            return Ok(None);
        }
        let rem = t % r;
        let cut = t - rem;
        if rem > 0 {
            self.copy(kst, kv.ptr + cut as u64 * row, rem * hd)?;
            self.copy(sst, sc.ptr + cut as u64 * row, rem * hd)?;
        }
        let out = self.pool(kv.ptr, sc.ptr, cut / r, r)?;
        self.rms(out, cut / r, &norm)?;
        Ok(Some((out, cut / r)))
    }

    /// Prefill: compressed KV and selection; the cache's source layer and top-k width.
    fn compress_kv_prefill(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        qr: Buf,
        t: usize,
    ) -> Result<(usize, usize)> {
        let c = self.cpu.cfg.clone();
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let clen = t / r;
        let mut latent = None;
        if c.is_kv_source(lid) {
            latent = self.compressor_prefill(p, lid, x, t)?;
            self.sh_ckv = lid;
        }
        if c.is_index_source(lid) {
            self.topk = if clen == 0 {
                None
            } else {
                let s = self.span("indexer");
                let r = self.indexer_prefill(p, lid, x, qr, latent, t)?;
                self.end(s);
                Some(r)
            };
        }
        let kc = self.topk.as_ref().map_or(0, |(_, k)| *k);
        if let Some((lat, n)) = latent {
            let at = At::Host {
                pos0: 0,
                stride: r,
                fixed: None,
            };
            self.rope(lat, n, 1, hd, lid, at, false)?;
            self.k.fp4_qdq(&self.s, lat.ptr, n * hd, 16, true)?;
            self.copy(self.ckv[&lid].ptr, lat.ptr, n * hd)?;
        }
        Ok((self.sh_ckv, kc))
    }

    fn indexer_prefill(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        qr: Buf,
        latent: Option<(Buf, usize)>,
        t: usize,
    ) -> Result<(Buf, usize)> {
        let c = self.cpu.cfg.clone();
        let (r, nh, ihd) = (c.compress_ratios[lid], c.index_n_heads, c.index_head_dim);
        let owns = c.is_kv_source(lid);
        if let (true, Some((lat, n))) = (owns, latent) {
            let k = self.lin(lat, n, &format!("{p}.attn.indexer.wk.weight"))?;
            self.rms(k, n, &format!("{p}.attn.indexer.k_norm.weight"))?;
            let at = At::Host {
                pos0: 0,
                stride: r,
                fixed: None,
            };
            self.rope(k, n, 1, ihd, lid, at, false)?;
            self.k.fp4_qdq(&self.s, k.ptr, n * ihd, 32, false)?;
            self.copy(self.idxk[&lid].ptr, k.ptr, n * ihd)?;
        }
        if owns {
            self.sh_idxk = lid;
        }
        let q = self.lin(qr, t, &format!("{p}.attn.indexer.wq_b.weight"))?;
        let at0 = At::Host {
            pos0: 0,
            stride: 1,
            fixed: None,
        };
        self.rope(q, t, nh, ihd, lid, at0, false)?;
        self.k.fp4_qdq(&self.s, q.ptr, t * nh * ihd, 32, false)?;
        let width = t / r;
        let wts = self.index_weights(p, x, t)?;
        let score = self.fbuf(t * width)?;
        let (ti, wi, nhi, ihdi) = (t as i32, width as i32, nh as i32, ihd as i32);
        let ik = self.idxk[&self.sh_idxk].ptr;
        unsafe {
            self.k.go(
                "ds_index_score",
                blocks(t * width * 32, 256),
                B256,
                0,
                &self.s,
                args![q.ptr, ik, wts.ptr, score.ptr, ti, wi, nhi, ihdi],
            )?
        };
        // masks, candidate blocks, top-k (as forward.rs), all on the device
        self.k
            .mask_scores(&self.s, score.ptr, 0, t, width, true, r)?;
        let src = c.candidate_source_layer;
        if lid as i64 == src {
            let cand = self.bytes(t * width)?;
            let (bs, kb) = (c.candidate_block_size, c.candidate_topk_blocks);
            let nb = width.div_ceil(bs);
            let (sc, sel) = (self.fbuf(t * nb)?, self.fbuf(t * nb)?);
            let (ci, ri, bsi, kbi) = (1i32, r as i32, bs as i32, kb as i32);
            unsafe {
                self.k.go(
                    "ds_candidates",
                    (t.max(1) as u32, 1, 1),
                    SEL,
                    0,
                    &self.s,
                    args![score.ptr, sc.ptr, sel.ptr, cand.ptr, ti, wi, ci, ri, bsi, kbi],
                )?
            };
            self.cand = Some((cand, width));
        } else if src >= 0 && (src as usize) < lid {
            let (cand, cw) = self.cand.context("no candidate mask")?;
            anyhow::ensure!(cw == width, "candidate mask width {cw} != {width}");
            self.k
                .mask_scores(&self.s, score.ptr, cand.ptr, t, width, false, r)?;
        }
        let k = c.index_topk.min(width);
        let out = self.fbuf(t * k)?;
        let scratch = self.fbuf(t * k)?;
        let (ki, si, oi, ci, ri) = (k as i32, k as i32, t as i32, 1i32, r as i32);
        unsafe {
            self.k.go(
                "ds_topk",
                (t.max(1) as u32, 1, 1),
                SEL,
                0,
                &self.s,
                args![score.ptr, scratch.ptr, out.ptr, ti, wi, ki, si, oi, ci, ri],
            )?
        };
        if std::env::var_os("DS_TAP").is_some() {
            self.sync()?;
            super::obs::d2h(t * k * 4);
            let v = out.dev().read::<i32>(0, t * k)?;
            self.taps
                .borrow_mut()
                .push((format!("{p}.topk"), v.iter().map(|&i| i as f32).collect()));
        }
        Ok((out, k))
    }

    /// The indexer's per-head weights, scaled and rounded.
    fn index_weights(&self, p: &str, x: Buf, t: usize) -> Result<Buf> {
        let (nh, ihd) = (self.cpu.cfg.index_n_heads, self.cpu.cfg.index_head_dim);
        let wts = self.lin(x, t, &format!("{p}.attn.indexer.weights_proj.weight"))?;
        let ws = ((ihd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32;
        let nw = (t * nh) as i64;
        unsafe {
            self.k.go(
                "ds_scale_round",
                blocks(t * nh, 256),
                B256,
                0,
                &self.s,
                args![wts.ptr, ws, nw],
            )?
        };
        Ok(wts)
    }

    /// A decode step's compressed KV and selection (as forward.rs `compress_kv_steps`: one
    /// position at a time), position-free on the host: the ratio-r compressor pools its
    /// partial group every position and writes it to the group's cache row, which is past
    /// every reader's width until the group completes and rewrites it. An index source
    /// writes [t][index_topk] rows (-1 holes past each row's min(k, width)); the candidate
    /// source a [t][max_seq_len / r] mask. Returns the cache's source layer.
    fn compress_dyn(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        qr: Buf,
        t: usize,
        offset: usize,
    ) -> Result<usize> {
        let c = self.cpu.cfg.clone();
        let (r, hd, d) = (c.compress_ratios[lid], c.head_dim, c.dim);
        let (is_kv, is_idx) = (c.is_kv_source(lid), c.is_index_source(lid));
        let (wmax, kmax) = (c.max_seq_len / r, c.index_topk);
        let src_l = c.candidate_source_layer;
        let makes_cand = lid as i64 == src_l;
        let eats_cand = is_idx && src_l >= 0 && (src_l as usize) < lid;
        let tk = if is_idx {
            Some(self.fbuf(t * kmax)?)
        } else {
            None
        };
        let cand_out = if makes_cand {
            Some(self.bytes(t * wmax)?)
        } else {
            None
        };
        let cand_in = if eats_cand {
            Some(self.cand.context("no candidate mask")?)
        } else {
            None
        };
        if comp_batched() {
            self.compress_rows(p, lid, x, t, is_kv, is_idx)?;
        }
        for e in 0..if comp_batched() { 0 } else { t } {
            let xe = x.at(e * d * 4);
            let latent = if is_kv {
                self.sh_ckv = lid;
                Some(self.compressor_dyn(p, lid, xe, e)?)
            } else {
                None
            };
            if is_idx {
                let s = self.span("indexer");
                self.indexer_keys(p, lid, latent, e)?;
                self.end(s);
            }
            if let Some(lat) = latent {
                let at = At::Dev {
                    add: e,
                    stride: 0,
                    fixed_r: r,
                };
                self.rope(lat, 1, 1, hd, lid, at, false)?;
                self.k.fp4_qdq(&self.s, lat.ptr, hd, 16, true)?;
                self.row_write(self.ckv[&lid].ptr, lat, hd, e, r)?;
            }
        }
        if let Some(tk) = tk {
            // the queries batched over the step: row e reads only the complete groups below
            // its width, all final once every position's key is written
            let s = self.span("indexer");
            self.indexer_dyn(p, lid, x, qr, t, offset, tk, cand_out, cand_in)?;
            self.end(s);
            self.topk = Some((tk, kmax));
        }
        if let Some(cb) = cand_out {
            self.cand = Some((cb, wmax));
        }
        Ok(self.sh_ckv)
    }

    /// The step's compressed rows of [`Self::compress_dyn`] for all `t` positions at once:
    /// the compressor GEMMs over the batch, `ds_comp_pool` (pool + round + rms per position
    /// against the ring as each position sees it), `ds_ring_last2`, the indexer key GEMV and
    /// `ds_comp_post` (rope, fp4 qdq, cache rows). Bit for bit the per-position loop.
    fn compress_rows(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
        is_kv: bool,
        is_idx: bool,
    ) -> Result<()> {
        if !is_kv {
            return Ok(());
        }
        let c = self.cpu.cfg.clone();
        let (r, hd, ihd, rd, eps) = (
            c.compress_ratios[lid],
            c.head_dim,
            c.index_head_dim,
            c.rope_head_dim,
            c.norm_eps,
        );
        self.sh_ckv = lid;
        let norm = format!("{p}.attn.compressor.norm.weight");
        let pp = self.p.ptr;
        let lat = if r == 1 {
            let kv = self.lin(x, t, &format!("{p}.attn.compressor.wkv.weight"))?;
            self.rms(kv, t, &norm)?;
            kv
        } else {
            let kv = self.raw_bf(x.ptr, t, &format!("{p}.attn.compressor.wkv.weight"))?;
            let sc = self.raw_bf(x.ptr, t, &format!("{p}.attn.compressor.wgate.weight"))?;
            let (kst, sst) = {
                let (a, bb) = &self.cst[&lid];
                (b(a), b(bb))
            };
            let lat = self.fbuf(t * hd)?;
            let nw = self.dw(&norm)?.v;
            let (ri, hdi) = (r as i32, hd as i32);
            unsafe {
                self.k.go(
                    "ds_comp_pool",
                    (t as u32, 1, 1),
                    (1024, 1, 1),
                    0,
                    &self.s,
                    args![kst.ptr, sst.ptr, kv.ptr, sc.ptr, lat.ptr, nw, pp, ri, hdi, eps],
                )?;
                let ti = t as i32;
                self.k.go(
                    "ds_ring_last2",
                    blocks(t * hd, 256),
                    B256,
                    0,
                    &self.s,
                    args![kst.ptr, kv.ptr, sst.ptr, sc.ptr, ti, hdi, ri, pp],
                )?;
            }
            for e in 0..t {
                self.jrows
                    .push((lid, e, kv.at(e * hd * 4), sc.at(e * hd * 4)));
            }
            lat
        };
        let key = if is_idx {
            self.sh_idxk = lid;
            Some(self.lin(lat, t, &format!("{p}.attn.indexer.wk.weight"))?)
        } else {
            None
        };
        let (kn, ikp, idxk) = match key {
            Some(k) => (
                self.dw(&format!("{p}.attn.indexer.k_norm.weight"))?.v,
                k.ptr,
                self.idxk[&lid].ptr,
            ),
            None => (self.dw(&norm)?.v, 0, 0),
        };
        let (hdi, ihdi, rdi, ri, ti) = (hd as i32, ihd as i32, rd as i32, r as i32, t as i32);
        let (ckv, tab) = (self.ckv[&lid].ptr, self.tabs[lid].ptr);
        unsafe {
            self.k.go(
                "ds_comp_post",
                (t as u32, 1 + key.is_some() as u32, 1),
                (1024, 1, 1),
                0,
                &self.s,
                args![lat.ptr, hdi, ckv, ikp, ihdi, kn, idxk, tab, rdi, pp, ri, ti, eps],
            )?
        };
        Ok(())
    }

    /// One position's compressed latent (pre-RoPE): see [`Self::compress_dyn`].
    fn compressor_dyn(&mut self, p: &str, lid: usize, xe: Buf, e: usize) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let norm = format!("{p}.attn.compressor.norm.weight");
        if r == 1 {
            let kv = self.lin(xe, 1, &format!("{p}.attn.compressor.wkv.weight"))?;
            self.rms(kv, 1, &norm)?;
            return Ok(kv);
        }
        let kv = self.raw(xe.ptr, 1, &format!("{p}.attn.compressor.wkv.weight"))?;
        let sc = self.raw(xe.ptr, 1, &format!("{p}.attn.compressor.wgate.weight"))?;
        let (kst, sst) = {
            let (a, bb) = &self.cst[&lid];
            (b(a), b(bb))
        };
        self.ring_write(kst, kv, 1, hd, r, e)?;
        self.ring_write(sst, sc, 1, hd, r, e)?;
        self.jrows.push((lid, e, kv, sc));
        let out = self.pool(kst.ptr, sst.ptr, 1, r)?;
        self.rms(out, 1, &norm)?;
        Ok(out)
    }

    /// One position's indexer key (see [`Self::compress_dyn`]), when this layer owns them.
    fn indexer_keys(&mut self, p: &str, lid: usize, latent: Option<Buf>, e: usize) -> Result<()> {
        let c = &self.cpu.cfg;
        let (r, ihd) = (c.compress_ratios[lid], c.index_head_dim);
        if !c.is_kv_source(lid) {
            return Ok(());
        }
        self.sh_idxk = lid;
        let Some(lat) = latent else { return Ok(()) };
        let k = self.lin(lat, 1, &format!("{p}.attn.indexer.wk.weight"))?;
        self.rms(k, 1, &format!("{p}.attn.indexer.k_norm.weight"))?;
        let at = At::Dev {
            add: e,
            stride: 0,
            fixed_r: r,
        };
        self.rope(k, 1, 1, ihd, lid, at, false)?;
        self.k.fp4_qdq(&self.s, k.ptr, ihd, 32, false)?;
        self.row_write(self.idxk[&lid].ptr, k, ihd, e, r)
    }

    /// The step's `t` indexer queries, scores and top-k rows into `out` [t][index_topk] (see
    /// [`Self::compress_dyn`]); the candidate mask out [t][wmax] or in [t][stride].
    #[allow(clippy::too_many_arguments)]
    fn indexer_dyn(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        qr: Buf,
        t: usize,
        offset: usize,
        out: Buf,
        cand_out: Option<Buf>,
        cand_in: Option<(Buf, usize)>,
    ) -> Result<()> {
        let c = self.cpu.cfg.clone();
        let (r, nh, ihd) = (c.compress_ratios[lid], c.index_n_heads, c.index_head_dim);
        let (wmax, kmax) = (c.max_seq_len / r, c.index_topk);
        let q = self.lin(qr, t, &format!("{p}.attn.indexer.wq_b.weight"))?;
        let at = At::Dev {
            add: 0,
            stride: 1,
            fixed_r: 0,
        };
        self.rope(q, t, nh, ihd, lid, at, false)?;
        self.k.fp4_qdq(&self.s, q.ptr, t * nh * ihd, 32, false)?;
        let wts = self.index_weights(p, x, t)?;
        let score = self.fbuf(t * wmax)?;
        let ik = self.idxk[&self.sh_idxk].ptr;
        let (pp, ri, nhi, ihdi, wi) = (self.p.ptr, r as i32, nh as i32, ihd as i32, wmax as i32);
        let (sg, ty) = (blocks(wmax * 32, 256), t as u32);
        unsafe {
            self.gk.go(
                "dg_index_score",
                (sg.0, ty, 1),
                B256,
                0,
                &self.s,
                args![q.ptr, ik, wts.ptr, score.ptr, pp, 0i32, ri, nhi, ihdi, wi],
            )?
        };
        if let Some(cb) = cand_out {
            let (bs, kb) = (c.candidate_block_size, c.candidate_topk_blocks);
            let nb = wmax.div_ceil(bs);
            let (sc, sel) = (self.fbuf(t * nb)?, self.fbuf(t * nb)?);
            let (bsi, kbi, nbi) = (bs as i32, kb as i32, nb as i32);
            unsafe {
                self.gk.go(
                    "dg_candidates",
                    (ty, 1, 1),
                    SEL,
                    0,
                    &self.s,
                    args![score.ptr, sc.ptr, sel.ptr, cb.ptr, pp, 0i32, ri, bsi, kbi, wi, nbi, wi],
                )?
            };
        } else if let Some((cb, st)) = cand_in {
            let (mg, sti) = (blocks(wmax, 256), st as i32);
            unsafe {
                self.gk.go(
                    "dg_mask_scores",
                    (mg.0, ty, 1),
                    B256,
                    0,
                    &self.s,
                    args![score.ptr, cb.ptr, pp, 0i32, ri, wi, sti],
                )?
            };
        }
        let sel = self.fbuf(t * kmax)?;
        let (ki, oi) = (kmax as i32, offset as i32);
        unsafe {
            self.gk.go(
                "dg_topk",
                (ty, 1, 1),
                SEL,
                0,
                &self.s,
                args![score.ptr, sel.ptr, out.ptr, pp, 0i32, ri, ki, oi, wi],
            )
        }
    }

    // moe

    /// An fp8 quant-dequantized copy of `x` (n values). Kept out of the GEMV: fused into its
    /// column loop (`qdq` in `ds_lin_*`, bit-identical) it costs 2-8x the GEMV at t = 1..8,
    /// since every warp redoes it.
    fn qdq(&self, x: Buf, n: usize) -> Result<Buf> {
        let xq = self.fbuf(n)?;
        self.copy(xq.ptr, x.ptr, n)?;
        self.k.fp8_qdq(&self.s, xq.ptr, n)?;
        Ok(xq)
    }

    fn swiglu_into(&self, g: Buf, u: Buf, w: u64, h: Buf, n: usize, inter: usize) -> Result<()> {
        let (nn, ii, lim) = ((n * inter) as i64, inter as i32, self.cpu.cfg.swiglu_limit);
        unsafe {
            self.k.go(
                "ds_swiglu",
                blocks(n * inter, 256),
                B256,
                0,
                &self.s,
                args![g.ptr, u.ptr, w, h.ptr, nn, ii, lim],
            )
        }
    }

    fn mlp(&self, p: &str, x: Buf, n: usize, w: u64) -> Result<Buf> {
        let (n1, n3) = (format!("{p}.w1.weight"), format!("{p}.w3.weight"));
        if self.fast {
            let (w1, w3) = (self.dw(&n1)?, self.dw(&n3)?);
            if Kernels::fast_ok(&w1.v) && Kernels::fast_ok(&w3.v) {
                let h = self.fbuf(n * w1.v.rows as usize)?;
                // the K-split GEMV quantizes the activation itself: skip the qdq pass
                let xq = if self.k.ks_takes(&w1.v) {
                    x
                } else {
                    self.qdq(x, n * w1.v.cols as usize)?
                };
                self.k.linear_fast(
                    &self.s,
                    &w1.v,
                    Some(&w3.v),
                    xq.ptr,
                    h.ptr,
                    n,
                    true,
                    false,
                    w,
                    self.cpu.cfg.swiglu_limit,
                )?;
                let w2 = self.dw(&format!("{p}.w2.weight"))?;
                if self.k.ks_takes(&w2.v) {
                    let y = self.fbuf(n * w2.v.rows as usize)?;
                    self.k
                        .linear_fast(&self.s, &w2.v, None, h.ptr, y.ptr, n, true, false, 0, 0.0)?;
                    return Ok(y);
                }
                return self.lin(h, n, &format!("{p}.w2.weight"));
            }
        }
        let g = self.lin(x, n, &n1)?;
        let u = self.lin(x, n, &n3)?;
        let inter = self.dw(&n1)?.v.rows as usize;
        let h = self.fbuf(n * inter)?;
        self.swiglu_into(g, u, w, h, n, inter)?;
        self.lin(h, n, &format!("{p}.w2.weight"))
    }

    /// `lin` over at most expert `e`'s row count (from the device CSR `off`) of `t` rows,
    /// into `y`, with `xq` as the activation-quant scratch.
    #[allow(clippy::too_many_arguments)]
    fn lin_csr(
        &self,
        x: Buf,
        t: usize,
        name: &str,
        off: Buf,
        e: usize,
        y: Buf,
        xq: Buf,
    ) -> Result<()> {
        let w = self.dw(name)?;
        let k = w.v.cols as usize;
        let xp = if w.v.fmt >= 2 {
            self.copy(xq.ptr, x.ptr, t * k)?;
            self.k.fp8_qdq(&self.s, xq.ptr, t * k)?;
            xq.ptr
        } else {
            x.ptr
        };
        let n = w.v.rows as usize;
        let (wv, ti, r, ei) = (w.v, t as i32, 1i32, e as i32);
        unsafe {
            self.k.go(
                "ds_linear_csr",
                blocks(t * n * 32, 256),
                B256,
                0,
                &self.s,
                args![wv, xp, y.ptr, ti, r, off.ptr, ei],
            )
        }
    }

    fn moe(&self, p: &str, x: Buf, t: usize, lid: usize) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (ne, topk) = c.moe(lid);
        let d = c.dim;
        let gw = self.dw(&format!("{p}.ffn.gate.weight"))?;
        let bias = self.dw(&format!("{p}.ffn.gate.bias"))?;
        let (ids, wts) = (self.fbuf(t * topk)?, self.fbuf(t * topk)?);
        let (counts, off) = (self.fbuf(ne)?, self.fbuf(ne + 1)?);
        let (rows, rw) = (self.fbuf(t * topk)?, self.fbuf(t * topk)?);
        let tab = if self.moe_grouped && self.expert_slots == 0 {
            self.moe_tab(p, lid, ne)?
        } else {
            None
        };
        // DS_MOE_GATE=1: the one-launch router (bit-identical; slower than the three launches
        // on the 3090 bench so far, so opt-in for sweeping on H200)
        let fused = std::env::var("DS_MOE_GATE").is_ok_and(|v| v == "1")
            && Kernels::moe_gate_ok(&gw.v, t, ne, topk);
        let sh = match &tab {
            Some(tb) => self.moe_sh_weights(p, tb, t, topk)?,
            None => None,
        };
        let mut elist = None;
        let mut preq = false;
        if fused {
            // gate GEMV + router + CSR (+ the grouped plan) in one launch
            let s = self.span("router");
            let raw = self.fbuf(t * ne)?;
            let plan = match &tab {
                Some(tb) => {
                    let el = self.fbuf((t * topk).min(tb.local))?;
                    elist = Some(el);
                    (tb.mask.ptr, el.ptr, (t * topk).min(tb.local))
                }
                None => (0, 0, 0),
            };
            self.k.moe_gate(
                &self.s,
                &gw.v,
                x.ptr,
                raw.ptr,
                bias.v.w,
                ids.ptr,
                wts.ptr,
                (counts.ptr, off.ptr, rows.ptr, rw.ptr),
                plan,
                t,
                topk,
                c.norm_topk_prob,
                c.route_scale,
            )?;
            self.end(s);
        } else {
            let s = self.span("router");
            // the gate GEMV in a relaxed order unless DS_EXACT / DS_ROUTER_EXACT=1
            let raw = if self.k.exact || self.router_exact {
                self.raw(x.ptr, t, &format!("{p}.ffn.gate.weight"))?
            } else {
                let raw = self.fbuf(t * ne)?;
                self.k.gemv_bf16(&self.s, &gw.v, x.ptr, raw.ptr, t)?;
                raw
            };
            self.k.router(
                &self.s,
                raw.ptr,
                bias.v.w,
                ids.ptr,
                wts.ptr,
                t,
                ne,
                topk,
                c.norm_topk_prob,
                c.route_scale,
            )?;
            self.end(s);
            // DS_ROUTE_DUMP=path (direct path, DS_GRAPH=0): append "layer t ids..." per MoE
            // call to path.<thread> (one file per rank's thread), for replaying the real
            // routing in mega_moe_slots
            if let Some(path) = std::env::var_os("DS_ROUTE_DUMP") {
                use std::io::Write;
                self.sync()?;
                let v = ids.dev().read::<i32>(0, t * topk)?;
                let mut line = format!("{lid} {t}");
                for i in &v {
                    line.push_str(&format!(" {i}"));
                }
                line.push('\n');
                let mut p = std::path::PathBuf::from(path).into_os_string();
                p.push(format!(".{:?}", std::thread::current().id()).replace(['(', ')'], ""));
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)?
                    .write_all(line.as_bytes())?;
            }
            // per-expert CSR on the device; with the shared expert folded, the plan and the
            // GU's activation quantization ride in the same launch
            let s = self.span("route");
            match (&tab, &sh) {
                (Some(tb), Some(_)) => {
                    let cap = (t * topk).min(tb.local);
                    let el = self.fbuf(cap)?;
                    self.k.route_plan_q(
                        &self.s,
                        (ids.ptr, wts.ptr),
                        (counts.ptr, off.ptr, rows.ptr, rw.ptr),
                        (tb.mask.ptr, el.ptr, cap),
                        x.ptr,
                        (t, topk, ne, d),
                    )?;
                    elist = Some(el);
                    preq = true;
                }
                _ => self.k.route(
                    &self.s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
                )?,
            }
            self.end(s);
        }
        if let Some(tab) = tab {
            if t <= 8 {
                let (y, folded) = self.moe_grouped_routed(
                    p, &tab, x, t, topk, ne, ids, counts, off, rows, rw, elist, sh, preq,
                )?;
                if folded {
                    return self.moe_finish_folded(x, t, y);
                }
                return self.moe_finish(p, x, t, y);
            }
            let y = self.moe_grouped_tiled(&tab, x, t, topk, ne, ids, counts, off, rows, rw)?;
            return self.moe_finish(p, x, t, y);
        }
        // the expert LRU uploads on the host's say: it alone reads the counts back, to skip
        // (and not upload) unrouted experts
        let live: Vec<bool> = if self.expert_slots > 0 {
            self.sync()?;
            super::obs::d2h(ne * 4);
            counts
                .dev()
                .read::<i32>(0, ne)?
                .iter()
                .map(|&n| n > 0)
                .collect()
        } else {
            vec![true; ne]
        };
        let s = self.span("experts");
        let y = self.zbuf(t * d)?;
        // one set of expert scratch, reused expert after expert (the stream orders it)
        let inter =
            (0..ne)
                .find(|&e| (self.owns_expert)(e))
                .map_or(Ok(1), |e| -> Result<usize> {
                    Ok(self
                        .cpu
                        .st
                        .info(&format!("{p}.ffn.experts.{e}.w1.weight"))?
                        .shape[0])
                })?;
        let (xs, w, out) = (self.fbuf(t * d)?, self.fbuf(t)?, self.fbuf(t * d)?);
        let (g, u, h) = (
            self.fbuf(t * inter)?,
            self.fbuf(t * inter)?,
            self.fbuf(t * inter)?,
        );
        let xq = self.fbuf(t * d.max(inter))?;
        let (ei_t, di) = (t as i32, d as i32);
        for e in (0..ne).filter(|&e| live[e] && (self.owns_expert)(e)) {
            let ei = e as i32;
            unsafe {
                self.k.go(
                    "ds_gather_csr",
                    blocks(t * d, 256),
                    B256,
                    0,
                    &self.s,
                    args![x.ptr, off.ptr, rows.ptr, rw.ptr, xs.ptr, w.ptr, ei, ei_t, di],
                )?
            };
            let pe = format!("{p}.ffn.experts.{e}");
            let n1 = format!("{pe}.w1.weight");
            self.lin_csr(xs, t, &n1, off, e, g, xq)?;
            self.lin_csr(xs, t, &format!("{pe}.w3.weight"), off, e, u, xq)?;
            self.swiglu_into(g, u, w.ptr, h, t, inter)?;
            self.lin_csr(h, t, &format!("{pe}.w2.weight"), off, e, out, xq)?;
            unsafe {
                self.k.go(
                    "ds_scatter_csr",
                    blocks(t * d, 256),
                    B256,
                    0,
                    &self.s,
                    args![y.ptr, out.ptr, off.ptr, rows.ptr, ei, ei_t, di],
                )?
            };
        }
        self.end(s);
        self.moe_finish(p, x, t, y)
    }

    /// The shared expert's weights when it folds into the grouped launches as one more slot
    /// (on rank 0 alone under EP, so the reduce sums it once): its MLP launches and the add
    /// gone. `DS_MOE_SH=0` keeps the MLP.
    fn moe_sh_weights(
        &self,
        p: &str,
        tab: &MoeTab,
        t: usize,
        topk: usize,
    ) -> Result<Option<([DsW; 3], bool)>> {
        if !self.moe_sh || t > 8 {
            return Ok(None);
        }
        let pe = format!("{p}.ffn.shared_experts");
        let ws = [
            self.dw(&format!("{pe}.w1.weight"))?,
            self.dw(&format!("{pe}.w3.weight"))?,
            self.dw(&format!("{pe}.w2.weight"))?,
        ];
        let ok = self.k.moe_sh_ok(
            [&ws[0].v, &ws[1].v, &ws[2].v],
            tab.fmt,
            t,
            topk,
            self.cpu.cfg.dim,
            tab.inter,
        );
        if !ok {
            return Ok(None);
        }
        let (rank, world) = (self.obs.rank.get(), self.ep_world);
        let (a, b, c) = (ws[0].v, ws[1].v, ws[2].v);
        let sl = tab.inter / world.max(1);
        // the slice: whole 32-row/column scale blocks (fp8), whole groups (both)
        if !self.sh_split || world <= 1 || tab.inter % (32 * world) != 0 || a.fmt != 2 || c.fmt != 2
        {
            return Ok(Some(([a, b, c], rank == 0)));
        }
        let rows = |w: DsW| {
            let mut v = w;
            v.rows = sl as i32;
            v.w += (rank * sl * w.cols as usize) as u64;
            v.s += ((rank * sl / 32) * (w.cols as usize / 32)) as u64;
            v
        };
        let key = format!("{pe}.w2.weight#rank{rank}of{world}");
        let c2 = match self.w.borrow().get(&key) {
            Some(w) => Some(w.v),
            None => None,
        };
        let c2 = match c2 {
            Some(v) => v,
            None => {
                // down's K slice [d][sl] (and its [d / 32][sl / 32] scales), copied once
                let (d, k) = (c.rows as usize, c.cols as usize);
                let wb = DevBuf::alloc(d * sl)?;
                let sb = DevBuf::alloc(d.div_ceil(32) * (sl / 32))?;
                for (dst, src, r, w, pitch, c0) in [
                    (wb.ptr, c.w, d, sl, k, rank * sl),
                    (sb.ptr, c.s, d.div_ceil(32), sl / 32, k / 32, rank * sl / 32),
                ] {
                    let (ri, wi, pi, ci) = (r as i32, w as i32, pitch as i32, c0 as i32);
                    unsafe {
                        self.k.go(
                            "ds_copy2d",
                            blocks(r * w, 256),
                            B256,
                            0,
                            &self.s,
                            args![dst, src, ri, wi, pi, ci],
                        )?
                    };
                }
                let mut v = c;
                (v.w, v.s, v.cols) = (wb.ptr, sb.ptr, sl as i32);
                self.w
                    .borrow_mut()
                    .insert(key, Rc::new(DevW::from_parts(wb, Some(sb), v)));
                v
            }
        };
        Ok(Some(([rows(a), rows(b), c2], true)))
    }

    /// The routed experts' partial sum y [t][d] by the grouped kernels: plan (live local
    /// experts), gate/up + SwiGLU, quantize, down to per-slot partials, ordered reduce.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_routed(
        &self,
        p: &str,
        tab: &MoeTab,
        x: Buf,
        t: usize,
        topk: usize,
        ne: usize,
        ids: Buf,
        counts: Buf,
        off: Buf,
        rows: Buf,
        rw: Buf,
        planned: Option<Buf>,
        sh: Option<([DsW; 3], bool)>,
        preq: bool,
    ) -> Result<(Buf, bool)> {
        let d = self.cpu.cfg.dim;
        let (n, inter) = (t * topk, tab.inter);
        let slots = n.min(tab.local);
        let s = self.span("grouped_gu");
        let elist = match planned {
            Some(e) => e,
            None => {
                let elist = self.fbuf(slots)?;
                let (nei, ci) = (ne as i32, slots as i32);
                unsafe {
                    self.k.go(
                        "ds_moe_plan",
                        (1, 1, 1),
                        (32, 1, 1),
                        0,
                        &self.s,
                        args![counts.ptr, tab.mask.ptr, elist.ptr, nei, ci],
                    )?
                };
                elist
            }
        };
        // the K-split kernels quantize their activation themselves (idempotent on a
        // quantized one), so the fp8 qdq passes are only for the exact kernels
        let xq = if self.k.moe_ks_takes(t, d, t) {
            x
        } else {
            self.qdq(x, t * d)?
        };
        let lim = self.cpu.cfg.swiglu_limit;
        if let Some((ws, on)) = sh {
            let h = self.fbuf((n + t) * inter)?;
            self.k.moe_gu_sh(
                &self.s,
                tab.fmt,
                (tab.w1.ptr, tab.w3.ptr, elist.ptr),
                (off.ptr, rows.ptr, rw.ptr),
                (&ws[0], &ws[1], on),
                if preq { 0 } else { x.ptr },
                h.ptr,
                (t, n, slots, inter, d),
                lim,
            )?;
            self.end(s);
            let s = self.span("grouped_down");
            let part = self.fbuf((n + t) * d)?;
            self.k.moe_down_sh(
                &self.s,
                tab.fmt,
                (tab.w2.ptr, elist.ptr, off.ptr),
                (&ws[2], on),
                h.ptr,
                part.ptr,
                (t, n, slots, d, inter),
            )?;
            let y = self.fbuf(t * d)?;
            let (ti, ki, di, shi) = (t as i32, topk as i32, d as i32, on as i32);
            unsafe {
                self.k.go(
                    "ds_moe_reduce_sh",
                    blocks(t * d, 256),
                    B256,
                    0,
                    &self.s,
                    args![
                        y.ptr,
                        part.ptr,
                        ids.ptr,
                        off.ptr,
                        rows.ptr,
                        tab.mask.ptr,
                        ti,
                        ki,
                        di,
                        shi
                    ],
                )?
            };
            self.end(s);
            return Ok((y, true));
        }
        let h = self.fbuf(n * inter)?;
        self.k.moe_gu(
            &self.s, tab.fmt, tab.w1.ptr, tab.w3.ptr, elist.ptr, off.ptr, rows.ptr, rw.ptr, xq.ptr,
            h.ptr, t, slots, inter, d, lim,
        )?;
        if !self.k.moe_ks_takes(t, inter, n) {
            self.k.fp8_qdq(&self.s, h.ptr, n * inter)?;
        }
        self.end(s);
        let s = self.span("grouped_down");
        let part = self.fbuf(n * d)?;
        self.k.moe_down(
            &self.s, tab.fmt, tab.w2.ptr, elist.ptr, off.ptr, h.ptr, part.ptr, t, slots, d, inter,
            topk,
        )?;
        let y = self.fbuf(t * d)?;
        let (ti, ki, di) = (t as i32, topk as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_moe_reduce",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![
                    y.ptr,
                    part.ptr,
                    ids.ptr,
                    off.ptr,
                    rows.ptr,
                    tab.mask.ptr,
                    ti,
                    ki,
                    di
                ],
            )?
        };
        self.end(s);
        Ok((y, false))
    }

    /// [`Self::moe_grouped_routed`] for t > 8 (prefill): the same plan and ordered reduce,
    /// the projections over 8-row tiles of each expert's segment ([`Kernels::moe_tiled`]),
    /// one launch per projection for the layer.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_tiled(
        &self,
        tab: &MoeTab,
        x: Buf,
        t: usize,
        topk: usize,
        ne: usize,
        ids: Buf,
        counts: Buf,
        off: Buf,
        rows: Buf,
        rw: Buf,
    ) -> Result<Buf> {
        let d = self.cpu.cfg.dim;
        let (n, inter) = (t * topk, tab.inter);
        let slots = n.min(tab.local);
        let s = self.span("grouped_tiled");
        let elist = self.fbuf(slots)?;
        let (nei, ci) = (ne as i32, slots as i32);
        unsafe {
            self.k.go(
                "ds_moe_plan",
                (1, 1, 1),
                (32, 1, 1),
                0,
                &self.s,
                args![counts.ptr, tab.mask.ptr, elist.ptr, nei, ci],
            )?
        };
        let tiles = self.fbuf(Kernels::moe_tile_cap(n, slots))?;
        // the K-split tiles quantize their activation themselves (as the t <= 8 path)
        let xq = if self.k.moe_tiled_ks(d, inter) {
            x
        } else {
            self.qdq(x, t * d)?
        };
        let (h, part) = (self.fbuf(n * inter)?, self.fbuf(n * d)?);
        let wide = n * d.max(inter);
        let (codes, scales) = (self.bytes(wide)?, self.fbuf(wide / 32)?);
        self.k.moe_tiled(
            &self.s,
            tab.fmt,
            (tab.w1.ptr, tab.w3.ptr, tab.w2.ptr),
            (elist.ptr, slots, tiles.ptr),
            (off.ptr, rows.ptr, rw.ptr),
            (xq.ptr, h.ptr, part.ptr),
            (codes.ptr, scales.ptr),
            (t, topk, d, inter),
            self.cpu.cfg.swiglu_limit,
        )?;
        let y = self.fbuf(t * d)?;
        let (ti, ki, di) = (t as i32, topk as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_moe_reduce",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![
                    y.ptr,
                    part.ptr,
                    ids.ptr,
                    off.ptr,
                    rows.ptr,
                    tab.mask.ptr,
                    ti,
                    ki,
                    di
                ],
            )?
        };
        self.end(s);
        Ok(y)
    }

    /// Layer `lid`'s grouped-MoE table, built (uploading every local expert) on first use;
    /// None when the experts are not one `linear_fast` format (the loop then runs).
    fn moe_tab(&self, p: &str, lid: usize, ne: usize) -> Result<Option<Rc<MoeTab>>> {
        if let Some(t) = self.moe_tabs.borrow().get(&lid) {
            return Ok(t.clone());
        }
        let zero = DsW::default();
        let (mut w1, mut w3, mut w2, mut mask) = (
            vec![zero; ne],
            vec![zero; ne],
            vec![zero; ne],
            vec![0i32; ne],
        );
        let mut keep = Vec::new();
        let mut ok = true;
        let mut shape: Option<(i32, i32, i32)> = None;
        for e in (0..ne).filter(|&e| (self.owns_expert)(e)) {
            let pe = format!("{p}.ffn.experts.{e}");
            let ws = [
                self.dw(&format!("{pe}.w1.weight"))?,
                self.dw(&format!("{pe}.w3.weight"))?,
                self.dw(&format!("{pe}.w2.weight"))?,
            ];
            let (a, b, c) = (ws[0].v, ws[1].v, ws[2].v);
            let sh = (a.fmt, a.rows, a.cols);
            ok &= Kernels::fast_ok(&a)
                && Kernels::fast_ok(&c)
                && (b.fmt, b.rows, b.cols) == sh
                && (c.fmt, c.rows, c.cols) == (a.fmt, a.cols, a.rows)
                && *shape.get_or_insert(sh) == sh;
            (w1[e], w3[e], w2[e], mask[e]) = (a, b, c, 1);
            keep.extend(ws);
        }
        let tab = match shape {
            Some((fmt, inter, _)) if ok => Some(Rc::new(MoeTab {
                w1: DevBuf::from_slice(&w1)?,
                w3: DevBuf::from_slice(&w3)?,
                w2: DevBuf::from_slice(&w2)?,
                mask: DevBuf::from_slice(&mask)?,
                fmt,
                inter: inter as usize,
                local: mask.iter().filter(|&&m| m != 0).count(),
                _keep: keep,
            })),
            _ => None,
        };
        self.moe_tabs.borrow_mut().insert(lid, tab.clone());
        Ok(tab)
    }

    /// The routed sum `y` [t][d] -> the MoE output: the EP reduce hook, then the shared
    /// expert added and the sum bf16-rounded once.
    fn moe_finish(&self, p: &str, x: Buf, t: usize, y: Buf) -> Result<Buf> {
        let d = self.cpu.cfg.dim;
        // a reduce that only enqueues runs on `side`, overlapping the shared expert (which
        // does not read `y`); one that syncs the host (Host) stays in order on `s`
        let forked = self.moe_reduce.is_some() && self.reduce_graphable;
        if let Some(reduce) = &self.moe_reduce {
            let rs = if forked { &self.side } else { &self.s };
            if forked {
                self.fork.record(&self.s)?;
                self.side.wait(&self.fork)?;
            }
            let s = self.obs.span(rs, "reduce", None);
            reduce(&y.dev(), t * d, rs)?;
            self.obs.end(rs, s);
            if forked {
                self.join.record(&self.side)?;
            }
        }
        let s = self.span("shared");
        let sh = self.mlp(&format!("{p}.ffn.shared_experts"), x, t, 0)?;
        self.end(s);
        if forked {
            self.s.wait(&self.join)?;
        }
        if self.defer_sh.get() {
            self.sh.set(Some(sh));
            return Ok(y);
        }
        let s = self.span("finish");
        let nn = (t * d) as i64;
        unsafe {
            self.k.go(
                "ds_add_round",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![y.ptr, sh.ptr, nn],
            )?
        };
        self.end(s);
        Ok(y)
    }

    /// [`Self::moe_finish`] when the grouped launches already summed the shared expert into
    /// `y` (unrounded): the EP reduce in order, then the one bf16 rounding — deferred to the mHC
    /// post-mix as `bf(y + 0)` (a zero `sh`) or here.
    fn moe_finish_folded(&self, _x: Buf, t: usize, y: Buf) -> Result<Buf> {
        let d = self.cpu.cfg.dim;
        if let Some(reduce) = &self.moe_reduce {
            let s = self.span("reduce");
            reduce(&y.dev(), t * d, &self.s)?;
            self.end(s);
        }
        if self.defer_sh.get() {
            self.sh.set(Some(Buf::from(&self.zero_rows)));
            return Ok(y);
        }
        let s = self.span("finish");
        let nn = (t * d) as i64;
        unsafe {
            self.k.go(
                "ds_round",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![y.ptr, nn],
            )?
        };
        self.end(s);
        Ok(y)
    }

    // mhc and blocks

    fn hc_mixes(&self, p: &str, kind: &str, x: Buf, t: usize) -> Result<Mix> {
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let mix = (2 + hc) * hc;
        let mixes = self.raw(x.ptr, t, &format!("{p}.hc_{kind}_fn"))?;
        let m = Mix {
            pre: self.fbuf(t * hc)?,
            post: self.fbuf(t * hc)?,
            comb: self.fbuf(t * hc * hc)?,
        };
        let sc = self.dw(&format!("{p}.hc_{kind}_scale"))?;
        let base = self.dw(&format!("{p}.hc_{kind}_base"))?;
        assert!(mix <= 80, "ds_hc_sinkhorn stages <= 80 mixes");
        // the row rsqrt and the Sinkhorn in one launch (ds_hc_scale + ds_sinkhorn, bit for bit)
        let (n, neps, hci) = ((hc * d) as i32, c.norm_eps, hc as i32);
        let (it, eps) = (c.hc_sinkhorn_iters as i32, c.hc_eps);
        let (scp, bp) = (sc.v.w, base.v.w);
        unsafe {
            self.k.go(
                "ds_hc_sinkhorn",
                (t as u32, 1, 1),
                B256,
                0,
                &self.s,
                args![
                    x.ptr, mixes.ptr, n, neps, scp, bp, m.pre.ptr, m.post.ptr, m.comb.ptr, hci, it,
                    eps
                ],
            )?
        };
        Ok(m)
    }

    fn hc_pre(&self, x: Buf, pre: Buf, t: usize) -> Result<Buf> {
        let (hc, d) = (self.cpu.cfg.hc_mult, self.cpu.cfg.dim);
        let out = self.fbuf(t * d)?;
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_hc_pre",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![x.ptr, pre.ptr, out.ptr, ti, hci, di],
            )?
        };
        Ok(out)
    }

    fn hc_post(&self, h: Buf, res: Buf, m: &Mix, t: usize) -> Result<Buf> {
        self.hc_post2(h, None, res, m, t)
    }

    /// [`GpuModel::hc_post`] of `bf(h + h2)` when `h2` is given.
    fn hc_post2(&self, h: Buf, h2: Option<Buf>, res: Buf, m: &Mix, t: usize) -> Result<Buf> {
        let (hc, d) = (self.cpu.cfg.hc_mult, self.cpu.cfg.dim);
        let out = self.fbuf(t * hc * d)?;
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        let h2 = h2.map_or(0, |b| b.ptr);
        unsafe {
            self.k.go(
                "ds_hc_post",
                blocks(t * hc * d, 256),
                B256,
                0,
                &self.s,
                args![h.ptr, h2, res.ptr, m.post.ptr, m.comb.ptr, out.ptr, ti, hci, di],
            )?
        };
        Ok(out)
    }

    /// `hc_post` and the next stage's mixes GEMM (`{p}.hc_{kind}_fn`) in one launch: the
    /// new stream and its unscaled mixes. With no `m`, the GEMM alone on the stream `res`
    /// (as [`GpuModel::raw`], bit for bit, spread over more SMs).
    #[allow(clippy::too_many_arguments)]
    fn hc_post_mix(
        &self,
        h: Buf,
        h2: Option<Buf>,
        res: Buf,
        m: Option<&Mix>,
        t: usize,
        p: &str,
        kind: &str,
    ) -> Result<(Buf, (Buf, i32))> {
        let (hc, d) = (self.cpu.cfg.hc_mult, self.cpu.cfg.dim);
        let w = self.dw(&format!("{p}.hc_{kind}_fn"))?;
        let out = match m {
            Some(_) => self.fbuf(t * hc * d)?,
            None => res,
        };
        let (post, comb) = m.map_or((0, 0), |m| (m.post.ptr, m.comb.ptr));
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        let (h2, wv) = (h2.map_or(0, |b| b.ptr), w.v);
        if !self.hc_exact && !self.k.exact {
            // the K-split GEMM (relaxed bar): fixed-order partials ds_hc_stage sums
            let np = (hc * d).div_ceil(256);
            let part = self.fbuf(t * np * w.v.rows as usize)?;
            unsafe {
                self.k.go(
                    "ds_hc_mix_ks",
                    (np as u32, t.div_ceil(8) as u32, 1),
                    B256,
                    0,
                    &self.s,
                    args![h.ptr, h2, res.ptr, post, comb, out.ptr, wv, part.ptr, ti, hci, di],
                )?
            };
            return Ok((out, (part, np as i32)));
        }
        let mixes = self.fbuf(t * w.v.rows as usize)?;
        unsafe {
            self.k.go(
                "ds_hc_post_mix",
                (w.v.rows as u32, t.div_ceil(8) as u32, 1),
                (32 * t.min(8) as u32, 1, 1),
                0,
                &self.s,
                args![h.ptr, h2, res.ptr, post, comb, out.ptr, wv, mixes.ptr, ti, hci, di],
            )?
        };
        Ok((out, (mixes, 1)))
    }

    /// A whole mHC stage in one launch on the K-split path (`ds_hc_fused_ks`): the post-mix
    /// of `h` (+ `h2`) into `res` by `m` (no `m`: the stream `res` as is), then this stage's
    /// mixes, Sinkhorn and `rms(hc_pre(stream, pre), norm)`. The stream, the coefficients
    /// and the block input.
    #[allow(clippy::too_many_arguments)]
    fn hc_fused(
        &self,
        h: Buf,
        h2: Option<Buf>,
        res: Buf,
        m: Option<&Mix>,
        t: usize,
        p: &str,
        kind: &str,
        pre: Buf,
        norm: &str,
    ) -> Result<(Buf, Mix, Buf)> {
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let w = self.dw(&format!("{p}.hc_{kind}_fn"))?;
        let rows = w.v.rows as usize;
        assert!(
            rows <= 80 && 256 % hc == 0,
            "ds_hc_fused_ks: <= 80 mixes, hc | 256"
        );
        let xo = match m {
            Some(_) => self.fbuf(t * hc * d)?,
            None => res,
        };
        let (post, comb) = m.map_or((0, 0), |m| (m.post.ptr, m.comb.ptr));
        let gx = d.div_ceil(256 / hc);
        let part = self.fbuf(t * gx * (rows + 2))?;
        let o = Mix {
            pre: self.fbuf(t * hc)?,
            post: self.fbuf(t * hc)?,
            comb: self.fbuf(t * hc * hc)?,
        };
        let hin = self.fbuf(t * d)?;
        let sc = self.dw(&format!("{p}.hc_{kind}_scale"))?;
        let base = self.dw(&format!("{p}.hc_{kind}_base"))?;
        let nw = self.dw(norm)?;
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        let (it, eps, neps) = (c.hc_sinkhorn_iters as i32, c.hc_eps, c.norm_eps);
        let (h2, wv, nv, scp, bp) = (h2.map_or(0, |b| b.ptr), w.v, nw.v, sc.v.w, base.v.w);
        let cnt = self.hc_cnt.ptr;
        unsafe {
            self.k.go(
                "ds_hc_fused_ks",
                (gx as u32, t.div_ceil(8) as u32, 1),
                B256,
                0,
                &self.s,
                args![
                    h.ptr, h2, res.ptr, post, comb, xo.ptr, wv, part.ptr, pre.ptr, hin.ptr, nv,
                    scp, bp, o.pre.ptr, o.post.ptr, o.comb.ptr, cnt, ti, hci, di, it, eps, neps,
                    neps, 0i32
                ],
            )?
        };
        Ok((xo, o, hin))
    }

    /// An mHC stage after its mixes GEMM, in one launch: the Sinkhorn of `mixes` (this
    /// stage's coefficients) and `rms(hc_pre(x, pre), norm)` (the block input).
    #[allow(clippy::too_many_arguments)]
    fn hc_stage(
        &self,
        p: &str,
        kind: &str,
        x: Buf,
        (mixes, np): (Buf, i32),
        pre: Buf,
        t: usize,
        norm: &str,
    ) -> Result<(Mix, Buf)> {
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        assert!((2 + hc) * hc <= 80, "ds_hc_stage stages <= 80 mixes");
        let m = Mix {
            pre: self.fbuf(t * hc)?,
            post: self.fbuf(t * hc)?,
            comb: self.fbuf(t * hc * hc)?,
        };
        let out = self.fbuf(t * d)?;
        let sc = self.dw(&format!("{p}.hc_{kind}_scale"))?;
        let base = self.dw(&format!("{p}.hc_{kind}_base"))?;
        let nw = self.dw(norm)?;
        let (hci, it, eps, neps) = (hc as i32, c.hc_sinkhorn_iters as i32, c.hc_eps, c.norm_eps);
        let (scp, bp, nv, di) = (sc.v.w, base.v.w, nw.v, d as i32);
        unsafe {
            self.k.go(
                "ds_hc_stage",
                (t as u32, 1, 1),
                (1024, 1, 1),
                0,
                &self.s,
                args![
                    x.ptr, mixes.ptr, neps, scp, bp, m.pre.ptr, m.post.ptr, m.comb.ptr, hci, it,
                    eps, pre.ptr, out.ptr, nv, di, neps, np
                ],
            )?
        };
        Ok((m, out))
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
        sp: usize,
        pre_mix: Buf,
        attn: Attn,
    ) -> Result<(Buf, Buf)> {
        let d = self.cpu.cfg.dim;
        if std::env::var_os("DS_TAP").is_none() && d % 32 == 0 && self.cpu.cfg.hc_mult <= 8 {
            return self.block_fused(p, lid, x, t, sp, pre_mix, attn);
        }
        let s = self.span("hc");
        let am = self.hc_mixes(p, "attn", x, t)?;
        let h = self.hc_pre(x, pre_mix, t)?;
        self.tap(&format!("{p}.attn_pre"), h, t * d)?;
        self.rms(h, t, &format!("{p}.attn_norm.weight"))?;
        self.tap(&format!("{p}.attn_in"), h, t * d)?;
        self.end(s);
        let s = self.span("attn");
        let h = match attn {
            Attn::Main => self.attention(p, lid, h, t, sp)?,
            Attn::Spark(main_x, mt) => self.dspark_attention(p, lid, h, t, sp, main_x, mt)?,
        };
        self.end(s);
        self.tap(&format!("{p}.attn"), h, t * d)?;
        let s = self.span("hc");
        let x = self.hc_post(h, x, &am, t)?;
        let fm = self.hc_mixes(p, "ffn", x, t)?;
        let h = self.hc_pre(x, am.pre, t)?;
        self.rms(h, t, &format!("{p}.ffn_norm.weight"))?;
        self.end(s);
        self.tap(&format!("{p}.moe_in"), h, t * d)?;
        let s = self.span("moe");
        let h = self.moe(p, h, t, lid)?;
        self.end(s);
        self.tap(&format!("{p}.moe"), h, t * d)?;
        let s = self.span("hc");
        let x = self.hc_post(h, x, &fm, t)?;
        self.end(s);
        Ok((x, fm.pre))
    }

    /// [`GpuModel::block`] with the mHC glue fused: per stage the mixes GEMM (the ffn
    /// stage's with the attention's post-mix) and [`GpuModel::hc_stage`]; the MoE's shared
    /// add folded into the last post-mix. Same values, bit for bit.
    #[allow(clippy::too_many_arguments)]
    fn block_fused(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
        sp: usize,
        pre_mix: Buf,
        attn: Attn,
    ) -> Result<(Buf, Buf)> {
        if !self.hc_exact && !self.k.exact && t <= 8 {
            return self.block_ks(p, lid, x, t, sp, pre_mix, attn);
        }
        let s = self.span("hc");
        let next = self.next_hc.take();
        let mixes = match self.pend_mixes.take() {
            Some(m) => m,
            None => self.hc_post_mix(x, None, x, None, t, p, "attn")?.1,
        };
        let norm = format!("{p}.attn_norm.weight");
        let (am, h) = self.hc_stage(p, "attn", x, mixes, pre_mix, t, &norm)?;
        self.end(s);
        let s = self.span("attn");
        let h = match attn {
            Attn::Main => self.attention(p, lid, h, t, sp)?,
            Attn::Spark(main_x, mt) => self.dspark_attention(p, lid, h, t, sp, main_x, mt)?,
        };
        self.end(s);
        let s = self.span("hc");
        let (x, mixes) = self.hc_post_mix(h, None, x, Some(&am), t, p, "ffn")?;
        let norm = format!("{p}.ffn_norm.weight");
        let (fm, h) = self.hc_stage(p, "ffn", x, mixes, am.pre, t, &norm)?;
        self.end(s);
        let s = self.span("moe");
        self.defer_sh.set(true);
        self.sh.set(None);
        let r = self.moe(p, h, t, lid);
        self.defer_sh.set(false);
        let h = r?;
        let sh = self.sh.take();
        self.end(s);
        let s = self.span("hc");
        let x = match next {
            Some(np) => {
                let (x, m) = self.hc_post_mix(h, sh, x, Some(&fm), t, &np, "attn")?;
                self.pend_mixes = Some(m);
                x
            }
            None => self.hc_post2(h, sh, x, &fm, t)?,
        };
        self.end(s);
        Ok((x, fm.pre))
    }

    /// [`GpuModel::block_fused`] on the K-split path: one launch per mHC stage
    /// ([`GpuModel::hc_fused`]), the last post-mix carrying the next block's attention stage.
    #[allow(clippy::too_many_arguments)]
    fn block_ks(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
        sp: usize,
        pre_mix: Buf,
        attn: Attn,
    ) -> Result<(Buf, Buf)> {
        let next = self.next_hc.take();
        let norm = format!("{p}.attn_norm.weight");
        if !matches!(attn, Attn::Main) || sp == 0 {
            self.glue_flush()?;
        }
        // a span only when there is a launch (the carried stage costs nothing here, and an
        // empty span still charges its event pair, ~5 us, to the op table)
        let (am, h) = match self.pend_stage.take() {
            Some(st) => st,
            None => {
                let s = self.span(hc_name("hc.pend"));
                let (_, am, h) = self.hc_fused(x, None, x, None, t, p, "attn", pre_mix, &norm)?;
                self.end(s);
                (am, h)
            }
        };
        let s = self.span("attn");
        let h = match attn {
            Attn::Main => self.attention(p, lid, h, t, sp)?,
            Attn::Spark(main_x, mt) => self.dspark_attention(p, lid, h, t, sp, main_x, mt)?,
        };
        self.end(s);
        let s = self.span(hc_name("hc.ffn"));
        let norm = format!("{p}.ffn_norm.weight");
        let (x, fm, h) = self.hc_fused(h, None, x, Some(&am), t, p, "ffn", am.pre, &norm)?;
        self.end(s);
        let s = self.span("moe");
        self.defer_sh.set(true);
        self.sh.set(None);
        let r = self.moe(p, h, t, lid);
        self.defer_sh.set(false);
        let h = r?;
        let sh = self.sh.take();
        self.end(s);
        let s = self.span(hc_name("hc.next"));
        let x = match next {
            Some(np) => {
                let norm = format!("{np}.attn_norm.weight");
                let (x, nm, nh) = match self.mega.is_some() {
                    true => self.glue_defer(h, sh, x, &fm, t, &np, fm.pre)?,
                    false => self.hc_fused(h, sh, x, Some(&fm), t, &np, "attn", fm.pre, &norm)?,
                };
                self.pend_stage = Some((nm, nh));
                x
            }
            None => self.hc_post2(h, sh, x, &fm, t)?,
        };
        self.end(s);
        Ok((x, fm.pre))
    }

    // engram

    /// Rows of an engram table, decoded on the host from the mapped checkpoint (fp8 with a
    /// per-row × 32 e8m0 scale): only the rows a step touches are ever read.
    fn engram_rows(&self, name: &str, rows: &[i64]) -> Result<Vec<f32>> {
        let info = self.cpu.st.info(name)?;
        let ehd = info.shape[1];
        let w = self.cpu.st.bytes(name)?;
        let s = self
            .cpu
            .st
            .bytes(&format!("{}scale", name.strip_suffix("weight").unwrap()))?;
        let mut out = Vec::with_capacity(rows.len() * ehd);
        for &r in rows {
            let r = r as usize;
            for i in 0..ehd {
                out.push(e4m3_to_f32(w[r * ehd + i]) * e8m0_to_f32(s[r * (ehd / 32) + i / 32]));
            }
        }
        Ok(out)
    }

    /// Engram's gate over `rows` (the step's staged table rows for this layer).
    fn engram(&self, p: &str, lid: usize, x: Buf, rows: Buf, t: usize) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        self.tap(&format!("{p}.eg_rows"), rows, rows.len / 4)?;
        let kv = self.lin(rows, t, &format!("{p}.engram.wkv.weight"))?;
        self.tap(&format!("{p}.eg_kv"), kv, t * (hc + 1) * d)?;
        let qk = {
            let cached = self.qk.borrow().get(&lid).cloned();
            match cached {
                Some(q) => q,
                None => {
                    let (qw, kw) = (
                        self.cpu.w(&format!("{p}.engram.q_weight"))?,
                        self.cpu.w(&format!("{p}.engram.k_weight"))?,
                    );
                    let v: Vec<f32> = qw.data.iter().zip(&kw.data).map(|(a, b)| a * b).collect();
                    let bq = Rc::new(DevBuf::from_slice(&v)?);
                    self.qk.borrow_mut().insert(lid, bq.clone());
                    bq
                }
            }
        };
        let out = self.fbuf(t * hc * d)?;
        let (hci, di, eps) = (hc as i32, d as i32, c.norm_eps);
        unsafe {
            self.k.go(
                "ds_engram_gate",
                ((t * hc) as u32, 1, 1),
                B256,
                0,
                &self.s,
                args![x.ptr, kv.ptr, qk.ptr, out.ptr, hci, di, eps],
            )?
        };
        Ok(out)
    }

    // the forward

    fn embed_hc(&self, ids: Buf, t: usize, copies: usize) -> Result<Buf> {
        let d = self.cpu.cfg.dim;
        let e = self.dw("embed.weight")?;
        let out = self.fbuf(t * copies * d)?;
        let (ti, ci, dd) = (t as i32, copies as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_embed",
                blocks(t * copies * d, 256),
                B256,
                0,
                &self.s,
                args![e.v.w, ids.ptr, out.ptr, ti, ci, dd],
            )?
        };
        Ok(out)
    }

    fn one_hot(&self, t: usize) -> Vec<f32> {
        let hc = self.cpu.cfg.hc_mult;
        (0..t * hc).map(|i| (i % hc == 0) as u8 as f32).collect()
    }

    /// Stage a step's host inputs: P, the ids, the mHC one-hot and the Engram rows. The
    /// first scratch of a step, so a graph's copies land at the same addresses every time.
    fn inputs(
        &mut self,
        ids: &[usize],
        sp: usize,
        hashes: Option<&[Vec<Vec<i64>>]>,
    ) -> Result<Inputs> {
        let c = self.cpu.cfg.clone();
        let t = ids.len();
        let mut copies = Vec::new();
        let (p, u) = self.stage(&[sp as i32, 0, 0, 0])?;
        copies.push(u);
        self.p = p;
        let (idb, u) = self.stage(&ids.iter().map(|&i| i as i32).collect::<Vec<_>>())?;
        copies.push(u);
        let (one_hot, u) = self.stage(&self.one_hot(t))?;
        copies.push(u);
        let mut engram = Vec::new();
        for (j, &l) in c.engram_layer_ids.iter().enumerate() {
            let hs = hashes.context("hashes")?;
            let flat: Vec<i64> = hs.iter().flat_map(|ht| ht[j].iter().copied()).collect();
            let rows = self.engram_rows(&format!("layers.{l}.engram.embed.weight"), &flat)?;
            let (bb, u) = self.stage(&rows)?;
            copies.push(u);
            engram.push(bb);
        }
        Ok(Inputs {
            ids: idb,
            one_hot,
            engram,
            copies,
        })
    }

    /// As [`Model::forward`]: the last position's logits and the DSpark target hiddens.
    /// `hook` (when given) sees the engram hashes and each block's stream, read back.
    pub fn forward(
        &mut self,
        ids: &[usize],
        sp: usize,
        hook: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, hook, false, true)
            .map(|(l, m, _)| (l, m))
    }

    /// As [`Model::forward_all`]: every position's logits [T][V] (a verify window).
    pub fn forward_all(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, None, true, true)
            .map(|(l, m, _)| (l, m))
    }

    /// A verify window that leaves its outputs on the device: only each row's argmax comes
    /// back (a few bytes instead of [T][V] f32 and the hiddens); [`Self::logits_row`] and
    /// [`Self::window_mains`] read parts of this window on demand.
    pub fn verify_window(&mut self, ids: &[usize], sp: usize) -> Result<Vec<usize>> {
        self.forward_impl(ids, sp, None, true, false)
            .map(|(_, _, a)| a)
    }

    /// Row `r` of the last [`Self::verify_window`]'s logits.
    pub fn logits_row(&self, r: usize) -> Result<Vec<f32>> {
        let v = self.cpu.cfg.vocab_size;
        let out = self.last.as_ref().context("logits_row: no verify window")?;
        if r >= out.rows {
            bail!("logits_row: row {r} of {}", out.rows);
        }
        super::obs::d2h(v * 4);
        Ok(out.logits.dev().read::<f32>(r * v * 4, v)?)
    }

    /// Whether a step at `sp` with these settings is captured (see the module doc).
    fn graphable(&self, sp: usize, hook: bool) -> bool {
        self.graph
            && sp > 0
            && !hook
            && self.force.is_none()
            && self.expert_slots == 0
            && (self.moe_reduce.is_none() || self.reduce_graphable)
            && std::env::var_os("DS_TAP").is_none()
            && !self.obs.direct()
    }

    fn forward_impl(
        &mut self,
        ids: &[usize],
        sp: usize,
        mut hook: Option<&mut dyn FnMut(&str, &[f32])>,
        all: bool,
        full: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>)> {
        let c = self.cpu.cfg.clone();
        let (d, t, v) = (c.dim, ids.len(), c.vocab_size);
        self.last = None;
        self.begin_step()?;
        let kind = match (sp, all && t > 1) {
            (0, _) => "prefill",
            (_, true) => "verify",
            _ => "decode",
        };
        self.obs.begin(&self.s, kind, t, sp);
        let hashes = if c.engram_layer_ids.is_empty() {
            None
        } else {
            Some(self.cpu.engram_hash(ids, sp))
        };
        if let (Some(hs), Some(h)) = (&hashes, hook.as_mut()) {
            h(
                "hash",
                &hs.iter()
                    .flatten()
                    .flatten()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>(),
            );
        }
        let inp = self.inputs(ids, sp, hashes.as_deref())?;
        // host code of a decode step never sees the position (see `DYN`)
        let bsp = if sp == 0 { 0 } else { DYN };
        let key = (t, all);
        let gen = self.gen();
        if std::env::var_os("DS_GRAPH_DEBUG").is_some() {
            eprintln!("step t={t} sp={sp} gen={gen:?} {:?}", self.graph_stats);
        }
        let out = if self.graphable(sp, hook.is_some()) {
            if let Some(cg) = self.graphs.get(&key) {
                self.obs.path(true, false);
                super::obs::replay();
                super::obs::h2d(inp.copies.iter().map(|u| u.n).sum());
                cg.g.launch(&self.s)?;
                self.graph_stats.replays += 1;
                cg.out.clone()
            } else if self.warm.get(&key) == Some(&gen) {
                // warmed up at this generation: every weight is resident and the arena fits
                // the step, so the capture allocates nothing
                let s = ManuallyDrop::new(Stream(self.s.0));
                // no timeline events inside the capture
                self.obs.path(true, true);
                super::obs::capture();
                self.arena.borrow_mut().frozen = true;
                let mut out = None;
                let g = Graph::capture(&s, |s| {
                    let r = (|| -> Result<StepOut> {
                        for u in &inp.copies {
                            u.enqueue(s)?;
                        }
                        self.body(&inp, t, bsp, all, None)
                    })();
                    let ok = r.is_ok();
                    out = Some(r);
                    if ok {
                        Ok(())
                    } else {
                        Err(tang_moe::gpu::Error("capture body failed".into()))
                    }
                });
                self.arena.borrow_mut().frozen = false;
                let out = out.context("capture")?.context("captured step")?;
                let g = g?;
                g.launch(&self.s)?;
                self.graph_stats.captures += 1;
                self.graphs.insert(
                    key,
                    Captured {
                        g,
                        out: out.clone(),
                        gen,
                    },
                );
                out
            } else {
                self.obs.path(false, false);
                let out = self.direct(&inp, t, bsp, all, None)?;
                // a step that grew the arena moves it at the next start: warm up again
                self.warm.insert(key, self.gen_after());
                out
            }
        } else {
            self.obs.path(false, false);
            self.direct(&inp, t, bsp, all, hook)?
        };
        self.obs.gpu_end(&self.s, self.arena.borrow().capacity());
        if self.journal_on && sp > 0 {
            self.journal_steps += 1;
            self.journal = out
                .jrows
                .iter()
                .map(|&(l, e, kv, sc)| (l, sp + e, kv, sc))
                .collect();
        }
        if self.skip_readback {
            // its events complete by the next step's start, which emits it
            self.obs.finish();
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        self.sync()?;
        let (logits, amax) = if full {
            super::obs::d2h((out.rows * v + out.mains.len() * t * d) * 4);
            (out.logits.dev().read::<f32>(0, out.rows * v)?, Vec::new())
        } else {
            super::obs::d2h(out.rows * 4);
            let a = out.amax.dev().read::<i32>(0, out.rows)?;
            (Vec::new(), a.into_iter().map(|i| i as usize).collect())
        };
        let main = if full {
            Self::read_mains(&out, t, d)?
        } else {
            self.last = Some(out);
            Vec::new()
        };
        self.obs.finish();
        Ok((logits, main, amax))
    }

    /// The first `rows` positions' DSpark target hiddens of `out`, [rows][targets][d].
    fn read_mains(out: &StepOut, rows: usize, d: usize) -> Result<Vec<f32>> {
        let n = out.mains.len();
        let mut main = vec![0f32; rows * d * n];
        for (j, m) in out.mains.iter().enumerate() {
            let mv = m.dev().read::<f32>(0, rows * d)?;
            for ti in 0..rows {
                main[(ti * n + j) * d..(ti * n + j + 1) * d]
                    .copy_from_slice(&mv[ti * d..(ti + 1) * d]);
            }
        }
        Ok(main)
    }

    /// The first `rows` positions' DSpark target hiddens of the last [`Self::verify_window`]
    /// (the accepted prefix: the rest are never read back).
    pub fn window_mains(&self, rows: usize) -> Result<Vec<f32>> {
        let out = self
            .last
            .as_ref()
            .context("window_mains: no verify window")?;
        let d = self.cpu.cfg.dim;
        super::obs::d2h(rows * d * out.mains.len() * 4);
        Self::read_mains(out, rows, d)
    }

    /// The generation the next step will start at (one more when this one overflowed).
    fn gen_after(&self) -> (u64, u64) {
        let (a, s) = (self.arena.borrow(), self.stage.borrow());
        (a.gen + a.overflowed() as u64, s.gen + s.overflowed() as u64)
    }

    fn direct(
        &mut self,
        inp: &Inputs,
        t: usize,
        sp: usize,
        all: bool,
        hook: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<StepOut> {
        for u in &inp.copies {
            u.enqueue(&self.s)?;
        }
        self.graph_stats.direct += 1;
        self.body(inp, t, sp, all, hook)
    }

    /// The step on the device, from staged inputs: no host sync, no host-side position when
    /// `sp` is [`DYN`].
    fn body(
        &mut self,
        inp: &Inputs,
        t: usize,
        sp: usize,
        all: bool,
        mut hook: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<StepOut> {
        let c = self.cpu.cfg.clone();
        let (hc, d) = (c.hc_mult, c.dim);
        self.jrows.clear();
        let sp_embed = self.span("embed");
        let mut h = self.embed_hc(inp.ids, t, hc)?;
        self.end(sp_embed);
        let mut pre_mix = inp.one_hot;
        let mut mains: Vec<Buf> = Vec::new();
        self.pend_mixes = None;
        self.pend_stage = None;
        self.pend_glue = None;
        for i in 0..c.n_layers {
            let p = format!("layers.{i}");
            let sp_layer = self.obs.span(&self.s, "layer", Some(i));
            if let Some(f) = &self.force {
                let (a, bb) = (f[i].0.clone(), f[i].1.clone());
                h = self.up(&a)?;
                pre_mix = self.up(&bb)?;
            }
            if let Some(j) = c.engram_layer_ids.iter().position(|&l| l == i) {
                let se = self.span("engram");
                h = self.engram(&p, i, h, inp.engram[j], t)?;
                self.end(se);
            }
            if c.dspark_target_layer_ids.contains(&i) {
                self.glue_flush()?;
                let sh = self.span("hc");
                let m = self.fbuf(t * d)?;
                let (ti, hci, di) = (t as i32, hc as i32, d as i32);
                unsafe {
                    self.k.go(
                        "ds_mean_hc",
                        blocks(t * d, 256),
                        B256,
                        0,
                        &self.s,
                        args![h.ptr, m.ptr, ti, hci, di],
                    )?
                };
                mains.push(m);
                self.end(sh);
            }
            let j = i + 1;
            self.next_hc =
                (j < c.n_layers && self.force.is_none() && !c.engram_layer_ids.contains(&j))
                    .then(|| format!("layers.{j}"));
            let (nh, npm) = self.block(&p, i, h, t, sp, pre_mix, Attn::Main)?;
            self.next_hc = None;
            h = nh;
            pre_mix = npm;
            self.end(sp_layer);
            if let Some(hk) = hook.as_mut() {
                self.glue_flush()?;
                hk(&format!("layer{i}"), &self.read(h, t * hc * d)?);
            }
        }
        let sp_head = self.span("head");
        let x = self.hc_pre(h, pre_mix, t)?;
        self.rms(x, t, "norm.weight")?;
        let rows = if all { t } else { 1 };
        let hw = self.dw("head.weight")?;
        let logits = self.fbuf(rows * hw.v.rows as usize)?;
        let xh = x.ptr + ((t - rows) * d * 4) as u64;
        self.k.head(&self.s, &hw.v, xh, logits.ptr, rows)?;
        let amax = self.fbuf(rows)?;
        let vi = hw.v.rows;
        unsafe {
            self.k.go(
                "ds_argmax_rows",
                (rows as u32, 1, 1),
                (1024, 1, 1),
                0,
                &self.s,
                args![logits.ptr, vi, amax.ptr],
            )?
        };
        self.end(sp_head);
        Ok(StepOut {
            logits,
            rows,
            amax,
            mains,
            jrows: std::mem::take(&mut self.jrows),
        })
    }

    // dspark

    #[allow(clippy::too_many_arguments)]
    /// DSpark attention over the ring and the block. `sp` is 0 (seeding: the main rows into
    /// the ring only) or [`DYN`]: the position is `P[0]` on the device, and the index rows are
    /// launched for the ring's full width (`-1` past the rows written), so a draft captures.
    #[allow(clippy::too_many_arguments)]
    fn dspark_attention(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        bl: usize,
        sp: usize,
        main_x: Buf,
        t: usize,
    ) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (h, hd, win) = (c.n_heads, c.head_dim, c.window_size);
        let dev = |add| At::Dev {
            add,
            stride: 1,
            fixed_r: 0,
        };
        let mkv = self.lin(main_x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(mkv, t, &format!("{p}.attn.kv_norm.weight"))?;
        if sp == 0 {
            let at0 = At::Host {
                pos0: 0,
                stride: 1,
                fixed: None,
            };
            self.rope(mkv, t, 1, hd, lid, at0, false)?;
            self.k.fp8_qdq(&self.s, mkv.ptr, t * hd)?;
            self.ring_seed(lid, mkv, t)?;
            return self.fbuf(1);
        }
        anyhow::ensure!(sp == DYN, "dspark_attention: sp > 0 reads P[0]");
        self.rope(mkv, t, 1, hd, lid, dev(0), false)?;
        self.k.fp8_qdq(&self.s, mkv.ptr, t * hd)?;
        // the block's rows sit at P[0] + t ..
        let qr = self.lin(x, bl, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(qr, bl, &format!("{p}.attn.q_norm.weight"))?;
        let q = self.lin(qr, bl, &format!("{p}.attn.wq_b.weight"))?;
        // q's rope is attn_out's
        let kv = self.lin(x, bl, &format!("{p}.attn.wkv.weight"))?;
        self.rms(kv, bl, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(kv, bl, 1, hd, lid, dev(t), false)?;
        self.k.fp8_qdq(&self.s, kv.ptr, bl * hd)?;
        // the ring holds positions < P[0] + t once the t main rows are written: its
        // min(win, P[0] + t) rows, then the block's, then -1 holes (as ds_win_idx mode 2)
        let nk = win + bl;
        let idx = self.fbuf(bl * nk)?;
        let (bi, ni, pp, ai, wi) = (bl as i32, nk as i32, self.p.ptr, t as i32 - 1, win as i32);
        unsafe {
            self.gk.go(
                "dg_spark_idx",
                blocks(bl * nk, 256),
                B256,
                0,
                &self.s,
                args![idx.ptr, bi, ni, pp, ai, wi],
            )?
        };
        let ring = b(&self.win[lid]);
        self.ring_write(ring, mkv, t, hd, win, 0)?;
        let all = self.fbuf((win + bl) * hd)?;
        self.copy(all.ptr, ring.ptr, win * hd)?;
        self.copy(all.ptr + (win * hd * 4) as u64, kv.ptr, bl * hd)?;
        self.attn_out(
            p,
            q,
            false,
            (all, win + bl, 0),
            Ix::Rows(idx),
            bl,
            nk,
            lid,
            dev(t),
        )
    }

    /// As [`Model::forward_spec`]: greedy drafts, drawn on the device.
    pub fn forward_spec(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
    ) -> Result<Option<super::forward::Draft>> {
        self.spark(token, main_hidden, sp, Draw::Greedy)
    }

    /// As [`Model::forward_spec_with`]: each draft is `draw(logits)` (Markov bias included),
    /// sampled on the host.
    pub fn forward_spec_with(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
        draw: &mut dyn FnMut(&[f32]) -> usize,
    ) -> Result<Option<super::forward::Draft>> {
        self.spark(token, main_hidden, sp, Draw::Host(draw))
    }

    /// The DSpark draft. On the device (greedy: `ds_spark_pick`, argmax; a [`SpecSampler`]:
    /// `ds_spark_sample`, its draw from the shared RNG state): each position's Markov bias
    /// added and its draft picked, then the next position's Markov embedding row; the
    /// drafts, logits, hiddens (and the advanced RNG) come back in one read at the end, and
    /// the whole draft is a CUDA graph per (main rows, sampler) when decode steps are (see
    /// the module doc). With [`Draw::Host`] the host samples, one row read back per position.
    fn spark(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
        mut how: Draw,
    ) -> Result<Option<super::forward::Draft>> {
        let c = self.cpu.cfg.clone();
        let (d, blk, v) = (c.dim, c.dspark_block_size, c.vocab_size);
        self.begin_step()?;
        self.obs.begin(&self.s, "draft", blk, sp);
        let t = main_hidden.len() / (d * c.dspark_target_layer_ids.len());
        let last = format!("mtp.{}", c.n_mtp_layers - 1);
        let rank = c.dspark_markov_rank;
        let memb_w = self.cpu.w(&format!("{last}.markov_head.embed.weight"))?;
        if self.spark_tab.is_none() {
            self.spark_tab = Some(DevBuf::from_slice(&memb_w.data)?);
        }
        // the inputs, staged in a fixed order (a captured draft's uploads read these slots)
        let mut copies = Vec::new();
        let (p, u) = self.stage(&[sp as i32, 0, 0, 0])?;
        copies.push(u);
        self.p = p;
        let (mh, u) = self.stage(main_hidden)?;
        copies.push(u);
        let mut ids = vec![c.dspark_noise_token_id as i32; blk];
        ids[0] = token as i32;
        let (idb, u) = self.stage(&ids)?;
        copies.push(u);
        let (oh, u) = self.stage(&self.one_hot(blk))?;
        copies.push(u);
        let rng0 = match &how {
            Draw::Sample(sm) => sm.rng,
            _ => 0,
        };
        let (rb, u) = self.stage(&[rng0])?;
        copies.push(u);
        let ins = SparkIn { mh, idb, oh, rb, t };
        if sp == 0 {
            // seeding: the main rows into the MTP layers' rings, nothing drafted
            for u in &copies {
                u.enqueue(&self.s)?;
            }
            let main_x = self.lin(mh, t, "mtp.0.main_proj.weight")?;
            self.rms(main_x, t, "mtp.0.main_norm.weight")?;
            let h = self.embed_hc(idb, blk, c.hc_mult)?;
            for s in 0..c.n_mtp_layers {
                let (lid, p) = (c.n_layers + s, format!("mtp.{s}"));
                self.dspark_attention(&p, lid, h, blk, 0, main_x, t)?;
            }
            self.obs.gpu_end(&self.s, self.arena.borrow().capacity());
            self.obs.finish();
            return Ok(None);
        }
        let proj = self.cpu.w(&format!("{last}.confidence_head.proj.weight"))?;
        let conf_of = |xh: &[f32], out: &[usize]| -> Vec<f32> {
            (0..blk)
                .map(|i| {
                    let memb = &memb_w.data[out[i] * rank..(out[i] + 1) * rank];
                    let feat: Vec<f32> =
                        xh[i * d..(i + 1) * d].iter().chain(memb).copied().collect();
                    super::forward::matmul(&feat, 1, &proj)[0]
                })
                .collect()
        };
        self.spec_q = false;
        if let Draw::Host(draw) = &mut how {
            // the Markov head samples left to right: each position's bias (its predecessor's
            // Markov embedding through the head, a rank x V GEMV) on the device, the row read
            // back for the draw; the confidence head (a dot product) on the host
            self.obs.path(false, false);
            for u in &copies {
                u.enqueue(&self.s)?;
            }
            let x = self.spark_trunk(&ins)?;
            let xh = self.read(x, blk * d)?;
            self.rms(x, blk, &format!("{last}.norm.weight"))?;
            let dlogits = self.raw(x.ptr, blk, "head.weight")?;
            let mhead = format!("{last}.markov_head.head.weight");
            let mut logits = vec![0f32; blk * v];
            let mut out = vec![token];
            for i in 0..blk {
                let memb = &memb_w.data[out[i] * rank..(out[i] + 1) * rank];
                let mx = self.up(memb)?;
                let bias = self.raw(mx.ptr, 1, &mhead)?;
                let row = dlogits.at(i * v * 4);
                let vn = v as i64;
                unsafe {
                    self.k.go(
                        "ds_add",
                        blocks(v, 256),
                        B256,
                        0,
                        &self.s,
                        args![row.ptr, bias.ptr, vn],
                    )?
                };
                let li = &mut logits[i * v..(i + 1) * v];
                li.copy_from_slice(&self.read(row, v)?);
                out.push(draw(li));
            }
            let conf = conf_of(&xh, &out);
            self.obs.gpu_end(&self.s, self.arena.borrow().capacity());
            self.obs.finish();
            return Ok(Some(super::forward::Draft {
                ids: out,
                logits,
                conf,
            }));
        }
        let samp = match &how {
            Draw::Sample(sm) => Some((sm.temp, sm.top_p)),
            _ => None,
        };
        let key = (t, samp.map(|(a, b)| (a.to_bits(), b.to_bits())));
        if samp.is_some() && self.spec_dev.is_none() {
            self.spec_dev = Some(SpecDev::new(blk, v)?);
        }
        self.spec_q = samp.is_some();
        let gen = self.gen();
        let lay = SparkPack::new(blk, v, d);
        let pk = if self.graphable(sp, false) {
            if let Some(cg) = self.spark_graphs.get(&key) {
                self.obs.path(true, false);
                super::obs::replay();
                super::obs::h2d(copies.iter().map(|u| u.n).sum());
                cg.g.launch(&self.s)?;
                self.graph_stats.replays += 1;
                cg.pk
            } else if self.spark_warm.get(&key) == Some(&gen) {
                let s = ManuallyDrop::new(Stream(self.s.0));
                self.obs.path(true, true);
                super::obs::capture();
                self.arena.borrow_mut().frozen = true;
                let mut out = None;
                let g = Graph::capture(&s, |s| {
                    let r = (|| -> Result<Buf> {
                        for u in &copies {
                            u.enqueue(s)?;
                        }
                        self.spark_device(&ins, samp)
                    })();
                    let ok = r.is_ok();
                    out = Some(r);
                    if ok {
                        Ok(())
                    } else {
                        Err(tang_moe::gpu::Error("draft capture body failed".into()))
                    }
                });
                self.arena.borrow_mut().frozen = false;
                let pk = out.context("capture")?.context("captured draft")?;
                let g = g?;
                g.launch(&self.s)?;
                self.graph_stats.captures += 1;
                self.spark_graphs.insert(key, SparkGraph { g, pk, gen });
                pk
            } else {
                self.obs.path(false, false);
                for u in &copies {
                    u.enqueue(&self.s)?;
                }
                let pk = self.spark_device(&ins, samp)?;
                self.spark_warm.insert(key, self.gen_after());
                pk
            }
        } else {
            self.obs.path(false, false);
            for u in &copies {
                u.enqueue(&self.s)?;
            }
            self.spark_device(&ins, samp)?
        };
        self.obs.gpu_end(&self.s, self.arena.borrow().capacity());
        let all = self.read(pk, lay.len)?;
        let out: Vec<usize> = all[lay.ids..lay.ids + blk + 1]
            .iter()
            .map(|f| f.to_bits() as i32 as usize)
            .collect();
        if let Draw::Sample(sm) = &mut how {
            sm.rng = all[lay.rng].to_bits() as u64 | (all[lay.rng + 1].to_bits() as u64) << 32;
        }
        let conf = conf_of(&all[lay.hid..lay.ids], &out);
        let mut logits = all;
        logits.truncate(blk * v);
        self.obs.finish();
        Ok(Some(super::forward::Draft {
            ids: out,
            logits,
            conf,
        }))
    }

    /// The draft's MTP layers over the staged inputs (position from `P[0]`): the last
    /// layer's pre-norm hiddens [blk][D].
    fn spark_trunk(&mut self, ins: &SparkIn) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (blk, t) = (c.dspark_block_size, ins.t);
        let main_x = self.lin(ins.mh, t, "mtp.0.main_proj.weight")?;
        self.rms(main_x, t, "mtp.0.main_norm.weight")?;
        let mut h = self.embed_hc(ins.idb, blk, c.hc_mult)?;
        let mut pre_mix = ins.oh;
        for s in 0..c.n_mtp_layers {
            let (lid, p) = (c.n_layers + s, format!("mtp.{s}"));
            let (nh, npm) = self.block(&p, lid, h, blk, DYN, pre_mix, Attn::Spark(main_x, t))?;
            h = nh;
            pre_mix = npm;
        }
        self.hc_pre(h, pre_mix, blk)
    }

    /// The device draft after the uploads: the trunk, the head into the pack, and the
    /// left-to-right picks (greedy, or `samp` = (temperature, top-p) draws). Host code
    /// independent of the position and the inputs' values, so it captures.
    fn spark_device(&mut self, ins: &SparkIn, samp: Option<(f32, f32)>) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (d, blk, v, rank) = (
            c.dim,
            c.dspark_block_size,
            c.vocab_size,
            c.dspark_markov_rank,
        );
        let last = format!("mtp.{}", c.n_mtp_layers - 1);
        let mhead = format!("{last}.markov_head.head.weight");
        let x = self.spark_trunk(ins)?;
        let lay = SparkPack::new(blk, v, d);
        let pk = self.fbuf(lay.len)?;
        let at = |o: usize| pk.ptr + (o * 4) as u64;
        self.copy(at(lay.hid), x.ptr, blk * d)?;
        self.rms(x, blk, &format!("{last}.norm.weight"))?;
        // the head writes the logits straight into the pack
        let hw = self.dw("head.weight")?;
        self.k.linear(&self.s, &hw.v, x.ptr, pk.ptr, blk, false)?;
        let tab = b(self.spark_tab.as_ref().context("spark_tab")?).ptr;
        let idp = at(lay.ids);
        let memb = self.fbuf(rank)?;
        let (vi, ri) = (v as i32, rank as i32);
        let null = 0u64;
        // slot 0 = the token (from the staged ids), and its Markov embedding row
        let s0 = 0i32;
        unsafe {
            self.k.go(
                "ds_spark_pick",
                (1, 1, 1),
                (1024, 1, 1),
                0,
                &self.s,
                args![null, vi, idp, s0, tab, ri, memb.ptr, null, ins.idb.ptr],
            )?
        };
        let sampled = match samp {
            Some((temp, top_p)) => {
                let rng = at(lay.rng);
                self.copy(rng, ins.rb.ptr, 2)?;
                Some((rng, self.fbuf(2 * v)?, temp, top_p))
            }
            None => None,
        };
        let (qi, qp, qn) = self
            .spec_dev
            .as_ref()
            .map_or((0u64, 0u64, 0u64), |sd| (sd.qi.ptr, sd.qp.ptr, sd.qn.ptr));
        for i in 0..blk {
            let bias = self.raw(memb.ptr, 1, &mhead)?;
            let row = pk.ptr + (i * v * 4) as u64;
            let slot = i as i32 + 1;
            unsafe {
                match sampled {
                    None => self.k.go(
                        "ds_spark_pick",
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        &self.s,
                        args![row, vi, idp, slot, tab, ri, memb.ptr, bias.ptr, null],
                    )?,
                    Some((rng, w, temp, top_p)) => self.k.go(
                        "ds_spark_sample",
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        &self.s,
                        args![
                            row, vi, idp, slot, tab, ri, memb.ptr, temp, top_p, rng, w.ptr,
                            bias.ptr, qi, qp, qn
                        ],
                    )?,
                }
            };
        }
        Ok(pk)
    }

    // speculative decoding

    /// As [`Model::snapshot`], into the model's one snapshot store.
    pub fn snapshot(&mut self) -> Result<GpuSnapshot> {
        let c = self.cpu.cfg.clone();
        let hd = c.head_dim;
        if self.snap.is_none() {
            let z = |n: usize| -> Result<DevBuf> { Ok(DevBuf::alloc(n * 4)?) };
            let win = (0..c.n_layers)
                .map(|_| z(c.window_size * hd))
                .collect::<Result<_>>()?;
            let mut cst = HashMap::new();
            for &l in self.cst.keys() {
                let n = c.compress_ratios[l] * hd;
                cst.insert(l, (z(n)?, z(n)?));
            }
            self.snap = Some((win, cst));
        }
        let (sw, sc) = self.snap.as_ref().unwrap();
        let (sw, sc): (Vec<Buf>, HashMap<usize, (Buf, Buf)>) = (
            sw.iter().map(b).collect(),
            sc.iter().map(|(&l, (a, bb))| (l, (b(a), b(bb)))).collect(),
        );
        for (l, dst) in sw.iter().enumerate() {
            self.copy(dst.ptr, self.win[l].ptr, c.window_size * hd)?;
        }
        for (l, (a, bb)) in &sc {
            let n = c.compress_ratios[*l] * hd;
            self.copy(a.ptr, self.cst[l].0.ptr, n)?;
            self.copy(bb.ptr, self.cst[l].1.ptr, n)?;
        }
        self.journal_on = true;
        self.journal_steps = 0;
        self.journal.clear();
        Ok(GpuSnapshot { win: sw, cst: sc })
    }

    /// As [`Model::rewind`]: back to only positions `< from` fed. At most one forward may
    /// run under the snapshot (the journal lives in that step's scratch).
    pub fn rewind(&mut self, snap: &GpuSnapshot, from: usize, end: usize) -> Result<()> {
        if self.journal_steps > 1 {
            bail!("rewind: {} forwards under one snapshot", self.journal_steps);
        }
        let c = self.cpu.cfg.clone();
        let (win, hd) = (c.window_size, c.head_dim);
        for (l, old) in snap.win.iter().enumerate() {
            for pos in from..end.min(from + win) {
                let o = ((pos % win) * hd * 4) as u64;
                self.copy(self.win[l].ptr + o, old.ptr + o, hd)?;
            }
        }
        for (l, (a, bb)) in &snap.cst {
            let n = c.compress_ratios[*l] * hd;
            let (ka, kb) = (self.cst[l].0.ptr, self.cst[l].1.ptr);
            self.copy(ka, a.ptr, n)?;
            self.copy(kb, bb.ptr, n)?;
        }
        let journal = std::mem::take(&mut self.journal);
        for (lid, pos, kv, sc) in &journal {
            if *pos < from {
                let o = ((pos % c.compress_ratios[*lid]) * hd * 4) as u64;
                let (ka, kb) = (self.cst[lid].0.ptr, self.cst[lid].1.ptr);
                self.copy(ka + o, kv.ptr, hd)?;
                self.copy(kb + o, sc.ptr, hd)?;
            }
        }
        self.journal_on = false;
        Ok(())
    }

    /// Speculative sampling's accept walk over the last verify window on the device
    /// (`ds_spec_dist` per row, then `ds_spec_walk` against the draft's kept Q rows): the
    /// accepted count, the tokens and the advanced RNG, from one small read-back; `None` when
    /// a row the walk reached has no device distribution (the host walks instead).
    fn spec_walk(
        &mut self,
        ids: &[usize],
        temp: f32,
        top_p: f32,
        rng: u64,
    ) -> Result<Option<(usize, Vec<usize>, u64)>> {
        let v = self.cpu.cfg.vocab_size;
        let n = ids.len();
        let out = self.last.as_ref().context("spec_walk: no verify window")?;
        if out.rows != n {
            bail!("spec_walk: {} rows for {n} ids", out.rows);
        }
        let lp = out.logits.ptr;
        let sd = self.spec_dev.as_ref().context("spec_walk: no draft")?;
        sd.ids
            .write(0, &ids.iter().map(|&i| i as i32).collect::<Vec<_>>())?;
        let (vi, ni) = (v as i32, n as i32);
        unsafe {
            self.k.go(
                "ds_spec_dist",
                (n as u32, 1, 1),
                (1024, 1, 1),
                0,
                &self.s,
                args![lp, vi, temp, top_p, sd.w.ptr, sd.pi.ptr, sd.pp.ptr, sd.pn.ptr],
            )?;
            self.k.go(
                "ds_spec_walk",
                (1, 1, 1),
                (1024, 1, 1),
                0,
                &self.s,
                args![
                    sd.ids.ptr, ni, sd.pi.ptr, sd.pp.ptr, sd.pn.ptr, sd.qi.ptr, sd.qp.ptr,
                    sd.qn.ptr, rng, sd.out.ptr
                ],
            )?;
        }
        self.sync()?;
        super::obs::d2h((n + 4) * 4);
        let o = sd.out.read::<i32>(0, n + 4)?;
        if o[0] != 1 {
            return Ok(None);
        }
        let k = o[1] as usize;
        let tokens = o[2..3 + k].iter().map(|&t| t as usize).collect();
        let rng = o[2 + n] as u32 as u64 | (o[3 + n] as u32 as u64) << 32;
        Ok(Some((k, tokens, rng)))
    }

    /// As [`Model::decode_spec`].
    pub fn decode_spec(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        edit: &mut dyn FnMut(usize, &mut [usize]),
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep> {
        let c = self.cpu.cfg.clone();
        let (v, w) = (c.vocab_size, c.dim * c.dspark_target_layer_ids.len());
        let pos0 = sp + main.len() / w;
        let (start, t0) = (std::time::SystemTime::now(), std::time::Instant::now());
        // the drafter's block for this mode (the spark reads it from the config)
        let greedy = pick.greedy();
        self.cpu.cfg.dspark_block_size = c.draft_block(greedy);
        let draft = if pick.greedy_draft() {
            self.spark(token, main, sp, Draw::Greedy)
        } else if let Some(sm) = pick.as_sampler() {
            self.spark(token, main, sp, Draw::Sample(sm))
        } else {
            self.spark(token, main, sp, Draw::Host(&mut |q: &[f32]| pick.draft(q)))
        };
        self.cpu.cfg.dspark_block_size = c.dspark_block_size;
        let draft = draft?.context("decode_spec needs sp > 0")?;
        let mut ids = draft.ids;
        ids.truncate(1 + c.draft_keep(&draft.conf, greedy));
        edit(pos0 + 1, &mut ids[1..]);
        let t1 = std::time::Instant::now();
        let snap = self.snapshot()?;
        let amax = self.verify_window(&ids, pos0)?;
        let n = ids.len();
        let dev = match pick.as_sampler() {
            Some(sm) if !SpecSampler::greedy(sm) && self.spec_q && !host_walk() => {
                let r = self.spec_walk(&ids, sm.temp, sm.top_p, sm.rng)?;
                if let Some((_, _, rng)) = &r {
                    sm.rng = *rng;
                }
                r
            }
            _ => None,
        };
        let (k, tokens) = if pick.greedy() {
            super::forward::verify_greedy(&amax, &ids)
        } else if let Some((k, tokens, _)) = dev {
            self.spec_walks.0 += 1;
            (k, tokens)
        } else {
            self.spec_walks.1 += 1;
            // only the rows the walk reaches are read back
            let mut err = None;
            let mut row = |r: usize| {
                self.logits_row(r).unwrap_or_else(|e| {
                    err.get_or_insert(e);
                    vec![0.0; v]
                })
            };
            let r = super::forward::verify_rows_with(&mut row, &draft.logits, &ids, v, pick);
            if let Some(e) = err {
                return Err(e);
            }
            r
        };
        let main = self.window_mains(k + 1)?;
        let t2 = std::time::Instant::now();
        self.rewind(&snap, pos0 + k + 1, pos0 + n)?;
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        if super::obs::level() > 0 || crate::telemetry::ds_enabled() {
            super::obs::spec(super::obs::SpecRec {
                drafted: n - 1,
                accepted: k,
                draft_ms: ms(t0, t1),
                verify_ms: ms(t1, t2),
                rewind_ms: ms(t2, std::time::Instant::now()),
                rank: self.obs.rank.get(),
                start,
            });
        }
        Ok(SpecStep {
            tokens,
            drafted: n - 1,
            accepted: k,
            main,
            sp: pos0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::tiny;
    use super::*;

    fn load_ref(name: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(format!("{}/ref/{name}.bin", tiny())).ok()?;
        Some(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let m = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / m
    }

    fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
        let ls = |l: &[f32]| {
            let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
            l.iter().map(|&x| x as f64 - z).collect::<Vec<f64>>()
        };
        let (p, q) = (ls(p_logits), ls(q_logits));
        p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum()
    }

    fn argmax(v: &[f32]) -> usize {
        (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
    }

    fn open() -> Option<(GpuModel, serde_json::Value)> {
        Gpu::for_test()?;
        let dir = Path::new(&tiny()).join("hf");
        if !Path::new(&tiny()).join("ref/step0_in0.bin").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return None;
        }
        let meta = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).unwrap(),
        )
        .unwrap();
        Some((GpuModel::open(&dir).unwrap(), meta))
    }

    /// Step 0 intermediates, device against the CPU reference on the same forced input.
    /// The launch inventory of a verify window (t = 6) on the direct path: the fixture's
    /// steps up to it, then that step between cuProfilerStart/Stop, for
    /// `nsys profile --capture-range=cudaProfilerApi` (`DS_MEGA=1` for the megakernel).
    #[test]
    #[ignore]
    fn gpu_tiny_inventory() {
        std::env::set_var("DS_GRAPH", "0");
        let Some((mut m, meta)) = open() else { return };
        for step in meta["steps"].as_array().unwrap() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let mark = ids.len() == 6;
            if mark {
                m.s.sync().unwrap();
                unsafe { cudarc::driver::sys::cuProfilerStart() };
            }
            m.forward(&ids, sp, None).unwrap();
            if mark {
                m.s.sync().unwrap();
                unsafe { cudarc::driver::sys::cuProfilerStop() };
            }
        }
    }

    /// `DS_MEGA=1`'s dense-glue megakernel against the separate launches: the same logits,
    /// bit for bit, over the fixture's steps (prefill, then decode and verify windows).
    #[test]
    fn gpu_tiny_mega_matches() {
        let Some((mut a, meta)) = open() else { return };
        if a.k.exact || a.hc_exact {
            eprintln!("skip: exact kernels");
            return;
        }
        let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
        a.mega = Some(super::super::mega::Mega::new(&Gpu::at(0).unwrap()).unwrap());
        let n0 = super::super::mega::LAUNCHES.load(std::sync::atomic::Ordering::Relaxed);
        for (s, step) in meta["steps"].as_array().unwrap().iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            for rep in 0..2 {
                let (la, _) = a.forward(&ids, sp, None).unwrap();
                let (lb, _) = b.forward(&ids, sp, None).unwrap();
                let diff = la.iter().zip(&lb).filter(|(x, y)| x != y).count();
                eprintln!(
                    "step {s} (t {}, sp {sp}) rep {rep}: {diff} logits differ",
                    ids.len()
                );
                assert_eq!(la, lb, "step {s} rep {rep}");
            }
        }
        let n = super::super::mega::LAUNCHES.load(std::sync::atomic::Ordering::Relaxed) - n0;
        eprintln!("{n} mg_glue launches");
        assert!(n > 0, "the megakernel never ran");
    }

    #[test]
    #[ignore]
    fn gpu_tiny_taps() {
        let Some((mut m, meta)) = open() else { return };
        std::env::set_var("DS_TAP", "1");
        let n_layers = m.cpu.cfg.n_layers;
        let ids: Vec<usize> = meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let force: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
            .map(|i| {
                (
                    load_ref(&format!("step0_in{i}")).unwrap(),
                    load_ref(&format!("step0_pm{i}")).unwrap(),
                )
            })
            .collect();
        let mut cpu = Model::open(&Path::new(&tiny()).join("hf")).unwrap();
        cpu.force = Some(force.clone());
        cpu.forward(&ids, 0, &mut |_, _| {}).unwrap();
        m.force = Some(force);
        m.forward(&ids, 0, None).unwrap();
        let want: HashMap<String, Vec<f32>> = cpu.taps.borrow().iter().cloned().collect();
        for (name, v) in m.taps.borrow().iter() {
            if let Some(w) = want.get(name) {
                eprintln!("{name}: rel {:.2e}", rel(v, w));
            }
        }
    }

    /// The device forward teacher-forced against the fixture, as the CPU reference's test.
    #[test]
    fn gpu_tiny_teacher_forced() {
        let Some((mut m, meta)) = open() else { return };
        let n_layers = m.cpu.cfg.n_layers;
        for (s, step) in meta["steps"].as_array().unwrap().iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            m.force = Some(
                (0..n_layers)
                    .map(|i| {
                        (
                            load_ref(&format!("step{s}_in{i}")).unwrap(),
                            load_ref(&format!("step{s}_pm{i}")).unwrap(),
                        )
                    })
                    .collect(),
            );
            let mut got: HashMap<String, Vec<f32>> = HashMap::new();
            let mut hook = |n: &str, t: &[f32]| {
                got.insert(n.to_string(), t.to_vec());
            };
            let (logits, _) = m.forward(&ids, sp, Some(&mut hook)).unwrap();
            let rels: Vec<f32> = (0..n_layers)
                .map(|i| {
                    rel(
                        &got[&format!("layer{i}")],
                        &load_ref(&format!("step{s}_layer{i}")).unwrap(),
                    )
                })
                .collect();
            let k = kl(&load_ref(&format!("step{s}_logits")).unwrap(), &logits);
            let main = load_ref(&format!("step{s}_main_hidden")).unwrap();
            let next = step["next_id"].as_u64().unwrap() as usize;
            let draft = m.forward_spec(next, &main, sp).unwrap();
            let spec = match (draft, step["spec_ids"].as_array()) {
                (Some(d), Some(want)) => {
                    let want: Vec<usize> =
                        want.iter().map(|v| v.as_u64().unwrap() as usize).collect();
                    let r = load_ref(&format!("step{s}_spec_logits")).unwrap();
                    format!(
                        "spec {} kl0 {:.0e}",
                        if d.ids == want {
                            "same".into()
                        } else {
                            format!("{:?} want {want:?}", d.ids)
                        },
                        kl(
                            &r[..m.cpu.cfg.vocab_size],
                            &d.logits[..m.cpu.cfg.vocab_size]
                        )
                    )
                }
                _ => String::new(),
            };
            eprintln!(
                "step {s}: [{}] KL {k:.1e} top1 {} {spec}",
                rels.iter()
                    .map(|r| format!("{r:.1e}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                argmax(&logits) == next
            );
            assert!(k < 1e-3, "step {s}: KL {k}");
            for (i, r) in rels.iter().enumerate() {
                assert!(*r < 5e-2, "step {s} layer {i}: rel {r}");
            }
        }
    }

    /// The expert LRU changes where weights live, never the math: a 3-slot cache gives the
    /// unbounded run's logits bit for bit.
    #[test]
    fn gpu_tiny_expert_cache_is_exact() {
        let Some((mut a, meta)) = open() else { return };
        let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
        b.expert_slots = 3;
        for step in meta["steps"].as_array().unwrap().iter().take(4) {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let (la, _) = a.forward(&ids, sp, None).unwrap();
            let (lb, _) = b.forward(&ids, sp, None).unwrap();
            assert_eq!(la, lb, "step at {sp}");
        }
        eprintln!(
            "misses: unbounded {} vs 3 slots {}",
            a.expert_misses(),
            b.expert_misses()
        );
        assert!(b.expert_misses() > a.expert_misses());
    }

    /// The grouped MoE kernels against the per-expert loop (exact `ds_linear_csr`), on decode
    /// steps, 6-token verify windows and the prefill (the 8-row tiled kernels at t > 8): bit for bit under
    /// DS_EXACT=1; on the K-split path (one bf16 ulp in rare elements per GEMV, compounding
    /// through the layers) the same argmax, logits KL < 1e-3 and hiddens rel < 2e-2.
    #[test]
    fn gpu_tiny_moe_grouped_matches_loop() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        let (mut a, mut b) = (GpuModel::open(&dir).unwrap(), GpuModel::open(&dir).unwrap());
        a.moe_grouped = true;
        b.moe_grouped = false;
        let (v, n) = (a.cpu.cfg.vocab_size, ids.len());
        let exact = a.k.exact;
        let same = |la: &[f32], lb: &[f32], ma: &[f32], mb: &[f32], what: &str| {
            if exact {
                assert_eq!(la, lb, "{what}");
                assert_eq!(ma, mb, "{what} hiddens");
                return;
            }
            let rh = rel(ma, mb);
            let mut kd = 0f64;
            for (pa, pb) in la.chunks(v).zip(lb.chunks(v)) {
                assert_eq!(argmax(pa), argmax(pb), "{what} argmax");
                kd = kd.max(kl(pb, pa));
            }
            eprintln!("{what}: logits KL {kd:.2e}, hiddens rel {rh:.2e}");
            assert!(kd < 1e-3, "{what}: KL {kd:e}");
            assert!(rh < 2e-2, "{what}: hiddens rel {rh:e}");
        };
        let (la, ma) = a.forward(&ids, 0, None).unwrap();
        let (lb, mb) = b.forward(&ids, 0, None).unwrap();
        same(&la, &lb, &ma, &mb, "prefill");
        let mut tok = argmax(&la);
        let mut pos = n;
        for i in 0..6 {
            let (la, ma) = a.forward(&[tok], pos, None).unwrap();
            let (lb, mb) = b.forward(&[tok], pos, None).unwrap();
            same(&la, &lb, &ma, &mb, &format!("decode step {i}"));
            tok = argmax(&la);
            pos += 1;
        }
        for round in 0..3 {
            let win: Vec<usize> = (0..6).map(|i| (tok + 31 * i + round) % v).collect();
            let (la, ma) = a.forward_all(&win, pos).unwrap();
            let (lb, mb) = b.forward_all(&win, pos).unwrap();
            same(&la, &lb, &ma, &mb, &format!("verify window {round}"));
            pos += 6;
            tok = argmax(&la[5 * v..]);
        }
        assert!(
            a.moe_tabs.borrow().values().any(|t| t.is_some()),
            "grouped path never taken"
        );
    }

    /// The K-split mHC glue (default path) against the exact one (`hc_exact`), on the
    /// fixture's teacher-forced steps (prefill, then decode): every layer's hidden within
    /// rel 2e-2, the same argmax, logits KL < 1e-3 (bit for bit under DS_EXACT=1).
    #[test]
    fn gpu_tiny_hc_ks_matches_exact() {
        let Some((mut a, meta)) = open() else { return };
        let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
        b.hc_exact = true;
        let exact = a.k.exact;
        let n_layers = a.cpu.cfg.n_layers;
        let mut worst = 0f32;
        for (s, step) in meta["steps"].as_array().unwrap().iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let force: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
                .map(|i| {
                    (
                        load_ref(&format!("step{s}_in{i}")).unwrap(),
                        load_ref(&format!("step{s}_pm{i}")).unwrap(),
                    )
                })
                .collect();
            let mut run = |m: &mut GpuModel| {
                m.force = Some(force.clone());
                let mut got: HashMap<String, Vec<f32>> = HashMap::new();
                let mut hook = |n: &str, t: &[f32]| {
                    got.insert(n.to_string(), t.to_vec());
                };
                let (l, _) = m.forward(&ids, sp, Some(&mut hook)).unwrap();
                (l, got)
            };
            let (la, ga) = run(&mut a);
            let (lb, gb) = run(&mut b);
            if exact {
                assert_eq!(la, lb, "step {s}");
                continue;
            }
            let rels: Vec<f32> = (0..n_layers)
                .map(|i| {
                    let k = format!("layer{i}");
                    rel(&ga[&k], &gb[&k])
                })
                .collect();
            let k = kl(&lb, &la);
            eprintln!(
                "step {s}: [{}] KL {k:.1e}",
                rels.iter()
                    .map(|r| format!("{r:.1e}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            assert_eq!(argmax(&la), argmax(&lb), "step {s} argmax");
            assert!(k < 1e-3, "step {s}: KL {k}");
            for (i, r) in rels.iter().enumerate() {
                assert!(*r < 2e-2, "step {s} layer {i}: rel {r}");
                worst = worst.max(*r);
            }
        }
        eprintln!("worst layer rel {worst:.1e}");
    }

    /// Free-running on the device, prefill then greedy decode; informational past top-1.
    #[test]
    fn gpu_tiny_free_running() {
        let Some((mut m, meta)) = open() else { return };
        // free-running top-1 against the CPU model is an exact-path property: the tiny random
        // fixture amplifies the K-split mixes' ulp-level drift over steps (6/10 top-1 on the
        // relaxed path, which `gpu_tiny_hc_ks_matches_exact` holds to a KL/argmax bar instead)
        m.hc_exact = true;
        let steps = meta["steps"].as_array().unwrap();
        let mut top1 = 0;
        for (s, step) in steps.iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let (logits, main) = m.forward(&ids, sp, None).unwrap();
            let next = step["next_id"].as_u64().unwrap() as usize;
            top1 += (argmax(&logits) == next) as usize;
            m.forward_spec(argmax(&logits), &main, sp).unwrap();
            eprintln!(
                "step {s}: KL {:.2e} top1 {}",
                kl(&load_ref(&format!("step{s}_logits")).unwrap(), &logits),
                argmax(&logits) == next
            );
        }
        assert!(top1 + 1 >= steps.len(), "top-1 {top1}/{}", steps.len());
    }

    fn prompt(meta: &serde_json::Value) -> Vec<usize> {
        meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }

    /// As forward.rs `tiny_chunk_and_rewind_exact`, on the device.
    #[test]
    fn gpu_tiny_chunk_and_rewind_exact() {
        let Some((mut a, meta)) = open() else { return };
        let ids = prompt(&meta);
        let (v, n) = (a.cpu.cfg.vocab_size, ids.len());
        a.forward(&ids, 0, None).unwrap();
        let stream: Vec<usize> = (0..14).map(|i| (37 * i + 11) % v).collect();
        let mut want = Vec::new();
        for (i, &t) in stream.iter().enumerate() {
            want.push(a.forward(&[t], n + i, None).unwrap().0);
        }
        drop(a);
        for keep in 1..=6 {
            let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
            b.forward(&ids, 0, None).unwrap();
            b.forward(&stream[..1], n, None).unwrap();
            let snap = b.snapshot().unwrap();
            let (lw, _) = b.forward_all(&stream[1..7], n + 1).unwrap();
            for e in 0..6 {
                assert_eq!(lw[e * v..(e + 1) * v], want[1 + e][..], "window row {e}");
            }
            b.rewind(&snap, n + 1 + keep, n + 7).unwrap();
            for i in 1 + keep..stream.len() {
                let (l, _) = b.forward(&[stream[i]], n + i, None).unwrap();
                assert_eq!(l, want[i], "keep {keep} step {i}");
            }
        }
    }

    /// Graphs change how a decode step is launched, never the math: captured and replayed
    /// decode steps and verify windows give the direct run's logits bit for bit.
    #[test]
    fn gpu_tiny_graph_matches_direct() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        let (mut a, mut b) = (GpuModel::open(&dir).unwrap(), GpuModel::open(&dir).unwrap());
        a.graph = true;
        b.graph = false;
        let (v, n) = (a.cpu.cfg.vocab_size, ids.len());
        let (la, _) = a.forward(&ids, 0, None).unwrap();
        let (lb, _) = b.forward(&ids, 0, None).unwrap();
        assert_eq!(la, lb, "prefill");
        let mut tok = argmax(&la);
        let mut pos = n;
        for i in 0..10 {
            let (la, ma) = a.forward(&[tok], pos, None).unwrap();
            let (lb, mb) = b.forward(&[tok], pos, None).unwrap();
            assert_eq!(la, lb, "decode step {i}");
            assert_eq!(ma, mb, "decode step {i} hiddens");
            tok = argmax(&la);
            pos += 1;
        }
        for round in 0..4 {
            let win: Vec<usize> = (0..6).map(|i| (tok + 31 * i + round) % v).collect();
            let (sa, sb) = (a.snapshot().unwrap(), b.snapshot().unwrap());
            let (la, ma) = a.forward_all(&win, pos).unwrap();
            let (lb, mb) = b.forward_all(&win, pos).unwrap();
            assert_eq!(la, lb, "verify window {round}");
            assert_eq!(ma, mb, "verify window {round} hiddens");
            let keep = 1 + round;
            a.rewind(&sa, pos + keep, pos + 6).unwrap();
            b.rewind(&sb, pos + keep, pos + 6).unwrap();
            pos += keep;
            let (la, _) = a.forward(&[tok], pos, None).unwrap();
            let (lb, _) = b.forward(&[tok], pos, None).unwrap();
            assert_eq!(la, lb, "decode after rewind {round}");
            tok = argmax(&la);
            pos += 1;
        }
        eprintln!("graph: {:?}, direct: {:?}", a.graph_stats, b.graph_stats);
        assert!(a.graph_stats.replays >= 8, "{:?}", a.graph_stats);
        assert_eq!(b.graph_stats.captures + b.graph_stats.replays, 0);
    }

    /// The forked expert-parallel reduce (a graphable `moe_reduce` on the side stream, joined
    /// before the sum) orders like the in-line one: a device "reduce" doubling `y` (what two
    /// identical ranks would all-reduce), forked, captured and replayed, or forked and direct,
    /// gives the logits of the same doubling done by a syncing host reduce on the main
    /// stream, bit for bit. Single-GPU stand-in for NCCL's all-reduce.
    #[test]
    fn gpu_tiny_forked_reduce_exact() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        let mut host = GpuModel::open(&dir).unwrap();
        host.moe_reduce = Some(Box::new(|y, n, s| {
            s.sync()?;
            let v: Vec<f32> = y.read::<f32>(0, n)?.iter().map(|a| a + a).collect();
            y.write(0, &v)?;
            Ok(())
        }));
        let dev = |graph: bool| {
            let mut m = GpuModel::open(&dir).unwrap();
            let k = Rc::new(Kernels::new(&Gpu::at(0).unwrap()).unwrap());
            m.moe_reduce = Some(Box::new(move |y, n, s| {
                let (p, nn) = (y.ptr, n as i64);
                unsafe { k.go("ds_add", blocks(n, 256), B256, 0, s, args![p, p, nn])? };
                Ok(())
            }));
            m.reduce_graphable = true;
            m.graph = graph;
            m
        };
        let (mut g, mut d) = (dev(true), dev(false));
        let v = host.cpu.cfg.vocab_size;
        let lh = host.forward(&ids, 0, None).unwrap().0;
        assert_eq!(lh, g.forward(&ids, 0, None).unwrap().0, "prefill graph");
        assert_eq!(lh, d.forward(&ids, 0, None).unwrap().0, "prefill direct");
        let mut tok = argmax(&lh);
        let mut pos = ids.len();
        for i in 0..4 {
            let lh = host.forward(&[tok], pos, None).unwrap().0;
            assert_eq!(
                lh,
                g.forward(&[tok], pos, None).unwrap().0,
                "decode {i} graph"
            );
            assert_eq!(
                lh,
                d.forward(&[tok], pos, None).unwrap().0,
                "decode {i} direct"
            );
            tok = argmax(&lh);
            pos += 1;
        }
        for round in 0..3 {
            let win: Vec<usize> = (0..6).map(|i| (tok + 31 * i + round) % v).collect();
            let lh = host.forward_all(&win, pos).unwrap().0;
            assert_eq!(
                lh,
                g.forward_all(&win, pos).unwrap().0,
                "verify {round} graph"
            );
            assert_eq!(
                lh,
                d.forward_all(&win, pos).unwrap().0,
                "verify {round} direct"
            );
            tok = argmax(&lh);
            pos += 6;
        }
        eprintln!("forked graph: {:?}", g.graph_stats);
        assert!(g.graph_stats.replays >= 3, "{:?}", g.graph_stats);
        assert_eq!(host.graph_stats.captures + host.graph_stats.replays, 0);
    }

    /// `DS_OBS=2` on a direct step: one `layer` span per backbone layer, and the ops inside
    /// the layers plus `embed` and `head` account for the step's device time (within 20%).
    #[test]
    fn gpu_tiny_obs_layers_sum_to_step() {
        use super::super::obs;
        let Some((mut m, meta)) = open() else { return };
        m.obs.over.set(Some((2, 1, true)));
        obs::collect(true);
        let ids = prompt(&meta);
        let (mut l, _) = m.forward(&ids, 0, None).unwrap();
        let mut pos = ids.len();
        for _ in 0..5 {
            l = m.forward(&[argmax(&l)], pos, None).unwrap().0;
            pos += 1;
        }
        let recs: Vec<_> = obs::take()
            .into_iter()
            .filter(|r| r.kind == "decode" && r.t == 1 && r.sp >= ids.len())
            .collect();
        obs::collect(false);
        assert_eq!(recs.len(), 5);
        let nl = m.cpu.cfg.n_layers;
        for r in &recs {
            assert_eq!(r.path, "direct");
            assert!(
                r.counts.launches > 0 && r.counts.syncs >= 1,
                "{:?}",
                r.counts
            );
            assert_eq!(r.spans.iter().filter(|s| s.name == "layer").count(), nl);
            let top: f64 = r
                .spans
                .iter()
                .filter(|s| s.name == "embed" || s.name == "head")
                .map(|s| s.gpu_ms)
                .sum();
            let ops = obs::layer_ops_ms(r) + top;
            let ratio = ops / r.gpu_ms;
            eprintln!(
                "step {} gpu {:.3} ms, ops {:.3} ms ({:.1}%), covered {:.3} ms, launches {}",
                r.step,
                r.gpu_ms,
                ops,
                100.0 * ratio,
                obs::covered_ms(r),
                r.counts.launches
            );
            for (n, ms, k) in obs::breakdown(r) {
                eprintln!("  {n:<26} {ms:>8.3} ms x{k}");
            }
            assert!(
                (0.8..=1.2).contains(&ratio),
                "ops {ops} vs gpu {}",
                r.gpu_ms
            );
        }
    }

    /// The timeline doesn't change the math, and graph capture works with it on.
    #[test]
    fn gpu_tiny_obs_graph_exact() {
        let Some((mut a, meta)) = open() else { return };
        let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
        b.obs.over.set(Some((2, 1, false)));
        let ids = prompt(&meta);
        let (la, _) = a.forward(&ids, 0, None).unwrap();
        let (lb, _) = b.forward(&ids, 0, None).unwrap();
        assert_eq!(la, lb);
        let (mut ta, mut pos) = (argmax(&la), ids.len());
        for _ in 0..6 {
            let la = a.forward(&[ta], pos, None).unwrap().0;
            let lb = b.forward(&[ta], pos, None).unwrap().0;
            assert_eq!(la, lb, "at {pos}");
            ta = argmax(&la);
            pos += 1;
        }
        assert!(
            b.graph_stats.captures >= 1 && b.graph_stats.replays >= 2,
            "{:?}",
            b.graph_stats
        );
    }

    /// Timeline overhead: decode us/step with it off, at 1 and at 2 (graph path), and off vs 2
    /// on the direct path.
    #[test]
    #[ignore]
    fn gpu_tiny_obs_overhead() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        // the fixture's context is short: every round restarts the sequence
        let ctx = GpuModel::open(&dir).unwrap().cpu.cfg.max_seq_len;
        let steps = ctx.saturating_sub(ids.len() + 8).max(8);
        for (lvl, direct) in [(0, false), (1, false), (2, false), (0, true), (2, true)] {
            let mut m = GpuModel::open(&dir).unwrap();
            m.obs.over.set(Some((lvl, 1, direct)));
            let (l, _) = m.forward(&ids, 0, None).unwrap();
            let mut tok = argmax(&l);
            let mut pos = ids.len();
            for _ in 0..4 {
                tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                pos += 1;
            }
            let mut best = f64::MAX;
            for _ in 0..10 {
                let t0 = std::time::Instant::now();
                for _ in 0..steps {
                    tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                    pos += 1;
                }
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / steps as f64);
                m.reset().unwrap();
                pos = 0;
                let (l, _) = m.forward(&ids, 0, None).unwrap();
                tok = argmax(&l);
                pos += ids.len();
            }
            eprintln!("DS_OBS={lvl} direct={direct}: {best:.0} us/step");
        }
    }

    /// Decode-step time on the tiny model, graphs on vs off (launch overhead is the whole
    /// story at this size). `cargo test ... gpu_tiny_graph_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_tiny_graph_bench() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        let steps = 50;
        for graph in [false, true] {
            let mut m = GpuModel::open(&dir).unwrap();
            m.graph = graph;
            let (l, _) = m.forward(&ids, 0, None).unwrap();
            let mut tok = argmax(&l);
            let mut pos = ids.len();
            // warm up: weights, arena, capture
            for _ in 0..4 {
                tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                pos += 1;
            }
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                pos += 1;
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;
            eprintln!("graph {graph}: {us:.0} us/step ({:?})", m.graph_stats);
        }
    }

    /// As forward.rs `tiny_spec_matches_greedy`, on the device.
    /// Sampled speculative steps read back only the logits rows the accept walk reaches
    /// (`verify_window` + `logits_row`): the same tokens, accepted counts and RNG state as the
    /// full read-back (`forward_all` + `verify_rows`) from the same state.
    #[test]
    fn gpu_tiny_spec_sampling_lazy_rows() {
        use super::super::forward::SpecSampler;
        let Some((_, meta)) = open() else { return };
        let ids = prompt(&meta);
        let n = ids.len();
        let start = |m: &mut GpuModel| {
            let (l, main) = m.forward(&ids, 0, None).unwrap();
            m.forward_spec(0, &main, 0).unwrap();
            let t0 = argmax(&l);
            let (_, main) = m.forward(&[t0], n, None).unwrap();
            (t0, main)
        };
        let dir = Path::new(&tiny()).join("hf");
        let (mut a, mut b) = (GpuModel::open(&dir).unwrap(), GpuModel::open(&dir).unwrap());
        let ((ta, mut ma), (tb, mut mb)) = (start(&mut a), start(&mut b));
        assert_eq!(ta, tb);
        let (mut tok, mut sp) = (ta, n);
        let c = a.cpu.cfg.clone();
        let (v, w) = (c.vocab_size, c.dim * c.dspark_target_layer_ids.len());
        let (mut pa, mut pb) = (
            SpecSampler::new(0.9, 0.95, 5),
            SpecSampler::new(0.9, 0.95, 5),
        );
        let mut rows = 0;
        for step in 0..8 {
            let st = a
                .decode_spec(tok, &ma, sp, &mut |_, _| {}, &mut pa)
                .unwrap();
            let pos0 = sp + mb.len() / w;
            b.cpu.cfg.dspark_block_size = c.draft_block(false);
            let mut d = b
                .forward_spec_with(tok, &mb, sp, &mut |q: &[f32]| pb.draft(q))
                .unwrap()
                .unwrap();
            // the window decode_spec verifies (DS_DRAFT_N* / DS_DRAFT_CONF)
            d.ids.truncate(1 + c.draft_keep(&d.conf, false));
            let snap = b.snapshot().unwrap();
            let (logits, mains) = b.forward_all(&d.ids, pos0).unwrap();
            let (k, tokens) =
                super::super::forward::verify_rows(&logits, &d.logits, &d.ids, v, &mut pb);
            b.rewind(&snap, pos0 + k + 1, pos0 + d.ids.len()).unwrap();
            assert_eq!((st.accepted, &st.tokens), (k, &tokens), "step {step}");
            assert_eq!(pa.rng, pb.rng, "step {step}");
            assert_eq!(st.main, mains[..(k + 1) * w], "step {step}");
            rows += k + 1;
            tok = *tokens.last().unwrap();
            (ma, mb, sp) = (st.main, mains[..(k + 1) * w].to_vec(), pos0);
        }
        eprintln!(
            "8 sampled steps: {rows} logits rows; walks device/host {:?}",
            a.spec_walks
        );
    }

    #[test]
    fn gpu_tiny_spec_matches_greedy() {
        let Some((mut m, meta)) = open() else { return };
        let ids = prompt(&meta);
        let (v, n, len) = (m.cpu.cfg.vocab_size, ids.len(), 40);
        let (mut l, _) = m.forward(&ids, 0, None).unwrap();
        let mut want = Vec::new();
        while want.len() < len {
            want.push(argmax(&l));
            l = m
                .forward(&[*want.last().unwrap()], n + want.len() - 1, None)
                .unwrap()
                .0;
        }
        drop(m);
        let (oracle, oracle2) = (want.clone(), want.clone());
        let mut rng = 1u64;
        type Edit = Box<dyn FnMut(usize, &mut [usize])>;
        let runs: Vec<(&str, Edit)> = vec![
            ("dspark", Box::new(|_, _: &mut [usize]| {})),
            (
                "oracle",
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle.get(pos + i - n) {
                            *x = w;
                        }
                    }
                }),
            ),
            (
                "wrong",
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle2.get(pos + i - n) {
                            *x = w;
                        }
                        rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        if (rng >> 33) % 5 == 0 {
                            *x = (*x + 1 + (rng >> 40) as usize % (v - 1)) % v;
                        }
                    }
                }),
            ),
        ];
        for (name, mut edit) in runs {
            let mut m = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
            let (l, main) = m.forward(&ids, 0, None).unwrap();
            let t0 = argmax(&l);
            m.forward_spec(t0, &main, 0).unwrap();
            let (l, mut main) = m.forward(&[t0], n, None).unwrap();
            let mut out = vec![t0, argmax(&l)];
            let (mut sp, mut drafted, mut accepted) = (n, 0, 0);
            while out.len() < len {
                let s = m
                    .decode_spec(
                        *out.last().unwrap(),
                        &main,
                        sp,
                        &mut *edit,
                        &mut super::super::forward::Greedy,
                    )
                    .unwrap();
                drafted += s.drafted;
                accepted += s.accepted;
                out.extend(s.tokens);
                main = s.main;
                sp = s.sp;
            }
            out.truncate(len);
            eprintln!(
                "{name}: accepted {accepted}/{drafted} ({:.1}%)",
                100.0 * accepted as f64 / drafted.max(1) as f64
            );
            assert_eq!(out, want, "{name}");
        }
    }

    /// `DS_DRAFT_N_GREEDY` below (3) and above (7, the drafter's block grown past the trained 5) the
    /// trained block, and the confidence early stop: greedy spec equals plain greedy, with
    /// windows of at most n + 1 rows.
    #[test]
    fn gpu_tiny_spec_draft_n() {
        let Some((mut m, meta)) = open() else { return };
        let ids = prompt(&meta);
        let (n, len) = (ids.len(), 40);
        let (mut l, _) = m.forward(&ids, 0, None).unwrap();
        let mut want = Vec::new();
        while want.len() < len {
            want.push(argmax(&l));
            l = m
                .forward(&[*want.last().unwrap()], n + want.len() - 1, None)
                .unwrap()
                .0;
        }
        drop(m);
        let keep = [
            "DS_DRAFT_N",
            "DS_DRAFT_N_GREEDY",
            "DS_DRAFT_N_SAMPLED",
            "DS_DRAFT_CONF",
        ]
        .map(|k| (k, std::env::var_os(k)));
        for (dn, conf) in [("3", "0"), ("7", "0"), ("7", "0.3")] {
            // the greedy length alone; sampled steps keep their own (3 here)
            std::env::remove_var("DS_DRAFT_N");
            std::env::set_var("DS_DRAFT_N_GREEDY", dn);
            std::env::set_var("DS_DRAFT_N_SAMPLED", "3");
            std::env::set_var("DS_DRAFT_CONF", conf);
            let m = GpuModel::open(&Path::new(&tiny()).join("hf"));
            for (k, v) in &keep {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
            let mut m = m.unwrap();
            let dn: usize = dn.parse().unwrap();
            assert_eq!((m.cpu.cfg.draft_n, m.cpu.cfg.draft_n_sampled), (dn, 3));
            assert_eq!(m.cpu.cfg.dspark_block_size, dn.max(5));
            let (l, main) = m.forward(&ids, 0, None).unwrap();
            let t0 = argmax(&l);
            m.forward_spec(t0, &main, 0).unwrap();
            let (l, mut main) = m.forward(&[t0], n, None).unwrap();
            let mut out = vec![t0, argmax(&l)];
            let (mut sp, mut drafted, mut accepted, mut steps) = (n, 0, 0, 0);
            while out.len() < len {
                let s = m
                    .decode_spec(
                        *out.last().unwrap(),
                        &main,
                        sp,
                        &mut |_, _| {},
                        &mut super::super::forward::Greedy,
                    )
                    .unwrap();
                assert!(s.drafted <= dn && s.drafted >= 1, "drafted {}", s.drafted);
                drafted += s.drafted;
                accepted += s.accepted;
                steps += 1;
                out.extend(s.tokens);
                main = s.main;
                sp = s.sp;
            }
            out.truncate(len);
            eprintln!("n {dn} conf {conf}: accepted {accepted}/{drafted} in {steps} steps");
            assert_eq!(out, want, "n {dn} conf {conf}");
        }
    }
}
