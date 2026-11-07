//! Run tang-compile tensor graphs natively on Metal: the MSL lowering
//! ([`TensorGraph::to_msl`]) plus a runtime that owns its Metal buffers
//! end to end. Native `half` buffers, and matrix-fragment kernels for
//! f16 contractions. WGSL ([`crate::ir_gpu`]) stays the portable path.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use tang_compile::tensor::msl::MslProgram;
use tang_compile::tensor::precision::{bf16_bits, bf16_from_bits, f16_bits, f16_from_bits};
use tang_compile::tensor::{ErrorBudget, ErrorStats, Format, Op, Precision, TensorGraph, TensorId};

/// A Metal device, its queue and a pipeline cache keyed by source.
pub struct MslDevice {
    pub device: Device,
    pub queue: CommandQueue,
    pub name: String,
    pipelines: RefCell<HashMap<String, ComputePipelineState>>,
    /// Compile with fast math (approximate `exp2`/`log2`/division).
    /// Off by default: the tests compare against an f64 reference.
    pub fast_math: bool,
}

impl MslDevice {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let name = device.name().to_string();
        Some(Self {
            device,
            queue,
            name,
            pipelines: RefCell::new(HashMap::new()),
            fast_math: false,
        })
    }

    fn pipeline(&self, source: &str, entry: &str) -> ComputePipelineState {
        if let Some(p) = self.pipelines.borrow().get(source) {
            return p.clone();
        }
        if let Ok(dir) = std::env::var("MSL_DUMP") {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut h);
            let _ = std::fs::write(format!("{dir}/{:016x}.metal", h.finish()), source);
        }
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(self.fast_math);
        let lib = self
            .device
            .new_library_with_source(source, &opts)
            .unwrap_or_else(|e| panic!("MSL compile failed: {e}\n{source}"));
        let f = lib.get_function(entry, None).unwrap();
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap();
        self.pipelines
            .borrow_mut()
            .insert(source.to_string(), p.clone());
        p
    }
}

/// Buffers for an [`MslProgram`]: graph inputs uploaded, intermediates
/// allocated, reused across runs.
pub struct Executable {
    pub program: MslProgram,
    pub buffers: Vec<metal::Buffer>,
    pipelines: Vec<ComputePipelineState>,
}

fn bytes_of<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl Executable {
    /// Whether every kernel's threadgroup memory fits the device.
    pub fn fits(dev: &MslDevice, program: &MslProgram) -> bool {
        let limit = dev.device.max_threadgroup_memory_length() as usize;
        program.kernels.iter().all(|k| k.threadgroup_bytes <= limit)
    }

    pub fn new(dev: &MslDevice, program: MslProgram, inputs: &[&[f32]]) -> Self {
        assert!(
            Self::fits(dev, &program),
            "a kernel's threadgroup memory exceeds the device limit"
        );
        let opts = MTLResourceOptions::StorageModeShared;
        let mk = |b: &[u8]| {
            dev.device
                .new_buffer_with_data(b.as_ptr() as *const _, b.len().max(4) as u64, opts)
        };
        let buffers = program
            .buffers
            .iter()
            .map(|b| match b.input {
                Some(k) => {
                    assert_eq!(inputs[k].len(), b.len, "input {k} has wrong length");
                    match b.format {
                        Format::F32 => mk(bytes_of(inputs[k])),
                        Format::F16 => {
                            let h: Vec<u16> =
                                inputs[k].iter().map(|&x| f16_bits(x as f64)).collect();
                            mk(bytes_of(&h))
                        }
                        Format::BF16 => {
                            let h: Vec<u16> =
                                inputs[k].iter().map(|&x| bf16_bits(x as f64)).collect();
                            mk(bytes_of(&h))
                        }
                        f @ (Format::Q4K | Format::Q5K | Format::Q8x32) => {
                            unimplemented!("MSL runtime: block format {f:?} inputs")
                        }
                    }
                }
                None => dev
                    .device
                    .new_buffer((b.len * b.format.bytes()).max(4) as u64, opts),
            })
            .collect();
        let pipelines = program
            .kernels
            .iter()
            .map(|k| dev.pipeline(&k.source, &k.entry))
            .collect();
        Self {
            program,
            buffers,
            pipelines,
        }
    }

    fn index(&self, t: TensorId) -> usize {
        let t = self.program.storage(t);
        self.program
            .buffers
            .iter()
            .position(|b| b.node == t)
            .unwrap()
    }

    pub fn format(&self, t: TensorId) -> Format {
        self.program.buffers[self.index(t)].format
    }

    pub fn buffer(&self, t: TensorId) -> &metal::Buffer {
        &self.buffers[self.index(t)]
    }

    fn encode(&self, enc: &metal::ComputeCommandEncoderRef, i: usize) {
        let k = &self.program.kernels[i];
        enc.set_compute_pipeline_state(&self.pipelines[i]);
        for (b, t) in k.reads.iter().enumerate() {
            enc.set_buffer(b as u64, Some(self.buffer(*t)), 0);
        }
        let n = k.reads.len() as u64;
        enc.set_buffer(n, Some(self.buffer(k.node)), 0);
        let params = [k.points, 0, 0, 0];
        enc.set_bytes(n + 1, 16, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(k.groups.0 as u64, k.groups.1 as u64, 1),
            MTLSize::new(k.threads as u64, 1, 1),
        );
    }

    /// Encode `reps` runs of kernels `range` into one command buffer and
    /// wait for it.
    fn submit(&self, dev: &MslDevice, kernels: std::ops::Range<usize>, reps: usize) {
        let cb = dev.queue.new_command_buffer();
        // serial dispatch: each kernel sees the previous one's writes
        let enc = cb.new_compute_command_encoder();
        for _ in 0..reps {
            for i in kernels.clone() {
                self.encode(enc, i);
            }
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    /// Run every kernel once and wait.
    pub fn run(&self, dev: &MslDevice) {
        self.submit(dev, 0..self.program.kernels.len(), 1);
    }

    /// Run kernel `i` only (its inputs must already be computed).
    pub fn run_kernel(&self, dev: &MslDevice, i: usize) {
        self.submit(dev, i..i + 1, 1);
    }

    /// Read `t` back as f32 (f16/bf16 widened exactly).
    pub fn read(&self, t: TensorId) -> Vec<f32> {
        let i = self.index(t);
        let spec = &self.program.buffers[i];
        let p = self.buffers[i].contents();
        unsafe {
            match spec.format {
                Format::F32 => std::slice::from_raw_parts(p as *const f32, spec.len).to_vec(),
                Format::F16 => std::slice::from_raw_parts(p as *const u16, spec.len)
                    .iter()
                    .map(|&h| f16_from_bits(h) as f32)
                    .collect(),
                Format::BF16 => std::slice::from_raw_parts(p as *const u16, spec.len)
                    .iter()
                    .map(|&h| bf16_from_bits(h) as f32)
                    .collect(),
                f @ (Format::Q4K | Format::Q5K | Format::Q8x32) => {
                    unimplemented!("MSL runtime: reading block format {f:?}")
                }
            }
        }
    }

    /// Mean ms of kernels `range` over `iters` back-to-back runs in one
    /// command buffer (after a warm-up submission): kernel-only time.
    pub fn time_kernels(
        &self,
        dev: &MslDevice,
        range: std::ops::Range<usize>,
        iters: usize,
    ) -> f64 {
        self.submit(dev, range.clone(), 3);
        let t = Instant::now();
        self.submit(dev, range, iters);
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    }

    /// Mean ms of one whole-program run with one command buffer per run
    /// (committed back to back, one wait at the end): the launch pattern
    /// of the wgpu path, which submits once per run.
    pub fn time_separate(&self, dev: &MslDevice, iters: usize) -> f64 {
        self.run(dev);
        let t = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for i in 0..self.program.kernels.len() {
                self.encode(enc, i);
            }
            enc.end_encoding();
            cb.commit();
            last = Some(cb.to_owned());
        }
        last.unwrap().wait_until_completed();
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    }

    /// Mean ms of one whole-program run.
    pub fn time(&self, dev: &MslDevice, iters: usize) -> f64 {
        self.time_kernels(dev, 0..self.program.kernels.len(), iters)
    }
}

/// Run and compare every output with `want` (f64).
pub fn gpu_error(
    dev: &MslDevice,
    exe: &Executable,
    outputs: &[TensorId],
    want: &[Vec<f64>],
) -> ErrorStats {
    exe.run(dev);
    let got: Vec<Vec<f64>> = outputs
        .iter()
        .map(|o| exe.read(*o).iter().map(|&v| v as f64).collect())
        .collect();
    ErrorStats::compare(&got, want)
}

/// Contractions that could run on f16 fragments if their operands were
/// stored in f16: every materialized comp whose `fragment_plan` succeeds
/// once its operand buffers are f16.
pub fn fragment_candidates(g: &TensorGraph, outputs: &[TensorId]) -> Vec<TensorId> {
    let prog = g.to_msl(outputs);
    prog.kernels
        .iter()
        .map(|k| k.node)
        .filter(|&t| {
            let v = with_f16_operands(g, t);
            v.fragment_plan(t).is_some() || v.attention_plan(t).is_some()
        })
        .collect()
}

/// `g` with `t`'s contraction operands' buffers in f16 storage.
fn with_f16_operands(g: &TensorGraph, t: TensorId) -> TensorGraph {
    let mut v = g.clone();
    for s in operand_buffers(g, t) {
        let mut p = v.precision(s);
        p.storage = Format::F16;
        v.set_precision(s, p);
    }
    v
}

/// Buffers holding the operands of `t`'s core contraction (the two
/// factors of its product body; unread args are left alone), or of its
/// attention (`Q`, `K`, `V`).
fn operand_buffers(g: &TensorGraph, t: TensorId) -> Vec<TensorId> {
    g.fragment_operands(t)
        .map(|o| o.to_vec())
        .or_else(|| g.attention_operands(t).map(|o| o.to_vec()))
        .unwrap_or_default()
}

/// Which backend a tuned graph runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Wgsl,
    Msl,
}

/// One candidate [`tune_precision`] tried.
pub struct PrecisionTrial {
    pub what: String,
    pub error: ErrorStats,
    pub ms: Option<f64>,
    pub kept: bool,
}

pub struct PrecisionTune {
    pub graph: TensorGraph,
    pub assignment: Vec<(TensorId, Precision)>,
    /// Nodes lowered onto matrix fragments.
    pub fragments: Vec<TensorId>,
    pub error: ErrorStats,
    pub base_error: ErrorStats,
    pub base_ms: f64,
    pub ms: f64,
    pub trials: Vec<PrecisionTrial>,
}

/// [`crate::ir_gpu::tune_precision`] on the native Metal backend, with one
/// more candidate dimension: **fragment lowering**. For every contraction
/// that fits [`TensorGraph::fragment_plan`], the candidate "operands in
/// f16 + f32 accumulator on simdgroup fragments" is tried as one move
/// (fragments need f16 operands; f16 operands alone are also tried by the
/// per-buffer search). Then every remaining buffer is tried in f16, largest
/// first. A change is kept if the program stays within `budget` against
/// the f64 reference and gets at least 3% faster; the final schedule is
/// re-timed against the f32 baseline and dropped if the gain is noise.
///
/// Unlike the WGSL search, f16 accumulators run natively here (each
/// combine rounds through `half`) and are timed and kept like any change.
pub fn tune_precision(
    dev: &MslDevice,
    g: TensorGraph,
    outputs: &[TensorId],
    inputs: &[&[f32]],
    want: &[Vec<f64>],
    budget: ErrorBudget,
    iters: usize,
) -> PrecisionTune {
    let measure = |g: &TensorGraph| {
        let exe = Executable::new(dev, g.to_msl(outputs), inputs);
        let err = gpu_error(dev, &exe, outputs, want);
        let ms = budget.admits(&err).then(|| {
            (0..3)
                .map(|_| exe.time(dev, iters))
                .fold(f64::INFINITY, f64::min)
        });
        (err, ms)
    };
    let _ = measure(&g);
    let (base_error, base_ms) = measure(&g);
    let base_ms =
        base_ms.unwrap_or_else(|| Executable::new(dev, g.to_msl(outputs), inputs).time(dev, iters));
    let base = g.clone();
    let mut cur = g;
    let (mut cur_err, mut cur_ms) = (base_error, base_ms);
    let mut trials = Vec::new();
    let mut try_move = |cur: &mut TensorGraph,
                        cur_err: &mut ErrorStats,
                        cur_ms: &mut f64,
                        v: TensorGraph,
                        what: String| {
        let (err, ms) = measure(&v);
        let kept = ms.is_some_and(|t| t < *cur_ms * 0.97);
        trials.push(PrecisionTrial {
            what,
            error: err,
            ms,
            kept,
        });
        if kept {
            *cur = v;
            *cur_err = err;
            *cur_ms = ms.unwrap();
        }
    };

    // fragments first: the largest single win when they apply
    for t in fragment_candidates(&cur, outputs) {
        let mut v = with_f16_operands(&cur, t);
        v.set_fragment(t, true);
        try_move(
            &mut cur,
            &mut cur_err,
            &mut cur_ms,
            v,
            format!("fragment {}", t.index()),
        );
    }
    // then f16 storage per buffer, largest first
    let mut bufs: Vec<(TensorId, usize)> = cur
        .to_msl(outputs)
        .buffers
        .iter()
        .map(|b| (b.node, cur.buffer_bytes(b.node)))
        .collect();
    bufs.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (node, _) in bufs {
        if cur.precision(node).storage == Format::F16 {
            continue;
        }
        let mut p = cur.precision(node);
        p.storage = Format::F16;
        let mut v = cur.clone();
        v.set_precision(node, p);
        try_move(
            &mut cur,
            &mut cur_err,
            &mut cur_ms,
            v,
            format!("f16 storage {}", node.index()),
        );
    }
    // then f16 accumulators (never on fragment nodes: they need f32)
    let reductions: Vec<TensorId> = (0..cur.nodes().len())
        .map(TensorId::from_index)
        .filter(|&t| !cur.nodes()[t.index()].fragment)
        .filter(|&t| matches!(&cur.nodes()[t.index()].op, Op::Comp(c) if c.monoid.is_some()))
        .collect();
    for node in reductions {
        let mut p = cur.precision(node);
        p.accum = Format::F16;
        let mut v = cur.clone();
        v.set_precision(node, p);
        try_move(
            &mut cur,
            &mut cur_err,
            &mut cur_ms,
            v,
            format!("f16 accum {}", node.index()),
        );
    }

    let assignment: Vec<(TensorId, Precision)> = (0..cur.nodes().len())
        .map(TensorId::from_index)
        .filter(|&t| cur.precision(t) != Precision::F32)
        .map(|t| (t, cur.precision(t)))
        .collect();
    let fragments: Vec<TensorId> = (0..cur.nodes().len())
        .map(TensorId::from_index)
        .filter(|&t| cur.nodes()[t.index()].fragment)
        .collect();
    let (mut base_ms, mut cur_ms) = (base_ms, cur_ms);
    if !assignment.is_empty() || !fragments.is_empty() {
        let b = Executable::new(dev, base.to_msl(outputs), inputs);
        let c = Executable::new(dev, cur.to_msl(outputs), inputs);
        base_ms = f64::INFINITY;
        cur_ms = f64::INFINITY;
        for _ in 0..3 {
            base_ms = base_ms.min(b.time(dev, iters));
            cur_ms = cur_ms.min(c.time(dev, iters));
        }
        if cur_ms >= base_ms * 0.99 {
            return PrecisionTune {
                graph: base,
                assignment: Vec::new(),
                fragments: Vec::new(),
                error: base_error,
                base_error,
                base_ms,
                ms: base_ms,
                trials,
            };
        }
    }
    PrecisionTune {
        graph: cur,
        assignment,
        fragments,
        error: cur_err,
        base_error,
        base_ms,
        ms: cur_ms,
        trials,
    }
}

/// `(split, tile, reg, fragment)`: one Metal thread mapping.
type MslCandidate = (u32, (u32, u32, u32), (u32, u32), bool);

/// One kernel's winning mapping on Metal.
pub struct TunedKernel {
    pub node: TensorId,
    pub split: u32,
    pub tile: (u32, u32, u32),
    pub reg: (u32, u32),
    pub fragment: bool,
    pub ms: f64,
}

/// Per-kernel autotuning on Metal: every thread mapping from
/// `schedule_candidates`, plus fragment lowering where the node's
/// precision admits it (f16 operands, f32 accumulator). Precision is not
/// changed here — that is [`tune_precision`]'s job.
pub fn autotune_kernels(
    dev: &MslDevice,
    mut g: TensorGraph,
    outputs: &[TensorId],
    inputs: &[&[f32]],
    iters: usize,
) -> (TensorGraph, Vec<TunedKernel>) {
    let base = Executable::new(dev, g.to_msl(outputs), inputs);
    let nodes: Vec<TensorId> = base.program.kernels.iter().map(|k| k.node).collect();
    let mut log = Vec::new();
    for (i, node) in nodes.into_iter().enumerate() {
        let mut cands: Vec<MslCandidate> = g
            .schedule_candidates(node)
            .into_iter()
            .map(|(s, t, r)| (s, t, r, false))
            .collect();
        if g.fragment_plan(node).is_some() {
            cands.push((0, (0, 0, 0), (1, 1), true));
        }
        let mut best: Option<(MslCandidate, f64)> = None;
        for (split, tile, reg, frag) in cands {
            let mut v = g.clone();
            v.set_schedule(node, split, tile, reg);
            v.set_fragment(node, frag);
            let prog = v.to_msl(outputs);
            if !Executable::fits(dev, &prog) {
                continue;
            }
            let exe = Executable::new(dev, prog, inputs);
            exe.submit(dev, 0..i + 1, 1);
            let ms = exe.time_kernels(dev, i..i + 1, iters);
            if best.is_none_or(|b| ms < b.1) {
                best = Some(((split, tile, reg, frag), ms));
            }
        }
        let ((split, tile, reg, fragment), ms) = best.unwrap();
        g.set_schedule(node, split, tile, reg);
        g.set_fragment(node, fragment);
        log.push(TunedKernel {
            node,
            split,
            tile,
            reg,
            fragment,
            ms,
        });
    }
    (g, log)
}

/// The backend dimension of the search: run the same scheduled graph
/// through WGSL (wgpu) and through native MSL, check both against `want`
/// under `budget`, and return the faster admitted one with both timings
/// (`None` where a backend could not run it or missed the budget).
#[allow(clippy::too_many_arguments)]
pub fn pick_backend(
    wgpu: &tang_gpu::GpuDevice,
    cache: &mut tang_gpu::KernelCache,
    dev: &MslDevice,
    g: &TensorGraph,
    outputs: &[TensorId],
    inputs: &[&[f32]],
    want: &[Vec<f64>],
    budget: ErrorBudget,
    iters: usize,
) -> (Backend, Option<f64>, Option<f64>) {
    use crate::ir_gpu;
    // WGSL has no fragments and no bf16; it ignores `fragment`
    let wgsl = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let exe = ir_gpu::Executable::new(wgpu, g.to_wgsl(outputs), inputs);
        let err = ir_gpu::gpu_error(wgpu, cache, &exe, outputs, want);
        budget.admits(&err).then(|| {
            (0..3)
                .map(|_| ir_gpu::time_program(wgpu, cache, &exe, outputs[0], iters))
                .fold(f64::INFINITY, f64::min)
        })
    }))
    .ok()
    .flatten();
    let prog = g.to_msl(outputs);
    let msl = Executable::fits(dev, &prog)
        .then(|| {
            let exe = Executable::new(dev, prog, inputs);
            let err = gpu_error(dev, &exe, outputs, want);
            budget.admits(&err).then(|| {
                (0..3)
                    .map(|_| exe.time(dev, iters))
                    .fold(f64::INFINITY, f64::min)
            })
        })
        .flatten();
    let backend = match (wgsl, msl) {
        (Some(w), Some(m)) if w <= m => Backend::Wgsl,
        (Some(_), None) => Backend::Wgsl,
        _ => Backend::Msl,
    };
    (backend, wgsl, msl)
}
