//! One runner for tensor-IR programs on every backend.
//!
//! A [`TensorGraph`] lowers to a [`WgslProgram`] (kernels + buffer specs)
//! for wgpu (`to_wgsl`, or MSL passthrough) or for CUDA (`to_cuda`: mma
//! tensor-core tiles plus transpiled WGSL). [`Runtime`] picks the backend
//! once — CUDA when the `cuda` feature is on and an NVIDIA device with
//! nvrtc is present, wgpu otherwise — and [`Program`] owns the buffers of
//! one lowered program on that backend.
//!
//! # Interop
//!
//! A program lives entirely on its runtime's backend: every buffer it
//! reads or writes is allocated there. Programs on the same runtime hand
//! tensors to each other with device-side copies ([`Program::copy`]);
//! crossing backends (or reaching the CPU) is an explicit host round trip
//! ([`Program::read`] / [`Program::write`]). On wgpu, [`Program::wgpu_buffer`]
//! exposes the buffers to tang-gpu's hand kernels exactly as before; on
//! CUDA the hand kernels (`llm.rs`, `nn.rs`, …) do not run, so a code path
//! that wants CUDA must be expressed as IR programs end to end — the
//! derived training step and Adam are (`tang-bench/examples/train_loop.rs`).
//! There is no zero-copy Vulkan/CUDA sharing.
//!
//! # Memory plans
//!
//! By default every materialized tensor gets its own allocation. A
//! [`BufferPlan`] maps the program's buffers onto fewer slots (tensors
//! with disjoint lifetimes sharing one allocation); [`Program::with_plan`]
//! allocates the slots instead and runs the kernels in the plan's order.
//! `TensorGraph::plan_memory`'s [`MemoryPlan`] converts directly
//! ([`BufferPlan::from_memory_plan`], [`Program::planned`]). On CUDA a
//! plan may update in place (an output sharing its input's slot); wgpu
//! cannot bind one buffer read-only and read-write in the same dispatch,
//! so there such plans are refused.
//!
//! # CUDA graphs
//!
//! On a runtime from [`Runtime::cuda_capturable`] (its own stream,
//! cudarc's event tracking off), [`Program::capture`] records one run as
//! a CUDA graph; later [`Program::run`]s replay it.

use std::cell::{RefCell, RefMut};

use tang_compile::tensor::memplan::MemoryPlan;
#[cfg(any(feature = "cuda", feature = "opencl"))]
use tang_compile::tensor::precision::{bf16_bits, bf16_from_bits};
use tang_compile::tensor::precision::{f16_bits, f16_from_bits};
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{Format, TensorGraph, TensorId};

use crate::{GpuBuffer, GpuDevice, GpuError, KernelCache};

#[cfg(feature = "cuda")]
use crate::cuda::CudaDevice;
#[cfg(feature = "opencl")]
use crate::opencl::{ClDevice, ClKernels, ClMem};

/// Which backend a [`Runtime`] runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Wgpu,
    Cuda,
    OpenCl,
}

/// A device plus what running programs on it needs.
pub enum Runtime {
    Wgpu {
        device: GpuDevice,
        cache: RefCell<KernelCache>,
    },
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
    /// OpenCL (feature `opencl`): `src/opencl.rs`.
    #[cfg(feature = "opencl")]
    OpenCl(ClDevice),
}

impl Runtime {
    /// CUDA when compiled with `cuda` and an NVIDIA device with nvrtc is
    /// present, else wgpu. `TANG_BACKEND=wgpu` or `=cuda` forces one.
    pub fn select() -> Result<Self, GpuError> {
        let want = std::env::var("TANG_BACKEND").ok();
        #[cfg(feature = "opencl")]
        if want.as_deref() == Some("opencl") {
            return Self::opencl().ok_or(GpuError::NoAdapter);
        }
        if want.as_deref() != Some("wgpu") {
            #[cfg(feature = "cuda")]
            if let Some(rt) = Self::cuda() {
                return Ok(rt);
            }
            if want.as_deref() == Some("cuda") {
                return Err(GpuError::NoAdapter);
            }
        }
        Self::wgpu()
    }

    pub fn wgpu() -> Result<Self, GpuError> {
        Ok(Self::from_wgpu(GpuDevice::new_sync()?))
    }

    /// Wrap an existing wgpu device (its own kernel cache).
    pub fn from_wgpu(device: GpuDevice) -> Self {
        Self::Wgpu {
            device,
            cache: RefCell::new(KernelCache::new()),
        }
    }

    /// CUDA device 0, if a driver, a device and nvrtc are all present.
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Option<Self> {
        if !CudaDevice::has_nvrtc() {
            return None;
        }
        CudaDevice::new(0).map(Self::Cuda)
    }

    /// CUDA device 0 set up for [`Program::capture`]
    /// ([`CudaDevice::new_capturable`]).
    #[cfg(feature = "cuda")]
    pub fn cuda_capturable() -> Option<Self> {
        if !CudaDevice::has_nvrtc() {
            return None;
        }
        CudaDevice::new_capturable(0).map(Self::Cuda)
    }

    /// A runtime on an existing cudarc context and stream, shared with the
    /// caller ([`CudaDevice::from_stream`]). `None` without nvrtc.
    #[cfg(feature = "cuda")]
    pub fn cuda_from_stream(
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Option<Self> {
        if !CudaDevice::has_nvrtc() {
            return None;
        }
        CudaDevice::from_stream(ctx, stream).map(Self::Cuda)
    }

    /// The first OpenCL GPU, if a driver and a device are present.
    #[cfg(feature = "opencl")]
    pub fn opencl() -> Option<Self> {
        ClDevice::new().map(Self::OpenCl)
    }

    #[cfg(feature = "opencl")]
    pub fn opencl_device(&self) -> Option<&ClDevice> {
        match self {
            Self::OpenCl(d) => Some(d),
            _ => None,
        }
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            Self::Wgpu { .. } => BackendKind::Wgpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendKind::Cuda,
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => BackendKind::OpenCl,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Wgpu { device, .. } => &device.name,
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => &d.name,
            #[cfg(feature = "opencl")]
            Self::OpenCl(d) => &d.name,
        }
    }

    /// The wgpu device and kernel cache, for hand kernels.
    pub fn wgpu_parts(&self) -> Option<(&GpuDevice, RefMut<'_, KernelCache>)> {
        match self {
            Self::Wgpu { device, cache } => Some((device, cache.borrow_mut())),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => None,
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> Option<&CudaDevice> {
        match self {
            Self::Cuda(d) => Some(d),
            _ => None,
        }
    }

    /// `g`'s program for this backend. wgpu: WGSL with the device's caps
    /// (subgroups, native f16), `fragment` nodes as Metal simdgroup-matrix
    /// kernels under MSL passthrough, or native MSL tiles for everything
    /// with `TANG_NATIVE=1`. CUDA: `to_cuda` (tensor cores on).
    pub fn lower(&self, g: &TensorGraph, outputs: &[TensorId]) -> WgslProgram {
        match self {
            Self::Wgpu { device, .. } => {
                let caps = device.wgsl_caps();
                if device.msl_passthrough && std::env::var("TANG_NATIVE").is_ok() {
                    g.to_msl_mma_caps(outputs, caps)
                } else if device.msl_passthrough {
                    // `fragment` nodes run their simdgroup-matrix kernel
                    g.to_wgsl_fragments(outputs, caps)
                } else {
                    g.to_wgsl_caps(outputs, caps)
                }
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => g.to_cuda(outputs),
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => g.to_opencl(outputs),
        }
    }

    /// Submit pending work and wait for it.
    pub fn sync(&self) {
        match self {
            Self::Wgpu { device, cache } => {
                cache.borrow_mut().flush(device);
                device.device.poll(wgpu::PollType::Wait).ok();
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => d.sync(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(d) => d.sync(),
        }
    }
}

/// Program buffers → allocations. `slot[i]` is the slot backing
/// `program.buffers[i]`; each slot is allocated once with `slot_len`
/// f32 words. Tensors sharing a slot must have disjoint live ranges and
/// graph inputs should keep slots of their own (the planner's job).
#[derive(Clone, Debug)]
pub struct BufferPlan {
    pub slot: Vec<usize>,
    pub slot_len: Vec<usize>,
    /// Kernel indices in execution order (a topological order the slot
    /// sharing is valid for).
    pub order: Vec<usize>,
}

impl BufferPlan {
    /// One slot per buffer, program order — what [`Program::new`] uses.
    pub fn identity(program: &WgslProgram) -> Self {
        Self {
            slot: (0..program.buffers.len()).collect(),
            slot_len: program.buffers.iter().map(words).collect(),
            order: (0..program.kernels.len()).collect(),
        }
    }

    /// The slots and order of a liveness plan from
    /// `TensorGraph::plan_memory`, in-place donations included.
    pub fn from_memory_plan(plan: &MemoryPlan) -> Self {
        Self {
            slot: plan.slot.clone(),
            slot_len: plan.slot_words.clone(),
            order: plan.order.clone(),
        }
    }
}

impl From<&MemoryPlan> for BufferPlan {
    fn from(plan: &MemoryPlan) -> Self {
        Self::from_memory_plan(plan)
    }
}

/// f32 words a buffer needs (f16 and bf16 pack two per word, element
/// `i` at half `i`: WGSL's packed words and CUDA's `unsigned short`
/// buffers share the layout).
///
/// Block formats (Q4K, Q5K, Q8x32) are passed in and read back as their
/// raw device words, except that reading a Q8x32 buffer dequantizes it.
fn words(b: &tang_compile::tensor::wgsl::BufferSpec) -> usize {
    b.format.words(b.len).max(1)
}

/// Host length of an input: elements, or raw words for a block format.
fn host_len(b: &tang_compile::tensor::wgsl::BufferSpec) -> usize {
    if b.format.is_block() {
        words(b)
    } else {
        b.len
    }
}

fn check_plan(program: &WgslProgram, inputs: &[&[f32]], plan: &BufferPlan, shared: &[usize]) {
    assert_eq!(plan.slot.len(), program.buffers.len(), "plan size");
    let mut order = plan.order.clone();
    order.sort_unstable();
    assert!(
        order.iter().copied().eq(0..program.kernels.len()),
        "plan order is not a permutation of the kernels"
    );
    for (i, b) in program.buffers.iter().enumerate() {
        assert!(words(b) <= plan.slot_len[plan.slot[i]], "slot too small");
        if let Some(k) = b.input.filter(|k| !shared.contains(k)) {
            assert!(
                inputs[k].is_empty() || inputs[k].len() == host_len(b),
                "input {k} has wrong length"
            );
        }
    }
}

/// One input of a program bound to another program's buffer instead of
/// an allocation of its own ([`Program::with_shared`]): weights and
/// caches shared by a prefill and a decode program.
#[derive(Clone, Copy)]
pub struct Shared<'a> {
    /// Graph input of the program being built.
    pub input: usize,
    /// The program that owns the buffer, and its tensor.
    pub from: &'a Program,
    pub tensor: TensorId,
}

/// A wgpu program recorded once ([`Program::capture`]): pipelines, bind
/// groups and uniforms built, so a run is one compute pass.
struct WgpuRecording {
    passes: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, u32, u32)>,
}

enum Storage {
    Wgpu(Vec<GpuBuffer>),
    #[cfg(feature = "cuda")]
    Cuda {
        /// Reference counted: programs over shared buffers hold the same
        /// slots ([`Program::with_shared`]).
        slots: Vec<std::rc::Rc<cudarc::driver::CudaSlice<f32>>>,
        functions: Vec<cudarc::driver::CudaFunction>,
        /// One captured run ([`Program::capture`]).
        graph: Option<cudarc::driver::CudaGraph>,
        /// The program as launches of kernels and persistent segments
        /// ([`Program::megakernel`]): `run` launches these instead.
        fused: Option<Vec<Item>>,
    },
    #[cfg(feature = "opencl")]
    OpenCl {
        /// Reference counted, as on CUDA (shared buffers).
        slots: Vec<std::rc::Rc<ClMem>>,
        /// One kernel object per kernel, arguments bound at construction.
        kernels: ClKernels,
    },
}

/// One launch of a fused program ([`Program::megakernel`]).
#[cfg(feature = "cuda")]
enum Item {
    Kernel(usize),
    Mega(MegaRun),
}

/// A run of kernels fused into one persistent kernel.
#[cfg(feature = "cuda")]
struct MegaRun {
    function: cudarc::driver::CudaFunction,
    /// Device pointer per slot.
    table: cudarc::driver::CudaSlice<u64>,
    /// Grid barrier state (two words, left zero by every launch).
    sync: cudarc::driver::CudaSlice<u32>,
    grid: u32,
    threads: u32,
    smem: u32,
}

/// A lowered program with its buffers on one backend; reusable across runs.
pub struct Program {
    pub program: WgslProgram,
    storage: Storage,
    /// `slot[i]` backs `program.buffers[i]`.
    slot: Vec<usize>,
    /// Per kernel: slots of its reads then its output.
    args: Vec<Vec<usize>>,
    /// Kernel indices in the order [`run`](Self::run) launches them.
    order: Vec<usize>,
    /// wgpu: the run recorded by [`capture`](Self::capture).
    recording: Option<WgpuRecording>,
}

/// What [`Program::tuned`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct TuneReport {
    /// Kernels with a tensor-core (native) version that kept it.
    pub native_kept: usize,
    /// Kernels with a native version at all.
    pub native_total: usize,
    /// The choices came from the tuning cache file.
    pub from_cache: bool,
}

impl Program {
    /// Lower `g` for `rt` and allocate it; `inputs` in graph input order.
    pub fn compile(rt: &Runtime, g: &TensorGraph, outputs: &[TensorId], inputs: &[&[f32]]) -> Self {
        Self::new(rt, rt.lower(g, outputs), inputs)
    }

    /// Allocate every buffer (one per tensor) and upload the inputs.
    pub fn new(rt: &Runtime, program: WgslProgram, inputs: &[&[f32]]) -> Self {
        let plan = BufferPlan::identity(&program);
        Self::with_plan(rt, program, inputs, &plan)
    }

    /// Allocate the slots of `plan` and upload the inputs; an empty input
    /// is left zeroed, to [`write`](Self::write) later (a large model
    /// streams its weights in without a second host copy).
    pub fn with_plan(
        rt: &Runtime,
        program: WgslProgram,
        inputs: &[&[f32]],
        plan: &BufferPlan,
    ) -> Self {
        Self::with_shared(rt, program, inputs, plan, &[])
    }

    /// [`with_plan`](Self::with_plan) with some inputs bound to other
    /// programs' buffers (same runtime): nothing is allocated or uploaded
    /// for them (their `inputs` entries may be empty), and writes either
    /// program makes — an in-place cache update — are seen by both. The
    /// shared tensor must have the input's length and format.
    pub fn with_shared(
        rt: &Runtime,
        program: WgslProgram,
        inputs: &[&[f32]],
        plan: &BufferPlan,
        shared: &[Shared<'_>],
    ) -> Self {
        let ks: Vec<usize> = shared.iter().map(|s| s.input).collect();
        check_plan(&program, inputs, plan, &ks);
        // slot of this program -> (program, its slot)
        let mut borrowed: Vec<Option<(&Program, usize)>> = vec![None; plan.slot_len.len()];
        for sh in shared {
            let i = program
                .buffers
                .iter()
                .position(|b| b.input == Some(sh.input))
                .expect("shared input not bound");
            let (b, theirs) = (
                &program.buffers[i],
                &sh.from.program.buffers[sh.from.buffer_index(sh.tensor)],
            );
            assert_eq!(
                (words(b), b.format),
                (words(theirs), theirs.format),
                "shared input {} differs from its buffer",
                sh.input
            );
            borrowed[plan.slot[i]] = Some((sh.from, sh.from.slot_of(sh.tensor)));
        }
        let own = |k: usize| !ks.contains(&k);
        match rt {
            Runtime::Wgpu { device, .. } => {
                Self::wgpu_with_plan(device, program, inputs, plan, &borrowed, &own)
            }
            #[cfg(feature = "cuda")]
            Runtime::Cuda(d) => {
                use std::rc::Rc;
                let mut slots: Vec<Rc<_>> = plan
                    .slot_len
                    .iter()
                    .zip(&borrowed)
                    .map(|(&n, b)| match b {
                        Some((p, s)) => match &p.storage {
                            Storage::Cuda { slots, .. } => slots[*s].clone(),
                            _ => panic!("programs on different backends"),
                        },
                        None => Rc::new(d.stream.alloc_zeros::<f32>(n).expect("alloc")),
                    })
                    .collect();
                for (i, b) in program.buffers.iter().enumerate() {
                    if let Some(k) = b.input.filter(|&k| own(k) && !inputs[k].is_empty()) {
                        let host = pack_words(b.format, inputs[k]);
                        let slot = Rc::get_mut(&mut slots[plan.slot[i]]).expect("own slot");
                        let mut view = slot.slice_mut(0..host.len());
                        d.stream.memcpy_htod(&host[..], &mut view).expect("upload");
                    }
                }
                // nvrtc the missing kernels on all cores, then load
                crate::cuda::precompile(program.kernels.iter().map(|k| k.source.as_str()), d.arch);
                let functions = program
                    .kernels
                    .iter()
                    .map(|k| d.function(&k.source))
                    .collect();
                let storage = Storage::Cuda {
                    slots,
                    functions,
                    graph: None,
                    fused: None,
                };
                Self::assemble(program, storage, plan)
            }
            #[cfg(feature = "opencl")]
            Runtime::OpenCl(d) => {
                use std::rc::Rc;
                let slots: Vec<Rc<ClMem>> = plan
                    .slot_len
                    .iter()
                    .zip(&borrowed)
                    .map(|(&n, b)| match b {
                        Some((p, s)) => match &p.storage {
                            Storage::OpenCl { slots, .. } => slots[*s].clone(),
                            _ => panic!("programs on different backends"),
                        },
                        None => Rc::new(d.alloc_zeros(n)),
                    })
                    .collect();
                for (i, b) in program.buffers.iter().enumerate() {
                    if let Some(k) = b.input.filter(|&k| own(k) && !inputs[k].is_empty()) {
                        d.write(&slots[plan.slot[i]], &pack_words(b.format, inputs[k]));
                    }
                }
                let kernels = d.kernels(program.kernels.iter().map(|k| k.source.as_str()));
                let storage = Storage::OpenCl { slots, kernels };
                let p = Self::assemble(program, storage, plan);
                if let Storage::OpenCl { slots, kernels } = &p.storage {
                    for (i, a) in p.args.iter().enumerate() {
                        let bufs: Vec<&ClMem> = a.iter().map(|&s| &*slots[s]).collect();
                        d.set_args(kernels, i, &bufs, p.program.kernels[i].points);
                    }
                }
                p
            }
        }
    }

    /// Lower nothing: allocate `program` with the slots and order of a
    /// [`MemoryPlan`] (in-place donations need CUDA).
    pub fn planned(
        rt: &Runtime,
        program: WgslProgram,
        inputs: &[&[f32]],
        plan: &MemoryPlan,
    ) -> Self {
        Self::with_plan(rt, program, inputs, &BufferPlan::from_memory_plan(plan))
    }

    /// [`Program::new`] on a bare wgpu device, for callers that keep
    /// their own device and [`KernelCache`] (run it with
    /// [`dispatch`](Self::dispatch)).
    pub fn on_wgpu(device: &GpuDevice, program: WgslProgram, inputs: &[&[f32]]) -> Self {
        let plan = BufferPlan::identity(&program);
        check_plan(&program, inputs, &plan, &[]);
        let borrowed = vec![None; plan.slot_len.len()];
        Self::wgpu_with_plan(device, program, inputs, &plan, &borrowed, &|_| true)
    }

    fn wgpu_with_plan(
        device: &GpuDevice,
        program: WgslProgram,
        inputs: &[&[f32]],
        plan: &BufferPlan,
        borrowed: &[Option<(&Program, usize)>],
        own: &dyn Fn(usize) -> bool,
    ) -> Self {
        let mut slots: Vec<GpuBuffer> = plan
            .slot_len
            .iter()
            .zip(borrowed)
            .map(|(&n, b)| match b {
                Some((p, s)) => match &p.storage {
                    Storage::Wgpu(slots) => GpuBuffer {
                        buffer: slots[*s].buffer.clone(),
                        len: slots[*s].len,
                    },
                    #[cfg(any(feature = "cuda", feature = "opencl"))]
                    _ => panic!("programs on different backends"),
                },
                None => GpuBuffer::uninit(device, n),
            })
            .collect();
        for (i, b) in program.buffers.iter().enumerate() {
            let Some(k) = b.input.filter(|&k| own(k) && !inputs[k].is_empty()) else {
                continue;
            };
            let s = plan.slot[i];
            // a whole-slot input is uploaded as its own buffer, as the old
            // executor did
            let fresh = match b.format {
                Format::F16 => GpuBuffer::from_u32_slice(device, &pack_f16(inputs[k])),
                Format::F32 => GpuBuffer::from_slice(device, inputs[k]),
                Format::BF16 => unreachable!("bf16 is interpreter-only"),
                Format::Q4K | Format::Q5K | Format::Q8x32 => {
                    panic!("block formats run only on the CUDA int8 lowering")
                }
                _ => unreachable!("bf16 and fp8 are not wgpu formats"),
            };
            if fresh.len() == plan.slot_len[s] {
                slots[s] = fresh;
            } else {
                write_wgpu(device, &slots[s], b.format, inputs[k]);
            }
        }
        let p = Self::assemble(program, Storage::Wgpu(slots), plan);
        for (i, a) in p.args.iter().enumerate() {
            let (out, reads) = a.split_last().unwrap();
            assert!(
                !reads.contains(out),
                "kernel {i} updates a slot in place: wgpu cannot bind one buffer \
                 read-only and read-write in one dispatch (plan without donations)"
            );
        }
        p
    }

    fn assemble(program: WgslProgram, storage: Storage, plan: &BufferPlan) -> Self {
        let mut p = Self {
            program,
            storage,
            slot: plan.slot.clone(),
            args: Vec::new(),
            order: plan.order.clone(),
            recording: None,
        };
        p.args = p
            .program
            .kernels
            .iter()
            .map(|k| {
                k.reads
                    .iter()
                    .chain(std::iter::once(&k.node))
                    .map(|&t| p.slot_of(t))
                    .collect()
            })
            .collect();
        p
    }

    fn buffer_index(&self, t: TensorId) -> usize {
        let t = self.program.storage(t);
        self.program
            .buffers
            .iter()
            .position(|b| b.node == t)
            .expect("tensor has no buffer")
    }

    fn slot_of(&self, t: TensorId) -> usize {
        self.slot[self.buffer_index(t)]
    }

    /// The tensor bound to graph input `k`.
    pub fn input(&self, k: usize) -> TensorId {
        self.program
            .buffers
            .iter()
            .find(|b| b.input == Some(k))
            .expect("input not bound")
            .node
    }

    /// Format of the buffer holding `t`.
    pub fn format(&self, t: TensorId) -> Format {
        self.program.buffers[self.buffer_index(t)].format
    }

    /// The wgpu buffer holding `t`, for tang-gpu's hand kernels.
    pub fn wgpu_buffer(&self, t: TensorId) -> Option<&GpuBuffer> {
        match &self.storage {
            Storage::Wgpu(s) => Some(&s[self.slot_of(t)]),
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => None,
        }
    }

    /// The CUDA buffer holding `t` (packed words of [`Self::format`]; f32
    /// for the default format), so a caller sharing the context
    /// ([`Runtime::cuda_from_stream`]) can copy device-to-device into or
    /// out of the program on its stream instead of round-tripping the host.
    #[cfg(feature = "cuda")]
    pub fn cuda_slice(&self, t: TensorId) -> Option<&cudarc::driver::CudaSlice<f32>> {
        match &self.storage {
            Storage::Cuda { slots, .. } => Some(&slots[self.slot_of(t)]),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// Run every kernel in plan order (one submission on wgpu,
    /// asynchronous launches on the CUDA stream, or one graph launch once
    /// [`capture`](Self::capture)d).
    pub fn run(&self, rt: &Runtime) {
        match rt {
            Runtime::Wgpu { device, cache } => match &self.recording {
                Some(rec) => {
                    let mut cache = cache.borrow_mut();
                    cache.flush(device);
                    let mut enc = device
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("tang-gpu program"),
                            timestamp_writes: None,
                        });
                        for (pipe, bind, x, y) in &rec.passes {
                            pass.set_pipeline(pipe);
                            pass.set_bind_group(0, bind, &[]);
                            pass.dispatch_workgroups(*x, *y, 1);
                        }
                    }
                    device.queue.submit(std::iter::once(enc.finish()));
                }
                None => self.dispatch(device, &mut cache.borrow_mut()),
            },
            #[cfg(feature = "cuda")]
            Runtime::Cuda(d) => {
                if let Storage::Cuda { graph: Some(g), .. } = &self.storage {
                    g.launch().expect("graph launch");
                    return;
                }
                self.launch_all(d);
            }
            #[cfg(feature = "opencl")]
            Runtime::OpenCl(d) => {
                #[cfg(feature = "otel")]
                if crate::opencl::tel::sample() {
                    return self.run_opencl_profiled(d);
                }
                for &i in &self.order {
                    self.run_opencl(d, i, false);
                }
                d.flush();
            }
        }
    }

    /// Enqueue kernel `i` (with `timed`: wait, and return its device ms).
    #[cfg(feature = "opencl")]
    fn run_opencl(&self, d: &ClDevice, i: usize, timed: bool) -> f64 {
        let Storage::OpenCl { kernels, .. } = &self.storage else {
            panic!("program and runtime are on different backends")
        };
        let k = &self.program.kernels[i];
        d.launch(
            kernels,
            i,
            k.count as usize,
            k.workgroup_size as usize,
            timed,
        )
    }

    /// [`run`](Self::run) with every kernel's event kept, then waited on
    /// and reported (telemetry's sampled runs).
    #[cfg(all(feature = "opencl", feature = "otel"))]
    fn run_opencl_profiled(&self, d: &ClDevice) {
        let Storage::OpenCl { kernels, .. } = &self.storage else {
            panic!("program and runtime are on different backends")
        };
        let start = std::time::SystemTime::now();
        let t0 = std::time::Instant::now();
        let evs: Vec<_> = self
            .order
            .iter()
            .map(|&i| {
                let k = &self.program.kernels[i];
                (
                    i,
                    d.launch_event(kernels, i, k.count as usize, k.workgroup_size as usize),
                )
            })
            .collect();
        let enqueue_ns = t0.elapsed().as_nanos() as u64;
        d.flush();
        let times: Vec<_> = evs
            .into_iter()
            .map(|(i, ev)| {
                let (q, s, e) = d.event_times(ev);
                (i, q, s, e)
            })
            .collect();
        crate::opencl::tel::run(start, enqueue_ns, &times);
    }

    #[cfg(feature = "cuda")]
    fn launch_all(&self, d: &CudaDevice) {
        if let Storage::Cuda {
            fused: Some(items), ..
        } = &self.storage
        {
            use cudarc::driver::{LaunchConfig, PushKernelArg};
            for it in items {
                match it {
                    Item::Kernel(i) => self.run_cuda(d, *i),
                    Item::Mega(m) => {
                        let cfg = LaunchConfig {
                            grid_dim: (m.grid, 1, 1),
                            block_dim: (m.threads, 1, 1),
                            shared_mem_bytes: m.smem,
                        };
                        let mut launch = d.stream.launch_builder(&m.function);
                        launch.arg(&m.table);
                        launch.arg(&m.sync);
                        unsafe { launch.launch(cfg) }.expect("megakernel launch");
                    }
                }
            }
            return;
        }
        for &i in &self.order {
            self.run_cuda(d, i);
        }
    }

    /// [`run`](Self::run) on a bare wgpu device and kernel cache: every
    /// kernel as one submission.
    pub fn dispatch(&self, device: &GpuDevice, cache: &mut KernelCache) {
        cache.begin_batch();
        for &i in &self.order {
            self.run_wgpu(device, cache, i);
        }
        // leaving the cache in batch mode would make a later
        // `to_vec_sync` (the hand path's readbacks) wait on work that was
        // never submitted
        cache.end_batch(device);
    }

    /// [`run_kernel`](Self::run_kernel) on a bare wgpu device.
    pub fn dispatch_kernel(&self, device: &GpuDevice, cache: &mut KernelCache, i: usize) {
        self.run_wgpu(device, cache, i);
    }

    /// Fuse the whole program into one persistent kernel
    /// (`tang_compile::tensor::cuda_mega`): every kernel a stage over its
    /// own grid, independent kernels sharing a phase, a grid barrier
    /// between phases (knobs: `MegaOpts`; slot sizes are filled in here),
    /// `per_sm` blocks per SM (0: as many as fit). With `max_blocks > 0`
    /// only kernels of at most that many blocks fuse, in maximal runs
    /// between the bigger ones (which keep their own launch and
    /// occupancy): a kernel boundary is then a per-edge choice, launch or
    /// grid barrier. Returns `(launches, phases, largest grid)`. Later [`run`](Self::run)s (and a
    /// [`capture`](Self::capture) after this) launch it instead of the
    /// kernels. CUDA only.
    #[cfg(feature = "cuda")]
    pub fn megakernel(
        &mut self,
        rt: &Runtime,
        opts: &tang_compile::tensor::cuda_mega::MegaOpts,
        per_sm: u32,
        max_blocks: u32,
    ) -> (usize, usize, u32) {
        use cudarc::driver::DevicePtr;
        use tang_compile::tensor::cuda_mega::{megakernel, Mega, MegaStage};
        let mut opts = opts.clone();
        if opts.threads == 0 {
            opts.threads = 256;
        }
        let threads = opts.threads;
        let Runtime::Cuda(d) = rt else {
            panic!("megakernel needs the CUDA runtime")
        };
        let Storage::Cuda { slots, .. } = &self.storage else {
            unreachable!()
        };
        opts.slot_bytes = slots.iter().map(|s| 4 * s.len()).collect();
        let stage = |i: usize| {
            let k = &self.program.kernels[i];
            MegaStage {
                source: &k.source,
                threads: k.workgroup_size,
                blocks: k.count.div_ceil(k.workgroup_size),
                count: k.points,
                args: self.args[i].clone(),
            }
        };
        // segments: maximal runs of fusible kernels (at most `max_blocks`
        // blocks, a block size dividing `threads`, no dynamic smem); a run
        // of one stays a launch
        let fusible = |i: usize| {
            let k = &self.program.kernels[i];
            let blocks = k.count.div_ceil(k.workgroup_size);
            (max_blocks == 0 || blocks <= max_blocks)
                && threads.is_multiple_of(k.workgroup_size)
                && threads / k.workgroup_size <= 15
                && crate::cuda::dynamic_smem(&k.source) == 0
        };
        let mut runs: Vec<Vec<usize>> = Vec::new();
        for &i in &self.order {
            match runs.last_mut() {
                Some(r) if fusible(i) && fusible(*r.last().unwrap()) => r.push(i),
                _ => runs.push(vec![i]),
            }
        }
        let megas: Vec<Option<Mega>> = runs
            .iter()
            .map(|r| {
                (r.len() > 1)
                    .then(|| megakernel(&r.iter().map(|&i| stage(i)).collect::<Vec<_>>(), &opts))
            })
            .collect();
        if let Ok(p) = std::env::var("TANG_MEGA_DUMP") {
            if let Some(m) = megas.iter().flatten().max_by_key(|m| m.phases.len()) {
                std::fs::write(p, &m.source).ok();
            }
        }
        crate::cuda::precompile(megas.iter().flatten().map(|m| m.source.as_str()), d.arch);
        let sms = d
            .ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .expect("SM count") as u32;
        let Storage::Cuda {
            slots,
            fused,
            graph,
            ..
        } = &mut self.storage
        else {
            unreachable!()
        };
        let mut items = Vec::new();
        let (mut phases, mut grid) = (0, 0);
        for (r, m) in runs.iter().zip(megas) {
            let Some(mega) = m else {
                items.push(Item::Kernel(r[0]));
                continue;
            };
            let function = d.function(&mega.source);
            let fit = function
                .occupancy_max_active_blocks_per_multiprocessor(threads, mega.smem as usize, None)
                .expect("occupancy");
            assert!(fit > 0, "megakernel does not fit on an SM");
            let g = sms * if per_sm == 0 { fit } else { per_sm.min(fit) };
            let ptrs: Vec<u64> = mega
                .table
                .iter()
                .map(|&s| {
                    let (p, _sync) = slots[s].device_ptr(&d.stream);
                    p
                })
                .collect();
            let table = d.stream.clone_htod(&ptrs).expect("pointer table");
            let sync = d
                .stream
                .alloc_zeros::<u32>(2 + 2 * (mega.phases.len() + 1))
                .expect("barrier state");
            phases += mega.phases.len();
            grid = grid.max(g);
            items.push(Item::Mega(MegaRun {
                function,
                table,
                sync,
                grid: g,
                threads,
                smem: mega.smem,
            }));
        }
        let launches = items.len();
        *graph = None;
        *fused = Some(items);
        (launches, phases, grid)
    }

    /// Kernel indices in launch order.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// With `MegaOpts::trace`: nanoseconds of each phase of the last run
    /// (block 0's view).
    #[cfg(feature = "cuda")]
    pub fn mega_trace(&self, rt: &Runtime) -> Vec<f64> {
        let (
            Runtime::Cuda(d),
            Storage::Cuda {
                fused: Some(items), ..
            },
        ) = (rt, &self.storage)
        else {
            return Vec::new();
        };
        let Some(m) = items.iter().find_map(|it| match it {
            Item::Mega(m) => Some(m),
            Item::Kernel(_) => None,
        }) else {
            return Vec::new();
        };
        let words = d.stream.clone_dtoh(&m.sync).expect("read trace");
        let t: Vec<u64> = words[2..]
            .chunks(2)
            .map(|c| c[0] as u64 | (c[1] as u64) << 32)
            .collect();
        t.windows(2)
            .map(|w| w[1].wrapping_sub(w[0]) as f64)
            .collect()
    }

    /// Record one run as a CUDA graph; later [`run`](Self::run)s replay
    /// it (same kernels, same slots — host writes between runs are seen).
    /// Needs a runtime from [`Runtime::cuda_capturable`]. On wgpu: build
    /// every kernel's pipeline, bind group and uniform once, so a run is
    /// one command buffer with one compute pass (the kernels in order).
    pub fn capture(&mut self, rt: &Runtime) {
        #[cfg(feature = "cuda")]
        if let Runtime::Cuda(d) = rt {
            use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
            assert!(
                d.capturable(),
                "capture needs Runtime::cuda_capturable (own stream, no event tracking)"
            );
            if let Storage::Cuda { graph, .. } = &mut self.storage {
                *graph = None;
            }
            d.sync();
            d.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .expect("begin capture");
            self.launch_all(d);
            let g = d
                .stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .expect("end capture");
            if let Storage::Cuda { graph, .. } = &mut self.storage {
                *graph = g;
            }
            return;
        }
        #[allow(irrefutable_let_patterns)]
        if let Runtime::Wgpu { device, cache } = rt {
            self.recording = None;
            let Storage::Wgpu(slots) = &self.storage else {
                panic!("program and runtime are on different backends")
            };
            let mut cache = cache.borrow_mut();
            let passes = self
                .order
                .iter()
                .map(|&i| {
                    let k = &self.program.kernels[i];
                    let bufs: Vec<&GpuBuffer> = self.args[i].iter().map(|&s| &slots[s]).collect();
                    let (pipe, layout) = cache.pipeline(
                        device,
                        &k.source,
                        bufs.len(),
                        &[bufs.len() - 1],
                        k.workgroup_size,
                    );
                    let bind = cache.bind(device, &layout, &bufs, &[k.points, 0, 0, 0]);
                    let groups = k.count.div_ceil(k.workgroup_size);
                    (pipe, bind, groups.min(65535), groups.div_ceil(65535))
                })
                .collect();
            self.recording = Some(WgpuRecording { passes });
        }
    }

    /// Run kernel `i` only (its inputs must already be computed).
    pub fn run_kernel(&self, rt: &Runtime, i: usize) {
        match rt {
            Runtime::Wgpu { device, cache } => self.run_wgpu(device, &mut cache.borrow_mut(), i),
            #[cfg(feature = "cuda")]
            Runtime::Cuda(d) => self.run_cuda(d, i),
            #[cfg(feature = "opencl")]
            Runtime::OpenCl(d) => {
                self.run_opencl(d, i, false);
            }
        }
    }

    fn run_wgpu(&self, device: &GpuDevice, cache: &mut KernelCache, i: usize) {
        #[allow(irrefutable_let_patterns)]
        let Storage::Wgpu(slots) = &self.storage
        else {
            panic!("program and runtime are on different backends")
        };
        let k = &self.program.kernels[i];
        let bufs: Vec<&GpuBuffer> = self.args[i].iter().map(|&s| &slots[s]).collect();
        cache.dispatch_buffers(
            device,
            &k.source,
            &bufs,
            &[bufs.len() - 1],
            &[k.points, 0, 0, 0],
            k.count,
            k.workgroup_size,
        );
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, d: &CudaDevice, i: usize) {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let Storage::Cuda {
            slots, functions, ..
        } = &self.storage
        else {
            panic!("program and runtime are on different backends")
        };
        let k = &self.program.kernels[i];
        let cfg = LaunchConfig {
            grid_dim: (k.count.div_ceil(k.workgroup_size), 1, 1),
            block_dim: (k.workgroup_size, 1, 1),
            shared_mem_bytes: crate::cuda::dynamic_smem(&k.source),
        };
        let mut launch = d.stream.launch_builder(&functions[i]);
        for &s in &self.args[i] {
            launch.arg(&*slots[s]);
        }
        launch.arg(&k.points);
        unsafe { launch.launch(cfg) }.expect("launch");
    }

    /// Read `t` back as f32 (waits for queued work; f16 widened exactly).
    pub fn read(&self, rt: &Runtime, t: TensorId) -> Vec<f32> {
        match (rt, &self.storage) {
            (Runtime::Wgpu { device, cache }, Storage::Wgpu(_)) => {
                cache.borrow_mut().flush(device);
                self.read_wgpu(device, t)
            }
            #[cfg(feature = "cuda")]
            (Runtime::Cuda(d), Storage::Cuda { slots, .. }) => {
                let b = &self.program.buffers[self.buffer_index(t)];
                let n = words(b);
                let mut out = vec![0.0f32; n];
                d.stream
                    .memcpy_dtoh(&slots[self.slot_of(t)].slice(0..n), &mut out)
                    .expect("download");
                d.sync();
                unpack_words(b.format, &out, b.len)
            }
            #[cfg(feature = "opencl")]
            (Runtime::OpenCl(d), Storage::OpenCl { slots, .. }) => {
                let b = &self.program.buffers[self.buffer_index(t)];
                let out = d.read(&slots[self.slot_of(t)], words(b));
                unpack_words(b.format, &out, b.len)
            }
            #[allow(unreachable_patterns)]
            _ => panic!("program and runtime are on different backends"),
        }
    }

    /// [`read`](Self::read) on a bare wgpu device (work queued in a
    /// [`KernelCache`] batch must be flushed first).
    pub fn read_wgpu(&self, device: &GpuDevice, t: TensorId) -> Vec<f32> {
        let b = &self.program.buffers[self.buffer_index(t)];
        let Some(buf) = self.wgpu_buffer(t) else {
            panic!("program is not on wgpu")
        };
        let raw = buf.to_vec_sync(device);
        match b.format {
            Format::F16 => unpack_f16(&raw, b.len),
            _ => raw[..b.len].to_vec(),
        }
    }

    /// Overwrite `t` (usually an input) from the host.
    pub fn write(&self, rt: &Runtime, t: TensorId, data: &[f32]) {
        let b = &self.program.buffers[self.buffer_index(t)];
        assert_eq!(data.len(), b.len, "write length");
        let s = self.slot_of(t);
        match (rt, &self.storage) {
            (Runtime::Wgpu { device, cache }, Storage::Wgpu(slots)) => {
                cache.borrow_mut().flush(device);
                write_wgpu(device, &slots[s], b.format, data);
                // the queue holds a staging copy of every write until a
                // submit retires it: streaming a model's weights in would
                // otherwise keep them all in host memory at once
                device.queue.submit([]);
                device.device.poll(wgpu::PollType::Wait).ok();
            }
            #[cfg(feature = "cuda")]
            (Runtime::Cuda(d), Storage::Cuda { slots, .. }) => {
                use cudarc::driver::DevicePtr;
                let host = pack_words(b.format, data);
                let (dst, _sync) = slots[s].device_ptr(&d.stream);
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(dst, &host[..], d.stream.cu_stream())
                }
                .expect("upload");
                // `data` is borrowed only for this call
                d.sync();
            }
            #[cfg(feature = "opencl")]
            (Runtime::OpenCl(d), Storage::OpenCl { slots, .. }) => {
                d.write(&slots[s], &pack_words(b.format, data));
            }
            #[allow(unreachable_patterns)]
            _ => panic!("program and runtime are on different backends"),
        }
    }

    /// Device-side copy of `src`'s tensor `src_t` into this program's
    /// tensor `dst` (same runtime; queued in order with the kernels).
    pub fn copy(&self, rt: &Runtime, dst: TensorId, src: &Program, src_t: TensorId) {
        let (db, sb) = (
            &self.program.buffers[self.buffer_index(dst)],
            &src.program.buffers[src.buffer_index(src_t)],
        );
        assert_eq!((db.len, db.format), (sb.len, sb.format), "copy shape");
        let (ds, ss) = (self.slot_of(dst), src.slot_of(src_t));
        match (rt, &self.storage, &src.storage) {
            (Runtime::Wgpu { device, cache }, Storage::Wgpu(d), Storage::Wgpu(s)) => {
                let mut cache = cache.borrow_mut();
                if d[ds].len() == s[ss].len() {
                    d[ds].copy_from(device, &mut cache, &s[ss]);
                } else {
                    copy_wgpu(device, &mut cache, &d[ds], &s[ss], words(db));
                }
            }
            #[cfg(feature = "cuda")]
            (
                Runtime::Cuda(dev),
                Storage::Cuda { slots: d, .. },
                Storage::Cuda { slots: s, .. },
            ) => {
                use cudarc::driver::DevicePtr;
                let (dp, _a) = d[ds].device_ptr(&dev.stream);
                let (sp, _b) = s[ss].device_ptr(&dev.stream);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dp,
                        sp,
                        words(db) * 4,
                        dev.stream.cu_stream(),
                    )
                }
                .expect("dtod");
            }
            #[cfg(feature = "opencl")]
            (
                Runtime::OpenCl(dev),
                Storage::OpenCl { slots: d, .. },
                Storage::OpenCl { slots: s, .. },
            ) => dev.copy(&d[ds], &s[ss], words(db)),
            #[allow(unreachable_patterns)]
            _ => panic!("programs and runtime are on different backends"),
        }
    }

    /// Milliseconds per run of the whole program: the minimum over five
    /// batches of `iters` (a shared GPU only ever adds time).
    pub fn time(&self, rt: &Runtime, iters: usize) -> f64 {
        time_with(rt, iters, || self.run(rt))
    }

    /// Milliseconds per launch of kernel `i` alone. On CUDA: the minimum
    /// over `5 * iters` single launches, each bracketed by events — on a
    /// time-sliced GPU a batch of launches almost always straddles another
    /// tenant's slice, a single short launch usually does not.
    pub fn time_kernel(&self, rt: &Runtime, i: usize, iters: usize) -> f64 {
        #[cfg(feature = "opencl")]
        if let Runtime::OpenCl(d) = rt {
            for _ in 0..3 {
                self.run_opencl(d, i, false);
            }
            d.sync();
            return (0..5 * iters)
                .map(|_| self.run_opencl(d, i, true))
                .fold(f64::INFINITY, f64::min);
        }
        #[cfg(feature = "cuda")]
        if let Runtime::Cuda(dev) = rt {
            use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
            for _ in 0..3 {
                self.run_kernel(rt, i);
            }
            rt.sync();
            return (0..5 * iters)
                .map(|_| {
                    let a = dev
                        .stream
                        .record_event(Some(CU_EVENT_DEFAULT))
                        .expect("event");
                    self.run_kernel(rt, i);
                    let b = dev
                        .stream
                        .record_event(Some(CU_EVENT_DEFAULT))
                        .expect("event");
                    b.synchronize().expect("sync");
                    a.elapsed_ms(&b).expect("elapsed") as f64
                })
                .fold(f64::INFINITY, f64::min);
        }
        time_with(rt, iters, || {
            self.run_kernel(rt, i);
            #[allow(irrefutable_let_patterns)]
            if let Runtime::Wgpu { device, cache } = rt {
                cache.borrow_mut().flush(device);
            }
        })
    }
    /// GPU-clock times (ms) of `n` launches of kernel `i` after `warmup`
    /// untimed ones: wgpu timestamp queries around each compute pass, CUDA
    /// events around each launch, OpenCL profiling events. Unlike
    /// [`time_kernel`](Self::time_kernel) nothing host-side (submission,
    /// polling, a busy CPU) lands in the number. `None` where the backend
    /// has no GPU timer (a wgpu adapter without `TIMESTAMP_QUERY`). The
    /// kernel's inputs must already be computed.
    pub fn kernel_samples(
        &self,
        rt: &Runtime,
        i: usize,
        warmup: usize,
        n: usize,
    ) -> Option<Vec<f64>> {
        self.gpu_samples(rt, &[i], warmup, n)
    }

    /// [`kernel_samples`](Self::kernel_samples) for one whole run (every
    /// kernel in plan order, gaps between them included).
    pub fn program_samples(&self, rt: &Runtime, warmup: usize, n: usize) -> Option<Vec<f64>> {
        let order = self.order.clone();
        self.gpu_samples(rt, &order, warmup, n)
    }

    fn gpu_samples(&self, rt: &Runtime, ks: &[usize], warmup: usize, n: usize) -> Option<Vec<f64>> {
        match rt {
            Runtime::Wgpu { device, cache } => {
                if !device.timestamps {
                    return None;
                }
                #[allow(irrefutable_let_patterns)]
                let Storage::Wgpu(slots) = &self.storage
                else {
                    panic!("program and runtime are on different backends")
                };
                let mut cache = cache.borrow_mut();
                cache.flush(device);
                #[allow(irrefutable_let_patterns)]
                let passes: Vec<_> = ks
                    .iter()
                    .map(|&i| {
                        let k = &self.program.kernels[i];
                        let bufs: Vec<&GpuBuffer> =
                            self.args[i].iter().map(|&s| &slots[s]).collect();
                        let (pipe, layout) = cache.pipeline(
                            device,
                            &k.source,
                            bufs.len(),
                            &[bufs.len() - 1],
                            k.workgroup_size,
                        );
                        let bind = cache.bind(device, &layout, &bufs, &[k.points, 0, 0, 0]);
                        let groups = k.count.div_ceil(k.workgroup_size);
                        (pipe, bind, groups.min(65535), groups.div_ceil(65535))
                    })
                    .collect();
                Some(wgpu_pass_times(device, &passes, warmup, n))
            }
            #[cfg(feature = "cuda")]
            Runtime::Cuda(dev) => {
                use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
                for _ in 0..warmup {
                    for &i in ks {
                        self.run_cuda(dev, i);
                    }
                }
                let evs: Vec<_> = (0..n)
                    .map(|_| {
                        let a = dev
                            .stream
                            .record_event(Some(CU_EVENT_DEFAULT))
                            .expect("event");
                        for &i in ks {
                            self.run_cuda(dev, i);
                        }
                        let b = dev
                            .stream
                            .record_event(Some(CU_EVENT_DEFAULT))
                            .expect("event");
                        (a, b)
                    })
                    .collect();
                dev.sync();
                Some(
                    evs.iter()
                        .map(|(a, b)| a.elapsed_ms(b).expect("elapsed") as f64)
                        .collect(),
                )
            }
            #[cfg(feature = "opencl")]
            Runtime::OpenCl(d) => {
                for _ in 0..warmup {
                    for &i in ks {
                        self.run_opencl(d, i, false);
                    }
                }
                d.sync();
                Some(
                    (0..n)
                        .map(|_| ks.iter().map(|&i| self.run_opencl(d, i, true)).sum())
                        .collect(),
                )
            }
        }
    }

    /// Lower `g` and tune it for `rt`: per kernel, the backend-native
    /// lowering is timed against the portable one and the faster kept.
    ///
    /// - CUDA: every tensor-core tile (in each `MMA_SHAPES` block shape)
    ///   against its transpiled-WGSL twin.
    /// - wgpu on Metal (MSL passthrough): every simdgroup-matrix MSL tile
    ///   (`to_msl_mma`, what `TANG_NATIVE=1` forces for all kernels)
    ///   against its WGSL kernel.
    /// - other wgpu adapters have one lowering: [`Program::compile`].
    ///
    /// The choices are stored in `cache` (keyed by the device and every
    /// variant's sources) so the measurement is paid once per chip. The
    /// WGSL *schedule* autotuner (`tang_bench::ir_gpu::autotune_kernels`,
    /// `TuningCache`) is a different search — over graph rewrites, before
    /// lowering — and composes with this one: tune the graph, then lower
    /// it here.
    pub fn tuned(
        rt: &Runtime,
        g: &TensorGraph,
        outputs: &[TensorId],
        inputs: &[&[f32]],
        cache: Option<&std::path::Path>,
    ) -> (Self, TuneReport) {
        match rt {
            Runtime::Wgpu { device, .. } => {
                let variants = vec![g.to_wgsl(outputs), g.to_msl_mma(outputs)];
                let same_bindings = variants[0].kernels.len() == variants[1].kernels.len()
                    && variants[0]
                        .kernels
                        .iter()
                        .zip(&variants[1].kernels)
                        .all(|(a, b)| a.node == b.node && a.reads == b.reads);
                if !device.msl_passthrough || !same_bindings {
                    let program = variants.into_iter().next().unwrap();
                    return (Self::new(rt, program, inputs), TuneReport::default());
                }
                Self::tune_variants(rt, variants, "tang-wgpu-tuning", inputs, cache)
            }
            #[cfg(feature = "cuda")]
            Runtime::Cuda(_) => {
                use tang_compile::tensor::cuda::{MMA_SHAPES, SM90_FLASH_SHAPES, SM90_MMA_SHAPES};
                // variant 0: every kernel transpiled; 1 + v: the
                // tensor-core tiles in block shape MMA_SHAPES[v]; on
                // Hopper, then every pairing of a wgmma tile shape
                // (SM90_MMA_SHAPES) with a flash shape (SM90_FLASH_SHAPES)
                let n = MMA_SHAPES.len()
                    + if matches!(rt, Runtime::Cuda(d) if d.arch == "sm_90a") {
                        SM90_MMA_SHAPES.len() * SM90_FLASH_SHAPES.len()
                    } else {
                        0
                    };
                let sm = match rt {
                    Runtime::Cuda(d) => d
                        .arch
                        .trim_start_matches("sm_")
                        .trim_end_matches('a')
                        .parse()
                        .unwrap_or(0),
                    #[allow(unreachable_patterns)]
                    _ => 0,
                };
                let variants: Vec<WgslProgram> =
                    tang_compile::tensor::cuda::with_cuda_sm(sm, || {
                        std::iter::once(None)
                            .chain((0..n).map(Some))
                            .map(|v| g.to_cuda_variant(outputs, v))
                            .collect()
                    });
                Self::tune_variants(rt, variants, "tang-cuda-tuning", inputs, cache)
            }
            // one lowering on OpenCL
            #[cfg(feature = "opencl")]
            Runtime::OpenCl(_) => (Self::compile(rt, g, outputs, inputs), TuneReport::default()),
        }
    }

    /// Pick a variant per kernel. `variants[0]` is the portable lowering,
    /// the rest native ones with the same kernels and bindings; kernels
    /// whose sources differ are timed, the others take variant 0. Cache
    /// file: a `<tag> <key>` header, then `kernel variant` for every tuned
    /// kernel not on variant 1.
    fn tune_variants(
        rt: &Runtime,
        mut variants: Vec<WgslProgram>,
        tag: &str,
        inputs: &[&[f32]],
        cache: Option<&std::path::Path>,
    ) -> (Self, TuneReport) {
        let n_k = variants[0].kernels.len();
        assert!(variants.iter().all(|p| p.kernels.len() == n_k));
        // per kernel, the variants worth timing: distinct sources only
        let candidates: Vec<Vec<usize>> = (0..n_k)
            .map(|i| {
                let mut c: Vec<usize> = Vec::new();
                for (v, p) in variants.iter().enumerate() {
                    let src = &p.kernels[i].source;
                    if !c.iter().any(|&u| variants[u].kernels[i].source == *src) {
                        c.push(v);
                    }
                }
                c
            })
            .collect();
        let differs: Vec<usize> = (0..n_k).filter(|&i| candidates[i].len() > 1).collect();
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            rt.name().hash(&mut h);
            for p in &variants {
                for k in &p.kernels {
                    k.source.hash(&mut h);
                }
            }
            h.finish()
        };
        let header = format!("{tag} {key:016x}");
        // cache lines: `kernel variant` for every tuned kernel not on variant 1
        let cached: Option<Vec<(usize, usize)>> = cache.and_then(|p| {
            let text = std::fs::read_to_string(p).ok()?;
            let mut lines = text.lines();
            (lines.next()? == header).then(|| {
                lines
                    .filter_map(|l| {
                        let (i, v) = l.trim().split_once(' ')?;
                        Some((i.parse().ok()?, v.parse().ok()?))
                    })
                    .collect()
            })
        });
        let from_cache = cached.is_some();
        let mut choice: Vec<usize> = (0..n_k)
            .map(|i| usize::from(candidates[i].len() > 1))
            .collect();
        match cached {
            Some(v) => {
                for (i, c) in v {
                    if i < n_k && c < variants.len() {
                        choice[i] = c;
                    }
                }
            }
            None => {
                let progs: Vec<Self> = variants
                    .drain(..)
                    .map(|p| Self::new(rt, p, inputs))
                    .collect();
                // every kernel's inputs exist once a whole run has gone through
                for p in &progs {
                    p.run(rt);
                }
                for &i in &differs {
                    let mut best = (f64::INFINITY, 0);
                    for &v in &candidates[i] {
                        let t = progs[v].time_kernel(rt, i, 10);
                        if t < best.0 {
                            best = (t, v);
                        }
                    }
                    choice[i] = best.1;
                }
                if let Some(p) = cache {
                    let mut text = header.clone();
                    for &i in &differs {
                        if choice[i] != 1 {
                            text.push_str(&format!("\n{i} {}", choice[i]));
                        }
                    }
                    text.push('\n');
                    if let Some(dir) = p.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    if let Err(e) = std::fs::write(p, text) {
                        eprintln!("tuning cache: could not write {}: {e}", p.display());
                    }
                }
                variants = progs.into_iter().map(|p| p.program).collect();
            }
        }
        let mut program = variants.remove(0);
        for (i, &c) in choice.iter().enumerate() {
            if c != 0 {
                std::mem::swap(&mut program.kernels[i], &mut variants[c - 1].kernels[i]);
            }
        }
        if std::env::var_os("TANG_TUNE_VERBOSE").is_some() {
            for &i in &differs {
                eprintln!("tune: kernel {i} -> variant {}", choice[i]);
            }
        }
        let report = TuneReport {
            native_kept: differs.iter().filter(|&&i| choice[i] != 0).count(),
            native_total: differs.len(),
            from_cache,
        };
        (Self::new(rt, program, inputs), report)
    }
}

/// `warmup + n` back-to-back compute passes, each running `passes` (one
/// dispatch per entry); the last `n` are bracketed by timestamp writes.
/// Returns their GPU times in ms. Passes go in submissions of at most
/// [`PASSES_PER_SUBMIT`]: wgpu's Metal backend opens a command buffer per
/// pass and a Metal queue has 64, so thousands of passes in one encoder
/// would wait forever for a buffer that is only freed once submitted.
fn wgpu_pass_times(
    device: &GpuDevice,
    passes: &[(wgpu::ComputePipeline, wgpu::BindGroup, u32, u32)],
    warmup: usize,
    n: usize,
) -> Vec<f64> {
    const PASSES_PER_SUBMIT: usize = 32;
    // one query set holds at most 4096 timestamps: chunk long runs
    const PER_SET: usize = 2048;
    let period = device.queue.get_timestamp_period() as f64;
    let new_enc = || {
        device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    };
    let record = |enc: &mut wgpu::CommandEncoder, tw: Option<wgpu::ComputePassTimestampWrites>| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tang-gpu timed"),
            timestamp_writes: tw,
        });
        for (pipe, bind, x, y) in passes {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(*x, *y, 1);
        }
    };
    let mut left = warmup;
    while left > 0 {
        let m = left.min(PASSES_PER_SUBMIT);
        left -= m;
        let mut enc = new_enc();
        for _ in 0..m {
            record(&mut enc, None);
        }
        device.queue.submit(std::iter::once(enc.finish()));
        // short waits: wgpu gives up on one long one
        if device.device.poll(wgpu::PollType::Wait).is_err() {
            return Vec::new();
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut left = n;
    while left > 0 {
        let m = left.min(PER_SET);
        left -= m;
        let qs = device.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("tang-gpu timer"),
            ty: wgpu::QueryType::Timestamp,
            count: 2 * m as u32,
        });
        let bytes = (2 * m * 8) as u64;
        let resolve = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tang-gpu timer resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tang-gpu timer read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut r = 0;
        while r < m {
            let mut enc = new_enc();
            let stop = (r + PASSES_PER_SUBMIT).min(m);
            for q in r..stop {
                record(
                    &mut enc,
                    Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: Some(2 * q as u32),
                        end_of_pass_write_index: Some(2 * q as u32 + 1),
                    }),
                );
            }
            r = stop;
            if r == m {
                enc.resolve_query_set(&qs, 0..2 * m as u32, &resolve, 0);
                enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
            }
            device.queue.submit(std::iter::once(enc.finish()));
            if device.device.poll(wgpu::PollType::Wait).is_err() {
                return Vec::new();
            }
        }
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        if device.device.poll(wgpu::PollType::Wait).is_err() || rx.try_recv() != Ok(true) {
            // a kernel that faulted or hung: no times, so it never wins
            return Vec::new();
        }
        {
            let data = slice.get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            for q in 0..m {
                let dt = ticks[2 * q + 1].saturating_sub(ticks[2 * q]) as f64;
                out.push(dt * period * 1e-6);
            }
        }
        read.unmap();
    }
    out
}

fn time_with(rt: &Runtime, iters: usize, f: impl Fn()) -> f64 {
    for _ in 0..3 {
        f();
    }
    rt.sync();
    (0..5)
        .map(|_| {
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            rt.sync();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// Write `data` at the start of `buf` (which may be a larger slot).
fn write_wgpu(device: &GpuDevice, buf: &GpuBuffer, format: Format, data: &[f32]) {
    match format {
        Format::F16 => {
            let words = pack_f16(data);
            device
                .queue
                .write_buffer(&buf.buffer, 0, bytemuck::cast_slice(&words));
        }
        _ => device
            .queue
            .write_buffer(&buf.buffer, 0, bytemuck::cast_slice(data)),
    }
}

/// Copy the first `words` of `src` into `dst` (buffers of different sizes).
fn copy_wgpu(
    device: &GpuDevice,
    cache: &mut KernelCache,
    dst: &GpuBuffer,
    src: &GpuBuffer,
    words: usize,
) {
    let mut enc = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, 0, (words * 4) as u64);
    cache.enqueue(enc.finish());
}

/// Host values in a buffer's device layout, as f32 words (f16/bf16 two
/// per word; f32 as is).
#[cfg(any(feature = "cuda", feature = "opencl"))]
fn pack_words(format: Format, data: &[f32]) -> std::borrow::Cow<'_, [f32]> {
    let bits = |x: f32| match format {
        Format::BF16 => bf16_bits(x as f64) as u32,
        _ => f16_bits(x as f64) as u32,
    };
    if format.is_fp8() {
        let x: Vec<f64> = data.iter().map(|&v| v as f64).collect();
        return tang_compile::tensor::precision::pack_fp8(format, &x)
            .into_iter()
            .map(f32::from_bits)
            .collect();
    }
    match format {
        Format::F32 => std::borrow::Cow::Borrowed(data),
        f if f.is_block() => std::borrow::Cow::Borrowed(data),
        _ => data
            .chunks(2)
            .map(|p| f32::from_bits(bits(p[0]) | p.get(1).map_or(0, |&x| bits(x) << 16)))
            .collect(),
    }
}

/// Inverse of [`pack_words`]: the first `n` elements, widened to f32.
#[cfg(any(feature = "cuda", feature = "opencl"))]
fn unpack_words(format: Format, words: &[f32], n: usize) -> Vec<f32> {
    match format {
        Format::F32 => words[..n].to_vec(),
        Format::Q8x32 => {
            let bytes: &[u8] = bytemuck::cast_slice(words);
            let sc: &[f32] = bytemuck::cast_slice(&bytes[n..n + n / 32 * 8]);
            (0..n)
                .map(|i| bytes[i] as i8 as f32 * sc[i / 32 * 2])
                .collect()
        }
        Format::Q4K | Format::Q5K => words[..format.words(n)].to_vec(),
        Format::F16 => unpack_f16(words, n),
        f if f.is_fp8() => {
            let w: Vec<u32> = words.iter().map(|x| x.to_bits()).collect();
            tang_compile::tensor::precision::unpack_fp8(f, &w, n)
                .into_iter()
                .map(|v| v as f32)
                .collect()
        }
        _ => words
            .iter()
            .flat_map(|w| {
                let b = w.to_bits();
                [b as u16, (b >> 16) as u16]
            })
            .take(n)
            .map(|h| bf16_from_bits(h) as f32)
            .collect(),
    }
}

/// f32 values → IEEE f16 (round to nearest even), two per u32 word,
/// little end first — the byte layout of a WGSL `array<f16>`.
pub fn pack_f16(v: &[f32]) -> Vec<u32> {
    v.chunks(2)
        .map(|p| {
            let lo = f16_bits(p[0] as f64) as u32;
            let hi = p.get(1).map_or(0, |&x| f16_bits(x as f64) as u32);
            lo | (hi << 16)
        })
        .collect()
}

/// Inverse of [`pack_f16`]: the first `n` halves of `words` (read back as
/// f32 bit patterns), widened to f32.
pub fn unpack_f16(words: &[f32], n: usize) -> Vec<f32> {
    words
        .iter()
        .flat_map(|w| {
            let b = w.to_bits();
            [b as u16, (b >> 16) as u16]
        })
        .take(n)
        .map(|h| f16_from_bits(h) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_graph() -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let a = g.input(&[8, 4]);
        let b = g.input(&[4, 6]);
        let c = g.matmul(a, b).unwrap();
        let (g, outs) = g.substitute_maps(&[c]);
        (g, outs[0])
    }

    fn inputs() -> (Vec<f32>, Vec<f32>) {
        (
            (0..32).map(|i| i as f32 * 0.1 - 1.0).collect(),
            (0..24).map(|i| (i % 5) as f32 * 0.2).collect(),
        )
    }

    #[test]
    fn program_matches_eval() {
        let Ok(rt) = Runtime::select() else { return };
        let (g, c) = matmul_graph();
        let (a, b) = inputs();
        let p = Program::compile(&rt, &g, &[c], &[&a, &b]);
        p.run(&rt);
        let got = p.read(&rt, c);
        let want = g.eval(
            &[
                a.iter().map(|&x| x as f64).collect(),
                b.iter().map(|&x| x as f64).collect(),
            ],
            &[c],
        );
        for (x, y) in got.iter().zip(&want[0]) {
            assert!((*x as f64 - y).abs() < 1e-3, "{x} vs {y}");
        }
    }

    #[test]
    fn write_copy_and_plan() {
        let Ok(rt) = Runtime::select() else { return };
        let (g, c) = matmul_graph();
        let (a, b) = inputs();
        let program = rt.lower(&g, &[c]);
        // all buffers in one oversized slot each: exercises slot != buffer sizes
        let mut plan = BufferPlan::identity(&program);
        for n in &mut plan.slot_len {
            *n += 3;
        }
        let p = Program::with_plan(&rt, program, &[&a, &b], &plan);
        p.run(&rt);
        let first = p.read(&rt, c);
        let q = Program::compile(&rt, &g, &[c], &[&a, &b]);
        q.write(&rt, q.input(0), &vec![0.0; a.len()]);
        q.run(&rt);
        assert!(q.read(&rt, c).iter().all(|&x| x == 0.0));
        q.copy(&rt, q.input(0), &p, p.input(0));
        q.run(&rt);
        assert_eq!(q.read(&rt, c), first);
    }

    /// CUDA: a chain of kernels fused into persistent segments
    /// (`Program::megakernel`), whole and split at big kernels, gives the
    /// same bits as launching them one by one, and stays correct across
    /// runs (the barrier state resets itself).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_shared_context_runs_on_callers_stream() {
        use cudarc::driver::CudaContext;
        if !CudaDevice::has_nvrtc() {
            return;
        }
        let Ok(ctx) = std::panic::catch_unwind(|| CudaContext::new(0)).map(|r| r.ok()) else {
            return;
        };
        let Some(ctx) = ctx else { return };
        // the caller's context and stream, as phyz-gpu would own them
        let stream = ctx.new_stream().unwrap();
        let rt = Runtime::cuda_from_stream(ctx.clone(), stream.clone()).unwrap();
        let d = rt.cuda_device().unwrap();
        assert!(std::sync::Arc::ptr_eq(&d.ctx, &ctx));
        assert!(std::sync::Arc::ptr_eq(&d.stream, &stream));
        let mut g = TensorGraph::new();
        let a = g.input(&[16, 8]);
        let b = g.input(&[8, 4]);
        let out = g.matmul(a, b).unwrap();
        let va: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let vb: Vec<f32> = (0..32).map(|i| 1.0 - i as f32 * 0.03).collect();
        let p = Program::compile(&rt, &g, &[out], &[&va, &vb]);
        p.run(&rt);
        // the caller takes the result device-to-device on its own stream
        let res = p.cuda_slice(out).unwrap();
        let copy = stream.clone_dtod(&res.slice(0..64)).unwrap();
        let got = stream.clone_dtoh(&copy).unwrap();
        assert_eq!(got, p.read(&rt, out));
        for i in 0..16 {
            for j in 0..4 {
                let want: f32 = (0..8).map(|k| va[i * 8 + k] * vb[k * 4 + j]).sum();
                assert!((got[i * 4 + j] - want).abs() < 1e-4, "{i},{j}");
            }
        }
    }

    #[test]
    fn cuda_megakernel_matches_launches() {
        use tang_compile::tensor::cuda_mega::MegaOpts;
        let Some(rt) = Runtime::cuda() else { return };
        let mut g = TensorGraph::new();
        let a = g.input(&[64, 32]);
        let b = g.input(&[32, 48]);
        let c = g.input(&[48, 40]);
        let ab = g.matmul(a, b).unwrap();
        let abc = g.matmul(ab, c).unwrap();
        let ct = g.transpose(c, &[1, 0]).unwrap();
        let out = g.matmul(abc, ct).unwrap();
        let (g, outs) = g.substitute_maps(&[out]);
        let fill = |n: usize, k: f32| {
            (0..n)
                .map(|i| (i as f32 * k).sin() * 0.5)
                .collect::<Vec<f32>>()
        };
        let (va, vb, vc) = (
            fill(64 * 32, 0.37),
            fill(32 * 48, 0.11),
            fill(48 * 40, 0.73),
        );
        // the decode lowering: warp kernels, the kind a megakernel fuses
        let lower = || {
            let dp = g.to_cuda_decode(
                &outs,
                &tang_compile::tensor::cuda_gemv::DecodeOpts::default(),
            );
            Program::new(&rt, dp.program, &[&va, &vb, &vc])
        };
        let want = {
            let p = lower();
            p.run(&rt);
            p.read(&rt, outs[0])
        };
        // all fused; then only the 640-block kernel is small enough, so
        // nothing fuses (a run of one stays a launch)
        for (max_blocks, fused) in [(0, 1), (700, 3)] {
            let mut p = lower();
            let (launches, phases, _) = p.megakernel(&rt, &MegaOpts::default(), 0, max_blocks);
            assert_eq!(launches, fused, "max_blocks {max_blocks}");
            assert_eq!(phases > 0, fused == 1);
            for _ in 0..3 {
                p.run(&rt);
                assert_eq!(p.read(&rt, outs[0]), want, "max_blocks {max_blocks}");
            }
        }
    }

    /// CUDA: f16 and bf16 storage (inputs, an intermediate and the
    /// output, with a half accumulator) matches the emulator.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_half_storage_matches_emulation() {
        use tang::Scalar;
        use tang_compile::tensor::Precision;
        use tang_expr::ExprId;
        let Some(rt) = Runtime::cuda() else { return };
        for f in [Format::F16, Format::BF16] {
            let mut g = TensorGraph::new();
            let a = g.input(&[37, 29]);
            let b = g.input(&[29, 23]);
            let c = g.matmul(a, b).unwrap();
            let d = g
                .map(&[c], |v| {
                    v[0] * ExprId::from_f64(0.5) + ExprId::from_f64(1.0)
                })
                .unwrap();
            let (mut g, outs) = g.substitute_maps(&[d]);
            g.schedule_auto(&outs);
            let d = outs[0];
            let full = Precision {
                storage: f,
                accum: f,
                operand: Format::F32,
            };
            for t in [a, b] {
                g.set_precision(t, Precision::storage(f));
            }
            g.set_precision(d, full);
            let av: Vec<f32> = (0..37 * 29)
                .map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.6)
                .collect();
            let bv: Vec<f32> = (0..29 * 23)
                .map(|i| ((i * 5) % 11) as f32 * 0.1 - 0.5)
                .collect();
            let p = Program::compile(&rt, &g, &[d], &[&av, &bv]);
            assert!(p.program.buffers.iter().any(|b| b.format == f));
            p.run(&rt);
            let got = p.read(&rt, d);
            let want = g.eval_emulated(
                &[
                    av.iter().map(|&x| x as f64).collect(),
                    bv.iter().map(|&x| x as f64).collect(),
                ],
                &[d],
            );
            let scale = want[0].iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let ulp = if f == Format::F16 { 1e-3 } else { 8e-3 };
            for (x, y) in got.iter().zip(&want[0]) {
                assert!(
                    ((*x as f64) - y).abs() <= 4.0 * ulp * scale,
                    "{f:?}: {x} vs {y}"
                );
            }
        }
    }

    /// CUDA: a matmul whose operands are f16 (bf16) runs on m16n8k16
    /// fragments in every block shape and matches the emulator (fp32
    /// accumulation, the output stored in the operands' format or f32);
    /// with an f32 B operand too (converted while staging).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_half_mma_matches_emulation() {
        use tang_compile::tensor::{cuda::MMA_SHAPES, Precision};
        let Some(rt) = Runtime::cuda() else { return };
        for f in [Format::F16, Format::BF16] {
            for (out_f, b_f) in [(Format::F32, f), (f, f), (Format::F32, Format::F32)] {
                let (m, k, n) = (200, 136, 72);
                let mut g = TensorGraph::new();
                let a = g.input(&[m, k]);
                let b = g.input(&[k, n]);
                let c = g.matmul(a, b).unwrap();
                let (mut g, outs) = g.substitute_maps(&[c]);
                g.schedule_auto(&outs);
                let c = outs[0];
                for (t, tf) in [(a, f), (b, b_f)] {
                    g.set_precision(t, Precision::storage(tf));
                }
                g.set_precision(c, Precision::storage(out_f));
                let av: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                let bv: Vec<f32> = (0..k * n)
                    .map(|i| ((i * 5) % 11) as f32 * 0.1 - 0.5)
                    .collect();
                let want = g.eval_emulated(
                    &[
                        av.iter().map(|&x| x as f64).collect(),
                        bv.iter().map(|&x| x as f64).collect(),
                    ],
                    &[c],
                );
                let scale = want[0].iter().fold(0.0f64, |m, x| m.max(x.abs()));
                for v in 0..MMA_SHAPES.len() {
                    let program = g.to_cuda_variant(&[c], Some(v));
                    assert!(
                        program.kernels[0].source.contains("m16n8k16"),
                        "{f:?} variant {v} is not on half mma"
                    );
                    let p = Program::new(&rt, program, &[&av, &bv]);
                    p.run(&rt);
                    let got = p.read(&rt, c);
                    // a mixed tile rounds its f32 operand to the format (the
                    // emulator does not)
                    let tol = if out_f == Format::F32 && b_f == f {
                        1e-5
                    } else if f == Format::F16 {
                        1e-3
                    } else {
                        8e-3
                    };
                    let err = got
                        .iter()
                        .zip(&want[0])
                        .map(|(x, y)| ((*x as f64) - y).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        err <= tol * scale,
                        "{f:?}/{b_f:?}/{out_f:?} v{v}: err {err:e}"
                    );
                }
            }
        }
    }

    /// CUDA: half tiles over transposed operands (B stored [n, k] like a
    /// weight, A stored [k, m] like a weight gradient's activation) are
    /// staged in their native m16n8k16 layouts and match the emulator.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_half_mma_layouts() {
        use tang_compile::tensor::{cuda::MMA_SHAPES, Precision};
        let Some(rt) = Runtime::cuda() else { return };
        let (m, k, n) = (136, 200, 72);
        for (ta, tb) in [(false, true), (true, false), (true, true)] {
            for (f, b_f) in [
                (Format::F16, Format::F16),
                (Format::BF16, Format::BF16),
                (Format::F16, Format::F32),
            ] {
                let mut g = TensorGraph::new();
                let a_shape = if ta { [k, m] } else { [m, k] };
                let a_in = g.input(&a_shape);
                let b_shape = if tb { [n, k] } else { [k, n] };
                let b_in = g.input(&b_shape);
                let a = if ta {
                    g.transpose(a_in, &[1, 0]).unwrap()
                } else {
                    a_in
                };
                let b = if tb {
                    g.transpose(b_in, &[1, 0]).unwrap()
                } else {
                    b_in
                };
                let c = g.matmul(a, b).unwrap();
                let (mut g, outs) = g.substitute_maps(&[c]);
                g.schedule_auto(&outs);
                let c = outs[0];
                g.set_precision(a_in, Precision::storage(f));
                g.set_precision(b_in, Precision::storage(b_f));
                let av: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                let bv: Vec<f32> = (0..k * n)
                    .map(|i| ((i * 5) % 11) as f32 * 0.1 - 0.5)
                    .collect();
                let want = g.eval_emulated(
                    &[
                        av.iter().map(|&x| x as f64).collect(),
                        bv.iter().map(|&x| x as f64).collect(),
                    ],
                    &[c],
                );
                let scale = want[0].iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let tol = if b_f == f { 1e-5 } else { 1e-3 };
                for v in 0..MMA_SHAPES.len() {
                    let program = g.to_cuda_variant(&[c], Some(v));
                    assert!(
                        program.kernels[0].source.contains("m16n8k16"),
                        "{ta}/{tb} {f:?} variant {v} is not on half mma"
                    );
                    let p = Program::new(&rt, program, &[&av, &bv]);
                    p.run(&rt);
                    let got = p.read(&rt, c);
                    let err = got
                        .iter()
                        .zip(&want[0])
                        .map(|(x, y)| ((*x as f64) - y).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        err <= tol * scale,
                        "{ta}/{tb} {f:?}/{b_f:?} v{v}: err {err:e}"
                    );
                }
            }
        }
    }
}
