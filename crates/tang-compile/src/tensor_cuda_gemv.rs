//! CUDA lowering for memory-bound reductions: decode-shaped programs.
//!
//! A decode step (one token, `m = 1`) is a chain of matrix–vector
//! products, short reductions (norms, softmax statistics) and a few
//! in-place updates (the KV cache). The tensor-core tiles of
//! `tensor_cuda.rs` are built for GEMMs; here every byte of weight is
//! read once and the kernel's job is to keep enough of them in flight.
//!
//! [`TensorGraph::to_cuda_decode`] lowers a graph like `to_cuda` does,
//! with three additions:
//!
//! - **Warp reductions.** A reduction (or the epilogue map reading it
//!   point for point) runs as one warp per point group: lanes stride the
//!   flattened reduce space a vector chunk at a time, accumulate with the
//!   comp's monoid, combine through shuffles (and shared memory when a
//!   block's warps split one group's reduction), and run the epilogue on
//!   the result. The body is evaluated generically: every leaf read is an
//!   affine address over the comp's space, inlined maps are recomputed
//!   per element, and leaves are classified per chunk — invariant along
//!   the inner reduce dim (one load per chunk), unit-stride and aligned
//!   (vector loads, `float4` or `uint4`), or anything else (a scalar
//!   load per element). A *register dim* — a short point dim the largest
//!   operand does not depend on (the query heads of a GQA group), or the
//!   weight rows of a GEMV — is unrolled per thread so the shared operand
//!   is loaded once for all of its points.
//! - **Packed storage.** Graph inputs may be stored packed
//!   ([`Packing`]): 4-bit unsigned integers eight to a word, or bf16 two
//!   to a word. The graph sees the logical values (`0..=15`, the bf16
//!   value); the lowering decodes them where it loads them, so a 4-bit
//!   weight's dequantization `q·s + b` is an inlined map evaluated in
//!   registers while the packed words stream in.
//! - **Runtime bounds.** A reduction body (or epilogue) of the form
//!   `select(cond, x, id)` with `id` the monoid's identity, `cond` affine
//!   in the coordinates and in scalars read from buffers, skips the
//!   elements where `cond` is false: a whole point when `cond` does not
//!   read a reduce coord, else the loop bounds on that coord. This is
//!   `map_zero_cond`'s skip with data-dependent bounds — a KV cache of
//!   fixed capacity read only up to the current length. A map that
//!   overwrites its donor input in place (`select(cond, y, old)`, see
//!   [`DecodeOpts::inplace`]) only visits the points where `cond` holds:
//!   the cache row of the current position.
//!
//! Kernels with no lowering here are WGSL transpiled (`wgsl_to_cuda`).
//! Calling convention and naming follow `to_cuda`: reads as `b{j}` in
//! `reads` order, `out`, `count`; sources are canonical.
use std::collections::HashMap;
use std::fmt::Write;

use tang_expr::{ExprGraph, ExprId, Node};

use super::msl_mma::Addr;
use super::tile::emit_scalar;
use super::wgsl::{Target, WgslProgram};
use super::{
    fixed, numel, Arg, Form, IntMma, Monoid, Op, Scan, ScanKind, Stage, TensorGraph, TensorId,
};

/// How a graph input is stored on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// Unsigned 4-bit integers, eight per 32-bit word, element `i` in
    /// bits `4·(i % 8)` of word `i / 8` (MLX's layout). Values `0..=15`.
    U4,
    /// Unsigned 3-bit integers as one little-endian bit stream: element
    /// `i` in bits `3i .. 3i+3` of the words (it straddles two when
    /// `3i % 32 > 29`; 32 elements are exactly 3 words). Values `0..=7`
    /// (the GPTQ 3-bit weights of bench/lowbit).
    U3,
    /// bfloat16, two per word, element `i` in the `i % 2` half (the raw
    /// little-endian bytes of a bf16 array).
    Bf16,
    /// IEEE binary16, two per word, element `i` in the `i % 2` half. An
    /// in-place output over an f16 input (a KV cache) is written as f16
    /// (round to nearest even).
    F16,
}

impl Packing {
    /// 32-bit words holding `len` elements.
    pub fn words(self, len: usize) -> usize {
        match self {
            Packing::U4 => len.div_ceil(8),
            Packing::U3 => (3 * len).div_ceil(32),
            Packing::Bf16 | Packing::F16 => len.div_ceil(2),
        }
    }

    /// The f32 value of the 16-bit `bits` (an unsigned expression).
    fn half(self, bits: &str) -> String {
        match self {
            Packing::Bf16 => format!("__uint_as_float(({bits}) << 16)"),
            Packing::F16 => format!("tang_h2f({bits})"),
            Packing::U4 | Packing::U3 => unreachable!(),
        }
    }

    /// Bytes per element in memory.
    fn bytes(self) -> f64 {
        match self {
            Packing::U4 => 0.5,
            Packing::U3 => 0.375,
            Packing::Bf16 | Packing::F16 => 2.0,
        }
    }

    /// A sub-byte integer weight (its GEMVs stream 32-element chunks).
    fn low_bit(self) -> bool {
        matches!(self, Packing::U4 | Packing::U3)
    }
}

/// Which backend a decode lowering spells its kernels for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecodeTarget {
    Cuda,
    Msl,
    Glsl,
    OpenCl,
}

/// Options of [`TensorGraph::to_cuda_decode`].
#[derive(Clone, Debug, Default)]
pub struct DecodeOpts {
    /// `(graph input index, packing)`: inputs stored packed.
    pub packed: Vec<(usize, Packing)>,
    /// `(output, graph input index)`: outputs that will share the input's
    /// buffer (the runner's memory plan must alias them). A predicated
    /// map `select(cond, y, input)` then only writes where `cond` holds.
    pub inplace: Vec<(TensorId, usize)>,
    /// Register tile along a GEMV's rows (points per thread when no
    /// short invariant dim exists). 0: 2.
    pub rows: usize,
    /// Warps per block. 0: 4.
    pub warps: usize,
    /// Unroll factor of the chunk loop (more loads in flight). 0: 1.
    pub unroll: usize,
    /// Try the tensor-core tile (`cuda_mma_kernel`) first for every
    /// matmul-shaped kernel: prefill, where a weight is read by many
    /// tokens.
    pub mma: bool,
    /// Block shape of those tiles (`MMA_SHAPES` index).
    pub mma_shape: usize,
    /// No tensor-core tiles for many-row f32 contractions (`TANG_STEP_MMA=0`
    /// for this program only): batched decode, where B streams' few rows
    /// were faster on the warp lowering.
    pub no_step_mma: bool,
}

/// What [`TensorGraph::to_cuda_decode`] produced.
pub struct DecodeProgram {
    pub program: WgslProgram,
    /// Per kernel: lowered here (`true`) or transpiled WGSL.
    pub native: Vec<bool>,
}

/// A value read by a body, seen from the comp's iteration space.
#[derive(Clone)]
enum Val {
    /// A buffer element at an affine address.
    Leaf(usize),
    /// A coordinate as a value.
    Iota(Form),
    /// An inlined map: its body over its own args.
    Map {
        body: ExprGraph,
        out: ExprId,
        args: Vec<Val>,
    },
    /// A value already computed into a variable (an inlined reduction
    /// of an in-place update, evaluated before the body).
    Var(String),
    /// Element `addr` of staged producer `i`'s shared-memory copy
    /// (`staged_map`).
    Staged(usize, Addr),
    /// A zero-padded read (`View::guard`) of `inner`: 0 where any
    /// source coordinate `forms[d]` falls outside `0..shape[d]`. Only
    /// the lowerings that set `Lowering::guards` build it.
    Guard {
        inner: Box<Val>,
        forms: Vec<Form>,
        shape: Vec<usize>,
    },
}

#[derive(Clone, PartialEq, Eq)]
struct Leaf {
    /// Storage node of the buffer.
    buf: usize,
    addr: Addr,
    pack: Option<Packing>,
}

/// An affine condition `konst + Σ rt + Σ coefs[d]·x_d > 0`, where `rt`
/// are runtime scalars (one-element reads) with their coefficients.
#[derive(Clone)]
struct Cond {
    coefs: Vec<f64>,
    konst: f64,
    rt: Vec<(f64, usize)>,
}

pub(super) struct Kernel {
    pub source: String,
    pub threads: usize,
    pub blocks: usize,
}

pub(super) const PRELUDE: &str = r#"#define select(f, t, c) ((c) ? (t) : (f))
__device__ __forceinline__ float tang_u4(const float* b, int a) {
    unsigned q = (__ldg((const unsigned*)b + (a >> 3)) >> ((a & 7) * 4)) & 15u;
    return __uint_as_float(0x4b000000u | q) - 8388608.0f;
}
__device__ __forceinline__ float tang_u3(const float* b, int a) {
    const unsigned* w = (const unsigned*)b;
    unsigned bit = 3u * (unsigned)a, o = bit & 31u, lo = __ldg(w + (bit >> 5)) >> o;
    unsigned q = (o > 29u ? lo | (__ldg(w + (bit >> 5) + 1) << (32u - o)) : lo) & 7u;
    return __uint_as_float(0x4b000000u | q) - 8388608.0f;
}
__device__ __forceinline__ float tang_bf16(const float* b, int a) {
    unsigned w = __ldg((const unsigned*)b + (a >> 1));
    return __uint_as_float((a & 1) ? (w & 0xffff0000u) : (w << 16));
}
__device__ __forceinline__ float tang_h2f(unsigned h) {
    float f;
    unsigned short s = (unsigned short)h;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(s));
    return f;
}
__device__ __forceinline__ unsigned tang_f2h(float f) {
    unsigned short s;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(s) : "f"(f));
    return (unsigned)s;
}
__device__ __forceinline__ unsigned tang_f2bf(float f) {
    unsigned b = __float_as_uint(f);
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ float tang_f16(const float* b, int a) {
    unsigned w = __ldg((const unsigned*)b + (a >> 1));
    return tang_h2f((a & 1) ? (w >> 16) : (w & 0xffffu));
}
"#;

impl TensorGraph {
    /// The program as CUDA C++ for a decode step: the reductions as warp
    /// kernels, packed inputs decoded at load, runtime bounds and
    /// in-place updates (see the module docs); everything else WGSL
    /// transpiled. Buffers of packed inputs are sized in words.
    /// The tuning cache's key for the kernel rooted at `root`: what it
    /// computes, not where it sits (layers with the same shapes share it).
    pub fn kernel_sig(&self, root: usize, reads: &[usize]) -> String {
        let n = &self.nodes[root];
        let red = self
            .comp_at(root)
            .map(|c| format!("{:?}", c.red))
            .unwrap_or_default();
        let rd: Vec<String> = reads
            .iter()
            .map(|&r| {
                format!(
                    "{:?}/{:?}",
                    self.nodes[r].shape, self.nodes[r].precision.storage
                )
            })
            .collect();
        super::tune::hash_hex(&format!(
            "{:?}|{:?}|{red}|{:?}|{}",
            n.shape,
            n.precision.storage,
            n.int_mma,
            rd.join(";")
        ))
    }

    pub fn to_cuda_decode(&self, outputs: &[TensorId], opts: &DecodeOpts) -> DecodeProgram {
        self.to_decode(outputs, opts, DecodeTarget::Cuda)
    }

    /// The decode lowering for `target`: the kernels are generated once
    /// (in CUDA spelling) and transliterated for Metal
    /// (`tensor_msl_gemv.rs`); kernels with no lowering here stay WGSL
    /// on Metal and are transpiled on CUDA.
    pub(super) fn to_decode(
        &self,
        outputs: &[TensorId],
        opts: &DecodeOpts,
        target: DecodeTarget,
    ) -> DecodeProgram {
        // OpenCL prefill: matmuls over unpacked operands as register tiles
        // (super::opencl_gemv::prefill_tiles), lowered by the generic tile
        // kernel and transpiled, not as per-token warp GEMVs
        let tiled;
        let mut tiles: Vec<usize> = Vec::new();
        let this = if target == DecodeTarget::OpenCl && opts.mma {
            let (g, t) = super::opencl_gemv::prefill_tiles(self, opts);
            tiled = g;
            tiles = t;
            &tiled
        } else {
            self
        };
        this.to_decode_with(outputs, opts, target, &tiles)
    }

    fn to_decode_with(
        &self,
        outputs: &[TensorId],
        opts: &DecodeOpts,
        target: DecodeTarget,
        tiles: &[usize],
    ) -> DecodeProgram {
        // a k-row delta step's scan, readout and last state as fused
        // kernels (no per-step states; see `tensor_cuda_delta.rs`)
        if target == DecodeTarget::Cuda {
            if let Some(g) = self.fuse_delta_steps(outputs) {
                return g.to_decode_with(outputs, opts, target, tiles);
            }
            // many-row contractions as tensor-core matmuls (attention's
            // p·v in a k-row step; see `tensor_cuda_split.rs`)
            // (not over packed inputs: the tile reads plain f32/f16 only)
            let packed_in: Vec<usize> = opts
                .packed
                .iter()
                .filter_map(|&(k, _)| {
                    (0..self.nodes.len())
                        .find(|&i| matches!(self.nodes[i].op, Op::Input(j) if j == k))
                })
                .collect();
            if !opts.no_step_mma && std::env::var("TANG_STEP_MMA").as_deref() != Ok("0") {
                if let Some(g) = self.split_bilinear_avoiding(outputs, &packed_in) {
                    // factors were appended after their readers: lower the
                    // graph renumbered in dependency order, then give the
                    // program this graph's ids back (new factors keep ids
                    // past this graph's nodes)
                    let (r, old) = g.renumber(outputs);
                    let mut new_of = vec![usize::MAX; g.nodes.len()];
                    for (n, &o) in old.iter().enumerate() {
                        new_of[o] = n;
                    }
                    let nid = |t: TensorId| TensorId(new_of[t.n()] as u32);
                    let oid = |t: TensorId| TensorId(old[t.n()] as u32);
                    let routs: Vec<TensorId> = outputs.iter().map(|&t| nid(t)).collect();
                    let mut ropts = opts.clone();
                    ropts.inplace = opts.inplace.iter().map(|&(t, k)| (nid(t), k)).collect();
                    let mut dp = r.to_decode_with(&routs, &ropts, target, tiles);
                    for k in &mut dp.program.kernels {
                        k.node = oid(k.node);
                        k.reads.iter_mut().for_each(|t| *t = oid(*t));
                    }
                    for b in &mut dp.program.buffers {
                        b.node = oid(b.node);
                    }
                    for (a, b) in &mut dp.program.aliases {
                        *a = oid(*a);
                        *b = oid(*b);
                    }
                    return dp;
                }
            }
        }
        // Block formats (GGUF weights, Q8x32 activations) on CUDA: laid out
        // as f32, then the buffers get their formats back; the int8 GEMV
        // (`IntMma`) reads them and Q8x32 writers store them (see
        // `to_cuda_variant_scheduled`)
        let blocky = |i: usize| self.nodes[i].precision.storage.is_block();
        let any_block = target == DecodeTarget::Cuda && (0..self.nodes.len()).any(blocky);
        let plain;
        let layout = if any_block {
            let mut p = self.clone();
            for n in &mut p.nodes {
                if n.precision.storage.is_block() {
                    n.precision.storage = super::Format::F32;
                }
            }
            plain = p;
            &plain
        } else {
            self
        };
        let mut program = layout.to_program_numbered(outputs, Target::Wgsl);
        if any_block {
            for b in &mut program.buffers {
                b.format = self.nodes[b.node.index()].precision.storage;
            }
        }
        let input_node = |k: usize| {
            (0..self.nodes.len())
                .find(|&i| matches!(self.nodes[i].op, Op::Input(j) if j == k))
                .expect("no such input")
        };
        let packed: HashMap<usize, Packing> = opts
            .packed
            .iter()
            .map(|&(k, p)| (input_node(k), p))
            .collect();
        let inplace: Vec<(usize, usize)> = opts
            .inplace
            .iter()
            .map(|&(t, k)| (self.storage(t).n(), input_node(k)))
            .collect();
        // an in-place output is stored like its donor
        let mut packed = packed;
        for &(o, d) in &inplace {
            if let Some(&p) = packed.get(&d) {
                packed.insert(o, p);
            }
        }
        let mut native = Vec::with_capacity(program.kernels.len());
        for (ki, k) in program.kernels.iter_mut().enumerate() {
            let root = k.node.index();
            let reads: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
            super::tune::set_kernel(ki, Some(self.kernel_sig(root, &reads)));
            if let Op::Scan(Scan {
                kind: ScanKind::Sort { experts },
                ..
            }) = &self.nodes[root].op
            {
                let (src, threads) = self.cuda_sort_kernel(root, *experts, &reads);
                k.source = src;
                k.workgroup_size = threads as u32;
                k.count = threads as u32;
                native.push(true);
                continue;
            }
            if let Op::Scan(Scan {
                kind: ScanKind::Delta(_),
                ..
            }) = &self.nodes[root].op
            {
                let (src, threads, blocks) = self.cuda_delta_kernel(root, &reads);
                k.source = src;
                k.workgroup_size = threads as u32;
                k.count = (threads * blocks) as u32;
                native.push(true);
                continue;
            }
            if target == DecodeTarget::Cuda
                && !opts.no_step_mma
                && std::env::var("TANG_STEP_MMA").as_deref() != Ok("0")
                && self.nodes[root].int_mma.is_none()
                && self.mma_candidate(root)
                && !reads.iter().any(|&r| blocky(r) || packed.contains_key(&r))
            {
                if let Some(src) = self.cuda_mma_kernel(root, &reads) {
                    let threads = super::cuda::src_define(&src, "TANG_THREADS");
                    let tiles = super::cuda::src_define(&src, "TANG_TILES");
                    k.source = src;
                    k.workgroup_size = threads as u32;
                    k.count = (threads * tiles) as u32;
                    native.push(true);
                    continue;
                }
            }
            if any_block {
                if let Some(s) = self.nodes[root].int_mma {
                    // a batched schedule (tensor cores, tiled positions) this
                    // contraction has no kernel for falls back to the decode GEMV
                    let gemv = IntMma {
                        stage: Stage::Gemv,
                        ksplit: 1,
                        pos_tile: 1,
                    };
                    let (src, threads, blocks) = self
                        .cuda_int8_kernel(root, &reads, s)
                        .or_else(|e| {
                            if s == gemv {
                                return Err(e);
                            }
                            if std::env::var("TANG_BATCH_LOG").is_ok() {
                                eprintln!("int8 node {root}: {s:?} -> Gemv ({e})");
                            }
                            self.cuda_int8_kernel(root, &reads, gemv)
                        })
                        .unwrap_or_else(|e| panic!("int8 GEMV schedule on node {root}: {e}"));
                    k.source = src;
                    k.workgroup_size = threads as u32;
                    k.count = (threads * blocks) as u32;
                    native.push(true);
                    continue;
                }
                assert!(
                    !reads.iter().any(|&r| blocky(r)),
                    "to_cuda_decode: node {root} reads a block-format buffer without an int8 GEMV"
                );
                if blocky(root) {
                    // a Q8x32 map reading staged statistics (the norm
                    // that feeds an int8 GEMV): one native kernel
                    let lw = Lowering {
                        g: self,
                        root,
                        guards: Default::default(),
                        reads: &reads,
                        packed: &packed,
                        opts,
                    };
                    if let Some(kern) = lw.staged_map(root) {
                        let kern = lw
                            .with_side_jobs(&kern, &k.writes)
                            .unwrap_or_else(|| panic!("node {root}: side jobs did not lower"));
                        k.source = kern.source.clone();
                        k.workgroup_size = kern.threads as u32;
                        k.count = (kern.blocks * kern.threads) as u32;
                        native.push(true);
                        continue;
                    }
                    assert!(
                        !self.hosts(root),
                        "node {root}: a multi-output kernel needs a native lowering"
                    );
                    let st = self.nodes[root].precision.storage;
                    let src = super::cuda::wgsl_to_cuda(&k.source, k.workgroup_size);
                    k.source = super::cuda_int8::q8x32_writer(
                        &src,
                        k.points as usize,
                        st == super::Format::Q8x32F,
                    )
                    .unwrap_or_else(|| {
                        panic!("node {root}: a Q8x32 store needs a one-thread-per-point kernel")
                    });
                    native.push(false);
                    continue;
                }
            }
            let lw = Lowering {
                g: self,
                root,
                guards: Default::default(),
                reads: &reads,
                packed: &packed,
                opts,
            };
            let donor = inplace.iter().find(|(o, _)| *o == root).map(|&(_, d)| d);
            let mma = || {
                // OpenCL prefill over packed 4-bit weights: the register
                // block with the dequantization at load
                if opts.mma
                    && target == DecodeTarget::OpenCl
                    && std::env::var("TANG_CL_MM").as_deref() != Ok("off")
                    && reads.iter().any(|j| packed.contains_key(j))
                {
                    return self.opencl_mma_kernel_q4(root, &reads, &packed);
                }
                if !opts.mma || reads.iter().any(|j| packed.contains_key(j)) {
                    return None;
                }
                // no matrix-tile lowering in GLSL yet: the warp kernels
                if target == DecodeTarget::Glsl {
                    return None;
                }
                if target == DecodeTarget::OpenCl {
                    // TANG_CL_MM=off: the tiled WGSL fallback only
                    if std::env::var("TANG_CL_MM").as_deref() == Ok("off") {
                        return None;
                    }
                    return self.opencl_mma_kernel(root, &reads);
                }
                if target == DecodeTarget::Msl {
                    return super::msl_gemv::mma_kernel(self, root, &reads);
                }
                let shape = super::cuda::MMA_SHAPES[opts.mma_shape % super::cuda::MMA_SHAPES.len()];
                let source = self.cuda_mma_kernel_shaped(root, &reads, shape)?;
                let define = |name: &str| -> usize {
                    source
                        .lines()
                        .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
                        .and_then(|v| v.trim().parse().ok())
                        .expect("define")
                };
                let (threads, blocks) = (define("TANG_THREADS"), define("TANG_TILES"));
                Some(Kernel {
                    source,
                    threads,
                    blocks,
                })
            };
            let got = match donor {
                Some(d) => lw.inplace_update(root, d),
                // OpenCL prefill: the register-block kernel, else the tiled WGSL one
                None if tiles.contains(&root) => mma(),
                // CUDA only: shared atomics, not in the OpenCL transpile
                None => (target == DecodeTarget::Cuda)
                    .then(|| lw.topk_select(root).or_else(|| lw.topk_warp(root)))
                    .flatten()
                    .or_else(mma)
                    .or_else(|| lw.shared_reduction(root))
                    .or_else(|| lw.split_fused(root))
                    .or_else(|| lw.warp_reduction(root))
                    .or_else(|| lw.staged_map(root))
                    .or_else(|| lw.point_map(root))
                    .or_else(|| lw.hosted_map(root))
                    .or_else(|| lw.thread_reduction(root)),
            };
            // TANG_CL_LOG=1: which lowering each reduction got (prefill)
            if target == DecodeTarget::OpenCl
                && opts.mma
                && std::env::var("TANG_CL_LOG").is_ok_and(|v| v == "1")
                && self
                    .comp_at(root)
                    .is_some_and(|c| c.monoid.is_some() || self.tiled_reduction(root).is_some())
            {
                let what = match &got {
                    Some(k) => k.source.lines().next().unwrap_or("").to_string(),
                    None => "WGSL fallback (tiled or generic)".to_string(),
                };
                let packed_reads = reads.iter().filter(|j| packed.contains_key(j)).count();
                eprintln!(
                    "tang-cl: node {root} {:?} packed reads {packed_reads}, tile {}: {what}",
                    self.nodes[root].shape,
                    tiles.contains(&root)
                );
            }
            if self.hosts(root) {
                assert!(
                    target == DecodeTarget::Cuda && got.is_some(),
                    "node {root}: a multi-output kernel needs the native CUDA lowering"
                );
            }
            match got {
                Some(kern) => {
                    let kern = if target == DecodeTarget::Cuda {
                        lw.with_side_jobs(&kern, &k.writes)
                            .unwrap_or_else(|| panic!("node {root}: side jobs did not lower"))
                    } else {
                        kern
                    };
                    k.source = match target {
                        DecodeTarget::Cuda => kern.source.clone(),
                        DecodeTarget::Msl => {
                            // Metal binds the donor once, read-write, as
                            // `out` (wgpu refuses one buffer bound both ways)
                            if let Some(d) = donor {
                                k.reads.retain(|t| t.index() != d);
                            }
                            let kept: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
                            super::msl_gemv::transliterate(&kern.source, &kept, donor)
                        }
                        DecodeTarget::Glsl => {
                            // as for Metal: the donor is bound once, as `out`
                            if let Some(d) = donor {
                                k.reads.retain(|t| t.index() != d);
                            }
                            let kept: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
                            super::glsl_gemv::transliterate(&kern.source, &kept, donor)
                        }
                        DecodeTarget::OpenCl
                            if kern.source.starts_with(super::opencl_gemv::OPENCL_NATIVE) =>
                        {
                            kern.source
                        }
                        DecodeTarget::OpenCl => {
                            // as for Metal: the donor is read through `out`
                            if let Some(d) = donor {
                                k.reads.retain(|t| t.index() != d);
                            }
                            let kept: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
                            super::opencl_gemv::transliterate(&kern.source, &kept, donor)
                        }
                    };
                    k.workgroup_size = kern.threads as u32;
                    k.count = (kern.blocks * kern.threads) as u32;
                    native.push(true);
                }
                None => {
                    if let Some(&j) = reads.iter().find(|j| packed.contains_key(j)) {
                        panic!(
                            "to_cuda_decode: a kernel with no decode lowering reads packed input {:?}",
                            self.nodes[j].op_input()
                        );
                    }
                    assert!(
                        donor.is_none(),
                        "to_cuda_decode: in-place output is not a predicated update"
                    );
                    if target == DecodeTarget::Cuda {
                        k.source = super::cuda::wgsl_to_cuda(&k.source, k.workgroup_size);
                    }
                    if target == DecodeTarget::OpenCl {
                        k.source = super::opencl::wgsl_to_opencl(&k.source, k.workgroup_size);
                    }
                    native.push(false);
                }
            }
        }
        for b in &mut program.buffers {
            if let Some(p) = packed.get(&b.node.index()) {
                b.len = p.words(b.len);
            }
        }
        super::tune::set_kernel(0, None);
        DecodeProgram {
            program: program.canonical(),
            native,
        }
    }
}

impl TensorGraph {
    /// Schedule: inline a reduction into the map that reads it point for
    /// point when that map reads no other reduction and is its only
    /// reader — the map becomes the reduction kernel's epilogue (a
    /// residual add, an activation over merged gate/up). Unlike
    /// `schedule_inline` this takes reductions over any number of reduce
    /// dims, which the warp kernels handle.
    pub fn schedule_epilogues(&mut self, outputs: &[TensorId]) {
        let live = self.live(outputs);
        let uses = self.use_counts(outputs, &live);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            if c.monoid.is_some() {
                continue;
            }
            let mut reds: Vec<usize> = c
                .args
                .iter()
                .map(|a| a.src.n())
                .filter(|&p| self.comp_at(p).is_some_and(|pc| pc.monoid.is_some()))
                .collect();
            reds.dedup();
            let [p] = reds[..] else { continue };
            let pc = self.comp_at(p).unwrap();
            let pk = pc.points().len();
            if uses[p] != 1
                || is_output(p)
                || self.nodes[p].inline
                || c.points().len() != pk
                || self.epilogue_read(c, p) != Some(super::View::identity(pk, pk).forms)
            {
                continue;
            }
            self.nodes[p].inline = true;
        }
    }

    /// Schedule: recompute a materialized map inside its readers when
    /// every reader is a matrix–vector product over a 4-bit weight
    /// (graph input `u4`; the GEMV is bound by the weight's bytes, so the
    /// map's few loads and flops per element are free: the RMSNorm that
    /// feeds a projection) or an in-place update in `inplace` (which only
    /// evaluates its predicated rows: the roped key of the cache row).
    /// Saves a kernel and a round trip per such map.
    pub fn schedule_decode_maps(
        &mut self,
        outputs: &[TensorId],
        u4: &[usize],
        inplace: &[TensorId],
    ) {
        let live = self.live(outputs);
        let n = self.nodes.len();
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let u4_nodes: Vec<usize> = (0..n)
            .filter(|&i| matches!(self.nodes[i].op, Op::Input(k) if u4.contains(&k)))
            .collect();
        let gemv = |c: &super::Comp| {
            c.monoid.is_some() && c.args.iter().any(|a| u4_nodes.contains(&a.src.n()))
        };
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut reshaped = vec![false; n];
        for i in (0..n).filter(|&i| live[i]) {
            match &self.nodes[i].op {
                Op::Comp(c) => {
                    for a in &c.args {
                        readers[a.src.n()].push(i);
                    }
                }
                Op::Reshape(src) => reshaped[src.n()] = true,
                _ => {}
            }
        }
        for i in (0..n).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            if c.monoid.is_some()
                || c.tuple != 0
                || self.nodes[i].inline
                || is_output(i)
                || reshaped[i]
                || readers[i].is_empty()
                // an epilogue (it holds an inlined reduction) stays put
                || c.args.iter().any(|a| {
                    let p = a.src.n();
                    self.nodes[p].inline && self.comp_at(p).is_some_and(|pc| pc.monoid.is_some())
                })
            {
                continue;
            }
            let ok = readers[i]
                .iter()
                .all(|&r| inplace.iter().any(|t| t.n() == r) || self.comp_at(r).is_some_and(gemv));
            if ok {
                self.nodes[i].inline = true;
            }
        }
    }
}

impl TensorGraph {
    /// Schedule (decode): a small Sum / Max reduction — an RMSNorm's sum
    /// of squares, per-head norm statistics, a scalar gate — read only
    /// through maps (directly or through inlined maps) by maps that are
    /// kernels of their own is marked inline, so each such map is one
    /// `staged_map` kernel that recomputes the statistic per block in
    /// shared memory (`ss` then `x·rsqrt(ss)·w` in one launch, also for a
    /// map stored as Q8x32). Bounded work: at most 2¹⁶ products a
    /// statistic and 2²¹ over all the blocks of a reader. `inplace` are
    /// the in-place cache updates (they visit one row of their longest
    /// dim). Returns how many reductions it inlined.
    pub fn schedule_staged_stats(&mut self, outputs: &[TensorId], inplace: &[TensorId]) -> usize {
        let n = self.nodes.len();
        let live = self.live(outputs);
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in (0..n).filter(|&i| live[i]) {
            match &self.nodes[i].op {
                Op::Comp(c) => {
                    for a in &c.args {
                        if !readers[a.src.n()].contains(&i) {
                            readers[a.src.n()].push(i);
                        }
                    }
                }
                // anything else reads whole buffers: not a staged reader
                Op::Reshape(t) => readers[t.n()].push(usize::MAX),
                Op::Scan(sc) => sc
                    .inputs
                    .iter()
                    .for_each(|t| readers[t.n()].push(usize::MAX)),
                Op::Operator(o) => o
                    .inputs
                    .iter()
                    .for_each(|t| readers[t.n()].push(usize::MAX)),
                _ => {}
            }
        }
        // the kernels a value reaches through inlined maps (`None`: a
        // reader that is not a map)
        fn finals(g: &TensorGraph, readers: &[Vec<usize>], i: usize, out: &mut Vec<usize>) -> bool {
            for &r in &readers[i] {
                if r == usize::MAX {
                    return false;
                }
                let Some(rc) = g.comp_at(r) else { return false };
                if rc.monoid.is_some() || rc.tuple != 0 || g.nodes[r].int_mma.is_some() {
                    return false;
                }
                if g.nodes[r].inline {
                    if !finals(g, readers, r, out) {
                        return false;
                    }
                } else if !out.contains(&r) {
                    out.push(r);
                }
            }
            true
        }
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let mut count = 0;
        for p in (0..n).filter(|&p| live[p]) {
            let Some(pc) = self.comp_at(p) else { continue };
            let kind = pc.monoid.as_ref().map(Monoid::kind);
            if !matches!(kind, Some(super::MonoidKind::Sum | super::MonoidKind::Max))
                || self.nodes[p].inline
                || is_output(p)
                || pc.body_outs.len() != 1
                || pc.tuple != 0
                || pc.red.is_empty()
                || pc.has_gather()
                || numel(&pc.out) > 1024
            {
                continue;
            }
            let mut fs = Vec::new();
            if readers[p].is_empty() || !finals(self, &readers, p, &mut fs) {
                continue;
            }
            let work = numel(&pc.out) * numel(&pc.red);
            let blocks = |f: usize| {
                let out = &self.comp_at(f).unwrap().out;
                let k = numel(out);
                let k = if inplace.iter().any(|t| t.n() == f) {
                    k / out.iter().map(|d| d.n()).max().unwrap_or(1)
                } else {
                    k
                };
                k.div_ceil(256)
            };
            if work > 1 << 16 || fs.iter().any(|&f| work * blocks(f) > 1 << 21) {
                continue;
            }
            self.nodes[p].inline = true;
            count += 1;
        }
        count
    }
}

impl TensorGraph {
    /// Schedule (decode): a small materialized map (≤ 1024 points; a GDN
    /// layer's per-head gates `α = exp(softplus(·)·A)`, `β = σ(·)`) read
    /// only by comps is evaluated inside each reader instead of launched
    /// on its own, when the readers' spaces total at most 2²² points (a
    /// few transcendentals per point are free next to the reader's loads).
    /// Not outputs, in-place updates' operands, maps read through
    /// reshapes or by non-comps, or maps over inlined reductions. Returns
    /// how many maps it inlined.
    pub fn schedule_small_maps(&mut self, outputs: &[TensorId], inplace: &[TensorId]) -> usize {
        let n = self.nodes.len();
        let live = self.live(outputs);
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in (0..n).filter(|&i| live[i]) {
            match &self.nodes[i].op {
                Op::Comp(c) => {
                    for a in &c.args {
                        if !readers[a.src.n()].contains(&i) {
                            readers[a.src.n()].push(i);
                        }
                    }
                }
                Op::Reshape(t) => readers[t.n()].push(usize::MAX),
                Op::Scan(sc) => sc
                    .inputs
                    .iter()
                    .for_each(|t| readers[t.n()].push(usize::MAX)),
                Op::Operator(o) => o
                    .inputs
                    .iter()
                    .for_each(|t| readers[t.n()].push(usize::MAX)),
                _ => {}
            }
        }
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let mut count = 0;
        for i in (0..n).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            if c.monoid.is_some()
                || c.tuple != 0
                || c.width() != 1
                || self.nodes[i].inline
                || is_output(i)
                || numel(&c.out) > 1024
                || self.nodes[i].precision.storage != super::Format::F32
                || readers[i].is_empty()
                || c.args.iter().any(|a| a.view.guard || a.view.gather.is_some())
                // over an inlined reduction (a staged statistic): stays
                || c.args.iter().any(|a| {
                    let p = a.src.n();
                    self.nodes[p].inline || self.comp_at(p).is_some_and(|pc| pc.monoid.is_some() && self.nodes[p].inline)
                })
            {
                continue;
            }
            let ok = readers[i].iter().all(|&r| {
                r != usize::MAX
                    && !inplace.iter().any(|t| t.n() == r)
                    && self.comp_at(r).is_some_and(|rc| {
                        rc.args
                            .iter()
                            .all(|a| a.src.n() != i || !(a.view.guard || a.view.gather.is_some()))
                    })
                    && self.nodes[r].int_mma.is_none()
            });
            let work: usize = readers[i]
                .iter()
                .filter(|&&r| r != usize::MAX)
                .map(|&r| self.comp_at(r).map_or(0, |rc| numel(&rc.space())))
                .sum();
            if ok && work <= 1 << 22 {
                self.nodes[i].inline = true;
                count += 1;
            }
        }
        count
    }
}

impl super::TNode {
    fn op_input(&self) -> Option<usize> {
        match self.op {
            Op::Input(k) => Some(k),
            _ => None,
        }
    }
}

/// A split reduction's combine fused into its partial (`split_fused`):
/// the combine node, the map it is the epilogue of (if any), the parts.
struct Fix {
    c: usize,
    e: Option<usize>,
    parts: usize,
}

struct Lowering<'a> {
    g: &'a TensorGraph,
    /// The kernel's node (what `TNode::host` names).
    root: usize,
    /// Zero-padded leaf reads allowed (`Val::Guard`): set while a
    /// lowering that evaluates every value with `emit_val` builds trees.
    guards: std::cell::Cell<bool>,
    reads: &'a [usize],
    packed: &'a HashMap<usize, Packing>,
    opts: &'a DecodeOpts,
}

/// `(int)(offset + Σ coef·coord)` with the coordinates as C expressions.
fn addr_expr(addr: &Addr, coords: &[String]) -> String {
    let mut e = format!("{}", addr.offset);
    for (c, x) in addr.coefs.iter().zip(coords) {
        match *c {
            0 => {}
            1 => write!(e, " + (int)({x})").unwrap(),
            c => write!(e, " + {c} * (int)({x})").unwrap(),
        }
    }
    e
}

fn form_expr(f: &Form, coords: &[String]) -> String {
    addr_expr(
        &Addr {
            offset: f.offset,
            coefs: f.coefs.clone(),
        },
        coords,
    )
}

fn load(buf: usize, pack: Option<Packing>, a: &str) -> String {
    match pack {
        None => format!("__ldg(b{buf} + ({a}))"),
        Some(Packing::U4) => format!("tang_u4(b{buf}, {a})"),
        Some(Packing::U3) => format!("tang_u3(b{buf}, {a})"),
        Some(Packing::Bf16) => format!("tang_bf16(b{buf}, {a})"),
        Some(Packing::F16) => format!("tang_f16(b{buf}, {a})"),
    }
}

/// `lit` with CUDA float spelling.
fn flit(v: f64) -> String {
    float_literals(&super::wgsl::lit(v))
}

/// Emit `outs` of `g` as `float` statements (CUDA spelling).
fn emit(
    s: &mut String,
    g: &ExprGraph,
    outs: &[ExprId],
    prefix: &str,
    indent: &str,
    var: &dyn Fn(u16) -> String,
) -> Vec<String> {
    let mut body = String::new();
    let vals = emit_scalar(&mut body, g, outs, prefix, var);
    for line in body.lines() {
        writeln!(
            s,
            "{indent}{}",
            float_literals(&line.replacen("let ", "float ", 1))
        )
        .unwrap();
    }
    vals.into_iter().map(|v| float_literals(&v)).collect()
}

/// Float literals get an `f` suffix (single precision arithmetic).
fn float_literals(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let prev_ident = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if c.is_ascii_digit() && !prev_ident {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let mut is_float = false;
            if i < b.len() && b[i] == b'.' {
                is_float = true;
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    is_float = true;
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            out.push_str(&s[start..i]);
            if is_float {
                out.push('f');
            }
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Whether value tree `v` reads point dim `d` (a leaf address or an iota
/// form with a coefficient on it).
fn val_uses(v: &Val, leaves: &[Leaf], d: usize) -> bool {
    match v {
        Val::Leaf(i) => leaves[*i].addr.coefs[d] != 0,
        Val::Iota(f) => f.coefs[d] != 0,
        Val::Map { args, .. } => args.iter().any(|a| val_uses(a, leaves, d)),
        Val::Var(_) => false,
        Val::Staged(_, a) => a.coefs[d] != 0,
        Val::Guard { inner, forms, .. } => {
            val_uses(inner, leaves, d) || forms.iter().any(|f| f.coefs[d] != 0)
        }
    }
}

fn a_guarded(a: &Arg) -> bool {
    a.view.guard || a.view.gather.is_some()
}

/// Per-slot combine of a monoid when every slot is a plain sum or max
/// of its own accumulator and element (`None`: a general combine, which
/// is emitted from the monoid's graph).
fn is_identity_lit(m: &Monoid, slot: usize, v: f64) -> bool {
    let id = m.identity[slot];
    v == id || (id == f64::NEG_INFINITY && v <= -1e30)
}

impl Lowering<'_> {
    fn storage(&self, j: usize) -> usize {
        self.g.storage(TensorId(j as u32)).n()
    }

    fn is_buffer(&self, j: usize) -> bool {
        self.reads.contains(&self.storage(j))
    }

    /// The value tree of `a` (a view over a space of `rank`), leaves
    /// interned in `leaves`.
    fn tree(&self, a: &Arg, rank: usize, leaves: &mut Vec<Leaf>) -> Option<Val> {
        // a guarded read, or a gathered one (data-dependent address, which
        // `flat_addr` cannot express): no decode lowering (WGSL handles it),
        // but a guarded buffer read where `guards` is on
        if a.view.gather.is_some() {
            return None;
        }
        let j = a.src.n();
        if a.view.guard {
            if !self.guards.get() || matches!(self.g.nodes[j].op, Op::Iota(_)) {
                return None;
            }
            let mut view = a.view.clone();
            view.guard = false;
            let inner = self.tree(&Arg { src: a.src, view }, rank, leaves)?;
            return Some(Val::Guard {
                inner: Box::new(inner),
                forms: a.view.forms.clone(),
                shape: fixed(&self.g.nodes[j].shape),
            });
        }
        if let Op::Iota(d) = self.g.nodes[j].op {
            return Some(Val::Iota(a.view.forms[d].clone()));
        }
        if self.is_buffer(j) {
            let buf = self.storage(j);
            let leaf = Leaf {
                buf,
                addr: self.g.flat_addr(a, rank),
                pack: self.packed.get(&buf).copied(),
            };
            let i = leaves.iter().position(|l| *l == leaf).unwrap_or_else(|| {
                leaves.push(leaf);
                leaves.len() - 1
            });
            return Some(Val::Leaf(i));
        }
        let mc = self.g.comp_at(j)?;
        if !self.g.inlined_in(j, self.root)
            || mc.monoid.is_some()
            || mc.width() != 1
            || mc.tuple != 0
        {
            return None;
        }
        let args = mc
            .args
            .iter()
            .map(|s| {
                let view = s.view.compose(&a.view, rank);
                self.tree(&Arg { src: s.src, view }, rank, leaves)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Val::Map {
            body: mc.body.clone(),
            out: mc.body_outs[0],
            args,
        })
    }

    /// `e` of `body` (vars: `args`) as an affine condition. A runtime
    /// scalar is a leaf whose address depends on none of the coordinates
    /// `pk..` (the reduce coords): it is loaded once per point.
    fn cond(
        &self,
        body: &ExprGraph,
        e: ExprId,
        args: &[Val],
        leaves: &[Leaf],
        rank: usize,
        pk: usize,
    ) -> Option<Cond> {
        let nodes = body.nodes_slice();
        let lit = |e: ExprId| nodes[e.index() as usize].as_f64();
        let zero = || Cond {
            coefs: vec![0.0; rank],
            konst: 0.0,
            rt: Vec::new(),
        };
        let scale = |mut c: Cond, k: f64| {
            c.coefs.iter_mut().for_each(|x| *x *= k);
            c.konst *= k;
            c.rt.iter_mut().for_each(|(x, _)| *x *= k);
            c
        };
        match nodes[e.index() as usize] {
            Node::Lit(b) => Some(Cond {
                konst: f64::from_bits(b),
                ..zero()
            }),
            Node::Var(v) => match &args[v as usize] {
                Val::Iota(f) => Some(Cond {
                    coefs: f.coefs.iter().map(|&c| c as f64).collect(),
                    konst: f.offset as f64,
                    rt: Vec::new(),
                }),
                Val::Leaf(i) if leaves[*i].addr.coefs[pk..].iter().all(|&c| c == 0) => Some(Cond {
                    rt: vec![(1.0, *i)],
                    ..zero()
                }),
                Val::Map { body, out, args } => self.cond(body, *out, args, leaves, rank, pk),
                _ => None,
            },
            Node::Add(x, y) => {
                let mut a = self.cond(body, x, args, leaves, rank, pk)?;
                let b = self.cond(body, y, args, leaves, rank, pk)?;
                a.konst += b.konst;
                a.coefs.iter_mut().zip(&b.coefs).for_each(|(p, q)| *p += q);
                a.rt.extend(b.rt);
                Some(a)
            }
            Node::Neg(x) => Some(scale(self.cond(body, x, args, leaves, rank, pk)?, -1.0)),
            Node::Mul(x, y) => match (lit(x), lit(y)) {
                (Some(k), _) => Some(scale(self.cond(body, y, args, leaves, rank, pk)?, k)),
                (_, Some(k)) => Some(scale(self.cond(body, x, args, leaves, rank, pk)?, k)),
                _ => None,
            },
            _ => None,
        }
    }

    /// The runtime part of `c` plus its terms over the coords in `dims`.
    fn cond_base(&self, c: &Cond, leaves: &[Leaf], dims: &[usize], coords: &[String]) -> String {
        let mut e = flit(c.konst);
        for (k, i) in &c.rt {
            let l = &leaves[*i];
            let a = if l.addr.coefs.iter().all(|&c| c == 0) {
                format!("{}", l.addr.offset)
            } else {
                addr_expr(&l.addr, coords)
            };
            write!(e, " + {} * {}", flit(*k), load(l.buf, l.pack, &a)).unwrap();
        }
        for &d in dims {
            if c.coefs[d] != 0.0 {
                write!(e, " + {} * (float)({})", flit(c.coefs[d]), coords[d]).unwrap();
            }
        }
        e
    }

    /// Emit a value tree; leaves through `leaf`.
    fn emit_val(
        &self,
        s: &mut String,
        v: &Val,
        coords: &[String],
        indent: &str,
        ctr: &mut usize,
        leaf: &dyn Fn(usize) -> String,
    ) -> String {
        match v {
            Val::Var(name) => name.clone(),
            Val::Staged(i, a) => format!("sm{i}[{}]", addr_expr(a, coords)),
            Val::Leaf(i) => leaf(*i),
            Val::Guard {
                inner,
                forms,
                shape,
            } => {
                // the value only where in range (its loads would not be)
                let inb: Vec<String> = forms
                    .iter()
                    .zip(shape)
                    .map(|(f, n)| {
                        let x = form_expr(f, coords);
                        format!("(unsigned)({x}) < {n}u")
                    })
                    .collect();
                *ctr += 1;
                let g = format!("g{ctr}_");
                writeln!(s, "{indent}float {g} = (float)0;").unwrap();
                writeln!(s, "{indent}if ({}) {{", inb.join(" && ")).unwrap();
                let deeper = format!("{indent}    ");
                let v = self.emit_val(s, inner, coords, &deeper, ctr, leaf);
                writeln!(s, "{deeper}{g} = {v};").unwrap();
                writeln!(s, "{indent}}}").unwrap();
                g
            }
            Val::Iota(f) => format!("(float)({})", form_expr(f, coords)),
            Val::Map { body, out, args } => {
                let vals: Vec<String> = args
                    .iter()
                    .map(|a| self.emit_val(s, a, coords, indent, ctr, leaf))
                    .collect();
                *ctr += 1;
                let prefix = format!("t{ctr}_");
                emit(s, body, &[*out], &prefix, indent, &|x| {
                    vals[x as usize].clone()
                })[0]
                    .clone()
            }
        }
    }

    /// A reduction whose body reads an inlined reduction (the *shared
    /// producer* `S`) that ignores the comp's last point dim `c`, folded
    /// with an online-softmax monoid whose slot 0 is `c`-free: decode
    /// attention written as one reduction over the keys (`q·k` shared by
    /// every output dim, `V` read once), though nothing here names it.
    ///
    /// One block per point group: all of `c` (`vc` consecutive values per
    /// thread) and all of a *register dim* `r` — a short point dim the
    /// streamed operands ignore and `S` reads (the query heads of a GQA
    /// group: one read of K and V serves all of them). Per chunk of keys:
    ///
    /// 1. `S` for every (r, key): `S`'s args that ignore the key are
    ///    evaluated once per block into shared memory (`q`); `lpk` lanes
    ///    share a (r, key), each over a contiguous, unrolled run of `S`'s
    ///    reduce space, combined through shuffles;
    /// 2. slot 0 of the element per (r, key) and its chunk max `m`; then
    ///    `p = exp(e₀ − m)` in shared memory;
    /// 3. each thread folds `Σ p·g` for its (r, c) slots ≥ 1 and combines
    ///    `(m, Σ p·g)` into its state once per chunk (the monoid's own
    ///    combine).
    ///
    /// Keys past a runtime bound shared by every slot are skipped. Split-KV
    /// is `split_reduction` on the graph (a chunk axis among the points).
    /// `TANG_SHARED_RED=0` disables it, `TANG_SR_VC` sets `vc` (default 2).
    fn shared_reduction(&self, root: usize) -> Option<Kernel> {
        if std::env::var("TANG_SHARED_RED").as_deref() == Ok("0") {
            return None;
        }
        let g = self.g;
        let rc = g.comp_at(root)?;
        let monoid = rc.monoid.as_ref()?;
        if g.nodes[root].inline
            || rc.red.len() != 1
            || monoid.kind() != super::MonoidKind::OnlineSoftmax
        {
            return None;
        }
        let pts = fixed(rc.points());
        let red = fixed(&rc.red);
        let (pk, rank) = (pts.len(), pts.len() + 1);
        let w = rc.width();
        if monoid.width != w || pk < 2 {
            return None;
        }
        let mut sh = None;
        for (ai, a) in rc.args.iter().enumerate() {
            let p = a.src.n();
            if g.nodes[p].inline && g.comp_at(p).is_some_and(|pc| pc.monoid.is_some()) {
                if sh.is_some() {
                    return None;
                }
                sh = Some(ai);
            }
        }
        let ai = sh?;
        let sa = &rc.args[ai];
        let sc = g.comp_at(sa.src.n())?;
        let smon = sc.monoid.as_ref()?;
        if sc.width() != 1 || sc.tuple != 0 || smon.width != 1 || a_guarded(sa) {
            return None;
        }
        let cdim = pk - 1;
        let nc = pts[cdim];
        if sa.view.forms.iter().any(|f| f.coefs[cdim] != 0)
            || !sa.view.forms.iter().any(|f| f.coefs[pk] != 0)
        {
            return None;
        }
        let mut leaves = Vec::new();
        let mut args = Vec::with_capacity(rc.args.len());
        for (i, a) in rc.args.iter().enumerate() {
            if i == ai {
                args.push(Val::Var("sv".into()));
            } else {
                args.push(self.tree(a, rank, &mut leaves)?);
            }
        }
        let spts = sa.view.forms.len();
        let sred = fixed(&sc.red);
        let srank = spts + sred.len();
        let mut sleaves = Vec::new();
        let sargs = sc
            .args
            .iter()
            .map(|a| self.tree(a, srank, &mut sleaves))
            .collect::<Option<Vec<_>>>()?;
        let nred_s: usize = sred.iter().product();
        // slot 0 must not read c
        let uses = |e: ExprId, d: usize| -> bool {
            let nodes = rc.body.nodes_slice();
            let mut stack = vec![e];
            while let Some(x) = stack.pop() {
                match nodes[x.index() as usize] {
                    Node::Var(v) => {
                        if val_uses(&args[v as usize], &leaves, d) {
                            return true;
                        }
                    }
                    Node::Lit(_) => {}
                    Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => stack.extend([a, b]),
                    Node::Neg(a)
                    | Node::Recip(a)
                    | Node::Sqrt(a)
                    | Node::Sin(a)
                    | Node::Floor(a)
                    | Node::Exp2(a)
                    | Node::Log2(a) => stack.push(a),
                    Node::Select(c, a, b) => stack.extend([c, a, b]),
                }
            }
            false
        };
        if uses(rc.body_outs[0], cdim) {
            return None;
        }
        // the register dim: S reads it, the leaves of slots >= 1 that vary
        // along the key do not
        let streamed: Vec<usize> = (0..leaves.len())
            .filter(|&i| leaves[i].addr.coefs[pk] != 0)
            .collect();
        let rdim = (0..cdim).filter(|&d| (2..=8).contains(&pts[d])).find(|&d| {
            sa.view.forms.iter().any(|f| f.coefs[d] != 0)
                && streamed.iter().all(|&i| leaves[i].addr.coefs[d] == 0)
        });
        let nr = rdim.map_or(1, |d| pts[d]);
        // 128 threads; a thread owns `vc` c's (default 8), `ks` lanes split
        // the keys of a chunk
        let vc: usize = std::env::var("TANG_SR_VC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let threads = 128usize;
        if vc == 0
            || !nc.is_multiple_of(vc)
            || nc / vc > threads
            || !threads.is_multiple_of(nc / vc)
        {
            return None;
        }
        let ks = threads / (nc / vc);
        if !ks.is_power_of_two() || ks > 32 {
            return None;
        }
        // a key bound shared by every slot
        let nodes = rc.body.nodes_slice();
        let mut cond_e: Option<ExprId> = None;
        for (slot, &o) in rc.body_outs.iter().enumerate() {
            let Node::Select(c, _, f) = nodes[o.index() as usize] else {
                cond_e = None;
                break;
            };
            let is_id = nodes[f.index() as usize]
                .as_f64()
                .is_some_and(|v| is_identity_lit(monoid, slot, v));
            if !is_id || cond_e.is_some_and(|e| e != c) {
                cond_e = None;
                break;
            }
            cond_e = Some(c);
        }
        let bound = cond_e
            .and_then(|c| self.cond(&rc.body, c, &args, &leaves, rank, pk))
            .filter(|cd| {
                cd.coefs[pk] < 0.0
                    && cd.coefs[cdim] == 0.0
                    && rdim.is_none_or(|d| cd.coefs[d] == 0.0)
            });

        let gdims: Vec<usize> = (0..cdim).filter(|&d| Some(d) != rdim).collect();
        let groups: usize = gdims.iter().map(|&d| pts[d]).product();
        let kc = threads;
        let warps = threads / 32;
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: shared-producer reduction (decode)"
        )
        .unwrap();
        writeln!(
            s,
            "// {groups} groups x {nr} x {nc} points ({vc} per thread), {} reduce, producer reduce {nred_s}",
            red[0]
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {groups}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        s.push_str(PRELUDE);
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(
            s,
            "    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;"
        )
        .unwrap();
        writeln!(s, "    const int grp = blockIdx.x;").unwrap();
        let mut rem = "grp".to_string();
        for &d in gdims.iter().rev() {
            writeln!(s, "    const int p{d} = ({rem}) % {};", pts[d]).unwrap();
            rem = format!("({rem}) / {}", pts[d]);
        }
        // coords with r, c and the key as given expressions
        let mk = |r: &str, c: &str, j: &str| -> Vec<String> {
            let mut v: Vec<String> = (0..cdim)
                .map(|d| {
                    if Some(d) == rdim {
                        r.to_string()
                    } else {
                        format!("p{d}")
                    }
                })
                .collect();
            v.push(c.to_string());
            v.push(j.to_string());
            v
        };
        let pt_dims: Vec<usize> = (0..pk).collect();
        match &bound {
            Some(cd) => writeln!(
                s,
                "    const int hi = min({}, max(0, (int)ceilf(({}) / {})));",
                red[0],
                self.cond_base(cd, &leaves, &pt_dims, &mk("0", "0", "0")),
                flit(-cd.coefs[pk])
            )
            .unwrap(),
            None => writeln!(s, "    const int hi = {};", red[0]).unwrap(),
        }
        writeln!(s, "    __shared__ float s_sh[{nr}][{kc}];").unwrap();
        writeln!(s, "    __shared__ float e_sh[{nr}][{kc}];").unwrap();
        writeln!(s, "    __shared__ float r_sh[{nr}][{warps}];").unwrap();
        writeln!(s, "    __shared__ float m_sh[{nr}];").unwrap();
        writeln!(s, "    float acc[{nr}][{vc}][{w}];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < {nr}; r++)").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int u = 0; u < {vc}; u++) {{").unwrap();
        for (slot, &id) in monoid.identity.iter().enumerate() {
            writeln!(s, "        acc[r][u][{slot}] = {};", flit(id)).unwrap();
        }
        writeln!(s, "    }}").unwrap();

        // S's space coords for (r, key) and a reduce index f
        let sdecomp = {
            let mut t = String::new();
            let mut srem = "f".to_string();
            for d in (0..sred.len()).rev() {
                writeln!(t, "            const int q{d} = ({srem}) % {};", sred[d]).unwrap();
                srem = format!("({srem}) / {}", sred[d]);
            }
            t
        };
        let scoords_for = |r: &str, j: &str| -> Vec<String> {
            let pc = mk(r, "0", j);
            let mut v: Vec<String> = sa
                .view
                .forms
                .iter()
                .map(|f| format!("({})", form_expr(f, &pc)))
                .collect();
            v.extend((0..sred.len()).map(|d| format!("q{d}")));
            v
        };
        let key_dims: Vec<usize> = (0..spts)
            .filter(|&k| sa.view.forms[k].coefs[pk] != 0)
            .collect();
        let hoist: Vec<usize> = (0..sargs.len())
            .filter(|&i| !key_dims.iter().any(|&d| val_uses(&sargs[i], &sleaves, d)))
            .collect();
        let mut ctr = 0usize;
        // a block with no live key writes the identity states and leaves
        // (split-KV chunks past the cache length)
        writeln!(s, "    if (hi <= 0) {{").unwrap();
        writeln!(s, "        for (int r = 0; r < {nr}; r++)").unwrap();
        writeln!(
            s,
            "        for (int cc = tid; cc < {nc}; cc += {threads}) {{"
        )
        .unwrap();
        {
            let oc = mk("r", "cc", "0");
            let mut flat = "0".to_string();
            for d in 0..pk {
                flat = format!("({flat}) * {} + ({})", pts[d], oc[d]);
            }
            writeln!(s, "            const int o = {flat};").unwrap();
        }
        for (slot, &id) in monoid.identity.iter().enumerate() {
            writeln!(s, "            out[o * {w} + {slot}] = {};", flit(id)).unwrap();
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        return;").unwrap();
        writeln!(s, "    }}").unwrap();
        // hoisted producer args, per (r, f)
        for &i in &hoist {
            writeln!(s, "    __shared__ float h_sh{i}[{nr}][{nred_s}];").unwrap();
        }
        if !hoist.is_empty() {
            writeln!(
                s,
                "    for (int t = tid; t < {}; t += {threads}) {{",
                nr * nred_s
            )
            .unwrap();
            writeln!(s, "        const int rr = t / {nred_s}, f = t % {nred_s};").unwrap();
            s.push_str(&sdecomp.replace("            ", "        "));
            let sc0 = scoords_for("rr", "0");
            let sel = |i: usize| {
                let l: &Leaf = &sleaves[i];
                load(l.buf, l.pack, &addr_expr(&l.addr, &sc0))
            };
            for &i in &hoist {
                let v = self.emit_val(&mut s, &sargs[i], &sc0, "        ", &mut ctr, &sel);
                writeln!(s, "        h_sh{i}[rr][f] = {v};").unwrap();
            }
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
        }
        // lanes per (r, key) item: 1 by default (a lane streams its key's
        // whole producer row; few serial rounds per warp), TANG_SR_LPK
        let lpk_want: usize = std::env::var("TANG_SR_LPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let lpk = [32usize, 16, 8, 4, 2, 1]
            .into_iter()
            .find(|&l| l <= lpk_want && nred_s.is_multiple_of(l))
            .unwrap();
        let kpw = 32 / lpk;
        let run = nred_s / lpk;
        writeln!(s, "    for (int j0 = 0; j0 < hi; j0 += {kc}) {{").unwrap();
        // 1. S per (r, key); several items per warp in flight
        writeln!(s, "    #pragma unroll 4").unwrap();
        writeln!(
            s,
            "    for (int it = warp * {kpw} + lane / {lpk}; it < {}; it += {}) {{",
            nr * kc,
            warps * kpw
        )
        .unwrap();
        writeln!(s, "        const int rr = it / {kc}, kk = it % {kc};").unwrap();
        writeln!(s, "        const int jk = j0 + kk, sub = lane % {lpk};").unwrap();
        writeln!(s, "        float sa_ = {};", flit(smon.identity[0])).unwrap();
        writeln!(s, "        if (jk < hi) {{").unwrap();
        // leaves contiguous along the producer's innermost reduce dim,
        // f16 or f32: the lane's run preloaded as 16-byte vectors
        let sin = srank - 1;
        let sinner = sred[sred.len() - 1];
        let vleaf = |l: &Leaf| -> Option<usize> {
            let per = match l.pack {
                Some(Packing::F16) | Some(Packing::Bf16) => 8,
                None => 4,
                _ => return None,
            };
            let aligned = l.addr.offset % per as i64 == 0
                && l.addr
                    .coefs
                    .iter()
                    .enumerate()
                    .all(|(d, &c)| d == sin || c % per as i64 == 0);
            // linear over the flattened reduce index: a run may span dims
            let mut stride = 1i64;
            let mut linear = true;
            for d in (0..sred.len()).rev() {
                linear &= l.addr.coefs[spts + d] == stride;
                stride *= sred[d] as i64;
            }
            let _ = (sin, sinner);
            (linear && run.is_multiple_of(per) && aligned).then_some(per)
        };
        let vec_leaves: Vec<(usize, usize)> = (0..sleaves.len())
            .filter_map(|i| vleaf(&sleaves[i]).map(|p| (i, p)))
            .collect();
        if !vec_leaves.is_empty() {
            writeln!(s, "        const int f = sub * {run};").unwrap();
            s.push_str(&sdecomp.replace("            ", "        "));
            let scj = scoords_for("rr", "jk");
            for &(i, per) in &vec_leaves {
                let l = &sleaves[i];
                let a = addr_expr(&l.addr, &scj);
                let ty = if per == 8 { "uint4" } else { "float4" };
                writeln!(s, "        {ty} pv{i}[{}];", run / per).unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(
                    s,
                    "        for (int v = 0; v < {}; v++) pv{i}[v] = __ldg((const {ty}*)(b{}) + (({a}) >> {}) + v);",
                    run / per,
                    l.buf,
                    if per == 8 { 3 } else { 2 }
                )
                .unwrap();
            }
        }
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int fi = 0; fi < {run}; fi++) {{").unwrap();
        writeln!(s, "            const int f = sub * {run} + fi;").unwrap();
        s.push_str(&sdecomp);
        {
            let scj = scoords_for("rr", "jk");
            let sel = |i: usize| {
                let l: &Leaf = &sleaves[i];
                match vec_leaves.iter().find(|v| v.0 == i) {
                    Some(&(_, 8)) => l.pack.unwrap().half(&format!(
                        "(((&pv{i}[fi >> 3].x)[(fi >> 1) & 3] >> (16 * (fi & 1))) & 0xffffu)"
                    )),
                    Some(_) => format!("(&pv{i}[fi >> 2].x)[fi & 3]"),
                    None => load(l.buf, l.pack, &addr_expr(&l.addr, &scj)),
                }
            };
            let svals: Vec<String> = sargs
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    if hoist.contains(&i) {
                        format!("h_sh{i}[rr][f]")
                    } else {
                        self.emit_val(&mut s, a, &scj, "            ", &mut ctr, &sel)
                    }
                })
                .collect();
            let se = emit(
                &mut s,
                &sc.body,
                &sc.body_outs,
                "x_",
                "            ",
                &|v| svals[v as usize].clone(),
            );
            self.combine(&mut s, smon, 1, "            ", |_| "sa_".to_string(), &se);
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(
            s,
            "        for (int off = {}; off > 0; off >>= 1) {{",
            lpk / 2
        )
        .unwrap();
        writeln!(
            s,
            "            float o0 = __shfl_xor_sync(0xffffffffu, sa_, off);"
        )
        .unwrap();
        self.combine(
            &mut s,
            smon,
            1,
            "            ",
            |_| "sa_".to_string(),
            &["o0".to_string()],
        );
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        if (sub == 0) s_sh[rr][kk] = sa_;").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        // 2. slot 0 per (r, key), its chunk max, p = exp(e0 - m)
        writeln!(s, "    {{").unwrap();
        writeln!(s, "        const int jk = j0 + tid;").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int rr = 0; rr < {nr}; rr++) {{").unwrap();
        writeln!(s, "            float e0 = {};", flit(monoid.identity[0])).unwrap();
        writeln!(s, "            if (jk < hi) {{").unwrap();
        writeln!(s, "                const float sv = s_sh[rr][tid];").unwrap();
        {
            let pc = mk("rr", "0", "jk");
            let el0 = |i: usize| {
                let l: &Leaf = &leaves[i];
                load(l.buf, l.pack, &addr_expr(&l.addr, &pc))
            };
            let v0: Vec<String> = args
                .iter()
                .map(|a| self.emit_val(&mut s, a, &pc, "                ", &mut ctr, &el0))
                .collect();
            let y0 = emit(
                &mut s,
                &rc.body,
                &rc.body_outs[..1],
                "z_",
                "                ",
                &|v| v0[v as usize].clone(),
            );
            writeln!(s, "                e0 = {};", y0[0]).unwrap();
        }
        writeln!(s, "            }}").unwrap();
        writeln!(s, "            e_sh[rr][tid] = e0;").unwrap();
        writeln!(s, "            float mx = e0;").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));").unwrap();
        writeln!(s, "            if (lane == 0) r_sh[rr][warp] = mx;").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        writeln!(s, "    if (tid < {nr}) {{").unwrap();
        writeln!(s, "        float mx = r_sh[tid][0];").unwrap();
        writeln!(
            s,
            "        for (int k = 1; k < {warps}; k++) mx = fmaxf(mx, r_sh[tid][k]);"
        )
        .unwrap();
        writeln!(s, "        m_sh[tid] = mx;").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int rr = 0; rr < {nr}; rr++) e_sh[rr][tid] = __expf(e_sh[rr][tid] - m_sh[rr]);").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        // 3. Σ p·g per (r, c): a thread owns `vc` consecutive c, `ks`
        // neighbouring lanes split the chunk's keys and add their partial
        // sums through shuffles (every term shares the chunk max); one
        // combine per chunk
        writeln!(s, "    const int jn = min({kc}, hi - j0);").unwrap();
        writeln!(s, "    const int kq = tid % {ks}, cg = tid / {ks};").unwrap();
        writeln!(s, "    float gs[{nr}][{vc}][{}];", w - 1).unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < {nr}; r++)").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int u = 0; u < {vc}; u++)").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(
            s,
            "    for (int k = 0; k < {}; k++) gs[r][u][k] = 0.0f;",
            w - 1
        )
        .unwrap();
        writeln!(s, "    #pragma unroll 4").unwrap();
        writeln!(s, "    for (int jj = kq; jj < jn; jj += {ks}) {{").unwrap();
        writeln!(s, "        const int jk = j0 + jj;").unwrap();
        // leaves unit-stride along c, f16: one 16-byte load for the vc c's
        let cvec: Vec<usize> = (0..leaves.len())
            .filter(|&i| {
                let l = &leaves[i];
                vc == 8
                    && l.pack == Some(Packing::F16)
                    && l.addr.coefs[cdim] == 1
                    && l.addr.offset % 8 == 0
                    && l.addr
                        .coefs
                        .iter()
                        .enumerate()
                        .all(|(d, &c)| d == cdim || c % 8 == 0)
            })
            .collect();
        {
            let pc0 = mk("0", &format!("(cg * {vc})"), "jk");
            for &i in &cvec {
                let l = &leaves[i];
                writeln!(
                    s,
                    "        const uint4 cv{i} = __ldg((const uint4*)(b{}) + (({}) >> 3));",
                    l.buf,
                    addr_expr(&l.addr, &pc0)
                )
                .unwrap();
            }
        }
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int u = 0; u < {vc}; u++) {{").unwrap();
        writeln!(s, "        const int cc = cg * {vc} + u;").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int r = 0; r < {nr}; r++) {{").unwrap();
        writeln!(s, "            const float sv = s_sh[r][jj];").unwrap();
        writeln!(s, "            const float pj = e_sh[r][jj];").unwrap();
        {
            let pc = mk("r", "cc", "jk");
            let el = |i: usize| {
                let l: &Leaf = &leaves[i];
                if cvec.contains(&i) {
                    Packing::F16.half(&format!(
                        "(((&cv{i}.x)[u >> 1] >> (16 * (u & 1))) & 0xffffu)"
                    ))
                } else {
                    load(l.buf, l.pack, &addr_expr(&l.addr, &pc))
                }
            };
            let vals: Vec<String> = args
                .iter()
                .map(|a| self.emit_val(&mut s, a, &pc, "            ", &mut ctr, &el))
                .collect();
            let y = emit(
                &mut s,
                &rc.body,
                &rc.body_outs[1..],
                "y_",
                "            ",
                &|v| vals[v as usize].clone(),
            );
            for (k, yk) in y.iter().enumerate() {
                writeln!(
                    s,
                    "            gs[r][u][{k}] = fmaf(pj, {yk}, gs[r][u][{k}]);"
                )
                .unwrap();
            }
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < {nr}; r++)").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int u = 0; u < {vc}; u++) {{").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int k = 0; k < {}; k++) {{", w - 1).unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int off = {}; off > 0; off >>= 1) gs[r][u][k] += __shfl_xor_sync(0xffffffffu, gs[r][u][k], off);", ks / 2).unwrap();
        writeln!(s, "        }}").unwrap();
        let mut elem = vec!["m_sh[r]".to_string()];
        elem.extend((0..w - 1).map(|k| format!("gs[r][u][{k}]")));
        self.combine(
            &mut s,
            monoid,
            w,
            "        ",
            |k| format!("acc[r][u][{k}]"),
            &elem,
        );
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    if (tid % {ks} != 0) return;").unwrap();
        writeln!(s, "    const int cg = tid / {ks};").unwrap();
        // out: flat point index over (gdims..., r, c) in the comp's order
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < {nr}; r++)").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int u = 0; u < {vc}; u++) {{").unwrap();
        writeln!(s, "        const int cc = cg * {vc} + u;").unwrap();
        let oc = mk("r", "cc", "0");
        let mut flat = "0".to_string();
        for d in 0..pk {
            flat = format!("({flat}) * {} + ({})", pts[d], oc[d]);
        }
        writeln!(s, "        const int o = {flat};").unwrap();
        for slot in 0..w {
            writeln!(s, "        out[o * {w} + {slot}] = acc[r][u][{slot}];").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads,
            blocks: groups,
        })
    }

    /// A reduction (or a map that is its epilogue) as warp kernels.
    fn warp_reduction(&self, root: usize) -> Option<Kernel> {
        self.warp_reduction_fix(root, None)
    }

    /// A split reduction (`split_reduction`: a partial over `[parts, …]`
    /// and a combine summing the parts, perhaps the epilogue of a map)
    /// as one kernel: the partial's warp reduction writes its part to a
    /// module-scope scratch, and the warp that finishes a point group's
    /// last part (a per-group arrival counter) sums the parts and runs
    /// the combine and its epilogue. The combine is a dependent launch
    /// otherwise. Taken when the schedule inlines the partial
    /// (`TANG_SPLIT_FUSE=1` in the Qwen decode schedule).
    fn split_fused(&self, root: usize) -> Option<Kernel> {
        if super::tune::knob("TANG_SPLIT_FUSE").as_deref() == Some("0") {
            return None;
        }
        let g = self.g;
        let rc = g.comp_at(root)?;
        let is_combine = |c: &super::Comp| -> Option<usize> {
            let m = c.monoid.as_ref()?;
            if m.kind() != super::MonoidKind::Sum
                || c.red.len() != 1
                || c.tuple != 0
                || c.body_outs.len() != 1
                || c.args.len() != 1
            {
                return None;
            }
            let p = c.args[0].src.n();
            let pc = g.comp_at(p)?;
            let k = c.points().len();
            // the split's view: (part, points…) from (points…, part)
            let mut forms = vec![super::Form::unit(k + 1, k)];
            forms.extend((0..k).map(|d| super::Form::unit(k + 1, d)));
            (g.nodes[p].inline
                && pc.monoid.is_some()
                && pc.tuple == 0
                && c.args[0].view.forms == forms
                && !c.args[0].view.guard
                && c.args[0].view.gather.is_none()
                && pc.out[0].n() == c.red[0].n())
            .then_some(p)
        };
        let (c, e) = if rc.monoid.is_some() {
            (root, None)
        } else {
            let cs: Vec<usize> = rc
                .args
                .iter()
                .map(|a| a.src.n())
                .filter(|&p| {
                    g.nodes[p].inline && g.comp_at(p).is_some_and(|pc| pc.monoid.is_some())
                })
                .collect();
            let [c] = cs[..] else { return None };
            let cc = g.comp_at(c)?;
            let k = cc.points().len();
            if rc.points().len() != k
                || g.epilogue_read(rc, c)? != super::View::identity(k, k).forms
            {
                return None;
            }
            (c, Some(root))
        };
        let cc = g.comp_at(c)?;
        let p = is_combine(cc)?;
        let parts = cc.red[0].n();
        self.warp_reduction_fix(p, Some(&Fix { c, e, parts }))
    }

    fn warp_reduction_fix(&self, root: usize, fix: Option<&Fix>) -> Option<Kernel> {
        let g = self.g;
        let c = g.comp_at(root)?;
        let (red_node, epi) = if c.monoid.is_some() {
            (root, None)
        } else {
            let mut found = None;
            for a in &c.args {
                let p = a.src.n();
                if g.nodes[p].inline && g.comp_at(p).is_some_and(|pc| pc.monoid.is_some()) {
                    if found.is_some_and(|f| f != p) {
                        return None;
                    }
                    found = Some(p);
                }
            }
            let p = found?;
            let pc = g.comp_at(p)?;
            let pk = pc.points().len();
            if c.points().len() != pk
                || g.epilogue_read(c, p)? != super::View::identity(pk, pk).forms
            {
                return None;
            }
            (p, Some(c))
        };
        let rc = g.comp_at(red_node)?;
        let monoid = rc.monoid.as_ref()?;
        let pts = fixed(rc.points());
        let red = fixed(&rc.red);
        let (pk, rank) = (pts.len(), pts.len() + red.len());
        let w = rc.width();
        if red.is_empty() || monoid.width != w {
            return None;
        }
        let mut leaves = Vec::new();
        let args = rc
            .args
            .iter()
            .map(|a| self.tree(a, rank, &mut leaves))
            .collect::<Option<Vec<_>>>()?;
        let nodes = rc.body.nodes_slice();

        // runtime bounds: body `select(c₁, select(c₂, …x…, id), id)` with
        // `id` the identity — every `cᵢ` on that chain: those over point
        // coords skip a point, one bounding the (single) reduce coord
        // from above and one from below set the loop bounds
        let mut point_conds: Vec<Cond> = Vec::new();
        let mut red_cond: Option<Cond> = None;
        let mut red_lo: Option<Cond> = None;
        if w == 1 {
            let mut e = rc.body_outs[0];
            while let Node::Select(cnd, t, f) = nodes[e.index() as usize] {
                if !nodes[f.index() as usize]
                    .as_f64()
                    .is_some_and(|v| is_identity_lit(monoid, 0, v))
                {
                    break;
                }
                if let Some(cd) = self.cond(&rc.body, cnd, &args, &leaves, rank, pk) {
                    let red_dims: Vec<usize> = (pk..rank).filter(|&d| cd.coefs[d] != 0.0).collect();
                    if red_dims.is_empty() {
                        point_conds.push(cd);
                    } else if red_dims == [rank - 1] && red.len() == 1 {
                        let slot = if cd.coefs[rank - 1] < 0.0 {
                            &mut red_cond
                        } else {
                            &mut red_lo
                        };
                        if slot.is_none() {
                            *slot = Some(cd);
                        }
                    }
                }
                e = t;
            }
        }
        // (the first bound found keeps its old name: `hi` or `lo`)
        let (red_cond, red_lo) = match (red_cond, red_lo) {
            (None, lo) => (lo, None),
            (hi, lo) => (hi, lo),
        };
        let point_cond = point_conds.first().cloned();

        // vector width along the inner reduce dim
        let kin = rank - 1;
        let ri = red[red.len() - 1];
        let aligned = |l: &Leaf, a: i64| {
            l.addr.offset % a == 0
                && l.addr
                    .coefs
                    .iter()
                    .enumerate()
                    .all(|(d, &c)| d == kin || c % a == 0)
        };
        let unit = |l: &Leaf| l.addr.coefs[kin] == 1;
        let vec = if leaves
            .iter()
            .any(|l| unit(l) && l.pack.is_some_and(Packing::low_bit) && aligned(l, 32))
            && ri.is_multiple_of(32)
        {
            32
        } else if ri.is_multiple_of(4) && leaves.iter().any(|l| unit(l) && l.pack.is_none()) {
            // wider (TANG_DECODE_VEC=8: 16-byte loads of f16, two float4
            // of f32) when every unit-stride leaf allows it
            let wide: usize = std::env::var("TANG_DECODE_VEC")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4);
            if wide > 4
                && wide.is_power_of_two()
                && ri.is_multiple_of(wide)
                && leaves
                    .iter()
                    .filter(|l| unit(l) && !l.pack.is_some_and(Packing::low_bit))
                    .all(|l| aligned(l, 4))
            {
                wide
            } else {
                4
            }
        } else {
            1
        };
        // classify leaves: 0 hoisted (invariant along k), 1 vector, 2 scalar
        let class: Vec<u8> = leaves
            .iter()
            .map(|l| {
                if l.addr.coefs[kin] == 0 {
                    0
                } else if vec > 1
                    && unit(l)
                    && match l.pack {
                        None => aligned(l, 4),
                        Some(Packing::U4 | Packing::U3) => vec == 32 && aligned(l, 32),
                        Some(Packing::Bf16 | Packing::F16) => aligned(l, 4),
                    }
                {
                    1
                } else {
                    2
                }
            })
            .collect();

        // register dim: short point dim the biggest operand ignores, else
        // the GEMV rows
        let heaviest = (0..leaves.len()).max_by_key(|&i| numel(&g.nodes[leaves[i].buf].shape));
        let mut reg: Option<(usize, usize)> = None;
        // a bound may not depend on the register dim (it is per group)
        let free = |d: usize| {
            point_conds
                .iter()
                .chain(&red_cond)
                .chain(&red_lo)
                .all(|cd| {
                    cd.coefs[d] == 0.0 && cd.rt.iter().all(|&(_, i)| leaves[i].addr.coefs[d] == 0)
                })
        };
        if let Some(h) = heaviest {
            let hl = &leaves[h];
            reg = (0..pk)
                .filter(|&d| (2..=8).contains(&pts[d]) && hl.addr.coefs[d] == 0 && free(d))
                .max_by_key(|&d| pts[d])
                .map(|d| (d, pts[d]));
            if reg.is_none() {
                let rows = if self.opts.rows == 0 {
                    2
                } else {
                    self.opts.rows
                };
                reg = (0..pk)
                    .rev()
                    .find(|&d| pts[d] > 1)
                    .filter(|&d| {
                        rows > 1 && pts[d].is_multiple_of(rows) && hl.addr.coefs[d] != 0 && free(d)
                    })
                    .map(|d| (d, rows));
            }
        }
        // the load model: bytes each lane loads per output point with dim
        // `d` tiled `t` in registers (a leaf that ignores `d` is loaded
        // once for all `t` points); only leaves that vary along the reduce
        // are counted. When another tiling loads a quarter fewer bytes
        // than the rule above (attention's P·V: tiling the heads shares V
        // but re-reads the f32 probabilities per `d`; tiling `d` shares
        // them), take it. `TANG_REG_MODEL=0` keeps the rule.
        if std::env::var("TANG_REG_MODEL").as_deref() != Ok("0") {
            let bytes = |l: &Leaf| l.pack.map_or(4.0, Packing::bytes);
            let streamed: Vec<&Leaf> = leaves
                .iter()
                .filter(|l| (pk..rank).any(|r| l.addr.coefs[r] != 0))
                .collect();
            let cost = |r: Option<(usize, usize)>| -> f64 {
                streamed
                    .iter()
                    .map(|l| match r {
                        Some((d, t)) if l.addr.coefs[d] == 0 => bytes(l) / t as f64,
                        _ => bytes(l),
                    })
                    .sum()
            };
            let rows = if self.opts.rows == 0 {
                2
            } else {
                self.opts.rows
            };
            let best = (0..pk)
                .filter(|&d| pts[d] > 1 && free(d))
                .filter_map(|d| {
                    let t = if (2..=8).contains(&pts[d]) {
                        pts[d]
                    } else if rows > 1 && pts[d].is_multiple_of(rows) {
                        rows
                    } else {
                        return None;
                    };
                    Some((d, t))
                })
                .min_by(|a, b| cost(Some(*a)).total_cmp(&cost(Some(*b))));
            if let Some(b) = best {
                if cost(Some(b)) < 0.75 * cost(reg) {
                    reg = Some(b);
                }
            }
        }
        // a second register dim when it cuts the streamed bytes by a
        // quarter more (attention's P·V: the GQA heads share V, a run of
        // output dims shares the probabilities); 4-bit GEMVs keep one.
        // `TANG_REG2=0` disables it
        let mut reg2: Option<(usize, usize)> = None;
        if let Some((d1, t1)) = reg {
            if super::tune::knob("TANG_REG2").as_deref() != Some("0")
                && !leaves.iter().any(|l| l.pack.is_some_and(Packing::low_bit))
            {
                let bytes = |l: &Leaf| l.pack.map_or(4.0, Packing::bytes);
                let cost = |d2: Option<(usize, usize)>| -> f64 {
                    leaves
                        .iter()
                        .filter(|l| (pk..rank).any(|r| l.addr.coefs[r] != 0))
                        .map(|l| {
                            let mut b = bytes(l);
                            if l.addr.coefs[d1] == 0 {
                                b /= t1 as f64;
                            }
                            if let Some((d, t)) = d2 {
                                if l.addr.coefs[d] == 0 {
                                    b /= t as f64;
                                }
                            }
                            b
                        })
                        .sum()
                };
                let base = cost(None);
                reg2 = (0..pk)
                    .filter(|&d| d != d1 && pts[d] > 1 && free(d))
                    .flat_map(|d| [2usize, 4, 8].into_iter().map(move |t| (d, t)))
                    .filter(|&(d, t)| pts[d].is_multiple_of(t) && t1 * t <= 16)
                    .min_by(|a, b| cost(Some(*a)).total_cmp(&cost(Some(*b))))
                    .filter(|&c| cost(Some(c)) < 0.75 * base);
            } else if super::tune::knob("TANG_REG2").as_deref() != Some("0") {
                // a 4-bit GEMV whose register dim the weight ignores (the
                // rows of a short window, `[k, rows]`): tile the weight
                // rows too, so each lane's activation loads serve several
                // rows and its packed words several tokens. Without it a
                // lane loads k·32 f32 activations per 16 bytes of weight
                // and a window is bound by those loads, not by DRAM. Up
                // to `TANG_WIN_TILE` (32) points a lane
                let cap: usize = super::tune::knob_parse("TANG_WIN_TILE").unwrap_or(32);
                let w4: Vec<&Leaf> = leaves
                    .iter()
                    .filter(|l| l.pack.is_some_and(Packing::low_bit))
                    .collect();
                if t1 > 1 && w4.iter().all(|l| l.addr.coefs[d1] == 0) {
                    reg2 = (0..pk)
                        .filter(|&d| {
                            d != d1
                                && pts[d] > 1
                                && free(d)
                                && w4.iter().all(|l| l.addr.coefs[d] != 0)
                        })
                        .flat_map(|d| [8usize, 4, 2].into_iter().map(move |t| (d, t)))
                        .find(|&(d, t)| pts[d].is_multiple_of(t) && t1 * t <= cap);
                    // then the token and row tiles together, by the bytes
                    // all groups load per reduce element (L2 traffic):
                    // each group loads every leaf once per register point
                    // it depends on; a 4-bit row re-read by k/t1 groups
                    // vs k f32 activations re-read by rows/t2 groups.
                    // 8×4 at k=8 is 8.5 B per weight element, 2×16 is 4
                    // (H200: window time tracks it). TANG_WIN_JOINT=0 off
                    // (only past 4 tokens: on the H200 2×16 took window-8
                    // from 17.9 to 13.4 ms but window-4 from 6.9 to 9.0)
                    if let Some((d2, _)) = reg2.filter(|_| pts[d1] > 4) {
                        if std::env::var("TANG_WIN_JOINT").as_deref() != Ok("0") {
                            let bytes = |l: &Leaf| match l.pack {
                                Some(Packing::U4) => 0.5,
                                Some(Packing::U3) => 0.375,
                                Some(Packing::Bf16 | Packing::F16) => 2.0,
                                None => 4.0,
                            };
                            let traffic = |a: usize, b: usize| -> f64 {
                                let groups = (pts[d1] / a * (pts[d2] / b)) as f64;
                                groups
                                    * leaves
                                        .iter()
                                        // (per-group scales, constant along
                                        // the innermost reduce dim, amortize)
                                        .filter(|l| l.addr.coefs[rank - 1] != 0)
                                        .map(|l| {
                                            let mut n = bytes(l);
                                            if l.addr.coefs[d1] != 0 {
                                                n *= a as f64;
                                            }
                                            if l.addr.coefs[d2] != 0 {
                                                n *= b as f64;
                                            }
                                            n
                                        })
                                        .sum::<f64>()
                            };
                            let best = [8usize, 4, 2]
                                .into_iter()
                                .filter(|&a| pts[d1].is_multiple_of(a))
                                .flat_map(|a| {
                                    [32usize, 16, 8, 4, 2].into_iter().map(move |b| (a, b))
                                })
                                .filter(|&(a, b)| pts[d2].is_multiple_of(b) && a * b <= cap)
                                .min_by(|x, y| traffic(x.0, x.1).total_cmp(&traffic(y.0, y.1)));
                            if let Some((a, b)) = best {
                                reg = Some((d1, a));
                                reg2 = Some((d2, b));
                            }
                        }
                    }
                }
            }
        }
        let (t1, t2) = (reg.map_or(1, |r| r.1), reg2.map_or(1, |r| r.1));
        let rt = t1 * t2;
        let on_reg = |cd: &Cond| {
            reg.is_some_and(|(d, _)| cd.coefs[d] != 0.0)
                || reg2.is_some_and(|(d, _)| cd.coefs[d] != 0.0)
        };
        // a runtime scalar of a bound is loaded per point group: it may not
        // vary along the register dim
        let rt_on_reg = |cd: &Cond| {
            [reg, reg2]
                .iter()
                .flatten()
                .any(|&(d, _)| cd.rt.iter().any(|&(_, i)| leaves[i].addr.coefs[d] != 0))
        };
        if point_conds
            .iter()
            .chain(&red_cond)
            .chain(&red_lo)
            .any(|cd| on_reg(cd) || rt_on_reg(cd))
        {
            return None;
        }

        let warps = if self.opts.warps == 0 {
            4
        } else {
            self.opts.warps
        };
        let n_points: usize = pts.iter().product();
        let groups = n_points / rt;
        // few groups: a block of warps splits each one's reduction
        // (TANG_KS_GROUPS: the threshold, default 512)
        let ks_below: usize = std::env::var("TANG_KS_GROUPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let chunks_inner = ri / vec;
        let n_chunks: usize = red[..red.len() - 1].iter().product::<usize>() * chunks_inner;
        // (only a reduction long enough to give each warp 32 chunks)
        let ks = if groups < ks_below && n_chunks >= 256 {
            8.max(warps)
        } else {
            1
        };
        let threads = if ks > 1 { 32 * ks } else { 32 * warps };
        // lanes per group: a short reduction (a split's combine over 4
        // partials, attention's 128-wide q·k) shares a warp among
        // 32 / lpg groups, each lane over ~4 chunks (`TANG_SUBWARP=0`: a
        // warp per group)
        // (a 4-bit weight's 32-wide chunks keep a warp per group: its
        // lanes stream the weight rows)
        let lpg = if ks == 1
            && !leaves.iter().any(|l| l.pack.is_some_and(Packing::low_bit))
            && super::tune::knob("TANG_SUBWARP").as_deref() != Some("0")
        {
            n_chunks.div_ceil(4).next_power_of_two().clamp(1, 32)
        } else {
            32
        };
        let gpw = 32 / lpg;
        let gpb = threads / 32 / ks * gpw;
        let blocks = if !point_conds.is_empty() && ks == 1 {
            // a grid-stride loop over the live groups
            groups.div_ceil(gpb).min(2048)
        } else {
            groups.div_ceil(gpb)
        };

        let w_out = epi.map_or(w, |e| e.width());
        // the fused split: one warp per group (no block split), the part
        // axis (dim 0) not tiled, no live-group bound
        if let Some(f) = fix {
            if ks != 1
                || lpg != 32
                || w != 1
                || epi.is_some()
                || !point_conds.is_empty()
                || pts[0] != f.parts
                || reg.is_some_and(|(d, _)| d == 0)
                || reg2.is_some_and(|(d, _)| d == 0)
            {
                return None;
            }
        }
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: warp reduction (decode)"
        )
        .unwrap();
        writeln!(
            s,
            "// {groups} groups x {rt} points, {} reduce in chunks of {vec}, {ks} warps per group{}",
            n_chunks * vec,
            if lpg < 32 {
                format!(", {lpg} lanes per group")
            } else {
                String::new()
            }
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {blocks}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        s.push_str(PRELUDE);
        if fix.is_some() {
            writeln!(s, "__device__ float tang_part[{n_points}];").unwrap();
            writeln!(s, "__device__ unsigned tang_ctr[{}];", groups / pts[0]).unwrap();
        }
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(
            s,
            "    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;"
        )
        .unwrap();
        // a point condition along one point dim bounds the live groups:
        // that dim is the slowest in the group order, and a grid-stride
        // loop stops at the bound
        let dyn_dim = point_cond.as_ref().and_then(|cd| {
            // (a dim of extent 1 is always coordinate 0: it bounds nothing)
            let dims: Vec<usize> = (0..pk)
                .filter(|&d| cd.coefs[d] != 0.0 && pts[d] > 1)
                .collect();
            let constant_rt = cd
                .rt
                .iter()
                .all(|&(_, i)| leaves[i].addr.coefs.iter().all(|&c| c == 0));
            match dims[..] {
                [d] if cd.coefs[d] < 0.0 && ks == 1 && point_conds.len() == 1 && constant_rt => {
                    Some(d)
                }
                _ => None,
            }
        });
        if ks == 1 {
            let limit = match dyn_dim {
                Some(d) => {
                    let cd = point_cond.as_ref().unwrap();
                    let base = self.cond_base(cd, &leaves, &[], &[]);
                    writeln!(
                        s,
                        "    const int glimit = min({}, max(0, (int)ceilf(({base}) / {}))) * {};",
                        pts[d],
                        flit(-cd.coefs[d]),
                        groups / pts[d]
                    )
                    .unwrap();
                    "glimit".to_string()
                }
                None => format!("{groups}"),
            };
            writeln!(s, "    const int ks = 0;").unwrap();
            writeln!(
                s,
                "    for (int grp0 = blockIdx.x * {gpb} + warp * {gpw}; grp0 < {limit}; grp0 += gridDim.x * {gpb}) {{"
            )
            .unwrap();
            // a sub-group past the limit still takes part in the shuffles
            writeln!(
                s,
                "    const int sl = lane & {}, gv = grp0 + lane / {lpg};",
                lpg - 1
            )
            .unwrap();
            writeln!(s, "    const bool valid = gv < {limit};").unwrap();
            writeln!(s, "    const int grp = valid ? gv : grp0;").unwrap();
        } else {
            writeln!(s, "    const int grp = blockIdx.x;").unwrap();
            writeln!(s, "    const int ks = warp, sl = lane;").unwrap();
            writeln!(s, "    const bool valid = true;").unwrap();
        }
        // group -> point coords (the register dim in steps of rt)
        let mut rem = "grp".to_string();
        let order: Vec<usize> = (0..pk)
            .rev()
            .filter(|&d| Some(d) != dyn_dim)
            .chain(dyn_dim)
            .collect();
        let tile = |d: usize| {
            if reg.is_some_and(|(r, _)| r == d) {
                t1
            } else if reg2.is_some_and(|(r, _)| r == d) {
                t2
            } else {
                1
            }
        };
        for d in order {
            let n = pts[d] / tile(d);
            let mul = if tile(d) > 1 {
                format!(" * {}", tile(d))
            } else {
                String::new()
            };
            writeln!(s, "    const int p{d} = (({rem}) % {n}){mul};").unwrap();
            rem = format!("({rem}) / {n}");
        }
        // register point r = (r1, r2) = (r / t2, r % t2)
        let d1 = reg.map(|r| r.0);
        let d2 = reg2.map(|r| r.0);
        let mut coords: Vec<String> = (0..pk)
            .map(|d| {
                if Some(d) == d1 {
                    if t2 > 1 {
                        format!("(p{d} + r / {t2})")
                    } else {
                        format!("(p{d} + r)")
                    }
                } else if Some(d) == d2 {
                    format!("(p{d} + r % {t2})")
                } else {
                    format!("p{d}")
                }
            })
            .collect();
        let mut outer_coords = Vec::new();
        for d in 0..red.len() - 1 {
            outer_coords.push(format!("q{}", pk + d));
        }
        coords.extend(outer_coords.iter().cloned());
        coords.push("(kb + e)".into());

        // accumulators
        writeln!(s, "    float acc[{rt}][{w}];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < {rt}; r++) {{").unwrap();
        for (slot, &id) in monoid.identity.iter().enumerate() {
            writeln!(s, "        acc[r][{slot}] = {};", flit(id)).unwrap();
        }
        writeln!(s, "    }}").unwrap();
        let pt_dims: Vec<usize> = (0..pk).collect();
        let mut lo_chunk = "0".to_string();
        let mut hi_chunk = format!("{n_chunks}");
        if !point_conds.is_empty() {
            let all: Vec<String> = point_conds
                .iter()
                .map(|cd| {
                    format!(
                        "({}) > 0.0f",
                        self.cond_base(cd, &leaves, &pt_dims, &coords)
                    )
                })
                .collect();
            writeln!(s, "    const bool live = valid && {};", all.join(" && ")).unwrap();
        } else {
            writeln!(s, "    const bool live = valid;").unwrap();
        }
        if let Some(cd) = &red_cond {
            // konst + rt + points + a·x > 0 on the reduce coord x
            let a = cd.coefs[kin];
            let base = self.cond_base(cd, &leaves, &pt_dims, &coords);
            if a < 0.0 {
                writeln!(
                    s,
                    "    const int hi = min({ri}, max(0, (int)ceilf(({base}) / {})));",
                    flit(-a)
                )
                .unwrap();
                hi_chunk = format!("((hi + {}) / {vec})", vec - 1);
            } else {
                writeln!(
                    s,
                    "    const int lo = min({ri}, max(0, (int)floorf(-({base}) / {}) + 1));",
                    flit(a)
                )
                .unwrap();
                lo_chunk = format!("(lo / {vec})");
            }
        }
        if let Some(cd) = &red_lo {
            // the second bound, from below (the first was from above)
            let a = cd.coefs[kin];
            let base = self.cond_base(cd, &leaves, &pt_dims, &coords);
            writeln!(
                s,
                "    const int lo = min({ri}, max(0, (int)floorf(-({base}) / {}) + 1));",
                flit(a)
            )
            .unwrap();
            lo_chunk = format!("(lo / {vec})");
        }
        writeln!(s, "    if (live) {{").unwrap();
        if self.opts.unroll > 1 {
            writeln!(s, "    #pragma unroll {}", self.opts.unroll).unwrap();
        }
        writeln!(
            s,
            "    for (int c = {lo_chunk} + sl + {lpg} * ks; c < {hi_chunk}; c += {}) {{",
            lpg * ks
        )
        .unwrap();
        // chunk -> outer reduce coords and the inner base
        writeln!(s, "        const int kb = (c % {chunks_inner}) * {vec};").unwrap();
        let mut rem = format!("(c / {chunks_inner})");
        for d in (0..red.len() - 1).rev() {
            writeln!(s, "        const int q{} = ({rem}) % {};", pk + d, red[d]).unwrap();
            rem = format!("({rem}) / {}", red[d]);
        }
        // preloads
        let dep1 = |l: &Leaf| d1.is_some_and(|d| l.addr.coefs[d] != 0);
        let dep2 = |l: &Leaf| d2.is_some_and(|d| l.addr.coefs[d] != 0);
        // a leaf's copies: one per register point of the tiled dims it reads
        let nr_of = |l: &Leaf| match (dep1(l), dep2(l)) {
            (true, true) => rt,
            (true, false) => t1,
            (false, true) => t2,
            (false, false) => 1,
        };
        // its coords while loading copy r
        let leaf_at0 = |l: &Leaf| -> Vec<String> {
            let mut c: Vec<String> = coords.iter().map(|x| x.replace("(kb + e)", "kb")).collect();
            // (a leaf reading one tiled dim: copy r is that dim's r; the
            // other dim's coefficient is 0)
            if !(dep1(l) && dep2(l)) {
                for d in [d1, d2].into_iter().flatten() {
                    c[d] = format!("(p{d} + r)");
                }
            }
            c
        };
        // f32 vector leaves tiled along a register dim (a window's
        // activations, one copy per token) stream a float4 at a time inside
        // the chunk instead of all `vec` up front: k·vec registers spill
        let stream = vec % 4 == 0
            && super::tune::knob("TANG_WIN_STREAM").as_deref() != Some("0")
            && leaves.iter().any(|l| l.pack.is_some_and(Packing::low_bit))
            && leaves
                .iter()
                .enumerate()
                .any(|(i, l)| class[i] == 1 && l.pack.is_none() && nr_of(l) > 1);
        let streamed = |i: usize| stream && class[i] == 1 && leaves[i].pack.is_none();
        for (i, l) in leaves.iter().enumerate() {
            let nr = nr_of(l);
            let at0 = leaf_at0(l);
            if streamed(i) {
                continue;
            }
            match class[i] {
                0 => {
                    writeln!(s, "        float h{i}[{nr}];").unwrap();
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(s, "        for (int r = 0; r < {nr}; r++) h{i}[r] = {};", {
                        let a = addr_expr(&l.addr, &at0);
                        load(l.buf, l.pack, &a)
                    })
                    .unwrap();
                }
                1 => {
                    let a = addr_expr(&l.addr, &at0);
                    match l.pack {
                        Some(Packing::U4) => {
                            writeln!(s, "        uint4 u{i}[{nr}];").unwrap();
                            writeln!(s, "        #pragma unroll").unwrap();
                            writeln!(
                                s,
                                "        for (int r = 0; r < {nr}; r++) u{i}[r] = __ldg((const uint4*)b{} + (({a}) >> 5));",
                                l.buf
                            )
                            .unwrap();
                        }
                        Some(Packing::U3) => {
                            // 32 codes = 3 words, chunk-aligned (no straddle)
                            writeln!(s, "        uint3 u{i}[{nr}];").unwrap();
                            writeln!(s, "        #pragma unroll").unwrap();
                            writeln!(
                                s,
                                "        for (int r = 0; r < {nr}; r++) {{ const unsigned* p = (const unsigned*)b{} + (({a}) >> 5) * 3; u{i}[r] = make_uint3(__ldg(p), __ldg(p + 1), __ldg(p + 2)); }}",
                                l.buf
                            )
                            .unwrap();
                        }
                        Some(Packing::Bf16 | Packing::F16) => {
                            // four 16-bit values per uint2
                            writeln!(s, "        uint2 w{i}[{nr}][{}];", vec / 4).unwrap();
                            writeln!(s, "        #pragma unroll").unwrap();
                            writeln!(s, "        for (int r = 0; r < {nr}; r++)").unwrap();
                            writeln!(s, "        #pragma unroll").unwrap();
                            writeln!(
                                s,
                                "        for (int q = 0; q < {}; q++) w{i}[r][q] = __ldg((const uint2*)b{} + (({a}) >> 2) + q);",
                                vec / 4,
                                l.buf
                            )
                            .unwrap();
                        }
                        None => {
                            writeln!(s, "        float4 v{i}[{nr}][{}];", vec / 4).unwrap();
                            writeln!(s, "        #pragma unroll").unwrap();
                            writeln!(s, "        for (int r = 0; r < {nr}; r++)").unwrap();
                            writeln!(s, "        #pragma unroll").unwrap();
                            writeln!(
                                s,
                                "        for (int q = 0; q < {}; q++) v{i}[r][q] = __ldg((const float4*)(b{} + ({a})) + q);",
                                vec / 4,
                                l.buf
                            )
                            .unwrap();
                        }
                    }
                }
                _ => {}
            }
        }
        if stream {
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int q = 0; q < {}; q++) {{", vec / 4).unwrap();
            for (i, l) in leaves.iter().enumerate() {
                if !streamed(i) {
                    continue;
                }
                let nr = nr_of(l);
                let a = addr_expr(&l.addr, &leaf_at0(l));
                writeln!(s, "        float4 v{i}[{nr}];").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(
                    s,
                    "        for (int r = 0; r < {nr}; r++) v{i}[r] = __ldg((const float4*)(b{} + ({a})) + q);",
                    l.buf
                )
                .unwrap();
            }
        }
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int r = 0; r < {rt}; r++) {{").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        if stream {
            writeln!(s, "        for (int e4 = 0; e4 < 4; e4++) {{").unwrap();
            writeln!(s, "        const int e = q * 4 + e4;").unwrap();
        } else {
            writeln!(s, "        for (int e = 0; e < {vec}; e++) {{").unwrap();
        }
        let ri_ = |l: &Leaf| -> String {
            match (dep1(l), dep2(l)) {
                (true, true) => "r".into(),
                (true, false) if t2 > 1 => format!("(r / {t2})"),
                (true, false) => "r".into(),
                (false, true) => format!("(r % {t2})"),
                (false, false) => "0".into(),
            }
        };
        let leaf_expr = |i: usize| -> String {
            let l = &leaves[i];
            match class[i] {
                0 => format!("h{i}[{}]", ri_(l)),
                1 => match l.pack {
                    // 2^23 + q as a float, minus 2^23: no int-to-float
                    // conversion (quarter rate) per weight
                    Some(Packing::U4) => format!(
                        "(__uint_as_float(0x4b000000u | (((&u{i}[{}].x)[e >> 3] >> ((e & 7) * 4)) & 15u)) - 8388608.0)",
                        ri_(l)
                    ),
                    // bit 3e of the chunk's 3 words; `e` is unrolled, so
                    // the word, shift and straddle fold to constants
                    Some(Packing::U3) => format!(
                        "(__uint_as_float(0x4b000000u | ((((&u{i}[{r}].x)[(3 * e) >> 5] >> ((3 * e) & 31)) | (((3 * e) & 31) > 29 ? (&u{i}[{r}].x)[((3 * e) >> 5) + 1] << (32 - ((3 * e) & 31)) : 0u)) & 7u)) - 8388608.0)",
                        r = ri_(l)
                    ),
                    Some(p @ (Packing::Bf16 | Packing::F16)) => p.half(&format!(
                        "(((&w{i}[{}][e >> 2].x)[(e >> 1) & 1] >> (16 * (e & 1))) & 0xffffu)",
                        ri_(l)
                    )),
                    None if streamed(i) => format!("(&v{i}[{}].x)[e & 3]", ri_(l)),
                    None => format!("(&v{i}[{}][e >> 2].x)[e & 3]", ri_(l)),
                },
                _ => load(l.buf, l.pack, &addr_expr(&l.addr, &coords)),
            }
        };
        let ind = "            ";
        let mut ctr = 0usize;
        let vals: Vec<String> = args
            .iter()
            .map(|a| self.emit_val(&mut s, a, &coords, ind, &mut ctr, &leaf_expr))
            .collect();
        let el = emit(&mut s, &rc.body, &rc.body_outs, "x_", ind, &|v| {
            vals[v as usize].clone()
        });
        self.combine(&mut s, monoid, w, ind, |k| format!("acc[r][{k}]"), &el);
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        }}").unwrap();
        if stream {
            writeln!(s, "        }}").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    }}").unwrap();

        // shuffle combine
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int off = {}; off > 0; off >>= 1) {{", lpg / 2).unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int r = 0; r < {rt}; r++) {{").unwrap();
        let others: Vec<String> = (0..w).map(|k| format!("o{k}")).collect();
        for (k, o) in others.iter().enumerate() {
            writeln!(
                s,
                "            float {o} = __shfl_xor_sync(0xffffffffu, acc[r][{k}], off);"
            )
            .unwrap();
        }
        self.combine(
            &mut s,
            monoid,
            w,
            "            ",
            |k| format!("acc[r][{k}]"),
            &others,
        );
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        if ks > 1 {
            writeln!(s, "    __shared__ float part[{ks}][{rt}][{w}];").unwrap();
            writeln!(s, "    if (lane == 0) {{").unwrap();
            writeln!(s, "        for (int r = 0; r < {rt}; r++)").unwrap();
            writeln!(
                s,
                "        for (int k = 0; k < {w}; k++) part[ks][r][k] = acc[r][k];"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (ks != 0 || grp >= {groups}) return;").unwrap();
            writeln!(s, "    for (int j = 1; j < {ks}; j++) {{").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int r = 0; r < {rt}; r++) {{").unwrap();
            let pj: Vec<String> = (0..w).map(|k| format!("part[j][r][{k}]")).collect();
            self.combine(
                &mut s,
                monoid,
                w,
                "            ",
                |k| format!("acc[r][{k}]"),
                &pj,
            );
            writeln!(s, "        }}").unwrap();
            writeln!(s, "    }}").unwrap();
        }

        // epilogue and write: lane r writes register point r
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < {rt}; r++) {{").unwrap();
        writeln!(s, "        if (!valid || sl != r % {lpg}) continue;").unwrap();
        let pcoords: Vec<String> = coords[..pk].to_vec();
        let vals: Vec<String> = match epi {
            Some(ec) => {
                let mut eleaves = Vec::new();
                let mut evals = Vec::new();
                for a in &ec.args {
                    if a.src.n() == red_node {
                        evals.push(format!("acc[r][{}]", g.epilogue_slot(a)));
                        continue;
                    }
                    let t = self.tree(a, pk, &mut eleaves)?;
                    let mut ctr = 100usize;
                    let el = |i: usize| {
                        let l: &Leaf = &eleaves[i];
                        load(l.buf, l.pack, &addr_expr(&l.addr, &pcoords))
                    };
                    let v = self.emit_val(&mut s, &t, &pcoords, "        ", &mut ctr, &el);
                    evals.push(v);
                }
                emit(&mut s, &ec.body, &ec.body_outs, "y_", "        ", &|v| {
                    evals[v as usize].clone()
                })
            }
            None => (0..w).map(|k| format!("acc[r][{k}]")).collect(),
        };
        let mut flat = "0".to_string();
        for (d, &np) in pts.iter().enumerate() {
            flat = format!("({flat}) * {np} + {}", pcoords[d]);
        }
        writeln!(s, "        const int o = {flat};").unwrap();
        let dst = if fix.is_some() { "tang_part" } else { "out" };
        for (slot, v) in vals.iter().enumerate() {
            writeln!(s, "        {dst}[o * {w_out} + {slot}] = {v};").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        if let Some(f) = fix {
            // the last part of this group to arrive sums them all
            let gr = groups / pts[0];
            let rest: usize = pts[1..].iter().product();
            writeln!(s, "    __threadfence();").unwrap();
            writeln!(s, "    unsigned lastw = 0u;").unwrap();
            writeln!(
                s,
                "    if (lane == 0 && valid) lastw = atomicAdd(&tang_ctr[grp % {gr}], 1u) == {}u;",
                f.parts - 1
            )
            .unwrap();
            writeln!(s, "    lastw = __shfl_sync(0xffffffffu, lastw, 0);").unwrap();
            writeln!(s, "    if (lastw) {{").unwrap();
            writeln!(s, "    __threadfence();").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int r = 0; r < {rt}; r++) {{").unwrap();
            writeln!(s, "        if (sl != r % {lpg}) continue;").unwrap();
            let ccoords: Vec<String> = pcoords[1..].to_vec();
            let mut cflat = "0".to_string();
            for (d, &np) in pts[1..].iter().enumerate() {
                cflat = format!("({cflat}) * {np} + {}", ccoords[d]);
            }
            writeln!(s, "        const int oc = {cflat};").unwrap();
            writeln!(s, "        float v = 0.0f;").unwrap();
            writeln!(
                s,
                "        for (int q = 0; q < {}; q++) v += __ldcg(&tang_part[q * {rest} + oc]);",
                f.parts
            )
            .unwrap();
            let cvals: Vec<String> = match f.e {
                Some(e) => {
                    let ec = g.comp_at(e)?;
                    let mut eleaves = Vec::new();
                    let mut evals = Vec::new();
                    for a in &ec.args {
                        if a.src.n() == f.c {
                            evals.push("v".to_string());
                            continue;
                        }
                        let tv = self.tree(a, pk - 1, &mut eleaves)?;
                        let mut ctr = 200usize;
                        let el = |i: usize| {
                            let l: &Leaf = &eleaves[i];
                            load(l.buf, l.pack, &addr_expr(&l.addr, &ccoords))
                        };
                        evals.push(self.emit_val(&mut s, &tv, &ccoords, "        ", &mut ctr, &el));
                    }
                    emit(&mut s, &ec.body, &ec.body_outs, "z_", "        ", &|v| {
                        evals[v as usize].clone()
                    })
                }
                None => vec!["v".to_string()],
            };
            let wc = cvals.len();
            for (slot, v) in cvals.iter().enumerate() {
                writeln!(s, "        out[oc * {wc} + {slot}] = {v};").unwrap();
            }
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    if (lane == 0) tang_ctr[grp % {gr}] = 0u;").unwrap();
            writeln!(s, "    }}").unwrap();
        }
        if ks == 1 {
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads,
            blocks,
        })
    }

    /// `acc_k ← combine(acc, el)` for every slot.
    fn combine(
        &self,
        s: &mut String,
        m: &Monoid,
        w: usize,
        indent: &str,
        acc: impl Fn(usize) -> String,
        el: &[String],
    ) {
        let accs: Vec<String> = (0..w).map(&acc).collect();
        if w > 2 * TOPK_UNROLL_MAX && super::step::gather::is_topk(m) {
            return topk_merge(s, w / 2, indent, &accs, el);
        }
        let out = emit(s, &m.combine, &m.outs, "m_", indent, &|v| {
            let v = v as usize;
            if v < w {
                accs[v].clone()
            } else {
                el[v - w].clone()
            }
        });
        // all new values first, then the assignment (slots read each other)
        let tmp: Vec<String> = (0..w).map(|k| format!("n{k}")).collect();
        let mut line = String::new();
        for (k, o) in out.iter().enumerate() {
            write!(line, "float {} = {o}; ", tmp[k]).unwrap();
        }
        writeln!(s, "{indent}{{ {line}").unwrap();
        for (k, a) in accs.iter().enumerate() {
            writeln!(s, "{indent}  {a} = {};", tmp[k]).unwrap();
        }
        writeln!(s, "{indent}}}").unwrap();
    }

    /// A [`Monoid::topk`]`(k)` reduction (argmax is `k = 1`) as a block
    /// select instead of the generic combine's k²/2 compare-selects a
    /// step: one block per point keeps its best `k` `(index, value)` pairs
    /// sorted in shared memory; each chunk of `P − k` reduce elements
    /// (body evaluated per element) lands behind them and a bitonic sort
    /// of all `P` (a power of two ≥ 2k) restores the order — value
    /// descending, the smaller index first on a tie, the monoid's order.
    /// Any row length; `P · 8` bytes of shared memory. The body's element
    /// must be one pair followed by identity pairs (as `comp_tuple` bodies
    /// for top-k write it).
    fn topk_select(&self, root: usize) -> Option<Kernel> {
        let c = self.g.comp_at(root)?;
        let m = c.monoid.as_ref()?;
        let k = m.topk_k()?;
        // `TANG_TOPK_SELECT=0`: the generic combine (a baseline to time)
        if std::env::var("TANG_TOPK_SELECT").is_ok_and(|v| v == "0") {
            return None;
        }
        // small k (the argmax) is cheaper as the generic warp combine
        if k < 32 || c.tuple != 1 || c.width() != 2 * k || c.red.is_empty() {
            return None;
        }
        // the element is (j, w) then identity pairs
        let nv = c.args.len();
        for probe in 0..3 {
            let v: Vec<f64> = (0..nv)
                .map(|i| ((i * 5 + probe * 11) % 13) as f64 * 0.7 - 2.1)
                .collect();
            let o = c.body.eval_many::<f64>(&c.body_outs[2..], &v);
            if o.chunks(2)
                .any(|p| p[0] != 0.0 || p[1] != -(f32::MAX as f64))
            {
                return None;
            }
        }
        let pts = fixed(c.points());
        let red = fixed(&c.red);
        let (pk, rk) = (pts.len(), red.len());
        let mut leaves = Vec::new();
        let args = c
            .args
            .iter()
            .map(|a| self.tree(a, pk + rk, &mut leaves))
            .collect::<Option<Vec<_>>>()?;
        let n_red: usize = red.iter().product();
        let n_pts: usize = pts.iter().product();
        let p = (2 * k).next_power_of_two().max(1024);
        let chunk = p - k;
        let threads = 256;
        let mut s = String::new();
        writeln!(s, "// Auto-generated by tang-compile: top-{k} block select").unwrap();
        writeln!(s, "#define TANG_TILES {n_pts}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        s.push_str(PRELUDE);
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    __shared__ float si[{p}];").unwrap();
        writeln!(s, "    __shared__ float sv[{p}];").unwrap();
        writeln!(s, "    const int tid = threadIdx.x;").unwrap();
        writeln!(s, "    const int pt = blockIdx.x;").unwrap();
        let mut rem = "pt".to_string();
        for d in (0..pk).rev() {
            writeln!(s, "    const int p{d} = ({rem}) % {};", pts[d]).unwrap();
            rem = format!("({rem}) / {}", pts[d]);
        }
        writeln!(s, "    __shared__ int cnt;").unwrap();
        writeln!(s, "    for (int i = tid; i < {p}; i += {threads}) {{ si[i] = 0.0f; sv[i] = -3.40282347e38f; }}").unwrap();
        writeln!(s, "    if (tid == 0) cnt = 0;").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        writeln!(
            s,
            "    for (int base = 0; base < {n_red}; base += {threads}) {{"
        )
        .unwrap();
        writeln!(s, "      {{").unwrap();
        writeln!(s, "        const int r = base + tid;").unwrap();
        writeln!(s, "        float ej = 0.0f, ew = -3.40282347e38f;").unwrap();
        writeln!(s, "        if (r < {n_red}) {{").unwrap();
        let mut rr = "r".to_string();
        for d in (0..rk).rev() {
            writeln!(s, "          const int p{} = ({rr}) % {};", pk + d, red[d]).unwrap();
            rr = format!("({rr}) / {}", red[d]);
        }
        let coords: Vec<String> = (0..pk + rk).map(|d| format!("p{d}")).collect();
        let mut ctr = 0usize;
        let el = |i: usize| {
            let l: &Leaf = &leaves[i];
            load(l.buf, l.pack, &addr_expr(&l.addr, &coords))
        };
        let vals: Vec<String> = args
            .iter()
            .map(|a| self.emit_val(&mut s, a, &coords, "          ", &mut ctr, &el))
            .collect();
        let y = emit(
            &mut s,
            &c.body,
            &c.body_outs[..2],
            "x_",
            "          ",
            &|v| vals[v as usize].clone(),
        );
        writeln!(s, "          ej = {}; ew = {};", y[0], y[1]).unwrap();
        writeln!(s, "        }}").unwrap();
        // only what beats the current k-th survives to the next sort
        writeln!(
            s,
            "        const float tv = sv[{}], tj = si[{}];",
            k - 1,
            k - 1
        )
        .unwrap();
        writeln!(
            s,
            "        if (r < {n_red} && (ew > tv || (ew == tv && ej < tj))) {{"
        )
        .unwrap();
        writeln!(s, "          const int q = atomicAdd(&cnt, 1);").unwrap();
        writeln!(s, "          si[{k} + q] = ej; sv[{k} + q] = ew;").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "      }}").unwrap();
        writeln!(s, "      __syncthreads();").unwrap();
        writeln!(s, "      const int c = cnt;").unwrap();
        writeln!(s, "      __syncthreads();").unwrap();
        writeln!(
            s,
            "      if (c > {} || base + {threads} >= {n_red}) {{",
            chunk - threads
        )
        .unwrap();
        writeln!(s, "      for (int i = {k} + c + tid; i < {p}; i += {threads}) {{ si[i] = 0.0f; sv[i] = -3.40282347e38f; }}").unwrap();
        writeln!(s, "      __syncthreads();").unwrap();
        // bitonic sort of all P, best first
        writeln!(s, "      for (int size = 2; size <= {p}; size <<= 1) {{").unwrap();
        writeln!(
            s,
            "        for (int stride = size >> 1; stride > 0; stride >>= 1) {{"
        )
        .unwrap();
        writeln!(s, "          for (int i = tid; i < {p}; i += {threads}) {{").unwrap();
        writeln!(s, "            const int j = i ^ stride;").unwrap();
        writeln!(s, "            if (j > i) {{").unwrap();
        writeln!(
            s,
            "              const bool jb = sv[j] > sv[i] || (sv[j] == sv[i] && si[j] < si[i]);"
        )
        .unwrap();
        writeln!(
            s,
            "              const bool ib = sv[i] > sv[j] || (sv[i] == sv[j] && si[i] < si[j]);"
        )
        .unwrap();
        writeln!(s, "              if ((i & size) == 0 ? jb : ib) {{").unwrap();
        writeln!(
            s,
            "                float a = si[i]; si[i] = si[j]; si[j] = a;"
        )
        .unwrap();
        writeln!(s, "                a = sv[i]; sv[i] = sv[j]; sv[j] = a;").unwrap();
        writeln!(s, "              }}").unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "          }}").unwrap();
        writeln!(s, "          __syncthreads();").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "      }}").unwrap();
        writeln!(s, "      if (tid == 0) cnt = 0;").unwrap();
        writeln!(s, "      __syncthreads();").unwrap();
        writeln!(s, "      }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    for (int i = tid; i < {k}; i += {threads}) {{").unwrap();
        writeln!(s, "      out[pt * {} + 2 * i] = si[i];", 2 * k).unwrap();
        writeln!(s, "      out[pt * {} + 2 * i + 1] = sv[i];", 2 * k).unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads,
            blocks: n_pts,
        })
    }

    /// A map reading packed inputs (the dequantization of a weight a
    /// tensor-core tile then reads): one thread per point.
    fn point_map(&self, root: usize) -> Option<Kernel> {
        self.point_map_with(root, true)
    }

    /// A plain map that hosts side jobs or inlined nodes (`TNode::host`):
    /// [`Self::point_map`] without the packed-input requirement.
    fn hosted_map(&self, root: usize) -> Option<Kernel> {
        if !self.g.hosts(root) {
            return None;
        }
        self.point_map_with(root, false)
    }

    fn point_map_with(&self, root: usize, need_packed: bool) -> Option<Kernel> {
        let c = self.g.comp_at(root)?;
        if c.monoid.is_some()
            || (need_packed && !self.reads.iter().any(|j| self.packed.contains_key(j)))
        {
            return None;
        }
        let pts = fixed(c.points());
        let pk = pts.len();
        let w = c.width();
        let mut leaves = Vec::new();
        self.guards.set(!need_packed);
        let args = c
            .args
            .iter()
            .map(|a| self.tree(a, pk, &mut leaves))
            .collect::<Option<Vec<_>>>();
        self.guards.set(false);
        let args = args?;
        let n: usize = pts.iter().product();
        // stored as f16 (a dequantized weight a half tile reads): a thread
        // writes a word, two neighbouring points
        let half = self.g.nodes[root].precision.storage == super::Format::F16;
        if half && (w != 1 || !n.is_multiple_of(2)) {
            return None;
        }
        let per = if half { 2 } else { 1 };
        let threads = 256;
        let blocks = (n / per).div_ceil(threads);
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: map over packed inputs"
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {blocks}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        s.push_str(PRELUDE);
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(
            s,
            "    const int tt = blockIdx.x * {threads} + threadIdx.x;"
        )
        .unwrap();
        writeln!(s, "    if (tt >= {}) return;", n / per).unwrap();
        writeln!(s, "    unsigned hw[2] = {{0u, 0u}};").unwrap();
        writeln!(s, "    for (int u = 0; u < {per}; u++) {{").unwrap();
        writeln!(s, "    const int t = tt * {per} + u;").unwrap();
        let mut rem = "t".to_string();
        for d in (0..pk).rev() {
            writeln!(s, "    const int p{d} = ({rem}) % {};", pts[d]).unwrap();
            rem = format!("({rem}) / {}", pts[d]);
        }
        let coords: Vec<String> = (0..pk).map(|d| format!("p{d}")).collect();
        let mut ctr = 0usize;
        let el = |i: usize| {
            let l: &Leaf = &leaves[i];
            load(l.buf, l.pack, &addr_expr(&l.addr, &coords))
        };
        let vals: Vec<String> = args
            .iter()
            .map(|a| self.emit_val(&mut s, a, &coords, "    ", &mut ctr, &el))
            .collect();
        let y = emit(&mut s, &c.body, &c.body_outs, "x_", "    ", &|v| {
            vals[v as usize].clone()
        });
        if half {
            writeln!(s, "    hw[u] = tang_f2h({});", y[0]).unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    ((unsigned*)out)[tt] = hw[0] | (hw[1] << 16);").unwrap();
        } else {
            for (slot, v) in y.iter().enumerate() {
                writeln!(s, "    out[t * {w} + {slot}] = {v};").unwrap();
            }
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads,
            blocks,
        })
    }

    /// Whether the staged producers fit: Sum or Max of width 1 with a
    /// reduce space, at most 4096 outputs in all.
    fn staged_ok(&self, staged: &[usize]) -> Option<()> {
        let mut total = 0usize;
        for &p in staged {
            let pc = self.g.comp_at(p)?;
            let kind = pc.monoid.as_ref()?.kind();
            if !matches!(kind, super::MonoidKind::Sum | super::MonoidKind::Max) || pc.red.is_empty()
            {
                return None;
            }
            total += numel(pc.points());
        }
        (total <= 4096).then_some(())
    }

    /// The staged producers into `__shared__ float sm{i}[…]` — one warp
    /// per output point, lanes over its reduce space — then a block
    /// barrier (emitted before any thread may return).
    fn emit_staged(
        &self,
        s: &mut String,
        staged: &[usize],
        threads: usize,
        ctr: &mut usize,
    ) -> Option<()> {
        let g = self.g;
        writeln!(
            s,
            "    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;"
        )
        .unwrap();
        for (i, &p) in staged.iter().enumerate() {
            let pc = g.comp_at(p)?;
            let m = pc.monoid.as_ref()?;
            let sum = m.kind() == super::MonoidKind::Sum;
            let ppts = fixed(pc.points());
            let pred = fixed(&pc.red);
            let (ppk, prank) = (ppts.len(), ppts.len() + pred.len());
            let pn: usize = ppts.iter().product();
            let rn: usize = pred.iter().product();
            let mut pleaves = Vec::new();
            let pargs = pc
                .args
                .iter()
                .map(|a| self.tree(a, prank, &mut pleaves))
                .collect::<Option<Vec<_>>>()?;
            // warps per point: all of them share a point when there are
            // few (an RMSNorm's one), partials combined through smem
            let nw = threads / 32;
            let mut wpp = 1;
            while wpp * 2 * pn <= nw {
                wpp *= 2;
            }
            let groups = nw / wpp;
            writeln!(s, "    __shared__ float sm{i}[{pn}];").unwrap();
            if wpp > 1 {
                writeln!(s, "    __shared__ float smp{i}[{nw}];").unwrap();
            }
            writeln!(
                s,
                "    for (int q = warp / {wpp}; q < {pn}; q += {groups}) {{"
            )
            .unwrap();
            let id = flit(m.identity[0]);
            writeln!(s, "        float acc = {id};").unwrap();
            writeln!(
                s,
                "        for (int r = lane + 32 * (warp % {wpp}); r < {rn}; r += {}) {{",
                32 * wpp
            )
            .unwrap();
            let mut rem = "q".to_string();
            for d in (0..ppk).rev() {
                writeln!(s, "        const int c{d} = ({rem}) % {};", ppts[d]).unwrap();
                rem = format!("({rem}) / {}", ppts[d]);
            }
            let mut rem = "r".to_string();
            for d in (0..pred.len()).rev() {
                writeln!(s, "        const int c{} = ({rem}) % {};", ppk + d, pred[d]).unwrap();
                rem = format!("({rem}) / {}", pred[d]);
            }
            let pcoords: Vec<String> = (0..prank).map(|d| format!("c{d}")).collect();
            let el = |k: usize| {
                let l: &Leaf = &pleaves[k];
                load(l.buf, l.pack, &addr_expr(&l.addr, &pcoords))
            };
            let vals: Vec<String> = pargs
                .iter()
                .map(|a| self.emit_val(s, a, &pcoords, "        ", ctr, &el))
                .collect();
            *ctr += 1;
            let y = emit(
                s,
                &pc.body,
                &pc.body_outs,
                &format!("r{}_", *ctr),
                "        ",
                &|v| vals[v as usize].clone(),
            );
            if sum {
                writeln!(s, "        acc += {};", y[0]).unwrap();
            } else {
                writeln!(s, "        acc = fmaxf(acc, {});", y[0]).unwrap();
            }
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        for (int o = 16; o > 0; o >>= 1) {{").unwrap();
            writeln!(
                s,
                "            float v = __shfl_xor_sync(0xffffffffu, acc, o);"
            )
            .unwrap();
            if sum {
                writeln!(s, "            acc += v;").unwrap();
            } else {
                writeln!(s, "            acc = fmaxf(acc, v);").unwrap();
            }
            writeln!(s, "        }}").unwrap();
            if wpp > 1 {
                writeln!(s, "        if (lane == 0) smp{i}[warp] = acc;").unwrap();
                writeln!(s, "    }}").unwrap();
                writeln!(s, "    __syncthreads();").unwrap();
                writeln!(s, "    if (threadIdx.x < {pn}) {{").unwrap();
                writeln!(s, "        float acc = smp{i}[threadIdx.x * {wpp}];").unwrap();
                writeln!(s, "        for (int k = 1; k < {wpp}; k++) {{").unwrap();
                let v = format!("smp{i}[threadIdx.x * {wpp} + k]");
                if sum {
                    writeln!(s, "            acc += {v};").unwrap();
                } else {
                    writeln!(s, "            acc = fmaxf(acc, {v});").unwrap();
                }
                writeln!(s, "        }}").unwrap();
                writeln!(s, "        sm{i}[threadIdx.x] = acc;").unwrap();
                writeln!(s, "    }}").unwrap();
            } else {
                writeln!(s, "        if (lane == 0) sm{i}[q] = acc;").unwrap();
                writeln!(s, "    }}").unwrap();
            }
        }
        writeln!(s, "    __syncthreads();").unwrap();
        Some(())
    }

    /// The value tree of `a` like [`Self::tree`], with an inlined
    /// reduction of width 1 (the *staged producer*) read as an element of
    /// a shared-memory copy of its whole output: `staged` collects the
    /// producers ([`Val::Staged`]).
    fn tree_staged(
        &self,
        a: &Arg,
        rank: usize,
        leaves: &mut Vec<Leaf>,
        staged: &mut Vec<usize>,
    ) -> Option<Val> {
        let j = a.src.n();
        if matches!(self.g.nodes[j].op, Op::Iota(_)) || self.is_buffer(j) {
            return self.tree(a, rank, leaves);
        }
        if a.view.guard {
            return None;
        }
        let mc = self.g.comp_at(j)?;
        if !self.g.inlined_in(j, self.root) || mc.width() != 1 || mc.tuple != 0 {
            return None;
        }
        if mc.monoid.is_some() {
            if a_guarded(a) {
                return None;
            }
            let i = staged.iter().position(|&p| p == j).unwrap_or_else(|| {
                staged.push(j);
                staged.len() - 1
            });
            return Some(Val::Staged(i, self.g.flat_addr(a, rank)));
        }
        let args = mc
            .args
            .iter()
            .map(|s| {
                let view = s.view.compose(&a.view, rank);
                self.tree_staged(&Arg { src: s.src, view }, rank, leaves, staged)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Val::Map {
            body: mc.body.clone(),
            out: mc.body_outs[0],
            args,
        })
    }

    /// A map that reads small inlined reductions (an RMSNorm's sum of
    /// squares, per-head q/k-norm statistics): every block first computes
    /// each such producer's whole output into shared memory — one warp
    /// per producer point, lanes over its reduce space, a shuffle tree —
    /// then evaluates its own points reading them from there. The
    /// producer is recomputed per block instead of launched as a kernel
    /// of its own (a decode step's statistics are a few thousand
    /// elements; the launch and the dependency cost more).
    /// Coder-next schedules them only with `TANG_STAGED_MAP=1`;
    /// `TANG_STAGED_MAP=0` disables it everywhere.
    fn staged_map(&self, root: usize) -> Option<Kernel> {
        if super::tune::knob("TANG_STAGED_MAP").as_deref() == Some("0") {
            return None;
        }
        let g = self.g;
        let c = g.comp_at(root)?;
        // stored as Q8x32: whole warps store blocks of 32 (`tang_q8x32_store`)
        let st = g.nodes[root].precision.storage;
        let q8 = matches!(st, super::Format::Q8x32 | super::Format::Q8x32F);
        if c.monoid.is_some()
            || self.packed.contains_key(&root)
            || !(st == super::Format::F32 || q8)
        {
            return None;
        }
        let pts = fixed(c.points());
        let pk = pts.len();
        let w = c.width();
        if q8 && (w != 1 || !numel(c.points()).is_multiple_of(32)) {
            return None;
        }
        let mut leaves = Vec::new();
        let mut staged = Vec::new();
        let args = c
            .args
            .iter()
            .map(|a| self.tree_staged(a, pk, &mut leaves, &mut staged))
            .collect::<Option<Vec<_>>>()?;
        if staged.is_empty() {
            return None;
        }
        self.staged_ok(&staged)?;
        let n: usize = pts.iter().product();
        let threads = 256usize;
        let blocks = n.div_ceil(threads);
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: map with staged reductions (decode)"
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {blocks}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        s.push_str(PRELUDE);
        if q8 {
            s.push_str(&super::cuda_int8::q8x32_store(st == super::Format::Q8x32F));
        }
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        let mut ctr = 0usize;
        self.emit_staged(&mut s, &staged, threads, &mut ctr)?;
        writeln!(s, "    const int t = blockIdx.x * {threads} + threadIdx.x;").unwrap();
        writeln!(s, "    if (t >= {n}) return;").unwrap();
        let mut rem = "t".to_string();
        for d in (0..pk).rev() {
            writeln!(s, "    const int p{d} = ({rem}) % {};", pts[d]).unwrap();
            rem = format!("({rem}) / {}", pts[d]);
        }
        let coords: Vec<String> = (0..pk).map(|d| format!("p{d}")).collect();
        let el = |i: usize| {
            let l: &Leaf = &leaves[i];
            load(l.buf, l.pack, &addr_expr(&l.addr, &coords))
        };
        let vals: Vec<String> = args
            .iter()
            .map(|a| self.emit_val(&mut s, a, &coords, "    ", &mut ctr, &el))
            .collect();
        let y = emit(&mut s, &c.body, &c.body_outs, "x_", "    ", &|v| {
            vals[v as usize].clone()
        });
        if q8 {
            // codes then the per-32 scales after the n code bytes
            writeln!(s, "    tang_q8x32_store(out, t, {}, {n}u);", y[0]).unwrap();
        } else {
            for (slot, v) in y.iter().enumerate() {
                writeln!(s, "    out[t * {w} + {slot}] = {v};").unwrap();
            }
        }
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads,
            blocks,
        })
    }

    /// `out = select(c₁, select(c₂, …y…, old), old)` written over `old`'s
    /// buffer (the donor `donor`): only the points where every `cᵢ` holds
    /// are visited, bounded along the one coordinate the conditions read.
    fn inplace_update(&self, root: usize, donor: usize) -> Option<Kernel> {
        let g = self.g;
        let c = g.comp_at(root)?;
        if c.monoid.is_some() || c.width() != 1 || c.tuple != 0 {
            return None;
        }
        let pts = fixed(c.points());
        let pk = pts.len();
        // an inlined reduction over one coord, read point for point (the
        // one-hot sum that picks a chunk's token for each cache row): a
        // loop at every point, with runtime bounds from its body's
        // `select(cond, x, identity)` chain; its coord follows the points
        let red_arg = c.args.iter().position(|a| {
            let p = a.src.n();
            g.nodes[p].inline && g.comp_at(p).is_some_and(|pc| pc.monoid.is_some())
        });
        let red = match red_arg {
            Some(ai) => {
                let p = c.args[ai].src.n();
                let rc = g.comp_at(p)?;
                let one = c.args.iter().filter(|a| a.src.n() == p).count() == 1;
                if !one
                    || rc.red.len() != 1
                    || rc.width() != 1
                    || rc.tuple != 0
                    || rc.points().len() != pk
                    || c.args[ai].view != super::View::identity(pk, pk)
                {
                    return None;
                }
                Some((ai, rc))
            }
            None => None,
        };
        let rank = pk + usize::from(red.is_some());
        // the in-place map's args over `rank` coords (none on the loop's)
        let widen = |a: &Arg| {
            let mut a = a.clone();
            for f in &mut a.view.forms {
                f.coefs.resize(rank, 0);
            }
            a
        };
        let mut leaves = Vec::new();
        // small reductions under the args (the roped key's norm
        // statistics) are staged in shared memory (`staged_map`)
        let mut staged = Vec::new();
        let args = c
            .args
            .iter()
            .enumerate()
            .map(|(ai, a)| {
                if red.is_some_and(|(r, _)| r == ai) {
                    Some(Val::Var(String::new()))
                } else {
                    self.tree_staged(&widen(a), rank, &mut leaves, &mut staged)
                }
            })
            .collect::<Option<Vec<_>>>()?;
        if !staged.is_empty() {
            self.staged_ok(&staged)?;
        }
        // the reduction's args are over its own space: the points, then its coord
        let red_args = match red {
            Some((_, rc)) => Some(
                rc.args
                    .iter()
                    .map(|a| self.tree(a, rank, &mut leaves))
                    .collect::<Option<Vec<_>>>()?,
            ),
            None => None,
        };
        let nodes = c.body.nodes_slice();
        let is_old = |e: ExprId| match nodes[e.index() as usize] {
            Node::Var(v) => {
                let a = &c.args[v as usize];
                self.storage(a.src.n()) == donor && a.view == super::View::identity(pk, pk)
            }
            _ => false,
        };
        let mut conds = Vec::new();
        let mut e = c.body_outs[0];
        while let Node::Select(cnd, t, f) = nodes[e.index() as usize] {
            if !is_old(f) {
                break;
            }
            conds.push(self.cond(&c.body, cnd, &args, &leaves, rank, pk)?);
            e = t;
        }
        if conds.is_empty() {
            return None;
        }
        // the one coordinate the conditions bound
        let dims: Vec<usize> = (0..pk)
            .filter(|&d| conds.iter().any(|cd| cd.coefs[d] != 0.0))
            .collect();
        let [dp] = dims[..] else { return None };
        // the bounds are computed before the loop over `dp`
        if conds
            .iter()
            .any(|cd| cd.rt.iter().any(|&(_, i)| leaves[i].addr.coefs[dp] != 0))
        {
            return None;
        }
        // the reduction's loop bounds: lo ≤ x < hi from its select chain
        let mut red_bounds: Vec<Cond> = Vec::new();
        if let (Some((_, rc)), Some(ra)) = (red, &red_args) {
            let monoid = rc.monoid.as_ref().unwrap();
            let rn = rc.body.nodes_slice();
            let mut e = rc.body_outs[0];
            while let Node::Select(cnd, t, f) = rn[e.index() as usize] {
                if !rn[f.index() as usize]
                    .as_f64()
                    .is_some_and(|v| is_identity_lit(monoid, 0, v))
                {
                    break;
                }
                if let Some(cd) = self.cond(&rc.body, cnd, ra, &leaves, rank, pk) {
                    if cd.coefs[pk] != 0.0 {
                        red_bounds.push(cd);
                    }
                }
                e = t;
            }
        }
        // a 16-bit output: a thread writes whole words — two neighbours
        // along the innermost dim, or (when that is the bounded dim, whose
        // points are all this thread's) one half by read-modify-write
        let pack = self.packed.get(&root).copied();
        if pack.is_some_and(Packing::low_bit) {
            return None;
        }
        let last = pk - 1;
        let pair = pack.is_some() && dp != last;
        if pair && !pts[last].is_multiple_of(2) {
            return None;
        }
        let extent = |d: usize| {
            if pair && d == last {
                pts[d] / 2
            } else {
                pts[d]
            }
        };
        let rest: usize = (0..pk).filter(|&d| d != dp).map(extent).product();
        let threads = 256;
        let blocks = rest.div_ceil(threads);
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: predicated in-place update"
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {blocks}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        s.push_str(PRELUDE);
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        let mut ctr = 0usize;
        if !staged.is_empty() {
            self.emit_staged(&mut s, &staged, threads, &mut ctr)?;
        }
        writeln!(s, "    const int t = blockIdx.x * {threads} + threadIdx.x;").unwrap();
        writeln!(s, "    if (t >= {rest}) return;").unwrap();
        let mut rem = "t".to_string();
        for d in (0..pk).rev().filter(|&d| d != dp) {
            let mul = if pair && d == last { " * 2" } else { "" };
            writeln!(s, "    const int p{d} = (({rem}) % {}){mul};", extent(d)).unwrap();
            rem = format!("({rem}) / {}", extent(d));
        }
        let coords: Vec<String> = (0..rank).map(|d| format!("p{d}")).collect();
        let others: Vec<usize> = (0..pk).filter(|&d| d != dp).collect();
        writeln!(s, "    int lo = 0, hi = {};", pts[dp]).unwrap();
        for cd in &conds {
            let a = cd.coefs[dp];
            let base = self.cond_base(cd, &leaves, &others, &coords);
            if a < 0.0 {
                writeln!(s, "    hi = min(hi, (int)ceilf(({base}) / {}));", flit(-a)).unwrap();
            } else {
                writeln!(
                    s,
                    "    lo = max(lo, (int)floorf(-({base}) / {}) + 1);",
                    flit(a)
                )
                .unwrap();
            }
        }
        writeln!(s, "    for (int p{dp} = lo; p{dp} < hi; p{dp}++) {{").unwrap();
        let all_pts: Vec<usize> = (0..pk).collect();
        let mut value = |s: &mut String, coords: &[String], prefix: &str| {
            let el = |i: usize| {
                let l: &Leaf = &leaves[i];
                load(l.buf, l.pack, &addr_expr(&l.addr, coords))
            };
            // the inlined reduction first, as a loop over its coord
            let red_val = match (red, &red_args) {
                (Some((_, rc)), Some(ra)) => {
                    let monoid = rc.monoid.as_ref().unwrap();
                    let acc = format!("{prefix}r");
                    let ind = "            ";
                    writeln!(s, "        float {acc} = {};", flit(monoid.identity[0])).unwrap();
                    writeln!(s, "        {{").unwrap();
                    writeln!(s, "        int rlo = 0, rhi = {};", rc.red[0].n()).unwrap();
                    for cd in &red_bounds {
                        let a = cd.coefs[pk];
                        let base = self.cond_base(cd, &leaves, &all_pts, coords);
                        if a < 0.0 {
                            writeln!(
                                s,
                                "        rhi = min(rhi, (int)ceilf(({base}) / {}));",
                                flit(-a)
                            )
                            .unwrap();
                        } else {
                            writeln!(
                                s,
                                "        rlo = max(rlo, (int)floorf(-({base}) / {}) + 1);",
                                flit(a)
                            )
                            .unwrap();
                        }
                    }
                    writeln!(s, "        for (int p{pk} = rlo; p{pk} < rhi; p{pk}++) {{").unwrap();
                    let rv: Vec<String> = ra
                        .iter()
                        .map(|a| self.emit_val(s, a, coords, ind, &mut ctr, &el))
                        .collect();
                    let x = emit(
                        s,
                        &rc.body,
                        &rc.body_outs,
                        &format!("{prefix}e_"),
                        ind,
                        &|v| rv[v as usize].clone(),
                    );
                    self.combine(s, monoid, 1, ind, |_| acc.clone(), &x);
                    writeln!(s, "        }}").unwrap();
                    writeln!(s, "        }}").unwrap();
                    Some(acc)
                }
                _ => None,
            };
            let vals: Vec<String> = args
                .iter()
                .map(|a| match a {
                    Val::Var(n) if n.is_empty() => red_val.clone().unwrap(),
                    _ => self.emit_val(s, a, coords, "        ", &mut ctr, &el),
                })
                .collect();
            emit(s, &c.body, &[e], prefix, "        ", &|v| {
                vals[v as usize].clone()
            })[0]
                .clone()
        };
        let mut flat = "0".to_string();
        for (d, &np) in pts.iter().enumerate() {
            flat = format!("({flat}) * {np} + p{d}");
        }
        let y = value(&mut s, &coords, "x_");
        let to16 = |v: &str| match pack {
            Some(Packing::F16) => format!("tang_f2h({v})"),
            _ => format!("tang_f2bf({v})"),
        };
        match pack {
            None => writeln!(s, "        out[{flat}] = {y};").unwrap(),
            Some(_) if pair => {
                let mut c1 = coords.clone();
                c1[last] = format!("(p{last} + 1)");
                let y1 = value(&mut s, &c1, "z_");
                writeln!(
                    s,
                    "        ((unsigned*)out)[({flat}) >> 1] = {} | ({} << 16);",
                    to16(&y),
                    to16(&y1)
                )
                .unwrap();
            }
            Some(_) => {
                writeln!(s, "        const int f = {flat};").unwrap();
                writeln!(s, "        unsigned* o = (unsigned*)out + (f >> 1);").unwrap();
                writeln!(s, "        const unsigned h = {};", to16(&y)).unwrap();
                writeln!(
                    s,
                    "        *o = (f & 1) ? ((*o & 0xffffu) | (h << 16)) : ((*o & 0xffff0000u) | h);"
                )
                .unwrap();
            }
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads,
            blocks,
        })
    }

    /// A small [`Monoid::topk`]`(k)` (2 ≤ k < 32) over a short row (at most
    /// 1024 elements) as one warp per point: each lane evaluates its
    /// `⌈n/32⌉` elements into registers, then `k` rounds of a warp argmax
    /// (value descending, the smaller index first on a tie: the monoid's
    /// order) pull the next best, and the lane that held it drops it. The
    /// generic warp combine instead merges two sorted k-lists at every
    /// shuffle step (a MoE router's top-10 of 512 took 40 µs). Same
    /// element contract as [`topk_select`](Self::topk_select).
    fn topk_warp(&self, root: usize) -> Option<Kernel> {
        let c = self.g.comp_at(root)?;
        let m = c.monoid.as_ref()?;
        let k = m.topk_k()?;
        if std::env::var("TANG_TOPK_SELECT").is_ok_and(|v| v == "0") {
            return None;
        }
        if !(2..32).contains(&k) || c.tuple != 1 || c.width() != 2 * k || c.red.is_empty() {
            return None;
        }
        let nv = c.args.len();
        for probe in 0..3 {
            let v: Vec<f64> = (0..nv)
                .map(|i| ((i * 5 + probe * 11) % 13) as f64 * 0.7 - 2.1)
                .collect();
            let o = c.body.eval_many::<f64>(&c.body_outs[2..], &v);
            if o.chunks(2)
                .any(|p| p[0] != 0.0 || p[1] != -(f32::MAX as f64))
            {
                return None;
            }
        }
        let pts = fixed(c.points());
        let red = fixed(&c.red);
        let (pk, rk) = (pts.len(), red.len());
        let n_red: usize = red.iter().product();
        if n_red > 1024 {
            return None;
        }
        let mut leaves = Vec::new();
        let args = c
            .args
            .iter()
            .map(|a| self.tree(a, pk + rk, &mut leaves))
            .collect::<Option<Vec<_>>>()?;
        let n_pts: usize = pts.iter().product();
        let per = n_red.div_ceil(32);
        let mut s = String::new();
        writeln!(s, "// Auto-generated by tang-compile: top-{k} warp select").unwrap();
        writeln!(s, "#define TANG_TILES {n_pts}").unwrap();
        writeln!(s, "#define TANG_THREADS 32").unwrap();
        s.push_str(PRELUDE);
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    const int lane = threadIdx.x;").unwrap();
        writeln!(s, "    const int pt = blockIdx.x;").unwrap();
        let mut rem = "pt".to_string();
        for d in (0..pk).rev() {
            writeln!(s, "    const int p{d} = ({rem}) % {};", pts[d]).unwrap();
            rem = format!("({rem}) / {}", pts[d]);
        }
        writeln!(s, "    float vj[{per}], vw[{per}];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int e = 0; e < {per}; e++) {{").unwrap();
        writeln!(s, "        const int r = lane + 32 * e;").unwrap();
        writeln!(s, "        float ej = 0.0f, ew = -3.40282347e38f;").unwrap();
        writeln!(s, "        if (r < {n_red}) {{").unwrap();
        let mut rr = "r".to_string();
        for d in (0..rk).rev() {
            writeln!(s, "          const int p{} = ({rr}) % {};", pk + d, red[d]).unwrap();
            rr = format!("({rr}) / {}", red[d]);
        }
        let coords: Vec<String> = (0..pk + rk).map(|d| format!("p{d}")).collect();
        let mut ctr = 0usize;
        let el = |i: usize| {
            let l: &Leaf = &leaves[i];
            load(l.buf, l.pack, &addr_expr(&l.addr, &coords))
        };
        let vals: Vec<String> = args
            .iter()
            .map(|a| self.emit_val(&mut s, a, &coords, "          ", &mut ctr, &el))
            .collect();
        let y = emit(
            &mut s,
            &c.body,
            &c.body_outs[..2],
            "x_",
            "          ",
            &|v| vals[v as usize].clone(),
        );
        writeln!(s, "          ej = {}; ew = {};", y[0], y[1]).unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        vj[e] = ej; vw[e] = ew;").unwrap();
        writeln!(s, "    }}").unwrap();
        // k rounds: the lane's best, then the warp's best of those
        writeln!(s, "    #pragma unroll 1").unwrap();
        writeln!(s, "    for (int q = 0; q < {k}; q++) {{").unwrap();
        writeln!(s, "        float bj = vj[0], bw = vw[0]; int bs = 0;").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int e = 1; e < {per}; e++)").unwrap();
        writeln!(s, "            if (vw[e] > bw || (vw[e] == bw && vj[e] < bj)) {{ bj = vj[e]; bw = vw[e]; bs = e; }}").unwrap();
        writeln!(s, "        int bl = lane;").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int o = 16; o > 0; o >>= 1) {{").unwrap();
        writeln!(
            s,
            "            const float oj = __shfl_xor_sync(0xffffffffu, bj, o);"
        )
        .unwrap();
        writeln!(
            s,
            "            const float ow = __shfl_xor_sync(0xffffffffu, bw, o);"
        )
        .unwrap();
        writeln!(
            s,
            "            const int ol = __shfl_xor_sync(0xffffffffu, bl, o);"
        )
        .unwrap();
        writeln!(s, "            if (ow > bw || (ow == bw && (oj < bj || (oj == bj && ol < bl)))) {{ bj = oj; bw = ow; bl = ol; }}").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        if (lane == bl) {{").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int e = 0; e < {per}; e++) if (e == bs) vw[e] = -3.40282347e38f, vj[e] = 3.40282347e38f;").unwrap();
        writeln!(s, "            out[pt * {} + 2 * q] = bj;", 2 * k).unwrap();
        writeln!(s, "            out[pt * {} + 2 * q + 1] = bw;", 2 * k).unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads: 32,
            blocks: n_pts,
        })
    }
}

/// Above this k a [`Monoid::topk`] combine is a loop, not the unrolled
/// k²/2 compare-select network (k = 512 is 2M lines nvrtc never finishes).
const TOPK_UNROLL_MAX: usize = 8;

/// `acc ← top-k(acc ∪ el)` as a plain two-pointer merge of the two sorted
/// lists (same order as [`Monoid::topk`]: larger value, then smaller
/// index). Minimal and looped so nvrtc compiles it; a dedicated top-k
/// kernel replaces it ([`Lowering::topk_select`] for k ≥ 32; this
/// remains for 8 < k < 32).
fn topk_merge(s: &mut String, k: usize, indent: &str, accs: &[String], el: &[String]) {
    let w = 2 * k;
    writeln!(s, "{indent}{{ float ta[{w}], tb[{w}], to[{w}];").unwrap();
    for i in 0..w {
        writeln!(s, "{indent}  ta[{i}] = {}; tb[{i}] = {};", accs[i], el[i]).unwrap();
    }
    // ia + ib = q < k, so neither runs past its list
    writeln!(s, "{indent}  int ia = 0, ib = 0;").unwrap();
    writeln!(s, "{indent}  #pragma unroll 1").unwrap();
    writeln!(s, "{indent}  for (int q = 0; q < {k}; q++) {{").unwrap();
    writeln!(
        s,
        "{indent}    float ai = ta[2 * ia], av = ta[2 * ia + 1], bi = tb[2 * ib], bv = tb[2 * ib + 1];"
    )
    .unwrap();
    writeln!(
        s,
        "{indent}    bool fa = av > bv || (av == bv && ai <= bi);"
    )
    .unwrap();
    writeln!(
        s,
        "{indent}    to[2 * q] = fa ? ai : bi; to[2 * q + 1] = fa ? av : bv;"
    )
    .unwrap();
    writeln!(s, "{indent}    if (fa) ia++; else ib++;").unwrap();
    writeln!(s, "{indent}  }}").unwrap();
    for i in 0..w {
        writeln!(s, "{indent}  {} = to[{i}];", accs[i]).unwrap();
    }
    writeln!(s, "{indent}}}").unwrap();
}

impl TensorGraph {
    /// Whether some node names `root` as its host (`TNode::host`).
    pub(super) fn hosts(&self, root: usize) -> bool {
        let r = TensorId(root as u32);
        self.nodes.iter().any(|n| n.host == Some(r))
    }
}

impl Lowering<'_> {
    /// A small reduction (at most 64 elements a point, Sum or Max, of
    /// width 1) as one thread per point looping over its reduce space: for a
    /// kernel that hosts inlined nodes or side jobs (`TNode::host`) and
    /// has no warp lowering (a depthwise conv over a short window).
    fn thread_reduction(&self, root: usize) -> Option<Kernel> {
        let g = self.g;
        if !g.hosts(root) {
            return None;
        }
        let c = g.comp_at(root)?;
        let m = c.monoid.as_ref()?;
        let kind = m.kind();
        if !matches!(kind, super::MonoidKind::Sum | super::MonoidKind::Max)
            || c.width() != 1
            || c.tuple != 0
            || c.body_outs.len() != 1
            || g.nodes[root].precision.storage != super::Format::F32
            || self.packed.contains_key(&root)
        {
            return None;
        }
        let pts = fixed(c.points());
        let red = fixed(&c.red);
        let (pk, rank) = (pts.len(), pts.len() + red.len());
        let n: usize = pts.iter().product();
        let rn: usize = red.iter().product();
        if rn > 64 {
            return None;
        }
        let mut leaves = Vec::new();
        self.guards.set(true);
        let args = c
            .args
            .iter()
            .map(|a| self.tree(a, rank, &mut leaves))
            .collect::<Option<Vec<_>>>();
        self.guards.set(false);
        let args = args?;
        let threads = 256usize;
        let blocks = n.div_ceil(threads);
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: thread reduction (decode)"
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {blocks}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        s.push_str(PRELUDE);
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    const int t = blockIdx.x * {threads} + threadIdx.x;").unwrap();
        writeln!(s, "    if (t >= {n}) return;").unwrap();
        let mut rem = "t".to_string();
        for d in (0..pk).rev() {
            writeln!(s, "    const int p{d} = ({rem}) % {};", pts[d]).unwrap();
            rem = format!("({rem}) / {}", pts[d]);
        }
        writeln!(s, "    float acc = {};", flit(m.identity[0])).unwrap();
        writeln!(s, "    for (int r = 0; r < {rn}; r++) {{").unwrap();
        let mut rem = "r".to_string();
        for d in (0..red.len()).rev() {
            writeln!(s, "    const int p{} = ({rem}) % {};", pk + d, red[d]).unwrap();
            rem = format!("({rem}) / {}", red[d]);
        }
        let coords: Vec<String> = (0..rank).map(|d| format!("p{d}")).collect();
        let el = |i: usize| {
            let l: &Leaf = &leaves[i];
            load(l.buf, l.pack, &addr_expr(&l.addr, &coords))
        };
        let mut ctr = 0usize;
        let vals: Vec<String> = args
            .iter()
            .map(|a| self.emit_val(&mut s, a, &coords, "    ", &mut ctr, &el))
            .collect();
        let y = emit(&mut s, &c.body, &c.body_outs, "x_", "    ", &|v| {
            vals[v as usize].clone()
        });
        if kind == super::MonoidKind::Sum {
            writeln!(s, "    acc += {};", y[0]).unwrap();
        } else {
            writeln!(s, "    acc = fmaxf(acc, {});", y[0]).unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    out[t] = acc;").unwrap();
        writeln!(s, "}}").unwrap();
        Some(Kernel {
            source: s,
            threads,
            blocks,
        })
    }

    /// Side job `u` of this kernel (`TNode::host`): a map evaluated over
    /// its own points by a grid-stride loop at the top of the kernel
    /// (every thread present, before any early return), its small
    /// reductions staged per block as in `staged_map`, stored to `b{u}`.
    /// One scope, so its names shadow nothing of the kernel's own.
    fn side_job(
        &self,
        u: usize,
        threads: usize,
        blocks: usize,
        bid: &str,
        nb: &str,
        ctr: &mut usize,
    ) -> Option<String> {
        let g = self.g;
        let c = g.comp_at(u)?;
        if c.monoid.is_some()
            || c.tuple != 0
            || g.nodes[u].precision.storage != super::Format::F32
            || self.packed.contains_key(&u)
            || !threads.is_multiple_of(32)
        {
            return None;
        }
        let pts = fixed(c.points());
        let pk = pts.len();
        let w = c.width();
        let n: usize = pts.iter().product();
        let mut leaves = Vec::new();
        let mut staged = Vec::new();
        self.guards.set(true);
        let args = c
            .args
            .iter()
            .map(|a| self.tree_staged(a, pk, &mut leaves, &mut staged))
            .collect::<Option<Vec<_>>>();
        self.guards.set(false);
        let args = args?;
        let mut s = String::new();
        writeln!(s, "    {{ // side job: node {u}").unwrap();
        if !staged.is_empty() {
            self.staged_ok(&staged)?;
            // every block recomputes the statistics: bounded as in
            // `schedule_staged_stats`
            let work: usize = staged
                .iter()
                .map(|&p| g.comp_at(p).map_or(0, |pc| numel(&pc.out) * numel(&pc.red)))
                .sum();
            if work * blocks > 1 << 21 {
                return None;
            }
            self.emit_staged(&mut s, &staged, threads, ctr)?;
        }
        writeln!(
            s,
            "    for (int t = ({bid}) * blockDim.x + threadIdx.x; t < {n}; t += ({nb}) * blockDim.x) {{"
        )
        .unwrap();
        let mut rem = "t".to_string();
        for d in (0..pk).rev() {
            writeln!(s, "    const int p{d} = ({rem}) % {};", pts[d]).unwrap();
            rem = format!("({rem}) / {}", pts[d]);
        }
        let coords: Vec<String> = (0..pk).map(|d| format!("p{d}")).collect();
        let el = |i: usize| {
            let l: &Leaf = &leaves[i];
            load(l.buf, l.pack, &addr_expr(&l.addr, &coords))
        };
        let vals: Vec<String> = args
            .iter()
            .map(|a| self.emit_val(&mut s, a, &coords, "    ", ctr, &el))
            .collect();
        *ctr += 1;
        let y = emit(
            &mut s,
            &c.body,
            &c.body_outs,
            &format!("s{}_", *ctr),
            "    ",
            &|v| vals[v as usize].clone(),
        );
        for (slot, v) in y.iter().enumerate() {
            writeln!(s, "    b{u}[t * {w} + {slot}] = {v};").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    }}").unwrap();
        Some(s)
    }

    /// `kern` with the side jobs `writes` added: their buffers as
    /// parameters after `out` (in order). By default each side job gets
    /// blocks of its own past the kernel's (`⌈points / threads⌉`, at most
    /// 1024), which run it and return, so it runs alongside the kernel's
    /// work instead of in front of it; the kernel body sees its own
    /// `gridDim.x`. `TANG_COKERNEL_BLOCKS=0`: the old layout, every side
    /// job a grid-stride loop at the top run by the kernel's own threads.
    /// `kern` unchanged when there are none.
    fn with_side_jobs(&self, kern: &Kernel, writes: &[TensorId]) -> Option<Kernel> {
        if writes.is_empty() {
            return Some(Kernel {
                source: kern.source.clone(),
                threads: kern.threads,
                blocks: kern.blocks,
            });
        }
        let src = &kern.source;
        let sig = src.find("main0(")?;
        let close = sig + src[sig..].find(", unsigned count)")?;
        let open = close + src[close..].find('{')?;
        let own = !std::env::var("TANG_COKERNEL_BLOCKS").is_ok_and(|v| v == "0");
        // temporaries numbered past anything the kernel uses
        let mut ctr = 1_000_000usize;
        let mut jobs = String::new();
        let mut extra = 0usize;
        if own {
            let b0 = kern.blocks;
            jobs.push_str(&format!("    if (blockIdx.x >= {b0}u) {{\n    const unsigned tang_sb = blockIdx.x - {b0}u;\n"));
            for (q, &u) in writes.iter().enumerate() {
                let n: usize = self
                    .g
                    .comp_at(u.n())
                    .map_or(1, |c| fixed(c.points()).iter().product());
                let e = n.div_ceil(kern.threads).clamp(1, 1024);
                jobs.push_str(&format!(
                    "    {}if (tang_sb < {}u) {{\n",
                    if q > 0 { "else " } else { "" },
                    extra + e
                ));
                jobs.push_str(&self.side_job(
                    u.n(),
                    kern.threads,
                    e,
                    &format!("tang_sb - {extra}u"),
                    &format!("{e}u"),
                    &mut ctr,
                )?);
                jobs.push_str("    }\n");
                extra += e;
            }
            jobs.push_str("    return;\n    }\n");
        } else {
            for &u in writes {
                jobs.push_str(&self.side_job(
                    u.n(),
                    kern.threads,
                    kern.blocks,
                    "blockIdx.x",
                    "gridDim.x",
                    &mut ctr,
                )?);
            }
        }
        let params: String = writes
            .iter()
            .map(|u| format!(", float* __restrict__ b{}", u.n()))
            .collect();
        // the kernel's own body: its grid, not the side jobs' blocks
        let body = if own {
            src[open + 1..].replace("gridDim.x", &format!("{}u", kern.blocks))
        } else {
            src[open + 1..].to_string()
        };
        let mut out = String::with_capacity(src.len() + jobs.len() + params.len());
        out.push_str(&src[..close]);
        out.push_str(&params);
        out.push_str(&src[close..=open]);
        out.push('\n');
        out.push_str(&jobs);
        out.push_str(&body);
        Some(Kernel {
            source: out,
            threads: kern.threads,
            blocks: kern.blocks + extra,
        })
    }
}

impl TensorGraph {
    /// The buffers kernel `root` binds (as `to_program_numbered` gives
    /// them): its inline closure's buffer-backed nodes, then its side
    /// jobs' (`TNode::host`).
    fn host_kernel_reads(&self, root: usize) -> Vec<usize> {
        let mut reads: Vec<usize> = self
            .inline_closure(root)
            .into_iter()
            .filter(|&j| {
                j != root
                    && !self.inlined_in(j, root)
                    && !matches!(self.nodes[j].op, Op::Iota(_) | Op::Reshape(_))
            })
            .collect();
        let r = TensorId(root as u32);
        for u in (0..self.nodes.len()).filter(|&u| self.nodes[u].host == Some(r)) {
            for j in self.host_reads(u, root) {
                if !reads.contains(&j) {
                    reads.push(j);
                }
            }
        }
        reads
    }

    /// Whether kernel `root` lowers natively on CUDA with its side jobs
    /// (the decode lowering's chain, no tensor-core tiles): its threads.
    fn probe_native(
        &self,
        root: usize,
        packed: &HashMap<usize, Packing>,
        opts: &DecodeOpts,
    ) -> Option<(usize, usize)> {
        let reads = self.host_kernel_reads(root);
        let lw = Lowering {
            g: self,
            root,
            guards: Default::default(),
            reads: &reads,
            packed,
            opts,
        };
        let kern = if self.nodes[root].precision.storage.is_block() {
            lw.staged_map(root)?
        } else {
            lw.shared_reduction(root)
                .or_else(|| lw.split_fused(root))
                .or_else(|| lw.warp_reduction(root))
                .or_else(|| lw.staged_map(root))
                .or_else(|| lw.point_map(root))
                .or_else(|| lw.hosted_map(root))
                .or_else(|| lw.thread_reduction(root))?
        };
        let r = TensorId(root as u32);
        let writes: Vec<TensorId> = (0..self.nodes.len())
            .filter(|&u| self.nodes[u].host == Some(r))
            .map(|u| TensorId(u as u32))
            .collect();
        let Some(all) = lw.with_side_jobs(&kern, &writes) else {
            if std::env::var("TANG_COKERNEL_DEBUG").is_ok() {
                eprintln!(
                    "cokernel: n{root} lowered ({}) but its side jobs {writes:?} did not",
                    kern.source.lines().next().unwrap_or("")
                );
            }
            return None;
        };
        // blocks with the side jobs' own
        Some((all.threads, all.blocks))
    }

    /// Schedule (decode, CUDA): multi-output kernels (`TNode::host`), so
    /// one launch writes several nodes. Two rules, both on graph
    /// structure alone:
    ///
    /// - **Inline and store.** A map `P` read by several kernels is
    ///   evaluated inside the first of them, `C` (recomputed from `P`'s
    ///   own inputs, which `C` binds instead of `P`), and `C` also stores
    ///   `P` for the later readers: a residual add folded into the norm
    ///   that reads it, a norm into the Q8x32 copy of it, a conv window
    ///   into the conv. Only when every other reader comes after `C`
    ///   and `C`'s space is at most 4× `P`'s (no broadcast recompute);
    ///   nodes `P` hosted are re-homed to `C`.
    /// - **Side job.** A map `U` (a recurrent state's update, a ring
    ///   buffer's shift) is evaluated by the first kernel `K` after its
    ///   last producer that comes before every reader of `U`, as a
    ///   grid-stride loop of its own at the top of `K`.
    ///
    /// A host must lower natively (checked by lowering it); in-place
    /// updates, their donors' readers, outputs stored packed and int8
    /// GEMVs stay as they are. Returns `(inlined, side jobs)`.
    pub fn schedule_cokernels(
        &mut self,
        outputs: &[TensorId],
        opts: &DecodeOpts,
    ) -> (usize, usize) {
        let n = self.nodes.len();
        let live = self.live(outputs);
        let input_node = |g: &TensorGraph, k: usize| {
            (0..n).find(|&i| matches!(g.nodes[i].op, Op::Input(j) if j == k))
        };
        let packed: HashMap<usize, Packing> = opts
            .packed
            .iter()
            .filter_map(|&(k, p)| input_node(self, k).map(|i| (i, p)))
            .collect();
        let inplace_out: Vec<usize> = opts
            .inplace
            .iter()
            .map(|&(t, _)| self.storage(t).n())
            .collect();
        let donors: Vec<usize> = opts
            .inplace
            .iter()
            .filter_map(|&(_, k)| input_node(self, k))
            .collect();
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut odd = vec![false; n];
        for i in (0..n).filter(|&i| live[i]) {
            match &self.nodes[i].op {
                Op::Comp(c) => {
                    for a in &c.args {
                        let p = a.src.n();
                        if !readers[p].contains(&i) {
                            readers[p].push(i);
                        }
                        // a gathered read has no decode tree (a guarded
                        // one does, where `Lowering::guards` is on)
                        if a.view.gather.is_some() {
                            odd[p] = true;
                        }
                    }
                }
                Op::Iota(_) | Op::Input(_) => {}
                _ => self.deps(i).into_iter().for_each(|p| odd[p] = true),
            }
        }
        let kernel = |g: &TensorGraph, i: usize| {
            live[i] && !g.nodes[i].inline && g.nodes[i].host.is_none() && g.comp_at(i).is_some()
        };
        let plain_map = |g: &TensorGraph, i: usize| {
            g.comp_at(i)
                .is_some_and(|c| c.monoid.is_none() && c.tuple == 0)
                && kernel(g, i)
                && !odd[i]
                && !inplace_out.contains(&i)
                && !donors.contains(&i)
                && g.nodes[i].int_mma.is_none()
                && g.nodes[i].precision.storage == super::Format::F32
                && g.nodes[i].split == 0
                && g.nodes[i].tile == (0, 0, 0)
                && !packed.contains_key(&i)
        };
        let host_ok = |g: &TensorGraph, c: usize| {
            kernel(g, c)
                && g.nodes[c].int_mma.is_none()
                && !inplace_out.contains(&c)
                && g.nodes[c].split == 0
                && g.nodes[c].tile == (0, 0, 0)
        };
        // where node `j`'s value is written: its host's kernel, or its own
        let at = |g: &TensorGraph, j: usize| g.nodes[j].host.map_or(j, |h| h.n());
        // the kernels that read `p`: an inlined reader (a norm's
        // statistic) stands for the kernels it is evaluated in
        let kernel_readers = |g: &TensorGraph, p: usize| {
            let mut kr: Vec<usize> = Vec::new();
            let mut stack = readers[p].clone();
            let mut seen = vec![false; n];
            while let Some(r) = stack.pop() {
                if std::mem::replace(&mut seen[r], true) {
                    continue;
                }
                if g.nodes[r].inline {
                    stack.extend(readers[r].iter().copied());
                } else {
                    kr.push(r);
                }
            }
            kr
        };
        let debug = std::env::var("TANG_COKERNEL_DEBUG").is_ok();
        let rules = std::env::var("TANG_COKERNEL_RULES").unwrap_or_else(|_| "ab".into());
        let mut inlined = 0;
        for p in (0..n).filter(|_| rules.contains('a')) {
            let is_output = outputs.iter().any(|o| o.n() == p);
            if !plain_map(self, p) || readers[p].is_empty() {
                if debug && self.comp_at(p).is_some_and(|c| c.monoid.is_none()) && kernel(self, p) {
                    eprintln!(
                        "cokernel: n{p} not a plain map (odd {} inplace {} st {:?} readers {:?})",
                        odd[p],
                        inplace_out.contains(&p),
                        self.nodes[p].precision.storage,
                        readers[p]
                    );
                }
                continue;
            }
            let kr = kernel_readers(self, p);
            let Some(&c) = kr.iter().min() else { continue };
            if !host_ok(self, c) || (kr.len() < 2 && !is_output) {
                if debug {
                    eprintln!(
                        "cokernel: n{p} into n{c}: host {} readers {kr:?}",
                        host_ok(self, c)
                    );
                }
                continue;
            }
            let pn = numel(self.comp_at(p).unwrap().points());
            let cn = numel(&self.comp_at(c).unwrap().space());
            if cn > 4 * pn {
                if debug {
                    eprintln!("cokernel: n{p} into n{c}: space {cn} > 4 x {pn}");
                }
                continue;
            }
            // a donor read inside `P` moves with it: not past an update
            if self.host_reads(p, p).iter().any(|j| donors.contains(j)) {
                if debug {
                    eprintln!("cokernel: n{p} into n{c}: reads a donor");
                }
                continue;
            }
            let tp = TensorId(p as u32);
            let moved: Vec<usize> = (0..n).filter(|&q| self.nodes[q].host == Some(tp)).collect();
            if moved.iter().any(|&q| {
                kernel_readers(self, q)
                    .iter()
                    .any(|&r| r != p && at(self, r) <= c)
            }) {
                if debug {
                    eprintln!("cokernel: n{p} into n{c}: a re-homed node is read before");
                }
                continue;
            }
            let tc = TensorId(c as u32);
            self.nodes[p].host = Some(tc);
            for &q in &moved {
                self.nodes[q].host = Some(tc);
            }
            if self.probe_native(c, &packed, opts).is_some() {
                inlined += 1;
            } else {
                if debug {
                    let cc = self.comp_at(c).unwrap();
                    eprintln!(
                        "cokernel: n{p} into n{c}: no native lowering (P {:?}, C {:?} red {:?} monoid {:?} args {:?} st {:?})",
                        self.nodes[p].shape,
                        self.nodes[c].shape,
                        cc.red,
                        cc.monoid.as_ref().map(|m| m.kind()),
                        cc.args.iter().map(|a| (a.src.n(), a.view.guard)).collect::<Vec<_>>(),
                        self.nodes[c].precision.storage
                    );
                }
                self.nodes[p].host = None;
                for &q in &moved {
                    self.nodes[q].host = Some(tp);
                }
            }
        }
        let mut jobs = 0;
        let own_blocks = !std::env::var("TANG_COKERNEL_BLOCKS").is_ok_and(|v| v == "0");
        // SMs a host may fill and still leave room (H200: 132)
        let sms: usize = std::env::var("TANG_COKERNEL_SMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(132);
        for u in (0..n).filter(|_| rules.contains('b')) {
            // a kernel that already hosts side jobs keeps its launch
            if !plain_map(self, u) || self.hosts(u) {
                continue;
            }
            let reads = self.host_reads(u, u);
            if reads.iter().any(|j| donors.contains(j)) {
                continue;
            }
            let lo = reads
                .iter()
                .filter(|&&j| self.comp_at(j).is_some() || matches!(self.nodes[j].op, Op::Scan(_)))
                .map(|&j| at(self, j))
                .max();
            let hi = readers[u].iter().map(|&r| at(self, r)).min().unwrap_or(n);
            let start = lo.map_or(0, |l| l + 1);
            let un = numel(self.comp_at(u).unwrap().points());
            for k in start..hi {
                if k == u || !host_ok(self, k) {
                    continue;
                }
                self.nodes[u].host = Some(TensorId(k as u32));
                let fits = |(t, b): (usize, usize)| {
                    if own_blocks {
                        // the side job's blocks run beside the host's only
                        // where the host leaves SMs idle
                        b <= sms
                    } else {
                        // enough threads that the loop stays short
                        t * b * 64 >= un
                    }
                };
                match self.probe_native(k, &packed, opts) {
                    Some(tb) if fits(tb) => {
                        jobs += 1;
                        break;
                    }
                    _ => self.nodes[u].host = None,
                }
            }
        }
        (inlined, jobs)
    }
}
