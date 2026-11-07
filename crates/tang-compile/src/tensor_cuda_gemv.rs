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
use super::{fixed, numel, Arg, Form, Monoid, Op, TensorGraph, TensorId};

/// How a graph input is stored on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// Unsigned 4-bit integers, eight per 32-bit word, element `i` in
    /// bits `4·(i % 8)` of word `i / 8` (MLX's layout). Values `0..=15`.
    U4,
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
            Packing::Bf16 | Packing::F16 => len.div_ceil(2),
        }
    }

    /// The f32 value of the 16-bit `bits` (an unsigned expression).
    fn half(self, bits: &str) -> String {
        match self {
            Packing::Bf16 => format!("__uint_as_float(({bits}) << 16)"),
            Packing::F16 => format!("tang_h2f({bits})"),
            Packing::U4 => unreachable!(),
        }
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
        let mut program = self.to_program_numbered(outputs, Target::Wgsl);
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
        for k in &mut program.kernels {
            let root = k.node.index();
            let reads: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
            let lw = Lowering {
                g: self,
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
                None => mma()
                    .or_else(|| lw.warp_reduction(root))
                    .or_else(|| lw.point_map(root)),
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
            match got {
                Some(kern) => {
                    k.source = match target {
                        DecodeTarget::Cuda => kern.source,
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

impl super::TNode {
    fn op_input(&self) -> Option<usize> {
        match self.op {
            Op::Input(k) => Some(k),
            _ => None,
        }
    }
}

struct Lowering<'a> {
    g: &'a TensorGraph,
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
        if a.view.guard {
            return None;
        }
        let j = a.src.n();
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
        if !self.g.nodes[j].inline || mc.monoid.is_some() || mc.width() != 1 || mc.tuple != 0 {
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
            Val::Leaf(i) => leaf(*i),
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

    /// A reduction (or a map that is its epilogue) as warp kernels.
    fn warp_reduction(&self, root: usize) -> Option<Kernel> {
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
            .any(|l| unit(l) && l.pack == Some(Packing::U4) && aligned(l, 32))
            && ri.is_multiple_of(32)
        {
            32
        } else if ri.is_multiple_of(4) && leaves.iter().any(|l| unit(l) && l.pack.is_none()) {
            4
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
                        Some(Packing::U4) => vec == 32 && aligned(l, 32),
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
            let bytes = |l: &Leaf| match l.pack {
                Some(Packing::U4) => 0.5,
                Some(Packing::Bf16 | Packing::F16) => 2.0,
                None => 4.0,
            };
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
        let rt = reg.map_or(1, |r| r.1);
        let on_reg = |cd: &Cond| reg.is_some_and(|(d, _)| cd.coefs[d] != 0.0);
        // a runtime scalar of a bound is loaded per point group: it may not
        // vary along the register dim
        let rt_on_reg = |cd: &Cond| {
            reg.is_some_and(|(d, _)| cd.rt.iter().any(|&(_, i)| leaves[i].addr.coefs[d] != 0))
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
        let ks = if groups < 512 { 8.max(warps) } else { 1 };
        let threads = if ks > 1 { 32 * ks } else { 32 * warps };
        let gpb = threads / 32 / ks;
        let blocks = if !point_conds.is_empty() && ks == 1 {
            // a grid-stride loop over the live groups
            groups.div_ceil(gpb).min(2048)
        } else {
            groups.div_ceil(gpb)
        };
        let chunks_inner = ri / vec;
        let n_chunks: usize = red[..red.len() - 1].iter().product::<usize>() * chunks_inner;

        let w_out = epi.map_or(w, |e| e.width());
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: warp reduction (decode)"
        )
        .unwrap();
        writeln!(
            s,
            "// {groups} groups x {rt} points, {} reduce in chunks of {vec}, {ks} warps per group",
            n_chunks * vec
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
                "    for (int grp = blockIdx.x * {gpb} + warp; grp < {limit}; grp += gridDim.x * {gpb}) {{"
            )
            .unwrap();
        } else {
            writeln!(s, "    const int grp = blockIdx.x;").unwrap();
            writeln!(s, "    const int ks = warp;").unwrap();
        }
        // group -> point coords (the register dim in steps of rt)
        let mut rem = "grp".to_string();
        let order: Vec<usize> = (0..pk)
            .rev()
            .filter(|&d| Some(d) != dyn_dim)
            .chain(dyn_dim)
            .collect();
        for d in order {
            let n = if reg.is_some_and(|(r, _)| r == d) {
                pts[d] / rt
            } else {
                pts[d]
            };
            let mul = if reg.is_some_and(|(r, _)| r == d) {
                format!(" * {rt}")
            } else {
                String::new()
            };
            writeln!(s, "    const int p{d} = (({rem}) % {n}){mul};").unwrap();
            rem = format!("({rem}) / {n}");
        }
        let mut coords: Vec<String> = (0..pk)
            .map(|d| {
                if reg.is_some_and(|(r, _)| r == d) {
                    format!("(p{d} + r)")
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
            writeln!(s, "    const bool live = {};", all.join(" && ")).unwrap();
        } else {
            writeln!(s, "    const bool live = true;").unwrap();
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
            "    for (int c = {lo_chunk} + lane + 32 * ks; c < {hi_chunk}; c += {}) {{",
            32 * ks
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
        let dep_r = |l: &Leaf| reg.is_some_and(|(d, _)| l.addr.coefs[d] != 0);
        let at0: Vec<String> = coords.iter().map(|x| x.replace("(kb + e)", "kb")).collect();
        for (i, l) in leaves.iter().enumerate() {
            let nr = if dep_r(l) { rt } else { 1 };
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
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int r = 0; r < {rt}; r++) {{").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int e = 0; e < {vec}; e++) {{").unwrap();
        let ri_ = |l: &Leaf| if dep_r(l) { "r" } else { "0" };
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
                    Some(p @ (Packing::Bf16 | Packing::F16)) => p.half(&format!(
                        "(((&w{i}[{}][e >> 2].x)[(e >> 1) & 1] >> (16 * (e & 1))) & 0xffffu)",
                        ri_(l)
                    )),
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
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    }}").unwrap();

        // shuffle combine
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int off = 16; off > 0; off >>= 1) {{").unwrap();
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
        writeln!(s, "        if (lane != r) continue;").unwrap();
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
        for (slot, v) in vals.iter().enumerate() {
            writeln!(s, "        out[o * {w_out} + {slot}] = {v};").unwrap();
        }
        writeln!(s, "    }}").unwrap();
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

    /// A map reading packed inputs (the dequantization of a weight a
    /// tensor-core tile then reads): one thread per point.
    fn point_map(&self, root: usize) -> Option<Kernel> {
        let c = self.g.comp_at(root)?;
        if c.monoid.is_some() || !self.reads.iter().any(|j| self.packed.contains_key(j)) {
            return None;
        }
        let pts = fixed(c.points());
        let pk = pts.len();
        let w = c.width();
        let mut leaves = Vec::new();
        let args = c
            .args
            .iter()
            .map(|a| self.tree(a, pk, &mut leaves))
            .collect::<Option<Vec<_>>>()?;
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
        let args = c
            .args
            .iter()
            .enumerate()
            .map(|(ai, a)| {
                if red.is_some_and(|(r, _)| r == ai) {
                    Some(Val::Var(String::new()))
                } else {
                    self.tree(&widen(a), rank, &mut leaves)
                }
            })
            .collect::<Option<Vec<_>>>()?;
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
        if pack == Some(Packing::U4) {
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
        let mut ctr = 0usize;
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
                    Val::Var(_) => red_val.clone().unwrap(),
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
}
