//! OpenCL lowering for decode-shaped programs: the OpenCL C twin of
//! `tensor_cuda_gemv.rs`, for GPUs with OpenCL and no CUDA/Metal (the
//! Booster K1's Adreno 740, where OpenCL dispatch is several times cheaper
//! than Vulkan's).
//!
//! [`TensorGraph::to_opencl_decode`] generates the kernels once in CUDA
//! spelling, as Metal and Vulkan do, and [`transliterate`]s them. OpenCL C
//! has pointers, so the shape is CUDA's:
//!
//! - **The calling convention is `to_opencl`'s**: `main0(b{j}…, out, uint
//!   count)` with the reads in `WgslKernel::reads` order, `count / threads`
//!   groups of `threads` items (one dimension). As for Metal, an in-place
//!   update does not bind its donor: its old value is read through `out`.
//! - **Warp butterflies** (`__shfl_xor_sync` over 32 lanes, XOR offsets
//!   ≤ 16) are `TANG_SHFL_XOR(x, off)`, which the runtime defines for the
//!   device: a subgroup shuffle where the driver has one (Adreno
//!   subgroups are 64 or 128 wide; an offset ≤ 16 stays inside an aligned
//!   run of 32 lanes, so the sums are a 32-lane warp's), else through
//!   local memory ([`OPENCL_SHFL_LOCAL`], for drivers without subgroups,
//!   e.g. Apple's).
//! - **16-bit conversions are core OpenCL** (`vload_half`,
//!   `vstore_half_rte`: round to nearest even, as CUDA's `cvt.rn`), so no
//!   `cl_khr_fp16` is needed.
//! - `(&v.x)[i]`, a vector's element by index, is `((T*)&v)[i]`.
//!
//! Unlowered kernels are `to_opencl`'s WGSL transpile; they never read
//! packed inputs (`to_decode` checks).
use std::fmt::Write;

use super::cuda_gemv::{DecodeOpts, DecodeProgram, DecodeTarget, PRELUDE};
use super::{TensorGraph, TensorId};

/// [`PRELUDE`] in OpenCL C.
const OPENCL_PRELUDE: &str = r#"#define select(f, t, c) ((c) ? (t) : (f))
#define TANG_EL4(v, i) (((i) & 3) == 0 ? (v).s0 : ((i) & 3) == 1 ? (v).s1 : ((i) & 3) == 2 ? (v).s2 : (v).s3)
#define TANG_EL2(v, i) (((i) & 1) == 0 ? (v).s0 : (v).s1)
inline float tang_u4(__global const float* b, int a) {
    uint q = (((__global const uint*)b)[a >> 3] >> ((a & 7) * 4)) & 15u;
    return as_float(0x4b000000u | q) - 8388608.0f;
}
inline float tang_u3(__global const float* b, int a) {
    __global const uint* w = (__global const uint*)b;
    uint bit = 3u * (uint)a, o = bit & 31u, lo = w[bit >> 5] >> o;
    uint q = (o > 29u ? lo | (w[(bit >> 5) + 1] << (32u - o)) : lo) & 7u;
    return as_float(0x4b000000u | q) - 8388608.0f;
}
inline float tang_bf16(__global const float* b, int a) {
    uint w = ((__global const uint*)b)[a >> 1];
    return as_float((a & 1) ? (w & 0xffff0000u) : (w << 16));
}
inline float tang_h2f(uint h) {
    ushort s = (ushort)h;
    return vload_half(0, (const __private half*)&s);
}
inline uint tang_f2h(float f) {
    ushort s;
    vstore_half_rte(f, 0, (__private half*)&s);
    return (uint)s;
}
inline uint tang_f2bf(float f) {
    uint b = as_uint(f);
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}
inline float tang_f16(__global const float* b, int a) {
    uint w = ((__global const uint*)b)[a >> 1];
    return tang_h2f((a & 1) ? (w >> 16) : (w & 0xffffu));
}
"#;

/// `TANG_SHFL_XOR` through local memory, for drivers without subgroup
/// shuffles. Its barriers need every work-item of the group to reach
/// them, which a warp kernel's grid-stride loop over groups does not
/// guarantee (warps of one group can run different trip counts): it was
/// right on Apple's driver and wrong on the Adreno, so it is a fallback
/// for checking on hosts, not a production path.
pub const OPENCL_SHFL_LOCAL: &str = r#"#define TANG_SHFL_LOCAL 1
#define TANG_SHFL_XOR(x, off) tang_shfl_xor(tang_xch, tang_tid, (x), (off))
inline float tang_shfl_xor(__local float* xch, int tid, float x, int off) {
    barrier(CLK_LOCAL_MEM_FENCE);
    xch[tid] = x;
    barrier(CLK_LOCAL_MEM_FENCE);
    return xch[tid ^ off];
}
"#;

/// `TANG_SHFL_XOR` as `cl_khr_subgroup_shuffle`'s `sub_group_shuffle_xor`.
pub const OPENCL_SHFL_KHR: &str = r#"#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define TANG_SHFL_XOR(x, off) sub_group_shuffle_xor((x), (uint)(off))
"#;

/// `TANG_SHFL_XOR` as `cl_qcom_subgroup_shuffle` (Adreno).
pub const OPENCL_SHFL_QCOM: &str = r#"#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define TANG_SHFL_XOR(x, off) qcom_sub_group_shuffle_xor((x), (uint)(off), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
"#;

impl TensorGraph {
    /// [`to_cuda_decode`](Self::to_cuda_decode) for OpenCL: the same
    /// kernels in OpenCL C (see the module docs). Each native kernel starts
    /// with `// TANG_SHFL` where the runtime puts its `TANG_SHFL_XOR`
    /// definition ([`OPENCL_SHFL_LOCAL`], [`OPENCL_SHFL_KHR`],
    /// [`OPENCL_SHFL_QCOM`]). Buffers of packed inputs are sized in words.
    pub fn to_opencl_decode(&self, outputs: &[TensorId], opts: &DecodeOpts) -> DecodeProgram {
        self.to_decode(outputs, opts, DecodeTarget::OpenCl)
    }
}

/// Prefill on OpenCL: the reductions to run as register tiles, and the
/// graph with their schedules. A sum over at least 64 terms whose two
/// innermost point dims are both at least 16 wide (tokens × outputs) and
/// that reads no packed input — the dequantized weights of a prefill are
/// materialized f32 maps — gets `TANG_CL_PREFILL_TILE` (default 8x8x16:
/// 8×8 threads, k chunk 16) with `TANG_CL_PREFILL_REG` registers (default
/// 4x8), the shape the Adreno 740 ran fastest for hear's projections. The
/// warp GEMV it replaces re-reads the weight once per token.
pub(super) fn prefill_tiles(g: &TensorGraph, opts: &DecodeOpts) -> (TensorGraph, Vec<usize>) {
    let parse = |k: &str, d: &[u32]| -> Vec<u32> {
        std::env::var(k)
            .ok()
            .map(|v| {
                v.split('x')
                    .filter_map(|x| x.parse().ok())
                    .collect::<Vec<u32>>()
            })
            .filter(|v| v.len() == d.len())
            .unwrap_or_else(|| d.to_vec())
    };
    let t = parse("TANG_CL_PREFILL_TILE", &[8, 8, 16]);
    let r = parse("TANG_CL_PREFILL_REG", &[4, 8]);
    let input_node = |k: usize| {
        (0..g.nodes.len()).find(|&i| matches!(g.nodes[i].op, super::Op::Input(j) if j == k))
    };
    let packed: Vec<usize> = opts
        .packed
        .iter()
        .filter_map(|&(k, _)| input_node(k))
        .collect();
    let mut out = g.clone();
    let mut tiles = Vec::new();
    for i in 0..g.nodes.len() {
        let n = &g.nodes[i];
        if n.inline {
            continue;
        }
        let Some(c) = g.comp_at(i) else { continue };
        let red: usize = c.red.iter().map(|d| d.n()).product();
        let shape: Vec<usize> = n.shape.iter().map(|d| d.n()).collect();
        let wide = shape.len() >= 2 && shape[shape.len() - 1] >= 16 && shape[shape.len() - 2] >= 16;
        let reads_packed = g.deps(i).iter().any(|d| packed.contains(d));
        // a product of two tensors (a matmul), not a norm's Σ x² over one:
        // the q/k head norms ([tokens, heads] over head_dim) ran as a
        // 64-item WGSL tile, 1.2 ms each on the Adreno, where the warp
        // reduction spreads them over every lane
        let mut srcs: Vec<usize> = c.args.iter().map(|a| a.src.n()).collect();
        srcs.sort_unstable();
        srcs.dedup();
        let bilinear = srcs.len() >= 2;
        if c.monoid.is_some() && red >= 64 && wide && !reads_packed && bilinear {
            out.set_schedule(TensorId::from_index(i), 0, (t[0], t[1], t[2]), (r[0], r[1]));
            tiles.push(i);
        }
    }
    (out, tiles)
}

/// Marker line (after the kernel's first line) that the runtime replaces
/// with its shuffle definition.
pub const SHFL_MARKER: &str = "// TANG_SHFL\n";

/// Index of the `)` closing the `(` at `open`.
fn close_paren(s: &str, open: usize) -> usize {
    let mut depth = 0;
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return open + i;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced parentheses in a decode kernel");
}

/// Replace every `head(ARGS)` by `f(ARGS)` (ARGS balanced).
fn rewrite_calls(s: &str, head: &str, f: impl Fn(&str) -> String) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(head) {
        out.push_str(&rest[..i]);
        let open = i + head.len() - 1;
        let close = close_paren(rest, open);
        out.push_str(&f(&rest[open + 1..close]));
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

/// Replace the identifier `from` (whole tokens only) by `to`.
fn rename_ident(s: &str, from: &str, to: &str) -> String {
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if ident(b[i]) {
            let start = i;
            while i < b.len() && ident(b[i]) {
                i += 1;
            }
            let tok = &s[start..i];
            out.push_str(if tok == from { to } else { tok });
        } else {
            let start = i;
            while i < b.len() && !ident(b[i]) {
                i += 1;
            }
            out.push_str(&s[start..i]);
        }
    }
    out
}

/// A decode kernel in CUDA spelling (from `tensor_cuda_gemv.rs`) as
/// OpenCL C. `reads` are the buffers it binds (graph node numbers, the
/// `b{j}` of the source) before `out`; `donor`'s loads (an in-place
/// update's old value) read `out`.
pub(super) fn transliterate(src: &str, reads: &[usize], donor: Option<usize>) -> String {
    assert!(
        src.contains(PRELUDE),
        "decode kernel without the CUDA prelude"
    );
    let threads: usize = src
        .lines()
        .find_map(|l| l.strip_prefix("#define TANG_THREADS "))
        .and_then(|v| v.trim().parse().ok())
        .expect("TANG_THREADS");
    let mut body = String::new();
    for line in src.replace(PRELUDE, OPENCL_PRELUDE).lines() {
        if line.starts_with("extern \"C\" __global__") {
            let mut params: Vec<String> = reads
                .iter()
                .map(|j| format!("__global const float* restrict b{j}"))
                .collect();
            params.push("__global float* restrict out".into());
            params.push("uint count".into());
            writeln!(
                body,
                "__kernel __attribute__((reqd_work_group_size({threads}, 1, 1))) void main0({}) {{",
                params.join(", ")
            )
            .unwrap();
            writeln!(body, "    const int tang_bid = (int)get_group_id(0);").unwrap();
            writeln!(body, "    const int tang_tid = (int)get_local_id(0);").unwrap();
            writeln!(body, "#ifdef TANG_SHFL_LOCAL").unwrap();
            writeln!(body, "    __local float tang_xch[{threads}];").unwrap();
            writeln!(body, "#endif").unwrap();
            writeln!(body, "    if (tang_bid >= TANG_TILES) return;").unwrap();
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    // __ldg(p) is *p
    let s = rewrite_calls(&body, "__ldg(", |a| format!("(*({a}))"));
    // (&v.x)[i] — a vector's element by index. The index is a constant once
    // the loops unroll; a select chain folds to a register read, where
    // taking the vector's address (`((T*)&v)[i]`) would put the whole
    // register array in private memory (0.95 ms per GEMV on the Adreno,
    // ~5x Vulkan). The width comes from the array's declaration.
    let width = |name: &str| -> usize {
        for (ty, w) in [("float4 ", 4), ("uint4 ", 4), ("uint2 ", 2)] {
            if s.contains(&format!("{ty}{name}[")) {
                return w;
            }
        }
        panic!("vector element of an undeclared array `{name}` in a decode kernel")
    };
    let mut out = String::with_capacity(s.len());
    let mut rest: &str = &s;
    while let Some(i) = rest.find("(&") {
        out.push_str(&rest[..i]);
        let close = close_paren(rest, i);
        let inner = &rest[i + 2..close];
        let Some(v) = inner.strip_suffix(".x") else {
            panic!("unexpected address-of in a decode kernel: {inner}")
        };
        let name: String = v
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let after = &rest[close + 1..];
        assert!(after.starts_with('['), "vector element without an index");
        let iclose = after.find(']').expect("index ]");
        let idx = &after[1..iclose];
        let m = if width(&name) == 4 {
            "TANG_EL4"
        } else {
            "TANG_EL2"
        };
        out.push_str(&format!("{m}({v}, {idx})"));
        rest = &after[iclose + 1..];
    }
    out.push_str(rest);
    let mut s = out;
    let subs: [(&str, &str); 15] = [
        ("__shfl_xor_sync(0xffffffffu, ", "TANG_SHFL_XOR("),
        ("__syncthreads()", "barrier(CLK_LOCAL_MEM_FENCE)"),
        ("__shared__ ", "__local "),
        ("__uint_as_float(", "as_float("),
        ("__float_as_uint(", "as_uint("),
        ("ceilf(", "ceil("),
        ("floorf(", "floor("),
        ("threadIdx.x", "tang_tid"),
        ("blockIdx.x", "tang_bid"),
        ("gridDim.x", "TANG_TILES"),
        ("(const uint4*)", "(__global const uint4*)"),
        ("(const uint2*)", "(__global const uint2*)"),
        ("(const float4*)", "(__global const float4*)"),
        ("(const unsigned*)", "(__global const unsigned*)"),
        ("(unsigned*)", "(__global unsigned*)"),
    ];
    for (a, b) in subs {
        s = s.replace(a, b);
    }
    s = s.replace("unsigned* o ", "__global unsigned* o ");
    if let Some(d) = donor {
        s = rename_ident(&s, &format!("b{d}"), "out");
    }
    for bad in [
        "__ldg",
        "__shfl",
        "__device__",
        "asm(",
        "extern",
        "__restrict__",
    ] {
        assert!(
            !s.contains(bad),
            "untransliterated `{bad}` in a decode kernel:\n{s}"
        );
    }
    // after the kernel's first line (its `// … groups x … points` summary)
    match s.split_once('\n') {
        Some((first, rest)) if first.starts_with("//") => format!("{first}\n{SHFL_MARKER}{rest}"),
        _ => format!("{SHFL_MARKER}{s}"),
    }
}

/// Register block and work-group of [`TensorGraph::opencl_mma_kernel`]:
/// `TANG_CL_MM=RMxRNxWG` (rows per item, columns per item, work-group
/// size); default 4x4x64.
pub fn mm_shape() -> (usize, usize, usize) {
    std::env::var("TANG_CL_MM")
        .ok()
        .and_then(|v| {
            let p: Vec<usize> = v.split('x').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 3 && p.iter().all(|&x| x > 0)).then(|| (p[0], p[1], p[2]))
        })
        .unwrap_or((4, 4, 64))
}

/// Register block of [`TensorGraph::opencl_mma_kernel_q4`] for `m` rows
/// (tokens): `TANG_CL_MM` (default 4x4x64; RM 0: automatic rows, a tuner
/// candidate — 7x4 ran 33% slower than 4x4 on the Adreno). Automatic rows split `m` into the fewest
/// blocks of at most 8 rows (20 tokens: 3 blocks of 7, so each packed
/// weight is read 3 times instead of 5), padding at most a row per block.
pub fn mm_shape_q4(m: usize) -> (usize, usize, usize) {
    let (rm, rn, wg) = std::env::var("TANG_CL_MM")
        .ok()
        .and_then(|v| {
            let p: Vec<usize> = v.split('x').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 3 && p[1] > 0 && p[2] > 0).then(|| (p[0], p[1], p[2]))
        })
        .unwrap_or((4, 4, 64));
    let rm = if rm == 0 {
        m.div_ceil(m.div_ceil(8))
    } else {
        rm
    };
    (rm.max(1), rn, wg)
}

/// fp16 arithmetic in the packed-weight prefill kernel
/// ([`TensorGraph::opencl_mma_kernel_q4`]): `TANG_CL_PREFILL_F16=1`.
/// Activations and dequantized weights are rounded to half and each
/// 8-term partial dot product is taken in half (`cl_khr_fp16`); the
/// partials accumulate in f32. Without `cl_khr_fp16` (hosts) the kernel
/// rounds the operands and partials to half and computes in f32 — the
/// same roundings, for accuracy checks.
pub fn prefill_f16() -> bool {
    std::env::var("TANG_CL_PREFILL_F16").is_ok_and(|v| v == "1")
}

/// The line of a kernel the runtime precedes with `#define TANG_FP16 1`
/// when the device lists `cl_khr_fp16` (Apple's compiler defines the
/// `cl_khr_fp16` macro without supporting half arithmetic).
pub const FP16_MARKER: &str = "// TANG_FP16?\n";

/// fp16 helpers of the packed-weight prefill kernel (see [`prefill_f16`]),
/// after [`FP16_MARKER`].
const OPENCL_F16: &str = r#"#ifdef TANG_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half4 tang_h4;
#define TANG_H4(v) convert_half4(v)
#define TANG_HDOT(a, b, c, d) ((float)(dot((a), (b)) + dot((c), (d))))
#else
typedef float4 tang_h4;
inline float tang_r1(float v) {
    return tang_h2f(tang_f2h(v));
}
inline float4 tang_r4(float4 v) {
    return (float4)(tang_r1(v.x), tang_r1(v.y), tang_r1(v.z), tang_r1(v.w));
}
#define TANG_H4(v) tang_r4(v)
#define TANG_HDOT(a, b, c, d) tang_r1(tang_r1(dot((a), (b))) + tang_r1(dot((c), (d))))
#endif
"#;

/// Work-items per register block of the packed-weight prefill kernel,
/// splitting k between them: `TANG_CL_MM_LANES` (default 16; 1 is one item
/// per block), at most and dividing the work-group size.
pub fn mm_lanes(wg: usize) -> usize {
    let l = std::env::var("TANG_CL_MM_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&l| l > 0)
        .unwrap_or(16)
        .min(wg);
    if wg.is_multiple_of(l) {
        l
    } else {
        1
    }
}

/// `None` from [`TensorGraph::opencl_mma_kernel_q4`], logged with the
/// line that rejected `root` under `TANG_CL_LOG=1`.
fn q4_none(root: usize, line: u32) -> Option<super::cuda_gemv::Kernel> {
    if std::env::var("TANG_CL_LOG").is_ok_and(|v| v == "1") {
        eprintln!("tang-cl: node {root}: no q4 register block (tensor_opencl_gemv.rs:{line})");
    }
    None
}

/// Packed words per lane step of the q4 prefill kernel: `TANG_CL_MM_VEC`
/// (1, the default: one word; 4: a uint4 per column, 32 weights, with the
/// affine `s·Σqx + b·Σx` form — 1.5x slower on the Mac, a tuner candidate).
pub fn mm_vec() -> usize {
    match std::env::var("TANG_CL_MM_VEC").as_deref() {
        Ok("4") => 4,
        _ => 1,
    }
}

/// The staged packed-weight prefill kernel, the default (`TANG_CL_MM_KIND=reg`: the register block):
/// `(WG, LANES, RN, KC)` from `TANG_CL_STAGE=WGxLANESxRNxKC` (default
/// 128x8x2x128: 32 columns per group, the fastest on the Mac). A work-group owns `WG/LANES·RN` columns and stages the
/// activations of up to `32/RN` rows (all of a 20-token chunk) through
/// local memory in k chunks of `KC`; each item keeps RN columns × those
/// rows in registers and its `LANES` take adjacent packed words, so each
/// packed weight is read once per pass, coalesced, and dequantized once
/// for every row.
pub fn stage_shape() -> Option<(usize, usize, usize, usize)> {
    // the default on OpenCL prefill (K1, 20 tokens: 374 vs 680 ms for the
    // register block); TANG_CL_MM_KIND=reg for the register block
    if std::env::var("TANG_CL_MM_KIND").as_deref() == Ok("reg") {
        return None;
    }
    let p: Vec<usize> = std::env::var("TANG_CL_STAGE")
        .ok()
        .map(|v| v.split('x').filter_map(|x| x.parse().ok()).collect())
        .filter(|p: &Vec<usize>| p.len() == 4 && p.iter().all(|&x| x > 0))
        .unwrap_or_else(|| vec![128, 8, 2, 128]);
    Some((p[0], p[1], p[2], p[3]))
}

/// First line of a kernel that is OpenCL C already (no transliteration).
pub const OPENCL_NATIVE: &str = "// tang-compile: OpenCL C (native)";

impl TensorGraph {
    /// The stores of an RM×RN register block `acc{i}_{j}` (rows `r0..`,
    /// columns `c0..`) of the reduction `red_node`, through the comp's
    /// epilogue when `epilogue`.
    #[allow(clippy::too_many_arguments)]
    fn mm_stores(
        &self,
        s: &mut String,
        c: &super::Comp,
        red_node: usize,
        epilogue: bool,
        pts: &[usize],
        (dy, dx): (usize, usize),
        (m, n): (usize, usize),
        (rm, rn): (usize, usize),
    ) {
        use super::tile::emit_scalar;
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let w_out = c.width();
        let prank = c.points().len();
        for i in 0..rm {
            for j in 0..rn {
                writeln!(s, "    {{").unwrap();
                writeln!(s, "        int p{dy} = r0 + {i}, p{dx} = c0 + {j};").unwrap();
                writeln!(s, "        if (p{dy} < {m} && p{dx} < {n}) {{").unwrap();
                writeln!(s, "            float red = acc{i}_{j};").unwrap();
                let vals: Vec<String> = if epilogue {
                    let mut arg_vals = Vec::new();
                    for a in &c.args {
                        if a.src.n() == red_node {
                            arg_vals.push("red".to_string());
                            continue;
                        }
                        let addr = self.flat_addr(a, prank);
                        let mut e = format!("{}", addr.offset);
                        for (d, &coef) in addr.coefs.iter().enumerate() {
                            if coef != 0 {
                                e.push_str(&format!(" + {coef} * p{d}"));
                            }
                        }
                        arg_vals.push(format!("b{}[{e}]", storage(a.src.n())));
                    }
                    let mut body = String::new();
                    let outs = emit_scalar(
                        &mut body,
                        &c.body,
                        &c.body_outs,
                        &format!("e{i}_{j}_"),
                        &|v| arg_vals[v as usize].clone(),
                    );
                    for line in body.lines() {
                        writeln!(s, "            {}", line.replacen("let ", "float ", 1)).unwrap();
                    }
                    outs
                } else {
                    vec!["red".to_string()]
                };
                let mut flat = String::from("0");
                for (d, &np) in pts.iter().enumerate() {
                    flat = format!("({flat}) * {np} + p{d}");
                }
                writeln!(s, "            int o = {flat};").unwrap();
                for (slot, v) in vals.iter().enumerate() {
                    writeln!(s, "            out[o * {w_out} + {slot}] = {v};").unwrap();
                }
                writeln!(s, "        }}").unwrap();
                writeln!(s, "    }}").unwrap();
            }
        }
    }

    /// A prefill matmul over 4-bit weights — `C[t][n] = Σ_{g,j} x[t][g·J+j]
    /// · W(q[n][g][j], s[n][g], b[n][g])`, where `W` is the inlined
    /// dequantization (MLX: `q·s + b`) of packed inputs — as the register
    /// block of [`opencl_mma_kernel`](Self::opencl_mma_kernel) with the
    /// dequantization at load: per k step of 8 each column's packed word
    /// is read once (4 bytes for 8 weights), unpacked and dequantized in
    /// registers and shared by the block's RM rows. No f32 copy of the
    /// weight is ever written (a materialized map is 8x the packed bytes,
    /// written and read once per pass). `packed`: storage node → packing.
    /// Needs `x` and the nibbles unit-stride along `j`, `J` a multiple of
    /// 8, and the per-group values constant along `j`.
    pub(super) fn opencl_mma_kernel_q4(
        &self,
        root: usize,
        reads: &[usize],
        packed: &std::collections::HashMap<usize, super::cuda_gemv::Packing>,
    ) -> Option<super::cuda_gemv::Kernel> {
        use super::cuda_gemv::Packing;
        use super::tile::emit_scalar;
        use tang_expr::Node;
        let c = self.comp_at(root)?;
        // the reduction: `root`, or the one inlined reduction its epilogue
        // reads (any red dims: `tiled_reduction` caps the minor ones at 32,
        // which a 6144-wide down projection's 96 groups exceed)
        let red_node = if c.monoid.is_some() {
            root
        } else {
            let mut found = None;
            for a in &c.args {
                let p = a.src.n();
                if self.nodes[p].inline && self.comp_at(p).is_some_and(|pc| pc.monoid.is_some()) {
                    if found.is_some_and(|f| f != p) {
                        return q4_none(root, line!());
                    }
                    found = Some(p);
                }
            }
            match found {
                Some(p) => p,
                None => return q4_none(root, line!()),
            }
        };
        let rc = self.comp_at(red_node)?;
        let epilogue = red_node != root;
        if rc.red.len() != 2 || rc.tuple != 0 || rc.body_outs.len() != 1 {
            return q4_none(root, line!());
        }
        if rc.monoid.as_ref().map(super::Monoid::kind) != Some(super::MonoidKind::Sum) {
            return q4_none(root, line!());
        }
        let pts: Vec<usize> = super::fixed(rc.points());
        if pts.len() != 2 || c.points().len() != 2 {
            return q4_none(root, line!());
        }
        if epilogue && self.epilogue_read(c, red_node)? != super::View::identity(2, 2).forms {
            return q4_none(root, line!());
        }
        let (gs, js) = (rc.red[0].n(), rc.red[1].n());
        if js % 8 != 0 {
            return q4_none(root, line!());
        }
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let rank = 4;
        // the body: x · E, E over packed args — substituted into the
        // reduction's body, or an inlined map (the dequantization) it reads
        let nodes = rc.body.nodes_slice();
        let Node::Mul(l, r) = nodes[rc.body_outs[0].index() as usize] else {
            return q4_none(root, line!());
        };
        let var_of = |e: tang_expr::ExprId| match nodes[e.index() as usize] {
            Node::Var(v) => Some(v as usize),
            _ => None,
        };
        let is_x = |v: usize| {
            let a = &rc.args[v];
            !a.view.guard
                && reads.contains(&storage(a.src.n()))
                && !packed.contains_key(&storage(a.src.n()))
        };
        let (vx, e) = match (var_of(l), var_of(r)) {
            (Some(v), _) if is_x(v) => (v, r),
            (_, Some(v)) if is_x(v) => (v, l),
            _ => return None,
        };
        let map = var_of(e).and_then(|v| {
            let a = &rc.args[v];
            let p = a.src.n();
            let mc = self.comp_at(p)?;
            (self.nodes[p].inline && !a.view.guard && mc.monoid.is_none() && mc.red.is_empty())
                .then_some((v, mc))
        });
        let (ebody, eout, eargs): (
            &tang_expr::ExprGraph,
            tang_expr::ExprId,
            Vec<Option<super::Arg>>,
        ) = match map {
            Some((v, mc)) if mc.body_outs.len() == 1 => {
                let outer = &rc.args[v];
                let args = mc
                    .args
                    .iter()
                    .map(|s| {
                        Some(super::Arg {
                            src: s.src,
                            view: s.view.compose(&outer.view, rank),
                        })
                    })
                    .collect();
                (&mc.body, mc.body_outs[0], args)
            }
            Some(_) => return None,
            None => {
                let live = rc.body.live_set(&[e]);
                if live
                    .iter()
                    .any(|&i| matches!(nodes[i], Node::Var(v) if v as usize == vx))
                {
                    return q4_none(root, line!());
                }
                let args = rc
                    .args
                    .iter()
                    .enumerate()
                    .map(|(v, a)| (v != vx).then(|| a.clone()))
                    .collect();
                (&rc.body, e, args)
            }
        };
        let ax = self.flat_addr(&rc.args[vx], rank);
        if ax.coefs[3] != 1 || ax.coefs[1] != 0 && ax.coefs[0] != 0 {
            return q4_none(root, line!());
        }
        // which point dim is the token (x varies along it): rows dy, columns dx
        let (dy, dx) = if ax.coefs[0] != 0 { (0, 1) } else { (1, 0) };
        let (m, n) = (pts[dy], pts[dx]);
        if m < 2 || ax.coefs[dx] != 0 {
            return q4_none(root, line!());
        }
        // E's args: the nibbles (U4, along j) and per-group values
        enum Role {
            Nibble {
                buf: usize,
                addr: super::msl_mma::Addr,
            },
            Group {
                buf: usize,
                addr: super::msl_mma::Addr,
                pack: Packing,
            },
        }
        let mut roles: Vec<Option<Role>> = Vec::new();
        let mut nibbles = 0;
        for a in &eargs {
            let Some(a) = a else {
                roles.push(None);
                continue;
            };
            if a.view.guard {
                return q4_none(root, line!());
            }
            let buf = storage(a.src.n());
            if !reads.contains(&buf) {
                return q4_none(root, line!());
            }
            let addr = self.flat_addr(a, rank);
            if addr.coefs[dy] != 0 {
                return q4_none(root, line!());
            }
            match packed.get(&buf) {
                Some(Packing::U4) => {
                    if addr.coefs[3] != 1
                        || addr.offset % 8 != 0
                        || addr.coefs[dx] % 8 != 0
                        || addr.coefs[2] % 8 != 0
                    {
                        return q4_none(root, line!());
                    }
                    nibbles += 1;
                    roles.push(Some(Role::Nibble { buf, addr }));
                }
                Some(&pack) if addr.coefs[3] == 0 => {
                    roles.push(Some(Role::Group { buf, addr, pack }))
                }
                _ => return None,
            }
        }
        if nibbles == 0 {
            return q4_none(root, line!());
        }
        'stage: {
            let Some((wg, lanes, rn, kc)) = stage_shape() else {
                break 'stage;
            };
            // the staged kernel (see `stage_shape`)
            let k_total = gs * js;
            let kc = kc.min(k_total);
            if lanes == 0
                || wg % lanes != 0
                || kc % (8 * lanes) != 0
                || k_total % kc != 0
                || kc % js.min(kc) != 0
                || (js % kc != 0 && kc % js != 0)
            {
                // infeasible shape: the register-block kernel
                q4_none(root, line!());
                break 'stage;
            }
            let ci = wg / lanes; // column items per group
            let tile = ci * rn; // columns per group
            let rmax = (32 / rn).max(1);
            let passes = m.div_ceil(rmax);
            let rp = m.div_ceil(passes); // rows per pass
            if rp * kc * 4 + wg * rp * rn * 4 > 32 << 10 {
                // infeasible shape: the register-block kernel
                q4_none(root, line!());
                break 'stage;
            }
            let ntiles = n.div_ceil(tile);
            let groups = ntiles * passes;
            let xbuf = storage(rc.args[vx].src.n());
            let mut s = String::new();
            writeln!(
                s,
                "{OPENCL_NATIVE}: {m}x{n}x{k_total} matmul over 4-bit weights, staged: {rp} rows x {rn} cols per item, {lanes} lanes along k, k chunk {kc}, groups of {wg}"
            )
            .unwrap();
            s.push_str(OPENCL_PRELUDE);
            let params: Vec<String> = reads
                .iter()
                .map(|j| format!("__global const float* restrict b{j}"))
                .chain([
                    "__global float* restrict out".to_string(),
                    "uint count".to_string(),
                ])
                .collect();
            writeln!(
                s,
                "__kernel __attribute__((reqd_work_group_size({wg}, 1, 1))) void main0({}) {{",
                params.join(", ")
            )
            .unwrap();
            writeln!(s, "    __local float xs[{}];", rp * kc).unwrap();
            writeln!(s, "    __local float part[{}];", wg * rp * rn).unwrap();
            writeln!(
                s,
                "    int t = (int)get_local_id(0), grp = (int)get_group_id(0);"
            )
            .unwrap();
            writeln!(s, "    int lane = t % {lanes}, ci = t / {lanes};").unwrap();
            writeln!(s, "    int tile = grp % {ntiles}, pass = grp / {ntiles};").unwrap();
            writeln!(
                s,
                "    int r0 = pass * {rp}, c0 = tile * {tile} + ci * {rn};"
            )
            .unwrap();
            for j in 0..rn {
                writeln!(s, "    int col{j} = min(c0 + {j}, {});", n - 1).unwrap();
            }
            for i in 0..rp {
                for j in 0..rn {
                    writeln!(s, "    float acc{i}_{j} = 0.0f;").unwrap();
                }
            }
            writeln!(s, "    for (int k0 = 0; k0 < {k_total}; k0 += {kc}) {{").unwrap();
            // stage x[r0.., k0..k0+kc] (rows clamped at the edge)
            writeln!(s, "        barrier(CLK_LOCAL_MEM_FENCE);").unwrap();
            writeln!(
                s,
                "        for (int e = t; e < {}; e += {wg}) {{",
                rp * kc / 4
            )
            .unwrap();
            writeln!(
                s,
                "            int r = e / {}, k4 = (e % {}) * 4;",
                kc / 4,
                kc / 4
            )
            .unwrap();
            writeln!(
                s,
                "            int kk = k0 + k4, row = min(r0 + r, {});",
                m - 1
            )
            .unwrap();
            // x address: offset + coefs[dy]·row + coefs[2]·(k / J) + (k % J)
            writeln!(
                s,
                "            vstore4(vload4(0, b{xbuf} + {} + {} * row + {} * (kk / {js}) + (kk % {js})), 0, xs + r * {kc} + k4);",
                ax.offset, ax.coefs[dy], ax.coefs[2]
            )
            .unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        barrier(CLK_LOCAL_MEM_FENCE);").unwrap();
            writeln!(
                s,
                "        for (int kw = lane; kw < {}; kw += {lanes}) {{",
                kc / 8
            )
            .unwrap();
            writeln!(
                s,
                "            int kk = k0 + kw * 8, kg = kk / {js}, kj = kk % {js};"
            )
            .unwrap();
            for j in 0..rn {
                for (v, role) in roles.iter().enumerate() {
                    match role {
                        Some(Role::Group { buf, addr, pack }) => {
                            let f = match pack {
                                Packing::Bf16 => "tang_bf16",
                                Packing::F16 => "tang_f16",
                                Packing::U4 => "tang_u4",
                                Packing::U3 => "tang_u3",
                            };
                            writeln!(
                                s,
                                "            float g{v}_{j} = {f}(b{buf}, {} + {} * col{j} + {} * kg);",
                                addr.offset, addr.coefs[dx], addr.coefs[2]
                            )
                            .unwrap();
                        }
                        Some(Role::Nibble { buf, addr }) => {
                            writeln!(
                                s,
                                "            uint w{v}_{j} = ((__global const uint*)b{buf})[({} + {} * col{j} + {} * kg + kj) >> 3];",
                                addr.offset, addr.coefs[dx], addr.coefs[2]
                            )
                            .unwrap();
                        }
                        None => {}
                    }
                }
                let mut ws = Vec::new();
                for q in 0..8 {
                    let var = |v: u16| -> String {
                        match &roles[v as usize] {
                            Some(Role::Nibble { .. }) => {
                                format!("(float)((w{v}_{j} >> {}) & 15u)", 4 * q)
                            }
                            Some(Role::Group { .. }) => format!("g{v}_{j}"),
                            None => unreachable!("x in the dequantization"),
                        }
                    };
                    let mut body = String::new();
                    let outs = emit_scalar(&mut body, ebody, &[eout], &format!("d{j}_{q}_"), &var);
                    for line in body.lines() {
                        writeln!(s, "            {}", line.replacen("let ", "float ", 1)).unwrap();
                    }
                    ws.push(outs[0].clone());
                }
                writeln!(
                    s,
                    "            float4 wl{j} = (float4)({}, {}, {}, {}), wh{j} = (float4)({}, {}, {}, {});",
                    ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7]
                )
                .unwrap();
            }
            writeln!(s, "            __local const float* xk = xs + kw * 8;").unwrap();
            for i in 0..rp {
                writeln!(
                    s,
                    "            {{ float4 xl = vload4(0, xk + {}), xh = vload4(1, xk + {});",
                    i * kc,
                    i * kc
                )
                .unwrap();
                for j in 0..rn {
                    writeln!(
                        s,
                        "              acc{i}_{j} += dot(xl, wl{j}) + dot(xh, wh{j});"
                    )
                    .unwrap();
                }
                writeln!(s, "            }}").unwrap();
            }
            writeln!(s, "        }}").unwrap();
            writeln!(s, "    }}").unwrap();
            if lanes > 1 {
                for i in 0..rp {
                    for j in 0..rn {
                        writeln!(s, "    part[{} + t] = acc{i}_{j};", (i * rn + j) * wg).unwrap();
                    }
                }
                writeln!(s, "    barrier(CLK_LOCAL_MEM_FENCE);").unwrap();
                writeln!(s, "    if (lane != 0) return;").unwrap();
                for i in 0..rp {
                    for j in 0..rn {
                        writeln!(s, "    acc{i}_{j} = 0.0f;").unwrap();
                        writeln!(
                            s,
                            "    for (int l = 0; l < {lanes}; l++) acc{i}_{j} += part[{} + t + l];",
                            (i * rn + j) * wg
                        )
                        .unwrap();
                    }
                }
            }
            self.mm_stores(
                &mut s,
                c,
                red_node,
                epilogue,
                &pts,
                (dy, dx),
                (m, n),
                (rp, rn),
            );
            writeln!(s, "}}").unwrap();
            return Some(super::cuda_gemv::Kernel {
                source: s,
                threads: wg,
                blocks: groups,
            });
        }
        let f16 = prefill_f16();
        let (rm, rn, wg) = mm_shape_q4(m);
        // the partials' local array within 32 KB (the Adreno's local memory)
        let lanes = if rm * rn * wg * 4 <= 32 << 10 {
            mm_lanes(wg)
        } else {
            1
        };
        let mb = m.div_ceil(rm);
        let nb = n.div_ceil(rn);
        let items = mb * nb;
        let mut s = String::new();
        writeln!(
            s,
            "{OPENCL_NATIVE}: {m}x{n}x{} matmul over 4-bit weights, {rm}x{rn} registers, {lanes} lanes along k, groups of {wg}{}{}",
            gs * js,
            if f16 { ", fp16" } else { "" },
            if mm_vec() == 4 && !f16 && js % 32 == 0 { ", uint4" } else { "" }
        )
        .unwrap();
        s.push_str(OPENCL_PRELUDE);
        if f16 {
            s.push_str(FP16_MARKER);
            s.push_str(OPENCL_F16);
        }
        let params: Vec<String> = reads
            .iter()
            .map(|j| format!("__global const float* restrict b{j}"))
            .chain([
                "__global float* restrict out".to_string(),
                "uint count".to_string(),
            ])
            .collect();
        writeln!(
            s,
            "__kernel __attribute__((reqd_work_group_size({wg}, 1, 1))) void main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        // `lanes` adjacent work-items share a block and split k word by
        // word, so a column's packed row is read in contiguous runs of
        // 4·lanes bytes (one item per block strides whole rows apart: an
        // uncoalesced 4-byte load per column, which small caches pay for)
        writeln!(
            s,
            "    int gid = (int)get_global_id(0), lane = gid % {lanes};"
        )
        .unwrap();
        writeln!(s, "    int blk = gid / {lanes}, live = blk < {items};").unwrap();
        writeln!(s, "    int g = min(blk, {});", items - 1).unwrap();
        writeln!(s, "    int cb = g % {nb}, rb = g / {nb};").unwrap();
        writeln!(s, "    int r0 = rb * {rm}, c0 = cb * {rn};").unwrap();
        let xbuf = storage(rc.args[vx].src.n());
        for i in 0..rm {
            writeln!(
                s,
                "    __global const float* a{i} = b{xbuf} + {} + {} * min(r0 + {i}, {});",
                ax.offset,
                ax.coefs[dy],
                m - 1
            )
            .unwrap();
        }
        for j in 0..rn {
            writeln!(s, "    int col{j} = min(c0 + {j}, {});", n - 1).unwrap();
        }
        for i in 0..rm {
            for j in 0..rn {
                writeln!(s, "    float acc{i}_{j} = 0.0f;").unwrap();
            }
        }
        let wpg = js / 8;
        // vec4: each lane takes 4 consecutive packed words (32 weights, one
        // group) per step: a 16-byte load per column, 8 float4 of x per row
        let vec4 = !f16 && wpg % 4 == 0 && mm_vec() == 4;
        let step = if vec4 { 4 } else { 1 };
        // affine dequantization `q·s + b` (MLX): Σ q·x and Σ x per step,
        // scaled once — `affine` = (nibble var, scale var, bias var)
        let affine = vec4
            .then(|| {
                let en = ebody.nodes_slice();
                let var = |e: tang_expr::ExprId| match en[e.index() as usize] {
                    Node::Var(v) => Some(v as usize),
                    _ => None,
                };
                let Node::Add(l, r) = en[eout.index() as usize] else {
                    return None;
                };
                let (m, b) = match (
                    en[l.index() as usize],
                    var(r),
                    en[r.index() as usize],
                    var(l),
                ) {
                    (Node::Mul(x, y), Some(b), _, _) => ((x, y), b),
                    (_, _, Node::Mul(x, y), Some(b)) => ((x, y), b),
                    _ => return None,
                };
                let (x, y) = (var(m.0)?, var(m.1)?);
                let is_n = |v: usize| matches!(roles[v], Some(Role::Nibble { .. }));
                let is_g = |v: usize| matches!(roles[v], Some(Role::Group { .. }));
                let (q, sc) = if is_n(x) && is_g(y) {
                    (x, y)
                } else if is_n(y) && is_g(x) {
                    (y, x)
                } else {
                    return None;
                };
                is_g(b).then_some((q, sc, b))
            })
            .flatten();
        writeln!(
            s,
            "    for (int kw = lane * {step}; kw < {}; kw += {}) {{",
            gs * wpg,
            lanes * step
        )
        .unwrap();
        writeln!(s, "        int kg = kw / {wpg}, kj = (kw % {wpg}) * 8;").unwrap();
        // per-group values of each column
        for (v, role) in roles.iter().enumerate() {
            if let Some(Role::Group { buf, addr, pack }) = role {
                let f = match pack {
                    Packing::Bf16 => "tang_bf16",
                    Packing::F16 => "tang_f16",
                    Packing::U4 => "tang_u4",
                    Packing::U3 => "tang_u3",
                };
                for j in 0..rn {
                    writeln!(
                        s,
                        "        float g{v}_{j} = {f}(b{buf}, {} + {} * col{j} + {} * kg);",
                        addr.offset, addr.coefs[dx], addr.coefs[2]
                    )
                    .unwrap();
                }
            }
        }
        if vec4 {
            writeln!(s, "        {{").unwrap();
            writeln!(s, "            int kx = {} * kg + kj;", ax.coefs[2]).unwrap();
            for j in 0..rn {
                for (v, role) in roles.iter().enumerate() {
                    if let Some(Role::Nibble { buf, addr }) = role {
                        writeln!(
                            s,
                            "            uint4 w{v}_{j} = vload4(0, (__global const uint*)b{buf} + (({} + {} * col{j} + {} * kg + kj) >> 3));",
                            addr.offset, addr.coefs[dx], addr.coefs[2]
                        )
                        .unwrap();
                    }
                }
            }
            // partial sums of this step: Σ q·x (affine) or Σ w·x, and Σ x
            for i in 0..rm {
                for j in 0..rn {
                    writeln!(s, "            float d{i}_{j} = 0.0f;").unwrap();
                }
                if affine.is_some() {
                    writeln!(s, "            float4 xs{i} = (float4)(0.0f);").unwrap();
                }
            }
            // nibbles as floats: 0x4b000000 | q is 2^23 + q
            let nib = |v: usize, j: usize, u: usize| {
                let c = ["x", "y", "z", "w"][u / 2];
                let sh = if u.is_multiple_of(2) {
                    "(uint4)(0u, 4u, 8u, 12u)"
                } else {
                    "(uint4)(16u, 20u, 24u, 28u)"
                };
                format!("(as_float4((uint4)(0x4b000000u) | (((uint4)(w{v}_{j}.{c}) >> {sh}) & (uint4)(15u))) - (float4)(8388608.0f))")
            };
            // one float4 of k at a time: RM x and RN weight vectors live
            for u in 0..8 {
                writeln!(s, "            {{").unwrap();
                for i in 0..rm {
                    writeln!(s, "                float4 x{i} = vload4({u}, a{i} + kx);").unwrap();
                    if affine.is_some() {
                        writeln!(s, "                xs{i} += x{i};").unwrap();
                    }
                }
                for j in 0..rn {
                    match affine {
                        Some((q, _, _)) => {
                            writeln!(s, "                float4 q{j} = {};", nib(q, j, u)).unwrap();
                        }
                        None => {
                            let mut ws = Vec::new();
                            for e in 0..4 {
                                let var = |v: u16| -> String {
                                    match &roles[v as usize] {
                                        Some(Role::Nibble { .. }) => {
                                            format!("{}.s{e}", nib(v as usize, j, u))
                                        }
                                        Some(Role::Group { .. }) => format!("g{v}_{j}"),
                                        None => unreachable!("x in the dequantization"),
                                    }
                                };
                                let mut body = String::new();
                                let outs = emit_scalar(
                                    &mut body,
                                    ebody,
                                    &[eout],
                                    &format!("d{j}_{u}_{e}_"),
                                    &var,
                                );
                                for line in body.lines() {
                                    writeln!(
                                        s,
                                        "                {}",
                                        line.replacen("let ", "float ", 1)
                                    )
                                    .unwrap();
                                }
                                ws.push(outs[0].clone());
                            }
                            writeln!(
                                s,
                                "                float4 q{j} = (float4)({}, {}, {}, {});",
                                ws[0], ws[1], ws[2], ws[3]
                            )
                            .unwrap();
                        }
                    }
                }
                for i in 0..rm {
                    for j in 0..rn {
                        writeln!(s, "                d{i}_{j} += dot(x{i}, q{j});").unwrap();
                    }
                }
                writeln!(s, "            }}").unwrap();
            }
            for i in 0..rm {
                if affine.is_some() {
                    writeln!(s, "            float xt{i} = dot(xs{i}, (float4)(1.0f));").unwrap();
                }
                for j in 0..rn {
                    match affine {
                        Some((_, sc, bi)) => writeln!(
                            s,
                            "            acc{i}_{j} += g{sc}_{j} * d{i}_{j} + g{bi}_{j} * xt{i};"
                        )
                        .unwrap(),
                        None => writeln!(s, "            acc{i}_{j} += d{i}_{j};").unwrap(),
                    }
                }
            }
        } else {
            writeln!(s, "        {{").unwrap();
            writeln!(s, "            int kx = {} * kg + kj;", ax.coefs[2]).unwrap();
            for i in 0..rm {
                if f16 {
                    writeln!(
                        s,
                        "            tang_h4 xl{i} = TANG_H4(vload4(0, a{i} + kx)), xh{i} = TANG_H4(vload4(1, a{i} + kx));"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "            float4 xl{i} = vload4(0, a{i} + kx), xh{i} = vload4(1, a{i} + kx);"
                    )
                    .unwrap();
                }
            }
            for j in 0..rn {
                for (v, role) in roles.iter().enumerate() {
                    if let Some(Role::Nibble { buf, addr }) = role {
                        writeln!(
                            s,
                            "            uint w{v}_{j} = ((__global const uint*)b{buf})[({} + {} * col{j} + {} * kg + kj) >> 3];",
                            addr.offset, addr.coefs[dx], addr.coefs[2]
                        )
                        .unwrap();
                    }
                }
                let mut ws = Vec::new();
                for q in 0..8 {
                    let var = |v: u16| -> String {
                        match &roles[v as usize] {
                            Some(Role::Nibble { .. }) => {
                                format!("(float)((w{v}_{j} >> {}) & 15u)", 4 * q)
                            }
                            Some(Role::Group { .. }) => format!("g{v}_{j}"),
                            None => unreachable!("x in the dequantization"),
                        }
                    };
                    let mut body = String::new();
                    let outs = emit_scalar(&mut body, ebody, &[eout], &format!("d{j}_{q}_"), &var);
                    for line in body.lines() {
                        writeln!(s, "            {}", line.replacen("let ", "float ", 1)).unwrap();
                    }
                    ws.push(outs[0].clone());
                }
                let ty = if f16 { "tang_h4" } else { "float4" };
                let wrap = |v: String| if f16 { format!("TANG_H4({v})") } else { v };
                writeln!(
                    s,
                    "            {ty} wl{j} = {}, wh{j} = {};",
                    wrap(format!(
                        "(float4)({}, {}, {}, {})",
                        ws[0], ws[1], ws[2], ws[3]
                    )),
                    wrap(format!(
                        "(float4)({}, {}, {}, {})",
                        ws[4], ws[5], ws[6], ws[7]
                    ))
                )
                .unwrap();
            }
            for i in 0..rm {
                for j in 0..rn {
                    if f16 {
                        writeln!(
                            s,
                            "            acc{i}_{j} += TANG_HDOT(xl{i}, wl{j}, xh{i}, wh{j});"
                        )
                        .unwrap();
                    } else {
                        writeln!(
                            s,
                            "            acc{i}_{j} += dot(xl{i}, wl{j}) + dot(xh{i}, wh{j});"
                        )
                        .unwrap();
                    }
                }
            }
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        if lanes > 1 {
            writeln!(s, "    __local float part[{}];", wg * rm * rn).unwrap();
            writeln!(s, "    int t = (int)get_local_id(0);").unwrap();
            for i in 0..rm {
                for j in 0..rn {
                    writeln!(s, "    part[{} + t] = acc{i}_{j};", (i * rn + j) * wg).unwrap();
                }
            }
            writeln!(s, "    barrier(CLK_LOCAL_MEM_FENCE);").unwrap();
            writeln!(s, "    if (lane != 0) return;").unwrap();
            for i in 0..rm {
                for j in 0..rn {
                    writeln!(s, "    acc{i}_{j} = 0.0f;").unwrap();
                    writeln!(
                        s,
                        "    for (int l = 0; l < {lanes}; l++) acc{i}_{j} += part[{} + t + l];",
                        (i * rn + j) * wg
                    )
                    .unwrap();
                }
            }
        }
        writeln!(s, "    if (!live) return;").unwrap();
        self.mm_stores(
            &mut s,
            c,
            red_node,
            epilogue,
            &pts,
            (dy, dx),
            (m, n),
            (rm, rn),
        );
        writeln!(s, "}}").unwrap();
        Some(super::cuda_gemv::Kernel {
            source: s,
            threads: wg,
            blocks: (items * lanes).div_ceil(wg),
        })
    }

    /// A prefill matmul (`C[m][n] = Σ_k A[m][k]·B[n][k]`, plus its
    /// epilogue) as an OpenCL register-block kernel, when both operands are
    /// contiguous along the reduce axis: each work-item keeps an RM×RN block
    /// of sums ([`mm_shape`]) and walks k four at a time with `vload4`, so
    /// every load is a contiguous 16-byte read and each pair of loaded
    /// vectors feeds RM·RN dot products — the dot-product form of the
    /// register-blocked sgemm the K1 probe ran at 510 GFLOP/s, without a
    /// local-memory stage (the Adreno's tiled kernels ran at ~54 GFLOP/s).
    pub(super) fn opencl_mma_kernel(
        &self,
        root: usize,
        reads: &[usize],
    ) -> Option<super::cuda_gemv::Kernel> {
        let c = self.comp_at(root)?;
        let t = self.bilinear_tile(root, reads)?;
        if t.scales.iter().any(|&k| k != 1.0) {
            return None;
        }
        if t.terms.len() != 1
            || t.outs.len() != 1
            || t.terms[0].map_a.is_some()
            || t.terms[0].map_b.is_some()
        {
            return None;
        }
        let (dx, dy, kdim) = (t.dx, t.dy, t.kdim);
        let (m, n, k_total) = (t.m, t.n, t.k_total);
        let (addr_a, addr_b) = (&t.addr_a, &t.addr_b);
        // both operands unit-stride along k, and a whole number of vec4s
        if addr_a.coefs[kdim] != 1 || addr_b.coefs[kdim] != 1 || k_total % 4 != 0 {
            return None;
        }
        let (rm, rn, wg) = mm_shape();
        let pts = &t.pts;
        let mb = m.div_ceil(rm);
        let nb = n.div_ceil(rn);
        let lead: usize = t.others.iter().map(|&d| pts[d]).product();
        let items = mb * nb * lead;
        let lead_term = |addr: &super::msl_mma::Addr| -> String {
            let mut s = format!("{}", addr.offset);
            for &d in &t.others {
                if addr.coefs[d] != 0 {
                    s.push_str(&format!(" + {} * p{d}", addr.coefs[d]));
                }
            }
            s
        };
        let mut s = String::new();
        writeln!(
            s,
            "{OPENCL_NATIVE}: {m}x{n}x{k_total} matmul, {rm}x{rn} registers, groups of {wg}"
        )
        .unwrap();
        s.push_str(OPENCL_PRELUDE);
        let params: Vec<String> = reads
            .iter()
            .map(|j| format!("__global const float* restrict b{j}"))
            .chain([
                "__global float* restrict out".to_string(),
                "uint count".to_string(),
            ])
            .collect();
        writeln!(
            s,
            "__kernel __attribute__((reqd_work_group_size({wg}, 1, 1))) void main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    int g = (int)get_global_id(0);").unwrap();
        writeln!(s, "    if (g >= {items}) return;").unwrap();
        writeln!(
            s,
            "    int cb = g % {nb}, rb = (g / {nb}) % {mb}, lead = g / {};",
            nb * mb
        )
        .unwrap();
        for &d in t.others.iter().rev() {
            writeln!(
                s,
                "    int p{d} = lead % {}; lead = lead / {};",
                pts[d], pts[d]
            )
            .unwrap();
        }
        writeln!(s, "    int r0 = rb * {rm}, c0 = cb * {rn};").unwrap();
        writeln!(
            s,
            "    int base_a = {}, base_b = {};",
            lead_term(addr_a),
            lead_term(addr_b)
        )
        .unwrap();
        let (cpa, cpb) = (addr_a.coefs[dy], addr_b.coefs[dx]);
        // row/column offsets, clamped at the edge (the stores are guarded)
        for i in 0..rm {
            writeln!(
                s,
                "    __global const float* a{i} = b{} + base_a + {cpa} * min(r0 + {i}, {});",
                t.buf_a,
                m - 1
            )
            .unwrap();
        }
        for j in 0..rn {
            writeln!(
                s,
                "    __global const float* q{j} = b{} + base_b + {cpb} * min(c0 + {j}, {});",
                t.buf_b,
                n - 1
            )
            .unwrap();
        }
        for i in 0..rm {
            for j in 0..rn {
                writeln!(s, "    float acc{i}_{j} = 0.0f;").unwrap();
            }
        }
        writeln!(s, "    for (int k = 0; k < {k_total}; k += 4) {{").unwrap();
        for i in 0..rm {
            writeln!(s, "        float4 x{i} = vload4(0, a{i} + k);").unwrap();
        }
        for j in 0..rn {
            writeln!(s, "        float4 y{j} = vload4(0, q{j} + k);").unwrap();
        }
        for i in 0..rm {
            for j in 0..rn {
                writeln!(s, "        acc{i}_{j} += dot(x{i}, y{j});").unwrap();
            }
        }
        writeln!(s, "    }}").unwrap();
        // epilogue and stores
        self.mm_stores(
            &mut s,
            c,
            t.red_node,
            t.epilogue,
            pts,
            (dy, dx),
            (m, n),
            (rm, rn),
        );
        writeln!(s, "}}").unwrap();
        Some(super::cuda_gemv::Kernel {
            source: s,
            threads: wg,
            blocks: items.div_ceil(wg),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transliterates_loads_and_shuffles() {
        let src = format!(
            "// k\n#define TANG_TILES 4\n#define TANG_THREADS 32\n{PRELUDE}extern \"C\" __global__ void __launch_bounds__(32) main0(const float* __restrict__ b3, float* __restrict__ out, unsigned count) {{\n    float x = __ldg((const float4*)(b3 + (2)) + 1).x;\n    float4 v0[1][2];\n    float y = (&v0[0][e >> 2].x)[e & 3];\n    y = __shfl_xor_sync(0xffffffffu, y, 4);\n    out[0] = b3[0] + x + y;\n}}\n"
        );
        let m = transliterate(&src, &[], Some(3));
        assert!(m.starts_with(&format!("// k\n{SHFL_MARKER}")));
        assert!(m.contains(
            "reqd_work_group_size(32, 1, 1))) void main0(__global float* restrict out, uint count)"
        ));
        assert!(m.contains("(*((__global const float4*)(out + (2)) + 1)).x"));
        assert!(m.contains("TANG_EL4(v0[0][e >> 2], e & 3)"));
        assert!(m.contains("TANG_SHFL_XOR(y, 4)"));
        assert!(m.contains("out[0] = out[0]"));
        assert!(!m.contains("b3"));
    }
}
