//! Multi-dialect compute shader code generation.
//!
//! Generates compute kernels in WGSL, MSL (Metal Shading Language),
//! CUDA C, and plain C from the same expression graph. The expression
//! syntax is nearly identical across dialects — only kernel boilerplate
//! and a few edge cases (select, literal suffixes) differ.

use std::fmt::Write;

use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

/// Target shader dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dialect {
    /// WebGPU Shading Language.
    Wgsl,
    /// Metal Shading Language.
    Msl,
    /// CUDA C (compiled via NVRTC).
    Cuda,
    /// Plain C (for CPU fallback / verification).
    C,
}

/// A generated compute kernel.
pub struct ComputeKernel {
    /// Complete kernel source code.
    pub source: String,
    /// Number of input values per work item.
    pub n_inputs: usize,
    /// Number of output values per work item.
    pub n_outputs: usize,
    /// Workgroup / threadgroup / block size.
    pub workgroup_size: u32,
    /// Which dialect this kernel was generated for.
    pub dialect: Dialect,
    /// Entry point function name.
    pub entry_point: &'static str,
}

impl ExprGraph {
    /// Generate a plain-C kernel in f64: `void k0(const double* inputs,
    /// double* outputs, int count)`. Literals keep full precision.
    pub fn to_c_f64(&self, outputs: &[ExprId], n_inputs: usize) -> ComputeKernel {
        let live = self.live_set(outputs);
        let max_id = live.iter().max().copied().unwrap_or(0);
        let mut src = String::new();
        emit_c(
            &mut src,
            self,
            outputs,
            n_inputs,
            outputs.len(),
            &live,
            max_id,
            "double",
            "",
        );
        ComputeKernel {
            source: src,
            n_inputs,
            n_outputs: outputs.len(),
            workgroup_size: 1,
            dialect: Dialect::C,
            entry_point: "k0",
        }
    }

    /// Generate a compute kernel for the given dialect.
    ///
    /// Each work item reads `n_inputs` f32 values and writes `outputs.len()` f32 values.
    /// Shared subexpressions are computed once per thread.
    pub fn to_kernel(
        &self,
        outputs: &[ExprId],
        n_inputs: usize,
        dialect: Dialect,
    ) -> ComputeKernel {
        let workgroup_size = 256u32;
        let n_outputs = outputs.len();
        let live = self.live_set(outputs);
        let max_id = if live.is_empty() {
            0
        } else {
            *live.iter().max().unwrap()
        };

        let mut src = String::with_capacity(2048);

        match dialect {
            Dialect::Wgsl => emit_wgsl(
                &mut src,
                self,
                outputs,
                n_inputs,
                n_outputs,
                &live,
                max_id,
                workgroup_size,
            ),
            Dialect::Msl => emit_msl(&mut src, self, outputs, n_inputs, n_outputs, &live, max_id),
            Dialect::Cuda => emit_cuda(&mut src, self, outputs, n_inputs, n_outputs, &live, max_id),
            Dialect::C => emit_c(
                &mut src, self, outputs, n_inputs, n_outputs, &live, max_id, "float", "f",
            ),
        }

        ComputeKernel {
            source: src,
            n_inputs,
            n_outputs,
            workgroup_size,
            dialect,
            entry_point: "k0",
        }
    }
}

// ---------------------------------------------------------------------------
// Shared body emission
// ---------------------------------------------------------------------------

/// Emit SSA evaluation lines shared by all dialects.
///
/// `decl` is the variable declaration prefix: `"let"` for WGSL, `"float"` for MSL/CUDA/C.
/// `lit_suffix` is appended to float literals: `""` for WGSL, `"f"` for MSL/CUDA/C.
fn emit_body(
    src: &mut String,
    graph: &ExprGraph,
    outputs: &[ExprId],
    n_inputs: usize,
    n_outputs: usize,
    live: &std::collections::HashSet<usize>,
    max_id: usize,
    indent: &str,
    decl: &str,
    lit_suffix: &str,
    thread_id: &str,
    dialect: Dialect,
) {
    // Load inputs
    if n_inputs > 0 {
        let base = format!("{thread_id} * {n_inputs}u");
        let base_wgsl = format!("{thread_id} * {n_inputs}");
        for i in 0..n_inputs {
            match dialect {
                Dialect::Wgsl => {
                    writeln!(src, "{indent}{decl} x{i} = inputs[{base} + {i}u];").unwrap();
                }
                Dialect::Msl => {
                    writeln!(src, "{indent}{decl} x{i} = inputs[{base_wgsl} + {i}];").unwrap();
                }
                Dialect::Cuda => {
                    writeln!(src, "{indent}{decl} x{i} = in{i}[{thread_id}];").unwrap();
                }
                Dialect::C => {
                    writeln!(src, "{indent}{decl} x{i} = inputs[i * {n_inputs} + {i}];").unwrap();
                }
            }
        }
        writeln!(src).unwrap();
    }

    emit_ssa(src, graph, live, max_id, indent, decl, lit_suffix, dialect);

    // Store outputs
    if n_outputs > 0 {
        for (k, out) in outputs.iter().enumerate() {
            let val = ref_str(graph, *out, lit_suffix);
            match dialect {
                Dialect::Wgsl => {
                    let base = format!("{thread_id} * {n_outputs}u");
                    writeln!(src, "{indent}outputs[{base} + {k}u] = {val};").unwrap();
                }
                Dialect::Msl => {
                    let base = format!("{thread_id} * {n_outputs}");
                    writeln!(src, "{indent}outputs[{base} + {k}] = {val};").unwrap();
                }
                Dialect::Cuda => {
                    let base = format!("{thread_id} * {n_outputs}");
                    writeln!(src, "{indent}outputs[{base} + {k}] = {val};").unwrap();
                }
                Dialect::C => {
                    writeln!(src, "{indent}outputs[i * {n_outputs} + {k}] = {val};").unwrap();
                }
            }
        }
    }
}

/// The SSA evaluation lines: one `{decl} t{i} = ...;` per live non-leaf
/// node, in topological order. Inputs are the caller's `x{n}`.
#[allow(clippy::too_many_arguments)]
fn emit_ssa(
    src: &mut String,
    graph: &ExprGraph,
    live: &std::collections::HashSet<usize>,
    max_id: usize,
    indent: &str,
    decl: &str,
    lit_suffix: &str,
    dialect: Dialect,
) {
    for i in 0..=max_id {
        if !live.contains(&i) {
            continue;
        }
        let node = graph.node(ExprId(i as u32));
        match node {
            Node::Var(_) | Node::Lit(_) => continue,
            _ => {}
        }
        let rhs = expr_str(graph, node, lit_suffix, dialect);
        writeln!(src, "{indent}{decl} t{i} = {rhs};").unwrap();
    }
    writeln!(src).unwrap();
}

/// Generate expression string for a node.
pub(crate) fn expr_str(graph: &ExprGraph, node: Node, suffix: &str, dialect: Dialect) -> String {
    match node {
        Node::Var(n) => format!("x{n}"),
        Node::Lit(bits) => format_literal(f64::from_bits(bits), suffix),
        Node::Add(a, b) => format!(
            "({} + {})",
            ref_str(graph, a, suffix),
            ref_str(graph, b, suffix)
        ),
        Node::Mul(a, b) => format!(
            "({} * {})",
            ref_str(graph, a, suffix),
            ref_str(graph, b, suffix)
        ),
        // The space keeps `-` of a negative literal from fusing into `--`.
        Node::Neg(a) => format!("(- {})", ref_str(graph, a, suffix)),
        Node::Recip(a) => format!("(1.0{suffix} / {})", ref_str(graph, a, suffix)),
        Node::Sqrt(a) => format!("sqrt({})", ref_str(graph, a, suffix)),
        Node::Sin(a) => format!("sin({})", ref_str(graph, a, suffix)),
        Node::Atan2(y, x) => format!(
            "atan2({}, {})",
            ref_str(graph, y, suffix),
            ref_str(graph, x, suffix)
        ),
        Node::Exp2(a) => format!("exp2({})", ref_str(graph, a, suffix)),
        Node::Log2(a) => format!("log2({})", ref_str(graph, a, suffix)),
        Node::Select(c, a, b) => {
            match dialect {
                Dialect::Wgsl => {
                    // WGSL: select(false_val, true_val, cond)
                    format!(
                        "select({}, {}, {} > 0.0)",
                        ref_str(graph, b, suffix),
                        ref_str(graph, a, suffix),
                        ref_str(graph, c, suffix),
                    )
                }
                _ => {
                    // MSL/CUDA/C: ternary
                    format!(
                        "({} > 0.0{suffix} ? {} : {})",
                        ref_str(graph, c, suffix),
                        ref_str(graph, a, suffix),
                        ref_str(graph, b, suffix),
                    )
                }
            }
        }
    }
}

/// Reference a node inline: Var → x{n}, Lit → literal, others → t{index}.
pub(crate) fn ref_str(graph: &ExprGraph, id: ExprId, suffix: &str) -> String {
    match graph.node(id) {
        Node::Var(n) => format!("x{n}"),
        Node::Lit(bits) => format_literal(f64::from_bits(bits), suffix),
        _ => format!("t{}", id.0),
    }
}

/// Format f64 as a float literal with optional suffix.
fn format_literal(v: f64, suffix: &str) -> String {
    let base = if v == 0.0 {
        "0.0".to_string()
    } else if v == 1.0 {
        "1.0".to_string()
    } else if v == -1.0 {
        "-1.0".to_string()
    } else if v == 2.0 {
        "2.0".to_string()
    } else {
        let s = format!("{v}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    };
    format!("{base}{suffix}")
}

// ---------------------------------------------------------------------------
// WGSL dialect
// ---------------------------------------------------------------------------

fn emit_wgsl(
    src: &mut String,
    graph: &ExprGraph,
    outputs: &[ExprId],
    n_inputs: usize,
    n_outputs: usize,
    live: &std::collections::HashSet<usize>,
    max_id: usize,
    workgroup_size: u32,
) {
    writeln!(src, "// Auto-generated by tang-expr").unwrap();
    writeln!(src).unwrap();

    // Params struct
    writeln!(src, "struct Params {{").unwrap();
    writeln!(src, "    count: u32,").unwrap();
    writeln!(src, "    _pad1: u32,").unwrap();
    writeln!(src, "    _pad2: u32,").unwrap();
    writeln!(src, "    _pad3: u32,").unwrap();
    writeln!(src, "}}").unwrap();
    writeln!(src).unwrap();

    // Bindings
    writeln!(
        src,
        "@group(0) @binding(0) var<storage, read> inputs: array<f32>;"
    )
    .unwrap();
    writeln!(
        src,
        "@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;"
    )
    .unwrap();
    writeln!(src, "@group(0) @binding(2) var<uniform> params: Params;").unwrap();
    writeln!(src).unwrap();

    // Entry point
    writeln!(src, "@compute @workgroup_size({workgroup_size})").unwrap();
    writeln!(
        src,
        "fn k0(@builtin(global_invocation_id) gid: vec3<u32>) {{"
    )
    .unwrap();
    writeln!(src, "    let idx = gid.x;").unwrap();
    writeln!(src, "    if (idx >= params.count) {{ return; }}").unwrap();
    writeln!(src).unwrap();

    emit_body(
        src,
        graph,
        outputs,
        n_inputs,
        n_outputs,
        live,
        max_id,
        "    ",
        "let",
        "",
        "idx",
        Dialect::Wgsl,
    );

    writeln!(src, "}}").unwrap();
}

// ---------------------------------------------------------------------------
// MSL dialect
// ---------------------------------------------------------------------------

fn emit_msl(
    src: &mut String,
    graph: &ExprGraph,
    outputs: &[ExprId],
    n_inputs: usize,
    n_outputs: usize,
    live: &std::collections::HashSet<usize>,
    max_id: usize,
) {
    writeln!(src, "// Auto-generated by tang-expr").unwrap();
    writeln!(src, "#include <metal_stdlib>").unwrap();
    writeln!(src, "using namespace metal;").unwrap();
    writeln!(src).unwrap();

    write!(src, "kernel void k0(").unwrap();
    writeln!(src, "device const float* inputs [[buffer(0)]],").unwrap();
    writeln!(src, "    device float* outputs [[buffer(1)]],").unwrap();
    writeln!(src, "    device const uint& count [[buffer(2)]],").unwrap();
    writeln!(src, "    uint gid [[thread_position_in_grid]]) {{").unwrap();
    writeln!(src, "    if (gid >= count) {{ return; }}").unwrap();
    writeln!(src).unwrap();

    emit_body(
        src,
        graph,
        outputs,
        n_inputs,
        n_outputs,
        live,
        max_id,
        "    ",
        "float",
        "f",
        "gid",
        Dialect::Msl,
    );

    writeln!(src, "}}").unwrap();
}

// ---------------------------------------------------------------------------
// CUDA dialect
// ---------------------------------------------------------------------------

fn emit_cuda(
    src: &mut String,
    graph: &ExprGraph,
    outputs: &[ExprId],
    n_inputs: usize,
    n_outputs: usize,
    live: &std::collections::HashSet<usize>,
    max_id: usize,
) {
    writeln!(src, "// Auto-generated by tang-expr").unwrap();
    // NVRTC provides all math builtins (expf, sqrtf, fmaxf, etc.) — no #include needed.
    writeln!(src).unwrap();

    // Separate input pointers: in0, in1, ... instead of one interleaved buffer
    write!(src, "extern \"C\" __global__ void k0(").unwrap();
    for i in 0..n_inputs {
        writeln!(src, "const float* __restrict__ in{i},").unwrap();
        write!(src, "    ").unwrap();
    }
    writeln!(src, "float* __restrict__ outputs,").unwrap();
    writeln!(src, "    const unsigned int count) {{").unwrap();
    writeln!(
        src,
        "    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;"
    )
    .unwrap();
    writeln!(src, "    if (gid >= count) {{ return; }}").unwrap();
    writeln!(src).unwrap();

    emit_body(
        src,
        graph,
        outputs,
        n_inputs,
        n_outputs,
        live,
        max_id,
        "    ",
        "float",
        "f",
        "gid",
        Dialect::Cuda,
    );

    writeln!(src, "}}").unwrap();
}

/// Arithmetic of a generated row kernel. Buffers stay f32 either way;
/// `F64` widens every load and evaluates the graph in double (literals at
/// full precision), rounding only on the per-row stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Real {
    F32,
    F64,
}

/// f64 atomic add by compare-and-swap: valid on every arch NVRTC may
/// target (native f64 `atomicAdd` needs sm_60).
const CUDA_ATOMIC_ADD_F64: &str = "__device__ void tang_atomic_add_f64(double* a, double v) {
    unsigned long long* p = (unsigned long long*)a;
    unsigned long long old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(v + __longlong_as_double(assumed)));
    } while (assumed != old);
}
";

/// Where a row kernel's `Var(k)` comes from for row `r`:
/// `buf[(period > 0 ? r % period : r) * stride + offset]`, `buf` indexing
/// the kernel's input buffers. Compile-time constants, so a strided column
/// costs a multiply and a tiled vector a constant modulo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowLoad {
    pub buf: usize,
    pub stride: usize,
    pub offset: usize,
    pub period: usize,
}

impl RowLoad {
    /// Element `r` of `buf`.
    pub fn each(buf: usize) -> Self {
        Self {
            buf,
            stride: 1,
            offset: 0,
            period: 0,
        }
    }
    /// Column `col` of a row-major `[rows, width]` matrix.
    pub fn col(buf: usize, width: usize, col: usize) -> Self {
        Self {
            buf,
            stride: width,
            offset: col,
            period: 0,
        }
    }
    /// Element `i` of `buf`, the same for every row.
    pub fn bcast(buf: usize, i: usize) -> Self {
        Self {
            buf,
            stride: 0,
            offset: i,
            period: 0,
        }
    }
    /// `buf[r % period]`: a vector tiled along flattened rows (a bias over
    /// the columns of a matrix walked element by element).
    pub fn tile(buf: usize, period: usize) -> Self {
        Self {
            buf,
            stride: 1,
            offset: 0,
            period,
        }
    }
    /// The largest index row `rows - 1` (or any earlier row) reads.
    pub fn max_index(&self, rows: usize) -> usize {
        let last = rows.saturating_sub(1);
        let last = if self.period > 0 {
            last.min(self.period - 1)
        } else {
            last
        };
        last * self.stride + self.offset
    }
}

/// Where a row kernel stores a per-row output: `buf[r * stride + offset]`
/// in the kernel's output buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowStore {
    pub buf: usize,
    pub stride: usize,
    pub offset: usize,
}

impl RowStore {
    /// Element `r` of `buf`.
    pub fn each(buf: usize) -> Self {
        Self {
            buf,
            stride: 1,
            offset: 0,
        }
    }
    /// Column `col` of a row-major `[rows, width]` matrix.
    pub fn col(buf: usize, width: usize, col: usize) -> Self {
        Self {
            buf,
            stride: width,
            offset: col,
        }
    }
}

impl ExprGraph {
    /// A CUDA **row kernel**: one thread per row `r` (grid-stride); the
    /// graph's `Var(k)` loaded as `loads[k]` says, each `row_outputs[j]`
    /// stored as its [`RowStore`] says, and each `sum_outputs[j]` summed
    /// over all rows into `sums[j]` (double: warp shuffle, then one atomic
    /// per warp; `sums` must start zeroed). Buffers are f32; `real` is the
    /// arithmetic in between.
    ///
    /// Signature: `k0(long long rows, const float* in0, …, float* out0, …,
    /// double* sums)`, one pointer per distinct `buf` index (inputs and
    /// outputs must not alias). Launch with 32-multiple blocks.
    pub fn to_cuda_rows(
        &self,
        loads: &[RowLoad],
        row_outputs: &[(ExprId, RowStore)],
        sum_outputs: &[ExprId],
        real: Real,
    ) -> ComputeKernel {
        let all: Vec<ExprId> = row_outputs
            .iter()
            .map(|o| o.0)
            .chain(sum_outputs.iter().copied())
            .collect();
        let live = self.live_set(&all);
        let max_id = live.iter().max().copied().unwrap_or(0);
        let (ty, suffix) = match real {
            Real::F32 => ("float", "f"),
            Real::F64 => ("double", ""),
        };
        let n_in = loads.iter().map(|l| l.buf + 1).max().unwrap_or(0);
        let n_out = row_outputs.iter().map(|o| o.1.buf + 1).max().unwrap_or(0);
        let ns = sum_outputs.len();
        let mut src = String::with_capacity(4096);
        writeln!(src, "// Auto-generated by tang-expr (row kernel)").unwrap();
        if ns > 0 {
            src.push_str(CUDA_ATOMIC_ADD_F64);
        }
        write!(src, "extern \"C\" __global__ void k0(long long rows").unwrap();
        for b in 0..n_in {
            write!(src, ",\n    const float* __restrict__ in{b}").unwrap();
        }
        for b in 0..n_out {
            write!(src, ",\n    float* __restrict__ out{b}").unwrap();
        }
        writeln!(src, ",\n    double* __restrict__ sums) {{").unwrap();
        for j in 0..ns {
            writeln!(src, "    double acc{j} = 0.0;").unwrap();
        }
        writeln!(
            src,
            "    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (long long)gridDim.x * blockDim.x) {{"
        )
        .unwrap();
        for (k, l) in loads.iter().enumerate() {
            let row = if l.period > 0 {
                format!("(r % {}LL)", l.period)
            } else {
                "r".to_string()
            };
            writeln!(
                src,
                "        {ty} x{k} = ({ty})in{}[{row} * {}LL + {}LL];",
                l.buf, l.stride, l.offset
            )
            .unwrap();
        }
        emit_ssa(
            &mut src,
            self,
            &live,
            max_id,
            "        ",
            ty,
            suffix,
            Dialect::Cuda,
        );
        for (out, st) in row_outputs {
            let v = ref_str(self, *out, suffix);
            writeln!(
                src,
                "        out{}[r * {}LL + {}LL] = (float)({v});",
                st.buf, st.stride, st.offset
            )
            .unwrap();
        }
        for (j, out) in sum_outputs.iter().enumerate() {
            let v = ref_str(self, *out, suffix);
            writeln!(src, "        acc{j} += (double)({v});").unwrap();
        }
        writeln!(src, "    }}").unwrap();
        for j in 0..ns {
            writeln!(src, "    for (int o = 16; o > 0; o >>= 1) acc{j} += __shfl_down_sync(0xffffffff, acc{j}, o);").unwrap();
            writeln!(
                src,
                "    if ((threadIdx.x & 31) == 0) tang_atomic_add_f64(&sums[{j}], acc{j});"
            )
            .unwrap();
        }
        writeln!(src, "}}").unwrap();
        ComputeKernel {
            source: src,
            n_inputs: loads.len(),
            n_outputs: all.len(),
            workgroup_size: 256,
            dialect: Dialect::Cuda,
            entry_point: "k0",
        }
    }
}

// ---------------------------------------------------------------------------
// C dialect
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_c(
    src: &mut String,
    graph: &ExprGraph,
    outputs: &[ExprId],
    n_inputs: usize,
    n_outputs: usize,
    live: &std::collections::HashSet<usize>,
    max_id: usize,
    real: &str,
    suffix: &str,
) {
    writeln!(src, "// Auto-generated by tang-expr").unwrap();
    writeln!(src, "#include <math.h>").unwrap();
    writeln!(src).unwrap();

    write!(src, "void k0(").unwrap();
    writeln!(src, "const {real}* inputs,").unwrap();
    writeln!(src, "    {real}* outputs,").unwrap();
    writeln!(src, "    int count) {{").unwrap();
    writeln!(src, "    for (int i = 0; i < count; i++) {{").unwrap();

    emit_body(
        src,
        graph,
        outputs,
        n_inputs,
        n_outputs,
        live,
        max_id,
        "        ",
        real,
        suffix,
        "i",
        Dialect::C,
    );

    writeln!(src, "    }}").unwrap();
    writeln!(src, "}}").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ExprGraph;

    #[test]
    fn row_kernel_f64_shape() {
        let mut g = ExprGraph::new();
        let (x, y) = (g.var(0), g.var(1));
        let p = g.mul(x, y);
        let c = g.lit(0.1);
        let q = g.add(p, c);
        let loads = [RowLoad::col(0, 3, 2), RowLoad::tile(1, 5)];
        let k = g.to_cuda_rows(&loads, &[(p, RowStore::col(0, 2, 1))], &[q], Real::F64);
        let s = &k.source;
        assert!(s.contains("double x0 = (double)in0[r * 3LL + 2LL];"), "{s}");
        assert!(
            s.contains("double x1 = (double)in1[(r % 5LL) * 1LL + 0LL];"),
            "{s}"
        );
        assert!(s.contains("out0[r * 2LL + 1LL] = (float)("), "{s}");
        assert!(s.contains("tang_atomic_add_f64(&sums[0], acc0)"), "{s}");
        assert!(
            s.contains("0.1") && !s.contains("0.1f"),
            "f64 literals carry no suffix: {s}"
        );
        assert_eq!(k.n_outputs, 2);
        assert_eq!(RowLoad::tile(1, 5).max_index(100), 4);
        assert_eq!(RowLoad::col(0, 3, 2).max_index(10), 29);
        // no sums: no atomic helper
        let k = g.to_cuda_rows(&loads, &[(p, RowStore::each(0))], &[], Real::F32);
        assert!(!k.source.contains("tang_atomic_add_f64") && k.source.contains("float x0"));
    }

    #[test]
    fn negated_negative_literal_is_not_a_decrement() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let m = g.lit(-9.81);
        let n = g.neg(m);
        let y = g.mul(x, n);
        for d in [Dialect::C, Dialect::Cuda, Dialect::Msl, Dialect::Wgsl] {
            let src = g.to_kernel(&[y], 1, d).source;
            assert!(!src.contains("--"), "{d:?}:\n{src}");
        }
    }

    #[test]
    fn c_f64_keeps_full_precision() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let c = g.lit(0.1234567890123);
        let y = g.mul(x, c);
        let src = g.to_c_f64(&[y], 1).source;
        assert!(src.contains("const double* inputs"), "{src}");
        assert!(
            src.contains("0.1234567890123") && !src.contains("0.1234567890123f"),
            "{src}"
        );
        assert!(!src.contains("float"), "{src}");
    }

    #[test]
    fn wgsl_matches_original() {
        // Verify that to_kernel(Wgsl) produces output matching to_wgsl()
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let xx = g.mul(x, x);
        let yy = g.mul(y, y);
        let sum = g.add(xx, yy);
        let dist = g.sqrt(sum);

        let old = g.to_wgsl(&[dist], 2);
        let new = g.to_kernel(&[dist], 2, Dialect::Wgsl);

        assert_eq!(new.n_inputs, old.n_inputs);
        assert_eq!(new.n_outputs, old.n_outputs);
        assert_eq!(new.workgroup_size, old.workgroup_size);
        // Both should contain core elements
        assert!(new.source.contains("@compute"));
        assert!(new.source.contains("sqrt("));
    }

    #[test]
    fn msl_entry_point() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        let kernel = g.to_kernel(&[sum], 2, Dialect::Msl);
        assert!(kernel.source.contains("kernel void k0("));
        assert!(kernel.source.contains("thread_position_in_grid"));
        assert!(kernel.source.contains("#include <metal_stdlib>"));
        assert_eq!(kernel.entry_point, "k0");
    }

    #[test]
    fn cuda_entry_point() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let prod = g.mul(x, y);
        let kernel = g.to_kernel(&[prod], 2, Dialect::Cuda);
        assert!(kernel.source.contains("extern \"C\" __global__ void k0("));
        assert!(kernel
            .source
            .contains("blockIdx.x * blockDim.x + threadIdx.x"));
        // Separate input pointers instead of interleaved buffer
        assert!(kernel.source.contains("const float* __restrict__ in0,"));
        assert!(kernel.source.contains("const float* __restrict__ in1,"));
        assert!(!kernel.source.contains("inputs["));
        // Loads from separate pointers
        assert!(kernel.source.contains("in0[gid]"));
        assert!(kernel.source.contains("in1[gid]"));
    }

    #[test]
    fn c_loop() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let s = g.sin(x);
        let kernel = g.to_kernel(&[s], 1, Dialect::C);
        assert!(kernel.source.contains("for (int i = 0; i < count; i++)"));
        assert!(kernel.source.contains("sin("));
    }

    #[test]
    fn msl_select_ternary() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let a = g.lit(3.0);
        let b = g.lit(7.0);
        let s = g.select(x, a, b);
        let kernel = g.to_kernel(&[s], 1, Dialect::Msl);
        // MSL should use ternary, not select()
        assert!(kernel.source.contains("?"));
        assert!(!kernel.source.contains("select("));
    }

    #[test]
    fn wgsl_select_builtin() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let a = g.lit(3.0);
        let b = g.lit(7.0);
        let s = g.select(x, a, b);
        let kernel = g.to_kernel(&[s], 1, Dialect::Wgsl);
        assert!(kernel.source.contains("select("));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn msl_literal_suffix() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let c = g.lit(3.14);
        let prod = g.mul(x, c);
        let kernel = g.to_kernel(&[prod], 1, Dialect::Msl);
        assert!(kernel.source.contains("3.14f"));
    }

    #[test]
    fn multiple_outputs_all_dialects() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let y = g.var(1);
        let sum = g.add(x, y);
        let prod = g.mul(x, y);

        for dialect in [Dialect::Wgsl, Dialect::Msl, Dialect::Cuda, Dialect::C] {
            let kernel = g.to_kernel(&[sum, prod], 2, dialect);
            assert_eq!(kernel.n_outputs, 2);
            assert_eq!(kernel.n_inputs, 2);
        }
    }

    #[test]
    fn full_pipeline_all_dialects() {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let xx = g.mul(x, x);
        let dx = g.diff(xx, 0);
        let dx = g.simplify(dx);

        for dialect in [Dialect::Wgsl, Dialect::Msl, Dialect::Cuda, Dialect::C] {
            let kernel = g.to_kernel(&[xx, dx], 1, dialect);
            assert_eq!(kernel.n_outputs, 2);
            assert!(!kernel.source.is_empty());
        }
    }
}
