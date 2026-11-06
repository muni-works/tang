//! Lower a [`TensorGraph`] to WGSL compute kernels.
//!
//! One kernel per materialized comp. Inlined comps become WGSL functions
//! called from their consumers, so a fused graph (e.g. online-softmax
//! attention with `q·k` recomputed inside) lowers to a single dispatch.
//! Views become index arithmetic.
//!
//! Schedules honoured: `inline` (recompute vs buffer) and `split` (one
//! workgroup per point, threads stride the reduce space and tree-combine
//! through workgroup memory — valid for any monoid, since combine is
//! associative). No tiling or shared-memory staging yet.

use std::collections::HashSet;
use std::fmt::Write;

use tang_expr::{ExprGraph, ExprId, Node};

use super::{fixed, numel, Arg, Comp, Form, Format, Monoid, Op, TensorGraph, TensorId};

/// Target language of the shared emitters. The kernel structure and the
/// index algebra are the same for WGSL and MSL; only buffer element
/// access (`load`/`store`/`quant`) differs per dialect, and `tensor_msl`
/// rewrites the remaining WGSL syntax. Set for the duration of one
/// lowering by [`with_dialect`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Dialect {
    Wgsl,
    /// Native `half` buffers: one element per index, no packing.
    Msl,
    /// WGSL text for `wgsl_to_cuda`: 16-bit buffers are native
    /// (`array<f16>` / `array<bf16>` become `unsigned short*`), one
    /// element per index, converted with PTX `cvt` (`tang_h2f` …).
    Cuda,
}

thread_local! {
    static DIALECT: std::cell::Cell<Dialect> = const { std::cell::Cell::new(Dialect::Wgsl) };
}

pub(super) fn dialect() -> Dialect {
    DIALECT.with(|d| d.get())
}

/// Run `f` with the emitters producing dialect `d`.
pub(super) fn with_dialect<T>(d: Dialect, f: impl FnOnce() -> T) -> T {
    let prev = DIALECT.with(|c| c.replace(d));
    struct Reset(Dialect);
    impl Drop for Reset {
        fn drop(&mut self) {
            DIALECT.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(prev);
    f()
}

/// One dispatch.
/// Which shading language a [`WgslProgram`]'s kernels are emitted in.
/// `MslMma` lowers matmul-shaped f32 tiles to simdgroup matrices (see
/// `tensor_msl_mma.rs`) and emits WGSL for the rest; a backend tells the
/// two apart by the source (MSL starts with `#include`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Wgsl,
    MslMma,
}

pub struct WgslKernel {
    /// WGSL, or MSL when it starts with `#include` (entry point `main0`).
    pub source: String,
    /// Node this kernel materializes.
    pub node: TensorId,
    /// Nodes bound as read-only storage, in binding order. The output is
    /// bound after these, then a uniform `{points, 0, 0, 0}`.
    pub reads: Vec<TensorId>,
    /// Output points; goes in the uniform as `params.count`.
    pub points: u32,
    /// Threads to launch (`points`, padded or multiplied for workgroup schedules).
    pub count: u32,
    pub workgroup_size: u32,
}

/// A buffer the program needs.
pub struct BufferSpec {
    pub node: TensorId,
    /// Length in elements.
    pub len: usize,
    /// Element format of the buffer (the node's storage precision):
    /// `F32` is `array<f32>`, `F16` is `array<f16>` (two per 32-bit word,
    /// needs `wgpu::Features::SHADER_F16`).
    pub format: Format,
    /// `Some(k)` if this is graph input `k`.
    pub input: Option<usize>,
}

pub struct WgslProgram {
    /// Kernels in dependency order.
    pub kernels: Vec<WgslKernel>,
    /// Every node that needs a buffer: inputs and materialized comps.
    pub buffers: Vec<BufferSpec>,
    /// `(reshape node, node whose buffer holds its bytes)`.
    pub aliases: Vec<(TensorId, TensorId)>,
}

/// Identifier prefixes the emitters follow with a graph node number:
/// buffers `b{j}` and per-node functions `read_{j}`, `body_{j}`,
/// `point_{j}`, `combine_{j}`, and the CUDA transpiler's buffer
/// parameters `p_b{j}`.
const NODE_PREFIXES: [&str; 6] = ["b", "read_", "body_", "point_", "combine_", "p_b"];

/// `src` with the node numbers in node-named identifiers (see
/// `NODE_PREFIXES`) renumbered `0, 1, …` in order of first appearance.
/// The renaming is a bijection on numbers applied to every such
/// identifier, so the kernel means the same thing; two kernels that
/// differ only in which nodes they touch become the same string. Binding
/// order is unchanged (`reads`), so one compiled source serves every
/// launch. Shapes, constants and views are literals and stay as they are.
pub fn canonical_source(src: &str) -> String {
    let bytes = src.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut out = String::with_capacity(src.len());
    let mut numbers: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !ident(bytes[i]) {
            let start = i;
            while i < bytes.len() && !ident(bytes[i]) {
                i += 1;
            }
            out.push_str(&src[start..i]);
            continue;
        }
        let start = i;
        while i < bytes.len() && ident(bytes[i]) {
            i += 1;
        }
        let tok = &src[start..i];
        let digits = tok.len() - tok.bytes().rev().take_while(u8::is_ascii_digit).count();
        let (head, num) = tok.split_at(digits);
        if num.is_empty() || !NODE_PREFIXES.contains(&head) || bytes[start].is_ascii_digit() {
            out.push_str(tok);
            continue;
        }
        let k = numbers.iter().position(|n| *n == num).unwrap_or_else(|| {
            numbers.push(num);
            numbers.len() - 1
        });
        write!(out, "{head}{k}").unwrap();
    }
    out
}

impl WgslProgram {
    /// Every kernel source made [canonical](canonical_source).
    pub fn canonical(mut self) -> Self {
        for k in &mut self.kernels {
            k.source = canonical_source(&k.source);
        }
        self
    }

    /// The buffer-owning node for `t` (itself unless `t` is a reshape).
    pub fn storage(&self, t: TensorId) -> TensorId {
        self.aliases
            .iter()
            .find(|(r, _)| *r == t)
            .map_or(t, |(_, s)| *s)
    }
}

impl TensorGraph {
    pub fn to_wgsl(&self, outputs: &[TensorId]) -> WgslProgram {
        self.to_program(outputs, Target::Wgsl)
    }

    /// Like [`to_wgsl`](Self::to_wgsl) with matmul-shaped f32 tiles
    /// lowered to Metal simdgroup matrices (`tensor_msl_mma.rs`, run
    /// through wgpu's MSL passthrough); everything else stays WGSL. The
    /// whole-program native backend is [`to_msl`](Self::to_msl).
    pub fn to_msl_mma(&self, outputs: &[TensorId]) -> WgslProgram {
        self.to_program(outputs, Target::MslMma)
    }

    /// Lower to kernels. Sources are [canonical](canonical_source): the
    /// same kernel at different node numbers (say, in another layer) is
    /// the same string, so source-keyed caches compile it once.
    pub fn to_program(&self, outputs: &[TensorId], target: Target) -> WgslProgram {
        self.to_program_numbered(outputs, target).canonical()
    }

    /// [`to_program`](Self::to_program) without
    /// [`canonical_source`]: identifiers still carry graph node numbers,
    /// which backends that rewrite the WGSL by node (MSL) rely on.
    pub(super) fn to_program_numbered(&self, outputs: &[TensorId], target: Target) -> WgslProgram {
        let live = self.live(outputs);
        let n = self.nodes.len();
        let needs_buffer = |i: usize| {
            live[i]
                && !self.nodes[i].inline
                && !matches!(self.nodes[i].op, Op::Iota(_) | Op::Reshape(_))
        };
        // reshapes alias their storage node's buffer
        let aliases: Vec<(TensorId, TensorId)> = (0..n)
            .filter(|&i| live[i] && matches!(self.nodes[i].op, Op::Reshape(_)))
            .map(|i| (TensorId(i as u32), self.storage(TensorId(i as u32))))
            .collect();
        let buffers = (0..n)
            .filter(|&i| needs_buffer(i))
            .map(|i| {
                let (len, input) = match &self.nodes[i].op {
                    Op::Input(k) => (numel(&self.nodes[i].shape), Some(*k)),
                    Op::Comp(c) => (numel(c.points()) * c.width(), None),
                    Op::Iota(_) | Op::Reshape(_) => unreachable!(),
                };
                let format = self.nodes[i].precision.storage;
                assert!(
                    format != Format::BF16 || dialect() != Dialect::Wgsl,
                    "bf16 storage is interpreter-only: WGSL has no bf16 type"
                );
                BufferSpec {
                    node: TensorId(i as u32),
                    len,
                    input,
                    format,
                }
            })
            .collect();

        let kernels = (0..n)
            .filter(|&i| needs_buffer(i) && self.comp_at(i).is_some())
            .map(|i| self.kernel_for(i, target))
            .collect();

        WgslProgram {
            kernels,
            buffers,
            aliases,
        }
    }

    /// Nodes reachable from `i` through inlined comps, including `i`,
    /// in dependency order.
    pub(super) fn inline_closure(&self, i: usize) -> Vec<usize> {
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![(i, false)];
        while let Some((j, done)) = stack.pop() {
            if done {
                order.push(j);
                continue;
            }
            if !seen.insert(j) {
                continue;
            }
            stack.push((j, true));
            let descend = match &self.nodes[j].op {
                Op::Comp(_) => j == i || self.nodes[j].inline,
                Op::Reshape(_) => true,
                Op::Input(_) | Op::Iota(_) => false,
            };
            if descend {
                stack.extend(self.deps(j).into_iter().map(|d| (d, false)));
            }
        }
        order
    }

    /// Bindings plus the per-node functions (`read_j`, `point_j`, …) of
    /// `root`'s kernel, and the buffer-backed nodes it reads (binding order).
    pub(super) fn kernel_preamble(&self, root: usize) -> (String, Vec<usize>) {
        let order = self.inline_closure(root);
        let reads: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&j| {
                j != root
                    && !self.nodes[j].inline
                    && !matches!(self.nodes[j].op, Op::Iota(_) | Op::Reshape(_))
            })
            .collect();

        let mut s = String::new();
        writeln!(s, "// Auto-generated by tang-compile from a TensorGraph").unwrap();
        writeln!(
            s,
            "struct Params {{ count: u32, _p1: u32, _p2: u32, _p3: u32 }}"
        )
        .unwrap();
        for (b, &j) in reads.iter().enumerate() {
            writeln!(
                s,
                "@group(0) @binding({b}) var<storage, read> b{j}: array<{}>;",
                elem_ty(self.nodes[j].precision.storage)
            )
            .unwrap();
        }
        writeln!(
            s,
            "@group(0) @binding({}) var<storage, read_write> out: array<{}>;",
            reads.len(),
            out_ty(self.nodes[root].precision.storage)
        )
        .unwrap();
        writeln!(
            s,
            "@group(0) @binding({}) var<uniform> params: Params;\n",
            reads.len() + 1
        )
        .unwrap();

        for &j in &order {
            self.emit_node(&mut s, j, &reads);
        }
        (s, reads)
    }

    fn kernel_for(&self, root: usize, target: Target) -> WgslKernel {
        let c = self.comp_at(root).unwrap();
        let (mut s, reads) = self.kernel_preamble(root);
        if target == Target::MslMma && self.nodes[root].tile.0 > 0 {
            if let Some(k) = self.msl_mma_kernel(root, &reads) {
                return k;
            }
        }

        let w = c.width();
        let k = c.points().len();
        let points = numel(c.points()) as u32;
        let split = self.nodes[root].split;
        let out_f = self.nodes[root].precision.storage;
        if self.nodes[root].tile.0 > 0 {
            if let Some(k) = self.tile_kernel(root, s.clone(), reads.clone()) {
                return k;
            }
        }
        let reads = reads.into_iter().map(|j| TensorId(j as u32)).collect();

        if split == 0 {
            let args = (0..k)
                .map(|d| format!("p{d}"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(s, "@compute @workgroup_size(256)").unwrap();
            writeln!(
                s,
                "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{"
            )
            .unwrap();
            writeln!(s, "    let g = gid.x + gid.y * {}u;", 65535 * 256).unwrap();
            writeln!(s, "    if (g >= params.count) {{ return; }}").unwrap();
            emit_decompose(&mut s, "g", &fixed(c.points()), "p");
            writeln!(s, "    let v = point_{root}({args});").unwrap();
            for slot in 0..w {
                let st = store(out_f, &format!("g * {w}u + {slot}u"), &format!("v[{slot}]"));
                writeln!(s, "    {st}").unwrap();
            }
            writeln!(s, "}}").unwrap();
            return WgslKernel {
                source: s,
                node: TensorId(root as u32),
                reads,
                points,
                count: points,
                workgroup_size: 256,
            };
        }

        // One workgroup per point; threads stride the flattened reduce
        // space, then tree-combine through workgroup memory. `root` may be
        // the epilogue map of the reduction it chunks (see
        // `tiled_reduction`); the map's body then runs on thread 0 after
        // the tree with the reduction's var bound to the result.
        let red_node = self.tiled_reduction(root).expect("split needs a reduction");
        let rc = self.comp_at(red_node).unwrap();
        let m = rc.monoid.as_ref().expect("split needs a reduction");
        let (w, w_out) = (rc.width(), c.width());
        let r_total = numel(&rc.red);
        writeln!(s, "var<workgroup> wg_acc: array<array<f32, {w}>, {split}>;").unwrap();
        writeln!(s, "@compute @workgroup_size({split})").unwrap();
        writeln!(
            s,
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{"
        )
        .unwrap();
        writeln!(s, "    let g = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    let tid = lid.x;").unwrap();
        writeln!(s, "    if (g >= params.count) {{ return; }}").unwrap();
        emit_decompose(&mut s, "g", &fixed(c.points()), "p");
        // the reduction's point coords as the epilogue reads them
        if red_node != root {
            let forms = self.epilogue_read(c, red_node).unwrap();
            for (d, f) in forms.iter().enumerate() {
                writeln!(s, "    let q{d} = {};", affine(f, "p")).unwrap();
            }
        }
        emit_scalar_acc(&mut s, "    ", m);
        writeln!(
            s,
            "    for (var ri: u32 = tid; ri < {r_total}u; ri = ri + {split}u) {{"
        )
        .unwrap();
        emit_decompose(&mut s, "ri", &fixed(&rc.red), "r");
        let rp = if red_node != root { "q" } else { "p" };
        emit_step(
            &mut s,
            "        ",
            &format!("body_{red_node}({})", arg_reads_with(self, rc, rp, None)),
            m,
            self.nodes[red_node].precision.accum,
        );
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    wg_acc[tid] = {};", acc_array(w)).unwrap();
        writeln!(s, "    workgroupBarrier();").unwrap();
        writeln!(
            s,
            "    for (var stride: u32 = {split}u / 2u; stride > 0u; stride = stride / 2u) {{"
        )
        .unwrap();
        writeln!(s, "        if (tid < stride) {{").unwrap();
        writeln!(
            s,
            "            wg_acc[tid] = combine_{red_node}(wg_acc[tid], wg_acc[tid + stride]);"
        )
        .unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        workgroupBarrier();").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    if (tid == 0u) {{").unwrap();
        if red_node == root {
            for slot in 0..w {
                let st = store(
                    out_f,
                    &format!("g * {w}u + {slot}u"),
                    &format!("wg_acc[0][{slot}]"),
                );
                writeln!(s, "        {st}").unwrap();
            }
        } else {
            let reads_epi = arg_reads_with(
                self,
                c,
                "p",
                Some(&|a: &Arg| {
                    (a.src.n() == red_node).then(|| format!("wg_acc[0][{}]", self.epilogue_slot(a)))
                }),
            );
            writeln!(s, "        let epi = body_{root}({reads_epi});").unwrap();
            for slot in 0..w_out {
                let st = store(
                    out_f,
                    &format!("g * {w_out}u + {slot}u"),
                    &format!("epi[{slot}]"),
                );
                writeln!(s, "        {st}").unwrap();
            }
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        WgslKernel {
            source: s,
            node: TensorId(root as u32),
            reads,
            points,
            count: points * split,
            workgroup_size: split,
        }
    }

    /// `read_j` for every node; `body_j`/`combine_j`/`point_j` for comps
    /// that are computed inside this kernel.
    fn emit_node(&self, s: &mut String, j: usize, reads: &[usize]) {
        let node = &self.nodes[j];
        let r = node.shape.len();
        let coords = (0..r)
            .map(|d| format!("c{d}: u32"))
            .collect::<Vec<_>>()
            .join(", ");
        match &node.op {
            Op::Input(_) => {
                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                let v = load(
                    node.precision.storage,
                    &format!("b{j}"),
                    &flat("c", 0, &fixed(&node.shape)),
                );
                writeln!(s, "    return {v};").unwrap();
                writeln!(s, "}}\n").unwrap();
            }
            Op::Iota(d) => {
                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                writeln!(s, "    return f32(c{d});").unwrap();
                writeln!(s, "}}\n").unwrap();
            }
            Op::Reshape(src) => {
                // same bytes: flat index here is the flat index there
                let src_shape = fixed(&self.nodes[src.n()].shape);
                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                writeln!(s, "    var p = {};", flat("c", 0, &fixed(&node.shape))).unwrap();
                let mut args = Vec::new();
                for d in (0..src_shape.len()).rev() {
                    writeln!(s, "    let s{d} = p % {}u;", src_shape[d]).unwrap();
                    writeln!(s, "    p = p / {}u;", src_shape[d]).unwrap();
                    args.push(format!("s{d}"));
                }
                args.reverse();
                writeln!(s, "    return read_{}({});", src.n(), args.join(", ")).unwrap();
                writeln!(s, "}}\n").unwrap();
            }
            Op::Comp(c) => {
                let w = c.width();
                let k = c.points().len();
                let materialized = reads.contains(&j);
                if !materialized {
                    emit_scalar_fn(s, &format!("body_{j}"), &c.body, &c.body_outs, c.args.len());
                    if let Some(m) = &c.monoid {
                        emit_combine(s, j, m, node.precision.accum);
                    }
                    self.emit_point(s, j, c);
                }

                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                let slot = flat("c", k, &fixed(&c.out[k..]));
                if materialized {
                    // buffer layout: [points][width]
                    let pt = flat("c", 0, &fixed(c.points()));
                    let v = load(
                        node.precision.storage,
                        &format!("b{j}"),
                        &format!("({pt}) * {w}u + ({slot})"),
                    );
                    writeln!(s, "    return {v};").unwrap();
                } else {
                    let args = (0..k)
                        .map(|d| format!("c{d}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(s, "    let v = point_{j}({args});").unwrap();
                    writeln!(s, "    return v[{slot}];").unwrap();
                }
                writeln!(s, "}}\n").unwrap();
            }
        }
    }

    /// `point_j(p…) -> array<f32, W>`: the comp at one output point.
    fn emit_point(&self, s: &mut String, j: usize, c: &Comp) {
        let w = c.width();
        let k = c.points().len();
        let params = (0..k)
            .map(|d| format!("p{d}: u32"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(s, "fn point_{j}({params}) -> array<f32, {w}> {{").unwrap();
        let reads = arg_reads(self, c);
        let Some(m) = &c.monoid else {
            writeln!(s, "    return body_{j}({reads});").unwrap();
            writeln!(s, "}}\n").unwrap();
            return;
        };
        emit_scalar_acc(s, "    ", m);
        let mut indent = String::from("    ");
        for (i, n) in fixed(&c.red).into_iter().enumerate() {
            writeln!(
                s,
                "{indent}for (var r{i}: u32 = 0u; r{i} < {n}u; r{i} = r{i} + 1u) {{"
            )
            .unwrap();
            indent.push_str("    ");
        }
        emit_step(
            s,
            &indent,
            &format!("body_{j}({reads})"),
            m,
            self.nodes[j].precision.accum,
        );
        for _ in &c.red {
            indent.truncate(indent.len() - 4);
            writeln!(s, "{indent}}}").unwrap();
        }
        writeln!(s, "    return {};", acc_array(w)).unwrap();
        writeln!(s, "}}\n").unwrap();
    }
}

/// `var acc0: f32 = id0; …` — accumulator slots as scalars, so the
/// combine runs in registers instead of copying `array<f32, W>` values.
fn emit_scalar_acc(s: &mut String, indent: &str, m: &Monoid) {
    for (i, v) in m.identity.iter().enumerate() {
        writeln!(s, "{indent}var acc{i}: f32 = {};", lit(*v)).unwrap();
    }
}

fn acc_array(w: usize) -> String {
    let slots = (0..w)
        .map(|i| format!("acc{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("array<f32, {w}>({slots})")
}

/// One reduction step: evaluate `elem` (an `array<f32, W>` expression)
/// and fold it into the scalar accumulators with the combine inlined.
fn emit_step(s: &mut String, indent: &str, elem: &str, m: &Monoid, accum: Format) {
    let w = m.width;
    writeln!(s, "{indent}let el = {elem};").unwrap();
    let outs = emit_scalar_mapped(s, indent, &m.combine, &m.outs, "k", &|v| {
        if (v as usize) < w {
            format!("acc{v}")
        } else {
            format!("el[{}]", v as usize - w)
        }
    });
    for (i, o) in outs.iter().enumerate() {
        writeln!(s, "{indent}let nx{i} = {o};").unwrap();
    }
    for i in 0..w {
        writeln!(s, "{indent}acc{i} = {};", quant(accum, &format!("nx{i}"))).unwrap();
    }
}

/// Emit `let {prefix}{i} = …;` for the live nodes of `g` and return the
/// expressions for `outs`; `var_expr` names input var `i`.
pub(super) fn emit_scalar_mapped(
    s: &mut String,
    indent: &str,
    g: &ExprGraph,
    outs: &[ExprId],
    prefix: &str,
    var_expr: &dyn Fn(u16) -> String,
) -> Vec<String> {
    let mut live: Vec<usize> = g.live_set(outs).into_iter().collect();
    live.sort_unstable();
    let nodes = g.nodes_slice();
    let e = |id: &ExprId| match nodes[id.index() as usize] {
        Node::Var(v) => var_expr(v),
        _ => format!("{prefix}{}", id.index()),
    };
    for i in live {
        let rhs = match &nodes[i] {
            Node::Var(_) => continue,
            Node::Lit(b) => lit(f64::from_bits(*b)),
            Node::Add(a, b) => format!("{} + {}", e(a), e(b)),
            Node::Mul(a, b) => format!("{} * {}", e(a), e(b)),
            Node::Neg(a) => format!("-{}", e(a)),
            Node::Recip(a) => format!("1.0 / {}", e(a)),
            Node::Sqrt(a) => format!("sqrt({})", e(a)),
            Node::Sin(a) => format!("sin({})", e(a)),
            Node::Atan2(a, b) => format!("atan2({}, {})", e(a), e(b)),
            Node::Exp2(a) => format!("exp2({})", e(a)),
            Node::Log2(a) => format!("log2({})", e(a)),
            Node::Select(c, a, b) => format!("select({}, {}, {} > 0.0)", e(b), e(a), e(c)),
        };
        writeln!(s, "{indent}let {prefix}{i} = {rhs};").unwrap();
    }
    outs.iter().map(e).collect()
}

pub(super) fn identity(m: &Monoid) -> String {
    m.identity
        .iter()
        .map(|v| lit(*v))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `read_src(view(p…, r…))` for every arg, as call arguments.
fn arg_reads(g: &TensorGraph, c: &Comp) -> String {
    arg_reads_with(g, c, "p", None)
}

/// `u32(offset + Σ coef·{prefix}d)`.
fn affine(f: &Form, prefix: &str) -> String {
    let mut terms = vec![format!("{}", f.offset)];
    for (d, &coef) in f.coefs.iter().enumerate() {
        if coef != 0 {
            terms.push(format!("{coef} * i32({prefix}{d})"));
        }
    }
    format!("u32({})", terms.join(" + "))
}

/// `arg_reads` with point coords named `{pts}d`; `replace` may
/// substitute an expression for an arg's read.
type ReadReplace<'a> = Option<&'a dyn Fn(&Arg) -> Option<String>>;

fn arg_reads_with(g: &TensorGraph, c: &Comp, pts: &str, replace: ReadReplace) -> String {
    c.args
        .iter()
        .map(|a| match replace.and_then(|r| r(a)) {
            Some(e) => e,
            None => arg_read_with(g, c, a, pts),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Value of one arg of `c` at point coords `p…` and reduce coords `r…`
/// (in scope as variables), zero-padded if the view is guarded.
pub(super) fn arg_read(g: &TensorGraph, c: &Comp, a: &super::Arg) -> String {
    arg_read_with(g, c, a, "p")
}

fn arg_read_with(g: &TensorGraph, c: &Comp, a: &super::Arg, pts: &str) -> String {
    let k = c.points().len();
    let coord_name = |d: usize| {
        if d < k {
            format!("{pts}{d}")
        } else {
            format!("r{}", d - k)
        }
    };
    let signed: Vec<String> = a
        .view
        .forms
        .iter()
        .map(|f| {
            let mut terms = vec![format!("{}", f.offset)];
            for (d, &coef) in f.coefs.iter().enumerate() {
                if coef != 0 {
                    terms.push(format!("{coef} * i32({})", coord_name(d)));
                }
            }
            format!("({})", terms.join(" + "))
        })
        .collect();
    // a gather adds `coef · index`, the index arg's value rounded to an
    // integer (index tensors are f32; exact below 2^24)
    let signed = match &a.view.gather {
        None => signed,
        Some(gt) => {
            let index = arg_read_with(g, c, &c.args[gt.arg], pts);
            let iv = format!("i32(round({index}))");
            signed
                .into_iter()
                .zip(&gt.coefs)
                .map(|(e, &coef)| match coef {
                    0 => e,
                    1 => format!("({e} + {iv})"),
                    _ => format!("({e} + {coef} * {iv})"),
                })
                .collect()
        }
    };
    let idx: Vec<String> = signed.iter().map(|e| format!("u32({e})")).collect();
    let read = format!("read_{}({})", a.src.n(), idx.join(", "));
    if !a.view.guard {
        return read;
    }
    // zero padding: the read itself may wrap (robust buffer
    // access keeps it in bounds); the select discards it
    let shape = fixed(&g.nodes[a.src.n()].shape);
    let cond = signed
        .iter()
        .zip(&shape)
        .map(|(e, n)| format!("(u32({e}) < {n}u)"))
        .collect::<Vec<_>>()
        .join(" && ");
    format!("select(0.0, {read}, {cond})")
}

/// Row-major flat index of coords `prefix{start}..` over `shape`.
fn flat(prefix: &str, start: usize, shape: &[usize]) -> String {
    if shape.is_empty() {
        return "0u".into();
    }
    let mut acc = format!("{prefix}{start}");
    for (i, n) in shape.iter().enumerate().skip(1) {
        acc = format!("({acc}) * {n}u + {prefix}{}", start + i);
    }
    acc
}

/// `g` → `p0..pk` over `shape`, row-major.
pub(super) fn emit_decompose(s: &mut String, g: &str, shape: &[usize], prefix: &str) {
    let mut rem = g.to_string();
    for d in (0..shape.len()).rev() {
        let n = shape[d];
        writeln!(s, "    let {prefix}{d} = ({rem}) % {n}u;").unwrap();
        rem = format!("({rem}) / {n}u");
    }
}

pub(super) fn lit(v: f64) -> String {
    // ±f32::MAX (the max monoid's identity stands in for ±∞) as an exact hex float. The
    // shortest decimal, 3.4028235e38, is above f32::MAX before rounding: naga accepts it,
    // but Tint (Chrome's WebGPU) rejects it as unrepresentable.
    if v == f64::NEG_INFINITY || (v as f32) == f32::MIN {
        return "-0x1.fffffep+127f".into();
    }
    if v == f64::INFINITY || (v as f32) == f32::MAX {
        return "0x1.fffffep+127f".into();
    }
    let s = format!("{:?}", v as f32);
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// `fn name(v0: f32, …) -> array<f32, W>` evaluating `outs` of `g`.
fn emit_scalar_fn(s: &mut String, name: &str, g: &ExprGraph, outs: &[ExprId], n_vars: usize) {
    let params = (0..n_vars)
        .map(|i| format!("v{i}: f32"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(s, "fn {name}({params}) -> array<f32, {}> {{", outs.len()).unwrap();
    emit_scalar_body(s, g, outs, Format::F32);
    writeln!(s, "}}\n").unwrap();
}

fn emit_combine(s: &mut String, j: usize, m: &Monoid, accum: Format) {
    let w = m.width;
    writeln!(
        s,
        "fn combine_{j}(a: array<f32, {w}>, e: array<f32, {w}>) -> array<f32, {w}> {{"
    )
    .unwrap();
    for i in 0..w {
        writeln!(s, "    let v{i} = a[{i}];").unwrap();
        writeln!(s, "    let v{} = e[{i}];", w + i).unwrap();
    }
    emit_scalar_body(s, &m.combine, &m.outs, accum);
    writeln!(s, "}}\n").unwrap();
}

fn emit_scalar_body(s: &mut String, g: &ExprGraph, outs: &[ExprId], round: Format) {
    let mut live: Vec<usize> = g.live_set(outs).into_iter().collect();
    live.sort_unstable();
    let nodes = g.nodes_slice();
    let e = |id: &ExprId| match nodes[id.index() as usize] {
        Node::Var(v) => format!("v{v}"),
        _ => format!("e{}", id.index()),
    };
    for i in live {
        let rhs = match &nodes[i] {
            Node::Var(_) => continue,
            Node::Lit(b) => lit(f64::from_bits(*b)),
            Node::Add(a, b) => format!("{} + {}", e(a), e(b)),
            Node::Mul(a, b) => format!("{} * {}", e(a), e(b)),
            Node::Neg(a) => format!("-{}", e(a)),
            Node::Recip(a) => format!("1.0 / {}", e(a)),
            Node::Sqrt(a) => format!("sqrt({})", e(a)),
            Node::Sin(a) => format!("sin({})", e(a)),
            Node::Atan2(a, b) => format!("atan2({}, {})", e(a), e(b)),
            Node::Exp2(a) => format!("exp2({})", e(a)),
            Node::Log2(a) => format!("log2({})", e(a)),
            Node::Select(c, a, b) => format!("select({}, {}, {} > 0.0)", e(b), e(a), e(c)),
        };
        writeln!(s, "    let e{i} = {rhs};").unwrap();
    }
    let vals = outs
        .iter()
        .map(|o| quant(round, &e(o)))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(s, "    return array<f32, {}>({vals});", outs.len()).unwrap();
}

/// WGSL element type of a read-only buffer in format `f`. f16 is packed
/// two per `u32` (naga 24 has no `enable f16`; `unpack2x16float` is core
/// WGSL), so it needs no device feature.
pub(super) fn elem_ty(f: Format) -> &'static str {
    if dialect() == Dialect::Msl {
        // header lines are rewritten by the MSL lowering; any name will do
        return "f32";
    }
    if dialect() == Dialect::Cuda {
        return match f {
            Format::F32 => "f32",
            Format::F16 => "f16",
            Format::BF16 => "bf16",
            Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
        };
    }
    match f {
        Format::F32 => "f32",
        Format::F16 => "u32",
        Format::BF16 => panic!("bf16 storage is interpreter-only: WGSL has no bf16 type"),
        Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
    }
}

/// Element type of the output buffer: packed f16 halves of one word are
/// written by different threads, so the word is atomic.
fn out_ty(f: Format) -> &'static str {
    match f {
        Format::F16 if dialect() == Dialect::Wgsl => "atomic<u32>",
        f => elem_ty(f),
    }
}

/// Element `idx` of buffer `buf` (format `f`), as the f32 the kernel
/// computes in.
pub(super) fn load(f: Format, buf: &str, idx: &str) -> String {
    if dialect() == Dialect::Msl {
        return match f {
            Format::F32 => format!("{buf}[{idx}]"),
            _ => format!("f32({buf}[{idx}])"),
        };
    }
    if dialect() == Dialect::Cuda {
        return match f {
            Format::F32 => format!("{buf}[{idx}]"),
            Format::F16 => format!("tang_h2f({buf}[{idx}])"),
            Format::BF16 => format!("tang_b2f({buf}[{idx}])"),
            Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
        };
    }
    match f {
        Format::F32 => format!("{buf}[{idx}]"),
        Format::F16 => format!("unpack2x16float({buf}[({idx}) >> 1u])[({idx}) & 1u]"),
        Format::BF16 => panic!("bf16 storage is interpreter-only: WGSL has no bf16 type"),
        Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
    }
}

/// Statement writing f32 value `v` to `out[idx]` in format `f`. An f16
/// half is merged into its word with two atomics on disjoint bits, so
/// the neighbour's half is never clobbered whatever the interleaving.
pub(super) fn store(f: Format, idx: &str, v: &str) -> String {
    if dialect() == Dialect::Msl {
        return match f {
            Format::F32 => format!("out[{idx}] = {v};"),
            Format::F16 => format!("out[{idx}] = half({v});"),
            Format::BF16 => format!("out[{idx}] = bfloat({v});"),
            Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
        };
    }
    if dialect() == Dialect::Cuda {
        return match f {
            Format::F32 => format!("out[{idx}] = {v};"),
            Format::F16 => format!("out[{idx}] = tang_f2h({v});"),
            Format::BF16 => format!("out[{idx}] = tang_f2b({v});"),
            Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
        };
    }
    match f {
        Format::F32 => format!("out[{idx}] = {v};"),
        Format::F16 => format!(
            "{{ let i_ = {idx}; let sh_ = (i_ & 1u) * 16u; \
             atomicAnd(&out[i_ >> 1u], ~(0xffffu << sh_)); \
             atomicOr(&out[i_ >> 1u], (pack2x16float(vec2<f32>({v}, 0.0)) & 0xffffu) << sh_); }}"
        ),
        Format::BF16 => panic!("bf16 storage is interpreter-only: WGSL has no bf16 type"),
        Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
    }
}

/// An f32 accumulator value `e` rounded to format `f` (kept in an f32
/// register): f16 accumulation as rounding after every combine, which
/// is what the interpreter emulates. It reproduces the numerics of an
/// f16 accumulator, not its speed.
pub(super) fn quant(f: Format, e: &str) -> String {
    if dialect() == Dialect::Msl {
        return match f {
            Format::F32 => e.to_string(),
            Format::F16 => format!("f32(half({e}))"),
            Format::BF16 => format!("f32(bfloat({e}))"),
            Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
        };
    }
    if dialect() == Dialect::Cuda {
        return match f {
            Format::F32 => e.to_string(),
            Format::F16 => format!("tang_h2f(tang_f2h({e}))"),
            Format::BF16 => format!("tang_b2f(tang_f2b({e}))"),
            Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
        };
    }
    match f {
        Format::F32 => e.to_string(),
        Format::F16 => format!("unpack2x16float(pack2x16float(vec2<f32>({e}, 0.0))).x"),
        Format::BF16 => panic!("bf16 accumulation is interpreter-only"),
        Format::Q4K | Format::Q5K | Format::Q8x32 => unreachable!("block formats lower only through the int8 MMA kernel"),
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::canonical_source;
    use crate::tensor::TensorGraph;

    #[test]
    fn renames_node_numbers_only() {
        let a = "var<storage> b12: array<vec4<f32>>; fn read_12() { point_7(b12[3u]); body_7(); combine_7(); }";
        let b = "var<storage> b40: array<vec4<f32>>; fn read_40() { point_9(b40[3u]); body_9(); combine_9(); }";
        assert_eq!(canonical_source(a), canonical_source(b));
        assert_eq!(
            canonical_source(a),
            "var<storage> b0: array<vec4<f32>>; fn read_0() { point_1(b0[3u]); body_1(); combine_1(); }"
        );
        // a different literal or wiring stays different
        assert_ne!(
            canonical_source(a),
            canonical_source(&a.replace("3u", "4u"))
        );
        assert_ne!(
            canonical_source(a),
            canonical_source(&a.replace("point_7(b12", "point_12(b12"))
        );
        // builtins and locals with digits are untouched
        let c = "let acc3 = vec4<f32>(); half2 x; float4 y; __shfl_xor_sync(1); m16n8k8";
        assert_eq!(canonical_source(c), c);
    }

    /// `x · W_l` for `layers` square weights: the same kernel per layer.
    fn chain(layers: usize, n: usize) -> (TensorGraph, Vec<crate::tensor::TensorId>) {
        let mut g = TensorGraph::new();
        let mut h = g.input(&[8, n]);
        for _ in 0..layers {
            let w = g.input(&[n, n]);
            h = g.matmul(h, w).unwrap();
        }
        (g, vec![h])
    }

    #[test]
    fn same_kernel_at_other_nodes_is_one_source() {
        let sources = |layers, n| {
            let (g, y) = chain(layers, n);
            let p = g.to_wgsl(&y);
            let mut s: Vec<String> = p.kernels.into_iter().map(|k| k.source).collect();
            s.sort();
            s.dedup();
            s
        };
        let (s3, s9) = (sources(3, 16), sources(9, 16));
        assert_eq!(s3, s9);
        // other shapes are other kernels
        let s9b = sources(9, 32);
        assert!(s9b.iter().all(|s| !s9.contains(s)));
    }
}
