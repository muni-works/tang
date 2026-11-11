//! One graph as several kernels, for graphs too large for one shader
//! (a traced contact step's adjoint is hundreds of thousands of nodes, which
//! shader compilers take minutes and tens of GB to optimize, or reject).

use std::collections::HashMap;
use std::fmt::Write;

use crate::codegen::{expr_str, ref_str};
use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

/// WGSL kernels that together compute a graph's outputs, run in order over
/// the same environments: each reads `inputs` (`n_inputs` per environment)
/// and a per-environment `scratch` (`n_scratch` floats), and the last writes
/// `outputs`. Bindings: 0 `inputs` (read), 1 `scratch` (read_write),
/// 2 `outputs` (read_write), 3 the uniform `count`; entry point `main`.
pub struct SplitKernels {
    /// Kernel sources, in launch order.
    pub parts: Vec<String>,
    /// Scratch floats per environment.
    pub n_scratch: usize,
    /// Workgroup size of every part.
    pub workgroup_size: u32,
}

fn children(node: Node) -> [Option<ExprId>; 3] {
    match node {
        Node::Var(_) | Node::Lit(_) => [None; 3],
        Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => [Some(a), Some(b), None],
        Node::Neg(a)
        | Node::Recip(a)
        | Node::Sqrt(a)
        | Node::Sin(a)
        | Node::Floor(a)
        | Node::Exp2(a)
        | Node::Log2(a) => [Some(a), None, None],
        Node::Select(c, a, b) => [Some(c), Some(a), Some(b)],
    }
}

impl ExprGraph {
    /// [`to_kernel`](Self::to_kernel) for WGSL, cut into kernels of at most
    /// `max_nodes` evaluated nodes each (topological order); a value used
    /// outside the kernel that computes it goes through `scratch`.
    pub fn to_wgsl_split(
        &self,
        outputs: &[ExprId],
        n_inputs: usize,
        max_nodes: usize,
    ) -> SplitKernels {
        self.split_wgsl(outputs, n_inputs, max_nodes, false)
    }

    /// [`to_wgsl_split`](Self::to_wgsl_split) in `f64` (buffers, scratch and
    /// arithmetic), for a device with wgpu's `SHADER_F64`. Vulkan has no f64
    /// `sin`/`exp2`/`log2`/`atan2`, so those come from WGSL helpers accurate
    /// to a few ulp. A buffer of `n` values takes `2n` f32 slots.
    pub fn to_wgsl_split_f64(
        &self,
        outputs: &[ExprId],
        n_inputs: usize,
        max_nodes: usize,
    ) -> SplitKernels {
        self.split_wgsl(outputs, n_inputs, max_nodes, true)
    }

    fn split_wgsl(
        &self,
        outputs: &[ExprId],
        n_inputs: usize,
        max_nodes: usize,
        f64: bool,
    ) -> SplitKernels {
        let wg = 256u32;
        let (ty, suffix) = if f64 { ("f64", "lf") } else { ("f32", "") };
        let live = self.live_set(outputs);
        let mut order: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&i| !matches!(self.node(ExprId(i as u32)), Node::Var(_) | Node::Lit(_)))
            .collect();
        order.sort_unstable();
        let chunks: Vec<&[usize]> = order.chunks(max_nodes.max(1)).collect();
        let nparts = chunks.len().max(1);
        let mut part_of: HashMap<usize, usize> = HashMap::new();
        for (p, c) in chunks.iter().enumerate() {
            for &i in *c {
                part_of.insert(i, p);
            }
        }
        // scratch slots: values read by a later part (or an output, read by
        // the last part) outside the part that computes them
        let mut slot: HashMap<usize, usize> = HashMap::new();
        let need = |i: usize, reader: usize, slot: &mut HashMap<usize, usize>| {
            if let Some(&p) = part_of.get(&i) {
                if p != reader && !slot.contains_key(&i) {
                    let n = slot.len();
                    slot.insert(i, n);
                }
            }
        };
        for (p, c) in chunks.iter().enumerate() {
            for &i in *c {
                for ch in children(self.node(ExprId(i as u32))).into_iter().flatten() {
                    need(ch.0 as usize, p, &mut slot);
                }
            }
        }
        for o in outputs {
            need(o.0 as usize, nparts - 1, &mut slot);
        }
        let ns = slot.len().max(1);
        let helpers = if f64 { F64_HELPERS } else { "" };
        let header = format!(
            "{helpers}struct Params {{ count: u32, _p1: u32, _p2: u32, _p3: u32 }}\n\
             @group(0) @binding(0) var<storage, read> inputs: array<{ty}>;\n\
             @group(0) @binding(1) var<storage, read_write> scratch: array<{ty}>;\n\
             @group(0) @binding(2) var<storage, read_write> outputs: array<{ty}>;\n\
             @group(0) @binding(3) var<uniform> params: Params;\n\
             @compute @workgroup_size({wg})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let idx = gid.x;\n    if (idx >= params.count) {{ return; }}\n"
        );
        let mut parts = Vec::with_capacity(nparts);
        for p in 0..nparts {
            let chunk: &[usize] = chunks.get(p).copied().unwrap_or(&[]);
            let mut src = header.clone();
            let mut loaded: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut body = String::new();
            let load = |id: ExprId,
                        pre: &mut String,
                        loaded: &mut std::collections::HashSet<usize>| {
                let i = id.0 as usize;
                if !loaded.insert(i) {
                    return;
                }
                match self.node(id) {
                    Node::Var(n) => {
                        writeln!(pre, "    let x{n} = inputs[idx * {n_inputs}u + {n}u];").unwrap()
                    }
                    Node::Lit(_) => {}
                    _ => {
                        if part_of.get(&i) != Some(&p) {
                            writeln!(pre, "    let t{i} = scratch[idx * {ns}u + {}u];", slot[&i])
                                .unwrap();
                        }
                    }
                }
            };
            let mut pre = String::new();
            for &i in chunk {
                let node = self.node(ExprId(i as u32));
                for ch in children(node).into_iter().flatten() {
                    load(ch, &mut pre, &mut loaded);
                }
                loaded.insert(i);
                let mut rhs = expr_str(self, node, suffix, crate::codegen::Dialect::Wgsl);
                if f64 {
                    for f in ["sin(", "exp2(", "log2(", "atan2("] {
                        if let Some(args) = rhs.strip_prefix(f) {
                            rhs = format!("{}_64({args}", &f[..f.len() - 1]);
                            break;
                        }
                    }
                }
                writeln!(body, "    let t{i} = {rhs};").unwrap();
                if let Some(s) = slot.get(&i) {
                    writeln!(body, "    scratch[idx * {ns}u + {s}u] = t{i};").unwrap();
                }
            }
            if p == nparts - 1 {
                for o in outputs {
                    load(*o, &mut pre, &mut loaded);
                }
                let no = outputs.len();
                for (k, o) in outputs.iter().enumerate() {
                    writeln!(
                        body,
                        "    outputs[idx * {no}u + {k}u] = {};",
                        ref_str(self, *o, suffix)
                    )
                    .unwrap();
                }
            }
            src += &pre;
            src += &body;
            src += "}\n";
            parts.push(src);
        }
        SplitKernels {
            parts,
            n_scratch: ns,
            workgroup_size: wg,
        }
    }
}

/// f64 `sin`, `exp2`, `log2`, `atan2` for WGSL on Vulkan (GLSL.std.450 has
/// them for 16- and 32-bit floats only): Cody-Waite reduction and
/// polynomials to a few ulp; `atan2` refines the f32 one by Newton steps.
const F64_HELPERS: &str = "fn sin_64(x: f64) -> f64 {
    let k = round(x * 0.6366197723675814lf);
    let r = (x - k * 1.5707963267948966lf) - k * 6.123233995736766e-17lf;
    let r2 = r * r;
    let s = r * (1.0lf + r2 * (-0.16666666666666666lf + r2 * (0.008333333333333333lf + r2 * (-0.0001984126984126984lf
        + r2 * (2.755731922398589e-6lf + r2 * (-2.505210838544172e-8lf + r2 * (1.6059043836821613e-10lf
        + r2 * (-7.647163731819816e-13lf))))))));
    let c = 1.0lf + r2 * (-0.5lf + r2 * (0.041666666666666664lf + r2 * (-0.001388888888888889lf
        + r2 * (2.48015873015873e-5lf + r2 * (-2.755731922398589e-7lf + r2 * (2.08767569878681e-9lf
        + r2 * (-1.1470745597729725e-11lf)))))));
    let q = i32(k - 4.0lf * floor(k * 0.25lf));
    if (q == 0) { return s; }
    if (q == 1) { return c; }
    if (q == 2) { return -s; }
    return -c;
}
fn pow2i64(n: i32) -> f64 {
    var m = n;
    var b = 2.0lf;
    if (m < 0) { b = 0.5lf; m = -m; }
    var p = 1.0lf;
    for (var i = 0; i < 11; i++) {
        if ((m & 1) == 1) { p = p * b; }
        b = b * b;
        m = m >> 1u;
    }
    return p;
}
fn exp2_64(x: f64) -> f64 {
    let n = floor(clamp(x, -1070.0lf, 1023.0lf) + 0.5lf);
    let y = (x - n) * 0.6931471805599453lf;
    var e = 1.0lf;
    var term = 1.0lf;
    for (var i = 1; i < 18; i++) { term = term * y / f64(i); e = e + term; }
    return e * pow2i64(i32(n));
}
fn log2_64(x: f64) -> f64 {
    var m = x;
    var e = 0.0lf;
    for (var i = 0; i < 20 && m >= 1.8446744073709552e19lf; i++) { m = m * 5.421010862427522e-20lf; e = e + 64.0lf; }
    for (var i = 0; i < 20 && m < 5.421010862427522e-20lf; i++) { m = m * 1.8446744073709552e19lf; e = e - 64.0lf; }
    for (var i = 0; i < 70 && m >= 1.4142135623730951lf; i++) { m = m * 0.5lf; e = e + 1.0lf; }
    for (var i = 0; i < 70 && m < 0.7071067811865476lf; i++) { m = m * 2.0lf; e = e - 1.0lf; }
    let t = (m - 1.0lf) / (m + 1.0lf);
    let t2 = t * t;
    var acc = 0.0lf;
    for (var k = 12; k >= 0; k--) { acc = acc * t2 + 1.0lf / f64(2 * k + 1); }
    return e + 2.0lf * t * acc * 1.4426950408889634lf;
}
fn atan2_64(y: f64, x: f64) -> f64 {
    var th = f64(atan2(f32(y), f32(x)));
    for (var i = 0; i < 2; i++) {
        let s = sin_64(th);
        let c = sin_64(th + 1.5707963267948966lf);
        th = th + (y * c - x * s) / (x * c + y * s);
    }
    return th;
}
";
