//! Native Metal lowering of matmul-shaped tiles: `simdgroup_matrix`.
//!
//! WGSL cannot name a tensor core. When a tiled reduction is a plain
//! bilinear sum — body `a·b` over one reduce axis, each operand read
//! along one of the two tile axes and the reduce axis — the same
//! comprehension lowers to Metal Shading Language with 8×8 simdgroup
//! matrix fragments: a 64×64 block per threadgroup, four simdgroups of
//! 32×32, operands staged through threadgroup memory from any affine
//! layout (transposed, strided, batched), and the comp's epilogue map
//! (bias, activation) run on the stored block before the write-out.
//!
//! Everything else — the staging, the bounds, the epilogue — is the
//! same comprehension semantics as the WGSL tile; only the inner
//! product is an intrinsic. A kernel that does not fit the pattern
//! falls back to WGSL, so a program is a mix.
use std::fmt::Write;

use tang_expr::Node;

use super::tile::emit_scalar;
use super::wgsl::WgslKernel;
use super::{fixed, numel, Arg, TensorGraph, TensorId};

/// Rows × cols of output per threadgroup.
const BLOCK: usize = 64;
/// Reduce elements staged per step.
const CHUNK: usize = 32;
const THREADS: usize = 128;
/// Row strides of the staged tiles, padded past the bank period so the
/// 8 rows of a fragment load land in different banks.
/// Padding of the staged tiles' row strides past the bank period, so
/// the 8 rows of a fragment load land in different banks.
const PAD: usize = 4;

/// An operand's flat address as an affine form over the comp's space.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct Addr {
    pub offset: i64,
    pub coefs: Vec<i64>,
}

/// One product of the bilinear body: buffers (storage nodes) of the two
/// operands and their addresses over the reduction's space; A has no
/// coefficient on `dx`, B none on `dy`.
///
/// An operand may instead be an inlined map over buffers (`map_a` /
/// `map_b`: the index of its arg in the reduction): the value is
/// recomputed where it is staged — dS of an attention backward, rebuilt
/// in every dQ/dK tile from the tensors it is a pointwise function of.
/// Its `buf` is then the map's node and its `addr` the map's row-major
/// address (for the staging order).
#[derive(Clone, PartialEq, Eq)]
pub(super) struct Term {
    pub buf_a: usize,
    pub addr_a: Addr,
    pub buf_b: usize,
    pub addr_b: Addr,
    pub map_a: Option<usize>,
    pub map_b: Option<usize>,
}

/// A matmul-shaped tile (see [`TensorGraph::bilinear_tile`]).
pub(super) struct Bilinear {
    pub red_node: usize,
    pub epilogue: bool,
    /// the first term (the Metal lowering handles one)
    pub buf_a: usize,
    pub buf_b: usize,
    pub addr_a: Addr,
    pub addr_b: Addr,
    /// every term of the sum (of the first output)
    pub terms: Vec<Term>,
    /// per output value (a tuple-valued reduction has several, e.g. a
    /// merged gate/up projection), the terms of its sum
    pub outs: Vec<Vec<Term>>,
    pub pts: Vec<usize>,
    /// tile axes: columns, rows; the remaining point dims are `others`
    pub dx: usize,
    pub dy: usize,
    pub others: Vec<usize>,
    pub m: usize,
    pub n: usize,
    pub k_total: usize,
    /// index of the reduce axis in the space
    pub kdim: usize,
    /// per output, a constant factor on its whole sum (literals in the
    /// products, e.g. the attention backward's 1/√d); 1 when there are none
    pub scales: Vec<f64>,
}

impl TensorGraph {
    /// Flat address of `a` (row-major over its source's shape) as an
    /// affine form over the iteration space.
    pub(super) fn flat_addr(&self, a: &Arg, rank: usize) -> Addr {
        let shape = fixed(&self.nodes[a.src.n()].shape);
        let mut stride = 1i64;
        let mut addr = Addr {
            offset: 0,
            coefs: vec![0; rank],
        };
        for (f, &n) in a.view.forms.iter().zip(&shape).rev() {
            addr.offset += f.offset * stride;
            for (c, &x) in addr.coefs.iter_mut().zip(&f.coefs) {
                *c += x * stride;
            }
            stride *= n as i64;
        }
        addr
    }

    /// The bilinear tile behind `root`, if `root` is (or is the epilogue
    /// of) a reduction over one axis whose body is a product of two leaf
    /// reads, each varying along one of the two tile axes and the reduce
    /// axis — the shape a matrix unit computes. Shared by the native
    /// lowerings (Metal simdgroup matrices, CUDA wmma).
    pub(super) fn bilinear_tile(&self, root: usize, reads: &[usize]) -> Option<Bilinear> {
        let c = self.comp_at(root)?;
        let red_node = self.tiled_reduction(root)?;
        let rc = self.comp_at(red_node)?;
        let epilogue = red_node != root;
        if rc.red.len() != 1 || rc.tuple > 1 {
            return None;
        }
        // the epilogue must read the reduction at its own coordinates
        if epilogue
            && self.epilogue_read(c, red_node)?
                != super::View::identity(rc.points().len(), rc.points().len()).forms
        {
            return None;
        }
        if c.points().len() != rc.points().len() {
            return None;
        }
        // body: a sum of products of two leaf reads (several terms when
        // an operand is split, e.g. rope's even/odd halves)
        let nodes = rc.body.nodes_slice();
        // (two vars, and the product of any literals among the factors)
        fn factors(
            nodes: &[Node],
            e: tang_expr::ExprId,
            vars: &mut Vec<usize>,
            k: &mut f64,
        ) -> bool {
            match nodes[e.index() as usize] {
                Node::Mul(a, b) => factors(nodes, a, vars, k) && factors(nodes, b, vars, k),
                Node::Var(p) => {
                    vars.push(p as usize);
                    true
                }
                Node::Lit(b) => {
                    *k *= f64::from_bits(b);
                    true
                }
                _ => false,
            }
        }
        fn products(
            nodes: &[Node],
            e: tang_expr::ExprId,
            out: &mut Vec<(usize, usize, f64)>,
        ) -> bool {
            match nodes[e.index() as usize] {
                Node::Add(a, b) => products(nodes, a, out) && products(nodes, b, out),
                Node::Mul(..) => {
                    let (mut vars, mut k) = (Vec::new(), 1.0);
                    if !factors(nodes, e, &mut vars, &mut k) || vars.len() != 2 {
                        return false;
                    }
                    out.push((vars[0], vars[1], k));
                    true
                }
                _ => false,
            }
        }
        let mut out_pairs: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut scales = Vec::new();
        for &o in &rc.body_outs {
            let mut pairs = Vec::new();
            if !products(nodes, o, &mut pairs) || pairs.is_empty() {
                return None;
            }
            // one factor for the whole sum
            let k = pairs[0].2;
            if pairs.iter().any(|t| t.2 != k) {
                return None;
            }
            scales.push(k);
            out_pairs.push(pairs.into_iter().map(|(a, b, _)| (a, b)).collect());
        }
        let pts = fixed(rc.points());
        let k_total = rc.red[0].n();
        let rank = rc.rank();
        let kdim = rank - 1;
        let mut non_unit = (0..pts.len()).rev().filter(|&d| pts[d] > 1);
        let dx = non_unit.next()?;
        let dy = super::tile::largest_other(&pts, dx)?;
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dx && d != dy).collect();
        let (n, m) = (pts[dx], pts[dy]);
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let is_buffer = |j: usize| reads.contains(&storage(j));
        // an inlined map over buffers and iotas, read through plain views:
        // recomputed at staging
        let is_map = |a: &Arg| -> bool {
            let p = a.src.n();
            if a.view.guard || !self.nodes[p].inline {
                return false;
            }
            let Some(mc) = self.comp_at(p) else {
                return false;
            };
            // a short contraction over buffers nested in the map (S = Q·Kᵀ
            // inside dS): lowerings that can run it per staged block do
            let nested = |j: usize| {
                self.nodes[j].inline
                    && self.comp_at(j).is_some_and(|r| {
                        r.monoid.as_ref().map(super::Monoid::kind) == Some(super::MonoidKind::Sum)
                            && r.red.len() == 1
                            && r.tuple == 0
                            && super::bilinear_terms(&r.body, r.body_outs[0]).is_some()
                            && r.args.iter().all(|x| !x.view.guard && is_buffer(x.src.n()))
                    })
            };
            mc.monoid.is_none()
                && mc.red.is_empty()
                && mc.width() == 1
                && mc.args.iter().all(|s| {
                    !s.view.guard
                        && (matches!(self.nodes[s.src.n()].op, super::Op::Iota(_))
                            || is_buffer(s.src.n())
                            || nested(s.src.n()))
                })
        };
        let mut outs: Vec<Vec<Term>> = Vec::with_capacity(out_pairs.len());
        for pairs in out_pairs {
            let mut terms = Vec::with_capacity(pairs.len());
            for (va, vb) in pairs {
                let addr_a = self.flat_addr(&rc.args[va], rank);
                let addr_b = self.flat_addr(&rc.args[vb], rank);
                // A varies along rows (dy) and k, B along cols (dx) and k
                let (a_arg, b_arg, addr_a, addr_b) =
                    if addr_a.coefs[dx] == 0 && addr_b.coefs[dy] == 0 {
                        (va, vb, addr_a, addr_b)
                    } else if addr_a.coefs[dy] == 0 && addr_b.coefs[dx] == 0 {
                        (vb, va, addr_b, addr_a)
                    } else {
                        return None;
                    };
                let operand = |v: usize| -> Option<(usize, Option<usize>)> {
                    let a = &rc.args[v];
                    if is_buffer(a.src.n()) {
                        Some((storage(a.src.n()), None))
                    } else if is_map(a) {
                        Some((a.src.n(), Some(v)))
                    } else {
                        None
                    }
                };
                let (buf_a, map_a) = operand(a_arg)?;
                let (buf_b, map_b) = operand(b_arg)?;
                // a map is staged in the order of the buffer it reads
                // along the most dims (its own layout may be transposed)
                let layout = |v: usize, addr: Addr| -> Addr {
                    let outer = &rc.args[v];
                    let mc = self.comp_at(outer.src.n()).unwrap();
                    mc.args
                        .iter()
                        .filter(|s| is_buffer(s.src.n()))
                        .map(|s| {
                            let view = s.view.compose(&outer.view, rank);
                            self.flat_addr(&Arg { src: s.src, view }, rank)
                        })
                        .max_by_key(|a| a.coefs.iter().filter(|&&c| c != 0).count())
                        .unwrap_or(addr)
                };
                let addr_a = match map_a {
                    Some(v) => layout(v, addr_a),
                    None => addr_a,
                };
                let addr_b = match map_b {
                    Some(v) => layout(v, addr_b),
                    None => addr_b,
                };
                terms.push(Term {
                    buf_a,
                    addr_a,
                    buf_b,
                    addr_b,
                    map_a,
                    map_b,
                });
            }
            outs.push(terms);
        }
        let terms = outs[0].clone();
        // epilogue args: the reduction, or buffers read at affine addresses
        if epilogue {
            for a in &c.args {
                if a.src.n() != red_node && !is_buffer(a.src.n()) {
                    return None;
                }
            }
        }
        Some(Bilinear {
            red_node,
            epilogue,
            buf_a: terms[0].buf_a,
            buf_b: terms[0].buf_b,
            addr_a: terms[0].addr_a.clone(),
            addr_b: terms[0].addr_b.clone(),
            terms,
            outs,
            pts,
            dx,
            dy,
            others,
            m,
            n,
            k_total,
            kdim,
            scales,
        })
    }

    /// The Metal simdgroup-matrix kernel for `root`, if it is (or is the
    /// epilogue of) a bilinear reduction with a 2-D tile; `None` means
    /// use the WGSL tile.
    pub(super) fn msl_mma_kernel(&self, root: usize, reads: &[usize]) -> Option<WgslKernel> {
        #[allow(non_snake_case)]
        let (LDA, LDB, LDC) = (CHUNK + PAD, BLOCK + PAD, BLOCK + PAD);
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
        let Bilinear {
            red_node,
            epilogue,
            buf_a,
            buf_b,
            addr_a,
            addr_b,
            pts,
            dx,
            dy,
            others,
            m,
            n,
            k_total,
            kdim,
            ..
        } = t;
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let tiles_x = n.div_ceil(BLOCK);
        let tiles_y = m.div_ceil(BLOCK);
        let n_tiles = tiles_x * tiles_y * others.iter().map(|&d| pts[d]).product::<usize>();
        let w_out = c.width();

        let mut s = String::new();
        writeln!(s, "#include <metal_stdlib>").unwrap();
        writeln!(s, "#include <metal_simdgroup_matrix>").unwrap();
        writeln!(s, "using namespace metal;").unwrap();
        writeln!(
            s,
            "// Auto-generated by tang-compile: simdgroup-matrix tile"
        )
        .unwrap();
        writeln!(
            s,
            "struct Params {{ uint count; uint p1; uint p2; uint p3; }};"
        )
        .unwrap();
        let mut params: Vec<String> = reads
            .iter()
            .enumerate()
            .map(|(b, &j)| format!("device const float* b{j} [[buffer({b})]]"))
            .collect();
        params.push(format!("device float* out [[buffer({})]]", reads.len()));
        params.push(format!(
            "constant Params& params [[buffer({})]]",
            reads.len() + 1
        ));
        params.push("uint3 wid [[threadgroup_position_in_grid]]".into());
        params.push("uint tid [[thread_index_in_threadgroup]]".into());
        params.push("uint sg [[simdgroup_index_in_threadgroup]]".into());
        writeln!(s, "kernel void main0({}) {{", params.join(", ")).unwrap();
        // one threadgroup arena: the A and B tiles during the reduce,
        // the stored block (C) afterwards
        let arena = (BLOCK * LDA + CHUNK * LDB).max(BLOCK * LDC);
        writeln!(s, "    threadgroup float smem[{arena}];").unwrap();
        writeln!(s, "    threadgroup float* As = smem;").unwrap();
        writeln!(s, "    threadgroup float* Bs = smem + {}u;", BLOCK * LDA).unwrap();
        writeln!(s, "    threadgroup float* Cs = smem;").unwrap();
        writeln!(s, "    uint tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        writeln!(s, "    uint tile_x = tile % {tiles_x}u;").unwrap();
        writeln!(s, "    uint tile_y = (tile / {tiles_x}u) % {tiles_y}u;").unwrap();
        writeln!(s, "    uint lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    uint p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    uint row0 = tile_y * {BLOCK}u;").unwrap();
        writeln!(s, "    uint col0 = tile_x * {BLOCK}u;").unwrap();
        // the leading-dim part of each operand's address, uniform per tile
        let lead_term = |addr: &Addr| -> String {
            let mut t = format!("{}", addr.offset);
            for &d in &others {
                if addr.coefs[d] != 0 {
                    t.push_str(&format!(" + {} * int(p{d})", addr.coefs[d]));
                }
            }
            t
        };
        writeln!(s, "    int base_a = {};", lead_term(&addr_a)).unwrap();
        writeln!(s, "    int base_b = {};", lead_term(&addr_b)).unwrap();
        writeln!(s, "    simdgroup_float8x8 acc[4][4];").unwrap();
        writeln!(s, "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);").unwrap();
        writeln!(s, "    uint sr = (sg / 2u) * 32u;").unwrap();
        writeln!(s, "    uint sc = (sg % 2u) * 32u;").unwrap();
        // Direct fragments: when every tile is full and each operand is
        // contiguous along its row or its reduce axis, the fragments are
        // loaded straight from device memory (the cache does the sharing
        // the threadgroup stage would) — no staging, no barriers.
        let (cpa, cka) = (addr_a.coefs[dy], addr_a.coefs[kdim]);
        let (cpb, ckb) = (addr_b.coefs[dx], addr_b.coefs[kdim]);
        let direct = m % BLOCK == 0
            && n % BLOCK == 0
            && k_total % 8 == 0
            && (cpa == 1 || cka == 1)
            && (cpb == 1 || ckb == 1);
        if direct {
            // A fragment (rows sr+8i.., k kk..): row-major when k is the
            // unit stride, else stored transposed
            let (lda, ta) = if cka == 1 {
                (cpa, "false")
            } else {
                (cka, "true")
            };
            let (ldb, tb) = if cpb == 1 {
                (ckb, "false")
            } else {
                (cpb, "true")
            };
            writeln!(s, "    for (uint kk = 0u; kk < {k_total}u; kk += 8u) {{").unwrap();
            writeln!(
                s,
                "        simdgroup_float8x8 a[4]; simdgroup_float8x8 b[4];"
            )
            .unwrap();
            writeln!(s, "        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], b{buf_a} + base_a + {cpa} * int(row0 + sr + 8u * i) + {cka} * int(kk), {lda}u, ulong2(0, 0), {ta});").unwrap();
            writeln!(s, "        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], b{buf_b} + base_b + {ckb} * int(kk) + {cpb} * int(col0 + sc + 8u * j), {ldb}u, ulong2(0, 0), {tb});").unwrap();
            writeln!(s, "        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);").unwrap();
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(
                s,
                "    for (uint k0 = 0u; k0 < {k_total}u; k0 += {CHUNK}u) {{"
            )
            .unwrap();
            // staging, walking the buffer in address order
            let stage = |s: &mut String,
                         name: &str,
                         addr: &Addr,
                         pdim: usize,
                         base: &str,
                         extent: usize,
                         buf: usize,
                         row_major: bool| {
                // row_major: stage[point*CHUNK + k]; else stage[k*BLOCK + point]
                let (cp, ck) = (addr.coefs[pdim], addr.coefs[kdim]);
                let k_inner = ck.unsigned_abs() <= cp.unsigned_abs();
                writeln!(
                    s,
                    "        for (uint idx = tid; idx < {}u; idx += {THREADS}u) {{",
                    BLOCK * CHUNK
                )
                .unwrap();
                if k_inner {
                    writeln!(
                        s,
                        "            uint kk = idx % {CHUNK}u; uint pt = idx / {CHUNK}u;"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "            uint pt = idx % {BLOCK}u; uint kk = idx / {BLOCK}u;"
                    )
                    .unwrap();
                }
                writeln!(s, "            uint gp = {base} + pt; uint gk = k0 + kk;").unwrap();
                let slot = if row_major {
                    format!("pt * {LDA}u + kk")
                } else {
                    format!("kk * {LDB}u + pt")
                };
                writeln!(
                s,
                "            {name}[{slot}] = (gp < {extent}u && gk < {k_total}u) ? b{buf}[uint(base_{} + {cp} * int(gp) + {ck} * int(gk))] : 0.0f;",
                if row_major { "a" } else { "b" }
            )
            .unwrap();
                writeln!(s, "        }}").unwrap();
            };
            stage(&mut s, "As", &addr_a, dy, "row0", m, buf_a, true);
            stage(&mut s, "Bs", &addr_b, dx, "col0", n, buf_b, false);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "        for (uint kk = 0u; kk < {CHUNK}u; kk += 8u) {{").unwrap();
            writeln!(
                s,
                "            simdgroup_float8x8 a[4]; simdgroup_float8x8 b[4];"
            )
            .unwrap();
            writeln!(s, "            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], As + (sr + 8u * i) * {LDA}u + kk, {LDA}u);").unwrap();
            writeln!(s, "            for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Bs + kk * {LDB}u + sc + 8u * j, {LDB}u);").unwrap();
            writeln!(s, "            for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], Cs + (sr + 8u * i) * {LDC}u + sc + 8u * j, {LDC}u);").unwrap();
        writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        // epilogue + write-out, one element per thread per step
        writeln!(
            s,
            "    for (uint idx = tid; idx < {}u; idx += {THREADS}u) {{",
            BLOCK * BLOCK
        )
        .unwrap();
        writeln!(s, "        uint p{dy} = row0 + idx / {BLOCK}u;").unwrap();
        writeln!(s, "        uint p{dx} = col0 + idx % {BLOCK}u;").unwrap();
        writeln!(s, "        if (p{dy} < {m}u && p{dx} < {n}u) {{").unwrap();
        writeln!(
            s,
            "            float red = Cs[(idx / {BLOCK}u) * {LDC}u + idx % {BLOCK}u];"
        )
        .unwrap();
        let vals: Vec<String> = if epilogue {
            let prank = c.points().len();
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
                        e.push_str(&format!(" + {coef} * int(p{d})"));
                    }
                }
                arg_vals.push(format!("b{}[uint({e})]", storage(a.src.n())));
            }
            let mut body = String::new();
            let outs = emit_scalar(&mut body, &c.body, &c.body_outs, "e0_", &|v| {
                arg_vals[v as usize].clone()
            });
            for line in body.lines() {
                writeln!(s, "            {}", line.replacen("let ", "float ", 1)).unwrap();
            }
            outs
        } else {
            vec!["red".to_string()]
        };
        let mut flat = String::from("0u");
        for (d, &np) in pts.iter().enumerate() {
            flat = format!("({flat}) * {np}u + p{d}");
        }
        writeln!(s, "            uint g = {flat};").unwrap();
        for (slot, v) in vals.iter().enumerate() {
            writeln!(s, "            out[g * {w_out}u + {slot}u] = {v};").unwrap();
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        Some(WgslKernel {
            source: s,
            node: TensorId(root as u32),
            reads: reads.iter().map(|&j| TensorId(j as u32)).collect(),
            points: numel(c.points()) as u32,
            count: (n_tiles * THREADS) as u32,
            workgroup_size: THREADS as u32,
            writes: Vec::new(),
        })
    }
}
