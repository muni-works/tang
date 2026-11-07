//! The attention backward as one kernel per key block (FlashAttention-2's
//! schedule), derived from the graph `backward` produces.
//!
//! After the pass pipeline the gradient of `softmax(q·kᵀ)·v` is a
//! tuple reduction over queries, `(dV, dK) = Σ_i (P̃(i,j)·G[i,·],
//! dS(i,j)·Q[i,·])`, plus `dQ = Σ_j dS(i,j)·K[j,·]`, where P̃ and dS are
//! inlined pointwise maps of two short contractions (`S = q·kᵀ`,
//! `dP̃ = G·vᵀ`), iotas (the mask) and per-row buffers (the softmax state,
//! `rowsum(dO·O)`). [`TensorGraph::attn_bwd_plan`] recognizes that shape
//! — any maps over any such contractions, not attention's formulas —
//! and [`TensorGraph::cuda_attn_bwd_kernel`] lowers it: a block owns
//! `BJ` keys, keeps their dV/dK accumulators in registers, and walks
//! the query chunks; per chunk the two contractions run on `mma`
//! m16n8k16 bf16, the maps' bodies run on the accumulator elements,
//! whose registers are the A operand of `dV += P̃ᵀ·G`, `dK += dSᵀ·Q`,
//! and dS (through shared memory) the A operand of the chunk's dQ
//! partial, added to an f32 buffer with atomics. Chunks the maps'
//! affine mask zeroes entirely (causal) are skipped.
//!
//! The dQ reduction becomes a zero-filled buffer the kernel accumulates
//! into ([`TensorGraph::to_cuda_variant`] patches the program), the
//! same in every variant so a tuner cannot mix the two halves.
use std::fmt::Write;

use super::super::msl_mma::Addr;
use super::super::tile::emit_scalar;
use super::super::wgsl::WgslProgram;
use super::super::{fixed, Arg, Op, TensorGraph, TensorId};
use super::{f32_lit, float_literals, mma16_prelude, HALF_PRELUDE, MMA_PRELUDE};

/// Keys per block (16 per warp).
const BJ: usize = 64;
/// Queries per chunk.
const BI: usize = 64;
const WARPS: usize = BJ / 16;
const THREADS: usize = WARPS * 32;
/// Shared-memory row padding (halves): keeps `ldmatrix` rows in
/// distinct banks.
const PADH: usize = 8;

/// An operand staged into shared memory as `[rows][e]` halves: rows
/// along the query (`side_i`) or key dim, `e` the contraction's lane.
/// Addresses are over the kv space plus a lane coord (index `rank`).
#[derive(Clone, PartialEq)]
struct Stage {
    buf: usize,
    addr: Addr,
    side_i: bool,
    e: usize,
}

/// A short contraction nested in the maps, `coef · Σ_e J[j,e]·I[i,e]`.
struct NestedGemm {
    node: usize,
    coef: f64,
    j_stage: usize,
    i_stage: usize,
    e: usize,
}

pub(crate) struct AttnBwd {
    /// the kv tile's root (materialized) and its tuple reduction
    pub root: usize,
    red: usize,
    /// the dQ reduction (materialized f32, accumulated with atomics)
    pub dq: usize,
    /// the dQ reduction's epilogue map (untiled by the plan), if any
    pub dq_epi: Option<usize>,
    /// contractions nested in the maps, to inline
    pub nested_nodes: Vec<usize>,
}

/// Per kv output: (map arg index in the reduction, B stage, scale).
struct Outs {
    map_arg: usize,
    b_stage: usize,
    scale: f64,
}

struct Plan {
    red: usize,
    root: usize,
    pts: Vec<usize>,
    dx: usize,
    dy: usize,
    kdim: usize,
    others: Vec<usize>,
    /// keys, queries, lanes of the outputs
    nk: usize,
    nq: usize,
    d: usize,
    stages: Vec<Stage>,
    nested: Vec<NestedGemm>,
    outs: Vec<Outs>,
    /// which kv output's map dQ reads, dQ's K stage, scale, dQ address
    dq_out: usize,
    dq_k: usize,
    dq_scale: f64,
    dq_addr: Addr,
    dq_node: usize,
    zero: Option<super::super::Form>,
}

fn trace<T>(k: u32) -> Option<T> {
    if std::env::var_os("TANG_ATTN_DEBUG").is_some() {
        eprintln!("attn_bwd: rejected at {k}");
    }
    None
}

fn aff(a: &Addr, coord: &dyn Fn(usize) -> String) -> String {
    let mut e = format!("{}", a.offset);
    for (d, &c) in a.coefs.iter().enumerate() {
        if c != 0 {
            e.push_str(&format!(" + {c} * {}", coord(d)));
        }
    }
    e
}

impl TensorGraph {
    fn kernel_reads(&self, root: usize) -> Vec<usize> {
        self.inline_closure(root)
            .into_iter()
            .filter(|&j| {
                j != root
                    && !self.nodes[j].inline
                    && !matches!(self.nodes[j].op, Op::Iota(_) | Op::Reshape(_))
            })
            .collect()
    }

    /// Per point dim of epilogue `root`, the point dim of its reduction
    /// `red` it reads at (`None`: a unit dim), when the epilogue reads
    /// the reduction through a pick of its points.
    fn epi_coords(&self, root: usize, red: usize) -> Option<Vec<Option<usize>>> {
        let c = self.comp_at(root)?;
        let rc = self.comp_at(red)?;
        let rp = fixed(rc.points());
        let cp = fixed(c.points());
        let forms = self.epilogue_read(c, red)?;
        let mut m: Vec<Option<usize>> = vec![None; cp.len()];
        for (d, f) in forms.iter().enumerate() {
            let nz: Vec<usize> = (0..f.coefs.len()).filter(|&e| f.coefs[e] != 0).collect();
            match nz[..] {
                [] if rp[d] == 1 && f.offset == 0 => {}
                [e] if f.coefs[e] == 1 && f.offset == 0 && e < cp.len() && cp[e] == rp[d] => {
                    m[e] = Some(d)
                }
                _ => return None,
            }
        }
        (0..cp.len())
            .all(|e| m[e].is_some() || cp[e] == 1)
            .then_some(m)
    }

    /// The fused attention-backward group in `outputs`' program, if any
    /// (see the module docs). `self` is scheduled for the CUDA lowering
    /// (nested contractions inlined). `TANG_NO_ATTN_BWD=1` turns it off.
    pub(crate) fn attn_bwd_plan(&self, outputs: &[TensorId]) -> Option<AttnBwd> {
        if std::env::var("TANG_NO_ATTN_BWD").ok().as_deref() == Some("1") {
            return None;
        }
        let live = self.live(outputs);
        for root in (0..self.nodes.len()).filter(|&i| live[i] && !self.nodes[i].inline) {
            if self.nodes[root].tile.0 == 0 {
                continue;
            }
            let Some(red) = self.tiled_reduction(root) else {
                continue;
            };
            let rc = self.comp_at(red).unwrap();
            if rc.body_outs.len() != 2 {
                continue;
            }
            // the dQ reduction: a sum over one dim of (one of the kv
            // tile's maps) × a buffer
            let maps: Vec<usize> = rc
                .args
                .iter()
                .filter(|a| self.nodes[a.src.n()].inline && self.comp_at(a.src.n()).is_some())
                .map(|a| a.src.n())
                .collect();
            for dq in (0..self.nodes.len()).filter(|&i| live[i] && i != red && i != root) {
                let Some(qc) = self.comp_at(dq) else { continue };
                if qc.red.len() != 1
                    || qc.tuple != 0
                    || qc.body_outs.len() != 1
                    || qc.monoid.as_ref().map(super::super::Monoid::kind)
                        != Some(super::super::MonoidKind::Sum)
                    || !qc.args.iter().any(|a| maps.contains(&a.src.n()))
                {
                    continue;
                }
                // its epilogue, if it is inlined under one
                let dq_epi = (0..self.nodes.len()).find(|&e| {
                    live[e] && e != dq && self.tiled_reduction(e) == Some(dq) && e != root
                });
                // the maps' short contractions over buffers, recomputed
                // in the kernel
                let mut nested_nodes = Vec::new();
                for &m in &maps {
                    for a in &self.comp_at(m).unwrap().args {
                        let j = a.src.n();
                        let Some(r) = self.comp_at(j) else { continue };
                        let ok = r.monoid.as_ref().map(super::super::Monoid::kind)
                            == Some(super::super::MonoidKind::Sum)
                            && r.red.len() == 1
                            && r.tuple == 0
                            && r.red[0].fixed().is_some_and(|e| e <= 128)
                            && super::super::bilinear_terms(&r.body, r.body_outs[0]).is_some()
                            && r.args.iter().all(|x| {
                                !x.view.guard
                                    && !self.nodes[x.src.n()].inline
                                    && !matches!(self.nodes[x.src.n()].op, Op::Iota(_))
                            });
                        if ok && !nested_nodes.contains(&j) {
                            nested_nodes.push(j);
                        }
                    }
                }
                let plan = AttnBwd {
                    root,
                    red,
                    dq,
                    dq_epi,
                    nested_nodes,
                };
                let mut g = self.clone();
                g.attn_bwd_schedule(&plan);
                let reads = g.kernel_reads(root);
                if g.attn_bwd_build(root, red, dq, &reads).is_some() {
                    return Some(plan);
                }
            }
        }
        None
    }

    /// Apply the plan's schedule: dQ materialized and untiled, its
    /// epilogue a plain map, the nested contractions inline.
    pub(crate) fn attn_bwd_schedule(&mut self, p: &AttnBwd) {
        self.nodes[p.dq].inline = false;
        self.nodes[p.dq].tile = (0, 0, 0);
        self.nodes[p.dq].split = 0;
        if let Some(e) = p.dq_epi {
            self.nodes[e].tile = (0, 0, 0);
        }
        for &n in &p.nested_nodes {
            self.nodes[n].inline = true;
            self.nodes[n].tile = (0, 0, 0);
            self.nodes[n].split = 0;
        }
    }

    /// Replace the kv tile's kernel with the fused one and the dQ
    /// reduction's with a zero fill launched before it.
    pub(crate) fn attn_bwd_patch(&self, p: &AttnBwd, program: &mut WgslProgram) -> bool {
        let Some(ki) = program
            .kernels
            .iter()
            .position(|k| k.node.index() == p.root)
        else {
            return false;
        };
        let Some(qi) = program.kernels.iter().position(|k| k.node.index() == p.dq) else {
            return false;
        };
        let reads: Vec<usize> = program.kernels[ki]
            .reads
            .iter()
            .map(|t| t.index())
            .collect();
        let Some(src) = self.cuda_attn_bwd_kernel(p.root, p.red, p.dq, &reads) else {
            return false;
        };
        let mut zk = program.kernels.remove(qi);
        let points = zk.points as usize;
        zk.source = format!(
            "// Auto-generated by tang-compile: zero fill (dQ accumulator)\n#define TANG_THREADS 256\nextern \"C\" __global__ void __launch_bounds__(256) main0(float* __restrict__ out, unsigned count) {{\n    unsigned i = blockIdx.x * 256u + threadIdx.x;\n    if (i < {points}u) out[i] = 0.0f;\n    (void)count;\n}}\n"
        );
        zk.reads.clear();
        zk.workgroup_size = 256;
        zk.count = (points.div_ceil(256) * 256) as u32;
        let ki = program
            .kernels
            .iter()
            .position(|k| k.node.index() == p.root)
            .unwrap();
        program.kernels.insert(ki, zk);
        let k = &mut program.kernels[ki + 1];
        k.source = src;
        k.reads.push(TensorId(p.dq as u32));
        k.workgroup_size = THREADS as u32;
        let tiles = super::src_define(&k.source, "TANG_TILES");
        k.count = (tiles * THREADS) as u32;
        true
    }

    fn attn_bwd_build(&self, root: usize, red: usize, dq: usize, reads: &[usize]) -> Option<Plan> {
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let is_buffer = |j: usize| reads.contains(&storage(j));
        let b = self.bilinear_tile(red, reads).or_else(|| trace(21))?;
        if root != red {
            self.epi_coords(root, red)?;
        }
        if b.red_node != red || b.outs.len() != 2 || b.outs.iter().any(|t| t.len() != 1) {
            return trace(1);
        }
        let rc = self.comp_at(red).or_else(|| trace(23))?;
        let rank = rc.rank();
        let (dx, dy, kdim) = (b.dx, b.dy, b.kdim);
        let (nk, d, nq) = (b.m, b.n, b.k_total);
        if !nk.is_multiple_of(BJ) || !nq.is_multiple_of(BI) || !(d == 64 || d == 128) {
            return trace(2);
        }
        // every buffer the kernel touches is bf16 or f32
        for &r in reads.iter().chain([&root]) {
            match self.nodes[r].precision.storage {
                super::super::Format::F32 | super::super::Format::BF16 => {}
                _ => return None,
            }
        }
        let mut stages: Vec<Stage> = Vec::new();
        let mut stage_of = |s: Stage| -> usize {
            stages.iter().position(|x| *x == s).unwrap_or_else(|| {
                stages.push(s);
                stages.len() - 1
            })
        };
        // a kv-space address with the lane at `from` moved to `rank`
        let lane_last = |a: &Addr, from: usize| -> Addr {
            let mut coefs = a.coefs.clone();
            coefs.resize(rank + 1, 0);
            if from != rank {
                coefs[rank] = coefs[from];
                coefs[from] = 0;
            }
            Addr {
                offset: a.offset,
                coefs,
            }
        };
        let mut nested: Vec<NestedGemm> = Vec::new();
        let mut outs = Vec::new();
        for (w, terms) in b.outs.iter().enumerate() {
            let t = &terms[0];
            // A: a map over (key, query), B: a buffer over (query, lane)
            let map_arg = t.map_a?;
            if t.map_b.is_some() {
                return trace(3);
            }
            let ba = lane_last(&t.addr_b, dx);
            if ba.coefs[dy] != 0 {
                return trace(4);
            }
            let b_stage = stage_of(Stage {
                buf: t.buf_b,
                addr: ba,
                side_i: true,
                e: d,
            });
            for nst in self
                .nested_plan(rc, rank, map_arg, dy, kdim, reads)
                .or_else(|| trace(25))?
            {
                let node = self
                    .comp_at(rc.args[map_arg].src.n())
                    .or_else(|| trace(27))?
                    .args[nst.arg]
                    .src
                    .n();
                if nested.iter().any(|n| n.node == node) {
                    continue;
                }
                let [term] = &nst.terms[..] else {
                    return trace(5);
                };
                if !(nst.e == 64 || nst.e == 128) {
                    return trace(6);
                }
                let j_stage = stage_of(Stage {
                    buf: term.p_buf,
                    addr: term.p_addr.clone(),
                    side_i: false,
                    e: nst.e,
                });
                let i_stage = stage_of(Stage {
                    buf: term.k_buf,
                    addr: term.k_addr.clone(),
                    side_i: true,
                    e: nst.e,
                });
                nested.push(NestedGemm {
                    node,
                    coef: term.coef,
                    j_stage,
                    i_stage,
                    e: nst.e,
                });
            }
            // the map's other args: iotas and buffers not along the lane
            let outer = &rc.args[map_arg];
            let mc = self.comp_at(outer.src.n()).or_else(|| trace(29))?;
            for sa in &mc.args {
                let j = sa.src.n();
                if nested.iter().any(|n| n.node == j) {
                    continue;
                }
                let view = sa.view.compose(&outer.view, rank);
                if view.guard || view.is_gather() {
                    return trace(7);
                }
                match &self.nodes[j].op {
                    Op::Iota(dd) => {
                        if view.forms[*dd].coefs[dx] != 0 {
                            return trace(8);
                        }
                    }
                    _ if is_buffer(j) => {
                        let a = self.flat_addr(&Arg { src: sa.src, view }, rank);
                        if a.coefs[dx] != 0 {
                            return trace(9);
                        }
                    }
                    _ => return None,
                }
            }
            outs.push(Outs {
                map_arg,
                b_stage,
                scale: b.scales[w],
            });
        }
        if nested.is_empty() {
            return trace(10);
        }
        // dQ: Σ_j map(i, j) · K[j, lane], seen from the kv space
        let qc = self.comp_at(dq).or_else(|| trace(31))?;
        let qrank = qc.rank();
        let qpts = fixed(qc.points());
        let (qa, qb) = {
            let terms =
                super::super::bilinear_terms(&qc.body, qc.body_outs[0]).or_else(|| trace(33))?;
            let [(coef, x, y)] = terms[..] else {
                return trace(11);
            };
            let (x, y) = (x as usize, y as usize);
            let is_map = |v: usize| {
                outs.iter()
                    .any(|o| rc.args[o.map_arg].src == qc.args[v].src)
            };
            let (m, k) = if is_map(x) {
                (x, y)
            } else if is_map(y) {
                (y, x)
            } else {
                return trace(12);
            };
            ((m, coef), k)
        };
        let ((qm, dq_scale), qk) = (qa, qb);
        let dq_out = outs
            .iter()
            .position(|o| rc.args[o.map_arg].src == qc.args[qm].src)
            .or_else(|| trace(35))?;
        let kv_view = &rc.args[outs[dq_out].map_arg].view;
        let q_view = &qc.args[qm].view;
        if q_view.guard || q_view.is_gather() || !is_buffer(qc.args[qk].src.n()) {
            return trace(13);
        }
        // dq space dim → kv dim, through the map's dims
        let unit = |f: &super::super::Form| -> Option<Option<usize>> {
            if f.offset != 0 {
                return trace(14);
            }
            let nz: Vec<usize> = (0..f.coefs.len()).filter(|&d| f.coefs[d] != 0).collect();
            match nz[..] {
                [] => Some(None),
                [d] if f.coefs[d] == 1 => Some(Some(d)),
                _ => None,
            }
        };
        let mut qmap: Vec<Option<usize>> = vec![None; qrank];
        for (qf, kf) in q_view.forms.iter().zip(&kv_view.forms) {
            match (
                unit(qf).or_else(|| trace(37))?,
                unit(kf).or_else(|| trace(39))?,
            ) {
                (Some(a), Some(b)) => qmap[a] = Some(b),
                (None, None) => {}
                _ => return None,
            }
        }
        // the dQ reduction runs over keys, one point dim over queries;
        // its one unmatched non-unit point dim is the lane
        if qmap[qrank - 1] != Some(dy) || !qmap.contains(&Some(kdim)) {
            return trace(15);
        }
        let mut lane = None;
        for qd in 0..qpts.len() {
            if qmap[qd].is_none() && qpts[qd] > 1 {
                if lane.is_some() || qpts[qd] != d {
                    return trace(16);
                }
                lane = Some(qd);
            }
        }
        let lane = lane?;
        // a dq-space address as a kv-space one (lane last)
        let to_kv = |a: &Addr| -> Option<Addr> {
            let mut coefs = vec![0i64; rank + 1];
            for (qd, &c) in a.coefs.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                if qd == lane {
                    coefs[rank] += c;
                } else {
                    coefs[qmap[qd]?] += c;
                }
            }
            Some(Addr {
                offset: a.offset,
                coefs,
            })
        };
        let ka = to_kv(&self.flat_addr(&qc.args[qk], qrank)).or_else(|| trace(41))?;
        if ka.coefs[kdim] != 0 {
            return trace(17);
        }
        let dq_k = stage_of(Stage {
            buf: storage(qc.args[qk].src.n()),
            addr: ka,
            side_i: false,
            e: d,
        });
        // dQ's flat address (row-major over its points)
        let mut flat = Addr {
            offset: 0,
            coefs: vec![0; qrank],
        };
        let mut stride = 1i64;
        for qd in (0..qpts.len()).rev() {
            flat.coefs[qd] = stride;
            stride *= qpts[qd] as i64;
        }
        let dq_addr = to_kv(&flat).or_else(|| trace(43))?;
        // every staged operand moves along its rows, the lane, and the
        // lead dims only
        for s in &stages {
            let row = if s.side_i { kdim } else { dy };
            for (dd, &c) in s.addr.coefs.iter().enumerate() {
                if c != 0 && dd != row && dd != rank && !b.others.contains(&dd) {
                    return trace(18);
                }
            }
        }
        // the skip condition: both maps zero there
        let zs: Vec<Option<super::super::Form>> = outs
            .iter()
            .map(|o| self.map_zero_cond(rc, o.map_arg))
            .collect();
        let zero = match &zs[..] {
            [Some(a), Some(b)] if a == b => Some(a.clone()),
            _ => None,
        }
        .filter(|f| f.coefs[dx] == 0);
        // shared memory: stages, plus dS
        let smem = stages
            .iter()
            .map(|s| (if s.side_i { BI } else { BJ }) * (s.e + PADH) * 2)
            .sum::<usize>()
            + BJ * (BI + PADH) * 2;
        if smem > 200 * 1024 {
            return trace(19);
        }
        Some(Plan {
            red,
            root,
            pts: b.pts.clone(),
            dx,
            dy,
            kdim,
            others: b.others.clone(),
            nk,
            nq,
            d,
            stages,
            nested,
            outs,
            dq_out,
            dq_k,
            dq_scale,
            dq_addr,
            dq_node: dq,
            zero,
        })
    }

    /// CUDA source of the fused attention backward for kv tile `root`
    /// (reads `reads`, then the dQ buffer as one more binding).
    pub(crate) fn cuda_attn_bwd_kernel(
        &self,
        root: usize,
        red: usize,
        dq: usize,
        reads: &[usize],
    ) -> Option<String> {
        let p = self.attn_bwd_build(root, red, dq, reads)?;
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let fmt = |j: usize| self.nodes[j].precision.storage;
        let ld = |j: usize, idx: &str| -> String {
            match fmt(j) {
                super::super::Format::BF16 => format!("tang_b2f(b{j}[{idx}])"),
                _ => format!("b{j}[{idx}]"),
            }
        };
        let elem = |j: usize| {
            if fmt(j) == super::super::Format::BF16 {
                "unsigned short"
            } else {
                "float"
            }
        };
        let rc = self.comp_at(p.red)?;
        let c = self.comp_at(p.root)?;
        let rank = rc.rank();
        let (dx, dy, kdim, d) = (p.dx, p.dy, p.kdim, p.d);
        let n_jb = p.nk / BJ;
        let n_lead: usize = p.others.iter().map(|&o| p.pts[o]).product();
        let n_tiles = n_jb * n_lead;
        let w_out = c.out[c.out.len() - c.tuple..]
            .iter()
            .map(|x| x.n())
            .product::<usize>();
        // shared layout: stages in order, then dS
        let mut offs = Vec::new();
        let mut off = 0usize;
        for s in &p.stages {
            offs.push(off);
            off += (if s.side_i { BI } else { BJ }) * (s.e + PADH);
        }
        let ds_off = off;
        let smem_halves = off + BJ * (BI + PADH);
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: attention backward (BF16 mma, fused dK/dV/dQ)"
        )
        .unwrap();
        writeln!(
            s,
            "// {BJ} keys per block, {BI}-query chunks, {WARPS} warps"
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {THREADS}").unwrap();
        writeln!(s, "#define TANG_SMEM {}", smem_halves * 2).unwrap();
        writeln!(s, "#define select(f, t, c) ((c) ? (t) : (f))").unwrap();
        s.push_str(MMA_PRELUDE);
        s.push_str(HALF_PRELUDE);
        s.push_str(&mma16_prelude(super::super::Format::BF16));
        s.push_str(LDM_PRELUDE);
        let mut params: Vec<String> = reads
            .iter()
            .map(|&j| format!("const {}* __restrict__ b{j}", elem(j)))
            .collect();
        params.push(format!("const float* b{}", p.dq_node));
        params.push(format!("{}* __restrict__ out", elem(root)));
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({THREADS}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    (void)count;").unwrap();
        writeln!(s, "    float* dqp = (float*)b{};", p.dq_node).unwrap();
        writeln!(
            s,
            "    extern __shared__ __align__(16) unsigned short smh[];"
        )
        .unwrap();
        writeln!(s, "    unsigned tile = blockIdx.x;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) return;").unwrap();
        writeln!(
            s,
            "    unsigned tid = threadIdx.x, warp = tid / 32u, lane = tid % 32u;"
        )
        .unwrap();
        writeln!(s, "    unsigned gq = lane / 4u, tq = lane % 4u;").unwrap();
        writeln!(s, "    unsigned j0 = (tile % {n_jb}u) * {BJ}u;").unwrap();
        writeln!(s, "    unsigned lead = tile / {n_jb}u;").unwrap();
        for &o in p.others.iter().rev() {
            writeln!(
                s,
                "    unsigned p{o} = lead % {}u; lead /= {}u;",
                p.pts[o], p.pts[o]
            )
            .unwrap();
        }
        writeln!(s, "    (void)lead;").unwrap();
        // coordinates over the kv space (+ lane at `rank`)
        let coord = |dd: usize, j: &str, i: &str, e: &str| -> String {
            if dd == rank {
                format!("(int)({e})")
            } else if dd == dy {
                format!("(int)({j})")
            } else if dd == kdim {
                format!("(int)({i})")
            } else {
                format!("(int)p{dd}")
            }
        };
        // stage `t` (rows from origin `r0`) into shared memory
        let stage = |s: &mut String, t: usize, r0: &str| {
            let st = &p.stages[t];
            let rows = if st.side_i { BI } else { BJ };
            let ld_s = st.e + PADH;
            let row_dim = if st.side_i { kdim } else { dy };
            let lane_fast = st.addr.coefs[rank] == 1 || st.addr.coefs[row_dim] != 1;
            let total = rows * st.e;
            let vec8 = fmt(st.buf) == super::super::Format::BF16
                && st.addr.coefs[rank] == 1
                && st.addr.offset % 8 == 0
                && st
                    .addr
                    .coefs
                    .iter()
                    .enumerate()
                    .all(|(dd, &c)| dd == rank || c % 8 == 0);
            writeln!(s, "        {{").unwrap();
            if vec8 {
                let per_row = st.e / 8;
                writeln!(
                    s,
                    "        for (unsigned idx = tid; idx < {}u; idx += {THREADS}u) {{",
                    total / 8
                )
                .unwrap();
                writeln!(
                    s,
                    "            unsigned rr = idx / {per_row}u, ee = (idx % {per_row}u) * 8u;"
                )
                .unwrap();
                let a = aff(&st.addr, &|dd| {
                    coord(dd, &format!("{r0} + rr"), &format!("{r0} + rr"), "ee")
                });
                writeln!(s, "            *(uint4*)(smh + {}u + rr * {ld_s}u + ee) = *(const uint4*)(b{} + ({a}));", offs[t], st.buf).unwrap();
                writeln!(s, "        }}").unwrap();
            } else {
                writeln!(
                    s,
                    "        for (unsigned idx = tid; idx < {total}u; idx += {THREADS}u) {{"
                )
                .unwrap();
                if lane_fast {
                    writeln!(
                        s,
                        "            unsigned rr = idx / {}u, ee = idx % {}u;",
                        st.e, st.e
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "            unsigned rr = idx % {rows}u, ee = idx / {rows}u;"
                    )
                    .unwrap();
                }
                let a = aff(&st.addr, &|dd| {
                    coord(dd, &format!("{r0} + rr"), &format!("{r0} + rr"), "ee")
                });
                let v = ld(st.buf, &format!("(unsigned)({a})"));
                writeln!(
                    s,
                    "            smh[{}u + rr * {ld_s}u + ee] = tang_f2b({v});",
                    offs[t]
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        }}").unwrap();
        };
        // key-side stages, once
        for t in (0..p.stages.len()).filter(|&t| !p.stages[t].side_i) {
            stage(&mut s, t, "j0");
        }
        let n_out = p.outs.len();
        let nfd = d / 8;
        let nfi = BI / 8;
        writeln!(s, "    float acc[{n_out}][{nfd}][4];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int w = 0; w < {n_out}; w++) for (int f = 0; f < {nfd}; f++) for (int r = 0; r < 4; r++) acc[w][f][r] = 0.0f;").unwrap();
        writeln!(s, "    unsigned jw = warp * 16u;").unwrap();
        // the chunk range: skip chunks the mask zeroes entirely
        writeln!(
            s,
            "    for (unsigned i0 = 0u; i0 < {}u; i0 += {BI}u) {{",
            p.nq
        )
        .unwrap();
        if let Some(f) = &p.zero {
            let lo_hi = |dd: usize, lo: &str, hi: &str| -> String {
                let cf = f.coefs[dd];
                match cf.cmp(&0) {
                    std::cmp::Ordering::Greater => format!(" + {cf} * (int)({lo})"),
                    std::cmp::Ordering::Less => format!(" + {cf} * (int)({hi})"),
                    std::cmp::Ordering::Equal => String::new(),
                }
            };
            let mut min = format!("{}", f.offset);
            min.push_str(&lo_hi(dy, "j0", &format!("j0 + {}u", BJ - 1)));
            min.push_str(&lo_hi(kdim, "i0", &format!("i0 + {}u", BI - 1)));
            for &o in &p.others {
                if f.coefs[o] != 0 {
                    min.push_str(&format!(" + {} * (int)p{o}", f.coefs[o]));
                }
            }
            writeln!(s, "        if ({min} > 0) continue;").unwrap();
        }
        writeln!(s, "        __syncthreads();").unwrap();
        for t in (0..p.stages.len()).filter(|&t| p.stages[t].side_i) {
            stage(&mut s, t, "i0");
        }
        writeln!(s, "        __syncthreads();").unwrap();
        // the nested contractions on the warp's 16 keys × the chunk
        for (q, ng) in p.nested.iter().enumerate() {
            let (ja, ia) = (offs[ng.j_stage], offs[ng.i_stage]);
            let le = ng.e + PADH;
            writeln!(s, "        float sa{q}[{nfi}][4];").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int f = 0; f < {nfi}; f++) for (int r = 0; r < 4; r++) sa{q}[f][r] = 0.0f;").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int ks = 0; ks < {}; ks++) {{", ng.e / 16).unwrap();
            writeln!(s, "            unsigned fa[4];").unwrap();
            writeln!(s, "            tang_ldm4(fa, smh + {ja}u + (jw + lane % 8u + 8u * ((lane / 8u) % 2u)) * {le}u + ks * 16u + 8u * (lane / 16u));").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int f = 0; f < {nfi}; f += 2) {{").unwrap();
            writeln!(s, "                unsigned fb[4];").unwrap();
            writeln!(s, "                tang_ldm4(fb, smh + {ia}u + (8u * f + lane % 8u + 8u * (lane / 16u)) * {le}u + ks * 16u + 8u * ((lane / 8u) % 2u));").unwrap();
            writeln!(s, "                tang_mma16(sa{q}[f], fa, fb);").unwrap();
            writeln!(s, "                tang_mma16(sa{q}[f + 1], fa, fb + 2);").unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        // the maps on each accumulator element
        let mut mv_names = Vec::new();
        for (w, o) in p.outs.iter().enumerate() {
            let outer = &rc.args[o.map_arg];
            let mc = self.comp_at(outer.src.n())?;
            let jexpr = "j0 + jw + gq + 8u * (r / 2)";
            let iexpr = "i0 + 8u * f + 2u * tq + (r % 2)";
            let vals: Vec<String> = mc
                .args
                .iter()
                .map(|sa| {
                    if let Some(q) = p.nested.iter().position(|n| n.node == sa.src.n()) {
                        let k = p.nested[q].coef;
                        return if k == 1.0 {
                            format!("sa{q}[f][r]")
                        } else {
                            format!("(sa{q}[f][r] * {})", f32_lit(k))
                        };
                    }
                    let view = sa.view.compose(&outer.view, rank);
                    let cd = |dd: usize| coord(dd, jexpr, iexpr, "0");
                    match &self.nodes[sa.src.n()].op {
                        Op::Iota(dd) => {
                            let f = &view.forms[*dd];
                            format!(
                                "(float)({})",
                                aff(
                                    &Addr {
                                        offset: f.offset,
                                        coefs: f.coefs.clone()
                                    },
                                    &cd
                                )
                            )
                        }
                        _ => {
                            let a = self.flat_addr(&Arg { src: sa.src, view }, rank);
                            ld(
                                storage(sa.src.n()),
                                &format!("(unsigned)({})", aff(&a, &cd)),
                            )
                        }
                    }
                })
                .collect();
            writeln!(s, "        float mv{w}[{nfi}][4];").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int f = 0; f < {nfi}; f++)").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int r = 0; r < 4; r++) {{").unwrap();
            let mut body = String::new();
            let res = emit_scalar(
                &mut body,
                &mc.body,
                &mc.body_outs,
                &format!("m{w}_"),
                &|x| vals[x as usize].clone(),
            );
            for line in body.lines() {
                writeln!(
                    s,
                    "            {}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            writeln!(s, "            mv{w}[f][r] = {};", res[0]).unwrap();
            writeln!(s, "        }}").unwrap();
            mv_names.push(format!("mv{w}"));
        }
        // dS (the map dQ reads) to shared memory as [key][query]
        let dqw = p.dq_out;
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int f = 0; f < {nfi}; f++) {{").unwrap();
        writeln!(s, "            *(unsigned*)(smh + {ds_off}u + (jw + gq) * {}u + 8u * f + 2u * tq) = tang_pack2(mv{dqw}[f][0], mv{dqw}[f][1]);", BI + PADH).unwrap();
        writeln!(s, "            *(unsigned*)(smh + {ds_off}u + (jw + gq + 8u) * {}u + 8u * f + 2u * tq) = tang_pack2(mv{dqw}[f][2], mv{dqw}[f][3]);", BI + PADH).unwrap();
        writeln!(s, "        }}").unwrap();
        // dV/dK += mapᵀ · B: the map's accumulator registers are the A
        // fragments (keys × queries), B from the query-side stage
        for (w, o) in p.outs.iter().enumerate() {
            let bo = offs[o.b_stage];
            let lb = p.stages[o.b_stage].e + PADH;
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int ks = 0; ks < {}; ks++) {{", BI / 16).unwrap();
            writeln!(s, "            unsigned fa[4] = {{tang_pack2(mv{w}[2 * ks][0], mv{w}[2 * ks][1]), tang_pack2(mv{w}[2 * ks][2], mv{w}[2 * ks][3]), tang_pack2(mv{w}[2 * ks + 1][0], mv{w}[2 * ks + 1][1]), tang_pack2(mv{w}[2 * ks + 1][2], mv{w}[2 * ks + 1][3])}};").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int f = 0; f < {nfd}; f += 2) {{").unwrap();
            writeln!(s, "                unsigned fb[4];").unwrap();
            writeln!(s, "                tang_ldm4t(fb, smh + {bo}u + (16u * ks + lane % 8u + 8u * ((lane / 8u) % 2u)) * {lb}u + 8u * f + 8u * (lane / 16u));").unwrap();
            writeln!(s, "                tang_mma16(acc[{w}][f], fa, fb);").unwrap();
            writeln!(
                s,
                "                tang_mma16(acc[{w}][f + 1], fa, fb + 2);"
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        writeln!(s, "        __syncthreads();").unwrap();
        // the chunk's dQ partial: warp w's 16 queries × the lanes, over
        // the block's keys; A = dS from shared (transposed), B = K
        let ko = offs[p.dq_k];
        let lk = d + PADH;
        let cols = 64usize.min(d);
        let nfc = cols / 8;
        writeln!(s, "        unsigned iw = warp * 16u;").unwrap();
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int c0 = 0; c0 < {d}; c0 += {cols}) {{").unwrap();
        writeln!(s, "            float qa[{nfc}][4];").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int f = 0; f < {nfc}; f++) for (int r = 0; r < 4; r++) qa[f][r] = 0.0f;").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int ks = 0; ks < {}; ks++) {{", BJ / 16).unwrap();
        writeln!(s, "                unsigned fa[4];").unwrap();
        writeln!(s, "                tang_ldm4t(fa, smh + {ds_off}u + (16u * ks + lane % 8u + 8u * (lane / 16u)) * {}u + iw + 8u * ((lane / 8u) % 2u));", BI + PADH).unwrap();
        writeln!(s, "                #pragma unroll").unwrap();
        writeln!(s, "                for (int f = 0; f < {nfc}; f += 2) {{").unwrap();
        writeln!(s, "                    unsigned fb[4];").unwrap();
        writeln!(s, "                    tang_ldm4t(fb, smh + {ko}u + (16u * ks + lane % 8u + 8u * ((lane / 8u) % 2u)) * {lk}u + c0 + 8u * f + 8u * (lane / 16u));").unwrap();
        writeln!(s, "                    tang_mma16(qa[f], fa, fb);").unwrap();
        writeln!(s, "                    tang_mma16(qa[f + 1], fa, fb + 2);").unwrap();
        writeln!(s, "                }}").unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int f = 0; f < {nfc}; f++)").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int r = 0; r < 4; r++) {{").unwrap();
        let qaddr = aff(&p.dq_addr, &|dd| {
            coord(
                dd,
                "0",
                "i0 + iw + gq + 8u * (r / 2)",
                "c0 + 8 * f + 2 * tq + (r % 2)",
            )
        });
        let qs = if p.dq_scale == 1.0 {
            String::new()
        } else {
            format!(" * {}f", f32_lit(p.dq_scale))
        };
        writeln!(
            s,
            "                atomicAdd(dqp + ({qaddr}), qa[f][r]{qs});"
        )
        .unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        // write dV/dK through the tile's epilogue
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int f = 0; f < {nfd}; f++)").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < 4; r++) {{").unwrap();
        writeln!(s, "        unsigned p{dy} = j0 + jw + gq + 8u * (r / 2);").unwrap();
        writeln!(s, "        unsigned p{dx} = 8u * f + 2u * tq + (r % 2);").unwrap();
        for (w, o) in p.outs.iter().enumerate() {
            if o.scale == 1.0 {
                writeln!(s, "        float red{w} = acc[{w}][f][r];").unwrap();
            } else {
                writeln!(
                    s,
                    "        float red{w} = acc[{w}][f][r] * {}f;",
                    f32_lit(o.scale)
                )
                .unwrap();
            }
        }
        // the epilogue's point coords from the reduction's
        let epi = if p.root != p.red {
            self.epi_coords(p.root, p.red)?
        } else {
            (0..p.pts.len()).map(Some).collect()
        };
        for (e, m) in epi.iter().enumerate() {
            match m {
                Some(dd) => writeln!(s, "        unsigned e{e} = p{dd};").unwrap(),
                None => writeln!(s, "        unsigned e{e} = 0u;").unwrap(),
            }
        }
        let vals: Vec<String> = if p.root != p.red {
            let prank = c.points().len();
            let mut arg_vals = Vec::new();
            for a in &c.args {
                if a.src.n() == p.red {
                    arg_vals.push(format!("red{}", self.epilogue_slot(a)));
                    continue;
                }
                let addr = self.flat_addr(a, prank);
                arg_vals.push(ld(
                    storage(a.src.n()),
                    &format!("(unsigned)({})", aff(&addr, &|dd| format!("(int)e{dd}"))),
                ));
            }
            let mut body = String::new();
            let outs = emit_scalar(&mut body, &c.body, &c.body_outs, "e0_", &|v| {
                arg_vals[v as usize].clone()
            });
            for line in body.lines() {
                writeln!(
                    s,
                    "        {}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            outs
        } else {
            (0..n_out).map(|w| format!("red{w}")).collect()
        };
        let mut flat = String::from("0u");
        for (e, np) in fixed(c.points()).into_iter().enumerate() {
            flat = format!("({flat}) * {np}u + e{e}");
        }
        writeln!(s, "        unsigned g = {flat};").unwrap();
        for (slot, v) in vals.iter().enumerate() {
            let v = match fmt(root) {
                super::super::Format::BF16 => format!("tang_f2b({v})"),
                _ => v.clone(),
            };
            writeln!(s, "        out[g * {w_out}u + {slot}u] = {v};").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(s)
    }
}

/// `ldmatrix` x4 (plain and transposed), and the nested mma's operand
/// order. `.b` and `16` are split: the canonical renaming would take
/// `b16` for a buffer.
const LDM_PRELUDE: &str = r#"static __device__ __forceinline__ void tang_ldm4(unsigned* r, const unsigned short* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared." "b" "16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
static __device__ __forceinline__ void tang_ldm4t(unsigned* r, const unsigned short* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared." "b" "16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
"#;
