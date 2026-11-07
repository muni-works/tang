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

/// Keys per block (16 per warp): the variants' choices, the first the
/// default.
const BJS: [usize; 2] = [64, 128];
/// Queries per chunk.
const BI: usize = 64;
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
    /// intermediate f32 buffers the nested contractions read, stored in
    /// bf16 instead (the tensor-core operand format: staged by copy)
    pub bf16_bufs: Vec<usize>,
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

/// A tile in shared memory: `rows × e` halves at offset `off`, stored
/// `[rows][e]` (row stride `e + PADH`) or, `tm`, `[e][rows]`.
struct Smem {
    off: usize,
    /// double-buffered: the runtime buffer offset `ib` is added
    dynb: bool,
    tm: bool,
    rows: usize,
    e: usize,
    /// 128-byte swizzled, unpadded (the `wgmma` operand layout): the
    /// contiguous dim in 64-element rows, 16-byte chunks XORed by the
    /// other index mod 8
    sw: bool,
}

impl Smem {
    fn halves(&self) -> usize {
        if self.sw {
            return self.rows * self.e;
        }
        if self.tm {
            self.e * (self.rows + PADH)
        } else {
            self.rows * (self.e + PADH)
        }
    }

    fn at(&self, row: &str, e: &str) -> String {
        let b = if self.dynb { "ib + " } else { "" };
        if self.sw {
            let (y, x, ny) = if self.tm {
                (e, row, self.e)
            } else {
                (row, e, self.rows)
            };
            return format!(
                "{b}{}u + (({x}) / 64u) * {}u + ({y}) * 64u + (((({x}) % 64u) / 8u ^ (({y}) % 8u)) * 8u) + ({x}) % 8u",
                self.off,
                ny * 64
            );
        }
        if self.tm {
            format!("{b}{}u + ({e}) * {}u + ({row})", self.off, self.rows + PADH)
        } else {
            format!("{b}{}u + ({row}) * {}u + ({e})", self.off, self.e + PADH)
        }
    }

    /// `ldmatrix` x4 into `dst`: an mma A fragment (16 × k16, `a`) or two
    /// B fragments (2 × n8 × k16) whose outer index (m or n) starts at
    /// `o0` and k at `k0`; `o_row`: the outer index runs along the tile's
    /// rows (else along e). Plain when k is the stored-contiguous dim,
    /// transposed when the outer index is.
    fn ldm(&self, dst: &str, a: bool, o_row: bool, o0: &str, k0: &str) -> String {
        let (oo, ko) = if a {
            ("8u * ((lane / 8u) % 2u)", "8u * (lane / 16u)")
        } else {
            ("8u * (lane / 16u)", "8u * ((lane / 8u) % 2u)")
        };
        // contiguous along k?
        let plain = o_row != self.tm;
        let (o, k) = if plain {
            (format!("{o0} + {oo} + lane % 8u"), format!("{k0} + {ko}"))
        } else {
            (format!("{o0} + {oo}"), format!("{k0} + {ko} + lane % 8u"))
        };
        let (row, e) = if o_row { (o, k) } else { (k, o) };
        format!(
            "{}({dst}, smh + {});",
            if plain { "tang_ldm4" } else { "tang_ldm4t" },
            self.at(&row, &e)
        )
    }
}

impl Smem {
    /// A `wgmma` shared-memory descriptor for this (swizzled) tile as an
    /// operand whose outer index (m or n) starts at `o0` and k at `k0`;
    /// `o_row`: the outer index runs along the tile's rows. Returns the
    /// descriptor and whether the operand is MN-major (the transpose
    /// immediate): K-major when k is the contiguous dim.
    fn desc(&self, o_row: bool, o0: &str, k0: &str) -> (String, bool) {
        let ny = if self.tm { self.e } else { self.rows };
        let b = if self.dynb { "ib * 2u + " } else { "" };
        if self.tm != o_row {
            (
                format!(
                    "tang_desc(sbase + {b}({}u + (({k0}) / 64u) * {}u + ({o0}) * 64u + ({k0}) % 64u) * 2u, 16u, 1024u)",
                    self.off,
                    ny * 64
                ),
                false,
            )
        } else {
            (
                format!(
                    "tang_desc(sbase + {b}({}u + (({o0}) / 64u) * {}u + ({k0}) * 64u) * 2u, {}u, 1024u)",
                    self.off,
                    ny * 64,
                    ny * 128
                ),
                true,
            )
        }
    }
}

/// `wgmma` m64nNk16 with A from registers (four bf16x2 fragments, the
/// `mma.sync` A layout per warp) and B from shared memory, accumulating.
fn wgmma_rs(n: usize, tb: bool) -> String {
    let regs = n / 2;
    let outs: Vec<String> = (0..regs).map(|i| format!("%{i}")).collect();
    let cons: Vec<String> = (0..regs).map(|i| format!("\"+f\"(d[{i}])")).collect();
    let tb = tb as u8;
    format!(
        "static __device__ __forceinline__ void tang_wgrs{n}_{tb}(float* d, const unsigned* a, unsigned long long b) {{\n    asm volatile(\"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{}}}, {{%{}, %{}, %{}, %{}}}, %{}, 1, 1, 1, {tb};\"\n        : {}\n        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(b));\n}}\n",
        outs.join(", "),
        regs,
        regs + 1,
        regs + 2,
        regs + 3,
        regs + 4,
        cons.join(", ")
    )
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

    /// A staged operand that is itself a materialized pointwise map of
    /// buffers (dO/l): evaluated while staging instead, from its own args.
    /// Returns the map's point coords over the kv space + lane (`rank`),
    /// the address the staging walks in (its largest arg's), and the
    /// buffers it reads. Opt-in (`TANG_ATTN_MAPSTAGE=1`): its element-wise
    /// loads are synchronous, so the kernel stalls on them (2.5–3× slower
    /// on an RTX 3090) where the map's own kernel costs ~0.1 ms.
    fn map_stage(&self, st: &Stage, red: usize) -> Option<(super::super::View, Addr, Vec<usize>)> {
        let rank = self.comp_at(red)?.rank();
        if std::env::var("TANG_ATTN_MAPSTAGE").ok().as_deref() != Some("1") {
            return None;
        }
        let m = st.buf;
        let mc = self.comp_at(m)?;
        if mc.monoid.is_some() || !mc.red.is_empty() || mc.tuple != 0 || mc.body_outs.len() != 1 {
            return None;
        }
        let shape = fixed(&self.nodes[m].shape);
        let mut strides = vec![1i64; shape.len()];
        for k in (0..shape.len().saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * shape[k + 1] as i64;
        }
        if st.addr.offset != 0 {
            return None;
        }
        let mut forms: Vec<Option<super::super::Form>> = vec![None; shape.len()];
        // the kv space's extent-1 dims are always 0
        let space = self.comp_at(red)?.space();
        for (dd, &c) in st.addr.coefs.iter().enumerate() {
            if c == 0 || (dd < rank && space[dd].fixed() == Some(1)) {
                continue;
            }
            let k = (0..shape.len())
                .find(|&k| shape[k] > 1 && strides[k] == c && forms[k].is_none())?;
            forms[k] = Some(super::super::Form::unit(rank + 1, dd));
        }
        let mut fs = Vec::new();
        for (k, f) in forms.into_iter().enumerate() {
            match f {
                Some(f) => fs.push(f),
                None if shape[k] == 1 => fs.push(super::super::Form {
                    offset: 0,
                    coefs: vec![0; rank + 1],
                }),
                None => return None,
            }
        }
        let view = super::super::View {
            guard: false,
            gather: None,
            forms: fs,
        };
        let mut bufs = Vec::new();
        let mut orient: Option<Addr> = None;
        for a in &mc.args {
            if a.view.guard || a.view.is_gather() {
                return None;
            }
            match &self.nodes[a.src.n()].op {
                Op::Iota(_) => {}
                _ => {
                    let stn = self.storage(a.src).n();
                    if self.nodes[stn].inline {
                        return None;
                    }
                    let addr = self.flat_addr(
                        &Arg {
                            src: a.src,
                            view: a.view.compose(&view, rank + 1),
                        },
                        rank + 1,
                    );
                    let nz = |x: &Addr| x.coefs.iter().filter(|&&c| c != 0).count();
                    if orient.as_ref().is_none_or(|o| nz(&addr) > nz(o)) {
                        orient = Some(addr);
                    }
                    if !bufs.contains(&stn) {
                        bufs.push(stn);
                    }
                }
            }
        }
        Some((view, orient?, bufs))
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
                let is_out = |j: usize| outputs.iter().any(|o| o.n() == j);
                let mut bf16_bufs = Vec::new();
                if std::env::var("TANG_ATTN_F32_OPS").is_err() {
                    for &n in &nested_nodes {
                        for a in &self.comp_at(n).unwrap().args {
                            let st = self.storage(a.src).n();
                            if matches!(self.nodes[st].op, Op::Comp(_))
                                && self.nodes[st].precision.storage == super::super::Format::F32
                                && !is_out(st)
                                && !bf16_bufs.contains(&st)
                            {
                                bf16_bufs.push(st);
                            }
                        }
                    }
                }
                let plan = AttnBwd {
                    root,
                    red,
                    dq,
                    dq_epi,
                    nested_nodes,
                    bf16_bufs,
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
        for &b in &p.bf16_bufs {
            self.nodes[b].precision.storage = super::super::Format::BF16;
        }
        for &n in &p.nested_nodes {
            self.nodes[n].inline = true;
            self.nodes[n].tile = (0, 0, 0);
            self.nodes[n].split = 0;
        }
        // the per-row reductions the kernel reads (rowsum(dO·O)): one
        // workgroup per row, its threads along the short reduce dim, so
        // the reads coalesce
        if std::env::var("TANG_ATTN_NOSPLIT").is_err() {
            let mut cand = self.kernel_reads(p.root);
            for j in cand.clone() {
                if self.comp_at(j).is_some_and(|c| c.monoid.is_none()) {
                    cand.extend(self.kernel_reads(j));
                }
            }
            for j in cand {
                let Some(c) = self.comp_at(j) else { continue };
                let short =
                    c.red.len() == 1 && c.red[0].fixed().is_some_and(|r| (32..=256).contains(&r));
                if short
                    && c.monoid.is_some()
                    && self.nodes[j].tile.0 == 0
                    && self.nodes[j].split == 0
                {
                    self.nodes[j].split = 32;
                }
            }
        }
    }

    /// Replace the kv tile's kernel with the fused one and the dQ
    /// reduction's with a zero fill launched before it.
    pub(crate) fn attn_bwd_patch(
        &self,
        p: &AttnBwd,
        program: &mut WgslProgram,
        variant: Option<usize>,
    ) -> bool {
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
        // staged maps (dO/l) are evaluated from their own buffers: those
        // join the bindings and the map's kernel goes away
        let Some(plan) = self.attn_bwd_build(p.root, p.red, p.dq, &reads) else {
            return false;
        };
        let mut reads = reads;
        let mut gone = Vec::new();
        for st in &plan.stages {
            if let Some((_, _, bufs)) = self.map_stage(st, p.red) {
                for b in bufs {
                    if !reads.contains(&b) {
                        reads.push(b);
                    }
                }
                gone.push(st.buf);
            }
        }
        let bj = BJS[variant.unwrap_or(0) % BJS.len()];
        // variants past MMA_SHAPES exist only on Hopper (227 KB of shared
        // memory per block); the rest must fit an sm_86's 99 KB
        let cap = match variant {
            Some(v) if v >= super::MMA_SHAPES.len() => 227 * 1024,
            _ => 99 * 1024,
        };
        // Hopper variants: every third one the wgmma kernel (two warpgroups
        // of 64 keys), the others mma.sync at 64 or 128 keys
        let wg = cap > 99 * 1024
            && std::env::var("TANG_ATTN_WG").ok().as_deref() != Some("0")
            && variant.is_some_and(|v| (v - super::MMA_SHAPES.len()).is_multiple_of(3));
        let bj = if wg { 128 } else { bj };
        let Some(src) = self
            .cuda_attn_bwd_kernel(p.root, p.red, p.dq, &reads, bj, cap, wg)
            .or_else(|| self.cuda_attn_bwd_kernel(p.root, p.red, p.dq, &reads, bj, cap, false))
            .or_else(|| self.cuda_attn_bwd_kernel(p.root, p.red, p.dq, &reads, BJS[0], cap, false))
        else {
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
        k.reads = reads.iter().map(|&j| TensorId(j as u32)).collect();
        k.reads.push(TensorId(p.dq as u32));
        let threads = super::src_define(&k.source, "TANG_THREADS");
        k.workgroup_size = threads as u32;
        let tiles = super::src_define(&k.source, "TANG_TILES");
        k.count = (tiles * threads) as u32;
        // the staged maps' own kernels, unless something else reads them
        for g in gone {
            let used = program
                .kernels
                .iter()
                .any(|k| k.node.index() != p.root && k.reads.iter().any(|r| r.index() == g));
            if !used {
                program.kernels.retain(|k| k.node.index() != g);
            }
        }
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
        if !nk.is_multiple_of(BJS[0]) || !nq.is_multiple_of(BI) || !(d == 64 || d == 128) {
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
            .map(|s| (if s.side_i { BI } else { BJS[0] }) * (s.e + PADH) * 2)
            .sum::<usize>()
            + BJS[0] * (BI + PADH) * 2;
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
        bj: usize,
        smem_cap: usize,
        wg: bool,
    ) -> Option<String> {
        let p = self.attn_bwd_build(root, red, dq, reads)?;
        if !p.nk.is_multiple_of(bj) || (wg && bj != 128) {
            return None;
        }
        let warps = bj / 16;
        let threads = warps * 32;
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
        let n_jb = p.nk / bj;
        let n_lead: usize = p.others.iter().map(|&o| p.pts[o]).product();
        let n_tiles = n_jb * n_lead;
        let w_out = c.out[c.out.len() - c.tuple..]
            .iter()
            .map(|x| x.n())
            .product::<usize>();
        // shared layout: stages in order (each stored with its global
        // contiguous dim innermost), then dS
        // key-side tiles, then the query-side ones (twice when the
        // chunks are double-buffered and it fits), then dS
        let mut tiles: Vec<Smem> = (0..p.stages.len())
            .map(|_| Smem {
                off: 0,
                dynb: false,
                tm: false,
                rows: 0,
                e: 0,
                sw: wg,
            })
            .collect();
        let mut off = 0usize;
        for side_i in [false, true] {
            for (t, st) in p.stages.iter().enumerate().filter(|x| x.1.side_i == side_i) {
                let (rows, row_dim) = if st.side_i { (BI, kdim) } else { (bj, dy) };
                tiles[t] = Smem {
                    off,
                    dynb: side_i,
                    tm: {
                        let a = self
                            .map_stage(st, p.red)
                            .map_or(st.addr.clone(), |(_, o, _)| o);
                        a.coefs[row_dim] == 1 && a.coefs[rank] != 1
                    },
                    rows,
                    e: st.e,
                    sw: wg,
                };
                off += tiles[t].halves();
            }
        }
        let i_region = off
            - tiles
                .iter()
                .filter(|t| !t.dynb)
                .map(Smem::halves)
                .sum::<usize>();
        let ds_h = if wg { bj * BI } else { bj * (BI + PADH) };
        // Hopper: dQ partials gathered in shared memory (f32, dQ's own
        // row layout) and added with one bulk async reduction per chunk
        let bulk = smem_cap > 99 * 1024
            && p.dq_addr.coefs[rank] == 1
            && std::env::var("TANG_ATTN_ATOMIC").is_err();
        // (wgmma: one partial per warpgroup, each over its own keys)
        let sq_h = if bulk {
            BI * d * 2 * if wg { 2 } else { 1 }
        } else {
            0
        };
        if wg && !bulk {
            return None;
        }
        let nbuf = if (off + i_region + ds_h + sq_h) * 2 <= smem_cap {
            2
        } else {
            1
        };
        let off = off + (nbuf - 1) * i_region;
        let ds = Smem {
            off,
            dynb: false,
            tm: false,
            rows: bj,
            e: BI,
            sw: wg,
        };
        let sq_off = off + ds_h;
        let smem_halves = sq_off + sq_h;
        if smem_halves * 2 + if wg { 1024 } else { 0 } > smem_cap {
            return None;
        }
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: attention backward (BF16 {}, fused dK/dV/dQ)",
            if wg { "wgmma" } else { "mma" }
        )
        .unwrap();
        writeln!(
            s,
            "// {bj} keys per block, {BI}-query chunks, {warps} warps, {nbuf} query buffers"
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        writeln!(
            s,
            "#define TANG_SMEM {}",
            smem_halves * 2 + if wg { 1024 } else { 0 }
        )
        .unwrap();
        writeln!(s, "#define select(f, t, c) ((c) ? (t) : (f))").unwrap();
        s.push_str(MMA_PRELUDE);
        s.push_str(HALF_PRELUDE);
        s.push_str(&mma16_prelude(super::super::Format::BF16));
        s.push_str(LDM_PRELUDE);
        if bulk {
            s.push_str(BULK_PRELUDE);
        }
        if wg {
            s.push_str(super::SM90_PRELUDE);
            for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
                s.push_str(&super::sm90_wgmma_ss("bf16", BI, ta, tb));
            }
            s.push_str(&wgmma_rs(d, false));
            s.push_str(&wgmma_rs(d, true));
        }
        let mut params: Vec<String> = reads
            .iter()
            .map(|&j| format!("const {}* __restrict__ b{j}", elem(j)))
            .collect();
        params.push(format!("const float* b{}", p.dq_node));
        params.push(format!("{}* __restrict__ out", elem(root)));
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    (void)count;").unwrap();
        writeln!(s, "    float* dqp = (float*)b{};", p.dq_node).unwrap();
        if wg {
            writeln!(
                s,
                "    extern __shared__ __align__(16) unsigned short smraw[];"
            )
            .unwrap();
            writeln!(s, "    unsigned short* smh = (unsigned short*)(((size_t)smraw + 1023) & ~(size_t)1023);").unwrap();
            writeln!(
                s,
                "    unsigned sbase = (unsigned)__cvta_generic_to_shared(smh);"
            )
            .unwrap();
        } else {
            writeln!(
                s,
                "    extern __shared__ __align__(16) unsigned short smh[];"
            )
            .unwrap();
        }
        writeln!(s, "    unsigned tile = blockIdx.x;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) return;").unwrap();
        writeln!(
            s,
            "    unsigned tid = threadIdx.x, warp = tid / 32u, lane = tid % 32u;"
        )
        .unwrap();
        writeln!(s, "    unsigned gq = lane / 4u, tq = lane % 4u;").unwrap();
        writeln!(s, "    unsigned j0 = (tile % {n_jb}u) * {bj}u;").unwrap();
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
        // stage `t` (rows from origin `r0`) into shared memory: 16-byte
        // cp.async along the stored-contiguous dim when it is bf16 and
        // aligned, else converted element by element
        let stage = |s: &mut String, t: usize, r0: &str| {
            let st = &p.stages[t];
            let tl = &tiles[t];
            let (rows, row_dim) = if st.side_i { (BI, kdim) } else { (bj, dy) };
            let cdim = if tl.tm { row_dim } else { rank };
            let (inner, outer) = if tl.tm { (rows, st.e) } else { (st.e, rows) };
            let vec8 = fmt(st.buf) == super::super::Format::BF16
                && st.addr.coefs[cdim] == 1
                && st.addr.offset % 8 == 0
                && inner.is_multiple_of(8)
                && st
                    .addr
                    .coefs
                    .iter()
                    .enumerate()
                    .all(|(dd, &c)| dd == cdim || c % 8 == 0);
            let src = |rr: &str, ee: &str| {
                aff(&st.addr, &|dd| {
                    coord(dd, &format!("{r0} + {rr}"), &format!("{r0} + {rr}"), ee)
                })
            };
            writeln!(s, "        {{").unwrap();
            if let Some((view, _, _)) = self.map_stage(st, p.red) {
                let mc = self.comp_at(st.buf).unwrap();
                writeln!(
                    s,
                    "        for (unsigned idx = tid; idx < {}u; idx += {threads}u) {{",
                    inner * outer
                )
                .unwrap();
                writeln!(
                    s,
                    "            unsigned io = idx / {inner}u, ii = idx % {inner}u;"
                )
                .unwrap();
                let (rr, ee) = if tl.tm { ("ii", "io") } else { ("io", "ii") };
                let cd =
                    |dd: usize| coord(dd, &format!("{r0} + {rr}"), &format!("{r0} + {rr}"), ee);
                let vals: Vec<String> = mc
                    .args
                    .iter()
                    .map(|a| {
                        let v = a.view.compose(&view, rank + 1);
                        match &self.nodes[a.src.n()].op {
                            Op::Iota(k) => {
                                let f = &v.forms[*k];
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
                                let ad = self.flat_addr(
                                    &Arg {
                                        src: a.src,
                                        view: v,
                                    },
                                    rank + 1,
                                );
                                ld(
                                    storage(a.src.n()),
                                    &format!("(unsigned)({})", aff(&ad, &cd)),
                                )
                            }
                        }
                    })
                    .collect();
                let mut body = String::new();
                let res = emit_scalar(
                    &mut body,
                    &mc.body,
                    &mc.body_outs,
                    &format!("s{t}_"),
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
                writeln!(
                    s,
                    "            smh[{}] = tang_f2b({});",
                    tl.at(rr, ee),
                    res[0]
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
            } else if vec8 {
                let per = inner / 8;
                writeln!(
                    s,
                    "        for (unsigned idx = tid; idx < {}u; idx += {threads}u) {{",
                    inner * outer / 8
                )
                .unwrap();
                writeln!(
                    s,
                    "            unsigned io = idx / {per}u, ii = (idx % {per}u) * 8u;"
                )
                .unwrap();
                let (rr, ee) = if tl.tm { ("ii", "io") } else { ("io", "ii") };
                writeln!(
                    s,
                    "            tang_cp16h(smh + {}, b{} + ({}), true);",
                    tl.at(rr, ee),
                    st.buf,
                    src(rr, ee)
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
            } else {
                writeln!(
                    s,
                    "        for (unsigned idx = tid; idx < {}u; idx += {threads}u) {{",
                    inner * outer
                )
                .unwrap();
                writeln!(
                    s,
                    "            unsigned io = idx / {inner}u, ii = idx % {inner}u;"
                )
                .unwrap();
                let (rr, ee) = if tl.tm { ("ii", "io") } else { ("io", "ii") };
                let v = ld(st.buf, &format!("(unsigned)({})", src(rr, ee)));
                writeln!(s, "            smh[{}] = tang_f2b({v});", tl.at(rr, ee)).unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        }}").unwrap();
        };
        // key-side stages, once
        for t in (0..p.stages.len()).filter(|&t| !p.stages[t].side_i) {
            stage(&mut s, t, "j0");
        }
        writeln!(s, "    tang_cp_commit();").unwrap();
        let n_out = p.outs.len();
        let nfd = d / 8;
        let nfi = BI / 8;
        writeln!(s, "    float acc[{n_out}][{nfd}][4];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int w = 0; w < {n_out}; w++) for (int f = 0; f < {nfd}; f++) for (int r = 0; r < 4; r++) acc[w][f][r] = 0.0f;").unwrap();
        if wg {
            // warpgroup g owns keys 64g…; warp w of it rows 16(w % 4)…
            writeln!(s, "    unsigned wgo = (warp / 4u) * 64u;").unwrap();
            writeln!(s, "    unsigned jw = wgo + (warp % 4u) * 16u;").unwrap();
        } else {
            writeln!(s, "    unsigned jw = warp * 16u;").unwrap();
        }
        // the chunk walk: chunks the mask zeroes entirely are skipped;
        // the next chunk's query-side stages load while this one computes
        // (double-buffered, offset `ib`) when they fit
        let mut skip = String::from("false");
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
            min.push_str(&lo_hi(dy, "j0", &format!("j0 + {}u", bj - 1)));
            min.push_str(&lo_hi(kdim, "i", &format!("i + {}u", BI - 1)));
            for &o in &p.others {
                if f.coefs[o] != 0 {
                    min.push_str(&format!(" + {} * (int)p{o}", f.coefs[o]));
                }
            }
            skip = format!("({min}) > 0");
        }
        let nq = p.nq;
        writeln!(s, "    auto next = [&](unsigned i) -> unsigned {{ while (i < {nq}u && ({skip})) i += {BI}u; return i; }};").unwrap();
        writeln!(s, "    unsigned ib = 0u;").unwrap();
        writeln!(s, "    unsigned i0 = next(0u);").unwrap();
        writeln!(s, "    if (i0 < {nq}u) {{").unwrap();
        for t in (0..p.stages.len()).filter(|&t| p.stages[t].side_i) {
            stage(&mut s, t, "i0");
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    tang_cp_commit();").unwrap();
        writeln!(s, "    while (i0 < {nq}u) {{").unwrap();
        writeln!(s, "        unsigned i1 = next(i0 + {BI}u);").unwrap();
        if nbuf == 2 {
            writeln!(s, "        ib ^= {i_region}u;").unwrap();
            writeln!(s, "        if (i1 < {nq}u) {{").unwrap();
            for t in (0..p.stages.len()).filter(|&t| p.stages[t].side_i) {
                stage(&mut s, t, "i1");
            }
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        ib ^= {i_region}u;").unwrap();
            writeln!(s, "        tang_cp_commit();").unwrap();
            writeln!(s, "        tang_cp_wait<1>();").unwrap();
        } else {
            writeln!(s, "        tang_cp_wait<0>();").unwrap();
        }
        if wg {
            writeln!(s, "        tang_fence_async();").unwrap();
        }
        writeln!(s, "        __syncthreads();").unwrap();
        // the nested contractions on the warp's 16 keys × the chunk
        // (wgmma: the warpgroup's 64 keys, from shared memory)
        if wg {
            for (q, _) in p.nested.iter().enumerate() {
                writeln!(s, "        float sa{q}[{nfi}][4];").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int f = 0; f < {nfi}; f++) for (int r = 0; r < 4; r++) sa{q}[f][r] = 0.0f;").unwrap();
            }
            writeln!(s, "        tang_wg_fence();").unwrap();
            for (q, ng) in p.nested.iter().enumerate() {
                let (tj, ti) = (&tiles[ng.j_stage], &tiles[ng.i_stage]);
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(
                    s,
                    "        for (unsigned ks = 0u; ks < {}u; ks++) {{",
                    ng.e / 16
                )
                .unwrap();
                let (da, ta) = tj.desc(true, "wgo", "ks * 16u");
                let (db, tb) = ti.desc(true, "0u", "ks * 16u");
                writeln!(
                    s,
                    "            tang_wg_ss{BI}_{}{}(&sa{q}[0][0], {da}, {db});",
                    ta as u8, tb as u8
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        tang_wg_commit();").unwrap();
            writeln!(s, "        tang_wg_wait<0>();").unwrap();
        }
        for (q, ng) in p.nested.iter().enumerate().filter(|_| !wg) {
            let (tj, ti) = (&tiles[ng.j_stage], &tiles[ng.i_stage]);
            writeln!(s, "        float sa{q}[{nfi}][4];").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int f = 0; f < {nfi}; f++) for (int r = 0; r < 4; r++) sa{q}[f][r] = 0.0f;").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int ks = 0; ks < {}; ks++) {{", ng.e / 16).unwrap();
            writeln!(s, "            unsigned fa[4];").unwrap();
            writeln!(
                s,
                "            {}",
                tj.ldm("fa", true, true, "jw", "ks * 16u")
            )
            .unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int f = 0; f < {nfi}; f += 2) {{").unwrap();
            writeln!(s, "                unsigned fb[4];").unwrap();
            writeln!(
                s,
                "                {}",
                ti.ldm("fb", false, true, "8u * f", "ks * 16u")
            )
            .unwrap();
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
            // leaves along the queries only are loaded once per column
            // (h), along the keys only once per row half (g), neither once
            let mut pre = String::new();
            let vals: Vec<String> = mc
                .args
                .iter()
                .enumerate()
                .map(|(ai, sa)| {
                    if let Some(q) = p.nested.iter().position(|n| n.node == sa.src.n()) {
                        let k = p.nested[q].coef;
                        return if k == 1.0 {
                            format!("sa{q}[f][r]")
                        } else {
                            format!("(sa{q}[f][r] * {})", f32_lit(k))
                        };
                    }
                    let view = sa.view.compose(&outer.view, rank);
                    let (addr, iota) = match &self.nodes[sa.src.n()].op {
                        Op::Iota(dd) => {
                            let f = &view.forms[*dd];
                            (
                                Addr {
                                    offset: f.offset,
                                    coefs: f.coefs.clone(),
                                },
                                true,
                            )
                        }
                        _ => (self.flat_addr(&Arg { src: sa.src, view }, rank), false),
                    };
                    let value = |j: &str, i: &str| -> String {
                        let e = aff(&addr, &|dd| coord(dd, j, i, "0"));
                        if iota {
                            format!("(float)({e})")
                        } else {
                            ld(storage(sa.src.n()), &format!("(unsigned)({e})"))
                        }
                    };
                    let (vj, vi) = (addr.coefs[dy] != 0, addr.coefs[kdim] != 0);
                    let name = format!("lv{w}_{ai}");
                    match (vj, vi) {
                        (false, true) => {
                            writeln!(pre, "        float {name}[{nfi}][2];").unwrap();
                            writeln!(pre, "        #pragma unroll").unwrap();
                            writeln!(pre, "        for (int f = 0; f < {nfi}; f++) for (int e = 0; e < 2; e++) {name}[f][e] = {};", value("0", "i0 + 8u * f + 2u * tq + e")).unwrap();
                            format!("{name}[f][r % 2]")
                        }
                        (true, false) => {
                            writeln!(pre, "        float {name}[2];").unwrap();
                            writeln!(pre, "        #pragma unroll").unwrap();
                            writeln!(pre, "        for (int hh = 0; hh < 2; hh++) {name}[hh] = {};", value("j0 + jw + gq + 8u * hh", "0")).unwrap();
                            format!("{name}[r / 2]")
                        }
                        (false, false) => {
                            writeln!(pre, "        float {name} = {};", value("0", "0")).unwrap();
                            name
                        }
                        (true, true) => value(jexpr, iexpr),
                    }
                })
                .collect();
            s.push_str(&pre);
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
        writeln!(
            s,
            "            *(unsigned*)(smh + {}) = tang_pack2(mv{dqw}[f][0], mv{dqw}[f][1]);",
            ds.at("jw + gq", "8u * f + 2u * tq")
        )
        .unwrap();
        writeln!(
            s,
            "            *(unsigned*)(smh + {}) = tang_pack2(mv{dqw}[f][2], mv{dqw}[f][3]);",
            ds.at("jw + gq + 8u", "8u * f + 2u * tq")
        )
        .unwrap();
        writeln!(s, "        }}").unwrap();
        // dV/dK += mapᵀ · B: the map's accumulator registers are the A
        // fragments (keys × queries), B from the query-side stage
        if wg {
            writeln!(s, "        tang_wg_fence();").unwrap();
            for (w, o) in p.outs.iter().enumerate() {
                let tb = &tiles[o.b_stage];
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int ks = 0; ks < {}; ks++) {{", BI / 16).unwrap();
                writeln!(s, "            unsigned fa[4] = {{tang_pack2(mv{w}[2 * ks][0], mv{w}[2 * ks][1]), tang_pack2(mv{w}[2 * ks][2], mv{w}[2 * ks][3]), tang_pack2(mv{w}[2 * ks + 1][0], mv{w}[2 * ks + 1][1]), tang_pack2(mv{w}[2 * ks + 1][2], mv{w}[2 * ks + 1][3])}};").unwrap();
                let (db, t) = tb.desc(false, "0u", "16u * (unsigned)ks");
                writeln!(
                    s,
                    "            tang_wgrs{d}_{}(&acc[{w}][0][0], fa, {db});",
                    t as u8
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        tang_wg_commit();").unwrap();
            writeln!(s, "        tang_wg_wait<0>();").unwrap();
        }
        for (w, o) in p.outs.iter().enumerate().filter(|_| !wg) {
            let tb = &tiles[o.b_stage];
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int ks = 0; ks < {}; ks++) {{", BI / 16).unwrap();
            writeln!(s, "            unsigned fa[4] = {{tang_pack2(mv{w}[2 * ks][0], mv{w}[2 * ks][1]), tang_pack2(mv{w}[2 * ks][2], mv{w}[2 * ks][3]), tang_pack2(mv{w}[2 * ks + 1][0], mv{w}[2 * ks + 1][1]), tang_pack2(mv{w}[2 * ks + 1][2], mv{w}[2 * ks + 1][3])}};").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int f = 0; f < {nfd}; f += 2) {{").unwrap();
            writeln!(s, "                unsigned fb[4];").unwrap();
            writeln!(
                s,
                "                {}",
                tb.ldm("fb", false, false, "8u * f", "16u * ks")
            )
            .unwrap();
            writeln!(s, "                tang_mma16(acc[{w}][f], fa, fb);").unwrap();
            writeln!(
                s,
                "                tang_mma16(acc[{w}][f + 1], fa, fb + 2);"
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        let leader = if wg { "tid % 128u == 0u" } else { "tid == 0u" };
        if bulk {
            writeln!(s, "        if ({leader}) tang_bulk_wait_read();").unwrap();
        }
        if wg {
            writeln!(s, "        tang_fence_async();").unwrap();
        }
        writeln!(s, "        __syncthreads();").unwrap();
        // wgmma: each warpgroup's dQ partial over its own 64 keys, 64
        // lanes at a time, into its own f32 block
        if wg {
            let qs = if p.dq_scale == 1.0 {
                String::new()
            } else {
                format!(" * {}f", f32_lit(p.dq_scale))
            };
            let tk = &tiles[p.dq_k];
            writeln!(
                s,
                "        float* sqw = (float*)(smh + {sq_off}u) + (warp / 4u) * {}u;",
                BI * d
            )
            .unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(
                s,
                "        for (unsigned c0 = 0u; c0 < {d}u; c0 += {BI}u) {{"
            )
            .unwrap();
            writeln!(s, "            float qa[{nfi}][4];").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int f = 0; f < {nfi}; f++) for (int r = 0; r < 4; r++) qa[f][r] = 0.0f;").unwrap();
            writeln!(s, "            tang_wg_fence();").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (unsigned ks = 0u; ks < 4u; ks++) {{").unwrap();
            let (da, ta) = ds.desc(false, "0u", "wgo + 16u * ks");
            let (db, tb) = tk.desc(false, "c0", "wgo + 16u * ks");
            writeln!(
                s,
                "                tang_wg_ss{BI}_{}{}(&qa[0][0], {da}, {db});",
                ta as u8, tb as u8
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "            tang_wg_commit();").unwrap();
            writeln!(s, "            tang_wg_wait<0>();").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int f = 0; f < {nfi}; f++)").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int r = 0; r < 4; r++) sqw[((warp % 4u) * 16u + gq + 8u * (r / 2)) * {d}u + c0 + 8u * f + 2u * tq + (r % 2)] = qa[f][r]{qs};").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        // the chunk's dQ partial: warp w's 16 queries × the lanes, over
        // the block's keys; A = dS from shared (transposed), B = K
        if !wg {
            let tk = &tiles[p.dq_k];
            let rgs = BI / 16;
            let ncg = (warps / rgs).max(1);
            let span = d / ncg;
            let cols = 64usize.min(span);
            let nfc = cols / 8;
            writeln!(s, "        unsigned iw = (warp % {rgs}u) * 16u;").unwrap();
            writeln!(s, "        unsigned cw = (warp / {rgs}u) * {span}u;").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            if std::env::var("TANG_ATTN_ABL")
                .unwrap_or_default()
                .contains("nodq")
            {
                writeln!(s, "        if (cw > 100000u)").unwrap();
            }
            writeln!(
                s,
                "        for (int c0 = cw; c0 < cw + {span}; c0 += {cols}) {{"
            )
            .unwrap();
            writeln!(s, "            float qa[{nfc}][4];").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int f = 0; f < {nfc}; f++) for (int r = 0; r < 4; r++) qa[f][r] = 0.0f;").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int ks = 0; ks < {}; ks++) {{", bj / 16).unwrap();
            writeln!(s, "                unsigned fa[4];").unwrap();
            writeln!(
                s,
                "                {}",
                ds.ldm("fa", true, false, "iw", "16u * ks")
            )
            .unwrap();
            writeln!(s, "                #pragma unroll").unwrap();
            writeln!(s, "                for (int f = 0; f < {nfc}; f += 2) {{").unwrap();
            writeln!(s, "                    unsigned fb[4];").unwrap();
            writeln!(
                s,
                "                    {}",
                tk.ldm("fb", false, false, "c0 + 8u * f", "16u * ks")
            )
            .unwrap();
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
            let abl = std::env::var("TANG_ATTN_ABL").unwrap_or_default();
            if bulk {
                writeln!(s, "                ((float*)(smh + {sq_off}u))[(iw + gq + 8u * (r / 2)) * {d}u + c0 + 8 * f + 2 * tq + (r % 2)] = qa[f][r]{qs};").unwrap();
            } else if abl.contains("noatomic") {
                writeln!(
                    s,
                    "                if (qa[f][r] == 12345.0f) dqp[0] = 1.0f;"
                )
                .unwrap();
            } else {
                writeln!(
                    s,
                    "                atomicAdd(dqp + ({qaddr}), qa[f][r]{qs});"
                )
                .unwrap();
            }
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        if bulk {
            writeln!(s, "        tang_bulk_fence();").unwrap();
            writeln!(s, "        __syncthreads();").unwrap();
            let row = |ii: &str| aff(&p.dq_addr, &|dd| coord(dd, "0", &format!("i0 + {ii}"), "0"));
            if wg {
                let src = format!("smh + {sq_off}u + (tid / 128u) * {}u", BI * d * 2);
                if p.dq_addr.coefs[kdim] == d as i64 {
                    writeln!(s, "        if ({leader}) {{ tang_bulk_add(dqp + ({}), {src}, {}u); tang_bulk_commit(); }}", row("0u"), BI * d * 4).unwrap();
                } else {
                    writeln!(s, "        if ({leader}) {{").unwrap();
                    writeln!(s, "            for (unsigned ii = 0u; ii < {BI}u; ii++) tang_bulk_add(dqp + ({}), {src} + ii * {}u, {}u);", row("ii"), 2 * d, d * 4).unwrap();
                    writeln!(s, "            tang_bulk_commit();").unwrap();
                    writeln!(s, "        }}").unwrap();
                }
            } else if p.dq_addr.coefs[kdim] == d as i64 {
                writeln!(s, "        if (tid == 0u) {{ tang_bulk_add(dqp + ({}), smh + {sq_off}u, {}u); tang_bulk_commit(); }}", row("0u"), BI * d * 4).unwrap();
            } else {
                writeln!(s, "        if (tid == 0u) {{").unwrap();
                writeln!(s, "            for (unsigned ii = 0u; ii < {BI}u; ii++) tang_bulk_add(dqp + ({}), smh + {sq_off}u + ii * {}u, {}u);", row("ii"), 2 * d, d * 4).unwrap();
                writeln!(s, "            tang_bulk_commit();").unwrap();
                writeln!(s, "        }}").unwrap();
            }
        }
        if nbuf == 2 {
            writeln!(s, "        ib ^= {i_region}u;").unwrap();
        } else {
            writeln!(s, "        __syncthreads();").unwrap();
            writeln!(s, "        if (i1 < {nq}u) {{").unwrap();
            for t in (0..p.stages.len()).filter(|&t| p.stages[t].side_i) {
                stage(&mut s, t, "i1");
            }
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        tang_cp_commit();").unwrap();
        }
        writeln!(s, "        i0 = i1;").unwrap();
        writeln!(s, "    }}").unwrap();
        if bulk {
            writeln!(s, "    if ({leader}) tang_bulk_wait();").unwrap();
        }
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

/// Hopper bulk async reduction (`cp.reduce.async.bulk`) of a shared
/// f32 block into global memory, and its fences and waits.
const BULK_PRELUDE: &str = r#"static __device__ __forceinline__ void tang_bulk_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
static __device__ __forceinline__ void tang_bulk_add(float* g, const unsigned short* sm, unsigned bytes) {
    unsigned a = (unsigned)__cvta_generic_to_shared(sm);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" :: "l"(g), "r"(a), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void tang_bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
static __device__ __forceinline__ void tang_bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
static __device__ __forceinline__ void tang_bulk_wait() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
"#;
