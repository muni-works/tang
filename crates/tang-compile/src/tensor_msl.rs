//! Lower a [`TensorGraph`] to Metal Shading Language.
//!
//! The scalar, split and tiled schedules share their emitters with WGSL:
//! the index algebra, staging layouts and epilogue fusion are produced
//! once (under `Dialect::Msl`, which switches buffer access to native
//! `half` / `float` elements) and the WGSL surface syntax is rewritten to
//! MSL here. MSL has no module-scope buffers, so every helper function
//! takes the kernel's buffers as a `Bufs` struct.
//!
//! On top of that, MSL gets a schedule WGSL cannot express: **fragment
//! lowering**. A contraction `C[m, n] = Σ_r A(m, r)·B(r, n)` whose
//! operands are stored in f16 and which accumulates in f32 runs on
//! `simdgroup_matrix` 8×8 fragments ([`TensorGraph::fragment_plan`]).
//! Operands are gathered through their views into threadgroup memory
//! (zero-padding guarded views and ragged edges at staging), so matmul,
//! batched matmul and conv-as-implicit-GEMM are one lowering. The result
//! tile goes through threadgroup memory, where an epilogue map is applied
//! before the write-out. The fragment geometry lives in
//! [`FragmentTarget`] so a CUDA `wmma` lowering can reuse the plan.

use std::fmt::Write;

use tang_expr::Node;

use super::wgsl::{
    arg_read, emit_decompose, emit_scalar_mapped, store, with_dialect, BufferSpec, Dialect,
};
use super::{fixed, tile_axes, Format, MonoidKind, Op, TensorGraph, TensorId};

/// Matrix-unit geometry of a target, for fragment lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragmentTarget {
    /// Fragment shape `(m, n, k)` of one multiply-accumulate.
    pub frag: (usize, usize, usize),
    /// Threads per SIMD group / warp.
    pub simd_width: usize,
    /// Threadgroup (shared) memory per threadgroup, bytes.
    pub threadgroup_mem: usize,
    /// Output tile per threadgroup `(m, n)` and reduce chunk staged per step.
    pub tile: (usize, usize, usize),
    /// SIMD groups per threadgroup along `(m, n)`.
    pub groups: (usize, usize),
}

impl FragmentTarget {
    /// Apple GPUs (M1+): `simdgroup_half8x8` / `simdgroup_float8x8`, 32 KiB
    /// threadgroup memory. 64×64 tiles over a 2×2 grid of SIMD groups, each
    /// owning 32×32 = 4×4 fragments (the spike's shape: 11 TFLOP/s at 1024³).
    pub const APPLE: FragmentTarget = FragmentTarget {
        frag: (8, 8, 8),
        simd_width: 32,
        threadgroup_mem: 32 * 1024,
        tile: (64, 64, 32),
        groups: (2, 2),
    };

    pub fn threads(&self) -> usize {
        self.groups.0 * self.groups.1 * self.simd_width
    }

    /// Threadgroup bytes of one fragment kernel: f16 operand tiles plus the
    /// f32 result tile used by the epilogue.
    pub fn staging_bytes(&self) -> usize {
        let (bm, bn, bk) = self.tile;
        2 * (bm * bk + bk * bn) + 4 * bm * bn
    }
}

/// How a comp maps onto fragments: which point dims are the GEMM's `m`
/// and `n`, which arg is `A(m, r)` and which is `B(r, n)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentPlan {
    /// The reduction (the root itself, or the reduction under an epilogue map).
    pub core: usize,
    /// Arg index of `A` (independent of `n`) and `B` (independent of `m`).
    pub a: usize,
    pub b: usize,
    /// Point dims mapped to `m` and `n`; every other point dim is a batch dim.
    pub dm: usize,
    pub dn: usize,
}

/// One Metal dispatch.
pub struct MslKernel {
    pub source: String,
    /// Kernel function name in `source`.
    pub entry: String,
    /// Node this kernel materializes.
    pub node: TensorId,
    /// Buffer-backed nodes read, in `[[buffer(i)]]` order; the output is
    /// bound next, then a `Params { points, 0, 0, 0 }` constant.
    pub reads: Vec<TensorId>,
    pub points: u32,
    /// Threadgroups along x and y.
    pub groups: (u32, u32),
    pub threads: u32,
    /// Lowered onto matrix fragments.
    pub fragment: bool,
    /// Threadgroup memory the kernel declares, bytes. A schedule whose
    /// staging does not fit the device (32 KiB on Apple GPUs) is not a
    /// valid candidate there.
    pub threadgroup_bytes: usize,
}

pub struct MslProgram {
    pub kernels: Vec<MslKernel>,
    pub buffers: Vec<BufferSpec>,
    pub aliases: Vec<(TensorId, TensorId)>,
}

impl MslProgram {
    /// The buffer-owning node for `t` (itself unless `t` is a reshape).
    pub fn storage(&self, t: TensorId) -> TensorId {
        self.aliases
            .iter()
            .find(|(r, _)| *r == t)
            .map_or(t, |(_, s)| *s)
    }
}

fn grid(groups: usize) -> (u32, u32) {
    let x = groups.clamp(1, 65535);
    (x as u32, groups.div_ceil(65535).max(1) as u32)
}

impl TensorGraph {
    /// Lower to MSL with the Apple fragment geometry.
    pub fn to_msl(&self, outputs: &[TensorId]) -> MslProgram {
        self.to_msl_for(outputs, &FragmentTarget::APPLE)
    }

    /// Lower to MSL. Same kernels and buffers as [`to_wgsl`](Self::to_wgsl);
    /// nodes scheduled `fragment` (and eligible, see
    /// [`fragment_plan`](Self::fragment_plan)) run on matrix fragments.
    ///
    /// Short contractions read only by tiles are nested in them, as in
    /// [`to_wgsl`](Self::to_wgsl).
    pub fn to_msl_for(&self, outputs: &[TensorId], target: &FragmentTarget) -> MslProgram {
        self.to_msl_with(outputs, target, true)
    }

    /// [`to_msl_for`](Self::to_msl_for), nesting short contractions or not.
    pub fn to_msl_with(
        &self,
        outputs: &[TensorId],
        target: &FragmentTarget,
        nested: bool,
    ) -> MslProgram {
        if !nested {
            return self.to_msl_scheduled(outputs, target);
        }
        let mut g = self.clone();
        g.schedule_nested_contractions_scalar(outputs);
        g.to_msl_scheduled(outputs, target)
    }

    fn to_msl_scheduled(&self, outputs: &[TensorId], target: &FragmentTarget) -> MslProgram {
        with_dialect(Dialect::Msl, || {
            let w = self.to_program_numbered(outputs, super::wgsl::Target::Wgsl);
            let kernels: Vec<MslKernel> = w
                .kernels
                .into_iter()
                .map(|k| {
                    let root = k.node.n();
                    let frag = self.nodes[root].fragment;
                    if let Some(plan) = frag.then(|| self.fragment_plan(k.node)).flatten() {
                        return self.fragment_kernel(root, &plan, target);
                    }
                    if let Some(plan) = frag.then(|| self.attention_plan(k.node)).flatten() {
                        return self.attention_kernel(root, &plan);
                    }
                    let reads: Vec<usize> = k.reads.iter().map(|t| t.n()).collect();
                    let groups = (k.count as usize).div_ceil(k.workgroup_size as usize);
                    let (source, threadgroup_bytes) = translate(self, &k.source, &reads, root);
                    MslKernel {
                        source,
                        threadgroup_bytes,
                        entry: "k_main".into(),
                        node: k.node,
                        reads: k.reads,
                        points: k.points,
                        groups: grid(groups),
                        threads: k.workgroup_size,
                        fragment: false,
                    }
                })
                .collect();
            let kernels = kernels
                .into_iter()
                .map(|mut k| {
                    k.source = super::wgsl::canonical_source(&k.source);
                    k
                })
                .collect();
            MslProgram {
                kernels,
                buffers: w.buffers,
                aliases: w.aliases,
            }
        })
    }

    /// Schedule: lower `t` onto matrix fragments where a backend has them.
    pub fn set_fragment(&mut self, t: TensorId, on: bool) {
        self.nodes[t.n()].fragment = on;
    }

    /// Whether `t` can run on f16 matrix fragments with an f32
    /// accumulator: a materialized sum-of-products `Σ_r A·B` (or an
    /// elementwise map over one, as epilogue) with two wide point dims,
    /// `A` independent of one and `B` of the other, both operands read
    /// straight from f16 buffers, and an f32 accumulator.
    pub fn fragment_plan(&self, t: TensorId) -> Option<FragmentPlan> {
        let root = t.n();
        let rc = self.comp_at(root)?;
        let core = if rc.monoid.is_none() {
            // fragments write the reduction's points: only an epilogue
            // that reads it point-for-point (identity view) fuses
            let r = self.tiled_reduction(root).filter(|&r| r != root)?;
            let k = rc.points().len();
            let read = self.epilogue_read(rc, r)?;
            if read != super::View::identity(k, k).forms || self.comp_at(r)?.out != rc.out {
                return None;
            }
            r
        } else {
            root
        };
        let c = self.comp_at(core)?;
        let m = c.monoid.as_ref()?;
        if m.kind() != MonoidKind::Sum || c.width() != 1 || c.tuple != 0 {
            return None;
        }
        if self.nodes[core].precision.accum != Format::F32 {
            return None;
        }
        let (u, v) = Self::product_operands(c)?;
        c.red.iter().try_for_each(|d| d.fixed().map(|_| ()))?;
        let pts: Vec<usize> = c
            .points()
            .iter()
            .map(|d| d.fixed())
            .collect::<Option<_>>()?;
        let (Some(dn), Some(dm)) = tile_axes(&pts) else {
            return None;
        };
        // operands: buffer-backed f16, or an inlined elementwise map
        // (materialized into threadgroup `half` while staging: the map runs
        // in f32 and its value is rounded to f16 there)
        for a in [&c.args[u], &c.args[v]] {
            if !(self.f16_buffer(a.src) || self.inline_map(a.src)) {
                return None;
            }
        }
        let dep = |i: usize, d: usize| c.args[i].view.depends_on(d);
        let (a, b) = if !dep(u, dn) && !dep(v, dm) {
            (u, v)
        } else if !dep(v, dn) && !dep(u, dm) {
            (v, u)
        } else {
            return None;
        };
        Some(FragmentPlan { core, a, b, dm, dn })
    }

    fn f16_buffer(&self, t: TensorId) -> bool {
        let s = self.storage(t).n();
        let backed = match &self.nodes[s].op {
            Op::Input(_) => true,
            Op::Comp(_) => !self.nodes[s].inline,
            _ => false,
        };
        backed && self.nodes[s].precision.storage == Format::F16
    }

    /// `t` is an inlined single-value map (no reduction).
    fn inline_map(&self, t: TensorId) -> bool {
        let n = &self.nodes[t.n()];
        n.inline
            && matches!(&n.op, Op::Comp(c) if c.monoid.is_none() && c.tuple == 0 && c.width() == 1)
    }

    /// Arg indices `(i, j)` of a body that is exactly `v_i · v_j`, `i ≠ j`.
    /// Other args may be present and unread (a gradient keeps the factor
    /// the product rule dropped as an argument).
    pub(crate) fn product_operands(c: &super::Comp) -> Option<(usize, usize)> {
        let nodes = c.body.nodes_slice();
        let (x, y) = match nodes[c.body_outs[0].index() as usize] {
            Node::Mul(x, y) => (x, y),
            _ => return None,
        };
        match (&nodes[x.index() as usize], &nodes[y.index() as usize]) {
            (Node::Var(i), Node::Var(j)) if i != j => Some((*i as usize, *j as usize)),
            _ => None,
        }
    }

    /// The contraction operands (storage buffers) `t` would read on
    /// fragments: the two factors of its core's `v_i · v_j` body.
    pub fn fragment_operands(&self, t: TensorId) -> Option<[TensorId; 2]> {
        let root = t.n();
        let rc = self.comp_at(root)?;
        let core = if rc.monoid.is_none() {
            self.tiled_reduction(root).filter(|&r| r != root)?
        } else {
            root
        };
        let c = self.comp_at(core)?;
        let (i, j) = Self::product_operands(c)?;
        Some([self.storage(c.args[i].src), self.storage(c.args[j].src)])
    }

    /// Whether every coordinate `arg` reads over the iteration box lies in
    /// its source: its guard never fires.
    fn view_in_bounds(&self, c: &super::Comp, which: usize) -> bool {
        let a = &c.args[which];
        let ext: Vec<i64> = c.space().iter().map(|d| d.n() as i64).collect();
        let shape = fixed(&self.nodes[a.src.n()].shape);
        a.view.forms.iter().zip(&shape).all(|(f, &n)| {
            let (mut lo, mut hi) = (f.offset, f.offset);
            for (k, e) in f.coefs.iter().zip(&ext) {
                let span = k * (e - 1);
                lo += span.min(0);
                hi += span.max(0);
            }
            lo >= 0 && hi < n as i64
        })
    }

    fn fragment_kernel(&self, root: usize, plan: &FragmentPlan, t: &FragmentTarget) -> MslKernel {
        let (pre, reads) = self.kernel_preamble(root);
        let c = self.comp_at(plan.core).unwrap();
        let pts = fixed(c.points());
        let red = fixed(&c.red);
        let kt: usize = red.iter().product();
        let (dm, dn) = (plan.dm, plan.dn);
        let (mm, nn) = (pts[dm], pts[dn]);
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dm && d != dn).collect();
        let win = self.window_operand(c, plan, &pts, t);
        // Gather staging walks the operand's contiguous device axis: k for
        // A (and m when A is stored transposed), n for B (and k when B is
        // stored transposed, e.g. a weight gradient's `x[c, t + d·k]`).
        // Walking the threadgroup tile's slow axis pads its rows by one
        // word so those stores do not conflict.
        let unit = |which: usize, d: usize| {
            self.affine_address(c, which)
                .is_some_and(|(_, co)| co[d].abs() == 1 && co.last().copied() != Some(1))
        };
        let a_mfast = unit(plan.a, dm);
        let b_kfast = self
            .affine_address(c, plan.b)
            .is_some_and(|(_, co)| co.last().copied() == Some(1) && co[dn].abs() != 1);
        let a_ld = if a_mfast { t.tile.2 + 2 } else { t.tile.2 };
        let b_ld = if b_kfast { t.tile.1 + 2 } else { t.tile.1 };
        let (bm, bn, bk) = t.tile;
        let (fm, fn_, fk) = t.frag;
        let (gm, gn) = t.groups;
        let threads = t.threads();
        let (sm, sn) = (bm / gm, bn / gn); // per-SIMD-group sub-tile
        let (am, an) = (sm / fm, sn / fn_); // fragments per SIMD group
        assert!(win.is_some() || t.staging_bytes() <= t.threadgroup_mem);
        let (tx, ty) = (nn.div_ceil(bn), mm.div_ceil(bm));
        let n_tiles = tx * ty * others.iter().map(|&d| pts[d]).product::<usize>();

        let mut s = pre;
        writeln!(s, "// fragment lowering: m = p{dm}, n = p{dn}, K = {kt}").unwrap();
        writeln!(s, "kernel void k_main(KERNEL_PARAMS uint3 wid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {{").unwrap();
        match &win {
            Some(w) => {
                // Cs aliases the operand staging: it is written after the
                // reduce loop's last barrier
                writeln!(s, "    threadgroup half smem[{}];", w.bytes(t) / 2).unwrap();
                writeln!(s, "    threadgroup half* As = smem;").unwrap();
                writeln!(
                    s,
                    "    threadgroup half* Xs = smem + {};",
                    w.nbuf * w.a_len(bm)
                )
                .unwrap();
                writeln!(s, "    threadgroup float* Cs = (threadgroup float*)smem;").unwrap();
            }
            None => {
                writeln!(s, "    threadgroup half As[{}];", bm * a_ld).unwrap();
                writeln!(s, "    threadgroup half Bs[{}];", bk * b_ld).unwrap();
            }
        }
        if win.is_none() {
            writeln!(s, "    threadgroup float Cs[{}];", bm * bn).unwrap();
        }
        writeln!(s, "    BUFS_INIT").unwrap();
        writeln!(s, "    const uint tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        let nlead: usize = others.iter().map(|&d| pts[d]).product();
        if win.is_some() {
            // batch dims fastest: neighbouring threadgroups (e.g. the
            // phases of a polyphase conv) read neighbouring weights
            writeln!(s, "    const uint tile_x = (tile / {nlead}u) % {tx}u;").unwrap();
            writeln!(s, "    const uint tile_y = tile / {}u;", nlead * tx).unwrap();
            writeln!(s, "    uint lead = tile % {nlead}u;").unwrap();
        } else {
            writeln!(s, "    const uint tile_x = tile % {tx}u;").unwrap();
            writeln!(s, "    const uint tile_y = (tile / {tx}u) % {ty}u;").unwrap();
            writeln!(s, "    uint lead = tile / {}u;", tx * ty).unwrap();
        }
        for &d in others.iter().rev() {
            writeln!(s, "    const uint p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    const uint m0 = tile_y * {bm}u;").unwrap();
        writeln!(s, "    const uint n0 = tile_x * {bn}u;").unwrap();
        writeln!(s, "    const uint sr = (sg / {gn}u) * {sm}u;").unwrap();
        writeln!(s, "    const uint sc = (sg % {gn}u) * {sn}u;").unwrap();
        writeln!(s, "    simdgroup_float8x8 acc[{am}][{an}];").unwrap();
        writeln!(s, "    for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);").unwrap();
        // Direct path: an operand whose address is affine with unit stride
        // along k or along its free axis is loaded straight from device
        // memory into fragments (no staging) on interior tiles.
        let direct = |which: usize, free: usize| -> Option<(usize, i64, i64, String)> {
            let a = &c.args[which];
            if !kt.is_multiple_of(fk) {
                return None;
            }
            let src = self.storage(a.src).n();
            let coef = self.affine_address(c, which)?;
            let k0 = pts.len();
            let sk = *coef.1.last().unwrap_or(&0);
            // the reduce dims must flatten row-major onto one stride
            let mut want = sk;
            for i in (0..red.len()).rev() {
                if coef.1[k0 + i] != want {
                    return None;
                }
                want *= red[i] as i64;
            }
            let sf = coef.1[free];
            if sk <= 0 || sf < 0 || !(sk == 1 || sf == 1) {
                return None;
            }
            let mut base = format!("{}", coef.0);
            for &d in &others {
                if coef.1[d] != 0 {
                    base.push_str(&format!(" + {} * int(p{d})", coef.1[d]));
                }
            }
            Some((src, sk, sf, base))
        };
        // a guarded operand loads directly only on chunks whose reads are
        // all in range (checked per chunk below)
        let da = direct(plan.a, dm);
        let db = direct(plan.b, dn);
        let (ga, gb) = (c.args[plan.a].view.guard, c.args[plan.b].view.guard);
        // fragment loads of a direct operand at reduce index `kexpr`
        let load_a = |k: &str| {
            let (ja, ska, sma, _) = da.as_ref().unwrap();
            let _ = ja;
            if *ska == 1 {
                format!("simdgroup_load(fa[i], pa + (sr + i * {fm}u) * {sma} + {k}, {sma})")
            } else {
                format!("simdgroup_load(fa[i], pa + (sr + i * {fm}u) + ({k}) * {ska}, {ska}, ulong2(0, 0), true)")
            }
        };
        let load_b = |k: &str| {
            let (_, skb, snb, _) = db.as_ref().unwrap();
            if *snb == 1 {
                format!("simdgroup_load(fb[j], pb + ({k}) * {skb} + sc + j * {fn_}u, {skb})")
            } else {
                format!("simdgroup_load(fb[j], pb + {k} + (sc + j * {fn_}u) * {snb}, {snb}, ulong2(0, 0), true)")
            }
        };
        // staging: A as [m][k], B as [k][n]; consecutive threads walk the
        // operand's fastest-moving tile axis
        let stage =
            |s: &mut String, which: usize, buf: &str, rows: usize, cols: usize, a_side: bool| {
                writeln!(
                    s,
                    "        for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
                    rows * cols
                )
                .unwrap();
                let (row_fast, ld) = if a_side {
                    (a_mfast, a_ld)
                } else {
                    (b_kfast, b_ld)
                };
                if row_fast {
                    writeln!(s, "            const uint ri = idx % {rows}u;").unwrap();
                    writeln!(s, "            const uint ci = idx / {rows}u;").unwrap();
                } else {
                    writeln!(s, "            const uint ri = idx / {cols}u;").unwrap();
                    writeln!(s, "            const uint ci = idx % {cols}u;").unwrap();
                }
                let (gpt, gk, ext) = if a_side {
                    ("m0 + ri", "k0 + ci", mm)
                } else {
                    ("n0 + ci", "k0 + ri", nn)
                };
                writeln!(s, "            const uint gp = {gpt};").unwrap();
                writeln!(s, "            const uint gk = {gk};").unwrap();
                writeln!(s, "            float v = 0.0f;").unwrap();
                writeln!(s, "            if (gp < {ext}u && gk < {kt}u) {{").unwrap();
                let (dp, dz) = if a_side { (dm, dn) } else { (dn, dm) };
                writeln!(s, "                const uint p{dp} = gp;").unwrap();
                writeln!(s, "                const uint p{dz} = 0u;").unwrap();
                let mut dec = String::new();
                emit_decompose(&mut dec, "gk", &red, "r");
                for line in dec.lines() {
                    writeln!(s, "            {line}").unwrap();
                }
                writeln!(
                    s,
                    "                v = {};",
                    arg_read(self, c, &c.args[which])
                )
                .unwrap();
                writeln!(s, "            }}").unwrap();
                writeln!(s, "            {buf}[ri * {ld}u + ci] = half(v);").unwrap();
                writeln!(s, "        }}").unwrap();
            };
        // One reduce loop, each operand either loaded straight from device
        // (`a_dev` / `b_dev`) or gathered into threadgroup memory.
        let emit_loop = |s: &mut String, a_dev: bool, b_dev: bool| {
            if a_dev {
                let (ja, _, sma, basea) = da.as_ref().unwrap();
                writeln!(
                    s,
                    "    device const half* pa = b{ja} + ({basea}) + int(m0) * {sma};"
                )
                .unwrap();
            }
            if b_dev {
                let (jb, _, snb, baseb) = db.as_ref().unwrap();
                writeln!(
                    s,
                    "    device const half* pb = b{jb} + ({baseb}) + int(n0) * {snb};"
                )
                .unwrap();
            }
            if a_dev && b_dev {
                writeln!(s, "    for (uint k = 0u; k < {kt}u; k += {fk}u) {{").unwrap();
                writeln!(s, "        simdgroup_half8x8 fa[{am}];").unwrap();
                writeln!(s, "        simdgroup_half8x8 fb[{an}];").unwrap();
                writeln!(
                    s,
                    "        for (uint i = 0u; i < {am}u; i++) {};",
                    load_a("k")
                )
                .unwrap();
                writeln!(
                    s,
                    "        for (uint j = 0u; j < {an}u; j++) {};",
                    load_b("k")
                )
                .unwrap();
                writeln!(s, "        for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
                writeln!(s, "    }}").unwrap();
                return;
            }
            writeln!(s, "    for (uint k0 = 0u; k0 < {kt}u; k0 += {bk}u) {{").unwrap();
            if !a_dev {
                stage(s, plan.a, "As", bm, bk, true);
            }
            if !b_dev {
                stage(s, plan.b, "Bs", bk, bn, false);
            }
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "        for (uint kk = 0u; kk < {bk}u; kk += {fk}u) {{").unwrap();
            writeln!(s, "            simdgroup_half8x8 fa[{am}];").unwrap();
            writeln!(s, "            simdgroup_half8x8 fb[{an}];").unwrap();
            if a_dev {
                writeln!(
                    s,
                    "            for (uint i = 0u; i < {am}u; i++) {};",
                    load_a("k0 + kk")
                )
                .unwrap();
            } else {
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) simdgroup_load(fa[i], &As[(sr + i * {fm}u) * {a_ld}u + kk], {a_ld});").unwrap();
            }
            if b_dev {
                writeln!(
                    s,
                    "            for (uint j = 0u; j < {an}u; j++) {};",
                    load_b("k0 + kk")
                )
                .unwrap();
            } else {
                writeln!(s, "            for (uint j = 0u; j < {an}u; j++) simdgroup_load(fb[j], &Bs[kk * {b_ld}u + sc + j * {fn_}u], {b_ld});").unwrap();
            }
            writeln!(s, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        };
        // A direct operand alongside a staged one reads whole chunks, so
        // the reduce extent must be a multiple of the chunk.
        let hybrid = kt.is_multiple_of(bk);
        let m_int = format!("m0 + {bm}u <= {mm}u");
        let n_int = format!("n0 + {bn}u <= {nn}u");
        let fast = match (da.is_some() && !ga, db.is_some() && !gb) {
            (true, true) => Some((format!("{m_int} && {n_int}"), true, true)),
            (true, false) if hybrid => Some((m_int.clone(), true, false)),
            (false, true) if hybrid => Some((n_int.clone(), false, true)),
            _ => None,
        };
        // per-chunk direct: both operands affine, one or both guarded, one
        // reduce dim (so a chunk is one interval of it)
        let chunked = win.is_none() && red.len() == 1 && da.is_some() && db.is_some() && (ga || gb);
        match (&win, fast) {
            (Some(w), _) => self.emit_window_loop(&mut s, c, plan, &pts, w, t),
            _ if chunked => {
                let k0 = pts.len();
                // in-range test of a guarded operand over one chunk
                let inb = |which: usize, free: usize, origin: &str, ext: usize| -> Vec<String> {
                    let a = &c.args[which];
                    if !a.view.guard {
                        return Vec::new();
                    }
                    let shape = fixed(&self.nodes[a.src.n()].shape);
                    a.view
                        .forms
                        .iter()
                        .zip(&shape)
                        .map(|(f, &n)| {
                            let mut e = format!("{}", f.offset);
                            for &d in &others {
                                if f.coefs[d] != 0 {
                                    e.push_str(&format!(" + {} * int(p{d})", f.coefs[d]));
                                }
                            }
                            let (cf, ck) = (f.coefs[free], f.coefs[k0]);
                            e.push_str(&format!(" + {cf} * int({origin}) + {ck} * int(k0)"));
                            let lo = cf.min(0) * (ext as i64 - 1) + ck.min(0) * (bk as i64 - 1);
                            let hi = cf.max(0) * (ext as i64 - 1) + ck.max(0) * (bk as i64 - 1);
                            format!("({e}) + {lo} >= 0 && ({e}) + {hi} < {n}")
                        })
                        .collect()
                };
                let mut conds = vec![
                    format!("{m_int} && {n_int}"),
                    format!("k0 + {bk}u <= {kt}u"),
                ];
                conds.extend(inb(plan.a, dm, "m0", bm));
                conds.extend(inb(plan.b, dn, "n0", bn));
                let (ja, _, sma, basea) = da.as_ref().unwrap();
                let (jb, _, snb, baseb) = db.as_ref().unwrap();
                writeln!(
                    s,
                    "    device const half* pa = b{ja} + ({basea}) + int(m0) * {sma};"
                )
                .unwrap();
                writeln!(
                    s,
                    "    device const half* pb = b{jb} + ({baseb}) + int(n0) * {snb};"
                )
                .unwrap();
                writeln!(s, "    for (uint k0 = 0u; k0 < {kt}u; k0 += {bk}u) {{").unwrap();
                writeln!(s, "        if ({}) {{", conds.join(" && ")).unwrap();
                writeln!(s, "        for (uint kk = 0u; kk < {bk}u; kk += {fk}u) {{").unwrap();
                writeln!(s, "            simdgroup_half8x8 fa[{am}];").unwrap();
                writeln!(s, "            simdgroup_half8x8 fb[{an}];").unwrap();
                writeln!(
                    s,
                    "            for (uint i = 0u; i < {am}u; i++) {};",
                    load_a("k0 + kk")
                )
                .unwrap();
                writeln!(
                    s,
                    "            for (uint j = 0u; j < {an}u; j++) {};",
                    load_b("k0 + kk")
                )
                .unwrap();
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
                writeln!(s, "        }}").unwrap();
                writeln!(s, "        continue;").unwrap();
                writeln!(s, "        }}").unwrap();
                stage(&mut s, plan.a, "As", bm, bk, true);
                stage(&mut s, plan.b, "Bs", bk, bn, false);
                writeln!(
                    s,
                    "        threadgroup_barrier(mem_flags::mem_threadgroup);"
                )
                .unwrap();
                writeln!(s, "        for (uint kk = 0u; kk < {bk}u; kk += {fk}u) {{").unwrap();
                writeln!(s, "            simdgroup_half8x8 fa[{am}];").unwrap();
                writeln!(s, "            simdgroup_half8x8 fb[{an}];").unwrap();
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) simdgroup_load(fa[i], &As[(sr + i * {fm}u) * {a_ld}u + kk], {a_ld});").unwrap();
                writeln!(s, "            for (uint j = 0u; j < {an}u; j++) simdgroup_load(fb[j], &Bs[kk * {b_ld}u + sc + j * {fn_}u], {b_ld});").unwrap();
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
                writeln!(s, "        }}").unwrap();
                writeln!(
                    s,
                    "        threadgroup_barrier(mem_flags::mem_threadgroup);"
                )
                .unwrap();
                writeln!(s, "    }}").unwrap();
            }
            (None, Some((cond, a_dev, b_dev))) => {
                writeln!(s, "    if ({cond}) {{").unwrap();
                emit_loop(&mut s, a_dev, b_dev);
                writeln!(s, "    }} else {{").unwrap();
                emit_loop(&mut s, false, false);
                writeln!(s, "    }}").unwrap();
            }
            (None, None) => emit_loop(&mut s, false, false),
        }
        writeln!(s, "    for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_store(acc[i][j], &Cs[(sr + i * {fm}u) * {bn}u + sc + j * {fn_}u], {bn});").unwrap();
        writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        // write-out (and epilogue) from the result tile
        writeln!(
            s,
            "    for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
            bm * bn
        )
        .unwrap();
        writeln!(s, "        const uint gm = m0 + idx / {bn}u;").unwrap();
        writeln!(s, "        const uint gn = n0 + idx % {bn}u;").unwrap();
        writeln!(s, "        if (gm < {mm}u && gn < {nn}u) {{").unwrap();
        writeln!(s, "        const uint p{dm} = gm;").unwrap();
        writeln!(s, "        const uint p{dn} = gn;").unwrap();
        writeln!(s, "        const float cval = Cs[idx];").unwrap();
        let mut point_flat = String::from("0u");
        for (d, n) in pts.iter().enumerate() {
            point_flat = format!("({point_flat}) * {n}u + p{d}");
        }
        writeln!(s, "        const uint g = {point_flat};").unwrap();
        let out_f = self.nodes[root].precision.storage;
        if root == plan.core {
            writeln!(s, "        {}", store(out_f, "g", "cval")).unwrap();
        } else {
            let mc = self.comp_at(root).unwrap();
            let args: Vec<String> = mc
                .args
                .iter()
                .map(|a| {
                    if a.src.n() == plan.core {
                        "cval".to_string()
                    } else {
                        arg_read(self, mc, a)
                    }
                })
                .collect();
            let vals =
                emit_scalar_mapped(&mut s, "        ", &mc.body, &mc.body_outs, "e_", &|v| {
                    args[v as usize].clone()
                });
            let mw = mc.width();
            for (slot, v) in vals.iter().enumerate() {
                writeln!(
                    s,
                    "        {}",
                    store(out_f, &format!("g * {mw}u + {slot}u"), v)
                )
                .unwrap();
            }
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        MslKernel {
            source: translate(self, &s, &reads, root).0,
            threadgroup_bytes: win
                .as_ref()
                .map_or(2 * (bm * a_ld + bk * b_ld) + 4 * bm * bn, |w| w.bytes(t)),
            entry: "k_main".into(),
            node: TensorId(root as u32),
            reads: reads.iter().map(|&j| TensorId(j as u32)).collect(),
            points: pts.iter().product::<usize>() as u32,
            groups: grid(n_tiles),
            threads: threads as u32,
            fragment: true,
        }
    }
}

/// A sliding-window operand of a fragment contraction (conv-shaped `B`):
/// its view reads one "row" reduce dim (channels) through forms of its
/// own, and the free dim `n` plus every other reduce dim ("taps") through
/// one "time" form `st·n + Σ coef·tap + …`. One zero-padded window per
/// (row chunk, n tile) is staged in threadgroup memory, de-interleaved by
/// phase `u mod st`, and each tap's `B` fragments load from it at a
/// shifted offset: the taps reuse one staged window instead of each
/// re-gathering its own copy (implicit-GEMM gather).
///
/// `A` is staged once per chunk for every tap, as tap planes
/// `[tap][m][c]`; with an affine, unguarded `A` (a weight) each thread
/// copies all taps of one `(m, c)` with constant offsets. Staging is
/// double-buffered when two copies fit: one barrier per chunk.
#[derive(Clone, Debug)]
pub(crate) struct Window {
    src: usize,
    shape: Vec<usize>,
    forms: Vec<super::Form>,
    tform: usize,
    row: usize,
    taps: Vec<usize>,
    st: i64,
    minoff: i64,
    span: i64,
    /// Row-dim chunk staged per step.
    pub cb: usize,
    /// Tap combinations (product of tap extents).
    ntaps: usize,
    /// Staging buffers (2: double-buffered).
    nbuf: usize,
}

impl Window {
    fn a_len(&self, bm: usize) -> usize {
        bm * self.cb * self.ntaps
    }
    fn wp(&self, bn: usize) -> usize {
        bn + (self.span / self.st) as usize + 1
    }
    fn row_stride(&self, bn: usize) -> usize {
        self.st as usize * self.wp(bn)
    }
    fn x_len(&self, bn: usize) -> usize {
        self.cb * self.row_stride(bn)
    }
    /// Threadgroup bytes: the staging buffers, which the f32 result tile
    /// aliases after the reduce loop.
    fn bytes(&self, t: &FragmentTarget) -> usize {
        let (bm, bn, _) = t.tile;
        (2 * self.nbuf * (self.a_len(bm) + self.x_len(bn))).max(4 * bm * bn)
    }
}

impl TensorGraph {
    /// The windowed reading of `B`, if its view is conv-shaped (see
    /// [`Window`]) and its staging fits the target. `TANG_MSL_NO_WINDOW`
    /// disables it (A/B against the implicit-GEMM gather).
    pub(crate) fn window_operand(
        &self,
        c: &super::Comp,
        plan: &FragmentPlan,
        pts: &[usize],
        t: &FragmentTarget,
    ) -> Option<Window> {
        if std::env::var_os("TANG_MSL_NO_WINDOW").is_some() {
            return None;
        }
        let arg = &c.args[plan.b];
        if !self.f16_buffer(arg.src) {
            return None;
        }
        let k0 = pts.len();
        let red = fixed(&c.red);
        let (dm, dn) = (plan.dm, plan.dn);
        let forms = &arg.view.forms;
        let tform = forms.iter().position(|f| f.coefs[dn] != 0)?;
        let tf = &forms[tform];
        let st = tf.coefs[dn];
        if st <= 0 || tf.coefs[dm] != 0 {
            return None;
        }
        let mut row = None;
        let mut taps = Vec::new();
        for r in 0..red.len() {
            let in_t = tf.coefs[k0 + r] != 0;
            let elsewhere = forms
                .iter()
                .enumerate()
                .any(|(i, f)| i != tform && f.coefs[k0 + r] != 0);
            match (in_t, elsewhere) {
                (true, true) => return None,
                (false, true) => {
                    if row.replace(r).is_some() {
                        return None;
                    }
                }
                _ => taps.push(r),
            }
        }
        let row = row?;
        if !taps.iter().any(|&r| tf.coefs[k0 + r] != 0) {
            return None;
        }
        for (i, f) in forms.iter().enumerate() {
            if i != tform && (f.coefs[dn] != 0 || f.coefs[dm] != 0) {
                return None;
            }
        }
        let (mut minoff, mut maxoff) = (0i64, 0i64);
        for &r in &taps {
            let e = tf.coefs[k0 + r] * (red[r] as i64 - 1);
            minoff += e.min(0);
            maxoff += e.max(0);
        }
        let mut w = Window {
            src: self.storage(arg.src).n(),
            shape: fixed(&self.nodes[arg.src.n()].shape),
            forms: forms.clone(),
            tform,
            row,
            ntaps: taps.iter().map(|&r| red[r]).product(),
            taps,
            st,
            minoff,
            span: maxoff - minoff,
            cb: 0,
            nbuf: 0,
        };
        // 16- and 8-channel chunks measured equal on op (a) (0.69 / 0.70
        // ms); double buffering (one barrier per chunk) measured slower
        // (0.74), so it is opt-in: `TANG_MSL_WINDOW_BUFFERS=2`
        let rows = red[row].next_multiple_of(8);
        let nbuf: usize = std::env::var("TANG_MSL_WINDOW_BUFFERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .clamp(1, 2);
        [16usize, 8].into_iter().find_map(|cb| {
            w.cb = cb.min(rows);
            w.nbuf = nbuf;
            (w.bytes(t) <= t.threadgroup_mem).then(|| w.clone())
        })
    }

    fn emit_window_loop(
        &self,
        s: &mut String,
        c: &super::Comp,
        plan: &FragmentPlan,
        pts: &[usize],
        w: &Window,
        t: &FragmentTarget,
    ) {
        let red = fixed(&c.red);
        let k0 = pts.len();
        let (dm, dn) = (plan.dm, plan.dn);
        let mm = pts[dm];
        let (bm, bn, _) = t.tile;
        let (fm, fn_, _) = t.frag;
        let threads = t.threads();
        let (am, an) = (bm / t.groups.0 / fm, bn / t.groups.1 / fn_);
        let (cb, st, ntaps) = (w.cb, w.st, w.ntaps);
        let wp = w.wp(bn);
        let rs = w.row_stride(bn);
        let (al, xl) = (w.a_len(bm), w.x_len(bn));
        let uw = st as usize * (bn - 1) + w.span as usize + 1;
        let crow = red[w.row];
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dm && d != dn).collect();
        let tf = &w.forms[w.tform];
        // tap ti's values of the tap dims
        let tap_vals = |ti: usize| {
            let mut rem = ti;
            let mut vals = vec![0i64; red.len()];
            for &r in w.taps.iter().rev() {
                vals[r] = (rem % red[r]) as i64;
                rem /= red[r];
            }
            vals
        };

        // ---- staging of chunk `c0` into `Ad` / `Xd`
        let mut stage = String::new();
        let st_ = &mut stage;
        // the window
        writeln!(
            st_,
            "        for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
            cb * uw
        )
        .unwrap();
        writeln!(st_, "            const uint cl = idx / {uw}u;").unwrap();
        writeln!(st_, "            const uint u = idx % {uw}u;").unwrap();
        writeln!(st_, "            const uint gc = c0 + cl;").unwrap();
        writeln!(st_, "            half v = half(0.0f);").unwrap();
        writeln!(st_, "            if (gc < {crow}u) {{").unwrap();
        let mut conds = Vec::new();
        for (i, f) in w.forms.iter().enumerate() {
            let mut e = format!("{}", f.offset);
            for &d in &others {
                if f.coefs[d] != 0 {
                    e.push_str(&format!(" + {} * int(p{d})", f.coefs[d]));
                }
            }
            if i == w.tform {
                e.push_str(&format!(" + {st} * int(n0) + {} + int(u)", w.minoff));
            } else if f.coefs[k0 + w.row] != 0 {
                e.push_str(&format!(" + {} * int(gc)", f.coefs[k0 + w.row]));
            }
            writeln!(st_, "                const int x{i} = {e};").unwrap();
            conds.push(format!("x{i} >= 0 && x{i} < {}", w.shape[i]));
        }
        let mut terms = Vec::new();
        let mut stride = 1i64;
        for i in (0..w.forms.len()).rev() {
            terms.push(format!("x{i} * {stride}"));
            stride *= w.shape[i] as i64;
        }
        let addr = if terms.is_empty() {
            "0".into()
        } else {
            terms.join(" + ")
        };
        // every coordinate is bounds-checked, guarded view or not: the
        // window overhangs the tile
        writeln!(
            st_,
            "                if ({}) {{ v = b{}[{addr}]; }}",
            conds.join(" && "),
            w.src
        )
        .unwrap();
        writeln!(st_, "            }}").unwrap();
        writeln!(
            st_,
            "            Xd[cl * {rs}u + (u % {st}u) * {wp}u + u / {st}u] = v;"
        )
        .unwrap();
        writeln!(st_, "        }}").unwrap();
        // A, every tap, as planes Ad[tap][m][c]
        let affine = (!c.args[plan.a].view.guard || self.view_in_bounds(c, plan.a))
            .then(|| self.affine_address(c, plan.a))
            .flatten()
            .filter(|(_, co)| co[dn] == 0);
        writeln!(
            st_,
            "        for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
            bm * cb
        )
        .unwrap();
        writeln!(st_, "            const uint mi = idx / {cb}u;").unwrap();
        writeln!(st_, "            const uint cl = idx % {cb}u;").unwrap();
        writeln!(
            st_,
            "            const bool ok = m0 + mi < {mm}u && c0 + cl < {crow}u;"
        )
        .unwrap();
        match affine {
            Some((off, co)) => {
                // one thread per (m, c), every tap at a constant offset;
                // consecutive threads walk c, so the plane stores never
                // conflict
                let ja = self.storage(c.args[plan.a].src).n();
                let mut base = format!("{off}");
                for &d in &others {
                    if co[d] != 0 {
                        base.push_str(&format!(" + {} * int(p{d})", co[d]));
                    }
                }
                writeln!(
                    st_,
                    "            device const half* q = b{ja} + ({base}) + int(m0 + mi) * {} + int(c0 + cl) * {};",
                    co[dm],
                    co[k0 + w.row]
                )
                .unwrap();
                for ti in 0..ntaps {
                    let vals = tap_vals(ti);
                    let o: i64 = w.taps.iter().map(|&r| co[k0 + r] * vals[r]).sum();
                    writeln!(
                        st_,
                        "            Ad[{}u + idx] = ok ? q[{o}] : half(0.0f);",
                        ti * bm * cb
                    )
                    .unwrap();
                }
            }
            None => {
                writeln!(st_, "            if (ok) {{").unwrap();
                writeln!(st_, "                const uint p{dm} = m0 + mi;").unwrap();
                writeln!(st_, "                const uint p{dn} = 0u;").unwrap();
                writeln!(st_, "                const uint r{} = c0 + cl;", w.row).unwrap();
                for ti in 0..ntaps {
                    let vals = tap_vals(ti);
                    writeln!(st_, "                {{").unwrap();
                    for &r in &w.taps {
                        writeln!(st_, "                const uint r{r} = {}u;", vals[r]).unwrap();
                    }
                    writeln!(
                        st_,
                        "                Ad[{}u + idx] = half({});",
                        ti * bm * cb,
                        arg_read(self, c, &c.args[plan.a])
                    )
                    .unwrap();
                    writeln!(st_, "                }}").unwrap();
                }
                writeln!(st_, "            }} else {{").unwrap();
                for ti in 0..ntaps {
                    writeln!(
                        st_,
                        "                Ad[{}u + idx] = half(0.0f);",
                        ti * bm * cb
                    )
                    .unwrap();
                }
                writeln!(st_, "            }}").unwrap();
            }
        }
        writeln!(st_, "        }}").unwrap();

        // ---- fragments of the staged chunk in `Ac` / `Xc`
        let mut comp = String::new();
        let cm = &mut comp;
        writeln!(cm, "        for (uint kk = 0u; kk < {cb}u; kk += 8u) {{").unwrap();
        writeln!(cm, "            simdgroup_half8x8 fa[{am}];").unwrap();
        writeln!(cm, "            simdgroup_half8x8 fb[{an}];").unwrap();
        for ti in 0..ntaps {
            let vals = tap_vals(ti);
            let off: i64 = w
                .taps
                .iter()
                .map(|&r| tf.coefs[k0 + r] * vals[r])
                .sum::<i64>()
                - w.minoff;
            let boff = (off % st) as usize * wp + (off / st) as usize;
            writeln!(cm, "            for (uint i = 0u; i < {am}u; i++) simdgroup_load(fa[i], &Ac[{}u + (sr + i * {fm}u) * {cb}u + kk], {cb});", ti * bm * cb).unwrap();
            writeln!(cm, "            for (uint j = 0u; j < {an}u; j++) simdgroup_load(fb[j], &Xc[kk * {rs}u + {boff}u + sc + j * {fn_}u], {rs});").unwrap();
            writeln!(cm, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
        }
        writeln!(cm, "        }}").unwrap();

        writeln!(s, "    // windowed operand: rows r{}, taps {:?}, stride {st}, window {uw}, chunk {cb}, {} buffer(s)", w.row, w.taps, w.nbuf).unwrap();
        if w.nbuf == 2 {
            writeln!(s, "    {{").unwrap();
            writeln!(s, "        const uint c0 = 0u;").unwrap();
            writeln!(s, "        threadgroup half* Ad = As;").unwrap();
            writeln!(s, "        threadgroup half* Xd = Xs;").unwrap();
            s.push_str(&stage);
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
            writeln!(
                s,
                "    for (uint cc = 0u, buf = 0u; cc < {crow}u; cc += {cb}u, buf ^= 1u) {{"
            )
            .unwrap();
            writeln!(s, "        if (cc + {cb}u < {crow}u) {{").unwrap();
            writeln!(s, "        const uint c0 = cc + {cb}u;").unwrap();
            writeln!(s, "        threadgroup half* Ad = As + (buf ^ 1u) * {al}u;").unwrap();
            writeln!(s, "        threadgroup half* Xd = Xs + (buf ^ 1u) * {xl}u;").unwrap();
            s.push_str(&stage);
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        threadgroup const half* Ac = As + buf * {al}u;").unwrap();
            writeln!(s, "        threadgroup const half* Xc = Xs + buf * {xl}u;").unwrap();
            s.push_str(&comp);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(s, "    for (uint c0 = 0u; c0 < {crow}u; c0 += {cb}u) {{").unwrap();
            writeln!(s, "        threadgroup half* Ad = As;").unwrap();
            writeln!(s, "        threadgroup half* Xd = Xs;").unwrap();
            s.push_str(&stage);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "        threadgroup const half* Ac = As;").unwrap();
            writeln!(s, "        threadgroup const half* Xc = Xs;").unwrap();
            s.push_str(&comp);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }
    }
}

impl TensorGraph {
    /// Element address of arg `which` of `c` in its storage buffer as
    /// `(offset, coefficient per iteration dim)` — `None` unless the
    /// source is a plain buffer (input, reshape of one, or a materialized
    /// single-value comp).
    fn affine_address(&self, c: &super::Comp, which: usize) -> Option<(i64, Vec<i64>)> {
        let a = &c.args[which];
        if a.view.is_gather() {
            return None;
        }
        let src = a.src.n();
        let storage = self.storage(a.src).n();
        match &self.nodes[storage].op {
            Op::Input(_) => {}
            Op::Comp(sc) if sc.tuple == 0 && !self.nodes[storage].inline => {}
            _ => return None,
        }
        // reshape keeps the flat index, so the reading node's own shape
        // gives the strides
        let shape = fixed(&self.nodes[src].shape);
        let mut stride = 1i64;
        let mut off = 0i64;
        let mut coefs = vec![0i64; c.rank()];
        for (f, &n) in a.view.forms.iter().zip(&shape).rev() {
            off += f.offset * stride;
            for (d, &k) in f.coefs.iter().enumerate() {
                coefs[d] += k * stride;
            }
            stride *= n as i64;
        }
        Some((off, coefs))
    }
}

// ---------------------------------------------------------------------------
// Attention on fragments
// ---------------------------------------------------------------------------

/// 8-row query fragments per SIMD group (4 SIMD groups, `32·ATT_RF` rows
/// per threadgroup), keys per tile. Measured on M4 Max: RF 2 and BK 64
/// are both slower (register pressure), so 32 × 32 it is.
const ATT_RF: usize = 1;
const ATT_BK: usize = 32;
/// Largest value-column block per threadgroup (wider `dv` is a grid axis).
const ATT_BC: usize = 64;
/// Largest score reduction (head dim) whose query fragments stay in registers.
const ATT_MAX_D: usize = 256;
/// Finite "minus infinity" (fast math need not honour ∞).
const ATT_NEG: &str = "-3.0e38f";

/// How an online-softmax reduction maps onto fragments (see
/// [`TensorGraph::attention_plan`]).
#[derive(Clone)]
pub struct AttentionPlan {
    /// The online-softmax reduction (the root, or under its epilogue).
    pub red: usize,
    /// Arg of `red` that reads the score comp, and that comp.
    pub score_arg: usize,
    pub score: usize,
    /// Literal factor of the score body (`v_i·v_j·lit`), applied in f32.
    pub scale: f64,
    /// `Q` (`A`, independent of the key) and `K` (independent of the
    /// row) reads, composed onto `red`'s space ++ `[head dim]`.
    pub q: super::Arg,
    pub k: super::Arg,
    /// Score reduction extent (head dim) and key extent.
    pub dk: usize,
    pub nk: usize,
    /// Point dims of `red`: query rows, value columns; the rest are grid
    /// axes (batch, heads).
    pub dr: usize,
    pub dc: usize,
    /// The state slot that is `Σ p·V` (its body a single read of `V`, arg
    /// `v_arg`); every other slot past 0 is a per-row sum `Σ p·g(i, j)`.
    pub pv_slot: usize,
    pub v_arg: usize,
    /// Affine mask condition over `red`'s space when slot 0 is
    /// `score + select(c, ≤ −1e20, 0)`: a key tile where `c > 0`
    /// everywhere contributes nothing and is skipped.
    pub mask: Option<super::Form>,
}

impl TensorGraph {
    /// Flatten a product body into its var factors and literal scale.
    fn product_factors(
        g: &tang_expr::ExprGraph,
        root: tang_expr::ExprId,
    ) -> Option<(Vec<u16>, f64)> {
        let nodes = g.nodes_slice();
        let mut vars = Vec::new();
        let mut scale = 1.0;
        let mut stack = vec![root];
        while let Some(e) = stack.pop() {
            match nodes[e.index() as usize] {
                Node::Mul(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                Node::Var(v) => vars.push(v),
                Node::Lit(b) => scale *= f64::from_bits(b),
                Node::Neg(a) => {
                    scale = -scale;
                    stack.push(a);
                }
                _ => return None,
            }
        }
        Some((vars, scale))
    }

    /// The single-red-dim inline sum `Σ_k A·B·lit` an arg reads: `(A arg,
    /// B arg, lit)` of that comp.
    fn score_comp(&self, src: usize) -> Option<(usize, usize, f64)> {
        let c = self.comp_at(src)?;
        if !self.nodes[src].inline
            || c.monoid.as_ref()?.kind() != MonoidKind::Sum
            || c.width() != 1
            || c.tuple != 0
            || c.red.len() != 1
        {
            return None;
        }
        let (vars, scale) = Self::product_factors(&c.body, c.body_outs[0])?;
        match vars[..] {
            [a, b] if a != b => Some((a as usize, b as usize, scale)),
            _ => None,
        }
    }

    /// The affine mask condition of an online-softmax reduction whose slot
    /// 0 is `score + select(cond, ≤ −1e20, 0)` (`score` the arg
    /// `score_arg`), `cond` affine in iotas: where `cond > 0` the element
    /// contributes nothing, so a tile where it holds everywhere is skipped.
    pub(super) fn score_mask(&self, c: &super::Comp, score_arg: usize) -> Option<super::Form> {
        let n = c.body.nodes_slice();
        let is_score = |e: tang_expr::ExprId| matches!(n[e.index() as usize], Node::Var(v) if v as usize == score_arg);
        let Node::Add(x, y) = n[c.body_outs[0].index() as usize] else {
            return None;
        };
        if !is_score(x) && !is_score(y) {
            return None;
        }
        let other = if is_score(x) { y } else { x };
        let Node::Select(cnd, a, b) = n[other.index() as usize] else {
            return None;
        };
        let lit = |e: tang_expr::ExprId| match n[e.index() as usize] {
            Node::Lit(b) => Some(f64::from_bits(b)),
            _ => None,
        };
        match (lit(a), lit(b)) {
            (Some(neg), Some(zero)) if neg <= -1e20 && zero == 0.0 => self
                .affine_of(c, cnd)
                .filter(|(_, k)| k.fract() == 0.0)
                .map(|(mut f, k)| {
                    f.offset += k as i64;
                    f
                }),
            _ => None,
        }
    }

    /// `e` as an affine form over `c`'s space when it is built from iota
    /// reads, literals, `+`, `−` and literal scaling.
    pub(super) fn affine_of(
        &self,
        c: &super::Comp,
        e: tang_expr::ExprId,
    ) -> Option<(super::Form, f64)> {
        let nodes = c.body.nodes_slice();
        let rank = c.rank();
        match nodes[e.index() as usize] {
            Node::Lit(b) => Some((super::Form::zero(rank), f64::from_bits(b))),
            Node::Var(v) => {
                let a = &c.args[v as usize];
                match self.nodes[a.src.n()].op {
                    Op::Iota(d) if !a.view.guard => Some((a.view.forms[d].clone(), 0.0)),
                    _ => None,
                }
            }
            Node::Add(x, y) => {
                let (fx, cx) = self.affine_of(c, x)?;
                let (fy, cy) = self.affine_of(c, y)?;
                let mut f = fx;
                f.offset += fy.offset;
                for (a, b) in f.coefs.iter_mut().zip(&fy.coefs) {
                    *a += b;
                }
                Some((f, cx + cy))
            }
            Node::Neg(x) => {
                let (f, k) = self.affine_of(c, x)?;
                Some((f.scaled(-1), -k))
            }
            Node::Mul(x, y) => {
                let lit = |e: tang_expr::ExprId| match nodes[e.index() as usize] {
                    Node::Lit(b) => {
                        let v = f64::from_bits(b);
                        (v.fract() == 0.0).then_some(v as i64)
                    }
                    _ => None,
                };
                let (k, other) = match (lit(x), lit(y)) {
                    (Some(k), _) => (k, y),
                    (_, Some(k)) => (k, x),
                    _ => return None,
                };
                let (f, c0) = self.affine_of(c, other)?;
                Some((f.scaled(k), c0 * k as f64))
            }
            _ => None,
        }
    }

    /// Whether `t` (scheduled `fragment`) is attention that runs on f16
    /// fragments with f32 accumulation: an online-softmax reduction over
    /// keys `j` (as the root, or under an epilogue map that reads it
    /// point-for-point) whose
    /// - slot 0 is `score + rest`, `score = Σ_k Q·K·lit` an inlined sum
    ///   over one dim with `Q` independent of `j` and `K` of the query row,
    ///   `rest` any function of args that do not vary along the value
    ///   column (a mask);
    /// - one slot is a single read of `V(j, c)` (`O += P·V`), the others
    ///   functions of `(i, j)` summed per row (the normalizer `1`);
    /// - `Q`, `K`, `V` are f16 buffers, the accumulator f32.
    pub fn attention_plan(&self, t: TensorId) -> Option<AttentionPlan> {
        let root = t.n();
        let red = self.tiled_reduction(root)?;
        let c = self.comp_at(red)?;
        let m = c.monoid.as_ref()?;
        if m.kind() != MonoidKind::OnlineSoftmax
            || c.red.len() != 1
            || c.tuple != 1
            || self.nodes[red].precision.accum != Format::F32
        {
            return None;
        }
        if red != root {
            self.attention_epilogue(root, red)?;
        }
        let pts: Vec<usize> = c
            .points()
            .iter()
            .map(|d| d.fixed())
            .collect::<Option<_>>()?;
        let pk = pts.len();
        let nk = c.red[0].fixed()?;
        let live_vars = |out: tang_expr::ExprId| -> Vec<usize> {
            let n = c.body.nodes_slice();
            c.body
                .live_set(&[out])
                .into_iter()
                .filter_map(|k| match n[k] {
                    Node::Var(v) => Some(v as usize),
                    _ => None,
                })
                .collect()
        };
        // slot 0: exactly one score read
        let s_vars = live_vars(c.body_outs[0]);
        let mut score = None;
        for &v in &s_vars {
            if let Some(sc) = self.score_comp(c.args[v].src.n()) {
                if score.replace((v, sc)).is_some() {
                    return None;
                }
            }
        }
        let (score_arg, (ia, ib, scale)) = score?;
        let sn = c.args[score_arg].src.n();
        let sc = self.comp_at(sn)?;
        let dk = sc.red[0].fixed()?;
        if dk > ATT_MAX_D {
            return None;
        }
        // compose the score's operand reads onto [red space ++ k]
        let rank = pk + 2;
        let ra = &c.args[score_arg];
        let mut inner: Vec<super::Form> = ra
            .view
            .forms
            .iter()
            .map(|f| {
                let mut g = f.clone();
                g.coefs.push(0);
                g
            })
            .collect();
        inner.push(super::Form::unit(rank, pk + 1));
        let inner = super::View {
            forms: inner,
            guard: ra.view.guard,
            gather: None,
        };
        let comp = |a: &super::Arg| super::Arg {
            src: a.src,
            view: a.view.compose(&inner, rank),
        };
        let (qa, ka) = (comp(&sc.args[ia]), comp(&sc.args[ib]));
        let j = pk;
        let (q, k) = if !qa.view.depends_on(j) {
            (qa, ka)
        } else {
            (ka, qa)
        };
        if q.view.depends_on(j) {
            return None;
        }
        // rows: what Q varies along and K does not — the innermost such
        // dim of Q's address
        let qaddr = self.flat_addr(&q, rank);
        let dr = (0..pk)
            .filter(|&d| pts[d] > 1 && q.view.depends_on(d) && !k.view.depends_on(d))
            .min_by_key(|&d| qaddr.coefs[d].unsigned_abs())?;
        // slots
        let mut pv = None;
        for (slot, &out) in c.body_outs.iter().enumerate().skip(1) {
            let vars = live_vars(out);
            let wide = vars.iter().any(|&v| {
                (0..pk).any(|d| {
                    d != dr && c.args[v].view.depends_on(d) && pts[d] > 1 && !q.view.depends_on(d)
                })
            });
            if !wide {
                continue;
            }
            let n = c.body.nodes_slice();
            let Node::Var(v) = n[out.index() as usize] else {
                return None;
            };
            if pv.replace((slot, v as usize)).is_some() {
                return None;
            }
        }
        let (pv_slot, v_arg) = pv?;
        let va = &c.args[v_arg];
        if va.view.depends_on(dr) {
            return None;
        }
        let dcs: Vec<usize> = (0..pk)
            .filter(|&d| pts[d] > 1 && va.view.depends_on(d) && !q.view.depends_on(d))
            .collect();
        let [dc] = dcs[..] else { return None };
        // nothing but the PV slot may vary along the value column
        for (slot, &out) in c.body_outs.iter().enumerate() {
            if slot == pv_slot {
                continue;
            }
            for v in live_vars(out) {
                if v != score_arg && c.args[v].view.depends_on(dc) {
                    return None;
                }
            }
        }
        if q.view.depends_on(dc) || k.view.depends_on(dc) {
            return None;
        }
        // f16 operands
        for a in [&q, &k, va] {
            if !self.f16_buffer(a.src) {
                return None;
            }
        }
        let mask = self.score_mask(c, score_arg);
        Some(AttentionPlan {
            red,
            score_arg,
            score: sn,
            scale,
            q,
            k,
            dk,
            nk,
            dr,
            dc,
            pv_slot,
            v_arg,
            mask,
        })
    }

    /// For an epilogue `root` over online-softmax `red`: for each root
    /// point dim, the `red` point dim it is (a pure pick), or `None` for a
    /// size-1 root dim. Every non-unit `red` point dim must be picked.
    fn attention_epilogue(&self, root: usize, red: usize) -> Option<Vec<Option<usize>>> {
        let c = self.comp_at(root)?;
        let rc = self.comp_at(red)?;
        let forms = self.epilogue_read(c, red)?;
        let rk = c.points().len();
        let rpts = fixed(rc.points());
        let mut map = vec![None; rk];
        for (r, f) in forms.iter().enumerate() {
            let nz: Vec<usize> = (0..rk).filter(|&d| f.coefs[d] != 0).collect();
            match nz[..] {
                [] if f.offset == 0 && rpts[r] == 1 => {}
                [d] if f.coefs[d] == 1 && f.offset == 0 && map[d].is_none() => map[d] = Some(r),
                _ => return None,
            }
        }
        let cpts = fixed(c.points());
        for (d, m) in map.iter().enumerate() {
            if m.is_none() && cpts[d] != 1 {
                return None;
            }
        }
        for (r, &n) in rpts.iter().enumerate() {
            if n > 1 && !map.contains(&Some(r)) {
                return None;
            }
        }
        Some(map)
    }

    /// Storage buffers of `t`'s attention operands `Q`, `K`, `V`.
    pub fn attention_operands(&self, t: TensorId) -> Option<[TensorId; 3]> {
        let root = t.n();
        let red = self.tiled_reduction(root)?;
        let c = self.comp_at(red)?;
        if c.monoid.as_ref()?.kind() != MonoidKind::OnlineSoftmax {
            return None;
        }
        // like the plan, without the precision gate
        let mut ops = Vec::new();
        for (v, a) in c.args.iter().enumerate() {
            if let Some((ia, ib, _)) = self.score_comp(a.src.n()) {
                let sc = self.comp_at(a.src.n())?;
                ops.push(self.storage(sc.args[ia].src));
                ops.push(self.storage(sc.args[ib].src));
                let _ = v;
            }
        }
        for &out in &c.body_outs[1..] {
            if let Node::Var(v) = c.body.nodes_slice()[out.index() as usize] {
                let s = self.storage(c.args[v as usize].src);
                if matches!(self.nodes[s.n()].op, Op::Input(_) | Op::Comp(_)) {
                    ops.push(s);
                }
            }
        }
        match ops[..] {
            [q, k, v] => Some([q, k, v]),
            _ => None,
        }
    }

    /// `a` as a direct f16 buffer read: the buffer node and the flat
    /// element offset as an affine form over the comp's coords (rank
    /// `rank`), when `a` reads an unguarded view of an f16 input.
    fn direct_f16(&self, a: &super::Arg, rank: usize) -> Option<(usize, super::Form)> {
        let j = a.src.n();
        if a.view.guard
            || self.storage(a.src).n() != j
            || !matches!(self.nodes[j].op, Op::Input(_))
            || self.nodes[j].precision.storage != Format::F16
        {
            return None;
        }
        let shape = fixed(&self.nodes[j].shape);
        let mut flat = super::Form::zero(rank);
        let mut stride = 1i64;
        for (f, &n) in a.view.forms.iter().zip(&shape).rev() {
            flat.offset += f.offset * stride;
            for (x, &c) in flat.coefs.iter_mut().zip(&f.coefs) {
                *x += c * stride;
            }
            stride *= n as i64;
        }
        Some((j, flat))
    }

    fn attention_kernel(&self, root: usize, plan: &AttentionPlan) -> MslKernel {
        let (pre, reads) = self.kernel_preamble(root);
        let c = self.comp_at(plan.red).unwrap();
        let pts = fixed(c.points());
        let pk = pts.len();
        let w = c.width();
        let (dr, dc) = (plan.dr, plan.dc);
        let (nq, nc, nk, dk) = (pts[dr], pts[dc], plan.nk, plan.dk);
        let others: Vec<usize> = (0..pk).filter(|&d| d != dr && d != dc).collect();
        let rf = ATT_RF;
        let (bq, bk) = (32 * rf, ATT_BK);
        let bc = nc.next_multiple_of(8).min(ATT_BC);
        let d8 = dk.next_multiple_of(8);
        let (ldq, ldk, ldv) = (d8 + 8, d8 + 8, bc + 8);
        // halves: Q staging, then K|V tiles
        let rk = pk + 2;
        let dq = self.direct_f16(&plan.q, rk);
        let dkk = self.direct_f16(&plan.k, rk);
        let dvv = self.direct_f16(&c.args[plan.v_arg], pk + 1);
        // K and V fragments straight from device memory (no staging, no
        // barriers) when both are direct unit-stride reads and every key
        // tile and value block is full: (K row stride, V row stride)
        let fragdev = match (&dkk, &dvv) {
            (Some((_, kf)), Some((_, vf)))
                if d8 == dk
                    && nk.is_multiple_of(bk)
                    && nc.is_multiple_of(bc)
                    && kf.coefs[pk + 1] == 1
                    && kf.coefs[dr] == 0
                    && kf.coefs[dc] == 0
                    && kf.coefs[pk] > 0
                    && vf.coefs[dc] == 1
                    && vf.coefs[dr] == 0
                    && vf.coefs[pk] > 0 =>
            {
                Some((kf.coefs[pk], vf.coefs[pk]))
            }
            _ => None,
        };
        let kv_halves = if fragdev.is_some() {
            0
        } else {
            bk * ldk + bk * ldv
        };
        let sm = (bq * ldq).max(kv_halves);
        let (tr, tc) = (nq.div_ceil(bq), nc.div_ceil(bc));
        let nlead: usize = others.iter().map(|&d| pts[d]).product();
        let n_tiles = tr * tc * nlead;
        let (nqf, ncf, nkf) = (d8 / 8, bc / 8, bk / 8);
        let scale = super::wgsl::lit(plan.scale);
        // synthetic comp over red space ++ [k]: r0 = key, r1 = head dim
        let syn = super::Comp {
            out: c.out.clone(),
            tuple: c.tuple,
            red: vec![c.red[0], super::Dim::Fixed(dk)],
            args: vec![plan.q.clone(), plan.k.clone()],
            body: c.body.clone(),
            body_outs: c.body_outs.clone(),
            monoid: None,
        };
        // red-space coords in scope as p{d}, r0
        let coords = |row: &str, col: &str, key: &str| -> String {
            format!("const uint p{dr} = {row}; const uint p{dc} = {col}; const uint r0 = {key};")
        };
        let unit_pts: String = others
            .iter()
            .filter(|&&d| pts[d] == 1)
            .map(|d| format!(" const uint p{d} = 0u;"))
            .collect();
        let _ = unit_pts;

        let mut s = pre;
        writeln!(
            s,
            "// attention on fragments: rows p{dr} ({nq}), value cols p{dc} ({nc}), keys {nk}, head dim {dk}, scale {scale}"
        )
        .unwrap();
        writeln!(s, "kernel void k_main(KERNEL_PARAMS uint3 wid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {{").unwrap();
        writeln!(s, "    threadgroup float smem_f[{}];", sm / 2).unwrap();
        writeln!(s, "    threadgroup half* smem = (threadgroup half*)smem_f;").unwrap();
        writeln!(s, "    BUFS_INIT").unwrap();
        writeln!(s, "    const uint tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        // rows slowest; under a mask that lightens low rows (causal: c = j − i)
        // the heavy row tiles are dispatched first
        let heavy_last = plan.mask.as_ref().is_some_and(|f| f.coefs[dr] < 0);
        writeln!(s, "    const uint tile_c = tile % {tc}u;").unwrap();
        writeln!(s, "    uint lead = (tile / {tc}u) % {nlead}u;").unwrap();
        if heavy_last {
            writeln!(
                s,
                "    const uint tile_r = {}u - tile / {}u;",
                tr - 1,
                tc * nlead
            )
            .unwrap();
        } else {
            writeln!(s, "    const uint tile_r = tile / {}u;", tc * nlead).unwrap();
        }
        for &d in others.iter().rev() {
            writeln!(s, "    const uint p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    const uint m0 = tile_r * {bq}u;").unwrap();
        writeln!(s, "    const uint c0 = tile_c * {bc}u;").unwrap();
        // stage Q (zero-padded) and keep this SIMD group's rows in registers
        let q_direct = d8 == dk
            && stage_direct(
                &mut s,
                dq.as_ref(),
                &others,
                (dr, "m0", bq, nq),
                (pk + 1, "0u", d8, dk),
                "smem",
                ldq,
                "    ",
            );
        if !q_direct {
            writeln!(
                s,
                "    for (uint idx = tid; idx < {}u; idx += 128u) {{",
                bq * d8
            )
            .unwrap();
            writeln!(
                s,
                "        const uint ri = idx / {d8}u; const uint ci = idx % {d8}u;"
            )
            .unwrap();
            writeln!(s, "        float v = 0.0f;").unwrap();
            writeln!(s, "        if (m0 + ri < {nq}u && ci < {dk}u) {{").unwrap();
            writeln!(
                s,
                "            {} const uint r1 = ci;",
                coords("m0 + ri", "0u", "0u")
            )
            .unwrap();
            writeln!(s, "            v = {};", arg_read(self, &syn, &syn.args[0])).unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        smem[ri * {ldq}u + ci] = half(v);").unwrap();
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(s, "    simdgroup_half8x8 fqa[{}];", rf * nqf).unwrap();
        writeln!(s, "    for (uint rr = 0u; rr < {rf}u; rr++) for (uint kk = 0u; kk < {nqf}u; kk++) simdgroup_load(fqa[rr * {nqf}u + kk], &smem[((sg * {rf}u + rr) * 8u) * {ldq}u + kk * 8u], {ldq});").unwrap();
        writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(s, "    threadgroup half* Ks = smem;").unwrap();
        writeln!(s, "    threadgroup half* Vs = smem + {};", bk * ldk).unwrap();
        // MLX's lane layout of an 8×8 fragment: each lane holds one row,
        // two adjacent columns; a row's 8 columns live on lanes ^1 and ^8
        writeln!(s, "    const uint qid = lane / 4u;").unwrap();
        writeln!(s, "    const uint fm = (qid & 4u) + (lane / 2u) % 4u;").unwrap();
        writeln!(s, "    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;").unwrap();
        // fragment rr of this SIMD group holds rows rowb + 8·rr
        writeln!(s, "    const uint rowb = m0 + sg * {}u + fm;", 8 * rf).unwrap();
        writeln!(s, "    float mrows[{rf}];").unwrap();
        writeln!(
            s,
            "    for (uint rr = 0u; rr < {rf}u; rr++) mrows[rr] = {ATT_NEG};"
        )
        .unwrap();
        let row_slots: Vec<usize> = (1..w).filter(|&k| k != plan.pv_slot).collect();
        for &k in &row_slots {
            writeln!(s, "    float lsums{k}[{rf}];").unwrap();
            writeln!(
                s,
                "    for (uint rr = 0u; rr < {rf}u; rr++) lsums{k}[rr] = 0.0f;"
            )
            .unwrap();
        }
        writeln!(s, "    simdgroup_float8x8 acca[{}];", rf * ncf).unwrap();
        writeln!(s, "    for (uint cf = 0u; cf < {}u; cf++) acca[cf] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);", rf * ncf).unwrap();
        let base_of = |f: &super::Form| {
            let mut e = format!("{}", f.offset);
            for &d in &others {
                if f.coefs[d] != 0 {
                    e.push_str(&format!(" + {} * int(p{d})", f.coefs[d]));
                }
            }
            e
        };
        if let (Some((sk, sv)), Some((kb, kf)), Some((vb, vf))) = (fragdev, &dkk, &dvv) {
            let _ = (sk, sv);
            writeln!(s, "    device const half* Kd = b{kb} + ({});", base_of(kf)).unwrap();
            writeln!(
                s,
                "    device const half* Vd = b{vb} + ({}) + c0;",
                base_of(vf)
            )
            .unwrap();
        }
        writeln!(s, "    for (uint j0 = 0u; j0 < {nk}u; j0 += {bk}u) {{").unwrap();
        if let Some(f) = &plan.mask {
            // min of the affine mask condition over the tile's box
            let mut e = format!("{}", f.offset);
            for &d in &others {
                if f.coefs[d] != 0 {
                    e.push_str(&format!(" + {} * int(p{d})", f.coefs[d]));
                }
            }
            let mut term = |coef: i64, lo: String, hi: String| {
                if coef != 0 {
                    let at = if coef > 0 { lo } else { hi };
                    e.push_str(&format!(" + {coef} * ({at})"));
                }
            };
            term(
                f.coefs[dr],
                "int(m0)".into(),
                format!("int(min(m0 + {bq}u, {nq}u)) - 1"),
            );
            term(
                f.coefs[dc],
                "int(c0)".into(),
                format!("int(min(c0 + {bc}u, {nc}u)) - 1"),
            );
            term(
                f.coefs[pk],
                "int(j0)".into(),
                format!("int(min(j0 + {bk}u, {nk}u)) - 1"),
            );
            writeln!(s, "        // the mask covers this whole key tile: skip it").unwrap();
            writeln!(s, "        if ({e} > 0) {{ continue; }}").unwrap();
        }
        // stage K [j][k] and V [j][c]
        if fragdev.is_none() {
            let k_direct = d8 == dk
                && stage_direct(
                    &mut s,
                    dkk.as_ref(),
                    &others,
                    (pk, "j0", bk, nk),
                    (pk + 1, "0u", d8, dk),
                    "Ks",
                    ldk,
                    "        ",
                );
            if !k_direct {
                writeln!(
                    s,
                    "        for (uint idx = tid; idx < {}u; idx += 128u) {{",
                    bk * d8
                )
                .unwrap();
                writeln!(
                    s,
                    "            const uint ji = idx / {d8}u; const uint ci = idx % {d8}u;"
                )
                .unwrap();
                writeln!(s, "            float v = 0.0f;").unwrap();
                writeln!(s, "            if (j0 + ji < {nk}u && ci < {dk}u) {{").unwrap();
                writeln!(
                    s,
                    "                {} const uint r1 = ci;",
                    coords("0u", "0u", "j0 + ji")
                )
                .unwrap();
                writeln!(
                    s,
                    "                v = {};",
                    arg_read(self, &syn, &syn.args[1])
                )
                .unwrap();
                writeln!(s, "            }}").unwrap();
                writeln!(s, "            Ks[ji * {ldk}u + ci] = half(v);").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            let v_direct = stage_direct(
                &mut s,
                dvv.as_ref(),
                &others,
                (pk, "j0", bk, nk),
                (dc, "c0", bc, nc),
                "Vs",
                ldv,
                "        ",
            );
            if !v_direct {
                writeln!(
                    s,
                    "        for (uint idx = tid; idx < {}u; idx += 128u) {{",
                    bk * bc
                )
                .unwrap();
                writeln!(
                    s,
                    "            const uint ji = idx / {bc}u; const uint ci = idx % {bc}u;"
                )
                .unwrap();
                writeln!(s, "            float v = 0.0f;").unwrap();
                writeln!(s, "            if (j0 + ji < {nk}u && c0 + ci < {nc}u) {{").unwrap();
                writeln!(s, "                {}", coords("0u", "c0 + ci", "j0 + ji")).unwrap();
                writeln!(
                    s,
                    "                v = {};",
                    arg_read(self, c, &c.args[plan.v_arg])
                )
                .unwrap();
                writeln!(s, "            }}").unwrap();
                writeln!(s, "            Vs[ji * {ldv}u + ci] = half(v);").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
        }
        // S = Q·Kᵀ
        writeln!(s, "        simdgroup_float8x8 sfa[{}];", rf * nkf).unwrap();
        writeln!(s, "        for (uint jf = 0u; jf < {}u; jf++) sfa[jf] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);", rf * nkf).unwrap();
        writeln!(s, "        for (uint kk = 0u; kk < {nqf}u; kk++) {{").unwrap();
        writeln!(s, "            for (uint jf = 0u; jf < {nkf}u; jf++) {{").unwrap();
        writeln!(s, "                simdgroup_half8x8 kf;").unwrap();
        match fragdev {
            Some((sk, _)) => writeln!(s, "                simdgroup_load(kf, Kd + (j0 + jf * 8u) * {sk}u + kk * 8u, {sk}, ulong2(0, 0), true);").unwrap(),
            None => writeln!(s, "                simdgroup_load(kf, &Ks[(jf * 8u) * {ldk}u + kk * 8u], {ldk}, ulong2(0, 0), true);").unwrap(),
        }
        writeln!(
            s,
            "                for (uint rr = 0u; rr < {rf}u; rr++) simdgroup_multiply_accumulate(sfa[rr * {nkf}u + jf], fqa[rr * {nqf}u + kk], kf, sfa[rr * {nkf}u + jf]);"
        )
        .unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "        }}").unwrap();
        // slot-0 element value at (row, key) in this lane's registers
        let var_of = |pfx: &'static str| {
            move |v: u16| -> String {
                if v as usize == plan.score_arg {
                    "sx".to_string()
                } else {
                    let _ = pfx;
                    arg_read(self, c, &c.args[v as usize])
                }
            }
        };
        writeln!(s, "        simdgroup_half8x8 pfa[{}];", rf * nkf).unwrap();
        // one copy per row fragment, every index constant (references or
        // pointers into the fragment arrays spill them)
        let blk_start = s.len();
        writeln!(s, "        {{").unwrap();
        writeln!(s, "        const uint row = rowb + RR8u;").unwrap();
        writeln!(s, "        float sv[{}];", 2 * nkf).unwrap();
        writeln!(s, "        float tmax = {ATT_NEG};").unwrap();
        writeln!(s, "        for (uint jf = 0u; jf < {nkf}u; jf++) {{").unwrap();
        writeln!(s, "            for (uint e = 0u; e < 2u; e++) {{").unwrap();
        writeln!(s, "                const uint key = j0 + jf * 8u + fn + e;").unwrap();
        writeln!(s, "                float val = {ATT_NEG};").unwrap();
        writeln!(s, "                if (row < {nq}u && key < {nk}u) {{").unwrap();
        writeln!(
            s,
            "                    const float sx = sf[jf].thread_elements()[e] * {scale};"
        )
        .unwrap();
        writeln!(s, "                    {}", coords("row", "0u", "key")).unwrap();
        let vf = var_of("s0_");
        let outs = emit_scalar_mapped(
            &mut s,
            "                    ",
            &c.body,
            &c.body_outs[..1],
            "s0_",
            &vf,
        );
        writeln!(s, "                    val = {};", outs[0]).unwrap();
        writeln!(s, "                }}").unwrap();
        writeln!(s, "                sv[jf * 2u + e] = val;").unwrap();
        writeln!(s, "                tmax = max(tmax, val);").unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        tmax = max(tmax, simd_shuffle_xor(tmax, 1u));").unwrap();
        writeln!(s, "        tmax = max(tmax, simd_shuffle_xor(tmax, 8u));").unwrap();
        writeln!(s, "        const float mnew = max(mrow, tmax);").unwrap();
        writeln!(s, "        const float alpha = exp(mrow - mnew);").unwrap();
        writeln!(s, "        mrow = mnew;").unwrap();
        for &k in &row_slots {
            writeln!(s, "        lsum{k} *= alpha;").unwrap();
        }
        writeln!(s, "        for (uint cf = 0u; cf < {ncf}u; cf++) {{ acc[cf].thread_elements()[0] *= alpha; acc[cf].thread_elements()[1] *= alpha; }}").unwrap();
        writeln!(s, "        for (uint jf = 0u; jf < {nkf}u; jf++) {{").unwrap();
        writeln!(s, "            for (uint e = 0u; e < 2u; e++) {{").unwrap();
        writeln!(s, "                const uint key = j0 + jf * 8u + fn + e;").unwrap();
        writeln!(
            s,
            "                const bool ok = row < {nq}u && key < {nk}u;"
        )
        .unwrap();
        writeln!(
            s,
            "                const float p = ok ? exp(sv[jf * 2u + e] - mnew) : 0.0f;"
        )
        .unwrap();
        writeln!(s, "                pf[jf].thread_elements()[e] = half(p);").unwrap();
        // per-row slots
        let simple: Vec<(usize, Option<f64>)> = row_slots
            .iter()
            .map(|&k| {
                let out = c.body_outs[k];
                let lit = match c.body.nodes_slice()[out.index() as usize] {
                    Node::Lit(b) => Some(f64::from_bits(b)),
                    _ => None,
                };
                (k, lit)
            })
            .collect();
        for (k, lit) in &simple {
            if let Some(v) = lit {
                writeln!(
                    s,
                    "                lsum{k} += p * {};",
                    super::wgsl::lit(*v)
                )
                .unwrap();
            }
        }
        let general: Vec<usize> = simple
            .iter()
            .filter(|x| x.1.is_none())
            .map(|x| x.0)
            .collect();
        if !general.is_empty() {
            writeln!(s, "                if (ok) {{").unwrap();
            writeln!(
                s,
                "                    const float sx = sf[jf].thread_elements()[e] * {scale};"
            )
            .unwrap();
            writeln!(s, "                    {}", coords("row", "0u", "key")).unwrap();
            let outs: Vec<tang_expr::ExprId> = general.iter().map(|&k| c.body_outs[k]).collect();
            let vf = var_of("g_");
            let vals =
                emit_scalar_mapped(&mut s, "                    ", &c.body, &outs, "g_", &vf);
            for (k, v) in general.iter().zip(vals) {
                writeln!(s, "                    lsum{k} += p * {v};").unwrap();
            }
            writeln!(s, "                }}").unwrap();
        }
        writeln!(s, "            }}").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        }}").unwrap();
        let blk = s.split_off(blk_start);
        for rr in 0..rf {
            let mut b = blk
                .replace("RR8u", &format!("{}u", 8 * rr))
                .replace("sf[", &format!("sfa[{}u + ", rr * nkf))
                .replace("pf[", &format!("pfa[{}u + ", rr * nkf))
                .replace("acc[", &format!("acca[{}u + ", rr * ncf))
                .replace("mrow", &format!("mrows[{rr}]"));
            for &k in &row_slots {
                b = b.replace(&format!("lsum{k}"), &format!("lsums{k}[{rr}]"));
            }
            s.push_str(&b);
        }
        // O += P·V
        writeln!(s, "        for (uint jf = 0u; jf < {nkf}u; jf++) {{").unwrap();
        writeln!(s, "            for (uint cf = 0u; cf < {ncf}u; cf++) {{").unwrap();
        writeln!(s, "                simdgroup_half8x8 vf;").unwrap();
        match fragdev {
            Some((_, sv)) => writeln!(
                s,
                "                simdgroup_load(vf, Vd + (j0 + jf * 8u) * {sv}u + cf * 8u, {sv});"
            )
            .unwrap(),
            None => writeln!(
                s,
                "                simdgroup_load(vf, &Vs[(jf * 8u) * {ldv}u + cf * 8u], {ldv});"
            )
            .unwrap(),
        }
        writeln!(
            s,
            "                for (uint rr = 0u; rr < {rf}u; rr++) simdgroup_multiply_accumulate(acca[rr * {ncf}u + cf], pfa[rr * {nkf}u + jf], vf, acca[rr * {ncf}u + cf]);"
        )
        .unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "        }}").unwrap();
        if fragdev.is_none() {
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
        }
        writeln!(s, "    }}").unwrap();
        // write-out (and epilogue) straight from the fragments: a lane
        // holds row fm, columns fn, fn + 1 of each 8×8 tile; per-row
        // slots are reduced across the row's lanes (^1, ^8)
        writeln!(s, "    for (uint rr = 0u; rr < {rf}u; rr++) {{").unwrap();
        writeln!(s, "    const uint row = rowb + rr * 8u;").unwrap();
        for &k in &row_slots {
            writeln!(s, "    float lsum{k} = lsums{k}[rr];").unwrap();
            writeln!(s, "    lsum{k} += simd_shuffle_xor(lsum{k}, 1u);").unwrap();
            writeln!(s, "    lsum{k} += simd_shuffle_xor(lsum{k}, 8u);").unwrap();
        }
        writeln!(s, "    for (uint cf = 0u; cf < {ncf}u; cf++) {{").unwrap();
        writeln!(s, "    for (uint e = 0u; e < 2u; e++) {{").unwrap();
        writeln!(s, "        const uint col = c0 + cf * 8u + fn + e;").unwrap();
        writeln!(s, "        if (row < {nq}u && col < {nc}u) {{").unwrap();
        writeln!(s, "        const uint p{dr} = row; const uint p{dc} = col;").unwrap();
        for k in 0..w {
            let v = if k == plan.pv_slot {
                format!("acca[rr * {ncf}u + cf].thread_elements()[e]")
            } else if k == 0 {
                "mrows[rr]".to_string()
            } else {
                format!("lsum{k}")
            };
            writeln!(s, "        const float sl{k} = {v};").unwrap();
        }
        let out_f = self.nodes[root].precision.storage;
        let flat = |prefix: &str, pts: &[usize]| {
            let mut e = String::from("0u");
            for (d, n) in pts.iter().enumerate() {
                e = format!("({e}) * {n}u + {prefix}{d}");
            }
            e
        };
        if root == plan.red {
            writeln!(s, "        const uint g = {};", flat("p", &pts)).unwrap();
            for k in 0..w {
                writeln!(
                    s,
                    "        {}",
                    store(out_f, &format!("g * {w}u + {k}u"), &format!("sl{k}"))
                )
                .unwrap();
            }
        } else {
            let map = self.attention_epilogue(root, plan.red).unwrap();
            let mc = self.comp_at(root).unwrap();
            let rpts = fixed(mc.points());
            for (d, m) in map.iter().enumerate() {
                match m {
                    Some(r) => writeln!(s, "        const uint t{d} = p{r};").unwrap(),
                    None => writeln!(s, "        const uint t{d} = 0u;").unwrap(),
                }
            }
            writeln!(s, "        {{").unwrap();
            for d in 0..rpts.len() {
                writeln!(s, "        const uint p{d} = t{d};").unwrap();
            }
            let args: Vec<String> = mc
                .args
                .iter()
                .map(|a| {
                    if a.src.n() == plan.red {
                        format!("sl{}", self.epilogue_slot(a))
                    } else {
                        arg_read(self, mc, a)
                    }
                })
                .collect();
            let vals =
                emit_scalar_mapped(&mut s, "        ", &mc.body, &mc.body_outs, "e_", &|v| {
                    args[v as usize].clone()
                });
            writeln!(s, "        const uint g = {};", flat("p", &rpts)).unwrap();
            let mw = mc.width();
            for (slot, v) in vals.iter().enumerate() {
                writeln!(
                    s,
                    "        {}",
                    store(out_f, &format!("g * {mw}u + {slot}u"), v)
                )
                .unwrap();
            }
            writeln!(s, "        }}").unwrap();
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        let points = numel_fixed(&fixed(self.comp_at(root).unwrap().points()));
        MslKernel {
            source: translate(self, &unroll_const_loops(&s), &reads, root).0,
            threadgroup_bytes: 2 * sm,
            entry: "k_main".into(),
            node: TensorId(root as u32),
            reads: reads.iter().map(|&j| TensorId(j as u32)).collect(),
            points: points as u32,
            groups: grid(n_tiles),
            threads: 128,
            fragment: true,
        }
    }
}

/// Stage a `rows × cols` tile of a direct f16 buffer read into threadgroup
/// memory with `half4` copies (see [`TensorGraph::direct_f16`]): `flat`
/// is unit-stride along `col_dim`; coords in `others` are in scope as
/// `p{d}`. Out-of-range rows/columns are zero. `false` (nothing emitted)
/// when the read is not 4-aligned.
#[allow(clippy::too_many_arguments)]
fn stage_direct(
    s: &mut String,
    direct: Option<&(usize, super::Form)>,
    others: &[usize],
    (row_dim, row0, rows, row_lim): (usize, &str, usize, usize),
    (col_dim, col0, cols, col_lim): (usize, &str, usize, usize),
    dst: &str,
    ld: usize,
    pad: &str,
) -> bool {
    let Some((buf, flat)) = direct else {
        return false;
    };
    let aligned = flat.offset % 4 == 0
        && flat.coefs[col_dim] == 1
        && flat
            .coefs
            .iter()
            .enumerate()
            .all(|(d, &c)| d == col_dim || c % 4 == 0)
        && cols.is_multiple_of(4)
        && col_lim.is_multiple_of(4)
        && ld.is_multiple_of(4);
    if !aligned {
        return false;
    }
    let mut base = format!("{}", flat.offset);
    for &d in others {
        if flat.coefs[d] != 0 {
            base.push_str(&format!(" + {} * int(p{d})", flat.coefs[d]));
        }
    }
    let rc = flat.coefs[row_dim];
    let c4 = cols / 4;
    writeln!(s, "{pad}{{").unwrap();
    writeln!(s, "{pad}    const int base = {base};").unwrap();
    writeln!(
        s,
        "{pad}    for (uint idx = tid; idx < {}u; idx += 128u) {{",
        rows * c4
    )
    .unwrap();
    writeln!(
        s,
        "{pad}        const uint ri = idx / {c4}u; const uint ci = (idx % {c4}u) * 4u;"
    )
    .unwrap();
    writeln!(s, "{pad}        half4 v = half4(0);").unwrap();
    writeln!(
        s,
        "{pad}        if ({row0} + ri < {row_lim}u && {col0} + ci < {col_lim}u) {{ v = *(device const half4*)(b{buf} + (base + {rc} * int({row0} + ri) + int({col0} + ci))); }}"
    )
    .unwrap();
    writeln!(
        s,
        "{pad}        *(threadgroup half4*)(&{dst}[ri * {ld}u + ci]) = v;"
    )
    .unwrap();
    writeln!(s, "{pad}    }}").unwrap();
    writeln!(s, "{pad}}}").unwrap();
    true
}

/// Unroll every `for (uint x = 0u; x < Nu; x++) body` in MSL text into N
/// scopes `{ const uint x = i; body }` (nested loops too): indices into
/// fragment arrays and `thread_elements()` become constants, which the
/// Metal compiler otherwise may leave dynamic and spill to the stack.
fn unroll_const_loops(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len() * 2);
    let mut i = 0;
    while let Some(off) = src[i..].find("for (uint ") {
        let at = i + off;
        out.push_str(&src[i..at]);
        let head_end = matching(src, at + 4);
        let head = &src[at + 5..head_end];
        let parsed = (|| {
            let mut parts = head.split(';').map(str::trim);
            let init = parts.next()?.strip_prefix("uint ")?;
            let (var, start) = init.split_once('=')?;
            let var = var.trim();
            if start.trim() != "0u" {
                return None;
            }
            let cond = parts.next()?;
            let n: usize = cond
                .strip_prefix(var)?
                .trim()
                .strip_prefix('<')?
                .trim()
                .strip_suffix('u')?
                .parse()
                .ok()?;
            (parts.next()? == format!("{var}++") && n <= 16).then(|| (var.to_string(), n))
        })();
        let Some((var, n)) = parsed else {
            out.push_str(&src[at..head_end + 1]);
            i = head_end + 1;
            continue;
        };
        let mut j = head_end + 1;
        while b[j] == b' ' {
            j += 1;
        }
        let (body, end) = if b[j] == b'{' {
            let mut depth = 0;
            let mut k = j;
            loop {
                match b[k] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            (&src[j + 1..k], k + 1)
        } else {
            let mut depth = 0;
            let mut k = j;
            loop {
                match b[k] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b';' if depth == 0 => break,
                    _ => {}
                }
                k += 1;
            }
            (&src[j..k + 1], k + 1)
        };
        let body = unroll_const_loops(body);
        for v in 0..n {
            write!(out, "{{ const uint {var} = {v}u; {body} }}").unwrap();
        }
        i = end;
    }
    out.push_str(&src[i..]);
    out
}

fn numel_fixed(p: &[usize]) -> usize {
    p.iter().product()
}

// ---------------------------------------------------------------------------
// WGSL surface syntax → MSL
// ---------------------------------------------------------------------------

fn msl_ty(f: Format) -> &'static str {
    match f {
        Format::F32 => "float",
        Format::F16 => "half",
        Format::BF16 => "bfloat",
    }
}

const PRELUDE: &str = "#include <metal_stdlib>
using namespace metal;
typedef uint u32;
typedef int i32;
typedef float f32;
template <typename T> using vec4 = vec<T, 4>;
struct Params { uint count; uint _p1; uint _p2; uint _p3; };
inline void workgroupBarrier() { threadgroup_barrier(mem_flags::mem_threadgroup); }
";

/// Rewrite an emitter's WGSL-syntax kernel (or a fragment kernel, already
/// MSL in its `main`) to MSL.
fn translate(g: &TensorGraph, src: &str, reads: &[usize], root: usize) -> (String, usize) {
    let mut out = String::from(PRELUDE);
    // buffers: helper functions see them through `B`
    writeln!(out, "struct Bufs {{").unwrap();
    for &j in reads {
        let f = g.nodes[j].precision.storage;
        writeln!(out, "    device const {}* b{j};", msl_ty(f)).unwrap();
    }
    let out_f = g.nodes[root].precision.storage;
    writeln!(out, "    device {}* out;", msl_ty(out_f)).unwrap();
    writeln!(out, "}};").unwrap();
    let mut params = String::new();
    for (i, &j) in reads.iter().enumerate() {
        let f = g.nodes[j].precision.storage;
        write!(params, "device const {}* b{j} [[buffer({i})]], ", msl_ty(f)).unwrap();
    }
    write!(
        params,
        "device {}* out [[buffer({})]], constant Params& params [[buffer({})]], ",
        msl_ty(out_f),
        reads.len(),
        reads.len() + 1
    )
    .unwrap();
    let names: Vec<String> = reads
        .iter()
        .map(|j| format!("b{j}"))
        .chain(std::iter::once("out".to_string()))
        .collect();
    let bufs_init = format!("Bufs B{{{}}};", names.join(", "));

    let mut shared = Vec::new();
    let mut tg_bytes = 0;
    let mut body = String::new();
    for line in src.lines() {
        let t = line.trim_start();
        if t.starts_with("struct Params") || t.starts_with("@group") || t.starts_with("@compute") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("var<workgroup> ") {
            // `name: array<T, N>;` → `threadgroup T name[N];`
            let (name, ty) = rest.trim_end_matches(';').split_once(": ").unwrap();
            let inner = ty
                .strip_prefix("array<")
                .unwrap()
                .strip_suffix('>')
                .unwrap();
            let (elem, n) = inner.rsplit_once(", ").unwrap();
            let width = match elem.strip_prefix("array<f32, ") {
                Some(w) => w.trim_end_matches('>').parse::<usize>().unwrap(),
                // vec4 point-minor stages (`tensor_tile` sites)
                None if elem.starts_with("vec4<") => 4,
                None => 1,
            };
            tg_bytes += 4 * width * n.parse::<usize>().unwrap();
            shared.push(format!("    threadgroup {elem} {name}[{n}];"));
            continue;
        }
        if t.starts_with("fn main(") {
            writeln!(
                body,
                "kernel void k_main({params}uint3 gid [[thread_position_in_grid]], uint3 wid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint sg_size [[threads_per_simdgroup]], uint sg_lane [[thread_index_in_simdgroup]]) {{"
            )
            .unwrap();
            for d in &shared {
                writeln!(body, "{d}").unwrap();
            }
            writeln!(body, "    {bufs_init}").unwrap();
            continue;
        }
        if t.starts_with("fn ") {
            writeln!(body, "{}", signature(t)).unwrap();
            continue;
        }
        if t.starts_with("//") {
            writeln!(body, "{line}").unwrap();
            continue;
        }
        let line = line
            .replace("KERNEL_PARAMS ", &params)
            .replace("BUFS_INIT", &bufs_init)
            .replace("subgroupAdd(", "simd_sum(")
            .replace("subgroupMax(", "simd_max(")
            .replace("subgroupShuffleXor(", "simd_shuffle_xor(");
        writeln!(body, "{}", statement(&line)).unwrap();
    }
    out.push_str(&body);
    (out, tg_bytes)
}

/// `fn name(a: T, …) -> R {` → `R name(thread const Bufs& B, T a, …) {`.
fn signature(t: &str) -> String {
    let rest = &t[3..];
    let open = rest.find('(').unwrap();
    let name = &rest[..open];
    let close = matching(rest, open);
    let args = &rest[open + 1..close];
    let ret = rest[close + 1..]
        .trim()
        .trim_start_matches("->")
        .trim()
        .trim_end_matches('{')
        .trim();
    let mut ps = vec!["thread const Bufs& B".to_string()];
    for a in split_top(args) {
        let a = a.trim();
        if a.is_empty() {
            continue;
        }
        let (n, ty) = a.split_once(':').unwrap();
        ps.push(format!("{} {}", ty.trim(), n.trim()));
    }
    format!("{} {name}({}) {{", fix_floats(ret), ps.join(", "))
}

/// Index of the bracket closing the one at `open`.
fn matching(s: &str, open: usize) -> usize {
    let b = s.as_bytes();
    let (o, c) = (b[open], if b[open] == b'(' { b')' } else { b'>' });
    let mut depth = 0;
    for (i, &ch) in b.iter().enumerate().skip(open) {
        if ch == o {
            depth += 1;
        } else if ch == c {
            depth -= 1;
            if depth == 0 {
                return i;
            }
        }
    }
    panic!("unbalanced: {s}");
}

/// Split on commas outside any `()`/`<>` nesting.
fn split_top(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// One statement line: declarations, constructors, selects, buffer and
/// helper-call rewrites, float literal suffixes.
fn statement(line: &str) -> String {
    let s = decls(line);
    let s = selects(&s);
    let s = array_ctors(&s);
    let s = calls_and_buffers(&s);
    fix_floats(&s)
}

/// `let x = e` → `const auto x = e`, `let x: T = e` → `const T x = e`,
/// `var x = e` → `auto x = e`, `var x: T[= e]` → `T x[= e]`.
fn decls(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let kw = if s[i..].starts_with("let ") {
            Some(true)
        } else if s[i..].starts_with("var ") {
            Some(false)
        } else {
            None
        };
        let boundary = i == 0 || !is_ident(b[i - 1]);
        if let (Some(is_let), true) = (kw, boundary) {
            let mut j = i + 4;
            let name_start = j;
            while j < b.len() && is_ident(b[j]) {
                j += 1;
            }
            let name = &s[name_start..j];
            let ty = if s[j..].starts_with(": ") {
                // type runs to ` =` or `;` at angle depth 0
                let ts = j + 2;
                let mut k = ts;
                let mut depth = 0;
                while k < b.len() {
                    match b[k] {
                        b'<' => depth += 1,
                        b'>' => depth -= 1,
                        b'=' | b';' if depth == 0 => break,
                        _ => {}
                    }
                    k += 1;
                }
                let t = s[ts..k].trim().to_string();
                j = k;
                Some(t)
            } else {
                None
            };
            let c = if is_let { "const " } else { "" };
            match ty {
                Some(t) => write!(out, "{c}{t} {name}").unwrap(),
                None => write!(out, "{c}auto {name}").unwrap(),
            }
            if !s[j..].starts_with(' ') && !s[j..].starts_with(';') {
                out.push(' ');
            }
            i = j;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// `select(f, t, c)` → `((c) ? (t) : (f))`: lazy, so a guarded read that
/// would leave its buffer is never performed (MSL has no robust access).
fn selects(s: &str) -> String {
    let Some(pos) = find_word(s, "select(") else {
        return s.to_string();
    };
    let open = pos + "select".len();
    let close = matching(s, open);
    let parts = split_top(&s[open + 1..close]);
    assert_eq!(parts.len(), 3, "select arity: {s}");
    let (f, t, c) = (
        selects(parts[0].trim()),
        selects(parts[1].trim()),
        selects(parts[2].trim()),
    );
    format!(
        "{}(({c}) ? ({t}) : ({f})){}",
        &s[..pos],
        selects(&s[close + 1..])
    )
}

fn find_word(s: &str, w: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut from = 0;
    while let Some(p) = s[from..].find(w) {
        let p = from + p;
        if p == 0 || !is_ident(b[p - 1]) {
            return Some(p);
        }
        from = p + 1;
    }
    None
}

/// `array<f32, N>(a, b)` → `array<f32, N>{a, b}`.
fn array_ctors(s: &str) -> String {
    let mut out = s.to_string();
    let mut from = 0;
    while let Some(p) = out[from..].find("array<") {
        let p = from + p;
        let lt = p + "array".len();
        let gt = matching(&out, lt);
        if out.as_bytes().get(gt + 1) == Some(&b'(') {
            let close = matching(&out, gt + 1);
            out.replace_range(close..close + 1, "}");
            out.replace_range(gt + 1..gt + 2, "{");
        }
        from = gt;
    }
    out
}

/// Insert `B` into helper calls; route buffer reads and the output
/// through it.
fn calls_and_buffers(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    while i < b.len() {
        let boundary = i == 0 || !is_ident(b[i - 1]);
        if boundary {
            // helper call: (read|point|body|combine)_<n>(
            let mut matched = false;
            for p in ["read_", "point_", "body_", "combine_"] {
                if s[i..].starts_with(p) {
                    let mut j = i + p.len();
                    let ds = j;
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > ds && b.get(j) == Some(&b'(') {
                        out.push_str(&s[i..=j]);
                        let empty = b.get(j + 1) == Some(&b')');
                        out.push_str(if empty { "B" } else { "B, " });
                        i = j + 1;
                        matched = true;
                    }
                    break;
                }
            }
            if matched {
                continue;
            }
            // buffer: b<n>[ or out[
            if b[i] == b'b' {
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                if j > i + 1 && b.get(j) == Some(&b'[') {
                    out.push_str("B.");
                    out.push_str(&s[i..j]);
                    i = j;
                    continue;
                }
            }
            if s[i..].starts_with("out[") {
                out.push_str("B.out");
                i += 3;
                continue;
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Give every float literal an `f` suffix (an unsuffixed literal is a
/// `double` in C++, which `auto` would then propagate).
fn fix_floats(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < b.len() {
        let starts = b[i].is_ascii_digit() && (i == 0 || !(is_ident(b[i - 1]) || b[i - 1] == b'.'));
        if !starts {
            out.push(b[i] as char);
            i += 1;
            continue;
        }
        let st = i;
        let mut float = false;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i < b.len() && b[i] == b'.' {
            float = true;
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
            let mut j = i + 1;
            if j < b.len() && (b[j] == b'-' || b[j] == b'+') {
                j += 1;
            }
            if j < b.len() && b[j].is_ascii_digit() {
                float = true;
                i = j;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
        }
        out.push_str(&s[st..i]);
        if float && !(i < b.len() && (b[i] == b'f' || b[i] == b'F')) {
            out.push('f');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statement_rewrites() {
        assert_eq!(
            statement("    let x = select(0.0, read_3(u32(p0)), (u32(p0) < 4u));"),
            "    const auto x = (((u32(p0) < 4u)) ? (read_3(B, u32(p0))) : (0.0f));"
        );
        assert_eq!(
            statement("for (var r0: u32 = 0u; r0 < 4u; r0 = r0 + 1u) {"),
            "for (u32 r0 = 0u; r0 < 4u; r0 = r0 + 1u) {"
        );
        assert_eq!(
            statement("var acc = array<f32, 2>(1e-5, b2[i]);"),
            "auto acc = array<f32, 2>{1e-5f, B.b2[i]};"
        );
        assert_eq!(
            statement("let v = point_4();"),
            "const auto v = point_4(B);"
        );
        assert_eq!(
            signature("fn combine_3(a: array<f32, 2>, e: array<f32, 2>) -> array<f32, 2> {"),
            "array<f32, 2> combine_3(thread const Bufs& B, array<f32, 2> a, array<f32, 2> e) {"
        );
    }
}
