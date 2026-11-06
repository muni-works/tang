//! Plain C lowering for CPUs: one translation unit per program, built by
//! the system compiler (gcc/clang `-O3`) and auto-vectorized for NEON or
//! AVX. Aimed at batch-1, latency-bound control steps, where a GPU round
//! trip costs more than the arithmetic (BENCHMARKS.md, Adreno 740).
//!
//! The comps are lowered directly, not transpiled from the WGSL kernels.
//! A WGSL kernel is written per thread: one point, the thread mapping
//! (`split`, `tile`, `reg`) baked into index arithmetic and barriers. On a
//! single CPU core that decomposition is pure overhead, and undoing it
//! textually would mean recovering loops from thread ids. Lowering the
//! comprehension itself gives the loops directly: the point space and the
//! reduce space are nested `for` loops with constant trip counts, every
//! read is an affine index into a `restrict` pointer, and the compiler
//! does the rest. GPU schedule fields are ignored; `inline` and
//! `precision` (storage and accumulation formats) are honoured.
//!
//! Per kernel (each materialized comp):
//! - The *vector dim* is the point dim most args read with unit stride
//!   (ties: the innermost). It is cut into constant chunks of `JB` points
//!   that form the innermost loop (`#pragma GCC ivdep`), so the compiler
//!   vectorizes across output points and keeps the accumulators in
//!   registers.
//! - The reduce loops sit outside that chunk loop: for `Σ_k x[i,k]·W[k,j]`
//!   the inner loop streams a contiguous row of `W` into `JB` accumulators.
//!   Any monoid works — combine is a per-point scalar graph.
//! - A *row block* of `RB = 4` points along a second point dim is
//!   unrolled in the source, so a weight row loaded once feeds four rows
//!   (batched GEMMs); the compiler merges the identical loads.
//! - Inlined comps are `static inline` helpers (`read_{j}`) evaluated per
//!   read, as in the interpreter.
//! - **Epilogue fusion**: a map that reads reductions point-for-point (and
//!   is their only reader) computes them in its own kernel and applies its
//!   body to the accumulators: the GRU step's six contractions and the
//!   gate math are one kernel.
//!
//! Kernels are emitted with node-numbered names, made
//! [canonical](crate::tensor::wgsl::canonical_source), and deduplicated: the same
//! comp in two layers is one C function. Buffers are static arrays
//! (aligned, allocated once) laid out by [`TensorGraph::plan_memory`];
//! `tang_run()` calls the kernels in plan order. Nothing allocates per
//! step. 16-bit storage is `__fp16`/`_Float16` (f16) or `uint16_t`
//! (bf16): loads widen to f32, arithmetic stays f32, an f16 accumulator
//! is rounded after every combine.
//!
//! Interface of the generated unit (all `tang_*`): `tang_run`,
//! `tang_n_inputs`, `tang_input(k)`, `tang_input_len(k)`,
//! `tang_input_format(k)` (0 f32, 1 f16, 2 bf16) and the same for outputs.
//! Callers write inputs in place and read outputs in place.
use std::collections::HashSet;
use std::fmt::Write as _;

use super::memplan::MemoryPlan;
use super::wgsl::{canonical_source, BufferSpec, WgslKernel, WgslProgram};
use super::{fixed, numel, Arg, Comp, Form, Format, Op, TensorGraph, TensorId};
use tang_expr::{ExprGraph, ExprId, Node};

/// A program lowered to C.
pub struct CProgram {
    /// Kernels (each `source` the canonical C of one kernel, entry
    /// `main0`), buffers and aliases, in the shape every backend shares.
    pub program: WgslProgram,
    /// Slots the buffers live in and the kernel order.
    pub plan: MemoryPlan,
    /// The whole translation unit.
    pub source: String,
    /// Program buffer index per output.
    pub outputs: Vec<usize>,
    /// Number of distinct C kernel functions after deduplication.
    pub unique_kernels: usize,
}

/// Accumulator floats per register block, `JB · RB · width ≤` this: 64
/// (16 NEON registers) without a row block, 128 with one (measured on
/// the Cortex-X3 with gcc 11: the batch-64 GRU is 1.34 ms at 128, 1.5–1.8
/// at 64, 1.66 at 32; batch 1 is fastest at 64). `$TANG_C_ACC_FLOATS`
/// overrides both, for tuning.
fn acc_floats(rb: usize) -> usize {
    std::env::var("TANG_C_ACC_FLOATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if rb > 1 { 128 } else { 64 })
}

/// Partial accumulators per point in lane mode (one f32 NEON vector).
const VL: usize = 4;
/// Chunk of a map's vector dim (no accumulators to keep in registers).
const MAP_CHUNK: usize = 256;

impl TensorGraph {
    /// Lower `outputs` to one C translation unit (see the module docs).
    /// The graph must be bound.
    pub fn to_c(&self, outputs: &[TensorId]) -> CProgram {
        assert!(self.is_bound(), "to_c needs a bound graph");
        // An inlined reduction is recomputed at every read. GPU tiles
        // stage it per chunk; a CPU loop nest would redo the whole sum per
        // point (per reduce step, inside another reduction), so every
        // inlined reduction that is not a fused epilogue's is materialized.
        let live = self.live(outputs);
        let out_storage: HashSet<usize> = outputs.iter().map(|o| self.storage(*o).n()).collect();
        let fused: HashSet<usize> = self
            .fused_reductions(&live, &out_storage)
            .into_iter()
            .flatten()
            .collect();
        let mut g = self.clone();
        for (i, node) in g.nodes.iter_mut().enumerate() {
            let reduces = matches!(&node.op, Op::Comp(c) if c.monoid.is_some());
            if live[i] && node.inline && reduces && !fused.contains(&i) {
                node.inline = false;
            }
        }
        g.to_c_scheduled(outputs)
    }

    fn to_c_scheduled(&self, outputs: &[TensorId]) -> CProgram {
        let live = self.live(outputs);
        let n = self.nodes.len();
        let out_storage: HashSet<usize> = outputs.iter().map(|o| self.storage(*o).n()).collect();
        let fused = self.fused_reductions(&live, &out_storage);
        let fused_any: HashSet<usize> = fused.iter().flatten().copied().collect();
        let needs_buffer = |i: usize| {
            live[i]
                && !self.nodes[i].inline
                && !fused_any.contains(&i)
                && !matches!(self.nodes[i].op, Op::Iota(_) | Op::Reshape(_))
        };
        let aliases: Vec<(TensorId, TensorId)> = (0..n)
            .filter(|&i| live[i] && matches!(self.nodes[i].op, Op::Reshape(_)))
            .map(|i| (TensorId(i as u32), self.storage(TensorId(i as u32))))
            .collect();
        let buffers: Vec<BufferSpec> = (0..n)
            .filter(|&i| needs_buffer(i))
            .map(|i| {
                let (len, input) = match &self.nodes[i].op {
                    Op::Input(k) => (numel(&self.nodes[i].shape), Some(*k)),
                    Op::Comp(c) => (numel(c.points()) * c.width(), None),
                    _ => unreachable!(),
                };
                BufferSpec {
                    node: TensorId(i as u32),
                    len,
                    input,
                    format: self.nodes[i].precision.storage,
                }
            })
            .collect();
        let kernels: Vec<WgslKernel> = (0..n)
            .filter(|&i| needs_buffer(i) && self.comp_at(i).is_some())
            .map(|i| {
                let (source, reads) = self.c_kernel(i, &fused[i]);
                let points = numel(self.comp_at(i).unwrap().points()) as u32;
                WgslKernel {
                    source: canonical_source(&source),
                    node: TensorId(i as u32),
                    reads: reads.into_iter().map(|r| TensorId(r as u32)).collect(),
                    points,
                    count: points,
                    workgroup_size: 1,
                }
            })
            .collect();
        let program = WgslProgram {
            kernels,
            buffers,
            aliases,
        };
        let plan = self.plan_memory(&program, outputs, &[]);
        let buf_of = |t: TensorId| {
            let s = program.storage(t);
            program
                .buffers
                .iter()
                .position(|b| b.node == s)
                .expect("output has a buffer")
        };
        let outs: Vec<usize> = outputs.iter().map(|o| buf_of(*o)).collect();
        let (source, unique_kernels) = c_unit(&program, &plan, &outs, &|t| buf_of(t));
        CProgram {
            program,
            plan,
            source,
            outputs: outs,
            unique_kernels,
        }
    }

    /// For every live materialized map: the reductions it computes in its
    /// own kernel. A reduction qualifies when it is not an output (inlined
    /// or not: `schedule_auto` inlines a GPU epilogue's reduction, which
    /// on the CPU is the same fusion), read by nothing but direct args of that map, over the
    /// same points, at the map's own coordinates (tuple slots constant).
    fn fused_reductions(&self, live: &[bool], out_storage: &HashSet<usize>) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in (0..n).filter(|&i| live[i]) {
            for d in self.deps(i) {
                if !readers[d].contains(&i) {
                    readers[d].push(i);
                }
            }
        }
        let mut fused = vec![Vec::new(); n];
        for m in (0..n).filter(|&m| live[m] && !self.nodes[m].inline) {
            let Some(cm) = self.comp_at(m) else { continue };
            if cm.monoid.is_some() {
                continue;
            }
            let np = cm.points().len();
            let pts = fixed(cm.points());
            for a in &cm.args {
                let r = a.src.n();
                if fused[m].contains(&r) || out_storage.contains(&r) {
                    continue;
                }
                let Some(cr) = self.comp_at(r) else { continue };
                if cr.monoid.is_none() || readers[r] != [m] || cr.points() != cm.points() {
                    continue;
                }
                let pointwise = cm.args.iter().filter(|b| b.src.n() == r).all(|b| {
                    !b.view.guard
                        && b.view.forms.len() == cr.out.len()
                        && b.view.forms[..np].iter().enumerate().all(|(d, f)| {
                            // the identity, up to coefficients on size-1 dims
                            f.offset == 0
                                && (0..np).all(|e| pts[e] == 1 || f.coefs[e] == (e == d) as i64)
                        })
                        && b.view.forms[np..]
                            .iter()
                            .all(|f| f.coefs.iter().all(|c| *c == 0))
                });
                if pointwise {
                    fused[m].push(r);
                }
            }
        }
        fused
    }

    /// Buffer nodes kernel `root` reads (sorted), fusing `fused`.
    fn c_reads(&self, root: usize, fused: &[usize]) -> Vec<usize> {
        let mut closure = self.inline_closure(root);
        for &r in fused {
            closure.extend(self.inline_closure(r));
        }
        let mut reads: Vec<usize> = closure
            .into_iter()
            .filter(|&j| j != root && !fused.contains(&j))
            .filter(|&j| match &self.nodes[j].op {
                Op::Input(_) => true,
                Op::Comp(_) => !self.nodes[j].inline,
                Op::Iota(_) | Op::Reshape(_) => false,
            })
            .map(|j| self.storage(TensorId(j as u32)).n())
            .collect();
        reads.sort_unstable();
        reads.dedup();
        reads
    }

    /// C source of the kernel materializing `root` (entry `main0`,
    /// params `b{read}…, b{root}`), and its reads.
    fn c_kernel(&self, root: usize, fused: &[usize]) -> (String, Vec<usize>) {
        let reads = self.c_reads(root, fused);
        let mut kx = Kx {
            g: self,
            reads: reads.clone(),
            helpers: String::new(),
            helped: HashSet::new(),
            tmp: 0,
        };
        let cm = self.comp_at(root).unwrap();
        let pts = fixed(cm.points());
        let np = pts.len();
        // reductions computed here: the root itself or the fused ones
        let accs: Vec<usize> = if cm.monoid.is_some() {
            vec![root]
        } else {
            fused.to_vec()
        };
        let wmax = accs
            .iter()
            .map(|&a| self.comp_at(a).unwrap().width())
            .max()
            .unwrap_or(1);

        // vector dim: most unit-stride buffer args, ties to the innermost
        let score = |d: usize| -> i64 {
            if pts[d] <= 1 {
                return -1;
            }
            let comps: Vec<&Comp> = if accs.is_empty() {
                vec![cm]
            } else {
                accs.iter().map(|&a| self.comp_at(a).unwrap()).collect()
            };
            comps
                .iter()
                .flat_map(|c| &c.args)
                .filter(|a| {
                    !matches!(self.nodes[a.src.n()].op, Op::Iota(_))
                        && !self.nodes[a.src.n()].inline
                })
                .filter(|a| self.flat_form(a).coefs.get(d) == Some(&1))
                .count() as i64
        };
        let jdim = (0..np)
            .rev()
            .max_by_key(|&d| score(d))
            .filter(|&d| pts[d] > 1);
        // lane mode: no point dim is contiguous, but the innermost reduce
        // dim is (`Σ_k x[i,k]·W[j,k]`, weights stored [out, in]): the
        // innermost loop runs over `VL` consecutive reduce steps into `VL`
        // partial accumulators per point, folded by the monoid at the end.
        let lane_ok = |a: usize| {
            let ca = self.comp_at(a).unwrap();
            let red = fixed(&ca.red);
            let Some(&last) = red.last() else {
                return false;
            };
            let d = np + red.len() - 1;
            last.is_multiple_of(VL)
                && ca.args.iter().any(|x| {
                    !matches!(self.nodes[x.src.n()].op, Op::Iota(_))
                        && !self.nodes[x.src.n()].inline
                        && self.flat_form(x).coefs.get(d) == Some(&1)
                })
        };
        let lane =
            !accs.is_empty() && jdim.map_or(0, score) == 0 && accs.iter().all(|&a| lane_ok(a));
        let rdim = if accs.is_empty() {
            None
        } else {
            (0..np)
                .rev()
                .find(|&d| Some(d) != jdim && pts[d] >= 4 && pts[d].is_multiple_of(4))
        };
        let rb = if rdim.is_some() { 4 } else { 1 };
        let jb = jdim.map_or(1, |d| {
            let cap = if accs.is_empty() {
                MAP_CHUNK
            } else if lane {
                (acc_floats(rb) / (rb * wmax * VL)).clamp(1, 4)
            } else {
                (acc_floats(rb) / (rb * wmax)).max(1)
            };
            (1..=cap.min(pts[d]))
                .rev()
                .find(|b| pts[d].is_multiple_of(*b))
                .unwrap()
        });

        let mut s = String::new();
        let mut ind = String::from("    ");
        let mut close = 0;
        for d in (0..np).filter(|&d| Some(d) != jdim && Some(d) != rdim) {
            writeln!(s, "{ind}for (int p{d} = 0; p{d} < {}; p{d}++) {{", pts[d]).unwrap();
            ind.push_str("    ");
            close += 1;
        }
        if let Some(d) = rdim {
            writeln!(s, "{ind}for (int q = 0; q < {}; q += {rb}) {{", pts[d]).unwrap();
            ind.push_str("    ");
            close += 1;
        }
        if let Some(d) = jdim {
            writeln!(s, "{ind}for (int j0 = 0; j0 < {}; j0 += {jb}) {{", pts[d]).unwrap();
            ind.push_str("    ");
            close += 1;
        }
        // coordinates of the point dims in row-block copy `k`, with `jj`
        // the offset in the vector chunk
        let point_vars = |k: usize, jj: &str| -> Vec<String> {
            (0..np)
                .map(|d| {
                    if Some(d) == jdim {
                        format!("(j0 + {jj})")
                    } else if Some(d) == rdim {
                        format!("(q + {k})")
                    } else {
                        format!("p{d}")
                    }
                })
                .collect()
        };

        for (ai, &a) in accs.iter().enumerate() {
            let ca = self.comp_at(a).unwrap();
            let m = ca.monoid.as_ref().unwrap();
            let w = ca.width();
            let accum16 = self.nodes[a].precision.accum == Format::F16;
            let red = fixed(&ca.red);
            let nr = red.len();
            let space: Vec<usize> = pts.iter().chain(&red).copied().collect();
            // `acc = combine(acc, elem)` for one accumulator set
            let combine =
                |kx: &mut Kx, s: &mut String, ind: &str, accv: &[String], elem: &[String]| {
                    let p = kx.fresh();
                    let outs = emit_scalar(s, ind, &m.combine, &m.outs, &p, &|v| {
                        let v = v as usize;
                        if v < w {
                            accv[v].clone()
                        } else {
                            elem[v - w].clone()
                        }
                    });
                    let p2 = kx.fresh();
                    for (v, o) in outs.iter().enumerate() {
                        writeln!(s, "{ind}const float {p2}{v} = {o};").unwrap();
                    }
                    for (v, acc) in accv.iter().enumerate() {
                        if accum16 {
                            writeln!(s, "{ind}{acc} = TANG_RH({p2}{v});").unwrap();
                        } else {
                            writeln!(s, "{ind}{acc} = {p2}{v};").unwrap();
                        }
                    }
                };
            let lanes = if lane {
                format!("[{VL}]")
            } else {
                String::new()
            };
            let accn = if lane {
                format!("accl{ai}")
            } else {
                format!("acc{ai}")
            };
            writeln!(s, "{ind}float {accn}[{w}][{rb}][{jb}]{lanes};").unwrap();
            writeln!(s, "{ind}for (int k = 0; k < {rb}; k++) {{").unwrap();
            writeln!(s, "{ind}    for (int jj = 0; jj < {jb}; jj++) {{").unwrap();
            for (v, id) in m.identity.iter().enumerate() {
                if lane {
                    writeln!(
                        s,
                        "{ind}        for (int v = 0; v < {VL}; v++) {accn}[{v}][k][jj][v] = {};",
                        lit(*id)
                    )
                    .unwrap();
                } else {
                    writeln!(s, "{ind}        {accn}[{v}][k][jj] = {};", lit(*id)).unwrap();
                }
            }
            writeln!(s, "{ind}    }}\n{ind}}}").unwrap();
            let mut rind = ind.clone();
            for (d, r) in red.iter().enumerate() {
                if lane && d == nr - 1 {
                    writeln!(s, "{rind}for (int rl = 0; rl < {}; rl++) {{", r / VL).unwrap();
                } else {
                    writeln!(s, "{rind}for (int r{d} = 0; r{d} < {r}; r{d}++) {{").unwrap();
                }
                rind.push_str("    ");
            }
            writeln!(s, "{rind}#pragma GCC ivdep").unwrap();
            if lane {
                writeln!(s, "{rind}for (int v = 0; v < {VL}; v++) {{").unwrap();
            } else {
                writeln!(s, "{rind}for (int jj = 0; jj < {jb}; jj++) {{").unwrap();
            }
            let bind = format!("{rind}    ");
            let copies: Vec<(usize, String, String)> = if lane {
                (0..rb)
                    .flat_map(|k| {
                        (0..jb).map(move |j| (k, j.to_string(), format!("[{k}][{j}][v]")))
                    })
                    .collect()
            } else {
                (0..rb)
                    .map(|k| (k, "jj".to_string(), format!("[{k}][jj]")))
                    .collect()
            };
            for (k, jj, at) in &copies {
                let mut vars = point_vars(*k, jj);
                vars.extend((0..nr).map(|d| {
                    if lane && d == nr - 1 {
                        format!("({VL} * rl + v)")
                    } else {
                        format!("r{d}")
                    }
                }));
                let elem = kx.body(&mut s, &bind, ca, &vars, &space, &|_| None);
                let accv: Vec<String> = (0..w).map(|v| format!("{accn}[{v}]{at}")).collect();
                combine(&mut kx, &mut s, &bind, &accv, &elem);
            }
            writeln!(s, "{rind}}}").unwrap();
            for _ in &red {
                rind.truncate(rind.len() - 4);
                writeln!(s, "{rind}}}").unwrap();
            }
            if lane {
                // fold the lanes
                writeln!(s, "{ind}float acc{ai}[{w}][{rb}][{jb}];").unwrap();
                writeln!(s, "{ind}for (int k = 0; k < {rb}; k++) {{").unwrap();
                writeln!(s, "{ind}    for (int jj = 0; jj < {jb}; jj++) {{").unwrap();
                let i3 = format!("{ind}        ");
                for v in 0..w {
                    writeln!(s, "{i3}acc{ai}[{v}][k][jj] = {accn}[{v}][k][jj][0];").unwrap();
                }
                writeln!(s, "{i3}for (int v = 1; v < {VL}; v++) {{").unwrap();
                let accv: Vec<String> = (0..w).map(|v| format!("acc{ai}[{v}][k][jj]")).collect();
                let elem: Vec<String> = (0..w).map(|v| format!("{accn}[{v}][k][jj][v]")).collect();
                combine(&mut kx, &mut s, &format!("{i3}    "), &accv, &elem);
                writeln!(s, "{i3}}}\n{ind}    }}\n{ind}}}").unwrap();
            }
        }

        // epilogue: the root's values per point
        let w = cm.width();
        let fmt = self.nodes[root].precision.storage;
        let strides = strides(&pts);
        writeln!(s, "{ind}#pragma GCC ivdep").unwrap();
        writeln!(s, "{ind}for (int jj = 0; jj < {jb}; jj++) {{").unwrap();
        let eind = format!("{ind}    ");
        for k in 0..rb {
            let vars = point_vars(k, "jj");
            let vals: Vec<String> = if cm.monoid.is_some() {
                (0..w).map(|v| format!("acc0[{v}][{k}][jj]")).collect()
            } else {
                let acc_of = |a: &Arg| -> Option<String> {
                    let ai = accs.iter().position(|&r| r == a.src.n())?;
                    let cr = self.comp_at(a.src.n()).unwrap();
                    let tup = fixed(&cr.out[np..]);
                    let slot = a.view.forms[np..]
                        .iter()
                        .zip(strides_of(&tup))
                        .map(|(f, st)| f.offset as usize * st)
                        .sum::<usize>();
                    Some(format!("acc{ai}[{slot}][{k}][jj]"))
                };
                kx.body(&mut s, &eind, cm, &vars, &pts, &acc_of)
            };
            let flat: Vec<String> = (0..np)
                .filter(|&d| pts[d] > 1)
                .map(|d| format!("{} * {}", strides[d] * w, vars[d]))
                .collect();
            let flat = if flat.is_empty() {
                "0".to_string()
            } else {
                flat.join(" + ")
            };
            for (v, val) in vals.iter().enumerate() {
                writeln!(s, "{eind}b{root}[{flat} + {v}] = {};", store(fmt, val)).unwrap();
            }
        }
        writeln!(s, "{ind}}}").unwrap();
        for _ in 0..close {
            ind.truncate(ind.len() - 4);
            writeln!(s, "{ind}}}").unwrap();
        }

        let mut params: Vec<String> = reads
            .iter()
            .map(|&b| {
                format!(
                    "const {}* restrict b{b}",
                    elem_ty(self.nodes[b].precision.storage)
                )
            })
            .collect();
        params.push(format!("{}* restrict b{root}", elem_ty(fmt)));
        let src = format!(
            "{}static TANG_KERNEL void main0({}) {{\n{s}}}\n",
            kx.helpers,
            params.join(", ")
        );
        (src, reads)
    }

    /// The flat buffer index an arg reads, as a form over its comp's space.
    fn flat_form(&self, a: &Arg) -> Form {
        let shape = fixed(&self.nodes[a.src.n()].shape);
        let rank = a.view.forms.first().map_or(0, |f| f.coefs.len());
        let mut flat = Form::zero(rank);
        for (f, st) in a.view.forms.iter().zip(strides_of(&shape)) {
            let f = f.scaled(st as i64);
            flat.offset += f.offset;
            for (o, c) in flat.coefs.iter_mut().zip(&f.coefs) {
                *o += c;
            }
        }
        flat
    }
}

/// Emission state of one kernel.
struct Kx<'a> {
    g: &'a TensorGraph,
    /// Buffer nodes the kernel reads: the leading params of every function.
    reads: Vec<usize>,
    helpers: String,
    helped: HashSet<usize>,
    tmp: usize,
}

impl Kx<'_> {
    fn fresh(&mut self) -> String {
        self.tmp += 1;
        format!("t{}_", self.tmp)
    }

    fn call_prefix(&self) -> String {
        self.reads.iter().map(|b| format!("b{b}, ")).collect()
    }

    /// Bind every arg of `c` at `vars` (one C expression per space dim of
    /// extent `space`) and emit its body; returns the body outputs.
    /// `acc_of` replaces an arg's read (the fused reductions).
    fn body(
        &mut self,
        s: &mut String,
        ind: &str,
        c: &Comp,
        vars: &[String],
        space: &[usize],
        acc_of: &dyn Fn(&Arg) -> Option<String>,
    ) -> Vec<String> {
        let pa = self.fresh();
        let used: HashSet<u16> = c
            .body
            .live_set(&c.body_outs)
            .into_iter()
            .filter_map(|i| match c.body.nodes_slice()[i] {
                Node::Var(v) => Some(v),
                _ => None,
            })
            .collect();
        for (j, a) in c.args.iter().enumerate() {
            if !used.contains(&(j as u16)) {
                continue;
            }
            let v = match acc_of(a) {
                Some(v) => v,
                None => self.read_arg(a, vars, space),
            };
            writeln!(s, "{ind}const float {pa}{j} = {v};").unwrap();
        }
        let pb = self.fresh();
        emit_scalar(s, ind, &c.body, &c.body_outs, &pb, &|v| format!("{pa}{v}"))
    }

    /// The value of `a` at `vars` as a C expression.
    fn read_arg(&mut self, a: &Arg, vars: &[String], space: &[usize]) -> String {
        let g = self.g;
        let src = a.src.n();
        let shape = fixed(&g.nodes[src].shape);
        let coords: Vec<String> = a.view.forms.iter().map(|f| form_str(f, vars)).collect();
        let val = match &g.nodes[src].op {
            Op::Iota(d) => format!("(float)({})", coords[*d]),
            Op::Comp(_) if g.nodes[src].inline => {
                self.helper(src);
                format!("read_{src}({}{})", self.call_prefix(), coords.join(", "))
            }
            _ => {
                let b = g.storage(a.src).n();
                let idx = form_str(&g.flat_form(a), vars);
                load(g.nodes[b].precision.storage, &format!("b{b}[{idx}]"))
            }
        };
        if !a.view.guard {
            return val;
        }
        let conds: Vec<String> = a
            .view
            .forms
            .iter()
            .zip(&shape)
            .zip(&coords)
            .filter_map(|((f, &n), c)| {
                let (lo, hi) = f.range(space);
                let mut cs = Vec::new();
                if lo < 0 {
                    cs.push(format!("{c} >= 0"));
                }
                if hi >= n as i64 {
                    cs.push(format!("{c} < {n}"));
                }
                (!cs.is_empty()).then(|| cs.join(" && "))
            })
            .collect();
        if conds.is_empty() {
            val
        } else {
            format!("(({}) ? {val} : 0.0f)", conds.join(" && "))
        }
    }

    /// `static inline float read_{s}(reads…, c0…)`: inlined comp `s` at
    /// full-shape coordinates `c…`.
    fn helper(&mut self, s_node: usize) {
        if !self.helped.insert(s_node) {
            return;
        }
        let g = self.g;
        let c = g.comp_at(s_node).unwrap();
        let pts = fixed(c.points());
        let red = fixed(&c.red);
        let np = pts.len();
        let rank = c.out.len();
        let w = c.width();
        let mut s = String::new();
        let mut vars: Vec<String> = (0..np).map(|d| format!("c{d}")).collect();
        vars.extend((0..red.len()).map(|d| format!("r{d}")));
        let space: Vec<usize> = pts.iter().chain(&red).copied().collect();
        let vals: Vec<String> = match &c.monoid {
            None => self.body(&mut s, "    ", c, &vars, &space, &|_| None),
            Some(m) => {
                for (v, id) in m.identity.iter().enumerate() {
                    writeln!(s, "    float a{v} = {};", lit(*id)).unwrap();
                }
                let mut ind = String::from("    ");
                for (d, r) in red.iter().enumerate() {
                    writeln!(s, "{ind}for (int r{d} = 0; r{d} < {r}; r{d}++) {{").unwrap();
                    ind.push_str("    ");
                }
                let elem = self.body(&mut s, &ind, c, &vars, &space, &|_| None);
                let p = self.fresh();
                let outs = emit_scalar(&mut s, &ind, &m.combine, &m.outs, &p, &|v| {
                    let v = v as usize;
                    if v < w {
                        format!("a{v}")
                    } else {
                        elem[v - w].clone()
                    }
                });
                let p2 = self.fresh();
                for (v, o) in outs.iter().enumerate() {
                    writeln!(s, "{ind}const float {p2}{v} = {o};").unwrap();
                }
                let accum16 = g.nodes[s_node].precision.accum == Format::F16;
                for v in 0..w {
                    if accum16 {
                        writeln!(s, "{ind}a{v} = TANG_RH({p2}{v});").unwrap();
                    } else {
                        writeln!(s, "{ind}a{v} = {p2}{v};").unwrap();
                    }
                }
                for _ in &red {
                    ind.truncate(ind.len() - 4);
                    writeln!(s, "{ind}}}").unwrap();
                }
                (0..w).map(|v| format!("a{v}")).collect()
            }
        };
        if w == 1 {
            writeln!(s, "    return {};", vals[0]).unwrap();
        } else {
            let tup = fixed(&c.out[np..]);
            let slot: Vec<String> = strides_of(&tup)
                .into_iter()
                .enumerate()
                .map(|(d, st)| format!("{st} * c{}", np + d))
                .collect();
            writeln!(s, "    const float v[{w}] = {{{}}};", vals.join(", ")).unwrap();
            writeln!(s, "    return v[{}];", slot.join(" + ")).unwrap();
        }
        let mut params: Vec<String> = self
            .reads
            .iter()
            .map(|&b| {
                format!(
                    "const {}* restrict b{b}",
                    elem_ty(g.nodes[b].precision.storage)
                )
            })
            .collect();
        params.extend((0..rank).map(|d| format!("int c{d}")));
        writeln!(
            self.helpers,
            "static inline float read_{s_node}({}) {{\n{s}}}",
            params.join(", ")
        )
        .unwrap();
    }
}

/// Row-major strides of `shape`.
fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        st[d] = st[d + 1] * shape[d + 1];
    }
    st
}

fn strides(shape: &[usize]) -> Vec<usize> {
    strides_of(shape)
}

/// `offset + Σ coef·var` as C.
fn form_str(f: &Form, vars: &[String]) -> String {
    let mut s = String::new();
    for (c, v) in f.coefs.iter().zip(vars) {
        match *c {
            0 => continue,
            1 => write!(s, "{}{v}", if s.is_empty() { "" } else { " + " }).unwrap(),
            -1 => write!(s, "{}{v}", if s.is_empty() { "-" } else { " - " }).unwrap(),
            c if c < 0 => {
                write!(s, "{}{} * {v}", if s.is_empty() { "-" } else { " - " }, -c).unwrap()
            }
            c => write!(s, "{}{c} * {v}", if s.is_empty() { "" } else { " + " }).unwrap(),
        }
    }
    if f.offset != 0 || s.is_empty() {
        if s.is_empty() {
            write!(s, "{}", f.offset).unwrap();
        } else if f.offset < 0 {
            write!(s, " - {}", -f.offset).unwrap();
        } else {
            write!(s, " + {}", f.offset).unwrap();
        }
    }
    format!("({s})")
}

fn elem_ty(f: Format) -> &'static str {
    match f {
        Format::F32 => "float",
        Format::F16 => "tang_f16",
        Format::BF16 => "uint16_t",
        Format::E4M3(_) | Format::E5M2(_) | Format::Fp8Accum => panic!("fp8 storage lowers only on sm90 CUDA tensor-core tiles"),
    }
}

fn load(f: Format, e: &str) -> String {
    match f {
        Format::F32 => e.to_string(),
        Format::F16 => format!("(float){e}"),
        Format::BF16 => format!("tang_b2f({e})"),
        Format::E4M3(_) | Format::E5M2(_) | Format::Fp8Accum => panic!("fp8 storage lowers only on sm90 CUDA tensor-core tiles"),
    }
}

fn store(f: Format, e: &str) -> String {
    match f {
        Format::F32 => e.to_string(),
        Format::F16 => format!("(tang_f16)({e})"),
        Format::BF16 => format!("tang_f2b({e})"),
        Format::E4M3(_) | Format::E5M2(_) | Format::Fp8Accum => panic!("fp8 storage lowers only on sm90 CUDA tensor-core tiles"),
    }
}

/// f32 literal; ±∞ as ±f32::MAX like the WGSL lowering, so the
/// online-softmax identity `exp(m − m)` stays finite.
fn lit(v: f64) -> String {
    if v.is_nan() {
        return "NAN".into();
    }
    if v >= f32::MAX as f64 {
        return "3.4028235e38f".into();
    }
    if v <= f32::MIN as f64 {
        return "(-3.4028235e38f)".into();
    }
    let s = format!("{:?}", v as f32);
    let s = if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    };
    if v < 0.0 {
        format!("({s}f)")
    } else {
        format!("{s}f")
    }
}

/// `const float {prefix}{i} = …;` for the live nodes of `g`; returns the
/// expressions for `outs`.
fn emit_scalar(
    s: &mut String,
    ind: &str,
    g: &ExprGraph,
    outs: &[ExprId],
    prefix: &str,
    var: &dyn Fn(u16) -> String,
) -> Vec<String> {
    let mut live: Vec<usize> = g.live_set(outs).into_iter().collect();
    live.sort_unstable();
    let nodes = g.nodes_slice();
    let e = |id: &ExprId| match nodes[id.index() as usize] {
        Node::Var(v) => var(v),
        Node::Lit(b) => lit(f64::from_bits(b)),
        _ => format!("{prefix}{}", id.index()),
    };
    for i in live {
        let rhs = match &nodes[i] {
            Node::Var(_) | Node::Lit(_) => continue,
            Node::Add(a, b) => format!("{} + {}", e(a), e(b)),
            Node::Mul(a, b) => format!("{} * {}", e(a), e(b)),
            Node::Neg(a) => format!("-{}", e(a)),
            Node::Recip(a) => format!("1.0f / {}", e(a)),
            Node::Sqrt(a) => format!("sqrtf({})", e(a)),
            Node::Sin(a) => format!("sinf({})", e(a)),
            Node::Atan2(a, b) => format!("atan2f({}, {})", e(a), e(b)),
            Node::Exp2(a) => format!("exp2f({})", e(a)),
            Node::Log2(a) => format!("log2f({})", e(a)),
            Node::Select(c, a, b) => format!("({} > 0.0f ? {} : {})", e(c), e(a), e(b)),
        };
        writeln!(s, "{ind}const float {prefix}{i} = {rhs};").unwrap();
    }
    outs.iter().map(e).collect()
}

const C_PRELUDE: &str = r#"// Auto-generated by tang-compile (to_c): a tensor-IR program as plain C.
// Build: cc -O3 -march=native -ffp-contract=fast -fno-strict-aliasing -c
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__aarch64__) || defined(__arm__)
typedef __fp16 tang_f16;
#else
typedef _Float16 tang_f16;
#endif
#define TANG_RH(x) ((float)(tang_f16)(x))
#define TANG_KERNEL __attribute__((noinline))
static inline float tang_b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t tang_f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
"#;

/// The translation unit: prelude, slots, deduplicated kernels, entry
/// point and the input/output tables.
fn c_unit(
    prog: &WgslProgram,
    plan: &MemoryPlan,
    outs: &[usize],
    buf_of: &dyn Fn(TensorId) -> usize,
) -> (String, usize) {
    let mut s = String::from(C_PRELUDE);
    for (i, w) in plan.slot_words.iter().enumerate() {
        writeln!(s, "static float S{i}[{w}] __attribute__((aligned(64)));").unwrap();
    }
    let ptr = |b: usize| -> String {
        let ty = elem_ty(prog.buffers[b].format);
        format!("(({ty}*)S{})", plan.slot[b])
    };
    let mut unique: Vec<&str> = Vec::new();
    let mut calls = String::new();
    let mut calls_n = 0usize;
    for &ki in &plan.order {
        let k = &prog.kernels[ki];
        let u = unique
            .iter()
            .position(|u| *u == k.source)
            .unwrap_or_else(|| {
                unique.push(&k.source);
                unique.len() - 1
            });
        let out = buf_of(k.node);
        let mut args: Vec<String> = k.reads.iter().map(|t| ptr(buf_of(*t))).collect();
        for t in &k.reads {
            assert_ne!(
                plan.slot[buf_of(*t)],
                plan.slot[out],
                "kernel reads its own output slot"
            );
        }
        args.push(ptr(out));
        let i = calls_n;
        calls_n += 1;
        writeln!(
            calls,
            "    TANG_PROF_BEGIN({i});\n    k{u}({});\n    TANG_PROF_END({i});",
            args.join(", ")
        )
        .unwrap();
    }
    for (u, src) in unique.iter().enumerate() {
        let src = src
            .replace("read_", &format!("k{u}_read_"))
            .replace("main0", &format!("k{u}"));
        writeln!(s, "{src}").unwrap();
    }
    // -DTANG_PROFILE: nanoseconds per kernel call, summed (tang_prof_ns)
    writeln!(
        s,
        "#ifdef TANG_PROFILE\n#include <time.h>\ndouble tang_prof_ns[{n}];\n\
         int tang_n_kernels(void) {{ return {n}; }}\n\
         static double tang_prof_t;\n\
         static inline double tang_now(void) {{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }}\n\
         #define TANG_PROF_BEGIN(i) tang_prof_t = tang_now()\n\
         #define TANG_PROF_END(i) tang_prof_ns[i] += tang_now() - tang_prof_t\n\
         #else\n#define TANG_PROF_BEGIN(i)\n#define TANG_PROF_END(i)\n#endif",
        n = calls_n.max(1)
    )
    .unwrap();
    writeln!(s, "void tang_run(void) {{\n{calls}}}").unwrap();
    let fmt_code = |f: Format| match f {
        Format::F32 => 0,
        Format::F16 => 1,
        Format::BF16 => 2,
        _ => panic!("fp8 storage lowers only on sm90 CUDA tensor-core tiles"),
    };
    let mut inputs: Vec<(usize, usize)> = prog
        .buffers
        .iter()
        .enumerate()
        .filter_map(|(b, spec)| spec.input.map(|k| (k, b)))
        .collect();
    inputs.sort_unstable();
    let table = |name: &str, bufs: &[usize]| -> String {
        let ptrs: Vec<String> = bufs
            .iter()
            .map(|&b| format!("(void*)S{}", plan.slot[b]))
            .collect();
        let lens: Vec<String> = bufs
            .iter()
            .map(|&b| prog.buffers[b].len.to_string())
            .collect();
        let fmts: Vec<String> = bufs
            .iter()
            .map(|&b| fmt_code(prog.buffers[b].format).to_string())
            .collect();
        let n = bufs.len();
        let pad = |v: Vec<String>| {
            if v.is_empty() {
                "0".to_string()
            } else {
                v.join(", ")
            }
        };
        format!(
            "static void* const tang_{name}_ptr[] = {{{}}};\n\
             static const size_t tang_{name}_lens[] = {{{}}};\n\
             static const int tang_{name}_fmts[] = {{{}}};\n\
             int tang_n_{name}s(void) {{ return {n}; }}\n\
             void* tang_{name}(int i) {{ return tang_{name}_ptr[i]; }}\n\
             size_t tang_{name}_len(int i) {{ return tang_{name}_lens[i]; }}\n\
             int tang_{name}_format(int i) {{ return tang_{name}_fmts[i]; }}\n",
            pad(ptrs),
            pad(lens),
            pad(fmts)
        )
    };
    // graph inputs never read by a live kernel have no buffer: skip-safe
    // because `input` indices are dense over the live inputs only if all
    // are live; callers index by graph input number.
    let n_in = inputs.iter().map(|(k, _)| k + 1).max().unwrap_or(0);
    assert!(
        inputs.iter().enumerate().all(|(i, (k, _))| i == *k) && inputs.len() == n_in,
        "every graph input must be live"
    );
    let in_bufs: Vec<usize> = inputs.iter().map(|(_, b)| *b).collect();
    s.push_str(&table("input", &in_bufs));
    s.push_str(&table("output", outs));
    (s, unique.len())
}

#[cfg(test)]
mod tests {
    use super::super::TensorGraph;
    use tang::Scalar;
    use tang_expr::ExprId;

    #[test]
    fn matmul_bias_relu_fuses_to_one_kernel() {
        let mut g = TensorGraph::new();
        let x = g.input(&[4, 8]);
        let w = g.input(&[8, 16]);
        let b = g.input(&[1, 16]);
        let y = g.matmul(x, w).unwrap();
        let bb = g.broadcast(b, &[4, 16]).unwrap();
        let z = g
            .map(&[y, bb], |v| {
                let s = v[0] + v[1];
                let zero = ExprId::from_f64(0.0);
                tang_expr::with_graph(|g| g.select(s, s, zero))
            })
            .unwrap();
        let (g, outs) = g.substitute_maps(&[z]);
        let p = g.to_c(&outs);
        assert_eq!(p.program.kernels.len(), 1);
        assert!(p.source.contains("void tang_run(void)"));
    }
}
