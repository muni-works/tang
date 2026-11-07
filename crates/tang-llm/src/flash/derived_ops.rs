//! Engine ops derived from the tensor IR (`TANG_CODER_DERIVED`): a graph and
//! its schedule, lowered by `tang-compile` to CUDA, compiled with NVRTC and
//! launched on the engine's own buffers.
//!
//! [`Op`] is the generic runner: the caller names which graph tensors are
//! *external* (bound to engine pointers at launch: every graph input, and the
//! outputs it wants), the runner allocates the rest (intermediates the
//! lowering materializes) once, and [`Op::launch`] enqueues the program's
//! kernels in order on a stream (capturable into the engine's CUDA graphs).
//!
//! The op builders below are the Coder-Next engine's small per-token ops,
//! written as the naive math (`docs/coder-derived.md` has the inventory).
//! Layouts are the engine's: row-major `[t][n]` f32 activations, `u32` expert
//! ids, f32 router weights.

use tang::Scalar;
use tang_compile::tensor::{
    Arg, Form, Format, IntMma, Monoid, Precision, Stage, TensorGraph, TensorId, View,
};
use tang_expr::ExprId;

/// The math of an op, before lowering: graph, external tensors in launch
/// pointer order, and the outputs to materialize.
pub struct Spec {
    pub g: TensorGraph,
    pub ext: Vec<TensorId>,
    pub outs: Vec<TensorId>,
    /// Run `schedule_auto` before lowering (off when the builder set the
    /// whole schedule itself).
    pub auto: bool,
}

/// One lowered kernel's source and bindings (device-free; the testable part).
pub struct Lowered {
    pub kernels: Vec<LKernel>,
    /// Scratch buffers in bytes (intermediates).
    pub scratch: Vec<usize>,
}

pub struct LKernel {
    pub source: String,
    pub args: Vec<Slot>,
    pub grid: u32,
    pub block: u32,
    pub smem: u32,
    pub points: u32,
}

/// Where a kernel parameter points: an external pointer (index into the
/// launch's pointer list) or a scratch buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Ext(usize),
    Scratch(usize),
}

fn define(src: &str, name: &str) -> Option<u32> {
    src.lines()
        .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
        .and_then(|v| v.trim().parse().ok())
}

impl Spec {
    /// Run the standard schedule and lower to CUDA.
    pub fn lower(mut self) -> Result<Lowered, String> {
        if self.auto {
            self.g.schedule_auto(&self.outs);
        }
        let p = self.g.to_cuda(&self.outs);
        // reshape aliases: a node whose bytes live in another node's buffer
        let resolve = |mut t: TensorId| {
            while let Some(&(_, b)) = p.aliases.iter().find(|(a, _)| *a == t) {
                t = b;
            }
            t
        };
        let ext: Vec<TensorId> = self.ext.iter().map(|&t| resolve(t)).collect();
        let mut scratch_ids: Vec<TensorId> = Vec::new();
        let mut scratch = Vec::new();
        let mut slot = |t: TensorId| -> Slot {
            let t = resolve(t);
            if let Some(i) = ext.iter().position(|&e| e == t) {
                return Slot::Ext(i);
            }
            if let Some(i) = scratch_ids.iter().position(|&e| e == t) {
                return Slot::Scratch(i);
            }
            scratch_ids.push(t);
            scratch.push(self.g.buffer_bytes(t).max(4));
            Slot::Scratch(scratch_ids.len() - 1)
        };
        let mut kernels = Vec::new();
        for k in &p.kernels {
            let mut args: Vec<Slot> = k.reads.iter().map(|&r| slot(r)).collect();
            args.push(slot(k.node));
            let (block, grid) = match (
                define(&k.source, "TANG_THREADS"),
                define(&k.source, "TANG_TILES"),
            ) {
                (Some(th), Some(ti)) => (th, ti),
                _ => (k.workgroup_size, k.count.div_ceil(k.workgroup_size.max(1))),
            };
            kernels.push(LKernel {
                source: k.source.clone(),
                args,
                grid,
                block,
                smem: define(&k.source, "TANG_SMEM").unwrap_or(0),
                points: k.points,
            });
        }
        Ok(Lowered { kernels, scratch })
    }
}

#[cfg(feature = "cuda")]
pub use gpu_impl::Op;

#[cfg(feature = "cuda")]
mod gpu_impl {
    use super::{Lowered, Slot};
    use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

    struct K {
        f: cudarc::driver::sys::CUfunction,
        _m: Module,
        args: Vec<Slot>,
        grid: u32,
        block: u32,
        smem: u32,
        points: u32,
    }

    /// A lowered op loaded on the device, with its scratch.
    pub struct Op {
        ks: Vec<K>,
        scratch: Vec<DevBuf>,
        pub n_ext: usize,
    }

    impl Op {
        pub fn load(gpu: &Gpu, l: Lowered, n_ext: usize) -> anyhow::Result<Self> {
            use anyhow::anyhow;
            let mut ks = Vec::new();
            for k in l.kernels {
                let m = gpu.module(&k.source).map_err(|e| anyhow!("{e}"))?;
                let f = m
                    .func(tang_compile::tensor::cuda::CUDA_ENTRY)
                    .map_err(|e| anyhow!("{e}"))?;
                if k.smem > 48 * 1024 {
                    unsafe {
                        gpu::check(
                            cudarc::driver::sys::cuFuncSetAttribute(
                                f,
                                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                k.smem as i32,
                            ),
                            "cuFuncSetAttribute",
                        )
                        .map_err(|e| anyhow!("{e}"))?;
                    }
                }
                ks.push(K {
                    f,
                    _m: m,
                    args: k.args,
                    grid: k.grid,
                    block: k.block,
                    smem: k.smem,
                    points: k.points,
                });
            }
            let scratch = l
                .scratch
                .iter()
                .map(|&b| DevBuf::zeroed(b).map_err(|e| anyhow!("{e}")))
                .collect::<anyhow::Result<_>>()?;
            Ok(Op { ks, scratch, n_ext })
        }

        pub fn kernels(&self) -> usize {
            self.ks.len()
        }

        /// Enqueue every kernel on `s`; `ext[i]` is external tensor `i`'s
        /// device address.
        pub fn launch(&self, s: &Stream, ext: &[u64]) -> anyhow::Result<()> {
            self.launch_range(s, ext, 0..self.ks.len())
        }

        /// Kernel `i`'s launch shape `(grid, block, smem)`.
        pub fn shape(&self, i: usize) -> (u32, u32, u32) {
            let k = &self.ks[i];
            (k.grid, k.block, k.smem)
        }

        /// Enqueue kernels `r` only (profiling).
        pub fn launch_range(
            &self,
            s: &Stream,
            ext: &[u64],
            r: std::ops::Range<usize>,
        ) -> anyhow::Result<()> {
            assert_eq!(ext.len(), self.n_ext, "derived op: external pointer count");
            for k in &self.ks[r] {
                let mut vals: Vec<u64> = k
                    .args
                    .iter()
                    .map(|a| match *a {
                        Slot::Ext(i) => ext[i],
                        Slot::Scratch(i) => self.scratch[i].ptr,
                    })
                    .collect();
                let mut pts = k.points;
                let mut args: Vec<*mut std::ffi::c_void> = vals
                    .iter_mut()
                    .map(|p| p as *mut u64 as *mut std::ffi::c_void)
                    .collect();
                args.push(&mut pts as *mut u32 as *mut std::ffi::c_void);
                unsafe { gpu::launch(k.f, (k.grid, 1, 1), (k.block, 1, 1), k.smem, s, &mut args) }
                    .map_err(|e| anyhow::anyhow!("derived op launch: {e}"))?;
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Coder-Next ops
// ---------------------------------------------------------------------------

/// `rmsnorm(r [+ y]) * w` over rows of `n`: ext `[r, (y,) w, r_out, x]`.
/// With `add`, `r_out = r + y` (the residual, written to its own buffer:
/// the IR has no in-place outputs) and `x = rmsnorm(r_out) * w`.
pub fn add_norm(t: usize, n: usize, eps: f64, add: bool) -> Spec {
    let mut g = TensorGraph::new();
    let r = g.input(&[t, n]);
    let y = add.then(|| g.input(&[t, n]));
    let w = g.input(&[n]);
    let ro = match y {
        Some(y) => g.map(&[r, y], |x| x[0] + x[1]).unwrap(),
        None => r,
    };
    let x = rmsnorm(&mut g, ro, w, t, n, eps);
    let mut ext = vec![r];
    ext.extend(y);
    ext.push(w);
    let mut outs = vec![];
    if add {
        ext.push(ro);
        outs.push(ro);
    }
    ext.push(x);
    outs.push(x);
    Spec {
        g,
        ext,
        outs,
        auto: true,
    }
}

fn rmsnorm(
    g: &mut TensorGraph,
    r: TensorId,
    w: TensorId,
    t: usize,
    n: usize,
    eps: f64,
) -> TensorId {
    // space [t, 1 | n]
    let ss = g
        .comp(
            &[t, 1],
            &[n],
            vec![Arg {
                src: r,
                view: View::pick(3, &[Some(0), Some(2)]),
            }],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[0]],
        )
        .unwrap();
    // a block a row (one thread would sum all n serially)
    g.set_schedule(ss, 256, (0, 0, 0), (1, 1));
    let (inv_n, eps) = (1.0 / n as f64, eps);
    g.comp(
        &[t, n],
        &[],
        vec![
            arg(r, View::identity(2, 2)),
            arg(ss, View::pick(2, &[Some(0), None])),
            arg(w, View::pick(2, &[Some(1)])),
        ],
        None,
        move |x| {
            let inv = (x[1] * ExprId::from_f64(inv_n) + ExprId::from_f64(eps))
                .sqrt()
                .recip();
            vec![x[0] * inv * x[2]]
        },
    )
    .unwrap()
}

fn arg(src: TensorId, view: View) -> Arg {
    Arg { src, view }
}

/// A view of rank `rank` whose source dim `d` reads `coefs[d] · coords + offsets[d]`.
fn affine(rank: usize, rows: &[(&[(usize, i64)], i64)]) -> View {
    View {
        forms: rows
            .iter()
            .map(|(cs, off)| {
                let mut f = Form::constant(rank, *off);
                for &(d, c) in cs.iter() {
                    f.coefs[d] = c;
                }
                f
            })
            .collect(),
        guard: false,
        gather: None,
    }
}

/// The MoE combine, residual and next norm (`cx_moe_add_norm`'s math):
/// `y[t] = Σ_k wts[t,k] parts[t·K+k] + sigmoid(lg[t, ne]) parts[T·K+t]`,
/// `r_out = r + y`, `x = rmsnorm(r_out) * w`.
/// ext `[r, parts, wts, lg, w, y, r_out, x]`; `lg` rows of `ls` floats.
pub fn moe_combine_norm(t: usize, n: usize, topk: usize, ne: usize, ls: usize, eps: f64) -> Spec {
    let mut g = TensorGraph::new();
    let r = g.input(&[t, n]);
    let parts = g.input(&[t * topk + t, n]);
    let wts = g.input(&[t, topk]);
    let lg = g.input(&[t, ls]);
    let w = g.input(&[n]);
    let k = topk as i64;
    // space [t, i | k]: wts[t, k] · parts[t·K + k, i]
    let routed = g
        .comp(
            &[t, n],
            &[topk],
            vec![
                arg(wts, View::pick(3, &[Some(0), Some(2)])),
                arg(parts, affine(3, &[(&[(0, k), (2, 1)], 0), (&[(1, 1)], 0)])),
            ],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[1]],
        )
        .unwrap();
    // space [t, i]: + sigmoid(lg[t, ne]) · parts[T·K + t, i]
    let y = g
        .comp(
            &[t, n],
            &[],
            vec![
                arg(routed, View::identity(2, 2)),
                arg(
                    parts,
                    affine(2, &[(&[(0, 1)], t as i64 * k), (&[(1, 1)], 0)]),
                ),
                arg(lg, affine(2, &[(&[(0, 1)], 0), (&[], ne as i64)])),
            ],
            None,
            |x| {
                let s = (ExprId::ONE + (-x[2]).exp()).recip();
                vec![x[0] + s * x[1]]
            },
        )
        .unwrap();
    let ro = g.map(&[r, y], |x| x[0] + x[1]).unwrap();
    let x = rmsnorm(&mut g, ro, w, t, n, eps);
    Spec {
        g,
        ext: vec![r, parts, wts, lg, w, y, ro, x],
        outs: vec![y, ro, x],
        auto: true,
    }
}

/// Greedy argmax over each row of `[t, v]` logits (ties to the lower id),
/// as `(id, value)` f32 pairs `[t, 2]`: ext `[logits, out]`.
pub fn argmax(t: usize, v: usize) -> Spec {
    let mut g = TensorGraph::new();
    let l = g.input(&[t, v]);
    let iv = g.iota(&[t, v], 1);
    let am = g
        .comp_tuple(
            &[t, 2],
            &[v],
            vec![arg(iv, View::identity(2, 2)), arg(l, View::identity(2, 2))],
            Some(Monoid::argmax()),
            |x| vec![x[0], x[1]],
        )
        .unwrap();
    g.set_schedule(am, 256, (0, 0, 0), (1, 1));
    Spec {
        g,
        ext: vec![l, am],
        outs: vec![am],
        auto: true,
    }
}

/// Softmax top-k routing (`cx_route`'s math): the `k` largest of each row's
/// first `ne` logits (ties to the lower id) as `[t, 2k]` (id, logit) pairs,
/// and their softmax renormalized over the top k, `[t, k]`.
/// ext `[lg, top, wts]`.
pub fn route(t: usize, ne: usize, ls: usize, k: usize) -> Spec {
    let mut g = TensorGraph::new();
    let lg = g.input(&[t, ls]);
    let ie = g.iota(&[t, ne], 1);
    let top = g
        .comp_tuple(
            &[t, 2 * k],
            &[ne],
            vec![
                arg(ie, View::identity(2, 2)),
                arg(lg, View::pick(2, &[Some(0), Some(1)])),
            ],
            Some(Monoid::topk(k)),
            |v| {
                let mut o = vec![v[0], v[1]];
                for _ in 1..k {
                    o.push(ExprId::ZERO);
                    o.push(ExprId::from_f64(-(f32::MAX as f64)));
                }
                o
            },
        )
        .unwrap();
    g.set_schedule(top, 256, (0, 0, 0), (1, 1));
    // wts[t, j] = exp(v_j - v_0) / Σ_i exp(v_i - v_0)   (v_0 the max)
    let ik = g.iota(&[t, k], 1);
    let mut args = vec![arg(ik, View::identity(2, 2))];
    for j in 0..k {
        args.push(arg(
            top,
            affine(2, &[(&[(0, 1)], 0), (&[], 2 * j as i64 + 1)]),
        ));
    }
    let wts = g
        .comp(&[t, k], &[], args, None, move |v| {
            let ex: Vec<ExprId> = (0..k).map(|j| (v[1 + j] - v[1]).exp()).collect();
            let norm = ex.iter().fold(ExprId::ZERO, |a, &b| a + b);
            let mut acc = ExprId::ZERO;
            for (j, &e) in ex.iter().enumerate() {
                let sel = |c, a, b| tang_expr::with_graph(|eg| eg.select(c, a, b));
                let d = v[0] - ExprId::from_f64(j as f64);
                let ge = sel(d + ExprId::from_f64(0.5), ExprId::ONE, ExprId::ZERO);
                let le = sel(ExprId::from_f64(0.5) - d, ExprId::ONE, ExprId::ZERO);
                let hit = ge * le;
                acc = acc + hit * e;
            }
            vec![acc / norm]
        })
        .unwrap();
    Spec {
        g,
        ext: vec![lg, top, wts],
        outs: vec![top, wts],
        auto: true,
    }
}

/// A dense GEMV on GGUF Q8_0 weights (`cx_gemvk`'s job): `out[t, r] =
/// Σ_k W[r, k] · q8(x)[t, k]`, `x` quantized to int8 per 32 in the kernel's
/// prologue (`Format::Q8x32` inlined), the dot on `dp4a`.
/// ext `[x, w, out]`; `w` the GGUF bytes, rows of `k / 32 · 34`.
pub fn gemv_q8_0(t: usize, rows: usize, k: usize) -> Spec {
    let mut g = TensorGraph::new();
    let x = g.input(&[t, k]);
    let w = g.input(&[rows, k]);
    let xq = g.map(&[x], |v| v[0]).unwrap();
    let out = g
        .comp(
            &[t, rows],
            &[k],
            vec![
                arg(w, View::pick(3, &[Some(1), Some(2)])),
                arg(xq, View::pick(3, &[Some(0), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    g.set_precision(w, Precision::storage(Format::Q8_0));
    g.set_precision(xq, Precision::storage(Format::Q8x32));
    // one row: the dp4a GEMV, x quantized in its prologue; more rows: int8
    // tensor cores (eight positions are the MMA's N), x quantized once
    let mma = t
        >= std::env::var("TANG_GEMV_MMA_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
    g.set_int_mma(
        out,
        Some(IntMma {
            stage: if mma { Stage::Mma } else { Stage::Gemv },
            ksplit: std::env::var("TANG_GEMV_KSPLIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(if mma { 2 } else { 1 }),
            pos_tile: std::env::var("TANG_GEMV_PT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or((if mma { t.min(64) } else { t.min(8) }) as u32),
        }),
    );
    g.set_inline(xq, t == 1);
    Spec {
        g,
        ext: vec![x, w, out],
        outs: vec![out],
        auto: false,
    }
}

/// Shapes and weight formats of one MoE layer for [`moe_block`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MoeShape {
    pub t: usize,
    pub n: usize,
    pub ff: usize,
    pub ne: usize,
    pub k: usize,
    /// Router logits row stride (`ne + 1`: the shared expert's gate last).
    pub ls: usize,
    /// gate / up expert stacks' format (Q4K or Q5K), down's (Q5K).
    pub gu: Format,
    pub down: Format,
    pub eps_bits: u64,
}

/// External tensors of [`moe_block`], in launch order.
pub const MOE_EXT: [&str; 13] = [
    "x", "lg", "wg", "wu", "wd", "sg", "su", "sd", "r", "wn", "y", "r_out", "x_out",
];

/// A decode window's whole MoE layer, as the naive math: softmax top-k
/// routing of the router logits `lg`, the routed experts' gate and up rows
/// (gathered by expert id, k-quant weights, x quantized per 32), SwiGLU
/// (quantized per 32), down rows, the shared expert (Q8_0) the same way,
/// then `y = Σ_j w_j·down_j + sigmoid(lg[ne])·shared`, `r_out = r + y`,
/// `x_out = rmsnorm(r_out)·wn`. Ext as [`MOE_EXT`] (`r_out` may alias `r`,
/// `x_out` may alias `x`: every read of them precedes the write).
pub fn moe_block(m: MoeShape) -> Spec {
    let MoeShape {
        t,
        n,
        ff,
        ne,
        k,
        ls,
        ..
    } = m;
    let eps = f64::from_bits(m.eps_bits);
    let mut g = TensorGraph::new();
    let x = g.input(&[t, n]);
    let lg = g.input(&[t, ls]);
    let wg = g.input(&[ne, ff, n]);
    let wu = g.input(&[ne, ff, n]);
    let wd = g.input(&[ne, n, ff]);
    let sg = g.input(&[ff, n]);
    let su = g.input(&[ff, n]);
    let sd = g.input(&[n, ff]);
    let r = g.input(&[t, n]);
    let wn = g.input(&[n]);

    // routing: top-k (id, logit) pairs, renormalized softmax weights
    let ie = g.iota(&[t, ne], 1);
    let top = g
        .comp_tuple(
            &[t, 2 * k],
            &[ne],
            vec![
                arg(ie, View::identity(2, 2)),
                arg(lg, View::pick(2, &[Some(0), Some(1)])),
            ],
            Some(Monoid::topk(k)),
            |v| {
                let mut o = vec![v[0], v[1]];
                for _ in 1..k {
                    o.push(ExprId::ZERO);
                    o.push(ExprId::from_f64(-(f32::MAX as f64)));
                }
                o
            },
        )
        .unwrap();
    let envn = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    g.set_schedule(top, envn("TANG_DMOE_TOPK_SPLIT", 256) as u32, (0, 0, 0), (1, 1));
    // wts[t, j] = exp(v_j − v_0) / Σ_i exp(v_i − v_0)
    let mut wargs = vec![arg(g.iota(&[t, k], 1), View::identity(2, 2))];
    for j in 0..k {
        wargs.push(arg(
            top,
            affine(2, &[(&[(0, 1)], 0), (&[], 2 * j as i64 + 1)]),
        ));
    }
    let wts = g
        .comp(&[t, k], &[], wargs, None, move |v| {
            let sel = |c, a, b| tang_expr::with_graph(|eg| eg.select(c, a, b));
            let ex: Vec<ExprId> = (0..k).map(|j| (v[1 + j] - v[1]).exp()).collect();
            let norm = ex.iter().fold(ExprId::ZERO, |a, &b| a + b);
            let mut acc = ExprId::ZERO;
            for (j, &e) in ex.iter().enumerate() {
                let d = v[0] - ExprId::from_f64(j as f64);
                let hit = sel(d + ExprId::from_f64(0.5), ExprId::ONE, ExprId::ZERO)
                    * sel(ExprId::from_f64(0.5) - d, ExprId::ONE, ExprId::ZERO);
                acc = acc + hit * e;
            }
            vec![acc / norm]
        })
        .unwrap();

    // routed experts: space [t, j, row | col], expert id = top[t, 2j]
    let xq = g.map(&[x], |v| v[0]).unwrap();
    let id_view = affine(4, &[(&[(0, 1)], 0), (&[(1, 2)], 0)]);
    let gathered = |g: &mut TensorGraph, w: TensorId, src: TensorId, rows: usize, cols: usize| {
        g.comp(
            &[t, k, rows],
            &[cols],
            vec![
                arg(w, View::pick(4, &[None, Some(2), Some(3)]).gather(0, 2, 1)),
                arg(src, View::pick(4, &[Some(0), Some(1), Some(3)])),
                arg(top, id_view.clone()),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    };
    // x is [t, n]: read it ignoring j
    let gx = |g: &mut TensorGraph, w: TensorId| {
        g.comp(
            &[t, k, ff],
            &[n],
            vec![
                arg(w, View::pick(4, &[None, Some(2), Some(3)]).gather(0, 2, 1)),
                arg(xq, View::pick(4, &[Some(0), Some(3)])),
                arg(top, id_view.clone()),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    };
    let eg = gx(&mut g, wg);
    let eu = gx(&mut g, wu);
    let silu = |v: &[ExprId]| v[0] / (ExprId::ONE + (-v[0]).exp()) * v[1];
    let h = g.map(&[eg, eu], silu).unwrap();
    let dn = gathered(&mut g, wd, h, n, ff);

    // shared expert: space [t, row | col]
    let dense = |g: &mut TensorGraph, w: TensorId, src: TensorId, rows: usize, cols: usize| {
        g.comp(
            &[t, rows],
            &[cols],
            vec![
                arg(w, View::pick(3, &[Some(1), Some(2)])),
                arg(src, View::pick(3, &[Some(0), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    };
    let shg = dense(&mut g, sg, xq, ff, n);
    let shu = dense(&mut g, su, xq, ff, n);
    let sh = g.map(&[shg, shu], silu).unwrap();
    let shd = dense(&mut g, sd, sh, n, ff);

    // combine: Σ_j wts[t, j] · dn[t, j, i]   space [t, i | j]
    let routed = g
        .comp(
            &[t, n],
            &[k],
            vec![
                arg(wts, View::pick(3, &[Some(0), Some(2)])),
                arg(dn, View::pick(3, &[Some(0), Some(2), Some(1)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let y = g
        .comp(
            &[t, n],
            &[],
            vec![
                arg(routed, View::identity(2, 2)),
                arg(shd, View::identity(2, 2)),
                arg(lg, affine(2, &[(&[(0, 1)], 0), (&[], ne as i64)])),
            ],
            None,
            |v| vec![v[0] + (ExprId::ONE + (-v[2]).exp()).recip() * v[1]],
        )
        .unwrap();
    let ro = g.map(&[r, y], |v| v[0] + v[1]).unwrap();
    let xo = rmsnorm(&mut g, ro, wn, t, n, eps);

    // schedule: storage formats, int8 contractions, x / SwiGLU quantized
    g.set_precision(wg, Precision::storage(m.gu));
    g.set_precision(wu, Precision::storage(m.gu));
    g.set_precision(wd, Precision::storage(m.down));
    for w in [sg, su, sd] {
        g.set_precision(w, Precision::storage(Format::Q8_0));
    }
    for a in [xq, h, sh] {
        g.set_precision(a, Precision::storage(Format::Q8x32));
    }
    let gemv = IntMma {
        stage: Stage::Gemv,
        ksplit: 1,
        pos_tile: 1,
    };
    for c in [eg, eu, dn] {
        g.set_int_mma(c, Some(gemv));
    }
    // the shared expert: the dp4a GEMV with every row on one weight read, the same
    // accumulation order at any T (a token's outputs bitwise the same in any window);
    // `TANG_CODER_DMOE_MMA=1`: int8 tensor cores past one row (faster, not bitwise across T)
    let shared = if t > 1 && std::env::var("TANG_CODER_DMOE_MMA").is_ok_and(|v| v == "1") {
        IntMma {
            stage: Stage::Mma,
            ksplit: 2,
            pos_tile: t.min(64) as u32,
        }
    } else {
        IntMma {
            pos_tile: t.min(8) as u32,
            ..gemv
        }
    };
    for c in [shg, shu, shd] {
        g.set_int_mma(c, Some(shared));
    }
    // one row: every GEMV quantizes its own activations in its prologue
    if t == 1 {
        g.set_inline(xq, envn("TANG_DMOE_INLINE_X", 0) == 1);
        g.set_inline(h, true);
        g.set_inline(sh, true);
    }
    g.set_schedule(routed, 0, (0, 0, 0), (1, 1));
    let _ = gathered;
    Spec {
        g,
        ext: vec![x, lg, wg, wu, wd, sg, su, sd, r, wn, y, ro, xo],
        outs: vec![y, ro, xo],
        auto: false,
    }
}

/// The engine's int8 activation layout for the next layer's hand kernels:
/// codes `[rows][n]` (a `Format::Q8x32` buffer's codes are exactly that),
/// then f32 scales at `rows · n`. Two ops, launched in order: `codes` (ext
/// `[x, q]`, `q` at the buffer) then `scales` (ext `[x, d]`, `d` at the
/// buffer + `rows · n`), which overwrites the Q8x32 scale pairs the first
/// leaves past the codes.
pub fn engine_q8(t: usize, n: usize) -> (Spec, Spec) {
    let mut g = TensorGraph::new();
    let x = g.input(&[t, n]);
    let q = g.map(&[x], |v| v[0]).unwrap();
    g.set_precision(q, Precision::storage(Format::Q8x32));
    let codes = Spec {
        g,
        ext: vec![x, q],
        outs: vec![q],
        auto: true,
    };
    let mut g = TensorGraph::new();
    let x = g.input(&[t, n]);
    // space [t, b | i]: max |x[t, 32 b + i]| / 127
    let d = g
        .comp(
            &[t, n / 32],
            &[32],
            vec![arg(
                x,
                affine(3, &[(&[(0, 1)], 0), (&[(1, 32), (2, 1)], 0)]),
            )],
            Some(Monoid::max()),
            |v| vec![v[0].abs()],
        )
        .unwrap();
    let d = g.map(&[d], |v| v[0] / ExprId::from_f64(127.0)).unwrap();
    let scales = Spec {
        g,
        ext: vec![x, d],
        outs: vec![d],
        auto: true,
    };
    (codes, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn eval(s: &Spec, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        s.g.eval(inputs, &s.outs)
    }

    fn norm_ref(r: &[f64], w: &[f64], n: usize, eps: f64) -> Vec<f64> {
        r.chunks(n)
            .flat_map(|row| {
                let ss: f64 = row.iter().map(|v| v * v).sum::<f64>() / n as f64;
                let inv = 1.0 / (ss + eps).sqrt();
                row.iter()
                    .zip(w)
                    .map(move |(v, w)| v * inv * w)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{i}: {x} vs {y}");
        }
    }

    #[test]
    fn add_norm_matches_reference() {
        let (t, n) = (3, 64);
        let (r, y, w) = (rnd(t * n, 1), rnd(t * n, 2), rnd(n, 3));
        let s = add_norm(t, n, 1e-6, true);
        let out = eval(&s, &[r.clone(), y.clone(), w.clone()]);
        let ro: Vec<f64> = r.iter().zip(&y).map(|(a, b)| a + b).collect();
        close(&out[0], &ro, 1e-12);
        close(&out[1], &norm_ref(&ro, &w, n, 1e-6), 1e-9);
    }

    #[test]
    fn moe_combine_norm_matches_reference() {
        let (t, n, k, ne) = (2, 32, 3, 8);
        let ls = ne + 1;
        let r = rnd(t * n, 4);
        let parts = rnd((t * k + t) * n, 5);
        let wts = rnd(t * k, 6);
        let lg = rnd(t * ls, 7);
        let w = rnd(n, 8);
        let s = moe_combine_norm(t, n, k, ne, ls, 1e-6);
        let out = eval(
            &s,
            &[r.clone(), parts.clone(), wts.clone(), lg.clone(), w.clone()],
        );
        let mut y = vec![0.0; t * n];
        for ti in 0..t {
            let gte = 1.0 / (1.0 + (-lg[ti * ls + ne]).exp());
            for i in 0..n {
                let mut a = 0.0;
                for j in 0..k {
                    a += wts[ti * k + j] * parts[(ti * k + j) * n + i];
                }
                y[ti * n + i] = a + gte * parts[(t * k + ti) * n + i];
            }
        }
        close(&out[0], &y, 1e-12);
        let ro: Vec<f64> = r.iter().zip(&y).map(|(a, b)| a + b).collect();
        close(&out[1], &ro, 1e-12);
        close(&out[2], &norm_ref(&ro, &w, n, 1e-6), 1e-9);
    }

    #[test]
    fn route_matches_reference() {
        let (t, ne, k) = (2, 40, 4);
        let ls = ne + 1;
        let mut lg = rnd(t * ls, 9);
        lg[5] = lg[7]; // a tie: the lower id first
        let s = route(t, ne, ls, k);
        let out = eval(&s, &[lg.clone()]);
        for ti in 0..t {
            let row = &lg[ti * ls..ti * ls + ne];
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            let top = &idx[..k];
            let m = row[top[0]];
            let z: f64 = top.iter().map(|&e| (row[e] - m).exp()).sum();
            for j in 0..k {
                assert_eq!(out[0][ti * 2 * k + 2 * j] as usize, top[j]);
                let want = (row[top[j]] - m).exp() / z;
                assert!((out[1][ti * k + j] - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn argmax_matches_reference() {
        let (t, v) = (3, 100);
        let mut l = rnd(t * v, 10);
        l[150] = 5.0;
        l[160] = 5.0;
        let out = eval(&argmax(t, v), &[l.clone()]);
        for ti in 0..t {
            let row = &l[ti * v..(ti + 1) * v];
            let mut best = 0;
            for (i, &x) in row.iter().enumerate() {
                if x > row[best] {
                    best = i;
                }
            }
            assert_eq!(out[0][2 * ti] as usize, best);
        }
    }

    fn coder_moe(t: usize) -> MoeShape {
        MoeShape {
            t,
            n: 2048,
            ff: 512,
            ne: 512,
            k: 10,
            ls: 513,
            gu: Format::Q4K,
            down: Format::Q5K,
            eps_bits: 1e-6f64.to_bits(),
        }
    }

    #[test]
    fn ops_lower_to_cuda() {
        for (name, s) in [
            ("add_norm", add_norm(4, 2048, 1e-6, true)),
            (
                "moe_combine_norm",
                moe_combine_norm(4, 2048, 10, 512, 513, 1e-6),
            ),
            ("route", route(4, 512, 513, 10)),
            ("argmax", argmax(2, 151936)),
            ("gemv_q8_0", gemv_q8_0(1, 4096, 2048)),
            ("gemv_q8_0 t8", gemv_q8_0(8, 4096, 2048)),
            ("moe_block t1", moe_block(coder_moe(1))),
            ("moe_block t8", moe_block(coder_moe(8))),
            ("engine_q8 codes", engine_q8(8, 2048).0),
            ("engine_q8 scales", engine_q8(8, 2048).1),
        ] {
            let l = s.lower().unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(!l.kernels.is_empty(), "{name}");
            eprintln!(
                "{name}: {} kernels, {} scratch",
                l.kernels.len(),
                l.scratch.len()
            );
        }
    }
}
#[cfg(test)]
mod dump {
    #[test]
    #[ignore]
    fn dump_add_norm_64() {
        let l = super::add_norm(64, 2048, 1e-6, true).lower().unwrap();
        for k in &l.kernels {
            eprintln!(
                "--- grid {} block {} smem {} points {} args {:?}\n{}",
                k.grid, k.block, k.smem, k.points, k.args, k.source
            );
        }
    }
}
#[cfg(test)]
mod moe_lowering {
    use super::*;
    #[test]
    fn moe_block_contractions_are_int8() {
        for t in [1, 8] {
            let l = moe_block(MoeShape {
                t,
                n: 2048,
                ff: 512,
                ne: 512,
                k: 10,
                ls: 513,
                gu: Format::Q4K,
                down: Format::Q5K,
                eps_bits: 1e-6f64.to_bits(),
            })
            .lower()
            .unwrap();
            let int8 = l
                .kernels
                .iter()
                .filter(|k| k.source.contains("__dp4a") || k.source.contains("mma.sync"))
                .count();
            for (i, k) in l.kernels.iter().enumerate() {
                eprintln!(
                    "t{t} k{i}: grid {} block {} int8 {} q8x32w {}",
                    k.grid,
                    k.block,
                    k.source.contains("__dp4a") || k.source.contains("mma.sync"),
                    k.source.contains("tang_q8x32_store")
                );
            }
            assert_eq!(int8, 6, "t={t}: six int8 contractions");
        }
    }
}
