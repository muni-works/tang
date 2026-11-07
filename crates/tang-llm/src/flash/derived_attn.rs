//! Wide-prefill attention for the Coder engine derived from the tensor IR
//! (`TANG_CODER_ATTN=derived`): the attention of one window written naively
//! — scores, an offset causal mask, softmax, the weighted sum — and lowered
//! by `tang-compile` (`schedule_auto` picks the Hopper flash shape at head
//! dim 256, #173). The engine compiles the generated CUDA with NVRTC and
//! launches it on its own buffers.
//!
//! Shapes are fixed per kernel, so the key count is padded: a window of `t`
//! query rows over keys `[0, n_pad)`, `n_pad` a multiple of [`KEY_STEP`]. The
//! mask reads each row's position from `pos` (keys `j > pos[i]` drop out,
//! padded rows have `pos = -1`), so one kernel serves any window offset whose
//! last position is below `n_pad`. The kernels are built lazily and cached by
//! `n_pad`.
//!
//! Inputs (fp16: the bf16 / int8 K/V cache and the f32 queries are converted
//! by `cx_attn_dpack` first, as fp16 holds bf16's 8 mantissa bits exactly and
//! keeps Q's and P's 3 more than bf16 would):
//! - `q [nkv, g, t, hd]` f16, the window's queries (heads grouped by kv head),
//! - `k [nkv, n_pad, hd]` f16, `v [nkv, n_pad, hd]` f16,
//! - `pos [1, 1, g·t, 1]` f32, each row's position, repeated for each of a
//!   group's g heads (−1: padding).
//!
//! Output `o [nkv, g, t, hd]` f32, softmax-normalized.

use tang::Scalar;
use tang_compile::tensor::{Arg, Format, Monoid, Precision, TensorGraph, TensorId, View};
use tang_expr::ExprId;

/// Key counts are padded to a multiple of this (fewer kernel variants).
pub const KEY_STEP: usize = 4096;

/// The attention problem of one window kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttnShape {
    pub t: usize,
    pub n_pad: usize,
    pub nkv: usize,
    pub g: usize,
    pub hd: usize,
}

/// The graph's inputs, in launch order, and its output.
pub struct AttnGraph {
    pub g: TensorGraph,
    pub q: TensorId,
    pub k: TensorId,
    pub v: TensorId,
    pub pos: TensorId,
    pub o: TensorId,
}

/// The naive window attention: `o[h,a,i,:] = softmax_j(q·k / √hd + mask)·v`
/// (`pos` holds each row's position once a head: `[1, 1, g·t, 1]`),
/// heads first (the layout the flash lowering tiles: queries the tile's rows,
/// head dim its lanes).
pub fn graph(s: AttnShape) -> AttnGraph {
    // GQA: a kv head's g query heads are folded into the query rows (row
    // a·t + i), so one K/V tile feeds every head of the group (FA2's GQA
    // packing; the flash lowering tiles one query dim). Same memory layout
    // as [nkv, g, t, hd].
    let AttnShape {
        t,
        n_pad,
        nkv,
        g: gq0,
        hd,
    } = s;
    let (t, gq) = (gq0 * t, 1);
    let mut g = TensorGraph::new();
    let q = g.input(&[nkv, gq, t, hd]);
    let k = g.input(&[nkv, n_pad, hd]);
    let v = g.input(&[nkv, n_pad, hd]);
    let pos = g.input(&[1, 1, t, 1]);
    let scale = 1.0 / (hd as f64).sqrt();
    let qs = g.map(&[q], |x| x[0] * ExprId::from_f64(scale)).unwrap();
    // space [h, a, i, j | c]
    let sc = g
        .comp(
            &[nkv, gq, t, n_pad],
            &[hd],
            vec![
                Arg {
                    src: qs,
                    view: View::pick(5, &[Some(0), Some(1), Some(2), Some(4)]),
                },
                Arg {
                    src: k,
                    view: View::pick(5, &[Some(0), Some(3), Some(4)]),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[1]],
        )
        .unwrap();
    // causal over positions: key j drops out past row i's position
    let full = [nkv, gq, t, n_pad];
    let sc = if std::env::var_os("DA_NOMASK").is_some() {
        sc
    } else {
        let j = g.iota(&full, 3);
        let posb = g.broadcast(pos, &full).unwrap();
        g.map(&[sc, j, posb], |x| {
            let past = x[1] - x[2];
            let m = tang_expr::with_graph(|eg| {
                let neg = eg.lit(-1e30);
                eg.select(past, neg, ExprId::ZERO)
            });
            x[0] + m
        })
        .unwrap()
    };
    let m = g.reduce(sc, 3, Monoid::max()).unwrap();
    let mb = g.broadcast(m, &full).unwrap();
    let p = g.map(&[sc, mb], |x| (x[0] - x[1]).exp()).unwrap();
    let l = g.reduce(p, 3, Monoid::sum()).unwrap();
    // space [h, a, i, c | j]
    let o = g
        .comp(
            &[nkv, gq, t, hd],
            &[n_pad],
            vec![
                Arg {
                    src: p,
                    view: View::pick(5, &[Some(0), Some(1), Some(2), Some(4)]),
                },
                Arg {
                    src: v,
                    view: View::pick(5, &[Some(0), Some(4), Some(3)]),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[1]],
        )
        .unwrap();
    let lb = g.broadcast(l, &[nkv, gq, t, hd]).unwrap();
    let o = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
    AttnGraph { g, q, k, v, pos, o }
}

/// The IR passes `hopper_attention` runs, then fp16 storage on the
/// operands. Returns the scheduled graph and its output.
pub fn scheduled(s: AttnShape) -> (TensorGraph, AttnGraph) {
    let a = graph(s);
    let outs = [a.o];
    let (g, outs) = a.g.stabilize_softmax(&outs);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    tile_reductions(&mut g);
    for &t in &[a.q, a.k, a.v] {
        g.set_precision(
            t,
            Precision::storage(if std::env::var_os("DA_BF16").is_some() {
                Format::BF16
            } else {
                Format::F16
            }),
        );
    }
    let o = outs[0];
    let ag = AttnGraph { o, ..a };
    (g, ag)
}

/// `hopper_attention`'s step after `schedule_auto`: every reduction left
/// untiled gets the 16×16 tile so the CUDA lowering tries its tensor-core
/// tiles on it.
fn tile_reductions(g: &mut TensorGraph) {
    for i in 0..g.nodes().len() {
        let n = &g.nodes()[i];
        let Some(c) = g.comp_at(i) else { continue };
        let red: usize = c.red.iter().map(|r| r.n()).product();
        if !n.inline && !c.red.is_empty() && red >= 16 && n.tile.0 == 0 && n.split == 0 {
            g.set_schedule(TensorId::from_index(i), 0, (16, 16, 32), (1, 1));
        }
    }
}

/// One compiled window kernel and how to launch it.
#[cfg(feature = "cuda")]
pub struct Kernel {
    pub f: cudarc::driver::sys::CUfunction,
    _m: tang_moe::gpu::Module,
    /// Each parameter's role, in parameter order (then `out`, `count`).
    pub params: Vec<Role>,
    pub threads: u32,
    pub tiles: u32,
    pub smem: u32,
    pub count: u32,
}

/// What a kernel parameter points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Q,
    K,
    V,
    Pos,
}

fn define(src: &str, name: &str) -> Option<u32> {
    src.lines()
        .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
        .and_then(|v| v.trim().parse().ok())
}

/// The CUDA source of the window kernel for `s` and the roles of its
/// parameters: lowering variant `TANG_CODER_ATTN_VARIANT` if set, else the
/// first variant that lowers to one Hopper (`wgmma`) flash kernel.
pub fn source(s: AttnShape) -> Result<(String, Vec<Role>, u32), String> {
    let (g, a) = scheduled(s);
    let forced: Option<usize> = std::env::var("TANG_CODER_ATTN_VARIANT")
        .ok()
        .and_then(|v| v.parse().ok());
    let candidates: Vec<usize> = match forced {
        Some(v) => vec![v],
        None => (0..64).collect(),
    };
    for v in candidates {
        let p = g.to_cuda_variant(&[a.o], Some(v));
        let [k] = &p.kernels[..] else { continue };
        if !k.source.contains("TANG_TILES") || (forced.is_none() && !k.source.contains("wgmma")) {
            continue;
        }
        let roles = k
            .reads
            .iter()
            .map(|&r| match r {
                r if r == a.q => Ok(Role::Q),
                r if r == a.k => Ok(Role::K),
                r if r == a.v => Ok(Role::V),
                r if r == a.pos => Ok(Role::Pos),
                r => Err(format!("unexpected read {r:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((k.source.clone(), roles, k.count));
    }
    Err(format!("{s:?}: no variant lowers to one flash kernel"))
}

#[cfg(feature = "cuda")]
impl Kernel {
    /// Generate, compile (NVRTC, `sm_90a`: `wgmma` is arch-specific) and load
    /// the window kernel for `s`.
    pub fn build(gpu: &tang_moe::gpu::Gpu, s: AttnShape) -> anyhow::Result<Self> {
        use anyhow::anyhow;
        let (src, params, count) = source(s).map_err(|e| anyhow!(e))?;
        let (threads, tiles, smem) = (
            define(&src, "TANG_THREADS").ok_or_else(|| anyhow!("no TANG_THREADS"))?,
            define(&src, "TANG_TILES").ok_or_else(|| anyhow!("no TANG_TILES"))?,
            define(&src, "TANG_SMEM").unwrap_or(0),
        );
        let include = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_90a"),
            include_paths: vec![format!("{include}/include")],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&src, opts)
            .map_err(|e| anyhow!("nvrtc (derived attention {s:?}): {e:?}"))?
            .to_src();
        let m = gpu.module_from_ptx(&ptx).map_err(|e| anyhow!("{e}"))?;
        let f = m.func("main0").map_err(|e| anyhow!("{e}"))?;
        if smem > 48 * 1024 {
            unsafe {
                tang_moe::gpu::check(
                    cudarc::driver::sys::cuFuncSetAttribute(
                        f,
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem as i32,
                    ),
                    "cuFuncSetAttribute",
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }
        Ok(Kernel {
            f,
            _m: m,
            params,
            threads,
            tiles,
            smem,
            count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small window against plain f64 attention through the interpreter.
    #[test]
    fn window_attention_matches_reference() {
        let s = AttnShape {
            t: 6,
            n_pad: 16,
            nkv: 2,
            g: 2,
            hd: 16,
        };
        let a = graph(s);
        let n = |d: &[usize]| d.iter().product::<usize>();
        let fill = |len: usize, seed: u64| -> Vec<f64> {
            (0..len)
                .map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f64 / 500.0) - 1.0)
                .collect()
        };
        let q = fill(n(&[s.nkv, s.g, s.t, s.hd]), 1);
        let k = fill(n(&[s.nkv, s.n_pad, s.hd]), 2);
        let v = fill(n(&[s.nkv, s.n_pad, s.hd]), 3);
        // a window at positions 7..12 (keys 0..=pos), the last row padding
        let pos: Vec<f64> = vec![7.0, 8.0, 9.0, 10.0, 11.0, -1.0];
        let pos_g: Vec<f64> = (0..s.g).flat_map(|_| pos.iter().copied()).collect();
        let got =
            a.g.eval(&[q.clone(), k.clone(), v.clone(), pos_g], &[a.o])
                .remove(0);
        let mut worst = 0f64;
        for i in 0..s.t - 1 {
            for h in 0..s.nkv {
                for b in 0..s.g {
                    let qr = &q[((h * s.g + b) * s.t + i) * s.hd..][..s.hd];
                    let keys = pos[i] as usize + 1;
                    let sc: Vec<f64> = (0..keys)
                        .map(|j| {
                            let kr = &k[(h * s.n_pad + j) * s.hd..][..s.hd];
                            qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f64>()
                                / (s.hd as f64).sqrt()
                        })
                        .collect();
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                    let l: f64 = w.iter().sum();
                    for c in 0..s.hd {
                        let want: f64 = (0..keys)
                            .map(|j| w[j] * v[(h * s.n_pad + j) * s.hd + c])
                            .sum::<f64>()
                            / l;
                        let g_ = got[((h * s.g + b) * s.t + i) * s.hd + c];
                        worst = worst.max((g_ - want).abs());
                    }
                }
            }
        }
        assert!(worst < 1e-9, "max abs err {worst}");
    }

    /// At the engine's shape the passes leave one flash kernel (plus the
    /// final normalization if it does not fuse), on the Hopper path.
    #[test]
    fn lowers_to_flash_at_head_dim_256() {
        let e = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let s = AttnShape {
            t: e("DA_T", 4096),
            n_pad: e("DA_N", 8192),
            nkv: e("DA_KV", 2),
            g: e("DA_G", 8),
            hd: 256,
        };
        let (g, a) = scheduled(s);
        let mut native = Vec::new();
        for v in 0..64 {
            let p = g.to_cuda_variant(&[a.o], Some(v));
            if p.kernels.iter().any(|k| k.source.contains("TANG_TILES")) {
                let k = p
                    .kernels
                    .iter()
                    .find(|k| k.source.contains("TANG_TILES"))
                    .unwrap();
                eprintln!(
                    "variant {v}: {} kernels, wgmma {}, threads {}",
                    p.kernels.len(),
                    k.source.contains("wgmma"),
                    k.workgroup_size
                );
                native.push(v);
            }
        }
        eprintln!("native variants {native:?}");
        if let Ok(dir) = std::env::var("DA_DUMP") {
            for &v in &native {
                let p = g.to_cuda_variant(&[a.o], Some(v));
                let k = &p.kernels[0];
                std::fs::write(format!("{dir}/v{v}.cu"), &k.source).unwrap();
                eprintln!(
                    "v{v}: reads {:?} wg {} count {} bufs {:?}",
                    k.reads,
                    k.workgroup_size,
                    k.count,
                    p.buffers
                        .iter()
                        .map(|b| (b.node.index(), b.format))
                        .collect::<Vec<_>>()
                );
            }
        }
        let p = g.to_cuda_variant(&[a.o], native.first().copied());
        assert!(
            p.kernels.iter().any(|k| k.source.contains("TANG_TILES")),
            "no native flash kernel"
        );
    }
}
