//! GLM-5.3-Flash MoE (GGUF arch `glm5-next`): sigmoid+bias router (`noaux_tc`), 288 routed experts
//! top-8 plus 1 ungated shared expert, expert FFN 2048 with clamped SwiGLU, and the dense FFN
//! (12288) of the first 3 layers. CPU f32 reference + naive CUDA kernels ([`SRC`]).
//!
//! Ground truth is HF `modeling_glm5_next.py` (`Glm5NextTextTopkRouter`, `Glm5NextTextExperts`,
//! `Glm5NextTextMLP`, `Glm5NextTextMoE`); llama.cpp `models/glm5-next.cpp` +
//! `llama-graph.cpp::build_moe_ffn` gives the GGUF names and the per-layer clamp arrays.
//!
//! Semantics (all f32):
//! - router: `s = sigmoid(W x)`; selection on `s + e_score_correction_bias`, after the group limit
//!   (`n_group`/`topk_group`: group score = sum of its top-2 biased scores; a no-op for this model,
//!   whose `n_group = topk_group = 1`); weights are the *unbiased* `s` of the chosen experts,
//!   divided by `sum + 1e-20` when `norm_topk_prob`, then always × `routed_scaling_factor` (2.5).
//! - SwiGLU with limit L: `silu(min(g, L)) * clamp(u, -L, L)`. Routed experts use
//!   `swiglu_clamp_exp[l]`; the shared expert *and* the dense FFN use `swiglu_clamp_shexp[l]`
//!   (llama.cpp `build_ffn`); HF has one `swiglu_limit` for all three. Both arrays are all 10 in
//!   the real GGUF. A missing or ≤ 1e-6 limit means no clamp (llama.cpp's rule).
//! - shared expert: a plain MLP on the same input, **no sigmoid gate** (unlike Qwen3-Next), added
//!   with weight 1. An expert id ≥ `n_expert` (HF's "no expert" sentinel) contributes nothing.
//!
//! GGUF tensors of layer `l` (ggml dims, `dims[0]` fastest): `ffn_gate_inp [hid, E]` F32,
//! `exp_probs_b.bias [E]` F32, `ffn_{gate,up}_exps [hid, ff, E]`, `ffn_down_exps [ff, hid, E]`,
//! `ffn_{gate,up}_shexp [hid, ff]`, `ffn_down_shexp [ff, hid]`; dense layers `ffn_{gate,up}
//! [hid, 12288]`, `ffn_down [12288, hid]`. Experts are read lazily, one at a time, through
//! [`Gguf::expert`]; nothing is dequantised up front.
//!
//! UD-Q4_K_XL (unsloth; read from the published shard headers, no weights downloaded): MoE layers
//! 3..=45 (45 is the MTP layer), 43 in all. Experts gate/up/down are Q4_K/Q4_K/Q5_K on 40 layers
//! (4,378,853,376 B each), Q5_K/Q5_K/Q6_K on layer 11 (5,303,697,408 B), Q4_K/Q4_K/Q6_K on layers
//! 12 and 44 (4,699,717,632 B each): 189,857,267,712 B in all. Shared experts and dense FFNs are
//! Q8_0; router and bias F32. [`expert_bytes`] measures any GGUF.
//!
//! Later (FAST, only after the RIGHT gates): int8 activations + dp4a / tensor-core expert GEMMs
//! (the Coder-Next `cx_moe_*` path), weight rows staged in shared memory, tokens grouped by
//! expert, router+select and gate/up+silu fusion, a warp-per-token top-k, CUDA graphs.

use crate::gguf::{Gguf, TensorInfo, Value};
use anyhow::{bail, ensure, Context, Result};
use std::borrow::Cow;

// ---------------------------------------------------------------------------------------------
// Weights and parameters
// ---------------------------------------------------------------------------------------------

/// A weight: row-major f32 in memory, or a GGUF tensor read (and dequantised) on demand.
#[derive(Clone, Copy)]
pub enum W<'a> {
    /// `[n_mat][rows][cols]` row-major (one matrix for a 2-D weight).
    F32(&'a [f32]),
    Gguf(&'a Gguf, &'a TensorInfo),
}

impl<'a> W<'a> {
    /// Matrix `e` (0 for a 2-D weight) as `rows × cols` row-major f32.
    pub fn mat(&self, e: usize, rows: usize, cols: usize) -> Result<Cow<'a, [f32]>> {
        let n = rows * cols;
        match *self {
            W::F32(v) => {
                ensure!(
                    v.len() >= (e + 1) * n,
                    "f32 weight: matrix {e} of {rows}x{cols} out of range ({})",
                    v.len()
                );
                Ok(Cow::Borrowed(&v[e * n..(e + 1) * n]))
            }
            W::Gguf(g, t) => {
                ensure!(
                    t.dims.len() >= 2 && t.dims[0] as usize == cols && t.dims[1] as usize == rows,
                    "{}: dims {:?}, want [{cols}, {rows}, ..]",
                    t.name,
                    t.dims
                );
                match t.dims.len() {
                    2 => {
                        ensure!(e == 0, "{}: 2-D tensor has no matrix {e}", t.name);
                        Ok(Cow::Owned(g.dequantize(t)?))
                    }
                    3 => Ok(Cow::Owned(g.expert(t, e)?)),
                    _ => bail!("{}: unexpected rank {:?}", t.name, t.dims),
                }
            }
        }
    }

    /// A 1-D weight of `n` values.
    pub fn vec(&self, n: usize) -> Result<Cow<'a, [f32]>> {
        match *self {
            W::F32(v) => {
                ensure!(v.len() == n, "f32 vector: {} values, want {n}", v.len());
                Ok(Cow::Borrowed(v))
            }
            W::Gguf(g, t) => {
                ensure!(
                    t.dims == [n as u64],
                    "{}: dims {:?}, want [{n}]",
                    t.name,
                    t.dims
                );
                Ok(Cow::Owned(g.dequantize(t)?))
            }
        }
    }
}

/// One layer's FFN: geometry, routing constants and weight handles. A dense layer has
/// `dense_*` set and no router or experts.
#[derive(Clone, Copy, Default)]
pub struct MoeParams<'a> {
    pub hidden: usize,
    pub n_expert: usize,
    pub top_k: usize,
    /// Group-limited routing; 1 / 1 (the real model) disables it.
    pub n_group: usize,
    pub topk_group: usize,
    pub ff_exp: usize,
    pub ff_shexp: usize,
    pub ff_dense: usize,
    pub scale: f32,
    pub norm_topk_prob: bool,
    /// SwiGLU limits (`f32::INFINITY` = no clamp): routed experts / shared expert + dense FFN.
    pub limit_exp: f32,
    pub limit_shexp: f32,
    /// `[E][hid]`.
    pub router: Option<W<'a>>,
    /// `[E]`; None = zero bias.
    pub bias: Option<W<'a>>,
    /// `[E][ff][hid]`, `[E][ff][hid]`, `[E][hid][ff]`.
    pub gate_exps: Option<W<'a>>,
    pub up_exps: Option<W<'a>>,
    pub down_exps: Option<W<'a>>,
    /// `[ff_sh][hid]`, `[ff_sh][hid]`, `[hid][ff_sh]`.
    pub sh_gate: Option<W<'a>>,
    pub sh_up: Option<W<'a>>,
    pub sh_down: Option<W<'a>>,
    /// `[ff_dense][hid]`, `[ff_dense][hid]`, `[hid][ff_dense]`.
    pub dense_gate: Option<W<'a>>,
    pub dense_up: Option<W<'a>>,
    pub dense_down: Option<W<'a>>,
}

fn limit(v: f64) -> f32 {
    if v > 1e-6 {
        v as f32
    } else {
        f32::INFINITY
    }
}

/// Layer `l`'s value of a scalar-or-array metadata key (llama.cpp `get_key_or_arr`).
fn per_layer_f64(g: &Gguf, key: &str, l: usize) -> Result<Option<f64>> {
    let Ok(v) = g.meta(key) else { return Ok(None) };
    let x = match v {
        Value::Array(a) => a
            .get(l)
            .with_context(|| format!("{key}: no entry for layer {l} ({})", a.len()))?,
        other => other,
    };
    Ok(Some(
        x.as_f64().with_context(|| format!("{key}: not a number"))?,
    ))
}

impl<'a> MoeParams<'a> {
    pub fn is_dense(&self) -> bool {
        self.dense_gate.is_some()
    }

    /// Layer `l`'s FFN from a `glm5-next` GGUF. Reads metadata and tensor handles only.
    pub fn from_gguf(g: &'a Gguf, l: usize) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        ensure!(arch == "glm5-next", "not a glm5-next GGUF ({arch})");
        let key = |s: &str| format!("{arch}.{s}");
        let u = |s: &str| g.meta_u64(&key(s)).map(|v| v as usize);
        let t = |s: &str| {
            g.info(&format!("blk.{l}.{s}.weight"))
                .map(|t| W::Gguf(g, t))
        };
        let lim_exp = per_layer_f64(g, &key("swiglu_clamp_exp"), l)?.unwrap_or(0.0);
        let lim_sh = per_layer_f64(g, &key("swiglu_clamp_shexp"), l)?.unwrap_or(lim_exp);
        let mut p = MoeParams {
            hidden: u("embedding_length")?,
            limit_exp: limit(lim_exp),
            limit_shexp: limit(lim_sh),
            n_group: 1,
            topk_group: 1,
            ..Default::default()
        };
        if l < u("leading_dense_block_count")? {
            p.ff_dense = u("feed_forward_length")?;
            p.dense_gate = Some(t("ffn_gate")?);
            p.dense_up = Some(t("ffn_up")?);
            p.dense_down = Some(t("ffn_down")?);
            return Ok(p);
        }
        let gating = u("expert_gating_func")?;
        ensure!(
            gating == 2,
            "expert_gating_func {gating}: only 2 (sigmoid) is implemented"
        );
        p.n_expert = u("expert_count")?;
        p.top_k = u("expert_used_count")?;
        p.n_group = u("expert_group_count").unwrap_or(1).max(1);
        p.topk_group = u("expert_group_used_count").unwrap_or(1).max(1);
        p.ff_exp = u("expert_feed_forward_length")?;
        p.scale = g.meta_f64(&key("expert_weights_scale")).unwrap_or(1.0) as f32;
        p.norm_topk_prob = matches!(g.meta(&key("expert_weights_norm")), Ok(Value::Bool(true)));
        p.router = Some(t("ffn_gate_inp")?);
        p.bias = g
            .get(&format!("blk.{l}.exp_probs_b.bias"))
            .map(|t| W::Gguf(g, t));
        p.gate_exps = Some(t("ffn_gate_exps")?);
        p.up_exps = Some(t("ffn_up_exps")?);
        p.down_exps = Some(t("ffn_down_exps")?);
        let n_sh = u("expert_shared_count").unwrap_or(0);
        if n_sh > 0 {
            p.ff_shexp = u("expert_shared_feed_forward_length").unwrap_or(p.ff_exp) * n_sh;
            p.sh_gate = Some(t("ffn_gate_shexp")?);
            p.sh_up = Some(t("ffn_up_shexp")?);
            p.sh_down = Some(t("ffn_down_shexp")?);
        }
        Ok(p)
    }
}

/// Bytes of layer `l`'s routed experts (gate + up + down) in a GGUF, from its tensor directory.
pub fn expert_bytes(g: &Gguf, l: usize) -> Result<u64> {
    ["gate", "up", "down"]
        .iter()
        .map(|n| {
            g.info(&format!("blk.{l}.ffn_{n}_exps.weight"))
                .map(|t| t.nbytes)
        })
        .sum()
}

// ---------------------------------------------------------------------------------------------
// CPU f32 reference
// ---------------------------------------------------------------------------------------------

/// `y[r] = Σ_j w[r][j] x[j]` for a `rows × cols` row-major matrix (f64 accumulation).
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    // parallel over output rows: each element's f64 dot is computed exactly as before
    // (same order), so the result is bit-identical to the serial loop
    w.par_chunks_exact(cols)
        .map(|r| {
            r.iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// One clamped-SwiGLU MLP on one token: `down · (silu(min(gate·x, L)) * clamp(up·x, ±L))`.
fn mlp(
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    ff: usize,
    hid: usize,
    x: &[f32],
    lim: f32,
) -> Vec<f32> {
    let g = matvec(gate, ff, hid, x);
    let u = matvec(up, ff, hid, x);
    let h: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| silu(g.min(lim)) * u.clamp(-lim, lim))
        .collect();
    matvec(down, hid, ff, &h)
}

/// Router for `t_len` tokens (`x` is `[t_len][hidden]`): per token the top-`k` expert ids and
/// their weights, `[t][k]`, highest biased score first.
pub fn route(p: &MoeParams, x: &[f32], t_len: usize) -> Result<(Vec<u32>, Vec<f32>)> {
    let (e, k, hid) = (p.n_expert, p.top_k, p.hidden);
    ensure!(
        x.len() == t_len * hid,
        "route: x has {} values, want {t_len}x{hid}",
        x.len()
    );
    let w = p.router.context("route: no router")?.mat(0, e, hid)?;
    let bias = match p.bias {
        Some(b) => b.vec(e)?.into_owned(),
        None => vec![0.0; e],
    };
    let ng = p.n_group.max(1);
    ensure!(
        e % ng == 0 && p.topk_group <= ng,
        "route: {e} experts in {ng} groups, keep {}",
        p.topk_group
    );
    let gs = e / ng;
    let (mut ids, mut wts) = (Vec::with_capacity(t_len * k), Vec::with_capacity(t_len * k));
    for xt in x.chunks_exact(hid) {
        let s: Vec<f32> = matvec(&w, e, hid, xt)
            .into_iter()
            .map(|a| 1.0 / (1.0 + (-a).exp()))
            .collect();
        let mut choice: Vec<f32> = s.iter().zip(&bias).map(|(s, b)| s + b).collect();
        if ng > 1 {
            let score = |gi: usize| {
                let mut v = choice[gi * gs..(gi + 1) * gs].to_vec();
                v.sort_by(|a, b| b.total_cmp(a));
                v.iter().take(2).sum::<f32>()
            };
            let mut groups: Vec<usize> = (0..ng).collect();
            groups.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
            for &gi in &groups[p.topk_group..] {
                choice[gi * gs..(gi + 1) * gs].fill(f32::NEG_INFINITY);
            }
        }
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| choice[b].total_cmp(&choice[a]).then(a.cmp(&b)));
        let chosen = &order[..k];
        let sum: f32 = chosen.iter().map(|&i| s[i]).sum();
        for &i in chosen {
            let w = if p.norm_topk_prob {
                s[i] / (sum + 1e-20)
            } else {
                s[i]
            };
            ids.push(i as u32);
            wts.push(w * p.scale);
        }
    }
    Ok((ids, wts))
}

/// The routed experts for a given routing (`ids` / `wts` `[t][k]`), without the shared expert:
/// `y[t] = Σ_j wts[t][j] · expert_{ids[t][j]}(x[t])`. Ids ≥ `n_expert` are skipped.
pub fn experts(
    p: &MoeParams,
    x: &[f32],
    t_len: usize,
    ids: &[u32],
    wts: &[f32],
) -> Result<Vec<f32>> {
    let (hid, ff, k) = (p.hidden, p.ff_exp, p.top_k);
    ensure!(
        ids.len() == t_len * k && wts.len() == t_len * k,
        "experts: routing is not [{t_len}][{k}]"
    );
    let (gw, uw, dw) = (
        p.gate_exps.context("no gate_exps")?,
        p.up_exps.context("no up_exps")?,
        p.down_exps.context("no down_exps")?,
    );
    // (token, slot) pairs per expert, so each expert is dequantised once
    let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); p.n_expert];
    for (i, &id) in ids.iter().enumerate() {
        if (id as usize) < p.n_expert {
            by_expert[id as usize].push((i / k, i));
        }
    }
    // the routed experts in parallel (dequant + every (token, slot) pair's MLP per task),
    // then the weighted sums into `y` in the serial loop's exact order (expert id, then
    // pair) — so the floats round identically and the oracle stays bit-exact
    use rayon::prelude::*;
    let active: Vec<(usize, &Vec<(usize, usize)>)> = by_expert
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let outs: Vec<Vec<Vec<f32>>> = active
        .par_iter()
        .map(|&(e, pairs)| -> Result<Vec<Vec<f32>>> {
            let (g, u, d) = (
                gw.mat(e, ff, hid)?,
                uw.mat(e, ff, hid)?,
                dw.mat(e, hid, ff)?,
            );
            Ok(pairs
                .iter()
                .map(|&(t, _)| mlp(&g, &u, &d, ff, hid, &x[t * hid..(t + 1) * hid], p.limit_exp))
                .collect())
        })
        .collect::<Result<_>>()?;
    let mut y = vec![0f32; t_len * hid];
    for ((_, pairs), outs) in active.iter().zip(outs) {
        for (&(t, i), out) in pairs.iter().zip(outs) {
            for (o, v) in y[t * hid..(t + 1) * hid].iter_mut().zip(out) {
                *o += wts[i] * v;
            }
        }
    }
    Ok(y)
}

/// The shared expert alone, `[t_len][hidden]` (zeros if the layer has none).
pub fn shared_expert(p: &MoeParams, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
    let hid = p.hidden;
    ensure!(
        x.len() == t_len * hid,
        "shared_expert: x has {} values, want {t_len}x{hid}",
        x.len()
    );
    let (Some(gw), Some(uw), Some(dw)) = (p.sh_gate, p.sh_up, p.sh_down) else {
        return Ok(vec![0.0; t_len * hid]);
    };
    let ff = p.ff_shexp;
    let (g, u, d) = (
        gw.mat(0, ff, hid)?,
        uw.mat(0, ff, hid)?,
        dw.mat(0, hid, ff)?,
    );
    Ok(x.chunks_exact(hid)
        .flat_map(|xt| mlp(&g, &u, &d, ff, hid, xt, p.limit_shexp))
        .collect())
}

/// One MoE layer FFN (`Glm5NextTextMoE.forward`): `x` `[t_len][hidden]` (the post-`ffn_norm`
/// input), returns routed + shared, `[t_len][hidden]`.
pub fn moe_layer(p: &MoeParams, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
    let (ids, wts) = route(p, x, t_len)?;
    let mut y = experts(p, x, t_len, &ids, &wts)?;
    for (o, s) in y.iter_mut().zip(shared_expert(p, x, t_len)?) {
        *o += s;
    }
    Ok(y)
}

/// Dense FFN of the first `leading_dense_block_count` layers: `[t_len][hidden]` in and out.
pub fn dense_ffn(p: &MoeParams, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
    let (hid, ff) = (p.hidden, p.ff_dense);
    ensure!(
        x.len() == t_len * hid,
        "dense_ffn: x has {} values, want {t_len}x{hid}",
        x.len()
    );
    let (g, u, d) = (
        p.dense_gate.context("no dense ffn_gate")?.mat(0, ff, hid)?,
        p.dense_up.context("no dense ffn_up")?.mat(0, ff, hid)?,
        p.dense_down.context("no dense ffn_down")?.mat(0, hid, ff)?,
    );
    Ok(x.chunks_exact(hid)
        .flat_map(|xt| mlp(&g, &u, &d, ff, hid, xt, p.limit_shexp))
        .collect())
}

/// The layer's FFN, dense or MoE: what W1's forward calls after `ffn_norm`.
pub fn ffn(p: &MoeParams, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
    if p.is_dense() {
        dense_ffn(p, x, t_len)
    } else {
        moe_layer(p, x, t_len)
    }
}

// ---------------------------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------------------------

/// The naive kernels (NVRTC source); launch plan in its header. Also compiled as host C++ and
/// executed on the fixtures by `scripts/glm5_moe_cuda_host.sh`.
pub const SRC: &str = include_str!("moe_cuda.cuh");

/// The KDA projection kernels ([`kda_dev::K5_SRC`]), appended to [`SRC`] at compile so
/// `g5_dot`/`g5_deq` are in scope for them.
pub const K5_SRC: &str = concat!(
    include_str!("moe_cuda.cuh"),
    "\n",
    include_str!("kda_k5.cuh")
);

#[cfg(feature = "cuda")]
#[path = "moe_gpu.rs"]
pub mod gpu;

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
