//! GLM-5.3-Flash KDA layers (Kimi Delta Attention), W4 of the port.
//!
//! KDA is the GDN of Qwen3-Next with two structural changes: the decay gate is
//! **per-channel (a vector over the head's key dims)** instead of per-head (a scalar), and
//! GLM-5.3-Flash runs it through the "safe gate" sigmoid form (`gate_lower_bound = -5.0`)
//! rather than the softplus form. Everything else — the short depthwise causal conv on
//! q/k/v with SiLU, FLA-style l2-norm on q/k, the delta-rule recurrence
//! `S ← g⊙S + k ⊗ β·(v − Sᵀk)`, and the gated output norm — matches the tang GDN reference
//! ([`super::super::reference`], `Hparams::gdn`), with `nh == nk == nv` (KDA is not grouped).
//!
//! The gate indexing matters and is easy to get wrong: in HF's
//! `recurrent_kimi_delta_attention`, `g` is `[b][t][nh][hd]` and `g_i = g[:, i][..., None]`,
//! so `state * g_i` scales state row `a` (the **key** dim) by `exp(g[a])`. In the chunked
//! path the same appears as `(k_i * (g_last - g_i).exp()).T @ v_new`. All math below uses
//! that convention: **gate channel = key dim**.
//!
//! Verified against Hugging Face `transformers` 5.20.2
//! (`models/glm5_next/modeling_glm5_next.py`, fp32 reference fallbacks) on the fixtures in
//! `tests/fixtures/glm5/kda/` (fixed seed; `meta.json` there documents the geometry):
//! `kda_decode` (token-serial) and `kda_chunk` (the FLA chunked decomposition HF's
//! `chunk_kimi_delta_attention` uses) agree with each other and with HF, and the full layer
//! (projections → conv → gate → scan → gated norm) matches HF's `Glm5NextTextLinearAttention`
//! including the committed conv/recurrent states, for prefill and decode.
//!
//! # Public API (what W1 wires)
//!
//! ```text
//! pub struct KdaConfig { heads, head_dim, d_conv, gate_lower_bound: Option<f32>, eps }
//! pub struct KdaState(pub Vec<f32>);   // [slot][head][key][value], f32
//! pub struct ConvState(pub Vec<f32>);  // [slot][chan][d_conv-1], oldest tap first
//!
//! // gates (input x is the post-attn_norm hidden, [b][t][hidden])
//! pub fn forget_gate(x: &[f32], w: &ForgetGateW, cfg: &KdaConfig, t: usize) -> Vec<f32>
//! //   -> [t][nh*hd], PRE-EXP, safe-gate or softplus form per cfg
//! pub fn beta_gate(x: &[f32], w: &[f32], cfg: &KdaConfig, b: usize, t: usize) -> Vec<f32>
//! //   -> [b][t][nh], post-sigmoid
//!
//! // conv on stacked qkv projections (x [b][t][3*nh*hd], q|k|v stacked per token)
//! pub fn conv1d_prefill(x: &[f32], w: &[f32], conv: &mut ConvState, cfg: &KdaConfig,
//!                       b: usize, t: usize) -> Result<Vec<f32>>     // [b][t][3*nh*hd]
//! pub fn conv1d_decode(x: &[f32], w: &[f32], conv: &mut ConvState,
//!                      cfg: &KdaConfig, b: usize) -> Result<Vec<f32>> // [b][3*nh*hd]
//!
//! // the delta-rule core. qkv: conv output (SiLU applied, NOT l2-normed — the norm is
//! // applied inside, per HF's use_qk_l2norm_in_kernel=True). g: pre-exp, beta: post-sigmoid.
//! // Both update `state` in place and return the core output o, pre-gated-norm:
//! pub fn kda_decode(qkv: &[f32], g: &[f32], beta: &[f32], state: &mut KdaState,
//!                   cfg: &KdaConfig, b: usize) -> Result<Vec<f32>>         // [b][nh*hd]
//! pub fn kda_chunk(qkv: &[f32], g: &[f32], beta: &[f32], state: &mut KdaState,
//!                  cfg: &KdaConfig, b: usize, t: usize, chunk: usize) -> Result<Vec<f32>>
//!
//! // everything above fused (W1 calls just this per KDA layer):
//! pub fn kda_layer(x: &[f32], w: &KdaLayerW, cfg: &KdaConfig, conv: &mut ConvState,
//!                  state: &mut KdaState, b: usize, t: usize) -> Result<Vec<f32>>
//! ```
//!
//! `t == 1` uses the decode conv + `kda_decode`; `t > 1` the prefill conv + `kda_chunk`
//! (chunk 64, HF's). CUDA: the kernels in [`cuda::SRC`] (feature `cuda`, NVRTC), a conv
//! pair and a per-head scan adapted from the GDN kernels (`gdn_step_body`), the scan valid
//! for any window width so decode and prefill share it; `cuda::CUDA_POD_CMDS` has the pod
//! parity commands.

use anyhow::{ensure, Result};

// ---------------------------------------------------------------------------
// config, weights, states

/// KDA geometry of one layer, from GGUF metadata: heads = `attention.head_count`
/// (KDA is not grouped: one k and one v per head), `head_dim` = `kda.head_dim` (128),
/// `d_conv` = `ssm.conv_kernel` (4), `gate_lower_bound` = `kda.gate_lower_bound`
/// (`Some(-5.0)` on GLM-5.3-Flash; `None` selects the Qwen3-Next softplus form),
/// `eps` = the model's rms_norm_eps (1e-5).
#[derive(Clone, Copy, Debug)]
pub struct KdaConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub d_conv: usize,
    pub gate_lower_bound: Option<f32>,
    pub eps: f32,
}

impl KdaConfig {
    /// Stacked q/k/v width of the conv.
    pub fn conv_dim(&self) -> usize {
        3 * self.heads * self.head_dim
    }

    /// The GLM-5.3-Flash geometry (64 heads × 128, conv 4, safe gate −5, eps 1e-5).
    pub fn glm53() -> Self {
        Self {
            heads: 64,
            head_dim: 128,
            d_conv: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1e-5,
        }
    }
}

/// Forget-gate weights, HF layout (row-major `out × in`). In the GGUF, `ssm_a` is already
/// **`-exp(A_log)`** (the converter negates it) — build `decay_rate` with
/// [`Self::decay_rate_from_gguf`].
#[derive(Clone, Debug)]
pub struct ForgetGateW {
    /// (ssm_f_a) `[head_dim][hidden]`
    pub f_a: Vec<f32>,
    /// (ssm_f_b) `[nh*hd][head_dim]`
    pub f_b: Vec<f32>,
    /// (ssm_dt) `[nh*hd]`
    pub dt_bias: Vec<f32>,
    /// `exp(A_log)` per head, `[heads]` (HF stores `A_log`; GGUF `ssm_a` = `-exp(A_log)`).
    pub decay_rate: Vec<f32>,
}

impl ForgetGateW {
    /// `exp(A_log)` from the GGUF's `ssm_a` (= `-exp(A_log)`).
    pub fn decay_rate_from_gguf(ssm_a: &[f32]) -> Vec<f32> {
        ssm_a.iter().map(|&a| -a).collect()
    }
}

/// All weights of one KDA layer, HF layout (GGUF names in parens). `conv` stacks the three
/// depthwise convs: `[3*nh*hd][d_conv]`, `w[chan * d_conv + tap]`, **tap 0 = newest**
/// (HF `Conv1d.weight` order; the GGUF stores them as `[1][chan][1][d_conv]`).
#[derive(Clone, Debug)]
pub struct KdaLayerW {
    /// (attn_q) `[nh*hd][hidden]`
    pub q: Vec<f32>,
    /// (attn_k) `[nh*hd][hidden]`
    pub k: Vec<f32>,
    /// (attn_v) `[nh*hd][hidden]`
    pub v: Vec<f32>,
    /// (ssm_conv1d_q|k|v stacked) `[3*nh*hd][d_conv]`
    pub conv: Vec<f32>,
    /// (ssm_f_a) `[head_dim][hidden]`
    pub f_a: Vec<f32>,
    /// (ssm_f_b) `[nh*hd][head_dim]`
    pub f_b: Vec<f32>,
    /// (ssm_dt) `[nh*hd]`
    pub dt_bias: Vec<f32>,
    /// exp(A_log), `[heads]` (GGUF `ssm_a` negated; see [`ForgetGateW`])
    pub decay_rate: Vec<f32>,
    /// (ssm_beta) `[heads][hidden]`
    pub b_proj: Vec<f32>,
    /// (ssm_g_a) `[head_dim][hidden]`
    pub g_a: Vec<f32>,
    /// (ssm_g_b) `[nh*hd][head_dim]`
    pub g_b: Vec<f32>,
    /// (ssm_norm) `[head_dim]`
    pub o_norm: Vec<f32>,
    /// (attn_out) `[hidden][nh*hd]`
    pub o_proj: Vec<f32>,
    /// block input width
    pub hidden: usize,
}

/// The recurrent state of one KDA layer: `[slot][head][key][value]` f32, row-major with the
/// key index slow (`s[a*d + b]` is key dim `a`'s contribution to value dim `b`) — HF's
/// `[b][nh][k][v]` contiguous layout, so HF cache tensors map 1:1.
///
/// GLM-5.3-Flash: 64 · 128 · 128 · 4 B = **4 MiB/layer**, 34 KDA layers = 141.4 MiB.
#[derive(Clone, Debug)]
pub struct KdaState(pub Vec<f32>);

impl KdaState {
    pub fn zeros(cfg: &KdaConfig, slots: usize) -> Self {
        Self(vec![0f32; slots * cfg.heads * cfg.head_dim * cfg.head_dim])
    }

    /// One head's `[key][value]` slice.
    pub fn head<'a>(&'a mut self, cfg: &KdaConfig, slot: usize, h: usize) -> &'a mut [f32] {
        let n = cfg.head_dim * cfg.head_dim;
        &mut self.0[(slot * cfg.heads + h) * n..(slot * cfg.heads + h + 1) * n]
    }
}

/// The depthwise conv history: `[slot][chan][d_conv - 1]`, the **pre-SiLU conv inputs** of
/// the last `d_conv - 1` tokens, oldest first (HF `conv_states`, `[b][conv_dim][d_conv-1]`).
/// GLM-5.3-Flash: 3·64·128 · 3 · 4 B = **288 KiB/layer**, 34 layers = 9.4 MiB.
#[derive(Clone, Debug)]
pub struct ConvState(pub Vec<f32>);

impl ConvState {
    pub fn zeros(cfg: &KdaConfig, slots: usize) -> Self {
        Self(vec![
            0f32;
            slots * cfg.conv_dim() * cfg.d_conv.saturating_sub(1)
        ])
    }

    fn chan<'a>(&'a mut self, cfg: &KdaConfig, slot: usize, c: usize) -> &'a mut [f32] {
        let m = cfg.d_conv - 1;
        &mut self.0[(slot * cfg.conv_dim() + c) * m..(slot * cfg.conv_dim() + c + 1) * m]
    }
}

// ---------------------------------------------------------------------------
// small math (the reference module's equivalents are private)

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ggml softplus: `log(1 + e^x)`, passed through above 20.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// FLA l2norm: `x / sqrt(Σx² + eps)` — eps *inside* the sqrt, on the squared norm; the sum
/// in f64 (HF computes fp32; f64 keeps the CPU reference well under the tolerance).
#[inline]
fn l2_norm(x: &mut [f32], eps: f32) {
    let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let r = 1.0 / ((s as f32) + eps).sqrt();
    for v in x {
        *v *= r;
    }
}

/// Strict-fp32 RMSNorm of `Glm5NextTextRMSNormGated` (mean in f64).
fn rms_norm_mul(x: &mut [f32], w: &[f32], eps: f32) {
    let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let r = 1.0 / ((s / x.len() as f64) as f32 + eps).sqrt();
    for (v, g) in x.iter_mut().zip(w) {
        *v = *v * r * g;
    }
}

/// `Y[t] = W X[t]`: `W` `[rows][cols]` row-major, `X` `[t][cols]`; → `[t][rows]`.
pub(crate) fn matmul(w: &[f32], rows: usize, cols: usize, x: &[f32], t: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "weight");
    assert_eq!(x.len(), t * cols, "input");
    let mut y = vec![0f32; t * rows];
    for ti in 0..t {
        let xr = &x[ti * cols..(ti + 1) * cols];
        for (r, yo) in y[ti * rows..(ti + 1) * rows].iter_mut().enumerate() {
            let wr = &w[r * cols..(r + 1) * cols];
            let mut s = 0f64;
            for (a, b) in wr.iter().zip(xr) {
                s += (*a as f64) * (*b as f64);
            }
            *yo = s as f32;
        }
    }
    y
}

// ---------------------------------------------------------------------------
// gates

/// The forget gate of `t` tokens → `[t][nh*hd]`, **pre-exp** (the scan applies `exp`).
/// With a lower bound (GLM-5.3-Flash):
/// `g = lb · σ(exp(A_log_h) · (f_b(f_a(x)) + dt_bias))`;
/// without (the Qwen3-Next form): `g = −exp(A_log_h) · softplus(…)` — already negative,
/// like tang's `ssm_a`.
pub fn forget_gate(x: &[f32], w: &ForgetGateW, cfg: &KdaConfig, t: usize) -> Vec<f32> {
    let hidden = w.f_a.len() / cfg.head_dim;
    let nh = w.decay_rate.len();
    let a = matmul(&w.f_a, cfg.head_dim, hidden, x, t);
    let g1 = matmul(&w.f_b, nh * cfg.head_dim, cfg.head_dim, &a, t);
    let hd = cfg.head_dim;
    let mut out = vec![0f32; t * nh * hd];
    for ti in 0..t {
        for h in 0..nh {
            let rate = w.decay_rate[h];
            for j in 0..hd {
                let pre = g1[ti * nh * hd + h * hd + j] + w.dt_bias[h * hd + j];
                out[ti * nh * hd + h * hd + j] = match cfg.gate_lower_bound {
                    Some(lb) => lb * sigmoid(rate * pre),
                    None => -rate * softplus(pre),
                };
            }
        }
    }
    out
}

/// `β = σ(b_proj x)` → `[b][t][nh]`, post-sigmoid. `w` = `[heads][hidden]`.
pub fn beta_gate(x: &[f32], w: &[f32], cfg: &KdaConfig, b: usize, t: usize) -> Vec<f32> {
    let raw = matmul(w, cfg.heads, w.len() / cfg.heads, x, b * t);
    raw.iter().map(|&v| sigmoid(v)).collect()
}

// ---------------------------------------------------------------------------
// conv

/// Depthwise causal conv + SiLU over the stacked projections. `x` = `[b][t][3*nh*hd]`
/// (q|k|v blocks per token, as `torch.cat([q_proj(x), k_proj(x), v_proj(x)], -1)`).
/// `w` = `[chan][d_conv]`, tap 0 = **newest** (HF `Conv1d` order). The last `d_conv − 1`
/// inputs are committed to `conv`. Returns `[b][t][3*nh*hd]`.
pub fn conv1d_prefill(
    x: &[f32],
    w: &[f32],
    conv: &mut ConvState,
    cfg: &KdaConfig,
    b: usize,
    t: usize,
) -> Result<Vec<f32>> {
    let ch = cfg.conv_dim();
    let kc = cfg.d_conv;
    ensure!(x.len() == b * t * ch && w.len() == ch * kc, "conv shapes");
    let mut y = vec![0f32; b * t * ch];
    for bi in 0..b {
        for c in 0..ch {
            let hist = conv.chan(cfg, bi, c).to_vec();
            // out[t] = Σ_i w[i]·e[t - (kc-1) + i] over e = [hist | x] (F.conv1d semantics)
            for ti in 0..t {
                let mut acc = 0f64;
                for i in 0..kc {
                    let p = ti as i64 - (kc as i64 - 1) + i as i64;
                    let e = if p < 0 {
                        hist[(p + (kc as i64 - 1)) as usize]
                    } else {
                        x[(bi * t + p as usize) * ch + c]
                    };
                    acc += (w[c * kc + i] as f64) * (e as f64);
                }
                y[(bi * t + ti) * ch + c] = silu(acc as f32);
            }
            // commit: conv <- the last kc-1 inputs of [hist | x]
            for s in 0..kc - 1 {
                let p = t as i64 + s as i64;
                conv.chan(cfg, bi, c)[s] = if p < (kc - 1) as i64 {
                    hist[p as usize]
                } else {
                    x[(bi * t + (p - (kc as i64 - 1)) as usize) * ch + c]
                };
            }
        }
    }
    Ok(y)
}

/// The one-token conv of decode. `x` = `[b][3*nh*hd]` → `[b][3*nh*hd]`, `conv` updated.
pub fn conv1d_decode(
    x: &[f32],
    w: &[f32],
    conv: &mut ConvState,
    cfg: &KdaConfig,
    b: usize,
) -> Result<Vec<f32>> {
    let ch = cfg.conv_dim();
    let kc = cfg.d_conv;
    ensure!(x.len() == b * ch && w.len() == ch * kc, "conv shapes");
    let mut y = vec![0f32; b * ch];
    for bi in 0..b {
        for c in 0..ch {
            let hist = conv.chan(cfg, bi, c).to_vec();
            // out = Σ_i w[i]·e[kc-1-i] over e = [hist | x_row] (tap 0 = newest)
            let mut acc = 0f64;
            for i in 0..kc {
                let e = if i == kc - 1 { x[bi * ch + c] } else { hist[i] };
                acc += (w[c * kc + i] as f64) * (e as f64);
            }
            y[bi * ch + c] = silu(acc as f32);
            // shift: the oldest drops, the new input becomes the newest
            for s in 0..kc - 1 {
                conv.chan(cfg, bi, c)[s] = if s + 1 < kc - 1 {
                    hist[s + 1]
                } else {
                    x[bi * ch + c]
                };
            }
        }
    }
    Ok(y)
}

// ---------------------------------------------------------------------------
// the delta-rule core

/// One token of one head: l2-norm q/k (eps 1e-6), `q /= √d`, then
/// `S ← exp(g)⊙S` (per key row `a`, `g[a]` the pre-exp gate channel), `δ = β(v − Sᵀk)`,
/// `S ← S + kδᵀ`, `o = Sᵀq`. Accumulation in f64.
#[inline]
#[allow(clippy::too_many_arguments)]
fn head_step(
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    g: &[f32],
    beta: f32,
    st: &mut [f32],
    o: &mut [f32],
    d: usize,
) {
    l2_norm(q, 1e-6);
    l2_norm(k, 1e-6);
    let scale = 1.0 / (d as f32).sqrt();
    for x in q.iter_mut() {
        *x *= scale;
    }
    // pass 1: decay per key row + sk[b] = Σ_a k[a]·S[a][b] (Sᵀk)
    let mut sk = vec![0f64; d];
    for a in 0..d {
        let ka = k[a] as f64;
        let decay = g[a].exp() as f64;
        let row = &mut st[a * d..(a + 1) * d];
        for (s, acc) in row.iter_mut().zip(&mut sk) {
            *s = ((*s as f64) * decay) as f32;
            *acc += ka * (*s as f64);
        }
    }
    // pass 2: δ = β(v − sk); S[a][b] += k[a]·δ[b]; o[b] += q[a]·S[a][b]
    let mut oo = vec![0f64; d];
    for a in 0..d {
        let (ka, qa) = (k[a] as f64, q[a] as f64);
        let row = &mut st[a * d..(a + 1) * d];
        for (((s, ob), vv), bk) in row.iter_mut().zip(&mut oo).zip(v).zip(&sk) {
            let delta = beta as f64 * (*vv as f64 - *bk);
            *s = (*s as f64 + ka * delta) as f32;
            *ob += qa * (*s as f64);
        }
    }
    for (ob, y) in oo.iter().zip(&mut o[..d]) {
        *y = *ob as f32;
    }
}

/// The token-serial decode step: `qkv` `[b][3*nh*hd]` (conv output, **not** l2-normed),
/// `g` `[b][nh*hd]` pre-exp, `beta` `[b][nh]` post-sigmoid. Updates `state` in place,
/// returns `o` = `[b][nh*hd]` (pre-gated-norm).
pub fn kda_decode(
    qkv: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut KdaState,
    cfg: &KdaConfig,
    b: usize,
) -> Result<Vec<f32>> {
    let (nh, hd) = (cfg.heads, cfg.head_dim);
    let ch = 3 * nh * hd;
    ensure!(
        qkv.len() == b * ch && g.len() == b * nh * hd && beta.len() == b * nh,
        "decode shapes"
    );
    let mut out = vec![0f32; b * nh * hd];
    for bi in 0..b {
        let row = &qkv[bi * ch..(bi + 1) * ch];
        for h in 0..nh {
            let mut q = row[h * hd..(h + 1) * hd].to_vec();
            let mut k = row[nh * hd + h * hd..nh * hd + (h + 1) * hd].to_vec();
            let v = &row[2 * nh * hd + h * hd..2 * nh * hd + (h + 1) * hd];
            let gh = &g[bi * nh * hd + h * hd..bi * nh * hd + (h + 1) * hd];
            let st = state.head(cfg, bi, h);
            let o = &mut out[bi * nh * hd + h * hd..bi * nh * hd + (h + 1) * hd];
            head_step(&mut q, &mut k, v, gh, beta[bi * nh + h], st, o, hd);
        }
    }
    Ok(out)
}

/// The FLA chunked prefill — a faithful port of HF's `chunk_kimi_delta_attention`
/// (the decomposition the fast kernels use; the CPU side pins its math down).
/// `qkv` `[b][t][3*nh*hd]`, `g` `[b][t][nh*hd]`, `beta` `[b][t][nh]`; `state` is both the
/// initial state (read) and the final state (written in place). Returns `o` = `[b][t][nh*hd]`.
///
/// Per chunk (zeros pad `t` up to a multiple of `chunk`, as HF does):
/// - cumsum gate `G`; `A[i][j] = −β_i·β_j·Σ_x k_i[x]·k_j[x]·exp(G_i[x] − G_j[x])` (i > j),
///   the strictly-lower delta-weighted gram, driven to its `(I + L)⁻¹`-style fixpoint
///   (HF's triangular loop), then `A ← A + I`;
/// - `V' = A·(v⊙β)`, `K' = A·(k⊙β·exp(G))`;
/// - `o_j = (q_j ⊙ exp(G_j))·S + Σ_{c ≤ j} (q_j·k_c·exp(G_j − G_c)) · (V' − K'·S)_c`;
/// - `S ← exp(G_last)⊙S + (k⊙exp(G_last − G))ᵀ·(V' − K'·S)`.
#[allow(clippy::too_many_arguments)]
pub fn kda_chunk(
    qkv: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut KdaState,
    cfg: &KdaConfig,
    b: usize,
    t: usize,
    chunk: usize,
) -> Result<Vec<f32>> {
    let (nh, hd, d) = (cfg.heads, cfg.head_dim, cfg.head_dim);
    let ch = 3 * nh * hd;
    ensure!(
        qkv.len() == b * t * ch && g.len() == b * t * nh * hd && beta.len() == b * t * nh,
        "chunk shapes"
    );
    ensure!(chunk > 0 && t > 0, "chunk size");
    let c = chunk;
    let pad = (c - t % c) % c;
    let tt = t + pad;
    let nc = tt / c;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0f32; b * t * nh * hd];

    for bi in 0..b {
        for h in 0..nh {
            // l2norm q/k, q /= sqrt(d); padded rows zero.
            let mut q = vec![0f32; tt * hd];
            let mut k = vec![0f32; tt * hd];
            let mut v = vec![0f32; tt * hd];
            for ti in 0..t {
                let row = &qkv[(bi * t + ti) * ch..(bi * t + ti + 1) * ch];
                q[ti * hd..(ti + 1) * hd].copy_from_slice(&row[h * hd..(h + 1) * hd]);
                k[ti * hd..(ti + 1) * hd]
                    .copy_from_slice(&row[nh * hd + h * hd..nh * hd + (h + 1) * hd]);
                v[ti * hd..(ti + 1) * hd]
                    .copy_from_slice(&row[2 * nh * hd + h * hd..2 * nh * hd + (h + 1) * hd]);
                l2_norm(&mut q[ti * hd..(ti + 1) * hd], 1e-6);
                l2_norm(&mut k[ti * hd..(ti + 1) * hd], 1e-6);
                for x in &mut q[ti * hd..(ti + 1) * hd] {
                    *x *= scale;
                }
            }
            // per-key-dim cumsum gate G and beta. G is the **within-chunk** cumsum (HF
            // pads g with 0 and cumsums AFTER reshaping to chunks, so G restarts at each
            // chunk boundary — the absolute exp(G) in the inter term, k_cumdecay and the
            // state decay is local; only differences would survive a global cumsum).
            // Pad rows add 0, carrying G forward; v/k/beta are 0 there so they're inert.
            let mut gc = vec![0f32; tt * hd];
            let mut bet = vec![0f32; tt];
            for ci in 0..nc {
                let o0 = ci * c;
                for ti in 0..c {
                    let r = o0 + ti;
                    bet[r] = if r < t {
                        beta[(bi * t + r) * nh + h]
                    } else {
                        0.0
                    };
                    for j in 0..hd {
                        let gv = if r < t {
                            g[(bi * t + r) * nh * hd + h * hd + j]
                        } else {
                            0.0
                        };
                        gc[r * hd + j] = gv + if ti > 0 { gc[(r - 1) * hd + j] } else { 0.0 };
                    }
                }
            }
            let st = state.head(cfg, bi, h);
            for ci in 0..nc {
                let o0 = ci * c;
                // A0: strictly-lower −(k_i·β_i)·k_jᵀ weighted by exp(G_i − G_j). Note β
                // applies to the ROW i only (HF's k_beta = key·β, contracted against raw
                // key_j), not to both indices.
                let mut a = vec![0f32; c * c];
                for i in 1..c {
                    for j in 0..i {
                        let mut s = 0f64;
                        for x in 0..hd {
                            let w = (k[(o0 + i) * hd + x] * bet[o0 + i]) as f64
                                * (k[(o0 + j) * hd + x]) as f64
                                * ((gc[(o0 + i) * hd + x] - gc[(o0 + j) * hd + x]).exp() as f64);
                            s += w;
                        }
                        a[i * c + j] = -(s as f32);
                    }
                }
                // HF's triangular fixpoint: row i is finalised from the ORIGINAL A0 row
                // (cloned before the update) against the already-finalised rows < i:
                // A[i][j] = A0[i][j] + Σ_{m<i} A0[i][m]·A[m][j].
                for i in 1..c {
                    let row: Vec<f32> = a[i * c..i * c + i].to_vec();
                    for j in 0..i {
                        let mut s = 0f32;
                        for m in 0..i {
                            s += row[m] * a[m * c + j];
                        }
                        a[i * c + j] = row[j] + s;
                    }
                }
                for i in 0..c {
                    a[i * c + i] += 1.0;
                }
                // V' = A·(v⊙β); K' = A·(k⊙β·exp(G))
                let mut vp = vec![0f32; c * hd];
                let mut kp = vec![0f32; c * hd];
                for i in 0..c {
                    for j in 0..c {
                        let w = a[i * c + j];
                        if w == 0.0 {
                            continue;
                        }
                        for x in 0..hd {
                            vp[i * hd + x] += w * v[(o0 + j) * hd + x] * bet[o0 + j];
                            kp[i * hd + x] += w
                                * k[(o0 + j) * hd + x]
                                * bet[o0 + j]
                                * gc[(o0 + j) * hd + x].exp();
                        }
                    }
                }
                // v_new = V' − K'·S
                let mut vnew = vec![0f64; c * hd];
                for (i, vn) in vnew.chunks_mut(hd).enumerate() {
                    for x in 0..hd {
                        let mut s = 0f64;
                        for (a2, sr) in (0..hd).zip(st.chunks(d)) {
                            s += (kp[i * hd + a2] as f64) * (sr[x] as f64);
                        }
                        vn[x] = vp[i * hd + x] as f64 - s;
                    }
                }
                // o_i = (q_i ⊙ exp(G_i))·S + Σ_{j ≤ i} (q_i·k_j·exp(G_i − G_j)) · v_new_j
                let mut oj = vec![0f64; c * hd];
                for (i, oir) in oj.chunks_mut(hd).enumerate() {
                    for (x, sr) in (0..hd).zip(st.chunks(d)) {
                        let w = (q[(o0 + i) * hd + x] * gc[(o0 + i) * hd + x].exp()) as f64;
                        if w == 0.0 {
                            continue;
                        }
                        for (o, vv) in oir.iter_mut().zip(0..hd) {
                            *o += w * (sr[vv] as f64);
                        }
                    }
                    for j in 0..=i {
                        let mut s = 0f64;
                        for x in 0..hd {
                            s += (q[(o0 + i) * hd + x] as f64)
                                * (k[(o0 + j) * hd + x] as f64)
                                * ((gc[(o0 + i) * hd + x] - gc[(o0 + j) * hd + x]).exp() as f64);
                        }
                        for (o, vn) in oir.iter_mut().zip(&vnew[j * hd..(j + 1) * hd]) {
                            *o += s * *vn;
                        }
                    }
                }
                // S ← exp(G_last)⊙S + (k⊙exp(G_last − G))ᵀ · v_new
                let last = (o0 + c - 1) * hd;
                for (x, sr) in (0..hd).zip(st.chunks_mut(d)) {
                    let dl = gc[last + x].exp();
                    let mut upd = vec![0f64; d];
                    for j in 0..c {
                        let w = (k[(o0 + j) * hd + x]
                            * (gc[last + x] - gc[(o0 + j) * hd + x]).exp())
                            as f64;
                        if w == 0.0 {
                            continue;
                        }
                        for (u, vn) in upd.iter_mut().zip(&vnew[j * hd..(j + 1) * hd]) {
                            *u += w * *vn;
                        }
                    }
                    for (s, u) in sr.iter_mut().zip(&upd) {
                        *s = ((*s as f64) * dl as f64 + *u) as f32;
                    }
                }
                for i in 0..c {
                    if o0 + i >= t {
                        break;
                    }
                    let base = (bi * t + o0 + i) * nh * hd + h * hd;
                    for x in 0..hd {
                        out[base + x] = oj[i * hd + x] as f32;
                    }
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// the full layer

/// One KDA layer over `t` tokens (`t == 1`: decode conv + [`kda_decode`]; else prefill conv
/// + [`kda_chunk`], HF's chunk size 64). `x` = `[b][t][hidden]`, the post-attn_norm hidden.
///
/// Updates `conv` and `state` in place; returns `[b][t][hidden]`.
pub fn kda_layer(
    x: &[f32],
    w: &KdaLayerW,
    cfg: &KdaConfig,
    conv: &mut ConvState,
    state: &mut KdaState,
    b: usize,
    t: usize,
) -> Result<Vec<f32>> {
    kda_layer_taps(x, w, cfg, conv, state, b, t, None)
}

/// [`kda_layer`], optionally capturing every intermediate into `taps` (the
/// `TANG_GLM5_KDA_DEBUG` shadow check compares them against the GPU's, see
/// [`super::kda_debug`]). The math is the same code path either way.
#[allow(clippy::too_many_arguments)]
pub fn kda_layer_taps(
    x: &[f32],
    w: &KdaLayerW,
    cfg: &KdaConfig,
    conv: &mut ConvState,
    state: &mut KdaState,
    b: usize,
    t: usize,
    mut taps: Option<&mut tang_verify::Taps>,
) -> Result<Vec<f32>> {
    let (nh, hd) = (cfg.heads, cfg.head_dim);
    let hidden = w.hidden;
    ensure!(x.len() == b * t * hidden, "layer input");
    // staging probe (CPU side): dump the same buffers the GPU path dumps, so the
    // staging diff runs without a GPU. Layout/units equal to kda_dev's dump.
    if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
        eprintln!("kda cpu x[0..8]: {:?}", &x[..8.min(x.len())]);
    }
    // projections, stacked q|k|v per token
    let qp = matmul(&w.q, nh * hd, hidden, x, b * t);
    let kp = matmul(&w.k, nh * hd, hidden, x, b * t);
    let vp = matmul(&w.v, nh * hd, hidden, x, b * t);
    let ch = 3 * nh * hd;
    let mut qkv = vec![0f32; b * t * ch];
    for i in 0..b * t {
        qkv[i * ch..i * ch + nh * hd].copy_from_slice(&qp[i * nh * hd..(i + 1) * nh * hd]);
        qkv[i * ch + nh * hd..i * ch + 2 * nh * hd]
            .copy_from_slice(&kp[i * nh * hd..(i + 1) * nh * hd]);
        qkv[i * ch + 2 * nh * hd..(i + 1) * ch]
            .copy_from_slice(&vp[i * nh * hd..(i + 1) * nh * hd]);
    }
    // pre-conv, like kda_dev's `kda gpu qkv` (read from qkvb before the core's conv)
    if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
        eprintln!("kda cpu qkv[0..16]: {:?}", &qkv[..16.min(qkv.len())]);
    }
    tang_verify::tap(&mut taps, super::kda_debug::tap::QKV, || qkv.clone());
    let qkv = if t == 1 {
        conv1d_decode(&qkv, &w.conv, conv, cfg, b)?
    } else {
        conv1d_prefill(&qkv, &w.conv, conv, cfg, b, t)?
    };
    tang_verify::tap(&mut taps, super::kda_debug::tap::CONV, || qkv.clone());
    // gates
    let g = forget_gate(
        x,
        &ForgetGateW {
            f_a: w.f_a.clone(),
            f_b: w.f_b.clone(),
            dt_bias: w.dt_bias.clone(),
            decay_rate: w.decay_rate.clone(),
        },
        cfg,
        b * t,
    );
    let beta = beta_gate(x, &w.b_proj, cfg, b, t);
    if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
        eprintln!("kda cpu g[0..8]: {:?}", &g[..8.min(g.len())]);
        eprintln!("kda cpu beta[0..4]: {:?}", &beta[..4.min(beta.len())]);
    }
    tang_verify::tap(&mut taps, super::kda_debug::tap::G, || g.clone());
    tang_verify::tap(&mut taps, super::kda_debug::tap::BETA, || beta.clone());
    // core
    let o = if t == 1 {
        kda_decode(&qkv, &g, &beta, state, cfg, b)?
    } else {
        kda_chunk(&qkv, &g, &beta, state, cfg, b, t, 64)?
    };
    if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
        // the core output and the final layer output, matching kda_dev's `kda gpu o`/`y`
        eprintln!("kda cpu o[0..8]: {:?}", &o[..8.min(o.len())]);
    }
    // gated output norm: y = RMSNorm(o) * o_norm * σ(g2), g2 = g_b(g_a(x))
    let ga = matmul(&w.g_a, hd, hidden, x, b * t);
    let g2 = matmul(&w.g_b, nh * hd, hd, &ga, b * t);
    let mut y = vec![0f32; b * t * nh * hd];
    for i in 0..b * t {
        for h in 0..nh {
            let dst = &mut y[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd];
            dst.copy_from_slice(&o[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd]);
            rms_norm_mul(dst, &w.o_norm, cfg.eps);
            for (yj, gj) in dst
                .iter_mut()
                .zip(&g2[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd])
            {
                *yj *= sigmoid(*gj);
            }
        }
    }
    let out = matmul(&w.o_proj, hidden, nh * hd, &y, b * t);
    if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
        eprintln!("kda cpu y[0..8]: {:?}", &out[..8.min(out.len())]);
    }
    if let Some(tp) = taps {
        use super::kda_debug::tap;
        tp.put(tap::O, o);
        tp.put(tap::G2, g2);
        tp.put(tap::Y_NORM, y);
        tp.put(tap::OUT, out.clone());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed-seed data generated with the HF module itself; see `meta.json`.
    const FX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glm5/kda");

    fn load(name: &str) -> Vec<f32> {
        let b = std::fs::read(format!("{FX}/{name}.bin")).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(b.len() % 4, 0);
        b.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    // geometry of the fixtures (meta.json): B=1, T=100, H=4, D=16, conv taps 4, lb −5,
    // chunk 64; the layer at hidden 48.
    fn cfg() -> KdaConfig {
        KdaConfig {
            heads: 4,
            head_dim: 16,
            d_conv: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1e-5,
        }
    }

    #[test]
    fn core_recurrent_matches_hf() {
        let (b, t, h, d) = (1, 100, 4, 16);
        let ch = 3 * h * d;
        // the fixtures store q, k, v RAW and SEPARATELY (pre-l2norm); stack them for us
        let (q, k, v) = (load("q"), load("k"), load("v"));
        let (g, beta, st0) = (load("g"), load("beta"), load("state0"));
        let mut qkv = vec![0f32; b * t * ch];
        for i in 0..b * t {
            qkv[i * ch..i * ch + h * d].copy_from_slice(&q[i * h * d..(i + 1) * h * d]);
            qkv[i * ch + h * d..i * ch + 2 * h * d].copy_from_slice(&k[i * h * d..(i + 1) * h * d]);
            qkv[i * ch + 2 * h * d..(i + 1) * ch].copy_from_slice(&v[i * h * d..(i + 1) * h * d]);
        }
        // feed tokens one at a time through kda_decode (conv already applied in HF here)
        let cfg = cfg();
        let mut state = KdaState(st0.clone());
        let mut out = vec![0f32; b * t * h * d];
        for ti in 0..t {
            let o = kda_decode(
                &qkv[ti * ch..(ti + 1) * ch],
                &g[ti * h * d..(ti + 1) * h * d],
                &beta[ti * h..(ti + 1) * h],
                &mut state,
                &cfg,
                b,
            )
            .unwrap();
            out[ti * h * d..(ti + 1) * h * d].copy_from_slice(&o);
        }
        let e = max_abs(&out, &load("out_recurrent"));
        let es = max_abs(&state.0, &load("state_recurrent"));
        assert!(e < 1e-4, "out vs HF: {e}");
        assert!(es < 1e-4, "state vs HF: {es}");
    }

    #[test]
    fn core_chunk_matches_hf_and_recurrent() {
        let (b, t, h, d) = (1, 100, 4, 16);
        let ch = 3 * h * d;
        let (q, k, v) = (load("q"), load("k"), load("v"));
        let (g, beta, st0) = (load("g"), load("beta"), load("state0"));
        let mut qkv = vec![0f32; b * t * ch];
        for i in 0..b * t {
            qkv[i * ch..i * ch + h * d].copy_from_slice(&q[i * h * d..(i + 1) * h * d]);
            qkv[i * ch + h * d..i * ch + 2 * h * d].copy_from_slice(&k[i * h * d..(i + 1) * h * d]);
            qkv[i * ch + 2 * h * d..(i + 1) * ch].copy_from_slice(&v[i * h * d..(i + 1) * h * d]);
        }
        let cfg = cfg();
        let mut state = KdaState(st0.clone());
        let out = kda_chunk(&qkv, &g, &beta, &mut state, &cfg, b, t, 64).unwrap();
        let e = max_abs(&out, &load("out_chunk"));
        assert!(e < 1e-4, "chunk out vs HF: {e}");
        let es = max_abs(&state.0, &load("state_chunk"));
        assert!(es < 1e-4, "chunk state vs HF: {es}");
        // and against the recurrent reference fixture
        let er = max_abs(&out, &load("out_recurrent"));
        let esr = max_abs(&state.0, &load("state_recurrent"));
        assert!(er < 1e-4, "chunk vs recurrent out: {er}");
        assert!(esr < 1e-4, "chunk vs recurrent state: {esr}");
    }

    #[test]
    fn conv_prefill_and_decode_match_hf() {
        let cfg = cfg();
        let (b, t) = (1, 100);
        let ch = cfg.conv_dim();
        // HF conv layout is [b][chan][t] — transpose to our [b][t][chan]
        let x = load("conv_x");
        let mut xt = vec![0f32; b * t * ch];
        for bi in 0..b {
            for c in 0..ch {
                for ti in 0..t {
                    xt[(bi * t + ti) * ch + c] = x[(bi * ch + c) * t + ti];
                }
            }
        }
        let w = load("conv_w"); // [chan][1][taps] == [chan][taps], tap 0 newest
        let mut conv = ConvState::zeros(&cfg, b);
        let y = conv1d_prefill(&xt, &w, &mut conv, &cfg, b, t).unwrap();
        // expected [b][chan][t]
        let want = load("conv_out");
        let mut e = 0f32;
        for bi in 0..b {
            for c in 0..ch {
                for ti in 0..t {
                    let yy = y[(bi * t + ti) * ch + c];
                    e = e.max((yy - want[(bi * ch + c) * t + ti]).abs());
                }
            }
        }
        assert!(e < 1e-4, "prefill conv vs HF: {e}");
        // decode: one more row through conv1d_decode matches HF causal_conv1d_update
        let row = load("conv_row"); // [b][chan][1] == [b][chan]
        let mut conv2 = ConvState(load("conv_state_in")); // [b][chan][3]
        let yd = conv1d_decode(&row, &w, &mut conv2, &cfg, b).unwrap();
        let wantd = load("conv_row_out");
        let mut ed = 0f32;
        for bi in 0..b {
            for c in 0..ch {
                ed = ed.max((yd[bi * ch + c] - wantd[bi * ch + c]).abs());
            }
        }
        assert!(ed < 1e-4, "decode conv vs HF: {ed}");
        let st_out = load("conv_state_out");
        let es = max_abs(&conv2.0, &st_out);
        assert!(es < 1e-4, "conv state after decode vs HF: {es}");
    }

    #[test]
    fn full_layer_prefill_and_decode_match_hf() {
        let cfg = cfg();
        let hidden = 48;
        let w = KdaLayerW {
            q: load("w_q_proj_weight"),
            k: load("w_k_proj_weight"),
            v: load("w_v_proj_weight"),
            conv: load("w_conv1d_weight"), // [chan][1][taps] == [chan][taps]
            f_a: load("w_forget_gate_f_a_proj_weight"),
            f_b: load("w_forget_gate_f_b_proj_weight"),
            dt_bias: load("w_forget_gate_dt_bias"),
            decay_rate: load("w_forget_gate_A_log")
                .iter()
                .map(|a| a.exp())
                .collect(),
            b_proj: load("w_b_proj_weight"),
            g_a: load("w_g_a_proj_weight"),
            g_b: load("w_g_b_proj_weight"),
            o_norm: load("w_o_norm_weight"),
            o_proj: load("w_o_proj_weight"),
            hidden,
        };
        let (b, t) = (1, 100);
        let x = load("layer_x");
        let mut conv = ConvState::zeros(&cfg, b);
        let mut state = KdaState::zeros(&cfg, b);
        let y = kda_layer(&x, &w, &cfg, &mut conv, &mut state, b, t).unwrap();
        let e = max_abs(&y, &load("layer_y"));
        assert!(e < 1e-4, "layer out vs HF: {e}");
        // conv state: HF [b][chan][4] = [hist(3) | last row]
        let hc = load("layer_conv_after");
        let mut ec = 0f32;
        for bi in 0..b {
            for c in 0..cfg.conv_dim() {
                // HF caches the last KERNEL inputs (4-wide); ours the last kernel-1
                // (3-wide), which is HF's entries 1..3.
                for s in 0..3 {
                    let ours = conv.0[(bi * cfg.conv_dim() + c) * 3 + s];
                    ec = ec.max((ours - hc[(bi * cfg.conv_dim() + c) * 4 + s + 1]).abs());
                }
            }
        }
        assert!(ec < 1e-4, "conv state vs HF: {ec}");
        // recurrent state [b][h][k][v]
        let hs = load("layer_rec_after");
        let es = max_abs(&state.0, &hs);
        assert!(es < 1e-4, "recurrent state vs HF: {es}");

        // decode continuation
        let x2 = load("layer_x2");
        let y2 = kda_layer(&x2, &w, &cfg, &mut conv, &mut state, b, 1).unwrap();
        let e2 = max_abs(&y2, &load("layer_y2"));
        assert!(e2 < 1e-4, "decode out vs HF: {e2}");
        let hc2 = load("layer_conv_after2");
        let mut ec2 = 0f32;
        for bi in 0..b {
            for c in 0..cfg.conv_dim() {
                for s in 0..3 {
                    let ours = conv.0[(bi * cfg.conv_dim() + c) * 3 + s];
                    ec2 = ec2.max((ours - hc2[(bi * cfg.conv_dim() + c) * 4 + s + 1]).abs());
                }
            }
        }
        assert!(ec2 < 1e-4, "decode conv state vs HF: {ec2}");
        let es2 = max_abs(&state.0, &load("layer_rec_after2"));
        assert!(es2 < 1e-4, "decode recurrent state vs HF: {es2}");
    }

    #[test]
    fn chunk_agrees_with_serial_from_zero() {
        // no fixture: chunk vs decode from a zero state, exercising the multi-chunk path
        let cfg = cfg();
        let (b, t, h, d) = (1, 130, 4, 16);
        let ch = 3 * h * d;
        let mut qkv = vec![0f32; b * t * ch];
        let mut g = vec![0f32; b * t * h * d];
        let mut beta = vec![0f32; b * t * h];
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut rf = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // f64 in [0,1) from the top 53 bits, then to [-0.5, 0.5)
            (((rng >> 11) as f64) / (1u64 << 53) as f64) as f32 - 0.5
        };
        for i in 0..b * t {
            for j in 0..ch {
                qkv[i * ch + j] = rf();
            }
            for j in 0..h * d {
                // safe-gate range: lb·σ(rate·x), keep g in [−5, 0)
                g[i * h * d + j] = -5.0 * sigmoid(2.0 * rf());
            }
            for j in 0..h {
                beta[i * h + j] = sigmoid(rf());
            }
        }
        let mut st1 = KdaState::zeros(&cfg, b);
        let out_c = kda_chunk(&qkv, &g, &beta, &mut st1, &cfg, b, t, 64).unwrap();
        let mut st2 = KdaState::zeros(&cfg, b);
        let mut out_r = vec![0f32; b * t * h * d];
        for ti in 0..t {
            let o = kda_decode(
                &qkv[ti * ch..(ti + 1) * ch],
                &g[ti * h * d..(ti + 1) * h * d],
                &beta[ti * h..(ti + 1) * h],
                &mut st2,
                &cfg,
                b,
            )
            .unwrap();
            out_r[ti * h * d..(ti + 1) * h * d].copy_from_slice(&o);
        }
        let e = max_abs(&out_c, &out_r);
        let es = max_abs(&st1.0, &st2.0);
        assert!(e < 1e-4, "chunk vs serial out: {e}");
        assert!(es < 1e-4, "chunk vs serial state: {es}");
    }
}
