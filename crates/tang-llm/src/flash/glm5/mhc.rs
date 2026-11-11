//! GLM-5.3-Flash mHC: manifold-constrained hyper-connections (Xie et al., 2026), the
//! Sinkhorn-normalized 4-stream residual, plus the final unweighted stream collapse.
//!
//! Ground truth: `transformers/models/glm5_next/modeling_glm5_next.py`
//! (`Glm5NextTextHyperConnection`, `Glm5NextTextHyperHead`, lines 209–336), checked against
//! llama.cpp `src/models/glm5-next.cpp` (`build_hc_pre` / `build_hc_sinkhorn` /
//! `build_hc_post` / `build_hc_mean`). Both were read line by line; the layout facts below
//! were verified numerically against the HF module (see `tests` below).
//!
//! # The block math, exactly
//!
//! The residual state is `hc = 4` streams: `r[t][s][0..n]`, stream-major (the layout of the
//! `streams` fixtures and of tang's Qwen4Exp reference). Each of the two blocks per decoder
//! layer (attention, FFN) owns one set of weights, read from the GGUF as:
//!
//! - `blk.N.hc_attn_fn.weight` / `blk.N.hc_ffn_fn.weight`: `[24][hc*n]`, row-major
//!   (`mix_dim = (2 + hc)*hc = 24`)
//! - `blk.N.hc_attn_base.weight` / `blk.N.hc_ffn_base.weight`: `[24]`
//! - `blk.N.hc_attn_scale.weight` / `blk.N.hc_ffn_scale.weight`: `[3]`
//!
//! (llama.cpp creates them as `{hc*n_embd, hc_mix_dim}`, i.e. dims[0] = 24, so `dequantize`
//! yields row-major `[24][hc*n]` with `dims[0]` fastest — `fn` rows as above.)
//!
//! **pre** (per token, all f32):
//! 1. `f = r[t]` flattened to `[hc*n]`;
//! 2. `f = f / sqrt(mean(f²) + rms_eps)` — an *unweighted* RMSNorm over the whole
//!    `hc*n` vector (eps inside the sqrt, on the mean; rms_eps = 1e-5, the model's
//!    `rms_norm_eps`, NOT the hc_eps);
//! 3. `mixes = fn @ f` — 24 numbers;
//! 4. `pre_w = mixes[0..4]`, `post_w = mixes[4..8]`, `comb_w = mixes[8..24]`;
//! 5. `pre[s]  = sigmoid(pre_w[s]  * scale[0] + base[0..4][s])  + hc_eps`;
//! 6. `post[s] = 2 * sigmoid(post_w[s] * scale[1] + base[4..8][s])`;
//! 7. `comb[s][d] = softmax_d(comb_w[s*4 + d] * scale[2] + base[8 + s*4 + d]) + hc_eps`
//!    — **comb is src-major**: `comb[src][dst]`, softmax over the last axis (dst), verified
//!    numerically (a `[4,4]` arange probe reproduces `comb.transpose(-1,-2) @ res` exactly);
//! 8. Sinkhorn-Knopp on comb: one **column** normalization (sum over src, i.e. axis 0),
//!    then `iters - 1` times a **row** (over dst, axis 1) then a column normalization,
//!    every denominator with `+ hc_eps` (hc_eps = 1e-6). 20 iterations at the real model:
//!    1 column pass + 19 × (row, column). Rows sum to 1 up to eps;
//! 9. `x[t] = Σ_s pre[s] * r[t][s]` — the block input (one vector; HF computes the
//!    sigmoid+eps-weighted sum directly on `hidden_streams`, skipping the mean a plain
//!    (non-manifold) HC would take).
//!
//! The softmax runs over the raw 4 logits — HF adds no max-subtraction, and 4-term sums
//! cannot overflow; the fixtures confirm bitwise-scale agreement.
//!
//! **post** (combine, after the block produced `y[t][0..n]`):
//! `r_out[t][d] = post[d] * y[t] + Σ_s comb[s][d] * r_in[t][s]`
//! (HF: `post.unsqueeze(-1) * y.unsqueeze(-2) + matmul(comb.transpose(-1,-2), residual)`;
//! llama.cpp's `build_hc_post` iterates the same `[dst][src]` after ggml's reversed-dim
//! views, and its graph comment "comb is [dst_hc, src_hc, n_tokens]" describes ggml's
//! fastest-first storage of our `[src][dst]`).
//!
//! **head** (once, before `output_norm`): `h[t] = mean_s r[t][s]` — an *unweighted* mean,
//! no learned weights (unlike DeepSeek-V4). `hc_head` has no GGUF tensors.
//!
//! # Semantics the port must not break
//!
//! - Every mixing weight (steps 2–8) is computed in **f32** even when the streams are bf16;
//!   only `collapsed`, `post` and `comb` are cast back (HF `.to(hidden_streams.dtype)`,
//!   `.to(dtype)`). The CPU reference is f32 throughout anyway.
//! - `hc_eps` appears twice with different signs of subtlety: **added** to comb after the
//!   softmax and pre after the sigmoid, and **added to every Sinkhorn denominator**.
//! - The Sinkhorn order is col, then (row, col) × (iters−1) — starting with a row pass
//!   diverges (though only at the eps scale, ~1e-6).
//! - The MTP/nextn layer has **no** HC (plain residual, llama.cpp `graph_mtp`); the trunk
//!   initializes all 4 streams to the embedding (HF `expand(-1, -1, hc_mult, -1)`).
//!
//! # CUDA
//!
//! Two kernels, one warp each, 4×4 comb and the 16 mixes in registers (see [`SRC`]):
//!
//! - `mhc_pre`: flat unweighted RMSNorm over `hc*n`, mixes, pre/post/comb, Sinkhorn,
//!   collapse; writes `x`, `post`, `comb`.
//! - `mhc_post`: `r_out = post[d]*y + Σ_s comb[s][d] * r_in[s]`.
//!
//! The mixes GEMV (`24 × hc*n` per token) is a plain dot per lane with warp shuffles — at
//! the real model `hc*n = 16384`, 32 lanes × 512 f32 each. One block per token, 32 threads.
//!
//! # Layout contract for W1
//!
//! Streams are stored `[t][s][n]` (token-major, then stream, feature fastest), the same as
//! tang's Qwen4Exp reference residual. `post` is `[t][4]`, `comb` is `[t][src*4 + dst]`.

use crate::flash::glm5::{HC_EPS, HC_STREAMS, SINKHORN_ITERS};
use crate::gguf::Gguf;
use anyhow::{bail, Context, Result};

// ---------------------------------------------------------------------------------------------
// CPU f32 reference
// ---------------------------------------------------------------------------------------------

/// One hyper-connection's weights (attention or FFN), dequantized to f32.
#[derive(Clone, Debug)]
pub struct MhcWeights {
    /// `[24][hc*n]` row-major: `fn[r*hc*n + f]`.
    pub f: Vec<f32>,
    /// `[24]`.
    pub b: Vec<f32>,
    /// `[3]` (pre, post, comb scales).
    pub s: Vec<f32>,
    /// `n` (hidden size). `hc` is [`HC_STREAMS`] (4).
    pub n: usize,
    /// Sinkhorn iterations (20 for the real model and W0's tiny one).
    pub iters: u32,
    /// `hc_eps` (1e-6).
    pub eps: f32,
    /// The unweighted-RMSNorm eps (`rms_norm_eps`, 1e-5 at the real model — NOT `eps`).
    pub rms_eps: f32,
}

/// The mHC hparams W1 parses from GGUF metadata (keys `glm5-next.hyper_connection.count`,
/// `.sinkhorn_iterations`, `.epsilon`, `glm5-next.attention.layer_norm_rms_epsilon`).
#[derive(Clone, Copy, Debug)]
pub struct MhcHparams {
    pub streams: usize,
    pub sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub rms_eps: f32,
}

impl MhcHparams {
    /// The model's defaults: 4 streams, 20 Sinkhorn iterations, hc_eps 1e-6, RMSNorm 1e-5.
    pub const REAL: Self = Self {
        streams: HC_STREAMS,
        sinkhorn_iters: SINKHORN_ITERS,
        hc_eps: HC_EPS,
        rms_eps: 1e-5,
    };
}

impl MhcWeights {
    /// Load `blk.{layer}.{prefix}_fn/base/scale.weight` from the GGUF, dequantized to f32.
    /// `prefix` is `"hc_attn"` or `"hc_ffn"`. `hp.n` is `n_embd`.
    pub fn from_gguf(
        g: &Gguf,
        layer: usize,
        prefix: &str,
        n: usize,
        hp: &MhcHparams,
    ) -> Result<Self> {
        let name = |suffix: &str| format!("blk.{layer}.{prefix}_{suffix}.weight");
        Self::from_slices(
            g.tensor_f32(&name("fn")).with_context(|| name("fn"))?,
            g.tensor_f32(&name("base")).with_context(|| name("base"))?,
            g.tensor_f32(&name("scale"))
                .with_context(|| name("scale"))?,
            n,
            hp,
        )
        .with_context(|| format!("blk.{layer}.{prefix}"))
    }

    /// Same as [`from_gguf`](Self::from_gguf), from already-dequantized weight vectors
    /// (fixtures, tests).
    pub fn from_slices(
        f: Vec<f32>,
        b: Vec<f32>,
        s: Vec<f32>,
        n: usize,
        hp: &MhcHparams,
    ) -> Result<Self> {
        let hc = hp.streams;
        let mix = (2 + hc) * hc;
        if f.len() != mix * hc * n {
            bail!("fn: {} values, want {mix}×{}", f.len(), hc * n);
        }
        if b.len() != mix {
            bail!("base: {} values, want {mix}", b.len());
        }
        if s.len() != 3 {
            bail!("scale: {} values, want 3", s.len());
        }
        Ok(Self {
            f,
            b,
            s,
            n,
            iters: hp.sinkhorn_iters,
            eps: hp.hc_eps,
            rms_eps: hp.rms_eps,
        })
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softmax4(x: &[f32; 4]) -> [f32; 4] {
    // HF's softmax over 4 raw logits: no max subtraction (cannot overflow), sum first.
    let z: f32 = x.iter().map(|v| v.exp()).sum();
    let mut out = [0f32; 4];
    for (o, v) in out.iter_mut().zip(x) {
        *o = v.exp() / z;
    }
    out
}

/// The 4×4 Sinkhorn-Knopp of the comb matrix: one column pass (over src), then
/// `iters - 1` × (row, column) — `comb[src][dst]`, both denominators `+ eps`.
/// `iters >= 1`.
pub fn sinkhorn4(comb: &mut [[f32; 4]; 4], iters: u32, eps: f32) {
    let col = |comb: &mut [[f32; 4]; 4], eps: f32| {
        for d in 0..4 {
            let z = comb[0][d] + comb[1][d] + comb[2][d] + comb[3][d] + eps;
            for s in 0..4 {
                comb[s][d] /= z;
            }
        }
    };
    let row = |comb: &mut [[f32; 4]; 4], eps: f32| {
        for s in 0..4 {
            let z = comb[s][0] + comb[s][1] + comb[s][2] + comb[s][3] + eps;
            comb[s] = [
                comb[s][0] / z,
                comb[s][1] / z,
                comb[s][2] / z,
                comb[s][3] / z,
            ];
        }
    };
    col(comb, eps);
    for _ in 1..iters {
        row(comb, eps);
        col(comb, eps);
    }
}

/// The per-token mixing weights [`mhc_pre`] returns: the collapsed block input plus the
/// `post`/`comb` the block's output is re-expanded with (via [`mhc_post`]). The `pre`
/// weights are consumed inside [`mhc_pre`].
pub struct Mix {
    /// The block input `[n]`: `Σ_s pre[s] * r[s]`.
    pub x: Vec<f32>,
    /// `[4]` per-stream post weights (`2 * sigmoid(...)`).
    pub post: [f32; 4],
    /// `[src][dst]`, Sinkhorn-normalized.
    pub comb: [[f32; 4]; 4],
}

/// The mHC pre step of one token: unweighted flat RMSNorm → `fn` projection →
/// pre/post/comb (+Sinkhorn) → collapse the 4 streams to the block input.
///
/// `r` is the token's 4 streams, `[4][n]` (stream-major slices of the state).
/// All f32; call for every token. Use [`mhc_post`] after the block.
pub fn mhc_pre(w: &MhcWeights, r: &[&[f32]]) -> Result<Mix> {
    let (n, eps, rms_eps) = (w.n, w.eps, w.rms_eps);
    if r.len() != 4 || r.iter().any(|s| s.len() != n) {
        bail!(
            "streams: {} slices of {:?}",
            r.len(),
            r.first().map(|s| s.len())
        );
    }
    // 1-2: flatten + unweighted RMSNorm over the whole hc*n vector. Sum of squares in f64
    // (tang's inv_rms convention) for parity margin.
    let flat_len = 4 * n;
    let mut f = Vec::with_capacity(flat_len);
    for s in r {
        f.extend_from_slice(s);
    }
    let sumsq: f64 = f.iter().map(|&v| v as f64 * v as f64).sum();
    let inv = 1.0 / (((sumsq / flat_len as f64) as f32) + rms_eps).sqrt();
    for v in &mut f {
        *v *= inv;
    }
    // 3: mixes = fn @ f (24 rows).
    let mut mixes = [0f32; 24];
    for (i, m) in mixes.iter_mut().enumerate() {
        let row = &w.f[i * flat_len..(i + 1) * flat_len];
        let mut acc = 0f32;
        for (a, b) in row.iter().zip(&f) {
            acc += a * b;
        }
        *m = acc;
    }
    // 4-7: pre, post, comb.
    let mut pre = [0f32; 4];
    let mut post = [0f32; 4];
    for s in 0..4 {
        pre[s] = sigmoid(mixes[s] * w.s[0] + w.b[s]) + eps;
        post[s] = 2.0 * sigmoid(mixes[4 + s] * w.s[1] + w.b[4 + s]);
    }
    let mut comb = [[0f32; 4]; 4];
    for s in 0..4 {
        let mut logits = [0f32; 4];
        for d in 0..4 {
            logits[d] = mixes[8 + s * 4 + d] * w.s[2] + w.b[8 + s * 4 + d];
        }
        comb[s] = softmax4(&logits);
        for d in 0..4 {
            comb[s][d] += eps;
        }
    }
    // 8: Sinkhorn.
    sinkhorn4(&mut comb, w.iters, eps);
    // 9: collapse (the pre weights, the streams as given).
    let mut x = vec![0f32; n];
    for s in 0..4 {
        for i in 0..n {
            x[i] += pre[s] * r[s][i];
        }
    }
    Ok(Mix { x, post, comb })
}

/// The mHC post step of one token: expand the block output `y` (`[n]`) back into 4 streams.
///
/// `r_out[d] = post[d] * y + Σ_s comb[s][d] * r_in[s]` — `r_in` is the *pre-block* state
/// (the same slices passed to [`mhc_pre`]); `r_out` must not alias it.
pub fn mhc_post(mix: &Mix, y: &[f32], r_in: &[&[f32]], r_out: &mut [&mut [f32]]) -> Result<()> {
    let n = mix.x.len();
    if y.len() != n || r_in.len() != 4 || r_out.len() != 4 {
        bail!(
            "post shapes: y {}, r_in {}, r_out {}",
            y.len(),
            r_in.len(),
            r_out.len()
        );
    }
    for s in r_in.iter() {
        if s.len() != n {
            bail!("stream len {} != {n}", s.len());
        }
    }
    for s in r_out.iter() {
        if s.len() != n {
            bail!("stream len {} != {n}", s.len());
        }
    }
    for d in 0..4 {
        let dst = &mut r_out[d];
        let pd = mix.post[d];
        for i in 0..n {
            dst[i] = pd * y[i];
        }
        for (src, crow) in r_in.iter().zip(mix.comb.iter()) {
            for (o, v) in dst.iter_mut().zip(src.iter()) {
                *o += crow[d] * v;
            }
        }
    }
    Ok(())
}

/// The final collapse before `output_norm`: the *unweighted* mean of the 4 streams.
/// `h[t] = mean_s r[t][s]`. `r` is one token's `[4][n]`.
pub fn hc_head(r: &[&[f32]]) -> Result<Vec<f32>> {
    let n = r.first().context("empty")?.len();
    if r.len() != 4 || r.iter().any(|s| s.len() != n) {
        bail!("streams: {} slices", r.len());
    }
    let mut h = vec![0f32; n];
    for s in r {
        for (o, v) in h.iter_mut().zip(s.iter()) {
            *o += v;
        }
    }
    for v in &mut h {
        *v *= 0.25;
    }
    Ok(h)
}

// ---------------------------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------------------------

/// The CUDA C of the two mHC kernels, NVRTC-compiled (the `kernels.rs`/`coder_kernels.rs`
/// pattern). One block per token, 32 threads (one warp); the 4×4 comb and the 24 mixes live
/// in registers, staged through 24 floats of shared memory. The mixes GEMV is a lane-dot
/// with warp shuffles: at the real model `hc*n = 16384`, each lane walks 512 columns.
///
/// Layouts match the CPU functions and the fixtures exactly: streams `[t][s][n]`,
/// `post` `[t][4]`, `comb` `[t][src*4+dst]`.
///
/// H200 pod: `cargo test --features cuda --release mhc_cuda -- --nocapture`
/// (`TANG_REQUIRE_CUDA=1` to fail instead of skip when no GPU is present).
pub const SRC: &str = r#"
#define HC 4

__device__ __forceinline__ float sigm_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// lane-sum over the full warp, fixed order
__device__ __forceinline__ float wsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One token's mHC pre step: flat unweighted RMSNorm over HC*n, mixes = fn @ f, pre/post/comb
// (softmax over dst + Sinkhorn), collapse to x. Grid: one block per token. Block: 32.
//   r     [t][HC][n] in   (token-major streams, feature fastest)
//   fn    [24][HC*n]      (row-major)
//   base  [24], scale [3]
//   x     [t][n] out (block input), post [t][4] out, comb [t][16] out (src*4+dst)
// iters >= 1; hc_eps, rms_eps as at the model. n a multiple of 32.
extern "C" __global__ void mhc_pre(const float* r, const float* fn, const float* base,
                                   const float* scale, float* x, float* post, float* comb,
                                   int n, int iters, float hc_eps, float rms_eps) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const int hn = HC * n;
    const float* rt = r + (size_t)t * hn;
    __shared__ float sh[32];   // mixes at [0..24); [24..28) pre; [28] spare
    // 1-2: sum of squares of the whole flat vector (f64 on the CPU; f32 here — the parity
    // tolerance absorbs the difference at these magnitudes).
    float ss = 0.f;
    for (int i = lane; i < hn; i += 32) ss = fmaf(rt[i], rt[i], ss);
    ss = wsum(ss);
    const float inv = 1.0f / sqrtf(ss / (float)hn + rms_eps);
    // 3: mixes = fn @ f. Streamed: each lane accumulates its strided share of every row
    // (w[i] * rt[i] * inv), one shuffle reduction per row, so no column cache is needed at
    // any n (the real model's hn = 16384 means 512 strided columns per lane).
    for (int m = 0; m < 24; m++) {
        const float* w = fn + (size_t)m * hn;
        float acc = 0.f;
        for (int i = lane; i < hn; i += 32) acc = fmaf(w[i], rt[i] * inv, acc);
        acc = wsum(acc);
        if (lane == 0) sh[m] = acc;
    }
    // 4-8: pre, post, comb (softmax over dst), Sinkhorn — serial on lane 0. A 4×4, ~150
    // flops, negligible next to the mixes GEMV, and op-for-op the CPU reference (no
    // cross-lane shared-memory races; a parallel version would need a warp sync between
    // the row reads and the writes).
    if (lane == 0) {
        for (int s = 0; s < 4; s++) {
            sh[24 + s] = sigm_f(sh[s] * scale[0] + base[s]) + hc_eps;          // pre
            post[(size_t)t * 4 + s] = 2.0f * sigm_f(sh[4 + s] * scale[1] + base[4 + s]);
        }
        for (int s = 0; s < 4; s++) {
            float e[4], z = 0.f;
            for (int d = 0; d < 4; d++) {
                e[d] = expf(sh[8 + s * 4 + d] * scale[2] + base[8 + s * 4 + d]);
                z += e[d];
            }
            for (int d = 0; d < 4; d++) sh[8 + s * 4 + d] = e[d] / z + hc_eps; // comb
        }
        // Sinkhorn: one column pass (sum over src), then iters-1 x (row, column).
        for (int d = 0; d < 4; d++) {
            float z = sh[8 + d] + sh[12 + d] + sh[16 + d] + sh[20 + d] + hc_eps;
            for (int s = 0; s < 4; s++) sh[8 + s * 4 + d] /= z;
        }
        for (int it = 1; it < iters; it++) {
            for (int s = 0; s < 4; s++) {
                float z = sh[8 + s * 4] + sh[9 + s * 4] + sh[10 + s * 4] + sh[11 + s * 4] + hc_eps;
                for (int d = 0; d < 4; d++) sh[8 + s * 4 + d] /= z;
            }
            for (int d = 0; d < 4; d++) {
                float z = sh[8 + d] + sh[12 + d] + sh[16 + d] + sh[20 + d] + hc_eps;
                for (int s = 0; s < 4; s++) sh[8 + s * 4 + d] /= z;
            }
        }
        for (int s = 0; s < 4; s++) {
            for (int d = 0; d < 4; d++) comb[(size_t)t * 16 + s * 4 + d] = sh[8 + s * 4 + d];
        }
    }
    __syncthreads();
    // 9: collapse: x = sum_s pre[s] * r[s].
    float* xt = x + (size_t)t * n;
    for (int i = lane; i < n; i += 32) {
        float acc = 0.f;
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            acc = fmaf(sh[24 + s], rt[(size_t)s * n + i], acc);
        }
        xt[i] = acc;
    }
}

// One token's mHC post step: r_out[d] = post[d]*y + sum_s comb[s][d]*r_in[s].
// Grid: one block per token. Block: 32. r_in is the pre-block state (the same the pre
// kernel read); r_out must not alias it.
extern "C" __global__ void mhc_post(const float* y, const float* r_in, const float* post,
                                    const float* comb, float* r_out, int n) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const float* yt = y + (size_t)t * n;
    const float* rit = r_in + (size_t)t * HC * n;
    float* rot = r_out + (size_t)t * HC * n;
    const float* pt = post + (size_t)t * 4;
    const float* ct = comb + (size_t)t * 16;
    for (int i = lane; i < HC * n; i += 32) {
        const int d = i / n;              // destination stream
        const int f = i - d * n;          // feature
        float acc = pt[d] * yt[f];
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            acc = fmaf(ct[s * 4 + d], rit[(size_t)s * n + f], acc);
        }
        rot[i] = acc;
    }
}
"#;

/// The two mHC kernels for one `n`, NVRTC-compiled once and cached (the `Kern` pattern of
/// `coder.rs`, minimal). Built with [`MhcKernels::new`]; `n` and the stream are arguments of
/// the launches, so one instance serves a whole model.
#[cfg(feature = "cuda")]
pub struct MhcKernels {
    module: tang_moe::gpu::Module,
}

#[cfg(feature = "cuda")]
impl MhcKernels {
    /// Compile [`SRC`] for this device's architecture. `Gpu::bind` first.
    pub fn new(gpu: &tang_moe::gpu::Gpu) -> Result<Self> {
        Ok(Self {
            module: gpu
                .module(SRC)
                .map_err(|e| anyhow::anyhow!("mhc kernels: {e}"))?,
        })
    }

    /// [`mhc_pre`](super::super::glm5::mhc) for `t` tokens. One 32-thread block per token.
    ///
    /// Buffers: `r` in `[t][4][n]` (token-major streams), `fn_` `[24][4n]` row-major,
    /// `base` `[24]`, `scale` `[3]`; out `x` `[t][n]`, `post` `[t][4]`,
    /// `comb` `[t][16]` (`src*4+dst`). `n % 32 == 0`.
    ///
    /// # Safety
    /// CUDA context current; every buffer sized as above and alive for the launch.
    pub unsafe fn pre(
        &self,
        stream: &tang_moe::gpu::Stream,
        r: u64,
        fn_: u64,
        base: u64,
        scale: u64,
        x: u64,
        post: u64,
        comb: u64,
        t: usize,
        n: usize,
        iters: u32,
        hc_eps: f32,
        rms_eps: f32,
    ) -> Result<()> {
        use tang_moe::gpu as g;
        let f = self
            .module
            .func("mhc_pre")
            .map_err(|e| anyhow::anyhow!("mhc_pre: {e}"))?;
        let (nt, it) = (n as i32, iters as i32);
        g::launch(
            f,
            (t as u32, 1, 1),
            (32, 1, 1),
            0,
            stream,
            tang_moe::args![r, fn_, base, scale, x, post, comb, nt, it, hc_eps, rms_eps],
        )
        .map_err(|e| anyhow::anyhow!("mhc_pre launch: {e}"))
    }

    /// [`mhc_post`](super::super::glm5::mhc) for `t` tokens. One 32-thread block per token.
    ///
    /// Buffers: `y` in `[t][n]`, `r_in` `[t][4][n]` (the pre-block state the `pre` kernel
    /// read), `post` `[t][4]`, `comb` `[t][16]`; `r_out` `[t][4][n]`, which must not alias
    /// `r_in`.
    ///
    /// # Safety
    /// CUDA context current; every buffer sized as above and alive for the launch.
    pub unsafe fn post(
        &self,
        stream: &tang_moe::gpu::Stream,
        y: u64,
        r_in: u64,
        post: u64,
        comb: u64,
        r_out: u64,
        t: usize,
        n: usize,
    ) -> Result<()> {
        use tang_moe::gpu as g;
        let f = self
            .module
            .func("mhc_post")
            .map_err(|e| anyhow::anyhow!("mhc_post: {e}"))?;
        let nt = n as i32;
        g::launch(
            f,
            (t as u32, 1, 1),
            (32, 1, 1),
            0,
            stream,
            tang_moe::args![y, r_in, post, comb, r_out, nt],
        )
        .map_err(|e| anyhow::anyhow!("mhc_post launch: {e}"))
    }
}

/// mHC with its weights **resident on the device**: one `DevBuf` per weight set (upload
/// once), a reusable stream, and a persistent scratch pool sized to the largest window, so
/// a per-call `mhc_pre`/`mhc_post` is kernel time + one small `r`/`y` transfer instead of
/// 7 `cuMemAlloc` + 1.6 MB of pageable HtoD + read-backs (the pod-13 profile's shape:
/// 90 per-window `from_slice(w.f)` calls at 1.6 MB each — the DevMoe fix pattern).
///
/// One per (device, weights) pair; built once per layer at engine construction. The scratch
/// is shared per instance: `pre`/`post` on the same instance must not overlap (the engine
/// calls them serially on one layer's block, so they don't).
#[cfg(feature = "cuda")]
pub struct MhcGpu {
    k: MhcKernels,
    s: tang_moe::gpu::Stream,
    /// Weights for the two blocks per layer, device-resident: `[attn, ffn]`.
    w: [DevWeights; 2],
    n: usize,
    iters: u32,
    eps: f32,
    rms_eps: f32,
    /// Scratch, sized to `max_t` tokens: r `[max_t][4][n]` (the pre/post input state), rout
    /// `[max_t][4][n]` (post's output — must not alias r), y `[max_t][n]`, x `[max_t][n]`,
    /// post `[max_t][4]`, comb `[max_t][16]`.
    r: tang_moe::gpu::DevBuf,
    rout: tang_moe::gpu::DevBuf,
    y: tang_moe::gpu::DevBuf,
    x: tang_moe::gpu::DevBuf,
    post: tang_moe::gpu::DevBuf,
    comb: tang_moe::gpu::DevBuf,
    max_t: usize,
}

#[cfg(feature = "cuda")]
struct DevWeights {
    fn_: tang_moe::gpu::DevBuf,
    base: tang_moe::gpu::DevBuf,
    scale: tang_moe::gpu::DevBuf,
}

#[cfg(feature = "cuda")]
impl MhcGpu {
    /// Upload `w`'s two blocks' weights (attention first, then FFN), allocate the scratch
    /// for `max_t` tokens. `Gpu::bind` first. Weights stay resident for the engine's life:
    /// ~1.6 MB × 2 per layer × 45 layers ≈ 146 MB at the real model, nothing at decode
    /// time.
    pub fn new(gpu: &tang_moe::gpu::Gpu, ws: [&MhcWeights; 2], max_t: usize) -> Result<Self> {
        let up = |v: &[f32]| {
            tang_moe::gpu::DevBuf::from_slice(v).map_err(|e| anyhow::anyhow!("mhc upload: {e}"))
        };
        let dw = |w: &MhcWeights| -> Result<DevWeights> {
            Ok(DevWeights {
                fn_: up(&w.f)?,
                base: up(&w.b)?,
                scale: up(&w.s)?,
            })
        };
        let n = ws[0].n;
        if ws.iter().any(|w| w.n != n) {
            bail!("mhc: mixed hidden sizes in the two blocks");
        }
        let (iters, eps, rms_eps) = (ws[0].iters, ws[0].eps, ws[0].rms_eps);
        Ok(Self {
            k: MhcKernels::new(gpu)?,
            s: tang_moe::gpu::Stream::blocking().map_err(|e| anyhow::anyhow!("mhc stream: {e}"))?,
            w: [dw(ws[0])?, dw(ws[1])?],
            n,
            iters,
            eps,
            rms_eps,
            r: tang_moe::gpu::DevBuf::alloc(max_t * 4 * n * 4)
                .map_err(|e| anyhow::anyhow!("mhc scratch r: {e}"))?,
            rout: tang_moe::gpu::DevBuf::alloc(max_t * 4 * n * 4)
                .map_err(|e| anyhow::anyhow!("mhc scratch rout: {e}"))?,
            y: tang_moe::gpu::DevBuf::alloc(max_t * n * 4)
                .map_err(|e| anyhow::anyhow!("mhc scratch y: {e}"))?,
            x: tang_moe::gpu::DevBuf::alloc(max_t * n * 4)
                .map_err(|e| anyhow::anyhow!("mhc scratch x: {e}"))?,
            post: tang_moe::gpu::DevBuf::alloc(max_t * 4 * 4)
                .map_err(|e| anyhow::anyhow!("mhc scratch post: {e}"))?,
            comb: tang_moe::gpu::DevBuf::alloc(max_t * 16 * 4)
                .map_err(|e| anyhow::anyhow!("mhc scratch comb: {e}"))?,
            max_t,
        })
    }

    /// Block `block`'s resident weights as device addresses `(fn, base, scale)` plus the
    /// shared `(iters, hc_eps, rms_eps)` — [`super::mhc_fused::MhcFused`] launches its own
    /// kernels on them. `block`: 0 = hc_attn, 1 = hc_ffn.
    pub(crate) fn dev_weights(&self, block: usize) -> ((u64, u64, u64), (u32, f32, f32)) {
        let w = &self.w[block];
        (
            (w.fn_.ptr, w.base.ptr, w.scale.ptr),
            (self.iters, self.eps, self.rms_eps),
        )
    }

    /// The resident block weights' fn row (for tests).
    #[cfg(test)]
    fn wfn_len(&self, block: usize) -> usize {
        self.w[block].fn_.len
    }

    /// [`mhc_pre`](super::super::glm5::mhc) for `t` tokens: upload `r` (the only per-call
    /// HtoD, `t·4n·4` bytes ≤ 256 KB at a 4096-token window), launch, and read back
    /// `x`/`post`/`comb`. `block` is 0 (attention) or 1 (FFN).
    pub fn pre(&self, block: usize, r: &[f32], t: usize) -> Result<MixSet> {
        if block > 1 || r.len() != t * 4 * self.n || t > self.max_t {
            bail!("mhc pre: block {block}, {} values, t {t}", r.len());
        }
        let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("mhc pre: {e}");
        self.r.write(0, r).map_err(gerr)?;
        let w = &self.w[block];
        unsafe {
            self.k
                .pre(
                    &self.s,
                    self.r.ptr,
                    w.fn_.ptr,
                    w.base.ptr,
                    w.scale.ptr,
                    self.x.ptr,
                    self.post.ptr,
                    self.comb.ptr,
                    t,
                    self.n,
                    self.iters,
                    self.eps,
                    self.rms_eps,
                )
                .map_err(|e| anyhow::anyhow!("mhc pre launch: {e}"))?;
        }
        self.s.sync().map_err(gerr)?;
        let n = self.n;
        let x: Vec<f32> = self.x.read(0, t * n).map_err(gerr)?;
        let post: Vec<f32> = self.post.read(0, t * 4).map_err(gerr)?;
        let comb: Vec<f32> = self.comb.read(0, t * 16).map_err(gerr)?;
        Ok(MixSet {
            x,
            post,
            comb,
            t,
            n,
        })
    }

    /// [`mhc_post`](super::super::glm5::mhc) from the engine's `Vec<Mix>` form: repack
    /// the per-token post/comb flat and delegate to [`Self::post`].
    pub fn post_mixes(&self, mixes: &[Mix], y: &[f32], r_in: &[f32], t: usize) -> Result<Vec<f32>> {
        if mixes.len() != t {
            bail!("mhc post_mixes: {} mixes, t {t}", mixes.len());
        }
        let mut post = Vec::with_capacity(t * 4);
        let mut comb = Vec::with_capacity(t * 16);
        for m in mixes {
            post.extend_from_slice(&m.post);
            for row in &m.comb {
                comb.extend_from_slice(row);
            }
        }
        let mix = MixSet {
            x: Vec::new(),
            post,
            comb,
            t,
            n: self.n,
        };
        self.post(&mix, y, r_in, t)
    }

    /// [`mhc_post`](super::super::glm5::mhc) for `t` tokens from a [`MixSet`]: upload
    /// `y`/`post`/`comb`, launch, read the 4 streams back. The device `r` (pre-block
    /// state) was overwritten by the caller's data only in host memory here — this path
    /// re-uploads it as `r_in`, `t·4n·4` bytes.
    pub fn post(&self, mix: &MixSet, y: &[f32], r_in: &[f32], t: usize) -> Result<Vec<f32>> {
        if y.len() != t * self.n
            || r_in.len() != t * 4 * self.n
            || mix.t != t
            || mix.n != self.n
            || t > self.max_t
        {
            bail!("mhc post shapes: y {}, r {}, t {t}", y.len(), r_in.len());
        }
        let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("mhc post: {e}");
        self.y.write(0, y).map_err(gerr)?;
        self.r.write(0, r_in).map_err(gerr)?;
        self.post.write(0, &mix.post).map_err(gerr)?;
        self.comb.write(0, &mix.comb).map_err(gerr)?;
        unsafe {
            self.k
                .post(
                    &self.s,
                    self.y.ptr,
                    self.r.ptr,
                    self.post.ptr,
                    self.comb.ptr,
                    self.rout.ptr,
                    t,
                    self.n,
                )
                .map_err(|e| anyhow::anyhow!("mhc post launch: {e}"))?;
        }
        self.s.sync().map_err(gerr)?;
        self.rout.read(0, t * 4 * self.n).map_err(gerr)
    }
}

/// The device `pre`'s outputs, host-side: the block input, the per-token `post` weights
/// (`[t][4]`) and `comb` (`[t][16]`, `src*4+dst`) — `mhc_post`'s inputs, kept flat so the
/// post upload needs no repacking.
#[cfg(feature = "cuda")]
pub struct MixSet {
    /// `[t][n]`.
    pub x: Vec<f32>,
    /// `[t][4]`.
    pub post: Vec<f32>,
    /// `[t][16]` (`src*4+dst`).
    pub comb: Vec<f32>,
    t: usize,
    n: usize,
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5/mhc")
    }

    fn read(name: &str, want: &[usize]) -> Vec<f32> {
        let b = std::fs::read(fixture_dir().join(format!("{name}.bin")))
            .with_context(|| name.to_string())
            .unwrap();
        assert_eq!(b.len() % 4, 0, "{name}: not f32");
        assert_eq!(
            b.len() / 4,
            want.iter().product::<usize>(),
            "{name}: shape {want:?}"
        );
        b.chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// The cases in `index.json`: (`streams`, `y`, `b`, `s`, `post`, `comb`, `collapsed`,
    /// `out`, `flat_norm`) f32s per case.
    struct Case {
        b: usize,
        s: usize,
        streams: Vec<f32>,
        y: Vec<f32>,
        post: Vec<f32>,
        comb: Vec<f32>,
        collapsed: Vec<f32>,
        out: Vec<f32>,
    }

    fn cases() -> Vec<Case> {
        let idx: Value = serde_json::from_str(
            &std::fs::read_to_string(fixture_dir().join("index.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(idx["hc_mult"], 4, "fixture hc_mult");
        (0..3)
            .map(|i| {
                let p = format!("c{i}_");
                let shape = |n: &str| -> Vec<usize> {
                    idx["tensors"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|t| t["name"] == n)
                        .unwrap()["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect()
                };
                let sshape = shape(&format!("{p}streams"));
                Case {
                    b: sshape[0],
                    s: sshape[1],
                    streams: read(&format!("{p}streams"), &sshape),
                    y: read(&format!("{p}y"), &shape(&format!("{p}y"))),
                    post: read(&format!("{p}post"), &shape(&format!("{p}post"))),
                    comb: read(&format!("{p}comb"), &shape(&format!("{p}comb"))),
                    collapsed: read(&format!("{p}collapsed"), &shape(&format!("{p}collapsed"))),
                    out: read(&format!("{p}out"), &shape(&format!("{p}out"))),
                }
            })
            .collect()
    }

    fn weights() -> MhcWeights {
        MhcWeights::from_slices(
            read("fn", &[24, 256]),
            read("base", &[24]),
            read("scale", &[3]),
            64,
            &MhcHparams::REAL,
        )
        .unwrap()
    }

    /// Max |got − want| over everything recorded.
    fn parity() -> f32 {
        let w = weights();
        let mut max = 0f32;
        for c in cases() {
            let (b, s, n, hc) = (c.b, c.s, 64, 4);
            for bi in 0..b {
                for si in 0..s {
                    let tok = (bi * s + si) * hc * n;
                    let streams: Vec<&[f32]> = (0..4)
                        .map(|st| &c.streams[tok + st * n..tok + (st + 1) * n])
                        .collect();
                    let mix = mhc_pre(&w, &streams).unwrap();
                    let t = bi * s + si;
                    for d in 0..4 {
                        max = max.max((mix.post[d] - c.post[t * 4 + d]).abs());
                    }
                    for src in 0..4 {
                        for d in 0..4 {
                            max = max.max((mix.comb[src][d] - c.comb[t * 16 + src * 4 + d]).abs());
                        }
                    }
                    for i in 0..n {
                        max = max.max((mix.x[i] - c.collapsed[t * n + i]).abs());
                    }
                    // post
                    let yt = &c.y[t * n..(t + 1) * n];
                    let mut out = vec![vec![0f32; n]; 4];
                    let mut out = {
                        let [o0, o1, o2, o3] = out.as_mut_slice() else {
                            unreachable!()
                        };
                        [
                            o0.as_mut_slice(),
                            o1.as_mut_slice(),
                            o2.as_mut_slice(),
                            o3.as_mut_slice(),
                        ]
                    };
                    mhc_post(&mix, yt, &streams, &mut out).unwrap();
                    for d in 0..4 {
                        for i in 0..n {
                            max = max.max((out[d][i] - c.out[tok + d * n + i]).abs());
                        }
                    }
                }
            }
        }
        max
    }

    #[test]
    fn hf_parity() {
        // Missing fixtures (not yet generated) skip rather than fail.
        if !fixture_dir().join("index.json").exists() {
            eprintln!("mhc fixtures not generated: skipping");
            return;
        }
        let max = parity();
        assert!(max <= 1e-4, "max abs err {max} > 1e-4");
        eprintln!("mhc parity: max abs err {max:.3e}");
    }

    #[test]
    fn sinkhorn_doubly_stochastic() {
        let mut c = [
            [0.3f32, 1.2, 0.1, 0.7],
            [1.1, 0.4, 0.9, 0.2],
            [0.5, 0.5, 1.5, 0.6],
            [0.8, 0.3, 0.4, 1.3],
        ];
        sinkhorn4(&mut c, 20, 1e-6);
        for s in 0..4 {
            let z: f32 = c[s].iter().sum();
            assert!((z - 1.0).abs() < 1e-5, "row {s} sums to {z}");
        }
        for d in 0..4 {
            let z: f32 = (0..4).map(|s| c[s][d]).sum();
            assert!((z - 1.0).abs() < 1e-5, "col {d} sums to {z}");
        }
    }

    #[test]
    fn head_is_unweighted_mean() {
        let r: Vec<Vec<f32>> = (0..4)
            .map(|s| (0..8).map(|i| (s * 8 + i) as f32).collect())
            .collect();
        let rs: Vec<&[f32]> = r.iter().map(|v| v.as_slice()).collect();
        let h = hc_head(&rs).unwrap();
        for (i, v) in h.iter().enumerate() {
            let want = (0..4).map(|s| (s * 8 + i) as f32).sum::<f32>() / 4.0;
            assert!((v - want).abs() < 1e-6);
        }
    }

    /// GPU parity on the same fixtures: the kernels against the CPU reference fns.
    /// Skips without a GPU unless `TANG_REQUIRE_CUDA=1`.
    #[cfg(feature = "cuda")]
    #[test]
    fn mhc_cuda() {
        use tang_compute::{ComputeBuffer, ComputeDevice, CudaComputeDevice};
        use tang_moe::gpu::{Gpu, Stream};

        if !fixture_dir().join("index.json").exists() {
            eprintln!("mhc fixtures not generated: skipping");
            return;
        }
        let dev = match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(d)) => d,
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => {
                eprintln!("no CUDA device: skipping");
                return;
            }
        };
        let gpu = Gpu {
            ctx: dev.cuda_context().clone(),
        };
        gpu.bind().unwrap();
        let stream = Stream::blocking().unwrap(); // own blocking stream (stages host fixtures), dropped at the end
        let k = MhcKernels::new(&gpu).unwrap();
        let w = weights();
        let n = w.n;
        for c in cases() {
            let (b, s) = (c.b, c.s);
            let t = b * s;
            let r = dev.upload_f32(&c.streams);
            let y = dev.upload_f32(&c.y);
            let fnb = dev.upload_f32(&w.f);
            let base = dev.upload_f32(&w.b);
            let scale = dev.upload_f32(&w.s);
            let x = dev.upload_f32(&vec![0f32; t * n]);
            let post = dev.upload_f32(&vec![0f32; t * 4]);
            let comb = dev.upload_f32(&vec![0f32; t * 16]);
            let out = dev.upload_f32(&vec![0f32; t * 4 * n]);
            let a = |v: &tang_compute::cuda::CudaBuffer| dev.buffer_addr(v);
            unsafe {
                k.pre(
                    &stream,
                    a(&r),
                    a(&fnb),
                    a(&base),
                    a(&scale),
                    a(&x),
                    a(&post),
                    a(&comb),
                    t,
                    n,
                    w.iters,
                    w.eps,
                    w.rms_eps,
                )
                .unwrap();
                k.post(&stream, a(&y), a(&r), a(&post), a(&comb), a(&out), t, n)
                    .unwrap();
            }
            // the kernels ran on `stream` (non-blocking): sync it, not just the device stream
            stream.sync().unwrap();
            dev.sync();
            let gx = x.to_vec();
            let gpost = post.to_vec();
            let gcomb = comb.to_vec();
            let gout = out.to_vec();
            let mut max = 0f32;
            for i in 0..t * n {
                max = max.max((gx[i] - c.collapsed[i]).abs());
            }
            for i in 0..t * 4 {
                max = max.max((gpost[i] - c.post[i]).abs());
            }
            for i in 0..t * 16 {
                max = max.max((gcomb[i] - c.comb[i]).abs());
            }
            for i in 0..t * 4 * n {
                max = max.max((gout[i] - c.out[i]).abs());
            }
            assert!(max <= 1e-4, "cuda parity: max abs err {max} > 1e-4");
            eprintln!("mhc cuda parity (b{b} s{s}): max abs err {max:.3e}");
        }
    }

    /// Deterministic pseudo-random f32 in [-1, 1) from an integer state (xorshift-ish; no
    /// external rng dep). Different `seed`s give independent sequences.
    #[allow(dead_code)]
    fn rnd(state: &mut u64) -> f32 {
        // xorshift64*
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let m = (state.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32;
        m / (1u32 << 23) as f32 - 1.0
    }

    /// Real-model shapes (n = 4096, hn = 16384, the kernel streams 512 strided columns per
    /// lane; the tiny fixtures are n = 64): CPU reference vs the CUDA kernels on a synthetic
    /// case, batch 1 token and 8 tokens. Skips without a GPU unless `TANG_REQUIRE_CUDA=1`.
    #[cfg(feature = "cuda")]
    #[test]
    fn mhc_cuda_real_shape() {
        use tang_compute::{ComputeBuffer, ComputeDevice, CudaComputeDevice};
        use tang_moe::gpu::{Gpu, Stream};

        let (t, n) = (8, 4096);
        let mut st = 0x9E3779B97F4A7C15u64;
        let w = MhcWeights::from_slices(
            (0..24 * 4 * n).map(|_| rnd(&mut st)).collect(),
            (0..24).map(|_| rnd(&mut st) * 0.5).collect(),
            [0.5f32, 0.5, 0.5].to_vec(),
            n,
            &MhcHparams::REAL,
        )
        .unwrap();
        // streams [t][4][n], y [t][n]
        let streams: Vec<f32> = (0..t * 4 * n).map(|_| rnd(&mut st) * 0.1).collect();
        let y: Vec<f32> = (0..t * n).map(|_| rnd(&mut st) * 0.1).collect();

        // CPU reference over the same buffers.
        let mut cpu_x = Vec::with_capacity(t * n);
        let mut cpu_out = Vec::with_capacity(t * 4 * n);
        for tok in 0..t {
            let rs: Vec<&[f32]> = (0..4)
                .map(|s| {
                    let o = tok * 4 * n + s * n;
                    &streams[o..o + n]
                })
                .collect();
            let mix = mhc_pre(&w, &rs).unwrap();
            cpu_x.extend_from_slice(&mix.x);
            let yt = &y[tok * n..(tok + 1) * n];
            let mut outs = vec![vec![0f32; n]; 4];
            let mut outs = {
                let [o0, o1, o2, o3] = outs.as_mut_slice() else {
                    unreachable!()
                };
                [
                    o0.as_mut_slice(),
                    o1.as_mut_slice(),
                    o2.as_mut_slice(),
                    o3.as_mut_slice(),
                ]
            };
            mhc_post(&mix, yt, &rs, &mut outs).unwrap();
            for o in outs {
                cpu_out.extend_from_slice(o);
            }
        }

        let dev = match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(d)) => d,
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => {
                eprintln!("no CUDA device: skipping");
                return;
            }
        };
        let gpu = Gpu {
            ctx: dev.cuda_context().clone(),
        };
        gpu.bind().unwrap();
        let stream = Stream::blocking().unwrap();
        let k = MhcKernels::new(&gpu).unwrap();
        let r = dev.upload_f32(&streams);
        let yb = dev.upload_f32(&y);
        let fnb = dev.upload_f32(&w.f);
        let base = dev.upload_f32(&w.b);
        let scale = dev.upload_f32(&w.s);
        let x = dev.upload_f32(&vec![0f32; t * n]);
        let post = dev.upload_f32(&vec![0f32; t * 4]);
        let comb = dev.upload_f32(&vec![0f32; t * 16]);
        let out = dev.upload_f32(&vec![0f32; t * 4 * n]);
        let a = |v: &tang_compute::cuda::CudaBuffer| dev.buffer_addr(v);
        unsafe {
            k.pre(
                &stream,
                a(&r),
                a(&fnb),
                a(&base),
                a(&scale),
                a(&x),
                a(&post),
                a(&comb),
                t,
                n,
                w.iters,
                w.eps,
                w.rms_eps,
            )
            .unwrap();
            k.post(&stream, a(&yb), a(&r), a(&post), a(&comb), a(&out), t, n)
                .unwrap();
        }
        stream.sync().unwrap();
        let mut max = 0f32;
        let gx = x.to_vec();
        for (g, c) in gx.iter().zip(&cpu_x) {
            max = max.max((*g - *c).abs());
        }
        let gout = out.to_vec();
        for (g, c) in gout.iter().zip(&cpu_out) {
            max = max.max((*g - *c).abs());
        }
        // The mixes GEMV is 16k-long f32 dots: CPU f64-sumsq vs GPU f32 differ a little more
        // than the tiny case; 1e-4 stays the gate.
        assert!(
            max <= 1e-4,
            "cuda real-shape parity: max abs err {max} > 1e-4"
        );
        eprintln!("mhc cuda real-shape (n={n}, t={t}): max abs err {max:.3e}");

        // Micro-bench: events timing of pre and post at batch 1 and 8, 50 reps each, the
        // median rep reported (first launches compile nothing extra but warm the L2).
        for (label, tb) in [("b1", 1usize), ("b8", 8usize)] {
            let reps = 50;
            let mut pre_us = Vec::with_capacity(reps);
            let mut post_us = Vec::with_capacity(reps);
            let e0 = tang_moe::gpu::Event::new(true).unwrap();
            let e1 = tang_moe::gpu::Event::new(true).unwrap();
            for _ in 0..reps {
                e0.record(&stream).unwrap();
                unsafe {
                    k.pre(
                        &stream,
                        a(&r),
                        a(&fnb),
                        a(&base),
                        a(&scale),
                        a(&x),
                        a(&post),
                        a(&comb),
                        tb,
                        n,
                        w.iters,
                        w.eps,
                        w.rms_eps,
                    )
                    .unwrap();
                }
                e1.record(&stream).unwrap();
                e1.sync().unwrap();
                pre_us.push(e1.since(&e0).unwrap_or(0.0).max(0.0) * 1e3);
                e0.record(&stream).unwrap();
                unsafe {
                    k.post(&stream, a(&yb), a(&r), a(&post), a(&comb), a(&out), tb, n)
                        .unwrap();
                }
                e1.record(&stream).unwrap();
                e1.sync().unwrap();
                post_us.push(e1.since(&e0).unwrap_or(0.0).max(0.0) * 1e3);
            }
            // medians
            pre_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            post_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "mhc bench n={n} {label}: pre median {:.1} µs, post median {:.1} µs",
                pre_us[reps / 2],
                post_us[reps / 2]
            );
        }
    }

    /// Regression test for the stream-ordering race in eval/tree/stream-blocking.md
    /// (frog/tree-eval 1b69a838): a synchronous `DevBuf` upload (legacy default stream)
    /// must be ordered before a kernel launched on a blocking stream. With a
    /// non-blocking data-path stream this flaked: the kernel read its input before the
    /// HtoD landed. Stages fresh host data, launches `mhc_pre`, reads back, and repeats —
    /// every rep must match the CPU reference.
    #[cfg(feature = "cuda")]
    #[test]
    fn mhc_cuda_stream_ordering() {
        use tang_compute::{ComputeBuffer, ComputeDevice, CudaComputeDevice};
        use tang_moe::gpu::{DevBuf, Gpu, Stream};

        let (t, n) = (1, 4096);
        let mut st = 0xDEADBEEFCAFEBABEu64;
        let w = MhcWeights::from_slices(
            (0..24 * 4 * n).map(|_| rnd(&mut st)).collect(),
            (0..24).map(|_| rnd(&mut st) * 0.5).collect(),
            [0.5f32, 0.5, 0.5].to_vec(),
            n,
            &MhcHparams::REAL,
        )
        .unwrap();
        let streams: Vec<f32> = (0..t * 4 * n).map(|_| rnd(&mut st) * 0.1).collect();

        let rs: Vec<&[f32]> = (0..4)
            .map(|s| {
                let o = s * n;
                &streams[o..o + n]
            })
            .collect();
        let cpu = mhc_pre(&w, &rs).unwrap();

        let dev = match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(d)) => d,
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => {
                eprintln!("no CUDA device: skipping");
                return;
            }
        };
        let gpu = Gpu {
            ctx: dev.cuda_context().clone(),
        };
        gpu.bind().unwrap();
        // The data-path stream the engine path uses: blocking, so the legacy-stream HtoD
        // of every from_slice below is ordered before the kernel.
        let stream = Stream::blocking().unwrap();
        let k = MhcKernels::new(&gpu).unwrap();
        let mut last_x = 0f32;
        for rep in 0..10 {
            // Fresh staging each rep, exactly the shape that raced before.
            let rb = DevBuf::from_slice(&streams).unwrap();
            let fnb = DevBuf::from_slice(&w.f).unwrap();
            let base = DevBuf::from_slice(&w.b).unwrap();
            let scale = DevBuf::from_slice(&w.s).unwrap();
            let xb = DevBuf::zeroed(t * n * 4).unwrap();
            let postb = DevBuf::zeroed(t * 4 * 4).unwrap();
            let combb = DevBuf::zeroed(t * 16 * 4).unwrap();
            unsafe {
                k.pre(
                    &stream, rb.ptr, fnb.ptr, base.ptr, scale.ptr, xb.ptr, postb.ptr, combb.ptr, t,
                    n, w.iters, w.eps, w.rms_eps,
                )
                .unwrap();
            }
            stream.sync().unwrap();
            let gx: Vec<f32> = xb.read(0, t * n).unwrap();
            for i in 0..n {
                let err = (gx[i] - cpu.x[i]).abs();
                assert!(
                    err <= 1e-4,
                    "rep {rep}: stream-ordering race, x[{i}] err {err}"
                );
                last_x = last_x.max(err);
            }
        }
        eprintln!("mhc stream-ordering: 10 reps, max abs err {last_x:.3e}");
    }

    /// The staging-vs-resident micro-bench (the pod-13 sprint task): per-call cost of the
    /// engine's current shape — a new blocking stream, 7 `DevBuf::from_slice`/`zeroed`
    /// allocs (incl. the 1.6 MB f32 weights), launch, sync, read-backs — against
    /// [`MhcGpu`], which keeps the weights resident and reuses a scratch pool (per-call:
    /// one small HtoD + kernel + small read-backs). Both paths checked against the CPU
    /// reference first; then 30 timed reps each (events, median). Real-model shape
    /// (n = 4096, t = 1 decode, t = 64 prefill window).
    #[cfg(feature = "cuda")]
    #[test]
    fn mhc_cuda_resident_bench() {
        use tang_compute::{ComputeBuffer, ComputeDevice, CudaComputeDevice};
        use tang_moe::gpu::{DevBuf, Event, Gpu, Stream};

        let dev = match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(d)) => d,
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => {
                eprintln!("no CUDA device: skipping");
                return;
            }
        };
        let gpu = Gpu {
            ctx: dev.cuda_context().clone(),
        };
        gpu.bind().unwrap();
        let n = 4096usize;
        let mut st = 0xABCDEF0123456789u64;
        let w = MhcWeights::from_slices(
            (0..24 * 4 * n).map(|_| rnd(&mut st)).collect(),
            (0..24).map(|_| rnd(&mut st) * 0.5).collect(),
            [0.5f32, 0.5, 0.5].to_vec(),
            n,
            &MhcHparams::REAL,
        )
        .unwrap();
        let resident = MhcGpu::new(&gpu, [&w, &w], 64).unwrap();
        assert_eq!(resident.wfn_len(0), 24 * 4 * n * 4);

        for t in [1usize, 64] {
            let streams: Vec<f32> = (0..t * 4 * n).map(|_| rnd(&mut st) * 0.1).collect();
            let y: Vec<f32> = (0..t * n).map(|_| rnd(&mut st) * 0.1).collect();
            // CPU reference of token 0 only (the parity target for the bench's checks).
            let rs: Vec<&[f32]> = (0..4)
                .map(|s| {
                    let o = s * n;
                    &streams[o..o + n]
                })
                .collect();
            let cpu = mhc_pre(&w, &rs).unwrap();

            // --- the staging path (the engine's current mhc_pre_gpu shape) ---
            let stage_pre = || -> anyhow::Result<()> {
                let s = Stream::blocking()?;
                let rb = DevBuf::from_slice(&streams)?;
                let fb = DevBuf::from_slice(&w.f)?;
                let bb = DevBuf::from_slice(&w.b)?;
                let sb = DevBuf::from_slice(&w.s)?;
                let xb = DevBuf::zeroed(t * n * 4)?;
                let pb = DevBuf::zeroed(t * 4 * 4)?;
                let cb = DevBuf::zeroed(t * 16 * 4)?;
                unsafe {
                    resident.k.pre(
                        &s, rb.ptr, fb.ptr, bb.ptr, sb.ptr, xb.ptr, pb.ptr, cb.ptr, t, n, w.iters,
                        w.eps, w.rms_eps,
                    )?;
                }
                s.sync()?;
                let _ = xb.read::<f32>(0, t * n)?;
                let _ = pb.read::<f32>(0, t * 4)?;
                let _ = cb.read::<f32>(0, t * 16)?;
                Ok(())
            };

            // --- the resident path ---
            let res_pre = || -> anyhow::Result<MixSet> { resident.pre(0, &streams, t) };

            // correctness first (resident vs CPU, token 0 only)
            let got = resident.pre(0, &streams, t).unwrap();
            for i in 0..n {
                assert!((got.x[i] - cpu.x[i]).abs() <= 1e-4, "resident pre x[{i}]");
            }
            // resident post vs CPU (per-token CPU post, the token-major streams; token 0)
            let mut outs = vec![vec![0f32; n]; 4];
            {
                let rsi: Vec<&[f32]> = (0..4)
                    .map(|s| {
                        let o = s * n;
                        &streams[o..o + n]
                    })
                    .collect();
                let mut o = {
                    let [o0, o1, o2, o3] = outs.as_mut_slice() else {
                        unreachable!()
                    };
                    [
                        o0.as_mut_slice(),
                        o1.as_mut_slice(),
                        o2.as_mut_slice(),
                        o3.as_mut_slice(),
                    ]
                };
                mhc_post(&cpu, &y[0..n], &rsi, &mut o).unwrap();
            }
            let outv = resident.post(&got, &y, &streams, t).unwrap();
            for d in 0..4 {
                for i in 0..n {
                    let (a, b) = (outv[d * n + i], outs[d][i]);
                    assert!(
                        (a - b).abs() <= 1e-4,
                        "resident post [{d}][{i}]: {a} vs {b}"
                    );
                }
            }

            // --- timing: 30 reps, median, events around the whole per-call body ---
            const REPS: usize = 30;
            let bench = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> Vec<f32> {
                let e0 = Event::new(true).unwrap();
                let e1 = Event::new(true).unwrap();
                let mut ms = Vec::with_capacity(REPS);
                for _ in 0..REPS {
                    e0.record(&resident.s).unwrap();
                    f().unwrap();
                    e1.record(&resident.s).unwrap();
                    e1.sync().unwrap();
                    ms.push(e1.since(&e0).unwrap());
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ms
            };
            // stage path uses its own stream internally; wall-clock (Instant) instead
            let wall = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> Vec<f32> {
                let mut ms = Vec::with_capacity(REPS);
                for _ in 0..REPS {
                    let t0 = std::time::Instant::now();
                    f().unwrap();
                    ms.push(t0.elapsed().as_secs_f32() * 1e3);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ms
            };
            let mut s1 = || stage_pre();
            let stg = wall(&mut s1);
            let mut r1 = || res_pre().map(|_: MixSet| ());
            let res = bench(&mut r1);
            eprintln!(
                "mhc resident-bench n={n} t={t}: staging median {:.0} µs, resident median {:.0} µs",
                stg[REPS / 2] * 1e3,
                res[REPS / 2] * 1e3
            );
        }
    }
}
