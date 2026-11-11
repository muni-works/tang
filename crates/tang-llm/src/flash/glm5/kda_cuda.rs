//! The CUDA kernels of the GLM-5.3-Flash KDA layers ([`super::kda`]), naive-but-correct
//! first (GATES.md): the GDN kernels' shape ([`super::super::coder_kernels`] pattern) with
//! KDA's differences — a **per-channel** gate (a vector over each head's key dims, not a
//! scalar per head), a β gate per head, and no fused output gate (KDA gates with a gated
//! RMSNorm over the whole head output, done by the CPU reference for now).
//!
//! Two kernels, mirroring the CPU reference exactly:
//! - `kda_conv`: depthwise causal conv + SiLU over the stacked qkv of a prefill window,
//!   reading `d_conv − 1` inputs of history per channel (naive: one thread per channel,
//!   loops the taps; commit of the new history is a separate kernel, `kda_conv_commit`).
//! - `kda_step`: one decode token per launch (conv fused into the step), grid (heads, b).
//! - `kda_prefill`: the chunked prefill, grid (heads, b), block `head_dim`; each block loops
//!   its chunks serially.
//!
//! All state layouts match the CPU reference 1:1 so the parity test is a byte compare:
//! recurrent state `[slot][head][key][value]` f32, conv history `[slot][chan][d_conv-1]`
//! oldest tap first.
//!
//! # Pod commands (batched by the manager into a 1×H200 session)
//! ```text
//! # build + run the KDA parity tests (cuda feature, no default features on Linux)
//! CARGO_TARGET_DIR=/tmp/glm5-kda-target cargo test -p tang-llm --lib \
//!     --no-default-features --features cuda glm5_kda_ -- --nocapture
//! ```
#![allow(clippy::too_many_arguments)]
use anyhow::{ensure, Result};

/// The `#define` prelude for the KDA kernels.
pub fn prelude(cfg: &super::kda::KdaConfig) -> String {
    format!(
        "#define KDA_H {}\n#define KDA_D {}\n#define KDA_KC {}\n#define KDA_CONV {}\n",
        cfg.heads,
        cfg.head_dim,
        cfg.d_conv,
        cfg.conv_dim(),
    )
}

/// `kda_conv`: depthwise causal conv + SiLU over `[t][KDA_CONV]` (one layer's qkv rows).
/// `X` is the pre-conv input (post-projections), `H0` the `d_conv − 1` rows of history per
/// channel (oldest first), `W` `[chan][d_conv]` tap 0 = newest. `Y` is the conv output
/// (pre-SiLU in `H`, post-SiLU in `Y`). One thread per channel; grid `ceil(KDA_CONV/256)`.
pub const CONV_SRC: &str = r#"
typedef unsigned long long u64;

extern "C" __global__ void kda_conv(const float* __restrict__ X, unsigned int stride,
                                    const float* __restrict__ H0, const float* __restrict__ W,
                                    float* __restrict__ Y, unsigned int T) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bi = blockIdx.y;
    if (ch >= KDA_CONV) return;
    X += (u64)bi * T * stride; Y += (u64)bi * T * KDA_CONV; H0 += (u64)bi * KDA_CONV * (KDA_KC - 1);
    float w[KDA_KC];
    for (unsigned int i = 0; i < KDA_KC; i++) w[i] = W[ch * KDA_KC + i];
    for (unsigned int t = 0; t < T; t++) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            long long p = (long long)t - (KDA_KC - 1) + i;
            float e = p < 0 ? H0[ch * (KDA_KC - 1) + (p + (KDA_KC - 1))]
                            : X[(u64)(p) * stride + ch];
            acc = __fmaf_rn(w[i], e, acc);
        }
        Y[t * KDA_CONV + ch] = acc / (1.0f + expf(-acc));
    }
}

// commit: H0 <- the last KDA_KC-1 inputs of [hist | X]. Grid ceil(KDA_CONV/256), block 256.
extern "C" __global__ void kda_conv_commit(float* __restrict__ H0, const float* __restrict__ X,
                                           unsigned int stride, unsigned int T) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bi = blockIdx.y;
    if (ch >= KDA_CONV) return;
    X += (u64)bi * T * stride; H0 += (u64)bi * KDA_CONV * (KDA_KC - 1);
    float old[KDA_KC - 1];
    for (unsigned int s = 0; s < KDA_KC - 1; s++) old[s] = H0[ch * (KDA_KC - 1) + s];
    for (unsigned int s = 0; s < KDA_KC - 1; s++) {
        unsigned long long p = T + s;
        H0[ch * (KDA_KC - 1) + s] = p < KDA_KC - 1 ? old[p] : X[(p - (KDA_KC - 1)) * stride + ch];
    }
}

// batched decode commit: same shift, but the row `bi` belongs to SLOT[bi] — the state
// offset is `SLOT[bi] * KDA_CONV * (KDA_KC-1)`, not `bi * ...` (T = 1, one row per slot).
extern "C" __global__ void kda_conv_commit_rows(float* __restrict__ H0, const float* __restrict__ X,
                                                const unsigned int* __restrict__ SLOT) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bi = blockIdx.y;
    if (ch >= KDA_CONV) return;
    X += (u64)bi * KDA_CONV;
    H0 += (u64)SLOT[bi] * KDA_CONV * (KDA_KC - 1);
    float old[KDA_KC - 1];
    for (unsigned int s = 0; s < KDA_KC - 1; s++) old[s] = H0[ch * (KDA_KC - 1) + s];
    for (unsigned int s = 0; s < KDA_KC - 1; s++) {
        unsigned long long p = 1 + s;
        H0[ch * (KDA_KC - 1) + s] = p < KDA_KC - 1 ? old[p] : X[ch];
    }
}
"#;

/// `kda_step`: the fused conv + delta-rule decode step for one token (all heads in one
/// launch). `QKV` is the raw (pre-conv) stacked projection `[KDA_CONV]`, `H0`/`W` the conv
/// history/weights, `G` the pre-exp per-channel gate `[KDA_H][KDA_D]`, `B` the post-sigmoid
/// beta `[KDA_H]`, `S` the recurrent state `[KDA_H][KDA_D][KDA_D]`, `O` the pre-norm output.
/// Grid `KDA_H`, block `KDA_D` (128) — naive: each thread owns one value column; the key
/// dim loops in registers.
pub const STEP_SRC: &str = r#"
typedef unsigned long long u64;

// One thread per value column x of one head: q/k in shared (normed), the state in GLOBAL
// (naive: read and rewritten per token; registers can't hold a row the column reduction
// needs). Decay per key row a uses g[a] = exp(G[h*D + a]) — the gate channel is the KEY dim.
extern "C" __global__ void kda_step(const float* __restrict__ QKV,
                                    const float* __restrict__ H0, const float* __restrict__ W,
                                    const float* __restrict__ G, const float* __restrict__ B,
                                    float* __restrict__ S, float* __restrict__ O) {
    unsigned int h = blockIdx.x, x = threadIdx.x, D = KDA_D;
    unsigned int bi = blockIdx.y;
    QKV += (u64)bi * KDA_CONV;
    H0 += (u64)bi * KDA_CONV * (KDA_KC - 1);
    S += (u64)bi * KDA_H * D * D;
    O += (u64)bi * KDA_H * D;
    G += (u64)bi * KDA_H * D;
    B += (u64)bi * KDA_H;
    // ---- fused conv (this head's q/k/v channels) + l2 norm + scale, in shared ----
    __shared__ float q[KDA_D], k[KDA_D], sk[KDA_D];
    float qv = 0.0f, kv = 0.0f, vv = 0.0f;
    {
        unsigned int qc = h * D + x;
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[qc * (KDA_KC - 1) + i] : QKV[qc];
            acc = __fmaf_rn(W[qc * KDA_KC + i], e, acc);
        }
        qv = acc / (1.0f + expf(-acc));
        unsigned int kch = KDA_H * D + h * D + x;
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[kch * (KDA_KC - 1) + i] : QKV[kch];
            acc = __fmaf_rn(W[kch * KDA_KC + i], e, acc);
        }
        kv = acc / (1.0f + expf(-acc));
        unsigned int vch = 2 * KDA_H * D + h * D + x;
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[vch * (KDA_KC - 1) + i] : QKV[vch];
            acc = __fmaf_rn(W[vch * KDA_KC + i], e, acc);
        }
        vv = acc / (1.0f + expf(-acc));
    }
    q[x] = qv; k[x] = kv;
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (unsigned int a = 0; a < D; a++) { qs += q[a] * q[a]; ks += k[a] * k[a]; }
    qs = rsqrtf(qs + 1e-6f); ks = rsqrtf(ks + 1e-6f);
    float scale = rsqrtf((float)D);
    q[x] *= qs * scale; k[x] *= ks;      // now q[]/k[] hold the normed values for ALL threads
    __syncthreads();
    // ---- delta rule: one thread owns value column x ----
    // pass 1: sk[x] = Σ_a k[a]·S[a][x] with S[a][x] *= exp(G[a]) first
    float p = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((u64)h * D + a) * D + x;
        float s = (*sp) * expf(G[h * D + a]);
        *sp = s;
        p = __fmaf_rn(k[a], s, p);
    }
    sk[x] = p;
    __syncthreads();
    float delta = B[h] * (vv - sk[x]);
    // pass 2: S[a][x] += k[a]·δ[x]; o[x] = Σ_a q[a]·S[a][x]
    float o = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((u64)h * D + a) * D + x;
        float s = __fmaf_rn(k[a], delta, *sp);
        *sp = s;
        o = __fmaf_rn(q[a], s, o);
    }
    O[h * D + x] = o;
}
"#;

/// `kda_prefill`: the chunked prefill (chunk 64), one block per head, the block loops the
/// chunks serially. Block = KDA_D threads, thread `a` = one **key dim** (the gate channel).
/// Scratch is `[nblk][6][C][D]` f32 (GC cumsum gates, QN/KN normed q/k, VP/KP, VN v-new),
/// `nblk` = gridDim.x, indexed `((blk*6 + s)*C + i)*D + a` — host-allocated, zeroed once.
/// `A` (C×C) and the reduction staging live in shared memory.
///
/// Matches `kda_chunk` (HF's `chunk_kimi_delta_attention`) step for step, in f32 (the CPU
/// reference's f64 accumulation is not reproduced; parity tolerance 1e-4).
pub const PREFILL_SRC: &str = r#"
typedef unsigned long long u64;

// One float per thread → the block sum (D is a power of two; naive tree over shared red).
// Each phase reads only values written before the previous __syncthreads().
__device__ __forceinline__ float block_red(float* red, unsigned int a, float p, unsigned int D) {
    red[a] = p;
    __syncthreads();
    for (unsigned int s2 = D >> 1; s2 > 0; s2 >>= 1) {
        float other = 0.0f;
        if (a < s2) other = red[a + s2];
        __syncthreads();
        if (a < s2) red[a] += other;
        __syncthreads();
    }
    float v = red[0];
    __syncthreads();
    return v;
}

extern "C" __global__ void __launch_bounds__(1024) kda_prefill(
        const float* __restrict__ QKV, unsigned int T, unsigned int stride, // [T][KDA_CONV]
        const float* __restrict__ G, unsigned int gstride,    // [T][KDA_H*KDA_D] pre-exp
        const float* __restrict__ BETA, unsigned int bstride, // [T][KDA_H]
        float* __restrict__ S,                                // [H][D][D]
        float* __restrict__ O,                                // [T][H*D]
        float* __restrict__ SCR, unsigned int C) {            // scratch [blk][6][C][D]
    unsigned int h = blockIdx.x, a = threadIdx.x, D = KDA_D;
    unsigned int bi = blockIdx.y;
    unsigned int nc = (T + C - 1) / C;
    QKV += (u64)bi * T * stride;
    G += (u64)bi * T * gstride;
    BETA += (u64)bi * T * bstride;
    O += (u64)bi * T * (KDA_H * D);
    S += (u64)bi * KDA_H * D * D;
    SCR += (u64)bi * gridDim.x * 6 * C * D;
    extern __shared__ float sh[];                // A [C][C] then red [D]
    float* A = sh;
    float* red = sh + C * C;
    float* gc = SCR + ((u64)blockIdx.x * 6 + 0) * C * D;
    float* qn = SCR + ((u64)blockIdx.x * 6 + 1) * C * D;
    float* kn = SCR + ((u64)blockIdx.x * 6 + 2) * C * D;
    float* vp = SCR + ((u64)blockIdx.x * 6 + 3) * C * D;
    float* kp = SCR + ((u64)blockIdx.x * 6 + 4) * C * D;
    float* vn = SCR + ((u64)blockIdx.x * 6 + 5) * C * D;
    for (unsigned int ci = 0; ci < nc; ci++) {
        unsigned int o0 = ci * C;
        unsigned int n = min(C, T - o0);
        // normed q/k (+scale on q) and within-chunk gate cumsum
        for (unsigned int i = 0; i < n; i++) {
            unsigned int qb = h * D + a, kb = KDA_H * D + h * D + a;
            float qv = QKV[(o0 + i) * stride + qb];
            float kv = QKV[(o0 + i) * stride + kb];
            // l2 norm over the head's dims: reduction over a
            float qs = block_red(red, a, qv * qv, D);
            float ks = block_red(red, a, kv * kv, D);
            qn[i * D + a] = qv * rsqrtf(qs + 1e-6f) * rsqrtf((float)D);
            kn[i * D + a] = kv * rsqrtf(ks + 1e-6f);
            float gv = G[(o0 + i) * gstride + h * D + a];
            gc[i * D + a] = i > 0 ? gv + gc[(i - 1) * D + a] : gv;
        }
        __syncthreads();
        // A0: strictly lower, beta on row i only. A is uninitialized shared memory — zero
        // it first (the CPU's `a` starts as zeros; the fixpoint reads row m's diagonal,
        // which must be 0 until the final +I).
        for (unsigned int i = 0; i < n; i++)
            for (unsigned int j = a; j < C; j += D) A[i * C + j] = 0.0f;
        __syncthreads();
        for (unsigned int i = 1; i < n; i++) {
            for (unsigned int j = 0; j < i; j++) {
                float p = kn[i * D + a] * kn[j * D + a]
                    * expf(gc[i * D + a] - gc[j * D + a]);
                float sum = block_red(red, a, p, D);
                if (a == 0) {
                    A[i * C + j] = -sum * BETA[(o0 + i) * bstride + h];
                }
            }
        }
        __syncthreads();
        // HF's triangular fixpoint (serial in one thread; naive): row i finalises from a
        // SNAPSHOT of A0's row i taken BEFORE this i-iteration, against the already-final
        // rows m < i — entries m < j are still A0 here (the CPU ref snapshots `row` too).
        if (a == 0) {
            for (unsigned int i = 1; i < n; i++) {
                float row[64];
                for (unsigned int m = 0; m < i; m++) row[m] = A[i * C + m];
                for (unsigned int j = 0; j < i; j++) {
                    float s = 0.0f;
                    for (unsigned int m = 0; m < i; m++) s += row[m] * A[m * C + j];
                    A[i * C + j] = row[j] + s;
                }
            }
            for (unsigned int i = 0; i < n; i++) A[i * C + i] += 1.0f;
        }
        __syncthreads();
        // VP = A·(v⊙β), KP = A·(k⊙β·exp(gc))
        float beta_i = a < n ? BETA[(o0 + a) * bstride + h] : 0.0f;
        for (unsigned int i = 0; i < n; i++) {
            float p = 0.0f;
            for (unsigned int j = 0; j <= i; j++) {
                float av = A[i * C + j];
                p = __fmaf_rn(av, QKV[(o0 + j) * stride + 2 * KDA_H * D + h * D + a]
                    * BETA[(o0 + j) * bstride + h], p);
            }
            vp[i * D + a] = p;
            p = 0.0f;
            for (unsigned int j = 0; j <= i; j++) {
                float av = A[i * C + j];
                p = __fmaf_rn(av, kn[j * D + a] * BETA[(o0 + j) * bstride + h]
                    * expf(gc[j * D + a]), p);
            }
            kp[i * D + a] = p;
        }
        __syncthreads();
        // VN = VP − KP·S. KP is over the KEY dims, S maps key→value: (K'·S)[x] =
        // Σ_a KP[a]·S[a][x]. Thread a owns the VALUE dim x = a here (the state update and
        // the output's VN reads below index it by value).
        for (unsigned int i = 0; i < n; i++) {
            float p = 0.0f;
            for (unsigned int a2 = 0; a2 < D; a2++)
                p = __fmaf_rn(kp[i * D + a2], S[((u64)h * D + a2) * D + a], p);
            vn[i * D + a] = vp[i * D + a] - p;
        }
        __syncthreads();
        // output: inter + intra (thread a owns value column a → wait, value dim ≠ key dim;
        // for the output we need Σ over key rows a of q·S[a][x] — thread a holds key a, so
        // stage per-(i) reductions over a into red, one x at a time)
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int x = 0; x < D; x++) {
                float p = qn[i * D + a] * expf(gc[i * D + a]) * S[((u64)h * D + a) * D + x];
                float o = block_red(red, a, p, D);
                for (unsigned int c2 = 0; c2 <= i; c2++) {
                    p = qn[i * D + a] * kn[c2 * D + a]
                        * expf(gc[i * D + a] - gc[c2 * D + a]);
                    float sum = block_red(red, a, p, D);
                    o += sum * vn[c2 * D + x];
                }
                if (a == 0) O[(o0 + i) * (KDA_H * D) + h * D + x] = o;
            }
        }
        // state update: S[a][x] = S[a][x]·exp(gc_last[a]) + Σ_c k_c[a]·exp(gl[a]-gc_c[a])·VN_c[x]
        float gl = gc[(n - 1) * D + a];
        for (unsigned int x = 0; x < D; x++) {
            float* sp = S + ((u64)h * D + a) * D + x;
            float s = *sp * expf(gl);
            for (unsigned int c2 = 0; c2 < n; c2++)
                s += kn[c2 * D + a] * expf(gl - gc[c2 * D + a]) * vn[c2 * D + x];
            *sp = s;
        }
        __syncthreads();
    }
}
"#;

/// Chunk length of the parallel prefill ([`PF2_SRC`]), HF's / `kda_chunk`'s 64.
pub const PF_C: usize = 64;
/// Threads per block of the parallel prefill kernels.
pub const PF_NT: usize = 256;
/// Widest value-column slice one `kda_pf_state` block owns (its shared S slice).
pub const PF_VS: usize = 32;

/// Floats of one `(chunk, head)` workspace slot of the parallel prefill: KP, VP, QG, KG
/// `[C][D]`, AQK `[C][C]`, GL `[D]` (see [`PF2_SRC`]).
pub fn pf_slot(d: usize) -> usize {
    4 * PF_C * d + PF_C * PF_C + d
}

/// The parallel prefill (`TANG_GLM5_KDA_PF2=1`): the same math as `kda_prefill` /
/// `kda_chunk`, split the way FLA's chunked kernels are — everything of a chunk that does
/// NOT depend on the recurrent state runs fully parallel over (chunk, head), and only the
/// state-carrying matmuls stay serial over chunks:
///
/// - `kda_conv_ip`: conv + SiLU **in place** on the stacked qkv (one thread per channel
///   keeps its last `d_conv − 1` inputs in registers), committing the new history to `H0`
///   in the same pass — no separate conv output buffer, no commit kernel. The fma order is
///   `kda_conv`'s exactly.
/// - `kda_pf_intra`, grid `(chunks, H)`, block [`PF_NT`]: l2-norms (a warp per row), the
///   per-key-dim gate cumsum, the gram pair `A0[i][j] = −β_i Σ_a k_i k_j e^{G_i−G_j}` and
///   `Aqk[i][j] = Σ_a q_i k_j e^{G_i−G_j}` (one warp per row, lanes over the key dim, the
///   shared exp computed once for both), the triangular fixpoint (row by row, a warp per
///   output, lanes over the inner index), then `KP = A·(k⊙β⊙e^G)`, `VP = A·(v⊙β)`,
///   `QG = q⊙e^G`, `KG = k⊙e^{G_last−G}` into the workspace slot.
/// - `kda_pf_state`, grid `(H, D / VS)`, block [`PF_NT`]: the block owns value columns
///   `[x0, x0+VS)` of one head's state, staged in shared memory across the chunks, and per
///   chunk does `VN = VP − KP·S`, `O = QG·S + Aqk·VN`, `S ← e^{G_last}⊙S + KGᵀ·VN`.
///
/// Any `T` (no multiple-of-64 requirement): a chunk's rows past `T` are simply not
/// computed — the CPU's zero padding makes them inert (k = v = β = 0, G carried), so
/// `G_last = G[n−1]` and every pad term vanishes.
///
/// Workspace slot layout (floats): `KP[C][D] VP[C][D] QG[C][D] KG[C][D] AQK[C][C] GL[D]`,
/// slot index `chunk_in_group · H + h`. During `kda_pf_intra` the planes double as
/// temporaries (QG ← q̂, KG ← k̂, KP ← G, VP ← k̂⊙β⊙e^G) — each reuse is fenced by a
/// `__syncthreads` or is the same thread rewriting its own element.
pub const PF2_SRC: &str = r#"
typedef unsigned long long u64;
#define PF_C 64
#define PF_NT 256
#define PF_VS 32
#define PF_SLOT (4 * PF_C * KDA_D + PF_C * PF_C + KDA_D)

__device__ __forceinline__ float pf_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// conv + SiLU in place over X [T][stride] (the KDA_CONV channels at each row's start), H0
// [chan][KC-1] oldest first, committed at the end. One thread per channel.
extern "C" __global__ void kda_conv_ip(float* X, unsigned int stride, float* H0,
                                       const float* __restrict__ W, unsigned int T) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= KDA_CONV) return;
    float w[KDA_KC], hs[KDA_KC - 1];
    for (unsigned int i = 0; i < KDA_KC; i++) w[i] = W[ch * KDA_KC + i];
    for (unsigned int s = 0; s < KDA_KC - 1; s++) hs[s] = H0[ch * (KDA_KC - 1) + s];
    for (unsigned int t = 0; t < T; t++) {
        float xv = X[(u64)t * stride + ch];
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC - 1; i++) acc = __fmaf_rn(w[i], hs[i], acc);
        acc = __fmaf_rn(w[KDA_KC - 1], xv, acc);
        for (unsigned int s = 0; s + 1 < KDA_KC - 1; s++) hs[s] = hs[s + 1];
        hs[KDA_KC - 2] = xv;
        X[(u64)t * stride + ch] = acc / (1.0f + expf(-acc));
    }
    for (unsigned int s = 0; s < KDA_KC - 1; s++) H0[ch * (KDA_KC - 1) + s] = hs[s];
}

// Y: conv output [T][ystride] (q | k | v at 0, H*D, 2*H*D). Chunk c0 + blockIdx.x, head
// blockIdx.y; slot blockIdx.x * gridDim.y + h.
extern "C" __global__ void __launch_bounds__(PF_NT) kda_pf_intra(
        const float* __restrict__ Y, unsigned int ystride,
        const float* __restrict__ G, unsigned int gstride,
        const float* __restrict__ BETA, unsigned int bstride,
        float* WS, unsigned int c0, unsigned int T) {
    const unsigned int D = KDA_D, C = PF_C, NW = PF_NT / 32;
    unsigned int h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned int o0 = (c0 + blockIdx.x) * C;
    if (o0 >= T) return;
    unsigned int n = min(C, T - o0);
    float* ws = WS + (u64)(blockIdx.x * gridDim.y + h) * PF_SLOT;
    float* KP = ws;
    float* VP = ws + C * D;
    float* QG = ws + 2 * C * D;
    float* KG = ws + 3 * C * D;
    float* AQK = ws + 4 * C * D;
    float* GL = AQK + C * C;
    __shared__ float A[PF_C * PF_C];
    __shared__ float row[PF_C];
    __shared__ float sb[PF_C];
    for (unsigned int e = tid; e < C * C; e += PF_NT) A[e] = 0.0f;
    if (tid < n) sb[tid] = BETA[(u64)(o0 + tid) * bstride + h];
    // 1. l2 norms (+ 1/sqrt(D) on q), a warp per row: QG <- q^, KG <- k^
    for (unsigned int i = warp; i < n; i += NW) {
        const float* yr = Y + (u64)(o0 + i) * ystride;
        float qs = 0.0f, ks = 0.0f;
        for (unsigned int a = lane; a < D; a += 32) {
            float q = yr[h * D + a], k = yr[KDA_H * D + h * D + a];
            qs += q * q;
            ks += k * k;
        }
        qs = pf_warp_sum(qs);
        ks = pf_warp_sum(ks);
        for (unsigned int a = lane; a < D; a += 32) {
            QG[i * D + a] = yr[h * D + a] * rsqrtf(qs + 1e-6f) * rsqrtf((float)D);
            KG[i * D + a] = yr[KDA_H * D + h * D + a] * rsqrtf(ks + 1e-6f);
        }
    }
    // 2. within-chunk gate cumsum, a thread per key dim: KP <- G
    if (tid < D) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < n; i++) {
            acc += G[(u64)(o0 + i) * gstride + h * D + tid];
            KP[i * D + tid] = acc;
        }
    }
    __syncthreads();
    // 3. the gram pair, a warp per row i, lanes over the key dim
    for (unsigned int i = warp; i < n; i += NW) {
        for (unsigned int j = 0; j <= i; j++) {
            float kk = 0.0f, qk = 0.0f;
            for (unsigned int a = lane; a < D; a += 32) {
                float e = expf(KP[i * D + a] - KP[j * D + a]);
                float kj = KG[j * D + a];
                kk += KG[i * D + a] * kj * e;
                qk += QG[i * D + a] * kj * e;
            }
            kk = pf_warp_sum(kk);
            qk = pf_warp_sum(qk);
            if (lane == 0) {
                if (j < i) A[i * C + j] = -kk * sb[i];
                AQK[i * C + j] = qk;
            }
        }
    }
    __syncthreads();
    // 4. per key dim (own column only): QG = q^ e^G, VP <- k^ b e^G, KG = k^ e^(Gl - G)
    if (tid < D) {
        float gl = KP[(n - 1) * D + tid];
        for (unsigned int i = 0; i < n; i++) {
            float g = KP[i * D + tid], qn = QG[i * D + tid], kn = KG[i * D + tid];
            QG[i * D + tid] = qn * expf(g);
            VP[i * D + tid] = kn * sb[i] * expf(g);
            KG[i * D + tid] = kn * expf(gl - g);
        }
        GL[tid] = gl;
    }
    // 5. HF's triangular fixpoint: row i from a snapshot of A0's row i against the final
    //    rows m < i (A[m][j] = 0 for m <= j, so the m-sum starts at j+1)
    for (unsigned int i = 1; i < n; i++) {
        __syncthreads();
        if (tid < i) row[tid] = A[i * C + tid];
        __syncthreads();
        for (unsigned int j = warp; j < i; j += NW) {
            float s = 0.0f;
            for (unsigned int m = j + 1 + lane; m < i; m += 32) s += row[m] * A[m * C + j];
            s = pf_warp_sum(s);
            if (lane == 0) A[i * C + j] = row[j] + s;
        }
    }
    __syncthreads();
    if (tid < n) A[tid * C + tid] += 1.0f;
    __syncthreads();
    // 6. KP = A (k^ b e^G)   (KP's G temporaries were consumed in 4.)
    for (unsigned int e = tid; e < n * D; e += PF_NT) {
        unsigned int i = e / D, a = e % D;
        float s = 0.0f;
        for (unsigned int j = 0; j <= i; j++) s = __fmaf_rn(A[i * C + j], VP[j * D + a], s);
        KP[i * D + a] = s;
    }
    __syncthreads();
    // 7. VP = A (v b)
    for (unsigned int e = tid; e < n * D; e += PF_NT) {
        unsigned int i = e / D, x = e % D;
        float s = 0.0f;
        for (unsigned int j = 0; j <= i; j++)
            s = __fmaf_rn(A[i * C + j], Y[(u64)(o0 + j) * ystride + 2 * KDA_H * D + h * D + x] * sb[j], s);
        VP[i * D + x] = s;
    }
}

// The state-carrying half: head blockIdx.x, value columns [x0, x0 + VS), chunks
// c0 .. c0 + nchunks (slot cc * KDA_H + h). S [H][D][D], O [T][H*D].
extern "C" __global__ void __launch_bounds__(PF_NT) kda_pf_state(
        const float* WS, float* __restrict__ S, float* __restrict__ O,
        unsigned int c0, unsigned int nchunks, unsigned int T, unsigned int VS) {
    const unsigned int D = KDA_D, C = PF_C;
    unsigned int h = blockIdx.x, x0 = blockIdx.y * VS, tid = threadIdx.x;
    __shared__ float Ss[KDA_D * PF_VS];
    __shared__ float VN[PF_C * PF_VS];
    float* Sg = S + (u64)h * D * D;
    for (unsigned int e = tid; e < D * VS; e += PF_NT) Ss[e] = Sg[(e / VS) * D + x0 + e % VS];
    __syncthreads();
    for (unsigned int cc = 0; cc < nchunks; cc++) {
        unsigned int o0 = (c0 + cc) * C;
        if (o0 >= T) break;
        unsigned int n = min(C, T - o0);
        const float* ws = WS + (u64)(cc * KDA_H + h) * PF_SLOT;
        const float* KP = ws;
        const float* VP = ws + C * D;
        const float* QG = ws + 2 * C * D;
        const float* KG = ws + 3 * C * D;
        const float* AQK = ws + 4 * C * D;
        const float* GL = AQK + C * C;
        // VN = VP - KP S ; O = QG S   (the OLD state)
        for (unsigned int e = tid; e < n * VS; e += PF_NT) {
            unsigned int i = e / VS, xl = e % VS;
            float p = 0.0f, q = 0.0f;
            for (unsigned int a = 0; a < D; a++) {
                float s = Ss[a * VS + xl];
                p = __fmaf_rn(KP[i * D + a], s, p);
                q = __fmaf_rn(QG[i * D + a], s, q);
            }
            VN[i * VS + xl] = VP[i * D + x0 + xl] - p;
            O[(u64)(o0 + i) * (KDA_H * D) + h * D + x0 + xl] = q;
        }
        __syncthreads();
        // O += Aqk VN (same thread, same element as above)
        for (unsigned int e = tid; e < n * VS; e += PF_NT) {
            unsigned int i = e / VS, xl = e % VS;
            u64 oi = (u64)(o0 + i) * (KDA_H * D) + h * D + x0 + xl;
            float o = O[oi];
            for (unsigned int c = 0; c <= i; c++) o = __fmaf_rn(AQK[i * C + c], VN[c * VS + xl], o);
            O[oi] = o;
        }
        // S <- e^Gl S + KG^T VN
        for (unsigned int e = tid; e < D * VS; e += PF_NT) {
            unsigned int a = e / VS, xl = e % VS;
            float s = Ss[e] * expf(GL[a]);
            for (unsigned int c = 0; c < n; c++) s = __fmaf_rn(KG[c * D + a], VN[c * VS + xl], s);
            Ss[e] = s;
        }
        __syncthreads();
    }
    for (unsigned int e = tid; e < D * VS; e += PF_NT) Sg[(e / VS) * D + x0 + e % VS] = Ss[e];
}
"#;

// ---------------------------------------------------------------------------
// cuda-gated launcher + parity tests

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::super::kda::{ConvState, KdaConfig, KdaState};
    use anyhow::{ensure, Result};
    use cudarc::driver::sys;
    use tang_moe::gpu::{DevBuf, Gpu, Stream};
    use tang_moe::{args, gpu::launch};

    /// All KDA kernels, NVRTC-compiled once.
    pub struct Kernels {
        pub conv: sys::CUfunction,
        pub conv_commit: sys::CUfunction,
        pub step: sys::CUfunction,
        pub prefill: sys::CUfunction,
        /// the parallel prefill ([`super::PF2_SRC`])
        pub conv_ip: sys::CUfunction,
        pub pf_intra: sys::CUfunction,
        pub pf_state: sys::CUfunction,
        /// the batched decode commit (`kda_conv_commit_rows`: the rows are slots)
        pub conv_commit_rows: sys::CUfunction,
        pub _mod: tang_moe::gpu::Module,
    }

    pub fn load(g: &Gpu, cfg: &KdaConfig) -> Result<Kernels> {
        let src = format!(
            "{}\n{}\n{}\n{}\n{}",
            super::prelude(cfg),
            super::CONV_SRC,
            super::STEP_SRC,
            super::PREFILL_SRC,
            super::PF2_SRC
        );
        g.bind()?;
        let m = g.module(&src)?;
        Ok(Kernels {
            conv: m.func("kda_conv")?,
            conv_commit: m.func("kda_conv_commit")?,
            step: m.func("kda_step")?,
            prefill: m.func("kda_prefill")?,
            conv_ip: m.func("kda_conv_ip")?,
            pf_intra: m.func("kda_pf_intra")?,
            pf_state: m.func("kda_pf_state")?,
            conv_commit_rows: m.func("kda_conv_commit_rows")?,
            _mod: m,
        })
    }

    fn sincos(g: &Gpu) -> Result<Stream> {
        let s = Stream::blocking()?;
        g.bind()?;
        Ok(s)
    }

    /// One KDA layer's resident device side: the conv history, recurrent state and conv
    /// weights stay on the GPU for the layer's whole life; the per-call uploads are the
    /// token's qkv row(s) and gates only, and the only read-back is the pre-norm output
    /// (the gated norm + o_proj run on the host). No per-call allocations, one persistent
    /// stream — the 117 ms/layer decode profile was 2× 4 MiB state uploads + 3 sync
    /// read-backs + a fresh stream per call, not the kernel.
    pub struct DevKda {
        pub wconv: DevBuf,
        pub h0: DevBuf,
        pub st: DevBuf,
        /// u32 slot ids of the current batched decode window ([`Self::decode_dev_rows`])
        slots_dev: DevBuf,
        /// preallocated: chunk-64 x/g/beta/o/scratch for prefill, reused for decode
        pub xb: DevBuf,
        pub gb: DevBuf,
        pub bb: DevBuf,
        pub ob: DevBuf,
        pub yb: DevBuf,
        pub scr: DevBuf,
        /// The parallel prefill's workspace: `pf_groups` chunks × heads slots of
        /// [`super::pf_slot`] floats (1 byte until [`Self::enable_pf2`]).
        pub ws: DevBuf,
        pf_groups: usize,
        pub s: Stream,
        ch_dim: usize,
        nh: usize,
        hd: usize,
        chunk: usize,
        /// `d_conv − 1` (the conv history's per-channel depth; slot pointer math).
        d_conv_m1: usize,
        /// the state buffers' slot count (the rows cap of [`Self::decode_dev_rows`]).
        slots: usize,
    }

    impl DevKda {
        /// Allocate everything once. `conv` / `state` are the CPU-side values to seed with
        /// (from the tiny-model / GGUF load or a CPU fallback window); they stay the source
        /// of truth on the host ONLY for read-back via [`Self::read_conv`] /
        /// [`Self::read_state`]. The state layout is `[slots][...]`; slot 0 is seeded with
        /// the caller's values (the single-sequence contract), the rest zeroed.
        pub fn new(
            g: &Gpu,
            k: &Kernels,
            cfg: &KdaConfig,
            wconv: &[f32],
            conv: &ConvState,
            state: &KdaState,
            slots: usize,
        ) -> Result<Self> {
            let ch_dim = cfg.conv_dim();
            let (nh, hd) = (cfg.heads, cfg.head_dim);
            let chunk = 64usize;
            g.bind()?;
            // fail fast if the weights are the wrong shape
            anyhow::ensure!(wconv.len() == ch_dim * cfg.d_conv, "conv weight shape");
            // BLOCKING: decode/prefill stage x/g/beta (and seed stages h0/st) with the
            // synchronous DevBuf::write, which lands on the legacy default stream — a
            // NON_BLOCKING stream would let the core kernels read them before the copy
            // completes (gpu.rs Stream::blocking; it bit the GLM-5.3 staging run too)
            let s = Stream::blocking()?;
            g.bind()?;
            // the state buffers hold `slots` sequences; slot 0 is seeded, the rest zeroed
            let mut h0 = conv.0.clone();
            let mut st = state.0.clone();
            if slots > 1 {
                h0.resize(slots * h0.len(), 0.0);
                st.resize(slots * st.len(), 0.0);
            }
            Ok(DevKda {
                wconv: DevBuf::from_slice(wconv)?,
                h0: DevBuf::from_slice(&h0)?,
                st: DevBuf::from_slice(&st)?,
                slots_dev: DevBuf::zeroed(slots * 4)?,
                xb: DevBuf::zeroed(slots * chunk * ch_dim * 4)?,
                gb: DevBuf::zeroed(slots * chunk * nh * hd * 4)?,
                bb: DevBuf::zeroed(slots * chunk * nh * 4)?,
                ob: DevBuf::zeroed(slots * chunk * nh * hd * 4)?,
                yb: DevBuf::zeroed(slots * chunk * ch_dim * 4)?,
                scr: DevBuf::zeroed(slots * nh * 6 * chunk * hd * 4)?,
                ws: DevBuf::zeroed(1)?,
                pf_groups: 0,
                s,
                ch_dim,
                nh,
                hd,
                chunk,
                d_conv_m1: cfg.d_conv.saturating_sub(1),
                slots,
            })
        }

        /// The one-token decode step: upload the qkv row + gates, launch, read `o` back.
        /// `conv`/`state` host copies are updated on the host too (cheap, keeps them in
        /// sync for CPU fallback), but nothing is read back from the device.
        pub fn decode(
            &mut self,
            k: &Kernels,
            cfg: &KdaConfig,
            x: &[f32],
            gpre: &[f32],
            beta: &[f32],
            conv: &mut ConvState,
            state: &mut KdaState,
            b: usize,
        ) -> Result<Vec<f32>> {
            let (nh, hd, ch_dim) = (self.nh, self.hd, self.ch_dim);
            anyhow::ensure!(x.len() == b * ch_dim, "decode x shape");
            self.s.sync()?;
            self.xb.write(0, x)?;
            self.gb.write(0, gpre)?;
            self.bb.write(0, beta)?;
            unsafe {
                launch(
                    k.step,
                    (nh as u32, b as u32, 1),
                    (hd as u32, 1, 1),
                    (3 * hd * 4) as u32,
                    &self.s,
                    args![
                        self.xb.ptr,
                        self.h0.ptr,
                        self.wconv.ptr,
                        self.gb.ptr,
                        self.bb.ptr,
                        self.st.ptr,
                        self.ob.ptr
                    ],
                )?;
            }
            self.s.sync()?;
            // conv history shift on the host (the device copy is not read back; the host's
            // is authoritative for CPU fallback — note the device's H0 is now stale until
            // the next write, so the shift must ALSO run on the device; done via
            // conv_commit with T=1 below to keep the two in sync).
            let m = cfg.d_conv - 1;
            let mut next = conv.0.clone();
            for bi in 0..b {
                for c in 0..ch_dim {
                    for s2 in 0..m {
                        next[(bi * ch_dim + c) * m + s2] = if s2 + 1 < m {
                            conv.0[(bi * ch_dim + c) * m + s2 + 1]
                        } else {
                            x[bi * ch_dim + c]
                        };
                    }
                }
            }
            conv.0 = next;
            // device H0: same shift via the commit kernel (T=1)
            unsafe {
                launch(
                    k.conv_commit,
                    (ch_dim.div_ceil(256) as u32, b as u32, 1),
                    (256, 1, 1),
                    0,
                    &self.s,
                    args![self.h0.ptr, self.xb.ptr, ch_dim as u32, 1u32],
                )?;
            }
            self.s.sync()?;
            Ok(self.ob.read::<f32>(0, b * nh * hd)?)
        }

        /// Prefill `t ≤ chunk` tokens (chunk ≤ 64): conv + chunked core, device-resident.
        /// `t` may be < the buffer's chunk; only `t` rows are uploaded/read.
        #[allow(clippy::too_many_arguments)]
        pub fn prefill(
            &mut self,
            k: &Kernels,
            cfg: &KdaConfig,
            x: &[f32],
            gpre: &[f32],
            beta: &[f32],
            conv: &mut ConvState,
            state: &mut KdaState,
            b: usize,
            t: usize,
        ) -> Result<Vec<f32>> {
            let (nh, hd, ch_dim) = (self.nh, self.hd, self.ch_dim);
            let chunk = self.chunk;
            anyhow::ensure!(t <= chunk && chunk == 64, "chunk window");
            self.s.sync()?;
            self.xb.write(0, x)?;
            self.gb.write(0, gpre)?;
            self.bb.write(0, beta)?;
            unsafe {
                launch(
                    k.conv,
                    (ch_dim.div_ceil(256) as u32, b as u32, 1),
                    (256, 1, 1),
                    0,
                    &self.s,
                    args![
                        self.xb.ptr,
                        ch_dim as u32,
                        self.h0.ptr,
                        self.wconv.ptr,
                        self.yb.ptr,
                        t as u32
                    ],
                )?;
                launch(
                    k.conv_commit,
                    (ch_dim.div_ceil(256) as u32, b as u32, 1),
                    (256, 1, 1),
                    0,
                    &self.s,
                    args![self.h0.ptr, self.xb.ptr, ch_dim as u32, t as u32],
                )?;
                launch(
                    k.prefill,
                    (nh as u32, b as u32, 1),
                    (hd as u32, 1, 1),
                    ((chunk * chunk + hd) * 4) as u32,
                    &self.s,
                    args![
                        self.yb.ptr,
                        t as u32,
                        ch_dim as u32,
                        self.gb.ptr,
                        (nh * hd) as u32,
                        self.bb.ptr,
                        nh as u32,
                        self.st.ptr,
                        self.ob.ptr,
                        self.scr.ptr,
                        chunk as u32
                    ],
                )?;
            }
            self.s.sync()?;
            Ok(self.ob.read::<f32>(0, b * t * nh * hd)?)
        }

        /// Allocate the parallel prefill's workspace for windows up to `max_t` tokens:
        /// `min(4, ⌈max_t/64⌉)` chunks are in flight per group (9.5 MB per chunk at the
        /// real 64 × 128 heads), longer windows loop groups.
        pub fn enable_pf2(&mut self, max_t: usize) -> Result<()> {
            let groups = max_t.div_ceil(super::PF_C).clamp(1, 4);
            if groups <= self.pf_groups {
                return Ok(());
            }
            ensure!(
                self.hd <= super::PF_NT,
                "pf2: head_dim {} > {}",
                self.hd,
                super::PF_NT
            );
            let vs = self.hd.min(super::PF_VS);
            ensure!(
                self.hd % vs == 0,
                "pf2: head_dim {} not a multiple of {vs}",
                self.hd
            );
            self.ws = DevBuf::zeroed_elems::<f32>(groups * self.nh * super::pf_slot(self.hd))?;
            self.pf_groups = groups;
            Ok(())
        }

        /// The workspace's size in floats (the debug path NaN-poisons it).
        pub fn ws_elems(&self) -> usize {
            self.pf_groups * self.nh * super::pf_slot(self.hd)
        }

        /// Device-resident prefill of `t ≥ 1` tokens for sequence `slot` (any length): conv
        /// in place on `qkv` (`[t][stride]`, pre-conv q|k|v → post-conv, history committed
        /// to this slot's h0), then the parallel chunked core writing `o` `[t][H·D]` and the
        /// slot's device state. All launches go on `s` (the caller's stream — no host sync,
        /// no host staging).
        #[allow(clippy::too_many_arguments)]
        pub fn prefill_dev(
            &self,
            k: &Kernels,
            s: &Stream,
            slot: usize,
            qkv: u64,
            stride: usize,
            g: u64,
            gstride: usize,
            beta: u64,
            bstride: usize,
            o: u64,
            t: usize,
        ) -> Result<()> {
            ensure!(
                self.pf_groups > 0,
                "pf2 workspace not allocated (enable_pf2)"
            );
            ensure!(t >= 1, "empty prefill");
            let (nh, hd, ch_dim) = (self.nh, self.hd, self.ch_dim);
            let vs = hd.min(super::PF_VS);
            let nc = t.div_ceil(super::PF_C);
            // the kernels index the state by `blockIdx.y` (kda_step/prefill) or take the
            // state pointer directly (conv_ip / pf_state): this slot's base offsets
            let (h0, st) = self.slot_state_ptrs(slot);
            unsafe {
                launch(
                    k.conv_ip,
                    (ch_dim.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![qkv, stride as u32, h0, self.wconv.ptr, t as u32],
                )?;
                let mut c0 = 0usize;
                while c0 < nc {
                    let ng = (nc - c0).min(self.pf_groups);
                    launch(
                        k.pf_intra,
                        (ng as u32, nh as u32, 1),
                        (super::PF_NT as u32, 1, 1),
                        0,
                        s,
                        args![
                            qkv,
                            stride as u32,
                            g,
                            gstride as u32,
                            beta,
                            bstride as u32,
                            self.ws.ptr,
                            c0 as u32,
                            t as u32
                        ],
                    )?;
                    launch(
                        k.pf_state,
                        (nh as u32, (hd / vs) as u32, 1),
                        (super::PF_NT as u32, 1, 1),
                        0,
                        s,
                        args![
                            self.ws.ptr,
                            st,
                            o,
                            c0 as u32,
                            ng as u32,
                            t as u32,
                            vs as u32
                        ],
                    )?;
                    c0 += ng;
                }
            }
            Ok(())
        }

        /// Device-resident decode step (one token, sequence `slot`): `qkv` the raw
        /// `[conv_dim]` row, `g` `[H·D]` pre-exp, `beta` `[H]`; writes `o` `[H·D]`, the
        /// slot's state, and shifts its device conv history. Launches on `s`; the host conv
        /// copy is NOT touched (the device is authoritative; `read_conv` syncs it when
        /// needed).
        pub fn decode_dev(
            &self,
            k: &Kernels,
            s: &Stream,
            slot: usize,
            qkv: u64,
            g: u64,
            beta: u64,
            o: u64,
        ) -> Result<()> {
            let (nh, hd, ch_dim) = (self.nh, self.hd, self.ch_dim);
            let (h0, st) = self.slot_state_ptrs(slot);
            unsafe {
                launch(
                    k.step,
                    (nh as u32, 1, 1),
                    (hd as u32, 1, 1),
                    (3 * hd * 4) as u32,
                    s,
                    args![qkv, h0, self.wconv.ptr, g, beta, st, o],
                )?;
                launch(
                    k.conv_commit,
                    (ch_dim.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![h0, qkv, ch_dim as u32, 1u32],
                )?;
            }
            Ok(())
        }

        /// Slot `slot`'s device state as `[(conv history ptr, bytes), (recurrent state ptr,
        /// bytes)]` (the resident MTP window checkpoints / restores them D2D).
        pub fn slot_state_regions(&self, slot: usize) -> [(u64, usize); 2] {
            let (h0, st) = self.slot_state_ptrs(slot);
            [
                (h0, self.ch_dim * self.d_conv_m1 * 4),
                (st, self.nh * self.hd * self.hd * 4),
            ]
        }

        /// This slot's conv-history / recurrent-state base pointers (the state layout is
        /// `[slot][...]`, so the pointers just offset into the one allocation).
        fn slot_state_ptrs(&self, slot: usize) -> (u64, u64) {
            let f = std::mem::size_of::<f32>() as u64;
            (
                self.h0.ptr + slot as u64 * self.ch_dim as u64 * self.d_conv_m1 as u64 * f,
                self.st.ptr + slot as u64 * self.nh as u64 * self.hd as u64 * self.hd as u64 * f,
            )
        }

        /// The batched decode step: row `i` (its qkv `[conv_dim]`, g `[H·D]`, beta `[H]`,
        /// o `[H·D]` at byte offset `i·<width>·4`) steps `slots[i]`'s state and commits its
        /// conv history — the SAME launches the per-row [`Self::decode_dev`] issues, N
        /// rows in two launches (`kda_step`'s `bi` already indexes QKV/G/S/O rows; only
        /// the commit's H0 offset needs the slot id). `slots` must be distinct (two rows
        /// on one slot would interleave the state updates — the engine never does it).
        pub fn decode_dev_rows(
            &mut self,
            k: &Kernels,
            s: &Stream,
            slots: &[usize],
            qkv: u64,
            g: u64,
            beta: u64,
            o: u64,
        ) -> Result<()> {
            let n = slots.len();
            ensure!(n > 0 && n <= self.slots, "k5 rows: {n} rows");
            let ids: Vec<u32> = slots.iter().map(|&s| s as u32).collect();
            self.slots_dev.write(0, &ids)?;
            let f = std::mem::size_of::<f32>() as u64;
            let (nh, hd) = (self.nh, self.hd);
            unsafe {
                launch(
                    k.step,
                    (nh as u32, n as u32, 1),
                    (hd as u32, 1, 1),
                    (3 * hd * 4) as u32,
                    s,
                    args![qkv, self.h0.ptr, self.wconv.ptr, g, beta, self.st.ptr, o],
                )?;
                launch(
                    k.conv_commit_rows,
                    (self.ch_dim.div_ceil(256) as u32, n as u32, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![self.h0.ptr, qkv, self.slots_dev.ptr],
                )?;
            }
            Ok(())
        }

        /// Read the conv history back (for a CPU fallback window or a checkpoint).
        pub fn read_conv(&self, conv: &mut ConvState) -> Result<()> {
            conv.0 = self.h0.read(0, conv.0.len())?;
            Ok(())
        }

        /// Read the recurrent state back (for a CPU fallback window or a checkpoint).
        pub fn read_state(&self, state: &mut KdaState) -> Result<()> {
            state.0 = self.st.read(0, state.0.len())?;
            Ok(())
        }

        /// Read one slot's conv history back.
        pub fn read_conv_at(&self, slot: usize, conv: &mut ConvState) -> Result<()> {
            let n = self.ch_dim * self.d_conv_m1;
            conv.0 = self.h0.read(slot * n * 4, n)?;
            Ok(())
        }

        /// Read one slot's recurrent state back.
        pub fn read_state_at(&self, slot: usize, state: &mut KdaState) -> Result<()> {
            let n = self.nh * self.hd * self.hd;
            state.0 = self.st.read(slot * n * 4, n)?;
            Ok(())
        }

        /// Push the host copies back onto the device, one slot (after a CPU fallback window
        /// on that slot; the device takes over from the host's values).
        pub fn seed_at(&self, slot: usize, conv: &ConvState, state: &KdaState) -> Result<()> {
            let n = self.nh * self.hd * self.hd;
            let m = self.d_conv_m1;
            let cn = self.ch_dim * m;
            self.h0.write(slot * cn * 4, &conv.0[..cn])?;
            self.st.write(slot * n * 4, &state.0[..n])?;
            Ok(())
        }
    }

    /// Prefill: conv + chunked core, both on GPU, against the CPU reference path.
    /// `x` = pre-conv stacked projections `[b][t][conv_dim]`; returns (o, state, conv).
    pub fn prefill(
        g: &Gpu,
        k: &Kernels,
        cfg: &KdaConfig,
        x: &[f32],
        wconv: &[f32],
        gpre: &[f32],
        beta: &[f32],
        conv: &mut ConvState,
        state: &mut KdaState,
        b: usize,
        t: usize,
        chunk: usize,
    ) -> Result<Vec<f32>> {
        let (nh, hd, ch_dim) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        ensure!(chunk == 64, "the CUDA prefill kernel is fixed at chunk 64");
        let s = sincos(g)?;
        let xb = DevBuf::from_slice(x)?;
        let wb = DevBuf::from_slice(wconv)?;
        let gb = DevBuf::from_slice(gpre)?;
        let bb = DevBuf::from_slice(beta)?;
        let h0 = DevBuf::from_slice(&conv.0)?;
        let st = DevBuf::from_slice(&state.0)?;
        let mut y = DevBuf::zeroed(b * t * ch_dim * 4)?;
        let mut o = DevBuf::zeroed(b * t * nh * hd * 4)?;
        // scratch: per (block, head) 6 planes of C*D — grid is (nh, b)
        let mut scr = DevBuf::zeroed(b * nh * 6 * chunk * hd * 4)?;
        unsafe {
            launch(
                k.conv,
                (ch_dim.div_ceil(256) as u32, b as u32, 1),
                (256, 1, 1),
                0,
                &s,
                args![xb.ptr, ch_dim as u32, h0.ptr, wb.ptr, y.ptr, t as u32],
            )?;
            launch(
                k.conv_commit,
                (ch_dim.div_ceil(256) as u32, b as u32, 1),
                (256, 1, 1),
                0,
                &s,
                args![h0.ptr, xb.ptr, ch_dim as u32, t as u32],
            )?;
            launch(
                k.prefill,
                (nh as u32, b as u32, 1),
                (hd as u32, 1, 1),
                ((chunk * chunk + hd) * 4) as u32,
                &s,
                args![
                    y.ptr,
                    t as u32,
                    ch_dim as u32,
                    gb.ptr,
                    (nh * hd) as u32,
                    bb.ptr,
                    nh as u32,
                    st.ptr,
                    o.ptr,
                    scr.ptr,
                    chunk as u32
                ],
            )?;
        }
        s.sync()?;
        conv.0 = h0.read(0, conv.0.len())?;
        state.0 = st.read(0, state.0.len())?;
        Ok(o.read::<f32>(0, b * t * nh * hd)?)
    }

    /// The parallel prefill through a resident [`DevKda`] (`enable_pf2` + `prefill_dev`)
    /// from host data, any `t`: returns (o, conv-output) and updates `conv`/`state` from
    /// the device. The parity tests' entry point to the exact engine path.
    pub fn prefill2(
        g: &Gpu,
        k: &Kernels,
        cfg: &KdaConfig,
        x: &[f32],
        wconv: &[f32],
        gpre: &[f32],
        beta: &[f32],
        conv: &mut ConvState,
        state: &mut KdaState,
        t: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (nh, hd, ch_dim) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        ensure!(x.len() == t * ch_dim, "prefill2 x shape");
        let mut dev = DevKda::new(g, k, cfg, wconv, conv, state, 1)?;
        dev.enable_pf2(t)?;
        let s = sincos(g)?;
        let xb = DevBuf::from_slice(x)?;
        let gb = DevBuf::from_slice(gpre)?;
        let bb = DevBuf::from_slice(beta)?;
        let o = DevBuf::from_slice(&vec![f32::NAN; t * nh * hd])?;
        // NaN-poison the workspace: a row the core reads but nothing wrote can't hide
        dev.ws.write(0, &vec![f32::NAN; dev.ws_elems()])?;
        dev.prefill_dev(
            k,
            &s,
            0,
            xb.ptr,
            ch_dim,
            gb.ptr,
            nh * hd,
            bb.ptr,
            nh,
            o.ptr,
            t,
        )?;
        s.sync()?;
        dev.read_conv(conv)?;
        dev.read_state(state)?;
        Ok((
            o.read::<f32>(0, t * nh * hd)?,
            xb.read::<f32>(0, t * ch_dim)?,
        ))
    }

    /// One decode step: fused conv + delta rule on GPU, vs the CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        g: &Gpu,
        k: &Kernels,
        cfg: &KdaConfig,
        x: &[f32],
        wconv: &[f32],
        gpre: &[f32],
        beta: &[f32],
        conv: &mut ConvState,
        state: &mut KdaState,
        b: usize,
    ) -> Result<Vec<f32>> {
        let (nh, hd, ch_dim) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        let s = sincos(g)?;
        let xb = DevBuf::from_slice(x)?;
        let wb = DevBuf::from_slice(wconv)?;
        let gb = DevBuf::from_slice(gpre)?;
        let bb = DevBuf::from_slice(beta)?;
        let h0 = DevBuf::from_slice(&conv.0)?;
        let st = DevBuf::from_slice(&state.0)?;
        let mut o = DevBuf::zeroed(b * nh * hd * 4)?;
        unsafe {
            launch(
                k.step,
                (nh as u32, b as u32, 1),
                (hd as u32, 1, 1),
                (3 * hd * 4) as u32,
                &s,
                args![xb.ptr, h0.ptr, wb.ptr, gb.ptr, bb.ptr, st.ptr, o.ptr],
            )?;
        }
        s.sync()?;
        // the step kernel does not update the conv history — shift it on the host (the
        // same semantics as `kda::conv1d_decode`: oldest drops, the new input is newest)
        let m = cfg.d_conv - 1;
        let mut next = conv.0.clone();
        for bi in 0..b {
            for c in 0..ch_dim {
                for s2 in 0..m {
                    next[(bi * ch_dim + c) * m + s2] = if s2 + 1 < m {
                        conv.0[(bi * ch_dim + c) * m + s2 + 1]
                    } else {
                        x[bi * ch_dim + c]
                    };
                }
            }
        }
        conv.0 = next;
        state.0 = st.read(0, state.0.len())?;
        Ok(o.read::<f32>(0, b * nh * hd)?)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::super::kda::{self, ConvState, KdaConfig, KdaState};
    use super::cuda::{decode, load, prefill, prefill2, DevKda};
    use tang_moe::gpu::Gpu;

    /// `Gpu::for_test`, but a missing GPU fails under `TANG_REQUIRE_CUDA=1` (the silent
    /// skip is how GPU bugs survived review before — engine.rs `skip_or_fail`).
    fn gpu() -> Option<Gpu> {
        let g = Gpu::for_test();
        if g.is_none() {
            assert!(
                !std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1"),
                "no CUDA GPU but TANG_REQUIRE_CUDA=1"
            );
            eprintln!("==================== SKIP: no CUDA GPU ====================");
        }
        g
    }

    /// Deterministic pseudo-random data (xorshift64*), uniform in [lo, hi).
    fn synth(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
                lo + (hi - lo) * u
            })
            .collect()
    }

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length");
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                if x.is_finite() {
                    (x - y).abs()
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, f32::max)
    }

    fn max_abs(a: &[f32]) -> f32 {
        a.iter().map(|x| x.abs()).fold(0.0, f32::max)
    }

    /// One parallel-prefill (PF2) window vs the CPU `conv1d_prefill` + `kda_chunk` from the
    /// same (possibly non-zero) initial conv + recurrent state: o, conv output, state and
    /// conv history must agree to `tol` relative to each tensor's max magnitude.
    #[allow(clippy::too_many_arguments)]
    fn pf2_case(
        g: &Gpu,
        kers: &super::cuda::Kernels,
        cfg: &KdaConfig,
        x: &[f32],
        wconv: &[f32],
        gpre: &[f32],
        beta: &[f32],
        conv0: &ConvState,
        state0: &KdaState,
        t: usize,
        tol: f32,
    ) {
        let (nh, hd, ch) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        let (x, gpre, beta) = (&x[..t * ch], &gpre[..t * nh * hd], &beta[..t * nh]);
        let (mut convg, mut stateg) = (conv0.clone(), KdaState(state0.0.clone()));
        let (og, yg) = prefill2(
            g,
            kers,
            cfg,
            x,
            wconv,
            gpre,
            beta,
            &mut convg,
            &mut stateg,
            t,
        )
        .unwrap();
        let (mut convc, mut statec) = (conv0.clone(), KdaState(state0.0.clone()));
        let yc = kda::conv1d_prefill(x, wconv, &mut convc, cfg, 1, t).unwrap();
        let oc = kda::kda_chunk(&yc, gpre, beta, &mut statec, cfg, 1, t, 64).unwrap();
        let rel = |e: f32, r: &[f32]| e / max_abs(r).max(1e-6);
        let (eo, ey, es, ec) = (
            rel(max_err(&og, &oc), &oc),
            rel(max_err(&yg, &yc), &yc),
            rel(max_err(&stateg.0, &statec.0), &statec.0),
            rel(max_err(&convg.0, &convc.0), &convc.0),
        );
        println!(
            "pf2 H={nh} D={hd} T={t}: rel out {eo:.3e} conv_out {ey:.3e} state {es:.3e} conv {ec:.3e} (|o| {:.3e} |S| {:.3e})",
            max_abs(&oc),
            max_abs(&statec.0)
        );
        assert!(eo < tol, "T={t} out {eo}");
        assert!(ey < tol, "T={t} conv_out {ey}");
        assert!(es < tol, "T={t} state {es}");
        assert!(ec < tol, "T={t} conv {ec}");
    }

    /// PF2 on the HF fixture (H=4, D=16), every tail shape the fixture's 100 tokens allow,
    /// from zeros and from the state after 37 tokens (a non-trivial S and conv history).
    #[test]
    fn glm5_kda_pf2_fixture_vs_cpu() {
        let Some(g) = gpu() else { return };
        let cfg = cfg();
        let (nh, hd, ch) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        let (x, wconv, gpre, beta) = (loadf("conv_x"), loadf("conv_w"), loadf("g"), loadf("beta"));
        let kers = load(&g, &cfg).unwrap();
        let (z0, s0) = (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1));
        for t in [1usize, 2, 3, 17, 63, 64, 65, 100] {
            pf2_case(&g, &kers, &cfg, &x, &wconv, &gpre, &beta, &z0, &s0, t, 1e-4);
        }
        // seeded: the CPU's state after the first 37 tokens, then the rest
        let (mut c1, mut s1) = (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1));
        let y = kda::conv1d_prefill(&x[..37 * ch], &wconv, &mut c1, &cfg, 1, 37).unwrap();
        kda::kda_chunk(
            &y,
            &gpre[..37 * nh * hd],
            &beta[..37 * nh],
            &mut s1,
            &cfg,
            1,
            37,
            64,
        )
        .unwrap();
        for t in [1usize, 17, 63] {
            pf2_case(
                &g,
                &kers,
                &cfg,
                &x[37 * ch..],
                &wconv,
                &gpre[37 * nh * hd..],
                &beta[37 * nh..],
                &c1,
                &s1,
                t,
                1e-4,
            );
        }
    }

    /// PF2 on synthetic data at the fixture's shape AND the real model's head shape
    /// (D = 128: 4 value slices per head, all 8 warps on the key dim), for the tail lengths
    /// 1, 17, 63, 65, 127, 150 and a multi-group window (300 = 5 chunks > 4 in flight),
    /// from a random non-zero state + conv history.
    #[test]
    fn glm5_kda_pf2_synth_vs_cpu() {
        let Some(g) = gpu() else { return };
        for (heads, head_dim) in [(4usize, 16usize), (3, 128)] {
            let cfg = KdaConfig {
                heads,
                head_dim,
                ..cfg()
            };
            let (nh, hd, ch) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
            let tmax = 300usize;
            let x = synth(tmax * ch, 1, -1.5, 1.5);
            let wconv = synth(ch * cfg.d_conv, 2, -0.6, 0.6);
            // pre-exp gates in (lb, 0) like lb·σ(·): mostly mild decay, some strong
            let gpre: Vec<f32> = synth(tmax * nh * hd, 3, 0.0, 1.0)
                .iter()
                .map(|u| -5.0 * u * u * u * u)
                .collect();
            let beta = synth(tmax * nh, 4, 0.05, 0.95);
            let conv0 = ConvState(synth(ConvState::zeros(&cfg, 1).0.len(), 5, -1.0, 1.0));
            let state0 = KdaState(synth(nh * hd * hd, 6, -0.3, 0.3));
            let kers = load(&g, &cfg).unwrap();
            for t in [1usize, 17, 63, 64, 65, 127, 150, 300] {
                pf2_case(
                    &g, &kers, &cfg, &x, &wconv, &gpre, &beta, &conv0, &state0, t, 2e-4,
                );
            }
        }
    }

    fn loadf(name: &str) -> Vec<f32> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glm5/kda/",);
        let b = std::fs::read(format!("{p}{name}.bin")).unwrap();
        b.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn cfg() -> KdaConfig {
        KdaConfig {
            heads: 4,
            head_dim: 16,
            d_conv: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1e-5,
        }
    }

    /// Core-only timing at the real GLM-5.3 shape (64 heads × 128, conv 4), per layer:
    /// the original window flow (host-staged 64-chunks through `DevKda::prefill` + the CPU
    /// `kda_chunk` tail, exactly `DevLayer::window`'s old core) vs PF2 (`prefill_dev`, data
    /// already on the device). Ignored by default; run with `-- --ignored glm5_kda_pf2_bench`.
    #[test]
    #[ignore]
    fn glm5_kda_pf2_bench() {
        let Some(g) = gpu() else { return };
        let cfg = KdaConfig::glm53();
        let (nh, hd, ch) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        let kers = load(&g, &cfg).unwrap();
        let tmax = 2048usize;
        let x = synth(tmax * ch, 1, -1.5, 1.5);
        let wconv = synth(ch * cfg.d_conv, 2, -0.6, 0.6);
        let gpre: Vec<f32> = synth(tmax * nh * hd, 3, 0.0, 1.0)
            .iter()
            .map(|u| -5.0 * u * u * u * u)
            .collect();
        let beta = synth(tmax * nh, 4, 0.05, 0.95);
        let (c0, s0) = (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1));
        let s = tang_moe::gpu::Stream::blocking().unwrap();
        for t in [20usize, 64, 512, 2048] {
            let (xs, gs, bs) = (&x[..t * ch], &gpre[..t * nh * hd], &beta[..t * nh]);
            // old: 64-chunks on the device from host data, CPU tail
            let mut dev = DevKda::new(&g, &kers, &cfg, &wconv, &c0, &s0, 1).unwrap();
            let mut conv = c0.clone();
            let mut dummy = KdaState::zeros(&cfg, 1);
            let t_old = std::time::Instant::now();
            let mut off = 0;
            while off + 64 <= t {
                dev.prefill(
                    &kers,
                    &cfg,
                    &xs[off * ch..(off + 64) * ch],
                    &gs[off * nh * hd..(off + 64) * nh * hd],
                    &bs[off * nh..(off + 64) * nh],
                    &mut conv,
                    &mut dummy,
                    1,
                    64,
                )
                .unwrap();
                off += 64;
            }
            if off < t {
                let (mut hc, mut hs) = (conv.clone(), KdaState::zeros(&cfg, 1));
                dev.read_conv(&mut hc).unwrap();
                dev.read_state(&mut hs).unwrap();
                let y = kda::conv1d_prefill(&xs[off * ch..], &wconv, &mut hc, &cfg, 1, t - off)
                    .unwrap();
                kda::kda_chunk(
                    &y,
                    &gs[off * nh * hd..],
                    &bs[off * nh..],
                    &mut hs,
                    &cfg,
                    1,
                    t - off,
                    64,
                )
                .unwrap();
                dev.h0.write(0, &hc.0).unwrap();
                dev.st.write(0, &hs.0).unwrap();
            }
            let old_ms = t_old.elapsed().as_secs_f64() * 1e3;
            // PF2: inputs device-resident (the engine's projections write them there)
            let mut dev2 = DevKda::new(&g, &kers, &cfg, &wconv, &c0, &s0, 1).unwrap();
            dev2.enable_pf2(t.min(512)).unwrap();
            let xb = tang_moe::gpu::DevBuf::from_slice(xs).unwrap();
            let gb = tang_moe::gpu::DevBuf::from_slice(gs).unwrap();
            let bb = tang_moe::gpu::DevBuf::from_slice(bs).unwrap();
            let ob = tang_moe::gpu::DevBuf::zeroed_elems::<f32>(t * nh * hd).unwrap();
            let mut best = f64::MAX;
            for _ in 0..3 {
                xb.write(0, xs).unwrap(); // the conv runs in place: restore the input
                let t0 = std::time::Instant::now();
                dev2.prefill_dev(
                    &kers,
                    &s,
                    0,
                    xb.ptr,
                    ch,
                    gb.ptr,
                    nh * hd,
                    bb.ptr,
                    nh,
                    ob.ptr,
                    t,
                )
                .unwrap();
                s.sync().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            println!(
                "pf2 bench T={t}: old core {old_ms:.2} ms/layer, pf2 {best:.3} ms/layer ({:.0}x) — x34 layers: old {:.0} ms, pf2 {:.1} ms",
                old_ms / best,
                old_ms * 34.0,
                best * 34.0
            );
        }
    }

    /// GPU prefill (conv + chunk 64) vs the CPU reference on the HF fixture, at several
    /// window lengths so the state handoff across chunk boundaries is exercised.
    #[test]
    fn glm5_kda_prefill_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (nh, hd) = (cfg.heads, cfg.head_dim);
        let x = loadf("conv_x");
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let kers = load(&g, &cfg).unwrap();
        for t in [1usize, 63, 64, 65, 100] {
            let b = 1usize;
            let mut conv = ConvState::zeros(&cfg, b);
            let mut state = KdaState::zeros(&cfg, b);
            let og = prefill(
                &g,
                &kers,
                &cfg,
                &x[..t * cfg.conv_dim()],
                &wconv,
                &gpre[..t * nh * hd],
                &beta[..t * nh],
                &mut conv,
                &mut state,
                b,
                t,
                64,
            )
            .unwrap();
            let mut conv2 = ConvState::zeros(&cfg, b);
            let mut state2 = KdaState::zeros(&cfg, b);
            let y = kda::conv1d_prefill(&x[..t * cfg.conv_dim()], &wconv, &mut conv2, &cfg, b, t)
                .unwrap();
            let oc = kda::kda_chunk(
                &y,
                &gpre[..t * nh * hd],
                &beta[..t * nh],
                &mut state2,
                &cfg,
                b,
                t,
                64,
            )
            .unwrap();
            let e: f32 = og
                .iter()
                .zip(&oc)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            let es: f32 = state
                .0
                .iter()
                .zip(&state2.0)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            let ec: f32 = conv
                .0
                .iter()
                .zip(&conv2.0)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            println!("T={t}: out {e:.3e} state {es:.3e} conv {ec:.3e}");
            assert!(e < 1e-4, "T={t} out {e}");
            assert!(es < 1e-4, "T={t} state {es}");
            assert!(ec < 1e-4, "T={t} conv {ec}");
        }
    }

    /// Cross-chunk isolation: seed both sides with the CPU's state after 64 tokens, then
    /// run tokens 64..65 through the FULL T=65 window on GPU (two chunks in one launch).
    #[test]
    fn glm5_kda_chunk1_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (nh, hd) = (cfg.heads, cfg.head_dim);
        let x = loadf("conv_x");
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let b = 1usize;
        // CPU state after the first 64 tokens
        let mut conv2 = ConvState::zeros(&cfg, b);
        let mut state2 = KdaState::zeros(&cfg, b);
        let y0 = kda::conv1d_prefill(&x[..64 * cfg.conv_dim()], &wconv, &mut conv2, &cfg, b, 64)
            .unwrap();
        let _ = kda::kda_chunk(
            &y0,
            &gpre[..64 * nh * hd],
            &beta[..64 * nh],
            &mut state2,
            &cfg,
            b,
            64,
            64,
        )
        .unwrap();
        let kers = load(&g, &cfg).unwrap();
        let t = 65usize;
        // GPU: full T=65 (2 chunks), state/conv seeded with the CPU's after-64 values
        let mut conv = conv2.clone();
        let mut state = KdaState(state2.0.clone());
        let og = prefill(
            &g,
            &kers,
            &cfg,
            &x[..t * cfg.conv_dim()],
            &wconv,
            &gpre[..t * nh * hd],
            &beta[..t * nh],
            &mut conv,
            &mut state,
            b,
            t,
            64,
        )
        .unwrap();
        let mut conv3 = conv2.clone();
        let mut state3 = KdaState(state2.0.clone());
        let y =
            kda::conv1d_prefill(&x[..t * cfg.conv_dim()], &wconv, &mut conv3, &cfg, b, t).unwrap();
        let oc = kda::kda_chunk(
            &y,
            &gpre[..t * nh * hd],
            &beta[..t * nh],
            &mut state3,
            &cfg,
            b,
            t,
            64,
        )
        .unwrap();
        for ti in 63..t {
            let mut et = 0f32;
            for j in 0..nh * hd {
                et = et.max((og[ti * nh * hd + j] - oc[ti * nh * hd + j]).abs());
            }
            println!("  t={ti} err={et:.3e}");
        }
        let e: f32 = og
            .iter()
            .zip(&oc)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        let es: f32 = state
            .0
            .iter()
            .zip(&state3.0)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        println!("chunk-cross: out {e:.3e} state {es:.3e}");
        assert!(es < 1e-4, "state {es}");
        assert!(e < 1e-4, "out {e}");
    }

    /// The resident `DevKda` path: 3 chained decodes + a 64-token prefill, all with no
    /// per-call allocations, vs the CPU reference. States must match to 1e-4.
    #[test]
    fn glm5_kda_resident_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (b, nh, hd, ch_dim) = (1usize, cfg.heads, cfg.head_dim, cfg.conv_dim());
        let x = loadf("conv_x");
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let kers = load(&g, &cfg).unwrap();
        let mut convg = ConvState::zeros(&cfg, b);
        let mut stateg = KdaState::zeros(&cfg, b);
        let mut dev = DevKda::new(&g, &kers, &cfg, &wconv, &convg, &stateg, b).unwrap();
        let mut convc = ConvState::zeros(&cfg, b);
        let mut statec = KdaState::zeros(&cfg, b);
        // 3 chained decodes
        for ti in 0..3 {
            let row = &x[ti * ch_dim..(ti + 1) * ch_dim];
            let og = dev
                .decode(
                    &kers,
                    &cfg,
                    row,
                    &gpre[ti * nh * hd..(ti + 1) * nh * hd],
                    &beta[ti * nh..(ti + 1) * nh],
                    &mut convg,
                    &mut stateg,
                    b,
                )
                .unwrap();
            let yc = kda::conv1d_decode(row, &wconv, &mut convc, &cfg, b).unwrap();
            let oc = kda::kda_decode(
                &yc,
                &gpre[ti * nh * hd..(ti + 1) * nh * hd],
                &beta[ti * nh..(ti + 1) * nh],
                &mut statec,
                &cfg,
                b,
            )
            .unwrap();
            let e: f32 = og
                .iter()
                .zip(&oc)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            assert!(e < 1e-4, "decode token {ti}: {e}");
        }
        // then a 64-token prefill on the same resident state
        let t = 64usize;
        let og = dev
            .prefill(
                &kers,
                &cfg,
                &x[3 * ch_dim..(3 + t) * ch_dim],
                &gpre[3 * nh * hd..(3 + t) * nh * hd],
                &beta[3 * nh..(3 + t) * nh],
                &mut convg,
                &mut stateg,
                b,
                t,
            )
            .unwrap();
        let yc = kda::conv1d_prefill(
            &x[3 * ch_dim..(3 + t) * ch_dim],
            &wconv,
            &mut convc,
            &cfg,
            b,
            t,
        )
        .unwrap();
        let oc = kda::kda_chunk(
            &yc,
            &gpre[3 * nh * hd..(3 + t) * nh * hd],
            &beta[3 * nh..(3 + t) * nh],
            &mut statec,
            &cfg,
            b,
            t,
            64,
        )
        .unwrap();
        let e: f32 = og
            .iter()
            .zip(&oc)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        println!("resident prefill err {e:.3e}");
        assert!(e < 1e-4, "prefill {e}");
        // read-back must agree with the CPU's states
        dev.read_state(&mut stateg).unwrap();
        dev.read_conv(&mut convg).unwrap();
        let es: f32 = stateg
            .0
            .iter()
            .zip(&statec.0)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        let ec: f32 = convg
            .0
            .iter()
            .zip(&convc.0)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        println!("resident states: state {es:.3e} conv {ec:.3e}");
        assert!(es < 1e-4, "state {es}");
        assert!(ec < 1e-4, "conv {ec}");
    }

    /// GPU decode step (fused conv + delta rule) vs the CPU reference, 3 tokens chained.
    #[test]
    fn glm5_kda_decode_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (b, nh, hd, ch) = (
            1usize,
            cfg.heads,
            cfg.head_dim,
            3 * cfg.heads * cfg.head_dim,
        );
        let x = loadf("conv_x"); // the conv input rows (qkv), [t][conv_dim]
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let kers = load(&g, &cfg).unwrap();
        let mut convg = ConvState::zeros(&cfg, b);
        let mut stateg = KdaState::zeros(&cfg, b);
        let mut convc = ConvState::zeros(&cfg, b);
        let mut statec = KdaState::zeros(&cfg, b);
        // run 3 tokens from the conv_x fixture (its conv state is zeroed, matching both)
        for ti in 0..3 {
            let row = &x[ti * ch..(ti + 1) * ch];
            let og = decode(
                &g,
                &kers,
                &cfg,
                row,
                &wconv,
                &gpre[ti * nh * hd..(ti + 1) * nh * hd],
                &beta[ti * nh..(ti + 1) * nh],
                &mut convg,
                &mut stateg,
                b,
            )
            .unwrap();
            let yc = kda::conv1d_decode(row, &wconv, &mut convc, &cfg, b).unwrap();
            let oc = kda::kda_decode(
                &yc,
                &gpre[ti * nh * hd..(ti + 1) * nh * hd],
                &beta[ti * nh..(ti + 1) * nh],
                &mut statec,
                &cfg,
                b,
            )
            .unwrap();
            let e: f32 = og
                .iter()
                .zip(&oc)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            println!("token {ti}: GPU vs CPU out {e:.3e}");
            assert!(e < 1e-4, "token {ti}: {e}");
        }
        let es: f32 = stateg
            .0
            .iter()
            .zip(&statec.0)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        assert!(es < 1e-4, "state {es}");
    }
}
