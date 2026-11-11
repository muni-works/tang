//! Fused, device-resident mHC (speed plan step 4, `TANG_GLM5_MHC_FUSED=1`).
//!
//! [`super::mhc::MhcGpu`] round-trips the host on every pre and post: upload `r`, launch,
//! read `x`/`post`/`comb` back; then repack, re-upload `r`/`y`/`post`/`comb` and read the
//! 4 streams back — 4 host round-trips per half block, 180 per decode step, plus a
//! one-warp-per-token mixes GEMV (24 × 16384 dots on 32 lanes).
//!
//! Here the 4 residual streams `r` `[t][4][n]` live on the device for a whole window:
//!
//! - `begin`: upload the embedded streams once.
//! - `pre(l, block)`: **one** kernel applies the PREVIOUS half's post in place
//!   (`r ← post ⊙ y + combᵀ r`, per feature across the 4 streams, so in-place is safe) and
//!   accumulates this half's 24 mix dots + the flat sum of squares over its feature
//!   chunk (`mhcf_part`, grid `(n/32 padded to 4, t)`, 8 warps splitting the 25 partials); a second (`mhcf_finish`, one block per
//!   token) reduces the chunks, runs pre/post/comb + Sinkhorn, collapses to `x` and
//!   applies the block's weighted RMSNorm (`attn_norm` / `ffn_norm`) in the same
//!   kernel. Only `x` (`t·n` f32) comes back to the host — the mixers take host input.
//! - `post(y)`: upload `y` (`t·n` f32); the post itself is deferred into the next pre.
//! - `end`: the last post + the head's unweighted stream mean in one launch; only
//!   `h` `[t][n]` comes back.
//!
//! Per half block that is 2 launches, one `t·n` upload and one `t·n` download (was:
//! 2 launches, 4 uploads and 4 downloads incl. two `t·4n`, plus a host repack and the
//! host norm).
//!
//! The math is op-for-op the CPU reference ([`super::mhc::mhc_pre`] / `mhc_post` /
//! `hc_head` and the engine's `norm_all`), except that the mixes are `inv · (fn · r)`
//! (the RMS factor applied after the dot instead of to every column; equal in exact
//! arithmetic) and the sums run in a tree instead of serially.
//!
//! `TANG_GLM5_MHC_DEBUG=1` shadows every pre/post on the CPU from the device's own `r`
//! (each step is checked alone, no accumulated drift) and prints `mhcdbg` lines plus a
//! `mhcdbg SUMMARY` at teardown. It also NaN-poisons every kernel output (`x`, `post`,
//! `comb`, `part`, `h`) before each launch, so a region no kernel wrote reads NaN, not
//! a plausible stale value.

use super::mhc::{MhcGpu, MhcWeights, Mix};
use anyhow::{anyhow, bail, Result};
use tang_moe::gpu::{DevBuf, Gpu, Module, Stream};

/// Features per `mhcf_part` block (one warp's worth; the block's PW warps split the 25
/// partials). The chunk count is padded to a multiple of 4 (= the old 128-feature chunk).
const FB: usize = 32;
/// Warps per `mhcf_part` block (`PW` in the kernel source).
const PW: usize = 8;
/// The finish's default when `TANG_GLM5_MHC_EXACT` is unset: false = exact.
const FAST_DEFAULT: bool = false;
/// `TANG_GLM5_MHC_FOLD`'s default (the folded kernel measured slower on the B300).
const FOLD_DEFAULT: bool = false;

/// The fused kernels' CUDA C (NVRTC). Layouts: `r` `[t][4][n]`, `y`/`x`/`h` `[t][n]`,
/// `post` `[t][4]`, `comb` `[t][16]` (`src*4+dst`), `part` `[t][ch][25]` (24 raw dots,
/// then the sum of squares).
pub const SRC: &str = r#"
#define HC 4
#define NP 25

__device__ __forceinline__ float sigm_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float wsum_f(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ double wsum_d(double v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Grid (ch32, t), block 32*PW (PW warps). Block c owns features [c*32, c*32+32) of all
// 4 streams; ch32 is a multiple of 4 (4 consecutive blocks = the old 128-feature chunk).
// Every warp loads the same 32 features (lane = feature) and applies the post in
// registers; warp w then computes the dots m = w, w+PW, ... (24 dots + the sum of
// squares at m = 24) with the same per-feature fmaf order and the same warp tree as
// before, so each warp partial is bit-identical to the old block's per-warp partial.
//  do_post: r <- post(prev) (warp 0 writes after a barrier: every warp has read r first),
//           and if h != 0 also h = mean_s r (the head collapse).
//  do_part: then this block's per-m warp sums -> part[t][c][m].
#define PW 8
extern "C" __global__ void mhcf_part(float* r, const float* y, const float* post,
                                     const float* comb, const float* fn, float* part,
                                     float* h, int n, int do_post, int do_part) {
    const int c = blockIdx.x, t = blockIdx.y, ch = gridDim.x;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int f = c * 32 + lane;
    const int hn = HC * n;
    float* rt = r + (size_t)t * hn;
    const bool ok = f < n;
    // this warp's fn rows (m = wid + j*PW) loaded first: independent of the r -> post
    // chain, so their latency overlaps it (same values, same fmaf order as before)
    #define MJ ((NP + PW - 1) / PW)
    float wf[MJ][4];
    #pragma unroll
    for (int j = 0; j < MJ; j++) {
        const int m = wid + j * PW;
        #pragma unroll
        for (int s = 0; s < 4; s++)
            wf[j][s] = (do_part && ok && m < 24) ? fn[(size_t)m * hn + (size_t)s * n + f] : 0.f;
    }
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (ok) {
        #pragma unroll
        for (int s = 0; s < 4; s++) v[s] = rt[(size_t)s * n + f];
        if (do_post) {
            const float* pt = post + (size_t)t * 4;
            const float* ct = comb + (size_t)t * 16;
            const float yv = y[(size_t)t * n + f];
            float o[4];
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                float acc = pt[d] * yv;
                #pragma unroll
                for (int s = 0; s < 4; s++) acc = fmaf(ct[s * 4 + d], v[s], acc);
                o[d] = acc;
            }
            #pragma unroll
            for (int d = 0; d < 4; d++) v[d] = o[d];
        }
    }
    if (do_post) {   // uniform
        __syncthreads();
        if (ok && wid == 0) {
            #pragma unroll
            for (int d = 0; d < 4; d++) rt[(size_t)d * n + f] = v[d];
            if (h) h[(size_t)t * n + f] = (((v[0] + v[1]) + v[2]) + v[3]) / 4.0f;
        }
    }
    if (!do_part) return;   // uniform across the grid
    #pragma unroll
    for (int j = 0; j < MJ; j++) {
        const int m = wid + j * PW;
        if (m >= NP) break;
        float a = 0.f;
        if (ok) {
            if (m < 24) {
                #pragma unroll
                for (int s = 0; s < 4; s++) a = fmaf(wf[j][s], v[s], a);
            } else {
                #pragma unroll
                for (int s = 0; s < 4; s++) a = fmaf(v[s], v[s], a);
            }
        }
        a = wsum_f(a);
        if (lane == 0) part[((size_t)t * ch + c) * NP + m] = a;
    }
}

// The pre's finish for token t, one block of bd threads (bd a multiple of 32, >= OW + 32,
// <= 1024), dynamic smem max(n, ch*NP) floats. Reduce the chunks, then pre/post/comb +
// Sinkhorn exactly as mhc.rs's mhc_pre, collapse x = sum_s pre[s] r[s], and if nw != 0
// the weighted RMSNorm x <- x * rsqrt(mean x^2 + eps) * nw.
//  fast == 0 (exact): every serial sum keeps the old one-block kernel's order: the chunk
//   reduction (25 threads), the Sinkhorn (the old lane-0 loop's order, IEEE divides) and
//   the f64 sum of squares (owner o < OW sums i = o + k*OW, k ascending, then the warp
//   tree and warps in order). Bit-identical to the CPU-ordered device path.
//  fast == 1: tree chunk reduction straight from global (no smem copy), __fdividef in the
//   Sinkhorn, and an f32 sum of squares from registers (per-thread, warp tree, warps in
//   order). Not bit-identical; same math.
// r and part are read with ld.global.cg: in the folded kernel other blocks wrote them.
// The Sinkhorn (warp 0) runs concurrently with the sum of squares (warps 1..): pre needs
// only the sigmoid, not the doubly-stochastic comb.
#define RK 6
#define OW 256
__device__ __forceinline__ void finish_body(
        const float* rt, const float* pt, int ch, const float* base, const float* scale,
        const float* nw, float* xt, float* postt, float* combt, int n, int iters, float hc_eps,
        float rms_eps, float norm_eps, int fast, float* sm) {
    const int tid = threadIdx.x, bd = blockDim.x;
    const int hn = HC * n;
    __shared__ float sp[NP];
    __shared__ float pre[4];
    __shared__ float sinv_s;
    __shared__ double red[32];
    __shared__ float redf[32];
    // prefetch this thread's r and norm weight (i = tid + k*bd, k < RK) so their latency
    // overlaps the reductions; anything past RK*bd is read late below
    float q[RK][4], g[RK];
    const int pw_ = bd - 32;   // the collapse threads (the last warp runs the Sinkhorn)
    #pragma unroll
    for (int k = 0; k < RK; k++) {
        const int i = tid < pw_ ? tid + k * pw_ : n;
        #pragma unroll
        for (int s = 0; s < 4; s++) q[k][s] = i < n ? __ldcg(rt + (size_t)s * n + i) : 0.f;
        g[k] = (nw && i < n) ? nw[i] : 0.f;
    }
    if (fast) {
        // warp w reduces columns m = w, w + nwarps, ...: lane-strided sums, then the warp tree
        const int w = tid >> 5, ln = tid & 31;
        for (int m = w; m < NP; m += bd >> 5) {
            float a = 0.f;
            for (int c = ln; c < ch; c += 32) a += __ldcg(pt + c * NP + m);
            a = wsum_f(a);
            if (ln == 0) sp[m] = a;
        }
        __syncthreads();
    } else {
        for (int i = tid; i < ch * NP; i += bd) sm[i] = __ldcg(pt + i);
        __syncthreads();
        if (tid < NP) {
            // ch counts 32-feature warp partials; 4 of them make one old 128-feature chunk,
            // summed first in warp order (the old block's order), then chunk by chunk.
            float s = 0.f;
            for (int c = 0; c < ch; c += 4) {
                const float* pc = sm + c * NP + tid;
                float b = 0.f;
                b += pc[0];
                b += pc[NP];
                b += pc[2 * NP];
                b += pc[3 * NP];
                s += b;
            }
            sp[tid] = s;
        }
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(sp[24] / (float)hn + rms_eps);
    if (tid < 4) {
        pre[tid] = sigm_f((sp[tid] * inv) * scale[0] + base[tid]) + hc_eps;
        postt[tid] = 2.0f * sigm_f((sp[4 + tid] * inv) * scale[1] + base[4 + tid]);
    }
    __syncthreads();   // pre visible; sp no longer written, sm (the part copy) free
    const int lw = (bd >> 5) - 1;
    if ((tid >> 5) == lw) {
        // the last warp: the Sinkhorn, off the collapse/norm path (pre needs only the
        // sigmoid; the comb is read by the next pre's post). Lane i (mod 16) owns comb[i] =
        // cb[s*4+d]; every sum gathers its 4 terms by shuffle and adds them in the old
        // serial loop's order (bit-identical when exact; a one-thread register loop is ~2x
        // slower: its divides serialize). The other warps sync on named barrier 1 only.
        const int i = tid & 15, s = i >> 2, d = i & 3;
        const float e = expf((sp[8 + i] * inv) * scale[2] + base[8 + i]);
        const unsigned F = 0xffffffffu;
        float z = 0.f;
        z += __shfl_sync(F, e, s * 4 + 0);
        z += __shfl_sync(F, e, s * 4 + 1);
        z += __shfl_sync(F, e, s * 4 + 2);
        z += __shfl_sync(F, e, s * 4 + 3);
        float cb;
        if (fast) {
            cb = __fdividef(e, z) + hc_eps;
            z = __shfl_sync(F, cb, d) + __shfl_sync(F, cb, 4 + d) + __shfl_sync(F, cb, 8 + d)
                + __shfl_sync(F, cb, 12 + d) + hc_eps;
            cb = __fdividef(cb, z);
            for (int it = 1; it < iters; it++) {
                z = __shfl_sync(F, cb, s * 4) + __shfl_sync(F, cb, s * 4 + 1)
                    + __shfl_sync(F, cb, s * 4 + 2) + __shfl_sync(F, cb, s * 4 + 3) + hc_eps;
                cb = __fdividef(cb, z);
                z = __shfl_sync(F, cb, d) + __shfl_sync(F, cb, 4 + d) + __shfl_sync(F, cb, 8 + d)
                    + __shfl_sync(F, cb, 12 + d) + hc_eps;
                cb = __fdividef(cb, z);
            }
        } else {
            cb = e / z + hc_eps;
            z = __shfl_sync(F, cb, d) + __shfl_sync(F, cb, 4 + d) + __shfl_sync(F, cb, 8 + d)
                + __shfl_sync(F, cb, 12 + d) + hc_eps;
            cb /= z;
            for (int it = 1; it < iters; it++) {
                z = __shfl_sync(F, cb, s * 4) + __shfl_sync(F, cb, s * 4 + 1)
                    + __shfl_sync(F, cb, s * 4 + 2) + __shfl_sync(F, cb, s * 4 + 3) + hc_eps;
                cb /= z;
                z = __shfl_sync(F, cb, d) + __shfl_sync(F, cb, 4 + d) + __shfl_sync(F, cb, 8 + d)
                    + __shfl_sync(F, cb, 12 + d) + hc_eps;
                cb /= z;
            }
        }
        if ((tid & 31) < 16) combt[i] = cb;
        return;
    }
    // warps 0..lw-1 (bd - 32 threads) from here; element i is owned by i mod (bd - 32)
    const int bw = bd - 32;
    #define NBAR() asm volatile("bar.sync 1, %0;" :: "r"(bw) : "memory")
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    float xa[RK];
    float ssf = 0.f;
    #pragma unroll
    for (int k = 0; k < RK; k++) {
        const int i = tid + k * bw;
        xa[k] = 0.f;
        if (i < n) {
            float a = 0.f;
            a = fmaf(p0, q[k][0], a);
            a = fmaf(p1, q[k][1], a);
            a = fmaf(p2, q[k][2], a);
            a = fmaf(p3, q[k][3], a);
            xa[k] = a;
            ssf = fmaf(a, a, ssf);
            if (!fast) sm[i] = a;
            if (!nw) xt[i] = a;
        }
    }
    for (int i = tid + RK * bw; i < n; i += bw) {
        float a = 0.f;
        a = fmaf(p0, __ldcg(rt + i), a);
        a = fmaf(p1, __ldcg(rt + (size_t)n + i), a);
        a = fmaf(p2, __ldcg(rt + (size_t)2 * n + i), a);
        a = fmaf(p3, __ldcg(rt + (size_t)3 * n + i), a);
        sm[i] = a;
        ssf = fmaf(a, a, ssf);
        if (!nw) xt[i] = a;
    }
    if (!nw) return;   // uniform
    if (fast) {
        ssf = wsum_f(ssf);
        if ((tid & 31) == 0) redf[tid >> 5] = ssf;
        NBAR();
        if (tid == 0) {
            float s = 0.f;
            for (int w = 0; w < lw; w++) s += redf[w];
            sinv_s = 1.0f / sqrtf(s / (float)n + norm_eps);
        }
    } else {
        NBAR();
        const int o = tid;   // owners are warps 0..OW/32-1
        if (o < OW) {
            double ss = 0.0;
            for (int i = o; i < n; i += OW) {
                const float a = sm[i];
                ss += (double)a * (double)a;
            }
            ss = wsum_d(ss);
            if ((o & 31) == 0) red[o >> 5] = ss;
        }
        NBAR();
        if (tid == 0) {
            double s = 0.0;
            for (int w = 0; w < OW / 32; w++) s += red[w];
            sinv_s = 1.0f / sqrtf((float)(s / (double)n) + norm_eps);
        }
    }
    NBAR();
    const float rr = sinv_s;
    #pragma unroll
    for (int k = 0; k < RK; k++) {
        const int i = tid + k * bw;
        if (i < n) xt[i] = xa[k] * rr * g[k];
    }
    for (int i = tid + RK * bw; i < n; i += bw) xt[i] = sm[i] * rr * nw[i];
    #undef NBAR
}

// Grid t, block 1024: the unfolded finish (TANG_GLM5_MHC_FOLD=0).
extern "C" __global__ void __launch_bounds__(1024) mhcf_finish(
        const float* r, const float* part, int ch, const float* base, const float* scale,
        const float* nw, float* x, float* post, float* comb, int n, int iters, float hc_eps,
        float rms_eps, float norm_eps, int fast) {
    extern __shared__ float sm[];
    const int t = blockIdx.x;
    finish_body(r + (size_t)t * HC * n, part + (size_t)t * ch * NP, ch, base, scale, nw,
                x + (size_t)t * n, post + (size_t)t * 4, comb + (size_t)t * 16, n, iters,
                hc_eps, rms_eps, norm_eps, fast, sm);
}

// Grid (ch32, t), block 1024: mhcf_part (warps 0..PW-1; the rest idle) + the finish in
// the last block to arrive for token t (threadfence-reduction pattern: every block's
// r / part writes are fenced before its arrival; the last block resets the counter, so a
// graph replay starts at 0). One launch per pre instead of two.
extern "C" __global__ void __launch_bounds__(1024) mhcf_fold(
        float* r, const float* y, float* post, float* comb, const float* fn, float* part,
        int n, int do_post, const float* base, const float* scale, const float* nw, float* x,
        int iters, float hc_eps, float rms_eps, float norm_eps, int fast, unsigned* cnt) {
    extern __shared__ float sm[];
    const int c = blockIdx.x, t = blockIdx.y, ch = gridDim.x;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int f = c * 32 + lane;
    const int hn = HC * n;
    float* rt = r + (size_t)t * hn;
    const bool act = wid < PW;
    const bool ok = act && f < n;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (ok) {
        #pragma unroll
        for (int s = 0; s < 4; s++) v[s] = rt[(size_t)s * n + f];
        if (do_post) {
            const float* pt = post + (size_t)t * 4;
            const float* ct = comb + (size_t)t * 16;
            const float yv = y[(size_t)t * n + f];
            float o[4];
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                float acc = pt[d] * yv;
                #pragma unroll
                for (int s = 0; s < 4; s++) acc = fmaf(ct[s * 4 + d], v[s], acc);
                o[d] = acc;
            }
            #pragma unroll
            for (int d = 0; d < 4; d++) v[d] = o[d];
        }
    }
    if (do_post) {   // uniform
        __syncthreads();
        if (ok && wid == 0) {
            #pragma unroll
            for (int d = 0; d < 4; d++) rt[(size_t)d * n + f] = v[d];
        }
    }
    if (act) {
        for (int m = wid; m < NP; m += PW) {
            float a = 0.f;
            if (ok) {
                if (m < 24) {
                    const float* w = fn + (size_t)m * hn + f;
                    #pragma unroll
                    for (int s = 0; s < 4; s++) a = fmaf(w[(size_t)s * n], v[s], a);
                } else {
                    #pragma unroll
                    for (int s = 0; s < 4; s++) a = fmaf(v[s], v[s], a);
                }
            }
            a = wsum_f(a);
            if (lane == 0) part[((size_t)t * ch + c) * NP + m] = a;
        }
    }
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(cnt + t, 1u) == (unsigned)(ch - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) cnt[t] = 0u;
    finish_body(rt, part + (size_t)t * ch * NP, ch, base, scale, nw, x + (size_t)t * n,
                post + (size_t)t * 4, comb + (size_t)t * 16, n, iters, hc_eps, rms_eps,
                norm_eps, fast, sm);
}
"#;

/// GPU-vs-CPU statistics of one tensor (local, so this file doesn't depend on where the
/// shadow-check helpers live: kda_debug today, tang-verify after that branch lands).
#[derive(Clone, Debug, Default)]
struct Stat {
    rel_l2: f64,
    max_abs: f64,
    max_idx: usize,
    nonfinite_gpu: usize,
    nonfinite_cpu: usize,
    len_mismatch: bool,
}

impl Stat {
    /// Off by more than `tol` relative L2, NaN/Inf the CPU doesn't have, or a length
    /// mismatch.
    fn bad(&self, tol: f64) -> bool {
        self.len_mismatch || !(self.rel_l2 <= tol) || self.nonfinite_gpu > self.nonfinite_cpu
    }
}

fn stat(gpu: &[f32], cpu: &[f32]) -> Stat {
    let mut s = Stat {
        len_mismatch: gpu.len() != cpu.len(),
        ..Default::default()
    };
    let (mut d2, mut c2) = (0f64, 0f64);
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        s.nonfinite_gpu += usize::from(!g.is_finite());
        s.nonfinite_cpu += usize::from(!c.is_finite());
        let d = if g.is_finite() && c.is_finite() {
            (g as f64 - c as f64).abs()
        } else {
            f64::INFINITY
        };
        if d > s.max_abs || i == 0 {
            (s.max_abs, s.max_idx) = (d, i);
        }
        if d.is_finite() {
            d2 += d * d;
            c2 += c as f64 * c as f64;
        }
    }
    s.rel_l2 = if s.nonfinite_gpu > s.nonfinite_cpu {
        f64::INFINITY
    } else if c2 > 1e-60 {
        (d2 / c2).sqrt()
    } else if d2 > 1e-60 {
        f64::INFINITY
    } else {
        0.0
    };
    s
}

/// The CPU shadow of the fused path (`TANG_GLM5_MHC_DEBUG=1`).
struct Dbg {
    tol: f64,
    /// The last pre's CPU mixes and the device `r` it read: the expected next `r` is
    /// `mhc_post(mixes, y, r_in)`.
    last: Option<(Vec<Mix>, Vec<f32>)>,
    y: Option<Vec<f32>>,
    checks: usize,
    failed: usize,
    first: String,
    worst: (f64, String),
}

impl Dbg {
    fn note(&mut self, what: &str, s: &Stat) {
        self.checks += 1;
        let bad = s.bad(self.tol);
        if s.rel_l2.is_finite() && s.rel_l2 > self.worst.0 {
            self.worst = (s.rel_l2, what.to_string());
        }
        if bad {
            self.failed += 1;
            if self.first.is_empty() {
                self.first = what.to_string();
            }
            eprintln!(
                "mhcdbg FAIL {what}: rel {:.3e} max {:.3e} @{} nonfinite {}/{}",
                s.rel_l2, s.max_abs, s.max_idx, s.nonfinite_gpu, s.nonfinite_cpu
            );
        }
    }
}

/// The CPU post of every token: `r_in` `[t][4][n]` → `[t][4][n]`.
fn cpu_post(mixes: &[Mix], y: &[f32], r_in: &[f32], n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; r_in.len()];
    for (i, m) in mixes.iter().enumerate() {
        let tok = &r_in[i * 4 * n..(i + 1) * 4 * n];
        let rs: Vec<&[f32]> = tok.chunks(n).collect();
        let mut os: Vec<&mut [f32]> = out[i * 4 * n..(i + 1) * 4 * n].chunks_mut(n).collect();
        super::mhc::mhc_post(m, &y[i * n..(i + 1) * n], &rs, &mut os)?;
    }
    Ok(out)
}

/// The CPU pre + weighted RMSNorm (`norm_all`'s `rms_norm_mul`) of every token.
fn cpu_pre(
    w: &MhcWeights,
    r: &[f32],
    nw: &[f32],
    eps: f32,
    n: usize,
) -> Result<(Vec<f32>, Vec<Mix>)> {
    let t = r.len() / (4 * n);
    let mut x = Vec::with_capacity(t * n);
    let mut mixes = Vec::with_capacity(t);
    for i in 0..t {
        let rs: Vec<&[f32]> = r[i * 4 * n..(i + 1) * 4 * n].chunks(n).collect();
        let mut m = super::mhc::mhc_pre(w, &rs)?;
        let s: f64 = m.x.iter().map(|&v| v as f64 * v as f64).sum();
        let inv = 1.0 / ((s / n as f64) as f32 + eps).sqrt();
        for (v, g) in m.x.iter_mut().zip(nw) {
            *v = *v * inv * g;
        }
        x.extend_from_slice(&m.x);
        mixes.push(m);
    }
    Ok((x, mixes))
}

/// The device-resident streams + the fused kernels (one per engine; single device).
pub struct MhcFused {
    module: Module,
    s: Stream,
    n: usize,
    max_t: usize,
    ch: usize,
    norm_eps: f32,
    r: DevBuf,
    y: DevBuf,
    x: DevBuf,
    h: DevBuf,
    post: DevBuf,
    comb: DevBuf,
    part: DevBuf,
    /// Per-token arrival counters of [`mhcf_fold`] (`[max_t]` u32, zero between launches).
    cnt: DevBuf,
    /// `TANG_GLM5_MHC_FOLD=1` (default off): the finish runs in the part kernel's last block.
    fold: bool,
    /// `TANG_GLM5_MHC_EXACT` (default see `new`): 0 = the fast finish (tree reductions,
    /// f32 norm, fast divides), 1 = the bit-exact finish.
    fast: bool,
    /// Per layer `[attn_norm, ffn_norm]`, device + host (the host copy feeds the shadow).
    norm: Vec<[DevBuf; 2]>,
    norm_h: Vec<[Vec<f32>; 2]>,
    /// The window in flight: `t`, and whether a post is waiting on `y`.
    t: usize,
    pending: bool,
    /// Where the pending post's `y` lives: `y.ptr` (host-uploaded by [`Self::post`]) or a
    /// producer's device buffer ([`Self::post_dev`], the resident decode step).
    y_src: u64,
    /// Windows completed (`end` calls): the tests' "the fused path actually ran" check.
    windows: usize,
    /// NaN-fill every output before its kernel (on with the shadow; tests set it).
    poison: bool,
    dbg: Option<Dbg>,
}

impl MhcFused {
    /// `norms[l] = [attn_norm, ffn_norm]` (each `[n]`); `norm_eps` the model's RMS eps.
    /// `Gpu::bind` first.
    pub fn new(
        gpu: &Gpu,
        n: usize,
        max_t: usize,
        norms: Vec<[Vec<f32>; 2]>,
        norm_eps: f32,
    ) -> Result<Self> {
        let ge =
            |what: &'static str| move |e: tang_moe::gpu::Error| anyhow!("mhc fused {what}: {e}");
        if norms.iter().flatten().any(|v| v.len() != n) {
            bail!("mhc fused: norm weights not {n} wide");
        }
        let ch = n.div_ceil(128) * 4;
        if n.max(ch * 25) * 4 > 46 * 1024 {
            bail!("mhc fused: n {n} needs more than 46 KiB of mhcf_finish shared memory");
        }
        let f4 = |k: usize| DevBuf::alloc(k * 4);
        let norm = norms
            .iter()
            .map(|[a, f]| Ok([DevBuf::from_slice(a)?, DevBuf::from_slice(f)?]))
            .collect::<std::result::Result<Vec<_>, tang_moe::gpu::Error>>()
            .map_err(ge("norm upload"))?;
        let dbg = std::env::var("TANG_GLM5_MHC_DEBUG")
            .is_ok_and(|v| v == "1")
            .then(|| Dbg {
                tol: std::env::var("TANG_GLM5_MHC_DEBUG_TOL")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1e-3),
                last: None,
                y: None,
                checks: 0,
                failed: 0,
                first: String::new(),
                worst: (0.0, String::new()),
            });
        Ok(Self {
            module: gpu.module(SRC).map_err(ge("compile"))?,
            s: Stream::blocking().map_err(ge("stream"))?,
            n,
            max_t,
            ch,
            norm_eps,
            r: f4(max_t * 4 * n).map_err(ge("r"))?,
            y: f4(max_t * n).map_err(ge("y"))?,
            x: f4(max_t * n).map_err(ge("x"))?,
            h: f4(max_t * n).map_err(ge("h"))?,
            post: f4(max_t * 4).map_err(ge("post"))?,
            comb: f4(max_t * 16).map_err(ge("comb"))?,
            part: f4(max_t * ch * 25).map_err(ge("part"))?,
            cnt: DevBuf::from_slice(&vec![0u32; max_t]).map_err(ge("cnt"))?,
            fold: std::env::var("TANG_GLM5_MHC_FOLD").map_or(FOLD_DEFAULT, |v| v == "1"),
            fast: std::env::var("TANG_GLM5_MHC_EXACT").map_or(FAST_DEFAULT, |v| v == "0"),
            norm,
            norm_h: norms,
            t: 0,
            pending: false,
            y_src: 0,
            windows: 0,
            poison: dbg.is_some(),
            dbg,
        })
    }

    pub fn max_t(&self) -> usize {
        self.max_t
    }

    /// NaN-poison every kernel output before its launch (see the module doc).
    pub fn set_poison(&mut self, on: bool) {
        self.poison = on;
    }

    /// Fill `buf`'s first `k` f32 with NaN (synchronous: the legacy-stream copy waits for
    /// the blocking stream's earlier kernels and the next launch waits for it).
    fn nan(buf: &DevBuf, k: usize) -> Result<()> {
        buf.write(0, &vec![f32::NAN; k])
            .map_err(|e| anyhow!("mhc fused poison: {e}"))
    }

    /// Windows completed so far.
    pub fn windows(&self) -> usize {
        self.windows
    }

    /// Start a window: upload the embedded streams `r` `[t][4][n]`.
    pub fn begin(&mut self, r: &[f32], t: usize) -> Result<()> {
        if t == 0 || t > self.max_t || r.len() != t * 4 * self.n {
            bail!(
                "mhc fused begin: t {t} (max {}), {} values",
                self.max_t,
                r.len()
            );
        }
        self.r
            .write(0, r)
            .map_err(|e| anyhow!("mhc fused begin: {e}"))?;
        self.t = t;
        self.pending = false;
        if let Some(d) = self.dbg.as_mut() {
            d.last = None;
            d.y = None;
        }
        Ok(())
    }

    /// `mhcf_part` over the window: the pending post (if any) and, with `fn_ != 0`, this
    /// half's partial dots; `h != 0` also writes the head mean.
    fn part(&self, fn_: u64, h: u64) -> Result<()> {
        let f = self
            .module
            .func("mhcf_part")
            .map_err(|e| anyhow!("mhcf_part: {e}"))?;
        let (r, y, post, comb, part) = (
            self.r.ptr,
            self.y_src,
            self.post.ptr,
            self.comb.ptr,
            self.part.ptr,
        );
        let n = self.n as i32;
        let do_post = i32::from(self.pending);
        let do_part = i32::from(fn_ != 0);
        unsafe {
            tang_moe::gpu::launch(
                f,
                (self.ch as u32, self.t as u32, 1),
                ((FB * PW) as u32, 1, 1),
                0,
                &self.s,
                tang_moe::args![r, y, post, comb, fn_, part, h, n, do_post, do_part],
            )
        }
        .map_err(|e| anyhow!("mhcf_part launch: {e}"))
    }

    /// Layer `l`'s half `block` (0 = attention / hc_attn + attn_norm, 1 = FFN / hc_ffn +
    /// ffn_norm): apply the pending post, then pre + norm. Returns the normed block input
    /// `[t][n]` on the host. `cpu_w` is the same block's host weights (shadow only).
    /// The launches of [`Self::pre`]: the pending post + this half's pre + norm, the
    /// normed `x` written to `x_dst` (device, `[t][n]` f32). Async on [`Self::stream`].
    fn pre_launch(&mut self, mh: &MhcGpu, l: usize, block: usize, x_dst: u64) -> Result<()> {
        if block > 1 || l >= self.norm.len() || self.t == 0 {
            bail!("mhc fused pre: layer {l} block {block} t {}", self.t);
        }
        let ((fn_, base, scale), (iters, hc_eps, rms_eps)) = mh.dev_weights(block);
        let (t, n) = (self.t, self.n);
        if self.poison {
            Self::nan(&self.part, t * self.ch * 25)?;
            Self::nan(&self.x, t * n)?;
        }
        let (r, part, nw, x, post, comb) = (
            self.r.ptr,
            self.part.ptr,
            self.norm[l][block].ptr,
            x_dst,
            self.post.ptr,
            self.comb.ptr,
        );
        let (ch, n, it, ne) = (self.ch as i32, self.n as i32, iters as i32, self.norm_eps);
        let fast = i32::from(self.fast);
        let smem = (self.n.max(self.ch * 25) * 4) as u32;
        if self.fold {
            let f = self
                .module
                .func("mhcf_fold")
                .map_err(|e| anyhow!("mhcf_fold: {e}"))?;
            let (y, cnt, do_post) = (self.y_src, self.cnt.ptr, i32::from(self.pending));
            self.pending = false;
            unsafe {
                tang_moe::gpu::launch(
                    f,
                    (self.ch as u32, self.t as u32, 1),
                    (1024, 1, 1),
                    smem,
                    &self.s,
                    tang_moe::args![
                        r, y, post, comb, fn_, part, n, do_post, base, scale, nw, x, it, hc_eps,
                        rms_eps, ne, fast, cnt
                    ],
                )
            }
            .map_err(|e| anyhow!("mhcf_fold launch: {e}"))?;
            return Ok(());
        }
        self.part(fn_, 0)?;
        self.pending = false;
        if self.poison {
            // after the part kernel: it reads the previous half's post/comb (the folded
            // kernel reads and then writes them, so it is not poisoned)
            Self::nan(&self.post, t * 4)?;
            Self::nan(&self.comb, t * 16)?;
        }
        let f = self
            .module
            .func("mhcf_finish")
            .map_err(|e| anyhow!("mhcf_finish: {e}"))?;
        unsafe {
            tang_moe::gpu::launch(
                f,
                (self.t as u32, 1, 1),
                (1024, 1, 1),
                smem,
                &self.s,
                tang_moe::args![
                    r, part, ch, base, scale, nw, x, post, comb, n, it, hc_eps, rms_eps, ne, fast
                ],
            )
        }
        .map_err(|e| anyhow!("mhcf_finish launch: {e}"))?;
        Ok(())
    }

    pub fn pre(
        &mut self,
        mh: &MhcGpu,
        l: usize,
        block: usize,
        cpu_w: Option<&MhcWeights>,
    ) -> Result<Vec<f32>> {
        let x_dst = self.x.ptr;
        self.pre_launch(mh, l, block, x_dst)?;
        let ge = |e: tang_moe::gpu::Error| anyhow!("mhc fused pre: {e}");
        self.s.sync().map_err(ge)?;
        let x: Vec<f32> = self.x.read(0, self.t * self.n).map_err(ge)?;
        if self.dbg.is_some() {
            self.shadow_pre(l, block, cpu_w, &x)?;
        }
        Ok(x)
    }

    fn shadow_pre(
        &mut self,
        l: usize,
        block: usize,
        cpu_w: Option<&MhcWeights>,
        x: &[f32],
    ) -> Result<()> {
        let (t, n) = (self.t, self.n);
        let ge = |e: tang_moe::gpu::Error| anyhow!("mhcdbg read: {e}");
        let r_dev: Vec<f32> = self.r.read(0, t * 4 * n).map_err(ge)?;
        let post: Vec<f32> = self.post.read(0, t * 4).map_err(ge)?;
        let comb: Vec<f32> = self.comb.read(0, t * 16).map_err(ge)?;
        let nw = self.norm_h[l][block].clone();
        let d = self.dbg.as_mut().unwrap();
        let half = if block == 0 { "attn" } else { "ffn" };
        if let (Some((mixes, r_in)), Some(y)) = (d.last.take(), d.y.take()) {
            let want = cpu_post(&mixes, &y, &r_in, n)?;
            d.note(&format!("l{l} post-before-{half} r"), &stat(&r_dev, &want));
        }
        let Some(w) = cpu_w else {
            bail!("mhcdbg: pre without the host weights");
        };
        let (x_cpu, mixes) = cpu_pre(w, &r_dev, &nw, self.norm_eps, n)?;
        d.note(&format!("l{l} pre({half}) x"), &stat(x, &x_cpu));
        let pc: Vec<f32> = mixes.iter().flat_map(|m| m.post).collect();
        let cc: Vec<f32> = mixes
            .iter()
            .flat_map(|m| m.comb.into_iter().flatten())
            .collect();
        d.note(&format!("l{l} pre({half}) post"), &stat(&post, &pc));
        d.note(&format!("l{l} pre({half}) comb"), &stat(&comb, &cc));
        d.last = Some((mixes, r_dev));
        Ok(())
    }

    /// The block output `y` `[t][n]` of the last `pre`'s half: uploaded now, applied by
    /// the next `pre` (or `end`).
    pub fn post(&mut self, y: &[f32]) -> Result<()> {
        if y.len() != self.t * self.n || self.pending {
            bail!(
                "mhc fused post: {} values (t {}), pending {}",
                y.len(),
                self.t,
                self.pending
            );
        }
        self.y
            .write(0, y)
            .map_err(|e| anyhow!("mhc fused post: {e}"))?;
        self.y_src = self.y.ptr;
        self.pending = true;
        if let Some(d) = self.dbg.as_mut() {
            d.y = Some(y.to_vec());
        }
        Ok(())
    }

    /// [`Self::begin`] with an async upload on [`Self::stream`] (the resident step: no
    /// legacy-stream copy, so no implicit wait on the other blocking streams).
    pub fn begin_async(&mut self, r: &[f32]) -> Result<()> {
        if r.is_empty() || r.len() % (4 * self.n) != 0 {
            bail!("mhc fused begin_async: {} values", r.len());
        }
        let t = r.len() / (4 * self.n);
        if t > self.max_t {
            bail!("mhc fused begin_async: t {t} > max {}", self.max_t);
        }
        self.r
            .write_async(0, r, &self.s)
            .map_err(|e| anyhow!("mhc fused begin_async: {e}"))?;
        self.set_window(t)
    }

    /// Set the window length and clear the pending post without touching the device (the
    /// step graph's capture pass; `0` closes the window after a replay).
    pub fn set_window(&mut self, t: usize) -> Result<()> {
        if t > self.max_t || self.dbg.is_some() {
            bail!("mhc fused set_window: t {t}, shadow {}", self.dbg.is_some());
        }
        self.t = t;
        self.pending = false;
        Ok(())
    }

    /// The device stream buffer `r` (`[4][n]` f32 for a 1-row window): the whole-step
    /// graph's device embed gather writes it in place of [`Self::begin_async`]'s upload.
    pub fn r_ptr(&self) -> u64 {
        self.r.ptr
    }

    /// The window's stream (every fused-mHC launch goes here).
    pub fn stream(&self) -> &Stream {
        &self.s
    }

    /// The device-resident [`Self::pre`]: the normed block input is written to `x_dst`
    /// (a consumer's device input buffer) and nothing is read back or synced.
    pub fn pre_dev(&mut self, mh: &MhcGpu, l: usize, block: usize, x_dst: u64) -> Result<()> {
        if self.dbg.is_some() {
            bail!("mhc fused pre_dev: no shadow check on the resident path");
        }
        self.pre_launch(mh, l, block, x_dst)
    }

    /// The device-resident [`Self::post`]: `y_dev` (`[t][n]` f32, a producer's output
    /// buffer, written on [`Self::stream`] before the next pre) is applied by the next
    /// `pre_dev` / `end_dev`. Nothing is copied.
    pub fn post_dev(&mut self, y_dev: u64) -> Result<()> {
        if self.pending || self.t == 0 {
            bail!("mhc fused post_dev: pending {}, t {}", self.pending, self.t);
        }
        self.y_src = y_dev;
        self.pending = true;
        Ok(())
    }

    /// The device-resident [`Self::end`]: the last post + the head mean into the device
    /// `h` buffer; returns its pointer (`[t][n]` f32, before `output_norm`). No sync.
    pub fn end_dev(&mut self) -> Result<u64> {
        if !self.pending || self.dbg.is_some() {
            bail!(
                "mhc fused end_dev: pending {}, shadow {}",
                self.pending,
                self.dbg.is_some()
            );
        }
        self.part(0, self.h.ptr)?;
        self.pending = false;
        self.t = 0;
        self.windows += 1;
        Ok(self.h.ptr)
    }

    /// End the window: the last post + the head's unweighted stream mean. Returns `h`
    /// `[t][n]` (before `output_norm`).
    pub fn end(&mut self) -> Result<Vec<f32>> {
        if !self.pending {
            bail!("mhc fused end: no pending post");
        }
        if self.poison {
            Self::nan(&self.h, self.t * self.n)?;
        }
        self.part(0, self.h.ptr)?;
        self.pending = false;
        let ge = |e: tang_moe::gpu::Error| anyhow!("mhc fused end: {e}");
        self.s.sync().map_err(ge)?;
        let (t, n) = (self.t, self.n);
        let h: Vec<f32> = self.h.read(0, t * n).map_err(ge)?;
        if let Some(d) = self.dbg.as_mut() {
            if let (Some((mixes, r_in)), Some(y)) = (d.last.take(), d.y.take()) {
                let want = cpu_post(&mixes, &y, &r_in, n)?;
                let r_dev: Vec<f32> = self.r.read(0, t * 4 * n).map_err(ge)?;
                d.note("final post r", &stat(&r_dev, &want));
                let hw: Vec<f32> = (0..t * n)
                    .map(|i| {
                        let (ti, j) = (i / n, i % n);
                        let mut a = 0f32;
                        for c in 0..4 {
                            a += want[(ti * 4 + c) * n + j];
                        }
                        a / 4.0
                    })
                    .collect();
                d.note("head mean h", &stat(&h, &hw));
            }
            eprintln!(
                "mhcdbg window t={t}: {} checks so far, {} failed, worst rel {:.3e} ({})",
                d.checks, d.failed, d.worst.0, d.worst.1
            );
        }
        self.t = 0;
        self.windows += 1;
        Ok(h)
    }
}

impl Drop for MhcFused {
    fn drop(&mut self) {
        if let Some(d) = &self.dbg {
            eprintln!(
                "mhcdbg SUMMARY: {} checks, {} failed{}; worst rel {:.3e} ({})",
                d.checks,
                d.failed,
                if d.first.is_empty() {
                    String::new()
                } else {
                    format!("; first: {}", d.first)
                },
                d.worst.0,
                d.worst.1
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::glm5::mhc::MhcHparams;

    fn rnd(st: &mut u64) -> f32 {
        *st ^= *st >> 12;
        *st ^= *st << 25;
        *st ^= *st >> 27;
        let m = (st.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32;
        m / (1u32 << 23) as f32 - 1.0
    }

    fn gpu() -> Option<Gpu> {
        let g = Gpu::for_test();
        if g.is_none() && std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") {
            panic!("TANG_REQUIRE_CUDA=1 but no CUDA device");
        }
        g
    }

    /// Two layers' chain on the device vs the CPU reference: pre(attn) → post → pre(ffn)
    /// → post → pre(attn, layer 1) → post → end, distinct weights per block (the block-
    /// index bug class: hc_attn vs hc_ffn AND attn_norm vs ffn_norm), real-model n = 4096
    /// and a tiny n not a multiple of the 128-feature chunk. Then a micro-bench of one
    /// fused pre against MhcGpu::pre at t = 1 and 64.
    #[test]
    fn fused_chain_matches_cpu() {
        let _env = crate::flash::glm5::env_read();
        let Some(g) = gpu() else {
            eprintln!("no CUDA device: skipping");
            return;
        };
        g.bind().unwrap();
        for (n, ts) in [(4096usize, &[1usize, 5, 64][..]), (96, &[3][..])] {
            let mut st = 0x1234_5678_9ABC_DEF1u64 ^ n as u64;
            // fn scaled so the mixes land at O(1) (unscaled, a 16k-long ±1 dot saturates
            // every sigmoid/softmax and the comparison degenerates to exact 0/1 weights)
            let fs = 2.0 / (4.0 * n as f32).sqrt();
            let mk = |st: &mut u64| {
                MhcWeights::from_slices(
                    (0..24 * 4 * n).map(|_| rnd(st) * fs).collect(),
                    (0..24).map(|_| rnd(st) * 0.5).collect(),
                    vec![0.5 + rnd(st) * 0.1, 0.5, 0.5 + rnd(st) * 0.1],
                    n,
                    &MhcHparams::REAL,
                )
                .unwrap()
            };
            let ws = [[mk(&mut st), mk(&mut st)], [mk(&mut st), mk(&mut st)]];
            let norms: Vec<[Vec<f32>; 2]> = (0..2)
                .map(|_| {
                    [
                        (0..n).map(|_| 1.0 + rnd(&mut st)).collect(),
                        (0..n).map(|_| 1.0 + rnd(&mut st)).collect(),
                    ]
                })
                .collect();
            let mhs: Vec<MhcGpu> = ws
                .iter()
                .map(|[a, f]| MhcGpu::new(&g, [a, f], 64).unwrap())
                .collect();
            let mut fz = MhcFused::new(&g, n, 64, norms.clone(), 1e-5).unwrap();
            fz.set_poison(true);
            for (fold, fast, &t) in [(true, false), (false, false), (true, true), (false, true)]
                .into_iter()
                .flat_map(|(a, b)| ts.iter().map(move |t| (a, b, t)))
            {
                (fz.fold, fz.fast) = (fold, fast);
                eprintln!("fused config fold={fold} fast={fast}");
                let r0: Vec<f32> = (0..t * 4 * n).map(|_| rnd(&mut st)).collect();
                let ys: Vec<Vec<f32>> = (0..3)
                    .map(|_| (0..t * n).map(|_| rnd(&mut st) * 0.5).collect())
                    .collect();
                // CPU chain
                let mut r = r0.clone();
                let mut cx = Vec::new();
                for (k, (l, b)) in [(0usize, 0usize), (0, 1), (1, 0)].into_iter().enumerate() {
                    let (x, mixes) = cpu_pre(&ws[l][b], &r, &norms[l][b], 1e-5, n).unwrap();
                    cx.push(x);
                    r = cpu_post(&mixes, &ys[k], &r, n).unwrap();
                }
                let h_cpu: Vec<f32> = (0..t * n)
                    .map(|i| {
                        let (ti, j) = (i / n, i % n);
                        (0..4).fold(0f32, |a, c| a + r[(ti * 4 + c) * n + j]) / 4.0
                    })
                    .collect();
                // device chain
                fz.begin(&r0, t).unwrap();
                let mut worst = 0f64;
                for (k, (l, b)) in [(0usize, 0usize), (0, 1), (1, 0)].into_iter().enumerate() {
                    let x = fz.pre(&mhs[l], l, b, Some(&ws[l][b])).unwrap();
                    let s = stat(&x, &cx[k]);
                    eprintln!(
                        "fused n={n} t={t} step {k} (l{l} b{b}): x rel {:.3e} max {:.3e}",
                        s.rel_l2, s.max_abs
                    );
                    assert!(
                        !s.bad(1e-4),
                        "fused pre step {k} (layer {l} block {b}) diverges: {s:?}"
                    );
                    worst = worst.max(s.rel_l2);
                    fz.post(&ys[k]).unwrap();
                }
                let h = fz.end().unwrap();
                let s = stat(&h, &h_cpu);
                eprintln!(
                    "fused n={n} t={t} head: rel {:.3e} max {:.3e}",
                    s.rel_l2, s.max_abs
                );
                assert!(!s.bad(1e-4), "fused head diverges: {s:?}");
                // the wrong block's weights must be visibly wrong (guards a constant index)
                fz.begin(&r0, t).unwrap();
                let x1 = fz.pre(&mhs[0], 0, 1, Some(&ws[0][1])).unwrap();
                let s = stat(&x1, &cx[0]);
                assert!(
                    s.rel_l2 > 1e-2,
                    "block 1 reproduced block 0's x (rel {:.3e})",
                    s.rel_l2
                );
                fz.post(&ys[0]).unwrap();
                fz.end().unwrap();
            }
            // micro-bench (real shape only): one fused pre vs MhcGpu::pre (host r upload +
            // one warp per token + x/post/comb read-back), wall clock, median of 30
            if n != 4096 {
                continue;
            }
            fz.set_poison(false);
            (fz.fold, fz.fast) = (FOLD_DEFAULT, FAST_DEFAULT);
            for t in [1usize, 64] {
                let r0: Vec<f32> = (0..t * 4 * n).map(|_| rnd(&mut st)).collect();
                let y: Vec<f32> = (0..t * n).map(|_| rnd(&mut st)).collect();
                let med = |f: &mut dyn FnMut()| {
                    let mut v: Vec<f64> = (0..30)
                        .map(|_| {
                            let t0 = std::time::Instant::now();
                            f();
                            t0.elapsed().as_secs_f64() * 1e6
                        })
                        .collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[15]
                };
                let old = med(&mut || {
                    let _ = mhs[0].pre(0, &r0, t).unwrap();
                });
                fz.begin(&r0, t).unwrap();
                let _ = fz.pre(&mhs[0], 0, 0, Some(&ws[0][0])).unwrap();
                let new = med(&mut || {
                    fz.post(&y).unwrap();
                    let _ = fz.pre(&mhs[0], 0, 0, Some(&ws[0][0])).unwrap();
                });
                fz.post(&y).unwrap();
                fz.end().unwrap();
                eprintln!("mhc fused bench n={n} t={t}: MhcGpu::pre {old:.0} µs, fused post+pre(+norm) {new:.0} µs");
            }
        }
    }
}
