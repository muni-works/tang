//! Top-k / top-p / presence-penalty sampling for the Flash engine's head, in the window graph.
//!
//! Per row (a window position), deterministically from that row's logits: subtract the presence
//! penalty from every token already in the output (the accepted output before the window, and
//! the window's own output tokens up to this row), keep the `top_k` best (ties to the lower id,
//! at most [`MAX_K`]), then the smallest prefix whose softmax mass at the temperature reaches
//! `top_p`, and take the Gumbel-max over it with the same `Philox(token, position, seed)` noise
//! as `fe_argmax1`. Greedy (temperature 0) is the first of the list. A sample is a function of
//! (seed, position, logits, output so far) only, so speculation stays exact: a draft is kept
//! only when it equals the sample, and then the window's earlier output tokens are the real ones.

use anyhow::{anyhow, ensure, Result};
use tang_compute::cuda::CudaBuffer;
use tang_compute::flash::MAX_T;
use tang_compute::{ComputeDevice, CudaComputeDevice};
use tang_moe::gpu::{self, Gpu, Module, Stream};

type B = CudaBuffer;

/// The most candidates kept per row.
pub const MAX_K: usize = 64;
/// Slices per row in the first pass.
const SLICES: usize = 64;

pub const SRC: &str = r#"
__device__ __forceinline__ unsigned philox(unsigned c0, unsigned c1, unsigned k0) {
    unsigned x0 = c0, x1 = c1, x2 = 0, x3 = 0, k1 = 0;
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        const unsigned lo0 = 0xD2511F53u * x0, hi0 = __umulhi(0xD2511F53u, x0);
        const unsigned lo1 = 0xCD9E8D57u * x2, hi1 = __umulhi(0xCD9E8D57u, x2);
        const unsigned y0 = hi1 ^ x1 ^ k0, y2 = hi0 ^ x3 ^ k1;
        x0 = y0; x1 = lo1; x2 = y2; x3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return x0;
}
// (v, i) ranks before (ov, oi): larger value, then lower id.
__device__ __forceinline__ bool before(float v, int i, float ov, int oi) { return v > ov || (v == ov && i < oi); }
__device__ __forceinline__ void best(float& v, int& i, float ov, int oi) { if (before(ov, oi, v, i)) { v = ov; i = oi; } }

// params: [0] top_k, [1] top_p (f32 bits), [2] presence penalty (f32 bits), [3] rows whose
// input is an output token start at this row (>= t: none), [4..12] the window's input tokens.
// seen[v]: 1 if token v is in the accepted output before the window.
__device__ __forceinline__ float pen_logit(const float* l, int i, int row, const unsigned* params, const unsigned* seen) {
    const float pen = __uint_as_float(params[2]);
    float x = l[i];
    if (pen != 0.f) {
        bool s = seen[i] != 0u;
        for (int r = (int)params[3]; r <= row && !s; r++) s = params[4 + r] == (unsigned)i;
        if (s) x -= pen;
    }
    return x;
}

// Pass 1, grid (SLICES, t), block 1024: each slice's k best as (value, id) pairs, best first.
extern "C" __global__ void fs_topk1(const float* logits, int n, float* part, const unsigned* params, const unsigned* seen) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int row = blockIdx.y, k = min((int)params[0], 64);
    const float* l = logits + (size_t)row * n;
    const int per = (n + gridDim.x - 1) / gridDim.x, lo = blockIdx.x * per, hi = min(n, lo + per);
    float* out = part + ((size_t)row * gridDim.x + blockIdx.x) * 64 * 2;
    float pv = __int_as_float(0x7f800000);
    int pi = -1;
    for (int j = 0; j < k; j++) {
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const float x = pen_logit(l, i, row, params, seen);
            if (before(pv, pi, x, i)) best(v, idx, x, i);
        }
        for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
        const int w = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) { bv[w] = v; bi[w] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < (int)(blockDim.x >> 5); q++) best(v, idx, bv[q], bi[q]);
            bv[0] = v; bi[0] = idx;
            out[2 * j] = v;
            out[2 * j + 1] = __int_as_float(idx);
        }
        __syncthreads();
        pv = bv[0]; pi = bi[0];
        __syncthreads();
    }
}

// Pass 2, grid (t), block 32: merge the slices' lists into the row's k best, cut at top_p, and
// take the Gumbel-max (or the first, greedy).
extern "C" __global__ void fs_topk2(const float* part, int nslices, const unsigned* params, const unsigned* ctl, unsigned* ids) {
    __shared__ float tv[64];
    __shared__ int ti[64];
    const int row = blockIdx.x, k = min((int)params[0], 64);
    const float* p = part + (size_t)row * nslices * 64 * 2;
    const int nc = nslices * k;
    float pv = __int_as_float(0x7f800000);
    int pi = -1;
    for (int j = 0; j < k; j++) {
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int c = threadIdx.x; c < nc; c += 32) {
            const int s = c / k, q = c % k;
            const float x = p[(s * 64 + q) * 2];
            const int xi = __float_as_int(p[(s * 64 + q) * 2 + 1]);
            if (xi != 0x7fffffff && before(pv, pi, x, xi)) best(v, idx, x, xi);
        }
        for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
        if (threadIdx.x == 0) { tv[j] = v; ti[j] = idx; }
        pv = v; pi = idx;
    }
    __syncwarp();
    if (threadIdx.x != 0) return;
    const float temp = __uint_as_float(ctl[4]);
    int pick = ti[0];
    if (temp > 0.f) {
        const float it = 1.0f / temp, top_p = __uint_as_float(params[1]);
        int keep = 0;
        while (keep < k && ti[keep] != 0x7fffffff) keep++;
        float sum = 0.f;
        for (int j = 0; j < keep; j++) sum += expf((tv[j] - tv[0]) * it);
        float cum = 0.f;
        int cut = keep;
        for (int j = 0; j < keep; j++) {
            cum += expf((tv[j] - tv[0]) * it);
            if (cum >= top_p * sum) { cut = j + 1; break; }
        }
        const unsigned pos = ctl[0] + row, seed = ctl[3];
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int j = 0; j < cut; j++) {
            const float u = ((float)(philox((unsigned)ti[j], pos, seed) >> 8) + 0.5f) * 5.9604644775390625e-8f;
            best(v, idx, tv[j] * it - logf(-logf(u)), ti[j]);
        }
        pick = idx;
    }
    ids[row] = (unsigned)pick;
}
"#;

pub struct Sampler {
    _m: Module,
    k1: cudarc::driver::sys::CUfunction,
    k2: cudarc::driver::sys::CUfunction,
    params: B,
    seen: B,
    part: B,
    pub top_k: usize,
    pub top_p: f32,
    pub presence: f32,
    /// The first output position; tokens from here on are output.
    gen_start: usize,
    /// Output tokens flagged in `seen` (to clear), and positions `gen_start..flagged` done.
    flagged: Vec<u32>,
    flagged_upto: usize,
    vocab: usize,
}

fn htod(addr: u64, bytes: &[u8]) -> Result<()> {
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyHtoD_v2(addr, bytes.as_ptr() as *const _, bytes.len())
    };
    ensure!(
        r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "cuMemcpyHtoD: {r:?}"
    );
    Ok(())
}

impl Sampler {
    pub fn new(gpu: &Gpu, dev: &CudaComputeDevice, vocab: usize) -> Result<Self> {
        let m = gpu.module(SRC).map_err(|e| anyhow!("{e}"))?;
        let k1 = m.func("fs_topk1").map_err(|e| anyhow!("{e}"))?;
        let k2 = m.func("fs_topk2").map_err(|e| anyhow!("{e}"))?;
        let mut seen = dev.alloc_f32(vocab);
        dev.zero_buffer(&mut seen);
        dev.sync();
        Ok(Sampler {
            _m: m,
            k1,
            k2,
            params: dev.alloc_f32(16),
            seen,
            part: dev.alloc_f32(MAX_T * SLICES * MAX_K * 2),
            top_k: 1,
            top_p: 1.0,
            presence: 0.0,
            gen_start: usize::MAX,
            flagged: Vec::new(),
            flagged_upto: 0,
            vocab,
        })
    }

    /// A new generation: output starts at position `gen_start` (prefill windows before it
    /// see no penalty).
    pub fn begin(&mut self, dev: &CudaComputeDevice, gen_start: usize) -> Result<()> {
        let base = dev.buffer_addr(&self.seen);
        for &tok in &self.flagged {
            htod(base + 4 * tok as u64, &0u32.to_le_bytes())?;
        }
        self.flagged.clear();
        self.gen_start = gen_start;
        self.flagged_upto = gen_start;
        Ok(())
    }

    /// Before a window over `tokens[pos0..pos0 + t]`: flag the output accepted before it and
    /// write the parameters and the window's tokens.
    pub fn stage(
        &mut self,
        dev: &CudaComputeDevice,
        tokens: &[u32],
        pos0: usize,
        t: usize,
    ) -> Result<()> {
        let base = dev.buffer_addr(&self.seen);
        if self.gen_start != usize::MAX
            && pos0 < self.flagged_upto
            && self.flagged_upto > self.gen_start
        {
            // The sequence went back past flagged output (a new prompt without `begin`): no
            // output history until the next `begin`.
            self.begin(dev, usize::MAX)?;
        }
        if self.gen_start != usize::MAX {
            while self.flagged_upto < pos0 {
                let tok = tokens[self.flagged_upto];
                if (tok as usize) < self.vocab && !self.flagged.contains(&tok) {
                    htod(base + 4 * tok as u64, &1u32.to_le_bytes())?;
                    self.flagged.push(tok);
                }
                self.flagged_upto += 1;
            }
        }
        let mut w = [0u32; 16];
        w[0] = self.top_k.clamp(1, MAX_K) as u32;
        w[1] = self.top_p.to_bits();
        w[2] = self.presence.to_bits();
        w[3] = if self.gen_start == usize::MAX {
            t as u32
        } else {
            self.gen_start.saturating_sub(pos0).min(t) as u32
        };
        for i in 0..t {
            w[4 + i] = tokens[pos0 + i];
        }
        let b: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        htod(dev.buffer_addr(&self.params), &b)
    }

    /// The two passes on `stream` (graph nodes): logits `[t][vocab]` to `ids[t]`.
    pub fn enqueue(
        &self,
        dev: &CudaComputeDevice,
        stream: &Stream,
        logits: &B,
        ctl: &B,
        ids: &B,
        t: usize,
    ) {
        let (lg, n, part, prm, seen) = (
            dev.buffer_addr(logits),
            self.vocab as i32,
            dev.buffer_addr(&self.part),
            dev.buffer_addr(&self.params),
            dev.buffer_addr(&self.seen),
        );
        let (ctlp, idp, ns) = (dev.buffer_addr(ctl), dev.buffer_addr(ids), SLICES as i32);
        unsafe {
            gpu::launch(
                self.k1,
                (SLICES as u32, t as u32, 1),
                (1024, 1, 1),
                0,
                stream,
                tang_moe::args![lg, n, part, prm, seen],
            )
            .expect("launch fs_topk1");
            gpu::launch(
                self.k2,
                (t as u32, 1, 1),
                (32, 1, 1),
                0,
                stream,
                tang_moe::args![part, ns, prm, ctlp, idp],
            )
            .expect("launch fs_topk2");
        }
    }
}

// ---- per-row sampling for batched windows (sequence slots)

/// Words per row in [`RowSampler`]'s parameter table.
pub const ROW_WORDS: usize = 8;
/// Slices per row in the row sampler's first pass.
const ROW_SLICES: usize = 16;

/// One sequence's sampling settings in a batched window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowSampling {
    /// <= 0: greedy (the window's argmax stands; no penalty).
    pub temperature: f32,
    /// 0: no top-k cut (and with `top_p` 1: Gumbel-max over the whole vocabulary); else the
    /// `min(top_k, MAX_K)` best.
    pub top_k: usize,
    /// < 1: the smallest prefix of the (at most `MAX_K`) best whose mass reaches it.
    pub top_p: f32,
    pub seed: u32,
    /// Subtracted from the logit of every token this slot has output since `set_slot_sampling`.
    pub presence: f32,
}

impl RowSampling {
    pub fn greedy() -> Self {
        RowSampling {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            presence: 0.0,
        }
    }
    pub fn sampled(&self) -> bool {
        self.temperature > 0.0
    }
    /// The row's parameter words: active, slot, position, temperature, top-k, top-p, seed,
    /// presence.
    pub fn words(&self, slot: usize, pos: usize) -> [u32; ROW_WORDS] {
        if !self.sampled() {
            return [0; ROW_WORDS];
        }
        let full = self.top_k == 0 && !(self.top_p > 0.0 && self.top_p < 1.0);
        let k = if full {
            0
        } else if self.top_k == 0 {
            MAX_K
        } else {
            self.top_k.min(MAX_K)
        };
        let p = if self.top_p > 0.0 && self.top_p <= 1.0 {
            self.top_p
        } else {
            1.0
        };
        [
            1,
            slot as u32,
            pos as u32,
            self.temperature.to_bits(),
            k as u32,
            p.to_bits(),
            self.seed,
            self.presence.to_bits(),
        ]
    }
}

pub const ROW_SRC: &str = r#"
__device__ __forceinline__ unsigned philox(unsigned c0, unsigned c1, unsigned k0) {
    unsigned x0 = c0, x1 = c1, x2 = 0, x3 = 0, k1 = 0;
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        const unsigned lo0 = 0xD2511F53u * x0, hi0 = __umulhi(0xD2511F53u, x0);
        const unsigned lo1 = 0xCD9E8D57u * x2, hi1 = __umulhi(0xCD9E8D57u, x2);
        const unsigned y0 = hi1 ^ x1 ^ k0, y2 = hi0 ^ x3 ^ k1;
        x0 = y0; x1 = lo1; x2 = y2; x3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return x0;
}
__device__ __forceinline__ bool before(float v, int i, float ov, int oi) { return v > ov || (v == ov && i < oi); }
__device__ __forceinline__ void best(float& v, int& i, float ov, int oi) { if (before(ov, oi, v, i)) { v = ov; i = oi; } }
__device__ __forceinline__ float gumbel(int i, unsigned pos, unsigned seed) {
    const float u = ((float)(philox((unsigned)i, pos, seed) >> 8) + 0.5f) * 5.9604644775390625e-8f;
    return -logf(-logf(u));
}
// p: the row's words (active, slot, pos, temp, k, top_p, seed, presence); seen: a bitmap of
// `words` u32 per slot.
__device__ __forceinline__ float pen(const float* l, int i, const unsigned* p, const unsigned* seen, int words) {
    const float pe = __uint_as_float(p[7]);
    float x = l[i];
    if (pe != 0.f && (seen[(size_t)p[1] * words + (i >> 5)] >> (i & 31)) & 1u) x -= pe;
    return x;
}

// Pass 1, grid (SLICES, t), block 512. k == 0: each slice's Gumbel-max (score, id) at the
// temperature; else its k best penalized logits, best first.
extern "C" __global__ void rs_pass1(const float* logits, int n, float* part, const unsigned* prm, const unsigned* seen, int words) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int row = blockIdx.y;
    const unsigned* p = prm + row * 8;
    if (p[0] == 0u) return;
    const int k = min((int)p[4], 64);
    const float* l = logits + (size_t)row * n;
    const int per = (n + gridDim.x - 1) / gridDim.x, lo = blockIdx.x * per, hi = min(n, lo + per);
    float* out = part + ((size_t)row * gridDim.x + blockIdx.x) * 64 * 2;
    const int nw = blockDim.x >> 5, w = threadIdx.x >> 5;
    if (k == 0) {
        const float it = 1.0f / __uint_as_float(p[3]);
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) best(v, idx, pen(l, i, p, seen, words) * it + gumbel(i, p[2], p[6]), i);
        for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
        if ((threadIdx.x & 31) == 0) { bv[w] = v; bi[w] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < nw; q++) best(v, idx, bv[q], bi[q]);
            out[0] = v;
            out[1] = __int_as_float(idx);
        }
        return;
    }
    float pv = __int_as_float(0x7f800000);
    int pi = -1;
    for (int j = 0; j < k; j++) {
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const float x = pen(l, i, p, seen, words);
            if (before(pv, pi, x, i)) best(v, idx, x, i);
        }
        for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
        if ((threadIdx.x & 31) == 0) { bv[w] = v; bi[w] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < nw; q++) best(v, idx, bv[q], bi[q]);
            bv[0] = v; bi[0] = idx;
            out[2 * j] = v;
            out[2 * j + 1] = __int_as_float(idx);
        }
        __syncthreads();
        pv = bv[0]; pi = bi[0];
        __syncthreads();
    }
}

// Pass 2, grid (t), block 32: merge, cut at top_p, Gumbel-max over the cut; write ids[row] and
// flag the pick in the slot's output bitmap. Inactive rows keep the argmax already in ids.
extern "C" __global__ void rs_pass2(const float* part, int nslices, const unsigned* prm, unsigned* seen, int words, unsigned* ids) {
    __shared__ float tv[64];
    __shared__ int ti[64];
    const int row = blockIdx.x;
    const unsigned* p = prm + row * 8;
    if (p[0] == 0u) return;
    const int k = min((int)p[4], 64);
    const float* pp = part + (size_t)row * nslices * 64 * 2;
    int pick;
    if (k == 0) {
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int s = threadIdx.x; s < nslices; s += 32) best(v, idx, pp[s * 128], __float_as_int(pp[s * 128 + 1]));
        for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
        pick = idx;
    } else {
        const int nc = nslices * k;
        float pv = __int_as_float(0x7f800000);
        int pi = -1;
        for (int j = 0; j < k; j++) {
            float v = __int_as_float(0xff800000);
            int idx = 0x7fffffff;
            for (int c = threadIdx.x; c < nc; c += 32) {
                const int s = c / k, q = c % k;
                const float x = pp[(s * 64 + q) * 2];
                const int xi = __float_as_int(pp[(s * 64 + q) * 2 + 1]);
                if (xi != 0x7fffffff && before(pv, pi, x, xi)) best(v, idx, x, xi);
            }
            for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
            if (threadIdx.x == 0) { tv[j] = v; ti[j] = idx; }
            pv = v; pi = idx;
        }
        __syncwarp();
        if (threadIdx.x != 0) return;
        const float it = 1.0f / __uint_as_float(p[3]), top_p = __uint_as_float(p[5]);
        int keep = 0;
        while (keep < k && ti[keep] != 0x7fffffff) keep++;
        float sum = 0.f;
        for (int j = 0; j < keep; j++) sum += expf((tv[j] - tv[0]) * it);
        float cum = 0.f;
        int cut = keep;
        for (int j = 0; j < keep; j++) {
            cum += expf((tv[j] - tv[0]) * it);
            if (cum >= top_p * sum) { cut = j + 1; break; }
        }
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int j = 0; j < cut; j++) best(v, idx, tv[j] * it + gumbel(ti[j], p[2], p[6]), ti[j]);
        pick = idx;
    }
    if (threadIdx.x != 0) return;
    ids[row] = (unsigned)pick;
    atomicOr(&seen[(size_t)p[1] * words + (pick >> 5)], 1u << (pick & 31));
}
"#;

/// Per-row sampling for batched windows: each row (a sequence slot's position) its own
/// temperature / top-k / top-p / seed and a presence penalty over that slot's own output.
/// Runs after the window's argmax and overwrites only the sampled rows' ids, so greedy rows
/// are bitwise the argmax. A row's pick is a function of (logits, seed, position, the slot's
/// output so far) only: a sequence samples the same alone as in any batch.
pub struct RowSampler {
    _m: Module,
    k1: cudarc::driver::sys::CUfunction,
    k2: cudarc::driver::sys::CUfunction,
    params: B,
    seen: B,
    part: B,
    words: usize,
    rows: usize,
    vocab: usize,
}

impl RowSampler {
    pub fn new(
        gpu: &Gpu,
        dev: &CudaComputeDevice,
        vocab: usize,
        rows: usize,
        slots: usize,
    ) -> Result<Self> {
        let m = gpu.module(ROW_SRC).map_err(|e| anyhow!("{e}"))?;
        let k1 = m.func("rs_pass1").map_err(|e| anyhow!("{e}"))?;
        let k2 = m.func("rs_pass2").map_err(|e| anyhow!("{e}"))?;
        let words = vocab.div_ceil(32);
        let mut seen = dev.alloc_f32(slots * words);
        dev.zero_buffer(&mut seen);
        let mut params = dev.alloc_f32(rows * ROW_WORDS);
        dev.zero_buffer(&mut params);
        dev.sync();
        Ok(RowSampler {
            _m: m,
            k1,
            k2,
            params,
            seen,
            part: dev.alloc_f32(rows * ROW_SLICES * MAX_K * 2),
            words,
            rows,
            vocab,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Forget slot `slot`'s output (a new request in it).
    pub fn clear_slot(&self, dev: &CudaComputeDevice, slot: usize) -> Result<()> {
        let addr = dev.buffer_addr(&self.seen) + (slot * self.words * 4) as u64;
        let r = unsafe { cudarc::driver::sys::cuMemsetD32_v2(addr, 0, self.words) };
        ensure!(
            r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "cuMemsetD32: {r:?}"
        );
        Ok(())
    }

    /// The window's per-row words (`ROW_WORDS` each, `t` rows); the device must be idle.
    pub fn stage(&self, dev: &CudaComputeDevice, words: &[u32]) -> Result<()> {
        ensure!(
            words.len() <= self.rows * ROW_WORDS,
            "{} row words",
            words.len()
        );
        let b: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
        htod(dev.buffer_addr(&self.params), &b)
    }

    /// Both passes on `stream` (graph nodes) over logits `[t][vocab]`, into `ids[t]`.
    pub fn enqueue(&self, dev: &CudaComputeDevice, stream: &Stream, logits: &B, ids: &B, t: usize) {
        let (lg, n, part, prm, seen, w) = (
            dev.buffer_addr(logits),
            self.vocab as i32,
            dev.buffer_addr(&self.part),
            dev.buffer_addr(&self.params),
            dev.buffer_addr(&self.seen),
            self.words as i32,
        );
        let (idp, ns) = (dev.buffer_addr(ids), ROW_SLICES as i32);
        unsafe {
            gpu::launch(
                self.k1,
                (ROW_SLICES as u32, t as u32, 1),
                (512, 1, 1),
                0,
                stream,
                tang_moe::args![lg, n, part, prm, seen, w],
            )
            .expect("launch rs_pass1");
            gpu::launch(
                self.k2,
                (t as u32, 1, 1),
                (32, 1, 1),
                0,
                stream,
                tang_moe::args![part, ns, prm, seen, w, idp],
            )
            .expect("launch rs_pass2");
        }
    }
}
