//! Sub-4-bit weight formats, emulated: per-group affine (scalar) codes and
//! per-matrix k-means codebooks (vector quantization), each optionally
//! behind a randomized block-Hadamard incoherence rotation.
//!
//! These are not [`super::precision::Format`]s: a codebook is data, not a
//! `Copy` tag, and the spike that measured them (bench/results/lowbit) only
//! needed the emulation — quantize, then dequantize back to f64 — to price
//! accuracy. Everything here is the reference the bench's torch harness
//! (bench/lowbit/lowbit.py) mirrors.

/// A low-bit weight format for an `[out, in]` matrix (row-major, groups run
/// along the input dim).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LowBit {
    /// `bits`-bit codes, one f16 scale and one f16 minimum per `group`.
    Affine { bits: u32, group: usize },
    /// One f16 RMS scale per `group`, then `dim`-element vectors coded as
    /// indices into one `k`-entry codebook for the whole matrix.
    Vq { dim: usize, k: usize, group: usize },
}

impl LowBit {
    /// Stored bits per weight, scales included, codebook amortized over
    /// `n` weights (f16 entries).
    pub fn bits_per_weight(self, n: usize) -> f64 {
        match self {
            LowBit::Affine { bits, group } => bits as f64 + 32.0 / group as f64,
            LowBit::Vq { dim, k, group } => {
                (k as f64).log2() / dim as f64
                    + 16.0 / group as f64
                    + (16 * k * dim) as f64 / n as f64
            }
        }
    }

    /// Table lookups per weight at dequant: a VQ code is one lookup that
    /// yields `dim` weights; affine needs none (a multiply-add).
    pub fn lookups_per_weight(self) -> f64 {
        match self {
            LowBit::Affine { .. } => 0.0,
            LowBit::Vq { dim, .. } => 1.0 / dim as f64,
        }
    }

    /// Quantize then dequantize `w` (`rows × cols`) in place.
    pub fn emulate(self, w: &mut [f64], cols: usize) {
        match self {
            LowBit::Affine { bits, group } => affine(w, bits, group),
            LowBit::Vq { dim, k, group } => vq(w, cols, dim, k, group),
        }
    }
}

fn f16(x: f64) -> f64 {
    super::precision::f16_from_bits(super::precision::f16_bits(x))
}

fn affine(w: &mut [f64], bits: u32, group: usize) {
    let qmax = ((1u32 << bits) - 1) as f64;
    for g in w.chunks_mut(group) {
        let lo = g.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let s = f16(((hi - lo) / qmax).max(1e-8));
        let lo = f16(lo);
        for x in g.iter_mut() {
            *x = ((*x - lo) / s).round().clamp(0.0, qmax) * s + lo;
        }
    }
}

fn nearest(v: &[f64], book: &[f64], dim: usize) -> usize {
    let d2 = |c: usize| {
        (0..dim)
            .map(|j| (v[j] - book[c * dim + j]).powi(2))
            .sum::<f64>()
    };
    (0..book.len() / dim)
        .min_by(|&a, &b| d2(a).total_cmp(&d2(b)))
        .unwrap()
}

/// Deterministic k-means (Lloyd) over `dim`-vectors; returns `k × dim`.
pub fn kmeans(v: &[f64], dim: usize, k: usize, iters: usize) -> Vec<f64> {
    let n = v.len() / dim;
    // Spread initial centers through the data (offset so a period that
    // divides n/k does not seed duplicates).
    let mut book: Vec<f64> = (0..k)
        .flat_map(|c| v[((c * n / k + c) % n) * dim..][..dim].to_vec())
        .collect();
    for _ in 0..iters {
        let mut sum = vec![0.0; k * dim];
        let mut cnt = vec![0usize; k];
        for p in v.chunks(dim) {
            let c = nearest(p, &book, dim);
            cnt[c] += 1;
            (0..dim).for_each(|j| sum[c * dim + j] += p[j]);
        }
        for c in (0..k).filter(|&c| cnt[c] > 0) {
            (0..dim).for_each(|j| book[c * dim + j] = sum[c * dim + j] / cnt[c] as f64);
        }
    }
    book
}

fn vq(w: &mut [f64], cols: usize, dim: usize, k: usize, group: usize) {
    assert!(cols.is_multiple_of(group) && group.is_multiple_of(dim));
    let scales: Vec<f64> = w
        .chunks(group)
        .map(|g| {
            f16((g.iter().map(|x| x * x).sum::<f64>() / group as f64)
                .sqrt()
                .max(1e-8))
        })
        .collect();
    let mut v: Vec<f64> = w.to_vec();
    v.chunks_mut(group)
        .zip(&scales)
        .for_each(|(g, s)| g.iter_mut().for_each(|x| *x /= s));
    let book = kmeans(&v, dim, k, 12);
    for (i, p) in v.chunks(dim).enumerate() {
        let c = nearest(p, &book, dim);
        let s = scales[i * dim / group];
        (0..dim).for_each(|j| w[i * dim + j] = book[c * dim + j] * s);
    }
}

/// Randomized block-Hadamard rotation of each row in blocks of `b` (a power
/// of two): `x ← H_b · (x ⊙ signs)`, orthonormal, so `inverse` undoes it.
/// At inference the activations get the same rotation, so `W'x' = Wx`.
pub fn rotate(w: &mut [f64], cols: usize, b: usize, signs: &[f64], inverse: bool) {
    assert!(b.is_power_of_two() && cols.is_multiple_of(b) && signs.len() == cols);
    let norm = 1.0 / (b as f64).sqrt();
    for row in w.chunks_mut(cols) {
        if !inverse {
            row.iter_mut().zip(signs).for_each(|(x, s)| *x *= s);
        }
        for blk in row.chunks_mut(b) {
            // in-place fast Walsh–Hadamard transform (symmetric, so H⁻¹ = H)
            let mut h = 1;
            while h < b {
                for i in (0..b).step_by(2 * h) {
                    for j in i..i + h {
                        let (a, c) = (blk[j], blk[j + h]);
                        blk[j] = a + c;
                        blk[j + h] = a - c;
                    }
                }
                h *= 2;
            }
            blk.iter_mut().for_each(|x| *x *= norm);
        }
        if inverse {
            row.iter_mut().zip(signs).for_each(|(x, s)| *x *= s);
        }
    }
}

// ------------------------------------------------------------------ decode
// The deployable form of `Vq { dim: 4, k: 2048, group: 128 }` behind the
// block-Hadamard rotation (the format the spike picked): packed codes, f16
// group scales and an f16 codebook, plus the decode GEMV that reads them.
// The rotation moves onto the activation — `y = W x = W' x'` with
// `x' = H(x ⊙ s)` per 128-block — so the kernel never un-rotates a weight:
// it rotates `x` once per step, then per weight row gathers one 4-vector
// per code and dots it with `x'`.

/// Elements per codebook entry.
pub const VQ_DIM: usize = 4;
/// Codebook entries; a code is `log2(VQ_K)` = 11 bits.
pub const VQ_K: usize = 2048;
/// Bits per code.
pub const VQ_BITS: usize = 11;
/// Weights per f16 scale (one warp lane per code: 32 codes).
pub const VQ_GROUP: usize = 128;
/// Hadamard block on the input dim.
pub const VQ_HB: usize = 128;

/// A `rows × cols` matrix in the packed VQ format.
#[derive(Clone, Debug)]
pub struct VqMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Codes, 11 bits each, row-major, as one little-endian bit stream in
    /// u32 words (code `i` starts at bit `11 i`), plus one pad word.
    pub codes: Vec<u32>,
    /// f16 bits, one per `VQ_GROUP` weights, row-major.
    pub scales: Vec<u16>,
    /// f16 bits, `VQ_K × VQ_DIM`.
    pub book: Vec<u16>,
    /// ±1 per input column (the rotation's random signs).
    pub signs: Vec<f32>,
}

fn h16(x: f64) -> u16 {
    super::precision::f16_bits(x)
}

fn f16v(b: u16) -> f64 {
    super::precision::f16_from_bits(b)
}

/// The rotation's signs for a matrix: deterministic in `seed`.
pub fn vq_signs(cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
    (0..cols)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s >> 63 == 1 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

impl VqMatrix {
    /// Stored bits per weight (codes + scales + codebook; signs are a seed).
    pub fn bits_per_weight(&self) -> f64 {
        let bytes = self.codes.len() * 4 + self.scales.len() * 2 + self.book.len() * 2;
        8.0 * bytes as f64 / (self.rows * self.cols) as f64
    }

    /// Bytes a decode step streams for this matrix.
    pub fn bytes(&self) -> usize {
        self.codes.len() * 4 + self.scales.len() * 2 + self.book.len() * 2
    }

    /// Encode `w` (`rows × cols`, row-major): rotate, scale per group, fit
    /// a codebook (k-means, `iters` rounds) unless `book` (`VQ_K × VQ_DIM`)
    /// is given, assign.
    pub fn encode(
        w: &[f64],
        rows: usize,
        cols: usize,
        seed: u64,
        book: Option<&[f64]>,
        iters: usize,
    ) -> Self {
        assert!(w.len() == rows * cols && cols.is_multiple_of(VQ_GROUP) && cols.is_multiple_of(VQ_HB));
        let signs = vq_signs(cols, seed);
        let s64: Vec<f64> = signs.iter().map(|&x| x as f64).collect();
        let mut v = w.to_vec();
        rotate(&mut v, cols, VQ_HB, &s64, false);
        let scales: Vec<u16> = v
            .chunks(VQ_GROUP)
            .map(|g| {
                h16((g.iter().map(|x| x * x).sum::<f64>() / VQ_GROUP as f64)
                    .sqrt()
                    .max(1e-8))
            })
            .collect();
        v.chunks_mut(VQ_GROUP)
            .zip(&scales)
            .for_each(|(g, &s)| g.iter_mut().for_each(|x| *x /= f16v(s)));
        let fit = match book {
            Some(b) => b.to_vec(),
            None => kmeans(&v, VQ_DIM, VQ_K, iters),
        };
        let book: Vec<u16> = fit.iter().map(|&x| h16(x)).collect();
        let bookq: Vec<f64> = book.iter().map(|&b| f16v(b)).collect();
        let n = rows * cols / VQ_DIM;
        let mut codes = vec![0u32; (n * VQ_BITS).div_ceil(32) + 1];
        for (i, p) in v.chunks(VQ_DIM).enumerate() {
            let c = nearest(p, &bookq, VQ_DIM) as u64;
            let (word, bit) = (i * VQ_BITS / 32, i * VQ_BITS % 32);
            let both = c << bit;
            codes[word] |= both as u32;
            codes[word + 1] |= (both >> 32) as u32;
        }
        VqMatrix {
            rows,
            cols,
            codes,
            scales,
            book,
            signs,
        }
    }

    /// Code `i` (row-major over all `rows·cols/4` codes).
    pub fn code(&self, i: usize) -> usize {
        let (word, bit) = (i * VQ_BITS / 32, i * VQ_BITS % 32);
        let both = self.codes[word] as u64 | (self.codes[word + 1] as u64) << 32;
        ((both >> bit) & (VQ_K as u64 - 1)) as usize
    }

    /// Dequantize to the original (un-rotated) basis: the reference.
    pub fn dequant(&self) -> Vec<f64> {
        let mut w: Vec<f64> = (0..self.rows * self.cols)
            .map(|e| {
                let c = self.code(e / VQ_DIM);
                f16v(self.book[c * VQ_DIM + e % VQ_DIM]) * f16v(self.scales[e / VQ_GROUP])
            })
            .collect();
        let s64: Vec<f64> = self.signs.iter().map(|&x| x as f64).collect();
        rotate(&mut w, self.cols, VQ_HB, &s64, true);
        w
    }

    /// The decode kernel on CPU, shaped like the GPU one: rotate `x` once,
    /// then per row, per group (one scale), per code (one gather), a
    /// 4-wide dot against `x'`. f32 throughout, like the device.
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        let xr = rotate_x(x, &self.signs);
        let book: Vec<f32> = self.book.iter().map(|&b| f16v(b) as f32).collect();
        let cpr = self.cols / VQ_DIM;
        (0..self.rows)
            .map(|r| {
                let mut acc = 0.0f32;
                for g in 0..self.cols / VQ_GROUP {
                    let mut part = 0.0f32;
                    for j in 0..VQ_GROUP / VQ_DIM {
                        let col = g * VQ_GROUP + j * VQ_DIM;
                        let c = self.code(r * cpr + col / VQ_DIM);
                        let e = &book[c * VQ_DIM..][..VQ_DIM];
                        part += (0..VQ_DIM).map(|t| e[t] * xr[col + t]).sum::<f32>();
                    }
                    acc += part * f16v(self.scales[r * self.cols / VQ_GROUP + g]) as f32;
                }
                acc
            })
            .collect()
    }
}

/// `x' = H(x ⊙ s)` per `VQ_HB` block, f32 (the kernel's prologue).
pub fn rotate_x(x: &[f32], signs: &[f32]) -> Vec<f32> {
    let mut v: Vec<f64> = x.iter().zip(signs).map(|(a, s)| (a * s) as f64).collect();
    let ones = vec![1.0; v.len()];
    rotate(&mut v, x.len(), VQ_HB, &ones, false);
    v.into_iter().map(|a| a as f32).collect()
}

/// CUDA source for the decode step of one VQ matrix: `vq_rotate` (one
/// block of 128 threads per Hadamard block: `x' = H(x ⊙ s)` in shared
/// memory, an in-register butterfly over 7 stages) and `vq_gemv` (one warp
/// per row; the codebook staged in shared memory as float4; lane `j` of a
/// group decodes code `j` — 32 codes × 4 = the 128-wide group — so every
/// lane reads one 11-bit code per group from a 64-bit window, gathers one
/// float4, dots it with a float4 of `x'`, and the warp reduces once per
/// row, scaling per group). Launch: `vq_rotate<<<cols/128, 128>>>`,
/// `vq_gemv<<<ceil(rows/8), 256, 2048*16>>>`.
pub fn vq_cuda_src() -> String {
    format!(
        r#"#include <cuda_fp16.h>
#define VQ_K {k}
#define VQ_BITS {bits}
#define VQ_GROUP {group}

extern "C" __global__ void vq_rotate(const float* __restrict__ x, const float* __restrict__ s,
                                     float* __restrict__ xr) {{
  __shared__ float b[128];
  int t = threadIdx.x, base = blockIdx.x * 128;
  b[t] = x[base + t] * s[base + t];
  __syncthreads();
  for (int h = 1; h < 128; h <<= 1) {{
    float a = b[t], c = b[t ^ h];
    __syncthreads();
    b[t] = (t & h) ? c - a : a + c;
    __syncthreads();
  }}
  xr[base + t] = b[t] * 0.08838834764831845f;  // 1/sqrt(128)
}}

extern "C" __global__ void vq_gemv(const unsigned* __restrict__ codes,
                                   const __half* __restrict__ scales,
                                   const __half* __restrict__ book,
                                   const float* __restrict__ xr, float* __restrict__ y,
                                   int rows, int cols) {{
  extern __shared__ float4 cb[];
  for (int i = threadIdx.x; i < VQ_K; i += blockDim.x) {{
    const __half* e = book + 4 * i;
    cb[i] = make_float4(__half2float(e[0]), __half2float(e[1]), __half2float(e[2]), __half2float(e[3]));
  }}
  __syncthreads();
  int lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) return;
  int groups = cols / VQ_GROUP;
  const float4* x4 = reinterpret_cast<const float4*>(xr);
  size_t c0 = (size_t)r * (cols / 4);
  float acc = 0.f;
  for (int g = 0; g < groups; ++g) {{
    size_t i = c0 + (size_t)g * 32 + lane;
    size_t bit = i * VQ_BITS;
    unsigned long long w = (unsigned long long)codes[bit >> 5] |
                           ((unsigned long long)codes[(bit >> 5) + 1] << 32);
    int c = (int)((w >> (bit & 31)) & (VQ_K - 1));
    float4 e = cb[c], v = x4[g * 32 + lane];
    acc += (e.x * v.x + e.y * v.y + e.z * v.z + e.w * v.w) *
           __half2float(scales[(size_t)r * groups + g]);
  }}
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) y[r] = acc;
}}
"#,
        k = VQ_K,
        bits = VQ_BITS,
        group = VQ_GROUP
    )
}

/// Weights per f16 scale/bias pair in the packed affine format.
pub const AFF_GROUP: usize = 64;
/// Weights per packed chunk: one GPU lane's unit of work, `bits` u32 words.
pub const AFF_CHUNK: usize = 32;
/// Columns a warp's 32 chunks span (the packing's stripe: rows must be a
/// multiple of this wide).
pub const AFF_SPAN: usize = 1024;

/// Where weight `col` of a row lives in the packing: (chunk within the
/// row, index `k` within the chunk). Lane `l` of superchunk `sc` holds, at
/// step `s` (0..8), columns `sc·1024 + s·128 + 4l .. +4` as `k = 4s..4s+4`
/// — so at every step the warp's 32 lanes read 128 contiguous columns of
/// `x'` (coalesced), and each lane still loads its codes as `bits`
/// adjacent words.
pub fn aff_slot(col: usize) -> (usize, usize) {
    let (sc, rem) = (col / AFF_SPAN, col % AFF_SPAN);
    let (step, lane, t) = (rem / 128, rem % 128 / 4, rem % 4);
    (sc * 32 + lane, 4 * step + t)
}

/// A `rows × cols` matrix in the packed affine format the GPTQ study
/// picked (`g3g64h`: 3-bit codes, f16 scale and bias per 64 weights,
/// behind the same block-Hadamard rotation as [`VqMatrix`], so 3.5 bits
/// per weight). `bits` is 3 or 4; 4 is the MLX-style baseline the decode
/// bench times against.
#[derive(Clone, Debug)]
pub struct AffineMatrix {
    pub rows: usize,
    pub cols: usize,
    pub bits: usize,
    /// Codes, row-major, in chunks of [`AFF_CHUNK`] weights placed by
    /// [`aff_slot`], `bits` u32 words per chunk, so a lane loads its chunk
    /// in one go. Within a chunk, by index `k`: 4-bit: word
    /// `t` holds weights `8t..8t+8`, a nibble each, low first. 3-bit (bit
    /// planes): words 0 and 1 hold the low two bits of weights 0..16 and
    /// 16..32 (2 bits each, low first), word 2 the high bit of weight `k`
    /// at bit `k`.
    pub codes: Vec<u32>,
    /// f16 bits, one per `AFF_GROUP` weights, row-major.
    pub scales: Vec<u16>,
    /// f16 bits (the group's minimum), one per `AFF_GROUP` weights.
    pub biases: Vec<u16>,
    /// ±1 per input column (the rotation's random signs); empty: none.
    pub signs: Vec<f32>,
}

impl AffineMatrix {
    /// Pack codes (each `< 2^bits`, row-major, already in the rotated basis)
    /// with their group scales and biases: how GPTQ output (bench/lowbit's
    /// `gptq_affine`) is loaded.
    pub fn from_codes(
        rows: usize,
        cols: usize,
        bits: usize,
        q: &[u8],
        scales: Vec<u16>,
        biases: Vec<u16>,
        signs: Vec<f32>,
    ) -> Self {
        assert!(bits == 3 || bits == 4);
        assert!(q.len() == rows * cols && cols.is_multiple_of(AFF_SPAN) && cols.is_multiple_of(VQ_HB));
        assert!(scales.len() == rows * cols / AFF_GROUP && biases.len() == scales.len());
        assert!(signs.is_empty() || signs.len() == cols);
        let mut codes = vec![0u32; q.len() / AFF_CHUNK * bits];
        for (i, &c) in q.iter().enumerate() {
            let c = c as u32;
            assert!(c < 1 << bits);
            let (ch, k) = aff_slot(i % cols);
            let w = (i / cols * (cols / AFF_CHUNK) + ch) * bits;
            if bits == 4 {
                codes[w + k / 8] |= c << (4 * (k % 8));
            } else {
                codes[w + k / 16] |= (c & 3) << (2 * (k % 16));
                codes[w + 2] |= (c >> 2) << k;
            }
        }
        AffineMatrix {
            rows,
            cols,
            bits,
            codes,
            scales,
            biases,
            signs,
        }
    }

    /// Round-to-nearest encode of `w` (`rows × cols`): rotate (when `seed`
    /// is given), then per group min/max affine codes with f16 params.
    pub fn encode(w: &[f64], rows: usize, cols: usize, bits: usize, seed: Option<u64>) -> Self {
        let signs = seed.map(|s| vq_signs(cols, s)).unwrap_or_default();
        let mut v = w.to_vec();
        if !signs.is_empty() {
            let s64: Vec<f64> = signs.iter().map(|&x| x as f64).collect();
            rotate(&mut v, cols, VQ_HB, &s64, false);
        }
        let qmax = ((1 << bits) - 1) as f64;
        let (mut q, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
        for g in v.chunks(AFF_GROUP) {
            let lo = g.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let (sc, b) = (h16(((hi - lo) / qmax).max(1e-8)), h16(lo));
            let (scf, bf) = (f16v(sc), f16v(b));
            q.extend(
                g.iter()
                    .map(|x| ((x - bf) / scf).round().clamp(0.0, qmax) as u8),
            );
            scales.push(sc);
            biases.push(b);
        }
        Self::from_codes(rows, cols, bits, &q, scales, biases, signs)
    }

    /// Stored bits per weight (codes + scales + biases; signs are a seed).
    pub fn bits_per_weight(&self) -> f64 {
        8.0 * self.bytes() as f64 / (self.rows * self.cols) as f64
    }

    /// Bytes a decode step streams for this matrix.
    pub fn bytes(&self) -> usize {
        self.codes.len() * 4 + (self.scales.len() + self.biases.len()) * 2
    }

    /// Code `i` (row-major).
    pub fn code(&self, i: usize) -> usize {
        let (ch, k) = aff_slot(i % self.cols);
        let w = (i / self.cols * (self.cols / AFF_CHUNK) + ch) * self.bits;
        let c = if self.bits == 4 {
            self.codes[w + k / 8] >> (4 * (k % 8)) & 15
        } else {
            (self.codes[w + k / 16] >> (2 * (k % 16)) & 3) | (self.codes[w + 2] >> k & 1) << 2
        };
        c as usize
    }

    /// Scale and bias interleaved per group (f16 bits): the kernel's
    /// `half2` parameter stream.
    pub fn scale_bias(&self) -> Vec<u16> {
        self.scales
            .iter()
            .zip(&self.biases)
            .flat_map(|(&s, &b)| [s, b])
            .collect()
    }

    /// Dequantize to the original (un-rotated) basis: the reference.
    pub fn dequant(&self) -> Vec<f64> {
        let mut w: Vec<f64> = (0..self.rows * self.cols)
            .map(|e| {
                let g = e / AFF_GROUP;
                self.code(e) as f64 * f16v(self.scales[g]) + f16v(self.biases[g])
            })
            .collect();
        if !self.signs.is_empty() {
            let s64: Vec<f64> = self.signs.iter().map(|&x| x as f64).collect();
            rotate(&mut w, self.cols, VQ_HB, &s64, true);
        }
        w
    }

    /// The decode kernel on CPU, shaped like the GPU one: rotate `x` once;
    /// per row, per group, `scale · Σ q x' + bias · Σ x'`, f32 throughout.
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        let xr = if self.signs.is_empty() {
            x.to_vec()
        } else {
            rotate_x(x, &self.signs)
        };
        let gpr = self.cols / AFF_GROUP;
        (0..self.rows)
            .map(|r| {
                let mut acc = 0.0f32;
                for g in 0..gpr {
                    let (mut qx, mut sx) = (0.0f32, 0.0f32);
                    for t in 0..AFF_GROUP {
                        let col = g * AFF_GROUP + t;
                        qx += self.code(r * self.cols + col) as f32 * xr[col];
                        sx += xr[col];
                    }
                    let gi = r * gpr + g;
                    acc += qx * f16v(self.scales[gi]) as f32 + sx * f16v(self.biases[gi]) as f32;
                }
                acc
            })
            .collect()
    }
}

/// Superchunks (1024 columns) each lane of [`affine_cuda_src`]'s kernels
/// loads before any math (a 4096-wide row is 4, so all of it).
pub const AFF_UNROLL: usize = 4;

/// CUDA source for the decode GEMV of an [`AffineMatrix`] with `bits`-bit
/// codes: `aff{bits}_gemv`, one warp per row, plus `aff_xsum` (the
/// per-group sums `Σ x'` over 64 columns: once per token, shared by every
/// matrix with that input).
///
/// Lane `l` first loads its chunk's `bits` adjacent words for
/// [`AFF_UNROLL`] superchunks (1024 columns each; the warp: contiguous
/// spans), then per superchunk, for steps `s = 0..8`, reads the float4 of
/// `x'` at columns `s·128 + 4l` (the warp: 512 contiguous bytes,
/// L1-resident). Codes go into the mantissa of 1.0f —
/// `as_float(0x3F800000 | q << (23 - bits)) = 1 + q/2^bits`, exact, one
/// shift and one OR, no int→float convert — so per group the row gets
/// `scale · Σ q x' + bias · Σ x' = A · D + B · Σ x'` with
/// `D = Σ (1 + q/2^bits) x'`, `A = 2^bits · scale`, `B = bias − A`. The
/// warp reduces once. `x'` comes from `vq_rotate` (same Hadamard block;
/// q/k/v and gate/up share one). Launch: `aff_xsum<<<cols/64/8, 256>>>(xr,
/// xsum, cols)` then `aff{bits}_gemv<<<ceil(rows/8), 256>>>(codes,
/// scale_bias, xr, xsum, y, rows, cols)`; `cols % 1024 == 0`.
pub fn affine_cuda_src(bits: usize) -> String {
    assert!(bits == 3 || bits == 4);
    let sh = 23 - bits;
    let (load, nib) = if bits == 4 {
        (
            "uint4 u = *reinterpret_cast<const uint4*>(codes + base);\n      uw[j][0] = u.x; uw[j][1] = u.y; uw[j][2] = u.z; uw[j][3] = u.w;",
            // step s: nibbles 16(s&1).. of word s>>1
            format!("unsigned c = uw[j][s >> 1] >> (16 * (s & 1));\n        float f0 = __uint_as_float(0x3F800000u | (c & 15u) << {sh}), f1 = __uint_as_float(0x3F800000u | (c >> 4 & 15u) << {sh}),\n              f2 = __uint_as_float(0x3F800000u | (c >> 8 & 15u) << {sh}), f3 = __uint_as_float(0x3F800000u | (c >> 12 & 15u) << {sh});"),
        )
    } else {
        (
            "uw[j][0] = codes[base]; uw[j][1] = codes[base + 1]; uw[j][2] = codes[base + 2];",
            // step s: byte (s&3) of word s>>2 (low 2 bits), nibble s of word 2 (high bit)
            format!("unsigned lo = uw[j][s >> 2] >> (8 * (s & 3)), hi = uw[j][2] >> (4 * s);\n        float f0 = __uint_as_float(0x3F800000u | ((lo & 3u) | (hi & 1u) << 2) << {sh}),\n              f1 = __uint_as_float(0x3F800000u | ((lo >> 2 & 3u) | (hi & 2u) << 1) << {sh}),\n              f2 = __uint_as_float(0x3F800000u | ((lo >> 4 & 3u) | (hi & 4u)) << {sh}),\n              f3 = __uint_as_float(0x3F800000u | ((lo >> 6 & 3u) | (hi & 8u) >> 1) << {sh});"),
        )
    };
    format!(
        r#"#include <cuda_fp16.h>
#ifndef AFF_XSUM_DEFINED
#define AFF_XSUM_DEFINED
extern "C" __global__ void aff_xsum(const float* __restrict__ xr, float* __restrict__ xsum, int cols) {{
  int lane = threadIdx.x & 31, g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (g >= cols / {group}) return;
  float a = xr[g * {group} + lane] + xr[g * {group} + 32 + lane];
  for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
  if (lane == 0) xsum[g] = a;
}}
#endif
extern "C" __global__ void aff{bits}_gemv(const unsigned* __restrict__ codes,
                                     const __half2* __restrict__ sb,
                                     const float* __restrict__ xr, const float* __restrict__ xsum,
                                     float* __restrict__ y, int rows, int cols) {{
  int lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) return;
  int supers = cols / {span}, groups = cols / {group};
  const float4* x4 = reinterpret_cast<const float4*>(xr);
  const __half2* p = sb + (size_t)r * groups;
  float acc = 0.f;
  for (int s0 = 0; s0 < supers; s0 += {nu}) {{
    unsigned uw[{nu}][{bits}];
#pragma unroll
    for (int j = 0; j < {nu}; ++j) {{
      size_t base = ((size_t)r * supers * 32 + min(s0 + j, supers - 1) * 32 + lane) * {bits};
      {load}
    }}
#pragma unroll
    for (int j = 0; j < {nu}; ++j) {{
      int sc = s0 + j;
      if (sc >= supers) break;
#pragma unroll
      for (int s = 0; s < 8; ++s) {{
        float4 v = __ldg(x4 + sc * 256 + s * 32 + lane);
        int g = sc * 16 + 2 * s + (lane >> 4);
        float2 sbg = __half22float2(p[g]);
        {nib}
        float a = sbg.x * {mul}.f, d = f0 * v.x + f1 * v.y + f2 * v.z + f3 * v.w;
        // each of the group's 16 lanes holds a quarter of its 64 columns: B·Σx' once per group
        acc += a * d + ((lane & 15) == 0 ? (sbg.y - a) * __ldg(xsum + g) : 0.f);
      }}
    }}
  }}
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) y[r] = acc;
}}
"#,
        bits = bits,
        nu = AFF_UNROLL,
        span = AFF_SPAN,
        group = AFF_GROUP,
        mul = 1 << bits,
        load = load,
        nib = nib,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift: gaussian-ish weights with a few outlier columns, like LLM
    /// projections.
    fn weights(rows: usize, cols: usize) -> Vec<f64> {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..rows * cols)
            .map(|i| {
                let g: f64 = (0..4).map(|_| u() - 0.5).sum();
                if i % cols % 61 == 0 {
                    12.0 * g
                } else {
                    g
                }
            })
            .collect()
    }

    fn mse(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64
    }

    fn signs(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if (i * 2654435761) % 7 < 3 { -1.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn rotate_roundtrips() {
        let w0 = weights(4, 256);
        let mut w = w0.clone();
        rotate(&mut w, 256, 128, &signs(256), false);
        assert!(mse(&w, &w0) > 0.01);
        rotate(&mut w, 256, 128, &signs(256), true);
        assert!(mse(&w, &w0) < 1e-24);
    }

    #[test]
    fn affine_error_shrinks_with_bits() {
        let w0 = weights(8, 256);
        let err = |bits| {
            let mut w = w0.clone();
            LowBit::Affine { bits, group: 64 }.emulate(&mut w, 256);
            mse(&w, &w0)
        };
        let (e2, e3, e4) = (err(2), err(3), err(4));
        assert!(e4 < e3 && e3 < e2, "{e2} {e3} {e4}");
    }

    #[test]
    fn hadamard_tames_outliers() {
        let (cols, w0) = (256, weights(8, 256));
        let f = LowBit::Affine { bits: 3, group: 64 };
        let mut plain = w0.clone();
        f.emulate(&mut plain, cols);
        let mut rot = w0.clone();
        rotate(&mut rot, cols, 128, &signs(cols), false);
        f.emulate(&mut rot, cols);
        rotate(&mut rot, cols, 128, &signs(cols), true);
        assert!(
            mse(&rot, &w0) < mse(&plain, &w0),
            "{} {}",
            mse(&rot, &w0),
            mse(&plain, &w0)
        );
    }

    #[test]
    fn vq_is_exact_when_codebook_covers_the_data() {
        // 4 distinct 2-vectors, repeated, one scale group: k = 4 recovers them.
        let pat = [1.0, -1.0, 0.5, 2.0, -2.0, 0.25, 0.0, 1.5];
        let w0: Vec<f64> = pat.iter().cycle().take(64).cloned().collect();
        let mut w = w0.clone();
        LowBit::Vq {
            dim: 2,
            k: 4,
            group: 64,
        }
        .emulate(&mut w, 64);
        assert!(mse(&w, &w0) < 1e-6, "{}", mse(&w, &w0));
    }

    #[test]
    fn vq_beats_scalar_at_two_bits() {
        let (cols, w0) = (256, weights(64, 256));
        let mut a = w0.clone();
        LowBit::Affine { bits: 2, group: 64 }.emulate(&mut a, cols);
        let mut v = w0.clone();
        LowBit::Vq {
            dim: 4,
            k: 256,
            group: 128,
        }
        .emulate(&mut v, cols);
        assert!(
            mse(&v, &w0) < mse(&a, &w0),
            "vq {} affine {}",
            mse(&v, &w0),
            mse(&a, &w0)
        );
    }

    #[test]
    fn bits_accounting() {
        let n = 4096 * 4096;
        assert_eq!(
            LowBit::Affine { bits: 4, group: 64 }.bits_per_weight(n),
            4.5
        );
        let b = LowBit::Vq {
            dim: 4,
            k: 256,
            group: 128,
        }
        .bits_per_weight(n);
        assert!((b - 2.125).abs() < 1e-3, "{b}");
        assert_eq!(
            LowBit::Vq {
                dim: 4,
                k: 256,
                group: 128
            }
            .lookups_per_weight(),
            0.25
        );
    }

    fn vq_fixture(rows: usize, cols: usize) -> (Vec<f64>, VqMatrix) {
        let w = weights(rows, cols);
        // a fixed codebook (no k-means) keeps the test fast: a 4-d grid
        // over [-2, 2] with 2048 points (8·8·8·4)
        let lv = |n: usize, i: usize| -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        let book: Vec<f64> = (0..VQ_K)
            .flat_map(|c| {
                [
                    lv(8, c % 8),
                    lv(8, c / 8 % 8),
                    lv(8, c / 64 % 8),
                    lv(4, c / 512),
                ]
            })
            .collect();
        let m = VqMatrix::encode(&w, rows, cols, 7, Some(&book), 0);
        (w, m)
    }

    #[test]
    fn vq_codes_pack_11_bits() {
        let (_, m) = vq_fixture(4, 256);
        assert_eq!(m.codes.len(), (4 * 256 / 4 * 11usize).div_ceil(32) + 1);
        assert!((0..256).all(|i| m.code(i) < VQ_K));
        // 2.875 bits/weight + 0.008 for the codebook at 4096×4096
        let bpw = 11.0 / 4.0 + 16.0 / 128.0 + (16 * VQ_K * 4) as f64 / (4096.0 * 4096.0);
        assert!((bpw - 2.883).abs() < 1e-3, "{bpw}");
        let fmt = LowBit::Vq {
            dim: 4,
            k: 2048,
            group: 128,
        }
        .bits_per_weight(4096 * 4096);
        assert!((fmt - bpw).abs() < 1e-12);
    }

    #[test]
    fn vq_gemv_matches_dequant_reference() {
        let (rows, cols) = (24, 512);
        let (w, m) = vq_fixture(rows, cols);
        let deq = m.dequant();
        // the format is a real approximation of w
        let rel = mse(&deq, &w) / (w.iter().map(|x| x * x).sum::<f64>() / w.len() as f64);
        assert!(rel < 0.2, "rel {rel}");
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let y = m.gemv(&x);
        for r in 0..rows {
            let want: f64 = (0..cols).map(|c| deq[r * cols + c] * x[c] as f64).sum();
            let scale: f64 = (0..cols)
                .map(|c| (deq[r * cols + c] * x[c] as f64).abs())
                .sum();
            assert!(
                (y[r] as f64 - want).abs() < 1e-5 * scale.max(1.0),
                "row {r}: {} vs {want}",
                y[r]
            );
        }
    }

    #[test]
    fn vq_rotation_is_moved_onto_x() {
        let s = vq_signs(256, 3);
        let x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let xr = rotate_x(&x, &s);
        let n0: f32 = x.iter().map(|a| a * a).sum();
        let n1: f32 = xr.iter().map(|a| a * a).sum();
        assert!((n0 - n1).abs() < 1e-3 * n0);
    }

    #[test]
    fn vq_cuda_src_names_both_kernels() {
        let s = vq_cuda_src();
        assert!(
            s.contains("vq_rotate") && s.contains("vq_gemv") && s.contains("#define VQ_BITS 11")
        );
    }

    #[test]
    fn affine_packs_3_bits_at_3_5_bpw() {
        let w = weights(8, 4096);
        let m = AffineMatrix::encode(&w, 8, 4096, 3, Some(5));
        assert!(
            (m.bits_per_weight() - 3.5).abs() < 1e-2,
            "{}",
            m.bits_per_weight()
        );
        assert!((0..8 * 4096).all(|i| m.code(i) < 8));
        let q: Vec<u8> = (0..8 * 4096).map(|i| m.code(i) as u8).collect();
        let n =
            AffineMatrix::from_codes(8, 4096, 3, &q, m.scales.clone(), m.biases.clone(), vec![]);
        assert_eq!(n.codes, m.codes);
        let fmt = LowBit::Affine { bits: 3, group: 64 }.bits_per_weight(1);
        assert!((fmt - 3.5).abs() < 1e-12);
    }

    #[test]
    fn affine_gemv_matches_dequant_reference() {
        for (bits, seed) in [(3, Some(11)), (4, Some(11)), (3, None)] {
            let (rows, cols) = (24, 2048);
            let w = weights(rows, cols);
            let m = AffineMatrix::encode(&w, rows, cols, bits, seed);
            let deq = m.dequant();
            let rel = mse(&deq, &w) / (w.iter().map(|x| x * x).sum::<f64>() / w.len() as f64);
            assert!(
                rel < if bits == 3 { 0.15 } else { 0.05 },
                "bits {bits} rel {rel}"
            );
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
                .collect();
            let y = m.gemv(&x);
            for r in 0..rows {
                let want: f64 = (0..cols).map(|c| deq[r * cols + c] * x[c] as f64).sum();
                let scale: f64 = (0..cols)
                    .map(|c| (deq[r * cols + c] * x[c] as f64).abs())
                    .sum();
                assert!(
                    (y[r] as f64 - want).abs() < 1e-5 * scale.max(1.0),
                    "bits {bits} row {r}: {} vs {want}",
                    y[r]
                );
            }
        }
    }

    #[test]
    fn affine_rotation_tames_outlier_columns() {
        // the outlier columns in `weights` blow up per-group ranges;
        // rotating first spreads them, so 3-bit error drops
        let w = weights(16, 1024);
        let plain = AffineMatrix::encode(&w, 16, 1024, 3, None).dequant();
        let rot = AffineMatrix::encode(&w, 16, 1024, 3, Some(9)).dequant();
        assert!(mse(&rot, &w) < 0.8 * mse(&plain, &w));
    }

    #[test]
    fn affine_cuda_src_names_kernels() {
        assert!(
            affine_cuda_src(3).contains("aff3_gemv")
                && affine_cuda_src(3).contains("uw[j][2] >> (4 * s)")
                && affine_cuda_src(3).contains("aff_xsum")
        );
        assert!(
            affine_cuda_src(4).contains("aff4_gemv") && affine_cuda_src(4).contains("uint4 u =")
        );
    }

    /// Writes GPU test vectors for bench/lowbit/vq_bench.cu's correctness
    /// check: `aff{3,4}.bin` = rows, cols (u32), codes (u32), scale/bias
    /// (f16 pairs), x' (f32, already rotated), y = the CPU gemv (f32).
    /// `VQ_CU_DIR=. cargo test -p tang-compile --lib affine_vectors_dump -- --ignored`
    #[test]
    #[ignore]
    fn affine_vectors_dump() {
        let dir = std::env::var("VQ_CU_DIR").unwrap_or_else(|_| ".".into());
        let (rows, cols) = (96, 4096);
        let w = weights(rows, cols);
        for bits in [3, 4] {
            let m = AffineMatrix::encode(&w, rows, cols, bits, Some(13));
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
                .collect();
            let (xr, y) = (rotate_x(&x, &m.signs), m.gemv(&x));
            let mut b = Vec::new();
            b.extend((rows as u32).to_le_bytes());
            b.extend((cols as u32).to_le_bytes());
            m.codes.iter().for_each(|v| b.extend(v.to_le_bytes()));
            m.scale_bias()
                .iter()
                .for_each(|v| b.extend(v.to_le_bytes()));
            xr.iter().chain(&y).for_each(|v| b.extend(v.to_le_bytes()));
            std::fs::write(format!("{dir}/aff{bits}.bin"), b).unwrap();
        }
    }

    /// Writes the kernel source for bench/lowbit/vq_bench.cu:
    /// `VQ_CU_OUT=vq.cu cargo test -p tang-compile --lib vq_cuda_src_dump -- --ignored`
    #[test]
    #[ignore]
    fn vq_cuda_src_dump() {
        let out = std::env::var("VQ_CU_OUT").unwrap_or_else(|_| "vq.cu".into());
        let src = vq_cuda_src()
            + &affine_cuda_src(3).replace("#include <cuda_fp16.h>\n", "")
            + &affine_cuda_src(4).replace("#include <cuda_fp16.h>\n", "");
        std::fs::write(out, src).unwrap();
    }
}
