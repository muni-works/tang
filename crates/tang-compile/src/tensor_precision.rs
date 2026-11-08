//! Precision as a schedule.
//!
//! The graph is math; number formats are a decision about how to run it,
//! like `inline` or `tile`. Each node carries a [`Precision`]: the format
//! its buffer is *stored* in (inputs and materialized comps) and the
//! format its reduction *accumulates* in. Everything defaults to f32.
//!
//! [`TensorGraph::eval_emulated`] rounds to the scheduled formats inside
//! the f64 interpreter, so the error of a schedule can be predicted
//! exactly-ish before running it; [`TensorGraph::precision_error`]
//! measures it against the pure-f64 reference.

use super::{TensorGraph, TensorId};

/// A number format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Format {
    #[default]
    F32,
    /// IEEE binary16: 10 explicit mantissa bits, exponent −14..15.
    F16,
    /// bfloat16: f32's exponent range, 7 explicit mantissa bits.
    /// Interpreter-only for now: WGSL has no bf16 type, and the lowering
    /// rejects it.
    BF16,
    /// OCP FP8 E4M3 (3 mantissa bits, max 448, no ∞) with one f32 scale
    /// per `block` consecutive elements (0: one scale for the tensor).
    /// The scale is part of the type: a block's scale is its largest
    /// magnitude over 448, the stored byte is `x / scale` rounded, and
    /// the scales live in the same buffer after the bytes (see
    /// [`Format::buffer_bytes`]), so a kernel reads the scale map like
    /// any other view of the operand.
    E4M3(u32),
    /// OCP FP8 E5M2 (2 mantissa bits, max 57344), block-scaled like
    /// [`Format::E4M3`].
    E5M2(u32),
    /// An accumulation format only: Hopper's FP8 tensor-core accumulator
    /// without promotion ("fast accumulation"): f32 range, but each
    /// addition keeps about [`FP8_ACCUM_BITS`] mantissa bits, truncated.
    /// A reduction with this `accum` lets an FP8 tile accumulate the
    /// whole reduction in the tensor cores; with `F32` the tile promotes
    /// its partial sum into an f32 register every 128 elements.
    Fp8Accum,
    /// GGUF Q4_K: super-blocks of 256 along the last dim, 144 bytes each
    /// (f16 `d`, `dmin`, 12 bytes of 6-bit sub-block scales and mins, 128
    /// bytes of 4-bit codes). Input storage only: the graph's values are the
    /// dequantized weights, the buffer holds the GGUF bytes as they are.
    Q4K,
    /// GGUF Q5_K: as [`Format::Q4K`] plus 32 bytes of fifth bits, 176 bytes
    /// a super-block.
    Q5K,
    /// GGUF Q8_0: blocks of 32 along the last dim, 34 bytes each (f16 `d`,
    /// 32 int8 codes). Input storage only, like [`Format::Q4K`].
    Q8_0,
    /// int8 codes with one f32 scale a 32 along the last dim (`d = amax /
    /// 127`, codes `round(v / d)`), the scale's companion `d · Σ codes`
    /// beside it: what an integer MMA reads as its activation operand. A
    /// buffer is `[codes: n bytes][(d, d·Σc): n / 32 × 8 bytes]`. The writer
    /// quantizes; [`TensorGraph::eval_emulated`] rounds the same way.
    Q8x32,
    /// [`Q8x32`](Format::Q8x32)'s codes then one f32 `d` a 32 (`[codes: n bytes][d: n / 32 ×
    /// 4 bytes]`, no companion): the int8 activation layout hand-written kernels read. Written
    /// only (no MMA reads it).
    Q8x32F,
}

/// Softmax weights rounded to an 8-bit operand format are first scaled by
/// `2^FP8_SOFTMAX_SHIFT`: `exp(s − m) ∈ (0, 1]` would sit mostly in
/// E4M3's subnormals (below 2⁻⁶), `2⁸·exp(s − m) ∈ (0, 256]` keeps three
/// mantissa bits down to `2⁻⁶·2⁻⁸` (FlashAttention-3's max offset). The
/// row sum is scaled alike, so the normalized result is not.
pub const FP8_SOFTMAX_SHIFT: i32 = 8;

/// The softmax weight `w = exp(s − m)` as an `operand`-format tile feeds it
/// to the tensor cores (scaled into range first when 8-bit).
pub fn round_weight(operand: Format, w: f64) -> f64 {
    if operand.is_fp8() {
        let k = 2f64.powi(FP8_SOFTMAX_SHIFT);
        operand.round(w * k) / k
    } else {
        operand.round(w)
    }
}

/// Explicit mantissa bits [`Format::Fp8Accum`] keeps (calibrated on H100:
/// truncation to 14 bits matches the GPU to 4e-4 relative, 13 or 15 bits
/// are off by 1e-2 and 4e-3;
/// see `examples/hopper_fp8.rs`).
pub const FP8_ACCUM_BITS: i32 = 14;

impl Format {
    /// Bytes per element in a buffer.
    pub fn bytes(self) -> usize {
        match self {
            Format::F32 => 4,
            Format::F16 | Format::BF16 => 2,
            Format::E4M3(_) | Format::E5M2(_) => 1,
            Format::Fp8Accum => 4,
            Format::Q4K | Format::Q5K | Format::Q8_0 | Format::Q8x32 | Format::Q8x32F => 1,
        }
    }

    /// Whether this is a block format (its rounding is per block, not per
    /// element, and its buffer is not an array of one element type).
    pub fn is_block(self) -> bool {
        matches!(
            self,
            Format::Q4K | Format::Q5K | Format::Q8_0 | Format::Q8x32 | Format::Q8x32F
        )
    }

    /// 32-bit words a buffer of `len` elements takes.
    pub fn words(self, len: usize) -> usize {
        match self {
            Format::F32 => len,
            Format::F16 | Format::BF16 => len.div_ceil(2),
            Format::Q4K => len / 256 * 36,
            Format::Q5K => len / 256 * 44,
            Format::Q8_0 => (len / 32 * 34).div_ceil(4),
            Format::Q8x32 => len / 4 + len / 32 * 2,
            Format::Q8x32F => len / 4 + len / 32,
            f @ (Format::E4M3(_) | Format::E5M2(_)) => f.buffer_bytes(len).div_ceil(4),
            Format::Fp8Accum => len,
        }
    }

    /// An 8-bit block-scaled format.
    pub fn is_fp8(self) -> bool {
        matches!(self, Format::E4M3(_) | Format::E5M2(_))
    }

    /// Elements per scale of a block-scaled format (`n` for a per-tensor
    /// scale of an `n`-element buffer); `None` for unscaled formats.
    pub fn block(self, n: usize) -> Option<usize> {
        match self {
            Format::E4M3(b) | Format::E5M2(b) => Some(if b == 0 { n.max(1) } else { b as usize }),
            _ => None,
        }
    }

    /// Largest finite value (of the unscaled format).
    pub fn max_finite(self) -> f64 {
        match self {
            Format::F32 | Format::Fp8Accum => f32::MAX as f64,
            Format::F16 => 65504.0,
            Format::BF16 => f32::from_bits(0x7f7f_0000) as f64,
            Format::E4M3(_) => 448.0,
            Format::E5M2(_) => 57344.0,
            Format::Q4K | Format::Q5K | Format::Q8_0 | Format::Q8x32 | Format::Q8x32F => f32::MAX as f64,
        }
    }

    /// Bytes of an `n`-element buffer: block-scaled formats store the
    /// bytes, padded to 16, then one f32 scale per block.
    pub fn buffer_bytes(self, n: usize) -> usize {
        match self.block(n) {
            Some(b) => n.next_multiple_of(16) + 4 * n.div_ceil(b),
            None if self.is_block() => self.words(n) * 4,
            None => n * self.bytes(),
        }
    }

    /// Byte offset of the scales in a block-scaled `n`-element buffer.
    pub fn scale_offset(n: usize) -> usize {
        n.next_multiple_of(16)
    }

    /// Per-block scales of `x` under a block-scaled format (f32 values):
    /// the block's largest magnitude over the format's largest finite
    /// value, 1 for an all-zero block.
    pub fn scales(self, x: &[f64]) -> Vec<f64> {
        let b = self.block(x.len()).expect("block-scaled format");
        let max = self.max_finite();
        x.chunks(b)
            .map(|c| {
                let a = c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                if a > 0.0 && a.is_finite() {
                    (a / max) as f32 as f64
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// A whole stored buffer rounded to this format: element-wise
    /// [`Format::round`], or for a block-scaled format `round(x/s)·s`
    /// with each block's scale `s` ([`Format::scales`]).
    pub fn round_buf(self, x: &mut [f64]) {
        match self.block(x.len()) {
            None => x.iter_mut().for_each(|v| *v = self.round(*v)),
            Some(b) => {
                let s = self.scales(x);
                for (c, s) in x.chunks_mut(b).zip(s) {
                    c.iter_mut().for_each(|v| *v = self.round(*v / s) * s);
                }
            }
        }
    }

    /// `x` rounded to this format: round to nearest, ties to even, with
    /// gradual underflow (subnormals) and overflow to ±∞.
    pub fn round(self, x: f64) -> f64 {
        match self {
            // the cast is IEEE round-to-nearest-even, subnormals and ∞ included
            Format::F32 => x as f32 as f64,
            Format::F16 => round_to(x, 10, -14, 65504.0),
            Format::BF16 => round_to(x, 7, -126, f32::from_bits(0x7f7f_0000) as f64),
            // saturating (cvt.rn.satfinite): fp8 has no ∞ to overflow to
            Format::E4M3(_) => sat(round_to(x, 3, -6, f64::INFINITY), 448.0),
            Format::E5M2(_) => sat(round_to(x, 2, -14, f64::INFINITY), 57344.0),
            Format::Fp8Accum => truncate_to(x as f32 as f64, FP8_ACCUM_BITS),
            // per block: see `round_blocks`
            Format::Q4K | Format::Q5K | Format::Q8_0 | Format::Q8x32 | Format::Q8x32F => x,
        }
    }
}

impl Format {
    /// Round a row-major buffer to this format block by block (identity
    /// for element formats, which [`Format::round`] covers). Q8x32 rounds
    /// each 32 of the last dim to `d · round(v / d)`, `d = amax / 127`, as
    /// f32 arithmetic does it on a GPU. Q4K / Q5K inputs are taken as
    /// already representable (they are dequantized GGUF values).
    pub fn round_blocks(self, buf: &mut [f64]) {
        if self != Format::Q8x32 && self != Format::Q8x32F {
            return;
        }
        for blk in buf.chunks_mut(32) {
            let amax = blk.iter().fold(0f32, |a, &v| a.max((v as f32).abs()));
            let d = amax / 127.0;
            for v in blk.iter_mut() {
                *v = if d == 0.0 {
                    0.0
                } else {
                    ((*v as f32 / d).round().clamp(-127.0, 127.0) * d) as f64
                };
            }
        }
    }
}

/// Round to a binary format with `mant` explicit mantissa bits, minimum
/// normal exponent `emin` and largest finite value `max`.
fn round_to(x: f64, mant: i32, emin: i32, max: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    let e = (biased - 1023).max(emin); // below emin: the subnormal quantum
    let quantum = 2f64.powi(e - mant);
    let r = (x / quantum).round_ties_even() * quantum;
    if r.abs() > max {
        f64::INFINITY.copysign(x)
    } else {
        r
    }
}

/// `x` truncated (toward zero) to `mant` explicit mantissa bits.
fn truncate_to(x: f64, mant: i32) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let quantum = 2f64.powi(e - mant);
    (x / quantum).trunc() * quantum
}

fn sat(x: f64, max: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        x.clamp(-max, max)
    }
}

/// FP8 bits of `x` (already scaled) in `f` (E4M3 or E5M2), round to
/// nearest even, saturating.
pub fn fp8_bits(f: Format, x: f64) -> u8 {
    let (mant, bias) = match f {
        Format::E4M3(_) => (3, 7),
        Format::E5M2(_) => (2, 15),
        _ => panic!("not fp8"),
    };
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let r = f.round(x).abs();
    let emin = 1 - bias;
    if r < 2f64.powi(emin) {
        return sign | (r * 2f64.powi(mant - emin)) as u8;
    }
    let e = ((r.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let m = (r * 2f64.powi(mant - e)) as u32 - (1 << mant);
    sign | (((e + bias) as u8) << mant) | m as u8
}

/// Value of FP8 bits in `f`.
pub fn fp8_from_bits(f: Format, b: u8) -> f64 {
    let (mant, bias) = match f {
        Format::E4M3(_) => (3, 7),
        Format::E5M2(_) => (2, 15),
        _ => panic!("not fp8"),
    };
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b & 0x7f) >> mant) as i32;
    let m = (b & ((1 << mant) - 1)) as f64;
    let one = (1u32 << mant) as f64;
    sign * if e == 0 {
        m / one * 2f64.powi(1 - bias)
    } else {
        (1.0 + m / one) * 2f64.powi(e - bias)
    }
}

/// A buffer in its block-scaled device layout: the FP8 bytes (padded to
/// 16), then the f32 scales, as little-endian 32-bit words.
pub fn pack_fp8(f: Format, x: &[f64]) -> Vec<u32> {
    let n = x.len();
    let b = f.block(n).expect("fp8");
    let s = f.scales(x);
    let mut bytes = vec![0u8; f.buffer_bytes(n).next_multiple_of(4)];
    for (i, v) in x.iter().enumerate() {
        bytes[i] = fp8_bits(f, v / s[i / b]);
    }
    let off = Format::scale_offset(n);
    for (j, sc) in s.iter().enumerate() {
        bytes[off + 4 * j..off + 4 * j + 4].copy_from_slice(&(*sc as f32).to_le_bytes());
    }
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Inverse of [`pack_fp8`]: the `n` dequantized values.
pub fn unpack_fp8(f: Format, words: &[u32], n: usize) -> Vec<f64> {
    let b = f.block(n).expect("fp8");
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let off = Format::scale_offset(n);
    (0..n)
        .map(|i| {
            let j = off + 4 * (i / b);
            let s = f32::from_le_bytes([bytes[j], bytes[j + 1], bytes[j + 2], bytes[j + 3]]);
            fp8_from_bits(f, bytes[i]) * s as f64
        })
        .collect()
}

/// IEEE binary16 bits of `x` (rounded to nearest even).
pub fn f16_bits(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    if x.is_nan() {
        return 0x7e00;
    }
    let r = Format::F16.round(x).abs();
    if r.is_infinite() {
        return sign | 0x7c00;
    }
    if r < 2f64.powi(-14) {
        // subnormal (or zero): r = m · 2⁻²⁴ exactly
        return sign | (r * 2f64.powi(24)) as u16;
    }
    let e = ((r.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let m = (r * 2f64.powi(10 - e)) as u16 - 1024;
    sign | (((e + 15) as u16) << 10) | m
}

/// Value of IEEE binary16 bits.
pub fn f16_from_bits(b: u16) -> f64 {
    let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as f64;
    sign * match e {
        0 => m * 2f64.powi(-24),
        31 if m == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// bfloat16 bits of `x` (rounded to nearest even).
pub fn bf16_bits(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    ((Format::BF16.round(x) as f32).to_bits() >> 16) as u16
}

/// Value of bfloat16 bits.
pub fn bf16_from_bits(b: u16) -> f64 {
    f32::from_bits((b as u32) << 16) as f64
}

/// Number formats of one node: `storage` for its buffer, `accum` for its
/// running reduction state (meaningless on a map), and `operand` for a
/// multiplicand the reduction computes itself rather than reads: the
/// softmax weights `exp(s − m)` of an online-softmax reduction (P in
/// attention's P·V), which a tensor-core tile rounds to its operand
/// format. `F32`: not rounded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Precision {
    pub storage: Format,
    pub accum: Format,
    pub operand: Format,
}

impl Precision {
    pub const F32: Precision = Precision {
        storage: Format::F32,
        accum: Format::F32,
        operand: Format::F32,
    };

    pub fn storage(f: Format) -> Self {
        Precision {
            storage: f,
            accum: Format::F32,
            operand: Format::F32,
        }
    }
}

/// Error of a computed result against a reference, per element, folded
/// over every output. Relative figures are relative to each output's
/// largest reference magnitude (so a zero crossing is not an infinity).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorStats {
    pub max_abs: f64,
    pub mean_abs: f64,
    pub max_rel: f64,
    pub mean_rel: f64,
}

impl ErrorStats {
    /// Compare `got` against `want`, output by output; the worst output wins.
    pub fn compare(got: &[Vec<f64>], want: &[Vec<f64>]) -> Self {
        assert_eq!(got.len(), want.len());
        let mut s = ErrorStats::default();
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.len(), w.len());
            let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
            let (mut max, mut sum) = (0.0f64, 0.0f64);
            for (a, b) in g.iter().zip(w) {
                let e = (a - b).abs();
                // NaN / ∞ in the result is an infinite error
                let e = if e.is_finite() { e } else { f64::INFINITY };
                max = max.max(e);
                sum += e;
            }
            let mean = sum / g.len().max(1) as f64;
            s.max_abs = s.max_abs.max(max);
            s.mean_abs = s.mean_abs.max(mean);
            s.max_rel = s.max_rel.max(max / scale);
            s.mean_rel = s.mean_rel.max(mean / scale);
        }
        s
    }
}

/// How much error a precision schedule may introduce.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorBudget {
    pub max_rel: f64,
    pub max_abs: f64,
}

impl ErrorBudget {
    /// A budget on relative error only.
    pub fn rel(max_rel: f64) -> Self {
        ErrorBudget {
            max_rel,
            max_abs: f64::INFINITY,
        }
    }

    pub fn admits(&self, s: &ErrorStats) -> bool {
        s.max_rel <= self.max_rel && s.max_abs <= self.max_abs
    }
}

impl TensorGraph {
    /// Error of this graph's precision schedule on `inputs`: the emulated
    /// interpreter ([`TensorGraph::eval_emulated`]) against the pure-f64
    /// one ([`TensorGraph::eval`]).
    pub fn precision_error(&self, inputs: &[Vec<f64>], outputs: &[TensorId]) -> ErrorStats {
        let want = self.eval(inputs, outputs);
        let got = self.eval_emulated(inputs, outputs);
        ErrorStats::compare(&got, &want)
    }

    /// Bytes `t`'s buffer occupies under its storage format (0 for
    /// nodes without a buffer of their own).
    pub fn buffer_bytes(&self, t: TensorId) -> usize {
        use super::{numel, Op};
        let n = &self.nodes[t.n()];
        let elems = match &n.op {
            Op::Input(_) => numel(&n.shape),
            Op::Comp(c) if !n.inline => numel(c.points()) * c.width(),
            _ => 0,
        };
        n.precision.storage.buffer_bytes(elems)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Monoid, TensorGraph};
    use tang::Scalar;

    #[test]
    fn fp8_round_trips_and_scales() {
        for f in [Format::E4M3(0), Format::E5M2(0)] {
            for b in 0u8..=255 {
                let v = fp8_from_bits(f, b);
                // E4M3 0x7f / 0xff are NaN; E5M2's top exponent is ∞/NaN
                let special = match f {
                    Format::E4M3(_) => b & 0x7f == 0x7f,
                    _ => (b >> 2) & 0x1f == 0x1f,
                };
                if special {
                    continue;
                }
                assert_eq!(f.round(v), v, "{f:?} {b:#x}");
                if v != 0.0 {
                    assert_eq!(fp8_bits(f, v), b, "{f:?} {b:#x} = {v}");
                }
            }
        }
        assert_eq!(Format::E4M3(0).round(1000.0), 448.0);
        assert_eq!(Format::E4M3(0).round(1.0 + 1.0 / 16.0), 1.0); // tie → even
        assert_eq!(Format::E4M3(0).round(2f64.powi(-9)), 2f64.powi(-9)); // min subnormal
        assert_eq!(Format::E5M2(0).round(1.0 + 1.0 / 8.0), 1.0);
        // block scales: each block's largest magnitude maps to 448
        let x: Vec<f64> = (0..256)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) * if i < 128 { 1e-3 } else { 10.0 })
            .collect();
        let mut r = x.clone();
        Format::E4M3(128).round_buf(&mut r);
        for (blk, (a, b)) in x.chunks(128).zip(r.chunks(128)).enumerate() {
            let amax = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (u, v) in a.iter().zip(b) {
                assert!((u - v).abs() <= amax / 448.0 * 16.0 + 1e-12, "block {blk}");
            }
        }
        let words = pack_fp8(Format::E4M3(128), &x);
        assert_eq!(words.len() * 4, Format::E4M3(128).buffer_bytes(256));
        let back = unpack_fp8(Format::E4M3(128), &words, 256);
        for (u, v) in back.iter().zip(&r) {
            assert!((u - v).abs() <= v.abs() * 1e-6, "{u} vs {v}");
        }
    }

    #[test]
    fn f16_known_bit_patterns() {
        let cases: &[(f64, u16)] = &[
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),                 // max finite
            (2f64.powi(-14), 0x0400),          // min normal
            (2f64.powi(-24), 0x0001),          // min subnormal
            (1023.0 * 2f64.powi(-24), 0x03ff), // max subnormal
            (1.0 / 3.0, 0x3555),
            (std::f64::consts::PI, 0x4248),
            (65520.0, 0x7c00), // halfway past max: ties to even → ∞
            (65519.0, 0x7bff),
            (1e6, 0x7c00),
            (-1e6, 0xfc00),
            (2f64.powi(-25), 0x0000), // half the min subnormal: tie → even (0)
            (3.0 * 2f64.powi(-25), 0x0002), // 1.5 ulp: tie → even (2)
            (1.0 + 2f64.powi(-11), 0x3c00), // tie between 1 and 1+2⁻¹⁰ → even
            (1.0 + 3.0 * 2f64.powi(-11), 0x3c02),
        ];
        for &(x, bits) in cases {
            assert_eq!(f16_bits(x), bits, "f16 bits of {x:e}");
            let back = f16_from_bits(bits);
            assert_eq!(Format::F16.round(x), back, "f16 round of {x:e}");
        }
        assert!(f16_from_bits(0x7e00).is_nan());
        // every finite f16 value round-trips
        for b in 0u16..0x7c00 {
            let v = f16_from_bits(b);
            assert_eq!(f16_bits(v), b);
            assert_eq!(Format::F16.round(v), v);
        }
    }

    #[test]
    fn bf16_known_bit_patterns() {
        let cases: &[(f64, u16)] = &[
            (1.0, 0x3f80),
            (-1.0, 0xbf80),
            (std::f64::consts::PI, 0x4049),
            (1.0 / 3.0, 0x3eab),
            (1.0 + 2f64.powi(-8), 0x3f80), // tie → even
            (1.0 + 3.0 * 2f64.powi(-8), 0x3f82),
            (3.0e38, 0x7f62),
            (3.4e38, 0x7f80),          // above max bf16 (3.39e38) + half ulp → ∞
            (2f64.powi(-133), 0x0001), // min subnormal
        ];
        for &(x, bits) in cases {
            assert_eq!(bf16_bits(x), bits, "bf16 bits of {x:e}");
            assert_eq!(Format::BF16.round(x), bf16_from_bits(bits));
        }
    }

    #[test]
    fn f32_round_matches_cast() {
        for x in [1.0 / 3.0, 1e-40, 1e39, -7.123456789] {
            assert_eq!(Format::F32.round(x), x as f32 as f64);
        }
    }

    /// Default precision is f32 everywhere and plain `eval` ignores the
    /// schedule entirely.
    #[test]
    fn defaults_change_nothing() {
        let mut g = TensorGraph::new();
        let x = g.input(&[4, 64]);
        let s = g.reduce(x, 1, Monoid::sum()).unwrap();
        assert!(g.nodes().iter().all(|n| n.precision == Precision::F32));
        let xs: Vec<f64> = (0..256).map(|i| (i as f64 * 0.37).sin()).collect();
        let a = g.eval(std::slice::from_ref(&xs), &[s]);
        g.set_precision(
            s,
            Precision {
                storage: Format::F16,
                accum: Format::F16,
                operand: Format::F32,
            },
        );
        assert_eq!(g.eval(std::slice::from_ref(&xs), &[s]), a);
        assert_ne!(g.eval_emulated(&[xs], &[s]), a);
    }

    /// A long sum in f16: the accumulator stops absorbing small terms
    /// once it is large (1 + 2⁻¹¹ rounds back to 1). The checker sees it.
    #[test]
    fn checker_rejects_f16_accumulation_of_a_long_sum() {
        let n = 4096;
        let mut g = TensorGraph::new();
        let x = g.input(&[1, n]);
        let s = g.reduce(x, 1, Monoid::sum()).unwrap();
        let xs = vec![(0..n)
            .map(|i| 0.01 + 0.001 * ((i % 7) as f64))
            .collect::<Vec<_>>()];
        let budget = ErrorBudget::rel(1e-2);
        // f16 storage of the input alone is fine (each term off by ≤ 2⁻¹¹)
        g.set_precision(x, Precision::storage(Format::F16));
        let st = g.precision_error(&xs, &[s]);
        assert!(budget.admits(&st), "{st:?}");
        // accumulating in f16 is not
        g.set_precision(
            s,
            Precision {
                storage: Format::F32,
                accum: Format::F16,
                operand: Format::F32,
            },
        );
        let st = g.precision_error(&xs, &[s]);
        assert!(!budget.admits(&st), "{st:?}");
        assert!(st.max_rel > 0.05, "{st:?}");
    }

    /// Emulation rounds exactly where the schedule says: an f16-stored
    /// map's output is the f16 rounding of the f64 value.
    #[test]
    fn emulated_storage_is_exact_rounding() {
        let mut g = TensorGraph::new();
        let x = g.input(&[8]);
        let y = g.map(&[x], |v| v[0] * v[0] + v[0]).unwrap();
        g.set_precision(y, Precision::storage(Format::F16));
        let xs: Vec<f64> = (0..8).map(|i| 0.1 + i as f64 * 0.731).collect();
        let want = g.eval(std::slice::from_ref(&xs), &[y]);
        let got = g.eval_emulated(&[xs], &[y]);
        for (a, b) in got[0].iter().zip(&want[0]) {
            assert_eq!(*a, Format::F16.round(*b));
        }
    }

    /// Snake `y = x + sin²(αx)/α`: `dα` sums over all of T in the
    /// backward graph. Precision set on backward nodes is honoured by the
    /// emulator and the lowering, and the forward input's f16 storage
    /// carries into the backward kernels that read it.
    #[test]
    fn backward_graph_respects_per_node_precision() {
        use crate::tensor::Op;
        let (c, t) = (2, 4096);
        let mut g = TensorGraph::new();
        let x = g.input(&[c, t]);
        let a = g.input(&[c, 1]);
        let ab = g.broadcast(a, &[c, t]).unwrap();
        let y = g
            .map(&[x, ab], |v| {
                let s = (v[1] * v[0]).sin();
                v[0] + s * s / v[1]
            })
            .unwrap();
        g.set_precision(x, Precision::storage(Format::F16));
        let grads = g.backward(y, &[x, a]).unwrap();
        let da = grads[1];
        let xs: Vec<f64> = (0..c * t)
            .map(|i| ((i * 7919) % 1000) as f64 / 500.0 - 1.0)
            .collect();
        let inputs = vec![xs, vec![0.8, 1.3]];
        let budget = ErrorBudget::rel(1e-3);
        // f16 x, f32 accumulation: within budget
        let st = g.precision_error(&inputs, &[da]);
        assert!(budget.admits(&st), "{st:?}");
        // every long reduction of the backward graph accumulating in f16
        let long: Vec<TensorId> = (0..g.nodes().len())
            .filter(|&i| matches!(&g.nodes()[i].op, Op::Comp(cc) if cc.monoid.is_some() && super::super::numel(&cc.red) >= t))
            .map(TensorId::from_index)
            .collect();
        assert!(!long.is_empty(), "dα reduces over t");
        for &r in &long {
            g.set_precision(
                r,
                Precision {
                    storage: Format::F32,
                    accum: Format::F16,
                    operand: Format::F32,
                },
            );
        }
        let st = g.precision_error(&inputs, &[da]);
        assert!(
            !budget.admits(&st),
            "f16 dα over T={t} should blow the budget: {st:?}"
        );
        // the lowering honours both: x read as packed f16, dα rounded per step
        let p = g.to_wgsl(&[y, da]);
        let xb = p.buffers.iter().find(|b| b.node == x).unwrap();
        assert_eq!(xb.format, Format::F16);
        assert!(p
            .kernels
            .iter()
            .any(|k| k.source.contains("pack2x16float(vec2<f32>(")));
        // canonical sources name read `i` `b{i}` (bindings come first)
        assert!(p
            .kernels
            .iter()
            .all(|k| match k.reads.iter().position(|r| *r == x) {
                Some(i) => k.source.contains(&format!("unpack2x16float(b{i}")),
                None => true,
            }));
    }
}
