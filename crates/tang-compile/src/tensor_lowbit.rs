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
    assert!(cols % group == 0 && group % dim == 0);
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
    assert!(b.is_power_of_two() && cols % b == 0 && signs.len() == cols);
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
}
