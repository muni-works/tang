//! CPU twins of the TCX grouped MoE (`ds_moe_tcx_*_f3` in kernels.cu, `DS_MOE_TCX=1`): the
//! operand decodes and the fragment layout the kernel relies on, and the per-32-group expert
//! projection it computes, so they are checked without a GPU.
//!
//! - [`f4h`] / [`f8h`]: the kernel's integer decodes of two e2m1 / e4m3 codes into an f16x2,
//!   exact as value / 2^14 and value / 2^8 (the kernel's `POW` undoes both).
//! - [`a_cols`] / [`b_cols`]: the columns of a 32-group each lane's A (weight) and B
//!   (activation) registers carry across the group's two m16n8k16 MMAs. The kernel loads 4
//!   weight bytes at 4 tig and 8 activation bytes at 8 tig, then reorders the activation bytes
//!   with `__byte_perm`; the dot product is right only if A and B carry the same column in
//!   every k slot.
//! - [`group_dot`]: one row × one token over the K columns as the kernel sums it: each group's
//!   exact product sum (f16 operands, the MMA's f32 accumulate) times weight scale × activation
//!   scale, accumulated in group order.

use super::quant::{e4m3_to_f32, e8m0_to_f32, fp4_to_f32};

/// IEEE binary16 bits -> f32 (exact; no NaN/inf handling needed for these decodes).
#[cfg(test)]
fn h2f(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        sign * m * 2f32.powi(-24)
    } else {
        sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

/// `f4h(u)` of kernels.cu: the e2m1 codes in bits 12-15 and 28-31 of `u` as two f16 bit
/// patterns (low half, high half), each its value / 2^14.
pub fn f4h(u: u32) -> u32 {
    (u & 0x8000_8000) | ((u >> 3) & 0x0e00_0e00)
}

/// `f8h(u)` of kernels.cu: the e4m3 codes in bits 8-15 and 24-31 of `u` as two f16 bit
/// patterns, each its value / 2^8.
pub fn f8h(u: u32) -> u32 {
    (u & 0x8000_8000) | ((u >> 1) & 0x3f80_3f80)
}

/// CUDA's `__byte_perm(x, y, s)` (no sign-extension modes).
pub fn byte_perm(x: u32, y: u32, s: u32) -> u32 {
    let b = ((y as u64) << 32) | x as u64;
    (0..4).fold(0u32, |r, i| {
        let sel = (s >> (4 * i)) & 7;
        r | ((((b >> (8 * sel)) & 0xff) as u32) << (8 * i))
    })
}

/// The 16 k slots (MMA 0's 16, then MMA 1's) of lane `tig`'s A row registers within a group,
/// as (a register's half: low/high value of a0..a3 for row g) -> column of the group. The
/// kernel's A operands for row g are `f4h(wa << 12), f4h(wa << 8)` (MMA 0: a0, a2) and
/// `f4h(wa << 4), f4h(wa)` (MMA 1), `wa` the 4 bytes at 4 tig = columns 8 tig .. 8 tig + 8,
/// low nibble first.
///
/// m16n8k16 (row-major A): a0 holds k = 2 tig, 2 tig + 1; a2 holds k = 2 tig + 8, 2 tig + 9.
pub fn a_cols(tig: usize) -> [(usize, usize); 8] {
    let mut out = [(0, 0); 8];
    // (mma, k) for the 8 values this lane carries for its row
    for (m, sh) in [(0usize, [12u32, 8]), (1, [4, 0])] {
        for (r, &s) in sh.iter().enumerate() {
            // f4h takes the nibbles at bits 12-15 (low half) and 28-31 (high half) of wa << s:
            // nibble index (12 - s) / 4 and (28 - s) / 4 of wa
            let lo = (12 - s as usize) / 4;
            let hi = (28 - s as usize) / 4;
            let k0 = 2 * tig + 8 * r;
            out[m * 4 + r * 2] = (m * 16 + k0, 8 * tig + lo);
            out[m * 4 + r * 2 + 1] = (m * 16 + k0 + 1, 8 * tig + hi);
        }
    }
    out
}

/// As [`a_cols`] for lane `tig`'s B registers (token g): b0 holds k = 2 tig, 2 tig + 1 and b1
/// k = 2 tig + 8, 2 tig + 9 (col-major B). The kernel loads X = 8 bytes at 8 tig (columns
/// 8 tig .. 8 tig + 8) and feeds `f8h(byte_perm(X.x, X.y, 0x4000))`,
/// `f8h(byte_perm(.., 0x5010))` to MMA 0 and `0x6020`, `0x7030` to MMA 1.
pub fn b_cols(tig: usize) -> [(usize, usize); 8] {
    let mut out = [(0, 0); 8];
    for (m, sels) in [(0usize, [0x4000u32, 0x5010]), (1, [0x6020, 0x7030])] {
        for (r, &sel) in sels.iter().enumerate() {
            // f8h reads bytes 1 and 3 of the permuted word: selector nibbles 1 and 3
            let lo = ((sel >> 4) & 7) as usize;
            let hi = ((sel >> 12) & 7) as usize;
            let k0 = 2 * tig + 8 * r;
            out[m * 4 + r * 2] = (m * 16 + k0, 8 * tig + lo);
            out[m * 4 + r * 2 + 1] = (m * 16 + k0 + 1, 8 * tig + hi);
        }
    }
    out
}

/// One fp4 weight row (`w` packed two codes a byte along the columns, low nibble first, e8m0
/// scales `ws` a 32-group) against one fp8 activation row (`xc` e4m3 codes, pow2 scales `xs` a
/// group), summed as `ds_moe_tcx_*` does for one K slice (`s0`, step `step`): per group the
/// exact product sum, then `acc += sum * (sw * sx)` in group order.
pub fn group_dot(w: &[u8], ws: &[u8], xc: &[u8], xs: &[f32], s0: usize, step: usize) -> f32 {
    let nch = xc.len() / 32;
    let mut acc = 0f32;
    let mut g = s0;
    while g < nch {
        // the products are exact in f32 (1 + 4 significant bits); the group's 32 of them
        // summed in f64 is the MMA's (exact here) sum before its f32 rounding
        let mut sum = 0f64;
        for c in 0..32 {
            let k = g * 32 + c;
            let code = (w[k / 2] >> (4 * (k & 1))) & 15;
            sum += fp4_to_f32(code) as f64 * e4m3_to_f32(xc[k]) as f64;
        }
        acc = (sum as f32).mul_add(e8m0_to_f32(ws[g]) * xs[g], acc);
        g += step;
    }
    acc
}

/// The same row × token from dequantized values in plain column order (the reference the
/// kernel is gated against, up to summation order).
pub fn dense_dot(w: &[u8], ws: &[u8], xc: &[u8], xs: &[f32]) -> f32 {
    (0..xc.len())
        .map(|k| {
            let code = (w[k / 2] >> (4 * (k & 1))) & 15;
            let wv = fp4_to_f32(code) as f64 * e8m0_to_f32(ws[k / 32]) as f64;
            wv * e4m3_to_f32(xc[k]) as f64 * xs[k / 32] as f64
        })
        .sum::<f64>() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f4h_is_value_over_2_14() {
        for code in 0u32..16 {
            // place the code in both halves' nibble 3 (bits 12-15, 28-31)
            let u = (code << 12) | (code << 28);
            let h = f4h(u);
            let want = fp4_to_f32(code as u8) / 16384.0;
            assert_eq!(h2f(h as u16), want, "code {code} low");
            assert_eq!(h2f((h >> 16) as u16), want, "code {code} high");
        }
    }

    #[test]
    fn f8h_is_value_over_2_8() {
        for v in 0u32..256 {
            if v & 0x7f == 0x7f {
                continue; // NaN
            }
            let u = (v << 8) | (v << 24);
            let h = f8h(u);
            let want = e4m3_to_f32(v as u8) / 256.0;
            assert_eq!(h2f(h as u16), want, "e4m3 {v:#x} low");
            assert_eq!(h2f((h >> 16) as u16), want, "e4m3 {v:#x} high");
        }
    }

    /// Every k slot of both MMAs carries the same group column in A and B, and the 4 lanes
    /// of a row cover the group's 32 columns once each.
    #[test]
    fn fragments_agree_and_cover_the_group() {
        let mut seen = [0u32; 32];
        let mut slot = [usize::MAX; 32];
        for tig in 0..4 {
            let (a, b) = (a_cols(tig), b_cols(tig));
            for i in 0..8 {
                assert_eq!(a[i].0, b[i].0, "tig {tig} slot order");
                assert_eq!(a[i].1, b[i].1, "tig {tig} k {} column", a[i].0);
                seen[a[i].1] += 1;
                assert_eq!(slot[a[i].0], usize::MAX, "k slot {} twice", a[i].0);
                slot[a[i].0] = a[i].1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "columns {seen:?}");
    }

    #[test]
    fn byte_perm_matches_cuda() {
        let (x, y) = (0x0302_0100u32, 0x0706_0504u32);
        assert_eq!(byte_perm(x, y, 0x4000), 0x0400_0000);
        assert_eq!(byte_perm(x, y, 0x5010), 0x0500_0100);
        assert_eq!(byte_perm(x, y, 0x7030), 0x0700_0300);
    }

    fn lcg(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    /// The group-ordered, K-sliced sum against the dense dequantized dot: within f32
    /// rounding of the K-length sum, for every S the kernel uses.
    #[test]
    fn group_dot_matches_dense() {
        let mut seed = 7u64;
        for &k in &[256usize, 2048, 4096] {
            let w: Vec<u8> = (0..k / 2).map(|_| lcg(&mut seed) as u8).collect();
            // weight scales around 2^-8, activation codes not NaN, pow2 activation scales
            let ws: Vec<u8> = (0..k / 32)
                .map(|_| 119 + (lcg(&mut seed) % 4) as u8)
                .collect();
            let xc: Vec<u8> = (0..k)
                .map(|_| {
                    let v = lcg(&mut seed) as u8;
                    if v & 0x7f == 0x7f {
                        0
                    } else {
                        v
                    }
                })
                .collect();
            let xs: Vec<f32> = (0..k / 32)
                .map(|_| 2f32.powi(-((lcg(&mut seed) % 6) as i32) - 6))
                .collect();
            let want = dense_dot(&w, &ws, &xc, &xs);
            let mag: f64 = (0..k)
                .map(|i| {
                    let code = (w[i / 2] >> (4 * (i & 1))) & 15;
                    (fp4_to_f32(code) * e8m0_to_f32(ws[i / 32]) * e4m3_to_f32(xc[i]) * xs[i / 32])
                        .abs() as f64
                })
                .sum();
            for s in [1usize, 2, 4, 8] {
                // the S slices summed in slice order, as the kernel's shared-memory reduce
                let got = (0..s).fold(0f32, |a, si| a + group_dot(&w, &ws, &xc, &xs, si, s));
                let err = (got - want).abs() as f64;
                assert!(
                    err <= 1e-6 * mag + 1e-30,
                    "k {k} S {s}: {got} vs {want} (|x| sum {mag})"
                );
            }
        }
    }
}
