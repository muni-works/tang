//! The number formats of DeepSeek-V4.1 and the quant-dequant steps its forward applies to
//! activations, matching `deepseek_ref.py` (and DeepSeek's tilelang kernels) bit for bit.

pub const FP8_MAX: f32 = 448.0;
pub const FP4_MAX: f32 = 6.0;
/// e2m1 magnitudes by code (bit 3 is the sign).
pub const FP4_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub fn fp4_to_f32(code: u8) -> f32 {
    let m = FP4_GRID[(code & 7) as usize];
    if code & 8 != 0 {
        -m
    } else {
        m
    }
}

/// OCP e4m3fn: bias 7, no infinities, 0x7f / 0xff are NaN.
pub fn e4m3_to_f32(v: u8) -> f32 {
    let sign = if v & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((v >> 3) & 0xF) as i32;
    let m = (v & 7) as f32;
    if e == 15 && v & 7 == 7 {
        return f32::NAN;
    }
    if e == 0 {
        sign * m * 2f32.powi(-9)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// [`e4m3_to_f32`] of every byte (the same values, looked up).
pub fn e4m3_lut() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|i| e4m3_to_f32(i as u8)))
}

/// e8m0: a bare exponent, bias 127 (0xff is NaN).
pub fn e8m0_to_f32(v: u8) -> f32 {
    if v == 0xFF {
        return f32::NAN;
    }
    f32::from_bits((v as u32) << 23).max(f32::from_bits(0x0040_0000) * (v == 0) as u8 as f32)
}

/// `x` rounded to bf16 (nearest, ties to even), back as f32.
pub fn bf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    f32::from_bits(r & 0xFFFF_0000)
}

pub fn bf_all(v: &mut [f32]) {
    for x in v {
        *x = bf(*x);
    }
}

/// 2^ceil(log2(x)) for positive normal x (the kernels' `fast_round_scale`).
pub fn pow2_ceil(x: f32) -> f32 {
    let b = x.to_bits();
    let e = (b >> 23) & 0xFF;
    let bump = (b & 0x7F_FFFF != 0) as u32;
    f32::from_bits((e + bump) << 23)
}

/// Nearest e4m3fn value, ties to even (the cast after the kernels' clamp to ±448).
pub fn round_e4m3(x: f32) -> f32 {
    let a = x.abs();
    if a == 0.0 {
        return 0.0 * x.signum();
    }
    let step = if a >= 2f32.powi(-6) {
        // three mantissa bits in this binade
        let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
        2f32.powi(e - 3)
    } else {
        2f32.powi(-9)
    };
    ((a / step).round_ties_even() * step)
        .min(FP8_MAX)
        .copysign(x)
}

/// Nearest e2m1 value, ties to the even code.
pub fn round_fp4(x: f32) -> f32 {
    let a = x.abs().min(FP4_MAX);
    let mut best = 0;
    for c in 1..8 {
        let (d, db) = ((a - FP4_GRID[c]).abs(), (a - FP4_GRID[best]).abs());
        if d < db || (d == db && c % 2 == 0 && best % 2 == 1) {
            best = c;
        }
    }
    FP4_GRID[best].copysign(x)
}

/// Activation fp8 quant-dequant per `block` with power-of-two scales (`act_quant(...,
/// ue8m0, inplace=True)`), applied to every row of `v` (len a multiple of `block`).
pub fn fp8_qdq(v: &mut [f32], block: usize) {
    for g in v.chunks_mut(block) {
        let amax = g.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-4);
        let s = pow2_ceil(amax / FP8_MAX);
        for x in g {
            *x = bf(round_e4m3((*x / s).clamp(-FP8_MAX, FP8_MAX)) * s);
        }
    }
}

/// fp4 quant-dequant per `block`: power-of-two scales (index keys and queries, block 32), or
/// e4m3 scales (compressed KV, block 16).
pub fn fp4_qdq(v: &mut [f32], block: usize, e4m3_scale: bool) {
    for g in v.chunks_mut(block) {
        let amax = g.iter().fold(0f32, |m, x| m.max(x.abs()));
        let s = if e4m3_scale {
            round_e4m3(amax.max(6.0 * 2f32.powi(-9)) / FP4_MAX)
        } else {
            pow2_ceil(amax.max(6.0 * 2f32.powi(-126)) / FP4_MAX)
        };
        for x in g {
            *x = bf(round_fp4((*x / s).clamp(-FP4_MAX, FP4_MAX)) * s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip() {
        for v in 0u8..=255 {
            let f = e4m3_to_f32(v);
            if f.is_nan() {
                continue;
            }
            assert_eq!(round_e4m3(f), f, "code {v:#x}");
        }
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
    }

    #[test]
    fn e8m0_values() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(120), 2f32.powi(-7));
        assert_eq!(e8m0_to_f32(130), 8.0);
    }

    #[test]
    fn fp4_ties_go_to_even_codes() {
        assert_eq!(round_fp4(0.25), 0.0);
        assert_eq!(round_fp4(0.75), 1.0);
        assert_eq!(round_fp4(1.25), 1.0);
        assert_eq!(round_fp4(1.75), 2.0);
        assert_eq!(round_fp4(2.5), 2.0);
        assert_eq!(round_fp4(3.5), 4.0);
        assert_eq!(round_fp4(5.0), 4.0);
        assert_eq!(round_fp4(-5.5), -6.0);
    }

    #[test]
    fn pow2_scales() {
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(1.5), 2.0);
        assert_eq!(pow2_ceil(0.3), 0.5);
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(bf(1.0), 1.0);
        assert_eq!(bf(1.0 + 2f32.powi(-8)), 1.0); // tie to even
        assert_eq!(bf(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
    }
}
