//! GGUF block formats on the CPU: the values a graph input stored as
//! [`Format::Q8_0`], [`Format::Q4K`], [`Format::Q5K`] or [`Format::Q6K`]
//! (or as raw bf16 / f16) stands for. The graph and the interpreter work
//! on these values; device buffers hold the bytes as they are. The
//! arithmetic is ggml's (and `tang_llm::gguf`'s), operation for operation,
//! so the values are bit-identical.

use super::Format;

/// Elements and bytes of one block of `f`, for the GGUF block formats.
pub fn block(f: Format) -> Option<(usize, usize)> {
    match f {
        Format::Q8_0 => Some((32, 34)),
        Format::Q4K => Some((256, 144)),
        Format::Q5K => Some((256, 176)),
        Format::Q6K => Some((256, 210)),
        Format::F16 | Format::BF16 => Some((1, 2)),
        _ => None,
    }
}

/// The f32 value of IEEE binary16 bits (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn half(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize `src` (whole blocks of `f`) into `out`. `None` for a format
/// that is not a GGUF block format, or a length that does not match.
pub fn dequantize(f: Format, src: &[u8], out: &mut [f32]) -> Option<()> {
    let (per, bytes) = block(f)?;
    if !out.len().is_multiple_of(per) || src.len() != out.len() / per * bytes {
        return None;
    }
    let blocks = src.chunks_exact(bytes).zip(out.chunks_exact_mut(per));
    match f {
        Format::F16 => {
            for (b, y) in blocks {
                y[0] = half(b, 0);
            }
        }
        Format::BF16 => {
            for (b, y) in blocks {
                y[0] = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
            }
        }
        Format::Q8_0 => {
            for (b, y) in blocks {
                let d = half(b, 0);
                for j in 0..32 {
                    y[j] = d * (b[2 + j] as i8) as f32;
                }
            }
        }
        Format::Q4K | Format::Q5K => {
            let five = f == Format::Q5K;
            for (b, y) in blocks {
                let d = half(b, 0);
                let min = half(b, 2);
                let sc = &b[4..16];
                let (qh, qs) = if five {
                    (&b[16..48], 48)
                } else {
                    (&b[0..0], 16)
                };
                let (mut u1, mut u2) = (1u8, 2u8);
                for j in 0..4 {
                    let q = &b[qs + j * 32..qs + j * 32 + 32];
                    let (s1, m1) = scale_min_k4(2 * j, sc);
                    let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                    let (d1, mm1) = (d * s1 as f32, min * m1 as f32);
                    let (d2, mm2) = (d * s2 as f32, min * m2 as f32);
                    for l in 0..32 {
                        let (h1, h2) = if five {
                            (
                                if qh[l] & u1 != 0 { 16 } else { 0 },
                                if qh[l] & u2 != 0 { 16 } else { 0 },
                            )
                        } else {
                            (0, 0)
                        };
                        y[j * 64 + l] = d1 * ((q[l] & 0xf) + h1) as f32 - mm1;
                        y[j * 64 + 32 + l] = d2 * ((q[l] >> 4) + h2) as f32 - mm2;
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        Format::Q6K => {
            for (b, y) in blocks {
                let d = half(b, 208);
                for n in 0..2 {
                    let ql = &b[n * 64..n * 64 + 64];
                    let qh = &b[128 + n * 32..128 + n * 32 + 32];
                    let sc = &b[192 + n * 8..192 + n * 8 + 8];
                    let yy = &mut y[n * 128..n * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        yy[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        yy[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        yy[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        yy[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        _ => return None,
    }
    Some(())
}
