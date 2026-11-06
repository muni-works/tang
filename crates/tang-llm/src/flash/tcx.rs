//! TCX: routed experts repacked from GGUF Q4_K / Q5_K into a tensor-core-native layout
//! (`TANG_CODER_TCX=1`), for `cx_moe_*_t` (int8 `mma.sync m16n8k32`).
//!
//! A matrix `[rows][k]` (rows % 16 == 0, k % 256 == 0) becomes tiles of 16 rows × 256 columns
//! (one superblock), ordered `[row tile][superblock]`, each tile:
//!
//! - codes, 2048 bytes: `[q 0..4][row 0..16][32 bytes]`, the GGUF `qs[32 q ..][..32]` of each
//!   row as they are. The m16n8k32 A-fragment register `c` of lane (g = lane / 4,
//!   tig = lane % 4) for steps 2q (low nibbles) and 2q + 1 (high) is the word at
//!   `(q, g + 8 (c & 1), 16 (c >> 1) + 4 tig)`: codes at row `g + 8 (c & 1)`, columns
//!   `32 s + 16 (c >> 1) + 4 tig + b`. The per-row dot reads its 16 bytes unchanged.
//! - Q5_K only, 512 bytes: the GGUF `qh` (32 bytes) of each of the 16 rows.
//! - scales, 320 bytes: `[16 rows] u32` (d | dmin << 16, the GGUF f16s) then
//!   `[16 rows][s0..s8, m0..m8]` (the unpacked 6-bit sub-block scales and mins).
//!
//! The values are the GGUF's exactly: `d * s_j * q - dmin * m_j`.

use anyhow::{ensure, Result};

pub const TY_Q4: i32 = 112;
pub const TY_Q5: i32 = 113;

/// Bytes of one 16 × 256 tile.
pub fn tile_bytes(q5: bool) -> usize {
    2048 + if q5 { 512 } else { 0 } + 320
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// One GGUF superblock's d, dmin, scales, mins and 256 codes (0..15 or 0..31).
fn decode_sb(q5: bool, b: &[u8]) -> (u16, u16, [u8; 8], [u8; 8], [u8; 256]) {
    let d = u16::from_le_bytes([b[0], b[1]]);
    let dm = u16::from_le_bytes([b[2], b[3]]);
    let sc = &b[4..16];
    let (mut s, mut m) = ([0u8; 8], [0u8; 8]);
    for j in 0..8 {
        (s[j], m[j]) = scale_min_k4(j, sc);
    }
    let mut q = [0u8; 256];
    let (qh, qs) = if q5 { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
    for j64 in 0..4 {
        for l in 0..32 {
            let v = qs[32 * j64 + l];
            let (mut lo, mut hi) = (v & 15, v >> 4);
            if q5 {
                lo |= ((qh[l] >> (2 * j64)) & 1) << 4;
                hi |= ((qh[l] >> (2 * j64 + 1)) & 1) << 4;
            }
            q[64 * j64 + l] = lo;
            q[64 * j64 + 32 + l] = hi;
        }
    }
    (d, dm, s, m, q)
}

/// Repack `ne` matrices of `rows × k` GGUF Q4_K (`q5` false) / Q5_K blocks.
pub fn repack(q5: bool, src: &[u8], ne: usize, rows: usize, k: usize) -> Result<Vec<u8>> {
    let bb = if q5 { 176 } else { 144 };
    let nsb = k / 256;
    ensure!(rows % 16 == 0 && k % 256 == 0, "tcx: {rows} x {k} isn't 16 x 256 tiled");
    ensure!(src.len() == ne * rows * nsb * bb, "tcx: {} bytes, want {}", src.len(), ne * rows * nsb * bb);
    let tb = tile_bytes(q5);
    let per = (rows / 16) * nsb * tb;
    let mut out = vec![0u8; ne * per];
    // Experts in parallel (one chunk of experts a thread).
    let nt = std::thread::available_parallelism().map_or(8, |n| n.get()).min(ne.max(1));
    let chunk = ne.div_ceil(nt);
    std::thread::scope(|sc| {
        for (ci, oc) in out.chunks_mut(chunk * per).enumerate() {
            sc.spawn(move || repack_into(q5, src, ci * chunk, oc, rows, k));
        }
    });
    Ok(out)
}

fn repack_into(q5: bool, src: &[u8], e0: usize, out: &mut [u8], rows: usize, k: usize) {
    let bb = if q5 { 176 } else { 144 };
    let nsb = k / 256;
    let tb = tile_bytes(q5);
    let per = (rows / 16) * nsb * tb;
    let mut dec = vec![(0u16, 0u16, [0u8; 8], [0u8; 8], [0u8; 256]); 16];
    for el in 0..out.len() / per {
        let e = e0 + el;
        for rt in 0..rows / 16 {
            for sb in 0..nsb {
                for (r, dr) in dec.iter_mut().enumerate() {
                    let row = e * rows + rt * 16 + r;
                    let o = (row * nsb + sb) * bb;
                    *dr = decode_sb(q5, &src[o..o + bb]);
                }
                let t = &mut out[((el * (rows / 16) + rt) * nsb + sb) * tb..][..tb];
                for lane in 0..32 {
                    let (g, tig) = (lane / 4, lane % 4);
                    for c in 0..4 {
                        let row = g + 8 * (c & 1);
                        for qp in 0..4 {
                            for b in 0..4 {
                                let col = |s: usize| 32 * s + 16 * (c >> 1) + 4 * tig + b;
                                let lo = dec[row].4[col(2 * qp)];
                                let hi = dec[row].4[col(2 * qp + 1)];
                                t[(qp * 16 + row) * 32 + 16 * (c >> 1) + 4 * tig + b] = (lo & 15) | ((hi & 15) << 4);
                            }
                        }
                    }
                }
                if q5 {
                    for r in 0..16 {
                        let row = e * rows + rt * 16 + r;
                        let o = (row * nsb + sb) * bb;
                        t[2048 + r * 32..2048 + r * 32 + 32].copy_from_slice(&src[o + 16..o + 48]);
                    }
                }
                let so = 2048 + if q5 { 512 } else { 0 };
                for (r, dr) in dec.iter().enumerate() {
                    let w = dr.0 as u32 | (dr.1 as u32) << 16;
                    t[so + r * 4..so + r * 4 + 4].copy_from_slice(&w.to_le_bytes());
                    t[so + 64 + r * 16..so + 64 + r * 16 + 8].copy_from_slice(&dr.2);
                    t[so + 64 + r * 16 + 8..so + 64 + r * 16 + 16].copy_from_slice(&dr.3);
                }
            }
        }
    }
}

/// Dequantize a repacked matrix set back to f32 (`[ne][rows][k]`), reading the tile layout as
/// the kernel does. For tests.
pub fn dequant(q5: bool, t: &[u8], ne: usize, rows: usize, k: usize) -> Vec<f32> {
    use crate::gguf::f16_to_f32;
    let (nsb, tb) = (k / 256, tile_bytes(q5));
    let mut out = vec![0f32; ne * rows * k];
    for e in 0..ne {
        for rt in 0..rows / 16 {
            for sb in 0..nsb {
                let tl = &t[((e * (rows / 16) + rt) * nsb + sb) * tb..][..tb];
                let so = 2048 + if q5 { 512 } else { 0 };
                for lane in 0..32 {
                    let (g, tig) = (lane / 4, lane % 4);
                    for qp in 0..4 {
                        for c in 0..4 {
                            let row = g + 8 * (c & 1);
                            let w = u32::from_le_bytes(tl[(qp * 16 + row) * 32 + 16 * (c >> 1) + 4 * tig..][..4].try_into().unwrap());
                            for (s, word) in [(2 * qp, w & 0x0F0F0F0F), (2 * qp + 1, (w >> 4) & 0x0F0F0F0F)] {
                                // qh bytes l = 16 (c >> 1) + 4 tig + b, bit s (the kernel's word load).
                                let hw = if q5 {
                                    u32::from_le_bytes(tl[2048 + row * 32 + 16 * (c >> 1) + 4 * tig..][..4].try_into().unwrap())
                                } else {
                                    0
                                };
                                let word = word | ((hw >> s) & 0x01010101) << 4;
                                let dw = u32::from_le_bytes(tl[so + row * 4..][..4].try_into().unwrap());
                                let (d, dm) = (f16_to_f32(dw as u16), f16_to_f32((dw >> 16) as u16));
                                let (sj, mj) = (tl[so + 64 + row * 16 + s], tl[so + 64 + row * 16 + 8 + s]);
                                for b in 0..4 {
                                    let q = (word >> (8 * b)) & 0xFF;
                                    let col = sb * 256 + 32 * s + 16 * (c >> 1) + 4 * tig + b;
                                    let v = d * sj as f32 * q as f32 - dm * mj as f32;
                                    out[(e * rows + rt * 16 + row) * k + col] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{dequantize, GgmlType};

    fn blocks(q5: bool, n: usize, seed: u64) -> Vec<u8> {
        let bb = if q5 { 176 } else { 144 };
        let mut x = seed;
        let mut v: Vec<u8> = (0..n * bb)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        for b in v.chunks_mut(bb) {
            // finite f16 d / dmin around 1e-3
            b[1] = 0x14 | (b[1] & 3);
            b[3] = 0x14 | (b[3] & 3);
        }
        v
    }

    #[test]
    fn repack_dequant_exact() {
        for q5 in [false, true] {
            let (ne, rows, k) = (2, 32, 512);
            let src = blocks(q5, ne * rows * k / 256, 0x9e3779b97f4a7c15 + q5 as u64);
            let mut want = vec![0f32; ne * rows * k];
            dequantize(if q5 { GgmlType::Q5K } else { GgmlType::Q4K }, &src, &mut want).unwrap();
            let t = repack(q5, &src, ne, rows, k).unwrap();
            assert_eq!(t.len(), ne * rows / 16 * k / 256 * tile_bytes(q5));
            // The per-row dot's view: words (q, c, 4g ..) are the GGUF qs bytes as they are.
            let (bb, tb, nsb) = (if q5 { 176 } else { 144 }, tile_bytes(q5), k / 256);
            for row in 0..ne * rows {
                let (e, r) = (row / rows, row % rows);
                for sb in 0..nsb {
                    let tl = &t[((e * (rows / 16) + r / 16) * nsb + sb) * tb..][..tb];
                    let g = &src[(row * nsb + sb) * bb..][..bb];
                    let qs = &g[if q5 { 48 } else { 16 }..];
                    for q in 0..4 {
                        for h in 0..2 {
                            let o = (q * 16 + r % 16) * 32 + 16 * h;
                            assert_eq!(&tl[o..o + 16], &qs[32 * q + 16 * h..][..16]);
                        }
                    }
                }
            }
            let got = dequant(q5, &t, ne, rows, k);
            for i in 0..want.len() {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "q5={q5} at {i}: {} vs {}", got[i], want[i]);
            }
        }
    }
}
