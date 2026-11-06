//! CPU reference for the TCX expert math in moe_tcx.cuh (`g5x_rows`, `g5x_plan`): the k-permuted
//! m16n8k32 fragments, the 6-bit scale/min unpack, the mins MMA and the two-level (A2)
//! activations, emulated lane by lane from the PTX fragment layout and checked against
//! `gguf::dequantize` dot products. Runs on any host (no GPU).

use crate::gguf::{dequantize, f32_to_f16, GgmlType};

/// xorshift for fixtures.
struct Rng(u64);
impl Rng {
    fn u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn f(&mut self) -> f32 {
        (self.u32() % 20001) as f32 / 10000.0 - 1.0
    }
}

/// A random Q4_K (ty 12) / Q5_K (13) row of `k` weights with sane d / dmin.
fn rand_row(ty: u32, k: usize, r: &mut Rng) -> Vec<u8> {
    let bb = if ty == 12 { 144 } else { 176 };
    let mut v = vec![0u8; k / 256 * bb];
    for b in v.chunks_mut(bb) {
        for x in b.iter_mut() {
            *x = r.u32() as u8;
        }
        b[0..2].copy_from_slice(&f32_to_f16(0.01 + 0.02 * r.f().abs()).to_le_bytes());
        b[2..4].copy_from_slice(&f32_to_f16(0.005 + 0.01 * r.f().abs()).to_le_bytes());
    }
    v
}

/// g5f_q8: codes, residual codes, per-32 scales.
fn q8(x: &[f32]) -> (Vec<i8>, Vec<i8>, Vec<f32>) {
    let (mut q, mut q2, mut d) = (Vec::new(), Vec::new(), Vec::new());
    for c in x.chunks(32) {
        let amax = c.iter().fold(0f32, |a, v| a.max(v.abs()));
        let dd = amax / 127.0;
        for &v in c {
            let qi = if dd == 0.0 {
                0
            } else {
                (v / dd).round().clamp(-127.0, 127.0) as i32
            };
            let rr = if dd == 0.0 {
                0.0
            } else {
                (v - dd * qi as f32) * 254.0 / dd
            };
            q.push(qi as i8);
            q2.push(rr.round().clamp(-127.0, 127.0) as i8);
        }
        d.push(dd);
    }
    (q, q2, d)
}

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn bperm(x: u32, sel_byte: u32) -> u32 {
    (x >> (8 * sel_byte)) & 0xFF
}
fn h2f(h: u16) -> f32 {
    let mut out = [0f32; 1];
    // f16 via the crate's F16 dequant
    dequantize(GgmlType::F16, &h.to_le_bytes(), &mut out).unwrap();
    out[0]
}

/// m16n8k32 s8 MMA from per-lane fragments (PTX layout): a[lane][4], b[lane][2] -> d[lane][4].
fn mma(a: &[[u32; 4]; 32], b: &[[u32; 2]; 32]) -> [[i32; 4]; 32] {
    let byte = |w: u32, i: usize| (w >> (8 * i)) as u8 as i8 as i32;
    let mut am = [[0i32; 32]; 16];
    let mut bm = [[0i32; 8]; 32];
    for lane in 0..32 {
        let (g, tig) = (lane / 4, lane % 4);
        for i in 0..4 {
            am[g][4 * tig + i] = byte(a[lane][0], i);
            am[g + 8][4 * tig + i] = byte(a[lane][1], i);
            am[g][16 + 4 * tig + i] = byte(a[lane][2], i);
            am[g + 8][16 + 4 * tig + i] = byte(a[lane][3], i);
            bm[4 * tig + i][g] = byte(b[lane][0], i);
            bm[16 + 4 * tig + i][g] = byte(b[lane][1], i);
        }
    }
    let dot = |r: usize, c: usize| (0..32).map(|k| am[r][k] * bm[k][c]).sum::<i32>();
    let mut d = [[0i32; 4]; 32];
    for lane in 0..32 {
        let (g, tig) = (lane / 4, lane % 4);
        d[lane] = [
            dot(g, 2 * tig),
            dot(g, 2 * tig + 1),
            dot(g + 8, 2 * tig),
            dot(g + 8, 2 * tig + 1),
        ];
    }
    d
}

/// `g5x_rows` for a 16-row tile against m <= 8 tokens: out[row][token].
fn tile(
    ty: u32,
    rows: &[Vec<u8>],
    xs: &[(Vec<i8>, Vec<i8>, Vec<f32>)],
    k: usize,
    a2: bool,
) -> Vec<[f32; 8]> {
    let bb = if ty == 12 { 144 } else { 176 };
    let qs = if ty == 12 { 16 } else { 48 };
    let m = xs.len();
    let mut acc = [[0f32; 4]; 32];
    let tok = |n: usize| &xs[n.min(m - 1)];
    let xw = |c: &[i8], o: usize| {
        u32::from_le_bytes([c[o] as u8, c[o + 1] as u8, c[o + 2] as u8, c[o + 3] as u8])
    };
    for sb in 0..k / 256 {
        for s in 0..8 {
            let (mut a, mut am, mut b, mut b2) = (
                [[0u32; 4]; 32],
                [[0u32; 4]; 32],
                [[0u32; 2]; 32],
                [[0u32; 2]; 32],
            );
            for lane in 0..32 {
                let (g, tig) = (lane / 4, lane % 4);
                for c in 0..4 {
                    let blk = &rows[g + 8 * (c & 1)][sb * bb..];
                    let w = u32le(blk, qs + (s >> 1) * 32 + tig * 8 + 4 * (c >> 1));
                    let mut v = if s & 1 == 1 {
                        (w >> 4) & 0x0F0F0F0F
                    } else {
                        w & 0x0F0F0F0F
                    };
                    if ty == 13 {
                        v |= ((u32le(blk, 16 + tig * 8 + 4 * (c >> 1)) >> s) & 0x01010101) << 4;
                    }
                    a[lane][c] = v;
                    let sc = [u32le(blk, 4), u32le(blk, 8), u32le(blk, 12)];
                    let m47 = ((sc[2] >> 4) & 0x0F0F0F0F) | (((sc[1] >> 6) & 0x03030303) << 4);
                    let mw = if s < 4 { sc[1] & 0x3F3F3F3F } else { m47 };
                    am[lane][c] = bperm(mw, (s & 3) as u32) * 0x01010101;
                }
                let t = tok(g);
                let o = sb * 256 + s * 32 + tig * 8;
                b[lane] = [xw(&t.0, o), xw(&t.0, o + 4)];
                b2[lane] = [xw(&t.1, o), xw(&t.1, o + 4)];
            }
            let (d, mm) = (mma(&a, &b), mma(&am, &b));
            let (d2, mm2) = (mma(&a, &b2), mma(&am, &b2));
            for lane in 0..32 {
                let (g, tig) = (lane / 4, lane % 4);
                for c in 0..4 {
                    let row = g + 8 * (c >> 1);
                    let n = 2 * tig + (c & 1);
                    if n >= m {
                        continue;
                    }
                    let blk = &rows[row][sb * bb..];
                    let sc = [u32le(blk, 4), u32le(blk, 8), u32le(blk, 12)];
                    let s47 = (sc[2] & 0x0F0F0F0F) | (((sc[0] >> 6) & 0x03030303) << 4);
                    let sw = if s < 4 { sc[0] & 0x3F3F3F3F } else { s47 };
                    let scale = bperm(sw, (s & 3) as u32) as f32;
                    let dw = h2f(u16::from_le_bytes([blk[0], blk[1]]));
                    let dm = h2f(u16::from_le_bytes([blk[2], blk[3]]));
                    let (mut fd, mut fm) = (d[lane][c] as f32, mm[lane][c] as f32);
                    if a2 {
                        fd += d2[lane][c] as f32 / 254.0;
                        fm += mm2[lane][c] as f32 / 254.0;
                    }
                    let dx = xs[n].2[sb * 8 + s];
                    acc[lane][c] += dx * (dw * scale * fd - dm * fm);
                }
            }
        }
    }
    let mut out = vec![[0f32; 8]; 16];
    for lane in 0..32 {
        let (g, tig) = (lane / 4, lane % 4);
        for c in 0..4 {
            let n = 2 * tig + (c & 1);
            if n < m {
                out[g + 8 * (c >> 1)][n] = acc[lane][c];
            }
        }
    }
    out
}

fn check(ty: u32, m: usize, a2: bool, seed: u64) {
    let k = 512;
    let mut r = Rng(seed);
    let rows: Vec<Vec<u8>> = (0..16).map(|_| rand_row(ty, k, &mut r)).collect();
    let xf: Vec<Vec<f32>> = (0..m).map(|_| (0..k).map(|_| r.f()).collect()).collect();
    let xs: Vec<_> = xf.iter().map(|x| q8(x)).collect();
    let got = tile(ty, &rows, &xs, k, a2);
    let gt = if ty == 12 {
        GgmlType::Q4K
    } else {
        GgmlType::Q5K
    };
    for (ri, row) in rows.iter().enumerate() {
        let mut w = vec![0f32; k];
        dequantize(gt, row, &mut w).unwrap();
        for n in 0..m {
            // the quantized activations' exact value, so only float rounding is left
            let (q, q2, d) = &xs[n];
            let xa: Vec<f32> = (0..k)
                .map(|i| d[i / 32] * (q[i] as f32 + if a2 { q2[i] as f32 / 254.0 } else { 0.0 }))
                .collect();
            let want: f32 = w.iter().zip(&xa).map(|(a, b)| a * b).sum();
            let tol = 1e-4
                * w.iter()
                    .zip(&xa)
                    .map(|(a, b)| (a * b).abs())
                    .sum::<f32>()
                    .max(1.0);
            assert!(
                (got[ri][n] - want).abs() <= tol,
                "ty {ty} m {m} a2 {a2} row {ri} tok {n}: {} vs {want}",
                got[ri][n]
            );
            // and the A2 dot tracks the f32 activations to the q8x2 band
            if a2 {
                let wf: f32 = w.iter().zip(&xf[n]).map(|(a, b)| a * b).sum();
                assert!(
                    (got[ri][n] - wf).abs() <= 1e-3 * tol / 1e-4,
                    "a2 vs f32: {} vs {wf}",
                    got[ri][n]
                );
            }
        }
    }
}

#[test]
fn tcx_q4k_matches_dequant() {
    for (m, a2) in [(1, false), (3, true), (8, false), (8, true)] {
        check(12, m, a2, 7 + m as u64);
    }
}

#[test]
fn tcx_q5k_matches_dequant() {
    for (m, a2) in [(1, true), (5, false), (8, true)] {
        check(13, m, a2, 99 + m as u64);
    }
}

/// `g5x_plan` on the host: same keys, ranks and group starts.
fn plan(ids: &[u32], k: usize, ne: usize, s: usize, n: usize) -> Vec<u32> {
    let key: Vec<usize> = (0..n)
        .map(|i| {
            let (t, j) = (i / s, i % s);
            if j < k {
                let e = ids[t * k + j] as usize;
                if e >= ne {
                    ne + 1
                } else {
                    e
                }
            } else {
                ne
            }
        })
        .collect();
    let mut out = vec![0u32; 2 + 1024 + n];
    let mut rank = vec![0usize; n];
    let mut start = vec![false; n];
    for i in 0..n {
        let lt = key.iter().filter(|&&q| q < key[i]).count();
        let pos = (0..i).filter(|&q| key[q] == key[i]).count();
        rank[i] = lt + pos;
        start[i] = pos % 8 == 0;
        out[2 + 1024 + rank[i]] = i as u32;
    }
    for i in (0..n).filter(|&i| start[i]) {
        let gid = (0..n).filter(|&q| start[q] && rank[q] < rank[i]).count();
        out[1 + gid] = rank[i] as u32;
    }
    let ng = start.iter().filter(|&&b| b).count();
    out[0] = ng as u32;
    out[1 + ng] = n as u32;
    out
}

#[test]
fn tcx_plan_groups() {
    let (k, ne, s) = (8, 16, 9);
    let mut r = Rng(3);
    for t in [1usize, 4, 40] {
        let ids: Vec<u32> = (0..t * k).map(|_| r.u32() % (ne as u32 + 2)).collect();
        let n = t * s;
        let p = plan(&ids, k, ne, s, n);
        let ng = p[0] as usize;
        assert!(
            ng <= super::tcx_groups_cpu(n, ne),
            "t {t}: {ng} groups > grid"
        );
        let mut seen = vec![false; n];
        for gi in 0..ng {
            let (a, b) = (p[1 + gi] as usize, p[2 + gi] as usize);
            assert!(b > a && b - a <= 8, "group {gi}: {a}..{b}");
            let k0 = |i: usize| {
                let pi = p[2 + 1024 + i] as usize;
                let (tt, j) = (pi / s, pi % s);
                if j < k {
                    let e = ids[tt * k + j];
                    if e >= ne as u32 {
                        ne as u32 + 1
                    } else {
                        e
                    }
                } else {
                    ne as u32
                }
            };
            for i in a..b {
                assert_eq!(k0(i), k0(a), "group {gi} mixes experts");
                let pi = p[2 + 1024 + i] as usize;
                assert!(!seen[pi]);
                seen[pi] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "t {t}: a pair is in no group");
    }
}
