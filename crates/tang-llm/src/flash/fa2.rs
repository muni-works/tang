//! Launch planning for `cx_attn_fa2` (the FA2-style wide-prefill attention of the Coder engine,
//! `TANG_CODER_ATTN=fa2`; design in docs/bench/fa2-prefill.md), and in the tests a CPU model of
//! the kernel + `cx_attn_merge`: the same GQA vector packing, key splits, 32-key tiles, causal
//! masking, online softmax and fp16 rounding of Q and P, checked against plain f64 attention. It
//! pins the tiling / split / merge arithmetic; the kernel's fragment layouts are checked on the
//! GPU by `flash-wide-check`.

/// Query vectors ((row, head) pairs of a kv group) per FA2 attention block (`FA_BM`).
pub const FA2_BM: usize = 64;
/// Keys per FA2 K/V tile (`FA_BN`).
pub const FA2_BN: usize = 32;

/// Dynamic shared memory of `cx_attn_fa2` (`FA_SMEM`): the Q tile plus double-buffered K and V
/// tiles, rows padded by 8 bf16. 101376 bytes at head dim 256.
pub fn fa2_smem(head_dim: usize) -> usize {
    (FA2_BM + 4 * FA2_BN) * (head_dim + 8) * 2
}

/// Key splits for an FA2 window: enough blocks (`tiles` query tiles × kv groups, times the
/// splits) for about two waves on `sms` SMs, but no split under 256 keys and no more than `cap`
/// (what the partials buffer holds). 0 when even one split doesn't fit.
pub fn fa2_splits(tiles: usize, keys: usize, sms: usize, cap: usize) -> usize {
    let want = (2 * sms).div_ceil(tiles.max(1));
    want.min(keys.div_ceil(256)).max(1).min(cap)
}

#[cfg(test)]
mod tests {
    const BM: usize = super::FA2_BM;
    const BN: usize = super::FA2_BN;

    fn bf16(x: f32) -> f32 {
        // Round to nearest even, as f2bf.
        let b = x.to_bits();
        f32::from_bits(((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16)
    }

    fn f16(x: f32) -> f32 {
        // Round to nearest even at fp16's 10 mantissa bits, as cvt.rn.f16x2.f32 (normal range;
        // the test's values stay in it).
        let b = x.to_bits();
        f32::from_bits(((b + 0xfff + ((b >> 13) & 1)) >> 13) << 13)
    }

    struct Shape {
        nh: usize,
        nkv: usize,
        hd: usize,
    }

    /// Gate-free attention outputs [t][nh][hd] for a window of `t` rows at positions `pos0..`, keys
    /// `k`/`v` [pos][nkv][hd] (already bf16 values), queries `q` [t][nh][hd].
    fn fa2_model(
        sh: &Shape,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t: usize,
        pos0: usize,
        nsplit: usize,
    ) -> Vec<f32> {
        let (nh, nkv, hd) = (sh.nh, sh.nkv, sh.hd);
        let g = nh / nkv;
        let nv = t * g;
        let scale = 1.0 / (hd as f32).sqrt();
        let top = pos0 + t - 1;
        let span = (top + nsplit) / nsplit;
        let span = span.div_ceil(BN) * BN;
        // part[row][grp][split][head] = (m, l, acc)
        let mut part = vec![(f32::NEG_INFINITY, 0f32, vec![0f32; hd]); t * nkv * nsplit * g];
        for grp in 0..nkv {
            for tile in 0..nv.div_ceil(BM) {
                let v0 = tile * BM;
                let vlast = (v0 + BM).min(nv) - 1;
                let hi = pos0 + vlast / g;
                for j in 0..nsplit {
                    let k0 = j * span;
                    let k1 = (k0 + span).min(hi + 1);
                    for vv in v0..=vlast {
                        let (row, h) = (vv / g, vv % g);
                        let pos = pos0 + row;
                        let qv: Vec<f32> = (0..hd)
                            .map(|d| f16(q[(row * nh + grp * g + h) * hd + d] * scale))
                            .collect();
                        let (mut m, mut l, mut o) = (f32::NEG_INFINITY, 0f32, vec![0f32; hd]);
                        let mut kb = k0;
                        while kb < k1 {
                            let s: Vec<f32> = (kb..kb + BN)
                                .map(|key| {
                                    if key >= k1 || key > pos {
                                        return f32::NEG_INFINITY;
                                    }
                                    (0..hd).map(|d| qv[d] * k[(key * nkv + grp) * hd + d]).sum()
                                })
                                .collect();
                            let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let nm = m.max(mx);
                            let sm = if nm == f32::NEG_INFINITY { 0.0 } else { nm };
                            let c = (m - sm).exp();
                            m = nm;
                            let p: Vec<f32> = s.iter().map(|&x| (x - sm).exp()).collect();
                            l = l * c + p.iter().sum::<f32>();
                            for d in 0..hd {
                                o[d] *= c;
                                for (i, &pi) in p.iter().enumerate() {
                                    let key = kb + i;
                                    if key < k1 {
                                        o[d] += f16(pi) * v[(key * nkv + grp) * hd + d];
                                    }
                                }
                            }
                            kb += BN;
                        }
                        part[((row * nkv + grp) * nsplit + j) * g + h] = (m, l, o);
                    }
                }
            }
        }
        // cx_attn_merge without the gate.
        let mut out = vec![0f32; t * nh * hd];
        for row in 0..t {
            for hh in 0..nh {
                let (grp, h) = (hh / g, hh % g);
                let ps: Vec<_> = (0..nsplit)
                    .map(|j| &part[((row * nkv + grp) * nsplit + j) * g + h])
                    .filter(|p| p.1 > 0.0)
                    .collect();
                let mm = ps.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
                let ll: f32 = ps.iter().map(|p| p.1 * (p.0 - mm).exp()).sum();
                for d in 0..hd {
                    let od: f32 = ps.iter().map(|p| p.2[d] * (p.0 - mm).exp()).sum();
                    out[(row * nh + hh) * hd + d] = od / ll;
                }
            }
        }
        out
    }

    fn naive(sh: &Shape, q: &[f32], k: &[f32], v: &[f32], t: usize, pos0: usize) -> Vec<f64> {
        let (nh, nkv, hd) = (sh.nh, sh.nkv, sh.hd);
        let g = nh / nkv;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut out = vec![0f64; t * nh * hd];
        for row in 0..t {
            let pos = pos0 + row;
            for hh in 0..nh {
                let grp = hh / g;
                let s: Vec<f64> = (0..=pos)
                    .map(|key| {
                        (0..hd)
                            .map(|d| {
                                q[(row * nh + hh) * hd + d] as f64
                                    * k[(key * nkv + grp) * hd + d] as f64
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
                for d in 0..hd {
                    out[(row * nh + hh) * hd + d] = s
                        .iter()
                        .enumerate()
                        .map(|(key, x)| (x - m).exp() / z * v[(key * nkv + grp) * hd + d] as f64)
                        .sum();
                }
            }
        }
        out
    }

    fn check(sh: Shape, t: usize, pos0: usize, nsplit: usize) {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let keys = pos0 + t;
        let q: Vec<f32> = (0..t * sh.nh * sh.hd).map(|_| rnd() * 3.0).collect();
        let k: Vec<f32> = (0..keys * sh.nkv * sh.hd)
            .map(|_| bf16(rnd() * 3.0))
            .collect();
        let v: Vec<f32> = (0..keys * sh.nkv * sh.hd).map(|_| bf16(rnd())).collect();
        let got = fa2_model(&sh, &q, &k, &v, t, pos0, nsplit);
        let want = naive(&sh, &q, &k, &v, t, pos0);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (a as f64 - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 4e-3,
            "t {t} pos0 {pos0} splits {nsplit}: max |err| {worst:.3e}"
        );
    }

    #[test]
    fn fa2_model_matches_naive_attention() {
        // Qwen3-Next's grouping (G = 8) at a small head dim; windows that start mid-tile, cross key
        // tiles, leave partial query tiles, and splits that end up empty for early rows.
        let sh = || Shape {
            nh: 16,
            nkv: 2,
            hd: 32,
        };
        check(sh(), 9, 0, 1);
        check(sh(), 20, 5, 1);
        check(sh(), 20, 77, 3);
        check(sh(), 13, 300, 8);
        check(
            Shape {
                nh: 4,
                nkv: 1,
                hd: 16,
            },
            40,
            31,
            2,
        );
    }

    #[test]
    fn fa2_splits_fill_the_gpu() {
        use super::fa2_splits;
        // 256-row window, G = 8, 2 kv groups: 32 tiles × 2 = 64 blocks; 132 SMs want ~264.
        assert_eq!(fa2_splits(64, 30_000, 132, 64), 5);
        // Short context: no split under 256 keys.
        assert_eq!(fa2_splits(64, 300, 132, 64), 2);
        // Capped by the partials buffer.
        assert_eq!(fa2_splits(64, 30_000, 132, 3), 3);
        assert_eq!(fa2_splits(64, 30_000, 132, 0), 0);
    }
}
