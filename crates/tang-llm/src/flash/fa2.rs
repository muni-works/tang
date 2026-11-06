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
/// tiles, rows padded by 8 bf16. 101376 bytes at head dim 256. Under `TANG_CODER_KV=int8`
/// (`int8`) the int8 K/V tiles land in their own double-buffered staging first
/// (`4 * FA_BN * head_dim` bytes more: 134144 at head dim 256).
pub fn fa2_smem(head_dim: usize) -> usize {
    fa2_smem_as(kv8(), head_dim)
}

pub fn fa2_smem_as(int8: bool, head_dim: usize) -> usize {
    (FA2_BM + 4 * FA2_BN) * (head_dim + 8) * 2 + if int8 { 4 * FA2_BN * head_dim } else { 0 }
}

/// The attention K/V cache format (`TANG_CODER_KV=bf16|int8`, read once at first use; default
/// bf16). `int8` stores each position's K (and V) as one row of `n_kv` heads of `head_dim` int8
/// codes followed by the heads' f32 absmax scales ([`kv_row_bytes`]): about half the bytes of a
/// bf16 row, so twice the sequence slots at a long `--ctx`. `cx_attn_prep` quantizes as it
/// writes; the split-K and FA2 attention kernels dequantize as they load.
pub fn kv8() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("TANG_CODER_KV").as_deref() {
        Ok("int8") => true,
        Ok("bf16") | Err(_) => false,
        Ok(v) => panic!("TANG_CODER_KV={v}: bf16 or int8"),
    })
}

/// Bytes one position of a layer's K (or V) cache takes: `n_kv * head_dim` bf16, or under
/// [`kv8`] `n_kv * head_dim` int8 codes, `n_kv` f32 scales, padded to 16 bytes (so every row,
/// and every head's codes in it, stay aligned for `cp.async`).
pub fn kv_row_bytes(n_kv: usize, head_dim: usize) -> usize {
    kv_row_bytes_as(kv8(), n_kv, head_dim)
}

pub fn kv_row_bytes_as(int8: bool, n_kv: usize, head_dim: usize) -> usize {
    if int8 {
        (n_kv * head_dim + n_kv * 4).next_multiple_of(16)
    } else {
        n_kv * head_dim * 2
    }
}

/// The int8 rule of `cx_attn_prep` for one head's `head_dim` values: codes and the scale
/// (absmax / 127; a zero head gets scale 0 and zero codes). Dequantized value = code * scale.
pub fn kv8_quant(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let (s, inv) = if amax > 0.0 {
        (amax / 127.0, 127.0 / amax)
    } else {
        (0.0, 0.0)
    };
    // __float2int_rn: round half to even.
    let q = x
        .iter()
        .map(|v| (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, s)
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

    /// K/V through the `TANG_CODER_KV=int8` rule (per position and kv head: absmax / 127 scale,
    /// round-half-even codes), dequantized.
    fn kv8_round(x: &[f32], hd: usize) -> Vec<f32> {
        x.chunks(hd)
            .flat_map(|h| {
                let (q, s) = super::kv8_quant(h);
                q.into_iter().map(move |c| c as f32 * s).collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn kv8_quant_rule() {
        use super::kv8_quant;
        let (q, s) = kv8_quant(&[0.0; 8]);
        assert_eq!((q, s), (vec![0; 8], 0.0));
        let x = [2.0f32, -1.0, 0.01, -2.0, 0.0];
        let (q, s) = kv8_quant(&x);
        assert_eq!((q[0], q[3], q[4]), (127, -127, 0));
        assert_eq!(s, 2.0 / 127.0);
        for (&c, &v) in q.iter().zip(&x) {
            assert!((c as f32 * s - v).abs() <= s / 2.0 + 1e-7);
        }
    }

    #[test]
    fn kv8_row_and_slot_math() {
        use super::kv_row_bytes_as;
        // Qwen3-Coder-Next: 2 kv heads of 256; 12 attention layers + the MTP layer hold K/V.
        assert_eq!(kv_row_bytes_as(false, 2, 256), 1024);
        assert_eq!(kv_row_bytes_as(true, 2, 256), 528);
        assert_eq!(kv_row_bytes_as(true, 1, 16), 32);
        let ctx = 131072;
        // 36 GDN layers: f32 state (32 v heads of 128 x 128) and the conv history.
        let gdn = 36 * (32 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 32 * 128) * 4);
        let slot = |row: usize| 13 * ctx * row * 2 + gdn;
        let (bf, i8) = (slot(1024), slot(528));
        assert!(bf > 3_550_000_000 && bf < 3_600_000_000, "{bf}");
        assert!(i8 < 1_900_000_000, "{i8}");
        // 24 slots fit at bf16 on the production H200: the same free memory holds 46+ at int8.
        let free = 4_000_000_000 + 24 * bf + bf / 2;
        assert_eq!(
            crate::batch::slots_that_fit(free, 4_000_000_000, bf, 256),
            24
        );
        assert!(crate::batch::slots_that_fit(free, 4_000_000_000, i8, 256) >= 46);
        assert_eq!(super::fa2_smem_as(false, 256), 101376);
        assert_eq!(super::fa2_smem_as(true, 256), 134144);
    }

    /// int8 K/V through the FA2 model against exact f64 attention on the unrounded K/V, beside
    /// the bf16 cache's error on the same inputs: head dim 256 (Coder-Next's), keys with a few
    /// outlier channels (as rmsnormed K has), long enough to cross many tiles and splits.
    #[test]
    fn kv8_attention_error_stays_small() {
        let sh = || Shape {
            nh: 16,
            nkv: 2,
            hd: 256,
        };
        let (t, pos0, nsplit) = (24, 700, 3);
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let (hd, keys) = (256, pos0 + t);
        let q: Vec<f32> = (0..t * 16 * hd).map(|_| rnd() * 3.0).collect();
        let k: Vec<f32> = (0..keys * 2 * hd)
            .map(|i| rnd() * if i % hd < 4 { 12.0 } else { 3.0 })
            .collect();
        let v: Vec<f32> = (0..keys * 2 * hd).map(|_| rnd()).collect();
        let want = naive(&sh(), &q, &k, &v, t, pos0);
        let err = |kk: &[f32], vv: &[f32]| {
            let got = fa2_model(&sh(), &q, kk, vv, t, pos0, nsplit);
            let (mut worst, mut sum) = (0f64, 0f64);
            for (&a, &b) in got.iter().zip(&want) {
                let e = (a as f64 - b).abs();
                worst = worst.max(e);
                sum += e;
            }
            (worst, sum / want.len() as f64)
        };
        let kb: Vec<f32> = k.iter().map(|&x| bf16(x)).collect();
        let vb: Vec<f32> = v.iter().map(|&x| bf16(x)).collect();
        let (bw, bm) = err(&kb, &vb);
        let (iw, im) = err(&kv8_round(&k, hd), &kv8_round(&v, hd));
        eprintln!("bf16 K/V: max {bw:.2e} mean {bm:.2e}; int8 K/V: max {iw:.2e} mean {im:.2e}");
        // Measured: bf16 max 6.9e-3 mean 7.2e-4, int8 max 4.4e-2 mean 4.7e-3 (sharp synthetic
        // softmax, 4x outlier channels in K). The GPU gates (flash-batch-check, flash-wide-check,
        // the needle, greedy prefixes) judge the model; this pins the rule's order of magnitude.
        assert!(
            iw < 10.0 * bw && im < 10.0 * bm,
            "int8 K/V: max {iw:.3e} mean {im:.3e}"
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
