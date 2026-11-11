//! The DSA/MLA shadow check (`TANG_GLM5_MLA_DEBUG=1`, or `TANG_KERNEL_VERIFY=mla`): after
//! each GPU DSA window ([`super::engine`]'s `dsa_gpu`), run the CPU reference
//! ([`super::mla`]) on the SAME inputs and the DEVICE's own cache rows, intermediate by
//! intermediate:
//!
//! | tap | GPU side | CPU side |
//! |---|---|---|
//! | `lat` | resident latent rows `[pos0, kv_len)` (bf16) | `gemv_kv_a(x)` rounded to bf16 |
//! | `lat.mirror` | the engine's host mirror, every row | the resident rows (exact) |
//! | `lat.cache` / `idx_k.cache` / `idx_g.cache` | every resident row `[0, kv_len)` | the shadow's OWN cache, appended from `x` window by window (a row the device lost or re-uploaded wrong after a realloc fails here) |
//! | `idx_k` / `idx_g` | resident indexer rows `[pos0, kv_len)` | [`super::mla::indexer_key`] |
//! | `idx_k.mirror` / `idx_g.mirror` / `valid.mirror` | host mirrors, every row | resident (exact) |
//! | `valid` | resident valid bytes | 1 |
//! | `pool` | `idx_pool`'s pooled keys | the pooling math over the resident rows |
//! | `out` | the GPU attention (+ host o_proj) | [`super::mla::mla_forward`] on the resident bf16 latents **with the GPU's selection** |
//!
//! The selection is compared on its own (it is a hard top-k, not a float tensor): the CPU
//! [`super::mla::indexer_select`] runs over the resident indexer rows, and every query whose
//! selected SET differs is classified by the f64 pool scores of the swapped pools: a
//! **near tie** when the scores differ by ≤ [`TIE_REL`] of the query's largest score
//! (f32 accumulation order decides those; both sides are right), else **real**. A real
//! mismatch, or any tap off its bar, fails the window. `out.cpusel` (info, not a tap):
//! the GPU output vs the CPU attention with the CPU's own selection, i.e. how much a
//! tie flip moves this layer.
//!
//! Poisoning (on by default): the resident rows this window writes and the attend/select
//! scratch are NaN-filled first ([`super::mla::cuda::MlaCuda::debug_poison`]).

use super::mla::{self, MlaWeights};
use tang_verify::{stat, Layout, Spec, Tap, Taps, Verifier};

pub const LEGACY: &str = "TANG_GLM5_MLA_DEBUG";

/// A selection mismatch is a near tie when the swapped pools' f64 scores differ by at
/// most this fraction of the query's largest |score|.
pub const TIE_REL: f64 = 1e-4;

/// What the GPU path produced for one window. Device rows are `[0, kv_len)` read off the
/// device; mirrors are the engine's host copies (same span).
pub struct GpuWindow<'a> {
    pub x: &'a [f32],
    pub q_resid: &'a [f32],
    pub t: usize,
    pub pos0: usize,
    pub lat_dev: Vec<u16>,
    pub idk_dev: Vec<f32>,
    pub idg_dev: Vec<f32>,
    pub val_dev: Vec<u8>,
    pub lat_mirror: &'a [u16],
    pub idk_mirror: &'a [f32],
    pub idg_mirror: &'a [f32],
    pub val_mirror: &'a [u8],
    pub pool: Option<Vec<f32>>,
    pub sel: Option<Vec<i32>>,
    pub out: &'a [f32],
    /// `q_resid` came off the device (the resident projections): check it against the
    /// CPU's `gemv_q_a(x)` (else it IS the host value).
    pub q_resid_dev: bool,
    /// the device's q (`q_b · q_resid`), when the projections are resident
    pub q_dev: Option<&'a [f32]>,
}

/// Per-window selection comparison.
#[derive(Clone, Debug, Default)]
pub struct SelCmp {
    pub queries: usize,
    pub differ: usize,
    pub ties: usize,
    pub real: usize,
    /// the widest swapped-pool score gap seen, relative to the query's largest |score|
    pub worst_gap: f64,
}

pub struct Shadow {
    v: Verifier,
    /// per layer: the shadow's own caches (latent bf16-rounded as f32, indexer k, gate)
    caches: Vec<[Vec<f32>; 3]>,
    sel: SelCmp,
    sel_windows_bad: usize,
}

pub fn bf2f(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round-to-nearest-even f32 → bf16, the `f2bf` of the CUDA source.
pub fn f2bf(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

/// Pooled indexer keys over `[n_pool]` complete pools from 0 (the `idx_pool_keys` math).
pub fn pool_keys(w: &MlaWeights, k: &[f32], g: &[f32], n_pool: usize) -> Vec<f32> {
    let (hd, kp) = (w.idx_dim, w.kpool);
    let mut out = vec![0f32; n_pool * hd];
    for p in 0..n_pool {
        for d in 0..hd {
            let lg: Vec<f32> = (0..kp)
                .map(|j| g[(p * kp + j) * hd + d] + w.i_ape[j * hd + d])
                .collect();
            let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = lg.iter().map(|l| (l - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            out[p * hd + d] = (0..kp).map(|j| e[j] / sum * k[(p * kp + j) * hd + d]).sum();
        }
    }
    out
}

/// f64 pool scores `[t][n_pool]` (`-inf` = not visible), the indexer's score in f64 so the
/// tie classification has an order-independent reference.
#[allow(clippy::too_many_arguments)]
pub fn pool_scores_f64(
    w: &MlaWeights,
    x: &[f32],
    q_resid: &[f32],
    pk: &[f32],
    n_pool: usize,
    t: usize,
    pos0: usize,
) -> Vec<f64> {
    let (hd, nh, kp) = (w.idx_dim, w.idx_heads, w.kpool);
    let dotd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>()
    };
    let ss = 1.0 / (hd as f64).sqrt();
    let hs = 1.0 / (nh as f64).sqrt();
    let mut out = vec![f64::NEG_INFINITY; t * n_pool];
    for qi in 0..t {
        let qr = &q_resid[qi * w.q_lora..(qi + 1) * w.q_lora];
        let iq: Vec<f64> = (0..nh * hd)
            .map(|r| dotd(&w.i_wq_b[r * w.q_lora..(r + 1) * w.q_lora], qr))
            .collect();
        let xi = &x[qi * w.hidden..(qi + 1) * w.hidden];
        let hw: Vec<f64> = (0..nh)
            .map(|h| dotd(&w.i_proj[h * w.hidden..(h + 1) * w.hidden], xi))
            .collect();
        for p in 0..n_pool {
            if p * kp + kp - 1 > pos0 + qi {
                continue;
            }
            let pkp = &pk[p * hd..(p + 1) * hd];
            let mut s = 0f64;
            for h in 0..nh {
                let d: f64 = (0..hd).map(|j| iq[h * hd + j] * pkp[j] as f64).sum::<f64>() * ss;
                s += hw[h] * hs * d.max(0.0);
            }
            out[qi * n_pool + p] = s;
        }
    }
    out
}

/// Compare two selections `[t][width]` as per-query sets; classify each differing query
/// by the f64 scores of the pools only one side picked.
pub fn compare_sel(
    w: &MlaWeights,
    a: &[i32],
    b: &[i32],
    scores: &[f64],
    n_pool: usize,
    t: usize,
    mut on_diff: impl FnMut(usize, &[usize], &[usize], f64),
) -> SelCmp {
    let width = w.select_width();
    let kp = w.kpool;
    let pools = |s: &[i32]| {
        let mut v: Vec<usize> = s[..(w.topk).min(s.len())]
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| i as usize / kp)
            .collect();
        v.sort();
        v.dedup();
        v
    };
    let set = |s: &[i32]| {
        let mut v: Vec<i32> = s.iter().copied().filter(|&i| i >= 0).collect();
        v.sort();
        v
    };
    let mut c = SelCmp {
        queries: t,
        ..Default::default()
    };
    for qi in 0..t {
        let (ra, rb) = (
            &a[qi * width..(qi + 1) * width],
            &b[qi * width..(qi + 1) * width],
        );
        if set(ra) == set(rb) {
            continue;
        }
        c.differ += 1;
        let (pa, pb) = (pools(ra), pools(rb));
        let only_a: Vec<usize> = pa.iter().copied().filter(|p| !pb.contains(p)).collect();
        let only_b: Vec<usize> = pb.iter().copied().filter(|p| !pa.contains(p)).collect();
        let row = &scores[qi * n_pool..(qi + 1) * n_pool];
        let scale = row
            .iter()
            .filter(|s| s.is_finite())
            .fold(0f64, |m, s| m.max(s.abs()))
            .max(1e-30);
        // the swap is justified iff every pool only one side took scores within the
        // bar of every pool only the other side took
        let mut gap = 0f64;
        for &i in &only_a {
            for &j in &only_b {
                gap = gap.max((row[i] - row[j]).abs());
            }
        }
        if only_a.is_empty() || only_b.is_empty() {
            // a different COUNT of pools, or the same pools with different tails, is
            // never a tie
            gap = f64::INFINITY;
        }
        let rel = gap / scale;
        c.worst_gap = c.worst_gap.max(rel);
        if rel <= TIE_REL {
            c.ties += 1;
        } else {
            c.real += 1;
        }
        on_diff(qi, &only_a, &only_b, rel);
    }
    c
}

impl Shadow {
    pub fn from_env() -> Option<Self> {
        let v = Verifier::from_env("mla", Some(LEGACY))?
            .with_tag("mladbg")
            .with_unit("l");
        let c = &v.cfg;
        v.log(format_args!(
            "ON: layers {:?} windows {:?} tol {:.1e} poison {} (selection near-tie bar {TIE_REL:.0e})",
            c.units, c.windows, c.tol, c.poison
        ));
        Some(Self {
            v,
            caches: Vec::new(),
            sel: SelCmp::default(),
            sel_windows_bad: 0,
        })
    }

    /// On regardless of the env (tests): default knobs, poison on, only failing windows'
    /// rows printed.
    pub fn always() -> Self {
        let mut v = Verifier::new("mla", tang_verify::Config::default())
            .with_tag("mladbg")
            .with_unit("l");
        v.verbose = false;
        Self {
            v,
            caches: Vec::new(),
            sel: SelCmp::default(),
            sel_windows_bad: 0,
        }
    }

    pub fn wants(&self, l: usize) -> bool {
        self.v.wants(l)
    }

    pub fn poison(&self) -> bool {
        self.v.cfg.poison
    }

    /// Totals so far: (windows checked, windows failed incl. selection, selection totals).
    pub fn totals(&self) -> (usize, usize, SelCmp) {
        (
            self.v.checked(),
            self.v.failed() + self.sel_windows_bad,
            self.sel.clone(),
        )
    }

    /// Check one window; returns whether it passed.
    pub fn check(&mut self, l: usize, w: &MlaWeights, g: &GpuWindow) -> bool {
        let (t, pos0) = (g.t, g.pos0);
        let kv = pos0 + t;
        let (dl, hd) = (w.kv_lora, w.idx_dim);
        let f = |v: &[u16]| v.iter().map(|&b| bf2f(b)).collect::<Vec<f32>>();
        let fb = |v: &[u8]| v.iter().map(|&b| b as f32).collect::<Vec<f32>>();
        let (mut gt, mut ct) = (Taps::new(), Taps::new());
        let lat_dev = f(&g.lat_dev);
        let tok = |d| Layout::new().dim("tok", 0).dim("d", d);
        let mut spec = Spec::new("mla");

        // projections (resident path only): q_resid = norm(q_a x), q = q_b q_resid
        if g.q_resid_dev {
            gt.put("q_resid", g.q_resid.to_vec());
            ct.put("q_resid", w.gemv_q_a(g.x, t));
            spec.push(Tap::new("q_resid", tok(w.q_lora)).what("g5f_gemv q_a + mla_rms_norm"));
        }
        if let Some(q) = g.q_dev {
            gt.put("q", q.to_vec());
            ct.put("q", w.gemv_q_b(g.q_resid, t));
            spec.push(
                Tap::new("q", tok(w.n_head * w.qk_nope))
                    .after(&["q_resid"])
                    .what("g5f_gemv q_b"),
            );
        }

        // latents: kv_a + RMSNorm, bf16
        let lat_cpu: Vec<f32> = w.gemv_kv_a(g.x, t).iter().map(|&v| bf2f(f2bf(v))).collect();
        gt.put("lat", lat_dev[pos0 * dl..].to_vec());
        ct.put("lat", lat_cpu.clone());
        spec.push(Tap::new("lat", tok(dl)).what("mla_kv_append"));
        gt.put("lat.mirror", f(g.lat_mirror));
        ct.put("lat.mirror", lat_dev.clone());
        spec.push(
            Tap::new("lat.mirror", tok(dl))
                .after(&["lat"])
                .what("kv_append read-back"),
        );

        // indexer rows
        let (mut ck, mut cg) = (vec![0f32; t * hd], vec![0f32; t * hd]);
        for i in 0..t {
            mla::indexer_key(
                w,
                &g.x[i * w.hidden..(i + 1) * w.hidden],
                &mut ck[i * hd..(i + 1) * hd],
                &mut cg[i * hd..(i + 1) * hd],
            );
        }
        // the shadow's own caches: appended window by window; a skipped window (WINDOWS
        // filter) or a new sequence re-seeds from the device rows
        if self.caches.len() <= l {
            self.caches.resize_with(l + 1, Default::default);
        }
        let sc = &mut self.caches[l];
        if sc[0].len() != pos0 * dl || sc[1].len() != pos0 * hd {
            sc[0] = lat_dev[..pos0 * dl].to_vec();
            sc[1] = g.idk_dev[..pos0 * hd].to_vec();
            sc[2] = g.idg_dev[..pos0 * hd].to_vec();
        }
        sc[0].extend_from_slice(&lat_cpu);
        sc[1].extend_from_slice(&ck);
        sc[2].extend_from_slice(&cg);
        for (n, dev, own) in [
            ("lat.cache", &lat_dev, &sc[0]),
            ("idx_k.cache", &g.idk_dev, &sc[1]),
            ("idx_g.cache", &g.idg_dev, &sc[2]),
        ] {
            gt.put(n, dev.clone());
            ct.put(n, own.clone());
        }
        gt.put("idx_k", g.idk_dev[pos0 * hd..].to_vec());
        ct.put("idx_k", ck);
        gt.put("idx_g", g.idg_dev[pos0 * hd..].to_vec());
        ct.put("idx_g", cg);
        spec.push(Tap::new("idx_k", tok(hd)).what("idx_key_append"));
        spec.push(Tap::new("idx_g", tok(hd)).what("idx_key_append"));
        for (n, m, d) in [
            ("idx_k.mirror", g.idk_mirror, &g.idk_dev),
            ("idx_g.mirror", g.idg_mirror, &g.idg_dev),
        ] {
            gt.put(n, m.to_vec());
            ct.put(n, d.clone());
        }
        spec.push(
            Tap::new("idx_k.mirror", tok(hd))
                .after(&["idx_k"])
                .what("idx_append read-back"),
        );
        spec.push(
            Tap::new("idx_g.mirror", tok(hd))
                .after(&["idx_g"])
                .what("idx_append read-back"),
        );
        spec.push(
            Tap::new("lat.cache", tok(dl))
                .after(&["lat"])
                .what("resident latent cache (realloc / re-upload)"),
        );
        spec.push(
            Tap::new("idx_k.cache", tok(hd))
                .after(&["idx_k"])
                .what("resident indexer cache (realloc / re-upload)"),
        );
        spec.push(
            Tap::new("idx_g.cache", tok(hd))
                .after(&["idx_g"])
                .what("resident indexer cache (realloc / re-upload)"),
        );
        gt.put("valid", fb(&g.val_dev));
        ct.put("valid", vec![1.0; kv]);
        spec.push(Tap::new("valid", Layout::flat()).what("idx_key_append"));
        gt.put("valid.mirror", fb(g.val_mirror));
        ct.put("valid.mirror", fb(&g.val_dev));
        spec.push(
            Tap::new("valid.mirror", Layout::flat())
                .after(&["valid"])
                .what("idx_append read-back"),
        );

        // pool + selection (sparse windows only)
        let n_pool = kv / w.kpool;
        let mut sel_ok = true;
        let mut cpu_sel = None;
        if let (Some(pool), Some(sel)) = (&g.pool, &g.sel) {
            let pk = pool_keys(w, &g.idk_dev, &g.idg_dev, n_pool);
            gt.put("pool", pool.clone());
            ct.put("pool", pk.clone());
            spec.push(
                Tap::new("pool", tok(hd))
                    .after(&["idx_k", "idx_g"])
                    .what("idx_pool_keys"),
            );
            let cs = mla::indexer_select(
                w,
                g.x,
                g.q_resid,
                &g.idk_dev,
                &g.idg_dev,
                &g.val_dev,
                &vec![1u8; t],
                t,
                kv,
                pos0,
            );
            // NaN-poisoned slots read back as 2143289344: an unwritten sel slot is a hard fail
            let unwritten = sel.iter().filter(|&&i| i >= kv as i32).count();
            let sc = pool_scores_f64(w, g.x, g.q_resid, &pk, n_pool, t, pos0);
            let mut shown = 0;
            let v = &self.v;
            let c = compare_sel(w, sel, &cs, &sc, n_pool, t, |qi, a, b, rel| {
                if shown < 4 {
                    shown += 1;
                    v.log(format_args!(
                        "l{l} t{t} pos0 {pos0} sel q{qi} (pos {}): gpu-only pools {a:?} cpu-only {b:?}, f64 score gap {rel:.2e} of max → {}",
                        pos0 + qi,
                        if rel <= TIE_REL { "near tie" } else { "REAL" }
                    ));
                }
            });
            v.log(format_args!(
                "l{l} t{t} pos0 {pos0} sel: {}/{} queries differ ({} near ties, {} real), worst gap {:.2e}, unwritten slots {unwritten}",
                c.differ, c.queries, c.ties, c.real, c.worst_gap
            ));
            sel_ok = c.real == 0 && unwritten == 0;
            self.sel.queries += c.queries;
            self.sel.differ += c.differ;
            self.sel.ties += c.ties;
            self.sel.real += c.real;
            self.sel.worst_gap = self.sel.worst_gap.max(c.worst_gap);
            cpu_sel = Some(cs);
        }

        // attention with the GPU's selection over the resident latents: the kernel alone
        let mut out = vec![0f32; t * w.hidden];
        mla::mla_forward(
            w,
            g.x,
            g.q_resid,
            &lat_dev,
            kv,
            t,
            pos0,
            None,
            g.sel.as_deref(),
            &mut out,
        );
        gt.put("out", g.out.to_vec());
        ct.put("out", out);
        spec.push(
            Tap::new("out", tok(w.hidden))
                .after(&["lat", "pool"])
                .what("mla_attn_naive"),
        );

        let r = self
            .v
            .check(l, &format!("t{t} pos0 {pos0}"), &spec, &gt, &ct);
        if let Some(cs) = cpu_sel {
            let mut o2 = vec![0f32; t * w.hidden];
            mla::mla_forward(
                w,
                g.x,
                g.q_resid,
                &lat_dev,
                kv,
                t,
                pos0,
                None,
                Some(&cs),
                &mut o2,
            );
            let s = stat(g.out, &o2, self.v.cfg.tol);
            self.v.log(format_args!(
                "l{l} t{t} pos0 {pos0} out.cpusel (info: GPU out vs CPU attention with the CPU's own selection) rel {:.3e} max {:.3e}",
                s.rel_l2, s.max_abs
            ));
        }
        if !sel_ok {
            self.sel_windows_bad += 1;
        }
        r.ok() && sel_ok
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        let s = &self.sel;
        self.v.log(format_args!(
            "SUMMARY selection: {} queries, {} differ ({} near ties ≤ {TIE_REL:.0e}, {} real), worst gap {:.2e}; {} windows failed on selection",
            s.queries, s.differ, s.ties, s.real, s.worst_gap, self.sel_windows_bad
        ));
    }
}
