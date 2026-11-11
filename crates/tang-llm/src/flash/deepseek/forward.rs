//! The CPU f32 reference forward of DeepSeek-V4.1-Flash: the parity truth for the GPU engine.
//!
//! A transcription of `scripts/deepseek_ref.py`, step for step, including where it rounds to
//! bf16: every GEMM output, norm and RoPE output, the attention output and its probabilities,
//! the indexer's score products, the MoE combine and the mHC collapse/expand. Batch 1, no
//! parallelism beyond rayon over GEMM rows; slow and plain on purpose.
//!
//! State is the reference's: a ring of `window_size` raw KV rows per layer, the compressed KV
//! and index keys owned by each `kv_source_layers` layer, the ratio-2 compressors' partial
//! group, and what attention layers hand down the stack (the last source's caches, top-k and
//! candidate blocks).

use super::quant::{bf, bf_all, fp4_qdq, fp8_qdq};
use super::st::{Dtype, St};
use super::{Config, EngramMeta};
use crate::engine::{Finish, Usage};
use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// A dequantized weight, [rows, cols] row-major. `quant` weights (fp8 / fp4 in the checkpoint)
/// make the GEMM quantize its activation first.
pub struct W {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub quant: bool,
}

/// `y[t][n] = Σ_k x[t][k] w[n][k]` in f32.
pub(crate) fn matmul(x: &[f32], t: usize, w: &W) -> Vec<f32> {
    let (n, k) = (w.rows, w.cols);
    assert_eq!(x.len(), t * k, "matmul: x {} != {t} x {k}", x.len());
    let mut y = vec![0f32; t * n];
    y.par_chunks_mut(n).enumerate().for_each(|(ti, row)| {
        let xr = &x[ti * k..(ti + 1) * k];
        for (ni, out) in row.iter_mut().enumerate() {
            let wr = &w.data[ni * k..(ni + 1) * k];
            *out = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    });
    y
}

/// Indices of the `k` largest, ties to the lowest index, in score order (`topk_stable`).
pub(crate) fn topk_stable(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch's softplus (beta 1, threshold 20).
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// RoPE tables `[seqlen][dim / 2]` of (cos, sin), YaRN-corrected when `orig > 0`.
pub(crate) fn yarn_freqs(
    c: &Config,
    dim: usize,
    seqlen: usize,
    orig: usize,
    base: f32,
) -> Vec<(f32, f32)> {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|j| 1.0 / base.powf((2 * j) as f32 / dim as f32))
        .collect();
    if orig > 0 {
        let corr = |rot: f64| {
            dim as f64 * (orig as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = corr(c.beta_fast as f64).floor().max(0.0);
        let high = corr(c.beta_slow as f64).ceil().min(dim as f64 - 1.0);
        let span = (high - low).max(1e-3) as f32;
        for (j, f) in freqs.iter_mut().enumerate() {
            let ramp = ((j as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / c.rope_factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut out = Vec::with_capacity(seqlen * half);
    for p in 0..seqlen {
        for &f in &freqs {
            let a = p as f32 * f;
            out.push((a.cos(), a.sin()));
        }
    }
    out
}

/// Rotate the last `rd` of every `d`-wide vector in `v` (rows of `per_row` vectors), row `r`
/// at table position `pos(r)`; adjacent pairs are (re, im). Rounds to bf16.
pub(crate) fn rope(
    v: &mut [f32],
    d: usize,
    rd: usize,
    per_row: usize,
    tab: &[(f32, f32)],
    pos: impl Fn(usize) -> usize,
    inverse: bool,
) {
    let half = rd / 2;
    for (r, row) in v.chunks_mut(d * per_row).enumerate() {
        let f = &tab[pos(r) * half..(pos(r) + 1) * half];
        for vec in row.chunks_mut(d) {
            let tail = &mut vec[d - rd..];
            for (j, &(c, s)) in f.iter().enumerate() {
                let s = if inverse { -s } else { s };
                let (a, b) = (tail[2 * j], tail[2 * j + 1]);
                tail[2 * j] = bf(a * c - b * s);
                tail[2 * j + 1] = bf(a * s + b * c);
            }
        }
    }
}

/// Gathered attention with a per-head sink in the denominator (`sparse_attn`). `q` is [T][H][d],
/// `kv` [n][d] shared by every head, `idxs[t]` the rows query t sees (-1 = none).
pub(crate) fn sparse_attn(
    q: &[f32],
    h: usize,
    d: usize,
    kv: &[f32],
    sink: &[f32],
    idxs: &[Vec<i32>],
    scale: f32,
) -> Vec<f32> {
    let t = idxs.len();
    let mut o = vec![0f32; t * h * d];
    o.par_chunks_mut(h * d).enumerate().for_each(|(ti, ot)| {
        let rows: Vec<usize> = idxs[ti]
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect();
        for hi in 0..h {
            let qv = &q[(ti * h + hi) * d..(ti * h + hi + 1) * d];
            let s: Vec<f32> = rows
                .iter()
                .map(|&r| {
                    qv.iter()
                        .zip(&kv[r * d..(r + 1) * d])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let mx = s
                .iter()
                .fold(f32::NEG_INFINITY, |m, &x| m.max(x))
                .max(-1e30);
            let p: Vec<f32> = s.iter().map(|&x| (x - mx).exp()).collect();
            let denom = p.iter().sum::<f32>() + (sink[hi] - mx).exp();
            let out = &mut ot[hi * d..(hi + 1) * d];
            for (&r, &pi) in rows.iter().zip(&p) {
                let pb = bf(pi);
                for (o, &k) in out.iter_mut().zip(&kv[r * d..(r + 1) * d]) {
                    *o += pb * k;
                }
            }
            for o in out.iter_mut() {
                *o = bf(*o / denom);
            }
        }
    });
    o
}

/// Sinkhorn-normalized mHC coefficients from one token's mixes: (pre, post, comb[j][k]).
pub(crate) fn sinkhorn(
    m: &[f32],
    scale: &[f32],
    base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pre: Vec<f32> = (0..hc)
        .map(|j| sigmoid(m[j] * scale[0] + base[j]) + eps)
        .collect();
    let post: Vec<f32> = (0..hc)
        .map(|j| 2.0 * sigmoid(m[hc + j] * scale[1] + base[hc + j]))
        .collect();
    let mut comb: Vec<f32> = (0..hc * hc)
        .map(|i| m[2 * hc + i] * scale[2] + base[2 * hc + i])
        .collect();
    for row in comb.chunks_mut(hc) {
        let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut s = 0.0;
        for x in row.iter_mut() {
            *x = (*x - mx).exp();
            s += *x;
        }
        for x in row.iter_mut() {
            *x = *x / s + eps;
        }
    }
    let colnorm = |comb: &mut [f32]| {
        for k in 0..hc {
            let s: f32 = (0..hc).map(|j| comb[j * hc + k]).sum();
            for j in 0..hc {
                comb[j * hc + k] /= s + eps;
            }
        }
    };
    colnorm(&mut comb);
    for _ in 0..iters.saturating_sub(1) {
        for row in comb.chunks_mut(hc) {
            let s: f32 = row.iter().sum();
            for x in row.iter_mut() {
                *x /= s + eps;
            }
        }
        colnorm(&mut comb);
    }
    (pre, post, comb)
}

/// What the layers hand down the stack: the most recent sources' caches, top-k and candidates.
#[derive(Default)]
struct Shared {
    ckv: usize,
    idxk: usize,
    topk: Vec<Vec<i32>>,
    cand: Vec<Vec<bool>>,
}

pub struct Model {
    pub cfg: Config,
    pub(crate) st: St,
    wcache: RefCell<HashMap<String, Rc<W>>>,
    pub(crate) freqs: Vec<Vec<(f32, f32)>>,
    win: Vec<Vec<f32>>,
    ckv: HashMap<usize, Vec<f32>>,
    idxk: HashMap<usize, Vec<f32>>,
    cstate: HashMap<usize, (Vec<f32>, Vec<f32>)>,
    shared: Shared,
    /// Intermediates by name, for debugging parity (`tap`).
    pub taps: RefCell<Vec<(String, Vec<f32>)>>,
    /// Debugging: replace each backbone layer's input stream and pre-mix (flat [T][hc]) with
    /// these, so a layer's error is its own (teacher forcing against the reference).
    pub force: Option<Vec<(Vec<f32>, Vec<f32>)>>,
    engram: Option<EngramMeta>,
    hash_cache: Vec<i64>,
    /// While a [`Snapshot`] is open: every compressor partial-group write (lid, pos, kv, score),
    /// so a rewind can replay the kept positions' writes over the snapshot.
    journal: Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>>,
}

/// The DSpark draft of one decode step.
pub struct Draft {
    /// The sampled token, then `block` drafts (greedy, or drawn from `logits` by
    /// [`Model::forward_spec_with`]).
    pub ids: Vec<usize>,
    /// [block][vocab] draft logits, Markov bias included: draft `i` was drawn from row `i`
    /// (the q of speculative sampling, before temperature / top-p).
    pub logits: Vec<f32>,
    /// [block]
    pub conf: Vec<f32>,
}

impl Model {
    /// `dir` holds the safetensors shards, `config.json` and (with Engram) `engram.json`.
    pub fn open(dir: &Path) -> Result<Self> {
        // DeepSeek's repo ships its runtime config as inference/config.json (the top-level one
        // is in Hugging Face's format); the tiny fixture writes it at the top
        let inf = dir.join("inference/config.json");
        let cfg = Config::load(&if inf.exists() {
            inf
        } else {
            dir.join("config.json")
        })?;
        let mut st = St::open(dir)?;
        let engram = if cfg.engram_layer_ids.is_empty() {
            None
        } else {
            // the tables stay on the host and are read by row per step: copy them into memory
            // now (`DS_ENGRAM_PIN=0` leaves them memory-mapped), or every cold row is a
            // disk/network read
            if std::env::var("DS_ENGRAM_PIN").map_or(true, |v| v != "0") {
                let t0 = std::time::Instant::now();
                for &l in &cfg.engram_layer_ids {
                    for w in ["weight", "scale"] {
                        st.pin(&format!("layers.{l}.engram.embed.{w}"))?;
                    }
                }
                eprintln!(
                    "deepseek: engram tables pinned in {:.1} s",
                    t0.elapsed().as_secs_f64()
                );
            }
            Some(EngramMeta::load(&dir.join("engram.json"))?)
        };
        let s = cfg.max_seq_len;
        let (hd, rd) = (cfg.head_dim, cfg.rope_head_dim);
        let n_all = cfg.n_layers + cfg.n_mtp_layers;
        ensure!(
            cfg.compress_ratios.len() == n_all,
            "compress_ratios has {} entries, want {n_all}",
            cfg.compress_ratios.len()
        );
        let freqs = (0..n_all)
            .map(|l| {
                if cfg.compress_ratios[l] > 0 {
                    yarn_freqs(&cfg, rd, s, cfg.original_seq_len, cfg.compress_rope_theta)
                } else {
                    yarn_freqs(&cfg, rd, s, 0, cfg.rope_theta)
                }
            })
            .collect();
        let mut ckv = HashMap::new();
        let mut idxk = HashMap::new();
        let mut cstate = HashMap::new();
        for &l in &cfg.kv_source_layers {
            let r = cfg.compress_ratios[l];
            ckv.insert(l, vec![0f32; s / r * hd]);
            idxk.insert(l, vec![0f32; s / r * cfg.index_head_dim]);
            if r > 1 {
                cstate.insert(l, (vec![0f32; r * hd], vec![f32::NEG_INFINITY; r * hd]));
            }
        }
        Ok(Self {
            win: vec![vec![0f32; cfg.window_size * hd]; n_all],
            hash_cache: vec![0; s],
            cfg,
            st,
            wcache: RefCell::new(HashMap::new()),
            freqs,
            ckv,
            idxk,
            cstate,
            shared: Shared::default(),
            taps: RefCell::new(Vec::new()),
            force: None,
            engram,
            journal: None,
        })
    }

    pub fn w(&self, name: &str) -> Result<Rc<W>> {
        if let Some(w) = self.wcache.borrow().get(name) {
            return Ok(w.clone());
        }
        let quant = matches!(self.st.info(name)?.dtype, Dtype::F8E4M3 | Dtype::I8);
        let (data, shape) = self.st.f32(name)?;
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            _ => (shape[0], shape[1..].iter().product()),
        };
        let w = Rc::new(W {
            data,
            rows,
            cols,
            quant,
        });
        self.wcache.borrow_mut().insert(name.to_string(), w.clone());
        Ok(w)
    }

    fn tap(&self, name: &str, v: &[f32]) {
        if std::env::var_os("DS_TAP").is_some() {
            self.taps.borrow_mut().push((name.to_string(), v.to_vec()));
        }
    }

    fn v(&self, name: &str) -> Result<Rc<W>> {
        self.w(name)
    }

    /// `linear`: fp8 / fp4 weights quantize the activation per 32 first; the output is bf16.
    pub(crate) fn linear(&self, x: &[f32], t: usize, name: &str) -> Result<Vec<f32>> {
        let w = self.w(name)?;
        let mut y = if w.quant {
            let mut xq = x.to_vec();
            fp8_qdq(&mut xq, 32);
            matmul(&xq, t, &w)
        } else {
            matmul(x, t, &w)
        };
        bf_all(&mut y);
        Ok(y)
    }

    /// RMSNorm over rows of the weight's width, rounded to bf16.
    pub(crate) fn rms(&self, x: &mut [f32], name: &str) -> Result<()> {
        let w = self.v(name)?;
        let n = w.cols;
        let eps = self.cfg.norm_eps;
        for row in x.chunks_mut(n) {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let r = 1.0 / (ms + eps).sqrt();
            for (v, &g) in row.iter_mut().zip(&w.data) {
                *v = bf(g * (*v * r));
            }
        }
        Ok(())
    }

    // attention

    fn attention(&mut self, p: &str, lid: usize, x: &[f32], sp: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (h, hd, rd) = (c.n_heads, c.head_dim, c.rope_head_dim);
        let r = c.compress_ratios[lid];
        let t = x.len() / c.dim;
        let mut qr = self.linear(x, t, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(&mut qr, &format!("{p}.attn.q_norm.weight"))?;
        self.tap(&format!("{p}.qr"), &qr);
        let mut q = self.linear(&qr, t, &format!("{p}.attn.wq_b.weight"))?;
        self.tap(&format!("{p}.q0"), &q);
        let tab = self.freqs[lid].clone();
        rope(&mut q, hd, rd, h, &tab, |i| sp + i, false);
        self.tap(&format!("{p}.q"), &q);
        let (mut kv, mut idxs) = self.window_kv(p, lid, x, sp)?;
        self.tap(&format!("{p}.wkv"), &kv);
        if r > 0 {
            let offset = kv.len() / hd;
            let (ckv, cidx) = self.compress_kv(p, lid, x, &qr, sp, offset)?;
            self.tap(
                &format!("{p}.cidx"),
                &cidx.iter().flatten().map(|&i| i as f32).collect::<Vec<_>>(),
            );
            self.tap(&format!("{p}.ckv"), &ckv);
            kv.extend_from_slice(&ckv);
            for (a, b) in idxs.iter_mut().zip(cidx) {
                a.extend(b);
            }
        }
        self.attn_out(p, &q, &kv, &idxs, &tab, sp)
    }

    fn attn_out(
        &self,
        p: &str,
        q: &[f32],
        kv: &[f32],
        idxs: &[Vec<i32>],
        tab: &[(f32, f32)],
        pos0: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (h, hd, rd, g, ol) = (
            c.n_heads,
            c.head_dim,
            c.rope_head_dim,
            c.o_groups,
            c.o_lora_rank,
        );
        let t = idxs.len();
        let sink = self.v(&format!("{p}.attn.attn_sink"))?;
        let mut o = sparse_attn(q, h, hd, kv, &sink.data, idxs, (hd as f32).powf(-0.5));
        self.tap(&format!("{p}.sa"), &o);
        rope(&mut o, hd, rd, h, tab, |i| pos0 + i, true);
        fp8_qdq(&mut o, 32);
        // wo_a is block-diagonal over groups; the checkpoint's fp8 is dequantized to bf16
        let wo_a = self.w(&format!("{p}.attn.wo_a.weight"))?;
        let gd = h * hd / g;
        let mut mid = vec![0f32; t * g * ol];
        for ti in 0..t {
            for gi in 0..g {
                let ov = &o[(ti * g + gi) * gd..(ti * g + gi + 1) * gd];
                for ri in 0..ol {
                    let wr = &wo_a.data[(gi * ol + ri) * gd..(gi * ol + ri + 1) * gd];
                    mid[(ti * g + gi) * ol + ri] =
                        bf(ov.iter().zip(wr).map(|(a, b)| a * bf(*b)).sum());
                }
            }
        }
        self.linear(&mid, t, &format!("{p}.attn.wo_b.weight"))
    }

    /// This layer's window KV (the chunk itself in prefill, the ring in decode) and the rows
    /// each query sees.
    fn window_kv(
        &mut self,
        p: &str,
        lid: usize,
        x: &[f32],
        sp: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        let c = &self.cfg;
        let (hd, rd, win) = (c.head_dim, c.rope_head_dim, c.window_size);
        let t = x.len() / c.dim;
        let mut kv = self.linear(x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(&mut kv, &format!("{p}.attn.kv_norm.weight"))?;
        rope(&mut kv, hd, rd, 1, &self.freqs[lid], |i| sp + i, false);
        fp8_qdq(&mut kv, 32);
        let ring = &mut self.win[lid];
        if sp == 0 {
            if t <= win {
                ring[..t * hd].copy_from_slice(&kv);
            } else {
                let cut = t % win;
                let tail = &kv[(t - win) * hd..];
                ring[cut * hd..].copy_from_slice(&tail[..(win - cut) * hd]);
                ring[..cut * hd].copy_from_slice(&tail[(win - cut) * hd..]);
            }
            let n = t.min(win);
            let idxs = (0..t)
                .map(|e| {
                    let lo = (e as i64 - win as i64 + 1).max(0);
                    (0..n as i64)
                        .map(|j| {
                            if lo + j > e as i64 {
                                -1
                            } else {
                                (lo + j) as i32
                            }
                        })
                        .collect()
                })
                .collect();
            return Ok((kv, idxs));
        }
        if t > 1 {
            // a chunk at sp > 0 (a verify window): query sp+e sees the positions of its own
            // window in time order, the older ones from the ring as it was before the chunk,
            // exactly the rows (and order) T single decode steps would see
            let mut all = ring.clone();
            all.extend_from_slice(&kv);
            let idxs = (0..t)
                .map(|e| {
                    let q = sp + e;
                    ((q + 1).saturating_sub(win)..=q)
                        .map(|j| if j < sp { j % win } else { win + j - sp } as i32)
                        .collect()
                })
                .collect();
            for e in 0..t {
                let s = (sp + e) % win;
                ring[s * hd..(s + 1) * hd].copy_from_slice(&kv[e * hd..(e + 1) * hd]);
            }
            return Ok((all, idxs));
        }
        ring[(sp % win) * hd..(sp % win + 1) * hd].copy_from_slice(&kv);
        let oldest = sp % win + 1;
        let idxs = (oldest..win)
            .chain(0..oldest)
            .map(|i| if i > sp { -1 } else { i as i32 })
            .collect();
        Ok((ring.clone(), vec![idxs]))
    }

    /// The source's compressed latent for the groups this chunk completes, pre-RoPE.
    fn compressor(
        &mut self,
        p: &str,
        lid: usize,
        x: &[f32],
        sp: usize,
    ) -> Result<Option<Vec<f32>>> {
        let c = &self.cfg;
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let t = x.len() / c.dim;
        let norm = format!("{p}.attn.compressor.norm.weight");
        if r == 1 {
            let mut kv = self.linear(x, t, &format!("{p}.attn.compressor.wkv.weight"))?;
            self.rms(&mut kv, &norm)?;
            return Ok(Some(kv));
        }
        // ratio > 1 pools in fp32
        let (wkv, wgate) = (
            self.w(&format!("{p}.attn.compressor.wkv.weight"))?,
            self.w(&format!("{p}.attn.compressor.wgate.weight"))?,
        );
        let (kv, sc) = (matmul(x, t, &wkv), matmul(x, t, &wgate));
        let pool = |kv: &[f32], sc: &[f32], n: usize| -> Vec<f32> {
            // softmax over the n tokens of a group, per channel
            (0..hd)
                .map(|ch| {
                    let mx = (0..n)
                        .map(|j| sc[j * hd + ch])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = (0..n).map(|j| (sc[j * hd + ch] - mx).exp()).collect();
                    let s: f32 = e.iter().sum();
                    (0..n).map(|j| kv[j * hd + ch] * (e[j] / s)).sum()
                })
                .collect()
        };
        let (kst, sst) = self.cstate.get_mut(&lid).unwrap();
        let mut out;
        if sp == 0 {
            if t < r {
                kst[..t * hd].copy_from_slice(&kv);
                sst[..t * hd].copy_from_slice(&sc);
                return Ok(None);
            }
            let rem = t % r;
            let cut = t - rem;
            if rem > 0 {
                kst[..rem * hd].copy_from_slice(&kv[cut * hd..]);
                sst[..rem * hd].copy_from_slice(&sc[cut * hd..]);
            }
            out = Vec::with_capacity(cut / r * hd);
            for gi in 0..cut / r {
                out.extend(pool(
                    &kv[gi * r * hd..(gi + 1) * r * hd],
                    &sc[gi * r * hd..(gi + 1) * r * hd],
                    r,
                ));
            }
        } else {
            let slot = sp % r;
            kst[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
            sst[slot * hd..(slot + 1) * hd].copy_from_slice(&sc);
            if let Some(j) = &mut self.journal {
                j.push((lid, sp, kv.clone(), sc.clone()));
            }
            if !(sp + 1).is_multiple_of(r) {
                return Ok(None);
            }
            out = pool(kst, sst, r);
        }
        bf_all(&mut out);
        self.rms(&mut out, &norm)?;
        Ok(Some(out))
    }

    fn compress_kv(
        &mut self,
        p: &str,
        lid: usize,
        x: &[f32],
        qr: &[f32],
        sp: usize,
        offset: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        let (r, rd, hd) = (
            self.cfg.compress_ratios[lid],
            self.cfg.rope_head_dim,
            self.cfg.head_dim,
        );
        let t = x.len() / self.cfg.dim;
        if sp > 0 && t > 1 {
            return self.compress_kv_steps(p, lid, x, qr, sp, offset);
        }
        let clen = (sp + t) / r;
        let mut latent = None;
        if self.cfg.is_kv_source(lid) {
            latent = self.compressor(p, lid, x, sp)?;
            self.shared.ckv = lid;
        }
        let idxs = if self.cfg.is_index_source(lid) {
            let idxs = if clen == 0 {
                vec![Vec::new(); t]
            } else {
                self.indexer(p, lid, x, qr, latent.as_deref(), sp, offset)?
            };
            self.shared.topk = idxs.clone();
            idxs
        } else {
            self.shared.topk.clone()
        };
        if let Some(mut lat) = latent {
            let n = lat.len() / hd;
            let pos = |j: usize| if sp == 0 { j * r } else { sp + 1 - r };
            rope(&mut lat, hd, rd, 1, &self.freqs[lid], pos, false);
            fp4_qdq(&mut lat, 16, true);
            let dst = &mut self.ckv.get_mut(&lid).unwrap()[(sp / r) * hd..(sp / r + n) * hd];
            dst.copy_from_slice(&lat);
        }
        Ok((self.ckv[&self.shared.ckv][..clen * hd].to_vec(), idxs))
    }

    /// A chunk at sp > 0 through the compressor and indexer one position at a time: group
    /// boundaries inside the chunk complete where a decode step would complete them, and each
    /// query's top-k sees only the groups complete at its own position.
    fn compress_kv_steps(
        &mut self,
        p: &str,
        lid: usize,
        x: &[f32],
        qr: &[f32],
        sp: usize,
        offset: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        let d = self.cfg.dim;
        let t = x.len() / d;
        let ql = qr.len() / t;
        let topk_in = std::mem::take(&mut self.shared.topk);
        let cand_in = std::mem::take(&mut self.shared.cand);
        let (mut idxs, mut topk, mut cand, mut ckv) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for e in 0..t {
            self.shared.topk = topk_in.get(e).cloned().into_iter().collect();
            self.shared.cand = cand_in.get(e).cloned().into_iter().collect();
            let (c, mut ie) = self.compress_kv(
                p,
                lid,
                &x[e * d..(e + 1) * d],
                &qr[e * ql..(e + 1) * ql],
                sp + e,
                offset,
            )?;
            ckv = c;
            idxs.push(ie.pop().unwrap_or_default());
            topk.append(&mut self.shared.topk);
            cand.append(&mut self.shared.cand);
        }
        self.shared.topk = topk;
        self.shared.cand = cand;
        Ok((ckv, idxs))
    }

    #[allow(clippy::too_many_arguments)]
    fn indexer(
        &mut self,
        p: &str,
        lid: usize,
        x: &[f32],
        qr: &[f32],
        latent: Option<&[f32]>,
        sp: usize,
        offset: usize,
    ) -> Result<Vec<Vec<i32>>> {
        let c = self.cfg.clone();
        let (r, rd, nh, ihd, hd) = (
            c.compress_ratios[lid],
            c.rope_head_dim,
            c.index_n_heads,
            c.index_head_dim,
            c.head_dim,
        );
        let t = x.len() / c.dim;
        let end = sp + t;
        let owns = c.is_kv_source(lid);
        if let (true, Some(lat)) = (owns, latent) {
            let n = lat.len() / hd;
            let mut k = self.linear(lat, n, &format!("{p}.attn.indexer.wk.weight"))?;
            self.rms(&mut k, &format!("{p}.attn.indexer.k_norm.weight"))?;
            let pos = |j: usize| if sp == 0 { j * r } else { sp + 1 - r };
            rope(&mut k, ihd, rd, 1, &self.freqs[lid], pos, false);
            fp4_qdq(&mut k, 32, false);
            self.idxk.get_mut(&lid).unwrap()[(sp / r) * ihd..(sp / r + n) * ihd]
                .copy_from_slice(&k);
        }
        if owns {
            self.shared.idxk = lid;
        }
        let mut q = self.linear(qr, t, &format!("{p}.attn.indexer.wq_b.weight"))?;
        rope(&mut q, ihd, rd, nh, &self.freqs[lid], |i| sp + i, false);
        fp4_qdq(&mut q, 32, false);
        let width = end / r;
        let ik = &self.idxk[&self.shared.idxk][..width * ihd];
        let mut wts = self.linear(x, t, &format!("{p}.attn.indexer.weights_proj.weight"))?;
        let ws = ((ihd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32;
        for v in wts.iter_mut() {
            *v = bf(*v * ws);
        }
        let mut score = vec![0f32; t * width];
        for ti in 0..t {
            for j in 0..width {
                let kj = &ik[j * ihd..(j + 1) * ihd];
                let mut s = 0f32;
                for hi in 0..nh {
                    let qv = &q[(ti * nh + hi) * ihd..(ti * nh + hi + 1) * ihd];
                    let d = bf(qv.iter().zip(kj).map(|(a, b)| a * b).sum());
                    s += bf(d.max(0.0) * wts[ti * nh + hi]);
                }
                score[ti * width + j] = bf(s);
            }
        }
        let lens: Vec<usize> = (0..t)
            .map(|ti| if sp == 0 { (ti + 1) / r } else { end / r })
            .collect();
        if sp == 0 {
            for ti in 0..t {
                for j in lens[ti]..width {
                    score[ti * width + j] = f32::NEG_INFINITY;
                }
            }
        }
        let src = c.candidate_source_layer;
        if lid as i64 == src {
            self.shared.cand = self.candidates(&score, width, &lens);
        } else if src >= 0 && (src as usize) < lid {
            for ti in 0..t {
                for j in 0..width {
                    if !self.shared.cand[ti][j] {
                        score[ti * width + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let k = c.index_topk.min(width);
        Ok((0..t)
            .map(|ti| {
                let mut sel = topk_stable(&score[ti * width..(ti + 1) * width], k);
                sel.sort();
                sel.into_iter()
                    .map(|i| {
                        if i < lens[ti] {
                            (i + offset) as i32
                        } else {
                            -1
                        }
                    })
                    .collect()
            })
            .collect())
    }

    /// Level one of the two-level top-k: the best `candidate_topk_blocks` blocks per query, the
    /// one holding its newest position pinned in.
    pub(crate) fn candidates(&self, score: &[f32], width: usize, lens: &[usize]) -> Vec<Vec<bool>> {
        let (bs, kb) = (
            self.cfg.candidate_block_size,
            self.cfg.candidate_topk_blocks,
        );
        let nb = width.div_ceil(bs);
        lens.iter()
            .enumerate()
            .map(|(ti, &len)| {
                let row = &score[ti * width..(ti + 1) * width];
                let mut sc: Vec<f32> = (0..nb)
                    .map(|b| {
                        row[b * bs..((b + 1) * bs).min(width)]
                            .iter()
                            .fold(f32::NEG_INFINITY, |m, &x| m.max(x))
                    })
                    .collect();
                let last = (len as i64 - 1).div_euclid(bs as i64);
                if last >= 0 && (last as usize) < nb {
                    sc[last as usize] = f32::INFINITY;
                }
                let mut keep = vec![false; nb];
                for b in topk_stable(&sc, kb.min(nb)) {
                    keep[b] = sc[b] > f32::NEG_INFINITY;
                }
                (0..width).map(|j| keep[j / bs]).collect()
            })
            .collect()
    }

    // moe

    fn moe(&self, p: &str, x: &[f32], lid: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (ne, topk) = c.moe(lid);
        let d = c.dim;
        let t = x.len() / d;
        let gate = self.v(&format!("{p}.ffn.gate.weight"))?;
        let bias = self.v(&format!("{p}.ffn.gate.bias"))?;
        let sc: Vec<f32> = matmul(x, t, &gate)
            .into_iter()
            .map(|s| softplus(s).sqrt())
            .collect();
        let mut routes: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
        for ti in 0..t {
            let row = &sc[ti * ne..(ti + 1) * ne];
            let biased: Vec<f32> = row.iter().zip(&bias.data).map(|(a, b)| a + b).collect();
            let sel = topk_stable(&biased, topk);
            let mut w: Vec<f32> = sel.iter().map(|&e| row[e]).collect();
            if c.norm_topk_prob && topk > 1 {
                let s: f32 = w.iter().sum::<f32>() + 1e-20;
                w.iter_mut().for_each(|v| *v /= s);
            }
            for (e, wv) in sel.into_iter().zip(w) {
                routes[e].push((ti, wv * c.route_scale));
            }
        }
        let mut y = vec![0f32; t * d];
        for (e, rs) in routes.iter().enumerate() {
            if rs.is_empty() {
                continue;
            }
            let xs: Vec<f32> = rs
                .iter()
                .flat_map(|&(ti, _)| x[ti * d..(ti + 1) * d].iter().copied())
                .collect();
            let wts: Vec<f32> = rs.iter().map(|&(_, w)| w).collect();
            let out = self.expert(&format!("{p}.ffn.experts.{e}"), &xs, Some(&wts))?;
            for (j, &(ti, _)) in rs.iter().enumerate() {
                for (a, b) in y[ti * d..(ti + 1) * d]
                    .iter_mut()
                    .zip(&out[j * d..(j + 1) * d])
                {
                    *a += b;
                }
            }
        }
        let sh = self.expert(&format!("{p}.ffn.shared_experts"), x, None)?;
        for (a, b) in y.iter_mut().zip(&sh) {
            *a = bf(*a + b);
        }
        Ok(y)
    }

    fn expert(&self, p: &str, x: &[f32], wts: Option<&[f32]>) -> Result<Vec<f32>> {
        let t = x.len() / self.cfg.dim;
        let lim = self.cfg.swiglu_limit;
        let g = self.linear(x, t, &format!("{p}.w1.weight"))?;
        let u = self.linear(x, t, &format!("{p}.w3.weight"))?;
        let inter = g.len() / t;
        let mut h: Vec<f32> = g
            .iter()
            .zip(&u)
            .enumerate()
            .map(|(i, (&g, &u))| {
                let (g, u) = if lim > 0.0 {
                    (g.min(lim), u.clamp(-lim, lim))
                } else {
                    (g, u)
                };
                let v = g / (1.0 + (-g).exp()) * u;
                bf(match wts {
                    Some(w) => w[i / inter] * v,
                    None => v,
                })
            })
            .collect();
        bf_all(&mut h);
        self.linear(&h, t, &format!("{p}.w2.weight"))
    }

    // hyper-connections and blocks

    fn hc_mixes(
        &self,
        p: &str,
        kind: &str,
        x: &[f32],
    ) -> Result<Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
        let c = &self.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let f = self.v(&format!("{p}.hc_{kind}_fn"))?;
        let scale = self.v(&format!("{p}.hc_{kind}_scale"))?;
        let base = self.v(&format!("{p}.hc_{kind}_base"))?;
        let t = x.len() / (hc * d);
        let mixes = matmul(x, t, &f);
        let mix = f.rows;
        Ok((0..t)
            .map(|ti| {
                let xr = &x[ti * hc * d..(ti + 1) * hc * d];
                let rs = 1.0
                    / (xr.iter().map(|v| v * v).sum::<f32>() / (hc * d) as f32 + c.norm_eps).sqrt();
                let m: Vec<f32> = mixes[ti * mix..(ti + 1) * mix]
                    .iter()
                    .map(|v| v * rs)
                    .collect();
                sinkhorn(
                    &m,
                    &scale.data,
                    &base.data,
                    hc,
                    c.hc_sinkhorn_iters,
                    c.hc_eps,
                )
            })
            .collect())
    }

    fn hc_pre(&self, x: &[f32], pre: &[Vec<f32>]) -> Vec<f32> {
        let (hc, d) = (self.cfg.hc_mult, self.cfg.dim);
        let mut out = vec![0f32; pre.len() * d];
        for (ti, pm) in pre.iter().enumerate() {
            for (ci, &w) in pm.iter().enumerate() {
                let xr = &x[(ti * hc + ci) * d..(ti * hc + ci + 1) * d];
                for (o, v) in out[ti * d..(ti + 1) * d].iter_mut().zip(xr) {
                    *o += w * v;
                }
            }
        }
        bf_all(&mut out);
        out
    }

    fn hc_post(
        &self,
        h: &[f32],
        res: &[f32],
        mixes: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> Vec<f32> {
        let (hc, d) = (self.cfg.hc_mult, self.cfg.dim);
        let mut out = vec![0f32; res.len()];
        for (ti, (_, post, comb)) in mixes.iter().enumerate() {
            for k in 0..hc {
                let o = &mut out[(ti * hc + k) * d..(ti * hc + k + 1) * d];
                for (i, v) in o.iter_mut().enumerate() {
                    let mut s = post[k] * h[ti * d + i];
                    let mut acc = 0f32;
                    for j in 0..hc {
                        acc += comb[j * hc + k] * res[(ti * hc + j) * d + i];
                    }
                    s += acc;
                    *v = bf(s);
                }
            }
        }
        out
    }

    /// One block: mHC-wrapped attention then MoE. Returns the stream and the FFN's pre-mix,
    /// which the next block's attention collapses with.
    fn block(
        &mut self,
        p: &str,
        lid: usize,
        x: Vec<f32>,
        pre_mix: &[Vec<f32>],
        attn: &mut dyn FnMut(&mut Self, &[f32]) -> Result<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let am = self.hc_mixes(p, "attn", &x)?;
        let mut h = self.hc_pre(&x, pre_mix);
        self.tap(&format!("{p}.attn_pre"), &h);
        self.rms(&mut h, &format!("{p}.attn_norm.weight"))?;
        self.tap(&format!("{p}.attn_in"), &h);
        let h = attn(self, &h)?;
        self.tap(&format!("{p}.attn"), &h);
        let x = self.hc_post(&h, &x, &am);
        let fm = self.hc_mixes(p, "ffn", &x)?;
        let a_pre: Vec<Vec<f32>> = am.iter().map(|m| m.0.clone()).collect();
        let mut h = self.hc_pre(&x, &a_pre);
        self.rms(&mut h, &format!("{p}.ffn_norm.weight"))?;
        self.tap(&format!("{p}.moe_in"), &h);
        let h = self.moe(p, &h, lid)?;
        self.tap(&format!("{p}.moe"), &h);
        let x = self.hc_post(&h, &x, &fm);
        Ok((x, fm.into_iter().map(|m| m.0).collect()))
    }

    // engram

    /// Hash ids [T][n_engram_layers][cols] of the n-grams ending at each new position.
    pub(crate) fn engram_hash(&mut self, ids: &[usize], sp: usize) -> Vec<Vec<Vec<i64>>> {
        let em = self.engram.as_ref().unwrap();
        let n = self.cfg.engram_max_ngram_size;
        for (i, &id) in ids.iter().enumerate() {
            self.hash_cache[sp + i] = em.token_map[id];
        }
        (0..ids.len())
            .map(|i| {
                let pos = sp + i;
                let toks: Vec<i64> = (0..n)
                    .map(|s| {
                        if pos < s {
                            em.pad
                        } else {
                            self.hash_cache[pos - s]
                        }
                    })
                    .collect();
                em.multipliers
                    .iter()
                    .enumerate()
                    .map(|(l, mult)| {
                        let mut rolling = toks[0].wrapping_mul(mult[0]);
                        let mut cols = Vec::new();
                        for g in 1..n {
                            rolling ^= toks[g].wrapping_mul(mult[g]);
                            for &pr in &em.primes[l][g - 1] {
                                cols.push(rolling.rem_euclid(pr));
                            }
                        }
                        cols.iter()
                            .zip(&em.offsets[l])
                            .map(|(a, b)| a + b)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn engram(&self, p: &str, x: &[f32], hashes: &[Vec<i64>]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let t = hashes.len();
        let table = self.w(&format!("{p}.engram.embed.weight"))?;
        let ehd = table.cols;
        let rows: Vec<f32> = hashes
            .iter()
            .flat_map(|cols| {
                cols.iter().flat_map(|&r| {
                    table.data[r as usize * ehd..(r as usize + 1) * ehd]
                        .iter()
                        .map(|v| bf(*v))
                })
            })
            .collect();
        self.tap(&format!("{p}.eg_rows"), &rows);
        let kv = self.linear(&rows, t, &format!("{p}.engram.wkv.weight"))?;
        self.tap(&format!("{p}.eg_kv"), &kv);
        let qw = self.v(&format!("{p}.engram.q_weight"))?;
        let kw = self.v(&format!("{p}.engram.k_weight"))?;
        let mut out = vec![0f32; x.len()];
        for ti in 0..t {
            let kvr = &kv[ti * d * (hc + 1)..(ti + 1) * d * (hc + 1)];
            let val = &kvr[hc * d..];
            for ci in 0..hc {
                let key = &kvr[ci * d..(ci + 1) * d];
                let h = &x[(ti * hc + ci) * d..(ti * hc + ci + 1) * d];
                let ms = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>() / d as f32;
                let rstd =
                    1.0 / (ms(h) + c.norm_eps).sqrt() * (1.0 / (ms(key) + c.norm_eps).sqrt());
                let mut dot = 0f32;
                for i in 0..d {
                    dot += h[i] * (qw.data[ci * d + i] * kw.data[ci * d + i]) * key[i];
                }
                let dot = dot * rstd * (d as f32).powf(-0.5);
                let gate = sigmoid(dot.abs().max(1e-6).sqrt().copysign(dot));
                for i in 0..d {
                    out[(ti * hc + ci) * d + i] = bf(h[i] + gate * val[i]);
                }
            }
        }
        Ok(out)
    }

    // the forward

    fn embed_hc(&self, ids: &[usize]) -> Result<Vec<f32>> {
        let (hc, d) = (self.cfg.hc_mult, self.cfg.dim);
        let e = self.w("embed.weight")?;
        let mut h = Vec::with_capacity(ids.len() * hc * d);
        for &id in ids {
            for _ in 0..hc {
                h.extend_from_slice(&e.data[id * d..(id + 1) * d]);
            }
        }
        Ok(h)
    }

    fn logits(&self, x: &[f32], head: &str) -> Result<Vec<f32>> {
        let w = self.w(head)?;
        Ok(matmul(x, x.len() / w.cols, &w))
    }

    /// Feed `ids` at `sp`. Returns the last position's logits [V] and the DSpark target hiddens
    /// [T][D * n_targets]. `hook(name, tensor)` sees the engram hashes and each block's stream.
    pub fn forward(
        &mut self,
        ids: &[usize],
        sp: usize,
        hook: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, hook, false)
    }

    /// [`Self::forward`] with every position's logits [T][V]: the verify window of speculative
    /// decoding. A chunk at sp > 0 is the same math as feeding its tokens one at a time.
    pub fn forward_all(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, &mut |_, _| {}, true)
    }

    fn forward_impl(
        &mut self,
        ids: &[usize],
        sp: usize,
        hook: &mut dyn FnMut(&str, &[f32]),
        all: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = self.cfg.clone();
        let (hc, d, t) = (c.hc_mult, c.dim, ids.len());
        let hashes = if self.engram.is_some() {
            Some(self.engram_hash(ids, sp))
        } else {
            None
        };
        if let Some(hs) = &hashes {
            let flat: Vec<f32> = hs.iter().flatten().flatten().map(|&v| v as f32).collect();
            hook("hash", &flat);
        }
        let mut h = self.embed_hc(ids)?;
        let mut pre_mix: Vec<Vec<f32>> = (0..t)
            .map(|_| (0..hc).map(|i| (i == 0) as u8 as f32).collect())
            .collect();
        let mut mains: Vec<Vec<f32>> = Vec::new();
        for i in 0..c.n_layers {
            let p = format!("layers.{i}");
            if let Some(f) = &self.force {
                h = f[i].0.clone();
                pre_mix = f[i].1.chunks(hc).map(|r| r.to_vec()).collect();
            }
            if let Some(j) = c.engram_layer_ids.iter().position(|&l| l == i) {
                let hs: Vec<Vec<i64>> = hashes
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|ht| ht[j].clone())
                    .collect();
                h = self.engram(&p, &h, &hs)?;
            }
            if c.dspark_target_layer_ids.contains(&i) {
                let mut m = vec![0f32; t * d];
                for ti in 0..t {
                    for ci in 0..hc {
                        for k in 0..d {
                            m[ti * d + k] += h[(ti * hc + ci) * d + k];
                        }
                    }
                }
                m.iter_mut().for_each(|v| *v = bf(*v / hc as f32));
                mains.push(m);
            }
            let pp = p.clone();
            let (nh, npm) = self.block(&p, i, h, &pre_mix, &mut |m: &mut Self, a: &[f32]| {
                m.attention(&pp, i, a, sp)
            })?;
            h = nh;
            pre_mix = npm;
            hook(&format!("layer{i}"), &h);
        }
        let mut x = self.hc_pre(&h, &pre_mix);
        self.rms(&mut x, "norm.weight")?;
        let logits = self.logits(&x[if all { 0 } else { (t - 1) * d }..], "head.weight")?;
        let n = mains.len();
        let mut main = vec![0f32; t * d * n];
        for ti in 0..t {
            for (j, m) in mains.iter().enumerate() {
                main[(ti * n + j) * d..(ti * n + j + 1) * d]
                    .copy_from_slice(&m[ti * d..(ti + 1) * d]);
            }
        }
        Ok((logits, main))
    }

    // dspark

    fn dspark_attention(
        &mut self,
        p: &str,
        lid: usize,
        x: &[f32],
        sp: usize,
        main_x: &[f32],
    ) -> Result<Vec<f32>> {
        let c = self.cfg.clone();
        let (h, hd, rd, win, d) = (c.n_heads, c.head_dim, c.rope_head_dim, c.window_size, c.dim);
        let t = main_x.len() / d;
        let mut mkv = self.linear(main_x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(&mut mkv, &format!("{p}.attn.kv_norm.weight"))?;
        let tab = self.freqs[lid].clone();
        rope(&mut mkv, hd, rd, 1, &tab, |i| sp + i, false);
        fp8_qdq(&mut mkv, 32);
        if sp == 0 {
            let ring = &mut self.win[lid];
            if t <= win {
                ring[..t * hd].copy_from_slice(&mkv);
            } else {
                let cut = t % win;
                let tail = &mkv[(t - win) * hd..];
                ring[cut * hd..].copy_from_slice(&tail[..(win - cut) * hd]);
                ring[..cut * hd].copy_from_slice(&tail[(win - cut) * hd..]);
            }
            return Ok(x.to_vec());
        }
        let b = x.len() / d;
        let pos0 = sp + t;
        let mut qr = self.linear(x, b, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(&mut qr, &format!("{p}.attn.q_norm.weight"))?;
        let mut q = self.linear(&qr, b, &format!("{p}.attn.wq_b.weight"))?;
        rope(&mut q, hd, rd, h, &tab, |i| pos0 + i, false);
        let mut kv = self.linear(x, b, &format!("{p}.attn.wkv.weight"))?;
        self.rms(&mut kv, &format!("{p}.attn.kv_norm.weight"))?;
        rope(&mut kv, hd, rd, 1, &tab, |i| pos0 + i, false);
        fp8_qdq(&mut kv, 32);
        let row: Vec<i32> = (0..win.min(sp + t) as i32)
            .chain((0..b as i32).map(|i| win as i32 + i))
            .collect();
        let idxs = vec![row; b];
        let ring = &mut self.win[lid];
        // main rows for positions sp..sp+t (one at decode, the accepted ones after a verify)
        for i in 0..t {
            let s = (sp + i) % win;
            ring[s * hd..(s + 1) * hd].copy_from_slice(&mkv[i * hd..(i + 1) * hd]);
        }
        let mut all = ring.clone();
        all.extend_from_slice(&kv);
        self.attn_out(p, &q, &all, &idxs, &tab, pos0)
    }

    /// The DSpark drafter after the main model sampled `token` with `main_hidden` from that
    /// step. Prefill (`sp` 0) only seeds each stage's window and returns `None`.
    pub fn forward_spec(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
    ) -> Result<Option<Draft>> {
        self.forward_spec_with(token, main_hidden, sp, &mut |q: &[f32]| argmax(q))
    }

    /// [`Self::forward_spec`] with the drafts drawn by `draw` from each position's draft logits
    /// (Markov bias included, the q of rejection sampling): argmax when greedy, a sample from
    /// q otherwise. The next position's Markov bias follows the drawn token.
    pub fn forward_spec_with(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
        draw: &mut dyn FnMut(&[f32]) -> usize,
    ) -> Result<Option<Draft>> {
        let c = self.cfg.clone();
        let (hc, d, blk, n_mtp) = (c.hc_mult, c.dim, c.dspark_block_size, c.n_mtp_layers);
        let t = main_hidden.len() / (d * c.dspark_target_layer_ids.len());
        let mut main_x = self.linear(main_hidden, t, "mtp.0.main_proj.weight")?;
        self.rms(&mut main_x, "mtp.0.main_norm.weight")?;
        let mut ids = vec![c.dspark_noise_token_id; blk];
        ids[0] = token;
        let mut h = self.embed_hc(&ids)?;
        let mut pre_mix: Vec<Vec<f32>> = (0..blk)
            .map(|_| (0..hc).map(|i| (i == 0) as u8 as f32).collect())
            .collect();
        for s in 0..n_mtp {
            let (lid, p) = (c.n_layers + s, format!("mtp.{s}"));
            if sp == 0 {
                self.dspark_attention(&p, lid, &h, 0, &main_x)?;
                continue;
            }
            let (pp, mx) = (p.clone(), main_x.clone());
            let (nh, npm) = self.block(&p, lid, h, &pre_mix, &mut |m: &mut Self, a: &[f32]| {
                m.dspark_attention(&pp, lid, a, sp, &mx)
            })?;
            h = nh;
            pre_mix = npm;
        }
        if sp == 0 {
            return Ok(None);
        }
        let last = format!("mtp.{}", n_mtp - 1);
        let x = self.hc_pre(&h, &pre_mix);
        let mut xn = x.clone();
        self.rms(&mut xn, &format!("{last}.norm.weight"))?;
        let mut logits = self.logits(&xn, "head.weight")?;
        let v = c.vocab_size;
        let rank = c.dspark_markov_rank;
        let memb_w = self.w(&format!("{last}.markov_head.embed.weight"))?;
        let mhead = self.w(&format!("{last}.markov_head.head.weight"))?;
        let proj = self.w(&format!("{last}.confidence_head.proj.weight"))?;
        let mut out = vec![token];
        let mut conf = Vec::with_capacity(blk);
        for i in 0..blk {
            let memb = &memb_w.data[out[i] * rank..(out[i] + 1) * rank];
            let bias = matmul(memb, 1, &mhead);
            let li = &mut logits[i * v..(i + 1) * v];
            for (a, b) in li.iter_mut().zip(&bias) {
                *a += b;
            }
            out.push(draw(li));
            let feat: Vec<f32> = x[i * d..(i + 1) * d].iter().chain(memb).copied().collect();
            conf.push(matmul(&feat, 1, &proj)[0]);
        }
        Ok(Some(Draft {
            ids: out,
            logits,
            conf,
        }))
    }
}

/// The state a verify window can damage past the accepted position: the backbone's window
/// rings (a slot written for a rejected position held the position one window earlier) and
/// the ratio-r compressors' partial group (a rejected position overwrites the slot of a kept
/// one in the same slot class; restored from the snapshot plus the journal of kept writes).
/// Everything else is position-indexed and rewritten before it is read again: compressed KV and
/// index keys are read only up to the groups complete at the query (`clen`), the Engram hash
/// cache is written before it is read, and `shared` is recomputed every forward.
pub struct Snapshot {
    win: Vec<Vec<f32>>,
    cstate: HashMap<usize, (Vec<f32>, Vec<f32>)>,
}

/// One speculative step's result.
pub struct SpecStep {
    /// The new tokens: the accepted drafts, then the main model's own next token.
    pub tokens: Vec<usize>,
    pub drafted: usize,
    pub accepted: usize,
    /// DSpark target hiddens [k+1][D * n] of the fed positions `sp..sp+k+1`, for the next
    /// [`Model::decode_spec`] (with `sp` as its start).
    pub main: Vec<f32>,
    pub sp: usize,
}

pub fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// Splitmix64, for seeded sampling.
pub fn next_u64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn uniform(rng: &mut u64) -> f64 {
    (next_u64(rng) >> 11) as f64 / (1u64 << 53) as f64
}

/// How [`Model::decode_spec`] picks tokens: the drafts from the drafter's logits, and at each
/// verify row the kept token. Greedy ([`Greedy`]) or speculative sampling ([`SpecSampler`]);
/// the GPU engine takes the same object.
pub trait SpecPick {
    /// A draft from draft logits `q` (Markov bias included).
    fn draft(&mut self, q: &[f32]) -> usize;
    /// The token at a verify row with target logits `p`. With `Some((q, d))` the row checks
    /// draft `d` drawn from `q`: returns `(d, true)` to accept it, else `(replacement, false)`.
    /// With `None` (every draft accepted, or no drafts) the bonus token from `p`.
    fn verify(&mut self, p: &[f32], draft: Option<(&[f32], usize)>) -> (usize, bool);
    /// The concrete sampler, for callers that must copy its state (expert parallelism runs
    /// one per rank).
    fn as_sampler(&mut self) -> Option<&mut SpecSampler> {
        None
    }
    /// Whether every pick is an argmax with no RNG draws (so a verify walk needs only each
    /// row's argmax, see [`verify_greedy`]).
    fn greedy(&self) -> bool {
        false
    }
    /// Whether [`Self::draft`] is the argmax (and consumes no state), so the engine may draw
    /// drafts on the device.
    fn greedy_draft(&self) -> bool {
        self.greedy()
    }
}

/// Temperature 0: argmax drafts, a draft is kept iff it is the target's argmax.
pub struct Greedy;

impl SpecPick for Greedy {
    fn greedy(&self) -> bool {
        true
    }
    fn draft(&mut self, q: &[f32]) -> usize {
        argmax(q)
    }
    fn verify(&mut self, p: &[f32], draft: Option<(&[f32], usize)>) -> (usize, bool) {
        let t = argmax(p);
        (t, draft.is_some_and(|(_, d)| d == t))
    }
}

/// Temperature + nucleus sampling, and speculative sampling (Leviathan et al. / Chen et al.)
/// over it. Both the target p and the draft q go through the same transform (softmax of
/// logits / temperature, then top-p truncation and renormalisation), and rejection sampling
/// runs between the transformed P and Q, so each emitted token is distributed exactly as a
/// plain draw from P: what [`Self::sample`] would give decoding one token at a time. At
/// temperature 0 it is [`Greedy`] (the transform is a point mass at the argmax).
#[derive(Clone, Debug)]
pub struct SpecSampler {
    pub temp: f32,
    pub top_p: f32,
    pub rng: u64,
}

impl SpecSampler {
    pub fn new(temp: f32, top_p: f32, seed: u64) -> Self {
        Self {
            temp,
            top_p,
            rng: seed,
        }
    }

    pub fn greedy(&self) -> bool {
        self.temp <= 0.0
    }

    /// The transformed distribution: (token, probability) by decreasing probability (ties
    /// by token), cut to the smallest prefix holding `top_p` of the mass, renormalised.
    pub fn dist(&self, logits: &[f32]) -> Vec<(usize, f64)> {
        if self.greedy() {
            return vec![(argmax(logits), 1.0)];
        }
        let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let w: Vec<f64> = logits
            .iter()
            .map(|&l| (((l - mx) / self.temp) as f64).exp())
            .collect();
        let z: f64 = w.iter().sum();
        let by = |a: &usize, b: &usize| w[*b].partial_cmp(&w[*a]).unwrap().then(a.cmp(b));
        let mut idx: Vec<usize> = (0..w.len()).collect();
        if self.top_p < 1.0 {
            // partial selection: grow the head until it holds top_p of the mass
            let mut k = 256.min(idx.len());
            loop {
                if k < idx.len() {
                    idx.select_nth_unstable_by(k - 1, by);
                }
                let head: f64 = idx[..k].iter().map(|&i| w[i]).sum();
                if head / z >= self.top_p as f64 || k == idx.len() {
                    break;
                }
                k = (k * 4).min(idx.len());
            }
            idx.truncate(k);
        }
        idx.sort_unstable_by(by);
        let mut keep = idx.len();
        if self.top_p < 1.0 {
            let mut acc = 0.0;
            for (j, &i) in idx.iter().enumerate() {
                acc += w[i] / z;
                if acc >= self.top_p as f64 {
                    keep = j + 1;
                    break;
                }
            }
        }
        idx.truncate(keep);
        let total: f64 = idx.iter().map(|&i| w[i]).sum();
        idx.into_iter().map(|i| (i, w[i] / total)).collect()
    }

    fn draw(&mut self, d: &[(usize, f64)]) -> usize {
        let mut r = uniform(&mut self.rng) * d.iter().map(|x| x.1).sum::<f64>();
        for &(i, p) in d {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        d.last().unwrap().0
    }

    /// One token from `logits`: argmax at temperature 0, else a draw from [`Self::dist`].
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        if self.greedy() {
            return argmax(logits);
        }
        let d = self.dist(logits);
        self.draw(&d)
    }
}

impl SpecPick for SpecSampler {
    fn as_sampler(&mut self) -> Option<&mut SpecSampler> {
        Some(self)
    }
    fn greedy(&self) -> bool {
        SpecSampler::greedy(self)
    }
    fn draft(&mut self, q: &[f32]) -> usize {
        self.sample(q)
    }

    fn verify(&mut self, p: &[f32], draft: Option<(&[f32], usize)>) -> (usize, bool) {
        if self.greedy() {
            return Greedy.verify(p, draft);
        }
        let pd = self.dist(p);
        let Some((q, d)) = draft else {
            return (self.draw(&pd), false);
        };
        let qd = self.dist(q);
        let at = |x: &[(usize, f64)], t: usize| x.iter().find(|e| e.0 == t).map_or(0.0, |e| e.1);
        let (pp, qq) = (at(&pd, d), at(&qd, d));
        // accept with probability min(1, P(d) / Q(d))
        if qq > 0.0 && uniform(&mut self.rng) * qq < pp {
            return (d, true);
        }
        // else draw from norm(max(0, P - Q))
        let mut qm: HashMap<usize, f64> = qd.into_iter().collect();
        let res: Vec<(usize, f64)> = pd
            .iter()
            .map(|&(i, x)| (i, (x - qm.remove(&i).unwrap_or(0.0)).max(0.0)))
            .filter(|e| e.1 > 0.0)
            .collect();
        if res.is_empty() {
            // P == Q up to rounding: the residual is empty, P itself is exact
            return (self.draw(&pd), false);
        }
        (self.draw(&res), false)
    }
}

impl Model {
    /// Open a rewind point (until the next [`Self::rewind`]).
    pub fn snapshot(&mut self) -> Snapshot {
        self.journal = Some(Vec::new());
        Snapshot {
            win: self.win[..self.cfg.n_layers].to_vec(),
            cstate: self.cstate.clone(),
        }
    }

    /// Undo a forward of positions `from..end` past `from`: afterwards the state is as if only
    /// positions `< from` had been fed.
    pub fn rewind(&mut self, snap: &Snapshot, from: usize, end: usize) {
        let (win, hd) = (self.cfg.window_size, self.cfg.head_dim);
        for (ring, old) in self.win.iter_mut().zip(&snap.win) {
            for pos in from..end.min(from + win) {
                let s = (pos % win) * hd;
                ring[s..s + hd].copy_from_slice(&old[s..s + hd]);
            }
        }
        self.cstate = snap.cstate.clone();
        for (lid, pos, kv, sc) in self.journal.take().unwrap_or_default() {
            if pos < from {
                let (kst, sst) = self.cstate.get_mut(&lid).unwrap();
                let s = (pos % self.cfg.compress_ratios[lid]) * hd;
                kst[s..s + hd].copy_from_slice(&kv);
                sst[s..s + hd].copy_from_slice(&sc);
            }
        }
    }

    /// One speculative step. `token` is the sampled, not yet fed, token at position
    /// `sp + rows(main)`; `main` the target hiddens of positions `sp..`. Drafts with DSpark
    /// (each draft from `pick.draft`), verifies `[token, drafts]` as one window, walks the rows
    /// with `pick.verify` until the first rejection (whose replacement, or the bonus token
    /// after a fully accepted window, ends the step) and rewinds the rest. With [`Greedy`] this
    /// is greedy decoding; with [`SpecSampler`] speculative sampling. `edit(pos, drafts)` may
    /// rewrite the drafts, the first at position `pos` (tests force wrong ones; sampling's
    /// guarantee assumes unedited drafts).
    pub fn decode_spec(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        edit: &mut dyn FnMut(usize, &mut [usize]),
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep> {
        let c = &self.cfg;
        let (v, w) = (c.vocab_size, c.dim * c.dspark_target_layer_ids.len());
        let pos0 = sp + main.len() / w;
        let (greedy, full) = (pick.greedy(), c.dspark_block_size);
        self.cfg.dspark_block_size = self.cfg.draft_block(greedy);
        let draft = self.forward_spec_with(token, main, sp, &mut |q: &[f32]| pick.draft(q));
        self.cfg.dspark_block_size = full;
        let draft = draft?.ok_or_else(|| anyhow::anyhow!("decode_spec needs sp > 0"))?;
        let mut ids = draft.ids;
        ids.truncate(1 + self.cfg.draft_keep(&draft.conf, greedy));
        edit(pos0 + 1, &mut ids[1..]);
        let snap = self.snapshot();
        let (logits, mains) = self.forward_all(&ids, pos0)?;
        let n = ids.len();
        let (k, tokens) = verify_rows(&logits, &draft.logits, &ids, v, pick);
        self.rewind(&snap, pos0 + k + 1, pos0 + n);
        Ok(SpecStep {
            tokens,
            drafted: n - 1,
            accepted: k,
            main: mains[..(k + 1) * w].to_vec(),
            sp: pos0,
        })
    }
}

/// The accept walk of a verify window: `logits` [n][v] of the window `ids` (the fed token,
/// then n-1 drafts drawn from the rows of `q` [n-1][v]). Returns the accepted count k and the
/// new tokens (`ids[1..=k]`, then the replacement or bonus token).
pub fn verify_rows(
    logits: &[f32],
    q: &[f32],
    ids: &[usize],
    v: usize,
    pick: &mut dyn SpecPick,
) -> (usize, Vec<usize>) {
    verify_rows_with(
        &mut |k| logits[k * v..(k + 1) * v].to_vec(),
        q,
        ids,
        v,
        pick,
    )
}

/// [`verify_rows`] with the target rows fetched on demand (`row(k)`, in order, each at most
/// once): a walk that stops at row k reads k + 1 rows.
pub fn verify_rows_with(
    row: &mut dyn FnMut(usize) -> Vec<f32>,
    q: &[f32],
    ids: &[usize],
    v: usize,
    pick: &mut dyn SpecPick,
) -> (usize, Vec<usize>) {
    let n = ids.len();
    let mut tokens = Vec::with_capacity(n);
    let mut k = 0;
    loop {
        let row = row(k);
        let row = &row[..];
        if k + 1 == n {
            tokens.push(pick.verify(row, None).0);
            return (k, tokens);
        }
        let (t, ok) = pick.verify(row, Some((&q[k * v..(k + 1) * v], ids[k + 1])));
        tokens.push(t);
        if !ok {
            return (k, tokens);
        }
        k += 1;
    }
}

/// [`verify_rows`] for a greedy pick from each row's argmax `amax` alone: drafts are kept
/// while they equal the target's argmax, then that row's argmax (the replacement, or the
/// bonus token after a fully accepted window).
pub fn verify_greedy(amax: &[usize], ids: &[usize]) -> (usize, Vec<usize>) {
    let n = ids.len();
    let mut k = 0;
    while k + 1 < n && ids[k + 1] == amax[k] {
        k += 1;
    }
    (k, amax[..=k].to_vec())
}

/// The model side of [`generate`]: the CPU [`Model`] here, the GPU engine in `serve`.
pub trait Decoder {
    fn config(&self) -> &Config;
    /// Feed `ids` at `sp`: the last logits and the DSpark target hiddens.
    fn feed(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)>;
    /// Seed the drafter's windows with the prompt's hiddens.
    fn seed_drafter(&mut self, main: &[f32]) -> Result<()>;
    /// Whether [`Self::spec_step`] supports this pick (greedy or sampled).
    fn spec_ok(&self, greedy: bool) -> bool;
    /// [`Model::decode_spec`] without the edit hook.
    fn spec_step(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep>;
}

impl Decoder for Model {
    fn config(&self) -> &Config {
        &self.cfg
    }
    fn feed(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward(ids, sp, &mut |_, _| {})
    }
    fn seed_drafter(&mut self, main: &[f32]) -> Result<()> {
        self.forward_spec(0, main, 0).map(|_| ())
    }
    fn spec_ok(&self, _: bool) -> bool {
        true
    }
    fn spec_step(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep> {
        self.decode_spec(token, main, sp, &mut |_, _| {}, pick)
    }
}

/// Decode after a prompt: prefill `ids` at position 0, then up to `max_new` tokens, each
/// handed to `on` (which ends the run by returning a [`Finish`]). With `spec`, every forward
/// after the first decode step is a DSpark verify window ([`Decoder::spec_step`]) at any
/// temperature the engine supports; the window's extra tokens are emitted one by one, so EOS,
/// `on`'s stop and `max_new` cut a window short exactly where plain decoding would stop.
/// Fills `usage`'s token counts, rates, draft/accepted counts and `decode_steps` (forwards
/// after the prefill; `completion_tokens / decode_steps` is the tokens per forward).
#[allow(clippy::too_many_arguments)]
pub fn generate(
    m: &mut dyn Decoder,
    ids: &[usize],
    max_new: usize,
    pick: &mut SpecSampler,
    spec: bool,
    eos: &[usize],
    usage: &mut Usage,
    on: &mut dyn FnMut(usize) -> Option<Finish>,
) -> Result<Finish> {
    let ctx = m.config().max_seq_len;
    let blk = m.config().dspark_block_size;
    let t0 = std::time::Instant::now();
    let (mut logits, mut main) = m.feed(ids, 0)?;
    usage.prompt_tokens = ids.len();
    usage.prefill_chunks += 1;
    usage.prefill_tok_s = ids.len() as f64 / t0.elapsed().as_secs_f64().max(1e-9);
    let spec = spec && m.spec_ok(pick.greedy());
    if spec {
        m.seed_drafter(&main)?;
    }
    let t1 = std::time::Instant::now();
    // tokens a verify window produced: (token, already fed)
    let mut queue: std::collections::VecDeque<(usize, bool)> = Default::default();
    // the start position of `main`'s rows, once it holds a decode step's hiddens
    let mut main_sp: Option<usize> = None;
    let mut pos = ids.len();
    let mut finish = Finish::Length;
    let mut n = 0;
    while n < max_new {
        let (next, fed) = queue
            .pop_front()
            .unwrap_or_else(|| (pick.sample(&logits), false));
        if eos.contains(&next) {
            finish = Finish::Stop;
            break;
        }
        n += 1;
        usage.completion_tokens += 1;
        if let Some(f) = on(next) {
            finish = f;
            break;
        }
        if n == max_new || pos + 1 >= ctx {
            break;
        }
        if fed {
            pos += 1;
            continue;
        }
        usage.decode_steps += 1;
        match main_sp {
            Some(sp) if spec && pos + blk + 1 < ctx => {
                let s = m.spec_step(next, &main, sp, pick)?;
                usage.draft_tokens += s.drafted;
                usage.accepted_tokens += s.accepted;
                let k = s.tokens.len();
                queue.extend(s.tokens.iter().enumerate().map(|(i, &t)| (t, i + 1 < k)));
                main = s.main;
                main_sp = Some(s.sp);
            }
            _ => {
                (logits, main) = m.feed(&[next], pos)?;
                main_sp = Some(pos);
            }
        }
        pos += 1;
    }
    usage.decode_tok_s = usage.completion_tokens as f64 / t1.elapsed().as_secs_f64().max(1e-9);
    Ok(finish)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The fixture `scripts/deepseek_tiny.py` writes (`DS41_TINY` overrides).
    pub(crate) fn tiny() -> String {
        std::env::var("DS41_TINY").unwrap_or_else(|_| "/private/tmp/ds41-tiny".into())
    }

    fn load_ref(name: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(format!("{}/ref/{name}.bin", tiny())).ok()?;
        Some(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let m = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / m
    }

    fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
        let ls = |l: &[f32]| {
            let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
            l.iter().map(|&x| x as f64 - z).collect::<Vec<f64>>()
        };
        let (p, q) = (ls(p_logits), ls(q_logits));
        p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum()
    }

    /// Each layer fed the reference's own input stream and pre-mix, and the drafter the
    /// reference's hiddens, so every error is that component's alone. This is the parity bar:
    /// free-running, the tiny random model turns ulp differences into indexer flips (as the
    /// reference does between its own prefill and decode paths; see `tiny_free_running`).
    #[test]
    fn tiny_teacher_forced() {
        if !Path::new(&tiny()).join("ref/step0_in0.bin").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return;
        }
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).unwrap(),
        )
        .unwrap();
        let steps = meta["steps"].as_array().unwrap();
        let mut m = Model::open(&Path::new(&tiny()).join("hf")).unwrap();
        for (s, step) in steps.iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            m.force = Some(
                (0..m.cfg.n_layers)
                    .map(|i| {
                        (
                            load_ref(&format!("step{s}_in{i}")).unwrap(),
                            load_ref(&format!("step{s}_pm{i}")).unwrap(),
                        )
                    })
                    .collect(),
            );
            let mut got: HashMap<String, Vec<f32>> = HashMap::new();
            let (logits, _) = m
                .forward(&ids, sp, &mut |n, t| {
                    got.insert(n.to_string(), t.to_vec());
                })
                .unwrap();
            let rels: Vec<String> = (0..m.cfg.n_layers)
                .map(|i| {
                    format!(
                        "{:.1e}",
                        rel(
                            &got[&format!("layer{i}")],
                            &load_ref(&format!("step{s}_layer{i}")).unwrap()
                        )
                    )
                })
                .collect();
            let k = kl(&load_ref(&format!("step{s}_logits")).unwrap(), &logits);
            // the drafter reads the reference's own hiddens, so its error is its own too
            let main = load_ref(&format!("step{s}_main_hidden")).unwrap();
            let next = step["next_id"].as_u64().unwrap() as usize;
            let draft = m.forward_spec(next, &main, sp).unwrap();
            let mut spec = String::new();
            if let (Some(d), Some(want)) = (draft, step["spec_ids"].as_array()) {
                let want: Vec<usize> = want.iter().map(|v| v.as_u64().unwrap() as usize).collect();
                let r = load_ref(&format!("step{s}_spec_logits")).unwrap();
                let v = m.cfg.vocab_size;
                let kls: Vec<String> = (0..m.cfg.dspark_block_size)
                    .map(|i| {
                        format!(
                            "{:.0e}",
                            kl(&r[i * v..(i + 1) * v], &d.logits[i * v..(i + 1) * v])
                        )
                    })
                    .collect();
                spec = format!(
                    "spec {} kl [{}] conf {:.1e}",
                    if d.ids == want {
                        "same".to_string()
                    } else {
                        format!("{:?} want {want:?}", d.ids)
                    },
                    kls.join(" "),
                    rel(&d.conf, &load_ref(&format!("step{s}_spec_conf")).unwrap())
                );
            }
            eprintln!("step {s}: [{}] KL {k:.1e} {spec}", rels.join(" "));
            assert!(k < 1e-6, "step {s}: KL {k}");
            for (i, r) in rels.iter().enumerate() {
                assert!(
                    r.parse::<f32>().unwrap() < 5e-2,
                    "step {s} layer {i}: rel {r}"
                );
            }
        }
    }

    /// The tiny fixture free-running with its own chunking: prefill, then one token at a time.
    /// Informational past top-1 (see `tiny_teacher_forced`). Skips without the fixture.
    #[test]
    fn tiny_free_running() {
        let dir = Path::new(&tiny()).join("hf");
        if !dir.join("model.safetensors").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return;
        }
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).unwrap(),
        )
        .unwrap();
        let mut m = Model::open(&dir).unwrap();
        let n_layers = m.cfg.n_layers;
        let mut top1 = 0;
        let mut kls = Vec::new();
        let steps = meta["steps"].as_array().unwrap();
        for (s, step) in steps.iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let mut got: HashMap<String, Vec<f32>> = HashMap::new();
            let (logits, main) = m
                .forward(&ids, sp, &mut |n, t| {
                    got.insert(n.to_string(), t.to_vec());
                })
                .unwrap();
            if let Some(h) = load_ref(&format!("step{s}_hash")) {
                assert_eq!(got["hash"], h, "step {s}: engram hashes");
            }
            let rels: Vec<String> = (0..n_layers)
                .map(|i| {
                    format!(
                        "{:.1e}",
                        rel(
                            &got[&format!("layer{i}")],
                            &load_ref(&format!("step{s}_layer{i}")).unwrap()
                        )
                    )
                })
                .collect();
            let r = load_ref(&format!("step{s}_logits")).unwrap();
            let k = kl(&r, &logits);
            kls.push(k);
            let next = step["next_id"].as_u64().unwrap() as usize;
            top1 += (argmax(&logits) == next) as usize;
            let mrel = rel(&main, &load_ref(&format!("step{s}_main_hidden")).unwrap());
            let draft = m.forward_spec(argmax(&logits), &main, sp).unwrap();
            let spec = match (&draft, step["spec_ids"].as_array()) {
                (Some(dr), Some(want)) => {
                    let want: Vec<usize> =
                        want.iter().map(|v| v.as_u64().unwrap() as usize).collect();
                    let sk = kl(
                        &load_ref(&format!("step{s}_spec_logits")).unwrap()[..m.cfg.vocab_size],
                        &dr.logits[..m.cfg.vocab_size],
                    );
                    format!("spec {:?} want {:?} kl0 {sk:.1e}", dr.ids, want)
                }
                _ => String::new(),
            };
            eprintln!(
                "step {s}: layers [{}] main {mrel:.1e} KL {k:.2e} top1 {} {spec}",
                rels.join(" "),
                argmax(&logits) == next
            );
        }
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        eprintln!("top1 {top1}/{} mean KL {mean:.2e}", steps.len());
        assert!(top1 + 1 >= steps.len(), "top-1 {top1}/{}", steps.len());
    }

    fn prompt() -> Option<(Model, Vec<usize>)> {
        let dir = Path::new(&tiny()).join("hf");
        if !dir.join("model.safetensors").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return None;
        }
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).unwrap(),
        )
        .unwrap();
        let ids = meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        Some((Model::open(&dir).unwrap(), ids))
    }

    /// A verify window then a rewind leaves the state exactly as feeding only the accepted
    /// tokens one at a time: the window's own logits and every later step's are bit-identical.
    #[test]
    fn tiny_chunk_and_rewind_exact() {
        let Some((mut a, ids)) = prompt() else { return };
        let v = a.cfg.vocab_size;
        let n = ids.len();
        a.forward(&ids, 0, &mut |_, _| {}).unwrap();
        // A: one at a time over a fixed token stream
        let stream: Vec<usize> = (0..14).map(|i| (37 * i + 11) % v).collect();
        let mut want = Vec::new();
        for (i, &t) in stream.iter().enumerate() {
            want.push(a.forward(&[t], n + i, &mut |_, _| {}).unwrap().0);
        }
        for keep in 1..=6 {
            let mut b = Model::open(&Path::new(&tiny()).join("hf")).unwrap();
            b.forward(&ids, 0, &mut |_, _| {}).unwrap();
            // a window of 6 at sp > 0 on top of a decode step
            b.forward(&stream[..1], n, &mut |_, _| {}).unwrap();
            let snap = b.snapshot();
            let (lw, _) = b.forward_all(&stream[1..7], n + 1).unwrap();
            for e in 0..6 {
                assert_eq!(lw[e * v..(e + 1) * v], want[1 + e][..], "window row {e}");
            }
            b.rewind(&snap, n + 1 + keep, n + 7);
            for i in 1 + keep..stream.len() {
                let (l, _) = b.forward(&[stream[i]], n + i, &mut |_, _| {}).unwrap();
                assert_eq!(l, want[i], "keep {keep} step {i}");
            }
        }
    }

    fn greedy(m: &mut Model, ids: &[usize], len: usize) -> Vec<usize> {
        let (mut l, _) = m.forward(ids, 0, &mut |_, _| {}).unwrap();
        let mut out = Vec::new();
        while out.len() < len {
            let t = argmax(&l);
            out.push(t);
            l = m
                .forward(&[t], ids.len() + out.len() - 1, &mut |_, _| {})
                .unwrap()
                .0;
        }
        out
    }

    /// Greedy speculative decoding: prefill, one plain step, then DSpark draft + verify.
    pub(crate) fn spec(
        m: &mut Model,
        ids: &[usize],
        len: usize,
        edit: &mut dyn FnMut(usize, &mut [usize]),
    ) -> (Vec<usize>, usize, usize) {
        let n = ids.len();
        let (l, main) = m.forward(ids, 0, &mut |_, _| {}).unwrap();
        let t0 = argmax(&l);
        m.forward_spec(t0, &main, 0).unwrap();
        let (l, mut main) = m.forward(&[t0], n, &mut |_, _| {}).unwrap();
        let mut out = vec![t0, argmax(&l)];
        let (mut sp, mut drafted, mut accepted) = (n, 0, 0);
        while out.len() < len {
            let s = m
                .decode_spec(*out.last().unwrap(), &main, sp, edit, &mut Greedy)
                .unwrap();
            drafted += s.drafted;
            accepted += s.accepted;
            out.extend(s.tokens);
            main = s.main;
            sp = s.sp;
        }
        out.truncate(len);
        (out, drafted, accepted)
    }

    #[test]
    fn tiny_spec_matches_greedy() {
        let Some((mut m, ids)) = prompt() else { return };
        let want = greedy(&mut m, &ids, 40);
        let v = m.cfg.vocab_size;
        let mut rng = 1u64;
        let n = ids.len();
        let (oracle, oracle2) = (want.clone(), want.clone());
        type Edit = Box<dyn FnMut(usize, &mut [usize])>;
        let runs: Vec<(&str, Edit)> = vec![
            ("dspark", Box::new(|_, _: &mut [usize]| {})),
            // the true continuation (as far as it is known): every draft accepted
            (
                "oracle",
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle.get(pos + i - n) {
                            *x = w;
                        }
                    }
                }),
            ),
            (
                "wrong",
                // the true continuation with random drafts broken: every acceptance length
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle2.get(pos + i - n) {
                            *x = w;
                        }
                        rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        if (rng >> 33).is_multiple_of(5) {
                            *x = (*x + 1 + (rng >> 40) as usize % (v - 1)) % v;
                        }
                    }
                }),
            ),
        ];
        for (name, mut edit) in runs {
            let mut m = Model::open(&Path::new(&tiny()).join("hf")).unwrap();
            let (got, drafted, accepted) = spec(&mut m, &ids, 40, &mut *edit);
            eprintln!(
                "{name}: accepted {accepted}/{drafted} ({:.1}%)",
                100.0 * accepted as f64 / drafted.max(1) as f64
            );
            assert_eq!(got, want, "{name}");
        }
    }

    /// Total variation between an empirical histogram (`n` draws) and exact probabilities,
    /// over the `top` most likely tokens plus everything else as one bucket; and the TV a
    /// sampler that is exactly right would show on average (sum of E|freq - p| / 2).
    fn tv(
        counts: &HashMap<usize, usize>,
        n: usize,
        exact: &[(usize, f64)],
        top: usize,
    ) -> (f64, f64) {
        let (mut tv, mut noise, mut rest_p) = (0.0, 0.0, 1.0);
        let mut rest_c = n;
        for &(t, p) in exact.iter().take(top) {
            let c = counts.get(&t).copied().unwrap_or(0);
            tv += (c as f64 / n as f64 - p).abs();
            noise += (2.0 * p * (1.0 - p) / (std::f64::consts::PI * n as f64)).sqrt();
            rest_p -= p;
            rest_c -= c;
        }
        let rest_p = rest_p.max(0.0);
        tv += (rest_c as f64 / n as f64 - rest_p).abs();
        noise += (2.0 * rest_p * (1.0 - rest_p) / (std::f64::consts::PI * n as f64)).sqrt();
        (tv / 2.0, noise / 2.0)
    }

    /// The rejection rule alone, on random logits: draft from Q, verify against P, and the
    /// kept token is distributed as P (temperature and top-p applied to both).
    #[test]
    fn spec_sampler_matches_target() {
        let v = 40;
        let mut r = 9u64;
        let mut logits = |scale: f32| -> Vec<f32> {
            (0..v)
                .map(|_| (uniform(&mut r) as f32 - 0.5) * scale)
                .collect()
        };
        for (temp, top_p) in [(1.0, 1.0), (0.7, 0.9), (1.3, 0.6)] {
            let (p, q) = (logits(6.0), logits(6.0));
            let mut s = SpecSampler::new(temp, top_p, 42);
            let exact = s.dist(&p);
            let n = 200_000;
            let mut counts = HashMap::new();
            let mut acc = 0;
            for _ in 0..n {
                let d = s.draft(&q);
                let (t, ok) = s.verify(&p, Some((&q, d)));
                acc += ok as usize;
                *counts.entry(t).or_insert(0) += 1;
            }
            let (tv, noise) = tv(&counts, n, &exact, 50);
            let (tvq, _) = tv_of(&s.dist(&q), &exact);
            eprintln!(
                "T {temp} top-p {top_p}: TV {tv:.4} (noise {noise:.4}), accept {:.3} = 1 - TV(P,Q) {:.3}",
                acc as f64 / n as f64,
                1.0 - tvq
            );
            assert!(tv < 0.01 && tv < 4.0 * noise, "TV {tv}");
            assert!((acc as f64 / n as f64 - (1.0 - tvq)).abs() < 0.01);
        }
    }

    fn tv_of(a: &[(usize, f64)], b: &[(usize, f64)]) -> (f64, ()) {
        let mut m: HashMap<usize, f64> = a.iter().copied().collect();
        let mut tv = 0.0;
        for &(i, p) in b {
            tv += (m.remove(&i).unwrap_or(0.0) - p).abs();
        }
        tv += m.values().sum::<f64>();
        (tv / 2.0, ())
    }

    /// The prompt, then one plain step: the state a speculative step starts from. Returns
    /// (model, the unfed token, its main hiddens, sp).
    fn spec_start() -> Option<(Model, usize, Vec<f32>, usize)> {
        let (mut m, ids) = prompt()?;
        let n = ids.len();
        let (l, main) = m.forward(&ids, 0, &mut |_, _| {}).unwrap();
        let t0 = argmax(&l);
        m.forward_spec(t0, &main, 0).unwrap();
        let (l, main) = m.forward(&[t0], n, &mut |_, _| {}).unwrap();
        Some((m, argmax(&l), main, n))
    }

    /// Speculative sampling on the tiny model: from one fixed state, 20000 spec steps (each
    /// rewound) against the exact target distributions. The first new token must follow P at
    /// that position, and the second (when the first was an accepted draft) P given the first.
    #[test]
    fn tiny_spec_sampling_distribution() {
        let Some((mut m, token, main, sp)) = spec_start() else {
            return;
        };
        let v = m.cfg.vocab_size;
        let w = main.len();
        let pos0 = sp + main.len() / w;
        let (temp, top_p) = (0.8, 0.95);
        let s = SpecSampler::new(temp, top_p, 0);
        // exact P at pos0 + 1 and, for its likely tokens, P at pos0 + 2
        let snap = m.snapshot();
        let (l0, _) = m.forward_all(&[token], pos0).unwrap();
        m.rewind(&snap, pos0, pos0 + 1);
        let p0 = s.dist(&l0);
        // draft distribution of the first draft, to show the test has power
        let dr = m.forward_spec(token, &main, sp).unwrap().unwrap();
        let q0 = s.dist(&dr.logits[..v]);
        let n = 20_000;
        let mut s = SpecSampler::new(temp, top_p, 7);
        let mut first: HashMap<usize, usize> = HashMap::new();
        let mut second: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        let (mut drafted, mut accepted) = (0, 0);
        for _ in 0..n {
            let outer = m.snapshot();
            let st = m
                .decode_spec(token, &main, sp, &mut |_, _| {}, &mut s)
                .unwrap();
            m.rewind(&outer, pos0, pos0 + st.drafted + 1);
            drafted += st.drafted;
            accepted += st.accepted;
            *first.entry(st.tokens[0]).or_insert(0) += 1;
            if st.tokens.len() > 1 {
                *second
                    .entry(st.tokens[0])
                    .or_default()
                    .entry(st.tokens[1])
                    .or_insert(0) += 1;
            }
        }
        // the state really was rewound: the next plain step matches a fresh one
        let (l0b, _) = m.forward_all(&[token], pos0).unwrap();
        assert_eq!(l0b, l0);
        let (tv0, noise0) = tv(&first, n, &p0, 50);
        let (tvq, _) = tv_of(&q0, &p0);
        eprintln!(
            "first token: TV {tv0:.4} (noise {noise0:.4}) over {} tokens; TV(P, Q) {tvq:.3}; accepted {accepted}/{drafted}",
            p0.len()
        );
        assert!(tv0 < 0.02, "first-token TV {tv0}");
        // the most common first token that was an accepted draft
        let (&t, cond) = second
            .iter()
            .max_by_key(|(_, c)| c.values().sum::<usize>())
            .expect("no draft was ever accepted");
        let cn: usize = cond.values().sum();
        let snap = m.snapshot();
        let (l1, _) = m.forward_all(&[token, t], pos0).unwrap();
        m.rewind(&snap, pos0, pos0 + 2);
        let p1 = SpecSampler::new(temp, top_p, 0).dist(&l1[v..]);
        let (tv1, noise1) = tv(cond, cn, &p1, 50);
        eprintln!("second token given {t}: TV {tv1:.4} (noise {noise1:.4}) from {cn} draws");
        assert!(tv1 < 4.0 * noise1 + 0.01, "second-token TV {tv1}");
    }

    /// After sampled spec steps the state is exactly that of feeding the same tokens one at a
    /// time: the next logits are bit-identical.
    #[test]
    fn tiny_spec_sampling_state_exact() {
        let Some((mut a, token, mut main, mut sp)) = spec_start() else {
            return;
        };
        let Some((mut b, ids)) = prompt() else { return };
        let n = ids.len();
        let mut s = SpecSampler::new(1.0, 0.9, 3);
        let mut out = vec![token];
        let (mut drafted, mut accepted) = (0, 0);
        while out.len() < 30 {
            let st = a
                .decode_spec(*out.last().unwrap(), &main, sp, &mut |_, _| {}, &mut s)
                .unwrap();
            drafted += st.drafted;
            accepted += st.accepted;
            out.extend(st.tokens);
            main = st.main;
            sp = st.sp;
        }
        eprintln!(
            "sampled {} tokens, accepted {accepted}/{drafted}",
            out.len()
        );
        // b: the prompt, the first plain token, then `out` one at a time
        let (l, _) = b.forward(&ids, 0, &mut |_, _| {}).unwrap();
        let mut fed = vec![argmax(&l)];
        fed.extend(&out);
        let mut want = Vec::new();
        for (i, &t) in fed.iter().enumerate() {
            want = b.forward(&[t], n + i, &mut |_, _| {}).unwrap().0;
        }
        // a has fed all but the last token of `out`
        let (got, _) = a
            .forward(&[*out.last().unwrap()], n + fed.len() - 1, &mut |_, _| {})
            .unwrap();
        assert_eq!(got, want);
        // and one more step stays equal
        let t = argmax(&want);
        let x = a.forward(&[t], n + fed.len(), &mut |_, _| {}).unwrap().0;
        let y = b.forward(&[t], n + fed.len(), &mut |_, _| {}).unwrap().0;
        assert_eq!(x, y);
    }

    fn run(
        spec: bool,
        temp: f32,
        max_new: usize,
        eos: &[usize],
        stop_after: usize,
    ) -> Option<(Vec<usize>, Finish, Usage)> {
        let (mut m, ids) = prompt()?;
        let mut out = Vec::new();
        let mut usage = Usage::default();
        let mut pick = SpecSampler::new(temp, 0.95, 11);
        let f = generate(
            &mut m,
            &ids,
            max_new,
            &mut pick,
            spec,
            eos,
            &mut usage,
            &mut |t| {
                out.push(t);
                (out.len() == stop_after).then_some(Finish::Stop)
            },
        )
        .unwrap();
        Some((out, f, usage))
    }

    /// The serve loop: spec == plain at temperature 0, and EOS, a stop and `max_new` cut a
    /// verify window exactly where plain decoding stops.
    #[test]
    fn tiny_generate_spec_limits() {
        let Some((plain, f, u0)) = run(false, 0.0, 40, &[], 0) else {
            return;
        };
        assert_eq!(f, Finish::Length);
        let (spec, _, u1) = run(true, 0.0, 40, &[], 0).unwrap();
        assert_eq!(spec, plain);
        assert_eq!(u1.completion_tokens, 40);
        eprintln!(
            "plain {} forwards, spec {} forwards ({}/{} accepted)",
            u0.decode_steps, u1.decode_steps, u1.accepted_tokens, u1.draft_tokens
        );
        assert_eq!(u0.decode_steps, 39);
        assert!(u1.decode_steps <= u0.decode_steps);
        for max in [1, 2, 7, 13] {
            let (o, f, u) = run(true, 0.0, max, &[], 0).unwrap();
            assert_eq!((o.as_slice(), f), (&plain[..max], Finish::Length));
            assert_eq!(u.completion_tokens, max);
        }
        for j in [3, 10, 25] {
            let e = plain[j];
            let first = plain.iter().position(|&t| t == e).unwrap();
            let (o, f, _) = run(true, 0.0, 40, &[e], 0).unwrap();
            assert_eq!((o.as_slice(), f), (&plain[..first], Finish::Stop));
        }
        let (o, f, _) = run(true, 0.0, 40, &[], 9).unwrap();
        assert_eq!((o.as_slice(), f), (&plain[..9], Finish::Stop));
        // sampled, seeded: reproducible, and spec runs at any temperature
        let (a, _, ua) = run(true, 0.8, 30, &[], 0).unwrap();
        let (b, _, _) = run(true, 0.8, 30, &[], 0).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 30);
        assert!(ua.draft_tokens > 0);
        eprintln!(
            "sampled: {} tokens in {} forwards ({}/{} accepted)",
            a.len(),
            ua.decode_steps,
            ua.accepted_tokens,
            ua.draft_tokens
        );
    }

    /// Acceptance on the real checkpoint (`DS41_REAL=/workspace/ds41`): greedy spec over a code
    /// prompt for 200 tokens; prints accepted/drafted and tokens per verify. CPU f32, slow:
    /// `cargo test --release --lib real_acceptance -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_acceptance() {
        let Ok(dir) = std::env::var("DS41_REAL") else {
            eprintln!("skip: set DS41_REAL to the checkpoint dir");
            return;
        };
        let dir = Path::new(&dir);
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let msgs = serde_json::json!([{"role": "user", "content": "Write a Rust function that \
            parses a CSV line into fields, handling quoted fields with escaped quotes, plus \
            unit tests."}]);
        let prompt = crate::chat::Template::load(dir)
            .unwrap()
            .render(&msgs, None, Some(false))
            .unwrap();
        let ids: Vec<usize> = tok
            .encode(prompt.as_str(), false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&i| i as usize)
            .collect();
        let n = std::env::var("DS41_REAL_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let mut m = Model::open(dir).unwrap();
        let mut usage = Usage::default();
        let mut out = Vec::new();
        let mut pick = SpecSampler::new(0.0, 1.0, 0);
        let t = std::time::Instant::now();
        generate(
            &mut m,
            &ids,
            n,
            &mut pick,
            true,
            &[],
            &mut usage,
            &mut |t| {
                out.push(t as u32);
                None
            },
        )
        .unwrap();
        eprintln!("{}", tok.decode(&out, false).unwrap());
        let verifies = usage.decode_steps.max(1);
        eprintln!(
            "{} tokens in {} forwards ({:.2} tokens/forward), accepted {}/{} drafts ({:.1}%), {:.1} s",
            usage.completion_tokens,
            verifies,
            usage.completion_tokens as f64 / verifies as f64,
            usage.accepted_tokens,
            usage.draft_tokens,
            100.0 * usage.accepted_tokens as f64 / usage.draft_tokens.max(1) as f64,
            t.elapsed().as_secs_f64()
        );
    }
}
