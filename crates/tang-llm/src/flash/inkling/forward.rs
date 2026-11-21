//! The Inkling 2 Mini CPU f32 reference forward, straight from the GGUF.
//!
//! A transcription of llama.cpp's `src/models/inkling2.cpp` (`graph::build_layer`) and
//! `inkling2-loop.cpp` (`graph::graph`), one token batch at a time, with every weight
//! dequantised on use: the 17.6 GB IQ4_NL file is only ever touched through the mmap, and the
//! routed experts are dequantised one at a time for the tokens that picked them.
//!
//! Per token the model is:
//!
//! ```text
//! x = rms_norm(embed(t))
//! for il in 0..6:                    // the prelude
//!     x = layer(il, x, no message)
//! prelude = x; prelude_blocks = blocks
//! for pass in 0..3:                  // the looped core, 12 layers × 3
//!     if pass > 0:
//!         bank = this pass's blocks past the first; bank_out = its output
//!         x = prelude; blocks = prelude_blocks
//!     for ip in 6..18:
//!         x = layer(ip, x, pass > 0 ? mix(bank + [bank_out], msg_query) : none)
//! for il in 18..24:
//!     x = layer(il, x, no message)
//! logits = output @ rms_norm(mix(blocks + [x], output_res_query))
//! ```
//!
//! and one layer, with `blocks` the residual streams started every sixth layer:
//!
//! ```text
//! inpSA = x
//! [q,k,v] = mix(blocks + [x], attn_res_query)        // 1 column local, 3 global
//! [q,k,v] = control(q,k,v, messages)                 // loop passes only
//! [q,k,v] = the layer's input norms
//! Q = clip(Wq q); K = conv(Wk k); V = rms(conv(Wv v)); R = clip(Wr q)
//! scores = Q·K/√d + bias(R, distance) - sink(q)      // causal (+ window if local)
//! x = (W o (attn · σ(Wg q))) convolved
//! x = if ip % 6 == 0 { blocks += [inpSA]; x } else { inpSA + x }
//! ffn_inp = x
//! x = control(mix(blocks + [x], ffn_res_query), messages[3])
//! x = conv(moe(rms_norm(x, ffn_norm)))
//! x = ffn_inp + x
//! ```

use super::{t, Hparams};
use crate::flash::reference::{sigmoid, Act, Mat};
use crate::gguf::{Gguf, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The open model: mmap'd GGUF + geometry.
pub struct InklingRef {
    pub g: Gguf,
    pub hp: Hparams,
    /// Print per-layer timings to stderr.
    pub verbose: bool,
    /// When set, every output position's full f32 logits are appended here.
    pub logits: Option<std::sync::Mutex<Vec<f32>>>,
}

impl InklingRef {
    pub fn open(path: &Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        let hp = Hparams::from_gguf(&g)?;
        hp.check()?;
        super::check_tensors(&g, &hp)?;
        Ok(Self {
            g,
            hp,
            verbose: false,
            logits: None,
        })
    }

    /// Logits for the last position, or for every position when `all`.
    pub fn forward(&self, ids: &[u32], all: bool) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "no tokens");
        for &id in ids {
            ensure!(
                (id as usize) < self.hp.n_vocab,
                "token {id} outside the {}-entry vocabulary",
                self.hp.n_vocab
            );
        }
        let mut run = Run::new(self, ids.len());
        let n = ids.len();
        let e = self.hp.n_embd;

        // Embedding: each row read on its own (the table is 201024 × 2304) and normalized.
        let mut x = vec![0f32; n * e];
        let te = self.g.info("token_embd.weight")?;
        let mut row = vec![0f32; e];
        for (i, &id) in ids.iter().enumerate() {
            self.g.read_row(te, id as usize, &mut row)?;
            x[i * e..(i + 1) * e].copy_from_slice(&row);
        }
        let tok_norm = self.g.tensor_f32("token_embd_norm.weight")?;
        rms_mul(&mut x, &tok_norm, self.hp.eps, e);

        // The prelude, the looped core, the tail.
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let hp = &self.hp;
        for il in 0..hp.loop_start {
            x = run.layer(il, il, x, &mut blocks, None)?;
        }
        let prelude = x.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..hp.n_passes {
            let (bank, bank_out) = if pass == 0 {
                (Vec::new(), Vec::new())
            } else {
                (
                    blocks[prelude_blocks.len() + 1..].to_vec(),
                    std::mem::take(&mut x),
                )
            };
            if pass > 0 {
                x = prelude.clone();
                blocks = prelude_blocks.clone();
            }
            for ip in hp.loop_start..hp.loop_end {
                let il = if pass == 0 {
                    ip
                } else {
                    hp.n_layer + (pass - 1) * (hp.loop_end - hp.loop_start) + (ip - hp.loop_start)
                };
                let msgs = if pass == 0 {
                    None
                } else {
                    Some(run.messages(&bank, &bank_out, ip)?)
                };
                x = run.layer(il, ip, x, &mut blocks, msgs.as_deref())?;
            }
        }
        for il in hp.loop_end..hp.n_layer {
            x = run.layer(il, il, x, &mut blocks, None)?;
        }

        // Output: mix the residual streams, normalize, project, soft-cap, drop the padding.
        let oq = self.g.tensor_f32("output_res_query.weight")?;
        let mut y = run.mix(&blocks, &x, &oq, 1);
        let out_norm = self.g.tensor_f32("output_norm.weight")?;
        rms_mul(&mut y, &out_norm, hp.eps, e);
        let out = dense(&self.g, "output.weight")?;
        ensure!(
            out.rows >= hp.n_vocab_unpadded && out.cols == e,
            "output.weight is {}×{}, want {}×{e}",
            out.rows,
            out.cols,
            hp.n_vocab_unpadded
        );
        let n_out = if all { n } else { 1 };
        let first = if all { 0 } else { n - 1 };
        let mut logits = Vec::with_capacity(n_out * hp.n_vocab_unpadded);
        for ti in first..first + n_out {
            let mut lg = out.apply(&y[ti * e..(ti + 1) * e], 1, false);
            lg.truncate(hp.n_vocab_unpadded);
            if hp.final_softcap != 0.0 {
                let cap = hp.final_softcap;
                for v in lg.iter_mut() {
                    *v = cap * (*v / cap).tanh();
                }
            }
            if let Some(l) = &self.logits {
                l.lock().unwrap().extend_from_slice(&lg);
            }
            logits.extend_from_slice(&lg);
        }
        Ok(logits)
    }
}

/// A `y = W x` matrix dequantised to f32; `rows` outputs of `cols` inputs.
fn dense(g: &Gguf, name: &str) -> Result<Mat> {
    let ti = g.info(name)?;
    ensure!(
        ti.dims.len() == 2,
        "{name}: not a matrix ({:?})",
        ti.dims
    );
    Ok(Mat {
        rows: ti.dims[1] as usize,
        cols: ti.dims[0] as usize,
        data: g.dequantize(ti)?,
        act: Act::F32,
    })
}

/// Expert `e` of a fused `[in, out, n_expert]` tensor, as a `Mat` of `out` rows.
fn expert_mat(g: &Gguf, ti: &TensorInfo, e: usize) -> Result<Mat> {
    Ok(Mat {
        rows: ti.dims[1] as usize,
        cols: ti.dims[0] as usize,
        data: g.expert(ti, e)?,
        act: Act::F32,
    })
}

/// `x = x / rms(x) * w`, per `e`-wide row.
fn rms_mul(x: &mut [f32], w: &[f32], eps: f32, e: usize) {
    for row in x.chunks_mut(e) {
        let s: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let r = 1.0 / ((s / e as f64) as f32 + eps).sqrt();
        for (v, g) in row.iter_mut().zip(w) {
            *v = *v * r * g;
        }
    }
}

/// `x / rms(x)` per `e`-wide row, no weight (the `mix` and value norms).
fn rms_only(x: &mut [f32], eps: f32, e: usize) {
    for row in x.chunks_mut(e) {
        let s: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let r = 1.0 / ((s / e as f64) as f32 + eps).sqrt();
        for v in row.iter_mut() {
            *v *= r;
        }
    }
}

/// In place: scale each head's `dh`-wide row by `scale` (if any), then scale the whole row down
/// to an L2 norm of at most `thr` (`llama_model_inkling2::graph::clip`).
fn clip_heads(x: &mut [f32], n_head: usize, dh: usize, scale: Option<&[f32]>, thr: f32) {
    for head in x.chunks_mut(n_head * dh) {
        for h in 0..n_head {
            let s = &mut head[h * dh..(h + 1) * dh];
            if let Some(sc) = scale {
                for (v, g) in s.iter_mut().zip(&sc[h * dh..(h + 1) * dh]) {
                    *v *= g;
                }
            }
            let n: f64 = s.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let norm = (n.sqrt()) as f32 + 1e-6;
            let f = (thr / norm).clamp(0.0, 1.0);
            for v in s.iter_mut() {
                *v *= f;
            }
        }
    }
}

/// The SITU gate: `β·tanh(g/β)·σ(g)`.
#[inline]
fn situ(g: f32, beta: f32) -> f32 {
    beta * (g / beta).tanh() * sigmoid(g)
}

/// The SITU linear branch: `λ·tanh(u/λ)`, or `u` when `λ` is 0.
#[inline]
fn situ_lin(u: f32, lambda: f32) -> f32 {
    if lambda > 0.0 {
        lambda * (u / lambda).tanh()
    } else {
        u
    }
}

/// Indices of the `k` largest, ties to the lowest index.
fn topk(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// One cache slot's state: its own K, V and convolution history (the looped core's passes each
/// get one, even though they share the weights).
struct Slot {
    /// `[n][kv_dim]`, one row per token position.
    k: Vec<f32>,
    v: Vec<f32>,
    /// `[n_conv-1][n_chan]` with the taps of each channel contiguous, as llama.cpp's `history`.
    hist: Vec<f32>,
    n: usize,
}

impl Slot {
    fn new(hp: &Hparams, n_chan: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            hist: vec![0f32; (hp.n_conv - 1) * n_chan],
            n: 0,
        }
    }
}

/// One forward pass over a whole prompt.
struct Run<'a> {
    m: &'a InklingRef,
    /// Tokens in the prompt.
    n: usize,
    e: usize,
    kv_dim: usize,
    slots: Vec<Slot>,
}

/// The four loop control gates, in the order the messages are indexed.
const CTRL: [&str; 4] = ["q", "k", "v", "ffn"];

impl<'a> Run<'a> {
    fn new(m: &'a InklingRef, n: usize) -> Self {
        let hp = &m.hp;
        let kv_dim = hp.d_head * hp.n_head_kv;
        // `hparams.n_sconv_channels`: K, V, the attention output and the FFN output.
        let n_chan = 2 * kv_dim + 2 * hp.n_embd;
        let slots = (0..hp.n_slot()).map(|_| Slot::new(hp, n_chan)).collect();
        Self {
            m,
            n,
            e: hp.n_embd,
            kv_dim,
            slots,
        }
    }

    fn g(&self) -> &Gguf {
        &self.m.g
    }

    /// `mix`: attention over the banked residual streams and `cur`, with `nq` learned queries.
    /// Returns `[n][nq][e]`; the streams are normalized, the queries are not.
    fn mix(&self, blocks: &[Vec<f32>], cur: &[f32], query: &[f32], nq: usize) -> Vec<f32> {
        let (n, e) = (self.n, self.e);
        let mut streams: Vec<Vec<f32>> = blocks.to_vec();
        streams.push(cur.to_vec());
        for s in streams.iter_mut() {
            rms_only(s, 1e-6, e);
        }
        let ns = streams.len();
        let scale = 1.0 / (e as f32).sqrt();
        let mut out = vec![0f32; n * nq * e];
        out.par_chunks_mut(nq * e).enumerate().for_each(|(ti, o)| {
            let mut scores = vec![0f32; ns * nq];
            for (s, stream) in streams.iter().enumerate() {
                let row = &stream[ti * e..(ti + 1) * e];
                for q in 0..nq {
                    let qv = &query[q * e..(q + 1) * e];
                    scores[s * nq + q] = crate::flash::reference::dot(row, qv) * scale;
                }
            }
            for q in 0..nq {
                let mut m = f32::NEG_INFINITY;
                for s in 0..ns {
                    m = m.max(scores[s * nq + q]);
                }
                let mut den = 0f32;
                for s in 0..ns {
                    scores[s * nq + q] = (scores[s * nq + q] - m).exp();
                    den += scores[s * nq + q];
                }
                let den = if den > 0.0 { den } else { 1.0 };
                for s in 0..ns {
                    let p = scores[s * nq + q] / den;
                    let src = &streams[s][ti * e..(ti + 1) * e];
                    for d in 0..e {
                        o[q * e + d] += p * src[d];
                    }
                }
            }
        });
        out
    }

    /// The four loop messages for one layer: `mix` over the previous pass's bank and output.
    /// Returns `[4][n][e]`, in [`CTRL`] order.
    fn messages(&self, bank: &[Vec<f32>], bank_out: &[f32], ip: usize) -> Result<Vec<Vec<f32>>> {
        let q = self.g().tensor_f32(&t(ip, "loop_msg_query.weight"))?;
        ensure!(q.len() == self.e * CTRL.len(), "loop_msg_query shape");
        let mixed = self.mix(bank, bank_out, &q, CTRL.len());
        let mut out = Vec::with_capacity(CTRL.len());
        for c in 0..CTRL.len() {
            let mut v = vec![0f32; self.n * self.e];
            for ti in 0..self.n {
                let src = (ti * CTRL.len() + c) * self.e;
                v[ti * self.e..(ti + 1) * self.e].copy_from_slice(&mixed[src..src + self.e]);
            }
            out.push(v);
        }
        Ok(out)
    }

    /// One control gate: `cur - tanh(forget·g)·cur + tanh(gain·g)·message`, with
    /// `g = down · [rms(cur); rms(message)]`.
    fn control(&self, cur: &[f32], msg: &[f32], ip: usize, c: usize) -> Result<Vec<f32>> {
        let (n, e) = (self.n, self.e);
        let mut g = vec![0f32; n * 2 * e];
        for ti in 0..n {
            let a = &cur[ti * e..(ti + 1) * e];
            let b = &msg[ti * e..(ti + 1) * e];
            let sa: f64 = a.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let sb: f64 = b.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let ra = 1.0 / ((sa / e as f64) as f32 + 1e-6).sqrt();
            let rb = 1.0 / ((sb / e as f64) as f32 + 1e-6).sqrt();
            for i in 0..e {
                g[ti * 2 * e + i] = a[i] * ra;
                g[ti * 2 * e + e + i] = b[i] * rb;
            }
        }
        let name = CTRL[c];
        let down = dense(self.g(), &t(ip, &format!("loop_{name}_ctrl_down.weight")))?;
        let gain = dense(self.g(), &t(ip, &format!("loop_{name}_ctrl_gain.weight")))?;
        let forget = dense(self.g(), &t(ip, &format!("loop_{name}_ctrl_forget.weight")))?;
        let h = down.apply(&g, n, false);
        let ga = gain.apply(&h, n, false);
        let fo = forget.apply(&h, n, false);
        let mut out = vec![0f32; n * e];
        for i in 0..n * e {
            out[i] = cur[i] - fo[i].tanh() * cur[i] + ga[i].tanh() * msg[i];
        }
        Ok(out)
    }

    /// A causal short convolution with a residual, at channel offset `off` of slot `il`'s
    /// history. `taps`, when given, are the input-dependent taps of the global layers
    /// (`[n][n_conv * n_heads]`, one `n_conv`-tap vector per head).
    fn conv(
        &mut self,
        il: usize,
        x: &[f32],
        w: &[f32],
        taps: Option<&[f32]>,
        n_heads: usize,
        off: usize,
    ) -> Vec<f32> {
        let (n, nc) = (self.n, x.len() / self.n);
        let k = self.m.hp.n_conv - 1;
        let n_conv = k + 1;
        let hist = self.slots[il].hist[off * k..off * k + nc * k].to_vec();
        // Row r of the padded input: r < k is history, r >= k is `cur[r - k]`.
        let row = |r: usize, c: usize| -> f32 {
            if r < k {
                hist[c * k + r]
            } else {
                x[(r - k) * nc + c]
            }
        };
        let mut out = vec![0f32; n * nc];
        out.par_chunks_mut(nc).enumerate().for_each(|(ti, o)| {
            for (c, ov) in o.iter_mut().enumerate() {
                let mut s = 0f32;
                for kk in 0..n_conv {
                    s += row(ti + kk, c) * w[c * n_conv + kk];
                }
                *ov = s;
            }
            if let Some(taps) = taps {
                let nd = nc / n_heads;
                for lag in 0..n_conv {
                    for c in 0..nc {
                        let tap = taps[ti * n_conv * n_heads + (c / nd) * n_conv + lag];
                        o[c] += row(ti + k - lag, c) * tap;
                    }
                }
            }
        });
        // Store this token batch as the next history.
        let h = &mut self.slots[il].hist[off * k..off * k + nc * k];
        for c in 0..nc {
            for kk in 0..k {
                h[c * k + kk] = row(n + kk, c);
            }
        }
        let scale = 1.0 / (((if taps.is_some() { 2 } else { 1 }) * n_conv + 1) as f32).sqrt();
        for (o, &v) in out.iter_mut().zip(x) {
            *o = (v + *o) * scale;
        }
        out
    }

    /// The routed MoE with SITU experts: 128 experts, top-8, sigmoid gating, weights normalized
    /// and scaled by `expert_weights_scale`.
    fn moe(&self, ip: usize, x: &[f32]) -> Result<Vec<f32>> {
        let hp = &self.m.hp;
        let (n, e) = (self.n, self.e);
        let ne = hp.n_expert;
        let n_ff = hp.n_ff_exp;
        let router = dense(self.g(), &t(ip, "ffn_gate_inp.weight"))?;
        ensure!(router.rows == ne, "ffn_gate_inp: {} experts", router.rows);
        let logits = router.apply(x, n, false);
        let bias = self.g().tensor_f32(&t(ip, "ffn_exp_probs_b.bias")).ok();
        if let Some(b) = &bias {
            ensure!(b.len() == ne, "ffn_exp_probs_b: {} entries", b.len());
        }

        // Per token: sigmoid probs, top-k on probs + bias, weights from the unbiased probs.
        let mut picked: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        for ti in 0..n {
            let l = &logits[ti * ne..(ti + 1) * ne];
            let p: Vec<f32> = l.iter().map(|&v| sigmoid(v)).collect();
            let sel: Vec<f32> = (0..ne)
                .map(|i| p[i] + bias.as_ref().map_or(0.0, |b| b[i]))
                .collect();
            let top = topk(&sel, hp.n_expert_used);
            let mut w: Vec<f32> = top.iter().map(|&i| p[i]).collect();
            let sum: f32 = w.iter().sum::<f32>() + 1e-20;
            for wi in w.iter_mut() {
                *wi = *wi / sum * hp.expert_weights_scale;
            }
            picked.push(top.into_iter().zip(w).collect());
        }

        let gu_t = self.g().info(&t(ip, "ffn_gate_up_exps.weight"))?;
        let dn_t = self.g().info(&t(ip, "ffn_down_exps.weight"))?;
        let mut out = vec![0f32; n * e];
        for ex in 0..ne {
            let rows: Vec<(usize, f32)> = (0..n)
                .filter_map(|ti| {
                    picked[ti]
                        .iter()
                        .find(|(i, _)| *i == ex)
                        .map(|&(_, w)| (ti, w))
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            let mut xe = Vec::with_capacity(rows.len() * e);
            for &(ti, _) in &rows {
                xe.extend_from_slice(&x[ti * e..(ti + 1) * e]);
            }
            let gu = expert_mat(self.g(), gu_t, ex)?;
            ensure!(gu.rows == 2 * n_ff, "gate_up expert {ex}: {} rows", gu.rows);
            let g = gu.apply(&xe, rows.len(), false);
            let mut act = vec![0f32; rows.len() * n_ff];
            for r in 0..rows.len() {
                for i in 0..n_ff {
                    act[r * n_ff + i] = situ(g[r * 2 * n_ff + i], hp.situ_beta)
                        * situ_lin(g[r * 2 * n_ff + n_ff + i], hp.situ_linear_beta);
                }
            }
            let dn = expert_mat(self.g(), dn_t, ex)?;
            ensure!(dn.rows == e, "down expert {ex}: {} rows", dn.rows);
            let y = dn.apply(&act, rows.len(), false);
            for (r, &(ti, w)) in rows.iter().enumerate() {
                for i in 0..e {
                    out[ti * e + i] += w * y[r * e + i];
                }
            }
        }
        Ok(out)
    }

    /// One layer application: `il` is the cache slot (which owns the K/V and conv state), `ip`
    /// the physical layer whose weights it reads.
    fn layer(
        &mut self,
        il: usize,
        ip: usize,
        cur: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        msgs: Option<&[Vec<f32>]>,
    ) -> Result<Vec<f32>> {
        let hp = &self.m.hp;
        let (n, e) = (self.n, self.e);
        let n_head = hp.n_head;
        let n_kv = hp.n_head_kv;
        let dh = hp.d_head;
        let kv_dim = self.kv_dim;
        let is_swa = hp.is_swa[ip];
        let g = &self.m.g;

        let inp_sa = cur.clone();

        // Attention inputs: one mixed stream, or three for the global layers.
        let mut inputs = [cur.clone(), cur.clone(), cur.clone()];
        if ip > 0 {
            let q = g.tensor_f32(&t(ip, "attn_res_query.weight"))?;
            let nq = q.len() / e;
            ensure!(nq == if is_swa { 1 } else { 3 }, "attn_res_query: {nq} columns");
            let mixed = self.mix(blocks, &cur, &q, nq);
            for (j, inp) in inputs.iter_mut().enumerate() {
                let col = if is_swa { 0 } else { j };
                let mut v = vec![0f32; n * e];
                for ti in 0..n {
                    let src = (ti * nq + col) * e;
                    v[ti * e..(ti + 1) * e].copy_from_slice(&mixed[src..src + e]);
                }
                *inp = v;
            }
        }
        if let Some(m) = msgs {
            for j in 0..3 {
                inputs[j] = self.control(&inputs[j], &m[j], ip, j)?;
            }
        }
        if is_swa {
            let norm = g.tensor_f32(&t(ip, "attn_norm.weight"))?;
            if msgs.is_some() {
                for inp in inputs.iter_mut() {
                    rms_mul(inp, &norm, hp.eps, e);
                }
            } else {
                rms_mul(&mut inputs[0], &norm, hp.eps, e);
                let (a, b) = inputs.split_at_mut(1);
                b[0].copy_from_slice(&a[0]);
                b[1].copy_from_slice(&a[0]);
            }
        } else {
            for (j, name) in ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"]
                .iter()
                .enumerate()
            {
                let norm = g.tensor_f32(&t(ip, &format!("{name}.weight")))?;
                rms_mul(&mut inputs[j], &norm, hp.eps, e);
            }
        }

        // Q, K, V and the relative-position projection.
        let wq = dense(g, &t(ip, "attn_q.weight"))?;
        let mut q = wq.apply(&inputs[0], n, false);
        let wk = dense(g, &t(ip, "attn_k.weight"))?;
        let k0 = wk.apply(&inputs[1], n, false);
        let wv = dense(g, &t(ip, "attn_v.weight"))?;
        let v0 = wv.apply(&inputs[2], n, false);
        let wr = dense(g, &t(ip, "attn_rel.weight"))?;
        let mut r = wr.apply(&inputs[0], n, false);

        let k_conv = g.tensor_f32(&t(ip, "attn_k_conv.weight"))?;
        let v_conv = g.tensor_f32(&t(ip, "attn_v_conv.weight"))?;
        let (dyn_k, dyn_v) = if is_swa {
            (None, None)
        } else {
            let pk = dense(g, &t(ip, "attn_k_conv_proj.weight"))?;
            let pv = dense(g, &t(ip, "attn_v_conv_proj.weight"))?;
            (
                Some(pk.apply(&inputs[1], n, false)),
                Some(pv.apply(&inputs[2], n, false)),
            )
        };
        let k = self.conv(il, &k0, &k_conv, dyn_k.as_deref(), n_kv, 0);
        let v = self.conv(il, &v0, &v_conv, dyn_v.as_deref(), n_kv, kv_dim);

        // Per-head scaling and clipping, the value norm, and the relative-position clip.
        let q_clip = g.tensor_f32(&t(ip, "attn_q_clip.weight"))?;
        clip_heads(&mut q, n_head, dh, Some(&q_clip), hp.qk_clip);
        let mut k = k;
        clip_heads(&mut k, n_kv, dh, None, hp.qk_clip);
        let mut v = v;
        rms_only(&mut v, hp.eps, dh);
        let r_clip = g.tensor_f32(&t(ip, "attn_rel_clip.weight"))?;
        clip_heads(&mut r, n_head, hp.n_rel, Some(&r_clip), hp.rel_clip);

        // The log-scaled positions of the full-attention layers.
        if !is_swa {
            let alpha = hp.log_scale_alpha;
            let floor = hp.log_scale_floor as f32;
            let mut tau = vec![0f32; n];
            for (ti, tv) in tau.iter_mut().enumerate() {
                let t = 1.0 + alpha * (((ti + 1) as f32) / floor).max(1.0).ln();
                *tv = t;
            }
            for ti in 0..n {
                let t = tau[ti];
                for x in q[ti * n_head * dh..(ti + 1) * n_head * dh].iter_mut() {
                    *x *= t;
                }
                for x in r[ti * n_head * hp.n_rel..(ti + 1) * n_head * hp.n_rel].iter_mut() {
                    *x *= t;
                }
            }
        }

        // The sink logit per head, and the relative bias table projected through R.
        let sink = dense(g, &t(ip, "attn_sink_proj.weight"))?;
        let sinks = sink.apply(&inputs[0], n, false);
        let rel_bias = g.tensor_f32(&t(ip, "attn_rel_bias.weight"))?;
        ensure!(
            rel_bias.len() == hp.n_rel * hp.n_rel_extent,
            "attn_rel_bias: {} entries",
            rel_bias.len()
        );

        // Append this batch to the slot's cache, then attend.
        {
            let slot = &mut self.slots[il];
            slot.k.extend_from_slice(&k);
            slot.v.extend_from_slice(&v);
            slot.n += n;
        }
        let cache_k = self.slots[il].k.clone();
        let cache_v = self.slots[il].v.clone();
        let total = self.slots[il].n;
        let first = total - n;

        let scale = 1.0 / (dh as f32).sqrt();
        let window = if is_swa { hp.n_swa } else { usize::MAX };
        // A local layer's bias table is only `n_swa` wide; distances past the extent take the
        // zero padding column llama.cpp appends to `attn_rel_bias` (`ggml_pad(..., 0, 1, ...)`).
        let extent = if is_swa {
            hp.n_swa.min(hp.n_rel_extent)
        } else {
            hp.n_rel_extent
        };
        let n_rel = hp.n_rel;
        let mut attn = vec![0f32; n * n_head * dh];
        attn.par_chunks_mut(n_head * dh).enumerate().for_each(|(ti, o)| {
            let pos = first + ti;
            let lo = (pos + 1).saturating_sub(window);
            let kv_lo = lo;
            let kv_hi = pos + 1;
            let n_kv_cells = kv_hi - kv_lo;
            let mut p = vec![0f32; n_kv_cells];
            for h in 0..n_head {
                let qv = &q[(ti * n_head + h) * dh..(ti * n_head + h + 1) * dh];
                let kvh = h % n_kv;
                // Scores, then the sink-stabilized softmax.
                let mut mx = 0f32;
                for (c, pv) in p.iter_mut().enumerate() {
                    let j = kv_lo + c;
                    let kv = &cache_k[j * kv_dim + kvh * dh..j * kv_dim + (kvh + 1) * dh];
                    let dot = crate::flash::reference::dot(qv, kv);
                    let d = (pos - j).min(extent);
                    let mut bias = 0f32;
                    let rv = &r[(ti * n_head + h) * n_rel..(ti * n_head + h + 1) * n_rel];
                    // `attn_rel_bias` is `[relative_length, extent]` in the file (dims 16 × 1024),
                    // but dequantized row-major it is `extent` rows of `n_rel`, so the entry for
                    // (rel, d) is `d * n_rel + rel`. `d == extent` is llama.cpp's zero column.
                    if d < extent {
                        for (rr, &b) in rv.iter().enumerate() {
                            bias += b * rel_bias[d * n_rel + rr];
                        }
                    }
                    let s = dot * scale + bias - sinks[ti * n_head + h];
                    *pv = s;
                    if s > mx {
                        mx = s;
                    }
                }
                let mx = mx.max(0.0);
                let mut den = (-mx).exp();
                for pv in p.iter_mut() {
                    *pv = (*pv - mx).exp();
                    den += *pv;
                }
                for (c, &pc) in p.iter().enumerate() {
                    let j = kv_lo + c;
                    let w = pc / den;
                    let vv = &cache_v[j * kv_dim + kvh * dh..j * kv_dim + (kvh + 1) * dh];
                    for d in 0..dh {
                        o[h * dh + d] += w * vv[d];
                    }
                }
            }
        });

        // The output gate, projection and convolution.
        let gate = dense(g, &t(ip, "attn_gate.weight"))?;
        let gv = gate.apply(&inputs[0], n, false);
        for (a, b) in attn.iter_mut().zip(&gv) {
            *a *= sigmoid(*b);
        }
        let wo = dense(g, &t(ip, "attn_output.weight"))?;
        let o = wo.apply(&attn, n, false);
        let post = g.tensor_f32(&t(ip, "attn_post_conv.weight"))?;
        let attn_out = self.conv(il, &o, &post, None, 0, 2 * kv_dim);

        // A block-start layer opens a new residual stream; the rest join the banked ones.
        let mut x = if ip.is_multiple_of(hp.attn_res_block) {
            blocks.push(inp_sa.clone());
            attn_out
        } else {
            inp_sa.iter().zip(&attn_out).map(|(a, b)| a + b).collect()
        };

        // The FFN: mix, the loop's FFN message, the norm, the experts, the convolution.
        let ffn_inp = x.clone();
        let fq = g.tensor_f32(&t(ip, "ffn_res_query.weight"))?;
        let mut f = self.mix(blocks, &x, &fq, 1);
        if let Some(m) = msgs {
            f = self.control(&f, &m[3], ip, 3)?;
        }
        let ffn_norm = g.tensor_f32(&t(ip, "ffn_norm.weight"))?;
        rms_mul(&mut f, &ffn_norm, hp.eps, e);
        let y = self.moe(ip, &f)?;
        let fconv = g.tensor_f32(&t(ip, "ffn_post_conv.weight"))?;
        let y = self.conv(il, &y, &fconv, None, 0, 2 * kv_dim + e);
        x = ffn_inp.iter().zip(&y).map(|(a, b)| a + b).collect();
        Ok(x)
    }
}

/// The top-k next-token log-probs of one position's logits (f64, ties to the lower id).
pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    crate::flash::reference::top_logprobs(logits, k)
}

/// `tang-llm inkling-ref <gguf> <ids...> [--ids-file F] [--top K] [--last N] [--all] [-v]
/// [--logits FILE]`: run the reference forward and print one JSON object per output position,
/// `{"pos": i, "top": [[id, logprob], ...]}`.
pub fn cli(args: &[String]) -> Result<()> {
    let usage = "usage: inkling-ref <gguf> <ids...> [--ids-file F] [--top K] [--last N] \
                 [--all] [-v] [--logits FILE]";
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context(usage)?);
    let mut ids: Vec<u32> = Vec::new();
    let mut top = 10;
    let mut all = false;
    let mut verbose = false;
    let mut logits_out: Option<PathBuf> = None;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--top" => top = it.next().context("--top K")?.parse().context("--top K")?,
            "--all" => all = true,
            "--ids-file" => {
                let f = it.next().context("--ids-file F")?;
                let s = std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?;
                for w in s.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']') {
                    if !w.is_empty() {
                        ids.push(
                            w.parse()
                                .with_context(|| format!("{f}: bad token id {w:?}"))?,
                        );
                    }
                }
            }
            "-v" | "--verbose" => verbose = true,
            "--logits" => logits_out = Some(PathBuf::from(it.next().context("--logits FILE")?)),
            s if s.starts_with("--") => bail!("unknown argument {s}; {usage}"),
            s => ids.push(
                s.parse()
                    .with_context(|| format!("bad token id {s:?}; {usage}"))?,
            ),
        }
    }
    if ids.is_empty() {
        bail!("no token ids; {usage}");
    }
    let mut m = InklingRef::open(&path)?;
    m.verbose = verbose;
    m.logits = logits_out.as_ref().map(|_| Default::default());
    let lg = m.forward(&ids, all)?;
    let n_vocab = m.hp.n_vocab_unpadded;
    let first = if all { 0 } else { ids.len() - 1 };
    let mut out = String::new();
    for (i, row) in lg.chunks(n_vocab).enumerate() {
        let t = top_logprobs(row, top);
        let top_s: Vec<String> = t.iter().map(|(id, lp)| format!("[{id},{lp:.6}]")).collect();
        writeln!(out, "{{\"pos\":{},\"top\":[{}]}}", first + i, top_s.join(","))?;
    }
    let mut stdout = std::io::stdout().lock();
    write!(stdout, "{out}")?;
    if let (Some(p), Some(l)) = (&logits_out, &m.logits) {
        let l = l.lock().unwrap();
        std::fs::write(
            p,
            l.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .with_context(|| format!("writing {}", p.display()))?;
    }
    Ok(())
}
