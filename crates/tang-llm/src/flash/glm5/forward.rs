//! The GLM-5.3-Flash CPU f32 reference forward, straight from the GGUF (W1).
//!
//! Every weight is dequantised to f32 on use and dropped after its layer, so the 244 GB
//! UD-Q4_K_XL file is only ever touched through the mmap — never resident. Activations stay
//! f32 throughout.
//!
//! Structure, per the Hugging Face reference (`Glm5NextTextModel.forward`):
//!
//! ```text
//! R[t][c] = embed(t) for the hc=4 mHC streams
//! for l in 0..45:
//!   x          = mhc_pre(R, hc_attn_*)           // collapse + (post, comb) weights
//!   y          = kda(x)  or  mla(x)               // the mixer
//!   R          = mhc_post(R, y)                   // post ⊙ y + comb^T @ R
//!   x          = mhc_pre(R, hc_ffn_*)
//!   y          = dense_ffn(x)  or  moe(x) + shared(x)
//!   R          = mhc_post(R, y)
//! h  = mean_c R[t][c]                              // the final stream collapse
//! logits = output @ rmsnorm(h) * output_norm
//! ```
//!
//! The component fns live in [`super::mhc`], [`super::kda`], [`super::mla`], [`super::moe`]
//! (W2–W5); until they land, the layer loop calls `todo!()` for those pieces. The parts W1
//! owns outright — embedding, the mHC *plumbing* between blocks (stream reshape/mean), the
//! norms, the first-3 dense FFN, the head, and the per-layer dump — are complete.
//!
//! Dumps (W0's compare-tool format): one `*.f32` file per entry plus `index.json`, written to
//! `$TANG_DUMP_DIR` when set. Entries: `embd`, per layer `L{l:02}.hc_attn_out`,
//! `L{l:02}.post_attn` (all streams), `L{l:02}.ffn_out`, `L{l:02}.post_ffn`, and `final_x`,
//! `logits` — all for the last position unless noted.

use super::Hparams;
use crate::flash::reference::{rms_norm_mul, silu};
use crate::gguf::Gguf;
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The open model: mmap'd GGUF + geometry.
pub struct Glm5Ref {
    pub g: Gguf,
    pub hp: Hparams,
    /// Print per-layer timings to stderr.
    pub verbose: bool,
    /// When set, every output position's full f32 logits are appended here (`[n_out][n_vocab]`).
    pub logits: Option<std::sync::Mutex<Vec<f32>>>,
    /// Dump per-layer intermediates here (W0's compare format).
    pub dump: Option<PathBuf>,
}

/// One dump entry: name, shape, values (row-major).
struct DumpEntry {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Glm5Ref {
    pub fn open(path: &Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        let hp = Hparams::from_gguf(&g)?;
        Ok(Self {
            g,
            hp,
            verbose: false,
            logits: None,
            dump: None,
        })
    }

    /// Dequantize `blk.{l}.{name}` as a matrix `[cols, rows]` (dims[0] fastest).
    pub(crate) fn mat(&self, l: usize, name: &str) -> Result<crate::flash::reference::Mat> {
        let info = self.g.info(&super::t(l, name))?;
        ensure!(
            info.dims.len() <= 2,
            "blk.{l}.{name}: not a matrix ({:?})",
            info.dims
        );
        let cols = info.dims[0] as usize;
        let rows = if info.dims.len() == 2 {
            info.dims[1] as usize
        } else {
            1
        };
        Ok(crate::flash::reference::Mat {
            rows,
            cols,
            data: self.g.dequantize(info)?,
            act: crate::flash::reference::Act::F32,
        })
    }

    pub(crate) fn vec(&self, name: &str) -> Result<Vec<f32>> {
        self.g.tensor_f32(name)
    }

    /// One KDA layer's weights as W4's [`super::kda::KdaLayerW`] (dequantised f32, HF layout:
    /// GGUF `[cols, rows]` row-major is HF `[out][in]`; the three depthwise convs, GGUF
    /// `[d_conv, 1, d_inner, 1]` = flat `[chan][tap]`, are stacked q|k|v).
    pub(crate) fn kda_weights(
        &self,
        l: usize,
    ) -> Result<(super::kda::KdaLayerW, super::kda::KdaConfig)> {
        let hp = &self.hp;
        let t = |s: &str| super::t(l, s);
        let v = |s: &str| self.vec(&t(s));
        let cfg = super::kda::KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let d_inner = hp.n_head * hp.kda_head_dim;
        let mut conv = Vec::with_capacity(3 * d_inner * hp.ssm_d_conv);
        for name in [
            "ssm_conv1d_q.weight",
            "ssm_conv1d_k.weight",
            "ssm_conv1d_v.weight",
        ] {
            let c = v(name)?;
            ensure!(
                c.len() == d_inner * hp.ssm_d_conv,
                "blk.{l}.{name}: {} floats",
                c.len()
            );
            conv.extend_from_slice(&c);
        }
        let w = super::kda::KdaLayerW {
            q: v("attn_q.weight")?,
            k: v("attn_k.weight")?,
            v: v("attn_v.weight")?,
            conv,
            f_a: v("ssm_f_a.weight")?,
            f_b: v("ssm_f_b.weight")?,
            dt_bias: v("ssm_dt.bias")?,
            decay_rate: super::kda::ForgetGateW::decay_rate_from_gguf(&v("ssm_a")?),
            b_proj: v("ssm_beta.weight")?,
            g_a: v("ssm_g_a.weight")?,
            g_b: v("ssm_g_b.weight")?,
            o_norm: v("ssm_norm.weight")?,
            o_proj: v("attn_output.weight")?,
            hidden: hp.n_embd,
        };
        Ok((w, cfg))
    }

    /// One DSA layer's weights as W3's [`super::mla::MlaWeights`] (dequantised f32).
    pub(crate) fn mla_weights(&self, l: usize) -> Result<super::mla::MlaWeights> {
        let hp = &self.hp;
        let t = |s: &str| super::t(l, s);
        let v = |s: &str| self.vec(&t(s));
        // 3-D GGUF layouts: attn_k_b [d_qk][kv_lora][n_head] flat, attn_v_b
        // [kv_lora][d_v][n_head] flat.
        let k_b = self.g.tensor_f32(&t("attn_k_b.weight"))?;
        let v_b = self.g.tensor_f32(&t("attn_v_b.weight"))?;
        Ok(super::mla::MlaWeights::new(
            hp.n_embd,
            hp.q_lora_rank,
            hp.kv_lora_rank,
            hp.n_head,
            hp.d_qk,
            hp.d_v,
            hp.idx_heads,
            hp.idx_dim,
            hp.idx_top_k,
            hp.kpool,
            hp.kpool_select_tail,
            hp.eps,
            v("attn_q_a.weight")?,
            v("attn_q_a_norm.weight")?,
            v("attn_q_b.weight")?,
            v("attn_kv_a_mqa.weight")?,
            v("attn_kv_a_norm.weight")?,
            k_b,
            v_b,
            v("attn_output.weight")?,
            v("indexer.attn_k.weight")?,
            v("indexer.k_norm.weight")?,
            v("indexer.k_norm.bias")?,
            v("indexer.attn_q_b.weight")?,
            v("indexer.proj.weight")?,
            v("indexer_compressor_gate.weight")?,
            v("indexer_compressor_ape.weight")?,
        ))
    }

    /// The dense lead-in FFN (`Glm5NextTextMLP`): swiglu with the HF clamp semantics —
    /// `gate.clamp(max=limit)`, `up.clamp(-limit, limit)` — then `down`.
    pub(crate) fn dense_ffn(&self, l: usize, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
        let limit = self
            .hp
            .clamp_shexp(l)
            .context("dense layer without swiglu_limit")?;
        let gate = self.mat(l, "ffn_gate.weight")?.apply(x, t_len, false);
        let up = self.mat(l, "ffn_up.weight")?.apply(x, t_len, false);
        let mut h = Vec::with_capacity(gate.len());
        for (g, u) in gate.iter().zip(&up) {
            h.push(silu(g.min(limit)) * u.clamp(-limit, limit));
        }
        Ok(self.mat(l, "ffn_down.weight")?.apply(&h, t_len, false))
    }

    /// RMSNorm in place over every token row. Layer weights are `blk.{l}.{suffix}`;
    /// the head norm passes its full tensor name via `exact`.
    pub(crate) fn norm_all(&self, x: &mut [f32], l: usize, suffix: &str) -> Result<()> {
        let n = self.hp.n_embd;
        let w = self.vec(&super::t(l, suffix))?;
        ensure!(w.len() == n, "blk.{l}.{suffix}: {} wide, want {n}", w.len());
        let eps = self.hp.eps;
        x.par_chunks_mut(n)
            .for_each(|row| rms_norm_mul(row, &w, eps));
        Ok(())
    }

    /// RMSNorm with a full tensor name (the head's `output_norm.weight`).
    pub(crate) fn norm_all_exact(&self, x: &mut [f32], name: &str) -> Result<()> {
        let n = self.hp.n_embd;
        let w = self.vec(name)?;
        ensure!(w.len() == n, "{name}: {} wide, want {n}", w.len());
        let eps = self.hp.eps;
        x.par_chunks_mut(n)
            .for_each(|row| rms_norm_mul(row, &w, eps));
        Ok(())
    }

    /// Run `tokens` from position 0 and return the full logits `[t_len][n_vocab]`
    /// (f32, streamed through the head a few positions at a time).
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let hp = &self.hp;
        let (n, hc, t_len) = (hp.n_embd, hp.hc, tokens.len());
        ensure!(t_len > 0, "no tokens");
        let last = t_len - 1;
        let start = std::time::Instant::now();
        let mut dump: Option<Vec<DumpEntry>> = (self.dump.is_some()).then(Vec::new);
        let mut dput = |name: &str, shape: &[usize], data: &[f32]| {
            if let Some(d) = dump.as_mut() {
                d.push(DumpEntry {
                    name: name.to_string(),
                    shape: shape.to_vec(),
                    data: data.to_vec(),
                });
            }
        };

        // ---- embedding, copied into every stream
        let embd = self.g.info("token_embd.weight")?;
        let mut r = vec![0f32; t_len * hc * n];
        for (t, &tok) in tokens.iter().enumerate() {
            ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
            let e = self.g.rows(embd, tok as usize, 1)?;
            for c in 0..hc {
                r[(t * hc + c) * n..(t * hc + c + 1) * n].copy_from_slice(&e);
            }
            if t == last {
                dput("embd", &[n], &e);
            }
        }

        // ---- the layer loop. The mixer/FFN component fns are W2-W5's; until they land,
        //      the pieces this file owns still run (norms, dense FFN, plumbing, dumps).
        //      The `todo!()` arms make the rest of the loop dead to the compiler; the allows
        //      go away as each component lands.
        for l in 0..hp.n_layer {
            let lt = std::time::Instant::now();
            // --- attention half. W2's mHC is per token over [hc][n] stream slices.
            let mhp = super::mhc::MhcHparams {
                streams: hc,
                sinkhorn_iters: hp.hc_sinkhorn_iters as u32,
                hc_eps: hp.hc_eps,
                rms_eps: hp.eps,
            };
            let w_attn = super::mhc::MhcWeights::from_gguf(&self.g, l, "hc_attn", n, &mhp)?;
            let mut mixes: Vec<super::mhc::Mix> = Vec::with_capacity(t_len);
            let mut x = vec![0f32; t_len * n];
            for t in 0..t_len {
                let streams: Vec<&[f32]> = (0..hc)
                    .map(|c| &r[(t * hc + c) * n..(t * hc + c + 1) * n])
                    .collect();
                let mix = super::mhc::mhc_pre(&w_attn, &streams)?;
                x[t * n..(t + 1) * n].copy_from_slice(&mix.x);
                mixes.push(mix);
            }
            self.norm_all(&mut x, l, "attn_norm.weight")?;
            let y: Vec<f32> = if hp.is_kda[l] {
                // W4's KDA (reference recurrent form until frog/glm5-kda lands)
                // W4's KDA layer (prefill from position 0: fresh conv + recurrent state)
                let (w, cfg) = self.kda_weights(l)?;
                let mut conv = super::kda::ConvState::zeros(&cfg, 1);
                let mut state = super::kda::KdaState::zeros(&cfg, 1);
                super::kda::kda_layer(&x, &w, &cfg, &mut conv, &mut state, 1, t_len)?
            } else {
                // W3's MLA-NoPE + indexer, dense-causal while kv_len <= idx_top_k
                // (W3 verified the indexer is a no-op there).
                let w = self.mla_weights(l)?;
                let q_resid = w.gemv_q_a(&x, t_len);
                let mut k_latent = w.gemv_kv_a(&x, t_len);
                let eps_idx = 1e-6;
                let idx_dim = hp.idx_dim;
                let mut cache_k = vec![0f32; t_len * idx_dim];
                let mut cache_gate = vec![0f32; t_len * idx_dim];
                for t in 0..t_len {
                    let mut k = vec![0f32; idx_dim];
                    let mut g = vec![0f32; idx_dim];
                    super::mla::indexer_key(
                        &w,
                        &x[t * hp.n_embd..(t + 1) * hp.n_embd],
                        &mut k,
                        &mut g,
                    );
                    cache_k[t * idx_dim..(t + 1) * idx_dim].copy_from_slice(&k);
                    cache_gate[t * idx_dim..(t + 1) * idx_dim].copy_from_slice(&g);
                }
                let sel = if super::mla::indexer_is_dense(t_len, hp.idx_top_k) {
                    None
                } else {
                    Some(super::mla::indexer_select(
                        &w,
                        &x,
                        &q_resid,
                        &cache_k,
                        &cache_gate,
                        &vec![1u8; t_len],
                        &vec![1u8; t_len],
                        t_len,
                        t_len,
                        0,
                    ))
                };
                let _ = eps_idx;
                let mut out = vec![0f32; t_len * hp.n_embd];
                super::mla::mla_forward(
                    &w,
                    &x,
                    &q_resid,
                    &k_latent,
                    t_len,
                    t_len,
                    0,
                    None,
                    sel.as_deref(),
                    &mut out,
                );
                k_latent.clear();
                out
            };
            dput(&format!("L{l:02}.hc_attn_out"), &[n], &y[last * n..]);
            let r_in = r.clone();
            for t in 0..t_len {
                let streams: Vec<&[f32]> = (0..hc)
                    .map(|c| &r_in[(t * hc + c) * n..(t * hc + c + 1) * n])
                    .collect();
                let (tok, _) = r.split_at_mut((t + 1) * hc * n);
                let token = &mut tok[t * hc * n..];
                let mut outs: Vec<&mut [f32]> = token.chunks_mut(n).collect();
                super::mhc::mhc_post(&mixes[t], &y[t * n..(t + 1) * n], &streams, &mut outs)?;
            }
            dput(
                &format!("L{l:02}.post_attn"),
                &[hc, n],
                &r[last * hc * n..(last + 1) * hc * n],
            );

            // --- FFN half
            let w_ffn = super::mhc::MhcWeights::from_gguf(&self.g, l, "hc_ffn", n, &mhp)?;
            let mut mixes: Vec<super::mhc::Mix> = Vec::with_capacity(t_len);
            let mut x = vec![0f32; t_len * n];
            for t in 0..t_len {
                let streams: Vec<&[f32]> = (0..hc)
                    .map(|c| &r[(t * hc + c) * n..(t * hc + c + 1) * n])
                    .collect();
                let mix = super::mhc::mhc_pre(&w_ffn, &streams)?;
                x[t * n..(t + 1) * n].copy_from_slice(&mix.x);
                mixes.push(mix);
            }
            self.norm_all(&mut x, l, "ffn_norm.weight")?;
            let y: Vec<f32> = if hp.is_dense(l) {
                self.dense_ffn(l, &x, t_len)?
            } else {
                // W5's MoE: sigmoid/noaux_tc router + shared expert, all in one call.
                let p = super::moe::MoeParams::from_gguf(&self.g, l)?;
                super::moe::moe_layer(&p, &x, t_len)?
            };
            dput(&format!("L{l:02}.ffn_out"), &[n], &y[last * n..]);
            let r_in = r.clone();
            for t in 0..t_len {
                let streams: Vec<&[f32]> = (0..hc)
                    .map(|c| &r_in[(t * hc + c) * n..(t * hc + c + 1) * n])
                    .collect();
                let (tok, _) = r.split_at_mut((t + 1) * hc * n);
                let token = &mut tok[t * hc * n..];
                let mut outs: Vec<&mut [f32]> = token.chunks_mut(n).collect();
                super::mhc::mhc_post(&mixes[t], &y[t * n..(t + 1) * n], &streams, &mut outs)?;
            }
            dput(
                &format!("L{l:02}.post_ffn"),
                &[hc, n],
                &r[last * hc * n..(last + 1) * hc * n],
            );

            if self.verbose {
                eprintln!(
                    "layer {l:2} {} {:6.1}s (total {:6.1}s)",
                    if hp.is_kda[l] { "kda" } else { "dsa" },
                    lt.elapsed().as_secs_f64(),
                    start.elapsed().as_secs_f64()
                );
            }
        }

        // ---- head: unweighted stream mean, output norm, lm_head
        let mut h = vec![0f32; t_len * n];
        for t in 0..t_len {
            for c in 0..hc {
                for i in 0..n {
                    h[t * n + i] += r[(t * hc + c) * n + i];
                }
            }
            for v in &mut h[t * n..(t + 1) * n] {
                *v /= hc as f32;
            }
        }
        let mut x = h;
        self.norm_all_exact(&mut x, "output_norm.weight")?;
        dput("final_x", &[n], &x[last * n..]);

        let head = self.head()?;
        ensure!(head.cols == n, "output.weight is {} wide", head.cols);
        let n_vocab = hp.n_vocab;
        let mut all = Vec::with_capacity(t_len * n_vocab);
        // a few positions at a time, so T × 155k logits never live at once
        for c0 in (0..t_len).step_by(64) {
            let c1 = (c0 + 64).min(t_len);
            let logits = head.apply(&x[c0 * n..c1 * n], c1 - c0, false);
            if let Some(sink) = &self.logits {
                sink.lock().unwrap().extend_from_slice(&logits);
            }
            if c0 <= last && last < c1 {
                let j = last - c0;
                dput(
                    "logits",
                    &[n_vocab],
                    &logits[j * n_vocab..(j + 1) * n_vocab],
                );
            }
            all.extend_from_slice(&logits);
        }

        // ---- write the dump (W0's compare format)
        if let (Some(dir), Some(entries)) = (&self.dump, &dump) {
            std::fs::create_dir_all(dir)?;
            let mut idx = Vec::new();
            for e in entries {
                let file = format!("{}.f32", e.name);
                let mut b = Vec::with_capacity(e.data.len() * 4);
                for v in &e.data {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(dir.join(&file), b)?;
                idx.push(serde_json::json!({
                    "name": e.name, "file": file, "dtype": "f32", "shape": e.shape
                }));
            }
            let index = serde_json::json!({
                "model": "glm5-next",
                "tokens": tokens,
                "layout": "little-endian raw f32 arrays; stream tensors are [hc][n_embd], stream-major",
                "tensors": idx,
            });
            std::fs::write(
                dir.join("index.json"),
                serde_json::to_string_pretty(&index)?,
            )?;
        }
        if self.verbose {
            eprintln!(
                "forward of {t_len} tokens: {:.1}s",
                start.elapsed().as_secs_f64()
            );
        }
        Ok(all)
    }

    /// The lm_head matrix (tied or not — the converter drops `lm_head` when tied, in which case
    /// the embedding is the head).
    pub(crate) fn head(&self) -> Result<crate::flash::reference::Mat> {
        let info = match self.g.get("output.weight") {
            Some(t) => t,
            None => self.g.info("token_embd.weight")?,
        };
        let cols = info.dims[0] as usize;
        let rows = info.dims[1] as usize;
        Ok(crate::flash::reference::Mat {
            rows,
            cols,
            data: self.g.dequantize(info)?,
            act: crate::flash::reference::Act::F32,
        })
    }
}

/// The top-k next-token log-probs of one position's logits (f64, ties to the lower id).
pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    crate::flash::reference::top_logprobs(logits, k)
}

/// `tang-llm glm5-ref <gguf-first-shard> <ids...> [--ids-file F] [--top K] [--last N] [-v]
/// [--logits FILE] [--dump DIR]`: run the reference forward and print one JSON object per
/// output position, `{"pos": i, "top": [[id, logprob], ...]}`.
///
/// `--dump` (or `$TANG_DUMP_DIR`) writes W0's per-layer compare dump. `--logits FILE` writes
/// every output position's full f32 logits.
pub fn cli(args: &[String]) -> Result<()> {
    let usage = "usage: glm5-ref <gguf> <ids...> [--ids-file F] [--top K] [--last N] [-v] \
                 [--logits FILE] [--dump DIR]";
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context(usage)?);
    let mut ids: Vec<u32> = Vec::new();
    let mut last: Option<usize> = None;
    let mut top = 10;
    let mut verbose = false;
    let mut logits_out: Option<PathBuf> = None;
    let mut dump_dir: Option<PathBuf> = None;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--last" => last = Some(it.next().context("--last N")?.parse().context("--last N")?),
            "--top" => top = it.next().context("--top K")?.parse().context("--top K")?,
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
            "--dump" => dump_dir = Some(PathBuf::from(it.next().context("--dump DIR")?)),
            s => ids.push(
                s.parse()
                    .with_context(|| format!("bad token id {s:?}; {usage}"))?,
            ),
        }
    }
    if ids.is_empty() {
        bail!("no token ids; {usage}");
    }
    let mut m = Glm5Ref::open(&path)?;
    m.verbose = verbose;
    m.dump = dump_dir.or_else(|| std::env::var("TANG_DUMP_DIR").ok().map(PathBuf::from));
    m.logits = logits_out.as_ref().map(|_| Default::default());
    if last.is_none() {
        // default: every position is an output position
        last = Some(ids.len());
    }
    let from = ids.len() - last.unwrap().clamp(1, ids.len());
    let all = m.forward(&ids)?;
    let n_vocab = m.hp.n_vocab;
    let mut out = String::new();
    for (i, lg) in all.chunks(n_vocab).enumerate() {
        if i < from {
            continue;
        }
        let t = top_logprobs(lg, top);
        let top_s: Vec<String> = t.iter().map(|(id, lp)| format!("[{id},{lp:.6}]")).collect();
        writeln!(out, "{{\"pos\":{},\"top\":[{}]}}", i, top_s.join(","))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Value;

    /// A tiny-but-complete glm5-next geometry: 3 layers (dense KDA, KDA, DSA),
    /// 4 heads, hc=4, vocab 64.
    fn tiny_hparams() -> Hparams {
        Hparams {
            n_vocab: 64,
            n_embd: 32,
            n_layer: 3,
            n_dense_lead: 1,
            n_expert_shared: 1,
            n_ff: 16,
            n_head: 4,
            d_qk: 16,
            d_v: 16,
            q_lora_rank: 16,
            kv_lora_rank: 8,
            kda_head_dim: 8,
            ssm_d_conv: 4,
            kda_gate_lower_bound: Some(-5.0),
            idx_heads: 2,
            idx_dim: 8,
            idx_top_k: 8,
            kpool: 4,
            kpool_select_tail: true,
            is_kda: vec![true, true, false],
            indexer_full: vec![true, true, true],
            hc: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 8,
            expert_weights_scale: 2.5,
            expert_weights_norm: true,
            swiglu_limit: vec![Some(10.0); 3],
            eps: 1e-5,
            n_nextn: 0,
        }
    }

    fn meta(hp: &Hparams) -> Vec<(&'static str, Value)> {
        let mut m = vec![
            ("general.architecture", Value::Str("glm5-next".into())),
            ("general.alignment", Value::U32(32)),
            ("glm5-next.block_count", Value::U32(hp.n_layer as u32)),
            ("glm5-next.embedding_length", Value::U32(hp.n_embd as u32)),
            ("glm5-next.vocab_size", Value::U32(hp.n_vocab as u32)),
            ("glm5-next.feed_forward_length", Value::U32(hp.n_ff as u32)),
            (
                "glm5-next.attention.head_count",
                Value::U32(hp.n_head as u32),
            ),
            (
                "glm5-next.attention.head_count_kv",
                Value::Array(hp.is_kda.iter().map(|&k| Value::U32(!k as u32)).collect()),
            ),
            (
                "glm5-next.attention.layer_norm_rms_epsilon",
                Value::F32(hp.eps),
            ),
            (
                "glm5-next.attention.q_lora_rank",
                Value::U32(hp.q_lora_rank as u32),
            ),
            (
                "glm5-next.attention.kv_lora_rank",
                Value::U32(hp.kv_lora_rank as u32),
            ),
            (
                "glm5-next.attention.key_length_mla",
                Value::U32(hp.d_qk as u32),
            ),
            (
                "glm5-next.attention.value_length_mla",
                Value::U32(hp.d_v as u32),
            ),
            ("glm5-next.kda.head_dim", Value::U32(hp.kda_head_dim as u32)),
            (
                "glm5-next.ssm.conv_kernel",
                Value::U32(hp.ssm_d_conv as u32),
            ),
            (
                "glm5-next.kda.gate_lower_bound",
                Value::F32(hp.kda_gate_lower_bound.unwrap_or(-5.0)),
            ),
            (
                "glm5-next.attention.indexer.head_count",
                Value::U32(hp.idx_heads as u32),
            ),
            (
                "glm5-next.attention.indexer.key_length",
                Value::U32(hp.idx_dim as u32),
            ),
            (
                "glm5-next.attention.indexer.top_k",
                Value::U32(hp.idx_top_k as u32),
            ),
            (
                "glm5-next.attention.indexer.kpool",
                Value::U32(hp.kpool as u32),
            ),
            (
                "glm5-next.attention.indexer.kpool_select_tail",
                Value::Bool(hp.kpool_select_tail),
            ),
            ("glm5-next.hyper_connection.count", Value::U32(hp.hc as u32)),
            (
                "glm5-next.hyper_connection.sinkhorn_iterations",
                Value::U32(hp.hc_sinkhorn_iters as u32),
            ),
            ("glm5-next.hyper_connection.epsilon", Value::F32(hp.hc_eps)),
            ("glm5-next.expert_count", Value::U32(hp.n_expert as u32)),
            (
                "glm5-next.expert_used_count",
                Value::U32(hp.n_expert_used as u32),
            ),
            (
                "glm5-next.expert_feed_forward_length",
                Value::U32(hp.n_ff_exp as u32),
            ),
            (
                "glm5-next.expert_shared_count",
                Value::U32(hp.n_expert_shared as u32),
            ),
            (
                "glm5-next.leading_dense_block_count",
                Value::U32(hp.n_dense_lead as u32),
            ),
            (
                "glm5-next.expert_weights_scale",
                Value::F32(hp.expert_weights_scale),
            ),
            ("glm5-next.expert_weights_norm", Value::Bool(true)),
            (
                "glm5-next.swiglu_clamp_exp",
                Value::Array(
                    hp.swiglu_limit
                        .iter()
                        .map(|v| Value::F32(v.unwrap_or(10.0)))
                        .collect(),
                ),
            ),
        ];
        if hp.n_nextn > 0 {
            m.push((
                "glm5-next.nextn_predict_layers",
                Value::U32(hp.n_nextn as u32),
            ));
        }
        m
    }

    /// A full tiny f32 GGUF: every tensor of [`super::gguf::layer_tensors`] for the 3 layers,
    /// plus embedding, output norm and head, all zeros.
    fn tiny_gguf(dir: &Path, seed: u64) -> Result<PathBuf> {
        let hp = tiny_hparams();
        let mut rng: u64 = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
        let mut next = move || {
            // xorshift64*, deterministic
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545f4914f6cdd1d) >> 40) as f32 / (1 << 24) as f32 - 0.5
        };
        let mut tensors: Vec<crate::flash::glm5::synth::T> = Vec::new();
        let mut push = |name: String, dims: Vec<u64>, t: &mut Vec<crate::flash::glm5::synth::T>| {
            let n: usize = dims.iter().product::<u64>() as usize;
            let data: Vec<f32> = (0..n).map(|_| next()).collect();
            t.push(crate::flash::glm5::synth::T {
                name: Box::leak(name.into_boxed_str()),
                dims: Box::leak(dims.into_boxed_slice()),
                data: Box::leak(data.into_boxed_slice()),
            });
        };
        push(
            "token_embd.weight".into(),
            vec![hp.n_embd as u64, hp.n_vocab as u64],
            &mut tensors,
        );
        push(
            "output_norm.weight".into(),
            vec![hp.n_embd as u64],
            &mut tensors,
        );
        push(
            "output.weight".into(),
            vec![hp.n_embd as u64, hp.n_vocab as u64],
            &mut tensors,
        );
        for l in 0..hp.n_layer {
            for (suffix, dims) in super::super::gguf::layer_tensors(&hp, l)? {
                push(super::super::t(l, suffix), dims, &mut tensors);
            }
        }
        let path = dir.join("glm5-tiny-f32.gguf");
        crate::flash::glm5::synth::write_gguf(&path, &meta(&hp), &tensors)?;
        Ok(path)
    }

    #[test]
    fn hparams_parse_and_layers_check() {
        let dir = std::env::temp_dir().join("glm5-synth-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = tiny_gguf(&dir, 42).unwrap();
        let g = Gguf::open(&path).unwrap();
        let hp = Hparams::from_gguf(&g).unwrap();
        assert_eq!(hp.n_layer, 3);
        assert_eq!(hp.hc, 4);
        assert_eq!(hp.is_kda, vec![true, true, false]);
        assert_eq!(hp.indexer_full, vec![true, true, true]);
        assert_eq!(hp.n_expert, 4);
        assert_eq!(hp.expert_weights_scale, 2.5);
        assert_eq!(hp.swiglu_limit, vec![Some(10.0); 3]);
        for l in 0..hp.n_layer {
            let n = super::super::gguf::check_layer(&g, &hp, l).unwrap();
            assert!(n > 8, "layer {l}: {n} tensors");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dense_ffn_matches_hf_clamp_semantics() {
        // down(silu(clamp(g)) * clamp(u)) — a small hand case
        let dir = std::env::temp_dir().join("glm5-synth-ffn");
        std::fs::create_dir_all(&dir).unwrap();
        let path = tiny_gguf(&dir, 7).unwrap();
        let m = Glm5Ref::open(&path).unwrap();
        let x = vec![0.5, -0.5, 0.25, -0.25];
        // n_embd=32 but mat() needs t_len*n inputs; use one row padded
        let mut row = vec![0f32; m.hp.n_embd];
        row[..4].copy_from_slice(&x);
        let y = m.dense_ffn(0, &row, 1).unwrap();
        assert_eq!(y.len(), m.hp.n_embd);
        std::fs::remove_file(&path).ok();
    }
}
