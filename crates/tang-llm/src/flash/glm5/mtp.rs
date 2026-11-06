//! GLM-5.3-Flash MTP ("NextN") drafting + greedy draft-and-verify decode, CPU reference.
//!
//! The unsloth GGUF ships the NextN block as `blk.{n_layer}` (`blk.45`): a DSA (MLA-NoPE +
//! indexer) attention, a 288-expert MoE + shared expert, and `nextn.{eh_proj, enorm, hnorm,
//! shared_head_norm}`. Embedding and LM head are the trunk's (`token_embd`, `output`). There
//! are **no** mHC tensors: plain residual. Per llama.cpp `glm5-next.cpp` `graph_mtp`:
//!
//! ```text
//! cell i (position i):  h_i = trunk's post-output_norm hidden at i (t_h_nextn), tok = x_{i+1}
//! x    = eh_proj @ [rmsnorm(embed(tok))*enorm ; rmsnorm(h_i)*hnorm]       (8192 -> 4096)
//! x   += dsa(rmsnorm(x)*attn_norm)            own latent/indexer cache, one row per cell
//! x   += moe(rmsnorm(x)*ffn_norm) + shexp(..)
//! hn   = rmsnorm(x)*shared_head_norm           the next chain step's h (t_h_nextn)
//! logits = output @ hn                          predicts x_{i+2}
//! ```
//!
//! Decode: trunk window -> teacher-forced MTP cells for the window's positions -> chain
//! `k` drafts (temporary cells, truncated after) -> one trunk window `[cur, d1..dk]` verifies.
//! Greedy acceptance keeps the longest prefix where `d_{j+1} == argmax(row j)`. A partial
//! accept rewinds the trunk ([`Glm5Engine::snapshot`]/`restore`) and re-runs the accepted
//! prefix (the KDA recurrence has no per-token checkpoints yet), so the pass count below is
//! the honest one; `ideal` counts verify passes only (what per-token KDA checkpoints buy).

use super::engine::{argmax, Glm5Engine};
use super::mla::MlaWeights;
use crate::flash::reference::{rms_norm_mul, Mat};
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

/// The NextN block's weights plus its own per-cell DSA cache.
pub struct Mtp {
    pub layer: usize,
    n: usize,
    eps: f32,
    eh: Mat,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    shnorm: Vec<f32>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mla: MlaWeights,
    head: Mat,
    idx_top_k: usize,
    // per-cell caches (positions 0..len)
    latent: Vec<f32>,
    idx_k: Vec<f32>,
    idx_g: Vec<f32>,
    /// The device home of this block (`TANG_GLM5_MTP_DEV`, default on with a GPU engine):
    /// its own MLA caches + projections, its MoE FFN resident, the trunk's device LM head.
    #[cfg(feature = "cuda")]
    dev: Option<MtpDev>,
}

/// The NextN block on the device: the same pieces a trunk DSA layer uses (W3's `MlaCuda`
/// with its own latent / indexer caches, the resident projections `MlaProj`, the W5 FFN
/// `DevMoe`), called through their host-in / host-out entry points.
#[cfg(feature = "cuda")]
struct MtpDev {
    cu: super::mla::cuda::MlaCuda,
    p: super::mla_dev::MlaProj,
    c: super::engine::DsaCache,
    moe: super::moe::gpu::DevMoe,
    /// eh_proj on the device (`k5_gemv`, GGUF storage type), its input `[t][2n]` / output
    /// `[t][n]` scratch (MAX_EH rows). Opt-in `TANG_GLM5_MTP_EH_DEV=1`: on the generic
    /// thread-per-row `k5_gemv` it measured 2.9 ms a cell vs 0.9 on the host (B300).
    eh: Option<(
        super::moe::gpu::DevW,
        super::moe::gpu::Kernels5,
        tang_moe::gpu::DevBuf,
        tang_moe::gpu::DevBuf,
    )>,
    /// The whole cell as one device graph (`resident::DraftCell`; `None`: host-staged cells).
    cell: Option<super::engine::resident::DraftCell>,
}

/// Rows the device eh_proj scratch holds (a cells call past it runs eh_proj on the host).
#[cfg(feature = "cuda")]
const MAX_EH: usize = 16;

/// What a batch of cells gives back.
pub struct CellOut {
    /// `hn` rows `[t][n]` (the next chain step's `h`).
    pub h: Vec<f32>,
    /// argmax draft per cell.
    pub top: Vec<u32>,
}

impl Mtp {
    pub fn load(e: &Glm5Engine) -> Result<Self> {
        let (m, hp) = (&e.m, &e.m.hp);
        ensure!(hp.n_nextn >= 1, "no NextN layer (nextn_predict_layers = 0)");
        let l = hp.n_layer;
        let v = |s: &str| m.vec(&super::t(l, s));
        let eh = m.mat(l, "nextn.eh_proj.weight")?;
        ensure!(
            eh.cols == 2 * hp.n_embd && eh.rows == hp.n_embd,
            "eh_proj is {}x{}",
            eh.rows,
            eh.cols
        );
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut me = Self {
            layer: l,
            n: hp.n_embd,
            eps: hp.eps,
            eh,
            enorm: v("nextn.enorm.weight")?,
            hnorm: v("nextn.hnorm.weight")?,
            shnorm: v("nextn.shared_head_norm.weight")?,
            attn_norm: v("attn_norm.weight")?,
            ffn_norm: v("ffn_norm.weight")?,
            mla: m.mla_weights(l)?,
            head: m.head()?,
            idx_top_k: hp.idx_top_k,
            latent: Vec::new(),
            idx_k: Vec::new(),
            idx_g: Vec::new(),
            #[cfg(feature = "cuda")]
            dev: None,
        };
        #[cfg(feature = "cuda")]
        me.load_dev(e)?;
        Ok(me)
    }

    /// Put the block on the engine's GPU 0 (no-op without one, or `TANG_GLM5_MTP_DEV=0`).
    #[cfg(feature = "cuda")]
    fn load_dev(&mut self, e: &Glm5Engine) -> Result<()> {
        if std::env::var("TANG_GLM5_MTP_DEV").is_ok_and(|v| v == "0") {
            return Ok(());
        }
        let Some(gf) = e.gpu.as_ref() else {
            return Ok(());
        };
        let (gpu, _) = gf.dev0();
        gpu.bind()
            .map_err(|er| anyhow::anyhow!("bind GPU 0 (mtp): {er}"))?;
        let g = &e.m.g;
        let cu = super::mla::cuda::MlaCuda::new_at(&self.mla, 0)
            .map_err(|er| anyhow::anyhow!("mtp MlaCuda: {er}"))?;
        let p = super::mla_dev::MlaProj::new(gpu, g, self.layer, &self.mla)?;
        let mp = super::moe::MoeParams::from_gguf(g, self.layer)?;
        let moe = super::moe::gpu::DevMoe::new(&mp)?;
        let eh = if !std::env::var("TANG_GLM5_MTP_EH_DEV").is_ok_and(|v| v == "1") {
            None
        } else {
            let info = g.info(&super::t(self.layer, "nextn.eh_proj.weight"))?;
            let n = self.n;
            let dw = super::moe::gpu::upload(&super::moe::W::Gguf(g, info), 1, n, 2 * n)?;
            let k5 = super::moe::gpu::Kernels5::new(gpu)?;
            let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp eh_proj: {er}");
            Some((
                dw,
                k5,
                tang_moe::gpu::DevBuf::zeroed_elems::<f32>(MAX_EH * 2 * n).map_err(ge)?,
                tang_moe::gpu::DevBuf::zeroed_elems::<f32>(MAX_EH * n).map_err(ge)?,
            ))
        };
        let cell = if super::engine::resident::cell_dev_on() {
            super::engine::resident::DraftCell::new(
                e,
                self.layer,
                [
                    &self.enorm,
                    &self.hnorm,
                    &self.attn_norm,
                    &self.ffn_norm,
                    &self.shnorm,
                ],
                self.eps,
            )?
        } else {
            None
        };
        self.dev = Some(MtpDev {
            cu,
            p,
            c: Default::default(),
            moe,
            eh,
            cell,
        });
        eprintln!(
            "glm5: MTP layer blk.{} resident on GPU 0 (DSA + FFN + head)",
            self.layer
        );
        Ok(())
    }

    /// Drop the device home: the host reference path (tests compare the two).
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    pub fn host_only(mut self) -> Self {
        #[cfg(feature = "cuda")]
        {
            self.dev = None;
        }
        self
    }

    /// Host LM-head logits of `hn` rows `[t][n]` (tests: the draft margin).
    pub fn host_logits(&self, hn: &[f32]) -> Vec<f32> {
        self.head.apply(hn, hn.len() / self.n, false)
    }

    /// Is the block on the device?
    pub fn on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        return self.dev.is_some();
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Device draft-cell graph replays so far (0: no device cell, tests).
    pub fn cell_replays(&self) -> usize {
        #[cfg(feature = "cuda")]
        if let Some(c) = self.dev.as_ref().and_then(|d| d.cell.as_ref()) {
            return c.replays;
        }
        0
    }

    /// Cells in the cache (= the next cell's position).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        #[cfg(feature = "cuda")]
        if let Some(d) = &self.dev {
            return d.c.valid_len();
        }
        self.latent.len() / self.mla.kv_lora
    }

    /// Drop cells at positions `>= pos`.
    pub fn truncate(&mut self, pos: usize) {
        #[cfg(feature = "cuda")]
        if let Some(d) = &mut self.dev {
            d.c.truncate_rows(pos, self.mla.kv_lora, self.mla.idx_dim);
            // the device watermarks too (rows past `pos` are overwritten by the next append)
            d.cu.set_seq(0);
            if d.cu.row_cap() > 0 {
                d.cu.append_done(pos, 0);
                d.cu.set_pool_upto(d.cu.pool_upto().min(pos / self.mla.kpool.max(1)));
            }
        }
        self.latent.truncate(pos * self.mla.kv_lora);
        self.idx_k.truncate(pos * self.mla.idx_dim);
        self.idx_g.truncate(pos * self.mla.idx_dim);
    }

    /// Run `toks.len()` cells at positions `len()..`, cell `i` pairing `h[i]` with `toks[i]`.
    pub fn cells(&mut self, e: &Glm5Engine, h: &[f32], toks: &[u32]) -> Result<CellOut> {
        let g = &e.m.g;
        let (n, t, eps) = (self.n, toks.len(), self.eps);
        ensure!(
            h.len() == t * n,
            "mtp: h is {} floats, want {}",
            h.len(),
            t * n
        );
        #[cfg(feature = "cuda")]
        if let Some(d) = self.dev.as_mut() {
            if let Some(c) = d.cell.as_ref() {
                c.stream()
                    .sync()
                    .map_err(|er| anyhow::anyhow!("mtp: {er}"))?;
            }
            super::engine::resident::refresh_cell_mirrors(&d.cu, &mut d.c)?;
        }
        let pos0 = self.len();
        let embd = g.info("token_embd.weight")?;
        // ---- [enorm(e) ; hnorm(h)] -> eh_proj
        let mut cat = vec![0f32; t * 2 * n];
        for (i, &tok) in toks.iter().enumerate() {
            let row = &mut cat[i * 2 * n..(i + 1) * 2 * n];
            row[..n].copy_from_slice(&g.rows(embd, tok as usize, 1)?);
            rms_norm_mul(&mut row[..n], &self.enorm, eps);
            row[n..].copy_from_slice(&h[i * n..(i + 1) * n]);
            rms_norm_mul(&mut row[n..], &self.hnorm, eps);
        }
        let pr = |k: &'static str, t0: Instant| {
            #[cfg(feature = "cuda")]
            if super::engine::resident::prof_on() {
                super::engine::resident::prof_add(k, t0.elapsed().as_secs_f64());
            }
            let _ = (k, t0);
        };
        let t0 = Instant::now();
        #[cfg(feature = "cuda")]
        let dev_x = match self
            .dev
            .as_ref()
            .and_then(|d| d.eh.as_ref())
            .filter(|_| t <= MAX_EH)
        {
            Some((dw, k5, xin, xout)) => {
                let (gpu, _) = e.gpu.as_ref().context("mtp: no GPU")?.dev0();
                gpu.bind()
                    .map_err(|er| anyhow::anyhow!("bind (mtp): {er}"))?;
                let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp eh_proj: {er}");
                xin.write(0, &cat).map_err(ge)?;
                let s = tang_moe::gpu::Stream::blocking().map_err(ge)?;
                unsafe {
                    tang_moe::gpu::launch(
                        k5.gemv,
                        ((t * n).div_ceil(128) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        &s,
                        tang_moe::args![
                            dw.w,
                            0u32,
                            xin.ptr,
                            xout.ptr,
                            n as u32,
                            t as u32,
                            n as u32,
                            (2 * n) as u32
                        ],
                    )
                    .map_err(ge)?;
                }
                s.sync().map_err(ge)?;
                Some(xout.read::<f32>(0, t * n).map_err(ge)?)
            }
            None => None,
        };
        #[cfg(not(feature = "cuda"))]
        let dev_x: Option<Vec<f32>> = None;
        let mut x = match dev_x {
            Some(x) => x,
            None => self.eh.apply(&cat, t, false),
        };
        pr("draft.eh_proj", t0);
        let t0 = Instant::now();
        // ---- attention half (the engine's dsa_cpu, on this block's own cache)
        let mut xa = x.clone();
        xa.par_chunks_mut(n)
            .for_each(|r| rms_norm_mul(r, &self.attn_norm, eps));
        #[cfg(feature = "cuda")]
        let y = match self.dev.as_mut() {
            Some(d) => {
                let (gpu, _) = e
                    .gpu
                    .as_ref()
                    .context("mtp: device block without a GPU")?
                    .dev0();
                gpu.bind()
                    .map_err(|er| anyhow::anyhow!("bind (mtp): {er}"))?;
                d.cu.set_seq(0);
                super::engine::dsa_gpu_resident(
                    self.layer, &self.mla, &d.cu, &d.p, &xa, t, pos0, &mut d.c, None,
                )?
            }
            None => self.dsa_host(&xa, t, pos0),
        };
        #[cfg(not(feature = "cuda"))]
        let y = self.dsa_host(&xa, t, pos0);
        x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        pr("draft.dsa", t0);
        let t0 = Instant::now();
        // ---- FFN half
        let mut xf = x.clone();
        xf.par_chunks_mut(n)
            .for_each(|r| rms_norm_mul(r, &self.ffn_norm, eps));
        #[cfg(feature = "cuda")]
        let y = match self.dev.as_ref() {
            Some(d) => {
                let (_, k) = e.gpu.as_ref().context("mtp: no GPU")?.dev0();
                super::moe::gpu::ffn_host(k, &d.moe, &xf, t)?
            }
            None => {
                let p = super::moe::MoeParams::from_gguf(g, self.layer)?;
                super::moe::moe_layer(&p, &xf, t)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let y = {
            let p = super::moe::MoeParams::from_gguf(g, self.layer)?;
            super::moe::moe_layer(&p, &xf, t)?
        };
        x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        pr("draft.moe", t0);
        let _t0 = Instant::now();
        // ---- shared head
        x.par_chunks_mut(n)
            .for_each(|r| rms_norm_mul(r, &self.shnorm, eps));
        let nv = self.head.rows;
        #[cfg(feature = "cuda")]
        if let Some(gf) = e.gpu.as_ref().filter(|_| self.dev.is_some()) {
            if let Some(top) = e.head_top(&x)? {
                pr("draft.head", t0);
                return Ok(CellOut { h: x, top });
            }
            let mut top = Vec::with_capacity(t);
            for i in 0..t {
                match gf.head_logits(&x[i * n..(i + 1) * n], n, nv)? {
                    Some(lg) => top.push(argmax(&lg)),
                    None => top.push(argmax(&self.head.apply(&x[i * n..(i + 1) * n], 1, false))),
                }
            }
            pr("draft.head", t0);
            return Ok(CellOut { h: x, top });
        }
        let logits = self.head.apply(&x, t, false);
        let top = (0..t)
            .map(|i| argmax(&logits[i * nv..(i + 1) * nv]))
            .collect();
        Ok(CellOut { h: x, top })
    }
}

impl Mtp {
    /// The attention half on the host (f32 caches).
    fn dsa_host(&mut self, xa: &[f32], t: usize, pos0: usize) -> Vec<f32> {
        let n = self.n;
        let xa = xa.to_vec();
        let w = &self.mla;
        let kv_len = pos0 + t;
        let q_resid = w.gemv_q_a(&xa, t);
        self.latent.extend_from_slice(&w.gemv_kv_a(&xa, t));
        let idd = w.idx_dim;
        for i in 0..t {
            let (mut k, mut gg) = (vec![0f32; idd], vec![0f32; idd]);
            super::mla::indexer_key(w, &xa[i * n..(i + 1) * n], &mut k, &mut gg);
            self.idx_k.extend_from_slice(&k);
            self.idx_g.extend_from_slice(&gg);
        }
        // index_share_for_mtp_iteration is about reusing the trunk's top-k across MTP
        // iterations; below idx_top_k the indexer is dense either way.
        let sel = if super::mla::indexer_is_dense(kv_len, self.idx_top_k) {
            None
        } else {
            Some(super::mla::indexer_select(
                w,
                &xa,
                &q_resid,
                &self.idx_k,
                &self.idx_g,
                &vec![1u8; kv_len],
                &vec![1u8; t],
                t,
                kv_len,
                pos0,
            ))
        };
        let mut y = vec![0f32; t * n];
        super::mla::mla_forward(
            w,
            &xa,
            &q_resid,
            &self.latent,
            kv_len,
            t,
            pos0,
            None,
            sel.as_deref(),
            &mut y,
        );
        y
    }
}

/// Something that proposes draft tokens. `observe` gets every committed trunk window's
/// `h` rows with the token that FOLLOWS each position; `draft` then proposes up to `k`
/// tokens after the last committed one.
pub trait Drafter {
    fn observe(&mut self, e: &Glm5Engine, h: &[f32], next_toks: &[u32]) -> Result<()>;
    fn draft(&mut self, e: &Glm5Engine, k: usize) -> Result<Vec<u32>>;
    /// The drafts enqueued on the device without waiting: (ids ptr, count, event fired when
    /// they are written). `None`: use [`Self::draft`].
    #[cfg(feature = "cuda")]
    fn draft_async(
        &mut self,
        _e: &Glm5Engine,
        _k: usize,
    ) -> Result<Option<(u64, usize, tang_moe::gpu::Event)>> {
        Ok(None)
    }
    /// [`Self::observe`] of the last resident window's first `rows` rows, read from the device
    /// (`h` / `toks` are the same rows on the host, for a drafter that needs them).
    #[cfg(feature = "cuda")]
    fn observe_window(&mut self, e: &Glm5Engine, h: &[f32], toks: &[u32]) -> Result<()> {
        self.observe(e, h, toks)
    }
    /// [`Self::observe`] of ONE row whose token / h live on the device (after a plain
    /// resident step): async; returns the event the next trunk launch must wait on, or
    /// `None` when the drafter can't (the caller has no host row to fall back to).
    #[cfg(feature = "cuda")]
    fn observe_ptrs(
        &mut self,
        _e: &Glm5Engine,
        _tok: u64,
        _h: u64,
    ) -> Result<Option<tang_moe::gpu::Event>> {
        Ok(None)
    }
    /// After [`Self::draft_async`]: wait for and read its `n` ids (the host-window fallback).
    #[cfg(feature = "cuda")]
    fn async_ids(&mut self, _n: usize) -> Result<Vec<u32>> {
        anyhow::bail!("no async drafts")
    }
}

/// The MTP drafter: teacher-forced cells for committed positions, chained cells for drafts.
pub struct MtpDrafter {
    pub mtp: Mtp,
    /// the last committed cell's (hn, argmax)
    last: Option<(Vec<f32>, u32)>,
    /// the last committed cell ran on the device (its hn / argmax live there)
    dev_last: bool,
}

impl MtpDrafter {
    pub fn new(mtp: Mtp) -> Self {
        Self {
            mtp,
            last: None,
            dev_last: false,
        }
    }

    /// Teacher-forced device cells for every row (async). `Ok(i)`: rows `[0, i)` ran
    /// (`i < t`: the rest must take the host cell).
    #[cfg(feature = "cuda")]
    fn observe_dev(&mut self, e: &Glm5Engine, h: &[f32], toks: &[u32]) -> Result<usize> {
        let n = self.mtp.n;
        let Some(d) = self.mtp.dev.as_mut() else {
            return Ok(0);
        };
        let Some(cell) = d.cell.as_mut() else {
            return Ok(0);
        };
        e.gpu
            .as_ref()
            .context("mtp: no GPU")?
            .dev0()
            .0
            .bind()
            .map_err(|er| anyhow::anyhow!("bind: {er}"))?;
        for (i, &tk) in toks.iter().enumerate() {
            if !cell.run(
                e,
                &d.cu,
                &d.p,
                &self.mtp.mla,
                &d.moe,
                &mut d.c,
                Some((tk, &h[i * n..(i + 1) * n])),
                None,
                None,
            )? {
                return Ok(i);
            }
        }
        Ok(toks.len())
    }

    /// The chained device draft after a device observe: `k` ids, one sync.
    #[cfg(feature = "cuda")]
    fn draft_dev(&mut self, e: &Glm5Engine, k: usize) -> Result<Vec<u32>> {
        let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp draft: {er}");
        let keep = self.mtp.len();
        let d = self.mtp.dev.as_mut().context("mtp: no device block")?;
        let cell = d.cell.as_mut().context("mtp: no device cell")?;
        let k = k.min(15);
        let mut got = 0usize;
        let tr = ProfT::new("draft.chain_enqueue");
        while got < k {
            if got > 0
                && !cell.run(
                    e,
                    &d.cu,
                    &d.p,
                    &self.mtp.mla,
                    &d.moe,
                    &mut d.c,
                    None,
                    None,
                    None,
                )?
            {
                break;
            }
            tang_moe::gpu::copy_d2d_async(
                cell.ids.ptr + 4 * got as u64,
                cell.tp.ptr,
                4,
                cell.stream(),
            )
            .map_err(ge)?;
            got += 1;
        }
        drop(tr);
        let tr = ProfT::new("draft.sync");
        cell.stream().sync().map_err(ge)?;
        drop(tr);
        let ids = cell.ids.read::<u32>(0, got).map_err(ge)?;
        self.mtp.truncate(keep);
        Ok(ids)
    }
}

impl Drafter for MtpDrafter {
    fn observe(&mut self, e: &Glm5Engine, h: &[f32], next_toks: &[u32]) -> Result<()> {
        self.dev_last = false;
        #[cfg(feature = "cuda")]
        {
            let done = self.observe_dev(e, h, next_toks)?;
            if done == next_toks.len() && done > 0 {
                self.dev_last = true;
                self.last = None;
                return Ok(());
            }
            if done > 0 {
                let n = self.mtp.n;
                return self.observe(e, &h[done * n..], &next_toks[done..]);
            }
        }
        let o = self.mtp.cells(e, h, next_toks)?;
        let n = self.mtp.n;
        let t = next_toks.len();
        self.last = Some((o.h[(t - 1) * n..].to_vec(), o.top[t - 1]));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn draft_async(
        &mut self,
        e: &Glm5Engine,
        k: usize,
    ) -> Result<Option<(u64, usize, tang_moe::gpu::Event)>> {
        if !self.dev_last || !std::env::var("TANG_GLM5_MTP_CHAIN").is_ok_and(|v| v == "1") {
            return Ok(None);
        }
        let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp draft: {er}");
        let keep = self.mtp.len();
        let d = self.mtp.dev.as_mut().context("mtp: no device block")?;
        let cell = d.cell.as_mut().context("mtp: no device cell")?;
        let k = k.min(15);
        let mut got = 0usize;
        while got < k {
            if got > 0
                && !cell.run(
                    e,
                    &d.cu,
                    &d.p,
                    &self.mtp.mla,
                    &d.moe,
                    &mut d.c,
                    None,
                    None,
                    None,
                )?
            {
                break;
            }
            tang_moe::gpu::copy_d2d_async(
                cell.ids.ptr + 4 * got as u64,
                cell.tp.ptr,
                4,
                cell.stream(),
            )
            .map_err(ge)?;
            got += 1;
        }
        let ev = tang_moe::gpu::Event::new(false).map_err(ge)?;
        ev.record(cell.stream()).map_err(ge)?;
        let ids = cell.ids.ptr;
        self.mtp.truncate(keep);
        Ok(Some((ids, got, ev)))
    }

    #[cfg(feature = "cuda")]
    fn observe_ptrs(
        &mut self,
        e: &Glm5Engine,
        _tok: u64,
        _h: u64,
    ) -> Result<Option<tang_moe::gpu::Event>> {
        if !self.dev_last {
            return Ok(None);
        }
        let n = self.mtp.n;
        let Some(d) = self.mtp.dev.as_mut() else {
            return Ok(None);
        };
        let Some(cell) = d.cell.as_mut() else {
            return Ok(None);
        };
        // the step's outputs staged on the trunk stream; the cell runs concurrently with the
        // next trunk launch (own MoE pool, own MLA caches): the trunk never waits for it
        let (st, sh) = cell.stage_next(n);
        let Some(ev) = e.step_out_stage(st, sh)? else {
            return Ok(None);
        };
        if !cell.run(
            e,
            &d.cu,
            &d.p,
            &self.mtp.mla,
            &d.moe,
            &mut d.c,
            None,
            Some((st, sh)),
            Some(&ev),
        )? {
            return Ok(None);
        }
        Ok(Some(ev))
    }

    #[cfg(feature = "cuda")]
    fn observe_window(&mut self, e: &Glm5Engine, h: &[f32], toks: &[u32]) -> Result<()> {
        let n = self.mtp.n;
        let ptrs = e.window_out_ptrs();
        let on = !std::env::var("TANG_GLM5_MTP_OBS_DEV").is_ok_and(|v| v == "0");
        if let (Some((ip, hp)), true, Some(d)) = (ptrs, on, self.mtp.dev.as_mut()) {
            if let Some(cell) = d.cell.as_mut() {
                let _t = ProfT::new("draft.observe_window");
                for i in 0..toks.len() {
                    let src = (ip + 4 * i as u64, hp + (i * n * 4) as u64);
                    if !cell.run(
                        e,
                        &d.cu,
                        &d.p,
                        &self.mtp.mla,
                        &d.moe,
                        &mut d.c,
                        None,
                        Some(src),
                        None,
                    )? {
                        self.dev_last = false;
                        return self.observe(e, &h[i * n..], &toks[i..]);
                    }
                }
                self.dev_last = true;
                self.last = None;
                return Ok(());
            }
        }
        self.observe(e, h, toks)
    }

    #[cfg(feature = "cuda")]
    fn async_ids(&mut self, n: usize) -> Result<Vec<u32>> {
        let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp draft: {er}");
        let d = self.mtp.dev.as_mut().context("mtp: no device block")?;
        let cell = d.cell.as_mut().context("mtp: no device cell")?;
        cell.stream().sync().map_err(ge)?;
        cell.ids.read::<u32>(0, n).map_err(ge)
    }

    fn draft(&mut self, e: &Glm5Engine, k: usize) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        if self.dev_last {
            return self.draft_dev(e, k);
        }
        let Some((mut h, mut d)) = self.last.clone() else {
            return Ok(vec![]);
        };
        let keep = self.mtp.len();
        let mut out = vec![d];
        while out.len() < k {
            let o = self.mtp.cells(e, &h, &[d])?;
            h = o.h;
            d = o.top[0];
            out.push(d);
        }
        self.mtp.truncate(keep);
        Ok(out)
    }
}

/// `TANG_GLM5_WIN_PROF=1`: time a scope into the resident profile.
#[allow(dead_code)]
struct ProfT(&'static str, Instant);

#[allow(dead_code)]
impl ProfT {
    fn new(k: &'static str) -> Self {
        Self(k, Instant::now())
    }
}

impl Drop for ProfT {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if super::engine::resident::prof_on() {
            super::engine::resident::prof_add(self.0, self.1.elapsed().as_secs_f64());
        }
    }
}

/// Decode statistics.
#[derive(Default, Debug, Clone)]
pub struct Stats {
    /// verify windows run (one per loop iteration)
    pub verify: usize,
    /// re-run windows after a partial accept (rewind)
    pub rerun: usize,
    /// drafts reached / accepted at each depth (index 0 = first draft; depth d is reached
    /// when every earlier draft in its window was accepted)
    pub proposed: Vec<usize>,
    pub accepted: Vec<usize>,
    /// tokens produced by the decode loop (excluding the prefill's token)
    pub tokens: usize,
    pub draft_s: f64,
    pub trunk_s: f64,
    /// verify windows that ran on the resident path ([`Glm5Engine::window_resident`]),
    /// the device rewinds after a partial accept, and their time (inside `trunk_s`)
    pub res_windows: usize,
    pub res_rewinds: usize,
    /// passes whose draft cells and verify window ran chained (one host wake)
    pub chained: usize,
    /// `--mtp auto`: plain resident steps taken (K = 0) and passes per K
    pub plain_steps: usize,
    pub auto_picks: Vec<usize>,
    /// the prompt's teacher-forced draft cells (once, after the prefill)
    pub observe0_s: f64,
    pub rewind_s: f64,
}

impl Stats {
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (d, (p, a)) in self.proposed.iter().zip(&self.accepted).enumerate() {
            s += &format!(
                "mtp: depth {} accept {}/{} = {:.1}%\n",
                d + 1,
                a,
                p,
                100.0 * *a as f64 / (*p).max(1) as f64
            );
        }
        let passes = self.verify + self.rerun;
        s += &format!(
            "mtp: {} tokens, {} verify + {} rerun trunk passes: {:.3} tok/pass (ideal, verify only: {:.3}); draft {:.1} s, trunk {:.1} s\n",
            self.tokens,
            self.verify,
            self.rerun,
            self.tokens as f64 / passes.max(1) as f64,
            self.tokens as f64 / self.verify.max(1) as f64,
            self.draft_s,
            self.trunk_s
        );
        if !self.auto_picks.is_empty() {
            s += &format!(
                "mtp: auto K picks (K=0..): {:?}; {} plain steps\n",
                self.auto_picks, self.plain_steps
            );
        }
        if self.res_windows > 0 {
            s += &format!(
                "mtp: resident verify: {} windows ({} chained), {} device rewinds ({:.3} ms each); trunk {:.2} ms/pass, draft {:.2} ms/pass; prompt cells {:.1} ms (prefill, not in draft)\n",
                self.res_windows,
                self.chained,
                self.res_rewinds,
                1e3 * self.rewind_s / self.res_rewinds.max(1) as f64,
                1e3 * self.trunk_s / passes.max(1) as f64,
                1e3 * self.draft_s / passes.max(1) as f64,
                1e3 * self.observe0_s
            );
        }
        s
    }
}

/// `--mtp auto` / `TANG_GLM5_MTP=auto`: [`generate`] picks K in `0..=cap` every pass.
pub static AUTO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The adaptive-K controller. Acceptance per draft depth: EWMA (alpha 0.1) of the conditional
/// accept. Cost per K: the MEDIAN wall ms of its last `WIN` passes (draft + verify + observe;
/// K = 0 = a measured plain resident step + its concurrent observe), each K's first pass
/// dropped (captures). Sticky: the current K stays unless another K's expected tokens per ms
/// `(1 + sum_d prod_{j<=d} p_j) / cost_K` beats it by >= `MARGIN` for `STREAK` consecutive
/// passes; one probe of an untried / stalest K at most every `PROBE` passes.
struct AutoK {
    cap: usize,
    p: Vec<f64>,
    samples: Vec<std::collections::VecDeque<f64>>,
    seen: Vec<usize>,
    last: Vec<usize>,
    pass: usize,
    cur: usize,
    streak: usize,
    cand: usize,
    probe_at: usize,
    probing: Option<usize>,
    picks: Vec<usize>,
    #[allow(dead_code)]
    log: bool,
}

#[allow(dead_code)]
const WIN: usize = 16;
const MARGIN: f64 = 1.05;
const STREAK: usize = 8;
const PROBE: usize = 64;

impl AutoK {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            p: vec![0.7; cap],
            samples: vec![Default::default(); cap + 1],
            seen: vec![0; cap + 1],
            last: vec![0; cap + 1],
            pass: 0,
            cur: 1.min(cap),
            streak: 0,
            cand: 0,
            probe_at: 0,
            probing: None,
            picks: vec![0; cap + 1],
            log: std::env::var("TANG_GLM5_MTP_AUTO_LOG").is_ok_and(|v| v == "1"),
        }
    }

    fn expect(&self, k: usize) -> f64 {
        let (mut e, mut q) = (1.0, 1.0);
        for d in 0..k {
            q *= self.p[d];
            e += q;
        }
        e
    }

    fn cost(&self, k: usize) -> Option<f64> {
        let mut v: Vec<f64> = self.samples[k].iter().copied().collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[v.len() / 2])
    }

    fn rate(&self, k: usize) -> Option<f64> {
        self.cost(k).map(|c| self.expect(k) / c)
    }

    fn pick(&mut self) -> usize {
        self.pass += 1;
        // warm-up: two passes of every K (the first is dropped) before any decision
        if let Some(k) = (0..=self.cap).find(|&k| self.seen[k] < 2) {
            self.probing = None;
            return self.take(k);
        }
        // a probe of the stalest K, at most once per PROBE passes
        if self.pass >= self.probe_at + PROBE {
            self.probe_at = self.pass;
            let k = (0..=self.cap)
                .filter(|&k| k != self.cur)
                .min_by_key(|&k| self.last[k])
                .unwrap();
            self.probing = Some(k);
            return self.take(k);
        }
        self.probing = None;
        let best = (0..=self.cap)
            .max_by(|&a, &b| {
                self.rate(a)
                    .unwrap_or(0.0)
                    .partial_cmp(&self.rate(b).unwrap_or(0.0))
                    .unwrap()
            })
            .unwrap();
        let (rb, rc) = (
            self.rate(best).unwrap_or(0.0),
            self.rate(self.cur).unwrap_or(0.0),
        );
        if best != self.cur && rb >= MARGIN * rc {
            if best == self.cand {
                self.streak += 1;
            } else {
                self.cand = best;
                self.streak = 1;
            }
            if self.streak >= STREAK {
                self.cur = best;
                self.streak = 0;
            }
        } else {
            self.streak = 0;
        }
        let k = self.cur;
        self.take(k)
    }

    fn take(&mut self, k: usize) -> usize {
        self.last[k] = self.pass;
        self.picks[k] += 1;
        k
    }

    /// A pass of `k` drafts took `ms`; `a` drafts were accepted.
    #[allow(dead_code)]
    fn update(&mut self, k: usize, a: usize, ms: f64) {
        self.seen[k] += 1;
        if self.seen[k] > 1 {
            let q = &mut self.samples[k];
            q.push_back(ms);
            while q.len() > WIN {
                q.pop_front();
            }
        }
        for d in 0..k.min(a + 1) {
            let v = if d < a { 1.0 } else { 0.0 };
            self.p[d] += 0.1 * (v - self.p[d]);
        }
        if self.log && self.pass.is_multiple_of(32) {
            eprintln!("{}", self.report());
        }
    }

    fn report(&self) -> String {
        let mut s = format!(
            "mtp auto @pass {} (cur K={}): p {:?}",
            self.pass,
            self.cur,
            self.p
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        for k in 0..=self.cap {
            s += &format!(
                "; K={k} cost {} ms E {:.3} rate {} tok/ms",
                self.cost(k).map_or("-".into(), |c| format!("{c:.2}")),
                self.expect(k),
                self.rate(k).map_or("-".into(), |r| format!("{r:.4}"))
            );
        }
        s
    }
}

/// Greedy draft-and-verify: prefill `prompt`, then generate until `n` tokens (the first is
/// the prefill's). Output equals plain greedy decode whatever the drafter proposes.
pub fn generate(
    e: &mut Glm5Engine,
    dr: &mut dyn Drafter,
    prompt: &[u32],
    n: usize,
    chunk: usize,
    k: usize,
) -> Result<(Vec<u32>, Stats)> {
    ensure!(
        !prompt.is_empty() && n >= 1 && k >= 1,
        "mtp generate: bad args"
    );
    let nv = e.m.hp.n_vocab;
    let mut st = Stats {
        proposed: vec![0; k],
        accepted: vec![0; k],
        ..Default::default()
    };
    // ---- prefill, collecting h for every prompt position
    let mut hs = Vec::new();
    let mut last = Vec::new();
    for c in prompt.chunks(chunk.max(1)) {
        let lg = e.window(c)?;
        hs.extend_from_slice(&e.last_h);
        last = lg[(c.len() - 1) * nv..].to_vec();
    }
    let mut cur = argmax(&last);
    let mut gen = vec![cur];
    let mut nexts: Vec<u32> = prompt[1..].to_vec();
    nexts.push(cur);
    let td = Instant::now();
    dr.observe(e, &hs, &nexts)?;
    // the prompt's teacher-forced cells are prefill work: reported apart, not in draft_s
    st.observe0_s = td.elapsed().as_secs_f64();
    let auto = AUTO.load(std::sync::atomic::Ordering::Relaxed);
    let mut ak = AutoK::new(k);
    #[cfg(feature = "cuda")]
    let mut pending: Option<tang_moe::gpu::Event> = None;
    while gen.len() < n {
        let td = Instant::now();
        // never draft past n: the window is cur + drafts
        let kk = if auto { ak.pick() } else { k }.min(n - gen.len());
        #[cfg(feature = "cuda")]
        let tp0 = td;
        #[cfg(feature = "cuda")]
        if let Some(ev) = pending.take() {
            e.trunk_wait(&ev)?;
        }
        // K = 0: one plain resident step; the drafter observes its h / next token from the
        // device (no host rows), ordered before the next trunk launch by `pending`
        #[cfg(feature = "cuda")]
        if kk == 0 {
            let next = e.step(cur)?;
            let ptrs = e.step_out_ptrs();
            let ev = match ptrs {
                Some((tp, hp)) => dr.observe_ptrs(e, tp, hp)?,
                None => None,
            };
            match ev {
                Some(_) => {}
                None => {
                    // the device cell can't take it (cache capacity): the host row
                    let h = e
                        .step_h_read()?
                        .context("mtp auto: K=0 without a device head")?;
                    dr.observe(e, &h, &[next])?;
                }
            }
            st.plain_steps += 1;
            gen.push(next);
            st.tokens += 1;
            cur = next;
            let ms = td.elapsed().as_secs_f64() * 1e3;
            st.trunk_s += ms / 1e3;
            ak.update(0, 0, ms);
            continue;
        }
        // chained: the draft cells and the verify window enqueued back to back, one host wake
        #[cfg(feature = "cuda")]
        let mut pre: Option<(Vec<u32>, Vec<f32>)> = None;
        #[cfg(feature = "cuda")]
        let drafts = match dr.draft_async(e, kk)? {
            Some((ids, got, ev)) => {
                st.draft_s += td.elapsed().as_secs_f64();
                let tw = Instant::now();
                let r = e.window_resident_dev(cur, ids, got, &ev)?;
                st.trunk_s += tw.elapsed().as_secs_f64();
                match r {
                    Some((all, g, h)) => {
                        st.chained += 1;
                        pre = Some((g, h));
                        all[1..].to_vec()
                    }
                    None => dr.async_ids(got)?,
                }
            }
            None => {
                let mut d = dr.draft(e, kk)?;
                d.truncate(kk);
                st.draft_s += td.elapsed().as_secs_f64();
                d
            }
        };
        #[cfg(not(feature = "cuda"))]
        let drafts = {
            let mut d = dr.draft(e, kk)?;
            d.truncate(kk);
            st.draft_s += td.elapsed().as_secs_f64();
            d
        };
        let tt = Instant::now();
        let mut win = vec![cur];
        win.extend_from_slice(&drafts);
        // the resident verify window: one device pass, per-row ids + h, device rewind
        #[cfg(feature = "cuda")]
        let rw = match pre.take() {
            Some(x) => Some(x),
            None => e.window_resident(&win)?,
        };
        #[cfg(feature = "cuda")]
        if let Some((g, hrows)) = rw {
            st.verify += 1;
            st.res_windows += 1;
            let mut a = 0;
            while a < drafts.len() && drafts[a] == g[a] {
                a += 1;
            }
            for d in 0..drafts.len().min(a + 1) {
                st.proposed[d] += 1;
                if d < a {
                    st.accepted[d] += 1;
                }
            }
            if a < drafts.len() {
                let tr = Instant::now();
                e.rewind_resident(win.len(), a + 1)?;
                st.rewind_s += tr.elapsed().as_secs_f64();
                st.res_rewinds += 1;
            }
            let new = g[..=a].to_vec();
            let h = &hrows[..(a + 1) * e.m.hp.n_embd];
            st.trunk_s += tt.elapsed().as_secs_f64();
            let td = Instant::now();
            dr.observe_window(e, h, &new)?;
            st.draft_s += td.elapsed().as_secs_f64();
            cur = *new.last().unwrap();
            gen.extend_from_slice(&new);
            st.tokens += new.len();
            ak.update(drafts.len(), a, tp0.elapsed().as_secs_f64() * 1e3);
            continue;
        }
        let snap = e.snapshot()?;
        let lg = e.window(&win)?;
        st.verify += 1;
        let g: Vec<u32> = (0..win.len())
            .map(|j| argmax(&lg[j * nv..(j + 1) * nv]))
            .collect();
        let mut a = 0;
        while a < drafts.len() && drafts[a] == g[a] {
            a += 1;
        }
        // conditional per depth: depth d counts only when depths < d were accepted
        for d in 0..drafts.len().min(a + 1) {
            st.proposed[d] += 1;
            if d < a {
                st.accepted[d] += 1;
            }
        }
        let mut h = e.last_h.clone();
        if a < drafts.len() {
            // rewind and re-run the accepted prefix (no per-token KDA checkpoints yet)
            e.restore(snap);
            let lg2 = e.window(&win[..a + 1])?;
            st.rerun += 1;
            let g2: Vec<u32> = (0..=a)
                .map(|j| argmax(&lg2[j * nv..(j + 1) * nv]))
                .collect();
            if g2[..] != g[..=a] {
                eprintln!(
                    "mtp: WARNING re-run argmax {:?} != verify {:?} (window-size numerics)",
                    g2,
                    &g[..=a]
                );
            }
            h = e.last_h.clone();
            // the re-run is the authority: its state is what the next window continues
            let new = g2;
            st.trunk_s += tt.elapsed().as_secs_f64();
            let td = Instant::now();
            dr.observe(e, &h, &new)?;
            st.draft_s += td.elapsed().as_secs_f64();
            cur = *new.last().unwrap();
            gen.extend_from_slice(&new);
            st.tokens += new.len();
            continue;
        }
        st.trunk_s += tt.elapsed().as_secs_f64();
        let new = g[..=a].to_vec();
        h.truncate((a + 1) * e.m.hp.n_embd);
        let td = Instant::now();
        dr.observe(e, &h, &new)?;
        st.draft_s += td.elapsed().as_secs_f64();
        cur = *new.last().unwrap();
        gen.extend_from_slice(&new);
        st.tokens += new.len();
    }
    gen.truncate(n);
    // the last window may run past n; count only the kept tokens
    st.tokens = gen.len() - 1;
    if auto {
        st.auto_picks = ak.picks.clone();
        eprintln!("{}", ak.report());
    }
    Ok((gen, st))
}

/// `flash-generate ... --mtp K` (or `TANG_GLM5_MTP=K`): the MTP decode, same output lines as
/// the plain CLI plus the acceptance report.
/// Default for `TANG_GLM5_KDA_WIN_STEP` when the env is unset: [`cli_run`] sets it to
/// the draft depth + 1 (the verify window length), so MTP's KDA windows run as serial
/// t=1 steps and the verify is exact against plain decode. Plain decode leaves it 0.
pub static WIN_STEP_DEFAULT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The effective KDA window-step threshold: `TANG_GLM5_KDA_WIN_STEP` if set and parses,
/// else [`WIN_STEP_DEFAULT`].
pub fn win_step() -> usize {
    std::env::var("TANG_GLM5_KDA_WIN_STEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| WIN_STEP_DEFAULT.load(std::sync::atomic::Ordering::Relaxed))
}

pub fn cli_run(
    e: &mut Glm5Engine,
    ids: &[u32],
    n: usize,
    chunk: usize,
    k: usize,
    out: Option<&Path>,
    quiet: bool,
) -> Result<()> {
    WIN_STEP_DEFAULT.store(k + 1, std::sync::atomic::Ordering::Relaxed);
    eprintln!(
        "glm5: KDA win_step={} (TANG_GLM5_KDA_WIN_STEP overrides)",
        win_step()
    );
    let t0 = Instant::now();
    let mut dr = MtpDrafter::new(Mtp::load(e)?);
    eprintln!(
        "glm5: MTP layer blk.{} loaded, k={k}, {:.1} s",
        dr.mtp.layer,
        t0.elapsed().as_secs_f64()
    );
    let t1 = Instant::now();
    let (gen, st) = generate(e, &mut dr, ids, n, chunk, k)?;
    let s = t1.elapsed().as_secs_f64();
    println!("prompt: {} tokens", ids.len());
    let line = gen
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    println!("ids: {line}");
    println!("decode: {} tokens in {:.2} s (incl. prefill)", gen.len(), s);
    print!("{}", st.report());
    #[cfg(feature = "cuda")]
    if super::engine::resident::prof_on() {
        print!("{}", super::engine::resident::prof_report());
    }
    if !quiet {
        let tk = crate::flash::tokenize::FlashTokenizer::from_gguf(&e.m.g);
        if let Ok(text) = tk.and_then(|t| t.decode(&gen)) {
            println!("text: {text}");
        }
    }
    if let Some(o) = out {
        std::fs::write(o, format!("{line}\n"))?;
    }
    Ok(())
}
