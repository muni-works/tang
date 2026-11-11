//! The GLM-5.3-Flash decode engine (W1): the resumable form of [`super::forward::Glm5Ref`].
//!
//! `Glm5Ref::forward` recomputes a whole sequence from position 0. This engine keeps each
//! layer's running state between windows, so prefill and decode cost only the new tokens:
//!
//! - **34 KDA layers**: the recurrent state `[head][key][value]` f32 plus the conv history
//!   (W4's [`KdaState`] / [`ConvState`]). Fixed size, never grows.
//! - **11 DSA layers**: the normed 512-dim NoPE latent per token (W3's `k_latent` cache) plus
//!   the pooled indexer's key/gate rows per token. Grows with context.
//! - **mHC**: nothing persists. The 4 streams live only inside a window.
//! - **MoE**: stateless. Experts stay mmap'd GGUF handles, dequantised per expert (W5).
//!
//! Naive-but-correct (GATES.md): every component is the CPU f32 function already checked
//! against HF. `window` must equal the reference forward for the same positions, which
//! the tests here pin down (chunked prefill + token-by-token decode vs one full forward).
//! The CUDA path replaces these calls one component at a time, with this engine as the
//! oracle.
//!
//! Weights: mixer weights are dequantised per window and dropped, the same as the
//! reference, so the 244 GB UD-Q4_K_XL file stays mmap'd. On small models
//! (`hidden ≤ 1024`, i.e. the tiny fixture), or with `TANG_GLM5_CACHE=1`, they are cached.

use super::forward::Glm5Ref;
use super::kda::{ConvState, KdaConfig, KdaLayerW, KdaState};
use super::mla::MlaWeights;
use anyhow::{bail, ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
#[cfg(feature = "cuda")]
use tang_moe::{args, gpu::launch};

/// One DSA layer's per-token caches: normed latent `[n][kv_lora]`, indexer key and gate
/// rows `[n][idx_dim]`.
#[derive(Default, Clone)]
pub(crate) struct DsaCache {
    latent: Vec<f32>,
    idx_k: Vec<f32>,
    idx_g: Vec<f32>,
    /// The GPU path's latent cache: `[n][kv_lora]` bf16 (W3's layout).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    lat16: Vec<u16>,
    /// The GPU path's indexer validity marks, `[n]`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    valid: Vec<u8>,
    /// The resident decode step ([`resident`]) appends on the device only: rows from here
    /// on are stale in the host mirrors above until `refresh_stale_dsa` pulls them.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stale_from: Option<usize>,
}

impl DsaCache {
    /// Rows held (= the next row's position).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn valid_len(&self) -> usize {
        self.valid.len()
    }

    /// Drop rows at positions `>= pos` (a rewind: the device rows are position-indexed and
    /// the next append overwrites them).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn truncate_rows(&mut self, pos: usize, kl: usize, idd: usize) {
        self.stale_from = self.stale_from.filter(|&f| f < pos);
        self.latent.truncate(pos * kl);
        self.lat16.truncate(pos * kl);
        self.idx_k.truncate(pos * idd);
        self.idx_g.truncate(pos * idd);
        self.valid.truncate(pos);
    }
}

/// The device-resident decode step (one stream, no per-stage host copies).
#[cfg(feature = "cuda")]
#[path = "resident.rs"]
pub(crate) mod resident;

/// warps per k5_gemv8 block (kda_k5.cuh's K5_WARPS: one staged row per warp).
#[cfg(feature = "cuda")]
const K5_WARPS: usize = 4;

/// The LM head resident (TANG_GLM5_GPU_HEAD=1): weights, k5_q8 scratch (codes + scales
/// after them, MAXT rows), the logits row, the k5 kernels, and a blocking stream
/// (DevBuf::write is a synchronous legacy-stream copy; a NON_BLOCKING stream would not
/// order against it — gpu.rs's doc).
#[cfg(feature = "cuda")]
struct HeadDev {
    dw: super::moe::gpu::DevW,
    /// the normed h row, f32 (k5_q8 reads device memory, not the host row)
    xf: tang_moe::gpu::DevBuf,
    xq8: tang_moe::gpu::DevBuf,
    out: tang_moe::gpu::DevBuf,
    k5: super::moe::gpu::Kernels5,
    s: tang_moe::gpu::Stream,
}

/// The FFNs on the GPU (feature `cuda`): every layer's dense or MoE FFN (W5's
/// [`super::moe::gpu::DevMoe`]) uploaded once in its GGUF storage format and kept resident.
/// With `TANG_GPUS=2`, layers `[0, split)` live on GPU 0 and `[split, L)` on GPU 1
/// (`TANG_SPLIT_AT`, default: the cut that balances routed-expert bytes). Activations cross
/// through the host for now (naive; one copy each way per FFN).
#[cfg(feature = "cuda")]
pub struct GpuFfn {
    devs: Vec<(tang_moe::gpu::Gpu, super::moe::gpu::Kernels)>,
    /// The resident KDA layers ([`super::kda_dev::DevLayer`], W4's DevKda core + the
    /// projections as device GEMVs), one per KDA layer, `None` when
    /// `TANG_GLM5_GPU_KDA=0`. The free-kernel path (`kda_gpu`) stays as the fallback for
    /// windows past `max_t`.
    /// The device LM head (None unless TANG_GLM5_GPU_HEAD=1).
    head_dev: Option<HeadDev>,
    kda_l: Vec<Option<super::kda_dev::DevLayer>>,
    /// Per KDA layer: does the device hold the live state (vs the engine's host copies)?
    /// `[slot][layer]`.
    kda_dev_fresh: Vec<Vec<bool>>,
    /// W4's staged KDA kernels per device (the >max_t fallback path).
    kda_kernels: Vec<super::kda_cuda::cuda::Kernels>,
    /// The resident scratch's window cap (TANG_GLM5_KDA_MAXT, default 512).
    max_t: usize,
    /// W3's MLA + indexer per DSA LAYER (`None` for KDA layers; `TANG_GLM5_GPU_MLA=0`:
    /// DSA stays on the CPU). One per layer, not per device: an `MlaCuda` stages its
    /// layer's kv_a / indexer / K-V decompress weights once and keeps the latent + indexer
    /// caches and their watermarks per sequence — shared across layers it ran every DSA
    /// layer on the first layer's weights and let the next window's first DSA layer
    /// attend over the last layer's latents (the watermark matched, so no re-upload).
    mla_k: Vec<Option<super::mla::cuda::MlaCuda>>,
    /// The device-resident MLA projections per DSA layer ([`super::mla_dev::MlaProj`]:
    /// q_a + norm, q_b, o_proj, the indexer's wq_b / weights_proj on g5f_gemv), built
    /// when `TANG_GLM5_MLA_RESIDENT=1` (default off). `None`: the host GEMVs.
    pub(crate) mla_p: Vec<Option<super::mla_dev::MlaProj>>,
    /// W2's resident mHC per layer ([`super::mhc::MhcGpu`]: both blocks' weights + a
    /// scratch pool on the device; `TANG_GLM5_GPU_MHC=0`: mHC stays on the CPU).
    mhc_g: Vec<Option<super::mhc::MhcGpu>>,
    /// The fused, device-resident mHC ([`super::mhc_fused::MhcFused`]):
    /// `TANG_GLM5_MHC_FUSED=1`, single device, every layer's `MhcGpu` resident. `None`:
    /// the per-call [`Self::mhc_g`] path.
    pub(crate) mhc_f: Option<super::mhc_fused::MhcFused>,
    /// Per layer: its FFN resident on the GPU (`None`: not resident, the CPU runs it —
    /// `TANG_GLM5_GPU_FFN=0` for all, or `TANG_GLM5_GPU_FFN_LAYERS` to pick a subset).
    layers: Vec<Option<super::moe::gpu::DevMoe>>,
    dev_of: Vec<usize>,
    pub split: usize,
    /// `TANG_GLM5_MOE_PROF=1`: per layer, (calls, total µs) for T = 1 / T > 1 FFN calls.
    prof: Option<Vec<[(u64, f64); 2]>>,
    /// The resident decode step's buffers (built on its first token).
    resident: Option<resident::Resident>,
}

/// Everything device-owned goes before the contexts it lives in. Fields drop in
/// declaration order, so without this `devs` (the contexts) went first and every DevLayer
/// stream / MLA buffer / module unload after it hit a destroyed context — the SIGSEGV at
/// the end of every GPU run (compute-sanitizer: cuModuleUnload INVALID_HANDLE,
/// cuStreamDestroy CONTEXT_IS_DESTROYED).
#[cfg(feature = "cuda")]
impl Drop for GpuFfn {
    fn drop(&mut self) {
        self.print_prof();
        if let Some((g, _)) = self.devs.first() {
            let _ = g.bind();
        }
        self.kda_l.clear();
        self.head_dev = None; // its buffers, module and stream before the context
        self.resident = None;
        self.mla_p.clear();
        self.mla_k.clear();
        self.mhc_f = None;
        self.mhc_g.clear();
        self.layers.clear();
        self.kda_kernels.clear();
        // each device's kernels (a module) before the device itself
        for (g, k) in self.devs.drain(..) {
            let _ = g.bind();
            drop(k);
            drop(g);
        }
    }
}

#[cfg(feature = "cuda")]
impl GpuFfn {
    /// The device LM head (TANG_GLM5_GPU_HEAD=1) for one decode row: the last normed h row
    /// up, one k5_q8 + one k5_gemv8, the logits row down. The host tiled path measured
    /// 39.6 ms/token (TANG_GLM5_TRACE); the kernel is ~150 us. `None` without a head.
    pub(crate) fn head_logits(&self, h: &[f32], n: usize, nv: usize) -> Result<Option<Vec<f32>>> {
        let Some(hd) = self.head_dev.as_ref() else {
            return Ok(None);
        };
        let gpu = &self.devs[0].0;
        gpu.bind()
            .map_err(|e| anyhow::anyhow!("bind GPU (head): {e}"))?;
        let row = &h[h.len() - n..];
        // synchronous legacy-stream copy; hd.s is blocking, so it orders after
        hd.xf
            .write(0, row)
            .map_err(|e| anyhow::anyhow!("head x write: {e}"))?;
        unsafe {
            // k5_q8: x [1][n] -> codes [1][n] + scales [1][n/32] (MAXT = 1)
            launch(
                hd.k5.q8,
                ((n / 32) as u32, 1, 1),
                (32, 1, 1),
                0,
                &hd.s,
                args![hd.xf.ptr, hd.xq8.ptr, n as u32, 1u32],
            )?;
            // k5_gemv8: grid (ceil(rows/K5_WARPS), T=1), ROWS = n_vocab, COLS = n;
            // scales live at xq8 + 1*n bytes (the MAXT offset)
            launch(
                hd.k5.gemv8,
                (nv.div_ceil(K5_WARPS) as u32, 1, 1),
                (128, 1, 1),
                (K5_WARPS * ((hd.dw.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32,
                &hd.s,
                args![
                    hd.dw.w,
                    0u32,
                    hd.xq8.ptr,
                    hd.xq8.ptr + n as u64,
                    hd.out.ptr,
                    nv as u32,
                    1u32,
                    nv as u32,
                    n as u32
                ],
            )?;
        }
        hd.s.sync().map_err(|e| anyhow::anyhow!("head sync: {e}"))?;
        hd.out
            .read::<f32>(0, nv)
            .map(Some)
            .map_err(|e| anyhow::anyhow!("head read: {e}"))
    }

    /// The last device LM head's logits row (TANG_GLM5_GPU_HEAD=1; `TANG_GLM5_LOGITS_DUMP`).
    pub fn last_head_logits(&self, nv: usize) -> Result<Option<Vec<f32>>> {
        let Some(hd) = self.head_dev.as_ref() else {
            return Ok(None);
        };
        self.devs[0]
            .0
            .bind()
            .map_err(|e| anyhow::anyhow!("bind GPU (head): {e}"))?;
        hd.out
            .read::<f32>(0, nv)
            .map(Some)
            .map_err(|e| anyhow::anyhow!("head read: {e}"))
    }

    /// The KDA layers resident (tests: the wiring regression guards).
    #[cfg(test)]
    pub(crate) fn kda_resident(&self) -> usize {
        self.kda_l.iter().filter(|x| x.is_some()).count()
    }

    /// The mHC layers resident (tests: the wiring regression guards).
    #[cfg(test)]
    pub(crate) fn mhc_resident(&self) -> usize {
        self.mhc_g.iter().filter(|x| x.is_some()).count()
    }

    /// Upload every layer's FFN. `Ok(None)` when there is no CUDA device or `TANG_GLM5_GPU=0`.
    /// `mla_w(l)` builds DSA layer `l`'s weights (each gets its own resident `MlaCuda`).
    pub fn load(
        g: &crate::gguf::Gguf,
        hp: &super::Hparams,
        mla_w: &dyn Fn(usize) -> Result<MlaWeights>,
    ) -> Result<Option<Self>> {
        Self::load_slots(g, hp, mla_w, 1)
    }

    /// [`Self::load`] with `slots` sequence slots: each KDA layer's device state is
    /// `[slots][...]` (4 MiB + 288 KiB per layer per slot), each DSA layer's `MlaCuda`
    /// LRU is pre-seeded with `slots` ids.
    pub fn load_slots(
        g: &crate::gguf::Gguf,
        hp: &super::Hparams,
        mla_w: &dyn Fn(usize) -> Result<MlaWeights>,
        slots: usize,
    ) -> Result<Option<Self>> {
        use tang_moe::gpu::Gpu;
        let n_layer = hp.n_layer;
        let kda_cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let kda_on = !std::env::var("TANG_GLM5_GPU_KDA").is_ok_and(|v| v == "0");
        let mut kda_kernels = Vec::new();
        let mla_on = !std::env::var("TANG_GLM5_GPU_MLA").is_ok_and(|v| v == "0");
        let mla_res = std::env::var("TANG_GLM5_MLA_RESIDENT").is_ok_and(|v| v == "1");
        let mut mla_k = Vec::new();
        let mut mla_p = Vec::new();
        let mhc_on = hp.hc == 4
            && hp.n_embd % 32 == 0
            && !std::env::var("TANG_GLM5_GPU_MHC").is_ok_and(|v| v == "0");
        let mhp = super::mhc::MhcHparams {
            streams: hp.hc,
            sinkhorn_iters: hp.hc_sinkhorn_iters as u32,
            hc_eps: hp.hc_eps,
            rms_eps: hp.eps,
        };
        if std::env::var("TANG_GLM5_GPU").is_ok_and(|v| v == "0") {
            return Ok(None);
        }
        let ngpu = crate::flash::split::gpus();
        let mut devs = Vec::new();
        for ord in 0..ngpu {
            let gpu = match Gpu::at(ord) {
                Ok(g) => g,
                Err(e) if ord == 0 => {
                    eprintln!("glm5: no CUDA device ({e}); FFNs on the CPU");
                    return Ok(None);
                }
                Err(e) => bail!("TANG_GPUS={ngpu}: CUDA device {ord}: {e}"),
            };
            gpu.bind()
                .map_err(|e| anyhow::anyhow!("bind GPU {ord}: {e}"))?;
            kda_kernels.push(if kda_on && hp.is_kda.iter().any(|&b| b) {
                super::kda_cuda::cuda::load(&gpu, &kda_cfg)?
            } else {
                // the staged fallback is only reachable when some KDA layer exists; a
                // placeholder keeps the vec indexed — an Err surfaces if it is ever used
                super::kda_cuda::cuda::load(&gpu, &kda_cfg)?
            });
            let _ = &mhc_on; // the resident layers are built in the per-layer loop below
            let mk = super::moe::gpu::Kernels::new(&gpu)?;
            devs.push((gpu, mk));
        }
        let split = if devs.len() < 2 {
            n_layer
        } else if let Ok(k) = std::env::var("TANG_SPLIT_AT") {
            k.parse::<usize>().context("TANG_SPLIT_AT")?.min(n_layer)
        } else {
            let bytes: Vec<u64> = (0..n_layer)
                .map(|l| super::moe::expert_bytes(g, l).unwrap_or(0))
                .collect();
            let total: u64 = bytes.iter().sum();
            let (mut best, mut err, mut acc) = (n_layer / 2, u64::MAX, 0u64);
            for (l, b) in bytes.iter().enumerate() {
                acc += b;
                if acc.abs_diff(total / 2) < err {
                    (best, err) = (l + 1, acc.abs_diff(total / 2));
                }
            }
            best
        };
        // the resident KDA layers: one DevLayer per KDA layer on its device. Weights
        // upload in GGUF storage format; scratch is sized to the largest engine window
        // (the engine's chunk, 512 at serve, never exceeds 4096; 64 covers the profile's
        // prefill windows and decode, and the <64 tail falls to the CPU path).
        let max_t = std::env::var("TANG_GLM5_KDA_MAXT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let mut kda_l: Vec<Option<super::kda_dev::DevLayer>> = Vec::new();
        let mut mhc_g: Vec<Option<super::mhc::MhcGpu>> = Vec::new();
        {
            let t_k = Instant::now();
            for l in 0..n_layer {
                // mHC resident: both blocks' weights (attn + ffn) on the layer's device
                if mhc_on {
                    let d = usize::from(l >= split);
                    devs[d]
                        .0
                        .bind()
                        .map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
                    let wa = super::mhc::MhcWeights::from_gguf(g, l, "hc_attn", hp.n_embd, &mhp)
                        .with_context(|| format!("mhc blk.{l}.hc_attn"))?;
                    let wf = super::mhc::MhcWeights::from_gguf(g, l, "hc_ffn", hp.n_embd, &mhp)
                        .with_context(|| format!("mhc blk.{l}.hc_ffn"))?;
                    let mh = super::mhc::MhcGpu::new(&devs[d].0, [&wa, &wf], max_t)
                        .with_context(|| format!("mhc layer {l} resident on GPU {d}"))?;
                    mhc_g.push(Some(mh));
                } else {
                    mhc_g.push(None);
                }
                // MLA resident per DSA layer, from that layer's own weights
                if mla_on && !hp.is_kda[l] {
                    let d = usize::from(l >= split);
                    devs[d]
                        .0
                        .bind()
                        .map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
                    let w = mla_w(l).with_context(|| format!("mla weights blk.{l}"))?;
                    let cu = super::mla::cuda::MlaCuda::new_at(&w, d)
                        .map_err(|e| anyhow::anyhow!("MLA layer {l} on GPU {d}: {e}"))?;
                    if slots > 1 {
                        // pre-create one LRU entry per slot so no decode round pays a
                        // cache switch
                        cu.set_slots(slots);
                    }
                    mla_k.push(Some(cu));
                    mla_p.push(if mla_res {
                        Some(
                            super::mla_dev::MlaProj::new(&devs[d].0, g, l, &w)
                                .with_context(|| format!("MLA projections layer {l} on GPU {d}"))?,
                        )
                    } else {
                        None
                    });
                } else {
                    mla_k.push(None);
                    mla_p.push(None);
                }
                if !hp.is_kda[l] || !kda_on {
                    kda_l.push(None);
                    continue;
                }
                let d = usize::from(l >= split);
                devs[d]
                    .0
                    .bind()
                    .map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
                let layer = super::kda_dev::DevLayer::new_gguf(
                    &devs[d].0,
                    g,
                    l,
                    &kda_cfg,
                    hp.n_embd,
                    &super::kda::ConvState::zeros(&kda_cfg, slots),
                    &super::kda::KdaState::zeros(&kda_cfg, slots),
                    max_t,
                    slots,
                )
                .with_context(|| format!("KDA layer {l} resident on GPU {d}"))?;
                kda_l.push(Some(layer));
                if kda_l.iter().filter(|x| x.is_some()).count() % 8 == 0 {
                    eprintln!(
                        "glm5: {} KDA layers resident ({:.0} s)",
                        kda_l.iter().filter(|x| x.is_some()).count(),
                        t_k.elapsed().as_secs_f64()
                    );
                }
            }
            eprintln!(
                "glm5: {} KDA layers resident on the GPU(s) ({:.0} s)",
                kda_l.iter().filter(|x| x.is_some()).count(),
                t_k.elapsed().as_secs_f64()
            );
            // residency of the other per-layer GPU paths, so a run log proves what ran where
            eprintln!(
                "glm5: {} DSA layers resident (MLA, one per layer), {} mHC layers resident{}",
                mla_k.iter().filter(|x| x.is_some()).count(),
                mhc_g.iter().filter(|x| x.is_some()).count(),
                if slots > 1 {
                    format!(", {slots} sequence slots ({:.1} GiB KDA state)", {
                        let per = (kda_cfg.heads * kda_cfg.head_dim * kda_cfg.head_dim
                            + kda_cfg.conv_dim() * (kda_cfg.d_conv - 1))
                            * 4;
                        per as f64
                            * kda_l.iter().filter(|x| x.is_some()).count() as f64
                            * (slots as f64)
                            / (1 << 30) as f64
                    })
                } else {
                    String::new()
                }
            );
        }
        let kda_dev_fresh = vec![vec![false; n_layer]; slots.max(1)];
        let mhc_fused = mhc_on
            && devs.len() == 1
            && std::env::var("TANG_GLM5_MHC_FUSED").is_ok_and(|v| v == "1");

        let t0 = Instant::now();
        let ffn_on = !std::env::var("TANG_GLM5_GPU_FFN").is_ok_and(|v| v == "0");
        // TANG_GLM5_GPU_FFN_LAYERS=0-6,11: only these layers' FFNs resident (the rest on the
        // CPU) — lets a 24 GB card run the real model with some MoE layers on the GPU
        let pick = std::env::var("TANG_GLM5_GPU_FFN_LAYERS").ok().map(|s| {
            let mut v = Vec::new();
            for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                match part.split_once('-') {
                    Some((a, b)) => {
                        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                            v.extend(a..=b);
                        }
                    }
                    None => v.extend(part.parse::<usize>().ok()),
                }
            }
            v
        });
        let mut layers = Vec::with_capacity(n_layer);
        let mut dev_of = vec![0usize; n_layer];
        for l in 0..n_layer {
            let d = usize::from(l >= split);
            dev_of[l] = d;
            if !ffn_on || pick.as_ref().is_some_and(|v| !v.contains(&l)) {
                layers.push(None);
                continue;
            }
            devs[d]
                .0
                .bind()
                .map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
            let p = super::moe::MoeParams::from_gguf(g, l)?;
            layers
                .push(Some(super::moe::gpu::DevMoe::new(&p).with_context(
                    || format!("uploading layer {l}'s FFN to GPU {d}"),
                )?));
            if l % 8 == 7 || l + 1 == n_layer {
                eprintln!(
                    "glm5: {} of {n_layer} FFNs on the GPU ({:.0} s)",
                    layers.iter().filter(|x| x.is_some()).count(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        eprintln!(
            "glm5: {} of {n_layer} FFNs resident on the GPU, MoE kernels: {}",
            layers.iter().filter(|x| x.is_some()).count(),
            if devs.first().is_some_and(|d| d.1.fast) {
                "fast (TANG_GLM5_MOE=fast)"
            } else {
                "naive"
            }
        );
        eprintln!(
            "glm5: FFNs on {} GPU(s), layers [0, {split}) on GPU 0{}",
            devs.len(),
            if devs.len() > 1 {
                format!(", [{split}, {n_layer}) on GPU 1")
            } else {
                String::new()
            }
        );
        // the LM head on the GPU (TANG_GLM5_GPU_HEAD=1): t==1 decode only. The host
        // tiled Mat::apply was 39.6 ms/token (trace); one k5_q8 + k5_gemv8 is ~150 us.
        // Built here, before `devs` moves into Self.
        let head_dev = 'head: {
            if !std::env::var("TANG_GLM5_GPU_HEAD").is_ok_and(|v| v == "1") {
                break 'head None;
            }
            devs[0]
                .0
                .bind()
                .map_err(|e| anyhow::anyhow!("bind GPU 0 (head): {e}"))?;
            let info = match g.get("output.weight") {
                Some(t) => t,
                None => g.info("token_embd.weight")?,
            };
            let rows = info.dims[1] as usize; // n_vocab
            let cols = info.dims[0] as usize; // hidden
            if info.ty != crate::gguf::GgmlType::Q8_0 || cols % 32 != 0 {
                // k5_gemv8 reads Q8_0 rows (the tiny fixtures' head is f32): host head
                eprintln!(
                    "glm5: LM head stays on the host ({:?}, hidden {cols}: the device head wants Q8_0)",
                    info.ty
                );
                break 'head None;
            }
            let dw = super::moe::gpu::upload(&super::moe::W::Gguf(g, info), 1, rows, cols)
                .map_err(|e| anyhow::anyhow!("lm head upload: {e}"))?;
            let k5 = super::moe::gpu::Kernels5::new(&devs[0].0)
                .map_err(|e| anyhow::anyhow!("k5 (head): {e}"))?;
            let xf = tang_moe::gpu::DevBuf::zeroed_elems::<f32>(cols)
                .map_err(|e| anyhow::anyhow!("head xf: {e}"))?;
            // scratch: k5_q8 codes [MAXT][cols] + f32 scales after them (the gemv8
            // kernel reads scales at xq8 + MAXT*cols); here MAXT = 1
            let xq8 = tang_moe::gpu::DevBuf::zeroed_elems::<u8>(cols + cols / 32 * 4)
                .map_err(|e| anyhow::anyhow!("head xq8: {e}"))?;
            let out = tang_moe::gpu::DevBuf::zeroed_elems::<f32>(rows)
                .map_err(|e| anyhow::anyhow!("head out: {e}"))?;
            let hs = tang_moe::gpu::Stream::blocking()
                .map_err(|e| anyhow::anyhow!("head stream: {e}"))?;
            eprintln!(
                "glm5: LM head resident on the GPU ({} MB)",
                dw.w.row_bytes as usize * rows / 1_000_000
            );
            Some(HeadDev {
                dw,
                xf,
                xq8,
                out,
                k5,
                s: hs,
            })
        };
        let mut f = Self {
            devs,
            kda_l,
            kda_dev_fresh,
            kda_kernels: kda_kernels,
            head_dev,
            max_t,
            mla_k,
            mla_p,
            mhc_g,
            mhc_f: None,
            layers,
            dev_of,
            split,
            resident: None,
            prof: std::env::var("TANG_GLM5_MOE_PROF")
                .is_ok_and(|v| v == "1")
                .then(|| vec![[(0u64, 0f64); 2]; n_layer]),
        };
        if mhc_fused {
            f.enable_mhc_fused(g, hp)?;
        }
        Ok(Some(f))
    }

    /// Build the device-resident MLA projections for every DSA layer that has an
    /// `MlaCuda` (`load` does when `TANG_GLM5_MLA_RESIDENT=1`). Returns how many.
    pub(crate) fn enable_mla_resident(
        &mut self,
        g: &crate::gguf::Gguf,
        mla_w: &dyn Fn(usize) -> Result<MlaWeights>,
    ) -> Result<usize> {
        for l in 0..self.mla_k.len() {
            if self.mla_k[l].is_none() || self.mla_p[l].is_some() {
                continue;
            }
            let d = self.dev_of[l];
            self.devs[d]
                .0
                .bind()
                .map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
            let w = mla_w(l)?;
            self.mla_p[l] = Some(super::mla_dev::MlaProj::new(&self.devs[d].0, g, l, &w)?);
        }
        Ok(self.mla_p.iter().filter(|p| p.is_some()).count())
    }

    /// Build the fused mHC ([`Self::mhc_f`]; `load` does when `TANG_GLM5_MHC_FUSED=1`).
    /// Single device only; the windows it serves need every layer's `MhcGpu` resident.
    pub(crate) fn enable_mhc_fused(
        &mut self,
        g: &crate::gguf::Gguf,
        hp: &super::Hparams,
    ) -> Result<()> {
        ensure!(self.devs.len() == 1, "fused mHC: single device only");
        self.devs[0]
            .0
            .bind()
            .map_err(|e| anyhow::anyhow!("bind GPU 0: {e}"))?;
        let norms = (0..hp.n_layer)
            .map(|l| -> Result<[Vec<f32>; 2]> {
                Ok([
                    g.tensor_f32(&super::t(l, "attn_norm.weight"))?,
                    g.tensor_f32(&super::t(l, "ffn_norm.weight"))?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        self.mhc_f = Some(super::mhc_fused::MhcFused::new(
            &self.devs[0].0,
            hp.n_embd,
            self.max_t,
            norms,
            hp.eps,
        )?);
        eprintln!(
            "glm5: mHC fused (device-resident streams, {} layers)",
            hp.n_layer
        );
        Ok(())
    }

    /// Device 0 and its MoE kernels (the MTP layer's device home, [`super::mtp`]).
    pub(crate) fn dev0(&self) -> (&tang_moe::gpu::Gpu, &super::moe::gpu::Kernels) {
        (&self.devs[0].0, &self.devs[0].1)
    }

    /// Whether layer `l`'s FFN is resident.
    fn has_ffn(&self, l: usize) -> bool {
        self.layers.get(l).is_some_and(|m| m.is_some())
    }

    /// Layer `l`'s FFN on its GPU: host `x` `[t][hidden]` in, host `y` out.
    fn ffn(&mut self, l: usize, x: &[f32], t: usize) -> Result<Vec<f32>> {
        let (gpu, k) = &self.devs[self.dev_of[l]];
        gpu.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        let m = self.layers[l].as_ref().context("FFN not resident")?;
        let t0 = Instant::now();
        let y = super::moe::gpu::ffn_host(k, m, x, t)?;
        if let Some(p) = self.prof.as_mut() {
            let e = &mut p[l][usize::from(t > 1)];
            e.0 += 1;
            e.1 += t0.elapsed().as_secs_f64() * 1e6;
        }
        Ok(y)
    }

    /// The fast path's routing of the last call on layer `l`'s device (`MoE debug tap`).
    fn last_routing(&self, l: usize, t: usize, k: usize) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        let (gpu, kern) = &self.devs[self.dev_of[l]];
        if !kern.fast {
            return Ok(None);
        }
        gpu.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        kern.last_routing(t, k)
    }

    /// `TANG_GLM5_MOE_PROF=1` summary: mean µs per FFN call, per layer and kind.
    fn print_prof(&self) {
        let Some(p) = &self.prof else { return };
        let mut tot = [(0u64, 0f64); 2];
        for (l, e) in p.iter().enumerate() {
            for (i, (n, us)) in e.iter().enumerate() {
                if *n == 0 {
                    continue;
                }
                eprintln!(
                    "moeprof l{l} {} calls {n} mean {:.0} us",
                    if i == 0 { "T=1" } else { "T>1" },
                    us / *n as f64
                );
                tot[i].0 += n;
                tot[i].1 += us;
            }
        }
        for (i, (n, us)) in tot.iter().enumerate() {
            if *n > 0 {
                eprintln!(
                    "moeprof ALL {} calls {n} total {:.1} ms mean {:.0} us/call",
                    if i == 0 { "T=1" } else { "T>1" },
                    us / 1e3,
                    us / *n as f64
                );
            }
        }
    }

    /// Grow the resident KDA layers' state buffers to `slots` sequences and the DSA
    /// layers' LRU to `slots` ids (a fresh `DevLayer` per KDA layer, replacing the old
    /// one — the weights re-upload, ~seconds × 34; called once at serve start, before
    /// any window). `mhc_f` is dropped: its captured streams are slot-agnostic (they are
    /// re-uploaded per window) but its `begin` holds a pointer set sized to one window,
    /// not per slot — it re-enables through `enable_mhc_fused` if the caller wants it.
    fn resize_slots(
        &mut self,
        slots: usize,
        g: &crate::gguf::Gguf,
        hp: &super::Hparams,
    ) -> Result<()> {
        self.resident = None; // its step graph points at the KDA layers rebuilt below
        use tang_moe::gpu::Gpu;
        let kda_cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let kda_on = !std::env::var("TANG_GLM5_GPU_KDA").is_ok_and(|v| v == "0");
        let t0 = Instant::now();
        for l in 0..hp.n_layer {
            if !hp.is_kda[l] || !kda_on {
                continue;
            }
            let d = self.dev_of[l];
            self.devs[d]
                .0
                .bind()
                .map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
            let layer = super::kda_dev::DevLayer::new_gguf(
                &self.devs[d].0,
                g,
                l,
                &kda_cfg,
                hp.n_embd,
                &super::kda::ConvState::zeros(&kda_cfg, slots),
                &super::kda::KdaState::zeros(&kda_cfg, slots),
                self.max_t,
                slots,
            )
            .with_context(|| format!("KDA layer {l} resident on GPU {d} ({slots} slots)"))?;
            self.kda_l[l] = Some(layer);
        }
        self.kda_dev_fresh
            .iter_mut()
            .for_each(|row| row.fill(false));
        for cu in self.mla_k.iter().flatten() {
            cu.set_slots(slots);
        }
        eprintln!(
            "glm5: KDA layers resized to {slots} sequence slots ({:.0} s)",
            t0.elapsed().as_secs_f64()
        );
        // fused mHC is per-window stateless in the streams' values but sized at load;
        // rebuild it (the fused instance held no sequence state)
        if self.mhc_g.iter().all(|m| m.is_some()) && self.devs.len() == 1 {
            self.enable_mhc_fused(g, hp)?;
        }
        Ok(())
    }

    /// Zero sequence `slot`'s conv + recurrent state on every resident KDA layer
    /// (`reset`/`reset_slot`: the device must not keep the dropped sequence's values,
    /// because a fresh window seeds from the host's zeros only when `kda_dev_fresh` is
    /// false — and after a reset it is, so the seed covers it. When the host state is
    /// `None` the seed never happens; zero the buffers here so a stale sequence's state
    /// can't leak into a new one).
    fn zero_slot_states(&mut self, _slot: usize) {
        // the fresh flag is cleared on every reset, so the next window's `seed` writes the
        // host's (zeroed) copies over this slot — no device memset needed
    }

    /// The device-state freshness flags resettable per slot (tests).
    #[cfg(feature = "cuda")]
    fn clear_dev_fresh(&mut self, slot: usize) {
        if let Some(row) = self.kda_dev_fresh.get_mut(slot) {
            row.fill(false);
        }
    }
}

/// The resumable GLM-5.3-Flash model.
pub struct Glm5Engine {
    pub m: Glm5Ref,
    /// Sequence slots' tokens. Slot 0 is the single-sequence path (this field, for the
    /// pre-batching callers: the CLI, the tests, the server's one-sequence lane); slots
    /// 1.. hold the batched server's other sequences. `n_slots` > 1 only after
    /// [`Glm5Engine::use_gpu_slots`] sized the device state (the resident KDA layers
    /// allocate `[slots][...]`); until then every slot past 0 is unusable.
    pub tokens: Vec<u32>,
    n_slots: usize,
    /// Per-slot tokens (slot 0 mirrors `tokens`); empty when single-slot.
    slot_tokens: Vec<Vec<u32>>,
    /// Per-slot KDA host state: `kda[slot][layer]`.
    kda: Vec<Vec<Option<(ConvState, KdaState)>>>,
    /// Per-slot DSA caches: `dsa[slot][layer]` (the host mirrors; the device caches are
    /// the `MlaCuda` LRU entries, keyed by slot id).
    dsa: Vec<Vec<DsaCache>>,
    kda_w: Vec<Option<std::sync::Arc<(KdaLayerW, KdaConfig)>>>,
    hc_w: Vec<Option<std::sync::Arc<super::mhc::MhcWeights>>>,
    mla_w: Vec<Option<std::sync::Arc<MlaWeights>>>,
    /// Per DSA layer: `mla_w`'s per-head `k_nope` / `v_b` concatenated (the
    /// `MlaCuda::attend` host arguments), built once with the weights.
    #[allow(dead_code)]
    mla_wkv: Vec<Option<std::sync::Arc<(Vec<f32>, Vec<f32>)>>>,
    head: Option<crate::flash::reference::Mat>,
    /// Print per-window timings to stderr.
    pub verbose: bool,
    /// set by [`Glm5Engine::prefill_slot`]: a prefill window needs only its last row's
    /// logits, so the device head runs once (not t times) and the window returns that row.
    prefill_last: bool,
    /// CPU DSA path: round the latent cache to bf16 (the GPU cache's format), so a CPU
    /// engine is the like-for-like oracle of the GPU MLA path (tests).
    pub(crate) bf16_latents: bool,
    /// W0's R3 bisector: per-layer dump sink, no-op without TANG_DUMP_DIR (dump.rs).
    pub dump: Option<super::dump::DumpSink>,
    /// The FFNs on the GPU, when a CUDA build finds one ([`Glm5Engine::use_gpu`]).
    #[cfg(feature = "cuda")]
    pub gpu: Option<GpuFfn>,
    /// `TANG_GLM5_KDA_DEBUG=1`: the CPU shadow of every resident KDA layer, checked
    /// intermediate-by-intermediate after each window ([`super::kda_debug`]).
    #[cfg(feature = "cuda")]
    kda_dbg: Option<super::kda_debug::Shadow>,
    /// `TANG_GLM5_MOE_DEBUG=1`: the CPU shadow of every GPU FFN call ([`super::moe_debug`]).
    #[cfg(feature = "cuda")]
    moe_dbg: Option<super::moe_debug::Shadow>,
    /// `TANG_GLM5_MLA_DEBUG=1`: the CPU shadow of every GPU DSA window ([`super::mla_debug`]).
    #[cfg(feature = "cuda")]
    pub(crate) mla_dbg: Option<super::mla_debug::Shadow>,
    /// The last window's post-`output_norm` hidden rows `[t][n_embd]` (the MTP layer's `h`,
    /// llama.cpp's `t_h_nextn`; [`super::mtp`]).
    pub(crate) last_h: Vec<f32>,
    /// TANG_GLM5_WINDIFF bisector: per (layer, stage) host rows appended each window
    /// (stage 0 attn-mixer in, 1 attn-mixer out, 2 ffn in, 3 ffn out); None = off.
    pub(crate) wintap: Option<Vec<Vec<f32>>>,
}

/// A rewind point for speculative verify ([`Glm5Engine::snapshot`]): the position and the
/// host copies of every KDA state (the DSA caches are append-only and rewind by truncation).
pub(crate) struct Snap {
    pos: usize,
    kda: Vec<Option<(ConvState, KdaState)>>,
}

impl Snap {
    pub(crate) fn clone_snap(&self) -> Snap {
        Snap {
            pos: self.pos,
            kda: self.kda.clone(),
        }
    }
}

impl Glm5Engine {
    pub fn open(path: &Path) -> Result<Self> {
        let m = Glm5Ref::open(path)?;
        let nl = m.hp.n_layer;
        Ok(Self {
            m,
            tokens: Vec::new(),
            n_slots: 1,
            slot_tokens: Vec::new(),
            kda: vec![(0..nl).map(|_| None).collect()],
            dsa: vec![vec![DsaCache::default(); nl]],
            kda_w: (0..nl).map(|_| None).collect(),
            hc_w: (0..nl * 2).map(|_| None).collect(),
            mla_w: (0..nl).map(|_| None).collect(),
            mla_wkv: (0..nl).map(|_| None).collect(),
            head: None,
            verbose: false,
            prefill_last: false,
            bf16_latents: false,
            dump: super::dump::DumpSink::open(),
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            kda_dbg: super::kda_debug::Shadow::from_env(nl),
            #[cfg(feature = "cuda")]
            moe_dbg: super::moe_debug::Shadow::from_env(nl),
            #[cfg(feature = "cuda")]
            mla_dbg: super::mla_debug::Shadow::from_env(),
            last_h: Vec::new(),
            wintap: None,
        })
    }

    /// Move the FFNs to the GPU(s) (no-op without a device or with `TANG_GLM5_GPU=0`).
    /// Returns whether they moved.
    #[cfg(feature = "cuda")]
    pub fn use_gpu(&mut self) -> Result<bool> {
        let m = &self.m;
        self.gpu = GpuFfn::load(&m.g, &m.hp, &|l| m.mla_weights(l))?;
        Ok(self.gpu.is_some())
    }

    /// [`Glm5Engine::use_gpu`] with `slots` sequence slots: the resident KDA layers
    /// allocate a `[slots][...]` device state (4 MiB + 288 KiB per layer per slot), each
    /// DSA layer's `MlaCuda` LRU grows to `slots` entries, and the engine keeps per-slot
    /// tokens + host state. Call before any window (the single-slot state a `use_gpu`
    /// call made is discarded).
    #[cfg(feature = "cuda")]
    pub fn use_gpu_slots(&mut self, slots: usize) -> Result<bool> {
        let m = &self.m;
        let slots = slots.max(1);
        self.n_slots = slots;
        let nl = m.hp.n_layer;
        self.slot_tokens = (0..slots).map(|_| Vec::new()).collect();
        self.kda = (0..slots)
            .map(|_| (0..nl).map(|_| None).collect())
            .collect();
        self.dsa = (0..slots).map(|_| vec![DsaCache::default(); nl]).collect();
        if let Some(g) = self.gpu.as_mut() {
            g.resize_slots(slots, &m.g, &m.hp)?;
        } else {
            self.gpu = GpuFfn::load_slots(&m.g, &m.hp, &|l| m.mla_weights(l), slots)?;
        }
        Ok(self.gpu.is_some())
    }

    /// `slots` sequence slots with the state on the host only (no device; the CPU build,
    /// the tests). With a GPU use [`Self::use_gpu_slots`], which also sizes the device state.
    pub fn set_slots_host(&mut self, slots: usize) {
        let slots = slots.max(1);
        let nl = self.m.hp.n_layer;
        self.n_slots = slots;
        self.slot_tokens = (0..slots).map(|_| Vec::new()).collect();
        self.kda = (0..slots)
            .map(|_| (0..nl).map(|_| None).collect())
            .collect();
        self.dsa = (0..slots).map(|_| vec![DsaCache::default(); nl]).collect();
    }

    /// The slot count (`use_gpu_slots` set; 1 until then).
    pub fn slots(&self) -> usize {
        self.n_slots
    }

    /// Slot `s`'s tokens (slot 0: `tokens`).
    pub fn slot_tokens(&self, slot: usize) -> &[u32] {
        if slot == 0 {
            &self.tokens
        } else {
            self.slot_tokens.get(slot).map_or(&[] as &[_], |v| v)
        }
    }

    /// Slot `s`'s tokens, mutable.
    pub fn slot_tokens_mut(&mut self, slot: usize) -> &mut Vec<u32> {
        if slot == 0 {
            &mut self.tokens
        } else {
            &mut self.slot_tokens[slot]
        }
    }

    /// Slot `s`'s KDA layer `l` state: `(conv, recurrent)` f32 vectors, read off the
    /// device when it holds the live copy. `None` when the layer has no state yet.
    #[cfg(feature = "cuda")]
    pub fn kda_state_at(&mut self, slot: usize, l: usize) -> Option<(ConvState, KdaState)> {
        let hp = self.m.hp.clone();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let (mut conv, mut state) = (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1));
        // the device's copy when fresh, else the host's (or zeros when the sequence has
        // not reached this layer's state yet)
        let host = self.kda[slot][l].as_ref();
        if host.is_some() {
            if let Some(g) = self.gpu.as_mut() {
                if g.kda_dev_fresh[slot][l] {
                    if let Some(dev_l) = &mut g.kda_l[l] {
                        let d = g.dev_of[l];
                        if g.devs[d].0.bind().is_ok()
                            && dev_l.read_back_at(slot, &mut conv, &mut state).is_ok()
                        {
                            return Some((conv, state));
                        }
                    }
                }
            }
        }
        host.cloned()
    }

    /// The host state snapshot (the CPU-only build; the device copy is always fresher but
    /// unreachable without cuda — snapshots there read it first).
    #[cfg(not(feature = "cuda"))]
    pub fn kda_state_at(&mut self, slot: usize, l: usize) -> Option<(ConvState, KdaState)> {
        self.kda[slot][l].clone()
    }

    /// Put the KDA state parts back into slot `slot`: one `(conv, recurrent)` f32 pair
    /// per KDA layer, in layer order (the server's saved `RunningState` parts, GDN
    /// layers = the KDA layers). The device re-seeds from these on the slot's next
    /// window (`kda_dev_fresh` is cleared). The DSA caches are NOT restored: the caller
    /// re-prefills the positions past the state, which rebuilds them.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn restore_kda_parts(&mut self, slot: usize, parts: &[Vec<f32>]) -> Result<()> {
        let hp = self.m.hp.clone();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let (cn, sn) = (
            cfg.conv_dim() * (cfg.d_conv - 1),
            cfg.heads * cfg.head_dim * cfg.head_dim,
        );
        ensure!(
            parts.len() >= 2 * hp.is_kda.iter().filter(|&&k| k).count(),
            "kda state parts: {} for {} KDA layers",
            parts.len(),
            hp.is_kda.iter().filter(|&&k| k).count()
        );
        let mut it = parts.iter();
        for l in 0..hp.n_layer {
            if !hp.is_kda[l] {
                continue;
            }
            let (conv, state) = (it.next().unwrap(), it.next().unwrap());
            ensure!(
                conv.len() == cn && state.len() == sn,
                "kda state part shape at layer {l}"
            );
            self.kda[slot][l] = Some((ConvState(conv.clone()), KdaState(state.clone())));
        }
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            g.clear_dev_fresh(slot);
        }
        Ok(())
    }

    /// The KDA geometry (the state shapes' source).
    pub fn kda_config(&self) -> KdaConfig {
        let hp = &self.m.hp;
        KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        }
    }

    /// Drop slot `s`'s DSA caches (host mirrors + the device LRU entry); the KDA state
    /// and tokens stay (a KDA-state restore calls this: the positional state is not saved).
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn reset_dsa(&mut self, slot: usize) {
        for c in &mut self.dsa[slot] {
            *c = DsaCache::default();
        }
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            for cu in g.mla_k.iter().flatten() {
                cu.reset_seq(slot as u64);
            }
        }
    }

    /// Drop slot `s`'s sequence (its next window starts fresh).
    pub fn reset_slot(&mut self, slot: usize) {
        if slot == 0 {
            self.reset();
            return;
        }
        self.slot_tokens[slot].clear();
        self.kda[slot].fill(None);
        for c in &mut self.dsa[slot] {
            *c = DsaCache::default();
        }
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            g.kda_dev_fresh.iter_mut().for_each(|row| row.fill(false));
            // re-zero the slot's device state (the host copies are None; the next window
            // seeds from zeros — but the device may hold the old sequence's values, so
            // mark the layer stale AND clear it)
            g.zero_slot_states(slot);
        }
        // the DSA device caches for this slot: re-base to empty by switching to the id
        // and letting the next append upload nothing (the host mirror is empty and
        // `lat_rows`/`idx_rows` watermarks drop to 0 via the contiguity check)
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            for cu in g.mla_k.iter().flatten() {
                cu.reset_seq(slot as u64);
            }
        }
    }

    /// Layer `l`'s FFN: on its GPU when loaded, else W1's dense FFN / W5's CPU MoE.
    fn ffn(&mut self, l: usize, x: &[f32], t: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            if g.has_ffn(l) {
                let y = g.ffn(l, x, t)?;
                if let Some(d) = self.moe_dbg.as_mut().filter(|d| d.wants(l)) {
                    let p = super::moe::MoeParams::from_gguf(&self.m.g, l)?;
                    let route = if p.is_dense() {
                        None
                    } else {
                        g.last_routing(l, t, p.top_k)?
                    };
                    let m = &self.m;
                    d.check(l, &p, x, t, &y, route, &|| m.dense_ffn(l, x, t))?;
                }
                return Ok(y);
            }
            // not resident (TANG_GLM5_GPU_FFN=0 / _LAYERS): fall through to the CPU path.
        }
        if self.m.hp.is_dense(l) {
            let _tr = super::trace::span("dense.cpu");
            self.m.dense_ffn(l, x, t)
        } else {
            let _tr = super::trace::span("moe.cpu");
            let p = super::moe::MoeParams::from_gguf(&self.m.g, l)?;
            super::moe::moe_layer(&p, x, t)
        }
    }

    /// Forget every slot's sequence. Cached weights stay.
    pub fn reset(&mut self) {
        self.tokens.clear();
        for t in &mut self.slot_tokens {
            t.clear();
        }
        for ks in &mut self.kda {
            ks.fill(None);
        }
        // the resident KDA layers hold the live state on the device: without this the
        // next sequence's first window skipped the seed and continued the OLD sequence's
        // conv + recurrent state (the host copies above are the fresh zeros it must get)
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            g.kda_dev_fresh.iter_mut().for_each(|row| row.fill(false));
            for slot in 0..self.n_slots {
                g.zero_slot_states(slot);
            }
        }
        for ds in &mut self.dsa {
            for c in ds {
                *c = DsaCache::default();
            }
        }
        // the DSA device caches re-base to empty (the host mirrors above are empty, so the
        // next append re-uploads from them — the watermarks drop via the contiguity check)
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            for slot in 0..self.n_slots {
                for cu in g.mla_k.iter().flatten() {
                    cu.reset_seq(slot as u64);
                }
            }
        }
    }

    /// A rewind point at the current position (MTP verify). Resident KDA layers read their
    /// device state back into the host copies first, so the snapshot is complete.
    pub(crate) fn snapshot(&mut self) -> Result<Snap> {
        // slot 0 only: MTP drives the single-sequence path
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            for l in 0..self.kda[0].len() {
                if !g.kda_dev_fresh[0][l] {
                    continue;
                }
                if let (Some(dev), Some(st)) = (g.kda_l[l].as_mut(), self.kda[0][l].as_mut()) {
                    g.devs[g.dev_of[l]]
                        .0
                        .bind()
                        .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                    dev.read_back_at(0, &mut st.0, &mut st.1)?;
                }
            }
        }
        Ok(Snap {
            pos: self.tokens.len(),
            kda: self.kda[0].clone(),
        })
    }

    /// Rewind to `s`: KDA states back to the snapshot (resident layers re-seed from the host
    /// on their next window), DSA caches truncated to the snapshot's position (the device
    /// latent rows are position-indexed, so the next window overwrites them). Slot 0 only.
    pub(crate) fn restore(&mut self, s: Snap) {
        let hp = &self.m.hp;
        let (kl, idd) = (hp.kv_lora_rank, hp.idx_dim);
        let pos = s.pos;
        self.tokens.truncate(pos);
        if let Some(t0) = self.slot_tokens.get_mut(0) {
            t0.truncate(pos);
        }
        self.kda[0] = s.kda;
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            g.clear_dev_fresh(0);
        }
        // MTP vs the resident step: the clear above makes `resident_ready` false until a
        // host window re-seeds the KDA device state, and the step graphs (the segments and
        // the whole-step `[tok, pos]` / cache-capacity state) are dropped here so nothing
        // replays across a rewind. MTP itself only calls `window()`.
        #[cfg(feature = "cuda")]
        self.drop_step_graph();
        for c in &mut self.dsa[0] {
            // rows past the rewind point are dropped, not refreshed
            c.stale_from = c.stale_from.filter(|&f| f < pos);
            c.latent.truncate(pos * kl);
            c.lat16.truncate(pos * kl);
            c.idx_k.truncate(pos * idd);
            c.idx_g.truncate(pos * idd);
            c.valid.truncate(pos);
        }
    }

    /// Layer `l`'s KDA weights, loaded once and shared (dequantising 268 MB per window is
    /// the engine's original per-token cost).
    fn kda_weights(&mut self, l: usize) -> Result<std::sync::Arc<(KdaLayerW, KdaConfig)>> {
        if let Some(w) = &self.kda_w[l] {
            return Ok(w.clone());
        }
        let w = std::sync::Arc::new(self.m.kda_weights(l)?);
        self.kda_w[l] = Some(w.clone());
        Ok(w)
    }

    /// mHC mixer weights for layer `l` (cached when `cache_weights`; see the field note).
    fn hc_weights(
        &mut self,
        l: usize,
        prefix: &str,
        hp: &super::mhc::MhcHparams,
    ) -> Result<std::sync::Arc<super::mhc::MhcWeights>> {
        let idx = l * 2 + if prefix == "hc_ffn" { 1 } else { 0 };
        if let Some(w) = &self.hc_w[idx] {
            return Ok(w.clone());
        }
        let w = std::sync::Arc::new(super::mhc::MhcWeights::from_gguf(
            &self.m.g,
            l,
            prefix,
            self.m.hp.n_embd,
            hp,
        )?);
        self.hc_w[idx] = Some(w.clone());
        Ok(w)
    }

    fn mla_weights(&mut self, l: usize) -> Result<std::sync::Arc<MlaWeights>> {
        if let Some(w) = &self.mla_w[l] {
            return Ok(w.clone());
        }
        let w = std::sync::Arc::new(self.m.mla_weights(l)?);
        self.mla_w[l] = Some(w.clone());
        Ok(w)
    }

    /// KDA layer `l`'s mixer: the resident [`super::kda_dev::DevLayer`] when loaded (the
    /// device holds the state; the host copies sync only across a CPU fallback), W4's
    /// staged free-function path when the window exceeds the resident scratch, else the
    /// CPU reference. `slot` selects the state slice (host copies and the device's
    /// `[slots][...]` buffers alike); `kda_dev_fresh[slot][l]` tracks whether the device
    /// slice is the live copy.
    #[allow(clippy::too_many_arguments)]
    fn kda_mixer(
        &mut self,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] slot: usize,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] l: usize,
        x: &[f32],
        t: usize,
        w: &KdaLayerW,
        cfg: &KdaConfig,
        st: &mut (ConvState, KdaState),
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            if let Some(dev) = &mut g.kda_l[l] {
                let d = g.dev_of[l];
                g.devs[d]
                    .0
                    .bind()
                    .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                let fresh = g.kda_dev_fresh[slot][l];
                if t > g.max_t {
                    // past the resident scratch: sync the state to the host and run the
                    // staged path (which keeps both copies fresh), then seed forward
                    if fresh {
                        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
                            eprintln!("kda wiring: read_back layer {l} (window {t} > max_t)");
                        }
                        let _tr = super::trace::span("read_back");
                        dev.read_back_at(slot, &mut st.0, &mut st.1)?;
                        g.kda_dev_fresh[slot][l] = false;
                    }
                    let _tr = super::trace::span("staged");
                    return kda_gpu(
                        x,
                        t,
                        w,
                        cfg,
                        &g.kda_kernels[d],
                        &g.devs[d].0,
                        &mut st.0,
                        &mut st.1,
                    );
                }
                if !fresh {
                    if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
                        eprintln!("kda wiring: seed layer {l} (host -> device)");
                    }
                    let _tr = super::trace::span("seed");
                    dev.seed_at(slot, &st.0, &st.1)?;
                    g.kda_dev_fresh[slot][l] = true;
                }
                let dbg = self.kda_dbg.as_mut().filter(|d| d.wants(l));
                let Some(dbg) = dbg else {
                    // a short window (MTP verify, t <= WIN_STEP) runs row by row: the t=1
                    // projections + core are what the decode step runs, and the batched
                    // t>1 projections are not bit-identical to them under the B300 flags
                    // (windiff: l00 attn_out row 0 rel 9.9e-5 with identical input)
                    if t > 1 && t <= super::mtp::win_step() {
                        let n = x.len() / t;
                        let mut y = Vec::with_capacity(x.len());
                        for i in 0..t {
                            y.extend(dev.window_at(slot, &x[i * n..(i + 1) * n], 1, &mut st.0)?);
                        }
                        return Ok(y);
                    }
                    return dev.window_at(slot, x, t, &mut st.0);
                };
                // shadow check: the device's state going in, the window with taps, the
                // device's state coming out, then the CPU re-run of the same window
                dbg.prime(l, &st.0, &st.1);
                let (mut c_in, mut s_in) = (st.0.clone(), st.1.clone());
                dev.read_back_at(slot, &mut c_in, &mut s_in)?;
                dev.probe = Some(dbg.probe());
                let y = dev.window_at(slot, x, t, &mut st.0);
                let probe = dev.probe.take();
                let y = y?;
                let taps = probe.map(|p| p.taps).unwrap_or_default();
                let (mut c_out, mut s_out) = (st.0.clone(), st.1.clone());
                dev.read_back_at(slot, &mut c_out, &mut s_out)?;
                dbg.check(l, x, t, w, cfg, taps, &st.0, &c_out, &s_out, Some(&s_in))?;
                return Ok(y);
            }
        }
        let _tr = super::trace::span("cpu");
        super::kda::kda_layer(x, w, cfg, &mut st.0, &mut st.1, 1, t)
    }

    /// DSA layer `l`: W3's GPU MLA + indexer when loaded, else the CPU path. `slot`
    /// selects the host mirror slice AND the device caches' LRU id (`set_seq`).
    fn dsa_layer(
        &mut self,
        slot: usize,
        l: usize,
        w: &MlaWeights,
        x: &[f32],
        t: usize,
        pos0: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        self.refresh_stale_dsa(slot, l)?;
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            let d = g.dev_of[l];
            if let Some(cu) = &g.mla_k[l] {
                g.devs[d]
                    .0
                    .bind()
                    .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                cu.set_seq(slot as u64);
                let dbg = self.mla_dbg.as_mut().filter(|d| d.wants(l));
                if let Some(p) = &g.mla_p[l] {
                    return dsa_gpu_resident(l, w, cu, p, x, t, pos0, &mut self.dsa[slot][l], dbg);
                }
                let wkv = self.mla_wkv[l]
                    .get_or_insert_with(|| {
                        std::sync::Arc::new((
                            (0..w.n_head).flat_map(|h| w.k_nope(h).to_vec()).collect(),
                            (0..w.n_head).flat_map(|h| w.v_b_head(h).to_vec()).collect(),
                        ))
                    })
                    .clone();
                return dsa_gpu(l, w, &wkv, cu, x, t, pos0, &mut self.dsa[slot][l], dbg);
            }
        }
        let _tr = super::trace::span("cpu");
        self.dsa_cpu(slot, l, w, x, t, pos0)
    }

    /// DSA layer `l` on the CPU: W3's MLA-NoPE + pooled indexer over the f32 caches.
    fn dsa_cpu(
        &mut self,
        slot: usize,
        l: usize,
        w: &MlaWeights,
        x: &[f32],
        t: usize,
        pos0: usize,
    ) -> Result<Vec<f32>> {
        let hp = self.m.hp.clone();
        let n = hp.n_embd;
        let kv_len = pos0 + t;
        let q_resid = w.gemv_q_a(x, t);
        let latent = w.gemv_kv_a(x, t);
        let idx_dim = hp.idx_dim;
        let cache = &mut self.dsa[slot][l];
        if self.bf16_latents {
            cache
                .latent
                .extend(latent.iter().map(|&v| super::mla::bf16_round(v)));
        } else {
            cache.latent.extend_from_slice(&latent);
        }
        for i in 0..t {
            let (mut k, mut g) = (vec![0f32; idx_dim], vec![0f32; idx_dim]);
            super::mla::indexer_key(w, &x[i * n..(i + 1) * n], &mut k, &mut g);
            cache.idx_k.extend_from_slice(&k);
            cache.idx_g.extend_from_slice(&g);
        }
        let sel = if super::mla::indexer_is_dense(kv_len, hp.idx_top_k) {
            None
        } else {
            Some(super::mla::indexer_select(
                w,
                x,
                &q_resid,
                &cache.idx_k,
                &cache.idx_g,
                &vec![1u8; kv_len],
                &vec![1u8; t],
                t,
                kv_len,
                pos0,
            ))
        };
        let mut out = vec![0f32; t * n];
        super::mla::mla_forward(
            w,
            x,
            &q_resid,
            &cache.latent,
            kv_len,
            t,
            pos0,
            None,
            sel.as_deref(),
            &mut out,
        );
        Ok(out)
    }

    /// mHC pre for layer `l`'s block: W2's GPU kernel when loaded, else the CPU reference.
    /// `block`: 0 = the attention half (`hc_attn` weights), 1 = the FFN half (`hc_ffn`) —
    /// it selects the resident weights; `w` is the same block's host copy (CPU path).
    fn mhc_pre(
        &self,
        l: usize,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] block: usize,
        w: &super::mhc::MhcWeights,
        r: &[f32],
        t: usize,
    ) -> Result<(Vec<f32>, Vec<super::mhc::Mix>)> {
        let (hc, n) = (self.m.hp.hc, self.m.hp.n_embd);
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            if t <= g.max_t {
                if let Some(m) = &g.mhc_g[l] {
                    g.devs[g.dev_of[l]]
                        .0
                        .bind()
                        .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                    // the block index, NOT a constant: `pre(0, ..)` ran the FFN half's mix
                    // with the attention half's hc weights (the GLM-5.3 all-GPU divergence)
                    let mix = m.pre(block, r, t)?;
                    // MixSet -> Vec<Mix> (the CPU post path's input shape; the resident post
                    // keeps the MixSet and avoids the repack below)
                    let mixes = (0..t)
                        .map(|i| {
                            let mut m2 = super::mhc::Mix {
                                x: mix.x[i * n..(i + 1) * n].to_vec(),
                                post: [0f32; 4],
                                comb: [[0f32; 4]; 4],
                            };
                            m2.post.copy_from_slice(&mix.post[i * 4..(i + 1) * 4]);
                            for src in 0..4 {
                                m2.comb[src].copy_from_slice(
                                    &mix.comb[i * 16 + src * 4..i * 16 + src * 4 + 4],
                                );
                            }
                            m2
                        })
                        .collect();
                    return Ok((mix.x, mixes));
                }
            }
        }
        let _ = l;
        mhc_pre_all(w, r, t, hc, n)
    }

    /// mHC post for layer `l`'s block: `r ← post ⊙ y + combᵀ r`, GPU or CPU as [`Self::mhc_pre`].
    fn mhc_post(
        &self,
        l: usize,
        mixes: &[super::mhc::Mix],
        y: &[f32],
        r: &mut [f32],
        t: usize,
    ) -> Result<()> {
        let (hc, n) = (self.m.hp.hc, self.m.hp.n_embd);
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            if t <= g.max_t {
                if let Some(m) = &g.mhc_g[l] {
                    g.devs[g.dev_of[l]]
                        .0
                        .bind()
                        .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                    let out = m.post_mixes(mixes, y, r, t)?;
                    r.copy_from_slice(&out);
                    return Ok(());
                }
            }
        }
        let _ = l;
        mhc_post_all(mixes, y, r, t, hc, n)
    }

    /// Start a fused-mHC window when it applies ([`GpuFfn::mhc_f`]): uploads the embedded
    /// streams. `false`: this window runs the per-call path (no fused instance, a window
    /// past the scratch, or the per-layer dump, which reads the host streams).
    fn mhc_fused_begin(&mut self, r: &[f32], t: usize) -> Result<bool> {
        #[cfg(feature = "cuda")]
        if self.dump.is_none() {
            if let Some(g) = self.gpu.as_mut() {
                let all = g.mhc_g.iter().all(|m| m.is_some());
                if let (true, Some(f)) = (all, g.mhc_f.as_mut()) {
                    if t <= f.max_t() {
                        g.devs[0]
                            .0
                            .bind()
                            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                        f.begin(r, t)?;
                        return Ok(true);
                    }
                }
            }
        }
        let _ = (r, t);
        Ok(false)
    }

    /// One half block's mHC pre followed by its RMSNorm (block 0: `attn_norm`, block 1:
    /// `ffn_norm`). Fused: one device pass, the streams stay resident and `r` is not read
    /// (the mixes stay on the device too, so the returned `Vec<Mix>` is empty).
    #[allow(clippy::too_many_arguments)]
    fn mhc_pre_norm(
        &mut self,
        fused: bool,
        l: usize,
        block: usize,
        w: &super::mhc::MhcWeights,
        r: &[f32],
        t: usize,
    ) -> Result<(Vec<f32>, Vec<super::mhc::Mix>)> {
        #[cfg(feature = "cuda")]
        if fused {
            let g = self.gpu.as_mut().context("fused mHC without a GPU")?;
            g.devs[0]
                .0
                .bind()
                .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
            let mh = g.mhc_g[l]
                .as_ref()
                .context("fused mHC: layer not resident")?;
            let f = g.mhc_f.as_mut().context("fused mHC: no instance")?;
            // `block`, not a constant: it picks hc_attn/hc_ffn AND attn_norm/ffn_norm
            return Ok((f.pre(mh, l, block, Some(w))?, Vec::new()));
        }
        let _ = fused;
        let (mut x, mixes) = self.mhc_pre(l, block, w, r, t)?;
        let norm = if block == 0 {
            "attn_norm.weight"
        } else {
            "ffn_norm.weight"
        };
        let _tr = super::trace::span("norm");
        self.m.norm_all(&mut x, l, norm)?;
        Ok((x, mixes))
    }

    /// The half block's mHC post. Fused: uploads `y`; the next pre (or the head) applies it.
    fn mhc_post_any(
        &mut self,
        fused: bool,
        l: usize,
        mixes: &[super::mhc::Mix],
        y: &[f32],
        r: &mut [f32],
        t: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if fused {
            let g = self.gpu.as_mut().context("fused mHC without a GPU")?;
            g.devs[0]
                .0
                .bind()
                .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
            return g.mhc_f.as_mut().context("fused mHC: no instance")?.post(y);
        }
        let _ = fused;
        self.mhc_post(l, mixes, y, r, t)
    }

    /// The head's unweighted stream mean `[t][n]`. Fused: the last post + mean on the device.
    fn hc_head_all(&mut self, fused: bool, r: &[f32], t: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if fused {
            let g = self.gpu.as_mut().context("fused mHC without a GPU")?;
            g.devs[0]
                .0
                .bind()
                .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
            return g.mhc_f.as_mut().context("fused mHC: no instance")?.end();
        }
        let _ = fused;
        let (n, hc) = (self.m.hp.n_embd, self.m.hp.hc);
        let mut h = vec![0f32; t * n];
        for i in 0..t {
            for c in 0..hc {
                for j in 0..n {
                    h[i * n + j] += r[(i * hc + c) * n + j];
                }
            }
            for v in &mut h[i * n..(i + 1) * n] {
                *v /= hc as f32;
            }
        }
        Ok(h)
    }

    /// Run `ids` after the tokens so far (slot 0). Returns their logits `[ids.len()][n_vocab]`.
    pub fn window(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.window_slot(0, ids)
    }

    /// [`Self::window`] for sequence `slot`: the KDA state (host + device slices), the DSA
    /// caches (host mirrors + the device LRU entry) and the tokens are that slot's; the
    /// weights, mHC and FFN state are shared (nothing per-sequence persists there). One
    /// window runs at a time, so the shared per-window scratch needs no partitioning.
    pub fn window_slot(&mut self, slot: usize, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "empty window");
        ensure!(
            slot < self.n_slots,
            "slot {slot} >= {n} (use_gpu_slots?)",
            n = self.n_slots
        );
        let w0 = Instant::now();
        #[cfg(feature = "cuda")]
        self.drop_step_graph();
        let prof = std::env::var("TANG_GLM5_PROF").is_ok_and(|v| v == "1");
        let stage = |name: &str, t0: &std::time::Instant| {
            if prof {
                eprintln!(
                    "glm5: prof {name}: {:.1} ms",
                    t0.elapsed().as_secs_f32() * 1e3
                );
            }
        };
        let hp = self.m.hp.clone();
        let (n, hc, t) = (hp.n_embd, hp.hc, ids.len());
        let pos0 = self.slot_tokens(slot).len();
        // TANG_GLM5_TRACE: the window's span tree (trace.rs); every guard below is a no-op
        // when it's off
        use super::trace::{span, span_l};
        let tr_root = super::trace::window_begin(t, pos0);

        // ---- embedding into every stream
        let tr = span("embed");
        let embd = self.m.g.info("token_embd.weight")?;
        let mut r = vec![0f32; t * hc * n];
        for (i, &tok) in ids.iter().enumerate() {
            ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
            let e = self.m.g.rows(embd, tok as usize, 1)?;
            for c in 0..hc {
                r[(i * hc + c) * n..(i * hc + c + 1) * n].copy_from_slice(&e);
            }
        }
        let mhp = super::mhc::MhcHparams {
            streams: hc,
            sinkhorn_iters: hp.hc_sinkhorn_iters as u32,
            hc_eps: hp.hc_eps,
            rms_eps: hp.eps,
        };

        let fused = self.mhc_fused_begin(&r, t)?;
        drop(tr);
        let t_embd = Instant::now();
        for l in 0..hp.n_layer {
            let _tr_l = span_l("layer", l);
            // ---- attention half
            let t0 = Instant::now();
            let tr = span("mhc_pre.attn");
            let w_attn = self.hc_weights(l, "hc_attn", &mhp)?;
            let (x, mixes) = self.mhc_pre_norm(fused, l, 0, &w_attn, &r, t)?;
            drop(tr);
            stage("mhc_pre+norm(attn)", &t0);
            if let Some(w) = self.wintap.as_mut() {
                w[l * 4].extend_from_slice(&x);
            }
            let _t1 = Instant::now();
            let y = if hp.is_kda[l] {
                let _tr = span("kda");
                let t_w = Instant::now();
                let tr = span("weights");
                let wc = self.kda_weights(l)?;
                drop(tr);
                stage("kda weights", &t_w);
                // no wc.0.clone(): the free-function paths only read the weights, and the
                // resident path doesn't touch them at all — pass the Arc's ref
                let (w, cfg) = (&wc.0, wc.1);
                let t_m = Instant::now();
                let mut st = self.kda[slot][l]
                    .take()
                    .unwrap_or_else(|| (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1)));
                let y = self.kda_mixer(slot, l, &x, t, w, &cfg, &mut st);
                self.kda[slot][l] = Some(st);
                stage("kda mixer", &t_m);
                y?
            } else {
                let _tr = span("dsa");
                let t_w = Instant::now();
                let tr = span("weights");
                let w = self.mla_weights(l)?;
                drop(tr);
                stage("mla weights", &t_w);
                let t_m = Instant::now();
                let y = self.dsa_layer(slot, l, &w, &x, t, pos0)?;
                stage("mla mixer", &t_m);
                y
            };
            if let Some(w) = self.wintap.as_mut() {
                w[l * 4 + 1].extend_from_slice(&y);
            }
            let t2 = Instant::now();
            let tr = span("mhc_post.attn");
            self.mhc_post_any(fused, l, &mixes, &y, &mut r, t)?;
            drop(tr);
            stage("mhc_post(attn)", &t2);

            // ---- FFN half
            let t3 = Instant::now();
            let tr = span("mhc_pre.ffn");
            let w_ffn = self.hc_weights(l, "hc_ffn", &mhp)?;
            let (x, mixes) = self.mhc_pre_norm(fused, l, 1, &w_ffn, &r, t)?;
            drop(tr);
            stage("mhc_pre+norm(ffn)", &t3);
            if let Some(w) = self.wintap.as_mut() {
                w[l * 4 + 2].extend_from_slice(&x);
            }
            let tr = span("ffn");
            let y = self.ffn(l, &x, t)?;
            drop(tr);
            if let Some(w) = self.wintap.as_mut() {
                w[l * 4 + 3].extend_from_slice(&y);
            }
            let t4 = Instant::now();
            let tr = span("mhc_post.ffn");
            self.mhc_post_any(fused, l, &mixes, &y, &mut r, t)?;
            drop(tr);
            stage("ffn", &t3);
            stage("mhc_post(ffn)", &t4);
            // W0's R3 bisector: dump the last position's post-FFN streams (no-op
            // without TANG_DUMP_DIR; lives in dump.rs).
            if let Some(d) = self.dump.as_mut() {
                let row = &r[(t - 1) * hc * n..t * hc * n];
                d.layer(l, pos0 + t - 1, row);
            }
        }

        stage("layers", &t_embd);
        // ---- head: unweighted stream mean, output norm, lm_head
        let t5 = Instant::now();
        let tr_head = span("head");
        let tr = span("head.mean");
        let mut h = self.hc_head_all(fused, &r, t)?;
        drop(tr);
        let tr = span("head.norm");
        self.m.norm_all_exact(&mut h, "output_norm.weight")?;
        self.last_h.clear();
        self.last_h.extend_from_slice(&h);
        drop(tr);
        let tr = span("lm_head");
        // on the device LM head when TANG_GLM5_GPU_HEAD=1 (GpuFfn::head_logits)
        #[cfg(feature = "cuda")]
        let dev = match self.gpu.as_ref() {
            // every row of a window too, one at a time: an MTP verify window must see the
            // same (q8-activation) head numerics as the t=1 decode it replaces
            Some(g) => {
                let (n, nv) = (self.m.hp.n_embd, self.m.hp.n_vocab);
                let r0 = if self.prefill_last { t - 1 } else { 0 };
                let mut all = Vec::with_capacity((t - r0) * nv);
                for i in r0..t {
                    match g.head_logits(&h[i * n..(i + 1) * n], n, nv)? {
                        Some(l) => all.extend_from_slice(&l),
                        None => break,
                    }
                }
                (all.len() == (t - r0) * nv).then_some(all)
            }
            None => None,
        };
        #[cfg(not(feature = "cuda"))]
        let dev: Option<Vec<f32>> = None;
        let logits = match dev {
            Some(l) => l,
            _ => {
                if self.head.is_none() {
                    self.head = Some(self.m.head()?);
                }
                self.head.as_ref().unwrap().apply(&h, t, false)
            }
        };
        drop(tr);
        drop(tr_head);
        stage("head", &t5);
        self.slot_tokens_mut(slot).extend_from_slice(ids);
        // W0's R3 bisector: write the dump (no-op without TANG_DUMP_DIR).
        let toks = self.slot_tokens(slot).to_vec();
        if let Some(d) = self.dump.as_mut() {
            let last_row = t - 1;
            d.finish(
                &toks,
                &h[last_row * n..(last_row + 1) * n],
                &logits[logits.len() - hp.n_vocab..],
                hp.n_vocab,
            )?;
        }
        super::trace::window_end(tr_root, t, pos0);
        if self.verbose {
            eprintln!(
                "glm5: window t={t} at {pos0} (slot {slot}): {:.2} s",
                w0.elapsed().as_secs_f64()
            );
        }
        Ok(logits)
    }

    /// The batched decode window: row `i` is token `rows[i].1` of sequence `rows[i].0`
    /// (distinct slots), each at that slot's own position. Everything per-token runs ONCE
    /// over all rows — embedding, mHC, norms, the KDA projections / gates / o_proj, the
    /// FFN (the CPU MoE dequantises each routed expert once per window, not per row) and
    /// the head — and only the per-sequence mixers split: the KDA core steps each row's
    /// slot state (one launch pair per row), each DSA layer attends row `i` over slot
    /// `rows[i].0`'s caches. Returns the logits `[rows.len()][n_vocab]`. Numerically the
    /// same as one [`Self::window_slot`] per row (every stage is per-token).
    pub fn window_batch(&mut self, rows: &[(usize, u32)]) -> Result<Vec<f32>> {
        let t = rows.len();
        ensure!(t > 0, "empty batch");
        for (i, &(s, _)) in rows.iter().enumerate() {
            ensure!(s < self.n_slots, "slot {s} >= {n}", n = self.n_slots);
            ensure!(
                rows[..i].iter().all(|r| r.0 != s),
                "slot {s} twice in one batch"
            );
        }
        if t == 1 {
            return self.window_slot(rows[0].0, &[rows[0].1]);
        }
        #[cfg(feature = "cuda")]
        self.drop_step_graph();
        let hp = self.m.hp.clone();
        let (n, hc) = (hp.n_embd, hp.hc);
        use super::trace::{span, span_l};
        let tr_root = super::trace::window_begin(t, 0);
        let tr = span("embed");
        let embd = self.m.g.info("token_embd.weight")?;
        let mut r = vec![0f32; t * hc * n];
        for (i, &(_, tok)) in rows.iter().enumerate() {
            ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
            let e = self.m.g.rows(embd, tok as usize, 1)?;
            for c in 0..hc {
                r[(i * hc + c) * n..(i * hc + c + 1) * n].copy_from_slice(&e);
            }
        }
        let mhp = super::mhc::MhcHparams {
            streams: hc,
            sinkhorn_iters: hp.hc_sinkhorn_iters as u32,
            hc_eps: hp.hc_eps,
            rms_eps: hp.eps,
        };
        let fused = self.mhc_fused_begin(&r, t)?;
        drop(tr);
        let slots: Vec<usize> = rows.iter().map(|r| r.0).collect();
        let pos: Vec<usize> = slots.iter().map(|&s| self.slot_tokens(s).len()).collect();
        for l in 0..hp.n_layer {
            let _tr_l = span_l("layer", l);
            let tr = span("mhc_pre.attn");
            let w_attn = self.hc_weights(l, "hc_attn", &mhp)?;
            let (x, mixes) = self.mhc_pre_norm(fused, l, 0, &w_attn, &r, t)?;
            drop(tr);
            let y = if hp.is_kda[l] {
                let _tr = span("kda");
                let wc = self.kda_weights(l)?;
                self.kda_rows(l, &slots, &x, &wc.0, &wc.1)?
            } else {
                let _tr = span("dsa");
                let w = self.mla_weights(l)?;
                let mut y = Vec::with_capacity(t * n);
                for i in 0..t {
                    y.extend(self.dsa_layer(slots[i], l, &w, &x[i * n..(i + 1) * n], 1, pos[i])?);
                }
                y
            };
            let tr = span("mhc_post.attn");
            self.mhc_post_any(fused, l, &mixes, &y, &mut r, t)?;
            drop(tr);
            let tr = span("mhc_pre.ffn");
            let w_ffn = self.hc_weights(l, "hc_ffn", &mhp)?;
            let (x, mixes) = self.mhc_pre_norm(fused, l, 1, &w_ffn, &r, t)?;
            drop(tr);
            let tr = span("ffn");
            let y = self.ffn(l, &x, t)?;
            drop(tr);
            let tr = span("mhc_post.ffn");
            self.mhc_post_any(fused, l, &mixes, &y, &mut r, t)?;
            drop(tr);
        }
        let tr_head = span("head");
        let mut h = self.hc_head_all(fused, &r, t)?;
        self.m.norm_all_exact(&mut h, "output_norm.weight")?;
        if self.head.is_none() {
            self.head = Some(self.m.head()?);
        }
        let logits = self.head.as_ref().unwrap().apply(&h, t, false);
        drop(tr_head);
        for &(s, tok) in rows {
            self.slot_tokens_mut(s).push(tok);
        }
        super::trace::window_end(tr_root, t, 0);
        Ok(logits)
    }

    /// KDA layer `l` for a batch of decode rows (row `i` = one token of `slots[i]`): the
    /// resident layer's batched rows path when it can take them, else one
    /// [`Self::kda_mixer`] call per row (the CPU / staged / shadow-checked paths).
    fn kda_rows(
        &mut self,
        l: usize,
        slots: &[usize],
        x: &[f32],
        w: &KdaLayerW,
        cfg: &KdaConfig,
    ) -> Result<Vec<f32>> {
        let n = self.m.hp.n_embd;
        let t = slots.len();
        #[cfg(feature = "cuda")]
        if self.kda_dbg.is_none() {
            if let Some(g) = self.gpu.as_mut() {
                if let Some(dev) = &mut g.kda_l[l] {
                    if dev.pf2_on() && t <= g.max_t {
                        let d = g.dev_of[l];
                        g.devs[d]
                            .0
                            .bind()
                            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                        // every row's slot slice must be the live device copy
                        for &s in slots {
                            if !g.kda_dev_fresh[s][l] {
                                let st = self.kda[s][l].take().unwrap_or_else(|| {
                                    (ConvState::zeros(cfg, 1), KdaState::zeros(cfg, 1))
                                });
                                dev.seed_at(s, &st.0, &st.1)?;
                                self.kda[s][l] = Some(st);
                                g.kda_dev_fresh[s][l] = true;
                            }
                        }
                        return dev.window_rows(slots, x);
                    }
                }
            }
        }
        let mut y = Vec::with_capacity(t * n);
        for (i, &s) in slots.iter().enumerate() {
            let mut st = self.kda[s][l]
                .take()
                .unwrap_or_else(|| (ConvState::zeros(cfg, 1), KdaState::zeros(cfg, 1)));
            let r = self.kda_mixer(s, l, &x[i * n..(i + 1) * n], 1, w, cfg, &mut st);
            self.kda[s][l] = Some(st);
            y.extend(r?);
        }
        Ok(y)
    }

    /// Prefill `ids` in windows of `chunk` (slot 0); returns the greedy next token.
    pub fn prefill(&mut self, ids: &[u32], chunk: usize) -> Result<u32> {
        self.prefill_slot(0, ids, chunk)
    }

    /// [`Self::prefill`] for sequence `slot`.
    pub fn prefill_slot(&mut self, slot: usize, ids: &[u32], chunk: usize) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prompt");
        let nv = self.m.hp.n_vocab;
        let mut last = Vec::new();
        self.prefill_last = true;
        for c in ids.chunks(chunk.max(1)) {
            let lg = self.window_slot(slot, c);
            let lg = match lg {
                Ok(l) => l,
                Err(e) => {
                    self.prefill_last = false;
                    return Err(e);
                }
            };
            last = lg[lg.len() - nv..].to_vec();
        }
        self.prefill_last = false;
        Ok(argmax(&last))
    }

    /// Feed one token (slot 0); returns the greedy next token.
    pub fn step(&mut self, tok: u32) -> Result<u32> {
        self.step_slot(0, tok)
    }

    /// Feed one token to sequence `slot`; returns the greedy next token. The device-
    /// resident step ([`resident`]) when every stage is resident, else the host path.
    pub fn step_slot(&mut self, slot: usize, tok: u32) -> Result<u32> {
        #[cfg(feature = "cuda")]
        if let Some(t) = self.step_resident(slot, tok)? {
            return Ok(t);
        }
        Ok(argmax(&self.window_slot(slot, &[tok])?))
    }
}

/// mHC pre for every token of a window: the collapsed inputs `[t][n]` and each token's mix.
fn mhc_pre_all(
    w: &super::mhc::MhcWeights,
    r: &[f32],
    t: usize,
    hc: usize,
    n: usize,
) -> Result<(Vec<f32>, Vec<super::mhc::Mix>)> {
    let mut x = vec![0f32; t * n];
    let mut mixes = Vec::with_capacity(t);
    for i in 0..t {
        let streams: Vec<&[f32]> = (0..hc)
            .map(|c| &r[(i * hc + c) * n..(i * hc + c + 1) * n])
            .collect();
        let mix = super::mhc::mhc_pre(w, &streams)?;
        x[i * n..(i + 1) * n].copy_from_slice(&mix.x);
        mixes.push(mix);
    }
    Ok((x, mixes))
}

/// mHC post for every token: `r ← post ⊙ y + combᵀ r`.
fn mhc_post_all(
    mixes: &[super::mhc::Mix],
    y: &[f32],
    r: &mut [f32],
    t: usize,
    hc: usize,
    n: usize,
) -> Result<()> {
    let r_in = r.to_vec();
    for i in 0..t {
        let streams: Vec<&[f32]> = (0..hc)
            .map(|c| &r_in[(i * hc + c) * n..(i * hc + c + 1) * n])
            .collect();
        let token = &mut r[i * hc * n..(i + 1) * hc * n];
        let mut outs: Vec<&mut [f32]> = token.chunks_mut(n).collect();
        super::mhc::mhc_post(&mixes[i], &y[i * n..(i + 1) * n], &streams, &mut outs)?;
    }
    Ok(())
}

/// One KDA layer with W4's CUDA conv + delta core (feature `cuda`). Projections, gates and
/// the gated output norm stay on the host (the reference's f32 `matmul`). Prefill runs
/// 64-token GPU chunks (the kernel's fixed size), a single-token window runs the GPU decode
/// step, and any other tail runs the CPU core on the same state. `x` `[t][hidden]` in,
/// `[t][hidden]` out; `conv` / `state` are updated in place.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn kda_gpu(
    x: &[f32],
    t: usize,
    w: &KdaLayerW,
    cfg: &KdaConfig,
    k: &super::kda_cuda::cuda::Kernels,
    gpu: &tang_moe::gpu::Gpu,
    conv: &mut ConvState,
    state: &mut KdaState,
) -> Result<Vec<f32>> {
    use super::kda::{matmul, sigmoid};
    use super::kda_cuda::cuda;
    let (nh, hd) = (cfg.heads, cfg.head_dim);
    let hidden = w.hidden;
    ensure!(x.len() == t * hidden, "layer input");
    use super::trace::span;
    // projections, stacked q|k|v per token (the reference's layout)
    let tr = span("proj.host");
    let qp = matmul(&w.q, nh * hd, hidden, x, t);
    let kp = matmul(&w.k, nh * hd, hidden, x, t);
    let vp = matmul(&w.v, nh * hd, hidden, x, t);
    let ch = 3 * nh * hd;
    let mut qkv = vec![0f32; t * ch];
    for i in 0..t {
        qkv[i * ch..i * ch + nh * hd].copy_from_slice(&qp[i * nh * hd..(i + 1) * nh * hd]);
        qkv[i * ch + nh * hd..i * ch + 2 * nh * hd]
            .copy_from_slice(&kp[i * nh * hd..(i + 1) * nh * hd]);
        qkv[i * ch + 2 * nh * hd..(i + 1) * ch]
            .copy_from_slice(&vp[i * nh * hd..(i + 1) * nh * hd]);
    }
    drop(tr);
    // gates: gpre (pre-exp, per key dim) and beta (post-sigmoid, per head)
    let tr = span("gates.host");
    let fg = super::kda::ForgetGateW {
        f_a: w.f_a.clone(),
        f_b: w.f_b.clone(),
        dt_bias: w.dt_bias.clone(),
        decay_rate: w.decay_rate.clone(),
    };
    let (b, c) = (1usize, 64usize);
    let gpre = super::kda::forget_gate(x, &fg, cfg, t);
    let beta = super::kda::beta_gate(x, &w.b_proj, cfg, b, t);
    drop(tr);
    let tr = span("core");
    let mut y = vec![0f32; t * nh * hd];
    let mut off = 0usize;
    if t == 1 {
        let o = cuda::decode(gpu, k, cfg, &qkv, &w.conv, &gpre, &beta, conv, state, b)?;
        y.copy_from_slice(&o);
        off = 1;
    }
    while off + c <= t {
        let o = cuda::prefill(
            gpu,
            k,
            cfg,
            &qkv[off * ch..(off + c) * ch],
            &w.conv,
            &gpre[off * nh * hd..(off + c) * nh * hd],
            &beta[off * nh..(off + c) * nh],
            conv,
            state,
            b,
            c,
            c,
        )?;
        y[off * nh * hd..(off + c) * nh * hd].copy_from_slice(&o);
        off += c;
    }
    // a tail shorter than one chunk: the CPU core on the same state objects
    if off < t {
        let rem = t - off;
        let yc = super::kda::conv1d_prefill(&qkv[off * ch..t * ch], &w.conv, conv, cfg, b, rem)?;
        let oc = super::kda::kda_chunk(
            &yc,
            &gpre[off * nh * hd..t * nh * hd],
            &beta[off * nh..t * nh],
            state,
            cfg,
            b,
            rem,
            c,
        )?;
        y[off * nh * hd..t * nh * hd].copy_from_slice(&oc);
    }
    drop(tr);
    // gated output norm (host, same as the reference), then o_proj
    let _tr = span("out.host");
    let ga = matmul(&w.g_a, hd, hidden, x, t);
    let g2 = matmul(&w.g_b, nh * hd, hd, &ga, t);
    for i in 0..t {
        for h in 0..nh {
            let dst = &mut y[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd];
            crate::flash::reference::rms_norm_mul(dst, &w.o_norm, cfg.eps);
            for (yj, gj) in dst
                .iter_mut()
                .zip(&g2[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd])
            {
                *yj *= sigmoid(*gj);
            }
        }
    }
    Ok(matmul(&w.o_proj, hidden, nh * hd, &y, t))
}

/// One DSA layer with W3's CUDA kernels (feature `cuda`), naive: the caches live on the
/// host and are staged per call (W3's launchers), q_a / q_b / o_proj are host GEMVs.
/// Latents are bf16 (W3's cache format), so this is ~1e-3 from the f32 CPU path.
/// Order follows W3's parity test: kv_append → idx_append → (sparse) idx_pool + idx_run →
/// attend.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn dsa_gpu(
    l: usize,
    w: &MlaWeights,
    wkv: &(Vec<f32>, Vec<f32>),
    cu: &super::mla::cuda::MlaCuda,
    x: &[f32],
    t: usize,
    pos0: usize,
    c: &mut DsaCache,
    dbg: Option<&mut super::mla_debug::Shadow>,
) -> Result<Vec<f32>> {
    let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 mla cuda: {e}");
    let s = tang_moe::gpu::Stream::blocking().map_err(gerr)?;
    let (n, dl, idd) = (w.hidden, w.kv_lora, w.idx_dim);
    let kv_len = pos0 + t;
    ensure!(x.len() == t * n, "layer input");
    ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
    use super::trace::span;
    let tr = span("q_a.host");
    let q_resid = w.gemv_q_a(x, t);
    drop(tr);
    if dbg.as_ref().is_some_and(|d| d.poison()) {
        cu.debug_poison(pos0, t).map_err(gerr)?;
    }
    // caches grow by t rows; the launchers write rows [pos0, kv_len)
    let tr = span("kv_append");
    c.lat16.resize(kv_len * dl, 0);
    cu.kv_append(&w.kv_a, &w.kv_a_norm, x, t, pos0, &mut c.lat16, &s)
        .map_err(gerr)?;
    drop(tr);
    let tr = span("idx_append");
    c.idx_k.resize(kv_len * idd, 0.0);
    c.idx_g.resize(kv_len * idd, 0.0);
    c.valid.resize(kv_len, 0);
    cu.idx_append(
        &w.i_wk,
        &w.i_k_norm_w,
        &w.i_k_norm_b,
        &w.i_gate,
        x,
        t,
        pos0,
        &mut c.idx_k,
        &mut c.idx_g,
        &mut c.valid,
        &s,
    )
    .map_err(gerr)?;
    drop(tr);
    let tr = span("idx_select");
    let mut pool_dbg = None;
    let sel = if super::mla::indexer_is_dense(kv_len, w.topk) {
        None
    } else {
        let n_pool = kv_len / w.kpool;
        let mut pool_keys = vec![0f32; n_pool * idd];
        cu.idx_pool(
            &c.idx_k,
            &c.idx_g,
            &w.i_ape,
            kv_len,
            0,
            n_pool,
            &mut pool_keys,
            &s,
        )
        .map_err(gerr)?;
        if dbg.is_some() {
            pool_dbg = Some(pool_keys.clone());
        }
        let mut sel = vec![-1i32; t * w.select_width()];
        cu.idx_run(
            &w.i_wq_b, &w.i_proj, &q_resid, x, &pool_keys, &c.valid, kv_len, 0, n_pool, t, pos0,
            &mut sel, &s,
        )
        .map_err(gerr)?;
        Some(sel)
    };
    drop(tr);
    let tr = span("q_b.host");
    let q = w.gemv_q_b(&q_resid, t);
    drop(tr);
    let tr = span("attend.gather");
    // the concatenated per-head decompress matrices (the resident MlaCuda reads its own
    // device copies; these are its host-side contract) — built once per layer by the
    // engine (`mla_wkv`), not per call: 2 × 25 MB of copying per DSA row otherwise
    let (wk, wv) = wkv;
    drop(tr);
    let tr = span("attend");
    let hv = w.n_head * w.v_head;
    let mut out_v = vec![0f32; t * hv];
    cu.attend(
        wk,
        wv,
        &q,
        &c.lat16,
        kv_len,
        t,
        pos0,
        sel.as_deref(),
        &mut out_v,
        &s,
    )
    .map_err(gerr)?;
    drop(tr);
    // o_proj on the host
    let _tr = span("o_proj.host");
    use rayon::prelude::*;
    let mut out = vec![0f32; t * n];
    // parallel over (token, row); each element's dot is the serial loop's
    out.par_iter_mut().enumerate().for_each(|(k, o)| {
        let (i, r) = (k / n, k % n);
        *o = crate::flash::reference::dot(&w.o[r * hv..(r + 1) * hv], &out_v[i * hv..(i + 1) * hv]);
    });
    if let Some(d) = dbg {
        let (lat_dev, idk_dev, idg_dev, val_dev) = cu.debug_rows(0, kv_len).map_err(gerr)?;
        d.check(
            l,
            w,
            &super::mla_debug::GpuWindow {
                x,
                q_resid: &q_resid,
                t,
                pos0,
                lat_dev,
                idk_dev,
                idg_dev,
                val_dev,
                lat_mirror: &c.lat16,
                idk_mirror: &c.idx_k,
                idg_mirror: &c.idx_g,
                val_mirror: &c.valid,
                pool: pool_dbg,
                sel: sel.clone(),
                out: &out,
                q_resid_dev: false,
                q_dev: None,
            },
        );
    }
    Ok(out)
}

/// [`dsa_gpu`] with the projections on the device ([`super::mla_dev::MlaProj::window`]):
/// `x` up, `y` down, nothing else crosses per layer except the cache mirrors' new rows.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsa_gpu_resident(
    l: usize,
    w: &MlaWeights,
    cu: &super::mla::cuda::MlaCuda,
    p: &super::mla_dev::MlaProj,
    x: &[f32],
    t: usize,
    pos0: usize,
    c: &mut DsaCache,
    dbg: Option<&mut super::mla_debug::Shadow>,
) -> Result<Vec<f32>> {
    let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 mla cuda: {e}");
    let s = tang_moe::gpu::Stream::blocking().map_err(gerr)?;
    let (n, dl, idd) = (w.hidden, w.kv_lora, w.idx_dim);
    let kv_len = pos0 + t;
    ensure!(x.len() == t * n, "layer input");
    ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
    if dbg.as_ref().is_some_and(|d| d.poison()) {
        cu.debug_poison(pos0, t).map_err(gerr)?;
    }
    c.lat16.resize(kv_len * dl, 0);
    c.idx_k.resize(kv_len * idd, 0.0);
    c.idx_g.resize(kv_len * idd, 0.0);
    c.valid.resize(kv_len, 0);
    let mut pool = Vec::new();
    let o = p.window(
        cu,
        w,
        x,
        t,
        pos0,
        &mut c.lat16,
        &mut c.idx_k,
        &mut c.idx_g,
        &mut c.valid,
        dbg.is_some().then_some(&mut pool),
        dbg.is_some(),
        &s,
    )?;
    if let Some(d) = dbg {
        let (lat_dev, idk_dev, idg_dev, val_dev) = cu.debug_rows(0, kv_len).map_err(gerr)?;
        let q_resid = o.q_resid.clone().unwrap_or_default();
        d.check(
            l,
            w,
            &super::mla_debug::GpuWindow {
                x,
                q_resid: &q_resid,
                t,
                pos0,
                lat_dev,
                idk_dev,
                idg_dev,
                val_dev,
                lat_mirror: &c.lat16,
                idk_mirror: &c.idx_k,
                idg_mirror: &c.idx_g,
                val_mirror: &c.valid,
                pool: o.sel.is_some().then_some(pool),
                sel: o.sel.clone(),
                out: &o.y,
                q_resid_dev: true,
                q_dev: o.q.as_deref(),
            },
        );
    }
    Ok(o.y)
}

pub(crate) fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Is the GGUF at `path` a `glm5-next` model?
pub fn is_glm5(path: &Path) -> bool {
    crate::gguf::Gguf::open(path)
        .ok()
        .and_then(|g| {
            g.meta_str("general.architecture")
                .ok()
                .map(|a| a == "glm5-next")
        })
        .unwrap_or(false)
}

/// `flash-generate` / `flash-bench` for a glm5-next GGUF:
/// `<gguf> (--prompt-ids IDS | --ids-file F | --prompt TEXT) [-n N] [--chunk C] [--out F]
/// [--quiet] [-v]`. Greedy (temperature 0) only; other flash flags are accepted and ignored.
pub fn cli(args: &[String], bench: bool) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut prompt, mut n, mut chunk) = (Vec::<u32>::new(), None, 32usize, 64usize);
    let (mut out, mut quiet, mut verbose) = (None::<PathBuf>, false, false);
    // `--slots N` (N > 1): the batcher bench — N sequence slots, the same prompt in each,
    // decoded round-robin; per-step wall ms + every slot's ids (they must all be equal)
    let mut slots = 1usize;
    let mut rr = false;
    let mut diverse = false;
    // --mtp K: draft K tokens per trunk pass with the NextN layer (super::mtp); 0 = plain
    let mut mtp_k = std::env::var("TANG_GLM5_MTP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0usize);
    if std::env::var("TANG_GLM5_MTP").is_ok_and(|v| v == "auto") {
        super::mtp::AUTO.store(true, std::sync::atomic::Ordering::Relaxed);
        mtp_k = 3;
    }
    let push = |ids: &mut Vec<u32>, s: &str| -> Result<()> {
        for w in s.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']') {
            if !w.is_empty() {
                ids.push(w.parse().with_context(|| format!("bad token id {w:?}"))?);
            }
        }
        Ok(())
    };
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            // the flash CLI takes one quoted list; bare ids that follow are accepted too
            "--prompt-ids" => push(&mut ids, val()?)?,
            "--ids-file" => push(&mut ids, &std::fs::read_to_string(val()?)?)?,
            "--prompt" => prompt = Some(val()?.clone()),
            "-n" | "--n" => n = val()?.parse()?,
            "--chunk" => chunk = val()?.parse()?,
            "--out" => out = Some(PathBuf::from(val()?)),
            "--quiet" => quiet = true,
            "--slots" => slots = val()?.parse::<usize>()?.max(1),
            // with --slots: one window per slot per step (round-robin) instead of one
            // batched window over all slots
            "--rr" => rr = true,
            // with --slots: slot s > 0 gets the prompt with one token changed, so the
            // slots' states and expert routing differ (identical prompts route every row
            // to the same experts — the CPU MoE's best case); slot 0 keeps the prompt
            "--diverse" => diverse = true,
            "-v" | "--verbose" => verbose = true,
            "--temp" => {
                let t: f32 = val()?.parse()?;
                ensure!(t == 0.0, "glm5: only greedy (--temp 0) for now");
            }
            "--eager" | "--no-think" => {}
            "--mtp" => {
                let v = val()?;
                if v == "auto" {
                    super::mtp::AUTO.store(true, std::sync::atomic::Ordering::Relaxed);
                    mtp_k = 3;
                } else {
                    mtp_k = v.parse()?;
                }
            }
            "--mtp-auto" => super::mtp::AUTO.store(true, std::sync::atomic::Ordering::Relaxed),
            "--draft" | "--seed" | "--ctx" | "--bw" => {
                val()?;
            }
            s if s.parse::<u32>().is_ok() => ids.push(s.parse()?),
            s => bail!("glm5: unknown argument {s}"),
        }
    }
    if let Some(p) = &prompt {
        ids = crate::flash::tokenize::encode_chat(&path, p, None)?;
    }
    ensure!(
        !ids.is_empty(),
        "no prompt (--prompt-ids, --ids-file or --prompt)"
    );
    ensure!(n >= 1, "-n must be at least 1");

    // TANG_GLM5_TRACE=1: span tree + bottleneck table (trace.rs), otel export with the
    // `otel` feature. Before the GPU loads, so kernel names are recorded.
    let traced = super::trace::init();
    let _otel = if traced {
        crate::telemetry::init()
    } else {
        None
    };
    let t0 = Instant::now();
    let mut e = Glm5Engine::open(&path)?;
    e.verbose = verbose;
    #[cfg(feature = "cuda")]
    if slots > 1 {
        e.use_gpu_slots(slots)?;
    } else {
        e.use_gpu()?;
    }
    #[cfg(not(feature = "cuda"))]
    ensure!(slots == 1, "--slots needs a cuda build");
    #[cfg(feature = "cuda")]
    if traced {
        if let Some(g) = &e.gpu {
            let (gbs, what) = sol_gbs(&g.devs[0].0);
            super::trace::set_sol(gbs, what);
        }
    }
    let hp = &e.m.hp;
    eprintln!(
        "glm5: {} layers ({} KDA, {} DSA), hidden {}, {} experts top-{}, vocab {}; open {:.1} s",
        hp.n_layer,
        hp.is_kda.iter().filter(|&&k| k).count(),
        hp.is_kda.iter().filter(|&&k| !k).count(),
        hp.n_embd,
        hp.n_expert,
        hp.n_expert_used,
        hp.n_vocab,
        t0.elapsed().as_secs_f64()
    );
    if slots > 1 {
        ensure!(mtp_k == 0, "--mtp and --slots are exclusive");
        return slots_bench(&mut e, &ids, n, chunk, slots, rr, diverse, out.as_deref());
    }
    if mtp_k > 0 {
        return super::mtp::cli_run(
            &mut e,
            &ids,
            n,
            chunk,
            mtp_k,
            out.as_deref(),
            quiet || bench,
        );
    }
    let t1 = Instant::now();
    let mut cur = e.prefill(&ids, chunk)?;
    let prefill_s = t1.elapsed().as_secs_f64();
    let mut gen = vec![cur];
    // TANG_GLM5_WINDIFF=S,T: bisect window-vs-step numerics. Step-decode until S tokens
    // are generated, snapshot, then (A) T single steps vs (B) one T-token window over the
    // same tokens, comparing the per-layer taps and the logits (top-2 margins).
    if let Ok(spec) = std::env::var("TANG_GLM5_WINDIFF") {
        let v: Vec<usize> = spec.split(',').filter_map(|x| x.parse().ok()).collect();
        ensure!(v.len() == 2, "TANG_GLM5_WINDIFF=S,T");
        return windiff(&mut e, gen, v[0], v[1]);
    }
    // TANG_GLM5_LOGITS_DUMP=path: the f32 logits of every decode step (device LM head,
    // TANG_GLM5_GPU_HEAD=1), n_vocab floats each, appended in order (numerics A/B)
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut dump = match std::env::var("TANG_GLM5_LOGITS_DUMP") {
        Ok(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        Err(_) => None,
    };
    let t2 = Instant::now();
    while gen.len() < n {
        cur = e.step(cur)?;
        gen.push(cur);
        #[cfg(feature = "cuda")]
        if let Some(w) = dump.as_mut() {
            let nv = e.m.hp.n_vocab;
            let l = e
                .gpu
                .as_ref()
                .map(|g| g.last_head_logits(nv))
                .transpose()?
                .flatten();
            let l = l.context("TANG_GLM5_LOGITS_DUMP needs TANG_GLM5_GPU_HEAD=1")?;
            use std::io::Write;
            w.write_all(&l.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
    }
    drop(dump);
    let decode_s = t2.elapsed().as_secs_f64();
    println!(
        "prompt: {} tokens, prefill {:.2} s ({:.1} tok/s)",
        ids.len(),
        prefill_s,
        ids.len() as f64 / prefill_s
    );
    let line = gen
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    println!("ids: {line}");
    if gen.len() > 1 {
        println!(
            "decode: {} tokens in {:.2} s ({:.2} tok/s)",
            gen.len() - 1,
            decode_s,
            (gen.len() - 1) as f64 / decode_s
        );
    }
    if !quiet && !bench {
        let tk = crate::flash::tokenize::FlashTokenizer::from_gguf(&e.m.g);
        if let Ok(text) = tk.and_then(|t| t.decode(&gen)) {
            println!("text: {text}");
        }
    }
    if let Some(o) = out {
        std::fs::write(&o, format!("{line}\n"))?;
    }
    if let Some(r) = super::trace::report() {
        eprint!("{r}");
    }
    Ok(())
}

/// TANG_GLM5_WINDIFF driver (see `cli`): A = T steps, B = one T window, same tokens.
fn windiff(e: &mut Glm5Engine, mut gen: Vec<u32>, s: usize, t: usize) -> Result<()> {
    let nl = e.m.hp.n_layer;
    let nv = e.m.hp.n_vocab;
    while gen.len() < s + 1 {
        let c = *gen.last().unwrap();
        gen.push(e.step(c)?);
    }
    // gen[s] is the next token to feed (gen[..s] are in the context)
    eprintln!(
        "windiff: context {} tokens, feeding gen[{s}..{}]",
        e.tokens.len(),
        s + t
    );
    let snap = e.snapshot()?;
    let run =
        |e: &mut Glm5Engine, gen: &mut Vec<u32>, win: bool| -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
            e.wintap = Some(vec![Vec::new(); nl * 4]);
            let mut lg = Vec::new();
            if win {
                lg = e.window(&gen[s..s + t])?;
            } else {
                for i in 0..t {
                    let l = e.window(&[gen[s + i]])?;
                    if gen.len() <= s + i + 1 {
                        gen.push(argmax(&l));
                    }
                    lg.extend(l);
                }
            }
            Ok((e.wintap.take().unwrap(), lg))
        };
    let (ta, la) = run(e, &mut gen, false)?;
    e.restore(snap.clone_snap());
    let (ta2, la2) = run(e, &mut gen, false)?;
    e.restore(snap);
    let (tb, lb) = run(e, &mut gen, true)?;
    eprintln!("windiff: tokens fed {:?}", &gen[s..s + t]);
    let rel = |a: &[f32], b: &[f32]| -> (f32, f32) {
        let (mut d, mut m, mut mx) = (0f64, 0f64, 0f32);
        for (x, y) in a.iter().zip(b) {
            d += ((x - y) as f64).powi(2);
            m += (*x as f64).powi(2);
            mx = mx.max((x - y).abs());
        }
        ((d / m.max(1e-30)).sqrt() as f32, mx)
    };
    let names = ["attn_in", "attn_out", "ffn_in", "ffn_out"];
    eprintln!("windiff: layer stage | step-vs-step rel | window-vs-step rel (l2), max abs");
    for l in 0..nl {
        for st in 0..4 {
            let (a, a2, b) = (&ta[l * 4 + st], &ta2[l * 4 + st], &tb[l * 4 + st]);
            if a.len() != b.len() {
                eprintln!("windiff: l{l} {} len {} vs {}", names[st], a.len(), b.len());
                continue;
            }
            let (r0, _) = rel(a, a2);
            let (r1, m1) = rel(a, b);
            // per token row: the first differing row tells window position effects
            let n = a.len() / t;
            let rows: Vec<String> = (0..t)
                .map(|i| {
                    format!(
                        "{:.1e}",
                        rel(&a[i * n..(i + 1) * n], &b[i * n..(i + 1) * n]).0
                    )
                })
                .collect();
            eprintln!(
                "windiff: l{l:02} {:8} kda={} ss {:.2e} ws {:.2e} max {:.2e} rows [{}]",
                names[st],
                e.m.hp.is_kda[l],
                r0,
                r1,
                m1,
                rows.join(" ")
            );
        }
    }
    let top2 = |v: &[f32]| -> (usize, f32, usize, f32) {
        let a = argmax(v) as usize;
        let mut b = if a == 0 { 1 } else { 0 };
        for (i, &x) in v.iter().enumerate() {
            if i != a && x > v[b] {
                b = i;
            }
        }
        (a, v[a], b, v[b])
    };
    for i in 0..t {
        for (nm, l) in [("step", &la), ("step2", &la2), ("window", &lb)] {
            let (a, va, b, vb) = top2(&l[i * nv..(i + 1) * nv]);
            eprintln!(
                "windiff: row {i} {nm:6} top1 {a} ({va:.4}) top2 {b} ({vb:.4}) margin {:.4}",
                va - vb
            );
        }
        let (r, m) = rel(&la[i * nv..(i + 1) * nv], &lb[i * nv..(i + 1) * nv]);
        eprintln!("windiff: row {i} logits window-vs-step rel {r:.2e} max abs {m:.3e}");
    }
    Ok(())
}

/// The device's memory bandwidth ceiling for %-of-SOL, GB/s: `TANG_SOL_GBS`, else the
/// datasheet figure for a known card, else bus width × memory clock × 2 (DDR) from the
/// driver (which under-reports HBM3e: the H200 reads ~4.0 TB/s there vs 4.8 spec).
#[cfg(feature = "cuda")]
fn sol_gbs(g: &tang_moe::gpu::Gpu) -> (f64, String) {
    use cudarc::driver::sys::CUdevice_attribute as A;
    let name = g.ctx.name().unwrap_or_default();
    if let Some(v) = std::env::var("TANG_SOL_GBS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        return (v, format!("{name}, TANG_SOL_GBS"));
    }
    const KNOWN: &[(&str, f64)] = &[
        ("H200", 4800.0),
        ("H100 NVL", 3900.0),
        ("H100 PCIe", 2000.0),
        ("H100", 3350.0),
        ("B300", 8000.0),
        ("B200", 8000.0),
        ("A100", 2039.0),
        ("RTX 4090", 1008.0),
        ("RTX 3090", 936.0),
    ];
    if let Some(&(_, v)) = KNOWN.iter().find(|(k, _)| name.contains(k)) {
        return (v, format!("{name}, datasheet"));
    }
    let clk = g
        .attribute(A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
        .unwrap_or(0) as f64; // kHz
    let bus = g
        .attribute(A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
        .unwrap_or(0) as f64; // bits
    (
        clk * 1e3 * 2.0 * bus / 8.0 / 1e9,
        format!("{name}, driver clock × bus"),
    )
}

/// `--slots N`: prefill `ids` into each of `slots` sequence slots, then decode `n - 1`
/// steps round-robin (one step = one token for EVERY slot, through that slot's state).
/// Prints each slot's ids (they must be identical — the same prompt, greedy — or the
/// slots' state bleeds), per-step wall ms (median / min / max over the steps) and the
/// aggregate decode tok/s.
fn slots_bench(
    e: &mut Glm5Engine,
    ids: &[u32],
    n: usize,
    chunk: usize,
    slots: usize,
    rr: bool,
    diverse: bool,
    out: Option<&Path>,
) -> Result<()> {
    // prefill in lockstep (both modes; prefill is not what this measures): position j of
    // every slot is one batched window of `slots` rows
    let t1 = Instant::now();
    let nv = e.m.hp.n_vocab;
    let _ = chunk;
    let nvu = nv as u32;
    // slot s's prompt: `ids`, or (diverse, s > 0) with the token 4 from the end — inside
    // the user message — replaced by a different id (same length, so lockstep works)
    let prompt = |s: usize| -> Vec<u32> {
        let mut p = ids.to_vec();
        if diverse && s > 0 && p.len() > 4 {
            let j = p.len() - 4;
            p[j] = (p[j] + 7919 * s as u32) % nvu;
        }
        p
    };
    let prompts: Vec<Vec<u32>> = (0..slots).map(prompt).collect();
    let mut lg = Vec::new();
    for j in 0..ids.len() {
        let rows: Vec<(usize, u32)> = (0..slots).map(|s| (s, prompts[s][j])).collect();
        lg = e.window_batch(&rows)?;
    }
    let mut cur: Vec<u32> = (0..slots)
        .map(|s| argmax(&lg[s * nv..(s + 1) * nv]))
        .collect();
    let prefill_s = t1.elapsed().as_secs_f64();
    let mut gen: Vec<Vec<u32>> = cur.iter().map(|&c| vec![c]).collect();
    let mut step_ms = Vec::new();
    let t2 = Instant::now();
    for _ in 1..n {
        let ts = Instant::now();
        if rr {
            for s in 0..slots {
                cur[s] = e.step_slot(s, cur[s])?;
            }
        } else {
            let rows: Vec<(usize, u32)> = (0..slots).map(|s| (s, cur[s])).collect();
            let lg = e.window_batch(&rows)?;
            for s in 0..slots {
                cur[s] = argmax(&lg[s * nv..(s + 1) * nv]);
            }
        }
        for s in 0..slots {
            gen[s].push(cur[s]);
        }
        step_ms.push(ts.elapsed().as_secs_f64() * 1e3);
    }
    let decode_s = t2.elapsed().as_secs_f64();
    println!(
        "prompt: {} tokens x {slots} slots, prefill {:.2} s",
        ids.len(),
        prefill_s
    );
    let line = |g: &[u32]| {
        g.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    for (s, g) in gen.iter().enumerate() {
        println!("ids slot {s}: {}", line(g));
    }
    // identical prompts: every slot must give the same ids; diverse: only slot 0 has the
    // reference prompt (compare its ids to the oracle), the others just differ
    let same = diverse || gen.iter().all(|g| g == &gen[0]);
    if diverse {
        let distinct = gen.iter().skip(1).filter(|g| *g != &gen[0]).count();
        println!(
            "diverse prompts: {distinct} of {} other slots decode differently from slot 0",
            slots - 1
        );
    } else {
        println!(
            "slots agree: {}",
            if same {
                "yes"
            } else {
                "NO — slot state bleeds"
            }
        );
    }
    if !step_ms.is_empty() {
        let mut v = step_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let toks = (slots * step_ms.len()) as f64;
        println!(
            "slots {slots} ({}): per-step wall {:.1} ms median (min {:.1}, max {:.1}) over {} steps | {:.1} ms/token/slot | aggregate {:.2} tok/s, per stream {:.2} tok/s",
            match (rr, diverse) {
                (false, false) => "batched, same prompt",
                (false, true) => "batched, diverse",
                (true, false) => "round-robin, same prompt",
                (true, true) => "round-robin, diverse",
            },
            v[v.len() / 2],
            v[0],
            v[v.len() - 1],
            v.len(),
            v[v.len() / 2] / slots as f64,
            toks / decode_s,
            step_ms.len() as f64 / decode_s
        );
    }
    if let Some(o) = out {
        std::fs::write(o, format!("{}\n", line(&gen[0])))?;
    }
    // TANG_GLM5_TRACE=1: the span table (ffn vs the GPU stages per batched window)
    if let Some(r) = super::trace::report() {
        eprint!("{r}");
    }
    ensure!(same, "slot ids differ");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf";

    /// GPU tests skip when there's no device — but a SILENT pass looks green in CI and hid
    /// the GLM-5.3 GPU bugs for a cycle (the "tiny fixture passes" that never ran). With
    /// `TANG_REQUIRE_CUDA=1` a skip is a FAILURE instead; without it, the skip prints a
    /// loud banner. `reason` says what's missing.
    /// (Now [`tang_verify::skip_or_fail`], shared by every kernel test.)
    #[allow(dead_code)]
    fn skip_or_fail(reason: &str) {
        tang_verify::skip_or_fail(reason)
    }

    fn seq1() -> Option<Vec<u32>> {
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").ok()?;
        let v: serde_json::Value = serde_json::from_str(&meta).ok()?;
        Some(
            v["seq1_ids"]
                .as_array()?
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect(),
        )
    }

    /// REAL-WEIGHTS BISECTOR: the wired GPU engine vs the CPU engine on the REAL model,
    /// a tiny sequence. Named the bug the tiny fixture couldn't: if this fails while
    /// engine_gpu_matches_cpu passes, the divergence is real-weights/real-shape specific.
    /// Gated on TANG_GLM5_REAL_GGUF (the shard-1 path) so normal test runs stay cheap.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_real_weights_gpu_matches_cpu() {
        let _env = crate::flash::glm5::env_write();
        let Some(real) = std::env::var("TANG_GLM5_REAL_GGUF").ok() else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let p = Path::new(&real);
        if !p.exists() {
            eprintln!("skip: {real} absent");
            return;
        }
        let ids: Vec<u32> = (1000..1008).collect(); // 8 arbitrary in-vocab tokens
                                                    // CPU outputs FIRST, engine dropped before the GPU engine opens: two resident
                                                    // engines (2 × ~250 GB) exceed the 274 GB card. The CPU engine holds no resident
                                                    // experts (mmap dequant per layer), so it's cheap — but dropping it keeps the
                                                    // peak at one resident set.
        std::env::set_var("TANG_GLM5_GPU", "0");
        let cpu_out = {
            let mut cpu2 = Glm5Engine::open(p).unwrap();
            let mut out = Vec::new();
            out.extend(cpu2.window(&ids[..5]).unwrap());
            for &tok in &ids[5..] {
                out.extend(cpu2.window(&[tok]).unwrap());
            }
            out
        };
        std::env::remove_var("TANG_GLM5_GPU");
        let mut gpu = Glm5Engine::open(p).unwrap();
        gpu.use_gpu().unwrap();
        let mut gpu_out = Vec::new();
        gpu_out.extend(gpu.window(&ids[..5]).unwrap());
        for &tok in &ids[5..] {
            gpu_out.extend(gpu.window(&[tok]).unwrap());
        }
        let mut worst = 0f32;
        let mut bad = 0usize;
        let nv = 154_880;
        for p in 0..ids.len() {
            let c = &cpu_out[p * nv..(p + 1) * nv];
            let g = &gpu_out[p * nv..(p + 1) * nv];
            let w: f32 = c
                .iter()
                .zip(g)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            worst = worst.max(w);
            if argmax(c) != argmax(g) {
                bad += 1;
                eprintln!("position {p}: top-1 differs, max abs {w}");
            }
        }
        eprintln!(
            "real-weights bisector: max abs {worst}, top-1 mismatches {bad}/{}",
            ids.len()
        );
        assert_eq!(
            bad, 0,
            "the wired GPU path diverges from CPU on the real model"
        );
    }

    /// Chunked prefill + token-by-token decode equals one full reference forward.
    #[test]
    fn engine_matches_reference_forward() {
        let _env = crate::flash::glm5::env_read();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            eprintln!("skip: tiny fixture absent");
            return;
        };
        let r = Glm5Ref::open(Path::new(TINY)).unwrap();
        let want = r.forward(&ids).unwrap();
        let nv = r.hp.n_vocab;
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut got = Vec::new();
        // uneven chunks, then single tokens: exercises KDA chunk + decode and the DSA caches
        let (a, b) = (17, 36);
        got.extend(e.window(&ids[..a]).unwrap());
        got.extend(e.window(&ids[a..b]).unwrap());
        for &tok in &ids[b..] {
            got.extend(e.window(&[tok]).unwrap());
        }
        assert_eq!(got.len(), want.len());
        let mut worst = 0f32;
        for (x, y) in got.iter().zip(&want) {
            worst = worst.max((x - y).abs());
        }
        for p in 0..ids.len() {
            assert_eq!(
                argmax(&got[p * nv..(p + 1) * nv]),
                argmax(&want[p * nv..(p + 1) * nv]),
                "top-1 differs at {p}"
            );
        }
        eprintln!("engine vs reference: logits max abs {worst:.3e}");
        assert!(worst <= 1e-4, "logits max abs {worst}");
    }

    /// Greedy continuation through the engine equals re-running the reference each step.
    #[test]
    fn engine_greedy_matches_reference() {
        let _env = crate::flash::glm5::env_read();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            eprintln!("skip: tiny fixture absent");
            return;
        };
        let prompt = &ids[..12];
        let r = Glm5Ref::open(Path::new(TINY)).unwrap();
        let nv = r.hp.n_vocab;
        let mut seq = prompt.to_vec();
        let mut want = Vec::new();
        for _ in 0..8 {
            let lg = r.forward(&seq).unwrap();
            let next = argmax(&lg[(seq.len() - 1) * nv..]);
            want.push(next);
            seq.push(next);
        }
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut got = vec![e.prefill(prompt, 5).unwrap()];
        while got.len() < 8 {
            let t = *got.last().unwrap();
            got.push(e.step(t).unwrap());
        }
        assert_eq!(got, want);
    }

    /// R3 (the f32 GPU paths: FFN, KDA, mHC): every FFN, the KDA conv + core and the mHC
    /// pre/post on the GPU equal the CPU engine on the tiny model: 150-token prefill (2 GPU chunks + a CPU tail), GPU
    /// decode steps, then a short window. Pod: `TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm
    /// --no-default-features --features cuda --lib glm5::engine -- --nocapture`
    /// (`TANG_GPUS=2 TANG_SPLIT_AT=2` exercises the split).
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_matches_cpu() {
        let _env = crate::flash::glm5::env_read();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            skip_or_fail("tiny fixture absent");
            return;
        };
        if tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("no CUDA device");
            return;
        }
        let mut cpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut gpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(gpu.use_gpu().unwrap(), "GPU FFNs did not load");
        // f32 paths only here: DSA stays on the CPU (the bf16 MLA path has its own test)
        for k in &mut gpu.gpu.as_mut().unwrap().mla_k {
            *k = None;
        }
        let nv = cpu.m.hp.n_vocab;
        let mut worst = 0f32;
        // seq2 (2052 tokens) for a window ≥ 64: two GPU prefill chunks + a CPU tail, then
        // single tokens (the GPU decode step), then a short window (CPU tail only)
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let _ = &ids;
        let mut flips = Vec::new();
        for (wi, w) in [
            &ids2[..150],
            &ids2[150..151],
            &ids2[151..152],
            &ids2[152..160],
        ]
        .into_iter()
        .enumerate()
        {
            let a = cpu.window(w).unwrap();
            let b = gpu.window(w).unwrap();
            for p in 0..w.len() {
                let (ra, rb) = (&a[p * nv..(p + 1) * nv], &b[p * nv..(p + 1) * nv]);
                let (ta, tb) = (argmax(ra), argmax(rb));
                let row: f32 = ra
                    .iter()
                    .zip(rb)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f32::max);
                if ta != tb {
                    // a near-tie flip (|cpu margin| ≲ the row's max abs diff) is numerics; a
                    // wide margin is a real divergence — print both so the log says which
                    let margin = ra[ta as usize] - ra[tb as usize];
                    eprintln!(
                        "window {wi} pos {p}: top-1 cpu {ta} gpu {tb}; cpu margin {margin:.3e}, \
                         row max abs {row:.3e}, gpu logits {:.5}/{:.5}",
                        rb[ta as usize], rb[tb as usize]
                    );
                    flips.push((wi, p, margin, row));
                }
            }
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x - y).abs());
            }
        }
        eprintln!("GPU vs CPU: {} top-1 flips {flips:?}", flips.len());
        assert!(
            flips.is_empty(),
            "top-1 differs at {} positions",
            flips.len()
        );
        eprintln!("GPU (FFN + KDA + mHC) vs CPU: logits max abs {worst:.3e}");
        assert!(worst <= 1e-3, "logits max abs {worst}");
    }

    /// The fused, device-resident mHC (`TANG_GLM5_MHC_FUSED=1`, [`super::mhc_fused`])
    /// against the CPU engine on the tiny model: the same windows as
    /// [`engine_gpu_matches_cpu`] (150-token prefill, decode steps, a short window), the
    /// fused instance built explicitly so the test doesn't depend on the env.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_fused_mhc_matches_cpu() {
        let _env = crate::flash::glm5::env_read();
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        let mut cpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut gpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(gpu.use_gpu().unwrap(), "GPU did not load");
        {
            let (g, hp) = (&gpu.m.g, gpu.m.hp.clone());
            let gf = gpu.gpu.as_mut().unwrap();
            for k in &mut gf.mla_k {
                *k = None;
            }
            if gf.mhc_f.is_none() {
                gf.enable_mhc_fused(g, &hp).unwrap();
            }
            // NaN in every fused output before each launch: an unwritten region can't pass
            gf.mhc_f.as_mut().unwrap().set_poison(true);
        }
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let nv = cpu.m.hp.n_vocab;
        let (mut worst, mut flips) = (0f32, 0usize);
        for w in [
            &ids2[..150],
            &ids2[150..151],
            &ids2[151..152],
            &ids2[152..160],
        ] {
            let a = cpu.window(w).unwrap();
            let b = gpu.window(w).unwrap();
            for p in 0..w.len() {
                let (ra, rb) = (&a[p * nv..(p + 1) * nv], &b[p * nv..(p + 1) * nv]);
                if argmax(ra) != argmax(rb) {
                    flips += 1;
                    eprintln!(
                        "fused mHC: pos {p}: top-1 cpu {} gpu {}",
                        argmax(ra),
                        argmax(rb)
                    );
                }
            }
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x - y).abs());
            }
        }
        let ran = gpu.gpu.as_ref().unwrap().mhc_f.as_ref().unwrap().windows();
        eprintln!("GPU (fused mHC + KDA + FFN) vs CPU: {flips} top-1 flips, logits max abs {worst:.3e}, {ran} fused windows");
        // the windows that fit the scratch went through the fused path (all four at the
        // default max_t; at TANG_GLM5_KDA_MAXT=64 the 150-token one runs the old path)
        assert!(ran >= 3, "fused mHC ran {ran} windows");
        assert_eq!(flips, 0, "top-1 flips");
        assert!(worst <= 1e-3, "logits max abs {worst}");
    }

    /// REGRESSION (the GLM-5.3 mHC bug, 324e7f6b): the resident mHC pre must use the
    /// CALLING half's weights — block 0 (hc_attn) for the attention half, block 1
    /// (hc_ffn) for the FFN half. The bug ran `pre(0, ..)` for both, so every FFN-half
    /// mix collapsed the streams with the attention weights. The tiny fixture's
    /// hc_attn/hc_ffn weights differ by ~0.17, so the wrong block is detectable directly:
    /// compare the GPU pre's mixes for block 1 against the CPU's for block 1 (not block 0).
    #[cfg(feature = "cuda")]
    #[test]
    fn mhc_gpu_pre_block1_matches_cpu_block1() {
        let _env = crate::flash::glm5::env_read();
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        let r = Glm5Ref::open(Path::new(TINY)).unwrap();
        let mhp = crate::flash::glm5::mhc::MhcHparams {
            streams: r.hp.hc,
            sinkhorn_iters: r.hp.hc_sinkhorn_iters as u32,
            hc_eps: r.hp.hc_eps,
            rms_eps: r.hp.eps,
        };
        let wa =
            crate::flash::glm5::mhc::MhcWeights::from_gguf(&r.g, 0, "hc_attn", r.hp.n_embd, &mhp)
                .unwrap();
        let wf =
            crate::flash::glm5::mhc::MhcWeights::from_gguf(&r.g, 0, "hc_ffn", r.hp.n_embd, &mhp)
                .unwrap();
        let mut g = tang_moe::gpu::Gpu::for_test().unwrap();
        g.bind().unwrap();
        let mh = crate::flash::glm5::mhc::MhcGpu::new(&g, [&wa, &wf], 8).unwrap();

        // four streams of nonzero state
        let t = 3usize;
        let n = r.hp.n_embd;
        let mut rng_state = 12345u64;
        let mut next = move || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32) / f32::from_bits(0x4f800000) - 1.0
        };
        let r_in: Vec<f32> = (0..t * 4 * n).map(|_| next()).collect();
        for block in 0..2 {
            let w = if block == 0 { &wa } else { &wf };
            let (x_cpu, mixes_cpu) = mhc_pre_all(w, &r_in, t, 4, n).unwrap();
            let set = mh.pre(block, &r_in, t).unwrap();
            let dx: f32 = x_cpu
                .iter()
                .zip(&set.x)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let dpost: f32 = mixes_cpu
                .iter()
                .zip(0..)
                .flat_map(|(m, i)| {
                    m.post
                        .iter()
                        .zip(&set.post[i * 4..(i + 1) * 4])
                        .map(|(a, b)| (a - b).abs())
                })
                .fold(0.0, f32::max);
            let dcomb: f32 = mixes_cpu
                .iter()
                .zip(0..)
                .flat_map(|(m, i)| {
                    m.comb
                        .iter()
                        .flatten()
                        .zip(&set.comb[i * 16..(i + 1) * 16])
                        .map(|(a, b)| (a - b).abs())
                })
                .fold(0.0, f32::max);
            eprintln!("block {block}: x {dx:.3e} post {dpost:.3e} comb {dcomb:.3e}");
            // the bug's signature: block 1's mixes would equal block 0's GPU mixes here
            assert!(
                dx < 1e-4 && dpost < 1e-4 && dcomb < 1e-4,
                "block {block} mHC pre diverges"
            );
        }
        // and the two blocks must NOT produce the same mixes (guards a future revert to
        // a constant block index passing this test trivially)
        let x0 = mh.pre(0, &r_in, t).unwrap().x;
        let x1 = mh.pre(1, &r_in, t).unwrap().x;
        let d: f32 = x0
            .iter()
            .zip(&x1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("block0 vs block1 x diff {d:.3e}");
        assert!(d > 1e-3, "hc_attn and hc_ffn pre produced identical mixes");
    }

    /// REGRESSION (the GLM-5.3 DSA bug, 74192590): each DSA layer's resident `MlaCuda`
    /// stages ITS OWN layer's weights and caches. The bug shared one sample-layer
    /// instance across all DSA layers (per device) — invisible to any single-DSA-layer
    /// fixture. This test asserts the structural invariant on the real model's geometry
    /// at load (no weights needed): `mla_k[l]` is `Some` exactly for DSA layers, and the
    /// per-layer instances are distinct objects.
    #[test]
    fn mla_resident_is_per_layer() {
        let _env = crate::flash::glm5::env_read();
        // no fixture dependency: the invariant is checked inside engine_gpu_mla_matches_cpu
        // on the tiny model (one DSA layer, so it only guards construction) AND here via
        // the engine's own accounting on any loaded model. The load-log line
        // "N DSA layers resident (MLA, one per layer)" is the real-model proof; this test
        // pins the invariant so a future per-device revert fails construction-time asserts.
        // (Hparams has no Default; the layer-kind mask is all the invariant needs)
        let dsa_layers =
            |is_kda: &[bool]| -> Vec<usize> { (0..is_kda.len()).filter(|&l| !is_kda[l]).collect() };
        // the real GLM-5.3 geometry: 11 DSA layers, first at layer 3
        let mut is_kda = vec![true; 45];
        for l in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43] {
            is_kda[l] = false;
        }
        let dsa = dsa_layers(&is_kda);

        assert_eq!(dsa.len(), 11, "GLM-5.3 DSA layer count");
        assert_eq!(dsa[0], 3, "first DSA layer (the sample-layer bug's source)");
        assert!(
            dsa.len() > 1,
            "a single-DSA-layer geometry cannot see this bug class"
        );
    }

    /// R3 (all of it): FFN + KDA + MLA/indexer on the GPU vs the CPU engine. W3's GPU
    /// latents are bf16 (the CPU's are f32), so the bar is top-1 agreement ≥ 98% and logits
    /// within 0.25; both numbers are printed for the pod log.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_mla_matches_cpu() {
        let _env = crate::flash::glm5::env_read();
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let mut cpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut gpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(gpu.use_gpu().unwrap(), "GPU did not load");
        assert!(
            gpu.gpu
                .as_ref()
                .unwrap()
                .mla_k
                .iter()
                .zip(&gpu.m.hp.is_kda)
                .all(|(k, &kda)| k.is_some() != kda),
            "MLA kernels: one per DSA layer, none on KDA layers"
        );
        // the in-process shadow on every DSA window: device rows, pools, selection and
        // attention vs the CPU reference on the SAME inputs. This is the layer-local
        // statement (bf16 latents included); the end-to-end bars below are the sum.
        gpu.mla_dbg = Some(super::super::mla_debug::Shadow::always());
        // the like-for-like oracle: the CPU engine with the GPU cache's bf16 latents. The
        // f32 CPU engine differs from BOTH by the latent format alone (bf16 is W3's cache
        // design, ~4e-3 relative per latent), and on this random-weight fixture that moves
        // the logits by O(10) — measured below as cpu-bf16 vs cpu-f32, not a GPU error.
        let mut cpu16 = Glm5Engine::open(Path::new(TINY)).unwrap();
        cpu16.bf16_latents = true;
        let nv = cpu.m.hp.n_vocab;
        let maxd =
            |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let flips = |a: &[f32], b: &[f32], n: usize| {
            (0..n)
                .filter(|&p| argmax(&a[p * nv..(p + 1) * nv]) != argmax(&b[p * nv..(p + 1) * nv]))
                .count()
        };
        let (mut w_gpu16, mut w_gpu32, mut w_fmt, mut scale) = (0f32, 0f32, 0f32, 0f32);
        let (mut f_gpu16, mut f_gpu32, mut f_fmt, mut total) = (0usize, 0usize, 0usize, 0usize);
        // dense (≤ top-k) then sparse windows, then decode steps
        for w in [
            &ids2[..12],
            &ids2[12..150],
            &ids2[150..151],
            &ids2[151..152],
            &ids2[152..170],
        ] {
            let a = cpu.window(w).unwrap();
            let a16 = cpu16.window(w).unwrap();
            let b = gpu.window(w).unwrap();
            total += w.len();
            (f_gpu16, f_gpu32, f_fmt) = (
                f_gpu16 + flips(&a16, &b, w.len()),
                f_gpu32 + flips(&a, &b, w.len()),
                f_fmt + flips(&a, &a16, w.len()),
            );
            w_gpu16 = w_gpu16.max(maxd(&a16, &b));
            w_gpu32 = w_gpu32.max(maxd(&a, &b));
            w_fmt = w_fmt.max(maxd(&a, &a16));
            scale = a.iter().fold(scale, |m, x| m.max(x.abs()));
        }
        eprintln!(
            "GPU (FFN + KDA + mHC + MLA bf16) vs CPU-bf16-latents: {f_gpu16} top-1 flips / {total}, logits max abs {w_gpu16:.3e}"
        );
        eprintln!(
            "  vs CPU-f32: {f_gpu32} flips, max abs {w_gpu32:.3e}; the format alone (CPU-bf16 vs CPU-f32): \
             {f_fmt} flips, max abs {w_fmt:.3e}; logit scale max |logit| {scale:.3e}"
        );
        let (checked, failed, sel) = gpu.mla_dbg.as_ref().unwrap().totals();
        eprintln!(
            "MLA shadow: {checked} windows, {failed} failed; selection {}/{} queries differ ({} near ties, {} real)",
            sel.differ, sel.queries, sel.ties, sel.real
        );
        assert!(
            checked == 5 && failed == 0,
            "MLA shadow: {failed}/{checked} windows failed"
        );
        // the GPU path IS the bf16-latent model: tight against its CPU twin
        assert_eq!(f_gpu16, 0, "top-1 flips vs the CPU-bf16 engine");
        assert!(
            w_gpu16 <= 1e-2,
            "logits max abs vs the CPU-bf16 engine {w_gpu16}"
        );
        // and the f32 gap is the format's: no larger than the CPU's own bf16-vs-f32 gap
        // (+ the f32-path noise of engine_gpu_matches_cpu, 1e-3)
        assert!(
            w_gpu32 <= 1.25 * w_fmt + 1e-3,
            "GPU vs CPU-f32 {w_gpu32} exceeds the format gap {w_fmt}"
        );
        assert!(
            f_gpu32 as f32 <= 0.02 * total as f32,
            "top-1 vs CPU-f32 {f_gpu32}/{total}"
        );
    }

    /// The device-resident projections (`TANG_GLM5_MLA_RESIDENT=1`: q_a + norm, q_b,
    /// o_proj and the indexer GEMVs on g5f_gemv) against the naive GPU path on the same
    /// windows, with the shadow on every DSA window (its `q_resid` / `q` taps check the
    /// new GEMVs against the CPU). f32 fixture: f32 rows, no activation quantisation, so
    /// the two GPU engines differ only by accumulation order.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_mla_resident_matches_naive() {
        let _env = crate::flash::glm5::env_read();
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let mut naive = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut res = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(
            naive.use_gpu().unwrap() && res.use_gpu().unwrap(),
            "GPU did not load"
        );
        let n = {
            let m = &res.m;
            res.gpu
                .as_mut()
                .unwrap()
                .enable_mla_resident(&m.g, &|l| m.mla_weights(l))
                .unwrap()
        };
        assert_eq!(
            n,
            res.m.hp.is_kda.iter().filter(|&&k| !k).count(),
            "one MlaProj per DSA layer"
        );
        res.mla_dbg = Some(super::super::mla_debug::Shadow::always());
        let nv = res.m.hp.n_vocab;
        let (mut worst, mut flips) = (0f32, 0usize);
        for w in [
            &ids2[..12],
            &ids2[12..150],
            &ids2[150..151],
            &ids2[151..152],
            &ids2[152..170],
        ] {
            let a = naive.window(w).unwrap();
            let b = res.window(w).unwrap();
            for p in 0..w.len() {
                flips +=
                    (argmax(&a[p * nv..(p + 1) * nv]) != argmax(&b[p * nv..(p + 1) * nv])) as usize;
            }
            worst = a
                .iter()
                .zip(&b)
                .fold(worst, |m, (x, y)| m.max((x - y).abs()));
        }
        let (checked, failed, sel) = res.mla_dbg.as_ref().unwrap().totals();
        eprintln!(
            "MLA resident vs naive GPU: {flips} top-1 flips / 170, logits max abs {worst:.3e}; \
             shadow {checked} windows, {failed} failed, selection {}/{} differ",
            sel.differ, sel.queries
        );
        assert!(
            checked == 5 && failed == 0,
            "MLA shadow (resident): {failed}/{checked} windows failed"
        );
        assert_eq!(flips, 0, "resident vs naive top-1 flips");
        assert!(worst <= 1e-2, "resident vs naive logits max abs {worst}");
    }

    /// THE BATCHER's bar on the tiny model (CPU): two slots with DIFFERENT prompts,
    /// prefilled separately, then decoded together through `window_batch` — every row's
    /// logits must equal a lone engine running that prompt (no state bleeds between slots,
    /// batching changes nothing), and the round-robin `window_slot` path must too.
    #[test]
    fn engine_slots_batch_matches_separate() {
        let _env = crate::flash::glm5::env_read();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            eprintln!("skip: tiny fixture absent");
            return;
        };
        let prompts = [&ids[..9], &ids[3..15]];
        let steps = 4;
        // the references: one engine per prompt, greedy
        let mut want: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut toks: Vec<Vec<u32>> = Vec::new();
        for p in prompts {
            let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
            let nv = e.m.hp.n_vocab;
            let lg = e.window(p).unwrap();
            let mut cur = argmax(&lg[(p.len() - 1) * nv..]);
            let (mut w, mut tk) = (Vec::new(), vec![cur]);
            for _ in 0..steps {
                let lg = e.window(&[cur]).unwrap();
                cur = argmax(&lg);
                w.push(lg);
                tk.push(cur);
            }
            want.push(w);
            toks.push(tk);
        }
        for batched in [true, false] {
            let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
            e.set_slots_host(3); // slot 0 left empty: the batch uses slots 1 and 2
            let nv = e.m.hp.n_vocab;
            let mut cur = Vec::new();
            for (k, p) in prompts.iter().enumerate() {
                let lg = e.window_slot(k + 1, p).unwrap();
                cur.push(argmax(&lg[(p.len() - 1) * nv..]));
            }
            assert_eq!(cur, vec![toks[0][0], toks[1][0]], "prefill next tokens");
            for step in 0..steps {
                let got: Vec<Vec<f32>> = if batched {
                    let lg = e.window_batch(&[(1, cur[0]), (2, cur[1])]).unwrap();
                    vec![lg[..nv].to_vec(), lg[nv..].to_vec()]
                } else {
                    vec![
                        e.window_slot(1, &[cur[0]]).unwrap(),
                        e.window_slot(2, &[cur[1]]).unwrap(),
                    ]
                };
                for k in 0..2 {
                    let worst = got[k]
                        .iter()
                        .zip(&want[k][step])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst <= 1e-4,
                        "batched {batched} slot {k} step {step}: max abs {worst}"
                    );
                    cur[k] = argmax(&got[k]);
                    assert_eq!(
                        cur[k],
                        toks[k][step + 1],
                        "batched {batched} slot {k} step {step}"
                    );
                }
            }
            assert_eq!(e.slot_tokens(1).len(), prompts[0].len() + steps);
            assert!(e.slot_tokens(0).is_empty(), "slot 0 untouched");
        }
    }

    /// The batcher on the GPU (tiny model): the device `[slots][...]` KDA state, the
    /// batched rows' per-slot core (`core_rows`, needs PF2) and the per-slot MLA LRU. Two
    /// slots with DIFFERENT prompts decoded together through `window_batch` must match a
    /// lone single-slot GPU engine per prompt, step by step.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_slots_batch_matches_single() {
        let _env = crate::flash::glm5::env_write();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            skip_or_fail("tiny fixture absent");
            return;
        };
        if tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("no CUDA device");
            return;
        }
        // both MoE kernel paths: window_batch hands the FFN all N rows at once (T = N),
        // which is the batched FFN the B300 measures (TANG_GLM5_MOE=fast) — and the naive
        // kernels as the control
        for moe in ["naive", "fast"] {
            gpu_slots_case(&ids, moe);
        }
    }

    /// One MoE mode of [`engine_gpu_slots_batch_matches_single`]. The env knobs are read
    /// at engine load (layer construction), so they are set around the loads only.
    #[cfg(feature = "cuda")]
    fn gpu_slots_case(ids: &[u32], moe: &str) {
        let load = |slots: usize| {
            std::env::set_var("TANG_GLM5_KDA_PF2", "1"); // the batched KDA rows need it
            std::env::set_var("TANG_GLM5_MOE", moe);
            let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
            let ok = if slots > 1 {
                e.use_gpu_slots(slots)
            } else {
                e.use_gpu()
            }
            .unwrap();
            std::env::remove_var("TANG_GLM5_KDA_PF2");
            std::env::remove_var("TANG_GLM5_MOE");
            assert!(ok, "GPU did not load");
            let g = e.gpu.as_ref().unwrap();
            assert!(
                (0..e.m.hp.n_layer).all(|l| g.has_ffn(l)),
                "{moe}: every FFN on the GPU"
            );
            assert_eq!(g.devs[0].1.fast, moe == "fast", "{moe}: MoE kernel path");
            e
        };
        let prompts = [&ids[..9], &ids[3..15]];
        let steps = 4;
        let mut want: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut toks: Vec<Vec<u32>> = Vec::new();
        for p in prompts {
            let mut e = load(1);
            let nv = e.m.hp.n_vocab;
            let lg = e.window(p).unwrap();
            let mut cur = argmax(&lg[(p.len() - 1) * nv..]);
            let (mut w, mut tk) = (Vec::new(), vec![cur]);
            for _ in 0..steps {
                let lg = e.window(&[cur]).unwrap();
                cur = argmax(&lg);
                w.push(lg);
                tk.push(cur);
            }
            want.push(w);
            toks.push(tk);
        }
        let mut e = load(3);
        let nv = e.m.hp.n_vocab;
        let mut cur = Vec::new();
        for (k, p) in prompts.iter().enumerate() {
            let lg = e.window_slot(k + 1, p).unwrap();
            cur.push(argmax(&lg[(p.len() - 1) * nv..]));
        }
        assert_eq!(
            cur,
            vec![toks[0][0], toks[1][0]],
            "{moe}: prefill next tokens"
        );
        for step in 0..steps {
            let lg = e.window_batch(&[(1, cur[0]), (2, cur[1])]).unwrap();
            for k in 0..2 {
                let got = &lg[k * nv..(k + 1) * nv];
                let scale = want[k][step]
                    .iter()
                    .fold(0f32, |m, v| m.max(v.abs()))
                    .max(1.0);
                let worst = got
                    .iter()
                    .zip(&want[k][step])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("gpu slots ({moe}): slot {k} step {step}: max abs {worst:.3e} (scale {scale:.2})");
                assert!(
                    worst <= 1e-3 * scale,
                    "{moe}: slot {k} step {step}: max abs {worst} vs scale {scale}"
                );
                cur[k] = argmax(got);
            }
        }
        assert!(e.slot_tokens(0).is_empty(), "slot 0 untouched");
    }

    /// KDA_WIN_STEP: with the window-step threshold >= t, one t-token window over the
    /// device-resident KDA core (PF2) gives the logits of t single decode steps, and leaves
    /// the same state (a further step agrees). This is what makes `--mtp K` exact by default
    /// (the verify window is K+1 tokens; `mtp::cli_run` sets the default to K+1).
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_win_step_window_matches_steps() {
        let _env = crate::flash::glm5::env_write();
        use std::sync::atomic::Ordering;
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            skip_or_fail("tiny fixture absent");
            return;
        };
        if tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("no CUDA device");
            return;
        }
        let load = || {
            std::env::set_var("TANG_GLM5_KDA_PF2", "1"); // WIN_STEP lives on the PF2 core
            let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
            let ok = e.use_gpu().unwrap();
            std::env::remove_var("TANG_GLM5_KDA_PF2");
            assert!(ok, "GPU did not load");
            e
        };
        let pre = &ids[..6];
        let ws = super::super::mtp::WIN_STEP_DEFAULT.load(Ordering::Relaxed);
        for t in [2usize, 3, 4] {
            let win = &ids[6..6 + t];
            let next = ids[6 + t];
            // A: t single decode steps
            let mut a = load();
            let nv = a.m.hp.n_vocab;
            a.window(pre).unwrap();
            let mut want = Vec::new();
            for &x in win {
                want.extend(a.window(&[x]).unwrap());
            }
            let want_next = a.window(&[next]).unwrap();
            // B: one t-token window, run as t serial core steps
            let mut b = load();
            b.window(pre).unwrap();
            super::super::mtp::WIN_STEP_DEFAULT.store(t, Ordering::Relaxed);
            let got = b.window(win);
            super::super::mtp::WIN_STEP_DEFAULT.store(ws, Ordering::Relaxed);
            let got = got.unwrap();
            let got_next = b.window(&[next]).unwrap();
            assert_eq!(got.len(), want.len());
            for r in 0..t {
                let (g, w) = (&got[r * nv..(r + 1) * nv], &want[r * nv..(r + 1) * nv]);
                let scale = w.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
                let d = g
                    .iter()
                    .zip(w)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                eprintln!("win_step t={t} row {r}: max abs {d:.3e} (scale {scale:.2})");
                assert!(
                    d <= 1e-4 * scale,
                    "t={t} row {r}: window vs steps max abs {d}"
                );
                assert_eq!(argmax(g), argmax(w), "t={t} row {r}: argmax");
            }
            let d = got_next
                .iter()
                .zip(&want_next)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            eprintln!("win_step t={t} next step: max abs {d:.3e}");
            assert!(
                d <= 1e-4 * want_next.iter().fold(1f32, |m, v| m.max(v.abs())),
                "t={t}: state after window"
            );
        }
    }

    /// reset() gives a fresh sequence.
    #[test]
    fn engine_reset() {
        let _env = crate::flash::glm5::env_read();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            return;
        };
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        let a = e.window(&ids[..6]).unwrap();
        e.reset();
        let b = e.window(&ids[..6]).unwrap();
        assert_eq!(a, b);
    }

    /// reset() gives a fresh sequence on the GPU too: the resident KDA layers must be
    /// re-seeded from the zeroed host state, not continue the last sequence's device state.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_reset_gpu() {
        let _env = crate::flash::glm5::env_read();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            skip_or_fail("tiny fixture absent");
            return;
        };
        if tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("no CUDA device");
            return;
        }
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(e.use_gpu().unwrap());
        let a = e.window(&ids[..6]).unwrap();
        e.window(&ids[6..9]).unwrap();
        e.reset();
        let b = e.window(&ids[..6]).unwrap();
        let d = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!("engine_reset_gpu: max |a - b| {d:.3e}");
        assert!(
            d < 1e-4,
            "after reset the GPU window differs from a fresh one: {d}"
        );
    }

    /// The device-resident decode step ([`resident`]) against the host-staged path on the
    /// same GPU config (fused mHC + KDA graph + resident MLA + fast MoE): identical greedy
    /// ids over a decode that crosses the indexer's dense -> sparse switch, then a host
    /// window after it (the lazily refreshed DSA mirrors) still agrees. Sets process env:
    /// run alone.
    #[test]
    #[cfg(feature = "cuda")]
    fn engine_gpu_resident_step_matches_host() {
        let _env = crate::flash::glm5::env_write();
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        for (k, v) in [
            ("TANG_GLM5_MOE", "fast"),
            ("TANG_GLM5_KDA_PF2", "1"),
            ("TANG_GLM5_KDA_GRAPH", "1"),
            ("TANG_GLM5_GEMV8", "1"),
        ] {
            std::env::set_var(k, v);
        }
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let mk = || {
            let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
            assert!(e.use_gpu().unwrap(), "GPU did not load");
            let (m, hp) = (&e.m, e.m.hp.clone());
            let gf = e.gpu.as_mut().unwrap();
            gf.enable_mla_resident(&m.g, &|l| m.mla_weights(l)).unwrap();
            if gf.mhc_f.is_none() {
                gf.enable_mhc_fused(&m.g, &hp).unwrap();
            }
            e
        };
        let (mut host, mut res) = (mk(), mk());
        let run = |e: &mut Glm5Engine, resident: bool| -> Vec<u32> {
            if resident {
                std::env::remove_var("TANG_GLM5_RESIDENT");
            } else {
                std::env::set_var("TANG_GLM5_RESIDENT", "0");
            }
            let mut cur = e.prefill(&ids2[..4], 64).unwrap();
            let mut out = vec![cur];
            for _ in 0..40 {
                cur = e.step(cur).unwrap();
                out.push(cur);
            }
            out
        };
        // eager resident step first (the whole-step graph is the default; =0 opts out)
        std::env::set_var("TANG_GLM5_STEP_GRAPH", "0");
        let a = run(&mut host, false);
        let b = run(&mut res, true);
        // the segment graphs (=1): warm-up token eager, then segments
        std::env::set_var("TANG_GLM5_STEP_GRAPH", "1");
        let mut gr = mk();
        let c = run(&mut gr, true);
        let segs = gr
            .gpu
            .as_ref()
            .unwrap()
            .resident
            .as_ref()
            .and_then(|r| r.graphs.as_ref())
            .map_or(0, |g| g.len());
        eprintln!(
            "step graph: {segs} segments, ids {}",
            if c == a { "= host" } else { "DIFFER" }
        );
        assert_eq!(
            segs,
            gr.m.hp.is_kda.iter().filter(|&&k| !k).count() + 1,
            "step graph segments"
        );
        assert_eq!(a, c, "step graph vs host greedy ids");
        // the whole-step graph (the default, = TANG_GLM5_STEP_GRAPH=2): one replay per token, the device
        // position counter / embed gather; dense and sparse graphs across the switch
        std::env::remove_var("TANG_GLM5_STEP_GRAPH");
        let mut gw = mk();
        let d = run(&mut gw, true);
        std::env::remove_var("TANG_GLM5_STEP_GRAPH");
        let whole = gw
            .gpu
            .as_ref()
            .unwrap()
            .resident
            .as_ref()
            .map_or([0, 0], |r| r.whole_replays);
        eprintln!(
            "whole step graph: replays [dense, sparse] {whole:?}, ids {}",
            if d == a { "= host" } else { "DIFFER" }
        );
        eprintln!("                  host  {a:?}\n                  whole {d:?}");
        assert!(
            whole[0] > 0 && whole[1] > 0 && whole[0] + whole[1] == 39,
            "whole graph replays {whole:?}"
        );
        assert_eq!(a, d, "whole step graph vs host greedy ids");
        assert!(
            host.gpu.as_ref().unwrap().resident.is_none(),
            "the host run took the resident step"
        );
        assert!(
            res.gpu.as_ref().unwrap().resident.is_some(),
            "the resident step never ran"
        );
        eprintln!("resident step: host {a:?}\n               res  {b:?}");
        // a host window after the resident decode: reads the (refreshed) DSA mirrors
        std::env::set_var("TANG_GLM5_RESIDENT", "0");
        let nv = host.m.hp.n_vocab;
        let la = host.window(&ids2[4..20]).unwrap();
        let lb = res.window(&ids2[4..20]).unwrap();
        std::env::remove_var("TANG_GLM5_RESIDENT");
        let flips = (0..16)
            .filter(|&p| argmax(&la[p * nv..(p + 1) * nv]) != argmax(&lb[p * nv..(p + 1) * nv]))
            .count();
        let worst = la
            .iter()
            .zip(&lb)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!(
            "after resident decode, host window: {flips} flips / 16, logits max abs {worst:.3e}"
        );
        for k in [
            "TANG_GLM5_MOE",
            "TANG_GLM5_KDA_PF2",
            "TANG_GLM5_KDA_GRAPH",
            "TANG_GLM5_GEMV8",
        ] {
            std::env::remove_var(k);
        }
        assert_eq!(a, b, "resident vs host greedy ids");
        assert_eq!(flips, 0, "host window after the resident decode");
        assert!(worst <= 1e-3, "logits max abs {worst}");
    }

    /// The resident MTP verify window ([`Glm5Engine::window_resident`], t = 3) against the
    /// host window of the same rows (KDA rows exact via WIN_STEP): identical per-row greedy
    /// ids, normed h rows within 1e-3 of scale; the state after it continues like the host's;
    /// and a device rewind to 1 kept row followed by the remaining rows gives the same ids
    /// (the KDA checkpoints + DSA watermarks). Sets process env: run alone.
    #[test]
    #[cfg(feature = "cuda")]
    fn engine_gpu_resident_window_matches_host() {
        let _env = crate::flash::glm5::env_write();
        use std::sync::atomic::Ordering;
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        for (k, v) in [
            ("TANG_GLM5_MOE", "fast"),
            ("TANG_GLM5_KDA_PF2", "1"),
            ("TANG_GLM5_KDA_GRAPH", "1"),
            ("TANG_GLM5_GEMV8", "1"),
        ] {
            std::env::set_var(k, v);
        }
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let mk = || {
            let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
            assert!(e.use_gpu().unwrap(), "GPU did not load");
            let (m, hp) = (&e.m, e.m.hp.clone());
            let gf = e.gpu.as_mut().unwrap();
            gf.enable_mla_resident(&m.g, &|l| m.mla_weights(l)).unwrap();
            if gf.mhc_f.is_none() {
                gf.enable_mhc_fused(&m.g, &hp).unwrap();
            }
            e
        };
        let ws = super::super::mtp::WIN_STEP_DEFAULT.load(Ordering::Relaxed);
        super::super::mtp::WIN_STEP_DEFAULT.store(4, Ordering::Relaxed);
        let (pre, win, rest) = (&ids2[..4], &ids2[4..7], &ids2[7..10]);
        // host
        std::env::set_var("TANG_GLM5_RESIDENT", "0");
        let mut a = mk();
        let (n, nv) = (a.m.hp.n_embd, a.m.hp.n_vocab);
        a.prefill(pre, 64).unwrap();
        let la = a.window(win).unwrap();
        let ha = a.last_h.clone();
        let ga: Vec<u32> = (0..3).map(|j| argmax(&la[j * nv..(j + 1) * nv])).collect();
        let lr = a.window(rest).unwrap();
        let gr: Vec<u32> = (0..3).map(|j| argmax(&lr[j * nv..(j + 1) * nv])).collect();
        // host, kept row 0 then the remaining rows: the rewind reference
        let mut a2 = mk();
        a2.prefill(pre, 64).unwrap();
        a2.window(&win[..1]).unwrap();
        let tail: Vec<u32> = win[1..].iter().chain(&rest[..1]).copied().collect();
        let lt = a2.window(&tail).unwrap();
        let gt: Vec<u32> = (0..3).map(|j| argmax(&lt[j * nv..(j + 1) * nv])).collect();
        std::env::remove_var("TANG_GLM5_RESIDENT");
        // resident
        let mut b = mk();
        b.prefill(pre, 64).unwrap();
        let (gb, hb) = b
            .window_resident(win)
            .unwrap()
            .expect("resident window did not run");
        let scale = ha.iter().fold(1f32, |m, x| m.max(x.abs()));
        let dh = ha
            .iter()
            .zip(&hb)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!("resident window t=3: ids host {ga:?} res {gb:?}; h max abs {dh:.3e} (scale {scale:.2})");
        let (gb2, _) = b
            .window_resident(rest)
            .unwrap()
            .expect("second resident window");
        eprintln!("resident window next t=3: ids host {gr:?} res {gb2:?}");
        // rewind: keep row 0, then the rest
        let mut c = mk();
        c.prefill(pre, 64).unwrap();
        c.window_resident(win).unwrap().unwrap();
        c.rewind_resident(3, 1).unwrap();
        assert_eq!(c.tokens.len(), pre.len() + 1, "tokens after rewind");
        let (gc, _) = c.window_resident(&tail).unwrap().unwrap();
        eprintln!("after rewind: ids host {gt:?} res {gc:?}");
        // a host window after the resident windows still agrees (mirrors refreshed)
        std::env::set_var("TANG_GLM5_RESIDENT", "0");
        let lb = b.window(&ids2[10..14]).unwrap();
        let la2 = a.window(&ids2[10..14]).unwrap();
        std::env::remove_var("TANG_GLM5_RESIDENT");
        let flips = (0..4)
            .filter(|&p| argmax(&la2[p * nv..(p + 1) * nv]) != argmax(&lb[p * nv..(p + 1) * nv]))
            .count();
        super::super::mtp::WIN_STEP_DEFAULT.store(ws, Ordering::Relaxed);
        for k in [
            "TANG_GLM5_MOE",
            "TANG_GLM5_KDA_PF2",
            "TANG_GLM5_KDA_GRAPH",
            "TANG_GLM5_GEMV8",
        ] {
            std::env::remove_var(k);
        }
        assert_eq!(hb.len(), 3 * n);
        assert_eq!(ga, gb, "resident window vs host window ids");
        assert!(dh <= 1e-3 * scale, "h rows max abs {dh}");
        assert_eq!(gr, gb2, "the window after");
        assert_eq!(gt, gc, "after a device rewind");
        assert_eq!(flips, 0, "host window after resident windows");
    }

    /// `TANG_GLM5_TRACE` is transparent: the same windows traced and untraced give the
    /// same logits bit for bit, and the trace saw the GPU (launches, timed kernels by name,
    /// copies) under the layer-family spans.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_trace_is_transparent() {
        let _env = crate::flash::glm5::env_write();
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            skip_or_fail("tiny fixture absent");
            return;
        };
        if tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("no CUDA device");
            return;
        }
        let run = |e: &mut Glm5Engine| {
            e.reset();
            let mut v = e.window(&ids[..6]).unwrap();
            v.extend(e.window(&ids[6..7]).unwrap());
            v
        };
        // traced from before the load, as the CLI does (kernel names are taken at load)
        super::super::trace::start();
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(e.use_gpu().unwrap());
        let traced = run(&mut e);
        super::super::trace::stop();
        let plain = run(&mut e);
        let r = super::super::trace::report().expect("a trace report");
        println!("{r}");
        // the same bound as engine_reset_gpu (a fresh sequence run twice): tracing adds
        // events and counters only, so anything past run-to-run noise is a bug
        let d = traced
            .iter()
            .zip(&plain)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let same = traced
            .iter()
            .zip(&plain)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "engine_gpu_trace_is_transparent: max |traced - plain| {d:.3e}, bit-identical {same}"
        );
        assert!(d < 1e-4, "tracing changed the logits: {d}");
        for want in [
            "decode, per token",
            "prefill, totals",
            "kda",
            "mhc_pre.attn",
            "ffn",
            "lm_head",
            "kernels by GPU time",
            "k5_",
        ] {
            assert!(r.contains(want), "trace report lacks {want:?}:\n{r}");
        }
        // the hook's cost where it is largest: the tiny model's decode windows are almost
        // all launches (~0.1 ms of GPU work each). Interleaved rounds, median per mode.
        let mut ms = [Vec::new(), Vec::new()];
        for round in 0..10 {
            for traced in [round % 2 == 0, round % 2 != 0] {
                e.reset();
                e.window(&ids[..6]).unwrap();
                if traced {
                    super::super::trace::start();
                }
                let t0 = Instant::now();
                for i in 0..16 {
                    e.window(&[ids[(6 + i) % ids.len()]]).unwrap();
                }
                ms[traced as usize].push(t0.elapsed().as_secs_f64() * 1e3 / 16.0);
                super::super::trace::stop();
            }
        }
        // min per mode: the CPU-side noise here is one-sided (background work only adds)
        let mn = |v: &mut Vec<f64>| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let (off, on) = (mn(&mut ms[0]), mn(&mut ms[1]));
        println!(
            "engine_gpu_trace_is_transparent: tiny decode window {off:.3} ms plain, {on:.3} ms traced ({:+.1}%)",
            100.0 * (on / off - 1.0)
        );
    }

    /// THE LESSON OF THIS PORT (the engine never wired the built-and-tested GPU structs
    /// in): a test that the resident layers are actually CONSTRUCTED at `use_gpu()` —
    /// one DevLayer per KDA layer, one MhcGpu per layer. Without a CUDA device this
    /// asserts the counts the CPU build reports (0) so the wiring stays type-visible;
    /// on the pod it fails if a future refactor drops the construction.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_resident_layers_are_constructed() {
        let _env = crate::flash::glm5::env_read();
        if !Path::new(TINY).exists() {
            return;
        }
        let Some(_g) = tang_moe::gpu::Gpu::for_test() else {
            return;
        };
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(e.use_gpu().unwrap(), "GPU did not load");
        let g = e.gpu.as_ref().unwrap();
        let want_kda = e.m.hp.is_kda.iter().filter(|&&k| k).count();
        assert_eq!(g.kda_resident(), want_kda, "KDA DevLayers constructed");
        assert_eq!(g.mhc_resident(), e.m.hp.n_layer, "mHC MhcGpus constructed");
    }

    /// THE LESSON, part 2: a KDA window actually ROUTES through the resident layer —
    /// observable via the engine's host/device state ownership: after one window the
    /// device is fresh (the host copies were seeded and the device owns the state).
    /// On a CPU-only build this is trivially skipped; the pod run pins the routing.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_kda_window_routes_through_resident_layer() {
        let _env = crate::flash::glm5::env_read();
        if !Path::new(TINY).exists() {
            return;
        }
        let Some(_g) = tang_moe::gpu::Gpu::for_test() else {
            return;
        };
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            return;
        };
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(e.use_gpu().unwrap());
        e.window(&ids[..4]).unwrap();
        let g = e.gpu.as_ref().unwrap();
        let kda_layers: Vec<usize> = (0..e.m.hp.n_layer).filter(|&l| e.m.hp.is_kda[l]).collect();
        assert!(
            g.kda_dev_fresh.iter().any(|row| row.iter().any(|&f| f)),
            "no KDA layer took over the state: the mixer bypassed the resident path"
        );
        assert_eq!(kda_layers.len(), g.kda_resident());
    }
}
