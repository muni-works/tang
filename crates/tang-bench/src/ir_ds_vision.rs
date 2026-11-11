//! DeepSeek-V4.1-Flash image input: host preprocessing, the vision encoder
//! (ViT + aligner) as a one-shot tang graph, and the splice of its rows into
//! the DS prefill ([`crate::ir_ds_prefill`] with `DsDims::vision`).
//!
//! Transcribed from the checkpoint's `inference/` code
//! (`deepseek-ai/DeepSeek-V4.1-Flash-1017`): `image_processor.py`,
//! `vision.py` and `model.py`'s `merge_image_embeddings`.
//!
//! - **Preprocessing** ([`plan_image_grid`], [`preprocess`]): an image of
//!   `W × H` is scaled up to at least `vision_min_pixels`, rounded up to
//!   multiples of the 14-pixel patch, shrunk if its token span would pass
//!   `vision_max_n_token`, then letterboxed into that size with PIL's
//!   `ImageOps.pad` (bicubic, gray 127, centered; [`pil_resize`] is PIL's
//!   fixed-point 8-bit resampler). Pixels go to `(x/255 − 0.5)/0.5`, rounded
//!   to bf16 like the official loader, then to patches `[N, 3·14·14]` in
//!   `(c, ky, kx)` order, row-major over the `n_h × n_w` patch grid.
//! - **2D RoPE**: per patch `(h, w)` the 32 angles `[h·f_j, w·f_j]`,
//!   `f_j = θ^(−2j/32)`, `j < 16`; they rotate each 64-wide head as two
//!   halves (`x1, x2 = chunk(2)`, not interleaved pairs).
//! - **Encoder** ([`encoder_graph`]): `patch_embed` (a linear with bias), 32
//!   pre-norm blocks (RMSNorm eps 1e-6, fused `wqkv` with bias, full
//!   bidirectional softmax attention, `wo` with bias, SwiGLU `w1 → [gate,
//!   up]`, `w2`), a final RMSNorm; then the aligner: the grid zero-padded to
//!   multiples of 3, every 3×3 block unfolded to `C·9` features (`c·9 + ky·3
//!   + kx`), `w1` (+bias) → exact-erf GELU → `w2` (+bias) to the LLM width.
//! - **Prompt**: each image placeholder becomes `[START] + ([IMAGE]·n_w' +
//!   [NEWLINE])·n_h' + [END]`, every position carrying `image_token_id`;
//!   IMAGE slots take the aligner rows in reading order, the delimiters the
//!   learned `image_start` / `image_newline` / `image_end` rows. Every span
//!   position routes with `ffn.gate.bias_vl` and has no Engram n-gram.
//!
//! [`EmbedCache`] keys encoder output by the SHA-256 of the image bytes, so a
//! repeated image (a multi-turn chat resending it) skips the encoder.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tang::Scalar;
use tang_compile::tensor::{Arg, Dim, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use crate::ir_ds_prefill::{arg, form, fx, ident, konst, lit, select, unit, PrefillInput};

/// Token types of an image span (`image_processor.py`); text is -1.
pub const TEXT: i8 = -1;
pub const IMAGE_START: i8 = 0;
pub const IMAGE: i8 = 1;
pub const IMAGE_NEW_LINE: i8 = 2;
pub const IMAGE_END: i8 = 3;

/// The vision tower's config (`inference/config.json`'s `vision_*`).
#[derive(Clone, Debug)]
pub struct VisionDims {
    pub dim: usize,
    pub n_heads: usize,
    pub inter: usize,
    pub n_layers: usize,
    pub patch: usize,
    pub rope_theta: f64,
    pub downsample: usize,
    pub max_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f64>,
    /// The language model's width (the aligner's output).
    pub llm_dim: usize,
    pub image_token_id: usize,
    pub eps: f64,
}

impl VisionDims {
    pub fn from_config(c: &serde_json::Value) -> Option<VisionDims> {
        let u = |k: &str| c[k].as_u64().map(|v| v as usize);
        let n_layers = u("vision_n_layers")?;
        (n_layers > 0).then(|| VisionDims {
            dim: u("vision_dim").unwrap(),
            n_heads: u("vision_n_heads").unwrap(),
            inter: u("vision_inter_dim").unwrap(),
            n_layers,
            patch: u("vision_patch_size").unwrap(),
            rope_theta: c["vision_rope_theta"].as_f64().unwrap_or(10000.0),
            downsample: u("vision_downsample_ratio").unwrap(),
            max_tokens: u("vision_max_n_token").unwrap(),
            min_pixels: u("vision_min_pixels").unwrap(),
            max_wh_ratio: c["vision_max_wh_ratio"].as_f64(),
            llm_dim: u("dim").unwrap(),
            image_token_id: u("image_token_id").unwrap(),
            eps: 1e-6,
        })
    }

    /// A small config for the graph-vs-reference checks on random weights.
    pub fn tiny(n_layers: usize) -> VisionDims {
        VisionDims {
            dim: 16,
            n_heads: 2,
            inter: 12,
            n_layers,
            patch: 2,
            rope_theta: 10000.0,
            downsample: 3,
            max_tokens: 64,
            min_pixels: 16,
            max_wh_ratio: None,
            llm_dim: 24,
            image_token_id: 7,
            eps: 1e-6,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// RoPE angles per patch: `head_dim / 2` (h half, then w half).
    pub fn rope_dim(&self) -> usize {
        self.head_dim() / 2
    }

    pub fn patch_len(&self) -> usize {
        3 * self.patch * self.patch
    }
}

// ---------------------------------------------------------------- planning

/// Where an image lands: its ViT patch grid, its LLM token grid and the
/// pixel size it is letterboxed into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagePlan {
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub best_h: usize,
    pub best_w: usize,
}

impl ImagePlan {
    pub fn patches(&self) -> usize {
        self.n_vit_h * self.n_vit_w
    }
    /// Aligner rows (IMAGE slots).
    pub fn rows(&self) -> usize {
        self.n_llm_h * self.n_llm_w
    }
    /// Prompt positions of the whole span.
    pub fn span(&self) -> usize {
        num_image_tokens(self.n_llm_h, self.n_llm_w)
    }
}

pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

fn llm_grid(best_h: usize, best_w: usize, p: usize, r: usize) -> (usize, usize) {
    ((best_h / p).div_ceil(r), (best_w / p).div_ceil(r))
}

/// `solve_resize_ratio`: the largest aspect-preserving size whose span fits.
fn solve_resize_ratio(h: f64, w: f64, p: usize, r: usize, max_tok: usize) -> (usize, usize) {
    let ratio = h / w;
    let max_w = (((max_tok - 2) as f64) / ratio + 0.25).sqrt() - 0.5;
    let max_h = max_w * ratio;
    let cell = p * r;
    if max_w < 1.0 {
        return ((max_tok - 2) / 2 * cell, cell);
    }
    if max_h < 1.0 {
        return (cell, (max_tok - 3) * cell);
    }
    let beta = (max_w.floor() * cell as f64 / w).min(max_h.floor() * cell as f64 / h);
    let pf = p as f64;
    (
        ((h * beta / pf).floor() * pf) as usize,
        ((w * beta / pf).floor() * pf) as usize,
    )
}

/// `plan_image_grid` for an image of `width × height` pixels.
pub fn plan_image_grid(width: usize, height: usize, vd: &VisionDims) -> ImagePlan {
    let p = vd.patch;
    let (mut w, mut h) = (width as f64, height as f64);
    if let Some(m) = vd.max_wh_ratio {
        if w > h * m {
            w = h * m;
        }
    }
    let px = w * h;
    if px > 0.0 && px < vd.min_pixels as f64 {
        let ratio = (vd.min_pixels as f64 / px).sqrt();
        // python's int(): truncation
        w = (w * ratio).trunc();
        h = (h * ratio).trunc();
    }
    let mut best_w = ((w / p as f64).ceil() as usize) * p;
    let mut best_h = ((h / p as f64).ceil() as usize) * p;
    let r = vd.downsample;
    let (mut lh, mut lw) = llm_grid(best_h, best_w, p, r);
    if num_image_tokens(lh, lw) > vd.max_tokens {
        (best_h, best_w) = solve_resize_ratio(h, w, p, r, vd.max_tokens);
        (lh, lw) = llm_grid(best_h, best_w, p, r);
        assert!(num_image_tokens(lh, lw) <= vd.max_tokens);
    }
    ImagePlan {
        n_vit_h: best_h / p,
        n_vit_w: best_w / p,
        n_llm_h: lh,
        n_llm_w: lw,
        best_h,
        best_w,
    }
}

/// The span's token types in order.
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i8> {
    let mut t = vec![IMAGE_START];
    for _ in 0..n_llm_h {
        t.extend(std::iter::repeat_n(IMAGE, n_llm_w));
        t.push(IMAGE_NEW_LINE);
    }
    t.push(IMAGE_END);
    t
}

// ---------------------------------------------------------------- pixels

/// An 8-bit RGB image, row-major.
#[derive(Clone, Debug)]
pub struct Rgb8 {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

/// PIL's bicubic kernel (`a = −0.5`, support 2).
fn bicubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// PIL's `precompute_coeffs` + `normalize_coeffs_8bpc`: per output pixel
/// the first source pixel and its fixed-point weights.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let fscale = scale.max(1.0);
    let support = 2.0 * fscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / fscale;
            // C's (int) cast truncates toward zero
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
            let k: Vec<f64> = (0..xmax)
                .map(|x| bicubic((x as f64 + xmin as f64 - center + 0.5) * ss))
                .collect();
            let ww: f64 = k.iter().sum();
            let one = (1u64 << PRECISION_BITS) as f64;
            let kk = k
                .iter()
                .map(|&v| {
                    let v = if ww != 0.0 { v / ww } else { v };
                    if v < 0.0 {
                        (-0.5 + v * one) as i32
                    } else {
                        (0.5 + v * one) as i32
                    }
                })
                .collect();
            (xmin, kk)
        })
        .collect()
}

fn clip8(ss: i32) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// PIL's `Image.resize(size, BICUBIC)` for 8-bit RGB: a horizontal pass,
/// then a vertical one, each rounded to 8 bits (`ImagingResample`).
pub fn pil_resize(im: &Rgb8, w: usize, h: usize) -> Rgb8 {
    if (w, h) == (im.w, im.h) {
        return im.clone();
    }
    let half = 1i32 << (PRECISION_BITS - 1);
    let mut cur = im.clone();
    if w != cur.w {
        let cs = coeffs(cur.w, w);
        let mut out = vec![0u8; w * cur.h * 3];
        for y in 0..cur.h {
            for (x, (x0, k)) in cs.iter().enumerate() {
                for c in 0..3 {
                    let mut ss = half;
                    for (i, &kv) in k.iter().enumerate() {
                        ss += cur.px[(y * cur.w + x0 + i) * 3 + c] as i32 * kv;
                    }
                    out[(y * w + x) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = Rgb8 { w, h: cur.h, px: out };
    }
    if h != cur.h {
        let cs = coeffs(cur.h, h);
        let mut out = vec![0u8; cur.w * h * 3];
        for (y, (y0, k)) in cs.iter().enumerate() {
            for x in 0..cur.w {
                for c in 0..3 {
                    let mut ss = half;
                    for (i, &kv) in k.iter().enumerate() {
                        ss += cur.px[((y0 + i) * cur.w + x) * 3 + c] as i32 * kv;
                    }
                    out[(y * cur.w + x) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = Rgb8 { w: cur.w, h, px: out };
    }
    cur
}

/// Python's `round()`: half to even.
fn round_half_even(v: f64) -> i64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - v.signum()) as i64
    } else {
        r as i64
    }
}

/// PIL's `ImageOps.pad(im, (w, h), BICUBIC, color=(127,127,127))`: fit
/// inside, keep the aspect, center on gray.
pub fn pil_pad(im: &Rgb8, w: usize, h: usize) -> Rgb8 {
    let im_ratio = im.w as f64 / im.h as f64;
    let dest_ratio = w as f64 / h as f64;
    if im_ratio == dest_ratio {
        return pil_resize(im, w, h);
    }
    let mut out = Rgb8 { w, h, px: vec![127; w * h * 3] };
    let (src, x0, y0) = if im_ratio > dest_ratio {
        let nh = round_half_even(im.h as f64 / im.w as f64 * w as f64) as usize;
        let src = if nh != h { pil_resize(im, w, nh) } else { im.clone() };
        (src, 0, round_half_even((h as f64 - nh as f64) * 0.5))
    } else {
        let nw = round_half_even(im.w as f64 / im.h as f64 * h as f64) as usize;
        let src = if nw != w { pil_resize(im, nw, h) } else { im.clone() };
        (src, round_half_even((w as f64 - nw as f64) * 0.5), 0)
    };
    // paste, clipped to the canvas
    for y in 0..src.h {
        let oy = y as i64 + y0;
        if !(0..h as i64).contains(&oy) {
            continue;
        }
        for x in 0..src.w {
            let ox = x as i64 + x0;
            if !(0..w as i64).contains(&ox) {
                continue;
            }
            let (o, s) = ((oy as usize * w + ox as usize) * 3, (y * src.w + x) * 3);
            out.px[o..o + 3].copy_from_slice(&src.px[s..s + 3]);
        }
    }
    out
}

/// f32 → bf16 → f32, round to nearest even (torch's `.to(bfloat16)`).
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

/// One image ready for the encoder.
#[derive(Clone, Debug)]
pub struct ImageInput {
    pub plan: ImagePlan,
    /// `[N, 3·p·p]`, `(c, ky, kx)` within a patch, bf16 values.
    pub patches: Vec<f64>,
    /// SHA-256 of the source bytes: the [`EmbedCache`] key.
    pub hash: [u8; 32],
}

pub fn decode(bytes: &[u8]) -> Result<Rgb8, String> {
    let im = image::load_from_memory(bytes).map_err(|e| e.to_string())?.to_rgb8();
    Ok(Rgb8 {
        w: im.width() as usize,
        h: im.height() as usize,
        px: im.into_raw(),
    })
}

pub fn image_hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// `load_image`: decode, plan, letterbox, normalize, patchify.
pub fn preprocess(bytes: &[u8], vd: &VisionDims) -> Result<ImageInput, String> {
    let im = decode(bytes)?;
    let mut inp = preprocess_rgb(&im, vd);
    inp.hash = image_hash(bytes);
    Ok(inp)
}

pub fn preprocess_rgb(im: &Rgb8, vd: &VisionDims) -> ImageInput {
    let plan = plan_image_grid(im.w, im.h, vd);
    let wide = vd
        .max_wh_ratio
        .is_some_and(|m| im.w as f64 >= m * im.h as f64);
    let fit = if wide {
        pil_resize(im, plan.best_w, plan.best_h)
    } else {
        pil_pad(im, plan.best_w, plan.best_h)
    };
    let p = vd.patch;
    let (nh, nw) = (plan.n_vit_h, plan.n_vit_w);
    let mut patches = vec![0.0; nh * nw * 3 * p * p];
    for py in 0..nh {
        for pxi in 0..nw {
            let base = (py * nw + pxi) * 3 * p * p;
            for c in 0..3 {
                for ky in 0..p {
                    for kx in 0..p {
                        let (y, x) = (py * p + ky, pxi * p + kx);
                        let v = fit.px[(y * fit.w + x) * 3 + c] as f32 / 255.0;
                        let v = bf16_round((v - 0.5) / 0.5);
                        patches[base + (c * p + ky) * p + kx] = v as f64;
                    }
                }
            }
        }
    }
    ImageInput {
        plan,
        patches,
        hash: [0; 32],
    }
}

/// `get_vision_cos_sin`'s angles, `[N, rope_dim]`: patch `(h, w)` gets
/// `h·f_j` for `j < rope_dim/2`, then `w·f_j`.
pub fn rope_angles(vd: &VisionDims, n_h: usize, n_w: usize) -> Vec<f64> {
    let rd = vd.rope_dim();
    let inv: Vec<f64> = (0..rd / 2)
        .map(|j| 1.0 / vd.rope_theta.powf((2 * j) as f64 / rd as f64))
        .collect();
    let mut out = Vec::with_capacity(n_h * n_w * rd);
    for h in 0..n_h {
        for w in 0..n_w {
            out.extend(inv.iter().map(|f| h as f64 * f));
            out.extend(inv.iter().map(|f| w as f64 * f));
        }
    }
    out
}

// ---------------------------------------------------------------- prompt

/// A prompt with its image spans expanded.
#[derive(Clone, Debug, Default)]
pub struct VlPrompt {
    pub ids: Vec<usize>,
    pub types: Vec<i8>,
    /// Per image: the span's first position.
    pub starts: Vec<usize>,
}

impl VlPrompt {
    /// `token_types >= 0`: every span position (selects the VL routing
    /// bias, shuts the Engram gate).
    pub fn mask(&self) -> Vec<f64> {
        self.types.iter().map(|&t| (t >= 0) as u8 as f64).collect()
    }
    pub fn dead(&self) -> Vec<bool> {
        self.types.iter().map(|&t| t >= 0).collect()
    }
}

/// `prepare_vl_inputs`: each `image_token_id` in `tokens` becomes its
/// image's span.
pub fn expand_prompt(
    tokens: &[usize],
    plans: &[ImagePlan],
    image_token_id: usize,
) -> Result<VlPrompt, String> {
    let n = tokens.iter().filter(|&&t| t == image_token_id).count();
    if n != plans.len() {
        return Err(format!("{n} image tokens but {} images", plans.len()));
    }
    let mut out = VlPrompt::default();
    let mut it = plans.iter();
    for &t in tokens {
        if t != image_token_id {
            out.ids.push(t);
            out.types.push(TEXT);
            continue;
        }
        let pl = it.next().unwrap();
        out.starts.push(out.ids.len());
        let ty = image_token_types(pl.n_llm_h, pl.n_llm_w);
        out.ids.extend(std::iter::repeat_n(image_token_id, ty.len()));
        out.types.extend(ty);
    }
    Ok(out)
}

/// The learned span delimiter rows (`image_start`, `image_newline`,
/// `image_end`), `[dim]` each.
#[derive(Clone, Debug)]
pub struct Delims {
    pub start: Vec<f64>,
    pub newline: Vec<f64>,
    pub end: Vec<f64>,
}

/// `merge_image_embeddings`: the prefill's `image_rows [T, dim]` (zero off
/// the spans) and `image_mask [T]`. `embeds[i]` is image `i`'s aligner
/// output, `[rows, dim]`.
pub fn splice_rows(
    pr: &VlPrompt,
    embeds: &[&[f64]],
    delims: &Delims,
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let t = pr.ids.len();
    let mut rows = vec![0.0; t * dim];
    let mut img = 0usize;
    let mut next = 0usize;
    for (pos, &ty) in pr.types.iter().enumerate() {
        let src: &[f64] = match ty {
            IMAGE_START => {
                next = 0;
                &delims.start
            }
            IMAGE_NEW_LINE => &delims.newline,
            IMAGE_END => {
                img += 1;
                &delims.end
            }
            IMAGE => {
                next += 1;
                &embeds[img][(next - 1) * dim..next * dim]
            }
            _ => continue,
        };
        rows[pos * dim..(pos + 1) * dim].copy_from_slice(src);
    }
    (rows, pr.mask())
}

// ---------------------------------------------------------------- cache

/// Encoder outputs by image hash, least recently used evicted past a byte
/// budget. The key covers the bytes only: one cache per model.
pub struct EmbedCache {
    budget: usize,
    used: usize,
    map: HashMap<[u8; 32], Arc<Vec<f32>>>,
    order: VecDeque<[u8; 32]>,
    pub hits: u64,
    pub misses: u64,
}

impl EmbedCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget: budget_bytes,
            used: 0,
            map: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&mut self, key: &[u8; 32]) -> Option<Arc<Vec<f32>>> {
        let hit = self.map.get(key).cloned();
        if hit.is_some() {
            self.hits += 1;
            self.order.retain(|k| k != key);
            self.order.push_back(*key);
        } else {
            self.misses += 1;
        }
        hit
    }

    pub fn put(&mut self, key: [u8; 32], rows: Vec<f32>) -> Arc<Vec<f32>> {
        let v = Arc::new(rows);
        let bytes = v.len() * 4;
        if let Some(old) = self.map.insert(key, v.clone()) {
            self.used -= old.len() * 4;
            self.order.retain(|k| *k != key);
        }
        self.used += bytes;
        self.order.push_back(key);
        while self.used > self.budget && self.order.len() > 1 {
            let k = self.order.pop_front().unwrap();
            if let Some(old) = self.map.remove(&k) {
                self.used -= old.len() * 4;
            }
        }
        v
    }

    /// The cached rows, or `encode()`'s (then cached).
    pub fn get_or_encode(
        &mut self,
        key: [u8; 32],
        encode: impl FnOnce() -> Vec<f32>,
    ) -> Arc<Vec<f32>> {
        match self.get(&key) {
            Some(v) => v,
            None => self.put(key, encode()),
        }
    }
}

// ---------------------------------------------------------------- graph

/// The encoder for one patch grid: inputs (host `patches`, `rope_angles`,
/// then checkpoint tensors by name) and the aligner output `[rows, llm]`.
pub struct EncoderGraph {
    pub g: TensorGraph,
    pub inputs: Vec<PrefillInput>,
    pub out: TensorId,
    /// The ViT output (after the final norm), `[N, dim]`.
    pub vit: TensorId,
    /// The residual stream after each sublayer: a device lowering takes
    /// these as outputs too (see [`EncoderGraph::lower_outputs`]).
    pub keep: Vec<TensorId>,
}

impl EncoderGraph {
    /// `[out, keep…]`: what a device lowering should materialize. The
    /// aligner output is first.
    pub fn lower_outputs(&self) -> Vec<TensorId> {
        std::iter::once(self.out).chain(self.keep.iter().copied()).collect()
    }
}

struct E {
    g: TensorGraph,
    inputs: Vec<PrefillInput>,
}

impl E {
    fn input(&mut self, name: String, shape: &[usize]) -> TensorId {
        let s: Vec<Dim> = shape.iter().map(|&n| fx(n)).collect();
        self.inputs.push(PrefillInput {
            name,
            shape: s.clone(),
        });
        self.g.input(&s)
    }

    fn n(&self, t: TensorId, d: usize) -> usize {
        self.g.shape(t)[d].n()
    }

    /// `y[r, n] = Σ_k x[r, k]·w[n, k] (+ b[n])`.
    fn linear(&mut self, x: TensorId, w: TensorId, b: Option<TensorId>) -> TensorId {
        let (r, n, k) = (self.n(x, 0), self.n(w, 0), self.n(w, 1));
        let y = self
            .g
            .comp(
                &[fx(r), fx(n)],
                &[fx(k)],
                vec![
                    arg(x, vec![unit(3, 0), unit(3, 2)]),
                    arg(w, vec![unit(3, 1), unit(3, 2)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        match b {
            None => y,
            Some(b) => self.map2(y, b, |a, b| a + b),
        }
    }

    /// `f(y[r, n], b[n])`.
    fn map2(&mut self, y: TensorId, b: TensorId, f: impl Fn(ExprId, ExprId) -> ExprId) -> TensorId {
        let (r, n) = (self.n(y, 0), self.n(y, 1));
        self.g
            .comp(
                &[fx(r), fx(n)],
                &[],
                vec![arg(y, ident(2)), arg(b, vec![unit(2, 1)])],
                None,
                |v| vec![f(v[0], v[1])],
            )
            .unwrap()
    }

    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (r, n) = (self.n(a, 0), self.n(a, 1));
        self.g
            .comp(
                &[fx(r), fx(n)],
                &[],
                vec![arg(a, ident(2)), arg(b, ident(2))],
                None,
                |v| vec![v[0] + v[1]],
            )
            .unwrap()
    }

    fn rms(&mut self, x: TensorId, w: TensorId, eps: f64) -> TensorId {
        let (r, n) = (self.n(x, 0), self.n(x, 1));
        let ss = self
            .g
            .comp(&[fx(r)], &[fx(n)], vec![arg(x, ident(2))], Some(Monoid::sum()), |v| {
                vec![v[0] * v[0]]
            })
            .unwrap();
        self.g
            .comp(
                &[fx(r), fx(n)],
                &[],
                vec![
                    arg(x, ident(2)),
                    arg(ss, vec![unit(2, 0)]),
                    arg(w, vec![unit(2, 1)]),
                ],
                None,
                |v| vec![v[2] * (v[0] / (v[1] / lit(n as f64) + lit(eps)).sqrt())],
            )
            .unwrap()
    }

    /// Half-split RoPE of the `heads` 64-wide heads at column `base` of
    /// `qkv` (`[N, ·]`), angles `[N, hd/2]`: `[N, heads·hd]`.
    fn rope(&mut self, qkv: TensorId, base: usize, heads: usize, hd: usize, ang: TensorId) -> TensorId {
        let n = self.n(qkv, 0);
        let half = hd / 2;
        let sp = [fx(n), fx(heads), fx(2), fx(half)];
        let par = self.g.iota(&sp, 2);
        let col = |o: usize| {
            arg(
                qkv,
                vec![unit(4, 0), form(4, &[(1, hd as i64), (3, 1)], (base + o) as i64)],
            )
        };
        let r = self
            .g
            .comp(
                &sp,
                &[],
                vec![
                    col(0),
                    col(half),
                    arg(ang, vec![unit(4, 0), unit(4, 3)]),
                    arg(par, ident(4)),
                ],
                None,
                |v| {
                    let (c, s) = (v[2].cos(), v[2].sin());
                    let lo = v[0] * c - v[1] * s;
                    let hi = v[1] * c + v[0] * s;
                    vec![select(v[3] - lit(0.5), hi, lo)]
                },
            )
            .unwrap();
        self.g.reshape(r, &[fx(n), fx(heads * hd)]).unwrap()
    }

    /// Full (bidirectional) softmax attention: `q`, `k` `[N, H·hd]`, `v`
    /// at column `vbase` of `qkv`; `[N, H·hd]`.
    fn attention(
        &mut self,
        q: TensorId,
        k: TensorId,
        qkv: TensorId,
        vbase: usize,
        heads: usize,
        hd: usize,
    ) -> TensorId {
        let n = self.n(q, 0);
        let scale = 1.0 / (hd as f64).sqrt();
        let hcol = |r: usize, row: usize, head: usize, d: usize| {
            vec![unit(r, row), form(r, &[(head, hd as i64), (d, 1)], 0)]
        };
        // scores s[h, i, j]
        let s = self
            .g
            .comp(
                &[fx(heads), fx(n), fx(n)],
                &[fx(hd)],
                vec![arg(q, hcol(4, 1, 0, 3)), arg(k, hcol(4, 2, 0, 3))],
                Some(Monoid::sum()),
                move |v| vec![v[0] * v[1] * lit(scale)],
            )
            .unwrap();
        // m and l keep the key dim as size 1 ([h, i, 1]): their spaces then
        // have p·v's rank, so the softmax fusion sees one score and makes
        // one online-softmax state (m, l, acc) — the flash tile's input
        let m = self
            .g
            .comp(
                &[fx(heads), fx(n), fx(1)],
                &[fx(n)],
                vec![arg(s, vec![unit(4, 0), unit(4, 1), unit(4, 3)])],
                Some(Monoid::max()),
                |v| vec![v[0]],
            )
            .unwrap();
        let p = self
            .g
            .comp(
                &[fx(heads), fx(n), fx(n)],
                &[],
                vec![arg(s, ident(3)), arg(m, vec![unit(3, 0), unit(3, 1), konst(3, 0)])],
                None,
                |v| vec![(v[0] - v[1]).exp()],
            )
            .unwrap();
        let z = self
            .g
            .comp(
                &[fx(heads), fx(n), fx(1)],
                &[fx(n)],
                vec![arg(p, vec![unit(4, 0), unit(4, 1), unit(4, 3)])],
                Some(Monoid::sum()),
                |v| vec![v[0]],
            )
            .unwrap();
        // o[h, i, d] = Σ_j p[h, i, j]·v[j, h, d], space (h, i, d, j)
        let o = self
            .g
            .comp(
                &[fx(heads), fx(n), fx(hd)],
                &[fx(n)],
                vec![
                    arg(p, vec![unit(4, 0), unit(4, 1), unit(4, 3)]),
                    arg(
                        qkv,
                        vec![unit(4, 3), form(4, &[(0, hd as i64), (2, 1)], vbase as i64)],
                    ),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        // normalized, written as [i, h, d] so the projection is a plain
        // linear over [N, H·hd]
        let y = self
            .g
            .comp(
                &[fx(n), fx(heads), fx(hd)],
                &[],
                vec![
                    arg(o, vec![unit(3, 1), unit(3, 0), unit(3, 2)]),
                    arg(z, vec![unit(3, 1), unit(3, 0), konst(3, 0)]),
                ],
                None,
                |v| vec![v[0] / v[1]],
            )
            .unwrap();
        self.g.reshape(y, &[fx(n), fx(heads * hd)]).unwrap()
    }

}

/// `|x|` as a select.
fn abs(x: ExprId) -> ExprId {
    select(x, x, -x)
}

fn poly(x: ExprId, c: &[f64]) -> ExprId {
    // Horner, highest degree first
    let mut acc = lit(c[0]);
    for &k in &c[1..] {
        acc = acc * x + lit(k);
    }
    acc
}

/// `erfc(y)` for `y ≥ 0.5`, Cody's rational forms (CALERF), and
/// `erf(x)` for `|x| ≤ 0.5`: about 1e-16 relative on f64.
pub fn erf_expr(x: ExprId) -> ExprId {
    const A: [f64; 5] = [
        1.85777706184603153e-1,
        3.16112374387056560e00,
        1.13864154151050156e02,
        3.77485237685302021e02,
        3.20937758913846947e03,
    ];
    const B: [f64; 5] = [
        1.0,
        2.36012909523441209e01,
        2.44024637934444173e02,
        1.28261652607737228e03,
        2.84423683343917062e03,
    ];
    const C: [f64; 9] = [
        2.15311535474403846e-8,
        5.64188496988670089e-1,
        8.88314979438837594e00,
        6.61191906371416295e01,
        2.98635138197400131e02,
        8.81952221241769090e02,
        1.71204761263407058e03,
        2.05107837782607147e03,
        1.23033935479799725e03,
    ];
    const D: [f64; 9] = [
        1.0,
        1.57449261107098347e01,
        1.17693950891312499e02,
        5.37181101862009858e02,
        1.62138957456669019e03,
        3.29079923573345963e03,
        4.36261909014324716e03,
        3.43936767414372164e03,
        1.23033935480374942e03,
    ];
    const P: [f64; 6] = [
        1.63153871373020978e-2,
        3.05326634961232344e-1,
        3.60344899949804439e-1,
        1.25781726111229246e-1,
        1.60837851487422766e-2,
        6.58749161529837803e-4,
    ];
    const Q: [f64; 6] = [
        1.0,
        2.56852019228982242e00,
        1.87295284992346725e00,
        5.27905102951428412e-1,
        6.05183413124413191e-2,
        2.33520497626869185e-3,
    ];
    const SQRPI: f64 = 5.6418958354775628695e-1;
    let y = abs(x);
    let ysq = x * x;
    // |x| ≤ 0.5
    let small = x * poly(ysq, &A) / poly(ysq, &B);
    // 0.5 < |x| ≤ 4: erfc = exp(−y²)·C(y)/D(y)
    let mid = poly(y, &C) / poly(y, &D);
    // |x| > 4 (y clamped so 1/y² stays finite where unused)
    let yb = select(y - lit(4.0), y, lit(4.0));
    let r = lit(1.0) / (yb * yb);
    let big = (lit(SQRPI) - r * poly(r, &P) / poly(r, &Q)) / yb;
    let tail = select(y - lit(4.0), big, mid) * (-ysq).exp();
    // erf = ±(1 − erfc)
    let erf_big = select(x, lit(1.0) - tail, tail - lit(1.0));
    select(y - lit(0.5), erf_big, small)
}

/// `F.gelu` (exact): `x/2 · (1 + erf(x/√2))`.
pub fn gelu_expr(x: ExprId) -> ExprId {
    lit(0.5) * x * (lit(1.0) + erf_expr(x * lit(std::f64::consts::FRAC_1_SQRT_2)))
}

/// The ViT + aligner for an `n_h × n_w` patch grid.
pub fn encoder_graph(vd: &VisionDims, n_h: usize, n_w: usize) -> EncoderGraph {
    let (c, h, hd, inter) = (vd.dim, vd.n_heads, vd.head_dim(), vd.inter);
    let n = n_h * n_w;
    let mut e = E {
        g: TensorGraph::new(),
        inputs: Vec::new(),
    };
    let patches = e.input("patches".into(), &[n, vd.patch_len()]);
    let ang = e.input("rope_angles".into(), &[n, vd.rope_dim()]);
    let pw = e.input("vision.patch_embed.proj.weight".into(), &[c, vd.patch_len()]);
    let pb = e.input("vision.patch_embed.proj.bias".into(), &[c]);
    let mut x = e.linear(patches, pw, Some(pb));
    // the residual stream, layer by layer: lowered as extra outputs so it
    // is written out — fused into its readers (each reads it once per
    // element), layer L would re-add every earlier layer's output
    let mut keep = vec![x];
    for l in 0..vd.n_layers {
        let p = format!("vision.blocks.{l}");
        let n1 = e.input(format!("{p}.norm1.weight"), &[c]);
        let wqkv = e.input(format!("{p}.attn.wqkv.weight"), &[3 * c, c]);
        let bqkv = e.input(format!("{p}.attn.wqkv.bias"), &[3 * c]);
        let wo = e.input(format!("{p}.attn.wo.weight"), &[c, c]);
        let bo = e.input(format!("{p}.attn.wo.bias"), &[c]);
        let n2 = e.input(format!("{p}.norm2.weight"), &[c]);
        let w1 = e.input(format!("{p}.mlp.w1.weight"), &[2 * inter, c]);
        let w2 = e.input(format!("{p}.mlp.w2.weight"), &[c, inter]);
        let xn = e.rms(x, n1, vd.eps);
        let qkv = e.linear(xn, wqkv, Some(bqkv));
        let q = e.rope(qkv, 0, h, hd, ang);
        let k = e.rope(qkv, c, h, hd, ang);
        let o = e.attention(q, k, qkv, 2 * c, h, hd);
        let a = e.linear(o, wo, Some(bo));
        x = e.add(x, a);
        keep.push(x);
        let xn = e.rms(x, n2, vd.eps);
        let gu = e.linear(xn, w1, None);
        let act = e
            .g
            .comp(
                &[fx(n), fx(inter)],
                &[],
                vec![
                    arg(gu, ident(2)),
                    arg(gu, vec![unit(2, 0), form(2, &[(1, 1)], inter as i64)]),
                ],
                None,
                |v| vec![v[0] / (lit(1.0) + (-v[0]).exp()) * v[1]],
            )
            .unwrap();
        let m = e.linear(act, w2, None);
        x = e.add(x, m);
        keep.push(x);
    }
    let nw = e.input("vision.norm.weight".into(), &[c]);
    let vit = e.rms(x, nw, vd.eps);

    // aligner: zero-pad to multiples of r, unfold r×r blocks
    let r = vd.downsample;
    let (bh, bw) = (n_h.div_ceil(r), n_w.div_ceil(r));
    let x3 = e.g.reshape(vit, &[fx(n_h), fx(n_w), fx(c)]).unwrap();
    let ri = r as i64;
    let unf = e
        .g
        .comp(
            &[fx(bh), fx(bw), fx(c), fx(r), fx(r)],
            &[],
            vec![Arg {
                src: x3,
                view: View {
                    forms: vec![
                        form(5, &[(0, ri), (3, 1)], 0),
                        form(5, &[(1, ri), (4, 1)], 0),
                        unit(5, 2),
                    ],
                    guard: true,
                    gather: None,
                },
            }],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let unf = e.g.reshape(unf, &[fx(bh * bw), fx(c * r * r)]).unwrap();
    let a1w = e.input("aligner.w1.weight".into(), &[vd.llm_dim, c * r * r]);
    let a1b = e.input("aligner.w1.bias".into(), &[vd.llm_dim]);
    let a2w = e.input("aligner.w2.weight".into(), &[vd.llm_dim, vd.llm_dim]);
    let a2b = e.input("aligner.w2.bias".into(), &[vd.llm_dim]);
    let y = e.linear(unf, a1w, None);
    let y = e.map2(y, a1b, |a, b| gelu_expr(a + b));
    let out = e.linear(y, a2w, Some(a2b));
    EncoderGraph {
        g: e.g,
        inputs: e.inputs,
        out,
        vit,
        keep,
    }
}

/// The prefill/training schedule for the encoder: softmax stabilized and
/// fused into one online-softmax state (the flash tile), maps substituted,
/// siblings merged, reduce dims merged/split, `schedule_auto`, and long
/// row reductions with few rows split across a workgroup. `bf16`: the
/// checkpoint tensors (every graph input past `patches` and `rope_angles`)
/// stored as bf16 — exact, the checkpoint is bf16 — so matmul tiles run on
/// bf16 fragments (and Hopper wgmma), activations rounded while staging.
pub fn prefill_schedule(g: &TensorGraph, outs: &[TensorId], bf16: Bf16) -> (TensorGraph, Vec<TensorId>) {
    use tang_compile::tensor::Op;
    let (g, o) = g.stabilize_softmax(outs);
    let (g, o) = g.substitute_maps(&o);
    let (g, o) = g.eliminate_common(&o);
    let (g, o) = g.merge_siblings(&o);
    let (g, o) = g.relayout(&o);
    let (g, o) = g.merge_reduce_dims(&o);
    let (mut g, o) = g.split_reduce_dims(&o);
    g.schedule_auto(&o);
    // an inlined map with several readers is recomputed in each — and the
    // residual stream `x + a` inlined layer after layer makes every
    // reader re-add the whole chain: write those out
    {
        let n = g.nodes().len();
        let mut readers = vec![0usize; n];
        for j in 0..n {
            if let Op::Comp(c) = &g.nodes()[j].op {
                for a in &c.args {
                    readers[a.src.index()] += 1;
                }
            }
        }
        let shared: Vec<usize> = (0..n)
            .filter(|&i| {
                let nd = &g.nodes()[i];
                matches!(&nd.op, Op::Comp(c) if c.monoid.is_none()) && nd.inline && readers[i] > 1
            })
            .collect();
        for i in shared {
            g.set_inline(TensorId::from_index(i), false);
        }
    }
    // row reductions with few rows and long rows (RMSNorm's Σx², 1521 rows
    // of 1024) run one thread per row unless split across a workgroup
    let numel = |d: &[Dim]| d.iter().map(|x| x.n()).product::<usize>();
    let split: Vec<usize> = (0..g.nodes().len())
        .filter(|&i| {
            let nd = &g.nodes()[i];
            let Op::Comp(c) = &nd.op else { return false };
            !nd.inline
                && nd.split == 0
                && nd.tile == (0, 0, 0)
                && c.monoid.is_some()
                && c.tuple == 0
                && numel(&c.red) >= 256
                && numel(&c.out) <= 1 << 13
        })
        .collect();
    for i in split {
        let nd = &g.nodes()[i];
        let (tile, reg) = (nd.tile, nd.reg);
        g.set_schedule(TensorId::from_index(i), 256, tile, reg);
    }
    let half = tang_compile::tensor::Precision::storage(tang_compile::tensor::Format::BF16);
    if bf16 != Bf16::Off {
        let weights: Vec<usize> = (0..g.nodes().len())
            .filter(|&i| matches!(g.nodes()[i].op, Op::Input(k) if k >= 2))
            .collect();
        for i in weights {
            g.set_precision(TensorId::from_index(i), half);
        }
    }
    if bf16 == Bf16::Activations {
        // every materialized activation but the outputs and the short row
        // reductions (Σx², softmax state) in bf16, as the official model
        // runs: Q, K, V half, so the Hopper flash and wgmma tiles apply
        let acts: Vec<usize> = (0..g.nodes().len())
            .filter(|&i| {
                let nd = &g.nodes()[i];
                let Op::Comp(c) = &nd.op else { return false };
                !nd.inline
                    && !o.iter().any(|t| t.index() == i)
                    && c.tuple == 0
                    && numel(&c.out) > 1 << 13
            })
            .collect();
        for i in acts {
            g.set_precision(TensorId::from_index(i), half);
        }
    }
    (g, o)
}

/// What [`prefill_schedule`] stores in bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16 {
    Off,
    /// The checkpoint tensors (exact).
    Weights,
    /// Weights and the materialized activations.
    Activations,
}

/// [`prefill_schedule`] lowered by `to_cuda`.
pub fn lower_prefill(
    g: &TensorGraph,
    outs: &[TensorId],
    bf16: Bf16,
) -> (tang_compile::tensor::wgsl::WgslProgram, Vec<TensorId>) {
    let (g, o) = prefill_schedule(g, outs, bf16);
    (g.to_cuda(&o), o)
}

/// The encoder's graph inputs in order: `patches`, `rope_angles`, then
/// each checkpoint tensor from `weight(name)`.
pub fn encoder_inputs(
    eg: &EncoderGraph,
    vd: &VisionDims,
    img: &ImageInput,
    mut weight: impl FnMut(&str) -> Vec<f64>,
) -> Vec<Vec<f64>> {
    eg.inputs
        .iter()
        .map(|s| match s.name.as_str() {
            "patches" => img.patches.clone(),
            "rope_angles" => rope_angles(vd, img.plan.n_vit_h, img.plan.n_vit_w),
            nm => weight(nm),
        })
        .collect()
}

/// The vision tower's checkpoint tensors (for loaders / budgets).
pub fn weight_names(vd: &VisionDims) -> Vec<String> {
    let mut v = vec![
        "vision.patch_embed.proj.weight".to_string(),
        "vision.patch_embed.proj.bias".into(),
    ];
    for l in 0..vd.n_layers {
        for t in [
            "norm1.weight",
            "attn.wqkv.weight",
            "attn.wqkv.bias",
            "attn.wo.weight",
            "attn.wo.bias",
            "norm2.weight",
            "mlp.w1.weight",
            "mlp.w2.weight",
        ] {
            v.push(format!("vision.blocks.{l}.{t}"));
        }
    }
    v.extend(
        [
            "vision.norm.weight",
            "aligner.w1.weight",
            "aligner.w1.bias",
            "aligner.w2.weight",
            "aligner.w2.bias",
        ]
        .map(String::from),
    );
    v
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ir_ds_prefill::tests::rel;
    use rayon::prelude::*;

    /// `y[r, n] = Σ_k x[r, k]·w[n, k] + b[n]`, rows in parallel.
    fn lin(x: &[f64], k: usize, w: &[f64], b: Option<&[f64]>) -> Vec<f64> {
        let n = w.len() / k;
        let mut y = vec![0.0; x.len() / k * n];
        y.par_chunks_mut(n).zip(x.par_chunks(k)).for_each(|(yr, xr)| {
            for (j, o) in yr.iter_mut().enumerate() {
                let wr = &w[j * k..(j + 1) * k];
                let mut s = 0.0;
                for i in 0..k {
                    s += xr[i] * wr[i];
                }
                *o = s + b.map_or(0.0, |b| b[j]);
            }
        });
        y
    }

    fn rms(x: &[f64], n: usize, w: &[f64], eps: f64) -> Vec<f64> {
        x.chunks(n)
            .flat_map(|r| {
                let ms = r.iter().map(|v| v * v).sum::<f64>() / n as f64;
                let s = 1.0 / (ms + eps).sqrt();
                r.iter().zip(w).map(move |(v, w)| w * (v * s)).collect::<Vec<_>>()
            })
            .collect()
    }

    /// `vision.py` + the aligner in f64: `(vit [N, C], aligner [rows, llm])`.
    pub(crate) fn reference(
        vd: &VisionDims,
        img: &ImageInput,
        w: &dyn Fn(&str) -> Vec<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let (c, h, hd, inter) = (vd.dim, vd.n_heads, vd.head_dim(), vd.inter);
        let (nh, nw) = (img.plan.n_vit_h, img.plan.n_vit_w);
        let n = nh * nw;
        // get_vision_cos_sin: inv_freq over rope_dim, positions (h, w)
        let rd = vd.rope_dim();
        let inv: Vec<f64> = (0..rd / 2)
            .map(|j| 1.0 / vd.rope_theta.powf((2 * j) as f64 / rd as f64))
            .collect();
        let freqs: Vec<f64> = (0..n)
            .flat_map(|i| {
                let (ph, pw) = ((i / nw) as f64, (i % nw) as f64);
                let inv = &inv;
                inv.iter().map(move |f| ph * f).chain(inv.iter().map(move |f| pw * f))
            })
            .collect();
        let mut x = lin(
            &img.patches,
            vd.patch_len(),
            &w("vision.patch_embed.proj.weight"),
            Some(&w("vision.patch_embed.proj.bias")),
        );
        for l in 0..vd.n_layers {
            let p = format!("vision.blocks.{l}");
            let xn = rms(&x, c, &w(&format!("{p}.norm1.weight")), vd.eps);
            let qkv = lin(
                &xn,
                c,
                &w(&format!("{p}.attn.wqkv.weight")),
                Some(&w(&format!("{p}.attn.wqkv.bias"))),
            );
            // apply_rotary on q and k: x1, x2 = chunk(2) of each head
            let rot = |part: usize| -> Vec<f64> {
                let mut out = vec![0.0; n * c];
                for i in 0..n {
                    for hh in 0..h {
                        let src = &qkv[i * 3 * c + part * c + hh * hd..][..hd];
                        let dst = &mut out[i * c + hh * hd..][..hd];
                        for j in 0..hd / 2 {
                            let (cs, sn) = (freqs[i * rd + j].cos(), freqs[i * rd + j].sin());
                            let (x1, x2) = (src[j], src[j + hd / 2]);
                            dst[j] = x1 * cs - x2 * sn;
                            dst[j + hd / 2] = x2 * cs + x1 * sn;
                        }
                    }
                }
                out
            };
            let (q, k) = (rot(0), rot(1));
            let scale = 1.0 / (hd as f64).sqrt();
            // scaled_dot_product_attention per head, no mask
            let mut o = vec![0.0; n * c];
            o.par_chunks_mut(c).enumerate().for_each(|(i, orow)| {
                let mut s = vec![0.0; n];
                for hh in 0..h {
                    let qi = &q[i * c + hh * hd..][..hd];
                    for (j, sj) in s.iter_mut().enumerate() {
                        let kj = &k[j * c + hh * hd..][..hd];
                        *sj = (0..hd).map(|d| qi[d] * kj[d]).sum::<f64>() * scale;
                    }
                    let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for d in 0..hd {
                        let mut acc = 0.0;
                        for (j, ej) in e.iter().enumerate() {
                            acc += ej * qkv[j * 3 * c + 2 * c + hh * hd + d];
                        }
                        orow[hh * hd + d] = acc / z;
                    }
                }
            });
            let a = lin(
                &o,
                c,
                &w(&format!("{p}.attn.wo.weight")),
                Some(&w(&format!("{p}.attn.wo.bias"))),
            );
            x.iter_mut().zip(&a).for_each(|(x, a)| *x += a);
            let xn = rms(&x, c, &w(&format!("{p}.norm2.weight")), vd.eps);
            let gu = lin(&xn, c, &w(&format!("{p}.mlp.w1.weight")), None);
            let act: Vec<f64> = gu
                .chunks(2 * inter)
                .flat_map(|r| {
                    (0..inter).map(move |j| {
                        let g = r[j];
                        g / (1.0 + (-g).exp()) * r[inter + j]
                    })
                })
                .collect();
            let m = lin(&act, inter, &w(&format!("{p}.mlp.w2.weight")), None);
            x.iter_mut().zip(&m).for_each(|(x, m)| *x += m);
        }
        let vit = rms(&x, c, &w("vision.norm.weight"), vd.eps);
        // Aligner: view(n_h, n_w, C).permute(2,0,1), pad, unfold(r, stride r)
        let r = vd.downsample;
        let (bh, bw) = (nh.div_ceil(r), nw.div_ceil(r));
        let mut unf = vec![0.0; bh * bw * c * r * r];
        for bi in 0..bh {
            for bj in 0..bw {
                let row = &mut unf[(bi * bw + bj) * c * r * r..][..c * r * r];
                for ch in 0..c {
                    for ki in 0..r {
                        for kj in 0..r {
                            let (yy, xx) = (bi * r + ki, bj * r + kj);
                            if yy < nh && xx < nw {
                                row[ch * r * r + ki * r + kj] = vit[(yy * nw + xx) * c + ch];
                            }
                        }
                    }
                }
            }
        }
        let y = lin(&unf, c * r * r, &w("aligner.w1.weight"), Some(&w("aligner.w1.bias")));
        let y: Vec<f64> = y
            .iter()
            .map(|&v| 0.5 * v * (1.0 + libm::erf(v / std::f64::consts::SQRT_2)))
            .collect();
        let out = lin(&y, vd.llm_dim, &w("aligner.w2.weight"), Some(&w("aligner.w2.bias")));
        (vit, out)
    }

    /// Seeded weights for the tiny tower.
    fn tiny_weights(eg: &EncoderGraph) -> HashMap<String, Vec<f64>> {
        eg.inputs
            .iter()
            .filter(|s| s.name != "patches" && s.name != "rope_angles")
            .map(|s| {
                let n: usize = s.shape.iter().map(|d| d.n()).product();
                let cols = s.shape.last().unwrap().n() as f64;
                let mut seed = s.name.bytes().fold(0xcbf29ce484222325u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x100000001b3)
                });
                let v = (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
                        if s.name.contains("norm") {
                            1.0 + 0.4 * (u - 0.5)
                        } else {
                            2.0 * (2.0 * u - 1.0) / cols.sqrt()
                        }
                    })
                    .collect();
                (s.name.clone(), v)
            })
            .collect()
    }

    fn gradient(w: usize, h: usize) -> Rgb8 {
        let mut px = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                px.push((x * 255 / w.max(1)) as u8);
                px.push((y * 255 / h.max(1)) as u8);
                px.push(((x * 7 + y * 13) % 256) as u8);
            }
        }
        Rgb8 { w, h, px }
    }

    #[test]
    fn erf_matches_libm() {
        let mut g = TensorGraph::new();
        let n = 4001;
        let x = g.input(&[n]);
        let y = g
            .comp(&[n], &[], vec![arg(x, ident(1))], None, |v| vec![erf_expr(v[0])])
            .unwrap();
        let xs: Vec<f64> = (0..n).map(|i| -10.0 + 20.0 * i as f64 / (n - 1) as f64).collect();
        let got = g.eval(&[xs.clone()], &[y]).remove(0);
        let worst = xs
            .iter()
            .zip(&got)
            .map(|(&x, &e)| (e - libm::erf(x)).abs() / libm::erf(x).abs().max(1e-300))
            .fold(0.0, f64::max);
        assert!(worst < 1e-14, "erf rel {worst:e}");
    }

    #[test]
    fn plan_matches_official_examples() {
        let vd = VisionDims {
            dim: 1024,
            n_heads: 16,
            inter: 2816,
            n_layers: 32,
            patch: 14,
            rope_theta: 10000.0,
            downsample: 3,
            max_tokens: 1024,
            min_pixels: 295936,
            max_wh_ratio: None,
            llm_dim: 5120,
            image_token_id: 129264,
            eps: 1e-6,
        };
        // values from image_processor.plan_image_grid (python, by hand):
        // 640x480: 307200 px ≥ min; best 644 x 490 → 46 x 35 patches → 16 x 12
        let p = plan_image_grid(640, 480, &vd);
        assert_eq!((p.best_w, p.best_h, p.n_vit_w, p.n_vit_h), (644, 490, 46, 35));
        assert_eq!((p.n_llm_w, p.n_llm_h, p.span()), (16, 12, 12 * 17 + 2));
        // small: scaled up to 544² → 546 x 546, 39² patches, 13² tokens
        let p = plan_image_grid(100, 100, &vd);
        assert_eq!((p.best_w, p.n_vit_w, p.n_llm_w, p.span()), (546, 39, 13, 184));
        // huge: shrunk under 1024 tokens
        let p = plan_image_grid(4000, 3000, &vd);
        assert!(p.span() <= 1024, "{p:?}");
        assert_eq!(image_token_types(2, 3).len(), num_image_tokens(2, 3));
    }

    #[test]
    fn resize_is_identity_and_flat_stays_flat() {
        let im = gradient(31, 17);
        assert_eq!(pil_resize(&im, 31, 17).px, im.px);
        let flat = Rgb8 { w: 9, h: 5, px: vec![200; 9 * 5 * 3] };
        let r = pil_resize(&flat, 23, 11);
        assert!(r.px.iter().all(|&v| v == 200));
        let pad = pil_pad(&flat, 20, 20);
        assert_eq!(pad.px[0], 127);
        assert_eq!(pad.px[(10 * 20 + 10) * 3], 200);
    }

    #[test]
    fn splice_fills_spans_in_reading_order() {
        let plans = [ImagePlan { n_vit_h: 6, n_vit_w: 3, n_llm_h: 2, n_llm_w: 1, best_h: 0, best_w: 0 }];
        let pr = expand_prompt(&[5, 7, 9], &plans, 7).unwrap();
        assert_eq!(pr.ids, vec![5, 7, 7, 7, 7, 7, 7, 9]);
        assert_eq!(pr.types, vec![TEXT, 0, 1, 2, 1, 2, 3, TEXT]);
        let dl = Delims { start: vec![10.0], newline: vec![20.0], end: vec![30.0] };
        let emb = [1.0, 2.0];
        let (rows, mask) = splice_rows(&pr, &[&emb], &dl, 1);
        assert_eq!(rows, vec![0.0, 10.0, 1.0, 20.0, 2.0, 20.0, 30.0, 0.0]);
        assert_eq!(mask, vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        assert!(expand_prompt(&[7, 7], &plans, 7).is_err());
    }

    #[test]
    fn cache_hits_and_evicts() {
        let mut c = EmbedCache::new(16);
        let (a, b) = (image_hash(b"a"), image_hash(b"b"));
        let mut calls = 0;
        c.get_or_encode(a, || {
            calls += 1;
            vec![1.0; 2]
        });
        c.get_or_encode(a, || {
            calls += 1;
            vec![1.0; 2]
        });
        assert_eq!((calls, c.hits, c.misses), (1, 1, 1));
        c.put(b, vec![2.0; 3]);
        // 8 + 12 bytes > 16: a (least recent) goes
        assert!(c.get(&a).is_none() && c.get(&b).is_some());
    }

    /// The encoder graph on the interpreter against the f64 transcription,
    /// tiny tower, a grid that needs aligner padding (7 × 5 patches).
    #[test]
    fn encoder_graph_matches_reference_tiny() {
        let vd = VisionDims::tiny(2);
        let im = gradient(13, 9);
        let mut img = preprocess_rgb(&im, &vd);
        // the tiny plan: force an odd grid so the aligner pads
        img.plan = ImagePlan { n_vit_h: 7, n_vit_w: 5, n_llm_h: 3, n_llm_w: 2, best_h: 14, best_w: 10 };
        img.patches = (0..35 * vd.patch_len()).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
        let eg = encoder_graph(&vd, 7, 5);
        let w = tiny_weights(&eg);
        let ins = encoder_inputs(&eg, &vd, &img, |n| w[n].clone());
        let got = eg.g.eval(&ins, &[eg.vit, eg.out]);
        let (vit, out) = reference(&vd, &img, &|n| w[n].clone());
        let (ev, eo) = (rel(&got[0], &vit), rel(&got[1], &out));
        assert_eq!(got[1].len(), 6 * vd.llm_dim);
        assert!(ev < 1e-13 && eo < 1e-13, "vit {ev:e} aligner {eo:e}");
    }

    /// The prefill with an image span against the reference on random tiny
    /// weights: rows spliced, VL bias routing, Engram gate shut.
    #[test]
    fn vl_prefill_matches_reference_tiny() {
        use crate::ir_ds_prefill::tests::{tiny_engram, weights, with_hashes_masked, Bugs, Ref};
        use crate::ir_ds_prefill::{prefill_graph, DsDims, SEQ};
        let dm = DsDims { vision: true, ..DsDims::tiny_full() };
        let pg = prefill_graph(&dm);
        let mut w = weights(&pg, &dm);
        // bias_vl must differ from bias for the check to see it
        for (k, v) in w.iter_mut() {
            if k.ends_with("gate.bias_vl") {
                v.iter_mut().enumerate().for_each(|(i, x)| *x = (i % 3) as f64 * 0.7 - 0.7);
            }
        }
        let t = 9;
        let toks: Vec<f64> = (0..t).map(|i| ((37 * i + 5) % dm.vocab) as f64).collect();
        let mask: Vec<f64> = (0..t).map(|i| (2..6).contains(&i) as u8 as f64).collect();
        let rows: Vec<f64> = (0..t * dm.dim)
            .map(|i| if mask[i / dm.dim] > 0.5 { ((i * 13 % 29) as f64 - 14.0) / 20.0 } else { 0.0 })
            .collect();
        let meta = tiny_engram(&dm);
        let dead: Vec<bool> = mask.iter().map(|&m| m > 0.5).collect();
        with_hashes_masked(&dm, &meta, &mut w, &toks, &dead);
        let want = Ref { dm: &dm, w: &w, bugs: Bugs::default(), load: None }
            .forward_vl(&toks, Some((&rows, &mask)))
            .0;
        let text_only = Ref { dm: &dm, w: &w, bugs: Bugs::default(), load: None }
            .forward_vl(&toks, None)
            .0;
        let ins: Vec<Vec<f64>> = pg
            .inputs
            .iter()
            .map(|s| match s.name.as_str() {
                "tokens" => toks.clone(),
                "image_rows" => rows.clone(),
                "image_mask" => mask.clone(),
                n if n.ends_with(".engram.hash") => w[n][..t * s.shape[1].n()].to_vec(),
                n => w[n].clone(),
            })
            .collect();
        let g = pg.g.bind(&[(SEQ, t)]).unwrap();
        let got = g.eval(&ins, &[pg.logits]).remove(0);
        let e = rel(&got, &want);
        assert!(e < 1e-11, "vl prefill {e:e}");
        assert!(rel(&text_only, &want) > 1e-3, "the image must change the logits");
    }

    // ------------------------------------------------ real checkpoint

    fn real() -> Option<(crate::ir_ds_real::St, serde_json::Value, VisionDims)> {
        let dir = crate::ir_ds_real::ds_dir();
        let p = std::path::Path::new(&dir);
        let cfg = crate::ir_ds_real::load_config(p).ok()?;
        let vd = VisionDims::from_config(&cfg)?;
        Some((crate::ir_ds_real::St::open(p).ok()?, cfg, vd))
    }

    fn test_image() -> (String, Vec<u8>) {
        let path = std::env::var("VISION_IMAGE").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cram-codec-wasm/site/examples/hurricane-TC-original.webp"
            )
            .into()
        });
        let b = std::fs::read(&path).unwrap();
        (path, b)
    }

    fn sum_abs(v: &[f64]) -> f64 {
        v.iter().map(|x| x.abs()).sum()
    }

    /// The encoder graph on the f64 interpreter against the f64
    /// transcription of `vision.py`, real weights, a real image at the real
    /// pixel budget (`VISION_LAYERS` limits the blocks). `VISION_DUMP`:
    /// write the patches (f32 LE) there for the PIL parity script.
    #[test]
    #[ignore]
    fn real_encoder_matches_reference() {
        let Some((st, _, mut vd)) = real() else { return eprintln!("no checkpoint") };
        if let Some(l) = std::env::var("VISION_LAYERS").ok().and_then(|v| v.parse().ok()) {
            vd.n_layers = l;
        }
        let (path, bytes) = test_image();
        let t0 = std::time::Instant::now();
        let img = preprocess(&bytes, &vd).unwrap();
        let pl = img.plan;
        eprintln!(
            "{path}: plan {pl:?}, {} span positions, preprocess {:.1?}",
            pl.span(),
            t0.elapsed()
        );
        if let Ok(d) = std::env::var("VISION_DUMP") {
            let b: Vec<u8> = img.patches.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect();
            std::fs::write(&d, b).unwrap();
        }
        let ws: std::sync::Mutex<HashMap<String, Vec<f64>>> = Default::default();
        let w = |n: &str| -> Vec<f64> {
            ws.lock().unwrap().entry(n.into()).or_insert_with(|| st.f64(n).unwrap()).clone()
        };
        let t0 = std::time::Instant::now();
        let (vit, out) = reference(&vd, &img, &w);
        eprintln!("reference: {} blocks in {:.1?}", vd.n_layers, t0.elapsed());
        let t0 = std::time::Instant::now();
        let eg = encoder_graph(&vd, pl.n_vit_h, pl.n_vit_w);
        let ins = encoder_inputs(&eg, &vd, &img, w);
        let got = eg.g.eval(&ins, &[eg.vit, eg.out]);
        let (ev, eo) = (rel(&got[0], &vit), rel(&got[1], &out));
        eprintln!(
            "graph: {} nodes on the interpreter in {:.1?}; vit rel {ev:.2e}, aligner rel {eo:.2e} \
             ({} x {} rows, mean |row| {:.4})",
            eg.g.nodes().len(),
            t0.elapsed(),
            pl.rows(),
            vd.llm_dim,
            sum_abs(&out) / out.len() as f64
        );
        assert!(ev < 1e-12 && eo < 1e-12, "vit {ev:e} aligner {eo:e}");
    }

    /// Image spliced into the real DS prefill: the real image planned at an
    /// 8-position budget (`VISION_SPAN`; 2 × 2 tokens, so the f64 expert
    /// tables fit), its
    /// real encoder rows and delimiters in the span, `DS_REAL_LAYERS` (0,2)
    /// real layers; graph prefill logits against the reference with the
    /// image present, then the derived step for the next text token.
    #[test]
    #[ignore]
    fn real_vl_prefill_matches_reference() {
        use crate::ir_ds_prefill::tests::{Bugs, Ref};
        use crate::ir_ds_prefill::{derived_step, prefill_graph, DsDims, SEQ};
        use crate::ir_ds_real::{dims_from_config, RealRun, PROMPT};
        use tang_compile::tensor::step::{seed_cache, StepInput};
        let Some(mut run) = RealRun::from_env() else { return eprintln!("no checkpoint") };
        let vd0 = VisionDims::from_config(&run.cfg).unwrap();
        let budget: usize = std::env::var("VISION_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let vd = VisionDims { max_tokens: budget, min_pixels: 0, ..vd0.clone() };
        run.vocab_ids.push(vd.image_token_id);
        run.vocab_ids.sort_unstable();
        run.vocab_ids.dedup();
        let (_, bytes) = test_image();
        let img = preprocess(&bytes, &vd).unwrap();
        let pl = img.plan;
        // the encoder (graph, checked against the reference)
        let w = |n: &str| run.st.f64(n).unwrap();
        let eg = encoder_graph(&vd, pl.n_vit_h, pl.n_vit_w);
        let emb = eg.g.eval(&encoder_inputs(&eg, &vd, &img, w), &[eg.out]).remove(0);
        let (_, emb_ref) = reference(&vd, &img, &w);
        let e_enc = rel(&emb, &emb_ref);
        eprintln!("encoder: plan {pl:?}, span {}, rel {e_enc:.2e}", pl.span());
        assert!(e_enc < 1e-12);

        // prompt: a text token, the image, a text token, then one decoded
        // (short: the f64 graph holds every routed expert of the run)
        let text = [PROMPT[0], vd.image_token_id, PROMPT[1], PROMPT[2]];
        let pr = expand_prompt(&text, &[pl], vd.image_token_id).unwrap();
        let dl = Delims {
            start: run.st.f64("image_start").unwrap(),
            newline: run.st.f64("image_newline").unwrap(),
            end: run.st.f64("image_end").unwrap(),
        };
        let d = vd.llm_dim;
        let (rows, mask) = splice_rows(&pr, &[&emb_ref], &dl, d);
        let t_all = pr.ids.len();
        let t_pre = t_all - 1;
        let toks = run.rows_of(&pr.ids);
        let mut dm0 = dims_from_config(&run.cfg, &run.layer_ids, run.vocab_ids.len(), 0, t_all.max(8));
        dm0.vision = true;
        crate::ir_ds_real::clamp_index_topk(&mut dm0);
        assert!(dm0.engram_layers.is_empty(), "pick DS_REAL_LAYERS without Engram (1, 14)");
        let pg0 = prefill_graph(&dm0);
        let mut wd = run.dense_inputs(&pg0.inputs, &dm0).unwrap();
        let used = std::cell::RefCell::new(vec![std::collections::BTreeSet::new(); dm0.n_layers]);
        let load = |l: usize, e: usize| {
            used.borrow_mut()[l].insert(e);
            run.expert(l, e)
        };
        let t0 = std::time::Instant::now();
        let want = Ref { dm: &dm0, w: &wd, bugs: Bugs::default(), load: Some(&load) }
            .forward_vl(&toks, Some((&rows, &mask)))
            .0;
        let text_only = Ref { dm: &dm0, w: &wd, bugs: Bugs::default(), load: Some(&load) }
            .forward_vl(&toks, None)
            .0;
        let used: Vec<Vec<usize>> =
            used.into_inner().into_iter().map(|s| s.into_iter().collect()).collect();
        eprintln!(
            "reference: {t_all} tokens (image span {}) in {:.1?}; experts per layer {:?}",
            pl.span(),
            t0.elapsed(),
            used.iter().map(|u| u.len()).collect::<Vec<_>>()
        );
        let nrows = used.iter().map(|u| u.len()).max().unwrap();
        let dm = DsDims { expert_rows: nrows, ..dm0.clone() };
        let pg = prefill_graph(&dm);
        for (l, u) in used.iter().enumerate() {
            run.expert_tables(l, u, nrows, &dm, &mut wd);
        }
        let mut base: Vec<Vec<f64>> = pg
            .inputs
            .iter()
            .map(|s| match s.name.as_str() {
                "tokens" => toks[..t_pre].to_vec(),
                "image_rows" => rows[..t_pre * d].to_vec(),
                "image_mask" => mask[..t_pre].to_vec(),
                n => wd.remove(n).unwrap_or_else(|| panic!("no input {n}")),
            })
            .collect();
        let v = dm.vocab;
        let t0 = std::time::Instant::now();
        let g = pg.g.bind(&[(SEQ, t_pre)]).unwrap();
        let st = derived_step(&pg, &dm, 1).unwrap();
        let mut outs = vec![pg.logits];
        outs.extend(st.caches.iter().map(|c| c.node));
        let got = g.eval(&base, &outs);
        let e_pre = rel(&got[0], &want[..t_pre * v]);
        let moved = rel(&text_only[..t_pre * v], &want[..t_pre * v]);
        eprintln!(
            "graph prefill T={t_pre} with the image: rel {e_pre:.2e} vs reference ({:.1?}); \
             text-only logits differ by {moved:.2e}",
            t0.elapsed()
        );
        // the derived step: the next text token, no image inputs
        let caches: Vec<Vec<f64>> = st
            .caches
            .iter()
            .zip(&got[1..])
            .map(|(c, val)| {
                let shape: Vec<usize> = pg
                    .g
                    .shape(c.node)
                    .iter()
                    .map(|dd| if *dd == Dim::Sym(SEQ) { t_pre } else { dd.n() })
                    .collect();
                seed_cache(c, &shape, val, t_pre, dm.ctx)
            })
            .collect();
        let sins: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&si| match si {
                StepInput::Pos => vec![t_pre as f64],
                StepInput::New(i) => match pg.inputs[i].name.as_str() {
                    "tokens" => toks[t_pre..].to_vec(),
                    "image_rows" => vec![0.0; d],
                    "image_mask" => vec![0.0],
                    n => panic!("new input {n}"),
                },
                StepInput::Same(i) => std::mem::take(&mut base[i]),
                StepInput::Cache(c) => caches[c].clone(),
            })
            .collect();
        let sgot = st.graph.eval(&sins, &st.outputs);
        let e_step = rel(&sgot[0], &want[t_pre * v..]);
        eprintln!("derived step at position {t_pre} after the image: rel {e_step:.2e}");
        assert!(moved > 1e-6, "the image must change the logits");
        assert!(e_pre < 1e-9, "prefill {e_pre:e}");
        assert!(e_step < 1e-9, "step {e_step:e}");
    }

    /// The kernels the prefill lowering makes of one real-size block
    /// (`VISION_LOWER_PRINT=1` prints each kernel's kind and points).
    #[test]
    #[ignore]
    fn encoder_lowering_kernels() {
        let nl = std::env::var("VISION_LOWER_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let vd = VisionDims { n_layers: nl, ..VisionDims::tiny(1) };
        let vd = VisionDims { dim: 1024, n_heads: 16, inter: 2816, patch: 14, llm_dim: 5120, ..vd };
        let eg = encoder_graph(&vd, 39, 39);
        let (g, o) = eg.g.eliminate_common(&eg.lower_outputs());
        let kinds = |g: &TensorGraph| {
            let mut c = std::collections::BTreeMap::<String, usize>::new();
            for n in g.nodes() {
                if let tang_compile::tensor::Op::Comp(cp) = &n.op {
                    if let Some(m) = &cp.monoid {
                        let k = format!("{:?}", m.kind());
                        let k = if k == "OnlineSoftmax" { format!("{k} out {:?} red {:?}", cp.out, cp.red) } else { k };
                        *c.entry(format!("{k}/w{}", m.width)).or_default() += 1;
                    }
                }
            }
            c
        };
        eprintln!("before: {:?}", kinds(&g));
        let (g2, o2) = g.stabilize_softmax(&o);
        eprintln!("after stabilize_softmax: {:?}", kinds(&g2));
        let (g2, o2) = g2.substitute_maps(&o2);
        let (g2, o2) = g2.eliminate_common(&o2);
        eprintln!("after substitute+cse: {:?}", kinds(&g2));
        let (g2, _) = g2.merge_siblings(&o2);
        eprintln!("after merge_siblings: {:?}", kinds(&g2));
        {
            // the reference: the training block's forward (flash there)
            let dims = crate::ir_block::BlockDims {
                seq: 256, kv_len: 256, rope: true, bias: true, d_model: 576,
                n_heads: 9, n_kv_heads: 3, head_dim: 64, ff_dim: 1536,
            };
            let (bg, bo) = crate::ir_block::transformer_block(&dims);
            eprintln!("block before: {:?}", kinds(&bg));
            let (bg, bo) = bg.stabilize_softmax(&[bo]);
            eprintln!("block after stabilize: {:?}", kinds(&bg));
            let (bg, bo) = bg.substitute_maps(&bo);
            let (bg, bo) = bg.eliminate_common(&bo);
            let (bg, _) = bg.merge_siblings(&bo);
            eprintln!("block after merge_siblings: {:?}", kinds(&bg));
        }
        let bf = match std::env::var("VISION_BF16").as_deref() {
            Ok("act") => Bf16::Activations,
            Ok(_) => Bf16::Weights,
            Err(_) => Bf16::Off,
        };
        let (p, _) = lower_prefill(&g, &o, bf);
        for (i, k) in p.kernels.iter().enumerate() {
            let t = k.source.lines().find(|l| l.contains("//")).unwrap_or("").trim();
            eprintln!("k{i:<3} pts {:<9} wg {:<4} n {:<6} {}", k.points, k.workgroup_size, k.count, t.trim_start_matches("// Auto-generated by tang-compile: "));
            if std::env::var("VISION_LOWER_SRC").ok().and_then(|v| v.parse::<usize>().ok()) == Some(i) {
                eprintln!("{}", k.source);
            }
        }
    }
}
