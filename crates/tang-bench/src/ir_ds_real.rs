//! DeepSeek-V4.1-Flash at real size from its derived graph
//! ([`crate::ir_ds_prefill`]): dims from DeepSeek's `inference/config.json`,
//! each graph input bound to its checkpoint tensor (dequantized from the
//! checkpoint's own formats), and the CPU check against the f64 reference.
//!
//! The checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, 48 shards, 510 GB):
//! dense weights `F8_E4M3` with `F8_E8M0` scales per 32 × 32 block, routed
//! experts e2m1 fp4 packed two to an `I8` byte (low nibble first) with one
//! `F8_E8M0` scale per row × 32, norms / embed / head / router / compressor
//! / indexer key projections `BF16`, mHC and sinks `F32`. The `safetensors`
//! crate can't read `F8_E8M0` headers, so [`St`] reads them itself.
//! `scripts/ds-fetch-layers.py` fetches a few layers by range requests.
//!
//! A CPU check at real size can't hold a layer's 384 experts in f64 (36 GB a
//! tensor), so the graph is built with [`DsDims::expert_rows`]: the expert
//! tensors hold only the experts the run routes to, and a slot map turns an
//! expert id into its row. Graph layer `i` reads checkpoint layer
//! `layer_ids[i]`, which lets a short graph skip an Engram layer (1, 14):
//! neither the graph nor the reference writes Engram.

use std::collections::HashMap;
use std::path::Path;

use tang_compile::tensor::Format;

use crate::ir_ds_prefill::DsDims;

/// The default checkpoint subset on mew (`DS41_DIR` overrides): config.json,
/// inference_config.json and the shards of layers 0–2 with embed / head.
pub const DEFAULT_DIR: &str = "/home/cam/ds41-l012";

pub fn ds_dir() -> String {
    std::env::var("DS41_DIR").unwrap_or_else(|_| DEFAULT_DIR.into())
}

/// OCP e4m3fn: bias 7, no infinities, 0x7f / 0xff are NaN.
pub fn e4m3(v: u8) -> f64 {
    let sign = if v & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((v >> 3) & 0xF) as i32;
    let m = (v & 7) as f64;
    if e == 15 && v & 7 == 7 {
        return f64::NAN;
    }
    if e == 0 {
        sign * m * 2f64.powi(-9)
    } else {
        sign * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// e8m0: a bare exponent, bias 127 (0xff is NaN; 0 is 2⁻¹²⁷).
pub fn e8m0(v: u8) -> f64 {
    if v == 0xFF {
        return f64::NAN;
    }
    2f64.powi(v as i32 - 127)
}

/// e2m1 by code (bit 3 the sign).
pub fn fp4(code: u8) -> f64 {
    const GRID: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = GRID[(code & 7) as usize];
    if code & 8 != 0 {
        -m
    } else {
        m
    }
}

struct Info {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// The safetensors shards of a directory, memory-mapped.
pub struct St {
    maps: Vec<memmap2::Mmap>,
    t: HashMap<String, Info>,
}

impl St {
    pub fn open(dir: &Path) -> Result<St, String> {
        let mut maps = Vec::new();
        let mut t = HashMap::new();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        for p in files {
            let f = std::fs::File::open(&p).map_err(|e| e.to_string())?;
            // SAFETY: read-only map of a file nothing else writes while we run
            let m = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?;
            let n = u64::from_le_bytes(m[..8].try_into().unwrap()) as usize;
            let hdr: serde_json::Value = serde_json::from_slice(&m[8..8 + n])
                .map_err(|e| format!("{}: {e}", p.display()))?;
            for (k, v) in hdr.as_object().unwrap() {
                if k == "__metadata__" {
                    continue;
                }
                let off = v["data_offsets"].as_array().unwrap();
                t.insert(
                    k.clone(),
                    Info {
                        file: maps.len(),
                        dtype: v["dtype"].as_str().unwrap().into(),
                        shape: v["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|d| d.as_u64().unwrap() as usize)
                            .collect(),
                        start: 8 + n + off[0].as_u64().unwrap() as usize,
                        end: 8 + n + off[1].as_u64().unwrap() as usize,
                    },
                );
            }
            maps.push(m);
        }
        Ok(St { maps, t })
    }

    pub fn has(&self, name: &str) -> bool {
        self.t.contains_key(name)
    }

    /// `(dtype, stored shape)`.
    pub fn info(&self, name: &str) -> Option<(&str, &[usize])> {
        self.t.get(name).map(|i| (i.dtype.as_str(), &i.shape[..]))
    }

    fn bytes(&self, name: &str) -> Result<&[u8], String> {
        let i = self
            .t
            .get(name)
            .ok_or_else(|| format!("{name}: not in the checkpoint"))?;
        Ok(&self.maps[i.file][i.start..i.end])
    }

    /// The logical shape (fp4 `I8` rows hold two values a byte).
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        let i = self.t.get(name)?;
        let mut s = i.shape.clone();
        if i.dtype == "I8" || i.dtype == "U8" {
            *s.last_mut().unwrap() *= 2;
        }
        Some(s)
    }

    /// Rows `[r0, r1)` of a tensor as f64 (every value exact), quantized
    /// weights dequantized with their `.scale` sibling: per 32 × 32 block,
    /// or per row × 32 when the scale has one row per weight row.
    pub fn rows(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f64>, String> {
        let i = &self.t[name];
        let shape = self.shape(name).unwrap();
        let cols: usize = shape[1..].iter().product();
        let b = self.bytes(name)?;
        let raw: Vec<f64> = match i.dtype.as_str() {
            "F32" => b[r0 * cols * 4..r1 * cols * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect(),
            "BF16" => b[r0 * cols * 2..r1 * cols * 2]
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
                .collect(),
            "F8_E4M3" => b[r0 * cols..r1 * cols].iter().map(|&v| e4m3(v)).collect(),
            "F8_E8M0" => b[r0 * cols..r1 * cols].iter().map(|&v| e8m0(v)).collect(),
            "I64" => b[r0 * cols * 8..r1 * cols * 8]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect(),
            "I8" | "U8" => b[r0 * cols / 2..r1 * cols / 2]
                .iter()
                .flat_map(|&v| [fp4(v & 0xF), fp4(v >> 4)])
                .collect(),
            d => return Err(format!("{name}: dtype {d}")),
        };
        if !matches!(i.dtype.as_str(), "F8_E4M3" | "I8" | "U8") {
            return Ok(raw);
        }
        let sname = format!(
            "{}scale",
            name.strip_suffix("weight")
                .ok_or_else(|| format!("{name}: quantized non-weight"))?
        );
        let ss = self
            .shape(&sname)
            .ok_or_else(|| format!("{sname}: missing"))?;
        let rows = shape[0];
        let s = self.rows(&sname, 0, ss[0])?;
        let mut out = raw;
        let per_row = ss == [rows, cols / 32];
        if !per_row && ss != [rows.div_ceil(32), cols.div_ceil(32)] {
            return Err(format!("{sname}: scale {ss:?} for weight {shape:?}"));
        }
        for r in r0..r1 {
            let sr = if per_row { r } else { r / 32 };
            for c in 0..cols {
                out[(r - r0) * cols + c] *= s[sr * ss[1] + c / 32];
            }
        }
        Ok(out)
    }

    /// The whole tensor as f64.
    pub fn f64(&self, name: &str) -> Result<Vec<f64>, String> {
        let n = self
            .shape(name)
            .ok_or_else(|| format!("{name}: not in the checkpoint"))?[0];
        self.rows(name, 0, n)
    }

    /// The device format of a checkpoint weight: [`Format::Fp8B32`] for an
    /// `F8_E4M3` weight (32 × 32 block scales), [`Format::Mxfp4`] for an fp4
    /// `I8` one (row × 32 scales); `None` for anything stored unquantized.
    pub fn native_format(&self, name: &str) -> Option<Format> {
        match self.t.get(name)?.dtype.as_str() {
            "F8_E4M3" => Some(Format::Fp8B32),
            "I8" | "U8" => Some(Format::Mxfp4),
            _ => None,
        }
    }

    /// `(codes, scales)`: the raw bytes of a quantized weight and of its
    /// `.scale` sibling, after checking the scale layout matches
    /// [`St::native_format`]'s.
    pub fn native_parts(&self, name: &str) -> Result<(Format, &[u8], &[u8]), String> {
        let f = self
            .native_format(name)
            .ok_or_else(|| format!("{name}: not a quantized weight"))?;
        let shape = self.shape(name).unwrap();
        let (rows, cols) = (shape[0], shape[1..].iter().product::<usize>());
        let sname = format!("{}scale", name.strip_suffix("weight").unwrap());
        let ss = self
            .shape(&sname)
            .ok_or_else(|| format!("{sname}: missing"))?;
        let want = match f {
            Format::Fp8B32 => vec![rows / 32, cols / 32],
            _ => vec![rows, cols / 32],
        };
        if ss != want || rows % 32 != 0 || cols % 32 != 0 {
            return Err(format!("{sname}: scale {ss:?} for {f:?} weight {shape:?}"));
        }
        Ok((f, self.bytes(name)?, self.bytes(&sname)?))
    }

    /// A quantized weight as its device buffer: codes then scales
    /// ([`Format::Fp8B32`] / [`Format::Mxfp4`] layout).
    pub fn native(&self, name: &str) -> Result<(Format, Vec<u8>), String> {
        let (f, c, s) = self.native_parts(name)?;
        let mut v = Vec::with_capacity(c.len() + s.len());
        v.extend_from_slice(c);
        v.extend_from_slice(s);
        Ok((f, v))
    }
}

/// A byte buffer as the f32 words a device upload takes (bit patterns,
/// zero-padded to a word).
pub fn words_of(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            f32::from_bits(u32::from_le_bytes(w))
        })
        .collect()
}

/// `dm` with its indexer's top-k clamped to the context: a context of
/// `ctx` positions has at most `ctx` compressed rows, so `index_topk ≥ ctx`
/// picks every row either way (the reference clamps the same way), but a
/// [`tang_compile::tensor::Monoid::topk`] of 512 lowers to a reduction
/// whose state is 1024 wide — a 245 MB kernel source at real size. For the
/// short-context checks.
pub fn clamp_index_topk(dm: &mut DsDims) {
    dm.index_topk = dm.index_topk.min(dm.ctx);
}

/// DeepSeek's `inference/config.json`.
pub fn load_config(dir: &Path) -> Result<serde_json::Value, String> {
    let p = dir.join("inference_config.json");
    let s = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

/// [`DsDims`] at real size for graph layers reading checkpoint layers
/// `layer_ids`, a vocab of `vocab` rows (a subset of the head, see
/// [`RealRun`]) and `expert_rows` bound expert rows (0: all of them).
pub fn dims_from_config(
    c: &serde_json::Value,
    layer_ids: &[usize],
    vocab: usize,
    expert_rows: usize,
    ctx: usize,
) -> DsDims {
    let u = |k: &str| c[k].as_u64().unwrap_or_else(|| panic!("config {k}")) as usize;
    let f = |k: &str| c[k].as_f64().unwrap_or_else(|| panic!("config {k}"));
    let list = |k: &str| -> Vec<usize> {
        c[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    let ratios = list("compress_ratios");
    let at = |src: &[usize]| -> Vec<usize> {
        (0..layer_ids.len())
            .filter(|&i| src.contains(&layer_ids[i]))
            .collect()
    };
    DsDims {
        vocab,
        dim: u("dim"),
        n_layers: layer_ids.len(),
        n_heads: u("n_heads"),
        head_dim: u("head_dim"),
        rope_head_dim: u("rope_head_dim"),
        q_lora_rank: u("q_lora_rank"),
        o_groups: u("o_groups"),
        o_lora_rank: u("o_lora_rank"),
        window: u("window_size"),
        compress_ratios: layer_ids.iter().map(|&l| ratios[l]).collect(),
        kv_source_layers: at(&list("kv_source_layers")),
        index_source_layers: at(&list("index_source_layers")),
        index_n_heads: u("index_n_heads"),
        index_head_dim: u("index_head_dim"),
        index_topk: u("index_topk"),
        n_experts: u("n_routed_experts"),
        n_active: u("n_activated_experts"),
        moe_inter: u("moe_inter_dim"),
        route_scale: f("route_scale"),
        norm_topk_prob: c
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        swiglu_limit: f("swiglu_limit"),
        hc: u("hc_mult"),
        sinkhorn_iters: u("hc_sinkhorn_iters"),
        hc_eps: f("hc_eps"),
        eps: f("norm_eps"),
        ctx,
        expert_rows,
        candidate_source: layer_ids
            .iter()
            .position(|&l| l as u64 == c["candidate_source_layer"].as_u64().unwrap_or(u64::MAX)),
        candidate_topk_blocks: u("candidate_topk_blocks"),
        candidate_block_size: u("candidate_block_size"),
        engram_layers: at(&list("engram_layer_ids")),
        engram_cols: (u("engram_max_ngram_size") - 1) * u("engram_n_heads"),
        engram_head_dim: u("engram_head_dim"),
        // a real run binds the rows its hashes hit (see `engram_tables`)
        engram_rows: 0,
        vision: false,
        dspark_targets: at(&list("dspark_target_layer_ids")),
        n_mtp: u("n_mtp_layers"),
        dspark_block: u("dspark_block_size"),
        dspark_noise: u("dspark_noise_token_id"),
        dspark_rank: u("dspark_markov_rank"),
        dspark_experts: u("dspark_n_routed_experts"),
        dspark_active: u("dspark_n_activated_experts"),
    }
}

/// RoPE frequencies of a `rd`-wide rotary part, YaRN-corrected when
/// `orig > 0` (`forward.rs`'s `yarn_freqs`, in f64).
pub fn yarn_freqs(c: &serde_json::Value, rd: usize, orig: usize, base: f64) -> Vec<f64> {
    let half = rd / 2;
    let mut fr: Vec<f64> = (0..half)
        .map(|j| 1.0 / base.powf((2 * j) as f64 / rd as f64))
        .collect();
    if orig > 0 {
        let factor = c["rope_factor"].as_f64().unwrap();
        let (bf, bs) = (
            c["beta_fast"].as_f64().unwrap(),
            c["beta_slow"].as_f64().unwrap(),
        );
        let corr = |rot: f64| {
            rd as f64 * (orig as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
        };
        let low = corr(bf).floor().max(0.0);
        let high = corr(bs).ceil().min(rd as f64 - 1.0);
        let span = (high - low).max(1e-3);
        for (j, f) in fr.iter_mut().enumerate() {
            let ramp = ((j as f64 - low) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    fr
}

/// The graph's `freqs` input for a `width`-wide head: zero frequency on the
/// no-rope pairs, then the rotary part's.
fn freqs_input(width: usize, rotary: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; width / 2 - rotary.len()];
    v.extend_from_slice(rotary);
    v
}

/// The checkpoint name of graph input `name` (graph layer `i` → checkpoint
/// layer `layer_ids[i]`).
pub fn ckpt_name(name: &str, layer_ids: &[usize]) -> String {
    match name.strip_prefix("layers.") {
        Some(rest) => {
            let (i, tail) = rest.split_once('.').unwrap();
            format!("layers.{}.{tail}", layer_ids[i.parse::<usize>().unwrap()])
        }
        None => name.into(),
    }
}

/// What a real-size CPU run binds: the checkpoint, the graph layers' source
/// layers and the vocab rows kept.
pub struct RealRun {
    pub st: St,
    pub cfg: serde_json::Value,
    pub layer_ids: Vec<usize>,
    /// Checkpoint token ids of graph vocab rows `0..vocab_ids.len()`.
    pub vocab_ids: Vec<usize>,
}

impl RealRun {
    /// Every non-expert graph input (`dm`'s graph) as f64: checkpoint
    /// tensors (embed / head restricted to `vocab_ids`), computed RoPE
    /// frequencies. Experts and slot maps are left to the caller.
    pub fn dense_inputs(
        &self,
        inputs: &[crate::ir_ds_prefill::PrefillInput],
        dm: &DsDims,
    ) -> Result<HashMap<String, Vec<f64>>, String> {
        let c = &self.cfg;
        let (hd, rd, ihd) = (dm.head_dim, dm.rope_head_dim, dm.index_head_dim);
        let orig = c["original_seq_len"].as_u64().unwrap() as usize;
        let plain = yarn_freqs(c, rd, 0, c["rope_theta"].as_f64().unwrap());
        let yarn = yarn_freqs(c, rd, orig, c["compress_rope_theta"].as_f64().unwrap());
        let mut out = HashMap::new();
        for i in inputs {
            let n = i.name.as_str();
            if i.shape.iter().any(|d| d.fixed().is_none())
                || n.contains(".ffn.experts.")
                || n.ends_with(".ffn.expert_slot")
                || n.ends_with(".engram.hash")
                || n.ends_with(".engram.embed.weight")
            {
                continue;
            }
            let shape: Vec<usize> = i.shape.iter().map(|d| d.n()).collect();
            let v = if n.ends_with(".freqs") && n.starts_with("mtp.") {
                // the MTP stages attend over a plain window: ratio 0
                freqs_input(hd, &plain)
            } else if n.ends_with(".freqs") {
                let l: usize = n.split('.').nth(1).unwrap().parse().unwrap();
                let rot = if dm.compress_ratios[l] > 0 {
                    &yarn
                } else {
                    &plain
                };
                freqs_input(if n.contains("indexer") { ihd } else { hd }, rot)
            } else if n == "embed.weight" || n == "head.weight" || n.contains(".markov_head.") {
                // vocab-indexed: only the kept rows
                let mut v = Vec::with_capacity(self.vocab_ids.len() * shape[1]);
                for &id in &self.vocab_ids {
                    v.extend(self.st.rows(n, id, id + 1)?);
                }
                v
            } else {
                let cn = ckpt_name(n, &self.layer_ids);
                let have = self
                    .st
                    .shape(&cn)
                    .ok_or_else(|| format!("{cn}: not in the checkpoint"))?;
                if have != shape {
                    return Err(format!("{cn}: checkpoint {have:?}, graph {shape:?}"));
                }
                self.st.f64(&cn)?
            };
            assert_eq!(v.len(), shape.iter().product::<usize>(), "{n}");
            out.insert(i.name.clone(), v);
        }
        Ok(out)
    }

    /// The Engram inputs of `ids` (checkpoint token ids, the whole run):
    /// per graph Engram layer, its hashes `[T · cols]` as rows of the
    /// compact table and the table itself, `(name, values)`, plus the
    /// table's row count. The compact table holds the rows
    /// `scripts/ds-fetch-engram-rows.py` fetched (hashed there by the
    /// Python reference); a hash this side computes that it lacks is an
    /// error, so the two hashings must agree.
    #[allow(clippy::type_complexity)]
    pub fn engram_inputs(&self, ids: &[usize]) -> Result<(Vec<(String, Vec<f64>)>, usize), String> {
        let ckpt: Vec<usize> = self.cfg["engram_layer_ids"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect())
            .unwrap_or_default();
        let mut out = Vec::new();
        let mut rows = 0;
        for (i, &l) in self.layer_ids.iter().enumerate() {
            let Some(j) = ckpt.iter().position(|&e| e == l) else {
                continue;
            };
            let meta =
                crate::ir_ds_prefill::EngramMeta::load(&Path::new(&ds_dir()).join("engram.json"))?;
            let have: Vec<i64> = self
                .st
                .f64(&format!("layers.{l}.engram.row_ids"))?
                .into_iter()
                .map(|v| v as i64)
                .collect();
            let hs: Vec<f64> = meta.hashes(ids)[j]
                .iter()
                .flatten()
                .map(|h| {
                    have.binary_search(h)
                        .map(|r| r as f64)
                        .map_err(|_| format!("layer {l}: hash row {h} not fetched"))
                })
                .collect::<Result<_, _>>()?;
            let table = self.st.f64(&format!("layers.{l}.engram.embed.weight"))?;
            if rows != 0 && rows != have.len() {
                return Err("Engram layers with different fetched row counts".into());
            }
            rows = have.len();
            out.push((format!("layers.{i}.engram.hash"), hs));
            out.push((format!("layers.{i}.engram.embed.weight"), table));
        }
        Ok((out, rows))
    }

    /// Expert `e` of graph layer `l`: `(w1 [inter, d], w3 [inter, d], w2 [d, inter])`.
    pub fn expert(&self, l: usize, e: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let p = format!("layers.{}.ffn.experts.{e}", self.layer_ids[l]);
        let w = |k: &str| self.st.f64(&format!("{p}.{k}.weight")).unwrap();
        (w("w1"), w("w3"), w("w2"))
    }

    /// Expert `e` under checkpoint prefix `p` (e.g. `mtp.1`).
    pub fn expert_at(&self, p: &str, e: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let w = |k: &str| {
            self.st
                .f64(&format!("{p}.ffn.experts.{e}.{k}.weight"))
                .unwrap()
        };
        (w("w1"), w("w3"), w("w2"))
    }

    /// The compact expert tensors of graph prefix `p` (checkpoint prefix
    /// `ckpt`) holding `used`, padded to `rows`, and the slot map of `ne`.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_tables_at(
        &self,
        p: &str,
        ckpt: &str,
        used: &[usize],
        rows: usize,
        ne: usize,
        dm: &DsDims,
        into: &mut HashMap<String, Vec<f64>>,
    ) {
        let (d, inter) = (dm.dim, dm.moe_inter);
        let mut w1 = vec![0.0; rows * inter * d];
        let mut w3 = vec![0.0; rows * inter * d];
        let mut w2 = vec![0.0; rows * d * inter];
        let mut slot = vec![0.0; ne];
        for (r, &e) in used.iter().enumerate() {
            let (a, b, c) = self.expert_at(ckpt, e);
            w1[r * inter * d..(r + 1) * inter * d].copy_from_slice(&a);
            w3[r * inter * d..(r + 1) * inter * d].copy_from_slice(&b);
            w2[r * d * inter..(r + 1) * d * inter].copy_from_slice(&c);
            slot[e] = r as f64;
        }
        into.insert(format!("{p}.ffn.experts.w1"), w1);
        into.insert(format!("{p}.ffn.experts.w3"), w3);
        into.insert(format!("{p}.ffn.experts.w2"), w2);
        into.insert(format!("{p}.ffn.expert_slot"), slot);
    }

    /// The compact expert tensors of graph layer `l` holding `used` (in
    /// order, padded with zero rows to `rows`) and the slot map.
    pub fn expert_tables(
        &self,
        l: usize,
        used: &[usize],
        rows: usize,
        dm: &DsDims,
        into: &mut HashMap<String, Vec<f64>>,
    ) {
        let (d, inter) = (dm.dim, dm.moe_inter);
        let mut w1 = vec![0.0; rows * inter * d];
        let mut w3 = vec![0.0; rows * inter * d];
        let mut w2 = vec![0.0; rows * d * inter];
        let mut slot = vec![0.0; dm.n_experts];
        for (r, &e) in used.iter().enumerate() {
            let (a, b, c) = self.expert(l, e);
            w1[r * inter * d..(r + 1) * inter * d].copy_from_slice(&a);
            w3[r * inter * d..(r + 1) * inter * d].copy_from_slice(&b);
            w2[r * d * inter..(r + 1) * d * inter].copy_from_slice(&c);
            slot[e] = r as f64;
        }
        let p = format!("layers.{l}.ffn");
        into.insert(format!("{p}.experts.w1"), w1);
        into.insert(format!("{p}.experts.w3"), w3);
        into.insert(format!("{p}.experts.w2"), w2);
        into.insert(format!("{p}.expert_slot"), slot);
    }
}

impl RealRun {
    /// [`RealRun::expert_tables`]'s `w1` / `w3` / `w2` as [`Format::Mxfp4`]
    /// device buffers (the checkpoint's own bytes: every expert's codes, then
    /// every expert's scales; padding rows zero), keyed like the graph inputs.
    pub fn native_expert_tables(
        &self,
        l: usize,
        used: &[usize],
        rows: usize,
    ) -> Result<Vec<(String, Vec<u8>)>, String> {
        let mut out = Vec::new();
        for k in ["w1", "w3", "w2"] {
            let name =
                |e: usize| format!("layers.{}.ffn.experts.{e}.{k}.weight", self.layer_ids[l]);
            let (_, c0, s0) = self.st.native_parts(&name(used[0]))?;
            let (cn, sn) = (c0.len(), s0.len());
            let mut v = vec![0u8; rows * (cn + sn)];
            for (r, &e) in used.iter().enumerate() {
                let (f, c, s) = self.st.native_parts(&name(e))?;
                if f != Format::Mxfp4 || c.len() != cn || s.len() != sn {
                    return Err(format!("{}: not like expert {}", name(e), used[0]));
                }
                v[r * cn..(r + 1) * cn].copy_from_slice(c);
                v[rows * cn + r * sn..rows * cn + (r + 1) * sn].copy_from_slice(s);
            }
            out.push((format!("layers.{l}.ffn.experts.{k}"), v));
        }
        Ok(out)
    }
}

/// A derived step lowered with its weights in the checkpoint's formats.
pub struct NativeStep {
    pub graph: tang_compile::tensor::TensorGraph,
    /// `[step outputs…, updated caches…]` in `graph`.
    pub outs: Vec<tang_compile::tensor::TensorId>,
    pub program: tang_compile::tensor::cuda_gemv::DecodeProgram,
    pub sched: tang_compile::tensor::int8_sched::BlockSchedule,
    /// Step input index → the format its buffer holds (`None`: f32 values,
    /// a quantized weight's dequantized when some kernel other than the int8
    /// GEMV reads it).
    pub storage: Vec<Option<Format>>,
    /// Quantized weights some other kernel reads, uploaded dequantized.
    pub demoted: Vec<String>,
}

/// The device format of prefill input `name` under [`RealRun`]'s binding:
/// the compact expert tables are [`Format::Mxfp4`], a dense weight its
/// checkpoint tensor's [`St::native_format`].
pub fn native_format_of(run: &RealRun, name: &str) -> Option<Format> {
    if name.contains(".ffn.experts.") {
        return Some(Format::Mxfp4);
    }
    if name == "embed.weight" || name == "head.weight" {
        return None;
    }
    run.st.native_format(&ckpt_name(name, &run.layer_ids))
}

/// The derived step with every quantized weight the int8 GEMV can read
/// stored as the checkpoint stores it ([`tang_compile::tensor::int8_sched`]:
/// Q8x32 activations, the GEMV dequantizing in registers), through the
/// decode passes and `to_cuda_decode`. Caches are written to fresh buffers.
pub fn lower_native_step(
    pg: &crate::ir_ds_prefill::PrefillGraph,
    st: &tang_compile::tensor::step::StepGraph,
    format_of: &dyn Fn(&str) -> Option<Format>,
) -> NativeStep {
    use tang_compile::tensor::step::StepInput;
    use tang_compile::tensor::{Op, Precision, TensorGraph, TensorId};
    let input_node = |g: &TensorGraph, k: usize| {
        (0..g.nodes().len())
            .find(|&i| matches!(g.nodes()[i].op, Op::Input(j) if j == k))
            .map(TensorId::from_index)
            .unwrap()
    };
    let name_of = |k: usize| match st.inputs[k] {
        StepInput::Same(i) => Some(pg.inputs[i].name.as_str()),
        _ => None,
    };
    let mut graph = st.graph.clone();
    for k in 0..st.inputs.len() {
        if let Some(f) = name_of(k).and_then(format_of) {
            graph.set_precision(input_node(&graph, k), Precision::storage(f));
        }
    }
    let mut outs = st.outputs.clone();
    outs.extend(st.caches.iter().map(|c| c.updated));
    let (graph, outs, sched) = graph.schedule_block_weights(&outs);
    let n = outs.len();
    let mut keep = outs.clone();
    keep.extend(&sched.copies);
    let (graph, keep) = graph.substitute_maps(&keep);
    let (graph, keep) = graph.eliminate_common(&keep);
    let outs = keep[..n].to_vec();
    let storage: Vec<Option<Format>> = (0..st.inputs.len())
        .map(|k| {
            let s = graph.nodes()[input_node(&graph, k).index()]
                .precision
                .storage;
            s.is_block().then_some(s)
        })
        .collect();
    let demoted = (0..st.inputs.len())
        .filter(|&k| storage[k].is_none() && name_of(k).and_then(format_of).is_some())
        .map(|k| name_of(k).unwrap().to_string())
        .collect();
    let program = graph.to_cuda_decode(&outs, &Default::default());
    NativeStep {
        graph,
        outs,
        program,
        sched,
        storage,
        demoted,
    }
}

impl RealRun {
    /// The device words of every step input of `ns`: a native-format weight's
    /// checkpoint bytes, anything else `vals[k]` as f32 (NaN as 0: unwritten
    /// cache rows, masked).
    pub fn native_words(
        &self,
        pg: &crate::ir_ds_prefill::PrefillGraph,
        st: &tang_compile::tensor::step::StepGraph,
        ns: &NativeStep,
        used: &[Vec<usize>],
        rows: usize,
        vals: &[Vec<f64>],
    ) -> Result<Vec<Vec<f32>>, String> {
        use tang_compile::tensor::step::StepInput;
        let mut experts: HashMap<String, Vec<u8>> = HashMap::new();
        for (l, u) in used.iter().enumerate() {
            if ns.storage.contains(&Some(Format::Mxfp4)) {
                experts.extend(self.native_expert_tables(l, u, rows)?);
            }
        }
        (0..st.inputs.len())
            .map(|k| match (ns.storage[k], st.inputs[k]) {
                (Some(f), StepInput::Same(i)) => {
                    let name = &pg.inputs[i].name;
                    let bytes = match experts.remove(name) {
                        Some(b) => b,
                        None => {
                            let (g, b) = self.st.native(&ckpt_name(name, &self.layer_ids))?;
                            if g != f {
                                return Err(format!("{name}: {g:?}, scheduled {f:?}"));
                            }
                            b
                        }
                    };
                    Ok(words_of(&bytes))
                }
                (Some(f), _) => Err(format!("step input {k}: {f:?} but not a weight")),
                (None, _) => Ok(vals[k]
                    .iter()
                    .map(|&x| if x.is_nan() { 0.0 } else { x as f32 })
                    .collect()),
            })
            .collect()
    }
}

/// Checkpoint token ids of the check's prompt (BOS, then ids from the
/// tokenizer's range), cycled to the length a run needs.
pub const PROMPT: [usize; 6] = [0, 3476, 25108, 6102, 1871, 89104];

/// The prompt's first `n` tokens.
pub fn prompt(n: usize) -> Vec<usize> {
    PROMPT.iter().copied().cycle().take(n).collect()
}

impl RealRun {
    /// `DS41_DIR` (default [`DEFAULT_DIR`]) with graph layers from
    /// `DS_REAL_LAYERS` (default `0,2`: window-only, then the first
    /// compressed-KV + indexer source; skips layer 1's Engram); the head
    /// rows kept are the prompt's plus the first 2048. `None` without a
    /// checkpoint directory.
    pub fn from_env() -> Option<RealRun> {
        let dir = ds_dir();
        let p = Path::new(&dir);
        if !p.join("inference_config.json").exists() {
            return None;
        }
        let layer_ids: Vec<usize> = std::env::var("DS_REAL_LAYERS")
            .unwrap_or_else(|_| "0,2".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let mut vocab_ids: Vec<usize> = (0..2048).collect();
        vocab_ids.extend(PROMPT);
        // DSpark's noise token: the drafter embeds it
        if let Some(n) = load_config(p)
            .ok()
            .and_then(|c| c["dspark_noise_token_id"].as_u64())
        {
            vocab_ids.push(n as usize);
        }
        vocab_ids.sort_unstable();
        vocab_ids.dedup();
        Some(RealRun {
            st: St::open(p).unwrap(),
            cfg: load_config(p).unwrap(),
            layer_ids,
            vocab_ids,
        })
    }

    /// Graph vocab rows of checkpoint token ids.
    pub fn rows_of(&self, ids: &[usize]) -> Vec<f64> {
        ids.iter()
            .map(|id| self.vocab_ids.binary_search(id).expect("a kept row") as f64)
            .collect()
    }

    /// Where the CPU check records the experts a run of `n` tokens routes
    /// to (one line per graph layer), for runs without the reference.
    pub fn routed_path(&self, n: usize) -> std::path::PathBuf {
        let l: Vec<String> = self.layer_ids.iter().map(|l| l.to_string()).collect();
        Path::new(&ds_dir()).join(format!("routed-{}-{n}.txt", l.join("_")))
    }

    pub fn write_routed(&self, n: usize, used: &[Vec<usize>]) -> std::io::Result<()> {
        let s: String = used
            .iter()
            .map(|u| {
                u.iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
                    + "\n"
            })
            .collect();
        std::fs::write(self.routed_path(n), s)
    }

    pub fn read_routed(&self, n: usize) -> std::io::Result<Vec<Vec<usize>>> {
        Ok(std::fs::read_to_string(self.routed_path(n))?
            .lines()
            .map(|l| l.split_whitespace().map(|e| e.parse().unwrap()).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_ds_prefill::tests::{rel, Bugs, Ref};
    use crate::ir_ds_prefill::{derived_step, prefill_graph, SEQ};
    use std::time::Instant;
    use tang_compile::tensor::step::seed_cache;
    use tang_compile::tensor::step::StepInput;
    use tang_compile::tensor::TensorId;

    fn open() -> Option<RealRun> {
        let r = RealRun::from_env();
        if r.is_none() {
            eprintln!("no {}/inference_config.json", ds_dir());
        }
        r
    }

    fn rss_gb() -> f64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM"))
                    .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
            })
            .map_or(0.0, |kb| kb / 1e6)
    }

    /// The decoders against hand-computed values and the checkpoint's own
    /// scale layout: every dense fp8 weight of a layer has a 32 × 32 block
    /// scale, every expert a per-row × 32 one.
    #[test]
    #[ignore]
    fn formats_decode() {
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0x01), 2f64.powi(-9));
        assert!(e4m3(0x7f).is_nan());
        assert_eq!(e8m0(127), 1.0);
        assert_eq!(e8m0(130), 8.0);
        assert_eq!(fp4(0x7), 6.0);
        assert_eq!(fp4(0xB), -1.5);
        let Some(run) = open() else { return };
        let mut seen: std::collections::BTreeMap<String, usize> = Default::default();
        for name in run.st.t.keys() {
            let (dt, _) = run.st.info(name).unwrap();
            *seen.entry(dt.to_string()).or_default() += 1;
        }
        eprintln!("dtypes in the subset: {seen:?}");
        let w = run.st.f64("layers.0.attn.wq_a.weight").unwrap();
        let ms = (w.iter().map(|v| v * v).sum::<f64>() / w.len() as f64).sqrt();
        eprintln!("layers.0.attn.wq_a: {} values, rms {ms:.3e}", w.len());
        assert!(w.iter().all(|v| v.is_finite()) && ms > 1e-4 && ms < 1.0);
        let e = run.st.f64("layers.0.ffn.experts.7.w2.weight").unwrap();
        let ms = (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
        eprintln!(
            "layers.0.ffn.experts.7.w2: {} values, rms {ms:.3e}",
            e.len()
        );
        assert!(e.iter().all(|v| v.is_finite()) && ms > 1e-4 && ms < 1.0);
    }

    /// The real-size graph (real dims, real weights, `DS_REAL_LAYERS`
    /// checkpoint layers, default 0 and 2) on the f64 interpreter against
    /// the f64 reference (`ir_ds_prefill`'s transcription of forward.rs):
    /// prefill logits over `T` tokens, then the step derived from the same
    /// graph, seeded from the prefill's caches, for token `T + 1`.
    #[test]
    #[ignore]
    fn real_layers_match_reference() {
        let Some(run) = open() else { return };
        let t_pre: usize = std::env::var("DS_REAL_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let toks = run.rows_of(&prompt(t_pre + 1));
        let ctx = 8.max(t_pre + 1);
        let mut dm0 = dims_from_config(&run.cfg, &run.layer_ids, run.vocab_ids.len(), 0, ctx);
        let (engram, erows) = run.engram_inputs(&prompt(t_pre + 1)).unwrap();
        dm0.engram_rows = erows;
        if !engram.is_empty() {
            eprintln!(
                "engram: graph layers {:?}, {erows} table rows bound (of the host-resident table)",
                dm0.engram_layers
            );
        }
        eprintln!(
            "real dims: layers {:?} (ratios {:?}), d {}, {} heads x {}, experts {} top {}, vocab rows {}",
            run.layer_ids,
            dm0.compress_ratios,
            dm0.dim,
            dm0.n_heads,
            dm0.head_dim,
            dm0.n_experts,
            dm0.n_active,
            dm0.vocab
        );
        let t0 = Instant::now();
        let pg0 = prefill_graph(&dm0);
        let mut w = run.dense_inputs(&pg0.inputs, &dm0).unwrap();
        w.extend(engram);
        eprintln!(
            "dense inputs: {} tensors, {:.2} G values, {:.1?}",
            w.len(),
            w.values().map(|v| v.len()).sum::<usize>() as f64 / 1e9,
            t0.elapsed()
        );

        // the reference over all T + 1 tokens, loading each routed expert
        // from the checkpoint as it is picked, recording the picks
        let t0 = Instant::now();
        let used = std::cell::RefCell::new(vec![std::collections::BTreeSet::new(); dm0.n_layers]);
        let load = |l: usize, e: usize| {
            used.borrow_mut()[l].insert(e);
            run.expert(l, e)
        };
        let want = Ref {
            dm: &dm0,
            w: &w,
            bugs: Bugs::default(),
            load: Some(&load),
        }
        .forward(&toks);
        let used: Vec<Vec<usize>> = used
            .into_inner()
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();
        run.write_routed(toks.len(), &used).unwrap();
        eprintln!(
            "reference: {} tokens in {:.1?}; experts routed per layer {:?}",
            toks.len(),
            t0.elapsed(),
            used.iter().map(|u| u.len()).collect::<Vec<_>>()
        );

        // the graph with compact expert tables
        let rows = used.iter().map(|u| u.len()).max().unwrap();
        let dm = DsDims {
            expert_rows: rows,
            ..dm0.clone()
        };
        let pg = prefill_graph(&dm);
        for (l, u) in used.iter().enumerate() {
            run.expert_tables(l, u, rows, &dm, &mut w);
        }
        // the graph's inputs, moved out of `w` (one copy of the weights)
        let mut hashes: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut base: Vec<Vec<f64>> = pg
            .inputs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if s.name == "tokens" {
                    toks[..t_pre].to_vec()
                } else if s.name.ends_with(".engram.hash") {
                    let c = s.shape[1].n();
                    hashes.insert(i, w[&s.name].clone());
                    w[&s.name][..t_pre * c].to_vec()
                } else {
                    w.remove(&s.name)
                        .unwrap_or_else(|| panic!("no input {}", s.name))
                }
            })
            .collect();
        drop(w);
        let v = dm.vocab;
        let argmax = |r: &[f64]| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
        let rows_agree = |got: &[f64], want: &[f64]| {
            got.chunks(v)
                .zip(want.chunks(v))
                .filter(|(a, b)| argmax(a) == argmax(b))
                .count()
        };

        let t0 = Instant::now();
        let g = pg.g.bind(&[(SEQ, t_pre)]).unwrap();
        let mut outs = vec![pg.logits];
        let st = derived_step(&pg, &dm, 1).unwrap();
        let cache_nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
        outs.extend(&cache_nodes);
        let got = g.eval(&base, &outs);
        let e_pre = rel(&got[0], &want[..t_pre * v]);
        eprintln!(
            "graph prefill T={t_pre}: max rel err {e_pre:.2e}, top-1 {}/{t_pre}, {:.1?}, peak RSS {:.1} GB",
            rows_agree(&got[0], &want[..t_pre * v]),
            t0.elapsed(),
            rss_gb()
        );

        // the derived step: k = 1 at position T, caches seeded from the prefill
        let t0 = Instant::now();
        eprintln!(
            "derived step: {} nodes, {} caches: {}",
            st.graph.nodes().len(),
            st.caches.len(),
            st.caches
                .iter()
                .map(|c| format!("{:?}", st.graph.shape(c.input)))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let caches: Vec<Vec<f64>> = cache_nodes
            .iter()
            .zip(&got[1..])
            .zip(&st.caches)
            .map(|((&n, val), c)| {
                let shape: Vec<usize> =
                    pg.g.shape(n)
                        .iter()
                        .map(|d| {
                            if *d == tang_compile::tensor::Dim::Sym(SEQ) {
                                t_pre
                            } else {
                                d.n()
                            }
                        })
                        .collect();
                seed_cache(c, &shape, val, t_pre, dm.ctx)
            })
            .collect();
        let sins: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&si| match si {
                StepInput::Pos => vec![t_pre as f64],
                StepInput::New(i) => match hashes.get(&i) {
                    Some(h) => {
                        let c = pg.inputs[i].shape[1].n();
                        h[t_pre * c..(t_pre + 1) * c].to_vec()
                    }
                    None => toks[t_pre..t_pre + 1].to_vec(),
                },
                StepInput::Same(i) => std::mem::take(&mut base[i]),
                StepInput::Cache(c) => caches[c].clone(),
            })
            .collect();
        let sgot = st.graph.eval(&sins, &st.outputs);
        let e_step = rel(&sgot[0], &want[t_pre * v..]);
        eprintln!(
            "derived step at position {t_pre}: max rel err {e_step:.2e}, top-1 {}/1, {:.1?}, peak RSS {:.1} GB",
            rows_agree(&sgot[0], &want[t_pre * v..]),
            t0.elapsed(),
            rss_gb()
        );
        assert!(e_pre < 1e-9, "prefill {e_pre:e}");
        assert!(e_step < 1e-9, "step {e_step:e}");
    }

    /// The DSpark drafter at real size on the checkpoint's three MTP
    /// stages against the f64 reference: drafts, logits and confidences
    /// after `DS_SPARK_T` main rows (default 1), then the drafter's derived
    /// step for the next row. The main hiddens are stand-ins (the prompt's
    /// embedding rows, one copy per target layer): real ones need all 40
    /// main layers, which a CPU check doesn't hold. Each eval binds only
    /// the experts its rows route to.
    #[test]
    #[ignore]
    fn real_dspark_matches_reference() {
        use crate::ir_ds_spark::tests::spark_ref;
        use crate::ir_ds_spark::{spark_graph, spark_step};
        let Some(mut run) = open() else { return };
        if !run.st.has("mtp.0.main_proj.weight") {
            eprintln!("no MTP stages in {}", ds_dir());
            return;
        }
        run.layer_ids = vec![37, 38, 39];
        let t_pre: usize = std::env::var("DS_SPARK_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let ids = prompt(t_pre + 2);
        let toks = run.rows_of(&ids);
        let mut dm0 = dims_from_config(
            &run.cfg,
            &run.layer_ids,
            run.vocab_ids.len(),
            0,
            8.max(t_pre + 2),
        );
        dm0.dspark_noise = run.rows_of(&[dm0.dspark_noise])[0] as usize;
        let (d, nt, blk) = (dm0.dim, dm0.dspark_targets.len(), dm0.dspark_block);
        eprintln!(
            "dspark real: {} stages, block {blk}, {} experts top {}, markov rank {}, targets {nt}",
            dm0.n_mtp, dm0.dspark_experts, dm0.dspark_active, dm0.dspark_rank
        );
        let t0 = Instant::now();
        let sg0 = spark_graph(&dm0);
        let mut w = run.dense_inputs(&sg0.inputs, &dm0).unwrap();
        eprintln!(
            "dense inputs: {} tensors, {:.2} G values, {:.1?}",
            w.len(),
            w.values().map(|v| v.len()).sum::<usize>() as f64 / 1e9,
            t0.elapsed()
        );
        // stand-in main hiddens: each row's token embedding, nt times
        let n = t_pre + 1;
        let emb = &w["embed.weight"];
        let mut mains = Vec::with_capacity(n * nt * d);
        for &r in &toks[..n] {
            for _ in 0..nt {
                mains.extend_from_slice(&emb[r as usize * d..(r as usize + 1) * d]);
            }
        }
        let next: Vec<u32> = toks[1..=n].iter().map(|&x| x as u32).collect();
        // the reference over n rows, recording each pick in order
        let calls = std::cell::RefCell::new(Vec::new());
        let load = |s: usize, e: usize| {
            calls.borrow_mut().push((s, e));
            run.expert_at(&format!("mtp.{s}"), e)
        };
        let t0 = Instant::now();
        let r = Ref {
            dm: &dm0,
            w: &w,
            bugs: Bugs::default(),
            load: Some(&load),
        };
        let n0 = {
            spark_ref(&r, &mains[..t_pre * nt * d], &next[..t_pre]);
            calls.borrow().len()
        };
        calls.borrow_mut().clear();
        let (rids, rlg, rcf) = spark_ref(&r, &mains, &next);
        let calls = calls.into_inner();
        eprintln!("reference: {n} rows in {:.1?}", t0.elapsed());
        let used = |cs: &[(usize, usize)]| -> Vec<Vec<usize>> {
            (0..dm0.n_mtp)
                .map(|s| {
                    let mut u: Vec<usize> = cs.iter().filter(|c| c.0 == s).map(|c| c.1).collect();
                    u.sort_unstable();
                    u.dedup();
                    u
                })
                .collect()
        };
        let (u_pre, u_step) = (used(&calls[..n0]), used(&calls[n0..]));
        let rows = u_pre.iter().chain(&u_step).map(|u| u.len()).max().unwrap();
        eprintln!(
            "experts per stage: prefill {:?}, step {:?}",
            u_pre.iter().map(|u| u.len()).collect::<Vec<_>>(),
            u_step.iter().map(|u| u.len()).collect::<Vec<_>>()
        );
        let dm = DsDims {
            expert_rows: rows,
            ..dm0.clone()
        };
        let sg = spark_graph(&dm);
        let bind = |w: &mut HashMap<String, Vec<f64>>, u: &[Vec<usize>]| {
            for (s, us) in u.iter().enumerate() {
                let p = format!("mtp.{s}");
                run.expert_tables_at(&p, &p, us, rows, dm.dspark_experts, &dm, w);
            }
        };
        bind(&mut w, &u_pre);
        // moved out of `w` (one copy of the weights); expert tables taken,
        // the rest cloned back for the step
        let ins = |w: &mut HashMap<String, Vec<f64>>, m: &[f64], nx: &[u32]| -> Vec<Vec<f64>> {
            sg.inputs
                .iter()
                .map(|s| match s.name.as_str() {
                    "mains" => m.to_vec(),
                    "next" => nx.iter().map(|&x| x as f64).collect(),
                    k => w.remove(k).unwrap_or_else(|| panic!("no input {k}")),
                })
                .collect()
        };
        let st = spark_step(&sg, &dm, 1).unwrap();
        let t0 = Instant::now();
        let g = sg.g.bind(&[(SEQ, t_pre)]).unwrap();
        let mut outs = vec![sg.ids, sg.logits, sg.conf];
        outs.extend(st.caches.iter().map(|c| c.node));
        let mut pins = ins(&mut w, &mains[..t_pre * nt * d], &next[..t_pre]);
        let got = g.eval(&pins, &outs);
        // the dense weights back for the step (the expert tables rebound)
        for (s, v) in sg.inputs.iter().zip(pins.iter_mut()) {
            if s.name != "mains" && s.name != "next" && !s.name.contains(".ffn.expert") {
                w.insert(s.name.clone(), std::mem::take(v));
            }
        }
        drop(pins);
        let gids: Vec<u32> = got[0].iter().map(|&x| x as u32).collect();
        let want_ids: Vec<u32> = rids[..t_pre].concat();
        let el = rel(&got[1], &rlg[..t_pre].concat());
        let ec = rel(&got[2], &rcf[..t_pre].concat());
        eprintln!(
            "graph prefill T={t_pre}: drafts {:?} (reference {:?}), logits err {el:.2e}, conf err {ec:.2e}, {:.1?}, peak RSS {:.1} GB",
            gids, want_ids, t0.elapsed(), rss_gb()
        );
        assert_eq!(gids, want_ids);
        // the derived step for row t_pre, with the step rows' experts bound
        let caches: Vec<Vec<f64>> = st
            .caches
            .iter()
            .zip(&got[3..])
            .map(|(c, val)| {
                let shape: Vec<usize> =
                    sg.g.shape(c.node)
                        .iter()
                        .map(|x| {
                            if *x == tang_compile::tensor::Dim::Sym(SEQ) {
                                t_pre
                            } else {
                                x.n()
                            }
                        })
                        .collect();
                seed_cache(c, &shape, val, t_pre, dm.ctx)
            })
            .collect();
        drop(got);
        bind(&mut w, &u_step);
        let mut step_ins = ins(&mut w, &mains[t_pre * nt * d..], &next[t_pre..]);
        let sins: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&si| match si {
                StepInput::Pos => vec![t_pre as f64],
                StepInput::New(i) | StepInput::Same(i) => std::mem::take(&mut step_ins[i]),
                StepInput::Cache(c) => caches[c].clone(),
            })
            .collect();
        let t0 = Instant::now();
        let sgot = st.graph.eval(&sins, &st.outputs);
        let sids: Vec<u32> = sgot[0].iter().map(|&x| x as u32).collect();
        let esl = rel(&sgot[1], &rlg[t_pre]);
        let esc = rel(&sgot[2], &rcf[t_pre]);
        eprintln!(
            "derived step at row {t_pre}: drafts {:?} (reference {:?}), logits err {esl:.2e}, conf err {esc:.2e}, {:.1?}, peak RSS {:.1} GB",
            sids, rids[t_pre], t0.elapsed(), rss_gb()
        );
        assert_eq!(sids, rids[t_pre]);
        assert!(el < 1e-9 && ec < 1e-9 && esl < 1e-9 && esc < 1e-9);
    }

    /// Rows `[0, rows)` of quantized weight `name` as a device buffer in its
    /// native format: the codes and scales of those rows.
    fn native_rows(run: &RealRun, name: &str, rows: usize) -> (Format, Vec<u8>) {
        let (f, c, sc) = run.st.native_parts(name).unwrap();
        let cols: usize = run.st.shape(name).unwrap()[1..].iter().product();
        let (cb, sb) = match f {
            Format::Fp8B32 => (rows * cols, rows / 32 * cols / 32),
            _ => (rows * cols / 2, rows * cols / 32),
        };
        let mut v = c[..cb].to_vec();
        v.extend_from_slice(&sc[..sb]);
        (f, v)
    }

    /// One int8 GEMV `y[p, r] = Σ_k W[(idx[p],) r, k] · X[p, k]` over real
    /// checkpoint rows (`experts`: the tables of those experts, gathered by
    /// `idx`; else one dense weight), lowered by `schedule_block_weights` +
    /// `to_cuda_decode` and run on the CPU, against the dequantized weight
    /// times the Q8x32-rounded activation in f64: `(rel err, rel err vs the
    /// unrounded activation)`.
    fn gemv_on_cpu(run: &RealRun, names: &[String], rows: usize, idx: &[usize]) -> (f64, f64) {
        use tang_compile::tensor::{Arg, Dim, Form, Monoid, Precision, TensorGraph, View};
        let gathered = names.len() > 1 || !idx.is_empty();
        let npos = idx.len().max(1);
        let k: usize = run.st.shape(&names[0]).unwrap()[1..].iter().product();
        let mut bytes = Vec::new();
        let mut fmt = Format::F32;
        let mut deq: Vec<Vec<f64>> = Vec::new();
        let (mut codes, mut scales) = (Vec::new(), Vec::new());
        for n in names {
            let (f, b) = native_rows(run, n, rows);
            let cb = match f {
                Format::Fp8B32 => rows * k,
                _ => rows * k / 2,
            };
            codes.extend_from_slice(&b[..cb]);
            scales.extend_from_slice(&b[cb..]);
            fmt = f;
            deq.push(run.st.rows(n, 0, rows).unwrap());
        }
        bytes.extend(codes);
        bytes.extend(scales);
        let mut g = TensorGraph::new();
        let ne = names.len();
        let (x, w);
        let y = if gathered {
            let ix = g.input(&[Dim::Fixed(npos)]);
            x = g.input(&[Dim::Fixed(npos), Dim::Fixed(k)]);
            w = g.input(&[Dim::Fixed(ne), Dim::Fixed(rows), Dim::Fixed(k)]);
            g.comp(
                &[Dim::Fixed(npos), Dim::Fixed(rows)],
                &[Dim::Fixed(k)],
                vec![
                    Arg {
                        src: ix,
                        view: View::pick(3, &[Some(0)]),
                    },
                    Arg {
                        src: x,
                        view: View::pick(3, &[Some(0), Some(2)]),
                    },
                    Arg {
                        src: w,
                        view: View::pick(3, &[None, Some(1), Some(2)]).gather(0, 0, 1),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[1] * v[2]],
            )
            .unwrap()
        } else {
            x = g.input(&[Dim::Fixed(npos), Dim::Fixed(k)]);
            w = g.input(&[Dim::Fixed(rows), Dim::Fixed(k)]);
            g.comp(
                &[Dim::Fixed(npos), Dim::Fixed(rows)],
                &[Dim::Fixed(k)],
                vec![
                    Arg {
                        src: x,
                        view: View {
                            gather: None,
                            guard: false,
                            forms: vec![Form::unit(3, 0), Form::unit(3, 2)],
                        },
                    },
                    Arg {
                        src: w,
                        view: View::pick(3, &[Some(1), Some(2)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
        };
        g.set_precision(w, Precision::storage(fmt));
        let (g, outs, sched) = g.schedule_block_weights(&[y]);
        assert_eq!(sched.gemvs, 1, "the contraction is not on the int8 GEMV");
        let mut keep = outs.clone();
        keep.extend(&sched.copies);
        let (g, keep) = g.substitute_maps(&keep);
        let dp = g.to_cuda_decode(&keep[..1], &Default::default());
        // activations: a smooth signal with a few outliers, as hidden states
        let xs: Vec<f64> = (0..npos * k)
            .map(|i| {
                let t = i as f64;
                (t * 0.37).sin() * 0.8 + if i % 97 == 0 { 6.0 } else { 0.0 }
            })
            .collect();
        let mut xq = xs.clone();
        Format::Q8x32.round_blocks(&mut xq);
        let mut ins: Vec<Vec<f32>> = Vec::new();
        if gathered {
            ins.push(idx.iter().map(|&e| e as f32).collect());
        }
        ins.push(xs.iter().map(|&v| v as f32).collect());
        ins.push(words_of(&bytes));
        let got = crate::cuda_cpu::run(&dp, &ins, &keep[..1]);
        let got: Vec<f64> = got[0].iter().map(|&v| v as f64).collect();
        let reference = |xv: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; npos * rows];
            for p in 0..npos {
                let wm = &deq[if gathered { idx[p] } else { 0 }];
                for r in 0..rows {
                    out[p * rows + r] = (0..k).map(|c| wm[r * k + c] * xv[p * k + c]).sum();
                }
            }
            out
        };
        (rel(&got, &reference(&xq)), rel(&got, &reference(&xs)))
    }

    /// The DS GEMV kernels (Fp8B32 dense, Mxfp4 gathered experts), run on
    /// the CPU (`cuda_cpu`) over real rows of checkpoint layers 0 and 2,
    /// against dequantize-then-f64 references.
    #[test]
    #[ignore]
    fn ds_gemv_kernels_match_dequantized() {
        let Some(run) = open() else { return };
        let (l0, l2) = (run.layer_ids[0], run.layer_ids[1]);
        let dense = [
            (format!("layers.{l0}.attn.wq_a.weight"), 64),
            (format!("layers.{l2}.ffn.shared_experts.w2.weight"), 64),
            (format!("layers.{l2}.attn.wq_b.weight"), 96),
        ];
        for (n, rows) in &dense {
            let (e, e_raw) = gemv_on_cpu(&run, std::slice::from_ref(n), *rows, &[]);
            eprintln!("Fp8B32 {n} [{rows} rows]: rel err {e:.2e} (vs unrounded X {e_raw:.2e})");
            assert!(e < 1e-5, "{n}: {e:e}");
        }
        for (l, k) in [(l0, "w1"), (l0, "w2"), (l2, "w3")] {
            let names: Vec<String> = [48usize, 86, 98]
                .iter()
                .map(|e| format!("layers.{l}.ffn.experts.{e}.{k}.weight"))
                .collect();
            let (e, e_raw) = gemv_on_cpu(&run, &names, 64, &[2, 0, 1, 2]);
            eprintln!(
                "Mxfp4 layers.{l} experts 48/86/98 {k}, gathered [2, 0, 1, 2], 64 rows: rel err {e:.2e} (vs unrounded X {e_raw:.2e})"
            );
            assert!(e < 1e-5, "{k}: {e:e}");
        }
    }

    /// The real-size derived step (`DS_REAL_LAYERS`, default checkpoint
    /// layers 0 and 2) lowered with its weights in the checkpoint's formats
    /// (`lower_native_step`: Fp8B32 dense, Mxfp4 experts, Q8x32 activations)
    /// and run on the CPU (`cuda_cpu`), against the interpreter's step — the
    /// step `real_layers_match_reference` checks against the f64 reference —
    /// pure and with the scheduled Q8x32 rounding emulated. Reads the routed
    /// experts that test recorded.
    #[test]
    #[ignore]
    fn native_step_matches_interpreter() {
        let Some(run) = open() else { return };
        let t_pre: usize = std::env::var("DS_REAL_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let toks = run.rows_of(&prompt(t_pre + 1));
        let used = run
            .read_routed(toks.len())
            .expect("run real_layers_match_reference first");
        let rows = used.iter().map(|u| u.len()).max().unwrap();
        let mut dm = dims_from_config(
            &run.cfg,
            &run.layer_ids,
            run.vocab_ids.len(),
            rows,
            8.max(t_pre + 1),
        );
        clamp_index_topk(&mut dm);
        let t0 = Instant::now();
        let pg = prefill_graph(&dm);
        let mut w = run.dense_inputs(&pg.inputs, &dm).unwrap();
        for (l, u) in used.iter().enumerate() {
            run.expert_tables(l, u, rows, &dm, &mut w);
        }
        let mut base: Vec<Vec<f64>> = pg
            .inputs
            .iter()
            .map(|s| {
                if s.name == "tokens" {
                    toks[..t_pre].to_vec()
                } else {
                    w.remove(&s.name).unwrap()
                }
            })
            .collect();
        drop(w);
        let st = derived_step(&pg, &dm, 1).unwrap();
        let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
        let pre = pg.g.bind(&[(SEQ, t_pre)]).unwrap().eval(&base, &nodes);
        let caches: Vec<Vec<f64>> = nodes
            .iter()
            .zip(pre)
            .zip(&st.caches)
            .map(|((&n, v), c)| {
                let shape: Vec<usize> =
                    pg.g.shape(n)
                        .iter()
                        .map(|d| {
                            if *d == tang_compile::tensor::Dim::Sym(SEQ) {
                                t_pre
                            } else {
                                d.n()
                            }
                        })
                        .collect();
                seed_cache(c, &shape, &v, t_pre, dm.ctx)
            })
            .collect();
        let sins: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&si| match si {
                StepInput::Pos => vec![t_pre as f64],
                StepInput::New(_) => toks[t_pre..t_pre + 1].to_vec(),
                StepInput::Same(i) => std::mem::take(&mut base[i]),
                StepInput::Cache(c) => caches[c].clone(),
            })
            .collect();
        let mut ids = st.outputs.clone();
        ids.extend(st.caches.iter().map(|c| c.updated));
        let want = st.graph.eval(&sins, &ids);
        eprintln!("prefill + interpreter step: {:.1?}", t0.elapsed());

        let t0 = Instant::now();
        let ns = lower_native_step(&pg, &st, &|n| native_format_of(&run, n));
        let nk = ns.program.program.kernels.len();
        let native = ns.program.native.iter().filter(|n| **n).count();
        let mut fmts: std::collections::BTreeMap<String, usize> = Default::default();
        for f in ns.storage.iter().flatten() {
            *fmts.entry(format!("{f:?}")).or_default() += 1;
        }
        eprintln!(
            "native lowering: {nk} kernels ({native} native), {} int8 GEMVs, inputs stored {fmts:?}, demoted to f32: {:?}, {:.1?}",
            ns.sched.gemvs,
            ns.demoted,
            t0.elapsed()
        );
        let emu = ns.graph.eval_emulated(&sins, &ns.outs);
        let words = run.native_words(&pg, &st, &ns, &used, rows, &sins).unwrap();
        drop(sins);
        let t0 = Instant::now();
        let got = crate::cuda_cpu::run(&ns.program, &words, &ns.outs);
        eprintln!("CUDA kernels on the CPU: {:.1?}", t0.elapsed());
        let v = dm.vocab;
        let argmax = |r: &[f64]| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
        let (mut worst_emu, mut worst_pure) = (0.0f64, 0.0f64);
        for (j, ((g, e), p)) in got.iter().zip(&emu).zip(&want).enumerate() {
            let keep = |x: &[f64]| -> Vec<f64> {
                x.iter()
                    .zip(p)
                    .filter(|(_, w)| !w.is_nan())
                    .map(|(a, _)| *a)
                    .collect()
            };
            let g64: Vec<f64> = g.iter().map(|&x| x as f64).collect();
            let (gk, ek, pk) = (keep(&g64), keep(e), keep(p));
            let (a, b) = (rel(&gk, &ek), rel(&gk, &pk));
            if j == 0 {
                assert_eq!(argmax(&gk[..v]), argmax(&pk[..v]), "top-1");
                eprintln!(
                    "logits: rel err vs emulated {a:.2e}, vs pure f64 step {b:.2e}, emulated vs pure {:.2e}; argmax native {} emulated {} pure {}",
                    rel(&ek, &pk),
                    argmax(&gk[..v]),
                    argmax(&ek[..v]),
                    argmax(&pk[..v])
                );
            }
            worst_emu = worst_emu.max(a);
            worst_pure = worst_pure.max(b);
        }
        eprintln!(
            "native step on the CPU: worst rel err vs emulated interpreter {worst_emu:.2e}, vs pure f64 step {worst_pure:.2e}"
        );
        // Q8x32 codes flip where the device's f32 activation and the
        // interpreter's f64 one straddle a rounding boundary: noise of the
        // quantization's own size (emulated vs pure is ~9e-3), not a kernel
        // error (the GEMVs alone match to ~1e-7, ds_gemv_kernels_match_dequantized)
        assert!(worst_emu < 2e-2, "{worst_emu:e}");
    }
}
