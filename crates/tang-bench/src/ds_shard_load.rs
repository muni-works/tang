//! Per-rank weight loading for the sharded DeepSeek-V4.1 step: each rank
//! reads only the slice of the checkpoint its [`RankProgram`] stores
//! ([`RankProgram::boxes`]), in the checkpoint's own formats.
//!
//! - Routed experts (expert parallel, dim 0 of `{p}.ffn.experts.w{1,2,3}`):
//!   the rank's block of experts, each expert's `Mxfp4` codes then every
//!   expert's scales (the [`RealRun::native_expert_tables`] layout), read
//!   expert by expert from the per-expert checkpoint tensors.
//! - Dense quantized weights (tensor parallel or replicated): the box's codes
//!   and the scale blocks under it, which is again an `Fp8B32` / `Mxfp4`
//!   tensor of the box's shape. A box that cuts through a scale block (rows
//!   or columns not on a 32 boundary) can't be stored natively; it is
//!   *demoted*: dequantized rows, cut to the box (reported).
//! - Unquantized tensors (BF16 / F32): the box's rows, then cut.
//! - Engram tables: left in the memory map ([`Shard::Host`]), never copied;
//!   a runtime maps them into host memory (they are 203 GB).
//! - Inputs the checkpoint doesn't hold (RoPE frequencies, slot maps): from
//!   the caller, cut to the box.
//!
//! [`classify`] decides what a box will be without reading it (planning a
//! 40-layer run on a CPU), [`Loader::load`] reads it. The CPU proof is
//! [`simulate_loaded`]: every rank's program run on the interpreter with
//! its weights from [`Loader::load`] only, against the unsharded step.
use std::collections::{HashMap, HashSet};

use tang_compile::tensor::shard::{
    assemble, cut, input_shapes, rank_program, Analysis, HostComm, RankProgram,
};
use tang_compile::tensor::{Format, TensorGraph, TensorId};

use crate::ir_ds_real::{ckpt_name, e4m3, e8m0, fp4, St};

/// One rank's slice of one graph input.
#[derive(Clone, Debug)]
pub enum Shard {
    /// Device bytes in `format` (codes, then scales) of a tensor of `extents`.
    Native {
        format: Format,
        bytes: Vec<u8>,
        extents: Vec<usize>,
    },
    /// Values (unquantized, computed, or a demoted quantized box).
    Values(Vec<f64>),
    /// Left in the checkpoint's memory map (Engram tables).
    Host { ckpt: String },
}

/// What a box will load as, without reading it.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Native {
        format: Format,
        bytes: usize,
    },
    /// A quantized tensor whose box cuts a scale block.
    Demoted {
        ckpt: String,
        values: usize,
    },
    Values {
        values: usize,
    },
    Host {
        ckpt: String,
    },
}

/// How graph inputs bind to the checkpoint.
pub struct Loader<'a> {
    pub st: &'a St,
    /// Graph layer `i` → checkpoint layer `layer_ids[i]` (`None`: same index).
    pub layer_ids: Option<&'a [usize]>,
    /// Checkpoint token ids of the graph's vocab rows (`None`: all).
    pub vocab_ids: Option<&'a [usize]>,
    /// Graph prefix (`layers.0`, `mtp.1`) → expert of each table row (`None`:
    /// a zero padding row). A prefix absent here: row `r` is expert `r`.
    pub experts: HashMap<String, Vec<Option<usize>>>,
    /// Full values of graph inputs the checkpoint doesn't hold.
    pub computed: &'a (dyn Fn(&str) -> Option<Vec<f64>> + Sync),
}

fn full(bx: &[(usize, usize)], shape: &[usize]) -> bool {
    bx.iter().zip(shape).all(|(&(a, b), &n)| a == 0 && b == n)
}

/// `(prefix, w1|w2|w3)` of a routed-expert table input.
fn expert_table(name: &str) -> Option<(&str, &str)> {
    let (p, k) = name.split_once(".ffn.experts.")?;
    (!k.contains('.')).then_some((p, k))
}

fn vocab_indexed(name: &str) -> bool {
    name == "embed.weight" || name == "head.weight" || name.contains(".markov_head.")
}

/// Rows box and column box of `shape` flattened to `[shape[0], rest]`, when
/// only dim 0 / dim 1 (2-D) is cut.
fn flat_box(
    shape: &[usize],
    bx: &[(usize, usize)],
) -> Option<((usize, usize), (usize, usize), usize, usize)> {
    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();
    match shape.len() {
        2 => Some((bx[0], bx[1], rows, cols)),
        n if n > 2 && full(&bx[1..], &shape[1..]) => Some((bx[0], (0, cols), rows, cols)),
        _ => None,
    }
}

/// Whether `format` can store this box natively.
fn aligned(format: Format, (r0, r1): (usize, usize), (c0, c1): (usize, usize)) -> bool {
    let cols_ok = c0 % 32 == 0 && c1 % 32 == 0;
    match format {
        Format::Fp8B32 => cols_ok && r0 % 32 == 0 && r1 % 32 == 0,
        _ => cols_ok,
    }
}

impl Loader<'_> {
    fn ckpt(&self, name: &str) -> String {
        match self.layer_ids {
            Some(ids) => ckpt_name(name, ids),
            None => name.into(),
        }
    }

    fn expert_ckpt(&self, prefix: &str, e: usize, k: &str) -> String {
        let p = self.ckpt(&format!("{prefix}.x"));
        format!(
            "{}.ffn.experts.{e}.{k}.weight",
            p.strip_suffix(".x").unwrap()
        )
    }

    fn rows_of(&self, prefix: &str, n: usize) -> Vec<Option<usize>> {
        match self.experts.get(prefix) {
            Some(m) => {
                assert_eq!(
                    m.len(),
                    n,
                    "{prefix}: expert map of {} rows, table {n}",
                    m.len()
                );
                m.clone()
            }
            None => (0..n).map(Some).collect(),
        }
    }

    /// What `name` (graph input of `shape`) cut to `bx` loads as.
    pub fn classify(
        &self,
        name: &str,
        shape: &[usize],
        bx: &[(usize, usize)],
    ) -> Result<Kind, String> {
        let ext: usize = bx.iter().map(|(a, b)| b - a).product();
        if name.ends_with(".engram.embed.weight") {
            if !full(bx, shape) {
                return Err(format!("{name}: an Engram table split {bx:?}"));
            }
            return Ok(Kind::Host {
                ckpt: self.ckpt(name),
            });
        }
        if let Some((p, k)) = expert_table(name) {
            if !full(&bx[1..], &shape[1..]) {
                return Err(format!("{name}: split inside an expert {bx:?}"));
            }
            let rows = self.rows_of(p, shape[0]);
            let Some(e0) = rows[bx[0].0..bx[0].1].iter().flatten().next() else {
                return Ok(Kind::Values { values: ext });
            };
            let cn = self.expert_ckpt(p, *e0, k);
            return Ok(match self.st.native_parts(&cn) {
                Ok((format, c, s)) => Kind::Native {
                    format,
                    bytes: (bx[0].1 - bx[0].0) * (c.len() + s.len()),
                },
                Err(_) => Kind::Values { values: ext },
            });
        }
        let cn = self.ckpt(name);
        if vocab_indexed(name) && self.vocab_ids.is_some() || !self.st.has(&cn) {
            return Ok(Kind::Values { values: ext });
        }
        let have = self.st.shape(&cn).unwrap();
        if have != shape {
            return Err(format!("{cn}: checkpoint {have:?}, graph {shape:?}"));
        }
        let Some(format) = self.st.native_format(&cn) else {
            return Ok(Kind::Values { values: ext });
        };
        match flat_box(shape, bx) {
            Some((rb, cb, _, _)) if aligned(format, rb, cb) => {
                let n = (rb.1 - rb.0) * (cb.1 - cb.0);
                let bytes = match format {
                    Format::Fp8B32 => n + n / 1024,
                    _ => n / 2 + n / 32,
                };
                Ok(Kind::Native { format, bytes })
            }
            _ => Ok(Kind::Demoted {
                ckpt: cn,
                values: ext,
            }),
        }
    }

    /// Read `name`'s box `bx` (graph input of `shape`) from the checkpoint.
    pub fn load(
        &self,
        name: &str,
        shape: &[usize],
        bx: &[(usize, usize)],
    ) -> Result<Shard, String> {
        let ext: Vec<usize> = bx.iter().map(|(a, b)| b - a).collect();
        let kind = self.classify(name, shape, bx)?;
        if let Kind::Host { ckpt } = kind {
            return Ok(Shard::Host { ckpt });
        }
        if let Some((p, k)) = expert_table(name) {
            let rows = self.rows_of(p, shape[0]);
            let mine = &rows[bx[0].0..bx[0].1];
            let per: usize = shape[1..].iter().product();
            if let Kind::Native { format, .. } = kind {
                let e0 = mine.iter().flatten().next().unwrap();
                let (_, c0, s0) = self.st.native_parts(&self.expert_ckpt(p, *e0, k))?;
                let (cn, sn) = (c0.len(), s0.len());
                let n = mine.len();
                let mut v = vec![0u8; n * (cn + sn)];
                for (r, e) in mine.iter().enumerate() {
                    let Some(e) = e else { continue };
                    let name = self.expert_ckpt(p, *e, k);
                    let (f, c, s) = self.st.native_parts(&name)?;
                    if f != format || c.len() != cn || s.len() != sn {
                        return Err(format!("{name}: not like expert {e0}"));
                    }
                    v[r * cn..(r + 1) * cn].copy_from_slice(c);
                    v[n * cn + r * sn..n * cn + (r + 1) * sn].copy_from_slice(s);
                }
                return Ok(Shard::Native {
                    format,
                    bytes: v,
                    extents: ext,
                });
            }
            let mut v = vec![0.0; mine.len() * per];
            for (r, e) in mine.iter().enumerate() {
                if let Some(e) = e {
                    v[r * per..(r + 1) * per]
                        .copy_from_slice(&self.st.f64(&self.expert_ckpt(p, *e, k))?);
                }
            }
            return Ok(Shard::Values(v));
        }
        if let Some(p) = name.strip_suffix(".ffn.expert_slot") {
            // the slot map of the table of the same prefix: expert → row
            let rows = self
                .experts
                .get(p)
                .ok_or_else(|| format!("{name}: no expert map for {p}"))?;
            let mut slot = vec![0.0; shape[0]];
            for (r, e) in rows.iter().enumerate() {
                if let Some(e) = e {
                    slot[*e] = r as f64;
                }
            }
            return Ok(Shard::Values(cut(&slot, shape, bx)));
        }
        let cn = self.ckpt(name);
        if let (true, Some(ids)) = (vocab_indexed(name), self.vocab_ids) {
            let mut v = Vec::new();
            for &id in &ids[bx[0].0..bx[0].1] {
                v.extend(self.st.rows(&cn, id, id + 1)?);
            }
            let mut sh = shape.to_vec();
            sh[0] = ext[0];
            let mut b = bx.to_vec();
            b[0] = (0, ext[0]);
            return Ok(Shard::Values(cut(&v, &sh, &b)));
        }
        if !self.st.has(&cn) {
            let v = (self.computed)(name)
                .ok_or_else(|| format!("{name}: not in the checkpoint, not computed"))?;
            return Ok(Shard::Values(cut(&v, shape, bx)));
        }
        if let Kind::Native { format, .. } = kind {
            let ((r0, r1), (c0, c1), _, cols) = flat_box(shape, bx).unwrap();
            let (_, codes, scales) = self.st.native_parts(&cn)?;
            let sc = cols / 32;
            let mut v = Vec::new();
            match format {
                Format::Fp8B32 => {
                    for r in r0..r1 {
                        v.extend_from_slice(&codes[r * cols + c0..r * cols + c1]);
                    }
                    for br in r0 / 32..r1 / 32 {
                        v.extend_from_slice(&scales[br * sc + c0 / 32..br * sc + c1 / 32]);
                    }
                }
                _ => {
                    for r in r0..r1 {
                        v.extend_from_slice(&codes[(r * cols + c0) / 2..(r * cols + c1) / 2]);
                    }
                    for r in r0..r1 {
                        v.extend_from_slice(&scales[r * sc + c0 / 32..r * sc + c1 / 32]);
                    }
                }
            }
            return Ok(Shard::Native {
                format,
                bytes: v,
                extents: ext,
            });
        }
        // unquantized or demoted: the box's rows, dequantized, then cut
        let v = self.st.rows(&cn, bx[0].0, bx[0].1)?;
        let mut sh = shape.to_vec();
        sh[0] = ext[0];
        let mut b = bx.to_vec();
        b[0] = (0, ext[0]);
        Ok(Shard::Values(cut(&v, &sh, &b)))
    }

    /// A shard's values (a host-mapped table read from the map).
    pub fn values(&self, s: &Shard) -> Result<Vec<f64>, String> {
        match s {
            Shard::Native {
                format,
                bytes,
                extents,
            } => Ok(dequant(*format, bytes, extents)),
            Shard::Values(v) => Ok(v.clone()),
            Shard::Host { ckpt } => self.st.f64(ckpt),
        }
    }
}

/// The values of a native buffer (codes, then scales) of `extents`
/// (flattened to `[extents[0], rest]`).
pub fn dequant(format: Format, bytes: &[u8], extents: &[usize]) -> Vec<f64> {
    let rows = extents[0];
    let cols: usize = extents[1..].iter().product();
    let sc = cols / 32;
    let mut out = Vec::with_capacity(rows * cols);
    match format {
        Format::Fp8B32 => {
            let (codes, scales) = bytes.split_at(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    out.push(e4m3(codes[r * cols + c]) * e8m0(scales[(r / 32) * sc + c / 32]));
                }
            }
        }
        _ => {
            let (codes, scales) = bytes.split_at(rows * cols / 2);
            for r in 0..rows {
                for c in 0..cols {
                    let b = codes[(r * cols + c) / 2];
                    let q = if c % 2 == 0 { b & 0xF } else { b >> 4 };
                    out.push(fp4(q) * e8m0(scales[r * sc + c / 32]));
                }
            }
        }
    }
    out
}

/// One rank's load: per step input its values (weights from the loader,
/// everything else cut from `full`), and what was read.
pub struct RankLoad {
    pub ins: Vec<Vec<f64>>,
    pub native_bytes: usize,
    pub value_elems: usize,
    pub native: usize,
    pub demoted: Vec<String>,
    pub host: Vec<String>,
    /// Per step input: the native buffer (codes, then scales) when the box
    /// loaded natively.
    pub shards: Vec<Option<(Format, Vec<u8>)>>,
}

/// Rank `p`'s inputs: step input `k` named `names[k]` (a weight) loaded
/// through `ld`, the rest cut from `full[k]` (a weight's `full[k]` is never
/// read; pass it empty). With `check`, every loaded weight's values are
/// compared bit for bit with the same box of `check[k]`.
pub fn load_rank(
    ld: &Loader,
    p: &RankProgram,
    shapes: &[Vec<usize>],
    names: &[Option<String>],
    full: &[Vec<f64>],
    check: Option<&[Vec<f64>]>,
) -> Result<RankLoad, String> {
    let mut out = RankLoad {
        ins: Vec::new(),
        native_bytes: 0,
        value_elems: 0,
        native: 0,
        demoted: Vec::new(),
        host: Vec::new(),
        shards: Vec::new(),
    };
    for k in 0..shapes.len() {
        let bx = &p.boxes[k];
        let Some(name) = &names[k] else {
            out.ins.push(cut(&full[k], &shapes[k], bx));
            out.shards.push(None);
            continue;
        };
        let s = ld.load(name, &shapes[k], bx)?;
        match (&s, ld.classify(name, &shapes[k], bx)?) {
            (Shard::Native { bytes, .. }, _) => {
                out.native += 1;
                out.native_bytes += bytes.len();
            }
            (Shard::Host { ckpt }, _) => out.host.push(ckpt.clone()),
            (Shard::Values(v), kind) => {
                out.value_elems += v.len();
                if let Kind::Demoted { ckpt, .. } = kind {
                    out.demoted.push(ckpt);
                }
            }
        }
        let v = ld.values(&s)?;
        if let Some(c) = check {
            let want = cut(&c[k], &shapes[k], bx);
            if v.len() != want.len() || v.iter().zip(&want).any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!(
                    "{name} box {bx:?}: loaded values differ from the full tensor's box"
                ));
            }
        }
        out.ins.push(v);
        out.shards.push(match s {
            Shard::Native { format, bytes, .. } => Some((format, bytes)),
            _ => None,
        });
    }
    Ok(out)
}

/// The sharded step on the interpreter (one thread a rank, collectives on
/// the host) with each rank's weights from `ld` only; returns the assembled
/// outputs and each rank's load.
#[allow(clippy::too_many_arguments)]
pub fn simulate_loaded(
    g: &TensorGraph,
    outputs: &[TensorId],
    an: &Analysis,
    weights: &HashSet<TensorId>,
    names: &[Option<String>],
    full: &[Vec<f64>],
    ld: &Loader,
    check: Option<&[Vec<f64>]>,
) -> Result<(Vec<Vec<f64>>, Vec<RankLoad>), String> {
    let comm = HostComm::new(an.world);
    let shapes = input_shapes(g);
    let progs: Vec<RankProgram> = (0..an.world)
        .map(|r| rank_program(g, outputs, an, weights, r, comm.clone()))
        .collect();
    let loads: Vec<RankLoad> = progs
        .iter()
        .map(|p| load_rank(ld, p, &shapes, names, full, check))
        .collect::<Result<_, _>>()?;
    let res: Vec<Vec<Vec<f64>>> = std::thread::scope(|s| {
        let hs: Vec<_> = progs
            .iter()
            .zip(&loads)
            .map(|(p, l)| s.spawn(move || p.graph.eval(&l.ins, &p.outputs)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    Ok((assemble(g, outputs, an, &res), loads))
}

/// A safetensors writer for test checkpoints in the real checkpoint's
/// formats (`F8_E4M3` + `F8_E8M0` 32 × 32 scales, fp4 `I8` + row × 32
/// scales, `F32`).
pub mod synth {
    use super::*;

    pub enum Entry {
        F32(Vec<usize>, Vec<f64>),
        /// Quantized to `Fp8B32` (rows, cols multiples of 32).
        Fp8(Vec<usize>, Vec<f64>),
        /// Quantized to `Mxfp4` (cols a multiple of 32).
        Fp4(Vec<usize>, Vec<f64>),
    }

    fn e8m0_for(amax: f64, top: f64) -> u8 {
        if amax == 0.0 {
            return 127;
        }
        ((amax / top).log2().ceil() as i32 + 127).clamp(1, 254) as u8
    }

    fn nearest(table: &[(u8, f64)], x: f64) -> u8 {
        table
            .iter()
            .min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs()))
            .unwrap()
            .0
    }

    /// `(name, dtype, shape, bytes)` tensors of `e` under `name`.
    fn tensors(name: &str, e: &Entry) -> Vec<(String, &'static str, Vec<usize>, Vec<u8>)> {
        match e {
            Entry::F32(s, v) => {
                vec![(
                    name.into(),
                    "F32",
                    s.clone(),
                    v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect(),
                )]
            }
            Entry::Fp8(s, v) => {
                let (rows, cols) = (s[0], s[1..].iter().product::<usize>());
                assert!(rows % 32 == 0 && cols % 32 == 0, "{name}: {s:?}");
                let t: Vec<(u8, f64)> = (0..=255u8)
                    .map(|c| (c, e4m3(c)))
                    .filter(|x| x.1.is_finite())
                    .collect();
                let (br, bc) = (rows / 32, cols / 32);
                let mut sc = vec![0u8; br * bc];
                for i in 0..br {
                    for j in 0..bc {
                        let mut m = 0.0f64;
                        for r in 0..32 {
                            for c in 0..32 {
                                m = m.max(v[(i * 32 + r) * cols + j * 32 + c].abs());
                            }
                        }
                        sc[i * bc + j] = e8m0_for(m, 448.0);
                    }
                }
                let codes = (0..rows * cols)
                    .map(|k| nearest(&t, v[k] / e8m0(sc[(k / cols / 32) * bc + (k % cols) / 32])))
                    .collect();
                vec![
                    (name.into(), "F8_E4M3", s.clone(), codes),
                    (
                        format!("{}scale", name.strip_suffix("weight").unwrap()),
                        "F8_E8M0",
                        vec![br, bc],
                        sc,
                    ),
                ]
            }
            Entry::Fp4(s, v) => {
                let (rows, cols) = (s[0], s[1..].iter().product::<usize>());
                assert!(rows % 32 == 0 && cols % 32 == 0, "{name}: {s:?}");
                let t: Vec<(u8, f64)> = (0..16u8).map(|c| (c, fp4(c))).collect();
                let sc: Vec<u8> = (0..rows * cols / 32)
                    .map(|b| {
                        e8m0_for(
                            v[b * 32..b * 32 + 32]
                                .iter()
                                .fold(0.0f64, |m, x| m.max(x.abs())),
                            6.0,
                        )
                    })
                    .collect();
                let q: Vec<u8> = (0..rows * cols)
                    .map(|k| nearest(&t, v[k] / e8m0(sc[k / 32])))
                    .collect();
                let codes = q.chunks_exact(2).map(|p| p[0] | (p[1] << 4)).collect();
                vec![
                    (name.into(), "I8", vec![rows, cols / 2], codes),
                    (
                        format!("{}scale", name.strip_suffix("weight").unwrap()),
                        "F8_E8M0",
                        vec![rows, cols / 32],
                        sc,
                    ),
                ]
            }
        }
    }

    /// Write `entries` as one safetensors file at `path`.
    pub fn write(path: &std::path::Path, entries: &[(String, Entry)]) -> std::io::Result<()> {
        let mut hdr = serde_json::Map::new();
        let mut data = Vec::new();
        for (n, e) in entries {
            for (name, dt, shape, b) in tensors(n, e) {
                hdr.insert(
                    name,
                    serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [data.len(), data.len() + b.len()]}),
                );
                data.extend(b);
            }
        }
        let mut h = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
        while h.len() % 8 != 0 {
            h.push(b' ');
        }
        let mut f = Vec::with_capacity(8 + h.len() + data.len());
        f.extend((h.len() as u64).to_le_bytes());
        f.extend(h);
        f.extend(data);
        std::fs::write(path, f)
    }
}

#[cfg(test)]
mod tests {
    use super::synth::{write, Entry};
    use super::*;
    use crate::derive_all::Model;
    use crate::ir_ds_prefill::DsDims;
    use crate::ir_shard::{case_with, ds_model, ds_wide, native_bytes, rel, summary, Case};
    use tang_compile::tensor::shard::{analyze, plan_auto, CostModel};
    use tang_compile::tensor::step::StepInput;

    /// `ds_wide` at widths the checkpoint formats can hold (32-multiples).
    fn tiny_dims() -> DsDims {
        DsDims {
            dim: 64,
            moe_inter: 32,
            q_lora_rank: 32,
            ..ds_wide()
        }
    }

    fn step_names(m: &Model, st: &tang_compile::tensor::step::StepGraph) -> Vec<Option<String>> {
        st.inputs
            .iter()
            .map(|x| match x {
                StepInput::Same(i) => Some(m.inputs[*i].0.clone()),
                _ => None,
            })
            .collect()
    }

    /// A synthetic checkpoint of `m`'s weights in the real formats: routed
    /// experts per expert as fp4, 32-aligned dense matrices as fp8, the rest
    /// F32. Returns its directory.
    fn synth_ckpt(m: &Model) -> std::path::PathBuf {
        let w = crate::derive_all::weights(m);
        let mut entries = Vec::new();
        for (name, shape) in &m.inputs {
            if name == "tokens" {
                continue;
            }
            let s: Vec<usize> = shape.iter().map(|d| d.n()).collect();
            let v = &w[name];
            if let Some((p, k)) = expert_table(name) {
                let per: usize = s[1..].iter().product();
                for e in 0..s[0] {
                    let x = v[e * per..(e + 1) * per].to_vec();
                    let n = format!("{p}.ffn.experts.{e}.{k}.weight");
                    let es = s[1..].to_vec();
                    entries.push((
                        n,
                        if es[0] % 32 == 0 && es[1] % 32 == 0 {
                            Entry::Fp4(es, x)
                        } else {
                            Entry::F32(es, x)
                        },
                    ));
                }
                continue;
            }
            let quant = name.ends_with("weight")
                && !vocab_indexed(name)
                && !name.contains("norm")
                && !name.contains("engram")
                && s.len() == 2
                && s[0] % 32 == 0
                && s[1] % 32 == 0;
            entries.push((
                name.clone(),
                if quant {
                    Entry::Fp8(s, v.clone())
                } else {
                    Entry::F32(s, v.clone())
                },
            ));
        }
        let dir = std::env::temp_dir().join(format!("tang-shard-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir.join("model.safetensors"), &entries).unwrap();
        dir
    }

    /// The tiny DS step's weights written in the checkpoint's formats, each
    /// rank loading only its box (EP experts, TP splits, replicated rest):
    /// every loaded box equals that box of the full tensor bit for bit, and
    /// the sharded step on those weights matches the unsharded one.
    #[test]
    fn tiny_loaded_shards_match() {
        let m = ds_model(tiny_dims());
        let dir = synth_ckpt(&m);
        let st = St::open(&dir).unwrap();
        // the weights as the checkpoint holds them (quantized, dequantized)
        let mut w = HashMap::new();
        for (name, shape) in &m.inputs {
            if name == "tokens" {
                continue;
            }
            let v = match expert_table(name) {
                Some((p, k)) => (0..shape[0].n())
                    .flat_map(|e| st.f64(&format!("{p}.ffn.experts.{e}.{k}.weight")).unwrap())
                    .collect(),
                None => st.f64(name).unwrap(),
            };
            w.insert(name.clone(), v);
        }
        let quantized = m
            .inputs
            .iter()
            .filter(|(n, _)| st.native_format(n).is_some())
            .count();
        let Case {
            st: step,
            ins,
            outs,
            want,
            weights,
            caches,
        } = case_with(&m, 1, 5, &w);
        assert!(want[0].iter().all(|x| x.is_finite()));
        let names = step_names(&m, &step);
        let bare: Vec<Vec<f64>> = ins
            .iter()
            .zip(&names)
            .map(|(v, n)| if n.is_some() { Vec::new() } else { v.clone() })
            .collect();
        let none = |_: &str| None;
        let ld = Loader {
            st: &st,
            layer_ids: None,
            vocab_ids: None,
            experts: HashMap::new(),
            computed: &none,
        };
        for n in [2usize, 4] {
            for (label, cm) in [("eager", CostModel::eager()), ("8xH200", CostModel::h200())] {
                let plan = plan_auto(&step.graph, &outs, &weights, &caches, n, &|_| 1.0, &cm);
                let an = analyze(&step.graph, &outs, &plan);
                let (got, loads) = simulate_loaded(
                    &step.graph,
                    &outs,
                    &an,
                    &weights,
                    &names,
                    &bare,
                    &ld,
                    Some(&ins),
                )
                .unwrap();
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| rel(a, b))
                    .fold(0.0, f64::max);
                let s = summary(&plan, &an);
                let l = &loads[0];
                eprintln!(
                    "tiny N={n} {label:6}: {} EP / {} TP inputs; rank 0 loads {} native ({} B), {} values, demoted {:?}; rel err {err:.2e}",
                    s.experts,
                    s.splits,
                    l.native,
                    l.native_bytes,
                    l.value_elems,
                    l.demoted
                );
                assert!(err < 1e-10, "N={n} {label}: {err:e}");
                assert!(
                    l.native > 0,
                    "nothing loaded natively ({quantized} quantized tensors)"
                );
            }
        }
        std::fs::remove_dir_all(dir).ok();
    }

    /// The native box cut agrees with dequantizing then cutting, on every
    /// aligned box of a 64 × 96 fp8 and fp4 matrix (rows and columns split
    /// 2 and 3 ways), and unaligned boxes are demoted.
    #[test]
    fn native_boxes_cut_like_values() {
        let (r, c) = (64usize, 96usize);
        let v: Vec<f64> = (0..r * c)
            .map(|k| ((k * 7919 % 1000) as f64 / 500.0 - 1.0) * 3.0)
            .collect();
        let dir = std::env::temp_dir().join(format!("tang-shard-box-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(
            &dir.join("m.safetensors"),
            &[
                ("a.weight".into(), Entry::Fp8(vec![r, c], v.clone())),
                ("b.weight".into(), Entry::Fp4(vec![r, c], v)),
            ],
        )
        .unwrap();
        let st = St::open(&dir).unwrap();
        let none = |_: &str| None;
        let ld = Loader {
            st: &st,
            layer_ids: None,
            vocab_ids: None,
            experts: HashMap::new(),
            computed: &none,
        };
        let mut native = 0;
        for name in ["a.weight", "b.weight"] {
            let fullv = st.f64(name).unwrap();
            for bx in [
                vec![(0, 32), (0, 96)],
                vec![(32, 64), (0, 96)],
                vec![(0, 64), (32, 64)],
                vec![(0, 64), (64, 96)],
                vec![(16, 48), (0, 96)],
                vec![(0, 64), (0, 48)],
            ] {
                let s = ld.load(name, &[r, c], &bx).unwrap();
                let got = ld.values(&s).unwrap();
                assert_eq!(got, cut(&fullv, &[r, c], &bx), "{name} {bx:?}");
                let k = ld.classify(name, &[r, c], &bx).unwrap();
                if let Shard::Native { bytes, .. } = &s {
                    native += 1;
                    assert_eq!(
                        Kind::Native {
                            format: st.native_format(name).unwrap(),
                            bytes: bytes.len()
                        },
                        k
                    );
                }
                let unaligned = bx[1].1 % 32 != 0 || (name == "a.weight" && bx[0].0 % 32 != 0);
                assert_eq!(
                    matches!(k, Kind::Demoted { .. }),
                    unaligned,
                    "{name} {bx:?}: {k:?}"
                );
            }
        }
        assert_eq!(native, 9);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Real layers (`DS41_DIR`, `DS_REAL_LAYERS`, routed tables as for
    /// `ds_multi_gpu`): the sharded step at world 2 and 4 with every rank's
    /// weights read by [`Loader`] from the checkpoint (compact expert
    /// tables, the run's vocab rows) matches the unsharded step built by
    /// the existing dense path.
    #[test]
    #[ignore]
    fn real_loaded_shards_match() {
        use crate::ir_ds_prefill::{derived_step, prefill_graph, SEQ};
        use crate::ir_ds_real::*;
        use tang_compile::tensor::step::seed_cache;
        use tang_compile::tensor::Dim;
        let run = RealRun::from_env().expect("DS41_DIR");
        let t_pre = 3;
        let toks = run.rows_of(&prompt(t_pre + 1));
        let used = run
            .read_routed(toks.len())
            .expect("routed tables (run real_layers_match_reference)");
        let rows = used.iter().map(|u| u.len()).max().unwrap().div_ceil(4) * 4;
        let dm = dims_from_config(&run.cfg, &run.layer_ids, run.vocab_ids.len(), rows, 8);
        let pg = prefill_graph(&dm);
        let t0 = std::time::Instant::now();
        let mut w = run.dense_inputs(&pg.inputs, &dm).unwrap();
        for (l, u) in used.iter().enumerate() {
            run.expert_tables(l, u, rows, &dm, &mut w);
        }
        let names0: Vec<String> = pg.inputs.iter().map(|s| s.name.clone()).collect();
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
                        .map(|d| if *d == Dim::Sym(SEQ) { t_pre } else { d.n() })
                        .collect();
                seed_cache(c, &shape, &v, t_pre, dm.ctx)
            })
            .collect();
        let names: Vec<Option<String>> = st
            .inputs
            .iter()
            .map(|x| match x {
                StepInput::Same(i) => Some(names0[*i].clone()),
                _ => None,
            })
            .collect();
        let ins: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&x| match x {
                StepInput::Pos => vec![t_pre as f64],
                StepInput::New(_) => toks[t_pre..t_pre + 1].to_vec(),
                StepInput::Same(i) => std::mem::take(&mut base[i]),
                StepInput::Cache(c) => caches[c].clone(),
            })
            .collect();
        let mut outs = st.outputs.clone();
        outs.extend(st.caches.iter().map(|c| c.updated));
        let want = st.graph.eval(&ins, &outs);
        eprintln!(
            "real layers {:?}: reference step in {:.1?}",
            run.layer_ids,
            t0.elapsed()
        );
        let weights: HashSet<TensorId> = names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_some())
            .map(|(k, _)| crate::ir_model::input_node(&st.graph, k))
            .collect();
        let by_node: HashMap<TensorId, String> = names
            .iter()
            .enumerate()
            .filter_map(|(k, n)| {
                n.clone()
                    .map(|n| (crate::ir_model::input_node(&st.graph, k), n))
            })
            .collect();
        let bare: Vec<Vec<f64>> = ins
            .iter()
            .zip(&names)
            .map(|(v, n)| if n.is_some() { Vec::new() } else { v.clone() })
            .collect();
        let computed = |n: &str| {
            let i = pg.inputs.iter().find(|i| i.name == n)?;
            run.dense_inputs(std::slice::from_ref(i), &dm)
                .ok()?
                .remove(n)
        };
        let experts: HashMap<String, Vec<Option<usize>>> = used
            .iter()
            .enumerate()
            .map(|(l, u)| {
                let mut m: Vec<Option<usize>> = u.iter().copied().map(Some).collect();
                m.resize(rows, None);
                (format!("layers.{l}"), m)
            })
            .collect();
        let ld = Loader {
            st: &run.st,
            layer_ids: Some(&run.layer_ids),
            vocab_ids: Some(&run.vocab_ids),
            experts,
            computed: &computed,
        };
        let caches_p = crate::ir_shard::cache_pairs(&st);
        for n in [2usize, 4] {
            let t0 = std::time::Instant::now();
            let elem = |t: TensorId| by_node.get(&t).map_or(1.0, |s| native_bytes(s));
            let plan = plan_auto(
                &st.graph,
                &outs,
                &weights,
                &caches_p,
                n,
                &elem,
                &CostModel::h200(),
            );
            let an = analyze(&st.graph, &outs, &plan);
            let s = summary(&plan, &an);
            let (got, loads) = simulate_loaded(
                &st.graph,
                &outs,
                &an,
                &weights,
                &names,
                &bare,
                &ld,
                Some(&ins),
            )
            .unwrap();
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| rel(a, b))
                .fold(0.0, f64::max);
            for (r, l) in loads.iter().enumerate() {
                eprintln!(
                    "  N={n} rank {r}: {} native tensors {:.1} MB, {:.1} M values, demoted {:?}, host {:?}",
                    l.native,
                    l.native_bytes as f64 / 1e6,
                    l.value_elems as f64 / 1e6,
                    l.demoted,
                    l.host
                );
            }
            eprintln!(
                "real N={n}: {} EP / {} TP inputs, {} all-reduce + {} all-gather; loaded sharded vs unsharded rel err {err:.2e} ({:.1?})",
                s.experts,
                s.splits,
                s.all_reduce,
                s.all_gather,
                t0.elapsed()
            );
            assert!(err < 1e-9, "N={n}: {err:e}");
        }
    }

    /// Readiness of a full-model run: real layers (`DS_REAL_LAYERS`, all 384
    /// experts, the whole vocab), planned for `DS_SHARD_WORLDS` (default
    /// `4`) on the 8×H200 cost model; every rank's every weight box
    /// classified (native / values / host) and none demoted. With
    /// `DS_SHARD_READ=1` every box is also read, its byte count checked
    /// against the classification and each rank's first expert of every
    /// table checked against the checkpoint tensor's values.
    #[test]
    #[ignore]
    fn real_full_boxes_load_native() {
        use crate::ir_ds_real::*;
        use tang_compile::tensor::shard::{input_shapes, rank_program};
        let run = RealRun::from_env().expect("DS41_DIR");
        let vocab = run.cfg["vocab_size"].as_u64().unwrap() as usize;
        let dm = dims_from_config(&run.cfg, &run.layer_ids, vocab, 0, 4096);
        let t0 = std::time::Instant::now();
        let m = Model {
            cap: 4096,
            ..ds_model(dm.clone())
        };
        let st = tang_compile::tensor::step::derive_step(
            &m.g,
            &[m.logits],
            &crate::derive_all::spec(&m, 1),
        )
        .expect("derives");
        let mut outs = st.outputs.clone();
        outs.extend(st.caches.iter().map(|c| c.updated));
        let names = step_names(&m, &st);
        let wi = crate::ir_shard::weight_inputs(&m, &st);
        let ws: HashSet<TensorId> = wi.iter().map(|(t, _)| *t).collect();
        let by_node: HashMap<TensorId, String> = wi.into_iter().collect();
        eprintln!(
            "full layers {:?}: step derived in {:.1?}",
            run.layer_ids,
            t0.elapsed()
        );
        let shapes = input_shapes(&st.graph);
        let read = std::env::var("DS_SHARD_READ").as_deref() == Ok("1");
        let computed = |n: &str| {
            let pg = crate::ir_ds_prefill::prefill_graph(&dm);
            let i = pg.inputs.iter().find(|i| i.name == n)?.clone();
            run.dense_inputs(std::slice::from_ref(&i), &dm)
                .ok()?
                .remove(n)
        };
        let ld = Loader {
            st: &run.st,
            layer_ids: Some(&run.layer_ids),
            vocab_ids: None,
            experts: HashMap::new(),
            computed: &computed,
        };
        let worlds: Vec<usize> = std::env::var("DS_SHARD_WORLDS")
            .unwrap_or_else(|_| "4".into())
            .split(',')
            .map(|w| w.parse().unwrap())
            .collect();
        for n in worlds {
            let t0 = std::time::Instant::now();
            let elem = |t: TensorId| by_node.get(&t).map_or(1.0, |s| native_bytes(s));
            let plan = crate::ir_shard::plan_auto_cached(
                &st.graph,
                &outs,
                &ws,
                &crate::ir_shard::cache_pairs(&st),
                n,
                &elem,
                &CostModel::h200(),
            );
            let an = analyze(&st.graph, &outs, &plan);
            let s = summary(&plan, &an);
            eprintln!(
                "N={n}: planned in {:.1?}: {} EP / {} TP inputs, {} all-reduce + {} all-gather a step",
                t0.elapsed(),
                s.experts,
                s.splits,
                s.all_reduce,
                s.all_gather
            );
            let comm = HostComm::new(n);
            let mut demoted = Vec::new();
            for r in 0..n {
                let p = rank_program(&st.graph, &outs, &an, &ws, r, comm.clone());
                let (mut nat, mut vals, mut host, mut experts) =
                    (0usize, 0usize, Vec::new(), 0usize);
                let t1 = std::time::Instant::now();
                for (k, name) in names.iter().enumerate() {
                    let Some(name) = name else { continue };
                    let bx = &p.boxes[k];
                    let kind = ld.classify(name, &shapes[k], bx).unwrap();
                    match &kind {
                        Kind::Native { bytes, .. } => nat += bytes,
                        Kind::Values { values } => vals += values,
                        Kind::Demoted { ckpt, values } => {
                            vals += values;
                            demoted.push(format!("rank {r} {ckpt} {bx:?}"));
                        }
                        Kind::Host { ckpt } => host.push(ckpt.clone()),
                    }
                    if let Some((pfx, w)) = expert_table(name) {
                        experts = bx[0].1 - bx[0].0;
                        if read {
                            let sh = ld.load(name, &shapes[k], bx).unwrap();
                            let Shard::Native {
                                format,
                                bytes,
                                extents,
                            } = &sh
                            else {
                                panic!("{name}: not native")
                            };
                            assert_eq!(
                                kind,
                                Kind::Native {
                                    format: *format,
                                    bytes: bytes.len()
                                }
                            );
                            // the box's first expert against the checkpoint
                            let per: usize = extents[1..].iter().product();
                            let e = bx[0].0;
                            let n_e = extents[0];
                            let (cn, sn) = (per / 2, per / 32);
                            assert_eq!(bytes.len(), n_e * (cn + sn));
                            let mut one = bytes[..cn].to_vec();
                            one.extend_from_slice(&bytes[n_e * cn..n_e * cn + sn]);
                            let got = dequant(*format, &one, &extents[1..]);
                            let want = run.st.f64(&ld.expert_ckpt(pfx, e, w)).unwrap();
                            assert_eq!(got.len(), per);
                            assert!(got == want, "{name}: rank {r} expert {e} differs");
                        }
                    } else if read {
                        let sh = ld.load(name, &shapes[k], bx).unwrap();
                        if let (Shard::Native { bytes, .. }, Kind::Native { bytes: b, .. }) =
                            (&sh, &kind)
                        {
                            assert_eq!(bytes.len(), *b, "{name}");
                        }
                    }
                }
                eprintln!(
                    "  rank {r}: {experts} experts a table; native {:.2} GB, values {:.2} G elems, host {:?}{}",
                    nat as f64 / 1e9,
                    vals as f64 / 1e9,
                    host,
                    if read { format!(", read in {:.1?}", t1.elapsed()) } else { String::new() }
                );
            }
            assert!(demoted.is_empty(), "N={n}: demoted boxes: {demoted:#?}");
        }
    }
}
