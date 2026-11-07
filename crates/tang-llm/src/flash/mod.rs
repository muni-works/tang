//! Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`): the model's geometry, read from GGUF
//! metadata, a tensor inventory, and a slow pure-Rust f32 reference forward
//! ([`reference`]) that the fast kernels are checked against.
//!
//! The block math, with its traps, is written up in `docs/strata.md`; what the GGUF actually holds
//! is in `docs/flash-next-tensors.md`.

#[cfg(feature = "cuda")]
pub mod cli;
#[cfg(feature = "cuda")]
pub mod coder;
#[cfg(feature = "cuda")]
pub mod coder_kernels;
pub mod coder_mtp;
pub mod deepseek;
pub mod derived_attn;
#[cfg(feature = "cuda")]
pub mod engine;
pub mod fa2;
pub mod glm5;
#[cfg(feature = "cuda")]
pub mod kernels;
pub mod mtp;
#[cfg(feature = "cuda")]
pub mod mtp_gpu;
#[cfg(feature = "mtp-train")]
pub mod mtp_train;
pub mod ngram;
pub mod pack;
pub mod reference;
pub mod requant;
#[cfg(feature = "cuda")]
pub mod sampler;
#[cfg(feature = "cuda")]
pub mod serve;
#[cfg(feature = "cuda")]
pub mod split;
pub mod tcx;
pub mod tokenize;
pub mod tree;

use crate::gguf::{Gguf, TensorInfo};
use anyhow::Result;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A tensor name with every layer index replaced by `N`, so the 48 copies of a layer tensor
/// group together: `blk.N.ffn_gate_exps.weight`.
pub fn name_pattern(name: &str) -> String {
    let mut out = String::new();
    for (i, part) in name.split('.').enumerate() {
        if i > 0 {
            out.push('.');
        }
        if !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()) {
            out.push('N');
        } else {
            out.push_str(part);
        }
    }
    out
}

/// Which accounting bucket a tensor falls in.
pub fn category(t: &TensorInfo) -> &'static str {
    let n = t.name.as_str();
    if n.contains("_exps") {
        "routed experts"
    } else if n.starts_with("per_layer_token_embd") {
        "n-gram table"
    } else if n == "output.weight" || n.starts_with("output_hc") || n.starts_with("hc_head") {
        "head"
    } else if n == "token_embd.weight" {
        "embedding"
    } else if n.contains(".hc_") {
        "hyper-connections"
    } else if n.contains("nextn") || n.starts_with("mtp") {
        "mtp"
    } else {
        "dense"
    }
}

/// A plain-text inventory: metadata (long arrays summarised), then one line per tensor pattern
/// with count, shape, types and bytes, then byte totals per category.
pub fn inventory(g: &Gguf) -> Result<String> {
    let mut s = String::new();
    writeln!(
        s,
        "shards: {}",
        g.shard_paths()
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(s, "\n## metadata ({} keys)", g.meta.len())?;
    for (k, v) in &g.meta {
        writeln!(s, "{k} = {}", v.summary())?;
    }
    // pattern -> (count, dims set, type counts, bytes, layers)
    #[derive(Default)]
    struct Row {
        count: usize,
        dims: BTreeMap<String, usize>,
        types: BTreeMap<String, usize>,
        bytes: u64,
        layers: Vec<u32>,
        cat: &'static str,
    }
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    let mut cats: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut types: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let r = rows.entry(name_pattern(&t.name)).or_default();
        r.count += 1;
        *r.dims.entry(format!("{:?}", t.dims)).or_default() += 1;
        *r.types.entry(t.ty.name()).or_default() += 1;
        r.bytes += t.nbytes;
        r.cat = category(t);
        if let Some(l) = t
            .name
            .strip_prefix("blk.")
            .and_then(|x| x.split('.').next())
            .and_then(|x| x.parse().ok())
        {
            r.layers.push(l);
        }
        *cats.entry(category(t)).or_default() += t.nbytes;
        let e = types.entry(t.ty.name()).or_default();
        e.0 += 1;
        e.1 += t.nbytes;
    }
    writeln!(s, "\n## tensors ({})", g.tensors.len())?;
    writeln!(
        s,
        "pattern | count | layers | dims | types | bytes | category"
    )?;
    for (p, r) in &rows {
        let layers = if r.layers.is_empty() {
            "-".to_string()
        } else {
            compress_ranges(&r.layers)
        };
        writeln!(
            s,
            "{p} | {} | {layers} | {} | {} | {} | {}",
            r.count,
            r.dims
                .iter()
                .map(|(d, n)| if *n > 1 {
                    format!("{d}×{n}")
                } else {
                    d.clone()
                })
                .collect::<Vec<_>>()
                .join(" "),
            r.types
                .iter()
                .map(|(d, n)| format!("{d}×{n}"))
                .collect::<Vec<_>>()
                .join(" "),
            r.bytes,
            r.cat
        )?;
    }
    writeln!(s, "\n## bytes by category")?;
    let total: u64 = cats.values().sum();
    for (c, b) in &cats {
        writeln!(s, "{c}: {b} ({:.3} GiB)", *b as f64 / (1u64 << 30) as f64)?;
    }
    writeln!(
        s,
        "total: {total} ({:.3} GiB)",
        total as f64 / (1u64 << 30) as f64
    )?;
    writeln!(s, "\n## bytes by type")?;
    for (t, (n, b)) in &types {
        writeln!(
            s,
            "{t}: {n} tensors, {b} ({:.3} GiB)",
            *b as f64 / (1u64 << 30) as f64
        )?;
    }
    Ok(s)
}

fn compress_ranges(v: &[u32]) -> String {
    let mut v = v.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if i == j {
            format!("{}", v[i])
        } else {
            format!("{}-{}", v[i], v[j])
        });
        i = j + 1;
    }
    out.join(",")
}
