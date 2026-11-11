//! Qwen3-Coder-Next at real size from its derived graph: dims from the GGUF
//! header, each graph input bound to its GGUF tensor in the tensor's own
//! format (`tang_compile::tensor::gguf`), and the derived decode step.
//!
//! The UD-Q4_K_XL file (unsloth): 843 tensors; dense weights Q8_0 (a few
//! BF16), routed experts Q4_K / Q5_K (one Q6_K `ffn_down_exps`), norms and
//! small tensors F32.

use tang_compile::tensor::{Dim, Format};
use tang_llm::gguf::{GgmlType, Gguf};

use crate::ir_coder_prefill::{CoderDims, CoderPrefill};

/// The default UD-Q4_K_XL path on mew (`CODER_GGUF` overrides).
pub const DEFAULT_GGUF: &str = "/home/cam/q3cn/Qwen3-Coder-Next-UD-Q4_K_XL.gguf";

pub fn gguf_path() -> String {
    std::env::var("CODER_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.into())
}

/// The storage format of a GGUF tensor type (`None`: F32, the graph's own).
pub fn format_of(ty: GgmlType) -> Option<Format> {
    match ty {
        GgmlType::Q8_0 => Some(Format::Q8_0),
        GgmlType::Q4K => Some(Format::Q4K),
        GgmlType::Q5K => Some(Format::Q5K),
        GgmlType::Q6K => Some(Format::Q6K),
        GgmlType::Bf16 => Some(Format::BF16),
        GgmlType::F16 => Some(Format::F16),
        _ => None,
    }
}

/// [`CoderDims`] from the `qwen3next.*` header.
pub fn dims_from_gguf(g: &Gguf) -> Result<CoderDims, String> {
    let u = |k: &str| -> Result<usize, String> {
        g.meta_u64(&format!("qwen3next.{k}"))
            .map(|v| v as usize)
            .map_err(|e| e.to_string())
    };
    let f = |k: &str| -> Result<f64, String> {
        g.meta_f64(&format!("qwen3next.{k}"))
            .map_err(|e| e.to_string())
    };
    let layers = u("block_count")?;
    let interval = u("full_attention_interval")?;
    let vocab = g.info("token_embd.weight").map_err(|e| e.to_string())?.dims[1] as usize;
    Ok(CoderDims {
        d_model: u("embedding_length")?,
        vocab,
        recurrent: (0..layers).map(|l| (l + 1) % interval != 0).collect(),
        eps: f("attention.layer_norm_rms_epsilon")?,
        n_head: u("attention.head_count")?,
        n_head_kv: u("attention.head_count_kv")?,
        head_dim: u("attention.key_length")?,
        n_rot: u("rope.dimension_count")?,
        rope_base: f("rope.freq_base")?,
        ssm_d_conv: u("ssm.conv_kernel")?,
        ssm_d_state: u("ssm.state_size")?,
        ssm_k_heads: u("ssm.group_count")?,
        ssm_v_heads: u("ssm.time_step_rank")?,
        n_expert: u("expert_count")?,
        n_expert_used: u("expert_used_count")?,
        n_ff_exp: u("expert_feed_forward_length")?,
        n_ff_shexp: u("expert_shared_feed_forward_length")?,
    })
}

/// For every graph input but `tokens`: its GGUF type, after checking the
/// file has it with the graph's shape (ggml `ne` reversed). Errors list
/// every mismatch.
pub fn bind(pg: &CoderPrefill, g: &Gguf) -> Result<Vec<Option<GgmlType>>, String> {
    let mut bad = Vec::new();
    let tys = pg
        .inputs
        .iter()
        .map(|i| {
            if i.name == "tokens" {
                return None;
            }
            let Some(t) = g.get(&i.name) else {
                bad.push(format!("{}: not in the file", i.name));
                return None;
            };
            let want: Vec<u64> = i.shape.iter().rev().map(|d: &Dim| d.n() as u64).collect();
            // ggml drops trailing 1s
            let mut have = t.dims.clone();
            while have.len() < want.len() {
                have.push(1);
            }
            if have != want {
                bad.push(format!("{}: file {:?}, graph {:?}", i.name, t.dims, want));
            }
            Some(t.ty)
        })
        .collect();
    if !bad.is_empty() {
        return Err(format!("binding: {}", bad.join("; ")));
    }
    Ok(tys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::Path;

    fn open() -> Option<Gguf> {
        let p = gguf_path();
        Path::new(&p)
            .exists()
            .then(|| Gguf::open(Path::new(&p)).unwrap())
    }

    /// `tang_compile::tensor::gguf::dequantize` against `tang_llm::gguf` on
    /// the first rows of the first tensor of every type in the file: the
    /// same f32 bits.
    #[test]
    #[ignore]
    fn dequant_matches_tang_llm_bit_for_bit() {
        let Some(g) = open() else {
            eprintln!("no {}", gguf_path());
            return;
        };
        let mut first: BTreeMap<String, &tang_llm::gguf::TensorInfo> = BTreeMap::new();
        for t in &g.tensors {
            first.entry(t.ty.name()).or_insert(t);
        }
        for (name, t) in first {
            let Some(fmt) = format_of(t.ty) else {
                continue;
            };
            let rows = t.n_rows().min(64);
            let rb = t.row_bytes().unwrap();
            let bytes = &g.bytes(t)[..rows * rb];
            let mut ours = vec![0f32; rows * t.row_len()];
            tang_compile::tensor::gguf::dequantize(fmt, bytes, &mut ours).expect("dequantize");
            let theirs = g.rows(t, 0, rows).unwrap();
            let same = ours
                .iter()
                .zip(&theirs)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            eprintln!(
                "{name:6} {}: {} values ({rows} rows) {}",
                t.name,
                ours.len(),
                if same { "bit-identical" } else { "DIFFER" }
            );
            assert!(same, "{name}");
        }
    }

    /// The real dims from the header, the real prefill graph built, and
    /// every input found in the file with its shape.
    #[test]
    #[ignore]
    fn real_graph_binds_to_the_file() {
        let Some(g) = open() else {
            eprintln!("no {}", gguf_path());
            return;
        };
        let dims = dims_from_gguf(&g).unwrap();
        eprintln!("dims: {dims:?}");
        let pg = crate::ir_coder_prefill::prefill_graph(&dims);
        let tys = bind(&pg, &g).unwrap();
        let mut count: BTreeMap<String, usize> = BTreeMap::new();
        for t in tys.iter().flatten() {
            *count.entry(t.name()).or_default() += 1;
        }
        eprintln!(
            "{} graph inputs bound ({} graph nodes); types {count:?}",
            tys.iter().flatten().count(),
            pg.g.nodes().len()
        );
    }
}
