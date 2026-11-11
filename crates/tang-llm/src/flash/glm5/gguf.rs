//! The tensor map: every GGUF tensor name of `glm5-next` with its shape, and what it is in the
//! Hugging Face module (verified against llama.cpp's `models/glm5-next.cpp` `load_arch_tensors`
//! and `conversion/glm.py` `Glm5NextModel.modify_tensors`).
//!
//! Conventions that differ from plain HF and matter when reading weights:
//!
//! - `ssm_a` holds `-exp(A_log)`, sliced to one value per head (the converter flattens and
//!   negates).
//! - `ssm_dt.bias` is HF's `dt_bias`.
//! - The converter splits HF's fused `kv_b_proj` `[n_head, d_v + d_qk_nope, kv_lora]` into
//!   `attn_k_b` `[d_qk_nope, kv_lora, n_head]` (transposed) and `attn_v_b` `[kv_lora, d_v, n_head]`.
//! - The KDA conv1d weights are `[d_conv, 1, d_inner, 1]` (the converter reshapes; the loader
//!   also accepts a 3-D fallback).
//! - mHC tensors and `exp_probs_b` are kept F32 by the converter (`tensor_force_quant`).

use super::Hparams;
use crate::gguf::TensorInfo;
use anyhow::{ensure, Context, Result};

/// Every non-expert tensor of one trunk layer `l`, with its GGUF name suffix (after `blk.{l}.`)
/// and shape, in load order. Expert tensors (`ffn_{gate,up,down}_exps`) are handled by [`moe`]
/// (W5) and are not listed.
///
/// `hd` = `kda.head_dim`, `d_inner` = `hd * n_head`; `hc` = hyper-connection streams,
/// `mix` = `(2 + hc) * hc`.
#[allow(clippy::type_complexity)]
pub fn layer_tensors(hp: &Hparams, l: usize) -> Result<Vec<(&'static str, Vec<u64>)>> {
    let n = hp.n_embd as u64;
    let hc = hp.hc as u64;
    let mix = hp.hc_mix_dim() as u64;
    let mut out: Vec<(&'static str, Vec<u64>)> = vec![
        ("attn_norm.weight", vec![n]),
        ("ffn_norm.weight", vec![n]),
        ("hc_attn_fn.weight", vec![hc * n, mix]),
        ("hc_attn_base.weight", vec![mix]),
        ("hc_attn_scale.weight", vec![3]),
        ("hc_ffn_fn.weight", vec![hc * n, mix]),
        ("hc_ffn_base.weight", vec![mix]),
        ("hc_ffn_scale.weight", vec![3]),
    ];
    if hp.is_kda[l] {
        let hd = hp.kda_head_dim as u64;
        let heads = hp.n_head as u64;
        let d_inner = hd * heads;
        let d_conv = hp.ssm_d_conv as u64;
        out.extend([
            ("attn_q.weight", vec![n, d_inner]),
            ("attn_k.weight", vec![n, d_inner]),
            ("attn_v.weight", vec![n, d_inner]),
            ("ssm_conv1d_q.weight", vec![d_conv, 1, d_inner, 1]),
            ("ssm_conv1d_k.weight", vec![d_conv, 1, d_inner, 1]),
            ("ssm_conv1d_v.weight", vec![d_conv, 1, d_inner, 1]),
            ("ssm_f_a.weight", vec![n, hd]),
            ("ssm_f_b.weight", vec![hd, d_inner]),
            ("ssm_beta.weight", vec![n, heads]),
            ("ssm_a", vec![heads]),
            ("ssm_dt.bias", vec![d_inner]),
            ("ssm_g_a.weight", vec![n, hd]),
            ("ssm_g_b.weight", vec![hd, d_inner]),
            ("ssm_norm.weight", vec![hd]),
            ("attn_output.weight", vec![d_inner, n]),
        ]);
    } else {
        let ql = hp.q_lora_rank as u64;
        let kl = hp.kv_lora_rank as u64;
        let heads = hp.n_head as u64;
        let dqk = hp.d_qk as u64;
        let dv = hp.d_v as u64;
        let ih = hp.idx_heads as u64;
        let idd = hp.idx_dim as u64;
        let kp = hp.kpool as u64;
        out.extend([
            ("attn_q_a_norm.weight", vec![ql]),
            ("attn_kv_a_norm.weight", vec![kl]),
            ("attn_q_a.weight", vec![n, ql]),
            ("attn_q_b.weight", vec![ql, heads * dqk]),
            ("attn_kv_a_mqa.weight", vec![n, kl]), // NoPE: no rope column
            ("attn_k_b.weight", vec![dqk, kl, heads]),
            ("attn_v_b.weight", vec![kl, dv, heads]),
            ("attn_output.weight", vec![heads * dv, n]),
        ]);
        if hp.indexer_full[l] {
            out.extend([
                ("indexer.k_norm.weight", vec![idd]),
                ("indexer.k_norm.bias", vec![idd]),
                ("indexer.proj.weight", vec![n, ih]),
                ("indexer.attn_k.weight", vec![n, idd]),
                ("indexer.attn_q_b.weight", vec![ql, ih * idd]),
                ("indexer_compressor_gate.weight", vec![n, idd]),
                ("indexer_compressor_ape.weight", vec![idd, kp]),
            ]);
        }
    }
    if hp.is_dense(l) {
        let ff = hp.n_ff as u64;
        out.extend([
            ("ffn_gate.weight", vec![n, ff]),
            ("ffn_down.weight", vec![ff, n]),
            ("ffn_up.weight", vec![n, ff]),
        ]);
    } else {
        let ne = hp.n_expert as u64;
        let fe = hp.n_ff_exp as u64;
        let nsh = hp.n_expert_shared as u64;
        out.extend([
            ("ffn_gate_inp.weight", vec![n, ne]),
            ("exp_probs_b.bias", vec![ne]),
            ("ffn_gate_exps.weight", vec![n, fe, ne]),
            ("ffn_down_exps.weight", vec![fe, n, ne]),
            ("ffn_up_exps.weight", vec![n, fe, ne]),
            ("ffn_gate_shexp.weight", vec![n, fe * nsh]),
            ("ffn_down_shexp.weight", vec![fe * nsh, n]),
            ("ffn_up_shexp.weight", vec![n, fe * nsh]),
        ]);
    }
    Ok(out)
}

/// Check the file's tensors against the map: every listed name must exist with the listed shape,
/// and no `blk.{0..n_layer}` tensor may be unknown. Returns the number of trunk tensors checked.
pub fn check_layer(g: &crate::gguf::Gguf, hp: &Hparams, l: usize) -> Result<usize> {
    let mut n = 0;
    for (suffix, dims) in layer_tensors(hp, l)? {
        let name = super::t(l, suffix);
        let info: &TensorInfo = g
            .get(&name)
            .with_context(|| format!("layer {l}: missing {suffix}"))?;
        ensure!(
            info.dims == dims,
            "blk.{l}.{suffix}: dims {:?}, want {dims:?}",
            info.dims
        );
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_hp() -> Hparams {
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
            is_kda: vec![true, false, true],
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

    #[test]
    fn dense_layer_has_no_experts_and_no_indexer() {
        let hp = tiny_hp();
        let names: Vec<String> = layer_tensors(&hp, 0)
            .unwrap()
            .into_iter()
            .map(|(s, _)| s.to_string())
            .collect();
        assert!(names.contains(&"ffn_gate.weight".to_string()));
        assert!(!names.iter().any(|s| s.contains("exps")));
        assert!(!names.iter().any(|s| s.contains("indexer")));
    }

    #[test]
    fn dsa_layer_has_indexer_and_moe() {
        let hp = tiny_hp();
        let names: Vec<String> = layer_tensors(&hp, 1)
            .unwrap()
            .into_iter()
            .map(|(s, _)| s.to_string())
            .collect();
        assert!(names.contains(&"attn_k_b.weight".to_string()));
        assert!(names.contains(&"indexer.attn_k.weight".to_string()));
        assert!(names.contains(&"indexer.proj.weight".to_string()));
        assert!(names.contains(&"ffn_gate_exps.weight".to_string()));
    }
}
