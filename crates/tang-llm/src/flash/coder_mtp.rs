//! The Coder engine's MTP sidecar contract (`scripts/qwen3next_mtp_sidecar.py`), checked on the
//! host: every `mtp.*` tensor the engine loads (`Coder::load`, `load_layer(.., "mtp.", ..)`),
//! with the shape the trunk's hyperparameters require. The engine checks a sidecar with it
//! before uploading; `flash-mtp-train` checks its exports with it.
use super::reference::Hparams;
use crate::gguf::Gguf;
use anyhow::{ensure, Context, Result};

/// `(name, rows, row length)` of every tensor the engine reads from a sidecar; experts are
/// `n_expert` stacked matrices of that shape. Rows 1 means a vector.
pub fn sidecar_tensors(hp: &Hparams) -> Vec<(&'static str, usize, usize, usize)> {
    let (n, hd, nh, nkv, ff, ne) = (
        hp.n_embd,
        hp.head_dim,
        hp.n_head,
        hp.n_head_kv,
        hp.n_ff_exp,
        hp.n_expert,
    );
    vec![
        ("mtp.fc.weight", n, 2 * n, 1),
        ("mtp.enorm.weight", 1, n, 1),
        ("mtp.hnorm.weight", 1, n, 1),
        ("mtp.norm.weight", 1, n, 1),
        ("mtp.attn_norm.weight", 1, n, 1),
        ("mtp.post_attention_norm.weight", 1, n, 1),
        ("mtp.attn_q.weight", 2 * nh * hd, n, 1),
        ("mtp.attn_k.weight", nkv * hd, n, 1),
        ("mtp.attn_v.weight", nkv * hd, n, 1),
        ("mtp.attn_output.weight", n, nh * hd, 1),
        ("mtp.attn_q_norm.weight", 1, hd, 1),
        ("mtp.attn_k_norm.weight", 1, hd, 1),
        ("mtp.ffn_gate_inp.weight", ne, n, 1),
        ("mtp.ffn_gate_inp_shexp.weight", 1, n, 1),
        ("mtp.ffn_gate_shexp.weight", ff, n, 1),
        ("mtp.ffn_up_shexp.weight", ff, n, 1),
        ("mtp.ffn_down_shexp.weight", n, ff, 1),
        ("mtp.ffn_gate_exps.weight", ff, n, ne),
        ("mtp.ffn_up_exps.weight", ff, n, ne),
        ("mtp.ffn_down_exps.weight", n, ff, ne),
    ]
}

/// Check that `g` holds a loadable MTP layer for a trunk with hyperparameters `hp`.
pub fn check_sidecar(g: &Gguf, hp: &Hparams) -> Result<()> {
    for (name, rows, k, ne) in sidecar_tensors(hp) {
        let t = g.info(name).context("MTP sidecar")?;
        ensure!(
            t.ty.can_dequantize(),
            "{name}: type {} unreadable",
            t.ty.name()
        );
        ensure!(
            t.row_len() == k && t.n_elements() as usize == rows * k * ne,
            "{name}: dims {:?}, want {ne} x {rows} x {k}",
            t.dims
        );
        ensure!(
            ne == 1 || t.dims.len() == 3,
            "{name}: not an expert stack ({:?})",
            t.dims
        );
    }
    Ok(())
}
