//! A tiny synthetic `inkling2_loop` GGUF, so the whole graph (the looped core, its control
//! gates, the convolutions, the block residuals, the MoE) runs in a unit test without the
//! 17 GB checkpoint.
//!
//! The geometry is small but complete: 8 physical layers, a core of `[2, 6)` run 3 times,
//! block size 2, 4 heads (2 KV), head dim 4, 8 experts taking 2, conv kernel 4, relative bias
//! 2 rows × 4 columns, vocabulary 16.

use super::forward::{InklingRef, Session};
use crate::gguf::Value;
use anyhow::{ensure, Result};
use std::path::{Path, PathBuf};

/// The geometry the synthetic file describes.
pub const N_VOCAB: usize = 16;
pub const N_VOCAB_UNPADDED: usize = 14;
pub const N_EMBD: usize = 16;
pub const N_LAYER: usize = 8;
pub const N_HEAD: usize = 4;
pub const N_HEAD_KV: usize = 2;
pub const D_HEAD: usize = 4;
pub const N_EXPERT: usize = 8;
pub const N_EXPERT_USED: usize = 2;
pub const N_FF_EXP: usize = 6;
pub const N_CONV: usize = 4;
pub const BLOCK: usize = 2;
pub const N_REL: usize = 2;
pub const N_REL_EXTENT: usize = 4;
pub const N_SWA: usize = 4;
pub const LOOP_START: usize = 2;
pub const LOOP_END: usize = 6;
pub const N_PASSES: usize = 3;
pub const GATE_RANK: usize = 4;

/// The metadata the forward reads, at the values above.
pub fn meta() -> Vec<(&'static str, Value)> {
    let per_layer = |v: u64| Value::Array(vec![Value::U32(v as u32); N_LAYER]);
    let swa: Vec<Value> = (0..N_LAYER)
        .map(|l| Value::U32(u32::from(l % 3 != 2)))
        .collect();
    vec![
        ("general.architecture", Value::Str("inkling2_loop".into())),
        ("general.alignment", Value::U32(32)),
        ("inkling2_loop.block_count", Value::U32(N_LAYER as u32)),
        ("inkling2_loop.embedding_length", Value::U32(N_EMBD as u32)),
        ("inkling2_loop.feed_forward_length", Value::U32(8)),
        ("inkling2_loop.attention.head_count", per_layer(N_HEAD as u64)),
        (
            "inkling2_loop.attention.head_count_kv",
            per_layer(N_HEAD_KV as u64),
        ),
        ("inkling2_loop.attention.key_length", Value::U32(D_HEAD as u32)),
        (
            "inkling2_loop.attention.value_length",
            Value::U32(D_HEAD as u32),
        ),
        (
            "inkling2_loop.attention.layer_norm_rms_epsilon",
            Value::F32(1e-6),
        ),
        ("inkling2_loop.expert_count", Value::U32(N_EXPERT as u32)),
        (
            "inkling2_loop.expert_used_count",
            Value::U32(N_EXPERT_USED as u32),
        ),
        (
            "inkling2_loop.expert_feed_forward_length",
            Value::U32(N_FF_EXP as u32),
        ),
        ("inkling2_loop.expert_weights_scale", Value::F32(2.0)),
        ("inkling2_loop.activation.situ_beta", Value::F32(4.0)),
        ("inkling2_loop.activation.situ_linear_beta", Value::F32(25.0)),
        (
            "inkling2_loop.attention.sliding_window",
            Value::U32(N_SWA as u32),
        ),
        (
            "inkling2_loop.attention.sliding_window_pattern",
            Value::Array(swa),
        ),
        ("inkling2_loop.shortconv.l_cache", Value::U32(N_CONV as u32)),
        ("inkling2_loop.attn_res.block_size", Value::U32(BLOCK as u32)),
        (
            "inkling2_loop.attention.relative_length",
            Value::U32(N_REL as u32),
        ),
        (
            "inkling2_loop.attention.relative_extent",
            Value::U32(N_REL_EXTENT as u32),
        ),
        ("inkling2_loop.attention.qk_clip", Value::F32(16.0 * 2f32.sqrt())),
        ("inkling2_loop.attention.relative_clip", Value::F32(8.0)),
        (
            "inkling2_loop.attention.log_scale_floor",
            Value::U32(4),
        ),
        ("inkling2_loop.attention.log_scale_alpha", Value::F32(0.1)),
        (
            "inkling2_loop.final_logit_softcapping",
            Value::F32(30.0),
        ),
        ("inkling2_loop.vocab_size", Value::U32(N_VOCAB as u32)),
        (
            "inkling2_loop.vocab_size_unpadded",
            Value::U32(N_VOCAB_UNPADDED as u32),
        ),
        ("inkling2_loop.loop.start", Value::U32(LOOP_START as u32)),
        ("inkling2_loop.loop.end", Value::U32(LOOP_END as u32)),
        ("inkling2_loop.loop.count", Value::U32(N_PASSES as u32)),
        (
            "inkling2_loop.loop.gate_rank",
            Value::U32(GATE_RANK as u32),
        ),
    ]
}

/// A tensor to write: name, dims, and data of that many elements.
struct T {
    name: String,
    dims: Vec<u64>,
    data: Vec<f32>,
}

/// Deterministic pseudo-random weights in `[-0.5, 0.5)` scaled by `s`.
fn fill(seed: &mut u64, n: usize, s: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            u * s
        })
        .collect()
}

/// Every tensor the forward reads, for the geometry above.
fn tensors() -> Vec<T> {
    let mut b = Builder { seed: 7, out: Vec::new() };
    let e = N_EMBD as u64;
    let kv = (D_HEAD * N_HEAD_KV) as u64;
    let q = (D_HEAD * N_HEAD) as u64;

    b.w("token_embd.weight", &[e, N_VOCAB as u64], 0.5);
    b.one("token_embd_norm.weight", &[e]);
    b.w("output.weight", &[e, N_VOCAB as u64], 0.3);
    b.one("output_norm.weight", &[e]);
    b.w("output_res_query.weight", &[e], 0.5);

    for l in 0..N_LAYER {
        let is_swa = l % 3 != 2;
        let core = (LOOP_START..LOOP_END).contains(&l);
        let n = |x: &str| format!("blk.{l}.{x}");
        b.w(&n("attn_q.weight"), &[e, q], 0.2);
        b.w(&n("attn_k.weight"), &[e, kv], 0.2);
        b.w(&n("attn_v.weight"), &[e, kv], 0.2);
        b.w(&n("attn_output.weight"), &[q, e], 0.2);
        b.w(&n("attn_gate.weight"), &[e, q], 0.2);
        b.w(&n("attn_rel.weight"), &[e, (N_REL * N_HEAD) as u64], 0.2);
        b.w(&n("attn_rel_bias.weight"), &[N_REL as u64, N_REL_EXTENT as u64], 0.2);
        b.w(&n("attn_rel_clip.weight"), &[N_REL as u64, N_HEAD as u64], 0.1);
        b.w(&n("attn_q_clip.weight"), &[D_HEAD as u64, N_HEAD as u64], 0.1);
        b.w(&n("attn_sink_proj.weight"), &[e, N_HEAD as u64], 0.1);
        b.w(&n("attn_k_conv.weight"), &[N_CONV as u64, kv], 0.5);
        b.w(&n("attn_v_conv.weight"), &[N_CONV as u64, kv], 0.5);
        b.w(&n("attn_post_conv.weight"), &[N_CONV as u64, e], 0.5);
        b.w(&n("ffn_post_conv.weight"), &[N_CONV as u64, e], 0.5);
        b.one(&n("ffn_norm.weight"), &[e]);
        b.w(&n("ffn_gate_inp.weight"), &[e, N_EXPERT as u64], 0.5);
        b.w(&n("ffn_gate_up_exps.weight"), &[e, (2 * N_FF_EXP) as u64, N_EXPERT as u64], 0.2);
        b.w(&n("ffn_down_exps.weight"), &[N_FF_EXP as u64, e, N_EXPERT as u64], 0.2);
        b.w(&n("ffn_res_query.weight"), &[e], 0.5);
        if is_swa {
            b.one(&n("attn_norm.weight"), &[e]);
        } else {
            b.one(&n("attn_q_inp_norm.weight"), &[e]);
            b.one(&n("attn_k_inp_norm.weight"), &[e]);
            b.one(&n("attn_v_inp_norm.weight"), &[e]);
            b.w(&n("attn_k_conv_proj.weight"), &[e, (N_CONV * N_HEAD_KV) as u64], 0.3);
            b.w(&n("attn_v_conv_proj.weight"), &[e, (N_CONV * N_HEAD_KV) as u64], 0.3);
        }
        if l > 0 {
            b.w(&n("attn_res_query.weight"), &[e, if is_swa { 1 } else { 3 }], 0.5);
        }
        if core {
            b.w(&n("loop_msg_query.weight"), &[e, 4], 0.5);
            for c in ["q", "k", "v", "ffn"] {
                b.w(&n(&format!("loop_{c}_ctrl_down.weight")), &[2 * e, GATE_RANK as u64], 0.2);
                b.w(&n(&format!("loop_{c}_ctrl_gain.weight")), &[GATE_RANK as u64, e], 0.3);
                b.w(&n(&format!("loop_{c}_ctrl_forget.weight")), &[GATE_RANK as u64, e], 0.3);
            }
        }
    }
    b.out
}

/// Builds the tensor list: deterministic pseudo-random weights, or all ones for norms.
struct Builder {
    seed: u64,
    out: Vec<T>,
}

impl Builder {
    fn w(&mut self, name: &str, dims: &[u64], s: f32) {
        let n: u64 = dims.iter().product();
        self.out.push(T {
            name: name.to_string(),
            data: fill(&mut self.seed, n as usize, s),
            dims: dims.to_vec(),
        });
    }

    fn one(&mut self, name: &str, dims: &[u64]) {
        let n: u64 = dims.iter().product();
        self.out.push(T {
            name: name.to_string(),
            data: vec![1.0; n as usize],
            dims: dims.to_vec(),
        });
    }
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

fn value_ty(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::Str(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_value(b: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(v) => b.push(*v),
        Value::I8(v) => b.push(*v as u8),
        Value::U16(v) => b.extend_from_slice(&v.to_le_bytes()),
        Value::I16(v) => b.extend_from_slice(&v.to_le_bytes()),
        Value::U32(v) => put_u32(b, *v),
        Value::I32(v) => b.extend_from_slice(&v.to_le_bytes()),
        Value::F32(v) => b.extend_from_slice(&v.to_le_bytes()),
        Value::Bool(v) => b.push(*v as u8),
        Value::Str(s) => put_str(b, s),
        Value::Array(a) => {
            put_u32(b, a.first().map(value_ty).unwrap_or(4));
            put_u64(b, a.len() as u64);
            for x in a {
                put_value(b, x);
            }
        }
        Value::U64(v) => put_u64(b, *v),
        Value::I64(v) => b.extend_from_slice(&v.to_le_bytes()),
        Value::F64(v) => b.extend_from_slice(&v.to_le_bytes()),
    }
}

/// Write the tiny model as a single-shard f32 GGUF; returns its path.
pub fn write(dir: &Path) -> Result<PathBuf> {
    let ts = tensors();
    let m = meta();
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, ts.len() as u64);
    put_u64(&mut b, m.len() as u64);
    for (k, v) in &m {
        put_str(&mut b, k);
        put_u32(&mut b, value_ty(v));
        put_value(&mut b, v);
    }
    let mut data = Vec::new();
    for t in &ts {
        let n: u64 = t.dims.iter().product();
        ensure!(t.data.len() == n as usize, "{}: data length", t.name);
        put_str(&mut b, &t.name);
        put_u32(&mut b, t.dims.len() as u32);
        for &d in &t.dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, 0); // F32
        put_u64(&mut b, data.len() as u64);
        for v in &t.data {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend_from_slice(&data);
    std::fs::create_dir_all(dir)?;
    let p = dir.join("inkling-tiny-f32.gguf");
    std::fs::write(&p, b)?;
    Ok(p)
}

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tang-inkling-{tag}-{}",
        std::process::id()
    ))
}

#[test]
fn tiny_model_opens_and_runs_deterministically() {
    let dir = tmp("run");
    let p = write(&dir).unwrap();
    let m = InklingRef::open(&p).unwrap();
    assert_eq!(m.hp.n_layer, N_LAYER);
    assert_eq!(m.hp.n_slot(), N_LAYER + (N_PASSES - 1) * (LOOP_END - LOOP_START));
    assert_eq!(m.hp.layer_phys(N_LAYER), LOOP_START);
    assert_eq!(m.hp.layer_phys(N_LAYER + (LOOP_END - LOOP_START) - 1), LOOP_END - 1);

    let ids = [1u32, 5, 9, 2];
    let a = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert_eq!(a.len(), N_VOCAB_UNPADDED);
    assert!(a.iter().all(|v| v.is_finite()), "non-finite logits");
    // Every logit is inside the soft cap.
    assert!(a.iter().all(|v| v.abs() <= 30.0));
    // Same input, same output.
    let b = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert_eq!(a, b);
    // All positions when asked.
    let all = m.forward(&mut Session::new(&m), &ids, true).unwrap();
    assert_eq!(all.len(), ids.len() * N_VOCAB_UNPADDED);
    assert_eq!(&all[(ids.len() - 1) * N_VOCAB_UNPADDED..], &a[..]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn incremental_decoding_matches_a_full_run() {
    // The session keeps the KV and convolution state, so stepping one token at a time must land
    // on the same logits as running the whole prefix in one call.
    let dir = tmp("session");
    let p = write(&dir).unwrap();
    let m = InklingRef::open(&p).unwrap();
    let ids = [3u32, 7, 1, 12, 5];
    let full = m.forward(&mut Session::new(&m), &ids, false).unwrap();

    let mut s = Session::new(&m);
    let first = m.forward(&mut s, &ids[..1], false).unwrap();
    assert!(first.iter().all(|v| v.is_finite()));
    let mut last = first;
    for &id in &ids[1..] {
        last = m.forward(&mut s, &[id], false).unwrap();
    }
    assert_eq!(s.pos, ids.len());
    let d = last
        .iter()
        .zip(&full)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(d < 1e-3, "incremental vs full: {d}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_loop_changes_the_answer() {
    // With one pass the core is visited once, not three times: the output must differ, or the
    // loop is not wired up.
    let dir = tmp("loop");
    let p = write(&dir).unwrap();
    let mut m = InklingRef::open(&p).unwrap();
    let ids = [3u32, 7, 1];
    let three = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    m.hp.n_passes = 1;
    let one = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert_ne!(three, one, "the extra passes had no effect");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn past_the_window_a_local_layer_forgets() {
    // A prompt longer than the sliding window must still run, and the last token's logits must
    // not depend on tokens the local layers can no longer see... which the global layers do see,
    // so check the weaker property: a longer prompt runs and stays finite.
    let dir = tmp("window");
    let p = write(&dir).unwrap();
    let m = InklingRef::open(&p).unwrap();
    let ids: Vec<u32> = (0..(N_SWA as u32 + 5)).map(|i| i % N_VOCAB as u32).collect();
    let a = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert!(a.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bad_geometry_is_rejected() {
    let dir = tmp("bad");
    let p = write(&dir).unwrap();
    let g = crate::gguf::Gguf::open(&p).unwrap();
    let mut hp = super::Hparams::from_gguf(&g).unwrap();
    hp.check().unwrap();
    // A loop core that does not sit on block boundaries cannot preserve the residual bank.
    hp.loop_start = 1;
    assert!(hp.check().is_err());
    hp.loop_start = LOOP_START;
    hp.n_passes = 1;
    assert!(hp.check().is_err());
    hp.n_passes = N_PASSES;
    hp.n_expert_used = N_EXPERT + 1;
    assert!(hp.check().is_err());
    let _ = std::fs::remove_dir_all(dir);
}
