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

/// A lowered real-size step and how its weights are stored.
pub struct RealStep {
    pub graph: tang_compile::tensor::TensorGraph,
    /// `[logits, updated caches…]` in `graph`.
    pub outs: Vec<tang_compile::tensor::TensorId>,
    pub program: tang_compile::tensor::cuda_gemv::DecodeProgram,
    pub sched: tang_compile::tensor::int8_sched::BlockSchedule,
    /// Step input index → the storage its buffer holds (`None`: f32 values,
    /// a block tensor's dequantized when its GGUF type is not f32).
    pub storage: Vec<Option<Format>>,
    /// Names of GGUF block tensors uploaded as dequantized f32.
    pub f32_inputs: Vec<String>,
    /// `(updated cache, step input)` pairs written in place.
    pub inplace: Vec<(tang_compile::tensor::TensorId, usize)>,
    /// The residual stream's nodes in `graph`, layer order (kept
    /// materialized).
    pub residual: Vec<tang_compile::tensor::TensorId>,
}

/// The derived step with its weights stored as the GGUF stores them where
/// the int8 GEMV reads them, through the decode passes and `to_cuda_decode`.
/// Full-capacity row caches update in place; rings and states are written
/// to fresh buffers (the runner swaps them in).
pub fn lower_real_step(
    pg: &CoderPrefill,
    st: &tang_compile::tensor::step::StepGraph,
    tys: &[Option<GgmlType>],
) -> RealStep {
    lower_real_step_with(pg, st, tys, &RealOpts::default())
}

/// Options of [`lower_real_step_with`]: a batched step
/// ([`derived_step_batched`](crate::ir_coder_prefill::derived_step_batched)).
#[derive(Clone, Copy, Debug, Default)]
pub struct RealOpts {
    /// Streams the step's batch dim `B` is bound to (`None`: an unbatched
    /// step).
    pub batch: Option<usize>,
    /// Routed experts grouped by expert from this many (stream, slot)
    /// positions (`schedule_expert_groups`; 0: never).
    pub group: usize,
    /// Q6_K tensors stored Q8_0 (the uploader requantizes them with
    /// [`q8_0_bytes`]) so the int8 GEMV reads them instead of an f32
    /// gather: coder-next's blk.46 routed down, 2.7 ms at B = 64 as f32.
    pub q6k_as_q8_0: bool,
    /// Full-capacity K/V caches stored f16 (`Packing::F16`, decoded at
    /// load, the in-place update stores halves): half the attention bytes.
    pub kv_f16: bool,
    /// Stages of the int8 contractions by their positions.
    /// `None`: `schedule_block_weights` (the decode GEMV, the MMA tile for
    /// many-row steps).
    pub block: Option<tang_compile::tensor::int8_sched::BlockOpts>,
}

/// [`lower_real_step`] with [`RealOpts`].
pub fn lower_real_step_with(
    pg: &CoderPrefill,
    st: &tang_compile::tensor::step::StepGraph,
    tys: &[Option<GgmlType>],
    ro: &RealOpts,
) -> RealStep {
    use tang_compile::tensor::step::StepInput;
    use tang_compile::tensor::{Op, Precision};
    let mut graph = match ro.batch {
        Some(b) => st.graph.bind(&[("B", b)]).expect("bind the batch dim"),
        None => st.graph.clone(),
    };
    let input_node = |g: &tang_compile::tensor::TensorGraph, k: usize| {
        (0..g.nodes().len())
            .find(|&i| matches!(g.nodes()[i].op, Op::Input(j) if j == k))
            .map(tang_compile::tensor::TensorId::from_index)
            .unwrap()
    };
    let gguf_of = |k: usize| match st.inputs[k] {
        StepInput::Same(i) => tys[i],
        _ => None,
    };
    if ro.batch.unwrap_or(1) > 1
        && !ro.q6k_as_q8_0
        && (0..st.inputs.len()).any(|k| gguf_of(k) == Some(GgmlType::Q6K))
    {
        // the f32 gather such a tensor becomes is wrong at B > 1 on the GPU
        // (coder-next blk.46 ffn_down_exps: B=4 vs B=1 top-1 0/4); right
        // on the CPU harness. Store it Q8_0 instead.
        eprintln!(
            "warning: batched step with Q6_K weights kept f32; set RealOpts::q6k_as_q8_0 \
             (and upload with weight_words)"
        );
    }
    for k in 0..st.inputs.len() {
        if let Some(f) = gguf_of(k).and_then(format_of) {
            let f = if ro.q6k_as_q8_0 && f == Format::Q6K {
                Format::Q8_0
            } else {
                f
            };
            if matches!(f, Format::Q4K | Format::Q5K | Format::Q8_0) {
                graph.set_precision(input_node(&graph, k), Precision::storage(f));
            }
        }
    }
    let mut outs = st.outputs.clone();
    outs.extend(st.caches.iter().map(|c| c.updated));
    let (graph, outs) = if ro.group > 0 {
        let (g, o, eg) = graph.schedule_expert_groups(&outs, ro.group, true);
        eprintln!(
            "expert groups: {} routed contractions sorted by expert",
            eg.grouped.len()
        );
        (g, o)
    } else {
        (graph, outs)
    };
    let (graph, outs, sched) = match &ro.block {
        Some(bo) => graph.schedule_block_weights_with(&outs, bo),
        None => graph.schedule_block_weights(&outs),
    };
    let n = outs.len();
    let mut keep = outs.clone();
    keep.extend(&sched.copies);
    // the residual stream stays materialized (as `DecodeGraph::keep` does
    // for Qwen3): a [k, d] map read by several kernels would otherwise be
    // substituted into each reader, every layer's norm recomputing the
    // whole stream back to the (gathered) embedding
    let rs = residual_stream(&graph, pg);
    let r0 = keep.len();
    keep.extend(&rs);
    let (graph, keep) = graph.substitute_maps(&keep);
    let (mut graph, keep) = graph.eliminate_common(&keep);
    let outs = keep[..n].to_vec();
    let residual = keep[r0..].to_vec();
    let storage: Vec<Option<Format>> = (0..st.inputs.len())
        .map(|k| {
            let s = graph.nodes()[input_node(&graph, k).index()]
                .precision
                .storage;
            s.is_block().then_some(s)
        })
        .collect();
    let f32_inputs = (0..st.inputs.len())
        .filter(|&k| storage[k].is_none() && gguf_of(k).and_then(format_of).is_some())
        .map(|k| match st.inputs[k] {
            StepInput::Same(i) => pg.inputs[i].name.clone(),
            _ => unreachable!(),
        })
        .collect();
    let input_of = |t: tang_compile::tensor::TensorId| match st.graph.nodes()[t.index()].op {
        Op::Input(i) => i,
        _ => unreachable!(),
    };
    let opts = tang_compile::tensor::cuda_gemv::DecodeOpts {
        // step tiles (TF32, and split_bilinear's softmax factor) off for
        // batched steps: B=4 0.4 ms slower, and at B=16 the race bar's max
        // logprob gap grew to 1.25 (vs 0.2 without). On from
        // `TANG_STEP_MMA_MIN_B` streams (B=64 ~6k: 38.7 vs 40.1 ms)
        no_step_mma: {
            let b = ro.batch.unwrap_or(1);
            let min_b: usize = std::env::var("TANG_STEP_MMA_MIN_B")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(usize::MAX);
            b > 1 && b < min_b
        },
        packed: if ro.kv_f16 {
            st.caches
                .iter()
                .filter(|s| !s.state && s.ring.is_none())
                .map(|s| {
                    (
                        input_of(s.input),
                        tang_compile::tensor::cuda_gemv::Packing::F16,
                    )
                })
                .collect()
        } else {
            Vec::new()
        },
        inplace: st
            .caches
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.state && s.ring.is_none())
            .map(|(c, s)| (outs[st.outputs.len() + c], input_of(s.input)))
            .collect(),
        ..Default::default()
    };
    // norms' and gates' statistics recomputed inside their maps (one
    // kernel a norm, Q8x32 stores included), small maps (the GDN gates)
    // evaluated in their readers. On by default: with cokernels (side jobs on
    // their own blocks) H200 5.12 -> 4.83 ms, 1649 -> 1385 kernels; TANG_STAGED_MAP=0 off
    // (not for B > 1 streams: there the staged maps cost more than they
    // save (B=4 H200: 9.1 vs 10.4 ms) and broke the predicated in-place
    // cache update)
    if ro.batch.unwrap_or(1) <= 1
        && tang_compile::tensor::tune::step_knob("coder-next", "TANG_STAGED_MAP").as_deref()
            != Some("0")
    {
        let inplace: Vec<_> = opts.inplace.iter().map(|&(t, _)| t).collect();
        graph.schedule_staged_stats(&outs, &inplace);
        graph.schedule_small_maps(&outs, &inplace);
    }
    // multi-output kernels: residual adds folded into their norms (and
    // norms into their Q8x32 copies), conv windows into the convs, ring
    // and state updates as side jobs on blocks of their own. On by default (H200: 5.30 -> 5.13 ms,
    // 1924 -> 1649 kernels); TANG_COKERNEL=0 off
    // (decode steps only: a prefill chunk's many-row hosts lower through paths without
    // side jobs, "a multi-output kernel needs a native lowering")
    let tok_rows = st
        .inputs
        .iter()
        .position(|s| matches!(s, StepInput::New(_)))
        .map(|k| {
            graph.nodes()[input_node(&graph, k).index()]
                .shape
                .iter()
                .map(|d| match d {
                    tang_compile::tensor::Dim::Fixed(n) => *n,
                    _ => 1,
                })
                .product::<usize>()
        });
    if tok_rows.unwrap_or(1) == 1
        && tang_compile::tensor::tune::step_knob("coder-next", "TANG_COKERNEL").as_deref()
            != Some("0")
    {
        let (a, b) = graph.schedule_cokernels(&outs, &opts);
        eprintln!("multi-output kernels: {a} nodes inlined and stored, {b} side jobs");
    }
    let program = graph.to_cuda_decode(&outs, &opts);
    RealStep {
        graph,
        outs,
        program,
        sched,
        storage,
        f32_inputs,
        inplace: opts.inplace,
        residual,
    }
}

/// The residual adds of shape `[k, d_model]` (`a + b` of two plain reads)
/// read by more than one comp: the residual stream of the step.
pub fn residual_stream(
    g: &tang_compile::tensor::TensorGraph,
    pg: &CoderPrefill,
) -> Vec<tang_compile::tensor::TensorId> {
    let dm = pg
        .inputs
        .iter()
        .find(|i| i.name == "token_embd.weight")
        .map(|i| i.shape[1].n())
        .unwrap();
    let n = g.nodes().len();
    let mut readers = vec![0usize; n];
    for i in 0..n {
        if let Some(c) = g.comp_at(i) {
            let mut seen = Vec::new();
            for a in &c.args {
                if !seen.contains(&a.src.index()) {
                    seen.push(a.src.index());
                    readers[a.src.index()] += 1;
                }
            }
        }
    }
    // a residual add: the sum of two args, each a plain (ungathered) read
    let is_add = |c: &tang_compile::tensor::Comp| {
        use tang_expr::Node;
        let nodes = c.body.nodes_slice();
        let var = |e: tang_expr::ExprId| matches!(nodes[e.index() as usize], Node::Var(_));
        c.args.len() == 2
            && c.args.iter().all(|a| !a.view.is_gather() && !a.view.guard)
            && c.body_outs.len() == 1
            && matches!(nodes[c.body_outs[0].index() as usize], Node::Add(x, y) if var(x) && var(y))
    };
    (0..n)
        .filter(|&i| {
            g.comp_at(i)
                .is_some_and(|c| c.red.is_empty() && c.monoid.is_none() && is_add(c))
                && g.nodes()[i].precision.storage == tang_compile::tensor::Format::F32
                // `[k, d]`, or `[B, k, d]` in a batched step
                && (2..=3).contains(&g.nodes()[i].shape.len())
                && g.nodes()[i].shape.last().unwrap().n() == dm
                && readers[i] >= 2
        })
        .map(tang_compile::tensor::TensorId::from_index)
        .collect()
}

/// The words to upload for GGUF tensor `t` into a step input stored as
/// `storage` (`RealStep::storage`): raw block words where the int8 GEMV reads
/// the file's format, a Q6_K tensor requantized to Q8_0 when it is stored
/// Q8_0 ([`RealOpts::q6k_as_q8_0`]), else the dequantized f32 values.
pub fn weight_words(g: &Gguf, t: &tang_llm::gguf::TensorInfo, storage: Option<Format>) -> Vec<f32> {
    let words = |b: &[u8]| -> Vec<f32> {
        b.chunks(4)
            .map(|c| {
                let mut a = [0u8; 4];
                a[..c.len()].copy_from_slice(c);
                f32::from_le_bytes(a)
            })
            .collect()
    };
    match (storage, t.ty) {
        (Some(Format::Q8_0), GgmlType::Q6K) => words(&q8_0_bytes(&g.dequantize(t).unwrap())),
        (Some(_), _) | (None, GgmlType::F32) => words(g.bytes(t)),
        (None, _) => g.dequantize(t).unwrap(),
    }
}

/// f32 → IEEE half bits, round to nearest even (finite, non-negative scales;
/// subnormals flush to zero, overflow saturates).
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7bff;
    }
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rest = m & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h.min(0x7bff) as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    let v = if e == 0 {
        m * 2f32.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    };
    if h & 0x8000 != 0 {
        -v
    } else {
        v
    }
}

/// `vals` (a multiple of 32) as GGUF Q8_0 blocks: per 32 values an f16
/// scale `max|x| / 127` and 32 int8 codes `round(x / d)`.
pub fn q8_0_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for blk in vals.chunks(32) {
        let a = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let d = a / 127.0;
        let dh = f32_to_f16_bits(d);
        let dq = f16_bits_to_f32(dh);
        out.extend_from_slice(&dh.to_le_bytes());
        for &x in blk {
            let q = if dq == 0.0 {
                0.0
            } else {
                (x / dq).round().clamp(-127.0, 127.0)
            };
            out.push(q as i8 as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::Path;

    /// The tiny coder step through the real-size lowering's map passes
    /// (residual stream kept), every kernel listed (`--nocapture`).
    #[test]
    fn tiny_step_kernels() {
        use crate::ir_coder_prefill::*;
        let pg = prefill_graph(&CoderDims::tiny());
        let st = derived_step(&pg, 1, 21).unwrap();
        let mut outs = st.outputs.clone();
        outs.extend(st.caches.iter().map(|c| c.updated));
        let n = outs.len();
        let mut keep = outs.clone();
        keep.extend(residual_stream(&st.graph, &pg));
        let (g, keep) = st.graph.substitute_maps(&keep);
        let (g, keep) = g.eliminate_common(&keep);
        let dp = g.to_cuda_decode(&keep[..n], &Default::default());
        for (k, nat) in dp.program.kernels.iter().zip(&dp.native) {
            let i = k.node.index();
            let what = match g.comp_at(i) {
                Some(c) => format!(
                    "red {:?} args {:?}",
                    c.red,
                    c.args
                        .iter()
                        .map(|a| (a.src.index(), g.nodes()[a.src.index()].shape.clone()))
                        .collect::<Vec<_>>()
                ),
                None => "not a comp".into(),
            };
            eprintln!(
                "K n{i} {:?} nat {nat} {what} | {}",
                g.nodes()[i].shape,
                k.source
                    .lines()
                    .find(|l| l.contains("tang-") || l.starts_with("//"))
                    .unwrap_or("")
            );
        }
        if let Ok(dir) = std::env::var("DUMP_KERNELS") {
            for (j, k) in dp.program.kernels.iter().enumerate() {
                std::fs::write(format!("{dir}/k{j:03}_n{}.cu", k.node.index()), &k.source).unwrap();
            }
        }
        eprintln!("{} kernels", dp.program.kernels.len());
    }

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

    /// The real-size decode step (k = 1, capacity 4096) derived from the
    /// real prefill graph, then lowered by `to_cuda_decode` with f32
    /// weights (no storage formats yet): kernels, native, reductions over
    /// the capacity, and (with `--features cuda`) nvrtc.
    #[test]
    #[ignore]
    fn real_step_derives_and_lowers_f32() {
        let Some(g) = open() else {
            eprintln!("no {}", gguf_path());
            return;
        };
        let dims = dims_from_gguf(&g).unwrap();
        let pg = crate::ir_coder_prefill::prefill_graph(&dims);
        let t0 = std::time::Instant::now();
        let st = crate::ir_coder_prefill::derived_step(&pg, 1, 4096).unwrap();
        let derive = t0.elapsed();
        let rows = st
            .caches
            .iter()
            .filter(|c| !c.state && c.ring.is_none())
            .count();
        let rings = st.caches.iter().filter(|c| c.ring.is_some()).count();
        let states = st.caches.iter().filter(|c| c.state).count();
        let mut outs = st.outputs.clone();
        outs.extend(st.caches.iter().map(|c| c.updated));
        let (sg, outs) = st.graph.substitute_maps(&outs);
        let (sg, outs) = sg.eliminate_common(&outs);
        let dp = sg.to_cuda_decode(&outs, &Default::default());
        let ks = &dp.program.kernels;
        let native = dp.native.iter().filter(|n| **n).count();
        let cap = ks
            .iter()
            .filter(|k| {
                sg.comp_at(k.node.index())
                    .is_some_and(|c| c.red.iter().any(|d| d.n() == 4096))
            })
            .count();
        eprintln!(
            "real coder step: derived in {derive:.2?} ({} step nodes); caches: {rows} rows, {rings} rings, {states} states; \
             CUDA: {} kernels, {native} native, {cap} reduce over the capacity; total {:.2?}",
            st.graph.nodes().len(),
            ks.len(),
            t0.elapsed()
        );
        #[cfg(feature = "cuda")]
        {
            let mut srcs: Vec<&str> = ks.iter().map(|k| k.source.as_str()).collect();
            srcs.sort_unstable();
            srcs.dedup();
            for s in &srcs {
                assert!(tang_gpu::cuda::ptx(s, "sm_90").contains(".entry"));
            }
            eprintln!(
                "  nvrtc: {} distinct sources compiled to sm_90 PTX",
                srcs.len()
            );
        }
    }

    /// The real-size step with the weights in their GGUF formats: Q4_K /
    /// Q5_K / Q8_0 contractions on the int8 GEMV (Q8x32 activation copies),
    /// everything else (the Q6_K experts, BF16, the Q8_0 embedding gather)
    /// uploaded as dequantized f32. Kernels, int8 GEMVs, PTX.
    #[test]
    #[ignore]
    fn real_step_lowers_with_gguf_formats() {
        let Some(g) = open() else {
            eprintln!("no {}", gguf_path());
            return;
        };
        let dims = dims_from_gguf(&g).unwrap();
        let pg = crate::ir_coder_prefill::prefill_graph(&dims);
        let tys = bind(&pg, &g).unwrap();
        let st = crate::ir_coder_prefill::derived_step(&pg, 1, 4096).unwrap();
        let low = lower_real_step(&pg, &st, &tys);
        if std::env::var("LIST").is_ok() {
            let g = &low.graph;
            for (k, nat) in low
                .program
                .program
                .kernels
                .iter()
                .zip(&low.program.native)
                .take(
                    std::env::var("LIST")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(60),
                )
            {
                let i = k.node.index();
                let what = g.comp_at(i).map_or("-".to_string(), |c| {
                    format!(
                        "red {:?} args {:?} st {:?}",
                        c.red,
                        c.args.iter().map(|a| a.src.index()).collect::<Vec<_>>(),
                        g.nodes()[i].precision.storage
                    )
                });
                eprintln!(
                    "K n{i} {:?} nat {nat} int8 {} {what} writes {:?} | {}",
                    g.nodes()[i].shape,
                    g.nodes()[i].int_mma.is_some(),
                    k.writes.iter().map(|t| t.index()).collect::<Vec<_>>(),
                    k.source.lines().next().unwrap_or("")
                );
            }
        }
        eprintln!(
            "real coder step, GGUF formats: {} kernels, {} native, {} int8 GEMVs, {} Q8x32 copies; \
             f32-uploaded block inputs: {:?}",
            low.program.program.kernels.len(),
            low.program.native.iter().filter(|n| **n).count(),
            low.sched.gemvs,
            low.sched.copies.len(),
            low.f32_inputs
        );
        #[cfg(feature = "cuda")]
        {
            let mut srcs: Vec<&str> = low
                .program
                .program
                .kernels
                .iter()
                .map(|k| k.source.as_str())
                .collect();
            srcs.sort_unstable();
            srcs.dedup();
            for s in &srcs {
                assert!(tang_gpu::cuda::ptx(s, "sm_90").contains(".entry"));
            }
            eprintln!(
                "  nvrtc: {} distinct sources compiled to sm_90 PTX",
                srcs.len()
            );
        }
    }
}
