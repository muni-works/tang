//! Precision search through [`tang_gpu::program::Program`], on whichever
//! backend the [`Runtime`] is: the CUDA counterpart of
//! `ir_msl::tune_precision`, with the same flow.
//!
//! 1. **Tile moves.** For every tensor-core kernel
//!    ([`TensorGraph::cuda_tile_operands`]), its plain operands go to f16
//!    (or bf16) together. That is what puts the kernel on m16n8k16 half
//!    fragments instead of TF32; one operand alone leaves it on the
//!    transpiled fallback.
//! 2. **Per buffer.** Every remaining f32 buffer, largest first.
//!    FP8 formats (`Format::E4M3(block)`, `E5M2(block)`) go to the same
//!    moves: an attention kernel's softmax weights follow ([`apply_for`]).
//!    An FP8 candidate is dropped before it reaches the GPU when no FP8
//!    tile lowers it, or when `eval_emulated` predicts it over budget.
//! 3. Each candidate is run against the f64 reference. It is kept only if
//!    the program stays within the [`ErrorBudget`] and gets `min_gain`
//!    faster than the schedule so far (a move "all tiles" comes first:
//!    single block-sized matmuls gain a few percent each). At the end, the result is re-timed
//!    against the f32 baseline, and the change is dropped if the gain is
//!    under 1%.
//!
//! Accumulators stay f32: the tensor-core kernels accumulate in fp32, and
//! an f16 accumulator would put a kernel on the transpiled path.
//!
//! On CUDA each kernel is costed at its best tile shape (what
//! `Program::tuned` picks), so run the result through `Program::tuned`.
use tang_compile::tensor::{ErrorBudget, ErrorStats, Format, Precision, TensorGraph, TensorId};
use tang_gpu::program::{Program, Runtime};

/// One candidate the search tried.
pub struct PrecisionTrial {
    pub what: String,
    pub error: ErrorStats,
    /// Sum of the program's per-kernel times (ms, fastest single launch
    /// each); `None` when over budget.
    pub ms: Option<f64>,
    /// Time relative to the schedule so far, measured in the same window.
    pub vs: Option<f64>,
    pub kept: bool,
}

pub struct PrecisionTune {
    pub graph: TensorGraph,
    /// Nodes whose precision changed from f32.
    pub assignment: Vec<(TensorId, Precision)>,
    pub error: ErrorStats,
    pub base_error: ErrorStats,
    /// Sums of per-kernel times (ms) of the f32 and chosen schedules.
    pub base_ms: f64,
    pub ms: f64,
    pub trials: Vec<PrecisionTrial>,
}

/// One move: buffers set to one storage format.
#[derive(Clone)]
pub struct Move {
    pub what: String,
    pub buffers: Vec<TensorId>,
}

/// The search's moves on `g`, in order: tile operand groups, then single
/// buffers, largest first. Buffers in `exclude` stay f32 (a training
/// step's weights and optimizer state: fp32 master copies; a half tile
/// converts an f32 operand while staging).
pub fn moves(g: &TensorGraph, outputs: &[TensorId], exclude: &[TensorId]) -> Vec<Move> {
    let mut out: Vec<Move> = g
        .cuda_tile_operands(outputs)
        .into_iter()
        .map(|(k, bufs)| {
            let bufs: Vec<TensorId> = bufs.into_iter().filter(|b| !exclude.contains(b)).collect();
            (k, bufs)
        })
        .filter(|(_, bufs)| !bufs.is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(k, bufs)| Move {
            what: format!(
                "tile {} operands {:?}",
                k.index(),
                bufs.iter().map(|b| b.index()).collect::<Vec<_>>()
            ),
            buffers: bufs,
        })
        .collect();
    // every tile at once first: each block-sized matmul is a few percent
    // of a program, so the tiles' gains only clear the bar together
    if out.len() > 1 {
        let mut all: Vec<TensorId> = out.iter().flat_map(|m| m.buffers.clone()).collect();
        all.sort_unstable_by_key(|t| t.index());
        all.dedup();
        out.insert(
            0,
            Move {
                what: format!("all {} tiles", out.len()),
                buffers: all,
            },
        );
    }
    let mut bufs: Vec<(TensorId, usize)> = g
        .to_cuda(outputs)
        .buffers
        .iter()
        .filter(|b| !exclude.contains(&b.node))
        .map(|b| (b.node, g.buffer_bytes(b.node)))
        .collect();
    bufs.sort_by_key(|b| std::cmp::Reverse(b.1));
    out.extend(bufs.into_iter().map(|(b, _)| Move {
        what: format!("buffer {}", b.index()),
        buffers: vec![b],
    }));
    out
}

/// `g` with `m`'s buffers stored in `f`.
pub fn apply(g: &TensorGraph, m: &Move, f: Format) -> TensorGraph {
    let mut v = g.clone();
    for &b in &m.buffers {
        let mut p = v.precision(b);
        p.storage = f;
        v.set_precision(b, p);
    }
    v
}

/// [`apply`], and for an FP8 move that covers an attention kernel's Q,
/// K and V (in the program for `outputs`), that kernel's softmax weights
/// rounded to E4M3 (the reduction's `operand` format), as its FP8 tile
/// does.
pub fn apply_for(g: &TensorGraph, m: &Move, f: Format, outputs: &[TensorId]) -> TensorGraph {
    let mut v = apply(g, m, f);
    if f.is_fp8() {
        for (red, bufs) in g.cuda_flash_reductions(outputs) {
            if bufs.iter().all(|b| v.precision(*b).storage.is_fp8()) {
                let mut p = v.precision(red);
                p.operand = Format::E4M3(0);
                v.set_precision(red, p);
            }
        }
    }
    v
}

/// GPU error of `p`'s `outputs` against `want`.
pub fn program_error(
    rt: &Runtime,
    p: &Program,
    outputs: &[TensorId],
    want: &[Vec<f64>],
) -> ErrorStats {
    p.run(rt);
    let got: Vec<Vec<f64>> = outputs
        .iter()
        .map(|&o| p.read(rt, o).iter().map(|&v| v as f64).collect())
        .collect();
    ErrorStats::compare(&got, want)
}

/// Every kernel source the search could compile: the baseline, each
/// move on its own and every prefix of the moves taken together, in each
/// of `formats`. Hand them to `tang_gpu::cuda::precompile` before the
/// search (no GPU needed), so that nvrtc does not run while the GPU is
/// held. Sources for other combinations of kept moves still compile
/// during the search.
pub fn candidate_sources(
    g: &TensorGraph,
    outputs: &[TensorId],
    exclude: &[TensorId],
    formats: &[Format],
) -> Vec<String> {
    let ms = moves(g, outputs, exclude);
    let mut graphs = vec![g.clone()];
    for &f in formats {
        let mut acc = g.clone();
        for m in &ms {
            graphs.push(apply(g, m, f));
            acc = apply(&acc, m, f);
            graphs.push(acc.clone());
        }
    }
    let mut out: Vec<String> = graphs
        .iter()
        .flat_map(|v| {
            std::iter::once(None)
                .chain((0..tang_compile::tensor::cuda::MMA_SHAPES.len()).map(Some))
                .flat_map(move |s| {
                    v.to_cuda_variant(outputs, s)
                        .kernels
                        .into_iter()
                        .map(|k| k.source)
                })
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Greedy precision search on `rt` (see the module docs). `formats` are
/// the storage formats tried per move, in order (e.g. `[F16, BF16]`).
/// The fastest admitted format is taken if it beats the current schedule
/// by `min_gain` (a fraction: 0.01 is 1%, about the cost's noise).
#[allow(clippy::too_many_arguments)]
pub fn tune_precision(
    rt: &Runtime,
    g: TensorGraph,
    outputs: &[TensorId],
    inputs: &[&[f32]],
    want: &[Vec<f64>],
    budget: ErrorBudget,
    exclude: &[TensorId],
    formats: &[Format],
    min_gain: f64,
    iters: usize,
) -> PrecisionTune {
    // A time-sliced GPU (another tenant at full load) halves or doubles
    // a whole program's wall time from one batch to the next; a single
    // short launch usually runs between the other tenant's slices. So a
    // program's cost is the sum over its kernels of the fastest of many
    // single launches (`Program::time_kernel`). On CUDA a kernel is
    // costed at its best lowering over every tile shape — what
    // `Program::tuned` will pick; half tiles want other shapes than TF32
    // ones — memoized by the kernel's sources, so a candidate only times
    // the kernels it changes.
    let mut memo: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let mut cost = |g: &TensorGraph| -> f64 {
        use std::hash::{Hash, Hasher};
        let variants: Vec<Option<usize>> = match rt.kind() {
            tang_gpu::program::BackendKind::Cuda => std::iter::once(None)
                .chain((0..tang_compile::tensor::cuda::MMA_SHAPES.len()).map(Some))
                .collect(),
            _ => vec![Some(0)],
        };
        let lower = |v: Option<usize>| match rt.kind() {
            tang_gpu::program::BackendKind::Cuda => g.to_cuda_variant(outputs, v),
            _ => rt.lower(g, outputs),
        };
        let progs: Vec<_> = variants.iter().map(|&v| lower(v)).collect();
        let n = progs[0].kernels.len();
        let keys: Vec<u64> = (0..n)
            .map(|i| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for p in &progs {
                    p.kernels[i].source.hash(&mut h);
                }
                h.finish()
            })
            .collect();
        let missing: Vec<usize> = (0..n).filter(|i| !memo.contains_key(&keys[*i])).collect();
        if !missing.is_empty() {
            let mut best = vec![f64::INFINITY; n];
            for prog in progs {
                let p = Program::new(rt, prog, inputs);
                p.run(rt);
                for &i in &missing {
                    best[i] = best[i].min(p.time_kernel(rt, i, iters));
                }
            }
            for &i in &missing {
                memo.insert(keys[i], best[i]);
            }
        }
        keys.iter().map(|k| memo[k]).sum()
    };
    let base_p = Program::compile(rt, &g, outputs, inputs);
    let base_error = program_error(rt, &base_p, outputs, want);
    drop(base_p);
    let base_ms = cost(&g);
    let base = g.clone();
    let mut cur = g;
    let mut cur_ms = base_ms;
    let mut cur_err = base_error;
    let mut trials = Vec::new();
    for m in moves(&cur, outputs, exclude) {
        if m.buffers
            .iter()
            .all(|&b| cur.precision(b).storage != Format::F32)
        {
            continue;
        }
        // (graph, error, cost, trial index)
        let mut best: Option<(TensorGraph, ErrorStats, f64, usize)> = None;
        for &f in formats {
            let v = apply_for(&cur, &m, f, outputs);
            if f.is_fp8() {
                // FP8 only where an FP8 tile reads it, and only if the
                // emulator predicts the error within budget: no GPU time
                // for a schedule that cannot be accepted
                let skip = if !v.cuda_fp8_lowers(outputs) {
                    Some(("no FP8 tile", ErrorStats::default()))
                } else {
                    let ins: Vec<Vec<f64>> = inputs
                        .iter()
                        .map(|x| x.iter().map(|&y| y as f64).collect())
                        .collect();
                    let pred = ErrorStats::compare(&v.eval_emulated(&ins, outputs), want);
                    (!budget.admits(&pred)).then_some(("predicted over budget", pred))
                };
                if let Some((why, pred)) = skip {
                    trials.push(PrecisionTrial {
                        what: format!("{} {f:?} ({why})", m.what),
                        error: pred,
                        ms: None,
                        vs: None,
                        kept: false,
                    });
                    continue;
                }
            }
            let p = Program::compile(rt, &v, outputs, inputs);
            let err = program_error(rt, &p, outputs, want);
            drop(p);
            let ms = budget.admits(&err).then(|| cost(&v));
            trials.push(PrecisionTrial {
                what: format!("{} {f:?}", m.what),
                error: err,
                ms,
                vs: ms.map(|t| t / cur_ms),
                kept: false,
            });
            if let Some(t) = ms {
                // formats within `min_gain` of each other: the more accurate
                let better = |b: &(TensorGraph, ErrorStats, f64, usize)| {
                    if (t - b.2).abs() <= b.2 * min_gain {
                        err.max_rel < b.1.max_rel
                    } else {
                        t < b.2
                    }
                };
                if best.as_ref().is_none_or(better) {
                    best = Some((v, err, t, trials.len() - 1));
                }
            }
        }
        if let Some((v, err, t, i)) = best {
            // a change must buy `min_gain` to be worth its error
            if t < cur_ms * (1.0 - min_gain) {
                trials[i].kept = true;
                cur = v;
                cur_err = err;
                cur_ms = t;
            }
        }
    }
    let assignment: Vec<(TensorId, Precision)> = (0..cur.nodes().len())
        .map(TensorId::from_index)
        .filter(|&t| cur.precision(t) != Precision::F32)
        .map(|t| (t, cur.precision(t)))
        .collect();
    if !assignment.is_empty() {
        // the memoized costs are fastest launches: a gain under 1% is noise
        if cur_ms >= base_ms * 0.99 {
            return PrecisionTune {
                graph: base,
                assignment: Vec::new(),
                error: base_error,
                base_error,
                base_ms,
                ms: base_ms,
                trials,
            };
        }
    }
    PrecisionTune {
        graph: cur,
        assignment,
        error: cur_err,
        base_error,
        base_ms,
        ms: cur_ms,
        trials,
    }
}

/// A precision assignment as text, one `node storage` per line (for
/// handing a search's result to a later, separately locked run).
pub fn write_assignment(assignment: &[(TensorId, Precision)]) -> String {
    assignment
        .iter()
        .map(|(t, p)| format!("{} {:?}\n", t.index(), p.storage))
        .collect()
}

/// Inverse of [`write_assignment`], applied to `g`.
pub fn read_assignment(g: &mut TensorGraph, text: &str) {
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(n), Some(f)) = (it.next(), it.next()) else {
            continue;
        };
        let t = TensorId::from_index(n.parse().expect("node"));
        let f = match f {
            "F16" => Format::F16,
            "BF16" => Format::BF16,
            _ => Format::F32,
        };
        let mut p = g.precision(t);
        p.storage = f;
        g.set_precision(t, p);
    }
}

/// f64 outputs to a file (a reference computed outside the GPU lock).
pub fn write_f64s(path: &str, v: &[Vec<f64>]) {
    let mut bytes = Vec::new();
    bytes.extend((v.len() as u64).to_le_bytes());
    for o in v {
        bytes.extend((o.len() as u64).to_le_bytes());
        for x in o {
            bytes.extend(x.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// Inverse of [`write_f64s`].
pub fn read_f64s(path: &str) -> Vec<Vec<f64>> {
    let b = std::fs::read(path).unwrap();
    let word = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
    let mut at = 8;
    (0..word(0))
        .map(|_| {
            let n = word(at) as usize;
            at += 8;
            let o = (0..n).map(|i| f64::from_bits(word(at + 8 * i))).collect();
            at += 8 * n;
            o
        })
        .collect()
}
