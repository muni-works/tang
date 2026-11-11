//! The four tiny prefill graphs (Qwen3, Qwen3-Coder-Next, DeepSeek V4.1,
//! GLM-5.3) through one `derive_step`: does each derive, its cache set,
//! chained steps against the grown prefill (NaN in unwritten cache rows),
//! and the derived step through the CUDA decode lowering (nvrtc with
//! `--features cuda`; no device). The builders are shared with
//! `examples/derive_tiny_gpu.rs`.

use std::collections::HashMap;

use tang_compile::tensor::step::{
    derive_step_with, seed_cache, CacheSlot, Rows, StepGraph, StepOptions, StepSpec,
};
use tang_compile::tensor::{Dim, TensorGraph, TensorId};

const T: Dim = Dim::Sym("T");
/// Cache capacity: no tiny model has a dim this size, so a reduction over
/// it is a reduction over the cache.
pub const CAP: usize = 21;

pub struct Model {
    pub name: &'static str,
    pub g: TensorGraph,
    pub inputs: Vec<(String, Vec<Dim>)>,
    pub logits: TensorId,
    pub named: Vec<(String, TensorId)>,
    pub cap: usize,
}

/// The four tiny models, by name.
pub fn all_models() -> Vec<Model> {
    vec![qwen(), coder(), deepseek(), glm()]
}

pub fn qwen() -> Model {
    use crate::ir_qwen::prefill::prefill_graph;
    let dims = crate::ir_qwen::QwenDims {
        d_model: 128,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        ff: 192,
        vocab: 256,
        layers: 2,
        ctx: 16,
        eps: 1e-6,
        theta: 1e4,
        kv_f16: false,
        vocab_chunks: 1,
        prefill_q4: false,
    };
    let pg = prefill_graph(&dims);
    let named =
        pg.k.iter()
            .enumerate()
            .map(|(l, &t)| (format!("blk.{l}.k"), t))
            .chain(
                pg.v.iter()
                    .enumerate()
                    .map(|(l, &t)| (format!("blk.{l}.v"), t)),
            )
            .collect();
    Model {
        name: "qwen3",
        inputs: pg
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.shape.clone()))
            .collect(),
        logits: pg.logits,
        g: pg.g,
        named,
        cap: CAP,
    }
}

pub fn coder() -> Model {
    use crate::ir_coder_prefill::*;
    let pg = prefill_graph(&CoderDims::tiny());
    let named =
        pg.k.iter()
            .enumerate()
            .map(|(l, &t)| (format!("attn.{l}.k"), t))
            .chain(
                pg.v.iter()
                    .enumerate()
                    .map(|(l, &t)| (format!("attn.{l}.v"), t)),
            )
            .chain(
                pg.gdn_state
                    .iter()
                    .enumerate()
                    .map(|(l, &t)| (format!("gdn.{l}.state"), t)),
            )
            .collect();
    Model {
        name: "coder-next",
        inputs: pg
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.shape.clone()))
            .collect(),
        logits: pg.logits,
        g: pg.g,
        named,
        cap: CAP,
    }
}

pub fn deepseek() -> Model {
    use crate::ir_ds_prefill::*;
    let dm = DsDims::tiny();
    let pg = prefill_graph(&dm);
    Model {
        name: "deepseek-v4.1",
        inputs: pg
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.shape.clone()))
            .collect(),
        logits: pg.logits,
        named: pg.named.clone(),
        g: pg.g,
        cap: CAP,
    }
}

pub fn glm() -> Model {
    use crate::ir_glm_prefill::*;
    let pg = prefill_graph(&GlmDims::tiny());
    Model {
        name: "glm-5.3",
        inputs: pg
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.shape.clone()))
            .collect(),
        logits: pg.logits,
        named: pg.named.clone(),
        g: pg.g,
        cap: CAP,
    }
}

pub fn vocab(m: &Model) -> usize {
    m.g.shape(m.logits).last().unwrap().n()
}

/// Deterministic weights by name, scaled so activations stay O(1).
pub fn weights(m: &Model) -> HashMap<String, Vec<f64>> {
    m.inputs
        .iter()
        .filter(|(n, _)| n != "tokens")
        .map(|(name, shape)| {
            let n: usize = shape.iter().map(|d| d.n()).product();
            let mut s = name.bytes().fold(1469598103934665603u64, |h, b| {
                (h ^ b as u64).wrapping_mul(1099511628211)
            });
            let cols = shape.last().map_or(1, |d| d.n()) as f64;
            let v = (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (s >> 11) as f64 / (1u64 << 53) as f64;
                    if name.contains("norm") || name.ends_with("decay_rate") {
                        0.8 + 0.4 * r
                    } else if name.ends_with("freqs") {
                        0.3 * r
                    } else {
                        1.5 * (2.0 * r - 1.0) / cols.sqrt()
                    }
                })
                .collect();
            (name.clone(), v)
        })
        .collect()
}

pub fn concrete(shape: &[Dim], t: usize) -> Vec<usize> {
    shape
        .iter()
        .map(|&d| if d == T { t } else { d.n() })
        .collect()
}

pub fn name_of(m: &Model, n: TensorId) -> String {
    m.named
        .iter()
        .find(|(_, t)| *t == n)
        .map_or(format!("{n:?}"), |(s, _)| s.clone())
}

pub fn spec(m: &Model, k: usize) -> StepSpec {
    StepSpec {
        seq: "T",
        new: k,
        capacity: m.cap,
    }
}

pub fn derive(m: &Model, k: usize, rows: Rows) -> Result<StepGraph, String> {
    derive_step_with(
        &m.g,
        &[m.logits],
        &spec(m, k),
        &StepOptions { rows, batch: None },
    )
    .map_err(|e| e.0)
}

/// The cache set: name, kind, step buffer shape, floats vs all-full.
pub fn report(m: &Model, st: &StepGraph) -> Vec<String> {
    st.caches
        .iter()
        .map(|c| {
            let sh: Vec<usize> = st.graph.shape(c.input).iter().map(|d| d.n()).collect();
            let kind = match (c.state, c.ring) {
                (true, _) => "state".to_string(),
                (false, Some(r)) => format!("ring {r}"),
                (false, None) => "rows".to_string(),
            };
            format!("{} ({kind}) {sh:?}", name_of(m, c.node))
        })
        .collect()
}

/// The step's cache buffer for slot `c` after a prefill of `len` rows.
pub fn seed(m: &Model, c: &CacheSlot, v: Vec<f64>, len: usize) -> Vec<f64> {
    let sh = concrete(m.g.shape(c.node), len);
    if !c.state {
        return seed_cache(c, &sh, &v, len, m.cap);
    }
    if !m.g.shape(c.node).contains(&T) {
        return v;
    }
    let outer: usize = sh[..c.axis].iter().product();
    let inner: usize = sh[c.axis + 1..].iter().product();
    (0..outer)
        .flat_map(|o| {
            let b = (o * len + len - 1) * inner;
            v[b..b + inner].to_vec()
        })
        .collect()
}

/// Worst `|a − b| / max|b|`; NaN or ∞ anywhere is ∞.
pub fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b
        .iter()
        .filter(|x| x.is_finite())
        .fold(1e-300f64, |s, x| s.max(x.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let e = (x - y).abs() / scale;
            if e.is_finite() {
                e
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_compile::tensor::cuda_gemv::DecodeOpts;
    use tang_compile::tensor::step::StepInput;

    /// Chained steps from `p0` against the grown prefill: the new rows'
    /// logits (or the last row's), every written cache row, every state.
    /// Returns the worst error and how many unwritten (NaN) cache rows the
    /// steps ran with.
    fn chain(m: &Model, p0: usize, ks: &[usize], rows: Rows) -> (f64, usize) {
        let w = weights(m);
        let total: usize = ks.iter().sum();
        let toks: Vec<f64> = (0..p0 + total)
            .map(|i| ((37 * i + 5) % vocab(m)) as f64)
            .collect();
        let ins = |t: &[f64]| -> Vec<Vec<f64>> {
            m.inputs
                .iter()
                .map(|(n, _)| {
                    if n == "tokens" {
                        t.to_vec()
                    } else {
                        w[n].clone()
                    }
                })
                .collect()
        };
        let first = derive(m, ks[0], rows).unwrap();
        let nodes: Vec<TensorId> = first.caches.iter().map(|c| c.node).collect();
        let fg = m.g.bind(&[("T", p0)]).unwrap();
        let mut caches: Vec<Vec<f64>> = first
            .caches
            .iter()
            .zip(fg.eval(&ins(&toks[..p0]), &nodes))
            .map(|(c, v)| seed(m, c, v, p0))
            .collect();
        let nan_rows = caches
            .iter()
            .zip(&first.caches)
            .filter(|(_, c)| !c.state)
            .map(|(b, c)| {
                let sh = concrete(m.g.shape(c.node), 1);
                let inner: usize = sh[c.axis + 1..].iter().product();
                b.chunks(inner.max(1))
                    .filter(|r| r.iter().any(|x| x.is_nan()))
                    .count()
            })
            .sum();
        let vocab = vocab(m);
        let mut worst = 0.0f64;
        let mut pos = p0;
        for &k in ks {
            let st = derive(m, k, rows).unwrap();
            let si: Vec<Vec<f64>> = st
                .inputs
                .iter()
                .map(|&x| match x {
                    StepInput::Pos => vec![pos as f64],
                    StepInput::New(_) => toks[pos..pos + k].to_vec(),
                    StepInput::Same(i) => w[&m.inputs[i].0].clone(),
                    StepInput::Cache(c) => caches[c].clone(),
                })
                .collect();
            let mut ids = st.outputs.clone();
            ids.extend(st.caches.iter().map(|c| c.updated));
            let got = st.graph.eval(&si, &ids);
            assert!(got[0].iter().all(|x| x.is_finite()), "{}: logits", m.name);
            let grown = pos + k;
            let fg = m.g.bind(&[("T", grown)]).unwrap();
            let mut rids = vec![m.logits];
            rids.extend(st.caches.iter().map(|c| c.node));
            let want = fg.eval(&ins(&toks[..grown]), &rids);
            let r0 = if rows == Rows::Last { grown - 1 } else { pos };
            worst = worst.max(rel(&got[0], &want[0][r0 * vocab..grown * vocab]));
            for (ci, c) in st.caches.iter().enumerate() {
                let (gv, wv) = (&got[1 + ci], &want[1 + ci]);
                let wsh = concrete(m.g.shape(c.node), grown);
                if c.state {
                    let e = rel(gv, &seed(m, c, wv.clone(), grown));
                    worst = worst.max(e);
                    continue;
                }
                // every written row of the buffer against its position
                let ax = c.axis;
                let outer: usize = wsh[..ax].iter().product();
                let inner: usize = wsh[ax + 1..].iter().product();
                let nrows = c.rows(m.cap);
                let (mut g_rows, mut w_rows) = (Vec::new(), Vec::new());
                for o in 0..outer {
                    for r in 0..nrows {
                        if let Some(p) = c.position(r, grown) {
                            let gb = (o * nrows + r) * inner;
                            let wb = (o * grown + p) * inner;
                            g_rows.extend_from_slice(&gv[gb..gb + inner]);
                            w_rows.extend_from_slice(&wv[wb..wb + inner]);
                        }
                    }
                }
                worst = worst.max(rel(&g_rows, &w_rows));
            }
            caches = got[1..].to_vec();
            pos = grown;
        }
        (worst, nan_rows)
    }

    const CHAINS: [&[usize]; 3] = [&[1], &[4], &[4, 1, 1, 2]];

    fn check(m: &Model, rows_kinds: &[Rows]) {
        let st = match derive(m, 1, Rows::All) {
            Ok(st) => st,
            Err(e) => panic!("{}: refused: {e}", m.name),
        };
        eprintln!("{}: derives; caches:", m.name);
        for line in report(m, &st) {
            eprintln!("  {line}");
        }
        for &rows in rows_kinds {
            for ks in CHAINS {
                let (e, nan) = chain(m, 5, ks, rows);
                eprintln!(
                    "  {rows:?} steps {ks:?} from 5: max rel err {e:.2e} ({nan} unwritten NaN cache rows)"
                );
                assert!(e < 1e-10, "{} {rows:?} {ks:?}: {e:e}", m.name);
            }
        }
        lower(m, &st);
    }

    /// The k = 1 step through the CUDA decode lowering (maps fused, common
    /// terms shared, in-place cache updates), with every kernel through
    /// nvrtc under `--features cuda`. Reductions over the whole capacity are
    /// counted: dense attention over its cache (runtime-bounded by the
    /// mask) is expected; anything else is reported.
    fn lower(m: &Model, st: &StepGraph) {
        let mut outs = st.outputs.clone();
        outs.extend(st.caches.iter().map(|c| c.updated));
        let (g, outs) = st.graph.substitute_maps(&outs);
        let (g, outs) = g.eliminate_common(&outs);
        let input_of = |t: TensorId| match st.graph.nodes()[t.index()].op {
            tang_compile::tensor::Op::Input(i) => i,
            _ => unreachable!(),
        };
        let opts = DecodeOpts {
            // full-capacity row caches update in place (a predicated
            // map); rings shift and states are rewritten whole, so those
            // go to a fresh buffer (the runner swaps it in)
            inplace: st
                .caches
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.state && s.ring.is_none())
                .map(|(c, s)| (outs[st.outputs.len() + c], input_of(s.input)))
                .collect(),
            ..Default::default()
        };
        let dp = g.to_cuda_decode(&outs, &opts);
        let ks = &dp.program.kernels;
        let native = dp.native.iter().filter(|&&n| n).count();
        let cap_red: Vec<String> = ks
            .iter()
            .filter_map(|k| {
                let c = g.comp_at(k.node.index())?;
                c.red.iter().any(|d| d.n() == m.cap).then(|| {
                    format!(
                        "{:?} out {:?} red {:?} monoid {:?}",
                        k.node,
                        g.nodes()[k.node.index()].shape,
                        c.red,
                        c.monoid.as_ref().map(|m| m.kind())
                    )
                })
            })
            .collect();
        eprintln!(
            "  CUDA (k = 1): {} kernels, {native} native; {} reduce over the capacity ({})",
            ks.len(),
            cap_red.len(),
            m.cap
        );
        for r in &cap_red {
            eprintln!("    over capacity: {r}");
        }
        #[cfg(feature = "cuda")]
        {
            let mut srcs: Vec<&str> = ks.iter().map(|k| k.source.as_str()).collect();
            srcs.sort_unstable();
            srcs.dedup();
            for s in &srcs {
                assert!(tang_gpu::cuda::ptx(s, "sm_86").contains(".entry"));
            }
            eprintln!(
                "  nvrtc: {} distinct sources compiled to sm_86 PTX",
                srcs.len()
            );
        }
    }

    #[test]
    fn qwen3_derives() {
        check(&qwen(), &[Rows::All, Rows::Last]);
    }

    #[test]
    fn coder_next_derives() {
        check(&coder(), &[Rows::All, Rows::Last]);
    }

    #[test]
    fn deepseek_derives() {
        check(&deepseek(), &[Rows::All, Rows::Last]);
    }

    #[test]
    fn glm_derives() {
        check(&glm(), &[Rows::All, Rows::Last]);
    }
}
