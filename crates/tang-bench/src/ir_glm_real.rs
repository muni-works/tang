//! GLM-5.3-Flash at real size from its derived graph
//! ([`crate::ir_glm_prefill`]): dims from the GGUF's hparams, every graph
//! input bound to its GGUF tensor (dequantized by tang-llm's reader), and
//! the check against tang-llm's hand engine (`flash::glm5::forward::
//! Glm5Ref::forward`, the CPU reference the GPU engine is gated on) run on
//! the same file.
//!
//! The checkpoint is `unsloth/GLM-5.3-Flash-GGUF` UD-Q4_K_XL (6 shards,
//! 186 GB): Q4_K / Q5_K / Q6_K experts, Q8_0 dense and KDA projections, F32
//! norms, mHC and router bias. `scripts/glm-fetch-layers.py` cuts a few
//! layers into one small, self-consistent GGUF (by HTTP range requests, or
//! from a local copy): checkpoint layer `LAYERS[i]` becomes `blk.{i}` and
//! the per-layer metadata (KDA/DSA kinds, clamps, dense lead-in) is cut to
//! match, so `Hparams::from_gguf` and `Glm5Ref` read it as a short model.
//! The default subset is layers 0 and 3: KDA + dense FFN, then the first
//! DSA layer (with its indexer) + the 288-expert MoE.
//!
//! The reference is the hand engine in f32 (dequantized weights, f32
//! activations, f64 dot accumulations); the graph runs on the f64
//! interpreter. They agree to f32 rounding, not to f64: the bound is
//! relative to the logits' scale.
//!
//! A CPU check can't hold a layer's 288 experts in f64 (58 GB), so the
//! graph is built with [`GlmDims::expert_rows`]: the expert tensors hold
//! only the experts the run routes to, found by evaluating the graph's own
//! `expert_ids` node layer by layer, and a slot map turns an id into its row.
use std::collections::HashMap;
use std::path::Path;

use crate::ir_glm_prefill::{Ffn, GlmDims, Mixer, PrefillInput};
use tang_llm::flash::glm5::Hparams;
use tang_llm::gguf::Gguf;

/// The default subset on mew (`GLM_REAL_GGUF` overrides): checkpoint
/// layers 0 and 3 with embed / head, cut from the local UD-Q4_K_XL copy.
pub const DEFAULT_GGUF: &str = "/home/cam/glm53-l03/glm53-l0_3.gguf";

pub fn gguf_path() -> String {
    std::env::var("GLM_REAL_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.into())
}

/// [`GlmDims`] at real size from `hp`, with a vocab of `vocab` rows (a
/// subset of the head) and `expert_rows` bound expert rows (0: all).
pub fn dims_from_hparams(hp: &Hparams, vocab: usize, expert_rows: usize) -> GlmDims {
    let limit = hp.clamp_exp(0).unwrap_or(f32::MAX) as f64;
    for l in 0..hp.n_layer {
        assert_eq!(
            hp.clamp_exp(l).map(|x| x as f64),
            Some(limit),
            "one swiglu clamp for every layer"
        );
    }
    GlmDims {
        vocab,
        n: hp.n_embd,
        hc: hp.hc,
        sinkhorn: hp.hc_sinkhorn_iters,
        hc_eps: hp.hc_eps as f64,
        eps: hp.eps as f64,
        layers: (0..hp.n_layer)
            .map(|l| {
                (
                    if hp.is_kda[l] { Mixer::Kda } else { Mixer::Dsa },
                    if hp.is_dense(l) { Ffn::Dense } else { Ffn::Moe },
                )
            })
            .collect(),
        heads: hp.n_head,
        dk: hp.d_qk,
        dv: hp.d_v,
        q_lora: hp.q_lora_rank,
        kv_lora: hp.kv_lora_rank,
        idx_heads: hp.idx_heads,
        idx_dim: hp.idx_dim,
        topk: hp.idx_top_k,
        kpool: hp.kpool,
        absorbed: true,
        sparse: true,
        kda_heads: hp.n_head,
        kda_hd: hp.kda_head_dim,
        d_conv: hp.ssm_d_conv,
        gate_lb: hp.kda_gate_lower_bound.expect("safe gate") as f64,
        ff_dense: hp.n_ff,
        n_expert: hp.n_expert,
        top_k: hp.n_expert_used,
        ff_exp: hp.n_ff_exp,
        ff_sh: hp.n_ff_exp * hp.n_expert_shared,
        route_scale: hp.expert_weights_scale as f64,
        limit,
        expert_rows,
    }
}

/// What a real-size CPU run binds: the GGUF subset, its hparams and the
/// vocab rows kept.
pub struct RealRun {
    pub g: Gguf,
    pub hp: Hparams,
    /// Checkpoint layer of each graph layer (for reports).
    pub layer_ids: Vec<usize>,
    /// Token ids of graph vocab rows `0..vocab_ids.len()`.
    pub vocab_ids: Vec<usize>,
}

/// Token ids of the check's prompt (the haiku gate's first output ids),
/// cycled to the length a run needs.
pub const PROMPT: [usize; 8] = [785, 1196, 6801, 264, 6386, 38038, 911, 279];

pub fn prompt(n: usize) -> Vec<usize> {
    PROMPT.iter().copied().cycle().take(n).collect()
}

impl RealRun {
    /// `GLM_REAL_GGUF` (default [`DEFAULT_GGUF`]); `GLM_REAL_LAYERS` names the
    /// checkpoint layers the subset holds (default `0,3`, reports only). The
    /// head rows kept are the first 2048 and the prompt's. `None` without
    /// the file.
    pub fn from_env() -> Option<RealRun> {
        let p = gguf_path();
        if !Path::new(&p).exists() {
            return None;
        }
        let g = Gguf::open(Path::new(&p)).unwrap();
        let hp = Hparams::from_gguf(&g).unwrap();
        let layer_ids: Vec<usize> = std::env::var("GLM_REAL_LAYERS")
            .unwrap_or_else(|_| "0,3".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(
            layer_ids.len(),
            hp.n_layer,
            "GLM_REAL_LAYERS vs block_count"
        );
        let mut vocab_ids: Vec<usize> = (0..2048).collect();
        vocab_ids.extend(PROMPT);
        vocab_ids.sort_unstable();
        vocab_ids.dedup();
        Some(RealRun {
            g,
            hp,
            layer_ids,
            vocab_ids,
        })
    }

    /// Graph vocab rows of token ids.
    pub fn rows_of(&self, ids: &[usize]) -> Vec<f64> {
        ids.iter()
            .map(|id| self.vocab_ids.binary_search(id).expect("a kept row") as f64)
            .collect()
    }

    fn f32s(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.g.info(name).map_err(|e| e.to_string())?;
        self.g.dequantize(t).map_err(|e| format!("{name}: {e}"))
    }

    /// Graph input `name` as f32 from the GGUF (every value exact: the
    /// dequantized weight), for everything but `tokens`, `sample.params`
    /// and compact expert tables / slot maps. Full expert tensors
    /// (`expert_rows = 0`) come whole.
    pub fn input32(&self, name: &str) -> Result<Vec<f32>, String> {
        let rows = |n: &str| -> Result<Vec<f32>, String> {
            let t = self.g.info(n).map_err(|e| e.to_string())?;
            let mut v = Vec::new();
            for &id in &self.vocab_ids {
                v.extend(self.g.rows(t, id, 1).map_err(|e| e.to_string())?);
            }
            Ok(v)
        };
        match name {
            "token_embd" => return rows("token_embd.weight"),
            "output" => return rows("output.weight"),
            "output_norm" => return self.f32s("output_norm.weight"),
            _ => {}
        }
        let rest = name
            .strip_prefix("blk.")
            .ok_or_else(|| format!("{name}: not a layer input"))?;
        let (l, tail) = rest.split_once('.').unwrap();
        let blk = |suffix: &str| self.f32s(&format!("blk.{l}.{suffix}"));
        let stack = |parts: [&str; 3]| -> Result<Vec<f32>, String> {
            let mut v = Vec::new();
            for p in parts {
                v.extend(blk(p)?);
            }
            Ok(v)
        };
        match tail {
            "attn_qkv" => stack(["attn_q.weight", "attn_k.weight", "attn_v.weight"]),
            "ssm_conv1d" => stack([
                "ssm_conv1d_q.weight",
                "ssm_conv1d_k.weight",
                "ssm_conv1d_v.weight",
            ]),
            "ssm_decay_rate" => Ok(blk("ssm_a")?.into_iter().map(|a| -a).collect()),
            "ssm_dt" => blk("ssm_dt.bias"),
            "exp_probs_b" => blk("exp_probs_b.bias"),
            "attn_out" => blk("attn_output.weight"),
            t if t.ends_with(".weight") || t.ends_with(".bias") => blk(t),
            t => blk(&format!("{t}.weight")),
        }
    }

    /// [`Self::input32`] as f64.
    pub fn input(&self, name: &str) -> Result<Vec<f64>, String> {
        Ok(self.input32(name)?.into_iter().map(|x| x as f64).collect())
    }

    /// Every graph input but `tokens`, `sample.params` and the MoE expert
    /// tensors / slot maps, checked against the graph's shapes.
    pub fn dense_inputs(
        &self,
        inputs: &[PrefillInput],
    ) -> Result<HashMap<String, Vec<f64>>, String> {
        let mut out = HashMap::new();
        for i in inputs {
            let n = i.name.as_str();
            if n == "tokens"
                || n == "sample.params"
                || n.ends_with("_exps")
                || n.ends_with(".ffn_expert_slot")
            {
                continue;
            }
            let v = self.input(n)?;
            let want: usize = i.shape.iter().map(|d| d.n()).product();
            if v.len() != want {
                return Err(format!("{n}: {} values, graph {:?}", v.len(), i.shape));
            }
            out.insert(i.name.clone(), v);
        }
        Ok(out)
    }

    /// The compact expert tensors of graph layer `l` holding `used` (in
    /// order, zero rows up to `rows`) and its slot map (id → row; unused
    /// ids read row 0).
    pub fn expert_tables(
        &self,
        l: usize,
        used: &[usize],
        rows: usize,
        d: &GlmDims,
        into: &mut HashMap<String, Vec<f64>>,
    ) -> Result<(), String> {
        let (n, ff) = (d.n, d.ff_exp);
        let per = n * ff;
        let mut slot = vec![0.0; d.n_expert];
        for (name, k) in [
            ("ffn_gate_exps", 0),
            ("ffn_up_exps", 1),
            ("ffn_down_exps", 2),
        ] {
            let t = self
                .g
                .info(&format!("blk.{l}.{name}.weight"))
                .map_err(|e| e.to_string())?;
            let mut w = vec![0.0; rows * per];
            for (r, &e) in used.iter().enumerate() {
                let v = self.g.expert(t, e).map_err(|e| e.to_string())?;
                assert_eq!(v.len(), per);
                for (o, x) in w[r * per..(r + 1) * per].iter_mut().zip(v) {
                    *o = x as f64;
                }
                if k == 0 {
                    slot[e] = r as f64;
                }
            }
            into.insert(format!("blk.{l}.{name}"), w);
        }
        into.insert(format!("blk.{l}.ffn_expert_slot"), slot);
        Ok(())
    }

    /// The experts each MoE layer routes `toks` (graph vocab rows) to,
    /// from the graph's own `blk.{l}.expert_ids`, layer by layer (a layer's
    /// routing needs the earlier layers' experts bound). `w` holds the
    /// dense inputs ([`Self::dense_inputs`]); the expert tables of the
    /// last pass are left in it.
    pub fn route(
        &self,
        toks: &[f64],
        d0: &GlmDims,
        w: &mut HashMap<String, Vec<f64>>,
    ) -> Result<Vec<Vec<usize>>, String> {
        use crate::ir_glm_prefill::{prefill_graph, SEQ};
        let moe: Vec<usize> = (0..d0.layers.len())
            .filter(|&l| d0.layers[l].1 == Ffn::Moe)
            .collect();
        let mut used: Vec<Vec<usize>> = vec![Vec::new(); d0.layers.len()];
        for &l in &moe {
            let rows = used.iter().map(|u| u.len()).max().unwrap().max(1);
            let d = GlmDims {
                expert_rows: rows,
                ..d0.clone()
            };
            let pg = prefill_graph(&d);
            for &m in &moe {
                self.expert_tables(m, &used[m], rows, &d, w)?;
            }
            let node = pg
                .named
                .iter()
                .find(|(n, _)| *n == format!("blk.{l}.expert_ids"))
                .unwrap()
                .1;
            let base: Vec<Vec<f64>> = pg
                .inputs
                .iter()
                .map(|s| {
                    if s.name == "tokens" {
                        toks.to_vec()
                    } else {
                        w[&s.name].clone()
                    }
                })
                .collect();
            let e =
                pg.g.bind(&[(SEQ, toks.len())])
                    .unwrap()
                    .eval(&base, &[node]);
            let mut u: Vec<usize> = e[0].iter().map(|&x| x as usize).collect();
            u.sort_unstable();
            u.dedup();
            used[l] = u;
        }
        Ok(used)
    }

    /// The hand engine on the same file: `Glm5Ref::forward`'s logits over
    /// `ids` (token ids), restricted to the kept vocab rows, as f64.
    pub fn reference(&self, ids: &[usize]) -> Vec<f64> {
        let r = tang_llm::flash::glm5::forward::Glm5Ref::open(Path::new(&gguf_path())).unwrap();
        let toks: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
        let all = r.forward(&toks).unwrap();
        let nv = r.hp.n_vocab;
        let mut out = Vec::with_capacity(ids.len() * self.vocab_ids.len());
        for t in 0..ids.len() {
            out.extend(self.vocab_ids.iter().map(|&v| all[t * nv + v] as f64));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_glm_prefill::{add_sampler, prefill_graph, seed_caches, try_step_sampled, SEQ};
    use crate::ir_sample::{sample_host, SampleSpec};
    use std::time::Instant;
    use tang_compile::tensor::step::StepInput;
    use tang_compile::tensor::TensorId;

    fn open() -> Option<RealRun> {
        let r = RealRun::from_env();
        if r.is_none() {
            eprintln!("no {}", gguf_path());
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

    fn rel(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = (x - y).abs() / scale;
                if d.is_nan() {
                    f64::INFINITY
                } else {
                    d
                }
            })
            .fold(0.0, f64::max)
    }

    /// The real-size graph (real dims, real weights of the subset's layers,
    /// default checkpoint layers 0 and 3) on the f64 interpreter against the
    /// hand engine: prefill logits over `T` tokens (`GLM_REAL_T`, default
    /// 3), then the step derived from the same graph — with the sampler —
    /// seeded from the prefill's caches, for token `T + 1`. The step's
    /// greedy token is its logits' argmax and the reference's; a sampled
    /// token equals the host sampler on the step's logits.
    #[test]
    #[ignore]
    fn real_layers_match_hand_engine() {
        let Some(run) = open() else { return };
        let t_pre: usize = std::env::var("GLM_REAL_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let ids = prompt(t_pre + 1);
        let toks = run.rows_of(&ids);
        let cap = 8.max(t_pre + 1);
        let v = run.vocab_ids.len();
        let d0 = dims_from_hparams(&run.hp, v, 1);
        eprintln!(
            "real dims: checkpoint layers {:?} = {:?}, n {}, hc {}, DSA {} heads x {} (q_lora {}, kv_lora {}), \
             KDA {} heads x {}, experts {} top {} x {}, dense ff {}, vocab rows {} of {}",
            run.layer_ids,
            d0.layers,
            d0.n,
            d0.hc,
            d0.heads,
            d0.dk,
            d0.q_lora,
            d0.kv_lora,
            d0.kda_heads,
            d0.kda_hd,
            d0.n_expert,
            d0.top_k,
            d0.ff_exp,
            d0.ff_dense,
            v,
            run.hp.n_vocab
        );

        // the reference over all T + 1 tokens
        let t0 = Instant::now();
        let want = run.reference(&ids);
        eprintln!(
            "hand engine (Glm5Ref::forward, f32): {} tokens in {:.1?}, peak RSS {:.1} GB",
            ids.len(),
            t0.elapsed(),
            rss_gb()
        );

        // the experts each MoE layer routes to, from the graph itself
        let t0 = Instant::now();
        let pg0 = prefill_graph(&d0);
        let mut w = run.dense_inputs(&pg0.inputs).unwrap();
        let moe: Vec<usize> = (0..d0.layers.len())
            .filter(|&l| d0.layers[l].1 == Ffn::Moe)
            .collect();
        let used = run.route(&toks, &d0, &mut w).unwrap();
        eprintln!(
            "routing (graph's expert_ids): experts per layer {:?} in {:.1?}",
            used.iter().map(|u| u.len()).collect::<Vec<_>>(),
            t0.elapsed()
        );

        // the graph with compact expert tables and the sampler
        let rows = used.iter().map(|u| u.len()).max().unwrap().max(1);
        let d = GlmDims {
            expert_rows: rows,
            ..d0.clone()
        };
        let mut pg = prefill_graph(&d);
        let token = add_sampler(&mut pg, SampleSpec::default());
        for &m in &moe {
            run.expert_tables(m, &used[m], rows, &d, &mut w).unwrap();
        }
        let greedy = [0.0, 1.0, 0.0, 0.0];
        let mut base: Vec<Vec<f64>> = pg
            .inputs
            .iter()
            .map(|s| match s.name.as_str() {
                "tokens" => toks[..t_pre].to_vec(),
                "sample.params" => greedy.to_vec(),
                n => w.remove(n).unwrap_or_else(|| panic!("no input {n}")),
            })
            .collect();
        drop(w);
        eprintln!(
            "graph inputs: {:.2} G values, {:.1?}",
            base.iter().map(|x| x.len()).sum::<usize>() as f64 / 1e9,
            t0.elapsed()
        );
        let argmax = |r: &[f64]| {
            (0..r.len())
                .max_by(|&a, &b| r[a].total_cmp(&r[b]).then(b.cmp(&a)))
                .unwrap()
        };
        let agree = |got: &[f64], want: &[f64]| {
            got.chunks(v)
                .zip(want.chunks(v))
                .filter(|(a, b)| argmax(a) == argmax(b))
                .count()
        };

        let t0 = Instant::now();
        let st = try_step_sampled(&pg, token, 1, cap).unwrap();
        let mut outs = vec![pg.logits, token];
        let cache_nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
        outs.extend(&cache_nodes);
        let got = pg.g.bind(&[(SEQ, t_pre)]).unwrap().eval(&base, &outs);
        let e_pre = rel(&got[0], &want[..t_pre * v]);
        let pre_tok: Vec<usize> = got[1].iter().map(|&x| x as usize).collect();
        for t in 0..t_pre {
            assert_eq!(pre_tok[t], argmax(&got[0][t * v..(t + 1) * v]));
        }
        eprintln!(
            "graph prefill T={t_pre}: max rel err {e_pre:.2e} vs hand engine, top-1 {}/{t_pre}, greedy tokens {pre_tok:?}, \
             {:.1?}, peak RSS {:.1} GB",
            agree(&got[0], &want[..t_pre * v]),
            t0.elapsed(),
            rss_gb()
        );

        let t0 = Instant::now();
        eprintln!(
            "derived step (+ sampler): {} nodes, {} caches: {}",
            st.graph.nodes().len(),
            st.caches.len(),
            st.caches
                .iter()
                .map(|c| format!(
                    "{}{:?}",
                    if c.state { "state" } else { "rows" },
                    st.graph.shape(c.input)
                ))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let caches = seed_caches(&pg, &st, &got[2..], t_pre, cap);
        let params_at = pg.input("sample.params");
        let mut sins: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&si| match si {
                StepInput::Pos => vec![t_pre as f64],
                StepInput::New(_) => toks[t_pre..t_pre + 1].to_vec(),
                StepInput::Same(i) => std::mem::take(&mut base[i]),
                StepInput::Cache(c) => caches[c].clone(),
            })
            .collect();
        let sgot = st.graph.eval(&sins, &st.outputs);
        let e_step = rel(&sgot[0], &want[t_pre * v..]);
        let stok = sgot[1][0] as usize;
        let ref_tok = argmax(&want[t_pre * v..]);
        eprintln!(
            "derived step at position {t_pre}: max rel err {e_step:.2e} vs hand engine, greedy token {} (id {}), \
             hand engine's argmax {} (id {}), {:.1?}, peak RSS {:.1} GB",
            stok,
            run.vocab_ids[stok],
            ref_tok,
            run.vocab_ids[ref_tok],
            t0.elapsed(),
            rss_gb()
        );
        assert_eq!(
            stok,
            argmax(&sgot[0]),
            "greedy token = the step logits' argmax"
        );

        // sampled: the same step with temperature / top-k / top-p / seed
        let k_params = st
            .inputs
            .iter()
            .position(|&si| si == StepInput::Same(params_at))
            .unwrap();
        let mut sampled = Vec::new();
        for params in [
            [0.8, 0.95, 40.0, 1.0],
            [1.0, 0.9, 0.0, 2.0],
            [0.6, 1.0, 8.0, 3.0],
        ] {
            sins[k_params] = params.to_vec();
            let s = st.graph.eval(&sins, &st.outputs);
            let (want_tok, kept) = sample_host(&s[0], params, t_pre as u64, 64.0);
            assert_eq!(s[1][0] as usize, want_tok, "sampled {params:?}");
            sampled.push((params, want_tok, kept));
        }
        eprintln!("sampled step tokens (params, row, kept set) = host sampler's: {sampled:?}");
        assert!(e_pre < 1e-3, "prefill {e_pre:e}");
        assert!(e_step < 1e-3, "step {e_step:e}");
    }
}
