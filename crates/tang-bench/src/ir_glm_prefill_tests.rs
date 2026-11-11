//! The GLM prefill graph against tang-llm's hand CPU reference
//! (`flash::glm5::{mhc, mla, kda, moe}`, composed as `Glm5Ref::forward`
//! composes them), and what `derive_step` makes of it.
use super::*;
use std::collections::HashMap;
use tang_compile::tensor::step::StepInput;
use tang_llm::flash::glm5::{kda as hk, mhc as hm, mla as hl, moe as hf};

type Weights = HashMap<String, Vec<f32>>;

/// Deterministic f32 weights per name (the graph gets the same values as f64).
fn weights(pg: &PrefillGraph) -> Weights {
    pg.inputs
        .iter()
        .filter(|s| s.name != "tokens")
        .map(|s| {
            let n: usize = s.shape.iter().map(|d| d.n()).product();
            let mut seed = s.name.bytes().fold(1469598103934665603u64, |h, b| {
                (h ^ b as u64).wrapping_mul(1099511628211)
            });
            let mut rnd = move || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            let nm = s.name.as_str();
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    let r = rnd();
                    (if nm.ends_with("norm") || nm.ends_with("k_norm.weight") {
                        0.8 + 0.4 * r
                    } else if nm.ends_with("decay_rate") {
                        0.5 + r
                    } else if nm == "token_embd" {
                        2.0 * r - 1.0
                    } else {
                        r - 0.5
                    }) as f32
                })
                .collect();
            (s.name.clone(), v)
        })
        .collect()
}

fn tokens(t: usize, vocab: usize) -> Vec<usize> {
    (0..t).map(|i| (7 * i + 3) % vocab).collect()
}

fn inputs(pg: &PrefillGraph, w: &Weights, toks: &[usize]) -> Vec<Vec<f64>> {
    pg.inputs
        .iter()
        .map(|s| {
            if s.name == "tokens" {
                toks.iter().map(|&t| t as f64).collect()
            } else {
                w[&s.name].iter().map(|&x| x as f64).collect()
            }
        })
        .collect()
}

fn graph_logits(pg: &PrefillGraph, w: &Weights, toks: &[usize]) -> Vec<f64> {
    let g = pg.g.bind(&[(SEQ, toks.len())]).unwrap();
    g.eval(&inputs(pg, w, toks), &[pg.logits]).remove(0)
}

/// `inv_rms` / `rms_norm_mul` of `flash::reference` (crate-private there).
fn rms_rows(x: &mut [f32], w: &[f32], eps: f32) {
    for row in x.chunks_mut(w.len()) {
        let s: f64 = row.iter().map(|&v| v as f64 * v as f64).sum();
        let inv = 1.0 / ((s / row.len() as f64) as f32 + eps).sqrt();
        for (v, g) in row.iter_mut().zip(w) {
            *v = *v * inv * g;
        }
    }
}

fn matvec_rows(w: &[f32], rows: usize, x: &[f32]) -> Vec<f32> {
    let cols = w.len() / rows;
    x.chunks(cols)
        .flat_map(|xr| {
            (0..rows).map(move |r| {
                w[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(xr)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32
            })
        })
        .collect()
}

/// The hand path: `Glm5Ref::forward`'s layer loop on the same weights, every
/// mixer/FFN/mHC step a tang-llm call. `dense` forces dense causal
/// attention (no indexer) at every length: the negative control.
fn reference(d: &GlmDims, w: &Weights, toks: &[usize], dense: bool) -> Vec<f32> {
    let (n, hc, t_len) = (d.n, d.hc, toks.len());
    let eps = d.eps as f32;
    let get = |s: String| -> Vec<f32> { w.get(&s).unwrap_or_else(|| panic!("{s}")).clone() };
    let embd = get("token_embd".into());
    let mut r = vec![0f32; t_len * hc * n];
    for (t, &tok) in toks.iter().enumerate() {
        for c in 0..hc {
            r[(t * hc + c) * n..(t * hc + c + 1) * n]
                .copy_from_slice(&embd[tok * n..(tok + 1) * n]);
        }
    }
    let mhp = hm::MhcHparams {
        streams: hc,
        sinkhorn_iters: d.sinkhorn as u32,
        hc_eps: d.hc_eps as f32,
        rms_eps: eps,
    };
    let block = |r: &mut Vec<f32>, pfx: &str, norm: &str, f: &dyn Fn(&[f32]) -> Vec<f32>| {
        let mw = hm::MhcWeights::from_slices(
            get(format!("{pfx}_fn")),
            get(format!("{pfx}_base")),
            get(format!("{pfx}_scale")),
            n,
            &mhp,
        )
        .unwrap();
        let mut mixes = Vec::new();
        let mut x = vec![0f32; t_len * n];
        for t in 0..t_len {
            let s: Vec<&[f32]> = (0..hc)
                .map(|c| &r[(t * hc + c) * n..(t * hc + c + 1) * n])
                .collect();
            let mix = hm::mhc_pre(&mw, &s).unwrap();
            x[t * n..(t + 1) * n].copy_from_slice(&mix.x);
            mixes.push(mix);
        }
        rms_rows(&mut x, &get(norm.to_string()), eps);
        let y = f(&x);
        let r_in = r.clone();
        for t in 0..t_len {
            let s: Vec<&[f32]> = (0..hc)
                .map(|c| &r_in[(t * hc + c) * n..(t * hc + c + 1) * n])
                .collect();
            let token = &mut r[t * hc * n..(t + 1) * hc * n];
            let mut outs: Vec<&mut [f32]> = token.chunks_mut(n).collect();
            hm::mhc_post(&mixes[t], &y[t * n..(t + 1) * n], &s, &mut outs).unwrap();
        }
    };
    for (l, &(mixer, ffn)) in d.layers.iter().enumerate() {
        let p = format!("blk.{l}");
        let g = |s: &str| get(format!("{p}.{s}"));
        let attn = |x: &[f32]| -> Vec<f32> {
            match mixer {
                Mixer::Dsa => {
                    let mw = hl::MlaWeights::new(
                        n,
                        d.q_lora,
                        d.kv_lora,
                        d.heads,
                        d.dk,
                        d.dv,
                        d.idx_heads,
                        d.idx_dim,
                        d.topk,
                        d.kpool,
                        true,
                        eps,
                        g("attn_q_a"),
                        g("attn_q_a_norm"),
                        g("attn_q_b"),
                        g("attn_kv_a_mqa"),
                        g("attn_kv_a_norm"),
                        g("attn_k_b"),
                        g("attn_v_b"),
                        g("attn_output"),
                        g("indexer.attn_k"),
                        g("indexer.k_norm.weight"),
                        g("indexer.k_norm.bias"),
                        g("indexer.attn_q_b"),
                        g("indexer.proj"),
                        g("indexer_compressor_gate"),
                        g("indexer_compressor_ape"),
                    );
                    let q_resid = mw.gemv_q_a(x, t_len);
                    let k_latent = mw.gemv_kv_a(x, t_len);
                    let idd = d.idx_dim;
                    let mut ck = vec![0f32; t_len * idd];
                    let mut cg = vec![0f32; t_len * idd];
                    for t in 0..t_len {
                        hl::indexer_key(
                            &mw,
                            &x[t * n..(t + 1) * n],
                            &mut ck[t * idd..(t + 1) * idd],
                            &mut cg[t * idd..(t + 1) * idd],
                        );
                    }
                    let sel = (!dense && !hl::indexer_is_dense(t_len, d.topk)).then(|| {
                        hl::indexer_select(
                            &mw,
                            x,
                            &q_resid,
                            &ck,
                            &cg,
                            &vec![1u8; t_len],
                            &vec![1u8; t_len],
                            t_len,
                            t_len,
                            0,
                        )
                    });
                    let mut out = vec![0f32; t_len * n];
                    hl::mla_forward(
                        &mw,
                        x,
                        &q_resid,
                        &k_latent,
                        t_len,
                        t_len,
                        0,
                        None,
                        sel.as_deref(),
                        &mut out,
                    );
                    out
                }
                Mixer::Kda => {
                    let (h, hd) = (d.kda_heads, d.kda_hd);
                    let qkv = g("attn_qkv");
                    let m = h * hd * n;
                    let kw = hk::KdaLayerW {
                        q: qkv[..m].to_vec(),
                        k: qkv[m..2 * m].to_vec(),
                        v: qkv[2 * m..].to_vec(),
                        conv: g("ssm_conv1d"),
                        f_a: g("ssm_f_a"),
                        f_b: g("ssm_f_b"),
                        dt_bias: g("ssm_dt"),
                        decay_rate: g("ssm_decay_rate"),
                        b_proj: g("ssm_beta"),
                        g_a: g("ssm_g_a"),
                        g_b: g("ssm_g_b"),
                        o_norm: g("ssm_norm"),
                        o_proj: g("attn_out"),
                        hidden: n,
                    };
                    let cfg = hk::KdaConfig {
                        heads: h,
                        head_dim: hd,
                        d_conv: d.d_conv,
                        gate_lower_bound: Some(d.gate_lb as f32),
                        eps,
                    };
                    let mut conv = hk::ConvState::zeros(&cfg, 1);
                    let mut state = hk::KdaState::zeros(&cfg, 1);
                    hk::kda_layer(x, &kw, &cfg, &mut conv, &mut state, 1, t_len).unwrap()
                }
            }
        };
        block(
            &mut r,
            &format!("{p}.hc_attn"),
            &format!("{p}.attn_norm"),
            &attn,
        );
        let (router, bias, ge, ue, de, sg, su, sd, dg, du, dd) = match ffn {
            Ffn::Moe => (
                g("ffn_gate_inp"),
                g("exp_probs_b"),
                g("ffn_gate_exps"),
                g("ffn_up_exps"),
                g("ffn_down_exps"),
                g("ffn_gate_shexp"),
                g("ffn_up_shexp"),
                g("ffn_down_shexp"),
                vec![],
                vec![],
                vec![],
            ),
            Ffn::Dense => (
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                g("ffn_gate"),
                g("ffn_up"),
                g("ffn_down"),
            ),
        };
        let mp = hf::MoeParams {
            hidden: n,
            n_expert: d.n_expert,
            top_k: d.top_k,
            n_group: 1,
            topk_group: 1,
            ff_exp: d.ff_exp,
            ff_shexp: d.ff_sh,
            ff_dense: d.ff_dense,
            scale: d.route_scale as f32,
            norm_topk_prob: true,
            limit_exp: d.limit as f32,
            limit_shexp: d.limit as f32,
            router: some(&router),
            bias: some(&bias),
            gate_exps: some(&ge),
            up_exps: some(&ue),
            down_exps: some(&de),
            sh_gate: some(&sg),
            sh_up: some(&su),
            sh_down: some(&sd),
            dense_gate: some(&dg),
            dense_up: some(&du),
            dense_down: some(&dd),
        };
        let ffn_f = |x: &[f32]| hf::ffn(&mp, x, t_len).unwrap();
        block(
            &mut r,
            &format!("{p}.hc_ffn"),
            &format!("{p}.ffn_norm"),
            &ffn_f,
        );
    }
    let mut h = vec![0f32; t_len * n];
    for t in 0..t_len {
        for c in 0..hc {
            for i in 0..n {
                h[t * n + i] += r[(t * hc + c) * n + i];
            }
        }
        for v in &mut h[t * n..(t + 1) * n] {
            *v /= hc as f32;
        }
    }
    rms_rows(&mut h, &get("output_norm".into()), eps);
    matvec_rows(&get("output".into()), d.vocab, &h)
}

fn some(v: &[f32]) -> Option<hf::W<'_>> {
    (!v.is_empty()).then_some(hf::W::F32(v))
}

/// Worst `|a − b| / max|b|`.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if (x - y).is_finite() {
                (x - y).abs() / scale
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

fn vs_reference(d: &GlmDims, lens: &[usize], dense_ref: bool) -> Vec<(usize, f64)> {
    let pg = prefill_graph(d);
    // weights of the sparse graph: a superset (the hand MLA always takes the indexer's)
    let w = weights(&prefill_graph(&GlmDims {
        sparse: true,
        ..d.clone()
    }));
    lens.iter()
        .map(|&t| {
            let toks = tokens(t, d.vocab);
            let got = graph_logits(&pg, &w, &toks);
            let want: Vec<f64> = reference(d, &w, &toks, dense_ref)
                .iter()
                .map(|&x| x as f64)
                .collect();
            (t, rel(&got, &want))
        })
        .collect()
}

const TOL: f64 = 2e-5;
const LENS: [usize; 6] = [1, 2, 4, 5, 7, 9];

/// The whole tiny model (DSA + dense, KDA + MoE, DSA + MoE; sparse indexer,
/// absorbed MLA) against the hand reference: `T ≤ 4` dense, `T ≥ 5` sparse.
#[test]
fn glm_prefill_matches_hand_reference() {
    let d = GlmDims::tiny();
    let errs = vs_reference(&d, &LENS, false);
    eprintln!("glm tiny (absorbed, sparse) vs hand f32 reference: {errs:?}");
    for (t, e) in errs {
        assert!(e < TOL, "T={t}: {e:e}");
    }
}

/// The expanded (per-head K/V) form is the same function.
#[test]
fn glm_prefill_expanded_matches_hand_reference() {
    let d = GlmDims {
        absorbed: false,
        ..GlmDims::tiny()
    };
    let errs = vs_reference(&d, &[3, 7], false);
    eprintln!("glm tiny (expanded, sparse) vs hand f32 reference: {errs:?}");
    for (t, e) in errs {
        assert!(e < TOL, "T={t}: {e:e}");
    }
}

/// Without the indexer the graph is the hand path's dense attention at
/// every length.
#[test]
fn glm_prefill_dense_matches_dense_reference() {
    let d = GlmDims {
        sparse: false,
        ..GlmDims::tiny()
    };
    let errs = vs_reference(&d, &[3, 9], true);
    eprintln!("glm tiny (absorbed, dense) vs hand dense reference: {errs:?}");
    for (t, e) in errs {
        assert!(e < TOL, "T={t}: {e:e}");
    }
}

/// Negative control: past `topk` the sparse graph must NOT match a
/// reference that skips the indexer (and must still match below it).
#[test]
fn glm_prefill_negative_control() {
    let d = GlmDims::tiny();
    let errs = vs_reference(&d, &[4, 9], true);
    eprintln!("negative control (sparse graph vs dense reference): {errs:?}");
    assert!(errs[0].1 < TOL, "T=4 is dense either way: {:e}", errs[0].1);
    assert!(
        errs[1].1 > 1e-2,
        "T=9: the indexer changed nothing ({:e})",
        errs[1].1
    );
}

fn describe(pg: &PrefillGraph, r: &Result<StepGraph, String>) -> String {
    match r {
        Err(e) => {
            // name the refused node: its op and shape, and the nearest named node after it
            let num: Option<usize> = e
                .split("node ")
                .nth(1)
                .and_then(|r| r.split(' ').next())
                .and_then(|x| x.parse().ok());
            let at = num.map(|i| {
                let nd = &pg.g.nodes()[i];
                let next = pg
                    .named
                    .iter()
                    .filter(|(_, t)| t.index() >= i)
                    .min_by_key(|(_, t)| t.index())
                    .map(|(s, t)| format!("{s} (node {})", t.index()));
                format!(
                    " [node {i}: {:?}, op {:?}; next named {next:?}]",
                    nd.shape,
                    std::mem::discriminant(&nd.op)
                )
            });
            format!("refused: {e}{}", at.unwrap_or_default())
        }
        Ok(s) => {
            let caches: Vec<String> = s
                .caches
                .iter()
                .map(|c| {
                    format!(
                        "{} {:?}",
                        pg.name_of(c.node).unwrap_or("<unnamed>"),
                        pg.g.shape(c.node)
                    )
                })
                .collect();
            format!("derived; caches: {caches:?}")
        }
    }
}

/// Prefill to `p0`, then derived steps of `k` against prefills of the grown
/// sequence (new rows' logits and every filled cache row).
fn step_matches_prefill(d: &GlmDims, p0: usize, ks: &[usize], cap: usize) -> f64 {
    let pg = prefill_graph(d);
    let w = weights(&pg);
    let all = tokens(cap, d.vocab);
    let shape_at = |t: TensorId, rows: usize| -> Vec<usize> {
        pg.g.shape(t)
            .iter()
            .map(|&x| if x == T { rows } else { x.n() })
            .collect()
    };
    let rows_of = |buf: &[f64], shape: &[usize], axis: usize, r0: usize, r1: usize| {
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out = Vec::new();
        for o in 0..outer {
            for r in r0..r1 {
                let base = (o * shape[axis] + r) * inner;
                out.extend_from_slice(&buf[base..base + inner]);
            }
        }
        out
    };
    // caches by prefill node, NaN past the filled rows
    let mut caches: HashMap<TensorId, Vec<f64>> = HashMap::new();
    let first = try_step(&pg, ks[0], cap).unwrap();
    {
        let g = pg.g.bind(&[(SEQ, p0)]).unwrap();
        let nodes: Vec<TensorId> = first.caches.iter().map(|c| c.node).collect();
        let vals = g.eval(&inputs(&pg, &w, &all[..p0]), &nodes);
        for (c, v) in first.caches.iter().zip(vals) {
            let shape = shape_at(c.node, p0);
            let ax = c.axis;
            let outer: usize = shape[..ax].iter().product();
            let inner: usize = shape[ax + 1..].iter().product();
            let mut buf = vec![f64::NAN; outer * cap * inner];
            for o in 0..outer {
                for r in 0..p0 {
                    let s = (o * p0 + r) * inner;
                    let dd = (o * cap + r) * inner;
                    buf[dd..dd + inner].copy_from_slice(&v[s..s + inner]);
                }
            }
            caches.insert(c.node, buf);
        }
    }
    let mut worst = 0.0f64;
    let mut p0 = p0;
    for &k in ks {
        let step = try_step(&pg, k, cap).unwrap();
        let ins: Vec<Vec<f64>> = step
            .inputs
            .iter()
            .map(|&si| match si {
                StepInput::Pos => vec![p0 as f64],
                StepInput::New(_) => all[p0..p0 + k].iter().map(|&t| t as f64).collect(),
                StepInput::Same(i) => w[&pg.inputs[i].name].iter().map(|&x| x as f64).collect(),
                StepInput::Cache(c) => caches[&step.caches[c].node].clone(),
            })
            .collect();
        let mut ids = step.outputs.clone();
        ids.extend(step.caches.iter().map(|c| c.updated));
        let got = step.graph.eval(&ins, &ids);
        assert!(
            got[0].iter().all(|x| x.is_finite()),
            "step logits not finite"
        );
        let grown = p0 + k;
        let fg = pg.g.bind(&[(SEQ, grown)]).unwrap();
        let mut rids = vec![pg.logits];
        rids.extend(step.caches.iter().map(|c| c.node));
        let want = fg.eval(&inputs(&pg, &w, &all[..grown]), &rids);
        let lw = rows_of(&want[0], &shape_at(pg.logits, grown), 0, p0, grown);
        worst = worst.max(rel(&got[0], &lw));
        for (c, slot) in step.caches.iter().enumerate() {
            let gr = rows_of(&got[1 + c], &shape_at(slot.node, cap), slot.axis, 0, grown);
            let wr = rows_of(
                &want[1 + c],
                &shape_at(slot.node, grown),
                slot.axis,
                0,
                grown,
            );
            worst = worst.max(rel(&gr, &wr));
            caches.insert(slot.node, got[1 + c].clone());
        }
        p0 = grown;
    }
    worst
}

/// What `derive_step` does with each GLM block kind.
#[test]
fn glm_derive_step_report() {
    let full = GlmDims::tiny();
    let dsa_dense = GlmDims {
        layers: vec![(Mixer::Dsa, Ffn::Dense), (Mixer::Dsa, Ffn::Moe)],
        sparse: false,
        ..GlmDims::tiny()
    };
    let cases = [
        ("full tiny (KDA + sparse DSA)", full.clone()),
        ("DSA dense, absorbed, + mHC + MoE", dsa_dense.clone()),
        (
            "DSA dense, expanded",
            GlmDims {
                absorbed: false,
                ..dsa_dense.clone()
            },
        ),
        (
            "DSA sparse (indexer)",
            GlmDims {
                sparse: true,
                ..dsa_dense.clone()
            },
        ),
        (
            "KDA only",
            GlmDims {
                layers: vec![(Mixer::Kda, Ffn::Moe)],
                ..GlmDims::tiny()
            },
        ),
    ];
    for (name, d) in cases {
        let pg = prefill_graph(&d);
        let r = try_step(&pg, 1, 16);
        eprintln!("derive_step [{name}]: {}", describe(&pg, &r));
    }
}

/// Where the step derives (dense DSA + mHC + MoE), it is the prefill's
/// continuation: decode and a 3-token window from p0 = 5, NaN tails.
#[test]
fn glm_dsa_dense_step_matches_prefill() {
    for absorbed in [true, false] {
        let d = GlmDims {
            layers: vec![(Mixer::Dsa, Ffn::Dense), (Mixer::Dsa, Ffn::Moe)],
            sparse: false,
            absorbed,
            ..GlmDims::tiny()
        };
        let e = step_matches_prefill(&d, 5, &[1, 3, 1], 12);
        eprintln!(
            "glm dsa-dense (absorbed {absorbed}): derived steps [1,3,1] from 5 vs prefill: {e:.2e}"
        );
        assert!(e < 1e-12, "{e:e}");
    }
}

/// Two GLM idioms, each alone: KDA's causal conv (a guarded shifted read
/// along `T`; derives with a ring cache) and a data-dependent attention
/// mask (the indexer's selection; refused).
#[test]
fn glm_derive_step_probes() {
    use tang_compile::tensor::Monoid;
    let spec = StepSpec {
        seq: SEQ,
        new: 1,
        capacity: 8,
    };
    // conv: y[t, c] = Σ_i w[c, i]·x[t − 3 + i, c]
    let mut g = TensorGraph::new();
    let x = g.input(&[T, fx(4)]);
    let w = g.input(&[fx(4), fx(4)]);
    let y = g
        .comp(
            &[T, fx(4)],
            &[fx(4)],
            vec![
                ag(x, vec![fm(3, &[(0, 1), (2, 1)], -3), u(3, 1)]),
                a(w, vec![u(3, 1), u(3, 2)]),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let r = derive_step(&g, &[y], &spec).map(|_| ()).map_err(|e| e.0);
    eprintln!("probe [causal conv, guarded shifted read]: {r:?}");
    // derives since ring caches (a 3-row buffer of x)
    assert!(r.is_ok());
    // data-dependent mask: o[i] = Σ_j select(m[i, j] − ½, x[j], 0), m from data and j ≤ i
    let mut g = TensorGraph::new();
    let x = g.input(&[T]);
    let sp = [T, T];
    let (ii, jj) = (g.iota(&sp, 0), g.iota(&sp, 1));
    let m = g
        .comp(
            &sp,
            &[],
            vec![
                a(x, vec![u(2, 0)]),
                a(x, vec![u(2, 1)]),
                a(ii, id(2)),
                a(jj, id(2)),
            ],
            None,
            |v| {
                vec![select(
                    v[2] - v[3] + lit(0.5),
                    select(v[0] - v[1], lit(1.0), lit(0.0)),
                    lit(0.0),
                )]
            },
        )
        .unwrap();
    let o = g
        .comp(
            &[T],
            &[T],
            vec![a(m, id(2)), a(x, vec![u(2, 1)])],
            Some(Monoid::sum()),
            |v| vec![select(v[0] - lit(0.5), v[1], lit(0.0))],
        )
        .unwrap();
    let r = derive_step(&g, &[o], &spec).map(|_| ()).map_err(|e| e.0);
    eprintln!("probe [data-dependent mask]: {r:?}");
    assert!(r.is_err());
}
