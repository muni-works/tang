//! derive_step's ring caches on the DeepSeek V4.1 and GLM-5.3 tiny prefill
//! graphs (`ir_ds_prefill`, `ir_glm_prefill`, unchanged): which caches
//! become rings, and that chained steps still continue the prefill.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use tang_compile::tensor::step::{derive_step, seed_cache, StepInput, StepSpec};
    use tang_compile::tensor::{Dim, TensorGraph, TensorId};

    const T: Dim = Dim::Sym("T");

    struct Model<'a> {
        g: &'a TensorGraph,
        inputs: Vec<(String, Vec<Dim>)>,
        logits: TensorId,
        named: &'a [(String, TensorId)],
    }

    fn weights(m: &Model) -> HashMap<String, Vec<f64>> {
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

    fn concrete(shape: &[Dim], t: usize) -> Vec<usize> {
        shape
            .iter()
            .map(|&d| if d == T { t } else { d.n() })
            .collect()
    }

    fn name_of(m: &Model, n: TensorId) -> String {
        m.named.iter().find(|(_, t)| *t == n).map_or_else(
            || {
                m.named
                    .iter()
                    .find(|(_, t)| matches!(m.g.nodes()[t.index()].op, tang_compile::tensor::Op::Reshape(s) if s == n))
                    .map_or(format!("{n:?}"), |(s, _)| s.clone())
            },
            |(s, _)| s.clone(),
        )
    }

    /// Cache report and the worst relative error of chained steps from `p0`.
    fn run(m: &Model, vocab: usize, cap: usize, p0: usize, chains: &[&[usize]]) -> f64 {
        let spec = |k| StepSpec {
            seq: "T",
            new: k,
            capacity: cap,
        };
        let st = derive_step(m.g, &[m.logits], &spec(1)).unwrap();
        let mut ring_floats = 0usize;
        let mut full_floats = 0usize;
        for c in &st.caches {
            let sh = st.graph.shape(c.input).to_vec();
            let n: usize = sh.iter().map(|d| d.n()).product();
            let at_cap: usize = concrete(m.g.shape(c.node), cap).iter().product();
            eprintln!(
                "  cache {:<34} ring {:?} shape {:?} (full would be {} floats, now {})",
                name_of(m, c.node),
                c.ring,
                sh,
                at_cap,
                n
            );
            ring_floats += n;
            full_floats += if c.state { n } else { at_cap };
        }
        eprintln!("  cache floats: {ring_floats} now vs {full_floats} all-full at capacity {cap}");
        let w = weights(m);
        let total: usize = chains
            .iter()
            .map(|c| c.iter().sum::<usize>())
            .max()
            .unwrap();
        let all: Vec<f64> = (0..p0 + total)
            .map(|i| ((37 * i + 5) % vocab) as f64)
            .collect();
        let ins = |toks: &[f64]| -> Vec<Vec<f64>> {
            m.inputs
                .iter()
                .map(|(n, _)| {
                    if n == "tokens" {
                        toks.to_vec()
                    } else {
                        w[n].clone()
                    }
                })
                .collect()
        };
        let mut worst = 0.0f64;
        for ks in chains {
            let mut pos = p0;
            let mut caches: Vec<Vec<f64>> = {
                let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
                let fg = m.g.bind(&[("T", pos)]).unwrap();
                let v = fg.eval(&ins(&all[..pos]), &nodes);
                st.caches
                    .iter()
                    .zip(v)
                    .map(|(c, v)| {
                        let sh = concrete(m.g.shape(c.node), pos);
                        if !c.state {
                            return seed_cache(c, &sh, &v, pos, cap);
                        }
                        if !m.g.shape(c.node).contains(&T) {
                            return v;
                        }
                        // a scan's state: row pos − 1 along its axis
                        let outer: usize = sh[..c.axis].iter().product();
                        let inner: usize = sh[c.axis + 1..].iter().product();
                        (0..outer)
                            .flat_map(|o| {
                                let b = (o * pos + pos - 1) * inner;
                                v[b..b + inner].to_vec()
                            })
                            .collect()
                    })
                    .collect()
            };
            for &k in *ks {
                let s = derive_step(m.g, &[m.logits], &spec(k)).unwrap();
                let si: Vec<Vec<f64>> = s
                    .inputs
                    .iter()
                    .map(|&x| match x {
                        StepInput::Pos => vec![pos as f64],
                        StepInput::New(_) => all[pos..pos + k].to_vec(),
                        StepInput::Same(i) => w[&m.inputs[i].0].clone(),
                        StepInput::Cache(c) => caches[c].clone(),
                    })
                    .collect();
                let mut ids = s.outputs.clone();
                ids.extend(s.caches.iter().map(|c| c.updated));
                let got = s.graph.eval(&si, &ids);
                let grown = pos + k;
                let fg = m.g.bind(&[("T", grown)]).unwrap();
                let want = fg.eval(&ins(&all[..grown]), &[m.logits]);
                let want = &want[0][pos * vocab..grown * vocab];
                let scale = want.iter().fold(1e-300f64, |a, x| a.max(x.abs()));
                let e = got[0]
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs() / scale)
                    .fold(0.0, |a: f64, x| {
                        if x.is_finite() {
                            a.max(x)
                        } else {
                            f64::INFINITY
                        }
                    });
                worst = worst.max(e);
                caches = got[1..].to_vec();
                pos = grown;
            }
            eprintln!("  steps {ks:?} from {p0}: max rel err {worst:.2e}");
        }
        worst
    }

    const CHAINS: [&[usize]; 4] = [&[1], &[4], &[4, 1, 1, 2], &[1, 1, 1, 1, 1, 1, 1, 1, 1]];

    #[test]
    fn ds_tiny_rings() {
        use crate::ir_ds_prefill::*;
        let dm = DsDims::tiny();
        let pg = prefill_graph(&dm);
        let m = Model {
            g: &pg.g,
            inputs: pg
                .inputs
                .iter()
                .map(|i| (i.name.clone(), i.shape.clone()))
                .collect(),
            logits: pg.logits,
            named: &pg.named,
        };
        eprintln!("ds tiny (window {}, ctx {}):", dm.window, dm.ctx);
        let e = run(&m, dm.vocab, dm.ctx, 5, &CHAINS);
        assert!(e < 1e-10, "{e:e}");
    }

    #[test]
    fn glm_kda_rings() {
        use crate::ir_glm_prefill::*;
        for (name, d) in [(
            "KDA only",
            GlmDims {
                layers: vec![(Mixer::Kda, Ffn::Moe)],
                ..GlmDims::tiny()
            },
        )] {
            let pg = prefill_graph(&d);
            let m = Model {
                g: &pg.g,
                inputs: pg
                    .inputs
                    .iter()
                    .map(|i| (i.name.clone(), i.shape.clone()))
                    .collect(),
                logits: pg.logits,
                named: &pg.named,
            };
            eprintln!("glm tiny [{name}]:");
            let e = run(&m, d.vocab, 24, 2, &CHAINS);
            assert!(e < 1e-10, "{e:e}");
        }
        // KDA feeding DSA: refused by the carried-state rule (not rings)
        let d = GlmDims {
            layers: vec![(Mixer::Kda, Ffn::Dense), (Mixer::Dsa, Ffn::Moe)],
            sparse: false,
            ..GlmDims::tiny()
        };
        eprintln!(
            "glm KDA + dense DSA: {:?}",
            try_step(&prefill_graph(&d), 1, 24).err()
        );
        // the full tiny (sparse indexer) still refuses: say where
        let pg = prefill_graph(&GlmDims::tiny());
        eprintln!("glm full tiny: {:?}", try_step(&pg, 1, 24).err());
    }
}
