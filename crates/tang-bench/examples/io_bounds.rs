//! IO lower bounds vs the traffic of the schedule tang picks.
//!
//! For each kernel class — matmul, causal attention forward and
//! forward+backward (FlashAttention-2 placement), row softmax, RMSNorm,
//! a gathered MoE expert GEMM — build the comprehension graph, schedule it
//! the way the GPU lowering does, and report (in MB at f32):
//!
//! * `bound`: the best lower bound on slow↔fast traffic over all schedules
//!   at fast-memory size `M` — the closed form from the literature where
//!   one exists (`tensor::io::bounds`), else the generic compulsory / HBL
//!   bound tang computes from the comprehensions (`io_report`);
//! * `generic`: tang's own bound from the IR alone (no kernel knowledge);
//! * `staged`: the schedule's traffic, tile re-reads included, no credit
//!   for cross-workgroup cache hits (`io_report().achieved()`);
//! * `once`: each kernel reading each buffer once — the `hbm_bytes` model
//!   of `memory_axis` (validated against `paper/tang/data/memory-axis.json`);
//! * `ratio` = staged / bound: ≥ 1 always; how far from IO-optimal the
//!   picked schedule can be.
//!
//! `M` is the fast memory one workgroup can stage: 3090 48 KB of shared
//! memory per block (12288 f32 words), M4 32 KB of threadgroup memory
//! (8192 words). One JSON object per line; `--md` prints a markdown table.
//!
//!   cargo run --release -p tang-bench --example io_bounds [-- --md]
use tang::Scalar;
use tang_compile::tensor::io::{achievable, bounds};
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{Arg, MemorySchedule, Monoid, TensorGraph, TensorId, View};
use tang_derive_wasm::{naive_graph, parse_variant};

struct Row {
    kernel: String,
    shape: String,
    device: &'static str,
    mem: f64,
    bound: f64,
    generic: f64,
    staged: f64,
    once: f64,
    kernels: usize,
    /// Best known schedule's traffic, where one is modelled.
    known: Option<f64>,
    /// `hbm_bytes / 4` exactly as `memory_axis` computes it from the
    /// lowered WGSL program (checks `once` and the kernel list).
    wgsl_once: Option<f64>,
    source: &'static str,
}

/// `memory_axis`'s `hbm_bytes` model, in words.
fn wgsl_once(p: &WgslProgram) -> f64 {
    let len = |t: TensorId| {
        let s = p.storage(t);
        p.buffers
            .iter()
            .find(|b| b.node == s)
            .map_or(0, |b| b.len as u64)
    };
    p.kernels
        .iter()
        .map(|k| {
            let mut r: Vec<TensorId> = k.reads.iter().map(|&t| p.storage(t)).collect();
            r.sort_by_key(|t| t.index());
            r.dedup();
            r.iter().map(|&t| len(t)).sum::<u64>() + len(k.node)
        })
        .sum::<u64>() as f64
}

/// Traffic of the graph exactly as `to_wgsl_with(outs, nest)` would lower it.
fn analyze(g: &TensorGraph, outs: &[TensorId], nest: bool, mem: f64) -> (f64, f64, f64, usize) {
    let mut g = g.clone();
    if nest {
        g.schedule_nested_contractions_scalar(outs);
    }
    let r = g.io_report(outs, mem);
    let once: f64 = r
        .kernels
        .iter()
        .map(|k| {
            k.reads
                .iter()
                .map(|(t, _)| tang_compile::tensor::numel(g.shape(*t)) as f64)
                .sum::<f64>()
                + k.write
        })
        .sum();
    (r.bound(), r.achieved(), once, r.kernels.len())
}

fn matmul(m: usize, n: usize, k: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let a = g.input(&[m, k]);
    let b = g.input(&[k, n]);
    let c = g.matmul(a, b).unwrap();
    g.schedule_auto(&[c]);
    (g, vec![c])
}

/// `y = softmax(x)` per row (`norm = false`) or RMSNorm `x / √(mean x² + ε)`.
fn row_norm(r: usize, c: usize, norm: bool) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[r, c]);
    let rowv = |rank| View::pick(rank, &[Some(0), Some(rank - 1)]);
    let y = if norm {
        let ss = g
            .comp(
                &[r, 1],
                &[c],
                vec![Arg {
                    src: x,
                    view: rowv(3),
                }],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        let inv_c = 1.0 / c as f64;
        g.comp(
            &[r, c],
            &[] as &[usize],
            vec![
                Arg {
                    src: x,
                    view: View::identity(2, 2),
                },
                Arg {
                    src: ss,
                    view: View::pick(2, &[Some(0), None]),
                },
            ],
            None,
            |v| {
                vec![
                    v[0] * (v[1] * tang_expr::ExprId::from_f64(inv_c)
                        + tang_expr::ExprId::from_f64(1e-6))
                    .sqrt()
                    .recip(),
                ]
            },
        )
        .unwrap()
    } else {
        let mx = g
            .comp(
                &[r, 1],
                &[c],
                vec![Arg {
                    src: x,
                    view: rowv(3),
                }],
                Some(Monoid::max()),
                |v| vec![v[0]],
            )
            .unwrap();
        let sm = g
            .comp(
                &[r, 1],
                &[c],
                vec![
                    Arg {
                        src: x,
                        view: rowv(3),
                    },
                    Arg {
                        src: mx,
                        view: View::pick(3, &[Some(0), None]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![(v[0] - v[1]).exp()],
            )
            .unwrap();
        g.comp(
            &[r, c],
            &[] as &[usize],
            vec![
                Arg {
                    src: x,
                    view: View::identity(2, 2),
                },
                Arg {
                    src: mx,
                    view: View::pick(2, &[Some(0), None]),
                },
                Arg {
                    src: sm,
                    view: View::pick(2, &[Some(0), None]),
                },
            ],
            None,
            |v| vec![(v[0] - v[1]).exp() * v[2].recip()],
        )
        .unwrap()
    };
    let (g, outs) = g.substitute_maps(&[y]);
    let (mut g, outs) = g.stabilize_softmax(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}

/// `out[t, k, f] = Σ_d W[sel[t, k], f, d] · x[t, d]`.
fn moe(t: usize, kk: usize, e: usize, f: usize, d: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let w = g.input(&[e, f, d]);
    let sel = g.input(&[t, kk]);
    let x = g.input(&[t, d]);
    // space (t, k, f | d)
    let wv = View::pick(4, &[None, Some(2), Some(3)]).gather(0, 1, 1);
    let out = g
        .comp(
            &[t, kk, f],
            &[d],
            vec![
                Arg { src: w, view: wv },
                Arg {
                    src: sel,
                    view: View::pick(4, &[Some(0), Some(1)]),
                },
                Arg {
                    src: x,
                    view: View::pick(4, &[Some(0), Some(3)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[2]],
        )
        .unwrap();
    g.schedule_auto(&[out]);
    (g, vec![out])
}

fn main() {
    let md = std::env::args().any(|a| a == "--md");
    let devices: [(&str, f64); 2] = [("3090", 12288.0), ("m4", 8192.0)];
    let mut rows: Vec<Row> = Vec::new();
    let mb = |w: f64| w * 4.0 / 1e6;
    for &(dev, mem) in &devices {
        for &(m, n, k) in &[
            (4096usize, 4096usize, 4096usize),
            (512, 1536, 576),
            (128, 576, 1536),
        ] {
            let (g, o) = matmul(m, n, k);
            let (generic, staged, once, nk) = analyze(&g, &o, false, mem);
            rows.push(Row {
                kernel: "matmul".into(),
                shape: format!("{m}×{k}·{k}×{n}"),
                device: dev,
                mem,
                bound: bounds::matmul(m as f64, n as f64, k as f64, mem).max(generic),
                generic,
                staged,
                once,
                kernels: nk,
                known: None,
                wgsl_once: None,
                source: "Kwasniewski'19 Thm 1",
            });
        }
        let v = parse_variant("s", "j <= i").unwrap();
        for &(n, d) in &[(1024usize, 64usize), (4096, 64), (4096, 128), (16384, 64)] {
            for bwd in [false, true] {
                let (g, outs) = naive_graph(&v, n, d, bwd).unwrap();
                let pl = g.place(&outs, &MemorySchedule::flash());
                let (generic, staged, once, nk) = analyze(&pl.graph, &pl.outputs, pl.nest, mem);
                let (nf, df) = (n as f64, d as f64);
                let known = achievable::flash_fwd(nf, df, mem, true)
                    + if bwd {
                        achievable::flash_bwd(nf, df, mem, true)
                    } else {
                        0.0
                    };
                let lit = if bwd {
                    // fwd+bwd: the forward is recomputed inside, both bounds hold
                    bounds::attention(n as f64, d as f64, mem, true)
                        .max(bounds::attention_bwd(n as f64, d as f64, mem, true))
                } else {
                    bounds::attention(n as f64, d as f64, mem, true)
                };
                rows.push(Row {
                    kernel: if bwd {
                        "attention fwd+bwd (flash)"
                    } else {
                        "attention fwd (flash)"
                    }
                    .into(),
                    shape: format!("N={n} d={d} causal"),
                    device: dev,
                    mem,
                    bound: lit.max(generic),
                    generic,
                    staged,
                    once,
                    kernels: nk,
                    known: Some(known),
                    wgsl_once: Some(wgsl_once(&pl.graph.to_wgsl_with(&pl.outputs, pl.nest))),
                    source: "Saha–Ye'24 (const. from Lemma)",
                });
            }
        }
        for norm in [false, true] {
            for &(r, c) in &[(4096usize, 4096usize), (4096, 32768)] {
                let (g, o) = row_norm(r, c, norm);
                let (generic, staged, once, nk) = analyze(&g, &o, false, mem);
                rows.push(Row {
                    kernel: if norm { "rmsnorm" } else { "softmax" }.into(),
                    shape: format!("{r}×{c}"),
                    device: dev,
                    mem,
                    bound: bounds::row_normalize(r as f64, c as f64, mem).max(generic),
                    generic,
                    staged,
                    once,
                    kernels: nk,
                    known: None,
                    wgsl_once: None,
                    source: "row argument (docs/proofs.md)",
                });
            }
        }
        for &(t, kk, e, f, d) in &[
            (8usize, 10usize, 128usize, 512usize, 2048usize),
            (4096, 8, 64, 1408, 2048),
        ] {
            let (g, o) = moe(t, kk, e, f, d);
            let (generic, staged, once, nk) = analyze(&g, &o, false, mem);
            // only the selected experts' weights are compulsory: at most
            // min(e, t·k) of them; balanced routing assumed for the GEMM part
            let active = (e.min(t * kk)) as f64;
            let lit = (active * (f * d) as f64 + (t * d) as f64 + (t * kk * f) as f64).max(
                bounds::moe_matmul(t as f64, kk as f64, active, d as f64, f as f64, mem),
            );
            rows.push(Row {
                kernel: "MoE expert GEMM (gather)".into(),
                shape: format!("T={t} k={kk} E={e} {d}→{f}"),
                device: dev,
                mem,
                bound: lit,
                generic,
                staged,
                once,
                kernels: nk,
                known: None,
                wgsl_once: None,
                source: "compulsory + Kwasniewski per expert",
            });
        }
    }
    if md {
        println!("| kernel | shape | device (M words) | bound MB | tang generic MB | best known MB | staged MB | once MB (wgsl check) | kernels | staged/bound | bound source |");
        println!("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|");
        let opt = |x: Option<f64>| x.map_or("–".to_string(), |v| format!("{:.2}", mb(v)));
        for r in &rows {
            let check = match r.wgsl_once {
                Some(w) if (w - r.once).abs() < 0.5 => " ✓".to_string(),
                Some(w) => format!(" ✗ {:.2}", mb(w)),
                None => String::new(),
            };
            println!(
                "| {} | {} | {} ({}) | {:.2} | {:.2} | {} | {:.2} | {:.2}{} | {} | {:.2} | {} |",
                r.kernel,
                r.shape,
                r.device,
                r.mem,
                mb(r.bound),
                mb(r.generic),
                opt(r.known),
                mb(r.staged),
                mb(r.once),
                check,
                r.kernels,
                r.staged / r.bound,
                r.source
            );
        }
    } else {
        for r in &rows {
            println!(
                "{{\"kernel\":\"{}\",\"shape\":\"{}\",\"device\":\"{}\",\"mem_words\":{},\"bound_bytes\":{:.0},\"generic_bytes\":{:.0},\"staged_bytes\":{:.0},\"once_bytes\":{:.0},\"kernels\":{},\"known_bytes\":{},\"wgsl_once_bytes\":{},\"ratio\":{:.4},\"source\":\"{}\"}}",
                r.kernel, r.shape, r.device, r.mem, r.bound * 4.0, r.generic * 4.0, r.staged * 4.0, r.once * 4.0, r.kernels,
                r.known.map_or("null".into(), |v| format!("{:.0}", v * 4.0)),
                r.wgsl_once.map_or("null".into(), |v| format!("{:.0}", v * 4.0)),
                r.staged / r.bound, r.source
            );
        }
    }
}
