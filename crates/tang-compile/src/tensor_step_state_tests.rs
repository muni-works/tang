//! Carried states with a per-channel gate (GLM's KDA), and ragged
//! batching: sessions at different lengths on a batch dim `B`, stepped
//! together with one `p0` each.

use std::collections::HashMap;

use tang::Scalar;
use tang_expr::ExprId;

use super::super::{Arg, Dim, Monoid, Op, TensorGraph, TensorId, View};
use super::{derive_step_with, Rows, StepInput, StepOptions, StepSpec};

const T: Dim = Dim::Sym("T");
const B: Dim = Dim::Sym("B");

fn spec(k: usize) -> StepSpec {
    StepSpec {
        seq: "T",
        new: k,
        capacity: 16,
    }
}

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

fn sigmoid(x: ExprId) -> ExprId {
    lit(1.0) / (lit(1.0) + (-x).exp())
}

/// Deterministic values in about [-scale, scale].
fn rand(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Rows `r0..r1` along `axis` of a row-major `shape` buffer.
fn rows(buf: &[f64], shape: &[usize], axis: usize, r0: usize, r1: usize) -> Vec<f64> {
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
}

/// Worst `|a − b| / (1 + |b|)`; NaN or ∞ anywhere is ∞.
fn max_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let e = (x - y).abs() / (1.0 + y.abs());
            if e.is_finite() {
                e
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

/// `y[b, t, n] = Σ_a x[b, t, a]·w[a, n]`.
fn bproj(g: &mut TensorGraph, x: TensorId, a: usize, n: usize) -> TensorId {
    let w = g.input(&[a, n]);
    g.comp(
        &[B, T, Dim::Fixed(n)],
        &[Dim::Fixed(a)],
        vec![
            Arg {
                src: x,
                view: View::pick(4, &[Some(0), Some(1), Some(3)]),
            },
            Arg {
                src: w,
                view: View::pick(4, &[Some(3), Some(2)]),
            },
        ],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

fn pick_arg(src: TensorId, picks: &[Option<usize>], rank: usize) -> Arg {
    Arg {
        src,
        view: View::pick(rank, picks),
    }
}

fn sel(c: ExprId, x: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|eg| eg.select(c, x, f))
}

fn vis(t: ExprId, j: ExprId) -> ExprId {
    t - j + lit(0.5)
}

/// Independent sessions on `B`: causal attention (select masks), then a
/// linear-attention scan layer, as residual blocks over `x [B, T, dm]`.
fn batched_model() -> (TensorGraph, Vec<TensorId>) {
    let (dm, dh, n) = (6, 4, 4);
    let scale = 1.0 / (dh as f64).sqrt();
    let mut g = TensorGraph::new();
    let x = g.input(&[B, T, Dim::Fixed(dm)]);
    let q = bproj(&mut g, x, dm, dh);
    let k = bproj(&mut g, x, dm, dh);
    let v = bproj(&mut g, x, dm, dh);
    let s = g
        .comp(
            &[B, T, T],
            &[Dim::Fixed(dh)],
            vec![
                pick_arg(q, &[Some(0), Some(1), Some(3)], 4),
                pick_arg(k, &[Some(0), Some(2), Some(3)], 4),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let (it, ij) = (g.iota(&[B, T, T], 1), g.iota(&[B, T, T], 2));
    let id3 = [Some(0), Some(1), Some(2)];
    let m = g
        .comp(
            &[B, T],
            &[T],
            vec![
                pick_arg(s, &id3, 3),
                pick_arg(it, &id3, 3),
                pick_arg(ij, &id3, 3),
            ],
            Some(Monoid::max()),
            move |v| vec![sel(vis(v[1], v[2]), v[0] * lit(scale), lit(-1e30))],
        )
        .unwrap();
    let l = g
        .comp(
            &[B, T],
            &[T],
            vec![
                pick_arg(s, &id3, 3),
                pick_arg(m, &[Some(0), Some(1)], 3),
                pick_arg(it, &id3, 3),
                pick_arg(ij, &id3, 3),
            ],
            Some(Monoid::sum()),
            move |v| {
                let p = (v[0] * lit(scale) - v[1]).exp();
                vec![sel(vis(v[2], v[3]), p, lit(0.0))]
            },
        )
        .unwrap();
    let sp = [B, T, Dim::Fixed(dh), T];
    let (it4, ij4) = (g.iota(&sp, 1), g.iota(&sp, 3));
    let id4 = [Some(0), Some(1), Some(2), Some(3)];
    let o = g
        .comp(
            &[B, T, Dim::Fixed(dh)],
            &[T],
            vec![
                pick_arg(s, &[Some(0), Some(1), Some(3)], 4),
                pick_arg(m, &[Some(0), Some(1)], 4),
                pick_arg(l, &[Some(0), Some(1)], 4),
                pick_arg(v, &[Some(0), Some(3), Some(2)], 4),
                pick_arg(it4, &id4, 4),
                pick_arg(ij4, &id4, 4),
            ],
            Some(Monoid::sum()),
            move |v| {
                let pv = (v[0] * lit(scale) - v[1]).exp() * v[3] / v[2];
                vec![sel(vis(v[4], v[5]), pv, lit(0.0))]
            },
        )
        .unwrap();
    let ao = bproj(&mut g, o, dh, dm);
    let h = g.map(&[x, ao], |v| v[0] + v[1]).unwrap();
    let q2 = bproj(&mut g, h, dm, n);
    let k2 = bproj(&mut g, h, dm, n);
    let v2 = bproj(&mut g, h, dm, n);
    let k2 = g
        .map(&[k2], |v| lit(2.0) * sigmoid(v[0]) - lit(1.0))
        .unwrap();
    let st = g.delta_scan(k2, v2, None, None, 1, false).unwrap();
    let ro = g.scan_readout(st, q2, 1.0 / (n as f64).sqrt()).unwrap();
    let lo = bproj(&mut g, ro, n, dm);
    let y = g.map(&[h, lo], |v| v[0] + v[1]).unwrap();
    (g, vec![y])
}

/// Sessions at lengths 3, 6 and 5 stepped together (one batched step,
/// `p0` per session) equal each session stepped alone, bit for bit
/// (outputs and every cache buffer, NaN tails included), and match the
/// prefill of each grown session. Steps of 4, then 1.
#[test]
fn ragged_sessions_step_together_bit_for_bit() {
    let (g, outs) = batched_model();
    let ps = [3usize, 6, 5];
    let ks = [4usize, 1];
    let cap = 16;
    let tmax = 12;
    let nb = ps.len();
    let mut shapes: Vec<(usize, Vec<Dim>)> = g
        .nodes()
        .iter()
        .filter_map(|nd| match nd.op {
            Op::Input(i) => Some((i, nd.shape.clone())),
            _ => None,
        })
        .collect();
    shapes.sort_by_key(|x| x.0);
    let dm = shapes[0].1[2].n();
    // x per session ([1, tmax, dm]), weights shared
    let xs: Vec<Vec<f64>> = (0..nb)
        .map(|b| rand(tmax * dm, 100 + b as u64, 0.8))
        .collect();
    let weights: Vec<Vec<f64>> = shapes
        .iter()
        .map(|(i, sh)| {
            let n: usize = sh.iter().map(|d| d.fixed().unwrap_or(1)).product();
            rand(n, 17 + *i as u64, 0.8)
        })
        .collect();
    let x_rows = |b: usize, r0: usize, r1: usize| xs[b][r0 * dm..r1 * dm].to_vec();
    let prefill = |b: usize, t: usize, ids: &[TensorId]| {
        let pg = g.bind(&[("B", 1), ("T", t)]).unwrap();
        let mut ins = weights.clone();
        ins[0] = x_rows(b, 0, t);
        pg.eval(&ins, ids)
    };
    let concrete1 = |sh: &[Dim], t: usize| -> Vec<usize> {
        sh.iter()
            .map(|&d| {
                if d == T {
                    t
                } else if d == B {
                    1
                } else {
                    d.n()
                }
            })
            .collect()
    };
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut pos = ps;
    // per-session cache buffers ([1, …]) by prefill node
    let mut caches: Vec<HashMap<TensorId, Vec<f64>>> = vec![HashMap::new(); nb];
    let mut worst = 0.0f64;
    let mut slots = (0, 0);
    for &k in &ks {
        let opts = StepOptions {
            rows: Rows::All,
            batch: Some("B"),
        };
        let step = derive_step_with(&g, &outs, &spec(k), &opts).unwrap();
        slots = (
            step.caches.iter().filter(|c| !c.state).count(),
            step.caches.iter().filter(|c| c.state).count(),
        );
        for c in &step.caches {
            assert_eq!(g.shape(c.node)[0], B, "batch dim leads {:?}", c.node);
        }
        if caches[0].is_empty() {
            let nodes: Vec<TensorId> = step.caches.iter().map(|c| c.node).collect();
            for b in 0..nb {
                let vals = prefill(b, pos[b], &nodes);
                for (c, v) in step.caches.iter().zip(vals) {
                    let shape = concrete1(g.shape(c.node), pos[b]);
                    let buf = if c.state {
                        rows(&v, &shape, c.axis, pos[b] - 1, pos[b])
                    } else {
                        let inner: usize = shape[c.axis + 1..].iter().product();
                        let outer: usize = shape[..c.axis].iter().product();
                        let n: usize = concrete1(g.shape(c.node), cap).iter().product();
                        let mut buf = vec![f64::NAN; n];
                        for o in 0..outer {
                            for r in 0..pos[b] {
                                let s0 = (o * pos[b] + r) * inner;
                                let d0 = (o * cap + r) * inner;
                                buf[d0..d0 + inner].copy_from_slice(&v[s0..s0 + inner]);
                            }
                        }
                        buf
                    };
                    caches[b].insert(c.node, buf);
                }
            }
        }
        let mut ids = step.outputs.clone();
        ids.extend(step.caches.iter().map(|c| c.updated));
        let inputs_for = |sessions: &[usize]| -> Vec<Vec<f64>> {
            step.inputs
                .iter()
                .map(|si| match *si {
                    StepInput::Pos => sessions.iter().map(|&b| pos[b] as f64).collect(),
                    StepInput::New(_) => sessions
                        .iter()
                        .flat_map(|&b| x_rows(b, pos[b], pos[b] + k))
                        .collect(),
                    StepInput::Same(i) => weights[i].clone(),
                    StepInput::Cache(c) => sessions
                        .iter()
                        .flat_map(|&b| caches[b][&step.caches[c].node].clone())
                        .collect(),
                })
                .collect()
        };
        let all: Vec<usize> = (0..nb).collect();
        let together = step
            .graph
            .bind(&[("B", nb)])
            .unwrap()
            .eval(&inputs_for(&all), &ids);
        let one = step.graph.bind(&[("B", 1)]).unwrap();
        let mut updates = Vec::new();
        for b in 0..nb {
            let alone = one.eval(&inputs_for(&[b]), &ids);
            for (t, a) in together.iter().zip(&alone) {
                let per = t.len() / nb;
                assert_eq!(bits(&t[b * per..(b + 1) * per]), bits(a), "session {b}");
            }
            // the grown session's prefill, new rows
            let want = prefill(b, pos[b] + k, &outs);
            let shape = concrete1(g.shape(outs[0]), pos[b] + k);
            let e = max_err(&alone[0], &rows(&want[0], &shape, 1, pos[b], pos[b] + k));
            worst = worst.max(e);
            updates.push(alone);
        }
        for (b, alone) in updates.into_iter().enumerate() {
            for (ci, c) in step.caches.iter().enumerate() {
                caches[b].insert(c.node, alone[step.outputs.len() + ci].clone());
            }
        }
        for p in pos.iter_mut() {
            *p += k;
        }
    }
    eprintln!(
        "ragged sessions {ps:?}, steps {ks:?}: {} row caches + {} states; together == alone \
         bit for bit; vs prefill max rel err {worst:.2e}",
        slots.0, slots.1
    );
    assert_eq!(slots, (2, 1));
    assert!(worst < 1e-12, "{worst}");
}

/// A comp that reads a position but has no batch coordinate is refused
/// in batched mode (it could not know whose `p0` it is).
#[test]
fn batched_position_without_batch_coordinate_is_refused() {
    let mut g = TensorGraph::new();
    let x = g.input(&[B, T, Dim::Fixed(2)]);
    // a [T, 2] position table, no batch dim, added to every session
    let t = g.iota(&[T, Dim::Fixed(2)], 0);
    let pe = g.map(&[t], |v| v[0] * lit(0.1)).unwrap();
    let y = g
        .comp(
            &[B, T, Dim::Fixed(2)],
            &[],
            vec![
                pick_arg(x, &[Some(0), Some(1), Some(2)], 3),
                pick_arg(pe, &[Some(1), Some(2)], 3),
            ],
            None,
            |v| vec![v[0] + v[1]],
        )
        .unwrap();
    let opts = StepOptions {
        rows: Rows::All,
        batch: Some("B"),
    };
    let msg = derive_step_with(&g, &[y], &spec(1), &opts).err().unwrap().0;
    assert!(msg.contains("batch"), "{msg}");
}

/// KDA (GLM-5.3's Kimi delta attention): the gated delta rule with a
/// per-channel gate, `S_t = diag(α_t)·(I − β_t·k_t·k_tᵀ)·S_{t−1} +
/// β_t·k_t·v_tᵀ`, written as a raw `linear_scan` over an explicit `A` (no
/// `DeltaForm`: `delta_scan`'s α is a scalar per row). Readout and output
/// projection over `x [T, dm]`.
fn kda_model() -> (TensorGraph, Vec<TensorId>, TensorId) {
    let (dm, n, m) = (6, 4, 3);
    let mut g = TensorGraph::new();
    let x = g.input(&[T, Dim::Fixed(dm)]);
    let (wq, wk, wv, wa, wb, wo) = (
        g.input(&[dm, n]),
        g.input(&[dm, n]),
        g.input(&[dm, m]),
        g.input(&[dm, n]),
        g.input(&[dm, 1]),
        g.input(&[m, dm]),
    );
    let q = g.matmul(x, wq).unwrap();
    let k = g.matmul(x, wk).unwrap();
    let v = g.matmul(x, wv).unwrap();
    let sq = g.map(&[k], |v| v[0] * v[0]).unwrap();
    let ss = g.reduce(sq, 1, Monoid::sum()).unwrap();
    let ssb = g.broadcast(ss, &[T, Dim::Fixed(n)]).unwrap();
    let k = g
        .map(&[k, ssb], |v| v[0] * (v[1] + lit(1e-6)).sqrt().recip())
        .unwrap();
    let a = g.matmul(x, wa).unwrap();
    let alpha = g.map(&[a], |v| sigmoid(v[0] + lit(2.0))).unwrap(); // [T, n]
    let b = g.matmul(x, wb).unwrap();
    let beta = g.map(&[b], |v| sigmoid(v[0])).unwrap(); // [T, 1]
                                                        // A[t, i, j] = α[t, i]·(δᵢⱼ − β[t]·k[t, i]·k[t, j])
    let (ii, jj) = (
        g.iota(&[Dim::Fixed(n), Dim::Fixed(n)], 0),
        g.iota(&[Dim::Fixed(n), Dim::Fixed(n)], 1),
    );
    let am = g
        .comp(
            &[T, Dim::Fixed(n), Dim::Fixed(n)],
            &[],
            vec![
                pick_arg(alpha, &[Some(0), Some(1)], 3),
                pick_arg(beta, &[Some(0), None], 3),
                pick_arg(k, &[Some(0), Some(1)], 3),
                pick_arg(k, &[Some(0), Some(2)], 3),
                pick_arg(ii, &[Some(1), Some(2)], 3),
                pick_arg(jj, &[Some(1), Some(2)], 3),
            ],
            None,
            |v| {
                let d = v[4] - v[5];
                let eye = sel(lit(0.5) - d * d, lit(1.0), lit(0.0));
                vec![v[0] * (eye - v[1] * v[2] * v[3])]
            },
        )
        .unwrap();
    let um = g
        .comp(
            &[T, Dim::Fixed(n), Dim::Fixed(m)],
            &[],
            vec![
                pick_arg(beta, &[Some(0), None], 3),
                pick_arg(k, &[Some(0), Some(1)], 3),
                pick_arg(v, &[Some(0), Some(2)], 3),
            ],
            None,
            |v| vec![v[0] * v[1] * v[2]],
        )
        .unwrap();
    let s = g.linear_scan(am, um, 0, false, false).unwrap();
    let o = g.scan_readout(s, q, 1.0 / (n as f64).sqrt()).unwrap();
    let y = g.matmul(o, wo).unwrap();
    (g, vec![y], s)
}

/// KDA's state is the cache: prefill to 5, then steps [1], [4] and
/// [4, 1, 2] each match the prefill of the grown sequence (outputs and the
/// state); a state one row stale does not.
#[test]
fn kda_per_channel_gate_state_matches_prefill() {
    let (g, outs, s) = kda_model();
    let tmax = 13;
    let mut shapes: Vec<(usize, Vec<Dim>)> = g
        .nodes()
        .iter()
        .filter_map(|nd| match nd.op {
            Op::Input(i) => Some((i, nd.shape.clone())),
            _ => None,
        })
        .collect();
    shapes.sort_by_key(|x| x.0);
    let data: Vec<Vec<f64>> = shapes
        .iter()
        .map(|(i, sh)| {
            let n: usize = sh
                .iter()
                .map(|&d| if d == T { tmax } else { d.n() })
                .product();
            rand(n, 31 + *i as u64, 0.8)
        })
        .collect();
    let dm = shapes[0].1[1].n();
    let inputs_at = |t: usize| {
        let mut ins = data.clone();
        ins[0] = data[0][..t * dm].to_vec();
        ins
    };
    let prefill =
        |t: usize, ids: &[TensorId]| g.bind(&[("T", t)]).unwrap().eval(&inputs_at(t), ids);
    let sshape = |t: usize| -> Vec<usize> {
        g.shape(s)
            .iter()
            .map(|&d| if d == T { t } else { d.n() })
            .collect()
    };
    let run = |p0: usize, ks: &[usize], stale: usize| -> f64 {
        let mut pos = p0;
        let st0 = p0 - stale;
        let mut state = rows(&prefill(st0, &[s])[0], &sshape(st0), 0, st0 - 1, st0);
        let mut worst = 0.0f64;
        for &k in ks {
            let step = derive_step_with(&g, &outs, &spec(k), &StepOptions::default()).unwrap();
            assert_eq!(step.caches.len(), 1);
            assert!(step.caches[0].state && step.caches[0].node == s);
            let ins: Vec<Vec<f64>> = step
                .inputs
                .iter()
                .map(|si| match *si {
                    StepInput::Pos => vec![pos as f64],
                    StepInput::New(_) => data[0][pos * dm..(pos + k) * dm].to_vec(),
                    StepInput::Same(i) => data[i].clone(),
                    StepInput::Cache(_) => state.clone(),
                })
                .collect();
            let got = step
                .graph
                .eval(&ins, &[step.outputs[0], step.caches[0].updated]);
            let grown = pos + k;
            let want = prefill(grown, &[outs[0], s]);
            let yshape = [grown, dm];
            worst = worst.max(max_err(&got[0], &rows(&want[0], &yshape, 0, pos, grown)));
            worst = worst.max(max_err(
                &got[1],
                &rows(&want[1], &sshape(grown), 0, grown - 1, grown),
            ));
            state = got[1].clone();
            pos = grown;
        }
        worst
    };
    for ks in [&[1usize][..], &[4], &[4, 1, 2]] {
        let e = run(5, ks, 0);
        eprintln!("kda ks={ks:?}: max rel err {e:.2e}");
        assert!(e < 1e-12, "{ks:?}: {e}");
    }
    let e = run(5, &[1], 1);
    eprintln!("kda, stale state: max rel err {e:.2e}");
    assert!(e > 1e-3, "{e}");
}

/// A scan's per-row output feeding attention whose K and V are cached
/// (KDA → DSA in GLM): row t's keys read the scan state at t, final once
/// computed, so the rows cache like any other. Prefill to 5, steps [1],
/// [4], [4, 1, 2], NaN past the filled cache rows.
#[test]
fn scan_rows_feed_cached_attention() {
    let (dm, n, dh) = (6, 4, 4);
    let mut g = TensorGraph::new();
    let x = g.input(&[T, Dim::Fixed(dm)]);
    // linear attention: S_t = S_{t−1} + k_t·v_tᵀ, o_t = S_tᵀ·q_t
    let (wq, wk, wv, wo) = (
        g.input(&[dm, n]),
        g.input(&[dm, n]),
        g.input(&[dm, n]),
        g.input(&[n, dm]),
    );
    let q = g.matmul(x, wq).unwrap();
    let k = g.matmul(x, wk).unwrap();
    let k = g
        .map(&[k], |v| lit(2.0) * sigmoid(v[0]) - lit(1.0))
        .unwrap();
    let v = g.matmul(x, wv).unwrap();
    let st = g.delta_scan(k, v, None, None, 0, false).unwrap();
    let ro = g.scan_readout(st, q, 0.5).unwrap();
    let lo = g.matmul(ro, wo).unwrap();
    let h = g.map(&[x, lo], |v| v[0] + v[1]).unwrap();
    // causal attention over h: its K, V rows come from the scan's rows
    let (aq, ak, av) = (g.input(&[dm, dh]), g.input(&[dm, dh]), g.input(&[dm, dh]));
    let q2 = g.matmul(h, aq).unwrap();
    let k2 = g.matmul(h, ak).unwrap();
    let v2 = g.matmul(h, av).unwrap();
    let kt = g.transpose(k2, &[1, 0]).unwrap();
    let s = g.matmul(q2, kt).unwrap();
    let (it, ij) = (g.iota(&[T, T], 0), g.iota(&[T, T], 1));
    let scale = 0.5;
    let p2 = |s: TensorId| pick_arg(s, &[Some(0), Some(1)], 2);
    let m = g
        .comp(
            &[T],
            &[T],
            vec![p2(s), p2(it), p2(ij)],
            Some(Monoid::max()),
            move |v| vec![sel(vis(v[1], v[2]), v[0] * lit(scale), lit(-1e30))],
        )
        .unwrap();
    let l = g
        .comp(
            &[T],
            &[T],
            vec![p2(s), pick_arg(m, &[Some(0)], 2), p2(it), p2(ij)],
            Some(Monoid::sum()),
            move |v| {
                vec![sel(
                    vis(v[2], v[3]),
                    (v[0] * lit(scale) - v[1]).exp(),
                    lit(0.0),
                )]
            },
        )
        .unwrap();
    let sp = [T, Dim::Fixed(dh), T];
    let (it3, ij3) = (g.iota(&sp, 0), g.iota(&sp, 2));
    let id3 = [Some(0), Some(1), Some(2)];
    let y = g
        .comp(
            &[T, Dim::Fixed(dh)],
            &[T],
            vec![
                pick_arg(s, &[Some(0), Some(2)], 3),
                pick_arg(m, &[Some(0)], 3),
                pick_arg(l, &[Some(0)], 3),
                pick_arg(v2, &[Some(2), Some(1)], 3),
                pick_arg(it3, &id3, 3),
                pick_arg(ij3, &id3, 3),
            ],
            Some(Monoid::sum()),
            move |v| {
                let pv = (v[0] * lit(scale) - v[1]).exp() * v[3] / v[2];
                vec![sel(vis(v[4], v[5]), pv, lit(0.0))]
            },
        )
        .unwrap();
    let outs = [y];
    let step = derive_step_with(&g, &outs, &spec(1), &StepOptions::default()).unwrap();
    let rows_n = step.caches.iter().filter(|c| !c.state).count();
    let states_n = step.caches.iter().filter(|c| c.state).count();
    assert_eq!((rows_n, states_n), (2, 1), "K/V rows + the scan state");

    let tmax = 13;
    let cap = 16;
    let mut shapes: Vec<(usize, Vec<Dim>)> = g
        .nodes()
        .iter()
        .filter_map(|nd| match nd.op {
            Op::Input(i) => Some((i, nd.shape.clone())),
            _ => None,
        })
        .collect();
    shapes.sort_by_key(|x| x.0);
    let data: Vec<Vec<f64>> = shapes
        .iter()
        .map(|(i, sh)| {
            let n: usize = sh
                .iter()
                .map(|&d| if d == T { tmax } else { d.n() })
                .product();
            rand(n, 53 + *i as u64, 0.8)
        })
        .collect();
    let inputs_at = |t: usize| {
        let mut ins = data.clone();
        ins[0] = data[0][..t * dm].to_vec();
        ins
    };
    let conc = |sh: &[Dim], t: usize| -> Vec<usize> {
        sh.iter().map(|&d| if d == T { t } else { d.n() }).collect()
    };
    let prefill =
        |t: usize, ids: &[TensorId]| g.bind(&[("T", t)]).unwrap().eval(&inputs_at(t), ids);
    for ks in [&[1usize][..], &[4], &[4, 1, 2]] {
        let mut pos = 5;
        let mut caches: HashMap<TensorId, Vec<f64>> = HashMap::new();
        let mut worst = 0.0f64;
        for &k in ks {
            let step = derive_step_with(&g, &outs, &spec(k), &StepOptions::default()).unwrap();
            if caches.is_empty() {
                let nodes: Vec<TensorId> = step.caches.iter().map(|c| c.node).collect();
                let vals = prefill(pos, &nodes);
                for (c, v) in step.caches.iter().zip(vals) {
                    let shape = conc(g.shape(c.node), pos);
                    let buf = if c.state {
                        rows(&v, &shape, c.axis, pos - 1, pos)
                    } else {
                        let inner: usize = shape[c.axis + 1..].iter().product();
                        let outer: usize = shape[..c.axis].iter().product();
                        let mut buf = vec![f64::NAN; conc(g.shape(c.node), cap).iter().product()];
                        for o in 0..outer {
                            for r in 0..pos {
                                let s0 = (o * pos + r) * inner;
                                let d0 = (o * cap + r) * inner;
                                buf[d0..d0 + inner].copy_from_slice(&v[s0..s0 + inner]);
                            }
                        }
                        buf
                    };
                    caches.insert(c.node, buf);
                }
            }
            let ins: Vec<Vec<f64>> = step
                .inputs
                .iter()
                .map(|si| match *si {
                    StepInput::Pos => vec![pos as f64],
                    StepInput::New(_) => data[0][pos * dm..(pos + k) * dm].to_vec(),
                    StepInput::Same(i) => data[i].clone(),
                    StepInput::Cache(c) => caches[&step.caches[c].node].clone(),
                })
                .collect();
            let mut ids = step.outputs.clone();
            ids.extend(step.caches.iter().map(|c| c.updated));
            let got = step.graph.eval(&ins, &ids);
            let grown = pos + k;
            let mut rids = vec![y];
            rids.extend(step.caches.iter().map(|c| c.node));
            let want = prefill(grown, &rids);
            worst = worst.max(max_err(
                &got[0],
                &rows(&want[0], &[grown, dh], 0, pos, grown),
            ));
            for (ci, c) in step.caches.iter().enumerate() {
                let (gv, wv) = (&got[1 + ci], &want[1 + ci]);
                let wshape = conc(g.shape(c.node), grown);
                let e = if c.state {
                    max_err(gv, &rows(wv, &wshape, c.axis, grown - 1, grown))
                } else {
                    max_err(
                        &rows(gv, &conc(g.shape(c.node), cap), c.axis, 0, grown),
                        &rows(wv, &wshape, c.axis, 0, grown),
                    )
                };
                worst = worst.max(e);
                caches.insert(c.node, gv.clone());
            }
            pos = grown;
        }
        eprintln!("scan rows -> cached attention, ks={ks:?}: max rel err {worst:.2e}");
        assert!(worst < 1e-12, "{ks:?}: {worst}");
    }
}
