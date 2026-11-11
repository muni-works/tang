//! Top-k attention through [`derive_step`]: written as a gather of the
//! picked rows, and as a dense data mask ANDed with the causal mask. Steps
//! are checked against prefills of the grown sequence with NaN in every
//! unfilled cache row (`check_steps_with`).

use tang::Scalar;
use tang_expr::ExprId;

use super::super::{Arg, Dim, Form, Monoid, TensorGraph, TensorId, View};
use super::tests::check_steps_with;
use super::{derive_step, Rows, StepSpec};

const T: Dim = Dim::Sym("T");
const KK: usize = 2;
const DM: usize = 4;
const DH: usize = 4;

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

fn sel(c: ExprId, t: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(c, t, f))
}

fn pick(a: ExprId, b: ExprId, x: ExprId) -> ExprId {
    let z = lit(0.0);
    sel(a - b + lit(0.5), sel(b - a + lit(0.5), x, z), z)
}

fn arg(src: TensorId, view: View) -> Arg {
    Arg { src, view }
}

fn pk(r: usize, p: &[Option<usize>]) -> View {
    View::pick(r, p)
}

/// `Σ coefs·x + offset` over a rank-`r` space.
fn form(r: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::constant(r, offset);
    for &(d, k) in terms {
        f.coefs[d] = k;
    }
    f
}

/// The top-k slot `slot(s)` of `top` (`[T, 2·KK]`) in a rank-`r` space whose
/// coordinate 0 is the query row.
fn top_at(top: TensorId, r: usize, slot: Form) -> Arg {
    arg(
        top,
        View {
            forms: vec![Form::unit(r, 0), slot],
            guard: false,
            gather: None,
        },
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sparse {
    /// Attention reads the picked rows through a gather.
    Gather,
    /// Dense over all keys, `select(causal, select(member, x, id), id)`.
    Mask,
    /// Negative: the dense data mask without the causal select.
    MaskOnly,
    /// Negative: a gather whose top-k index slot is `j` unmasked (the
    /// values still are), so causality is not provable from the index.
    GatherRawIndex,
}

/// One top-k attention layer over `x [T, DM]`: indexer scores `q·k`, top
/// `KK` keys `j ≤ i`, softmax attention over them (scores `q·k / √DH`).
fn layer(mode: Sparse) -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let x = g.input(&[T, Dim::Fixed(DM)]);
    let wq = g.input(&[DM, DH]);
    let wk = g.input(&[DM, DH]);
    let wv = g.input(&[DM, DH]);
    let wi = g.input(&[DM, DH]);
    let q = g.matmul(x, wq).unwrap();
    let k = g.matmul(x, wk).unwrap();
    let v = g.matmul(x, wv).unwrap();
    let iq = g.matmul(x, wi).unwrap();
    // indexer scores [T(i), T(j)] = iq_i · k_j
    let s = g
        .comp(
            &[T, T],
            &[Dim::Fixed(DH)],
            vec![
                arg(iq, pk(3, &[Some(0), Some(2)])),
                arg(k, pk(3, &[Some(1), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let (ti, tj) = (g.iota(&[T, T], 0), g.iota(&[T, T], 1));
    let fmax = -(f32::MAX as f64);
    let raw = mode == Sparse::GatherRawIndex;
    let top = g
        .comp_tuple(
            &[T, Dim::Fixed(2 * KK)],
            &[T],
            vec![
                arg(s, View::identity(2, 2)),
                arg(ti, View::identity(2, 2)),
                arg(tj, View::identity(2, 2)),
            ],
            Some(Monoid::topk(KK)),
            move |v| {
                let causal = v[1] - v[2] + lit(0.5);
                let idx = if raw {
                    v[2]
                } else {
                    sel(causal, v[2], lit(0.0))
                };
                let mut o = vec![idx, sel(causal, v[0], lit(fmax))];
                for _ in 1..KK {
                    // padding slots, masked too (the recognizer wants every
                    // contribution as select(causal, x, identity))
                    o.push(sel(causal, lit(0.0), lit(0.0)));
                    o.push(sel(causal, lit(fmax), lit(fmax)));
                }
                o
            },
        )
        .unwrap();
    let scale = 1.0 / (DH as f64).sqrt();
    let out = match mode {
        Sparse::Gather | Sparse::GatherRawIndex => gather_attention(&mut g, q, k, v, top, scale),
        Sparse::Mask | Sparse::MaskOnly => {
            mask_attention(&mut g, q, k, v, top, scale, mode == Sparse::Mask)
        }
    };
    (g, out)
}

/// Attention over the `KK` picked rows: `k` and `v` gathered by index.
fn gather_attention(
    g: &mut TensorGraph,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    top: TensorId,
    scale: f64,
) -> TensorId {
    let kp = Dim::Fixed(KK);
    // sg[i, s] = Σ_c q[i, c]·k[idx(i, s), c]: space (i, s, c); arg 2 the index
    let kg = pk(3, &[None, Some(2)]).gather(0, 2, 1);
    let sg = g
        .comp(
            &[T, kp],
            &[Dim::Fixed(DH)],
            vec![
                arg(q, pk(3, &[Some(0), Some(2)])),
                arg(k, kg),
                top_at(top, 3, form(3, &[(1, 2)], 0)),
            ],
            Some(Monoid::sum()),
            move |v| vec![v[0] * v[1] * lit(scale)],
        )
        .unwrap();
    // validity of slot s (a real pick): value > −1e29
    let val2 = |r| top_at(top, r, form(r, &[(1, 2)], 1));
    let m = g
        .comp(
            &[T],
            &[kp],
            vec![arg(sg, View::identity(2, 2)), val2(2)],
            Some(Monoid::max()),
            |v| vec![sel(v[1] + lit(1e29), v[0], lit(-1e30))],
        )
        .unwrap();
    let p = g
        .comp(
            &[T, kp],
            &[],
            vec![
                arg(sg, View::identity(2, 2)),
                arg(m, pk(2, &[Some(0)])),
                val2(2),
            ],
            None,
            |v| vec![sel(v[2] + lit(1e29), (v[0] - v[1]).exp(), lit(0.0))],
        )
        .unwrap();
    let l = g.reduce(p, 1, Monoid::sum()).unwrap();
    // o[i, c] = Σ_s p[i, s]·v[idx(i, s), c] / l[i]: space (i, c, s)
    let vg = pk(3, &[None, Some(1)]).gather(0, 2, 1);
    g.comp(
        &[T, Dim::Fixed(DH)],
        &[kp],
        vec![
            arg(p, pk(3, &[Some(0), Some(2)])),
            arg(v, vg),
            top_at(top, 3, form(3, &[(2, 2)], 0)),
            arg(l, pk(3, &[Some(0), None])),
        ],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1] / v[3]],
    )
    .unwrap()
}

/// Dense attention over every key `j`, visible iff picked (and `j ≤ i`
/// when `causal`).
fn mask_attention(
    g: &mut TensorGraph,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    top: TensorId,
    scale: f64,
    causal: bool,
) -> TensorId {
    // scores [T(i), T(j)]
    let sc = g
        .comp(
            &[T, T],
            &[Dim::Fixed(DH)],
            vec![
                arg(q, pk(3, &[Some(0), Some(2)])),
                arg(k, pk(3, &[Some(1), Some(2)])),
            ],
            Some(Monoid::sum()),
            move |v| vec![v[0] * v[1] * lit(scale)],
        )
        .unwrap();
    // the args every masked reduction appends (space with i = 0, j = last):
    // iota i, iota j, then the 2·KK top-k slots of row i
    let vis = |g: &mut TensorGraph, sp: &[Dim]| -> Vec<Arg> {
        let r = sp.len();
        let mut a = vec![
            arg(g.iota(sp, 0), View::identity(r, r)),
            arg(g.iota(sp, r - 1), View::identity(r, r)),
        ];
        a.extend((0..2 * KK).map(|s| top_at(top, r, Form::constant(r, s as i64))));
        a
    };
    // visible(v, at) wraps x: causal select outside, membership inside
    let wrap = move |v: &[ExprId], at: usize, x: ExprId, id: f64| {
        let (i, j) = (v[at], v[at + 1]);
        let mut member = lit(-0.5);
        for s in 0..KK {
            let (ix, val) = (v[at + 2 + 2 * s], v[at + 3 + 2 * s]);
            member = member + sel(val + lit(1e29), pick(ix, j, lit(1.0)), lit(0.0));
        }
        let inner = sel(member, x, lit(id));
        if causal {
            sel(i - j + lit(0.5), inner, lit(id))
        } else {
            inner
        }
    };
    let sp = [T, T];
    let mut a = vec![arg(sc, View::identity(2, 2))];
    a.extend(vis(g, &sp));
    let m = g
        .comp(&[T], &[T], a, Some(Monoid::max()), move |v| {
            vec![wrap(v, 1, v[0], -1e30)]
        })
        .unwrap();
    let mut a = vec![arg(sc, View::identity(2, 2)), arg(m, pk(2, &[Some(0)]))];
    a.extend(vis(g, &sp));
    let l = g
        .comp(&[T], &[T], a, Some(Monoid::sum()), move |v| {
            vec![wrap(v, 2, (v[0] - v[1]).exp(), 0.0)]
        })
        .unwrap();
    // o[i, c] over j: space (i, c, j)
    let sp = [T, Dim::Fixed(DH), T];
    let mut a = vec![
        arg(sc, pk(3, &[Some(0), Some(2)])),
        arg(m, pk(3, &[Some(0)])),
        arg(l, pk(3, &[Some(0)])),
        arg(v, pk(3, &[Some(2), Some(1)])),
    ];
    a.extend(vis(g, &sp));
    g.comp(
        &[T, Dim::Fixed(DH)],
        &[T],
        a,
        Some(Monoid::sum()),
        move |v| vec![wrap(v, 4, (v[0] - v[1]).exp() / v[2] * v[3], 0.0)],
    )
    .unwrap()
}

fn spec(k: usize) -> StepSpec {
    StepSpec {
        seq: "T",
        new: k,
        capacity: 16,
    }
}

fn check(g: &TensorGraph, o: TensorId, p0: usize, ks: &[usize], skew: usize) -> f64 {
    check_steps_with(g, &[o], p0, ks, (skew, 0), Rows::All)
}

#[test]
fn topk_attention_steps_match_prefill() {
    for mode in [Sparse::Gather, Sparse::Mask] {
        let (g, o) = layer(mode);
        let step = derive_step(&g, &[o], &spec(1)).unwrap();
        let caches: Vec<String> = step
            .caches
            .iter()
            .map(|c| format!("{:?} {:?}", c.node, g.shape(c.node)))
            .collect();
        eprintln!("top-k {mode:?}: caches {caches:?}");
        // k and v (k also feeds the indexer scores)
        assert_eq!(step.caches.len(), 2, "{caches:?}");
        // p0 = 1: fewer than KK candidates, padding slots must stay masked
        for (p0, ks) in [
            (5, &[1usize][..]),
            (5, &[4]),
            (3, &[4, 1, 1, 2]),
            (1, &[1, 1]),
        ] {
            let e = check(&g, o, p0, ks, 0);
            eprintln!("top-k {mode:?} p0={p0} ks={ks:?}: max rel err {e:.2e}");
            assert!(e < 1e-12, "{mode:?} {ks:?}: {e}");
        }
        // last-row outputs too
        let e = check_steps_with(&g, &[o], 5, &[4, 1], (0, 0), Rows::Last);
        assert!(e < 1e-12, "{mode:?} last rows: {e}");
    }
}

/// The gather and the dense mask are the same function (prefill).
#[test]
fn gather_equals_dense_mask() {
    let (gg, og) = layer(Sparse::Gather);
    let (gm, om) = layer(Sparse::Mask);
    let t = 9;
    let ins: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            let n = if i == 0 { t * DM } else { DM * DH };
            (0..n)
                .map(|j| ((j * 7 + i * 13) % 11) as f64 / 11.0 - 0.5)
                .collect()
        })
        .collect();
    let a = gg.bind(&[("T", t)]).unwrap().eval(&ins, &[og]);
    let b = gm.bind(&[("T", t)]).unwrap().eval(&ins, &[om]);
    let e = a[0]
        .iter()
        .zip(&b[0])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    eprintln!("gather vs dense mask, T = {t}: max abs diff {e:.2e}");
    assert!(e < 1e-12, "{e}");
}

/// Negative controls: a wrong position fails the comparison; a data mask
/// without the causal select, and a gather whose index is not provably
/// causal, are refused.
#[test]
fn topk_negative_controls() {
    for mode in [Sparse::Gather, Sparse::Mask] {
        let (g, o) = layer(mode);
        let e = check(&g, o, 5, &[1], 1);
        eprintln!("top-k {mode:?} skewed position: max rel err {e:.2e}");
        assert!(e > 1e-3, "{mode:?}: {e}");
    }
    for mode in [Sparse::MaskOnly, Sparse::GatherRawIndex] {
        let (g, o) = layer(mode);
        let r = derive_step(&g, &[o], &spec(1));
        let msg = r.err().expect("must be refused").0;
        eprintln!("top-k {mode:?}: {msg}");
        assert!(msg.contains("causal"), "{msg}");
    }
}

const B: Dim = Dim::Sym("B");

fn bproj(g: &mut TensorGraph, x: TensorId, a: usize, n: usize) -> TensorId {
    let w = g.input(&[a, n]);
    g.comp(
        &[B, T, Dim::Fixed(n)],
        &[Dim::Fixed(a)],
        vec![
            arg(x, pk(4, &[Some(0), Some(1), Some(3)])),
            arg(w, pk(4, &[Some(3), Some(2)])),
        ],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

/// The gather layer over independent sessions `x [B, T, DM]`.
fn batched_layer() -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let x = g.input(&[B, T, Dim::Fixed(DM)]);
    let q = bproj(&mut g, x, DM, DH);
    let k = bproj(&mut g, x, DM, DH);
    let v = bproj(&mut g, x, DM, DH);
    let iq = bproj(&mut g, x, DM, DH);
    let s = g
        .comp(
            &[B, T, T],
            &[Dim::Fixed(DH)],
            vec![
                arg(iq, pk(4, &[Some(0), Some(1), Some(3)])),
                arg(k, pk(4, &[Some(0), Some(2), Some(3)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let (ti, tj) = (g.iota(&[B, T, T], 1), g.iota(&[B, T, T], 2));
    let fmax = -(f32::MAX as f64);
    let id3 = || View::identity(3, 3);
    let top = g
        .comp_tuple(
            &[B, T, Dim::Fixed(2 * KK)],
            &[T],
            vec![arg(s, id3()), arg(ti, id3()), arg(tj, id3())],
            Some(Monoid::topk(KK)),
            move |v| {
                let causal = v[1] - v[2] + lit(0.5);
                let mut o = vec![sel(causal, v[2], lit(0.0)), sel(causal, v[0], lit(fmax))];
                for _ in 1..KK {
                    o.push(sel(causal, lit(0.0), lit(0.0)));
                    o.push(sel(causal, lit(fmax), lit(fmax)));
                }
                o
            },
        )
        .unwrap();
    // slot form `2·s + off` of top in a rank-r space (b = 0, i = 1)
    let slot = |r: usize, sd: usize, off: i64| {
        arg(
            top,
            View {
                forms: vec![Form::unit(r, 0), Form::unit(r, 1), form(r, &[(sd, 2)], off)],
                guard: false,
                gather: None,
            },
        )
    };
    // source row idx of session b: forms (b, ·, c), gathered on dim 1 by arg `ix`
    let gat = |src: TensorId, r: usize, c: usize, ix: usize| {
        arg(
            src,
            View {
                forms: vec![Form::unit(r, 0), Form::zero(r), Form::unit(r, c)],
                guard: false,
                gather: None,
            }
            .gather(1, ix, 1),
        )
    };
    let kp = Dim::Fixed(KK);
    let scale = 1.0 / (DH as f64).sqrt();
    // sg[b, i, s]: space (b, i, s, c)
    let sg = g
        .comp(
            &[B, T, kp],
            &[Dim::Fixed(DH)],
            vec![
                arg(q, pk(4, &[Some(0), Some(1), Some(3)])),
                gat(k, 4, 3, 2),
                slot(4, 2, 0),
            ],
            Some(Monoid::sum()),
            move |v| vec![v[0] * v[1] * lit(scale)],
        )
        .unwrap();
    let m = g
        .comp(
            &[B, T],
            &[kp],
            vec![arg(sg, id3()), slot(3, 2, 1)],
            Some(Monoid::max()),
            |v| vec![sel(v[1] + lit(1e29), v[0], lit(-1e30))],
        )
        .unwrap();
    let l = g
        .comp(
            &[B, T],
            &[kp],
            vec![
                arg(sg, id3()),
                arg(m, pk(3, &[Some(0), Some(1)])),
                slot(3, 2, 1),
            ],
            Some(Monoid::sum()),
            |v| vec![sel(v[2] + lit(1e29), (v[0] - v[1]).exp(), lit(0.0))],
        )
        .unwrap();
    // o[b, i, c]: space (b, i, c, s)
    let o = g
        .comp(
            &[B, T, Dim::Fixed(DH)],
            &[kp],
            vec![
                arg(sg, pk(4, &[Some(0), Some(1), Some(3)])),
                arg(m, pk(4, &[Some(0), Some(1)])),
                arg(l, pk(4, &[Some(0), Some(1)])),
                gat(v, 4, 2, 4),
                slot(4, 3, 0),
                slot(4, 3, 1),
            ],
            Some(Monoid::sum()),
            |v| {
                vec![sel(
                    v[5] + lit(1e29),
                    (v[0] - v[1]).exp() / v[2] * v[3],
                    lit(0.0),
                )]
            },
        )
        .unwrap();
    (g, o)
}

/// Sessions at lengths 3, 6 and 1 stepped together through the gather
/// layer (`p0` is `[B]`): each equals its session stepped alone, bit for
/// bit, and the grown session's prefill. Steps of 4, then 1.
#[test]
fn batched_gather_ragged_sessions() {
    use super::super::Op;
    use super::{derive_step_with, StepInput, StepOptions};
    use std::collections::HashMap;
    let (g, o) = batched_layer();
    let (ps, ks, cap, tmax) = ([3usize, 6, 1], [4usize, 1], 16usize, 12usize);
    let nb = ps.len();
    let n_in = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input(_)))
        .count();
    let val = |n: usize, seed: usize| -> Vec<f64> {
        (0..n)
            .map(|j| (((j * 7919 + seed * 104729) % 1009) as f64 / 1009.0 - 0.5) * 1.6)
            .collect()
    };
    let xs: Vec<Vec<f64>> = (0..nb).map(|b| val(tmax * DM, 100 + b)).collect();
    let mut weights: Vec<Vec<f64>> = (0..n_in).map(|i| val(DM * DH, 17 + i)).collect();
    weights[0] = Vec::new();
    let x_rows = |b: usize, r0: usize, r1: usize| xs[b][r0 * DM..r1 * DM].to_vec();
    let prefill = |b: usize, t: usize, ids: &[TensorId]| {
        let mut ins = weights.clone();
        ins[0] = x_rows(b, 0, t);
        g.bind(&[("B", 1), ("T", t)]).unwrap().eval(&ins, ids)
    };
    // per session: cache buffers [1, cap, DH], NaN past the filled rows
    let mut caches: Vec<HashMap<TensorId, Vec<f64>>> = vec![HashMap::new(); nb];
    let mut pos = ps;
    let mut worst = 0.0f64;
    let opts = StepOptions {
        rows: Rows::All,
        batch: Some("B"),
    };
    for &k in &ks {
        let step = derive_step_with(&g, &[o], &spec(k), &opts).unwrap();
        assert_eq!(step.caches.len(), 2, "k and v");
        if caches[0].is_empty() {
            let nodes: Vec<TensorId> = step.caches.iter().map(|c| c.node).collect();
            for b in 0..nb {
                for (c, v) in step.caches.iter().zip(prefill(b, pos[b], &nodes)) {
                    let mut buf = vec![f64::NAN; cap * DH];
                    buf[..v.len()].copy_from_slice(&v);
                    caches[b].insert(c.node, buf);
                }
            }
        }
        let mut ids = step.outputs.clone();
        ids.extend(step.caches.iter().map(|c| c.updated));
        let inputs_for = |ss: &[usize]| -> Vec<Vec<f64>> {
            step.inputs
                .iter()
                .map(|si| match *si {
                    StepInput::Pos => ss.iter().map(|&b| pos[b] as f64).collect(),
                    StepInput::New(_) => ss
                        .iter()
                        .flat_map(|&b| x_rows(b, pos[b], pos[b] + k))
                        .collect(),
                    StepInput::Same(i) => weights[i].clone(),
                    StepInput::Cache(c) => ss
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
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut alones = Vec::new();
        for b in 0..nb {
            let alone = one.eval(&inputs_for(&[b]), &ids);
            for (t, a) in together.iter().zip(&alone) {
                let per = t.len() / nb;
                assert_eq!(bits(&t[b * per..(b + 1) * per]), bits(a), "session {b}");
            }
            let want = prefill(b, pos[b] + k, &[o]);
            let w = &want[0][pos[b] * DH..(pos[b] + k) * DH];
            let e = alone[0]
                .iter()
                .zip(w)
                .map(|(x, y)| {
                    let e = (x - y).abs() / (1.0 + y.abs());
                    if e.is_finite() {
                        e
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(0.0, f64::max);
            worst = worst.max(e);
            alones.push(alone);
        }
        for (b, alone) in alones.into_iter().enumerate() {
            for (ci, c) in step.caches.iter().enumerate() {
                caches[b].insert(c.node, alone[1 + ci].clone());
            }
        }
        for p in pos.iter_mut() {
            *p += k;
        }
    }
    eprintln!("batched gather, sessions {ps:?}, steps {ks:?}: together == alone bit for bit; vs prefill {worst:.2e}");
    assert!(worst < 1e-12, "{worst}");
}
