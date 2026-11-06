//! Gather views: data-dependent addressing as a comprehension read.

use super::*;

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "[{i}] {x} vs {y}");
    }
}

fn ramp(n: usize, scale: f64, off: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 7919) % 97) as f64 * scale + off)
        .collect()
}

#[test]
fn embedding_lookup() {
    let (v, d, t) = (7, 3, 5);
    let mut g = TensorGraph::new();
    let table = g.input(&[v, d]);
    let ids = g.input(&[t]);
    let out = g.gather(table, 0, ids, false).unwrap();
    assert_eq!(g.shape(out), &[Dim::Fixed(t), Dim::Fixed(d)]);
    let tv = ramp(v * d, 0.1, -1.0);
    let iv = vec![3.0, 0.0, 6.0, 3.0, 1.0];
    let got = &g.eval(&[tv.clone(), iv.clone()], &[out])[0];
    let want: Vec<f64> = (0..t)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .map(|(i, j)| tv[iv[i] as usize * d + j])
        .collect();
    close(got, &want, 0.0);
}

#[test]
fn guarded_gather_pads_with_zero() {
    let mut g = TensorGraph::new();
    let table = g.input(&[4, 2]);
    let ids = g.input(&[3]);
    let out = g.gather(table, 0, ids, true).unwrap();
    let tv = ramp(8, 1.0, 1.0);
    let got = &g.eval(&[tv.clone(), vec![2.0, -1.0, 4.0]], &[out])[0];
    close(got, &[tv[4], tv[5], 0.0, 0.0, 0.0, 0.0], 0.0);
}

#[test]
#[should_panic] // "unguarded read …", rethrown by the eval thread scope
fn unguarded_out_of_range_is_caught() {
    let mut g = TensorGraph::new();
    let table = g.input(&[4, 2]);
    let ids = g.input(&[1]);
    let out = g.gather(table, 0, ids, false).unwrap();
    g.eval(&[vec![0.0; 8], vec![9.0]], &[out]);
}

#[test]
fn gather_must_name_an_affine_sibling() {
    let mut g = TensorGraph::new();
    let table = g.input(&[4]);
    let ids = g.input(&[2]);
    let bad = vec![Arg {
        src: table,
        view: View::pick(1, &[None]).gather(0, 5, 1),
    }];
    assert!(g.comp(&[2], &[], bad, None, |v| vec![v[0]]).is_err());
    let nested = vec![
        Arg {
            src: table,
            view: View::pick(1, &[None]).gather(0, 1, 1),
        },
        Arg {
            src: ids,
            view: View::pick(1, &[None]).gather(0, 0, 1),
        },
    ];
    assert!(g.comp(&[2], &[], nested, None, |v| vec![v[0]]).is_err());
}

/// Paged KV: `cache[blocks, bs, d]`, `table[seq, nblk]` of block ids;
/// the logical row `t = blk·bs + off` of sequence `s` is
/// `cache[table[s, blk], off, :]`. One gathered dim, two affine ones.
#[test]
fn paged_kv_read_and_scores() {
    let (blocks, bs, d, seqs, nblk) = (6, 4, 3, 2, 3);
    let mut g = TensorGraph::new();
    let cache = g.input(&[blocks, bs, d]);
    let table = g.input(&[seqs, nblk]);
    let q = g.input(&[seqs, d]);
    // scores[s, blk, off] = Σ_c q[s, c] · cache[table[s, blk], off, c]
    let args = vec![
        Arg {
            src: cache,
            view: View::pick(4, &[None, Some(2), Some(3)]).gather(0, 1, 1),
        },
        Arg {
            src: table,
            view: View::pick(4, &[Some(0), Some(1)]),
        },
        Arg {
            src: q,
            view: View::pick(4, &[Some(0), Some(3)]),
        },
    ];
    let scores = g
        .comp(&[seqs, nblk, bs], &[d], args, Some(Monoid::sum()), |v| {
            vec![v[0] * v[2]]
        })
        .unwrap();
    let cv = ramp(blocks * bs * d, 0.05, -2.0);
    let tv = vec![5.0, 0.0, 2.0, 1.0, 4.0, 3.0];
    let qv = ramp(seqs * d, 0.3, -1.0);
    let got = &g.eval(&[cv.clone(), tv.clone(), qv.clone()], &[scores])[0];
    let mut want = Vec::new();
    for s in 0..seqs {
        for b in 0..nblk {
            let blk = tv[s * nblk + b] as usize;
            for o in 0..bs {
                want.push(
                    (0..d)
                        .map(|c| qv[s * d + c] * cv[(blk * bs + o) * d + c])
                        .sum(),
                );
            }
        }
    }
    close(got, &want, 1e-12);
}

/// A map read through a gather fuses (the gather is pushed into the map's
/// args); a gather read through a map fuses (the map carries the gather
/// and its index arg); the index tensor itself stays a buffer.
#[test]
fn substitute_fuses_through_gathers() {
    let (v, d, t) = (5, 4, 3);
    let mut g = TensorGraph::new();
    let table = g.input(&[v, d]);
    let ids_raw = g.input(&[t]);
    let w = g.input(&[d, 2]);
    // computed index: still a buffer after substitution
    let ids = g.map(&[ids_raw], |x| x[0] + ExprId::from_f64(1.0)).unwrap();
    let scaled = g.map(&[table], |x| x[0] * ExprId::from_f64(2.0)).unwrap();
    let emb = g.gather(scaled, 0, ids, false).unwrap();
    let act = g.map(&[emb], |x| x[0] * x[0]).unwrap();
    let y = g.matmul(act, w).unwrap();
    let inputs = vec![
        ramp(v * d, 0.1, -0.5),
        vec![0.0, 3.0, 1.0],
        ramp(d * 2, 0.2, -0.7),
    ];
    let want = g.eval(&inputs, &[y]);
    let (h, outs) = g.substitute_maps_with(&[y], false);
    // the matmul (with scale, gather and square spliced in) and the index map
    assert_eq!(h.count_comps(&outs), 2);
    let mm = h.comp_at(outs[0].n()).unwrap();
    assert!(mm.has_gather());
    close(&h.eval(&inputs, &outs)[0], &want[0], 1e-12);
}

/// A gathering map read through another gather is a nested gather: it
/// stays materialized, and the result is unchanged.
#[test]
fn nested_gather_is_materialized() {
    let mut g = TensorGraph::new();
    let table = g.input(&[6]);
    let a = g.input(&[4]);
    let b = g.input(&[3]);
    let first = g.gather(table, 0, a, false).unwrap(); // table[a[i]]
    let second = g.gather(first, 0, b, false).unwrap(); // table[a[b[j]]]
    let inputs = vec![
        ramp(6, 1.0, 0.0),
        vec![5.0, 2.0, 0.0, 4.0],
        vec![3.0, 1.0, 1.0],
    ];
    let want = g.eval(&inputs, &[second]);
    close(&want[0], &[inputs[0][4], inputs[0][2], inputs[0][2]], 0.0);
    let (h, outs) = g.substitute_maps_with(&[second], false);
    assert_eq!(h.count_comps(&outs), 2);
    close(&h.eval(&inputs, &outs)[0], &want[0], 0.0);
}

/// `d table[v, :] = Σ_t [ids[t] == v] · dy[t, :]`: repeated ids add up,
/// padded (-1, guarded) ids contribute nothing.
#[test]
fn backward_is_a_scatter_add() {
    let (v, d, t) = (5, 3, 6);
    let mut g = TensorGraph::new();
    let table = g.input(&[v, d]);
    let ids = g.input(&[t]);
    let wt = g.input(&[t, d]);
    let emb = g.gather(table, 0, ids, true).unwrap();
    let prod = g.map(&[emb, wt], |x| x[0] * x[1]).unwrap();
    let r1 = g.reduce(prod, 1, Monoid::sum()).unwrap();
    let loss = g.reduce(r1, 0, Monoid::sum()).unwrap();
    let grads = g.backward(loss, &[table, wt]).unwrap();
    let tv = ramp(v * d, 0.1, -1.0);
    let iv = vec![1.0, 4.0, 1.0, -1.0, 0.0, 1.0];
    let wv = ramp(t * d, 0.2, -0.3);
    let got = g.eval(&[tv.clone(), iv.clone(), wv.clone()], &grads);
    let mut dtable = vec![0.0; v * d];
    let mut dw = vec![0.0; t * d];
    for i in 0..t {
        if iv[i] < 0.0 {
            continue;
        }
        let r = iv[i] as usize;
        for j in 0..d {
            dtable[r * d + j] += wv[i * d + j];
            dw[i * d + j] = tv[r * d + j];
        }
    }
    close(&got[0], &dtable, 1e-12);
    close(&got[1], &dw, 1e-12);
    // and it survives the optimizer pipeline
    let (h, outs) = g.substitute_maps(&grads);
    close(&h.eval(&[tv, iv, wv], &outs)[0], &dtable, 1e-12);
}

/// Gradient through a paged (scaled + offset) gather: the gathered dim's
/// coordinate is `bs · table[blk] + off`.
#[test]
fn backward_through_scaled_gather() {
    let (rows, bs, nblk) = (8, 2, 3);
    let mut g = TensorGraph::new();
    let x = g.input(&[rows]);
    let table = g.input(&[nblk]);
    let args = vec![
        Arg {
            src: x,
            view: View {
                forms: vec![Form::unit(2, 1)],
                guard: false,
                gather: None,
            }
            .gather(0, 1, bs as i64),
        },
        Arg {
            src: table,
            view: View::pick(2, &[Some(0)]),
        },
    ];
    let y = g
        .comp(&[nblk, bs], &[], args, None, |v| vec![v[0] * v[0]])
        .unwrap();
    let r = g.reduce(y, 1, Monoid::sum()).unwrap();
    let loss = g.reduce(r, 0, Monoid::sum()).unwrap();
    let dx = g.backward(loss, &[x]).unwrap();
    let xv = ramp(rows, 0.3, -1.0);
    let tv = vec![3.0, 0.0, 3.0];
    let got = &g.eval(&[xv.clone(), tv.clone()], &dx)[0];
    let mut want = vec![0.0; rows];
    for b in 0..nblk {
        for o in 0..bs {
            let row = tv[b] as usize * bs + o;
            want[row] += 2.0 * xv[row];
        }
    }
    close(got, &want, 1e-12);
}

#[test]
fn cse_tells_gathers_apart() {
    let mut g = TensorGraph::new();
    let table = g.input(&[4]);
    let ids = g.input(&[2]);
    let plain = g
        .comp(
            &[2],
            &[],
            vec![
                Arg {
                    src: table,
                    view: View::pick(1, &[None]),
                },
                Arg {
                    src: ids,
                    view: View::identity(1, 1),
                },
            ],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let gathered = g.gather(table, 0, ids, false).unwrap();
    let sum = g.map(&[plain, gathered], |v| v[0] + v[1]).unwrap();
    let inputs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![3.0, 1.0]];
    let want = g.eval(&inputs, &[sum]);
    close(&want[0], &[5.0, 3.0], 0.0);
    let (h, outs) = g.eliminate_common(&[sum]);
    close(&h.eval(&inputs, &outs)[0], &want[0], 0.0);
}

#[test]
fn wgsl_reads_the_index() {
    let mut g = TensorGraph::new();
    let table = g.input(&[7, 3]);
    let ids = g.input(&[5]);
    let out = g.gather(table, 0, ids, true).unwrap();
    let src = &g.to_wgsl(&[out]).kernels[0].source;
    assert!(src.contains("round(read_"), "{src}");
}
