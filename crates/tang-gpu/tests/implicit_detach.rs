//! The implicit gradient of a Newton solve (`TensorGraph::detach`, see
//! `implicit_gradient_of_a_newton_solve` in tang-compile) on the device,
//! f32, against the f64 interpreter. Skipped without a GPU adapter.
//!
//! The graph is ribt's KL projection: Newton on the convex dual, the
//! tridiagonal Hessian solved by parallel cyclic reduction, steps clamped
//! to ±4, exp arguments capped at 80.

use tang::Scalar;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;
use tang_gpu::{Program, Runtime};

fn sh(off: i64) -> View {
    let mut f = Form::unit(1, 0);
    f.offset = off;
    View {
        forms: vec![f],
        guard: true,
        gather: None,
    }
}

/// `y[m] = Σ_k W[k,m]·x[k]`
fn to_bands(g: &mut TensorGraph, w: TensorId, x: TensorId, m: usize, k: usize) -> TensorId {
    g.comp(
        &[m],
        &[k],
        vec![
            Arg {
                src: x,
                view: View::pick(2, &[Some(1)]),
            },
            Arg {
                src: w,
                view: View::pick(2, &[Some(1), Some(0)]),
            },
        ],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

/// `y[k] = Σ_m W[k,m]·z[m]`
fn to_bins(g: &mut TensorGraph, w: TensorId, z: TensorId, m: usize, k: usize) -> TensorId {
    g.comp(
        &[k],
        &[m],
        vec![
            Arg {
                src: z,
                view: View::pick(2, &[Some(1)]),
            },
            Arg {
                src: w,
                view: View::pick(2, &[Some(0), Some(1)]),
            },
        ],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

/// Symmetric tridiagonal solve over `[M]` by parallel cyclic reduction
/// (`up[m]` couples m and m+1), as ribt's `glottis::pcr`.
fn pcr(
    g: &mut TensorGraph,
    valid: TensorId,
    up: TensorId,
    d: TensorId,
    r: TensorId,
    m: usize,
) -> TensorId {
    let id = || View::identity(1, 1);
    let mut lo = g
        .comp(
            &[m],
            &[],
            vec![Arg {
                src: up,
                view: sh(-1),
            }],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let (mut d, mut up, mut r) = (d, up, r);
    let mut s = 1i64;
    while (s as usize) < m {
        let q = [lo, d, up, r];
        let args = || {
            let mut a: Vec<Arg> = q.iter().map(|&t| Arg { src: t, view: id() }).collect();
            for o in [-s, s] {
                for &t in &q {
                    a.push(Arg {
                        src: t,
                        view: sh(o),
                    });
                }
                a.push(Arg {
                    src: valid,
                    view: sh(o),
                });
            }
            a
        };
        // v: 0 lo,1 d,2 up,3 r | 4 lo⁻,5 d⁻,6 up⁻,7 r⁻,8 valid⁻ | 9 lo⁺,10 d⁺,11 up⁺,12 r⁺,13 valid⁺
        let k = |v: &[ExprId]| {
            (
                v[0] / (v[5] + ExprId::ONE - v[8]),
                v[2] / (v[10] + ExprId::ONE - v[13]),
            )
        };
        let nlo = g
            .comp(&[m], &[], args(), None, |v| vec![-(k(v).0 * v[4])])
            .unwrap();
        let nd = g
            .comp(&[m], &[], args(), None, |v| {
                let (k1, k2) = k(v);
                vec![v[1] - k1 * v[6] - k2 * v[9]]
            })
            .unwrap();
        let nup = g
            .comp(&[m], &[], args(), None, |v| vec![-(k(v).1 * v[11])])
            .unwrap();
        let nr = g
            .comp(&[m], &[], args(), None, |v| {
                let (k1, k2) = k(v);
                vec![v[3] - k1 * v[7] - k2 * v[12]]
            })
            .unwrap();
        (lo, d, up, r) = (nlo, nd, nup, nr);
        s *= 2;
    }
    g.map(&[r, d], |v| v[0] / v[1]).unwrap()
}

/// The KL projection `s = x ⊙ exp(F λ)` with `Fᵀ s = exp(mel)`, by
/// `newton` Newton steps on the convex dual (Hessian `Fᵀ diag(s) F`,
/// tridiagonal), as ribt's `glottis::project_keep`: steps clamped to
/// ±4, exp arguments capped at 80. `implicit`: detach the iteration
/// and take one more step with a detached Hessian
/// ([`TensorGraph::detach`]). Returns `s` and the last step `Δλ`.
/// Inputs: x [K], mel [M], F, F², F·F₊ [K, M], valid [M].
fn project(
    g: &mut TensorGraph,
    m: usize,
    k: usize,
    newton: usize,
    implicit: bool,
) -> (TensorId, TensorId) {
    let x = g.input(&[k]);
    let mel = g.input(&[m]);
    let fb = g.input(&[k, m]);
    let fb2 = g.input(&[k, m]);
    let fbn = g.input(&[k, m]);
    let valid = g.input(&[m]);
    let e = g.map(&[mel], |v| v[0].exp()).unwrap();
    let fx = to_bands(g, fb, x, m, k);
    let lit = ExprId::from_f64;
    let mut lam = g
        .map(&[mel, fx], |v| v[0] - (v[1] + lit(1e-30)).ln())
        .unwrap();
    let s_of = |g: &mut TensorGraph, lam: TensorId| {
        let fl = to_bins(g, fb, lam, m, k);
        g.map(&[x, fl], |v| {
            let z = ExprId::select(v[1] - lit(80.0), lit(80.0), v[1]);
            v[0] * z.exp()
        })
        .unwrap()
    };
    let step = |g: &mut TensorGraph, lam: TensorId, detach_h: bool| {
        let s = s_of(g, lam);
        let fs = to_bands(g, fb, s, m, k);
        let mut d = to_bands(g, fb2, s, m, k);
        let mut up = to_bands(g, fbn, s, m, k);
        if detach_h {
            d = g.detach(d).unwrap();
            up = g.detach(up).unwrap();
        }
        let r = g.map(&[e, fs], |v| v[0] - v[1]).unwrap();
        let dl = pcr(g, valid, up, d, r, m);
        let next = g
            .map(&[lam, dl], |v| {
                let hi = ExprId::select(v[1] - lit(4.0), lit(4.0), v[1]);
                v[0] + ExprId::select(lit(-4.0) - v[1], lit(-4.0), hi)
            })
            .unwrap();
        (next, dl)
    };
    let mut dl = lam;
    for _ in 0..newton {
        (lam, dl) = step(g, lam, false);
    }
    if implicit {
        let fixed = g.detach(lam).unwrap();
        (lam, dl) = step(g, fixed, true);
    }
    (s_of(g, lam), dl)
}

#[test]
fn implicit_gradient_matches_interpreter() {
    let Ok(rt) = Runtime::select() else {
        eprintln!("skipped: no GPU adapter");
        return;
    };
    let (m, k, newton) = (8usize, 33usize, 8usize);
    let mut fb = vec![0.0; k * m];
    for kk in 0..k {
        for c in 0..m {
            let w = 1.0 - (kk as f64 - (4 * c + 2) as f64).abs() / 4.0;
            fb[kk * m + c] = w.max(0.0);
        }
    }
    let fb2: Vec<f64> = fb.iter().map(|w| w * w).collect();
    let fbn: Vec<f64> = (0..k * m)
        .map(|i| {
            if i % m + 1 < m {
                fb[i] * fb[i + 1]
            } else {
                0.0
            }
        })
        .collect();
    let x: Vec<f64> = (0..k)
        .map(|i| ((i * 7919) % 97) as f64 * 0.02 + 0.4)
        .collect();
    let mel: Vec<f64> = (0..m).map(|c| 0.3 * ((c * 5) % 7) as f64 - 0.5).collect();
    let ins = vec![x, mel, fb, fb2, fbn, vec![1.0; m]];

    let mut g = TensorGraph::new();
    let (s, _) = project(&mut g, m, k, newton, true);
    let wts = g.iota(&[k], 0);
    let y = g
        .map(&[s, wts], |v| v[0] * (v[1] * ExprId::from_f64(0.37)).sin())
        .unwrap();
    let grads = g
        .backward(y, &[TensorId::from_index(0), TensorId::from_index(1)])
        .unwrap();
    let outs = [y, grads[0], grads[1]];
    let want = g.eval(&ins, &outs);

    let (g, o) = g.substitute_maps(&outs);
    let (g, o) = g.eliminate_common(&o);
    let ins32: Vec<Vec<f32>> = ins
        .iter()
        .map(|v| v.iter().map(|&a| a as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = ins32.iter().map(|v| v.as_slice()).collect();
    let p = Program::compile(&rt, &g, &o, &refs);
    p.run(&rt);
    for (name, (t, w)) in ["y", "dy/dx", "dy/dmel"].iter().zip(o.iter().zip(&want)) {
        let got = p.read(&rt, *t);
        let scale = w.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let err = got
            .iter()
            .zip(w)
            .fold(0.0f64, |a, (u, v)| a.max((*u as f64 - v).abs()));
        println!(
            "{} {name}: max abs err {err:.3e}, rel to max |ref| {:.3e}",
            rt.name(),
            err / scale
        );
        assert!(err <= 1e-4 * scale, "{name}: {err} vs scale {scale}");
    }
}
