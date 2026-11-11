//! Step derivation over bodies that use `floor`: a causal mask stays the
//! affine `select(i − j + ½, x, 0)`; a mask that is only expressible with
//! `floor` is refused (not mis-derived), and a `floor`/`mod` condition
//! *inside* an affine causal select is just part of the masked value.

use tang::Scalar;
use tang_compile::tensor::step::{derive_step, StepSpec};
use tang_compile::tensor::{Arg, Dim, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

const T: Dim = Dim::Sym("T");

fn a(src: TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View {
            gather: None,
            guard: false,
            forms,
        },
    }
}

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

fn select(c: ExprId, t: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(c, t, f))
}

fn imod(x: ExprId, r: f64) -> ExprId {
    tang_expr::with_graph(|g| {
        let rl = g.lit(r);
        g.imod(x, rl)
    })
}

/// `y[i] = Σ_j body(x[j], i, j)` over `[T]`, derived for 1 new token.
fn derive(body: fn(ExprId, ExprId, ExprId) -> ExprId) -> bool {
    let mut g = TensorGraph::new();
    let x = g.input(&[T]);
    let (i, j) = (g.iota(&[T, T], 0), g.iota(&[T, T], 1));
    let u = |d| Form::unit(2, d);
    let y = g
        .comp(
            &[T],
            &[T],
            vec![
                a(x, vec![u(1)]),
                a(i, vec![u(0), u(1)]),
                a(j, vec![u(0), u(1)]),
            ],
            Some(Monoid::sum()),
            |v| vec![body(v[0], v[1], v[2])],
        )
        .unwrap();
    let spec = StepSpec {
        seq: "T",
        new: 1,
        capacity: 16,
    };
    derive_step(&g, &[y], &spec).is_ok()
}

#[test]
fn affine_causal_mask_derives() {
    assert!(derive(|x, i, j| select(i - j + lit(0.5), x, lit(0.0))));
}

#[test]
fn floor_inside_a_causal_select_derives() {
    // only group ends ((j + 1) mod 4 == 0) contribute, causally
    assert!(derive(|x, i, j| {
        let end = lit(0.5) - imod(j + lit(1.0), 4.0);
        select(i - j + lit(0.5), select(end, x, lit(0.0)), lit(0.0))
    }));
}

#[test]
fn floor_only_mask_is_refused() {
    // j ≤ i written as floor(i + ½) − j + ½: causal in value, but the
    // checker reads only affine conditions, so it refuses
    assert!(!derive(|x, i, j| {
        let q = tang_expr::with_graph(|g| {
            let h = g.lit(0.5);
            let d = g.add(i, h);
            g.floor(d)
        });
        select(q - j + lit(0.5), x, lit(0.0))
    }));
}

/// `[t mod 3, floor(t / 32)]` over a fixed axis: the tensor interpreter is
/// exact, and every tensor emitter writes `floor`. With `FLOOR_CUDA_DUMP`
/// set, the CUDA kernels are written there (for an NVRTC compile check).
#[test]
fn tensor_backends() {
    let n = 1000;
    let mut g = TensorGraph::new();
    let t = g.iota(&[n], 0);
    let y = g
        .map(&[t], |v| {
            let m = imod(v[0], 3.0);
            let d = ((v[0] + lit(0.5)) * lit(1.0 / 32.0)).floor();
            m * lit(1000.0) + d
        })
        .unwrap();
    let got = g.eval(&[], &[y]);
    for (i, &v) in got[0].iter().enumerate() {
        assert_eq!(v, ((i % 3) * 1000 + i / 32) as f64, "t={i}");
    }
    let c = g.to_c(&[y]);
    assert!(c.source.contains("floorf("), "{}", c.source);
    let w: String = g
        .to_wgsl(&[y])
        .kernels
        .iter()
        .map(|k| k.source.clone())
        .collect();
    assert!(w.contains("floor("), "{w}");
    let cu: String = g
        .to_cuda(&[y])
        .kernels
        .iter()
        .map(|k| k.source.clone())
        .collect();
    assert!(cu.contains("floor"), "{cu}");
    if let Ok(dir) = std::env::var("FLOOR_CUDA_DUMP") {
        for (i, k) in g.to_cuda(&[y]).kernels.iter().enumerate() {
            std::fs::write(format!("{dir}/tensor{i}.cu"), &k.source).unwrap();
        }
    }
}
