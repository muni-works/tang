//! DeepSeek-style fp8 (e4m3) per-32-block quantize → dequantize written in
//! the tensor IR with `floor`, checked bit-for-bit against a direct
//! implementation (nearest of the 253 finite positive e4m3 values, ties to
//! the even mantissa; a ue8m0 power-of-two scale `2^⌈log2(amax/448)⌉`).

use tang::Scalar;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

const BLOCK: usize = 32;
const FP8_MAX: f64 = 448.0;

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

fn sel(c: ExprId, t: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(c, t, f))
}

fn log2(x: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.log2(x))
}

fn exp2(x: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.exp2(x))
}

/// `⌊log2 y⌋` for `y > 0`, corrected for `log2` rounding up to an integer.
fn ilog2(y: ExprId) -> ExprId {
    let e = log2(y).floor();
    sel(exp2(e) - y, e - lit(1.0), e)
}

/// Round to nearest, ties to even, with `floor`.
fn rne(z: ExprId) -> ExprId {
    let r = (z + lit(0.5)).floor();
    let odd = r - lit(2.0) * ((r + lit(0.5)) * lit(0.5)).floor();
    // a tie lands exactly on r − z = ½
    let tie = (r - z) - lit(0.5) + lit(1e-300);
    sel(tie, r - odd, r)
}

/// e4m3 (no inf, max 448, subnormals below 2^−6) of a scaled value `y`.
fn e4m3(y: ExprId) -> ExprId {
    let m = y.abs();
    let e = ilog2(m.max(lit(1e-300))).max(lit(-6.0));
    let step = exp2(e - lit(3.0));
    let q = (rne(m / step) * step).min(lit(FP8_MAX));
    sel(y, q, -q)
}

/// The graph: `x [nb, 32]` → per-block `amax`, scale, dequantized values.
fn graph(nb: usize) -> (TensorGraph, TensorId, TensorId) {
    let mut g = TensorGraph::new();
    let x = g.input(&[nb, BLOCK]);
    let amax = g
        .comp(
            &[nb],
            &[BLOCK],
            vec![a(x, vec![Form::unit(2, 0), Form::unit(2, 1)])],
            Some(Monoid::max()),
            |v| vec![v[0].abs()],
        )
        .unwrap();
    let u = |d| Form::unit(2, d);
    let y = g
        .comp(
            &[nb, BLOCK],
            &[] as &[usize],
            vec![a(x, vec![u(0), u(1)]), a(amax, vec![u(0)])],
            None,
            |v| {
                // ue8m0 scale 2^⌈log2(amax/448)⌉ = 2^−⌊−log2(…)⌋
                let am = v[1].max(lit(1e-30));
                let s = exp2(-(-log2(am / lit(FP8_MAX))).floor());
                vec![e4m3(v[0] / s) * s]
            },
        )
        .unwrap();
    (g, x, y)
}

fn e4m3_values() -> Vec<f64> {
    let mut v = vec![];
    for e in 0..16i32 {
        for m in 0..8i32 {
            if e == 15 && m == 7 {
                continue; // NaN
            }
            v.push(if e == 0 {
                m as f64 / 8.0 * 2f64.powi(-6)
            } else {
                (1.0 + m as f64 / 8.0) * 2f64.powi(e - 7)
            });
        }
    }
    v
}

fn direct(x: &[f64]) -> Vec<f64> {
    let vals = e4m3_values();
    x.chunks(BLOCK)
        .flat_map(|blk| {
            let amax = blk.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
            let s = (amax / FP8_MAX).log2().ceil().exp2();
            let vals = &vals;
            blk.iter().map(move |&v| {
                let y = (v / s).abs().min(FP8_MAX);
                // nearest; on a tie the even code (index) wins
                let (mut bi, mut bd) = (0, f64::INFINITY);
                for (i, &c) in vals.iter().enumerate() {
                    let d = (c - y).abs();
                    if d < bd || (d == bd && i % 2 == 0) {
                        (bi, bd) = (i, d);
                    }
                }
                vals[bi].copysign(v) * s
            })
        })
        .collect()
}

#[test]
fn fp8_per32_matches_direct() {
    let nb = 64;
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut x: Vec<f64> = (0..nb * BLOCK)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            // blocks spanning many magnitudes
            u * 10f64.powi((i / BLOCK) as i32 % 9 - 4)
        })
        .collect();
    // exact ties and edges in block 0 (scale 1 once amax = 448)
    let ties = [
        448.0,
        17.0,
        19.0,
        0.5 + 1.0 / 32.0,
        2f64.powi(-9) * 1.5,
        -0.0,
    ];
    x[..ties.len()].copy_from_slice(&ties);
    let (g, _x, y) = graph(nb);
    let got = g.eval(&[x.clone()], &[y]);
    let want = direct(&x);
    for (i, (&gv, &wv)) in got[0].iter().zip(&want).enumerate() {
        assert_eq!(gv, wv, "i={i} x={}", x[i]);
    }
    // the rounding is real: some values moved
    assert!(x.iter().zip(&want).filter(|(a, b)| a != b).count() > nb * BLOCK / 2);
}
