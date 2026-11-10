//! The discovery catalog: the operator zoo run through
//! [`online_lift`], [`scan_lift`] and [`chunk_lift`]. `docs/discovery.md`
//! is generated from [`catalog_markdown`] plus the literature notes.

#![allow(
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::type_complexity,
    clippy::neg_cmp_op_on_partial_ord
)]

use tang::Scalar;
use tang_expr::ExprId;

use super::discover::*;
use super::Monoid;

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Componentwise sums of width `k`.
pub fn sums(k: usize) -> Monoid {
    Monoid::build(k, vec![0.0; k], |a, e| {
        a.iter().zip(e).map(|(&x, &y)| x + y).collect()
    })
}

/// Product monoid (state = concatenation).
pub fn prod(ms: &[Monoid]) -> Monoid {
    let w: usize = ms.iter().map(|m| m.width).sum();
    let id: Vec<f64> = ms.iter().flat_map(|m| m.identity.clone()).collect();
    let ms = ms.to_vec();
    Monoid::build(w, id, move |a, e| {
        let mut out = vec![];
        let mut o = 0;
        for m in &ms {
            let mut v = a[o..o + m.width].to_vec();
            v.extend_from_slice(&e[o..o + m.width]);
            out.extend(inline(&m.combine, &m.outs, &v));
            o += m.width;
        }
        out
    })
}

fn c(v: f64) -> ExprId {
    lit(v)
}

fn sigmoid(x: ExprId) -> ExprId {
    c(1.0) / (c(1.0) + eexp(-x))
}

fn absx(x: ExprId) -> ExprId {
    sel(x, x, -x)
}

/// `occupied` for a max slot (`> 0` unless `−∞`).
fn occ_max(i: usize) -> Build {
    build(move |s| vec![s[i] + c(1e30)])
}

/// `occupied` for a count slot.
fn occ_count(i: usize) -> Build {
    build(move |s| vec![s[i]])
}

fn normals(k: usize, scale: f64) -> Sampler {
    sampler(move |r| (0..k).map(|_| scale * r.normal()).collect())
}

/// Online spec over a max-of-scores first pass: `x = (s, extra…)`,
/// `score(x)` the logit, `g(x, m)`.
fn softmax_spec(
    name: &str,
    n_x: usize,
    score: impl Fn(&[ExprId]) -> ExprId + 'static,
    g: impl Fn(&[ExprId], ExprId) -> Vec<ExprId> + 'static,
    sample: Sampler,
) -> OnlineSpec {
    OnlineSpec {
        name: name.into(),
        n_x,
        r1: Monoid::max(),
        lift: build(move |x| vec![score(x)]),
        reference: build(|s| vec![s[0]]),
        occupied: occ_max(0),
        g: build(move |v| g(&v[..n_x], v[n_x])),
        sample,
    }
}

/// `(n, Σx)` first pass, reference the mean.
fn mean_spec(
    name: &str,
    n_x: usize,
    g: impl Fn(&[ExprId], ExprId) -> Vec<ExprId> + 'static,
    sample: Sampler,
) -> OnlineSpec {
    OnlineSpec {
        name: name.into(),
        n_x,
        r1: sums(2),
        lift: build(|x| vec![c(1.0), x[0]]),
        reference: build(|s| vec![s[1] / s[0]]),
        occupied: occ_count(0),
        g: build(move |v| g(&v[..n_x], v[n_x])),
        sample,
    }
}

// ---------------------------------------------------------------------------
// Online catalog
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    Found,
    NotFound,
}

pub struct OnlineCase {
    pub spec: OnlineSpec,
    pub expect: Expect,
    pub note: &'static str,
}

pub fn online_catalog() -> Vec<OnlineCase> {
    use Expect::*;
    let mut v = vec![];
    let mut add = |spec, expect, note| v.push(OnlineCase { spec, expect, note });

    add(
        softmax_spec(
            "softmax attention (Z, Σ p·v)",
            2,
            |x| x[0],
            |x, m| {
                let e = eexp(x[0] - m);
                vec![e, e * x[1]]
            },
            normals(2, 3.0),
        ),
        Found,
        "online softmax / FlashAttention",
    );
    add(
        softmax_spec(
            "logsumexp",
            1,
            |x| x[0],
            |x, m| vec![eexp(x[0] - m)],
            normals(1, 4.0),
        ),
        Found,
        "lse = m + ln S",
    );
    add(
        mean_spec(
            "variance (Welford)",
            1,
            |x, mu| vec![(x[0] - mu) * (x[0] - mu)],
            sampler(|r| vec![5.0 + 2.0 * r.normal()]),
        ),
        Found,
        "Welford / Chan pairwise",
    );
    add(
        mean_spec(
            "skewness (3rd central moment)",
            1,
            |x, mu| {
                let d = x[0] - mu;
                vec![d * d, d * d * d]
            },
            sampler(|r| vec![1.0 + r.normal()]),
        ),
        Found,
        "Pébay higher-order moments",
    );
    add(
        mean_spec(
            "kurtosis (4th central moment)",
            1,
            |x, mu| {
                let d = x[0] - mu;
                vec![d * d, d * d * d * d]
            },
            sampler(|r| vec![1.0 + r.normal()]),
        ),
        Found,
        "Pébay higher-order moments",
    );
    add(
        OnlineSpec {
            name: "covariance (co-moment)".into(),
            n_x: 2,
            r1: sums(3),
            lift: build(|x| vec![c(1.0), x[0], x[1]]),
            reference: build(|s| vec![s[1] / s[0], s[2] / s[0]]),
            occupied: occ_count(0),
            g: build(|v| vec![(v[0] - v[2]) * (v[1] - v[3])]),
            sample: sampler(|r| {
                let a = r.normal();
                vec![2.0 + a, -1.0 + 0.5 * a + r.normal()]
            }),
        },
        Found,
        "Chan co-moment update",
    );
    add(
        OnlineSpec {
            name: "LayerNorm → matmul (one column)".into(),
            n_x: 2,
            r1: sums(3),
            lift: build(|x| vec![c(1.0), x[0], x[0] * x[0]]),
            reference: build(|s| {
                let mu = s[1] / s[0];
                vec![mu, (s[2] / s[0] - mu * mu + c(1e-5)).sqrt()]
            }),
            occupied: occ_count(0),
            g: build(|v| vec![(v[0] - v[2]) / v[3] * v[1]]),
            sample: normals(2, 1.5),
        },
        Found,
        "LN folding: (x·W − μ·1ᵀW)/σ",
    );
    add(
        OnlineSpec {
            name: "RMSNorm → matmul (one column)".into(),
            n_x: 2,
            r1: sums(2),
            lift: build(|x| vec![c(1.0), x[0] * x[0]]),
            reference: build(|s| vec![(s[1] / s[0] + c(1e-6)).sqrt()]),
            occupied: occ_count(0),
            g: build(|v| vec![v[0] * v[1] / v[2]]),
            sample: normals(2, 1.0),
        },
        Found,
        "hoistable rescale",
    );
    add(
        OnlineSpec {
            name: "L2-normalize → dot (cosine)".into(),
            n_x: 2,
            r1: sums(1),
            lift: build(|x| vec![x[0] * x[0]]),
            reference: build(|s| vec![s[0].sqrt()]),
            occupied: build(|s| vec![s[0]]),
            g: build(|v| vec![v[0] * v[1] / v[2]]),
            sample: normals(2, 1.0),
        },
        Found,
        "hoistable rescale",
    );
    add(
        OnlineSpec {
            name: "stable 2-norm (scaled sum of squares)".into(),
            n_x: 1,
            r1: Monoid::max(),
            lift: build(|x| vec![absx(x[0])]),
            reference: build(|s| vec![s[0]]),
            occupied: occ_max(0),
            g: build(|v| {
                let t = v[0] / v[1];
                vec![t * t]
            }),
            sample: sampler(|r| vec![r.normal() * (r.range(-20.0, 20.0)).exp()]),
        },
        Found,
        "Blue 1978 / LAPACK dnrm2",
    );
    add(
        OnlineSpec {
            name: "power mean, max-scaled (p = 3)".into(),
            n_x: 1,
            r1: Monoid::max(),
            lift: build(|x| vec![x[0]]),
            reference: build(|s| vec![s[0]]),
            occupied: occ_max(0),
            g: build(|v| {
                let t = v[0] / v[1];
                vec![t * t * t]
            }),
            sample: sampler(|r| vec![r.range(0.1, 10.0)]),
        },
        Found,
        "",
    );
    add(
        OnlineSpec {
            name: "absmax-scale → dot (no rounding)".into(),
            n_x: 2,
            r1: Monoid::max(),
            lift: build(|x| vec![absx(x[0])]),
            reference: build(|s| vec![s[0]]),
            occupied: occ_max(0),
            g: build(|v| vec![c(127.0) * v[0] / v[2] * v[1]]),
            sample: normals(2, 1.0),
        },
        Found,
        "hoistable rescale",
    );
    add(
        OnlineSpec {
            name: "absmax-quantize (int4 rounding) → dot".into(),
            n_x: 2,
            r1: Monoid::max(),
            lift: build(|x| vec![absx(x[0])]),
            reference: build(|s| vec![s[0]]),
            occupied: occ_max(0),
            g: build(|v| {
                // round(7·x/a) as a select staircase on [−7, 7]
                let z = c(7.0) * v[0] / v[2];
                let mut q = c(0.0);
                for k in -7..7 {
                    let t = k as f64 + 0.5;
                    q = q + sel(z - c(t), c(1.0), c(0.0));
                }
                vec![(q - c(7.0)) * v[2] / c(7.0) * v[1]]
            }),
            sample: normals(2, 1.0),
        },
        NotFound,
        "rounding is not shift-equivariant in the scale",
    );
    add(
        softmax_spec(
            "softcapped attention (Gemma 2)",
            2,
            |x| c(30.0) * (x[0] / c(30.0)).tanh(),
            |x, m| {
                let e = eexp(c(30.0) * (x[0] / c(30.0)).tanh() - m);
                vec![e, e * x[1]]
            },
            normals(2, 20.0),
        ),
        Found,
        "online softmax on transformed scores",
    );
    add(
        softmax_spec(
            "ALiBi attention",
            3,
            |x| x[0] - c(0.25) * x[2],
            |x, m| {
                let e = eexp(x[0] - c(0.25) * x[2] - m);
                vec![e, e * x[1]]
            },
            sampler(|r| vec![2.0 * r.normal(), r.normal(), r.below(64) as f64]),
        ),
        Found,
        "online softmax on biased scores",
    );
    add(
        OnlineSpec {
            name: "sigmoid attention".into(),
            n_x: 2,
            r1: sums(1),
            lift: build(|_| vec![c(0.0)]),
            reference: build(|_| vec![]),
            occupied: build(|_| vec![c(1.0)]),
            g: build(|v| vec![sigmoid(v[0] - c(2.0)) * v[1]]),
            sample: normals(2, 2.0),
        },
        Found,
        "no normalizer: already one pass",
    );
    add(
        softmax_spec(
            "z-loss (lse²)",
            1,
            |x| x[0],
            |x, m| vec![eexp(x[0] - m)],
            normals(1, 5.0),
        ),
        Found,
        "epilogue (m + ln S)² of lse",
    );
    add(
        softmax_spec(
            "cross-entropy (lse, target logit)",
            2,
            |x| x[0],
            |x, m| vec![eexp(x[0] - m), x[1] * x[0]],
            sampler(|r| vec![3.0 * r.normal(), (r.below(8) == 0) as u8 as f64]),
        ),
        Found,
        "two spectral blocks (λ = −1, 0)",
    );
    add(
        softmax_spec(
            "softmax-weighted mean E_p[x]",
            2,
            |x| x[0],
            |x, m| {
                let e = eexp(x[0] - m);
                vec![e, e * x[1]]
            },
            normals(2, 3.0),
        ),
        Found,
        "",
    );
    add(
        softmax_spec(
            "softmax entropy",
            1,
            |x| x[0],
            |x, m| {
                let e = eexp(x[0] - m);
                vec![e, e * (x[0] - m)]
            },
            normals(1, 3.0),
        ),
        Found,
        "Jordan block: e^{s−m}·(s−m) shifts with a nilpotent part",
    );
    add(
        OnlineSpec {
            name: "KL(softmax a ‖ softmax b) from logits".into(),
            n_x: 2,
            r1: prod(&[Monoid::max(), Monoid::max()]),
            lift: build(|x| vec![x[0], x[1]]),
            reference: build(|s| vec![s[0], s[1]]),
            occupied: occ_max(0),
            g: build(|v| {
                let ea = eexp(v[0] - v[2]);
                vec![ea, eexp(v[1] - v[3]), ea * (v[0] - v[1])]
            }),
            sample: sampler(|r| {
                let a = 2.0 * r.normal();
                vec![a, a + r.normal()]
            }),
        },
        Found,
        "distillation loss in one pass",
    );
    add(
        OnlineSpec {
            name: "softmax-weighted variance Var_p[v]".into(),
            n_x: 2,
            r1: Monoid::online_softmax(2),
            lift: build(|x| vec![x[0], c(1.0), x[1]]),
            reference: build(|s| vec![s[0], s[2] / s[1]]),
            occupied: occ_max(0),
            g: build(|v| {
                let d = v[1] - v[3];
                vec![eexp(v[0] - v[2]) * d * d]
            }),
            sample: sampler(|r| vec![3.0 * r.normal(), 4.0 + r.normal()]),
        },
        Found,
        "two references (m, μ_p): e^{−δm}·binomial(δμ)",
    );
    add(
        OnlineSpec {
            name: "softmax score variance Var_p[s]".into(),
            n_x: 1,
            r1: Monoid::online_softmax(2),
            lift: build(|x| vec![x[0], c(1.0), x[0]]),
            reference: build(|s| vec![s[0], s[2] / s[1]]),
            occupied: occ_max(0),
            g: build(|v| {
                let d = v[0] - v[2];
                vec![eexp(v[0] - v[1]) * d * d]
            }),
            sample: normals(1, 3.0),
        },
        Found,
        "",
    );
    add(
        OnlineSpec {
            name: "softmax-weighted covariance Cov_p[u, v]".into(),
            n_x: 3,
            r1: Monoid::online_softmax(3),
            lift: build(|x| vec![x[0], c(1.0), x[1], x[2]]),
            reference: build(|s| vec![s[0], s[2] / s[1], s[3] / s[1]]),
            occupied: occ_max(0),
            g: build(|v| vec![eexp(v[0] - v[3]) * (v[1] - v[4]) * (v[2] - v[5])]),
            sample: sampler(|r| {
                let a = r.normal();
                vec![2.0 * r.normal(), 1.0 + a, a - r.normal()]
            }),
        },
        Found,
        "",
    );
    add(
        OnlineSpec {
            name: "standardized-logit attention softmax((s−μ)/σ)".into(),
            n_x: 2,
            r1: prod(&[sums(3), Monoid::max()]),
            lift: build(|x| vec![c(1.0), x[0], x[0] * x[0], x[0]]),
            reference: build(|s| {
                let mu = s[1] / s[0];
                vec![(s[2] / s[0] - mu * mu + c(1e-3)).sqrt(), s[3]]
            }),
            occupied: occ_count(0),
            g: build(|v| {
                let e = eexp((v[0] - v[3]) / v[2]);
                vec![e, e * v[1]]
            }),
            sample: normals(2, 2.0),
        },
        NotFound,
        "σ scales the exponent: {e^{t·s}}ₜ is infinite-dimensional",
    );
    add(
        mean_spec(
            "mean absolute deviation about the mean",
            1,
            |x, mu| vec![absx(x[0] - mu)],
            normals(1, 1.0),
        ),
        NotFound,
        "kink at x = μ",
    );
    add(
        softmax_spec(
            "max-relative ReLU attention (sparsemax proxy)",
            2,
            |x| x[0],
            |x, m| {
                let t = x[0] - m + c(2.0);
                let r = sel(t, t, c(0.0));
                vec![r, r * x[1]]
            },
            normals(2, 2.0),
        ),
        NotFound,
        "threshold moves with m: piecewise-linear",
    );
    add(
        OnlineSpec {
            name: "MoE top-2 renormalized weight sum".into(),
            n_x: 1,
            r1: prod(&[Monoid::max(), Monoid::topk(2)]),
            lift: build(|x| vec![x[0], c(0.0), x[0], c(0.0), c(-(f32::MAX as f64))]),
            reference: build(|s| vec![s[0], s[4]]),
            occupied: occ_max(0),
            g: build(|v| vec![eexp(v[0] - v[1]) * sel(v[0] - v[2] + c(1e-12), c(1.0), c(0.0))]),
            sample: normals(1, 2.0),
        },
        NotFound,
        "the top-k state already holds the answer; a second pass over a threshold is not a lift",
    );
    add(
        OnlineSpec {
            name: "Gaussian NLL at the running mean, σ fixed".into(),
            n_x: 1,
            r1: sums(2),
            lift: build(|x| vec![c(1.0), x[0]]),
            reference: build(|s| vec![s[1] / s[0]]),
            occupied: occ_count(0),
            g: build(|v| {
                let d = v[0] - v[1];
                vec![c(0.5) * d * d + c(0.9189385332046727)]
            }),
            sample: normals(1, 2.0),
        },
        Found,
        "",
    );
    add(
        softmax_spec(
            "log-softmax-weighted sum Σ p·log p·v",
            2,
            |x| x[0],
            |x, m| {
                let e = eexp(x[0] - m);
                vec![e, e * x[1], e * (x[0] - m) * x[1]]
            },
            normals(2, 2.0),
        ),
        Found,
        "needs ln Z epilogue; Jordan block",
    );
    add(
        OnlineSpec {
            name: "cos-similarity attention weights exp(cos(s − a))".into(),
            n_x: 1,
            r1: Monoid::max(),
            lift: build(|x| vec![x[0]]),
            reference: build(|s| vec![s[0]]),
            occupied: occ_max(0),
            g: build(|v| vec![(v[0] - v[1]).cos()]),
            sample: normals(1, 2.0),
        },
        NotFound,
        "closed (dim 2) but rotation spectrum ±i: not emitted",
    );
    v
}

// ---------------------------------------------------------------------------
// Scan catalog
// ---------------------------------------------------------------------------

pub struct ScanCase {
    pub spec: ScanSpec,
    pub expect: Expect,
    pub note: &'static str,
}

fn scan(
    name: &str,
    d: usize,
    n_x: usize,
    step: impl Fn(&[ExprId], &[ExprId]) -> Vec<ExprId> + 'static,
    sample_x: Sampler,
    sample_h: Sampler,
) -> ScanSpec {
    ScanSpec {
        name: name.into(),
        d,
        n_x,
        step: build(move |v| step(&v[..d], &v[d..])),
        sample_x,
        sample_h,
        mat: None,
    }
}

/// `S' = diag(a)(I − [delta]·β·k·kᵀ)S + β·k·vᵀ` on a 2×2 state, inputs
/// `(a₀, a₁, β, k₀, k₁, v₀, v₁)`; `a` tied for scalar decay.
fn delta_like(name: &str, decay: u8, beta: bool, delta: bool) -> ScanSpec {
    let (n, m) = (2usize, 2usize);
    ScanSpec {
        name: name.into(),
        d: n * m,
        n_x: 7,
        step: build(move |v| {
            let (s, x) = v.split_at(n * m);
            let a = [x[0], if decay == 1 { x[0] } else { x[1] }];
            let b = if beta { x[2] } else { c(1.0) };
            let k = [x[3], x[4]];
            let vv = [x[5], x[6]];
            let mut out = vec![];
            for i in 0..n {
                for cc in 0..m {
                    let mut y = s[i * m + cc];
                    if delta {
                        let kts = k[0] * s[cc] + k[1] * s[m + cc];
                        y = y - b * k[i] * kts;
                    }
                    let y = if decay == 0 { y } else { a[i] * y };
                    out.push(y + b * k[i] * vv[cc]);
                }
            }
            out
        }),
        sample_x: sampler(|r| {
            let (k0, k1) = (r.normal(), r.normal());
            let kn = (k0 * k0 + k1 * k1).sqrt().max(1e-3);
            vec![
                r.range(0.8, 1.0),
                r.range(0.8, 1.0),
                r.range(0.1, 0.9),
                k0 / kn,
                k1 / kn,
                r.normal(),
                r.normal(),
            ]
        }),
        sample_h: normals(4, 1.0),
        mat: Some((n, m)),
    }
}

pub fn scan_catalog() -> Vec<ScanCase> {
    use Expect::*;
    let mut v = vec![];
    let mut add = |spec, expect, note| v.push(ScanCase { spec, expect, note });
    let pos =
        |lo: f64, hi: f64, k: usize| sampler(move |r| (0..k).map(|_| r.range(lo, hi)).collect());

    add(
        scan(
            "cumsum",
            1,
            1,
            |h, x| vec![h[0] + x[0]],
            normals(1, 1.0),
            normals(1, 1.0),
        ),
        Found,
        "",
    );
    add(
        scan(
            "EMA",
            1,
            2,
            |h, x| vec![x[0] * h[0] + (c(1.0) - x[0]) * x[1]],
            sampler(|r| vec![r.range(0.5, 1.0), r.normal()]),
            normals(1, 1.0),
        ),
        Found,
        "",
    );
    add(
        scan(
            "cumprod",
            1,
            1,
            |h, x| vec![h[0] * x[0]],
            pos(0.5, 1.5, 1),
            pos(0.5, 2.0, 1),
        ),
        Found,
        "",
    );
    add(
        scan(
            "logcumsumexp",
            1,
            1,
            |h, x| vec![eln(eexp(h[0]) + eexp(x[0]))],
            normals(1, 2.0),
            normals(1, 2.0),
        ),
        Found,
        "affine in exp(h)",
    );
    add(
        scan(
            "softplus recurrence ln(1 + e^{h+x})",
            1,
            1,
            |h, x| vec![eln(c(1.0) + eexp(h[0] + x[0]))],
            normals(1, 1.0),
            normals(1, 1.0),
        ),
        Found,
        "affine in exp(h)",
    );
    add(
        scan(
            "harmonic h/(1 + h·x)",
            1,
            1,
            |h, x| vec![h[0] / (c(1.0) + h[0] * x[0])],
            pos(0.1, 1.0, 1),
            pos(0.1, 2.0, 1),
        ),
        Found,
        "affine in 1/h",
    );
    add(
        scan(
            "geometric smoothing x·√h",
            1,
            1,
            |h, x| vec![x[0] * h[0].sqrt()],
            pos(0.5, 2.0, 1),
            pos(0.5, 2.0, 1),
        ),
        Found,
        "affine in ln h",
    );
    add(
        scan(
            "cummax",
            1,
            1,
            |h, x| vec![emax(h[0], x[0])],
            normals(1, 1.0),
            normals(1, 1.0),
        ),
        Found,
        "",
    );
    add(
        scan(
            "decayed running max",
            1,
            2,
            |h, x| vec![emax(h[0] - x[0], x[1])],
            sampler(|r| vec![r.range(0.0, 0.5), r.normal()]),
            normals(1, 1.0),
        ),
        Found,
        "",
    );
    add(
        scan(
            "Viterbi (2 states)",
            2,
            2,
            |h, x| {
                let t = [[-0.1, -2.3], [-1.6, -0.2]];
                (0..2)
                    .map(|i| emax(h[0] + c(t[i][0]), h[1] + c(t[i][1])) + x[i])
                    .collect()
            },
            normals(2, 1.0),
            normals(2, 1.0),
        ),
        Found,
        "",
    );
    add(
        scan(
            "HMM forward (2 states, unnormalized)",
            2,
            2,
            |h, x| {
                let t = [[0.9, 0.2], [0.1, 0.8]];
                (0..2)
                    .map(|i| (c(t[i][0]) * h[0] + c(t[i][1]) * h[1]) * x[i])
                    .collect()
            },
            pos(0.1, 1.0, 2),
            pos(0.1, 1.0, 2),
        ),
        Found,
        "",
    );
    add(
        scan(
            "cumulative rotation",
            2,
            1,
            |h, x| {
                let (s, co) = (x[0].sin(), x[0].cos());
                vec![co * h[0] - s * h[1], s * h[0] + co * h[1]]
            },
            normals(1, 1.0),
            normals(2, 1.0),
        ),
        Found,
        "",
    );
    add(
        scan(
            "minGRU",
            1,
            2,
            |h, x| {
                let z = sigmoid(x[0]);
                vec![(c(1.0) - z) * h[0] + z * x[1]]
            },
            normals(2, 1.0),
            normals(1, 1.0),
        ),
        Found,
        "gates independent of h",
    );
    add(
        scan(
            "mLSTM cell (c, n), unstabilized",
            2,
            3,
            |h, x| {
                let (f, i) = (sigmoid(x[0]), eexp(x[1]));
                vec![f * h[0] + i * x[2], f * h[1] + i]
            },
            sampler(|r| vec![r.normal() + 2.0, 0.3 * r.normal(), r.normal()]),
            pos(0.1, 1.0, 2),
        ),
        Found,
        "",
    );
    add(
        scan(
            "running mean with count (μ, n)",
            2,
            1,
            |h, x| {
                let n1 = h[1] + c(1.0);
                vec![h[0] + (x[0] - h[0]) / n1, n1]
            },
            normals(1, 1.0),
            sampler(|r| vec![r.normal(), (1 + r.below(20)) as f64]),
        ),
        Found,
        "observables (μ·n, n)",
    );
    add(
        scan(
            "scalar Kalman variance (Riccati)",
            1,
            3,
            |h, x| {
                let (a, q, r) = (x[0], x[1], x[2]);
                vec![a * a * h[0] * r / (h[0] + r) + q]
            },
            sampler(|r| vec![r.range(0.5, 1.2), r.range(0.01, 0.5), r.range(0.1, 2.0)]),
            pos(0.1, 3.0, 1),
        ),
        Found,
        "Möbius",
    );
    add(
        scan(
            "continued fraction 1/(x + h)",
            1,
            1,
            |h, x| vec![c(1.0) / (x[0] + h[0])],
            pos(1.0, 3.0, 1),
            pos(0.1, 2.0, 1),
        ),
        Found,
        "Möbius",
    );
    add(
        scan(
            "tanh RNN",
            1,
            1,
            |h, x| vec![(c(0.8) * h[0] + x[0]).tanh()],
            normals(1, 1.0),
            normals(1, 0.8),
        ),
        NotFound,
        "",
    );
    add(
        scan(
            "GRU (scalar)",
            1,
            1,
            |h, x| {
                let z = sigmoid(c(0.5) * h[0] + x[0]);
                let r = sigmoid(c(-0.4) * h[0] + c(0.3) * x[0]);
                let ht = (c(0.9) * r * h[0] + c(0.7) * x[0]).tanh();
                vec![(c(1.0) - z) * h[0] + z * ht]
            },
            normals(1, 1.0),
            normals(1, 0.8),
        ),
        NotFound,
        "",
    );
    add(
        scan(
            "LSTM (scalar, h-dependent gates)",
            2,
            1,
            |s, x| {
                let (cc, h) = (s[0], s[1]);
                let f = sigmoid(c(0.6) * h + x[0]);
                let i = sigmoid(c(-0.3) * h + c(0.5) * x[0]);
                let o = sigmoid(c(0.2) * h - x[0]);
                let g = (c(0.8) * h + x[0]).tanh();
                let c2 = f * cc + i * g;
                vec![c2, o * c2.tanh()]
            },
            normals(1, 1.0),
            normals(2, 0.8),
        ),
        NotFound,
        "",
    );
    add(
        scan(
            "logistic map",
            1,
            1,
            |h, x| vec![x[0] * h[0] * (c(1.0) - h[0])],
            pos(2.5, 3.9, 1),
            pos(0.05, 0.95, 1),
        ),
        NotFound,
        "",
    );
    add(
        scan(
            "leaky integrate-and-fire with reset",
            1,
            1,
            |h, x| {
                let v = c(0.9) * h[0] + x[0];
                vec![sel(v - c(1.0), c(0.0), v)]
            },
            pos(0.0, 0.4, 1),
            pos(0.0, 1.5, 1),
        ),
        NotFound,
        "",
    );
    add(
        scan(
            "Welford running (μ, M₂, n)",
            3,
            1,
            |h, x| {
                let n1 = h[2] + c(1.0);
                let mu = h[0] + (x[0] - h[0]) / n1;
                vec![mu, h[1] + (x[0] - h[0]) * (x[0] - mu), n1]
            },
            normals(1, 1.0),
            sampler(|r| vec![r.normal(), r.range(0.0, 5.0), (1 + r.below(20)) as f64]),
        ),
        NotFound,
        "outside the searched families (observable library is d ≤ 2)",
    );
    add(
        delta_like("linear attention S + k·vᵀ", 0, false, false),
        Found,
        "",
    );
    add(
        delta_like("Mamba-2 SSD αS + β·k·vᵀ", 1, true, false),
        Found,
        "",
    );
    add(
        delta_like("GLA / Mamba-1 diag(a)S + k·vᵀ", 2, false, false),
        Found,
        "",
    );
    add(
        delta_like("DeltaNet (I − β·k·kᵀ)S + β·k·vᵀ", 0, true, true),
        Found,
        "",
    );
    add(
        delta_like("gated delta rule α(I − β·k·kᵀ)S + β·k·vᵀ", 1, true, true),
        Found,
        "",
    );
    add(
        {
            let mut s = delta_like(
                "diag-gated delta (KDA-like) diag(a)(I − β·k·kᵀ)S + β·k·vᵀ",
                2,
                true,
                true,
            );
            s.name = s.name.clone();
            s
        },
        Found,
        "affine; A is General for the chunk lift",
    );
    v
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

pub struct Row {
    pub kind: &'static str,
    pub name: String,
    pub found: bool,
    pub expect: Expect,
    pub result: String,
    pub detail: String,
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

pub fn run_catalog() -> Vec<Row> {
    let mut rows = vec![];
    for case in online_catalog() {
        let s = &case.spec;
        let (found, result, detail) = match online_lift(s) {
            Ok(l) => {
                let err = verify_online(s, &l, 99, 40);
                let spec: Vec<String> = l
                    .blocks
                    .iter()
                    .map(|(lams, _)| {
                        format!(
                            "({})",
                            lams.iter()
                                .map(|x| format!("{x}"))
                                .collect::<Vec<_>>()
                                .join(",")
                        )
                    })
                    .collect();
                (
                    true,
                    format!(
                        "found: dim {}, coords {:?}, λ {}{}",
                        l.dim,
                        l.coords,
                        spec.join(" "),
                        if l.separable { ", separable" } else { "" }
                    ),
                    format!(
                        "fuzz {err:.1e}; relations {}; basis [{}]",
                        if l.symbolic {
                            "simplify→0"
                        } else {
                            "numeric"
                        },
                        l.basis
                            .iter()
                            .map(|b| trunc(b, 40))
                            .collect::<Vec<_>>()
                            .join("; ")
                    ),
                )
            }
            Err(e) => (false, format!("not found: {e:?}"), String::new()),
        };
        rows.push(Row {
            kind: "online",
            name: s.name.clone(),
            found,
            expect: case.expect,
            result,
            detail: format!("{detail} {}", case.note).trim().to_string(),
        });
    }
    for case in scan_catalog() {
        let s = &case.spec;
        let (found, result, detail) = match scan_lift(s) {
            Ok((l, err)) => {
                let sym = match &l {
                    ScanLift::Affine { symbolic, .. } => {
                        if *symbolic {
                            "∂f/∂h independent of h (structural)"
                        } else {
                            "∂f/∂h independent of h (numeric)"
                        }
                    }
                    ScanLift::Mobius { .. } => "Schwarzian = 0",
                    _ => "numeric certificate",
                };
                let chunk = chunk_lift(s)
                    .map(|c| format!("; chunked {:?} err {:.1e}", c.form, c.err))
                    .unwrap_or_default();
                (
                    true,
                    format!("found: {}", l.family()),
                    format!("{sym}; fuzz {err:.1e}{chunk}"),
                )
            }
            Err(w) => (
                false,
                "not found in family".into(),
                w.iter()
                    .map(|w| format!("{}: {}", w.family, trunc(&w.why, 70)))
                    .collect::<Vec<_>>()
                    .join(" | "),
            ),
        };
        rows.push(Row {
            kind: "scan",
            name: s.name.clone(),
            found,
            expect: case.expect,
            result,
            detail: format!("{detail} {}", case.note).trim().to_string(),
        });
    }
    rows
}

pub fn catalog_markdown(rows: &[Row]) -> String {
    let mut s = String::from("| kind | op | result | evidence |\n|---|---|---|---|\n");
    for r in rows {
        s += &format!(
            "| {} | {} | {} | {} |\n",
            r.kind,
            r.name,
            r.result.replace('|', "/"),
            r.detail.replace('|', "/")
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_softmax_is_rediscovered() {
        let cat = online_catalog();
        let l = online_lift(&cat[0].spec).unwrap_or_else(|e| panic!("{e:?}"));
        assert_eq!(l.dim, 2);
        assert_eq!(l.blocks.len(), 1);
        assert_eq!(l.blocks[0].0, vec![-1.0]);
        assert!(verify_online(&cat[0].spec, &l, 1, 50) < 1e-10);
    }

    #[test]
    fn welford_is_rediscovered() {
        let cat = online_catalog();
        let l = online_lift(&cat[2].spec).unwrap_or_else(|e| panic!("{e:?}"));
        // (x−μ)², (x−μ), 1: one block, λ = 0, nilpotent of order 3
        assert_eq!(l.dim, 3);
        assert_eq!(l.blocks[0].0, vec![0.0]);
    }

    #[test]
    fn online_lift_matches_two_pass_in_the_interpreter() {
        for case in online_catalog()
            .into_iter()
            .filter(|c| c.expect == Expect::Found)
            .take(12)
        {
            let l = online_lift(&case.spec)
                .ok()
                .unwrap_or_else(|| panic!("{}", case.spec.name));
            let (rows, n) = (3, 37);
            let (g2, two, g1, one) = online_graphs(&case.spec, &l, rows, n);
            let mut rng = Rng(5);
            let inputs: Vec<Vec<f64>> = {
                let mut cols: Vec<Vec<f64>> = vec![vec![]; case.spec.n_x];
                for _ in 0..rows * n {
                    for (c, x) in cols.iter_mut().zip((case.spec.sample)(&mut rng)) {
                        c.push(x);
                    }
                }
                cols
            };
            let a = &g2.eval(&inputs, &[two])[0];
            let b = &g1.eval(&inputs, &[one])[0];
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() <= 1e-8 * (x.abs() + 1e-6),
                    "{}: two-pass {x} vs one-pass {y}",
                    case.spec.name
                );
            }
        }
    }

    #[test]
    fn delta_family_is_chunked() {
        for (name, decay, beta, delta, form) in [
            ("lin", 0, false, false, ChunkForm::Cumulative),
            ("ssd", 1, true, false, ChunkForm::ScalarDecay),
            ("gla", 2, false, false, ChunkForm::DiagDecay),
            ("delta", 0, true, true, ChunkForm::ScalarDecayDelta),
            ("gdn", 1, true, true, ChunkForm::ScalarDecayDelta),
        ] {
            let s = delta_like(name, decay, beta, delta);
            let c = chunk_lift(&s).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(c.form, form, "{name}");
            assert!(c.err < 1e-9, "{name}: {}", c.err);
        }
    }

    #[test]
    fn catalog_meets_expectations() {
        let rows = run_catalog();
        let md = catalog_markdown(&rows);
        println!("{md}");
        if let Ok(p) = std::env::var("TANG_DISCOVERY_MD") {
            std::fs::write(p, &md).unwrap();
        }
        let bad: Vec<String> = rows
            .iter()
            .filter(|r| r.found != (r.expect == Expect::Found))
            .map(|r| format!("{} {}: {} {}", r.kind, r.name, r.result, r.detail))
            .collect();
        assert!(rows.len() >= 30);
        assert!(bad.is_empty(), "unexpected:\n{}", bad.join("\n"));
    }

    /// Backward of `Var_p[v]` is elementwise from the saved forward
    /// state `(m, Z, μ, Var)`: `∂/∂vᵢ = 2pᵢ(vᵢ − μ)`,
    /// `∂/∂sᵢ = pᵢ((vᵢ − μ)² − Var)`. Checked by central differences.
    #[test]
    fn weighted_variance_backward_is_elementwise() {
        let var = |s: &[f64], v: &[f64]| {
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            let p: Vec<f64> = s.iter().map(|x| (x - m).exp() / z).collect();
            let mu: f64 = p.iter().zip(v).map(|(a, b)| a * b).sum();
            (
                p.iter()
                    .zip(v)
                    .map(|(a, b)| a * (b - mu) * (b - mu))
                    .sum::<f64>(),
                p,
                mu,
            )
        };
        let mut rng = Rng(11);
        let s: Vec<f64> = (0..17).map(|_| 2.0 * rng.normal()).collect();
        let v: Vec<f64> = (0..17).map(|_| 1.0 + rng.normal()).collect();
        let (vv, p, mu) = var(&s, &v);
        let h = 1e-6;
        for i in 0..17 {
            let (mut sp, mut sm, mut vp, mut vm) = (s.clone(), s.clone(), v.clone(), v.clone());
            sp[i] += h;
            sm[i] -= h;
            vp[i] += h;
            vm[i] -= h;
            let ds = (var(&sp, &v).0 - var(&sm, &v).0) / (2.0 * h);
            let dv = (var(&s, &vp).0 - var(&s, &vm).0) / (2.0 * h);
            let d = v[i] - mu;
            assert!((ds - p[i] * (d * d - vv)).abs() < 1e-7, "ds {i}");
            assert!((dv - 2.0 * p[i] * d).abs() < 1e-7, "dv {i}");
        }
    }

    /// The backward of every discovered one-pass graph is derived
    /// (generic monoid vjp: prefix/suffix scans + combine vjp) and matches
    /// central differences of the interpreter, and the backward of the
    /// two-pass definition.
    #[test]
    fn discovered_lifts_have_derived_backward() {
        let mut checked = 0;
        for case in online_catalog()
            .into_iter()
            .filter(|c| c.expect == Expect::Found)
        {
            let s = &case.spec;
            let l = online_lift(s).unwrap_or_else(|e| panic!("{e:?}"));
            let (rows, n) = (2, 9);
            let (mut g2, two, mut g1, one) = online_graphs(s, &l, rows, n);
            let xs1: Vec<_> = (0..s.n_x).map(super::super::TensorId::from_index).collect();
            let gr1 = g1
                .backward(one, &xs1)
                .unwrap_or_else(|e| panic!("{}: {e:?}", s.name));
            assert!(
                l.monoid.saved.is_some(),
                "{}: no saved-state structure",
                s.name
            );
            // the saved-state path: no prefix/suffix scans in the one-pass backward
            assert!(
                (0..g1.nodes().len())
                    .all(|i| g1.scan_at(super::super::TensorId::from_index(i)).is_none()),
                "{}: backward fell back to scans",
                s.name
            );
            let gr2 = g2
                .backward(two, &xs1)
                .unwrap_or_else(|e| panic!("{}: {e:?}", s.name));
            let mut rng = Rng(21);
            let mut inputs: Vec<Vec<f64>> = vec![vec![]; s.n_x];
            for _ in 0..rows * n {
                for (c, x) in inputs.iter_mut().zip((s.sample)(&mut rng)) {
                    c.push(x);
                }
            }
            let a = g1.eval(&inputs, &gr1);
            let b = g2.eval(&inputs, &gr2);
            let total = |inp: &Vec<Vec<f64>>| g1.eval(inp, &[one])[0].iter().sum::<f64>();
            for xi in 0..s.n_x {
                // integer positions (ALiBi) have no meaningful derivative
                if s.name.starts_with("ALiBi") && xi == 2 {
                    continue;
                }
                for p in 0..rows * n {
                    let h = 1e-6 * (1.0 + inputs[xi][p].abs());
                    let (mut up, mut dn) = (inputs.clone(), inputs.clone());
                    up[xi][p] += h;
                    dn[xi][p] -= h;
                    let fd = (total(&up) - total(&dn)) / (2.0 * h);
                    let scale = 1e-5 * (1.0 + fd.abs());
                    assert!(
                        (a[xi][p] - fd).abs() <= scale,
                        "{} x{xi}[{p}]: derived {} fd {fd}",
                        s.name,
                        a[xi][p]
                    );
                    assert!(
                        (a[xi][p] - b[xi][p]).abs() <= 1e-8 * (1.0 + fd.abs()),
                        "{} one vs two pass",
                        s.name
                    );
                }
            }
            checked += 1;
        }
        assert!(checked >= 20);
    }
}
