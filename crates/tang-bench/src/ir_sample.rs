//! Token sampling written into the prefill graph, so [`derive_step`] carries
//! it into the step and the device picks the next token itself: temperature,
//! top-k, top-p (nucleus) and greedy, with a counter-based seeded RNG.
//!
//! [`append_sampler`] takes logits `[T, V]` and one input `params [4]` =
//! `(temperature, top_p, top_k, seed)` and returns the token ids `[T]` (as
//! floats). Per row `t` (the absolute position, which the derived step gets
//! from its `pos` input):
//!
//! - **greedy** (`temperature ≤ 0`): the first maximum of the logits, by the
//!   [`Monoid::argmax`] reduction over the raw logit values. No arithmetic
//!   touches the logits on this path (the score is a `select` of the logit
//!   itself), so the id is bit-identical to a host argmax (first maximum)
//!   of the same logits.
//! - **top-k** (`top_k ≥ 1`; `0` keeps everything): a bisection on a logit
//!   threshold `θ` in `[max − range, max]`, `iters` rounds of
//!   `#{v : l_v ≥ θ} ≥ k`. The kept set is `{l ≥ θ}`: the exact top-k, plus
//!   any logits within `range / 2^iters` of the k-th (ties all kept).
//! - **top-p** (`0 < top_p < 1`; otherwise off): over the top-k set, the
//!   tempered weights `exp((l − max)/temperature)`; a second bisection finds
//!   the largest threshold whose kept mass is `≥ top_p` of the total — the
//!   smallest nucleus, as a sort-and-cumsum implementation keeps it.
//! - **sample**: Gumbel-max over the kept set: the argmax of
//!   `(l − max)/temperature − ln(−ln u_v)` is a draw from the renormalized
//!   `softmax(l/temperature)` restricted to the set. No prefix sum, no sort:
//!   one more argmax reduction.
//!
//! The RNG: `u_v = hash(seed, t, v)`, rounds of `h ← (h³ + a·x + c) mod
//! 2039` (a permutation of `h` for each digit `x`) over the base-2048
//! digits of the seed, the position and the token id, three passes, three
//! lanes. Every intermediate is an integer below 2²³,
//! so f32 (device) and f64 (interpreter) compute the same `h`; only the
//! final `ln` differs in rounding. Same `(seed, position)` ⇒ same token.
//!
//! Logits below `max − range` are dropped; at `range = 64` that is a weight
//! `< e^(−64/temperature)` each (negligible for temperature ≤ 2).
use tang::Scalar;
use tang_compile::tensor::{Arg, Dim, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

/// Bisection rounds and window of the threshold searches.
#[derive(Clone, Copy, Debug)]
pub struct SampleSpec {
    pub iters: usize,
    pub range: f64,
}

impl Default for SampleSpec {
    fn default() -> Self {
        Self {
            iters: 24,
            range: 64.0,
        }
    }
}

/// `params` layout.
pub const TEMPERATURE: usize = 0;
pub const TOP_P: usize = 1;
pub const TOP_K: usize = 2;
pub const SEED: usize = 3;

/// What [`append_sampler`] added.
pub struct Sampler {
    /// The `params [4]` input.
    pub params: TensorId,
    /// Token ids `[T]`.
    pub token: TensorId,
    /// The kept-set threshold `[T]` (logits `≥` it are candidates), for checks.
    pub threshold: TensorId,
}

/// The base-2048 digit hash's modulus (prime, `3 ∤ p − 1`, so `x ↦ x³ + k`
/// permutes the residues), passes over the digits, and round constants
/// `(a, c)` per lane.
const P: f64 = 2039.0;
const PASSES: usize = 3;
const LANES: [[(f64, f64); 6]; 3] = [
    [
        (1103.0, 17.0),
        (1871.0, 941.0),
        (409.0, 1307.0),
        (1597.0, 61.0),
        (733.0, 1999.0),
        (1291.0, 577.0),
    ],
    [
        (1747.0, 389.0),
        (311.0, 1123.0),
        (1979.0, 7.0),
        (857.0, 1663.0),
        (1429.0, 251.0),
        (199.0, 1801.0),
    ],
    [
        (617.0, 1531.0),
        (1361.0, 83.0),
        (977.0, 1213.0),
        (1811.0, 449.0),
        (263.0, 1877.0),
        (1559.0, 991.0),
    ],
];

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

fn select(c: ExprId, t: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(c, t, f))
}

/// `x mod p` for an integer `0 ≤ x < 2²³`: a float quotient (off by at
/// most one) and an exact correction.
fn modp(x: ExprId) -> ExprId {
    let q = (x * lit(1.0 / P)).floor();
    let r = x - q * lit(P);
    let r = r + select(-r, lit(P), lit(0.0));
    r - select(r - lit(P - 0.5), lit(P), lit(0.0))
}

/// `x mod 2048` and `⌊x / 2048⌋` (exact: a power-of-two scale).
fn digits(x: ExprId) -> (ExprId, ExprId) {
    let hi = (x * lit(1.0 / 2048.0)).floor();
    (x - hi * lit(2048.0), hi)
}

/// `u ∈ (0, 1)` from integer `seed < 2²²`, position `< 2²²`, id `< 2²²`.
pub fn uniform(seed: ExprId, pos: ExprId, v: ExprId) -> ExprId {
    let (s0, s1) = digits(seed);
    let (p0, p1) = digits(pos);
    let (v0, v1) = digits(v);
    let xs = [s0, p0, v0, s1, p1, v1];
    let mut h = [lit(0.0); 3];
    for (j, lane) in LANES.iter().enumerate() {
        let mut x = lit(lane[5].1);
        // three passes over the digits, each lane in its own order
        for pass in 0..PASSES {
            for (r, &(a, c)) in lane.iter().enumerate() {
                let d = xs[(r + j + pass * 3) % 6];
                x = modp(modp(x * x) * x + lit(a) * d + lit(c));
            }
        }
        h[j] = x;
    }
    let u = (h[0] + (h[1] + (h[2] + lit(0.5)) * lit(1.0 / P)) * lit(1.0 / P)) * lit(1.0 / P);
    u.min(lit(1.0 - 1.0 / 1048576.0))
}

/// The same hash on the host (f64, every step exact).
pub fn uniform_host(seed: u64, pos: u64, v: u64) -> f64 {
    let modp = |x: f64| {
        let q = (x * (1.0 / P)).floor();
        let mut r = x - q * P;
        if r < 0.0 {
            r += P;
        }
        if r > P - 0.5 {
            r -= P;
        }
        r
    };
    let d = |x: u64| ((x % 2048) as f64, (x / 2048) as f64);
    let ((s0, s1), (p0, p1), (v0, v1)) = (d(seed), d(pos), d(v));
    let xs = [s0, p0, v0, s1, p1, v1];
    let mut h = [0.0; 3];
    for (j, lane) in LANES.iter().enumerate() {
        let mut x = lane[5].1;
        for pass in 0..PASSES {
            for (r, &(a, c)) in lane.iter().enumerate() {
                x = modp(modp(x * x) * x + a * xs[(r + j + pass * 3) % 6] + c);
            }
        }
        h[j] = x;
    }
    let u = (h[0] + (h[1] + (h[2] + 0.5) * (1.0 / P)) * (1.0 / P)) * (1.0 / P);
    u.min(1.0 - 1.0 / 1048576.0)
}

fn arg(src: TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View {
            gather: None,
            guard: false,
            forms,
        },
    }
}

fn u(rank: usize, d: usize) -> Form {
    Form::unit(rank, d)
}

/// Sampling over `logits [seq, V]`; adds the `params [4]` input to `g`.
pub fn append_sampler(
    g: &mut TensorGraph,
    logits: TensorId,
    spec: SampleSpec,
) -> Sampler {
    let sh = g.shape(logits).to_vec();
    let (t, v) = (sh[0], sh[1]);
    let vn = v.n();
    let params = g.input(&[Dim::Fixed(4)]);
    let p = |rank: usize, i: usize| arg(params, vec![Form::constant(rank, i as i64)]);
    let row = |x: TensorId| arg(x, vec![u(2, 0)]);
    let lv = arg(logits, vec![u(2, 0), u(2, 1)]);
    let comp = |g: &mut TensorGraph,
                out: &[Dim],
                red: &[Dim],
                args: Vec<Arg>,
                m: Option<Monoid>,
                f: &dyn Fn(&[ExprId]) -> ExprId| {
        g.comp(out, red, args, m, |x| vec![f(x)])
            .unwrap_or_else(|e| panic!("sampler comp {out:?}/{red:?}: {e:?}"))
    };
    let m = comp(g, &[t], &[v], vec![lv.clone()], Some(Monoid::max()), &|x| x[0]);
    let mrow = || arg(m, vec![u(1, 0)]);
    let (it, range) = (spec.iters, spec.range);

    // top-k: bisection on θ with #{l ≥ θ} ≥ k (k = 0: V + 1, never met)
    let mut lo = comp(g, &[t], &[], vec![mrow()], None, &move |x| x[0] - lit(range));
    let mut hi = comp(g, &[t], &[], vec![mrow()], None, &|x| x[0]);
    let keff = move |k: ExprId| select(k - lit(0.5), k, lit(vn as f64 + 1.0));
    for _ in 0..it {
        let cnt = comp(
            g,
            &[t],
            &[v],
            vec![lv.clone(), row(lo), row(hi)],
            Some(Monoid::sum()),
            &|x| {
                let mid = (x[1] + x[2]) * lit(0.5);
                select(mid - x[0], lit(0.0), lit(1.0))
            },
        );
        let one = |x: TensorId| arg(x, vec![u(1, 0)]);
        let args = vec![one(lo), one(hi), one(cnt), p(1, TOP_K)];
        let nlo = comp(g, &[t], &[], args.clone(), None, &move |x| {
            let mid = (x[0] + x[1]) * lit(0.5);
            select(x[2] - keff(x[3]) + lit(0.5), mid, x[0])
        });
        let nhi = comp(g, &[t], &[], args, None, &move |x| {
            let mid = (x[0] + x[1]) * lit(0.5);
            select(x[2] - keff(x[3]) + lit(0.5), x[1], mid)
        });
        lo = nlo;
        hi = nhi;
    }
    let theta_k = lo;

    // top-p over the top-k set: tempered weights, total, then bisection on τ
    let temp = |x: ExprId| select(x, x, lit(1.0));
    let weight = move |l: ExprId, m: ExprId, th: ExprId, tp: ExprId| {
        select(th - l, lit(0.0), ((l - m) / temp(tp)).exp())
    };
    let z = comp(
        g,
        &[t],
        &[v],
        vec![lv.clone(), row(m), row(theta_k), p(2, TEMPERATURE)],
        Some(Monoid::sum()),
        &move |x| weight(x[0], x[1], x[2], x[3]),
    );
    let mut lo = theta_k;
    let mut hi = comp(g, &[t], &[], vec![mrow()], None, &|x| x[0]);
    for _ in 0..it {
        let mass = comp(
            g,
            &[t],
            &[v],
            vec![
                lv.clone(),
                row(m),
                row(theta_k),
                p(2, TEMPERATURE),
                row(lo),
                row(hi),
            ],
            Some(Monoid::sum()),
            &move |x| {
                let mid = (x[4] + x[5]) * lit(0.5);
                select(mid - x[0], lit(0.0), weight(x[0], x[1], x[2], x[3]))
            },
        );
        let one = |x: TensorId| arg(x, vec![u(1, 0)]);
        let args = vec![one(lo), one(hi), one(mass), one(z), p(1, TOP_P)];
        let nlo = comp(g, &[t], &[], args.clone(), None, &|x| {
            let mid = (x[0] + x[1]) * lit(0.5);
            select(x[2] - x[4] * x[3], mid, select(x[4] * x[3] - x[2], x[0], mid))
        });
        let nhi = comp(g, &[t], &[], args, None, &|x| {
            let mid = (x[0] + x[1]) * lit(0.5);
            select(x[2] - x[4] * x[3], x[1], select(x[4] * x[3] - x[2], mid, x[1]))
        });
        lo = nlo;
        hi = nhi;
    }
    // top_p in (0, 1): the nucleus threshold; otherwise the top-k one
    let one = |x: TensorId| arg(x, vec![u(1, 0)]);
    let threshold = comp(
        g,
        &[t],
        &[],
        vec![one(lo), one(theta_k), p(1, TOP_P)],
        None,
        &|x| select(x[2], select(lit(1.0) - x[2], x[0], x[1]), x[1]),
    );

    // Gumbel-max over the kept set (greedy: the raw logit)
    let sp = [t, v];
    let (pos, ids) = (g.iota(&sp, 0), g.iota(&sp, 1));
    let best = g
        .comp_tuple(
            &[t, Dim::Fixed(2)],
            &[v],
            vec![
                lv,
                arg(ids, vec![u(2, 0), u(2, 1)]),
                arg(pos, vec![u(2, 0), u(2, 1)]),
                row(m),
                row(threshold),
                p(2, TEMPERATURE),
                p(2, SEED),
            ],
            Some(Monoid::argmax()),
            |x| {
                let (l, id, pos, m, th, tp, seed) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
                let uu = uniform(seed, pos, id);
                let gumbel = -(-(uu.ln())).ln();
                let s = select(th - l, lit(-1e30), (l - m) / temp(tp) + gumbel);
                vec![id, select(tp, s, l)]
            },
        )
        .unwrap();
    let token = comp(
        g,
        &[t],
        &[],
        vec![arg(best, vec![u(1, 0), Form::constant(1, 0)])],
        None,
        &|x| x[0],
    );
    Sampler {
        params,
        token,
        threshold,
    }
}

/// Candidates [`append_sampler_topk`] keeps: the hand sampler's `MAX_K`.
pub const TOPK_CANDIDATES: usize = 64;

fn form(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::constant(rank, offset);
    for &(d, c) in terms {
        f.coefs[d] += c;
    }
    f
}

/// [`append_sampler`] without the vocabulary-wide bisections: ONE top-`K`
/// reduction (`K` = [`TOPK_CANDIDATES`], the block select on CUDA) picks the
/// candidates sorted best first (ties to the lower id), and everything after
/// runs over those `K`:
///
/// - top-k: the k-th candidate's logit is the threshold (`k` capped at `K`,
///   `0` = all `K`), exact, no bisection;
/// - top-p: prefix masses over the sorted candidates; the kept set is the
///   ranks whose preceding mass is below `top_p` of the total (the smallest
///   nucleus, as the host sampler's sort-and-cumsum);
/// - the draw: Gumbel-max over the kept candidates with the same
///   `uniform(seed, position, id)` as [`append_sampler`], so the same kept set
///   gives the same token;
/// - `top_k = 0` with `top_p ≥ 1` (no cut at all): Gumbel-max over the whole
///   vocabulary (one more reduction), as the hand sampler does;
/// - greedy: the first candidate (the first maximum).
///
/// Outputs as [`append_sampler`]; `params` is `[seq, 4]` (one row of
/// `(temperature, top_p, top_k, seed)` per position). Unlike it, a nucleus that
/// would need more than `K` tokens stops at `K` (the hand sampler's cap).
pub fn append_sampler_topk(g: &mut TensorGraph, logits: TensorId) -> Sampler {
    let sh = g.shape(logits).to_vec();
    let (t, v) = (sh[0], sh[1]);
    let kc = TOPK_CANDIDATES.min(v.n());
    let kd = Dim::Fixed(kc);
    // vocabulary reductions in two passes: `nb` chunks of `cl` (the largest `nb <= 128`
    // dividing V), so a one-row step spreads them over the SMs instead of one block
    let vn = v.n();
    let nb = (1..=128.min(vn)).rev().find(|b| vn % b == 0).unwrap();
    let (cl, bd) = (vn / nb, Dim::Fixed(vn / nb));
    let nbd = Dim::Fixed(nb);
    let _ = bd;
    // a `[t, v]` tensor read at chunk b, element c (space (t, b, c))
    let chunked = |x: TensorId| arg(x, vec![u(3, 0), form(3, &[(1, cl as i64), (2, 1)], 0)]);
    // per row: a batched step (`batch_graph`) gives every stream its own parameters
    let params = g.input(&[t, Dim::Fixed(4)]);
    let p = |rank: usize, i: usize| arg(params, vec![u(rank, 0), Form::constant(rank, i as i64)]);
    let comp = |g: &mut TensorGraph,
                out: &[Dim],
                red: &[Dim],
                args: Vec<Arg>,
                m: Option<Monoid>,
                f: &dyn Fn(&[ExprId]) -> ExprId| {
        g.comp(out, red, args, m, |x| vec![f(x)])
            .unwrap_or_else(|e| panic!("sampler comp {out:?}/{red:?}: {e:?}"))
    };
    // the K best (id, logit) pairs, best first
    let vid = g.iota(&[t, v], 1);
    let pair = move |x: &[ExprId]| {
        let mut o = vec![x[0], x[1]];
        for _ in 1..kc {
            o.push(lit(0.0));
            o.push(lit(-(f32::MAX as f64)));
        }
        o
    };
    // each chunk's K best, then the K best of those (the merge is exact: same set, same order)
    let cand1 = g
        .comp_tuple(
            &[t, nbd, Dim::Fixed(2 * kc)],
            &[Dim::Fixed(cl)],
            vec![chunked(vid), chunked(logits)],
            Some(Monoid::topk(kc)),
            pair,
        )
        .unwrap_or_else(|e| panic!("sampler top-k (chunks): {e:?}"));
    let cand = g
        .comp_tuple(
            &[t, Dim::Fixed(2 * kc)],
            &[nbd, kd],
            vec![
                arg(cand1, vec![u(3, 0), u(3, 1), form(3, &[(2, 2)], 0)]),
                arg(cand1, vec![u(3, 0), u(3, 1), form(3, &[(2, 2)], 1)]),
            ],
            Some(Monoid::topk(kc)),
            pair,
        )
        .unwrap_or_else(|e| panic!("sampler top-k (merge): {e:?}"));
    // candidate r's logit / id, in a rank-`rank` space whose axis `ax` is r
    let cv = |rank: usize, ax: usize| arg(cand, vec![u(rank, 0), form(rank, &[(ax, 2)], 1)]);
    let ci = |rank: usize, ax: usize| arg(cand, vec![u(rank, 0), form(rank, &[(ax, 2)], 0)]);
    let row = |x: TensorId, rank: usize| arg(x, vec![u(rank, 0)]);
    let m = comp(g, &[t], &[], vec![arg(cand, vec![u(1, 0), Form::constant(1, 1)])], None, &|x| x[0]);
    let first = comp(g, &[t], &[], vec![arg(cand, vec![u(1, 0), Form::constant(1, 0)])], None, &|x| x[0]);
    let kf = kc as f64;
    // top-k threshold: candidate min(k, K) − 1 (k = 0: K − 1)
    let rk = g.iota(&[t, kd], 1);
    let theta_k = comp(
        g,
        &[t],
        &[kd],
        vec![cv(2, 1), arg(rk, vec![u(2, 0), u(2, 1)]), p(2, TOP_K)],
        Some(Monoid::sum()),
        &move |x| {
            let k = select(x[2] - lit(0.5), x[2].min(lit(kf)), lit(kf));
            let d = x[1] - (k - lit(1.0));
            select(lit(0.25) - d * d, x[0], lit(0.0))
        },
    );
    // the top-k set's floor over the whole vocabulary: top_k = 0 keeps every logit within
    // `range` of the max (the host sampler's set), else the k-th candidate
    let range = 64.0;
    let theta_k = comp(
        g,
        &[t],
        &[],
        vec![row(theta_k, 1), row(m, 1), p(1, TOP_K)],
        None,
        &move |x| select(x[2] - lit(0.5), x[0], x[1] - lit(range)),
    );
    // tempered weights of the top-k set, their total, and prefix masses
    let temp = |x: ExprId| select(x, x, lit(1.0));
    let w = comp(
        g,
        &[t, kd],
        &[],
        vec![cv(2, 1), row(m, 2), row(theta_k, 2), p(2, TEMPERATURE)],
        None,
        &move |x| select(x[2] - x[0], lit(0.0), ((x[0] - x[1]) / temp(x[3])).exp()),
    );
    // the total over the whole top-k set (all of the vocabulary's kept logits, not just the K)
    let z1 = comp(
        g,
        &[t, nbd],
        &[Dim::Fixed(cl)],
        vec![
            chunked(logits),
            row(m, 3),
            row(theta_k, 3),
            p(3, TEMPERATURE),
        ],
        Some(Monoid::sum()),
        &move |x| select(x[2] - x[0], lit(0.0), ((x[0] - x[1]) / temp(x[3])).exp()),
    );
    let z = comp(g, &[t], &[nbd], vec![arg(z1, vec![u(2, 0), u(2, 1)])], Some(Monoid::sum()), &|x| x[0]);
    let r3 = g.iota(&[t, kd, kd], 1);
    let q3 = g.iota(&[t, kd, kd], 2);
    // mass strictly before rank r
    let before = comp(
        g,
        &[t, kd],
        &[kd],
        vec![
            arg(w, vec![u(3, 0), u(3, 2)]),
            arg(r3, vec![u(3, 0), u(3, 1), u(3, 2)]),
            arg(q3, vec![u(3, 0), u(3, 1), u(3, 2)]),
        ],
        Some(Monoid::sum()),
        &|x| select(x[1] - x[2] - lit(0.5), x[0], lit(0.0)),
    );
    // nucleus threshold: the smallest logit among ranks with mass-before < top_p · z
    // (as a max of negated logits)
    let neg_th = comp(
        g,
        &[t],
        &[kd],
        vec![
            cv(2, 1),
            arg(before, vec![u(2, 0), u(2, 1)]),
            row(z, 2),
            p(2, TOP_P),
            row(theta_k, 2),
        ],
        Some(Monoid::max()),
        &|x| {
            let kept = x[3] * x[2] - x[1];
            // inside the nucleus and the top-k set
            select(kept, select(x[4] - x[0], lit(-1e30), -x[0]), lit(-1e30))
        },
    );
    let one = |x: TensorId| arg(x, vec![u(1, 0)]);
    let threshold = comp(
        g,
        &[t],
        &[],
        vec![one(neg_th), one(theta_k), p(1, TOP_P)],
        None,
        &|x| select(x[2], select(lit(1.0) - x[2], -x[0], x[1]), x[1]),
    );
    // Gumbel-max over the kept candidates
    let pos2 = g.iota(&[t, kd], 0);
    let bestk = g
        .comp_tuple(
            &[t, Dim::Fixed(2)],
            &[kd],
            vec![
                cv(2, 1),
                ci(2, 1),
                arg(pos2, vec![u(2, 0), u(2, 1)]),
                row(m, 2),
                row(threshold, 2),
                p(2, TEMPERATURE),
                p(2, SEED),
            ],
            Some(Monoid::argmax()),
            |x| {
                let (l, id, pos, m, th, tp, seed) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
                let uu = uniform(seed, pos, id);
                let gumbel = -(-(uu.ln())).ln();
                vec![id, select(th - l, lit(-1e30), (l - m) / temp(tp) + gumbel)]
            },
        )
        .unwrap();
    // no cut at all: Gumbel-max over the whole vocabulary
    let posv = g.iota(&[t, v], 0);
    let bestv1 = g
        .comp_tuple(
            &[t, nbd, Dim::Fixed(2)],
            &[Dim::Fixed(cl)],
            vec![
                chunked(logits),
                chunked(vid),
                chunked(posv),
                row(m, 3),
                p(3, TEMPERATURE),
                p(3, SEED),
            ],
            Some(Monoid::argmax()),
            |x| {
                let (l, id, pos, m, tp, seed) = (x[0], x[1], x[2], x[3], x[4], x[5]);
                let uu = uniform(seed, pos, id);
                vec![id, (l - m) / temp(tp) - (-(uu.ln())).ln()]
            },
        )
        .unwrap();
    // chunks in id order: argmax's first-index tie rule keeps the lowest id
    let bestv = g
        .comp_tuple(
            &[t, Dim::Fixed(2)],
            &[nbd],
            vec![
                arg(bestv1, vec![u(2, 0), u(2, 1), Form::constant(2, 0)]),
                arg(bestv1, vec![u(2, 0), u(2, 1), Form::constant(2, 1)]),
            ],
            Some(Monoid::argmax()),
            |x| vec![x[0], x[1]],
        )
        .unwrap();
    let token = comp(
        g,
        &[t],
        &[],
        vec![
            arg(bestk, vec![u(1, 0), Form::constant(1, 0)]),
            arg(bestv, vec![u(1, 0), Form::constant(1, 0)]),
            one(first),
            p(1, TEMPERATURE),
            p(1, TOP_K),
            p(1, TOP_P),
        ],
        None,
        &|x| {
            // all of the vocabulary: top_k = 0 (< 0.5) and top_p ≥ 1 (not in (0, 1))
            let nucleus = select(x[5], select(lit(1.0) - x[5], lit(1.0), lit(0.0)), lit(0.0));
            let full = select(lit(0.5) - x[4], lit(1.0) - nucleus, lit(0.0));
            select(x[3], select(full - lit(0.5), x[1], x[0]), x[2])
        },
    );
    Sampler {
        params,
        token,
        threshold,
    }
}

/// The host sampler with the same semantics, exact sets (sort-based), for
/// checks: `None` params fields mean off. Returns `(token, kept set size)`.
pub fn sample_host(l: &[f64], params: [f64; 4], pos: u64, range: f64) -> (usize, usize) {
    let [tp, top_p, top_k, seed] = params;
    let first_max = (0..l.len())
        .max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a)))
        .unwrap();
    if tp <= 0.0 {
        return (first_max, 1);
    }
    let m = l[first_max];
    let mut order: Vec<usize> = (0..l.len()).filter(|&i| l[i] >= m - range).collect();
    order.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
    let k = top_k as usize;
    if k >= 1 && k < order.len() {
        let kth = l[order[k - 1]];
        order.retain(|&i| l[i] >= kth);
    }
    if top_p > 0.0 && top_p < 1.0 {
        let w: Vec<f64> = order.iter().map(|&i| ((l[i] - m) / tp).exp()).collect();
        let z: f64 = w.iter().sum();
        let mut acc = 0.0;
        let mut keep = order.len();
        for (j, wj) in w.iter().enumerate() {
            acc += wj;
            if acc >= top_p * z {
                keep = j + 1;
                break;
            }
        }
        let last = l[order[keep - 1]];
        order.retain(|&i| l[i] >= last);
    }
    let n = order.len();
    let best = order
        .into_iter()
        .map(|i| {
            let u = uniform_host(seed as u64, pos, i as u64);
            (i, (l[i] - m) / tp - (-(u.ln())).ln())
        })
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .unwrap();
    (best.0, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: &str = "T";

    fn graph(v: usize, spec: SampleSpec) -> (TensorGraph, TensorId, Sampler) {
        let mut g = TensorGraph::new();
        let l = g.input(&[Dim::Sym(SEQ), Dim::Fixed(v)]);
        let s = append_sampler(&mut g, l, spec);
        (g, l, s)
    }

    fn run(v: usize, rows: &[Vec<f64>], params: [f64; 4]) -> (Vec<usize>, Vec<f64>) {
        let (g, _, s) = graph(v, SampleSpec::default());
        let b = g.bind(&[(SEQ, rows.len())]).unwrap();
        let flat: Vec<f64> = rows.concat();
        let out = b.eval(&[flat, params.to_vec()], &[s.token, s.threshold]);
        (out[0].iter().map(|&x| x as usize).collect(), out[1].clone())
    }

    fn rows(n: usize, v: usize, salt: u64) -> Vec<Vec<f64>> {
        (0..n)
            .map(|r| {
                (0..v)
                    .map(|i| {
                        let u = uniform_host(salt, r as u64, i as u64);
                        // a few heavy logits, ties on purpose
                        8.0 * u * u - 3.0 + if i % 7 == 0 { 0.5 } else { 0.0 }
                    })
                    .collect()
            })
            .collect()
    }

    /// The hash is the same function in the graph and on the host, and its
    /// outputs look uniform (mean, variance, a 16-bin histogram).
    #[test]
    fn hash_matches_host_and_is_uniform() {
        let mut g = TensorGraph::new();
        let n = 4096;
        let a = g.input(&[Dim::Fixed(n)]);
        let b2 = g.input(&[Dim::Fixed(n)]);
        let out = g
            .comp(
                &[Dim::Fixed(n)],
                &[],
                vec![arg(a, vec![u(1, 0)]), arg(b2, vec![u(1, 0)])],
                None,
                |x| vec![uniform(lit(12345.0), x[0], x[1])],
            )
            .unwrap();
        let pos: Vec<f64> = (0..n).map(|i| (i / 64) as f64 * 977.0).collect();
        let ids: Vec<f64> = (0..n).map(|i| (i % 64) as f64 * 2411.0).collect();
        let got = g.eval(&[pos.clone(), ids.clone()], &[out]).remove(0);
        let mut hist = [0usize; 16];
        for i in 0..n {
            let h = uniform_host(12345, pos[i] as u64, ids[i] as u64);
            assert_eq!(got[i], h, "hash {i}");
            hist[(h * 16.0) as usize] += 1;
        }
        let mean = got.iter().sum::<f64>() / n as f64;
        let var = got.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let chi: f64 = hist
            .iter()
            .map(|&c| (c as f64 - n as f64 / 16.0).powi(2) / (n as f64 / 16.0))
            .sum();
        eprintln!("hash: mean {mean:.4} var {var:.4} (1/12 = 0.0833) chi2(15) {chi:.1} hist {hist:?}");
        assert!((mean - 0.5).abs() < 0.02 && (var - 1.0 / 12.0).abs() < 0.01 && chi < 40.0);
    }

    /// Greedy is the host's first maximum, ties included.
    #[test]
    fn greedy_is_first_argmax() {
        let v = 300;
        let mut rs = rows(12, v, 1);
        rs[3][17] = 100.0;
        rs[3][201] = 100.0; // tie: the lower id wins
        let (tok, _) = run(v, &rs, [0.0, 0.9, 40.0, 7.0]);
        for (r, row) in rs.iter().enumerate() {
            assert_eq!(tok[r], sample_host(row, [0.0, 0.0, 0.0, 0.0], r as u64, 64.0).0);
        }
        assert_eq!(tok[3], 17);
    }

    fn run_topk(v: usize, rows: &[Vec<f64>], params: [f64; 4]) -> (Vec<usize>, Vec<f64>) {
        let mut g = TensorGraph::new();
        let l = g.input(&[Dim::Sym(SEQ), Dim::Fixed(v)]);
        let s = append_sampler_topk(&mut g, l);
        let b = g.bind(&[(SEQ, rows.len())]).unwrap();
        let per_row: Vec<f64> = rows.iter().flat_map(|_| params).collect();
        let out = b.eval(&[rows.concat(), per_row], &[s.token, s.threshold]);
        (out[0].iter().map(|&x| x as usize).collect(), out[1].clone())
    }

    /// The top-K sampler: greedy is the first maximum; sampled tokens and kept sets equal
    /// the exact host sampler's wherever its kept set fits in the K candidates (and the
    /// uncut case, top_k 0 with top_p 1, draws over the whole vocabulary like the host).
    #[test]
    fn topk_sampler_matches_exact_host_sampler() {
        check_topk(257);
    }

    /// V = 1000: the vocabulary reductions split in 125 chunks of 8.
    #[test]
    fn topk_sampler_split_matches_exact_host_sampler() {
        check_topk(1000);
    }

    fn check_topk(v: usize) {
        let mut rs = rows(40, v, 2);
        rs[3][17] = 100.0;
        rs[3][201] = 100.0;
        let (tok, _) = run_topk(v, &rs, [0.0, 0.9, 40.0, 7.0]);
        for (r, row) in rs.iter().enumerate() {
            assert_eq!(tok[r], sample_host(row, [0.0; 4], r as u64, 64.0).0, "greedy row {r}");
        }
        assert_eq!(tok[3], 17);
        let rs = rows(40, v, 2);
        let (mut n, mut skipped) = (0, 0);
        for &tp in &[0.3, 0.7, 1.0, 1.6] {
            for &k in &[0.0, 1.0, 5.0, 40.0, 64.0] {
                for &pp in &[1.0, 0.95, 0.5, 0.1] {
                    for &seed in &[1.0, 99.0] {
                        let params = [tp, pp, k, seed];
                        let (tok, th) = run_topk(v, &rs, params);
                        let uncut = k == 0.0 && pp >= 1.0;
                        for (r, row) in rs.iter().enumerate() {
                            let (want, kept) = sample_host(row, params, r as u64, 64.0);
                            if !uncut && kept > TOPK_CANDIDATES {
                                skipped += 1;
                                continue;
                            }
                            if !uncut {
                                let got_kept = row.iter().filter(|&&x| x >= th[r]).count();
                                assert_eq!(got_kept, kept, "kept set {params:?} row {r}");
                            }
                            assert_eq!(tok[r], want, "token {params:?} row {r}");
                            n += 1;
                        }
                    }
                }
            }
        }
        eprintln!("top-K sampler: {n} draws equal to the exact host sampler ({skipped} nuclei past K skipped)");
        assert!(n > 4000);
    }

    /// Sampling matches the exact (sort-based) host sampler: same kept sets
    /// and same draws, over temperatures, top-k, top-p and seeds.
    #[test]
    fn sampling_matches_exact_host_sampler() {
        let v = 257;
        let rs = rows(40, v, 2);
        let mut n = 0;
        for &tp in &[0.3, 0.7, 1.0, 1.6] {
            for &k in &[0.0, 1.0, 5.0, 40.0] {
                for &pp in &[1.0, 0.95, 0.5, 0.1] {
                    for &seed in &[1.0, 99.0] {
                        let params = [tp, pp, k, seed];
                        let (tok, th) = run(v, &rs, params);
                        for (r, row) in rs.iter().enumerate() {
                            let (want, kept) = sample_host(row, params, r as u64, 64.0);
                            let got_kept = row.iter().filter(|&&x| x >= th[r]).count();
                            assert_eq!(got_kept, kept, "kept set {params:?} row {r}");
                            assert_eq!(tok[r], want, "token {params:?} row {r}");
                            n += 1;
                        }
                    }
                }
            }
        }
        eprintln!("{n} draws equal to the exact host sampler");
    }

    /// Draws over many positions follow the renormalized tempered softmax of
    /// the kept set (one fixed logit row repeated; position = RNG counter).
    #[test]
    fn draws_follow_the_distribution() {
        let v = 12;
        let base: Vec<f64> = (0..v).map(|i| (i as f64 * 0.37).sin() * 2.0).collect();
        let n = 20000;
        let rs = vec![base.clone(); n];
        for params in [[1.0, 1.0, 0.0, 5.0], [0.7, 0.9, 6.0, 11.0]] {
            let (tok, th) = run(v, &rs, params);
            let kept: Vec<usize> = (0..v).filter(|&i| base[i] >= th[0]).collect();
            let m = base.iter().cloned().fold(f64::MIN, f64::max);
            let w: Vec<f64> = kept.iter().map(|&i| ((base[i] - m) / params[0]).exp()).collect();
            let z: f64 = w.iter().sum();
            let mut chi = 0.0;
            for (j, &i) in kept.iter().enumerate() {
                let c = tok.iter().filter(|&&t| t == i).count() as f64;
                let e = w[j] / z * n as f64;
                chi += (c - e).powi(2) / e;
            }
            let outside = tok.iter().filter(|t| !kept.contains(t)).count();
            eprintln!(
                "params {params:?}: kept {} of {v}, chi2({}) = {chi:.1}, outside {outside}",
                kept.len(),
                kept.len() - 1
            );
            assert_eq!(outside, 0);
            assert!(chi < 3.0 * kept.len() as f64 + 10.0, "chi2 {chi}");
        }
    }

    /// Same seed, same tokens; another seed, other tokens.
    #[test]
    fn seeded_and_reproducible() {
        let v = 64;
        let rs = rows(64, v, 3);
        let a = run(v, &rs, [1.0, 1.0, 0.0, 42.0]).0;
        let b = run(v, &rs, [1.0, 1.0, 0.0, 42.0]).0;
        let c = run(v, &rs, [1.0, 1.0, 0.0, 43.0]).0;
        assert_eq!(a, b);
        let diff = a.iter().zip(&c).filter(|(x, y)| x != y).count();
        eprintln!("seed 42 vs 43: {diff} of {} tokens differ", a.len());
        assert!(diff > a.len() / 4);
    }
}
