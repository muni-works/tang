//! Algorithm discovery: lifting multi-pass reductions and sequential
//! recurrences to single associative monoids.
//!
//! Three searches, each a candidate generator (symbolic, in `tang-expr`)
//! followed by a verifier (fuzzing in f64, and in the [`TensorGraph`]
//! interpreter):
//!
//! - **Online lifts** ([`online_lift`]). A two-pass reduction
//!   `y = Σᵢ g(xᵢ, ρ(R₁(x)))` — a first-pass monoid `R₁`, a reference `ρ`
//!   read off its state (a max, a mean, a norm), and a second-pass sum —
//!   becomes one pass when the family `{g(·, a)}` indexed by the
//!   reference is a finite-dimensional space closed under change of
//!   reference: `D(x, a') = C(a, a')·D(x, a)` for a vector `D` of basis
//!   functions with `D₀ = g`. The state is then `(R₁ state, Σ D)`, and a
//!   combine moves both halves' sums to the new reference with `C`.
//!   Online softmax (`D = exp(s − m)`, `C = exp(m − m')`) and Welford/Chan
//!   (`D = ((x − μ)ʲ)ⱼ₌₀..₂`, `C` the binomial shift) are the two classical
//!   instances. The search differentiates `g` in the reference
//!   coordinates `b = φ(a)` (`φ` = identity or `ln`) until the derivatives
//!   close under constant-coefficient linear relations `∂ᵢD = AᵢD`
//!   (checked numerically at sampled points with a least-squares fit, the
//!   relation's dependence structure checked symbolically); then
//!   `C = exp(Σ δᵢAᵢ)`, emitted in closed form from the joint spectral
//!   decomposition of the commuting `Aᵢ` (real spectra). By the
//!   classical theorem on translation-invariant function spaces, the
//!   closing families are exactly the exponential polynomials in `b`, so
//!   "not closed" is a statement about a well-defined family.
//! - **Scan lifts** ([`scan_lift`]). A recurrence `h' = f(h, x)` becomes
//!   an associative scan when `f` is, in some coordinates, an element of
//!   a monoid of maps with a finite representation: affine maps (the
//!   Jacobian `∂f/∂h` is independent of `h` — checked structurally on the
//!   symbolic derivative), affine after an elementwise change of
//!   coordinates (`exp`, `ln`, `1/h`), max-plus affine (tropical), or
//!   Möbius (the Schwarzian derivative vanishes), or affine in a small
//!   set of observables (`hᵢhⱼ`, ratios — a finite Koopman lift). When a
//!   family fails, the search records a witness (a point with
//!   `∂²f/∂h² ≠ 0`, a nonzero Schwarzian, a counterexample), so "no lift
//!   in the family" comes with evidence.
//! - **Chunked lifts** ([`chunk_lift`]). An affine scan on a matrix
//!   state `S_t = A_t·S_{t−1} + U_t` with a structured `A_t` (identity,
//!   scalar decay, diagonal decay, scalar decay × `(I − β·k·kᵀ)`) and a
//!   linear readout gets the chunked matmul form (SSD, GLA, the WY/UT
//!   form of the delta rule), verified against the sequential scan; the
//!   scalar-decay families also map onto [`TensorGraph::delta_scan`], whose
//!   readout the CUDA chunked schedule recognizes.
//!
//! The catalog (`tensor_discover_catalog.rs`) runs all three over the
//! operator zoo; `docs/discovery.md` has the results.

#![allow(
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::type_complexity,
    clippy::neg_cmp_op_on_partial_ord
)]

use std::rc::Rc;

use tang_expr::{trace, ExprGraph, ExprId, Node};

use super::{Arg, Form, Monoid, TensorGraph, TensorId, View};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// A symbolic builder: input `ExprId`s to output `ExprId`s, run inside a
/// `tang-expr` trace.
pub type Build = Rc<dyn Fn(&[ExprId]) -> Vec<ExprId>>;
/// A sampler of one element's inputs.
pub type Sampler = Rc<dyn Fn(&mut Rng) -> Vec<f64>>;

pub fn build(f: impl Fn(&[ExprId]) -> Vec<ExprId> + 'static) -> Build {
    Rc::new(f)
}

pub fn sampler(f: impl Fn(&mut Rng) -> Vec<f64> + 'static) -> Sampler {
    Rc::new(f)
}

/// xorshift64*.
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// A traced function, evaluated in f64.
#[derive(Clone)]
pub struct Fun {
    pub g: ExprGraph,
    pub outs: Vec<ExprId>,
}

impl Fun {
    pub fn trace(n_in: usize, f: &Build) -> Self {
        let (g, outs) = trace(|| {
            let v: Vec<ExprId> = (0..n_in).map(|i| ExprId::var(i as u16)).collect();
            f(&v)
        });
        Self { g, outs }
    }
    pub fn call(&self, x: &[f64]) -> Vec<f64> {
        self.g.eval_many(&self.outs, x)
    }
}

/// Re-emit `outs` of `g` into the current trace with `Var(i)` := `vars[i]`.
pub fn inline(g: &ExprGraph, outs: &[ExprId], vars: &[ExprId]) -> Vec<ExprId> {
    let nodes = g.nodes_slice();
    let need = outs
        .iter()
        .map(|o| o.index() as usize + 1)
        .max()
        .unwrap_or(0);
    let mut m: Vec<ExprId> = Vec::with_capacity(need);
    let w = |f: &mut dyn FnMut(&mut ExprGraph) -> ExprId| tang_expr::with_graph(f);
    for n in &nodes[..need] {
        let id = match *n {
            Node::Var(i) => vars[i as usize],
            Node::Lit(bits) => lit(f64::from_bits(bits)),
            Node::Add(a, b) => {
                let (a, b) = (m[a.index() as usize], m[b.index() as usize]);
                w(&mut |g| g.add(a, b))
            }
            Node::Mul(a, b) => {
                let (a, b) = (m[a.index() as usize], m[b.index() as usize]);
                w(&mut |g| g.mul(a, b))
            }
            Node::Neg(a) => {
                let a = m[a.index() as usize];
                w(&mut |g| g.neg(a))
            }
            Node::Recip(a) => {
                let a = m[a.index() as usize];
                w(&mut |g| g.recip(a))
            }
            Node::Sqrt(a) => {
                let a = m[a.index() as usize];
                w(&mut |g| g.sqrt(a))
            }
            Node::Sin(a) => {
                let a = m[a.index() as usize];
                w(&mut |g| g.sin(a))
            }
            Node::Exp2(a) => {
                let a = m[a.index() as usize];
                w(&mut |g| g.exp2(a))
            }
            Node::Log2(a) => {
                let a = m[a.index() as usize];
                w(&mut |g| g.log2(a))
            }
            Node::Atan2(a, b) => {
                let (a, b) = (m[a.index() as usize], m[b.index() as usize]);
                w(&mut |g| g.atan2(a, b))
            }
            Node::Select(c, a, b) => {
                let (c, a, b) = (
                    m[c.index() as usize],
                    m[a.index() as usize],
                    m[b.index() as usize],
                );
                w(&mut |g| g.select(c, a, b))
            }
        };
        m.push(id);
    }
    outs.iter().map(|o| m[o.index() as usize]).collect()
}

pub fn lit(v: f64) -> ExprId {
    tang_expr::with_graph(|g| g.lit(v))
}

pub fn sel(c: ExprId, a: ExprId, b: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(c, a, b))
}

/// `max` via select (exact at `±∞`).
pub fn emax(a: ExprId, b: ExprId) -> ExprId {
    sel(a - b, a, b)
}

pub fn eexp(a: ExprId) -> ExprId {
    use tang::Scalar;
    a.exp()
}

pub fn eln(a: ExprId) -> ExprId {
    use tang::Scalar;
    a.ln()
}

/// Snap `v` to a nearby simple rational (denominator ≤ 24) when within
/// `1e-9` relative: least-squares coefficients come out as `-0.9999999999`.
fn snap(v: f64) -> f64 {
    if v.abs() < 1e-11 {
        return 0.0;
    }
    for d in 1..=24 {
        let n = (v * d as f64).round();
        if ((n / d as f64) - v).abs() <= 1e-9 * v.abs().max(1.0) {
            return n / d as f64;
        }
    }
    v
}

pub type Mat = Vec<Vec<f64>>;

fn eye(n: usize) -> Mat {
    (0..n)
        .map(|i| (0..n).map(|j| (i == j) as u8 as f64).collect())
        .collect()
}

fn matmul(a: &Mat, b: &Mat) -> Mat {
    let (n, k, m) = (a.len(), b.len(), b.first().map_or(0, |r| r.len()));
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for l in 0..k {
            if a[i][l] != 0.0 {
                for j in 0..m {
                    c[i][j] += a[i][l] * b[l][j];
                }
            }
        }
    }
    c
}

fn madd(a: &Mat, b: &Mat, s: f64) -> Mat {
    a.iter()
        .zip(b)
        .map(|(r, q)| r.iter().zip(q).map(|(x, y)| x + s * y).collect())
        .collect()
}

fn mnorm(a: &Mat) -> f64 {
    a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
}

/// Least squares `min ‖M·c − y‖` by modified Gram–Schmidt on the columns
/// of `M` (`rows × r`). Returns `(c, residual norm)`.
fn lstsq(m: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64) {
    let rows = y.len();
    let r = m.first().map_or(0, |x| x.len());
    // column-major copy
    let mut q: Vec<Vec<f64>> = (0..r)
        .map(|j| (0..rows).map(|i| m[i][j]).collect())
        .collect();
    let mut rr = vec![vec![0.0; r]; r];
    let mut keep = vec![true; r];
    for j in 0..r {
        for i in 0..j {
            if !keep[i] {
                continue;
            }
            let d: f64 = (0..rows).map(|t| q[i][t] * q[j][t]).sum();
            rr[i][j] = d;
            for t in 0..rows {
                q[j][t] -= d * q[i][t];
            }
        }
        let nn: f64 = q[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        let orig: f64 = (0..rows).map(|i| m[i][j] * m[i][j]).sum::<f64>().sqrt();
        if nn <= 1e-10 * orig.max(1e-300) {
            keep[j] = false;
            continue;
        }
        rr[j][j] = nn;
        for t in 0..rows {
            q[j][t] /= nn;
        }
    }
    let mut res = y.to_vec();
    let mut qy = vec![0.0; r];
    for j in 0..r {
        if !keep[j] {
            continue;
        }
        qy[j] = (0..rows).map(|t| q[j][t] * res[t]).sum();
        for t in 0..rows {
            res[t] -= qy[j] * q[j][t];
        }
    }
    let mut c = vec![0.0; r];
    for j in (0..r).rev() {
        if !keep[j] {
            continue;
        }
        let mut s = qy[j];
        for l in j + 1..r {
            s -= rr[j][l] * c[l];
        }
        c[j] = s / rr[j][j];
    }
    (c, res.iter().map(|x| x * x).sum::<f64>().sqrt())
}

/// Real eigenvalue groups `(λ, multiplicity)` of `a`, or `None` when the
/// spectrum is complex. Characteristic polynomial by Faddeev–LeVerrier,
/// roots by Durand–Kerner, clustered.
fn real_spectrum(a: &Mat) -> Option<Vec<(f64, usize)>> {
    let n = a.len();
    if n == 0 {
        return Some(vec![]);
    }
    // one eigenvalue: A = λI + nilpotent (repeated roots are where
    // root finding is least accurate, and the common case)
    let lam = (0..n).map(|i| a[i][i]).sum::<f64>() / n as f64;
    let shifted = madd(a, &eye(n), -lam);
    let mut pw = shifted.clone();
    for _ in 1..n {
        pw = matmul(&pw, &shifted);
    }
    if mnorm(&pw) <= 1e-9 * (1.0 + mnorm(a)).powi(n as i32) {
        return Some(vec![(snap(lam), n)]);
    }
    // c[k]: coefficient of λ^(n-k), c[0] = 1
    let mut c = vec![1.0];
    let mut mk = vec![vec![0.0; n]; n];
    for k in 1..=n {
        let am = matmul(a, &mk);
        mk = madd(&am, &eye(n), c[k - 1]);
        let amk = matmul(a, &mk);
        let tr: f64 = (0..n).map(|i| amk[i][i]).sum();
        c.push(-tr / k as f64);
    }
    // Durand–Kerner on monic poly
    type C = (f64, f64);
    let mul = |x: C, y: C| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let sub = |x: C, y: C| (x.0 - y.0, x.1 - y.1);
    let div = |x: C, y: C| {
        let d = y.0 * y.0 + y.1 * y.1;
        ((x.0 * y.0 + x.1 * y.1) / d, (x.1 * y.0 - x.0 * y.1) / d)
    };
    let eval = |z: C| {
        let mut v = (1.0, 0.0);
        for &ck in &c[1..] {
            v = mul(v, z);
            v.0 += ck;
        }
        v
    };
    let scale = 1.0 + c.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let mut z: Vec<C> = (0..n)
        .map(|k| {
            let t = 0.4 + 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (scale * 0.5 * t.cos(), scale * 0.5 * t.sin())
        })
        .collect();
    for _ in 0..2000 {
        let mut moved = 0.0f64;
        for i in 0..n {
            let mut den = (1.0, 0.0);
            for j in 0..n {
                if i != j {
                    den = mul(den, sub(z[i], z[j]));
                }
            }
            let step = div(eval(z[i]), den);
            z[i] = sub(z[i], step);
            moved = moved.max(step.0.abs() + step.1.abs());
        }
        if moved < 1e-15 {
            break;
        }
    }
    // cluster (repeated roots converge to ~ε^(1/m))
    let mut groups: Vec<(f64, f64, usize)> = vec![]; // (sum re, max |im|, count)
    let mut used = vec![false; n];
    for i in 0..n {
        if used[i] {
            continue;
        }
        let mut s = (z[i].0, z[i].1.abs(), 1);
        used[i] = true;
        for j in i + 1..n {
            if !used[j] && (z[j].0 - z[i].0).abs() + (z[j].1.abs() - z[i].1.abs()).abs() < 1e-3 {
                used[j] = true;
                s.0 += z[j].0;
                s.1 = s.1.max(z[j].1.abs());
                s.2 += 1;
            }
        }
        groups.push(s);
    }
    let mut out = vec![];
    for (sr, im, cnt) in groups {
        let mean_im = im;
        if mean_im > 1e-3 * cnt as f64 {
            // a cluster of a conjugate pair has mean im 0 but each root
            // is far from the axis: complex
            return None;
        }
        out.push((snap(sr / cnt as f64), cnt));
    }
    Some(out)
}

/// The spectral projector of `a` onto the generalized eigenspace of
/// `lam` (Hermite interpolation: `e(x) ≡ 1 mod (x−λ)^m`, `≡ 0 mod
/// (x−μ)^{m_μ}`, `P = e(A)`).
fn projector(a: &Mat, spec: &[(f64, usize)], which: usize) -> Mat {
    let n = a.len();
    // unknown coefficients e(x) = Σ c_j x^j, j < n
    let mut rows: Vec<Vec<f64>> = vec![];
    let mut rhs = vec![];
    for (gi, &(mu, mult)) in spec.iter().enumerate() {
        for d in 0..mult {
            // d-th derivative of x^j at mu
            let row: Vec<f64> = (0..n)
                .map(|j| {
                    if j < d {
                        0.0
                    } else {
                        let mut f = 1.0;
                        for t in 0..d {
                            f *= (j - t) as f64;
                        }
                        f * mu.powi((j - d) as i32)
                    }
                })
                .collect();
            rows.push(row);
            rhs.push(if gi == which && d == 0 { 1.0 } else { 0.0 });
        }
    }
    let (cs, _) = lstsq(&rows, &rhs);
    let mut p = vec![vec![0.0; n]; n];
    let mut pw = eye(n);
    for &cj in &cs {
        p = madd(&p, &pw, cj);
        pw = matmul(&pw, a);
    }
    p
}

// ---------------------------------------------------------------------------
// Online lifts
// ---------------------------------------------------------------------------

/// Coordinate of a reference parameter in which its change of reference
/// is a translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coord {
    /// `b = a`.
    Add,
    /// `b = ln a` (a scale: norms, normalizers, absmax).
    Log,
}

/// A two-pass reduction `y = Σᵢ g(xᵢ, ρ(R₁ᵢ lift(xᵢ)))`.
#[derive(Clone)]
pub struct OnlineSpec {
    pub name: String,
    /// Scalars per element.
    pub n_x: usize,
    /// The first pass.
    pub r1: Monoid,
    /// Element → `R₁` element (`r1.width` values).
    pub lift: Build,
    /// `R₁` state → reference parameters.
    pub reference: Build,
    /// `R₁` state → one value, `> 0` iff the state is not the identity.
    pub occupied: Build,
    /// `(x…, reference…)` → the values summed by the second pass.
    pub g: Build,
    /// Element sampler.
    pub sample: Sampler,
}

impl OnlineSpec {
    pub fn n_ref(&self) -> usize {
        let f = Fun::trace(self.r1.width, &self.reference);
        f.outs.len()
    }
    pub fn n_out(&self) -> usize {
        Fun::trace(self.n_x + self.n_ref(), &self.g).outs.len()
    }
}

/// A found online lift.
#[derive(Clone)]
pub struct OnlineLift {
    pub coords: Vec<Coord>,
    /// Basis size `r` (the second-pass state width).
    pub dim: usize,
    /// `∂_{bᵢ} D = Aᵢ D`.
    pub gens: Vec<Mat>,
    /// Joint spectral blocks: eigenvalue per parameter and projector.
    pub blocks: Vec<(Vec<f64>, Mat)>,
    /// Output `c` is `Σⱼ readout[c][j]·Sⱼ`.
    pub readout: Mat,
    /// The one-pass monoid: state `(R₁ state, S₀…S_{r−1})`.
    pub monoid: Monoid,
    /// Element → monoid element.
    pub element: Build,
    /// Basis functions, printed.
    pub basis: Vec<String>,
    /// Whether every `∂ᵢDⱼ − Σ AᵢDₗ` relation was confirmed structurally
    /// (`simplify` to a literal 0) rather than only numerically.
    pub symbolic: bool,
    /// `g(x, a) = u(a)·v(x)` exactly (rank 1 in `(x, a)`): the correction
    /// is a pure rescale and could also be hoisted to an epilogue.
    pub separable: bool,
}

#[derive(Clone, Debug)]
pub enum OnlineFail {
    /// The derivative closure exceeded `cap` dimensions in every
    /// coordinate choice.
    NotClosed { cap: usize },
    /// Closed, but a generator has a complex spectrum (rotations):
    /// not emitted.
    Complex,
    /// Closed and emitted, but the fuzz rejected it (e.g. the closure
    /// was only valid almost everywhere: kinks, thresholds).
    Rejected { err: f64 },
}

const CAP: usize = 8;

/// Search for an online lift of `spec` (see the module docs).
pub fn online_lift(spec: &OnlineSpec) -> Result<OnlineLift, OnlineFail> {
    let p = spec.n_ref();
    let mut last = OnlineFail::NotClosed { cap: CAP };
    for mask in 0..(1usize << p) {
        let coords: Vec<Coord> = (0..p)
            .map(|i| {
                if mask >> i & 1 == 1 {
                    Coord::Log
                } else {
                    Coord::Add
                }
            })
            .collect();
        match try_coords(spec, &coords) {
            Ok(l) => match verify_online(spec, &l, 7, 60) {
                e if e <= 1e-8 => return Ok(l),
                e => last = OnlineFail::Rejected { err: e },
            },
            Err(OnlineFail::Complex) => last = OnlineFail::Complex,
            Err(e @ OnlineFail::Rejected { .. }) => last = e,
            Err(OnlineFail::NotClosed { .. }) => {}
        }
    }
    Err(last)
}

/// Sample realistic `(x, b)` points: `b` from folds of 1..8 elements.
fn sample_points(spec: &OnlineSpec, coords: &[Coord], n: usize, seed: u64) -> Vec<Vec<f64>> {
    let lift = Fun::trace(spec.n_x, &spec.lift);
    let comb = Fun {
        g: spec.r1.combine.clone(),
        outs: spec.r1.outs.clone(),
    };
    let rf = Fun::trace(spec.r1.width, &spec.reference);
    let mut rng = Rng(seed);
    let mut pts = vec![];
    while pts.len() < n {
        let k = 1 + rng.below(8);
        let mut st = spec.r1.identity.clone();
        for _ in 0..k {
            let e = lift.call(&(spec.sample)(&mut rng));
            let mut v = st.clone();
            v.extend(e);
            st = comb.call(&v);
        }
        let a = rf.call(&st);
        let b: Option<Vec<f64>> = a
            .iter()
            .zip(coords)
            .map(|(&a, c)| match c {
                Coord::Add => Some(a),
                Coord::Log => (a > 0.0).then(|| a.ln()),
            })
            .collect();
        let Some(b) = b else { continue };
        let mut x = (spec.sample)(&mut rng);
        x.extend(b);
        if x.iter().all(|v| v.is_finite()) {
            pts.push(x);
        }
    }
    pts
}

fn try_coords(spec: &OnlineSpec, coords: &[Coord]) -> Result<OnlineLift, OnlineFail> {
    let (nx, p) = (spec.n_x, coords.len());
    let gb = spec.g.clone();
    let cs = coords.to_vec();
    // G(x, b) = g(x, φ⁻¹(b))
    let (mut graph, gouts) = trace(|| {
        let x: Vec<ExprId> = (0..nx).map(|i| ExprId::var(i as u16)).collect();
        let a: Vec<ExprId> = (0..p)
            .map(|i| {
                let b = ExprId::var((nx + i) as u16);
                match cs[i] {
                    Coord::Add => b,
                    Coord::Log => eexp(b),
                }
            })
            .collect();
        let mut v = x;
        v.extend(a);
        gb(&v)
    });
    let pts = sample_points(spec, coords, 64, 0x5eed ^ (p as u64 * 31));
    let evals =
        |g: &ExprGraph, e: ExprId| -> Vec<f64> { pts.iter().map(|x| g.eval(e, x)).collect() };
    let mut basis: Vec<ExprId> = vec![];
    let mut cols: Vec<Vec<f64>> = vec![]; // per basis: values at points
                                          // express `vals` in the basis; None when outside the span
    let in_span = |cols: &Vec<Vec<f64>>, vals: &[f64]| -> Option<Vec<f64>> {
        let yn = vals.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !vals.iter().all(|v| v.is_finite()) {
            return None;
        }
        if yn == 0.0 {
            return Some(vec![0.0; cols.len()]);
        }
        if cols.is_empty() {
            return None;
        }
        let m: Vec<Vec<f64>> = (0..vals.len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        let (c, res) = lstsq(&m, vals);
        (res <= 1e-9 * yn).then(|| c.into_iter().map(snap).collect())
    };
    let mut readout: Mat = vec![];
    for &o in &gouts {
        let v = evals(&graph, o);
        match in_span(&cols, &v) {
            Some(c) => readout.push(c),
            None => {
                let mut row = vec![0.0; basis.len() + 1];
                row[basis.len()] = 1.0;
                readout.push(row);
                basis.push(o);
                cols.push(v);
            }
        }
    }
    // derivative closure
    let mut gens: Vec<Mat> = vec![vec![]; p];
    let mut symbolic = true;
    let mut j = 0;
    while j < basis.len() {
        for i in 0..p {
            let d = graph.diff(basis[j], (nx + i) as u16);
            let d = graph.simplify(d);
            let v = evals(&graph, d);
            let row = match in_span(&cols, &v) {
                Some(c) => {
                    if graph.node(d) != Node::Lit(0f64.to_bits()) && !c.iter().all(|x| *x == 0.0) {
                        // structural confirmation: d − Σ cₗ Dₗ simplifies to 0?
                        let mut acc = d;
                        for (l, &cl) in c.iter().enumerate() {
                            if cl != 0.0 {
                                let k = graph.lit(-cl);
                                let t = graph.mul(k, basis[l]);
                                acc = graph.add(acc, t);
                            }
                        }
                        let s = graph.simplify(acc);
                        if graph.node(s) != Node::Lit(0f64.to_bits()) {
                            symbolic = false;
                        }
                    }
                    c
                }
                None => {
                    if basis.len() >= CAP {
                        return Err(OnlineFail::NotClosed { cap: CAP });
                    }
                    basis.push(d);
                    cols.push(v);
                    let mut c = vec![0.0; basis.len()];
                    c[basis.len() - 1] = 1.0;
                    c
                }
            };
            gens[i].push(row);
        }
        j += 1;
    }
    let r = basis.len();
    for gm in gens.iter_mut() {
        for row in gm.iter_mut() {
            row.resize(r, 0.0);
        }
    }
    for row in readout.iter_mut() {
        row.resize(r, 0.0);
    }
    // the generators of a translation action commute
    for a in 0..p {
        for b in a + 1..p {
            let ab = matmul(&gens[a], &gens[b]);
            let ba = matmul(&gens[b], &gens[a]);
            if mnorm(&madd(&ab, &ba, -1.0)) > 1e-6 * (1.0 + mnorm(&ab)) {
                return Err(OnlineFail::NotClosed { cap: CAP });
            }
        }
    }
    // joint spectral blocks
    let mut blocks: Vec<(Vec<f64>, Mat)> = vec![(vec![], eye(r))];
    for gm in &gens {
        let spec_i = real_spectrum(gm).ok_or(OnlineFail::Complex)?;
        let projs: Vec<Mat> = (0..spec_i.len())
            .map(|w| projector(gm, &spec_i, w))
            .collect();
        let mut next = vec![];
        for (lams, q) in &blocks {
            for (w, pw) in projs.iter().enumerate() {
                let qq = matmul(q, pw);
                if mnorm(&qq) > 1e-8 {
                    let mut l = lams.clone();
                    l.push(spec_i[w].0);
                    next.push((
                        l,
                        qq.iter()
                            .map(|r| r.iter().map(|x| snap(*x)).collect())
                            .collect(),
                    ));
                }
            }
        }
        blocks = next;
    }
    let separable = rank_one(spec, coords, &pts);
    let basis_str = basis.iter().map(|&b| graph.fmt_expr(b)).collect();
    let mut lift = OnlineLift {
        coords: coords.to_vec(),
        dim: r,
        gens,
        blocks,
        readout,
        monoid: Monoid::sum(),
        element: build(|_| vec![]),
        basis: basis_str,
        symbolic,
        separable,
    };
    emit_online(spec, &mut lift, &graph, &basis);
    Ok(lift)
}

/// `g(x,a)·g(x',a') == g(x,a')·g(x',a)` at sampled pairs (first output).
fn rank_one(spec: &OnlineSpec, coords: &[Coord], pts: &[Vec<f64>]) -> bool {
    let f = Fun::trace(spec.n_x + coords.len(), &spec.g);
    let nx = spec.n_x;
    let cv = |x: &[f64]| -> Vec<f64> {
        let mut v = x[..nx].to_vec();
        v.extend(x[nx..].iter().zip(coords).map(|(&b, c)| match c {
            Coord::Add => b,
            Coord::Log => b.exp(),
        }));
        v
    };
    pts.windows(2).all(|w| {
        let (x1, a1) = (&w[0][..nx], &w[0][nx..]);
        let (x2, a2) = (&w[1][..nx], &w[1][nx..]);
        let mk = |x: &[f64], a: &[f64]| {
            let mut v = x.to_vec();
            v.extend_from_slice(a);
            f.call(&cv(&v))[0]
        };
        let l = mk(x1, a1) * mk(x2, a2);
        let rr = mk(x1, a2) * mk(x2, a1);
        (l - rr).abs() <= 1e-9 * (l.abs() + rr.abs() + 1e-300)
    })
}

/// `C(δ)·S` symbolically: `Σ_blocks e^{λ·δ} Q (Σ_k Mᵏ/k!) S`,
/// `M = Σᵢ δᵢ Nᵢ`, `Nᵢ = (Aᵢ − λᵢ)Q`.
fn shift(lift: &OnlineLift, delta: &[ExprId], s: &[ExprId]) -> Vec<ExprId> {
    let r = lift.dim;
    let mut out: Vec<Option<ExprId>> = vec![None; r];
    let addo = |o: &mut Option<ExprId>, v: ExprId| {
        *o = Some(match *o {
            None => v,
            Some(a) => a + v,
        })
    };
    for (lams, q) in &lift.blocks {
        let ns: Vec<Mat> = lift
            .gens
            .iter()
            .zip(lams)
            .map(|(a, &l)| {
                let n = matmul(&madd(a, &eye(r), -l), q);
                n.iter()
                    .map(|row| row.iter().map(|x| snap(*x)).collect())
                    .collect()
            })
            .collect();
        // v = Q·S, then accumulate Σ_k M^k v / k!
        let matvec_lit = |m: &Mat, v: &[Option<ExprId>]| -> Vec<Option<ExprId>> {
            (0..r)
                .map(|i| {
                    let mut acc: Option<ExprId> = None;
                    for (j, vj) in v.iter().enumerate() {
                        if let (Some(vj), c) = (vj, m[i][j]) {
                            if c != 0.0 {
                                let t = if c == 1.0 { *vj } else { lit(c) * *vj };
                                acc = Some(match acc {
                                    None => t,
                                    Some(a) => a + t,
                                });
                            }
                        }
                    }
                    acc
                })
                .collect()
        };
        let sv: Vec<Option<ExprId>> = s.iter().map(|&x| Some(x)).collect();
        let mut term = matvec_lit(q, &sv);
        let mut total = term.clone();
        for k in 1..r {
            // term ← M·term / k
            let mut next: Vec<Option<ExprId>> = vec![None; r];
            for (i, n) in ns.iter().enumerate() {
                let nv = matvec_lit(n, &term);
                for (t, x) in next.iter_mut().zip(nv) {
                    if let Some(x) = x {
                        addo(t, delta[i] * x);
                    }
                }
            }
            if next.iter().all(Option::is_none) {
                break;
            }
            let inv = 1.0 / k as f64;
            term = next.into_iter().map(|t| t.map(|t| lit(inv) * t)).collect();
            for (a, t) in total.iter_mut().zip(&term) {
                if let Some(t) = t {
                    addo(a, *t);
                }
            }
        }
        // e^{λ·δ}
        let mut ex: Option<ExprId> = None;
        for (&l, &d) in lams.iter().zip(delta) {
            if l != 0.0 {
                let t = if l == 1.0 { d } else { lit(l) * d };
                ex = Some(match ex {
                    None => t,
                    Some(a) => a + t,
                });
            }
        }
        let scale = ex.map(eexp);
        for (o, t) in out.iter_mut().zip(total) {
            if let Some(t) = t {
                addo(o, scale.map_or(t, |s| s * t));
            }
        }
    }
    out.into_iter()
        .map(|o| o.unwrap_or_else(|| lit(0.0)))
        .collect()
}

fn emit_online(spec: &OnlineSpec, lift: &mut OnlineLift, graph: &ExprGraph, basis: &[ExprId]) {
    let (w1, r, nx) = (spec.r1.width, lift.dim, spec.n_x);
    let coords = lift.coords.clone();
    let refb = spec.reference.clone();
    let phi = move |state: &[ExprId]| -> Vec<ExprId> {
        refb(state)
            .into_iter()
            .zip(&coords)
            .map(|(a, c)| match c {
                Coord::Add => a,
                Coord::Log => eln(a),
            })
            .collect()
    };
    let r1 = spec.r1.clone();
    let occ = spec.occupied.clone();
    let l2 = lift.clone();
    let phi2 = Rc::new(phi);
    let phi_m = phi2.clone();
    let mut id = spec.r1.identity.clone();
    id.extend(std::iter::repeat_n(0.0, r));
    lift.monoid = Monoid::build(w1 + r, id, move |a, e| {
        let (s1, x1) = a.split_at(w1);
        let (s2, x2) = e.split_at(w1);
        let mut v = s1.to_vec();
        v.extend_from_slice(s2);
        let s3 = inline(&r1.combine, &r1.outs, &v);
        // an empty side takes the merged state as its reference: δ = 0,
        // nothing NaN (0/0 means) reaches the combine or its derivative
        let (o1, o2) = (occ(s1)[0], occ(s2)[0]);
        let safe = |o: ExprId, s: &[ExprId]| -> Vec<ExprId> {
            s.iter().zip(&s3).map(|(&x, &y)| sel(o, x, y)).collect()
        };
        let (b1, b2, b3) = (phi_m(&safe(o1, s1)), phi_m(&safe(o2, s2)), phi_m(&s3));
        let d1: Vec<ExprId> = b3.iter().zip(&b1).map(|(&x, &y)| x - y).collect();
        let d2: Vec<ExprId> = b3.iter().zip(&b2).map(|(&x, &y)| x - y).collect();
        let t1 = shift(&l2, &d1, x1);
        let t2 = shift(&l2, &d2, x2);
        let z = lit(0.0);
        let mut out = s3;
        out.extend(
            t1.into_iter()
                .zip(t2)
                .map(|(p, q)| sel(o1, p, z) + sel(o2, q, z)),
        );
        out
    });
    if let Some(blocks) = r1_blocks(&spec.r1) {
        let nref = lift.coords.len();
        let coords = lift.coords.clone();
        let refb = spec.reference.clone();
        let phi_b = build(move |st| {
            refb(st)
                .into_iter()
                .zip(&coords)
                .map(|(a, c)| match c {
                    Coord::Add => a,
                    Coord::Log => eln(a),
                })
                .collect()
        });
        let _ = nref;
        lift.monoid.saved = Some(std::sync::Arc::new(SavedBwd {
            n_x: nx,
            w1,
            r,
            blocks,
            lift: Fun::trace(nx, &spec.lift),
            phi: Fun::trace(w1, &phi_b),
            basis: Fun {
                g: graph.clone(),
                outs: basis.to_vec(),
            },
            gens: lift.gens.clone(),
            element: Fun {
                g: ExprGraph::new(),
                outs: vec![],
            },
        }));
    }
    let lb = spec.lift.clone();
    let g = graph.clone();
    let bs = basis.to_vec();
    lift.element = build(move |x| {
        let st = lb(&x[..nx]);
        let b = phi2(&st);
        let mut v = x[..nx].to_vec();
        v.extend(b);
        let mut out = st;
        out.extend(inline(&g, &bs, &v));
        out
    });
    if let Some(sv) = lift.monoid.saved.take() {
        if let Ok(mut sv) = std::sync::Arc::try_unwrap(sv) {
            sv.element = Fun::trace(nx, &lift.element);
            lift.monoid.saved = Some(std::sync::Arc::new(sv));
        }
    }
}

/// Max relative error of the one-pass monoid (random combine trees,
/// identities mixed in) against the two-pass definition.
pub fn verify_online(spec: &OnlineSpec, lift: &OnlineLift, seed: u64, trials: usize) -> f64 {
    let lift_f = Fun::trace(spec.n_x, &spec.lift);
    let r1c = Fun {
        g: spec.r1.combine.clone(),
        outs: spec.r1.outs.clone(),
    };
    let rf = Fun::trace(spec.r1.width, &spec.reference);
    let gf = Fun::trace(spec.n_x + spec.n_ref(), &spec.g);
    let el = Fun::trace(spec.n_x, &lift.element);
    let mc = Fun {
        g: lift.monoid.combine.clone(),
        outs: lift.monoid.outs.clone(),
    };
    let w = lift.monoid.width;
    let w1 = spec.r1.width;
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    for _ in 0..trials {
        let n = 1 + rng.below(40);
        let xs: Vec<Vec<f64>> = (0..n).map(|_| (spec.sample)(&mut rng)).collect();
        // two-pass
        let mut st = spec.r1.identity.clone();
        for x in &xs {
            let mut v = st.clone();
            v.extend(lift_f.call(x));
            st = r1c.call(&v);
        }
        let a = rf.call(&st);
        let mut want = vec![0.0; lift.readout.len()];
        for x in &xs {
            let mut v = x.clone();
            v.extend_from_slice(&a);
            for (wv, gv) in want.iter_mut().zip(gf.call(&v)) {
                *wv += gv;
            }
        }
        // one pass, random tree
        let leaves: Vec<Vec<f64>> = xs.iter().map(|x| el.call(x)).collect();
        fn tree(l: &[Vec<f64>], rng: &mut Rng, mc: &Fun, id: &[f64]) -> Vec<f64> {
            let with_id = |v: Vec<f64>, rng: &mut Rng| {
                if rng.below(5) == 0 {
                    let mut a = id.to_vec();
                    a.extend(v);
                    mc.call(&a)
                } else {
                    v
                }
            };
            if l.len() == 1 {
                return with_id(l[0].clone(), rng);
            }
            let k = 1 + rng.below(l.len() - 1);
            let a = tree(&l[..k], rng, mc, id);
            let b = tree(&l[k..], rng, mc, id);
            let mut v = a;
            v.extend(b);
            with_id(mc.call(&v), rng)
        }
        let s = tree(&leaves, &mut rng, &mc, &lift.monoid.identity);
        debug_assert_eq!(s.len(), w);
        for (c, row) in lift.readout.iter().enumerate() {
            let got: f64 = row.iter().zip(&s[w1..]).map(|(b, x)| b * x).sum();
            // backward-error scale: Σ|summands|
            let mut scale = 1e-300;
            for x in &xs {
                let mut v = x.clone();
                v.extend_from_slice(&a);
                scale += gf.call(&v)[c].abs();
            }
            let e = (got - want[c]).abs() / scale;
            worst = worst.max(if e.is_nan() { f64::INFINITY } else { e });
        }
    }
    worst
}

/// The two-pass graph and the one-pass graph over inputs `xᵢ [rows, n]`
/// (one per element scalar), outputs `[rows, k]`. Evaluated by the
/// [`TensorGraph`] interpreter in the tests.
pub fn online_graphs(
    spec: &OnlineSpec,
    lift: &OnlineLift,
    rows: usize,
    n: usize,
) -> (TensorGraph, TensorId, TensorGraph, TensorId) {
    let k = lift.readout.len();
    let (nx, w1, p) = (spec.n_x, spec.r1.width, spec.n_ref());
    let pick = View::pick(2, &[Some(0), Some(1)]);
    let slot = |rank: usize, j: usize| View {
        forms: vec![Form::unit(rank, 0), Form::constant(rank, j as i64)],
        guard: false,
        gather: None,
    };
    let sumk = Monoid::build(k, vec![0.0; k], |a, e| {
        a.iter().zip(e).map(|(&x, &y)| x + y).collect()
    });
    // two-pass
    let mut g2 = TensorGraph::new();
    let xs: Vec<TensorId> = (0..nx).map(|_| g2.input(&[rows, n])).collect();
    let args = |xs: &[TensorId]| -> Vec<Arg> {
        xs.iter()
            .map(|&src| Arg {
                src,
                view: pick.clone(),
            })
            .collect()
    };
    let lf = spec.lift.clone();
    let p1 = g2
        .comp_tuple(
            &[rows, w1],
            &[n],
            args(&xs),
            Some(spec.r1.clone()),
            move |v| lf(v),
        )
        .unwrap();
    let mut a2 = args(&xs);
    a2.extend((0..w1).map(|j| Arg {
        src: p1,
        view: slot(2, j),
    }));
    let (rf, gf) = (spec.reference.clone(), spec.g.clone());
    let two = g2
        .comp_tuple(&[rows, k], &[n], a2, Some(sumk.clone()), move |v| {
            let a = rf(&v[nx..]);
            debug_assert_eq!(a.len(), p);
            let mut u = v[..nx].to_vec();
            u.extend(a);
            gf(&u)
        })
        .unwrap();
    // one-pass
    let mut g1 = TensorGraph::new();
    let xs1: Vec<TensorId> = (0..nx).map(|_| g1.input(&[rows, n])).collect();
    let ef = lift.element.clone();
    let st = g1
        .comp_tuple(
            &[rows, lift.monoid.width],
            &[n],
            args(&xs1),
            Some(lift.monoid.clone()),
            move |v| ef(v),
        )
        .unwrap();
    let ro = lift.readout.clone();
    let r = lift.dim;
    let one = g1
        .comp_tuple(
            &[rows, k],
            &[] as &[usize],
            (0..r)
                .map(|j| Arg {
                    src: st,
                    view: slot(1, w1 + j),
                })
                .collect(),
            None,
            move |v| {
                ro.iter()
                    .map(|row| {
                        let mut acc = lit(0.0);
                        for (b, &x) in row.iter().zip(v) {
                            if *b != 0.0 {
                                acc = acc + lit(*b) * x;
                            }
                        }
                        acc
                    })
                    .collect()
            },
        )
        .unwrap();
    (g2, two, g1, one)
}

// ---------------------------------------------------------------------------
// Scan lifts
// ---------------------------------------------------------------------------

/// A recurrence `h' = f(h, x)`, `h ∈ ℝᵈ`.
#[derive(Clone)]
pub struct ScanSpec {
    pub name: String,
    pub d: usize,
    pub n_x: usize,
    /// `(h…, x…)` → `h'`.
    pub step: Build,
    pub sample_x: Sampler,
    /// Sampler of states in the recurrence's domain.
    pub sample_h: Sampler,
    /// `Some((n, m))`: `h` is an `n×m` matrix (row-major), for the
    /// chunked lift.
    pub mat: Option<(usize, usize)>,
}

/// Elementwise coordinate change `u = φ(h)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conj {
    Id,
    Exp,
    Ln,
    Recip,
}

impl Conj {
    fn fwd(self, h: ExprId) -> ExprId {
        match self {
            Conj::Id => h,
            Conj::Exp => eexp(h),
            Conj::Ln => eln(h),
            Conj::Recip => lit(1.0) / h,
        }
    }
    fn inv(self, u: ExprId) -> ExprId {
        match self {
            Conj::Id => u,
            Conj::Exp => eln(u),
            Conj::Ln => eexp(u),
            Conj::Recip => lit(1.0) / u,
        }
    }
    fn fwd_f(self, h: f64) -> f64 {
        match self {
            Conj::Id => h,
            Conj::Exp => h.exp(),
            Conj::Ln => h.ln(),
            Conj::Recip => 1.0 / h,
        }
    }
}

/// Structure of the linear part `A` of an affine step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AStruct {
    Identity,
    Scalar,
    Diagonal,
    /// `α(I − β·k·kᵀ)` (rank-1 symmetric update of a scaled identity).
    ScaledDelta,
    General,
}

#[derive(Clone)]
pub enum ScanLift {
    /// `u' = A(x)·u + b(x)` in coordinates `u = φ(h)`.
    Affine {
        conj: Conj,
        a: AStruct,
        /// The Jacobian's independence of `h` was shown structurally.
        symbolic: bool,
        monoid: Monoid,
        element: Build,
    },
    /// `h'ᵢ = maxⱼ(hⱼ + aᵢⱼ(x)) ∨ bᵢ(x)`.
    Tropical,
    /// `h' = (αh + β)/(γh + δ)`.
    Mobius { monoid: Monoid, element: Build },
    /// Affine in observables `ψ(h)` (numeric certificate).
    Observable { obs: Vec<String> },
}

impl ScanLift {
    pub fn family(&self) -> String {
        match self {
            ScanLift::Affine { conj, a, .. } => match conj {
                Conj::Id => format!("affine, A {a:?}"),
                c => format!("affine in {c:?}(h), A {a:?}"),
            },
            ScanLift::Tropical => "max-plus affine".into(),
            ScanLift::Mobius { .. } => "Möbius (2×2 projective)".into(),
            ScanLift::Observable { obs } => format!("affine in observables ({})", obs.join(", ")),
        }
    }
}

/// Why a family was rejected.
#[derive(Clone, Debug)]
pub struct Witness {
    pub family: String,
    pub why: String,
}

/// Search for an associative representation of `spec`'s step. On
/// failure, one witness per family tried.
pub fn scan_lift(spec: &ScanSpec) -> Result<(ScanLift, f64), Vec<Witness>> {
    let mut wit = vec![];
    for conj in [Conj::Id, Conj::Exp, Conj::Ln, Conj::Recip] {
        match affine(spec, conj) {
            Ok(l) => {
                let e = verify_scan(spec, &l, 3, 40);
                if e <= 1e-8 {
                    return Ok((l, e));
                }
                wit.push(Witness {
                    family: format!("affine in {conj:?}"),
                    why: format!("structurally affine but fuzz err {e:.1e}"),
                });
            }
            Err(w) => wit.push(w),
        }
    }
    match tropical(spec) {
        Ok(()) => return Ok((ScanLift::Tropical, 0.0)),
        Err(w) => wit.push(w),
    }
    if spec.d == 1 {
        match mobius(spec) {
            Ok(l) => {
                let e = verify_scan(spec, &l, 5, 40);
                if e <= 1e-7 {
                    return Ok((l, e));
                }
                wit.push(Witness {
                    family: "Möbius".into(),
                    why: format!("Schwarzian ≈ 0 but fuzz err {e:.1e}"),
                });
            }
            Err(w) => wit.push(w),
        }
    }
    if spec.d <= 2 {
        match observable(spec) {
            Ok(obs) => return Ok((ScanLift::Observable { obs }, 0.0)),
            Err(w) => wit.push(w),
        }
    }
    Err(wit)
}

/// Step in coordinates `u = φ(h)`: `u' = φ(f(φ⁻¹(u), x))`.
fn conj_step(spec: &ScanSpec, conj: Conj) -> (ExprGraph, Vec<ExprId>) {
    let (d, nx) = (spec.d, spec.n_x);
    let f = spec.step.clone();
    trace(|| {
        let mut v: Vec<ExprId> = (0..d).map(|i| conj.inv(ExprId::var(i as u16))).collect();
        v.extend((0..nx).map(|i| ExprId::var((d + i) as u16)));
        f(&v).into_iter().map(|y| conj.fwd(y)).collect()
    })
}

fn sample_u(spec: &ScanSpec, conj: Conj, rng: &mut Rng) -> Option<Vec<f64>> {
    let mut v: Vec<f64> = (spec.sample_h)(rng)
        .into_iter()
        .map(|h| conj.fwd_f(h))
        .collect();
    v.extend((spec.sample_x)(rng));
    v.iter().all(|x| x.is_finite()).then_some(v)
}

fn affine(spec: &ScanSpec, conj: Conj) -> Result<ScanLift, Witness> {
    let d = spec.d;
    let fam = format!("affine in {conj:?}(h)");
    let (mut g, outs) = conj_step(spec, conj);
    let hmask: u64 = if d >= 64 { u64::MAX } else { (1u64 << d) - 1 };
    let mut jac = vec![vec![ExprId::ZERO; d]; d];
    let mut structural = true;
    for i in 0..d {
        for j in 0..d {
            let dj = g.diff(outs[i], j as u16);
            jac[i][j] = g.simplify(dj);
            if g.deps(jac[i][j]) & hmask != 0 {
                structural = false;
            }
        }
    }
    let mut rng = Rng(0xaff1);
    if !structural {
        // numeric: J must not change with h at fixed x
        for _ in 0..12 {
            let Some(p) = sample_u(spec, conj, &mut rng) else {
                continue;
            };
            let Some(mut q) = sample_u(spec, conj, &mut rng) else {
                continue;
            };
            q[d..].copy_from_slice(&p[d..]);
            for i in 0..d {
                for j in 0..d {
                    let (a, b) = (g.eval(jac[i][j], &p), g.eval(jac[i][j], &q));
                    if (a - b).abs() > 1e-9 * (a.abs() + b.abs() + 1e-9) {
                        return Err(Witness {
                            family: fam,
                            why: format!(
                                "∂f{i}/∂h{j} varies with h at fixed x: {a:.4} vs {b:.4} (h = {:?} vs {:?})",
                                &p[..d.min(3)],
                                &q[..d.min(3)]
                            ),
                        });
                    }
                }
            }
        }
    }
    // structure of A
    let mut astruct = AStruct::Identity;
    let rank = |s: AStruct| match s {
        AStruct::Identity => 0,
        AStruct::Scalar => 1,
        AStruct::Diagonal => 2,
        AStruct::ScaledDelta => 3,
        AStruct::General => 4,
    };
    let structure_of = |a: &Mat| -> AStruct {
        let n = a.len();
        let tol = 1e-10;
        let offdiag = (0..n).all(|i| (0..n).all(|j| i == j || a[i][j].abs() < tol));
        if offdiag {
            if (0..n).all(|i| (a[i][i] - 1.0).abs() < tol) {
                return AStruct::Identity;
            }
            if (0..n).all(|i| (a[i][i] - a[0][0]).abs() < tol) {
                return AStruct::Scalar;
            }
            return AStruct::Diagonal;
        }
        let sym = (0..n).all(|i| (0..n).all(|j| (a[i][j] - a[j][i]).abs() < tol));
        if sym && n >= 2 {
            if let Some(sp) = real_spectrum(a) {
                if let Some(&(alpha, _)) = sp.iter().find(|s| s.1 >= n - 1) {
                    let m = madd(&eye(n), a, -1.0 / alpha.max(1e-300));
                    // m = β k kᵀ, rank ≤ 1: all 2×2 minors vanish
                    let ok = (0..n).all(|i| {
                        (0..n).all(|j| (m[i][i] * m[j][j] - m[i][j] * m[j][i]).abs() < 1e-8)
                    });
                    if ok {
                        return AStruct::ScaledDelta;
                    }
                }
            }
        }
        AStruct::General
    };
    let mat = spec.mat;
    for _ in 0..8 {
        let Some(p) = sample_u(spec, conj, &mut rng) else {
            continue;
        };
        let j: Mat = (0..d)
            .map(|i| (0..d).map(|k| g.eval(jac[i][k], &p)).collect())
            .collect();
        let s = match mat {
            Some((n, m)) => {
                // J must be A ⊗ I_m
                let a: Mat = (0..n)
                    .map(|i| (0..n).map(|k| j[i * m][k * m]).collect())
                    .collect();
                let kron = (0..d).all(|r| {
                    (0..d).all(|c| {
                        let want = if r % m == c % m { a[r / m][c / m] } else { 0.0 };
                        (j[r][c] - want).abs() < 1e-10
                    })
                });
                if kron {
                    structure_of(&a)
                } else {
                    AStruct::General
                }
            }
            None => structure_of(&j),
        };
        if rank(s) > rank(astruct) {
            astruct = s;
        }
    }
    // emit: element = (A, b), b = f(0, x) in u coordinates; for a
    // diagonal A the state is (diag, b)
    let diag = matches!(
        astruct,
        AStruct::Identity | AStruct::Scalar | AStruct::Diagonal
    );
    let width = if diag { 2 * d } else { d * d + d };
    let mut id = vec![0.0; width];
    if diag {
        id[..d].iter_mut().for_each(|x| *x = 1.0);
    } else {
        for i in 0..d {
            id[i * d + i] = 1.0;
        }
    }
    // combine(a = earlier, e = later) = e ∘ a
    let monoid = Monoid::build(width, id, move |a, e| {
        if diag {
            let (aa, ab) = a.split_at(d);
            let (ea, eb) = e.split_at(d);
            let mut out: Vec<ExprId> = (0..d).map(|i| ea[i] * aa[i]).collect();
            out.extend((0..d).map(|i| ea[i] * ab[i] + eb[i]));
            out
        } else {
            let (aa, ab) = a.split_at(d * d);
            let (ea, eb) = e.split_at(d * d);
            let mut out = vec![];
            for i in 0..d {
                for k in 0..d {
                    let mut s = ea[i * d] * aa[k];
                    for l in 1..d {
                        s = s + ea[i * d + l] * aa[l * d + k];
                    }
                    out.push(s);
                }
            }
            for i in 0..d {
                let mut s = eb[i];
                for l in 0..d {
                    s = s + ea[i * d + l] * ab[l];
                }
                out.push(s);
            }
            out
        }
    });
    let jg = g.clone();
    let jouts: Vec<ExprId> = if diag {
        (0..d).map(|i| jac[i][i]).collect()
    } else {
        jac.iter().flatten().copied().collect()
    };
    let f_outs = outs.clone();
    let element = build(move |x| {
        // A at u_ref (A is u-independent), b = f(u_ref) − A·u_ref with
        // u_ref = φ(1) inside every coordinate's domain
        let ur = conj.fwd_f(1.0);
        let mut v: Vec<ExprId> = (0..d).map(|_| lit(ur)).collect();
        v.extend_from_slice(x);
        let a = inline(&jg, &jouts, &v);
        let fy = inline(&jg, &f_outs, &v);
        let b: Vec<ExprId> = (0..d)
            .map(|i| {
                let mut s = fy[i];
                if diag {
                    s = s - a[i] * lit(ur);
                } else {
                    for l in 0..d {
                        s = s - a[i * d + l] * lit(ur);
                    }
                }
                s
            })
            .collect();
        let mut out = a;
        out.extend(b);
        out
    });
    Ok(ScanLift::Affine {
        conj,
        a: astruct,
        symbolic: structural,
        monoid,
        element,
    })
}

fn tropical(spec: &ScanSpec) -> Result<(), Witness> {
    let d = spec.d;
    let fam = "max-plus affine".to_string();
    let f = Fun::trace(spec.d + spec.n_x, &spec.step);
    let mut rng = Rng(0x7a0f);
    let big = 40.0;
    for _ in 0..16 {
        let x = (spec.sample_x)(&mut rng);
        let call = |h: &[f64]| {
            let mut v = h.to_vec();
            v.extend_from_slice(&x);
            f.call(&v)
        };
        let b = call(&vec![f64::NEG_INFINITY; d]);
        if b.iter().any(|v| v.is_nan()) {
            return Err(Witness {
                family: fam,
                why: "f(−∞, x) undefined".into(),
            });
        }
        let mut a = vec![vec![0.0; d]; d];
        for j in 0..d {
            for (t, big) in [big, 2.0 * big].into_iter().enumerate() {
                let mut h = vec![f64::NEG_INFINITY; d];
                h[j] = big;
                let y = call(&h);
                for i in 0..d {
                    let aij = y[i] - big;
                    if !aij.is_finite()
                        || (t == 1 && (aij - a[i][j]).abs() > 1e-9 * (1.0 + aij.abs()))
                    {
                        return Err(Witness {
                            family: fam,
                            why: format!(
                                "slope of f{i} in h{j} is not 1 far out (no max-plus form)"
                            ),
                        });
                    }
                    a[i][j] = aij;
                }
            }
        }
        for _ in 0..4 {
            let h = (spec.sample_h)(&mut rng);
            let y = call(&h);
            for i in 0..d {
                let mut w = b[i];
                for j in 0..d {
                    w = w.max(h[j] + a[i][j]);
                }
                if !((y[i] - w).abs() <= 1e-9 * (1.0 + w.abs())) {
                    return Err(Witness {
                        family: fam,
                        why: format!(
                            "f(h)={:.4} ≠ max-plus fit {w:.4} at h={:?}",
                            y[i],
                            &h[..d.min(3)]
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

fn mobius(spec: &ScanSpec) -> Result<ScanLift, Witness> {
    let fam = "Möbius".to_string();
    let (mut g, outs) = conj_step(spec, Conj::Id);
    let f1 = g.diff(outs[0], 0);
    let f1 = g.simplify(f1);
    let f2 = g.diff(f1, 0);
    let f2 = g.simplify(f2);
    let f3 = g.diff(f2, 0);
    let f3 = g.simplify(f3);
    let mut rng = Rng(0x30b1);
    for _ in 0..12 {
        let Some(p) = sample_u(spec, Conj::Id, &mut rng) else {
            continue;
        };
        let (a, b, c) = (g.eval(f1, &p), g.eval(f2, &p), g.eval(f3, &p));
        let s = c / a - 1.5 * (b / a) * (b / a);
        if !(s.abs() <= 1e-7 * (1.0 + (c / a).abs())) {
            return Err(Witness {
                family: fam,
                why: format!("Schwarzian {s:.3e} ≠ 0 at h={:.4}", p[0]),
            });
        }
    }
    // (α, β, γ, 1) from f(0), f(1), f(−1)
    let f = spec.step.clone();
    let element = build(move |x| {
        let at = |h: f64| {
            let mut v = vec![lit(h)];
            v.extend_from_slice(x);
            f(&v)[0]
        };
        let (f0, f1, fm) = (at(0.0), at(1.0), at(-1.0));
        let beta = f0;
        let gamma = (lit(2.0) * beta - f1 - fm) / (f1 - fm);
        let alpha = f1 * (gamma + lit(1.0)) - beta;
        vec![alpha, beta, gamma, lit(1.0)]
    });
    // [[α, β], [γ, δ]] row-major; combine(a, e) = e·a, normalized by
    // the max |entry| (projective, keeps the scan in range)
    let monoid = Monoid::build(4, vec![1.0, 0.0, 0.0, 1.0], |a, e| {
        let m = [
            e[0] * a[0] + e[1] * a[2],
            e[0] * a[1] + e[1] * a[3],
            e[2] * a[0] + e[3] * a[2],
            e[2] * a[1] + e[3] * a[3],
        ];
        m.to_vec()
    });
    let _ = g;
    Ok(ScanLift::Mobius { monoid, element })
}

/// Observable library over `h` (`d ≤ 2`): `(name, ψ)`.
fn library(d: usize) -> Vec<(String, Rc<dyn Fn(&[f64]) -> f64>)> {
    let mut l: Vec<(String, Rc<dyn Fn(&[f64]) -> f64>)> = vec![];
    for i in 0..d {
        l.push((format!("h{i}"), Rc::new(move |h: &[f64]| h[i])));
        l.push((format!("h{i}²"), Rc::new(move |h: &[f64]| h[i] * h[i])));
        l.push((format!("1/h{i}"), Rc::new(move |h: &[f64]| 1.0 / h[i])));
        l.push((format!("exp h{i}"), Rc::new(move |h: &[f64]| h[i].exp())));
        l.push((format!("ln h{i}"), Rc::new(move |h: &[f64]| h[i].ln())));
    }
    if d == 2 {
        l.push(("h0·h1".into(), Rc::new(|h: &[f64]| h[0] * h[1])));
        l.push(("h0/h1".into(), Rc::new(|h: &[f64]| h[0] / h[1])));
        l.push(("h1/h0".into(), Rc::new(|h: &[f64]| h[1] / h[0])));
        l.push(("h0²·h1".into(), Rc::new(|h: &[f64]| h[0] * h[0] * h[1])));
        l.push(("h0·h1²".into(), Rc::new(|h: &[f64]| h[0] * h[1] * h[1])));
    }
    l
}

/// `h` recoverable from the chosen observables (by name).
fn recoverable(d: usize, names: &[&str]) -> bool {
    let has = |s: &str| names.contains(&s);
    (0..d).all(|i| {
        let hi = format!("h{i}");
        let j = 1 - i.min(1);
        has(&hi)
            || has(&format!("exp h{i}"))
            || has(&format!("ln h{i}"))
            || has(&format!("1/h{i}"))
            || (d == 2
                && (has(&format!("h{j}")) || has(&format!("1/h{j}")))
                && (has("h0·h1") || has(&format!("h{i}/h{j}"))))
    })
}

fn observable(spec: &ScanSpec) -> Result<Vec<String>, Witness> {
    let d = spec.d;
    let lib = library(d);
    let f = Fun::trace(d + spec.n_x, &spec.step);
    let mut rng = Rng(0x0b5e);
    // fixed x samples, many h per x
    let xs: Vec<Vec<f64>> = (0..6).map(|_| (spec.sample_x)(&mut rng)).collect();
    let hs: Vec<Vec<f64>> = (0..24).map(|_| (spec.sample_h)(&mut rng)).collect();
    let n = lib.len();
    let mut subsets: Vec<Vec<usize>> = vec![];
    for a in 0..n {
        subsets.push(vec![a]);
        for b in a + 1..n {
            subsets.push(vec![a, b]);
            for c in b + 1..n {
                subsets.push(vec![a, b, c]);
            }
        }
    }
    'sub: for s in subsets {
        let names: Vec<&str> = s.iter().map(|&i| lib[i].0.as_str()).collect();
        if !recoverable(d, &names) {
            continue;
        }
        for x in &xs {
            let mut rows = vec![];
            let mut tgt: Vec<Vec<f64>> = vec![vec![]; s.len()];
            for h in &hs {
                let mut v = h.clone();
                v.extend_from_slice(x);
                let y = f.call(&v);
                let mut row: Vec<f64> = s.iter().map(|&i| (lib[i].1)(h)).collect();
                row.push(1.0);
                if !row.iter().all(|v| v.is_finite()) {
                    continue 'sub;
                }
                rows.push(row);
                for (t, &i) in tgt.iter_mut().zip(&s) {
                    t.push((lib[i].1)(&y));
                }
            }
            for t in &tgt {
                if !t.iter().all(|v| v.is_finite()) {
                    continue 'sub;
                }
                let (_, res) = lstsq(&rows, t);
                let yn = t.iter().map(|v| v * v).sum::<f64>().sqrt();
                if res > 1e-9 * (yn + 1e-12) {
                    continue 'sub;
                }
            }
        }
        // a single coordinate that is just h (or φ(h)) is the affine
        // family, already tried
        if s.len() == d
            && names
                .iter()
                .all(|n| !n.contains('·') && !n.contains('/') || n.starts_with("1/"))
        {
            continue;
        }
        return Ok(names.into_iter().map(String::from).collect());
    }
    Err(Witness {
        family: "observable lift (≤3 of the library)".into(),
        why: "no subset closes".into(),
    })
}

/// Max relative error of scanning with the lift (random association,
/// as a parallel scan would) against the sequential recurrence.
pub fn verify_scan(spec: &ScanSpec, lift: &ScanLift, seed: u64, trials: usize) -> f64 {
    let (monoid, element, conj, mobius) = match lift {
        ScanLift::Affine {
            monoid,
            element,
            conj,
            ..
        } => (monoid, element, *conj, false),
        ScanLift::Mobius { monoid, element } => (monoid, element, Conj::Id, true),
        _ => return 0.0,
    };
    let d = spec.d;
    let f = Fun::trace(d + spec.n_x, &spec.step);
    let el = Fun::trace(spec.n_x, element);
    let mc = Fun {
        g: monoid.combine.clone(),
        outs: monoid.outs.clone(),
    };
    let w = monoid.width;
    let diag = !mobius && w == 2 * d;
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    for _ in 0..trials {
        let t = 1 + rng.below(24);
        let h0 = (spec.sample_h)(&mut rng);
        let xs: Vec<Vec<f64>> = (0..t).map(|_| (spec.sample_x)(&mut rng)).collect();
        let mut h = h0.clone();
        for x in &xs {
            let mut v = h.clone();
            v.extend_from_slice(x);
            h = f.call(&v);
        }
        let leaves: Vec<Vec<f64>> = xs.iter().map(|x| el.call(x)).collect();
        fn tree(l: &[Vec<f64>], rng: &mut Rng, mc: &Fun) -> Vec<f64> {
            if l.len() == 1 {
                return l[0].clone();
            }
            let k = 1 + rng.below(l.len() - 1);
            let mut v = tree(&l[..k], rng, mc);
            v.extend(tree(&l[k..], rng, mc));
            mc.call(&v)
        }
        let s = tree(&leaves, &mut rng, &mc);
        let got: Vec<f64> = if mobius {
            let x = h0[0];
            vec![(s[0] * x + s[1]) / (s[2] * x + s[3])]
        } else {
            let u0: Vec<f64> = h0.iter().map(|&x| conj.fwd_f(x)).collect();
            let u: Vec<f64> = if diag {
                (0..d).map(|i| s[i] * u0[i] + s[d + i]).collect()
            } else {
                (0..d)
                    .map(|i| (0..d).map(|l| s[i * d + l] * u0[l]).sum::<f64>() + s[d * d + i])
                    .collect()
            };
            u.into_iter()
                .map(|u| match conj {
                    Conj::Id => u,
                    Conj::Exp => u.ln(),
                    Conj::Ln => u.exp(),
                    Conj::Recip => 1.0 / u,
                })
                .collect()
        };
        for i in 0..d {
            let e = (got[i] - h[i]).abs() / (h[i].abs() + 1e-6);
            worst = worst.max(if e.is_nan() { f64::INFINITY } else { e });
        }
        let _ = w;
    }
    worst
}

// ---------------------------------------------------------------------------
// Chunked lifts
// ---------------------------------------------------------------------------

/// The chunked form of an affine matrix-state scan
/// `S_t = A_t·S_{t−1} + β_t·k_t·v_tᵀ`, readout `o_t = S_tᵀ·q_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkForm {
    /// `A = I`: `O = tril(QKᵀ)·V + Q·S₀` (linear attention).
    Cumulative,
    /// `A = αI`: `O = (tril(QKᵀ) ⊙ Γ)·V + γ⊙Q·S₀` (SSD, RetNet).
    ScalarDecay,
    /// `A = diag(a)`: `O = tril(Q̃K̃ᵀ)·V + Q̃·S₀`, `Q̃ = Q⊙Γ`, `K̃ = K/Γ` (GLA).
    DiagDecay,
    /// `A = α(I − β·k·kᵀ)`: WY/UT, `(I + tril₋(diag β·KKᵀ))·U = diag β·(V̂ − K·Ŝ₀)` (gated delta rule).
    ScalarDecayDelta,
}

pub fn chunk_form(a: AStruct) -> Option<ChunkForm> {
    match a {
        AStruct::Identity => Some(ChunkForm::Cumulative),
        AStruct::Scalar => Some(ChunkForm::ScalarDecay),
        AStruct::Diagonal => Some(ChunkForm::DiagDecay),
        AStruct::ScaledDelta => Some(ChunkForm::ScalarDecayDelta),
        AStruct::General => None,
    }
}

/// One step of a chunk-form scan, numerically: decay per state row
/// `a` (`n`; all equal for scalar decay), `β`, `k` (`n`), `v` (`m`), `q`
/// (`n`).
#[derive(Clone, Debug)]
pub struct Step {
    pub a: Vec<f64>,
    pub beta: f64,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub q: Vec<f64>,
}

/// Sequential reference: `S ← diag(a)(I − [delta]βkkᵀ)S + βkvᵀ`, `o = Sᵀq`.
pub fn scan_sequential(steps: &[Step], n: usize, m: usize, delta: bool) -> Vec<Vec<f64>> {
    let mut s = vec![vec![0.0; m]; n];
    let mut outs = vec![];
    for st in steps {
        let kts: Vec<f64> = (0..m)
            .map(|c| (0..n).map(|i| st.k[i] * s[i][c]).sum())
            .collect();
        for i in 0..n {
            for c in 0..m {
                let mut x = s[i][c];
                if delta {
                    x -= st.beta * st.k[i] * kts[c];
                }
                s[i][c] = st.a[i] * x + st.beta * st.k[i] * st.v[c];
            }
        }
        outs.push(
            (0..m)
                .map(|c| (0..n).map(|i| s[i][c] * st.q[i]).sum())
                .collect(),
        );
    }
    outs
}

/// The chunked algorithm for `form` with chunk length `chunk`: only
/// matmuls, one triangular solve per chunk (delta), and a chunk-level
/// state recurrence.
pub fn scan_chunked(
    steps: &[Step],
    n: usize,
    m: usize,
    form: ChunkForm,
    chunk: usize,
) -> Vec<Vec<f64>> {
    let mut s0 = vec![vec![0.0; m]; n];
    let mut outs = vec![];
    for ch in steps.chunks(chunk) {
        let c = ch.len();
        // cumulative per-row decay Γ_t (n), within the chunk
        let mut gam = vec![vec![1.0; n]; c];
        for t in 0..c {
            for i in 0..n {
                gam[t][i] = ch[t].a[i] * if t > 0 { gam[t - 1][i] } else { 1.0 };
            }
        }
        match form {
            ChunkForm::Cumulative | ChunkForm::ScalarDecay | ChunkForm::DiagDecay => {
                // Ŝ_t = S_t / Γ_t (rowwise): Ŝ_t = Ŝ₀ + Σ_{s≤t} (βk/Γ_s) vᵀ
                // o_t = Ŝ_tᵀ(Γ_t ⊙ q_t)
                let qt: Vec<Vec<f64>> = (0..c)
                    .map(|t| (0..n).map(|i| ch[t].q[i] * gam[t][i]).collect())
                    .collect();
                let kt: Vec<Vec<f64>> = (0..c)
                    .map(|t| {
                        (0..n)
                            .map(|i| ch[t].beta * ch[t].k[i] / gam[t][i])
                            .collect()
                    })
                    .collect();
                // P = tril(Q̃ K̃ᵀ)
                for t in 0..c {
                    let mut o: Vec<f64> = (0..m)
                        .map(|cc| (0..n).map(|i| qt[t][i] * s0[i][cc]).sum())
                        .collect();
                    for s in 0..=t {
                        let p: f64 = (0..n).map(|i| qt[t][i] * kt[s][i]).sum();
                        for cc in 0..m {
                            o[cc] += p * ch[s].v[cc];
                        }
                    }
                    outs.push(o);
                }
                // S_C = Γ_C ⊙ (S₀ + K̃ᵀV)
                for i in 0..n {
                    for cc in 0..m {
                        let x: f64 = (0..c).map(|s| kt[s][i] * ch[s].v[cc]).sum();
                        s0[i][cc] = gam[c - 1][i] * (s0[i][cc] + x);
                    }
                }
            }
            ChunkForm::ScalarDecayDelta => {
                // scalar decay γ_t; Ŝ = S/γ: Ŝ_t = (I − βkkᵀ)Ŝ_{t−1} + (β/γ_t) k vᵀ
                let g: Vec<f64> = (0..c).map(|t| gam[t][0]).collect();
                // (I + tril₋(diag β KKᵀ)) U = diag β (V/γ − K Ŝ₀)
                let mut u = vec![vec![0.0; m]; c];
                for t in 0..c {
                    let ks0: Vec<f64> = (0..m)
                        .map(|cc| (0..n).map(|i| ch[t].k[i] * s0[i][cc]).sum())
                        .collect();
                    for cc in 0..m {
                        u[t][cc] = ch[t].beta * (ch[t].v[cc] / g[t] - ks0[cc]);
                    }
                    for r in 0..t {
                        let kk: f64 = (0..n).map(|i| ch[t].k[i] * ch[r].k[i]).sum();
                        for cc in 0..m {
                            u[t][cc] -= ch[t].beta * kk * u[r][cc];
                        }
                    }
                }
                // O = diag γ (Q Ŝ₀ + tril(QKᵀ) U)
                for t in 0..c {
                    let mut o: Vec<f64> = (0..m)
                        .map(|cc| (0..n).map(|i| ch[t].q[i] * s0[i][cc]).sum())
                        .collect();
                    for s in 0..=t {
                        let p: f64 = (0..n).map(|i| ch[t].q[i] * ch[s].k[i]).sum();
                        for cc in 0..m {
                            o[cc] += p * u[s][cc];
                        }
                    }
                    outs.push(o.into_iter().map(|x| x * g[t]).collect());
                }
                for i in 0..n {
                    for cc in 0..m {
                        let x: f64 = (0..c).map(|s| ch[s].k[i] * u[s][cc]).sum();
                        s0[i][cc] = g[c - 1] * (s0[i][cc] + x);
                    }
                }
            }
        }
    }
    outs
}

/// Role binding of a matrix-state scan's inputs, recovered numerically
/// from its affine lift: per step `A_t` and `U_t = f(0, x)`.
pub struct ChunkLift {
    pub form: ChunkForm,
    /// Max relative error of the chunked algorithm vs the sequential
    /// scan, on steps recovered from the spec's own `A_t`, `U_t`.
    pub err: f64,
}

/// Chunked lift of a matrix-state spec: find the affine lift, classify
/// `A`, recover `(a, β, k, v)` per step from `(A_t, U_t)`, and check the
/// chunked algorithm against the sequential recurrence with readouts.
pub fn chunk_lift(spec: &ScanSpec) -> Option<ChunkLift> {
    let (n, m) = spec.mat?;
    let (lift, _) = scan_lift(spec).ok()?;
    let ScanLift::Affine {
        a,
        conj: Conj::Id,
        element,
        ..
    } = &lift
    else {
        return None;
    };
    let form = chunk_form(*a)?;
    let d = spec.d;
    let el = Fun::trace(spec.n_x, element);
    let diag = matches!(
        form,
        ChunkForm::Cumulative | ChunkForm::ScalarDecay | ChunkForm::DiagDecay
    );
    let mut rng = Rng(0xc4c4);
    let mut worst = 0.0f64;
    for _ in 0..6 {
        let t = 5 + rng.below(40);
        let mut steps = vec![];
        for _ in 0..t {
            let e = el.call(&(spec.sample_x)(&mut rng));
            let (am, u): (Mat, Mat) = if diag {
                let a: Mat = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| if i == j { e[i * m] } else { 0.0 })
                            .collect()
                    })
                    .collect();
                (
                    a,
                    (0..n)
                        .map(|i| (0..m).map(|c| e[d + i * m + c]).collect())
                        .collect(),
                )
            } else {
                let a: Mat = (0..n)
                    .map(|i| (0..n).map(|j| e[(i * m) * d + j * m]).collect())
                    .collect();
                (
                    a,
                    (0..n)
                        .map(|i| (0..m).map(|c| e[d * d + i * m + c]).collect())
                        .collect(),
                )
            };
            // U = β k vᵀ: rank 1; take k ∝ U's dominant column structure
            let (beta, k, v, av) = match form {
                ChunkForm::ScalarDecayDelta => {
                    let sp = real_spectrum(&am)?;
                    let alpha = sp.iter().find(|s| s.1 >= n - 1)?.0;
                    // βkkᵀ = I − A/α
                    let mm = madd(&eye(n), &am, -1.0 / alpha);
                    let i0 = (0..n).max_by(|&x, &y| mm[x][x].partial_cmp(&mm[y][y]).unwrap())?;
                    let bk2 = mm[i0][i0];
                    if bk2 <= 1e-14 {
                        // β k = 0: U = 0 too
                        (0.0, vec![0.0; n], vec![0.0; m], vec![alpha; n])
                    } else {
                        // choose ‖k‖ so that k matches: k_i = mm[i][i0]/sqrt(bk2)·s, β = 1/s²
                        let kk: Vec<f64> = (0..n).map(|i| mm[i][i0] / bk2.sqrt()).collect();
                        // β = 1 with this k; v = Uᵀk / (kᵀk)
                        let kn: f64 = kk.iter().map(|x| x * x).sum();
                        let v: Vec<f64> = (0..m)
                            .map(|c| (0..n).map(|i| u[i][c] * kk[i]).sum::<f64>() / kn)
                            .collect();
                        (1.0, kk, v, vec![alpha; n])
                    }
                }
                _ => {
                    // β = 1, k = column scale, v = row: pick the largest row of U
                    let i0 = (0..n).max_by(|&x, &y| {
                        let nx: f64 = u[x].iter().map(|v| v * v).sum();
                        let ny: f64 = u[y].iter().map(|v| v * v).sum();
                        nx.partial_cmp(&ny).unwrap()
                    })?;
                    let v = u[i0].clone();
                    let vn: f64 = v.iter().map(|x| x * x).sum();
                    let k: Vec<f64> = if vn == 0.0 {
                        vec![0.0; n]
                    } else {
                        (0..n)
                            .map(|i| u[i].iter().zip(&v).map(|(a, b)| a * b).sum::<f64>() / vn)
                            .collect()
                    };
                    (1.0, k, v, (0..n).map(|i| am[i][i]).collect())
                }
            };
            let q: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            steps.push(Step {
                a: av,
                beta,
                k,
                v,
                q,
            });
        }
        let delta = form == ChunkForm::ScalarDecayDelta;
        // sequential reference uses the spec's own step, not the recovered
        // roles: S ← f(S, x)
        let want = scan_sequential(&steps, n, m, delta);
        let got = scan_chunked(&steps, n, m, form, 8);
        for (a, b) in want.iter().zip(&got) {
            for (x, y) in a.iter().zip(b) {
                let e = (x - y).abs() / (x.abs() + 1e-6);
                worst = worst.max(if e.is_nan() { f64::INFINITY } else { e });
            }
        }
    }
    Some(ChunkLift { form, err: worst })
}

// ---------------------------------------------------------------------------
// Saved-state backward of online lifts
// ---------------------------------------------------------------------------

/// A block of a first-pass monoid's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum R1Block {
    /// Slot `j` is a running sum.
    Sum(usize),
    /// Slot `j` is a running max.
    Max(usize),
    /// Slots `(m, acc₁…)`: an online softmax (`accₗ = Σ gₗ·e^{s−m}`).
    Softmax(Vec<usize>),
}

/// Decompose `m` into independent sum / max / online-softmax blocks
/// (dependence structure from `deps`, block kind by probing the combine).
pub fn r1_blocks(m: &Monoid) -> Option<Vec<R1Block>> {
    let w = m.width;
    let mut parent: Vec<usize> = (0..w).collect();
    fn find(p: &mut Vec<usize>, x: usize) -> usize {
        if p[x] != x {
            let r = find(p, p[x]);
            p[x] = r;
        }
        p[x]
    }
    for i in 0..w {
        let d = m.combine.deps(m.outs[i]);
        for j in 0..w {
            if d >> j & 1 == 1 || d >> (w + j) & 1 == 1 {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![];
    for i in 0..w {
        let r = find(&mut parent, i);
        match groups
            .iter_mut()
            .find(|g| find(&mut parent.clone(), g[0]) == r)
        {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let f = Fun {
        g: m.combine.clone(),
        outs: m.outs.clone(),
    };
    let mut rng = Rng(0xb10c);
    let probes: Vec<(Vec<f64>, Vec<f64>)> = (0..6)
        .map(|_| {
            let v: Vec<f64> = (0..2 * w).map(|_| rng.normal()).collect();
            let o = f.call(&v);
            (v, o)
        })
        .collect();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
    let mut out = vec![];
    for g in groups {
        if g.len() == 1 {
            let j = g[0];
            if probes.iter().all(|(v, o)| close(o[j], v[j] + v[w + j])) {
                out.push(R1Block::Sum(j));
            } else if probes.iter().all(|(v, o)| close(o[j], v[j].max(v[w + j]))) {
                out.push(R1Block::Max(j));
            } else {
                return None;
            }
            continue;
        }
        // online softmax: max slot first, then accumulators
        let ok = probes.iter().all(|(v, o)| {
            let (ma, me) = (v[g[0]], v[w + g[0]]);
            let mm = ma.max(me);
            close(o[g[0]], mm)
                && g[1..]
                    .iter()
                    .all(|&l| close(o[l], v[l] * (ma - mm).exp() + v[w + l] * (me - mm).exp()))
        });
        if !ok {
            return None;
        }
        out.push(R1Block::Softmax(g));
    }
    Some(out)
}

/// What the backward needs from an online lift.
pub struct SavedBwd {
    pub n_x: usize,
    pub w1: usize,
    pub r: usize,
    pub blocks: Vec<R1Block>,
    /// `x → R₁ element`.
    pub lift: Fun,
    /// `R₁ state → b` (reference in search coordinates).
    pub phi: Fun,
    /// Basis `D(x, b)`: vars `x…, b…`.
    pub basis: Fun,
    pub gens: Vec<Mat>,
    /// `x → monoid element`.
    pub element: Fun,
}

impl SavedBwd {
    /// `x̄` for one element from the saved final state `T = (T₁, S)`
    /// and its cotangent: `x̄ = ∇ₓ[S̄·D(x, b_T) + w·T₁lin(x)]` with
    /// `w = T̄₁ + ∂(μ·b(T₁))/∂T₁`, `μ_p = S̄·A_p·S` — because
    /// `S_T = Σⱼ D(xⱼ, b_T)` identically, so `dS_T/dxᵢ = ∂ₓD(xᵢ, b_T) +
    /// A·S_T·db_T/dxᵢ`, and `dT₁/dxᵢ` is the block-wise derivative of the
    /// first pass (sums: the lift; max: routed to the argmax; online
    /// softmax: the same identity one level down). Vars: `x` (n_x), `T`
    /// (w1 + r), `T̄` (w1 + r); outputs `n_x`.
    pub fn element_grad(&self) -> (ExprGraph, Vec<ExprId>) {
        let (nx, w1, r) = (self.n_x, self.w1, self.r);
        let p = self.gens.len();
        let w = w1 + r;
        // ∂(μ·b(T₁))/∂T₁ as a graph over (T₁, μ)
        let (hg, hv) = {
            let (mut g, outs) = trace(|| {
                let t: Vec<ExprId> = (0..w1).map(|i| ExprId::var(i as u16)).collect();
                let b = inline(&self.phi.g, &self.phi.outs, &t);
                let mut acc = lit(0.0);
                for (k, &bk) in b.iter().enumerate() {
                    acc = acc + ExprId::var((w1 + k) as u16) * bk;
                }
                vec![acc]
            });
            let one = g.lit(1.0);
            let v = g.vjp(&outs, &[one], w1);
            (g, v)
        };
        let (mut g, f) = trace(|| {
            let x: Vec<ExprId> = (0..nx).map(|i| ExprId::var(i as u16)).collect();
            let t: Vec<ExprId> = (0..w).map(|i| ExprId::var((nx + i) as u16)).collect();
            let tb: Vec<ExprId> = (0..w).map(|i| ExprId::var((nx + w + i) as u16)).collect();
            let (t1, s) = t.split_at(w1);
            let (tb1, sb) = tb.split_at(w1);
            let b = inline(&self.phi.g, &self.phi.outs, t1);
            // μ_p = S̄·A_p·S
            let mu: Vec<ExprId> = (0..p)
                .map(|k| {
                    let mut acc = lit(0.0);
                    for i in 0..r {
                        for j in 0..r {
                            let a = self.gens[k][i][j];
                            if a != 0.0 {
                                acc = acc + sb[i] * lit(a) * s[j];
                            }
                        }
                    }
                    acc
                })
                .collect();
            let mut hin = t1.to_vec();
            hin.extend_from_slice(&mu);
            let dh = inline(&hg, &hv, &hin);
            let wv: Vec<ExprId> = (0..w1).map(|i| tb1[i] + dh[i]).collect();
            // S̄·D(x, b_T)
            let mut xb = x.clone();
            xb.extend_from_slice(&b);
            let d = inline(&self.basis.g, &self.basis.outs, &xb);
            let mut fsum = lit(0.0);
            for i in 0..r {
                fsum = fsum + sb[i] * d[i];
            }
            // w·T₁lin(x)
            let l = inline(&self.lift.g, &self.lift.outs, &x);
            let eq = |a: ExprId, b: ExprId| sel(a - b, lit(-1.0), sel(b - a, lit(-1.0), lit(1.0)));
            for blk in &self.blocks {
                match blk {
                    R1Block::Sum(j) => fsum = fsum + wv[*j] * l[*j],
                    R1Block::Max(j) => {
                        fsum = fsum + wv[*j] * sel(eq(l[*j], t1[*j]), l[*j], lit(0.0))
                    }
                    R1Block::Softmax(sl) => {
                        let (mj, sv) = (sl[0], l[sl[0]]);
                        let am = sel(eq(sv, t1[mj]), sv, lit(0.0));
                        fsum = fsum + wv[mj] * am;
                        for &a in &sl[1..] {
                            fsum = fsum + wv[a] * (l[a] * eexp(sv - t1[mj]) - t1[a] * am);
                        }
                    }
                }
            }
            vec![fsum]
        });
        let one = g.lit(1.0);
        let xb = g.vjp(&f, &[one], nx);
        (g, xb)
    }
}
