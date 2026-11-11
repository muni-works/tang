//! `Scalar` trait implementation for `ExprId`.
//!
//! Every Scalar method decomposes into the 9 RISC primitives by inserting
//! nodes into the thread-local expression graph.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use tang::Scalar;

use crate::node::ExprId;
use crate::with_graph;

// --- Operator impls (all delegate to graph ops) ---

impl Add for ExprId {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        with_graph(|g| g.add(self, rhs))
    }
}

impl Sub for ExprId {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // sub(a, b) = add(a, neg(b))
        let nb = with_graph(|g| g.neg(rhs));
        with_graph(|g| g.add(self, nb))
    }
}

impl Mul for ExprId {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        with_graph(|g| g.mul(self, rhs))
    }
}

impl Div for ExprId {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // div(a, b) = mul(a, recip(b))
        let rb = with_graph(|g| g.recip(rhs));
        with_graph(|g| g.mul(self, rb))
    }
}

impl Neg for ExprId {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        with_graph(|g| g.neg(self))
    }
}

impl AddAssign for ExprId {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for ExprId {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for ExprId {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for ExprId {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// --- Scalar impl ---

impl Scalar for ExprId {
    const ZERO: Self = ExprId::ZERO;
    const ONE: Self = ExprId::ONE;
    const TWO: Self = ExprId::TWO;
    // These can't be const because they need graph insertion.
    // We use ZERO as placeholder — the actual values are injected lazily.
    const HALF: Self = ExprId(u32::MAX - 1);
    const PI: Self = ExprId(u32::MAX - 2);
    const TAU: Self = ExprId(u32::MAX - 3);
    const FRAC_PI_2: Self = ExprId(u32::MAX - 4);
    const EPSILON: Self = ExprId(u32::MAX - 5);
    const INFINITY: Self = ExprId(u32::MAX - 6);
    const NEG_INFINITY: Self = ExprId(u32::MAX - 7);

    #[inline]
    fn sqrt(self) -> Self {
        with_graph(|g| g.sqrt(self))
    }

    #[inline]
    fn abs(self) -> Self {
        // abs(x) = select(x, x, -x). Not sqrt(x * x): that overflows for
        // |x| > ~1e154 (f64) or ~1e19 (f32) and underflows to 0 below their
        // square roots of the smallest normal.
        let neg = with_graph(|g| g.neg(self));
        with_graph(|g| g.select(self, self, neg))
    }

    #[inline]
    fn sin(self) -> Self {
        with_graph(|g| g.sin(self))
    }

    #[inline]
    fn cos(self) -> Self {
        // cos(x) = sin(x + PI/2)
        let half_pi = Self::from_f64(std::f64::consts::FRAC_PI_2);
        let shifted = with_graph(|g| g.add(self, half_pi));
        with_graph(|g| g.sin(shifted))
    }

    #[inline]
    fn tan(self) -> Self {
        // tan(x) = sin(x) * recip(cos(x))
        let s = self.sin();
        let c = self.cos();
        let rc = with_graph(|g| g.recip(c));
        with_graph(|g| g.mul(s, rc))
    }

    #[inline]
    fn asin(self) -> Self {
        // asin(x) = atan2(x, sqrt(1 - x*x))
        let one = ExprId::ONE;
        let xx = with_graph(|g| g.mul(self, self));
        let diff = with_graph(|g| {
            let neg_xx = g.neg(xx);
            g.add(one, neg_xx)
        });
        let sq = with_graph(|g| g.sqrt(diff));
        with_graph(|g| g.atan2(self, sq))
    }

    #[inline]
    fn acos(self) -> Self {
        // acos(x) = atan2(sqrt(1 - x*x), x)
        let one = ExprId::ONE;
        let xx = with_graph(|g| g.mul(self, self));
        let diff = with_graph(|g| {
            let neg_xx = g.neg(xx);
            g.add(one, neg_xx)
        });
        let sq = with_graph(|g| g.sqrt(diff));
        with_graph(|g| g.atan2(sq, self))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        with_graph(|g| g.atan2(self, other))
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        // min(a, b) = select(a - b, b, a): exact. The old
        // 0.5 * (a + b - sqrt((a-b)^2)) rounded (min(1, 1e-17) was 0) and
        // squared the difference, which under/overflows in f32.
        let diff = self - other;
        with_graph(|g| g.select(diff, other, self))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        // max(a, b) = select(a - b, a, b): exact (see `min`).
        let diff = self - other;
        with_graph(|g| g.select(diff, self, other))
    }

    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    #[inline]
    fn recip(self) -> Self {
        with_graph(|g| g.recip(self))
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        match n {
            0 => ExprId::ONE,
            1 => self,
            2 => with_graph(|g| g.mul(self, self)),
            3 => {
                let sq = with_graph(|g| g.mul(self, self));
                with_graph(|g| g.mul(sq, self))
            }
            4 => {
                let sq = with_graph(|g| g.mul(self, self));
                with_graph(|g| g.mul(sq, sq))
            }
            -1 => with_graph(|g| g.recip(self)),
            -2 => {
                let sq = with_graph(|g| g.mul(self, self));
                with_graph(|g| g.recip(sq))
            }
            _ => self.powf(Self::from_f64(n as f64)),
        }
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        // copysign(x, s) = abs(x) * signum(s)
        let ax = self.abs();
        let ss = sign.signum();
        with_graph(|g| g.mul(ax, ss))
    }

    #[inline]
    fn signum(self) -> Self {
        // signum(x) = x * recip(sqrt(x * x))
        let xx = with_graph(|g| g.mul(self, self));
        let abs_x = with_graph(|g| g.sqrt(xx));
        let r = with_graph(|g| g.recip(abs_x));
        with_graph(|g| g.mul(self, r))
    }

    #[inline]
    fn floor(self) -> Self {
        with_graph(|g| g.floor(self))
    }

    #[inline]
    fn ceil(self) -> Self {
        with_graph(|g| {
            if let Some(v) = g.node(self).as_f64() {
                g.lit(v.ceil())
            } else {
                panic!("ceil() requires a literal expression")
            }
        })
    }

    #[inline]
    fn round(self) -> Self {
        with_graph(|g| {
            if let Some(v) = g.node(self).as_f64() {
                g.lit(v.round())
            } else {
                panic!("round() requires a literal expression")
            }
        })
    }

    #[inline]
    fn exp(self) -> Self {
        // exp(x) = exp2(x * log2(e))
        let log2_e = Self::from_f64(std::f64::consts::LOG2_E);
        let scaled = with_graph(|g| g.mul(self, log2_e));
        with_graph(|g| g.exp2(scaled))
    }

    #[inline]
    fn ln(self) -> Self {
        // ln(x) = log2(x) * ln(2)
        let ln_2 = Self::from_f64(std::f64::consts::LN_2);
        let l = with_graph(|g| g.log2(self));
        with_graph(|g| g.mul(l, ln_2))
    }

    #[inline]
    fn powf(self, p: Self) -> Self {
        // powf(x, p) = exp2(p * log2(x))
        let l = with_graph(|g| g.log2(self));
        let pl = with_graph(|g| g.mul(p, l));
        with_graph(|g| g.exp2(pl))
    }

    #[inline]
    fn sinh(self) -> Self {
        // sinh(x) = 0.5 * (exp(x) - exp(-x))
        let half = Self::from_f64(0.5);
        let ex = self.exp();
        let neg_x = with_graph(|g| g.neg(self));
        let enx = Scalar::exp(neg_x);
        let diff = ex - enx;
        with_graph(|g| g.mul(half, diff))
    }

    #[inline]
    fn cosh(self) -> Self {
        // cosh(x) = 0.5 * (exp(x) + exp(-x))
        let half = Self::from_f64(0.5);
        let ex = self.exp();
        let neg_x = with_graph(|g| g.neg(self));
        let enx = Scalar::exp(neg_x);
        let sum = ex + enx;
        with_graph(|g| g.mul(half, sum))
    }

    #[inline]
    fn tanh(self) -> Self {
        // sinh(x) / cosh(x) is inf / inf = NaN once |x| > ~710 (f64; ~89 in
        // f32). Instead, with a = min(|x|, 20) (tanh is 1 to f64 precision
        // past 20, with slope 0):
        // - a < 0.05: the odd series to a^9, truncation < 1e-16 relative;
        // - else 1 - 2 / (e^(2a) + 1), cancellation <= ~2e-15 relative there.
        let lit = Self::from_f64;
        let a = self.abs().min(lit(20.0));
        let e = (a * Self::TWO).exp();
        let big = Self::ONE - Self::TWO / (e + Self::ONE);
        let a2 = a * a;
        let series = a
            * (Self::ONE
                + a2 * (lit(-1.0 / 3.0)
                    + a2 * (lit(2.0 / 15.0)
                        + a2 * (lit(-17.0 / 315.0) + a2 * lit(62.0 / 2835.0)))));
        let r = Self::select(a - lit(0.05), big, series);
        Self::select(self, r, -r)
    }

    #[inline]
    fn acosh(self) -> Self {
        // acosh(x) = ln(x + sqrt(x*x - 1))
        let one = ExprId::ONE;
        let xx = with_graph(|g| g.mul(self, self));
        let diff = with_graph(|g| {
            let neg_one = g.neg(one);
            g.add(xx, neg_one)
        });
        let sq = with_graph(|g| g.sqrt(diff));
        let sum = with_graph(|g| g.add(self, sq));
        Scalar::ln(sum)
    }

    #[inline]
    fn asinh(self) -> Self {
        // asinh(x) = ln(x + sqrt(x*x + 1))
        let one = ExprId::ONE;
        let xx = with_graph(|g| g.mul(self, self));
        let sum_inner = with_graph(|g| g.add(xx, one));
        let sq = with_graph(|g| g.sqrt(sum_inner));
        let sum = with_graph(|g| g.add(self, sq));
        Scalar::ln(sum)
    }

    #[inline]
    fn atanh(self) -> Self {
        // atanh(x) = 0.5 * ln((1+x) / (1-x))
        let half = Self::from_f64(0.5);
        let one = ExprId::ONE;
        let one_plus = with_graph(|g| g.add(one, self));
        let neg_x = with_graph(|g| g.neg(self));
        let one_minus = with_graph(|g| g.add(one, neg_x));
        let ratio = one_plus / one_minus;
        let l = Scalar::ln(ratio);
        with_graph(|g| g.mul(half, l))
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        with_graph(|g| g.lit(v))
    }

    #[inline]
    fn to_f64(self) -> f64 {
        panic!("cannot evaluate symbolic ExprId to f64 — use ExprGraph::eval() instead")
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        with_graph(|g| g.lit(v as f64))
    }

    #[inline]
    fn select(cond: Self, a: Self, b: Self) -> Self {
        with_graph(|g| g.select(cond, a, b))
    }
}

// --- Display for ExprId (needed by Scalar bound) ---
// Note: ExprId already has Display in node.rs, just showing "eN".
// The detailed expression display is in display.rs via ExprGraph::fmt_expr.

#[cfg(test)]
mod tests {
    use tang::Scalar;

    /// Evaluate `f` traced over one input.
    fn eval1(f: impl Fn(crate::ExprId) -> crate::ExprId, x: f64) -> f64 {
        let (g, out) = crate::trace(|| f(crate::ExprId::var(0)));
        g.compile(out)(&[x])
    }

    #[test]
    fn min_max_abs_are_exact() {
        for (a, b) in [(1.0, 1e-17), (-3.0, 2.5), (1e200, -1e200), (1e-30, 2e-30)] {
            let lo = eval1(|x| x.min(crate::ExprId::from_f64(b)), a);
            let hi = eval1(|x| x.max(crate::ExprId::from_f64(b)), a);
            assert_eq!(lo, a.min(b), "min({a}, {b})");
            assert_eq!(hi, a.max(b), "max({a}, {b})");
        }
        for x in [-1e200, -2.5, 0.0, 1e-300, 3.0] {
            assert_eq!(eval1(|v| v.abs(), x), x.abs(), "abs({x})");
        }
    }

    #[test]
    fn tanh_is_finite_and_accurate() {
        for x in [
            -2000.0, -30.0, -1.0, -0.04, -1e-8, 0.0, 1e-8, 0.03, 0.05, 0.5, 5.0, 25.0, 2000.0,
        ] {
            let got = eval1(|v| v.tanh(), x);
            let want: f64 = x.tanh();
            assert!(got.is_finite(), "tanh({x}) = {got}");
            let err = (got - want).abs() / want.abs().max(1e-300);
            assert!(
                want == 0.0 && got == 0.0 || err < 4e-15,
                "tanh({x}): {got} vs {want}, rel {err:e}"
            );
        }
    }

    use crate::{trace, ExprId};

    #[test]
    fn basic_arithmetic() {
        let (g, result) = trace(|| {
            let x = ExprId::from_f64(3.0);
            let y = ExprId::from_f64(4.0);
            x + y
        });
        // Should be a single Add node
        let val = g.eval::<f64>(result, &[]);
        assert!((val - 7.0).abs() < 1e-10);
    }

    #[test]
    fn var_trace() {
        let (g, result) = trace(|| {
            let x: ExprId = Scalar::from_f64(0.0); // creates a Lit(0.0) = ZERO
            x
        });
        assert_eq!(result, ExprId::ZERO);
        assert_eq!(g.len(), 3); // just ZERO, ONE, TWO
    }

    #[test]
    fn constants_are_lits() {
        let (_g, (half, pi)) = trace(|| {
            let h = ExprId::from_f64(0.5);
            let p = ExprId::from_f64(std::f64::consts::PI);
            (h, p)
        });
        assert_ne!(half, pi);
    }
}
