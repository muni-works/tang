//! Summary statistics: mean, variance, stddev, skewness, kurtosis.
//!
//! Generic over `Scalar`. For `f32` and `f64` slices, the kernels dispatch
//! to SIMD paths (AVX2 on x86_64, NEON on aarch64) when available at
//! runtime; other targets and other scalar types use a portable fallback
//! that LLVM auto-vectorizes when it can.
//!
//! The two-pass algorithm is used everywhere: first the mean, then the
//! central sums M2/M3/M4. This is a few ulps less accurate than Welford
//! on pathological inputs but ~2x the throughput and trivially
//! vectorizable.

use core::any::TypeId;
use tang::Scalar;

/// All four moments computed together in two passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments<S> {
    /// Sample size.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: S,
    /// Population variance (divisor `n`).
    pub variance: S,
    /// Method-of-moments skewness: `m3 / m2^(3/2)`.
    pub skewness: S,
    /// Excess kurtosis: `m4 / m2^2 - 3`.
    pub kurtosis: S,
}

/// Arithmetic mean of a slice. Panics if empty.
#[inline]
pub fn mean<S: Scalar>(x: &[S]) -> S {
    assert!(!x.is_empty(), "mean: empty slice");
    if let Some(v) = dispatch_f32(x, mean_f32) {
        return v;
    }
    if let Some(v) = dispatch_f64(x, mean_f64) {
        return v;
    }
    sum_generic(x) / S::from_f64(x.len() as f64)
}

/// Population variance (divisor `n`).
#[inline]
pub fn variance<S: Scalar>(x: &[S]) -> S {
    let m = moments(x);
    m.variance
}

/// Sample variance (divisor `n - 1`). Requires `len >= 2`.
#[inline]
pub fn variance_sample<S: Scalar>(x: &[S]) -> S {
    assert!(x.len() >= 2, "variance_sample: need at least 2 samples");
    let (_, m2, _, _) = central_sums(x);
    m2 / S::from_f64((x.len() - 1) as f64)
}

/// Population standard deviation.
#[inline]
pub fn stddev<S: Scalar>(x: &[S]) -> S {
    variance(x).sqrt()
}

/// Sample standard deviation (divisor `n - 1`).
#[inline]
pub fn stddev_sample<S: Scalar>(x: &[S]) -> S {
    variance_sample(x).sqrt()
}

/// Method-of-moments skewness: `(M3 / n) / (M2 / n)^(3/2)`.
///
/// This is the biased "population" estimator. Returns `0` if the variance
/// underflows to zero (constant input).
#[inline]
pub fn skewness<S: Scalar>(x: &[S]) -> S {
    moments(x).skewness
}

/// Excess kurtosis: `(M4 / n) / (M2 / n)^2 - 3`.
///
/// Normal distribution has excess kurtosis 0. Add 3 for raw (Pearson)
/// kurtosis, or use [`kurtosis_raw`].
#[inline]
pub fn kurtosis<S: Scalar>(x: &[S]) -> S {
    moments(x).kurtosis
}

/// Raw (non-excess) kurtosis: `(M4 / n) / (M2 / n)^2`.
///
/// Normal distribution has raw kurtosis 3.
#[inline]
pub fn kurtosis_raw<S: Scalar>(x: &[S]) -> S {
    kurtosis(x) + S::from_f64(3.0)
}

/// Central moment of arbitrary integer order.
///
/// Returns `sum((x_i - mean)^order) / n`. Order 0 is 1, order 1 is 0,
/// order 2 is population variance.
pub fn central_moment<S: Scalar>(x: &[S], order: u32) -> S {
    assert!(!x.is_empty(), "central_moment: empty slice");
    let n = S::from_f64(x.len() as f64);
    match order {
        0 => S::ONE,
        1 => S::ZERO,
        2 => variance(x),
        3 => {
            let (_, _, m3, _) = central_sums(x);
            m3 / n
        }
        4 => {
            let (_, _, _, m4) = central_sums(x);
            m4 / n
        }
        k => {
            let m = mean(x);
            let mut acc = S::ZERO;
            for &v in x {
                acc += (v - m).powi(k as i32);
            }
            acc / n
        }
    }
}

/// Compute mean + all four central moments in two passes.
pub fn moments<S: Scalar>(x: &[S]) -> Moments<S> {
    assert!(!x.is_empty(), "moments: empty slice");
    let n_us = x.len();
    let (mean_, m2, m3, m4) = central_sums(x);
    let n = S::from_f64(n_us as f64);
    let variance = m2 / n;
    let (skewness, kurtosis) = if variance > S::ZERO {
        let v32 = variance.sqrt() * variance;
        let sk = (m3 / n) / v32;
        let kurt = (m4 / n) / (variance * variance) - S::from_f64(3.0);
        (sk, kurt)
    } else {
        (S::ZERO, S::ZERO)
    };
    Moments {
        n: n_us,
        mean: mean_,
        variance,
        skewness,
        kurtosis,
    }
}

/// Two-pass: mean then (M2, M3, M4) = sum((x-mean)^k) for k=2,3,4.
#[inline]
fn central_sums<S: Scalar>(x: &[S]) -> (S, S, S, S) {
    let m = mean(x);
    if let Some(v) = dispatch_f32_moments(x, m, moments_f32) {
        return v;
    }
    if let Some(v) = dispatch_f64_moments(x, m, moments_f64) {
        return v;
    }
    let mut m2 = S::ZERO;
    let mut m3 = S::ZERO;
    let mut m4 = S::ZERO;
    for &v in x {
        let d = v - m;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    (m, m2, m3, m4)
}

#[inline]
fn sum_generic<S: Scalar>(x: &[S]) -> S {
    let mut s = S::ZERO;
    for &v in x {
        s += v;
    }
    s
}

// ── Type-dispatch helpers ────────────────────────────────────────────

#[inline]
fn dispatch_f32<S: Scalar>(x: &[S], f: fn(&[f32]) -> f32) -> Option<S> {
    if TypeId::of::<S>() == TypeId::of::<f32>() {
        // SAFETY: S == f32 verified by TypeId.
        let xs: &[f32] = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let r = f(xs);
        Some(unsafe { core::mem::transmute_copy::<f32, S>(&r) })
    } else {
        None
    }
}

#[inline]
fn dispatch_f64<S: Scalar>(x: &[S], f: fn(&[f64]) -> f64) -> Option<S> {
    if TypeId::of::<S>() == TypeId::of::<f64>() {
        let xs: &[f64] = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let r = f(xs);
        Some(unsafe { core::mem::transmute_copy::<f64, S>(&r) })
    } else {
        None
    }
}

#[inline]
fn dispatch_f32_moments<S: Scalar>(
    x: &[S],
    mean: S,
    f: fn(&[f32], f32) -> (f32, f32, f32),
) -> Option<(S, S, S, S)> {
    if TypeId::of::<S>() == TypeId::of::<f32>() {
        let xs: &[f32] = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let m: f32 = unsafe { core::mem::transmute_copy::<S, f32>(&mean) };
        let (m2, m3, m4) = f(xs, m);
        let cvt = |v: f32| -> S { unsafe { core::mem::transmute_copy::<f32, S>(&v) } };
        Some((mean, cvt(m2), cvt(m3), cvt(m4)))
    } else {
        None
    }
}

#[inline]
fn dispatch_f64_moments<S: Scalar>(
    x: &[S],
    mean: S,
    f: fn(&[f64], f64) -> (f64, f64, f64),
) -> Option<(S, S, S, S)> {
    if TypeId::of::<S>() == TypeId::of::<f64>() {
        let xs: &[f64] = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let m: f64 = unsafe { core::mem::transmute_copy::<S, f64>(&mean) };
        let (m2, m3, m4) = f(xs, m);
        let cvt = |v: f64| -> S { unsafe { core::mem::transmute_copy::<f64, S>(&v) } };
        Some((mean, cvt(m2), cvt(m3), cvt(m4)))
    } else {
        None
    }
}

// ── f32/f64 kernels with SIMD dispatch ───────────────────────────────

fn mean_f32(x: &[f32]) -> f32 {
    sum_f32(x) / x.len() as f32
}

fn mean_f64(x: &[f64]) -> f64 {
    sum_f64(x) / x.len() as f64
}

// the scalar fallback is unreachable on aarch64+neon (not on CI's x86_64)
#[allow(unreachable_code)]
fn sum_f32(x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified at runtime.
            return unsafe { x86::sum_f32_avx2(x) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: NEON is part of the aarch64 baseline when target_feature=neon.
        return unsafe { neon::sum_f32_neon(x) };
    }
    sum_f32_scalar(x)
}

// the scalar fallback is unreachable on aarch64+neon (not on CI's x86_64)
#[allow(unreachable_code)]
fn sum_f64(x: &[f64]) -> f64 {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::sum_f64_avx2(x) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon::sum_f64_neon(x) };
    }
    sum_f64_scalar(x)
}

// the scalar fallback is unreachable on aarch64+neon (not on CI's x86_64)
#[allow(unreachable_code)]
fn moments_f32(x: &[f32], mean: f32) -> (f32, f32, f32) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::moments_f32_avx2(x, mean) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon::moments_f32_neon(x, mean) };
    }
    moments_f32_scalar(x, mean)
}

// the scalar fallback is unreachable on aarch64+neon (not on CI's x86_64)
#[allow(unreachable_code)]
fn moments_f64(x: &[f64], mean: f64) -> (f64, f64, f64) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::moments_f64_avx2(x, mean) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon::moments_f64_neon(x, mean) };
    }
    moments_f64_scalar(x, mean)
}

// ── Portable scalar fallbacks (4-way unrolled; auto-vectorizes well) ──

fn sum_f32_scalar(x: &[f32]) -> f32 {
    let mut a = [0.0f32; 4];
    let chunks = x.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        a[0] += c[0];
        a[1] += c[1];
        a[2] += c[2];
        a[3] += c[3];
    }
    let mut s = (a[0] + a[1]) + (a[2] + a[3]);
    for &v in rem {
        s += v;
    }
    s
}

fn sum_f64_scalar(x: &[f64]) -> f64 {
    let mut a = [0.0f64; 4];
    let chunks = x.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        a[0] += c[0];
        a[1] += c[1];
        a[2] += c[2];
        a[3] += c[3];
    }
    let mut s = (a[0] + a[1]) + (a[2] + a[3]);
    for &v in rem {
        s += v;
    }
    s
}

fn moments_f32_scalar(x: &[f32], mean: f32) -> (f32, f32, f32) {
    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    let mut m4 = 0.0f32;
    for &v in x {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    (m2, m3, m4)
}

fn moments_f64_scalar(x: &[f64], mean: f64) -> (f64, f64, f64) {
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;
    for &v in x {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    (m2, m3, m4)
}

// ── x86_64 AVX2 ──────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        let low = _mm256_castps256_ps128(v);
        let high = _mm256_extractf128_ps::<1>(v);
        let s128 = _mm_add_ps(low, high);
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_pd(v: __m256d) -> f64 {
        let low = _mm256_castpd256_pd128(v);
        let high = _mm256_extractf128_pd::<1>(v);
        let s128 = _mm_add_pd(low, high);
        let hi = _mm_unpackhi_pd(s128, s128);
        _mm_cvtsd_f64(_mm_add_sd(s128, hi))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_f32_avx2(x: &[f32]) -> f32 {
        let n = x.len();
        let ptr = x.as_ptr();
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(ptr.add(i)));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(ptr.add(i + 8)));
            a2 = _mm256_add_ps(a2, _mm256_loadu_ps(ptr.add(i + 16)));
            a3 = _mm256_add_ps(a3, _mm256_loadu_ps(ptr.add(i + 24)));
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(ptr.add(i)));
            i += 8;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut s = hsum_ps(acc);
        while i < n {
            s += *ptr.add(i);
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_f64_avx2(x: &[f64]) -> f64 {
        let n = x.len();
        let ptr = x.as_ptr();
        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd();
        let mut a3 = _mm256_setzero_pd();
        let mut i = 0usize;
        while i + 16 <= n {
            a0 = _mm256_add_pd(a0, _mm256_loadu_pd(ptr.add(i)));
            a1 = _mm256_add_pd(a1, _mm256_loadu_pd(ptr.add(i + 4)));
            a2 = _mm256_add_pd(a2, _mm256_loadu_pd(ptr.add(i + 8)));
            a3 = _mm256_add_pd(a3, _mm256_loadu_pd(ptr.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            a0 = _mm256_add_pd(a0, _mm256_loadu_pd(ptr.add(i)));
            i += 4;
        }
        let acc = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
        let mut s = hsum_pd(acc);
        while i < n {
            s += *ptr.add(i);
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn moments_f32_avx2(x: &[f32], mean: f32) -> (f32, f32, f32) {
        let n = x.len();
        let ptr = x.as_ptr();
        let mv = _mm256_set1_ps(mean);
        let mut m2 = _mm256_setzero_ps();
        let mut m3 = _mm256_setzero_ps();
        let mut m4 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(ptr.add(i));
            let d = _mm256_sub_ps(xv, mv);
            let d2 = _mm256_mul_ps(d, d);
            let d3 = _mm256_mul_ps(d2, d);
            let d4 = _mm256_mul_ps(d2, d2);
            m2 = _mm256_add_ps(m2, d2);
            m3 = _mm256_add_ps(m3, d3);
            m4 = _mm256_add_ps(m4, d4);
            i += 8;
        }
        let mut m2s = hsum_ps(m2);
        let mut m3s = hsum_ps(m3);
        let mut m4s = hsum_ps(m4);
        while i < n {
            let d = *ptr.add(i) - mean;
            let d2 = d * d;
            m2s += d2;
            m3s += d2 * d;
            m4s += d2 * d2;
            i += 1;
        }
        (m2s, m3s, m4s)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn moments_f64_avx2(x: &[f64], mean: f64) -> (f64, f64, f64) {
        let n = x.len();
        let ptr = x.as_ptr();
        let mv = _mm256_set1_pd(mean);
        let mut m2 = _mm256_setzero_pd();
        let mut m3 = _mm256_setzero_pd();
        let mut m4 = _mm256_setzero_pd();
        let mut i = 0usize;
        while i + 4 <= n {
            let xv = _mm256_loadu_pd(ptr.add(i));
            let d = _mm256_sub_pd(xv, mv);
            let d2 = _mm256_mul_pd(d, d);
            let d3 = _mm256_mul_pd(d2, d);
            let d4 = _mm256_mul_pd(d2, d2);
            m2 = _mm256_add_pd(m2, d2);
            m3 = _mm256_add_pd(m3, d3);
            m4 = _mm256_add_pd(m4, d4);
            i += 4;
        }
        let mut m2s = hsum_pd(m2);
        let mut m3s = hsum_pd(m3);
        let mut m4s = hsum_pd(m4);
        while i < n {
            let d = *ptr.add(i) - mean;
            let d2 = d * d;
            m2s += d2;
            m3s += d2 * d;
            m4s += d2 * d2;
            i += 1;
        }
        (m2s, m3s, m4s)
    }
}

// ── aarch64 NEON ─────────────────────────────────────────────────────

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use core::arch::aarch64::*;

    pub unsafe fn sum_f32_neon(x: &[f32]) -> f32 {
        let n = x.len();
        let ptr = x.as_ptr();
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 16 <= n {
            a0 = vaddq_f32(a0, vld1q_f32(ptr.add(i)));
            a1 = vaddq_f32(a1, vld1q_f32(ptr.add(i + 4)));
            a2 = vaddq_f32(a2, vld1q_f32(ptr.add(i + 8)));
            a3 = vaddq_f32(a3, vld1q_f32(ptr.add(i + 12)));
            i += 16;
        }
        let acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        let mut s = vaddvq_f32(acc);
        while i < n {
            s += *ptr.add(i);
            i += 1;
        }
        s
    }

    pub unsafe fn sum_f64_neon(x: &[f64]) -> f64 {
        let n = x.len();
        let ptr = x.as_ptr();
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            a0 = vaddq_f64(a0, vld1q_f64(ptr.add(i)));
            a1 = vaddq_f64(a1, vld1q_f64(ptr.add(i + 2)));
            a2 = vaddq_f64(a2, vld1q_f64(ptr.add(i + 4)));
            a3 = vaddq_f64(a3, vld1q_f64(ptr.add(i + 6)));
            i += 8;
        }
        let acc = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        let mut s = vaddvq_f64(acc);
        while i < n {
            s += *ptr.add(i);
            i += 1;
        }
        s
    }

    pub unsafe fn moments_f32_neon(x: &[f32], mean: f32) -> (f32, f32, f32) {
        let n = x.len();
        let ptr = x.as_ptr();
        let mv = vdupq_n_f32(mean);
        let mut m2 = vdupq_n_f32(0.0);
        let mut m3 = vdupq_n_f32(0.0);
        let mut m4 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let xv = vld1q_f32(ptr.add(i));
            let d = vsubq_f32(xv, mv);
            let d2 = vmulq_f32(d, d);
            let d3 = vmulq_f32(d2, d);
            let d4 = vmulq_f32(d2, d2);
            m2 = vaddq_f32(m2, d2);
            m3 = vaddq_f32(m3, d3);
            m4 = vaddq_f32(m4, d4);
            i += 4;
        }
        let mut m2s = vaddvq_f32(m2);
        let mut m3s = vaddvq_f32(m3);
        let mut m4s = vaddvq_f32(m4);
        while i < n {
            let d = *ptr.add(i) - mean;
            let d2 = d * d;
            m2s += d2;
            m3s += d2 * d;
            m4s += d2 * d2;
            i += 1;
        }
        (m2s, m3s, m4s)
    }

    pub unsafe fn moments_f64_neon(x: &[f64], mean: f64) -> (f64, f64, f64) {
        let n = x.len();
        let ptr = x.as_ptr();
        let mv = vdupq_n_f64(mean);
        let mut m2 = vdupq_n_f64(0.0);
        let mut m3 = vdupq_n_f64(0.0);
        let mut m4 = vdupq_n_f64(0.0);
        let mut i = 0usize;
        while i + 2 <= n {
            let xv = vld1q_f64(ptr.add(i));
            let d = vsubq_f64(xv, mv);
            let d2 = vmulq_f64(d, d);
            let d3 = vmulq_f64(d2, d);
            let d4 = vmulq_f64(d2, d2);
            m2 = vaddq_f64(m2, d2);
            m3 = vaddq_f64(m3, d3);
            m4 = vaddq_f64(m4, d4);
            i += 2;
        }
        let mut m2s = vaddvq_f64(m2);
        let mut m3s = vaddvq_f64(m3);
        let mut m4s = vaddvq_f64(m4);
        while i < n {
            let d = *ptr.add(i) - mean;
            let d2 = d * d;
            m2s += d2;
            m3s += d2 * d;
            m4s += d2 * d2;
            i += 1;
        }
        (m2s, m3s, m4s)
    }
}

// ── DVec adapter ─────────────────────────────────────────────────────

use crate::DVec;

impl<S: Scalar> DVec<S> {
    /// Arithmetic mean of the elements.
    #[inline]
    pub fn mean_stat(&self) -> S {
        mean(self.as_slice())
    }

    /// Population variance.
    #[inline]
    pub fn variance(&self) -> S {
        variance(self.as_slice())
    }

    /// Sample variance (divisor `n - 1`).
    #[inline]
    pub fn variance_sample(&self) -> S {
        variance_sample(self.as_slice())
    }

    /// Population standard deviation.
    #[inline]
    pub fn stddev(&self) -> S {
        stddev(self.as_slice())
    }

    /// Sample standard deviation.
    #[inline]
    pub fn stddev_sample(&self) -> S {
        stddev_sample(self.as_slice())
    }

    /// Method-of-moments skewness.
    #[inline]
    pub fn skewness(&self) -> S {
        skewness(self.as_slice())
    }

    /// Excess kurtosis (normal = 0).
    #[inline]
    pub fn kurtosis(&self) -> S {
        kurtosis(self.as_slice())
    }

    /// Raw Pearson kurtosis (normal = 3).
    #[inline]
    pub fn kurtosis_raw(&self) -> S {
        kurtosis_raw(self.as_slice())
    }

    /// Mean + all four moments in two passes.
    #[inline]
    pub fn moments(&self) -> Moments<S> {
        moments(self.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use approx::assert_relative_eq;

    fn ref_moments(x: &[f64]) -> (f64, f64, f64, f64) {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
        for &v in x {
            let d = v - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
        }
        (mean, m2 / n, m3 / n, m4 / n)
    }

    #[test]
    fn mean_matches_reference() {
        let x: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let (ref_mean, _, _, _) = ref_moments(&x);
        assert_relative_eq!(mean(&x), ref_mean, max_relative = 1e-12);
    }

    #[test]
    fn variance_matches_reference() {
        let x: Vec<f64> = (0..1000).map(|i| (i as f64).sin() * 3.0 + 1.0).collect();
        let (_, m2, _, _) = ref_moments(&x);
        assert_relative_eq!(variance(&x), m2, max_relative = 1e-10);
    }

    #[test]
    fn skewness_matches_reference() {
        // Exponential-ish: positively skewed
        let x: Vec<f64> = (0..2048).map(|i| ((i as f64) * 0.01).exp()).collect();
        let (_, m2, m3, _) = ref_moments(&x);
        let expected = m3 / (m2.sqrt() * m2);
        assert_relative_eq!(skewness(&x), expected, max_relative = 1e-9);
    }

    #[test]
    fn kurtosis_matches_reference() {
        let x: Vec<f64> = (0..4096).map(|i| ((i as f64) * 0.003).sin()).collect();
        let (_, m2, _, m4) = ref_moments(&x);
        let expected = m4 / (m2 * m2) - 3.0;
        assert_relative_eq!(kurtosis(&x), expected, max_relative = 1e-10);
    }

    #[test]
    fn exact_constant_input_has_zero_moments() {
        // Values exactly representable: the mean divides cleanly back to
        // the original and higher moments are identically zero.
        let x = alloc::vec![4.0f64; 128];
        let m = moments(&x);
        assert_eq!(m.mean, 4.0);
        assert_eq!(m.variance, 0.0);
        assert_eq!(m.skewness, 0.0);
        assert_eq!(m.kurtosis, 0.0);

        let xf = alloc::vec![2.5f32; 257]; // odd length exercises tail
        let mf = moments(&xf);
        assert_eq!(mf.mean, 2.5);
        assert_eq!(mf.variance, 0.0);
    }

    #[test]
    fn f32_matches_f64_loosely() {
        let xf64: Vec<f64> = (0..777).map(|i| ((i as f64) * 0.013).sin() * 7.0).collect();
        let xf32: Vec<f32> = xf64.iter().map(|&v| v as f32).collect();
        let a = moments(&xf64);
        let b = moments(&xf32);
        assert_relative_eq!(a.mean, b.mean as f64, max_relative = 1e-5);
        assert_relative_eq!(a.variance, b.variance as f64, max_relative = 1e-4);
        assert_relative_eq!(a.skewness, b.skewness as f64, max_relative = 1e-3);
        assert_relative_eq!(a.kurtosis, b.kurtosis as f64, max_relative = 1e-2);
    }

    #[test]
    fn central_moment_orders() {
        let x: Vec<f64> = (0..512).map(|i| (i as f64 * 0.02).cos()).collect();
        assert_eq!(central_moment(&x, 0), 1.0);
        assert_relative_eq!(central_moment(&x, 1), 0.0, epsilon = 1e-12);
        assert_relative_eq!(central_moment(&x, 2), variance(&x), max_relative = 1e-12);
        let n = x.len() as f64;
        let m = mean(&x);
        let m5_ref: f64 = x.iter().map(|&v| (v - m).powi(5)).sum::<f64>() / n;
        assert_relative_eq!(central_moment(&x, 5), m5_ref, max_relative = 1e-10);
    }

    #[test]
    fn dvec_bindings() {
        let v = DVec::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(v.mean_stat(), 3.0);
        assert_relative_eq!(v.variance(), 2.0);
        assert_relative_eq!(v.variance_sample(), 2.5);
        assert_relative_eq!(v.skewness(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn small_sizes_and_tails() {
        // Exercise SIMD tail handling at many sizes around lane boundaries
        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sqrt()).collect();
            let (_, m2r, m3r, m4r) = ref_moments(&x);
            let nn = n as f64;
            let want = Moments {
                n,
                mean: x.iter().sum::<f64>() / nn,
                variance: m2r,
                skewness: if m2r > 0.0 {
                    m3r / (m2r.sqrt() * m2r)
                } else {
                    0.0
                },
                kurtosis: if m2r > 0.0 {
                    m4r / (m2r * m2r) - 3.0
                } else {
                    0.0
                },
            };
            let got = moments(&x);
            assert_relative_eq!(got.mean, want.mean, max_relative = 1e-12);
            assert_relative_eq!(got.variance, want.variance, max_relative = 1e-10);
            if want.variance > 1e-10 {
                assert_relative_eq!(got.skewness, want.skewness, max_relative = 1e-9);
                assert_relative_eq!(got.kurtosis, want.kurtosis, max_relative = 1e-9);
            }
        }
    }
}
