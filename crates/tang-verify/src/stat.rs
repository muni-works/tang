//! The per-tensor GPU-vs-CPU comparison.

/// GPU-vs-CPU statistics of one tensor, over every element.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stat {
    pub n: usize,
    /// ‖gpu − cpu‖₂ / ‖cpu‖₂ (0 when both are ~0; ∞ when only the CPU is ~0)
    pub rel_l2: f64,
    pub max_abs: f64,
    /// Where the worst element is. NaN/Inf elements win it: with poisoning they are the news.
    pub max_idx: usize,
    pub norm_gpu: f64,
    pub norm_cpu: f64,
    pub cos: f64,
    /// exact 0.0 counts: the "kernel wrote nothing" / "kernel wrote zeros" signal
    pub zeros_gpu: usize,
    pub zeros_cpu: usize,
    /// NaN/Inf counts: with poisoning, NaN in gpu = a region no kernel wrote
    pub nonfinite_gpu: usize,
    pub nonfinite_cpu: usize,
    /// elements with |d| > tol·(|cpu| + rms(cpu)) (non-finite d counts)
    pub bad_elems: usize,
    pub len_mismatch: bool,
}

/// Compare `gpu` against the reference `cpu`. `tol` sets the per-element bar of
/// [`Stat::bad_elems`]; the tensor verdict is [`Stat::bad`].
pub fn stat(gpu: &[f32], cpu: &[f32], tol: f64) -> Stat {
    let mut s = Stat {
        n: gpu.len().min(cpu.len()),
        len_mismatch: gpu.len() != cpu.len(),
        ..Default::default()
    };
    let (mut d2, mut g2, mut c2, mut gc) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &c) in gpu.iter().zip(cpu) {
        s.zeros_gpu += (g == 0.0) as usize;
        s.zeros_cpu += (c == 0.0) as usize;
        s.nonfinite_gpu += (!g.is_finite()) as usize;
        s.nonfinite_cpu += (!c.is_finite()) as usize;
        if !(g.is_finite() && c.is_finite()) {
            continue;
        }
        let (g, c) = (g as f64, c as f64);
        let d = (g - c).abs();
        if d > s.max_abs {
            s.max_abs = d;
        }
        d2 += d * d;
        g2 += g * g;
        c2 += c * c;
        gc += g * c;
    }
    // the max's index: a second pass so NaN entries can win it
    let mut best = -1f64;
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        let d = if g.is_finite() && c.is_finite() {
            ((g as f64) - (c as f64)).abs()
        } else {
            f64::INFINITY
        };
        if d > best {
            best = d;
            s.max_idx = i;
        }
    }
    s.norm_gpu = g2.sqrt();
    s.norm_cpu = c2.sqrt();
    s.rel_l2 = if s.norm_cpu > 1e-30 {
        d2.sqrt() / s.norm_cpu
    } else if s.norm_gpu > 1e-30 {
        f64::INFINITY
    } else {
        0.0
    };
    s.cos = if s.norm_gpu > 0.0 && s.norm_cpu > 0.0 {
        gc / (s.norm_gpu * s.norm_cpu)
    } else {
        0.0
    };
    let rms = s.rms_cpu();
    for (&g, &c) in gpu.iter().zip(cpu) {
        s.bad_elems += elem_bad(g, c, tol, rms) as usize;
    }
    s
}

/// The per-element bar shared by [`stat`] and the provenance search.
pub(crate) fn elem_bad(g: f32, c: f32, tol: f64, rms: f64) -> bool {
    let d = ((g as f64) - (c as f64)).abs();
    !d.is_finite() || d > tol * ((c as f64).abs() + rms)
}

impl Stat {
    /// A tensor fails when it's off by more than `tol` (relative L2), has NaN/Inf the CPU
    /// doesn't, has notably more exact zeros (>1% of elements), or the lengths differ.
    pub fn bad(&self, tol: f64) -> bool {
        self.len_mismatch
            || self.rel_l2 > tol
            || self.extra_nonfinite() > 0
            || self.extra_zeros() > self.n / 100
    }

    /// GPU NaN/Inf beyond the reference's: with poisoning, elements no kernel wrote.
    pub fn extra_nonfinite(&self) -> usize {
        self.nonfinite_gpu.saturating_sub(self.nonfinite_cpu)
    }

    /// GPU exact zeros beyond the reference's: a kernel that wrote zeros (or, unpoisoned,
    /// didn't write a zero-initialised buffer).
    pub fn extra_zeros(&self) -> usize {
        self.zeros_gpu.saturating_sub(self.zeros_cpu)
    }

    pub fn rms_cpu(&self) -> f64 {
        if self.n > 0 {
            self.norm_cpu / (self.n as f64).sqrt()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_clean() {
        let a = [1.0f32, -2.0, 3.0, 0.0];
        let s = stat(&a, &a, 1e-3);
        assert_eq!(s.rel_l2, 0.0);
        assert_eq!(s.zeros_gpu, 1);
        assert!(!s.bad(1e-3));
        assert!((s.cos - 1.0).abs() < 1e-12);
    }

    #[test]
    fn flags_dead_output() {
        // the "kernel wrote zeros" shape: gpu all zero where cpu is not
        let cpu: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let gpu = vec![0f32; 1000];
        let s = stat(&gpu, &cpu, 1e-3);
        assert!(s.bad(1e-3));
        assert!((s.rel_l2 - 1.0).abs() < 1e-9);
        assert!(s.extra_zeros() > 10);
    }

    #[test]
    fn flags_unwritten_poison() {
        let cpu = vec![1f32; 8];
        let mut gpu = vec![1f32; 8];
        gpu[5] = f32::NAN;
        let s = stat(&gpu, &cpu, 1e-3);
        assert!(s.bad(1e-3));
        assert_eq!(s.extra_nonfinite(), 1);
        assert_eq!(s.max_idx, 5, "the NaN is the headline");
    }

    #[test]
    fn near_zero_cpu_is_not_infinite_noise() {
        let s = stat(&[0.0, 0.0], &[0.0, 0.0], 1e-3);
        assert_eq!(s.rel_l2, 0.0);
        let s = stat(&[1e-3, 0.0], &[0.0, 0.0], 1e-3);
        assert!(s.rel_l2.is_infinite() && s.bad(1e-3));
    }

    #[test]
    fn length_mismatch_fails() {
        assert!(stat(&[1.0, 2.0], &[1.0], 1e-3).bad(1e-3));
    }
}
