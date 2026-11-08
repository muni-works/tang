//! Row kernels: a traced tang-expr graph run once per row, with strided /
//! broadcast / tiled inputs, per-row outputs and f64 sums over rows
//! ([`tang_expr::ExprGraph::to_cuda_rows`]).
//!
//! `elementwise` covers `out[i] = f(in0[i], in1[i], …)`; a row kernel is
//! for the shapes that does not reach: a loss head reading `k` columns of
//! a row-major activation plus a per-column vector shared by every row,
//! writing several gradients per row, and reducing the loss terms over the
//! batch in double. The graph is traced and compiled once
//! ([`CudaComputeDevice::row_kernel`]); a launch is argument binding only.

use std::cell::RefCell;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use tang_expr::codegen::Real;
use tang_expr::node::ExprId;
use tang_expr::trace;

use super::{CudaBuffer, CudaComputeDevice};

/// A compiled row kernel (see the module docs).
pub struct RowKernel {
    func: CudaFunction,
    n_inputs: usize,
    n_rows_out: usize,
    n_sums: usize,
    sums: RefCell<CudaSlice<f64>>,
    source: String,
}

impl RowKernel {
    /// The generated CUDA source.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Number of f64 sums the kernel produces.
    pub fn n_sums(&self) -> usize {
        self.n_sums
    }
}

/// Where row `r` reads its input: `buf[(period > 0 ? r % period : r) *
/// stride + offset]`.
#[derive(Clone, Copy)]
pub struct RowIn<'a> {
    pub buf: &'a CudaBuffer,
    pub stride: usize,
    pub offset: usize,
    pub period: usize,
}

impl<'a> RowIn<'a> {
    /// Element `r` of `buf`.
    pub fn each(buf: &'a CudaBuffer) -> Self {
        Self {
            buf,
            stride: 1,
            offset: 0,
            period: 0,
        }
    }
    /// Column `col` of a row-major `[rows, width]` matrix.
    pub fn col(buf: &'a CudaBuffer, width: usize, col: usize) -> Self {
        Self {
            buf,
            stride: width,
            offset: col,
            period: 0,
        }
    }
    /// The same element `buf[i]` for every row.
    pub fn bcast(buf: &'a CudaBuffer, i: usize) -> Self {
        Self {
            buf,
            stride: 0,
            offset: i,
            period: 0,
        }
    }
    /// `buf[r % period]`: a vector tiled along flattened rows (a bias over
    /// the columns of a matrix walked element by element).
    pub fn tile(buf: &'a CudaBuffer, period: usize) -> Self {
        Self {
            buf,
            stride: 1,
            offset: 0,
            period,
        }
    }
    fn max_index(&self, rows: usize) -> usize {
        let last = if self.period > 0 {
            (self.period - 1).min(rows.saturating_sub(1))
        } else {
            rows.saturating_sub(1)
        };
        last * self.stride + self.offset
    }
}

/// Where row `r` writes an output: `buf[r * stride + offset]`.
pub struct RowOut<'a> {
    pub buf: &'a mut CudaBuffer,
    pub stride: usize,
    pub offset: usize,
}

impl<'a> RowOut<'a> {
    /// Element `r` of `buf`.
    pub fn each(buf: &'a mut CudaBuffer) -> Self {
        Self {
            buf,
            stride: 1,
            offset: 0,
        }
    }
    /// Column `col` of a row-major `[rows, width]` matrix.
    pub fn col(buf: &'a mut CudaBuffer, width: usize, col: usize) -> Self {
        Self {
            buf,
            stride: width,
            offset: col,
        }
    }
}

impl CudaComputeDevice {
    /// Trace `f` over `n_inputs` variables and compile it as a row kernel.
    /// `f` returns `(row_outputs, sum_outputs)`; it may differentiate
    /// inside the trace (`tang_expr::with_graph(|g| g.vjp(..))`).
    pub fn row_kernel(
        &self,
        n_inputs: usize,
        real: Real,
        f: &dyn Fn(&[ExprId]) -> (Vec<ExprId>, Vec<ExprId>),
    ) -> RowKernel {
        let (graph, (rows_out, sums)) = trace(|| {
            let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
            f(&vars)
        });
        let kernel = graph.to_cuda_rows(&rows_out, &sums, n_inputs, real);
        let (_module, func) = self.get_func(&kernel.source, kernel.entry_point);
        let acc = self
            .stream
            .alloc_zeros::<f64>(sums.len().max(1))
            .expect("row kernel sums");
        RowKernel {
            func,
            n_inputs,
            n_rows_out: rows_out.len(),
            n_sums: sums.len(),
            sums: RefCell::new(acc),
            source: kernel.source,
        }
    }

    /// Run `k` over `rows` rows, in order on this device's stream. The
    /// kernel's sums are zeroed first and stay on the device until
    /// [`CudaComputeDevice::row_sums`].
    pub fn run_rows(&self, k: &RowKernel, rows: usize, ins: &[RowIn], outs: &mut [RowOut]) {
        assert_eq!(ins.len(), k.n_inputs, "row kernel: one RowIn per input");
        assert_eq!(
            outs.len(),
            k.n_rows_out,
            "row kernel: one RowOut per row output"
        );
        if rows == 0 {
            return;
        }
        for (i, a) in ins.iter().enumerate() {
            assert!(
                a.max_index(rows) < a.buf.len,
                "row kernel input {i} reads past its buffer"
            );
        }
        for (j, o) in outs.iter().enumerate() {
            assert!(
                (rows - 1) * o.stride + o.offset < o.buf.len,
                "row kernel output {j} writes past its buffer"
            );
        }
        let mut sums = k.sums.borrow_mut();
        if k.n_sums > 0 {
            self.stream
                .memset_zeros(&mut *sums)
                .expect("row kernel: zero sums");
        }
        let args: Vec<[i64; 3]> = ins
            .iter()
            .map(|a| [a.stride as i64, a.offset as i64, a.period as i64])
            .collect();
        let oargs: Vec<[i64; 2]> = outs
            .iter()
            .map(|o| [o.stride as i64, o.offset as i64])
            .collect();
        let rows_i = rows as i64;
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(256).clamp(1, 8192) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut b = self.stream.launch_builder(&k.func);
            b.arg(&rows_i);
            for (a, x) in ins.iter().zip(&args) {
                b.arg(a.buf.f32_data()).arg(&x[0]).arg(&x[1]).arg(&x[2]);
            }
            for (o, x) in outs.iter_mut().zip(&oargs) {
                b.arg(o.buf.f32_data_mut()).arg(&x[0]).arg(&x[1]);
            }
            b.arg(&mut *sums);
            b.launch(cfg).expect("row kernel launch");
        }
    }

    /// The sums of `k`'s last [`run_rows`](Self::run_rows) (synchronizes).
    pub fn row_sums(&self, k: &RowKernel) -> Vec<f64> {
        let sums = k.sums.borrow();
        let v = self
            .stream
            .clone_dtoh(&*sums)
            .expect("row kernel: read sums");
        self.stream.synchronize().expect("row kernel: sync");
        v[..k.n_sums].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::ComputeDevice;
    use tang::Scalar;

    /// Strided, broadcast and tiled inputs; a column output; f64 sums and
    /// a vjp, against the host.
    #[test]
    fn row_kernel_matches_host() {
        let Ok(Ok(dev)) = std::panic::catch_unwind(CudaComputeDevice::new) else {
            eprintln!("no CUDA device, skipped");
            return;
        };
        let (rows, w) = (10_000usize, 3usize);
        let m: Vec<f32> = (0..rows * w)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let v = [0.5f32, -1.25, 2.0];
        let (mb, vb) = (dev.upload(&m), dev.upload(&v));
        // per row: y = Σ_c m[r,c]·v[c]; loss = exp(y)·m[r,0]; out = dloss/dm[r,1]
        let k = dev.row_kernel(6, Real::F64, &|x| {
            let y = x[0] * x[3] + x[1] * x[4] + x[2] * x[5];
            let loss = y.exp() * x[0];
            let g = tang_expr::with_graph(|gr| gr.vjp(&[loss], &[ExprId::ONE], 6));
            (vec![g[1]], vec![loss, y])
        });
        let mut out = dev.alloc(rows * 2);
        let ins: Vec<RowIn> = (0..w)
            .map(|c| RowIn::col(&mb, w, c))
            .chain((0..w).map(|c| RowIn::bcast(&vb, c)))
            .collect();
        dev.run_rows(&k, rows, &ins, &mut [RowOut::col(&mut out, 2, 1)]);
        let sums = dev.row_sums(&k);
        let got = dev.download(&out);
        let (mut sl, mut sy) = (0.0f64, 0.0f64);
        for r in 0..rows {
            let row = &m[r * w..r * w + w];
            let y: f64 = row.iter().zip(&v).map(|(a, b)| *a as f64 * *b as f64).sum();
            sl += y.exp() * row[0] as f64;
            sy += y;
            let g = (y.exp() * row[0] as f64 * v[1] as f64) as f32;
            assert!(
                (got[r * 2 + 1] - g).abs() <= 1e-6 * g.abs().max(1.0),
                "row {r}: {} vs {g}",
                got[r * 2 + 1]
            );
            assert_eq!(got[r * 2], 0.0);
        }
        assert!(
            (sums[0] - sl).abs() <= 1e-9 * sl.abs().max(1.0),
            "{} vs {sl}",
            sums[0]
        );
        assert!(
            (sums[1] - sy).abs() <= 1e-9 * sy.abs().max(1.0),
            "{} vs {sy}",
            sums[1]
        );
        // tiled: out[i] = m[i] + v[i % 3] over the flattened matrix
        let kt = dev.row_kernel(2, Real::F32, &|x| (vec![x[0] + x[1]], vec![]));
        let mut o2 = dev.alloc(rows * w);
        dev.run_rows(
            &kt,
            rows * w,
            &[RowIn::each(&mb), RowIn::tile(&vb, w)],
            &mut [RowOut::each(&mut o2)],
        );
        let o2 = dev.download(&o2);
        for i in 0..rows * w {
            assert_eq!(o2[i], m[i] + v[i % w]);
        }
    }
}
