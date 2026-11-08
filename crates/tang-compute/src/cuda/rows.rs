//! Row kernels: a traced tang-expr graph run once per row, with strided /
//! broadcast / tiled inputs, per-row outputs and f64 sums over rows
//! ([`tang_expr::ExprGraph::to_cuda_rows`]).
//!
//! `elementwise` covers `out[i] = f(in0[i], in1[i], …)`; a row kernel is
//! for the shapes that does not reach: a loss head reading `k` columns of
//! a row-major activation plus a per-column vector shared by every row,
//! writing several gradients per row, and reducing the loss terms over the
//! batch in double; or a bias + activation epilogue (`tile`). The graph and
//! its layout are traced and compiled once ([`CudaComputeDevice::row_kernel`]);
//! a launch binds buffers only.

use std::cell::RefCell;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
pub use tang_expr::codegen::{Real, RowLoad, RowStore};
use tang_expr::node::ExprId;
use tang_expr::trace;

use super::{CudaBuffer, CudaComputeDevice};

/// A compiled row kernel (see the module docs).
pub struct RowKernel {
    func: CudaFunction,
    loads: Vec<RowLoad>,
    stores: Vec<RowStore>,
    n_in: usize,
    n_out: usize,
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

impl CudaComputeDevice {
    /// Trace `f` over one variable per `loads` entry and compile it as a row
    /// kernel. `f` returns `(row_outputs, sum_outputs)`, row output `j`
    /// stored per `stores[j]`; it may differentiate inside the trace
    /// (`tang_expr::with_graph(|g| g.vjp(..))`).
    pub fn row_kernel(
        &self,
        loads: &[RowLoad],
        stores: &[RowStore],
        real: Real,
        f: &dyn Fn(&[ExprId]) -> (Vec<ExprId>, Vec<ExprId>),
    ) -> RowKernel {
        let n = loads.len();
        let (graph, (rows_out, sums)) = trace(|| {
            let vars: Vec<ExprId> = (0..n as u16).map(ExprId::var).collect();
            f(&vars)
        });
        assert_eq!(
            rows_out.len(),
            stores.len(),
            "row kernel: one RowStore per row output"
        );
        let outs: Vec<(ExprId, RowStore)> =
            rows_out.into_iter().zip(stores.iter().copied()).collect();
        let kernel = graph.to_cuda_rows(loads, &outs, &sums, real);
        let (_module, func) = self.get_func(&kernel.source, kernel.entry_point);
        let acc = self
            .stream
            .alloc_zeros::<f64>(sums.len().max(1))
            .expect("row kernel sums");
        RowKernel {
            func,
            loads: loads.to_vec(),
            stores: stores.to_vec(),
            n_in: loads.iter().map(|l| l.buf + 1).max().unwrap_or(0),
            n_out: stores.iter().map(|s| s.buf + 1).max().unwrap_or(0),
            n_sums: sums.len(),
            sums: RefCell::new(acc),
            source: kernel.source,
        }
    }

    /// Run `k` over `rows` rows on this device's stream (`ins`/`outs` are
    /// the buffers the layout's `buf` indices name). Sums are zeroed first
    /// and stay on the device until [`row_sums`](Self::row_sums).
    pub fn run_rows(
        &self,
        k: &RowKernel,
        rows: usize,
        ins: &[&CudaBuffer],
        outs: &mut [&mut CudaBuffer],
    ) {
        assert_eq!(ins.len(), k.n_in, "row kernel: input buffer count");
        assert_eq!(outs.len(), k.n_out, "row kernel: output buffer count");
        if rows == 0 {
            return;
        }
        for (i, l) in k.loads.iter().enumerate() {
            assert!(
                l.max_index(rows) < ins[l.buf].len,
                "row kernel load {i} reads past buffer {}",
                l.buf
            );
        }
        for (j, s) in k.stores.iter().enumerate() {
            assert!(
                (rows - 1) * s.stride + s.offset < outs[s.buf].len,
                "row kernel store {j} writes past buffer {}",
                s.buf
            );
        }
        let mut sums = k.sums.borrow_mut();
        if k.n_sums > 0 {
            self.stream
                .memset_zeros(&mut *sums)
                .expect("row kernel: zero sums");
        }
        let rows_i = rows as i64;
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(256).clamp(1, 8192) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut b = self.stream.launch_builder(&k.func);
            b.arg(&rows_i);
            for buf in ins {
                b.arg(buf.f32_data());
            }
            for buf in outs.iter_mut() {
                b.arg(buf.f32_data_mut());
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
        let loads: Vec<RowLoad> = (0..w)
            .map(|c| RowLoad::col(0, w, c))
            .chain((0..w).map(|c| RowLoad::bcast(1, c)))
            .collect();
        let k = dev.row_kernel(&loads, &[RowStore::col(0, 2, 1)], Real::F64, &|x| {
            let y = x[0] * x[3] + x[1] * x[4] + x[2] * x[5];
            let loss = y.exp() * x[0];
            let g = tang_expr::with_graph(|gr| gr.vjp(&[loss], &[ExprId::ONE], 6));
            (vec![g[1]], vec![loss, y])
        });
        let mut out = dev.alloc(rows * 2);
        dev.run_rows(&k, rows, &[&mb, &vb], &mut [&mut out]);
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
        let kt = dev.row_kernel(
            &[RowLoad::each(0), RowLoad::tile(1, w)],
            &[RowStore::each(0)],
            Real::F32,
            &|x| (vec![x[0] + x[1]], vec![]),
        );
        let mut o2 = dev.alloc(rows * w);
        dev.run_rows(&kt, rows * w, &[&mb, &vb], &mut [&mut o2]);
        let o2 = dev.download(&o2);
        for i in 0..rows * w {
            assert_eq!(o2[i], m[i] + v[i % w]);
        }
    }
}
