//! CPU runtime for tensor-IR programs: [`TensorGraph::to_c`] compiled by
//! the system C compiler into a shared library at load time and
//! `dlopen`ed. Buffers are the library's static arrays (allocated once),
//! so a step is one call to `tang_run` with no allocation and no copies:
//! inputs are written in place, outputs read in place.
//!
//! The compiler is `$TANG_CC` (default `cc`), with `-O3`, the host's own
//! CPU (`-mcpu=native` on aarch64, `-march=native` elsewhere),
//! `-ffp-contract=fast` and `-fno-strict-aliasing`; `$TANG_CFLAGS`
//! replaces those flags. For a target without a compiler at run time (or a
//! robot without a Rust toolchain), build the same source ahead of time:
//! `CProgram::source` is one self-contained C file.
use std::path::{Path, PathBuf};
use std::process::Command;

use tang_compile::tensor::c::CProgram;
use tang_compile::tensor::{Format, TensorGraph, TensorId};

use crate::program::{pack_f16, unpack_f16};

type RunFn = unsafe extern "C" fn();
type PtrFn = unsafe extern "C" fn(i32) -> *mut core::ffi::c_void;

/// A program compiled for the host CPU.
pub struct CpuProgram {
    pub program: CProgram,
    lib: libloading::Library,
    run: RunFn,
    input: PtrFn,
    output: PtrFn,
    dir: PathBuf,
}

/// Default compiler flags for the host.
pub fn default_cflags() -> Vec<String> {
    let cpu = if cfg!(target_arch = "aarch64") {
        "-mcpu=native"
    } else {
        "-march=native"
    };
    ["-O3", cpu, "-ffp-contract=fast", "-fno-strict-aliasing"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

impl CpuProgram {
    /// Lower `outputs` with [`TensorGraph::to_c`], compile, load, and
    /// write `inputs` (graph input order, f32; converted to each input's
    /// storage format).
    pub fn compile(
        g: &TensorGraph,
        outputs: &[TensorId],
        inputs: &[&[f32]],
    ) -> Result<Self, String> {
        let program = g.to_c(outputs);
        let p = Self::load(program)?;
        for (k, x) in inputs.iter().enumerate() {
            p.write_input(k, x);
        }
        Ok(p)
    }

    /// Compile and load an already-lowered program.
    pub fn load(program: CProgram) -> Result<Self, String> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        program.source.hash(&mut h);
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tang-cpu-{}-{}-{:x}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            h.finish()
        ));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let src = dir.join("prog.c");
        std::fs::write(&src, &program.source).map_err(|e| e.to_string())?;
        let lib_path = dir.join(if cfg!(target_os = "macos") {
            "prog.dylib"
        } else {
            "prog.so"
        });
        compile_shared(&src, &lib_path)?;
        // SAFETY: the library is the unit we just generated; its symbols
        // have the signatures declared above.
        unsafe {
            let lib = libloading::Library::new(&lib_path).map_err(|e| e.to_string())?;
            let run = *lib.get::<RunFn>(b"tang_run\0").map_err(|e| e.to_string())?;
            let input = *lib
                .get::<PtrFn>(b"tang_input\0")
                .map_err(|e| e.to_string())?;
            let output = *lib
                .get::<PtrFn>(b"tang_output\0")
                .map_err(|e| e.to_string())?;
            Ok(Self {
                program,
                lib,
                run,
                input,
                output,
                dir,
            })
        }
    }

    fn input_spec(&self, k: usize) -> (usize, Format) {
        let b = self
            .program
            .program
            .buffers
            .iter()
            .find(|b| b.input == Some(k))
            .expect("graph input has a buffer");
        (b.len, b.format)
    }

    /// Overwrite graph input `k`.
    pub fn write_input(&self, k: usize, data: &[f32]) {
        let (len, format) = self.input_spec(k);
        assert_eq!(data.len(), len, "input {k} length");
        // SAFETY: `tang_input(k)` points at a static array of `len`
        // elements of `format`.
        unsafe {
            let p = (self.input)(k as i32);
            match format {
                Format::F32 => std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut f32, len),
                Format::F16 => {
                    let w = pack_f16(data);
                    std::ptr::copy_nonoverlapping(w.as_ptr() as *const u8, p as *mut u8, 2 * len)
                }
                Format::BF16 => {
                    let h: Vec<u16> = data.iter().map(|x| bf16(*x)).collect();
                    std::ptr::copy_nonoverlapping(h.as_ptr(), p as *mut u16, len)
                }
                _ => panic!("fp8 storage lowers only on sm90 CUDA"),
            }
        }
    }

    /// One step: every kernel in order.
    pub fn run(&self) {
        // SAFETY: `tang_run` touches only the library's own buffers.
        unsafe { (self.run)() }
    }

    /// Output `i` (in `outputs` order) as f32.
    pub fn read_output(&self, i: usize) -> Vec<f32> {
        let b = &self.program.program.buffers[self.program.outputs[i]];
        // SAFETY: as for `write_input`.
        unsafe {
            let p = (self.output)(i as i32);
            match b.format {
                Format::F32 => std::slice::from_raw_parts(p as *const f32, b.len).to_vec(),
                Format::F16 => {
                    let words = std::slice::from_raw_parts(p as *const f32, b.len.div_ceil(2));
                    unpack_f16(words, b.len)
                }
                Format::BF16 => std::slice::from_raw_parts(p as *const u16, b.len)
                    .iter()
                    .map(|h| f32::from_bits((*h as u32) << 16))
                    .collect(),
                _ => panic!("fp8 storage lowers only on sm90 CUDA"),
            }
        }
    }

    /// Mean seconds per [`run`](Self::run) over `iters` after one warm-up.
    pub fn time(&self, iters: usize) -> f64 {
        self.run();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            self.run();
        }
        t.elapsed().as_secs_f64() / iters.max(1) as f64
    }
}

impl Drop for CpuProgram {
    fn drop(&mut self) {
        let _ = &self.lib;
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn compile_shared(src: &Path, out: &Path) -> Result<(), String> {
    let cc = std::env::var("TANG_CC").unwrap_or_else(|_| "cc".into());
    let flags: Vec<String> = match std::env::var("TANG_CFLAGS") {
        Ok(f) => f.split_whitespace().map(str::to_string).collect(),
        Err(_) => default_cflags(),
    };
    let o = Command::new(&cc)
        .args(&flags)
        .args(["-shared", "-fPIC", "-o"])
        .arg(out)
        .arg(src)
        .arg("-lm")
        .output()
        .map_err(|e| format!("{cc}: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "{cc} failed:\n{}",
            String::from_utf8_lossy(&o.stderr)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang::Scalar;
    use tang_compile::tensor::{Format, Monoid, Precision};
    use tang_expr::ExprId;

    fn ramp(n: usize, s: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7919) % 97) as f32 / 97.0 - 0.5) * s)
            .collect()
    }

    fn check(g: &TensorGraph, outs: &[TensorId], inputs: &[Vec<f32>], tol: f64) {
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let p = match CpuProgram::compile(g, outs, &refs) {
            Ok(p) => p,
            Err(e) if e.contains("No such file") => return, // no compiler
            Err(e) => panic!("{e}"),
        };
        p.run();
        let f64_in: Vec<Vec<f64>> = inputs
            .iter()
            .map(|v| v.iter().map(|x| *x as f64).collect())
            .collect();
        let want = g.eval(&f64_in, outs);
        for (i, w) in want.iter().enumerate() {
            let got = p.read_output(i);
            let scale = w.iter().fold(1e-30f64, |m, x| m.max(x.abs()));
            let err = got
                .iter()
                .zip(w)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0, f64::max);
            assert!(err / scale < tol, "output {i}: rel err {}", err / scale);
        }
    }

    /// GRU step (as `tang_bench::ir_control::gru_step`), small.
    fn gru(batch: usize, obs: usize, hidden: usize) -> (TensorGraph, Vec<TensorId>) {
        let mut g = TensorGraph::new();
        let x = g.input(&[batch, obs]);
        let h = g.input(&[batch, hidden]);
        let gate = |g: &mut TensorGraph| {
            let w_i = g.input(&[obs, hidden]);
            let w_h = g.input(&[hidden, hidden]);
            let b_i = g.input(&[1, hidden]);
            let gi = g.matmul(x, w_i).unwrap();
            let gh = g.matmul(h, w_h).unwrap();
            let b_i = g.broadcast(b_i, &[batch, hidden]).unwrap();
            let gi = g.map(&[gi, b_i], |v| v[0] + v[1]).unwrap();
            (gi, gh)
        };
        let (ir, hr) = gate(&mut g);
        let (iz, hz) = gate(&mut g);
        let (inn, hn) = gate(&mut g);
        let one = || ExprId::from_f64(1.0);
        let sig = |e: ExprId| one() / (one() + (-e).exp());
        let y = g
            .map(&[ir, hr, iz, hz, inn, hn, h], |v| {
                let r = sig(v[0] + v[1]);
                let z = sig(v[2] + v[3]);
                let n = (v[4] + r * v[5]).tanh();
                (one() - z) * n + z * v[6]
            })
            .unwrap();
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.eliminate_common(&outs);
        g.schedule_auto(&outs);
        (g, outs)
    }

    fn inputs_for(g: &TensorGraph, outs: &[TensorId]) -> Vec<Vec<f32>> {
        let p = g.to_c(outs);
        let n = p
            .program
            .buffers
            .iter()
            .filter(|b| b.input.is_some())
            .count();
        (0..n)
            .map(|k| {
                let len = p
                    .program
                    .buffers
                    .iter()
                    .find(|b| b.input == Some(k))
                    .unwrap()
                    .len;
                ramp(len, 0.5)
            })
            .collect()
    }

    #[test]
    fn gru_matches_eval() {
        for batch in [1, 8] {
            let (g, outs) = gru(batch, 12, 20);
            let p = g.to_c(&outs);
            assert_eq!(p.program.kernels.len(), 1, "GRU fuses into one kernel");
            check(&g, &outs, &inputs_for(&g, &outs), 1e-5);
        }
    }

    /// `x · Wᵀ` with `W [out, in]`: no point dim is contiguous, so the
    /// kernel runs in lane mode (partial sums along the reduce dim).
    #[test]
    fn transposed_weight_matmul_matches_eval() {
        let (m, k, n) = (8, 24, 12);
        let mut g = TensorGraph::new();
        let x = g.input(&[m, k]);
        let w = g.input(&[n, k]);
        let wt = g.transpose(w, &[1, 0]).unwrap();
        let y = g.matmul(x, wt).unwrap();
        let z = g.map(&[y], |v| v[0].tanh()).unwrap();
        let (g, outs) = g.substitute_maps(&[z]);
        assert!(g.to_c(&outs).source.contains("accl0"), "lane mode");
        check(&g, &outs, &[ramp(m * k, 1.0), ramp(n * k, 1.0)], 1e-5);
    }

    #[test]
    fn softmax_and_f16_storage_match_eval() {
        let (n, d) = (6, 10);
        let mut g = TensorGraph::new();
        let x = g.input(&[n, d]);
        let m = g.reduce(x, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[n, d]).unwrap();
        let e = g.map(&[x, mb], |v| (v[0] - v[1]).exp()).unwrap();
        let s = g.reduce(e, 1, Monoid::sum()).unwrap();
        let sb = g.broadcast(s, &[n, d]).unwrap();
        let y = g.map(&[e, sb], |v| v[0] / v[1]).unwrap();
        let t = g.transpose(y, &[1, 0]).unwrap();
        let (g, outs) = g.substitute_maps(&[t]);
        let (g2, outs2) = g.stabilize_softmax(&outs);
        let inputs = vec![ramp(n * d, 4.0)];
        check(&g, &outs, &inputs, 1e-5);
        check(&g2, &outs2, &inputs, 1e-5);
        let mut h = g.clone();
        h.set_precision(TensorId::from_index(0), Precision::storage(Format::F16));
        check(&h, &outs, &inputs, 5e-3);
    }
}
