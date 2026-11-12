//! A vendor GEMM (`TANG_VENDOR_GEMM`, `tang_gpu::vendor`: dequant to f16 +
//! cuBLASLt) against tang's own int8 MMA tile and an f64 product, on GGUF
//! block weights at prefill shapes. Per case: both programs' GEMM kernel
//! time (best of 20) and the relative error of each against f64 over the
//! dequantized weight and the exact f32 activation.
//!
//!     VG_CASES=rowsxkxnpos,... VG_FMT=Q4K|Q5K|Q8_0 \
//!       cargo run --release -p tang-bench --features cuda --example vendor_gemm_check
use tang_compile::tensor::{gguf, Format, Precision, TensorGraph};
use tang_gpu::program::{Program, Runtime};

/// Random block bytes for `n` elements of `f` with sane f16 scales.
fn random_blocks(f: Format, n: usize, rng: &mut tang_bench::Rng) -> Vec<u8> {
    let (per, bytes) = gguf::block(f).expect("a block format");
    let mut out = vec![0u8; n / per * bytes];
    for b in out.chunks_mut(bytes) {
        for x in b.iter_mut() {
            *x = (rng.f32() * 256.0) as u8;
        }
        // d (and dmin): small positive f16 (exponent 2^-10 .. 2^-7)
        let h = |rng: &mut tang_bench::Rng| -> [u8; 2] {
            let e = 5 + (rng.f32() * 3.0) as u16;
            let m = (rng.f32() * 1024.0) as u16;
            ((e << 10) | m).to_le_bytes()
        };
        let d = h(rng);
        b[..2].copy_from_slice(&d);
        if f != Format::Q8_0 {
            let dm = h(rng);
            b[2..4].copy_from_slice(&dm);
        }
    }
    out
}

fn main() {
    let rt = Runtime::select().expect("gpu");
    let fmt = match std::env::var("VG_FMT").as_deref() {
        Ok("Q5K") => Format::Q5K,
        Ok("Q8_0") => Format::Q8_0,
        _ => Format::Q4K,
    };
    let cases: Vec<(usize, usize, usize)> = std::env::var("VG_CASES")
        .unwrap_or("2048x2048x256,8192x2048x256,2048x4096x256,8192x2048x2048,2048x4096x4096,12288x2048x4096".into())
        .split(',')
        .map(|c| {
            let d: Vec<usize> = c.split('x').map(|x| x.parse().unwrap()).collect();
            (d[0], d[1], d[2])
        })
        .collect();
    let mut rng = tang_bench::Rng::new(7);
    println!("fmt {fmt:?} on {}", rt.name());
    for (rows, k, npos) in cases {
        let wb = random_blocks(fmt, rows * k, &mut rng);
        let mut wf = vec![0f32; rows * k];
        gguf::dequantize(fmt, &wb, &mut wf).expect("dequantize");
        let words: Vec<f32> = wb
            .chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                f32::from_le_bytes(w)
            })
            .collect();
        let xv: Vec<f32> = (0..npos * k).map(|_| rng.f32() - 0.5).collect();

        let build = || {
            let mut g = TensorGraph::new();
            let w = g.input(&[rows, k]);
            let x = g.input(&[npos, k]);
            let wt = g.transpose(w, &[1, 0]).unwrap();
            let y = g.matmul(x, wt).unwrap();
            let (mut g, outs) = g.substitute_maps(&[y]);
            g.set_precision(w, Precision::storage(fmt));
            let (g, outs, s) = g.schedule_block_weights(&outs);
            assert_eq!(s.gemvs, 1, "the contraction is not on the int8 path");
            let mut keep = outs.clone();
            keep.extend(&s.copies);
            let (g, keep) = g.substitute_maps(&keep);
            (g, keep[0])
        };
        // f64 reference on 128 sampled entries
        let samples: Vec<(usize, usize)> = (0..128usize)
            .map(|s| ((s * 7919) % npos, (s * 104729) % rows))
            .collect();
        let want: Vec<f64> = samples
            .iter()
            .map(|&(p, r)| {
                (0..k)
                    .map(|j| wf[r * k + j] as f64 * xv[p * k + j] as f64)
                    .sum()
            })
            .collect();
        let scale = want.iter().fold(0f64, |a, w| a.max(w.abs()));
        let mut line = format!("{rows}x{k} npos {npos}:");
        let mut outs: Vec<Vec<f32>> = Vec::new();
        for vendor in [false, true] {
            // the lowering reads TANG_VENDOR_GEMM
            if vendor {
                std::env::set_var("TANG_VENDOR_GEMM", "1");
            } else {
                std::env::remove_var("TANG_VENDOR_GEMM");
            }
            let (g, y) = build();
            let prog = g.to_cuda(&[y]);
            let gi = prog
                .kernels
                .iter()
                .position(|kk| kk.node == y)
                .expect("the GEMM kernel");
            let is_vendor = prog.kernels[gi]
                .source
                .starts_with(tang_compile::tensor::VENDOR_TAG);
            assert_eq!(is_vendor, vendor, "vendor lowering not taken");
            let p = Program::new(&rt, prog, &[&words, &xv]);
            p.run(&rt);
            let got = p.read(&rt, y);
            let err = samples
                .iter()
                .zip(&want)
                .map(|(&(pp, r), w)| (got[pp * rows + r] as f64 - w).abs())
                .fold(0f64, f64::max)
                / scale;
            let t = p.time_kernel(&rt, gi, 20);
            let tflops = 2.0 * (rows * k * npos) as f64 / t / 1e9;
            line.push_str(&format!(
                "  {} {t:.4} ms {tflops:.1} TFLOP/s rel_err {err:.2e}",
                if vendor { "vendor" } else { "tang" }
            ));
            outs.push(got);
        }
        let diff = outs[0]
            .iter()
            .zip(&outs[1])
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0f64, f64::max)
            / scale;
        println!("{line}  |tang-vendor| {diff:.2e}");
    }
}
