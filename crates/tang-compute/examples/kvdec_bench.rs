//! TKV / KIVI decode on Metal or CUDA: bit-identity vs the CPU decoder and throughput.
//!
//!     cargo run --release -p tang-compute --features metal --example kvdec_bench -- \
//!         <x.f32> <heads>x<T>x<hd> <k|v> <rel> [--copies N] [--block B] [--lane W] [--reps R]
//!     cargo run --release -p tang-compute --features cuda --example kvdec_bench -- <...>
//!
//! `--copies N` decodes N copies of the stream in one dispatch (≈ N layer×kind streams of a
//! model's cache) to measure whole-cache decode time.
use cram_codec::kvcodec::{encode, Kind, KvParams, KvStream};
use cram_codec::kvgpu::{Batch, Placement};
use std::time::Instant;
#[cfg(feature = "metal")]
use tang_compute::MetalDevice as Device;
#[cfg(all(feature = "cuda", not(feature = "metal")))]
use tang_compute::CudaComputeDevice as Device;
use tang_compute::ComputeDevice;

fn flag(a: &[String], f: &str) -> Option<String> {
    a.iter().position(|x| x == f).map(|i| a[i + 1].clone())
}

/// New device: Metal when that feature is on (the Mac), else CUDA. With NEITHER
/// feature, fall back to the CPU device so the example still configures (the
/// CPU path has no tkv_decode and the run fails loudly there).
#[cfg(feature = "metal")]
fn new_device() -> Device {
    MetalDevice::new().expect("metal")
}
#[cfg(feature = "metal")]
use tang_compute::MetalDevice;
#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn new_device() -> Device {
    CudaComputeDevice::new().expect("no CUDA GPU")
}
#[cfg(all(feature = "cuda", not(feature = "metal")))]
use tang_compute::CudaComputeDevice;
#[cfg(not(any(feature = "metal", feature = "cuda")))]
fn new_device() -> Device {
    tang_compute::CpuDevice::new()
}
#[cfg(not(any(feature = "metal", feature = "cuda")))]
use tang_compute::CpuDevice as Device;


fn main() {
    let a: Vec<String> = std::env::args().collect();
    let x: Vec<f32> = std::fs::read(&a[1])
        .unwrap()
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let sh: Vec<usize> = a[2].split('x').map(|v| v.parse().unwrap()).collect();
    let (h, n, d) = (sh[0], sh[1], sh[2]);
    let kind = if a[3] == "k" { Kind::K } else { Kind::V };
    let rel: f64 = a[4].parse().unwrap();
    let copies: usize = flag(&a, "--copies").map_or(1, |v| v.parse().unwrap());
    let reps: usize = flag(&a, "--reps").map_or(10, |v| v.parse().unwrap());
    // step from the centered rms (as bench/kv_base.py)
    let mut sse = 0.0;
    for hh in 0..h {
        for c in 0..d {
            let m: f64 = (0..n).map(|t| x[(hh * n + t) * d + c] as f64).sum::<f64>() / n as f64;
            sse += (0..n)
                .map(|t| (x[(hh * n + t) * d + c] as f64 - m).powi(2))
                .sum::<f64>();
        }
    }
    let mut p = KvParams::new(
        kind,
        h,
        d,
        rel * (sse / x.len() as f64).sqrt() * 12f64.sqrt(),
    );
    p.pre_rope = false;
    p.gpu = true;
    if let Some(w) = flag(&a, "--lane") {
        p.lane_ch = w.parse().unwrap();
    }
    if let Some(b) = flag(&a, "--block") {
        p.block = b.parse().unwrap();
    }
    let bytes = encode(p, &x, n);
    let s = KvStream::parse(&bytes).unwrap();
    let t = Instant::now();
    let cpu = s.decode_range(0, n);
    let cpu_s = t.elapsed().as_secs_f64();
    let mut b = Batch::new();
    for i in 0..copies {
        let row = move |hh: usize, tt: usize| ((i * h + hh) * n + tt) * d;
        b.add(
            &bytes,
            &Placement {
                row: &row,
                rope_theta: None,
                decode_only: false,
            },
            0,
            n,
        )
        .unwrap();
    }
    let dev = new_device();
    let up = |v: &[u32]| dev.upload_u32(v);
    let (sb, lb, cb, pb) = (
        up(&b.stream),
        up(&b.lanes),
        up(&b.chans),
        up(&Batch::prior()),
    );
    let scr = dev.alloc(b.scratch_words.max(1));
    let out = dev.alloc(copies * x.len());
    dev.tkv_decode(&sb, &lb, &cb, &pb, &scr, &out, None, b.n_lanes());
    let g = dev.download(&out);
    let same = (0..copies).all(|i| {
        g[i * x.len()..(i + 1) * x.len()]
            .iter()
            .zip(&cpu)
            .all(|(u, v)| u.to_bits() == v.to_bits())
    });
    let t = Instant::now();
    for _ in 0..reps {
        dev.tkv_decode(&sb, &lb, &cb, &pb, &scr, &out, None, b.n_lanes());
    }
    dev.sync();
    let gpu_s = t.elapsed().as_secs_f64() / reps as f64;
    let vals = (copies * x.len()) as f64;
    println!(
        "{{\"bpv\":{:.4},\"lanes\":{},\"bit_identical\":{same},\"cpu_ns_per_value\":{:.2},\"gpu_ms\":{:.3},\"gpu_gvalues_s\":{:.3},\"scratch_mb\":{:.1}}}",
        8.0 * bytes.len() as f64 / x.len() as f64,
        b.n_lanes(),
        1e9 * cpu_s / x.len() as f64,
        1e3 * gpu_s,
        vals / gpu_s / 1e9,
        b.scratch_words as f64 * 4.0 / 1e6
    );
}
