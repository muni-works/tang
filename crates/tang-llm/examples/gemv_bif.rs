//! Bytes in flight for decode-shaped GEMVs: what a weight stream can reach on this GPU.
//!
//! 1. `stream`: a pure read of a large buffer (uint4 loads, U unrolled a thread, grid-stride):
//!    the practical ceiling, with no math.
//! 2. `derived`: the derived Q8_0 GEMV (`derived_ops::gemv_q8_0`, GGUF block layout: 34-byte
//!    blocks, halfword loads).
//! 3. `repacked`: the same GEMV on a repacked layout (codes `[rows][k]` 16-byte aligned, f16
//!    scales `[rows][k/32]` apart). A warp takes R rows, a lane four uint4 code loads a row,
//!    all issued before any math (4·R·16 bytes in flight a lane); x staged once in shared memory.
//!
//! Prints GB/s and the fraction of the given peak (`PEAK_GBS`, default 4800 for H200).
//!
//! `cargo run --release -p tang-llm --features cuda --example gemv_bif [rows...]`

use tang_llm::flash::derived_ops::{self as dop, Op};
use tang_moe::gpu::{self, DevBuf, Event, Gpu, Stream};
#[allow(unused_imports)]
use tang_moe::gpu::Module;

const SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void stream_u1(const uint4* __restrict__ p, unsigned long long n, unsigned* out) {
    unsigned acc = 0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        const uint4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}
extern "C" __global__ void stream_u4(const uint4* __restrict__ p, unsigned long long n, unsigned* out) {
    unsigned acc = 0;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i + 3 * stride < n; i += 4 * stride) {
        uint4 v[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) v[u] = __ldg(p + i + u * stride);
        #pragma unroll
        for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}
// repacked Q8 GEMV, k = 2048 (128 uint4 a row: 4 a lane), R rows a warp
__device__ __forceinline__ void l2_prefetch(const unsigned char* pf, unsigned long long pf_bytes) {
#if __CUDA_ARCH__ >= 900
    // this block's share of [pf, pf + pf_bytes), 4 KiB a thread a step, bulk prefetch into L2
    if (pf == nullptr || pf_bytes == 0) return;
    const unsigned long long share = (pf_bytes + gridDim.x - 1) / gridDim.x;
    const unsigned long long b0 = blockIdx.x * share, b1 = min(pf_bytes, b0 + share);
    for (unsigned long long o = b0 + threadIdx.x * 4096ull; o < b1; o += blockDim.x * 4096ull) {
        const unsigned n = (unsigned)min(4096ull, b1 - o) & ~15u;
        if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pf + o), "r"(n) : "memory");
    }
#endif
}
template <int R>
__device__ void gq8r(const uint4* __restrict__ codes, const __half* __restrict__ sc,
                     const signed char* __restrict__ xq, const float* __restrict__ xs,
                     float* __restrict__ out, int rows, const unsigned char* pf, unsigned long long pf_bytes) {
    l2_prefetch(pf, pf_bytes);
    __shared__ __align__(16) signed char sx[2048];
    __shared__ float ss[64];
    for (int i = threadIdx.x; i < 2048 / 16; i += blockDim.x) ((uint4*)sx)[i] = ((const uint4*)xq)[i];
    for (int i = threadIdx.x; i < 64; i += blockDim.x) ss[i] = xs[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * R;
    if (r0 >= rows) return;
    uint4 w[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const int row = min(r0 + r, rows - 1);
        #pragma unroll
        for (int j = 0; j < 4; j++) w[r][j] = __ldg(codes + (size_t)row * 128 + lane + 32 * j);
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const int row = min(r0 + r, rows - 1);
        float acc = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int c = lane + 32 * j;            // uint4 index: columns 16c .. 16c+15
            const int4 xv = ((const int4*)sx)[c];
            int d = 0;
            d = __dp4a((int)w[r][j].x, xv.x, d); d = __dp4a((int)w[r][j].y, xv.y, d);
            d = __dp4a((int)w[r][j].z, xv.z, d); d = __dp4a((int)w[r][j].w, xv.w, d);
            const int b = c >> 1;                   // 32-column block
            acc = fmaf(__half2float(sc[(size_t)row * 64 + b]) * ss[b], (float)d, acc);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0 && r0 + r < rows) out[r0 + r] = acc;
    }
}
extern "C" __global__ void __launch_bounds__(256) gq8r1(const uint4* c, const __half* s, const signed char* x, const float* xs, float* o, int rows, const unsigned char* pf, unsigned long long pfb) { gq8r<1>(c, s, x, xs, o, rows, pf, pfb); }
extern "C" __global__ void __launch_bounds__(256) gq8r2(const uint4* c, const __half* s, const signed char* x, const float* xs, float* o, int rows, const unsigned char* pf, unsigned long long pfb) { gq8r<2>(c, s, x, xs, o, rows, pf, pfb); }
extern "C" __global__ void __launch_bounds__(256) gq8r4(const uint4* c, const __half* s, const signed char* x, const float* xs, float* o, int rows, const unsigned char* pf, unsigned long long pfb) { gq8r<4>(c, s, x, xs, o, rows, pf, pfb); }
"#;

fn best_us(s: &Stream, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    s.sync().unwrap();
    let mut best = f64::MAX;
    for _ in 0..20 {
        let (a, b) = (Event::new(true).unwrap(), Event::new(true).unwrap());
        a.record(s).unwrap();
        for _ in 0..10 {
            f();
        }
        b.record(s).unwrap();
        b.sync().unwrap();
        best = best.min(b.since(&a).unwrap() as f64 * 100.0);
    }
    best
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() {
    let peak: f64 = std::env::var("PEAK_GBS").ok().and_then(|v| v.parse().ok()).unwrap_or(4800.0);
    let rows_list: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let rows_list = if rows_list.is_empty() { vec![4096, 20480, 151936] } else { rows_list };
    let gpu = Gpu::new().expect("cuda");
    let s = Stream::new().unwrap();
    let m = gpu.module(SRC).expect("nvrtc");
    let sms = 132u32.max(1);
    let line = |name: &str, bytes: f64, us: f64| {
        let gbs = bytes / us / 1e3;
        println!("| {name} | {:.1} | {us:.1} | {gbs:.0} | {:.2} |", bytes / 1e6, gbs / peak);
    };
    println!("| kernel | MB | µs | GB/s | fraction of {peak:.0} GB/s |");
    println!("|---|---|---|---|---|");

    // 1. the streaming ceiling
    let n16: usize = (512usize << 20) / 16;
    let buf = DevBuf::zeroed(n16 * 16).unwrap();
    let sink = DevBuf::zeroed(16).unwrap();
    for (name, blocks) in [("stream_u1", sms * 8), ("stream_u4", sms * 8), ("stream_u4", sms * 16)] {
        let f = m.func(name).unwrap();
        let (mut p, mut n, mut o) = (buf.ptr, n16 as u64, sink.ptr);
        let us = best_us(&s, || unsafe {
            let mut a = [&mut p as *mut u64 as *mut _, &mut n as *mut u64 as *mut _, &mut o as *mut u64 as *mut _];
            gpu::launch(f, (blocks, 1, 1), (256, 1, 1), 0, &s, &mut a).unwrap();
        });
        line(&format!("{name} ×{blocks} blocks (512 MB)"), (n16 * 16) as f64, us);
    }

    // 2./3. the GEMVs, k = 2048
    let k = 2048usize;
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let xf: Vec<f32> = (0..k).map(|_| (rng.next() % 2001) as f32 / 1000.0 - 1.0).collect();
    // x as Q8x32: codes then per-32 f32 scale (the repacked kernel's operand)
    let mut xq = vec![0i8; k];
    let mut xs = vec![0f32; k / 32];
    for b in 0..k / 32 {
        let amax = xf[32 * b..32 * b + 32].iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        xs[b] = d;
        for i in 0..32 {
            xq[32 * b + i] = if d == 0.0 { 0 } else { (xf[32 * b + i] / d).round() as i8 };
        }
    }
    for &rows in &rows_list {
        // GGUF Q8_0 blocks: f16 d then 32 codes
        let mut gg = Vec::with_capacity(rows * k / 32 * 34);
        let mut codes = vec![0i8; rows * k];
        let mut sc = vec![0u16; rows * k / 32];
        for r in 0..rows {
            for b in 0..k / 32 {
                let d = 0.001 + (rng.next() % 1000) as f32 * 1e-5;
                let h = half_bits(d);
                gg.extend_from_slice(&h.to_le_bytes());
                sc[r * k / 32 + b] = h;
                for i in 0..32 {
                    let q = (rng.next() % 255) as i32 - 127;
                    gg.push(q as i8 as u8);
                    codes[r * k + 32 * b + i] = q as i8;
                }
            }
        }
        // derived (GGUF layout)
        let op = Op::load(&gpu, dop::gemv_q8_0(1, rows, k).lower().unwrap(), 3).unwrap();
        let (dx, dw, dout) = (
            DevBuf::from_slice(&xf).unwrap(),
            DevBuf::from_slice(&gg).unwrap(),
            DevBuf::zeroed(rows * 4).unwrap(),
        );
        let ptrs = [dx.ptr, dw.ptr, dout.ptr];
        let us = best_us(&s, || op.launch(&s, &ptrs).unwrap());
        line(&format!("derived Q8_0 GEMV {rows}×{k} ({} kernels)", op.kernels()), gg.len() as f64, us);
        let want = dout.read::<f32>(0, rows).unwrap();
        // repacked
        let (rc, rs, rx, rxs, ro) = (
            DevBuf::from_slice(&codes).unwrap(),
            DevBuf::from_slice(&sc).unwrap(),
            DevBuf::from_slice(&xq).unwrap(),
            DevBuf::from_slice(&xs).unwrap(),
            DevBuf::zeroed(rows * 4).unwrap(),
        );
        let bytes = (rows * k + rows * k / 32 * 2) as f64;
        for (name, r) in [("gq8r1", 1usize), ("gq8r2", 2), ("gq8r4", 4)] {
            let f = m.func(name).unwrap();
            let blocks = rows.div_ceil(8 * r) as u32;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (rc.ptr, rs.ptr, rx.ptr, rxs.ptr, ro.ptr, rows as i32);
            let (mut a6, mut a7) = (0u64, 0u64);
            let us = best_us(&s, || unsafe {
                let mut a = [
                    &mut a0 as *mut u64 as *mut _, &mut a1 as *mut u64 as *mut _, &mut a2 as *mut u64 as *mut _,
                    &mut a3 as *mut u64 as *mut _, &mut a4 as *mut u64 as *mut _, &mut a5 as *mut i32 as *mut _,
                    &mut a6 as *mut u64 as *mut _, &mut a7 as *mut u64 as *mut _,
                ];
                gpu::launch(f, (blocks, 1, 1), (256, 1, 1), 0, &s, &mut a).unwrap();
            });
            let got = ro.read::<f32>(0, rows).unwrap();
            let err = got.iter().zip(&want).map(|(g, w)| ((g - w) / w.abs().max(1e-3)).abs()).fold(0f32, f32::max);
            line(&format!("repacked {name} {rows}×{k} (vs derived max rel {err:.1e})"), bytes, us);
        }
    }
    fill_drain(&gpu, &s, &m, peak);
}

/// Repacked Q8 GEMV operands for `rows` × 2048 (contents irrelevant to timing).
struct RepackedMat {
    c: DevBuf,
    s: DevBuf,
    bytes: usize,
}
fn repacked(rows: usize) -> RepackedMat {
    let k = 2048;
    RepackedMat {
        c: DevBuf::from_slice(&vec![1i8; rows * k]).unwrap(),
        s: DevBuf::from_slice(&vec![0x3c00u16; rows * k / 32]).unwrap(),
        bytes: rows * k + rows * k / 32 * 2,
    }
}

/// Launch gq8r2 over `m`, prefetching `pf` (codes then scales, as two ranges? one: codes).
unsafe fn launch_r2(f: cudarc::driver::sys::CUfunction, s: &Stream, m: &RepackedMat, rows: usize, x: &DevBuf, xs: &DevBuf, out: &DevBuf, pf: Option<&RepackedMat>) {
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (m.c.ptr, m.s.ptr, x.ptr, xs.ptr, out.ptr, rows as i32);
    let (mut a6, mut a7) = pf.map_or((0u64, 0u64), |p| (p.c.ptr, (p.bytes - p.bytes / 33) as u64));
    let mut a = [
        &mut a0 as *mut u64 as *mut _, &mut a1 as *mut u64 as *mut _, &mut a2 as *mut u64 as *mut _,
        &mut a3 as *mut u64 as *mut _, &mut a4 as *mut u64 as *mut _, &mut a5 as *mut i32 as *mut _,
        &mut a6 as *mut u64 as *mut _, &mut a7 as *mut u64 as *mut _,
    ];
    gpu::launch(f, (rows.div_ceil(16) as u32, 1, 1), (256, 1, 1), 0, s, &mut a).unwrap();
}

/// Fill/drain: one GEMV's time against its bytes over a sweep of sizes (fit t = τ + bytes/BW),
/// then the L2-prefetch pair: GEMV A (prefetching B) followed by GEMV B, B's time with and
/// without the prefetch, an unrelated 512 MB stream in between each pair so nothing is warm.
fn fill_drain(gpu: &Gpu, s: &Stream, m: &gpu::Module, peak: f64) {
    let f = m.func("gq8r2").unwrap();
    let k = 2048;
    let x = DevBuf::from_slice(&vec![1i8; k]).unwrap();
    let xs = DevBuf::from_slice(&vec![1f32; k / 32]).unwrap();
    let out = DevBuf::zeroed(200_000 * 4).unwrap();
    let flush = DevBuf::zeroed(512 << 20).unwrap();
    let fl = m.func("stream_u4").unwrap();
    let do_flush = || unsafe {
        let (mut p, mut n, mut o) = (flush.ptr, ((512u64 << 20) / 16), out.ptr);
        let mut a = [&mut p as *mut u64 as *mut _, &mut n as *mut u64 as *mut _, &mut o as *mut u64 as *mut _];
        gpu::launch(fl, (1056, 1, 1), (256, 1, 1), 0, s, &mut a).unwrap();
    };
    let _ = gpu;
    println!("\n| fill/drain: rows × 2048 (repacked R=2, cold L2) | MB | µs | GB/s |");
    println!("|---|---|---|---|");
    let mut pts = Vec::new();
    for rows in [128usize, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] {
        let mat = repacked(rows);
        let mut best = f64::MAX;
        for _ in 0..15 {
            do_flush();
            let (a, b) = (Event::new(true).unwrap(), Event::new(true).unwrap());
            a.record(s).unwrap();
            unsafe { launch_r2(f, s, &mat, rows, &x, &xs, &out, None) };
            b.record(s).unwrap();
            b.sync().unwrap();
            best = best.min(b.since(&a).unwrap() as f64 * 1000.0);
        }
        pts.push((mat.bytes as f64, best));
        println!("| {rows} | {:.2} | {best:.2} | {:.0} |", mat.bytes as f64 / 1e6, mat.bytes as f64 / best / 1e3);
    }
    // least squares t = tau + bytes / bw over the sizes
    let n = pts.len() as f64;
    let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
    let sxx = pts.iter().map(|p| p.0 * p.0).sum::<f64>();
    let sxy = pts.iter().map(|p| p.0 * p.1).sum::<f64>();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let tau = (sy - slope * sx) / n;
    println!("\nfit: tau (fill + drain, incl. launch in a stream) = {tau:.2} µs, streaming BW = {:.0} GB/s ({:.2} of peak)", 1e-3 / slope, 1e-3 / slope / peak);
    // empty-kernel reference: the same launch with rows = 0 work
    println!("\n| L2 prefetch pair: A then B, rows each | B alone µs | B after A (no prefetch) µs | B after A prefetching B µs | A µs (no pf / pf) |");
    println!("|---|---|---|---|---|");
    for rows in [1024usize, 2048, 4096, 8192] {
        let (ma, mb) = (repacked(rows), repacked(rows));
        let time = |pf: bool, with_a: bool| -> (f64, f64) {
            let mut best = (f64::MAX, f64::MAX);
            for _ in 0..15 {
                do_flush();
                let (e0, e1, e2) = (Event::new(true).unwrap(), Event::new(true).unwrap(), Event::new(true).unwrap());
                e0.record(s).unwrap();
                if with_a {
                    unsafe { launch_r2(f, s, &ma, rows, &x, &xs, &out, pf.then_some(&mb)) };
                }
                e1.record(s).unwrap();
                unsafe { launch_r2(f, s, &mb, rows, &x, &xs, &out, None) };
                e2.record(s).unwrap();
                e2.sync().unwrap();
                let (ta, tb) = (e1.since(&e0).unwrap() as f64 * 1000.0, e2.since(&e1).unwrap() as f64 * 1000.0);
                if tb < best.1 {
                    best = (ta, tb);
                }
            }
            best
        };
        let (_, alone) = time(false, false);
        let (a0, b0) = time(false, true);
        let (a1, b1) = time(true, true);
        println!("| {rows} ({:.1} MB) | {alone:.2} | {b0:.2} | {b1:.2} | {a0:.2} / {a1:.2} |", ma.bytes as f64 / 1e6);
    }
}

fn half_bits(x: f32) -> u16 {
    tang_compile::tensor::precision::f16_bits(x as f64)
}
