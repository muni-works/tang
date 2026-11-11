//! The Coder-Next engine's small per-token ops, hand kernel vs the op derived
//! from the tensor IR (`flash::derived_ops`), on the same random inputs at
//! the engine's shapes (hidden 2048, 512 experts, top-10): max error and
//! time per launch (min over warm launches).
//!
//! `cargo run --release -p tang-llm --features cuda --example coder_derived_ops [T...]`

use std::ffi::c_void;

use tang_llm::flash::coder_kernels::{self as ck, Shapes};
use tang_llm::flash::derived_ops::{self as dop, Op};
use tang_llm::flash::kernels;
use tang_moe::gpu::{self, DevBuf, Event, Gpu, Stream};

const N: usize = 2048;
const NE: usize = 512;
const K: usize = 10;
const EPS: f32 = 1e-6;

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    let scale = want
        .iter()
        .fold(0f64, |m, &v| m.max(v.abs() as f64))
        .max(1e-30);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w as f64).abs() / scale)
        .fold(0.0, f64::max)
}

/// Min µs a launch of `f` over `iters` timed single launches.
fn time(s: &Stream, mut f: impl FnMut()) -> f64 {
    // `ONCE=1`: three launches, no timing (for ncu / nsys)
    if std::env::var_os("ONCE").is_some() {
        for _ in 0..3 {
            f();
        }
        s.sync().unwrap();
        return f64::NAN;
    }
    for _ in 0..10 {
        f();
    }
    s.sync().unwrap();
    let mut best = f64::MAX;
    for _ in 0..50 {
        let (a, b) = (Event::new(true).unwrap(), Event::new(true).unwrap());
        a.record(s).unwrap();
        for _ in 0..20 {
            f();
        }
        b.record(s).unwrap();
        b.sync().unwrap();
        best = best.min(b.since(&a).unwrap() as f64 * 1000.0 / 20.0);
    }
    best
}

fn launch(
    f: cudarc::driver::sys::CUfunction,
    grid: u32,
    block: u32,
    s: &Stream,
    args: &mut [*mut c_void],
) {
    unsafe { gpu::launch(f, (grid, 1, 1), (block, 1, 1), 0, s, args) }.unwrap();
}

fn p<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

fn main() {
    let ts: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let ts = if ts.is_empty() { vec![1, 8] } else { ts };
    let gpu = Gpu::new().expect("cuda");
    let s = Stream::new().unwrap();
    let maxt = 64;
    let sh = Shapes {
        hidden: N,
        n_head: 16,
        n_kv: 2,
        head_dim: 256,
        n_rot: 64,
        ff: 512,
        topk: K,
        n_expert: NE,
    };
    let src = format!("{}{}{}", ck::prelude(&sh, maxt), kernels::GEMV_SRC, ck::SRC);
    let m = gpu.module(&src).expect("hand kernels");
    let (k_norm, k_addn, k_route) = (
        m.func("cx_norm_q8").unwrap(),
        m.func("cx_moe_add_norm").unwrap(),
        m.func("cx_route").unwrap(),
    );
    let ls = NE + 1;
    println!("| op | T | hand µs | derived µs | derived kernels | max rel err |");
    println!("|---|---|---|---|---|---|");
    for &t in &ts {
        let r0 = rnd(t * N, 1);
        let y = rnd(t * N, 2);
        let w: Vec<f32> = rnd(N, 3).iter().map(|v| 1.0 + 0.5 * v).collect();
        let (b_r, b_y, b_w) = (
            DevBuf::from_slice(&r0).unwrap(),
            DevBuf::from_slice(&y).unwrap(),
            DevBuf::from_slice(&w).unwrap(),
        );
        let b_x = DevBuf::zeroed_elems::<f32>(t * N).unwrap();
        let b_xq = DevBuf::zeroed(maxt * N + maxt * N / 32 * 4).unwrap();
        let (b_ro, b_dx) = (
            DevBuf::zeroed_elems::<f32>(t * N).unwrap(),
            DevBuf::zeroed_elems::<f32>(t * N).unwrap(),
        );

        // ---- add + rmsnorm
        let hand = |r: u64| {
            let (mut r, mut yy, mut ww, mut x, mut xq) = (r, b_y.ptr, b_w.ptr, b_x.ptr, b_xq.ptr);
            let (mut n, mut eps, mut add) = (N as i32, EPS, 1i32);
            launch(
                k_norm,
                t as u32,
                256,
                &s,
                &mut [
                    p(&mut r),
                    p(&mut yy),
                    p(&mut ww),
                    p(&mut x),
                    p(&mut xq),
                    p(&mut n),
                    p(&mut eps),
                    p(&mut add),
                ],
            );
        };
        let scratch_r = DevBuf::from_slice(&r0).unwrap();
        hand(scratch_r.ptr);
        s.sync().unwrap();
        let (want_r, want_x) = (
            scratch_r.read::<f32>(0, t * N).unwrap(),
            b_x.read::<f32>(0, t * N).unwrap(),
        );
        let op = Op::load(
            &gpu,
            dop::add_norm(t, N, EPS as f64, true).lower().unwrap(),
            5,
        )
        .unwrap();
        let ext = [b_r.ptr, b_y.ptr, b_w.ptr, b_ro.ptr, b_dx.ptr];
        op.launch(&s, &ext).unwrap();
        s.sync().unwrap();
        let (er, ex) = (
            max_rel(&b_ro.read::<f32>(0, t * N).unwrap(), &want_r),
            max_rel(&b_dx.read::<f32>(0, t * N).unwrap(), &want_x),
        );
        if std::env::var_os("DBG").is_some() {
            let d = b_dx.read::<f32>(0, t * N).unwrap();
            let bad: Vec<usize> = (0..t)
                .filter(|&ti| (0..N).any(|i| (d[ti * N + i] - want_x[ti * N + i]).abs() > 1e-3))
                .collect();
            let cpu: Vec<f32> = r0.iter().zip(&y).map(|(a, b)| a + b).collect();
            eprintln!(
                "add_norm T={t}: err r {er:.2e} x {ex:.2e}; bad rows {bad:?}; vs cpu: derived {:.2e} hand {:.2e}",
                max_rel(&b_ro.read::<f32>(0, t * N).unwrap(), &cpu),
                max_rel(&want_r, &cpu)
            );
        }
        let err = er.max(ex);
        // timing: the hand kernel adds in place, so time it on a copy (values drift; the time does not)
        let th = time(&s, || hand(scratch_r.ptr));
        let td = time(&s, || op.launch(&s, &ext).unwrap());
        println!(
            "| add_norm | {t} | {th:.2} | {td:.2} | {} | {err:.2e} |",
            op.kernels()
        );

        // ---- MoE combine + residual + norm
        let parts = rnd((t * K + t) * N, 4);
        let wts = rnd(t * K, 5);
        let lg = rnd(t * ls, 6);
        let (b_parts, b_wts, b_lg) = (
            DevBuf::from_slice(&parts).unwrap(),
            DevBuf::from_slice(&wts).unwrap(),
            DevBuf::from_slice(&lg).unwrap(),
        );
        let b_hy = DevBuf::zeroed_elems::<f32>(t * N).unwrap();
        let b_dy = DevBuf::zeroed_elems::<f32>(t * N).unwrap();
        let hand2 = |r: u64| {
            let (mut r, mut pa, mut wt, mut l, mut yy, mut ww, mut x, mut xq) = (
                r,
                b_parts.ptr,
                b_wts.ptr,
                b_lg.ptr,
                b_hy.ptr,
                b_w.ptr,
                b_x.ptr,
                b_xq.ptr,
            );
            let (mut lsv, mut tv, mut eps) = (ls as i32, t as i32, EPS);
            launch(
                k_addn,
                t as u32,
                1024,
                &s,
                &mut [
                    p(&mut r),
                    p(&mut pa),
                    p(&mut wt),
                    p(&mut l),
                    p(&mut lsv),
                    p(&mut tv),
                    p(&mut yy),
                    p(&mut ww),
                    p(&mut x),
                    p(&mut xq),
                    p(&mut eps),
                ],
            );
        };
        scratch_r.write(0, &r0).unwrap();
        hand2(scratch_r.ptr);
        s.sync().unwrap();
        let (want_y, want_r, want_x) = (
            b_hy.read::<f32>(0, t * N).unwrap(),
            scratch_r.read::<f32>(0, t * N).unwrap(),
            b_x.read::<f32>(0, t * N).unwrap(),
        );
        let op = Op::load(
            &gpu,
            dop::moe_combine_norm(t, N, K, NE, ls, EPS as f64)
                .lower()
                .unwrap(),
            8,
        )
        .unwrap();
        let ext = [
            b_r.ptr,
            b_parts.ptr,
            b_wts.ptr,
            b_lg.ptr,
            b_w.ptr,
            b_dy.ptr,
            b_ro.ptr,
            b_dx.ptr,
        ];
        op.launch(&s, &ext).unwrap();
        s.sync().unwrap();
        let err = max_rel(&b_dy.read::<f32>(0, t * N).unwrap(), &want_y)
            .max(max_rel(&b_ro.read::<f32>(0, t * N).unwrap(), &want_r))
            .max(max_rel(&b_dx.read::<f32>(0, t * N).unwrap(), &want_x));
        let th = time(&s, || hand2(scratch_r.ptr));
        let td = time(&s, || op.launch(&s, &ext).unwrap());
        println!(
            "| moe_combine_norm | {t} | {th:.2} | {td:.2} | {} | {err:.2e} |",
            op.kernels()
        );

        // ---- routing
        let b_ids = DevBuf::zeroed_elems::<u32>(t * K).unwrap();
        let b_hw = DevBuf::zeroed_elems::<f32>(t * K).unwrap();
        let hand3 = || {
            let (mut l, mut ids, mut ww) = (b_lg.ptr, b_ids.ptr, b_hw.ptr);
            let (mut lsv, mut sc) = (ls as i32, 1.0f32);
            launch(
                k_route,
                t as u32,
                32,
                &s,
                &mut [p(&mut l), p(&mut lsv), p(&mut ids), p(&mut ww), p(&mut sc)],
            );
        };
        hand3();
        s.sync().unwrap();
        let (want_ids, want_w) = (
            b_ids.read::<u32>(0, t * K).unwrap(),
            b_hw.read::<f32>(0, t * K).unwrap(),
        );
        let b_top = DevBuf::zeroed_elems::<f32>(t * 2 * K).unwrap();
        let b_dw = DevBuf::zeroed_elems::<f32>(t * K).unwrap();
        let op = Op::load(&gpu, dop::route(t, NE, ls, K).lower().unwrap(), 3).unwrap();
        let ext = [b_lg.ptr, b_top.ptr, b_dw.ptr];
        op.launch(&s, &ext).unwrap();
        s.sync().unwrap();
        let top = b_top.read::<f32>(0, t * 2 * K).unwrap();
        let ids_ok = (0..t * K).all(|i| top[2 * i] as u32 == want_ids[i]);
        let err = max_rel(&b_dw.read::<f32>(0, t * K).unwrap(), &want_w);
        let th = time(&s, hand3);
        let td = time(&s, || op.launch(&s, &ext).unwrap());
        println!(
            "| route | {t} | {th:.2} | {td:.2} | {} | {err:.2e} (ids {}) |",
            op.kernels(),
            if ids_ok { "equal" } else { "DIFFER" }
        );

        // ---- dense Q8_0 GEMV (attn_qkv-sized: 2048 -> 8192 rows... use 4096)
        for &(rows, k) in &[(4096usize, 2048usize), (2048, 4096), (151936, 2048)] {
            let x = rnd(t * k, 11);
            let mut wb = vec![0u8; rows * k / 32 * 34];
            let mut seed = 12u64;
            let mut wref = vec![0f32; rows * k];
            for blk in 0..rows * k / 32 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let d = 0.001 + (seed >> 40) as f32 / (1u64 << 24) as f32 * 0.01;
                let dh = half_bits(d);
                let dv = half_val(dh);
                wb[blk * 34..blk * 34 + 2].copy_from_slice(&dh.to_le_bytes());
                for j in 0..32 {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let q = ((seed >> 33) % 255) as i32 - 127;
                    wb[blk * 34 + 2 + j] = q as i8 as u8;
                    wref[blk * 32 + j] = dv * q as f32;
                }
            }
            // reference over the int8-rounded activations
            let mut want = vec![0f32; t * rows];
            for ti in 0..t {
                let mut xq = vec![0f64; k];
                for b in 0..k / 32 {
                    let blk = &x[ti * k + b * 32..][..32];
                    let a = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let d = a / 127.0;
                    for j in 0..32 {
                        xq[b * 32 + j] = if d == 0.0 {
                            0.0
                        } else {
                            ((blk[j] / d).round().clamp(-127.0, 127.0) * d) as f64
                        };
                    }
                }
                for r in 0..rows {
                    let row = &wref[r * k..(r + 1) * k];
                    want[ti * rows + r] = row
                        .iter()
                        .zip(&xq)
                        .map(|(&a, &b)| a as f64 * b)
                        .sum::<f64>() as f32;
                }
            }
            let (b_x, b_w) = (
                DevBuf::from_slice(&x).unwrap(),
                DevBuf::from_slice(&wb).unwrap(),
            );
            let b_o = DevBuf::zeroed_elems::<f32>(t * rows).unwrap();
            let op = Op::load(&gpu, dop::gemv_q8_0(t, rows, k).lower().unwrap(), 3).unwrap();
            let ext = [b_x.ptr, b_w.ptr, b_o.ptr];
            op.launch(&s, &ext).unwrap();
            s.sync().unwrap();
            let err = max_rel(&b_o.read::<f32>(0, t * rows).unwrap(), &want);
            let td = time(&s, || op.launch(&s, &ext).unwrap());
            let gbs = wb.len() as f64 / (td * 1e-6) / 1e9;
            println!(
                "| gemv_q8_0 {rows}x{k} | {t} | - | {td:.2} ({gbs:.0} GB/s) | {} | {err:.2e} |",
                op.kernels()
            );
        }
    }
}

fn half_bits(x: f32) -> u16 {
    tang_compile::tensor::precision::f16_bits(x as f64)
}
fn half_val(b: u16) -> f32 {
    tang_compile::tensor::precision::f16_from_bits(b) as f32
}
