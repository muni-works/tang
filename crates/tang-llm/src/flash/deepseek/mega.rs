//! The DS_MEGA lane: measuring what a step's structure costs (graph nodes, launches, grid
//! barriers) and the persistent dense-glue kernel that replaces a layer's small launches
//! (`mg_glue`, see kernels_mega.cu). Opt-in with `DS_MEGA=1`.

use super::gpu::DsW;
use anyhow::{ensure, Result};
use cudarc::driver::sys;
use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

/// kernels.cu (for its device helpers and `lin_mma`), then kernels_mega.cu.
pub const SRC: &str = concat!(include_str!("kernels.cu"), include_str!("kernels_mega.cu"));

/// Whether the opt-in mega paths are on (`DS_MEGA=1`).
pub fn on() -> bool {
    std::env::var("DS_MEGA").is_ok_and(|v| v == "1")
}

/// `mg_glue` launches so far (enqueued or captured), for tests.
pub static LAUNCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `struct MgArgs` in kernels_mega.cu.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MgArgs {
    pub h: u64,
    pub h2: u64,
    pub res: u64,
    pub post: u64,
    pub comb: u64,
    pub xo: u64,
    pub hpart: u64,
    pub pre_i: u64,
    pub hin: u64,
    pub scale: u64,
    pub base: u64,
    pub pre_o: u64,
    pub post_o: u64,
    pub comb_o: u64,
    pub xc: u64,
    pub xs: u64,
    pub ring: u64,
    pub all: u64,
    pub qr: u64,
    pub kv: u64,
    pub qc: u64,
    pub qsc: u64,
    pub tab: u64,
    pub p: u64,
    pub q: u64,
    pub part1: u64,
    pub part2: u64,
    pub part3: u64,
    pub cnt1: u64,
    pub cnt2: u64,
    pub cnt3: u64,
    pub gbar: u64,
    pub hbar: u64,
    pub w: DsW,
    pub nw: DsW,
    pub wqa: DsW,
    pub wkv: DsW,
    pub wqb: DsW,
    pub qn: DsW,
    pub kn: DsW,
    pub t: i32,
    pub hc: i32,
    pub d: i32,
    pub iters: i32,
    pub gx: i32,
    pub rd: i32,
    pub win: i32,
    pub ncopy: i32,
    pub eps: f32,
    pub neps: f32,
    pub reps: f32,
    pub qeps: f32,
    pub s1: i32,
    pub sb1: i32,
    pub nb1: i32,
    pub s2: i32,
    pub sb2: i32,
    pub nb2: i32,
    pub s3: i32,
    pub sb3: i32,
    pub nb3: i32,
    pub pf_bytes: i32,
}

/// gpu.rs's `ks_split` (the K-split GEMV's work split), so the partial sums, and with them
/// the results, are those of `Kernels::ks_gemv`.
fn ks_split(tiles: usize, groups: usize, sms: usize) -> (usize, usize, usize) {
    let target = 6 * sms;
    let mut sp = 1;
    while sp < 8 && tiles.div_ceil(8 / sp) < target && groups >= sp * 2 * 4 {
        sp *= 2;
    }
    let br = tiles.div_ceil(8 / sp).max(1);
    let mut sb = 1;
    while br * sb < target && groups >= (sb + 1) * sp * 4 && sb < 16 {
        sb += 1;
    }
    (sp, br, sb)
}

const PART: usize = 1 << 21;
/// `mg_glue`'s dynamic shared memory cap (the largest staged activation is ~47 KB; leaves
/// room under the 99 KB opt-in for the phases' static arrays, which grew when mg_hc took
/// ds_hc_fused_ks's shared body).
const SMEM_MAX: usize = 64 * 1024;
const CNT: usize = 1 << 13;

/// The megakernel's module and persistent scratch.
pub struct Mega {
    _m: Module,
    f: [sys::CUfunction; 2],
    sms: usize,
    part: [DevBuf; 3],
    cnt: [DevBuf; 3],
    bar: DevBuf,
    /// Blocks per SM (`DS_MEGA_BPS`, capped by the occupancy).
    bps: usize,
    pf: i32,
}

impl Mega {
    pub fn new(g: &Gpu) -> Result<Self> {
        let m = g.module(SRC)?;
        let f = [m.func("mg_glue_f2")?, m.func("mg_glue_f3")?];
        for &fi in &f {
            // dynamic shared memory past the 48 KB default (the staged activation of a
            // 5120-column projection at t = 8 is 46 KB, on top of the phases' static arrays)
            gpu::check(
                unsafe {
                    sys::cuFuncSetAttribute(
                        fi,
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        SMEM_MAX as i32,
                    )
                },
                "cuFuncSetAttribute",
            )?;
        }
        let sms = g.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
            as usize;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Ok(Mega {
            _m: m,
            f,
            sms,
            part: [
                DevBuf::zeroed(PART * 4)?,
                DevBuf::zeroed(PART * 4)?,
                DevBuf::zeroed(PART * 4)?,
            ],
            cnt: [
                DevBuf::zeroed(CNT * 4)?,
                DevBuf::zeroed(CNT * 4)?,
                DevBuf::zeroed(CNT * 4)?,
            ],
            bar: DevBuf::zeroed(64)?,
            bps: env("DS_MEGA_BPS", 2),
            pf: (env("DS_MEGA_PF_MB", 16) << 20) as i32,
        })
    }

    /// Whether `mg_glue` takes these shapes.
    pub fn ok(&self, a: &MgArgs) -> bool {
        let f = a.wqa.fmt;
        let ci = 256 / a.hc.max(1);
        (1..=8).contains(&a.t)
            && (f == 2 || f == 3)
            && a.wkv.fmt == f
            && a.wqb.fmt == f
            && 256 % a.hc == 0
            && ci % 32 == 0
            && a.d % ci == 0
            && a.w.rows <= 80
            && (a.gx as usize) < self.sms
    }

    /// Launch `mg_glue` on `a` (the scratch, barrier and split fields are filled here).
    pub fn launch(&self, s: &Stream, mut a: MgArgs) -> Result<()> {
        ensure!(self.ok(&a), "mg_glue: unsupported shapes");
        let t = a.t as usize;
        let mut smem = 0usize;
        let mut split = |w: &DsW| -> (i32, i32, i32) {
            let (rows, cols) = (w.rows as usize, w.cols as usize);
            let nch = cols / 32;
            let pitch = cols.div_ceil(128) * 128 + 32;
            smem = smem.max(t * pitch + t * nch * 4);
            let (sp, br, mut sb) = ks_split(rows.div_ceil(16), nch, self.sms);
            if sb * t * rows > PART || br > CNT {
                sb = 1;
            }
            (sp as i32, sb as i32, br as i32)
        };
        (a.s1, a.sb1, a.nb1) = split(&a.wqa);
        (a.s2, a.sb2, a.nb2) = split(&a.wkv);
        (a.s3, a.sb3, a.nb3) = split(&a.wqb);
        ensure!(smem <= SMEM_MAX, "mg_glue: {smem} B of shared memory");
        [a.part1, a.part2, a.part3] = [self.part[0].ptr, self.part[1].ptr, self.part[2].ptr];
        [a.cnt1, a.cnt2, a.cnt3] = [self.cnt[0].ptr, self.cnt[1].ptr, self.cnt[2].ptr];
        a.gbar = self.bar.ptr;
        a.hbar = self.bar.ptr + 32;
        a.pf_bytes = self.pf;
        let f = self.f[(a.wqa.fmt - 2) as usize];
        let mut per_sm = 0i32;
        gpu::check(
            unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut per_sm, f, 256, smem) },
            "occupancy",
        )?;
        ensure!(per_sm >= 1, "mg_glue: not resident");
        let grid = self.sms * self.bps.min(per_sm as usize).max(1);
        LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            super::obs::launch(
                f,
                (grid as u32, 1, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![a],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_moe::args;
    use tang_moe::gpu::{Event, Graph};

    fn time(s: &Stream, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f32> {
        let (a, b) = (Event::new(true)?, Event::new(true)?);
        f()?;
        s.sync()?;
        let mut best = f32::MAX;
        for _ in 0..reps {
            a.record(s)?;
            f()?;
            b.record(s)?;
            b.sync()?;
            best = best.min(b.since(&a)?);
        }
        Ok(best * 1000.0) // µs
    }

    /// Launch floor on this GPU: µs per node of a captured chain of N dependent trivial
    /// kernels, vs plain stream launches, and a persistent kernel's grid barrier. Run with
    /// `--ignored --nocapture` (one GPU; CUDA_VISIBLE_DEVICES picks it).
    #[test]
    #[ignore]
    fn mega_launch_floor() -> Result<()> {
        let Some(g) = Gpu::for_test() else {
            return Ok(());
        };
        g.bind()?;
        let m = g.module(SRC)?;
        let tick = m.func("mg_tick")?;
        let bar = m.func("mg_barrier_loop")?;
        let sms =
            g.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let s = Stream::new()?;
        let p = DevBuf::zeroed(4 * 4096)?;
        let cnt = DevBuf::zeroed(8)?;
        let src = DevBuf::zeroed(256)?;
        let dst = DevBuf::zeroed(256)?;
        println!("sms {sms}");
        let chain = |s: &Stream, n: usize, nb: u32, memcpy: bool| -> gpu::Result<()> {
            for i in 0..n {
                if memcpy && i % 10 == 9 {
                    gpu::check(
                        unsafe { sys::cuMemcpyDtoDAsync_v2(dst.ptr, src.ptr, 64, s.0) },
                        "memcpy",
                    )?;
                    continue;
                }
                unsafe { gpu::launch(tick, (nb, 1, 1), (128, 1, 1), 0, s, args![p.ptr])? };
            }
            Ok(())
        };
        println!("| N | graph 1-block | graph {sms}-block | graph {sms}-block + memcpy/10 | stream 1-block | stream {sms}-block |");
        println!("|---|---|---|---|---|---|");
        for n in [300usize, 600, 1200, 2400] {
            let mut row = vec![];
            for (nb, mc) in [(1, false), (sms, false), (sms, true)] {
                let gr = Graph::capture(&s, |s| chain(s, n, nb, mc))?;
                let us = time(&s, 20, || Ok(gr.launch(&s)?))?;
                row.push(us / n as f32);
            }
            for nb in [1, sms] {
                let us = time(&s, 10, || Ok(chain(&s, n, nb, false)?))?;
                row.push(us / n as f32);
            }
            println!(
                "| {n} | {} |",
                row.iter()
                    .map(|v| format!("{v:.2}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
        }
        println!("\n| blocks x threads | µs per grid barrier |");
        println!("|---|---|");
        for (nb, th) in [
            (sms, 256u32),
            (sms, 1024),
            (2 * sms, 256),
            (64, 256),
            (8, 256),
        ] {
            let r1 = 1i32;
            let r = 1000i32;
            let one = time(&s, 20, || unsafe {
                Ok(gpu::launch(
                    bar,
                    (nb, 1, 1),
                    (th, 1, 1),
                    0,
                    &s,
                    args![p.ptr, cnt.ptr, r1],
                )?)
            })?;
            let many = time(&s, 20, || unsafe {
                Ok(gpu::launch(
                    bar,
                    (nb, 1, 1),
                    (th, 1, 1),
                    0,
                    &s,
                    args![p.ptr, cnt.ptr, r],
                )?)
            })?;
            println!("| {nb} x {th} | {:.2} |", (many - one) / (r - 1) as f32);
        }
        Ok(())
    }

    fn rnd(n: usize, seed: u64, f: impl Fn(u64) -> f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                f(x)
            })
            .collect()
    }

    fn unif(x: u64) -> f32 {
        (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    /// A random fp8 weight (e4m3 codes without NaN, unit e8m0 32 x 32 block scales).
    fn fp8(rows: usize, cols: usize, seed: u64) -> (DevBuf, DevBuf, DsW) {
        let b: Vec<u8> = rnd(rows * cols, seed, |x| (x >> 33) as f32)
            .iter()
            .map(|v| (*v as u32 as u8) & 0xb7)
            .collect();
        let sc = vec![127u8; rows.div_ceil(32) * cols.div_ceil(32)];
        let (w, s) = (
            DevBuf::from_slice(&b).unwrap(),
            DevBuf::from_slice(&sc).unwrap(),
        );
        let mut v = DsW::default();
        (v.w, v.s, v.rows, v.cols, v.fmt) = (w.ptr, s.ptr, rows as i32, cols as i32, 2);
        (w, s, v)
    }

    fn f32w(rows: usize, cols: usize, seed: u64, amp: f32) -> (DevBuf, DsW) {
        let w = DevBuf::from_slice(&rnd(rows * cols, seed, |x| amp * unif(x))).unwrap();
        let mut v = DsW::default();
        (v.w, v.rows, v.cols, v.fmt) = (w.ptr, rows as i32, cols as i32, 0);
        (w, v)
    }

    /// `mg_glue` against the seven launches it replaces at the real model's shapes (dim
    /// 5120, hc 4, q_lora 1280, 64 x 512 heads, window 128), one GPU: µs per layer, graph
    /// replay of 40 layers' worth, and the outputs compared bit for bit.
    /// `--ignored --nocapture`; `DS_MEGA_BPS` / `DS_MEGA_PF_MB` tune it.
    #[test]
    #[ignore]
    fn mega_glue_bench() -> Result<()> {
        use super::super::gpu::Kernels;
        let Some(g) = Gpu::for_test() else {
            return Ok(());
        };
        g.bind()?;
        let k = Kernels::new(&g)?;
        let mega = Mega::new(&g)?;
        let s = Stream::new()?;
        let (d, hc, ql, nh, hd, win, rd) = (
            5120usize, 4usize, 1280usize, 64usize, 512usize, 128usize, 64usize,
        );
        let rows = (2 + hc) * hc;
        let gx = d / (256 / hc);
        let f = |n: usize, seed: u64, amp: f32| {
            DevBuf::from_slice(&rnd(n, seed, |x| amp * unif(x))).unwrap()
        };
        let (_hw, hw) = f32w(rows, hc * d, 1, 0.02);
        let (_nw, nw) = f32w(1, d, 2, 2.0);
        let (_qn, qn) = f32w(1, ql, 3, 2.0);
        let (_kn, kn) = f32w(1, hd, 4, 2.0);
        // three weight sets, rotated over the layers (151 MB: no set stays in L2)
        let sets: Vec<_> = (0..3u64)
            .map(|i| {
                (
                    fp8(ql, d, 5 + 10 * i),
                    fp8(hd, d, 6 + 10 * i),
                    fp8(nh * hd, ql, 7 + 10 * i),
                )
            })
            .collect();
        let ws: Vec<[DsW; 3]> = sets.iter().map(|(a, b, c)| [a.2, b.2, c.2]).collect();
        let scale = f(3, 8, 1.0);
        let base = f(rows, 9, 1.0);
        let tab = f(2 * 4096 * rd / 2, 10, 2.0);
        let pp = DevBuf::from_slice(&[200i32])?;
        println!("| t | separate µs | mg_glue µs | bit-exact |");
        println!("|---|---|---|---|");
        for t in [1usize, 6] {
            let h = f(t * d, 11, 1.0);
            let res = f(t * hc * d, 12, 1.0);
            let post = f(t * hc, 13, 1.0);
            let comb = f(t * hc * hc, 14, 0.5);
            let pre = f(t * hc, 15, 1.0);
            let ring0 = rnd(win * hd, 16, unif);
            // outputs: [separate, mega]
            let mk = || -> Result<Vec<DevBuf>> {
                Ok(vec![
                    DevBuf::zeroed(4 * t * hc * d)?,           // 0 xo
                    DevBuf::zeroed(4 * t * gx * (rows + 2))?,  // 1 hpart
                    DevBuf::zeroed(4 * t * hc)?,               // 2 pre_o
                    DevBuf::zeroed(4 * t * hc)?,               // 3 post_o
                    DevBuf::zeroed(4 * t * hc * hc)?,          // 4 comb_o
                    DevBuf::zeroed(4 * t * d)?,                // 5 hin
                    DevBuf::from_slice(&ring0)?,               // 6 ring
                    DevBuf::zeroed(4 * (win + t) * hd)?,       // 7 all
                    DevBuf::zeroed(4 * t * ql)?,               // 8 qr
                    DevBuf::zeroed(4 * t * hd)?,               // 9 kv
                    DevBuf::zeroed(4 * t * nh * hd)?,          // 10 q
                    DevBuf::zeroed(t * d + 4 * t * d / 32)?,   // 11 xc | xs
                    DevBuf::zeroed(t * ql + 4 * t * ql / 32)?, // 12 qc | qsc
                ])
            };
            let (o1, o2) = (mk()?, mk()?);
            let cnt = DevBuf::zeroed(8)?;
            let all = |o: &[DevBuf]| if t > 1 { o[7].ptr } else { 0 };
            let sep = |s: &Stream, l: usize| -> Result<()> {
                let [wqa, wkv, wqb] = ws[l % 3];
                let o = &o1;
                let (ti, hci, di) = (t as i32, hc as i32, d as i32);
                let (it, eps, neps) = (20i32, 1e-6f32, 1e-20f32);
                let (h2, xc, xs) = (0u64, o[11].ptr, o[11].ptr + (t * d) as u64);
                let (qc, qs) = (o[12].ptr, o[12].ptr + (t * ql) as u64);
                unsafe {
                    k.go(
                        "ds_hc_fused_ks",
                        (gx as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        s,
                        args![
                            h.ptr, h2, res.ptr, post.ptr, comb.ptr, o[0].ptr, hw, o[1].ptr,
                            pre.ptr, o[5].ptr, nw, scale.ptr, base.ptr, o[2].ptr, o[3].ptr,
                            o[4].ptr, cnt.ptr, ti, hci, di, it, eps, neps, neps, 0i32
                        ],
                    )?;
                }
                k.ks_quant_to(s, o[5].ptr, xc, xs, t * d)?;
                k.ks_gemv(s, &wqa, None, xc, xs, o[8].ptr, t, true, 0, 0.0)?;
                k.ks_gemv(s, &wkv, None, xc, xs, o[9].ptr, t, true, 0, 0.0)?;
                if t > 1 {
                    gpu::check(
                        unsafe { sys::cuMemcpyDtoDAsync_v2(o[7].ptr, o[6].ptr, 4 * win * hd, s.0) },
                        "copy",
                    )?;
                }
                let (rdi, wi, allp) = (rd as i32, win as i32, all(o));
                unsafe {
                    k.go(
                        "ds_qkv_post",
                        (t as u32, 2, 1),
                        (1024, 1, 1),
                        0,
                        s,
                        args![
                            o[8].ptr, qn, qc, qs, o[9].ptr, kn, tab.ptr, rdi, pp.ptr, o[6].ptr, wi,
                            allp, neps
                        ],
                    )?;
                }
                k.ks_gemv(s, &wqb, None, qc, qs, o[10].ptr, t, true, 0, 0.0)?;
                Ok(())
            };
            let o = &o2;
            let a = MgArgs {
                h: h.ptr,
                res: res.ptr,
                post: post.ptr,
                comb: comb.ptr,
                xo: o[0].ptr,
                hpart: o[1].ptr,
                pre_i: pre.ptr,
                hin: o[5].ptr,
                scale: scale.ptr,
                base: base.ptr,
                pre_o: o[2].ptr,
                post_o: o[3].ptr,
                comb_o: o[4].ptr,
                xc: o[11].ptr,
                xs: o[11].ptr + (t * d) as u64,
                ring: o[6].ptr,
                all: all(o),
                qr: o[8].ptr,
                kv: o[9].ptr,
                qc: o[12].ptr,
                qsc: o[12].ptr + (t * ql) as u64,
                tab: tab.ptr,
                p: pp.ptr,
                q: o[10].ptr,
                w: hw,
                nw,
                wqa: ws[0][0],
                wkv: ws[0][1],
                wqb: ws[0][2],
                qn,
                kn,
                t: t as i32,
                hc: hc as i32,
                d: d as i32,
                iters: 20,
                gx: gx as i32,
                rd: rd as i32,
                win: win as i32,
                ncopy: if t > 1 { (win * hd) as i32 } else { 0 },
                eps: 1e-6,
                neps: 1e-20,
                reps: 1e-20,
                qeps: 1e-20,
                ..Default::default()
            };
            let meg = |s: &Stream, l: usize| {
                let [wqa, wkv, wqb] = ws[l % 3];
                mega.launch(s, MgArgs { wqa, wkv, wqb, ..a })
            };
            sep(&s, 0)?;
            meg(&s, 0)?;
            s.sync()?;
            let mut same = true;
            for i in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
                let n = o1[i].len / 4;
                let (x, y) = (o1[i].read::<u32>(0, n)?, o2[i].read::<u32>(0, n)?);
                let bad = x.iter().zip(&y).filter(|(p, q)| p != q).count();
                if bad > 0 {
                    println!("t {t}: buffer {i}: {bad} of {n} words differ");
                    same = false;
                }
            }
            let layers = 40;
            let g1 = Graph::capture(&s, |s| {
                for l in 0..layers {
                    sep(s, l).map_err(|e| gpu::Error(e.to_string()))?;
                }
                Ok(())
            })?;
            let g2 = Graph::capture(&s, |s| {
                for l in 0..layers {
                    meg(s, l).map_err(|e| gpu::Error(e.to_string()))?;
                }
                Ok(())
            })?;
            let us1 = time(&s, 20, || Ok(g1.launch(&s)?))? / layers as f32;
            let us2 = time(&s, 20, || Ok(g2.launch(&s)?))? / layers as f32;
            println!("| {t} | {us1:.1} | {us2:.1} | {same} |");
        }
        Ok(())
    }

    fn bytes_w(rows: usize, cols: usize, fmt: i32, seed: u64) -> (Vec<DevBuf>, DsW) {
        let r = |n: usize| -> Vec<u8> {
            rnd(n.div_ceil(4), seed, |x| (x >> 32) as u32 as f32)
                .iter()
                .flat_map(|v| ((*v as u32) & 0x37373737).to_le_bytes())
                .take(n)
                .collect()
        };
        let mut v = DsW::default();
        (v.rows, v.cols, v.fmt) = (rows as i32, cols as i32, fmt);
        let (w, sc) = match fmt {
            0 => (
                DevBuf::from_slice(&rnd(rows * cols, seed, |x| 0.02 * unif(x))).unwrap(),
                None,
            ),
            1 => {
                let h: Vec<u16> = rnd(rows * cols, seed, |x| 0.02 * unif(x))
                    .iter()
                    .map(|f| (f.to_bits() >> 16) as u16)
                    .collect();
                (DevBuf::from_slice(&h).unwrap(), None)
            }
            2 => (
                DevBuf::from_slice(&r(rows * cols)).unwrap(),
                Some(vec![127u8; rows.div_ceil(32) * cols.div_ceil(32)]),
            ),
            _ => (
                DevBuf::from_slice(&r(rows * cols / 2)).unwrap(),
                Some(vec![127u8; rows * (cols / 32)]),
            ),
        };
        v.w = w.ptr;
        let mut keep = vec![w];
        if let Some(sc) = sc {
            let b = DevBuf::from_slice(&sc).unwrap();
            v.s = b.ptr;
            keep.push(b);
        }
        (keep, v)
    }

    fn wbytes(w: &DsW) -> usize {
        let (r, c) = (w.rows as usize, w.cols as usize);
        match w.fmt {
            0 => r * c * 4,
            1 => r * c * 2,
            2 => r * c + r.div_ceil(32) * c.div_ceil(32),
            _ => r * c / 2 + r * (c / 32),
        }
    }

    /// One EP rank's grouped MoE launches at the decode shape (t = 7, top-6, 384 experts on
    /// rank `e % 8`, the fp8 shared expert folded in): `moe_gu_sh` and `moe_down_sh` (with its
    /// h quant) timed apart, CUDA graph replay, expert weights drawn from a pool of 48 so
    /// consecutive launches miss L2. Synthetic cases (`live` local experts, one token each),
    /// then with `DS_ROUTE_FILE` (a `DS_ROUTE_DUMP` file) the real routing of its first
    /// verify step (t = 7 lines, one per MoE layer) on every rank: per-rank ms per step and
    /// the step's critical path (Σ over layers of the slowest rank).
    /// `--ignored --nocapture mega_moe_slots`.
    #[test]
    #[ignore]
    fn mega_moe_slots() -> Result<()> {
        use super::super::gpu::Kernels;
        let Some(g) = Gpu::for_test() else {
            return Ok(());
        };
        g.bind()?;
        let k = Kernels::new(&g)?;
        let s = Stream::new()?;
        let (t, topk, ne, d, inter, world) = (7usize, 6usize, 384usize, 5120usize, 2304usize, 8);
        let x = DevBuf::from_slice(&rnd(t * d, 3, unif))?;
        let (xc, xs) = k.ks_x_ptrs();
        let pool_n = 48usize;
        let mut keep = vec![];
        let mut pool = vec![];
        for i in 0..pool_n {
            let sd = i as u64 * 3;
            let (a, b, c) = (
                bytes_w(inter, d, 3, sd + 1),
                bytes_w(inter, d, 3, sd + 2),
                bytes_w(d, inter, 3, sd + 3),
            );
            pool.push((a.1, b.1, c.1));
            keep.extend(a.0.into_iter().chain(b.0).chain(c.0));
        }
        // DS_SH_SPLIT=w: the shared expert as one rank's 1/w slice (gate/up rows, down's K)
        let shw: usize = std::env::var("DS_SH_SPLIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let ish = inter / shw;
        let mut shs = vec![];
        for c in 0..2u64 {
            let (s1, s3, s2) = (
                bytes_w(ish, d, 2, c * 7 + 50),
                bytes_w(ish, d, 2, c * 7 + 51),
                bytes_w(d, ish, 2, c * 7 + 52),
            );
            shs.push((s1.1, s3.1, s2.1));
            keep.extend(s1.0.into_iter().chain(s3.0).chain(s2.0));
        }
        let slot = |r: usize, c: usize, fmt: i32| {
            let mut v = DsW::default();
            (v.rows, v.cols, v.fmt) = (r as i32, c as i32, fmt);
            wbytes(&v)
        };
        k.ks_quant_to(&s, x.ptr, xc, xs, t * d)?;
        let mut next = 0usize;
        // (gu µs, down µs, live experts, bytes) of one routing on rank `rank`
        let mut case = |ids: &[i32], rank: usize| -> Result<(f32, f32, usize, usize)> {
            let local: std::collections::BTreeSet<usize> = ids
                .iter()
                .map(|&e| e as usize)
                .filter(|e| e % world == rank)
                .collect();
            let live = local.len();
            let copies = 2usize;
            let mut tabs = vec![];
            for c in 0..copies {
                let (mut w1, mut w3, mut w2) = (
                    vec![DsW::default(); ne],
                    vec![DsW::default(); ne],
                    vec![DsW::default(); ne],
                );
                for &e in &local {
                    let p = &pool[next % pool_n];
                    next += 1;
                    (w1[e], w3[e], w2[e]) = (p.0, p.1, p.2);
                }
                tabs.push((
                    DevBuf::from_slice(&w1)?,
                    DevBuf::from_slice(&w3)?,
                    DevBuf::from_slice(&w2)?,
                    shs[c].clone(),
                ));
            }
            let ids = DevBuf::from_slice(ids)?;
            let wts = DevBuf::from_slice(&vec![0.2f32; t * topk])?;
            let (counts, off) = (DevBuf::zeroed(ne * 4)?, DevBuf::zeroed((ne + 1) * 4)?);
            let (rows, rw) = (DevBuf::zeroed(t * topk * 4)?, DevBuf::zeroed(t * topk * 4)?);
            k.route(
                &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
            )?;
            let mask: Vec<i32> = (0..ne).map(|e| (e % world == rank) as i32).collect();
            let mask = DevBuf::from_slice(&mask)?;
            let cap = (t * topk).min(ne / world);
            let elist = DevBuf::zeroed(cap * 4)?;
            let (nei, ci) = (ne as i32, cap as i32);
            unsafe {
                k.go(
                    "ds_moe_plan",
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    &s,
                    args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
                )?
            };
            let nall = t * topk;
            let h = DevBuf::zeroed((nall + t) * inter * 4)?;
            let part = DevBuf::zeroed((nall + t) * d * 4)?;
            let gu = |s: &Stream, i: usize| -> Result<()> {
                let tb = &tabs[i % copies];
                k.moe_gu_sh(
                    s,
                    3,
                    (tb.0.ptr, tb.1.ptr, elist.ptr),
                    (off.ptr, rows.ptr, rw.ptr),
                    (&tb.3 .0, &tb.3 .1, true),
                    0,
                    h.ptr,
                    (t, nall, cap, inter, d),
                    10.0,
                )
            };
            let down = |s: &Stream, i: usize| -> Result<()> {
                let tb = &tabs[i % copies];
                k.moe_down_sh(
                    s,
                    3,
                    (tb.2.ptr, elist.ptr, off.ptr),
                    (&tb.3 .2, true),
                    h.ptr,
                    part.ptr,
                    (t, nall, cap, d, inter),
                )
            };
            let reps = 10usize;
            let mut us = [0f32; 2];
            let fs: [&dyn Fn(&Stream, usize) -> Result<()>; 2] = [&gu, &down];
            for (j, f) in fs.into_iter().enumerate() {
                f(&s, 0)?;
                s.sync()?;
                let gr = Graph::capture(&s, |s| {
                    for i in 0..reps {
                        f(s, i).map_err(|e| gpu::Error(e.to_string()))?;
                    }
                    Ok(())
                })?;
                us[j] = time(&s, 5, || Ok(gr.launch(&s)?))? / reps as f32;
            }
            let bytes = live * (2 * slot(inter, d, 3) + slot(d, inter, 3))
                + 2 * slot(ish, d, 2)
                + slot(d, ish, 2);
            Ok((us[0], us[1], live, bytes))
        };
        println!("| live | gu µs | down µs | GB/s | % of 3350 |");
        println!("|---|---|---|---|---|");
        for live in [1usize, 5, 8] {
            // token i's first pick -> expert 8 * i (rank 0) for i < live, the rest elsewhere
            let ids: Vec<i32> = (0..t * topk)
                .map(|j| {
                    if j % topk == 0 && j / topk < live {
                        (8 * (j / topk)) as i32
                    } else {
                        (8 * j + 1 + j % 7) as i32 % ne as i32 | 1
                    }
                })
                .collect();
            let (a, b, _, by) = case(&ids, 0)?;
            let gbs = by as f32 / (a + b) / 1e3;
            println!(
                "| {live} | {a:.1} | {b:.1} | {gbs:.0} | {:.0}% |",
                gbs / 33.5
            );
        }
        let Ok(path) = std::env::var("DS_ROUTE_FILE") else {
            return Ok(());
        };
        // the first verify step: consecutive distinct t = 7 lines until a layer id repeats
        let text = std::fs::read_to_string(path)?;
        let mut step: Vec<(usize, Vec<i32>)> = vec![];
        for l in text.lines() {
            let v: Vec<i64> = l.split(' ').filter_map(|w| w.parse().ok()).collect();
            if v.len() != 2 + t * topk || v[1] as usize != t {
                continue;
            }
            let lid = v[0] as usize;
            let ids: Vec<i32> = v[2..].iter().map(|&e| e as i32).collect();
            if step.last().is_some_and(|(l, i)| *l == lid && *i == ids) {
                continue; // the other ranks' copy of the same line
            }
            if step.iter().any(|(l, _)| *l == lid) {
                break;
            }
            step.push((lid, ids));
        }
        println!(
            "\nreal routing, {} MoE layers of one verify step",
            step.len()
        );
        let mut rank_ms = vec![0f32; world];
        let mut crit = 0f32;
        let mut lives = vec![0usize; world];
        for (lid, ids) in &step {
            let mut worst = 0f32;
            let mut row = vec![];
            for r in 0..world {
                let (a, b, live, _) = case(ids, r)?;
                rank_ms[r] += (a + b) / 1e3;
                lives[r] += live;
                worst = worst.max(a + b);
                row.push(format!("{:.0}({live})", a + b));
            }
            crit += worst / 1e3;
            println!("layer {lid}: {}", row.join(" "));
        }
        println!(
            "per-rank gu+down ms/step: {}",
            rank_ms
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!("per-rank live experts/step: {lives:?}");
        println!("critical path (Σ layers of the slowest rank): {crit:.2} ms/step");
        drop(keep);
        Ok(())
    }

    /// Achieved weight bandwidth of every weight-streaming projection of a verify step
    /// (t = 6) at the real shapes, through the launcher the engine uses; copies rotated so
    /// nothing stays in L2. `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn mega_bw_table() -> Result<()> {
        use super::super::gpu::Kernels;
        let Some(g) = Gpu::for_test() else {
            return Ok(());
        };
        g.bind()?;
        let k = Kernels::new(&g)?;
        let s = Stream::new()?;
        let t = std::env::var("DS_BW_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6usize);
        let x = DevBuf::from_slice(&rnd(t * 32768, 3, unif))?;
        let y = DevBuf::zeroed(4 * t * 129280)?;
        let y2 = DevBuf::zeroed(4 * t * 129280)?;
        let (xc, xs) = k.ks_x_ptrs();
        // (name, rows, cols, fmt, kind): kind 0 ks_gemv, 1 ks_gemv gu (w1 + w3), 2 wo_ks,
        // 3 linear, 4 head, 5 ks_gemv2 (w and a 512-row wkv on one activation), 6 gemv_bf16
        let rows: &[(&str, usize, usize, i32, u8)] = &[
            ("wq_a (fp8, ds_ks)", 1280, 5120, 2, 0),
            ("wkv (fp8, ds_ks)", 512, 5120, 2, 0),
            ("wq_a + wkv (fp8, ds_ks2)", 1280, 5120, 2, 5),
            ("wq_b (fp8, ds_ks)", 32768, 1280, 2, 0),
            ("wo_a (fp8, ds_wo_ks, 8 groups)", 8192, 4096, 2, 2),
            ("wo_b (fp8, ds_ks)", 5120, 8192, 2, 0),
            ("indexer wq_b (fp8, ds_ks)", 4096, 1280, 2, 0),
            ("compressor wkv (bf16, ds_linear_t8)", 512, 5120, 1, 3),
            ("compressor wgate (bf16, ds_linear_t8)", 512, 5120, 1, 3),
            ("indexer weights_proj (bf16, ds_linear_t8)", 32, 5120, 1, 3),
            ("router gate (bf16, ds_linear_t8)", 384, 5120, 1, 3),
            ("router gate (bf16, ds_gemv_bf16_r1)", 384, 5120, 1, 6),
            ("hc mixes (f32, ds_linear_t8)", 24, 20480, 0, 3),
            ("expert gu, one slot (fp4, ds_ks gu)", 2304, 5120, 3, 1),
            ("expert down, one slot (fp4, ds_ks)", 5120, 2304, 3, 0),
            ("head (bf16, ds_head_bf16)", 129280, 5120, 1, 4),
        ];
        println!("| projection | rows x cols | MB | µs | GB/s | % of 3350 |");
        println!("|---|---|---|---|---|---|");
        for &(name, r, c, fmt, kind) in rows {
            let one = wbytes(&bytes_w(r.min(64), c, fmt, 1).1) * r / r.min(64);
            let copies = (200usize << 20).div_ceil(one).clamp(1, 32);
            let ws: Vec<_> = (0..copies)
                .map(|i| {
                    let a = bytes_w(r, c, fmt, 10 + i as u64);
                    let b = match kind {
                        1 => Some(bytes_w(r, c, fmt, 1000 + i as u64)),
                        5 => Some(bytes_w(512, c, fmt, 1000 + i as u64)),
                        _ => None,
                    };
                    (a, b)
                })
                .collect();
            if kind <= 1 || kind == 5 {
                k.ks_quant_to(&s, x.ptr, xc, xs, t * c)?;
            }
            let run = |s: &Stream, i: usize| -> Result<()> {
                let (a, b) = &ws[i % copies];
                let w = &a.1;
                match kind {
                    0 => k.ks_gemv(s, w, None, xc, xs, y.ptr, t, true, 0, 0.0)?,
                    1 => k.ks_gemv(
                        s,
                        w,
                        Some(&b.as_ref().unwrap().1),
                        xc,
                        xs,
                        y.ptr,
                        t,
                        true,
                        0,
                        10.0,
                    )?,
                    2 => k.wo_ks(s, w, x.ptr, y.ptr, t, 8, 1024)?,
                    3 => k.linear(s, w, x.ptr, y.ptr, t, false)?,
                    4 => k.head(s, w, x.ptr, y.ptr, t)?,
                    5 => {
                        k.ks_gemv2(s, w, &b.as_ref().unwrap().1, xc, xs, y.ptr, y2.ptr, t, true)?
                    }
                    _ => k.gemv_bf16(s, w, x.ptr, y.ptr, t)?,
                }
                Ok(())
            };
            if kind <= 1 || kind == 5 {
                ensure!(k.ks_ok(&ws[0].0 .1), "{name}: not on the K-split path");
            }
            run(&s, 0)?;
            s.sync()?;
            let n = copies.max(20);
            let gr = Graph::capture(&s, |s| {
                for i in 0..n {
                    run(s, i).map_err(|e| gpu::Error(e.to_string()))?;
                }
                Ok(())
            })?;
            let us = time(&s, 10, || Ok(gr.launch(&s)?))? / n as f32;
            let bytes = match kind {
                1 => 2 * one,
                5 => one + one * 512 / r,
                _ => one,
            };
            let gbs = bytes as f32 / us / 1e3;
            println!(
                "| {name} | {r} x {c} | {:.1} | {us:.1} | {gbs:.0} | {:.0}% |",
                bytes as f32 / 1048576.0,
                gbs / 33.5
            );
        }
        Ok(())
    }

    /// Programmatic dependent launch: µs per kernel of a dependent chain whose kernels each
    /// stream `mb` MB of their own weights before waiting on their predecessor, plain vs
    /// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`, captured in a graph and on
    /// the stream; the counters check the order held. `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn mega_pdl_floor() -> Result<()> {
        let Some(g) = Gpu::for_test() else {
            return Ok(());
        };
        g.bind()?;
        let m = g.module(SRC)?;
        let f = m.func("mg_pdl_step")?;
        let sms =
            g.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let s = Stream::new()?;
        let pool = DevBuf::zeroed(256 << 20)?;
        let p = DevBuf::zeroed(4 * 4096)?;
        let sink = DevBuf::zeroed(16)?;
        println!("| per-kernel weights | graph plain µs | graph PDL µs | stream plain µs | stream PDL µs | ideal (bytes at 3.35 TB/s) µs |");
        println!("|---|---|---|---|---|---|");
        let n = 400usize;
        let mut done = 0i32;
        for kb in [0usize, 256, 1024, 4096, 16384] {
            let bytes = kb << 10;
            let slots = if bytes == 0 {
                1
            } else {
                ((256usize << 20) / bytes).max(1)
            };
            let chain = |s: &Stream, pdl: bool| -> gpu::Result<()> {
                for i in 0..n {
                    let w = pool.ptr + ((i % slots) * bytes) as u64;
                    let (n16, pp, sk) = ((bytes / 16) as i64, p.ptr, sink.ptr);
                    unsafe {
                        super::super::pdl::launch(
                            f,
                            (sms * 2, 1, 1),
                            (256, 1, 1),
                            0,
                            s,
                            args![w, n16, pp, sk],
                            pdl,
                        )?
                    };
                }
                Ok(())
            };
            let mut row = vec![];
            for pdl in [false, true] {
                let gr = Graph::capture(&s, |s| chain(s, pdl))?;
                row.push(time(&s, 10, || Ok(gr.launch(&s)?))? / n as f32);
            }
            for pdl in [false, true] {
                row.push(time(&s, 5, || Ok(chain(&s, pdl)?))? / n as f32);
            }
            s.sync()?;
            // order check: every block's counter advanced by the same total
            let c = p.read::<i32>(0, 2 * sms as usize)?;
            done += 34 * n as i32; // (1 + 10) graph and (1 + 5) stream runs, plain and PDL
            ensure!(
                c.iter().all(|&v| v == done),
                "PDL lost increments: {:?} vs {done}",
                &c[..4]
            );
            println!(
                "| {kb} KB | {} | {:.2} |",
                row.iter()
                    .map(|v| format!("{v:.2}"))
                    .collect::<Vec<_>>()
                    .join(" | "),
                bytes as f32 / 3.35e6
            );
        }
        Ok(())
    }
}
