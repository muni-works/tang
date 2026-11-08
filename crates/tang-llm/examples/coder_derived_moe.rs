//! The derived decode MoE layer (`flash::derived_ops::moe_block`) on the GPU
//! against a plain-Rust reference: random GGUF Q4_K gate/up and Q5_K down
//! expert stacks, Q8_0 shared expert, the same int8 rounding of x and the
//! SwiGLU output. Small expert count for the check (the reference
//! dequantizes), then timing at the engine's shape (512 experts, top-10).
//!
//! `cargo run --release -p tang-llm --features cuda --example coder_derived_moe [T...]`

use tang_compile::tensor::precision::{f16_bits, f16_from_bits};
use tang_compile::tensor::Format;
use tang_llm::flash::derived_ops::{self as dop, MoeShape, Op};
use tang_moe::gpu::{DevBuf, Event, Gpu, Stream};

struct Rng(u64);
impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn sym(&mut self) -> f32 {
        self.f32() * 2.0 - 1.0
    }
}

fn h(x: f32) -> u16 {
    f16_bits(x as f64)
}
fn hv(b: u16) -> f32 {
    f16_from_bits(b) as f32
}

fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn gen_kquant(n: usize, five: bool, amp: f32, rng: &mut Rng) -> Vec<u8> {
    let bb = if five { 176 } else { 144 };
    let qmax = if five { 31.0 } else { 15.0 };
    let mut out = vec![0u8; n / 256 * bb];
    for b in out.chunks_mut(bb) {
        let d = amp * (0.5 + rng.f32()) / (40.0 * qmax / 2.0);
        let dmin = d * qmax / 2.0 * (0.8 + 0.4 * rng.f32());
        b[0..2].copy_from_slice(&h(d).to_le_bytes());
        b[2..4].copy_from_slice(&h(dmin).to_le_bytes());
        for x in b[4..].iter_mut() {
            *x = (rng.f32() * 256.0) as u8;
        }
    }
    out
}

fn dequant_k(bytes: &[u8], five: bool) -> Vec<f32> {
    let bb = if five { 176 } else { 144 };
    let mut y = Vec::with_capacity(bytes.len() / bb * 256);
    for b in bytes.chunks(bb) {
        let d = hv(u16::from_le_bytes([b[0], b[1]]));
        let dmin = hv(u16::from_le_bytes([b[2], b[3]]));
        let sc = &b[4..16];
        let (qh, qs) = if five {
            (&b[16..48], &b[48..176])
        } else {
            (&b[0..0], &b[16..144])
        };
        for j in 0..4 {
            let q = &qs[32 * j..32 * j + 32];
            for half in 0..2 {
                let s = 2 * j + half;
                let (scv, mv) = scale_min(s, sc);
                let (d1, m1) = (d * scv as f32, dmin * mv as f32);
                for l in 0..32 {
                    let mut v = if half == 0 { q[l] & 0xF } else { q[l] >> 4 } as u32;
                    if five && (qh[l] >> s) & 1 == 1 {
                        v += 16;
                    }
                    y.push(d1 * v as f32 - m1);
                }
            }
        }
    }
    y
}

fn gen_q8_0(n: usize, amp: f32, rng: &mut Rng) -> (Vec<u8>, Vec<f32>) {
    let mut b = vec![0u8; n / 32 * 34];
    let mut v = vec![0f32; n];
    for blk in 0..n / 32 {
        let d = hv(h(amp * (0.5 + rng.f32()) / 127.0));
        b[blk * 34..blk * 34 + 2].copy_from_slice(&h(d).to_le_bytes());
        for j in 0..32 {
            let q = (rng.sym() * 127.0).round() as i32;
            b[blk * 34 + 2 + j] = q as i8 as u8;
            v[blk * 32 + j] = d * q as f32;
        }
    }
    (b, v)
}

fn q8x32(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    for blk in out.chunks_mut(32) {
        let amax = blk.iter().fold(0f32, |a, x| a.max(x.abs()));
        let d = amax / 127.0;
        for x in blk.iter_mut() {
            *x = if d == 0.0 {
                0.0
            } else {
                (*x / d).round().clamp(-127.0, 127.0) * d
            };
        }
    }
    out
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32
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

struct Bufs {
    ptrs: Vec<u64>,
    _b: Vec<DevBuf>,
    y: usize,
    ro: usize,
    xo: usize,
}

fn run(gpu: &Gpu, s: &Stream, m: MoeShape, check: bool) {
    let MoeShape {
        t,
        n,
        ff,
        ne,
        k,
        ls,
        ..
    } = m;
    let mut rng = Rng(0x1234_5678 ^ (t as u64 * 77) ^ ne as u64);
    let x: Vec<f32> = (0..t * n).map(|_| rng.sym()).collect();
    let lg: Vec<f32> = (0..t * ls).map(|_| 2.0 * rng.sym()).collect();
    let wg = gen_kquant(ne * ff * n, false, 0.05, &mut rng);
    let wu = gen_kquant(ne * ff * n, false, 0.05, &mut rng);
    let wd = gen_kquant(ne * n * ff, true, 0.05, &mut rng);
    let (sg, sgv) = gen_q8_0(ff * n, 0.05, &mut rng);
    let (su, suv) = gen_q8_0(ff * n, 0.05, &mut rng);
    let (sd, sdv) = gen_q8_0(n * ff, 0.05, &mut rng);
    let r: Vec<f32> = (0..t * n).map(|_| rng.sym()).collect();
    let wn: Vec<f32> = (0..n).map(|_| 1.0 + 0.3 * rng.sym()).collect();
    let up = |v: &[u8]| DevBuf::from_slice(v).unwrap();
    let upf = |v: &[f32]| DevBuf::from_slice(v).unwrap();
    let z = |n: usize| DevBuf::zeroed_elems::<f32>(n).unwrap();
    // gate ‖ up in one buffer (the engine's layout): wu / su point into it
    let cat = |a: &[u8], b: &[u8]| [a, b].concat();
    let b = vec![
        upf(&x),
        upf(&lg),
        up(&cat(&wg, &wu)),
        up(&[0u8; 4]),
        up(&wd),
        up(&cat(&sg, &su)),
        up(&[0u8; 4]),
        up(&sd),
        upf(&r),
        upf(&wn),
        z(t * n),
        z(t * n),
        z(t * n),
        z(t * n / 4 + t * n / 32),
    ];
    let mut ptrs: Vec<u64> = b.iter().map(|b| b.ptr).collect();
    ptrs[3] = ptrs[2] + wg.len() as u64;
    ptrs[6] = ptrs[5] + sg.len() as u64;
    let xq = ptrs.pop().unwrap();
    ptrs.extend([ptrs[2], ptrs[5], xq]);
    let bufs = Bufs {
        ptrs,
        y: 10,
        ro: 11,
        xo: 12,
        _b: b,
    };
    let t0 = std::time::Instant::now();
    let op = Op::load(gpu, dop::moe_block(m).lower().unwrap(), dop::MOE_EXT.len()).unwrap();
    let build = t0.elapsed().as_secs_f64();
    op.launch(s, &bufs.ptrs).unwrap();
    s.sync().unwrap();
    let mut err = f64::NAN;
    if check {
        let gu = dequant_k(&wg, false);
        let uu = dequant_k(&wu, false);
        let dd = dequant_k(&wd, true);
        let mut want_y = vec![0f32; t * n];
        for ti in 0..t {
            let xq = q8x32(&x[ti * n..(ti + 1) * n]);
            let l = &lg[ti * ls..ti * ls + ne];
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
            let top = &idx[..k];
            let z: f64 = top.iter().map(|&e| ((l[e] - l[top[0]]) as f64).exp()).sum();
            let ffn = |wg: &[f32], wu: &[f32], wd: &[f32]| {
                let g = matvec(wg, &xq, ff, n);
                let u = matvec(wu, &xq, ff, n);
                let hh: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                matvec(wd, &q8x32(&hh), n, ff)
            };
            for &e in top {
                let w = (((l[e] - l[top[0]]) as f64).exp() / z) as f32;
                let o = ffn(
                    &gu[e * ff * n..(e + 1) * ff * n],
                    &uu[e * ff * n..(e + 1) * ff * n],
                    &dd[e * n * ff..(e + 1) * n * ff],
                );
                for i in 0..n {
                    want_y[ti * n + i] += w * o[i];
                }
            }
            let sh = ffn(&sgv, &suv, &sdv);
            let gate = 1.0 / (1.0 + (-lg[ti * ls + ne]).exp());
            for i in 0..n {
                want_y[ti * n + i] += gate * sh[i];
            }
        }
        let want_ro: Vec<f32> = r.iter().zip(&want_y).map(|(a, b)| a + b).collect();
        let mut want_xo = vec![0f32; t * n];
        for ti in 0..t {
            let row = &want_ro[ti * n..(ti + 1) * n];
            let ss = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64;
            let inv = 1.0 / (ss + 1e-6).sqrt();
            for i in 0..n {
                want_xo[ti * n + i] = (row[i] as f64 * inv * wn[i] as f64) as f32;
            }
        }
        let rd = |i: usize| bufs._b[i].read::<f32>(0, t * n).unwrap();
        err = max_rel(&rd(bufs.y), &want_y)
            .max(max_rel(&rd(bufs.ro), &want_ro))
            .max(max_rel(&rd(bufs.xo), &want_xo));
        // xq_out: the GPU's x_out as codes then one f32 scale a 32
        let xo = rd(bufs.xo);
        let raw = bufs._b[13].read::<u8>(0, t * n + t * n / 8).unwrap();
        let mut bad = 0;
        for (bi, blk) in xo.chunks(32).enumerate() {
            let d = f32::from_le_bytes(raw[t * n + 4 * bi..t * n + 4 * bi + 4].try_into().unwrap());
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            if (d - amax / 127.0).abs() > 1e-6 * amax.max(1e-30) {
                bad += 1;
            }
            for (i, &v) in blk.iter().enumerate() {
                let q = raw[bi * 32 + i] as i8 as i32;
                if (q - (v / d).round() as i32).abs() > 1 {
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "xq_out disagrees with x_out in {bad} places");
    }
    for _ in 0..5 {
        op.launch(s, &bufs.ptrs).unwrap();
    }
    s.sync().unwrap();
    let mut best = f64::MAX;
    for _ in 0..20 {
        let (a, e) = (Event::new(true).unwrap(), Event::new(true).unwrap());
        a.record(s).unwrap();
        for _ in 0..10 {
            op.launch(s, &bufs.ptrs).unwrap();
        }
        e.record(s).unwrap();
        e.sync().unwrap();
        best = best.min(e.since(&a).unwrap() as f64 * 100.0);
    }
    if std::env::var_os("PER_KERNEL").is_some() {
        for i in 0..op.kernels() {
            let mut b = f64::MAX;
            for _ in 0..20 {
                let (a, e) = (Event::new(true).unwrap(), Event::new(true).unwrap());
                a.record(s).unwrap();
                op.launch_range(s, &bufs.ptrs, i..i + 1).unwrap();
                e.record(s).unwrap();
                e.sync().unwrap();
                b = b.min(e.since(&a).unwrap() as f64 * 1000.0);
            }
            eprintln!(
                "  T={t} ne={ne} {i:2} {:<10} {:?}: {b:.1} µs",
                op.name(i),
                op.shape(i)
            );
        }
    }
    println!(
        "| {t} | {ne} | {} | {best:.1} | {build:.1} s | {} |",
        op.kernels(),
        if check {
            format!("{err:.2e}")
        } else {
            "-".into()
        }
    );
}

fn main() {
    let ts: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let ts = if ts.is_empty() { vec![1, 2, 8, 16] } else { ts };
    let gpu = Gpu::new().expect("cuda");
    let s = Stream::new().unwrap();
    println!("| T | experts | kernels | µs a layer | build | max rel err vs reference |");
    println!("|---|---|---|---|---|---|");
    for &t in &ts {
        let base = MoeShape {
            t,
            n: 2048,
            ff: 512,
            ne: 16,
            k: 4,
            ls: 17,
            gu: Format::Q4K,
            down: Format::Q5K,
            eps_bits: 1e-6f64.to_bits(),
            keep_y: true,
            rows: t,
        };
        run(&gpu, &s, base, true);
        run(
            &gpu,
            &s,
            MoeShape {
                ne: 512,
                k: 10,
                ls: 513,
                ..base
            },
            false,
        );
    }
}
