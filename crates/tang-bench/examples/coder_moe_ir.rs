//! One MoE layer of tang-llm's Qwen3-Coder-Next engine (80B-A3B, GGUF
//! UD-Q4_K_XL: hidden 2048, 512 experts + a shared one, top-10, expert FFN
//! 512) as one tensor-IR graph, its expert GEMMs on int8 tensor cores
//! through schedule fields — no new op:
//!
//! router GEMM → top-k (k argmax rounds) → softmax weights, sigmoid shared
//! gate → pairs sorted by expert (rank and inverse permutation as comps) →
//! `gu[i, r] = Σ_d Wgu[key[i], r, d] · xq[tok[i], d]` (gather views) →
//! SwiGLU → `dn[i, d] = Σ_f Wd[key[i], d, f] · h[i, f]` → weighted combine
//! reading `dn[rank[t, j]]`.
//!
//! The schedule: `Wgu` stored GGUF Q4_K and `Wd` Q5_K (the GGUF bytes as
//! they are), `xq` and `h` stored [`Format::Q8x32`] (int8 a 32, quantized
//! by their writers), and [`IntMma`] on `gu` and `dn`: m16n8k32 s8 MMAs
//! with the k-quant dequantization fused, activations in fragment order.
//!
//! Checks: interpreter (`eval_emulated`, the same int8 rounding) vs a
//! plain-Rust reference at a small size; GPU vs the reference at every
//! timed size. Timings per kernel; `CODER_MOE_DUMP=dir` writes the layer's
//! inputs and the derived outputs for `tang-llm --example coder_moe_hand`
//! (the hand kernels on the same bytes).
//!
//! `cargo run --release -p tang-bench --features cuda --example coder_moe_ir`
//! (`CODER_MOE_B=1,8,64`, `CODER_MOE_STAGE=mma` for the MMA k order,
//! `CODER_MOE_KSPLIT=gu,dn`). Without CUDA it runs the interpreter check only.

use tang_compile::tensor::{Arg, Form, Format, IntMma, Monoid, Precision, Stage, TensorGraph, TensorId, View};
use tang::Scalar;
use tang_expr::ExprId;

#[derive(Clone, Copy, Debug)]
struct Dims {
    t: usize,
    d: usize,
    e: usize,
    k: usize,
    f: usize,
}

impl Dims {
    fn k1(&self) -> usize {
        self.k + 1
    }
    fn p(&self) -> usize {
        self.t * self.k1()
    }
}

// ---------------------------------------------------------------------------
// GGUF k-quants: random valid super-blocks and their dequantization
// ---------------------------------------------------------------------------

fn f16_bits(x: f32) -> u16 {
    tang_compile::tensor::precision::f16_bits(x as f64)
}
fn f16_val(b: u16) -> f32 {
    tang_compile::tensor::precision::f16_from_bits(b) as f32
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

/// `rows × cols` weights as Q4_K (`five = false`) or Q5_K super-blocks with
/// random codes, scales and mins; values about `±amp`.
fn gen_kquant(rows: usize, cols: usize, five: bool, amp: f32, rng: &mut tang_bench::Rng) -> Vec<u8> {
    let bb = if five { 176 } else { 144 };
    let nb = rows * cols / 256;
    let mut out = vec![0u8; nb * bb];
    let qmax = if five { 31.0 } else { 15.0 };
    for b in out.chunks_mut(bb) {
        // value = d·sc·q − dmin·m, sc, m < 64: centre it with dmin·m ≈ d·sc·qmax/2
        let d = amp * (0.5 + rng.f32()) / (40.0 * qmax / 2.0);
        let dmin = d * qmax / 2.0 * (0.8 + 0.4 * rng.f32());
        b[0..2].copy_from_slice(&f16_bits(d).to_le_bytes());
        b[2..4].copy_from_slice(&f16_bits(dmin).to_le_bytes());
        for x in b[4..].iter_mut() {
            *x = (rng.f32() * 256.0) as u8;
        }
    }
    out
}

/// Dequantize `rows × cols` k-quant weights (ggml's dequantize_row_q{4,5}_K).
fn dequant(bytes: &[u8], five: bool) -> Vec<f32> {
    let bb = if five { 176 } else { 144 };
    let mut y = Vec::with_capacity(bytes.len() / bb * 256);
    for b in bytes.chunks(bb) {
        let d = f16_val(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_val(u16::from_le_bytes([b[2], b[3]]));
        let sc = &b[4..16];
        let (qh, qs) = if five { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
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

/// The Q8x32 rounding of the writers (f32 arithmetic): per 32, d = amax/127.
fn q8x32(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    for blk in out.chunks_mut(32) {
        let amax = blk.iter().fold(0f32, |a, x| a.max(x.abs()));
        let d = amax / 127.0;
        for x in blk.iter_mut() {
            *x = if d == 0.0 { 0.0 } else { (*x / d).round().clamp(-127.0, 127.0) * d };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

fn eq(a: ExprId, b: ExprId) -> ExprId {
    let (up, down) = (a - b, b - a);
    tang_expr::with_graph(|g| {
        let inner = g.select(down, ExprId::ZERO, ExprId::ONE);
        g.select(up, ExprId::ZERO, inner)
    })
}

fn arg(src: TensorId, view: View) -> Arg {
    Arg { src, view }
}

struct Layer {
    g: TensorGraph,
    y: TensorId,
    sel: TensorId,
    wts: TensorId,
    logits: TensorId,
    rank: TensorId,
    gu: TensorId,
    dn: TensorId,
    xq: TensorId,
    h: TensorId,
}

/// Inputs: x[T, D] f32, wr[E + 1, D] f32 (router rows, then the shared
/// expert's gate row), wgu[E + 1, 2F, D] (gate rows then up rows; expert E
/// is the shared one), wd[E + 1, D, F].
fn layer_graph(m: Dims) -> Layer {
    let Dims { t, d, e, k, f } = m;
    let (k1, p) = (m.k1(), m.p());
    let mut g = TensorGraph::new();
    let x = g.input(&[t, d]);
    let wr = g.input(&[e + 1, d]);
    let wgu = g.input(&[e + 1, 2 * f, d]);
    let wd = g.input(&[e + 1, d, f]);
    let sum = || Some(Monoid::sum());
    let logits = g
        .comp(
            &[t, e + 1],
            &[d],
            vec![arg(x, View::pick(3, &[Some(0), Some(2)])), arg(wr, View::pick(3, &[Some(1), Some(2)]))],
            sum(),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    // top-k over the routed logits: round j takes the argmax with earlier picks masked
    let iota_e = g.iota(&[t, e], 1);
    let mut picks: Vec<TensorId> = Vec::new();
    let slot = |j: i64| View {
        forms: vec![Form::unit(2, 0), Form::constant(2, j)],
        guard: false,
        gather: None,
    };
    for _ in 0..k {
        let mut args = vec![arg(logits, View::identity(2, 2)), arg(iota_e, View::identity(2, 2))];
        args.extend(picks.iter().map(|&q| arg(q, slot(0))));
        let n_prev = picks.len();
        let masked = g
            .comp(&[t, e], &[], args, None, |v| {
                let mut hit = ExprId::ZERO;
                for j in 0..n_prev {
                    hit = hit + eq(v[1], v[2 + j]);
                }
                vec![v[0] - hit * ExprId::from_f64(1e9)]
            })
            .unwrap();
        let am = g
            .comp_tuple(
                &[t, 2],
                &[e],
                vec![arg(iota_e, View::identity(2, 2)), arg(masked, View::identity(2, 2))],
                Some(Monoid::argmax()),
                |v| vec![v[0], v[1]],
            )
            .unwrap();
        picks.push(am);
    }
    // sel[t, j] (j = k: the shared expert, index e) and the combine weights
    // (softmax renormalized over the top k; the shared expert's sigmoid gate)
    let iota_k = g.iota(&[t, k1], 1);
    let gate_view = View {
        forms: vec![Form::unit(2, 0), Form::constant(2, e as i64)],
        guard: false,
        gather: None,
    };
    let stack = |g: &mut TensorGraph, weights: bool| {
        let mut args = vec![arg(iota_k, View::identity(2, 2)), arg(logits, gate_view.clone())];
        args.extend(picks.iter().map(|&q| arg(q, slot(if weights { 1 } else { 0 }))));
        g.comp(&[t, k1], &[], args, None, |v| {
            let mut acc = ExprId::ZERO;
            let mut norm = ExprId::ZERO;
            let vals: Vec<ExprId> = (0..k)
                .map(|j| {
                    if weights {
                        let ex = (v[2 + j] - v[2]).exp(); // v[2] is the max
                        norm = norm + ex;
                        ex
                    } else {
                        v[2 + j]
                    }
                })
                .collect();
            for (j, &val) in vals.iter().enumerate() {
                let val = if weights { val / norm } else { val };
                acc = acc + eq(v[0], ExprId::from_f64(j as f64)) * val;
            }
            let shared = if weights {
                ExprId::ONE / (ExprId::ONE + (-v[1]).exp())
            } else {
                ExprId::from_f64(e as f64)
            };
            vec![acc + eq(v[0], ExprId::from_f64(k as f64)) * shared]
        })
        .unwrap()
    };
    let sel = stack(&mut g, false);
    let wts = stack(&mut g, true);
    // sort the pairs by expert: key = sel·P + pair, rank = #keys below,
    // then the expert and token at each sorted position
    let iota_t = g.iota(&[t, k1], 0);
    let key = g
        .map(&[sel, iota_t, iota_k], |v| {
            v[0] * ExprId::from_f64(p as f64) + v[1] * ExprId::from_f64(k1 as f64) + v[2]
        })
        .unwrap();
    let rank = g
        .comp(
            &[t, k1],
            &[t, k1],
            vec![
                arg(key, View::pick(4, &[Some(0), Some(1)])),
                arg(key, View::pick(4, &[Some(2), Some(3)])),
            ],
            sum(),
            |v| {
                let diff = v[0] - v[1];
                vec![tang_expr::with_graph(|eg| eg.select(diff, ExprId::ONE, ExprId::ZERO))]
            },
        )
        .unwrap();
    let iota_p = g.iota(&[p], 0);
    let at_rank = |g: &mut TensorGraph, val: TensorId| {
        g.comp(
            &[p],
            &[t, k1],
            vec![
                arg(rank, View::pick(3, &[Some(1), Some(2)])),
                arg(iota_p, View::pick(3, &[Some(0)])),
                arg(val, View::pick(3, &[Some(1), Some(2)])),
            ],
            sum(),
            |v| vec![eq(v[0], v[1]) * v[2]],
        )
        .unwrap()
    };
    let skey = at_rank(&mut g, sel);
    let stok = at_rank(&mut g, iota_t);
    // the expert half
    let xq = g.map(&[x], |v| v[0]).unwrap();
    let gu = g
        .comp(
            &[p, 2 * f],
            &[d],
            vec![
                arg(wgu, View::pick(3, &[None, Some(1), Some(2)]).gather(0, 2, 1)),
                arg(xq, View::pick(3, &[None, Some(2)]).gather(0, 3, 1)),
                arg(skey, View::pick(3, &[Some(0)])),
                arg(stok, View::pick(3, &[Some(0)])),
            ],
            sum(),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let up = View {
        forms: vec![Form::unit(2, 0), Form { coefs: vec![0, 1], offset: f as i64 }],
        guard: false,
        gather: None,
    };
    let h = g
        .comp(
            &[p, f],
            &[],
            vec![arg(gu, View::identity(2, 2)), arg(gu, up)],
            None,
            |v| vec![v[0] / (ExprId::ONE + (-v[0]).exp()) * v[1]],
        )
        .unwrap();
    let dn = g
        .comp(
            &[p, d],
            &[f],
            vec![
                arg(wd, View::pick(3, &[None, Some(1), Some(2)]).gather(0, 2, 1)),
                arg(h, View::pick(3, &[Some(0), Some(2)])),
                arg(skey, View::pick(3, &[Some(0)])),
            ],
            sum(),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let y = g
        .comp(
            &[t, d],
            &[k1],
            vec![
                arg(wts, View::pick(3, &[Some(0), Some(2)])),
                arg(dn, View::pick(3, &[None, Some(1)]).gather(0, 2, 1)),
                arg(rank, View::pick(3, &[Some(0), Some(2)])),
            ],
            sum(),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    Layer { g, y, sel, wts, logits, rank, gu, dn, xq, h }
}

/// The schedule: storage formats and the int8 MMA fields.
fn schedule(l: &mut Layer, stage: Stage, ksplit: (u32, u32)) {
    let g = &mut l.g;
    g.set_precision(TensorId::from_index(2), Precision::storage(Format::Q4K));
    g.set_precision(TensorId::from_index(3), Precision::storage(Format::Q5K));
    g.set_precision(l.xq, Precision::storage(Format::Q8x32));
    g.set_precision(l.h, Precision::storage(Format::Q8x32));
    g.set_int_mma(l.gu, Some(IntMma { stage, ksplit: ksplit.0 }));
    g.set_int_mma(l.dn, Some(IntMma { stage, ksplit: ksplit.1 }));
    // the router: a 2048-long dot a point, split across a warp
    g.set_schedule(l.logits, 32, (0, 0, 0), (1, 1));
}

// ---------------------------------------------------------------------------
// Plain-Rust reference (f64 sums over the same int8-rounded activations)
// ---------------------------------------------------------------------------

struct Ref {
    y: Vec<f64>,
    gu: Vec<f64>, // [pair][2F], pair = t·K1 + j
    dn: Vec<f64>, // [pair][D]
}

/// Given the routing (sel, wts per pair), the expert half.
fn reference(m: Dims, x: &[f32], wgu: &[f32], wd: &[f32], sel: &[usize], wts: &[f64]) -> Ref {
    let Dims { t, d, f, .. } = m;
    let (k1, p) = (m.k1(), m.p());
    let xq = q8x32(x);
    let mut gu = vec![0.0; p * 2 * f];
    let mut dn = vec![0.0; p * d];
    let mut y = vec![0.0; t * d];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = p.div_ceil(threads);
    std::thread::scope(|sc| {
        for (c, (guc, dnc)) in gu.chunks_mut(per * 2 * f).zip(dn.chunks_mut(per * d)).enumerate() {
            let xq = &xq;
            sc.spawn(move || {
                for (i, (gup, dnp)) in guc.chunks_mut(2 * f).zip(dnc.chunks_mut(d)).enumerate() {
                    let pair = c * per + i;
                    let (ti, ex) = (pair / k1, sel[pair]);
                    for r in 0..2 * f {
                        let w = &wgu[(ex * 2 * f + r) * d..][..d];
                        gup[r] = (0..d).map(|j| w[j] as f64 * xq[ti * d + j] as f64).sum();
                    }
                    let hv: Vec<f32> = (0..f)
                        .map(|j| {
                            let (a, b) = (gup[j] as f32, gup[f + j] as f32);
                            a / (1.0 + (-a).exp()) * b
                        })
                        .collect();
                    let hq = q8x32(&hv);
                    for r in 0..d {
                        let w = &wd[(ex * d + r) * f..][..f];
                        dnp[r] = (0..f).map(|j| w[j] as f64 * hq[j] as f64).sum();
                    }
                }
            });
        }
    });
    for pair in 0..p {
        let ti = pair / k1;
        for r in 0..d {
            y[ti * d + r] += wts[pair] * dn[pair * d + r];
        }
    }
    Ref { y, gu, dn }
}

/// Top-k and weights from logits (f64), as the graph defines them.
fn route(m: Dims, logits: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let Dims { t, e, k, .. } = m;
    let (mut sel, mut wts) = (Vec::new(), Vec::new());
    for ti in 0..t {
        let l = &logits[ti * (e + 1)..(ti + 1) * (e + 1)];
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
        let ex: Vec<f64> = order[..k].iter().map(|&i| (l[i] - l[order[0]]).exp()).collect();
        let norm: f64 = ex.iter().sum();
        for j in 0..k {
            sel.push(order[j]);
            wts.push(ex[j] / norm);
        }
        sel.push(e);
        wts.push(1.0 / (1.0 + (-l[e]).exp()));
    }
    (sel, wts)
}

fn max_rel(got: impl Iterator<Item = f64> + Clone, want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-30);
    got.zip(want).map(|(g, w)| (g - w).abs() / scale).fold(0.0, f64::max)
}

struct Data {
    x: Vec<f32>,
    wr: Vec<f32>,
    wgu_q: Vec<u8>,
    wd_q: Vec<u8>,
    wgu: Vec<f32>,
    wd: Vec<f32>,
}

fn data(m: Dims, seed: u64) -> Data {
    let Dims { t, d, e, f, .. } = m;
    let mut rng = tang_bench::Rng::new(seed);
    let x: Vec<f32> = (0..t * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let wr: Vec<f32> = (0..(e + 1) * d).map(|_| (rng.f32() * 2.0 - 1.0) / (d as f32).sqrt()).collect();
    let wgu_q = gen_kquant((e + 1) * 2 * f, d, false, 2.0 / (d as f32).sqrt(), &mut rng);
    let wd_q = gen_kquant((e + 1) * d, f, true, 2.0 / (f as f32).sqrt(), &mut rng);
    let wgu = dequant(&wgu_q, false);
    let wd = dequant(&wd_q, true);
    Data { x, wr, wgu_q, wd_q, wgu, wd }
}

fn words(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn interpreter_check() {
    // small: T 3, D 256, E 6 (+ shared), K 2, F 256
    let m = Dims { t: 3, d: 256, e: 6, k: 2, f: 256 };
    let dt = data(m, 5);
    let mut l = layer_graph(m);
    schedule(&mut l, Stage::Fragment, (1, 1));
    let ins: Vec<Vec<f64>> = [&dt.x, &dt.wr, &dt.wgu, &dt.wd]
        .iter()
        .map(|v| v.iter().map(|&a| a as f64).collect())
        .collect();
    let ev = l.g.eval_emulated(&ins, &[l.y, l.sel, l.wts, l.logits, l.gu, l.rank]);
    let (sel, wts) = route(m, &ev[3]);
    let sel_ok = ev[1].iter().map(|&s| s as usize).eq(sel.iter().copied());
    let r = reference(m, &dt.x, &dt.wgu, &dt.wd, &sel, &wts);
    // the interpreter's gu is in sorted order: back through rank
    let rank: Vec<usize> = ev[5].iter().map(|&v| v as usize).collect();
    let f2 = 2 * m.f;
    let gu_pairs: Vec<f64> = (0..m.p()).flat_map(|q| ev[4][rank[q] * f2..][..f2].to_vec()).collect();
    println!(
        "interpreter (emulated int8) vs reference, {m:?}: top-k match {sel_ok}, gu max rel {:.2e}, y max rel {:.2e}",
        max_rel(gu_pairs.iter().copied(), &r.gu),
        max_rel(ev[0].iter().copied(), &r.y)
    );
    assert!(sel_ok && max_rel(ev[0].iter().copied(), &r.y) < 1e-4);
}

fn main() {
    interpreter_check();
    #[cfg(feature = "cuda")]
    gpu::run();
    #[cfg(not(feature = "cuda"))]
    println!("(built without --features cuda: GPU part skipped)");
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use tang_gpu::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    use tang_gpu::program::{Program, Runtime};

    /// Per launch of kernel `i`: (min of warm launches, median with L2
    /// flushed before each launch), ms.
    fn time(rt: &Runtime, exe: &Program, i: usize, flush: &mut tang_gpu::cudarc::driver::CudaSlice<u8>, iters: usize) -> (f64, f64) {
        let Runtime::Cuda(dev) = rt else { unreachable!() };
        let warm = exe.time_kernel(rt, i, iters);
        let mut v: Vec<f64> = (0..iters)
            .map(|_| {
                dev.stream.memset_zeros(flush).unwrap();
                let a = dev.stream.record_event(Some(CU_EVENT_DEFAULT)).unwrap();
                exe.run_kernel(rt, i);
                let b = dev.stream.record_event(Some(CU_EVENT_DEFAULT)).unwrap();
                b.synchronize().unwrap();
                a.elapsed_ms(&b).unwrap() as f64
            })
            .collect();
        v.sort_by(f64::total_cmp);
        (warm, v[v.len() / 2])
    }

    fn put(dir: &std::path::Path, name: &str, bytes: &[u8]) {
        std::fs::write(dir.join(name), bytes).unwrap();
    }
    fn f32s(v: impl Iterator<Item = f32>) -> Vec<u8> {
        v.flat_map(|x| x.to_le_bytes()).collect()
    }

    pub fn run() {
        let rt = Runtime::select().expect("gpu");
        println!("backend: {:?} — {}", rt.kind(), rt.name());
        let Runtime::Cuda(dev) = &rt else {
            println!("not CUDA: the int8 MMA lowering is CUDA-only");
            return;
        };
        let mut flush = dev.stream.alloc_zeros::<u8>(256 << 20).unwrap();
        let bs: Vec<usize> = std::env::var("CODER_MOE_B")
            .unwrap_or_else(|_| "1,8,64".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let stage = match std::env::var("CODER_MOE_STAGE").as_deref() {
            Ok("mma") => Stage::Mma,
            _ => Stage::Fragment,
        };
        let iters = 100;
        // one weight set (the largest B's routing reuses it)
        let full = Dims { t: 1, d: 2048, e: 512, k: 10, f: 512 };
        let dt = data(Dims { t: *bs.iter().max().unwrap(), ..full }, 7);
        let dump = std::env::var("CODER_MOE_DUMP").ok().map(std::path::PathBuf::from);
        for &b in &bs {
            let m = Dims { t: b, ..full };
            let ks: (u32, u32) = match std::env::var("CODER_MOE_KSPLIT") {
                Ok(s) => {
                    let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect();
                    (v[0], v[1])
                }
                Err(_) => (2, 1),
            };
            let mut l = layer_graph(m);
            schedule(&mut l, stage, ks);
            let outs = [l.y, l.sel, l.wts, l.logits, l.gu, l.dn, l.rank];
            let x = &dt.x[..b * m.d];
            let (wgu_w, wd_w) = (words(&dt.wgu_q), words(&dt.wd_q));
            let exe = Program::compile(&rt, &l.g, &outs, &[x, &dt.wr, &wgu_w, &wd_w]);
            exe.run(&rt);
            let rd = |t: TensorId| -> Vec<f64> { exe.read(&rt, t).iter().map(|&v| v as f64).collect() };
            let (y, gsel, gwts, logits, gu, dn, rank) =
                (rd(l.y), rd(l.sel), rd(l.wts), rd(l.logits), rd(l.gu), rd(l.dn), rd(l.rank));
            let (sel_r, _) = route(m, &logits);
            let sel: Vec<usize> = gsel.iter().map(|&s| s as usize).collect();
            let topk_ok = sel == sel_r;
            let r = reference(m, x, &dt.wgu, &dt.wd, &sel, &gwts);
            let rank: Vec<usize> = rank.iter().map(|&v| v as usize).collect();
            let (f2, d) = (2 * m.f, m.d);
            let unsort = |v: &[f64], w: usize| -> Vec<f64> {
                (0..m.p()).flat_map(|q| v[rank[q] * w..][..w].to_vec()).collect()
            };
            let (gu_p, dn_p) = (unsort(&gu, f2), unsort(&dn, d));
            println!(
                "\nB={b} {stage:?} ksplit {ks:?}: top-k vs f64 routing of the GPU logits {topk_ok}; gpu vs reference max rel: gu {:.2e}, dn {:.2e}, y {:.2e}",
                max_rel(gu_p.iter().copied(), &r.gu),
                max_rel(dn_p.iter().copied(), &r.dn),
                max_rel(y.iter().copied(), &r.y)
            );
            // per kernel
            let mut expert = (0.0, 0.0);
            let mut total = (0.0, 0.0);
            for (i, k) in exe.program.kernels.iter().enumerate() {
                let (w, c) = time(&rt, &exe, i, &mut flush, iters);
                let n = k.node;
                let name = if n == l.gu {
                    "gate|up (int8 MMA)"
                } else if n == l.dn {
                    "down (int8 MMA)"
                } else if n == l.h {
                    "SwiGLU → Q8x32"
                } else if n == l.xq {
                    "x → Q8x32"
                } else if n == l.y {
                    "combine"
                } else if n == l.logits {
                    "router"
                } else {
                    ""
                };
                total.0 += w;
                total.1 += c;
                if !name.is_empty() && name != "router" {
                    expert.0 += w;
                    expert.1 += c;
                }
                println!("  k{i:2} {:>8.4} ms warm  {:>8.4} ms cold  {name}", w, c);
            }
            println!(
                "  expert half (x→Q8x32, gate|up, SwiGLU, down, combine): {:.4} warm / {:.4} cold ms; all {} kernels {:.4} / {:.4}",
                expert.0,
                expert.1,
                exe.program.kernels.len(),
                total.0,
                total.1
            );
            if let Some(dir) = &dump {
                let dir = dir.join(format!("b{b}"));
                std::fs::create_dir_all(&dir).unwrap();
                put(&dir, "meta.txt", format!("{} {} {} {} {}\n", m.t, m.d, m.e, m.k, m.f).as_bytes());
                put(&dir, "x.f32", &f32s(x.iter().copied()));
                put(&dir, "logits.f32", &f32s(logits.iter().map(|&v| v as f32)));
                // routed pairs only, the hand kernels' ids / wts layout [t][K]
                let routed = |v: &[f64]| -> Vec<f64> {
                    (0..m.p()).filter(|q| q % m.k1() < m.k).map(|q| v[q]).collect()
                };
                let ids: Vec<u8> = routed(&gsel).iter().flat_map(|&s| (s as u32).to_le_bytes()).collect();
                put(&dir, "ids.u32", &ids);
                put(&dir, "wts.f32", &f32s(routed(&gwts).iter().map(|&v| v as f32)));
                put(&dir, "wgu.q4k", &dt.wgu_q);
                put(&dir, "wd.q5k", &dt.wd_q);
                // derived outputs in the hand kernels' pair order: routed t·K + k, then shared T·K + t
                let hand_order = |v: &[f64], w: usize| -> Vec<f32> {
                    let mut o = Vec::with_capacity(m.p() * w);
                    for q in (0..m.p()).filter(|q| q % m.k1() < m.k).chain((0..m.t).map(|t| t * m.k1() + m.k)) {
                        o.extend(v[q * w..][..w].iter().map(|&a| a as f32));
                    }
                    o
                };
                put(&dir, "ir_gu.f32", &f32s(hand_order(&gu_p, f2).into_iter()));
                put(&dir, "ir_dn.f32", &f32s(hand_order(&dn_p, d).into_iter()));
                put(&dir, "ir_y.f32", &f32s(y.iter().map(|&v| v as f32)));
                put(&dir, "ref_y.f32", &f32s(r.y.iter().map(|&v| v as f32)));
                println!("  wrote {}", dir.display());
            }
        }
    }
}
