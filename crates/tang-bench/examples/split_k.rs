//! Split-K for monoid reductions, timed: batch-1 decode attention (one
//! query row per head against a long KV cache) and a long-K matmul, each
//! unsplit and split into P chunks by `TensorGraph::split_reduction` — the
//! partial states folded by the monoid's own combine (online softmax's
//! rescale-and-add for attention, `+` for the matmul). The fastest P per
//! length is the autotuned pick.
//!
//! ```text
//! cargo run --release -p tang-bench --example split_k [--features cuda] -- [attn|matmul|show] [kv…]
//! ```
//! `TANG_BACKEND=wgpu|cuda` picks the runtime (CUDA when built with it).
use tang::Scalar;
use tang_compile::tensor::{Arg, Monoid, MonoidKind, Op, TensorGraph, TensorId, View};
use tang_expr::ExprId;
use tang_gpu::program::{Program, Runtime};

/// q `[hk, g, 1, d]`, k `[hk, n, d]`, v `[hk, n, dv]` → `[hk, g, 1, dv]`,
/// naive, then the fusion pipeline (one online-softmax reduction).
fn decode_attention(
    hk: usize,
    gq: usize,
    n: usize,
    d: usize,
    dv: usize,
) -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let q = g.input(&[hk, gq, 1, d]);
    let k = g.input(&[hk, n, d]);
    let v = g.input(&[hk, n, dv]);
    let scale = 1.0 / (d as f64).sqrt();
    let qs = g.map(&[q], |x| x[0] * ExprId::from_f64(scale)).unwrap();
    let s = g
        .comp(
            &[hk, gq, 1, n],
            &[d],
            vec![
                Arg {
                    src: qs,
                    view: View::pick(5, &[Some(0), Some(1), Some(2), Some(4)]),
                },
                Arg {
                    src: k,
                    view: View::pick(5, &[Some(0), Some(3), Some(4)]),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[1]],
        )
        .unwrap();
    let m = g.reduce(s, 3, Monoid::max()).unwrap();
    let mb = g.broadcast(m, &[hk, gq, 1, n]).unwrap();
    let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
    let l = g.reduce(p, 3, Monoid::sum()).unwrap();
    let o = g
        .comp(
            &[hk, gq, 1, dv],
            &[n],
            vec![
                Arg {
                    src: p,
                    view: View::pick(5, &[Some(0), Some(1), Some(2), Some(4)]),
                },
                Arg {
                    src: v,
                    view: View::pick(5, &[Some(0), Some(4), Some(3)]),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![x[0] * x[1]],
        )
        .unwrap();
    let lb = g.broadcast(l, &[hk, gq, 1, dv]).unwrap();
    let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
    let (g, outs) = g.stabilize_softmax(&[y]);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (mut g, outs) = g.merge_reduce_dims(&outs);
    g.schedule_inline_up_to(&outs, dv);
    (g, outs[0])
}

/// The live reduction with the longest reduce axis (the one to split).
fn longest_reduction(g: &TensorGraph, y: TensorId) -> TensorId {
    let (g2, _) = g.eliminate_dead(&[y]);
    assert_eq!(g2.nodes().len(), g.nodes().len(), "graph has dead nodes");
    (0..g.nodes().len())
        .filter_map(|i| match &g.nodes()[i].op {
            Op::Comp(c) if c.monoid.is_some() && c.red.len() == 1 => Some((i, c.red[0].n())),
            _ => None,
        })
        .max_by_key(|&(_, n)| n)
        .map(|(i, _)| TensorId::from_index(i))
        .expect("a reduction")
}

fn show(g: &TensorGraph, y: TensorId, label: &str) {
    println!("-- {label}");
    let (g, _) = g.eliminate_dead(&[y]);
    for (i, nd) in g.nodes().iter().enumerate() {
        if let Op::Comp(c) = &nd.op {
            let kind = c.monoid.as_ref().map(Monoid::kind);
            println!(
                "  %{i} out {:?} tuple {} red {:?} {:?} inline {} split {} tile {:?} reg {:?}",
                c.out.iter().map(|d| d.n()).collect::<Vec<_>>(),
                c.tuple,
                c.red.iter().map(|d| d.n()).collect::<Vec<_>>(),
                kind,
                nd.inline,
                nd.split,
                nd.tile,
                nd.reg
            );
        }
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

/// f64 decode attention on the host.
fn attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hk: usize,
    gq: usize,
    n: usize,
    d: usize,
) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0; hk * gq * d];
    for h in 0..hk {
        for gi in 0..gq {
            let qr = &q[(h * gq + gi) * d..][..d];
            let s: Vec<f64> = (0..n)
                .map(|j| {
                    let kr = &k[(h * n + j) * d..][..d];
                    qr.iter()
                        .zip(kr)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let l: f64 = p.iter().sum();
            let o = &mut out[(h * gq + gi) * d..][..d];
            for (j, pj) in p.iter().enumerate() {
                let vr = &v[(h * n + j) * d..][..d];
                for c in 0..d {
                    o[c] += pj * vr[c] as f64 / l;
                }
            }
        }
    }
    out
}

/// Max abs error relative to the largest reference value.
fn max_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0, f64::max)
        / scale
}

/// Time `g` (ms per run) and check it against `want`.
fn run(
    rt: &Runtime,
    g: &TensorGraph,
    y: TensorId,
    inputs: &[&[f32]],
    want: &[f64],
    iters: usize,
) -> (f64, f64) {
    let prog = Program::compile(rt, g, &[y], inputs);
    prog.run(rt);
    let err = max_err(&prog.read(rt, y), want);
    (prog.time(rt, iters), err)
}

/// A schedule to time: graph, output, label.
type Variant = (TensorGraph, TensorId, String);

/// The split graph's partial: the reduction whose out gained the chunk axis.
fn partial_of(g: &TensorGraph, y: TensorId, rank: usize) -> Option<TensorId> {
    let (g2, _) = g.eliminate_dead(&[y]);
    debug_assert_eq!(g2.nodes().len(), g.nodes().len());
    (0..g.nodes().len())
        .find(|&i| matches!(&g.nodes()[i].op, Op::Comp(c) if c.monoid.is_some() && c.out.len() == rank + 1))
        .map(TensorId::from_index)
}

/// Unsplit, then every split-K candidate × the partial's tile shapes;
/// prints the best tile per P and the overall pick.
#[allow(clippy::too_many_arguments)]
fn sweep(
    rt: &Runtime,
    label: &str,
    g: &TensorGraph,
    y: TensorId,
    inputs: &[&[f32]],
    want: &[f64],
    bytes: f64,
    tol: f64,
) {
    let red = longest_reduction(g, y);
    let rank = g.shape(red).len();
    let mut variants: Vec<(usize, Vec<Variant>)> = Vec::new();
    let mut base = g.clone();
    base.schedule_auto(&[y]);
    variants.push((1, vec![(base, y, "auto".into())]));
    for p in g.split_k_candidates(red, 32, 1 << 20) {
        let (mut sg, so) = g.split_reduction(&[y], red, p).unwrap();
        sg.schedule_auto(&so);
        let mut vs = vec![(sg.clone(), so[0], "auto".to_string())];
        if let Some(part) = partial_of(&sg, so[0], rank) {
            let mut cands = sg.schedule_candidates(part);
            let n0 = sg.nodes()[part.index()].tile;
            if n0.0 > 0 {
                for by in [1, 2, 4, 8].into_iter().filter(|b| n0.0 * b <= 256) {
                    for ch in [32, 64] {
                        cands.push((0, (n0.0, by, ch), (1, 1)));
                    }
                }
            }
            for (split, tile, reg) in cands {
                let mut v = sg.clone();
                v.set_schedule(part, split, tile, reg);
                vs.push((v, so[0], format!("split {split} tile {tile:?} reg {reg:?}")));
            }
        }
        variants.push((p, vs));
    }
    let iters = 50;
    let mut best = (f64::INFINITY, 0usize, String::new());
    let mut unsplit = 0.0;
    for (p, vs) in &variants {
        let mut pbest = (f64::INFINITY, String::new(), 0.0);
        for (vg, vy, name) in vs {
            let (ms, err) = run(rt, vg, *vy, inputs, want, iters);
            assert!(err < tol, "{label} P={p} {name}: err {err}");
            if ms < pbest.0 {
                pbest = (ms, name.clone(), err);
            }
        }
        let (ms, name, err) = pbest;
        if *p == 1 {
            unsplit = ms;
        }
        if ms < best.0 {
            best = (ms, *p, name.clone());
        }
        println!(
            "{label:>28}  P={p:<4} {:>9.1} us  {:>6.0} GB/s  rel err {err:.1e}  [{name}]",
            ms * 1e3,
            bytes / (ms * 1e-3) / 1e9
        );
    }
    // the untuned rule: fill 2^16 points
    let (mut rg, ro) = g.schedule_split_k(&[y], 1 << 16);
    rg.schedule_auto(&ro);
    let (ms, err) = run(rt, &rg, ro[0], inputs, want, iters);
    assert!(err < tol, "{label} rule: err {err}");
    let rp = partial_of(&rg, ro[0], rank).map_or(1, |p| rg.shape(p)[0].n());
    println!(
        "{label:>28}  rule   P={rp:<4} {:>6.1} us, {:.2}x vs unsplit",
        ms * 1e3,
        unsplit / ms
    );
    println!(
        "{label:>28}  autotuned P={} [{}] {:.1} us, {:.2}x vs unsplit",
        best.1,
        best.2,
        best.0 * 1e3,
        unsplit / best.0
    );
}

/// tang-llm's hand decode attention (`cx_attn_part` + `cx_attn_merge` in
/// `flash/coder_kernels.rs`), ported to f32 caches in this example's
/// head-major layout (k `[hk][n][d]`) with the gate and causal/tree
/// offsets dropped: block (group g, chunk j) runs the group's G query
/// heads over CHUNK keys with 4 warps, the merge folds the chunks.
#[cfg(feature = "cuda")]
fn hand_attention(
    n: usize,
    chunk: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    want: &[f64],
) -> (f64, f64) {
    use tang_gpu::cuda::{
        cudarc::driver::{LaunchConfig, PushKernelArg},
        CudaDevice,
    };
    const SRC: &str = r#"
#define NEG_INF_F __int_as_float(0xff800000)
#define DL (HD / 32)
extern "C" __global__ void __launch_bounds__(128) main0(const float* q, const float* kc, const float* vc,
                                                       float* part, int n, int nchunk) {
    __shared__ float wm[4][G], wl[4][G];
    __shared__ float wacc[4][HD];
    const int g = blockIdx.y, j = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* out = part + ((size_t)g * nchunk + j) * G * (HD + 2);
    const int lo = j * CHUNK;
    const float scale = 1.0f / sqrtf((float)HD);
    float qv[G][DL];
    for (int h = 0; h < G; h++)
        for (int i = 0; i < DL; i++) qv[h][i] = q[((size_t)g * G + h) * HD + lane * DL + i] * scale;
    float m[G], l[G], acc[G][DL];
    for (int h = 0; h < G; h++) {
        m[h] = NEG_INF_F; l[h] = 0.f;
        for (int i = 0; i < DL; i++) acc[h][i] = 0.f;
    }
    const int hi = min(n, lo + CHUNK);
    for (int p = lo + warp; p < hi; p += 4) {
        const float* kr = kc + ((size_t)g * n + p) * HD + lane * DL;
        const float* vr = vc + ((size_t)g * n + p) * HD + lane * DL;
        float kv[DL], vv[DL];
        #pragma unroll
        for (int i = 0; i < DL; i++) { kv[i] = kr[i]; vv[i] = vr[i]; }
        #pragma unroll
        for (int h = 0; h < G; h++) {
            float sc = 0.f;
            #pragma unroll
            for (int i = 0; i < DL; i++) sc = fmaf(qv[h][i], kv[i], sc);
            for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o);
            const float mn = fmaxf(m[h], sc), corr = expf(m[h] - mn), pr = expf(sc - mn);
            #pragma unroll
            for (int i = 0; i < DL; i++) acc[h][i] = fmaf(pr, vv[i], acc[h][i] * corr);
            l[h] = l[h] * corr + pr;
            m[h] = mn;
        }
    }
    for (int h = 0; h < G; h++)
        if (lane == 0) { wm[warp][h] = m[h]; wl[warp][h] = l[h]; }
    for (int h = 0; h < G; h++) {
        __syncthreads();
        for (int i = 0; i < DL; i++) wacc[warp][lane * DL + i] = acc[h][i];
        __syncthreads();
        float M = NEG_INF_F;
        for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) M = fmaxf(M, wm[w][h]);
        float L = 0.f;
        for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) L += wl[w][h] * expf(wm[w][h] - M);
        for (int d = threadIdx.x; d < HD; d += 128) {
            float o = 0.f;
            for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) o += wacc[w][d] * expf(wm[w][h] - M);
            out[h * (HD + 2) + 2 + d] = o;
        }
        if (threadIdx.x == 0) { out[h * (HD + 2)] = M; out[h * (HD + 2) + 1] = L; }
    }
}
"#;
    const MERGE: &str = r#"
#define NEG_INF_F __int_as_float(0xff800000)
extern "C" __global__ void main0(const float* part, int nchunk, float* out) {
    const int h = blockIdx.x, g = h / G, hh = h % G, d = threadIdx.x;
    const size_t hs = (size_t)G * (HD + 2);
    const float* pp = part + (size_t)g * nchunk * hs + hh * (HD + 2);
    float M = NEG_INF_F;
    for (int j = 0; j < nchunk; j++) if (pp[j * hs + 1] > 0.f) M = fmaxf(M, pp[j * hs]);
    float L = 0.f, o = 0.f;
    for (int j = 0; j < nchunk; j++) {
        const float* qq = pp + j * hs;
        if (qq[1] > 0.f) { const float f = expf(qq[0] - M); L += qq[1] * f; o += qq[2 + d] * f; }
    }
    out[(size_t)h * HD + d] = o / L;
}
"#;
    let (hk, gq, hd) = (8usize, 4usize, 128usize);
    let defs = format!("#define G {gq}\n#define HD {hd}\n#define CHUNK {chunk}\n");
    let dev = CudaDevice::new(0).expect("cuda");
    let part_f = dev.function(&format!("{defs}{SRC}"));
    let merge_f = dev.function(&format!("{defs}{MERGE}"));
    let s = &dev.stream;
    let nchunk = n.div_ceil(chunk);
    let (dq, dk, dv) = (
        s.memcpy_stod(q).unwrap(),
        s.memcpy_stod(k).unwrap(),
        s.memcpy_stod(v).unwrap(),
    );
    let mut part = s.alloc_zeros::<f32>(hk * nchunk * gq * (hd + 2)).unwrap();
    let mut out = s.alloc_zeros::<f32>(hk * gq * hd).unwrap();
    let (n_i, nc_i) = (n as i32, nchunk as i32);
    let mut launch = || unsafe {
        s.launch_builder(&part_f)
            .arg(&dq)
            .arg(&dk)
            .arg(&dv)
            .arg(&mut part)
            .arg(&n_i)
            .arg(&nc_i)
            .launch(LaunchConfig {
                grid_dim: (nchunk as u32, hk as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        s.launch_builder(&merge_f)
            .arg(&part)
            .arg(&nc_i)
            .arg(&mut out)
            .launch(LaunchConfig {
                grid_dim: ((hk * gq) as u32, 1, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    };
    launch();
    dev.sync();
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..50 {
            launch();
        }
        dev.sync();
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / 50.0);
    }
    let err = max_err(&s.memcpy_dtov(&out).unwrap(), want);
    (best, err)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("attn");
    let lens: Vec<usize> = args.iter().skip(1).filter_map(|s| s.parse().ok()).collect();
    let lens = if lens.is_empty() {
        vec![1024, 4096, 16384, 32768]
    } else {
        lens
    };
    // Llama-3-8B decode: 8 kv heads, 4 query heads each, head dim 128
    let (hk, gq, d) = (8, 4, 128);
    if mode == "show" {
        let n = lens[0];
        let (g, y) = decode_attention(hk, gq, n, d, d);
        let mut a = g.clone();
        a.schedule_auto(&[y]);
        show(&a, y, "unsplit");
        let red = longest_reduction(&g, y);
        assert_eq!(
            g.comp_at(red.index())
                .unwrap()
                .monoid
                .as_ref()
                .unwrap()
                .kind(),
            MonoidKind::OnlineSoftmax
        );
        let (mut sg, so) = g.split_reduction(&[y], red, 8).unwrap();
        sg.schedule_auto(&so);
        show(&sg, so[0], "split 8");
        if std::env::var("DUMP").is_ok() {
            for k in sg.to_cuda(&so).kernels {
                println!("// kernel node {}\n{}", k.node.index(), k.source);
            }
        }
        return;
    }
    let rt = Runtime::select().expect("gpu");
    println!("runtime: {}", rt.name());
    let mut rng = Rng(7);
    if mode == "prefill" {
        // regression check for the cooperative staged loads: causal
        // prefill attention, schedule_auto (TANG_LOAD_LANES=1: old load)
        for &n in &lens {
            let (mut g, y) = tang_bench::ir_gpu::fused_attention_heads(hk, gq, n, d, d, true);
            g.schedule_auto(&[y]);
            let q = rng.vec(hk * gq * n * d);
            let k = rng.vec(hk * n * d);
            let v = rng.vec(hk * n * d);
            let prog = Program::compile(&rt, &g, &[y], &[&q, &k, &v]);
            println!("prefill causal n={n}: {:.1} us", prog.time(&rt, 10) * 1e3);
        }
        return;
    }
    if mode == "attn" {
        for &n in &lens {
            let q = rng.vec(hk * gq * d);
            let k = rng.vec(hk * n * d);
            let v = rng.vec(hk * n * d);
            let want = attention_ref(&q, &k, &v, hk, gq, n, d);
            let (g, y) = decode_attention(hk, gq, n, d, d);
            let bytes = 4.0 * (2 * hk * n * d) as f64;
            sweep(
                &rt,
                &format!("decode attn kv={n}"),
                &g,
                y,
                &[&q, &k, &v],
                &want,
                bytes,
                1e-4,
            );
            #[cfg(feature = "cuda")]
            if matches!(rt, Runtime::Cuda(_)) {
                for chunk in [128, 256, 512] {
                    let (ms, err) = hand_attention(n, chunk, &q, &k, &v, &want);
                    println!(
                        "{:>28}  chunk {chunk:<4} {:>6.1} us  {:>6.0} GB/s  rel err {err:.1e}",
                        format!("hand cx_attn kv={n}"),
                        ms * 1e3,
                        bytes / (ms * 1e-3) / 1e9
                    );
                }
            }
        }
    } else {
        // long-K matmul: [m, K] · [K, n]
        for &kk in &lens {
            let (m, nn) = (16, 256);
            let mut g = TensorGraph::new();
            let a = g.input(&[m, kk]);
            let b = g.input(&[kk, nn]);
            let c = g.matmul(a, b).unwrap();
            let av = rng.vec(m * kk);
            let bv = rng.vec(kk * nn);
            let mut want = vec![0.0f64; m * nn];
            for i in 0..m {
                for p in 0..kk {
                    let x = av[i * kk + p] as f64;
                    for j in 0..nn {
                        want[i * nn + j] += x * bv[p * nn + j] as f64;
                    }
                }
            }
            let bytes = 4.0 * (m * kk + kk * nn) as f64;
            // CUDA lowers matmul-shaped tiles to TF32 tensor cores
            sweep(
                &rt,
                &format!("matmul {m}x{kk}x{nn}"),
                &g,
                c,
                &[&av, &bv],
                &want,
                bytes,
                2e-3,
            );
        }
    }
}
