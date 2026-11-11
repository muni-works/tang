//! DS pieces on a CUDA device (`--features cuda`, ignored tests): the
//! top-k block select against the interpreter and the generic combine,
//! and a host-mapped Engram-sized table gathered by the step.

use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_gpu::program::{Program, Runtime};

fn arg(src: TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View { gather: None, guard: false, forms },
    }
}

fn topk_graph(rows: usize, n: usize, k: usize, argmax: bool) -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let x = g.input(&[rows, n]);
    let j = g.iota(&[rows, n], 1);
    let m = if argmax { Monoid::argmax() } else { Monoid::topk(k) };
    let f = || vec![Form::unit(2, 0), Form::unit(2, 1)];
    let top = g
        .comp_tuple(&[rows, 2 * k], &[n], vec![arg(x, f()), arg(j, f())], Some(m), |v| {
            let mut o = vec![v[1], v[0]];
            for _ in 1..k {
                o.push(<tang_expr::ExprId as tang::Scalar>::from_f64(0.0));
                o.push(<tang_expr::ExprId as tang::Scalar>::from_f64(-(f32::MAX as f64)));
            }
            o
        })
        .unwrap();
    (g, top)
}

/// Top-512 over indexer-length rows and the vocab argmax: the block select
/// on the GPU equals the interpreter bit for bit; ms per launch against
/// the generic combine (`TANG_TOPK_SELECT=0`).
#[test]
#[ignore]
fn topk_select_on_gpu() {
    let rt = Runtime::cuda().expect("cuda");
    // (rows, row length, k): DS indexer rows at 8k / 64k context, the
    // 129280-vocab argmax over a verify window of 8
    for (rows, n, k, am) in [(1usize, 8192usize, 64usize, false), (1, 8192, 512, false), (1, 65536, 512, false), (8, 129280, 1, true)] {
        let (g, top) = topk_graph(rows, n, k, am);
        let vals: Vec<f64> = (0..rows * n)
            .map(|i| ((i.wrapping_mul(2654435761) >> 7) % 4093) as f64 * 0.125 - 256.0)
            .collect();
        let t0 = std::time::Instant::now();
        let want = g.eval(&[vals.clone()], &[top]).remove(0);
        eprintln!("  interpreter {:.1} s", t0.elapsed().as_secs_f64());
        let vf: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
        let mut ms = Vec::new();
        // the generic combine unrolls k² compare-selects: NVRTC takes
        // minutes past k = 64, so it is timed only up to there
        let modes: &[&str] = if k <= 64 { &["1", "0"] } else { &["1"] };
        for &select in modes {
            std::env::set_var("TANG_TOPK_SELECT", select);
            let dp = g.to_cuda_decode(&[top], &DecodeOpts::default());
            let src: String = dp.program.kernels.iter().map(|k| k.source.as_str()).collect();
            assert_eq!(src.contains("block select"), select == "1" && k >= 32);
            let p = Program::new(&rt, dp.program, &[&vf]);
            p.run(&rt);
            let got = p.read(&rt, top);
            let bad = got.iter().zip(&want).filter(|(a, b)| **a as f64 != **b).count();
            assert_eq!(bad, 0, "select={select} rows {rows} n {n} k {k}");
            ms.push(p.time(&rt, 20));
        }
        std::env::remove_var("TANG_TOPK_SELECT");
        match ms[..] {
            [a, b] => eprintln!("top-{k} over {rows} x {n}: kernel {a:.3} ms, generic combine {b:.3} ms ({:.1}x), both = interpreter", b / a),
            [a] => eprintln!("top-{k} over {rows} x {n}: kernel {a:.3} ms = interpreter"),
            _ => unreachable!(),
        }
    }
}

/// An Engram-sized table (`DS_ENGRAM_GB`, default 8 GB of f32) kept in
/// host-pinned, device-mapped memory, gathered by index rows on the GPU:
/// equals the interpreter-style CPU gather; ms per step vs the table in
/// device memory; device bytes held.
#[test]
#[ignore]
fn engram_host_mapped_gather() {
    let rt = Runtime::cuda().expect("cuda");
    let gb: usize = std::env::var("DS_ENGRAM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let d = 1024usize;
    let rows = gb * (1 << 30) / (4 * d);
    // a step: 8 tokens x 16 hashed heads of rows, summed per token
    let (t, heads) = (8usize, 16usize);
    let mut g = TensorGraph::new();
    let table = g.input(&[rows, d]);
    let idx = g.input(&[t * heads]);
    let picked = g.gather(table, 0, idx, false).unwrap();
    let out = g.map(&[picked], |v| v[0] * <tang_expr::ExprId as tang::Scalar>::from_f64(2.0)).unwrap();
    let tab: Vec<f32> = (0..rows * d).map(|i| ((i % 1009) as f32) * 0.001).collect();
    let ix: Vec<f32> = (0..t * heads).map(|i| ((i * 2654435761usize) % rows) as f32).collect();
    let want: Vec<f32> = ix
        .iter()
        .flat_map(|&r| tab[r as usize * d..(r as usize + 1) * d].iter().map(|x| x * 2.0))
        .collect();
    let dp = g.to_cuda_decode(&[out], &DecodeOpts::default());
    let host = Program::with_host(&rt, dp.program, &[&tab, &ix], &[0]);
    host.run(&rt);
    assert_eq!(host.read(&rt, out), want, "host-mapped gather");
    let mh = host.time(&rt, 50);
    eprintln!(
        "engram {gb} GB ({rows} x {d}) host-mapped: {} GB on host, {:.3} ms/step for {} rows ({:.1} MB read)",
        host.host_bytes() >> 30,
        mh,
        t * heads,
        (t * heads * d * 4) as f64 / 1e6
    );
    drop(host);
    let dp = g.to_cuda_decode(&[out], &DecodeOpts::default());
    let dev = Program::new(&rt, dp.program, &[&tab, &ix]);
    dev.run(&rt);
    assert_eq!(dev.read(&rt, out), want, "device gather");
    eprintln!("engram {gb} GB in device memory: {:.3} ms/step", dev.time(&rt, 50));
}

/// The derived DS step (tiny_full: Engram layers 1 and 4, ratio-1 KV,
/// candidate blocks; and a top-512 variant at 600 positions, where the
/// indexer runs the block select) as one CUDA decode program with the
/// Engram tables host-mapped, against the interpreter's step; ms per step
/// host-mapped vs device-resident.
#[test]
#[ignore]
fn ds_step_engram_host_mapped() {
    use crate::ir_ds_prefill::tests::step_case;
    use crate::ir_ds_prefill::DsDims;
    let rt = Runtime::cuda().expect("cuda");
    // ratio-1 layers 3..5 hold 600 compressed rows: the indexer drops some
    let topk = DsDims {
        index_topk: 512,
        ctx: 608,
        ..DsDims::tiny_full()
    };
    for (name, dm, p0) in [("tiny_full", DsDims::tiny_full(), 5usize), ("top-512", topk, 600)] {
        let (st, ins, ids, want, engram) = step_case(&dm, p0, 1);
        assert!(!engram.is_empty(), "{name}: no Engram input");
        let words: Vec<Vec<f32>> = ins
            .iter()
            .map(|v| v.iter().map(|&x| if x.is_nan() { 0.0 } else { x as f32 }).collect())
            .collect();
        let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
        let lower = || {
            let dp = st.graph.to_cuda_decode(&ids, &DecodeOpts::default());
            let src: String = dp.program.kernels.iter().map(|k| k.source.as_str()).collect();
            (dp, src.contains("block select"))
        };
        let (dp, select) = lower();
        let host = Program::with_host(&rt, dp.program, &refs, &engram);
        host.run(&rt);
        let mut worst = 0.0f64;
        for (t, w) in ids.iter().zip(&want) {
            let g = host.read(&rt, *t);
            let (a, b): (Vec<f64>, Vec<f64>) =
                g.iter().zip(w).filter(|(_, w)| !w.is_nan()).map(|(a, b)| (*a as f64, *b)).unzip();
            worst = worst.max(crate::ir_ds_prefill::tests::rel(&a, &b));
        }
        let mh = host.time(&rt, 20);
        let hb = host.host_bytes();
        drop(host);
        let dev = Program::new(&rt, lower().0.program, &refs);
        let md = dev.time(&rt, 20);
        eprintln!(
            "ds step {name} (pos {p0}): rel err {worst:.2e} vs interpreter, {} Engram inputs host-mapped ({hb} B), \
             block select {select}; {mh:.3} ms/step host-mapped, {md:.3} ms device-resident",
            engram.len()
        );
        assert!(worst < 1e-4, "{name}: {worst:e}");
        assert_eq!(select, dm.index_topk >= 32, "{name}: block select");
    }
}
