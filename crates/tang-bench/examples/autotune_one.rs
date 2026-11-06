//! Tune one problem `RUNS` times from scratch (no cache, no model) with
//! verbose output: the contenders and their spreads per kernel.
//! `cargo run --release -p tang-bench --example autotune_one -- bwd 128`
use tang_bench::ir_gpu::naive_attention_heads;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_gpu::autotune::{autotune, TuneOptions};
use tang_gpu::program::Runtime;

fn passes(g: TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, o) = g.stabilize_softmax(outs);
    let (g, o) = g.substitute_maps(&o);
    let (g, o) = g.eliminate_common(&o);
    let (g, o) = g.merge_siblings(&o);
    let (g, o) = g.relayout(&o);
    let (g, o) = g.merge_reduce_dims(&o);
    let (mut g, o) = g.split_reduce_dims(&o);
    g.schedule_auto(&o);
    (g, o)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map_or("bwd", |s| s.as_str());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let rt = Runtime::select().expect("gpu");
    let (g, o) = match kind {
        "mm" => {
            let mut g = TensorGraph::new();
            let a = g.input(&[n, n]);
            let b = g.input(&[n, n]);
            let c = g.matmul(a, b).unwrap();
            passes(g, &[c])
        }
        "fwd" => {
            let (g, y) = naive_attention_heads(8, 1, n, 64, 64, true);
            passes(g, &[y])
        }
        _ => {
            let (mut g, y) = naive_attention_heads(4, 1, n, 64, 64, true);
            let ins: Vec<TensorId> = (0..3).map(TensorId::from_index).collect();
            let dy = g.input(g.shape(y).to_vec().as_slice());
            let l = g.map(&[y, dy], |x| x[0] * x[1]).unwrap();
            let grads = g.backward(l, &ins).unwrap();
            passes(g, &grads)
        }
    };
    let prog = rt.lower(&g, &o);
    let mut rng = tang_bench::Rng::new(7);
    let inputs: Vec<Vec<f32>> = (0..prog.buffers.iter().filter(|b| b.input.is_some()).count())
        .map(|k| {
            let len = prog
                .buffers
                .iter()
                .find(|b| b.input == Some(k))
                .unwrap()
                .len;
            (0..len).map(|_| rng.f32() - 0.5).collect()
        })
        .collect();
    let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    for r in 0..runs {
        eprintln!("=== run {r}");
        let opts = TuneOptions {
            dir: None,
            verbose: true,
            ..Default::default()
        };
        autotune(&rt, g.clone(), &o, &ins, &opts);
    }
}
