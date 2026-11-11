//! The derived steps of the four tiny prefill graphs (Qwen3, coder-next,
//! DeepSeek V4.1, GLM-5.3; `tang_bench::derive_all`) on the GPU against the
//! interpreter: prefill 5 tokens on the CPU, seed the caches (NaN in
//! unwritten rows), run one derived step of `k` tokens on the device, and
//! compare its logits and updated caches with the interpreter's step.
//!
//! ```text
//! derive_tiny_gpu [k…]      default: 1 4
//! ```
//!
//! Prints one line per model and `k` and exits 1 if any disagrees by more
//! than 1e-3 of the output's scale (the device runs f32, the reference
//! f64), after running every check.
use tang_bench::derive_all::*;
use tang_bench::qwen_session::{lower, runtime};
use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_compile::tensor::step::{Rows, StepInput};
use tang_compile::tensor::TensorId;
use tang_gpu::program::Program;

fn main() {
    let ks: Vec<usize> = {
        let v: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|a| a.parse().ok())
            .collect();
        if v.is_empty() {
            vec![1, 4]
        } else {
            v
        }
    };
    let rt = runtime();
    eprintln!("runtime: {}", rt.name());
    let p0 = 5;
    let mut failed = 0;
    for m in all_models() {
        let w = weights(&m);
        let toks: Vec<f64> = (0..p0 + 8)
            .map(|i| ((37 * i + 5) % vocab(&m)) as f64)
            .collect();
        let ins = |t: &[f64]| -> Vec<Vec<f64>> {
            m.inputs
                .iter()
                .map(|(n, _)| {
                    if n == "tokens" {
                        t.to_vec()
                    } else {
                        w[n].clone()
                    }
                })
                .collect()
        };
        for &k in &ks {
            let st = match derive(&m, k, Rows::All) {
                Ok(st) => st,
                Err(e) => {
                    println!("{} k={k}: refused: {e}", m.name);
                    failed += 1;
                    continue;
                }
            };
            let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
            let pre =
                m.g.bind(&[("T", p0)])
                    .unwrap()
                    .eval(&ins(&toks[..p0]), &nodes);
            let caches: Vec<Vec<f64>> = st
                .caches
                .iter()
                .zip(pre)
                .map(|(c, v)| seed(&m, c, v, p0))
                .collect();
            let step_ins: Vec<Vec<f64>> = st
                .inputs
                .iter()
                .map(|&x| match x {
                    StepInput::Pos => vec![p0 as f64],
                    StepInput::New(_) => toks[p0..p0 + k].to_vec(),
                    StepInput::Same(i) => w[&m.inputs[i].0].clone(),
                    StepInput::Cache(c) => caches[c].clone(),
                })
                .collect();
            let mut ids = st.outputs.clone();
            ids.extend(st.caches.iter().map(|c| c.updated));
            let want = st.graph.eval(&step_ins, &ids);

            // the decode lowering (no in-place: every output its own buffer)
            let (g, outs) = st.graph.substitute_maps(&ids);
            let (g, outs) = g.eliminate_common(&outs);
            let dp = lower(&rt, &g, &outs, &DecodeOpts::default());
            let native = dp.native.iter().filter(|n| **n).count();
            let nk = dp.program.kernels.len();
            let words: Vec<Vec<f32>> = step_ins
                .iter()
                .map(|v| v.iter().map(|&x| x as f32).collect())
                .collect();
            let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
            let prog = Program::new(&rt, dp.program, &refs);
            prog.run(&rt);
            let mut worst = 0.0f64;
            for (o, wv) in outs.iter().zip(&want) {
                let got: Vec<f64> = prog.read(&rt, *o).iter().map(|&x| x as f64).collect();
                // NaN where the reference has NaN (unwritten cache rows)
                let (gk, wk): (Vec<f64>, Vec<f64>) = got
                    .iter()
                    .zip(wv)
                    .filter(|(_, w)| !w.is_nan())
                    .map(|(a, b)| (*a, *b))
                    .unzip();
                worst = worst.max(rel(&gk, &wk));
            }
            let ms = prog.time(&rt, 20);
            let ok = worst < 1e-3;
            failed += usize::from(!ok);
            println!(
                "{} k={k}: {nk} kernels ({native} native), GPU vs interpreter rel err {worst:.2e}, \
                 {ms:.3} ms device: {}",
                m.name,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
