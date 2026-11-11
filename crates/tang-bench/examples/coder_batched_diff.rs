//! Two lowerings of coder-next's batched step side by side on one GPU (no
//! memory plan: every node its own buffer), one step from zero caches,
//! the residual stream compared layer by layer: where a batched schedule
//! first departs from the plain one.
//!
//! `coder_batched_diff <file.gguf> [--batch B] [--ctx C]`; schedule A is the
//! plain batched step, B takes the `TANG_BATCH_*` env (`TANG_BATCH_GROUP`
//! etc.), read once per side via `A_`/`B_`-prefixed copies.
use std::path::Path;

use tang_bench::ir_coder_prefill::{derived_step_batched, prefill_graph};
use tang_bench::ir_coder_real::*;
use tang_bench::qwen_session::runtime;
use tang_compile::tensor::int8_sched::BlockOpts;
use tang_compile::tensor::step::StepInput;
use tang_gpu::program::Program;
use tang_llm::gguf::{GgmlType, Gguf};

fn words_of_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| {
            let mut a = [0u8; 4];
            a[..c.len()].copy_from_slice(c);
            f32::from_le_bytes(a)
        })
        .collect()
}

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let path = args.first().cloned().unwrap_or_else(gguf_path);
    let nb: usize = opt("--batch").map_or(1, |v| v.parse().unwrap());
    let ctx: usize = opt("--ctx").map_or(256, |v| v.parse().unwrap());
    let g = Gguf::open(Path::new(&path)).unwrap();
    let dims = dims_from_gguf(&g).unwrap();
    let pg = prefill_graph(&dims);
    let tys = bind(&pg, &g).unwrap();
    let st = derived_step_batched(&pg, ctx).unwrap();
    let rt = runtime();
    let plain = RealOpts {
        batch: Some(nb),
        group: 0,
        block: BlockOpts::default(),
    };
    let test = RealOpts {
        batch: Some(nb),
        group: env("TANG_BATCH_GROUP", 0),
        block: BlockOpts {
            pos_tile: env("TANG_BATCH_PT", 1),
            mma_min: env("TANG_BATCH_MMA_MIN", 0),
            ksplit: env("TANG_BATCH_KSPLIT", 2) as u32,
        },
    };
    let pos: Vec<f32> = vec![0.0; nb];
    let toks: Vec<f32> = (0..nb).map(|b| (151 + 7 * b) as f32).collect();
    let mut sides = Vec::new();
    for (side, ro) in [plain, test].into_iter().enumerate() {
        // `B_ENV="K=V,K2=V2"`: env set for side B's lowering only (a control)
        if side == 1 {
            if let Ok(e) = std::env::var("B_ENV") {
                for kv in e.split(',').filter(|s| !s.is_empty()) {
                    let (k, v) = kv.split_once('=').unwrap();
                    std::env::set_var(k, v);
                }
            }
        }
        let low = lower_real_step_with(&pg, &st, &tys, &ro);
        let empty: Vec<&[f32]> = vec![&[]; st.inputs.len()];
        let res = low.residual.clone();
        let out0 = low.outs[0];
        let names: Vec<String> = res
            .iter()
            .map(|&t| format!("{:?}", low.graph.nodes()[t.index()].shape))
            .collect();
        // the routers' top-k (expert, p) pairs: tuple comps over `[.., 2k]`
        let kk = 2 * dims.n_expert_used;
        let tops: Vec<tang_compile::tensor::TensorId> = (0..low.graph.nodes().len())
            .filter(|&i| {
                low.graph.comp_at(i).is_some_and(|c| {
                    c.tuple != 0 && c.out.last().and_then(|d| d.fixed()) == Some(kk)
                })
            })
            .map(tang_compile::tensor::TensorId::from_index)
            .collect();
        let prog = Program::new(&rt, low.program.program, &empty);
        for (k, si) in st.inputs.iter().enumerate() {
            match *si {
                StepInput::Same(i) => {
                    let t = g.info(&pg.inputs[i].name).unwrap();
                    let w = match (low.storage[k], t.ty) {
                        (Some(_), _) | (None, GgmlType::F32) => words_of_bytes(g.bytes(t)),
                        (None, _) => g.dequantize(t).unwrap(),
                    };
                    prog.write(&rt, prog.input(k), &w);
                }
                StepInput::Pos => prog.write(&rt, prog.input(k), &pos),
                StepInput::New(_) => prog.write(&rt, prog.input(k), &toks),
                StepInput::Cache(_) => {}
            }
        }
        prog.run(&rt);
        let r: Vec<Vec<f32>> = res.iter().map(|&t| prog.read(&rt, t)).collect();
        let l = prog.read(&rt, out0);
        let tk: Vec<Vec<f32>> = tops.iter().map(|&t| prog.read(&rt, t)).collect();
        sides.push((r, l, names, tk));
        drop(prog);
    }
    let rel = |a: &[f32], b: &[f32]| {
        let s = b.iter().fold(1e-30f64, |m, &x| m.max(x.abs() as f64));
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y) as f64).abs())
            .fold(0.0, f64::max)
            / s
    };
    let (a, b) = (&sides[0], &sides[1]);
    println!("residual nodes: {} vs {}", a.0.len(), b.0.len());
    for (i, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
        println!("  residual {i:3} {}: rel {:.2e}", a.2[i], rel(y, x));
    }
    println!("logits rel {:.2e}", rel(&b.1, &a.1));
    println!("routers: {} vs {}", a.3.len(), b.3.len());
    let k = dims.n_expert_used;
    for (l, (x, y)) in a.3.iter().zip(&b.3).enumerate() {
        for s in 0..x.len() / (2 * k) {
            let ids = |v: &[f32]| {
                let mut i: Vec<i64> = (0..k).map(|j| v[s * 2 * k + 2 * j] as i64).collect();
                i.sort();
                i
            };
            let (ix, iy) = (ids(x), ids(y));
            if ix != iy {
                let p = |v: &[f32], j: usize| v[s * 2 * k + 2 * j + 1];
                println!(
                    "  router {l} stream {s}: experts differ {ix:?} vs {iy:?}; p at the cut {:.6e} / {:.6e} (A), {:.6e} / {:.6e} (B)",
                    p(x, k - 2), p(x, k - 1), p(y, k - 2), p(y, k - 1)
                );
            }
        }
    }
}
