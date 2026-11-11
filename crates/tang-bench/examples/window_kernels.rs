//! The derived Qwen3-8B step's CUDA kernels at window k = 1 and k = K, no
//! device needed: per kernel its node shape, launch, lowering (first source
//! line) and a byte estimate of what it streams (its reads' sizes), to see
//! where a window costs more than a step.
//!
//! ```text
//! window_kernels [K] [--ctx 4352] [--rows last]
//! ```
use tang_bench::ir_qwen::prefill::{as_decode_graph, derived_step_rows, prefill_graph};
use tang_bench::ir_qwen::*;
use tang_compile::tensor::step::Rows;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let k: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(4);
    let opt = |n: &str| {
        args.iter()
            .position(|a| a == n)
            .map(|i| args[i + 1].clone())
    };
    let ctx: usize = opt("--ctx").map_or(4352, |v| v.parse().unwrap());
    let rows = if opt("--rows").as_deref() == Some("last") {
        Rows::Last
    } else {
        Rows::All
    };
    let dims = QwenDims::qwen3_8b(ctx);
    let pg = prefill_graph(&dims);
    for seq in [1, k] {
        let step = derived_step_rows(&pg, &dims, seq, rows);
        let dg = as_decode_graph(&pg, &step, &dims);
        let (g, outs) = compiled_decode(&dg, true);
        let mut opts = decode_opts(&dg, &outs);
        if seq == 1 || dg.gemv_window() {
            opts.rows = 8;
            opts.warps = 2;
        }
        let dp = g.to_cuda_decode(&outs, &opts);
        let mut total = 0f64;
        println!("== window {seq}: {} kernels", dp.program.kernels.len());
        let mut lines = Vec::new();
        for (i, kern) in dp.program.kernels.iter().enumerate() {
            let shape: Vec<usize> = g.shape(kern.node).iter().map(|d| d.n()).collect();
            // bytes read: each read buffer once (packed inputs at their
            // stored size are approximated by their element count / 8 when
            // the name says 4-bit; this is an upper-level sketch)
            let bytes: f64 = kern
                .reads
                .iter()
                .map(|r| {
                    let n: usize = g.shape(*r).iter().map(|d| d.n()).product();
                    let packed = opts.packed.iter().any(|(j, _)| {
                        g.nodes().iter().position(
                            |nd| matches!(nd.op, tang_compile::tensor::Op::Input(x) if x == *j),
                        ) == Some(r.index())
                    });
                    n as f64 * if packed { 0.5 } else { 4.0 }
                })
                .sum();
            total += bytes;
            let first = kern
                .source
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(70)
                .collect::<String>();
            lines.push((
                bytes,
                format!(
                    "  k{i:03} {shape:?} wg {} count {} native {} reads {:.1} MB  {first}\n      reads {}",
                    kern.workgroup_size,
                    kern.count,
                    dp.native[i],
                    bytes / 1e6,
                    kern.reads
                        .iter()
                        .map(|r| {
                            let n = &g.nodes()[r.index()];
                            let sh: Vec<usize> = n.shape.iter().map(|d| d.n()).collect();
                            let what = match n.op {
                                tang_compile::tensor::Op::Input(x) => {
                                    let p = opts.packed.iter().any(|(j, _)| *j == x);
                                    format!("in{x}{}", if p { "/packed" } else { "" })
                                }
                                _ => format!("n{}", r.index()),
                            };
                            format!("{what}{sh:?}")
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
            ));
        }
        lines.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, l) in lines.iter().take(25) {
            println!("{l}");
        }
        println!("  total reads {:.0} MB", total / 1e6);
        if let Ok(dir) = std::env::var("DUMP") {
            for (i, kern) in dp.program.kernels.iter().enumerate() {
                std::fs::write(format!("{dir}/s{seq}_k{i:03}.cu"), &kern.source).unwrap();
            }
        }
    }
}
