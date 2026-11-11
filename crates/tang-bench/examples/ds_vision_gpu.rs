//! DeepSeek-V4.1-Flash's vision encoder (ViT + aligner,
//! `tang_bench::ir_ds_vision`) on one GPU, real weights (bf16 → f32).
//!
//! 1. Check: the same graph on a small patch sub-grid with
//!    `VISION_CHECK_LAYERS` blocks (default 2) on the device against the
//!    f64 interpreter (fails above 1e-3 of the output scale).
//! 2. Time: the full encoder for the image (`argv[1]`, default the repo's
//!    hurricane webp), `VISION_ITERS` timed runs (default 10): ms per image.
//!
//! Each `VISION_MODES` lowering (default `mma,prefill`; see [`lower_as`])
//! (`prefill`, `bf16`, `wgN`, `mma`, `f32`) is checked, timed and profiled (`VISION_PROFILE=0`: no per-kernel
//! profile). Tensor-core modes pass under 1e-2 (TF32), `f32` under 1e-3.
//!
//! ```text
//! DS41_DIR=/models/ds41-flash-links ds_vision_gpu [image]
//! ```
use std::time::Instant;

use tang_bench::ir_ds_real::{ds_dir, load_config, St};
use tang_bench::ir_ds_vision::*;
use tang_bench::qwen_session::{lower, runtime};
use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_gpu::program::{Program, Runtime};

/// `VISION_LOWER`: `prefill` (default) runs the training/prefill pipeline
/// (softmax fusion to an online-softmax flash tile, TF32 MMA tiles,
/// `schedule_auto`); `decode` the GEMV-oriented decode lowering, with
/// `mma` its tensor-core tile.
fn lower_as(
    rt: &Runtime,
    g: &TensorGraph,
    outs: &[TensorId],
    how: &str,
) -> (WgslProgram, Vec<TensorId>) {
    match how {
        "prefill" => lower_prefill(g, outs, Bf16::Off),
        "bf16" => lower_prefill(g, outs, Bf16::Weights),
        "bf16act" => lower_prefill(g, outs, Bf16::Activations),
        // bf16 weights, contraction tiles in Hopper wgmma shape h (and
        // flash shape h / |wgmma shapes|), the to_cuda_variant numbering
        // Program::tuned uses
        // wgN: bf16 weights; awgN: bf16 weights and activations
        m if m.starts_with("wg") || m.starts_with("awg") => {
            use tang_compile::tensor::cuda::{with_cuda_sm, MMA_SHAPES};
            let act = m.starts_with('a');
            let h: usize = m.trim_start_matches('a')[2..].parse().unwrap();
            let bf = if act {
                Bf16::Activations
            } else {
                Bf16::Weights
            };
            let (sg, o) = prefill_schedule(g, outs, bf);
            let p = with_cuda_sm(90, || sg.to_cuda_variant(&o, Some(MMA_SHAPES.len() + h)));
            (p, o)
        }
        _ => {
            let opts = DecodeOpts {
                mma: how == "mma",
                ..Default::default()
            };
            (lower(rt, g, outs, &opts).program, outs.to_vec())
        }
    }
}

/// The `n` slowest kernels (ms, points, first comment), time summed by
/// kernel kind, and (`VISION_DUMP_SLOWEST=1`) the slowest one's source.
fn profile(rt: &Runtime, prog: &Program, tags: &[(u32, String, String)], n: usize) {
    let mut t: Vec<(f64, usize)> = (0..tags.len())
        .map(|i| (prog.time_kernel(rt, i, 3), i))
        .collect();
    let total: f64 = t.iter().map(|x| x.0).sum();
    t.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!(
        "profile: {} kernels, sum of single launches {total:.2} ms",
        t.len()
    );
    let mut kinds: std::collections::BTreeMap<String, (f64, usize)> = Default::default();
    for &(ms, i) in &t {
        let e = kinds
            .entry(format!("{} pts {}", tags[i].1, tags[i].0))
            .or_default();
        e.0 += ms;
        e.1 += 1;
    }
    let mut kv: Vec<_> = kinds.into_iter().collect();
    kv.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    for (k, (ms, c)) in kv.iter().take(14) {
        eprintln!("  {ms:8.2} ms  x{c:<4} {k}");
    }
    for &(ms, i) in t.iter().take(n) {
        eprintln!("  {ms:8.3} ms  k{i:<4} pts {:<9} {}", tags[i].0, tags[i].1);
    }
    if env("VISION_DUMP_SLOWEST", 0) > 0 {
        let src = &tags[t[0].1].2;
        eprintln!("--- slowest kernel source (first 120 lines) ---");
        for l in src.lines().take(120) {
            eprintln!("{l}");
        }
    }
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(
            0.0,
            |m, d| if d.is_nan() { f64::INFINITY } else { m.max(d) },
        )
}

fn env(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let dir = ds_dir();
    let p = std::path::Path::new(&dir);
    let cfg = load_config(p).unwrap();
    let vd = VisionDims::from_config(&cfg).expect("no vision tower in the config");
    let st = St::open(p).unwrap();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cram-codec-wasm/site/examples/hurricane-TC-original.webp"
        )
        .into()
    });
    let bytes = std::fs::read(&path).unwrap();
    let t0 = Instant::now();
    let img = preprocess(&bytes, &vd).unwrap();
    let pl = img.plan;
    eprintln!(
        "{path}: {} x {} patches → {} x {} tokens ({} span positions), preprocess {:.1?}",
        pl.n_vit_h,
        pl.n_vit_w,
        pl.n_llm_h,
        pl.n_llm_w,
        pl.span(),
        t0.elapsed()
    );
    let t0 = Instant::now();
    let mut cache = std::collections::HashMap::<String, Vec<f32>>::new();
    let mut weight = |n: &str| -> Vec<f32> {
        cache
            .entry(n.to_string())
            .or_insert_with(|| st.f64(n).unwrap().into_iter().map(|v| v as f32).collect())
            .clone()
    };
    let rt = runtime();

    // 1. check on a 6 x 5 sub-grid, VISION_CHECK_LAYERS blocks
    let check_layers = env("VISION_CHECK_LAYERS", 2);
    let (sh, sw) = (6usize, 5usize);
    let sub_vd = VisionDims {
        n_layers: check_layers,
        ..vd.clone()
    };
    let plen = vd.patch_len();
    let mut sub = img.clone();
    sub.plan = ImagePlan {
        n_vit_h: sh,
        n_vit_w: sw,
        n_llm_h: 2,
        n_llm_w: 2,
        best_h: sh * vd.patch,
        best_w: sw * vd.patch,
    };
    sub.patches = (0..sh)
        .flat_map(|y| {
            let r0 = y * pl.n_vit_w * plen;
            img.patches[r0..r0 + sw * plen].to_vec()
        })
        .collect();
    let eg = encoder_graph(&sub_vd, sh, sw);
    let ins64 = encoder_inputs(&eg, &sub_vd, &sub, |n| {
        weight(n).into_iter().map(|v| v as f64).collect()
    });
    let want = eg.g.eval(&ins64, &[eg.out]).remove(0);
    let modes: Vec<String> = std::env::var("VISION_MODES")
        .unwrap_or_else(|_| "prefill,bf16,bf16act".into())
        .split(',')
        .map(String::from)
        .collect();
    let tol = |m: &str| match m {
        "f32" => 1e-3,
        m if m.contains("act") || m.starts_with('a') => 5e-2,
        _ => 1e-2,
    };
    let (g, outs) = eg.g.eliminate_common(&eg.lower_outputs());
    let words: Vec<Vec<f32>> = ins64
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
    for m in &modes {
        let (p, o) = lower_as(&rt, &g, &outs, m);
        let prog = Program::new(&rt, p, &refs);
        prog.run(&rt);
        let got: Vec<f64> = prog.read(&rt, o[0]).iter().map(|&x| x as f64).collect();
        let err = rel(&got, &want);
        let ok = err < tol(m);
        println!(
            "check [{m}]: {sh} x {sw} patches, {check_layers} blocks, device vs f64 interpreter rel err {err:.2e}: {}",
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            std::process::exit(1);
        }
    }

    // 2. the full encoder for the image
    let eg = encoder_graph(&vd, pl.n_vit_h, pl.n_vit_w);
    let words: Vec<Vec<f32>> = eg
        .inputs
        .iter()
        .map(|s| match s.name.as_str() {
            "patches" => img.patches.iter().map(|&v| v as f32).collect(),
            "rope_angles" => rope_angles(&vd, pl.n_vit_h, pl.n_vit_w)
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            n => weight(n),
        })
        .collect();
    eprintln!(
        "weights: {} tensors, {:.2} G values, loaded in {:.1?}",
        words.len() - 2,
        words.iter().map(|w| w.len()).sum::<usize>() as f64 / 1e9,
        t0.elapsed()
    );
    let (g, outs) = eg.g.eliminate_common(&eg.lower_outputs());
    let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
    let n = pl.patches();
    // FLOPs: per block qkv + wo + mlp (2·N·C·(4C + 3I)) + attention 4·N²·C
    let (c, i) = (vd.dim as f64, vd.inter as f64);
    let nf = n as f64;
    let flops = vd.n_layers as f64 * (2.0 * nf * c * (4.0 * c + 3.0 * i) + 4.0 * nf * nf * c)
        + 2.0 * pl.rows() as f64 * vd.llm_dim as f64 * (9.0 * c + vd.llm_dim as f64);
    let mut first: Option<Vec<f32>> = None;
    for m in &modes {
        let t0 = Instant::now();
        let (p, o) = lower_as(&rt, &g, &outs, m);
        let nk = p.kernels.len();
        let tags: Vec<(u32, String, String)> = p
            .kernels
            .iter()
            .map(|k| {
                let t = k
                    .source
                    .lines()
                    .find(|l| l.contains("//"))
                    .unwrap_or("")
                    .trim();
                let t = t.trim_start_matches("// Auto-generated by tang-compile: ");
                (k.points, t.chars().take(70).collect(), k.source.clone())
            })
            .collect();
        let prog = Program::new(&rt, p, &refs);
        eprintln!(
            "{} [{m}]: {nk} kernels, lowered + uploaded in {:.1?}",
            rt.name(),
            t0.elapsed()
        );
        prog.run(&rt);
        let out = prog.read(&rt, o[0]);
        let finite = out.iter().all(|v| v.is_finite());
        let vs = match &first {
            None => String::new(),
            Some(f) => {
                let a: Vec<f64> = out.iter().map(|&v| v as f64).collect();
                let b: Vec<f64> = f.iter().map(|&v| v as f64).collect();
                format!(", vs first mode rel {:.2e}", rel(&a, &b))
            }
        };
        let iters = env("VISION_ITERS", 10);
        let ms = prog.time(&rt, iters);
        println!(
            "ds41 vision encoder [{m}]: {n} patches → {} rows x {}, {nk} kernels, {ms:.2} ms per image \
             ({:.1} TFLOP/s), finite: {finite}{vs}",
            pl.rows(),
            vd.llm_dim,
            flops / ms / 1e9
        );
        if env("VISION_PROFILE", 1) > 0 {
            profile(&rt, &prog, &tags, 12);
        }
        first.get_or_insert(out);
    }
}
