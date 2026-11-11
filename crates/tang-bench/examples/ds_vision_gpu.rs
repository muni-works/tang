//! DeepSeek-V4.1-Flash's vision encoder (ViT + aligner,
//! `tang_bench::ir_ds_vision`) on one GPU, real weights (bf16 → f32).
//!
//! 1. Check: the same graph on a small patch sub-grid with
//!    `VISION_CHECK_LAYERS` blocks (default 2) on the device against the
//!    f64 interpreter (fails above 1e-3 of the output scale).
//! 2. Time: the full encoder for the image (`argv[1]`, default the repo's
//!    hurricane webp), `VISION_ITERS` timed runs (default 10): ms per image.
//!
//! Both run twice: the plain lowering, then with `DecodeOpts::mma` (the
//! tensor-core tile for matmul-shaped kernels, TF32; `VISION_MMA_SHAPE`
//! picks the block shape). The MMA check passes under 1e-2 (TF32).
//!
//! ```text
//! DS41_DIR=/models/ds41-flash-links ds_vision_gpu [image]
//! ```
use std::time::Instant;

use tang_bench::ir_ds_real::{ds_dir, load_config, St};
use tang_bench::ir_ds_vision::*;
use tang_bench::qwen_session::{lower, runtime};
use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_gpu::program::Program;

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0, |m, d| if d.is_nan() { f64::INFINITY } else { m.max(d) })
}

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
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
    let optss = [
        ("f32", DecodeOpts::default(), 1e-3),
        (
            "mma",
            DecodeOpts {
                mma: true,
                mma_shape: env("VISION_MMA_SHAPE", 0),
                ..Default::default()
            },
            1e-2,
        ),
    ];
    let (g, outs) = eg.g.eliminate_common(&[eg.out]);
    let words: Vec<Vec<f32>> = ins64.iter().map(|v| v.iter().map(|&x| x as f32).collect()).collect();
    let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
    for (name, opts, tol) in &optss {
        let dp = lower(&rt, &g, &outs, opts);
        let prog = Program::new(&rt, dp.program, &refs);
        prog.run(&rt);
        let got: Vec<f64> = prog.read(&rt, outs[0]).iter().map(|&x| x as f64).collect();
        let err = rel(&got, &want);
        let ok = err < *tol;
        println!(
            "check [{name}]: {sh} x {sw} patches, {check_layers} blocks, device vs f64 interpreter rel err {err:.2e}: {}",
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
    let (g, outs) = eg.g.eliminate_common(&[eg.out]);
    let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
    let n = pl.patches();
    // FLOPs: per block qkv + wo + mlp (2·N·C·(4C + 3I)) + attention 4·N²·C
    let (c, i) = (vd.dim as f64, vd.inter as f64);
    let nf = n as f64;
    let flops = vd.n_layers as f64 * (2.0 * nf * c * (4.0 * c + 3.0 * i) + 4.0 * nf * nf * c)
        + 2.0 * pl.rows() as f64 * vd.llm_dim as f64 * (9.0 * c + vd.llm_dim as f64);
    let mut first: Option<Vec<f32>> = None;
    for (name, opts, _) in &optss {
        let t0 = Instant::now();
        let dp = lower(&rt, &g, &outs, opts);
        let nk = dp.program.kernels.len();
        let prog = Program::new(&rt, dp.program, &refs);
        eprintln!("{} [{name}]: {nk} kernels, lowered + uploaded in {:.1?}", rt.name(), t0.elapsed());
        prog.run(&rt);
        let out = prog.read(&rt, outs[0]);
        let finite = out.iter().all(|v| v.is_finite());
        let vs = match &first {
            None => String::new(),
            Some(f) => {
                let a: Vec<f64> = out.iter().map(|&v| v as f64).collect();
                let b: Vec<f64> = f.iter().map(|&v| v as f64).collect();
                format!(", vs f32 lowering rel {:.2e}", rel(&a, &b))
            }
        };
        let iters = env("VISION_ITERS", 10);
        let ms = prog.time(&rt, iters);
        println!(
            "ds41 vision encoder [{name}]: {n} patches → {} rows x {}, {nk} kernels, {ms:.2} ms per image \
             ({:.1} TFLOP/s), finite: {finite}{vs}",
            pl.rows(),
            vd.llm_dim,
            flops / ms / 1e9
        );
        first.get_or_insert(out);
    }
}
