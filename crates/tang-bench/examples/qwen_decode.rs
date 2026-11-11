//! Qwen3 through tang's tensor IR: one graph per step
//! (`tang_bench::ir_qwen`), lowered by `to_msl_decode` (Metal, through
//! wgpu's MSL passthrough) or `to_cuda_decode`, run through
//! `tang_gpu::program::Program`. One set of weight buffers and KV caches
//! serves the decode program and every prefill program
//! (`Program::with_shared`); the caches and the `state` (next token,
//! position) are updated in place, so a decode step is one run and one
//! 8-byte readback.
//!
//! ```text
//! qwen_decode tiny [--derived]           random small model: GPU vs the interpreter
//!     --derived        (any mode) the steps derived from the one prefill graph
//!                      (`ir_qwen::prefill`) instead of the hand-written ones
//! qwen_decode model <dir> <ids> [opts]  MLX 4-bit checkpoint: prompt, then greedy decode
//!     --gen N          greedy tokens after the prompt (default 64)
//!     --ctx C          KV capacity (default 4352)
//!     --chunks a,b,…   prefill the prompt in chunks of these sizes (one
//!                      prefill program per size; default: decode steps)
//!     --time-prefill   device time of each prefill program
//!     --out FILE       write the generated ids
//!     --logits FILE    with --chunks: the last chunk's next-token logits (f32)
//!     --force FILE     teacher-force these ids through decode steps instead,
//!                      writing every step's logits to --logits from --from on
//!     --from P         (default 0)
//!     --bench L        device time of a decode step at context lengths L (comma list)
//!     --bench-window K with --bench: also a K-token window (a draft verify)
//!                      at each length
//!     --profile        per-kernel times of a decode step
//! qwen_decode reference <dir> <ids> <logits out> [--from P]
//!                                       CPU f32 forward (f64 accumulation), the
//!                                       checkpoint dequantized exactly
//! qwen_decode compare <a> <b> <vocab>   two logits files, row by row
//! qwen_decode bench <dir> [--sizes 20,49,512] [--gen 128] [--check]
//!                                       prefill of each size (device ms, tok/s), then
//!                                       decode; --check: the OpenCL prefill variants'
//!                                       logits against the f32-map one
//! qwen_decode tune <dir> [budget_s=300]  measure the decode GEMV shape and the
//!                                       OpenCL prefill block on this device, write
//!                                       the winners to ~/.config/tang/tune.json
//!                                       (`$TANG_TUNE`), read by `model` and Session
//! ```
//!
//! `<ids>` is a file of whitespace-separated token ids (`qwen_tok`).
use std::time::Instant;

use tang_bench::ir_qwen::*;
use tang_bench::qwen_session::*;
use tang_compile::tensor::cuda_gemv::Packing;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_gpu::program::{BackendKind, Runtime};

fn main() {
    #[cfg(feature = "otel")]
    let _otel = tang_bench::otel_init("qwen_decode");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--derived") {
        // read by `Step::new` (qwen_session::derived_mode)
        std::env::set_var("QWEN_DERIVED", "1");
    }
    match args.first().map(String::as_str) {
        Some("tiny") if derived_mode() => tiny_derived(),
        Some("tiny") => tiny(),
        Some("model") => model(&args[1..]),
        Some("reference") => reference(&args[1..]),
        Some("compare") => compare(&args[1..]),
        // the K1 prefill/decode bench: see prefill_bench
        Some("bench") => {
            let dir = &args[1];
            let opt = |k: &str| {
                args.iter()
                    .position(|a| a == k)
                    .map(|i| args[i + 1].clone())
            };
            let sizes: Vec<usize> = opt("--sizes")
                .unwrap_or_else(|| "20,49,512".into())
                .split(',')
                .map(|v| v.parse().unwrap())
                .collect();
            let gen = opt("--gen").map_or(128, |v| v.parse().unwrap());
            prefill_bench(dir, &sizes, gen, args.iter().any(|a| a == "--check"));
        }
        Some("tune") => {
            let budget = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(300.0);
            tune(&args[1], budget);
        }
        _ => eprintln!("usage: qwen_decode tiny | model | reference | compare (see the docs)"),
    }
}

fn f16_round(x: f64) -> f64 {
    use tang_compile::tensor::precision::{f16_bits, f16_from_bits};
    f16_from_bits(f16_bits(x))
}

fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// bf16 values (already representable) packed two per word.
fn pack_bf16(v: &[f32]) -> Vec<f32> {
    v.chunks(2)
        .map(|c| {
            let lo = c[0].to_bits() >> 16;
            let hi = c.get(1).map_or(0, |x| x.to_bits() >> 16);
            f32::from_bits(lo | (hi << 16))
        })
        .collect()
}

/// Nibbles 0..=15 packed eight per word, low first.
fn pack_u4(v: &[f32]) -> Vec<f32> {
    v.chunks(8)
        .map(|c| {
            let w = c
                .iter()
                .enumerate()
                .fold(0u32, |w, (i, &x)| w | ((x as u32) << (4 * i)));
            f32::from_bits(w)
        })
        .collect()
}

/// Device words of f16 values (two per word, element `i` in half `i % 2`).
fn unpack_f16_words(raw: &[f32]) -> Vec<f64> {
    use tang_compile::tensor::precision::f16_from_bits;
    raw.iter()
        .flat_map(|w| {
            let b = w.to_bits();
            [b as u16, (b >> 16) as u16].map(f16_from_bits)
        })
        .collect()
}

// ----------------------------------------------------------------------
// tiny: the GPU against the interpreter

fn tiny_dims() -> QwenDims {
    QwenDims {
        d_model: 256,
        heads: 4,
        kv_heads: 2,
        head_dim: 64,
        ff: 384,
        vocab: 320,
        layers: 2,
        ctx: 24,
        eps: 1e-6,
        theta: 1e4,
        kv_f16: std::env::var("TANG_KV_F32").is_err(),
        // QWEN_VOCAB_CHUNKS=2 exercises the split embedding and LM head
        vocab_chunks: env("QWEN_VOCAB_CHUNKS").max(1),
        prefill_q4: false,
    }
}

/// The interpreter's `dg.outputs`, with `[0]` the full logits (every
/// vocabulary block concatenated).
fn eval_full(g: &TensorGraph, inputs: &[Vec<f64>], dg: &DecodeGraph) -> Vec<Vec<f64>> {
    let all: Vec<TensorId> = dg.outputs.iter().chain(&dg.logit_chunks).copied().collect();
    let mut res = g.eval_emulated(inputs, &all);
    let extra: Vec<Vec<f64>> = res.split_off(dg.outputs.len());
    for e in extra {
        res[0].extend(e);
    }
    res
}

/// Random logical inputs for `dg` (caches and the step inputs zero).
fn tiny_logical(dims: &QwenDims, dg: &DecodeGraph, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = tang_bench::Rng::new(seed);
    dg.inputs
        .iter()
        .map(|s| {
            let n: usize = s.shape.iter().product();
            match s.packing {
                Some(Packing::U4) => (0..n)
                    .map(|_| ((rng.f32() + 1.0) * 7.999).floor())
                    .collect(),
                Some(Packing::Bf16) if s.name.ends_with(".s") => (0..n)
                    .map(|_| bf16_round(0.01 + 0.01 * rng.f32()))
                    .collect(),
                Some(Packing::Bf16) => (0..n)
                    .map(|_| bf16_round(-0.08 * (1.0 + 0.1 * rng.f32())))
                    .collect(),
                _ if s.name == "rope" => rope_table(dims),
                Some(Packing::F16) | None
                    if s.name.ends_with("cache") || s.name == "state" || s.name == "chunk" =>
                {
                    vec![0.0; n]
                }
                Some(Packing::F16) => unreachable!("only the caches are f16"),
                None => (0..n).map(|_| 1.0 + 0.2 * rng.f32()).collect(),
            }
        })
        .collect()
}

fn device_words(dg: &DecodeGraph, logical: &[Vec<f32>]) -> Vec<Vec<f32>> {
    dg.inputs
        .iter()
        .zip(logical)
        .map(|(s, v)| match s.packing {
            Some(Packing::U4) => pack_u4(v),
            Some(Packing::Bf16) => pack_bf16(v),
            Some(Packing::F16) => vec![0.0; Packing::F16.words(v.len())],
            None => v.clone(),
        })
        .collect()
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&a, &b)| m.max((a as f64 - b).abs()));
    err / scale
}

/// Random small model: decode steps, then a prefill continuing the
/// cache at an offset, each against the interpreter (f64, f16 caches
/// emulated) on the same logical inputs.
fn tiny() {
    let dims = tiny_dims();
    let rt = runtime();
    let dg = decode_graph(&dims);
    let logical = tiny_logical(&dims, &dg, 11);
    let device = device_words(&dg, &logical);
    let dec = Step::new(&rt, &dims, 1, dims.ctx, &|_| device.clone(), None);
    let mut cpu: Vec<Vec<f64>> = logical
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();
    // the reference stores the updated caches in f16 like the device
    // (the interpreter's precision emulation)
    let emulate = |dg: &DecodeGraph| {
        let mut g = dg.g.clone();
        if dims.kv_f16 {
            use tang_compile::tensor::{Format, Precision};
            for &t in &dg.outputs[2..] {
                g.set_precision(t, Precision::storage(Format::F16));
            }
        }
        g
    };
    let gref = emulate(&dg);
    let kv = |v: &Vec<f64>| -> Vec<f64> {
        if dims.kv_f16 {
            v.iter().map(|&x| f16_round(x)).collect()
        } else {
            v.clone()
        }
    };
    let (mut worst, mut cache_ulps) = (0.0f64, 0.0f64);
    let mut rng = tang_bench::Rng::new(5);
    let ulp = if dims.kv_f16 {
        2f64.powi(-10)
    } else {
        2f64.powi(-22)
    };
    let check_caches = |step: &Step, res: &[Vec<f64>], ulps: &mut f64| {
        // the device caches against the reference's (layer 0), in units
        // of the cache format's rounding at the cache's own scale (its
        // rms): an entry that is small after cancellation carries the
        // absolute error of its inputs, so a per-entry relative bound
        // would read a 1e-3 absolute slip on a 2e-3 entry as hundreds of
        // ulps.
        for which in [2usize, 3] {
            let raw = step.prog.read(&rt, step.outs[which]);
            let dev: Vec<f64> = if dims.kv_f16 {
                unpack_f16_words(&raw)
            } else {
                raw.iter().map(|&x| x as f64).collect()
            };
            let (ss, n) = res[which]
                .iter()
                .filter(|b| **b != 0.0)
                .fold((0.0f64, 0usize), |(s, n), b| (s + b * b, n + 1));
            let rms = (ss / n.max(1) as f64).sqrt().max(1e-3);
            for (&a, &b) in dev.iter().zip(&res[which]) {
                *ulps = ulps.max((a - b).abs() / (b.abs().max(rms) * ulp));
            }
        }
    };
    // decode positions 0..p0 (teacher-forced random tokens), then the
    // prefill of the rest at p0; then decode again after it
    let p0 = 9;
    let seq = 6;
    for pos in 0..p0 {
        let tok = ((rng.f32() + 1.0) * 0.5 * dims.vocab as f32) as usize % dims.vocab;
        dec.set_state(&rt, tok, pos);
        dec.prog.run(&rt);
        let got = dec.logits(&rt);
        let (next, npos) = dec.state(&rt);
        cpu[dg.input("state")] = vec![tok as f64, pos as f64];
        let res = eval_full(&gref, &cpu, &dg);
        worst = worst.max(rel_err(&got, &res[0]));
        let check_at: usize = std::env::var("QWEN_KEEP_ALL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if pos == check_at && dec.outs.len() > dec.named {
            // QWEN_KEEP_ALL=POS: each kernel's result against the interpreter
            let extra = &dec.outs[dec.named..];
            let want = dec.g.eval_emulated(&cpu, extra);
            for (i, (&t, w)) in extra.iter().zip(&want).enumerate() {
                let got: Vec<f32> = dec.prog.read(&rt, t);
                let e = rel_err(&got, w);
                println!(
                    "  kernel result {i:3} ({t:?}, {} values): rel err {e:.2e}",
                    w.len()
                );
            }
        }
        if dims.kv_f16 {
            check_caches(&dec, &res, &mut cache_ulps);
        }
        assert_eq!(npos, pos + 1, "state position");
        assert_eq!(
            next, res[1][0] as usize,
            "on-device argmax vs the interpreter's"
        );
        assert_eq!(next, argmax(&got), "on-device argmax vs the logits'");
        for l in 0..dims.layers {
            let k = dg.input(&format!("model.layers.{l}.k_cache"));
            let v = dg.input(&format!("model.layers.{l}.v_cache"));
            cpu[k] = kv(&res[2 + 2 * l]);
            cpu[v] = kv(&res[3 + 2 * l]);
        }
        println!(
            "decode pos {pos:2}: rel logit err {:.2e}, next token {next} (interpreter {})",
            rel_err(&got, &res[0]),
            res[1][0]
        );
    }
    // prefill continuing at p0 over a window of the whole cache
    let pre = Step::new(
        &rt,
        &dims,
        seq,
        dims.ctx,
        &|dg| vec![Vec::new(); dg.inputs.len()],
        Some(&dec),
    );
    let pg = emulate(&pre.dg);
    let tokens: Vec<usize> = (0..seq).map(|i| (37 * i + 5) % dims.vocab).collect();
    pre.prefill(&rt, &tokens, p0);
    let got = pre.logits(&rt);
    let (next, npos) = pre.state(&rt);
    let mut pcpu: Vec<Vec<f64>> = pre
        .dg
        .inputs
        .iter()
        .map(|s| {
            if s.name == "chunk" {
                chunk_input(&tokens, p0).iter().map(|&x| x as f64).collect()
            } else {
                cpu[dg.input(&s.name)].clone()
            }
        })
        .collect();
    let res = eval_full(&pg, &pcpu, &pre.dg);
    let perr = rel_err(&got, &res[0]);
    let mut pre_ulps = 0.0;
    // (f32 caches hold f32 arithmetic: compared through the logits)
    if dims.kv_f16 {
        check_caches(&pre, &res, &mut pre_ulps);
    }
    // rows of the cache the prefill did not own are untouched
    println!(
        "prefill {seq} tokens at {p0}: rel logit err {perr:.2e}, caches within {pre_ulps:.1} ulp, next {next} (interpreter {}), position {npos}",
        res[1][0]
    );
    assert_eq!(
        (next, npos),
        (res[1][0] as usize, p0 + seq),
        "prefill state"
    );
    // one decode step after the prefill (shared caches and state)
    for l in 0..dims.layers {
        let k = dg.input(&format!("model.layers.{l}.k_cache"));
        let v = dg.input(&format!("model.layers.{l}.v_cache"));
        cpu[k] = kv(&res[2 + 2 * l]);
        cpu[v] = kv(&res[3 + 2 * l]);
    }
    dec.prog.run(&rt);
    let got = dec.logits(&rt);
    cpu[dg.input("state")] = vec![next as f64, npos as f64];
    let res = eval_full(&gref, &cpu, &dg);
    let derr = rel_err(&got, &res[0]);
    println!("decode after the prefill: rel logit err {derr:.2e}");
    pcpu.clear();
    worst = worst.max(derr);
    println!(
        "tiny: worst relative logit error {worst:.2e} (decode), {perr:.2e} (prefill), caches within {:.1} ulp ({})",
        cache_ulps.max(pre_ulps),
        if dims.kv_f16 { "f16" } else { "f32" }
    );
    // f16 caches: a one-ulp flip of a cached value moves the logits by
    // about 1e-4 of their scale
    let tol = if dims.kv_f16 { 1e-3 } else { 1e-4 };
    let ptol = window_tol(&rt, tol);
    report_tiny(
        "tiny",
        worst < tol && perr < ptol && cache_ulps.max(pre_ulps) <= 1.0,
        tol,
        ptol,
    );
}

// ----------------------------------------------------------------------
// MLX checkpoint
/// `tiny` for the derived steps (`--derived`): decode steps, a 6-token
/// window continuing the cache at p0, and a decode step after it, each
/// against the interpreter on the same derived graph (f16 caches emulated).
fn tiny_derived() {
    use tang_bench::ir_qwen::prefill::{as_decode_graph, derived_step, prefill_graph};
    let dims = tiny_dims();
    let rt = runtime();
    let pg = prefill_graph(&dims);
    let dg = as_decode_graph(&pg, &derived_step(&pg, &dims, 1), &dims);
    let logical = tiny_logical(&dims, &dg, 11);
    let device = device_words(&dg, &logical);
    let dec = Step::new(&rt, &dims, 1, dims.ctx, &|_| device.clone(), None);
    assert!(dec.is_derived());
    let mut cpu: Vec<Vec<f64>> = logical
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();
    let emulate = |dg: &DecodeGraph| {
        let mut g = dg.g.clone();
        if dims.kv_f16 {
            use tang_compile::tensor::{Format, Precision};
            for &t in &dg.outputs[1..] {
                g.set_precision(t, Precision::storage(Format::F16));
            }
        }
        g
    };
    let kv = |v: &Vec<f64>| -> Vec<f64> {
        if dims.kv_f16 {
            v.iter().map(|&x| f16_round(x)).collect()
        } else {
            v.clone()
        }
    };
    // the reference's updated caches back into its inputs
    let store = |cpu: &mut Vec<Vec<f64>>, res: &[Vec<f64>]| {
        for l in 0..dims.layers {
            cpu[dg.input(&format!("model.layers.{l}.k_cache"))] = kv(&res[1 + 2 * l]);
            cpu[dg.input(&format!("model.layers.{l}.v_cache"))] = kv(&res[2 + 2 * l]);
        }
    };
    let gref = emulate(&dg);
    let mut rng = tang_bench::Rng::new(5);
    let (p0, seq) = (9, 6);
    let mut worst = 0.0f64;
    let mut agree = 0;
    for pos in 0..p0 {
        let tok = ((rng.f32() + 1.0) * 0.5 * dims.vocab as f32) as usize % dims.vocab;
        dec.feed(&rt, &[tok], pos);
        dec.prog.run(&rt);
        let got = dec.logits(&rt);
        cpu[dg.input("tokens")] = vec![tok as f64];
        cpu[dg.input("pos")] = vec![pos as f64];
        let res = gref.eval_emulated(&cpu, &dg.outputs);
        let e = rel_err(&got, &res[0]);
        worst = worst.max(e);
        agree += usize::from(argmax(&got) == argmax_f64(&res[0]));
        store(&mut cpu, &res);
        println!("derived decode pos {pos:2}: rel logit err {e:.2e}");
    }
    // a window of `seq` tokens at p0 (shares weights and caches)
    let pre = Step::new(
        &rt,
        &dims,
        seq,
        dims.ctx,
        &|dg| vec![Vec::new(); dg.inputs.len()],
        Some(&dec),
    );
    let tokens: Vec<usize> = (0..seq).map(|i| (37 * i + 5) % dims.vocab).collect();
    pre.feed(&rt, &tokens, p0);
    pre.prog.run(&rt);
    let got = pre.logits(&rt);
    let pcpu: Vec<Vec<f64>> = pre
        .dg
        .inputs
        .iter()
        .map(|s| match s.name.as_str() {
            "tokens" => tokens.iter().map(|&t| t as f64).collect(),
            "pos" => vec![p0 as f64],
            n => cpu[dg.input(n)].clone(),
        })
        .collect();
    let pres = emulate(&pre.dg).eval_emulated(&pcpu, &pre.dg.outputs);
    let werr = rel_err(&got, &pres[0]);
    println!("derived window of {seq} at {p0}: rel logit err {werr:.2e} (all {seq} rows)");
    store(&mut cpu, &pres);
    // one decode step after the window
    let next = pre.next_token(&rt);
    dec.feed(&rt, &[next], p0 + seq);
    dec.prog.run(&rt);
    let got = dec.logits(&rt);
    cpu[dg.input("tokens")] = vec![next as f64];
    cpu[dg.input("pos")] = vec![(p0 + seq) as f64];
    let res = gref.eval_emulated(&cpu, &dg.outputs);
    let derr = rel_err(&got, &res[0]);
    println!("derived decode after the window: rel logit err {derr:.2e}");
    worst = worst.max(derr);
    println!(
        "tiny --derived: worst relative logit error {worst:.2e} (decode), {werr:.2e} (window); argmax {agree}/{p0} ({})",
        if dims.kv_f16 { "f16 caches" } else { "f32 caches" }
    );
    let tol = if dims.kv_f16 { 1e-3 } else { 1e-4 };
    let wtol = window_tol(&rt, tol);
    report_tiny("tiny --derived", worst < tol && werr < wtol, tol, wtol);
}

/// Tolerance of a prefill / window step's logits against the interpreter.
/// On CUDA its projections are f16 tensor-core tiles (`compiled_decode`
/// stores the dequantized weights f16, and the tile stages the f32
/// activations as f16) unless `TANG_PREFILL_TF32` is set; the reference is
/// the f32 graph, so each weight and activation carries an f16 rounding
/// (2⁻¹¹ relative): a few 1e-4 of the logit scale even with f32 caches
/// (RTX PRO 6000: 2e-4 hand, 5.6e-4 derived). Decode has no tiles and keeps
/// the decode tolerance.
fn window_tol(rt: &Runtime, decode_tol: f64) -> f64 {
    let f16_tiles =
        rt.kind() == BackendKind::Cuda && std::env::var_os("TANG_PREFILL_TF32").is_none();
    if f16_tiles {
        decode_tol.max(1e-3)
    } else {
        decode_tol
    }
}

/// The verdict of a `tiny` check: a line, and exit status 1 on failure
/// (after printing, so a script can carry on to its next check).
fn report_tiny(name: &str, ok: bool, tol: f64, window_tol: f64) {
    println!(
        "{name}: {} (tolerance {tol:.0e} decode, {window_tol:.0e} prefill/window)",
        if ok {
            "PASS"
        } else {
            "FAIL: the GPU disagrees with the interpreter"
        }
    );
    if !ok {
        std::process::exit(1);
    }
}

fn argmax_f64(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn read_ids(path: &str) -> Vec<usize> {
    std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn write_f32(path: &str, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn model(args: &[String]) {
    let dir = &args[0];
    let ids = read_ids(&args[1]);
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let gen: usize = opt("--gen").map_or(64, |v| v.parse().unwrap());
    let ctx: usize = opt("--ctx").map_or(4352, |v| v.parse().unwrap());
    let mut dims = dims_for(dir, ctx);
    eprintln!("dims: {dims:?}");
    dims.kv_f16 = std::env::var("TANG_KV_F32").is_err();
    let rt = runtime();
    for (k, v) in apply_tuning(&rt, &dims) {
        eprintln!("tuned {k}={v}");
    }
    // a 311 MB 4-bit LM head exceeds Vulkan's 128 MiB storage binding on
    // the Adreno: four blocks of the vocabulary there (QWEN_VOCAB_CHUNKS
    // overrides)
    dims.vocab_chunks = match env("QWEN_VOCAB_CHUNKS") {
        0 => match &rt {
            Runtime::Wgpu { device, .. } if !device.msl_passthrough => 4,
            // OpenCL buffers top out at CL_DEVICE_MAX_MEM_ALLOC_SIZE (256 MiB on the Adreno)
            _ if rt.kind() == BackendKind::OpenCl => {
                (dims.vocab * dims.d_model / 2).div_ceil(255 << 20).max(1)
            }
            _ => 1,
        },
        n => n,
    };
    eprintln!("runtime: {}", rt.name());
    let t = Instant::now();
    let st = St::open(&format!("{dir}/model.safetensors"));
    eprintln!("checkpoint read in {:.1}s", t.elapsed().as_secs_f64());
    // weights stream in one tensor at a time: converting them all first
    // would hold a second 4.6 GB copy, which the 10 GB robot swaps to zram
    let dec = Step::new(
        &rt,
        &dims,
        1,
        ctx,
        &|dg| vec![Vec::new(); dg.inputs.len()],
        None,
    );
    let t = Instant::now();
    for (k, v) in load_inputs(&st, &dims, &dec.dg).enumerate() {
        if !v.is_empty() {
            dec.prog.write(&rt, dec.prog.input(k), &v);
        }
    }
    eprintln!("weights uploaded in {:.1}s", t.elapsed().as_secs_f64());
    let mut total = dec.bytes;

    if let Some(force) = opt("--force") {
        // teacher forcing through decode steps: every step's logits
        let ids = read_ids(&force);
        let from: usize = opt("--from").map_or(0, |v| v.parse().unwrap());
        let mut out = Vec::new();
        let mut agree = 0;
        for (pos, &tok) in ids.iter().enumerate() {
            dec.feed(&rt, &[tok], pos);
            dec.prog.run(&rt);
            if pos >= from {
                let l = dec.logits(&rt);
                let next = dec.next_token(&rt);
                agree += usize::from(next == argmax(&l));
                out.extend(l);
            }
        }
        println!(
            "teacher-forced {} tokens; on-device argmax equals the logits' argmax at {agree}/{}",
            ids.len(),
            ids.len() - from
        );
        if let Some(p) = opt("--logits") {
            write_f32(&p, &out);
        }
        return;
    }

    // the prompt: prefill chunks (shared weights and caches) or decode steps
    let t = Instant::now();
    let mut prefill_ms = Vec::new();
    // the first generated token (a derived step has no `state`)
    let mut first_tok = 0;
    match opt("--chunks") {
        Some(cs) => {
            let sizes: Vec<usize> = cs.split(',').map(|v| v.parse().unwrap()).collect();
            assert_eq!(
                sizes.iter().sum::<usize>(),
                ids.len(),
                "chunks must cover the prompt"
            );
            let mut steps: Vec<Step> = Vec::new();
            let mut p0 = 0;
            for &seq in &sizes {
                // attention reads the cache up to the chunk's end, rounded up
                let kctx = (p0 + seq).next_multiple_of(64).min(ctx);
                let pre = Step::new(
                    &rt,
                    &dims,
                    seq,
                    kctx,
                    &|dg| vec![Vec::new(); dg.inputs.len()],
                    Some(&dec),
                );
                pre.feed(&rt, &ids[p0..p0 + seq], p0);
                pre.prog.run(&rt);
                first_tok = pre.next_token(&rt);
                if args.iter().any(|a| a == "--time-prefill") {
                    let ms = pre.prog.time(&rt, 3);
                    prefill_ms.push((seq, p0, ms));
                }
                p0 += seq;
                total += pre.bytes;
                // only the last chunk's program is read afterwards: drop
                // the others (each holds its own activations)
                steps.clear();
                steps.push(pre);
            }
            // the last chunk's next-token logits
            if let (Some(p), Some(last)) = (opt("--logits"), steps.last()) {
                write_f32(&p, &last.logits(&rt));
            }
        }
        None => {
            for (pos, &tok) in ids.iter().enumerate() {
                dec.feed(&rt, &[tok], pos);
                dec.prog.run(&rt);
            }
            first_tok = dec.next_token(&rt);
        }
    }
    let (first, pos) = if dec.is_derived() {
        (first_tok, ids.len())
    } else {
        dec.state(&rt)
    };
    assert_eq!(pos, ids.len());
    let prompt_s = t.elapsed().as_secs_f64();
    for (seq, p0, ms) in &prefill_ms {
        println!(
            "prefill {seq} tokens at {p0}: {ms:.1} ms device ({:.0} tok/s)",
            *seq as f64 * 1e3 / ms
        );
    }
    println!(
        "prompt {} tokens in {prompt_s:.2}s (incl. building the prefill programs); allocated {:.2} GB",
        ids.len(),
        total as f64 / 1e9
    );
    // greedy decode: per token one run and one 8-byte readback
    let mut out = vec![first];
    let t = Instant::now();
    let mut p = pos;
    for _ in 1..gen {
        let tok = if dec.is_derived() {
            // token and position in, logits out, argmax on the host
            dec.feed(&rt, &[*out.last().unwrap()], p);
            dec.prog.run(&rt);
            p += 1;
            dec.next_token(&rt)
        } else {
            dec.prog.run(&rt);
            let (tok, np) = dec.state(&rt);
            p = np;
            tok
        };
        if p >= ctx {
            break;
        }
        out.push(tok);
    }
    let dec_s = t.elapsed().as_secs_f64();
    println!(
        "decode {} tokens in {dec_s:.3}s: {:.1} tok/s end to end (run + state readback per token)",
        out.len() - 1,
        (out.len() - 1) as f64 / dec_s
    );
    println!(
        "ids: {}",
        out.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if let Some(p) = opt("--out") {
        let s: Vec<String> = out.iter().map(|t| t.to_string()).collect();
        std::fs::write(p, s.join(" ")).unwrap();
    }
    // device time of one step at a few context lengths
    if let Some(ls) = opt("--bench") {
        let kind = if dec.is_derived() { "derived" } else { "hand" };
        // a K-token window (draft verify) over the whole cache, sharing the
        // decode program's weights and caches
        let window = opt("--bench-window").map(|k| {
            let k: usize = k.parse().unwrap();
            Step::new(
                &rt,
                &dims,
                k,
                ctx,
                &|dg| vec![Vec::new(); dg.inputs.len()],
                Some(&dec),
            )
        });
        for l in ls.split(',').map(|v| v.parse::<usize>().unwrap()) {
            let pos = l.min(ctx - 200) - 1;
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                dec.feed(&rt, &[first], pos);
                best = best.min(dec.prog.time(&rt, 20));
            }
            println!(
                "decode step at len {l} ({kind}): {best:.3} ms device ({:.1} tok/s)",
                1e3 / best
            );
            if let Some(w) = &window {
                let k = w.dg.seq;
                let mut wbest = f64::INFINITY;
                for _ in 0..3 {
                    w.feed(&rt, &vec![first; k], pos);
                    wbest = wbest.min(w.prog.time(&rt, 20));
                }
                println!(
                    "window of {k} at len {l} ({kind}): {wbest:.3} ms device ({:.1} tok/s)",
                    k as f64 * 1e3 / wbest
                );
            }
            // TANG_TRACE_OUT=prefix: per-phase ns of the megakernel (min
            // over 20 runs), or every kernel's ms alone, at this length
            if let Ok(pre) = std::env::var("TANG_TRACE_OUT") {
                #[cfg(feature = "cuda")]
                let v: Vec<f64> = if env("TANG_MEGA") == 1 {
                    let mut m: Vec<f64> = Vec::new();
                    for _ in 0..20 {
                        dec.prog.run(&rt);
                        rt.sync();
                        let t = dec.prog.mega_trace(&rt);
                        m = if m.is_empty() {
                            t
                        } else {
                            m.iter().zip(&t).map(|(a, b)| a.min(*b)).collect()
                        };
                    }
                    m
                } else {
                    let p = &dec.prog;
                    p.order()
                        .iter()
                        .map(|&i| 1e6 * p.time_kernel(&rt, i, 3))
                        .collect()
                };
                #[cfg(not(feature = "cuda"))]
                let v: Vec<f64> = Vec::new();
                let s: Vec<String> = v.iter().map(|x| format!("{x:.0}")).collect();
                std::fs::write(format!("{pre}_{l}.txt"), s.join("\n")).unwrap();
            }
        }
    }
    if args.iter().any(|a| a == "--profile") {
        profile(&rt, &dec);
    }
}

/// Per-kernel device time of one step, grouped by kernel kind.
fn profile(rt: &Runtime, step: &Step) {
    let p = &step.prog;
    let mut groups: Vec<(String, f64, usize)> = Vec::new();
    let mut total = 0.0;
    for (i, k) in p.program.kernels.iter().enumerate() {
        let ms = p.time_kernel(rt, i, 3);
        total += ms;
        let head = k
            .source
            .lines()
            .find(|l| l.starts_with("// ") && !l.contains("Auto-generated"))
            .unwrap_or("")
            .to_string();
        let kind = if !step.native[i] {
            format!("unlowered map, {} points", k.points)
        } else if k.source.contains("in-place") {
            "in-place update".to_string()
        } else {
            head.trim_start_matches("// ").to_string()
        };
        match groups.iter_mut().find(|g| g.0 == kind) {
            Some(g) => {
                g.1 += ms;
                g.2 += 1
            }
            None => groups.push((kind, ms, 1)),
        }
    }
    groups.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("per-kernel (sum of isolated launches {total:.3} ms):");
    for (k, ms, n) in groups {
        println!("  {ms:8.3} ms  {n:4}x  {k}");
    }
}

// ----------------------------------------------------------------------
// CPU reference

/// `y[t][r] = Σ_k w[r][k]·x[t][k]` for every token, f64 accumulation,
/// rows split across threads.
fn matmul(w: &[f32], rows: usize, cols: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = rows.div_ceil(threads);
    let mut out = vec![vec![0.0f32; rows]; xs.len()];
    let parts: Vec<Vec<Vec<f32>>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|c| {
                s.spawn(move || {
                    let r0 = (c * chunk).min(rows);
                    let r1 = ((c + 1) * chunk).min(rows);
                    xs.iter()
                        .map(|x| {
                            (r0..r1)
                                .map(|r| {
                                    let row = &w[r * cols..(r + 1) * cols];
                                    let mut acc = 0.0f64;
                                    for (a, b) in row.iter().zip(x) {
                                        acc += (*a as f64) * (*b as f64);
                                    }
                                    acc as f32
                                })
                                .collect()
                        })
                        .collect()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (c, part) in parts.into_iter().enumerate() {
        let r0 = (c * chunk).min(rows);
        for (t, p) in part.into_iter().enumerate() {
            out[t][r0..r0 + p.len()].copy_from_slice(&p);
        }
    }
    out
}

fn rms(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| (v as f64 * inv) as f32 * g)
        .collect()
}

/// Qwen3 forward over `ids` on the CPU: f32 activations, the checkpoint
/// dequantized exactly to f32, f64 accumulation in every dot product;
/// rope from f64 angles. Logits of positions `from..`.
fn reference(args: &[String]) {
    let dir = &args[0];
    let ids = read_ids(&args[1]);
    let out_path = &args[2];
    let from: usize = args
        .iter()
        .position(|a| a == "--from")
        .map_or(0, |i| args[i + 1].parse().unwrap());
    let dims = dims_for(dir, ids.len());
    let QwenDims {
        d_model: d,
        heads,
        kv_heads: kvh,
        head_dim: hd,
        layers,
        eps,
        ..
    } = dims;
    let half = hd / 2;
    let per = heads / kvh;
    let st = St::open(&format!("{dir}/model.safetensors"));
    let t0 = Instant::now();
    let (emb, _, _) = st.dequant("model.embed_tokens");
    let mut x: Vec<Vec<f32>> = ids
        .iter()
        .map(|&t| emb[t * d..(t + 1) * d].to_vec())
        .collect();
    drop(emb);
    let n = ids.len();
    let angles: Vec<_> = (0..n).map(|p| rope_angles(&dims, p)).collect();
    let rope = |v: &mut [f32], p: usize| {
        let (c, s) = &angles[p];
        for i in 0..half {
            let (a, b) = (v[i], v[i + half]);
            v[i] = a * c[i] - b * s[i];
            v[i + half] = b * c[i] + a * s[i];
        }
    };
    for l in 0..layers {
        let pre = format!("model.layers.{l}.");
        let ln1 = st.bf16(&format!("{pre}input_layernorm.weight"));
        let ln2 = st.bf16(&format!("{pre}post_attention_layernorm.weight"));
        let qn = st.bf16(&format!("{pre}self_attn.q_norm.weight"));
        let kn = st.bf16(&format!("{pre}self_attn.k_norm.weight"));
        let h: Vec<Vec<f32>> = x.iter().map(|v| rms(v, &ln1, eps)).collect();
        let mm = |stem: &str, xs: &[Vec<f32>]| {
            let (w, r, c) = st.dequant(&format!("{pre}{stem}"));
            matmul(&w, r, c, xs)
        };
        let mut q = mm("self_attn.q_proj", &h);
        let mut k = mm("self_attn.k_proj", &h);
        let v = mm("self_attn.v_proj", &h);
        for (p, (qt, kt)) in q.iter_mut().zip(k.iter_mut()).enumerate() {
            for hh in 0..heads {
                let s = &mut qt[hh * hd..(hh + 1) * hd];
                let r = rms(s, &qn, eps);
                s.copy_from_slice(&r);
                rope(s, p);
            }
            for hh in 0..kvh {
                let s = &mut kt[hh * hd..(hh + 1) * hd];
                let r = rms(s, &kn, eps);
                s.copy_from_slice(&r);
                rope(s, p);
            }
        }
        let scale = 1.0 / (hd as f64).sqrt();
        let attn: Vec<Vec<f32>> = (0..n)
            .map(|t| {
                let mut o = vec![0.0f32; heads * hd];
                for hh in 0..heads {
                    let g = hh / per;
                    let qh = &q[t][hh * hd..(hh + 1) * hd];
                    let sc: Vec<f64> = (0..=t)
                        .map(|j| {
                            let kj = &k[j][g * hd..(g + 1) * hd];
                            qh.iter()
                                .zip(kj)
                                .map(|(&a, &b)| a as f64 * b as f64)
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for c in 0..hd {
                        let mut acc = 0.0f64;
                        for (j, &p) in e.iter().enumerate() {
                            acc += p * v[j][g * hd + c] as f64;
                        }
                        o[hh * hd + c] = (acc / z) as f32;
                    }
                }
                o
            })
            .collect();
        let proj = mm("self_attn.o_proj", &attn);
        for (xt, pt) in x.iter_mut().zip(&proj) {
            for (a, b) in xt.iter_mut().zip(pt) {
                *a += b;
            }
        }
        let h2: Vec<Vec<f32>> = x.iter().map(|v| rms(v, &ln2, eps)).collect();
        let gate = mm("mlp.gate_proj", &h2);
        let up = mm("mlp.up_proj", &h2);
        let act: Vec<Vec<f32>> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| {
                g.iter()
                    .zip(u)
                    .map(|(&a, &b)| ((a as f64 / (1.0 + (-a as f64).exp())) * b as f64) as f32)
                    .collect()
            })
            .collect();
        let down = mm("mlp.down_proj", &act);
        for (xt, dt) in x.iter_mut().zip(&down) {
            for (a, b) in xt.iter_mut().zip(dt) {
                *a += b;
            }
        }
        eprintln!("reference: layer {l} ({:.0}s)", t0.elapsed().as_secs_f64());
    }
    let norm = st.bf16("model.norm.weight");
    let h: Vec<Vec<f32>> = x[from..].iter().map(|v| rms(v, &norm, eps)).collect();
    let (w, r, c) = st.dequant("lm_head");
    let logits = matmul(&w, r, c, &h);
    let flat: Vec<f32> = logits.iter().flatten().copied().collect();
    write_f32(out_path, &flat);
    println!(
        "reference: {} rows of logits from position {from} in {:.0}s",
        logits.len(),
        t0.elapsed().as_secs_f64()
    );
}

/// Two logits files (rows of `vocab`): worst and mean absolute error,
/// largest logit, argmax agreement.
fn compare(args: &[String]) {
    let (a, b) = (read_f32(&args[0]), read_f32(&args[1]));
    let vocab: usize = args[2].parse().unwrap();
    assert_eq!(a.len(), b.len(), "different sizes");
    let rows = a.len() / vocab;
    let (mut worst, mut sum, mut scale, mut agree) = (0.0f64, 0.0f64, 0.0f64, 0);
    for r in 0..rows {
        let (x, y) = (
            &a[r * vocab..(r + 1) * vocab],
            &b[r * vocab..(r + 1) * vocab],
        );
        for (&p, &q) in x.iter().zip(y) {
            let e = (p as f64 - q as f64).abs();
            worst = worst.max(e);
            sum += e;
            scale = scale.max(q.abs() as f64);
        }
        agree += usize::from(argmax(x) == argmax(y));
    }
    println!(
        "{rows} rows: max |err| {worst:.3e}, mean {:.3e}, max |logit| {scale:.1}, argmax agree {agree}/{rows}",
        sum / a.len() as f64
    );
}
