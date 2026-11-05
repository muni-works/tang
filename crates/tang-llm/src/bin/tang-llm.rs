//! `tang-llm logits <model-dir> <token ids...>` — dump logits for every position as JSON (for
//! checking against a reference implementation).
//! `tang-llm generate <model-dir> <token ids...> [-n N]` — greedy decode, print ids and speed.
//! `tang-llm serve <model-dir | hf-repo-id> [--host H] [--port P] [--ctx N] [--api-key-file F]`
//! — OpenAI-compatible server (on 127.0.0.1 unless `--host` says otherwise). With a key (from
//! the file, or `TANG_API_KEY`), requests need `Authorization: Bearer <key>`.
//! For frog's scheduler it also serves `GET /node` (hardware, free memory, model, measured
//! rates, queue, KV blocks), `POST /models/load` / `/models/unload` (swap the model, only into
//! free memory) and takes `x-frog-priority: interactive | background` (see `docs/node.md`).
//! `--kv-ttl <secs>` (or `TANG_KV_TTL`) drops everything derived from prompts (KV caches in
//! memory and on disk, saved states, the draft store) once unused that long (default: off;
//! see `docs/retention.md`).
//! `--kv-slots N` keeps N conversations' KV caches (default 1); a request's `prompt_cache_key`
//! picks its conversation's cache. Caches grow with their conversations. `--kv-slots auto`
//! keeps up to 8 within half of RAM where the GPU shares it (Metal, CPU), dropping the least
//! recently used to make room, and keeps 1 on CUDA. `TANG_KV_SLOTS` sets the default;
//! `TANG_KV_BUDGET=<GB>` sets the memory cap (with any slot count). Up to one request per slot
//! decodes at once, batched into the same forward passes (`TANG_MAX_BATCH` caps it; see
//! `docs/batching.md`). While streams decode, a step takes at most `TANG_STEP_MS` (default 50;
//! 0 for no limit): prefill chunks shrink to fit, sized by a step-time model calibrated at load
//! (`TANG_CALIBRATE=0` skips that) and refined as steps run. KV is kept in 256-position
//! blocks shared by content: a conversation whose prompt starts like another's reuses that one's
//! blocks instead of prefilling them. Keyed conversations' blocks are also saved to
//! `~/.cache/tang/kv/` after each turn (each block once) and read back instead of prefilled;
//! `TANG_KV_DISK=<GB>` caps that (default 8, 0 turns it off). KV caches hold bf16 on the GPU
//! (half of f32's memory; attention accumulates in f32); `TANG_KV_F32=1` keeps them f32 (gpt-oss
//! defaults to f32; `TANG_KV_F32=0` makes it bf16).
//! Speculative decoding (suffix drafts from the prompt and earlier completions, verified in one
//! forward; outputs unchanged) is on for GPUs: `--no-speculate` or `TANG_SPECULATE=0` turns it
//! off, `--speculate` forces it on. Earlier completions are kept in
//! `~/.cache/tang/drafts/<model>.tok` (`TANG_DRAFT_STORE=<file>`, or `0` for memory only);
//! `TANG_DRAFT_MAX`, `_ALPHA`, `_MIN_PROB`, `_MIN_MATCH`, `_GLOBAL`, `_COST` tune drafting.
//! `--draft-model <model>` (or `TANG_DRAFT_MODEL`) adds a small model with the same tokenizer
//! (Qwen3-0.6B for Qwen3-32B) that drafts when the suffix drafter has nothing;
//! `TANG_DRAFT_MODEL_MAX` (8) caps its drafts and `_COST` (0.07) is its step's time relative to
//! the target's; drafts grow while expected tokens per unit time improve.
//! `tang-llm image-features <model-dir> <pixels.f32>` — the projector's output for an image
//! (`[896, 896, 3]` f32, normalized), as JSON.
//! `tang-llm logits-image <model-dir> <pixels.f32> <token ids...> [--last N]` — logits with the
//! image standing in for its placeholder tokens.
//! `tang-llm vl-trace <model-dir> <patches.f32> <grid-h> <grid-w> [--f32]` — Qwen3-VL's tower on
//! preprocessed patches (`[n, 3·16·16]`, merge-block order): JSON with the hidden state after
//! the patch embedding and each block, the merged embeddings and the DeepStack features.
//! `tang-llm vl-logits <model-dir> (--image F | --patches F GH GW) <token ids...> [--step K]
//! [--last N] [--f32]` — Qwen3-VL logits with the image in its placeholders; `--step K`
//! prefills the first K ids (with the image) and feeds the rest one at a time.
//! `tang-llm vl-preprocess <model-dir> <image> <out.f32>` — the patches the server would feed.
//!
//! `tang-llm bench-prefill <model> [--n 1024,4096,9216] [--reps R]` — cold prefill (empty
//! cache, 512-token chunks like the server) of N synthetic tokens: ms and tok/s.
//! `tang-llm bench-verify <model> [--ctx 4096,16384] [--k 1,2,4,8,16,32] [--reps R]` — one
//! forward of k tokens on top of a ctx-token cache (a speculative-decoding verify step), with
//! logits for all k rows: ms (median and min over interleaved rounds) and cost relative to the
//! first k.
//!
//! `tang-llm bench-spec <model> <requests.jsonl> [--ctx N] [--passes off,on] [--greedy]` — run
//! chat completion bodies (one JSON per line) with speculative decoding off, then on: decode
//! tok/s, draft acceptance, and whether outputs matched. Drafting settings come from
//! `TANG_DRAFT_*`; the global store starts empty (no file).
//!
//! `tang-llm sim-spec <dump.jsonl>...` — replay `bench-spec --dump` token logs through the
//! drafter without a model: tokens per verify forward and the speedup the cost curve predicts
//! (`TANG_DRAFT_*` settings; `TANG_DRAFT_COST` defaults to the CUDA curve).
//!
//! `tang-llm gguf-info <file.gguf>` — metadata and a tensor inventory of a GGUF (all shards).
//! `tang-llm gguf-rows <file.gguf> <tensor> <first> <n> <out.f32>` — rows dequantized to raw f32.
//! `tang-llm flash-mtp <main.gguf> <mtp.gguf> --ids-file F [--from I] [--depth D] [--joint-hnorm]
//! [--main-head] [--dump DIR]` — teacher-forced acceptance of chained MTP drafts (`flash::mtp`).
//! `tang-llm flash-requant <file.gguf> [--no-errors] [policy...]` — per source type, the error of
//! requantizing dense weights to Q4X / searched Q4X / Q8_0, and the dense bytes a decode step reads
//! under each policy (`flash::requant::DensePolicy`).
//! `tang-llm flash-ref <first shard.gguf> <token ids...> [--ids-file F] [--top K] [--last N]
//! [--dump DIR] [-v]` — Qwen3.8-Flash-Next's slow f32 reference forward on the CPU: top-K
//! next-token logprobs per position as JSON lines, and with `--dump` the last position's
//! per-layer intermediates (see `flash::reference`).
//!
//! Every command takes `--device auto|metal|cuda|cpu` (auto: Metal, then CUDA, then the CPU,
//! whichever is built in and present).

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use tang_compute::{ComputeDevice, CpuDevice};
use tang_llm::draft_model::DraftModel;
use tang_llm::{Dtype, Engine, Model};

/// Where the model runs.
#[derive(Clone, Copy, Debug)]
enum Backend {
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
    Cpu,
}

impl Backend {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "auto" => Self::detect(),
            #[cfg(feature = "metal")]
            "metal" => Self::Metal,
            #[cfg(feature = "cuda")]
            "cuda" => Self::Cuda,
            "cpu" => Self::Cpu,
            other if ["metal", "cuda"].contains(&other) => {
                bail!("this build has no {other} support (rebuild with --features {other})")
            }
            other => bail!("unknown device {other} (auto, metal, cuda or cpu)"),
        })
    }

    /// The first backend that's built in and has a device.
    fn detect() -> Self {
        #[cfg(feature = "metal")]
        if tang_compute::MetalDevice::new().is_some() {
            return Self::Metal;
        }
        #[cfg(feature = "cuda")]
        if new_cuda().is_ok() {
            return Self::Cuda;
        }
        eprintln!("tang-llm: no GPU found, running on the CPU (slow)");
        Self::Cpu
    }
}

#[cfg(feature = "metal")]
fn new_metal() -> Result<tang_compute::MetalDevice> {
    tang_compute::MetalDevice::new().context("no Metal device")
}

/// A CUDA device, or an error when there's no driver (cudarc panics when it can't load
/// libcuda) or no GPU.
#[cfg(feature = "cuda")]
fn new_cuda() -> Result<tang_compute::CudaComputeDevice> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let dev = std::panic::catch_unwind(tang_compute::CudaComputeDevice::new);
    std::panic::set_hook(hook);
    match dev {
        Ok(Ok(dev)) => {
            // TF32 tensor cores for the prefill GEMMs: ~1.1-1.3x prefill on an RTX 3090 at
            // KL ~1e-5 against FP32 (decode's GEMVs don't use cuBLAS). `GAIA_TF32=0` opts out.
            dev.set_tf32(std::env::var("GAIA_TF32").as_deref() != Ok("0"));
            Ok(dev)
        }
        Ok(Err(e)) => bail!("no CUDA device: {e}"),
        Err(_) => bail!("no CUDA device (couldn't load the driver or cuBLAS)"),
    }
}

fn read_f32(path: &str) -> Result<Vec<f32>> {
    Ok(std::fs::read(path)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// Qwen3-VL checks (see the module docs: `vl-trace`, `vl-logits`, `vl-preprocess`).
fn vl<D: ComputeDevice>(
    make: fn() -> Result<D>,
    cmd: &str,
    dir: &std::path::Path,
    args: &[String],
) -> Result<()> {
    use tang_llm::qwen3vl::Patches;
    let mut dtype = Dtype::Bf16;
    // Images in prompt order: encoded bytes, or preprocessed patches.
    let mut images: Vec<Result<Vec<u8>, Patches>> = Vec::new();
    let (mut step, mut last, mut ids) = (None, None, vec![]);
    let mut pos = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--f32" => dtype = Dtype::F32,
            "--image" => images.push(Ok(std::fs::read(it.next().context("--image FILE")?)?)),
            "--patches" => {
                let f = read_f32(it.next().context("--patches F GH GW")?)?;
                let gh: usize = it.next().context("GH")?.parse()?;
                let gw: usize = it.next().context("GW")?.parse()?;
                images.push(Err(Patches {
                    data: f,
                    grid_h: gh,
                    grid_w: gw,
                }));
            }
            "--step" => step = Some(it.next().context("--step K")?.parse::<usize>()?),
            "--last" => last = Some(it.next().context("--last N")?.parse::<usize>()?),
            _ => pos.push(a.clone()),
        }
    }
    let model = Model::load(make()?, dir, 8192, dtype)?;
    let v = model
        .qwen_vision
        .as_ref()
        .context("not a Qwen3-VL checkpoint")?;
    match cmd {
        "vl-preprocess" => {
            let [img, out] = pos.as_slice() else {
                bail!("vl-preprocess <dir> <image> <out.f32>")
            };
            let p = tang_llm::qwen3vl::preprocess(&std::fs::read(img)?, &v.sizing)?;
            let bytes: Vec<u8> = p.data.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(out, bytes)?;
            println!("{} {}", p.grid_h, p.grid_w);
        }
        "vl-trace" => {
            let (f, gh, gw, out) = match pos.as_slice() {
                [f, gh, gw] => (f, gh, gw, None),
                [f, gh, gw, out] => (f, gh, gw, Some(out)),
                _ => bail!("vl-trace <dir> <patches.f32> <grid-h> <grid-w> [out-dir]"),
            };
            let p = Patches {
                data: read_f32(f)?,
                grid_h: gh.parse()?,
                grid_w: gw.parse()?,
            };
            let mut trace = Vec::new();
            let t = Instant::now();
            let e = v.encode_traced(&model.dev, &p, Some(&mut trace));
            let embeds = model.dev.download(&e.embeds);
            eprintln!(
                "encoded {} patches in {:.0}ms",
                p.count(),
                t.elapsed().as_secs_f64() * 1e3
            );
            let ds: Vec<Vec<f32>> = e.deepstack.iter().map(|d| model.dev.download(d)).collect();
            // With an output directory: raw f32 files (block_N, embeds, deepstack_N) instead
            // of JSON, for real-size images.
            if let Some(out) = out {
                let write = |name: String, v: &[f32]| -> Result<()> {
                    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    Ok(std::fs::write(std::path::Path::new(out).join(name), b)?)
                };
                for (i, b) in trace.iter().enumerate() {
                    write(format!("block_{i}.f32"), b)?;
                }
                write("embeds.f32".into(), &embeds)?;
                for (i, d) in ds.iter().enumerate() {
                    write(format!("deepstack_{i}.f32"), d)?;
                }
                println!("{} {}", e.grid.0, e.grid.1);
                return Ok(());
            }
            println!(
                "{}",
                serde_json::json!({"blocks": trace, "embeds": embeds, "deepstack": ds, "grid": [e.grid.0, e.grid.1]})
            );
        }
        "vl-logits" => {
            for a in &pos {
                ids.push(a.parse::<u32>()?);
            }
            let t = Instant::now();
            let mut feats = Vec::new();
            for img in images {
                feats.push(match img {
                    Ok(b) => model.encode_image_bytes(&b)?,
                    Err(p) => model.qwen_features(v.encode(&model.dev, &p)),
                });
            }
            let n: usize = feats.iter().map(|f| f.tokens).sum();
            eprintln!(
                "images: {n} tokens in {:.0}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            let mut cache = model.new_cache();
            let k = step.unwrap_or(ids.len()).min(ids.len());
            let mut logits = model.forward_images(&ids[..k], &feats, &mut cache, true)?;
            for &id in &ids[k..] {
                logits.extend(model.forward(&[id], &mut cache, true)?);
            }
            let vsz = model.cfg.vocab_size;
            let mut rows: Vec<&[f32]> = logits.chunks(vsz).collect();
            if let Some(n) = last {
                rows = rows.split_off(rows.len().saturating_sub(n));
            }
            println!("{}", serde_json::to_string(&rows)?);
        }
        _ => bail!("unknown command {cmd}"),
    }
    Ok(())
}

fn new_cpu() -> Result<CpuDevice> {
    Ok(CpuDevice::new())
}

/// Call `$f(make_device, args...)` with the constructor for `$backend`'s device type.
macro_rules! on_backend {
    ($backend:expr, $f:ident($($arg:expr),*)) => {
        match $backend {
            #[cfg(feature = "metal")]
            Backend::Metal => $f(new_metal, $($arg),*),
            #[cfg(feature = "cuda")]
            Backend::Cuda => $f(new_cuda, $($arg),*),
            Backend::Cpu => $f(new_cpu, $($arg),*),
        }
    };
}

/// Pull `--device X` out of the arguments.
fn take_device(args: &mut Vec<String>) -> Result<Backend> {
    match args.iter().position(|a| a == "--device") {
        Some(i) => {
            let spec = args
                .get(i + 1)
                .context("--device auto|metal|cuda|cpu")?
                .clone();
            args.drain(i..i + 2);
            Backend::parse(&spec)
        }
        None => Ok(Backend::detect()),
    }
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let backend = take_device(&mut args)?;
    let (cmd, dir) = match args.as_slice() {
        [c, d, ..] => (c.as_str(), PathBuf::from(d)),
        _ => bail!("usage: tang-llm <logits|generate> <model-dir> <ids...> [-n N] [--f32]"),
    };
    if cmd == "calibrate-image-preview" {
        return on_backend!(backend, calibrate_image_preview(&args[1..]));
    }
    if args.first().map(String::as_str) == Some("serve-images") {
        return on_backend!(backend, serve_images(&args[1..]));
    }
    if args.first().map(String::as_str) == Some("serve") {
        return serve(backend, &args[1..]);
    }
    if cmd == "gguf-info" {
        let g = tang_llm::gguf::Gguf::open(&dir)?;
        print!("{}", tang_llm::flash::inventory(&g)?);
        return Ok(());
    }
    if cmd == "gguf-rows" {
        // gguf-rows <gguf> <tensor> <first row> <n rows> <out.f32>: dequantized rows, raw f32
        let usage = "gguf-rows <gguf> <tensor> <first row> <n rows> <out.f32>";
        let g = tang_llm::gguf::Gguf::open(&dir)?;
        let t = g.info(args.get(2).context(usage)?)?;
        let first: usize = args.get(3).context(usage)?.parse().context(usage)?;
        let n: usize = args.get(4).context(usage)?.parse().context(usage)?;
        let v = g.rows(t, first, n)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(args.get(5).context(usage)?, bytes)?;
        return Ok(());
    }
    #[cfg(feature = "mtp-train")]
    if cmd == "flash-mtp-train" {
        return tang_llm::flash::mtp_train::cli(&args[1..]);
    }
    if cmd == "flash-mtp" {
        return tang_llm::flash::mtp::cli(&args[1..]);
    }
    if cmd == "flash-requant" {
        use tang_llm::flash::requant::{report, DensePolicy};
        let g = tang_llm::gguf::Gguf::open(&dir)?;
        let errors = !args[2..].iter().any(|a| a == "--no-errors");
        let extra = args[2..]
            .iter()
            .filter(|a| *a != "--no-errors")
            .map(|a| DensePolicy::parse(a))
            .collect::<Result<Vec<_>>>()?;
        print!("{}", report(&g, &extra, errors)?);
        return Ok(());
    }
    if cmd == "flash-tokenize" {
        return tang_llm::flash::tokenize::cli(&args[1..]);
    }
    if cmd == "flash-ref" {
        return tang_llm::flash::reference::cli(&args[1..]);
    }
    if cmd == "glm5-ref" {
        return tang_llm::flash::glm5::forward::cli(&args[1..]);
    }
    // DeepSeek-V4.1-Flash: deepseek-serve <checkpoint dir> [--host H] [--port P] [--ctx N]
    #[cfg(feature = "cuda")]
    if cmd == "deepseek-serve" {
        return deepseek_serve(&args[1..]);
    }
    // deepseek-profile <dir> [--devices ..] [--steps N] [--prompt-tokens N] [--direct]
    #[cfg(feature = "cuda")]
    if cmd == "deepseek-profile" {
        return deepseek_profile(&args[1..]);
    }
    // GLM-5.3-Flash (`glm5-next`) runs on its own engine, on any build.
    if matches!(cmd, "flash-generate" | "flash-bench")
        && tang_llm::flash::glm5::engine::is_glm5(&dir)
    {
        return tang_llm::flash::glm5::engine::cli(&args[1..], cmd == "flash-bench");
    }
    #[cfg(feature = "cuda")]
    match cmd {
        "flash-mtp-prefix" => return tang_llm::flash::cli::mtp_prefix(&args[1..]),
        "flash-coder-dump" => return tang_llm::flash::cli::coder_dump(&args[1..]),
        "flash-generate" => return tang_llm::flash::cli::generate(&args[1..]),
        "flash-parity" => return tang_llm::flash::cli::parity(&args[1..]),
        "flash-bench-panel" => return tang_llm::flash::cli::bench_panel(&args[1..]),
        "flash-bench" => return tang_llm::flash::cli::bench(&args[1..]),
        "flash-spec-test" => return tang_llm::flash::cli::spec_test(&args[1..]),
        "flash-tcheck" => return tang_llm::flash::cli::tcheck(&args[1..]),
        "flash-batch-check" => return tang_llm::flash::cli::batch_check(&args[1..]),
        "flash-spec-batch-check" => return tang_llm::flash::cli::spec_batch_check(&args[1..]),
        "flash-sample-check" => return tang_llm::flash::cli::sample_check(&args[1..]),
        "flash-wide-check" => return tang_llm::flash::cli::wide_check(&args[1..]),
        "flash-state-kl" => return tang_llm::flash::cli::state_kl(&args[1..]),
        "flash-tree-check" => return tang_llm::flash::cli::tree_check(&args[1..]),
        "flash-aa" => return tang_llm::flash::cli::aa(&args[1..]),
        "flash-resume-test" => return tang_llm::flash::serve::resume_test(&args[1..]),
        "flash-sampler-test" => return tang_llm::flash::serve::sampler_test(&args[1..]),
        "flash-prefill-bench" => return tang_llm::flash::serve::prefill_bench(&args[1..]),
        "flash-serve-replay" => return tang_llm::flash::serve::serve_replay(&args[1..]),
        "flash-gemv-check" => return tang_llm::flash::engine::gemv_check(&dir),
        _ => {}
    }
    if cmd == "sim-spec" {
        return sim_spec(&args[1..]);
    }
    if cmd == "bench-spec" {
        return bench_spec(backend, &args[1..]);
    }
    if cmd.starts_with("vl-") {
        let cmd = cmd.to_string();
        return on_backend!(backend, vl(&cmd, &dir, &args[2..]));
    }
    if matches!(cmd, "bench-prefill" | "bench-verify") {
        let cmd = cmd.to_string();
        return bench(backend, &cmd, &args[1..]);
    }
    // Multimodal checks take a raw f32 pixel file before the token ids.
    let mut pixels: Option<Vec<f32>> = None;
    let mut n = 32;
    // `logits --last N`: only the last N positions (long prompts).
    let mut last: Option<usize> = None;
    let mut dtype = Dtype::Bf16;
    let mut ids = Vec::new();
    let mut rest = args[2..].iter();
    if matches!(cmd, "image-features" | "logits-image") {
        let path = rest.next().context("a pixels file")?;
        let bytes = std::fs::read(path)?;
        pixels = Some(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        );
    }
    while let Some(a) = rest.next() {
        if a == "--f32" {
            dtype = Dtype::F32;
        } else if a == "--q4" {
            dtype = Dtype::Q4;
        } else if a == "--last" {
            last = Some(rest.next().context("--last N")?.parse()?);
        } else if a == "-n" {
            n = rest.next().context("-n N")?.parse()?;
        } else {
            ids.push(a.parse::<u32>()?);
        }
    }
    on_backend!(backend, run(cmd, &dir, pixels, n, last, dtype, ids))
}

#[allow(clippy::too_many_arguments)]
fn run<D: ComputeDevice>(
    make: fn() -> Result<D>,
    cmd: &str,
    dir: &std::path::Path,
    pixels: Option<Vec<f32>>,
    n: usize,
    last: Option<usize>,
    dtype: Dtype,
    ids: Vec<u32>,
) -> Result<()> {
    let t = Instant::now();
    let model = Model::load(make()?, dir, 4096, dtype)?;
    eprintln!("loaded in {:.2}s", t.elapsed().as_secs_f32());
    let mut cache = model.new_cache();
    match cmd {
        "image-features" => {
            let v = model.vision.as_ref().context("no vision tower")?;
            let t = Instant::now();
            let f = v.project(&model.dev, pixels.as_ref().unwrap());
            let out = model.dev.download(&f);
            eprintln!("encoded in {:.0}ms", t.elapsed().as_secs_f64() * 1e3);
            let rows: Vec<&[f32]> = out.chunks(model.cfg.hidden_size).collect();
            println!("{}", serde_json::to_string(&rows)?);
        }
        "logits-image" => {
            let img = model.encode_image(pixels.as_ref().unwrap())?;
            let v = model.vision.as_ref().context("no vision tower")?;
            let img = tang_llm::model::ImageFeatures::plain(img, v.tokens);
            let logits = model.forward_images(&ids, &[img], &mut cache, true)?;
            let v = model.cfg.vocab_size;
            let mut rows: Vec<&[f32]> = logits.chunks(v).collect();
            if let Some(n) = last {
                rows = rows.split_off(rows.len().saturating_sub(n));
            }
            println!("{}", serde_json::to_string(&rows)?);
        }
        "logits" | "logits-step" => {
            let logits = if cmd == "logits" {
                model.forward(&ids, &mut cache, true)?
            } else {
                let mut all = Vec::new();
                for &id in &ids {
                    all.extend(model.forward(&[id], &mut cache, true)?);
                }
                all
            };
            let v = model.cfg.vocab_size;
            let mut rows: Vec<&[f32]> = logits.chunks(v).collect();
            if let Some(n) = last {
                rows = rows.split_off(rows.len().saturating_sub(n));
            }
            println!("{}", serde_json::to_string(&rows)?);
        }
        "generate" => {
            let t = Instant::now();
            let mut logits = model.forward(&ids, &mut cache, false)?;
            let prefill = t.elapsed();
            let eos = model.cfg.eos();
            let t = Instant::now();
            let mut out = Vec::new();
            for _ in 0..n {
                let next = argmax(&logits);
                out.push(next);
                if eos.contains(&next) {
                    break;
                }
                logits = model.forward(&[next], &mut cache, false)?;
            }
            let dt = t.elapsed().as_secs_f64();
            println!("{}", serde_json::to_string(&out)?);
            eprintln!(
                "prefill {} tok in {:.0}ms ({:.0} tok/s) · decode {} tok at {:.1} tok/s",
                ids.len(),
                prefill.as_secs_f64() * 1e3,
                ids.len() as f64 / prefill.as_secs_f64(),
                out.len(),
                out.len() as f64 / dt
            );
        }
        c => bail!("unknown command {c}"),
    }
    Ok(())
}

/// `tang-llm serve <first shard.gguf> [--mtp M] [--host H] [--port P] [--ctx N] [--api-key-file F]
/// [--temp T] [--snapshots N] [--dump-ids DIR] [--no-suffix] [--warm-dir DIR] [--batch N]`: the Flash engine (`flash::serve`):
/// Flash-Next, or qwen3next (Qwen3-Coder-Next) by the GGUF's architecture. Without MTP, suffix
/// drafts are on unless `--no-suffix`.
#[cfg(feature = "cuda")]
fn serve_flash(args: &[String]) -> Result<()> {
    use tang_llm::flash::serve::{default_disk, FlashServe, Settings};
    let gguf = PathBuf::from(&args[0]);
    let mut s = Settings {
        sampling: tang_llm::flash::serve::SamplingMode::Greedy,
        disk: default_disk(&gguf),
        gguf,
        mtp: None,
        ctx: 32_768,
        temp: 0.0,
        snapshots: 6,
        warm: Vec::new(),
        profile: None,
        suffix: true,
        warm_dir: None,
        batch: 0,
        snapshot_bytes: std::env::var("TANG_FLASH_SNAP_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(12e9, |g| g * 1e9) as usize,
        dump: None,
    };
    let (mut host, mut port) = ("127.0.0.1".to_string(), 8911u16);
    let mut key = std::env::var("TANG_API_KEY").ok();
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--host" => host = val()?.clone(),
            // live OTLP/HTTP export (otel builds); the default is the local file only
            "--otel-live" => std::env::set_var("TANG_OTEL_LIVE", "1"),
            "--port" => port = val()?.parse()?,
            "--ctx" => s.ctx = val()?.parse()?,
            "--mtp" => s.mtp = Some(PathBuf::from(val()?)),
            "--temp" => s.temp = val()?.parse()?,
            "--sampling" => {
                s.sampling = match val()?.as_str() {
                    "greedy" => tang_llm::flash::serve::SamplingMode::Greedy,
                    "model-card" => tang_llm::flash::serve::SamplingMode::ModelCard,
                    m => bail!("--sampling greedy|model-card, not {m}"),
                }
            }
            "--snapshots" => s.snapshots = val()?.parse()?,
            "--warm-request" => s.warm.push(PathBuf::from(val()?)),
            "--profile" => s.profile = Some(PathBuf::from(val()?)),
            "--dump-ids" => s.dump = Some(PathBuf::from(val()?)),
            "--no-suffix" => s.suffix = false,
            "--warm-dir" => s.warm_dir = Some(PathBuf::from(val()?)),
            "--batch" => s.batch = val()?.parse()?,
            "--api-key-file" => {
                let path = val()?;
                key =
                    Some(std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?);
            }
            other => bail!("unknown option {other}"),
        }
    }
    let key = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    if key.is_none() && !host.starts_with("127.") && host != "localhost" {
        eprintln!("tang-llm: warning: listening on {host} without an API key; anyone who can reach it can use it");
    }
    let name = s
        .gguf
        .file_name()
        .map(|n| {
            n.to_string_lossy()
                .split("-0000")
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .unwrap_or_default();
    let opts = tang_llm::server::Options {
        addr: format!("{host}:{port}"),
        model: Some(name),
        key,
        dtype: Dtype::Q4,
        hardware: probe(Backend::Cuda),
    };
    tang_llm::server::serve(opts, move |_name: &str| FlashServe::load(&s))
}

/// Where a DeepSeek step's time goes: prefill + N decode/verify forwards with the step
/// timeline on (docs/observability.md, "DeepSeek").
#[cfg(feature = "cuda")]
fn deepseek_profile(args: &[String]) -> Result<()> {
    let usage = "deepseek-profile <checkpoint dir> [--devices 0,1,..] [--steps N] [--prompt-tokens N] [--direct] [--graph-debug] [--no-spec] [--ctx N]";
    let dir = PathBuf::from(args.first().context(usage)?);
    let mut devices: Vec<usize> = Vec::new();
    let mut o = tang_llm::flash::deepseek::serve::ProfileOpts {
        steps: 20,
        prompt_tokens: 64,
        direct: false,
        graph_debug: false,
        spec: std::env::var("DS_SPEC").map_or(true, |v| v != "0"),
    };
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--steps" => o.steps = val()?.parse()?,
            "--prompt-tokens" => o.prompt_tokens = val()?.parse()?,
            "--direct" => o.direct = true,
            "--graph-debug" => o.graph_debug = true,
            "--no-spec" => o.spec = false,
            "--ctx" => std::env::set_var("DS41_CTX", val()?),
            "--devices" => {
                devices = val()?
                    .split(',')
                    .map(|d| d.trim().parse::<usize>())
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            other => bail!("unknown option {other} ({usage})"),
        }
    }
    let _otel = tang_llm::telemetry::init();
    let table = tang_llm::flash::deepseek::serve::profile(&dir, &devices, &o)?;
    print!("{table}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn deepseek_serve(args: &[String]) -> Result<()> {
    let usage =
        "deepseek-serve <checkpoint dir> [--host H] [--port P] [--ctx N] [--expert-slots N] [--devices 0,1,..]";
    let dir = PathBuf::from(args.first().context(usage)?);
    let (mut host, mut port) = ("127.0.0.1".to_string(), 8911u16);
    let mut key = std::env::var("TANG_API_KEY").ok();
    let mut devices: Vec<usize> = Vec::new();
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--host" => host = val()?.clone(),
            "--port" => port = val()?.parse()?,
            "--ctx" => std::env::set_var("DS41_CTX", val()?),
            "--expert-slots" => std::env::set_var("DS_EXPERT_SLOTS", val()?),
            // GPU ordinals for expert parallelism (needs --features ep), e.g. 0,1,2,3
            "--devices" => {
                devices = val()?
                    .split(',')
                    .map(|d| d.trim().parse::<usize>())
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            "--api-key-file" => {
                let path = val()?;
                key =
                    Some(std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?);
            }
            other => bail!("unknown option {other} ({usage})"),
        }
    }
    let key = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    if key.is_none() && !host.starts_with("127.") && host != "localhost" {
        eprintln!("tang-llm: warning: listening on {host} without an API key; anyone who can reach it can use it");
    }
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let opts = tang_llm::server::Options {
        addr: format!("{host}:{port}"),
        model: Some(name),
        key,
        dtype: Dtype::Q4,
        hardware: probe(Backend::Cuda),
    };
    tang_llm::server::serve(opts, move |_name: &str| {
        tang_llm::flash::deepseek::serve::DsServe::load_on(&dir, &devices)
    })
}

fn serve(backend: Backend, args: &[String]) -> Result<()> {
    // `--kv-ttl <secs>` (or TANG_KV_TTL): how long prompt-derived state outlives its last use,
    // for every backend (see `docs/retention.md`).
    let mut ttl = tang_llm::retention::parse(&std::env::var("TANG_KV_TTL").unwrap_or_default())?;
    let mut rest = Vec::with_capacity(args.len());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--kv-ttl" => ttl = tang_llm::retention::parse(it.next().context("--kv-ttl <secs>")?)?,
            _ => rest.push(a.clone()),
        }
    }
    let args = &rest[..];
    tang_llm::retention::set_ttl(ttl);
    if let Some(t) = ttl {
        eprintln!(
            "tang-llm: prompt-derived state (KV caches, saved states, drafts) dropped {} s after last use",
            t.as_secs()
        );
    }
    #[cfg(feature = "cuda")]
    if args.first().is_some_and(|a| a.ends_with(".gguf")) {
        return serve_flash(args);
    }
    let spec = args
        .first()
        .context("usage: tang-llm serve <model> [--host H] [--port P] [--ctx N] [--api-key-file F] [--f32 | --q4] [--no-speculate] [--draft-model M] [--kv-slots N|auto] [--kv-ttl SECS] [--otel-live]")?
        .clone();
    let (mut port, mut ctx, mut dtype) = (8911u16, 32_768usize, Dtype::Bf16);
    let mut host = "127.0.0.1".to_string();
    let mut image_pipeline = None;
    let mut slots = kv_slots(&std::env::var("TANG_KV_SLOTS").unwrap_or_else(|_| "1".into()))?;
    // Never on the command line itself, where `ps` would show it.
    let mut key = std::env::var("TANG_API_KEY").ok();
    // Speculative decoding (suffix drafts): on for GPUs unless TANG_SPECULATE=0 or --no-speculate.
    let mut speculate = match std::env::var("TANG_SPECULATE").as_deref() {
        Ok("0") | Ok("off") | Ok("false") => false,
        Ok(_) => true,
        Err(_) => !matches!(backend, Backend::Cpu),
    };
    let mut draft_model = std::env::var("TANG_DRAFT_MODEL")
        .ok()
        .filter(|m| !m.is_empty());
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--host" => host = it.next().context("--host H")?.clone(),
            // live OTLP/HTTP export (otel builds); the default is the local file only
            "--otel-live" => std::env::set_var("TANG_OTEL_LIVE", "1"),
            "--port" => port = it.next().context("--port P")?.parse()?,
            "--ctx" => ctx = it.next().context("--ctx N")?.parse()?,
            "--image-pipeline" => {
                image_pipeline = Some(PathBuf::from(it.next().context("--image-pipeline DIR")?))
            }
            "--kv-slots" => slots = kv_slots(it.next().context("--kv-slots N|auto")?)?,
            "--api-key-file" => {
                let path = it.next().context("--api-key-file F")?;
                key =
                    Some(std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?);
            }
            "--f32" => dtype = Dtype::F32,
            "--q4" => dtype = Dtype::Q4,
            "--speculate" => speculate = true,
            "--no-speculate" => speculate = false,
            "--draft-model" => {
                draft_model = Some(it.next().context("--draft-model <model>")?.clone())
            }
            other => bail!("unknown option {other}"),
        }
    }
    // The GPU's memory is the system's (so RAM bounds the KV slots).
    let unified = match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => false,
        _ => true,
    };
    let key = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    tang_llm::resolve_model(&spec)?;
    if let Some(m) = &draft_model {
        tang_llm::resolve_model(m)?;
    }
    let addr = format!("{host}:{port}");
    if key.is_none() && !host.starts_with("127.") && host != "localhost" {
        eprintln!("tang-llm: warning: listening on {host} without an API key; anyone who can reach it can use it");
    }
    let opts = tang_llm::server::Options {
        addr,
        key,
        model: Some(spec),
        dtype,
        hardware: probe(backend),
    };
    on_backend!(
        backend,
        serve_on(
            opts,
            backend,
            ctx,
            speculate,
            draft_model,
            slots,
            unified,
            image_pipeline
        )
    )
}

/// Reads the device and its free memory, for `/node` and the fit check before a load.
fn probe(backend: Backend) -> tang_llm::node::Probe {
    use std::sync::Arc;
    use tang_llm::node::{host_memory, Hardware};
    let host = || host_memory().unwrap_or((0, 0));
    match backend {
        #[cfg(feature = "metal")]
        Backend::Metal => {
            // The same GPU the model's device uses: its working set counts this process's
            // buffers.
            let dev = std::sync::Mutex::new(new_metal().ok());
            Arc::new(move || {
                let (total, available) = host();
                let dev = dev.lock().unwrap_or_else(|e| e.into_inner());
                let (name, gpu_free) = dev.as_ref().map_or((String::new(), available), |d| {
                    (d.device_name(), d.free_memory_bytes() as u64)
                });
                Hardware {
                    kind: "metal",
                    name,
                    unified: true,
                    total_bytes: total,
                    free_bytes: available.min(gpu_free),
                }
            })
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => Arc::new(|| {
            let (name, free, total) = tang_compute::cuda_memory_info().unwrap_or_default();
            Hardware {
                kind: "cuda",
                name,
                unified: false,
                total_bytes: total as u64,
                free_bytes: free as u64,
            }
        }),
        Backend::Cpu => Arc::new(move || {
            let (total, available) = host();
            Hardware {
                kind: "cpu",
                name: "cpu".into(),
                unified: true,
                total_bytes: total,
                free_bytes: available,
            }
        }),
    }
}

/// Drafting settings for `backend`: its verify cost curve, a store of earlier completions under
/// `~/.cache/tang/drafts/` (`TANG_DRAFT_STORE`: another file, or 0 for none), `TANG_DRAFT_*`.
fn draft_config(backend: Backend, model: &str) -> tang_llm::draft::DraftConfig {
    use tang_llm::draft::{DraftConfig, COST_CUDA, COST_METAL};
    let cost = match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => COST_CUDA,
        #[cfg(feature = "metal")]
        Backend::Metal => COST_METAL,
        // A CPU forward costs about k times one token: drafts only pay when nearly certain.
        _ => &[(1, 1.0), (2, 1.9), (32, 30.0)],
    };
    let _ = (COST_CUDA, COST_METAL);
    let mut cfg = DraftConfig::new(cost).from_env();
    let name = file_name(model);
    cfg.store = match std::env::var("TANG_DRAFT_STORE").as_deref() {
        Ok("0") | Ok("") => None,
        Ok(p) => Some(PathBuf::from(p)),
        Err(_) => std::env::var_os("HOME").map(|h| {
            PathBuf::from(h)
                .join(".cache/tang/drafts")
                .join(format!("{name}.tok"))
        }),
    };
    cfg
}

/// `model` as a file name.
fn file_name(model: &str) -> String {
    model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Where conversations' KV caches go on disk, and how many bytes: `~/.cache/tang/kv/`, 8 GB
/// unless `TANG_KV_DISK=<GB>` (0 turns it off).
fn kv_disk(model: &str, dtype: Dtype) -> Option<(PathBuf, u64)> {
    let gb = match std::env::var("TANG_KV_DISK") {
        Ok(v) => v.parse::<f64>().ok()?,
        Err(_) => 8.0,
    };
    if gb <= 0.0 {
        return None;
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    let dir = home
        .join(".cache/tang/kv")
        .join(format!("{}-{dtype:?}", file_name(model)));
    Some((dir, (gb * 1e9) as u64))
}

/// `--kv-slots N|auto`; None is auto.
fn kv_slots(v: &str) -> Result<Option<usize>> {
    match v {
        "auto" => Ok(None),
        n => Ok(Some(n.parse().context("--kv-slots N|auto")?)),
    }
}

/// `--kv-slots auto`: up to 8 caches within half of RAM where the GPU shares RAM; 1 elsewhere
/// (nothing here measures VRAM). Slots and a memory budget.
fn auto_slots(unified: bool) -> (usize, Option<usize>) {
    match unified.then(physical_ram).flatten() {
        Some(ram) => (8, Some(ram / 2)),
        None => (1, None),
    }
}

fn physical_ram() -> Option<usize> {
    if cfg!(target_os = "macos") {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        return String::from_utf8(out.stdout).ok()?.trim().parse().ok();
    }
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: usize = info
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kb * 1024)
}

/// Serve with `opts`, loading each model (the first, and any `/models/load` asks for) with the
/// command line's settings.
fn serve_on<D: ComputeDevice + 'static>(
    make: fn() -> Result<D>,
    opts: tang_llm::server::Options,
    backend: Backend,
    ctx: usize,
    speculate: bool,
    draft_model: Option<String>,
    slots: Option<usize>,
    unified: bool,
    image_pipeline: Option<PathBuf>,
) -> Result<()> {
    let dtype = opts.dtype;
    let loader = move |name: &str| {
        let t = Instant::now();
        let dir = tang_llm::resolve_model(name)?;
        let disk = kv_disk(name, dtype);
        let draft = speculate.then(|| draft_config(backend, name));
        let mut e = Engine::load(make()?, &dir, ctx, dtype)?;
        if let Some(d) = &draft {
            eprintln!(
                "tang-llm: speculative decoding on (suffix drafts, up to {}; store {})",
                d.max_draft,
                d.store
                    .as_ref()
                    .map_or("off".into(), |p| p.display().to_string())
            );
        }
        e.set_speculation(draft);
        if let Some(m) = draft_model.as_ref().filter(|_| speculate) {
            let d = DraftModel::load(make()?, &tang_llm::resolve_model(m)?, ctx, e.vocab_size())?;
            eprintln!(
                "tang-llm: draft model {m} (up to {} tokens, a step {:.2} of the target's)",
                d.max, d.step_cost
            );
            e.set_draft_model(Some(d));
        }
        let (n, budget) = slots.map_or_else(|| auto_slots(unified), |n| (n, None));
        let budget = std::env::var("TANG_KV_BUDGET")
            .ok()
            .and_then(|gb| gb.parse::<f64>().ok())
            .map(|gb| (gb * 1e9) as usize)
            .or(budget);
        e.set_slots(n);
        e.set_kv_budget(budget);
        if let Some(b) = std::env::var("TANG_MAX_BATCH")
            .ok()
            .and_then(|b| b.parse().ok())
        {
            e.set_max_batch(b);
        }
        // A step's time budget while streams decode, in ms (0: whole prefill chunks always).
        let step_ms = std::env::var("TANG_STEP_MS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(tang_llm::engine::STEP_BUDGET * 1e3);
        e.set_step_budget((step_ms > 0.0).then_some(step_ms / 1e3));
        if e.max_batch() > 1 && std::env::var("TANG_CALIBRATE").as_deref() != Ok("0") {
            let t = std::time::Instant::now();
            e.calibrate()?;
            let c = e.step_cost().coef();
            eprintln!(
                "tang-llm: step cost calibrated in {:.1}s: {:.1} ms + {:.2} ms a decode row; a prefill chunk {:.1} ms + {:.2} ms a token (+{:.3} per 1k of prefix); step budget {}",
                t.elapsed().as_secs_f32(),
                c[0] * 1e3,
                c[1] * 1e3,
                c[2] * 1e3,
                c[3],
                c[4],
                if step_ms > 0.0 { format!("{step_ms:.0} ms") } else { "off".into() },
            );
        }
        if let Some((dir, bytes)) = disk {
            e.set_kv_store(dir.clone(), bytes)?;
            eprintln!(
                "tang-llm: KV caches saved in {} (up to {:.0} GB)",
                dir.display(),
                bytes as f64 / 1e9
            );
        }
        if let Some(s) = e.speculation() {
            eprintln!("tang-llm: draft store has {} tokens", s.global.tokens());
        }
        eprintln!(
            "tang-llm: loaded {} in {:.1}s ({} ctx, {} KV slots{}, batch {}; KV {} KiB a token, {})",
            dir.display(),
            t.elapsed().as_secs_f32(),
            e.context_window(),
            e.slots(),
            budget.map_or(String::new(), |b| format!(
                " within {:.1} GB",
                b as f64 / 1e9
            )),
            e.max_batch(),
            e.model.row_bytes() / 1024,
            if e.model.kv_bf16() { "bf16" } else { "f32" },
        );
        Ok(e)
    };
    tang_llm::server::serve_with_images(opts, loader, image_pipeline.map(|root| (root, make)))
}

/// Options for the bench commands.
struct BenchOpts {
    ns: Vec<usize>,
    ks: Vec<usize>,
    reps: usize,
}

fn bench(backend: Backend, cmd: &str, args: &[String]) -> Result<()> {
    let list = |s: &String| -> Result<Vec<usize>> {
        s.split(',').map(|v| Ok(v.trim().parse()?)).collect()
    };
    let spec = args
        .first()
        .context("a model directory or Hugging Face repo id")?;
    let prefill = cmd == "bench-prefill";
    let mut o = BenchOpts {
        ns: if prefill {
            vec![1024, 4096, 9216]
        } else {
            vec![4096, 16384]
        },
        ks: vec![1, 2, 4, 8, 16, 32],
        reps: if prefill { 2 } else { 10 },
    };
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" | "--ctx" => o.ns = list(it.next().context("--ctx N,N,...")?)?,
            "--k" => o.ks = list(it.next().context("--k K,K,...")?)?,
            "--reps" => o.reps = it.next().context("--reps R")?.parse()?,
            other => bail!("unknown option {other}"),
        }
    }
    let dir = tang_llm::resolve_model(spec)?;
    eprintln!("{cmd} {} on {backend:?}", dir.display());
    on_backend!(backend, bench_on(&dir, prefill, o))
}

/// Deterministic, tokenizer-free token ids (timing doesn't depend on the text).
fn synthetic_ids(n: usize, vocab: usize) -> Vec<u32> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (1000 + x % (vocab.min(150_000) as u64 - 1000)) as u32
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn bench_on<D: ComputeDevice>(
    make: fn() -> Result<D>,
    dir: &std::path::Path,
    prefill: bool,
    o: BenchOpts,
) -> Result<()> {
    let max_n = *o.ns.iter().max().context("no sizes")?;
    let max_k = if prefill {
        0
    } else {
        *o.ks.iter().max().context("no k")?
    };
    let model = Model::load(make()?, dir, max_n + max_k + 1, Dtype::Bf16)?;
    let vocab = model.cfg.vocab_size;
    let mut cache = model.new_cache();
    // Warm up: kernel compilation, pools, cuBLAS/Metal pipeline setup.
    for m in [1, 4, 16, 512] {
        cache.truncate(0);
        model.forward(&synthetic_ids(m, vocab), &mut cache, false)?;
    }
    let fill = |cache: &mut tang_llm::Cache<D::Buffer>, ids: &[u32]| -> Result<()> {
        cache.truncate(0);
        for chunk in ids.chunks(512) {
            model.forward(chunk, cache, false)?;
        }
        Ok(())
    };
    if prefill {
        println!("| prompt tokens | ms | tok/s |");
        println!("|---:|---:|---:|");
        for &n in &o.ns {
            let ids = synthetic_ids(n, vocab);
            let mut times = Vec::new();
            for _ in 0..o.reps.max(1) {
                let t = Instant::now();
                fill(&mut cache, &ids)?;
                times.push(t.elapsed().as_secs_f64());
            }
            let t = median(times);
            println!("| {n} | {:.0} | {:.0} |", t * 1e3, n as f64 / t);
        }
        return Ok(());
    }
    // Rounds interleave the k values so that load from elsewhere on the machine (a shared
    // GPU) hits every k alike; the median and the minimum are both reported.
    println!("| ctx | k | ms/forward (median) | ms (min) | vs k=1 (median) | vs k=1 (min) |");
    println!("|---:|---:|---:|---:|---:|---:|");
    for &ctx in &o.ns {
        let ids = synthetic_ids(ctx + max_k, vocab);
        fill(&mut cache, &ids[..ctx])?;
        let mut times = vec![Vec::new(); o.ks.len()];
        for r in 0..o.reps.max(1) + 2 {
            for (i, &k) in o.ks.iter().enumerate() {
                cache.truncate(ctx);
                let t = Instant::now();
                let logits = model.forward(&ids[ctx..ctx + k], &mut cache, true)?;
                let dt = t.elapsed().as_secs_f64() * 1e3;
                anyhow::ensure!(logits.len() == k * vocab);
                if r >= 2 {
                    times[i].push(dt);
                }
            }
        }
        let stats: Vec<(f64, f64)> = times
            .into_iter()
            .map(|t| {
                (
                    median(t.clone()),
                    t.into_iter().fold(f64::INFINITY, f64::min),
                )
            })
            .collect();
        let (b_med, b_min) = stats[0];
        for (&k, &(med, min)) in o.ks.iter().zip(&stats) {
            println!(
                "| {ctx} | {k} | {med:.2} | {min:.2} | {:.2} | {:.2} |",
                med / b_med,
                min / b_min
            );
        }
    }
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn bench_spec(backend: Backend, args: &[String]) -> Result<()> {
    let spec = args
        .first()
        .context("bench-spec <model> <requests.jsonl>")?;
    let file = args.get(1).context("bench-spec <model> <requests.jsonl>")?;
    let (mut ctx, mut passes, mut greedy) = (
        16_384usize,
        vec!["off".to_string(), "on".to_string()],
        false,
    );
    let mut dump: Option<String> = None;
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ctx" => ctx = it.next().context("--ctx N")?.parse()?,
            "--passes" => {
                passes = it
                    .next()
                    .context("--passes off,on")?
                    .split(',')
                    .map(str::to_string)
                    .collect()
            }
            "--greedy" => greedy = true,
            "--dump" => dump = Some(it.next().context("--dump FILE")?.clone()),
            other => bail!("unknown option {other}"),
        }
    }
    let mut reqs = Vec::new();
    for line in std::fs::read_to_string(file)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let body: serde_json::Value = serde_json::from_str(line)?;
        let mut r = tang_llm::server::parse(&body).map_err(|e| anyhow::anyhow!(e))?;
        if greedy {
            r.sampling.temperature = 0.0;
        }
        reqs.push(r);
    }
    let dir = tang_llm::resolve_model(spec)?;
    eprintln!(
        "bench-spec {} on {backend:?}: {} requests",
        dir.display(),
        reqs.len()
    );
    let mut draft = draft_config(backend, spec);
    draft.store = None;
    on_backend!(backend, bench_spec_on(&dir, ctx, reqs, passes, draft, dump))
}

fn bench_spec_on<D: ComputeDevice>(
    make: fn() -> Result<D>,
    dir: &std::path::Path,
    ctx: usize,
    reqs: Vec<tang_llm::Request>,
    passes: Vec<String>,
    draft: tang_llm::draft::DraftConfig,
    dump: Option<String>,
) -> Result<()> {
    let mut dump = match dump {
        Some(f) => Some(std::io::BufWriter::new(std::fs::File::create(f)?)),
        None => None,
    };
    let mut fwd: Vec<Vec<(usize, f64)>> = vec![vec![(0, 0.0); 65]; passes.len()];
    let mut e = Engine::load(make()?, dir, ctx, Dtype::Bf16)?;
    // Warm up kernels.
    e.set_speculation(None);
    let warm = tang_llm::server::parse(&serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8
    }))
    .map_err(|e| anyhow::anyhow!(e))?;
    e.complete(&warm, |_| true)?;
    // Passes interleave per request (off, on, off, on, ...) so that load from elsewhere on a
    // shared GPU hits both alike. Each pass keeps its own drafting state across requests.
    e.set_speculation(Some(draft.clone()));
    let mut states: Vec<Option<tang_llm::engine::Speculation>> = passes
        .iter()
        .map(|p| {
            (p == "on").then(|| {
                e.set_speculation(Some(draft.clone()));
                e.take_speculation().unwrap()
            })
        })
        .collect();
    let mut texts: Vec<Vec<String>> = vec![Vec::new(); passes.len()];
    let mut totals: Vec<(String, usize, f64, usize, usize)> =
        passes.iter().map(|p| (p.clone(), 0, 0.0, 0, 0)).collect();
    for (i, r) in reqs.iter().enumerate() {
        for (j, pass) in passes.iter().enumerate() {
            e.put_speculation(states[j].take());
            let before = e.forwards.clone();
            let mut text = String::new();
            let (finish, u) = e.complete(r, |p| {
                match p {
                    tang_llm::chat::Piece::Reasoning(t) | tang_llm::chat::Piece::Text(t) => {
                        text.push_str(&t)
                    }
                    tang_llm::chat::Piece::ToolCall { name, arguments } => {
                        text.push_str(&format!("[{name} {arguments}]"))
                    }
                }
                true
            })?;
            states[j] = e.take_speculation();
            for (w, (a, b)) in e.forwards.iter().zip(&before).enumerate() {
                fwd[j][w].0 += a.0 - b.0;
                fwd[j][w].1 += a.1 - b.1;
            }
            if j == 0 {
                if let Some(d) = dump.as_mut() {
                    use std::io::Write;
                    let (p, o) = e.last_tokens();
                    writeln!(d, "{}", serde_json::json!({ "prompt": p, "out": o }))?;
                }
            }
            let s = u.completion_tokens as f64 / u.decode_tok_s.max(1e-9);
            eprintln!(
                "{pass} #{i}: prompt {} (cached {}) → {} tok {finish:?}, {:.1} tok/s, drafts {}/{}",
                u.prompt_tokens,
                u.cached_tokens,
                u.completion_tokens,
                u.decode_tok_s,
                u.accepted_tokens,
                u.draft_tokens
            );
            let t = &mut totals[j];
            t.1 += u.completion_tokens;
            t.2 += s;
            t.3 += u.draft_tokens;
            t.4 += u.accepted_tokens;
            texts[j].push(text);
        }
    }
    println!("| pass | tokens | decode s | tok/s | vs first | drafted | accepted | accept % | tok/forward |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    let base = totals[0].1 as f64 / totals[0].2;
    for (pass, toks, secs, drafted, accepted) in &totals {
        let rate = *toks as f64 / secs;
        println!(
            "| {pass} | {toks} | {secs:.1} | {rate:.1} | {:.2}x | {drafted} | {accepted} | {:.0}% | {:.2} |",
            rate / base,
            100.0 * *accepted as f64 / (*drafted).max(1) as f64,
            *toks as f64 / (toks - accepted).max(1) as f64,
        );
    }
    println!("\n| pass | forward width | forwards | mean ms | vs width 1 |");
    println!("|---|---:|---:|---:|---:|");
    for (pass, f) in passes.iter().zip(&fwd) {
        let one = f[1].1 / f[1].0.max(1) as f64;
        for (w, &(n, secs)) in f.iter().enumerate() {
            if n > 0 {
                let ms = secs / n as f64;
                println!("| {pass} | {w} | {n} | {:.2} | {:.2} |", ms * 1e3, ms / one);
            }
        }
    }
    if texts.len() >= 2 {
        let same = texts[0]
            .iter()
            .zip(&texts[1])
            .filter(|(a, b)| a == b)
            .count();
        println!("\noutputs identical: {same}/{}", texts[0].len());
        for (i, (a, b)) in texts[0].iter().zip(&texts[1]).enumerate() {
            if a != b {
                let at = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
                println!("  #{i} differs at byte {at} of {}/{}", a.len(), b.len());
            }
        }
    }
    Ok(())
}

fn sim_spec(files: &[String]) -> Result<()> {
    use tang_llm::draft::{Calibration, CostModel, DraftConfig, Global, Session, COST_CUDA};
    let cfg = DraftConfig::new(COST_CUDA).from_env();
    let mut global = Global::new(cfg.global_tokens, None);
    let mut calib = Calibration::default();
    let cost = CostModel::new(cfg.max_draft + 1).table(&cfg, cfg.max_draft + 1);
    let (mut toks, mut fwds, mut units, mut drafted, mut accepted) =
        (0usize, 0usize, 0f64, 0usize, 0usize);
    for f in files {
        for line in std::fs::read_to_string(f)?.lines() {
            let v: serde_json::Value = serde_json::from_str(line)?;
            let ids = |k: &str| -> Vec<u32> {
                v[k].as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|x| x.as_u64().map(|x| x as u32))
                    .collect()
            };
            let (prompt, out) = (ids("prompt"), ids("out"));
            let mut s = Session::new(&cfg, &global, calib.clone(), cost.clone(), &prompt);
            let mut j = 0;
            while j < out.len() {
                let d = s.propose(out.len() - j);
                let mut a = 0;
                while a < d.tokens.len() && d.tokens[a] == out[j + a] {
                    s.calib.observe(d.match_len, d.shares[a], true);
                    a += 1;
                }
                if a < d.tokens.len() {
                    s.calib.observe(d.match_len, d.shares[a], false);
                }
                let take = (a + 1).min(out.len() - j);
                for &t in &out[j..j + take] {
                    s.push(t);
                }
                fwds += 1;
                units += cost[1 + d.tokens.len()] as f64;
                drafted += d.tokens.len();
                accepted += a.min(take);
                j += take;
            }
            toks += out.len();
            calib = s.calib.clone();
            global.push(&out);
        }
    }
    println!(
        "tokens {toks} forwards {fwds} tok/forward {:.2} accept {:.0}% ({accepted}/{drafted}) predicted speedup {:.2}x",
        toks as f64 / fwds.max(1) as f64,
        100.0 * accepted as f64 / drafted.max(1) as f64,
        toks as f64 / units.max(1e-9)
    );
    Ok(())
}

fn serve_images<D: ComputeDevice + 'static, F: Fn() -> Result<D> + Send + 'static>(
    make: F,
    args: &[String],
) -> Result<()> {
    let root = PathBuf::from(
        args.first()
            .context("serve-images <pipeline-dir> [--host H] [--port P] [--api-key-file F]")?,
    );
    let (mut host, mut port) = ("127.0.0.1".to_owned(), 8913u16);
    // Same as `serve`: never on the command line itself, where `ps` would show it.
    let mut key = std::env::var("TANG_API_KEY").ok();
    if let Some(k) = &key {
        ensure_nonempty_key(k)?;
    }
    let mut options = args[1..].iter();
    while let Some(option) = options.next() {
        match option.as_str() {
            "--host" => host = options.next().context("--host H")?.clone(),
            "--port" => port = options.next().context("--port P")?.parse()?,
            "--api-key-file" => {
                let value = std::fs::read_to_string(options.next().context("--api-key-file F")?)?;
                ensure_nonempty_key(&value)?;
                key = Some(value.trim().to_owned());
            }
            _ => bail!("unknown image server option {option}"),
        }
    }
    tang_llm::image_server::serve(&format!("{host}:{port}"), root, key, make)
}
fn ensure_nonempty_key(key: &str) -> Result<()> {
    anyhow::ensure!(!key.trim().is_empty(), "empty image server API key");
    Ok(())
}

fn calibrate_image_preview<D: ComputeDevice, F: Fn() -> Result<D>>(
    make: F,
    args: &[String],
) -> Result<()> {
    use std::io::Write;
    anyhow::ensure!(
        args.len() == 2,
        "calibrate-image-preview <vae-dir> <new-output.json>"
    );
    let output = PathBuf::from(&args[1]);
    anyhow::ensure!(!output.exists(), "preview output already exists");
    let record = tang_llm::image_preview::calibrate(make()?, std::path::Path::new(&args[0]))?;
    let bytes = serde_json::to_vec_pretty(&record)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    println!(
        "preview calibration saved to {}; held-out MSE {:.6}, constant baseline {:.6}",
        output.display(),
        record.held_out_mse,
        record.constant_baseline_mse
    );
    Ok(())
}
