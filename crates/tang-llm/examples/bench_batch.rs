//! Per-stream and total decode tok/s at several batch sizes (see `docs/batching.md`).
//!
//! `cargo run --release --example bench_batch --no-default-features --features cuda -- <model>
//! [--q4] [--batches 1,2,4,8] [--tokens 256] [--speculate]`
//!
//! Each batch gets that many different prompts, prefilled first; then decode is timed until
//! every stream has `--tokens` tokens (greedy; essays rarely stop that early, and a stream
//! that does just counts fewer). Per-stream tok/s is
//! a stream's tokens over the decode time; total is the batch's tokens over it. It also
//! checks each stream's tokens against the same prompt decoded alone.
use anyhow::Result;
use serde_json::json;
use std::time::Instant;
use tang_compute::ComputeDevice;
use tang_llm::draft::{DraftConfig, COST_CUDA};
use tang_llm::engine::{Control, Engine, Lane, Request, Stream};
use tang_llm::sample::Sampling;
use tang_llm::Dtype;

const TOPICS: [&str; 8] = [
    "the history of lighthouses",
    "how a compiler's register allocator works",
    "growing tomatoes on a balcony",
    "the rules of cricket",
    "why the sky is blue",
    "the design of a bicycle gearbox",
    "how vaccines train the immune system",
    "writing a good bug report",
];

fn device() -> impl ComputeDevice {
    #[cfg(feature = "metal")]
    return tang_compute::MetalDevice::new().expect("Metal");
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return tang_compute::CudaComputeDevice::new().expect("CUDA");
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    return tang_compute::CpuDevice::new();
}

fn request(i: usize, tokens: usize) -> Request {
    Request {
        messages: json!([{ "role": "user", "content": format!("Write a long essay about {}.", TOPICS[i % 8]) }]),
        images: vec![],
        tools: None,
        response_schema: None,
        think: Some(false),
        thinking_budget: None,
        sampling: Sampling {
            temperature: 0.0,
            ..Default::default()
        },
        temperature_set: true,
        top_k_set: false,
        top_p_set: false,
        presence_penalty: None,
        max_tokens: Some(tokens),
        stop: vec![],
        cache_key: None,
        tenant: None,
        prefill_only: false,
    }
}

/// Generated tokens, and the decode-phase seconds and steps.
fn run<D: ComputeDevice>(
    e: &mut Engine<D>,
    reqs: &[Request],
) -> Result<(Vec<Vec<u32>>, f64, usize)> {
    e.reset();
    let mut streams: Vec<Stream<D::Buffer>> = reqs
        .iter()
        .map(|r| e.admit(r, None))
        .collect::<Result<_>>()?;
    let mut decode: Option<Instant> = None;
    let mut steps = 0;
    while streams.iter().any(|s| !s.done()) {
        if decode.is_none() && !streams.iter().any(|s| s.prefilling()) {
            e.model.dev.sync();
            decode = Some(Instant::now());
        }
        let mut ons: Vec<Box<dyn FnMut(tang_llm::chat::Piece) -> bool>> = (0..streams.len())
            .map(|_| Box::new(|_| true) as Box<_>)
            .collect();
        let mut ctls = vec![(); streams.len()];
        let mut lanes: Vec<Lane<_>> = streams
            .iter_mut()
            .filter(|s| !s.done())
            .zip(ons.iter_mut())
            .zip(ctls.iter_mut())
            .map(|((s, on), c)| Lane {
                stream: s,
                on: &mut **on,
                ctl: c as &mut dyn Control,
            })
            .collect();
        e.step(&mut lanes)?;
        if decode.is_some() {
            steps += 1;
        }
    }
    let secs = decode.map_or(0.0, |t| t.elapsed().as_secs_f64());
    let mut outs = Vec::new();
    for s in streams {
        e.retire(s)?;
        outs.push(e.last_tokens().1.to_vec());
    }
    Ok((outs, secs, steps))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: bench_batch <model> [--q4] [--batches 1,2,4,8] [--tokens N] [--speculate] [--long N]");
    let (mut dtype, mut batches, mut tokens, mut speculate) =
        (Dtype::Bf16, vec![1, 2, 4, 8], 256, false);
    let mut long = None;
    let mut step_ms: Option<f64> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--q4" => dtype = Dtype::Q4,
            "--batches" => {
                batches = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|b| b.parse().unwrap())
                    .collect()
            }
            "--tokens" => tokens = args.next().unwrap().parse()?,
            "--speculate" => speculate = true,
            "--long" => long = Some(args.next().unwrap().parse::<usize>()?),
            "--step-ms" => step_ms = Some(args.next().unwrap().parse()?),
            "--hook-cost" => return hook_cost(),
            "--probe" => return probe(&tang_llm::resolve_model(&model)?, dtype),
            o => anyhow::bail!("unknown option {o}"),
        }
    }
    // With `--features otel` and OTEL_EXPORTER_OTLP_ENDPOINT set, steps record metrics (to
    // measure their cost; see docs/observability.md).
    let _otel = tang_llm::telemetry::init();
    let dir = tang_llm::resolve_model(&model)?;
    let ctx = long.map_or(4096, |n| (n + 2048).max(4096));
    let mut e = Engine::load(device(), &dir, ctx, dtype)?;
    let max = *batches.iter().max().unwrap();
    e.set_slots(max + 1);
    if let Some(n) = long {
        // The step budget (0: none, whole chunks), after calibrating the step cost.
        let ms = step_ms.unwrap_or(tang_llm::engine::STEP_BUDGET * 1e3);
        e.set_step_budget((ms > 0.0).then_some(ms / 1e3));
        // Twice: the first also compiles every kernel it touches.
        for _ in 0..2 {
            let t = Instant::now();
            e.calibrate()?;
            println!(
                "calibrated in {:.1}s: {}",
                t.elapsed().as_secs_f64(),
                e.step_cost().json()
            );
        }
        return latency(&mut e, max, n);
    }
    if speculate {
        e.set_speculation(Some(DraftConfig::new(COST_CUDA)));
    }
    // Never stop early, so every stream decodes exactly `tokens`.
    let reqs: Vec<Request> = (0..max).map(|i| request(i, tokens)).collect();
    // Warm up kernels, then each prompt alone for the agreement check.
    run(&mut e, &reqs[..1])?;
    let alone: Vec<Vec<u32>> = (0..max)
        .map(|i| run(&mut e, &reqs[i..i + 1]).map(|r| r.0.into_iter().next().unwrap()))
        .collect::<Result<_>>()?;
    println!("batch | per-stream tok/s | total tok/s | ms/step | same as alone");
    for &b in &batches {
        let (outs, secs, steps) = run(&mut e, &reqs[..b])?;
        let n: usize = outs.iter().map(|o| o.len()).sum();
        let same = outs.iter().zip(&alone).filter(|(a, b)| a == b).count();
        println!(
            "{b:5} | {:16.1} | {:11.1} | {:7.2} | {same}/{b}",
            n as f64 / b as f64 / secs,
            n as f64 / secs,
            secs * 1e3 / steps.max(1) as f64,
        );
    }
    Ok(())
}

/// What the per-step telemetry hook costs, alone: ns per `telemetry::step` call (exporting
/// with `--features otel` and OTEL_EXPORTER_OTLP_ENDPOINT set; an empty call otherwise).
fn hook_cost() -> Result<()> {
    let _otel = tang_llm::telemetry::init();
    let stats = tang_llm::engine::StepStats {
        decode_rows: 8,
        decode_streams: 7,
        prefill_tokens: 256,
        prefill_chunks: 1,
        secs: 0.012,
    };
    let n = 2_000_000;
    for _ in 0..2 {
        let t = Instant::now();
        for _ in 0..n {
            tang_llm::telemetry::step(std::hint::black_box(&stats));
        }
        println!(
            "telemetry::step: {:.1} ns/call",
            t.elapsed().as_secs_f64() * 1e9 / n as f64
        );
    }
    Ok(())
}

/// Decode inter-token latency while an `n`-token prompt prefills alongside `b` decoding
/// streams: each step's time from the long prompt's admission until its prefill is done.
fn latency<D: ComputeDevice>(e: &mut Engine<D>, b: usize, n: usize) -> Result<()> {
    let words: Vec<&str> = (0..n)
        .map(|i| ["alpha", "beta", "gamma", "delta", "kappa"][(i * 7 + i / 3) % 5])
        .collect();
    let mut long = request(0, 4);
    long.messages = json!([{ "role": "user", "content": words.join(" ") + "\n\nSummarize." }]);
    let reqs: Vec<Request> = (0..b).map(|i| request(i, 100_000)).collect();
    // Warm up, including a long prefill.
    run(e, &reqs[..1])?;
    run(e, std::slice::from_ref(&long))?;
    e.reset();
    let mut streams: Vec<Stream<D::Buffer>> = reqs
        .iter()
        .map(|r| e.admit(r, None))
        .collect::<Result<_>>()?;
    let step = |e: &mut Engine<D>, streams: &mut [Stream<D::Buffer>]| -> Result<f64> {
        let mut ons: Vec<Box<dyn FnMut(tang_llm::chat::Piece) -> bool>> = (0..streams.len())
            .map(|_| Box::new(|_| true) as Box<_>)
            .collect();
        let mut ctls = vec![(); streams.len()];
        let mut lanes: Vec<Lane<_>> = streams
            .iter_mut()
            .filter(|s| !s.done())
            .zip(ons.iter_mut())
            .zip(ctls.iter_mut())
            .map(|((s, on), c)| Lane {
                stream: s,
                on: &mut **on,
                ctl: c as &mut dyn Control,
            })
            .collect();
        let t = Instant::now();
        e.step(&mut lanes)?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    };
    // Decoding steadily first.
    let mut plain = Vec::new();
    for i in 0..60 {
        let ms = step(e, &mut streams)?;
        if i >= 20 {
            plain.push(ms);
        }
    }
    streams.push(e.admit(&long, None)?);
    let t = Instant::now();
    let mut during = Vec::new();
    while streams.last().unwrap().prefilling() {
        during.push(step(e, &mut streams)?);
    }
    let prefill_s = t.elapsed().as_secs_f64();
    let stats = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (mean, v[v.len() / 2], v[v.len() * 99 / 100], v[v.len() - 1])
    };
    let (pm, pp, _, px) = stats(&mut plain);
    let (dm, dp, d99, dx) = stats(&mut during);
    let prompt = streams.last().unwrap().progress().prompt_tokens;
    println!("{b} decoding streams; a {prompt}-token prompt joins");
    println!("  decode alone:   step mean {pm:.1} ms, p50 {pp:.1}, max {px:.1}");
    println!(
        "  during prefill: step mean {dm:.1} ms, p50 {dp:.1}, p99 {d99:.1}, max {dx:.1} ({} steps)",
        during.len()
    );
    println!(
        "  prefill took {prefill_s:.2} s ({:.0} tok/s)",
        prompt as f64 / prefill_s
    );
    println!("  fit after: {}", e.step_cost().json());
    for s in streams {
        e.retire(s)?;
    }
    Ok(())
}

/// Step time against prefill chunk size: 4 decode rows plus a chunk of q tokens after c.
fn probe(dir: &std::path::Path, dtype: Dtype) -> Result<()> {
    use tang_llm::model::Segment;
    let e = Engine::load(device(), dir, 20_000, dtype)?;
    let m = &e.model;
    let ids: Vec<u32> = (0..18_000).map(|i| 100 + (i % 1000) as u32).collect();
    let mut rows: Vec<_> = (0..4).map(|_| m.new_cache()).collect();
    for c in rows.iter_mut() {
        m.forward(&ids[..64], c, false)?;
    }
    let one = [7u32];
    let mut long = m.new_cache();
    for c in [0usize, 4096, 16_000] {
        while long.len < c {
            let k = long.len;
            m.forward(&ids[k..(k + 512).min(c)], &mut long, false)?;
        }
        print!("c={c:6}:");
        for q in [0usize, 4, 8, 9, 16, 32, 33, 64, 128, 256, 512] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                long.truncate(c);
                let mut segs: Vec<_> = rows
                    .iter_mut()
                    .map(|r| Segment {
                        tokens: &one,
                        cache: r,
                        all: false,
                    })
                    .collect();
                if q > 0 {
                    segs.push(Segment {
                        tokens: &ids[c..c + q],
                        cache: &mut long,
                        all: false,
                    });
                }
                let t = Instant::now();
                m.forward_batch(&mut segs)?;
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
                drop(segs);
                for r in rows.iter_mut() {
                    r.truncate(64);
                }
            }
            print!(" q{q}={best:.1}");
        }
        println!();
    }
    Ok(())
}
