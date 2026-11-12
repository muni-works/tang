//! Batched decoding through the engine's stream API and the server (see `docs/batching.md`):
//! a long prompt prefills in chunks between the decode steps of streams already running, a
//! background prefill gives way inside a batch and goes on later, and concurrent HTTP
//! requests decode together. Every output must equal the same request run alone.
//!
//! Needs `TANG_LLM_TEST_MODEL` (e.g. `mlx-community/Qwen3-0.6B-4bit`); skipped without it.
//! Run with `--release`.
#![cfg(any(feature = "metal", feature = "cuda"))]

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use tang_compute::ComputeDevice;
use tang_llm::chat::Piece;
use tang_llm::engine::{Control, Engine, Lane, Outcome, Request, Stream};
use tang_llm::sample::Sampling;
use tang_llm::Dtype;

fn model() -> Option<String> {
    let Ok(spec) = std::env::var("TANG_LLM_TEST_MODEL") else {
        eprintln!("TANG_LLM_TEST_MODEL not set: skipping");
        return None;
    };
    Some(spec)
}

fn device() -> impl ComputeDevice {
    #[cfg(feature = "metal")]
    return tang_compute::MetalDevice::new().expect("Metal");
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return tang_compute::CudaComputeDevice::new().expect("CUDA");
}

fn engine(spec: &str, slots: usize) -> Engine<impl ComputeDevice> {
    let dir = tang_llm::resolve_model(spec).unwrap();
    let mut e = Engine::load(device(), &dir, 8192, Dtype::Bf16).unwrap();
    e.set_slots(slots);
    e
}

fn req(user: &str, max: usize) -> Request {
    Request {
        messages: json!([{ "role": "user", "content": user }]),
        images: vec![],
        tools: None,
        response_schema: None,
        think: Some(false),
        thinking_budget: None,
        temperature_set: true,
        top_k_set: false,
        top_p_set: false,
        presence_penalty: None,
        sampling: Sampling {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens: Some(max),
        stop: vec![],
        cache_key: None,
        tenant: None,
        prefill_only: false,
    }
}

fn long_text() -> String {
    (0..3000)
        .map(|i| ["alpha", "beta", "gamma", "delta", "kappa"][(i * 7 + i / 3) % 5])
        .collect::<Vec<_>>()
        .join(" ")
        + "\n\nHow many words above are \"gamma\"? Guess."
}

/// Never yields, or yields at its first chance when `give_way` is set.
#[derive(Default)]
struct Ctl {
    give_way: bool,
}

impl Control for Ctl {
    fn keep_prefilling(&mut self) -> bool {
        !self.give_way
    }
}

/// One step for `streams` (the `None`s sit out).
fn step<D: ComputeDevice>(
    e: &mut Engine<D>,
    streams: &mut [Option<Stream<D::Buffer>>],
    ctls: &mut [Ctl],
) {
    let mut ons: Vec<Box<dyn FnMut(Piece) -> bool>> = Vec::new();
    let mut lanes: Vec<Lane<_>> = Vec::new();
    let mut live: Vec<(&mut Stream<D::Buffer>, &mut Ctl)> = streams
        .iter_mut()
        .zip(ctls.iter_mut())
        .filter_map(|(s, c)| s.as_mut().map(|s| (s, c)))
        .filter(|(s, _)| !s.done())
        .collect();
    for _ in 0..live.len() {
        ons.push(Box::new(|_| true));
    }
    for ((s, c), on) in live.iter_mut().zip(ons.iter_mut()) {
        lanes.push(Lane {
            stream: &mut **s,
            on: &mut **on,
            ctl: &mut **c,
        });
    }
    e.step(&mut lanes).unwrap();
}

fn alone<D: ComputeDevice>(e: &mut Engine<D>, r: &Request) -> Vec<u32> {
    e.reset();
    e.complete(r, |_| true).unwrap();
    e.last_tokens().1.to_vec()
}

/// The finished stream's tokens.
fn retire<D: ComputeDevice>(e: &mut Engine<D>, s: Stream<D::Buffer>) -> Vec<u32> {
    match e.retire(s).unwrap() {
        Outcome::Done(..) => e.last_tokens().1.to_vec(),
        o => panic!("{o:?}"),
    }
}

#[test]
fn a_long_prompt_prefills_between_decode_steps() {
    let Some(spec) = model() else { return };
    let mut e = engine(&spec, 3);
    let short = [
        req("Count from one to forty in words.", 64),
        req("List the planets with one fact each.", 64),
    ];
    let long = req(&long_text(), 8);
    let want: Vec<Vec<u32>> = short
        .iter()
        .chain([&long])
        .map(|r| alone(&mut e, r))
        .collect();
    e.reset();

    let mut streams: Vec<Option<Stream<_>>> = vec![
        Some(e.admit(&short[0], None).unwrap()),
        Some(e.admit(&short[1], None).unwrap()),
        None,
    ];
    let mut ctls: Vec<Ctl> = (0..3).map(|_| Ctl::default()).collect();
    while streams[..2]
        .iter()
        .any(|s| s.as_ref().unwrap().prefilling())
    {
        step(&mut e, &mut streams, &mut ctls);
    }
    step(&mut e, &mut streams, &mut ctls);
    streams[2] = Some(e.admit(&long, None).unwrap());
    let mut prefill_steps = 0;
    while streams[2].as_ref().unwrap().prefilling() {
        let before: Vec<usize> = streams[..2]
            .iter()
            .map(|s| s.as_ref().unwrap().progress().generated)
            .collect();
        step(&mut e, &mut streams, &mut ctls);
        prefill_steps += 1;
        for (i, b) in before.iter().enumerate() {
            let s = streams[i].as_ref().unwrap();
            assert!(
                s.done() || s.progress().generated > *b,
                "stream {i} made no progress while the long prompt prefilled"
            );
        }
        let p = streams[2].as_ref().unwrap().progress();
        assert!(p.prefilled <= prefill_steps * 512, "{p:?}");
    }
    assert!(
        prefill_steps >= 4,
        "{prefill_steps} steps for a 3000-word prompt"
    );
    while streams.iter().any(|s| !s.as_ref().unwrap().done()) {
        step(&mut e, &mut streams, &mut ctls);
    }
    for (i, s) in streams.into_iter().enumerate() {
        assert_eq!(retire(&mut e, s.unwrap()), want[i], "stream {i}");
    }
}

#[test]
fn a_background_prefill_gives_way_inside_a_batch() {
    let Some(spec) = model() else { return };
    let mut e = engine(&spec, 2);
    let chat = req("Write a haiku about rain.", 32);
    let bg = req(&long_text(), 8);
    let want_chat = alone(&mut e, &chat);
    let want_bg = alone(&mut e, &bg);
    e.reset();

    let mut streams = vec![
        Some(e.admit(&chat, None).unwrap()),
        Some(e.admit(&bg, None).unwrap()),
    ];
    let mut ctls = vec![Ctl::default(), Ctl::default()];
    // A chunk in, the background stream is told to give way.
    step(&mut e, &mut streams, &mut ctls);
    ctls[1].give_way = true;
    step(&mut e, &mut streams, &mut ctls);
    assert!(streams[1].as_ref().unwrap().done());
    let out = e.retire(streams[1].take().unwrap()).unwrap();
    let Outcome::Yielded { prefilled, .. } = out else {
        panic!("expected a yield, got {out:?}");
    };
    assert!(prefilled > 0);
    while !streams[0].as_ref().unwrap().done() {
        step(&mut e, &mut streams, &mut ctls);
    }
    assert_eq!(retire(&mut e, streams[0].take().unwrap()), want_chat);
    // It goes on from what it prefilled (its slot kept the cache).
    let (_, u) = e.complete(&bg, |_| true).unwrap();
    assert_eq!(u.cached_tokens, prefilled);
    assert_eq!(e.last_tokens().1, &want_bg[..]);
}

/// One HTTP/1.1 request; the status and the body as JSON.
fn call(addr: &str, method: &str, path: &str, body: &str) -> (u16, Value) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status = out[9..12].parse().unwrap();
    let (head, rest) = out.split_once("\r\n\r\n").unwrap();
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        rest.split_once("\r\n")
            .map_or("", |x| x.1)
            .trim_end_matches("\r\n0\r\n\r\n")
    } else {
        rest
    };
    (status, serde_json::from_str(body).unwrap_or(Value::Null))
}

#[test]
fn the_server_decodes_concurrent_requests_together() {
    let Some(spec) = model() else { return };
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let addr = format!("127.0.0.1:{port}");
    let opts = tang_llm::server::Options {
        addr: addr.clone(),
        key: None,
        model: Some(spec.clone()),
        dtype: Dtype::Bf16,
        hardware: std::sync::Arc::new(tang_llm::node::Hardware::default),
    };
    std::thread::spawn(move || {
        tang_llm::server::serve(opts, |spec: &str| {
            let mut e = Engine::load(device(), &tang_llm::resolve_model(spec)?, 4096, Dtype::Bf16)?;
            e.set_slots(4);
            Ok(e)
        })
    });
    for _ in 0..3000 {
        if TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let prompts = [
        "Name five rivers.",
        "Explain recursion to a child.",
        "Write a limerick about a cat.",
        "What is the capital of Peru? Answer in a sentence.",
    ];
    let body = |p: &str| {
        json!({
            "messages": [{ "role": "user", "content": p }],
            "max_tokens": 48,
            "temperature": 0,
            "chat_template_kwargs": { "enable_thinking": false },
        })
        .to_string()
    };
    let content = |v: &Value| {
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string()
    };
    // One at a time, then all at once.
    let alone: Vec<String> = prompts
        .iter()
        .map(|p| {
            let (status, v) = call(&addr, "POST", "/v1/chat/completions", &body(p));
            assert_eq!(status, 200, "{v}");
            content(&v)
        })
        .collect();
    let watch = {
        let addr = addr.clone();
        std::thread::spawn(move || {
            let mut most = 0;
            for _ in 0..400 {
                let (_, v) = call(&addr, "GET", "/node", "");
                most = most.max(v["queue"]["running"].as_array().map_or(0, |r| r.len()));
                std::thread::sleep(Duration::from_millis(5));
            }
            most
        })
    };
    let runs: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (addr, b) = (addr.clone(), body(p));
            std::thread::spawn(move || call(&addr, "POST", "/v1/chat/completions", &b))
        })
        .collect();
    let together: Vec<String> = runs
        .into_iter()
        .map(|h| {
            let (status, v) = h.join().unwrap();
            assert_eq!(status, 200, "{v}");
            content(&v)
        })
        .collect();
    assert_eq!(together, alone);
    assert!(
        watch.join().unwrap() > 1,
        "/node never showed more than one running"
    );
}

#[test]
fn decode_steps_keep_their_time_budget_while_a_long_prompt_prefills() {
    let Some(spec) = model() else { return };
    let mut e = engine(&spec, 3);
    e.calibrate().unwrap();
    e.calibrate().unwrap();
    let mut streams: Vec<Option<Stream<_>>> = vec![
        Some(
            e.admit(&req("Count from one to a thousand in words.", 4000), None)
                .unwrap(),
        ),
        Some(
            e.admit(
                &req("Tell a very long story about a lighthouse.", 4000),
                None,
            )
            .unwrap(),
        ),
        None,
    ];
    let mut ctls: Vec<Ctl> = (0..3).map(|_| Ctl::default()).collect();
    let timed = |e: &mut Engine<_>, streams: &mut [Option<Stream<_>>], ctls: &mut [Ctl]| {
        let t = std::time::Instant::now();
        step(e, streams, ctls);
        t.elapsed().as_secs_f64()
    };
    let mut plain = Vec::new();
    for i in 0..40 {
        let s = timed(&mut e, &mut streams, &mut ctls);
        if i >= 10 {
            plain.push(s);
        }
    }
    plain.sort_by(f64::total_cmp);
    // A budget a few decode steps long: room for small chunks.
    let budget = (4.0 * plain[plain.len() / 2]).max(0.015);
    e.set_step_budget(Some(budget));
    streams[2] = Some(e.admit(&req(&long_text(), 4), None).unwrap());
    let mut during = Vec::new();
    while streams[2].as_ref().unwrap().prefilling() {
        during.push(timed(&mut e, &mut streams, &mut ctls));
        assert!(during.len() < 100_000);
    }
    during.sort_by(f64::total_cmp);
    let p90 = during[during.len() * 9 / 10];
    eprintln!(
        "decode {:.1} ms; budget {:.1} ms; during prefill p50 {:.1} p90 {p90:.4} max {:.1} ({} steps)",
        plain[plain.len() / 2] * 1e3,
        budget * 1e3,
        during[during.len() / 2] * 1e3,
        during.last().unwrap() * 1e3,
        during.len()
    );
    assert!(
        during.len() > 4,
        "the prompt went in {} steps",
        during.len()
    );
    assert!(p90 <= 1.3 * budget, "p90 {p90} over a {budget} budget");
}
