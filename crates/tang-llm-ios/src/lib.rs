//! A C ABI over tang-llm for iOS, where the model has to run in-process instead of behind the
//! `tang-llm serve` sidecar. Strings cross as UTF-8; every string tang returns is freed with
//! [`tang_string_free`]. Results and errors come back as JSON.
#![cfg(target_vendor = "apple")]

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::time::Instant;
use tang_compute::MetalDevice;
use tang_llm::chat::Piece;
use tang_llm::sample::Sampling;
use tang_llm::{Dtype, Engine, Request};

pub struct Handle {
    engine: Engine<MetalDevice>,
    load_s: f64,
}

/// Something to prefill: code-shaped, so tokens per character look like a real agent prompt.
const FILLER: &str = "fn parse_line(line: &str) -> Option<(String, u32)> {\n    \
    let (name, count) = line.split_once(':')?;\n    \
    Some((name.trim().to_string(), count.trim().parse().ok()?))\n}\n\n";

fn out(v: Value) -> *mut c_char {
    CString::new(v.to_string()).unwrap_or_default().into_raw()
}

fn err(e: anyhow::Error) -> *mut c_char {
    out(json!({ "error": format!("{e:#}") }))
}

unsafe fn str_arg<'a>(p: *const c_char) -> Result<&'a str> {
    anyhow::ensure!(!p.is_null(), "null string");
    Ok(CStr::from_ptr(p).to_str()?)
}

/// Load a model directory (config.json, tokenizer.json, safetensors). `q4` quantizes bf16
/// weights at load; 4-bit MLX checkpoints load as 4-bit either way. On failure returns null
/// and, if `error` isn't null, a message to free with [`tang_string_free`].
///
/// # Safety
/// `dir` and `error` must be valid C strings/buffers, or null for `error`.
#[no_mangle]
pub unsafe extern "C" fn tang_load(
    dir: *const c_char,
    max_ctx: u32,
    q4: bool,
    error: *mut *mut c_char,
) -> *mut Handle {
    let load = || -> Result<Handle> {
        let dir = str_arg(dir)?;
        let dev = MetalDevice::new().context("no Metal device")?;
        let t = Instant::now();
        let dtype = if q4 { Dtype::Q4 } else { Dtype::Bf16 };
        let engine = Engine::load(dev, Path::new(dir), max_ctx as usize, dtype)?;
        Ok(Handle {
            engine,
            load_s: t.elapsed().as_secs_f64(),
        })
    };
    match load() {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            if !error.is_null() {
                *error = CString::new(format!("{e:#}"))
                    .unwrap_or_default()
                    .into_raw();
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `h` must be a handle from [`tang_load`], freed at most once, or null.
#[no_mangle]
pub unsafe extern "C" fn tang_free(h: *mut Handle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// # Safety
/// `s` must be a string handed out by this API, freed at most once, or null.
#[no_mangle]
pub unsafe extern "C" fn tang_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// What the model is and how long it took to load.
/// # Safety
/// `h` must be a live handle from [`tang_load`] or null.
#[no_mangle]
pub unsafe extern "C" fn tang_info(h: *mut Handle) -> *mut c_char {
    let Some(h) = h.as_ref() else {
        return err(anyhow!("null handle"));
    };
    let c = &h.engine.model.cfg;
    out(json!({
        "model_type": c.model_type,
        "layers": c.num_hidden_layers,
        "hidden": c.hidden_size,
        "context_window": h.engine.context_window(),
        "load_s": h.load_s,
    }))
}

fn usage(u: &tang_llm::engine::Usage, seconds: f64) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "cached_tokens": u.cached_tokens,
        "completion_tokens": u.completion_tokens,
        "prefill_tok_s": u.prefill_tok_s,
        "decode_tok_s": u.decode_tok_s,
        "seconds": seconds,
    })
}

fn run(e: &mut Engine<MetalDevice>, messages: Value, max_tokens: usize) -> Result<(Value, String)> {
    let req = Request {
        messages,
        images: Vec::new(),
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
            ..Sampling::default()
        },
        max_tokens: Some(max_tokens),
        stop: Vec::new(),
        cache_key: None,
        prefill_only: false,
    };
    let mut text = String::new();
    let t = Instant::now();
    let (_, u) = e.complete(&req, |p| {
        if let Piece::Text(s) | Piece::Reasoning(s) = p {
            text.push_str(&s);
        }
        true
    })?;
    Ok((usage(&u, t.elapsed().as_secs_f64()), text))
}

/// A user message of about `tokens` tokens, ending in a request that keeps the model talking.
fn prompt_of(e: &Engine<MetalDevice>, tokens: usize) -> Result<String> {
    let ask = "\nList every function above, then count upward from 1, one number per line.";
    let per = e.count_tokens(FILLER)?.max(1);
    let body = FILLER.repeat(tokens.saturating_sub(40) / per);
    Ok(format!("{body}{ask}"))
}

/// Cold prefill of about `prompt_tokens`, then `gen_tokens` of decode; then a warm follow-up
/// turn that reuses the KV cache, as an agent's next step would.
/// # Safety
/// `h` must be a live handle from [`tang_load`].
#[no_mangle]
pub unsafe extern "C" fn tang_bench(
    h: *mut Handle,
    prompt_tokens: u32,
    gen_tokens: u32,
) -> *mut c_char {
    let Some(h) = h.as_mut() else {
        return err(anyhow!("null handle"));
    };
    let mut bench = || -> Result<Value> {
        let e = &mut h.engine;
        e.reset();
        let user = prompt_of(e, prompt_tokens as usize)?;
        let mut messages = json!([{ "role": "user", "content": user }]);
        let (cold, reply) = run(e, messages.clone(), gen_tokens as usize)?;
        let arr = messages.as_array_mut().unwrap();
        arr.push(json!({ "role": "assistant", "content": reply }));
        arr.push(json!({ "role": "user", "content": "Now only the even numbers." }));
        let (warm, _) = run(e, messages, gen_tokens as usize)?;
        Ok(json!({ "cold": cold, "warm": warm }))
    };
    bench().map(out).unwrap_or_else(err)
}

/// Stream a chat reply. `messages` is an OpenAI-style JSON array. `on_text` gets each piece of
/// text and returns false to stop. Returns the usage as JSON.
/// # Safety
/// `h` must be a live handle from [`tang_load`]; `on_text` and `ctx` must stay
/// valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn tang_chat(
    h: *mut Handle,
    messages: *const c_char,
    max_tokens: u32,
    on_text: extern "C" fn(*const c_char, *mut c_void) -> bool,
    ctx: *mut c_void,
) -> *mut c_char {
    let Some(h) = h.as_mut() else {
        return err(anyhow!("null handle"));
    };
    let mut chat = || -> Result<Value> {
        let messages: Value = serde_json::from_str(str_arg(messages)?)?;
        let req = Request {
            messages,
            images: Vec::new(),
            tools: None,
            response_schema: None,
            think: Some(false),
            thinking_budget: None,
            temperature_set: false,
            top_k_set: false,
            top_p_set: false,
            presence_penalty: None,
            sampling: Sampling::default(),
            max_tokens: Some(max_tokens as usize),
            stop: Vec::new(),
            cache_key: None,
            prefill_only: false,
        };
        let t = Instant::now();
        let (_, u) = h.engine.complete(&req, |p| match p {
            Piece::Text(s) => {
                let c = CString::new(s).unwrap_or_default();
                on_text(c.as_ptr(), ctx)
            }
            _ => true,
        })?;
        Ok(usage(&u, t.elapsed().as_secs_f64()))
    };
    chat().map(out).unwrap_or_else(err)
}
