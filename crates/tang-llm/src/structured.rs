//! Local JSON/schema token constraints, with bounded request compilation.
// The flash server (cuda-only) is the host-side constraint's user.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]
use anyhow::{bail, ensure, Result};
use llguidance::{
    api::TopLevelGrammar,
    toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie},
    JsonCompileOptions, Matcher, ParserFactory,
};
use serde_json::{json, Value};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, LazyLock},
};

const MAX_SCHEMA_BYTES: usize = 64 * 1024;
static VALIDATOR: LazyLock<ParserFactory> = LazyLock::new(|| {
    let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env())
        .expect("byte tokenizer factory");
    factory.quiet();
    factory
});

fn grammar(mut schema: Value) -> TopLevelGrammar {
    if schema.is_boolean() {
        schema = json!({"allOf":[schema]});
    }
    // A skip lexeme is allowed once between grammar terminals. Bound it so a
    // model cannot spend its entire response repeating indentation before a value.
    // Whitespace inside JSON strings remains part of their content, unrestricted.
    JsonCompileOptions {
        whitespace_pattern: Some(r"[ \t\r\n]{1,8}".into()),
        ..Default::default()
    }
    .apply_to(&mut schema);
    TopLevelGrammar::from_json_schema(schema)
}

pub fn response_schema(body: &Value) -> Result<Option<Value>> {
    let format = &body["response_format"];
    if format.is_null() {
        return Ok(None);
    }
    let schema = match format["type"].as_str() {
        Some("text") => return Ok(None),
        Some("json_object") => json!({"type":"object"}),
        Some("json_schema") => format["json_schema"]
            .get("schema")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("response_format.json_schema.schema is required"))?,
        _ => bail!("unsupported response_format type"),
    };
    ensure!(
        schema.is_object() || schema.is_boolean(),
        "schema must be an object or boolean"
    );
    ensure!(
        serde_json::to_vec(&schema)?.len() <= MAX_SCHEMA_BYTES,
        "schema exceeds 64 KiB"
    );
    ensure!(
        body["stop"].is_null() || body["stop"].as_array().is_some_and(Vec::is_empty),
        "stop strings cannot be combined with structured output"
    );
    // Validate without a model or network access, before this request enters the queue.
    catch_unwind(AssertUnwindSafe(|| {
        VALIDATOR.create_parser(grammar(schema.clone()))
    }))
    .map_err(|_| anyhow::anyhow!("invalid structured output schema"))??;
    Ok(Some(schema))
}

pub fn model_factory(
    tokenizer: &tokenizers::Tokenizer,
    vocab: usize,
    eos: u32,
) -> Result<ParserFactory> {
    let json: Value = serde_json::from_str(
        &tokenizer
            .to_string(false)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?,
    )?;
    let mut bytes = catch_unwind(AssertUnwindSafe(|| {
        llguidance::token_bytes_from_tokenizer_json(&json)
    }))
    .map_err(|_| anyhow::anyhow!("unsupported structured-output tokenizer"))??;
    ensure!(
        bytes.len() <= vocab && (eos as usize) < vocab,
        "tokenizer vocabulary differs from model"
    );
    bytes.resize(vocab, Vec::new());
    let env = Arc::new(ApproximateTokEnv::new(TokTrie::from(
        &TokRxInfo::new(vocab as u32, eos),
        &bytes,
    )));
    let mut factory = ParserFactory::new_simple(&(env as llguidance::toktrie::TokEnv))?;
    factory.quiet();
    Ok(factory)
}

pub fn matcher(factory: &ParserFactory, schema: Value) -> Result<Matcher> {
    ensure!(
        serde_json::to_vec(&schema)?.len() <= MAX_SCHEMA_BYTES,
        "schema exceeds 64 KiB"
    );
    let parser = factory.create_parser(grammar(schema))?;
    Ok(Matcher::new(Ok(parser)))
}

pub fn mask(matcher: &mut Matcher, logits: &mut [f32], eos: &[u32]) -> Result<()> {
    let accepting = matcher.is_accepting()?;
    let allowed = matcher.compute_mask_or_eos()?;
    let mut any = false;
    for (token, logit) in logits.iter_mut().enumerate() {
        if (eos.contains(&(token as u32)) && accepting)
            || (!eos.contains(&(token as u32)) && allowed.is_allowed(token as u32))
        {
            any |= logit.is_finite();
        } else {
            *logit = f32::NEG_INFINITY;
        }
    }
    ensure!(any, "structured output has no finite allowed token");
    Ok(())
}

/// How a host-side pick samples when the device's token breaks the grammar.
#[derive(Clone, Copy, Debug)]
pub struct HostSampling {
    /// 0: greedy.
    pub temperature: f32,
    /// 0: no cut.
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for HostSampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Sample a token from (already masked) logits on the host: argmax at temperature 0, else
/// softmax over the top-k / top-p of the finite logits, seeded by `(seed, pos)`.
pub fn host_pick(logits: &[f32], s: &HostSampling, pos: u64) -> u32 {
    let finite = logits.iter().enumerate().filter(|(_, v)| v.is_finite());
    if s.temperature <= 0.0 {
        return finite
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i as u32);
    }
    let mut c: Vec<(usize, f32)> = finite.map(|(i, &v)| (i, v)).collect();
    if c.is_empty() {
        return 0;
    }
    c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if s.top_k > 0 {
        c.truncate(s.top_k);
    }
    let top = c[0].1;
    let w: Vec<f32> = c
        .iter()
        .map(|&(_, v)| ((v - top) / s.temperature).exp())
        .collect();
    let sum: f32 = w.iter().sum();
    let mut keep = w.len();
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let mut cum = 0.0;
        for (j, &x) in w.iter().enumerate() {
            cum += x;
            if cum >= s.top_p * sum {
                keep = j + 1;
                break;
            }
        }
    }
    let total: f32 = w[..keep].iter().sum();
    let u = (splitmix(s.seed ^ splitmix(pos)) >> 40) as f32 / (1u64 << 24) as f32 * total;
    let mut cum = 0.0;
    for j in 0..keep {
        cum += w[j];
        if u < cum {
            return c[j].0 as u32;
        }
    }
    c[keep - 1].0 as u32
}

/// One request's grammar state over the model's vocabulary, for engines that pick tokens on
/// the device. The device's token is kept when the grammar allows it: masking only removes
/// candidates, so an allowed greedy pick is the masked argmax, and for sampling, keeping an
/// allowed draw and redrawing from the masked distribution otherwise is exactly sampling from
/// the masked distribution. Only a rejected token costs a full mask on the host.
pub struct Constraint {
    m: Matcher,
    eos: Vec<u32>,
    /// Tokens the host had to pick (the device's pick broke the grammar).
    pub host_picks: usize,
}

impl Constraint {
    pub fn new(factory: &ParserFactory, schema: Value, eos: Vec<u32>) -> Result<Self> {
        ensure!(!eos.is_empty(), "model has no EOS token");
        Ok(Self {
            m: matcher(factory, schema)?,
            eos,
            host_picks: 0,
        })
    }

    /// Whether `tok` may come next (an end of turn only once the value is complete).
    pub fn allows(&mut self, tok: u32) -> Result<bool> {
        if self.eos.contains(&tok) {
            return self.m.is_accepting();
        }
        Ok(!self.m.is_stopped() && self.m.validate_tokens(&[tok])? == 1)
    }

    /// The token to emit: `proposed` (the device's pick) when allowed, else a host pick from
    /// `logits` (this position's row; masked in place).
    pub fn choose(
        &mut self,
        proposed: u32,
        logits: &mut [f32],
        s: &HostSampling,
        pos: u64,
    ) -> Result<u32> {
        if self.allows(proposed)? {
            return Ok(proposed);
        }
        self.host_picks += 1;
        mask(&mut self.m, logits, &self.eos)?;
        let tok = host_pick(logits, s, pos);
        ensure!(logits[tok as usize].is_finite(), "no allowed token");
        Ok(tok)
    }

    /// Advance past an emitted token (end-of-turn tokens aren't fed to the grammar).
    pub fn accept(&mut self, tok: u32) -> Result<()> {
        if !self.eos.contains(&tok) {
            self.m.consume_token(tok)?;
        }
        Ok(())
    }

    /// The JSON value is complete and nothing may follow it: the reply is done.
    pub fn finished(&self) -> bool {
        self.m.is_stopped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_formats_are_bounded_and_validated() {
        assert!(response_schema(&json!({})).unwrap().is_none());
        assert!(response_schema(&json!({"response_format":{"type":"text"}}))
            .unwrap()
            .is_none());
        assert!(
            response_schema(&json!({"response_format":{"type":"json_object"}}))
                .unwrap()
                .is_some()
        );
        for format in [
            json!({"type":"bad"}),
            json!({"type":"json_schema"}),
            json!({"type":"json_schema","json_schema":{"schema":{"type":"bogus"}}}),
        ] {
            assert!(response_schema(&json!({"response_format":format})).is_err());
        }
        assert!(
            response_schema(&json!({"stop":"}","response_format":{"type":"json_object"}})).is_err()
        );
        assert!(response_schema(&json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"description":"x".repeat(MAX_SCHEMA_BYTES)}}}})).is_err());
    }
    #[test]
    fn schema_masks_bad_keys_values_and_premature_eos() {
        let schema = json!({"type":"object","properties":{"tool":{"const":"Write"},"args":{"type":"object","properties":{"content":{"type":"string"}},"required":["content"],"additionalProperties":false}},"required":["tool","args"],"additionalProperties":false});
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let valid = b"{\"tool\":\"Write\",\"args\":{\"content\":\"<div>frog</div>\\n\"}}";
        let mut grammar = matcher(&factory, schema.clone()).unwrap();
        for &byte in valid {
            let mut logits = vec![0.; 257];
            mask(&mut grammar, &mut logits, &[256]).unwrap();
            assert!(logits[byte as usize].is_finite());
            grammar.consume_token(byte as u32).unwrap();
        }
        assert!(grammar.is_accepting().unwrap());
        let mut logits = vec![0.; 257];
        mask(&mut grammar, &mut logits, &[255, 256]).unwrap();
        assert!(logits[255].is_finite() && logits[256].is_finite());
        for invalid in [
            b"{\"tool\":\"Bash\"}".as_slice(),
            b"{\"args\":{\"bad\":1}}".as_slice(),
            b"{\"tool\"<".as_slice(),
        ] {
            let mut grammar = matcher(&factory, schema.clone()).unwrap();
            let mut rejected = false;
            for &byte in invalid {
                let allowed = grammar.compute_mask().unwrap();
                if !allowed.is_allowed(byte as u32) {
                    rejected = true;
                    break;
                }
                grammar.consume_token(byte as u32).unwrap();
            }
            assert!(rejected);
        }
    }

    #[test]
    fn accepting_number_can_continue_and_eos_is_blocked_before_acceptance() {
        let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
        let mut number = matcher(&factory, json!({"type":"integer"})).unwrap();
        number.consume_token(b'1' as u32).unwrap();
        let mut logits = vec![0.; 257];
        mask(&mut number, &mut logits, &[256]).unwrap();
        assert!(logits[b'0' as usize].is_finite() && logits[256].is_finite());
        let mut object = matcher(&factory, json!({"type":"object"})).unwrap();
        object.consume_token(b'{' as u32).unwrap();
        logits.fill(0.);
        mask(&mut object, &mut logits, &[255, 256]).unwrap();
        assert!(!logits[255].is_finite() && !logits[256].is_finite());
    }

    #[test]
    fn indentation_cannot_loop_but_string_whitespace_is_preserved() {
        let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
        let mut object = matcher(&factory, json!({"type":"object","properties":{"count":{"const":7}},"required":["count"],"additionalProperties":false})).unwrap();
        for &byte in b"{\"count\":" {
            object.consume_token(byte as u32).unwrap();
        }
        for _ in 0..8 {
            let allowed = object.compute_mask().unwrap();
            assert!(allowed.is_allowed(b' ' as u32));
            object.consume_token(b' ' as u32).unwrap();
        }
        let allowed = object.compute_mask().unwrap();
        assert!(!allowed.is_allowed(b' ' as u32));
        assert!(allowed.is_allowed(b'7' as u32));
        let mut string = matcher(&factory, json!({"type":"string"})).unwrap();
        for &byte in serde_json::to_vec(&format!("{}\n", " ".repeat(30)))
            .unwrap()
            .iter()
        {
            let allowed = string.compute_mask().unwrap();
            assert!(allowed.is_allowed(byte as u32));
            string.consume_token(byte as u32).unwrap();
        }
        assert!(string.is_accepting().unwrap());
        assert!(response_schema(
            &json!({"response_format":{"type":"json_schema","json_schema":{"schema":true}}})
        )
        .unwrap()
        .is_some());
    }

    /// A small BPE-like vocabulary: every byte, some multi-byte pieces (several of them
    /// spanning JSON structure), and an end-of-turn token last.
    fn fake_vocab() -> (Vec<Vec<u8>>, u32) {
        let mut words: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for w in [
            "{\"", "\":", "\",", "\"}", "\": \"", "true", "false", "null", "name", "age", "[1",
            "},{", "\"a", "12", ".5", "e-", "  ", "\n", "\\n", "\\u00", "ok\"", "\"]", "]}",
            "tags", "Write", "id\":",
        ] {
            words.push(w.as_bytes().to_vec());
        }
        words.push(b"\xFF<|im_end|>".to_vec());
        let eos = words.len() as u32 - 1;
        (words, eos)
    }

    fn fake_factory() -> (ParserFactory, Vec<Vec<u8>>, u32) {
        let (words, eos) = fake_vocab();
        let env = Arc::new(ApproximateTokEnv::new(TokTrie::from(
            &TokRxInfo::new(words.len() as u32, eos),
            &words,
        )));
        let mut f = ParserFactory::new_simple(&(env as llguidance::toktrie::TokEnv)).unwrap();
        f.quiet();
        (f, words, eos)
    }

    /// Enough of JSON Schema to check the supported subset.
    fn valid(v: &Value, s: &Value) -> bool {
        if let Some(c) = s.get("const") {
            return v == c;
        }
        if let Some(e) = s.get("enum").and_then(Value::as_array) {
            return e.contains(v);
        }
        let ty_ok = match s.get("type").and_then(Value::as_str) {
            None => true,
            Some("object") => v.is_object(),
            Some("array") => v.is_array(),
            Some("string") => v.is_string(),
            Some("number") => v.is_number(),
            Some("integer") => v.is_i64() || v.is_u64(),
            Some("boolean") => v.is_boolean(),
            Some("null") => v.is_null(),
            Some(_) => false,
        };
        if !ty_ok {
            return false;
        }
        if let Some(o) = v.as_object() {
            let props = s.get("properties").and_then(Value::as_object);
            for r in s
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if !o.contains_key(r.as_str().unwrap()) {
                    return false;
                }
            }
            for (k, x) in o {
                match props.and_then(|p| p.get(k)) {
                    Some(ps) if !valid(x, ps) => return false,
                    Some(_) => {}
                    None if s.get("additionalProperties") == Some(&json!(false)) => return false,
                    None => {}
                }
            }
        }
        if let (Some(a), Some(items)) = (v.as_array(), s.get("items")) {
            if !a.iter().all(|x| valid(x, items)) {
                return false;
            }
            if let Some(n) = s.get("maxItems").and_then(Value::as_u64) {
                if a.len() as u64 > n {
                    return false;
                }
            }
        }
        if let (Some(t), Some(n)) = (v.as_str(), s.get("maxLength").and_then(Value::as_u64)) {
            if t.chars().count() as u64 > n {
                return false;
            }
        }
        true
    }

    /// Decode with random logits the way the flash server does: the "device" proposes the
    /// raw argmax (or a random token), the constraint keeps it or picks on the host.
    fn decode_random(
        f: &ParserFactory,
        words: &[Vec<u8>],
        eos: u32,
        schema: &Value,
        seed: u64,
        hs: &HostSampling,
    ) -> (Vec<u8>, usize) {
        let mut c = Constraint::new(f, schema.clone(), vec![eos]).unwrap();
        let mut out = Vec::new();
        let mut rng = seed;
        for step in 0..20_000u64 {
            let mut logits: Vec<f32> = (0..words.len())
                .map(|_| {
                    rng = splitmix(rng);
                    (rng >> 40) as f32 / (1u64 << 24) as f32 * 8.0
                })
                .collect();
            // Make the end of turn tempting, so streams don't wander forever.
            logits[eos as usize] += 4.0;
            let proposed = if step % 3 == 0 {
                (rng % words.len() as u64) as u32
            } else {
                host_pick(&logits, &HostSampling::default(), 0)
            };
            let tok = c.choose(proposed, &mut logits, hs, step).unwrap();
            if tok == eos {
                return (out, c.host_picks);
            }
            c.accept(tok).unwrap();
            out.extend_from_slice(&words[tok as usize]);
            if c.finished() {
                assert!(c.allows(eos).unwrap(), "finished but EOS not allowed");
                return (out, c.host_picks);
            }
        }
        panic!(
            "no end after 20000 tokens: {}",
            String::from_utf8_lossy(&out)
        );
    }

    #[test]
    fn fake_vocab_masks_structure_and_multi_byte_tokens() {
        let (f, words, eos) = fake_factory();
        let id = |w: &str| words.iter().position(|x| x == w.as_bytes()).unwrap() as u32;
        let schema = json!({"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false});
        let mut c = Constraint::new(&f, schema, vec![eos]).unwrap();
        assert!(c.allows(id("{\"")).unwrap());
        assert!(c.allows(id("{")).unwrap());
        assert!(!c.allows(id("[1")).unwrap());
        assert!(!c.allows(eos).unwrap());
        c.accept(id("{\"")).unwrap();
        assert!(c.allows(id("name")).unwrap());
        assert!(!c.allows(id("age")).unwrap());
        // A rejected proposal is replaced by the best allowed token.
        let mut logits = vec![0.0f32; words.len()];
        logits[id("age") as usize] = 9.0;
        logits[id("n") as usize] = 5.0;
        logits[id("name") as usize] = 4.0;
        let t = c
            .choose(id("age"), &mut logits, &HostSampling::default(), 0)
            .unwrap();
        assert_eq!(t, id("n"));
        assert_eq!(c.host_picks, 1);
        assert!(!logits[id("age") as usize].is_finite());
        for w in ["name", "\": \"", "ok\"", "}"] {
            assert!(c.allows(id(w)).unwrap(), "{w}");
            c.accept(id(w)).unwrap();
        }
        assert!(c.finished());
        assert!(c.allows(eos).unwrap());
        assert!(!c.allows(id(" ")).unwrap());
        // An allowed proposal stands, whatever the logits say.
        let mut c = Constraint::new(&f, json!({"type":"integer"}), vec![eos]).unwrap();
        let mut logits = vec![0.0f32; words.len()];
        logits[id("7") as usize] = 9.0;
        assert_eq!(
            c.choose(id("12"), &mut logits, &HostSampling::default(), 0)
                .unwrap(),
            id("12")
        );
        assert_eq!(c.host_picks, 0);
    }

    #[test]
    fn random_logits_always_give_schema_valid_json() {
        let (f, words, eos) = fake_factory();
        let schemas = [
            json!({"type":"object"}),
            json!({"type":"object","properties":{"name":{"type":"string","maxLength":12},"age":{"type":"integer"},"ok":{"type":"boolean"}},"required":["name","age","ok"],"additionalProperties":false}),
            json!({"type":"array","items":{"type":"number"},"maxItems":5}),
            json!({"type":"object","properties":{"tool":{"const":"Write"},"mode":{"enum":["a","b",3,null]},"args":{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string","maxLength":6},"maxItems":3},"n":{"type":"null"}},"required":["tags","n"],"additionalProperties":false}},"required":["tool","mode","args"],"additionalProperties":false}),
            json!({"type":"string","maxLength":20}),
            json!({"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"array","items":{"type":"object","properties":{"c":{"type":"integer"}},"required":["c"],"additionalProperties":false},"maxItems":2}},"required":["b"],"additionalProperties":false}},"required":["a"],"additionalProperties":false}),
        ];
        let mut host = 0;
        for (k, schema) in schemas.iter().enumerate() {
            for seed in 0..40u64 {
                let hs = HostSampling {
                    temperature: if seed % 2 == 0 { 0.0 } else { 0.8 },
                    top_k: 20,
                    top_p: 0.95,
                    seed,
                };
                let (out, picks) =
                    decode_random(&f, &words, eos, schema, seed * 1000 + k as u64, &hs);
                host += picks;
                let text = String::from_utf8(out).expect("utf-8");
                let v: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    // Valid JSON (RFC 8259 puts no bound on exponents) that serde's f64
                    // can't hold: random logits love `7e71204882`.
                    Err(e) if e.to_string().starts_with("number out of range") => continue,
                    Err(e) => panic!("schema {k} seed {seed}: {e}: {text}"),
                };
                assert!(valid(&v, schema), "schema {k} seed {seed}: {text}");
            }
        }
        assert!(host > 0, "the host never had to pick");
    }

    #[test]
    fn host_pick_is_greedy_at_zero_and_stays_in_top_k() {
        let mut logits = vec![f32::NEG_INFINITY; 10];
        logits[3] = 1.0;
        logits[7] = 2.0;
        logits[8] = 0.5;
        assert_eq!(host_pick(&logits, &HostSampling::default(), 0), 7);
        let hs = HostSampling {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            seed: 9,
        };
        let mut seen = [0usize; 10];
        for pos in 0..400 {
            seen[host_pick(&logits, &hs, pos) as usize] += 1;
        }
        assert_eq!(seen[8], 0);
        assert!(seen[3] > 0 && seen[7] > seen[3]);
        assert_eq!(seen.iter().sum::<usize>(), 400);
    }

    #[test]
    fn chat_request_parsing_carries_the_schema() {
        let body = json!({
            "model": "qwen3-coder-next",
            "messages": [{"role":"user","content":"Give me a person"}],
            "response_format": {"type":"json_schema","json_schema":{"name":"person","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}}}
        });
        let req = crate::server::parse(&body).unwrap();
        let schema = req.response_schema.clone().unwrap();
        assert_eq!(schema["required"], json!(["name"]));
        let (f, _, eos) = fake_factory();
        Constraint::new(&f, schema, vec![eos]).unwrap();
        let mut obj = body.clone();
        obj["response_format"] = json!({"type":"json_object"});
        assert_eq!(
            crate::server::parse(&obj).unwrap().response_schema,
            Some(json!({"type":"object"}))
        );
        obj["response_format"] = json!({"type":"text"});
        assert!(crate::server::parse(&obj)
            .unwrap()
            .response_schema
            .is_none());
        obj["response_format"] = json!({"type":"json_schema","json_schema":{"name":"x"}});
        assert!(crate::server::parse(&obj).is_err());
        obj["response_format"] = json!({"type":"json_object"});
        obj["stop"] = json!(["}"]);
        assert!(crate::server::parse(&obj).is_err());
    }
}
