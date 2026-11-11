//! Text ↔ ids for Flash-Next from the GGUF's own tokenizer metadata (byte-level BPE,
//! `tokenizer.ggml.pre = "qwen35"`), and its chat template.
//!
//! The GGUF carries the vocabulary, merges and token types; this builds the equivalent Hugging
//! Face `tokenizer.json` (NFC, the Qwen 3.5 split regex, byte-level) in memory and loads it with
//! the `tokenizers` crate. Checked against llama.cpp's tokenization of the truth track's prompts
//! (`flash-tokenize --check`).

use crate::chat::Template;
use crate::gguf::Gguf;
use anyhow::{anyhow, ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

/// Qwen 3.5's pre-tokenizer split (llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN35`).
pub const QWEN35_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// `tokenizer.ggml.pre = "glm4"` (GLM-5.3-Flash): like Qwen's but numbers chunk in 1–3 digits.
pub const GLM4_SPLIT: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct FlashTokenizer {
    pub tok: tokenizers::Tokenizer,
    template: Option<Template>,
}

impl FlashTokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arr = |k: &str| -> Result<&[crate::gguf::Value]> {
            g.meta(k)?
                .as_array()
                .with_context(|| format!("{k} is not an array"))
        };
        let tokens: Vec<&str> = arr("tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().unwrap_or(""))
            .collect();
        let merges: Vec<&str> = arr("tokenizer.ggml.merges")?
            .iter()
            .map(|v| v.as_str().unwrap_or(""))
            .collect();
        let types: Vec<u64> = arr("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(1))
            .collect();
        let pre = g.meta_str("tokenizer.ggml.pre").unwrap_or("qwen35");
        ensure!(
            g.meta_str("tokenizer.ggml.model")? == "gpt2",
            "only byte-level BPE (gpt2) GGUF tokenizers are supported"
        );
        ensure!(
            pre.starts_with("qwen") || pre == "glm4",
            "pre-tokenizer {pre} isn't supported"
        );
        let split = if pre == "glm4" {
            GLM4_SPLIT
        } else {
            QWEN35_SPLIT
        };
        let mut vocab = serde_json::Map::new();
        let mut added = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t.to_string(), json!(i));
            // 3 = control, 4 = user-defined: matched literally, never split.
            if matches!(types.get(i), Some(3) | Some(4)) {
                added.push(
                    json!({"id": i, "content": t, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true}),
                );
            }
        }
        let spec = json!({
            "version": "1.0",
            "added_tokens": added,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": split}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": Value::Object(vocab), "merges": merges}
        });
        let tok: tokenizers::Tokenizer = spec
            .to_string()
            .parse()
            .map_err(|e| anyhow!("building the tokenizer: {e}"))?;
        let template = match g.meta_str("tokenizer.chat_template") {
            Ok(src) => {
                let name = |k: &str| {
                    g.meta_u64(k)
                        .ok()
                        .and_then(|i| tokens.get(i as usize))
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                };
                Some(Template::from_source(
                    src.to_string(),
                    name("tokenizer.ggml.bos_token_id"),
                    name("tokenizer.ggml.eos_token_id"),
                )?)
            }
            Err(_) => None,
        };
        Ok(Self { tok, template })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok
            .decode(ids, false)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// The chat template over a conversation (OpenAI messages, contents as strings, tool-call
    /// arguments as objects) and tools, with the generation prompt.
    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        think: Option<bool>,
    ) -> Result<String> {
        let t = self
            .template
            .as_ref()
            .context("the GGUF has no chat template")?;
        t.render(messages, tools, think)
    }

    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.tok.token_to_id(piece)
    }

    /// The chat template applied to one user message (generation prompt added, thinking on
    /// unless `think` says otherwise).
    pub fn chat(&self, user: &str, think: Option<bool>) -> Result<String> {
        let t = self
            .template
            .as_ref()
            .context("the GGUF has no chat template")?;
        t.render(&json!([{"role": "user", "content": user}]), None, think)
    }
}

/// The chat template applied to one user message, tokenized.
pub fn encode_chat(gguf: &Path, text: &str, think: Option<bool>) -> Result<Vec<u32>> {
    let g = Gguf::open(gguf)?;
    let t = FlashTokenizer::from_gguf(&g)?;
    t.encode(&t.chat(text, think)?)
}

/// `tang-llm flash-tokenize <gguf> (--text T | --chat T | --check IDS_FILE...)`: print ids, or
/// check that encode(decode(ids)) == ids for id files tokenized by llama.cpp.
pub fn cli(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let g = Gguf::open(Path::new(it.next().context("gguf path")?))?;
    let t = FlashTokenizer::from_gguf(&g)?;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--text" => println!("{:?}", t.encode(it.next().context("--text T")?)?),
            "--chat" => {
                let s = t.chat(it.next().context("--chat T")?, None)?;
                println!("{s}");
                println!("{:?}", t.encode(&s)?);
            }
            "--check" => {
                for f in it.by_ref() {
                    let s = std::fs::read_to_string(f)?;
                    let ids: Vec<u32> = s
                        .split_whitespace()
                        .map(|w| w.parse())
                        .collect::<Result<_, _>>()?;
                    let text = t.decode(&ids)?;
                    let back = t.encode(&text)?;
                    let first = ids.iter().zip(&back).position(|(a, b)| a != b);
                    println!(
                        "{f}: {} ids, round trip {}{}",
                        ids.len(),
                        if back == ids { "identical" } else { "DIFFERS" },
                        match first {
                            Some(i) => format!(
                                " (first at {i}: {:?} vs {:?})",
                                &ids[i..(i + 5).min(ids.len())],
                                &back[i..(i + 5).min(back.len())]
                            ),
                            None if back.len() != ids.len() =>
                                format!(" (lengths {} vs {})", ids.len(), back.len()),
                            None => String::new(),
                        }
                    );
                }
            }
            s => anyhow::bail!("unknown argument {s}"),
        }
    }
    Ok(())
}
