//! Chat templates (the checkpoint's own Jinja template, rendered with minijinja) and a streaming
//! parser that splits model output into reasoning, text and tool calls.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

pub struct Template {
    env: minijinja::Environment<'static>,
    bos: String,
    eos: String,
}

/// Drop Hugging Face's `{% generation %}` / `{% endgeneration %}` markers: they tell the
/// tokenizer which span to mask when training, and render as nothing. minijinja has no such tag,
/// so a template carrying them (Inkling 2 Mini's does) would fail to parse.
fn strip_generation_tags(src: &str) -> String {
    src.replace("{%- generation -%}", "")
        .replace("{% generation %}", "")
        .replace("{%- endgeneration -%}", "")
        .replace("{% endgeneration %}", "")
}

impl Template {
    /// Load `chat_template` (and bos/eos strings) from `tokenizer_config.json`.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer_config.json")).context("tokenizer_config.json")?,
        )?;
        let source = match &cfg["chat_template"] {
            Value::String(s) => s.clone(),
            // A list of named templates: use "default".
            Value::Array(list) => list
                .iter()
                .find(|t| t["name"] == "default")
                .or(list.first())
                .and_then(|t| t["template"].as_str())
                .context("chat_template list")?
                .to_string(),
            // Or beside it: `chat_template.jinja`, or `chat_template.json` (Qwen3-VL).
            _ => match std::fs::read_to_string(dir.join("chat_template.jinja")) {
                Ok(t) => t,
                Err(_) => serde_json::from_slice::<Value>(
                    &std::fs::read(dir.join("chat_template.json"))
                        .context("no chat_template in tokenizer_config.json")?,
                )?["chat_template"]
                    .as_str()
                    .context("chat_template.json has no chat_template")?
                    .to_string(),
            },
        };
        let source = strip_generation_tags(&source);
        let special = |k: &str| match &cfg[k] {
            Value::String(s) => s.clone(),
            v => v["content"].as_str().unwrap_or_default().to_string(),
        };
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", source)?;
        Ok(Self {
            env,
            bos: special("bos_token"),
            eos: special("eos_token"),
        })
    }

    /// A template from its Jinja source and bos/eos strings (e.g. a GGUF's metadata).
    pub fn from_source(source: String, bos: String, eos: String) -> Result<Self> {
        let source = strip_generation_tags(&source);
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", source)?;
        Ok(Self { env, bos, eos })
    }

    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        think: Option<bool>,
    ) -> Result<String> {
        let t = self.env.get_template("chat")?;
        // No tools: undefined, not none (Qwen3-Coder-Next's template takes `tools | length`
        // whenever `tools is defined`).
        let tools = tools
            .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
            .map_or(
                minijinja::Value::UNDEFINED,
                minijinja::Value::from_serialize,
            );
        // Null fields dropped, not passed as none: the same template takes
        // `message.tool_calls | length` when `tool_calls is defined`, and a plain assistant turn
        // arrives with `"tool_calls": null` from the request normalizer and many clients.
        let mut messages = messages.clone();
        if let Some(ms) = messages.as_array_mut() {
            for m in ms.iter_mut().filter_map(Value::as_object_mut) {
                m.retain(|_, v| !v.is_null());
            }
        }
        Ok(t.render(minijinja::context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => true,
            enable_thinking => think.unwrap_or(true),
            bos_token => self.bos,
            eos_token => self.eos,
        })?)
    }
}

/// transformers' `strftime_now` for templates that print today's date (gpt-oss): UTC, with
/// `%Y %m %d %B %b %H %M %S`.
fn strftime_now(fmt: String) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    strftime(&fmt, secs)
}

fn strftime(fmt: &str, secs: i64) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('B') => out.push_str(MONTHS[m as usize - 1]),
            Some('b') => out.push_str(&MONTHS[m as usize - 1][..3]),
            Some('H') => out.push_str(&format!("{:02}", rem / 3600)),
            Some('M') => out.push_str(&format!("{:02}", rem / 60 % 60)),
            Some('S') => out.push_str(&format!("{:02}", rem % 60)),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Python's `json.dumps` spacing (`", "` and `": "`), which is what templates were trained with.
fn tojson(v: minijinja::Value) -> Result<minijinja::Value, minijinja::Error> {
    struct Py;
    impl serde_json::ser::Formatter for Py {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Py);
    v.serialize(&mut ser).map_err(|e| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
    })?;
    Ok(minijinja::Value::from_safe_string(
        String::from_utf8(out).unwrap_or_default(),
    ))
}

/// What the model is producing, as it streams.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Text(String),
    /// A complete `<tool_call>` body: `{"name": ..., "arguments": ...}`.
    ToolCall {
        name: String,
        arguments: Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Text,
    Think,
    Tool,
}

/// Splits a completion into reasoning, text and tool calls. Tags (`<think>`, `</think>`,
/// `<tool_call>`, `</tool_call>`) come in by token id through [`push_tag`](Self::push_tag);
/// [`push_text`](Self::push_text) is always literal, so a tag the model spells out in ordinary
/// tokens (or quotes from a user message) never opens a tool call or closes the reasoning.
///
/// A tool call's body is Hermes JSON (`{"name": ..., "arguments": {...}}`) or Qwen XML
/// (`<function=NAME><parameter=KEY>\nVALUE\n</parameter>...</function>`, see
/// [`parse_xml_call`]). Inside a call only `</tool_call>` is a tag, and an XML call's
/// `</tool_call>` only closes it once the body parses, so values may contain any of the tags.
/// Inside `<think>`, tool-call tags are reasoning text.
pub struct Parser {
    /// gpt-oss's harmony format instead of think/tool-call tags.
    harmony: Option<Harmony>,
    mode: Mode,
    tool: String,
    /// Text seen so far (to trim the blank lines the template puts around tags).
    text_started: bool,
    /// Tool name → its JSON schema `parameters`, for typing XML parameter values.
    schemas: std::collections::HashMap<String, Value>,
}

pub const TAGS: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];

/// gpt-oss's harmony format: every message is `<|start|>{header}<|message|>{body}` closed by
/// `<|end|>`, `<|call|>` (a tool call) or `<|return|>` (the last one). The header names the
/// channel (`<|channel|>analysis`, `commentary` or `final`) and, for a tool call, the recipient
/// (`to=functions.NAME`, with `<|constrain|>json`).
pub const HARMONY_TAGS: [&str; 7] = [
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|call|>",
    "<|return|>",
];

/// Where a harmony completion is: in a header (collecting it) or in a message body.
#[derive(Debug, Clone, PartialEq)]
enum Harmony {
    Header(String),
    Body(Body),
}

#[derive(Debug, Clone, PartialEq)]
enum Body {
    Reasoning,
    Text,
    /// A call to `name`, its arguments so far.
    Tool {
        name: String,
        args: String,
    },
}

/// The model's tag tokens: id → tag. A tag the vocabulary has no single token for is never
/// recognised.
#[derive(Debug, Clone, Default)]
pub struct Tags(Vec<(u32, &'static str)>);

impl Tags {
    pub fn new(id: impl Fn(&str) -> Option<u32>) -> Self {
        Self(
            TAGS.iter()
                .chain(&HARMONY_TAGS)
                .filter_map(|t| Some((id(t)?, *t)))
                .collect(),
        )
    }

    /// Whether the vocabulary speaks harmony (gpt-oss).
    pub fn harmony(&self) -> bool {
        self.0.iter().any(|(_, t)| *t == "<|channel|>")
    }

    pub fn get(&self, id: u32) -> Option<&'static str> {
        self.0.iter().find(|(i, _)| *i == id).map(|(_, t)| *t)
    }
}

/// Decoded output: text from ordinary tokens, or a tag token.
#[derive(Debug, Clone, PartialEq)]
pub enum Chunk {
    Text(String),
    Tag(&'static str),
}

/// Incremental detokenization: decode a small window, emit what's new, hold back broken
/// UTF-8. Tag tokens are never decoded into the text: they come out as [`Chunk::Tag`].
#[derive(Debug, Default)]
pub struct Detok {
    prefix: usize,
    read: usize,
    seen: usize,
}

impl Detok {
    /// What `out` (all tokens so far) adds since the last call.
    pub fn push(
        &mut self,
        out: &[u32],
        tags: &Tags,
        decode: impl Fn(&[u32]) -> Result<String>,
    ) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let new_text = |prefix: usize, read: usize, end: usize| -> Result<Option<String>> {
            let before = decode(&out[prefix..read])?;
            let after = decode(&out[prefix..end])?;
            Ok(after
                .get(before.len()..)
                .filter(|n| !n.is_empty())
                .map(str::to_string))
        };
        for i in self.seen..out.len() {
            if let Some(tag) = tags.get(out[i]) {
                // Everything before the tag, broken UTF-8 or not.
                chunks.extend(new_text(self.prefix, self.read, i)?.map(Chunk::Text));
                chunks.push(Chunk::Tag(tag));
                self.prefix = i + 1;
                self.read = i + 1;
            }
        }
        self.seen = out.len();
        if self.read < out.len() {
            if let Some(new) =
                new_text(self.prefix, self.read, out.len())?.filter(|n| !n.ends_with('\u{fffd}'))
            {
                chunks.push(Chunk::Text(new));
                self.prefix = self.read;
                self.read = out.len();
            }
        }
        Ok(chunks)
    }
}

impl Parser {
    /// `thinking`: the prompt already opened a `<think>` block (some templates do).
    pub fn new(thinking: bool) -> Self {
        Self {
            harmony: None,
            mode: if thinking { Mode::Think } else { Mode::Text },
            tool: String::new(),
            text_started: false,
            schemas: Default::default(),
        }
    }

    /// A parser for harmony output (gpt-oss), after a prompt that ends `<|start|>assistant`.
    pub fn harmony() -> Self {
        Self {
            harmony: Some(Harmony::Header("assistant".into())),
            ..Self::new(false)
        }
    }

    /// [`harmony`](Self::harmony) if `tags` are harmony's, else [`new`](Self::new).
    pub fn for_tags(tags: &Tags, thinking: bool) -> Self {
        if tags.harmony() {
            Self::harmony()
        } else {
            Self::new(thinking)
        }
    }

    /// The request's `tools` (OpenAI format), so XML parameter values get their schema's types.
    pub fn with_tools(mut self, tools: Option<&Value>) -> Self {
        for t in tools.and_then(Value::as_array).into_iter().flatten() {
            let f = if t["function"].is_object() {
                &t["function"]
            } else {
                t
            };
            if let Some(name) = f["name"].as_str() {
                self.schemas
                    .insert(name.to_string(), f["parameters"].clone());
            }
        }
        self
    }

    /// Text from ordinary tokens: never a tag, whatever it spells.
    pub fn push_text(&mut self, s: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.harmony.is_some() {
            self.harmony_text(s, &mut out);
            return out;
        }
        self.emit(s, &mut out);
        out
    }

    /// A tag token.
    pub fn push_tag(&mut self, tag: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.harmony.is_some() {
            self.harmony_tag(tag, &mut out);
            return out;
        }
        if self.is_text(tag) {
            self.emit(tag, &mut out);
        } else {
            self.tag(tag, &mut out);
        }
        out
    }

    pub fn push_chunk(&mut self, c: &Chunk) -> Vec<Piece> {
        match c {
            Chunk::Text(s) => self.push_text(s),
            Chunk::Tag(t) => self.push_tag(t),
        }
    }

    /// Flush at end of generation (an unterminated tool call is returned as text, unless it is
    /// a complete XML call that only lacks its `</tool_call>`).
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.harmony.is_some() {
            // Generation stops at `<|call|>` or `<|return|>` before the parser sees them.
            self.harmony_close(&mut out);
            return out;
        }
        if self.mode == Mode::Tool && !self.tool.trim().is_empty() {
            let body = std::mem::take(&mut self.tool);
            match parse_xml_call(&body, &self.schemas) {
                Some((name, arguments)) => out.push(Piece::ToolCall { name, arguments }),
                None => out.push(Piece::Text(format!("<tool_call>{body}"))),
            }
        }
        out
    }

    /// A tag that is plain text where it appears: tool-call tags inside reasoning, and anything
    /// but a closing `</tool_call>` that completes the call inside a call.
    fn is_text(&self, tag: &str) -> bool {
        match self.mode {
            Mode::Think => tag == "<tool_call>" || tag == "</tool_call>",
            Mode::Tool if tag != "</tool_call>" => true,
            Mode::Tool => {
                let b = self.tool.trim_start();
                b.starts_with("<function=") && parse_xml_call(&self.tool, &self.schemas).is_none()
            }
            Mode::Text => false,
        }
    }

    fn tag(&mut self, tag: &str, out: &mut Vec<Piece>) {
        match tag {
            "<think>" => self.mode = Mode::Think,
            "</think>" => self.mode = Mode::Text,
            "<tool_call>" => {
                self.mode = Mode::Tool;
                self.tool.clear();
            }
            _ => {
                self.mode = Mode::Text;
                let body = std::mem::take(&mut self.tool);
                if let Some((name, arguments)) = parse_xml_call(&body, &self.schemas) {
                    out.push(Piece::ToolCall { name, arguments });
                    return;
                }
                match serde_json::from_str::<Value>(body.trim()) {
                    Ok(v) if v["name"].is_string() => out.push(Piece::ToolCall {
                        name: v["name"].as_str().unwrap_or_default().to_string(),
                        arguments: match &v["arguments"] {
                            Value::String(s) => {
                                serde_json::from_str(s).unwrap_or(Value::String(s.clone()))
                            }
                            Value::Null => Value::Object(Default::default()),
                            a => a.clone(),
                        },
                    }),
                    _ => out.push(Piece::Text(format!("<tool_call>{body}</tool_call>"))),
                }
            }
        }
    }

    fn harmony_text(&mut self, s: &str, out: &mut Vec<Piece>) {
        match self.harmony.as_mut() {
            Some(Harmony::Header(h)) => h.push_str(s),
            Some(Harmony::Body(Body::Tool { args, .. })) => args.push_str(s),
            Some(Harmony::Body(Body::Reasoning)) => push(out, Piece::Reasoning(s.to_string())),
            Some(Harmony::Body(Body::Text)) => self.emit(s, out),
            None => {}
        }
    }

    fn harmony_tag(&mut self, tag: &str, out: &mut Vec<Piece>) {
        match (self.harmony.as_mut(), tag) {
            (Some(Harmony::Header(h)), "<|channel|>" | "<|constrain|>") => {
                h.push(' ');
                h.push_str(tag);
            }
            (Some(Harmony::Header(h)), "<|message|>") => {
                let body = harmony_body(h);
                self.harmony = Some(Harmony::Body(body));
            }
            (_, "<|start|>") => {
                self.harmony_close(out);
                self.harmony = Some(Harmony::Header(String::new()));
            }
            (_, "<|end|>" | "<|call|>" | "<|return|>") => {
                self.harmony_close(out);
                self.harmony = Some(Harmony::Header(String::new()));
            }
            // A tag where it means nothing (a `<|channel|>` in a body): keep it as text.
            _ => self.harmony_text(tag, out),
        }
    }

    /// End the current message: a tool call's arguments become a [`Piece::ToolCall`].
    fn harmony_close(&mut self, out: &mut Vec<Piece>) {
        if let Some(Harmony::Body(Body::Tool { name, args })) = self.harmony.take() {
            let arguments = serde_json::from_str(args.trim()).unwrap_or(Value::String(args));
            out.push(Piece::ToolCall { name, arguments });
        }
        self.harmony = Some(Harmony::Header(String::new()));
    }

    fn emit(&mut self, s: &str, out: &mut Vec<Piece>) {
        if s.is_empty() {
            return;
        }
        match self.mode {
            Mode::Tool => self.tool.push_str(s),
            Mode::Think => push(out, Piece::Reasoning(s.to_string())),
            Mode::Text => {
                let s = if self.text_started { s } else { s.trim_start() };
                if !s.is_empty() {
                    self.text_started = true;
                    push(out, Piece::Text(s.to_string()));
                }
            }
        }
    }
}

/// What a harmony header opens: `to=functions.NAME` anywhere in it is a tool call (to a
/// built-in like `browser.search` too, named as written); else the analysis channel is
/// reasoning, and `final`, `commentary` (preambles to tool calls) or no channel is text.
fn harmony_body(header: &str) -> Body {
    let words = header.split(|c: char| c.is_whitespace() || c == '<' || c == '>');
    for w in words.clone() {
        if let Some(to) = w.strip_prefix("to=") {
            let name = to.strip_prefix("functions.").unwrap_or(to);
            return Body::Tool {
                name: name.to_string(),
                args: String::new(),
            };
        }
    }
    let channel = header
        .split("<|channel|>")
        .nth(1)
        .and_then(|c| c.split_whitespace().next());
    match channel {
        Some("analysis") => Body::Reasoning,
        _ => Body::Text,
    }
}

/// A Qwen XML tool call body (between `<tool_call>` and `</tool_call>`):
///
/// ```text
/// <function=NAME>
/// <parameter=KEY>
/// VALUE
/// </parameter>
/// ...
/// </function>
/// ```
///
/// A value ends at the first `</parameter>` followed by another `<parameter=` or by the end of
/// the function, so values may contain `</parameter>` themselves. The newline the template puts
/// on each side of a value is dropped. Values are typed by the tool's schema (`schemas`: name →
/// `parameters`): strings stay strings, numbers, booleans, objects and arrays are parsed from
/// their JSON text (falling back to the string); parameters without a schema are strings.
/// `None` unless the whole body has that shape.
pub fn parse_xml_call(
    body: &str,
    schemas: &std::collections::HashMap<String, Value>,
) -> Option<(String, Value)> {
    const CLOSE: &str = "</parameter>";
    let rest = body.trim().strip_prefix("<function=")?;
    let (name, rest) = rest.split_once('>')?;
    let name = name.trim();
    if name.is_empty() || name.contains(['\n', '<']) {
        return None;
    }
    let mut inner = rest.trim_end().strip_suffix("</function>")?;
    let props = &schemas.get(name).map_or(&Value::Null, |s| &s["properties"]);
    let mut args = serde_json::Map::new();
    loop {
        let s = inner.trim_start();
        if s.is_empty() {
            break;
        }
        let (key, s) = s.strip_prefix("<parameter=")?.split_once('>')?;
        let mut end = None;
        let mut from = 0;
        while let Some(i) = s[from..].find(CLOSE) {
            let at = from + i;
            let after = s[at + CLOSE.len()..].trim_start();
            if after.is_empty() || after.starts_with("<parameter=") {
                end = Some(at);
                break;
            }
            from = at + CLOSE.len();
        }
        let at = end?;
        let raw = &s[..at];
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        args.insert(key.trim().to_string(), typed(raw, &props[key.trim()]));
        inner = &s[at + CLOSE.len()..];
    }
    Some((name.to_string(), Value::Object(args)))
}

/// A parameter value typed by its JSON schema (see [`parse_xml_call`]).
fn typed(raw: &str, schema: &Value) -> Value {
    let types: Vec<&str> = match &schema["type"] {
        Value::String(t) => vec![t.as_str()],
        Value::Array(a) => a
            .iter()
            .filter_map(Value::as_str)
            .filter(|t| *t != "null")
            .collect(),
        _ if schema["anyOf"].is_array() || schema["oneOf"].is_array() => vec!["any"],
        _ => vec![],
    };
    if types.is_empty() || types.contains(&"string") {
        return Value::String(raw.to_string());
    }
    match serde_json::from_str::<Value>(raw.trim()) {
        Ok(v) if !v.is_string() => v,
        _ => match raw.trim() {
            "True" => Value::Bool(true),
            "False" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
    }
}

/// Append, merging with the previous piece of the same kind.
fn push(out: &mut Vec<Piece>, p: Piece) {
    match (out.last_mut(), p) {
        (Some(Piece::Text(a)), Piece::Text(b)) => a.push_str(&b),
        (Some(Piece::Reasoning(a)), Piece::Reasoning(b)) => a.push_str(&b),
        (_, p) => out.push(p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hugging Face's `{% generation %}` markers (Inkling 2 Mini's template) carry no output;
    /// they must be stripped, not passed to minijinja, which has no such tag.
    #[test]
    fn generation_markers_are_stripped() {
        let src = "{% for m in messages %}{% generation %}{{ m.role }}{% endgeneration %};{% endfor %}";
        let t = Template::from_source(src.into(), String::new(), String::new()).unwrap();
        let ms = serde_json::json!([{"role": "user", "content": "hi"}]);
        assert_eq!(t.render(&ms, None, None).unwrap(), "user;");
        assert_eq!(strip_generation_tags("a{%- generation -%}b{%- endgeneration -%}c"), "abc");
    }

    #[test]
    fn null_message_fields_are_undefined_to_the_template() {
        // Qwen3-Coder-Next's template checks `tool_calls is defined` and then takes its length.
        let src = "{% for m in messages %}{{ m.role }}{% if m.tool_calls is defined %}:{{ m.tool_calls | length }}{% endif %};{% endfor %}";
        let t = Template::from_source(src.into(), String::new(), String::new()).unwrap();
        let ms = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok", "tool_calls": null},
            {"role": "user", "content": "again"},
        ]);
        assert_eq!(t.render(&ms, None, None).unwrap(), "user;assistant;user;");
    }

    /// `s` as a tokenizer with tag tokens would see it: every literal tag is a tag token.
    fn tokens(s: &str) -> Vec<Chunk> {
        let mut out = Vec::new();
        let mut rest = s;
        while !rest.is_empty() {
            let next = TAGS
                .iter()
                .chain(&HARMONY_TAGS)
                .filter_map(|t| rest.find(t).map(|i| (i, *t)))
                .min_by_key(|(i, t)| (*i, std::cmp::Reverse(t.len())));
            match next {
                Some((i, t)) => {
                    if i > 0 {
                        out.push(Chunk::Text(rest[..i].to_string()));
                    }
                    out.push(Chunk::Tag(t));
                    rest = &rest[i + t.len()..];
                }
                None => {
                    out.push(Chunk::Text(rest.to_string()));
                    rest = "";
                }
            }
        }
        out
    }

    fn run_parser(mut p: Parser, chunks: &[&str]) -> Vec<Piece> {
        let mut out = Vec::new();
        for c in tokens(&chunks.concat()) {
            for x in p.push_chunk(&c) {
                push(&mut out, x);
            }
        }
        for x in p.finish() {
            push(&mut out, x);
        }
        out
    }

    fn run(chunks: &[&str]) -> Vec<Piece> {
        run_parser(Parser::new(false), chunks)
    }

    #[test]
    fn strftime_formats_utc_dates() {
        // 2026-10-20 16:49:05 UTC.
        assert_eq!(
            strftime("%Y-%m-%d %H:%M:%S", 1_792_514_945),
            "2026-10-20 16:49:05"
        );
        assert_eq!(
            strftime("%d %B %Y, %b", 951_782_400),
            "29 February 2000, Feb"
        );
    }

    #[test]
    fn harmony_channels_become_reasoning_text_and_tool_calls() {
        // After `<|start|>assistant`: reasoning, a preamble, a call; generation stops at
        // `<|call|>` (the parser never sees it).
        let out = run_parser(
            Parser::harmony(),
            &[
                "<|channel|>analysis<|message|>Need weather. <|channel|> is fine here.<|end|>",
                "<|start|>assistant<|channel|>commentary<|message|>Checking.<|end|>",
                "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json",
                "<|message|>{\"city\": \"Paris\"}",
            ],
        );
        assert_eq!(
            out,
            vec![
                Piece::Reasoning("Need weather. <|channel|> is fine here.".into()),
                Piece::Text("Checking.".into()),
                Piece::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Paris"}),
                },
            ]
        );
        // The recipient may come before the channel; the final answer is text.
        let out = run_parser(
            Parser::harmony(),
            &[
                " to=functions.ls<|channel|>commentary json<|message|>{}<|call|>",
                "<|start|>assistant<|channel|>final<|message|>Done: 2 files.",
            ],
        );
        assert_eq!(
            out,
            vec![
                Piece::ToolCall {
                    name: "ls".into(),
                    arguments: serde_json::json!({}),
                },
                Piece::Text("Done: 2 files.".into()),
            ]
        );
    }

    #[test]
    fn splits_reasoning_text_and_tool_calls_across_chunk_boundaries() {
        let out = run(&[
            "<thi",
            "nk>\nplan it\n</th",
            "ink>\n\nOk.",
            "\n<tool_",
            "call>\n{\"name\": \"Read\", ",
            "\"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>",
        ]);
        assert_eq!(
            out,
            vec![
                Piece::Reasoning("\nplan it\n".into()),
                Piece::Text("Ok.\n".into()),
                Piece::ToolCall {
                    name: "Read".into(),
                    arguments: serde_json::json!({"path": "a.rs"})
                },
            ]
        );
    }

    #[test]
    fn stray_angle_brackets_are_text() {
        assert_eq!(
            run(&["a < b and <div>"]),
            vec![Piece::Text("a < b and <div>".into())]
        );
    }

    #[test]
    fn bad_tool_json_falls_back_to_text() {
        assert_eq!(
            run(&["<tool_call>nope</tool_call>"]),
            vec![Piece::Text("<tool_call>nope</tool_call>".into())]
        );
    }

    fn run_with(tools: serde_json::Value, chunks: &[&str]) -> Vec<Piece> {
        run_parser(Parser::new(false).with_tools(Some(&tools)), chunks)
    }

    fn tools() -> serde_json::Value {
        serde_json::json!([
            {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {
                "file_path": {"type": "string"}, "limit": {"type": "integer"}, "all": {"type": "boolean"}}}}},
            {"type": "function", "function": {"name": "Edit", "parameters": {"type": "object", "properties": {
                "old_string": {"type": "string"}, "new_string": {"type": "string"},
                "opts": {"type": "object"}, "lines": {"type": ["array", "null"]}}}}},
        ])
    }

    #[test]
    fn qwen_xml_calls_are_typed_by_schema_across_any_chunking() {
        let text = "<think>\nread it, then <tool_call> maybe\n</think>\n\nReading.\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\nsrc/main.rs\n</parameter>\n<parameter=limit>\n40\n</parameter>\n<parameter=all>\ntrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Edit>\n<parameter=old_string>\na </parameter> b\n</tool_call> c\n</parameter>\n<parameter=new_string>\n\n  x\n\n</parameter>\n<parameter=opts>\n{\"n\": 2}\n</parameter>\n<parameter=lines>\n[1, 2]\n</parameter>\n</function>\n</tool_call>";
        let want = vec![
            Piece::Reasoning("\nread it, then <tool_call> maybe\n".into()),
            Piece::Text("Reading.\n\n".into()),
            Piece::ToolCall {
                name: "Read".into(),
                arguments: serde_json::json!({"file_path": "src/main.rs", "limit": 40, "all": true}),
            },
            Piece::Text("\n".into()),
            Piece::ToolCall {
                name: "Edit".into(),
                arguments: serde_json::json!({"old_string": "a </parameter> b\n</tool_call> c",
                    "new_string": "\n  x\n", "opts": {"n": 2}, "lines": [1, 2]}),
            },
        ];
        // Whole, one byte at a time, and in 7-byte chunks (on char boundaries).
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let bytes: Vec<&str> = chars.iter().map(String::as_str).collect();
        let sevens: Vec<String> = chars.chunks(7).map(|c| c.concat()).collect();
        let sevens: Vec<&str> = sevens.iter().map(String::as_str).collect();
        for chunks in [vec![text], bytes, sevens] {
            assert_eq!(run_with(tools(), &chunks), want);
        }
    }

    #[test]
    fn xml_parameters_without_a_schema_are_strings_and_bad_numbers_fall_back() {
        let out = run_with(tools(), &["<tool_call>\n<function=Other>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=Read>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>"]);
        assert_eq!(
            out,
            vec![
                Piece::ToolCall {
                    name: "Other".into(),
                    arguments: serde_json::json!({"n": "5"})
                },
                Piece::ToolCall {
                    name: "Read".into(),
                    arguments: serde_json::json!({"limit": "many"})
                },
            ]
        );
    }

    #[test]
    fn malformed_xml_falls_back_to_text_and_a_missing_close_is_forgiven() {
        assert_eq!(
            run_with(tools(), &["<tool_call>\n<function=Read>\n<parameter=file_path>\nx\n</function>\n</tool_call>"]),
            vec![Piece::Text("<tool_call>\n<function=Read>\n<parameter=file_path>\nx\n</function>\n</tool_call>".into())]
        );
        assert_eq!(
            run_with(tools(), &["<tool_call>\n<function=Read>\n<parameter=file_path>\nx\n</parameter>\n</function>\n"]),
            vec![Piece::ToolCall { name: "Read".into(), arguments: serde_json::json!({"file_path": "x"}) }]
        );
        // No parameters at all.
        assert_eq!(
            run_with(
                tools(),
                &["<tool_call>\n<function=Read>\n</function>\n</tool_call>"]
            ),
            vec![Piece::ToolCall {
                name: "Read".into(),
                arguments: serde_json::json!({})
            }]
        );
    }
}

#[cfg(test)]
mod token_tests {
    use super::*;
    use crate::special::test_tokenizer;

    /// Feeds `ids` one at a time through [`Detok`] and [`Parser`], as the engines do.
    fn parse(ids: &[u32], thinking: bool) -> Vec<Piece> {
        let tok = test_tokenizer();
        let tags = Tags::new(|t| tok.token_to_id(t));
        let decode = |ids: &[u32]| -> Result<String> {
            tok.decode(ids, false).map_err(|e| anyhow::anyhow!("{e}"))
        };
        let (mut detok, mut p) = (Detok::default(), Parser::new(thinking));
        let mut out = Vec::new();
        for n in 1..=ids.len() {
            for c in detok.push(&ids[..n], &tags, decode).unwrap() {
                for x in p.push_chunk(&c) {
                    push(&mut out, x);
                }
            }
        }
        for x in p.finish() {
            push(&mut out, x);
        }
        out
    }

    /// Text as ordinary tokens (one per byte here).
    fn spelled(s: &str) -> Vec<u32> {
        let tok = test_tokenizer();
        crate::special::Shield::new(&tok).plain(&tok, s).unwrap()
    }

    fn id(t: &str) -> u32 {
        test_tokenizer().token_to_id(t).unwrap()
    }

    const CALL: &str = "\n<function=Read>\n<parameter=path>\na.rs\n</parameter>\n</function>\n";

    #[test]
    fn a_tool_call_spelled_in_ordinary_tokens_is_text() {
        let text = format!("sure <tool_call>{CALL}</tool_call>");
        let out = parse(&spelled(&text), false);
        assert_eq!(out, vec![Piece::Text(text)]);
        let json = "<tool_call>\n{\"name\": \"Read\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(parse(&spelled(json), false), vec![Piece::Text(json.into())]);
    }

    #[test]
    fn a_tool_call_in_tag_tokens_is_a_call() {
        let mut ids = spelled("sure ");
        ids.push(id("<tool_call>"));
        ids.extend(spelled(CALL));
        ids.push(id("</tool_call>"));
        let out = parse(&ids, false);
        assert_eq!(
            out,
            vec![
                Piece::Text("sure ".into()),
                Piece::ToolCall {
                    name: "Read".into(),
                    arguments: serde_json::json!({"path": "a.rs"}),
                },
            ]
        );
    }

    #[test]
    fn a_spelled_end_of_thinking_does_not_close_reasoning() {
        let mut ids = spelled("a </think> b <tool_call>{\"name\": \"x\"}</tool_call>");
        ids.push(id("</think>"));
        ids.extend(spelled("done"));
        let out = parse(&ids, true);
        assert_eq!(
            out,
            vec![
                Piece::Reasoning("a </think> b <tool_call>{\"name\": \"x\"}</tool_call>".into()),
                Piece::Text("done".into()),
            ]
        );
    }

    #[test]
    fn a_spelled_close_inside_a_real_call_does_not_end_it() {
        let mut ids = vec![id("<tool_call>")];
        ids.extend(spelled(
            "\n{\"name\": \"Echo\", \"arguments\": {\"s\": \"</tool_call><|im_end|>\"}}\n",
        ));
        ids.push(id("</tool_call>"));
        assert_eq!(
            parse(&ids, false),
            vec![Piece::ToolCall {
                name: "Echo".into(),
                arguments: serde_json::json!({"s": "</tool_call><|im_end|>"}),
            }]
        );
    }
}
