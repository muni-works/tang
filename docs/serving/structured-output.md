# Structured output on the flash server

`tang-llm serve` (the flash engine; Qwen3-Coder-Next through the qwen3next `Coder`) enforces
OpenAI's `response_format` at the token level:

- `{"type":"json_schema","json_schema":{"schema":{...}}}`: the reply is a JSON value valid for the schema.
- `{"type":"json_object"}`: the reply is a JSON object (schema `{"type":"object"}`).
- `{"type":"text"}` or no `response_format`: unconstrained, as before.

The JSON goes out verbatim as `message.content` (and as content deltas when streaming). Tool-call
and think markers inside string values are left as data, not parsed as structure.

## Design

**Request parsing.** `crate::server::parse` calls `structured::response_schema`, which checks the
format, bounds the schema at 64 KiB, rejects `stop` strings combined with a schema, and compiles the
schema once against a byte vocabulary, so a bad schema is a 400 before the request queues. The dense
engine (`engine.rs`) already used this; the flash server ignored it until now.

**Grammar.** [llguidance](https://github.com/guidance-ai/llguidance) (MIT, already a dependency)
compiles the schema to a grammar and keeps an Earley parser plus a token trie over the model's
vocabulary (`structured::model_factory`, built from the GGUF's tokenizer on the first structured
request and kept). Whitespace between tokens is bounded to 1..=8 characters so a model can't spend
its budget on indentation.

**Picking tokens.** The flash engine picks tokens on the GPU (greedy, or the GPU sampler), so the
mask can't be applied before the pick. `structured::Constraint::choose` does this instead:

1. If the grammar allows the device's token (`Matcher::validate_tokens`; an end of turn only once
   the value is complete), keep it. Masking only removes candidates, so an allowed greedy pick is
   exactly the masked argmax. With sampling, keeping an allowed draw and redrawing from the masked
   distribution otherwise is rejection sampling: the result is the masked distribution exactly.
2. Otherwise compute the full mask, apply it to that row's logits on the host, and pick there
   (argmax, or temperature/top-k/top-p with a seeded RNG).

Only rejected tokens pay for a full mask. On ordinary prompts the model mostly writes valid JSON
anyway, so most steps cost one `validate_tokens` call.

**Logits.** The window's logits are read before the MTP pass reuses the head buffer:
`Coder::step_logits` (single-stream path) and `Coder::decode_batch_keep` (batched path). The
Flash-Next `Engine` doesn't implement them, so a structured request there fails with an error
instead of returning unconstrained text.

**First token.** A constrained request prefills all but the prompt's last token and runs that one
as a decode step, so the first output token is checked like every other one. Prefix reuse still
works: parked slots and saved states up to the second-to-last token are used as usual.

**Speculative decoding.** Constrained streams never draft. An accepted draft is committed to the
recurrent state inside the verify window before the host could reject it, so drafts can't be
undone. In a batch, constrained streams decode one row each in windows of their own, and
unconstrained streams keep MTP drafts in theirs, so the two kinds coexist. Overriding a token leaves
that slot's MTP cache built from the device's token; that only lowers draft acceptance if a later
unconstrained request reuses the parked slot (verification stays exact).

**Stopping.** When the grammar is complete and nothing can follow it (`Matcher::is_stopped`, e.g.
after the closing `}`), the stream finishes with `stop` without another window. A bare number at
top level can still go on, so the model's end of turn ends it (allowed only once the value is
complete). `max_tokens` can still cut a reply short (`finish_reason: "length"`), in which case the
JSON is incomplete.

**Thinking.** A request with a schema renders the template with thinking off, so the reply is the
JSON alone.

## Supported schema subset

Whatever llguidance's JSON-schema compiler accepts, which covers: `type` (object, array, string,
number, integer, boolean, null, and type lists), `properties`, `required`,
`additionalProperties` (false or a schema), `items`/`prefixItems`, `minItems`/`maxItems`, `enum`,
`const`, `anyOf`/`oneOf`/`allOf`, `$ref`/`$defs`, string `minLength`/`maxLength`/`pattern`/common
`format`s, numeric `minimum`/`maximum`, nesting, and all JSON string escapes. Schemas it can't
compile are rejected with a 400 at parse time. Unit tests cover object/array/string/number/
integer/boolean/null, enum, const, required, additionalProperties:false, nested objects and arrays.

## Performance cost

Per constrained token:

- One `validate_tokens` on the grammar (microseconds).
- A logits download of the window's head buffer (`MAX_T` rows × vocab f32, ~4.9 MB for Qwen's
  151,936 tokens): roughly 0.2–0.5 ms over PCIe. This is the main fixed cost.
- Only when the device's token is rejected: a full mask (on the order of 0.1–1 ms for a 150k vocab
  with llguidance) plus a host argmax/sample over one row.
- No drafts: a constrained stream decodes one token a window, so it loses MTP's speed-up (on the
  order of 1.5–2× for this model). Unconstrained streams in the same batch keep theirs.

Grammar compilation happens once per request (milliseconds for typical schemas); the vocabulary
trie is built once per process.

These numbers are estimates; they haven't been measured on the serving GPU yet.

## Verifying it live

Hugging Face Inference Providers re-tests structured output every 6 hours. zest checks the same
thing against the live endpoint:

```sh
curl -s https://api.frog.toys/v1/chat/completions \
  -H "Authorization: Bearer $FROG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coder-next",
    "messages": [{"role": "user", "content": "Describe a frog called Ribbit, age 3."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "frog",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "colors": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["name", "age", "colors"],
          "additionalProperties": false
        }
      }
    }
  }' | jq -e '.choices[0].message.content | fromjson | (.name|type=="string") and (.age|type=="number") and (.colors|type=="array")'
```

`jq -e` exits non-zero unless the content parses as JSON with the required fields. Run it again
with `"stream": true` (join the `delta.content` pieces before parsing) and with
`{"type": "json_object"}`. The server log shows each request's line; a constrained request
reports `drafts 0/0 accepted`.
