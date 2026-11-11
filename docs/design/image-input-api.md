# Image input through Frog Cloud: API sketch

Status: design note only. Nothing in the gateway changes with this PR.
The engine side (`tang_bench::ir_ds_vision`) is in place for
DeepSeek-V4.1-Flash: preprocessing, the encoder graph, the splice and the
embedding cache.

## What clients send

The OpenAI chat format, content parts on a user message:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What storm is this?"},
  {"type": "image_url", "image_url": {"url": "data:image/webp;base64,UklGR...", "detail": "auto"}}
]}
```

- `url` is either a `data:` URL (base64) or `https://`.
- Anthropic-style `{"type": "image", "source": {"type": "base64", ...}}` maps to
  the same thing. The official `load_image_bytes` accepts both shapes.
- `detail` is accepted and ignored. DeepSeek has a single pixel budget
  (`vision_max_n_token` 1024); `low` could later mean a smaller budget.

## Gateway (zest)

1. **Validate and normalize.** Parse the parts. Check the media type against
   png / jpeg / webp / gif (first frame). Cap the decoded size, e.g. 20 MB and
   8 images per request. Reject with 400 `invalid_image` and do not forward
   anything.
2. **Fetch `https://` URLs at the gateway, not the engine.** Use a timeout,
   size cap, no redirects to private ranges, and an SSRF denylist. The
   engine pods never make outbound requests for user content.
3. **Hash.** SHA-256 of the raw image bytes is the image id. It is the same
   key the engine's `EmbedCache` uses. Bytes are content-addressed, so a
   multi-turn chat that resends the same base64 every turn hashes to the
   same id.
4. **Forward to the engine** as a sidecar, not inline in the prompt text:

   ```json
   {"messages": [...text parts with an image placeholder per image...],
    "images": [{"id": "sha256:9f2c...", "bytes_b64": "..."}]}
   ```

   With an `images_have` handshake (below), repeat images travel as id only.
5. **Billing.** An image costs its span length in prompt tokens:
   `n_llm_h · (n_llm_w + 1) + 2`, from `plan_image_grid` on the image's
   size, at most 1024. The gateway can compute this from the image header
   (width and height) before forwarding. It then shows up in `usage` like
   text and is charged in dollars per token. A 640×480 image is 206 tokens;
   the 544² floor is 184.

## Engine

1. **Template.** The chat template places DeepSeek's image placeholder
   (`image_token_id` 129264) where each image part was. `expand_prompt`
   turns each placeholder into its span:
   `[START] ([IMAGE]·w [NEWLINE])·h [END]`, every position carrying 129264.
2. **Encode.** `EmbedCache::get_or_encode(id, ...)`. On a miss: `preprocess`
   (PIL-exact letterbox), then the encoder graph for that patch grid. The
   graph is cached per `(n_vit_h, n_vit_w)` grid. The cache stores
   `[rows, 5120]` f32: 169 rows at the floor is 3.5 MB, so 1 GB holds about
   300 images. LRU by bytes.
3. **Prefill.** Build the `DsDims::vision` graph with inputs `image_rows` and
   `image_mask`. The span rows replace the embeddings, span tokens route with
   `bias_vl`, and Engram is shut on the span. Host-side n-gram hashes use
   `EngramMeta::hashes_masked`. The image must sit inside the first prefill
   chunk (`model.py` asserts `start_pos == 0`). With chunked prefill,
   relaxing this only needs the rows and mask sliced per chunk, since nothing
   in the graph looks across chunk edges except attention.
4. **Decode** is unchanged. It feeds zero rows and zero mask, and the
   derived step is the same graph.
5. **Prefix cache.** A cached KV prefix that contains an image must key on
   the image id as well as the token ids. Every span position carries the
   same id 129264, so token ids alone would wrongly match two different
   images.

## `images_have` handshake (optional, later)

Each engine node reports the image ids in its `EmbedCache` (bloom filter or
recent list). The gateway skips `bytes_b64` for ids the target node holds.
If the node evicted one in between, it replies 409 `need_image` and the
gateway resends with bytes. This saves upload bandwidth on long
image-heavy chats. It doesn't change correctness.

## Errors

| case | status |
|---|---|
| bad media type / undecodable | 400 `invalid_image` |
| too large / too many | 413 `image_too_large` |
| URL fetch fails | 400 `image_fetch_failed` |
| model without vision | 400 `model_has_no_vision` |
