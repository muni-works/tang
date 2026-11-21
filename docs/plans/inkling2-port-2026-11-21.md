# Inkling 2 Mini: port status (2026-11-21)

Thinking Machines Lab's looped transformer, GGUF architecture `inkling2_loop`. Reference:
llama.cpp `src/models/inkling2.cpp` + `inkling2-loop.cpp`. Port: `crates/tang-llm/src/flash/inkling/`.

**Status: it works.** The graph runs on the real 17.6 GB checkpoint and generates coherent
reasoning — on "What is 2+2? Answer with just the number." it writes *"The user asks \"What is
2+2? Answer with just the number.\" Need answer just number. 2+2=4. Need final only \"4\"."*, and
on a haiku prompt it reasons about 5-7-5 syllable counts.

Against llama.cpp on the same tokens: layer 0 is bit-identical (Q/K/V `maxΔ` 0.0000), and the
logits agree to 0.96-0.99 correlation with 7/8 top-1 on an 8-token prompt. The residual is
llama.cpp's f16-accumulated CPU kernels: my Rust matches an independent f32 Python transcription
of the same math to 0.066, while both sit ~0.4 from llama.cpp's own expert down projection. That
is the floor for an f32 reference, not a bug.

## What is here

- `mod.rs`: `Hparams` from GGUF metadata (`inkling2_loop.*`), `check()` for the assumptions the
  forward makes, `layer_tensors`/`check_tensors` (every tensor at the shape the forward reads),
  `inventory`, and the cache-slot mapping (`layer_phys`, `n_slot`).
- `forward.rs`: the CPU f32 reference forward — the 24-layer trunk with the middle 12 layers
  (`loop.start..loop.end`) run `loop.count` times, the four low-rank control gates that mix each
  pass's banked residual streams into Q/K/V/FFN, the causal short convolutions (K, V, attention
  out, FFN out) with per-slot history, the block attention residuals (`attn_res.block_size`), the
  relative-position bias and log-scaled positions, the QK/relative clips, and the SITU MoE
  (128 experts, top-8, sigmoid, `expert_weights_scale`).
- `tests.rs`: a synthetic tiny `inkling2_loop` GGUF (8 layers, core `[2,6)` × 3) so the whole graph
  runs without the 17.6 GB checkpoint; checks determinism, that the loop changes the answer, a
  prompt past the sliding window, and that bad geometry is rejected.
- Tokenizer: `flash/tokenize.rs` `INKLING2_SPLIT` (llama.cpp maps `pre = "inkling2"` to GPT-4o's
  pattern; the checkpoint's own `tokenizer.json` agrees). **Verified token-for-token against
  `llama-tokenize`** on the real GGUF.
- Chat template: the checkpoint's template uses `{% generation %}` markers, which minijinja has no
  tag for; `chat.rs::strip_generation_tags` drops them (they render as nothing). The rendered
  prompt and ids are correct.
- CLI: `tang-llm inkling-ref <gguf> <ids...> [--all] [--top K] [--logits FILE]`.

## Verified against llama.cpp

Measured on `ggml-org/Inkling-2-Mini-GGUF` `IQ4_NL` (17.6 GB), 8 tokens, with a small harness
(`parity.cpp`) that feeds llama.cpp explicit token ids and dumps named intermediates through the
eval callback.

| quantity | agreement |
|---|---|
| `token_embd` rows | bit-identical |
| `Qcur` / `Kcur` / `Vcur` (layer 0) | ~0.03–0.04 (IQ4_NL noise) |
| `Rcur` (layer 0) | ~0.036 |
| `sinks` = `attn_sink_proj · x` | ~1e-4 |
| relative bias, solved from `kq_b` | ~0.007 (quantization noise) |
| `kq_soft_max` (layer 0, token 7, head 0) | ~1% per cell |
| final logits | **wrong**: 0/8 top-1, correlation ~0.3 |

So the bias table gather (`d * n_rel + rel`) is correct, as are the clips, the sink and the
softmax. The error is downstream of the attention probabilities.

## The two bugs that were left (both fixed)

1. **GQA head mapping.** ggml's `mul_mat` broadcast repeats each KV head over a *contiguous* run
   of query heads, so query head `h` reads KV head `h / (n_head / n_kv)`. The port had `h % n_kv`
   (tiled, the Qwen3 convention). Layer 0's attention output was 5.03 off the reference; grouped
   it is 0.016.
2. **The value norm is per head.** `ggml_rms_norm` runs over `[dh][n_kv][n_tokens]`, so each
   head's 128 values normalize to rms 1 independently. `rms_only` was chunking by the whole token
   width, which left per-head rms varying 0.56–1.40.

Both were found by dumping llama.cpp's intermediates (`kq_soft_max`, `Qcur`, `Vcur`, `ffn_inp`)
and diffing, not by reading code.

## Generating

`tang-llm inkling-ref <gguf> --chat "prompt" --gen N` renders the checkpoint's template and
generates greedily. `Session` keeps the KV and convolution state between calls, so each token is
one layer pass rather than a whole prefix; a test checks incremental decoding lands on the same
logits as a full run.

On CPU this is about 1.5 s/token for a short prompt.

## If the parity gap ever matters

Re-run the oracle diff with the harness below. The remaining suspects, if a >0.4 disagreement
ever shows up where there should not be one:

- `attn_out-<il>` (post-conv attention output; block-start layers 0, 6, 12, ... are missing from
  the dump).
- `ffn_moe_logits-<il>`, `ffn_moe_probs-<il>`, `ffn_moe_weights_scaled-<il>`, `ffn_moe_out-<il>`.
- `result_norm`, `result_output`.

Build the harness with:

```
clang++ -O2 -std=c++17 -Iinclude -Iggml/include -Isrc -o parity parity.cpp \
  -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/bin
PARITY_DUMP_DIR=/tmp/refdump ./parity model.gguf out.bin <ids...>
```

and run tang with `TANG_DUMP_DIR=/tmp/tangdump` (the dumps were scaffolding and have since been
removed from `forward.rs`; re-add them where the comparison needs them).

## Geometry (from the real GGUF)

- 24 physical layers, core `[6, 18)` run 3 times → 48 layer applications, 48 KV/conv slots.
- Hidden 2304, 24 heads / 8 KV heads, head dim 128, vocab 201024 (200058 unpadded).
- 128 experts, top-8, expert FFN 1152, `expert_weights_scale` 8.0, SITU β 4.0 / λ 25.0.
- Sliding window 1024; layers 5, 11, 17, 23 are global (`sliding_window_pattern`).
- No RoPE: `rope.dimension_count` is 0.

## The compiled graph (in progress)

`crates/tang-bench/src/ir_inkling_prefill.rs` writes the model once as a full-sequence graph over
a symbolic `T`, for `derive_step` to take decode from — the same shape as `ir_glm_prefill.rs`.
It builds, binds and evaluates, and covers the whole structure: the looped core with its four
control gates, the block attention residuals, the relative-bias gather (an affine view of the
table by distance `i − j`, so no gather is needed), the sink, the clips, the log-scaled positions,
the causal short convolutions, and the top-k MoE with SITU.

**It does not yet match the hand path.** On the tiny model the logits agree to 0.011 on one token
and drift with position (0.026 at 2, 0.62 at 3), and `blk.0.attn_out` comes out **zero** while the
layer output and the logits are non-zero — so the attention output is being lost somewhere the
rest of the pipeline compensates for. That is the bug to chase first: a zero there means the
`attn` comp's reduce dims or its views are wrong, and everything downstream is riding on the FFN
alone.

`cargo test -p tang-bench --lib ir_inkling` runs the comparison (it asserts < 0.5, not 1e-3,
until this is fixed). `tang_llm::flash::inkling::tiny` is public now so the bench crate can feed
the same weights to both paths.

**Next step.** The zero is the whole bug: `attn_out` feeds the block-start residual and the FFN
mix, so if it is zero the graph is effectively running the FFN alone and still landing near the
hand path only because the residuals dominate. Bisect inside `layer()` by naming and evaluating
each stage in turn — `p3` (the softmax), then `attn` (P·V), then the gate, then the post-conv —
against the hand path's `attn_out`. The first one that is zero localizes it. The likely suspects,
in order: the `attn` comp's reduce dims not trailing its point dims (`[kvh, T]` after `[T, h, dh]`
is right, but the views must address the space in that order), and the GQA `pick` comparing an
iota value against `floor(h / group)` where the iota's own axis is the KV head. With
`heads == kv_heads` the mapping is the identity, so setting `tiny::N_HEAD = N_HEAD_KV` takes GQA
out of the picture while bisecting.

## Running it here

This session is cloud-only: no route to `mew` (SSH host key verification fails), so nothing has
been run on the 3090. The hand path runs on this machine's CPU at ~1.5 s/token, which is enough
for correctness work but not for throughput. Getting the fast path onto a GPU is the next job
after the graph matches.
