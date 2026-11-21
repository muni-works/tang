# Inkling 2 Mini: port status (2026-11-21)

Thinking Machines Lab's looped transformer, GGUF architecture `inkling2_loop`. Reference:
llama.cpp `src/models/inkling2.cpp` + `inkling2-loop.cpp`. Port: `crates/tang-llm/src/flash/inkling/`.

**Status: the graph is complete and runs on the real 17.6 GB checkpoint, and every input to the
attention (embedding, norms, Q/K/V/R, clips, sinks, relative bias, softmax) is verified against
llama.cpp. The logits are still wrong** (top-1 agreement 0/8, correlation ~0.3), so one step
between attention and the logits remains incorrect. Not usable for inference yet.

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

## Where the bug is

`attn_out-1` (layer 1, post-conv) has the right rms (0.473 vs the reference's 0.468) but
uncorrelated per-element values, which is the signature of a **permutation** rather than a scale
error. The candidates, in order:

1. The `wo` input layout: `attn_output.weight` is `[3072, 2304]`, so `Mat` rows = 2304, cols =
   3072, and the gated attention vector must be `(head, dim)` with `dim` fastest. The gate
   (`attn_gate`, `[2304, 3072]`) produces `(head, dim)` the same way. Transposing the gated
   vector to `(dim, head)` moved rms from 0.066 to 0.055 against a target of 0.096, so neither is
   right.
2. The conv tap order in `conv()`: token 0 uses only tap 3 of the 4-tap kernel with zero history,
   per `ggml_ssm_conv`; tap 0 gave a worse match, so the tap order is probably fine.
3. The MoE: `ffn_moe_weights_scaled` shows llama.cpp normalizes the top-8 sigmoid weights and
   scales by 8.0, which is what `moe()` does; but the router logits disagree at the source, so
   this cannot be checked until attention is fixed.

## How to finish it

Use llama.cpp's own dumps as oracles, in order, fixing one before moving to the next:

- `attn_out-<il>` (post-conv attention output; note the block-start layers 0, 6, 12, ... are
  missing from the dump) — compare against `Run::layer`'s `attn_out`.
- `ffn_inp-<il>`, `ffn_out-<il>`, `l_out-<il>` — the residual stream at each step.
- `ffn_moe_logits-<il>`, `ffn_moe_probs-<il>`, `ffn_moe_weights_scaled-<il>`, `ffn_moe_out-<il>`
  — the router and the experts.
- `result_norm`, `result_output` — the head.

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
