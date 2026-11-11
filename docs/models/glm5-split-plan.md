# GLM-5.3-Flash on 2×H200: layer-split plan (W6)

The multi-GPU layer split lives in `crates/tang-llm/src/flash/split.rs` and the stage wiring
in `crates/tang-llm/src/flash/coder.rs` (`TANG_GPUS=2`, `TANG_SPLIT_AT=k`): layers `[0, k)`
on GPU 0, `[k, L)` on GPU 1 plus the head, output norm and rope, each device with its own
kernels (NVRTC-compiled per context), scratch, pinned staging, side stream, captured graphs
and per-slot state. The per-window boundary copies GPU 0's `x` (normed hidden), `mxq` (int8
codes+scales) and `r` (residual) to GPU 1 with `cuMemcpyPeerAsync`, ordered by an event —
GPU 0's last layer reads GPU 1's first `attn_norm` over the peer link, so stage 1 continues
exactly where stage 0's `moe_add_norm` left the state. `TANG_GPUS=1` builds none of it: the
same kernels on the same stream, byte-for-byte the pre-split engine (verified by
`tests/flash_split.rs`, which also demands bitwise equality between the two loads).

Validated on Qwen3-Coder-Next (`qwen3next`, the engine the split runs on today); the GLM-5.3
port takes it unchanged once the glm5 forward lands (same engine loop, `Engine` splits the
same way — its layer loop is one `for l in 0..n_stages()` too).

## Weights: UD-Q4_K_XL, from the shard headers (no download of tensor data)

Per-tensor types read from the GGUF headers of
`unsloth/GLM-5.3-Flash-GGUF` (UD-Q4_K_XL, 6 shards, 1412 tensors; ranges only, ~10 MB):

- experts `ffn_gate_exps`/`ffn_up_exps` mostly Q4_K (12), some Q5_K (13); `ffn_down_exps`
  Q6_K (14) — per-layer mix at 4176.0 MiB/layer for a KDA+MoE layer,
- attention/projection matrices Q8_0, norms/router/biases F32,
- `token_embd` and `output` 642.8 MiB each (154,880 × 4096, Q8_0),
- totals: **185.98 GiB** weights (186.61 GB), 46 blocks (45 + MTP layer 45).

| layer band | layers | each (MiB) | notes |
|---|---|---|---|
| dense FFN | 0–2 | 293.7 | no experts |
| KDA + MoE | 4,5,6,8,…,44 | 4346.7 | ssm_* tensors |
| DSA (MLA) + MoE | 3,7,…,43 | 4333.8 | attn_kv_a_mqa etc. |
| DSA + indexer + MoE | 11 | 5215.8 | indexer tensors |
| KDA + Q6K down | 12 | 4652.7 | down_exps Q6_K |
| MTP (blk.45) | 45 | 4367.0 | nextn.* tensors |

## Split point (2×H200, 141 GiB usable each)

GPU 0 = embedding + head + layers `[0, k)`; GPU 1 = layers `[k, 45)`. Balanced cut
(≈92.5 GiB each) is **k = 23** (layers 0–22 vs 23–44). In tang's notation the layer list
is `blk.0..blk.44` (46 entries including MTP); the Coder-style engine would count the MTP
layer as a normal layer, so `TANG_SPLIT_AT` lands at 23 of 46.

| | GPU 0 | GPU 1 |
|---|---|---|
| weights at k=23 | 92.34 GiB | 93.64 GiB |
| free (of 141) | 48.7 GiB | 47.4 GiB |

## KV / state headroom at k=23 (per stream)

Per layer, per stream:

- **34 KDA layers**: recurrent state 64 heads × 128 × 128 f32 = **4 MiB/layer fixed**
  (136 MiB all 34; GPU 0 holds 18 KDA layers = 72 MiB, GPU 1 16 = 64 MiB) + conv state
  3 × (d_conv−1) × 8192 f32 = 288 KiB/layer fixed (9.6 MiB all 34). Never grows with
  context.
- **11 DSA layers**: the 512-dim NoPE latent per token at bf16 = **1 KiB/token/layer**,
  11.0 KiB/token per stream for all 11; plus the pooled indexer's key|gate rows
  3 × 128 f32 = 1.5 KiB/token/layer (16.5 KiB/token all 11) and pooled rows at
  1/4 the token count. Total ≈ **27.5 KiB/token/stream** counting the indexer.
  (W3's absorbed-MLA plan keeps only the latent: 11.0 KiB/token.)
- MTP layer 45: one more DSA layer's worth.

With ~47.4 GiB free after weights on each side, minus scratch + pinned staging + captured
graphs (≈3 GiB on the Coder-Next geometry; GLM-5's hidden-4096 FA2 partials scale it to
≈4–6 GiB), call it **~40 GiB of KV headroom per GPU**. At k=23, GPU 0 owns the DSA layers
3, 7, 11, 15, 19 (5) and GPU 1 the DSA layers 23, 27, 31, 35, 39, 43 (6): each device
pays only its own layers' KV.

**Max context × streams** (latent-only, 11.0 KiB/token): a single stream reaches ~3.7 M
tokens; 131k context × ~2,900 streams. With the indexer rows too (27.5 KiB/token):
~1.5 M tokens single-stream, 131k × ~1,100 streams per device. The engine's MAX_SLOTS
(16 in the Coder engine) is the practical batching limit until raised; 16 streams of
131k cost ~57 GiB of KV — over the headroom, so the slots cap is safe but 131k × 16 needs
the latent-only plan or a higher k. 32k × 16 streams is 14 GiB: comfortable.

## Pod commands (2×H200, e.g. RunPod)

```bash
# build (once per pod; a persistent volume at /root/tang speeds this up)
git clone <tang> && cd tang
cargo build --release -p tang-llm --no-default-features --features cuda

# 1. the split parity gate (bitwise, ~20 min with load)
TANG_LLM_TEST_GGUF=/root/Qwen3-Coder-Next-480B-A35B-Instruct-UD-Q4_K_XL-00001-of-00003.gguf \
TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm \
    --no-default-features --features cuda --test flash_split -- --nocapture

# 2. single vs split decode throughput (dart capacity)
TANG_GPUS=1 target/release/tang-llm flash-bench <gguf> --prompt-ids <ids> --n 128
TANG_GPUS=2 TANG_SPLIT_AT=<half> target/release/tang-llm flash-bench <gguf> --prompt-ids <ids> --n 128

# 3. eager (no graphs) sanity if 2 misbehaves
TANG_GPUS=2 TANG_SPLIT_AT=<half> target/release/tang-llm flash-generate <gguf> --eager
```

## Open items

- Micro-batch pipelining (GPU 0 works on window n+1 while GPU 1 finishes n): the boundary
  events are already per-window; needs double staging buffers. Plan only, not built.
- Sampling stays greedy under `TANG_GPUS=2` (`enable_sampler` declines): the sampler's
  kernels are compiled on GPU 0's context and the head runs on GPU 1; a per-stage
  sampler is the fix. Greedy generation (bench, MTP verify) is unaffected.
- `Engine` (Flash-Next) gets the same treatment when GLM-5 lands on it; its `serve` path
  (doorbell/CPU misses) is single-GPU for now — Qwen3-Coder-Next has no misses on H200,
  so it doesn't gate the parity result.
- The GLM-5 DSA indexer's per-token rows are counted at 384 f32; verify against W3's
  fixtures before trusting the >2048-token numbers to 3 significant figures.
