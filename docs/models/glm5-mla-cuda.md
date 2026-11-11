# GLM-5.3-Flash DSA layer on CUDA (W3) — kernels, parity, pod commands

Status: naive-but-correct (GATES rule). No absorbed MLA, no tensor-core tuning yet.

## Kernels (NVRTC `CUDA_SRC` in `flash/glm5/mla.rs`, loaded by `MlaCuda`)

| kernel | grid/block | does |
|---|---|---|
| `mla_kv_append` | t × 256 | kv_a GEMV + RMSNorm per token → **bf16** 512-d latent row into the cache |
| `idx_key_append` | t × 256 | indexer key = LayerNorm(wk·x) (weight **and bias**, eps 1e-6) + gate row, marks valid |
| `idx_pool_keys` | n_pool × 256 | per-dim softmax over the 4 slots (gate + APE), weighted key average |
| `idx_select` | t × 256 | per-query: Σ_h w_h·relu(q·pool)·scale over all pools (serial pool loop, nh·hd dots), running top-512 in smem (4 KB), emits [2051] i32 (invisible pools −1, tail after slot 2048) |
| `mla_attn_naive` | (nh, t) × 256 | per (query, head): decompress each selected latent (512→256 K via `wk_h`, 512→256 V via `wv_h`), 2-pass softmax (max, then sum) |

Shared memory: attention (dk+dv)·4 B = 2 KB at 256/256; selection 512·8 = 4 KB.
The host GEMVs (`iq`, `hw`, `q`) stay on the host for now — small ([t][1536] worst).
Cache layout per DSA layer: `[cap][512]` u16 latents + `[cap][128]` f32 k + `[cap][128]`
f32 gate + `[cap]` u8 valid.

## Parity tests (cuda-gated, skip without a device; `TANG_REQUIRE_CUDA=1` makes skip fail)

`glm5::mla::cuda::cuda_naive_matches_cpu` on the seed-7 tiny fixture (S=130, 4 heads):
kv_append vs CPU f32 latents (bf16 bar 3e-3), idx keys (1e-5), pooled keys (1e-5),
selection sets vs CPU ref (≤15% tie-row slack; tie *order* is W0's study — kernel uses
lower-index-first, same as CPU), attention dense+sparse on the bf16 cache, and
end-to-end o_proj'd output vs HF (bar 3e-2 because the latents are bf16).

## Pod commands (1×H200, release)

```sh
cargo test -p tang-llm --lib --release \
  --no-default-features --features cuda glm5::mla -- --nocapture
# explicit gate (fails instead of skipping):
TANG_REQUIRE_CUDA=1 cargo test -p tang-llm --lib --release \
  --no-default-features --features cuda glm5::mla::cuda -- --nocapture
```

## > 2048 tokens (R7)

`indexer_is_dense(kv_len, topk)` is false past 2048 keys; the CPU path already runs
`indexer_select` there (validated against HF on the 2100-token tiny fixture:
`glm5::mla::tests::indexer_long_context_matches_hf`, 2100 rows, all set mismatches
explained by exact relu-0.0 ties). The GPU path: pool keys from the ring via
`idx_pool_keys` (n_pool ≤ cap/4), then `idx_select` per query. Decode-window semantics:
the caller appends the window's rows (kv_append + idx_append), recomputes pool keys
only for newly completed pools, and runs idx_run + attend for the window's queries.

## Still open (FAST only, after R1–R7)

Absorbed projection (latent-space scores, one Wv_b GEMV per query — needs cx_attn_fa2
HD=512), batched pool refresh, moving the indexer GEMVs on-device.
