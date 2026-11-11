# GLM-5.3-Flash GPU speed plan (from the naive-kernel baseline)

Status: 2026-10-26. The correctness verdict is in (see `crates/tang-llm/NOTES.md`):
the wired GPU path matches the CPU oracle exactly on the real model. These kernels were
deliberately NAIVE (GATES.md: WORK → RIGHT → FAST) — this plan is the FAST step.

## The measured baseline (B300, tree-verdict-b300-4, all-GPU, naive kernels)

| stage | naive | where the time is |
|---|---|---|
| prefill 20 tok | 22.9 s (0.9 tok/s) | per-window: 34 KDA layers × (9 G5W GEMVs + core) + 45 FFN + 45 mHC |
| decode 1 tok | ~1.9 s (0.52 tok/s) | same launches, T=1: every kernel launch-bound |

These are single-request numbers; the naive kernels' launch count dominates decode.

## Where the time goes (per decode step, per layer)

Every projection is `k5_gemv`/`g5_expert_gu`: **one thread per output value**, a serial
row dot that dequantises each weight element from its GGUF bytes (Q8_0: 1 byte + f16
scale per element). For one token:

- KDA layer: q/k/v (8192 outs each), f_a/f_b, beta, g_a/g_b, o_proj (4096 outs) —
  ~40k output threads/layer, each reading 4096 quantised bytes serially.
- MoE layer: router (288), top-8 experts' gate/up/down (2048·2 + 4096 outs per expert
  pair), shared expert, combine.
- mHC pre: 24 mixes × 16384-length dots, 32 lanes per token, strided.

Three structural costs: (1) one dequant pass per dot with no reuse, (2) no shared
memory/warp cooperation — a 4096-long row is streamed by ONE thread, (3) launch count
(~40 launches/layer/step naive).

## The plan (ordered by expected payoff per effort)

1. **Dequant-reuse GEMV ("q8 gemv8")** — port the Flash engine's `fe_gemv8` shape
   (crates/tang-llm/src/flash/coder.rs:75: two rows a warp, shared staging, int8
   activations) to the G5W types. The Flash engine's measured result on the same idea:
   Coder-Next decode went from naive to 162 tok/s at 4K ctx on an H200 — 11.9% of
   speed-of-light, with the same int8-row-dot structure. Expected: 5–20× on every GEMV.
   The B300's memory bandwidth is the real ceiling: a decode step reads ~3.5 GB of
   weights → ~1.2 ms at 3 TB/s effective, i.e. ~800 tok/s single-stream speed-of-light.
2. **Batch the per-token launches (decode)** — at T=1, grid sizes are 4096–8192 threads
   = one SM-wave; kernel launch overhead (~5 µs × ~1800 launches/step) is comparable to
   the work. Two moves: (a) fuse the q/k/v + gate + beta launches into one kernel per
   KDA layer; (b) CUDA-graph capture of a decode step (tang-moe's `Graph` exists for
   Flash; the KDA stream is already blocking-ordered for it).
3. **KDA core** — the decode `kda_step` is already fused (conv + delta rule in one
   launch); the prefill core is chunk-64 serial. For serving (short prefills, long
   decode) the core is fine; for 4k+ prefill, the chunked scan needs a split-K form.
   Second-order.
4. **mHC pre** — 24 GEMVs of 16384 each with 32 lanes/token is ~1/512 of the GPU.
   Batch tokens: one block per 8 tokens, or fold the mixes dot into the q/k/v GEMV's
   activation staging. Second-order; measure after (1)+(2).
   **Done behind `TANG_GLM5_MHC_FUSED=1`** (`mhc_fused.rs`, NOTES.md 2026-10-26): streams
   device-resident per window, post fused into the next pre, norm fused in; 3090 mHC
   time per decode step 170.8 → 10.5 ms, ids = oracle.
5. **MoE expert batch** — `g5_expert_gu` at T=1 launches 8 experts × 2 kernels; a
   grouped kernel (all selected experts in one launch, ids in shared) removes 16
   launches/layer. The Flash engine's grouped expert kernel is the template.

## Sequencing

Each step keeps the parity bar: the shadow check (`TANG_GLM5_KDA_DEBUG=1`) +
`engine_gpu_matches_cpu` + the real-model ids (785 1196 6801 264 6386 38038 911 279).
Measure with `TANG_GLM5_PROF=1` per-stage ms on the B300 (or mew's 3090 for free) before
and after each step. Steps 1–2 are the difference between 0.5 and ~100+ tok/s; do them
before anything clever.
