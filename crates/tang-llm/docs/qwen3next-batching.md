# Continuous batching for Qwen3-Coder-Next

**Goal:** one H200 serves 8 concurrent streams at more than 100 tok/s each (800+ aggregate),
not one stream at about 240 tok/s.

**Why:** at one sequence per GPU, 240 tok/s is about 0.86 M output tokens an hour. Against
$3.6–5.3 an hour that costs $4–6 per million tokens, which is 3–4× the $1.50 price.

## Where the time goes at batch B

From the measured window costs on H200 (single sequence, UD-Q4_K_XL):

| | T = 1 | T = 8 (one sequence's verify window) |
|---|---|---|
| window | 4.1 ms | ~13.5 ms (3.3×) |

A window of B rows costs about the same whether the rows come from one sequence or from B
sequences: the kernels are row-independent. What grows with B is the expert reads. Each row
picks 10 of 512 experts per layer, so 8 rows touch ~75 distinct experts instead of 10.

Today's expert kernels also read a pair's weights once per (token, expert) pair, not once per
expert. So the step reads:

- dense weights: ~1.5 GB, shared by all rows;
- experts: ~1.9 MB per (row, expert) pair per layer, i.e. 0.9 GB per row per step.

At B = 8 that is about 13 ms a step, or 590 tok/s aggregate, **74 a stream**. That is short of
the target, and the gap is the expert reads.

With a **grouped expert kernel** (each distinct expert's rows read once, looping over the
tokens routed to it) the expert bytes become ~75 experts × 1.9 MB × 48 layers ≈ 6.8 GB a step at
B = 8. At the ~2.5 TB/s a well-fed grouped kernel reaches, a step costs:

| part | ms |
|---|---|
| experts | 2.7 |
| dense (T = 8 GEMV, ~2 TB/s) | 0.8 |
| GDN + attention, per sequence | ~1 |
| **step** | **≈ 4.5–5.5** |

That is **~1,500 tok/s aggregate, ~190 a stream**, which clears the target with margin.
Flash-Next already plans grouped experts this way (`MoePlan`, `moe_grouped_into`), so the
pattern exists in the codebase.

## Design

1. **Sequence slots.** Each slot owns its recurrent and positional state, preallocated:
   - GDN state and conv history: 36 layers × (32 × 128 × 128 + 3 × 8192) f32 ≈ 75 MB a slot;
   - attention K/V: 12 layers × 2 KB a position, ≈ 0.8 GB a slot at 32k context.

   8 slots at 32k is ~7 GB beside the 50 GB of weights. `save_running` / `restore_running` /
   positional save and restore become per slot, so prefix reuse keeps working.
2. **Batched window.** A window is a list of rows. Each row is (slot, position, token), and
   rows of one slot are consecutive (its decode token, or its draft run, or a prefill chunk).
   A per-row control record carries slot, position, and the per-slot `n_keep`, temperature and
   seed. Every dense GEMV, norm, router and combine kernel is already row-independent and only
   needs T up to 16–32 (the wide kernels).
3. **Attention.** `cx_attn_prep` writes K/V at (slot, position) through a per-row slot base
   pointer. `cx_attn_part` / `cx_attn_merge` read their row's slot cache up to its position.
   The split-K grid is per row already.
4. **GDN.** `fl_gdn_conv_step` gets a slot dimension: grid (v head, slot), and each block runs
   its slot's rows in order, with state and history at the slot's base. The commit (`n_keep`)
   is per slot, from the control record. The conv-history commit is per slot too.
5. **Experts.** Plan per layer (on the GPU, as `moe_route_into` does): sort the (row, rank)
   pairs by expert, then one block group per distinct expert. Each group stages that expert's
   gate|up rows once and applies them to all its tokens, then the same for down. Per-row
   accumulation order stays fixed (rank order in the combine), so a row's result doesn't depend
   on its batch.
6. **Sampling.** The sampler already takes per-row control. Each row gets its own temperature,
   seed and top-k/top-p from its request.
7. **Scheduler** (`flash serve`). Each step:
   - takes every active slot's next token, with its suffix drafts when they're priced in,
     within a row budget (16, then 32);
   - fills spare rows with prefill chunks of waiting requests (chunked prefill, so a new long
     prompt doesn't stall decode);
   - replays the CUDA graph for that shape. Decode-only shapes B × 1, B = 1..8, are captured;
     mixed shapes run eagerly or padded.
8. **Correctness gate.** `flash-batch-check`: a sequence's tokens must be bitwise identical
   whether it runs alone or batched with others, at any position in the batch. This is the
   batch analogue of `flash-tcheck`, and it holds if every kernel is row-independent with a
   fixed reduction order. Plus `flash-spec-test` per slot and `flash-parity`.

## Milestones

| | what | expected at 8 streams (H200) |
|---|---|---|
| M1 | slots + batched decode windows (B × 1), today's per-pair expert kernels, scheduler without drafts | ~590 aggregate, ~74 a stream |
| M2 | grouped expert kernel | ~1,200–1,500 aggregate, 150–190 a stream |
| M3 | per-slot drafts in the batch, chunked prefill interleave, per-slot prefix reuse | holds M2 under agent traffic, cuts TTFT |

Correctness gates at every milestone: batch-check, parity and spec-test.

## Status (2026-11-03): batch-64 windows

`Coder::enable_batch(rows)` builds a third module (rows 32..256, next to narrow and wide) and
`decode_batch` runs up to that many sequences as one window: K-split GEMVs 32 rows a launch
(`cx_gemvkb`, the narrow `cx_gemvk`'s arithmetic), grouped experts on by default (big groups —
the shared expert — split into groups of 8), multi-segment GDN, split-K attention for every row,
head and argmax for every row. `MAX_SLOTS` is 128. `flash serve --batch N` with no drafts groups
plain decode by the batch width. `flash-batch-check` (`TANG_BATCH_N`, `TANG_BATCH_CTX`) passes
**bitwise** at 64 sequences.

Speed: 866 tok/s aggregate at 64 at first; 1,251 with the streaming expert kernels (gate|up ~1.5–2 TB/s, down ~1–1.2 TB/s). Dense GEMVs, GDN and attention are now about half the step.

Tensor cores (tolerance-gated, see `speed-log.md`): int8 MMA down experts and Q8X dense layers
take B=64 to 40.0 ms (1,600 tok/s), B=32 to 25.8 ms. Left: gate|up experts (~10.7 ms), router
and plan (~3.2 ms), GDN conv+step (4.3 ms).

**Proposal: bf16 GDN state for batch windows.** The recurrent state is f32, 32 heads × 128 ×
128 per layer and sequence (2 MB); a decode step reads and writes all of it, ~270 MB a layer at
B=64, which is the 4.3 ms GDN conv+step (36 layers, ≈ bandwidth). Keeping it in bf16 (f32
accumulate inside the step, round on store) halves the traffic: ~2.1 ms saved at B=64, ~1 ms at
B=32, and half the per-slot state memory (75 → 38 MB). It's a numerics change on every step of
every sequence (the error compounds through the recurrence, unlike a reordered sum), so it needs
its own gate: long-context greedy agreement and KL against the f32 state, not just batch-check. The expert
kernels read distinct experts at ~0.5–0.9 TB/s (target ~2.4) and the 32-row GEMVs are L2-bound.
Numbers in `speed-log.md`. Next: a bandwidth-bound grouped expert kernel for 1–2 tokens an expert
(stream weights once, no per-block staging stall; down is the worse half), and dense layers
that don't re-read activations per weight row (activation tile in shared memory, or an MMA path
with a tolerance gate).

**Effort:** about 1.5–2 weeks. M1 is mostly plumbing. M2 is the one hard kernel. M3 is
scheduler work.

## Serving fixes 2026-11-04

GPU checks below are still to run (written without a GPU; CPU tests green). `$G` is the
Qwen3-Coder-Next first GGUF shard; build with `cargo build --release -p tang-llm
--no-default-features --features cuda`.

1. **A failed step ends only its streams.** `engine.step` used to retire every stream with the
   error. Now `batch_step` wraps failures in `server::StepFailure { lanes, fatal }`: a prefill,
   prefix snapshot or token-take error names its one lane, a decode/verify window's error names
   the window's lanes. The worker retires those (their slots are freed, not parked: `broken`) and
   the rest keep decoding. A device fault (`CUDA_ERROR…` in the message, an unattributed error
   that reads as one, or a panic out of the compute layer, which `catch_unwind` now catches)
   ends every stream, sets `NodeState::unhealthy`, and `/health` answers 503 `unhealthy: …`;
   new requests fail with "restart the server" instead of joining. Tests: `server::step_failures`.
   GPU check: `tang-llm serve $G --batch 64 --ctx 32768`, then 64 concurrent requests where one
   has a prompt near `--ctx` and `max_tokens` past it (if the slot overflows mid-decode, only that
   request errors; the other 63 finish), and `curl -s localhost:8911/health` stays `ok`. A forced
   fault (`TANG_FORCE_STEP_FAULT` is not wired; kill -SEGV is not the same) can't be staged
   cheaply: check the log line `tang-llm: the engine is unhealthy:` on the next real one.

2. **`--batch` clamped to the slots that fit.** `Coder::fit_slots` counts slots from free VRAM,
   `slot_bytes()` at `--ctx` and `TANG_SLOT_HEADROOM_GB` (default 4), and is the cap
   `alloc_slot` enforces. `flash serve` calls it at startup and again after building the batch
   and wide windows (they take memory), and sets `batch = min(batch, cap - 1)`
   (`batch::clamp_batch`; one slot spare so admitting can always evict a parked one), logging
   `--batch N clamped to M: C sequence slots fit in free VRAM at --ctx X`. `batch_can_admit`
   bounds busy slots by that `batch`, so admission stays under the cap. Test:
   `batch::tests::batch_stays_under_the_slots_that_fit`. GPU check: `tang-llm serve $G --batch 128
   --ctx 131072` on an H200 logs the clamp (slots ≈ (free − 4 GB) / slot size) and then 128
   concurrent requests run without an allocation failure; `--batch 64 --ctx 32768` is unchanged
   (no warning, 64 at once).
