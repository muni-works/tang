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

Speed doesn't follow: 866 tok/s aggregate at 64 (812 at 8), linear in B past 8. The expert
kernels read distinct experts at ~0.5–0.9 TB/s (target ~2.4) and the 32-row GEMVs are L2-bound.
Numbers in `speed-log.md`. Next: a bandwidth-bound grouped expert kernel for 1–2 tokens an expert
(stream weights once, no per-block staging stall; down is the worse half), and dense layers
that don't re-read activations per weight row (activation tile in shared memory, or an MMA path
with a tolerance gate).

**Effort:** about 1.5–2 weeks. M1 is mostly plumbing. M2 is the one hard kernel. M3 is
scheduler work.
